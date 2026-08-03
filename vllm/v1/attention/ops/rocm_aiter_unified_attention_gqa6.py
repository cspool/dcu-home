# SPDX-License-Identifier: Apache-2.0
from vllm.triton_utils import tl, triton

_ARGS = "out q k v block_table seqused_k cu_seqlens_q softmax_scale".split()


@triton.jit
def _gqa6(
    output,
    query,
    key_cache,
    value_cache,
    table,
    seq_lens,
    query_starts,
    scale,
    CACHE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    query_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_group = tl.program_id(2)
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 256)
    query_len = tl.load(query_starts + 1)
    if query_block * BLOCK_Q >= query_len:
        return
    query_pos = query_block * BLOCK_Q + rows // 2
    query_head = kv_head * 6 + head_group * 2 + rows % 2
    query_offset = query_pos[:, None] * 6144
    query_offset += query_head[:, None] * 256 + dims[None, :]
    query_mask = query_pos < query_len
    q = tl.load(query + query_offset, mask=query_mask[:, None], other=0.0)
    maximum = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    denominator = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, 256], dtype=tl.float32)
    context_len = tl.load(seq_lens) - query_len
    query_stop = tl.minimum((query_block + 1) * BLOCK_Q, query_len)
    num_blocks = (context_len + query_stop + BLOCK_SIZE - 1) // BLOCK_SIZE
    width: tl.constexpr = 32 if BLOCK_SIZE == 64 else BLOCK_SIZE
    for block in range(0, num_blocks):
        for subtile in tl.static_range(0, BLOCK_SIZE // width):
            columns = tl.arange(0, width)
            start = block * BLOCK_SIZE + subtile * width
            if CACHE_SIZE == 784:
                logical_page = start // CACHE_SIZE
                first_offset = start % CACHE_SIZE
                first_count = CACHE_SIZE - first_offset
                first_page = tl.load(table + logical_page)
                second_page = tl.load(
                    table + logical_page + 1,
                    mask=first_count < width,
                    other=first_page,
                )
                use_second = columns >= first_count
                page = tl.where(use_second, second_page, first_page)
                offset = tl.where(
                    use_second, columns - first_count, first_offset + columns
                )
            else:
                page = tl.load(table + start // CACHE_SIZE)
                offset = (start + columns) % CACHE_SIZE
            base = (page * CACHE_SIZE + offset) * 1024 + kv_head * 256
            k = tl.load(key_cache + base[None, :] + dims[:, None])
            v = tl.load(value_cache + base[:, None] + dims[None, :])
            causal = start + columns[None, :] < context_len + query_pos[:, None] + 1
            scores = scale * tl.dot(q, k)
            scores = tl.where(query_mask[:, None] & causal, scores, float("-inf"))
            block_max = tl.maximum(maximum, tl.max(scores, axis=1))
            block_max = tl.where(block_max > float("-inf"), block_max, 0.0)
            probabilities = tl.exp(scores - block_max[:, None])
            correction = tl.exp(maximum - block_max)
            accumulator *= correction[:, None]
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            maximum = block_max
            accumulator += tl.dot(probabilities.to(v.dtype), v)
    result = accumulator / denominator[:, None]
    tl.store(output + query_offset, result, mask=query_mask[:, None])


def prefill(**kwargs):
    query = kwargs["q"]
    cache_size = kwargs["k"].shape[1]
    block_m = 64 if cache_size == 784 and query.shape[0] >= 128 else 16
    config = {
        "num_warps": 4 if block_m == 64 or query.shape[0] < 128 else 2,
        "num_stages": 2 if query.shape[0] < 128 else 1,
        "waves_per_eu": 1,
    }
    if query.shape[0] >= 128:
        config |= {"matrix_instr_nonkdim": 16, "kpack": 2}
    operands = tuple(kwargs[name] for name in _ARGS)
    _gqa6[(query.shape[0] // (block_m // 2) + 1, 4, 3)](
        *operands,
        CACHE_SIZE=cache_size,
        BLOCK_SIZE=block_m * (2 if cache_size == 64 else 1),
        BLOCK_Q=block_m // 2,
        BLOCK_M=block_m,
        **config,
    )
