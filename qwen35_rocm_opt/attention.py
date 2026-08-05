# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl

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
    BLOCK_M: tl.constexpr,
    STRIDES: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.program_id(2)
    kv_head = head // 3
    head_group = head % 3
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 256)
    block_q: tl.constexpr = BLOCK_M // 2
    block_size: tl.constexpr = BLOCK_M * (2 if CACHE_SIZE == 64 else 1)
    query_start = tl.load(query_starts + sequence)
    query_len = tl.load(query_starts + sequence + 1) - query_start
    if query_block * block_q >= query_len:
        return
    local_query_pos = query_block * block_q + rows // 2
    query_pos = query_start + local_query_pos
    query_head = kv_head * 6 + head_group * 2 + rows % 2
    query_offset = query_pos[:, None] * STRIDES[1]
    query_offset += query_head[:, None] * 256 + dims[None, :]
    query_mask = local_query_pos < query_len
    q = tl.load(query + query_offset, mask=query_mask[:, None], other=0.0)
    maximum = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    denominator = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, 256], dtype=tl.float32)
    context_len = tl.load(seq_lens + sequence) - query_len
    query_stop = tl.minimum((query_block + 1) * block_q, query_len)
    num_blocks = (context_len + query_stop + block_size - 1) // block_size
    width: tl.constexpr = 32 if block_size == 64 else block_size
    for block in range(0, num_blocks):
        for subtile in tl.static_range(0, block_size // width):
            columns = tl.arange(0, width)
            start = block * block_size + subtile * width
            logical_page = start // CACHE_SIZE
            table_offset = sequence * STRIDES[0] + logical_page
            first_page = tl.load(table + table_offset)
            first_offset = start % CACHE_SIZE
            boundary = CACHE_SIZE - first_offset
            second_page = tl.load(
                table + table_offset + 1,
                mask=boundary < width,
                other=first_page,
            )
            on_first_page = columns < boundary
            page = tl.where(on_first_page, first_page, second_page)
            offset = tl.where(
                on_first_page, first_offset + columns, columns - boundary
            )
            k_base = page * STRIDES[3] + offset * STRIDES[4]
            k_base += kv_head * STRIDES[5]
            v_base = page * STRIDES[6] + offset * STRIDES[7]
            v_base += kv_head * STRIDES[8]
            k = tl.load(key_cache + k_base[None, :] + dims[:, None])
            v = tl.load(value_cache + v_base[:, None] + dims[None, :])
            causal = (
                start + columns[None, :]
                < context_len + local_query_pos[:, None] + 1
            )
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
    output_offset = query_pos[:, None] * STRIDES[2]
    output_offset += query_head[:, None] * 256 + dims[None, :]
    tl.store(output + output_offset, result, mask=query_mask[:, None])


def prefill(**kwargs) -> None:
    """Launch the Qwen3.5 GQA6 prefill kernel using host-owned metadata."""
    query = kwargs["q"]
    cache_size = kwargs["k"].shape[1]
    num_sequences = kwargs["seqused_k"].shape[0]
    wide = (cache_size, num_sequences, query.shape[0] >= 128) == (784, 1, True)
    block_m = 64 if wide else 16
    config = {
        "num_warps": 4 if block_m == 64 or query.shape[0] < 128 else 2,
        "num_stages": 2 if query.shape[0] < 128 else 1,
        "waves_per_eu": 1,
    }
    if query.shape[0] >= 128:
        config |= {"matrix_instr_nonkdim": 16, "kpack": 2}
    strides = (
        kwargs["block_table"].stride(0),
        query.stride(0),
        kwargs["out"].stride(0),
        *kwargs["k"].stride()[:3],
        *kwargs["v"].stride()[:3],
    )
    grid = (triton.cdiv(kwargs["max_seqlen_q"], block_m // 2), 12, num_sequences)
    _gqa6[grid](
        *(kwargs[name] for name in _ARGS),
        CACHE_SIZE=cache_size,
        BLOCK_M=block_m,
        STRIDES=strides,
        **config,
    )
