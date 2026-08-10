# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.triton_utils import tl, triton

# Call chain: Qwen3NextAttention -> ROCm AITER backend -> prefill -> _gqa6_prefill.
# Role: handle Qwen3.5 Q24/KV4/D256 prefill when page784 declines the request.
# Work logic: group six Q heads into three pairs per KV head, reuse each paged-KV
# tile for a pair, and update FP32 online-softmax state before writing BF16 output.
# Correctness boundary: the backend owns shape/dtype/batch gates and AITER fallback.


@triton.jit
def _gqa6_prefill(
    output,
    query,
    key_cache,
    value_cache,
    block_table,
    query_len,
    context_len,
    scale,
    CACHE_STRIDES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    query_block, query_group = tl.program_id(0), tl.program_id(1)
    kv_head, head_pair = query_group // 3, query_group % 3
    rows, dimensions = tl.arange(0, BLOCK_ROWS), tl.arange(0, 256)
    tokens_per_block: tl.constexpr = BLOCK_ROWS // 2

    local_token = query_block * tokens_per_block + rows // 2
    query_head = kv_head * 6 + head_pair * 2 + rows % 2
    query_offset = local_token[:, None] * 6144
    query_offset += query_head[:, None] * 256 + dimensions[None, :]
    valid_query = local_token < query_len
    query_values = tl.load(query + query_offset, mask=valid_query[:, None], other=0.0)

    max_score = tl.full([BLOCK_ROWS], float("-inf"), dtype=tl.float32)
    normalizer = tl.full([BLOCK_ROWS], 1.0, dtype=tl.float32)
    weighted_values = tl.zeros([BLOCK_ROWS, 256], dtype=tl.float32)
    query_stop = tl.minimum((query_block + 1) * tokens_per_block, query_len)
    cache_blocks = (context_len + query_stop + BLOCK_ROWS - 1) // BLOCK_ROWS
    # Performance: wide CTAs reuse each KV tile for two paired Q heads.
    tile_width: tl.constexpr = 32 if BLOCK_ROWS == 64 else BLOCK_ROWS

    for cache_block in range(0, cache_blocks):
        for subtile in tl.static_range(0, BLOCK_ROWS // tile_width):
            columns = tl.arange(0, tile_width)
            start = cache_block * BLOCK_ROWS + subtile * tile_width
            logical_page = start // 784
            first_page = tl.load(block_table + logical_page)
            first_position = start % 784
            first_page_tokens = 784 - first_position
            second_page = tl.load(
                block_table + logical_page + 1,
                mask=first_page_tokens < tile_width,
                other=first_page,
            )
            use_first_page = columns < first_page_tokens
            page = tl.where(use_first_page, first_page, second_page)
            position = tl.where(
                use_first_page,
                first_position + columns,
                columns - first_page_tokens,
            )
            # Correctness: hybrid KV caches are strided views, not packed pages.
            key_offset = page * CACHE_STRIDES[0] + position * CACHE_STRIDES[1]
            key_offset += kv_head * CACHE_STRIDES[2]
            value_offset = page * CACHE_STRIDES[4] + position * CACHE_STRIDES[5]
            value_offset += kv_head * CACHE_STRIDES[6]
            keys = tl.load(
                key_cache
                + key_offset[None, :]
                + dimensions[:, None] * CACHE_STRIDES[3]
            )
            values = tl.load(
                value_cache
                + value_offset[:, None]
                + dimensions[None, :] * CACHE_STRIDES[7]
            )
            causal = start + columns[None, :] < context_len + local_token[:, None] + 1
            scores = scale * tl.dot(query_values, keys)
            scores = tl.where(valid_query[:, None] & causal, scores, float("-inf"))
            # Correctness: rescale prior online-softmax state for each new maximum.
            next_max = tl.maximum(max_score, tl.max(scores, axis=1))
            next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
            probabilities = tl.exp(scores - next_max[:, None])
            correction = tl.exp(max_score - next_max)
            weighted_values *= correction[:, None]
            normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
            max_score = next_max
            weighted_values += tl.dot(probabilities.to(values.dtype), values)

    result = weighted_values / normalizer[:, None]
    output_offset = local_token[:, None] * 6144
    output_offset += query_head[:, None] * 256 + dimensions[None, :]
    tl.store(output + output_offset, result, mask=valid_query[:, None])


def prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata,
    scale: float,
) -> None:
    # This host launcher runs only after the backend proves the exact target shape.
    query = query[: metadata.num_actual_tokens]
    output = output[: metadata.num_actual_tokens]
    # Performance: short tails avoid a 64-row CTA; long queries amortize it.
    block_rows = 64 if query.shape[0] >= 128 else 16
    options = {
        "num_warps": 4,
        "num_stages": 2 if query.shape[0] < 128 else 1,
        "waves_per_eu": 1,
    }
    if query.shape[0] >= 128:
        options |= {"matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (triton.cdiv(metadata.max_query_len, block_rows // 2), 12, 1)
    _gqa6_prefill[grid](
        output,
        query,
        key_cache,
        value_cache,
        metadata.block_table,
        metadata.max_query_len,
        metadata.max_seq_len - metadata.max_query_len,
        scale,
        CACHE_STRIDES=(*key_cache.stride(), *value_cache.stride()),
        BLOCK_ROWS=block_rows,
        **options,
    )
