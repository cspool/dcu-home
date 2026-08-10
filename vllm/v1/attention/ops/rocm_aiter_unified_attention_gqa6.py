# SPDX-License-Identifier: Apache-2.0
"""gfx936 GQA6 prefill kernels, including the 784-token page fast path."""

import torch
import flash_attn_2_cuda
from flash_attn.flash_attn_interface import varlen_fwd_unified

from vllm.triton_utils import tl, triton

_PAGE_WORKSPACE: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}
_PAGE_METADATA: dict[tuple, tuple[torch.Tensor, ...]] = {}


@triton.jit
def _pack_page784(
    key_cache,
    value_cache,
    block_table,
    packed_key,
    packed_value,
    tail_tokens,
    page_stride,
    token_stride,
    head_stride,
    dim_stride,
):
    """Pack each 16-token page tail plus the final partial page."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    dimensions = tl.arange(0, 256)

    from_tail = token < tail_tokens
    logical_page = tl.where(from_tail, token // 16, tail_tokens // 16)
    position = tl.where(from_tail, 768 + token % 16, token - tail_tokens)

    source = tl.load(block_table + logical_page) * page_stride
    source += position * token_stride + head * head_stride
    source += dimensions * dim_stride
    target = token * 1024 + head * 256 + dimensions
    tl.store(packed_key + target, tl.load(key_cache + source))
    tl.store(packed_value + target, tl.load(value_cache + source))


@triton.jit
def _merge_page784(
    output,
    main,
    main_lse,
    residual,
    residual_lse,
    current,
    current_lse,
    row_count,
    query_len,
):
    """Merge main-page, residual, and causal-current attention using FP32 LSE."""
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    valid = row < row_count
    token = row // 24
    head = row % 24
    output_offset = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    lse_offset = head * query_len + token

    main_score = tl.load(main_lse + lse_offset, mask=valid, other=-float("inf"))
    residual_score = tl.load(
        residual_lse + lse_offset,
        mask=valid,
        other=-float("inf"),
    )
    current_score = tl.load(
        current_lse + lse_offset,
        mask=valid,
        other=-float("inf"),
    )
    max_score = tl.maximum(main_score, tl.maximum(residual_score, current_score))
    main_weight = tl.exp(main_score - max_score)
    residual_weight = tl.exp(residual_score - max_score)
    current_weight = tl.exp(current_score - max_score)
    normalizer = main_weight + residual_weight + current_weight
    main_weight /= normalizer
    residual_weight /= normalizer
    current_weight /= normalizer

    mask = valid[:, None]
    main_value = tl.load(main + output_offset, mask=mask).to(tl.float32)
    residual_value = tl.load(residual + output_offset, mask=mask).to(tl.float32)
    current_value = tl.load(current + output_offset, mask=mask).to(tl.float32)
    result = main_value * main_weight[:, None]
    result += residual_value * residual_weight[:, None]
    result += current_value * current_weight[:, None]
    tl.store(output + output_offset, result, mask=mask)


def _page_workspace(
    query: torch.Tensor,
    page_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (query.device, query.dtype)
    if key not in _PAGE_WORKSPACE:
        options = {"device": query.device, "dtype": query.dtype}
        main = torch.empty((4096, 24, 256), **options)
        residual = torch.empty_like(main)
        packed_key = torch.empty((96, 64, 4, 256), **options)
        packed_value = torch.empty_like(packed_key)
        _PAGE_WORKSPACE[key] = (main, residual, packed_key, packed_value)

    main, residual, packed_key, packed_value = _PAGE_WORKSPACE[key]
    return (
        main[: len(query)],
        residual[: len(query)],
        packed_key[:page_count],
        packed_value[:page_count],
    )


def _current_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_starts: torch.Tensor,
    query_len: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The direct ABI keeps the caller-provided output buffer. Output and LSE
    # are elements zero and five of the result tuple.
    result = flash_attn_2_cuda.varlen_fwd(
        query,
        key,
        value,
        output,
        query_starts,
        query_starts,
        None,
        None,
        None,
        None,
        query_len,
        query_len,
        0.0,
        scale,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    return result[0], result[5]


def page784_prefill(
    query: torch.Tensor,
    current_key: torch.Tensor | None,
    current_value: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata,
    scale: float,
) -> bool:
    """Run the page784 split-and-merge path, returning whether it took over."""
    query_len = metadata.max_query_len
    context_len = metadata.max_seq_len - query_len
    supported = (
        current_key is not None
        and current_value is not None
        and query_len >= 128
        and context_len >= 784
        and metadata.query_start_loc.numel() == 2
    )
    if not supported:
        return False

    full_pages, boundary_tokens = divmod(context_len, 784)
    tail_tokens = full_pages * 16
    residual_tokens = tail_tokens + boundary_tokens
    packed_pages = (residual_tokens + 63) // 64
    if (
        query_len > 4096
        or packed_pages > 96
        or metadata.num_actual_tokens != query_len
    ):
        return False

    query = query[:query_len]
    current_key = current_key[:query_len]
    current_value = current_value[:query_len]
    output = output[:query_len]
    main, residual, packed_key, packed_value = _page_workspace(query, packed_pages)
    packed_key_tokens = packed_key.view(-1, 4, 256)
    packed_value_tokens = packed_value.view(-1, 4, 256)
    block_table = metadata.block_table
    query_starts = metadata.query_start_loc

    _pack_page784[(residual_tokens, 4)](
        key_cache,
        value_cache,
        block_table,
        packed_key_tokens,
        packed_value_tokens,
        tail_tokens,
        *key_cache.stride(),
        num_warps=4,
    )

    metadata_key = (query.device, full_pages, residual_tokens, packed_pages)
    if metadata_key not in _PAGE_METADATA:
        options = {"dtype": torch.int32, "device": query.device}
        main_len = torch.tensor([full_pages * 768], **options)
        residual_len = torch.tensor([residual_tokens], **options)
        residual_table = torch.arange(packed_pages, **options)[None]
        _PAGE_METADATA[metadata_key] = (main_len, residual_len, residual_table)
    main_len, residual_len, residual_table = _PAGE_METADATA[metadata_key]

    attention_options = {
        "softmax_scale": scale,
        "window_size": (-1, -1),
        "return_softmax_lse": True,
    }
    main, main_lse = varlen_fwd_unified(
        query,
        key_cache[:, :768],
        value_cache[:, :768],
        query_starts,
        main_len,
        block_table[:, :full_pages],
        query_len,
        full_pages * 768,
        causal=False,
        out=main,
        **attention_options,
    )
    residual, residual_lse = varlen_fwd_unified(
        query,
        packed_key,
        packed_value,
        query_starts,
        residual_len,
        residual_table,
        query_len,
        residual_tokens,
        causal=False,
        out=residual,
        **attention_options,
    )
    current, current_lse = _current_attention(
        query,
        current_key,
        current_value,
        output,
        query_starts,
        query_len,
        scale,
    )
    main_lse, residual_lse, current_lse = (
        lse[0] if lse.ndim == 3 else lse
        for lse in (main_lse, residual_lse, current_lse)
    )

    row_count = query_len * 24
    _merge_page784[(triton.cdiv(row_count, 4),)](
        output,
        main,
        main_lse,
        residual,
        residual_lse,
        current,
        current_lse,
        row_count,
        query_len,
        num_warps=4,
    )
    return True


@triton.jit
def _gqa6_prefill(
    output,
    query,
    key_cache,
    value_cache,
    block_table,
    sequence_lengths,
    query_starts,
    scale,
    page_stride: tl.constexpr,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    dim_stride: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_group = tl.program_id(1)
    kv_head = query_group // 3
    head_pair = query_group % 3
    rows = tl.arange(0, BLOCK_ROWS)
    dimensions = tl.arange(0, 256)
    tokens_per_block: tl.constexpr = BLOCK_ROWS // 2

    query_start = tl.load(query_starts)
    query_len = tl.load(query_starts + 1) - query_start
    if query_block * tokens_per_block >= query_len:
        return

    local_token = query_block * tokens_per_block + rows // 2
    query_token = query_start + local_token
    query_head = kv_head * 6 + head_pair * 2 + rows % 2
    query_offset = (
        query_token[:, None] * 6144
        + query_head[:, None] * 256
        + dimensions[None, :]
    )
    valid_query = local_token < query_len
    query_values = tl.load(
        query + query_offset,
        mask=valid_query[:, None],
        other=0.0,
    )

    max_score = tl.full([BLOCK_ROWS], float("-inf"), dtype=tl.float32)
    normalizer = tl.full([BLOCK_ROWS], 1.0, dtype=tl.float32)
    weighted_values = tl.zeros([BLOCK_ROWS, 256], dtype=tl.float32)
    context_len = tl.load(sequence_lengths) - query_len
    query_stop = tl.minimum((query_block + 1) * tokens_per_block, query_len)
    cache_blocks = (context_len + query_stop + BLOCK_ROWS - 1) // BLOCK_ROWS
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
            cache_offset = (
                page * page_stride
                + position * token_stride
                + kv_head * head_stride
            )
            keys = tl.load(
                key_cache
                + cache_offset[None, :]
                + dimensions[:, None] * dim_stride
            )
            values = tl.load(
                value_cache
                + cache_offset[:, None]
                + dimensions[None, :] * dim_stride
            )
            causal = (
                start + columns[None, :]
                < context_len + local_token[:, None] + 1
            )
            scores = scale * tl.dot(query_values, keys)
            scores = tl.where(
                valid_query[:, None] & causal,
                scores,
                float("-inf"),
            )
            next_max = tl.maximum(max_score, tl.max(scores, axis=1))
            next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
            probabilities = tl.exp(scores - next_max[:, None])
            correction = tl.exp(max_score - next_max)
            weighted_values *= correction[:, None]
            normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
            max_score = next_max
            weighted_values += tl.dot(probabilities.to(values.dtype), values)

    result = weighted_values / normalizer[:, None]
    output_offset = (
        query_token[:, None] * 6144
        + query_head[:, None] * 256
        + dimensions[None, :]
    )
    tl.store(output + output_offset, result, mask=valid_query[:, None])


def prefill(
    query: torch.Tensor,
    current_key: torch.Tensor | None,
    current_value: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata,
    scale: float,
) -> None:
    """Run page784 when eligible, otherwise the general single-sequence GQA6 path."""
    if page784_prefill(
        query,
        current_key,
        current_value,
        key_cache,
        value_cache,
        output,
        metadata,
        scale,
    ):
        return

    query = query[: metadata.num_actual_tokens]
    output = output[: metadata.num_actual_tokens]
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
        metadata.seq_lens,
        metadata.query_start_loc,
        scale,
        *key_cache.stride(),
        BLOCK_ROWS=block_rows,
        **options,
    )
