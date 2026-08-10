# SPDX-License-Identifier: Apache-2.0
import torch
from flash_attn.flash_attn_interface import varlen_fwd_unified

from vllm.triton_utils import tl, triton

_PAGE_WORKSPACE: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}
_PAGE_METADATA: dict[tuple, tuple[torch.Tensor, ...]] = {}


@triton.jit
def _pack_page784(
    key_cache,
    value_cache,
    current_key,
    current_value,
    block_table,
    packed_key,
    packed_value,
    residual_tokens,
    tail_tokens,
    full_pages,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dimensions = tl.arange(0, 256)
    from_cache = token < residual_tokens
    from_page_tail = token < tail_tokens
    remainder_token = token - tail_tokens
    logical_page = tl.where(from_page_tail, token // 16, full_pages)
    position = tl.where(from_page_tail, 768 + token % 16, remainder_token % 784)

    physical_page = tl.load(block_table + logical_page, mask=from_cache, other=0)
    source = physical_page * (784 * 4 * 256)
    source += position * (4 * 256) + head * 256 + dimensions
    current_token = token - residual_tokens
    current_offset = current_token * 1024 + head * 256 + dimensions
    key = tl.load(key_cache + source, mask=from_cache, other=0.0)
    value = tl.load(value_cache + source, mask=from_cache, other=0.0)
    key = tl.where(
        from_cache,
        key,
        tl.load(current_key + current_offset, mask=~from_cache, other=0.0),
    )
    value = tl.where(
        from_cache,
        value,
        tl.load(current_value + current_offset, mask=~from_cache, other=0.0),
    )
    target = token * 1024 + head * 256 + dimensions
    tl.store(packed_key + target, key)
    tl.store(packed_value + target, value)


@triton.jit
def _merge_page784(
    output,
    main,
    main_lse,
    residual,
    residual_lse,
    row_count,
    query_len,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    valid = row < row_count
    token = row // 24
    head = row % 24
    output_offset = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    lse_offset = head * query_len + token
    main_score = tl.load(main_lse + lse_offset, mask=valid, other=-float("inf"))
    residual_score = tl.load(residual_lse + lse_offset, mask=valid, other=-float("inf"))
    max_score = tl.maximum(main_score, residual_score)
    main_weight = tl.exp(main_score - max_score)
    residual_weight = tl.exp(residual_score - max_score)
    normalizer = main_weight + residual_weight

    mask = valid[:, None]
    main_value = tl.load(main + output_offset, mask=mask).to(tl.float32)
    residual_value = tl.load(residual + output_offset, mask=mask).to(tl.float32)
    result = main_value * (main_weight / normalizer)[:, None]
    result += residual_value * (residual_weight / normalizer)[:, None]
    tl.store(output + output_offset, result, mask=mask)


def _page_workspace(
    query: torch.Tensor, page_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (query.device, query.dtype)
    if key not in _PAGE_WORKSPACE:
        options = {"device": query.device, "dtype": query.dtype}
        token_buffers = (torch.empty((4096, 24, 256), **options) for _ in range(2))
        page_buffers = (torch.empty((160, 64, 4, 256), **options) for _ in range(2))
        _PAGE_WORKSPACE[key] = (*token_buffers, *page_buffers)

    main, residual, packed_key, packed_value = _PAGE_WORKSPACE[key]
    return (
        main[: len(query)],
        residual[: len(query)],
        packed_key[:page_count],
        packed_value[:page_count],
    )


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
    query_len = metadata.max_query_len
    context_len = metadata.max_seq_len - query_len
    supported = current_key is not None and current_value is not None
    supported &= query_len >= 128 and context_len >= 784
    supported &= metadata.query_start_loc.numel() == 2
    if not supported:
        return False

    full_pages, boundary_tokens = divmod(context_len, 784)
    tail_tokens = full_pages * 16
    residual_tokens = tail_tokens + boundary_tokens
    combined_tokens = residual_tokens + query_len
    packed_pages = (combined_tokens + 63) // 64
    if (
        query_len > 4096
        or packed_pages > 160
        or metadata.num_actual_tokens != query_len
    ):
        return False

    query, output = (tensor[:query_len] for tensor in (query, output))
    main, residual, packed_key, packed_value = _page_workspace(query, packed_pages)
    packed_key_tokens = packed_key.view(-1, 4, 256)
    packed_value_tokens = packed_value.view(-1, 4, 256)
    block_table = metadata.block_table
    query_starts = metadata.query_start_loc

    _pack_page784[(combined_tokens, 4)](
        key_cache,
        value_cache,
        current_key,
        current_value,
        block_table,
        packed_key_tokens,
        packed_value_tokens,
        residual_tokens,
        tail_tokens,
        full_pages,
        num_warps=4,
    )

    metadata_key = (query.device, full_pages, combined_tokens, packed_pages)
    if metadata_key not in _PAGE_METADATA:
        options = {"dtype": torch.int32, "device": query.device}
        main_len = torch.tensor([full_pages * 768], **options)
        residual_len = torch.tensor([combined_tokens], **options)
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
        combined_tokens,
        causal=True,
        out=residual,
        **attention_options,
    )
    main_lse, residual_lse = (
        lse[0] if lse.ndim == 3 else lse for lse in (main_lse, residual_lse)
    )
    row_count = query_len * 24
    _merge_page784[(triton.cdiv(row_count, 4),)](
        output,
        main,
        main_lse,
        residual,
        residual_lse,
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
    query_len,
    context_len,
    scale,
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
            cache_offset = page * (784 * 4 * 256) + position * (4 * 256)
            cache_offset += kv_head * 256
            keys = tl.load(key_cache + cache_offset[None, :] + dimensions[:, None])
            values = tl.load(value_cache + cache_offset[:, None] + dimensions[None, :])
            causal = start + columns[None, :] < context_len + local_token[:, None] + 1
            scores = scale * tl.dot(query_values, keys)
            scores = tl.where(valid_query[:, None] & causal, scores, float("-inf"))
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
    current_key: torch.Tensor | None,
    current_value: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata,
    scale: float,
) -> None:
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
        metadata.max_query_len,
        metadata.max_seq_len - metadata.max_query_len,
        scale,
        BLOCK_ROWS=block_rows,
        **options,
    )
