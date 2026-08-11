# SPDX-License-Identifier: Apache-2.0
import torch
from flash_attn import varlen_fwd_unified

from vllm.triton_utils import tl, triton

# Call chain: Qwen3NextAttention -> ROCm AITER backend -> prefill -> pack -> two
# official FlashAttention calls -> _merge_page784; False returns route to GQA6.
# Role: accelerate later prefill by splitting each 784-token page into a regular
# 768-token main region and a small residual containing tails, boundary and new KV.
# Work logic: custom Triton packs the residual, official FA computes main/residual
# attention, and a custom LSE-weighted merge produces the complete output.

_PAGE_WORKSPACE: dict[tuple, tuple[torch.Tensor, ...]] = {}
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
    CACHE_STRIDES: tl.constexpr,
    CURRENT_STRIDES: tl.constexpr,
    PACKED_TOKEN_STRIDE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dimensions = tl.arange(0, 256)
    from_cache = token < residual_tokens
    # Correctness: residual order is page tails, boundary tokens, then current KV.
    from_page_tail = token < tail_tokens
    remainder_token = token - tail_tokens
    logical_page = tl.where(from_page_tail, token // 16, full_pages)
    position = tl.where(from_page_tail, 768 + token % 16, remainder_token % 784)

    # Correctness: cache and current KV may be independent interleaved views.
    physical_page = tl.load(block_table + logical_page, mask=from_cache, other=0)
    key_source = physical_page * CACHE_STRIDES[0]
    key_source += position * CACHE_STRIDES[1] + head * CACHE_STRIDES[2]
    key_source += dimensions * CACHE_STRIDES[3]
    value_source = physical_page * CACHE_STRIDES[4]
    value_source += position * CACHE_STRIDES[5] + head * CACHE_STRIDES[6]
    value_source += dimensions * CACHE_STRIDES[7]
    current_token = token - residual_tokens
    current_key_offset = current_token * CURRENT_STRIDES[0]
    current_key_offset += head * CURRENT_STRIDES[1] + dimensions * CURRENT_STRIDES[2]
    current_value_offset = current_token * CURRENT_STRIDES[3]
    current_value_offset += head * CURRENT_STRIDES[4] + dimensions * CURRENT_STRIDES[5]
    key = tl.load(key_cache + key_source, mask=from_cache, other=0.0)
    value = tl.load(value_cache + value_source, mask=from_cache, other=0.0)
    key = tl.where(
        from_cache,
        key,
        tl.load(current_key + current_key_offset, mask=~from_cache, other=0.0),
    )
    value = tl.where(
        from_cache,
        value,
        tl.load(current_value + current_value_offset, mask=~from_cache, other=0.0),
    )
    target = token * PACKED_TOKEN_STRIDE + head * 256 + dimensions
    tl.store(packed_key + target, key)
    tl.store(packed_value + target, value)


@triton.jit
def _merge_page784(
    output,
    main,
    main_lse,
    residual,
    residual_lse,
    query_len,
    QUERY_HEADS: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    token = row // QUERY_HEADS
    head = row % QUERY_HEADS
    output_offset = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    lse_offset = head * query_len + token
    main_score = tl.load(main_lse + lse_offset)
    residual_score = tl.load(residual_lse + lse_offset)
    max_score = tl.maximum(main_score, residual_score)
    main_weight = tl.exp(main_score - max_score)
    residual_weight = tl.exp(residual_score - max_score)
    normalizer = main_weight + residual_weight
    main_value = tl.load(main + output_offset).to(tl.float32)
    residual_value = tl.load(residual + output_offset).to(tl.float32)
    result = main_value * (main_weight / normalizer)[:, None]
    result += residual_value * (residual_weight / normalizer)[:, None]
    tl.store(output + output_offset, result)


def _page_workspace(
    query: torch.Tensor, page_count: int, query_heads: int, kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (query.device, query.dtype, query_heads, kv_heads)
    if key not in _PAGE_WORKSPACE:
        # Performance: allocate fixed workspaces once; returned slices are views.
        options = {"device": query.device, "dtype": query.dtype}
        token_buffers = (
            torch.empty((4096, query_heads, 256), **options) for _ in range(2)
        )
        page_buffers = (
            torch.empty((160, 64, kv_heads, 256), **options) for _ in range(2)
        )
        _PAGE_WORKSPACE[key] = (*token_buffers, *page_buffers)

    main, residual, packed_key, packed_value = _PAGE_WORKSPACE[key]
    return (
        main[: len(query)],
        residual[: len(query)],
        packed_key[:page_count],
        packed_value[:page_count],
    )


def prefill(
    query: torch.Tensor,
    current_key: torch.Tensor | None,
    current_value: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata,
    scale: float,
) -> bool:
    # Correctness: every False return happens before any output-writing kernel.
    # The backend can therefore run GQA6 without clearing or repairing output.
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
    query_heads, kv_heads = query.shape[1], current_key.shape[1]
    main, residual, packed_key, packed_value = _page_workspace(
        query, packed_pages, query_heads, kv_heads
    )
    packed_key_tokens = packed_key.view(-1, kv_heads, 256)
    packed_value_tokens = packed_value.view(-1, kv_heads, 256)
    block_table = metadata.block_table
    query_starts = metadata.query_start_loc

    _pack_page784[(combined_tokens, kv_heads)](
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
        CACHE_STRIDES=(*key_cache.stride(), *value_cache.stride()),
        CURRENT_STRIDES=(*current_key.stride(), *current_value.stride()),
        PACKED_TOKEN_STRIDE=packed_key_tokens.stride(0),
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

    # Performance: reuse one option set for the two official FA launches.
    attention_options = {
        "softmax_scale": scale,
        "window_size": (-1, -1),
        "return_softmax_lse": True,
    }
    # Correctness: history is noncausal; residual includes current KV and is causal.
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
    # Correctness: combine partial attention with LSE weights, not a plain average.
    _merge_page784[(query_len * query_heads // 4,)](
        output,
        main,
        main_lse,
        residual,
        residual_lse,
        query_len,
        QUERY_HEADS=query_heads,
        num_warps=4,
    )
    return True
