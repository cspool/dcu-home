# SPDX-License-Identifier: Apache-2.0
"""Aligned page784 wrapper for the vendor unified page-Prefill FA kernel."""

from __future__ import annotations

import flash_attn_2_cuda
import torch
from flash_attn.flash_attn_interface import varlen_fwd_unified

from vllm.triton_utils import tl, triton


PAGE_SIZE = 784
MAIN_SIZE = 768
TAIL_SIZE = PAGE_SIZE - MAIN_SIZE


@triton.jit
def _pack_page784_residual_kernel(
    source_k,
    source_v,
    source_block_table,
    packed_k,
    packed_v,
    source_block_stride,
    source_token_stride,
    source_head_stride,
    source_dim_stride,
    packed_token_stride,
    packed_head_stride,
    packed_dim_stride,
    tail_tokens,
    full_pages,
    MAIN_SIZE_CONST: tl.constexpr,
    TAIL_SIZE_CONST: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    packed_token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK_DIM)
    dim_mask = dim < HEAD_DIM

    is_tail = packed_token < tail_tokens
    logical_page = tl.where(
        is_tail,
        packed_token // TAIL_SIZE_CONST,
        full_pages,
    )
    token_in_page = tl.where(
        is_tail,
        MAIN_SIZE_CONST + packed_token % TAIL_SIZE_CONST,
        packed_token - tail_tokens,
    )
    physical_page = tl.load(source_block_table + logical_page)
    source_offset = (
        physical_page * source_block_stride
        + token_in_page * source_token_stride
        + head * source_head_stride
        + dim * source_dim_stride
    )
    packed_offset = (
        packed_token * packed_token_stride
        + head * packed_head_stride
        + dim * packed_dim_stride
    )
    k_value = tl.load(source_k + source_offset, mask=dim_mask)
    v_value = tl.load(source_v + source_offset, mask=dim_mask)
    tl.store(packed_k + packed_offset, k_value, mask=dim_mask)
    tl.store(packed_v + packed_offset, v_value, mask=dim_mask)


@triton.jit
def _merge_three_attn_states_kernel(
    output,
    output_0,
    lse_0,
    output_1,
    lse_1,
    output_2,
    lse_2,
    out_token_stride,
    out_head_stride,
    out_dim_stride,
    state0_token_stride,
    state0_head_stride,
    state0_dim_stride,
    state1_token_stride,
    state1_head_stride,
    state1_dim_stride,
    state2_token_stride,
    state2_head_stride,
    state2_dim_stride,
    lse0_head_stride,
    lse0_token_stride,
    lse1_head_stride,
    lse1_token_stride,
    lse2_head_stride,
    lse2_token_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK_DIM)
    dim_mask = dim < HEAD_DIM

    score_0 = tl.load(lse_0 + head * lse0_head_stride + token * lse0_token_stride)
    score_1 = tl.load(lse_1 + head * lse1_head_stride + token * lse1_token_stride)
    score_2 = tl.load(lse_2 + head * lse2_head_stride + token * lse2_token_stride)
    maximum = tl.maximum(score_0, tl.maximum(score_1, score_2))
    exp_0 = tl.exp(score_0 - maximum)
    exp_1 = tl.exp(score_1 - maximum)
    exp_2 = tl.exp(score_2 - maximum)
    denominator = exp_0 + exp_1 + exp_2
    weight_0 = exp_0 / denominator
    weight_1 = exp_1 / denominator
    weight_2 = exp_2 / denominator

    offset_0 = (
        token * state0_token_stride
        + head * state0_head_stride
        + dim * state0_dim_stride
    )
    offset_1 = (
        token * state1_token_stride
        + head * state1_head_stride
        + dim * state1_dim_stride
    )
    offset_2 = (
        token * state2_token_stride
        + head * state2_head_stride
        + dim * state2_dim_stride
    )
    value_0 = tl.load(output_0 + offset_0, mask=dim_mask).to(tl.float32)
    value_1 = tl.load(output_1 + offset_1, mask=dim_mask).to(tl.float32)
    value_2 = tl.load(output_2 + offset_2, mask=dim_mask).to(tl.float32)
    merged = value_0 * weight_0 + value_1 * weight_1 + value_2 * weight_2
    output_offset = (
        token * out_token_stride + head * out_head_stride + dim * out_dim_stride
    )
    tl.store(output + output_offset, merged, mask=dim_mask)


_metadata_cache: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}
MAX_QUERY_TOKENS = 4096
MAX_CONTEXT_TOKENS = 262144
MAX_RESIDUAL_PAGES = (
    (MAX_CONTEXT_TOKENS // PAGE_SIZE) * TAIL_SIZE + (PAGE_SIZE - 1) + 63
) // 64

_workspace_cache: dict[
    tuple[int, int, int, int, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _get_metadata(
    device: torch.device,
    main_tokens: int,
    residual_tokens: int,
    residual_pages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (_device_index(device), main_tokens, residual_tokens, residual_pages)
    cached = _metadata_cache.get(key)
    if cached is None:
        cached = (
            torch.tensor([main_tokens], dtype=torch.int32, device=device),
            torch.tensor([residual_tokens], dtype=torch.int32, device=device),
            torch.arange(residual_pages, dtype=torch.int32, device=device)[None],
        )
        _metadata_cache[key] = cached
    return cached


def _get_workspace(
    query: torch.Tensor,
    residual_pages: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Cache one fixed-capacity workspace per device/layout.  The old exact-shape
    # key retained two Q-sized outputs for every prompt remainder and eventually
    # consumed the entire 0.95-memory-utilization margin during full `all`.
    assert query.shape[0] <= MAX_QUERY_TOKENS
    assert residual_pages <= MAX_RESIDUAL_PAGES
    key = (
        _device_index(query.device),
        query.shape[1],
        query.shape[2],
        num_kv_heads,
        query.dtype,
    )
    cached = _workspace_cache.get(key)
    if cached is None:
        cached = (
            torch.empty(
                (MAX_QUERY_TOKENS, query.shape[1], query.shape[2]),
                dtype=query.dtype,
                device=query.device,
            ),
            torch.empty(
                (MAX_QUERY_TOKENS, query.shape[1], query.shape[2]),
                dtype=query.dtype,
                device=query.device,
            ),
            torch.empty(
                (MAX_RESIDUAL_PAGES, 64, num_kv_heads, query.shape[2]),
                dtype=query.dtype,
                device=query.device,
            ),
            torch.empty(
                (MAX_RESIDUAL_PAGES, 64, num_kv_heads, query.shape[2]),
                dtype=query.dtype,
                device=query.device,
            ),
        )
        _workspace_cache[key] = cached
    return (
        cached[0][: query.shape[0]],
        cached[1][: query.shape[0]],
        cached[2][:residual_pages],
        cached[3][:residual_pages],
    )


def _canonical_lse(lse: torch.Tensor) -> torch.Tensor:
    if lse.ndim == 3:
        assert lse.shape[0] == 1
        return lse[0]
    assert lse.ndim == 2
    return lse


def _contiguous_current_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = flash_attn_2_cuda.varlen_fwd(
        query,
        key,
        value,
        output,
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        None,
        None,
        query.shape[0],
        key.shape[0],
        0.0,
        softmax_scale,
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
    return result[0], _canonical_lse(result[5])


def page784_split_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> None:
    """Run the aligned three-part later-Prefill path for one request."""
    assert query.shape == output.shape
    assert query.shape[0] == max_seqlen_q
    assert key.shape[0] == value.shape[0] == max_seqlen_q
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert key_cache.dtype == value_cache.dtype == torch.bfloat16
    assert query.shape[1:] == (24, 256)
    assert key.shape[1:] == value.shape[1:] == (4, 256)
    assert key_cache.ndim == value_cache.ndim == 4
    assert key_cache.shape[1:] == value_cache.shape[1:] == (PAGE_SIZE, 4, 256)
    assert cu_seqlens_q.numel() == 2
    assert block_table.ndim == 2 and block_table.shape[0] == 1
    assert query.stride(-1) == key.stride(-1) == value.stride(-1) == 1

    context_len = max_seqlen_k - max_seqlen_q
    full_pages = context_len // PAGE_SIZE
    boundary_len = context_len - full_pages * PAGE_SIZE
    assert full_pages > 0
    assert block_table.shape[1] * PAGE_SIZE >= max_seqlen_k
    tail_tokens = full_pages * TAIL_SIZE
    residual_tokens = tail_tokens + boundary_len
    residual_pages = (residual_tokens + 63) // 64
    assert residual_tokens > 0 and residual_pages > 0

    main_len_tensor, residual_len_tensor, residual_table = _get_metadata(
        query.device,
        full_pages * MAIN_SIZE,
        residual_tokens,
        residual_pages,
    )
    main_output, residual_output, packed_k, packed_v = _get_workspace(
        query, residual_pages, key.shape[1]
    )

    packed_flat_k = packed_k.view(-1, key.shape[1], key.shape[2])
    packed_flat_v = packed_v.view(-1, value.shape[1], value.shape[2])
    _pack_page784_residual_kernel[(residual_tokens, key.shape[1])](
        key_cache,
        value_cache,
        block_table[0],
        packed_flat_k,
        packed_flat_v,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        packed_flat_k.stride(0),
        packed_flat_k.stride(1),
        packed_flat_k.stride(2),
        tail_tokens,
        full_pages,
        MAIN_SIZE_CONST=MAIN_SIZE,
        TAIL_SIZE_CONST=TAIL_SIZE,
        HEAD_DIM=key.shape[2],
        BLOCK_DIM=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )

    main_state = varlen_fwd_unified(
        query,
        key_cache[:, :MAIN_SIZE],
        value_cache[:, :MAIN_SIZE],
        cu_seqlens_q,
        main_len_tensor,
        block_table[:, :full_pages],
        max_seqlen_q,
        full_pages * MAIN_SIZE,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        out=main_output,
        return_softmax_lse=True,
    )
    residual_state = varlen_fwd_unified(
        query,
        packed_k,
        packed_v,
        cu_seqlens_q,
        residual_len_tensor,
        residual_table,
        max_seqlen_q,
        residual_tokens,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        out=residual_output,
        return_softmax_lse=True,
    )
    current_state = _contiguous_current_fa(
        query,
        key,
        value,
        output,
        cu_seqlens_q,
        softmax_scale,
    )

    state0_output, state0_lse = main_state
    state1_output, state1_lse = residual_state
    state2_output, state2_lse = current_state
    state0_lse = _canonical_lse(state0_lse)
    state1_lse = _canonical_lse(state1_lse)
    state2_lse = _canonical_lse(state2_lse)
    _merge_three_attn_states_kernel[(query.shape[0], query.shape[1])](
        output,
        state0_output,
        state0_lse,
        state1_output,
        state1_lse,
        state2_output,
        state2_lse,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        state0_output.stride(0),
        state0_output.stride(1),
        state0_output.stride(2),
        state1_output.stride(0),
        state1_output.stride(1),
        state1_output.stride(2),
        state2_output.stride(0),
        state2_output.stride(1),
        state2_output.stride(2),
        state0_lse.stride(0),
        state0_lse.stride(1),
        state1_lse.stride(0),
        state1_lse.stride(1),
        state2_lse.stride(0),
        state2_lse.stride(1),
        HEAD_DIM=query.shape[2],
        BLOCK_DIM=triton.next_power_of_2(query.shape[2]),
        num_warps=4,
    )
