# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx936 single-request decode specialization for AITER GQA6 attention.

The stock AITER decode path splits one query over 16 context segments.  With
four KV heads that creates only 64 main workgroups on an 80-CU gfx936 device.
This specialization uses 20 active segments, producing exactly 80 main
workgroups.  Triton reduction axes must be powers of two, so the final kernel
reduces a 32-lane vector while masking lanes 20--31 and retaining a 20-segment
storage stride.

Only temporary FP32 segment buffers and their reduction geometry change.  The
AITER main kernel, BF16 Q/K/V tensors, KV-cache layout, and model weights are
unchanged.

The main kernel is imported from AMD AITER.  The reduction below is derived
from AITER's Apache-2.0 ``reduce_segments`` implementation; provenance and
version details are recorded in ``THIRD_PARTY_NOTICES.md``.
"""

import torch
import triton
import triton.language as tl

from aiter.ops.triton.unified_attention import (
    cdiv_fn,
    find_seq_idx,
    kernel_unified_attention_3d,
)


_ACTIVE_SEGMENTS = 20
_REDUCTION_SEGMENTS = 32
_NUM_QUERY_HEADS = 24
_NUM_KV_HEADS = 4
_HEAD_SIZE = 256
_CACHE_BLOCK_SIZE = 784
_BLOCK_SIZE = 16
_BLOCK_M = 16
_BLOCK_Q = 2


@triton.jit
def reduce_segments_gqa6_80cu(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    block_table_stride: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    REDUCTION_SEGMENTS: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    seq_idx = find_seq_idx(
        query_start_len_ptr,
        query_token_idx,
        num_seqs,
        BLOCK_Q,
        False,
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    blocks_per_segment = cdiv_fn(
        seq_len, NUM_SEGMENTS_PER_SEQ * BLOCK_SIZE
    )
    active_segments = cdiv_fn(
        seq_len, blocks_per_segment * BLOCK_SIZE
    )

    segment_offsets = tl.arange(0, REDUCTION_SEGMENTS)
    segment_mask = (segment_offsets < active_segments) & (
        segment_offsets < NUM_SEGMENTS_PER_SEQ
    )
    dimension_offsets = tl.arange(0, HEAD_SIZE_PADDED)
    dimension_mask = dimension_offsets < HEAD_SIZE

    segment_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + segment_offsets
    )
    segment_max = tl.load(
        segm_max_ptr + segment_offset,
        mask=segment_mask,
        other=float("-inf"),
    )
    overall_max = tl.max(segment_max)

    segment_expsum = tl.load(
        segm_expsum_ptr + segment_offset,
        mask=segment_mask,
        other=0.0,
    )
    segment_expsum *= tl.exp(segment_max - overall_max)
    overall_expsum = tl.sum(segment_expsum)

    segment_output_offset = (
        query_token_idx.to(tl.int64)
        * (
            num_query_heads
            * NUM_SEGMENTS_PER_SEQ
            * HEAD_SIZE_PADDED
        )
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segment_offsets[:, None] * HEAD_SIZE_PADDED
        + dimension_offsets[None, :]
    )
    segment_output = tl.load(
        segm_output_ptr + segment_output_offset,
        mask=segment_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    segment_output *= tl.exp(segment_max - overall_max)[:, None]
    output_sum = tl.sum(segment_output, axis=0)
    result = tl.where(
        overall_expsum == 0.0,
        0.0,
        output_sum / overall_expsum,
    )

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + dimension_offsets
    )
    tl.store(output_ptr + output_offset, result, mask=dimension_mask)


def unified_attention_gqa6_decode_80cu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    """Run the exact single-request gfx936 GQA6 decode specialization."""
    del max_seqlen_k
    assert causal
    assert q_descale is None
    assert alibi_slopes is None
    assert max_seqlen_q == 1
    assert q.shape == (1, _NUM_QUERY_HEADS, _HEAD_SIZE)
    assert out.ndim == 3 and out.shape == q.shape
    assert k.ndim == 4 and k.shape[1:] == (
        _CACHE_BLOCK_SIZE,
        _NUM_KV_HEADS,
        _HEAD_SIZE,
    )
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert cu_seqlens_q.numel() == 2
    assert seqused_k.numel() == 1
    assert block_table.ndim == 2 and block_table.shape[0] == 1
    assert softcap in (0, 0.0, None)
    assert window_size == (-1, -1)

    segment_output = torch.empty(
        (1, _NUM_QUERY_HEADS, _ACTIVE_SEGMENTS, _HEAD_SIZE),
        dtype=torch.float32,
        device=q.device,
    )
    segment_max = torch.empty(
        (1, _NUM_QUERY_HEADS, _ACTIVE_SEGMENTS),
        dtype=torch.float32,
        device=q.device,
    )
    segment_expsum = torch.empty_like(segment_max)

    kernel_unified_attention_3d[
        (1, _NUM_KV_HEADS, _ACTIVE_SEGMENTS)
    ](
        segm_output_ptr=segment_output,
        segm_max_ptr=segment_max,
        segm_expsum_ptr=segment_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=0.0,
        num_query_heads=_NUM_QUERY_HEADS,
        num_queries_per_kv=_NUM_QUERY_HEADS // _NUM_KV_HEADS,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        CACHE_BLOCK_SIZE=_CACHE_BLOCK_SIZE,
        HEAD_SIZE=_HEAD_SIZE,
        HEAD_SIZE_PADDED=_HEAD_SIZE,
        USE_ALIBI_SLOPES=False,
        USE_SOFTCAP=False,
        SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=_BLOCK_Q,
        num_seqs=1,
        BLOCK_M=_BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=_ACTIVE_SEGMENTS,
    )
    reduce_segments_gqa6_80cu[(1, _NUM_QUERY_HEADS)](
        output_ptr=out,
        segm_output_ptr=segment_output,
        segm_max_ptr=segment_max,
        segm_expsum_ptr=segment_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=1,
        num_query_heads=_NUM_QUERY_HEADS,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        BLOCK_SIZE=_BLOCK_SIZE,
        HEAD_SIZE=_HEAD_SIZE,
        HEAD_SIZE_PADDED=_HEAD_SIZE,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=_BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=_ACTIVE_SEGMENTS,
        REDUCTION_SEGMENTS=_REDUCTION_SEGMENTS,
    )
