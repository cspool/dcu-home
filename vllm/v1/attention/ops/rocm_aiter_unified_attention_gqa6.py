# SPDX-License-Identifier: Apache-2.0
"""gfx936/BF16/head256/GQA6 specialization of AITER unified attention.

This is a narrowly-scoped copy of AITER's non-segmented 2D attention kernel.
It preserves the original online-softmax math and cache-block selection while
fixing the GQA6 row overlap caused by AITER's ``BLOCK_M=16, BLOCK_Q=2`` mapping.

H11.3 partitions the six query heads belonging to each KV head into three
two-head groups.  Each program therefore covers exactly ``8 tokens * 2 heads``
with a legal power-of-two ``tl.arange(0, 16)``.  The third launch-grid axis is
only a GQA head-group axis; it is not AITER's segmented 3D decode algorithm and
does not require temporary segment buffers or a reduction kernel.
"""
from functools import lru_cache
import torch
import triton
import triton.language as tl
_H11_3_SHORT_PREFILL_COMPILER_CONFIG = {'num_warps': 4, 'num_stages': 2, 'waves_per_eu': 1}
_H11_4_LONG_PREFILL_COMPILER_CONFIG = {'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}
_H11_5_WIDE_CAUSAL_PREFILL_COMPILER_CONFIG = {'num_warps': 4, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}

@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y

@triton.jit
def _apply_softcap(scores, softcap):
    scaled = scores / softcap
    pos = tl.exp(scaled)
    neg = tl.exp(-scaled)
    return softcap * (pos - neg) / (pos + neg)

@triton.jit
def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        value = tl.load(query_start_len_ptr + mid)
        mid_value = value // BLOCK_Q + mid if use_q_block_mode else value
        if mid_value <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1

@triton.jit
def kernel_unified_attention_2d_gqa6(output_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, seq_lens_ptr, alibi_slopes_ptr, scale, k_scale, v_scale, softcap, num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr, block_table_stride: tl.int64, query_stride_0: tl.int64, query_stride_1: tl.int64, output_stride_0: tl.int64, output_stride_1: tl.int64, BLOCK_SIZE: tl.constexpr, TOKENS_PER_BLOCK: tl.constexpr, CACHE_BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, USE_ALIBI_SLOPES: tl.constexpr, USE_SOFTCAP: tl.constexpr, SLIDING_WINDOW: tl.constexpr, stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64, stride_k_cache_2: tl.int64, stride_k_cache_3: tl.constexpr, stride_v_cache_0: tl.int64, stride_v_cache_1: tl.int64, stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr, query_start_len_ptr, BLOCK_Q: tl.constexpr, num_seqs: tl.int32, BLOCK_M: tl.constexpr, HEADS_PER_CTA: tl.constexpr, GQA_SPLITS: tl.constexpr, NUMERIC_WIDTH: tl.constexpr, NUMERIC_SUBTILES: tl.constexpr, NUMERIC_SUBTILE: tl.constexpr=0):
    tl.static_assert(TOKENS_PER_BLOCK <= BLOCK_SIZE, 'TOKENS_PER_BLOCK must fit in the padded BLOCK_SIZE')
    tl.static_assert(num_queries_per_kv == HEADS_PER_CTA * GQA_SPLITS, 'GQA heads must be partitioned exactly')
    tl.static_assert(BLOCK_M == BLOCK_Q * HEADS_PER_CTA, 'BLOCK_M must contain only complete token/head groups')
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    q_head_group_idx = tl.program_id(2)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_start = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_stop - cur_batch_start
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // HEADS_PER_CTA
    query_offset_0 = cur_batch_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + q_head_group_idx * HEADS_PER_CTA + offs_m % HEADS_PER_CTA
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)
    query = tl.load(query_ptr + query_offset, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0)
    block_table_offset = seq_idx * block_table_stride
    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_sum = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)
    causal_query_stop = tl.minimum((q_block_local_idx + 1) * BLOCK_Q, cur_batch_query_len)
    num_blocks = _cdiv(context_len + causal_query_stop, TOKENS_PER_BLOCK)
    tl.static_assert(BLOCK_SIZE % NUMERIC_WIDTH == 0, 'numeric width must divide the logical I/O tile')
    tl.static_assert(NUMERIC_SUBTILES == BLOCK_SIZE // NUMERIC_WIDTH, 'numeric subtile count must cover the logical I/O tile')
    if NUMERIC_SUBTILE != 0:
        tl.static_assert(TOKENS_PER_BLOCK == BLOCK_SIZE, 'numeric subtiles require a full logical I/O tile')
        tl.static_assert(NUMERIC_SUBTILE == NUMERIC_WIDTH, 'legacy numeric-subtile width must match NUMERIC_WIDTH')
    else:
        tl.static_assert(NUMERIC_WIDTH == BLOCK_SIZE and NUMERIC_SUBTILES == 1, 'the unsplit path must contain exactly one numerical tile')
    for block_idx in range(0, num_blocks):
        for subtile_idx in tl.static_range(0, NUMERIC_SUBTILES):
            offs_n = tl.arange(0, NUMERIC_WIDTH)
            subtile_start = subtile_idx * NUMERIC_WIDTH
            token_mask = subtile_start + offs_n < TOKENS_PER_BLOCK
            start_n = block_idx * TOKENS_PER_BLOCK + subtile_start
            if TOKENS_PER_BLOCK == CACHE_BLOCK_SIZE:
                physical_block_idx = tl.load(block_tables_ptr + block_table_offset + start_n // CACHE_BLOCK_SIZE)
                offset_in_page = start_n % CACHE_BLOCK_SIZE + offs_n
                value_offset = physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + offset_in_page[:, None] * stride_v_cache_1
                key_offset = physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + offset_in_page[None, :] * stride_k_cache_1
            elif CACHE_BLOCK_SIZE % TOKENS_PER_BLOCK == 0:
                physical_block_idx = tl.load(block_tables_ptr + block_table_offset + start_n // CACHE_BLOCK_SIZE)
                offset_in_page = (start_n + offs_n) % CACHE_BLOCK_SIZE
                value_offset = physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + offset_in_page[:, None] * stride_v_cache_1
                key_offset = physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + offset_in_page[None, :] * stride_k_cache_1
            else:
                logical_page_idx = start_n // CACHE_BLOCK_SIZE
                offset_in_first_page = start_n % CACHE_BLOCK_SIZE
                tokens_in_first_page = CACHE_BLOCK_SIZE - offset_in_first_page
                first_physical_block_idx = tl.load(block_tables_ptr + block_table_offset + logical_page_idx)
                crosses_page = tokens_in_first_page < NUMERIC_WIDTH
                second_physical_block_idx = tl.load(block_tables_ptr + block_table_offset + logical_page_idx + 1, mask=crosses_page, other=first_physical_block_idx)
                use_second_page = offs_n >= tokens_in_first_page
                physical_block_idx = tl.where(use_second_page, second_physical_block_idx, first_physical_block_idx)
                offset_in_page = tl.where(use_second_page, offs_n - tokens_in_first_page, offset_in_first_page + offs_n)
                value_offset = physical_block_idx[:, None] * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + offset_in_page[:, None] * stride_v_cache_1
                key_offset = physical_block_idx[None, :] * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + offset_in_page[None, :] * stride_k_cache_1
            key_load = tl.load(key_cache_ptr + key_offset, mask=dim_mask[:, None] & token_mask[None, :], other=0.0)
            if key_load.dtype.is_fp8():
                if query.dtype.is_fp8():
                    key = key_load
                else:
                    key = (key_load.to(tl.float32) * tl.load(k_scale)).to(query.dtype)
            else:
                key = key_load
            value_load = tl.load(value_cache_ptr + value_offset, mask=token_mask[:, None] & dim_mask[None, :], other=0.0)
            if value_load.dtype.is_fp8():
                if query.dtype.is_fp8():
                    value = value_load
                else:
                    value = (value_load.to(tl.float32) * tl.load(v_scale)).to(query.dtype)
            else:
                value = value_load
            seq_offset = start_n + offs_n
            seq_mask = token_mask[None, :] & (seq_offset[None, :] < context_len + query_pos[:, None] + 1)
            scores = tl.zeros((BLOCK_M, NUMERIC_WIDTH), dtype=tl.float32)
            scores += scale * tl.dot(query, key)
            if USE_SOFTCAP:
                scores = _apply_softcap(scores, softcap)
            scores = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, scores, float('-inf'))
            if SLIDING_WINDOW > 0:
                scores = tl.where(context_len + query_pos[:, None] - seq_offset < SLIDING_WINDOW, scores, float('-inf'))
            if USE_ALIBI_SLOPES:
                scores += alibi_slope[:, None] * (seq_offset - context_len)
            block_max = tl.maximum(running_max, tl.max(scores, axis=1))
            block_max = tl.where(block_max > float('-inf'), block_max, 0.0)
            probabilities = tl.exp(scores - block_max[:, None])
            block_sum = tl.sum(probabilities, axis=1)
            correction = tl.exp(running_max - block_max)
            acc *= correction[:, None]
            running_sum = running_sum * correction + block_sum
            running_max = block_max
            acc += tl.dot(probabilities.to(value.dtype), value)
    acc /= running_sum[:, None]
    output_offset = query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 + offs_d[None, :]
    tl.store(output_ptr + output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])

@lru_cache
def _find_block(cache_block_size: int, max_block_size: int) -> int | None:
    if cache_block_size < 16 or max_block_size < 16:
        return None
    for exponent in range(max_block_size.bit_length() - 1, 3, -1):
        block_size = 1 << exponent
        if cache_block_size % block_size == 0:
            return block_size
    return None

def unified_attention_gqa6_prefill(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale, alibi_slopes=None):
    """Run the H11.3 non-overlapping prefill specialization."""
    del max_seqlen_k
    assert causal, 'Only causal attention is supported'
    assert q_descale is None, 'Q scales are not supported'
    assert max_seqlen_q > 1, 'H11.3 is prefill-only'
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert q.ndim == 3 and q.shape[1:] == (24, 256)
    assert k.ndim == 4 and k.shape[2:] == (4, 256)
    assert v.ndim == 4 and v.shape[2:] == (4, 256)
    use_alibi_slopes = alibi_slopes is not None
    element_size = v.element_size()
    cache_block_size = v.shape[1]
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    block_size = cache_block_size
    if block_size * head_size_padded * element_size > 16384:
        block_size = 16384 // (head_size_padded * element_size)
        block_size = _find_block(cache_block_size, block_size)
        assert block_size is not None, 'cannot find a suitable block size for H11.3 unified attention'
    assert element_size >= 2 or block_size >= 32
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    tokens_per_block = block_size
    heads_per_cta = 2
    gqa_splits = 3
    assert num_queries_per_kv == heads_per_cta * gqa_splits
    numeric_width = block_size
    numeric_subtiles = 1
    numeric_subtile = 0
    use_h11_5_wide_causal = cache_block_size == 784 and num_seqs == 1 and (max_seqlen_q >= 128)
    if use_h11_5_wide_causal:
        block_m = 64
        block_q = block_m // heads_per_cta
        tokens_per_block = 64
        block_size = 64
        numeric_width = 32
        numeric_subtiles = 2
        numeric_subtile = 32
        compiler_config = _H11_5_WIDE_CAUSAL_PREFILL_COMPILER_CONFIG
    elif max_seqlen_q >= 128:
        block_m = 16
        block_q = block_m // heads_per_cta
        compiler_config = _H11_4_LONG_PREFILL_COMPILER_CONFIG
    else:
        block_m = 16
        block_q = block_m // heads_per_cta
        compiler_config = _H11_3_SHORT_PREFILL_COMPILER_CONFIG
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
    kernel_unified_attention_2d_gqa6[total_num_q_blocks, num_kv_heads, gqa_splits](output_ptr=out, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, block_tables_ptr=block_table, seq_lens_ptr=seqused_k, alibi_slopes_ptr=alibi_slopes, scale=softmax_scale, k_scale=k_descale, v_scale=v_descale, softcap=softcap, num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv, block_table_stride=block_table.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0), output_stride_1=out.stride(1), BLOCK_SIZE=block_size, TOKENS_PER_BLOCK=tokens_per_block, CACHE_BLOCK_SIZE=cache_block_size, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=head_size_padded, USE_ALIBI_SLOPES=use_alibi_slopes, USE_SOFTCAP=softcap > 0, SLIDING_WINDOW=1 + window_size[0], stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=cu_seqlens_q, BLOCK_Q=block_q, num_seqs=num_seqs, BLOCK_M=block_m, HEADS_PER_CTA=heads_per_cta, GQA_SPLITS=gqa_splits, NUMERIC_WIDTH=numeric_width, NUMERIC_SUBTILES=numeric_subtiles, NUMERIC_SUBTILE=numeric_subtile, **compiler_config)
