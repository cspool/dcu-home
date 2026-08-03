# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch
from flash_attn import flash_attn_interface as flash

from vllm.triton_utils import tl, triton



@triton.jit
def _pack_residual(source_k, source_v, table, packed_k, packed_v, tail, full):
    packed_token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, 256)
    is_tail = packed_token < tail
    logical_page = tl.where(is_tail, packed_token // 16, full)
    token = tl.where(is_tail, 768 + packed_token % 16, packed_token - tail)
    physical_page = tl.load(table + logical_page)
    source_offset = ((physical_page * 784 + token) * 4 + head) * 256 + dim
    packed_offset = (packed_token * 4 + head) * 256 + dim
    tl.store(packed_k + packed_offset, tl.load(source_k + source_offset))
    tl.store(packed_v + packed_offset, tl.load(source_v + source_offset))


@triton.jit
def _merge_states(output, output0, lse0, output1, lse1, output2, lse2):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    token = row // 24
    head = row % 24
    lse_offset = head * (tl.num_programs(0) // 6) + token
    score0 = tl.load(lse0 + lse_offset)
    score1 = tl.load(lse1 + lse_offset)
    score2 = tl.load(lse2 + lse_offset)
    maximum = tl.maximum(score0, tl.maximum(score1, score2))
    exp0 = tl.exp(score0 - maximum)
    exp1 = tl.exp(score1 - maximum)
    exp2 = tl.exp(score2 - maximum)
    denominator = exp0 + exp1 + exp2
    dim = tl.arange(0, 256)
    offset = row[:, None] * 256 + dim[None, :]
    value0 = tl.load(output0 + offset).to(tl.float32)
    value1 = tl.load(output1 + offset).to(tl.float32)
    value2 = tl.load(output2 + offset).to(tl.float32)
    merged = value0 * (exp0 / denominator)[:, None]
    merged += value1 * (exp1 / denominator)[:, None]
    merged += value2 * (exp2 / denominator)[:, None]
    tl.store(output + offset, merged)


@cache
def _workspace(device, dtype, main_tokens, residual_tokens, residual_pages):
    return (
        torch.empty((2, 4096, 24, 256), dtype=dtype, device=device),
        torch.empty((2, 96, 64, 4, 256), dtype=dtype, device=device),
        torch.tensor((main_tokens, residual_tokens), dtype=torch.int32, device=device),
        torch.arange(residual_pages, dtype=torch.int32, device=device)[None],
    )


def prefill(qkv, kv_cache, output, metadata):
    query = qkv[0]
    key_cache, value_cache = kv_cache
    cu_seqlens_q, block_table, max_seqlen_q, max_seqlen_k, softmax_scale = metadata
    full_pages, boundary = divmod(max_seqlen_k - max_seqlen_q, 784)
    tail_tokens = full_pages * 16
    residual_tokens = tail_tokens + boundary
    residual_pages = triton.cdiv(residual_tokens, 64)
    main_tokens = full_pages * 768
    key = (query.device, query.dtype, main_tokens, residual_tokens, residual_pages)
    outputs, packed, lengths, residual_table = _workspace(*key)
    main_output, residual_output = outputs[:, :max_seqlen_q]
    packed_k, packed_v = packed[:, :residual_pages]
    pack_args = (*kv_cache, block_table[0], packed_k, packed_v, tail_tokens, full_pages)
    _pack_residual[(residual_tokens, 4)](*pack_args, num_warps=4)
    def attention(k, v, length, table, tokens, out):
        return flash.varlen_fwd_unified(
            query, k, v, cu_seqlens_q, length, table,
            max_seqlen_q, tokens, softmax_scale,
            out=out, return_softmax_lse=True,
        )
    main_state = attention(
        key_cache[:, :768], value_cache[:, :768],
        lengths[:1], block_table[:, :full_pages], main_tokens, main_output,
    )
    residual_state = attention(
        packed_k, packed_v, lengths[1:], residual_table,
        residual_tokens, residual_output,
    )
    current_args = (*qkv, max_seqlen_q, cu_seqlens_q, max_seqlen_q)
    current_config = dict(
        cu_seqlens_k=cu_seqlens_q, softmax_scale=softmax_scale, causal=True,
        return_softmax_lse=True, out=output, is_prefix_cache=False,
    )
    current_state = flash.vllm_flash_attn_varlen_func(*current_args, **current_config)
    states = (main_state, residual_state, current_state)
    state_args = (output, *(tensor for state in states for tensor in state[:2]))
    _merge_states[(max_seqlen_q * 6,)](*state_args, num_warps=4)
