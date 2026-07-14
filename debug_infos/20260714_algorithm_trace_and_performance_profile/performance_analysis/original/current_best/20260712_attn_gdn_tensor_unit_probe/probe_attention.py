#!/usr/bin/env python3
"""Launch one current Qwen3.5 GQA6 attention-core kernel without vLLM serve."""

from __future__ import annotations

import argparse
import math
import socket
import sys
from pathlib import Path


REPO = Path("/public/home/tangyu408/vllm_cscc")
CACHE_BLOCK = 784


def service_is_active() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=0.2):
            return True
    except OSError:
        return False


def make_inputs(torch, query_len: int, sequence_len: int):
    torch.manual_seed(20260712 + query_len + sequence_len)
    num_blocks = math.ceil(sequence_len / CACHE_BLOCK)
    query = torch.randn(
        (query_len, 24, 256), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        (num_blocks, CACHE_BLOCK, 4, 256), device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    query_start = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor(
        [sequence_len], device="cuda", dtype=torch.int32
    )
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32)[
        None, :
    ]
    return query, key, value, output, query_start, sequence_lengths, block_table


def launch_prefill(torch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_unified_attention_gqa6 import (
        kernel_unified_attention_2d_gqa6,
    )

    query_len, sequence_len = 512, 12000
    query, key, value, output, query_start, seq_lens, table = make_inputs(
        torch, query_len, sequence_len
    )
    grid = (query_len // 32 + 1, 4, 3)
    kernel_unified_attention_2d_gqa6[grid](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key,
        value_cache_ptr=value,
        block_tables_ptr=table,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=256**-0.5,
        k_scale=None,
        v_scale=None,
        softcap=0.0,
        num_query_heads=24,
        num_queries_per_kv=6,
        block_table_stride=table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=64,
        TOKENS_PER_BLOCK=56,
        CACHE_BLOCK_SIZE=CACHE_BLOCK,
        HEAD_SIZE=256,
        HEAD_SIZE_PADDED=256,
        USE_ALIBI_SLOPES=False,
        USE_SOFTCAP=False,
        SLIDING_WINDOW=0,
        stride_k_cache_0=key.stride(0),
        stride_k_cache_1=key.stride(1),
        stride_k_cache_2=key.stride(2),
        stride_k_cache_3=key.stride(3),
        stride_v_cache_0=value.stride(0),
        stride_v_cache_1=value.stride(1),
        stride_v_cache_2=value.stride(2),
        stride_v_cache_3=value.stride(3),
        query_start_len_ptr=query_start,
        BLOCK_Q=32,
        num_seqs=1,
        BLOCK_M=64,
        HEADS_PER_CTA=2,
        GQA_SPLITS=3,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    torch.cuda.synchronize()
    print("mode=prefill", "q=512", "seq=12000", "finite", bool(torch.isfinite(output).all()))


def launch_decode(torch) -> None:
    # Initialize Triton's ROCm driver through a current vLLM Triton kernel
    # before AITER calls the same driver.  Direct AITER-first initialization in
    # this container resolves an incompatible hipGetProcAddress header/runtime
    # pair, while production vLLM has already initialized Triton at this point.
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    heads, value_heads, dim = 16, 48, 128
    packed = torch.zeros(
        (1, 2 * heads * dim + value_heads * dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.zeros((1, value_heads), device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(
        (1, value_heads, dim, dim), device="cuda", dtype=torch.float32
    )
    gdn_out = torch.empty(
        (1, 1, value_heads, dim), device="cuda", dtype=torch.bfloat16
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        packed,
        gate,
        gate,
        torch.zeros(value_heads, device="cuda"),
        torch.zeros(value_heads, device="cuda"),
        dim**-0.5,
        state,
        gdn_out,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        validate=True,
    )
    torch.cuda.synchronize()

    from aiter.ops.triton.unified_attention import unified_attention

    query_len, sequence_len = 1, 12000
    query, key, value, output, query_start, seq_lens, table = make_inputs(
        torch, query_len, sequence_len
    )
    unified_attention(
        query,
        key,
        value,
        output,
        query_start,
        query_len,
        seq_lens,
        sequence_len,
        256**-0.5,
        True,
        (-1, -1),
        table,
        0.0,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    print("mode=decode", "q=1", "seq=12000", "finite", bool(torch.isfinite(output).all()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prefill", "decode"))
    args = parser.parse_args()
    if service_is_active():
        raise SystemExit("refuse: vLLM service is active on port 8001")
    # Prefill tests the current source-only H11.5 specialization.  Decode is
    # unchanged AITER code and uses the installed production package so its
    # native extensions participate in the same import order as vLLM serve.
    if args.mode == "prefill":
        sys.path.insert(0, str(REPO))
    import torch

    print("torch", torch.__version__)
    print("arch", torch.cuda.get_device_properties(0).gcnArchName)
    if args.mode == "prefill":
        launch_prefill(torch)
    else:
        launch_decode(torch)


if __name__ == "__main__":
    main()
