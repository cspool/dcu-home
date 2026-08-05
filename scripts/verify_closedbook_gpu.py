#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import statistics
from pathlib import Path

import torch

from qwen35_rocm_opt import attention as portable_attention
from qwen35_rocm_opt.gdn import qwen35_gdn_rmsnorm
from qwen35_rocm_opt.gemv import qwen35_gemv
from qwen35_rocm_opt.native import load_k5120_provider
from vllm.v1.attention.ops import rocm_aiter_unified_attention_gqa6 as frozen_attention


def timed_pair(left, right, weight, x, rows, calls=1000, rounds=11):
    """Time both providers in alternating order to cancel clock ramp effects."""
    for _ in range(100):
        left(weight, x, rows)
        right(weight, x, rows)
    torch.cuda.synchronize()
    samples = {left: [], right: []}

    def measure(provider):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            provider(weight, x, rows)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / calls

    for round_index in range(rounds):
        order = (left, right) if round_index % 2 else (right, left)
        for provider in order:
            samples[provider].append(measure(provider))
    return tuple(statistics.median(samples[provider]) for provider in (left, right))


def check_attention() -> None:
    tokens, context, page = 128, 760, 784
    q = torch.randn((tokens, 24, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, page, 4, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    seq = torch.tensor([context + tokens], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    frozen = torch.empty_like(q)
    portable = torch.empty_like(q)
    args = dict(
        q=q,
        k=k,
        v=v,
        block_table=block_table,
        seqused_k=seq,
        cu_seqlens_q=starts,
        max_seqlen_q=tokens,
        softmax_scale=256**-0.5,
    )
    frozen_attention.prefill(out=frozen, **args)
    portable_attention.prefill(out=portable, **args)
    torch.cuda.synchronize()
    if not torch.equal(frozen, portable):
        raise AssertionError("portable page784 attention differs from frozen kernel")


def check_gdn() -> None:
    class Norm:
        def __init__(self):
            self.weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, z):
            value = x.float()
            value *= torch.rsqrt((value * value).mean(-1, keepdim=True) + self.eps)
            gate = z.float()
            return (value * self.weight.float() * gate * torch.sigmoid(gate)).to(x.dtype)

    norm = Norm()
    core = torch.randn((32, 48, 128), device="cuda", dtype=torch.bfloat16)
    storage = torch.randn((32, 16384), device="cuda", dtype=torch.bfloat16)
    z = storage.as_strided((32, 48, 128), (16384, 128, 1))
    output = qwen35_gdn_rmsnorm(norm, core, z)
    reference = norm(core.reshape(-1, 128), z.reshape(-1, 128)).reshape_as(core)
    if not torch.allclose(output, reference, atol=0.02, rtol=0.02):
        raise AssertionError("portable fused GDN norm differs from reference")


def check_output_gemv() -> None:
    x = torch.randn((1, 17408), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((5120, 17408), device="cuda", dtype=torch.bfloat16)
    output = qwen35_gemv(weight, x)
    reference = torch.nn.functional.linear(x, weight)
    if output is None or not torch.allclose(output, reference, atol=2.0, rtol=0.02):
        raise AssertionError("portable K=17408 GEMV differs from torch linear")


def check_native(frozen_library: Path) -> None:
    portable = load_k5120_provider()
    torch.ops.load_library(str(frozen_library))
    frozen = torch.ops._rocm_C.LLMM1
    for output_features in (96, 14336, 16384, 34816):
        rows = 4 if output_features == 96 else 2
        x = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            (output_features, 5120),
            device="cuda",
            dtype=torch.bfloat16,
        )
        portable_output = portable(weight, x, rows)
        frozen_output = frozen(weight, x, rows)
        if not torch.equal(portable_output, frozen_output):
            raise AssertionError(f"K=5120 output mismatch for M={output_features}")
        portable_ms, frozen_ms = timed_pair(portable, frozen, weight, x, rows)
        delta = (portable_ms / frozen_ms - 1) * 100
        print(
            f"K5120 M={output_features}: portable={portable_ms:.6f} ms, "
            f"frozen={frozen_ms:.6f} ms, delta={delta:+.3f}%"
        )
        if delta > 1.0:
            raise AssertionError(f"portable K=5120 provider regressed {delta:.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frozen-rocm-library", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260805)
    check_attention()
    check_gdn()
    check_output_gemv()
    if args.frozen_rocm_library:
        check_native(args.frozen_rocm_library)
    print("verify_closedbook_gpu: OK")


if __name__ == "__main__":
    main()
