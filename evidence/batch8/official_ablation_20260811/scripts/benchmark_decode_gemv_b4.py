#!/usr/bin/env python3
"""Test whether the batch-1 decode GEMV kernels still beat official GEMM at B=4."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as functional
from vllm.triton_utils import tl, triton


@triton.jit
def _k17408_b4_gemv(weight, x, output, M: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    batch = tl.program_id(1)
    offsets = tl.arange(0, 2048)
    accumulator = tl.zeros((2048,), dtype=tl.float32)
    for start in range(0, 17408, 2048):
        columns = start + offsets
        mask = columns < 17408
        accumulator += tl.load(
            x + batch * 17408 + columns, mask=mask, other=0.0
        ).to(tl.float32) * tl.load(
            weight + row * 17408 + columns, mask=mask, other=0.0
        ).to(tl.float32)
    tl.store(output + batch * M + row, tl.sum(accumulator))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args()


def elapsed_us(function: Callable[[], torch.Tensor], iterations: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def paired(functions: dict[str, Callable[[], torch.Tensor]], groups: int, iterations: int, seed: int) -> dict:
    for _ in range(20):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(seed)
    for _ in range(groups):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            samples[name].append(elapsed_us(functions[name], iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline = medians["official_f_linear"]
    candidate = medians["batch4_custom"]
    return {
        "samples_us": samples,
        "medians_us": medians,
        "candidate_time_reduction_percent": 100.0 * (baseline - candidate) / baseline,
    }


def correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    result = {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "allclose_atol_0.25_rtol_0.02": bool(
            torch.allclose(actual, expected, atol=0.25, rtol=0.02)
        ),
    }
    if not result["finite"] or not result["allclose_atol_0.25_rtol_0.02"]:
        raise AssertionError(result)
    return result


def llmm1_rows(weight: torch.Tensor, x: torch.Tensor, rows_per_block: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.ops._rocm_C.LLMM1(weight, x[row : row + 1], rows_per_block)
            for row in range(x.shape[0])
        ],
        dim=0,
    )


def benchmark_llmm1_shape(
    output_features: int,
    groups: int,
    iterations: int,
    qwen35_gemv,
) -> dict:
    batch = 4
    input_features = 5120
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9364500 + output_features)
    x = torch.randn(
        (batch, input_features), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    weight = 0.02 * torch.randn(
        (output_features, input_features), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    rows_per_block = 4 if output_features == 96 else 2

    def official() -> torch.Tensor:
        return functional.linear(x, weight)

    def candidate() -> torch.Tensor:
        return llmm1_rows(weight, x, rows_per_block)

    expected = official()
    actual = candidate()
    torch.cuda.synchronize()
    result = {
        "family": "LLMM1_K5120",
        "shape": {"input": list(x.shape), "weight": list(weight.shape)},
        "current_batch4_gate_returns_none": qwen35_gemv(weight, x) is None,
        "hypothetical_migration": "four independent M=1 LLMM1 launches plus concatenate",
        "correctness": correctness(actual, expected),
        "performance": paired(
            {"official_f_linear": official, "batch4_custom": candidate},
            groups,
            iterations,
            9364600 + output_features,
        ),
    }
    print(
        json.dumps(
            {
                "shape": result["shape"],
                "medians_us": result["performance"]["medians_us"],
                "gain_percent": result["performance"]["candidate_time_reduction_percent"],
            }
        ),
        flush=True,
    )
    del x, weight, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_k17408(groups: int, iterations: int, qwen35_gemv) -> dict:
    batch = 4
    input_features = 17408
    output_features = 5120
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9364517)
    x = torch.randn(
        (batch, input_features), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    weight = 0.02 * torch.randn(
        (output_features, input_features), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )

    def official() -> torch.Tensor:
        return functional.linear(x, weight)

    def candidate() -> torch.Tensor:
        output = torch.empty(
            (batch, output_features), device="cuda", dtype=torch.bfloat16
        )
        _k17408_b4_gemv[(output_features, batch)](
            weight,
            x,
            output,
            M=output_features,
            B=batch,
            num_warps=16,
            num_stages=1,
            waves_per_eu=1,
        )
        return output

    expected = official()
    actual = candidate()
    torch.cuda.synchronize()
    result = {
        "family": "Triton_K17408",
        "shape": {"input": list(x.shape), "weight": list(weight.shape)},
        "current_batch4_gate_returns_none": qwen35_gemv(weight, x) is None,
        "hypothetical_migration": "native B4 grid=(5120,4), 16 warps, one stage",
        "correctness": correctness(actual, expected),
        "performance": paired(
            {"official_f_linear": official, "batch4_custom": candidate},
            groups,
            iterations,
            9364617,
        ),
    }
    print(
        json.dumps(
            {
                "shape": result["shape"],
                "medians_us": result["performance"]["medians_us"],
                "gain_percent": result["performance"]["candidate_time_reduction_percent"],
            }
        ),
        flush=True,
    )
    del x, weight, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import vllm._rocm_C  # noqa: F401
    from vllm.model_executor.layers.fla.ops.gfx936 import qwen35_gemv

    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")

    records = [benchmark_k17408(args.groups, args.iterations, qwen35_gemv)] + [
        benchmark_llmm1_shape(output_features, args.groups, args.iterations, qwen35_gemv)
        for output_features in (96, 14336, 16384, 34816, 248320)
    ]
    result = {
        "schema": "official-relative-decode-gemv-b4-v1",
        "device": prop.gcnArchName,
        "source": str(args.source.resolve()),
        "groups": args.groups,
        "iterations": args.iterations,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
