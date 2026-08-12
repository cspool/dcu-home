#!/usr/bin/env python3
"""Measure the M=4096 TunableOp profile against the official default GEMM path."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as functional


SHAPES = (
    (14336, 4096, 5120),
    (16384, 4096, 5120),
    (34816, 4096, 5120),
    (5120, 4096, 17408),
    (5120, 4096, 6144),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def run(function, enabled: bool, iterations: int) -> float:
    torch.cuda.tunable.enable(enabled)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def benchmark_shape(n: int, m: int, k: int, groups: int, iterations: int) -> dict:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9364700 + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = 0.02 * torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16, generator=generator
    )

    def operation() -> torch.Tensor:
        return functional.linear(x, weight)

    torch.cuda.tunable.enable(False)
    official = operation()
    torch.cuda.tunable.enable(True)
    candidate = operation()
    torch.cuda.synchronize()
    delta = candidate.float() - official.float()
    correctness = {
        "finite": bool(torch.isfinite(candidate).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "allclose_atol_0.5_rtol_0.02": bool(
            torch.allclose(candidate, official, atol=0.5, rtol=0.02)
        ),
    }
    if not correctness["finite"] or not correctness["allclose_atol_0.5_rtol_0.02"]:
        raise AssertionError(correctness)

    for enabled in (False, True):
        for _ in range(8):
            run(operation, enabled, 1)
    samples = {"official_disabled": [], "profile_enabled": []}
    rng = random.Random(9364800 + n + k)
    for _ in range(groups):
        order = [False, True]
        rng.shuffle(order)
        for enabled in order:
            name = "profile_enabled" if enabled else "official_disabled"
            samples[name].append(run(operation, enabled, iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    gain = 100.0 * (
        medians["official_disabled"] - medians["profile_enabled"]
    ) / medians["official_disabled"]
    result = {
        "shape": {"N": n, "M_total_tokens": m, "K": k},
        "dp2_interpretation": "M remains 4096 when four local requests share the prefill token budget",
        "correctness": correctness,
        "samples_us": samples,
        "medians_us": medians,
        "profile_time_reduction_percent": gain,
    }
    print(json.dumps({"shape": result["shape"], "medians_us": medians, "gain_percent": gain}), flush=True)
    del x, weight, official, candidate, delta
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    torch.empty(0, device="cuda")
    torch.cuda.tunable.enable(False)
    torch.cuda.tunable.set_filename(str(args.profile.resolve()))
    torch.cuda.tunable.enable(True)
    loaded = torch.cuda.tunable.get_results()
    if not loaded:
        raise RuntimeError(f"empty TunableOp profile: {args.profile}")
    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    records = [
        benchmark_shape(n, m, k, args.groups, args.iterations)
        for n, m, k in SHAPES
    ]
    torch.cuda.tunable.enable(False)
    result = {
        "schema": "official-relative-m4096-tunable-b4-v1",
        "device": prop.gcnArchName,
        "profile": str(args.profile.resolve()),
        "profile_results_loaded": len(loaded),
        "groups": args.groups,
        "iterations": args.iterations,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
