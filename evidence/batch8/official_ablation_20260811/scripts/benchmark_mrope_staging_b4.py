#!/usr/bin/env python3
"""Source-faithful M-RoPE staging traffic model for a fixed 4096-token graph."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=101)
    return parser.parse_args()


def elapsed_us(function: Callable[[], object], iterations: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def paired(functions: dict[str, Callable[[], object]], groups: int, iterations: int, seed: int) -> dict:
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
    baseline = medians["official_noncontiguous"]
    candidate = medians["contiguous_fixed_graph"]
    return {
        "samples_us": samples,
        "medians_us": medians,
        "candidate_time_reduction_percent": 100.0 * (baseline - candidate) / baseline,
    }


def benchmark_case(scheduled_tokens: int, groups: int, iterations: int) -> dict:
    max_tokens = 4096
    official_cpu = torch.arange(3 * (max_tokens + 1), dtype=torch.int64).reshape(
        3, max_tokens + 1
    ).pin_memory()
    official_gpu = torch.empty_like(official_cpu, device="cuda")
    official_materialized = torch.empty((3, max_tokens), device="cuda", dtype=torch.int64)

    candidate_cpu = official_cpu[:, :max_tokens].contiguous().pin_memory()
    candidate_gpu = torch.empty_like(candidate_cpu, device="cuda")

    def official() -> torch.Tensor:
        official_gpu[:, :scheduled_tokens].copy_(
            official_cpu[:, :scheduled_tokens], non_blocking=True
        )
        # The fixed graph consumes a contiguous [3,4096] input.  The official
        # +1-stride buffer therefore needs the 96-KiB materialization represented here.
        return official_materialized.copy_(official_gpu[:, :max_tokens])

    def candidate() -> torch.Tensor:
        # The migrated exact-4096 path copies once into the persistent graph input.
        candidate_gpu.copy_(candidate_cpu, non_blocking=True)
        return candidate_gpu

    official()
    candidate()
    torch.cuda.synchronize()
    correctness = {
        "active_positions_equal": bool(
            torch.equal(
                official_materialized[:, :scheduled_tokens],
                candidate_gpu[:, :scheduled_tokens],
            )
        ),
        "official_stride": list(official_gpu[:, :max_tokens].stride()),
        "candidate_stride": list(candidate_gpu.stride()),
    }
    result = paired(
        {"official_noncontiguous": official, "contiguous_fixed_graph": candidate},
        groups,
        iterations,
        9365000 + scheduled_tokens,
    )
    return {
        "scheduled_tokens": scheduled_tokens,
        "global_batch_interpretation": "4 decode tokens on one DP rank" if scheduled_tokens == 4 else "full 4096-token prefill budget",
        "correctness": correctness,
        "traffic_bytes": {
            "official_h2d_active": 3 * scheduled_tokens * 8,
            "official_d2d_materialization": 3 * max_tokens * 8,
            "candidate_h2d_fixed": 3 * max_tokens * 8,
            "candidate_d2d_materialization": 0,
        },
        "performance": result,
    }


def main() -> int:
    args = parse_args()
    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    records = [
        benchmark_case(tokens, args.groups, args.iterations) for tokens in (4, 4096)
    ]
    result = {
        "schema": "official-relative-mrope-staging-b4-traffic-model-v1",
        "device": prop.gcnArchName,
        "scope_warning": "This isolates the source-level staging and required contiguous graph materialization; it is not a full model E2E ablation.",
        "groups": args.groups,
        "iterations": args.iterations,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for record in records:
        print(json.dumps({"tokens": record["scheduled_tokens"], **record["performance"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
