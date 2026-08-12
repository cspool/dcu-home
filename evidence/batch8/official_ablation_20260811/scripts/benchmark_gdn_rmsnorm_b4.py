#!/usr/bin/env python3
"""Paired official-vs-fused Qwen3.5 GDN epilogue benchmark at B=1/4/8."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=31)
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


def benchmark(tokens: int, norm, fused, groups: int, iterations: int) -> dict:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9364000 + tokens)
    core = torch.randn(
        (tokens, 48, 128), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    gate_storage = torch.randn(
        (tokens, 16384), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    gate = gate_storage.as_strided((tokens, 48, 128), (16384, 128, 1))

    def official() -> torch.Tensor:
        return norm(core.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(core)

    def candidate() -> torch.Tensor:
        return fused(norm, core, gate)

    expected = official()
    actual = candidate()
    torch.cuda.synchronize()
    delta = actual.float() - expected.float()
    correctness = {
        "allclose_atol_0.02_rtol_0.02": bool(
            torch.allclose(actual, expected, atol=0.02, rtol=0.02)
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
    }
    if not correctness["allclose_atol_0.02_rtol_0.02"]:
        raise AssertionError(correctness)

    for _ in range(20):
        official()
        candidate()
    torch.cuda.synchronize()

    functions = {"official": official, "fused": candidate}
    samples = {name: [] for name in functions}
    rng = random.Random(9364100 + tokens)
    for _ in range(groups):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            samples[name].append(elapsed_us(functions[name], iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    gain = 100.0 * (medians["official"] - medians["fused"]) / medians["official"]
    result = {
        "tokens": tokens,
        "logical_batch": tokens,
        "shape": list(core.shape),
        "gate_stride": list(gate.stride()),
        "official_gate_reshape_copies": gate.reshape(-1, 128).data_ptr() != gate.data_ptr(),
        "correctness": correctness,
        "samples_us": samples,
        "medians_us": medians,
        "candidate_time_reduction_percent": gain,
    }
    print(json.dumps({"B": tokens, "medians_us": medians, "gain_percent": gain}), flush=True)
    return result


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from vllm.model_executor.layers.fla.ops.gfx936 import qwen35_gdn_rmsnorm
    from vllm.model_executor.layers.fla.ops.layernorm_guard import RMSNormGated

    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    norm = RMSNormGated(
        128,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    norm.weight.data.normal_()
    result = {
        "schema": "official-relative-gdn-rmsnorm-b4-v1",
        "device": prop.gcnArchName,
        "source": str(args.source.resolve()),
        "groups": args.groups,
        "iterations": args.iterations,
        "records": [
            benchmark(tokens, norm, qwen35_gdn_rmsnorm, args.groups, args.iterations)
            for tokens in (1, 4, 8, 16, 4096)
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
