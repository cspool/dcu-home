#!/usr/bin/env python3
"""Compare the official and migrated packed-GDN decode schedules at B=1/4/8."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

H = 16
HV = 48
K = 128
V = 128
BK = 128
BV = 32
QKV_DIM = 2 * H * K + HV * V
SCALE = K**-0.5
CONFIGS = {
    "official_1w3s": {"num_warps": 1, "num_stages": 3},
    "migrated_4w1s": {"num_warps": 4, "num_stages": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ring", type=int, default=16)
    parser.add_argument("--graph-replays", type=int, default=8)
    parser.add_argument("--groups", type=int, default=41)
    return parser.parse_args()


def launch(kernel, tensors: dict[str, torch.Tensor], config: dict[str, int]) -> None:
    mixed = tensors["mixed"]
    a = tensors["a"]
    b = tensors["b"]
    state = tensors["state"]
    indices = tensors["indices"]
    batch = mixed.shape[0]
    kernel[(V // BV, batch * HV)](
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        o=tensors["out"],
        h0=state,
        ht=state,
        ssm_state_indices=indices,
        scale=SCALE,
        stride_mixed_qkv_tok=mixed.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0),
        stride_indices_seq=indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        **config,
    )


def make_correctness_inputs(batch: int, steps: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return {
        "mixed": torch.randn(
            (steps, batch, QKV_DIM), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "a": torch.randn(
            (steps, batch, HV), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "b": torch.randn(
            (steps, batch, HV), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "A_log": torch.full((HV,), -3.0, device="cuda", dtype=torch.bfloat16),
        "dt_bias": torch.full((HV,), -1.0, device="cuda", dtype=torch.bfloat16),
        "state": 0.05 * torch.randn(
            (batch, HV, V, K), device="cuda", dtype=torch.float32,
            generator=generator,
        ),
    }


def run_correctness(kernel, data: dict[str, torch.Tensor], config: dict[str, int]):
    batch = data["mixed"].shape[1]
    state = data["state"].clone()
    outputs = torch.empty(
        (data["mixed"].shape[0], batch, 1, HV, V),
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    for step in range(data["mixed"].shape[0]):
        launch(
            kernel,
            {
                "mixed": data["mixed"][step],
                "a": data["a"][step],
                "b": data["b"][step],
                "A_log": data["A_log"],
                "dt_bias": data["dt_bias"],
                "out": outputs[step],
                "state": state,
                "indices": indices,
            },
            config,
        )
    torch.cuda.synchronize()
    return outputs, state


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "allclose_atol_0.02_rtol_0.02": bool(
            torch.allclose(actual, expected, atol=0.02, rtol=0.02)
        ),
    }


def correctness(kernel, batch: int) -> dict:
    data = make_correctness_inputs(batch, steps=16, seed=9364200 + batch)
    official_out, official_state = run_correctness(
        kernel, data, CONFIGS["official_1w3s"]
    )
    migrated_out, migrated_state = run_correctness(
        kernel, data, CONFIGS["migrated_4w1s"]
    )
    result = {
        "output": metrics(migrated_out, official_out),
        "state": metrics(migrated_state, official_state),
    }
    if not result["output"]["allclose_atol_0.02_rtol_0.02"]:
        raise AssertionError(result)
    return result


def make_ring(batch: int, ring: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    indices = torch.arange(ring * batch, device="cuda", dtype=torch.int32)
    return {
        "mixed": torch.randn(
            (ring, batch, QKV_DIM), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "a": torch.randn(
            (ring, batch, HV), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "b": torch.randn(
            (ring, batch, HV), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ),
        "A_log": torch.full((HV,), -3.0, device="cuda", dtype=torch.bfloat16),
        "dt_bias": torch.full((HV,), -1.0, device="cuda", dtype=torch.bfloat16),
        "state": 0.05 * torch.randn(
            (ring * batch, HV, V, K), device="cuda", dtype=torch.float32,
            generator=generator,
        ),
        "out": torch.empty(
            (ring, batch, 1, HV, V), device="cuda", dtype=torch.bfloat16
        ),
        "indices": indices.reshape(ring, batch),
    }


def slot(tensors: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {
        "mixed": tensors["mixed"][index],
        "a": tensors["a"][index],
        "b": tensors["b"][index],
        "A_log": tensors["A_log"],
        "dt_bias": tensors["dt_bias"],
        "state": tensors["state"],
        "out": tensors["out"][index],
        "indices": tensors["indices"][index],
    }


def capture(kernel, tensors: dict[str, torch.Tensor], config: dict[str, int]):
    launch(kernel, slot(tensors, 0), config)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(tensors["mixed"].shape[0]):
            launch(kernel, slot(tensors, index), config)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graph: torch.cuda.CUDAGraph, replays: int, ring: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replays / ring


def benchmark(kernel, batch: int, ring: int, replays: int, groups: int) -> dict:
    tensor_sets = {
        name: make_ring(batch, ring, 9364300 + batch * 10 + offset)
        for offset, name in enumerate(CONFIGS)
    }
    graphs = {
        name: capture(kernel, tensor_sets[name], config)
        for name, config in CONFIGS.items()
    }
    samples = {name: [] for name in CONFIGS}
    rng = random.Random(9364400 + batch)
    for _ in range(groups):
        order = list(CONFIGS)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(graphs[name], replays, ring))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    official = medians["official_1w3s"]
    migrated = medians["migrated_4w1s"]
    gain = 100.0 * (official - migrated) / official
    result = {
        "batch": batch,
        "grid": [V // BV, batch * HV],
        "ring": ring,
        "graph_replays_per_sample": replays,
        "samples_us_per_batch_launch": samples,
        "medians_us_per_batch_launch": medians,
        "candidate_time_reduction_percent": gain,
        "candidate_saved_us_per_layer": official - migrated,
        "candidate_optimistic_saved_ms_per_48_layers": (official - migrated) * 48 / 1000,
    }
    print(json.dumps({"B": batch, "medians_us": medians, "gain_percent": gain}), flush=True)
    return result


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode_kernel,
    )

    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    records = []
    for batch in range(1, 9):
        records.append(
            {
                "batch": batch,
                "correctness": correctness(
                    fused_recurrent_gated_delta_rule_packed_decode_kernel, batch
                ),
                "performance": benchmark(
                    fused_recurrent_gated_delta_rule_packed_decode_kernel,
                    batch,
                    args.ring,
                    args.graph_replays,
                    args.groups,
                ),
            }
        )
        torch.cuda.empty_cache()
    result = {
        "schema": "official-relative-gdn-packed-decode-b4-v1",
        "device": prop.gcnArchName,
        "source": str(args.source.resolve()),
        "configs": CONFIGS,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
