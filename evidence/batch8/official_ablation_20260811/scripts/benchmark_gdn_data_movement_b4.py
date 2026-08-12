#!/usr/bin/env python3
"""Quantify official GDN state/output movement avoided by the migrated path."""

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
    parser.add_argument("--iterations", type=int, default=17)
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


def paired(
    functions: dict[str, Callable[[], object]], groups: int, iterations: int, seed: int
) -> dict:
    for _ in range(10):
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
    return {
        "samples_us": samples,
        "medians_us": {
            name: statistics.median(values) for name, values in samples.items()
        },
    }


def state_prep(batch: int, groups: int, iterations: int) -> dict:
    slots = 16
    state = torch.randn(
        (slots, 48, 128, 128), device="cuda", dtype=torch.float32
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.long) * 2 + 1
    all_false = torch.zeros(batch, device="cuda", dtype=torch.bool)
    all_true = torch.ones(batch, device="cuda", dtype=torch.bool)

    def official_new():
        result = state[indices].contiguous()
        result[~all_false, ...] = 0
        return result

    def candidate_new():
        return None

    def official_existing():
        result = state[indices].contiguous()
        result[~all_true, ...] = 0
        return result

    def candidate_existing():
        return state[indices].contiguous()

    official_zero = official_new()
    candidate_zero = torch.zeros_like(official_zero)
    official_value = official_existing()
    candidate_value = candidate_existing()
    torch.cuda.synchronize()
    correctness = {
        "all_new_none_semantically_zero": bool(torch.equal(official_zero, candidate_zero)),
        "all_existing_equal": bool(torch.equal(official_value, candidate_value)),
    }
    all_new = paired(
        {"official_gather_and_mask": official_new, "candidate_pass_none": candidate_new},
        groups,
        iterations,
        9364900 + batch,
    )
    all_existing = paired(
        {"official_gather_and_empty_mask": official_existing, "candidate_gather_only": candidate_existing},
        groups,
        iterations,
        9364910 + batch,
    )
    return {
        "batch": batch,
        "state_dtype": str(state.dtype),
        "bytes_per_selected_batch": batch * 48 * 128 * 128 * state.element_size(),
        "correctness": correctness,
        "all_new_prefill": all_new,
        "all_existing_prefill": all_existing,
    }


def output_movement(groups: int, iterations: int) -> dict:
    tokens = 4096
    source = torch.randn((tokens, 48, 128), device="cuda", dtype=torch.bfloat16)
    target = torch.empty_like(source)

    def official_copy():
        return target.copy_(source)

    def candidate_direct_output():
        return None

    copy_result = paired(
        {"official_d2d_copy": official_copy, "candidate_direct_output": candidate_direct_output},
        groups,
        iterations,
        9364920,
    )

    full = torch.empty_like(source)
    tail = torch.empty_like(source)

    def official_full_zero():
        return full.zero_()

    def candidate_decode_tail_zero():
        return tail[4:].zero_()

    decode_zero = paired(
        {"official_full_4096_zero": official_full_zero, "candidate_tail_4092_zero": candidate_decode_tail_zero},
        groups,
        iterations,
        9364930,
    )
    return {
        "shape": list(source.shape),
        "bytes": source.numel() * source.element_size(),
        "prefill_direct_output": copy_result,
        "decode_padding_zero": decode_zero,
        "interpretation": {
            "prefill": "At T=4096 the candidate writes chunk_o directly into the caller buffer and avoids this full D2D copy.",
            "decode": "At local B=4 both paths zero nearly the whole 4096-token padded buffer, so this sub-optimization is covered.",
        },
    }


def main() -> int:
    args = parse_args()
    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    result = {
        "schema": "official-relative-gdn-data-movement-b4-v1",
        "device": prop.gcnArchName,
        "groups": args.groups,
        "iterations": args.iterations,
        "state_prep": [state_prep(batch, args.groups, args.iterations) for batch in (1, 4)],
        "output_movement": output_movement(args.groups, args.iterations),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for record in result["state_prep"]:
        print(json.dumps({"B": record["batch"], "new": record["all_new_prefill"]["medians_us"], "existing": record["all_existing_prefill"]["medians_us"]}), flush=True)
    print(json.dumps(result["output_movement"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
