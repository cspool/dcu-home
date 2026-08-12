#!/usr/bin/env python3
"""Screen BM32 versus the current fallback for transient local DP batches 1-8."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.rocm_aiter_unified_attention_gqa6 import _gqa6


CASES = {
    1: ([128], [24576]),
    2: ([64, 128], [16384, 24576]),
    3: ([16, 64, 128], [4096, 16384, 24576]),
    4: ([16, 32, 64, 128], [4096, 8192, 16384, 24576]),
    5: ([8, 16, 32, 64, 128], [4096, 8192, 12288, 16384, 24576]),
    6: ([8, 16, 32, 64, 128, 16], [4096, 8192, 12288, 16384, 24576, 8192]),
    7: ([8, 16, 32, 64, 128, 16, 32], [4096, 8192, 12288, 16384, 24576, 8192, 16384]),
    8: ([8, 16, 32, 64, 128, 16, 32, 64], [4096, 8192, 12288, 16384, 24576, 8192, 16384, 24576]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=17)
    parser.add_argument("--calls", type=int, default=3)
    return parser.parse_args()


def elapsed_us(function: Callable[[], None], calls: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(calls):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / calls


def make_case(batch: int) -> dict[str, torch.Tensor | list[int]]:
    q_lens, contexts = CASES[batch]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9365100 + batch)
    total_q = sum(q_lens)
    query = 0.02 * torch.randn(
        (total_q, 24, 256), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    starts = [0]
    for length in q_lens:
        starts.append(starts[-1] + length)
    query_starts = torch.tensor(starts, device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor(
        [context + q_len for context, q_len in zip(contexts, q_lens, strict=True)],
        device="cuda",
        dtype=torch.int32,
    )

    logical_pages = [math.ceil((context + q_len) / 784) + 1 for context, q_len in zip(contexts, q_lens, strict=True)]
    table = torch.zeros((batch, max(logical_pages)), device="cuda", dtype=torch.int32)
    physical = 0
    for sequence, pages in enumerate(logical_pages):
        table[sequence, :pages] = torch.arange(
            physical, physical + pages, device="cuda", dtype=torch.int32
        )
        physical += pages
    key = 0.02 * torch.randn(
        (physical, 784, 4, 256), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    value = 0.02 * torch.randn(
        (physical, 784, 4, 256), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    return {
        "q_lens": q_lens,
        "contexts": contexts,
        "query": query,
        "query_starts": query_starts,
        "seq_lens": seq_lens,
        "table": table,
        "key": key,
        "value": value,
    }


def launch(data: dict, output: torch.Tensor, block_m: int) -> None:
    query = data["query"]
    table = data["table"]
    key = data["key"]
    value = data["value"]
    batch = len(data["q_lens"])
    strides = (
        table.stride(0),
        query.stride(0),
        output.stride(0),
        *key.stride()[:3],
        *value.stride()[:3],
    )
    _gqa6[(triton.cdiv(max(data["q_lens"]), block_m // 2), 12, batch)](
        output,
        query,
        key,
        value,
        table,
        data["seq_lens"],
        data["query_starts"],
        256**-0.5,
        CACHE_SIZE=784,
        BLOCK_M=block_m,
        STRIDES=strides,
        num_warps=2,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )


def benchmark(batch: int, groups: int, calls: int) -> dict:
    data = make_case(batch)
    fallback_block_m = 64 if batch == 1 and data["query"].shape[0] >= 128 else 16
    outputs = {
        "current_fallback": torch.empty_like(data["query"]),
        "candidate_bm32": torch.empty_like(data["query"]),
    }
    functions = {
        "current_fallback": lambda: launch(data, outputs["current_fallback"], fallback_block_m),
        "candidate_bm32": lambda: launch(data, outputs["candidate_bm32"], 32),
    }
    for _ in range(8):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    functions["current_fallback"]()
    functions["candidate_bm32"]()
    torch.cuda.synchronize()
    delta = outputs["candidate_bm32"].float() - outputs["current_fallback"].float()
    correctness = {
        "finite": bool(torch.isfinite(outputs["candidate_bm32"]).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "allclose_atol_0.02_rtol_0.02": bool(
            torch.allclose(
                outputs["candidate_bm32"], outputs["current_fallback"],
                atol=0.02, rtol=0.02,
            )
        ),
    }
    if not correctness["finite"] or not correctness["allclose_atol_0.02_rtol_0.02"]:
        raise AssertionError(correctness)
    samples = {name: [] for name in functions}
    rng = random.Random(9365200 + batch)
    for _ in range(groups):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            samples[name].append(elapsed_us(functions[name], calls))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    gain = 100.0 * (medians["current_fallback"] - medians["candidate_bm32"]) / medians["current_fallback"]
    result = {
        "local_batch": batch,
        "current_fallback_block_m": fallback_block_m,
        "q_lens": data["q_lens"],
        "contexts": data["contexts"],
        "total_query_tokens": int(data["query"].shape[0]),
        "cache_strides": list(data["key"].stride()),
        "correctness": correctness,
        "samples_us": samples,
        "medians_us": medians,
        "bm32_time_reduction_percent": gain,
    }
    print(json.dumps({"B": batch, "medians_us": medians, "gain_percent": gain}), flush=True)
    del data, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    prop = torch.cuda.get_device_properties(0)
    if "gfx936" not in prop.gcnArchName:
        raise RuntimeError(f"gfx936 required, got {prop.gcnArchName}")
    result = {
        "schema": "gqa-page784-local-batch-bm32-screen-v1",
        "device": prop.gcnArchName,
        "groups": args.groups,
        "calls": args.calls,
        "records": [benchmark(batch, args.groups, args.calls) for batch in CASES],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
