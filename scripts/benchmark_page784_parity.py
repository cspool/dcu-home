#!/usr/bin/env python3
"""Benchmark modular page784 only against immutable 3k history.

The script never imports or launches the 3k tree.  It recreates the historical
random inputs and reads the recorded 3k medians from the experiment JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py"
)
SPEC = importlib.util.spec_from_file_location("modular_gqa6", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
page784 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(page784)
PAGE_SIZE = 784
HISTORY = (
    REPO_ROOT.parent
    / "experiments/decode_bank_dcu0/page784_combined_residual_r1/results.json"
)


@dataclass
class Case:
    name: str
    query_len: int
    context_len: int
    query: torch.Tensor
    current_k: torch.Tensor
    current_v: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    block_table: torch.Tensor
    cu_q: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY,
        help="3k historical results JSON (default: %(default)s)",
    )
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--case", action="append")
    return parser.parse_args()


def check_environment() -> None:
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name) != "0":
            raise RuntimeError(f"{name}=0 is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one masked DCU is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx936" not in arch:
        raise RuntimeError(f"gfx936 is required, got {arch}")


def make_case(name: str, query_len: int, context_len: int, seed: int) -> Case:
    torch.manual_seed(seed)
    total_len = query_len + context_len
    logical_pages = (total_len + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = logical_pages + 3
    query = torch.randn(
        (query_len, 24, 256), device="cuda", dtype=torch.bfloat16
    )
    current_k = torch.randn(
        (query_len, 4, 256), device="cuda", dtype=torch.bfloat16
    )
    current_v = torch.randn_like(current_k)
    cache_k = torch.randn(
        (physical_pages, PAGE_SIZE, 4, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache_v = torch.randn_like(cache_k)
    pages = torch.randperm(physical_pages, device="cuda")[:logical_pages]
    return Case(
        name=name,
        query_len=query_len,
        context_len=context_len,
        query=query,
        current_k=current_k,
        current_v=current_v,
        cache_k=cache_k,
        cache_v=cache_v,
        block_table=pages.to(torch.int32).view(1, logical_pages),
        cu_q=torch.tensor([0, query_len], device="cuda", dtype=torch.int32),
        output=torch.empty_like(query),
    )


def launch(case: Case) -> None:
    meta = SimpleNamespace(
        max_query_len=case.query_len,
        max_seq_len=case.query_len + case.context_len,
        query_start_loc=case.cu_q,
        block_table=case.block_table,
        num_actual_tokens=case.query_len,
    )
    assert page784.page784_prefill(
        case.query, case.current_k, case.current_v, case.cache_k, case.cache_v,
        case.output, meta, 256**-0.5,
    )


def elapsed_us(case: Case, calls: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(calls):
        launch(case)
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / calls


def main() -> int:
    args = parse_args()
    check_environment()
    definitions = (
        ("4-8K_q3437_ctx4096", 3437, 4096),
        ("8-16K_q4096_ctx4096", 4096, 4096),
        ("8-16K_q4096_ctx8192", 4096, 8192),
        ("8-16K_q2231_ctx12288", 2231, 12288),
    )
    if args.case:
        requested = set(args.case)
        definitions = tuple(item for item in definitions if item[0] in requested)
        missing = requested - {item[0] for item in definitions}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")

    history = json.loads(args.history.read_text())
    historical_cases = {case["name"]: case for case in history["cases"]}
    result = {
        "schema": "modular-page784-vs-immutable-3k-history-v1",
        "historical_source": str(args.history),
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "triton": triton.__version__,
        "groups": args.groups,
        "calls": args.calls,
        "cases": [],
    }
    for index, (name, query_len, context_len) in enumerate(definitions):
        case = make_case(name, query_len, context_len, 9362700 + index)
        launch(case)
        torch.cuda.synchronize()
        first = case.output.clone()
        launch(case)
        torch.cuda.synchronize()
        repeat_bitwise = bool(torch.equal(first, case.output))
        rng = random.Random(9362800 + index)
        samples: list[float] = []
        for group in range(args.groups):
            samples.append(elapsed_us(case, args.calls))
            rng.random()
            print(f"case={name} group={group + 1}/{args.groups}", flush=True)
        median = statistics.median(samples)
        baseline = historical_cases[name]["medians_us"]["baseline"]
        record = {
            "name": name,
            "query_len": query_len,
            "context_len": context_len,
            "samples_us": samples,
            "modular_median_us": median,
            "historical_3k_median_us": baseline,
            "modular_delta_percent": (median / baseline - 1.0) * 100.0,
            "finite": bool(torch.isfinite(case.output).all().item()),
            "repeat_bitwise": repeat_bitwise,
        }
        result["cases"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del case
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
