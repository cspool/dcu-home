#!/usr/bin/env python3
"""Extract GPU-free evidence from frozen all3 and profiler artifacts."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
from pathlib import Path


DEFAULT_ALL3 = Path(
    "/public/home/tangyu408/testdata/goal_runs/"
    "20260712_h11_5_h10_8_all3_10m/results"
)
DEFAULT_TRACE = Path(
    "/public/home/tangyu408/testdata/profile_runs/torch_profile_20260707_codex/"
    "contexts/16-32K/raw_traces/"
    "rank0.1783405413530600899.pt.trace.json.gz"
)
DEFAULT_PMC = Path(
    "/public/home/tangyu408/testdata/goal_runs/"
    "20260712_prefill_tensor_unit_probe/rocprof_mmop.csv"
)
DEFAULT_BACKENDS = Path(
    "/public/home/tangyu408/testdata/goal_runs/"
    "20260712_prefill_gemm_backend_probe/torch_full_results.json"
)


def extract_all3(result_root: Path) -> dict:
    rows = []
    for path in sorted(result_root.glob("*/result.json")):
        payload = json.loads(path.read_text())
        rows.append({"context": path.parent.name, "input_lens": payload["input_lens"]})
    input_lens = [length for row in rows for length in row["input_lens"]]
    chunks: list[int] = []
    for length in input_lens:
        chunks.extend([4096] * (length // 4096))
        if length % 4096:
            chunks.append(length % 4096)
    counts = collections.Counter(chunks)
    return {
        "contexts": rows,
        "input_lens": input_lens,
        "total_input_tokens": sum(input_lens),
        "chunk_count": len(chunks),
        "chunk_histogram": [
            {"tokens": tokens, "count": count}
            for tokens, count in sorted(counts.items(), reverse=True)
        ],
        "full_chunk_token_fraction": (
            counts[4096] * 4096 / sum(input_lens) if input_lens else 0.0
        ),
    }


def extract_trace(trace_path: Path) -> dict:
    with gzip.open(trace_path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernel_by_external_id = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and "kernel" in event.get("args", {})
    }
    contexts = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("execute_context_")
        and "_generation_0" in event.get("name", "")
    ]
    linked = []
    for event in events:
        if event.get("cat") != "cpu_op" or event.get("name") != "aten::mm":
            continue
        external_id = event.get("args", {}).get("External id")
        launch = kernel_by_external_id.get(external_id)
        if launch is None:
            continue
        context = next(
            (
                item
                for item in contexts
                if item["ts"] <= event["ts"] <= item["ts"] + item["dur"]
            ),
            None,
        )
        if context is None or "(4096)" not in context["name"]:
            continue
        args = launch["args"]
        kernel = args["kernel"]
        macro_tile = re.search(r"_MT(\d+)x(\d+)x(\d+)", kernel)
        wgm = re.search(r"_WGM(\d+)", kernel)
        grid = args["grid"]
        block = args["block"]
        workgroups = (grid[0] * grid[1] * grid[2]) // (
            block[0] * block[1] * block[2]
        )
        linked.append(
            (
                tuple(map(int, macro_tile.groups())) if macro_tile else None,
                int(wgm.group(1)) if wgm else None,
                workgroups,
                tuple(grid),
                tuple(block),
                kernel,
            )
        )
    grouped = collections.Counter(linked)
    source_by_workgroups = {
        320: "all output/down projections: 64 MLP down + 48 GDN out + 16 attention out per five chunks",
        2176: "MLP gate_up: 64 layers per chunk",
        64: "GDN in_proj_ba: 48 layers per chunk",
        1024: "GDN in_proj_qkvz: 48 layers per chunk",
        896: "full-attention qkv: 16 layers per chunk",
        62080: "LM-head/logits path; outside the six transformer-core projection families",
    }
    return {
        "linked_prefill_mm_count": len(linked),
        "kernel_groups": [
            {
                "calls": calls,
                "macro_tile": macro_tile,
                "wgm": wgm,
                "workgroups": workgroups,
                "grid": grid,
                "block": block,
                "kernel": kernel,
                "uses_mmac_by_name": "MAC_MMAC" in kernel,
                "inferred_source": source_by_workgroups.get(workgroups, "unknown"),
            }
            for (
                macro_tile,
                wgm,
                workgroups,
                grid,
                block,
                kernel,
            ), calls in sorted(grouped.items(), key=lambda item: (-item[1], item[0][2]))
        ],
    }


def extract_pmc(path: Path) -> dict:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    keys = [
        "KernelName",
        "SQ_INSTS_MMOP",
        "SQ_ACTIVE_INST_MMOP",
        "SQ_INSTS_VALU",
        "SQ_ACTIVE_INST_VALU",
        "SQ_WAVES",
    ]
    result = {key: int(row[key]) if key != "KernelName" else row[key] for key in keys}
    result["raw_mmop_per_valu"] = result["SQ_INSTS_MMOP"] / result["SQ_INSTS_VALU"]
    result["raw_active_mmop_per_active_valu"] = (
        result["SQ_ACTIVE_INST_MMOP"] / result["SQ_ACTIVE_INST_VALU"]
    )
    result["occupancy_percentage_valid"] = False
    return result


def extract_existing_backend_diagnostic(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for result in payload["results"]:
        case = result["case"]
        if case["n"] != 4096:
            continue
        rocblas = result["backends"]["torch_rocblas"]["timing"]
        hipblaslt = result["backends"]["torch_hipblaslt"]["timing"]
        rows.append(
            {
                "name": case["name"],
                "tokens": case["n"],
                "out_features": case["m"],
                "in_features": case["k"],
                "layers": case["layers"],
                "rocblas_median_ms": rocblas["median_ms"],
                "rocblas_median_tflops": rocblas["median_tflops"],
                "hipblaslt_median_ms": hipblaslt["median_ms"],
                "diagnostic_only_under_600_seconds": True,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all3", type=Path, default=DEFAULT_ALL3)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--pmc", type=Path, default=DEFAULT_PMC)
    parser.add_argument("--backends", type=Path, default=DEFAULT_BACKENDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "gpu_initialized": False,
        "all3": extract_all3(args.all3),
        "phase_labelled_trace": extract_trace(args.trace),
        "existing_one_dispatch_pmc": extract_pmc(args.pmc),
        "existing_backend_diagnostic": extract_existing_backend_diagnostic(
            args.backends
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
