#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FIELDS = [
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "total_token_throughput",
    "output_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {"raw_type": type(data).__name__}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result_files = sorted(RESULTS.glob("*/*_throughput/result.json"))
    if not result_files:
        print(f"No result.json files found under {RESULTS}")
        print("This script only summarizes existing benchmark outputs.")
        return 0

    print("| case | context | " + " | ".join(FIELDS) + " |")
    print("| --- | --- | " + " | ".join(["---:"] * len(FIELDS)) + " |")
    for path in result_files:
        case_id = path.relative_to(RESULTS).parts[0]
        context = path.parent.name.removesuffix("_throughput")
        row = load_json(path)
        values = [fmt(row.get(field)) for field in FIELDS]
        print(f"| {case_id} | {context} | " + " | ".join(values) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

