#!/usr/bin/env python3
"""Run a q_len sweep before widening the page784 BM32 local-batch gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=9)
    parser.add_argument("--calls", type=int, default=2)
    return parser.parse_args()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("gqa_local_batch_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    base = load_base(args.base_script.resolve())
    contexts = {
        3: [4096, 16384, 24576],
        5: [4096, 8192, 12288, 16384, 24576],
    }
    sweeps = {
        3: ([8] * 3, [16] * 3, [64] * 3, [256] * 3, [1024] * 3),
        5: (
            [8] * 5,
            [16] * 5,
            [64] * 5,
            [256] * 5,
            [1024, 768, 768, 768, 768],
        ),
    }
    records = []
    for batch, q_len_cases in sweeps.items():
        for q_lens in q_len_cases:
            base.CASES[batch] = (list(q_lens), contexts[batch])
            record = base.benchmark(batch, args.groups, args.calls)
            record["case_name"] = f"b{batch}_q" + "_".join(map(str, q_lens))
            records.append(record)
    result = {
        "schema": "gqa-page784-b3-b5-qlen-sweep-v1",
        "groups": args.groups,
        "calls": args.calls,
        "records": records,
        "all_positive": all(
            record["bm32_time_reduction_percent"] > 0 for record in records
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
