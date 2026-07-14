#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT.parent

CHECKED_FILES = [
    ROOT / "env.example",
    ROOT / ".env",
    ROOT / "benchmark_matrix.tsv",
    ROOT / "scripts" / "render_commands.sh",
    TESTDATA / "start_vllm.sh",
    TESTDATA / "run_throughput.sh",
]

FORBIDDEN = [
    re.compile(r"--speculative", re.IGNORECASE),
    re.compile(r"num[-_]?speculative", re.IGNORECASE),
    re.compile(r"speculative[-_]?model", re.IGNORECASE),
    re.compile(r"speculative[-_]?method", re.IGNORECASE),
    re.compile(r"draft[-_]?model", re.IGNORECASE),
    re.compile(r"\bEAGLE\b", re.IGNORECASE),
]


def main() -> int:
    violations: list[str] = []
    for path in CHECKED_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in FORBIDDEN:
                if pattern.search(line):
                    violations.append(f"{path}:{line_no}: {line.strip()}")

    if violations:
        print("forbidden speculative decoding config found:", file=sys.stderr)
        for item in violations:
            print(f"  {item}", file=sys.stderr)
        return 1

    print("OK: no speculative decoding config in benchmark environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

