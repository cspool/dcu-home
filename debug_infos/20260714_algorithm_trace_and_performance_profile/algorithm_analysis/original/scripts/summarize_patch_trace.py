#!/usr/bin/env python3
"""Summarize vLLM patch trace JSONL files for the three context runs."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED = {
    "4-8K": {"prefill_chunks": 2, "decode_steps": 88},
    "8-16K": {"prefill_chunks": 4, "decode_steps": 92},
    "16-32K": {"prefill_chunks": 6, "decode_steps": 23},
}

JOIN_KEY_REQUIREMENTS = {
    "scheduler_step": ("engine_step_id",),
    "model_execute_begin": ("engine_step_id", "forward_id"),
    "model_execute_end": ("engine_step_id", "forward_id"),
    "batch_constructed": ("engine_step_id", "forward_id", "req_ids"),
    "attention_batch_metadata": ("engine_step_id", "forward_id"),
    "attention_forward_begin": ("engine_step_id", "forward_id", "layer_name"),
    "attention_forward_end": ("engine_step_id", "forward_id", "layer_name"),
    "kv_allocate_slots": ("engine_step_id", "request_id"),
    "kv_get_computed_blocks": ("engine_step_id", "request_id"),
    "sampler_call": ("engine_step_id", "forward_id"),
    "sample_tokens": ("engine_step_id", "forward_id"),
    "scheduler_update_output": ("engine_step_id",),
}


def read_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("events.*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    events.sort(key=lambda e: (e.get("ts_ns", 0), e.get("pid", 0), e.get("event_index", 0)))
    return events


def first_req_tokens(event: dict[str, Any]) -> int | None:
    tokens = event.get("num_scheduled_tokens") or {}
    if not tokens:
        return None
    try:
        return int(next(iter(tokens.values())))
    except Exception:
        return None


def load_result(context_dir: Path) -> dict[str, Any]:
    result_path = context_dir / "result.json"
    if not result_path.exists():
        return {}
    with result_path.open(encoding="utf-8") as f:
        result = json.load(f)
    return {
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "total_input_tokens": result.get("total_input_tokens"),
        "total_output_tokens": result.get("total_output_tokens"),
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "mean_tpot_ms": result.get("mean_tpot_ms"),
        "mean_e2el_ms": result.get("mean_e2el_ms"),
    }


def missing_join_keys(events: list[dict[str, Any]]) -> dict[str, int]:
    missing: Counter[str] = Counter()
    for event in events:
        event_type = event.get("event_type")
        required = JOIN_KEY_REQUIREMENTS.get(event_type)
        if not required:
            continue
        for key in required:
            value = event.get(key)
            if value is None or value == []:
                missing[f"{event_type}.{key}"] += 1
        if event_type == "scheduler_step" and event.get("phase") != "empty":
            reqs = event.get("requests") or []
            if not reqs or any(not req.get("request_id") for req in reqs):
                missing["scheduler_step.requests.request_id"] += 1
        if event_type == "scheduler_update_output" and event.get("output_count", 0) > 0:
            outputs = event.get("outputs") or []
            if any(not out.get("request_id") for out in outputs):
                missing["scheduler_update_output.outputs.request_id"] += 1
    return dict(sorted(missing.items()))


def summarize_context(label: str, context_dir: Path) -> dict[str, Any]:
    trace_dir = context_dir / "patch_trace"
    events = read_events(trace_dir)
    bench_result = load_result(context_dir)
    counts = Counter(e.get("event_type") for e in events)
    scheduler_steps = [e for e in events if e.get("event_type") == "scheduler_step"]
    phases = Counter(e.get("phase") for e in scheduler_steps)
    prefill_steps = [
        e
        for e in scheduler_steps
        if e.get("phase") in {"prefill_chunk", "mixed"}
        and (first_req_tokens(e) or 0) > 1
    ]
    decode_steps = [
        e
        for e in scheduler_steps
        if e.get("phase") in {"decode", "decode_or_spec"}
        and (first_req_tokens(e) or 0) == 1
    ]
    update_events = [e for e in events if e.get("event_type") == "scheduler_update_output"]
    finished = []
    output_token_total_by_req: defaultdict[str, int] = defaultdict(int)
    for event in update_events:
        for out in event.get("outputs") or []:
            req_id = out.get("request_id")
            if req_id:
                output_token_total_by_req[req_id] += int(out.get("new_token_count") or 0)
            if out.get("finished"):
                finished.append(out)
    errors = [e for e in events if e.get("event_type") == "patch_error"]
    required_event_types = {
        "scheduler_step",
        "model_execute_begin",
        "batch_constructed",
        "attention_forward_begin",
        "kv_allocate_slots",
        "scheduler_update_output",
    }
    missing = sorted(t for t in required_event_types if counts.get(t, 0) == 0)
    expected = EXPECTED.get(label, {})
    validation = {
        "has_events": bool(events),
        "missing_required_event_types": missing,
        "missing_join_keys": missing_join_keys(events),
        "patch_errors": len(errors),
        "benchmark_completed": bench_result.get("completed") == 1,
        "benchmark_failed_zero": bench_result.get("failed") == 0,
        "prefill_chunks_match_expected": (
            expected.get("prefill_chunks") is None
            or len(prefill_steps) == expected["prefill_chunks"]
        ),
        "decode_steps_match_expected": (
            expected.get("decode_steps") is None
            or len(decode_steps) == expected["decode_steps"]
        ),
        "has_finished_output": bool(finished),
    }
    return {
        "label": label,
        "event_count": len(events),
        "event_type_counts": dict(sorted(counts.items())),
        "phase_counts": dict(sorted(phases.items())),
        "prefill_chunk_count": len(prefill_steps),
        "prefill_chunk_tokens": [first_req_tokens(e) for e in prefill_steps],
        "decode_step_count": len(decode_steps),
        "benchmark_result": bench_result,
        "finished_outputs": finished,
        "output_token_total_by_req": dict(output_token_total_by_req),
        "validation": validation,
        "trace_files": [str(p) for p in sorted(trace_dir.glob("events.*.jsonl"))],
    }


def write_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    (run_dir / "patch_trace_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Patch Trace Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "| Context | Events | Prefill chunks | Decode steps | Finished outputs | Patch errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {label} | {events} | {prefill} {prefill_tokens} | {decode} | {finished} | {errors} |".format(
                label=summary["label"],
                events=summary["event_count"],
                prefill=summary["prefill_chunk_count"],
                prefill_tokens=summary["prefill_chunk_tokens"],
                decode=summary["decode_step_count"],
                finished=len(summary["finished_outputs"]),
                errors=summary["validation"]["patch_errors"],
            )
        )
    lines.extend(["", "## Validation", ""])
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        lines.append(f"- `benchmark_result`: `{summary['benchmark_result']}`")
        for key, value in summary["validation"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append(f"- `event_type_counts`: `{summary['event_type_counts']}`")
        lines.append(f"- `phase_counts`: `{summary['phase_counts']}`")
        lines.append(
            f"- `output_token_total_by_req`: `{summary['output_token_total_by_req']}`"
        )
        if summary["finished_outputs"]:
            lines.append(f"- `finished_outputs`: `{summary['finished_outputs']}`")
        lines.append("")
    (run_dir / "patch_trace_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_patch_trace.py RUN_DIR")
    run_dir = Path(sys.argv[1])
    contexts_dir = run_dir / "contexts"
    labels = sorted(
        p.name for p in contexts_dir.iterdir() if p.is_dir()
    ) if contexts_dir.exists() else []
    if not labels:
        labels = ["4-8K", "8-16K", "16-32K"]
    summaries = []
    for label in labels:
        summaries.append(summarize_context(label, run_dir / "contexts" / label))
    write_summary(run_dir, summaries)
    failed = [
        (s["label"], key)
        for s in summaries
        for key, value in s["validation"].items()
        if value is False or (isinstance(value, list) and value)
    ]
    if failed:
        raise SystemExit(f"patch trace validation failed: {failed}")


if __name__ == "__main__":
    main()
