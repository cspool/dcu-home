#!/usr/bin/env python3
"""Summarize and validate selected-layer FX trace artifacts."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_EVENT_FILES = [
    "fx_graph.py",
    "fx_graph.txt",
    "fx_nodes.json",
    "fx_graph_module.pt",
    "fx_graph_module",
    "fx_trace_metadata.json",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_trace_dir(trace_dir: str, context_dir: Path) -> Path:
    path = Path(trace_dir)
    if path.is_absolute() or path.exists():
        return path
    for parent in (context_dir, *context_dir.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def validate_event_dir(trace_dir: str, context_dir: Path) -> list[str]:
    if not trace_dir:
        return ["missing trace_dir"]
    root = resolve_trace_dir(trace_dir, context_dir)
    missing: list[str] = []
    for name in EXPECTED_EVENT_FILES:
        path = root / name
        if not path.exists():
            missing.append(str(path))
    return missing


def _sorted_event_ids(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row.get("event_id", "") for row in rows if row.get("event_id")})


def _captured_by_layer(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        layer_id = row.get("layer_id")
        if layer_id:
            counts[layer_id] += 1
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
        )
    )


def derive_run_metadata(
    *,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    layer_rows: list[dict[str, str]],
) -> dict[str, Any]:
    repaired = dict(metadata)
    ok_rows = [row for row in manifest_rows if row.get("status") == "ok"]
    error_rows = [row for row in manifest_rows if row.get("status") == "error"]
    event_ids = _sorted_event_ids(manifest_rows)

    if event_ids:
        repaired["captured_events"] = event_ids
        repaired["captured_by_layer"] = _captured_by_layer(manifest_rows)
        repaired["fx_sample_count"] = len(manifest_rows)
        repaired["fx_trace_count"] = len(ok_rows)
        repaired["fx_trace_error_count"] = len(error_rows)
    repaired["observed_layer_event_count"] = len(layer_rows)
    repaired["metadata_aggregation"] = {
        "source": "fx_layer_trace_manifest.csv and fx_layer_events.csv",
        "reason": "vLLM worker processes can exit after the tracing worker and otherwise overwrite run metadata.",
        "updated_by": "summarize_selected_layer_fx_trace.py",
    }
    return repaired


def write_run_metadata_if_changed(path: Path, metadata: dict[str, Any]) -> None:
    existing = read_json(path)
    if existing == metadata:
        return
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_context(context_dir: Path) -> dict[str, Any]:
    label = context_dir.name
    fx_dir = context_dir / "fx_trace"
    metadata_path = fx_dir / "run_metadata.json"
    manifest_path = fx_dir / "fx_layer_trace_manifest.csv"
    layer_events_path = fx_dir / "fx_layer_events.csv"
    result_path = context_dir / "result.json"
    metadata = read_json(metadata_path)
    manifest_rows = read_csv(manifest_path)
    layer_rows = read_csv(layer_events_path)
    result = read_json(result_path)
    if metadata:
        metadata = derive_run_metadata(
            metadata=metadata,
            manifest_rows=manifest_rows,
            layer_rows=layer_rows,
        )
        write_run_metadata_if_changed(metadata_path, metadata)

    ok_rows = [row for row in manifest_rows if row.get("status") == "ok"]
    error_rows = [row for row in manifest_rows if row.get("status") == "error"]
    missing_artifacts: dict[str, list[str]] = {}
    node_counts: dict[str, int] = {}
    for row in ok_rows:
        event_id = row.get("event_id") or "<missing-event-id>"
        missing = validate_event_dir(row.get("trace_dir", ""), context_dir)
        if missing:
            missing_artifacts[event_id] = missing
        try:
            node_counts[event_id] = int(row.get("node_count") or 0)
        except ValueError:
            node_counts[event_id] = 0

    return {
        "label": label,
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "layer_events_path": str(layer_events_path),
        "result_path": str(result_path),
        "metadata_present": bool(metadata),
        "target_event_keys": metadata.get("target_event_keys", []),
        "captured_events": metadata.get("captured_events", []),
        "observed_layer_event_count": metadata.get("observed_layer_event_count", len(layer_rows)),
        "fx_sample_count": metadata.get("fx_sample_count"),
        "fx_trace_count": metadata.get("fx_trace_count"),
        "fx_trace_error_count": metadata.get("fx_trace_error_count"),
        "manifest_row_count": len(manifest_rows),
        "ok_row_count": len(ok_rows),
        "error_row_count": len(error_rows),
        "node_counts": node_counts,
        "missing_artifacts": missing_artifacts,
        "benchmark": {
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "total_input_tokens": result.get("total_input_tokens"),
            "total_output_tokens": result.get("total_output_tokens"),
            "mean_ttft_ms": result.get("mean_ttft_ms"),
            "mean_tpot_ms": result.get("mean_tpot_ms"),
            "mean_e2el_ms": result.get("mean_e2el_ms"),
        },
        "validation": {
            "metadata_present": bool(metadata),
            "has_manifest_rows": bool(manifest_rows),
            "all_manifest_rows_ok": bool(manifest_rows) and not error_rows,
            "all_ok_rows_have_artifacts": not missing_artifacts,
            "benchmark_completed": result.get("completed") == 1,
            "benchmark_failed_zero": result.get("failed") == 0,
        },
        "errors": [
            {
                "event_id": row.get("event_id"),
                "layer_id": row.get("layer_id"),
                "forward_id": row.get("forward_id"),
                "error": row.get("error"),
            }
            for row in error_rows
        ],
    }


def write_markdown(run_dir: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Selected-Layer FX Trace Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "| Context | Targets | Captured | FX OK | FX Errors | Missing artifacts | Benchmark |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in summary:
        benchmark = item["benchmark"]
        bench_text = (
            f"completed={benchmark.get('completed')}, failed={benchmark.get('failed')}, "
            f"input={benchmark.get('total_input_tokens')}, output={benchmark.get('total_output_tokens')}"
        )
        lines.append(
            "| {label} | `{targets}` | `{captured}` | {ok} | {err} | {missing} | {bench} |".format(
                label=item["label"],
                targets=",".join(item.get("target_event_keys") or []),
                captured=",".join(item.get("captured_events") or []),
                ok=item["ok_row_count"],
                err=item["error_row_count"],
                missing=sum(len(value) for value in item["missing_artifacts"].values()),
                bench=bench_text,
            )
        )

    lines.extend(["", "## Per-Context Details", ""])
    for item in summary:
        lines.append(f"### {item['label']}")
        lines.append(f"- metadata: `{item['metadata_path']}`")
        lines.append(f"- manifest: `{item['manifest_path']}`")
        lines.append(f"- observed layer events: `{item['observed_layer_event_count']}`")
        lines.append(f"- fx samples/traces/errors: `{item['fx_sample_count']}` / `{item['fx_trace_count']}` / `{item['fx_trace_error_count']}`")
        if item["node_counts"]:
            node_text = ", ".join(f"{key}={value}" for key, value in sorted(item["node_counts"].items()))
            lines.append(f"- node counts: `{node_text}`")
        if item["errors"]:
            lines.append("- errors:")
            for error in item["errors"]:
                lines.append(
                    f"  - `{error.get('event_id')}` layer={error.get('layer_id')} "
                    f"forward={error.get('forward_id')}: `{error.get('error')}`"
                )
        if item["missing_artifacts"]:
            lines.append("- missing artifacts:")
            for event_id, missing in sorted(item["missing_artifacts"].items()):
                lines.append(f"  - `{event_id}`: `{len(missing)}` missing")
        lines.append("")

    (run_dir / "selected_layer_fx_trace_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_selected_layer_fx_trace.py RUN_DIR")
    run_dir = Path(sys.argv[1])
    contexts_dir = run_dir / "contexts"
    if not contexts_dir.exists():
        raise SystemExit(f"contexts directory not found: {contexts_dir}")
    summary = [
        summarize_context(path)
        for path in sorted(contexts_dir.iterdir())
        if path.is_dir()
    ]
    (run_dir / "selected_layer_fx_trace_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(run_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
