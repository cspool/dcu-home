#!/usr/bin/env python3
"""Export normalized Workflow05 intervals to Perfetto-supported Chrome JSON.

This is a format adapter, not a trace viewer or an analysis engine.  It keeps
source rows and evidence semantics auditable so the resulting trace can be
opened by the upstream Perfetto UI and queried by Trace Processor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EVIDENCE_CLASSES = {
    "observed",
    "inferred",
    "estimate",
    "replay_projected",
    "heuristic",
    "unavailable",
}


class ExportError(RuntimeError):
    """Fail-closed normalized-table to Perfetto format error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExportError(f"{path} must contain one JSON object")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ExportError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{label} must be a nonempty string")
    return value.strip()


def string_list(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ExportError(f"{label} must be a list of nonempty strings")
    if nonempty and not value:
        raise ExportError(f"{label} must not be empty")
    if len(value) != len(set(value)):
        raise ExportError(f"{label} contains duplicates")
    return value


def number(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ExportError(f"invalid numeric {label}: {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ExportError(f"non-finite numeric {label}: {value!r}")
    return result


def to_ns(value: str, unit: str, label: str) -> int:
    scale = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(unit)
    if scale is None:
        raise ExportError(f"unsupported timestamp unit: {unit}")
    result = number(value, label) * scale
    rounded = int(round(result))
    if abs(result - rounded) > 1e-6:
        raise ExportError(f"timestamp cannot be represented as integer ns: {label}")
    return rounded


def chrome_us(value_ns: int) -> int | float:
    if value_ns % 1_000 == 0:
        return value_ns // 1_000
    return round(value_ns / 1_000.0, 3)


def source_path(raw: str, spec_path: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise ExportError(f"track source is missing: {resolved}")
    return resolved


def row_value(row: dict[str, str], field: str, label: str) -> str:
    if field not in row:
        raise ExportError(f"missing column {field!r} in {label}")
    return row[field]


def allocate_sublanes(rows: list[dict[str, Any]]) -> None:
    """Give overlapping complete events distinct Chrome trace thread lanes."""
    by_lane: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lane[row["base_lane"]].append(row)
    for lane in sorted(by_lane):
        lane_ends: list[int] = []
        for row in sorted(
            by_lane[lane],
            key=lambda item: (
                item["begin_ns"],
                item["end_ns"],
                item["row_key"],
            ),
        ):
            slot = next(
                (
                    index
                    for index, end_ns in enumerate(lane_ends)
                    if end_ns <= row["begin_ns"]
                ),
                len(lane_ends),
            )
            if slot == len(lane_ends):
                lane_ends.append(row["end_ns"])
            else:
                lane_ends[slot] = row["end_ns"]
            row["sublane"] = slot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt exact normalized Workflow05 interval CSVs to the Chrome "
            "JSON interface supported by Perfetto UI and Trace Processor."
        )
    )
    parser.add_argument("--track-spec", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    spec_path = args.track_spec.resolve()
    output_trace = args.output_trace.resolve()
    output_manifest = args.output_manifest.resolve()
    if not spec_path.is_file():
        raise ExportError(f"track specification is missing: {spec_path}")
    if output_trace.exists() or output_manifest.exists():
        raise ExportError("refusing to overwrite trace or manifest output")
    if output_trace.parent != output_manifest.parent:
        raise ExportError("trace and manifest must share one output directory")
    output_trace.parent.mkdir(parents=True, exist_ok=True)

    spec = load_object(spec_path)
    if spec.get("schema_version") != 1:
        raise ExportError("track specification schema_version must equal 1")
    clock = spec.get("clock")
    if not isinstance(clock, dict):
        raise ExportError("track specification requires a clock object")
    clock_domain = required_text(clock.get("domain"), "clock.domain")
    if clock.get("parent_clock_mergeable") is not False:
        raise ExportError("Workflow05 parent and child clocks must not merge")
    tracks = spec.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ExportError("track specification requires a nonempty tracks list")

    all_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ExportError(f"tracks[{track_index}] must be an object")
        group = required_text(track.get("track_group"), "track_group")
        if group in seen_groups:
            raise ExportError(f"duplicate track_group: {group}")
        seen_groups.add(group)
        category = required_text(track.get("category"), f"{group}.category")
        path = source_path(
            required_text(track.get("source_csv"), f"{group}.source_csv"),
            spec_path,
        )
        expected_sha = track.get("source_sha256")
        observed_sha = sha256_file(path)
        if expected_sha is not None and expected_sha != observed_sha:
            raise ExportError(f"source SHA-256 mismatch: {path}")
        fields, rows = read_csv(path)
        required = bool(track.get("required", True))
        if required and not rows:
            raise ExportError(f"required track source is empty: {path}")
        name_columns = string_list(
            track.get("name_columns"), f"{group}.name_columns"
        )
        lane_columns = string_list(
            track.get("lane_columns"), f"{group}.lane_columns"
        )
        row_key_columns = string_list(
            track.get("row_key_columns"), f"{group}.row_key_columns"
        )
        args_columns = string_list(
            track.get("args_columns", []),
            f"{group}.args_columns",
            nonempty=False,
        )
        begin_column = required_text(
            track.get("begin_column"), f"{group}.begin_column"
        )
        end_column = str(track.get("end_column", "")).strip()
        duration_column = str(track.get("duration_column", "")).strip()
        if bool(end_column) == bool(duration_column):
            raise ExportError(
                f"{group} requires exactly one end_column or duration_column"
            )
        unit = required_text(track.get("timestamp_unit"), f"{group}.unit")
        evidence_column = str(track.get("evidence_column", "")).strip()
        evidence_value = str(track.get("evidence_value", "")).strip()
        if bool(evidence_column) == bool(evidence_value):
            raise ExportError(
                f"{group} requires one evidence_column or evidence_value"
            )
        timing_column = str(track.get("timing_semantics_column", "")).strip()
        timing_value = str(track.get("timing_semantics_value", "")).strip()
        if bool(timing_column) == bool(timing_value):
            raise ExportError(
                f"{group} requires one timing semantics column or value"
            )
        referenced = set(
            name_columns
            + lane_columns
            + row_key_columns
            + args_columns
            + [begin_column]
            + ([end_column] if end_column else [duration_column])
            + ([evidence_column] if evidence_column else [])
            + ([timing_column] if timing_column else [])
        )
        missing = sorted(referenced - set(fields))
        if missing:
            raise ExportError(f"{group} source lacks columns: {missing}")
        seen_keys: set[tuple[str, ...]] = set()
        normalized_rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, 2):
            label = f"{path}:{row_number}"
            key = tuple(row_value(row, field, label) for field in row_key_columns)
            if key in seen_keys:
                raise ExportError(f"duplicate row key in {group}: {key}")
            seen_keys.add(key)
            begin_ns = to_ns(row[begin_column], unit, f"{label}:{begin_column}")
            if end_column:
                end_ns = to_ns(row[end_column], unit, f"{label}:{end_column}")
            else:
                end_ns = begin_ns + to_ns(
                    row[duration_column], unit, f"{label}:{duration_column}"
                )
            if end_ns < begin_ns:
                raise ExportError(f"negative interval duration: {label}")
            evidence = row[evidence_column].strip() if evidence_column else evidence_value
            if evidence not in EVIDENCE_CLASSES:
                raise ExportError(f"invalid evidence class in {label}: {evidence!r}")
            timing = row[timing_column].strip() if timing_column else timing_value
            if not timing:
                raise ExportError(f"empty timing semantics in {label}")
            normalized_rows.append(
                {
                    "track_index": track_index,
                    "track_group": group,
                    "category": category,
                    "name": " / ".join(row[field] for field in name_columns),
                    "base_lane": tuple(row[field] for field in lane_columns),
                    "row_key": key,
                    "begin_ns": begin_ns,
                    "end_ns": end_ns,
                    "evidence_class": evidence,
                    "timing_semantics": timing,
                    "args": {field: row[field] for field in args_columns},
                    "source_csv": str(path),
                    "source_row_number": row_number,
                }
            )
        allocate_sublanes(normalized_rows)
        all_rows.extend(normalized_rows)
        source_records.append(
            {
                "track_group": group,
                "path": str(path),
                "sha256": observed_sha,
                "row_count": len(rows),
                "header": fields,
                "event_count": len(normalized_rows),
            }
        )

    if not all_rows:
        raise ExportError("track specification produced no interval events")
    observed_min_ns = min(row["begin_ns"] for row in all_rows)
    observed_max_ns = max(row["end_ns"] for row in all_rows)
    configured_origin = clock.get("origin_ns")
    origin_ns = (
        observed_min_ns
        if configured_origin is None
        else int(configured_origin)
    )
    if origin_ns > observed_min_ns:
        raise ExportError("clock.origin_ns is later than the first event")

    lane_names = sorted(
        {
            (
                row["track_index"],
                row["track_group"],
                row["base_lane"],
                row["sublane"],
            )
            for row in all_rows
        }
    )
    lane_ids = {lane: index + 1 for index, lane in enumerate(lane_names)}
    events: list[dict[str, Any]] = []
    for track_index, group in sorted(
        {(row["track_index"], row["track_group"]) for row in all_rows}
    ):
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": track_index + 1_000,
                "tid": 0,
                "args": {"name": group},
            }
        )
    for lane, tid in lane_ids.items():
        track_index, group, base_lane, sublane = lane
        label = " / ".join(base_lane) or group
        if sublane:
            label += f" / overlap-{sublane}"
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": track_index + 1_000,
                "tid": tid,
                "args": {"name": label},
            }
        )
    for row in sorted(
        all_rows,
        key=lambda item: (
            item["begin_ns"],
            item["track_index"],
            item["base_lane"],
            item["sublane"],
            item["row_key"],
        ),
    ):
        lane = (
            row["track_index"],
            row["track_group"],
            row["base_lane"],
            row["sublane"],
        )
        event_args = {
            **row["args"],
            "workflow05_track_group": row["track_group"],
            "evidence_class": row["evidence_class"],
            "timing_semantics": row["timing_semantics"],
            "source_csv": row["source_csv"],
            "source_row_number": row["source_row_number"],
            "source_begin_ns": row["begin_ns"],
            "source_end_ns": row["end_ns"],
            "clock_domain": clock_domain,
        }
        events.append(
            {
                "ph": "X",
                "name": row["name"],
                "cat": row["category"],
                "pid": row["track_index"] + 1_000,
                "tid": lane_ids[lane],
                "ts": chrome_us(row["begin_ns"] - origin_ns),
                "dur": chrome_us(row["end_ns"] - row["begin_ns"]),
                "args": event_args,
            }
        )

    trace_payload = {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "metadata": {
            "schema": "workflow05-perfetto-chrome-json-v1",
            "clock_domain": clock_domain,
            "origin_ns": origin_ns,
            "parent_clock_mergeable": False,
            "viewer": "Perfetto UI or Trace Processor",
        },
    }
    output_trace.write_text(
        json.dumps(trace_payload, ensure_ascii=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "ready_for_perfetto_parse_attempt",
        "adapter_role": "format_adapter_only_not_viewer_or_analysis_engine",
        "output_format": "chrome_json_supported_by_perfetto",
        "official_format_reference": (
            "https://perfetto.dev/docs/getting-started/other-formats"
        ),
        "intended_open_source_consumers": [
            "Perfetto UI",
            "Perfetto Trace Processor",
        ],
        "track_spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "clock": {
            "domain": clock_domain,
            "origin_ns": origin_ns,
            "observed_min_ns": observed_min_ns,
            "observed_max_ns": observed_max_ns,
            "parent_clock_mergeable": False,
        },
        "sources": source_records,
        "track_group_count": len(seen_groups),
        "interval_event_count": len(all_rows),
        "metadata_event_count": len(events) - len(all_rows),
        "overlap_sublane_count": len(lane_ids),
        "dropped_source_row_count": 0,
        "trace": {
            "path": str(output_trace),
            "sha256": sha256_file(output_trace),
            "size_bytes": output_trace.stat().st_size,
        },
        "perfetto_parse_verified": False,
        "next_required_step": (
            "run probe_workflow05_open_source_trace.py and retain its "
            "attempt/fallback manifest"
        ),
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
