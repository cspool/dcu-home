#!/usr/bin/env python3
"""Export exact Workflow05 process windows through hipprof's native DB bridge.

This script never launches an application or touches a GPU.  It consumes the
R07 deferred-export contract, copies the immutable capture DB for each attempt,
and asks hipprof to emit bounded PFTrace and/or Chrome JSON traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT_TYPES = {"pftrace": 2, "chrome-json": 0}
SOURCE_PRIORITY = [
    "hipprof_bounded_native_pftrace",
    "hipprof_bounded_native_chrome_json_plus_exact_db_marker_overlay",
    "normalized_perfetto_chrome_overlay",
    "custom_plotly_timeline_fallback",
]


class ExportError(RuntimeError):
    """Fail-closed native hipprof window export error."""


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


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ExportError(f"invalid SQLite identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def database_snapshot(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
        ]
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {quote_identifier(table)}"
                ).fetchone()[0]
            )
            for table in tables
        }
        return {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
            "schema_version": int(
                connection.execute("PRAGMA schema_version").fetchone()[0]
            ),
            "tables": tables,
            "table_counts": counts,
        }


def validate_copy_mutation(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_tables = set(before["tables"])
    after_tables = set(after["tables"])
    added = sorted(after_tables - before_tables)
    removed = sorted(before_tables - after_tables)
    changed = {
        table: {
            "before": before["table_counts"][table],
            "after": after["table_counts"][table],
        }
        for table in sorted(before_tables & after_tables)
        if before["table_counts"][table] != after["table_counts"][table]
    }
    allowed = not removed and not changed and set(added) <= {"RANGE_SUMMARY"}
    result = {
        "added_tables": added,
        "removed_tables": removed,
        "existing_table_count_changes": changed,
        "known_range_summary_mutation_only": allowed,
        "before": before,
        "after": after,
    }
    if not allowed:
        raise ExportError(f"unexpected hipprof DB-copy mutation: {result}")
    return result


def aggregate_stats(
    connection: sqlite3.Connection,
    table: str,
    begin_index: int,
    end_index: int,
) -> dict[str, int | None]:
    row = connection.execute(
        f"SELECT COUNT(*), COALESCE(SUM(DurationNs), 0), "
        f"MIN(BeginNs), MAX(EndNs) FROM {quote_identifier(table)} "
        "WHERE _Index BETWEEN ? AND ?",
        (begin_index, end_index),
    ).fetchone()
    return {
        "count": int(row[0]),
        "duration_sum_ns": int(row[1]),
        "min_begin_ns": None if row[2] is None else int(row[2]),
        "max_end_ns": None if row[3] is None else int(row[3]),
    }


def source_window_stats(
    database: Path,
    source: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    begin_index = int(window["runtime_index_begin"])
    end_index = int(window["runtime_index_end"])
    hip_table = str(source["hip_table"])
    ops_table = str(source["hipops_table"])
    tx_table = str(source["hiptx_table"])
    with connect_readonly(database) as connection:
        runtime = aggregate_stats(connection, hip_table, begin_index, end_index)
        kernels = aggregate_stats(connection, ops_table, begin_index, end_index)
        marker_rows = connection.execute(
            f"SELECT message, BeginNs, EndNs, pid, tid FROM "
            f"{quote_identifier(tx_table)} "
            "WHERE begin_Index = ? AND end_Index = ? ORDER BY _Index",
            (begin_index, end_index),
        ).fetchall()
    marker_names = [str(row[0]) for row in marker_rows]
    expected_marker = str(window["exact_process_range"])
    checks = {
        "runtime_count_matches_contract": runtime["count"]
        == int(window["strict_contained_runtime_call_count"]),
        "kernel_count_matches_contract": kernels["count"]
        == int(window["strict_owned_kernel_count"]),
        "kernel_duration_matches_contract": kernels["duration_sum_ns"]
        == int(window["strict_owned_kernel_duration_ns"]),
        "exact_process_marker_once": marker_names.count(expected_marker) == 1,
    }
    if not all(checks.values()):
        raise ExportError(
            f"source-window contract mismatch for {expected_marker}: {checks}"
        )
    return {
        "runtime": runtime,
        "kernels": kernels,
        "marker_names": marker_names,
        "marker_bounds": [
            {
                "message": str(row[0]),
                "begin_ns": int(row[1]),
                "end_ns": int(row[2]),
                "pid": int(row[3]),
                "tid": int(row[4]),
            }
            for row in marker_rows
        ],
        "checks": checks,
    }


def chrome_trace_validation(
    path: Path,
    window: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    payload = load_object(path)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ExportError(f"native Chrome JSON has no traceEvents list: {path}")
    phase_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    process_names: list[str] = []
    slice_names: list[str] = []
    exact_marker_events: list[dict[str, Any]] = []
    raw_bounds: dict[str, list[int]] = {"HIP": [], "HIPOPS": []}
    malformed = 0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("ph"), str):
            malformed += 1
            continue
        phase = event["ph"]
        phase_counts[phase] += 1
        category = event.get("cat")
        if isinstance(category, str):
            category_counts[category] += 1
        if phase == "M" and event.get("name") == "process_name":
            args = event.get("args")
            if isinstance(args, dict) and isinstance(args.get("name"), str):
                process_names.append(args["name"])
        if phase == "X" and isinstance(event.get("name"), str):
            slice_names.append(event["name"])
            if event["name"] == str(window["exact_process_range"]):
                exact_marker_events.append(event)
            if category in raw_bounds and isinstance(event.get("args"), dict):
                begin = event["args"].get("BeginNs")
                end = event["args"].get("EndNs")
                try:
                    raw_bounds[category].extend([int(begin), int(end)])
                except (TypeError, ValueError):
                    pass
    expected_marker = str(window["exact_process_range"])
    runtime_count = int(stats["runtime"]["count"])
    kernel_count = int(stats["kernels"]["count"])
    checks = {
        "no_malformed_events": malformed == 0,
        "runtime_slice_count_exact": category_counts["HIP"] == runtime_count,
        "kernel_slice_count_exact": category_counts["HIPOPS"] == kernel_count,
        "exact_process_marker_once": slice_names.count(expected_marker) == 1,
        "hiptx_slice_count_exact": category_counts["HIPTX"] == 1,
        "grouped_stream_track_present": any(
            "Stream on Device" in name for name in process_names
        ),
        "flow_start_step_balanced": phase_counts["s"] == phase_counts["t"],
        "flow_covers_kernels": phase_counts["s"] >= kernel_count,
    }
    if len(exact_marker_events) == 1:
        marker_args = exact_marker_events[0].get("args")
        marker_bound = stats["marker_bounds"][0]
        checks["process_marker_raw_bounds_exact"] = isinstance(
            marker_args, dict
        ) and (
            int(marker_args.get("BeginNs", -1)) == marker_bound["begin_ns"]
            and int(marker_args.get("EndNs", -1)) == marker_bound["end_ns"]
        )
    for category, source_key in (("HIP", "runtime"), ("HIPOPS", "kernels")):
        bounds = raw_bounds[category]
        checks[f"{category.lower()}_raw_bounds_present"] = bool(bounds)
        if bounds:
            checks[f"{category.lower()}_min_begin_exact"] = min(bounds) == int(
                stats[source_key]["min_begin_ns"]
            )
            checks[f"{category.lower()}_max_end_exact"] = max(bounds) == int(
                stats[source_key]["max_end_ns"]
            )
    if not all(checks.values()):
        raise ExportError(f"native Chrome JSON semantic mismatch: {checks}")
    return {
        "status": "pass",
        "event_count": len(events),
        "phase_counts": dict(sorted(phase_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "process_names": sorted(set(process_names)),
        "checks": checks,
    }


def semantic_expectations(
    window: dict[str, Any],
    stats: dict[str, Any],
    *,
    require_process_marker: bool,
) -> dict[str, Any]:
    categories = {
            "HIP": int(stats["runtime"]["count"]),
            "HIPOPS": int(stats["kernels"]["count"]),
    }
    if require_process_marker:
        categories["HIPTX"] = 1
    return {
        "required_category_min_counts": categories,
        "required_slice_names": (
            [str(window["exact_process_range"])]
            if require_process_marker
            else []
        ),
        "required_track_name_substrings": ["Runtime API", "Stream on Device"],
        "required_arg_key_suffixes": ["BeginNs", "EndNs", "index"],
        "minimum_flow_count": int(stats["kernels"]["count"]),
    }


def add_exact_process_marker_overlay(
    native_path: Path,
    output_path: Path,
    window: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Preserve native events and append one exact marker omitted by hipprof."""
    payload = load_object(native_path)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ExportError("native Chrome JSON has no traceEvents list")
    marker_name = str(window["exact_process_range"])
    existing = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "X"
        and event.get("name") == marker_name
    ]
    if len(existing) > 1:
        raise ExportError("native Chrome JSON contains duplicate process markers")
    if existing:
        return {
            "mode": "native_marker_already_present",
            "native_path": str(native_path),
            "candidate_path": str(native_path),
            "appended_event_count": 0,
        }

    origins: set[int] = set()
    numeric_pids: set[int] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("pid"), int):
            numeric_pids.add(int(event["pid"]))
        args = event.get("args")
        if (
            event.get("ph") == "X"
            and isinstance(args, dict)
            and "BeginNs" in args
            and isinstance(event.get("ts"), (int, float))
        ):
            origins.add(
                int(args["BeginNs"]) - round(float(event["ts"]) * 1_000)
            )
    if len(origins) != 1:
        raise ExportError(f"cannot prove one native Chrome clock origin: {origins}")
    origin_ns = origins.pop()
    marker_bound = stats["marker_bounds"][0]
    marker_pid = max(numeric_pids, default=0) + 1
    begin_ns = int(marker_bound["begin_ns"])
    end_ns = int(marker_bound["end_ns"])
    appended = [
        {
            "args": {"name": "[Workflow05] Exact process range from HIPTX DB"},
            "ph": "M",
            "pid": marker_pid,
            "name": "process_name",
            "sort_index": marker_pid,
        },
        {
            "ph": "X",
            "name": marker_name,
            "pid": marker_pid,
            "tid": f"Thread{int(marker_bound['tid'])}",
            "ts": (begin_ns - origin_ns) / 1_000.0,
            "dur": (end_ns - begin_ns) / 1_000.0,
            "cat": "HIPTX",
            "args": {
                "BeginNs": str(begin_ns),
                "EndNs": str(end_ns),
                "pid": int(marker_bound["pid"]),
                "tid": int(marker_bound["tid"]),
                "index_begin": int(window["runtime_index_begin"]),
                "index_end": int(window["runtime_index_end"]),
                "stable_key": str(window["stable_key"]),
                "evidence_class": "observed_exact_hiptx_db",
            },
        },
    ]
    payload["traceEvents"] = [*events, *appended]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "mode": "native_events_plus_exact_db_marker_overlay",
        "native_path": str(native_path),
        "native_sha256": sha256_file(native_path),
        "candidate_path": str(output_path),
        "clock_origin_ns": origin_ns,
        "appended_event_count": len(appended),
        "appended_slice_count": 1,
        "native_event_count": len(events),
        "candidate_event_count": len(payload["traceEvents"]),
    }


def maybe_merge_json_chunks(
    trace_files: list[Path], output_prefix: Path
) -> tuple[list[Path], dict[str, Any] | None]:
    if len(trace_files) <= 1:
        return trace_files, None
    expected = [output_prefix.with_name(f"{output_prefix.name}_{i}.json") for i in range(1, len(trace_files) + 1)]
    if sorted(trace_files) != expected:
        raise ExportError(f"unexpected native JSON chunk names: {trace_files}")
    merged = output_prefix.with_suffix(".json")
    merge_manifest = output_prefix.with_name(output_prefix.name + "_merge.json")
    merger = Path(__file__).with_name("merge_hipprof_trace_chunks.py")
    result = subprocess.run(
        [
            sys.executable,
            str(merger),
            "--output",
            str(merged),
            "--manifest",
            str(merge_manifest),
            "--chunks",
            *(str(path) for path in trace_files),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExportError(f"native JSON chunk merge failed: {result.stderr}")
    return [merged], load_object(merge_manifest)


def run_attempt(
    *,
    hipprof: Path,
    source_database: Path,
    output_root: Path,
    window: dict[str, Any],
    stats: dict[str, Any],
    format_name: str,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    rank = int(window["selection_rank"])
    attempt_root = output_root / f"window_{rank:02d}" / format_name
    attempt_root.mkdir(parents=True, exist_ok=False)
    prefix = attempt_root / "observed_hipprof"
    log_path = attempt_root / "hipprof_export.log"
    command: list[str]
    with tempfile.TemporaryDirectory(
        prefix="db-copy-", dir=attempt_root
    ) as temporary_value:
        copy_path = Path(temporary_value) / "capture.db"
        shutil.copy2(source_database, copy_path)
        before = database_snapshot(copy_path)
        command = [
            str(hipprof),
            "--db",
            str(copy_path),
            "--output-type",
            str(FORMAT_TYPES[format_name]),
            "--group-stream",
            "--index-range",
            f"{int(window['runtime_index_begin'])}:{int(window['runtime_index_end'])}",
            "-o",
            str(prefix),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=timeout_seconds,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                command,
                124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
            timed_out = True
        log_path.write_text(
            "COMMAND\n"
            + json.dumps(command, ensure_ascii=False)
            + "\nSTDOUT\n"
            + result.stdout
            + "\nSTDERR\n"
            + result.stderr,
            encoding="utf-8",
        )
        after = database_snapshot(copy_path)
        mutation = validate_copy_mutation(before, after)
    generated = sorted(
        path
        for path in attempt_root.iterdir()
        if path.is_file() and path != log_path
    )
    trace_suffix = ".pftrace" if format_name == "pftrace" else ".json"
    traces = [path for path in generated if path.suffix == trace_suffix]
    merge_record = None
    if format_name == "chrome-json" and len(traces) > 1:
        traces, merge_record = maybe_merge_json_chunks(traces, prefix)
    status = "pass"
    reason = "hipprof native bounded export completed"
    semantic_validation = None
    if result.returncode != 0:
        status = "fail"
        reason = f"hipprof exited with {result.returncode}"
    elif len(traces) != 1 or traces[0].stat().st_size == 0:
        status = "fail"
        reason = f"expected one nonempty {trace_suffix} trace, found {traces}"
    marker_overlay = None
    if result.returncode == 0 and len(traces) == 1 and format_name == "chrome-json":
        try:
            native_trace = traces[0]
            overlay_path = prefix.with_name(
                prefix.name + "_with_process_marker.json"
            )
            marker_overlay = add_exact_process_marker_overlay(
                native_trace, overlay_path, window, stats
            )
            traces = [Path(marker_overlay["candidate_path"])]
            semantic_validation = chrome_trace_validation(traces[0], window, stats)
        except ExportError as exc:
            status = "fail"
            reason = str(exc)
    expectations = semantic_expectations(
        window,
        stats,
        require_process_marker=(format_name == "chrome-json"),
    )
    return {
        "format": format_name,
        "status": status,
        "reason": reason,
        "command": command,
        "exit_status": result.returncode,
        "timed_out": timed_out,
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "copy_mutation": mutation,
        "generated_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "trace_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "semantic_expectations": expectations,
            }
            for path in traces
        ],
        "json_chunk_merge": merge_record,
        "exact_process_marker_overlay": marker_overlay,
        "semantic_validation": semantic_validation,
        "semantic_expectations": expectations,
        "gpu_or_model_activity": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export exact process windows from an immutable hipprof DB copy "
            "without launching an application or touching a GPU."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hipprof-bin", type=Path, default=Path("/opt/dtk/bin/hipprof"))
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=tuple(FORMAT_TYPES),
        default=list(FORMAT_TYPES),
    )
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    output_root = args.output_dir.resolve()
    hipprof = args.hipprof_bin.expanduser().resolve()
    if not contract_path.is_file():
        raise ExportError(f"deferred export contract is missing: {contract_path}")
    if not hipprof.is_file() or not os.access(hipprof, os.X_OK):
        raise ExportError(f"hipprof is not executable: {hipprof}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ExportError(f"refusing nonempty output directory: {output_root}")
    if args.timeout_seconds <= 0:
        raise ExportError("timeout must be positive")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ExportError("max-windows must be positive")
    formats = list(dict.fromkeys(args.formats))

    contract = load_object(contract_path)
    source = contract.get("source_database")
    windows = contract.get("windows")
    if not isinstance(source, dict) or not isinstance(windows, list) or not windows:
        raise ExportError("contract lacks source_database or windows")
    source_database = Path(str(source.get("path", ""))).resolve()
    if not source_database.is_file():
        raise ExportError(f"source database is missing: {source_database}")
    expected_sha = source.get("sha256_after_validation") or source.get(
        "sha256_before_validation"
    )
    source_sha_before = sha256_file(source_database)
    if source_sha_before != expected_sha:
        raise ExportError("source database SHA-256 differs from the contract")
    ranks = [int(window["selection_rank"]) for window in windows]
    if ranks != list(range(1, len(windows) + 1)):
        raise ExportError(f"window ranks are not contiguous: {ranks}")
    if len({str(window["stable_key"]) for window in windows}) != len(windows):
        raise ExportError("window stable keys are not unique")
    selected_windows = windows[: args.max_windows] if args.max_windows else windows

    environment = dict(os.environ)
    library_dirs = [str(path.expanduser().resolve()) for path in args.library_dir]
    if any(not Path(path).is_dir() for path in library_dirs):
        raise ExportError("one or more library directories are missing")
    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(
        library_dirs + ([existing_library_path] if existing_library_path else [])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for window in selected_windows:
        stats = source_window_stats(source_database, source, window)
        attempts = [
            run_attempt(
                hipprof=hipprof,
                source_database=source_database,
                output_root=output_root,
                window=window,
                stats=stats,
                format_name=format_name,
                environment=environment,
                timeout_seconds=args.timeout_seconds,
            )
            for format_name in formats
        ]
        records.append(
            {
                "selection_rank": int(window["selection_rank"]),
                "stable_key": str(window["stable_key"]),
                "event_id": str(window["event_id"]),
                "exact_process_range": str(window["exact_process_range"]),
                "runtime_index_begin": int(window["runtime_index_begin"]),
                "runtime_index_end": int(window["runtime_index_end"]),
                "source_window_stats": stats,
                "attempts": attempts,
            }
        )
    source_sha_after = sha256_file(source_database)
    if source_sha_after != source_sha_before:
        raise ExportError("source database changed during offline export")
    failed = [
        (record["selection_rank"], attempt["format"], attempt["reason"])
        for record in records
        for attempt in record["attempts"]
        if attempt["status"] != "pass"
    ]
    manifest = {
        "schema_version": 1,
        "status": "pass" if not failed else "degraded_attempts_recorded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "native_hipprof_bounded_window_export_no_capture",
        "observed_trace_source_priority": SOURCE_PRIORITY,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "source_database": {
            "path": str(source_database),
            "sha256_before": source_sha_before,
            "sha256_after": source_sha_after,
            "unchanged": True,
        },
        "exporter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "hipprof": {"path": str(hipprof), "sha256": sha256_file(hipprof)},
        "library_directories": library_dirs,
        "formats_attempted": formats,
        "requested_window_count": len(selected_windows),
        "contract_window_count": len(windows),
        "partial_smoke_export": len(selected_windows) != len(windows),
        "windows": records,
        "failed_attempts": failed,
        "gpu_or_model_activity": False,
    }
    manifest_path = output_root / "workflow05_native_hipprof_window_exports.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
