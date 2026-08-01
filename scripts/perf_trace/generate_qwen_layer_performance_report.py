#!/usr/bin/env python3
"""Generate strict launch-owned Qwen layer evidence from a fresh hipprof DB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LAYER_RE = re.compile(
    r"^pra\.layer\.input(?P<forward>\d+)_layer(?P<layer>\d+)\."
    r"(?P<phase>prefill_chunk|decode)\."
    r"(?P<workload>linear_attention|full_attention)$"
)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty required CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _kernel_family(name: str) -> str:
    low = name.lower()
    if "mmac" in low or low.startswith("cijk_"):
        return "TunableOp_MMAC_GEMM"
    if "llmm" in low or "gemv" in low:
        return "LLMM1_GEMV"
    if "gdn" in low or "gated_delta" in low:
        return "GDN_recurrent"
    if "unified_attention" in low or "paged_attention" in low:
        return "KV_cache_attention"
    if "rotary" in low or "rope" in low:
        return "RoPE"
    if "rms" in low or "norm" in low:
        return "RMSNorm"
    if "silu" in low:
        return "fused_SiLU_mul"
    if "reduce" in low or "softmax" in low:
        return "reduction_softmax"
    if "copy" in low or "memcpy" in low:
        return "copy_cache"
    if "elementwise" in low or "vectorized" in low or "sigmoid" in low:
        return "elementwise"
    if "gemm" in low or "matmul" in low:
        return "GEMM_other"
    return "other"


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"event line {line_number} is not an object")
        events.append(value)
    if not events:
        raise RuntimeError("runtime layer event file is empty")
    return events


def _first_count_layer_ranges(
    database: Path,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """The mandatory first audit: count all layer HIPTX ranges."""
    connection = sqlite3.connect(database)
    tables = [
        row[0]
        for row in connection.execute(
            "select name from sqlite_master "
            "where type='table' and name like 'HIPTX_%' order by name"
        )
    ]
    if not tables:
        connection.close()
        raise RuntimeError("fresh hipprof DB has no HIPTX table")
    details: list[dict[str, Any]] = []
    total = 0
    for table in tables:
        query = (
            f"SELECT COUNT(*) FROM {_quote(table)} "
            "WHERE message LIKE 'pra.layer.%'"
        )
        count = int(connection.execute(query).fetchone()[0])
        details.append({"table": table, "query": query, "count": count})
        total += count
    connection.close()
    return total, details, tables


def _read_hipprof(
    database: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    tables = [
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type='table' order by name"
        )
    ]
    configs = [
        dict(row)
        for row in connection.execute("select * from CONFIG")
    ]
    layers: list[dict[str, Any]] = []
    process_markers: list[dict[str, Any]] = []
    request_markers: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []

    for table in tables:
        if table.startswith("HIPTX_"):
            config_key = table[len("HIPTX_") :]
            for row in connection.execute(f"select * from {_quote(table)}"):
                message = str(row["message"] or "")
                base = {
                    "source_db": str(database),
                    "config_key": config_key,
                    "pid": int(row["pid"]),
                    "tid": int(row["tid"]),
                    "marker_index": int(row["_Index"]),
                    "begin_ns": int(row["BeginNs"]),
                    "end_ns": int(row["EndNs"]),
                    "duration_ns": int(row["EndNs"]) - int(row["BeginNs"]),
                    "begin_runtime_index": int(row["begin_Index"]),
                    "end_runtime_index": int(row["end_Index"]),
                    "range_name": message,
                }
                match = LAYER_RE.fullmatch(message)
                if match:
                    layers.append(
                        {
                            **base,
                            "forward_id": int(match.group("forward")),
                            "layer_idx": int(match.group("layer")),
                            "phase": match.group("phase"),
                            "workload_type": match.group("workload"),
                        }
                    )
                elif message.startswith("pra.fx_process."):
                    process_markers.append(base)
                elif message.startswith("pra.request."):
                    request_markers.append(base)
        elif table.startswith("HIP_"):
            config_key = table[len("HIP_") :]
            for row in connection.execute(f"select * from {_quote(table)}"):
                args = str(row["args"] or "")
                runtime_calls.append(
                    {
                        "source_db": str(database),
                        "config_key": config_key,
                        "pid": int(row["pid"]),
                        "tid": int(row["tid"]),
                        "runtime_index": int(row["_Index"]),
                        "begin_ns": int(row["BeginNs"]),
                        "end_ns": int(row["EndNs"]),
                        "duration_ns": int(row["DurationNs"]),
                        "api_name": (
                            args.split("(", 1)[0]
                            if "(" in args
                            else f"hip_api_name_id_{row['Name']}"
                        ),
                        "args": args,
                    }
                )
        elif table.startswith("HIPOPS_"):
            config_key = table[len("HIPOPS_") :]
            for kernel_id, row in enumerate(
                connection.execute(f"select * from {_quote(table)}"),
                1,
            ):
                name = str(row["Name"])
                kernels.append(
                    {
                        "kernel_id": (
                            f"{config_key}:{row['pid']}:{kernel_id}:"
                            f"{row['BeginNs']}"
                        ),
                        "source_db": str(database),
                        "config_key": config_key,
                        "pid": int(row["pid"]),
                        "runtime_index": int(row["_Index"]),
                        "begin_ns": int(row["BeginNs"]),
                        "end_ns": int(row["EndNs"]),
                        "duration_ns": int(row["DurationNs"]),
                        "device_id": int(row["dev_id"]),
                        "queue_id": str(row["queue_id"]),
                        "kernel_name": name,
                        "kernel_family": _kernel_family(name),
                        "launch_parameters": str(row["PARS"] or ""),
                    }
                )
    connection.close()
    return (
        layers,
        process_markers,
        request_markers,
        runtime_calls,
        kernels,
        configs,
    )


def _write_normalized_sqlite(
    path: Path,
    metadata: dict[str, Any],
    tables: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    connection = sqlite3.connect(path)
    connection.execute("create table metadata (key text primary key, value text)")
    connection.executemany(
        "insert into metadata values (?, ?)",
        [
            (
                key,
                value
                if isinstance(value, str)
                else json.dumps(value, sort_keys=True),
            )
            for key, value in metadata.items()
        ],
    )
    for table_name, rows in tables:
        if not rows:
            raise RuntimeError(f"normalized table {table_name} would be empty")
        fields = list(rows[0])
        types: list[str] = []
        for field in fields:
            sample = next(
                (row[field] for row in rows if row.get(field) is not None),
                "",
            )
            if isinstance(sample, int):
                types.append("INTEGER")
            elif isinstance(sample, float):
                types.append("REAL")
            else:
                types.append("TEXT")
        schema = ", ".join(
            f"{_quote(field)} {field_type}"
            for field, field_type in zip(fields, types)
        )
        connection.execute(f"create table {_quote(table_name)} ({schema})")
        placeholders = ", ".join("?" for _ in fields)
        connection.executemany(
            f"insert into {_quote(table_name)} values ({placeholders})",
            [[row.get(field) for field in fields] for row in rows],
        )
    connection.commit()
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--event-jsonl", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-layers", type=int, default=64)
    parser.add_argument("--expected-device-id", type=int, default=1)
    args = parser.parse_args()

    for required in (
        args.raw_db,
        args.raw_trace,
        args.event_jsonl,
        args.run_metadata,
        args.contract,
    ):
        if not required.is_file() or required.stat().st_size == 0:
            raise RuntimeError(f"missing or empty fresh artifact: {required}")

    # Required audit ordering: this is deliberately the first evidence check.
    layer_range_count, first_queries, hiptx_tables = _first_count_layer_ranges(
        args.raw_db
    )

    contract = _load_object(args.contract)
    run_metadata = _load_object(args.run_metadata)
    events = _read_events(args.event_jsonl)
    (
        layer_ranges,
        process_markers,
        request_markers,
        runtime_calls,
        kernels,
        configs,
    ) = _read_hipprof(args.raw_db)

    if layer_range_count != len(layer_ranges):
        raise RuntimeError("first HIPTX count does not equal parsed layer ranges")
    if process_markers:
        raise RuntimeError("process profiling is off but process ranges were captured")
    if len(request_markers) != 1:
        raise RuntimeError(
            f"expected one synchronized request marker, found {len(request_markers)}"
        )
    if run_metadata.get("tag") != args.tag:
        raise RuntimeError("run metadata tag mismatch")
    if run_metadata.get("contract_id") != contract.get("contract_id"):
        raise RuntimeError("run metadata contract_id mismatch")
    if run_metadata.get("contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("run metadata contract_sha256 mismatch")
    if run_metadata.get("process_profile") != "off":
        raise RuntimeError("run metadata does not prove process profiling off")

    event_by_range: dict[str, dict[str, Any]] = {}
    occurrence_keys: set[str] = set()
    for event in events:
        range_name = str(event["range_name"])
        if range_name in event_by_range:
            raise RuntimeError(f"duplicate runtime event range: {range_name}")
        occurrence_key = str(event["occurrence_key"])
        if occurrence_key in occurrence_keys:
            raise RuntimeError(f"duplicate occurrence key: {occurrence_key}")
        occurrence_keys.add(occurrence_key)
        event_by_range[range_name] = event
    range_names = {row["range_name"] for row in layer_ranges}
    event_names = set(event_by_range)
    missing_event_rows = sorted(range_names - event_names)
    missing_range_rows = sorted(event_names - range_names)
    if missing_event_rows or missing_range_rows:
        raise RuntimeError(
            "layer event/range join failed: "
            f"missing_events={missing_event_rows[:5]}, "
            f"missing_ranges={missing_range_rows[:5]}"
        )

    runtime_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for call in runtime_calls:
        runtime_by_key[
            (call["config_key"], call["pid"], call["runtime_index"])
        ].append(call)
    kernels_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for kernel in kernels:
        kernels_by_key[
            (kernel["config_key"], kernel["pid"], kernel["runtime_index"])
        ].append(kernel)

    ownership_rows: list[dict[str, Any]] = []
    owned_by_range: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_launch_indices_by_range: dict[str, set[int]] = defaultdict(set)
    for layer_range in layer_ranges:
        for runtime_index in range(
            layer_range["begin_runtime_index"],
            layer_range["end_runtime_index"] + 1,
        ):
            key = (
                layer_range["config_key"],
                layer_range["pid"],
                runtime_index,
            )
            calls = [
                call
                for call in runtime_by_key.get(key, [])
                if layer_range["begin_ns"]
                <= call["begin_ns"]
                <= layer_range["end_ns"]
            ]
            for call in calls:
                matched = kernels_by_key.get(key, [])
                for kernel in matched:
                    row = {
                        "contract_id": contract["contract_id"],
                        "forward_id": layer_range["forward_id"],
                        "layer_idx": layer_range["layer_idx"],
                        "occurrence": 0,
                        "range_name": layer_range["range_name"],
                        "runtime_index": runtime_index,
                        "runtime_api": call["api_name"],
                        "runtime_begin_ns": call["begin_ns"],
                        "runtime_end_ns": call["end_ns"],
                        "runtime_host_start_inside_range": 1,
                        "runtime_index_inside_marker_bounds": 1,
                        "runtime_end_inside_range": int(
                            call["end_ns"] <= layer_range["end_ns"]
                        ),
                        "kernel_id": kernel["kernel_id"],
                        "kernel_name": kernel["kernel_name"],
                        "kernel_family": kernel["kernel_family"],
                        "kernel_begin_ns": kernel["begin_ns"],
                        "kernel_end_ns": kernel["end_ns"],
                        "kernel_duration_ns": kernel["duration_ns"],
                        "device_id": kernel["device_id"],
                        "queue_id": kernel["queue_id"],
                        "ownership_method": (
                            "HIPTX_host_range_to_HIP_runtime_host_start_"
                            "to_HIPOPS_identical__Index"
                        ),
                    }
                    ownership_rows.append(row)
                    owned_by_range[layer_range["range_name"]].append(row)
                    runtime_launch_indices_by_range[
                        layer_range["range_name"]
                    ].add(runtime_index)

    layer_rows: list[dict[str, Any]] = []
    kernel_family_rows: list[dict[str, Any]] = []
    launch_order_rows: list[dict[str, Any]] = []
    failed_ownership: list[str] = []
    for layer_range in sorted(
        layer_ranges,
        key=lambda row: (
            row["forward_id"],
            row["layer_idx"],
            row["begin_ns"],
        ),
    ):
        event = event_by_range[layer_range["range_name"]]
        for key in (
            "forward_id",
            "layer_idx",
            "phase",
            "workload_type",
        ):
            if event[key] != layer_range[key]:
                raise RuntimeError(
                    f"event/HIPTX mismatch for {layer_range['range_name']}:{key}"
                )
        if int(event["pid"]) != layer_range["pid"]:
            raise RuntimeError("event and HIPTX marker PID differ")

        unique_owned = {
            row["kernel_id"]: row
            for row in owned_by_range[layer_range["range_name"]]
        }
        owned = sorted(
            unique_owned.values(),
            key=lambda row: (row["kernel_begin_ns"], row["kernel_id"]),
        )
        if not owned:
            failed_ownership.append(layer_range["range_name"])
        durations = [row["kernel_duration_ns"] for row in owned]
        busy_union_ns = _merge_duration_ns(
            (row["kernel_begin_ns"], row["kernel_end_ns"]) for row in owned
        )
        kernel_span_ns = (
            max(row["kernel_end_ns"] for row in owned)
            - min(row["kernel_begin_ns"] for row in owned)
            if owned
            else 0
        )
        layer_row = {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract["contract_sha256"],
            "forward_id": layer_range["forward_id"],
            "layer_idx": layer_range["layer_idx"],
            "occurrence": int(event["occurrence"]),
            "occurrence_key": event["occurrence_key"],
            "phase": event["phase"],
            "q_len": int(event["q_len"]),
            "past_len": int(event["past_len"]),
            "kv_len": int(event["kv_len"]),
            "workload_type": event["workload_type"],
            "range_name": layer_range["range_name"],
            "hiptx_begin_ns": layer_range["begin_ns"],
            "hiptx_end_ns": layer_range["end_ns"],
            "hiptx_host_range_duration_ms": layer_range["duration_ns"] / 1e6,
            "hiptx_begin_runtime_index": layer_range[
                "begin_runtime_index"
            ],
            "hiptx_end_runtime_index": layer_range["end_runtime_index"],
            "hipprof_launch_owned_kernel_sum_ms": sum(durations) / 1e6,
            "hipprof_launch_owned_kernel_busy_union_ms": busy_union_ns / 1e6,
            "hipprof_launch_owned_kernel_span_ms": kernel_span_ns / 1e6,
            "launch_owned_kernel_count": len(owned),
            "launch_runtime_index_count": len(
                runtime_launch_indices_by_range[layer_range["range_name"]]
            ),
            "device_ids": ";".join(
                str(value) for value in sorted({row["device_id"] for row in owned})
            ),
            "attribution_status": "pass" if owned else "failed_join",
            "ownership_method": (
                "HIPTX host range -> HIP Runtime host start -> "
                "HIPOPS identical runtime _Index"
            ),
        }
        layer_rows.append(layer_row)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for order, row in enumerate(owned, 1):
            grouped[row["kernel_family"]].append(row)
            launch_order_rows.append(
                {
                    "contract_id": contract["contract_id"],
                    "forward_id": layer_range["forward_id"],
                    "layer_idx": layer_range["layer_idx"],
                    "occurrence": int(event["occurrence"]),
                    "launch_order": order,
                    "runtime_index": row["runtime_index"],
                    "kernel_id": row["kernel_id"],
                    "kernel_family": row["kernel_family"],
                    "kernel_name": row["kernel_name"],
                    "kernel_duration_ms": row["kernel_duration_ns"] / 1e6,
                    "device_id": row["device_id"],
                    "queue_id": row["queue_id"],
                }
            )
        layer_total_ns = sum(durations)
        for family, family_owned in sorted(
            grouped.items(),
            key=lambda item: min(
                row["kernel_begin_ns"] for row in item[1]
            ),
        ):
            family_ns = sum(row["kernel_duration_ns"] for row in family_owned)
            kernel_family_rows.append(
                {
                    "contract_id": contract["contract_id"],
                    "forward_id": layer_range["forward_id"],
                    "layer_idx": layer_range["layer_idx"],
                    "occurrence": int(event["occurrence"]),
                    "phase": event["phase"],
                    "q_len": int(event["q_len"]),
                    "past_len": int(event["past_len"]),
                    "kv_len": int(event["kv_len"]),
                    "workload_type": event["workload_type"],
                    "range_name": layer_range["range_name"],
                    "kernel_family": family,
                    "kernel_count": len(family_owned),
                    "launch_owned_kernel_duration_ms": family_ns / 1e6,
                    "layer_launch_owned_kernel_sum_ms": layer_total_ns / 1e6,
                    "pct_of_layer_launch_owned_kernel_sum": (
                        100.0 * family_ns / layer_total_ns
                        if layer_total_ns
                        else 0.0
                    ),
                    "runtime_index_examples": ";".join(
                        str(row["runtime_index"]) for row in family_owned[:3]
                    ),
                    "kernel_name_examples": ";".join(
                        row["kernel_name"] for row in family_owned[:2]
                    ),
                    "ownership_method": (
                        "launch-owned full HIPOPS durations by identical _Index"
                    ),
                }
            )

    if failed_ownership:
        raise RuntimeError(
            f"layer markers with failed strict ownership: {failed_ownership[:8]}"
        )
    if len(layer_rows) != layer_range_count:
        raise RuntimeError("layer row count differs from first HIPTX count")

    forwards: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        forwards[row["forward_id"]].append(row)
    forward_ids = sorted(forwards)
    if forward_ids != list(range(1, len(forward_ids) + 1)):
        raise RuntimeError(f"non-contiguous forward IDs: {forward_ids}")
    expected_layer_indices = list(range(args.expected_layers))
    expected_layer_types = contract["model"]["layer_types"]
    if len(expected_layer_types) != args.expected_layers:
        raise RuntimeError("contract model layer_types length mismatch")
    expected_past_len = 0
    forward_rows: list[dict[str, Any]] = []
    for forward_id in forward_ids:
        rows = sorted(forwards[forward_id], key=lambda row: row["layer_idx"])
        layer_indices = [row["layer_idx"] for row in rows]
        if layer_indices != expected_layer_indices:
            raise RuntimeError(
                f"forward {forward_id} layer coverage mismatch: {layer_indices}"
            )
        identity = {
            (
                row["phase"],
                row["q_len"],
                row["past_len"],
                row["kv_len"],
            )
            for row in rows
        }
        if len(identity) != 1:
            raise RuntimeError(f"forward {forward_id} metadata is inconsistent")
        phase, q_len, past_len, kv_len = next(iter(identity))
        if past_len != expected_past_len or kv_len != past_len + q_len:
            raise RuntimeError(f"forward {forward_id} sequence lengths are invalid")
        if phase == "prefill_chunk" and q_len <= 1:
            raise RuntimeError("prefill chunk must contain more than one token")
        if phase == "decode" and q_len != 1:
            raise RuntimeError("decode forward must contain exactly one token")
        for row in rows:
            if row["workload_type"] != expected_layer_types[row["layer_idx"]]:
                raise RuntimeError(
                    f"layer type mismatch at layer {row['layer_idx']}"
                )
        expected_past_len += q_len
        forward_rows.append(
            {
                "forward_id": forward_id,
                "phase": phase,
                "q_len": q_len,
                "past_len": past_len,
                "kv_len": kv_len,
                "layer_count": len(rows),
                "launch_owned_kernel_sum_ms": sum(
                    row["hipprof_launch_owned_kernel_sum_ms"] for row in rows
                ),
                "hiptx_host_range_sum_ms": sum(
                    row["hiptx_host_range_duration_ms"] for row in rows
                ),
            }
        )

    device_ids = sorted(
        {
            row["device_id"]
            for row in ownership_rows
        }
    )
    if device_ids != [args.expected_device_id]:
        raise RuntimeError(
            f"owned HIPOPS device IDs {device_ids} != "
            f"[{args.expected_device_id}]"
        )
    if len(events) != layer_range_count:
        raise RuntimeError("runtime event count differs from HIPTX layer count")

    metric_rows: list[dict[str, Any]] = []
    metric_specs = (
        (
            "hiptx_host_range_duration_ms",
            "hiptx_host_range_duration_ms",
            "host_context",
        ),
        (
            "hipprof_launch_owned_kernel_sum_ms",
            "hipprof_launch_owned_kernel_sum_ms",
            "downstream_denominator",
        ),
        (
            "hipprof_launch_owned_kernel_busy_union_ms",
            "hipprof_launch_owned_kernel_busy_union_ms",
            "diagnostic_only",
        ),
    )
    for row in layer_rows:
        for metric, source_field, role in metric_specs:
            metric_rows.append(
                {
                    "contract_id": row["contract_id"],
                    "contract_sha256": row["contract_sha256"],
                    "forward_id": row["forward_id"],
                    "layer_idx": row["layer_idx"],
                    "occurrence": row["occurrence"],
                    "occurrence_key": row["occurrence_key"],
                    "metric": metric,
                    "metric_value_ms": row[source_field],
                    "metric_role": role,
                    "phase": row["phase"],
                    "q_len": row["q_len"],
                    "past_len": row["past_len"],
                    "kv_len": row["kv_len"],
                    "workload_type": row["workload_type"],
                    "range_name": row["range_name"],
                    "launch_owned_kernel_count": row[
                        "launch_owned_kernel_count"
                    ],
                    "attribution_status": row["attribution_status"],
                    "ownership_method": row["ownership_method"],
                }
            )
    metric_keys = {
        (
            row["contract_id"],
            row["forward_id"],
            row["layer_idx"],
            row["occurrence"],
            row["metric"],
        )
        for row in metric_rows
    }
    if len(metric_keys) != len(metric_rows):
        raise RuntimeError("duplicate contract/forward/layer/occurrence/metric key")
    if len(metric_rows) != layer_range_count * len(metric_specs):
        raise RuntimeError("all input-layer metric coverage is incomplete")

    api_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "duration_ns": 0}
    )
    for call in runtime_calls:
        api_stats[call["api_name"]]["calls"] += 1
        api_stats[call["api_name"]]["duration_ns"] += call["duration_ns"]
    api_stat_rows = [
        {
            "api_name": name,
            "calls": values["calls"],
            "total_duration_ms": values["duration_ns"] / 1e6,
        }
        for name, values in sorted(
            api_stats.items(),
            key=lambda item: item[1]["duration_ns"],
            reverse=True,
        )
    ]
    phase_stat_rows: list[dict[str, Any]] = []
    for phase in sorted({row["phase"] for row in layer_rows}):
        phase_rows = [row for row in layer_rows if row["phase"] == phase]
        total_ms = sum(
            row["hipprof_launch_owned_kernel_sum_ms"] for row in phase_rows
        )
        phase_stat_rows.append(
            {
                "phase": phase,
                "forward_count": len(
                    {row["forward_id"] for row in phase_rows}
                ),
                "layer_event_count": len(phase_rows),
                "launch_owned_kernel_sum_ms": total_ms,
                "mean_launch_owned_kernel_sum_ms": total_ms / len(phase_rows),
                "hiptx_host_range_sum_ms": sum(
                    row["hiptx_host_range_duration_ms"]
                    for row in phase_rows
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    layer_events_path = output_dir / f"{args.tag}_layer_events.csv"
    breakdown_csv_path = (
        output_dir / f"{args.tag}_layer_kernel_breakdown.csv"
    )
    breakdown_json_path = (
        output_dir / f"{args.tag}_layer_kernel_breakdown.json"
    )
    all_layers_path = (
        output_dir / f"{args.tag}_all_input_layer_performance.csv"
    )
    ownership_path = output_dir / f"{args.tag}_strict_ownership.csv"
    launch_order_path = (
        output_dir / f"{args.tag}_layer_kernel_launch_order.csv"
    )
    stats_api_path = output_dir / f"{args.tag}_stats_hip_api.csv"
    stats_phase_path = output_dir / f"{args.tag}_stats_phase.csv"
    normalized_db_path = output_dir / f"{args.tag}.sqlite"
    audit_path = output_dir / f"{args.tag}_audit.json"
    report_path = (
        report_dir
        / "SAME_INPUT_QWEN3_5_27B_VLLM_PRA_LAYER_PERFORMANCE_REPORT.md"
    )

    _write_csv(layer_events_path, layer_rows)
    _write_csv(breakdown_csv_path, kernel_family_rows)
    _write_json(breakdown_json_path, kernel_family_rows)
    _write_csv(all_layers_path, metric_rows)
    _write_csv(ownership_path, ownership_rows)
    _write_csv(launch_order_path, launch_order_rows)
    _write_csv(stats_api_path, api_stat_rows)
    _write_csv(stats_phase_path, phase_stat_rows)

    raw_db_sha256 = _sha256(args.raw_db)
    raw_trace_sha256 = _sha256(args.raw_trace)
    event_sha256 = _sha256(args.event_jsonl)
    _write_normalized_sqlite(
        normalized_db_path,
        {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract["contract_sha256"],
            "raw_db": str(args.raw_db.resolve()),
            "raw_db_sha256": raw_db_sha256,
            "raw_trace": str(args.raw_trace.resolve()),
            "raw_trace_sha256": raw_trace_sha256,
            "runtime_events": str(args.event_jsonl.resolve()),
            "runtime_events_sha256": event_sha256,
            "ownership_rule": (
                "layer HIPTX host range -> HIP Runtime call whose host start "
                "lies inside range -> HIPOPS identical runtime _Index -> "
                "sum full kernel durations as launch-owned kernel time"
            ),
            "device_timestamp_overlap_attribution": "forbidden",
        },
        [
            ("layer_events", layer_rows),
            ("strict_ownership", ownership_rows),
            ("layer_kernel_breakdown", kernel_family_rows),
            ("all_input_layer_performance", metric_rows),
        ],
    )

    summary = {
        "status": "pass",
        "first_check": {
            "description": "count all layer HIPTX ranges in queryable trace",
            "database": str(args.raw_db.resolve()),
            "hiptx_tables": hiptx_tables,
            "queries": first_queries,
            "count": layer_range_count,
        },
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "raw_db_sha256": raw_db_sha256,
        "raw_trace_sha256": raw_trace_sha256,
        "runtime_events_sha256": event_sha256,
        "forward_count": len(forward_ids),
        "expected_layer_count_per_forward": args.expected_layers,
        "layer_range_count": layer_range_count,
        "runtime_layer_event_count": len(events),
        "occurrence_key_count": len(occurrence_keys),
        "strict_ownership_rows": len(ownership_rows),
        "unique_launch_owned_kernels": len(
            {row["kernel_id"] for row in ownership_rows}
        ),
        "failed_ownership_ranges": failed_ownership,
        "missing_event_rows": missing_event_rows,
        "missing_range_rows": missing_range_rows,
        "all_input_layer_metric_rows": len(metric_rows),
        "downstream_denominator_metric": (
            "hipprof_launch_owned_kernel_sum_ms"
        ),
        "device_ids": device_ids,
        "process_marker_count": len(process_markers),
        "request_marker_count": len(request_markers),
        "request_synchronized_latency_ms": run_metadata[
            "request_synchronized_latency_ms"
        ],
        "ownership_rule": (
            "layer HIPTX host range -> HIP Runtime host start inside range "
            "and marker index bounds -> HIPOPS identical runtime _Index -> "
            "full durations summed as launch-owned kernel time"
        ),
        "device_timestamp_overlap_attribution_used": False,
        "nested_total_attn_mlp_summed_as_independent_costs": False,
        "forward_rows": forward_rows,
        "outputs": {
            "layer_events": str(layer_events_path),
            "layer_kernel_breakdown_csv": str(breakdown_csv_path),
            "layer_kernel_breakdown_json": str(breakdown_json_path),
            "all_input_layer_performance": str(all_layers_path),
            "normalized_queryable_trace": str(normalized_db_path),
            "report": str(report_path),
        },
    }
    _write_json(audit_path, summary)

    phase_table = [
        "| phase | forwards | layer events | launch-owned kernel sum (ms) | HIPTX host range sum (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in phase_stat_rows:
        phase_table.append(
            f"| {row['phase']} | {row['forward_count']} | "
            f"{row['layer_event_count']} | "
            f"{row['launch_owned_kernel_sum_ms']:.6f} | "
            f"{row['hiptx_host_range_sum_ms']:.6f} |"
        )
    report = f"""# SAME_INPUT Qwen3.5-27B vLLM/PRA Layer Performance Report

Status: **PASS**

## Frozen SAME_INPUT Contract

- Contract: `{contract['contract_id']}` (`{contract['contract_sha256']}`).
- Source revision: `{contract['source']['revision']}`.
- Variant/config: `{contract['variant']}` / `{contract['config']['name']}`.
- Model: `{contract['model']['model_root']}`; dtype `bfloat16`.
- Attention backend: `ROCM_AITER_UNIFIED_ATTN`; eager execution.
- Input: dataset row {contract['same_input']['prompt']['dataset_row']}, rendered prompt SHA-256 `{contract['same_input']['prompt']['rendered_prompt_sha256']}`.
- Sampling: greedy temperature 0, seed 0, MAX_NEW_TOKENS=32.
- Warmup: 1 identical unprofiled request.
- Device: physical DCU 1; strict-owned HIPOPS device IDs `{device_ids}`.
- Tag/output: `{args.tag}` / `{output_dir}`.

## Strict Attribution

The runtime-resolved ownership chain is:

`layer HIPTX host range -> HIP Runtime call whose host start lies inside the range and whose _Index lies inside marker bounds -> HIPOPS kernel with identical runtime _Index -> full durations summed as launch-owned kernel time`.

Device timestamp overlap attribution was not used. Nested `total`, `attn`, and
`mlp` rows were not summed as independent costs.

The following metrics remain distinct:

- synchronized request latency: {run_metadata['request_synchronized_latency_ms']:.6f} ms;
- HIPTX host range duration: recorded once per layer event;
- hipprof HIP launch-owned kernel sum: downstream layer denominator.

## Completeness Audit

- First check—count all layer HIPTX ranges: {layer_range_count}.
- Complete forwards: {len(forward_ids)}.
- Expected layers in every forward: {args.expected_layers}; observed exact indices 0..{args.expected_layers - 1}.
- Runtime layer events: {len(events)}; unique occurrence keys: {len(occurrence_keys)}.
- Strict ownership rows: {len(ownership_rows)}; unique launch-owned kernels: {summary['unique_launch_owned_kernels']}.
- All input-layer rows: {len(metric_rows)} = {layer_range_count} layer events × {len(metric_specs)} distinct metrics.
- Missing event rows: 0; missing HIPTX rows: 0; failed joins: 0.
- Raw DB, raw trace, runtime events, and exported tables share contract `{contract['contract_id']}` and fresh DB SHA-256 `{raw_db_sha256}`.

The complete all input-layer table is `{all_layers_path.name}`. Its downstream
denominator metric is `hipprof_launch_owned_kernel_sum_ms`; the host-duration
and busy-union rows are context/diagnostic metrics and must not be added to it.

## Phase Summary

{chr(10).join(phase_table)}

## Evidence Boundary

This report measures layer totals only. It does not establish strict
process-wise timing and does not split layer totals among processes.
"""
    report_path.write_text(report, encoding="utf-8")

    run_metadata["status"] = "pass"
    run_metadata["analysis"] = summary
    run_metadata["analysis"]["report"] = str(report_path)
    _write_json(args.run_metadata, run_metadata)
    print(
        json.dumps(
            {
                "status": "pass",
                "first_layer_hiptx_count": layer_range_count,
                "forwards": len(forward_ids),
                "layers_per_forward": args.expected_layers,
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
