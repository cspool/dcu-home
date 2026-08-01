#!/usr/bin/env python3
"""Independently audit Qwen3.5 FX-process HIPTX instrumentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def table_with_prefix(connection: sqlite3.Connection, prefix: str) -> str:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE ? ORDER BY name",
            (f"{prefix}%",),
        )
        if row[0] == prefix or row[0].startswith(f"{prefix}_")
    ]
    if len(names) != 1:
        raise RuntimeError(
            f"expected one {prefix} table, observed {names}"
        )
    return names[0]


def quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def comparable_result(metadata: dict[str, Any]) -> dict[str, Any]:
    result = metadata["measured_result"]
    keys = (
        "prompt_token_count",
        "prompt_token_ids_sha256",
        "output_token_count",
        "output_token_ids_sha256",
        "output_text_sha256",
        "finish_reason",
    )
    return {key: result[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--event-jsonl", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--disabled-run-metadata", type=Path, required=True)
    parser.add_argument("--r01-run-metadata", type=Path, required=True)
    parser.add_argument("--r01-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.raw_db,
        args.run_metadata,
        args.event_jsonl,
        args.inventory,
        args.disabled_run_metadata,
        args.r01_run_metadata,
        args.r01_contract,
    ]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty audit input: {path}")

    metadata = load_object(args.run_metadata)
    disabled = load_object(args.disabled_run_metadata)
    r01_metadata = load_object(args.r01_run_metadata)
    contract = load_object(args.r01_contract)
    if metadata["process_profile"] != "on":
        raise RuntimeError("process run metadata is not process_profile=on")
    if disabled["process_profile"] != "off":
        raise RuntimeError("disabled metadata is not process_profile=off")
    if disabled.get("expected_process_range_count") != 0:
        raise RuntimeError("disabled run expected process ranges")
    for observed in (metadata, disabled, r01_metadata):
        if observed["contract_id"] != contract["contract_id"]:
            raise RuntimeError("same-input contract_id mismatch")
        if observed["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError("same-input contract SHA-256 mismatch")

    result_process = comparable_result(metadata)
    result_disabled = comparable_result(disabled)
    result_r01 = comparable_result(r01_metadata)
    if result_process != result_disabled or result_process != result_r01:
        raise RuntimeError("process/off/R01 measured outputs are not equivalent")
    if metadata["same_input"] != disabled["same_input"]:
        raise RuntimeError("process and disabled prompt provenance differ")
    if metadata["same_input"] != r01_metadata["same_input"]:
        raise RuntimeError("process and R01 prompt provenance differ")

    inventory = list(
        csv.DictReader(args.inventory.open(encoding="utf-8", newline=""))
    )
    if not inventory:
        raise RuntimeError("process inventory is empty")
    inventory_names = [row["nvtx_range_name"] for row in inventory]
    if len(inventory_names) != len(set(inventory_names)):
        raise RuntimeError("inventory range identities are not unique")
    expected_names = set(inventory_names)
    metadata_expected = set(metadata["expected_process_ranges"])
    if expected_names != metadata_expected:
        raise RuntimeError(
            "inventory and runtime metadata expected ranges differ"
        )
    if metadata["expected_process_range_count"] != len(inventory):
        raise RuntimeError("runtime expected range count differs from inventory")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        grouped[row["aggregation_key"]].append(row)
        fragment = row["fragment_id"]
        if fragment and not row["nvtx_range_name"].endswith(f".{fragment}"):
            raise RuntimeError(
                f"fragment identity missing from range: {row['nvtx_range_name']}"
            )
        if not row["expected_kernel_families"].startswith(
            "HYPOTHESIS ONLY:"
        ):
            raise RuntimeError("kernel family inventory is not hypothesis-labeled")
    for aggregation_key, rows in grouped.items():
        parents = {row["range_parent"] for row in rows}
        if len(parents) != 1:
            raise RuntimeError(
                f"aggregation key crosses parents: {aggregation_key}"
            )

    runtime_events = [
        json.loads(line)
        for line in args.event_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(runtime_events) != metadata["observed_layer_events"]:
        raise RuntimeError("runtime event row count mismatch")
    event_expected = {
        name
        for event in runtime_events
        for name in event["expected_process_range_names"]
    }
    if event_expected != expected_names:
        raise RuntimeError("runtime event expectations differ from inventory")
    selected_event_ids = {
        event["event_id"]
        for event in runtime_events
        if event["expected_process_range_names"]
    }
    if selected_event_ids != set(metadata["process_targets"]):
        raise RuntimeError("runtime selected event coverage mismatch")

    connection = sqlite3.connect(f"file:{args.raw_db.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    hiptx_table = table_with_prefix(connection, "HIPTX")
    hip_table = table_with_prefix(connection, "HIP")
    hipops_table = table_with_prefix(connection, "HIPOPS")
    hiptx_columns = [
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({quoted(hiptx_table)})"
        )
    ]
    hip_columns = [
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({quoted(hip_table)})"
        )
    ]
    hipops_columns = [
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({quoted(hipops_table)})"
        )
    ]
    for required, columns, label in (
        (
            {
                "_Index",
                "BeginNs",
                "EndNs",
                "pid",
                "tid",
                "message",
                "begin_Index",
                "end_Index",
            },
            set(hiptx_columns),
            "HIPTX",
        ),
        (
            {"_Index", "BeginNs", "EndNs", "pid", "tid"},
            set(hip_columns),
            "HIP Runtime",
        ),
        (
            {"_Index", "BeginNs", "EndNs", "pid", "tid", "Name"},
            set(hipops_columns),
            "HIPOPS",
        ),
    ):
        if not required.issubset(columns):
            raise RuntimeError(
                f"{label} schema lacks {sorted(required - columns)}"
            )

    marker_rows = list(
        connection.execute(
            f"SELECT _Index, BeginNs, EndNs, pid, tid, message, "
            f"begin_Index, end_Index FROM {quoted(hiptx_table)} "
            "WHERE message LIKE 'pra.fx_process.%' "
            "OR message LIKE 'pra.layer.%' ORDER BY _Index"
        )
    )
    process_rows = [
        row
        for row in marker_rows
        if row["message"].startswith("pra.fx_process.")
    ]
    layer_rows = [
        row
        for row in marker_rows
        if row["message"].startswith("pra.layer.")
    ]
    if not process_rows:
        raise RuntimeError("zero pra.fx_process HIPTX events")
    actual_counts = Counter(row["message"] for row in process_rows)
    missing = sorted(expected_names - set(actual_counts))
    extra = sorted(set(actual_counts) - expected_names)
    duplicate = {
        name: count for name, count in actual_counts.items() if count != 1
    }
    if missing or extra or duplicate:
        raise RuntimeError(
            f"process range identity failure: missing={missing}, "
            f"extra={extra}, duplicate={duplicate}"
        )

    process_by_name = {row["message"]: row for row in process_rows}
    layer_by_name: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in layer_rows:
        layer_by_name[row["message"]].append(row)

    nested_count = 0
    runtime_rows_total = 0
    hipops_rows_total = 0
    ranges_with_runtime = 0
    ranges_with_hipops = 0
    for inventory_row in inventory:
        process = process_by_name[inventory_row["nvtx_range_name"]]
        parents = layer_by_name[inventory_row["range_parent"]]
        if len(parents) != 1:
            raise RuntimeError(
                f"expected one parent {inventory_row['range_parent']}, "
                f"observed {len(parents)}"
            )
        parent = parents[0]
        if process["pid"] != parent["pid"] or process["tid"] != parent["tid"]:
            raise RuntimeError("process and parent pid/tid differ")
        if not (
            parent["BeginNs"] <= process["BeginNs"]
            and process["EndNs"] <= parent["EndNs"]
            and parent["begin_Index"] <= process["begin_Index"]
            and process["end_Index"] <= parent["end_Index"]
        ):
            raise RuntimeError(
                f"process range not nested under intended parent: "
                f"{inventory_row['nvtx_range_name']}"
            )
        nested_count += 1

        runtime_rows = list(
            connection.execute(
                f"SELECT _Index FROM {quoted(hip_table)} "
                "WHERE pid=? AND tid=? AND BeginNs>=? AND BeginNs<=? "
                "AND _Index>=? AND _Index<=?",
                (
                    process["pid"],
                    process["tid"],
                    process["BeginNs"],
                    process["EndNs"],
                    process["begin_Index"],
                    process["end_Index"],
                ),
            )
        )
        runtime_indices = {row["_Index"] for row in runtime_rows}
        runtime_rows_total += len(runtime_rows)
        if runtime_rows:
            ranges_with_runtime += 1
        if runtime_indices:
            placeholders = ",".join("?" for _ in runtime_indices)
            hipops_count = connection.execute(
                f"SELECT COUNT(*) FROM {quoted(hipops_table)} "
                f"WHERE _Index IN ({placeholders})",
                tuple(runtime_indices),
            ).fetchone()[0]
        else:
            hipops_count = 0
        hipops_rows_total += hipops_count
        if hipops_count:
            ranges_with_hipops += 1
    connection.close()

    audit = {
        "schema_version": 1,
        "runtime_goal": "R02",
        "status": "pass",
        "evidence_boundary": (
            "Instrumentation validation only; no process kernel durations "
            "were calculated."
        ),
        "inputs": {
            str(path.resolve()): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        },
        "counts": {
            "process_hiptx_rows": len(process_rows),
            "unique_process_messages": len(actual_counts),
            "inventory_rows": len(inventory),
            "nested_process_rows": nested_count,
            "layer_hiptx_rows": len(layer_rows),
            "selected_events": len(selected_event_ids),
            "missing_process_messages": 0,
            "extra_process_messages": 0,
            "duplicate_process_messages": 0,
        },
        "range_identity": {
            "namespace": "pra.fx_process.",
            "unique_join": True,
            "every_expected_range_once": True,
            "every_range_under_intended_parent": True,
            "fragment_identity_in_range_name": True,
            "aggregation_keys_parent_local": True,
        },
        "disabled_behavior": {
            "process_profile": disabled["process_profile"],
            "expected_process_range_count": disabled[
                "expected_process_range_count"
            ],
            "process_enabled_matches_disabled_output": True,
            "process_enabled_matches_r01_output": True,
            "exact_comparable_result": result_process,
        },
        "same_input": {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract["contract_sha256"],
            "prompt": metadata["same_input"],
            "sampling": metadata["sampling"],
            "warmup_iters": metadata["warmup_iters"],
            "max_new_tokens": metadata["max_new_tokens"],
            "measured_outputs_exactly_equivalent": True,
        },
        "strict_launch_ownership_schema": {
            "hiptx_table": hiptx_table,
            "hiptx_columns": hiptx_columns,
            "hip_runtime_table": hip_table,
            "hip_runtime_columns": hip_columns,
            "hipops_table": hipops_table,
            "hipops_columns": hipops_columns,
            "correlation_identity": "_Index",
            "runtime_rows_inside_process_ranges": runtime_rows_total,
            "process_ranges_with_runtime_rows": ranges_with_runtime,
            "hipops_rows_joined_by_runtime_index": hipops_rows_total,
            "process_ranges_with_joined_hipops": ranges_with_hipops,
            "ownership_rule": (
                "process HIPTX range -> HIP Runtime host BeginNs inside range "
                "and runtime _Index inside marker bounds -> HIPOPS identical "
                "runtime _Index"
            ),
            "kernel_durations_calculated": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(args.output),
                "process_hiptx_rows": len(process_rows),
                "inventory_rows": len(inventory),
                "nested_process_rows": nested_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
