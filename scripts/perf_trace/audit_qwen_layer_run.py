#!/usr/bin/env python3
"""Independently audit one Qwen SAME_INPUT layer-wise evidence directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_METRICS = {
    "hiptx_host_range_duration_ms",
    "hipprof_launch_owned_kernel_sum_ms",
    "hipprof_launch_owned_kernel_busy_union_ms",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value, f"not a nonempty object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(rows, f"CSV has no data rows: {path}")
    return rows


def _close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=tolerance)


def _key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["contract_id"]),
        int(row["forward_id"]),
        int(row["layer_idx"]),
        int(row["occurrence"]),
    )


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _connect_immutable(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    _require(not temporary.exists(), f"stale temporary output: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    contract_path = args.contract.resolve()
    output_path = args.output.resolve()
    _require(output_path.parent == artifact_dir, "audit output must be in artifact dir")
    _require(not output_path.exists(), f"refusing to overwrite audit: {output_path}")

    tag = args.tag
    paths = {
        "raw_db": artifact_dir / f"{tag}.hipprof.db",
        "raw_trace": artifact_dir / f"{tag}.hipprof.json",
        "raw_trace_merge_manifest": (
            artifact_dir / f"{tag}.raw_trace_merge_manifest.json"
        ),
        "runtime_events": artifact_dir / f"{tag}.layer_events.runtime.jsonl",
        "run_metadata": artifact_dir / f"{tag}.json",
        "generator_audit": artifact_dir / f"{tag}_audit.json",
        "layer_events": artifact_dir / f"{tag}_layer_events.csv",
        "layer_kernel_breakdown_csv": (
            artifact_dir / f"{tag}_layer_kernel_breakdown.csv"
        ),
        "layer_kernel_breakdown_json": (
            artifact_dir / f"{tag}_layer_kernel_breakdown.json"
        ),
        "all_input_layer_performance": (
            artifact_dir / f"{tag}_all_input_layer_performance.csv"
        ),
        "strict_ownership": artifact_dir / f"{tag}_strict_ownership.csv",
        "launch_order": artifact_dir / f"{tag}_layer_kernel_launch_order.csv",
        "stats_hip_api": artifact_dir / f"{tag}_stats_hip_api.csv",
        "stats_phase": artifact_dir / f"{tag}_stats_phase.csv",
        "normalized_db": artifact_dir / f"{tag}.sqlite",
        "report": (
            artifact_dir
            / "SAME_INPUT_QWEN3_5_27B_VLLM_PRA_LAYER_PERFORMANCE_REPORT.md"
        ),
        "tool_provenance": artifact_dir / f"{tag}.tool_provenance.txt",
        "initial_generator_log": artifact_dir / f"{tag}.generator.log",
        "recovery_generator_log": (
            artifact_dir / f"{tag}.generator_recovery.log"
        ),
    }
    optional_paths = {
        "raw_trace_merge_manifest",
        "recovery_generator_log",
    }
    for name, path in {"contract": contract_path, **paths}.items():
        if name in optional_paths:
            continue
        _require(path.is_file() and path.stat().st_size > 0, f"missing {name}: {path}")

    # Required evidence ordering: count every layer HIPTX range first.
    raw_connection = _connect_immutable(paths["raw_db"])
    hiptx_tables = [
        row[0]
        for row in raw_connection.execute(
            "select name from sqlite_master "
            "where type='table' and name like 'HIPTX_%' order by name"
        )
    ]
    _require(hiptx_tables, "raw DB has no HIPTX tables")
    first_queries: list[dict[str, Any]] = []
    first_count = 0
    for table in hiptx_tables:
        query = (
            f"SELECT COUNT(*) FROM {_quote(table)} "
            "WHERE message LIKE 'pra.layer.%'"
        )
        count = int(raw_connection.execute(query).fetchone()[0])
        first_queries.append({"table": table, "query": query, "count": count})
        first_count += count
    raw_connection.close()

    contract = _load_object(contract_path)
    contract_without_hash = dict(contract)
    recorded_contract_hash = str(contract_without_hash.pop("contract_sha256"))
    canonical_contract_hash = hashlib.sha256(
        json.dumps(
            contract_without_hash,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    _require(
        canonical_contract_hash == recorded_contract_hash,
        "frozen contract canonical SHA-256 mismatch",
    )
    _require(contract["runtime_goal"] == "R01", "contract runtime_goal mismatch")
    _require(contract["same_input"]["warmup_count"] == 1, "warmup contract mismatch")
    _require(
        contract["same_input"]["sampling"]["max_new_tokens"] == 32,
        "MAX_NEW_TOKENS contract mismatch",
    )
    _require(contract["model"]["num_hidden_layers"] == 64, "model layer mismatch")
    _require(
        contract["config"]["process_profile_assignment"]
        == "PRA_BACKEND_PERF_PROCESS_PROFILE=0",
        "process-profile-off contract mismatch",
    )

    run_metadata = _load_object(paths["run_metadata"])
    generator_audit = _load_object(paths["generator_audit"])
    merge_manifest = (
        _load_object(paths["raw_trace_merge_manifest"])
        if paths["raw_trace_merge_manifest"].is_file()
        else None
    )
    _require(run_metadata["status"] == "pass", "run metadata is not pass")
    _require(generator_audit["status"] == "pass", "generator audit is not pass")
    _require(run_metadata["tag"] == tag, "run tag mismatch")
    _require(run_metadata["contract_id"] == contract["contract_id"], "contract ID mismatch")
    _require(
        run_metadata["contract_sha256"] == recorded_contract_hash,
        "run contract hash mismatch",
    )
    _require(run_metadata["max_new_tokens"] == 32, "run MAX_NEW_TOKENS mismatch")
    _require(run_metadata["warmup_iters"] == 1, "run warmup count mismatch")
    _require(run_metadata["process_profile"] == "off", "process profiling is on")
    _require(run_metadata["expected_layers"] == 64, "run expected layer mismatch")
    _require(
        len(run_metadata["warmup_results"]) == 1
        and run_metadata["warmup_results"][0] == run_metadata["measured_result"],
        "warmup and measured output differ",
    )
    _require(
        run_metadata["model_root"] == contract["model"]["resolved_model_root"],
        "resolved model root mismatch",
    )

    core_hashes = {
        name: _sha256(paths[name])
        for name in ("raw_db", "raw_trace", "runtime_events")
    }
    _require(
        core_hashes["raw_db"] == generator_audit["raw_db_sha256"],
        "raw DB hash mismatch",
    )
    _require(
        core_hashes["raw_trace"] == generator_audit["raw_trace_sha256"],
        "raw trace hash mismatch",
    )
    _require(
        core_hashes["runtime_events"] == generator_audit["runtime_events_sha256"],
        "runtime event hash mismatch",
    )

    chunks: list[dict[str, Any]] = []
    if merge_manifest is not None:
        chunks = merge_manifest["chunks"]
        _require(
            isinstance(chunks, list) and len(chunks) >= 1,
            "segmented raw trace manifest has no chunks",
        )
        _require(
            merge_manifest["output"]["path"] == str(paths["raw_trace"]),
            "merged trace path mismatch",
        )
        _require(
            merge_manifest["output"]["sha256"] == core_hashes["raw_trace"],
            "merged trace manifest hash mismatch",
        )
        for expected_number, chunk in enumerate(chunks, 1):
            chunk_path = Path(chunk["path"]).resolve()
            _require(chunk_path.parent == artifact_dir, "trace chunk outside artifact dir")
            _require(chunk["chunk_number"] == expected_number, "chunk order mismatch")
            _require(
                chunk_path.stat().st_size == chunk["size_bytes"],
                f"trace chunk size mismatch: {chunk_path}",
            )
            _require(
                _sha256(chunk_path) == chunk["sha256"],
                f"trace chunk hash mismatch: {chunk_path}",
            )
            with chunk_path.open("rb") as stream:
                _require(
                    stream.read(16) == b'{"traceEvents":[',
                    f"trace chunk prefix mismatch: {chunk_path}",
                )
                stream.seek(-2, 2)
                _require(stream.read() == b"]}", f"trace chunk suffix mismatch: {chunk_path}")

    event_rows = _load_rows(paths["layer_events"])
    _require(len(event_rows) == first_count, "layer event count differs from raw HIPTX")
    event_by_key = {_key(row): row for row in event_rows}
    _require(len(event_by_key) == len(event_rows), "duplicate layer event key")
    occurrence_keys = {row["occurrence_key"] for row in event_rows}
    _require(len(occurrence_keys) == len(event_rows), "duplicate occurrence key")
    expected_layer_types = contract["model"]["layer_types"]
    forward_to_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        forward_to_rows[int(row["forward_id"])].append(row)
        layer = int(row["layer_idx"])
        _require(row["workload_type"] == expected_layer_types[layer], "layer type mismatch")
        _require(
            int(row["occurrence"]) == int(row["forward_id"]),
            "stable occurrence does not equal the 1-based forward ID",
        )
        _require(
            int(row["source_range_occurrence"]) == 0,
            "unexpected repeated source range occurrence",
        )
        _require(
            int(row["past_len"]) + int(row["q_len"]) == int(row["kv_len"]),
            "q_len/past_len/kv_len mismatch",
        )
        _require(row["attribution_status"] == "pass", "failed layer attribution")
        _require(int(row["launch_owned_kernel_count"]) > 0, "layer owns no kernels")
        _require(row["device_ids"] == "1", "unexpected strict-owned device ID")
        _require("_Index" in row["ownership_method"], "ownership method omits _Index")
    forward_ids = sorted(forward_to_rows)
    _require(forward_ids == list(range(1, len(forward_ids) + 1)), "forward IDs not contiguous")
    for forward_id, rows in forward_to_rows.items():
        _require(len(rows) == 64, f"forward {forward_id} does not have 64 rows")
        _require(
            {int(row["layer_idx"]) for row in rows} == set(range(64)),
            f"forward {forward_id} layer indices incomplete",
        )
        context = {
            (
                row["phase"],
                int(row["q_len"]),
                int(row["past_len"]),
                int(row["kv_len"]),
            )
            for row in rows
        }
        _require(len(context) == 1, f"forward {forward_id} context is inconsistent")
    prefill_rows = [rows[0] for rows in forward_to_rows.values() if rows[0]["phase"] == "prefill_chunk"]
    decode_rows = [rows[0] for rows in forward_to_rows.values() if rows[0]["phase"] == "decode"]
    _require(
        sum(int(row["q_len"]) for row in prefill_rows)
        == run_metadata["measured_result"]["prompt_token_count"],
        "prefill chunks do not cover the prompt",
    )
    _require(
        len(decode_rows) == run_metadata["measured_result"]["output_token_count"],
        "decode forward count differs from output token count",
    )

    breakdown_rows = _load_rows(paths["layer_kernel_breakdown_csv"])
    breakdown_by_key: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in breakdown_rows:
        breakdown_by_key[_key(row)].append(row)
    _require(set(breakdown_by_key) == set(event_by_key), "kernel breakdown join dropped rows")
    for key, rows in breakdown_by_key.items():
        event = event_by_key[key]
        _require(
            sum(int(row["kernel_count"]) for row in rows)
            == int(event["launch_owned_kernel_count"]),
            "breakdown kernel count mismatch",
        )
        _require(
            _close(
                sum(float(row["launch_owned_kernel_duration_ms"]) for row in rows),
                float(event["hipprof_launch_owned_kernel_sum_ms"]),
            ),
            "breakdown duration mismatch",
        )
        _require(
            _close(sum(float(row["pct_of_layer_launch_owned_kernel_sum"]) for row in rows), 100.0),
            "breakdown percentages do not total 100",
        )
    breakdown_json = json.loads(
        paths["layer_kernel_breakdown_json"].read_text(encoding="utf-8")
    )
    _require(
        isinstance(breakdown_json, list)
        and len(breakdown_json) == len(breakdown_rows),
        "breakdown CSV/JSON row count mismatch",
    )

    metric_rows = _load_rows(paths["all_input_layer_performance"])
    metric_by_key: dict[tuple[str, int, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metric_rows:
        key = _key(row)
        metric = row["metric"]
        _require(metric not in metric_by_key[key], "duplicate layer metric row")
        metric_by_key[key][metric] = row
    _require(set(metric_by_key) == set(event_by_key), "all-input table dropped layer keys")
    for key, metrics in metric_by_key.items():
        _require(set(metrics) == EXPECTED_METRICS, "layer metric set mismatch")
        event = event_by_key[key]
        for metric in EXPECTED_METRICS:
            _require(
                _close(
                    float(metrics[metric]["metric_value_ms"]),
                    float(event[metric]),
                ),
                f"metric value mismatch: {metric}",
            )
    _require(
        len(metric_rows) == len(event_rows) * len(EXPECTED_METRICS),
        "all-input metric row count mismatch",
    )

    ownership_rows = _load_rows(paths["strict_ownership"])
    ownership_by_key: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    kernel_ids: set[str] = set()
    for row in ownership_rows:
        ownership_by_key[_key(row)].append(row)
        _require(row["runtime_host_start_inside_range"] == "1", "host-start rule failed")
        _require(row["runtime_index_inside_marker_bounds"] == "1", "index-bounds rule failed")
        _require(int(row["device_id"]) == 1, "strict ownership includes another device")
        _require(int(row["kernel_duration_ns"]) > 0, "non-positive owned kernel duration")
        _require(row["kernel_id"] not in kernel_ids, "owned kernel assigned more than once")
        kernel_ids.add(row["kernel_id"])
    _require(set(ownership_by_key) == set(event_by_key), "strict ownership dropped layers")
    for key, rows in ownership_by_key.items():
        event = event_by_key[key]
        _require(
            len(rows) == int(event["launch_owned_kernel_count"]),
            "strict ownership kernel count mismatch",
        )
        _require(
            _close(
                sum(int(row["kernel_duration_ns"]) for row in rows) / 1e6,
                float(event["hipprof_launch_owned_kernel_sum_ms"]),
            ),
            "strict ownership duration sum mismatch",
        )

    normalized = _connect_immutable(paths["normalized_db"])
    normalized_counts = {
        table: int(normalized.execute(f"select count(*) from {_quote(table)}").fetchone()[0])
        for table in (
            "metadata",
            "layer_events",
            "strict_ownership",
            "layer_kernel_breakdown",
            "all_input_layer_performance",
        )
    }
    normalized_metadata = dict(normalized.execute("select key, value from metadata"))
    normalized.close()
    _require(normalized_counts["metadata"] == 10, "normalized metadata count mismatch")
    _require(normalized_counts["layer_events"] == len(event_rows), "normalized event count mismatch")
    _require(
        normalized_counts["strict_ownership"] == len(ownership_rows),
        "normalized ownership count mismatch",
    )
    _require(
        normalized_counts["layer_kernel_breakdown"] == len(breakdown_rows),
        "normalized breakdown count mismatch",
    )
    _require(
        normalized_counts["all_input_layer_performance"] == len(metric_rows),
        "normalized metric count mismatch",
    )
    _require(normalized_metadata["raw_db_sha256"] == core_hashes["raw_db"], "normalized raw DB hash mismatch")
    _require(
        normalized_metadata["raw_trace_sha256"] == core_hashes["raw_trace"],
        "normalized raw trace hash mismatch",
    )
    _require(
        normalized_metadata["runtime_events_sha256"] == core_hashes["runtime_events"],
        "normalized event hash mismatch",
    )
    _require(
        normalized_metadata["device_timestamp_overlap_attribution"] == "forbidden",
        "normalized trace permits device-overlap attribution",
    )

    report = paths["report"].read_text(encoding="utf-8")
    for required_text in ("_Index", "launch-owned", "all input-layer", "Status: **PASS**"):
        _require(required_text in report, f"report missing required text: {required_text}")
    _require(
        "does not establish strict\nprocess-wise timing" in report,
        "report evidence boundary is missing",
    )
    _require(
        '"status": "pass"'
        in paths["initial_generator_log"].read_text(encoding="utf-8"),
        "generator pass is not preserved",
    )
    if paths["recovery_generator_log"].is_file():
        _require(
            '"status": "pass"'
            in paths["recovery_generator_log"].read_text(encoding="utf-8"),
            "recovery generator pass is not preserved",
        )

    _require(
        first_count == int(run_metadata["observed_layer_events"]),
        "formal raw HIPTX count differs from this request metadata",
    )
    _require(
        len(forward_ids) == int(run_metadata["observed_forwards"]),
        "formal forward count differs from this request metadata",
    )
    _require(
        first_count == len(forward_ids) * 64,
        "formal raw HIPTX count is not current forwards times 64",
    )
    _require(generator_audit["first_check"]["count"] == first_count, "generator first count mismatch")
    _require(generator_audit["failed_ownership_ranges"] == [], "failed ownership ranges present")
    _require(generator_audit["missing_event_rows"] == [], "missing event rows present")
    _require(generator_audit["missing_range_rows"] == [], "missing range rows present")
    _require(generator_audit["process_marker_count"] == 0, "process markers present")
    _require(generator_audit["request_marker_count"] == 1, "request marker count mismatch")
    _require(not generator_audit["device_timestamp_overlap_attribution_used"], "device overlap was used")
    _require(
        not generator_audit["nested_total_attn_mlp_summed_as_independent_costs"],
        "nested totals were independently summed",
    )

    record = {
        "schema_version": 1,
        "status": "pass",
        "runtime_goal": "R01",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "audit_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "contract": {
            "path": str(contract_path),
            "contract_id": contract["contract_id"],
            "canonical_sha256": canonical_contract_hash,
        },
        "first_check": {
            "description": "count all layer HIPTX ranges in formal raw queryable DB",
            "database": str(paths["raw_db"]),
            "tables": hiptx_tables,
            "queries": first_queries,
            "count": first_count,
        },
        "coverage": {
            "forward_count": len(forward_ids),
            "layers_per_forward": 64,
            "layer_event_count": len(event_rows),
            "unique_occurrence_key_count": len(occurrence_keys),
            "prefill_forward_count": len(prefill_rows),
            "decode_forward_count": len(decode_rows),
            "all_input_layer_metric_rows": len(metric_rows),
            "metrics_per_layer_occurrence": sorted(EXPECTED_METRICS),
            "missing_event_rows": 0,
            "missing_range_rows": 0,
            "failed_joins": 0,
        },
        "strict_attribution": {
            "ownership_rule": generator_audit["ownership_rule"],
            "strict_ownership_rows": len(ownership_rows),
            "unique_launch_owned_kernels": len(kernel_ids),
            "device_ids": [1],
            "downstream_denominator_metric": (
                "hipprof_launch_owned_kernel_sum_ms"
            ),
            "device_timestamp_overlap_attribution_used": False,
            "nested_total_attn_mlp_summed_as_independent_costs": False,
        },
        "request": {
            "synchronized_latency_ms": run_metadata["request_synchronized_latency_ms"],
            "prompt_tokens": run_metadata["measured_result"]["prompt_token_count"],
            "output_tokens": run_metadata["measured_result"]["output_token_count"],
            "warmup_count": run_metadata["warmup_iters"],
            "warmup_and_measured_output_identical": True,
        },
        "same_run_binding": {
            "raw_db_sha256": core_hashes["raw_db"],
            "raw_trace_sha256": core_hashes["raw_trace"],
            "runtime_events_sha256": core_hashes["runtime_events"],
            "raw_trace_chunk_count": len(chunks),
            "normalized_sqlite_counts": normalized_counts,
        },
        "recovery_provenance": {
            "generator_status": "pass",
            "raw_trace_was_segmented": merge_manifest is not None,
            "raw_trace_chunks_preserved": bool(chunks),
            "lossless_merge_manifest": (
                str(paths["raw_trace_merge_manifest"])
                if merge_manifest is not None
                else None
            ),
            "recovery_generator_log": (
                str(paths["recovery_generator_log"])
                if paths["recovery_generator_log"].is_file()
                else None
            ),
            "same_raw_db_and_runtime_events_used": True,
            "additional_model_or_profiler_run_performed": False,
        },
        "evidence_boundary": (
            "Layer totals only; no strict process-wise timing and no split "
            "of layer totals among processes."
        ),
        "outputs": {
            name: str(path)
            for name, path in paths.items()
            if path.is_file()
        },
    }
    _atomic_json(output_path, record)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
