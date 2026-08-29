#!/usr/bin/env python3
"""Freeze an auditable R08 recovery after an observed literal-filter failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("pmc", "pmc-read", "pmc-write")


class RecoveryError(RuntimeError):
    """Fail-closed R08 recovery planning error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RecoveryError(f"expected non-empty JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RecoveryError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    result = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    result.update(extra)
    return result


def require_record(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RecoveryError(f"{label} is missing or changed: {path}")
    return path


def require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RecoveryError(f"{label} is outside the R08 artifact root: {path}") from exc


def git_tracking_state(source_root: Path, path: Path) -> str:
    if not path.resolve().is_relative_to(source_root.resolve()):
        return "external_system_tool"
    relative = path.resolve().relative_to(source_root.resolve())
    result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--error-unmatch", str(relative)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return "tracked" if result.returncode == 0 else "untracked_frozen_stage_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze failed-filter R08 recovery.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--recovery-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve()
    recovery_id = args.recovery_id
    if not recovery_id or "/" in recovery_id or ".." in recovery_id:
        raise RecoveryError("unsafe recovery ID")

    initial_contract_path = artifact_root / "R08_RUN_CONTRACT.json"
    initial_source_path = artifact_root / "R08_SOURCE_LINEAGE.json"
    initial_manifest_path = artifact_root / "capture_manifest.json"
    initial_plan_path = artifact_root / "targeted_family_plan.csv"
    initial_plan_json_path = artifact_root / "targeted_family_plan.json"
    progress_path = artifact_root / "capture_progress.json"
    for path in (
        initial_contract_path,
        initial_source_path,
        initial_manifest_path,
        initial_plan_path,
        initial_plan_json_path,
        progress_path,
    ):
        if not path.is_file():
            raise RecoveryError(f"missing initial R08 evidence: {path}")

    initial_contract = load_json(initial_contract_path)
    initial_source = load_json(initial_source_path)
    initial_manifest = load_json(initial_manifest_path)
    initial_plan_json = load_json(initial_plan_json_path)
    progress = load_json(progress_path)
    if (
        initial_contract.get("runtime_goal") != "R08"
        or initial_contract.get("status") != "ready"
        or initial_manifest.get("status") != "ready"
        or progress.get("status") != "failed"
        or initial_contract.get("lineage_id") != initial_source.get("lineage_id")
        or initial_contract.get("lineage_id") != initial_manifest.get("lineage_id")
        or progress.get("lineage_id") != initial_contract.get("lineage_id")
    ):
        raise RecoveryError("initial R08 contract/lineage/progress is not recoverable")
    if (
        sha256_file(initial_manifest_path)
        != initial_contract["capture_manifest"]["sha256"]
        or sha256_file(initial_plan_path)
        != initial_contract["targeted_family_plan_csv"]["sha256"]
        or sha256_file(initial_plan_json_path)
        != initial_contract["targeted_family_plan"]["sha256"]
        or sha256_file(initial_source_path)
        != initial_contract["source_lineage"]["sha256"]
    ):
        raise RecoveryError("initial frozen R08 planning artifacts drifted")

    captures = initial_manifest.get("captures", [])
    accepted = progress.get("captures", [])
    completed_count = len(accepted)
    if (
        int(progress.get("completed_capture_count", -1)) != completed_count
        or int(progress.get("capture_count", -1)) != len(captures)
        or not 0 <= completed_count < len(captures)
    ):
        raise RecoveryError("failed progress count is inconsistent")
    for record, planned in zip(accepted, captures[:completed_count], strict=True):
        if (
            record.get("status") != "complete"
            or record.get("capture_id") != planned.get("capture_id")
            or record.get("capture_batch_id") != planned.get("capture_batch_id")
            or record.get("mode") != planned.get("mode")
        ):
            raise RecoveryError("accepted capture prefix identity drift")
        for label in (
            "preflight",
            "launcher_preflight",
            "metadata",
            "runtime_events",
            "raw_db",
            "raw_pmc",
            "provenance",
            "driver_log",
            "trace_summary",
            "pmc_summary",
            "hardware_kernel_metrics",
            "discarded_superset_matches",
            "analysis_compaction_manifest",
        ):
            path = require_record(record.get(label, {}), f"accepted capture {label}")
            require_under(path, artifact_root, f"accepted capture {label}")

    failed = captures[completed_count]
    if progress.get("failed_capture_id") != failed.get("capture_id"):
        raise RecoveryError("failed capture does not equal the first pending capture")
    failed_root = Path(failed["output_dir"]).resolve()
    require_under(failed_root, artifact_root, "failed capture root")
    tag = failed["tag"]
    metadata_path = failed_root / f"{tag}.json"
    events_path = failed_root / f"{tag}.layer_events.runtime.jsonl"
    provenance_path = failed_root / "tool_provenance.txt"
    profile_exit_path = failed_root / "profile.exit_code"
    diagnostic_root = failed_root / "diagnostic_trace_analysis"
    diagnostic_summary_path = diagnostic_root / "process_trace_summary.json"
    diagnostic_ownership_path = diagnostic_root / "selected_strict_ownership.csv"
    raw_dbs = sorted((failed_root / "raw").glob("*.db"))
    raw_pmcs = sorted((failed_root / "raw").glob("*.txt"))
    if (
        len(raw_dbs) != 1
        or len(raw_pmcs) != 1
        or raw_pmcs[0].stat().st_size != 0
        or raw_dbs[0].stat().st_size <= 0
        or profile_exit_path.read_text(encoding="utf-8").strip() != "0"
    ):
        raise RecoveryError("failed capture is not the observed zero-PMC filter case")
    for path in (
        metadata_path,
        events_path,
        provenance_path,
        diagnostic_summary_path,
        diagnostic_ownership_path,
    ):
        if not path.is_file():
            raise RecoveryError(f"missing failed-attempt diagnostic evidence: {path}")
    metadata = load_json(metadata_path)
    diagnostic_summary = load_json(diagnostic_summary_path)
    if (
        metadata.get("measured_result") != initial_contract["expected_measured_result"]
        or diagnostic_summary.get("status") != "PASS"
    ):
        raise RecoveryError("failed attempt request or trace semantics are not intact")
    selected_plan_path = require_record(
        failed["selection_plan"], "failed capture selection plan"
    )
    selected_row = read_csv(selected_plan_path)[0]
    ownership_rows = read_csv(diagnostic_ownership_path)
    matching_target_rows = [
        row
        for row in ownership_rows
        if row.get("event_id") == selected_row["event_id"]
        and row.get("stage") == selected_row["stage"]
        and row.get("kernel_family") == selected_row["matched_kernel_family"]
        and row.get("kernel_name") in json.loads(selected_row["r07_kernel_names_json"])
    ]
    if len(matching_target_rows) != int(selected_row["r07_kernel_instance_count"]):
        raise RecoveryError("failed trace does not contain the exact planned target family")

    recovery_root = artifact_root / "recovery" / recovery_id
    recovery_root.mkdir(parents=True, exist_ok=False)
    failed_snapshot_path = recovery_root / "capture_progress.failed.json"
    shutil.copyfile(progress_path, failed_snapshot_path)

    initial_rows = read_csv(initial_plan_path)
    if len(initial_rows) != int(initial_plan_json["selected_family_count"]):
        raise RecoveryError("initial target plan count drift")
    amended_rows: list[dict[str, Any]] = []
    amended_batches: set[str] = set()
    amendment_records: list[dict[str, Any]] = []
    rows_by_batch: dict[str, dict[str, Any]] = {}
    for original in initial_rows:
        row: dict[str, Any] = dict(original)
        literal = original["kernel_name_filter_literal"]
        if literal.startswith("void "):
            names = json.loads(original["r07_kernel_names_json"])
            if not names or not all("_" in name for name in names):
                raise RecoveryError(
                    f"audited '_' superset does not cover {original['capture_batch_id']}"
                )
            row["superseded_kernel_name_filter_literal"] = literal
            row["kernel_name_filter_literal"] = "_"
            row["kernel_filter_resolution"] = (
                "r08_observed_complex_demangled_literal_zero_pmc_"
                "plan_bounded_superset_recovery"
            )
            row["recovery_id"] = recovery_id
            amended_batches.add(original["capture_batch_id"])
            amendment_records.append(
                {
                    "capture_batch_id": original["capture_batch_id"],
                    "selection_rank": int(original["selection_rank"]),
                    "original_literal": literal,
                    "effective_literal": "_",
                    "selected_family_unchanged": True,
                    "exact_post_attribution_required": True,
                }
            )
        amended_rows.append(row)
        rows_by_batch[original["capture_batch_id"]] = row
    if failed["capture_batch_id"] not in amended_batches:
        raise RecoveryError("failed literal is not in the complex-signature recovery class")

    targeted_csv_path = artifact_root / f"targeted_family_plan.{recovery_id}.csv"
    write_csv_x(targeted_csv_path, amended_rows)
    original_capture_by_batch = {
        capture["capture_batch_id"]: capture
        for capture in captures
        if capture["mode"] == "pmc"
    }
    planning_records: dict[str, dict[str, Any]] = {}
    for batch_id in amended_batches:
        original_capture = original_capture_by_batch[batch_id]
        batch_root = recovery_root / "planning" / "batches" / batch_id
        selection_path = batch_root / "selection_plan.csv"
        expected_path = batch_root / "expected_family_order.json"
        write_csv_x(selection_path, [rows_by_batch[batch_id]])
        expected = load_json(Path(original_capture["expected_family_order"]["path"]))
        expected["kernel_name_filter_literal"] = "_"
        expected["kernel_filter_resolution"] = rows_by_batch[batch_id][
            "kernel_filter_resolution"
        ]
        expected["recovery_id"] = recovery_id
        expected["superseded_kernel_name_filter_literal"] = rows_by_batch[batch_id][
            "superseded_kernel_name_filter_literal"
        ]
        write_json_x(expected_path, expected)
        planning_records[batch_id] = {
            "selection_plan": file_record(selection_path),
            "expected_family_order": file_record(expected_path),
        }

    recovery_captures: list[dict[str, Any]] = []
    for index, original_capture in enumerate(captures):
        capture = dict(original_capture)
        batch_id = capture["capture_batch_id"]
        if batch_id in amended_batches:
            capture["kernel_name_filter_literal"] = "_"
            capture["selection_plan"] = planning_records[batch_id]["selection_plan"]
            capture["expected_family_order"] = planning_records[batch_id][
                "expected_family_order"
            ]
        if batch_id == failed["capture_batch_id"] and index >= completed_count:
            capture["attempt"] = 2
            capture["capture_id"] = f"{batch_id}:{capture['mode']}:attempt-002"
            capture["output_dir"] = str(
                artifact_root / "raw" / batch_id / capture["mode"] / "attempt-002"
            )
            capture["tag"] = f"{capture['tag']}_{recovery_id.replace('-', '_')}"
        output_dir = Path(capture["output_dir"])
        if index >= completed_count and output_dir.exists() and any(output_dir.iterdir()):
            raise RecoveryError(f"recovery capture root is not empty: {output_dir}")
        recovery_captures.append(capture)

    batch_records = []
    for row in amended_rows:
        batch_id = row["capture_batch_id"]
        batch_captures = [
            capture for capture in recovery_captures
            if capture["capture_batch_id"] == batch_id
        ]
        if {capture["mode"] for capture in batch_captures} != set(MODES):
            raise RecoveryError(f"recovery mode coverage drift: {batch_id}")
        batch_records.append(
            {
                "capture_batch_id": batch_id,
                "selection_rank": int(row["selection_rank"]),
                "hardware_family_key": row["hardware_family_key"],
                "kernel_name_filter_literal": row["kernel_name_filter_literal"],
                "r06_kernel_name_filter_literal": row[
                    "r06_kernel_name_filter_literal"
                ],
                "kernel_filter_resolution": row["kernel_filter_resolution"],
                "one_literal_filter": True,
                "capture_count": 3,
                "captures": batch_captures,
            }
        )

    targeted_json_path = artifact_root / f"targeted_family_plan.{recovery_id}.json"
    write_json_x(
        targeted_json_path,
        {
            "schema_version": 1,
            "status": "ready",
            "lineage_id": initial_contract["lineage_id"],
            "recovery_id": recovery_id,
            "selection_batch_id": initial_plan_json["selection_batch_id"],
            "source_plan": initial_plan_json["source_plan"],
            "selected_family_count": len(amended_rows),
            "capture_batch_count": len(batch_records),
            "capture_count": len(recovery_captures),
            "unique_literal_filter_count": len(
                {row["kernel_name_filter_literal"] for row in amended_rows}
            ),
            "modes": list(MODES),
            "minimum_name_order_match_rate": initial_contract[
                "minimum_name_order_match_rate"
            ],
            "one_literal_kernel_name_filter_per_capture_batch": True,
            "pmc_collection_policy": (
                "bounded_family_superset_exact_post_attribution"
            ),
            "latency_axis": "R07_non_replay_same_request_only",
            "replay_duration_is_latency_evidence": False,
            "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
            "targeted_family_csv": file_record(targeted_csv_path),
            "recovery_amended_batch_count": len(amended_batches),
            "batches": batch_records,
        },
    )

    recovery_manifest_path = artifact_root / f"capture_manifest.{recovery_id}.json"
    write_json_x(
        recovery_manifest_path,
        {
            "schema_version": 1,
            "status": "ready",
            "lineage_id": initial_contract["lineage_id"],
            "recovery_id": recovery_id,
            "serial_gpu_collection_required": True,
            "physical_device_id": 1,
            "capture_count": len(recovery_captures),
            "completed_capture_prefix_count": completed_count,
            "superseded_failed_capture_id": failed["capture_id"],
            "captures": recovery_captures,
        },
    )

    initial_preflight = (
        artifact_root
        / "preflight"
        / (
            f"{completed_count + 1:03d}_{failed['capture_batch_id']}"
            f"_{failed['mode']}.json"
        )
    )
    initial_log = (
        artifact_root
        / "logs"
        / f"{completed_count + 1:03d}_{failed['tag']}.driver.log"
    )
    for path in (initial_preflight, initial_log):
        if not path.is_file():
            raise RecoveryError(f"missing failed-attempt driver evidence: {path}")
    recovery_evidence_path = artifact_root / "R08_CAPTURE_RECOVERY_001.json"
    write_json_x(
        recovery_evidence_path,
        {
            "schema_version": 1,
            "status": "ready",
            "runtime_goal": "R08",
            "recovery_id": recovery_id,
            "lineage_id": initial_contract["lineage_id"],
            "reason": (
                "hipprof exited zero and produced a valid fixed-input trace, but "
                "the full demangled C++ --kernel-name literal emitted zero PMC bytes"
            ),
            "failure_class": "collector_literal_representation_incompatibility",
            "scope_change": {
                "selected_family_count_changed": False,
                "selected_family_identity_changed": False,
                "capture_modes_changed": False,
                "effective_literal_changed_for_complex_demangled_signatures": True,
                "replacement_literal": "_",
                "policy": "plan_bounded_superset_exact_post_attribution",
            },
            "failed_attempt_is_success_evidence": False,
            "failed_capture_id": failed["capture_id"],
            "replacement_capture_id": recovery_captures[completed_count]["capture_id"],
            "completed_capture_prefix_count": completed_count,
            "failed_attempt": {
                "root": str(failed_root),
                "profile_exit_code": file_record(profile_exit_path),
                "driver_preflight": file_record(initial_preflight),
                "driver_log": file_record(initial_log),
                "metadata": file_record(metadata_path),
                "runtime_events": file_record(events_path),
                "provenance": file_record(provenance_path),
                "raw_db": file_record(raw_dbs[0]),
                "raw_pmc": file_record(raw_pmcs[0]),
                "diagnostic_trace_summary": file_record(diagnostic_summary_path),
                "diagnostic_selected_strict_ownership": file_record(
                    diagnostic_ownership_path
                ),
                "fixed_input_output_matches_r07": True,
                "target_family_present_in_trace": True,
                "raw_pmc_size_bytes": 0,
            },
            "failed_progress_snapshot": file_record(failed_snapshot_path),
            "literal_amendments": amendment_records,
            "initial_contract": file_record(initial_contract_path),
            "initial_source_lineage": file_record(initial_source_path),
            "initial_capture_manifest": file_record(initial_manifest_path),
            "initial_targeted_plan": file_record(initial_plan_path),
            "recovery_capture_manifest": file_record(recovery_manifest_path),
            "recovery_targeted_plan": file_record(targeted_csv_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    tool_paths = {
        "initial_planner": source_root
        / "scripts/perf_trace/prepare_qwen_r08_targeted_hardware.py",
        "capture_recovery_planner": Path(__file__).resolve(),
        "capture_launcher": source_root
        / "scripts/perf_trace/run_qwen_hardware_profile_single_request.sh",
        "profile_entry": source_root
        / "scripts/perf_trace/profile_qwen_same_input_layer.py",
        "trace_analyzer": source_root
        / "scripts/perf_trace/analyze_qwen_hipprof_process_trace.py",
        "pmc_analyzer": source_root / "scripts/perf_trace/analyze_qwen_hipprof_pmc.py",
        "capability_probe": source_root / "scripts/perf_trace/probe_dcu_capabilities.py",
        "consolidator": source_root
        / "scripts/perf_trace/consolidate_qwen_r08_targeted_hardware.py",
        "model_builder": source_root
        / "scripts/perf_trace/build_traffic_resource_model.py",
        "completion_auditor": source_root
        / "scripts/perf_trace/audit_qwen_r08_targeted_hardware.py",
        "handoff_writer": source_root
        / "scripts/perf_trace/finalize_qwen_r08_handoff.py",
        "capture_driver": source_root
        / "scripts/perf_trace/run_qwen_r08_targeted_capture.py",
        "hipprof": Path("/opt/dtk/bin/hipprof").resolve(),
        "cxxfilt": Path("/usr/bin/c++filt").resolve(),
    }
    tools: dict[str, dict[str, Any]] = {}
    for role, path in tool_paths.items():
        if not path.is_file():
            raise RecoveryError(f"missing recovery tool {role}: {path}")
        tools[role] = file_record(
            path,
            role=role,
            git_tracking_state=git_tracking_state(source_root, path),
        )
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_branch = subprocess.check_output(
        ["git", "-C", str(source_root), "branch", "--show-current"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "-z"]
    )
    final_source_path = artifact_root / "R08_SOURCE_LINEAGE_RECOVERY_001.json"
    final_inputs = dict(initial_source["inputs"])
    final_inputs.update(
        {
            "initial_r08_source_lineage": file_record(initial_source_path),
            "initial_r08_run_contract": file_record(initial_contract_path),
            "initial_r08_capture_manifest": file_record(initial_manifest_path),
            "initial_r08_targeted_plan": file_record(initial_plan_path),
            "failed_capture_progress_snapshot": file_record(failed_snapshot_path),
            "capture_recovery_evidence": file_record(recovery_evidence_path),
        }
    )
    write_json_x(
        final_source_path,
        {
            "schema_version": 1,
            "status": "frozen",
            "runtime_goal": "R08",
            "lineage_id": initial_contract["lineage_id"],
            "recovery_id": recovery_id,
            "stage_source_revision": (
                source_revision
                + "+r08recovery."
                + ".".join(record["sha256"][:12] for record in tools.values())
            ),
            "source_change_policy": "stage_trace_instrumentation_allowed",
            "source_hash_equality_required": False,
            "model_input_sampling_device_semantics_changed": False,
            "r07_process_family_identity_changed": False,
            "recovery_scope": (
                "capture-driver resume support and collector-literal representation "
                "adapter only; completed capture bytes and selected family identities "
                "remain unchanged"
            ),
            "source_root": str(source_root),
            "git_revision": source_revision,
            "git_branch": source_branch,
            "git_status_porcelain_v1_z_sha256": hashlib.sha256(
                source_status
            ).hexdigest(),
            "tools": tools,
            "inputs": final_inputs,
        },
    )

    final_contract_path = artifact_root / "R08_RUN_CONTRACT_RECOVERY_001.json"
    final_contract = dict(initial_contract)
    final_contract.update(
        {
            "status": "ready",
            "recovery_id": recovery_id,
            "source_lineage": file_record(final_source_path),
            "targeted_family_plan": file_record(targeted_json_path),
            "targeted_family_plan_csv": file_record(targeted_csv_path),
            "capture_manifest": file_record(recovery_manifest_path),
            "recovery_evidence": file_record(recovery_evidence_path),
            "initial_run_contract": file_record(initial_contract_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_x(final_contract_path, final_contract)
    print(
        json.dumps(
            {
                "status": "ready",
                "recovery_id": recovery_id,
                "completed_capture_prefix_count": completed_count,
                "amended_batch_count": len(amended_batches),
                "replacement_capture_id": recovery_captures[completed_count][
                    "capture_id"
                ],
                "run_contract": str(final_contract_path),
                "capture_manifest": str(recovery_manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
