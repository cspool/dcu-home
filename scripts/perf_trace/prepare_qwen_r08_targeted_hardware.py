#!/usr/bin/env python3
"""Freeze the same-lineage R08 bounded PMC capture contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("pmc", "pmc-read", "pmc-write")


class PlanError(RuntimeError):
    """Fail-closed R08 planning error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise PlanError(f"expected non-empty JSON object: {path}")
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
        raise PlanError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_lines_x(path: Path, values: list[str]) -> None:
    if not values or any(not value or "\n" in value or "\r" in value for value in values):
        raise PlanError(f"invalid newline target file payload: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(values) + "\n")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verified_ref(record: dict[str, Any], runtime_root: Path, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or not is_under(path, runtime_root):
        raise PlanError(f"{label} is missing or outside this runtime tree: {path}")
    observed = sha256_file(path)
    if observed != record.get("sha256"):
        raise PlanError(f"{label} SHA-256 drift: {observed}")
    return path


def file_record(path: Path, *, role: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "size_bytes": path.stat().st_size,
    }
    if role:
        result["role"] = role
    return result


def git_tracking_state(source_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(source_root.resolve())
    result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--error-unmatch", str(relative)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return "tracked" if result.returncode == 0 else "untracked_frozen_stage_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze fresh R08 targeted PMC work.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--r06-handoff", type=Path, required=True)
    parser.add_argument("--r07-handoff", type=Path, required=True)
    parser.add_argument("--selection-batch-id", required=True)
    parser.add_argument("--workflow05-policy-version", required=True)
    parser.add_argument("--maximum-targeted-pmc-family-count", type=int, required=True)
    parser.add_argument("--maximum-profiling-wall-time-seconds", type=float, required=True)
    parser.add_argument("--maximum-trace-bundle-bytes", type=int, required=True)
    parser.add_argument("--minimum-match-rate", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_under(source_root, project_root) or not is_under(runtime_root, project_root):
        raise PlanError("source/runtime roots must remain under project_root")
    if not is_under(artifact_root, runtime_root):
        raise PlanError("R08 artifact root must remain under this runtime tree")
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise PlanError(f"refusing non-empty R08 artifact root: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    if args.maximum_targeted_pmc_family_count <= 0:
        raise PlanError("maximum targeted family count must be positive")
    if not 0 < args.minimum_match_rate <= 1:
        raise PlanError("minimum match rate must be in (0, 1]")

    r06_path = args.r06_handoff.resolve()
    r07_path = args.r07_handoff.resolve()
    for path in (r06_path, r07_path):
        if not path.is_file() or not is_under(path, runtime_root):
            raise PlanError(f"handoff is outside the current runtime: {path}")
    r06 = load_json(r06_path)
    r07 = load_json(r07_path)
    for goal, handoff in (("R06", r06), ("R07", r07)):
        if (
            handoff.get("runtime_goal") != goal
            or handoff.get("status") != "complete"
            or handoff.get("execution_status") != "complete"
            or handoff.get("evidence_status") != "complete"
            or handoff.get("evidence_acquisition_mode")
            != "fresh_no_prior_runtime_reuse"
            or handoff.get("branch") != args.branch
            or handoff.get("run_id") != args.run_id
        ):
            raise PlanError(f"{goal} handoff is not a compatible complete input")

    lineage_path = verified_ref(
        r06["fresh_e2e_evidence"]["fresh_run_lineage_manifest"],
        runtime_root,
        "R06 lineage",
    )
    target_manifest_path = verified_ref(
        r06["fresh_e2e_evidence"]["full_request_target_manifest"],
        runtime_root,
        "R06 target manifest",
    )
    bounded_plan_path = verified_ref(
        r06["primary_outputs"]["bounded_hardware_plan"],
        runtime_root,
        "R06 bounded hardware plan",
    )
    metadata_path = verified_ref(
        r07["fresh_e2e_evidence"]["full_request_profile_metadata"],
        runtime_root,
        "R07 request metadata",
    )
    process_path = verified_ref(
        r07["observed_process_timeline"]["process_performance"],
        runtime_root,
        "R07 process performance",
    )
    gpu_path = verified_ref(
        r07["observed_process_timeline"]["process_gpu_timeline"],
        runtime_root,
        "R07 process GPU timeline",
    )
    ownership_path = verified_ref(
        r07["observed_process_timeline"]["strict_ownership"],
        runtime_root,
        "R07 strict ownership",
    )
    adapter_path = verified_ref(
        r07["fresh_e2e_evidence"]["fresh_run_dependency_adapter"],
        runtime_root,
        "R07 dependency adapter",
    )
    r07_source_path = verified_ref(
        r07["fresh_e2e_evidence"]["source_lineage"],
        runtime_root,
        "R07 source lineage",
    )
    lineage = load_json(lineage_path)
    metadata = load_json(metadata_path)
    adapter = load_json(adapter_path)
    r07_source = load_json(r07_source_path)
    lineage_id = lineage.get("lineage_id")
    if (
        lineage.get("status") != "PASS"
        or lineage.get("evidence_source_policy") != "current_run_only"
        or lineage.get("source_change_policy")
        != "stage_trace_instrumentation_allowed"
        or lineage.get("source_hash_equality_required") is not False
        or lineage_id != r06["fresh_e2e_evidence"].get("lineage_id")
        or lineage_id != r07["fresh_e2e_evidence"].get("lineage_id")
        or lineage_id != adapter.get("lineage_id")
    ):
        raise PlanError("R06/R07 fresh-run lineage mismatch")

    contract_path = verified_ref(
        r07["same_input_parent"]["contract"], runtime_root, "semantic contract"
    )
    contract = load_json(contract_path)
    contract_id = contract.get("contract_id")
    contract_sha = contract.get("contract_sha256")
    if (
        contract_id != metadata.get("contract_id")
        or contract_sha != metadata.get("contract_sha256")
        or contract_id != adapter.get("contract_id")
        or contract_sha != adapter.get("contract_sha256")
    ):
        raise PlanError("semantic contract identity drift")
    if (
        metadata.get("lineage_id") != lineage_id
        or metadata.get("runtime", {}).get("HIP_VISIBLE_DEVICES") != "1"
        or metadata.get("runtime", {}).get("CUDA_VISIBLE_DEVICES") != "1"
        or contract.get("device", {}).get("physical_device_id") != 1
        or contract.get("device", {}).get("unique_id") != "TS5V0409030401"
    ):
        raise PlanError("R07 metadata/device is not the required physical DCU 1 lineage")

    inventory_record = r07_source.get("inputs", {}).get("r02_process_inventory", {})
    inventory_path = verified_ref(inventory_record, runtime_root, "R02 process inventory")
    inventory_rows = read_csv(inventory_path)
    inventory_by_range = {row["nvtx_range_name"]: row for row in inventory_rows}
    if len(inventory_by_range) != len(inventory_rows):
        raise PlanError("R02 inventory contains duplicate range identities")

    process_rows = read_csv(process_path)
    process_by_range = {row["process_range"]: row for row in process_rows}
    if len(process_by_range) != len(process_rows):
        raise PlanError("R07 process table contains duplicate range identities")
    gpu_rows = read_csv(gpu_path)
    gpu_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in gpu_rows:
        key = (
            row["event_id"],
            row["stage"],
            row["matched_kernel_family"],
            row["process_range"],
        )
        gpu_by_key.setdefault(key, []).append(row)

    source_plan = [
        row for row in read_csv(bounded_plan_path)
        if row.get("selected", "").strip().lower() in {"1", "true", "yes"}
    ]
    if not source_plan or len(source_plan) > args.maximum_targeted_pmc_family_count:
        raise PlanError("R06 selected family count violates the R08 cap")
    if len({row["hardware_family_key"] for row in source_plan}) != len(source_plan):
        raise PlanError("R06 bounded plan contains duplicate family keys")
    if len({row["capture_batch_id"] for row in source_plan}) != len(source_plan):
        raise PlanError("R06 bounded plan contains duplicate capture batch IDs")
    source_plan.sort(key=lambda row: int(row["selection_rank"]))

    targeted_rows: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    planning_root = artifact_root / "planning" / "batches"
    for source_row in source_plan:
        event_id = source_row["representative_event_id"]
        stage = source_row["process_stage"]
        family = source_row["matched_kernel_family"]
        marker = source_row["representative_process_range"]
        batch_id = source_row["capture_batch_id"]
        r06_literal = source_row["kernel_name_filter_literal"]
        if not r06_literal or r06_literal.startswith("-") or "\n" in r06_literal or "\r" in r06_literal:
            raise PlanError(f"unsafe literal filter in batch {batch_id}")
        inventory_row = inventory_by_range.get(marker)
        process_row = process_by_range.get(marker)
        key = (event_id, stage, family, marker)
        selected_gpu = gpu_by_key.get(key, [])
        if inventory_row is None or process_row is None or not selected_gpu:
            raise PlanError(f"R06 family is absent from the R07 identity: {key}")
        if all(r06_literal in row["kernel_name"] for row in selected_gpu):
            literal = r06_literal
            filter_resolution = "r06_literal_covers_all_r07_selected_family_instances"
        else:
            literal = "_"
            filter_resolution = (
                "r07_plan_bounded_superset_fallback_for_complete_family_coverage"
            )
            if not all(literal in row["kernel_name"] for row in selected_gpu):
                raise PlanError(
                    f"no single audited fallback literal covers the R07 family: {key}"
                )
        selected_gpu.sort(key=lambda row: int(row["kernel_launch_order_in_process"]))
        row = {
            "measurement_contract_id": contract_id,
            "measurement_contract_sha256": contract_sha,
            "selection_batch_id": batch_id,
            "contract_relation": "current_measurement",
            "collection_required": "true",
            "targeted_eligible": "true",
            "expected_no_kernel": "false",
            "capture_batch_id": batch_id,
            "selection_rank": int(source_row["selection_rank"]),
            "selection_group_id": args.selection_batch_id,
            "hardware_family_key": source_row["hardware_family_key"],
            "event_id": event_id,
            "stage": stage,
            "hiptx_range": marker,
            "process_range": marker,
            "process_id": inventory_row["process_id"],
            "fragment_id": inventory_row.get("fragment_id", ""),
            "aggregation_key": inventory_row["aggregation_key"],
            "matched_kernel_family": family,
            "r06_kernel_name_filter_literal": r06_literal,
            "kernel_name_filter_literal": literal,
            "kernel_filter_resolution": filter_resolution,
            "one_literal_kernel_name_filter_per_capture_batch": "true",
            "r07_kernel_instance_count": len(selected_gpu),
            "r07_kernel_duration_ms": format(
                sum(float(item["kernel_duration_ms"]) for item in selected_gpu), ".15f"
            ),
            "r07_kernel_names_json": json.dumps(
                [item["kernel_name"] for item in selected_gpu],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "r07_runtime_indices_json": json.dumps(
                [int(item["runtime_index"]) for item in selected_gpu],
                separators=(",", ":"),
            ),
            "r07_process_hiptx_cpu_ms": process_row["hiptx_cpu_ms"],
            "latency_timing_source": "R07_non_replay_same_request",
            "pmc_replay_duration_is_latency_evidence": "false",
            "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
        }
        targeted_rows.append(row)

        batch_root = planning_root / batch_id
        selection_path = batch_root / "selection_plan.csv"
        event_target_path = batch_root / "process_targets.txt"
        range_target_path = batch_root / "process_range_targets.txt"
        batch_inventory_path = batch_root / "process_inventory.csv"
        expected_path = batch_root / "expected_family_order.json"
        write_csv_x(selection_path, [row])
        write_lines_x(event_target_path, [event_id])
        write_lines_x(range_target_path, [marker])
        write_csv_x(batch_inventory_path, [inventory_row])
        write_json_x(
            expected_path,
            {
                "schema_version": 1,
                "lineage_id": lineage_id,
                "capture_batch_id": batch_id,
                "kernel_name_filter_literal": literal,
                "r06_kernel_name_filter_literal": r06_literal,
                "kernel_filter_resolution": filter_resolution,
                "family_identity": {
                    "hardware_family_key": source_row["hardware_family_key"],
                    "event_id": event_id,
                    "stage": stage,
                    "process_range": marker,
                    "matched_kernel_family": family,
                },
                "r07_non_replay_kernel_order": [
                    {
                        "kernel_launch_order_in_process": int(
                            item["kernel_launch_order_in_process"]
                        ),
                        "kernel_name": item["kernel_name"],
                    }
                    for item in selected_gpu
                ],
            },
        )
        captures = []
        for mode in MODES:
            output_dir = artifact_root / "raw" / batch_id / mode / "attempt-001"
            tag = f"r08_{int(source_row['selection_rank']):02d}_{mode.replace('-', '_')}_{args.run_id}"
            captures.append(
                {
                    "capture_id": f"{batch_id}:{mode}:attempt-001",
                    "capture_batch_id": batch_id,
                    "selection_rank": int(source_row["selection_rank"]),
                    "mode": mode,
                    "attempt": 1,
                    "tag": tag,
                    "kernel_name_filter_literal": literal,
                    "output_dir": str(output_dir),
                    "selection_plan": file_record(selection_path),
                    "process_targets": file_record(event_target_path),
                    "process_range_targets": file_record(range_target_path),
                    "process_inventory": file_record(batch_inventory_path),
                    "expected_family_order": file_record(expected_path),
                }
            )
        batch_records.append(
            {
                "capture_batch_id": batch_id,
                "selection_rank": int(source_row["selection_rank"]),
                "hardware_family_key": source_row["hardware_family_key"],
                "kernel_name_filter_literal": literal,
                "r06_kernel_name_filter_literal": r06_literal,
                "kernel_filter_resolution": filter_resolution,
                "one_literal_filter": True,
                "capture_count": len(captures),
                "captures": captures,
            }
        )

    targeted_csv = artifact_root / "targeted_family_plan.csv"
    write_csv_x(targeted_csv, targeted_rows)
    targeted_json = artifact_root / "targeted_family_plan.json"
    write_json_x(
        targeted_json,
        {
            "schema_version": 1,
            "status": "ready",
            "lineage_id": lineage_id,
            "selection_batch_id": args.selection_batch_id,
            "source_plan": file_record(bounded_plan_path),
            "selected_family_count": len(targeted_rows),
            "capture_batch_count": len(batch_records),
            "capture_count": len(batch_records) * len(MODES),
            "unique_literal_filter_count": len(
                {row["kernel_name_filter_literal"] for row in targeted_rows}
            ),
            "modes": list(MODES),
            "minimum_name_order_match_rate": args.minimum_match_rate,
            "one_literal_kernel_name_filter_per_capture_batch": True,
            "pmc_collection_policy": "bounded_family_superset_exact_post_attribution",
            "latency_axis": "R07_non_replay_same_request_only",
            "replay_duration_is_latency_evidence": False,
            "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
            "targeted_family_csv": file_record(targeted_csv),
            "batches": batch_records,
        },
    )

    tool_paths = {
        "planner": Path(__file__).resolve(),
        "capture_launcher": source_root / "scripts/perf_trace/run_qwen_hardware_profile_single_request.sh",
        "profile_entry": source_root / "scripts/perf_trace/profile_qwen_same_input_layer.py",
        "trace_analyzer": source_root / "scripts/perf_trace/analyze_qwen_hipprof_process_trace.py",
        "pmc_analyzer": source_root / "scripts/perf_trace/analyze_qwen_hipprof_pmc.py",
        "capability_probe": source_root / "scripts/perf_trace/probe_dcu_capabilities.py",
        "consolidator": source_root / "scripts/perf_trace/consolidate_qwen_r08_targeted_hardware.py",
        "model_builder": source_root / "scripts/perf_trace/build_traffic_resource_model.py",
        "completion_auditor": source_root / "scripts/perf_trace/audit_qwen_r08_targeted_hardware.py",
        "handoff_writer": source_root / "scripts/perf_trace/finalize_qwen_r08_handoff.py",
        "capture_driver": source_root / "scripts/perf_trace/run_qwen_r08_targeted_capture.py",
        "hipprof": Path("/opt/dtk/bin/hipprof").resolve(),
        "cxxfilt": Path("/usr/bin/c++filt").resolve(),
    }
    for role, path in tool_paths.items():
        if not path.is_file():
            raise PlanError(f"missing current R08 tool {role}: {path}")
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_branch = subprocess.check_output(
        ["git", "-C", str(source_root), "branch", "--show-current"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "-z"]
    )
    tools = {}
    for role, path in tool_paths.items():
        record = file_record(path, role=role)
        if is_under(path, source_root):
            record["git_tracking_state"] = git_tracking_state(source_root, path)
        else:
            record["git_tracking_state"] = "external_system_tool"
        tools[role] = record

    source_lineage_path = artifact_root / "R08_SOURCE_LINEAGE.json"
    source_lineage = {
        "schema_version": 1,
        "status": "frozen",
        "runtime_goal": "R08",
        "lineage_id": lineage_id,
        "stage_source_revision": (
            source_revision
            + "+r08trace."
            + ".".join(record["sha256"][:12] for record in tools.values())
        ),
        "source_change_policy": "stage_trace_instrumentation_allowed",
        "source_hash_equality_required": False,
        "model_input_sampling_device_semantics_changed": False,
        "r07_process_family_identity_changed": False,
        "untracked_substitution_policy": (
            "all current stage tools are frozen by absolute path and SHA-256; "
            "the completion audit rejects any post-freeze drift"
        ),
        "source_root": str(source_root),
        "git_revision": source_revision,
        "git_branch": source_branch,
        "git_status_porcelain_v1_z_sha256": hashlib.sha256(source_status).hexdigest(),
        "tools": tools,
        "inputs": {
            "r06_handoff": file_record(r06_path),
            "r07_handoff": file_record(r07_path),
            "r06_lineage": file_record(lineage_path),
            "r06_target_manifest": file_record(target_manifest_path),
            "r06_bounded_hardware_plan": file_record(bounded_plan_path),
            "r07_metadata": file_record(metadata_path),
            "r07_process_performance": file_record(process_path),
            "r07_process_gpu_timeline": file_record(gpu_path),
            "r07_strict_ownership": file_record(ownership_path),
            "r07_dependency_adapter": file_record(adapter_path),
            "r02_process_inventory": file_record(inventory_path),
            "semantic_contract": file_record(contract_path),
        },
    }
    write_json_x(source_lineage_path, source_lineage)

    capture_manifest_path = artifact_root / "capture_manifest.json"
    all_captures = [capture for batch in batch_records for capture in batch["captures"]]
    write_json_x(
        capture_manifest_path,
        {
            "schema_version": 1,
            "status": "ready",
            "lineage_id": lineage_id,
            "serial_gpu_collection_required": True,
            "physical_device_id": 1,
            "capture_count": len(all_captures),
            "captures": all_captures,
        },
    )

    model = contract.get("model", {})
    tunable_path = source_root / "vllm/platforms/tunable_profiles/gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
    run_contract_path = artifact_root / "R08_RUN_CONTRACT.json"
    write_json_x(
        run_contract_path,
        {
            "schema_version": 1,
            "status": "ready",
            "runtime_goal": "R08",
            "branch": args.branch,
            "run_id": args.run_id,
            "workflow05_policy_version": args.workflow05_policy_version,
            "evidence_acquisition_mode": "fresh_no_prior_runtime_reuse",
            "lineage_id": lineage_id,
            "contract_id": contract_id,
            "contract_sha256": contract_sha,
            "contract_path": str(contract_path),
            "source_root": str(source_root),
            "model_root": model.get("resolved_model_root"),
            "served_model_name": model.get("served_model_name"),
            "physical_device_id": 1,
            "device_unique_id": contract["device"]["unique_id"],
            "expected_measured_result": metadata["measured_result"],
            "expected_same_input": metadata["same_input"],
            "expected_sampling": metadata["sampling"],
            "expected_max_new_tokens": metadata["max_new_tokens"],
            "expected_warmup_iters": metadata["warmup_iters"],
            "current_tunable_profile": file_record(tunable_path),
            "source_lineage": file_record(source_lineage_path),
            "targeted_family_plan": file_record(targeted_json),
            "targeted_family_plan_csv": file_record(targeted_csv),
            "capture_manifest": file_record(capture_manifest_path),
            "minimum_name_order_match_rate": args.minimum_match_rate,
            "maximum_targeted_pmc_family_count": args.maximum_targeted_pmc_family_count,
            "maximum_profiling_wall_time_seconds": args.maximum_profiling_wall_time_seconds,
            "maximum_trace_bundle_bytes": args.maximum_trace_bundle_bytes,
            "pmc_collection_policy": "bounded_family_superset_exact_post_attribution",
            "collector_side_process_window_filter_required": False,
            "final_process_family_hardware_attribution_required": True,
            "latency_axis": "R07_non_replay_same_request_only",
            "pmc_replay_duration_is_latency_evidence": False,
            "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "lineage_id": lineage_id,
                "selected_family_count": len(targeted_rows),
                "capture_count": len(all_captures),
                "run_contract": str(run_contract_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
