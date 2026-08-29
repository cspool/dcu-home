#!/usr/bin/env python3
"""Write the scheduler-assigned R08 handoff only after PASS completion audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"expected non-empty JSON object: {path}")
    return value


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def record(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "size_bytes": path.stat().st_size,
    }
    result.update(extra)
    return result


def artifact_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize fresh R08 handoff.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--handoff-output", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--device-capabilities", type=Path, required=True)
    parser.add_argument("--targeted-plan", type=Path, required=True)
    parser.add_argument("--hardware-metrics", type=Path, required=True)
    parser.add_argument("--hardware-family-metrics", type=Path, required=True)
    parser.add_argument("--hardware-coverage", type=Path, required=True)
    parser.add_argument("--traffic-resource-model", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    handoff_output = args.handoff_output.resolve()
    if not artifact_root.is_relative_to(runtime_root):
        raise RuntimeError("artifact root is outside the assigned runtime")
    expected_handoff = runtime_root / "handoffs" / "R08.json"
    if handoff_output != expected_handoff or handoff_output.exists():
        raise RuntimeError("refusing non-assigned or existing R08 handoff output")
    paths = [
        args.run_contract.resolve(),
        args.source_lineage.resolve(),
        args.capture_summary.resolve(),
        args.device_capabilities.resolve(),
        args.targeted_plan.resolve(),
        args.hardware_metrics.resolve(),
        args.hardware_family_metrics.resolve(),
        args.hardware_coverage.resolve(),
        args.traffic_resource_model.resolve(),
        args.completion_audit.resolve(),
    ]
    if any(not path.is_file() or not path.is_relative_to(runtime_root) for path in paths):
        raise RuntimeError("handoff input is missing or outside the assigned runtime")
    contract = load_json(args.run_contract.resolve())
    source = load_json(args.source_lineage.resolve())
    captures = load_json(args.capture_summary.resolve())
    capabilities = load_json(args.device_capabilities.resolve())
    coverage = load_json(args.hardware_coverage.resolve())
    model = load_json(args.traffic_resource_model.resolve())
    audit = load_json(args.completion_audit.resolve())
    lineage_id = contract.get("lineage_id")
    if (
        contract.get("runtime_goal") != "R08"
        or source.get("lineage_id") != lineage_id
        or captures.get("lineage_id") != lineage_id
        or coverage.get("lineage_id") != lineage_id
        or model.get("lineage_id") != lineage_id
        or audit.get("lineage_id") != lineage_id
        or audit.get("status") != "PASS"
        or audit.get("failure_checks") != []
        or capabilities.get("status") != "verified"
        or capabilities.get("physical_device_id") != 1
        or capabilities.get("architecture") != "gfx936"
        or coverage.get("status") != "PASS"
        or coverage.get("complete_selected_family_coverage") is not True
        or coverage.get("pmc_replay_duration_is_latency_evidence") is not False
        or model.get("status") != "complete"
        or model.get("model_type")
        != "fresh_run_fx_visible_traffic_and_dcu_family_resource"
        or model.get("traffic_boundary", {}).get("hbm_or_dram_traffic_claimed")
        is not False
        or model.get("resource_boundary", {}).get("achieved_occupancy_claimed")
        is not False
    ):
        raise RuntimeError("R08 completion conditions are not all satisfied")
    bytes_now = artifact_bytes(artifact_root)
    if (
        float(captures.get("profiling_wall_seconds", float("inf")))
        > float(contract["maximum_profiling_wall_time_seconds"])
        or bytes_now > int(contract["maximum_trace_bundle_bytes"])
    ):
        raise RuntimeError("R08 runtime budget exceeded before handoff")

    traffic_path = Path(model["outputs"]["traffic"]["path"])
    resource_path = Path(model["outputs"]["resource"]["path"])
    payload = {
        "schema_version": 1,
        "runtime_goal": "R08",
        "status": "complete",
        "execution_status": "complete",
        "evidence_status": "complete",
        "coverage_target_met": True,
        "next_authorization_required": False,
        "skill": "qwen-dcu-workflow05-targeted-hardware-gap-analysis",
        "branch": contract["branch"],
        "run_id": contract["run_id"],
        "workflow05_policy_version": contract["workflow05_policy_version"],
        "evidence_acquisition_mode": "fresh_no_prior_runtime_reuse",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime_root),
        "runtime_artifact_root": str(artifact_root),
        "handoff_output": str(handoff_output),
        "fresh_e2e_evidence": {
            "schema_version": 1,
            "status": "complete",
            "lineage_id": lineage_id,
            "device_capabilities": record(args.device_capabilities.resolve()),
            "traffic_resource_model": record(args.traffic_resource_model.resolve()),
            "source_lineage": record(args.source_lineage.resolve()),
            "hardware_coverage": record(args.hardware_coverage.resolve()),
            "completion_audit": record(args.completion_audit.resolve(), status="PASS"),
        },
        "collection": {
            "physical_device_id": 1,
            "logical_device_id": 0,
            "HIP_VISIBLE_DEVICES": "1",
            "CUDA_VISIBLE_DEVICES": "1",
            "selected_family_count": coverage["selected_family_count"],
            "capture_count": captures["completed_capture_count"],
            "capture_modes": ["pmc", "pmc-read", "pmc-write"],
            "one_literal_kernel_name_filter_per_capture_batch": True,
            "pmc_collection_policy": contract["pmc_collection_policy"],
            "minimum_name_order_match_rate_required": contract[
                "minimum_name_order_match_rate"
            ],
            "minimum_name_order_match_rate_observed": coverage[
                "minimum_name_order_match_rate_observed"
            ],
            "selected_exact_attribution_rate": coverage[
                "minimum_selected_exact_attribution_rate"
            ],
            "selected_ambiguity_count": coverage["selected_ambiguity_count"],
            "discarded_superset_match_count": coverage[
                "discarded_superset_match_count"
            ],
            "discarded_rows_audited": coverage["discarded_rows_audited"],
            "serial_gpu_collection": True,
        },
        "primary_outputs": {
            "run_contract": record(args.run_contract.resolve()),
            "source_lineage": record(args.source_lineage.resolve()),
            "targeted_family_plan": record(args.targeted_plan.resolve()),
            "capture_summary": record(args.capture_summary.resolve()),
            "device_capabilities": record(args.device_capabilities.resolve()),
            "hardware_metrics": record(
                args.hardware_metrics.resolve(), rows=csv_rows(args.hardware_metrics.resolve())
            ),
            "hardware_metrics_by_kernel_family": record(
                args.hardware_family_metrics.resolve(),
                rows=csv_rows(args.hardware_family_metrics.resolve()),
            ),
            "hardware_coverage": record(args.hardware_coverage.resolve(), status="PASS"),
            "process_traffic_model": record(traffic_path, rows=csv_rows(traffic_path)),
            "kernel_family_resource_model": record(
                resource_path, rows=csv_rows(resource_path)
            ),
            "traffic_resource_model": record(args.traffic_resource_model.resolve()),
            "completion_audit": record(args.completion_audit.resolve(), status="PASS"),
        },
        "same_run_binding": {
            "lineage_id": lineage_id,
            "contract_id": contract["contract_id"],
            "contract_sha256": contract["contract_sha256"],
            "r06_handoff_sha256": source["inputs"]["r06_handoff"]["sha256"],
            "r07_handoff_sha256": source["inputs"]["r07_handoff"]["sha256"],
            "r06_lineage_sha256": source["inputs"]["r06_lineage"]["sha256"],
            "r06_bounded_hardware_plan_sha256": source["inputs"][
                "r06_bounded_hardware_plan"
            ]["sha256"],
            "r07_metadata_sha256": source["inputs"]["r07_metadata"]["sha256"],
            "r07_process_performance_sha256": source["inputs"][
                "r07_process_performance"
            ]["sha256"],
            "r07_process_gpu_timeline_sha256": source["inputs"][
                "r07_process_gpu_timeline"
            ]["sha256"],
            "r07_dependency_adapter_sha256": source["inputs"][
                "r07_dependency_adapter"
            ]["sha256"],
            "capture_summary_sha256": sha256_file(args.capture_summary.resolve()),
            "hardware_metrics_sha256": sha256_file(args.hardware_metrics.resolve()),
            "traffic_resource_model_sha256": sha256_file(
                args.traffic_resource_model.resolve()
            ),
            "completion_audit_sha256": sha256_file(args.completion_audit.resolve()),
        },
        "artifact_budget": {
            "profiling_wall_time_seconds": captures["profiling_wall_seconds"],
            "maximum_profiling_wall_time_seconds": contract[
                "maximum_profiling_wall_time_seconds"
            ],
            "artifact_bytes_before_handoff": bytes_now,
            "maximum_trace_bundle_bytes": contract["maximum_trace_bundle_bytes"],
            "within_limit": True,
        },
        "validation": {
            "status": "PASS",
            "independent_check_count": audit["independent_check_count"],
            "independent_failure_check_count": 0,
            "selected_family_count": coverage["selected_family_count"],
            "final_disposition_count": coverage["final_disposition_count"],
            "capture_count": captures["completed_capture_count"],
            "coverage_fraction": coverage["selected_family_coverage_fraction"],
            "model_family_count": model["coverage"]["kernel_family_count"],
            "model_resource_complete_family_count": model["coverage"][
                "resource_complete_family_count"
            ],
        },
        "downstream_consumption": {
            "consumer_goal": "R09",
            "device_capabilities": str(args.device_capabilities.resolve()),
            "hardware_metrics": str(args.hardware_metrics.resolve()),
            "hardware_metrics_by_kernel_family": str(
                args.hardware_family_metrics.resolve()
            ),
            "traffic_resource_model": str(args.traffic_resource_model.resolve()),
            "strict_consumer_rule": (
                "use R07 non-replay timing as the only latency axis; attach R08 "
                "PMC/resource values as replay_projected and never merge replay clocks"
            ),
        },
        "evidence_boundary": {
            "establishes": (
                "fresh same-lineage replay-projected compute/cache/stall/resource "
                "diagnostics for every R06-selected R07 family and an FX-visible "
                "traffic/DCU resource model"
            ),
            "does_not_establish": (
                "replay latency, HBM/DRAM traffic or bandwidth, achieved occupancy, "
                "cross-capture concurrency, or optimization causality"
            ),
            "latency_axis": "R07_non_replay_same_request_only",
            "pmc_replay_duration_is_latency_evidence": False,
            "hbm_or_dram_traffic_claimed": False,
            "achieved_occupancy_claimed": False,
            "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
        },
    }
    handoff_output.parent.mkdir(parents=True, exist_ok=True)
    with handoff_output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "handoff": str(handoff_output),
                "sha256": sha256_file(handoff_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
