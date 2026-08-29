#!/usr/bin/env python3
"""Freeze an analysis-only R08 recovery after a fail-closed model build."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def git_status_sha(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--recovery-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve()
    runtime_root = args.runtime_root.resolve()
    recovery_id = args.recovery_id
    prior_contract_path = artifact_root / "R08_RUN_CONTRACT_RECOVERY_001.json"
    prior_source_path = artifact_root / "R08_SOURCE_LINEAGE_RECOVERY_001.json"
    recovery_path = artifact_root / "R08_ANALYSIS_RECOVERY_002.json"
    source_path = artifact_root / "R08_SOURCE_LINEAGE_RECOVERY_002.json"
    contract_path = artifact_root / "R08_RUN_CONTRACT_RECOVERY_002.json"
    for path in (recovery_path, source_path, contract_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite recovery output: {path}")

    prior_contract = load_json(prior_contract_path)
    prior_source = load_json(prior_source_path)
    if prior_contract.get("runtime_goal") != "R08" or prior_contract.get("status") != "ready":
        raise RuntimeError("prior R08 recovery contract is not ready")
    if prior_source.get("lineage_id") != prior_contract.get("lineage_id"):
        raise RuntimeError("prior R08 contract/source lineage mismatch")

    builder = source_root / "scripts/perf_trace/build_traffic_resource_model.py"
    self_path = Path(__file__).resolve()
    old_builder = prior_source.get("tools", {}).get("model_builder", {})
    if old_builder.get("sha256") != "53f90c8599651665ee52742daa0b207b5366b81818896140bb81d9c63ae0f5be":
        raise RuntimeError("unexpected pre-recovery model-builder identity")
    new_builder_record = {
        **record(builder),
        "role": "model_builder",
        "git_tracking_state": old_builder.get(
            "git_tracking_state", "untracked_frozen_stage_tool"
        ),
    }
    if new_builder_record["sha256"] == old_builder.get("sha256"):
        raise RuntimeError("analysis recovery did not change the model builder")

    for role, item in prior_source.get("tools", {}).items():
        if role == "model_builder":
            continue
        path = Path(str(item.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"unrelated frozen tool drifted: {role}")
    for label, item in prior_source.get("inputs", {}).items():
        path = Path(str(item.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"frozen input drifted: {label}")

    py_compile.compile(str(builder), doraise=True)
    test_files = [
        source_root / "tests/perf_trace/test_workflow05_fresh_evidence_components.py",
        source_root / "tests/perf_trace/test_workflow05_pmc_bounded_superset.py",
        source_root / "tests/perf_trace/test_workflow05_process_range_filter.py",
    ]
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", *(str(path) for path in test_files)],
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if tests.returncode != 0 or "Ran 14 tests" not in tests.stdout or "OK" not in tests.stdout:
        raise RuntimeError("analysis recovery unit tests did not pass")

    r06_lineage = runtime_root / "artifacts/R06/resume-001/fresh_run_lineage_manifest.json"
    r07_adapter = runtime_root / "artifacts/R07/dependency/fresh_run_dependency_adapter.json"
    hardware_metrics = artifact_root / "hardware_metrics_by_kernel_family.csv"
    capabilities = artifact_root / "device_capabilities.json"
    with tempfile.TemporaryDirectory(prefix="r08-analysis-recovery-") as raw_tmp:
        tmp = Path(raw_tmp)
        build = subprocess.run(
            [
                sys.executable,
                str(builder),
                "--lineage-manifest",
                str(r06_lineage),
                "--dependency-adapter",
                str(r07_adapter),
                "--hardware-metrics",
                str(hardware_metrics),
                "--device-capabilities",
                str(capabilities),
                "--output-dir",
                str(tmp / "model"),
            ],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if build.returncode != 0:
            raise RuntimeError(f"analysis recovery integration build failed: {build.stdout}")
        model = load_json(tmp / "model/traffic_resource_model.json")
        coverage = model.get("coverage", {})
        if (
            model.get("status") != "complete"
            or coverage.get("process_stage_count") != 17168
            or coverage.get("kernel_family_count") != 32
            or coverage.get("resource_complete_family_count") != 32
            or model.get("traffic_boundary", {}).get("hbm_or_dram_traffic_claimed") is not False
            or model.get("resource_boundary", {}).get("achieved_occupancy_claimed") is not False
        ):
            raise RuntimeError("analysis recovery integration model failed evidence gates")

    failed_output_dir = artifact_root / "traffic_resource_model"
    failed_output_file_count = (
        sum(1 for path in failed_output_dir.rglob("*") if path.is_file())
        if failed_output_dir.exists()
        else 0
    )
    if failed_output_file_count != 0:
        raise RuntimeError("failed analysis attempt unexpectedly produced files")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    recovery = {
        "schema_version": 1,
        "status": "PASS",
        "runtime_goal": "R08",
        "recovery_id": recovery_id,
        "created_utc": now,
        "lineage_id": prior_contract["lineage_id"],
        "failure_stage": "traffic_resource_model_build",
        "failure_mode": "fail_closed_before_formal_model_output",
        "failure_reason": (
            "recorded duration-weighted per-dispatch theoretical occupancy was "
            "incorrectly compared with a nonlinear recomputation over separately "
            "weighted aggregate resource descriptors"
        ),
        "failed_output_dir": str(failed_output_dir),
        "failed_output_file_count": failed_output_file_count,
        "recovery_scope": "analysis-only occupancy aggregation semantics",
        "recovery_action": (
            "retain the consolidated duration-weighted mean of per-dispatch gfx936 "
            "theoretical occupancy upper bounds; require exact formula agreement for "
            "single-dispatch rows; never label the result as achieved occupancy"
        ),
        "measurement_or_profiler_run_performed": False,
        "capture_bytes_rewritten": False,
        "selected_family_identity_changed": False,
        "r07_latency_axis_changed": False,
        "pmc_replay_duration_promoted_to_latency": False,
        "before_model_builder": old_builder,
        "after_model_builder": new_builder_record,
        "prior_contract": record(prior_contract_path),
        "prior_source_lineage": record(prior_source_path),
        "capture_summary": record(artifact_root / "CAPTURE_RUN_SUMMARY.json"),
        "hardware_coverage": record(artifact_root / "hardware_coverage.json"),
        "hardware_metrics": record(hardware_metrics),
        "validation": {
            "py_compile": "PASS",
            "unit_test_count": 14,
            "unit_tests": "PASS",
            "integration_process_stage_count": 17168,
            "integration_kernel_family_count": 32,
            "integration_resource_complete_family_count": 32,
            "hbm_or_dram_traffic_claimed": False,
            "achieved_occupancy_claimed": False,
        },
    }
    write_exclusive(recovery_path, recovery)

    source = copy.deepcopy(prior_source)
    source.update(
        {
            "status": "frozen",
            "recovery_id": recovery_id,
            "recovery_scope": "analysis-only traffic/resource occupancy aggregation semantics",
            "model_input_sampling_device_semantics_changed": False,
            "r07_process_family_identity_changed": False,
            "git_status_porcelain_v1_z_sha256": git_status_sha(source_root),
        }
    )
    source["tools"]["model_builder"] = new_builder_record
    source["tools"]["analysis_recovery_planner"] = {
        **record(self_path),
        "role": "analysis_recovery_planner",
        "git_tracking_state": "untracked_frozen_stage_tool",
    }
    source["inputs"].update(
        {
            "prior_r08_recovery_contract": record(prior_contract_path),
            "prior_r08_recovery_source_lineage": record(prior_source_path),
            "analysis_recovery_evidence": record(recovery_path),
            "r08_capture_summary": record(artifact_root / "CAPTURE_RUN_SUMMARY.json"),
            "r08_hardware_coverage": record(artifact_root / "hardware_coverage.json"),
            "r08_hardware_metrics": record(hardware_metrics),
        }
    )
    source["stage_source_revision"] = (
        f"{prior_source['stage_source_revision']}"
        f".model{new_builder_record['sha256'][:12]}"
        f".analysisrecovery{source['tools']['analysis_recovery_planner']['sha256'][:12]}"
    )
    write_exclusive(source_path, source)

    contract = copy.deepcopy(prior_contract)
    prior_capture_recovery = copy.deepcopy(contract.get("recovery_evidence"))
    contract.update(
        {
            "status": "ready",
            "created_utc": now,
            "recovery_id": recovery_id,
            "prior_run_contract": record(prior_contract_path),
            "capture_recovery_evidence": prior_capture_recovery,
            "recovery_evidence": record(recovery_path),
            "source_lineage": record(source_path),
        }
    )
    write_exclusive(contract_path, contract)
    print(
        json.dumps(
            {
                "status": "PASS",
                "recovery": str(recovery_path),
                "source_lineage": str(source_path),
                "run_contract": str(contract_path),
                "model_builder_sha256": new_builder_record["sha256"],
                "unit_test_count": 14,
                "integration_process_stage_count": 17168,
                "integration_kernel_family_count": 32,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
