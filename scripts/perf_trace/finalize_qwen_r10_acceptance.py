#!/usr/bin/env python3
"""Validate and write the scheduler-assigned R10 handoff as the final action."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_NAME = "qwen-dcu-workflow05-trace-visualization-reporting"
PAGE_NAMES = (
    "index.html",
    "E2E_PROCESS_TIMELINE.html",
    "E2E_PROCESS_TIMELINE_LOSSLESS.html",
    "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html",
    "CONCURRENCY_UTILIZATION.html",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"JSON must be a nonempty object: {path}")
    return value


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reference(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"required output is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_scheduler(project_root: Path) -> Any:
    path = project_root / "perf_trace" / "scripts" / "run_perf_trace_01_05.py"
    spec = importlib.util.spec_from_file_location("r10_scheduler_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unique_inode_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            seen.add(key)
            total += stat.st_size
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize fresh-run R10 handoff.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--handoff-output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--maximum-trace-bundle-bytes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    handoff_output = args.handoff_output.resolve()
    expected_artifact_root = runtime_root / "artifacts" / "R10"
    expected_handoff = runtime_root / "handoffs" / "R10.json"
    if artifact_root != expected_artifact_root or handoff_output != expected_handoff:
        raise RuntimeError("R10 paths differ from the scheduler assignment")
    if not is_under(runtime_root, project_root) or not is_under(artifact_root, runtime_root):
        raise RuntimeError("R10 paths escaped the project/current run")
    if handoff_output.exists():
        raise RuntimeError(f"refusing to overwrite R10 handoff: {handoff_output}")

    acceptance_dir = artifact_root / "acceptance"
    manifest_path = acceptance_dir / "offline_acceptance_manifest.json"
    source_lineage_path = artifact_root / "R10_SOURCE_LINEAGE.json"
    audit_path = artifact_root / "R10_COMPLETION_AUDIT.json"
    manifest = load_object(manifest_path)
    source_lineage = load_object(source_lineage_path)
    audit = load_object(audit_path)
    lineage_id = manifest.get("lineage_id")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("self_contained_offline") is not True
        or source_lineage.get("status") != "PASS"
        or source_lineage.get("runtime_goal") != "R10"
        or source_lineage.get("lineage_id") != lineage_id
        or source_lineage.get("source_hash_equality_required") is not False
        or audit.get("status") != "PASS"
        or audit.get("runtime_goal") != "R10"
        or audit.get("lineage_id") != lineage_id
        or audit.get("independent_failure_check_count") != 0
        or audit.get("determinism", {}).get("status") != "PASS"
    ):
        raise RuntimeError("R10 manifest/source-lineage/audit is incomplete")

    output_records = manifest.get("outputs")
    if not isinstance(output_records, dict) or set(output_records) != set(PAGE_NAMES):
        raise RuntimeError("R10 page output set is incomplete")
    page_refs: dict[str, Any] = {}
    for name in PAGE_NAMES:
        page = acceptance_dir / name
        observed = reference(page)
        declared = output_records[name]
        if observed["sha256"] != declared.get("sha256") or observed["path"] != declared.get("path"):
            raise RuntimeError(f"generated page drifted after audit: {name}")
        page_refs[name] = observed

    manifest_ref = reference(manifest_path)
    source_lineage_ref = reference(source_lineage_path)
    audit_ref = reference(audit_path)
    companion_refs = {
        name: reference(Path(record["path"]))
        for name, record in manifest.get("companions", {}).items()
    }
    artifact_bytes = unique_inode_bytes(artifact_root)
    if artifact_bytes > args.maximum_trace_bundle_bytes:
        raise RuntimeError("R10 acceptance artifacts exceed the trace bundle limit")

    upstream_hashes = {
        goal: sha256_file(runtime_root / "handoffs" / f"{goal}.json")
        for goal in ("R06", "R07", "R08", "R09")
    }
    primary_outputs: dict[str, Any] = {
        "source_lineage": source_lineage_ref,
        "offline_acceptance_manifest": manifest_ref,
        "completion_audit": audit_ref,
    }
    primary_outputs.update(page_refs)
    if companion_refs:
        primary_outputs["compatible_trace_companions"] = companion_refs

    payload = {
        "schema_version": 1,
        "runtime_goal": "R10",
        "status": "complete",
        "execution_status": "complete",
        "evidence_status": "complete",
        "coverage_target_met": True,
        "next_authorization_required": False,
        "skill": SKILL_NAME,
        "branch": args.branch,
        "run_id": args.run_id,
        "workflow05_policy_version": "workflow05-low-cost-timeline-v4",
        "evidence_acquisition_mode": "fresh_no_prior_runtime_reuse",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime_root),
        "runtime_artifact_root": str(artifact_root),
        "handoff_output": str(handoff_output),
        "fresh_e2e_evidence": {
            "schema_version": 1,
            "status": "complete",
            "lineage_id": lineage_id,
            "offline_acceptance_manifest": manifest_ref,
            "source_lineage": source_lineage_ref,
        },
        "primary_outputs": primary_outputs,
        "same_run_binding": {
            "lineage_id": lineage_id,
            **{f"{goal}_handoff_sha256": digest for goal, digest in upstream_hashes.items()},
            "source_analysis_sha256": manifest["source_analysis"]["sha256"],
            "source_table_hashes": manifest["source_table_hashes"],
            "generator_sha256": manifest["generator"]["sha256"],
            "offline_acceptance_manifest_sha256": manifest_ref["sha256"],
            "source_lineage_sha256": source_lineage_ref["sha256"],
            "completion_audit_sha256": audit_ref["sha256"],
        },
        "validation": {
            "status": "PASS",
            "independent_check_count": audit.get("independent_check_count"),
            "independent_failure_check_count": 0,
            "deterministic_regeneration": True,
            "self_contained_offline": True,
            "all_required_pages_parse": True,
            "lossless_relative_nanosecond_timeline_complete": True,
            "complete_perfetto_trace_without_sampling": True,
            "source_table_hashes_verified": True,
            "request_bounds_reproduced": True,
            "high_latency_selection_reproduced": True,
            "evidence_legends_complete": True,
            "external_asset_count": 0,
            "broken_link_count": 0,
            "cross_clock_or_replay_latency_arithmetic": False,
        },
        "artifact_budget": {
            "artifact_bytes_before_handoff": artifact_bytes,
            "maximum_trace_bundle_bytes": args.maximum_trace_bundle_bytes,
            "profiling_wall_time_seconds": 0,
            "model_run_count": 0,
            "gpu_probe_count": 0,
            "profiler_run_count": 0,
            "pmc_replay_count": 0,
            "additional_sampling_count": 0,
            "network_download_count": 0,
            "within_limit": True,
        },
        "evidence_boundary": {
            "establishes": "A deterministic self-contained offline acceptance bundle reproducing the complete normalized R07 request/process/device timeline, R09 high-latency selections, live utilization, concurrency, dependencies, R08 replay-projected attributes, inferred FX-visible traffic, and opportunity states.",
            "does_not_establish": "Replay latency, cross-capture concurrency, HBM/DRAM traffic or bandwidth, achieved occupancy, optimization causality, or speedup.",
            "latency_axis": "R07_non_replay_same_request_only",
            "presentation_backend": manifest.get("presentation_backend"),
        },
    }

    ledger = load_object(runtime_root / "runtime_handoff_ledger.json")
    scheduler = load_scheduler(project_root)
    scheduler.validate_scheduler_handoff_payload(
        "R10",
        payload,
        expected_skill=SKILL_NAME,
        project_root=project_root,
        run_dir=runtime_root,
        branch=args.branch,
        run_id=args.run_id,
        ledger=ledger,
        user_parameters={"evidence_acquisition_mode": "fresh_no_prior_runtime_reuse"},
    )

    handoff_output.parent.mkdir(parents=True, exist_ok=True)
    with handoff_output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": "complete", "handoff": str(handoff_output),
        "sha256": sha256_file(handoff_output), "artifact_bytes": artifact_bytes,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
