#!/usr/bin/env python3
"""Independently audit the fresh R08 hardware and traffic/resource evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = {"pmc", "pmc-read", "pmc-write"}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    parser = argparse.ArgumentParser(description="Audit fresh R08 evidence.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--device-capabilities", type=Path, required=True)
    parser.add_argument("--hardware-coverage", type=Path, required=True)
    parser.add_argument("--hardware-metrics", type=Path, required=True)
    parser.add_argument("--hardware-family-metrics", type=Path, required=True)
    parser.add_argument("--traffic-resource-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    contract = load_json(args.run_contract.resolve())
    source = load_json(args.source_lineage.resolve())
    captures = load_json(args.capture_summary.resolve())
    capabilities = load_json(args.device_capabilities.resolve())
    coverage = load_json(args.hardware_coverage.resolve())
    model = load_json(args.traffic_resource_model.resolve())
    metrics = read_csv(args.hardware_metrics.resolve())
    family_metrics = read_csv(args.hardware_family_metrics.resolve())
    failures: list[str] = []
    check_count = 0

    def check(condition: bool, message: str) -> None:
        nonlocal check_count
        check_count += 1
        if not condition:
            failures.append(message)

    check(contract.get("runtime_goal") == "R08", "run contract goal")
    check(contract.get("status") == "ready", "run contract status")
    lineage_id = contract.get("lineage_id")
    check(source.get("lineage_id") == lineage_id, "source lineage identity")
    check(captures.get("lineage_id") == lineage_id, "capture lineage identity")
    check(coverage.get("lineage_id") == lineage_id, "coverage lineage identity")
    check(model.get("lineage_id") == lineage_id, "model lineage identity")
    check(source.get("source_hash_equality_required") is False, "source hash policy")
    check(
        source.get("model_input_sampling_device_semantics_changed") is False,
        "semantic source boundary",
    )
    check(
        source.get("r07_process_family_identity_changed") is False,
        "R07 identity boundary",
    )
    for role, record in source.get("tools", {}).items():
        path = Path(str(record.get("path", ""))).resolve()
        check(path.is_file(), f"tool missing: {role}")
        if path.is_file():
            check(sha256_file(path) == record.get("sha256"), f"tool drift: {role}")
    for label, record in source.get("inputs", {}).items():
        path = Path(str(record.get("path", ""))).resolve()
        check(path.is_file(), f"input missing: {label}")
        if path.is_file():
            check(sha256_file(path) == record.get("sha256"), f"input drift: {label}")

    check(capabilities.get("status") == "verified", "device capability status")
    check(capabilities.get("physical_device_id") == 1, "physical device binding")
    check(capabilities.get("architecture") == "gfx936", "gfx936 architecture")
    counters = capabilities.get("counter_availability", {})
    check(counters.get("status") == "verified_collector_modes", "counter mode probe")
    for key in ("compute_cache_stall", "l2_read", "l2_write", "literal_kernel_filter"):
        check(counters.get(key, {}).get("available") is True, f"counter availability: {key}")
    unavailable = capabilities.get("unavailable_quantities", {})
    check("hbm_or_dram_bytes" in unavailable, "explicit HBM/DRAM unavailability")
    check("achieved_occupancy_pct" in unavailable, "explicit achieved occupancy unavailability")

    target_count = int(coverage.get("selected_family_count", -1))
    check(coverage.get("status") == "PASS", "hardware coverage status")
    check(coverage.get("complete_selected_family_coverage") is True, "complete family coverage")
    check(float(coverage.get("selected_family_coverage_fraction", 0.0)) == 1.0, "family coverage fraction")
    check(target_count == len(metrics) == len(family_metrics), "family metric row count")
    check(target_count <= int(contract["maximum_targeted_pmc_family_count"]), "family cap")
    check(coverage.get("disposition_counts", {}).get("collected") == target_count, "collected disposition count")
    check(coverage.get("disposition_counts", {}).get("no_kernel") == 0, "unexpected no-kernel disposition")
    check(coverage.get("disposition_counts", {}).get("unavailable") == 0, "unavailable disposition")
    check(coverage.get("disposition_counts", {}).get("failed") == 0, "failed disposition")
    check(coverage.get("discarded_rows_audited") is True, "superset discard audit")
    check(float(coverage.get("minimum_name_order_match_rate_observed", 0.0)) >= float(contract["minimum_name_order_match_rate"]), "minimum name/order rate")
    check(float(coverage.get("minimum_selected_exact_attribution_rate", 0.0)) == 1.0, "exact selected attribution")
    check(int(coverage.get("selected_ambiguity_count", -1)) == 0, "selected ambiguity")
    check(int(coverage.get("unmatched_selected_block_count", -1)) == 0, "unmatched selected PMC block")
    check(int(coverage.get("unmatched_selected_trace_kernel_count", -1)) == 0, "unmatched selected trace kernel")
    check(coverage.get("pmc_replay_duration_is_latency_evidence") is False, "coverage replay timing boundary")

    family_keys: set[str] = set()
    for row in metrics:
        key = row.get("hardware_family_key", "")
        check(bool(key) and key not in family_keys, f"duplicate or empty family key: {key}")
        family_keys.add(key)
        check(row.get("final_disposition") == "collected", f"family disposition: {key}")
        check(row.get("dcu_pmc_status") == "complete", f"PMC status: {key}")
        check(row.get("hardware_evidence_class") == "replay_projected_current_family", f"hardware evidence class: {key}")
        check(row.get("timing_source") == "R07_non_replay_same_request_family", f"R07 timing source: {key}")
        check(row.get("latency_axis") == "R07_non_replay_same_request_only", f"latency axis: {key}")
        check(row.get("pmc_replay_timing_used_as_latency", "").lower() == "false", f"replay latency flag: {key}")
        check(row.get("cross_capture_timeline_policy") == "separate_clock_axes_no_merge", f"clock policy: {key}")
        check(row.get("DRAM_throughput") == "unavailable", f"DRAM boundary: {key}")
        check(row.get("achieved_occupancy_pct") == "unavailable", f"achieved occupancy boundary: {key}")
        check(row.get("theoretical_occupancy_upper_bound_pct") not in ("", "unavailable"), f"theoretical occupancy: {key}")
        check(float(row.get("minimum_selected_name_order_match_rate", 0.0)) >= float(contract["minimum_name_order_match_rate"]), f"row name/order rate: {key}")
        check(float(row.get("selected_exact_attribution_rate", 0.0)) == 1.0, f"row exact attribution: {key}")
    check(metrics == family_metrics, "hardware metrics/family metrics deterministic equality")

    expected_capture_count = target_count * len(MODES)
    check(captures.get("status") == "complete", "capture summary status")
    check(int(captures.get("completed_capture_count", -1)) == expected_capture_count, "capture count")
    roots: set[str] = set()
    mode_batch_keys: set[tuple[str, str]] = set()
    for capture in captures.get("captures", []):
        batch = capture.get("capture_batch_id")
        mode = capture.get("mode")
        key = (str(batch), str(mode))
        check(mode in MODES and key not in mode_batch_keys, f"capture identity: {key}")
        mode_batch_keys.add(key)
        for label in (
            "preflight",
            "launcher_preflight",
            "metadata",
            "runtime_events",
            "raw_db",
            "raw_pmc",
            "provenance",
            "trace_summary",
            "pmc_summary",
            "hardware_kernel_metrics",
            "discarded_superset_matches",
            "analysis_compaction_manifest",
        ):
            record = capture.get(label, {})
            path = Path(str(record.get("path", ""))).resolve()
            check(path.is_file(), f"capture evidence missing {key}/{label}")
            if path.is_file():
                check(sha256_file(path) == record.get("sha256"), f"capture evidence drift {key}/{label}")
                check(path.is_relative_to(artifact_root), f"capture path outside R08 {key}/{label}")
            if label == "raw_db" and path.is_file():
                roots.add(str(path.parent.parent))
        preflight_path = Path(capture["preflight"]["path"])
        if preflight_path.is_file():
            preflight = load_json(preflight_path)
            card = preflight.get("card", {})
            check(preflight.get("status") == "idle_verified", f"preflight status {key}")
            check(preflight.get("physical_device_id") == 1, f"preflight device {key}")
            check(card.get("Unique ID") == contract["device_unique_id"], f"preflight identity {key}")
            check(float(card.get("HCU use (%)", "nan")) == 0.0, f"preflight use {key}")
            check(float(card.get("HCU memory use (%)", "nan")) == 0.0, f"preflight memory {key}")
        summary_path = Path(capture["pmc_summary"]["path"])
        if summary_path.is_file():
            summary = load_json(summary_path)
            check(summary.get("status") == "PASS", f"PMC summary status {key}")
            check(summary.get("capture_batch_id") == batch, f"PMC batch binding {key}")
            check(summary.get("kind") == mode, f"PMC mode binding {key}")
            check(int(summary.get("ambiguous_pair_count", -1)) == 0, f"PMC ambiguity {key}")
            check(float(summary.get("selected_exact_attribution_rate", 0.0)) == 1.0, f"PMC attribution {key}")
            check(summary.get("pmc_is_latency_evidence") is False, f"PMC latency boundary {key}")
    check(len(roots) == expected_capture_count, "fresh distinct capture roots")

    check(model.get("status") == "complete", "traffic/resource model status")
    check(model.get("model_type") == "fresh_run_fx_visible_traffic_and_dcu_family_resource", "traffic/resource model type")
    check(model.get("traffic_boundary", {}).get("hbm_or_dram_traffic_claimed") is False, "model traffic boundary")
    check(model.get("resource_boundary", {}).get("achieved_occupancy_claimed") is False, "model occupancy boundary")
    check(int(model.get("coverage", {}).get("kernel_family_count", -1)) == target_count, "model family coverage")
    check(int(model.get("coverage", {}).get("resource_complete_family_count", -1)) == target_count, "model resource completeness")
    for label in ("traffic", "resource"):
        record = model.get("outputs", {}).get(label, {})
        path = Path(str(record.get("path", ""))).resolve()
        check(path.is_file() and path.is_relative_to(artifact_root), f"model output path: {label}")
        if path.is_file():
            check(sha256_file(path) == record.get("sha256"), f"model output drift: {label}")

    current_bytes = artifact_bytes(artifact_root)
    check(float(captures.get("profiling_wall_seconds", float("inf"))) <= float(contract["maximum_profiling_wall_time_seconds"]), "profiling wall-time cap")
    check(current_bytes < int(contract["maximum_trace_bundle_bytes"]), "artifact byte cap before audit")
    status = "PASS" if not failures else "FAIL"
    payload = {
        "schema_version": 1,
        "status": status,
        "runtime_goal": "R08",
        "lineage_id": lineage_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "failure_checks": failures,
        "independent_check_count": check_count,
        "selected_family_count": target_count,
        "capture_count": expected_capture_count,
        "minimum_name_order_match_rate": coverage.get("minimum_name_order_match_rate_observed"),
        "selected_exact_attribution_rate": coverage.get("minimum_selected_exact_attribution_rate"),
        "selected_ambiguity_count": coverage.get("selected_ambiguity_count"),
        "profiling_wall_seconds": captures.get("profiling_wall_seconds"),
        "maximum_profiling_wall_time_seconds": contract["maximum_profiling_wall_time_seconds"],
        "artifact_bytes_before_audit": current_bytes,
        "maximum_trace_bundle_bytes": contract["maximum_trace_bundle_bytes"],
        "coverage_target_met": not failures,
        "latency_axis": "R07_non_replay_same_request_only",
        "pmc_replay_duration_is_latency_evidence": False,
        "hbm_or_dram_traffic_claimed": False,
        "achieved_occupancy_claimed": False,
        "source_hash_drift_detected": any("drift" in failure for failure in failures),
        "inputs": {
            "run_contract": {"path": str(args.run_contract.resolve()), "sha256": sha256_file(args.run_contract.resolve())},
            "source_lineage": {"path": str(args.source_lineage.resolve()), "sha256": sha256_file(args.source_lineage.resolve())},
            "capture_summary": {"path": str(args.capture_summary.resolve()), "sha256": sha256_file(args.capture_summary.resolve())},
            "device_capabilities": {"path": str(args.device_capabilities.resolve()), "sha256": sha256_file(args.device_capabilities.resolve())},
            "hardware_coverage": {"path": str(args.hardware_coverage.resolve()), "sha256": sha256_file(args.hardware_coverage.resolve())},
            "hardware_metrics": {"path": str(args.hardware_metrics.resolve()), "sha256": sha256_file(args.hardware_metrics.resolve())},
            "hardware_family_metrics": {"path": str(args.hardware_family_metrics.resolve()), "sha256": sha256_file(args.hardware_family_metrics.resolve())},
            "traffic_resource_model": {"path": str(args.traffic_resource_model.resolve()), "sha256": sha256_file(args.traffic_resource_model.resolve())},
        },
    }
    if args.output.exists():
        raise RuntimeError(f"refusing existing audit output: {args.output}")
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
