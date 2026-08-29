#!/usr/bin/env python3
"""Consolidate serial R08 PMC batches onto the R07 non-replay latency axis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


MODES = ("pmc", "pmc-read", "pmc-write")
UNAVAILABLE = "unavailable"


class ConsolidationError(RuntimeError):
    """Fail-closed R08 consolidation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ConsolidationError(f"expected non-empty JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ConsolidationError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def require_record(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ConsolidationError(f"{label} is missing or changed: {path}")
    return path


def number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, "", UNAVAILABLE):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def display(value: float | None, digits: int = 9) -> float | str:
    return round(value, digits) if value is not None and math.isfinite(value) else UNAVAILABLE


def weighted(
    rows: list[dict[str, str]],
    field: str | None = None,
    derived: Callable[[dict[str, str]], float | None] | None = None,
) -> tuple[float | None, int]:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = derived(row) if derived else number(row, str(field))
        weight = number(row, "kernel_time")
        if value is not None and weight is not None and weight > 0:
            values.append((value, weight))
    if not values:
        return None, 0
    denominator = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / denominator, len(values)


def mean_available(rows: list[dict[str, str]], field: str) -> tuple[float | None, int]:
    values = [value for row in rows if (value := number(row, field)) is not None]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def occupancy_upper(row: dict[str, str]) -> float | None:
    workgroup = number(row, "work_group_size")
    vgpr = number(row, "vgpr_count")
    shared = number(row, "shared_memory_size")
    if not workgroup or not vgpr or shared is None:
        return None
    workgroup_i = int(workgroup)
    vgpr_i = int(vgpr)
    if workgroup_i <= 0 or vgpr_i <= 0 or shared < 0:
        return None
    waves_per_group = math.ceil(workgroup_i / 64)
    by_wave = 40 // waves_per_group
    by_thread = 2560 // workgroup_i
    by_vgpr = 196608 // (vgpr_i * workgroup_i)
    by_shared = 10**9 if shared == 0 else 65536 // max(1, int(shared))
    groups = max(0, min(by_wave, by_thread, by_vgpr, by_shared))
    return min(100.0, 100.0 * groups * waves_per_group / 40.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate fresh R08 PMC batches.")
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--targeted-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    contract = load_json(args.run_contract.resolve())
    capture_summary = load_json(args.capture_summary.resolve())
    plan_path = args.targeted_plan.resolve()
    plan = read_csv(plan_path)
    if (
        contract.get("runtime_goal") != "R08"
        or capture_summary.get("status") != "complete"
        or capture_summary.get("lineage_id") != contract.get("lineage_id")
        or not plan
    ):
        raise ConsolidationError("invalid R08 contract, capture summary, or plan")
    if sha256_file(plan_path) != contract["targeted_family_plan_csv"]["sha256"]:
        raise ConsolidationError("targeted plan changed after contract freeze")
    if len(plan) > int(contract["maximum_targeted_pmc_family_count"]):
        raise ConsolidationError("target family cap exceeded")
    plan_by_batch = {row["capture_batch_id"]: row for row in plan}
    if len(plan_by_batch) != len(plan):
        raise ConsolidationError("duplicate target batch identity")
    capture_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for capture in capture_summary.get("captures", []):
        key = (capture["capture_batch_id"], capture["mode"])
        if key in capture_by_key or capture.get("status") != "complete":
            raise ConsolidationError(f"duplicate or incomplete capture: {key}")
        capture_by_key[key] = capture
    expected_capture_count = len(plan) * len(MODES)
    if (
        len(capture_by_key) != expected_capture_count
        or int(capture_summary.get("completed_capture_count", -1))
        != expected_capture_count
    ):
        raise ConsolidationError("capture count does not cover every batch/mode")

    output_rows: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    per_capture_evidence: list[dict[str, Any]] = []
    for target in sorted(plan, key=lambda row: int(row["selection_rank"])):
        batch_id = target["capture_batch_id"]
        family_key = (
            target["event_id"],
            target["stage"],
            target["matched_kernel_family"],
        )
        metrics_by_mode: dict[str, list[dict[str, str]]] = {}
        mode_summaries: dict[str, dict[str, Any]] = {}
        capture_refs: dict[str, Any] = {}
        discarded_count = 0
        for mode in MODES:
            capture = capture_by_key.get((batch_id, mode))
            if capture is None:
                raise ConsolidationError(f"missing capture for {batch_id}/{mode}")
            summary_path = require_record(capture["pmc_summary"], f"{batch_id}/{mode} summary")
            metrics_path = require_record(
                capture["hardware_kernel_metrics"], f"{batch_id}/{mode} metrics"
            )
            discarded_path = require_record(
                capture["discarded_superset_matches"],
                f"{batch_id}/{mode} discarded superset audit",
            )
            summary = load_json(summary_path)
            if (
                summary.get("status") != "PASS"
                or summary.get("kind") != mode
                or summary.get("collection_policy") != "bounded-family-superset"
                or summary.get("capture_batch_id") != batch_id
                or summary.get("kernel_name_filter")
                != target["kernel_name_filter_literal"]
                or float(summary.get("name_order_match_rate", 0.0))
                < float(contract["minimum_name_order_match_rate"])
                or int(summary.get("ambiguous_pair_count", -1)) != 0
                or int(summary.get("unmatched_selected_block_count", -1)) != 0
                or int(summary.get("unmatched_selected_trace_kernel_count", -1)) != 0
                or int(summary.get("missing_selected_kernel_count", -1)) != 0
                or int(summary.get("selected_target_family_count", -1)) != 1
                or int(summary.get("covered_selected_family_count", -1)) != 1
                or float(summary.get("selected_exact_attribution_rate", 0.0)) != 1.0
                or summary.get("final_process_family_attribution_exact") is not True
                or summary.get("fuzzy_name_matching_used") is not False
                or summary.get("pmc_is_latency_evidence") is not False
            ):
                raise ConsolidationError(f"strict PMC evidence failed: {batch_id}/{mode}")
            rows = read_csv(metrics_path)
            if not rows:
                raise ConsolidationError(f"empty selected metrics: {batch_id}/{mode}")
            for row in rows:
                observed_key = (row["event_id"], row["stage"], row["kernel_family"])
                if (
                    observed_key != family_key
                    or row["process_marker"] != target["hiptx_range"]
                    or row["capture_batch_id"] != batch_id
                    or row["kernel_name_filter"]
                    != target["kernel_name_filter_literal"]
                ):
                    raise ConsolidationError(
                        f"selected family/owner drift in {batch_id}/{mode}: {observed_key}"
                    )
            expected_instances = int(target["r07_kernel_instance_count"])
            if len(rows) != expected_instances:
                raise ConsolidationError(
                    f"replay instance count changed for {batch_id}/{mode}: "
                    f"{len(rows)} != {expected_instances}"
                )
            metrics_by_mode[mode] = rows
            mode_summaries[mode] = summary
            discarded_rows = read_csv(discarded_path)
            if len(discarded_rows) != int(summary["discarded_superset_match_count"]):
                raise ConsolidationError(f"discarded superset audit count drift: {batch_id}/{mode}")
            discarded_count += len(discarded_rows)
            capture_refs[mode] = {
                "raw_db": capture["raw_db"],
                "raw_pmc": capture["raw_pmc"],
                "metadata": capture["metadata"],
                "trace_summary": capture["trace_summary"],
                "pmc_summary": capture["pmc_summary"],
                "hardware_kernel_metrics": capture["hardware_kernel_metrics"],
                "discarded_superset_matches": capture[
                    "discarded_superset_matches"
                ],
            }
            per_capture_evidence.append(
                {
                    "capture_batch_id": batch_id,
                    "mode": mode,
                    "name_order_match_rate": summary["name_order_match_rate"],
                    "selected_exact_attribution_rate": summary[
                        "selected_exact_attribution_rate"
                    ],
                    "selected_kernel_count": len(rows),
                    "discarded_superset_match_count": len(discarded_rows),
                    "ambiguous_pair_count": summary["ambiguous_pair_count"],
                    "unmatched_selected_block_count": summary[
                        "unmatched_selected_block_count"
                    ],
                    "unmatched_selected_trace_kernel_count": summary[
                        "unmatched_selected_trace_kernel_count"
                    ],
                }
            )

        compute = metrics_by_mode["pmc"]
        reads = metrics_by_mode["pmc-read"]
        writes = metrics_by_mode["pmc-write"]
        alu, alu_samples = weighted(compute, "processed_alu_instructions")
        l2_hit, l2_hit_samples = weighted(compute, "l2_cache_hit_rate")
        workgroup, workgroup_samples = weighted(compute, "work_group_size")
        vgpr, vgpr_samples = weighted(compute, "vgpr_count")
        sgpr, sgpr_samples = weighted(compute, "sgpr_count")
        shared, shared_samples = weighted(compute, "shared_memory_size")
        occupancy, occupancy_samples = weighted(compute, derived=occupancy_upper)
        read_kb, read_samples = mean_available(reads, "size_of_l2_cache_read")
        write_kb, write_samples = mean_available(writes, "size_of_l2_cache_write")
        required_values = {
            "processed_alu_instructions": alu,
            "l2_cache_hit_rate": l2_hit,
            "work_group_size": workgroup,
            "vgpr_count": vgpr,
            "sgpr_count": sgpr,
            "shared_memory_size": shared,
            "theoretical_occupancy": occupancy,
            "size_of_l2_cache_read": read_kb,
            "size_of_l2_cache_write": write_kb,
        }
        unavailable_required = [name for name, value in required_values.items() if value is None]
        if unavailable_required:
            raise ConsolidationError(
                f"unverified required counter semantics for {batch_id}: {unavailable_required}"
            )

        stall_fields = {
            "L1_cache_stall": "l1_cache_unit_is_stalled",
            "L2_write_stall": "l2_cache_write_unit_is_stalled",
            "shared_memory_bank_conflict": "shared_memory_bank_conflict",
        }
        stalls: dict[str, tuple[float, int]] = {}
        for label, field in stall_fields.items():
            value, samples = weighted(compute, field)
            if value is not None:
                stalls[label] = (value, samples)
        strongest_stall = max(stalls, key=lambda key: stalls[key][0]) if stalls else UNAVAILABLE
        r07_instances = int(target["r07_kernel_instance_count"])
        r07_duration_ms = float(target["r07_kernel_duration_ms"])
        projected_l2_bytes = (float(read_kb) + float(write_kb)) * 1024.0 * r07_instances
        projected_l2_gbps = (
            projected_l2_bytes / (r07_duration_ms / 1000.0) / 1e9
            if r07_duration_ms > 0
            else None
        )
        profiled_names = sorted(
            {row["kernel_name"] for mode in MODES for row in metrics_by_mode[mode]}
        )
        output_rows.append(
            {
                "selection_rank": int(target["selection_rank"]),
                "selection_group_id": target["selection_group_id"],
                "capture_batch_id": batch_id,
                "hardware_family_key": target["hardware_family_key"],
                "event_id": target["event_id"],
                "stage": target["stage"],
                "process_range": target["hiptx_range"],
                "process_id": target["process_id"],
                "fragment_id": target["fragment_id"],
                "aggregation_key": target["aggregation_key"],
                "matched_kernel_family": target["matched_kernel_family"],
                "hardware_join_key": "event_id+stage+matched_kernel_family",
                "final_disposition": "collected",
                "dcu_pmc_status": "complete",
                "hardware_evidence_class": "replay_projected_current_family",
                "row_reuse_or_path_state": "fresh_same_lineage_replay_collected",
                "kernel_name_filter_literal": target["kernel_name_filter_literal"],
                "pmc_profiled_kernel_names": ";".join(profiled_names),
                "r07_non_replay_process_hiptx_host_duration_ms": float(
                    target["r07_process_hiptx_cpu_ms"]
                ),
                "r07_non_replay_kernel_family_instance_count": r07_instances,
                "r07_non_replay_kernel_family_duration_ms": r07_duration_ms,
                "timing_source": "R07_non_replay_same_request_family",
                "latency_axis": "R07_non_replay_same_request_only",
                "pmc_replay_timing_used_as_latency": False,
                "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
                "pmc_kernel_family_instance_count": len(compute),
                "pmc_read_kernel_family_instance_count": len(reads),
                "pmc_write_kernel_family_instance_count": len(writes),
                "DCU_activity_processed_ALU_pct": display(alu),
                "DCU_activity_sample_count": alu_samples,
                "DCU_matrix_core_utilization_proxy_pct": (
                    display(alu)
                    if target["matched_kernel_family"] == "TunableOp_MMAC_GEMM"
                    else "not_applicable"
                ),
                "DCU_matrix_proxy_definition": (
                    "processed_ALU activity; diagnostic MMAC-scoped proxy"
                    if target["matched_kernel_family"] == "TunableOp_MMAC_GEMM"
                    else "not_applicable"
                ),
                "L2_hit_rate_pct": display(l2_hit),
                "L2_hit_rate_sample_count": l2_hit_samples,
                "mean_L2_read_KB_per_replay_instance": display(read_kb),
                "L2_read_sample_count": read_samples,
                "mean_L2_write_KB_per_replay_instance": display(write_kb),
                "L2_write_sample_count": write_samples,
                "projected_L2_bytes_on_R07_instance_count": display(projected_l2_bytes, 3),
                "projected_L2_throughput_GBps_on_R07_latency_axis": display(projected_l2_gbps),
                "DRAM_throughput": UNAVAILABLE,
                "DRAM_unavailable_reason": (
                    "selected hipprof modes expose no verified HBM/DRAM equivalent"
                ),
                "work_group_size": display(workgroup),
                "weighted_work_group_size": display(workgroup),
                "work_group_size_sample_count": workgroup_samples,
                "VGPR_count": display(vgpr),
                "weighted_VGPR_count": display(vgpr),
                "VGPR_sample_count": vgpr_samples,
                "SGPR_count": display(sgpr),
                "weighted_SGPR_count": display(sgpr),
                "SGPR_sample_count": sgpr_samples,
                "shared_memory_size_bytes": display(shared),
                "weighted_shared_memory_size_bytes": display(shared),
                "shared_memory_sample_count": shared_samples,
                "theoretical_occupancy_upper_bound_pct": display(occupancy),
                "occupancy_sample_count": occupancy_samples,
                "occupancy_interpretation": (
                    "theoretical gfx936 resource upper bound; not achieved occupancy"
                ),
                "achieved_occupancy_pct": UNAVAILABLE,
                "strongest_available_stall_proxy": strongest_stall,
                "strongest_available_stall_proxy_value": (
                    display(stalls[strongest_stall][0])
                    if strongest_stall != UNAVAILABLE
                    else UNAVAILABLE
                ),
                "stall_proxy_sample_count": (
                    stalls[strongest_stall][1]
                    if strongest_stall != UNAVAILABLE
                    else 0
                ),
                "minimum_selected_name_order_match_rate": min(
                    float(mode_summaries[mode]["name_order_match_rate"])
                    for mode in MODES
                ),
                "selected_exact_attribution_rate": min(
                    float(mode_summaries[mode]["selected_exact_attribution_rate"])
                    for mode in MODES
                ),
                "discarded_superset_match_count": discarded_count,
                "capture_evidence_json": json.dumps(
                    capture_refs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        dispositions.append(
            {
                "selection_rank": int(target["selection_rank"]),
                "hardware_family_key": target["hardware_family_key"],
                "capture_batch_id": batch_id,
                "event_id": target["event_id"],
                "stage": target["stage"],
                "matched_kernel_family": target["matched_kernel_family"],
                "disposition": "collected",
                "disposition_reason": "all three fresh modes passed exact post-attribution",
            }
        )

    if len(output_rows) != len(plan) or len(dispositions) != len(plan):
        raise ConsolidationError("final family disposition is not exactly-once complete")
    if len({row["hardware_family_key"] for row in dispositions}) != len(plan):
        raise ConsolidationError("duplicate final family disposition")
    hardware_path = output_root / "hardware_metrics.csv"
    family_path = output_root / "hardware_metrics_by_kernel_family.csv"
    disposition_path = output_root / "targeted_family_disposition.csv"
    coverage_path = output_root / "hardware_coverage.json"
    write_csv_x(hardware_path, output_rows)
    write_csv_x(family_path, output_rows)
    write_csv_x(disposition_path, dispositions)
    coverage = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_id": contract["lineage_id"],
        "selected_family_count": len(plan),
        "final_disposition_count": len(dispositions),
        "disposition_counts": {
            "collected": len(dispositions),
            "no_kernel": 0,
            "unavailable": 0,
            "failed": 0,
        },
        "selected_family_coverage_fraction": 1.0,
        "complete_selected_family_coverage": True,
        "capture_count": expected_capture_count,
        "minimum_name_order_match_rate_required": contract[
            "minimum_name_order_match_rate"
        ],
        "minimum_name_order_match_rate_observed": min(
            float(row["name_order_match_rate"]) for row in per_capture_evidence
        ),
        "minimum_selected_exact_attribution_rate": min(
            float(row["selected_exact_attribution_rate"])
            for row in per_capture_evidence
        ),
        "selected_ambiguity_count": sum(
            int(row["ambiguous_pair_count"]) for row in per_capture_evidence
        ),
        "unmatched_selected_block_count": sum(
            int(row["unmatched_selected_block_count"])
            for row in per_capture_evidence
        ),
        "unmatched_selected_trace_kernel_count": sum(
            int(row["unmatched_selected_trace_kernel_count"])
            for row in per_capture_evidence
        ),
        "discarded_superset_match_count": sum(
            int(row["discarded_superset_match_count"])
            for row in per_capture_evidence
        ),
        "discarded_rows_audited": True,
        "matching_rule": (
            "same-replay pid + exact observed-name subsequence + dispatch order + "
            "exact demangled name, then strict HIPTX/runtime/_Index/HIPOPS ownership"
        ),
        "latency_axis": "R07_non_replay_same_request_only",
        "pmc_replay_duration_is_latency_evidence": False,
        "cross_capture_timeline_policy": "separate_clock_axes_no_merge",
        "hbm_or_dram_traffic_claimed": False,
        "achieved_occupancy_claimed": False,
        "counter_semantics_verified_per_selected_row": True,
        "targeted_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "capture_summary": {
            "path": str(args.capture_summary.resolve()),
            "sha256": sha256_file(args.capture_summary.resolve()),
        },
        "outputs": {
            "hardware_metrics": {
                "path": str(hardware_path),
                "sha256": sha256_file(hardware_path),
            },
            "hardware_metrics_by_kernel_family": {
                "path": str(family_path),
                "sha256": sha256_file(family_path),
            },
            "family_disposition": {
                "path": str(disposition_path),
                "sha256": sha256_file(disposition_path),
            },
        },
        "per_capture_evidence": per_capture_evidence,
    }
    write_json_x(coverage_path, coverage)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
