#!/usr/bin/env python3
"""Independent completion audit for the current Qwen R04 hardware run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("pmc", "pmc-read", "pmc-write")
REQUIRED_TOP_LEVEL = (
    "dcu_process_selection_plan.csv",
    "hardware_replay_kernel_metrics.csv",
    "hardware_metrics_by_kernel_family.csv",
    "hardware_metrics.csv",
    "hardware_coverage.json",
    "DCU_HARDWARE_METRICS_REPORT.md",
    "SAME_INPUT_PRA_QWEN35_FULL_EAGER_PROCESS_WISE_DCU_REPORT.md",
)


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


def csv_value_counter(path: Path, field: str) -> Counter[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter(row[field] for row in csv.DictReader(handle))


def family_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["event_id"],
        row["stage"],
        row["matched_kernel_family"],
    )


def report_target_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        heading = lines.index(
            "## Kernel-family Hardware Attributes by Representative Layer"
        )
    except ValueError as exc:
        raise RuntimeError(f"missing primary report table in {path}") from exc
    header_index = next(
        index
        for index in range(heading + 1, len(lines))
        if lines[index].startswith("| ")
    )
    ids: list[str] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        ids.append(cells[-1])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    audit_path = root / "R04_INDEPENDENT_COMPLETION_AUDIT.json"
    checks: dict[str, Any] = {}
    failures: list[str] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}
        if not passed:
            failures.append(name)

    record(
        "output_scope",
        root.parent.name == "R04"
        and "perf_trace_bk" not in root.parts
        and source_root.name == "pra2026-bh408",
        {"output_root": str(root), "source_root": str(source_root)},
    )
    top_level: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_TOP_LEVEL:
        path = root / name
        exists = path.is_file() and path.stat().st_size > 0
        record(f"required_output:{name}", exists, str(path))
        if exists:
            top_level[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    run_contract = load_json(root / "R04_RUN_CONTRACT.json")
    coverage = load_json(root / "hardware_coverage.json")
    expected = run_contract["expected_denominator"]
    selection = read_csv(root / "dcu_process_selection_plan.csv")
    pre_selection = read_csv(
        root / "dcu_process_selection_plan.pre_collection.csv"
    )
    ledger_path = Path(
        run_contract["upstream_bindings"]["non_replay_family_ledger"]["path"]
    )
    expected_rows = read_csv(ledger_path)
    hardware_rows = read_csv(root / "hardware_metrics_by_kernel_family.csv")
    process_rows = read_csv(root / "hardware_metrics.csv")

    record(
        "run_contract_projection_status",
        run_contract["status"] == "hardware_projection_pass",
        run_contract["status"],
    )
    record("coverage_status", coverage["status"] == "PASS", coverage["status"])
    record(
        "coverage_failure_reasons_empty",
        coverage["failure_reasons"] == [],
        coverage["failure_reasons"],
    )
    record(
        "pre_collection_plan_preserved",
        sha256_file(root / "dcu_process_selection_plan.pre_collection.csv")
        == run_contract["selection_plan"]["pre_collection_sha256"],
        run_contract["selection_plan"],
    )
    record(
        "final_selection_plan_bound",
        sha256_file(root / "dcu_process_selection_plan.csv")
        == run_contract["selection_plan"]["sha256"],
        run_contract["selection_plan"]["sha256"],
    )
    record(
        "pre_collection_plan_was_pending",
        all(
            row["collection_status"] in {"pending", "no_kernel"}
            and not any(
                column in row
                for column in (
                    "pmc_collection_status",
                    "pmc_read_collection_status",
                    "pmc_write_collection_status",
                )
            )
            for row in pre_selection
        ),
        Counter(row["collection_status"] for row in pre_selection),
    )
    record(
        "final_selection_status",
        all(
            row["collection_status"]
            == ("no_kernel" if row["expected_no_kernel"] == "true" else "complete")
            and row["pmc_collection_status"] == "PASS"
            and row["pmc_read_collection_status"] == "PASS"
            and row["pmc_write_collection_status"] == "PASS"
            for row in selection
        ),
        Counter(row["collection_status"] for row in selection),
    )
    record(
        "selection_denominator",
        len(selection) == int(expected["process_fragment_targets"]),
        len(selection),
    )
    record(
        "launch_target_denominator",
        sum(row["collection_required"] == "true" for row in selection)
        == int(expected["launch_owning_capture_targets"]),
        sum(row["collection_required"] == "true" for row in selection),
    )
    record(
        "no_kernel_target_denominator",
        sum(row["expected_no_kernel"] == "true" for row in selection)
        == int(expected["no_kernel_process_targets"]),
        sum(row["expected_no_kernel"] == "true" for row in selection),
    )
    record(
        "representative_parent_denominator",
        len({row["parent_layer_range"] for row in selection})
        == int(expected["representative_parent_layers"]),
        len({row["parent_layer_range"] for row in selection}),
    )

    expected_keys = [family_key(row) for row in expected_rows]
    hardware_keys = [family_key(row) for row in hardware_rows]
    record(
        "family_ledger_file_binding",
        sha256_file(ledger_path)
        == run_contract["upstream_bindings"]["non_replay_family_ledger"][
            "sha256"
        ],
        str(ledger_path),
    )
    record(
        "projection_row_denominator",
        len(hardware_rows) == int(expected["all_projection_rows"]),
        len(hardware_rows),
    )
    record(
        "projection_preserves_exact_family_order",
        hardware_keys == expected_keys,
        {
            "expected_rows": len(expected_keys),
            "observed_rows": len(hardware_keys),
        },
    )
    record(
        "projection_family_keys_unique",
        len(hardware_keys) == len(set(hardware_keys)),
        len(set(hardware_keys)),
    )
    kernel_rows = [
        row
        for row in hardware_rows
        if row["matched_kernel_family"] != "no_kernel"
    ]
    no_kernel_rows = [
        row
        for row in hardware_rows
        if row["matched_kernel_family"] == "no_kernel"
    ]
    record(
        "all_required_family_rows_complete",
        len(kernel_rows) == int(expected["kernel_family_rows"])
        and all(row["dcu_pmc_status"] == "complete" for row in kernel_rows),
        Counter(row["dcu_pmc_status"] for row in kernel_rows),
    )
    record(
        "all_expected_no_kernel_rows_explicit",
        len(no_kernel_rows) == int(expected["no_kernel_family_rows"])
        and all(row["dcu_pmc_status"] == "no_kernel" for row in no_kernel_rows),
        Counter(row["dcu_pmc_status"] for row in no_kernel_rows),
    )
    record(
        "all_mode_family_counts_present",
        all(
            int(row["pmc_kernel_family_instance_count"]) > 0
            and int(row["pmc_read_kernel_family_instance_count"]) > 0
            and int(row["pmc_write_kernel_family_instance_count"]) > 0
            for row in kernel_rows
        ),
    )
    record(
        "no_replay_instance_count_drift",
        all(row["replay_instance_count_changed"].lower() == "false" for row in hardware_rows)
        and coverage["replay_instance_count_changed_rows"] == 0,
        coverage["replay_instance_count_changed_target_ids"],
    )
    record(
        "timing_source_boundary",
        all(
            row["timing_source"] == "workflow02_non_replay_family_row"
            and row["hardware_join_key"]
            == "event_id+stage+matched_kernel_family"
            and row["pmc_replay_timing_used_as_latency"].lower() == "false"
            for row in hardware_rows
        )
        and coverage["pmc_replay_timing_used_as_latency"] is False
        and coverage["pmc_is_latency_evidence"] is False,
    )
    record(
        "dram_unavailable_not_inferred",
        all(row["DRAM_throughput"] == "unavailable" for row in hardware_rows),
    )
    available_fields = (
        "DCU_activity_processed_ALU_pct",
        "L2_hit_rate_pct",
        "mean_L2_read_KB_per_replay_instance",
        "mean_L2_write_KB_per_replay_instance",
        "L2_projected_throughput_GBps",
        "theoretical_occupancy_upper_bound_pct",
        "weighted_VGPR_count",
        "weighted_shared_memory_size_bytes",
        "strongest_available_stall_proxy",
    )
    record(
        "verified_metrics_available_for_kernel_rows",
        all(
            all(row[field] not in {"", "unavailable"} for field in available_fields)
            for row in kernel_rows
        ),
        {
            field: sum(row[field] == "unavailable" for row in kernel_rows)
            for field in available_fields
        },
    )
    record(
        "matrix_proxy_scope",
        all(
            (
                row["DCU_matrix_core_utilization_proxy_pct"]
                not in {"not_applicable", "unavailable", ""}
            )
            == (row["matched_kernel_family"] == "TunableOp_MMAC_GEMM")
            for row in kernel_rows
        ),
        {
            "mmac_rows": sum(
                row["matched_kernel_family"] == "TunableOp_MMAC_GEMM"
                for row in kernel_rows
            )
        },
    )
    record(
        "live_gfx936_occupancy_binding",
        str(coverage["device"]["gcn_arch_name"]).startswith("gfx936")
        and coverage["device"]["physical_device"] == 1
        and coverage["device"]["unique_id"]
        == run_contract["contract"]["device"]["unique_id"],
        coverage["device"],
    )
    record(
        "no_execution_path_change",
        coverage["unexpected_replay_families_or_execution_paths"] == {}
        and coverage["missing_replay_families_or_execution_paths"] == {},
        {
            "unexpected": coverage[
                "unexpected_replay_families_or_execution_paths"
            ],
            "missing": coverage[
                "missing_replay_families_or_execution_paths"
            ],
        },
    )
    record(
        "process_summary_denominator",
        len(process_rows) == int(expected["process_fragment_targets"])
        and len({(row["event_id"], row["stage"]) for row in process_rows})
        == len(process_rows),
        len(process_rows),
    )

    replay_counter = csv_value_counter(
        root / "hardware_replay_kernel_metrics.csv", "replay_source"
    )
    per_mode_evidence: dict[str, Any] = {}
    for mode in MODES:
        analysis = Path(
            run_contract["replay_bindings"][mode]["analysis_dir"]
        ).resolve()
        record(
            f"{mode}:analysis_scope",
            analysis.is_relative_to(root)
            and "perf_trace_bk" not in analysis.parts,
            str(analysis),
        )
        trace_summary = load_json(analysis / "process_trace_summary.json")
        pmc_summary = load_json(analysis / "hardware_metric_summary.json")
        replay_family_rows = read_csv(
            analysis / "process_launch_owned_kernel_family_order.csv"
        )
        mode_required = (
            "annotations.csv",
            "runtime_calls.csv",
            "kernels.csv",
            "strict_ownership.csv",
            "process_kernel_launch_order.csv",
            "process_launch_owned_kernel_family_order.csv",
            "pmc_blocks.csv",
            "pmc_name_order_matches.csv",
            "hardware_kernel_metrics.csv",
            "discarded_superset_matches.csv",
            "hardware_metric_summary.json",
            "unmatched_pmc_blocks.json",
            "unmatched_trace_kernels.json",
            "unmatched_selected_target_blocks.json",
            "unmatched_selected_trace_kernels.json",
            "ambiguous_pmc_pairs.json",
        )
        files_ok = all(
            (analysis / name).is_file() and (analysis / name).stat().st_size > 0
            for name in mode_required
        )
        empty_lists_ok = all(
            json.loads((analysis / name).read_text(encoding="utf-8")) == []
            for name in (
                "unmatched_pmc_blocks.json",
                "unmatched_trace_kernels.json",
                "unmatched_selected_target_blocks.json",
                "unmatched_selected_trace_kernels.json",
                "ambiguous_pmc_pairs.json",
            )
        )
        summary_ok = (
            trace_summary["status"] == "PASS"
            and pmc_summary["status"] == "PASS"
            and float(pmc_summary["name_order_match_rate"]) >= 0.99
            and int(pmc_summary["unmatched_pmc_block_count"]) == 0
            and int(pmc_summary["unmatched_selected_block_count"]) == 0
            and int(pmc_summary["ambiguous_pair_count"]) == 0
            and int(pmc_summary["covered_selected_target_count"])
            == int(expected["kernel_family_rows"])
            and int(trace_summary["checks"]["process_marker_count"])
            == int(expected["process_fragment_targets"])
        )
        replay_keys = [family_key(row) for row in replay_family_rows]
        replay_by_key = {family_key(row): row for row in replay_family_rows}
        expected_by_key = {family_key(row): row for row in expected_rows}
        family_ok = (
            len(replay_keys) == len(set(replay_keys))
            and set(replay_keys) == set(expected_keys)
        )
        counts_ok = all(
            int(replay_by_key[key]["kernel_family_instance_count"])
            == int(expected_by_key[key]["kernel_family_instance_count"])
            for key in expected_keys
        )
        selection_binding_ok = (
            trace_summary["selection_plan_sha256"]
            == run_contract["selection_plan"]["pre_collection_sha256"]
            and pmc_summary["selection_plan_sha256"]
            == run_contract["selection_plan"]["pre_collection_sha256"]
        )
        record(f"{mode}:normalized_output_set", files_ok)
        record(f"{mode}:unmatched_files_empty", empty_lists_ok)
        record(f"{mode}:strict_summary_pass", summary_ok, pmc_summary)
        record(
            f"{mode}:family_identity_set",
            family_ok,
            {
                "replay_row_order_equals_non_replay": replay_keys
                == expected_keys,
                "row_position_join_used": False,
            },
        )
        record(f"{mode}:family_instance_counts", counts_ok)
        record(f"{mode}:pre_collection_plan_binding", selection_binding_ok)
        record(
            f"{mode}:raw_replay_row_count",
            replay_counter[mode] == int(pmc_summary["strict_owned_metric_rows"]),
            replay_counter[mode],
        )
        per_mode_evidence[mode] = {
            "trace_summary": {
                "path": str(analysis / "process_trace_summary.json"),
                "sha256": sha256_file(
                    analysis / "process_trace_summary.json"
                ),
            },
            "pmc_summary": {
                "path": str(analysis / "hardware_metric_summary.json"),
                "sha256": sha256_file(
                    analysis / "hardware_metric_summary.json"
                ),
            },
            "name_order_match_rate": pmc_summary["name_order_match_rate"],
            "strict_owned_metric_rows": pmc_summary[
                "strict_owned_metric_rows"
            ],
        }

    expected_target_ids = [
        f"{row['event_id']}:{row['stage']}:{row['matched_kernel_family']}"
        for row in expected_rows
    ]
    required_report_labels = (
        "timing_source=workflow02_non_replay_family_row",
        "hardware_join_key=event_id+stage+matched_kernel_family",
        "pmc_replay_timing_used_as_latency=false",
        "theoretical gfx936 resource upper bound",
        "DCU MMAC activity proxy",
        "DRAM throughput is `unavailable`",
    )
    forbidden_report_fields = (
        "kernel_duration_ns_replay_diagnostic_only",
        "request_synchronized_latency_ms",
        "hiptx_cpu_ms",
        "strict_owned_hipops_ms",
        "workflow02_non_replay_family_duration_ms_context",
    )
    for name in (
        "DCU_HARDWARE_METRICS_REPORT.md",
        "SAME_INPUT_PRA_QWEN35_FULL_EAGER_PROCESS_WISE_DCU_REPORT.md",
    ):
        path = root / name
        text = path.read_text(encoding="utf-8")
        ids = report_target_ids(path)
        record(
            f"{name}:one_row_per_expected_family",
            ids == expected_target_ids,
            {"rows": len(ids), "unique": len(set(ids))},
        )
        record(
            f"{name}:timing_and_metric_labels",
            all(label in text for label in required_report_labels),
        )
        record(
            f"{name}:forbidden_timing_fields_absent",
            not any(field in text for field in forbidden_report_fields),
        )

    record(
        "archive_excluded",
        coverage["archive_used_as_current_evidence"] is False
        and "perf_trace_bk"
        not in json.dumps(run_contract, ensure_ascii=False),
    )
    record(
        "strict_join_boundaries",
        coverage["device_timestamp_overlap_attribution_used"] is False
        and coverage["replay_kernel_id_cross_run_join_used"] is False,
    )

    source_files = {
        name: source_root / "scripts" / "perf_trace" / name
        for name in (
            "profile_qwen_same_input_layer.py",
            "run_qwen_hardware_profile_single_request.sh",
            "analyze_qwen_hipprof_process_trace.py",
            "analyze_qwen_hipprof_pmc.py",
            "prepare_qwen_dcu_hardware_plan.py",
            "consolidate_qwen_dcu_hardware_metrics.py",
            "audit_qwen_dcu_hardware_run.py",
            "finalize_qwen_dcu_hardware_handoff.py",
        )
    }
    record(
        "live_source_tools_present",
        all(path.is_file() and "perf_trace_bk" not in path.parts for path in source_files.values()),
    )
    status = "PASS" if not failures else "FAIL"
    audit = {
        "schema_version": 1,
        "runtime_goal": "R04",
        "status": status,
        "completed_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "independent_from_generation": True,
        "failure_checks": failures,
        "checks": checks,
        "derived_denominator": expected,
        "coverage_sha256": sha256_file(root / "hardware_coverage.json"),
        "run_contract_sha256": sha256_file(root / "R04_RUN_CONTRACT.json"),
        "top_level_outputs": top_level,
        "per_mode_evidence": per_mode_evidence,
        "source_tools": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_files.items()
        },
        "evidence_boundary": {
            "timing_source": "workflow02_non_replay_family_row",
            "hardware_join_key": "event_id+stage+matched_kernel_family",
            "pmc_replay_timing_used_as_latency": False,
            "archive_used_as_current_evidence": False,
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
