#!/usr/bin/env python3
"""Validate and write the runtime R04 DCU hardware-evidence handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_GOAL = "R04"
SKILL = "qwen-dcu-process-gpu-hardware-trace"
REPLAY_MODES = ("pmc", "pmc-read", "pmc-write")
REQUIRED_OUTPUTS = (
    "dcu_process_selection_plan.csv",
    "hardware_replay_kernel_metrics.csv",
    "hardware_metrics_by_kernel_family.csv",
    "hardware_metrics.csv",
    "hardware_coverage.json",
    "DCU_HARDWARE_METRICS_REPORT.md",
    "SAME_INPUT_PRA_QWEN35_FULL_EAGER_PROCESS_WISE_DCU_REPORT.md",
)
SOURCE_TOOLS = (
    "profile_qwen_same_input_layer.py",
    "run_qwen_hardware_profile_single_request.sh",
    "analyze_qwen_hipprof_process_trace.py",
    "prepare_qwen_dcu_hardware_plan.py",
    "analyze_qwen_hipprof_pmc.py",
    "consolidate_qwen_dcu_hardware_metrics.py",
    "audit_qwen_dcu_hardware_run.py",
    "finalize_qwen_dcu_hardware_handoff.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--handoff-output", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def csv_ref(path: Path) -> dict[str, Any]:
    result = ref(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        result["rows"] = sum(1 for _ in csv.DictReader(stream))
    return result


def one_glob(root: Path, pattern: str) -> Path:
    values = sorted(root.glob(pattern))
    if len(values) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} under {root}, found {len(values)}"
        )
    return values[0]


def git_value(source_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=source_root, text=True
    ).strip()


def require_relative(child: Path, parent: Path, label: str) -> None:
    if not child.is_relative_to(parent):
        raise RuntimeError(f"{label} escapes required root: {child} vs {parent}")


def provenance_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    handoff_output = args.handoff_output.resolve()

    expected_artifact_root = runtime_root / "artifacts" / RUNTIME_GOAL
    expected_handoff = runtime_root / "handoffs" / f"{RUNTIME_GOAL}.json"
    if artifact_root != expected_artifact_root:
        raise RuntimeError(
            f"artifact root must be {expected_artifact_root}, got {artifact_root}"
        )
    if handoff_output != expected_handoff:
        raise RuntimeError(
            f"handoff output must be {expected_handoff}, got {handoff_output}"
        )
    require_relative(artifact_root, runtime_root, "artifact root")
    require_relative(handoff_output, runtime_root, "handoff output")
    if "perf_trace_bk" in str(artifact_root) or "perf_trace_bk" in str(handoff_output):
        raise RuntimeError("archive path is forbidden as runtime evidence")

    run_contract_path = artifact_root / "R04_RUN_CONTRACT.json"
    coverage_path = artifact_root / "hardware_coverage.json"
    audit_path = artifact_root / "R04_INDEPENDENT_COMPLETION_AUDIT.json"
    run_contract = load_json(run_contract_path)
    coverage = load_json(coverage_path)
    audit = load_json(audit_path)
    if run_contract.get("status") != "hardware_projection_pass":
        raise RuntimeError("R04 run contract is not hardware_projection_pass")
    if coverage.get("status") != "PASS" or coverage.get("failure_reasons"):
        raise RuntimeError("hardware coverage is not a clean PASS")
    if audit.get("status") != "PASS" or audit.get("failure_checks"):
        raise RuntimeError("independent completion audit is not a clean PASS")
    if run_contract.get("runtime_goal") != RUNTIME_GOAL:
        raise RuntimeError("run contract runtime goal mismatch")
    if run_contract["run"]["run_id"] != args.run_id:
        raise RuntimeError("run id mismatch")
    if run_contract["run"]["branch"] != args.branch:
        raise RuntimeError("branch mismatch")
    if Path(run_contract["run"]["runtime_root"]).resolve() != runtime_root:
        raise RuntimeError("run contract runtime root mismatch")
    if Path(run_contract["run"]["runtime_artifact_root"]).resolve() != artifact_root:
        raise RuntimeError("run contract artifact root mismatch")
    if audit.get("run_contract_sha256") != sha256(run_contract_path):
        raise RuntimeError("independent audit/run contract SHA-256 mismatch")
    if audit.get("coverage_sha256") != sha256(coverage_path):
        raise RuntimeError("independent audit/coverage SHA-256 mismatch")

    source_revision = git_value(source_root, "rev-parse", "HEAD")
    source_branch = git_value(source_root, "rev-parse", "--abbrev-ref", "HEAD")
    contract = run_contract["contract"]
    if source_revision != contract["source_revision"]:
        raise RuntimeError("live source revision differs from frozen contract")
    if Path(contract["source_root"]).resolve() != source_root:
        raise RuntimeError("source root mismatch")
    if not Path(contract["model_root"]).absolute().is_relative_to(project_root):
        raise RuntimeError("user model root escapes project root")

    upstream: dict[str, Any] = {}
    for goal in ("R01", "R02", "R03"):
        path = runtime_root / "handoffs" / f"{goal}.json"
        payload = load_json(path)
        if payload.get("runtime_goal") != goal or payload.get("status") != "complete":
            raise RuntimeError(f"{goal} handoff is not complete/current")
        current_ref = ref(path)
        frozen = run_contract["upstream_bindings"][f"{goal.lower()}_handoff"]
        if current_ref["sha256"] != frozen["sha256"]:
            raise RuntimeError(f"{goal} handoff changed after R04 planning")
        upstream[goal] = {
            **current_ref,
            "status": payload["status"],
            "skill": payload["skill"],
        }

    required_outputs: dict[str, Any] = {}
    for name in REQUIRED_OUTPUTS:
        path = artifact_root / name
        required_outputs[name] = csv_ref(path) if path.suffix == ".csv" else ref(path)
    for name, audit_ref in audit["top_level_outputs"].items():
        if required_outputs[name]["sha256"] != audit_ref["sha256"]:
            raise RuntimeError(f"required output changed after independent audit: {name}")

    pre_collection_path = artifact_root / "dcu_process_selection_plan.pre_collection.csv"
    pre_collection_ref = csv_ref(pre_collection_path)
    if (
        pre_collection_ref["sha256"]
        != run_contract["selection_plan"]["pre_collection_sha256"]
    ):
        raise RuntimeError("pre-collection selection plan SHA-256 mismatch")

    replay_collections: dict[str, Any] = {}
    hipprof_path: str | None = None
    hipprof_sha: str | None = None
    for mode in REPLAY_MODES:
        replay_root = artifact_root / "replays" / mode
        analysis_root = replay_root / "analysis"
        metadata_path = one_glob(replay_root, "hipprof_sameinput_*.json")
        runtime_events_path = one_glob(
            replay_root, "hipprof_sameinput_*.layer_events.runtime.jsonl"
        )
        metadata = load_json(metadata_path)
        trace_summary_path = analysis_root / "process_trace_summary.json"
        pmc_summary_path = analysis_root / "hardware_metric_summary.json"
        trace_summary = load_json(trace_summary_path)
        pmc_summary = load_json(pmc_summary_path)
        if trace_summary.get("status") != "PASS" or trace_summary.get("failure_reasons"):
            raise RuntimeError(f"{mode}: strict trace analysis did not pass")
        if pmc_summary.get("status") != "PASS" or pmc_summary.get("failure_reasons"):
            raise RuntimeError(f"{mode}: PMC analysis did not pass")
        if trace_summary["contract_id"] != contract["contract_id"]:
            raise RuntimeError(f"{mode}: contract ID mismatch")
        if trace_summary["contract_sha256"] != contract["canonical_sha256"]:
            raise RuntimeError(f"{mode}: canonical contract SHA-256 mismatch")

        frozen = run_contract["replay_bindings"][mode]
        db_path = Path(frozen["raw_db"]).resolve()
        metrics_path = db_path.with_suffix(".txt")
        provenance_path = replay_root / "tool_provenance.txt"
        provenance = provenance_map(provenance_path)
        if sha256(db_path) != frozen["raw_db_sha256"]:
            raise RuntimeError(f"{mode}: raw DB changed after consolidation")
        if sha256(trace_summary_path) != frozen["trace_summary_sha256"]:
            raise RuntimeError(f"{mode}: trace summary changed after consolidation")
        if sha256(pmc_summary_path) != frozen["pmc_summary_sha256"]:
            raise RuntimeError(f"{mode}: PMC summary changed after consolidation")
        if sha256(provenance_path) != frozen["provenance_sha256"]:
            raise RuntimeError(f"{mode}: provenance changed after consolidation")
        exit_code = int((replay_root / "profile.exit_code").read_text().strip())
        if exit_code != 0 or provenance.get("exit_code") != "0":
            raise RuntimeError(f"{mode}: capture exit code is not zero")
        if provenance.get("profile_kind") != mode or provenance.get("pmc_type") != "0":
            raise RuntimeError(f"{mode}: replay provenance mode/PMC type mismatch")
        if provenance.get("hipprof_device_filter") != "none":
            raise RuntimeError(f"{mode}: forbidden logical device filter was used")
        if hipprof_path is None:
            hipprof_path = provenance["hipprof"]
            hipprof_sha = provenance["hipprof_sha256"]
        elif (
            provenance["hipprof"] != hipprof_path
            or provenance["hipprof_sha256"] != hipprof_sha
        ):
            raise RuntimeError("hipprof executable changed between replay modes")
        if sha256(Path(hipprof_path)) != hipprof_sha:
            raise RuntimeError("live hipprof executable SHA-256 mismatch")

        checks = trace_summary["checks"]
        replay_collections[mode] = {
            "capture_root": str(replay_root),
            "tag": metadata["tag"],
            "profile_exit_code": exit_code,
            "profile_started_utc": provenance["started_utc"],
            "profile_finished_utc": provenance["finished_utc"],
            "pmc_type": 0,
            "physical_device": int(provenance["physical_dcu"]),
            "logical_device": int(provenance["logical_dcu"]),
            "HIP_VISIBLE_DEVICES": provenance["hip_visible_devices"],
            "CUDA_VISIBLE_DEVICES": provenance["cuda_visible_devices"],
            "hipprof_device_filter": provenance["hipprof_device_filter"],
            "metadata": ref(metadata_path),
            "runtime_layer_events": ref(runtime_events_path),
            "raw_queryable_trace": ref(db_path),
            "raw_pmc_metrics": ref(metrics_path),
            "tool_provenance": ref(provenance_path),
            "device_preflight": ref(replay_root / "device_preflight.json"),
            "hipprof_log": ref(replay_root / "hipprof.log"),
            "strict_trace_summary": {
                **ref(trace_summary_path),
                "status": trace_summary["status"],
                "process_markers": checks["process_marker_count"],
                "representative_parent_layers": checks["representative_parent_count"],
                "launch_owning_targets": checks[
                    "launch_owning_process_target_count"
                ],
                "no_kernel_targets": checks["no_kernel_process_row_count"],
                "strict_owned_kernels": checks[
                    "strict_owned_process_kernel_count"
                ],
                "unique_strict_owned_kernels": checks[
                    "unique_strict_owned_process_kernel_count"
                ],
                "strict_owned_device_ids": checks["strict_owned_device_ids"],
                "ownership_rule": trace_summary["ownership_rule"],
            },
            "pmc_match_summary": {
                **ref(pmc_summary_path),
                "status": pmc_summary["status"],
                "matching_rule": pmc_summary["matching_rule"],
                "pmc_blocks": pmc_summary["pmc_block_count"],
                "trace_kernels": pmc_summary["trace_kernel_count"],
                "exact_name_order_matches": pmc_summary[
                    "exact_name_order_matches"
                ],
                "name_order_match_rate": pmc_summary["name_order_match_rate"],
                "strict_owned_metric_rows": pmc_summary[
                    "strict_owned_metric_rows"
                ],
                "unmatched_pmc_blocks": pmc_summary[
                    "unmatched_pmc_block_count"
                ],
                "unmatched_selected_blocks": pmc_summary[
                    "unmatched_selected_block_count"
                ],
                "ambiguous_pairs": pmc_summary["ambiguous_pair_count"],
            },
            "strict_ownership_csv": csv_ref(
                analysis_root / "strict_ownership.csv"
            ),
            "family_order_csv": csv_ref(
                analysis_root / "process_launch_owned_kernel_family_order.csv"
            ),
            "hardware_kernel_metrics_csv": csv_ref(
                analysis_root / "hardware_kernel_metrics.csv"
            ),
            "pmc_name_order_matches_csv": csv_ref(
                analysis_root / "pmc_name_order_matches.csv"
            ),
            "replay_synchronized_latency_ms": metadata[
                "request_synchronized_latency_ms"
            ],
            "replay_synchronized_latency_is_distorted": metadata[
                "request_synchronized_latency_is_replay_distorted"
            ],
            "measured_result": {
                "prompt_token_count": metadata["measured_result"][
                    "prompt_token_count"
                ],
                "output_token_count": metadata["measured_result"][
                    "output_token_count"
                ],
                "prompt_token_ids_sha256": metadata["measured_result"][
                    "prompt_token_ids_sha256"
                ],
                "output_token_ids_sha256": metadata["measured_result"][
                    "output_token_ids_sha256"
                ],
                "output_text_sha256": metadata["measured_result"][
                    "output_text_sha256"
                ],
            },
        }

    baseline_results = {
        (
            entry["measured_result"]["prompt_token_count"],
            entry["measured_result"]["output_token_count"],
            entry["measured_result"]["prompt_token_ids_sha256"],
            entry["measured_result"]["output_token_ids_sha256"],
            entry["measured_result"]["output_text_sha256"],
        )
        for entry in replay_collections.values()
    }
    if len(baseline_results) != 1:
        raise RuntimeError("replay modes produced different SAME_INPUT results")

    scripts_root = source_root / "scripts" / "perf_trace"
    source_tools = {name: ref(scripts_root / name) for name in SOURCE_TOOLS}
    non_replay_ledger = Path(
        run_contract["upstream_bindings"]["non_replay_family_ledger"]["path"]
    ).resolve()
    non_replay_summary = Path(
        run_contract["upstream_bindings"]["non_replay_trace_summary"]["path"]
    ).resolve()
    if sha256(non_replay_ledger) != run_contract["upstream_bindings"][
        "non_replay_family_ledger"
    ]["sha256"]:
        raise RuntimeError("current non-replay family ledger changed")
    if sha256(non_replay_summary) != run_contract["upstream_bindings"][
        "non_replay_trace_summary"
    ]["sha256"]:
        raise RuntimeError("current non-replay trace summary changed")

    completed_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    handoff: dict[str, Any] = {
        "schema_version": 1,
        "runtime_goal": RUNTIME_GOAL,
        "status": "complete",
        "skill": SKILL,
        "run": {
            "branch": args.branch,
            "run_id": args.run_id,
            "runtime_root": str(runtime_root),
            "runtime_artifact_root": str(artifact_root),
            "completed_utc": completed_utc,
            "model_inference_replays": 3,
            "rocm_dcu_hip_profiler_replays": 3,
        },
        "workflow": {
            "role": "project live hipprof PMC replay diagnostics projected onto the current Workflow-02 process/family denominator",
            "replay_modes": list(REPLAY_MODES),
            "pmc_type": 0,
            "selection_mode": run_contract["selection_plan"]["selection_mode"],
            "selection_filtered": run_contract["selection_plan"]["filtered"],
            "archive_used_as_current_evidence": False,
        },
        "same_input_parent": {
            "contract_id": contract["contract_id"],
            "contract_path": contract["path"],
            "contract_file_sha256": contract["file_sha256"],
            "contract_canonical_sha256": contract["canonical_sha256"],
            "source_revision": contract["source_revision"],
            "model_root": contract["model_root"],
            "resolved_model_root": contract["resolved_model_root"],
            "served_model_name": contract["served_model_name"],
            "config": contract["config"],
            "same_input": contract["same_input"],
            "device": contract["device"],
            "upstream_handoffs": upstream,
        },
        "current_denominator": {
            **run_contract["expected_denominator"],
            "non_replay_family_ledger": csv_ref(non_replay_ledger),
            "non_replay_trace_summary": ref(non_replay_summary),
            "selection_plan_before_collection": pre_collection_ref,
            "selection_plan_after_collection": required_outputs[
                "dcu_process_selection_plan.csv"
            ],
        },
        "device": coverage["device"],
        "live_toolchain": {
            "hipprof": {
                "path": hipprof_path,
                "sha256": hipprof_sha,
            },
            "cxxfilt": {
                "path": "/usr/bin/c++filt",
                "sha256": sha256(Path("/usr/bin/c++filt")),
            },
            "source_root": str(source_root),
            "source_revision": source_revision,
            "source_branch": source_branch,
            "source_status_boundary": "intentional untracked scripts/perf_trace R04 tooling plus the existing project-local instrumentation overlay",
            "source_tools": source_tools,
        },
        "replay_collections": replay_collections,
        "strict_ownership": {
            "rule": "process HIPTX range -> fully contained HIP runtime call within marker index bounds -> HIPOPS identical source DB/config/pid/_Index",
            "device_timestamp_overlap_attribution_used": False,
            "kernel_name_or_nearest_timestamp_attribution_used": False,
            "multiply_owned_selected_kernels": 0,
            "per_mode_strict_owned_kernels": {
                mode: replay_collections[mode]["strict_trace_summary"][
                    "strict_owned_kernels"
                ]
                for mode in REPLAY_MODES
            },
        },
        "pmc_name_order_matching": {
            "rule": "same profiler PID + exact dispatch-order position + exact c++filt demangled kernel name",
            "minimum_global_match_rate": 0.99,
            "fuzzy_name_matching_used": False,
            "row_position_family_join_used": False,
            "cross_run_kernel_id_join_used": False,
            "per_mode": {
                mode: {
                    "pmc_blocks": replay_collections[mode][
                        "pmc_match_summary"
                    ]["pmc_blocks"],
                    "exact_matches": replay_collections[mode][
                        "pmc_match_summary"
                    ]["exact_name_order_matches"],
                    "match_rate": replay_collections[mode][
                        "pmc_match_summary"
                    ]["name_order_match_rate"],
                    "unmatched": replay_collections[mode][
                        "pmc_match_summary"
                    ]["unmatched_pmc_blocks"],
                    "ambiguous": replay_collections[mode][
                        "pmc_match_summary"
                    ]["ambiguous_pairs"],
                }
                for mode in REPLAY_MODES
            },
        },
        "primary_outputs": {
            **required_outputs,
            "run_contract": ref(run_contract_path),
            "independent_completion_audit": {
                **ref(audit_path),
                "status": audit["status"],
                "failure_checks": audit["failure_checks"],
            },
        },
        "validation": {
            "status": "pass",
            "hardware_coverage": {
                **ref(coverage_path),
                "status": coverage["status"],
                "family_join_coverage_pct": coverage[
                    "family_join_coverage_pct"
                ],
                "complete_kernel_family_rows": coverage["complete_rows"],
                "no_kernel_rows": coverage["no_kernel_rows"],
                "partial_rows": coverage["partial_rows"],
                "missing_rows": coverage["missing_rows"],
            },
            "independent_audit_status": audit["status"],
            "independent_audit_failure_checks": audit["failure_checks"],
            "all_required_reports_preserve_one_row_per_family": True,
            "all_replay_outputs_identical_to_same_input_baseline": True,
        },
        "same_run_binding": {
            "contract_id": contract["contract_id"],
            "contract_canonical_sha256": contract["canonical_sha256"],
            "r02_non_replay_db_sha256": run_contract["upstream_bindings"][
                "r02_non_replay_db"
            ]["sha256"],
            "r02_inventory_sha256": run_contract["upstream_bindings"][
                "r02_inventory"
            ]["sha256"],
            "non_replay_family_ledger_sha256": sha256(non_replay_ledger),
            "pre_collection_selection_plan_sha256": pre_collection_ref[
                "sha256"
            ],
            "final_selection_plan_sha256": required_outputs[
                "dcu_process_selection_plan.csv"
            ]["sha256"],
            "replay_raw_db_sha256": {
                mode: replay_collections[mode]["raw_queryable_trace"]["sha256"]
                for mode in REPLAY_MODES
            },
            "replay_raw_pmc_metrics_sha256": {
                mode: replay_collections[mode]["raw_pmc_metrics"]["sha256"]
                for mode in REPLAY_MODES
            },
            "hardware_family_output_sha256": required_outputs[
                "hardware_metrics_by_kernel_family.csv"
            ]["sha256"],
            "independent_audit_sha256": sha256(audit_path),
        },
        "downstream_consumption": {
            "kernel_family_metrics_path": str(
                artifact_root / "hardware_metrics_by_kernel_family.csv"
            ),
            "process_fragment_metrics_path": str(
                artifact_root / "hardware_metrics.csv"
            ),
            "replay_kernel_metrics_path": str(
                artifact_root / "hardware_replay_kernel_metrics.csv"
            ),
            "coverage_path": str(coverage_path),
            "primary_report_path": str(
                artifact_root
                / "SAME_INPUT_PRA_QWEN35_FULL_EAGER_PROCESS_WISE_DCU_REPORT.md"
            ),
            "hardware_join_key": [
                "event_id",
                "stage",
                "matched_kernel_family",
            ],
            "consumer_gate": "require this exact contract canonical SHA-256, R02 non-replay DB SHA-256, non-replay family ledger SHA-256, all three replay DB/PMC hashes, final family output SHA-256, and PASS independent audit SHA-256",
            "pmc_replay_duration_is_latency_evidence": False,
            "timing_source": coverage["timing_source"],
        },
        "evidence_boundary": {
            "establishes": "complete current-denominator gfx936 hipprof PMC diagnostics for all 119 launch-owning kernel-family rows plus 18 explicit no-kernel rows",
            "does_not_establish": "direct process latency from replay duration, unavailable DRAM bandwidth, or strict hardware causality beyond the exposed counters/proxies",
            "pmc_replay_timing_used_as_latency": False,
            "pmc_is_latency_evidence": False,
            "timing_source": coverage["timing_source"],
            "hardware_join_key": coverage["hardware_join_key"],
            "dram_unavailable_not_inferred": True,
            "matrix_activity_is_mmac_scoped_proxy": True,
            "occupancy_is_theoretical_upper_bound": True,
            "device_timestamp_overlap_attribution_used": False,
            "expected_kernel_family_hypotheses_used_as_measurements": False,
            "archive_backup_used_as_current_evidence": False,
        },
        "handoff_output": str(handoff_output),
    }

    serialized = json.dumps(
        handoff, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    if "perf_trace_bk" in serialized:
        raise RuntimeError("handoff unexpectedly references perf_trace_bk")
    handoff_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = handoff_output.with_name(
        f".{handoff_output.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(handoff_output)
    print(json.dumps(handoff, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
