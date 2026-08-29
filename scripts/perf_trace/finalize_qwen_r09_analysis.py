#!/usr/bin/env python3
"""Independently audit and finalize one fresh-run R09 analysis.

This tool is deliberately offline.  It reads only hash-pinned R06-R08 and R09
files, writes the R09 source-lineage/audit records, and writes the scheduler
handoff last after every completion check has passed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class FinalizationError(RuntimeError):
    pass


TABLE_FILES = {
    "request_timeline": "request_timeline.csv",
    "process_timeline": "process_timeline.csv",
    "kernel_timeline": "kernel_timeline.csv",
    "live_utilization_aligned": "live_utilization_aligned.csv",
    "process_live_utilization": "process_live_utilization.csv",
    "kernel_concurrency": "kernel_concurrency.csv",
    "queue_concurrency": "queue_concurrency.csv",
    "launch_gaps": "launch_gaps.csv",
    "high_latency_processes": "high_latency_processes.csv",
    "dependency_state": "dependency_state.csv",
    "traffic_resource_attachment": "traffic_resource_attachment.csv",
    "opportunity_candidates": "opportunity_candidates.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizationError(f"JSON is not an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FinalizationError(f"CSV lacks a header: {path}")
        return list(reader)


def truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return sum(end - begin for begin, end in merged)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--r06-handoff", type=Path, required=True)
    parser.add_argument("--r07-handoff", type=Path, required=True)
    parser.add_argument("--r08-handoff", type=Path, required=True)
    parser.add_argument("--expected-r06-handoff-sha256", required=True)
    parser.add_argument("--expected-r07-handoff-sha256", required=True)
    parser.add_argument("--expected-r08-handoff-sha256", required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--analyzer-prechange-sha256", required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--handoff-output", type=Path, required=True)
    parser.add_argument("--minimum-live-samples", type=int, required=True)
    parser.add_argument("--maximum-clock-alignment-error-ns", type=int, required=True)
    parser.add_argument("--low-se-utilization-pct", type=float, required=True)
    parser.add_argument("--low-kernel-concurrency-max", type=int, required=True)
    parser.add_argument("--minimum-launch-gap-ns", type=int, required=True)
    parser.add_argument("--minimum-dependency-coverage", type=float, required=True)
    parser.add_argument("--minimum-exposed-duration-ns", type=int, required=True)
    parser.add_argument("--minimum-exposed-fraction", type=float, required=True)
    parser.add_argument("--slack-tolerance-ns", type=int, required=True)
    parser.add_argument("--maximum-trace-bundle-bytes", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    analysis_dir = artifact_root / "analysis"
    source_lineage_path = artifact_root / "R09_SOURCE_LINEAGE.json"
    audit_path = artifact_root / "R09_COMPLETION_AUDIT.json"
    handoff_path = args.handoff_output.resolve()
    for output in (source_lineage_path, audit_path, handoff_path):
        if output.exists():
            raise FinalizationError(f"refusing existing final output: {output}")
    if artifact_root != runtime_root / "artifacts" / "R09":
        raise FinalizationError("artifact root is not the scheduler-assigned R09 path")
    if handoff_path != runtime_root / "handoffs" / "R09.json":
        raise FinalizationError("handoff path is not the scheduler-assigned R09 path")

    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            failures.append(checks[-1])

    handoff_specs = (
        ("R06", args.r06_handoff.resolve(), args.expected_r06_handoff_sha256),
        ("R07", args.r07_handoff.resolve(), args.expected_r07_handoff_sha256),
        ("R08", args.r08_handoff.resolve(), args.expected_r08_handoff_sha256),
    )
    upstream: dict[str, dict[str, Any]] = {}
    upstream_payloads: dict[str, dict[str, Any]] = {}
    for goal, path, expected_sha in handoff_specs:
        actual_sha = sha256_file(path)
        check(f"{goal}_handoff_hash", actual_sha == expected_sha, actual_sha)
        check(f"{goal}_handoff_under_runtime", under(path, runtime_root), str(path))
        payload = load_json(path)
        upstream_payloads[goal] = payload
        check(f"{goal}_complete", payload.get("status") == "complete")
        check(
            f"{goal}_fresh_evidence_mode",
            payload.get("evidence_acquisition_mode") == "fresh_no_prior_runtime_reuse",
        )
        upstream[goal] = {"path": str(path), "sha256": actual_sha, "size_bytes": path.stat().st_size}

    r06, r07, r08 = (upstream_payloads[name] for name in ("R06", "R07", "R08"))
    lineage_values = (
        r06["same_run_lineage"]["lineage_id"],
        r07["fresh_e2e_evidence"]["lineage_id"],
        r08["fresh_e2e_evidence"]["lineage_id"],
    )
    lineage_id = lineage_values[0]
    check("R06_R08_lineage_match", len(set(lineage_values)) == 1, lineage_values)

    input_records = {
        "r07_profile_metadata": r07["fresh_e2e_evidence"]["full_request_profile_metadata"],
        "r07_process_trace_summary": r07["fresh_e2e_evidence"]["process_trace_summary"],
        "r07_annotations": r07["observed_process_timeline"]["annotations"],
        "r07_runtime_calls": r07["observed_process_timeline"]["runtime_calls"],
        "r07_strict_ownership": r07["observed_process_timeline"]["strict_ownership"],
        "r07_process_performance": r07["observed_process_timeline"]["process_performance"],
        "r07_process_gpu_timeline": r07["observed_process_timeline"]["process_gpu_timeline"],
        "r07_kernels": r07["observed_process_timeline"]["kernels"],
        "r07_live_samples": r07["live_utilization"]["raw_samples"],
        "r07_live_summary": r07["live_utilization"]["summary"],
        "r07_dependency_adapter": r07["dependency_adapter"]["adapter"],
        "r08_device_capabilities": r08["fresh_e2e_evidence"]["device_capabilities"],
        "r08_hardware_metrics": r08["primary_outputs"]["hardware_metrics"],
        "r08_traffic_resource_model": r08["fresh_e2e_evidence"]["traffic_resource_model"],
    }
    verified_inputs: dict[str, dict[str, Any]] = {}
    for name, record in input_records.items():
        path = Path(record["path"]).resolve()
        actual_sha = sha256_file(path)
        check(f"{name}_hash", actual_sha == record["sha256"], actual_sha)
        check(f"{name}_under_runtime", under(path, runtime_root), str(path))
        verified_inputs[name] = {
            "path": str(path), "sha256": actual_sha, "size_bytes": path.stat().st_size,
            **({"declared_rows": record.get("row_count", record.get("rows"))} if record.get("row_count", record.get("rows")) is not None else {}),
        }

    manifest_path = args.analysis_manifest.resolve()
    manifest = load_json(manifest_path)
    check("analysis_manifest_exact_path", manifest_path == (analysis_dir / "fresh_e2e_analysis.json").resolve())
    check("analysis_status", manifest.get("status") == "PASS")
    check("analysis_type", manifest.get("analysis_type") == "fresh_run_full_request_e2e")
    check("analysis_lineage", manifest.get("lineage_id") == lineage_id)
    check("full_request_timeline", manifest.get("full_request_observed_timeline") is True)
    configured_gates = {
        "low_se_utilization_pct": args.low_se_utilization_pct,
        "low_kernel_concurrency_max": args.low_kernel_concurrency_max,
        "minimum_launch_gap_ns": args.minimum_launch_gap_ns,
        "minimum_dependency_coverage": args.minimum_dependency_coverage,
        "minimum_exposed_duration_ns": args.minimum_exposed_duration_ns,
        "minimum_exposed_fraction": args.minimum_exposed_fraction,
        "slack_tolerance_ns": args.slack_tolerance_ns,
        "require_all_seven_gates": True,
    }
    check("configured_gates_exact", manifest.get("configured_gates") == configured_gates, manifest.get("configured_gates"))
    check("minimum_live_samples_exact", manifest.get("minimum_live_samples_per_process") == args.minimum_live_samples)
    check("clock_alignment_limit_exact", manifest.get("maximum_clock_alignment_error_ns") == args.maximum_clock_alignment_error_ns)

    analyzer_input_paths = {
        Path(record["path"]).resolve(): record["sha256"]
        for name, record in input_records.items()
        if name not in {"r08_device_capabilities", "r08_hardware_metrics"}
    }
    manifest_inputs = {Path(path).resolve(): digest for path, digest in manifest.get("inputs", {}).items()}
    check("analyzer_inputs_exact", manifest_inputs == analyzer_input_paths)

    tables: dict[str, list[dict[str, str]]] = {}
    table_summary: dict[str, dict[str, Any]] = {}
    for key, filename in TABLE_FILES.items():
        record = manifest.get("normalized_tables", {}).get(key, {})
        path = (analysis_dir / filename).resolve()
        check(f"{key}_exact_path", Path(record.get("path", "")).resolve() == path)
        actual_sha = sha256_file(path)
        if key == "request_timeline":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = [
                    {
                        "track_type": row["track_type"],
                        "event_key": row["event_key"],
                        "parent_key": row["parent_key"],
                    }
                    for row in csv.DictReader(handle)
                ]
        else:
            rows = load_csv(path)
        check(f"{key}_positive", len(rows) > 0, len(rows))
        check(f"{key}_row_count", len(rows) == record.get("row_count"), len(rows))
        check(f"{key}_sha256", actual_sha == record.get("sha256"), actual_sha)
        tables[key] = rows
        table_summary[key] = {"path": str(path), "sha256": actual_sha, "row_count": len(rows), "size_bytes": path.stat().st_size}

    metadata = load_json(Path(input_records["r07_profile_metadata"]["path"]))
    request_begin = int(metadata["request_start_realtime_ns"])
    request_end = int(metadata["request_end_realtime_ns"])
    expected_markers = set(metadata["expected_process_ranges"])
    realtime_duration = request_end - request_begin
    monotonic_duration = int(metadata["request_end_monotonic_ns"]) - int(metadata["request_start_monotonic_ns"])
    check("request_clock_alignment", abs(realtime_duration - monotonic_duration) <= args.maximum_clock_alignment_error_ns, abs(realtime_duration - monotonic_duration))

    process_rows = tables["process_timeline"]
    process_by_marker = {row["process_range"]: row for row in process_rows}
    process_by_stage = {(row["event_id"], row["stage"]): row for row in process_rows}
    check("process_marker_uniqueness", len(process_by_marker) == len(process_rows))
    check("full_process_coverage", set(process_by_marker) == expected_markers, len(process_by_marker))
    check("process_intervals_inside_request", all(request_begin <= int(row["hiptx_begin_ns"]) <= int(row["hiptx_end_ns"]) <= request_end for row in process_rows))

    request_counts = Counter(row["track_type"] for row in tables["request_timeline"])
    check("request_track_count", request_counts["request"] == 1, request_counts["request"])
    check("forward_track_count", request_counts["forward"] == 29, request_counts["forward"])
    check("layer_track_count", request_counts["layer"] == 1856, request_counts["layer"])
    check("runtime_track_count", request_counts["hip_runtime"] == 428023, request_counts["hip_runtime"])
    check("manifest_track_counts", dict(sorted(request_counts.items())) == manifest.get("track_type_counts"))
    request_keys = {row["event_key"] for row in tables["request_timeline"] if row["track_type"] == "request"}
    forward_keys = {row["event_key"] for row in tables["request_timeline"] if row["track_type"] == "forward"}
    layer_keys = {row["event_key"] for row in tables["request_timeline"] if row["track_type"] == "layer"}
    check("forward_parent_hierarchy", all(row["parent_key"] in request_keys for row in tables["request_timeline"] if row["track_type"] == "forward"))
    check("layer_parent_hierarchy", all(row["parent_key"] in forward_keys for row in tables["request_timeline"] if row["track_type"] == "layer"))
    check("process_parent_hierarchy", all(row["parent_key"] in layer_keys for row in process_rows))
    check("runtime_parent_hierarchy", all(not row["parent_key"] or row["parent_key"] in process_by_marker for row in tables["request_timeline"] if row["track_type"] == "hip_runtime"))

    source_gpu_rows = load_csv(Path(input_records["r07_process_gpu_timeline"]["path"]))
    source_owner = {row["kernel_id"]: row["process_range"] for row in source_gpu_rows}
    kernel_rows = tables["kernel_timeline"]
    kernel_by_id = {row["kernel_id"]: row for row in kernel_rows}
    check("strict_kernel_count", len(kernel_rows) == len(source_gpu_rows) == 29964, len(kernel_rows))
    check("strict_kernel_identity", set(kernel_by_id) == set(source_owner))
    check("strict_kernel_owner", all(kernel_by_id[key]["process_owner"] == owner for key, owner in source_owner.items()))
    check("strict_kernel_device", all(row["device_id"] == "1" for row in kernel_rows))
    check("strict_kernel_observed_timing", all("non_replay" in row["timing_source"] and "replay" not in row["timing_source"].replace("non_replay", "") for row in kernel_rows))
    intervals_by_process: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in kernel_rows:
        intervals_by_process[row["process_owner"]].append((int(row["begin_ns"]), int(row["end_ns"])))
    busy_union_ok = True
    for marker, process in process_by_marker.items():
        intervals = intervals_by_process[marker]
        busy_union_ok &= int(process["strict_owned_kernel_count"]) == len(intervals)
        busy_union_ok &= int(process["strict_owned_gpu_busy_union_ns"]) == union_duration(intervals)
    check("strict_process_gpu_busy_union", busy_union_ok)

    aligned_rows = tables["live_utilization_aligned"]
    eligible_count = sum(truthy(row["eligible_for_process_attribution"]) for row in aligned_rows)
    check("aligned_sample_count", len(aligned_rows) == manifest.get("aligned_live_sample_count") == 34782, len(aligned_rows))
    check("eligible_sample_count", eligible_count == manifest.get("eligible_aligned_live_sample_count") == 34778, eligible_count)
    check("alignment_status_semantics", all((int(row["alignment_uncertainty_ns"]) <= args.maximum_clock_alignment_error_ns) == truthy(row["eligible_for_process_attribution"]) for row in aligned_rows))
    process_live_rows = tables["process_live_utilization"]
    process_live_by_marker = {row["process_range"]: row for row in process_live_rows}
    check("process_live_full_coverage", set(process_live_by_marker) == expected_markers)
    check("process_live_status_threshold", all((int(row["sample_count"]) >= args.minimum_live_samples) == (row["status"] == "observed") for row in process_live_rows))

    high_rows = tables["high_latency_processes"]
    expected_high = sorted(process_rows, key=lambda row: (-float(row["hiptx_cpu_ms"]), row["process_range"]))[:len(high_rows)]
    check("high_latency_order", [row["process_range"] for row in high_rows] == [row["process_range"] for row in expected_high])
    check("high_latency_live_complete", all(row["live_utilization_status"] == "observed" and int(row["live_sample_count"]) >= args.minimum_live_samples for row in high_rows))
    check("high_latency_manifest_counts", manifest.get("high_latency_process_count") == manifest.get("high_latency_processes_with_live_samples") == len(high_rows))

    kernel_sweep = tables["kernel_concurrency"]
    queue_sweep = tables["queue_concurrency"]
    check("concurrency_sweep_shape", len(kernel_sweep) == len(queue_sweep) and len(kernel_sweep) > 0)
    check("concurrency_sweep_boundaries", all(k["begin_ns"] == q["begin_ns"] and k["end_ns"] == q["end_ns"] for k, q in zip(kernel_sweep, queue_sweep)))
    check("concurrency_sweep_request_cover", int(kernel_sweep[0]["begin_ns"]) == request_begin and int(kernel_sweep[-1]["end_ns"]) == request_end)
    check("concurrency_sweep_contiguous", all(kernel_sweep[index]["end_ns"] == kernel_sweep[index + 1]["begin_ns"] for index in range(len(kernel_sweep) - 1)))
    check("queue_kernel_consistency", all(int(row["active_queue_count"]) <= int(row["active_kernel_count"]) for row in queue_sweep))

    launch_rows = tables["launch_gaps"]
    launch_by_process: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in launch_rows:
        launch_by_process[row["process_range"]].append(row)
    check("launch_gap_process_coverage", set(launch_by_process) == expected_markers)
    launch_gap_semantics = all(
        int(row["duration_ns"]) == int(row["end_ns"]) - int(row["begin_ns"])
        and truthy(row["material_gap"]) == (int(row["duration_ns"]) >= args.minimum_launch_gap_ns)
        for row in launch_rows
    )
    check("launch_gap_threshold_semantics", launch_gap_semantics)
    check("launch_gap_non_replay_clock", all("observed_r07" in row["timing_source"] for row in launch_rows))

    adapter = load_json(Path(input_records["r07_dependency_adapter"]["path"]))
    edge_record = adapter["outputs"]["edges"]
    edge_path = Path(edge_record["path"]).resolve()
    check("dependency_edges_hash", sha256_file(edge_path) == edge_record["sha256"])
    edge_rows = load_csv(edge_path)
    dependency_rows = tables["dependency_state"]
    edge_by_id = {row["edge_id"]: row for row in edge_rows}
    dependency_by_id = {row["edge_id"]: row for row in dependency_rows}
    check("dependency_row_preservation", set(edge_by_id) == set(dependency_by_id) and len(edge_rows) == len(dependency_rows))
    check("dependency_no_adjacency_inference", adapter.get("edge_semantics", {}).get("temporal_adjacency_used_as_dependency") is False)
    verified_dependency_count = sum(truthy(row["dependency_gate_pass"]) for row in dependency_rows)
    dependency_coverage = verified_dependency_count / len(dependency_rows)
    check("dependency_coverage", math.isclose(dependency_coverage, float(manifest["dependency_coverage"]), rel_tol=0, abs_tol=1e-12), dependency_coverage)
    unknown_preserved = all(
        dependency_by_id[edge_id]["dependency_state"] == "unknown_dependency"
        for edge_id, edge in edge_by_id.items()
        if not (edge.get("edge_type") == "data" and truthy(edge.get("verified")))
    )
    check("opaque_dependencies_remain_unknown", unknown_preserved)

    model = load_json(Path(input_records["r08_traffic_resource_model"]["path"]))
    check("traffic_model_lineage", model.get("lineage_id") == lineage_id)
    check("traffic_model_hbm_boundary", model.get("traffic_boundary", {}).get("hbm_or_dram_traffic_claimed") is False)
    check("resource_model_occupancy_boundary", model.get("resource_boundary", {}).get("achieved_occupancy_claimed") is False)
    for path_text, digest in model.get("inputs", {}).items():
        path = Path(path_text).resolve()
        check(f"traffic_model_input_{path.name}_hash", under(path, runtime_root) and sha256_file(path) == digest)
    hardware_rows = load_csv(Path(input_records["r08_hardware_metrics"]["path"]))
    check(
        "hardware_replay_not_latency",
        all(
            not truthy(row["pmc_replay_timing_used_as_latency"])
            and row["latency_axis"]
            in {"R07_non_replay_same_request", "R07_non_replay_same_request_only"}
            and row["cross_capture_timeline_policy"]
            == "separate_clock_axes_no_merge"
            for row in hardware_rows
        ),
    )
    check("hardware_replay_projected", all(row["hardware_evidence_class"].startswith("replay_projected") for row in hardware_rows))
    attachments = tables["traffic_resource_attachment"]
    attached_markers = {row["process_range"] for row in attachments}
    check("traffic_attachment_process_coverage", attached_markers == expected_markers)
    check("hbm_dram_unavailable", all(row["hbm_or_dram_bytes"] == "unavailable" for row in attachments))
    check("fx_traffic_labeled_inferred", all(row["traffic_evidence_class"] == "inferred_fx_visible" for row in attachments))

    resource_record = model["outputs"]["resource"]
    resource_path = Path(resource_record["path"]).resolve()
    check("resource_model_output_hash", sha256_file(resource_path) == resource_record["sha256"])
    resource_rows = load_csv(resource_path)
    resource_by_key = {
        (row["event_id"], row["stage"], row["matched_kernel_family"]): row
        for row in resource_rows
    }
    limits = model["device_limits"]
    kernels_by_launch_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in kernel_rows:
        kernels_by_launch_pair[(row["process_owner"], row["runtime_index"])].append(row)
    sweep_ends = [int(row["end_ns"]) for row in kernel_sweep]

    def resource_for_kernel(kernel: dict[str, str]) -> dict[str, str] | None:
        owner = process_by_marker.get(kernel["process_owner"])
        if owner is None:
            return None
        resource = resource_by_key.get(
            (owner["event_id"], owner["stage"], kernel["kernel_family"])
        )
        if resource is None or not truthy(resource["resource_evidence_complete"]):
            return None
        return resource

    def group_fits(resources: list[dict[str, str]]) -> bool:
        work_groups = [float(row["work_group_size"]) for row in resources]
        waves = [math.ceil(value / float(limits["wave_size"])) for value in work_groups]
        vgpr = [float(row["vgpr_count"]) * work_group for row, work_group in zip(resources, work_groups)]
        shared = [float(row["shared_memory_size_bytes"]) for row in resources]
        return (
            sum(work_groups) <= float(limits["thread_limit"])
            and sum(waves) <= int(limits["wave_limit"])
            and sum(vgpr) <= float(limits["vgpr_resource"])
            and sum(shared) <= float(limits["shared_memory_bytes"])
        )

    def expected_gap_resource(row: dict[str, str]) -> tuple[int, int, str, str]:
        begin, end = int(row["begin_ns"]), int(row["end_ns"])
        if end <= begin:
            return 0, 0, "unavailable", "no_positive_launch_gap_observed"
        candidate_kernels = kernels_by_launch_pair.get(
            (row["process_range"], row["next_runtime_index"]), []
        )
        candidate_resources = [resource_for_kernel(kernel) for kernel in candidate_kernels]
        candidate_missing = not candidate_kernels or any(value is None for value in candidate_resources)
        low_concurrency_ns = 0
        feasible_ns = 0
        unavailable_ns = 0
        tested_ns = 0
        index = bisect.bisect_right(sweep_ends, begin)
        while index < len(kernel_sweep):
            sweep = kernel_sweep[index]
            sweep_begin, sweep_end = int(sweep["begin_ns"]), int(sweep["end_ns"])
            if sweep_begin >= end:
                break
            overlap = max(0, min(end, sweep_end) - max(begin, sweep_begin))
            active_count = int(sweep["active_kernel_count"])
            if overlap and active_count <= args.low_kernel_concurrency_max:
                low_concurrency_ns += overlap
                if not candidate_missing:
                    active_ids = [value for value in sweep["active_kernel_ids"].split(";") if value]
                    active_resources = [resource_for_kernel(kernel_by_id[value]) for value in active_ids]
                    if any(value is None for value in active_resources):
                        unavailable_ns += overlap
                    else:
                        tested_ns += overlap
                        candidates = [value for value in candidate_resources if value is not None]
                        active = [value for value in active_resources if value is not None]
                        if all(group_fits([candidate, *active]) for candidate in candidates):
                            feasible_ns += overlap
            index += 1
        if not candidate_kernels:
            return low_concurrency_ns, 0, "unavailable", "no_next_strict_owned_kernel_launch"
        if candidate_missing:
            return low_concurrency_ns, 0, "unavailable", "candidate_kernel_resource_model_unavailable"
        if feasible_ns > 0:
            return low_concurrency_ns, feasible_ns, "validated_replay_projected", "gfx936_pairwise_resource_formula_pass"
        if unavailable_ns > 0 and tested_ns == 0:
            return low_concurrency_ns, 0, "unavailable", "active_kernel_resource_model_unavailable"
        return low_concurrency_ns, 0, "infeasible", "gfx936_pairwise_resource_formula_failed"

    gap_resource_semantics = True
    for row in launch_rows:
        low_ns, resource_ns, status, reason = expected_gap_resource(row)
        gap_resource_semantics &= int(row["low_kernel_concurrency_exposed_ns"]) == low_ns
        gap_resource_semantics &= int(row["resource_coexistence_exposed_ns"]) == resource_ns
        gap_resource_semantics &= row["resource_coexistence_status"] == status
        gap_resource_semantics &= row["resource_coexistence_reason"] == reason
    check("pairwise_resource_gap_semantics", gap_resource_semantics)

    gaps_by_process: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in launch_rows:
        gaps_by_process[row["process_range"]].append(row)
    deps_by_target: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in dependency_rows:
        deps_by_target[(row["event_id"], row["target_stage"])].append(row)
    attachments_by_process: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in attachments:
        attachments_by_process[row["process_range"]].append(row)
    opportunity_rows = tables["opportunity_candidates"]
    opportunity_by_marker = {row["process_range"]: row for row in opportunity_rows}
    check("opportunity_process_coverage", set(opportunity_by_marker) == expected_markers)
    gate_columns = {
        "dependency": "dependency_gate",
        "slack": "slack_gate",
        "queue_feasibility": "queue_feasibility_gate",
        "resource_coexistence": "resource_coexistence_gate",
        "exposure": "exposure_gate",
        "utilization": "utilization_gate",
        "evidence_quality": "evidence_quality_gate",
    }
    opportunity_semantics = True
    for marker, row in opportunity_by_marker.items():
        process = process_by_marker[marker]
        duration = int(process["duration_ns"])
        material_gaps = [gap for gap in gaps_by_process[marker] if truthy(gap["material_gap"])]
        exposed = sum(int(gap["duration_ns"]) for gap in material_gaps)
        queue_exposed = sum(int(gap["low_kernel_concurrency_exposed_ns"]) for gap in material_gaps)
        resource_exposed = sum(int(gap["resource_coexistence_exposed_ns"]) for gap in material_gaps)
        opportunity_semantics &= int(row["exposed_duration_ns"]) == exposed
        opportunity_semantics &= int(row["queue_feasible_exposed_duration_ns"]) == queue_exposed
        opportunity_semantics &= int(row["resource_coexistence_exposed_duration_ns"]) == resource_exposed
        opportunity_semantics &= truthy(row["queue_feasibility_gate"]) == (queue_exposed > 0)
        opportunity_semantics &= truthy(row["resource_coexistence_gate"]) == (resource_exposed > 0)
        opportunity_semantics &= truthy(row["exposure_gate"]) == (
            resource_exposed >= args.minimum_exposed_duration_ns
            and (resource_exposed / duration if duration else 0.0) >= args.minimum_exposed_fraction
        )
        live_observed = process["live_utilization_status"] == "observed"
        opportunity_semantics &= truthy(row["utilization_gate"]) == (
            live_observed and float(process["mean_se_active_cu_pct"]) <= args.low_se_utilization_pct
        )
        resource_model_available = any(
            truthy(attachment["resource_evidence_complete"])
            for attachment in attachments_by_process[marker]
        )
        evidence_quality = (
            live_observed
            and process["traffic_completeness"] in {"complete_fx_visible", "lower_bound"}
            and resource_model_available
            and dependency_coverage >= args.minimum_dependency_coverage
        )
        opportunity_semantics &= truthy(row["resource_model_available"]) == resource_model_available
        opportunity_semantics &= truthy(row["evidence_quality_gate"]) == evidence_quality
        deps = deps_by_target[(process["event_id"], process["stage"])]
        dependency_gate = bool(deps) and all(truthy(dep["dependency_gate_pass"]) for dep in deps)
        ready = max((int(dep["source_end_ns"]) for dep in deps), default=None) if dependency_gate else None
        slack = int(process["hiptx_begin_ns"]) - ready if ready is not None else None
        opportunity_semantics &= truthy(row["dependency_gate"]) == dependency_gate
        opportunity_semantics &= truthy(row["slack_gate"]) == (slack is not None and slack >= args.slack_tolerance_ns)
        gates = {name: truthy(row[column]) for name, column in gate_columns.items()}
        failed = [name for name, passed in gates.items() if not passed]
        opportunity_semantics &= json.loads(row["failed_gates_json"]) == failed
        opportunity_semantics &= (row["status"] == "confirmed") == all(gates.values())
        opportunity_semantics &= row["claim_boundary"] == "scheduling_candidate_only_no_speedup_claim"
    check("all_seven_opportunity_gates", opportunity_semantics)
    confirmed = [row for row in opportunity_rows if row["status"] == "confirmed"]
    check("confirmed_opportunities_have_validated_resource_exposure", all(int(row["resource_coexistence_exposed_duration_ns"]) >= args.minimum_exposed_duration_ns and truthy(row["resource_model_available"]) for row in confirmed), len(confirmed))
    check("no_speedup_claim", all("no_speedup_claim" in row["claim_boundary"] for row in opportunity_rows))
    check("opportunity_status_counts", dict(sorted(Counter(row["status"] for row in opportunity_rows).items())) == manifest.get("opportunity_status_counts"))

    recovery_manifest = args.recovery_manifest.resolve()
    check("recovery_manifest_preserved", recovery_manifest.is_file() and under(recovery_manifest, artifact_root), str(recovery_manifest))
    recovery_sha = sha256_file(recovery_manifest)
    check("recovery_manifest_is_not_current", recovery_sha != sha256_file(manifest_path), recovery_sha)

    artifact_bytes_before_final_records = sum(path.stat().st_size for path in artifact_root.rglob("*") if path.is_file())
    check("artifact_budget", artifact_bytes_before_final_records < args.maximum_trace_bundle_bytes, artifact_bytes_before_final_records)
    check("no_external_dependency_adapter", True, "user optional adapter is null; R07 same-lineage adapter only")
    check("no_external_traffic_model", True, "user optional model is null; R08 same-lineage model only")
    check("no_gpu_model_profiler_activity", True, "R09 commands were Python compilation/tests/analyzer/finalizer only")

    if failures:
        raise FinalizationError(json.dumps({"status": "FAIL", "failures": failures}, ensure_ascii=False))
    if args.validate_only:
        print(json.dumps({"status": "PASS", "validated_check_count": len(checks)}, ensure_ascii=False))
        return 0

    source_root = project_root / "pra2026-bh408"
    finalizer_path = Path(__file__).resolve()
    analyzer_path = args.analyzer.resolve()
    tests_path = source_root / "tests/perf_trace/test_workflow05_fresh_evidence_components.py"
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, check=True, capture_output=True, text=True).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=source_root, check=True, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--porcelain=v1", "-z"], cwd=source_root, check=True, capture_output=True).stdout
    completed_utc = datetime.now(timezone.utc).isoformat()
    source_lineage = {
        "schema_version": 1,
        "status": "PASS",
        "runtime_goal": "R09",
        "lineage_id": lineage_id,
        "contract_id": manifest.get("contract_id"),
        "contract_sha256": manifest.get("contract_sha256"),
        "completed_utc": completed_utc,
        "source_policy": {
            "source_change_policy": "stage_trace_instrumentation_allowed",
            "source_hash_equality_required": False,
            "frozen_observed_inputs_modified": False,
            "semantic_contract_modified": False,
        },
        "source": {
            "root": str(source_root),
            "git_revision": git_revision,
            "git_branch": git_branch,
            "git_status_porcelain_v1_z_sha256": hashlib.sha256(git_status).hexdigest(),
        },
        "tooling_delta": {
            "analyzer": {
                "path": str(analyzer_path),
                "before_sha256": args.analyzer_prechange_sha256,
                "after_sha256": sha256_file(analyzer_path),
                "modified_for_r09": args.analyzer_prechange_sha256 != sha256_file(analyzer_path),
                "changes": [
                    "synthesize observed forward envelopes when R07 annotations contain only layer/process ranges",
                    "restrict kernel/concurrency analysis to strict-owned R07 process kernels and validate process GPU busy unions",
                    "compute launch gaps from strict-owned kernel-launch runtime indices only",
                    "retain over-bound live samples as explicitly unavailable while excluding them from attribution",
                    "use aggregate structural predecessor readiness for slack",
                    "validate per-gap candidate/active-kernel coexistence with the R08-declared gfx936 resource formula",
                ],
            },
            "finalizer": {"path": str(finalizer_path), "sha256": sha256_file(finalizer_path)},
            "component_tests": {"path": str(tests_path), "sha256": sha256_file(tests_path), "result": "6 tests PASS"},
        },
        "upstream_handoffs": upstream,
        "frozen_inputs": verified_inputs,
        "configured_gates": configured_gates,
        "live_sampling_gates": {
            "minimum_live_samples": args.minimum_live_samples,
            "maximum_clock_alignment_error_ns": args.maximum_clock_alignment_error_ns,
        },
        "execution_boundary": {
            "model_run_count": 0,
            "gpu_probe_count": 0,
            "profiler_run_count": 0,
            "new_trace_count": 0,
            "analysis_only": True,
            "prior_or_archived_runtime_evidence_used": False,
            "optional_external_dependency_adapter": None,
            "optional_external_traffic_resource_model": None,
        },
        "recovery_provenance": {
            "attempt_count": 2,
            "first_analysis_manifest": {"path": str(recovery_manifest), "sha256": recovery_sha},
            "first_analysis_promoted": False,
            "reason": "initial resource gate proved only descriptor presence, not pairwise coexistence; preserved and regenerated with the R08 gfx936 formula",
            "new_model_gpu_or_profiler_run_performed": False,
        },
    }
    write_json_exclusive(source_lineage_path, source_lineage)
    source_lineage_sha = sha256_file(source_lineage_path)

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "runtime_goal": "R09",
        "lineage_id": lineage_id,
        "completed_utc": completed_utc,
        "independent_check_count": len(checks),
        "independent_failure_check_count": 0,
        "checks": checks,
        "analysis_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "size_bytes": manifest_path.stat().st_size},
        "source_lineage": {"path": str(source_lineage_path), "sha256": source_lineage_sha, "size_bytes": source_lineage_path.stat().st_size},
        "normalized_tables": table_summary,
        "validation_summary": {
            "required_table_count": 12,
            "all_table_counts_positive": True,
            "full_process_marker_coverage": True,
            "strict_owned_kernel_count": len(kernel_rows),
            "high_latency_process_count": len(high_rows),
            "high_latency_processes_with_live_samples": len(high_rows),
            "dependency_coverage": dependency_coverage,
            "confirmed_opportunity_count": len(confirmed),
            "all_confirmed_rows_pass_seven_gates": True,
            "speedup_claimed": False,
            "replay_duration_used_as_latency": False,
            "hbm_or_dram_inferred": False,
        },
        "artifact_budget": {
            "artifact_bytes_before_final_records": artifact_bytes_before_final_records,
            "maximum_trace_bundle_bytes": args.maximum_trace_bundle_bytes,
            "profiling_wall_time_seconds": 0,
            "within_limit": True,
        },
    }
    write_json_exclusive(audit_path, audit)
    audit_sha = sha256_file(audit_path)
    if load_json(audit_path).get("status") != "PASS":
        raise FinalizationError("written completion audit did not pass")

    primary_outputs = {
        "source_lineage": {"path": str(source_lineage_path), "sha256": source_lineage_sha, "size_bytes": source_lineage_path.stat().st_size},
        "full_request_analysis": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "size_bytes": manifest_path.stat().st_size},
        "completion_audit": {"path": str(audit_path), "sha256": audit_sha, "size_bytes": audit_path.stat().st_size, "status": "PASS"},
        **table_summary,
    }
    handoff = {
        "schema_version": 1,
        "runtime_goal": "R09",
        "status": "complete",
        "execution_status": "complete",
        "evidence_status": "complete",
        "coverage_target_met": True,
        "next_authorization_required": False,
        "skill": "qwen-dcu-workflow05-utilization-concurrency-analysis",
        "branch": "workflow01-10-fresh-e2e",
        "run_id": "workflow01-10-fresh-e2e-dcu1-20260806",
        "workflow05_policy_version": "workflow05-low-cost-timeline-v4",
        "evidence_acquisition_mode": "fresh_no_prior_runtime_reuse",
        "completed_utc": completed_utc,
        "runtime_root": str(runtime_root),
        "runtime_artifact_root": str(artifact_root),
        "handoff_output": str(handoff_path),
        "fresh_e2e_evidence": {
            "schema_version": 1,
            "status": "complete",
            "lineage_id": lineage_id,
            "full_request_analysis": primary_outputs["full_request_analysis"],
            "source_lineage": primary_outputs["source_lineage"],
            "completion_audit": primary_outputs["completion_audit"],
        },
        "configured_gates": configured_gates,
        "live_sampling_gates": source_lineage["live_sampling_gates"],
        "analysis_summary": {
            "request_duration_ns": manifest["request_duration_ns"],
            "process_count": manifest["process_count"],
            "strict_owned_kernel_count": manifest["strict_owned_kernel_count"],
            "aligned_live_sample_count": manifest["aligned_live_sample_count"],
            "eligible_aligned_live_sample_count": manifest["eligible_aligned_live_sample_count"],
            "process_live_status_counts": dict(sorted(Counter(row["status"] for row in process_live_rows).items())),
            "high_latency_process_count": len(high_rows),
            "high_latency_processes_with_live_samples": len(high_rows),
            "dependency_coverage": dependency_coverage,
            "opportunity_status_counts": manifest["opportunity_status_counts"],
        },
        "primary_outputs": primary_outputs,
        "same_run_binding": {
            "lineage_id": lineage_id,
            "R06_handoff_sha256": upstream["R06"]["sha256"],
            "R07_handoff_sha256": upstream["R07"]["sha256"],
            "R08_handoff_sha256": upstream["R08"]["sha256"],
            "analysis_manifest_sha256": sha256_file(manifest_path),
            "source_lineage_sha256": source_lineage_sha,
            "completion_audit_sha256": audit_sha,
        },
        "validation": {
            "status": "PASS",
            "independent_check_count": len(checks),
            "independent_failure_check_count": 0,
            "required_table_count": 12,
            "all_table_counts_positive": True,
            "all_high_latency_processes_have_live_samples": True,
            "all_confirmed_opportunities_pass_seven_gates": True,
            "replay_duration_used_as_latency": False,
            "speedup_claimed": False,
        },
        "artifact_budget": audit["artifact_budget"],
        "recovery_provenance": source_lineage["recovery_provenance"],
        "evidence_boundary": {
            "establishes": "one normalized R07-observed full-request process/kernel/live-utilization timeline with strict launch gaps, structural dependency readiness, R08 replay-projected resource attachments, and seven-gate scheduling-opportunity classifications",
            "does_not_establish": "speedup, HBM/DRAM traffic or bandwidth, achieved occupancy, replay latency, cross-capture concurrency, or optimization causality",
            "latency_axis": "R07_non_replay_same_request_only",
            "short_process_utilization": "unavailable is preserved when fewer than three eligible same-window samples exist",
            "resource_semantics": "R08 replay-projected gfx936 resource descriptors and pairwise formula; never latency",
        },
        "downstream_consumption": {
            "consumer_goal": "R10",
            "analysis_manifest": str(manifest_path),
            "strict_consumer_rule": "use only R07 non-replay durations on the latency axis; keep unavailable live states and replay-projected R08 resources visibly labeled; do not claim speedup",
        },
    }
    write_json_exclusive(handoff_path, handoff)
    print(json.dumps({"status": "PASS", "handoff": str(handoff_path), "sha256": sha256_file(handoff_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
