#!/usr/bin/env python3
"""Build the complete normalized analysis for one fresh-run request.

Every latency interval comes from the R07 non-replay clock. Live utilization is
attached only from aligned RSMI samples. R08 PMC/resource values remain
replay-projected attributes and never replace observed duration.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


class AnalysisError(RuntimeError):
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
        raise AnalysisError(f"JSON must be an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisError(f"CSV lacks a header: {path}")
        return list(reader)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(row: dict[str, Any], name: str, default: int | None = None) -> int:
    raw = row.get(name, "")
    if raw in {None, ""} and default is not None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid integer {name}: {raw!r}") from exc


def as_float(row: dict[str, Any], name: str, default: float | None = None) -> float:
    raw = row.get(name, "")
    if raw in {None, ""} and default is not None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid number {name}: {raw!r}") from exc


def truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def checked_output(container: dict[str, Any], key: str) -> Path:
    record = container.get("outputs", {}).get(key, {})
    if not isinstance(record, dict):
        raise AnalysisError(f"missing output reference: {key}")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise AnalysisError(f"changed output reference: {key}: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one fresh full-request process/hardware trace."
    )
    parser.add_argument("--profile-metadata", type=Path, required=True)
    parser.add_argument("--process-trace-summary", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--runtime-calls", type=Path, required=True)
    parser.add_argument("--strict-ownership", type=Path, required=True)
    parser.add_argument("--process-performance", type=Path, required=True)
    parser.add_argument("--process-gpu-timeline", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--live-samples", type=Path, required=True)
    parser.add_argument("--live-summary", type=Path, required=True)
    parser.add_argument("--dependency-adapter", type=Path, required=True)
    parser.add_argument("--traffic-resource-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-live-samples", type=int, default=3)
    parser.add_argument("--maximum-clock-alignment-error-ns", type=int, default=1_000_000)
    parser.add_argument("--high-latency-count", type=int, default=128)
    parser.add_argument("--low-se-utilization-pct", type=float, default=50.0)
    parser.add_argument("--low-kernel-concurrency-max", type=int, default=1)
    parser.add_argument("--minimum-launch-gap-ns", type=int, default=100_000)
    parser.add_argument("--minimum-dependency-coverage", type=float, default=0.0)
    parser.add_argument("--minimum-exposed-duration-ns", type=int, default=100_000)
    parser.add_argument("--minimum-exposed-fraction", type=float, default=0.01)
    parser.add_argument("--slack-tolerance-ns", type=int, default=1_000)
    args = parser.parse_args()
    if args.minimum_live_samples < 1 or args.high_latency_count < 1:
        parser.error("sample and high-latency counts must be positive")
    if args.maximum_clock_alignment_error_ns < 0 or args.minimum_launch_gap_ns < 0:
        parser.error("alignment error and launch gap must be nonnegative")
    if args.low_kernel_concurrency_max < 0 or args.minimum_exposed_duration_ns < 0:
        parser.error("concurrency and exposed duration thresholds must be nonnegative")
    if not 0 <= args.minimum_dependency_coverage <= 1:
        parser.error("minimum dependency coverage must be in [0, 1]")
    if not 0 <= args.minimum_exposed_fraction <= 1:
        parser.error("minimum exposed fraction must be in [0, 1]")
    return args


def interval_sweep(
    kernels: list[dict[str, Any]], request_begin: int, request_end: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for row in kernels:
        kernel_id = str(row["kernel_id"])
        queue = str(row.get("queue_id", ""))
        changes[int(row["begin_ns"])].append((1, kernel_id, queue))
        changes[int(row["end_ns"])].append((-1, kernel_id, queue))
    points = sorted(set(changes).union({request_begin, request_end}))
    active: set[str] = set()
    active_queues: Counter[str] = Counter()
    kernel_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for index, point in enumerate(points[:-1]):
        for delta, kernel_id, queue in sorted(changes.get(point, [])):
            if delta < 0:
                active.discard(kernel_id)
                active_queues[queue] -= 1
                if active_queues[queue] <= 0:
                    del active_queues[queue]
        for delta, kernel_id, queue in sorted(changes.get(point, [])):
            if delta > 0:
                active.add(kernel_id)
                active_queues[queue] += 1
        next_point = points[index + 1]
        if next_point <= point:
            continue
        common = {
            "begin_ns": point,
            "end_ns": next_point,
            "duration_ns": next_point - point,
            "timing_source": "observed_non_replay_hipops_interval_sweep",
            "evidence_class": "observed",
        }
        kernel_rows.append(
            {
                **common,
                "active_kernel_count": len(active),
                "active_kernel_ids": ";".join(sorted(active)),
            }
        )
        queue_rows.append(
            {
                **common,
                "active_queue_count": len(active_queues),
                "active_queue_ids": ";".join(sorted(active_queues)),
                "active_kernel_count": len(active),
            }
        )
    return kernel_rows, queue_rows


def interval_union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return sum(end - begin for begin, end in merged)


def main() -> int:
    args = parse_args()
    inputs = (
        args.profile_metadata,
        args.process_trace_summary,
        args.annotations,
        args.runtime_calls,
        args.strict_ownership,
        args.process_performance,
        args.process_gpu_timeline,
        args.kernels,
        args.live_samples,
        args.live_summary,
        args.dependency_adapter,
        args.traffic_resource_model,
    )
    for path in inputs:
        if not path.is_file():
            raise AnalysisError(f"missing input: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise AnalysisError(f"refusing nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(args.profile_metadata)
    trace_summary = load_json(args.process_trace_summary)
    live_summary = load_json(args.live_summary)
    adapter = load_json(args.dependency_adapter)
    model = load_json(args.traffic_resource_model)
    expected = metadata.get("expected_process_ranges")
    emitted = metadata.get("emitted_process_ranges")
    if (
        metadata.get("status") != "profile_complete_analysis_pending"
        or metadata.get("process_profile") != "on"
        or not isinstance(expected, list)
        or not expected
        or expected != emitted
        or metadata.get("expected_process_range_count") != len(expected)
    ):
        raise AnalysisError("profile metadata does not prove full process coverage")
    if trace_summary.get("status") != "PASS":
        raise AnalysisError("process trace summary did not pass")
    if trace_summary.get("contract_id") != metadata.get("contract_id"):
        raise AnalysisError("process trace and profile contract differ")
    empirical = live_summary.get("empirical_sub_millisecond_cadence", {})
    if (
        live_summary.get("status") != "complete"
        or live_summary.get("physical_device_index") != 1
        or live_summary.get("metric") != "se_active_cu_pct"
        or not isinstance(empirical, dict)
        or empirical.get("p50") is not True
        or empirical.get("p95") is not True
    ):
        raise AnalysisError("live utilization capture lacks empirical sub-ms proof")
    lineage_id = metadata.get("lineage_id")
    if not isinstance(lineage_id, str) or not lineage_id:
        raise AnalysisError("profile metadata lacks fresh-run lineage_id")
    if (
        adapter.get("status") != "complete"
        or adapter.get("adapter_type") != "fresh_run_fixed_input_fx_process_dependency"
        or model.get("status") != "complete"
        or model.get("model_type") != "fresh_run_fx_visible_traffic_and_dcu_family_resource"
        or adapter.get("lineage_id") != lineage_id
        or model.get("lineage_id") != lineage_id
        or adapter.get("contract_id") != metadata.get("contract_id")
        or model.get("contract_id") != metadata.get("contract_id")
    ):
        raise AnalysisError("R07/R08 evidence does not share one fresh-run contract")

    annotations = load_csv(args.annotations)
    runtime_calls = load_csv(args.runtime_calls)
    strict_ownership = load_csv(args.strict_ownership)
    process_rows = load_csv(args.process_performance)
    process_gpu_rows = load_csv(args.process_gpu_timeline)
    kernel_rows = load_csv(args.kernels)
    edge_rows = load_csv(checked_output(adapter, "edges"))
    traffic_rows = load_csv(checked_output(model, "traffic"))
    resource_rows = load_csv(checked_output(model, "resource"))

    markers = [row.get("process_range", "") for row in process_rows]
    if len(markers) != len(set(markers)) or set(markers) != set(expected):
        raise AnalysisError("process-performance rows do not equal expected markers")
    request_begin = int(metadata.get("request_start_realtime_ns", 0))
    request_end = int(metadata.get("request_end_realtime_ns", 0))
    if request_begin <= 0 or request_end <= request_begin:
        raise AnalysisError("invalid request realtime window")
    anchor_names = (
        "request_start_realtime_ns",
        "request_end_realtime_ns",
        "request_start_monotonic_ns",
        "request_end_monotonic_ns",
    )
    present_anchor_count = sum(metadata.get(name) is not None for name in anchor_names)
    if present_anchor_count not in {2, 4}:
        raise AnalysisError("request clock anchors are incomplete")
    if present_anchor_count == 4:
        monotonic_duration = (
            int(metadata["request_end_monotonic_ns"])
            - int(metadata["request_start_monotonic_ns"])
        )
        alignment_delta = abs((request_end - request_begin) - monotonic_duration)
        if alignment_delta > args.maximum_clock_alignment_error_ns:
            raise AnalysisError("request realtime/monotonic anchors do not align")

    samples: list[dict[str, Any]] = []
    raw_live_sample_count = 0
    for line_number, line in enumerate(args.live_samples.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw_live_sample_count += 1
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnalysisError(f"live sample {line_number} is not an object")
        if value.get("read_status") != 0:
            continue
        timestamp = value.get("realtime_midpoint_ns")
        uncertainty = value.get("alignment_uncertainty_ns")
        if (
            not isinstance(timestamp, int)
            or not isinstance(uncertainty, int)
            or uncertainty < 0
        ):
            raise AnalysisError(f"live sample {line_number} lacks clock fields")
        if value.get("physical_device_index", 1) != 1:
            raise AnalysisError(f"live sample {line_number} is from the wrong device")
        if value.get("metric", "se_active_cu_pct") != "se_active_cu_pct":
            raise AnalysisError(f"live sample {line_number} has the wrong metric")
        samples.append(value)
    if live_summary.get("sample_count") not in {None, raw_live_sample_count}:
        raise AnalysisError("live sample row count differs from its summary")
    if live_summary.get("successful_sample_count") not in {None, len(samples)}:
        raise AnalysisError("successful live sample count differs from its summary")
    samples.sort(key=lambda row: row["realtime_midpoint_ns"])
    sample_times = [row["realtime_midpoint_ns"] for row in samples]

    traffic_by_stage = {
        (row.get("event_id", ""), row.get("stage", "")): row for row in traffic_rows
    }
    resource_by_stage: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    resource_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in resource_rows:
        stage_key = (row.get("event_id", ""), row.get("stage", ""))
        resource_by_stage[stage_key].append(row)
        resource_key = (*stage_key, row.get("matched_kernel_family", ""))
        if resource_key in resource_by_key:
            raise AnalysisError(f"duplicate resource-model key: {resource_key}")
        resource_by_key[resource_key] = row
    edge_counts: Counter[tuple[str, str]] = Counter()
    unknown_counts: Counter[tuple[str, str]] = Counter()
    for row in edge_rows:
        key = (row.get("event_id", ""), row.get("target_stage", "") or row.get("source_stage", ""))
        if row.get("edge_type") == "data" and truthy(row.get("verified")):
            edge_counts[key] += 1
        else:
            unknown_counts[key] += 1

    normalized_process: list[dict[str, Any]] = []
    for row in process_rows:
        begin = as_int(row, "hiptx_begin_ns")
        end = as_int(row, "hiptx_end_ns")
        if end < begin:
            raise AnalysisError(f"negative process interval: {row['process_range']}")
        left = bisect.bisect_left(sample_times, begin)
        right = bisect.bisect_right(sample_times, end)
        inside = [
            sample for sample in samples[left:right]
            if sample["alignment_uncertainty_ns"] <= args.maximum_clock_alignment_error_ns
        ]
        means = [float(s["mean_se_active_cu_pct"]) for s in inside if s.get("mean_se_active_cu_pct") is not None]
        maxima = [float(s["max_se_active_cu_pct"]) for s in inside if s.get("max_se_active_cu_pct") is not None]
        live_available = len(means) >= args.minimum_live_samples
        key = (row.get("event_id", ""), row.get("stage", ""))
        traffic = traffic_by_stage.get(key, {})
        resources = resource_by_stage.get(key, [])
        normalized_process.append({
            **row,
            "duration_ns": end - begin,
            "live_sample_count": len(means),
            "live_utilization_status": "observed" if live_available else "unavailable_too_few_samples",
            "mean_se_active_cu_pct": sum(means) / len(means) if live_available else "unavailable",
            "max_se_active_cu_pct": max(maxima) if live_available and maxima else "unavailable",
            "maximum_sample_alignment_uncertainty_ns": max((s["alignment_uncertainty_ns"] for s in inside), default=""),
            "verified_dependency_edge_count": edge_counts[key],
            "unknown_dependency_count": unknown_counts[key],
            "fx_visible_total_io_bytes": traffic.get("fx_visible_total_io_bytes", "unavailable"),
            "traffic_completeness": traffic.get("traffic_completeness", "unavailable"),
            "hbm_or_dram_bytes": "unavailable",
            "resource_family_count": len(resources),
            "resource_families_json": json.dumps([
                {
                    "family": resource.get("matched_kernel_family"),
                    "occupancy_upper_bound_pct": resource.get("theoretical_occupancy_upper_bound_pct"),
                    "evidence": resource.get("hardware_evidence_class"),
                }
                for resource in resources
            ], separators=(",", ":")),
            "timing_source": "observed_non_replay_hiptx_fresh_run",
            "evidence_class": "observed",
        })
    normalized_process.sort(key=lambda row: (int(row["hiptx_begin_ns"]), int(row["hiptx_end_ns"]), row["process_range"]))
    process_by_stage = {
        (row.get("event_id", ""), row.get("stage", "")): row for row in normalized_process
    }
    process_by_marker = {row["process_range"]: row for row in normalized_process}

    owner_by_kernel: dict[str, str] = {}
    launch_runtime_pairs: set[tuple[str, str]] = set()
    for row in process_gpu_rows:
        kernel_id = row.get("kernel_id", "")
        marker = row.get("process_range", "")
        if not kernel_id or not marker:
            raise AnalysisError("process GPU timeline contains an empty ownership key")
        prior = owner_by_kernel.setdefault(kernel_id, marker)
        if prior != marker:
            raise AnalysisError(f"kernel has multiple process owners: {kernel_id}")
        if row.get("runtime_index", ""):
            launch_runtime_pairs.add((marker, row["runtime_index"]))
    strict_owner_by_kernel: dict[str, str] = {}
    for row in strict_ownership:
        kernel_id = row.get("kernel_id", "")
        marker = row.get("marker", "")
        if kernel_id and marker and row.get("kind") == "process":
            prior = strict_owner_by_kernel.setdefault(kernel_id, marker)
            if prior != marker:
                raise AnalysisError(f"strict ownership conflicts for kernel: {kernel_id}")
            if row.get("runtime_index", ""):
                launch_runtime_pairs.add((marker, row["runtime_index"]))
    if strict_owner_by_kernel != owner_by_kernel:
        raise AnalysisError(
            "process GPU timeline and strict process ownership do not identify "
            "the same kernels"
        )

    normalized_kernels: list[dict[str, Any]] = []
    seen_owned_kernel_ids: set[str] = set()
    for row in kernel_rows:
        kernel_id = row.get("kernel_id", "")
        if kernel_id not in owner_by_kernel:
            continue
        if as_int(row, "device_id") != 1:
            raise AnalysisError(f"strict-owned kernel is not on physical DCU 1: {kernel_id}")
        begin = max(request_begin, as_int(row, "begin_ns"))
        end = min(request_end, as_int(row, "end_ns"))
        if end <= begin:
            raise AnalysisError(f"strict-owned kernel is outside the request: {kernel_id}")
        if kernel_id in seen_owned_kernel_ids:
            raise AnalysisError(f"duplicate strict-owned kernel row: {kernel_id}")
        seen_owned_kernel_ids.add(kernel_id)
        normalized_kernels.append({
            **row,
            "begin_ns": begin,
            "end_ns": end,
            "duration_ns": end - begin,
            "process_owner": owner_by_kernel[kernel_id],
            "timing_source": "observed_non_replay_hipops_fresh_run",
            "evidence_class": "observed",
        })
    missing_owned_kernel_ids = set(owner_by_kernel).difference(seen_owned_kernel_ids)
    if missing_owned_kernel_ids:
        raise AnalysisError(
            f"strict-owned kernels missing from kernel timeline: "
            f"{len(missing_owned_kernel_ids)}"
        )
    normalized_kernels.sort(key=lambda row: (int(row["begin_ns"]), int(row["end_ns"]), str(row["kernel_id"])))
    kernel_concurrency, queue_concurrency = interval_sweep(normalized_kernels, request_begin, request_end)
    kernel_by_id = {str(row["kernel_id"]): row for row in normalized_kernels}
    kernels_by_launch_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_kernels:
        kernels_by_launch_pair[(row["process_owner"], str(row.get("runtime_index", "")))].append(row)

    sweep_ends = [int(row["end_ns"]) for row in kernel_concurrency]

    def concurrency_window(begin: int, end: int) -> tuple[int, int, int]:
        """Return kernel peak, queue peak, and low-concurrency exposure."""
        kernels_peak = 0
        queues_peak = 0
        low_concurrency_ns = 0
        index = bisect.bisect_right(sweep_ends, begin)
        while index < len(kernel_concurrency):
            kernel_row = kernel_concurrency[index]
            queue_row = queue_concurrency[index]
            row_begin = int(kernel_row["begin_ns"])
            row_end = int(kernel_row["end_ns"])
            if row_begin >= end:
                break
            overlap_begin = max(begin, row_begin)
            overlap_end = min(end, row_end)
            if overlap_end > overlap_begin:
                active_kernels = int(kernel_row["active_kernel_count"])
                kernels_peak = max(kernels_peak, active_kernels)
                queues_peak = max(queues_peak, int(queue_row["active_queue_count"]))
                if active_kernels <= args.low_kernel_concurrency_max:
                    low_concurrency_ns += overlap_end - overlap_begin
            index += 1
        return kernels_peak, queues_peak, low_concurrency_ns

    device_limits = model.get("device_limits", {})
    required_device_limits = (
        "wave_size",
        "wave_limit",
        "thread_limit",
        "vgpr_resource",
        "shared_memory_bytes",
    )
    if any(name not in device_limits for name in required_device_limits):
        raise AnalysisError("traffic/resource model lacks device coexistence limits")

    def kernel_resource(kernel: dict[str, Any]) -> dict[str, str] | None:
        owner = process_by_marker.get(kernel.get("process_owner", ""))
        if owner is None:
            return None
        resource = resource_by_key.get(
            (
                owner.get("event_id", ""),
                owner.get("stage", ""),
                kernel.get("kernel_family", ""),
            )
        )
        if resource is None or not truthy(resource.get("resource_evidence_complete")):
            return None
        return resource

    def resource_vector(resource: dict[str, str]) -> tuple[float, int, float, float]:
        work_group_size = float(resource["work_group_size"])
        waves = math.ceil(work_group_size / float(device_limits["wave_size"]))
        vgpr_use = float(resource["vgpr_count"]) * work_group_size
        shared_use = float(resource["shared_memory_size_bytes"])
        return work_group_size, waves, vgpr_use, shared_use

    def resource_group_fits(resources: list[dict[str, str]]) -> bool:
        vectors = [resource_vector(resource) for resource in resources]
        return (
            sum(vector[0] for vector in vectors) <= float(device_limits["thread_limit"])
            and sum(vector[1] for vector in vectors) <= int(device_limits["wave_limit"])
            and sum(vector[2] for vector in vectors) <= float(device_limits["vgpr_resource"])
            and sum(vector[3] for vector in vectors) <= float(device_limits["shared_memory_bytes"])
        )

    def resource_coexistence_window(
        marker: str, next_runtime_index: str, begin: int, end: int
    ) -> tuple[int, str, str]:
        candidate_kernels = kernels_by_launch_pair.get((marker, next_runtime_index), [])
        if not candidate_kernels:
            return 0, "unavailable", "no_next_strict_owned_kernel_launch"
        candidate_resources = [kernel_resource(kernel) for kernel in candidate_kernels]
        if any(resource is None for resource in candidate_resources):
            return 0, "unavailable", "candidate_kernel_resource_model_unavailable"
        complete_candidate_resources = [
            resource for resource in candidate_resources if resource is not None
        ]
        feasible_ns = 0
        unavailable_ns = 0
        tested_ns = 0
        index = bisect.bisect_right(sweep_ends, begin)
        while index < len(kernel_concurrency):
            row = kernel_concurrency[index]
            row_begin = int(row["begin_ns"])
            row_end = int(row["end_ns"])
            if row_begin >= end:
                break
            overlap_begin = max(begin, row_begin)
            overlap_end = min(end, row_end)
            overlap = max(0, overlap_end - overlap_begin)
            active_count = int(row["active_kernel_count"])
            if overlap and active_count <= args.low_kernel_concurrency_max:
                active_ids = [value for value in row["active_kernel_ids"].split(";") if value]
                active_resources = [kernel_resource(kernel_by_id[value]) for value in active_ids]
                if any(resource is None for resource in active_resources):
                    unavailable_ns += overlap
                else:
                    tested_ns += overlap
                    complete_active_resources = [
                        resource for resource in active_resources if resource is not None
                    ]
                    if all(
                        resource_group_fits([candidate_resource, *complete_active_resources])
                        for candidate_resource in complete_candidate_resources
                    ):
                        feasible_ns += overlap
            index += 1
        if feasible_ns > 0:
            return feasible_ns, "validated_replay_projected", "gfx936_pairwise_resource_formula_pass"
        if unavailable_ns > 0 and tested_ns == 0:
            return 0, "unavailable", "active_kernel_resource_model_unavailable"
        return 0, "infeasible", "gfx936_pairwise_resource_formula_failed"

    kernel_intervals_by_process: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in normalized_kernels:
        kernel_intervals_by_process[row["process_owner"]].append(
            (int(row["begin_ns"]), int(row["end_ns"]))
        )
    for process in normalized_process:
        marker = process["process_range"]
        intervals = kernel_intervals_by_process[marker]
        busy_union_ns = interval_union_duration_ns(intervals)
        if process.get("strict_owned_kernel_count", "") not in {"", None}:
            if int(process["strict_owned_kernel_count"]) != len(intervals):
                raise AnalysisError(f"strict-owned kernel count changed: {marker}")
        if process.get("strict_owned_kernel_busy_union_ms", "") not in {"", None}:
            observed_ns = round(float(process["strict_owned_kernel_busy_union_ms"]) * 1_000_000)
            if observed_ns != busy_union_ns:
                raise AnalysisError(f"strict-owned GPU busy union changed: {marker}")
        process["strict_owned_gpu_busy_union_ns"] = busy_union_ns
        process["strict_owned_gpu_busy_union_source"] = (
            "observed_non_replay_strict_owned_hipops_interval_union"
        )

    request_key = str(metadata.get("request_id") or metadata.get("contract_id"))
    request_timeline: list[dict[str, Any]] = [{
        "track_type": "request",
        "event_key": request_key,
        "parent_key": "",
        "label": "full request",
        "begin_ns": request_begin,
        "end_ns": request_end,
        "duration_ns": request_end - request_begin,
        "event_id": "",
        "forward_id": "",
        "layer": "",
        "occurrence": "",
        "phase": "",
        "process_owner": "",
        "runtime_index": "",
        "queue_id": "",
        "timing_source": "observed_r07_request_anchor",
        "evidence_class": "observed",
    }]
    layer_annotations = [row for row in annotations if row.get("kind") == "layer"]
    forward_annotations = [row for row in annotations if row.get("kind") == "forward"]
    if not layer_annotations:
        raise AnalysisError("annotations contain no observed layer intervals")
    forward_bounds: dict[str, dict[str, Any]] = {}
    for row in layer_annotations:
        forward_id = row.get("forward_id", "")
        if not forward_id:
            raise AnalysisError("layer annotation lacks a forward identity")
        begin, end = as_int(row, "begin_ns"), as_int(row, "end_ns")
        bound = forward_bounds.setdefault(
            forward_id,
            {"begin_ns": begin, "end_ns": end, "phase": row.get("phase", "")},
        )
        bound["begin_ns"] = min(int(bound["begin_ns"]), begin)
        bound["end_ns"] = max(int(bound["end_ns"]), end)
        if bound["phase"] != row.get("phase", ""):
            bound["phase"] = "mixed"

    emitted_forward_ids: set[str] = set()
    for row in forward_annotations:
        forward_id = row.get("forward_id", "") or row.get("annotation_id", "")
        if not forward_id or forward_id in emitted_forward_ids:
            raise AnalysisError(f"invalid duplicate forward annotation: {forward_id!r}")
        emitted_forward_ids.add(forward_id)
        begin, end = as_int(row, "begin_ns"), as_int(row, "end_ns")
        request_timeline.append({
            "track_type": "forward",
            "event_key": f"forward:{forward_id}",
            "parent_key": request_key,
            "label": row.get("message", "") or f"forward {forward_id}",
            "begin_ns": begin,
            "end_ns": end,
            "duration_ns": end - begin,
            "event_id": "",
            "forward_id": forward_id,
            "layer": "",
            "occurrence": row.get("occurrence", ""),
            "phase": row.get("phase", ""),
            "process_owner": "",
            "runtime_index": "",
            "queue_id": "",
            "timing_source": "observed_non_replay_hiptx_annotation",
            "evidence_class": "observed",
        })
    for forward_id, bound in sorted(forward_bounds.items()):
        if forward_id in emitted_forward_ids:
            continue
        request_timeline.append({
            "track_type": "forward",
            "event_key": f"forward:{forward_id}",
            "parent_key": request_key,
            "label": f"forward {forward_id}",
            "begin_ns": bound["begin_ns"],
            "end_ns": bound["end_ns"],
            "duration_ns": int(bound["end_ns"]) - int(bound["begin_ns"]),
            "event_id": "",
            "forward_id": forward_id,
            "layer": "",
            "occurrence": forward_id,
            "phase": bound["phase"],
            "process_owner": "",
            "runtime_index": "",
            "queue_id": "",
            "timing_source": "observed_non_replay_layer_envelope",
            "evidence_class": "observed_derived_hierarchy",
        })

    layer_key_by_event: dict[str, str] = {}
    for row in layer_annotations:
        begin, end = as_int(row, "begin_ns"), as_int(row, "end_ns")
        event_id = row.get("event_id", "")
        layer_key = row.get("message", "") or f"layer:{event_id}"
        if not event_id or event_id in layer_key_by_event:
            raise AnalysisError(f"invalid duplicate layer event identity: {event_id!r}")
        layer_key_by_event[event_id] = layer_key
        request_timeline.append({
            "track_type": "layer",
            "event_key": layer_key,
            "parent_key": f"forward:{row.get('forward_id', '')}",
            "label": row.get("message", ""),
            "begin_ns": begin,
            "end_ns": end,
            "duration_ns": end - begin,
            "event_id": row.get("event_id", ""),
            "forward_id": row.get("forward_id", ""),
            "layer": row.get("layer", ""),
            "occurrence": row.get("occurrence", ""),
            "phase": row.get("phase", ""),
            "process_owner": "",
            "runtime_index": "",
            "queue_id": "",
            "timing_source": "observed_non_replay_hiptx_annotation",
            "evidence_class": "observed",
        })
    for process in normalized_process:
        parent_key = layer_key_by_event.get(process.get("event_id", ""))
        if not parent_key:
            raise AnalysisError(
                f"process lacks an observed layer parent: {process['process_range']}"
            )
        process["track_type"] = "process"
        process["event_key"] = process["process_range"]
        process["parent_key"] = parent_key
    normalized_runtime_calls: list[dict[str, Any]] = []
    observed_launch_runtime_pairs: set[tuple[str, str]] = set()
    for row in runtime_calls:
        if not row.get("begin_ns") or not row.get("end_ns"):
            continue
        begin, end = as_int(row, "begin_ns"), as_int(row, "end_ns")
        if end < begin:
            raise AnalysisError(f"negative HIP runtime interval: {row.get('runtime_index', '')}")
        if begin < request_begin or end > request_end:
            raise AnalysisError(
                f"HIP runtime interval lies outside request: {row.get('runtime_index', '')}"
            )
        call = {
            "track_type": "hip_runtime",
            "event_key": row.get("runtime_index", ""),
            "parent_key": row.get("process_owner", ""),
            "label": row.get("api_name", ""),
            "begin_ns": begin,
            "end_ns": end,
            "duration_ns": end - begin,
            "event_id": "",
            "forward_id": "",
            "layer": "",
            "occurrence": "",
            "phase": "",
            "process_owner": row.get("process_owner", ""),
            "runtime_index": row.get("runtime_index", ""),
            "queue_id": "",
            "timing_source": "observed_non_replay_hip_runtime",
            "evidence_class": "observed",
        }
        request_timeline.append(call)
        normalized_runtime_calls.append(call)
        pair = (call["process_owner"], str(call["runtime_index"]))
        if pair in launch_runtime_pairs:
            observed_launch_runtime_pairs.add(pair)
    missing_launch_pairs = launch_runtime_pairs.difference(observed_launch_runtime_pairs)
    if missing_launch_pairs:
        raise AnalysisError(
            f"strict-owned kernel launches missing from HIP runtime calls: "
            f"{len(missing_launch_pairs)}"
        )
    request_timeline.sort(key=lambda row: (int(row["begin_ns"]), int(row["end_ns"]), row["track_type"], str(row["event_key"])))

    normalized_samples = [{
        "sequence": sample.get("sequence"),
        "realtime_midpoint_ns": sample["realtime_midpoint_ns"],
        "alignment_uncertainty_ns": sample["alignment_uncertainty_ns"],
        "mean_se_active_cu_pct": sample.get("mean_se_active_cu_pct"),
        "max_se_active_cu_pct": sample.get("max_se_active_cu_pct"),
        "se_active_cu_pct_json": json.dumps(sample.get("se_active_cu_pct"), separators=(",", ":")),
        "alignment_status": (
            "eligible"
            if sample["alignment_uncertainty_ns"] <= args.maximum_clock_alignment_error_ns
            else "unavailable_alignment_error"
        ),
        "eligible_for_process_attribution": (
            sample["alignment_uncertainty_ns"] <= args.maximum_clock_alignment_error_ns
        ),
        "evidence_class": (
            "observed"
            if sample["alignment_uncertainty_ns"] <= args.maximum_clock_alignment_error_ns
            else "unavailable"
        ),
        "metric_semantics": "instantaneous_active_cu_ratio_per_shader_engine",
    } for sample in samples if request_begin <= sample["realtime_midpoint_ns"] <= request_end]

    process_live = [{
        "process_range": row["process_range"],
        "event_id": row.get("event_id", ""),
        "stage": row.get("stage", ""),
        "begin_ns": row["hiptx_begin_ns"],
        "end_ns": row["hiptx_end_ns"],
        "sample_count": row["live_sample_count"],
        "mean_se_active_cu_pct": row["mean_se_active_cu_pct"],
        "max_se_active_cu_pct": row["max_se_active_cu_pct"],
        "maximum_alignment_uncertainty_ns": row["maximum_sample_alignment_uncertainty_ns"],
        "status": row["live_utilization_status"],
        "evidence_class": "observed" if row["live_utilization_status"] == "observed" else "unavailable",
    } for row in normalized_process]

    runtime_by_process: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_runtime_calls:
        if (row["process_owner"], str(row["runtime_index"])) in launch_runtime_pairs:
            runtime_by_process[row["process_owner"]].append(row)
    launch_gaps: list[dict[str, Any]] = []
    for process in normalized_process:
        marker = process["process_range"]
        begin, end = int(process["hiptx_begin_ns"]), int(process["hiptx_end_ns"])
        calls = sorted(runtime_by_process.get(marker, []), key=lambda row: (int(row["begin_ns"]), int(row["end_ns"])))
        boundaries: list[tuple[int, int, str, str, str]] = []
        if calls:
            boundaries.append((begin, int(calls[0]["begin_ns"]), "before_first_launch", "", str(calls[0]["runtime_index"])))
            for previous, current in zip(calls, calls[1:]):
                boundaries.append((int(previous["end_ns"]), int(current["begin_ns"]), "between_launches", str(previous["runtime_index"]), str(current["runtime_index"])))
            boundaries.append((int(calls[-1]["end_ns"]), end, "after_last_launch", str(calls[-1]["runtime_index"]), ""))
        else:
            boundaries.append((begin, end, "no_runtime_launch_observed", "", ""))
        emitted_gap_count = 0
        for index, (gap_begin, gap_end, kind, previous_index, next_index) in enumerate(boundaries):
            if gap_end <= gap_begin:
                continue
            duration = gap_end - gap_begin
            kernels_peak, queues_peak, low_concurrency_ns = concurrency_window(
                gap_begin, gap_end
            )
            resource_exposed_ns, resource_status, resource_reason = (
                resource_coexistence_window(
                    marker, next_index, gap_begin, gap_end
                )
            )
            launch_gaps.append({
                "gap_id": f"{marker}:gap:{index}",
                "process_range": marker,
                "event_id": process.get("event_id", ""),
                "stage": process.get("stage", ""),
                "gap_kind": kind,
                "begin_ns": gap_begin,
                "end_ns": gap_end,
                "duration_ns": duration,
                "previous_runtime_index": previous_index,
                "next_runtime_index": next_index,
                "material_gap": duration >= args.minimum_launch_gap_ns,
                "peak_active_kernel_count": kernels_peak,
                "peak_active_queue_count": queues_peak,
                "low_kernel_concurrency_exposed_ns": low_concurrency_ns,
                "resource_coexistence_exposed_ns": resource_exposed_ns,
                "resource_coexistence_status": resource_status,
                "resource_coexistence_reason": resource_reason,
                "timing_source": "observed_r07_host_runtime_gap",
                "evidence_class": "observed",
            })
            emitted_gap_count += 1
        if emitted_gap_count == 0:
            launch_gaps.append({
                "gap_id": f"{marker}:gap:none",
                "process_range": marker,
                "event_id": process.get("event_id", ""),
                "stage": process.get("stage", ""),
                "gap_kind": "no_positive_launch_gap_observed",
                "begin_ns": begin,
                "end_ns": begin,
                "duration_ns": 0,
                "previous_runtime_index": "",
                "next_runtime_index": "",
                "material_gap": False,
                "peak_active_kernel_count": 0,
                "peak_active_queue_count": 0,
                "low_kernel_concurrency_exposed_ns": 0,
                "resource_coexistence_exposed_ns": 0,
                "resource_coexistence_status": "unavailable",
                "resource_coexistence_reason": "no_positive_launch_gap_observed",
                "timing_source": "observed_r07_host_runtime_gap_audit",
                "evidence_class": "observed_absence",
            })

    dependency_rows: list[dict[str, Any]] = []
    for edge in edge_rows:
        event_id = edge.get("event_id", "")
        source_stage = edge.get("source_stage", "")
        target_stage = edge.get("target_stage", "")
        source = process_by_stage.get((event_id, source_stage))
        target = process_by_stage.get((event_id, target_stage))
        verified = edge.get("edge_type") == "data" and truthy(edge.get("verified"))
        interval_available = source is not None and target is not None
        ready_ns = int(source["hiptx_end_ns"]) if verified and interval_available else None
        target_begin_ns = int(target["hiptx_begin_ns"]) if interval_available else None
        slack_ns = target_begin_ns - ready_ns if ready_ns is not None and target_begin_ns is not None else None
        if not verified:
            state = "unknown_dependency"
        elif not interval_available:
            state = "unavailable_process_interval"
        elif slack_ns is not None and slack_ns >= -args.slack_tolerance_ns:
            state = "verified_ready_state"
        else:
            state = "verified_overlap_or_wait"
        dependency_rows.append({
            **edge,
            "source_process_range": source.get("process_range", "") if source else "",
            "target_process_range": target.get("process_range", "") if target else "",
            "source_end_ns": ready_ns if ready_ns is not None else "",
            "target_begin_ns": target_begin_ns if target_begin_ns is not None else "",
            "ready_time_ns": ready_ns if ready_ns is not None else "",
            "slack_ns": slack_ns if slack_ns is not None else "",
            "dependency_state": state,
            "dependency_gate_pass": verified and interval_available,
            "slack_gate_pass": slack_ns is not None and slack_ns >= args.slack_tolerance_ns,
            "timing_source": "observed_r07_process_intervals_plus_structural_fx_edge",
        })
    verified_dependency_count = sum(truthy(row["dependency_gate_pass"]) for row in dependency_rows)
    dependency_coverage = verified_dependency_count / len(dependency_rows) if dependency_rows else 0.0

    attachments: list[dict[str, Any]] = []
    for process in normalized_process:
        key = (process.get("event_id", ""), process.get("stage", ""))
        traffic = traffic_by_stage.get(key, {})
        resources = resource_by_stage.get(key, []) or [{}]
        for resource in resources:
            attachments.append({
                "process_range": process["process_range"],
                "event_id": process.get("event_id", ""),
                "stage": process.get("stage", ""),
                "fx_visible_total_io_bytes": traffic.get("fx_visible_total_io_bytes", "unavailable"),
                "traffic_completeness": traffic.get("traffic_completeness", "unavailable"),
                "hbm_or_dram_bytes": "unavailable",
                "matched_kernel_family": resource.get("matched_kernel_family", ""),
                "work_group_size": resource.get("work_group_size", "unavailable"),
                "vgpr_count": resource.get("vgpr_count", "unavailable"),
                "sgpr_count": resource.get("sgpr_count", "unavailable"),
                "shared_memory_size_bytes": resource.get("shared_memory_size_bytes", "unavailable"),
                "theoretical_occupancy_upper_bound_pct": resource.get("theoretical_occupancy_upper_bound_pct", "unavailable"),
                "resource_evidence_complete": resource.get("resource_evidence_complete", False),
                "traffic_evidence_class": "inferred_fx_visible",
                "resource_evidence_class": resource.get("hardware_evidence_class", "unavailable"),
            })

    opportunities: list[dict[str, Any]] = []
    deps_by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dependency_rows:
        deps_by_target[(row.get("event_id", ""), row.get("target_stage", ""))].append(row)
    attachments_by_process: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attachments:
        attachments_by_process[row["process_range"]].append(row)
    gaps_by_process: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in launch_gaps:
        gaps_by_process[row["process_range"]].append(row)
    for process in normalized_process:
        marker = process["process_range"]
        begin, end = int(process["hiptx_begin_ns"]), int(process["hiptx_end_ns"])
        duration = max(0, end - begin)
        exposed = sum(int(row["duration_ns"]) for row in gaps_by_process[marker] if truthy(row["material_gap"]))
        exposed_fraction = exposed / duration if duration else 0.0
        queue_feasible_exposed = sum(
            int(row["low_kernel_concurrency_exposed_ns"])
            for row in gaps_by_process[marker]
            if truthy(row["material_gap"])
        )
        queue_feasible_exposed_fraction = (
            queue_feasible_exposed / duration if duration else 0.0
        )
        resource_coexistence_exposed = sum(
            int(row["resource_coexistence_exposed_ns"])
            for row in gaps_by_process[marker]
            if truthy(row["material_gap"])
        )
        resource_coexistence_exposed_fraction = (
            resource_coexistence_exposed / duration if duration else 0.0
        )
        kernel_peak, queue_peak, _ = concurrency_window(begin, end)
        deps = deps_by_target[(process.get("event_id", ""), process.get("stage", ""))]
        resource_rows_for_process = attachments_by_process[marker]
        dependency_gate = bool(deps) and all(truthy(row["dependency_gate_pass"]) for row in deps)
        ready_time_ns = (
            max(int(row["source_end_ns"]) for row in deps)
            if dependency_gate
            else None
        )
        dependency_slack_ns = begin - ready_time_ns if ready_time_ns is not None else None
        slack_gate = (
            dependency_slack_ns is not None
            and dependency_slack_ns >= args.slack_tolerance_ns
        )
        queue_gate = queue_feasible_exposed > 0
        resource_model_available = any(
            truthy(row.get("resource_evidence_complete"))
            for row in resource_rows_for_process
        )
        resource_gate = resource_coexistence_exposed > 0
        exposure_gate = (
            resource_coexistence_exposed >= args.minimum_exposed_duration_ns
            and resource_coexistence_exposed_fraction >= args.minimum_exposed_fraction
        )
        live_observed = process["live_utilization_status"] == "observed"
        utilization_gate = live_observed and float(process["mean_se_active_cu_pct"]) <= args.low_se_utilization_pct
        evidence_gate = (
            live_observed
            and process.get("traffic_completeness") in {"complete_fx_visible", "lower_bound"}
            and resource_model_available
            and dependency_coverage >= args.minimum_dependency_coverage
        )
        gates = {
            "dependency": dependency_gate,
            "slack": slack_gate,
            "queue_feasibility": queue_gate,
            "resource_coexistence": resource_gate,
            "exposure": exposure_gate,
            "utilization": utilization_gate,
            "evidence_quality": evidence_gate,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if all(gates.values()):
            status = "confirmed"
        elif live_observed or deps or resource_gate:
            status = "candidate"
        else:
            status = "unavailable"
        opportunities.append({
            "process_range": marker,
            "event_id": process.get("event_id", ""),
            "stage": process.get("stage", ""),
            "status": status,
            "dependency_gate": dependency_gate,
            "slack_gate": slack_gate,
            "queue_feasibility_gate": queue_gate,
            "resource_coexistence_gate": resource_gate,
            "exposure_gate": exposure_gate,
            "utilization_gate": utilization_gate,
            "evidence_quality_gate": evidence_gate,
            "failed_gates_json": json.dumps(failed, separators=(",", ":")),
            "exposed_duration_ns": exposed,
            "exposed_fraction": exposed_fraction,
            "queue_feasible_exposed_duration_ns": queue_feasible_exposed,
            "queue_feasible_exposed_fraction": queue_feasible_exposed_fraction,
            "resource_coexistence_exposed_duration_ns": resource_coexistence_exposed,
            "resource_coexistence_exposed_fraction": resource_coexistence_exposed_fraction,
            "resource_model_available": resource_model_available,
            "ready_time_ns": ready_time_ns if ready_time_ns is not None else "unavailable",
            "dependency_slack_ns": (
                dependency_slack_ns
                if dependency_slack_ns is not None
                else "unavailable"
            ),
            "peak_active_kernel_count": kernel_peak,
            "peak_active_queue_count": queue_peak,
            "mean_se_active_cu_pct": process["mean_se_active_cu_pct"],
            "claim_boundary": "scheduling_candidate_only_no_speedup_claim",
        })

    high_latency = sorted(normalized_process, key=lambda row: (-float(row["hiptx_cpu_ms"]), row["process_range"]))[: min(args.high_latency_count, len(normalized_process))]
    if not high_latency:
        raise AnalysisError("no high-latency processes")
    high_with_live = sum(row["live_utilization_status"] == "observed" for row in high_latency)
    if high_with_live != len(high_latency):
        raise AnalysisError(
            "a required high-latency process lacks enough aligned live samples"
        )

    table_rows: dict[str, list[dict[str, Any]]] = {
        "request_timeline": request_timeline,
        "process_timeline": normalized_process,
        "kernel_timeline": normalized_kernels,
        "live_utilization_aligned": normalized_samples,
        "process_live_utilization": process_live,
        "kernel_concurrency": kernel_concurrency,
        "queue_concurrency": queue_concurrency,
        "launch_gaps": launch_gaps,
        "high_latency_processes": high_latency,
        "dependency_state": dependency_rows,
        "traffic_resource_attachment": attachments,
        "opportunity_candidates": opportunities,
    }
    empty_tables = sorted(key for key, rows in table_rows.items() if not rows)
    if empty_tables:
        raise AnalysisError(f"required normalized tables are empty: {empty_tables}")
    default_fields = {
        "request_timeline": ["track_type", "event_key", "parent_key", "label", "begin_ns", "end_ns", "duration_ns", "event_id", "forward_id", "layer", "occurrence", "phase", "process_owner", "runtime_index", "queue_id", "timing_source", "evidence_class"],
        "process_timeline": ["process_range", "event_id", "stage", "hiptx_begin_ns", "hiptx_end_ns", "hiptx_cpu_ms", "duration_ns", "live_sample_count", "live_utilization_status", "mean_se_active_cu_pct", "max_se_active_cu_pct", "verified_dependency_edge_count", "unknown_dependency_count", "fx_visible_total_io_bytes", "traffic_completeness", "resource_family_count", "resource_families_json", "timing_source", "evidence_class"],
        "kernel_timeline": ["kernel_id", "begin_ns", "end_ns", "duration_ns", "device_id", "queue_id", "kernel_name", "kernel_family", "process_owner", "timing_source", "evidence_class"],
        "live_utilization_aligned": ["sequence", "realtime_midpoint_ns", "alignment_uncertainty_ns", "mean_se_active_cu_pct", "max_se_active_cu_pct", "se_active_cu_pct_json", "alignment_status", "eligible_for_process_attribution", "evidence_class", "metric_semantics"],
        "process_live_utilization": ["process_range", "event_id", "stage", "begin_ns", "end_ns", "sample_count", "mean_se_active_cu_pct", "max_se_active_cu_pct", "maximum_alignment_uncertainty_ns", "status", "evidence_class"],
        "kernel_concurrency": ["begin_ns", "end_ns", "duration_ns", "active_kernel_count", "active_kernel_ids", "timing_source", "evidence_class"],
        "queue_concurrency": ["begin_ns", "end_ns", "duration_ns", "active_queue_count", "active_queue_ids", "active_kernel_count", "timing_source", "evidence_class"],
        "launch_gaps": ["gap_id", "process_range", "event_id", "stage", "gap_kind", "begin_ns", "end_ns", "duration_ns", "previous_runtime_index", "next_runtime_index", "material_gap", "peak_active_kernel_count", "peak_active_queue_count", "low_kernel_concurrency_exposed_ns", "resource_coexistence_exposed_ns", "resource_coexistence_status", "resource_coexistence_reason", "timing_source", "evidence_class"],
        "high_latency_processes": ["process_range", "event_id", "stage", "hiptx_begin_ns", "hiptx_end_ns", "hiptx_cpu_ms", "duration_ns", "live_sample_count", "live_utilization_status", "mean_se_active_cu_pct", "max_se_active_cu_pct", "verified_dependency_edge_count", "unknown_dependency_count", "fx_visible_total_io_bytes", "traffic_completeness", "resource_family_count", "resource_families_json", "timing_source", "evidence_class"],
        "dependency_state": ["edge_id", "event_id", "source_stage", "target_stage", "edge_type", "verified", "source_process_range", "target_process_range", "source_end_ns", "target_begin_ns", "ready_time_ns", "slack_ns", "dependency_state", "dependency_gate_pass", "slack_gate_pass", "evidence_class", "timing_source", "reason"],
        "traffic_resource_attachment": ["process_range", "event_id", "stage", "fx_visible_total_io_bytes", "traffic_completeness", "hbm_or_dram_bytes", "matched_kernel_family", "work_group_size", "vgpr_count", "sgpr_count", "shared_memory_size_bytes", "theoretical_occupancy_upper_bound_pct", "resource_evidence_complete", "traffic_evidence_class", "resource_evidence_class"],
        "opportunity_candidates": ["process_range", "event_id", "stage", "status", "dependency_gate", "slack_gate", "queue_feasibility_gate", "resource_coexistence_gate", "exposure_gate", "utilization_gate", "evidence_quality_gate", "failed_gates_json", "exposed_duration_ns", "exposed_fraction", "queue_feasible_exposed_duration_ns", "queue_feasible_exposed_fraction", "resource_coexistence_exposed_duration_ns", "resource_coexistence_exposed_fraction", "resource_model_available", "ready_time_ns", "dependency_slack_ns", "peak_active_kernel_count", "peak_active_queue_count", "mean_se_active_cu_pct", "claim_boundary"],
    }
    normalized_tables: dict[str, dict[str, Any]] = {}
    for key, filename in TABLE_FILES.items():
        path = args.output_dir / filename
        rows = table_rows[key]
        fields = list(rows[0]) if rows else default_fields[key]
        write_csv(path, fields, rows)
        normalized_tables[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "row_count": len(rows),
        }

    track_counts = Counter(row["track_type"] for row in request_timeline)
    if track_counts.get("request") != 1:
        raise AnalysisError("request hierarchy must contain exactly one request track")
    for required_track in ("forward", "layer", "hip_runtime"):
        if track_counts.get(required_track, 0) <= 0:
            raise AnalysisError(f"request hierarchy lacks {required_track} tracks")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_type": "fresh_run_full_request_e2e",
        "lineage_id": lineage_id,
        "contract_id": metadata.get("contract_id"),
        "contract_sha256": metadata.get("contract_sha256"),
        "full_request_observed_timeline": True,
        "request_begin_realtime_ns": request_begin,
        "request_end_realtime_ns": request_end,
        "request_duration_ns": request_end - request_begin,
        "process_count": len(normalized_process),
        "expected_process_count": len(expected),
        "strict_owned_kernel_count": len(normalized_kernels),
        "strict_owned_kernel_timeline": True,
        "strict_owned_process_gpu_busy_union_validated": True,
        "high_latency_process_count": len(high_latency),
        "high_latency_processes_with_live_samples": high_with_live,
        "all_high_latency_processes_have_live_samples": True,
        "minimum_live_samples_per_process": args.minimum_live_samples,
        "maximum_clock_alignment_error_ns": args.maximum_clock_alignment_error_ns,
        "aligned_live_sample_count": len(normalized_samples),
        "eligible_aligned_live_sample_count": sum(
            truthy(row["eligible_for_process_attribution"])
            for row in normalized_samples
        ),
        "fresh_run_dependency_adapter_consumed": True,
        "traffic_resource_model_consumed": True,
        "dependency_coverage": dependency_coverage,
        "track_type_counts": dict(sorted(track_counts.items())),
        "opportunity_status_counts": dict(sorted(Counter(row["status"] for row in opportunities).items())),
        "configured_gates": {
            "low_se_utilization_pct": args.low_se_utilization_pct,
            "low_kernel_concurrency_max": args.low_kernel_concurrency_max,
            "minimum_launch_gap_ns": args.minimum_launch_gap_ns,
            "minimum_dependency_coverage": args.minimum_dependency_coverage,
            "minimum_exposed_duration_ns": args.minimum_exposed_duration_ns,
            "minimum_exposed_fraction": args.minimum_exposed_fraction,
            "slack_tolerance_ns": args.slack_tolerance_ns,
            "require_all_seven_gates": True,
        },
        "latency_boundary": "all durations are observed R07 non-replay intervals",
        "hardware_boundary": "SE utilization is same-request observed; PMC resources are replay-projected; FX bytes are logical and not HBM traffic",
        "inputs": {str(path.resolve()): sha256_file(path.resolve()) for path in inputs},
        "normalized_tables": normalized_tables,
    }
    manifest_path = args.output_dir / "fresh_e2e_analysis.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
