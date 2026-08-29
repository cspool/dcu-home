#!/usr/bin/env python3
"""Generate deterministic, self-contained fresh-run R10 acceptance views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TABLES = (
    "request_timeline",
    "process_timeline",
    "kernel_timeline",
    "live_utilization_aligned",
    "process_live_utilization",
    "kernel_concurrency",
    "queue_concurrency",
    "launch_gaps",
    "high_latency_processes",
    "dependency_state",
    "traffic_resource_attachment",
    "opportunity_candidates",
)
REQUIRED_TRACK_GROUPS = (
    "request",
    "forward",
    "layer",
    "process",
    "hip_runtime",
    "gpu_queue",
    "strict_owned_kernel",
    "live_utilization",
    "hardware_attributes",
    "dependency",
    "opportunity",
)
PAGE_NAMES = (
    "index.html",
    "E2E_PROCESS_TIMELINE.html",
    "E2E_PROCESS_TIMELINE_LOSSLESS.html",
    "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html",
    "CONCURRENCY_UTILIZATION.html",
)
PERFETTO_TRACE = "E2E_PROCESS_TIMELINE.full.perfetto.json"
FULL_TIMELINE_MANIFEST = "full_timeline_manifest.json"
TOP_LATENCY_PROCESS_COLOR_COUNT = 10
TOP_LATENCY_PROCESS_PALETTE = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
REQUEST_SPAN_RATIO_CAVEAT = (
    "Overlapping process intervals are not additive end-to-end attribution."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV lacks a header: {path}")
        return list(reader)


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def compact_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def checked_reference(
    record: Any, *, name: str, required_root: Path
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"missing evidence reference: {name}")
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"invalid evidence path: {name}")
    path = Path(path_value).expanduser().resolve()
    if not is_under(path, required_root):
        raise RuntimeError(f"evidence escaped current run: {name}: {path}")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"evidence is missing or changed: {name}: {path}")
    return path


def discover_runtime_root(analysis_manifest: Path) -> Path | None:
    path = analysis_manifest.resolve()
    if (
        path.parent.name == "analysis"
        and path.parents[1].name == "R09"
        and path.parents[2].name == "artifacts"
    ):
        return path.parents[3]
    return None


def load_runtime_context(
    analysis_manifest: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Strictly bind production paths to R06-R09; allow isolated test fixtures."""
    runtime_root = discover_runtime_root(analysis_manifest)
    if runtime_root is None:
        return {
            "strict_same_run_validation": False,
            "runtime_root": None,
            "handoff_hashes": {},
            "r09_input_hashes_verified": False,
            "hardware_rows": [],
            "device_capabilities": {},
            "presentation_backend": {
                "policy": "open_source_first_with_labeled_custom_fallback",
                "selected_backend": "custom_canvas_timeline_fallback",
                "official_perfetto_python_status": "not_probed_in_fixture",
                "official_perfetto_cli_status": "not_probed_in_fixture",
                "compatible_trace_classification": "complete_structural_trace_unparsed",
                "official_parse_performed": False,
            },
            "evidence_references": {},
        }

    handoffs: dict[str, dict[str, Any]] = {}
    handoff_hashes: dict[str, str] = {}
    lineage_id = manifest.get("lineage_id")
    for goal in ("R06", "R07", "R08", "R09"):
        path = runtime_root / "handoffs" / f"{goal}.json"
        if not path.is_file():
            raise RuntimeError(f"required same-run handoff is missing: {path}")
        value = load_json(path)
        if (
            value.get("runtime_goal") != goal
            or value.get("status") != "complete"
            or value.get("execution_status") != "complete"
            or value.get("fresh_e2e_evidence", {}).get("lineage_id") != lineage_id
        ):
            raise RuntimeError(f"same-run handoff is incomplete or mismatched: {goal}")
        handoffs[goal] = value
        handoff_hashes[goal] = sha256_file(path)

    binding = handoffs["R09"].get("same_run_binding", {})
    for goal in ("R06", "R07", "R08"):
        if binding.get(f"{goal}_handoff_sha256") != handoff_hashes[goal]:
            raise RuntimeError(f"R09 binding drifted for {goal}")
    if binding.get("analysis_manifest_sha256") != sha256_file(analysis_manifest):
        raise RuntimeError("R09 analysis manifest hash drifted")

    for raw_path, expected_hash in manifest.get("inputs", {}).items():
        path = Path(raw_path).expanduser().resolve()
        if (
            not is_under(path, runtime_root)
            or not path.is_file()
            or sha256_file(path) != expected_hash
        ):
            raise RuntimeError(f"R09 input is outside the run, missing, or changed: {path}")

    r06 = handoffs["R06"]
    capability = r06.get("visualization_capability", {})
    if (
        capability.get("policy") != "open_source_first_with_labeled_custom_fallback"
        or capability.get("selected_backend") != "custom_plotly_timeline_fallback"
        or capability.get("official_perfetto_python_status") != "unavailable"
        or capability.get("official_perfetto_cli_status") != "unavailable"
        or capability.get("network_download_performed") is not False
        or capability.get("runtime_measurement_reference_count") != 0
    ):
        raise RuntimeError("R06 visualization capability contract is incompatible")
    attempts_path = checked_reference(
        r06.get("primary_outputs", {}).get("open_source_trace_attempts"),
        name="R06 open-source trace attempts",
        required_root=runtime_root,
    )
    probe_path = checked_reference(
        r06.get("primary_outputs", {}).get("tool_capability_probe"),
        name="R06 tool capability probe",
        required_root=runtime_root,
    )

    r08_outputs = handoffs["R08"].get("primary_outputs", {})
    device_path = checked_reference(
        r08_outputs.get("device_capabilities"),
        name="R08 device capabilities",
        required_root=runtime_root,
    )
    hardware_path = checked_reference(
        r08_outputs.get("hardware_metrics_by_kernel_family"),
        name="R08 hardware metrics",
        required_root=runtime_root,
    )
    traffic_path = checked_reference(
        r08_outputs.get("traffic_resource_model"),
        name="R08 traffic/resource model",
        required_root=runtime_root,
    )
    hardware_rows = read_csv(hardware_path)
    if not hardware_rows:
        raise RuntimeError("R08 hardware metrics are empty")
    for row in hardware_rows:
        if (
            truth(row.get("pmc_replay_timing_used_as_latency"))
            or row.get("latency_axis") != "R07_non_replay_same_request_only"
            or row.get("cross_capture_timeline_policy")
            != "separate_clock_axes_no_merge"
        ):
            raise RuntimeError("R08 hardware row conflates replay and R07 clocks")
    device = load_json(device_path)
    traffic = load_json(traffic_path)
    if (
        traffic.get("lineage_id") != lineage_id
        or traffic.get("traffic_boundary", {}).get("hbm_or_dram_traffic_claimed")
        is not False
        or traffic.get("resource_boundary", {}).get("achieved_occupancy_claimed")
        is not False
    ):
        raise RuntimeError("R08 traffic/resource semantics are incompatible")

    r07 = handoffs["R07"]
    r07_refs = {
        "raw_queryable_trace": r07.get("capture", {}).get("raw_queryable_trace"),
        "runtime_layer_events": r07.get("capture", {}).get("runtime_layer_events"),
        "runtime_calls": r07.get("observed_process_timeline", {}).get("runtime_calls"),
        "process_performance": r07.get("observed_process_timeline", {}).get("process_performance"),
        "process_gpu_timeline": r07.get("observed_process_timeline", {}).get("process_gpu_timeline"),
        "live_samples": r07.get("live_utilization", {}).get("raw_samples"),
    }
    for name, record in r07_refs.items():
        checked_reference(record, name=f"R07 {name}", required_root=runtime_root)

    evidence_references = {
        "R06_open_source_trace_attempts": {
            "path": str(attempts_path), "sha256": sha256_file(attempts_path)
        },
        "R06_tool_capability_probe": {
            "path": str(probe_path), "sha256": sha256_file(probe_path)
        },
        "R07_raw_queryable_trace": r07_refs["raw_queryable_trace"],
        "R07_process_performance": r07_refs["process_performance"],
        "R07_process_gpu_timeline": r07_refs["process_gpu_timeline"],
        "R07_live_samples": r07_refs["live_samples"],
        "R08_device_capabilities": r08_outputs["device_capabilities"],
        "R08_hardware_metrics": r08_outputs["hardware_metrics_by_kernel_family"],
        "R08_traffic_resource_model": r08_outputs["traffic_resource_model"],
    }
    return {
        "strict_same_run_validation": True,
        "runtime_root": runtime_root,
        "handoff_hashes": handoff_hashes,
        "r09_input_hashes_verified": True,
        "hardware_rows": hardware_rows,
        "device_capabilities": device,
        "presentation_backend": {
            "policy": capability["policy"],
            "r06_selected_backend": capability["selected_backend"],
            "selected_backend": "self_contained_custom_canvas_timeline_fallback",
            "official_perfetto_python_status": capability[
                "official_perfetto_python_status"
            ],
            "official_perfetto_cli_status": capability[
                "official_perfetto_cli_status"
            ],
            "compatible_trace_classification": (
                "complete_normalized_perfetto_chrome_structural_trace"
            ),
            "official_parse_performed": False,
            "presentation_capability_degradation": True,
            "runtime_evidence_degraded": False,
        },
        "evidence_references": evidence_references,
    }


def checked_table(
    manifest: dict[str, Any], key: str, runtime_root: Path | None
) -> tuple[Path, list[dict[str, str]]]:
    record = manifest.get("normalized_tables", {}).get(key)
    if not isinstance(record, dict):
        raise RuntimeError(f"analysis manifest lacks normalized table: {key}")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if runtime_root is not None and not is_under(path, runtime_root):
        raise RuntimeError(f"normalized table escaped current run: {key}: {path}")
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"normalized table is missing or changed: {key}: {path}")
    rows = read_csv(path)
    if record.get("row_count") != len(rows):
        raise RuntimeError(f"normalized table row count changed: {key}")
    return path, rows


def process_phase(row: dict[str, str]) -> str:
    direct = row.get("phase", "")
    if direct:
        return direct
    parent = row.get("parent_layer_range", "").lower()
    if "prefill" in parent:
        return "prefill"
    if "decode" in parent:
        return "decode"
    return ""


def selected_hardware_row(row: dict[str, str]) -> dict[str, str]:
    fields = (
        "event_id", "stage", "process_range", "matched_kernel_family",
        "hardware_evidence_class", "DCU_activity_processed_ALU_pct",
        "DCU_matrix_core_utilization_proxy_pct", "L2_hit_rate_pct",
        "mean_L2_read_KB_per_replay_instance",
        "mean_L2_write_KB_per_replay_instance",
        "projected_L2_bytes_on_R07_instance_count",
        "projected_L2_throughput_GBps_on_R07_latency_axis",
        "DRAM_throughput", "DRAM_unavailable_reason", "work_group_size",
        "VGPR_count", "SGPR_count", "shared_memory_size_bytes",
        "theoretical_occupancy_upper_bound_pct", "occupancy_interpretation",
        "achieved_occupancy_pct", "strongest_available_stall_proxy",
        "strongest_available_stall_proxy_value", "latency_axis",
        "pmc_replay_timing_used_as_latency", "cross_capture_timeline_policy",
        "minimum_selected_name_order_match_rate", "selected_exact_attribution_rate",
    )
    return {field: row.get(field, "") for field in fields}


def build_top_latency_process_contract(
    process_rows: list[dict[str, str]], request_begin: int, request_end: int
) -> dict[str, Any]:
    request_span = request_end - request_begin
    if request_span <= 0:
        raise RuntimeError("request span must be positive for process ranking")
    seen: set[str] = set()
    ranked: list[tuple[int, int, str]] = []
    for index, row in enumerate(process_rows):
        process_range = row.get("process_range", "").strip()
        if not process_range:
            raise RuntimeError(f"process timeline row {index} lacks process_range")
        if process_range in seen:
            raise RuntimeError(
                f"duplicate process_range in process timeline: {process_range}"
            )
        seen.add(process_range)
        try:
            begin = int(row["hiptx_begin_ns"])
            end = int(row["hiptx_end_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"process timeline row has invalid HIPTX bounds: {process_range}"
            ) from exc
        if end < begin:
            raise RuntimeError(
                f"process timeline row has negative duration: {process_range}"
            )
        ranked.append((end - begin, begin, process_range))
    total_duration = sum(duration for duration, _, _ in ranked)
    if total_duration <= 0:
        raise RuntimeError("observed process duration total must be positive")
    ranked.sort(key=lambda value: (-value[0], value[1], value[2]))
    selected = []
    for rank, (duration, begin, process_range) in enumerate(
        ranked[:TOP_LATENCY_PROCESS_COLOR_COUNT], start=1
    ):
        selected.append(
            {
                "rank": rank,
                "process_range": process_range,
                "hiptx_begin_ns": begin,
                "observed_duration_ns": duration,
                "observed_process_duration_share": duration / total_duration,
                "observed_request_span_ratio": duration / request_span,
                "request_span_ratio_caveat": REQUEST_SPAN_RATIO_CAVEAT,
                "color": TOP_LATENCY_PROCESS_PALETTE[rank - 1],
            }
        )
    return {
        "schema_version": 1,
        "ranking_source": "complete_immutable_R09_process_timeline",
        "ranking_duration": "hiptx_end_ns - hiptx_begin_ns",
        "ranking_order": [
            "observed_duration_ns_descending",
            "hiptx_begin_ns_ascending",
            "process_range_ascending",
        ],
        "configured_count": TOP_LATENCY_PROCESS_COLOR_COUNT,
        "selected_count": len(selected),
        "palette": list(TOP_LATENCY_PROCESS_PALETTE),
        "observed_process_duration_total_ns": total_duration,
        "observed_request_span_ns": request_span,
        "request_span_ratio_caveat": REQUEST_SPAN_RATIO_CAVEAT,
        "selected": selected,
    }


def build_lossless_timeline_payload(
    begin: int,
    end: int,
    timeline_rows: list[dict[str, Any]],
    top_latency_process_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "origin_ns": str(begin),
        "begin": 0,
        "end": end - begin,
        "groups": ["request", "forward", "layer", "process", "hip_runtime",
                   "gpu_queue", "strict_owned_kernel"],
        "top_latency_processes": top_latency_process_contract["selected"],
        "top_latency_process_policy": {
            key: value
            for key, value in top_latency_process_contract.items()
            if key != "selected"
        },
        "rows": [
            {
                **{key: value for key, value in row.items() if key not in {"b", "e"}},
                "b": row["b"] - begin,
                "e": row["e"] - begin,
                "b_abs": str(row["b"]),
                "e_abs": str(row["e"]),
            }
            for row in timeline_rows
        ],
    }


def build_payloads(
    manifest: dict[str, Any],
    tables: dict[str, list[dict[str, str]]],
    runtime_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    begin = integer(manifest["request_begin_realtime_ns"])
    end = integer(manifest["request_end_realtime_ns"])
    top_latency_process_contract = build_top_latency_process_contract(
        tables["process_timeline"], begin, end
    )
    timeline_rows = [
        {
            "g": row["track_type"], "n": row.get("label", ""),
            "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
            "event": row.get("event_id", ""),
            "forward": row.get("forward_id", ""),
            "layer": row.get("layer", ""), "phase": row.get("phase", ""),
            "process": row.get("process_owner", ""),
            "runtime": row.get("runtime_index", ""),
            "queue": row.get("queue_id", ""),
            "family": "", "evidence": row.get("evidence_class", "observed"),
            "timing": row.get("timing_source", ""),
        }
        for row in tables["request_timeline"]
    ]
    timeline_rows.extend(
        {
            "g": "process", "n": row["process_range"],
            "b": integer(row["hiptx_begin_ns"]),
            "e": integer(row["hiptx_end_ns"]),
            "event": row.get("event_id", ""),
            "forward": row.get("forward_id", ""),
            "layer": row.get("layer", ""), "phase": process_phase(row),
            "process": row["process_range"], "runtime": "",
            "queue": row.get("strict_owned_queue_ids", ""), "family": "",
            "stage": row.get("stage", ""),
            "evidence": row.get("evidence_class", "observed"),
            "timing": row.get("timing_source", ""),
        }
        for row in tables["process_timeline"]
    )
    kernel_rows = [
        {
            "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
            "process": row.get("process_owner", ""),
            "queue": row.get("queue_id", ""),
            "n": row.get("kernel_name", ""),
            "family": row.get("kernel_family", ""),
            "runtime": row.get("runtime_index", ""),
            "kernel_id": row.get("kernel_id", ""),
            "evidence": row.get("evidence_class", "observed"),
            "timing": row.get("timing_source", ""),
        }
        for row in tables["kernel_timeline"]
    ]
    timeline_rows.extend(
        {"g": "strict_owned_kernel", "event": "", "forward": "", "layer": "",
         "phase": "", **row}
        for row in kernel_rows
    )
    timeline_rows.extend(
        {**row, "g": "gpu_queue", "n": "queue " + row["queue"], "event": "",
         "forward": "", "layer": "", "phase": ""}
        for row in kernel_rows
    )

    families_by_process: dict[str, set[str]] = defaultdict(set)
    for row in kernel_rows:
        if row["family"]:
            families_by_process[row["process"]].add(row["family"])
    high_processes = [
        {
            "process": row["process_range"],
            "b": integer(row["hiptx_begin_ns"]),
            "e": integer(row["hiptx_end_ns"]),
            "ms": number(row["hiptx_cpu_ms"]),
            "samples": integer(row.get("live_sample_count")),
            "mean": row.get("mean_se_active_cu_pct", "unavailable"),
            "max": row.get("max_se_active_cu_pct", "unavailable"),
            "live_status": row.get("live_utilization_status", "unavailable"),
            "event": row.get("event_id", ""),
            "forward": row.get("forward_id", ""),
            "layer": row.get("layer", ""),
            "phase": process_phase(row), "stage": row.get("stage", ""),
            "families": sorted(families_by_process[row["process_range"]]),
            "owned_kernel_count": integer(row.get("strict_owned_kernel_count")),
            "owned_kernel_busy_union_ms": number(
                row.get("strict_owned_kernel_busy_union_ms")
            ),
            "traffic_bytes": row.get("fx_visible_total_io_bytes", "unavailable"),
            "traffic_completeness": row.get("traffic_completeness", ""),
            "evidence": row.get("evidence_class", "observed"),
        }
        for row in tables["high_latency_processes"]
    ]
    high_names = {row["process"] for row in high_processes}
    live_rows = [
        {
            "t": integer(row["realtime_midpoint_ns"]),
            "mean": number(row.get("mean_se_active_cu_pct")),
            "max": number(row.get("max_se_active_cu_pct")),
            "uncertainty": integer(row.get("alignment_uncertainty_ns")),
            "eligible": truth(row.get("eligible_for_process_attribution")),
            "status": row.get("alignment_status", ""),
            "evidence": row.get("evidence_class", "observed"),
        }
        for row in tables["live_utilization_aligned"]
    ]
    high_live = [
        row for row in tables["process_live_utilization"]
        if row.get("process_range") in high_names
    ]
    high_opportunities = [
        row for row in tables["opportunity_candidates"]
        if row.get("process_range") in high_names
    ]

    concurrency_samples = live_rows
    gaps = [
        {
            "id": row.get("gap_id", ""), "process": row.get("process_range", ""),
            "event": row.get("event_id", ""), "stage": row.get("stage", ""),
            "kind": row.get("gap_kind", ""), "b": integer(row["begin_ns"]),
            "e": integer(row["end_ns"]), "duration": integer(row["duration_ns"]),
            "material": truth(row.get("material_gap")),
            "peak_kernels": integer(row.get("peak_active_kernel_count")),
            "peak_queues": integer(row.get("peak_active_queue_count")),
            "resource_status": row.get("resource_coexistence_status", ""),
            "resource_reason": row.get("resource_coexistence_reason", ""),
            "timing": row.get("timing_source", ""),
            "evidence": row.get("evidence_class", "observed"),
        }
        for row in tables["launch_gaps"]
    ]
    dependencies = [
        {
            "edge_id": row.get("edge_id", ""),
            "event_id": row.get("event_id", ""),
            "source_process_range": row.get("source_process_range", ""),
            "target_process_range": row.get("target_process_range", ""),
            "dependency_state": row.get("dependency_state", ""),
            "ready_time_ns": row.get("ready_time_ns", ""),
            "slack_ns": row.get("slack_ns", ""),
            "dependency_gate_pass": row.get("dependency_gate_pass", ""),
            "slack_gate_pass": row.get("slack_gate_pass", ""),
            "evidence_class": row.get("evidence_class", ""),
            "verified": row.get("verified", ""),
            "timing_source": row.get("timing_source", ""),
        }
        for row in tables["dependency_state"]
    ]
    kc = [
        {"b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
         "n": integer(row["active_kernel_count"]),
         "evidence": row.get("evidence_class", "observed"),
         "timing": row.get("timing_source", "")}
        for row in tables["kernel_concurrency"]
    ]
    qc = [
        {"b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
         "n": integer(row["active_queue_count"]),
         "kernels": integer(row.get("active_kernel_count")),
         "evidence": row.get("evidence_class", "observed"),
         "timing": row.get("timing_source", "")}
        for row in tables["queue_concurrency"]
    ]

    process_live_counts = dict(
        sorted(Counter(row.get("status", "unknown") for row in tables[
            "process_live_utilization"
        ]).items())
    )
    opportunity_counts = dict(
        sorted(Counter(row.get("status", "unknown") for row in tables[
            "opportunity_candidates"
        ]).items())
    )
    hardware_rows = [
        selected_hardware_row(row) for row in runtime_context["hardware_rows"]
    ]
    device = runtime_context["device_capabilities"]
    device_summary = {
        key: device.get(key)
        for key in (
            "physical_device_id", "architecture", "cu_count", "wave_size",
            "wave_limit", "thread_limit", "vgpr_resource", "shared_memory_bytes",
            "resource_semantics", "unavailable_quantities",
        )
        if key in device
    }

    return {
        "index.html": {
            "row_counts": {key: len(tables[key]) for key in REQUIRED_TABLES},
            "track_counts": manifest.get("track_type_counts", {}),
            "process_live_status_counts": process_live_counts,
            "opportunity_status_counts": opportunity_counts,
            "request_begin_ns": begin, "request_end_ns": end,
            "request_duration_ns": end - begin,
        },
        "E2E_PROCESS_TIMELINE.html": {
            "begin": begin, "end": end,
            "groups": ["request", "forward", "layer", "process", "hip_runtime",
                       "gpu_queue", "strict_owned_kernel"],
            "top_latency_processes": top_latency_process_contract["selected"],
            "top_latency_process_policy": {
                key: value
                for key, value in top_latency_process_contract.items()
                if key != "selected"
            },
            "rows": timeline_rows,
        },
        "E2E_PROCESS_TIMELINE_LOSSLESS.html": build_lossless_timeline_payload(
            begin, end, timeline_rows, top_latency_process_contract
        ),
        "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html": {
            "processes": high_processes, "kernels": kernel_rows,
            "samples": live_rows,
            "process_live": high_live,
            "attachments": tables["traffic_resource_attachment"],
            "hardware": hardware_rows, "device": device_summary,
            "opportunities": high_opportunities,
        },
        "CONCURRENCY_UTILIZATION.html": {
            "begin": begin, "end": end, "kernel_concurrency": kc,
            "queue_concurrency": qc, "samples": concurrency_samples,
            "gaps": gaps, "dependencies": dependencies,
            "opportunities": tables["opportunity_candidates"],
            "process_live_status_counts": process_live_counts,
            "opportunity_status_counts": opportunity_counts,
        },
    }


CSS = """
:root{color-scheme:dark;--bg:#0a0f1d;--panel:#121b2e;--line:#293957;--text:#eef4ff;--muted:#a6b5cc;--obs:#55d6be;--runtime:#7aa2f7;--kernel:#ffb454;--queue:#c099ff;--live:#66c7ff;--replay:#d38cff;--inferred:#e8cc68;--missing:#7d879b;--bad:#ff6b7a;--good:#59d499}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}main{max-width:1700px;margin:auto;padding:18px}a{color:#8abaff}nav{display:flex;gap:13px;flex-wrap:wrap;padding:8px 0}.note,.panel,.backend{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px;margin:11px 0}.backend{border-color:#8b6bc6}.controls{display:flex;gap:11px;align-items:center;flex-wrap:wrap;margin:12px 0}.controls input[type=text],.controls input[type=number],.controls select{min-width:175px;background:#0d1424;color:var(--text);border:1px solid var(--line);padding:7px}.controls input[type=number]{min-width:130px;width:160px}.controls input.search{min-width:310px}.controls input[type=range]{width:205px}button{background:#26395c;color:var(--text);border:1px solid #49658f;border-radius:5px;padding:7px 11px}canvas{width:100%;height:auto;background:#0d1424;border:1px solid var(--line);border-radius:8px}.lossless-canvas{touch-action:none;cursor:grab}.lossless-canvas.dragging{cursor:grabbing}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}.legend span{color:var(--muted)}.dot{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}.top-process-legend h2{margin:0 0 8px}.top-process-items{display:flex;gap:8px 14px;flex-wrap:wrap}.top-process-item{display:inline-flex;align-items:center;max-width:100%;color:var(--text)}.top-process-name{overflow-wrap:anywhere}.top-process-swatch{display:inline-block;width:18px;height:12px;border-radius:2px;margin-right:6px;border:1px solid #f4f7ff}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#17223a}.scroll{max-height:440px;overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:9px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:11px}.big{font-size:24px;font-weight:700}.muted{color:var(--muted)}code{word-break:break-all}pre{white-space:pre-wrap;word-break:break-word}.badge{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:2px 8px;margin:2px}.provenance summary{cursor:pointer;font-weight:700}.warning{color:#ffd37d}.pass{color:var(--good)}
"""


def navigation() -> str:
    links = "".join(
        f"<a href='{html.escape(name)}'>{html.escape(name)}</a>" for name in PAGE_NAMES
    )
    return (
        "<nav>" + links
        + f"<a href='{PERFETTO_TRACE}' download>Complete Perfetto-compatible trace</a>"
        + f"<a href='{FULL_TIMELINE_MANIFEST}'>Full timeline manifest</a>"
        + "</nav>"
    )


def evidence_legend() -> str:
    items = (
        ("observed", "var(--obs)", "R07 same-request non-replay intervals"),
        ("observed live utilization", "var(--live)", "eligible RSMI SE snapshots"),
        ("replay-projected PMC/resources", "var(--replay)", "R08 attributes only; never replay latency"),
        ("inferred FX-visible traffic", "var(--inferred)", "logical tensor IO; not HBM traffic"),
        ("unavailable", "var(--missing)", "preserved missing evidence; never coerced to zero"),
    )
    spans = "".join(
        "<span data-evidence-class='" + html.escape(label) + "' title='"
        + html.escape(tip) + "'><i class='dot' style='background:" + color
        + "'></i>" + html.escape(label) + "</span>"
        for label, color, tip in items
    )
    return "<div class='legend' data-evidence-legend='complete'>" + spans + "</div>"


def top_latency_process_legend(payload: dict[str, Any]) -> str:
    selected = payload.get("top_latency_processes")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError("timeline payload lacks the top-latency process contract")
    items = []
    for entry in selected:
        rank = integer(entry.get("rank"))
        process_range = str(entry.get("process_range", ""))
        color = str(entry.get("color", ""))
        duration_ns = integer(entry.get("observed_duration_ns"))
        if rank <= 0 or not process_range or color not in TOP_LATENCY_PROCESS_PALETTE:
            raise RuntimeError("invalid top-latency process legend entry")
        items.append(
            "<span class='top-process-item' data-rank='"
            + str(rank)
            + "' data-process-range='"
            + html.escape(process_range)
            + "' data-color='"
            + html.escape(color)
            + "' title='observed HIPTX duration "
            + f"{duration_ns:,} ns'><i class='top-process-swatch' style='background:"
            + html.escape(color)
            + "'></i><span class='top-process-name'>#"
            + str(rank)
            + " "
            + html.escape(process_range)
            + "</span></span>"
        )
    return (
        "<section class='panel top-process-legend' "
        "data-top-latency-process-count='"
        + str(len(selected))
        + "'><h2>Top latency processes</h2><p class='muted'>Distinct fill "
        "colors identify the ten largest observed process HIPTX durations. "
        "Owned HIP runtime, GPU queue and strict-owned kernel rectangles use "
        "the same color as an outline. Zoom until a process rectangle is wide "
        "enough to show its complete name. Overlapping process intervals are "
        "not additive end-to-end attribution.</p><div class='top-process-items'>"
        + "".join(items)
        + "</div></section>"
    )


def provenance_panel(metadata: dict[str, Any]) -> str:
    rows = "".join(
        "<tr><td>" + html.escape(key) + "</td><td><code>"
        + html.escape(metadata["source_table_hashes"][key]) + "</code></td><td>"
        + str(metadata["source_table_row_counts"][key]) + "</td></tr>"
        for key in REQUIRED_TABLES
    )
    return (
        "<details class='panel provenance'><summary>Evidence provenance and immutable hashes</summary>"
        "<p>Lineage <code>" + html.escape(metadata["lineage_id"]) + "</code></p>"
        "<p>R09 analysis <code>" + html.escape(metadata["source_analysis_sha256"])
        + "</code></p><div class='scroll'><table><tr><th>table</th><th>SHA-256</th>"
        "<th>rows</th></tr>" + rows + "</table></div></details>"
    )


def page(
    *, title: str, heading: str, body: str, metadata: dict[str, Any],
    payload: dict[str, Any], app_javascript: str = ""
) -> str:
    backend = metadata["presentation_backend"]
    backend_text = (
        "SELF-CONTAINED CUSTOM CANVAS TIMELINE FALLBACK — official Perfetto "
        "Python/CLI unavailable. The bundled complete Chrome JSON retains every "
        "normalized interval but is not an official Perfetto parse."
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
        + navigation() + f"<h1>{html.escape(heading)}</h1>"
        + "<div class='backend' data-viewer-backend='"
        + html.escape(str(backend.get("selected_backend"))) + "'><strong>"
        + html.escape(backend_text) + "</strong></div>"
        + evidence_legend() + body + provenance_panel(metadata)
        + "<script type='application/json' id='acceptance-metadata'>"
        + compact_json(metadata) + "</script>"
        + "<script type='application/json' id='page-payload'>"
        + compact_json(payload) + "</script>"
        + ("<script>" + app_javascript + "</script>" if app_javascript else "")
        + "</main></body></html>"
    )


E2E_JS = r"""
const D=JSON.parse(document.getElementById('page-payload').textContent),C=document.getElementById('chart'),X=C.getContext('2d'),S=document.getElementById('start'),E=document.getElementById('end'),W=document.getElementById('window'),Z=document.getElementById('detail');
const F={q:document.getElementById('search'),track:document.getElementById('track-filter'),process:document.getElementById('process-filter'),event:document.getElementById('event-filter'),layer:document.getElementById('layer-filter'),phase:document.getElementById('phase-filter'),family:document.getElementById('family-filter')};
const colors={request:'#55d6be',forward:'#72ddb0',layer:'#43b8a5',process:'#4bc9b1',hip_runtime:'#7aa2f7',gpu_queue:'#c099ff',strict_owned_kernel:'#ffb454'};
const topByProcess=new Map((D.top_latency_processes||[]).map(item=>[item.process_range,item]));
const ownedGroups=new Set(['hip_runtime','gpu_queue','strict_owned_kernel']);let hits=[];
function includes(v,q){return !q||String(v??'').toLowerCase().includes(q.toLowerCase())}function match(r){return includes(JSON.stringify(r),F.q.value)&&includes(r.g,F.track.value)&&includes(r.process,F.process.value)&&includes(r.event,F.event.value)&&includes(r.layer,F.layer.value)&&includes(r.phase,F.phase.value)&&includes(r.family,F.family.value)}
function topFor(r){return topByProcess.get(String(r.process||''))}
function contrast(hex){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return .299*r+.587*g+.114*b>155?'#09111f':'#ffffff'}
function decorate(r,g,x,y,w,h,top){if(top&&ownedGroups.has(g)){X.save();X.strokeStyle=top.color;X.lineWidth=2;X.strokeRect(x+.5,y+.5,Math.max(.5,w-1),Math.max(.5,h-1));X.restore()}if(g!=='process'||w<48)return;const name=String(r.process||r.n||'');X.save();X.font='11px system-ui';const tw=X.measureText(name).width;if(w>=tw+8){X.beginPath();X.rect(x,y,w,h);X.clip();X.fillStyle=top?contrast(top.color):'#07131d';X.textBaseline='middle';X.fillText(name,x+4,y+h/2)}X.restore()}
function draw(){let lo=D.begin+(D.end-D.begin)*(+S.value/100),hi=D.begin+(D.end-D.begin)*(+E.value/100);if(hi<=lo)hi=lo+1;W.textContent=((lo-D.begin)/1e6).toFixed(3)+'–'+((hi-D.begin)/1e6).toFixed(3)+' ms on the R07 clock';let rows=D.rows.filter(r=>r.e>lo&&r.b<hi&&match(r));X.clearRect(0,0,C.width,C.height);X.font='12px system-ui';hits=[];const by={};for(const r of rows)(by[r.g]??=[]).push(r);D.groups.forEach((g,i)=>{let y=35+i*76;X.fillStyle='#a6b5cc';X.fillText(g,6,y+18);X.strokeStyle='#293957';X.beginPath();X.moveTo(140,y+30);X.lineTo(C.width-15,y+30);X.stroke();for(const r of (by[g]||[])){let x=140+(Math.max(r.b,lo)-lo)/(hi-lo)*(C.width-160),w=Math.max(1,(Math.min(r.e,hi)-Math.max(r.b,lo))/(hi-lo)*(C.width-160)),top=topFor(r),fill=g==='process'&&top?top.color:(colors[g]||'#7d879b');X.fillStyle=fill;X.fillRect(x,y,w,29);decorate(r,g,x,y,w,29,top);hits.push({x,y,w,h:29,r});}});document.getElementById('visible-count').textContent=rows.length.toLocaleString()+' visible intervals';}
Object.values(F).concat([S,E]).forEach(x=>x.oninput=draw);document.getElementById('reset').onclick=()=>{Object.values(F).forEach(x=>x.value='');S.value=0;E.value=100;draw()};C.onclick=e=>{let b=C.getBoundingClientRect(),x=(e.clientX-b.left)*C.width/b.width,y=(e.clientY-b.top)*C.height/b.height,h=hits.find(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);if(h)Z.textContent=JSON.stringify(h.r,null,2)};draw();
"""


LOSSLESS_E2E_JS = r"""
const D=JSON.parse(document.getElementById('page-payload').textContent);
const C=document.getElementById('chart'),X=C.getContext('2d'),W=document.getElementById('window'),Z=document.getElementById('detail'),VC=document.getElementById('visible-count');
const START=document.getElementById('start-ns'),END=document.getElementById('end-ns');
const F={q:document.getElementById('search'),track:document.getElementById('track-filter'),process:document.getElementById('process-filter'),event:document.getElementById('event-filter'),layer:document.getElementById('layer-filter'),phase:document.getElementById('phase-filter'),family:document.getElementById('family-filter')};
const colors={request:'#55d6be',forward:'#72ddb0',layer:'#43b8a5',process:'#4bc9b1',hip_runtime:'#7aa2f7',gpu_queue:'#c099ff',strict_owned_kernel:'#ffb454'};
const topByProcess=new Map((D.top_latency_processes||[]).map(item=>[item.process_range,item]));
const ownedGroups=new Set(['hip_runtime','gpu_queue','strict_owned_kernel']);
const labelWidth=170,laneHeight=15,history=[];
let lo=D.begin,hi=D.end,filtered=D.rows,hits=[],drag=null,filterTimer=null,framePending=false;
function inc(v,q){return !q||String(v??'').toLowerCase().includes(q.toLowerCase())}
for(const r of D.rows){r._search=[r.g,r.n,r.process,r.event,r.forward,r.layer,r.phase,r.runtime,r.queue,r.family,r.evidence,r.timing].join('\n').toLowerCase()}
function match(r){return (!F.q.value||r._search.includes(F.q.value.toLowerCase()))&&inc(r.g,F.track.value)&&inc(r.process,F.process.value)&&inc(r.event,F.event.value)&&inc(r.layer,F.layer.value)&&inc(r.phase,F.phase.value)&&inc(r.family,F.family.value)}
const layout={};
for(const g of D.groups){const ends=[];const rows=D.rows.filter(r=>r.g===g).sort((a,b)=>a.b-b.b||a.e-b.e);for(const r of rows){let lane=ends.findIndex(value=>value<=r.b);if(lane<0){lane=ends.length;ends.push(r.e)}else ends[lane]=r.e;r._lane=lane}layout[g]={lanes:Math.max(1,ends.length)}}
function canvasHeight(){return D.groups.reduce((sum,g)=>sum+Math.max(58,38+layout[g].lanes*laneHeight),28)}
function resize(){const dpr=window.devicePixelRatio||1,width=Math.max(900,C.clientWidth||1600),height=canvasHeight();if(C.width!==Math.round(width*dpr)||C.height!==Math.round(height*dpr)){C.width=Math.round(width*dpr);C.height=Math.round(height*dpr);C.style.height=height+'px'}X.setTransform(dpr,0,0,dpr,0,0);return {width,height}}
function schedule(){if(framePending)return;framePending=true;requestAnimationFrame(()=>{framePending=false;draw()})}
function clampView(a,b){const duration=D.end-D.begin;let span=Math.max(1,Math.min(duration,b-a));let left=Math.max(D.begin,Math.min(D.end-span,a));return [left,left+span]}
function setView(a,b,remember=true){if(remember)history.push([lo,hi]);[lo,hi]=clampView(a,b);START.value=(lo/1e6).toFixed(9);END.value=(hi/1e6).toFixed(9);schedule()}
function fitFiltered(){if(!filtered.length)return;let a=Infinity,b=-Infinity;for(const r of filtered){if(r.b<a)a=r.b;if(r.e>b)b=r.e}const pad=Math.max(1,(b-a)*0.05);setView(a-pad,b+pad)}
function updateFilter(autoFit=true){filtered=D.rows.filter(match);if(autoFit)fitFiltered();else schedule()}
function px(t,width){return labelWidth+(t-lo)/(hi-lo)*(width-labelWidth-16)}
function exact(r){const value={};for(const [key,item] of Object.entries(r)){if(!key.startsWith('_'))value[key]=item}value.begin_ns_exact=r.b_abs;value.end_ns_exact=r.e_abs;value.duration_ns_exact=String(BigInt(r.e_abs)-BigInt(r.b_abs));return value}
function topFor(r){return topByProcess.get(String(r.process||''))}
function contrast(hex){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return .299*r+.587*g+.114*b>155?'#09111f':'#ffffff'}
function decorate(r,g,x,y,w,h,top){if(top&&ownedGroups.has(g)){X.save();X.strokeStyle=top.color;X.lineWidth=2;X.strokeRect(x+.5,y+.5,Math.max(.5,w-1),Math.max(.5,h-1));X.restore()}if(g!=='process'||w<48)return;const name=String(r.process||r.n||'');X.save();X.font='10px system-ui';const tw=X.measureText(name).width;if(w>=tw+8){X.beginPath();X.rect(x,y,w,h);X.clip();X.fillStyle=top?contrast(top.color):'#07131d';X.textBaseline='middle';X.fillText(name,x+4,y+h/2)}X.restore()}
function draw(){const {width,height}=resize();X.clearRect(0,0,width,height);X.font='12px system-ui';hits=[];const span=hi-lo;W.textContent=`${(lo/1e6).toFixed(6)}–${(hi/1e6).toFixed(6)} ms relative to request begin; ${span.toLocaleString()} ns; ${(span/Math.max(1,width-labelWidth-16)).toFixed(3)} ns/px`;const visible=filtered.filter(r=>r.e>lo&&r.b<hi);VC.textContent=`${visible.length.toLocaleString()} visible / ${filtered.length.toLocaleString()} matching / ${D.rows.length.toLocaleString()} total intervals`;const by={};for(const r of visible)(by[r.g]??=[]).push(r);let topY=24;for(const g of D.groups){const groupHeight=Math.max(58,38+layout[g].lanes*laneHeight);X.fillStyle='#a6b5cc';X.fillText(`${g} (${layout[g].lanes} lanes)`,6,topY+18);X.strokeStyle='#293957';X.beginPath();X.moveTo(labelWidth,topY+25);X.lineTo(width-16,topY+25);X.stroke();for(const r of (by[g]||[])){const x=px(Math.max(r.b,lo),width),right=px(Math.min(r.e,hi),width),w=Math.max(.5,right-x),y=topY+30+r._lane*laneHeight,h=Math.max(7,laneHeight-3),top=topFor(r),fill=g==='process'&&top?top.color:(colors[g]||'#7d879b');X.fillStyle=fill;X.fillRect(x,y,w,h);decorate(r,g,x,y,w,h,top);hits.push({x,y,w,h,r})}topY+=groupHeight}X.fillStyle='#a6b5cc';X.font='12px system-ui';X.textBaseline='alphabetic';for(let i=0;i<=10;i++){const t=lo+span*i/10,x=px(t,width);X.fillText((t/1e6).toFixed(span<1e6?6:3)+' ms',Math.min(width-78,Math.max(labelWidth,x-28)),14)}}
function zoomAt(anchor,factor){const span=Math.max(1,(hi-lo)*factor);const ratio=(anchor-lo)/(hi-lo);setView(anchor-span*ratio,anchor+span*(1-ratio))}
Object.values(F).forEach(input=>input.addEventListener('input',()=>{clearTimeout(filterTimer);filterTimer=setTimeout(()=>updateFilter(true),250)}));
document.getElementById('fit-filter').onclick=fitFiltered;
document.getElementById('zoom-in').onclick=()=>zoomAt((lo+hi)/2,.5);
document.getElementById('zoom-out').onclick=()=>zoomAt((lo+hi)/2,2);
document.getElementById('back').onclick=()=>{const v=history.pop();if(v){[lo,hi]=v;START.value=(lo/1e6).toFixed(9);END.value=(hi/1e6).toFixed(9);schedule()}};
document.getElementById('reset').onclick=()=>{Object.values(F).forEach(input=>input.value='');filtered=D.rows;history.length=0;setView(D.begin,D.end,false)};
[START,END].forEach(input=>input.addEventListener('change',()=>setView(Number(START.value)*1e6,Number(END.value)*1e6)));
C.addEventListener('wheel',event=>{event.preventDefault();const rect=C.getBoundingClientRect(),x=event.clientX-rect.left,anchor=lo+(x-labelWidth)/Math.max(1,rect.width-labelWidth-16)*(hi-lo);zoomAt(Math.max(lo,Math.min(hi,anchor)),Math.exp(event.deltaY*.0015))},{passive:false});
C.addEventListener('pointerdown',event=>{const rect=C.getBoundingClientRect();drag={x:event.clientX,lo,hi};history.push([lo,hi]);C.classList.add('dragging');C.setPointerCapture(event.pointerId)});
C.addEventListener('pointermove',event=>{if(!drag)return;const rect=C.getBoundingClientRect(),shift=-(event.clientX-drag.x)/Math.max(1,rect.width-labelWidth-16)*(drag.hi-drag.lo);[lo,hi]=clampView(drag.lo+shift,drag.hi+shift);START.value=(lo/1e6).toFixed(9);END.value=(hi/1e6).toFixed(9);schedule()});
C.addEventListener('pointerup',event=>{drag=null;C.classList.remove('dragging');C.releasePointerCapture(event.pointerId)});
C.addEventListener('dblclick',event=>{const rect=C.getBoundingClientRect(),anchor=lo+(event.clientX-rect.left-labelWidth)/Math.max(1,rect.width-labelWidth-16)*(hi-lo);zoomAt(anchor,.25)});
C.addEventListener('click',event=>{if(drag)return;const rect=C.getBoundingClientRect(),sx=(event.clientX-rect.left)*((C.clientWidth||rect.width)/rect.width),sy=(event.clientY-rect.top)*((parseFloat(C.style.height)||rect.height)/rect.height),found=hits.filter(h=>sx>=h.x&&sx<=h.x+h.w&&sy>=h.y&&sy<=h.y+h.h);if(found.length)Z.textContent=JSON.stringify({overlap_count:found.length,events:found.slice(0,200).map(h=>exact(h.r)),truncated:found.length>200},null,2)});
window.addEventListener('resize',schedule);setView(D.begin,D.end,false);
"""


LOSSLESS_E2E_BODY = """
<div class='note'>Full-resolution, no-sampling view of every normalized request, forward, layer, process HIPTX, HIP runtime, GPU queue and strict-owned kernel interval. Times used for rendering are exact integer nanoseconds relative to request begin; original absolute nanoseconds remain decimal strings. Wheel to zoom at the pointer, drag to pan, double-click to zoom 4×, or fit the current filter.</div>
<div class='controls'><label>Search <input class='search' id='search' type='text' placeholder='any field'></label><label>Track <input id='track-filter' type='text'></label><label>Process <input id='process-filter' type='text'></label><label>Event <input id='event-filter' type='text'></label><label>Layer <input id='layer-filter' type='text'></label><label>Phase <input id='phase-filter' type='text'></label><label>Family <input id='family-filter' type='text'></label></div>
<div class='controls'><button id='fit-filter'>Fit filtered events</button><button id='zoom-in'>Zoom in 2×</button><button id='zoom-out'>Zoom out 2×</button><button id='back'>Back</button><button id='reset'>Reset all</button><label>Start relative ms <input id='start-ns' type='number' step='0.000001'></label><label>End relative ms <input id='end-ns' type='number' step='0.000001'></label></div>
<div class='controls'><span id='window'></span><span id='visible-count' class='badge'></span></div>
<div data-track-groups='request,forward,layer,process,hip_runtime,gpu_queue,strict_owned_kernel' data-sampling-performed='false'></div><canvas id='chart' class='lossless-canvas' width='1600' height='650'></canvas><pre id='detail' class='panel muted'>Click a rectangle to list every overlapping event at that pixel.</pre>
"""


HIGH_JS = r"""
const D=JSON.parse(document.getElementById('page-payload').textContent),P=document.getElementById('pick'),Q=document.getElementById('search'),EV=document.getElementById('event-filter'),LY=document.getElementById('layer-filter'),PH=document.getElementById('phase-filter'),FA=document.getElementById('family-filter'),C=document.getElementById('chart'),X=C.getContext('2d'),O=document.getElementById('details'),ST=document.getElementById('high-stats');
function inc(v,q){return !q||String(v??'').toLowerCase().includes(q.toLowerCase())}function eligible(r){return inc(JSON.stringify(r),Q.value)&&inc(r.event,EV.value)&&inc(r.layer,LY.value)&&inc(r.phase,PH.value)&&inc(r.families.join(','),FA.value)}
function options(){let old=P.value;P.innerHTML='';D.processes.filter(eligible).forEach(r=>{let o=document.createElement('option');o.value=r.process;o.textContent=r.process+' — '+r.ms.toFixed(4)+' ms';P.appendChild(o)});if([...P.options].some(o=>o.value===old))P.value=old;draw()}
function draw(){let r=D.processes.find(x=>x.process===P.value)||D.processes.filter(eligible)[0];if(!r)return;P.value=r.process;let pad=Math.max(1,(r.e-r.b)*.1),lo=r.b-pad,hi=r.e+pad,px=t=>125+(t-lo)/(hi-lo)*(C.width-150);X.clearRect(0,0,C.width,C.height);X.font='12px system-ui';[['process HIPTX',70],['strict-owned kernels',205],['eligible SE active CU %',410]].forEach(v=>{X.fillStyle='#a6b5cc';X.fillText(v[0],5,v[1])});X.fillStyle='#55d6be';X.fillRect(px(r.b),45,Math.max(2,px(r.e)-px(r.b)),45);let kernels=D.kernels.filter(k=>k.process===r.process&&k.e>=lo&&k.b<=hi);kernels.forEach((k,i)=>{X.fillStyle='#ffb454';X.fillRect(px(k.b),165+(i%5)*18,Math.max(2,px(k.e)-px(k.b)),14)});let samples=D.samples.filter(s=>s.eligible&&s.t>=r.b&&s.t<=r.e);X.strokeStyle='#66c7ff';X.lineWidth=2;X.beginPath();samples.forEach((s,i)=>{let x=px(s.t),y=490-s.mean*2.7;if(i===0)X.moveTo(x,y);else X.lineTo(x,y)});X.stroke();let a=D.attachments.filter(a=>a.process_range===r.process),families=new Set(a.map(a=>a.matched_kernel_family).filter(Boolean)),hw=D.hardware.filter(h=>h.event_id===r.event&&h.stage===r.stage&&families.has(h.matched_kernel_family)),live=D.process_live.find(x=>x.process_range===r.process),opp=D.opportunities.find(x=>x.process_range===r.process);ST.textContent=`${kernels.length} exact owned kernels; ${samples.length} eligible in-window SE samples; ${hw.length} replay-projected PMC/resource rows`;O.textContent=JSON.stringify({process:r,process_live:live,exact_owned_kernels:kernels,observed_in_window_samples:samples,replay_projected_pmc_resources:hw,inferred_fx_visible_traffic:a,opportunity:opp,device_limits:D.device},null,2)}
[Q,EV,LY,PH,FA].forEach(x=>x.oninput=options);P.onchange=draw;options();
"""


CONCURRENCY_JS = r"""
const D=JSON.parse(document.getElementById('page-payload').textContent),C=document.getElementById('chart'),X=C.getContext('2d'),S=document.getElementById('start'),E=document.getElementById('end'),Q=document.getElementById('search'),OS=document.getElementById('status-filter'),W=document.getElementById('window');
function esc(v){return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;')}function table(id,rows,fields){let q=Q.value.toLowerCase(),filtered=rows.filter(x=>(!q||JSON.stringify(x).toLowerCase().includes(q))&&(!OS.value||x.status===OS.value));let shown=filtered.slice(0,500);document.getElementById(id+'-count').textContent=`showing ${shown.length.toLocaleString()} of ${filtered.length.toLocaleString()} matching rows`;document.getElementById(id).innerHTML='<tr>'+fields.map(f=>'<th>'+esc(f)+'</th>').join('')+'</tr>'+shown.map(x=>'<tr>'+fields.map(f=>'<td>'+esc(x[f])+'</td>').join('')+'</tr>').join('')}
function draw(){let lo=D.begin+(D.end-D.begin)*(+S.value/100),hi=D.begin+(D.end-D.begin)*(+E.value/100);if(hi<=lo)hi=lo+1;W.textContent=((lo-D.begin)/1e6).toFixed(3)+'–'+((hi-D.begin)/1e6).toFixed(3)+' ms on the R07 clock';let px=t=>105+(t-lo)/(hi-lo)*(C.width-130);X.clearRect(0,0,C.width,C.height);X.font='12px system-ui';[['active kernels / overlap',120],['active queues',285],['eligible SE active CU %',465],['material launch gaps',585]].forEach(v=>{X.fillStyle='#a6b5cc';X.fillText(v[0],5,v[1])});let mk=Math.max(1,...D.kernel_concurrency.map(r=>r.n)),mq=Math.max(1,...D.queue_concurrency.map(r=>r.n));D.kernel_concurrency.filter(r=>r.e>lo&&r.b<hi).forEach(r=>{X.fillStyle=r.n>1?'#ff6b7a':'#ffb454';X.fillRect(px(r.b),120-r.n*105/mk,Math.max(1,px(r.e)-px(r.b)),r.n*105/mk)});D.queue_concurrency.filter(r=>r.e>lo&&r.b<hi).forEach(r=>{X.fillStyle='#c099ff';X.fillRect(px(r.b),285-r.n*105/mq,Math.max(1,px(r.e)-px(r.b)),r.n*105/mq)});let samples=D.samples.filter(s=>s.eligible&&s.t>=lo&&s.t<=hi);X.strokeStyle='#66c7ff';X.lineWidth=2;X.beginPath();samples.forEach((s,i)=>{let x=px(s.t),y=530-s.mean*2.3;if(i===0)X.moveTo(x,y);else X.lineTo(x,y)});X.stroke();D.gaps.filter(g=>g.material&&g.e>lo&&g.b<hi).forEach(g=>{X.fillStyle='#ff6b7a';X.fillRect(px(g.b),560,Math.max(2,px(g.e)-px(g.b)),20)});table('opps',D.opportunities,['process_range','event_id','stage','status','dependency_gate','slack_gate','queue_feasibility_gate','resource_coexistence_gate','exposure_gate','utilization_gate','evidence_quality_gate','failed_gates_json','claim_boundary']);table('deps',D.dependencies,['edge_id','source_process_range','target_process_range','dependency_state','ready_time_ns','slack_ns','dependency_gate_pass','slack_gate_pass','evidence_class']);}
[S,E,Q,OS].forEach(x=>x.oninput=draw);draw();
"""


def render_pages(
    metadata: dict[str, Any], payloads: dict[str, dict[str, Any]]
) -> dict[str, str]:
    cards = "".join(
        f"<div class='card'><div class='big'>{payloads['index.html']['row_counts'][key]:,}</div>"
        f"<div>{html.escape(key.replace('_', ' '))}</div></div>"
        for key in REQUIRED_TABLES
    )
    index_body = (
        "<div class='note'>All latency is the same R07 non-replay request. Live SE "
        "samples are observed. R08 PMC/resources are replay-projected attributes "
        "without replay timing. FX traffic is inferred logical tensor IO, never HBM "
        "bytes. Missing values remain unavailable. No speedup is claimed.</div>"
        f"<div class='cards'>{cards}</div>"
        "<div class='panel'><h2>Offline views</h2><ul>"
        "<li><a href='E2E_PROCESS_TIMELINE.html'>Complete request/process/device timeline</a></li>"
        "<li><a href='E2E_PROCESS_TIMELINE_LOSSLESS.html'>Full-resolution lossless drill-down timeline</a></li>"
        "<li><a href='HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html'>High-latency hardware windows</a></li>"
        "<li><a href='CONCURRENCY_UTILIZATION.html'>Concurrency, utilization and opportunity gates</a></li>"
        f"<li><a href='{PERFETTO_TRACE}'>Complete Perfetto-compatible Chrome JSON trace</a> — every normalized interval is retained; structurally checked but not an official parse.</li>"
        f"<li><a href='{FULL_TIMELINE_MANIFEST}'>Full timeline completeness manifest</a></li>"
        "</ul></div>"
    )
    e2e_body = """
<div class='note'>Complete observed request, forward, layer, process HIPTX, HIP runtime, GPU queue and strict-owned kernel intervals on the single R07 realtime axis. Rectangle details expose evidence and timing semantics. No replay duration appears on this axis.</div>
<div class='controls'><label>Search <input class='search' id='search' type='text' placeholder='any field'></label><label>Track <input id='track-filter' type='text'></label><label>Process <input id='process-filter' type='text'></label><label>Event <input id='event-filter' type='text'></label><label>Layer <input id='layer-filter' type='text'></label><label>Phase <input id='phase-filter' type='text'></label><label>Family <input id='family-filter' type='text'></label><button id='reset'>Reset</button></div>
<div class='controls'><label>Zoom start <input id='start' type='range' min='0' max='99' value='0'></label><label>Zoom end <input id='end' type='range' min='1' max='100' value='100'></label><span id='window'></span><span id='visible-count' class='badge'></span></div>
<div data-track-groups='request,forward,layer,process,hip_runtime,gpu_queue,strict_owned_kernel'></div><canvas id='chart' width='1600' height='650'></canvas><pre id='detail' class='panel muted'>Click a rectangle for its exact observed fields.</pre>
"""
    high_body = """
<div class='note'>Every R09 high-latency process window is reproduced with its exact strict-owned kernels and eligible in-window R07 SE samples. Purple records are replay-projected resource and PMC attributes with no replay timestamps; yellow records are inferred traffic from FX-visible logical IO; unavailable values are retained.</div>
<div class='controls'><label>Process <select id='pick'></select></label><label>Search <input class='search' id='search' type='text'></label><label>Event <input id='event-filter' type='text'></label><label>Layer <input id='layer-filter' type='text'></label><label>Phase <input id='phase-filter' type='text'></label><label>Family <input id='family-filter' type='text'></label></div><div id='high-stats' class='badge'></div>
<canvas id='chart' width='1600' height='520'></canvas><pre id='details' class='panel'></pre>
"""
    concurrency_body = """
<div class='note'>Kernel/queue interval sweeps, overlap, launch gaps, dependency readiness, observed utilization and opportunity state all use the R07 clock. R08 resource attributes participate only in the precomputed gate state. A confirmed row passes all seven gates but still does not claim speedup.</div>
<div class='controls'><label>Zoom start <input id='start' type='range' min='0' max='99' value='0'></label><label>Zoom end <input id='end' type='range' min='1' max='100' value='100'></label><label>Search <input class='search' id='search' type='text'></label><label>Opportunity <select id='status-filter'><option value=''>all</option><option>confirmed</option><option>candidate</option><option>unavailable</option></select></label><span id='window'></span></div><canvas id='chart' width='1600' height='620'></canvas>
<div class='panel'><h2>Opportunity gates</h2><p id='opps-count' class='muted'></p><div class='scroll'><table id='opps'></table></div></div><div class='panel'><h2>Dependency / ready / slack</h2><p id='deps-count' class='muted'></p><div class='scroll'><table id='deps'></table></div></div>
"""
    return {
        "index.html": page(
            title="Fresh E2E acceptance", heading="Fresh-run E2E performance acceptance",
            body=index_body, metadata=metadata, payload=payloads["index.html"]
        ),
        "E2E_PROCESS_TIMELINE.html": page(
            title="E2E process timeline", heading="Observed end-to-end request timeline",
            body=(
                top_latency_process_legend(
                    payloads["E2E_PROCESS_TIMELINE.html"]
                )
                + e2e_body
            ), metadata=metadata,
            payload=payloads["E2E_PROCESS_TIMELINE.html"], app_javascript=E2E_JS
        ),
        "E2E_PROCESS_TIMELINE_LOSSLESS.html": page(
            title="Lossless E2E process timeline",
            heading="Full-resolution observed end-to-end request timeline",
            body=(
                top_latency_process_legend(
                    payloads["E2E_PROCESS_TIMELINE_LOSSLESS.html"]
                )
                + LOSSLESS_E2E_BODY
            ), metadata=metadata,
            payload=payloads["E2E_PROCESS_TIMELINE_LOSSLESS.html"],
            app_javascript=LOSSLESS_E2E_JS,
        ),
        "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html": page(
            title="High-latency hardware timeline",
            heading="High-latency process hardware timeline", body=high_body,
            metadata=metadata,
            payload=payloads["HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html"],
            app_javascript=HIGH_JS
        ),
        "CONCURRENCY_UTILIZATION.html": page(
            title="Concurrency and opportunity analysis",
            heading="Concurrency, utilization, launch gaps and opportunity gates",
            body=concurrency_body, metadata=metadata,
            payload=payloads["CONCURRENCY_UTILIZATION.html"],
            app_javascript=CONCURRENCY_JS
        ),
    }


def build_full_perfetto_trace(
    manifest: dict[str, Any], payloads: dict[str, dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    begin = integer(manifest["request_begin_realtime_ns"])
    timeline_payload = payloads["E2E_PROCESS_TIMELINE.html"]
    rows = timeline_payload["rows"]
    top_processes = timeline_payload.get("top_latency_processes", [])
    top_by_process = {
        str(entry["process_range"]): entry for entry in top_processes
    }
    track_ids = {
        "request": 1,
        "forward": 2,
        "layer": 3,
        "process": 4,
        "hip_runtime": 5,
        "gpu_queue": 6,
        "strict_owned_kernel": 7,
    }
    category_counts = Counter(row["g"] for row in rows)
    events: list[dict[str, Any]] = []
    for row in rows:
        b = integer(row["b"])
        e = integer(row["e"])
        event_args = {
            key: value
            for key, value in row.items()
            if key not in {"g", "n", "b", "e"}
        }
        event_args.update({
            "begin_ns": str(b),
            "end_ns": str(e),
            "duration_ns": str(max(0, e - b)),
            "relative_begin_ns": str(b - begin),
            "relative_end_ns": str(e - begin),
        })
        top_owner = top_by_process.get(str(row.get("process", "")))
        if top_owner is not None and row["g"] in {
            "process", "hip_runtime", "gpu_queue", "strict_owned_kernel"
        }:
            prefix = (
                "top_latency_process"
                if row["g"] == "process"
                else "top_latency_owner"
            )
            event_args.update({
                f"{prefix}_rank": top_owner["rank"],
                f"{prefix}_process_range": top_owner["process_range"],
                f"{prefix}_color": top_owner["color"],
                f"{prefix}_observed_duration_ns": str(
                    top_owner["observed_duration_ns"]
                ),
            })
        events.append({
            "name": row["n"],
            "cat": row["g"],
            "ph": "X",
            "ts": round((b - begin) / 1000.0, 3),
            "dur": round(max(0, e - b) / 1000.0, 3),
            "pid": 1,
            "tid": track_ids[row["g"]],
            "args": event_args,
        })
    return {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "metadata": {
            "schema_version": 1,
            "lineage_id": manifest["lineage_id"],
            "source_clock": "R07_non_replay_realtime_only",
            "source_analysis_sha256": metadata["source_analysis_sha256"],
            "classification": "complete_normalized_perfetto_chrome_structural_trace",
            "official_perfetto_parse_performed": False,
            "complete_timeline": True,
            "sampling_performed": False,
            "complete_timeline_location": "E2E_PROCESS_TIMELINE.html",
            "lossless_timeline_location": "E2E_PROCESS_TIMELINE_LOSSLESS.html",
            "timestamp_encoding": "relative_microseconds_with_three_decimal_nanosecond_precision",
            "absolute_timestamp_encoding": "decimal_strings_in_event_args",
            "event_count": len(events),
            "event_count_by_category": dict(sorted(category_counts.items())),
            "top_latency_process_policy": timeline_payload[
                "top_latency_process_policy"
            ],
            "top_latency_processes": top_processes,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fresh E2E acceptance HTML.")
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_path = args.analysis_manifest.expanduser().resolve()
    if not analysis_path.is_file():
        raise RuntimeError("analysis manifest is missing")
    manifest = load_json(analysis_path)
    lineage_id = manifest.get("lineage_id")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("full_request_observed_timeline") is not True
        or manifest.get("analysis_type") != "fresh_run_full_request_e2e"
        or not isinstance(lineage_id, str)
        or not lineage_id
        or set(manifest.get("normalized_tables", {})) != set(REQUIRED_TABLES)
    ):
        raise RuntimeError("analysis manifest is not one complete fresh-run lineage")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output: {args.output_dir}")

    runtime_context = load_runtime_context(analysis_path, manifest)
    runtime_root = runtime_context["runtime_root"]
    tables: dict[str, list[dict[str, str]]] = {}
    for key in REQUIRED_TABLES:
        _, tables[key] = checked_table(manifest, key, runtime_root)
        if not tables[key]:
            raise RuntimeError(f"acceptance-critical table is empty: {key}")
    track_counts = manifest.get("track_type_counts", {})
    if any(integer(track_counts.get(key)) <= 0 for key in (
        "request", "forward", "layer", "hip_runtime"
    )):
        raise RuntimeError("analysis lacks a required observed track type")
    if (
        len(tables["process_timeline"]) != integer(manifest.get("process_count"))
        or len(tables["kernel_timeline"])
        != integer(manifest.get("strict_owned_kernel_count"))
        or len(tables["high_latency_processes"])
        != integer(manifest.get("high_latency_process_count"))
    ):
        raise RuntimeError("R09 manifest summary does not reproduce its tables")

    payloads = build_payloads(manifest, tables, runtime_context)
    top_latency_process_contract = {
        **payloads["E2E_PROCESS_TIMELINE.html"][
            "top_latency_process_policy"
        ],
        "selected": payloads["E2E_PROCESS_TIMELINE.html"][
            "top_latency_processes"
        ],
    }
    metadata = {
        "schema_version": 1,
        "lineage_id": lineage_id,
        "contract_id": manifest.get("contract_id"),
        "source_analysis_sha256": sha256_file(analysis_path),
        "source_table_hashes": {
            key: manifest["normalized_tables"][key]["sha256"] for key in REQUIRED_TABLES
        },
        "source_table_row_counts": {
            key: len(tables[key]) for key in REQUIRED_TABLES
        },
        "request_bounds": {
            "begin_ns": integer(manifest["request_begin_realtime_ns"]),
            "end_ns": integer(manifest["request_end_realtime_ns"]),
            "duration_ns": integer(manifest["request_duration_ns"]),
            "clock": "R07_non_replay_realtime_only",
        },
        "high_latency_process_count": len(tables["high_latency_processes"]),
        "track_groups": list(REQUIRED_TRACK_GROUPS),
        "presentation_backend": runtime_context["presentation_backend"],
        "strict_same_run_validation": runtime_context["strict_same_run_validation"],
        "top_latency_process_contract": top_latency_process_contract,
        "evidence_boundaries": {
            "latency": "observed R07 non-replay same request only",
            "live_utilization": "observed eligible RSMI SE active-CU snapshots",
            "hardware_resources": "R08 replay-projected attributes without replay timestamps",
            "traffic": "inferred FX-visible logical bytes, not HBM/DRAM traffic",
            "missing": "unavailable is retained and never coerced to zero",
            "opportunity": "seven-gate classification without speedup claim",
        },
    }
    pages = render_pages(metadata, payloads)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_records: dict[str, dict[str, Any]] = {}
    forbidden = ("<script src=", "http://", "https://", "fetch(", "XMLHttpRequest", "WebSocket")
    for name in PAGE_NAMES:
        content = pages[name]
        if any(token in content for token in forbidden):
            raise RuntimeError(f"page is not self-contained: {name}")
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        output_records[name] = {
            "path": str(path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    complete_trace = build_full_perfetto_trace(manifest, payloads, metadata)
    trace_path = output_dir / PERFETTO_TRACE
    trace_path.write_text(
        json.dumps(complete_trace, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    companion = {
        "path": str(trace_path), "sha256": sha256_file(trace_path),
        "size_bytes": trace_path.stat().st_size,
        "format": "complete Perfetto-compatible Chrome JSON trace",
        "classification": "complete_structural_trace_unparsed",
        "official_perfetto_parse_performed": False,
        "complete_timeline": True,
        "sampling_performed": False,
        "event_count": len(complete_trace["traceEvents"]),
        "event_count_by_category": complete_trace["metadata"][
            "event_count_by_category"
        ],
    }
    full_timeline_manifest_path = output_dir / FULL_TIMELINE_MANIFEST
    full_timeline_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_class": "formal_r10_full_resolution_timeline",
        "formal_r09_r10_regeneration": True,
        "lineage_id": lineage_id,
        "sampling_performed": False,
        "event_count": len(complete_trace["traceEvents"]),
        "event_count_formula": (
            "request_timeline + process_timeline + 2 * kernel_timeline"
        ),
        "event_count_by_category": complete_trace["metadata"][
            "event_count_by_category"
        ],
        "timestamp_encoding": {
            "rendering": "integer nanoseconds relative to request begin",
            "absolute": "decimal strings per event",
            "perfetto": (
                "relative microseconds with three decimal nanosecond precision"
            ),
        },
        "source_analysis_sha256": metadata["source_analysis_sha256"],
        "source_table_hashes": metadata["source_table_hashes"],
        "source_table_row_counts": metadata["source_table_row_counts"],
        "top_latency_process_contract": top_latency_process_contract,
        "outputs": {
            "lossless_page": {
                "path": "E2E_PROCESS_TIMELINE_LOSSLESS.html",
                "sha256": output_records[
                    "E2E_PROCESS_TIMELINE_LOSSLESS.html"
                ]["sha256"],
            },
            "complete_trace": {
                "path": PERFETTO_TRACE,
                "sha256": companion["sha256"],
            },
        },
    }
    full_timeline_manifest_path.write_text(
        json.dumps(full_timeline_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    full_manifest_record = {
        "path": str(full_timeline_manifest_path),
        "sha256": sha256_file(full_timeline_manifest_path),
        "size_bytes": full_timeline_manifest_path.stat().st_size,
        "format": "full-resolution timeline completeness manifest",
        "event_count": len(complete_trace["traceEvents"]),
        "sampling_performed": False,
    }

    generator_path = Path(__file__).resolve()
    acceptance = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_id": lineage_id,
        "self_contained_offline": True,
        "generator": {"path": str(generator_path), "sha256": sha256_file(generator_path)},
        "source_analysis": {"path": str(analysis_path), "sha256": sha256_file(analysis_path)},
        "source_table_hashes": metadata["source_table_hashes"],
        "row_counts": metadata["source_table_row_counts"],
        "top_latency_process_contract": top_latency_process_contract,
        "outputs": output_records,
        "companions": {
            PERFETTO_TRACE: companion,
            FULL_TIMELINE_MANIFEST: full_manifest_record,
        },
        "view_coverage": {
            "track_groups": list(REQUIRED_TRACK_GROUPS),
            "filters_search_zoom": True,
            "filter_dimensions": ["process", "event", "layer", "phase", "family"],
            "source_table_hashes_verified": True,
            "evidence_legends_complete": True,
            "complete_request_timeline_embedded": True,
            "lossless_relative_nanosecond_timeline_embedded": True,
            "complete_perfetto_trace_without_sampling": True,
            "high_latency_selection_reproduced": True,
            "top_latency_process_colors": True,
            "top_latency_owned_interval_outlines": True,
            "zoom_reveals_process_names_inside_rectangles": True,
        },
        "presentation_backend": runtime_context["presentation_backend"],
        "same_run_inputs": {
            "strict_validation": runtime_context["strict_same_run_validation"],
            "handoff_hashes": runtime_context["handoff_hashes"],
            "r09_input_hashes_verified": runtime_context["r09_input_hashes_verified"],
            "evidence_references": runtime_context["evidence_references"],
        },
        "request_bounds": metadata["request_bounds"],
        "high_latency_process_count": len(tables["high_latency_processes"]),
        "evidence_boundaries": metadata["evidence_boundaries"],
        "determinism_contract": {
            "timestamps_embedded": False,
            "source_iteration_order": "R09 CSV row order",
            "output_encoding": "UTF-8",
        },
    }
    acceptance_path = output_dir / "offline_acceptance_manifest.json"
    acceptance_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS", "manifest": str(acceptance_path),
        "sha256": sha256_file(acceptance_path),
        "page_bytes": sum(record["size_bytes"] for record in output_records.values()),
        "strict_same_run_validation": runtime_context["strict_same_run_validation"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
