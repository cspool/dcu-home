#!/usr/bin/env python3
"""Independently audit the fresh-run R10 offline acceptance bundle."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


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
TIMELINE_RECTANGLE_LABEL_GROUPS = (
    "request",
    "forward",
    "layer",
    "process",
    "hip_runtime",
    "gpu_queue",
    "strict_owned_kernel",
)
PAGE_NAMES = (
    "index.html",
    "E2E_PROCESS_TIMELINE.html",
    "E2E_PROCESS_TIMELINE_LOSSLESS.html",
    "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html",
    "CONCURRENCY_UTILIZATION.html",
)
MANIFEST_NAME = "offline_acceptance_manifest.json"
PERFETTO_TRACE = "E2E_PROCESS_TIMELINE.full.perfetto.json"
FULL_TIMELINE_MANIFEST = "full_timeline_manifest.json"
DETERMINISTIC_FILES = PAGE_NAMES + (
    PERFETTO_TRACE,
    FULL_TIMELINE_MANIFEST,
    MANIFEST_NAME,
)
TOP_LATENCY_PROCESS_PALETTE = (
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
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


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"JSON must be a nonempty object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV lacks a header: {path}")
        return list(reader)


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def process_phase(row: dict[str, str]) -> str:
    if row.get("phase"):
        return row["phase"]
    parent = row.get("parent_layer_range", "").lower()
    if "prefill" in parent:
        return "prefill"
    if "decode" in parent:
        return "decode"
    return ""


def expected_top_latency_process_contract(
    process_rows: list[dict[str, str]], begin: int, end: int
) -> dict[str, Any]:
    request_span = end - begin
    if request_span <= 0:
        raise RuntimeError("independent R10 check failed: invalid request span")
    ranked = [
        (
            integer(row["hiptx_end_ns"]) - integer(row["hiptx_begin_ns"]),
            integer(row["hiptx_begin_ns"]),
            row["process_range"],
        )
        for row in process_rows
    ]
    if (
        any(duration < 0 for duration, _, _ in ranked)
        or len({name for _, _, name in ranked}) != len(ranked)
    ):
        raise RuntimeError(
            "independent R10 check failed: process ranking input is invalid"
        )
    total = sum(duration for duration, _, _ in ranked)
    if total <= 0:
        raise RuntimeError(
            "independent R10 check failed: process duration total is invalid"
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [
        {
            "rank": rank,
            "process_range": process_range,
            "hiptx_begin_ns": process_begin,
            "observed_duration_ns": duration,
            "observed_process_duration_share": duration / total,
            "observed_request_span_ratio": duration / request_span,
            "request_span_ratio_caveat": REQUEST_SPAN_RATIO_CAVEAT,
            "color": TOP_LATENCY_PROCESS_PALETTE[rank - 1],
        }
        for rank, (duration, process_begin, process_range) in enumerate(
            ranked[: len(TOP_LATENCY_PROCESS_PALETTE)], start=1
        )
    ]
    return {
        "schema_version": 1,
        "ranking_source": "complete_immutable_R09_process_timeline",
        "ranking_duration": "hiptx_end_ns - hiptx_begin_ns",
        "ranking_order": [
            "observed_duration_ns_descending",
            "hiptx_begin_ns_ascending",
            "process_range_ascending",
        ],
        "configured_count": len(TOP_LATENCY_PROCESS_PALETTE),
        "selected_count": len(selected),
        "palette": list(TOP_LATENCY_PROCESS_PALETTE),
        "observed_process_duration_total_ns": total,
        "observed_request_span_ns": request_span,
        "request_span_ratio_caveat": REQUEST_SPAN_RATIO_CAVEAT,
        "selected": selected,
    }


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


class AcceptanceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str, str]] = []
        self.tags: Counter[str] = Counter()
        self.scripts: dict[str, list[str]] = defaultdict(list)
        self.current_script: str | None = None
        self.evidence_classes: set[str] = set()
        self.title_attribute_count = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.tags[tag] += 1
        values = {key.lower(): value or "" for key, value in attrs}
        for key in ("href", "src", "action", "poster"):
            if key in values:
                self.links.append((tag, key, values[key]))
        if "data-evidence-class" in values:
            self.evidence_classes.add(values["data-evidence-class"])
        if "title" in values:
            self.title_attribute_count += 1
        if tag == "script" and values.get("id"):
            self.current_script = values["id"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.current_script = None

    def handle_data(self, data: str) -> None:
        if self.current_script is not None:
            self.scripts[self.current_script].append(data)


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise RuntimeError(f"independent R10 check failed: {message}")


def parse_page(path: Path, checker: Checker) -> tuple[str, dict[str, Any], dict[str, Any], AcceptanceHTMLParser]:
    text = path.read_text(encoding="utf-8")
    parser = AcceptanceHTMLParser()
    parser.feed(text)
    parser.close()
    checker.require(parser.tags["html"] == 1, f"{path.name} has one html element")
    checker.require(parser.tags["head"] == 1, f"{path.name} has one head element")
    checker.require(parser.tags["body"] == 1, f"{path.name} has one body element")
    checker.require("acceptance-metadata" in parser.scripts, f"{path.name} embeds metadata")
    checker.require("page-payload" in parser.scripts, f"{path.name} embeds page payload")
    metadata = json.loads("".join(parser.scripts["acceptance-metadata"]))
    payload = json.loads("".join(parser.scripts["page-payload"]))
    checker.require(isinstance(metadata, dict), f"{path.name} metadata parses")
    checker.require(isinstance(payload, dict), f"{path.name} payload parses")
    return text, metadata, payload, parser


def checked_reference(record: Any, runtime_root: Path, checker: Checker, name: str) -> Path:
    checker.require(isinstance(record, dict), f"{name} reference is an object")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    checker.require(is_under(path, runtime_root), f"{name} remains inside current run")
    checker.require(path.is_file(), f"{name} exists")
    checker.require(sha256_file(path) == record.get("sha256"), f"{name} hash matches")
    return path


def request_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "g": row["track_type"], "n": row.get("label", ""),
        "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
        "event": row.get("event_id", ""), "forward": row.get("forward_id", ""),
        "layer": row.get("layer", ""), "phase": row.get("phase", ""),
        "process": row.get("process_owner", ""),
        "runtime": row.get("runtime_index", ""), "queue": row.get("queue_id", ""),
        "family": "", "evidence": row.get("evidence_class", "observed"),
        "timing": row.get("timing_source", ""),
    }


def process_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "g": "process", "n": row["process_range"],
        "b": integer(row["hiptx_begin_ns"]), "e": integer(row["hiptx_end_ns"]),
        "event": row.get("event_id", ""), "forward": row.get("forward_id", ""),
        "layer": row.get("layer", ""), "phase": process_phase(row),
        "process": row["process_range"], "runtime": "",
        "queue": row.get("strict_owned_queue_ids", ""), "family": "",
        "stage": row.get("stage", ""),
        "evidence": row.get("evidence_class", "observed"),
        "timing": row.get("timing_source", ""),
    }


def kernel_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
        "process": row.get("process_owner", ""), "queue": row.get("queue_id", ""),
        "n": row.get("kernel_name", ""), "family": row.get("kernel_family", ""),
        "runtime": row.get("runtime_index", ""), "kernel_id": row.get("kernel_id", ""),
        "evidence": row.get("evidence_class", "observed"),
        "timing": row.get("timing_source", ""),
    }


def high_expected(
    row: dict[str, str], families: set[str]
) -> dict[str, Any]:
    return {
        "process": row["process_range"],
        "b": integer(row["hiptx_begin_ns"]), "e": integer(row["hiptx_end_ns"]),
        "ms": number(row["hiptx_cpu_ms"]),
        "samples": integer(row.get("live_sample_count")),
        "mean": row.get("mean_se_active_cu_pct", "unavailable"),
        "max": row.get("max_se_active_cu_pct", "unavailable"),
        "live_status": row.get("live_utilization_status", "unavailable"),
        "event": row.get("event_id", ""), "forward": row.get("forward_id", ""),
        "layer": row.get("layer", ""), "phase": process_phase(row),
        "stage": row.get("stage", ""), "families": sorted(families),
        "owned_kernel_count": integer(row.get("strict_owned_kernel_count")),
        "owned_kernel_busy_union_ms": number(row.get("strict_owned_kernel_busy_union_ms")),
        "traffic_bytes": row.get("fx_visible_total_io_bytes", "unavailable"),
        "traffic_completeness": row.get("traffic_completeness", ""),
        "evidence": row.get("evidence_class", "observed"),
    }


def sample_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "t": integer(row["realtime_midpoint_ns"]),
        "mean": number(row.get("mean_se_active_cu_pct")),
        "max": number(row.get("max_se_active_cu_pct")),
        "uncertainty": integer(row.get("alignment_uncertainty_ns")),
        "eligible": truth(row.get("eligible_for_process_attribution")),
        "status": row.get("alignment_status", ""),
        "evidence": row.get("evidence_class", "observed"),
    }


def kc_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
        "n": integer(row["active_kernel_count"]),
        "evidence": row.get("evidence_class", "observed"),
        "timing": row.get("timing_source", ""),
    }


def qc_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
        "b": integer(row["begin_ns"]), "e": integer(row["end_ns"]),
        "n": integer(row["active_queue_count"]),
        "kernels": integer(row.get("active_kernel_count")),
        "evidence": row.get("evidence_class", "observed"),
        "timing": row.get("timing_source", ""),
    }


def gap_expected(row: dict[str, str]) -> dict[str, Any]:
    return {
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


def dependency_expected(row: dict[str, str]) -> dict[str, Any]:
    fields = (
        "edge_id", "event_id", "source_process_range", "target_process_range",
        "dependency_state", "ready_time_ns", "slack_ns", "dependency_gate_pass",
        "slack_gate_pass", "evidence_class", "verified", "timing_source",
    )
    return {field: row.get(field, "") for field in fields}


def require_sequence(
    checker: Checker, actual: list[Any], expected: list[Any], name: str
) -> None:
    checker.require(len(actual) == len(expected), f"{name} row count reproduces source")
    for index, (observed, wanted) in enumerate(zip(actual, expected)):
        if observed != wanted:
            raise RuntimeError(
                f"independent R10 check failed: {name} row {index} differs from source"
            )
    checker.count += 1


def unique_inode_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fresh-run R10 acceptance.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-dir", type=Path, required=True)
    parser.add_argument("--determinism-reference-dir", type=Path, required=True)
    parser.add_argument("--source-lineage-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--pre-generator-sha256", required=True)
    parser.add_argument("--maximum-trace-bundle-bytes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = Checker()
    project_root = args.project_root.resolve()
    source_root = args.source_root.resolve()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    analysis_path = args.analysis_manifest.resolve()
    acceptance_dir = args.acceptance_dir.resolve()
    reference_dir = args.determinism_reference_dir.resolve()
    source_lineage_output = args.source_lineage_output.resolve()
    audit_output = args.audit_output.resolve()

    checker.require(is_under(source_root, project_root), "source root remains in project")
    checker.require(is_under(runtime_root, project_root), "runtime root remains in project")
    checker.require(artifact_root == runtime_root / "artifacts" / "R10", "artifact root is scheduler R10")
    checker.require(acceptance_dir == artifact_root / "acceptance", "acceptance directory is canonical")
    checker.require(is_under(reference_dir, artifact_root), "determinism reference remains in R10")
    checker.require(source_lineage_output == artifact_root / "R10_SOURCE_LINEAGE.json", "source lineage path is canonical")
    checker.require(audit_output == artifact_root / "R10_COMPLETION_AUDIT.json", "audit path is canonical")
    checker.require(not source_lineage_output.exists(), "source lineage does not overwrite a file")
    checker.require(not audit_output.exists(), "audit does not overwrite a file")
    checker.require(re.fullmatch(r"[0-9a-f]{64}", args.pre_generator_sha256) is not None, "pre-generator hash is valid")

    handoffs = {
        goal: load_object(runtime_root / "handoffs" / f"{goal}.json")
        for goal in ("R06", "R07", "R08", "R09")
    }
    handoff_hashes = {
        goal: sha256_file(runtime_root / "handoffs" / f"{goal}.json")
        for goal in handoffs
    }
    lineage_id = handoffs["R09"]["fresh_e2e_evidence"]["lineage_id"]
    for goal, handoff in handoffs.items():
        checker.require(handoff.get("runtime_goal") == goal, f"{goal} goal identity")
        checker.require(handoff.get("status") == "complete", f"{goal} status complete")
        checker.require(handoff.get("execution_status") == "complete", f"{goal} execution complete")
        checker.require(handoff.get("fresh_e2e_evidence", {}).get("lineage_id") == lineage_id, f"{goal} lineage matches")
    r09_binding = handoffs["R09"]["same_run_binding"]
    for goal in ("R06", "R07", "R08"):
        checker.require(r09_binding[f"{goal}_handoff_sha256"] == handoff_hashes[goal], f"R09 binds {goal} hash")

    manifest = load_object(analysis_path)
    checker.require(is_under(analysis_path, runtime_root), "analysis remains inside current run")
    checker.require(sha256_file(analysis_path) == r09_binding["analysis_manifest_sha256"], "analysis hash matches R09")
    checker.require(manifest.get("status") == "PASS", "R09 analysis passed")
    checker.require(manifest.get("analysis_type") == "fresh_run_full_request_e2e", "analysis type is fresh E2E")
    checker.require(manifest.get("lineage_id") == lineage_id, "analysis lineage matches")
    checker.require(manifest.get("full_request_observed_timeline") is True, "full observed timeline is declared")
    checker.require(set(manifest.get("normalized_tables", {})) == set(REQUIRED_TABLES), "all twelve normalized tables exist")

    tables: dict[str, list[dict[str, str]]] = {}
    for name in REQUIRED_TABLES:
        record = manifest["normalized_tables"][name]
        path = Path(record["path"]).resolve()
        checker.require(is_under(path, runtime_root), f"{name} remains inside current run")
        checker.require(path.is_file(), f"{name} exists")
        checker.require(sha256_file(path) == record["sha256"], f"{name} hash matches R09")
        tables[name] = read_csv(path)
        checker.require(len(tables[name]) == record["row_count"], f"{name} row count matches R09")
        checker.require(len(tables[name]) > 0, f"{name} is nonempty")
    for raw_path, digest in manifest.get("inputs", {}).items():
        path = Path(raw_path).resolve()
        checker.require(is_under(path, runtime_root), "R09 input remains in current run")
        checker.require(path.is_file() and sha256_file(path) == digest, "R09 input hash remains immutable")

    r06 = handoffs["R06"]
    capability = r06["visualization_capability"]
    checker.require(capability["official_perfetto_python_status"] == "unavailable", "R06 official Python parser unavailable")
    checker.require(capability["official_perfetto_cli_status"] == "unavailable", "R06 official CLI unavailable")
    checker.require(capability["selected_backend"] == "custom_plotly_timeline_fallback", "R06 selected labeled custom fallback")
    checker.require(capability["network_download_performed"] is False, "R06 performed no network download")
    checked_reference(r06["primary_outputs"]["open_source_trace_attempts"], runtime_root, checker, "R06 open-source attempts")
    checked_reference(r06["primary_outputs"]["tool_capability_probe"], runtime_root, checker, "R06 capability probe")

    r08_outputs = handoffs["R08"]["primary_outputs"]
    device_path = checked_reference(r08_outputs["device_capabilities"], runtime_root, checker, "R08 device capabilities")
    hardware_path = checked_reference(r08_outputs["hardware_metrics_by_kernel_family"], runtime_root, checker, "R08 hardware metrics")
    traffic_path = checked_reference(r08_outputs["traffic_resource_model"], runtime_root, checker, "R08 traffic/resource model")
    device = load_object(device_path)
    hardware_rows = read_csv(hardware_path)
    traffic = load_object(traffic_path)
    checker.require(traffic.get("lineage_id") == lineage_id, "R08 traffic model lineage matches")
    checker.require(traffic["traffic_boundary"]["hbm_or_dram_traffic_claimed"] is False, "R08 does not claim HBM traffic")
    checker.require(traffic["resource_boundary"]["achieved_occupancy_claimed"] is False, "R08 does not claim achieved occupancy")
    for row in hardware_rows:
        checker.require(not truth(row.get("pmc_replay_timing_used_as_latency")), "PMC replay timing is not latency")
        checker.require(row.get("latency_axis") == "R07_non_replay_same_request_only", "hardware row retains R07 latency axis")
        checker.require(row.get("cross_capture_timeline_policy") == "separate_clock_axes_no_merge", "hardware row prohibits cross-clock merge")

    acceptance_manifest_path = acceptance_dir / MANIFEST_NAME
    acceptance = load_object(acceptance_manifest_path)
    checker.require(acceptance.get("status") == "PASS", "generator manifest status passed")
    checker.require(acceptance.get("lineage_id") == lineage_id, "acceptance lineage matches")
    checker.require(acceptance.get("self_contained_offline") is True, "acceptance declares offline self-containment")
    checker.require(acceptance.get("source_analysis", {}).get("path") == str(analysis_path), "acceptance analysis path matches")
    checker.require(acceptance.get("source_analysis", {}).get("sha256") == sha256_file(analysis_path), "acceptance analysis hash matches")
    expected_hashes = {key: manifest["normalized_tables"][key]["sha256"] for key in REQUIRED_TABLES}
    expected_counts = {key: len(tables[key]) for key in REQUIRED_TABLES}
    expected_timeline_rectangle_count = (
        expected_counts["request_timeline"]
        + expected_counts["process_timeline"]
        + 2 * expected_counts["kernel_timeline"]
    )
    begin = integer(manifest["request_begin_realtime_ns"])
    end = integer(manifest["request_end_realtime_ns"])
    expected_top_contract = expected_top_latency_process_contract(
        tables["process_timeline"], begin, end
    )
    checker.require(acceptance.get("source_table_hashes") == expected_hashes, "acceptance records all source hashes")
    checker.require(acceptance.get("row_counts") == expected_counts, "acceptance records all source row counts")
    checker.require(
        acceptance.get("top_latency_process_contract") == expected_top_contract,
        "acceptance records the exact top-latency process contract",
    )
    checker.require(acceptance.get("view_coverage", {}).get("track_groups") == list(REQUIRED_TRACK_GROUPS), "acceptance track groups complete")
    checker.require(acceptance.get("view_coverage", {}).get("filters_search_zoom") is True, "acceptance has filter/search/zoom")
    checker.require(acceptance.get("view_coverage", {}).get("filter_dimensions") == ["process", "event", "layer", "phase", "family"], "five filter dimensions declared")
    checker.require(acceptance.get("view_coverage", {}).get("evidence_legends_complete") is True, "evidence legends declared complete")
    checker.require(acceptance.get("view_coverage", {}).get("top_latency_process_colors") is True, "top-latency process colors declared")
    checker.require(acceptance.get("view_coverage", {}).get("top_latency_owned_interval_outlines") is True, "top-latency ownership outlines declared")
    checker.require(acceptance.get("view_coverage", {}).get("zoom_reveals_process_names_inside_rectangles") is True, "zoom-dependent process labels declared")
    checker.require(acceptance.get("view_coverage", {}).get("rectangle_label_groups") == list(TIMELINE_RECTANGLE_LABEL_GROUPS), "all timeline rectangle label groups declared")
    checker.require(acceptance.get("view_coverage", {}).get("zoom_reveals_all_timeline_labels_inside_rectangles") is True, "zoom-dependent labels for every timeline group declared")
    checker.require(acceptance.get("view_coverage", {}).get("labeled_timeline_rectangle_count") == expected_timeline_rectangle_count, "all timeline rectangles are counted as labeled")
    checker.require(acceptance.get("view_coverage", {}).get("unlabeled_timeline_rectangle_count") == 0, "no timeline rectangle is unlabeled")
    checker.require(acceptance.get("same_run_inputs", {}).get("strict_validation") is True, "production same-run validation was strict")
    checker.require(acceptance.get("same_run_inputs", {}).get("handoff_hashes") == handoff_hashes, "acceptance binds R06-R09 hashes")

    outputs = acceptance.get("outputs")
    checker.require(isinstance(outputs, dict) and set(outputs) == set(PAGE_NAMES), "exact required page outputs")
    expected_evidence_classes = {
        "observed", "observed live utilization", "replay-projected PMC/resources",
        "inferred FX-visible traffic", "unavailable",
    }
    parsed_pages: dict[str, tuple[str, dict[str, Any], dict[str, Any], AcceptanceHTMLParser]] = {}
    forbidden_tokens = (
        "<script src=", "http://", "https://", "file://", "//cdn.",
        "fetch(", "XMLHttpRequest", "WebSocket", "@import url",
    )
    required_control_ids = {
        "E2E_PROCESS_TIMELINE.html": (
            "search", "track-filter", "process-filter", "event-filter", "layer-filter",
            "phase-filter", "family-filter", "start", "end",
        ),
        "E2E_PROCESS_TIMELINE_LOSSLESS.html": (
            "search", "track-filter", "process-filter", "event-filter", "layer-filter",
            "phase-filter", "family-filter", "start-ns", "end-ns", "fit-filter",
            "zoom-in", "zoom-out", "back", "reset",
        ),
        "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html": (
            "pick", "search", "event-filter", "layer-filter", "phase-filter", "family-filter",
        ),
        "CONCURRENCY_UTILIZATION.html": ("search", "status-filter", "start", "end"),
    }
    required_static_markers = {
        "index.html": ("Fresh-run E2E performance acceptance",),
        "E2E_PROCESS_TIMELINE.html": (
            "data-track-groups=", "hip_runtime", "gpu_queue", "strict_owned_kernel",
            "data-top-latency-process-count=", "g==='process'&&top?top.color",
            "ownedGroups.has(g)", "data-rectangle-label-groups=",
            "labeledGroups=new Set(D.groups)", "function rectangleLabel",
            "function fitLabel", "X.fillText(label",
        ),
        "E2E_PROCESS_TIMELINE_LOSSLESS.html": (
            "Full-resolution observed end-to-end request timeline",
            "data-sampling-performed='false'", "relative to request begin",
            "data-top-latency-process-count=", "g==='process'&&top?top.color",
            "ownedGroups.has(g)", "data-rectangle-label-groups=",
            "labeledGroups=new Set(D.groups)", "function rectangleLabel",
            "function fitLabel", "X.fillText(label",
        ),
        "HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html": (
            "replay-projected resource", "inferred traffic", "unavailable",
        ),
        "CONCURRENCY_UTILIZATION.html": (
            "Opportunity gates", "Dependency / ready / slack", "material launch gaps",
        ),
    }
    for name in PAGE_NAMES:
        path = acceptance_dir / name
        checker.require(path.is_file(), f"{name} exists")
        checker.require(outputs[name].get("path") == str(path), f"{name} manifest path matches")
        checker.require(outputs[name].get("sha256") == sha256_file(path), f"{name} manifest hash matches")
        text, metadata, payload, parser = parse_page(path, checker)
        parsed_pages[name] = (text, metadata, payload, parser)
        checker.require(not any(token in text for token in forbidden_tokens), f"{name} has no external/runtime loader")
        checker.require(parser.evidence_classes == expected_evidence_classes, f"{name} legend classes complete")
        checker.require(parser.title_attribute_count >= 5, f"{name} evidence tooltips present")
        checker.require(metadata.get("lineage_id") == lineage_id, f"{name} embedded lineage matches")
        checker.require(metadata.get("source_analysis_sha256") == sha256_file(analysis_path), f"{name} embedded analysis hash matches")
        checker.require(metadata.get("source_table_hashes") == expected_hashes, f"{name} embedded table hashes match")
        checker.require(metadata.get("source_table_row_counts") == expected_counts, f"{name} embedded row counts match")
        checker.require(metadata.get("track_groups") == list(REQUIRED_TRACK_GROUPS), f"{name} embedded track groups match")
        checker.require(metadata.get("rectangle_label_groups") == list(TIMELINE_RECTANGLE_LABEL_GROUPS), f"{name} embeds every timeline rectangle label group")
        checker.require(metadata.get("strict_same_run_validation") is True, f"{name} records strict same-run validation")
        checker.require(metadata.get("top_latency_process_contract") == expected_top_contract, f"{name} embeds top-latency process contract")
        checker.require("SELF-CONTAINED CUSTOM CANVAS TIMELINE FALLBACK" in text, f"{name} visibly labels custom viewer")
        checker.require("not an official Perfetto parse" in text, f"{name} distinguishes structural trace")
        for marker in required_static_markers[name]:
            checker.require(marker in text, f"{name} contains scheduler marker: {marker}")
        for tag, attribute, link in parser.links:
            parsed = urlsplit(link)
            checker.require(not parsed.scheme and not parsed.netloc and not link.startswith("/"), f"{name} {attribute} link is relative")
            target_text = parsed.path
            if target_text:
                target = (acceptance_dir / target_text).resolve()
                checker.require(is_under(target, acceptance_dir), f"{name} link remains inside acceptance")
                checker.require(target.is_file(), f"{name} relative link resolves: {target_text}")
        for control_id in required_control_ids.get(name, ()):
            checker.require(f"id='{control_id}'" in text, f"{name} contains control {control_id}")

    for name, (_, metadata, _, _) in parsed_pages.items():
        checker.require(metadata["request_bounds"] == {
            "begin_ns": begin, "end_ns": end,
            "duration_ns": integer(manifest["request_duration_ns"]),
            "clock": "R07_non_replay_realtime_only",
        }, f"{name} reproduces request bounds")

    index_payload = parsed_pages["index.html"][2]
    checker.require(index_payload["row_counts"] == expected_counts, "index payload contains all twelve row counts")
    checker.require(index_payload["request_begin_ns"] == begin and index_payload["request_end_ns"] == end, "index request bounds reproduce R09")
    checker.require(index_payload["process_live_status_counts"] == dict(sorted(Counter(r.get("status", "unknown") for r in tables["process_live_utilization"]).items())), "index process-live statuses reproduce R09")
    checker.require(index_payload["opportunity_status_counts"] == dict(sorted(Counter(r.get("status", "unknown") for r in tables["opportunity_candidates"]).items())), "index opportunity statuses reproduce R09")

    e2e = parsed_pages["E2E_PROCESS_TIMELINE.html"][2]
    checker.require(e2e["begin"] == begin and e2e["end"] == end, "E2E payload request bounds match")
    checker.require(e2e["groups"] == list(TIMELINE_RECTANGLE_LABEL_GROUPS), "E2E observed groups complete")
    e2e_top_contract = {
        **e2e["top_latency_process_policy"],
        "selected": e2e["top_latency_processes"],
    }
    checker.require(
        e2e_top_contract == expected_top_contract,
        "E2E payload reproduces exact top-latency ranking and palette",
    )
    checker.require(
        len({row["color"] for row in e2e["top_latency_processes"]})
        == len(e2e["top_latency_processes"]),
        "E2E top-latency colors are distinct",
    )
    rows = e2e["rows"]
    checker.require(
        all(
            row.get("g") in TIMELINE_RECTANGLE_LABEL_GROUPS
            and bool(
                str(
                    (row.get("process") or row.get("n") or "")
                    if row.get("g") == "process"
                    else (row.get("n") or "")
                )
            )
            for row in rows
        ),
        "every E2E rectangle has a semantic label",
    )
    request_count = len(tables["request_timeline"])
    process_count = len(tables["process_timeline"])
    kernel_count = len(tables["kernel_timeline"])
    checker.require(len(rows) == request_count + process_count + 2 * kernel_count, "E2E embeds complete normalized intervals")
    require_sequence(checker, rows[:request_count], [request_expected(row) for row in tables["request_timeline"]], "E2E request/forward/layer/runtime")
    process_start = request_count
    kernel_start = process_start + process_count
    queue_start = kernel_start + kernel_count
    require_sequence(checker, rows[process_start:kernel_start], [process_expected(row) for row in tables["process_timeline"]], "E2E process HIPTX")
    expected_kernels = [kernel_expected(row) for row in tables["kernel_timeline"]]
    require_sequence(checker, rows[kernel_start:queue_start], [{"g": "strict_owned_kernel", "event": "", "forward": "", "layer": "", "phase": "", **row} for row in expected_kernels], "E2E strict-owned kernels")
    require_sequence(checker, rows[queue_start:], [{**row, "g": "gpu_queue", "n": "queue " + row["queue"], "event": "", "forward": "", "layer": "", "phase": ""} for row in expected_kernels], "E2E GPU queue projection")
    checker.require(Counter(row["g"] for row in rows)["request"] == 1, "E2E contains one request")
    checker.require(Counter(row["g"] for row in rows)["forward"] == manifest["track_type_counts"]["forward"], "E2E forward count matches")
    checker.require(Counter(row["g"] for row in rows)["layer"] == manifest["track_type_counts"]["layer"], "E2E layer count matches")
    checker.require(Counter(row["g"] for row in rows)["hip_runtime"] == manifest["track_type_counts"]["hip_runtime"], "E2E HIP runtime count matches")

    lossless = parsed_pages["E2E_PROCESS_TIMELINE_LOSSLESS.html"][2]
    checker.require(lossless["origin_ns"] == str(begin), "lossless payload preserves absolute request origin as a string")
    checker.require(lossless["begin"] == 0 and lossless["end"] == end - begin, "lossless payload uses relative integer nanoseconds")
    checker.require(lossless["groups"] == e2e["groups"], "lossless payload track groups match complete E2E")
    checker.require(
        {
            **lossless["top_latency_process_policy"],
            "selected": lossless["top_latency_processes"],
        }
        == expected_top_contract,
        "lossless payload reproduces exact top-latency contract",
    )
    lossless_rows = lossless["rows"]
    checker.require(len(lossless_rows) == len(rows), "lossless payload retains every E2E interval")
    for index, (row, relative) in enumerate(zip(rows, lossless_rows)):
        expected = {
            **{key: value for key, value in row.items() if key not in {"b", "e"}},
            "b": row["b"] - begin,
            "e": row["e"] - begin,
            "b_abs": str(row["b"]),
            "e_abs": str(row["e"]),
        }
        if relative != expected:
            raise RuntimeError(
                "independent R10 check failed: lossless E2E row "
                f"{index} differs from the exact relative-ns projection"
            )
    checker.count += 1

    high_payload = parsed_pages["HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html"][2]
    kernel_families: dict[str, set[str]] = defaultdict(set)
    for row in tables["kernel_timeline"]:
        if row.get("kernel_family"):
            kernel_families[row.get("process_owner", "")].add(row["kernel_family"])
    expected_high = [high_expected(row, kernel_families[row["process_range"]]) for row in tables["high_latency_processes"]]
    require_sequence(checker, high_payload["processes"], expected_high, "high-latency selection")
    require_sequence(checker, high_payload["kernels"], expected_kernels, "high-page exact kernels")
    expected_samples = [sample_expected(row) for row in tables["live_utilization_aligned"]]
    require_sequence(checker, high_payload["samples"], expected_samples, "high-page live samples")
    high_names = {row["process"] for row in expected_high}
    require_sequence(checker, high_payload["process_live"], [row for row in tables["process_live_utilization"] if row.get("process_range") in high_names], "high-page process live records")
    require_sequence(checker, high_payload["attachments"], tables["traffic_resource_attachment"], "high-page traffic/resource attachments")
    require_sequence(checker, high_payload["opportunities"], [row for row in tables["opportunity_candidates"] if row.get("process_range") in high_names], "high-page opportunity states")
    require_sequence(checker, high_payload["hardware"], [selected_hardware_row(row) for row in hardware_rows], "high-page replay-projected PMC/resources")
    expected_device = {key: device.get(key) for key in (
        "physical_device_id", "architecture", "cu_count", "wave_size", "wave_limit",
        "thread_limit", "vgpr_resource", "shared_memory_bytes", "resource_semantics",
        "unavailable_quantities",
    ) if key in device}
    checker.require(high_payload["device"] == expected_device, "high-page device limits reproduce R08")
    eligible_times = sorted(row["t"] for row in expected_samples if row["eligible"])
    owned_counts = Counter(row["process"] for row in expected_kernels)
    for row in expected_high:
        count = bisect.bisect_right(eligible_times, row["e"]) - bisect.bisect_left(eligible_times, row["b"])
        checker.require(count == row["samples"], f"high process {row['process']} in-window sample count")
        checker.require(owned_counts[row["process"]] == row["owned_kernel_count"], f"high process {row['process']} exact owned-kernel count")
    checker.require(len(expected_high) == manifest["high_latency_process_count"], "high selection count matches manifest")
    checker.require(all(row["samples"] >= manifest["minimum_live_samples_per_process"] for row in expected_high), "every high process has sufficient live samples")
    checker.require(all(row.get("pmc_replay_timing_used_as_latency") == "False" for row in high_payload["hardware"]), "high-page PMC attributes exclude replay latency")
    checker.require(all(row.get("cross_capture_timeline_policy") == "separate_clock_axes_no_merge" for row in high_payload["hardware"]), "high-page hardware retains separate clocks")
    checker.require(all("begin_ns" not in row and "end_ns" not in row for row in high_payload["hardware"]), "replay hardware carries no overlay timestamps")

    concurrency = parsed_pages["CONCURRENCY_UTILIZATION.html"][2]
    checker.require(concurrency["begin"] == begin and concurrency["end"] == end, "concurrency request bounds match")
    require_sequence(checker, concurrency["kernel_concurrency"], [kc_expected(row) for row in tables["kernel_concurrency"]], "kernel concurrency")
    require_sequence(checker, concurrency["queue_concurrency"], [qc_expected(row) for row in tables["queue_concurrency"]], "queue concurrency")
    require_sequence(checker, concurrency["samples"], expected_samples, "concurrency live utilization")
    require_sequence(checker, concurrency["gaps"], [gap_expected(row) for row in tables["launch_gaps"]], "launch gaps")
    require_sequence(checker, concurrency["dependencies"], [dependency_expected(row) for row in tables["dependency_state"]], "dependency states")
    require_sequence(checker, concurrency["opportunities"], tables["opportunity_candidates"], "opportunity candidates")
    checker.require(any(row["n"] > 1 for row in concurrency["kernel_concurrency"]), "kernel overlap is represented")
    checker.require(any(row["material"] for row in concurrency["gaps"]), "material launch gaps are represented")
    checker.require(set(row["status"] for row in concurrency["opportunities"]) == {"candidate", "confirmed", "unavailable"}, "all opportunity states are represented")
    checker.require(all(row.get("claim_boundary") == "scheduling_candidate_only_no_speedup_claim" for row in concurrency["opportunities"]), "opportunities do not claim speedup")

    combined_text = "\n".join(parsed_pages[name][0] for name in PAGE_NAMES)
    checker.require("cross-capture clocks merged" not in combined_text.lower(), "no cross-clock merge claim appears")
    checker.require("achieved speedup" not in combined_text.lower(), "no achieved speedup claim appears")
    checker.require("achieved occupancy" not in combined_text.lower() or "not achieved occupancy" in combined_text.lower(), "achieved occupancy is not asserted")

    companions = acceptance.get("companions", {})
    checker.require(
        set(companions) == {PERFETTO_TRACE, FULL_TIMELINE_MANIFEST},
        "complete trace and completeness manifest companions are indexed",
    )
    trace_path = acceptance_dir / PERFETTO_TRACE
    trace_record = companions[PERFETTO_TRACE]
    checker.require(trace_record.get("path") == str(trace_path), "complete trace path matches")
    checker.require(trace_record.get("sha256") == sha256_file(trace_path), "complete trace hash matches")
    checker.require(trace_record.get("official_perfetto_parse_performed") is False, "complete trace is not called an official parse")
    checker.require(trace_record.get("sampling_performed") is False, "complete trace record denies sampling")
    complete_trace = load_object(trace_path)
    events = complete_trace.get("traceEvents")
    checker.require(isinstance(events, list) and len(events) == len(rows), "complete trace retains every E2E interval")
    checker.require(all(isinstance(row, dict) and row.get("ph") == "X" and number(row.get("dur"), -1) >= 0 for row in events), "complete trace is structurally Chrome JSON compatible")
    trace_metadata = complete_trace.get("metadata", {})
    checker.require(trace_metadata.get("source_clock") == "R07_non_replay_realtime_only", "complete trace uses only R07 clock")
    checker.require(trace_metadata.get("official_perfetto_parse_performed") is False, "complete trace metadata denies official parse")
    checker.require(trace_metadata.get("complete_timeline") is True, "complete trace declares full coverage")
    checker.require(trace_metadata.get("sampling_performed") is False, "complete trace metadata denies sampling")
    expected_category_counts = dict(sorted(Counter(row["g"] for row in rows).items()))
    checker.require(trace_metadata.get("event_count_by_category") == expected_category_counts, "complete trace category counts reproduce E2E")
    checker.require(
        trace_metadata.get("top_latency_processes")
        == expected_top_contract["selected"],
        "complete trace metadata embeds top-latency process mapping",
    )
    checker.require(
        {
            **trace_metadata.get("top_latency_process_policy", {}),
            "selected": trace_metadata.get("top_latency_processes"),
        }
        == expected_top_contract,
        "complete trace metadata embeds the complete top-latency contract",
    )
    checker.require(Counter(row.get("cat") for row in events) == Counter(row["g"] for row in rows), "complete trace category rows reproduce E2E")
    top_by_process = {
        row["process_range"]: row
        for row in expected_top_contract["selected"]
    }
    annotated_process_count = 0
    annotated_owner_count = 0
    for index, (source, event) in enumerate(zip(rows, events)):
        if (
            event.get("cat") != source["g"]
            or event.get("name") != source["n"]
            or event.get("args", {}).get("begin_ns") != str(source["b"])
            or event.get("args", {}).get("end_ns") != str(source["e"])
        ):
            raise RuntimeError(
                "independent R10 check failed: complete trace event "
                f"{index} differs from E2E"
            )
        top_owner = top_by_process.get(str(source.get("process", "")))
        args = event["args"]
        if top_owner is not None and source["g"] in {
            "process", "hip_runtime", "gpu_queue", "strict_owned_kernel"
        }:
            prefix = (
                "top_latency_process"
                if source["g"] == "process"
                else "top_latency_owner"
            )
            checker.require(
                args.get(f"{prefix}_rank") == top_owner["rank"]
                and args.get(f"{prefix}_process_range")
                == top_owner["process_range"]
                and args.get(f"{prefix}_color") == top_owner["color"]
                and args.get(f"{prefix}_observed_duration_ns")
                == str(top_owner["observed_duration_ns"]),
                f"complete trace event {index} has exact top-owner annotations",
            )
            if source["g"] == "process":
                annotated_process_count += 1
            else:
                annotated_owner_count += 1
    checker.count += 1
    checker.require(
        annotated_process_count == len(expected_top_contract["selected"]),
        "complete trace annotates every selected process exactly once",
    )
    checker.require(
        annotated_owner_count > 0,
        "complete trace annotates selected-process owned intervals",
    )

    full_manifest_path = acceptance_dir / FULL_TIMELINE_MANIFEST
    full_manifest_record = companions[FULL_TIMELINE_MANIFEST]
    checker.require(
        full_manifest_record.get("path") == str(full_manifest_path),
        "full timeline manifest path matches",
    )
    checker.require(
        full_manifest_record.get("sha256") == sha256_file(full_manifest_path),
        "full timeline manifest hash matches",
    )
    full_manifest = load_object(full_manifest_path)
    checker.require(full_manifest.get("status") == "PASS", "full timeline manifest passes")
    checker.require(
        full_manifest.get("formal_r09_r10_regeneration") is True,
        "full timeline manifest identifies formal R09/R10 regeneration",
    )
    checker.require(
        full_manifest.get("sampling_performed") is False,
        "full timeline manifest denies sampling",
    )
    checker.require(
        full_manifest.get("event_count") == len(rows),
        "full timeline manifest event count reproduces E2E",
    )
    checker.require(
        full_manifest.get("event_count_by_category") == expected_category_counts,
        "full timeline manifest category counts reproduce E2E",
    )
    checker.require(
        full_manifest.get("source_analysis_sha256")
        == acceptance["source_analysis"]["sha256"],
        "full timeline manifest binds source analysis",
    )
    checker.require(
        full_manifest.get("source_table_hashes")
        == acceptance["source_table_hashes"],
        "full timeline manifest binds source tables",
    )
    checker.require(
        full_manifest.get("top_latency_process_contract")
        == expected_top_contract,
        "full timeline manifest binds top-latency process contract",
    )
    checker.require(
        full_manifest.get("rectangle_label_groups")
        == list(TIMELINE_RECTANGLE_LABEL_GROUPS),
        "full timeline manifest binds every rectangle label group",
    )
    checker.require(
        full_manifest.get("outputs")
        == {
            "lossless_page": {
                "path": "E2E_PROCESS_TIMELINE_LOSSLESS.html",
                "sha256": acceptance["outputs"][
                    "E2E_PROCESS_TIMELINE_LOSSLESS.html"
                ]["sha256"],
            },
            "complete_trace": {
                "path": PERFETTO_TRACE,
                "sha256": trace_record["sha256"],
            },
        },
        "full timeline manifest binds lossless outputs",
    )

    deterministic_hashes: dict[str, str] = {}
    for name in DETERMINISTIC_FILES:
        primary = acceptance_dir / name
        reference = reference_dir / name
        checker.require(primary.is_file() and reference.is_file(), f"determinism pair exists: {name}")
        primary_hash = sha256_file(primary)
        reference_hash = sha256_file(reference)
        checker.require(primary_hash == reference_hash, f"deterministic regeneration matches: {name}")
        deterministic_hashes[name] = primary_hash

    generator_path = source_root / "scripts" / "perf_trace" / "generate_fresh_e2e_visualization.py"
    auditor_path = Path(__file__).resolve()
    finalizer_path = source_root / "scripts" / "perf_trace" / "finalize_qwen_r10_acceptance.py"
    checker.require(generator_path.is_file(), "maintained generator exists")
    checker.require(auditor_path.is_file(), "independent auditor exists")
    checker.require(finalizer_path.is_file(), "handoff finalizer exists")
    checker.require(acceptance["generator"] == {"path": str(generator_path), "sha256": sha256_file(generator_path)}, "acceptance binds maintained generator")

    git_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "-C", str(source_root), "branch", "--show-current"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "-z"]
    )
    git_status_sha256 = hashlib.sha256(git_status).hexdigest()
    source_lineage = {
        "schema_version": 1,
        "runtime_goal": "R10",
        "status": "PASS",
        "lineage_id": lineage_id,
        "source_change_policy": "stage_trace_instrumentation_allowed",
        "source_hash_equality_required": False,
        "model_input_sampling_device_semantics_changed": False,
        "r09_data_or_semantics_changed": False,
        "upstream_handoffs_rewritten": False,
        "presentation_only_stage": True,
        "source_root": str(source_root),
        "git_revision": git_revision,
        "git_branch": git_branch,
        "git_status_porcelain_v1_z_sha256": git_status_sha256,
        "source_files": {
            "generator": {
                "path": str(generator_path),
                "pre_stage_sha256": args.pre_generator_sha256,
                "sha256": sha256_file(generator_path),
                "change": "strict same-run inputs, complete offline payloads, arbitrary-depth relative-ns viewer, visible evidence classes, R08 PMC/resources, and deterministic unsampled full trace",
            },
            "independent_auditor": {
                "path": str(auditor_path), "sha256": sha256_file(auditor_path)
            },
            "handoff_finalizer": {
                "path": str(finalizer_path), "sha256": sha256_file(finalizer_path)
            },
        },
        "immutable_inputs": {
            "R06_handoff_sha256": handoff_hashes["R06"],
            "R07_handoff_sha256": handoff_hashes["R07"],
            "R08_handoff_sha256": handoff_hashes["R08"],
            "R09_handoff_sha256": handoff_hashes["R09"],
            "R09_analysis_sha256": sha256_file(analysis_path),
            "R09_normalized_table_hashes": expected_hashes,
        },
        "presentation_backend": acceptance["presentation_backend"],
        "runtime_activity": {
            "model_run_count": 0, "gpu_probe_count": 0, "profiler_run_count": 0,
            "trace_capture_count": 0, "pmc_replay_count": 0,
            "additional_sampling_count": 0, "network_download_count": 0,
        },
    }
    write_new_json(source_lineage_output, source_lineage)

    primary_acceptance_bytes = unique_inode_bytes(acceptance_dir)
    checker.require(primary_acceptance_bytes <= args.maximum_trace_bundle_bytes, "primary acceptance remains within bundle budget")
    audit = {
        "schema_version": 1,
        "runtime_goal": "R10",
        "status": "PASS",
        "lineage_id": lineage_id,
        "independent_check_count": checker.count,
        "independent_failure_check_count": 0,
        "validation": {
            "page_count": len(PAGE_NAMES),
            "all_pages_parsed": True,
            "external_script_or_resource_url_count": 0,
            "broken_relative_link_count": 0,
            "source_table_hashes_verified": True,
            "embedded_row_counts_and_hashes_verified": True,
            "request_bounds_reproduced": True,
            "high_latency_process_count": len(expected_high),
            "high_latency_selection_reproduced": True,
            "all_high_latency_processes_have_exact_owned_kernels": True,
            "all_high_latency_processes_have_exact_in_window_live_samples": True,
            "evidence_legends_complete": True,
            "filters_search_zoom_complete": True,
            "top_latency_process_ranking_reproduced": True,
            "top_latency_process_distinct_fill_colors": True,
            "top_latency_owned_interval_outlines": True,
            "zoom_reveals_process_names_inside_rectangles": True,
            "all_timeline_rectangle_label_groups_verified": True,
            "zoom_reveals_all_timeline_labels_inside_rectangles": True,
            "labeled_timeline_rectangle_count": len(rows),
            "unlabeled_timeline_rectangle_count": 0,
            "lossless_relative_nanosecond_timeline_complete": True,
            "complete_perfetto_event_count": len(events),
            "complete_perfetto_sampling_performed": False,
            "cross_clock_or_replay_latency_arithmetic": False,
            "speedup_claimed": False,
            "official_perfetto_parse_claimed": False,
            "complete_compatible_trace_structural_status": "PASS",
        },
        "determinism": {
            "status": "PASS",
            "method": "two independent generator invocations at the canonical output path with byte-for-byte comparison",
            "compared_file_count": len(DETERMINISTIC_FILES),
            "identical_sha256_by_file": deterministic_hashes,
            "reference_location_at_validation": str(reference_dir),
        },
        "outputs": {
            "offline_acceptance_manifest": {
                "path": str(acceptance_manifest_path),
                "sha256": sha256_file(acceptance_manifest_path),
            },
            "source_lineage": {
                "path": str(source_lineage_output),
                "sha256": sha256_file(source_lineage_output),
            },
            **{
                name: {"path": str(acceptance_dir / name), "sha256": sha256_file(acceptance_dir / name)}
                for name in PAGE_NAMES
            },
        },
        "artifact_budget": {
            "primary_acceptance_bytes": primary_acceptance_bytes,
            "maximum_trace_bundle_bytes": args.maximum_trace_bundle_bytes,
            "profiling_wall_time_seconds": 0,
            "within_limit": True,
        },
        "evidence_boundary": {
            "latency_axis": "R07_non_replay_same_request_only",
            "live_utilization": "observed eligible R07 SE samples",
            "hardware": "R08 replay-projected attributes with no replay timestamps",
            "traffic": "inferred FX-visible logical bytes, not HBM/DRAM traffic",
            "opportunity": "seven-gate scheduling classification without speedup claim",
            "presentation": "custom offline lossless viewer; complete unsampled Chrome JSON structurally checked, not officially parsed",
        },
    }
    write_new_json(audit_output, audit)
    print(json.dumps({
        "status": "PASS", "checks": checker.count,
        "source_lineage": str(source_lineage_output),
        "audit": str(audit_output), "audit_sha256": sha256_file(audit_output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
