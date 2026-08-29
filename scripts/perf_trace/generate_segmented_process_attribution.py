#!/usr/bin/env python3
"""Generate a conservative full-layer Qwen process attribution package.

The script combines the complete R01 layer denominators with the representative
R03 process distributions.  Representative distributions are normalized to
each target layer metric; representative absolute latencies are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


getcontext().prec = 50

DECIMAL_ZERO = Decimal("0")
OUTPUT_QUANTUM = Decimal("0.000000000001")
VARIANT_REPORT_NAME = (
    "SAME_INPUT_QWEN3_5_27B_VLLM_PRA_FULL_EAGER_"
    "FULL_LAYER_PROCESS_ATTRIBUTION_REPORT.md"
)
BREAKDOWN_REPORT_NAME = "SAME_INPUT_FULL_LAYER_PROCESS_ATTRIBUTION_BREAKDOWN.md"
TYPE_MAP_NAME = "full_layer_attribution_type_map.csv"
ASSIGNMENT_NAME = "full_layer_template_assignment.csv"
ATTRIBUTION_NAME = "full_layer_process_attribution.csv"
AGGREGATION_NAME = "full_layer_process_aggregation.csv"
COVERAGE_NAME = "full_layer_coverage_and_risk.csv"
RUN_CONTRACT_NAME = "R05_RUN_CONTRACT.json"
GENERATION_AUDIT_NAME = "R05_GENERATION_AUDIT.json"

REQUIRED_OUTPUT_NAMES = (
    VARIANT_REPORT_NAME,
    BREAKDOWN_REPORT_NAME,
    TYPE_MAP_NAME,
    ASSIGNMENT_NAME,
    ATTRIBUTION_NAME,
    AGGREGATION_NAME,
    COVERAGE_NAME,
)

METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "metric": "hiptx_host_range_duration_ms",
        "metric_role": "host_context",
        "metric_semantic": "HIPTX host range duration",
        "template_field": "allocated_nvtx_cpu_ms",
        "template_semantic": (
            "R03 allocated NVTX-compatible CPU distribution, resolved to "
            "HIPTX host-range duration"
        ),
        "fraction_status": "same_host_metric_template",
    },
    {
        "metric": "hipprof_launch_owned_kernel_sum_ms",
        "metric_role": "downstream_denominator",
        "metric_semantic": "hipprof HIPOPS launch-owned kernel duration sum",
        "template_field": "allocated_cupti_kernel_ms",
        "template_semantic": (
            "R03 allocated CUPTI-compatible GPU distribution, resolved to "
            "hipprof HIPOPS launch-owned kernel duration"
        ),
        "fraction_status": "same_launch_owned_kernel_metric_template",
    },
    {
        "metric": "hipprof_launch_owned_kernel_busy_union_ms",
        "metric_role": "diagnostic_only",
        "metric_semantic": (
            "hipprof launch-owned kernel device busy-time union diagnostic"
        ),
        "template_field": "allocated_cupti_kernel_ms",
        "template_semantic": (
            "explicit diagnostic proxy using the representative launch-owned "
            "kernel-sum fractions"
        ),
        "fraction_status": "diagnostic_kernel_sum_fraction_proxy",
    },
)

METRIC_BY_NAME = {spec["metric"]: spec for spec in METRIC_SPECS}
EVENT_RE = re.compile(r"^input(?P<forward>[0-9]+)_layer(?P<layer>[0-9]+)$")

TYPE_DEFINITIONS: dict[str, dict[str, str]] = {
    "A1_observed_fx_op_exact": {
        "attribution_type_id": "A1_observed_fx_op_exact",
        "attribution_source": "observed_fx_op",
        "target_is_representative": "true",
        "representative_match": "exact",
        "target_template_shape_relation": "exact",
        "evidence_class": "observed_representative",
        "confidence": "representative_exact",
        "direct_full_layer_timing": "false",
        "description": (
            "Exact target representative with exact R03 shape matching. "
            "The representative process/FX structure is observed; process "
            "timing columns remain R03 attribution rather than direct "
            "per-process timing."
        ),
    },
    "A2_observed_fx_op_nearest_shape_fallback": {
        "attribution_type_id": "A2_observed_fx_op_nearest_shape_fallback",
        "attribution_source": "fallback_nearest_shape_observed_fx_op",
        "target_is_representative": "true",
        "representative_match": "nearest_shape",
        "target_template_shape_relation": "exact_perf_source",
        "evidence_class": "observed_structure_exploratory_shape",
        "confidence": "exploratory",
        "direct_full_layer_timing": "false",
        "description": (
            "The target is a representative event, but the R03 structural FX "
            "shape differs from its matched R01 performance shape. The "
            "nearest-shape fallback is explicit."
        ),
    },
    "A3_template_scaled_exact_shape": {
        "attribution_type_id": "A3_template_scaled_exact_shape",
        "attribution_source": "template_scaled",
        "target_is_representative": "false",
        "representative_match": "exact",
        "target_template_shape_relation": "exact",
        "evidence_class": "template_estimate",
        "confidence": "estimate_exact_shape",
        "direct_full_layer_timing": "false",
        "description": (
            "A same-phase, same-attention-path exact-shape template is "
            "normalized to the target layer denominator."
        ),
    },
    "A4_template_scaled_from_nearest_fx_fallback": {
        "attribution_type_id": "A4_template_scaled_from_nearest_fx_fallback",
        "attribution_source": "fallback_nearest_shape_template_scaled",
        "target_is_representative": "false",
        "representative_match": "nearest_shape",
        "target_template_shape_relation": "exact_perf_source",
        "evidence_class": "template_estimate_exploratory_fx_shape",
        "confidence": "exploratory",
        "direct_full_layer_timing": "false",
        "description": (
            "The target matches the representative performance-source shape, "
            "but that representative uses an exploratory nearest-shape FX "
            "structural match."
        ),
    },
    "A5_template_scaled_nearest_target_shape_fallback": {
        "attribution_type_id": "A5_template_scaled_nearest_target_shape_fallback",
        "attribution_source": "fallback_nearest_shape_template_scaled",
        "target_is_representative": "false",
        "representative_match": "exact",
        "target_template_shape_relation": "nearest",
        "evidence_class": "template_estimate_nearest_target_shape",
        "confidence": "exploratory",
        "direct_full_layer_timing": "false",
        "description": (
            "The representative itself is exact, but the target q_len/kv_len "
            "shape is different. The target-shape distance is exposed."
        ),
    },
    "A6_template_scaled_dual_shape_fallback": {
        "attribution_type_id": "A6_template_scaled_dual_shape_fallback",
        "attribution_source": "fallback_nearest_shape_template_scaled",
        "target_is_representative": "false",
        "representative_match": "nearest_shape",
        "target_template_shape_relation": "nearest",
        "evidence_class": "template_estimate_dual_shape_fallback",
        "confidence": "exploratory",
        "direct_full_layer_timing": "false",
        "description": (
            "Both the target-to-template performance shape and the template's "
            "FX-to-performance shape differ. Both mismatches are explicit."
        ),
    },
}


class AttributionError(RuntimeError):
    """Raised when an evidence or conservation gate fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def nested(data: Mapping[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        require(isinstance(current, Mapping) and key in current,
                f"missing handoff field: {'.'.join(keys)}")
        current = current[key]
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_state(source_root: Path) -> dict[str, str]:
    def git_output(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
        )
        require(
            result.returncode == 0,
            f"git {' '.join(args)} failed: "
            f"{result.stderr.decode(errors='replace').strip()}",
        )
        return result.stdout

    status = git_output("status", "--porcelain=v1", "-z")
    return {
        "revision": git_output("rev-parse", "HEAD").decode().strip(),
        "branch": git_output(
            "rev-parse", "--abbrev-ref", "HEAD"
        ).decode().strip(),
        "status_porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
    }


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and value, f"JSON must be a non-empty object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"missing CSV input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        rows = list(reader)
    require(rows, f"CSV is empty: {path}")
    return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: Sequence[str],
              rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def decimal_value(value: str, context: str) -> Decimal:
    try:
        result = Decimal(value)
    except Exception as exc:  # pragma: no cover - exact exception varies
        raise AttributionError(f"invalid decimal {value!r} in {context}") from exc
    require(result.is_finite(), f"non-finite decimal {value!r} in {context}")
    return result


def format_decimal(value: Decimal, places: int = 12) -> str:
    quantum = Decimal(1).scaleb(-places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")


def integer_value(value: str, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise AttributionError(f"invalid integer {value!r} in {context}") from exc


def validate_bound_file(given: Path, metadata: Mapping[str, Any],
                        label: str) -> dict[str, Any]:
    expected_path = Path(str(nested(metadata, "path"))).resolve()
    actual_path = given.resolve()
    require(actual_path == expected_path,
            f"{label} path is not the handoff-bound path: {actual_path} != {expected_path}")
    require(actual_path.is_file(), f"missing {label}: {actual_path}")
    expected_hash = str(nested(metadata, "sha256"))
    actual_hash = sha256_file(actual_path)
    require(actual_hash == expected_hash,
            f"{label} SHA-256 mismatch: {actual_hash} != {expected_hash}")
    return {
        "path": str(actual_path),
        "sha256": actual_hash,
        "size_bytes": actual_path.stat().st_size,
    }


def event_sort_key(event_id: str) -> tuple[int, int]:
    match = EVENT_RE.fullmatch(event_id)
    require(match is not None, f"invalid representative event ID: {event_id}")
    return int(match.group("forward")), int(match.group("layer"))


def layer_region(layer_idx: int, layer_count: int) -> str:
    require(layer_count > 0, "layer count must be positive")
    first_boundary = layer_count // 3
    second_boundary = (2 * layer_count) // 3
    if layer_idx < first_boundary:
        return "early"
    if layer_idx < second_boundary:
        return "middle"
    return "late"


def shape_class(phase: str, q_len: int, kv_len: int) -> str:
    if phase == "decode" and q_len == 1:
        return "decode_q1"
    if phase == "prefill_chunk" and q_len == 4096:
        return "prefill_full_chunk_q4096"
    if phase == "prefill_chunk" and q_len < 4096:
        return f"prefill_tail_q{q_len}"
    return f"{phase}_q{q_len}_kv{kv_len}"


def relative_delta(left: int, right: int) -> Decimal:
    denominator = max(abs(left), abs(right), 1)
    return Decimal(abs(left - right)) / Decimal(denominator)


def allocate_conserving(total: Decimal,
                        weights: Sequence[Decimal]) -> list[Decimal]:
    require(total >= DECIMAL_ZERO, f"negative target metric: {total}")
    require(weights, "cannot allocate with an empty process template")
    require(all(value >= DECIMAL_ZERO for value in weights),
            "negative representative process weight")
    weight_sum = sum(weights, DECIMAL_ZERO)
    require(weight_sum > DECIMAL_ZERO,
            "representative process weights sum to zero")
    positive_indices = [index for index, value in enumerate(weights) if value > 0]
    require(positive_indices, "representative process template has no positive row")
    residual_index = positive_indices[-1]
    allocations: list[Decimal] = [DECIMAL_ZERO for _ in weights]
    running = DECIMAL_ZERO
    for index, weight in enumerate(weights):
        if index == residual_index or weight == 0:
            continue
        value = (total * weight / weight_sum).quantize(
            OUTPUT_QUANTUM, rounding=ROUND_HALF_EVEN
        )
        allocations[index] = value
        running += value
    residual = total - running
    require(residual >= DECIMAL_ZERO,
            f"rounding produced a negative conservation residual: {residual}")
    allocations[residual_index] = residual
    require(sum(allocations, DECIMAL_ZERO) == total,
            "internal allocation failed exact Decimal conservation")
    return allocations


def output_metadata(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "path_scope": "output_dir_relative",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-layer Qwen process attribution from compatible "
            "R01/R02/R03 SAME_INPUT evidence."
        )
    )
    parser.add_argument("--r01-handoff", required=True, type=Path)
    parser.add_argument("--r02-handoff", required=True, type=Path)
    parser.add_argument("--r03-handoff", required=True, type=Path)
    parser.add_argument("--r04-handoff", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--full-input-layer-csv", required=True, type=Path)
    parser.add_argument("--layer-kernel-csv", required=True, type=Path)
    parser.add_argument("--representative-process-csv", required=True, type=Path)
    parser.add_argument("--representative-report", required=True, type=Path)
    parser.add_argument("--layer-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--runtime-goal", default="R05")
    parser.add_argument("--tolerance-ms", default="1e-9")
    parser.add_argument(
        "--workflow05-policy-version",
        default="workflow05-low-cost-timeline-v4",
    )
    parser.add_argument(
        "--evidence-acquisition-mode",
        default="fresh_no_prior_runtime_reuse",
    )
    parser.add_argument("--target-cumulative-latency-coverage", default="1.0")
    parser.add_argument("--maximum-selected-layer-input-count", type=int,
                        default=4096)
    parser.add_argument("--maximum-selected-process-count", type=int,
                        default=25000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tolerance = decimal_value(args.tolerance_ms, "--tolerance-ms")
    require(tolerance > 0, "tolerance must be positive")
    require(args.runtime_goal == "R05", "this generator invocation is restricted to R05")
    target_coverage = decimal_value(
        args.target_cumulative_latency_coverage,
        "--target-cumulative-latency-coverage",
    )
    require(DECIMAL_ZERO <= target_coverage <= Decimal("1"),
            "target cumulative latency coverage must be in [0, 1]")
    require(args.maximum_selected_layer_input_count > 0,
            "maximum selected layer-input count must be positive")
    require(args.maximum_selected_process_count > 0,
            "maximum selected process count must be positive")
    require(
        args.evidence_acquisition_mode == "fresh_no_prior_runtime_reuse",
        "R05 fresh full-request mode requires fresh_no_prior_runtime_reuse",
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.resolve()
    require(source_root.is_dir(), f"source root is missing: {source_root}")
    require(
        Path(__file__).resolve().is_relative_to(source_root),
        "R05 generator is outside the requested source root",
    )
    r05_source_state = current_source_state(source_root)

    r01_path = args.r01_handoff.resolve()
    r02_path = args.r02_handoff.resolve()
    r03_path = args.r03_handoff.resolve()
    r04_path = args.r04_handoff.resolve()
    r01 = read_json(r01_path)
    r02 = read_json(r02_path)
    r03 = read_json(r03_path)
    r04 = read_json(r04_path)

    require(r01.get("runtime_goal") == "R01" and r01.get("status") == "complete",
            "R01 handoff is not complete")
    require(r02.get("runtime_goal") == "R02" and r02.get("status") == "complete",
            "R02 handoff is not complete")
    require(r03.get("runtime_goal") == "R03" and r03.get("status") == "complete",
            "R03 handoff is not complete")
    require(r04.get("runtime_goal") == "R04" and r04.get("status") == "complete",
            "R04 handoff is not complete")
    for goal, handoff in (
        ("R01", r01), ("R02", r02), ("R03", r03), ("R04", r04)
    ):
        require(handoff.get("run_id") == args.run_id,
                f"{goal} handoff run ID differs from R05")
        require(handoff.get("branch") == args.branch,
                f"{goal} handoff branch differs from R05")
        require(
            handoff.get("workflow05_policy_version")
            == args.workflow05_policy_version,
            f"{goal} handoff Workflow05 policy differs from R05",
        )
    require(nested(r01, "run", "fresh_non_replay") is True,
            "R01 denominator is not marked fresh/non-replay")
    for goal, handoff in (("R02", r02), ("R03", r03), ("R04", r04)):
        require(
            handoff.get("evidence_acquisition_mode")
            == args.evidence_acquisition_mode,
            f"{goal} evidence acquisition mode differs from R05",
        )
    require(nested(r03, "upstream", "prior_runtime_evidence_used") is False,
            "R03 reports prior-runtime evidence use")
    require(
        nested(r03, "evidence_boundary", "historical_runtime_evidence_used")
        is False,
        "R03 reports historical runtime evidence use",
    )

    r01_hash = sha256_file(r01_path)
    r02_hash = sha256_file(r02_path)
    r03_hash = sha256_file(r03_path)
    r04_hash = sha256_file(r04_path)
    require(r02_hash == nested(r03, "component_source", "handoff_file_sha256"),
            "R02 handoff hash does not match the R03 binding")
    require(Path(nested(r03, "component_source", "handoff_path")).resolve() == r02_path,
            "R03 binds a different R02 handoff path")

    contract_id = str(nested(r01, "contract", "contract_id"))
    contract_sha = str(nested(r01, "contract", "canonical_sha256"))
    require(nested(r02, "contract", "contract_id") == contract_id,
            "R02 parent contract differs from R01")
    require(nested(r02, "contract", "canonical_sha256") == contract_sha,
            "R02 parent contract SHA differs from R01")
    require(nested(r03, "same_input_parent", "contract_id") == contract_id,
            "R03 parent contract differs from R01")
    require(nested(r03, "same_input_parent", "contract_canonical_sha256") == contract_sha,
            "R03 parent contract SHA differs from R01")
    require(nested(r03, "component_source", "parent_contract_id") == contract_id,
            "R03 component source parent contract differs from R01")
    require(nested(r04, "same_input_parent", "contract_id") == contract_id,
            "R04 parent contract differs from R01")
    require(
        nested(r04, "same_input_parent", "contract_canonical_sha256")
        == contract_sha,
        "R04 parent contract SHA differs from R01",
    )

    r01_source_revision = str(nested(r01, "source", "git_revision"))
    stage_source_revisions = {
        "R01": r01_source_revision,
        "R02": str(nested(r02, "source", "git_revision")),
        "R03": str(nested(r03, "source", "git_revision")),
        "R04": str(nested(r04, "live_toolchain", "source_revision")),
        "R05": r05_source_state["revision"],
    }

    input_bindings = {
        "full_input_layer_performance": validate_bound_file(
            args.full_input_layer_csv,
            nested(r01, "primary_outputs", "all_input_layer_performance"),
            "full input-layer performance CSV",
        ),
        "layer_kernel_breakdown": validate_bound_file(
            args.layer_kernel_csv,
            nested(r01, "primary_outputs", "layer_kernel_breakdown_csv"),
            "layer kernel-breakdown CSV",
        ),
        "representative_process_attribution": validate_bound_file(
            args.representative_process_csv,
            nested(r03, "primary_outputs", "per_variant_process_attribution"),
            "representative process attribution CSV",
        ),
        "representative_process_report": validate_bound_file(
            args.representative_report,
            nested(r03, "primary_outputs", "per_variant_report"),
            "representative process report",
        ),
        "layer_performance_report": validate_bound_file(
            args.layer_report,
            nested(r01, "primary_outputs", "report"),
            "layer performance report",
        ),
    }

    r02_db_binding = validate_bound_file(
        Path(nested(r02, "primary_outputs", "queryable_process_trace", "path")),
        nested(r02, "primary_outputs", "queryable_process_trace"),
        "R02 strict process trace database",
    )
    require(
        r02_db_binding["sha256"]
        == nested(r03, "component_source", "process_db", "sha256"),
        "R02 process DB hash differs from the R03 binding",
    )
    r02_inventory_binding = validate_bound_file(
        Path(nested(r02, "primary_outputs", "process_range_inventory_csv", "path")),
        nested(r02, "primary_outputs", "process_range_inventory_csv"),
        "R02 process inventory",
    )
    require(
        r02_inventory_binding["sha256"]
        == nested(r03, "component_source", "process_inventory", "sha256"),
        "R02 inventory hash differs from the R03 binding",
    )

    require(
        nested(r03, "evidence_boundary", "cupti_compatibility_field")
        == "hipprof HIPOPS launch-owned kernel duration",
        "R03 does not confirm the CUPTI-compatible hipprof HIPOPS semantic",
    )
    require(
        nested(r03, "evidence_boundary", "nvtx_compatibility_field")
        == "HIPTX host range duration",
        "R03 does not confirm the NVTX-compatible HIPTX semantic",
    )
    require(
        nested(r03, "evidence_boundary", "device_timestamp_overlap_attribution_used")
        is False,
        "R03 used prohibited device timestamp-overlap attribution",
    )
    require(
        nested(r03, "projection_method", "output_is_direct_process_timing")
        is False,
        "unexpected R03 direct-timing claim",
    )
    require(nested(r02, "trace_binding", "correlation_identity") == "_Index",
            "R02 strict process trace does not use runtime _Index")
    require(nested(r03, "component_source", "correlation_identity") == "_Index",
            "R03 component source does not use runtime _Index")
    require(nested(r02, "trace_binding", "kernel_durations_calculated") is False,
            "R02 unexpectedly claims process kernel durations")
    require(nested(r03, "component_source", "multiply_owned_runtime_indices") == 0,
            "R03 contains multiply owned runtime indices")
    require(
        nested(r03, "component_source", "strict_owned_kernel_count")
        == nested(r03, "component_source", "unique_strict_owned_runtime_indices"),
        "R03 strict kernel/runtime-index counts disagree",
    )

    denominator_header, denominator_rows = read_csv(
        args.full_input_layer_csv.resolve()
    )
    required_denominator_columns = {
        "contract_id",
        "contract_sha256",
        "forward_id",
        "layer_idx",
        "occurrence",
        "occurrence_key",
        "metric",
        "metric_value_ms",
        "metric_role",
        "phase",
        "q_len",
        "past_len",
        "kv_len",
        "workload_type",
        "range_name",
        "launch_owned_kernel_count",
        "attribution_status",
        "ownership_method",
    }
    require(required_denominator_columns.issubset(denominator_header),
            "full input-layer CSV schema is incomplete")

    denominator_by_group: dict[
        tuple[int, int, int], dict[str, dict[str, str]]
    ] = defaultdict(dict)
    context_by_group: dict[tuple[int, int, int], dict[str, Any]] = {}
    occurrence_key_to_group: dict[str, tuple[int, int, int]] = {}

    for row_number, row in enumerate(denominator_rows, start=2):
        context = f"denominator row {row_number}"
        require(row["contract_id"] == contract_id,
                f"{context}: mixed contract ID")
        require(row["contract_sha256"] == contract_sha,
                f"{context}: mixed contract SHA")
        require(row["metric"] in METRIC_BY_NAME,
                f"{context}: unsupported metric {row['metric']!r}")
        spec = METRIC_BY_NAME[row["metric"]]
        require(row["metric_role"] == spec["metric_role"],
                f"{context}: metric role mismatch")
        metric_value = decimal_value(row["metric_value_ms"], context)
        require(metric_value >= 0, f"{context}: negative metric")
        forward_id = integer_value(row["forward_id"], context)
        layer_idx = integer_value(row["layer_idx"], context)
        occurrence = integer_value(row["occurrence"], context)
        q_len = integer_value(row["q_len"], context)
        past_len = integer_value(row["past_len"], context)
        kv_len = integer_value(row["kv_len"], context)
        require(row["attribution_status"] == "pass",
                f"{context}: upstream attribution did not pass")
        require("HIPTX host range" in row["ownership_method"]
                and "HIP Runtime" in row["ownership_method"]
                and "HIPOPS" in row["ownership_method"],
                f"{context}: unresolved ROCm/HIP ownership semantic")
        group_key = (forward_id, layer_idx, occurrence)
        require(row["metric"] not in denominator_by_group[group_key],
                f"{context}: duplicate occurrence metric")
        denominator_by_group[group_key][row["metric"]] = row
        current_context = {
            "forward_id": forward_id,
            "layer_idx": layer_idx,
            "occurrence": occurrence,
            "occurrence_key": row["occurrence_key"],
            "phase": row["phase"],
            "q_len": q_len,
            "past_len": past_len,
            "kv_len": kv_len,
            "workload_type": row["workload_type"],
            "range_name": row["range_name"],
            "launch_owned_kernel_count": integer_value(
                row["launch_owned_kernel_count"], context
            ),
        }
        if group_key in context_by_group:
            require(context_by_group[group_key] == current_context,
                    f"{context}: metric rows disagree on occurrence context")
        else:
            context_by_group[group_key] = current_context
        prior_group = occurrence_key_to_group.setdefault(row["occurrence_key"], group_key)
        require(prior_group == group_key,
                f"{context}: occurrence key is not unique")

    expected_metrics = set(METRIC_BY_NAME)
    for group_key, rows in denominator_by_group.items():
        require(set(rows) == expected_metrics,
                f"occurrence {group_key} does not contain exactly three metrics")
        kernel_sum = decimal_value(
            rows["hipprof_launch_owned_kernel_sum_ms"]["metric_value_ms"],
            f"kernel sum {group_key}",
        )
        busy_union = decimal_value(
            rows["hipprof_launch_owned_kernel_busy_union_ms"]["metric_value_ms"],
            f"busy union {group_key}",
        )
        require(busy_union <= kernel_sum + tolerance,
                f"busy union exceeds kernel sum for {group_key}")

    forward_ids = sorted({key[0] for key in denominator_by_group})
    require(forward_ids == list(range(1, len(forward_ids) + 1)),
            "forward IDs are not contiguous from 1")
    layer_indices = sorted({key[1] for key in denominator_by_group})
    expected_layer_count = int(nested(r01, "model", "num_hidden_layers"))
    require(layer_indices == list(range(expected_layer_count)),
            "layer indices do not cover the full model")
    for forward_id in forward_ids:
        groups = [key for key in denominator_by_group if key[0] == forward_id]
        require(len(groups) == expected_layer_count,
                f"forward {forward_id} is not a complete layer schedule")
        require({key[1] for key in groups} == set(layer_indices),
                f"forward {forward_id} has incomplete layer indices")
    require(len(denominator_rows) == int(
        nested(r01, "primary_outputs", "all_input_layer_performance", "rows")
    ), "denominator row count differs from R01")

    kernel_header, kernel_rows = read_csv(args.layer_kernel_csv.resolve())
    required_kernel_columns = {
        "contract_id",
        "forward_id",
        "layer_idx",
        "occurrence",
        "phase",
        "q_len",
        "past_len",
        "kv_len",
        "workload_type",
        "kernel_family",
        "launch_owned_kernel_duration_ms",
        "layer_launch_owned_kernel_sum_ms",
        "ownership_method",
    }
    require(required_kernel_columns.issubset(kernel_header),
            "layer kernel-breakdown CSV schema is incomplete")
    kernel_sum_by_group: dict[tuple[int, int, int], Decimal] = defaultdict(
        lambda: DECIMAL_ZERO
    )
    for row_number, row in enumerate(kernel_rows, start=2):
        context = f"layer kernel row {row_number}"
        require(row["contract_id"] == contract_id,
                f"{context}: mixed contract ID")
        group_key = (
            integer_value(row["forward_id"], context),
            integer_value(row["layer_idx"], context),
            integer_value(row["occurrence"], context),
        )
        require(group_key in denominator_by_group,
                f"{context}: no denominator occurrence")
        layer_context = context_by_group[group_key]
        require(
            (
                row["phase"],
                integer_value(row["q_len"], context),
                integer_value(row["past_len"], context),
                integer_value(row["kv_len"], context),
                row["workload_type"],
            )
            == (
                layer_context["phase"],
                layer_context["q_len"],
                layer_context["past_len"],
                layer_context["kv_len"],
                layer_context["workload_type"],
            ),
            f"{context}: context differs from denominator",
        )
        require("identical _Index" in row["ownership_method"],
                f"{context}: unresolved launch ownership")
        kernel_sum_by_group[group_key] += decimal_value(
            row["launch_owned_kernel_duration_ms"], context
        )
        declared_total = decimal_value(
            row["layer_launch_owned_kernel_sum_ms"], context
        )
        denominator_total = decimal_value(
            denominator_by_group[group_key][
                "hipprof_launch_owned_kernel_sum_ms"
            ]["metric_value_ms"],
            context,
        )
        require(abs(declared_total - denominator_total) <= tolerance,
                f"{context}: declared layer total differs from denominator")
    require(set(kernel_sum_by_group) == set(denominator_by_group),
            "kernel breakdown does not cover every denominator occurrence")
    for group_key, total in kernel_sum_by_group.items():
        denominator_total = decimal_value(
            denominator_by_group[group_key][
                "hipprof_launch_owned_kernel_sum_ms"
            ]["metric_value_ms"],
            f"kernel group {group_key}",
        )
        require(abs(total - denominator_total) <= tolerance,
                f"kernel-family sum differs from denominator for {group_key}")

    representative_header, representative_rows = read_csv(
        args.representative_process_csv.resolve()
    )
    required_representative_columns = {
        "variant",
        "display_name",
        "fx_event_id",
        "layer",
        "phase",
        "process",
        "title",
        "allocated_cupti_kernel_ms",
        "allocated_nvtx_cpu_ms",
        "fx_q_len",
        "fx_kv_len",
        "match",
        "perf_q_len",
        "perf_kv_len",
        "nodes",
        "bucket",
        "source_total_cupti_kernel_ms",
        "source_total_nvtx_cpu_ms",
        "source_forward_id",
        "source_occurrence",
        "source_occurrence_key",
    }
    require(required_representative_columns.issubset(representative_header),
            "representative process CSV schema is incomplete")
    variants = {row["variant"] for row in representative_rows}
    display_names = {row["display_name"] for row in representative_rows}
    require(len(variants) == 1 and len(display_names) == 1,
            "representative process CSV mixes variants")
    variant = next(iter(variants))
    display_name = next(iter(display_names))

    representative_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in representative_rows:
        representative_by_event[row["fx_event_id"]].append(row)
    require(representative_by_event,
            "representative process evidence contains no events")

    templates: dict[str, dict[str, Any]] = {}
    process_metadata: dict[str, tuple[str, str]] = {}
    process_sets_by_layer_type: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for event_id in sorted(representative_by_event, key=event_sort_key):
        rows = representative_by_event[event_id]
        event_match = EVENT_RE.fullmatch(event_id)
        require(event_match is not None, f"invalid event ID: {event_id}")
        stable_fields = (
            "variant",
            "display_name",
            "layer",
            "phase",
            "fx_q_len",
            "fx_kv_len",
            "match",
            "perf_q_len",
            "perf_kv_len",
            "source_total_cupti_kernel_ms",
            "source_total_nvtx_cpu_ms",
            "source_forward_id",
            "source_occurrence",
            "source_occurrence_key",
        )
        for field in stable_fields:
            require(len({row[field] for row in rows}) == 1,
                    f"representative event {event_id} mixes {field}")
        layer_idx = integer_value(rows[0]["layer"], event_id)
        source_forward_id = integer_value(rows[0]["source_forward_id"], event_id)
        source_occurrence = integer_value(rows[0]["source_occurrence"], event_id)
        require(int(event_match.group("forward")) == source_forward_id,
                f"{event_id}: event input and source forward differ")
        require(int(event_match.group("layer")) == layer_idx,
                f"{event_id}: event and source layer differ")
        source_group = (source_forward_id, layer_idx, source_occurrence)
        require(source_group in denominator_by_group,
                f"{event_id}: source occurrence is absent from R01")
        source_context = context_by_group[source_group]
        require(source_context["occurrence_key"] == rows[0]["source_occurrence_key"],
                f"{event_id}: source occurrence key differs from R01")
        perf_q_len = integer_value(rows[0]["perf_q_len"], event_id)
        perf_kv_len = integer_value(rows[0]["perf_kv_len"], event_id)
        fx_q_len = integer_value(rows[0]["fx_q_len"], event_id)
        fx_kv_len = integer_value(rows[0]["fx_kv_len"], event_id)
        require(
            (
                rows[0]["phase"],
                perf_q_len,
                perf_kv_len,
            )
            == (
                source_context["phase"],
                source_context["q_len"],
                source_context["kv_len"],
            ),
            f"{event_id}: R03 performance source context differs from R01",
        )
        match_kind = rows[0]["match"]
        require(match_kind in {"exact", "nearest_shape"},
                f"{event_id}: unsupported representative match {match_kind!r}")
        if match_kind == "exact":
            require((fx_q_len, fx_kv_len) == (perf_q_len, perf_kv_len),
                    f"{event_id}: exact label hides a shape mismatch")
        else:
            require((fx_q_len, fx_kv_len) != (perf_q_len, perf_kv_len),
                    f"{event_id}: nearest_shape has no visible mismatch")

        process_ids = [row["process"] for row in rows]
        require(all(process_ids), f"{event_id}: empty process ID")
        require(len(process_ids) == len(set(process_ids)),
                f"{event_id}: duplicate process ID")
        for row in rows:
            require(row["title"], f"{event_id}: empty process title")
            require(row["bucket"], f"{event_id}: empty process bucket")
            require(integer_value(row["nodes"], event_id) >= 0,
                    f"{event_id}: negative node count")
            metadata = (row["title"], row["bucket"])
            prior_metadata = process_metadata.setdefault(row["process"], metadata)
            require(prior_metadata == metadata,
                    f"process metadata is unstable for {row['process']}")
            for field in ("allocated_cupti_kernel_ms", "allocated_nvtx_cpu_ms"):
                require(decimal_value(row[field], event_id) >= 0,
                        f"{event_id}: negative process allocation")

        kernel_total = sum(
            (decimal_value(row["allocated_cupti_kernel_ms"], event_id)
             for row in rows),
            DECIMAL_ZERO,
        )
        host_total = sum(
            (decimal_value(row["allocated_nvtx_cpu_ms"], event_id)
             for row in rows),
            DECIMAL_ZERO,
        )
        declared_kernel_total = decimal_value(
            rows[0]["source_total_cupti_kernel_ms"], event_id
        )
        declared_host_total = decimal_value(
            rows[0]["source_total_nvtx_cpu_ms"], event_id
        )
        source_kernel_total = decimal_value(
            denominator_by_group[source_group][
                "hipprof_launch_owned_kernel_sum_ms"
            ]["metric_value_ms"],
            event_id,
        )
        source_host_total = decimal_value(
            denominator_by_group[source_group][
                "hiptx_host_range_duration_ms"
            ]["metric_value_ms"],
            event_id,
        )
        require(abs(kernel_total - declared_kernel_total) <= tolerance,
                f"{event_id}: process kernel rows do not conserve R03 total")
        require(abs(host_total - declared_host_total) <= tolerance,
                f"{event_id}: process host rows do not conserve R03 total")
        require(abs(declared_kernel_total - source_kernel_total) <= tolerance,
                f"{event_id}: R03 kernel total differs from R01")
        require(abs(declared_host_total - source_host_total) <= tolerance,
                f"{event_id}: R03 host total differs from R01")

        layer_type = source_context["workload_type"]
        process_sets_by_layer_type[layer_type].add(tuple(process_ids))
        templates[event_id] = {
            "event_id": event_id,
            "rows": rows,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "phase": rows[0]["phase"],
            "fx_q_len": fx_q_len,
            "fx_kv_len": fx_kv_len,
            "perf_q_len": perf_q_len,
            "perf_kv_len": perf_kv_len,
            "match": match_kind,
            "source_forward_id": source_forward_id,
            "source_occurrence": source_occurrence,
            "source_occurrence_key": rows[0]["source_occurrence_key"],
        }
    for current_layer_type, process_sets in process_sets_by_layer_type.items():
        require(len(process_sets) == 1,
                f"process IDs/order are unstable for {current_layer_type}")

    candidate_classes = {
        (template["phase"], template["layer_type"]) for template in templates.values()
    }
    target_classes = {
        (context["phase"], context["workload_type"])
        for context in context_by_group.values()
    }
    require(target_classes.issubset(candidate_classes),
            "representative templates do not cover every phase/attention path")

    assignments: list[dict[str, str]] = []
    template_use_counts: Counter[str] = Counter()
    sorted_groups = sorted(denominator_by_group)
    max_forward_id = max(forward_ids)
    for group_key in sorted_groups:
        context = context_by_group[group_key]
        forward_id, layer_idx, occurrence = group_key
        target_event_id = f"input{forward_id}_layer{layer_idx}"
        candidates = [
            template
            for template in templates.values()
            if template["phase"] == context["phase"]
            and template["layer_type"] == context["workload_type"]
        ]
        require(candidates, f"no compatible template for {group_key}")

        if target_event_id in templates:
            chosen = templates[target_event_id]
            require(chosen in candidates,
                    f"exact event {target_event_id} is structurally incompatible")
            target_is_representative = True
        else:
            target_is_representative = False

            def candidate_key(template: Mapping[str, Any]) -> tuple[Any, ...]:
                shape_distance = (
                    relative_delta(context["q_len"], template["perf_q_len"])
                    + relative_delta(context["kv_len"], template["perf_kv_len"])
                ) / Decimal(2)
                return (
                    shape_distance,
                    abs(layer_idx - int(template["layer_idx"])),
                    abs(forward_id - int(template["source_forward_id"])),
                    str(template["event_id"]),
                )

            chosen = min(candidates, key=candidate_key)

        q_delta = context["q_len"] - int(chosen["perf_q_len"])
        kv_delta = context["kv_len"] - int(chosen["perf_kv_len"])
        layer_delta = layer_idx - int(chosen["layer_idx"])
        forward_delta = forward_id - int(chosen["source_forward_id"])
        target_shape_exact = q_delta == 0 and kv_delta == 0
        representative_exact = chosen["match"] == "exact"
        template_distance = (
            relative_delta(context["q_len"], int(chosen["perf_q_len"]))
            + relative_delta(context["kv_len"], int(chosen["perf_kv_len"]))
        ) / Decimal(2)

        if target_is_representative and representative_exact:
            type_id = "A1_observed_fx_op_exact"
            status = "observed_representative_structure"
            risk = (
                "representative process/FX evidence is exact; R03 timing rows "
                "remain allocated attribution rather than direct per-process timing"
            )
        elif target_is_representative:
            type_id = "A2_observed_fx_op_nearest_shape_fallback"
            status = "exploratory_nearest_shape_representative"
            risk = (
                "representative FX structure uses an explicit nearest-shape "
                "match to the R01 performance source"
            )
        elif target_shape_exact and representative_exact:
            type_id = "A3_template_scaled_exact_shape"
            status = "estimate_template_scaled"
            risk = (
                "template transfer across layer/occurrence; target metric "
                "remains authoritative"
            )
        elif target_shape_exact:
            type_id = "A4_template_scaled_from_nearest_fx_fallback"
            status = "exploratory_template_from_nearest_fx"
            risk = (
                "template transfer plus representative FX/performance "
                "nearest-shape mismatch"
            )
        elif representative_exact:
            type_id = "A5_template_scaled_nearest_target_shape_fallback"
            status = "exploratory_nearest_target_shape"
            risk = (
                "target q_len/kv_len differs from the exact representative "
                "performance shape"
            )
        else:
            type_id = "A6_template_scaled_dual_shape_fallback"
            status = "exploratory_dual_shape_fallback"
            risk = (
                "target/template performance shape differs and the "
                "representative FX/performance shape also differs"
            )
        type_definition = TYPE_DEFINITIONS[type_id]
        assignment_reason = (
            "exact representative event selected"
            if target_is_representative
            else (
                "deterministic nearest compatible template: same phase and "
                "attention path; lexicographic minimum of normalized "
                "q_len/kv_len distance, layer distance, forward distance, "
                "then event ID"
            )
        )
        assignment = {
            "variant": variant,
            "contract_id": contract_id,
            "contract_sha256": contract_sha,
            "event_id": target_event_id,
            "forward_id": str(forward_id),
            "layer_idx": str(layer_idx),
            "layer": str(layer_idx),
            "occurrence": str(forward_id),
            "source_range_occurrence": str(occurrence),
            "occurrence_key": context["occurrence_key"],
            "layer_type": context["workload_type"],
            "phase": context["phase"],
            "q_len": str(context["q_len"]),
            "past_len": str(context["past_len"]),
            "kv_len": str(context["kv_len"]),
            "shape_class": shape_class(
                context["phase"], context["q_len"], context["kv_len"]
            ),
            "layer_region": layer_region(layer_idx, expected_layer_count),
            "attention_backend": str(nested(r01, "contract", "attention_backend")),
            "pruning_or_selection_state": "none_in_frozen_contract",
            "template_event_id": str(chosen["event_id"]),
            "template_source_occurrence_key": str(chosen["source_occurrence_key"]),
            "template_process_count": str(len(chosen["rows"])),
            "representative_match": str(chosen["match"]),
            "representative_fx_q_len": str(chosen["fx_q_len"]),
            "representative_fx_kv_len": str(chosen["fx_kv_len"]),
            "representative_perf_q_len": str(chosen["perf_q_len"]),
            "representative_perf_kv_len": str(chosen["perf_kv_len"]),
            "target_template_q_len_delta": str(q_delta),
            "target_template_kv_len_delta": str(kv_delta),
            "target_template_layer_delta": str(layer_delta),
            "target_template_forward_delta": str(forward_delta),
            "template_distance": format_decimal(template_distance, 15),
            "representative_fx_perf_q_len_delta": str(
                int(chosen["fx_q_len"]) - int(chosen["perf_q_len"])
            ),
            "representative_fx_perf_kv_len_delta": str(
                int(chosen["fx_kv_len"]) - int(chosen["perf_kv_len"])
            ),
            "assignment_reason": assignment_reason,
            "evidence_class": type_definition["evidence_class"],
            "confidence": type_definition["confidence"],
            "attribution_source": type_definition["attribution_source"],
            "attribution_type_id": type_id,
            "risk": risk,
            "status": status,
        }
        assignments.append(assignment)
        template_use_counts[str(chosen["event_id"])] += 1

    require(len(assignments) == len(denominator_by_group),
            "not every denominator occurrence received one assignment")
    require(len({row["occurrence_key"] for row in assignments}) == len(assignments),
            "assignment occurrence keys are not unique")
    require(all(row["attribution_source"] and row["attribution_type_id"]
                for row in assignments),
            "assignment has an empty attribution source or type")
    assignment_coverage = (
        Decimal(len(assignments)) / Decimal(len(denominator_by_group))
    )
    require(assignment_coverage >= target_coverage,
            "assignment coverage is below the requested target")
    require(
        len(assignments) <= args.maximum_selected_layer_input_count,
        "selected layer-input count exceeds the configured limit",
    )
    selected_process_targets = sum(
        len(templates[row["template_event_id"]]["rows"])
        for row in assignments
    )
    require(
        selected_process_targets <= args.maximum_selected_process_count,
        "selected process target count exceeds the configured limit",
    )

    used_type_ids = sorted({row["attribution_type_id"] for row in assignments})
    type_rows: list[dict[str, str]] = []
    type_counts = Counter(row["attribution_type_id"] for row in assignments)
    for type_id in used_type_ids:
        row = dict(TYPE_DEFINITIONS[type_id])
        row["assignment_count"] = str(type_counts[type_id])
        type_rows.append(row)

    assignment_fieldnames = list(assignments[0])
    type_fieldnames = [
        "attribution_type_id",
        "attribution_source",
        "target_is_representative",
        "representative_match",
        "target_template_shape_relation",
        "evidence_class",
        "confidence",
        "direct_full_layer_timing",
        "description",
        "assignment_count",
    ]
    coverage_fieldnames = [
        "variant",
        "event_id",
        "forward_id",
        "layer_idx",
        "occurrence",
        "source_range_occurrence",
        "occurrence_key",
        "layer_type",
        "phase",
        "q_len",
        "kv_len",
        "template_event_id",
        "representative_match",
        "target_template_q_len_delta",
        "target_template_kv_len_delta",
        "target_template_layer_delta",
        "target_template_forward_delta",
        "template_distance",
        "representative_fx_perf_q_len_delta",
        "representative_fx_perf_kv_len_delta",
        "attribution_source",
        "attribution_type_id",
        "evidence_class",
        "confidence",
        "status",
        "risk",
        "busy_union_fraction_status",
    ]
    coverage_rows: list[dict[str, str]] = []
    for assignment in assignments:
        coverage = {field: assignment[field] for field in coverage_fieldnames
                    if field != "busy_union_fraction_status"}
        coverage["busy_union_fraction_status"] = (
            "diagnostic_kernel_sum_fraction_proxy"
        )
        coverage_rows.append(coverage)

    attribution_rows: list[dict[str, str]] = []
    conservation_sums: dict[
        tuple[str, str, str, str, str], Decimal
    ] = defaultdict(lambda: DECIMAL_ZERO)
    conservation_sources: dict[
        tuple[str, str, str, str, str], Decimal
    ] = {}
    metric_group_counts: Counter[str] = Counter()

    assignment_by_group = {
        (
            integer_value(row["forward_id"], "assignment"),
            integer_value(row["layer_idx"], "assignment"),
            integer_value(row["source_range_occurrence"], "assignment"),
        ): row
        for row in assignments
    }
    for group_key in sorted_groups:
        assignment = assignment_by_group[group_key]
        template = templates[assignment["template_event_id"]]
        template_rows = template["rows"]
        for spec in METRIC_SPECS:
            metric = spec["metric"]
            source_value = decimal_value(
                denominator_by_group[group_key][metric]["metric_value_ms"],
                f"source metric {group_key} {metric}",
            )
            weights = [
                decimal_value(row[spec["template_field"]],
                              f"{template['event_id']} {metric}")
                for row in template_rows
            ]
            weight_sum = sum(weights, DECIMAL_ZERO)
            allocations = allocate_conserving(source_value, weights)
            for template_row, weight, process_ms in zip(
                    template_rows, weights, allocations, strict=True):
                fraction = weight / weight_sum
                metric_evidence_label = assignment["attribution_source"]
                if spec["fraction_status"] == "diagnostic_kernel_sum_fraction_proxy":
                    metric_evidence_label = (
                        f"{metric_evidence_label}"
                        "+fallback_gpu_busy_union_kernel_fraction_proxy"
                    )
                attribution_row = {
                    "variant": variant,
                    "contract_id": contract_id,
                    "contract_sha256": contract_sha,
                    "phase": assignment["phase"],
                    "forward_id": assignment["forward_id"],
                    "layer": assignment["layer"],
                    "layer_idx": assignment["layer_idx"],
                    "occurrence": assignment["occurrence"],
                    "source_range_occurrence": assignment[
                        "source_range_occurrence"
                    ],
                    "occurrence_key": assignment["occurrence_key"],
                    "event_id": assignment["event_id"],
                    "layer_type": assignment["layer_type"],
                    "q_len": assignment["q_len"],
                    "past_len": assignment["past_len"],
                    "kv_len": assignment["kv_len"],
                    "process_id": template_row["process"],
                    "process_title": template_row["title"],
                    "bucket": template_row["bucket"],
                    "nodes": template_row["nodes"],
                    "metric": metric,
                    "metric_role": spec["metric_role"],
                    "metric_semantic": spec["metric_semantic"],
                    "ms": format_decimal(process_ms),
                    "source_layer_metric_ms": format_decimal(source_value),
                    "template_metric_value_ms": format_decimal(weight),
                    "template_metric_total_ms": format_decimal(weight_sum),
                    "normalized_template_fraction": format_decimal(fraction, 15),
                    "template_metric_field": spec["template_field"],
                    "template_metric_semantic": spec["template_semantic"],
                    "metric_fraction_status": spec["fraction_status"],
                    "metric_evidence_label": metric_evidence_label,
                    "template_event_id": assignment["template_event_id"],
                    "representative_match": assignment["representative_match"],
                    "attribution_source": assignment["attribution_source"],
                    "attribution_type_id": assignment["attribution_type_id"],
                    "evidence_class": assignment["evidence_class"],
                    "confidence": assignment["confidence"],
                    "aggregation_key": (
                        f"{variant}|{template_row['process']}|{metric}"
                    ),
                }
                attribution_rows.append(attribution_row)
                conservation_key = (
                    variant,
                    assignment["phase"],
                    assignment["layer"],
                    assignment["occurrence"],
                    metric,
                )
                conservation_sums[conservation_key] += process_ms
                prior_source = conservation_sources.setdefault(
                    conservation_key, source_value
                )
                require(prior_source == source_value,
                        f"source metric is unstable for {conservation_key}")
            metric_group_counts[metric] += 1

    conservation_errors = {
        key: abs(conservation_sums[key] - source)
        for key, source in conservation_sources.items()
    }
    max_conservation_error = max(
        conservation_errors.values(), default=DECIMAL_ZERO
    )
    require(max_conservation_error <= tolerance,
            f"maximum conservation error exceeds tolerance: {max_conservation_error}")
    require(len(conservation_sources) == len(denominator_rows),
            "attribution metric groups do not cover every denominator row")

    attribution_fieldnames = list(attribution_rows[0])
    aggregate_accumulator: dict[
        tuple[str, str, str, str, str], dict[str, Any]
    ] = {}
    for row in attribution_rows:
        for scope, phase in (
            ("phase", row["phase"]),
            ("full_sequence", "all"),
        ):
            key = (
                row["variant"],
                scope,
                phase,
                row["process_id"],
                row["metric"],
            )
            if key not in aggregate_accumulator:
                aggregate_accumulator[key] = {
                    "variant": row["variant"],
                    "scope": scope,
                    "phase": phase,
                    "process_id": row["process_id"],
                    "process_title": row["process_title"],
                    "metric": row["metric"],
                    "metric_role": row["metric_role"],
                    "metric_semantic": row["metric_semantic"],
                    "ms_decimal": DECIMAL_ZERO,
                    "process_row_count": 0,
                    "target_groups": set(),
                }
            accumulator = aggregate_accumulator[key]
            require(accumulator["process_title"] == row["process_title"],
                    f"unstable process title for aggregation key {key}")
            accumulator["ms_decimal"] += decimal_value(row["ms"], "aggregation")
            accumulator["process_row_count"] += 1
            accumulator["target_groups"].add(
                (
                    row["phase"],
                    row["forward_id"],
                    row["layer"],
                    row["occurrence"],
                    row["metric"],
                )
            )

    aggregation_rows: list[dict[str, str]] = []
    for key in sorted(aggregate_accumulator):
        accumulator = aggregate_accumulator[key]
        aggregation_rows.append({
            "variant": accumulator["variant"],
            "scope": accumulator["scope"],
            "phase": accumulator["phase"],
            "process_id": accumulator["process_id"],
            "process_title": accumulator["process_title"],
            "metric": accumulator["metric"],
            "metric_role": accumulator["metric_role"],
            "metric_semantic": accumulator["metric_semantic"],
            "ms": format_decimal(accumulator["ms_decimal"]),
            "process_row_count": str(accumulator["process_row_count"]),
            "target_group_count": str(len(accumulator["target_groups"])),
            "aggregation_key": "|".join(key),
        })
    aggregation_fieldnames = list(aggregation_rows[0])

    assignment_path = output_dir / ASSIGNMENT_NAME
    type_map_path = output_dir / TYPE_MAP_NAME
    attribution_path = output_dir / ATTRIBUTION_NAME
    aggregation_path = output_dir / AGGREGATION_NAME
    coverage_path = output_dir / COVERAGE_NAME
    write_csv(assignment_path, assignment_fieldnames, assignments)
    write_csv(type_map_path, type_fieldnames, type_rows)
    write_csv(attribution_path, attribution_fieldnames, attribution_rows)
    write_csv(aggregation_path, aggregation_fieldnames, aggregation_rows)
    write_csv(coverage_path, coverage_fieldnames, coverage_rows)

    source_totals = {
        metric: sum(
            (
                decimal_value(row["metric_value_ms"], metric)
                for row in denominator_rows
                if row["metric"] == metric
            ),
            DECIMAL_ZERO,
        )
        for metric in METRIC_BY_NAME
    }
    max_error_by_metric = {
        metric: max(
            (
                error for key, error in conservation_errors.items()
                if key[-1] == metric
            ),
            default=DECIMAL_ZERO,
        )
        for metric in METRIC_BY_NAME
    }
    source_counts = Counter(row["attribution_source"] for row in assignments)
    representative_match_counts = Counter(
        row["representative_match"] for row in assignments
    )
    status_counts = Counter(row["status"] for row in assignments)
    phase_counts = Counter(row["phase"] for row in assignments)
    layer_type_counts = Counter(row["layer_type"] for row in assignments)

    generator_path = Path(__file__).resolve()
    run_contract = {
        "schema_version": 1,
        "runtime_goal": args.runtime_goal,
        "run_id": args.run_id,
        "branch": args.branch,
        "workflow05_policy_version": args.workflow05_policy_version,
        "evidence_acquisition_mode": args.evidence_acquisition_mode,
        "variant": variant,
        "display_name": display_name,
        "contract": {
            "contract_id": contract_id,
            "canonical_sha256": contract_sha,
            "r01_source_revision": r01_source_revision,
            "stage_source_revisions": stage_source_revisions,
            "r05_source_state": r05_source_state,
            "source_hash_equality_required": False,
            "prompt_rendered_sha256": nested(
                r01, "contract", "prompt", "rendered_prompt_sha256"
            ),
            "prompt_token_ids_sha256": nested(
                r01, "contract", "prompt", "rendered_prompt_token_ids_sha256"
            ),
            "prompt_token_count": nested(
                r01, "contract", "prompt", "rendered_prompt_token_count"
            ),
            "max_new_tokens": nested(r01, "contract", "max_new_tokens"),
            "dtype": nested(r01, "contract", "dtype"),
            "attention_backend": nested(r01, "contract", "attention_backend"),
            "physical_device_id": nested(
                r01, "contract", "device", "physical_device_id"
            ),
        },
        "upstream_handoffs": {
            "R01": {"path": str(r01_path), "sha256": r01_hash},
            "R02": {"path": str(r02_path), "sha256": r02_hash},
            "R03": {"path": str(r03_path), "sha256": r03_hash},
            "R04": {"path": str(r04_path), "sha256": r04_hash},
        },
        "inputs": input_bindings,
        "strict_process_evidence": {
            "r02_process_db": r02_db_binding,
            "r02_process_inventory": r02_inventory_binding,
            "ownership_rule": nested(
                r02, "trace_binding", "ownership_rule"
            ),
            "strict_owned_kernel_count": nested(
                r03, "component_source", "strict_owned_kernel_count"
            ),
            "multiply_owned_runtime_indices": nested(
                r03, "component_source", "multiply_owned_runtime_indices"
            ),
        },
        "generator": {
            "path": str(generator_path),
            "sha256": sha256_file(generator_path),
        },
        "method": {
            "template_compatibility_key": ["phase", "workload_type"],
            "selection_order": [
                "normalized q_len/kv_len performance-shape distance",
                "absolute layer distance",
                "absolute source-forward distance",
                "representative event ID",
            ],
            "allocation": (
                "normalize the selected representative process metric "
                "distribution to each target layer's own metric"
            ),
            "occurrence_identity": (
                "output occurrence is the 1-based input/forward occurrence; "
                "the R01 within-range occurrence is preserved separately as "
                "source_range_occurrence"
            ),
            "numeric_tolerance_ms": str(tolerance),
            "busy_union_policy": (
                "retain the target busy-union metric identity and use the "
                "representative launch-owned kernel-sum fractions only as an "
                "explicit diagnostic proxy"
            ),
            "selection_policy": {
                "candidate_selection_policy": (
                    "latency_coverage_with_feature_diversity"
                ),
                "feature_diversity_budget_fraction": "0.0",
                "target_cumulative_latency_coverage": str(target_coverage),
                "observed_assignment_coverage": str(assignment_coverage),
                "maximum_selected_layer_input_count": (
                    args.maximum_selected_layer_input_count
                ),
                "selected_layer_input_count": len(assignments),
                "maximum_selected_process_count": (
                    args.maximum_selected_process_count
                ),
                "selected_process_target_count": selected_process_targets,
                "coverage_target_met": assignment_coverage >= target_coverage,
            },
            "ranking_metrics": {
                "selected": "hiptx_host_range_duration_ms",
                "secondary": [
                    "hipprof_launch_owned_kernel_busy_union_ms",
                    "hipprof_launch_owned_kernel_sum_ms",
                ],
            },
        },
        "evidence_boundary": {
            "template_scaled_is_direct_full_layer_trace": False,
            "r03_is_direct_process_timing": False,
            "nearest_shape_is_strict_evidence": False,
            "layer_totals_are_authoritative": True,
            "cpu_gpu_metric_fractions_mixed": False,
            "hardware_replay_used_as_timing_denominator": False,
            "perf_trace_bk_used_as_current_evidence": False,
            "prior_runtime_evidence_used_for_measurement_or_attribution": False,
            "r05_role": "planning evidence for R06",
            "r07_observed_process_timeline_is_authoritative": True,
        },
    }
    run_contract_path = output_dir / RUN_CONTRACT_NAME
    write_json(run_contract_path, run_contract)

    def markdown_count_table(counter: Counter[str], first_column: str) -> str:
        lines = [f"| {first_column} | assignments |", "|---|---:|"]
        for key in sorted(counter):
            lines.append(f"| `{key}` | {counter[key]} |")
        return "\n".join(lines)

    def top_rows(metric: str, limit: int = 12) -> list[dict[str, str]]:
        selected = [
            row for row in aggregation_rows
            if row["scope"] == "full_sequence" and row["metric"] == metric
        ]
        return sorted(
            selected,
            key=lambda row: (-decimal_value(row["ms"], "report"), row["process_id"]),
        )[:limit]

    report_lines = [
        "# Qwen3.5-27B vLLM/PRA Full-layer Process Attribution",
        "",
        "Status: **PASS**",
        "",
        "## Frozen SAME_INPUT Sources",
        "",
        f"- Contract: `{contract_id}` (`{contract_sha}`).",
        "- Stage source revisions: `"
        + json.dumps(stage_source_revisions, sort_keys=True)
        + "`; equality across stages is not required.",
        (
            f"- Complete denominator: `{input_bindings['full_input_layer_performance']['path']}` "
            f"(`{input_bindings['full_input_layer_performance']['sha256']}`)."
        ),
        (
            f"- Layer kernel breakdown: `{input_bindings['layer_kernel_breakdown']['path']}` "
            f"(`{input_bindings['layer_kernel_breakdown']['sha256']}`)."
        ),
        (
            f"- Representative process attribution: "
            f"`{input_bindings['representative_process_attribution']['path']}` "
            f"(`{input_bindings['representative_process_attribution']['sha256']}`)."
        ),
        (
            f"- Representative process report: "
            f"`{input_bindings['representative_process_report']['path']}`."
        ),
        f"- Complete layer report: `{input_bindings['layer_performance_report']['path']}`.",
        (
            "- Compatibility vocabulary is resolved as follows: R03 "
            "`allocated_cupti_kernel_ms` is the hipprof HIPOPS launch-owned "
            "kernel-duration distribution, and R03 `allocated_nvtx_cpu_ms` "
            "is the HIPTX host-range distribution."
        ),
        "",
        "## Assignment Method",
        "",
        (
            "Templates are restricted to the same phase and attention path "
            "(`linear_attention` or `full_attention`). Selection is "
            "deterministic: minimum normalized q_len/kv_len distance, then "
            "layer distance, forward distance, and event ID. Every selected "
            "distribution is normalized to the target layer's own metric."
        ),
        (
            "Output `occurrence` is the 1-based input/forward occurrence "
            "(1–29); R01's within-range `occurrence` remains available as "
            "`source_range_occurrence`."
        ),
        "",
        (
            "`observed_fx_op` marks an exact representative process/FX event, "
            "not direct per-process timing. `template_scaled` is a normalized "
            "full-layer estimate, never a direct full-layer trace. "
            "`fallback_nearest_shape_*` labels expose weaker shape mappings."
        ),
        "",
        "## Coverage and Risk",
        "",
        f"- Denominator occurrences: {len(denominator_by_group)}.",
        f"- Denominator metric rows: {len(denominator_rows)}.",
        f"- Representative events: {len(templates)}.",
        f"- Representative process rows: {len(representative_rows)}.",
        f"- Generated process rows: {len(attribution_rows)}.",
        "",
        markdown_count_table(source_counts, "attribution_source"),
        "",
        markdown_count_table(representative_match_counts, "representative_match"),
        "",
        (
            "All nearest-shape cases remain exploratory. The busy-union "
            "process distribution is separately marked "
            "`diagnostic_kernel_sum_fraction_proxy`; only its target layer "
            "busy-union total is measured."
        ),
        "",
        "## Conservation",
        "",
        "| metric | groups | source total ms | max absolute error ms |",
        "|---|---:|---:|---:|",
    ]
    for spec in METRIC_SPECS:
        metric = spec["metric"]
        report_lines.append(
            f"| `{metric}` | {metric_group_counts[metric]} | "
            f"{format_decimal(source_totals[metric], 6)} | "
            f"{format_decimal(max_error_by_metric[metric], 15)} |"
        )
    report_lines.extend([
        "",
        (
            f"Tolerance: `{tolerance}` ms. All {len(conservation_sources)} "
            "groups pass. CPU and GPU metrics use separate representative "
            "columns and are never combined."
        ),
        "",
        "## Full-sequence Process Attribution",
        "",
        (
            "The following tables rank process estimates independently for "
            "each metric. Values from different metrics must not be added."
        ),
    ])
    for spec in METRIC_SPECS:
        metric = spec["metric"]
        report_lines.extend([
            "",
            f"### {metric}",
            "",
            "| process_id | full-sequence ms | share of metric |",
            "|---|---:|---:|",
        ])
        total = source_totals[metric]
        for row in top_rows(metric):
            value = decimal_value(row["ms"], "report")
            share = DECIMAL_ZERO if total == 0 else value * Decimal(100) / total
            report_lines.append(
                f"| `{row['process_id']}` | {format_decimal(value, 6)} | "
                f"{format_decimal(share, 3)}% |"
            )
    report_lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        (
            "- R01 layer totals remain authoritative. Representative absolute "
            "latencies are not copied to target layers."
        ),
        (
            "- R03 supplies process-stage distributions. Its exact rows are "
            "shape-exact attribution, not direct per-process timing; its "
            "nearest-shape rows are exploratory."
        ),
        (
            "- `observed_fx_op` is limited to exact representative target "
            "events. `template_scaled` and every fallback remain estimates."
        ),
        (
            "- The busy-union target metric is preserved, but process shares "
            "use an explicit kernel-sum fraction proxy because no direct "
            "process busy-union template exists."
        ),
        (
            "- ROCm/DCU hardware replay diagnostics do not alter these timing "
            "denominators. No archive or `perf_trace_bk` artifact is runtime "
            "evidence."
        ),
        "",
    ])
    variant_report_path = output_dir / VARIANT_REPORT_NAME
    variant_report_path.write_text("\n".join(report_lines), encoding="utf-8")

    breakdown_lines = [
        "# SAME_INPUT Full-layer Process Attribution Breakdown",
        "",
        "Status: **PASS**",
        "",
        (
            "Each metric is conserved independently to the complete R01 "
            "denominator. `template_scaled` and fallback rows are estimates, "
            "not direct full-layer process traces."
        ),
        "",
        "| process_id | metric | full-sequence ms | share of source metric |",
        "|---|---|---:|---:|",
    ]
    full_sequence_rows = [
        row for row in aggregation_rows if row["scope"] == "full_sequence"
    ]
    for row in sorted(
        full_sequence_rows,
        key=lambda item: (
            item["metric"],
            -decimal_value(item["ms"], "breakdown"),
            item["process_id"],
        ),
    ):
        value = decimal_value(row["ms"], "breakdown")
        metric_total = source_totals[row["metric"]]
        share = DECIMAL_ZERO if metric_total == 0 else (
            value * Decimal(100) / metric_total
        )
        breakdown_lines.append(
            f"| `{row['process_id']}` | `{row['metric']}` | "
            f"{format_decimal(value, 6)} | {format_decimal(share, 3)}% |"
        )
    breakdown_lines.extend([
        "",
        (
            "The host-range metric uses only host-range fractions; GPU metrics "
            "use only GPU fractions. Busy-union rows carry the explicit "
            "`diagnostic_kernel_sum_fraction_proxy` limitation."
        ),
        "",
    ])
    breakdown_report_path = output_dir / BREAKDOWN_REPORT_NAME
    breakdown_report_path.write_text(
        "\n".join(breakdown_lines), encoding="utf-8"
    )

    generated_output_rows = {
        TYPE_MAP_NAME: len(type_rows),
        ASSIGNMENT_NAME: len(assignments),
        ATTRIBUTION_NAME: len(attribution_rows),
        AGGREGATION_NAME: len(aggregation_rows),
        COVERAGE_NAME: len(coverage_rows),
    }
    generated_outputs: dict[str, dict[str, Any]] = {}
    for name in (*REQUIRED_OUTPUT_NAMES, RUN_CONTRACT_NAME):
        row_count = generated_output_rows.get(name)
        generated_outputs[name] = output_metadata(output_dir / name, row_count)

    generation_audit = {
        "schema_version": 1,
        "runtime_goal": args.runtime_goal,
        "status": "PASS",
        "contract_id": contract_id,
        "contract_sha256": contract_sha,
        "r01_source_revision": r01_source_revision,
        "stage_source_revisions": stage_source_revisions,
        "r05_source_state": r05_source_state,
        "source_hash_equality_required": False,
        "variant": variant,
        "counts": {
            "forward_count": len(forward_ids),
            "layers_per_forward": expected_layer_count,
            "denominator_occurrences": len(denominator_by_group),
            "denominator_metric_rows": len(denominator_rows),
            "representative_events": len(templates),
            "representative_process_rows": len(representative_rows),
            "assignment_rows": len(assignments),
            "attribution_rows": len(attribution_rows),
            "aggregation_rows": len(aggregation_rows),
            "coverage_rows": len(coverage_rows),
            "metric_groups": len(conservation_sources),
            "selected_process_targets": selected_process_targets,
        },
        "assignment_counts": {
            "by_attribution_source": dict(sorted(source_counts.items())),
            "by_attribution_type_id": dict(sorted(type_counts.items())),
            "by_representative_match": dict(
                sorted(representative_match_counts.items())
            ),
            "by_status": dict(sorted(status_counts.items())),
            "by_phase": dict(sorted(phase_counts.items())),
            "by_layer_type": dict(sorted(layer_type_counts.items())),
            "by_template_event_id": dict(sorted(template_use_counts.items())),
        },
        "conservation": {
            "tolerance_ms": str(tolerance),
            "group_count": len(conservation_sources),
            "failure_count": sum(
                error > tolerance for error in conservation_errors.values()
            ),
            "max_absolute_error_ms": format_decimal(
                max_conservation_error, 15
            ),
            "max_absolute_error_by_metric_ms": {
                metric: format_decimal(error, 15)
                for metric, error in max_error_by_metric.items()
            },
            "source_totals_ms": {
                metric: format_decimal(total)
                for metric, total in source_totals.items()
            },
        },
        "checks": {
            "all_upstream_handoffs_complete": True,
            "all_bound_input_hashes_match": True,
            "single_contract": True,
            "source_revisions_recorded_without_equality_gate": True,
            "rocm_hip_metric_semantics_resolved": True,
            "every_denominator_occurrence_has_one_assignment": True,
            "every_assignment_has_source_and_type": True,
            "every_referenced_template_exists": True,
            "process_ids_stable_by_attention_path": True,
            "coverage_reconciles_with_assignments": True,
            "all_metric_groups_conserve": True,
            "cpu_gpu_metric_fractions_mixed": False,
            "reports_distinguish_observed_scaled_and_fallback": True,
            "assignment_coverage_target_met": (
                assignment_coverage >= target_coverage
            ),
            "selected_layer_input_count_within_limit": True,
            "selected_process_target_count_within_limit": True,
            "fresh_current_runtime_only": True,
        },
        "metric_fraction_policy": {
            spec["metric"]: {
                "template_field": spec["template_field"],
                "fraction_status": spec["fraction_status"],
                "metric_semantic": spec["metric_semantic"],
            }
            for spec in METRIC_SPECS
        },
        "inputs": {
            "R01_handoff": {
                "path": str(r01_path),
                "sha256": r01_hash,
            },
            "R02_handoff": {
                "path": str(r02_path),
                "sha256": r02_hash,
            },
            "R03_handoff": {
                "path": str(r03_path),
                "sha256": r03_hash,
            },
            **input_bindings,
        },
        "outputs": generated_outputs,
        "evidence_boundary": run_contract["evidence_boundary"],
    }
    write_json(output_dir / GENERATION_AUDIT_NAME, generation_audit)

    print(json.dumps({
        "status": "PASS",
        "output_dir": str(output_dir),
        "assignment_rows": len(assignments),
        "attribution_rows": len(attribution_rows),
        "metric_groups": len(conservation_sources),
        "max_conservation_error_ms": format_decimal(
            max_conservation_error, 15
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AttributionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
