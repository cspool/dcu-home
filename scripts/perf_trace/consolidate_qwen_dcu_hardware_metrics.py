#!/usr/bin/env python3
"""Consolidate three strict Qwen hipprof PMC replays onto non-replay families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


MODES = ("pmc", "pmc-read", "pmc-write")
UNAVAILABLE = "unavailable"


class ConsolidationError(RuntimeError):
    """Fail-closed replay projection error."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ConsolidationError(f"expected non-empty JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw in ("", None, UNAVAILABLE):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def display(value: float | str | None, digits: int = 6) -> float | str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return UNAVAILABLE
    return round(value, digits)


def require_equal(label: str, *values: Any) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise ConsolidationError(f"{label} mismatch: {values!r}")


def parse_provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def family_key(row: dict[str, str], family_field: str) -> tuple[str, str, str]:
    return (row["event_id"], row["stage"], row[family_field])


def occupancy_upper(row: dict[str, Any]) -> float | None:
    workgroup = number(row, "work_group_size")
    vgpr = number(row, "vgpr_count")
    shared = number(row, "shared_memory_size")
    if not workgroup or not vgpr:
        return None
    workgroup_i = int(workgroup)
    vgpr_i = int(vgpr)
    if workgroup_i <= 0 or vgpr_i <= 0:
        return None
    waves_per_group = math.ceil(workgroup_i / 64)
    by_wave = 40 // waves_per_group
    by_thread = 2560 // workgroup_i
    by_vgpr = 196608 // (vgpr_i * workgroup_i)
    by_shared = (
        10**9
        if shared in (None, 0)
        else 65536 // max(1, int(shared))
    )
    groups = max(0, min(by_wave, by_thread, by_vgpr, by_shared))
    return min(100.0, 100.0 * groups * waves_per_group / 40.0)


def weighted(
    rows: list[dict[str, Any]],
    field: str | None = None,
    derived: Callable[[dict[str, Any]], float | None] | None = None,
) -> tuple[float | None, int]:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = derived(row) if derived else number(row, str(field))
        weight = number(row, "kernel_time")
        if value is None or weight is None or weight <= 0:
            continue
        values.append((float(value), weight))
    if not values:
        return None, 0
    return (
        sum(value * weight for value, weight in values)
        / sum(weight for _, weight in values),
        len(values),
    )


def mean_available(
    rows: list[dict[str, Any]], field: str
) -> tuple[float | None, int]:
    values = [
        value
        for row in rows
        if (value := number(row, field)) is not None
    ]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def exact_metadata_contract(
    mode: str,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    run_contract: dict[str, Any],
) -> None:
    contract = run_contract["contract"]
    require_equal(
        f"{mode} contract_id",
        metadata["contract_id"],
        contract["contract_id"],
    )
    require_equal(
        f"{mode} contract SHA",
        metadata["contract_sha256"],
        contract["canonical_sha256"],
    )
    require_equal(
        f"{mode} source root",
        str(Path(metadata["source_root"]).resolve()),
        str(Path(contract["source_root"]).resolve()),
    )
    require_equal(
        f"{mode} model root",
        str(Path(metadata["model_root"]).resolve()),
        str(Path(contract["resolved_model_root"]).resolve()),
    )
    require_equal(
        f"{mode} served model",
        metadata["served_model_name"],
        contract["served_model_name"],
    )
    require_equal(f"{mode} process profile", metadata["process_profile"], "on")
    require_equal(
        f"{mode} process targets",
        sorted(metadata["process_targets"]),
        sorted(baseline["process_targets"]),
    )
    require_equal(
        f"{mode} expected process ranges",
        int(metadata["expected_process_range_count"]),
        int(baseline["expected_process_range_count"]),
    )
    require_equal(
        f"{mode} max new tokens",
        int(metadata["max_new_tokens"]),
        int(baseline["max_new_tokens"]),
    )
    require_equal(
        f"{mode} warmup",
        int(metadata["warmup_iters"]),
        int(baseline["warmup_iters"]),
    )
    require_equal(
        f"{mode} same input", metadata["same_input"], baseline["same_input"]
    )
    require_equal(
        f"{mode} sampling", metadata["sampling"], baseline["sampling"]
    )
    require_equal(
        f"{mode} measured result",
        metadata["measured_result"],
        baseline["measured_result"],
    )
    require_equal(
        f"{mode} warmup results",
        metadata["warmup_results"],
        baseline["warmup_results"],
    )
    require_equal(
        f"{mode} HIP visibility",
        metadata["runtime"]["HIP_VISIBLE_DEVICES"],
        "1",
    )
    require_equal(
        f"{mode} CUDA visibility",
        metadata["runtime"]["CUDA_VISIBLE_DEVICES"],
        "1",
    )
    require_equal(
        f"{mode} device name", metadata["runtime"]["device_name"], "BW"
    )
    session = metadata.get("profiler_session_control", {})
    require_equal(f"{mode} session enabled", session.get("enabled"), True)
    require_equal(
        f"{mode} session profile kind", session.get("profile_kind"), mode
    )
    require_equal(
        f"{mode} replay latency flag",
        metadata.get("request_synchronized_latency_is_replay_distorted"),
        True,
    )


def probe_device() -> dict[str, Any]:
    os.environ["HIP_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import torch

    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", ""))
    if not arch.startswith("gfx936"):
        raise ConsolidationError(
            f"occupancy constants are not verified for live arch {arch!r}"
        )
    if torch.cuda.device_count() != 1:
        raise ConsolidationError("visibility probe did not expose exactly one DCU")
    smi = json.loads(
        subprocess.check_output(
            [
                "/opt/hyhal/bin/hy-smi",
                "-d",
                "1",
                "--showuniqueid",
                "--showproductname",
                "--showserial",
                "--showuse",
                "--showmemuse",
                "--json",
            ],
            text=True,
        )
    )
    card = smi.get("card1", {})
    require_equal("live device unique ID", card.get("Unique ID"), "TCH19625050401")
    require_equal("live device name", card.get("Card Series"), "BW")
    return {
        "physical_device": 1,
        "logical_device": 0,
        "device_count_under_visibility": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "gcn_arch_name": arch,
        "total_memory_bytes": int(properties.total_memory),
        "unique_id": card.get("Unique ID"),
        "serial_number": card.get("Serial Number"),
        "probe_hcu_use_pct": card.get("HCU use (%)"),
        "probe_hcu_memory_use_pct": card.get("HCU memory use (%)"),
        "occupancy_constants": {
            "wave_size": 64,
            "wave_limit": 40,
            "thread_limit": 2560,
            "vgpr_resource": 196608,
            "shared_memory_bytes": 65536,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--non-replay-family-ledger", type=Path, required=True)
    parser.add_argument("--r02-run-metadata", type=Path, required=True)
    for mode in MODES:
        option = mode.replace("-", "_")
        parser.add_argument(f"--{mode}-analysis", type=Path, required=True)
        parser.add_argument(f"--{mode}-metadata", type=Path, required=True)
        parser.add_argument(f"--{mode}-provenance", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    run_contract_path = args.run_contract.resolve()
    selection_path = args.selection_plan.resolve()
    non_replay_path = args.non_replay_family_ledger.resolve()
    if output_root.name != "R04" or "perf_trace_bk" in output_root.parts:
        raise ConsolidationError("invalid live R04 output root")
    run_contract = load_json(run_contract_path)
    selection = read_csv(selection_path)
    expected_rows = read_csv(non_replay_path)
    baseline = load_json(args.r02_run_metadata.resolve())
    if not selection or not expected_rows:
        raise ConsolidationError("empty selection or non-replay family ledger")
    require_equal("run contract goal", run_contract["runtime_goal"], "R04")
    require_equal(
        "pre-collection selection SHA",
        run_contract["selection_plan"]["sha256"],
        sha256_file(selection_path),
    )
    require_equal(
        "non-replay family ledger SHA",
        run_contract["upstream_bindings"]["non_replay_family_ledger"][
            "sha256"
        ],
        sha256_file(non_replay_path),
    )
    expected_denominator = run_contract["expected_denominator"]
    require_equal(
        "selection target count",
        len(selection),
        int(expected_denominator["process_fragment_targets"]),
    )
    require_equal(
        "projection row count",
        len(expected_rows),
        int(expected_denominator["all_projection_rows"]),
    )

    mode_analysis: dict[str, Path] = {}
    mode_metadata: dict[str, dict[str, Any]] = {}
    mode_provenance: dict[str, dict[str, str]] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}
    mode_trace_summaries: dict[str, dict[str, Any]] = {}
    mode_metrics: dict[str, list[dict[str, str]]] = {}
    mode_replay_families: dict[str, list[dict[str, str]]] = {}
    mode_db_paths: dict[str, str] = {}
    mode_db_hashes: dict[str, str] = {}
    for mode in MODES:
        option = mode.replace("-", "_")
        analysis = Path(getattr(args, f"{option}_analysis")).resolve()
        metadata_path = Path(getattr(args, f"{option}_metadata")).resolve()
        provenance_path = Path(getattr(args, f"{option}_provenance")).resolve()
        for path in (analysis, metadata_path, provenance_path):
            if "perf_trace_bk" in path.parts:
                raise ConsolidationError(f"{mode} input points into archive")
        mode_analysis[mode] = analysis
        mode_metadata[mode] = load_json(metadata_path)
        mode_provenance[mode] = parse_provenance(provenance_path)
        mode_summaries[mode] = load_json(
            analysis / "hardware_metric_summary.json"
        )
        mode_trace_summaries[mode] = load_json(
            analysis / "process_trace_summary.json"
        )
        mode_metrics[mode] = read_csv(
            analysis / "hardware_kernel_metrics.csv"
        )
        mode_replay_families[mode] = read_csv(
            analysis / "process_launch_owned_kernel_family_order.csv"
        )
        require_equal(
            f"{mode} PMC analyzer status", mode_summaries[mode]["status"], "PASS"
        )
        require_equal(
            f"{mode} trace analyzer status",
            mode_trace_summaries[mode]["status"],
            "PASS",
        )
        require_equal(
            f"{mode} summary kind", mode_summaries[mode]["kind"], mode
        )
        require_equal(
            f"{mode} trace mode",
            mode_trace_summaries[mode]["capture_mode"],
            mode,
        )
        require_equal(
            f"{mode} profiler exit",
            mode_provenance[mode].get("exit_code"),
            "0",
        )
        require_equal(
            f"{mode} profiler kind",
            mode_provenance[mode].get("profile_kind"),
            mode,
        )
        require_equal(
            f"{mode} physical device",
            mode_provenance[mode].get("physical_dcu"),
            "1",
        )
        require_equal(
            f"{mode} PMC type",
            mode_provenance[mode].get("pmc_type"),
            "0",
        )
        exact_metadata_contract(
            mode, mode_metadata[mode], baseline, run_contract
        )
        mode_db_paths[mode] = mode_trace_summaries[mode]["source_db"]
        mode_db_hashes[mode] = mode_trace_summaries[mode]["source_db_sha256"]
        require_equal(
            f"{mode} DB file SHA",
            sha256_file(Path(mode_db_paths[mode])),
            mode_db_hashes[mode],
        )
    if len({str(path.parent) for path in mode_analysis.values()}) != 3:
        raise ConsolidationError("the three replay captures are not separate")

    device_probe = probe_device()
    expected_keys: list[tuple[str, str, str]] = []
    expected_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in expected_rows:
        key = family_key(row, "matched_kernel_family")
        if key in expected_by_key:
            raise ConsolidationError(f"duplicate expected family key: {key}")
        expected_keys.append(key)
        expected_by_key[key] = row
    expected_key_set = set(expected_keys)

    grouped_metrics: dict[
        str, dict[tuple[str, str, str], list[dict[str, str]]]
    ] = {}
    replay_family_by_key: dict[
        str, dict[tuple[str, str, str], dict[str, str]]
    ] = {}
    unexpected_by_mode: dict[str, list[tuple[str, str, str]]] = {}
    missing_by_mode: dict[str, list[tuple[str, str, str]]] = {}
    for mode in MODES:
        metric_groups: dict[
            tuple[str, str, str], list[dict[str, str]]
        ] = defaultdict(list)
        for row in mode_metrics[mode]:
            metric_groups[family_key(row, "kernel_family")].append(row)
        grouped_metrics[mode] = metric_groups
        replay_map: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in mode_replay_families[mode]:
            key = family_key(row, "matched_kernel_family")
            if key in replay_map:
                raise ConsolidationError(
                    f"{mode} duplicate replay family key: {key}"
                )
            replay_map[key] = row
        replay_family_by_key[mode] = replay_map
        unexpected_by_mode[mode] = sorted(set(replay_map) - expected_key_set)
        missing_by_mode[mode] = sorted(expected_key_set - set(replay_map))

    hardware_rows: list[dict[str, Any]] = []
    count_drift_rows: list[str] = []
    for key in expected_keys:
        meta = expected_by_key[key]
        expected_no_kernel = key[2] == "no_kernel"
        rows_by_mode = {
            mode: grouped_metrics[mode].get(key, []) for mode in MODES
        }
        replay_rows = {
            mode: replay_family_by_key[mode].get(key) for mode in MODES
        }
        if expected_no_kernel:
            status = (
                "no_kernel"
                if all(
                    replay_rows[mode] is not None
                    and replay_rows[mode]["matched_kernel_family"]
                    == "no_kernel"
                    and not rows_by_mode[mode]
                    for mode in MODES
                )
                else "partial"
            )
        else:
            present = [bool(rows_by_mode[mode]) for mode in MODES]
            status = (
                "complete"
                if all(present)
                else "partial"
                if any(present)
                else "missing"
            )
        timing_instances = int(meta["kernel_family_instance_count"])
        replay_counts = {
            mode: (
                int(replay_rows[mode]["kernel_family_instance_count"])
                if replay_rows[mode] is not None
                else 0
            )
            for mode in MODES
        }
        instance_changed = (
            not expected_no_kernel
            and any(count != timing_instances for count in replay_counts.values())
        )
        target_id = f"{key[0]}:{key[1]}:{key[2]}"
        if instance_changed:
            count_drift_rows.append(target_id)

        base = rows_by_mode["pmc"]
        read = rows_by_mode["pmc-read"]
        write = rows_by_mode["pmc-write"]
        alu, alu_samples = weighted(base, "processed_alu_instructions")
        l2_hit, l2_hit_samples = weighted(base, "l2_cache_hit_rate")
        occupancy, occupancy_samples = weighted(
            base, derived=occupancy_upper
        )
        vgpr, vgpr_samples = weighted(base, "vgpr_count")
        sgpr, sgpr_samples = weighted(base, "sgpr_count")
        shared, shared_samples = weighted(base, "shared_memory_size")
        read_mean, read_samples = mean_available(
            read, "size_of_l2_cache_read"
        )
        write_mean, write_samples = mean_available(
            write, "size_of_l2_cache_write"
        )
        projected_l2_bytes: float | None = None
        projected_l2_gbps: float | None = None
        timing_ms = float(meta["hipprof_kernel_duration_ms"])
        if (
            read_mean is not None
            and write_mean is not None
            and not expected_no_kernel
        ):
            projected_l2_bytes = (
                (read_mean + write_mean) * 1024.0 * timing_instances
            )
            if timing_ms > 0:
                projected_l2_gbps = (
                    projected_l2_bytes / (timing_ms / 1000.0) / 1e9
                )

        stall_fields = {
            "L1_cache_stall": "l1_cache_unit_is_stalled",
            "L2_write_stall": "l2_cache_write_unit_is_stalled",
            "shared_memory_bank_conflict": "shared_memory_bank_conflict",
        }
        stall_values: dict[str, float] = {}
        stall_samples: dict[str, int] = {}
        for label, field in stall_fields.items():
            value, samples = weighted(base, field)
            if value is not None:
                stall_values[label] = value
                stall_samples[label] = samples
        strongest_stall_name = (
            max(stall_values, key=stall_values.get)
            if stall_values
            else UNAVAILABLE
        )
        strongest_stall_value: float | None = (
            stall_values.get(strongest_stall_name)
            if strongest_stall_name != UNAVAILABLE
            else None
        )

        if expected_no_kernel:
            interpretation = (
                "Expected CPU-only/bookkeeping process range; no DCU replay "
                "was launched solely for this row."
            )
        elif status != "complete":
            interpretation = "Required replay source is missing; no bottleneck claim."
        else:
            evidence: list[str] = []
            if key[2] == "TunableOp_MMAC_GEMM" and alu is not None and alu >= 65:
                evidence.append("compute-active MMAC proxy")
            if l2_hit is not None and l2_hit < 50:
                evidence.append("low L2 hit/cache-memory pressure")
            if occupancy is not None and occupancy < 50:
                evidence.append("theoretical occupancy resource limit")
            if strongest_stall_name != UNAVAILABLE:
                evidence.append(f"strongest stall proxy {strongest_stall_name}")
            interpretation = (
                "; ".join(evidence)
                if evidence
                else "Available counters do not identify a dominant bottleneck."
            )

        profiled_names = sorted(
            {
                row["kernel_name"]
                for mode in MODES
                for row in rows_by_mode[mode]
            }
        )
        hardware_rows.append(
            {
                "parent_layer_range": meta["parent_layer_range"],
                "forward_id": meta["forward_id"],
                "layer": meta["layer"],
                "event_id": meta["event_id"],
                "stage": meta["stage"],
                "first_kernel_launch_order_in_parent": meta[
                    "first_kernel_launch_order_in_parent"
                ],
                "process_gpu_order": meta["process_gpu_order"],
                "process_gpu_start_offset_us": meta[
                    "process_gpu_start_offset_us"
                ],
                "process_id": meta["process_id"],
                "process_title": meta["process_title"],
                "fragment_id": meta["fragment_id"],
                "aggregation_key": meta["aggregation_key"],
                "first_kernel_launch_order_in_process": meta[
                    "first_kernel_launch_order_in_process"
                ],
                "matched_kernel_family": meta["matched_kernel_family"],
                "kernel_family_instance_count": timing_instances,
                "workflow02_non_replay_family_duration_ms_context": timing_ms,
                "hipprof_kernel_name_examples": meta[
                    "hipprof_kernel_name_examples"
                ],
                "dcu_pmc_status": status,
                "pmc_kernel_family_instance_count": replay_counts["pmc"],
                "pmc_read_kernel_family_instance_count": replay_counts[
                    "pmc-read"
                ],
                "pmc_write_kernel_family_instance_count": replay_counts[
                    "pmc-write"
                ],
                "pmc_profiled_kernel_names": ";".join(profiled_names),
                "DCU_activity_processed_ALU_pct": display(alu),
                "DCU_activity_sample_count": alu_samples,
                "DCU_matrix_core_utilization_proxy_pct": (
                    display(alu)
                    if key[2] == "TunableOp_MMAC_GEMM"
                    else "not_applicable"
                ),
                "DCU_matrix_proxy_definition": (
                    "processed_ALU activity; diagnostic proxy, not NVIDIA Tensor Core"
                    if key[2] == "TunableOp_MMAC_GEMM"
                    else "not_applicable"
                ),
                "L2_hit_rate_pct": display(l2_hit),
                "L2_hit_rate_sample_count": l2_hit_samples,
                "mean_L2_read_KB_per_replay_instance": display(read_mean),
                "L2_read_sample_count": read_samples,
                "mean_L2_write_KB_per_replay_instance": display(write_mean),
                "L2_write_sample_count": write_samples,
                "projected_L2_bytes": display(projected_l2_bytes, 3),
                "L2_projected_throughput_GBps": display(
                    projected_l2_gbps
                ),
                "DRAM_throughput": UNAVAILABLE,
                "DRAM_unavailable_reason": (
                    "selected hipprof derived PMC set exposes no verified DRAM equivalent"
                ),
                "theoretical_occupancy_upper_bound_pct": display(occupancy),
                "occupancy_sample_count": occupancy_samples,
                "occupancy_interpretation": (
                    "theoretical gfx936 resource upper bound; not achieved occupancy"
                ),
                "weighted_VGPR_count": display(vgpr),
                "VGPR_sample_count": vgpr_samples,
                "weighted_SGPR_count": display(sgpr),
                "SGPR_sample_count": sgpr_samples,
                "weighted_shared_memory_size_bytes": display(shared),
                "shared_memory_sample_count": shared_samples,
                "strongest_available_stall_proxy": strongest_stall_name,
                "strongest_available_stall_proxy_value": display(
                    strongest_stall_value
                ),
                "stall_proxy_sample_count": (
                    stall_samples.get(strongest_stall_name, 0)
                ),
                "hardware_bottleneck_interpretation": interpretation,
                "target_id": target_id,
                "timing_source": "workflow02_non_replay_family_row",
                "hardware_join_key": "event_id+stage+matched_kernel_family",
                "pmc_metric_weighting": (
                    "replay kernel_time weighted; diagnostic only"
                ),
                "pmc_replay_timing_used_as_latency": False,
                "replay_instance_count_changed": instance_changed,
            }
        )

    unexpected_execution_paths = {
        mode: [":".join(key) for key in unexpected_by_mode[mode]]
        for mode in MODES
        if unexpected_by_mode[mode]
    }
    missing_execution_paths = {
        mode: [":".join(key) for key in missing_by_mode[mode]]
        for mode in MODES
        if missing_by_mode[mode]
    }
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
    complete_rows = [
        row for row in kernel_rows if row["dcu_pmc_status"] == "complete"
    ]
    partial_rows = [
        row for row in kernel_rows if row["dcu_pmc_status"] == "partial"
    ]
    missing_rows = [
        row for row in kernel_rows if row["dcu_pmc_status"] == "missing"
    ]
    invalid_no_kernel = [
        row for row in no_kernel_rows if row["dcu_pmc_status"] != "no_kernel"
    ]
    required_targets = [
        row
        for row in selection
        if row["collection_required"].strip().lower() == "true"
    ]
    expected_no_kernel_targets = [
        row
        for row in selection
        if row["expected_no_kernel"].strip().lower() == "true"
    ]
    parent_layers = list(
        dict.fromkeys(row["parent_layer_range"] for row in selection)
    )
    per_mode_join = {
        mode: {
            "joined_expected_kernel_family_rows": sum(
                bool(grouped_metrics[mode].get(key))
                for key in expected_keys
                if key[2] != "no_kernel"
            ),
            "expected_kernel_family_rows": len(kernel_rows),
            "coverage_pct": (
                100.0
                * sum(
                    bool(grouped_metrics[mode].get(key))
                    for key in expected_keys
                    if key[2] != "no_kernel"
                )
                / len(kernel_rows)
                if kernel_rows
                else 0.0
            ),
            "name_order_match_rate": mode_summaries[mode][
                "name_order_match_rate"
            ],
            "pmc_block_count": mode_summaries[mode]["pmc_block_count"],
            "exact_name_order_matches": mode_summaries[mode][
                "exact_name_order_matches"
            ],
            "unmatched_pmc_block_count": mode_summaries[mode][
                "unmatched_pmc_block_count"
            ],
            "unmatched_selected_block_count": mode_summaries[mode][
                "unmatched_selected_block_count"
            ],
            "unmatched_selected_trace_kernel_count": mode_summaries[mode][
                "unmatched_selected_trace_kernel_count"
            ],
            "ambiguous_pair_count": mode_summaries[mode][
                "ambiguous_pair_count"
            ],
            "strict_owned_metric_rows": mode_summaries[mode][
                "strict_owned_metric_rows"
            ],
            "replay_db": mode_db_paths[mode],
            "replay_db_sha256": mode_db_hashes[mode],
        }
        for mode in MODES
    }
    failure_reasons: list[str] = []
    if len(parent_layers) != int(
        expected_denominator["representative_parent_layers"]
    ):
        failure_reasons.append("representative parent coverage mismatch")
    if len(required_targets) != int(
        expected_denominator["launch_owning_capture_targets"]
    ):
        failure_reasons.append("launch-owning target denominator mismatch")
    if len(expected_no_kernel_targets) != int(
        expected_denominator["no_kernel_process_targets"]
    ):
        failure_reasons.append("no-kernel target denominator mismatch")
    if len(complete_rows) != len(kernel_rows) or partial_rows or missing_rows:
        failure_reasons.append("not every required family row is complete")
    if invalid_no_kernel:
        failure_reasons.append("expected no-kernel row is not no_kernel")
    if unexpected_execution_paths or missing_execution_paths:
        failure_reasons.append("replay family/execution path differs from non-replay")
    for mode in MODES:
        summary = mode_summaries[mode]
        if float(summary["name_order_match_rate"]) < 0.99:
            failure_reasons.append(f"{mode} name/order match below 0.99")
        if (
            int(summary["unmatched_selected_block_count"]) != 0
            or int(summary["unmatched_selected_trace_kernel_count"]) != 0
            or int(summary["ambiguous_pair_count"]) != 0
        ):
            failure_reasons.append(f"{mode} selected-target PMC ambiguity")
    status = "FAIL" if failure_reasons else "PASS"

    replay_kernel_rows: list[dict[str, Any]] = []
    for mode in MODES:
        replay_kernel_rows.extend(
            {"replay_source": mode, **row} for row in mode_metrics[mode]
        )
    write_csv(
        output_root / "hardware_replay_kernel_metrics.csv",
        replay_kernel_rows,
    )
    write_csv(
        output_root / "hardware_metrics_by_kernel_family.csv", hardware_rows
    )

    process_groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in hardware_rows:
        process_groups[(row["event_id"], row["stage"])].append(row)
    process_rows: list[dict[str, Any]] = []
    for (event_id, stage), rows in process_groups.items():
        statuses = {str(row["dcu_pmc_status"]) for row in rows}
        process_rows.append(
            {
                "event_id": event_id,
                "stage": stage,
                "process_id": rows[0]["process_id"],
                "process_title": rows[0]["process_title"],
                "fragment_id": rows[0]["fragment_id"],
                "process_gpu_order": rows[0]["process_gpu_order"],
                "kernel_family_rows": len(rows),
                "complete_family_rows": sum(
                    row["dcu_pmc_status"] in {"complete", "no_kernel"}
                    for row in rows
                ),
                "dcu_pmc_process_status": (
                    "no_kernel"
                    if statuses == {"no_kernel"}
                    else "complete"
                    if statuses <= {"complete"}
                    else "incomplete"
                ),
                "family_target_ids": ";".join(
                    str(row["target_id"]) for row in rows
                ),
                "hardware_source": (
                    "hipprof PMC family projection; no replay latency"
                ),
            }
        )
    process_rows.sort(
        key=lambda row: (
            int(next(
                item["forward_id"]
                for item in selection
                if item["event_id"] == row["event_id"]
                and item["stage"] == row["stage"]
            )),
            int(next(
                item["layer"]
                for item in selection
                if item["event_id"] == row["event_id"]
                and item["stage"] == row["stage"]
            )),
            int(row["process_gpu_order"]),
        )
    )
    write_csv(output_root / "hardware_metrics.csv", process_rows)

    coverage = {
        "schema_version": 1,
        "status": status,
        "failure_reasons": failure_reasons,
        "run_id": run_contract["run"]["run_id"],
        "branch": run_contract["run"]["branch"],
        "contract_id": run_contract["contract"]["contract_id"],
        "contract_sha256": run_contract["contract"]["canonical_sha256"],
        "expected_representative_parent_layers": int(
            expected_denominator["representative_parent_layers"]
        ),
        "observed_representative_parent_layers": len(parent_layers),
        "representative_parent_layer_ranges": parent_layers,
        "expected_process_fragment_targets": int(
            expected_denominator["process_fragment_targets"]
        ),
        "observed_process_fragment_targets": len(selection),
        "expected_launch_owning_capture_targets": int(
            expected_denominator["launch_owning_capture_targets"]
        ),
        "observed_launch_owning_capture_targets": len(required_targets),
        "expected_no_kernel_process_targets": int(
            expected_denominator["no_kernel_process_targets"]
        ),
        "observed_no_kernel_process_targets": len(
            expected_no_kernel_targets
        ),
        "expected_kernel_family_rows": len(kernel_rows),
        "expected_no_kernel_rows": len(no_kernel_rows),
        "all_projection_rows": len(hardware_rows),
        "complete_rows": len(complete_rows),
        "partial_rows": len(partial_rows),
        "missing_rows": len(missing_rows),
        "no_kernel_rows": sum(
            row["dcu_pmc_status"] == "no_kernel" for row in no_kernel_rows
        ),
        "family_join_coverage_pct": (
            100.0 * len(complete_rows) / len(kernel_rows)
            if kernel_rows
            else 0.0
        ),
        "per_mode_join_coverage": per_mode_join,
        "replay_instance_count_changed_rows": len(count_drift_rows),
        "replay_instance_count_changed_target_ids": count_drift_rows,
        "unexpected_replay_families_or_execution_paths": (
            unexpected_execution_paths
        ),
        "missing_replay_families_or_execution_paths": missing_execution_paths,
        "unmatched_pmc_block_counts": {
            mode: mode_summaries[mode]["unmatched_pmc_block_count"]
            for mode in MODES
        },
        "unmatched_selected_target_block_counts": {
            mode: mode_summaries[mode]["unmatched_selected_block_count"]
            for mode in MODES
        },
        "ambiguous_pmc_block_counts": {
            mode: mode_summaries[mode]["ambiguous_pair_count"]
            for mode in MODES
        },
        "device": device_probe,
        "timing_source": "workflow02_non_replay_family_row",
        "hardware_join_key": "event_id+stage+matched_kernel_family",
        "pmc_replay_timing_used_as_latency": False,
        "pmc_is_latency_evidence": False,
        "replay_kernel_id_cross_run_join_used": False,
        "device_timestamp_overlap_attribution_used": False,
        "archive_used_as_current_evidence": False,
    }
    coverage_path = output_root / "hardware_coverage.json"
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_columns = [
        "parent_layer_range",
        "forward_id",
        "layer",
        "first_kernel_launch_order_in_parent",
        "process_gpu_order",
        "process_gpu_start_offset_us",
        "process_id",
        "process_title",
        "fragment_id",
        "matched_kernel_family",
        "kernel_family_instance_count",
        "dcu_pmc_status",
        "pmc_kernel_family_instance_count",
        "pmc_read_kernel_family_instance_count",
        "pmc_write_kernel_family_instance_count",
        "DCU_activity_processed_ALU_pct",
        "DCU_matrix_core_utilization_proxy_pct",
        "L2_hit_rate_pct",
        "mean_L2_read_KB_per_replay_instance",
        "mean_L2_write_KB_per_replay_instance",
        "L2_projected_throughput_GBps",
        "DRAM_throughput",
        "theoretical_occupancy_upper_bound_pct",
        "weighted_VGPR_count",
        "weighted_shared_memory_size_bytes",
        "strongest_available_stall_proxy",
        "hardware_bottleneck_interpretation",
        "target_id",
    ]

    def cell(value: Any, limit: int = 100) -> str:
        text = str(value).replace("|", "/").replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    lines = [
        "# SAME_INPUT PRA Qwen3.5 Process-wise DCU Hardware Report",
        "",
        f"Status: **{status}**",
        "",
        "- Device: physical DCU 1, live-verified gfx936.",
        "- Main row unit: current Workflow-02 launch-owned `matched_kernel_family`.",
        "- `timing_source=workflow02_non_replay_family_row`.",
        "- `hardware_join_key=event_id+stage+matched_kernel_family`.",
        "- `pmc_replay_timing_used_as_latency=false`.",
        (
            "- PMC `kernel_time` is used only to weight diagnostics; replay "
            "duration is excluded from the primary table and all latency claims."
        ),
        (
            "- Occupancy is a theoretical gfx936 resource upper bound, not "
            "achieved occupancy."
        ),
        (
            "- Matrix-core utilization is a DCU MMAC activity proxy only, not "
            "an NVIDIA Tensor Core metric."
        ),
        (
            "- DRAM throughput is `unavailable`; L2 traffic is not promoted to "
            "a DRAM inference."
        ),
        (
            f"- Coverage: {len(complete_rows)}/{len(kernel_rows)} required "
            f"family rows complete; {len(no_kernel_rows)} explicit no-kernel rows."
        ),
        "",
        "## Kernel-family Hardware Attributes by Representative Layer",
        "",
        "| " + " | ".join(report_columns) + " |",
        "|" + "|".join("---" for _ in report_columns) + "|",
    ]
    for row in hardware_rows:
        lines.append(
            "| "
            + " | ".join(cell(row.get(column, "")) for column in report_columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            (
                "The table is a hardware-diagnostic projection over the frozen "
                "non-replay family/order ledger. It does not replace process, "
                "layer, or end-to-end latency and does not change downstream "
                "segmented-attribution denominators."
            ),
            "",
        ]
    )
    report_text = "\n".join(lines)
    (output_root / "DCU_HARDWARE_METRICS_REPORT.md").write_text(
        report_text, encoding="utf-8"
    )
    (
        output_root
        / "SAME_INPUT_PRA_QWEN35_FULL_EAGER_PROCESS_WISE_DCU_REPORT.md"
    ).write_text(report_text, encoding="utf-8")

    pre_collection_path = (
        output_root / "dcu_process_selection_plan.pre_collection.csv"
    )
    if not pre_collection_path.exists():
        pre_collection_path.write_bytes(selection_path.read_bytes())
    pre_collection_sha = sha256_file(pre_collection_path)
    require_equal(
        "preserved pre-collection plan SHA",
        pre_collection_sha,
        run_contract["selection_plan"]["sha256"],
    )
    for row in selection:
        if row["expected_no_kernel"].strip().lower() == "true":
            row["collection_status"] = "no_kernel"
        else:
            target_keys = [
                key
                for key in expected_keys
                if key[0] == row["event_id"] and key[1] == row["stage"]
            ]
            target_statuses = {
                next(
                    item["dcu_pmc_status"]
                    for item in hardware_rows
                    if item["target_id"] == ":".join(key)
                )
                for key in target_keys
            }
            row["collection_status"] = (
                "complete" if target_statuses == {"complete"} else "incomplete"
            )
        row["pmc_collection_status"] = mode_summaries["pmc"]["status"]
        row["pmc_read_collection_status"] = mode_summaries["pmc-read"]["status"]
        row["pmc_write_collection_status"] = mode_summaries["pmc-write"]["status"]
    write_csv(selection_path, selection)
    run_contract["status"] = (
        "hardware_projection_pass" if status == "PASS" else "hardware_projection_fail"
    )
    run_contract["selection_plan"]["pre_collection_path"] = str(
        pre_collection_path
    )
    run_contract["selection_plan"]["pre_collection_sha256"] = pre_collection_sha
    run_contract["selection_plan"]["path"] = str(selection_path)
    run_contract["selection_plan"]["sha256"] = sha256_file(selection_path)
    run_contract["selection_plan"]["collection_status"] = (
        "complete" if status == "PASS" else "incomplete"
    )
    run_contract["replay_bindings"] = {
        mode: {
            "analysis_dir": str(mode_analysis[mode]),
            "metadata_sha256": sha256_file(
                Path(getattr(args, f"{mode.replace('-', '_')}_metadata"))
            ),
            "provenance_sha256": sha256_file(
                Path(getattr(args, f"{mode.replace('-', '_')}_provenance"))
            ),
            "raw_db": mode_db_paths[mode],
            "raw_db_sha256": mode_db_hashes[mode],
            "trace_summary_sha256": sha256_file(
                mode_analysis[mode] / "process_trace_summary.json"
            ),
            "pmc_summary_sha256": sha256_file(
                mode_analysis[mode] / "hardware_metric_summary.json"
            ),
        }
        for mode in MODES
    }
    run_contract["hardware_coverage"] = {
        "path": str(coverage_path),
        "sha256": sha256_file(coverage_path),
        "status": status,
    }
    run_contract_path.write_text(
        json.dumps(run_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(coverage, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
