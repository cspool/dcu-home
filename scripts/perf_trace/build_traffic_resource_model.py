#!/usr/bin/env python3
"""Build an evidence-graded tensor-traffic and DCU resource model.

Tensor byte counts are FX-visible logical tensor bytes.  They are not HBM
traffic: cache reuse, fusion, aliasing, mutation and opaque custom-op internals
remain explicit.  Kernel resource rows preserve current PMC/resource evidence
and expose the gfx936 upper-bound formula without relabeling it achieved
occupancy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8": 1,
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "float": 4,
    "int32": 4,
    "uint32": 4,
    "float64": 8,
    "double": 8,
    "int64": 8,
    "uint64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def number(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name, "").strip()
        if not value or value.lower().startswith("unavailable"):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def tensor_bytes(node: dict[str, str]) -> tuple[int | None, str | None]:
    try:
        shape = json.loads(node.get("shape_json", "null"))
    except json.JSONDecodeError:
        return None, "invalid_shape_json"
    if not isinstance(shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        return None, "shape_unavailable_or_symbolic"
    dtype = node.get("dtype", "").replace("torch.", "").lower()
    item_size = DTYPE_BYTES.get(dtype)
    if item_size is None:
        return None, f"dtype_size_unavailable:{dtype or 'empty'}"
    elements = math.prod(shape)
    return elements * item_size, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Workflow05 traffic/resource evidence."
    )
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--dependency-adapter", type=Path, required=True)
    parser.add_argument("--hardware-metrics", type=Path, required=True)
    parser.add_argument("--device-capabilities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_device(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise RuntimeError("device capabilities must be an object")
    if payload.get("architecture") != "gfx936":
        raise RuntimeError("device capabilities must prove architecture=gfx936")
    if payload.get("physical_device_id") != 1:
        raise RuntimeError("device capabilities must bind physical DCU 1")
    required = (
        "wave_size",
        "wave_limit",
        "thread_limit",
        "vgpr_resource",
        "shared_memory_bytes",
    )
    limits: dict[str, int] = {}
    for name in required:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"device capability {name} is invalid")
        limits[name] = value
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("device capabilities must retain source provenance")
    return limits


def occupancy_upper_bound(
    workgroup: float | None,
    vgpr: float | None,
    shared: float | None,
    limits: dict[str, int],
) -> float | None:
    if workgroup is None or vgpr is None or shared is None:
        return None
    workgroup_i = int(round(workgroup))
    vgpr_i = int(round(vgpr))
    shared_i = int(round(shared))
    if workgroup_i <= 0 or vgpr_i <= 0 or shared_i < 0:
        return None
    waves_per_group = math.ceil(workgroup_i / limits["wave_size"])
    candidates = [
        limits["wave_limit"] // waves_per_group,
        limits["thread_limit"] // workgroup_i,
        limits["vgpr_resource"] // (vgpr_i * workgroup_i),
    ]
    if shared_i:
        candidates.append(limits["shared_memory_bytes"] // shared_i)
    groups = min(candidates)
    return min(
        100.0,
        100.0 * groups * waves_per_group / limits["wave_limit"],
    )


def main() -> int:
    args = parse_args()
    for path in (
        args.lineage_manifest,
        args.dependency_adapter,
        args.hardware_metrics,
        args.device_capabilities,
    ):
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lineage = load_json(args.lineage_manifest)
    if (
        not isinstance(lineage, dict)
        or lineage.get("status") != "PASS"
        or not isinstance(lineage.get("lineage_id"), str)
        or lineage.get("source_hash_equality_required") is not False
    ):
        raise RuntimeError("fresh-run lineage manifest is invalid")
    adapter = load_json(args.dependency_adapter)
    if adapter.get("schema_version") != 1 or adapter.get("status") != "complete":
        raise RuntimeError("dependency adapter is not complete schema version 1")
    if (
        adapter.get("adapter_type")
        != "fresh_run_fixed_input_fx_process_dependency"
        or adapter.get("lineage_id") != lineage.get("lineage_id")
    ):
        raise RuntimeError("dependency adapter does not match the fresh-run lineage")
    output_records = adapter.get("outputs", {})
    node_record = output_records.get("nodes", {})
    edge_record = output_records.get("edges", {})
    node_path = Path(node_record.get("path", "")).expanduser().resolve()
    edge_path = Path(edge_record.get("path", "")).expanduser().resolve()
    for path, record in ((node_path, node_record), (edge_path, edge_record)):
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"dependency adapter output changed: {path}")
    limits = validate_device(load_json(args.device_capabilities))

    nodes = load_csv(node_path)
    edges = load_csv(edge_path)
    node_by_identity = {
        (row.get("event_id", ""), row.get("template_node_name", "")): row
        for row in nodes
    }
    stage_nodes: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in nodes:
        stage_nodes[(row.get("event_id", ""), row.get("process_stage", ""))].append(row)

    reads: dict[tuple[str, str], set[str]] = defaultdict(set)
    writes: dict[tuple[str, str], set[str]] = defaultdict(set)
    unknown_counts: dict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        event = edge.get("event_id", "")
        source_stage = edge.get("source_stage", "")
        target_stage = edge.get("target_stage", "")
        if edge.get("edge_type") == "data" and parse_bool(edge.get("verified")):
            source_name = edge.get("source_node", "")
            reads[(event, target_stage)].add(source_name)
            writes[(event, source_stage)].add(source_name)
        else:
            affected = target_stage or source_stage
            unknown_counts[(event, affected)] += 1

    traffic_rows: list[dict[str, Any]] = []
    for event_stage in sorted(stage_nodes):
        event, stage = event_stage
        visible_bytes = 0
        unavailable_node_count = 0
        unavailable_reasons: set[str] = set()
        for node in stage_nodes[event_stage]:
            value, reason = tensor_bytes(node)
            if value is None:
                unavailable_node_count += 1
                if reason:
                    unavailable_reasons.add(reason)
            else:
                visible_bytes += value

        def sum_nodes(names: set[str]) -> tuple[int, int, list[str]]:
            total = 0
            missing = 0
            reasons: set[str] = set()
            for name in sorted(names):
                node = node_by_identity.get((event, name))
                if node is None:
                    missing += 1
                    reasons.add(f"node_missing:{name}")
                    continue
                value, reason = tensor_bytes(node)
                if value is None:
                    missing += 1
                    if reason:
                        reasons.add(reason)
                else:
                    total += value
            return total, missing, sorted(reasons)

        read_bytes, missing_reads, read_reasons = sum_nodes(reads[event_stage])
        write_bytes, missing_writes, write_reasons = sum_nodes(writes[event_stage])
        unknown = unknown_counts[event_stage]
        complete_visible = not (
            unavailable_node_count or missing_reads or missing_writes or unknown
        )
        reasons = sorted(
            unavailable_reasons.union(read_reasons).union(write_reasons)
        )
        if unknown:
            reasons.append("opaque_or_unresolved_dependency")
        traffic_rows.append(
            {
                "contract_id": adapter.get("contract_id"),
                "event_id": event,
                "stage": stage,
                "fx_visible_read_bytes": read_bytes,
                "fx_visible_write_bytes": write_bytes,
                "fx_visible_total_io_bytes": read_bytes + write_bytes,
                "fx_visible_stage_tensor_bytes": visible_bytes,
                "read_tensor_count": len(reads[event_stage]),
                "write_tensor_count": len(writes[event_stage]),
                "fx_node_count": len(stage_nodes[event_stage]),
                "unavailable_node_count": unavailable_node_count,
                "unknown_dependency_count": unknown,
                "traffic_completeness": (
                    "complete_fx_visible" if complete_visible else "lower_bound"
                ),
                "traffic_semantics": (
                    "logical_fixed_input_fx_visible_tensor_bytes_not_hbm_traffic"
                ),
                "hbm_or_dram_bytes": "unavailable",
                "evidence_class": (
                    "inferred_fixed_input_fx_current_source"
                    if complete_visible
                    else "inferred_lower_bound"
                ),
                "unavailable_reasons_json": json.dumps(reasons, separators=(",", ":")),
            }
        )

    hardware = load_csv(args.hardware_metrics)
    resource_rows: list[dict[str, Any]] = []
    for row in hardware:
        workgroup = number(row, "weighted_work_group_size", "work_group_size")
        vgpr = number(row, "weighted_VGPR_count", "VGPR_count", "vgpr_count")
        sgpr = number(row, "weighted_SGPR_count", "SGPR_count", "sgpr_count")
        shared = number(
            row,
            "weighted_shared_memory_size_bytes",
            "shared_memory_size_bytes",
            "shared_memory_size",
            "lds_size_bytes",
        )
        recorded_occupancy = number(
            row,
            "theoretical_occupancy_upper_bound_pct",
            "occupancy_upper_bound_pct",
        )
        occupancy_sample_count = number(row, "occupancy_sample_count")
        recomputed_occupancy = occupancy_upper_bound(
            workgroup, vgpr, shared, limits
        )
        if recorded_occupancy is not None and not 0.0 <= recorded_occupancy <= 100.0:
            raise RuntimeError(
                "recorded occupancy is outside [0, 100]: "
                f"{row.get('event_id')} {row.get('stage')} "
                f"{row.get('matched_kernel_family')}"
            )
        if recorded_occupancy is not None and recomputed_occupancy is not None:
            # A family row may aggregate several dispatch shapes.  Its recorded
            # value is the duration-weighted mean of the per-dispatch theoretical
            # upper bounds.  Reapplying the nonlinear occupancy formula to the
            # independently weighted resource descriptors is not equivalent.
            # Exact recomputation is therefore a valid cross-check only for a
            # single contributing dispatch; multi-dispatch families retain the
            # already validated per-dispatch aggregate from consolidation.
            if (
                occupancy_sample_count is not None
                and occupancy_sample_count <= 1.0
                and abs(recorded_occupancy - recomputed_occupancy) > 1e-6
            ):
                raise RuntimeError(
                    "recorded occupancy differs from the current device formula: "
                    f"{row.get('event_id')} {row.get('stage')} "
                    f"{row.get('matched_kernel_family')}"
                )
        selected_occupancy = (
            recorded_occupancy
            if recorded_occupancy is not None
            else recomputed_occupancy
        )
        resource_complete = all(
            value is not None
            for value in (workgroup, vgpr, shared, selected_occupancy)
        )
        resource_rows.append(
            {
                "contract_id": adapter.get("contract_id"),
                "event_id": row.get("event_id", ""),
                "stage": row.get("stage", ""),
                "matched_kernel_family": row.get("matched_kernel_family", ""),
                "hardware_join_key": "event_id+stage+matched_kernel_family",
                "work_group_size": workgroup if workgroup is not None else "unavailable",
                "vgpr_count": vgpr if vgpr is not None else "unavailable",
                "sgpr_count": sgpr if sgpr is not None else "unavailable",
                "shared_memory_size_bytes": (
                    shared if shared is not None else "unavailable"
                ),
                "theoretical_occupancy_upper_bound_pct": (
                    selected_occupancy
                    if selected_occupancy is not None
                    else "unavailable"
                ),
                "occupancy_semantics": (
                    "duration_weighted_per_dispatch_gfx936_resource_upper_bound_"
                    "not_achieved_occupancy"
                    if recorded_occupancy is not None
                    else "single_resource_tuple_gfx936_resource_upper_bound_"
                    "not_achieved_occupancy"
                ),
                "occupancy_aggregation_source": (
                    "consolidated_duration_weighted_per_dispatch_upper_bounds"
                    if recorded_occupancy is not None
                    else "recomputed_from_current_resource_tuple"
                ),
                "resource_evidence_complete": resource_complete,
                "hardware_evidence_class": row.get(
                    "hardware_evidence_class", "unavailable"
                ),
                "row_reuse_or_path_state": row.get(
                    "row_reuse_or_path_state", "unavailable"
                ),
                "timing_source": row.get("timing_source", ""),
                "pmc_replay_timing_used_as_latency": row.get(
                    "pmc_replay_timing_used_as_latency", ""
                ),
                "source_row_sha256": hashlib.sha256(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "unavailable_reason": (
                    ""
                    if resource_complete
                    else "one or more current family resource fields unavailable"
                ),
            }
        )

    traffic_path = args.output_dir / "process_traffic_model.csv"
    resource_path = args.output_dir / "kernel_family_resource_model.csv"
    model_path = args.output_dir / "traffic_resource_model.json"
    write_csv(
        traffic_path,
        [
            "contract_id",
            "event_id",
            "stage",
            "fx_visible_read_bytes",
            "fx_visible_write_bytes",
            "fx_visible_total_io_bytes",
            "fx_visible_stage_tensor_bytes",
            "read_tensor_count",
            "write_tensor_count",
            "fx_node_count",
            "unavailable_node_count",
            "unknown_dependency_count",
            "traffic_completeness",
            "traffic_semantics",
            "hbm_or_dram_bytes",
            "evidence_class",
            "unavailable_reasons_json",
        ],
        traffic_rows,
    )
    write_csv(
        resource_path,
        [
            "contract_id",
            "event_id",
            "stage",
            "matched_kernel_family",
            "hardware_join_key",
            "work_group_size",
            "vgpr_count",
            "sgpr_count",
            "shared_memory_size_bytes",
            "theoretical_occupancy_upper_bound_pct",
            "occupancy_semantics",
            "occupancy_aggregation_source",
            "resource_evidence_complete",
            "hardware_evidence_class",
            "row_reuse_or_path_state",
            "timing_source",
            "pmc_replay_timing_used_as_latency",
            "source_row_sha256",
            "unavailable_reason",
        ],
        resource_rows,
    )

    model = {
        "schema_version": 1,
        "status": "complete",
        "model_type": "fresh_run_fx_visible_traffic_and_dcu_family_resource",
        "lineage_id": lineage.get("lineage_id"),
        "contract_id": adapter.get("contract_id"),
        "contract_sha256": adapter.get("contract_sha256"),
        "stage_source_revision": adapter.get("stage_source_revision"),
        "source_revision_policy": adapter.get("source_revision_policy"),
        "traffic_boundary": {
            "quantity": "logical fixed-input FX-visible tensor bytes",
            "hbm_or_dram_traffic_claimed": False,
            "cache_or_fusion_reuse_modeled": False,
            "opaque_custom_op_internal_traffic_modeled": False,
            "lower_bound_retained_when_incomplete": True,
        },
        "resource_boundary": {
            "quantity": "per-family replay-projected resource requirement",
            "occupancy": (
                "duration-weighted per-dispatch theoretical gfx936 upper bound; "
                "weighted aggregate resource descriptors are not inputs to a "
                "second nonlinear occupancy recomputation"
            ),
            "achieved_occupancy_claimed": False,
            "coexistence_formula": (
                "waves_per_group=ceil(work_group_size/wave_size); "
                "groups=min(wave_limit//waves_per_group, "
                "thread_limit//work_group_size, "
                "vgpr_resource//(vgpr_count*work_group_size), "
                "shared_memory_bytes//shared_memory_size when nonzero)"
            ),
        },
        "device_limits": limits,
        "coverage": {
            "process_stage_count": len(traffic_rows),
            "complete_fx_visible_traffic_count": sum(
                row["traffic_completeness"] == "complete_fx_visible"
                for row in traffic_rows
            ),
            "lower_bound_traffic_count": sum(
                row["traffic_completeness"] == "lower_bound"
                for row in traffic_rows
            ),
            "kernel_family_count": len(resource_rows),
            "resource_complete_family_count": sum(
                bool(row["resource_evidence_complete"]) for row in resource_rows
            ),
        },
        "inputs": {
            str(path.resolve()): sha256_file(path.resolve())
            for path in (
                args.lineage_manifest,
                args.dependency_adapter,
                args.hardware_metrics,
                args.device_capabilities,
                node_path,
                edge_path,
            )
        },
        "outputs": {
            "traffic": {
                "path": str(traffic_path.resolve()),
                "sha256": sha256_file(traffic_path),
            },
            "resource": {
                "path": str(resource_path.resolve()),
                "sha256": sha256_file(resource_path),
            },
        },
    }
    model_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "model": str(model_path.resolve()),
                "sha256": sha256_file(model_path),
                "coverage": model["coverage"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
