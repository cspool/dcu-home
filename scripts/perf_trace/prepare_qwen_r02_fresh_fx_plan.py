#!/usr/bin/env python3
"""Prepare complete R02 FX-template and process-range targets from R01."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


LINEAR_STAGES = (
    ("inputs", ""),
    ("input_rmsnorm", ""),
    ("qkv_projection", ""),
    ("gdn_recurrent_core", ""),
    ("gdn_gated_rmsnorm", ""),
    ("output_projection", "part01_mixer_out_proj"),
    (
        "output_projection__post_attention_rmsnorm_fused",
        "part02_shared_fusion",
    ),
    ("mlp", ""),
    ("layer_output", ""),
)

FULL_STAGES = (
    ("inputs", ""),
    ("input_rmsnorm", ""),
    ("qkv_projection", ""),
    ("rope", ""),
    ("kv_cache_attention", ""),
    ("attention_output", ""),
    ("output_projection", "part01_attention_o_proj"),
    (
        "output_projection__post_attention_rmsnorm_fused",
        "part02_shared_fusion",
    ),
    ("mlp", ""),
    ("layer_output", ""),
)

SELECTED_FIELDS = (
    "selection_id",
    "source_event_id",
    "run_id",
    "contract_id",
    "rank",
    "worker_id",
    "request_id",
    "engine_step_id",
    "forward_id",
    "layer_idx",
    "layer_occurrence",
    "phase",
    "q_len",
    "past_len",
    "kv_len",
    "layer_type",
)

ASSIGNMENT_FIELDS = (
    "event_id",
    "template_event_id",
    "phase",
    "layer_type",
    "q_len",
    "kv_len",
    "template_q_len",
    "template_kv_len",
    "target_template_q_len_delta",
    "target_template_kv_len_delta",
    "relation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def write_csv_exclusive(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_lines_exclusive(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(value + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--r01-contract", type=Path, required=True)
    parser.add_argument("--r01-layer-events", type=Path, required=True)
    parser.add_argument("--r01-run-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-process-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    contract_path = args.r01_contract.resolve()
    layer_events_path = args.r01_layer_events.resolve()
    run_metadata_path = args.r01_run_metadata.resolve()
    output_dir = args.output_dir.resolve()
    for path in (source_root, contract_path, layer_events_path, run_metadata_path):
        if not path.exists():
            raise RuntimeError(f"required R02 plan input is missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_object(contract_path)
    metadata = load_object(run_metadata_path)
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    r01_revision = str(contract.get("source", {}).get("revision", ""))
    if not r01_revision:
        raise RuntimeError("R01 contract lacks source revision provenance")
    if (
        metadata.get("contract_id") != contract.get("contract_id")
        or metadata.get("contract_sha256") != contract.get("contract_sha256")
    ):
        raise RuntimeError("R01 metadata and contract identities differ")

    with layer_events_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1856:
        raise RuntimeError(f"expected 1856 R01 events, observed {len(rows)}")
    event_ids: list[str] = []
    by_event: dict[str, dict[str, str]] = {}
    shape_key_by_event: dict[str, tuple[str, str, int, int]] = {}
    representative_by_class: dict[tuple[str, str, int, int], dict[str, str]] = {}
    forward_layers: Counter[int] = Counter()
    for row in rows:
        forward_id = int(row["forward_id"])
        layer_idx = int(row["layer_idx"])
        event_id = f"input{forward_id}_layer{layer_idx}"
        if event_id in by_event:
            raise RuntimeError(f"duplicate R01 event identity: {event_id}")
        key = (
            row["phase"],
            row["workload_type"],
            int(row["q_len"]),
            int(row["kv_len"]),
        )
        event_ids.append(event_id)
        by_event[event_id] = row
        shape_key_by_event[event_id] = key
        representative_by_class.setdefault(key, row)
        forward_layers[forward_id] += 1
    if set(forward_layers.values()) != {64} or len(forward_layers) != 29:
        raise RuntimeError("R01 does not contain 29 complete 64-layer forwards")
    if len(representative_by_class) != 58:
        raise RuntimeError(
            f"expected 58 exact phase/type/q/kv classes, observed "
            f"{len(representative_by_class)}"
        )

    selected_rows: list[dict[str, Any]] = []
    template_event_by_class: dict[tuple[str, str, int, int], str] = {}
    ordered_classes = sorted(
        representative_by_class,
        key=lambda key: (
            int(representative_by_class[key]["forward_id"]),
            0 if key[1] == "linear_attention" else 1,
        ),
    )
    for ordinal, key in enumerate(ordered_classes, 1):
        row = representative_by_class[key]
        event_id = f"input{row['forward_id']}_layer{row['layer_idx']}"
        template_event_by_class[key] = event_id
        selected_rows.append(
            {
                "selection_id": f"r02-fresh-shape-{ordinal:03d}",
                "source_event_id": event_id,
                "run_id": metadata["tag"],
                "contract_id": contract["contract_id"],
                "rank": 0,
                "worker_id": "rank0",
                "request_id": "",
                "engine_step_id": row["forward_id"],
                "forward_id": row["forward_id"],
                "layer_idx": row["layer_idx"],
                "layer_occurrence": 0,
                "phase": row["phase"],
                "q_len": row["q_len"],
                "past_len": row["past_len"],
                "kv_len": row["kv_len"],
                "layer_type": row["workload_type"],
            }
        )

    assignments: list[dict[str, Any]] = []
    process_ranges: list[str] = []
    for event_id in event_ids:
        row = by_event[event_id]
        key = shape_key_by_event[event_id]
        template_event = template_event_by_class[key]
        assignments.append(
            {
                "event_id": event_id,
                "template_event_id": template_event,
                "phase": row["phase"],
                "layer_type": row["workload_type"],
                "q_len": row["q_len"],
                "kv_len": row["kv_len"],
                "template_q_len": row["q_len"],
                "template_kv_len": row["kv_len"],
                "target_template_q_len_delta": 0,
                "target_template_kv_len_delta": 0,
                "relation": (
                    "same_event"
                    if event_id == template_event
                    else "exact_shape_template_transfer"
                ),
            }
        )
        stages = (
            LINEAR_STAGES
            if row["workload_type"] == "linear_attention"
            else FULL_STAGES
        )
        for stage, fragment in stages:
            name = f"pra.fx_process.{event_id}.{stage}"
            if fragment:
                name += f".{fragment}"
            process_ranges.append(name)
    if len(process_ranges) != 17168 or len(process_ranges) != len(set(process_ranges)):
        raise RuntimeError(
            f"unexpected full-request process range inventory: {len(process_ranges)}"
        )
    if len(process_ranges) > args.maximum_process_count:
        raise RuntimeError("full-request process plan exceeds the user limit")

    selected_path = output_dir / "fresh_fx_selected_manifest.csv"
    assignments_path = output_dir / "full_request_template_assignments.csv"
    process_targets_path = output_dir / "full_request_process_targets.txt"
    process_ranges_path = output_dir / "full_request_process_range_targets.txt"
    plan_path = output_dir / "R02_FRESH_FX_SELECTION_PLAN.json"
    write_csv_exclusive(selected_path, SELECTED_FIELDS, selected_rows)
    write_csv_exclusive(assignments_path, ASSIGNMENT_FIELDS, assignments)
    write_lines_exclusive(process_targets_path, event_ids)
    write_lines_exclusive(process_ranges_path, process_ranges)

    file_inputs = {
        str(path): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in (contract_path, layer_events_path, run_metadata_path)
    }
    plan = {
        "schema_version": 1,
        "runtime_goal": "R02",
        "status": "complete",
        "policy": "fresh_run_exact_shape_class_coverage",
        "source_revision": revision,
        "r01_source_revision": r01_revision,
        "source_revision_matches_r01": revision == r01_revision,
        "source_hash_equality_required": False,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "inputs": file_inputs,
        "coverage": {
            "r01_layer_event_count": len(rows),
            "r01_forward_count": len(forward_layers),
            "unique_phase_layer_type_q_len_kv_len_class_count": len(
                representative_by_class
            ),
            "fresh_fx_template_count": len(selected_rows),
            "assigned_event_count": len(assignments),
            "exact_shape_transfer_count": sum(
                row["relation"] == "exact_shape_template_transfer"
                for row in assignments
            ),
            "process_target_event_count": len(event_ids),
            "process_range_target_count": len(process_ranges),
            "target_coverage_fraction": 1.0,
        },
        "transfer_guard": (
            "cross-event transfer is permitted only when phase, layer_type, "
            "q_len, and kv_len all match exactly; every transferred row is labeled"
        ),
        "dependency_guard": {
            "temporal_adjacency_is_data_dependency": False,
            "stream_order_is_data_dependency": False,
            "queue_order_is_data_dependency": False,
        },
        "outputs": {
            "selected_manifest": str(selected_path),
            "template_assignments": str(assignments_path),
            "process_targets": str(process_targets_path),
            "process_range_targets": str(process_ranges_path),
        },
    }
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "plan": str(plan_path),
                "fresh_fx_templates": len(selected_rows),
                "process_ranges": len(process_ranges),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
