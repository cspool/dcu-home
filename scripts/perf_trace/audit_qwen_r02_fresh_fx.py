#!/usr/bin/env python3
"""Audit fresh full-request Qwen3.5 FX/process reconstruction evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(r"^input(?P<forward>\d+)_layer(?P<layer>\d+)$")
RANGE_RE = re.compile(
    r"^pra\.fx_process\.(?P<event>input\d+_layer\d+)\."
    r"(?P<process>[A-Za-z0-9_]+)(?:\.(?P<fragment>[A-Za-z0-9_]+))?$"
)
COMPARABLE_RESULT_KEYS = (
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "output_token_count",
    "output_token_ids_sha256",
    "output_text_sha256",
    "finish_reason",
)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def comparable(metadata: dict[str, Any]) -> dict[str, Any]:
    result = metadata["measured_result"]
    return {key: result[key] for key in COMPARABLE_RESULT_KEYS}


def range_name(event_id: str, process: str, fragment: str) -> str:
    value = f"pra.fx_process.{event_id}.{process}"
    return f"{value}.{fragment}" if fragment else value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--r01-contract", type=Path, required=True)
    parser.add_argument("--r01-run-metadata", type=Path, required=True)
    parser.add_argument("--r01-layer-events", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--template-assignments", type=Path, required=True)
    parser.add_argument("--process-targets", type=Path, required=True)
    parser.add_argument("--process-range-targets", type=Path, required=True)
    parser.add_argument("--capture-handoff", type=Path, required=True)
    parser.add_argument("--fx-root", type=Path, required=True)
    parser.add_argument("--reconstruction-manifest", type=Path, required=True)
    parser.add_argument(
        "--runtime-patch", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    fx_root = args.fx_root.resolve()
    paths = [
        args.r01_contract,
        args.r01_run_metadata,
        args.r01_layer_events,
        args.selection_plan,
        args.selected_manifest,
        args.template_assignments,
        args.process_targets,
        args.process_range_targets,
        args.capture_handoff,
        args.reconstruction_manifest,
        *args.runtime_patch,
    ]
    paths = [path.resolve() for path in paths]
    for path in [source_root, fx_root, *paths]:
        if not path.exists():
            raise RuntimeError(f"required audit input is missing: {path}")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing audit output: {output}")
    require("perf_trace_bk" not in fx_root.parts, "archived FX root is forbidden")

    contract = load_object(args.r01_contract.resolve())
    contract_payload = dict(contract)
    recorded_contract_sha = str(contract_payload.pop("contract_sha256"))
    require(
        canonical_sha256(contract_payload) == recorded_contract_sha,
        "R01 canonical contract hash mismatch",
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    r01_revision = str(contract["source"]["revision"])
    current_status_sha = hashlib.sha256(
        subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain=v1", "-z"]
        )
    ).hexdigest()
    r01_status_sha = contract["source"]["git_status_porcelain_v1_z_sha256"]

    traced_sources = contract["source"]["traced_source_files"]
    runtime_source_relations: dict[str, dict[str, Any]] = {}
    for relative in (
        "vllm/model_executor/models/qwen3_5.py",
        "vllm/model_executor/models/qwen3_next.py",
        "vllm/v1/worker/gpu_model_runner.py",
    ):
        path = source_root / relative
        observed = sha256_file(path)
        expected = traced_sources[relative]
        runtime_source_relations[relative] = {
            "path": str(path),
            "r01_sha256": expected,
            "current_sha256": observed,
            "matches": observed == expected,
        }
    profile_relative = (
        "vllm/platforms/tunable_profiles/"
        "gfx936_qwen3_5_27b_bf16_tn_m4096.csv"
    )
    profile_path = source_root / profile_relative
    current_profile_sha = sha256_file(profile_path)
    r01_profile_sha = traced_sources[profile_relative]

    r01_metadata = load_object(args.r01_run_metadata.resolve())
    require(
        r01_metadata["contract_id"] == contract["contract_id"]
        and r01_metadata["contract_sha256"] == recorded_contract_sha,
        "R01 metadata identity mismatch",
    )
    r01_rows = load_csv(args.r01_layer_events.resolve())
    require(len(r01_rows) == 1856, "R01 event count is not 1856")
    r01_by_event: dict[str, dict[str, str]] = {}
    r01_shape_classes: dict[tuple[str, str, int, int], str] = {}
    for row in r01_rows:
        event_id = f"input{row['forward_id']}_layer{row['layer_idx']}"
        require(event_id not in r01_by_event, f"duplicate R01 event: {event_id}")
        r01_by_event[event_id] = row
        shape_key = (
            row["phase"],
            row["workload_type"],
            int(row["q_len"]),
            int(row["kv_len"]),
        )
        r01_shape_classes.setdefault(shape_key, event_id)
    require(len(r01_shape_classes) == 58, "R01 shape-class count is not 58")

    plan = load_object(args.selection_plan.resolve())
    require(plan.get("status") == "complete", "selection plan is incomplete")
    coverage = plan["coverage"]
    expected_coverage = {
        "r01_layer_event_count": 1856,
        "r01_forward_count": 29,
        "unique_phase_layer_type_q_len_kv_len_class_count": 58,
        "fresh_fx_template_count": 58,
        "assigned_event_count": 1856,
        "process_target_event_count": 1856,
        "process_range_target_count": 17168,
    }
    for key, expected in expected_coverage.items():
        require(int(coverage[key]) == expected, f"plan coverage mismatch: {key}")
    require(
        float(coverage["target_coverage_fraction"]) == 1.0,
        "plan target coverage is not complete",
    )

    selected_rows = load_csv(args.selected_manifest.resolve())
    require(len(selected_rows) == 58, "selected FX manifest is not 58 rows")
    selected_by_event: dict[str, dict[str, str]] = {}
    for row in selected_rows:
        event_id = f"input{row['forward_id']}_layer{row['layer_idx']}"
        require(event_id == row["source_event_id"], "fresh template is not same-event")
        require(event_id not in selected_by_event, "duplicate selected FX event")
        selected_by_event[event_id] = row
        source = r01_by_event[event_id]
        for left, right in (
            ("phase", "phase"),
            ("q_len", "q_len"),
            ("past_len", "past_len"),
            ("kv_len", "kv_len"),
            ("layer_type", "workload_type"),
        ):
            require(str(row[left]) == str(source[right]), f"selected drift: {event_id}")
    require(
        set(selected_by_event) == set(r01_shape_classes.values()),
        "selected templates are not the first exact R01 shape representatives",
    )

    assignment_rows = load_csv(args.template_assignments.resolve())
    require(len(assignment_rows) == 1856, "template assignment count is not 1856")
    assignment_by_event: dict[str, dict[str, str]] = {}
    relation_counts: Counter[str] = Counter()
    for row in assignment_rows:
        event_id = row["event_id"]
        template_event = row["template_event_id"]
        require(event_id in r01_by_event, f"unknown assignment event: {event_id}")
        require(template_event in selected_by_event, "assignment template is not fresh")
        require(event_id not in assignment_by_event, "duplicate assignment event")
        assignment_by_event[event_id] = row
        source = r01_by_event[event_id]
        template = selected_by_event[template_event]
        require(
            (
                row["phase"],
                row["layer_type"],
                int(row["q_len"]),
                int(row["kv_len"]),
            )
            == (
                template["phase"],
                template["layer_type"],
                int(template["q_len"]),
                int(template["kv_len"]),
            ),
            f"non-exact template transfer: {event_id}",
        )
        require(
            int(row["target_template_q_len_delta"]) == 0
            and int(row["target_template_kv_len_delta"]) == 0,
            f"non-zero shape delta: {event_id}",
        )
        require(
            str(source["phase"]) == row["phase"]
            and str(source["workload_type"]) == row["layer_type"]
            and int(source["q_len"]) == int(row["q_len"])
            and int(source["kv_len"]) == int(row["kv_len"]),
            f"assignment differs from R01: {event_id}",
        )
        expected_relation = (
            "same_event"
            if event_id == template_event
            else "exact_shape_template_transfer"
        )
        require(row["relation"] == expected_relation, "transfer label mismatch")
        relation_counts[row["relation"]] += 1
    require(set(assignment_by_event) == set(r01_by_event), "assignment coverage gap")

    process_targets = load_lines(args.process_targets.resolve())
    require(
        process_targets == list(r01_by_event),
        "process target order/coverage differs from R01",
    )
    process_ranges = load_lines(args.process_range_targets.resolve())
    require(len(process_ranges) == 17168, "process range count is not 17168")
    require(len(process_ranges) == len(set(process_ranges)), "duplicate process range")
    expected_ranges: list[str] = []
    for event_id, row in r01_by_event.items():
        stages = LINEAR_STAGES if row["workload_type"] == "linear_attention" else FULL_STAGES
        expected_ranges.extend(
            range_name(event_id, process, fragment)
            for process, fragment in stages
        )
    require(process_ranges == expected_ranges, "process range target order/content mismatch")
    require(all(RANGE_RE.fullmatch(name) for name in process_ranges), "invalid range name")

    capture_handoff = load_object(args.capture_handoff.resolve())
    require(
        capture_handoff.get("status") == "complete"
        and capture_handoff.get("runtime_goal") == "R02"
        and capture_handoff.get("source_goal") == "R02_FX_CAPTURE",
        "fresh capture handoff is incomplete",
    )
    require(
        Path(capture_handoff["downstream_contract"]["fx_root"]).resolve()
        == fx_root,
        "capture handoff FX root mismatch",
    )
    require(
        capture_handoff["selection"]["selected_event_count"] == 58,
        "capture handoff selected count mismatch",
    )
    for record in capture_handoff["artifacts"].values():
        path = Path(record["path"]).resolve()
        require(path.is_file(), f"capture artifact missing: {path}")
        require(sha256_file(path) == record["sha256"], "capture artifact hash mismatch")

    fx_metadata = load_object(fx_root / "run_metadata.json")
    require(fx_metadata.get("status") == "complete", "FX run metadata is incomplete")
    require(fx_metadata["contract_id"] == contract["contract_id"], "FX contract mismatch")
    require(
        fx_metadata["source_identity"]["revision"] == revision,
        "FX source revision mismatch",
    )
    require(
        fx_metadata["fx_sample_count"] == 58
        and fx_metadata["fx_trace_count"] == 58
        and fx_metadata["fx_trace_error_count"] == 0,
        "FX sample/trace counts failed",
    )
    require(
        fx_metadata["patch_errors"] == []
        and fx_metadata["capture_errors"] == []
        and fx_metadata["lifecycle"]["wrapper_restore_errors"] == [],
        "FX patch/capture/restore errors are present",
    )
    lifecycle = fx_metadata["lifecycle"]
    require(
        lifecycle["wrappers_restored_before_offline_fx"] is True
        and lifecycle["active_execute_count_at_finalize"] == 0,
        "offline FX lifecycle guard failed",
    )
    require(
        set(fx_metadata["captured_events"]) == set(selected_by_event),
        "captured event set differs from selection",
    )

    request_result = load_object(fx_root / "request/result.json")
    capture_comparable = {
        key: request_result["measured_result"][key]
        for key in COMPARABLE_RESULT_KEYS
    }
    require(
        capture_comparable == comparable(r01_metadata),
        "fresh FX response differs from R01",
    )
    require(
        request_result["warmup_result"] == request_result["measured_result"],
        "fresh FX warmup and measured outputs differ",
    )
    require(
        request_result["fx_graph_used_for_response"] is False
        and request_result["response_source"]
        == "original eager Qwen3.5 decoder layers",
        "FX replay contaminated the runtime response",
    )

    fx_layer_rows = load_csv(fx_root / "fx_layer_events.csv")
    require(len(fx_layer_rows) == 1856, "FX runtime layer row count is not 1856")
    selected_runtime_rows = [row for row in fx_layer_rows if row["matched"] == "True"]
    require(len(selected_runtime_rows) == 58, "FX matched runtime row count is not 58")
    for row in selected_runtime_rows:
        require(row["event_id"] in selected_by_event, "unexpected matched FX event")
        require(row["source_contract_match"] == "True", "FX source join failed")
        require(row["fx_sampled"] == "True" and row["fx_traced"] == "True", "FX trace failed")
        require(row["fx_trace_status"] == "ok", "FX trace status is not ok")

    reconstruction_manifest = load_object(args.reconstruction_manifest.resolve())
    require(
        reconstruction_manifest.get("runtime_goal") == "R02"
        and reconstruction_manifest.get("processed") == 58,
        "reconstruction manifest identity/count mismatch",
    )
    results = reconstruction_manifest.get("results")
    require(isinstance(results, list) and len(results) == 58, "reconstruction result count mismatch")
    reconstruction_ids: set[str] = set()
    total_nodes = 0
    total_stages = 0
    opaque_counts: Counter[str] = Counter()
    for result in results:
        require(result.get("status") == "ok", "reconstruction result failed")
        event_id = result["event_id"]
        require(event_id in selected_by_event, "unknown reconstruction event")
        require(event_id not in reconstruction_ids, "duplicate reconstruction event")
        reconstruction_ids.add(event_id)
        record = result["json"]
        path = Path(record["path"]).resolve()
        require(sha256_file(path) == record["sha256"], "reconstruction hash mismatch")
        payload = load_object(path)
        identity = payload["event_identity"]
        selected = selected_by_event[event_id]
        require(identity["event_id"] == event_id, "reconstruction event mismatch")
        for left, right in (
            ("phase", "phase"),
            ("q_len", "q_len"),
            ("past_len", "past_len"),
            ("kv_len", "kv_len"),
            ("layer_type", "layer_type"),
        ):
            require(str(identity[left]) == str(selected[right]), "reconstruction shape drift")
        nodes = payload["nodes"]
        stages = payload["stages"]
        require(nodes and stages, "empty reconstruction")
        require([node["index"] for node in nodes] == list(range(len(nodes))), "node indices are unstable")
        names = [node["name"] for node in nodes]
        require(len(names) == len(set(names)), "node names are not unique")
        name_set = set(names)
        stage_names = {stage["stage"] for stage in stages}
        assigned: list[int] = []
        for stage in stages:
            assigned.extend(range(int(stage["start_index"]), int(stage["end_index"]) + 1))
        require(sorted(assigned) == list(range(len(nodes))), "stages do not partition nodes")
        require(len(assigned) == len(set(assigned)), "stage ranges overlap")
        for node in nodes:
            require(
                {"name", "args", "users", "process_stage", "shape", "dtype"}.issubset(node),
                "reconstruction node lacks required fields",
            )
            require(node["process_stage"] in stage_names, "node has unknown stage")
            require(set(node["args"]).issubset(name_set), "node has unknown args")
            require(set(node["users"]).issubset(name_set), "node has unknown users")
        observed_opaque = sorted(
            {node["target"] for node in nodes if node["target"].startswith("vllm.")}
        )
        guards = payload["evidence_guards"]
        require(
            observed_opaque == sorted(guards["opaque_custom_ops"]),
            "opaque custom-op guard mismatch",
        )
        require(
            guards["opaque_custom_op_internals_reconstructed"] is False
            and guards["measured_latency_reported"] is False,
            "reconstruction exceeded its evidence boundary",
        )
        total_nodes += len(nodes)
        total_stages += len(stages)
        opaque_counts.update(observed_opaque)
    require(reconstruction_ids == set(selected_by_event), "reconstruction coverage gap")

    patch_records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        for path in args.runtime_patch
    ]
    audit_inputs = {
        str(path): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "runtime_goal": "R02",
        "status": "pass",
        "evidence_boundary": (
            "Same-lineage fresh-run fixed-input FX/process structure only; no "
            "process DCU kernel duration or temporal-order dependency claim."
        ),
        "contract": {
            "contract_id": contract["contract_id"],
            "contract_sha256": recorded_contract_sha,
            "source_revision": revision,
            "r01_source_revision": r01_revision,
            "source_revision_matches_r01": revision == r01_revision,
            "source_hash_equality_required": False,
            "fresh_response_exactly_matches_r01": True,
            "comparable_result": capture_comparable,
        },
        "source_state_relation": {
            "r01_git_status_porcelain_v1_z_sha256": r01_status_sha,
            "current_git_status_porcelain_v1_z_sha256": current_status_sha,
            "git_status_matches_r01": current_status_sha == r01_status_sha,
            "runtime_python_sources": runtime_source_relations,
            "r01_tunable_profile_sha256": r01_profile_sha,
            "current_tunable_profile_sha256": current_profile_sha,
            "tunable_profile_matches_r01": current_profile_sha == r01_profile_sha,
            "runtime_patches": patch_records,
        },
        "coverage": {
            "r01_layer_events": 1856,
            "r01_forwards": 29,
            "unique_exact_shape_classes": 58,
            "fresh_fx_templates": 58,
            "full_request_assignments": 1856,
            "same_event_assignments": relation_counts["same_event"],
            "exact_shape_transfer_assignments": relation_counts[
                "exact_shape_template_transfer"
            ],
            "process_target_events": len(process_targets),
            "process_range_targets": len(process_ranges),
            "target_coverage_fraction": 1.0,
            "reconstructed_nodes": total_nodes,
            "reconstructed_stages": total_stages,
            "opaque_custom_op_counts": dict(sorted(opaque_counts.items())),
        },
        "lifecycle": {
            "response_from_original_eager_path": True,
            "wrapper_restore_errors": [],
            "active_execute_count_at_finalize": 0,
            "wrappers_restored_before_offline_fx": True,
            "fx_sample_count": 58,
            "fx_trace_count": 58,
            "fx_trace_error_count": 0,
        },
        "dependency_guards": {
            "temporal_adjacency_is_data_dependency": False,
            "stream_order_is_data_dependency": False,
            "queue_order_is_data_dependency": False,
        },
        "r07_compatibility": {
            "builder": str(
                source_root
                / "scripts/perf_trace/build_fresh_run_dependency_adapter.py"
            ),
            "fx_manifest": str(args.reconstruction_manifest.resolve()),
            "fx_manifest_sha256": sha256_file(args.reconstruction_manifest.resolve()),
            "template_assignments": str(args.template_assignments.resolve()),
            "template_assignments_sha256": sha256_file(args.template_assignments.resolve()),
            "stage_source_revision": revision,
            "allow_exact_shape_template_transfer_required": True,
            "manifest_schema_compatible": True,
        },
        "inputs": audit_inputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "fresh_fx_templates": 58,
                "full_request_assignments": 1856,
                "process_ranges": 17168,
                "reconstructed_nodes": total_nodes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
