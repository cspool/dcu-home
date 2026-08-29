#!/usr/bin/env python3
"""Build auditable fresh-run process dependencies from fixed-input FX.

The adapter never turns temporal adjacency into a data edge.  It transfers an
FX edge to a measured event only when the layer/phase family and shape relation
are explicit. Stage instrumentation changes are carried by a run-lineage
manifest; source revision/hash equality is not a dependency-validity gate.
Opaque custom operations and missing endpoints remain first-class unknown rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROCESS_MARKER_RE = re.compile(
    r"^pra\.fx_process\.(?P<event>input\d+_layer\d+)\."
    r"(?P<stage>[A-Za-z0-9_]+)(?:\.(?P<fragment>[A-Za-z0-9_]+))?$"
)


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


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def as_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "").strip()
    return int(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Workflow05 fresh-run dependency adapter."
    )
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--measurement-contract", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--runtime-calls", type=Path, required=True)
    parser.add_argument("--strict-ownership", type=Path, required=True)
    parser.add_argument("--template-assignments", type=Path, required=True)
    parser.add_argument("--fx-manifest", type=Path, required=True)
    parser.add_argument(
        "--stage-source-revision",
        required=True,
        help="Recorded provenance only; equality with an earlier stage is not required.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-exact-shape-template-transfer",
        action="store_true",
        help=(
            "Permit a different event to use a current-revision FX template "
            "only when q_len and kv_len deltas are both zero."
        ),
    )
    return parser.parse_args()


def manifest_reconstructions(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    results = manifest.get("results") if isinstance(manifest, dict) else None
    if not isinstance(results, list):
        raise RuntimeError("FX manifest does not contain results")
    output: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        event_id = result.get("event_id")
        json_record = result.get("json")
        path_value = (
            json_record.get("path") if isinstance(json_record, dict) else None
        )
        recorded_hash = (
            json_record.get("sha256") if isinstance(json_record, dict) else None
        )
        if not isinstance(event_id, str) or not isinstance(path_value, str):
            raise RuntimeError("FX manifest result lacks an event/path")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"FX reconstruction is missing: {path}")
        observed_hash = sha256_file(path)
        if recorded_hash is not None and recorded_hash != observed_hash:
            raise RuntimeError(f"FX reconstruction hash mismatch: {path}")
        if event_id in output:
            raise RuntimeError(f"duplicate FX event: {event_id}")
        reconstruction = load_json(path)
        metadata_path = path.with_name("fx_trace_metadata.json")
        metadata = load_json(metadata_path)
        output[event_id] = {
            "path": path,
            "sha256": observed_hash,
            "payload": reconstruction,
            "metadata_path": metadata_path,
            "metadata_sha256": sha256_file(metadata_path),
            "metadata": metadata,
        }
    if not output:
        raise RuntimeError("FX manifest contains no successful reconstruction")
    return output


def shape_relation(row: dict[str, str], same_event: bool) -> tuple[str, bool]:
    if same_event:
        return "same_event", True
    q_delta = as_int(row, "target_template_q_len_delta")
    kv_delta = as_int(row, "target_template_kv_len_delta")
    if q_delta == 0 and kv_delta == 0:
        return "exact_shape_template", True
    return "non_exact_shape_template", False


def main() -> int:
    args = parse_args()
    inputs = [
        args.lineage_manifest,
        args.measurement_contract,
        args.annotations,
        args.runtime_calls,
        args.strict_ownership,
        args.template_assignments,
        args.fx_manifest,
    ]
    for path in inputs:
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_json(args.measurement_contract)
    lineage = load_json(args.lineage_manifest)
    if (
        not isinstance(lineage, dict)
        or lineage.get("status") != "PASS"
        or lineage.get("evidence_source_policy") != "current_run_only"
        or lineage.get("source_change_policy")
        != "stage_trace_instrumentation_allowed"
        or lineage.get("source_hash_equality_required") is not False
    ):
        raise RuntimeError("lineage manifest does not authorize a fresh-run adapter")
    lineage_id = lineage.get("lineage_id")
    if not isinstance(lineage_id, str) or not lineage_id:
        raise RuntimeError("lineage manifest lacks lineage_id")
    contract_id = contract.get("contract_id")
    contract_sha256 = contract.get("contract_sha256") or contract.get(
        "canonical_sha256"
    )
    if not isinstance(contract_id, str) or not isinstance(contract_sha256, str):
        raise RuntimeError("measurement contract lacks identity")
    contract_source = contract.get("source", {})
    contract_revision = (
        contract_source.get("revision")
        if isinstance(contract_source, dict)
        else None
    ) or contract.get("source_revision")
    if lineage.get("semantic_contract_id") != contract_id:
        raise RuntimeError(
            "lineage semantic contract differs from the measurement contract"
        )

    annotations = [
        row for row in load_csv(args.annotations) if row.get("kind") == "process"
    ]
    if not annotations:
        raise RuntimeError("fresh run contains no process annotations")
    marker_rows: dict[str, dict[str, str]] = {}
    stages_by_event: dict[str, set[str]] = defaultdict(set)
    for row in annotations:
        marker = row.get("message", "")
        match = PROCESS_MARKER_RE.fullmatch(marker)
        if match is None:
            raise RuntimeError(f"invalid process marker: {marker}")
        if marker in marker_rows:
            raise RuntimeError(f"duplicate current process marker: {marker}")
        if row.get("event_id") and row["event_id"] != match.group("event"):
            raise RuntimeError(f"process marker event mismatch: {marker}")
        stage = match.group("stage")
        fragment = match.group("fragment")
        marker_suffix = stage + (f".{fragment}" if fragment else "")
        if row.get("stage") and row["stage"] not in {stage, marker_suffix}:
            raise RuntimeError(f"process marker stage mismatch: {marker}")
        marker_rows[marker] = row
        # FX process_stage values name the semantic stage, while the exact
        # HIPTX range may append a fragment identity.  Keep the fragment in
        # marker identity/ownership but normalize it away for structural FX
        # endpoint matching.
        stages_by_event[match.group("event")].add(stage)

    runtime_calls = load_csv(args.runtime_calls)
    ownership = load_csv(args.strict_ownership)
    owned_markers = {
        row.get("marker", "") for row in ownership if row.get("kind") == "process"
    }
    runtime_process_owners = {
        row.get("process_owner", "") for row in runtime_calls if row.get("process_owner")
    }
    for marker in sorted(owned_markers.union(runtime_process_owners)):
        if marker not in marker_rows:
            raise RuntimeError(f"runtime evidence references unknown marker: {marker}")

    assignments = load_csv(args.template_assignments)
    assignment_by_event: dict[str, dict[str, str]] = {}
    for row in assignments:
        event_id = row.get("event_id", "")
        if not event_id:
            continue
        if event_id in assignment_by_event:
            raise RuntimeError(f"duplicate template assignment: {event_id}")
        assignment_by_event[event_id] = row
    fx = manifest_reconstructions(args.fx_manifest)
    stage_revision = args.stage_source_revision.strip()
    fx_source_revisions: set[str] = set()
    for event_id, record in fx.items():
        source_revision = record["metadata"].get("source_revision")
        if isinstance(source_revision, str) and source_revision:
            fx_source_revisions.add(source_revision)

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    event_audit: list[dict[str, Any]] = []
    unknown_by_event: Counter[str] = Counter()
    verified_edge_count = 0
    transferred_edge_count = 0

    for current_event in sorted(stages_by_event):
        assignment = assignment_by_event.get(current_event)
        if current_event in fx:
            template_event = current_event
            assignment = assignment or {"event_id": current_event}
        elif assignment is not None:
            template_event = assignment.get("template_event_id", "")
        else:
            template_event = ""
        if template_event not in fx:
            unknown_by_event[current_event] += 1
            edge_rows.append(
                {
                    "edge_id": f"{current_event}:unknown:no_fx_template",
                    "contract_id": contract_id,
                    "event_id": current_event,
                    "template_event_id": template_event,
                    "source_node": "",
                    "source_stage": "",
                    "target_node": "",
                    "target_stage": "",
                    "edge_type": "unknown_dependency",
                    "evidence_class": "unavailable",
                    "verified": False,
                    "shape_relation": "unavailable",
                    "provenance": "missing_current_revision_fx_template",
                    "reason": "no current-revision FX reconstruction",
                }
            )
            continue

        relation, shape_exact = shape_relation(
            assignment or {}, current_event == template_event
        )
        transfer_allowed = current_event == template_event or (
            shape_exact and args.allow_exact_shape_template_transfer
        )
        reconstruction = fx[template_event]["payload"]
        identity = reconstruction.get("event_identity", {})
        current_annotation = next(
            row for row in annotations if row.get("event_id") == current_event
        )
        current_phase = current_annotation.get("phase")
        current_layer_type = current_annotation.get("workload_type")
        compatible_context = (
            identity.get("phase") == current_phase
            and identity.get("layer_type") == current_layer_type
        )
        verified_context = transfer_allowed and compatible_context
        nodes = reconstruction.get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError(f"FX reconstruction has no nodes: {template_event}")
        node_by_name = {
            node.get("name"): node
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("name"), str)
        }
        if len(node_by_name) != len(nodes):
            raise RuntimeError(f"FX node identity is not unique: {template_event}")
        captured_stages = stages_by_event[current_event]
        visible_node_count = 0
        for node in nodes:
            stage = node.get("process_stage")
            captured = stage in captured_stages
            if captured:
                visible_node_count += 1
            node_rows.append(
                {
                    "contract_id": contract_id,
                    "event_id": current_event,
                    "template_event_id": template_event,
                    "node_id": f"{current_event}:{stage}:{node.get('name')}",
                    "template_node_name": node.get("name"),
                    "process_stage": stage,
                    "op": node.get("op"),
                    "target": node.get("target"),
                    "shape_json": json.dumps(node.get("shape"), separators=(",", ":")),
                    "dtype": node.get("dtype"),
                    "captured_process_stage": captured,
                    "shape_relation": relation,
                    "evidence_class": (
                        "observed_fixed_input_fx_current_source"
                        if current_event == template_event and verified_context
                        else (
                            "inferred_fixed_input_fx_current_source_exact_shape"
                            if verified_context
                            else "unavailable"
                        )
                    ),
                    "fx_reconstruction_sha256": fx[template_event]["sha256"],
                }
            )

        local_edge_count = 0
        local_verified = 0
        for target in nodes:
            target_name = target.get("name")
            target_stage = target.get("process_stage")
            args_list = target.get("args")
            if not isinstance(args_list, list):
                args_list = []
            for source_name in sorted(set(str(value) for value in args_list)):
                source = node_by_name.get(source_name)
                if source is None:
                    continue
                source_stage = source.get("process_stage")
                if source_stage == target_stage:
                    continue
                endpoint_captured = (
                    source_stage in captured_stages and target_stage in captured_stages
                )
                verified = verified_context and endpoint_captured
                evidence = (
                    "observed_fixed_input_fx_current_source"
                    if verified and current_event == template_event
                    else (
                        "inferred_fixed_input_fx_current_source_exact_shape"
                        if verified
                        else "unavailable"
                    )
                )
                edge_id = (
                    f"{current_event}:data:{source_stage}:{source_name}->"
                    f"{target_stage}:{target_name}"
                )
                edge_rows.append(
                    {
                        "edge_id": edge_id,
                        "contract_id": contract_id,
                        "event_id": current_event,
                        "template_event_id": template_event,
                        "source_node": source_name,
                        "source_stage": source_stage,
                        "target_node": target_name,
                        "target_stage": target_stage,
                        "edge_type": "data" if verified else "unknown_dependency",
                        "evidence_class": evidence,
                        "verified": verified,
                        "shape_relation": relation,
                        "provenance": (
                            "fx_node_args_users_cross_stage_current_revision"
                        ),
                        "reason": (
                            ""
                            if verified
                            else (
                                "process endpoint not captured"
                                if not endpoint_captured
                                else "non-exact or incompatible FX context"
                            )
                        ),
                    }
                )
                local_edge_count += 1
                if verified:
                    local_verified += 1
                    if current_event == template_event:
                        verified_edge_count += 1
                    else:
                        transferred_edge_count += 1
                else:
                    unknown_by_event[current_event] += 1

        guards = reconstruction.get("evidence_guards", {})
        opaque_ops = guards.get("opaque_custom_ops", [])
        if not isinstance(opaque_ops, list):
            opaque_ops = []
        for opaque in opaque_ops:
            opaque_nodes = [
                node
                for node in nodes
                if str(node.get("target", "")) == str(opaque)
            ]
            for node in opaque_nodes or [{"name": "", "process_stage": ""}]:
                unknown_by_event[current_event] += 1
                edge_rows.append(
                    {
                        "edge_id": (
                            f"{current_event}:unknown:opaque:{opaque}:"
                            f"{node.get('name', '')}"
                        ),
                        "contract_id": contract_id,
                        "event_id": current_event,
                        "template_event_id": template_event,
                        "source_node": node.get("name", ""),
                        "source_stage": node.get("process_stage", ""),
                        "target_node": "",
                        "target_stage": node.get("process_stage", ""),
                        "edge_type": "unknown_dependency",
                        "evidence_class": "unavailable",
                        "verified": False,
                        "shape_relation": relation,
                        "provenance": "fx_opaque_custom_operation_boundary",
                        "reason": "opaque custom-op internal dependencies are hidden",
                    }
                )
        event_audit.append(
            {
                "event_id": current_event,
                "template_event_id": template_event,
                "shape_relation": relation,
                "phase_compatible": identity.get("phase") == current_phase,
                "layer_type_compatible": (
                    identity.get("layer_type") == current_layer_type
                ),
                "captured_stage_count": len(captured_stages),
                "visible_fx_node_count": visible_node_count,
                "cross_stage_edge_count": local_edge_count,
                "verified_cross_stage_edge_count": local_verified,
                "unknown_dependency_count": unknown_by_event[current_event],
            }
        )

    edge_ids = [str(row["edge_id"]) for row in edge_rows]
    if len(edge_ids) != len(set(edge_ids)):
        duplicates = [key for key, count in Counter(edge_ids).items() if count > 1]
        raise RuntimeError(f"dependency edge identity is not unique: {duplicates[:8]}")

    node_path = args.output_dir / "fresh_run_dependency_nodes.csv"
    edge_path = args.output_dir / "fresh_run_dependency_edges.csv"
    audit_path = args.output_dir / "fresh_run_dependency_event_audit.csv"
    adapter_path = args.output_dir / "fresh_run_dependency_adapter.json"
    write_csv(
        node_path,
        [
            "contract_id",
            "event_id",
            "template_event_id",
            "node_id",
            "template_node_name",
            "process_stage",
            "op",
            "target",
            "shape_json",
            "dtype",
            "captured_process_stage",
            "shape_relation",
            "evidence_class",
            "fx_reconstruction_sha256",
        ],
        node_rows,
    )
    write_csv(
        edge_path,
        [
            "edge_id",
            "contract_id",
            "event_id",
            "template_event_id",
            "source_node",
            "source_stage",
            "target_node",
            "target_stage",
            "edge_type",
            "evidence_class",
            "verified",
            "shape_relation",
            "provenance",
            "reason",
        ],
        edge_rows,
    )
    write_csv(
        audit_path,
        [
            "event_id",
            "template_event_id",
            "shape_relation",
            "phase_compatible",
            "layer_type_compatible",
            "captured_stage_count",
            "visible_fx_node_count",
            "cross_stage_edge_count",
            "verified_cross_stage_edge_count",
            "unknown_dependency_count",
        ],
        event_audit,
    )

    verified = sum(bool(row["verified"]) for row in edge_rows)
    unknown = sum(row["edge_type"] == "unknown_dependency" for row in edge_rows)
    denominator = verified + unknown
    adapter = {
        "schema_version": 1,
        "status": "complete",
        "adapter_type": "fresh_run_fixed_input_fx_process_dependency",
        "lineage_id": lineage_id,
        "contract_id": contract_id,
        "contract_sha256": contract_sha256,
        "stage_source_revision": stage_revision,
        "contract_source_revision": contract_revision,
        "fx_source_revisions": sorted(fx_source_revisions),
        "source_revision_policy": (
            "recorded_stage_instrumentation_delta; revision/hash equality not required"
        ),
        "clock_policy": "dependency structure only; no timestamp adjacency inference",
        "edge_semantics": {
            "data": "current-revision fixed-input FX args/users cross-stage edge",
            "unknown_dependency": (
                "opaque, incompatible, uncaptured, or otherwise unresolved relation"
            ),
            "temporal_adjacency_used_as_dependency": False,
            "same_queue_order_used_as_data_dependency": False,
        },
        "coverage": {
            "current_process_marker_count": len(marker_rows),
            "current_event_count": len(stages_by_event),
            "fx_node_row_count": len(node_rows),
            "dependency_edge_row_count": len(edge_rows),
            "verified_dependency_edge_count": verified,
            "same_event_verified_edge_count": verified_edge_count,
            "exact_shape_transferred_edge_count": transferred_edge_count,
            "unknown_dependency_count": unknown,
            "verified_dependency_fraction": (
                verified / denominator if denominator else None
            ),
        },
        "inputs": {
            str(path.resolve()): sha256_file(path.resolve()) for path in inputs
        },
        "outputs": {
            "nodes": {"path": str(node_path.resolve()), "sha256": sha256_file(node_path)},
            "edges": {"path": str(edge_path.resolve()), "sha256": sha256_file(edge_path)},
            "event_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
        "parameters": {
            "allow_exact_shape_template_transfer": (
                args.allow_exact_shape_template_transfer
            )
        },
    }
    adapter["canonical_payload_sha256"] = canonical_sha256(adapter)
    adapter_path.write_text(
        json.dumps(adapter, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "adapter": str(adapter_path.resolve()),
                "sha256": sha256_file(adapter_path),
                "verified_dependency_fraction": adapter["coverage"][
                    "verified_dependency_fraction"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
