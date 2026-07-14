#!/usr/bin/env python3
"""Reconstruct process groups from selected-layer Qwen3.5 FX traces.

This script produces evidence artifacts only: process groups, node tables,
node ranges, targets, users, and shape/dtype metadata. Process explanation and
manual tensor-axis diagrams belong in a separate visualization document.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessDef:
    process_id: str
    title: str
    rule: str
    node_indices: tuple[int, ...]


PROCESS_DEFS = [
    ProcessDef(
        "runtime_inputs",
        "Runtime FX inputs",
        "Fixed sampled layer inputs and control tensors exposed as FX placeholders.",
        tuple(range(0, 3)),
    ),
    ProcessDef(
        "pre_attention_residual_rmsnorm",
        "Pre-attention residual add and input RMSNorm",
        "Combine hidden_states with residual, then run RMSNorm over the Hidden axis.",
        tuple(range(3, 16)),
    ),
    ProcessDef(
        "qkv_projection_and_split",
        "Fused Q/gate/K/V projection and head reshape",
        "Project normalized hidden states, split fused projection into Q/gate/K/V branches, and reshape Q heads.",
        tuple(range(17, 32)),
    ),
    ProcessDef(
        "q_head_rmsnorm",
        "Q head RMSNorm",
        "Normalize Q head vectors over head_dim before rotary embedding.",
        tuple(range(32, 46)),
    ),
    ProcessDef(
        "k_head_rmsnorm",
        "K head RMSNorm",
        "Normalize K head vectors over head_dim before rotary embedding.",
        tuple(range(46, 60)),
    ),
    ProcessDef(
        "mrope_table_lookup",
        "MROPE cos/sin table lookup and axis remap",
        "Index rotary tables with sampled positions, split cos/sin halves, and remap MROPE axis slices.",
        tuple(range(60, 85)),
    ),
    ProcessDef(
        "q_rope_apply",
        "Q RoPE application",
        "Apply rotary transform to the normalized Q branch and restore flattened Q layout.",
        tuple(range(85, 102)),
    ),
    ProcessDef(
        "k_rope_apply",
        "K RoPE application",
        "Apply rotary transform to the normalized K branch and restore flattened K layout.",
        tuple(range(102, 119)),
    ),
    ProcessDef(
        "vllm_attention_and_kv_cache",
        "vLLM attention and KV cache update",
        "View Q/K/V/output buffers by head, update KV cache, then call vLLM attention output custom op.",
        tuple(range(119, 126)),
    ),
    ProcessDef(
        "attention_gate_projection_residual",
        "Attention gate, output projection, and residual",
        "Flatten attention output, gate it with the fused Q-side branch, project to Hidden width, and add the pre-attention residual.",
        (16, *range(126, 133), 135),
    ),
    ProcessDef(
        "post_attention_rmsnorm",
        "Post-attention RMSNorm",
        "Normalize the post-attention residual over the Hidden axis before the MLP projection.",
        (133, 134, *range(136, 146)),
    ),
    ProcessDef(
        "mlp_and_layer_output",
        "MLP and layer output tuple",
        "Run fused gate/up projection, SiLU-and-multiply activation, down projection, and package layer outputs.",
        tuple(range(146, 155)),
    ),
]


NODE_FIELDS = [
    "process_order",
    "process_id",
    "process_title",
    "node_index",
    "node_name",
    "op",
    "target",
    "shape",
    "dtype",
    "users",
    "args",
    "kwargs",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def node_shape(node: dict[str, Any]) -> list[int] | None:
    val = node.get("meta", {}).get("val")
    if isinstance(val, dict):
        shape = val.get("shape")
        if isinstance(shape, list):
            return shape
    tensor_meta = node.get("meta", {}).get("tensor_meta")
    if isinstance(tensor_meta, list) and tensor_meta and isinstance(tensor_meta[0], list):
        return tensor_meta[0]
    return None


def node_dtype(node: dict[str, Any]) -> str | None:
    val = node.get("meta", {}).get("val")
    if isinstance(val, dict):
        dtype = val.get("dtype")
        if dtype:
            return str(dtype)
    tensor_meta = node.get("meta", {}).get("tensor_meta")
    if isinstance(tensor_meta, list) and len(tensor_meta) > 1:
        return str(tensor_meta[1])
    return None


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": node.get("index"),
        "name": node.get("name"),
        "op": node.get("op"),
        "target": node.get("target"),
        "shape": node_shape(node),
        "dtype": node_dtype(node),
        "users": node.get("users", []),
        "args": node.get("args", ""),
        "kwargs": node.get("kwargs", ""),
    }


def index_ranges(indices: list[int]) -> list[str]:
    if not indices:
        return []
    ranges: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = idx
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ranges


def process_shapes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for node in nodes:
        shape = node_shape(node)
        if shape is not None:
            shapes[str(node.get("name"))] = shape
    return shapes


def process_targets(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(node.get("target")) for node in nodes)
    return dict(sorted(counts.items()))


def resolve_trace_dir(trace_dir: str, run_dir: Path) -> Path:
    path = Path(trace_dir)
    if path.is_absolute() or path.exists():
        return path
    for parent in (run_dir, *run_dir.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def load_trace_dirs(run_dir: Path) -> list[tuple[str, dict[str, str], Path]]:
    contexts_dir = run_dir / "contexts"
    if not contexts_dir.exists():
        raise SystemExit(f"contexts directory not found: {contexts_dir}")
    items: list[tuple[str, dict[str, str], Path]] = []
    for context_dir in sorted(contexts_dir.iterdir()):
        if not context_dir.is_dir():
            continue
        manifest = context_dir / "fx_trace" / "fx_layer_trace_manifest.csv"
        for row in read_csv(manifest):
            if row.get("status") != "ok":
                continue
            trace_dir = resolve_trace_dir(row.get("trace_dir", ""), run_dir)
            items.append((context_dir.name, row, trace_dir))
    return items


def reconstruct_event(context: str, manifest_row: dict[str, str], trace_dir: Path) -> dict[str, Any]:
    nodes_path = trace_dir / "fx_nodes.json"
    metadata_path = trace_dir / "fx_trace_metadata.json"
    graph_path = trace_dir / "fx_graph.py"
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    nodes: list[dict[str, Any]] = read_json(nodes_path)
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    by_index = {int(node["index"]): node for node in nodes}

    assigned: dict[int, str] = {}
    processes: list[dict[str, Any]] = []
    for order, proc in enumerate(PROCESS_DEFS, start=1):
        proc_nodes = [by_index[idx] for idx in proc.node_indices if idx in by_index]
        for node in proc_nodes:
            assigned[int(node["index"])] = proc.process_id
        indices = [int(node["index"]) for node in proc_nodes]
        processes.append(
            {
                "order": order,
                "process_id": proc.process_id,
                "title": proc.title,
                "rule": proc.rule,
                "node_indices": indices,
                "node_ranges": index_ranges(indices),
                "targets": process_targets(proc_nodes),
                "shapes": process_shapes(proc_nodes),
                "nodes": [compact_node(node) for node in proc_nodes],
            }
        )

    all_indices = {int(node["index"]) for node in nodes}
    unassigned = sorted(all_indices - set(assigned))
    duplicate_assignments = [
        idx
        for idx in sorted(all_indices)
        if sum(idx in proc.node_indices for proc in PROCESS_DEFS) > 1
    ]

    reconstruction = {
        "analysis_type": "qwen35_selected_layer_fx_process_reconstruction",
        "context": context,
        "event_id": manifest_row.get("event_id"),
        "layer_id": manifest_row.get("layer_id"),
        "layer_type": manifest_row.get("layer_type"),
        "forward_id": manifest_row.get("forward_id"),
        "phase": manifest_row.get("phase"),
        "q_len": manifest_row.get("q_len") or metadata.get("q_len"),
        "source_files": {
            "fx_nodes_json": str(nodes_path),
            "fx_graph_py": str(graph_path),
            "fx_trace_metadata_json": str(metadata_path),
            "fx_layer_trace_manifest": "fx_layer_trace_manifest.csv",
        },
        "input_binding": metadata.get("input_binding", {}),
        "specialization": metadata.get("specialization", {}),
        "evidence_boundary": {
            "runtime_sampling": "selected decoder-layer inputs were cloned during real vLLM eager inference",
            "fx_dag": "process groups are reconstructed from the fixed-input FX DAG and node metadata",
            "process_labels": "labels are reconstruction rules, not official FX or vLLM module ownership",
        },
        "node_count": len(nodes),
        "assigned_node_count": len(assigned),
        "unassigned_node_indices": unassigned,
        "duplicate_assignment_indices": duplicate_assignments,
        "process_count": len(processes),
        "processes": processes,
    }
    write_event_outputs(trace_dir, reconstruction)
    return {
        "context": context,
        "event_id": manifest_row.get("event_id"),
        "layer_id": manifest_row.get("layer_id"),
        "forward_id": manifest_row.get("forward_id"),
        "q_len": reconstruction["q_len"],
        "trace_dir": str(trace_dir),
        "node_count": len(nodes),
        "process_count": len(processes),
        "assigned_node_count": len(assigned),
        "unassigned_node_indices": unassigned,
        "duplicate_assignment_indices": duplicate_assignments,
        "outputs": {
            "fx_process_reconstruction_json": str(trace_dir / "fx_process_reconstruction.json"),
            "fx_process_reconstruction_md": str(trace_dir / "fx_process_reconstruction.md"),
            "fx_process_nodes_csv": str(trace_dir / "fx_process_nodes.csv"),
        },
    }


def write_event_outputs(trace_dir: Path, reconstruction: dict[str, Any]) -> None:
    write_json(trace_dir / "fx_process_reconstruction.json", reconstruction)
    write_nodes_csv(trace_dir / "fx_process_nodes.csv", reconstruction)
    write_reconstruction_md(trace_dir / "fx_process_reconstruction.md", reconstruction)


def write_nodes_csv(path: Path, reconstruction: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_FIELDS)
        writer.writeheader()
        for proc in reconstruction["processes"]:
            for node in proc["nodes"]:
                writer.writerow(
                    {
                        "process_order": proc["order"],
                        "process_id": proc["process_id"],
                        "process_title": proc["title"],
                        "node_index": node["index"],
                        "node_name": node["name"],
                        "op": node["op"],
                        "target": node["target"],
                        "shape": json.dumps(node["shape"], separators=(",", ":")),
                        "dtype": node["dtype"] or "",
                        "users": json.dumps(node["users"], ensure_ascii=False),
                        "args": node["args"],
                        "kwargs": node["kwargs"],
                    }
                )


def write_reconstruction_md(path: Path, reconstruction: dict[str, Any]) -> None:
    lines = [
        "# FX Process Reconstruction",
        "",
        f"Event: `{reconstruction['event_id']}`",
        f"Context: `{reconstruction['context']}`",
        f"Layer: `{reconstruction['layer_id']}` (`{reconstruction['layer_type']}`)",
        f"Forward/phase/q_len: `{reconstruction['forward_id']}` / `{reconstruction['phase']}` / `{reconstruction['q_len']}`",
        f"Node coverage: `{reconstruction['assigned_node_count']}/{reconstruction['node_count']}`",
        "",
        "This file is reconstruction evidence only: process labels are rule-based labels over the fixed-input FX DAG.",
        "",
        "## Process Table",
        "",
        "| Order | Process | Node ranges | Targets |",
        "| ---: | --- | --- | ---: |",
    ]
    for proc in reconstruction["processes"]:
        lines.append(
            f"| {proc['order']} | `{proc['process_id']}` {proc['title']} | "
            f"`{', '.join(proc['node_ranges'])}` | {len(proc['targets'])} |"
        )
    if reconstruction["unassigned_node_indices"]:
        lines.extend(
            [
                "",
                f"Unassigned nodes: `{reconstruction['unassigned_node_indices']}`",
            ]
        )
    if reconstruction["duplicate_assignment_indices"]:
        lines.extend(
            [
                "",
                f"Duplicate assignments: `{reconstruction['duplicate_assignment_indices']}`",
            ]
        )
    lines.append("")

    for proc in reconstruction["processes"]:
        lines.extend(
            [
                f"## {proc['order']}. {proc['title']}",
                "",
                f"- process_id: `{proc['process_id']}`",
                f"- rule: {proc['rule']}",
                f"- node ranges: `{', '.join(proc['node_ranges'])}`",
                f"- target set: `{', '.join(proc['targets'])}`",
                "",
                "```text",
            ]
        )
        for node in proc["nodes"]:
            shape = node["shape"] if node["shape"] is not None else "-"
            dtype = node["dtype"] or "-"
            users = ",".join(node["users"]) if node["users"] else "-"
            lines.append(
                f"#{node['index']:03d} {node['name']:<30} "
                f"{node['op']:<14} {node['target']:<45} "
                f"shape={shape} dtype={dtype} users={users}"
            )
        lines.extend(["```", ""])

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_run_summary(run_dir: Path, items: list[dict[str, Any]]) -> None:
    summary = {
        "analysis_type": "qwen35_selected_layer_fx_process_reconstruction_summary",
        "run_dir": str(run_dir),
        "event_count": len(items),
        "all_nodes_assigned": all(not item["unassigned_node_indices"] for item in items),
        "no_duplicate_assignments": all(not item["duplicate_assignment_indices"] for item in items),
        "process_count_set": sorted({item["process_count"] for item in items}),
        "node_count_set": sorted({item["node_count"] for item in items}),
        "items": items,
        "visualization": str(run_dir / "selected_layer_fx_process_visualization.md"),
    }
    write_json(run_dir / "selected_layer_fx_process_summary.json", summary)

    lines = [
        "# Selected-Layer FX Process Reconstruction Summary",
        "",
        f"Run directory: `{run_dir}`",
        f"Events reconstructed: `{len(items)}`",
        f"Node counts: `{summary['node_count_set']}`",
        f"Process counts: `{summary['process_count_set']}`",
        f"All nodes assigned: `{summary['all_nodes_assigned']}`",
        f"No duplicate assignments: `{summary['no_duplicate_assignments']}`",
        "",
        "| Context | Event | Layer | Forward | q_len | Nodes | Processes | Unassigned |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted(items, key=lambda x: (x["context"], x["event_id"] or "")):
        lines.append(
            "| {context} | `{event}` | {layer} | {forward} | {q_len} | {nodes} | {processes} | `{unassigned}` |".format(
                context=item["context"],
                event=item["event_id"],
                layer=item["layer_id"],
                forward=item["forward_id"],
                q_len=item["q_len"],
                nodes=item["node_count"],
                processes=item["process_count"],
                unassigned=item["unassigned_node_indices"],
            )
        )
    lines.extend(
        [
            "",
            "Per-event outputs:",
            "",
        ]
    )
    for item in sorted(items, key=lambda x: (x["context"], x["event_id"] or "")):
        lines.append(f"- `{item['context']}/{item['event_id']}`: `{item['trace_dir']}`")
    (run_dir / "selected_layer_fx_process_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: reconstruct_selected_layer_fx_processes.py RUN_DIR")
    run_dir = Path(sys.argv[1])
    trace_items = load_trace_dirs(run_dir)
    if not trace_items:
        raise SystemExit(f"no ok trace rows found under: {run_dir}")
    summary_items = [
        reconstruct_event(context, row, trace_dir)
        for context, row, trace_dir in trace_items
    ]
    write_run_summary(run_dir, summary_items)
    print(json.dumps(summary_items, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
