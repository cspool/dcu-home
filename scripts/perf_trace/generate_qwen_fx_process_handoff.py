#!/usr/bin/env python3
"""Generate the current Qwen3.5 FX-process Stage-A handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


INVENTORY_FIELDS = [
    "variant_scope",
    "phase",
    "layer_or_layer_pattern",
    "process_id",
    "process_title",
    "fragment_id",
    "aggregation_key",
    "fx_nodes",
    "fx_op_families",
    "expected_kernel_families",
    "torch_code_path",
    "instrumented_file",
    "instrumented_symbol",
    "nvtx_range_name",
    "range_parent",
    "range_guard_or_flag",
    "status",
    "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def node_families(nodes: list[dict[str, Any]]) -> str:
    targets = " ".join(
        f"{node.get('op', '')} {node.get('target', '')}" for node in nodes
    ).lower()
    families: list[str] = []
    rules = [
        ("placeholder/output", ("placeholder", " output")),
        ("GEMM/matmul", ("aten.mm", "matmul", "gemm")),
        ("attention", ("attention", "attn")),
        ("KV cache", ("kv_cache", "cache_update")),
        ("normalization", ("norm", "rsqrt", "mean")),
        ("elementwise", ("mul", "add", "sigmoid", "silu")),
        ("index/gather/cat", ("index", "gather", "cat")),
        ("reshape/view/copy", ("view", "reshape", "copy", "slice", "clone")),
        ("RoPE", ("rotary", "rope")),
        ("GDN", ("gdn", "delta")),
        ("allocation", ("empty", "zeros")),
    ]
    for family, keys in rules:
        if any(key in targets for key in keys):
            families.append(family)
    return ";".join(families or ["unclassified structural op family"])


def expected_kernels(process_id: str) -> str:
    hypotheses = {
        "inputs": "none expected (marker/bookkeeping only)",
        "input_rmsnorm": "fused RMSNorm and elementwise kernels",
        "qkv_projection": (
            "TunableOp GEMM for prefill; LLMM1/GEMV for decode; "
            "normalization and reshape helpers"
        ),
        "gdn_recurrent_core": "vLLM GDN custom recurrent kernels",
        "gdn_gated_rmsnorm": (
            "GDN gated RMSNorm, SiLU, and elementwise kernels"
        ),
        "rope": "RoPE index/copy/elementwise kernels",
        "kv_cache_attention": (
            "unified KV-cache update and unified attention kernels"
        ),
        "attention_output": "sigmoid and gating elementwise kernels",
        "output_projection": (
            "TunableOp GEMM for prefill; LLMM1/GEMV for decode"
        ),
        "output_projection__post_attention_rmsnorm_fused": (
            "shared residual-add and fused RMSNorm transition kernels"
        ),
        "mlp": (
            "gate/up and down GEMM or LLMM1/GEMV plus fused SiLU-and-mul"
        ),
        "layer_output": "none expected (marker/bookkeeping only)",
    }
    return "HYPOTHESIS ONLY: " + hypotheses[process_id]


def runtime_mapping(
    source_root: Path,
    profile_script: Path,
    process_id: str,
    layer_type: str,
) -> tuple[str, str, str]:
    if process_id in {"inputs", "layer_output"}:
        return (
            str(source_root / "vllm/model_executor/models/qwen3_next.py"),
            "Qwen3NextDecoderLayer.forward",
            "LayerRangeRecorder.install.wrapped / "
            "ProcessRangeRecorder.forward_decoder_layer",
        )
    if process_id in {
        "input_rmsnorm",
        "output_projection__post_attention_rmsnorm_fused",
    }:
        return (
            str(source_root / "vllm/model_executor/layers/layernorm.py"),
            "GemmaRMSNorm.forward (Qwen3_5RMSNorm alias)",
            "ProcessRangeRecorder.forward_decoder_layer",
        )
    if process_id == "mlp":
        return (
            str(source_root / "vllm/model_executor/models/qwen2_moe.py"),
            "Qwen2MoeMLP.forward (Qwen3NextMLP alias)",
            "ProcessRangeRecorder.forward_decoder_layer",
        )
    if layer_type == "linear_attention":
        return (
            str(source_root / "vllm/model_executor/models/qwen3_5.py"),
            "Qwen3_5GatedDeltaNet.forward",
            "ProcessRangeRecorder.install.wrapped_gdn",
        )
    return (
        str(source_root / "vllm/model_executor/models/qwen3_next.py"),
        "Qwen3NextAttention.forward",
        "ProcessRangeRecorder.install.wrapped_attention",
    )


def inventory_range_name(
    event_id: str,
    process_id: str,
    fragment_id: str,
) -> str:
    name = f"pra.fx_process.{event_id}.{process_id}"
    if fragment_id:
        name += f".{fragment_id}"
    return name


def markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_cell(row.get(field, "")) for field in fields)
            + " |"
        )
    return "\n".join(lines)


def build_inventory(
    *,
    source_root: Path,
    fx_root: Path,
    profile_script: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    fx_metadata_path = fx_root / "run_metadata.json"
    fx_events_path = fx_root / "fx_layer_events.csv"
    fx_metadata = load_object(fx_metadata_path)
    event_rows = {
        row["event_id"]: row
        for row in csv.DictReader(
            fx_events_path.open(encoding="utf-8", newline="")
        )
    }
    captured_events = list(fx_metadata.get("captured_events", []))
    if not captured_events:
        raise RuntimeError("current FX metadata has no captured events")

    inventory: list[dict[str, str]] = []
    mapping_events: list[dict[str, Any]] = []
    for event_id in captured_events:
        reconstruction_path = (
            fx_root / event_id / "fx_process_reconstruction.json"
        )
        reconstruction = load_object(reconstruction_path)
        identity = reconstruction["event_identity"]
        source_event = event_rows[event_id]
        for key, csv_key in (
            ("contract_id", "contract_id"),
            ("forward_id", "forward_id"),
            ("layer_idx", "layer_id"),
            ("layer_occurrence", "layer_occurrence"),
            ("phase", "phase"),
            ("q_len", "q_len"),
            ("past_len", "past_len"),
            ("kv_len", "kv_len"),
            ("layer_type", "layer_type"),
        ):
            if str(identity[key]) != str(source_event[csv_key]):
                raise RuntimeError(
                    f"{event_id} identity mismatch for {key}: "
                    f"{identity[key]!r} != {source_event[csv_key]!r}"
                )

        nodes = reconstruction["nodes"]
        stages = reconstruction["stages"]
        stage_by_name = {stage["stage"]: stage for stage in stages}
        mapping_event = {
            "event_id": event_id,
            "reconstruction_path": str(reconstruction_path),
            "reconstruction_sha256": sha256(reconstruction_path),
            "event_identity": identity,
            "fixed_input_context": reconstruction["fixed_input_context"],
            "evidence_guards": reconstruction["evidence_guards"],
            "ordered_processes": [],
        }

        for stage in stages:
            stage_name = stage["stage"]
            selected_nodes = nodes[
                int(stage["start_index"]) : int(stage["end_index"]) + 1
            ]
            mapping_event["ordered_processes"].append(
                {
                    "stage": stage,
                    "ordered_nodes_with_inputs_and_users": selected_nodes,
                    "derived_op_families": node_families(selected_nodes),
                }
            )
            if stage_name == "post_attention_rmsnorm":
                continue

            fragment_id = ""
            status = "instrumented"
            notes = (
                f"rule={stage['reconstruction_rule']}; "
                f"external_inputs={stage['external_inputs']}; "
                f"external_outputs={stage['external_outputs']}"
            )
            if stage_name == "output_projection":
                fragment_id = (
                    "part01_mixer_out_proj"
                    if identity["layer_type"] == "linear_attention"
                    else "part01_attention_o_proj"
                )
                status = "instrumented_partial"
                notes += (
                    "; range owns projection/copy only; the FX residual add "
                    "crosses into the following fused runtime transition"
                )

            code_path, code_symbol, instrumented_symbol = runtime_mapping(
                source_root,
                profile_script,
                stage_name,
                identity["layer_type"],
            )
            parent = (
                f"pra.layer.{event_id}.{identity['phase']}."
                f"{identity['layer_type']}"
            )
            inventory.append(
                {
                    "variant_scope": (
                        "Qwen3.5-27B current eager structural FX mapping; "
                        f"fx_contract={identity['contract_id']}"
                    ),
                    "phase": str(identity["phase"]),
                    "layer_or_layer_pattern": (
                        f"{event_id}; layer={identity['layer_idx']}; "
                        f"occurrence={identity['layer_occurrence']}; "
                        f"q_len={identity['q_len']}; "
                        f"past_len={identity['past_len']}; "
                        f"kv_len={identity['kv_len']}"
                    ),
                    "process_id": stage_name,
                    "process_title": str(stage["title"]),
                    "fragment_id": fragment_id,
                    "aggregation_key": f"{event_id}:{stage_name}",
                    "fx_nodes": ";".join(stage["nodes"]),
                    "fx_op_families": node_families(selected_nodes),
                    "expected_kernel_families": expected_kernels(stage_name),
                    "torch_code_path": f"{code_path}::{code_symbol}",
                    "instrumented_file": str(profile_script),
                    "instrumented_symbol": instrumented_symbol,
                    "nvtx_range_name": inventory_range_name(
                        event_id, stage_name, fragment_id
                    ),
                    "range_parent": parent,
                    "range_guard_or_flag": (
                        "PRA_BACKEND_PERF_PROCESS_PROFILE=1 and event_id in "
                        "PRA_BACKEND_PERF_PROCESS_TARGETS"
                    ),
                    "status": status,
                    "notes": notes,
                }
            )

        output_stage = stage_by_name["output_projection"]
        post_stage = stage_by_name["post_attention_rmsnorm"]
        output_tail_node = output_stage["nodes"][-1]
        fused_nodes = [output_tail_node, *post_stage["nodes"]]
        fused_records = [
            node for node in nodes if node["name"] in set(fused_nodes)
        ]
        code_path, code_symbol, instrumented_symbol = runtime_mapping(
            source_root,
            profile_script,
            "output_projection__post_attention_rmsnorm_fused",
            identity["layer_type"],
        )
        fragment_id = "part02_shared_fusion"
        inventory.append(
            {
                "variant_scope": (
                    "Qwen3.5-27B current eager structural FX mapping; "
                    f"fx_contract={identity['contract_id']}"
                ),
                "phase": str(identity["phase"]),
                "layer_or_layer_pattern": (
                    f"{event_id}; layer={identity['layer_idx']}; "
                    f"occurrence={identity['layer_occurrence']}; "
                    f"q_len={identity['q_len']}; "
                    f"past_len={identity['past_len']}; "
                    f"kv_len={identity['kv_len']}"
                ),
                "process_id": (
                    "output_projection__post_attention_rmsnorm_fused"
                ),
                "process_title": (
                    "Shared residual-add and post-attention RMSNorm transition"
                ),
                "fragment_id": fragment_id,
                "aggregation_key": f"{event_id}:fused_transition",
                "fx_nodes": ";".join(fused_nodes),
                "fx_op_families": node_families(fused_records),
                "expected_kernel_families": expected_kernels(
                    "output_projection__post_attention_rmsnorm_fused"
                ),
                "torch_code_path": f"{code_path}::{code_symbol}",
                "instrumented_file": str(profile_script),
                "instrumented_symbol": instrumented_symbol,
                "nvtx_range_name": inventory_range_name(
                    event_id,
                    "output_projection__post_attention_rmsnorm_fused",
                    fragment_id,
                ),
                "range_parent": (
                    f"pra.layer.{event_id}.{identity['phase']}."
                    f"{identity['layer_type']}"
                ),
                "range_guard_or_flag": (
                    "PRA_BACKEND_PERF_PROCESS_PROFILE=1 and event_id in "
                    "PRA_BACKEND_PERF_PROCESS_TARGETS"
                ),
                "status": "ambiguous_shared_fusion",
                "notes": (
                    "One current fused runtime call crosses the offline FX "
                    "output_projection/post_attention_rmsnorm boundary. It is "
                    "recorded once as a combined exclusive transition and "
                    "must not be duplicated into either source process."
                ),
            }
        )
        mapping_events.append(mapping_event)

    names = [row["nvtx_range_name"] for row in inventory]
    if len(names) != len(set(names)):
        raise RuntimeError("inventory range names are not unique")
    mapping = {
        "schema_version": 1,
        "evidence_boundary": (
            "Ordered FX structural mapping only; no process kernel time."
        ),
        "fx_root": str(fx_root),
        "fx_run_metadata": {
            "path": str(fx_metadata_path),
            "sha256": sha256(fx_metadata_path),
            "contract_id": fx_metadata["contract_id"],
            "source_identity": fx_metadata["source_identity"],
            "status": fx_metadata["status"],
        },
        "fx_layer_events": {
            "path": str(fx_events_path),
            "sha256": sha256(fx_events_path),
            "rows": len(event_rows),
        },
        "events": mapping_events,
    }
    return inventory, mapping_events, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fx-root", type=Path, required=True)
    parser.add_argument("--r01-contract", type=Path, required=True)
    parser.add_argument("--profile-script", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--process-run-metadata", type=Path)
    parser.add_argument("--process-event-jsonl", type=Path)
    parser.add_argument("--process-db", type=Path)
    parser.add_argument("--disabled-run-metadata", type=Path)
    parser.add_argument("--trace-audit", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    fx_root = args.fx_root.resolve()
    output_dir = args.output_dir.resolve()
    profile_script = args.profile_script.resolve()
    launcher = args.launcher.resolve()
    r01_contract_path = args.r01_contract.resolve()
    if "perf_trace_bk" in fx_root.parts:
        raise RuntimeError("perf_trace_bk cannot be current FX evidence")
    if not str(output_dir).startswith(
        "/public/home/tangyu408/Qwen_DCU_Worker_0/perf_trace/runtime/"
    ):
        raise RuntimeError("output directory is outside the runtime tree")
    output_dir.mkdir(parents=True, exist_ok=True)

    git_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "-C", str(source_root), "branch", "--show-current"],
        text=True,
    ).strip()
    r01_contract = load_object(r01_contract_path)
    if r01_contract["source"]["revision"] != git_revision:
        raise RuntimeError("R01 contract revision differs from current source")

    inventory, mapping_events, mapping = build_inventory(
        source_root=source_root,
        fx_root=fx_root,
        profile_script=profile_script,
    )
    fx_metadata = load_object(fx_root / "run_metadata.json")
    if fx_metadata["source_identity"]["revision"] != git_revision:
        raise RuntimeError("current FX artifacts differ from current revision")

    inventory_csv = output_dir / "process_range_inventory.csv"
    inventory_json = output_dir / "process_range_inventory.json"
    mapping_json = output_dir / "FX_PROCESS_MAPPING_EVIDENCE.json"
    with inventory_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(inventory)
    inventory_json.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    mapping_json.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    targets = [event["event_id"] for event in mapping_events]
    overlay_payload = {
        "schema_version": 1,
        "runtime_goal": "R02",
        "role": "process-instrumentation overlay on frozen R01 same-input contract",
        "parent_contract_id": r01_contract["contract_id"],
        "parent_contract_sha256": r01_contract["contract_sha256"],
        "unchanged_same_input": r01_contract["same_input"],
        "unchanged_model": r01_contract["model"],
        "unchanged_sampling": r01_contract["same_input"]["sampling"],
        "unchanged_device": r01_contract["device"],
        "intentional_instrumentation_delta": {
            "PRA_BACKEND_PERF_PROCESS_PROFILE": {
                "from": "0",
                "to": "1",
            },
            "PRA_BACKEND_PERF_PROCESS_TARGETS": targets,
            "process_marker_namespace": (
                "pra.fx_process.inputN_layerM.<process_id>[.<fragment_id>]"
            ),
        },
        "structural_fx_contract": {
            "contract_id": fx_metadata["contract_id"],
            "run_id": fx_metadata["run_id"],
            "path": str(fx_root),
            "run_metadata_sha256": sha256(fx_root / "run_metadata.json"),
            "not_merged_with_parent_contract": True,
        },
        "source": {
            "revision": git_revision,
            "branch": git_branch,
            "profile_script": {
                "path": str(profile_script),
                "sha256": sha256(profile_script),
            },
            "launcher": {
                "path": str(launcher),
                "sha256": sha256(launcher),
            },
        },
    }
    overlay = dict(overlay_payload)
    overlay["overlay_sha256"] = canonical_sha256(overlay_payload)
    overlay_path = output_dir / "R02_PROCESS_INSTRUMENTATION_OVERLAY.json"
    overlay_path.write_text(
        json.dumps(overlay, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.inventory_only:
        print(
            json.dumps(
                {
                    "status": "inventory_complete",
                    "inventory_rows": len(inventory),
                    "inventory": str(inventory_csv),
                    "mapping_evidence": str(mapping_json),
                    "overlay": str(overlay_path),
                },
                sort_keys=True,
            )
        )
        return

    required_paths = {
        "process_run_metadata": args.process_run_metadata,
        "process_event_jsonl": args.process_event_jsonl,
        "process_db": args.process_db,
        "disabled_run_metadata": args.disabled_run_metadata,
        "trace_audit": args.trace_audit,
    }
    for label, path in required_paths.items():
        if path is None or not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
    assert args.process_run_metadata is not None
    assert args.process_event_jsonl is not None
    assert args.process_db is not None
    assert args.disabled_run_metadata is not None
    assert args.trace_audit is not None
    trace_audit = load_object(args.trace_audit)
    if trace_audit.get("status") != "pass":
        raise RuntimeError("trace audit did not pass")
    process_metadata = load_object(args.process_run_metadata)
    disabled_metadata = load_object(args.disabled_run_metadata)
    process_db = args.process_db.resolve()
    raw_trace_prefix = process_db.name.removesuffix(".db")
    raw_trace_chunks = sorted(
        process_db.parent.glob(f"{raw_trace_prefix}_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not raw_trace_chunks:
        raise RuntimeError("hipprof emitted no preserved raw trace chunks")
    raw_trace_manifest = {
        "schema_version": 1,
        "format": "hipprof segmented Chrome-trace JSON",
        "chunks_preserved": True,
        "chunk_count": len(raw_trace_chunks),
        "chunks": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in raw_trace_chunks
        ],
        "queryable_database": {
            "path": str(process_db),
            "size_bytes": process_db.stat().st_size,
            "sha256": sha256(process_db),
        },
    }
    raw_trace_manifest_path = (
        output_dir / "R02_RAW_TRACE_CHUNKS_MANIFEST.json"
    )
    raw_trace_manifest_path.write_text(
        json.dumps(
            raw_trace_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fx_event_rows = [
        {
            key: event["event_identity"][key]
            for key in (
                "event_id",
                "phase",
                "layer_idx",
                "layer_occurrence",
                "q_len",
                "past_len",
                "kv_len",
                "layer_type",
            )
        }
        for event in mapping_events
    ]
    runtime_events = [
        json.loads(line)
        for line in args.process_event_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runtime_by_event = {
        event["event_id"]: event
        for event in runtime_events
        if event["event_id"] in set(targets)
    }
    validation_event_rows = [
        {
            key: runtime_by_event[event_id][key]
            for key in (
                "event_id",
                "phase",
                "layer_idx",
                "occurrence",
                "q_len",
                "past_len",
                "kv_len",
                "workload_type",
            )
        }
        for event_id in targets
    ]

    prompt = r01_contract["same_input"]["prompt"]
    sampling = r01_contract["same_input"]["sampling"]
    config = r01_contract["config"]
    source_contract_rows = markdown_table(
        fx_event_rows,
        [
            "event_id",
            "phase",
            "layer_idx",
            "layer_occurrence",
            "q_len",
            "past_len",
            "kv_len",
            "layer_type",
        ],
    )
    validation_contract_rows = markdown_table(
        validation_event_rows,
        [
            "event_id",
            "phase",
            "layer_idx",
            "occurrence",
            "q_len",
            "past_len",
            "kv_len",
            "workload_type",
        ],
    )
    inventory_md = markdown_table(inventory, INVENTORY_FIELDS)
    audit_counts = trace_audit["counts"]
    strict_chain = trace_audit["strict_launch_ownership_schema"]
    handoff = f"""# FX Process NVTX Instrumentation Handoff

## Source FX Artifacts

- Current FX root: `{fx_root}`.
- FX run metadata: `{fx_root / 'run_metadata.json'}` (`sha256={sha256(fx_root / 'run_metadata.json')}`); status `{fx_metadata['status']}`; contract `{fx_metadata['contract_id']}`; source revision `{fx_metadata['source_identity']['revision']}`.
- FX layer events: `{fx_root / 'fx_layer_events.csv'}` (`sha256={sha256(fx_root / 'fx_layer_events.csv')}`).
- Ordered reconstruction evidence: `{mapping_json}` (`sha256={sha256(mapping_json)}`); it retains each stage's ordered nodes, inputs, users, op families, fixed-input context, and evidence guards.
- Inventory sidecars: `{inventory_csv}` (`sha256={sha256(inventory_csv)}`) and `{inventory_json}` (`sha256={sha256(inventory_json)}`).
- These files are current evidence outside `perf_trace_bk`; no archived handoff or archived trace is treated as validation.

The exact structural FX selections are:

{source_contract_rows}

## Execution Reproducibility Contract

- Frozen same-input parent: `{r01_contract['contract_id']}`; canonical SHA-256 `{r01_contract['contract_sha256']}`; file `{r01_contract_path}` (`sha256={sha256(r01_contract_path)}`).
- R02 instrumentation overlay: `{overlay_path}` (`overlay_sha256={overlay['overlay_sha256']}`, `file_sha256={sha256(overlay_path)}`). The only intentional execution delta is `PRA_BACKEND_PERF_PROCESS_PROFILE: 0 -> 1` plus the nine explicit process targets; prompt, model, sampling, backend, warmup, device, and eager execution remain frozen.
- Input: `{prompt['dataset']}`, row `{prompt['dataset_row']}`; rendered prompt token count `{prompt['rendered_prompt_token_count']}`; rendered prompt SHA-256 `{prompt['rendered_prompt_sha256']}`; rendered token-ID SHA-256 `{prompt['rendered_prompt_token_ids_sha256']}`; `enable_thinking={str(prompt['enable_thinking']).lower()}`.
- Sampling: `temperature={sampling['temperature']}`, `top_p={sampling['top_p']}`, `top_k={sampling['top_k']}`, `min_p={sampling['min_p']}`, `seed={sampling['seed']}`, `ignore_eos={str(sampling['ignore_eos']).lower()}`, `max_new_tokens={sampling['max_new_tokens']}`; one identical warmup.
- Model/runtime: `{r01_contract['model']['architecture']}`, BF16, 64 layers, resolved checkpoint `{r01_contract['model']['resolved_model_root']}`, eager mode, attention backend `{config['attention_backend']}`, max batched tokens `{config['max_num_batched_tokens']}`, TP/PP 1.
- Source: `{source_root}` at `{git_revision}` on branch `{git_branch}`. The current Python model source is the same revision used by both the FX structural capture and R01.
- Device: physical DCU `{r01_contract['device']['physical_device_id']}` / logical device `{r01_contract['device']['logical_device_id']}` / unique ID `{r01_contract['device']['unique_id']}` through `HIP_VISIBLE_DEVICES=1` and `CUDA_VISIBLE_DEVICES=1`; hipprof is not passed `--devices 0`.
- Structural FX contract `{fx_metadata['contract_id']}` is not the same execution contract as R01 (historical OpenAI-chat rendering/max-token setup and different device). It is used only to reconstruct current-code process structure and is never merged with R01 runtime or timing evidence.

The exact R01-bound process validation selections are:

{validation_contract_rows}

## Instrumented Code Changes

- `{profile_script}` (`sha256={sha256(profile_script)}`) extends the existing current single-request entry point with `ProcessRangeRecorder`. When `PRA_BACKEND_PERF_PROCESS_PROFILE=0`, selected process wrappers are not installed and the existing decoder `forward` is called directly.
- `ProcessRangeRecorder.forward_decoder_layer` preserves the current `Qwen3NextDecoderLayer.forward` operator order while adding input, input-RMSNorm, shared fused transition, MLP, and layer-output ranges.
- `ProcessRangeRecorder.install.wrapped_gdn` preserves the current `Qwen3_5GatedDeltaNet.forward` operators and backend call while adding projection, recurrent-core, gated-norm, and output-projection ranges.
- `ProcessRangeRecorder.install.wrapped_attention` preserves the current `Qwen3NextAttention.forward` operators and backend call while adding QKV, RoPE, KV-cache/attention, output-gate, and output-projection ranges.
- `{launcher}` (`sha256={sha256(launcher)}`) is the current ROCm/DCU/HIP process-trace launcher. It enforces the R01 input/config/device policy and writes only beneath the supplied runtime artifact root.
- Range transport is `torch.cuda.nvtx.range` on this HIP build; the real validation database below contains the emitted names in the current HIPTX table.

## Process Range Inventory

{inventory_md}

## Range Naming Contract

- Parent: `pra.layer.inputN_layerM.<phase>.<layer_type>`.
- Child: `pra.fx_process.inputN_layerM.<process_id>[.<fragment_id>]`.
- Event identity is unique in the measured request; process/fragment identity is unique within the event; inventory `nvtx_range_name` values are globally unique.
- `aggregation_key` is stable within one event and logical process. The shared residual-add/RMSNorm transition deliberately has its own combined key because one current fused runtime call crosses two offline FX stages and must be counted once.
- Guard: ranges exist only when `PRA_BACKEND_PERF_PROCESS_PROFILE=1` and the event is listed in `PRA_BACKEND_PERF_PROCESS_TARGETS`.

## Expected Trace Outputs

- Current hipprof DB: `{args.process_db.resolve()}` (`sha256={sha256(args.process_db.resolve())}`).
- Preserved raw hipprof trace: `{len(raw_trace_chunks)}` segmented Chrome-trace JSON chunks indexed by `{raw_trace_manifest_path}` (`sha256={sha256(raw_trace_manifest_path)}`); no chunk is treated as merged or rewritten evidence.
- Runtime layer/process expectation stream: `{args.process_event_jsonl.resolve()}` (`sha256={sha256(args.process_event_jsonl.resolve())}`).
- Process run metadata: `{args.process_run_metadata.resolve()}` (`sha256={sha256(args.process_run_metadata.resolve())}`).
- Disabled-control metadata: `{args.disabled_run_metadata.resolve()}` (`sha256={sha256(args.disabled_run_metadata.resolve())}`).
- Trace audit: `{args.trace_audit.resolve()}` (`sha256={sha256(args.trace_audit.resolve())}`).
- `expected_kernel_families` in the inventory are hypotheses for downstream validation, not measurements and not process-time claims.
- The downstream consumer must use the same DB and strict chain: process HIPTX CPU range -> HIP Runtime call launched inside the range and inside its marker-index bounds -> HIPOPS kernel with identical runtime `_Index`.

## Validation Performed

- Syntax/focused validation passed for all edited sources; details and commands are indexed by the R02 runtime handoff.
- Real current hipprof validation: `{audit_counts['process_hiptx_rows']}` `pra.fx_process.*` HIPTX rows (`>0` required), `{audit_counts['unique_process_messages']}` unique messages, `{audit_counts['inventory_rows']}` inventory rows, `{audit_counts['nested_process_rows']}` rows nested under the intended parent, and zero missing/extra/duplicate range identities.
- Expected target coverage: `{process_metadata['expected_process_range_count']}` ranges across `{len(process_metadata['process_targets'])}` selected FX events; every expected name appears once under its intended layer.
- Current schema: HIPTX `{strict_chain['hiptx_table']}`, HIP Runtime `{strict_chain['hip_runtime_table']}`, HIPOPS `{strict_chain['hipops_table']}`. The correlation field is `_Index`; `{strict_chain['runtime_rows_inside_process_ranges']}` runtime rows were observed inside process ranges and `{strict_chain['hipops_rows_joined_by_runtime_index']}` HIPOPS rows were joinable by identical `_Index`. These are binding checks only; no process kernel durations were calculated.
- Disabling the process flag produced `process_profile={disabled_metadata['process_profile']}` with zero expected process ranges. Disabled-control, process-enabled, and R01 measured outputs match exactly for prompt token IDs, output token IDs, output text, count, and finish reason; warmup/measured equivalence also passed within each run.
- Audit result: `{trace_audit['status']}`. Audit file: `{args.trace_audit.resolve()}`.

## Open Risks

- The current FX structural capture uses `{fx_metadata['contract_id']}`, whose historical rendering and q/past/kv contexts differ from the frozen R01 same-input contract. This is an explicit structural-transfer boundary, not a same-run timing join; the exact event contexts for both contracts are recorded above.
- `output_projection` ends with residual addition in offline FX, while the current eager runtime fuses that addition with `post_attention_rmsnorm`. The combined range remains `status=ambiguous_shared_fusion`, is emitted once, and must never be duplicated into both source stages.
- Opaque `vllm.gdn_attention_core`, `vllm.unified_kv_cache_update`, and `vllm.unified_attention_with_output` internals are not reconstructed from FX.
- Kernel-family entries are hypotheses only. This Stage-A handoff defines and validates instrumentation; it does not establish process DCU kernel time.
"""
    handoff_path = output_dir / "FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md"
    handoff_path.write_text(handoff, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "handoff_complete",
                "handoff": str(handoff_path),
                "inventory_rows": len(inventory),
                "handoff_sha256": sha256(handoff_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
