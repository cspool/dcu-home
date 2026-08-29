#!/usr/bin/env python3
"""Capture fresh Qwen3.5 fixed-input FX templates from the R01 request.

The measured response always comes from the original eager model.  Selected
decoder-layer inputs are cloned at entry and replayed through ``make_fx`` only
after the request has returned and the temporary wrappers have been restored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


COMPARABLE_RESULT_KEYS = (
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "output_token_count",
    "output_token_ids_sha256",
    "output_text_sha256",
    "finish_reason",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def activate_current_build(root_dir: Path) -> dict[str, str]:
    sys.path.insert(0, str(root_dir))
    import vllm

    candidates = sorted(root_dir.glob("build/lib.*-cpython-*/vllm"))
    for candidate in candidates:
        if (candidate / "_C.abi3.so").is_file():
            candidate_text = str(candidate.resolve())
            if candidate_text not in vllm.__path__:
                vllm.__path__.append(candidate_text)
            break
    else:
        raise RuntimeError("current checkout has no built vllm/_C.abi3.so")
    import vllm._C
    import vllm._rocm_C

    return {
        "vllm_python": str(Path(vllm.__file__).resolve()),
        "vllm_C": str(Path(vllm._C.__file__).resolve()),
        "vllm_rocm_C": str(Path(vllm._rocm_C.__file__).resolve()),
    }


def request_result(output: Any) -> dict[str, Any]:
    if len(output) != 1 or len(output[0].outputs) != 1:
        raise RuntimeError("expected one request with one completion")
    request = output[0]
    completion = request.outputs[0]
    prompt_ids = list(request.prompt_token_ids or [])
    output_ids = list(completion.token_ids)
    text = str(completion.text)
    return {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "output_token_count": len(output_ids),
        "output_token_ids_sha256": canonical_sha256(output_ids),
        "output_text": text,
        "output_text_sha256": sha256_bytes(text.encode("utf-8")),
        "finish_reason": completion.finish_reason,
    }


def comparable(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in COMPARABLE_RESULT_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-row", type=int, default=0)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r01-run-metadata", type=Path, required=True)
    parser.add_argument("--r01-layer-events", type=Path, required=True)
    parser.add_argument("--r01-handoff", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--runtime-patch-dir", type=Path, required=True)
    parser.add_argument("--runtime-artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-handoff-output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--finalize-timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    model_root = args.model_root.resolve()
    dataset = args.dataset.resolve()
    contract_path = args.contract.resolve()
    r01_metadata_path = args.r01_run_metadata.resolve()
    r01_layer_events = args.r01_layer_events.resolve()
    r01_handoff = args.r01_handoff.resolve()
    selected_manifest = args.selected_manifest.resolve()
    patch_dir = args.runtime_patch_dir.resolve()
    runtime_artifact_root = args.runtime_artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    capture_handoff_output = args.capture_handoff_output.resolve()
    required = (
        root_dir,
        model_root,
        dataset,
        contract_path,
        r01_metadata_path,
        r01_layer_events,
        r01_handoff,
        selected_manifest,
        patch_dir / "r032_selected_layer_fx_patch.py",
        patch_dir / "vllm_selected_layer_fx_patch.py",
        runtime_artifact_root,
    )
    for path in required:
        if not path.exists():
            raise RuntimeError(f"required capture input is missing: {path}")
    for role, path in (
        ("output directory", output_dir),
        ("capture handoff", capture_handoff_output),
    ):
        if path != runtime_artifact_root and not path.is_relative_to(
            runtime_artifact_root
        ):
            raise RuntimeError(f"{role} escapes runtime_artifact_root: {path}")
    if args.max_new_tokens != 32 or args.warmup_iters != 1:
        raise RuntimeError("fresh R02 capture requires max_new_tokens=32, warmup=1")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "1":
        raise RuntimeError("fresh R02 capture requires physical DCU 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("fresh R02 capture requires CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("FX capture requires in-process V1 execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    reserved = [
        output_dir / "run_metadata.json",
        output_dir / "fx_layer_events.csv",
        output_dir / "fx_layer_trace_manifest.csv",
        output_dir / "FINALIZE_DONE.json",
        output_dir / "FINALIZE_REQUESTED",
        output_dir / "FX_ARMED",
        output_dir / "request" / "result.json",
        capture_handoff_output,
    ]
    if any(path.exists() for path in reserved):
        raise RuntimeError("refusing to overwrite fresh FX capture outputs")

    contract = load_object(contract_path)
    contract_payload = dict(contract)
    recorded_contract_sha256 = str(contract_payload.pop("contract_sha256"))
    if canonical_sha256(contract_payload) != recorded_contract_sha256:
        raise RuntimeError("R01 contract canonical SHA-256 mismatch")
    r01_metadata = load_object(r01_metadata_path)
    if (
        r01_metadata.get("contract_id") != contract.get("contract_id")
        or r01_metadata.get("contract_sha256") != recorded_contract_sha256
    ):
        raise RuntimeError("R01 run metadata differs from the frozen contract")
    revision = subprocess.check_output(
        ["git", "-C", str(root_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    r01_revision = str(contract.get("source", {}).get("revision", ""))
    if not r01_revision:
        raise RuntimeError("R01 contract lacks source revision provenance")

    raw_lines = dataset.read_bytes().splitlines()
    raw_row = raw_lines[args.dataset_row]
    prompt = json.loads(raw_row).get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("selected dataset row has no string prompt")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered_ids = tokenizer(rendered_prompt, add_special_tokens=False).input_ids
    observed_prompt = {
        "dataset_row_raw_sha256": sha256_bytes(raw_row),
        "prompt_text_sha256": sha256_bytes(prompt.encode("utf-8")),
        "rendered_prompt_sha256": sha256_bytes(rendered_prompt.encode("utf-8")),
        "rendered_prompt_token_count": len(rendered_ids),
        "rendered_prompt_token_ids_sha256": canonical_sha256(rendered_ids),
    }
    for key, value in observed_prompt.items():
        if contract.get("same_input", {}).get("prompt", {}).get(key) != value:
            raise RuntimeError(f"R01 prompt drift for {key}")

    build_binding = activate_current_build(root_dir)
    os.environ.update(
        {
            "VLLM_R032_FX_ENABLE": "1",
            "VLLM_R032_FX_ANALYSIS_TYPE": (
                "qwen35_fresh_run_full_request_shape_class_fx_trace"
            ),
            "VLLM_SELECTED_LAYER_FX_DIR": str(output_dir),
            "VLLM_SELECTED_LAYER_FX_ARM_FILE": str(output_dir / "FX_ARMED"),
            "VLLM_SELECTED_LAYER_FX_CANONICAL_MANIFEST": str(selected_manifest),
            "VLLM_SELECTED_LAYER_FX_SELECTION_HANDOFF": str(r01_handoff),
            "VLLM_SELECTED_LAYER_FX_SOURCE_TRACE": str(r01_layer_events),
            "VLLM_R032_FX_FINALIZE_FILE": str(output_dir / "FINALIZE_REQUESTED"),
            "VLLM_R032_FX_DONE_FILE": str(output_dir / "FINALIZE_DONE.json"),
            "VLLM_R032_FINALIZE_PROTOCOL": (
                "controller creates FINALIZE_REQUESTED after the exact R01 request returns"
            ),
            "VLLM_TRACE_RUN_ID": args.tag,
            "VLLM_TRACE_CONTRACT_ID": str(contract["contract_id"]),
            "VLLM_R032_SOURCE_ROOT": str(root_dir),
            "VLLM_R032_SOURCE_REVISION": revision,
            "VLLM_R032_MODEL_PATH": str(model_root),
            "VLLM_R032_MODEL_CONFIG_SHA256": sha256_file(model_root / "config.json"),
            "VLLM_R032_DEVICE_ID": "physical_dcu_1_logical_0",
            "VLLM_R032_DEVICE_SERIAL": "resolved_in_capture_runtime",
        }
    )
    sys.path.insert(0, str(patch_dir))
    fx_patch = importlib.import_module("r032_selected_layer_fx_patch")
    fx_patch.apply_patches()

    import torch
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=0,
        ignore_eos=False,
        max_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=str(model_root),
        served_model_name=args.served_model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        seed=0,
        attention_config={
            "backend": "ROCM_AITER_UNIFIED_ATTN",
            "use_prefill_decode_attention": False,
        },
        disable_log_stats=True,
    )

    warmup_output = llm.chat(
        messages,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    torch.cuda.synchronize()
    warmup_result = request_result(warmup_output)
    if comparable(warmup_result) != comparable(r01_metadata["measured_result"]):
        raise RuntimeError("unarmed warmup output differs from R01")

    write_json_exclusive(
        output_dir / "FX_ARMED",
        {
            "contract_id": contract["contract_id"],
            "contract_sha256": recorded_contract_sha256,
            "selected_manifest": str(selected_manifest),
            "selected_manifest_sha256": sha256_file(selected_manifest),
            "armed_realtime_ns": time.time_ns(),
        },
    )
    started_realtime_ns = time.time_ns()
    started_perf_ns = time.perf_counter_ns()
    measured_output = llm.chat(
        messages,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    torch.cuda.synchronize()
    ended_perf_ns = time.perf_counter_ns()
    ended_realtime_ns = time.time_ns()
    measured_result = request_result(measured_output)
    if comparable(measured_result) != comparable(r01_metadata["measured_result"]):
        raise RuntimeError("FX-sampled eager output differs from R01")
    if comparable(warmup_result) != comparable(measured_result):
        raise RuntimeError("FX capture warmup and measured outputs differ")

    request_record = {
        "schema_version": 1,
        "status": "eager_request_complete_fx_finalize_pending",
        "tag": args.tag,
        "contract_id": contract["contract_id"],
        "contract_sha256": recorded_contract_sha256,
        "same_input": observed_prompt,
        "sampling": r01_metadata["sampling"],
        "warmup_iters": args.warmup_iters,
        "warmup_result": warmup_result,
        "measured_result": measured_result,
        "request_start_realtime_ns": started_realtime_ns,
        "request_end_realtime_ns": ended_realtime_ns,
        "request_latency_ms": (ended_perf_ns - started_perf_ns) / 1e6,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "logical_device": 0,
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **build_binding,
        },
        "source_revision": revision,
        "r01_source_revision": r01_revision,
        "source_revision_matches_r01": revision == r01_revision,
        "source_hash_equality_required": False,
        "selected_manifest": str(selected_manifest),
        "selected_manifest_sha256": sha256_file(selected_manifest),
        "response_source": "original eager Qwen3.5 decoder layers",
        "fx_graph_used_for_response": False,
    }
    write_json_exclusive(output_dir / "request" / "result.json", request_record)
    write_json_exclusive(
        output_dir / "request" / "request_contract.json",
        {
            "schema_version": 1,
            "contract_path": str(contract_path),
            "contract_file_sha256": sha256_file(contract_path),
            "contract_id": contract["contract_id"],
            "contract_sha256": recorded_contract_sha256,
            "r01_run_metadata": str(r01_metadata_path),
            "r01_run_metadata_sha256": sha256_file(r01_metadata_path),
            "r01_layer_events": str(r01_layer_events),
            "r01_layer_events_sha256": sha256_file(r01_layer_events),
            "source_revision": revision,
            "r01_source_revision": r01_revision,
            "source_revision_matches_r01": revision == r01_revision,
            "source_hash_equality_required": False,
        },
    )
    write_json_exclusive(
        output_dir / "FINALIZE_REQUESTED",
        {
            "request_returned_realtime_ns": ended_realtime_ns,
            "requested_realtime_ns": time.time_ns(),
        },
    )

    deadline = time.monotonic() + args.finalize_timeout_seconds
    done_path = output_dir / "FINALIZE_DONE.json"
    while not done_path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("FX offline finalization timed out")
        time.sleep(0.5)
    done = load_object(done_path)
    expected_count = sum(
        1
        for line in selected_manifest.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    )
    if (
        done.get("status") != "complete"
        or done.get("fx_sample_count") != expected_count
        or done.get("fx_trace_count") != expected_count
        or done.get("fx_trace_error_count") != 0
    ):
        raise RuntimeError(f"FX finalization failed: {done}")
    with selected_manifest.open(encoding="utf-8", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))
    ordered_event_ids = [
        f"input{row['forward_id']}_layer{row['layer_idx']}"
        for row in selected_rows
    ]
    ordered_source_event_ids = [row["source_event_id"] for row in selected_rows]
    ordered_selection_ids = [row["selection_id"] for row in selected_rows]
    run_metadata_path = output_dir / "run_metadata.json"
    trace_manifest_path = output_dir / "fx_layer_trace_manifest.csv"
    layer_events_path = output_dir / "fx_layer_events.csv"
    capture_handoff = {
        "schema_version": 1,
        "status": "complete",
        "source_goal": "R02_FX_CAPTURE",
        "runtime_goal": "R02",
        "skill": "qwen-dcu-fx-process-nvtx-instrumentation",
        "runtime": {"run_id": args.tag},
        "selection": {
            "ordered_selection_ids": ordered_selection_ids,
            "selected_event_count": len(ordered_event_ids),
        },
        "source_binding": {
            "revision": revision,
            "r01_revision": r01_revision,
            "source_revision_matches_r01": revision == r01_revision,
            "source_hash_equality_required": False,
            "contract_id": contract["contract_id"],
            "contract_sha256": recorded_contract_sha256,
            "r01_run_metadata": str(r01_metadata_path),
            "r01_run_metadata_sha256": sha256_file(r01_metadata_path),
            "r01_layer_events": str(r01_layer_events),
            "r01_layer_events_sha256": sha256_file(r01_layer_events),
        },
        "artifacts": {
            "run_metadata": {
                "path": str(run_metadata_path),
                "sha256": sha256_file(run_metadata_path),
            },
            "fx_layer_events": {
                "path": str(layer_events_path),
                "sha256": sha256_file(layer_events_path),
            },
            "fx_layer_trace_manifest": {
                "path": str(trace_manifest_path),
                "sha256": sha256_file(trace_manifest_path),
            },
            "request_result": {
                "path": str(output_dir / "request" / "result.json"),
                "sha256": sha256_file(output_dir / "request" / "result.json"),
            },
        },
        "downstream_contract": {
            "consume_as_is": True,
            "fx_root": str(output_dir),
            "ordered_fx_event_ids": ordered_event_ids,
            "ordered_source_event_ids": ordered_source_event_ids,
            "capture_run_id": args.tag,
            "evidence_boundary": (
                "same-lineage fresh-run fixed-input FX structure; no process timing"
            ),
        },
    }
    write_json_exclusive(capture_handoff_output, capture_handoff)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "fresh_fx_template_count": expected_count,
                "request_result": str(output_dir / "request" / "result.json"),
                "run_metadata": str(output_dir / "run_metadata.json"),
                "capture_handoff": str(capture_handoff_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
