#!/usr/bin/env python3
"""Legacy existing-evidence parent/child selective-capture preparation.

This helper is intentionally not part of workflow01-10-fresh-e2e.  The fresh
branch keeps R01-R10 in one lineage and uses build_fresh_run_lineage_manifest.py
plus the R06 newline target files instead.  This tool remains available only
for workflow05-existing-evidence, where R01-R05 really are external historical
planning evidence and a child measurement contract is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_SOURCE_ROOT = Path(
    "/public/home/tangyu408/Qwen_DCU_Worker_0/pra2026-bh408"
)
PROCESS_RANGE_RE = re.compile(
    r"^pra\.fx_process\."
    r"(?P<event>input(?P<forward>\d+)_layer(?P<layer>\d+))\."
    r"(?P<process>[A-Za-z0-9_]+)"
    r"(?:\.(?P<fragment>[A-Za-z0-9_]+))?$"
)
TRUE_VALUES = {"1", "true", "yes"}
OUTPUT_NAMES = {
    "contract": "current_measurement_contract.json",
    "plan": "normalized_selective_trace_plan.csv",
    "inventory": "current_process_range_inventory.csv",
    "environment": "selective_capture_targets.env",
    "manifest": "selective_capture_preparation_manifest.json",
}


class PreparationError(RuntimeError):
    """Fail-closed preparation error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"{path} must contain a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_git(source_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise PreparationError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def current_source_binding(source_root: Path) -> dict[str, Any]:
    revision = run_git(source_root, "rev-parse", "HEAD").decode().strip()
    branch = run_git(
        source_root, "rev-parse", "--abbrev-ref", "HEAD"
    ).decode().strip()
    status_raw = run_git(source_root, "status", "--porcelain=v1", "-z")
    diff_raw = run_git(source_root, "diff", "--binary", "HEAD", "--")
    changed_raw = run_git(
        source_root,
        "ls-files",
        "-m",
        "-o",
        "--exclude-standard",
        "-z",
    )
    changed_paths = sorted(
        item.decode(errors="surrogateescape")
        for item in changed_raw.split(b"\0")
        if item
    )
    changed_files = []
    for relative in changed_paths:
        path = source_root / relative
        changed_files.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    source_state = {
        "revision": revision,
        "branch": branch,
        "status_porcelain_sha256": sha256_bytes(status_raw),
        "git_diff_binary_sha256": sha256_bytes(diff_raw),
        "changed_files": changed_files,
    }
    source_state["source_state_sha256"] = canonical_sha256(source_state)
    return source_state


def current_build_binding(source_root: Path) -> dict[str, Any]:
    candidates = sorted(source_root.glob("build/lib.*-cpython-*/vllm"))
    selected = next(
        (
            path
            for path in candidates
            if (path / "_C.abi3.so").is_file()
            and (path / "_rocm_C.abi3.so").is_file()
        ),
        None,
    )
    if selected is None:
        raise PreparationError("no complete current-checkout vLLM build pair")
    result: dict[str, Any] = {"selected_build_dir": str(selected.resolve())}
    for key, filename in (
        ("vllm_C", "_C.abi3.so"),
        ("vllm_rocm_C", "_rocm_C.abi3.so"),
    ):
        path = (selected / filename).resolve()
        result[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    result["build_binding_sha256"] = canonical_sha256(result)
    return result


def validate_parent_contract(parent: dict[str, Any]) -> str:
    payload = dict(parent)
    recorded = str(payload.pop("contract_sha256", ""))
    computed = canonical_sha256(payload)
    if not recorded or recorded != computed:
        raise PreparationError("parent contract canonical SHA-256 mismatch")
    return recorded


def validate_same_input_files(parent: dict[str, Any]) -> dict[str, Any]:
    prompt = parent.get("same_input", {}).get("prompt", {})
    dataset = Path(str(prompt.get("dataset", ""))).resolve()
    expected_dataset_sha = str(prompt.get("dataset_sha256", ""))
    if not dataset.is_file() or sha256_file(dataset) != expected_dataset_sha:
        raise PreparationError("current dataset does not match parent contract")

    model = parent.get("model", {})
    model_candidates = [
        Path(str(model.get("resolved_model_root", ""))),
        Path(str(model.get("model_root", ""))),
    ]
    model_root = next((path.resolve() for path in model_candidates if path.is_dir()), None)
    if model_root is None:
        raise PreparationError("parent model root is unavailable")
    file_checks = {
        "config.json": model.get("config_sha256"),
        "generation_config.json": model.get("generation_config_sha256"),
        "tokenizer_config.json": model.get("tokenizer_config_sha256"),
    }
    observed: dict[str, Any] = {
        "dataset": str(dataset),
        "dataset_sha256": expected_dataset_sha,
        "model_root": str(model_root),
        "model_files": {},
    }
    for filename, expected in file_checks.items():
        path = model_root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise PreparationError(
                f"current model file does not match parent contract: {filename}"
            )
        observed["model_files"][filename] = {
            "path": str(path),
            "sha256": expected,
        }
    return observed


def marker_values(row: dict[str, str]) -> list[str]:
    for field in (
        "hiptx_range",
        "planned_exact_range_name",
        "target_process_range_identities",
    ):
        value = row.get(field, "").strip()
        if value:
            return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
    raise PreparationError(
        "selected plan row lacks hiptx_range, planned_exact_range_name, "
        "or target_process_range_identities"
    )


def selected_plan_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise PreparationError("selection plan is empty")
    selected: list[dict[str, str]] = []
    for row in rows:
        selected_flag = row.get("selected", "").strip().lower()
        collection_flag = row.get("collection_required", "").strip().lower()
        if collection_flag:
            take = collection_flag in TRUE_VALUES
        else:
            take = selected_flag in TRUE_VALUES
        if not take:
            continue
        for marker in marker_values(row):
            expanded = dict(row)
            expanded["hiptx_range"] = marker
            selected.append(expanded)
    if not selected:
        raise PreparationError("selection plan contains no capture target")
    markers = [row["hiptx_range"] for row in selected]
    if len(markers) != len(set(markers)):
        raise PreparationError("selection plan contains duplicate exact markers")
    return selected


def template_for(
    row: dict[str, str],
    match: re.Match[str],
    inventory: list[dict[str, str]],
) -> dict[str, str]:
    process_id = match.group("process")
    fragment_id = match.group("fragment") or ""
    template_event = row.get("template_event_id", "").strip()
    candidates = [
        item
        for item in inventory
        if item.get("process_id") == process_id
        and item.get("fragment_id", "") == fragment_id
    ]
    if template_event:
        prefix = f"pra.fx_process.{template_event}."
        exact = [
            item
            for item in candidates
            if item.get("nvtx_range_name", "").startswith(prefix)
        ]
        if exact:
            candidates = exact
    layer_type = row.get("layer_type", "").strip()
    if layer_type:
        typed = [
            item
            for item in candidates
            if layer_type in item.get("layer_or_layer_pattern", "")
            or layer_type in item.get("variant_scope", "")
        ]
        if typed:
            candidates = typed
    semantic = {
        (
            item.get("process_title", ""),
            item.get("fx_op_families", ""),
            item.get("instrumented_symbol", ""),
        )
        for item in candidates
    }
    if not candidates or len(semantic) != 1:
        raise PreparationError(
            "template inventory join is missing or ambiguous for "
            f"{row['hiptx_range']}: {len(candidates)} candidates"
        )
    return sorted(candidates, key=lambda item: item["nvtx_range_name"])[0]


def relative_aggregation_key(template: dict[str, str], event_id: str) -> str:
    value = template.get("aggregation_key", "")
    suffix = value.split(":", 1)[1] if ":" in value else value
    if not suffix:
        raise PreparationError("template aggregation_key is empty")
    return f"{event_id}:{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--baseline-run-metadata", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--template-inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-batch-id")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if source_root != CANONICAL_SOURCE_ROOT.resolve():
        raise PreparationError(
            f"source root must equal {CANONICAL_SOURCE_ROOT.resolve()}"
        )
    inputs = (
        args.parent_contract,
        args.baseline_run_metadata,
        args.selection_plan,
        args.template_inventory,
    )
    if any(not path.resolve().is_file() for path in inputs):
        raise PreparationError("one or more required input files are missing")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: output_dir / value for key, value in OUTPUT_NAMES.items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise PreparationError(f"refusing to overwrite outputs: {existing}")

    parent_path = args.parent_contract.resolve()
    baseline_path = args.baseline_run_metadata.resolve()
    plan_path = args.selection_plan.resolve()
    inventory_path = args.template_inventory.resolve()
    parent = load_json(parent_path)
    parent_sha = validate_parent_contract(parent)
    baseline = load_json(baseline_path)
    if baseline.get("contract_id") != parent.get("contract_id"):
        raise PreparationError("baseline metadata contract_id mismatch")
    if baseline.get("contract_sha256") != parent_sha:
        raise PreparationError("baseline metadata contract SHA mismatch")
    expected_output = baseline.get("measured_result")
    if not isinstance(expected_output, dict):
        raise PreparationError("baseline metadata lacks measured_result")
    required_output = {
        "prompt_token_count",
        "prompt_token_ids_sha256",
        "output_token_count",
        "output_token_ids_sha256",
        "output_text_sha256",
        "finish_reason",
    }
    if not required_output.issubset(expected_output):
        raise PreparationError("baseline measured_result is incomplete")
    same_input_files = validate_same_input_files(parent)

    plan_rows = selected_plan_rows(plan_path)
    template_inventory = read_csv(inventory_path)
    if not template_inventory:
        raise PreparationError("template inventory is empty")
    resolved_rows: list[tuple[dict[str, str], re.Match[str], dict[str, str]]] = []
    for row in plan_rows:
        match = PROCESS_RANGE_RE.fullmatch(row["hiptx_range"])
        if match is None:
            raise PreparationError(
                f"invalid exact process marker: {row['hiptx_range']}"
            )
        for field, observed in (
            ("event_id", match.group("event")),
            ("process_id", match.group("process")),
        ):
            planned = row.get(field, "").strip()
            if planned and planned != observed:
                raise PreparationError(
                    f"{field} disagrees with exact marker: {planned} != {observed}"
                )
        resolved_rows.append(
            (row, match, template_for(row, match, template_inventory))
        )

    batches = {
        row.get("selection_batch_id", "").strip()
        for row, _, _ in resolved_rows
        if row.get("selection_batch_id", "").strip()
    }
    if args.selection_batch_id:
        selection_batch = args.selection_batch_id
        if batches and batches != {selection_batch}:
            raise PreparationError("selection batch override disagrees with plan")
    elif len(batches) == 1:
        selection_batch = next(iter(batches))
    else:
        raise PreparationError("selection plan must identify one selection batch")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", selection_batch):
        raise PreparationError("selection batch ID has unsafe syntax")

    source_binding = current_source_binding(source_root)
    build_binding = current_build_binding(source_root)
    plan_sha = sha256_file(plan_path)
    markers = sorted(row["hiptx_range"] for row, _, _ in resolved_rows)
    event_ids = sorted({match.group("event") for _, match, _ in resolved_rows})
    parent_revision = str(parent.get("source", {}).get("revision", ""))
    current_revision = source_binding["revision"]
    relation_type = (
        "same_request_same_revision_new_instrumentation"
        if parent_revision == current_revision
        else "same_request_cross_revision"
    )
    created_utc = datetime.now(timezone.utc).isoformat()

    child = json.loads(json.dumps(parent))
    child.pop("contract_sha256", None)
    child["schema_version"] = 2
    child["runtime_goal"] = "R07"
    child["contract_id"] = (
        f"{parent['contract_id']}/workflow05-current@"
        f"{current_revision[:8]}-{plan_sha[:8]}"
    )
    child["created_utc"] = created_utc
    child["parent_contract"] = {
        "contract_id": parent["contract_id"],
        "canonical_sha256": parent_sha,
        "file_sha256": sha256_file(parent_path),
        "path": str(parent_path),
        "evidence_role": "historical_planning_only",
    }
    child["contract_relation"] = {
        "type": relation_type,
        "same_request": True,
        "same_execution_path": parent_revision == current_revision,
        "historical_and_current_clocks_mergeable": False,
        "historical_timing_may_be_claimed_as_current_observed": False,
    }
    child["source"] = {
        "source_root": str(source_root),
        **source_binding,
        "current_build": build_binding,
        "instrumentation": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "profile_entry": source_root
                / "scripts/perf_trace/profile_qwen_same_input_layer.py",
                "launcher": source_root
                / "scripts/perf_trace/run_qwen_process_profile_single_request.sh",
                "analyzer": source_root
                / "scripts/perf_trace/analyze_qwen_hipprof_process_trace.py",
                "capture_preparer": Path(__file__).resolve(),
            }.items()
        },
    }
    child["same_input"]["expected_output"] = {
        key: expected_output[key] for key in sorted(required_output)
    }
    child["same_input"]["current_file_validation"] = same_input_files
    child["config"]["process_profile_assignment"] = (
        "PRA_BACKEND_PERF_PROCESS_PROFILE=1"
    )
    child["config"]["process_event_targets"] = event_ids
    child["config"]["exact_process_range_filter"] = {
        "environment": "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS",
        "required": True,
        "target_count": len(markers),
        "targets_sha256": canonical_sha256(markers),
    }
    child["selection"] = {
        "selection_batch_id": selection_batch,
        "historical_selection_plan": str(plan_path),
        "historical_selection_plan_sha256": plan_sha,
        "exact_process_range_targets": markers,
        "event_targets": event_ids,
    }
    child["output"] = {
        "preparation_root": str(output_dir),
        "capture_output_must_be_fresh": True,
    }
    child["evidence_boundary"] = (
        "Current R07 measurements are observed only under this child contract. "
        "R01-R05 timing and R4 hardware remain historical planning/template "
        "evidence and are not merged into the current clock or execution path."
    )
    child_sha = canonical_sha256(child)
    child["contract_sha256"] = child_sha
    outputs["contract"].write_text(
        json.dumps(child, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    normalized_rows: list[dict[str, Any]] = []
    current_inventory: list[dict[str, Any]] = []
    inventory_fields = list(template_inventory[0])
    for row, match, template in resolved_rows:
        event_id = match.group("event")
        process_id = match.group("process")
        fragment_id = match.group("fragment") or ""
        aggregation_key = relative_aggregation_key(template, event_id)
        normalized = dict(row)
        normalized.update(
            {
                "schema_version": 2,
                "measurement_contract_id": child["contract_id"],
                "measurement_contract_sha256": child_sha,
                "contract_relation": relation_type,
                "collection_required": "true",
                "hiptx_range": row["hiptx_range"],
                "planned_exact_range_name": row["hiptx_range"],
                "event_id": event_id,
                "forward_id": match.group("forward"),
                "layer_idx": match.group("layer"),
                "process_id": process_id,
                "fragment_id": fragment_id,
                "aggregation_key": aggregation_key,
            }
        )
        normalized_rows.append(normalized)

        inventory_row = dict(template)
        phase = row.get("phase", template.get("phase", ""))
        layer_type = row.get("layer_type", "")
        inventory_row.update(
            {
                "variant_scope": (
                    f"Workflow05 current child contract={child['contract_id']}"
                ),
                "phase": phase,
                "layer_or_layer_pattern": (
                    f"{event_id}; layer={match.group('layer')}; "
                    f"occurrence={row.get('occurrence', '0')}; "
                    f"layer_type={layer_type}; q_len={row.get('q_len', '')}; "
                    f"past_len={row.get('past_len', '')}; "
                    f"kv_len={row.get('kv_len', '')}"
                ),
                "process_id": process_id,
                "fragment_id": fragment_id,
                "aggregation_key": aggregation_key,
                "instrumented_file": str(
                    source_root
                    / "scripts/perf_trace/profile_qwen_same_input_layer.py"
                ),
                "nvtx_range_name": row["hiptx_range"],
                "range_parent": f"pra.layer.{event_id}.{phase}.{layer_type}",
                "range_guard_or_flag": (
                    "PRA_BACKEND_PERF_PROCESS_PROFILE=1; event_id in "
                    "PRA_BACKEND_PERF_PROCESS_TARGETS; exact name in "
                    "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS"
                ),
                "notes": (
                    template.get("notes", "")
                    + "; current exact marker derived from historical template "
                    + template.get("nvtx_range_name", "")
                ).strip("; "),
            }
        )
        current_inventory.append(inventory_row)

    plan_extra = [
        "measurement_contract_id",
        "measurement_contract_sha256",
        "contract_relation",
        "collection_required",
        "hiptx_range",
        "planned_exact_range_name",
        "fragment_id",
        "aggregation_key",
    ]
    plan_fields = list(plan_rows[0])
    for field in plan_extra:
        if field not in plan_fields:
            plan_fields.append(field)
    write_csv(outputs["plan"], normalized_rows, plan_fields)
    write_csv(outputs["inventory"], current_inventory, inventory_fields)

    environment = {
        "PRA_BACKEND_PERF_PROCESS_PROFILE": "1",
        "PRA_BACKEND_PERF_PROCESS_TARGETS": ",".join(event_ids),
        "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS": ",".join(markers),
        "CONTRACT_PATH": str(outputs["contract"]),
        "WORKFLOW05_NORMALIZED_SELECTION_PLAN": str(outputs["plan"]),
        "WORKFLOW05_CURRENT_PROCESS_INVENTORY": str(outputs["inventory"]),
        "WORKFLOW05_MEASUREMENT_CONTRACT_ID": child["contract_id"],
        "WORKFLOW05_MEASUREMENT_CONTRACT_SHA256": child_sha,
    }
    outputs["environment"].write_text(
        "".join(
            f"export {key}={shlex.quote(value)}\n"
            for key, value in environment.items()
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "status": "ready_for_selective_capture",
        "created_utc": created_utc,
        "contract_relation": relation_type,
        "historical_evidence_role": "planning_only",
        "current_measurement_role": "observed_after_capture_validation",
        "source_root": str(source_root),
        "selection_batch_id": selection_batch,
        "selected_event_count": len(event_ids),
        "selected_exact_process_range_count": len(markers),
        "selected_event_targets": event_ids,
        "selected_exact_process_range_targets": markers,
        "inputs": {
            "parent_contract": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
            },
            "baseline_run_metadata": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "historical_selection_plan": {
                "path": str(plan_path),
                "sha256": plan_sha,
            },
            "template_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
        },
        "outputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
        "invariants": {
            "predecessor_artifacts_modified": False,
            "exact_range_filter_required": True,
            "historical_and_current_timelines_separate": True,
            "historical_hardware_is_current_observation": False,
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
