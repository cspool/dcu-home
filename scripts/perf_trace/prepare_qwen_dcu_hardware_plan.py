#!/usr/bin/env python3
"""Validate the current R01/R02/R03 bindings and freeze the R04 target plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


class PlanError(RuntimeError):
    """Fail-closed current-run plan error."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise PlanError(f"expected non-empty JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_contract_sha(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    recorded = str(payload.pop("contract_sha256", ""))
    computed = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if recorded != computed:
        raise PlanError("R01 contract canonical SHA mismatch")
    return computed


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise PlanError(f"missing JSON field: {'.'.join(keys)}")
        current = current[key]
    return current


def handoff_runtime_root(handoff: dict[str, Any]) -> str:
    value = handoff.get("runtime_root")
    if value:
        return str(value)
    handoff_output = handoff.get("handoff_output")
    if handoff_output:
        return str(Path(str(handoff_output)).resolve().parent.parent)
    raise PlanError("handoff lacks runtime_root and handoff_output")


def require_equal(label: str, *values: Any) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise PlanError(f"{label} mismatch: {values!r}")


def require_under(path: Path, root: Path, label: str) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PlanError(f"{label} escapes required root: {resolved}") from exc
    if "perf_trace_bk" in resolved.parts:
        raise PlanError(f"{label} points into archived evidence: {resolved}")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--r01-handoff", type=Path, required=True)
    parser.add_argument("--r02-handoff", type=Path, required=True)
    parser.add_argument("--r03-handoff", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r02-run-metadata", type=Path, required=True)
    parser.add_argument("--family-ledger", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--selection-batch-id", required=True)
    parser.add_argument(
        "--pmc-collection-policy",
        choices=("bounded_family_superset_exact_post_attribution",),
        required=True,
    )
    parser.add_argument("--kernel-name-filter", required=True)
    parser.add_argument(
        "--maximum-targeted-pmc-family-count", type=int, required=True
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_root = args.source_root.resolve()
    model_root = args.model_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_root.resolve()
    if project_root != Path(
        "/public/home/tangyu408/Qwen_DCU_Worker_0"
    ).resolve():
        raise PlanError("unexpected project root")
    require_under(source_root, project_root, "source root")
    require_under(runtime_root, project_root / "perf_trace", "runtime root")
    require_under(output_root, runtime_root, "R04 output root")
    expected_artifact_parent = runtime_root / "artifacts" / "R04"
    require_under(output_root, expected_artifact_parent, "R04 attempt output root")
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "r01_handoff": args.r01_handoff.resolve(),
        "r02_handoff": args.r02_handoff.resolve(),
        "r03_handoff": args.r03_handoff.resolve(),
        "contract": args.contract.resolve(),
        "r02_run_metadata": args.r02_run_metadata.resolve(),
        "family_ledger": args.family_ledger.resolve(),
        "trace_summary": args.trace_summary.resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise PlanError(f"missing current input {label}: {path}")
        require_under(path, project_root, label)

    r01 = load_json(paths["r01_handoff"])
    r02 = load_json(paths["r02_handoff"])
    r03 = load_json(paths["r03_handoff"])
    contract = load_json(paths["contract"])
    r02_meta = load_json(paths["r02_run_metadata"])
    trace_summary = load_json(paths["trace_summary"])
    for label, handoff in (("R01", r01), ("R02", r02), ("R03", r03)):
        require_equal(f"{label} status", handoff.get("status"), "complete")
        require_equal(f"{label} branch", handoff.get("branch"), args.branch)
        require_equal(f"{label} run_id", handoff.get("run_id"), args.run_id)
        require_equal(
            f"{label} runtime_root",
            str(Path(handoff_runtime_root(handoff)).resolve()),
            str(runtime_root),
        )

    contract_sha = canonical_contract_sha(contract)
    contract_id = str(contract["contract_id"])
    r01_source_revision = str(nested(contract, "source", "revision"))
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    require_equal(
        "contract source root",
        str(Path(nested(contract, "source", "source_root")).resolve()),
        str(source_root),
    )
    require_equal(
        "user model root",
        str(Path(nested(contract, "model", "model_root"))),
        str(args.model_root),
    )
    require_equal(
        "resolved model root",
        str(Path(nested(contract, "model", "resolved_model_root")).resolve()),
        str(model_root),
    )
    require_equal(
        "contract branch/run",
        contract["runtime_branch"],
        args.branch,
    )
    require_equal("contract run_id", contract["run_id"], args.run_id)
    require_equal(
        "R01 handoff contract",
        nested(r01, "contract", "contract_id"),
        contract_id,
    )
    require_equal(
        "R01 handoff contract SHA",
        nested(r01, "contract", "canonical_sha256"),
        contract_sha,
    )
    require_equal(
        "R02 parent contract",
        nested(r02, "contract", "contract_id"),
        contract_id,
    )
    require_equal(
        "R02 parent contract SHA",
        nested(r02, "contract", "canonical_sha256"),
        contract_sha,
    )
    require_equal(
        "R03 R01 contract",
        nested(r03, "same_input_parent", "contract_id"),
        contract_id,
    )
    require_equal(
        "R03 R01 contract SHA",
        nested(r03, "same_input_parent", "contract_canonical_sha256"),
        contract_sha,
    )
    require_equal(
        "R03 R02 parent contract",
        nested(r03, "component_source", "parent_contract_id"),
        contract_id,
    )

    require_equal("R02 metadata contract", r02_meta["contract_id"], contract_id)
    require_equal(
        "R02 metadata contract SHA", r02_meta["contract_sha256"], contract_sha
    )
    require_equal(
        "R02 metadata source root",
        str(Path(r02_meta["source_root"]).resolve()),
        str(source_root),
    )
    require_equal(
        "served model",
        r02_meta["served_model_name"],
        nested(contract, "model", "served_model_name"),
    )
    require_equal("process profile", r02_meta["process_profile"], "on")
    require_equal(
        "max new tokens",
        int(r02_meta["max_new_tokens"]),
        int(nested(contract, "same_input", "sampling", "max_new_tokens")),
    )
    require_equal(
        "warmup",
        int(r02_meta["warmup_iters"]),
        int(nested(contract, "same_input", "warmup_count")),
    )
    for field in (
        "dataset_row_raw_sha256",
        "prompt_text_sha256",
        "rendered_prompt_sha256",
        "rendered_prompt_token_count",
        "rendered_prompt_token_ids_sha256",
    ):
        require_equal(
            f"prompt {field}",
            r02_meta["same_input"][field],
            nested(contract, "same_input", "prompt", field),
        )
    sampling = nested(contract, "same_input", "sampling")
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "ignore_eos",
    ):
        require_equal(
            f"sampling {field}", r02_meta["sampling"][field], sampling[field]
        )
    config = contract["config"]
    expected_config = {
        "name": "qwen3.5-27b-vllm-pra-eager-gfx936",
        "dtype": "bfloat16",
        "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
        "attention_use_prefill_decode_attention": False,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.95,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "vllm_enable_v1_multiprocessing": 0,
    }
    for field, expected in expected_config.items():
        require_equal(f"frozen config {field}", config[field], expected)
    require_equal(
        "physical device",
        int(nested(contract, "device", "physical_device_id")),
        1,
    )
    require_equal(
        "R02 device",
        int(nested(r02, "contract", "device", "physical_device_id")),
        1,
    )
    require_equal(
        "logical device",
        int(nested(contract, "device", "logical_device_id")),
        0,
    )
    require_equal(
        "visible device",
        r02_meta["runtime"]["HIP_VISIBLE_DEVICES"],
        "1",
    )

    r02_primary = nested(r02, "primary_outputs")
    r02_db_record = r02_primary.get("queryable_process_trace") or r02_primary.get(
        "queryable_trace"
    )
    if not isinstance(r02_db_record, dict):
        raise PlanError("R02 handoff lacks a queryable process trace")
    r02_db_path = Path(str(r02_db_record["path"]))
    r02_db_sha = str(r02_db_record["sha256"])
    require_equal("R02 DB file SHA", sha256_file(r02_db_path), r02_db_sha)
    require_equal(
        "R03 component DB path",
        str(Path(nested(r03, "component_source", "process_db", "path")).resolve()),
        str(r02_db_path.resolve()),
    )
    require_equal(
        "R03 component DB SHA",
        nested(r03, "component_source", "process_db", "sha256"),
        r02_db_sha,
    )
    inventory_path = Path(
        nested(r02, "primary_outputs", "process_range_inventory_csv", "path")
    )
    inventory_sha = nested(
        r02, "primary_outputs", "process_range_inventory_csv", "sha256"
    )
    require_equal("R02 inventory file SHA", sha256_file(inventory_path), inventory_sha)
    require_equal(
        "R03 inventory SHA",
        nested(r03, "component_source", "process_inventory", "sha256"),
        inventory_sha,
    )
    require_equal("non-replay trace status", trace_summary["status"], "PASS")
    require_equal(
        "non-replay trace DB SHA",
        trace_summary["source_db_sha256"],
        r02_db_sha,
    )
    require_equal(
        "non-replay trace inventory SHA",
        trace_summary["inventory_sha256"],
        inventory_sha,
    )
    require_equal(
        "non-replay trace contract",
        trace_summary["contract_id"],
        contract_id,
    )
    require_equal(
        "non-replay trace contract SHA",
        trace_summary["contract_sha256"],
        contract_sha,
    )

    family_rows = read_csv(paths["family_ledger"])
    family_ledger_sha256 = sha256_file(paths["family_ledger"])
    required_family_fields = {
        "parent_layer_range",
        "forward_id",
        "layer",
        "event_id",
        "stage",
        "first_kernel_launch_order_in_parent",
        "process_gpu_order",
        "process_gpu_start_offset_us",
        "process_id",
        "process_title",
        "fragment_id",
        "aggregation_key",
        "first_kernel_launch_order_in_process",
        "matched_kernel_family",
        "kernel_family_instance_count",
        "hipprof_kernel_duration_ms",
        "hipprof_kernel_name_examples",
        "gpu_order_basis",
    }
    if not family_rows or not required_family_fields.issubset(family_rows[0]):
        raise PlanError("non-replay family ledger lacks required schema")
    family_keys: set[tuple[str, str, str]] = set()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in family_rows:
        key = (row["event_id"], row["stage"], row["matched_kernel_family"])
        if key in family_keys:
            raise PlanError(f"duplicate non-replay family key: {key}")
        family_keys.add(key)
        grouped[(row["event_id"], row["stage"])].append(row)
    kernel_family_names = {
        row["matched_kernel_family"]
        for row in family_rows
        if row["matched_kernel_family"] != "no_kernel"
    }
    if args.maximum_targeted_pmc_family_count <= 0:
        raise PlanError("maximum targeted PMC family count must be positive")
    if len(kernel_family_names) > args.maximum_targeted_pmc_family_count:
        raise PlanError(
            "current family denominator exceeds the authorized PMC family cap: "
            f"{len(kernel_family_names)} > "
            f"{args.maximum_targeted_pmc_family_count}"
        )
    kernel_name_filter = args.kernel_name_filter.strip()
    if not kernel_name_filter or "\n" in kernel_name_filter or "\r" in kernel_name_filter:
        raise PlanError("kernel-name filter must be one non-empty literal")
    uncovered_filter_rows = [
        (row["event_id"], row["stage"], row["matched_kernel_family"])
        for row in family_rows
        if row["matched_kernel_family"] != "no_kernel"
        and not any(
            kernel_name_filter in name
            for name in row["hipprof_kernel_name_examples"].split(";")
            if name
        )
    ]
    if uncovered_filter_rows:
        raise PlanError(
            "the single literal kernel-name filter does not cover every "
            f"expected family row: {uncovered_filter_rows[:10]}"
        )
    expected_process_count = int(
        trace_summary["checks"]["expected_process_marker_count"]
    )
    if len(grouped) != expected_process_count:
        raise PlanError("family ledger does not cover every process marker")

    selection_rows: list[dict[str, Any]] = []
    for (event_id, stage), rows in grouped.items():
        first = rows[0]
        invariant_fields = (
            "parent_layer_range",
            "forward_id",
            "layer",
            "process_gpu_order",
            "process_gpu_start_offset_us",
            "process_id",
            "process_title",
            "fragment_id",
            "aggregation_key",
            "gpu_order_basis",
        )
        for field in invariant_fields:
            if any(row[field] != first[field] for row in rows[1:]):
                raise PlanError(f"process-family invariant drift: {event_id}.{stage}.{field}")
        no_kernel = [row for row in rows if row["matched_kernel_family"] == "no_kernel"]
        kernel_rows = [
            row for row in rows if row["matched_kernel_family"] != "no_kernel"
        ]
        if bool(no_kernel) == bool(kernel_rows) or len(no_kernel) > 1:
            raise PlanError(f"invalid kernel/no-kernel state: {event_id}.{stage}")
        expected_families = [row["matched_kernel_family"] for row in kernel_rows]
        selection_rows.append(
            {
                "capture_target_id": f"{event_id}:{stage}",
                "parent_layer_range": first["parent_layer_range"],
                "forward_id": first["forward_id"],
                "layer": first["layer"],
                "event_id": event_id,
                "stage": stage,
                "first_kernel_launch_order_in_parent": (
                    min(
                        int(row["first_kernel_launch_order_in_parent"])
                        for row in kernel_rows
                    )
                    if kernel_rows
                    else ""
                ),
                "process_gpu_order": first["process_gpu_order"],
                "process_gpu_start_offset_us": first["process_gpu_start_offset_us"],
                "process_id": first["process_id"],
                "process_title": first["process_title"],
                "fragment_id": first["fragment_id"],
                "aggregation_key": first["aggregation_key"],
                "hiptx_range": f"pra.fx_process.{event_id}.{stage}",
                "expected_kernel_families": ";".join(expected_families),
                "expected_kernel_family_row_count": len(kernel_rows),
                "expected_kernel_instance_count": sum(
                    int(row["kernel_family_instance_count"]) for row in kernel_rows
                ),
                "expected_non_replay_hipprof_duration_ms": sum(
                    float(row["hipprof_kernel_duration_ms"]) for row in kernel_rows
                ),
                "gpu_order_basis": first["gpu_order_basis"],
                "collection_required": bool_text(bool(kernel_rows)),
                "expected_no_kernel": bool_text(bool(no_kernel)),
                "selection_mode": "all_same_run_r02_representatives",
                "collection_status": "pending" if kernel_rows else "no_kernel",
                "selection_batch_id": args.selection_batch_id,
                "capture_batch_id": args.selection_batch_id,
                "contract_relation": "current_measurement",
                "measurement_contract_id": contract_id,
                "measurement_contract_sha256": contract_sha,
                "pmc_collection_policy": args.pmc_collection_policy,
                "kernel_name_filter": kernel_name_filter,
                "run_id": args.run_id,
                "branch": args.branch,
                "contract_id": contract_id,
                "contract_sha256": contract_sha,
                "source_revision": source_revision,
                "r02_non_replay_db_sha256": r02_db_sha,
                "non_replay_family_ledger_sha256": family_ledger_sha256,
            }
        )
    selection_rows.sort(
        key=lambda row: (
            int(row["forward_id"]),
            int(row["layer"]),
            int(row["process_gpu_order"]),
        )
    )
    by_parent: dict[str, list[int]] = defaultdict(list)
    for row in selection_rows:
        by_parent[row["parent_layer_range"]].append(int(row["process_gpu_order"]))
    for parent, orders in by_parent.items():
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise PlanError(f"non-contiguous process_gpu_order for {parent}")

    selection_fields = list(selection_rows[0])
    selection_path = output_root / "dcu_process_selection_plan.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields)
        writer.writeheader()
        writer.writerows(selection_rows)

    # The full-request contract uses newline files so the complete target set
    # is never truncated by an environment-variable size limit.  Preserve the
    # already-frozen process order for exact ranges and first-seen event order
    # for parent event targets.
    event_ids = list(dict.fromkeys(row["event_id"] for row in selection_rows))
    process_targets_path = output_root / "full_request_process_targets.txt"
    range_targets_path = output_root / "full_request_process_range_targets.txt"
    process_targets_path.write_text(
        "\n".join(event_ids) + "\n", encoding="utf-8"
    )
    range_targets_path.write_text(
        "\n".join(row["hiptx_range"] for row in selection_rows) + "\n",
        encoding="utf-8",
    )

    launch_targets = [
        row for row in selection_rows if row["collection_required"] == "true"
    ]
    no_kernel_targets = [
        row for row in selection_rows if row["expected_no_kernel"] == "true"
    ]
    expected_kernel_rows = [
        row for row in family_rows if row["matched_kernel_family"] != "no_kernel"
    ]
    expected_no_kernel_rows = [
        row for row in family_rows if row["matched_kernel_family"] == "no_kernel"
    ]
    snapshot = {
        "schema_version": 1,
        "runtime_goal": "R04",
        "status": "selection_plan_ready",
        "run": {
            "branch": args.branch,
            "run_id": args.run_id,
            "runtime_root": str(runtime_root),
            "runtime_artifact_root": str(output_root),
        },
        "contract": {
            "contract_id": contract_id,
            "canonical_sha256": contract_sha,
            "file_sha256": sha256_file(paths["contract"]),
            "path": str(paths["contract"]),
            "source_revision": source_revision,
            "r01_source_revision": r01_source_revision,
            "source_revision_matches_r01": (
                source_revision == r01_source_revision
            ),
            "source_hash_equality_required": False,
            "source_root": str(source_root),
            "model_root": str(args.model_root),
            "resolved_model_root": str(model_root),
            "served_model_name": nested(contract, "model", "served_model_name"),
            "runtime": "vLLM/PRA",
            "accelerator": "ROCm/DCU/HIP",
            "config": config,
            "same_input": contract["same_input"],
            "device": contract["device"],
        },
        "upstream_bindings": {
            "r01_handoff": {
                "path": str(paths["r01_handoff"]),
                "sha256": sha256_file(paths["r01_handoff"]),
            },
            "r02_handoff": {
                "path": str(paths["r02_handoff"]),
                "sha256": sha256_file(paths["r02_handoff"]),
            },
            "r03_handoff": {
                "path": str(paths["r03_handoff"]),
                "sha256": sha256_file(paths["r03_handoff"]),
            },
            "r02_non_replay_db": {
                "path": str(r02_db_path.resolve()),
                "sha256": r02_db_sha,
            },
            "r02_inventory": {
                "path": str(inventory_path.resolve()),
                "sha256": inventory_sha,
            },
            "non_replay_family_ledger": {
                "path": str(paths["family_ledger"]),
                "sha256": family_ledger_sha256,
                "role": (
                    "schema-equivalent same-run R02 family/order table "
                    "materialized from the exact R02 non-replay DB"
                ),
            },
            "non_replay_trace_summary": {
                "path": str(paths["trace_summary"]),
                "sha256": sha256_file(paths["trace_summary"]),
            },
        },
        "expected_denominator": {
            "representative_parent_layers": len(by_parent),
            "process_fragment_targets": len(selection_rows),
            "launch_owning_capture_targets": len(launch_targets),
            "no_kernel_process_targets": len(no_kernel_targets),
            "kernel_family_rows": len(expected_kernel_rows),
            "no_kernel_family_rows": len(expected_no_kernel_rows),
            "all_projection_rows": len(family_rows),
        },
        "selection_plan": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selection_mode": "all_same_run_r02_representatives",
            "filtered": False,
            "collection_status": "pending",
            "process_target_transport": "newline_file",
            "process_targets": {
                "path": str(process_targets_path),
                "sha256": sha256_file(process_targets_path),
                "rows": len(event_ids),
            },
            "exact_process_range_targets": {
                "path": str(range_targets_path),
                "sha256": sha256_file(range_targets_path),
                "rows": len(selection_rows),
            },
        },
        "pmc_collection": {
            "policy": args.pmc_collection_policy,
            "capture_batch_count_per_mode": 1,
            "capture_batch_id": args.selection_batch_id,
            "one_literal_kernel_name_filter_per_capture_batch": True,
            "kernel_name_filter": kernel_name_filter,
            "kernel_name_filter_coverage": {
                "expected_kernel_family_rows": len(expected_kernel_rows),
                "covered_kernel_family_rows": len(expected_kernel_rows),
                "unique_kernel_families": len(kernel_family_names),
                "maximum_targeted_pmc_family_count": (
                    args.maximum_targeted_pmc_family_count
                ),
            },
        },
        "timing_boundary": {
            "timing_source": "workflow02_non_replay_family_row",
            "hardware_join_key": "event_id+stage+matched_kernel_family",
            "pmc_replay_timing_used_as_latency": False,
        },
        "archive_used_as_current_evidence": False,
    }
    snapshot_path = output_root / "R04_RUN_CONTRACT.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "selection_plan": str(selection_path),
        "selection_plan_sha256": sha256_file(selection_path),
        "process_targets": str(process_targets_path),
        "process_targets_sha256": sha256_file(process_targets_path),
        "exact_process_range_targets": str(range_targets_path),
        "exact_process_range_targets_sha256": sha256_file(range_targets_path),
        "run_contract": str(snapshot_path),
        "expected_denominator": snapshot["expected_denominator"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
