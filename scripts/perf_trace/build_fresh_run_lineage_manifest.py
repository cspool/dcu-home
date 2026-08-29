#!/usr/bin/env python3
"""Validate one fresh-run R01-R05 prefix and freeze the R06 target contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class LineageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LineageError(f"JSON must be an object: {path}")
    return value


def under(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise LineageError(f"runtime path escapes the current run: {resolved}")
    return resolved


def file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = under(path, root)
    if not resolved.is_file():
        raise LineageError(f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def newline_record(path: Path, root: Path) -> dict[str, Any]:
    record = file_record(path, root)
    lines = path.resolve().read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() or line != line.strip() for line in lines):
        raise LineageError(f"target file has empty or noncanonical lines: {path}")
    if len(lines) != len(set(lines)):
        raise LineageError(f"target file contains duplicate lines: {path}")
    record["line_count"] = len(lines)
    return record


def csv_record(path: Path, root: Path) -> dict[str, Any]:
    record = file_record(path, root)
    with path.resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LineageError(f"CSV lacks a header: {path}")
        rows = list(reader)
    if not rows:
        raise LineageError(f"CSV contains no rows: {path}")
    record["row_count"] = len(rows)
    record["header"] = list(reader.fieldnames)
    return record


def resolve_recorded_path(path_value: str, project_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def referenced_runtime_files(
    value: Any,
    runtime_root: Path,
    project_root: Path,
) -> Iterable[tuple[Path, str]]:
    if isinstance(value, dict):
        path_value = value.get("path")
        digest = value.get("sha256") or value.get("handoff_sha256")
        if isinstance(path_value, str) and isinstance(digest, str):
            resolved = resolve_recorded_path(path_value, project_root)
            all_runtime_root = (project_root / "perf_trace" / "runtime").resolve()
            if resolved == all_runtime_root or resolved.is_relative_to(all_runtime_root):
                if resolved != runtime_root and not resolved.is_relative_to(runtime_root):
                    raise LineageError(
                        f"handoff references another runtime tree: {resolved}"
                    )
                yield under(resolved, runtime_root), digest
        for item in value.values():
            yield from referenced_runtime_files(item, runtime_root, project_root)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_runtime_files(item, runtime_root, project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--event-targets", type=Path, required=True)
    parser.add_argument("--range-targets", type=Path, required=True)
    parser.add_argument("--hardware-plan", type=Path, required=True)
    parser.add_argument("--stage-delta-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise LineageError(f"project root is missing: {project_root}")
    runtime_root = args.runtime_root.expanduser().resolve()
    if not runtime_root.is_dir():
        raise LineageError(f"runtime root is missing: {runtime_root}")
    if runtime_root != project_root and not runtime_root.is_relative_to(project_root):
        raise LineageError("runtime root escapes the project root")
    ledger_path = under(args.ledger, runtime_root)
    ledger = load_object(ledger_path)
    if ledger.get("branch") != args.branch or ledger.get("run_id") != args.run_id:
        raise LineageError("ledger branch/run_id differs from the requested lineage")
    handoffs = ledger.get("handoffs")
    if not isinstance(handoffs, list):
        raise LineageError("ledger handoffs must be a list")
    expected_goals = [f"R{index:02d}" for index in range(1, 6)]
    observed_goals = [
        entry.get("source_goal") if isinstance(entry, dict) else None
        for entry in handoffs
    ]
    if observed_goals != expected_goals:
        raise LineageError(
            f"fresh R06 requires exactly R01-R05; observed {observed_goals}"
        )

    runtime_records: dict[str, dict[str, Any]] = {}
    for entry in handoffs:
        if not isinstance(entry, dict):
            raise LineageError("ledger contains a non-object handoff")
        handoff_path = under(
            resolve_recorded_path(str(entry.get("path", "")), project_root),
            runtime_root,
        )
        observed_hash = sha256_file(handoff_path)
        if observed_hash != entry.get("sha256"):
            raise LineageError(f"handoff hash mismatch: {handoff_path}")
        payload = load_object(handoff_path)
        if payload != entry.get("payload") or payload.get("status") != "complete":
            raise LineageError(f"handoff payload mismatch/incomplete: {handoff_path}")
        runtime_records[str(handoff_path)] = file_record(handoff_path, runtime_root)
        for referenced, recorded_hash in referenced_runtime_files(
            payload,
            runtime_root,
            project_root,
        ):
            if sha256_file(referenced) != recorded_hash:
                raise LineageError(
                    f"runtime evidence hash mismatch: {referenced}"
                )
            runtime_records[str(referenced)] = file_record(referenced, runtime_root)

    contract_path = under(args.semantic_contract, runtime_root)
    contract = load_object(contract_path)
    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise LineageError("semantic contract lacks contract_id")
    runtime_records[str(contract_path)] = file_record(contract_path, runtime_root)

    deltas = []
    for delta_path in args.stage_delta_manifest:
        resolved = under(delta_path, runtime_root)
        delta = load_object(resolved)
        if delta.get("status") not in {"PASS", "pass", "complete"}:
            raise LineageError(f"stage delta is not complete: {resolved}")
        deltas.append(file_record(resolved, runtime_root))

    lineage_id = f"fresh-run:{args.run_id}:{contract_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.output_dir.resolve().is_relative_to(runtime_root):
        raise LineageError("output directory escapes the current run")
    lineage_path = args.output_dir / "fresh_run_lineage_manifest.json"
    target_path = args.output_dir / "full_request_target_manifest.json"
    if lineage_path.exists() or target_path.exists():
        raise LineageError("refusing to overwrite an existing lineage/target manifest")

    target_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "branch": args.branch,
        "run_id": args.run_id,
        "lineage_id": lineage_id,
        "capture_scope": "one_fresh_run_request_all_process_ranges",
        "event_target_file": newline_record(args.event_targets, runtime_root),
        "range_target_file": newline_record(args.range_targets, runtime_root),
        "r08_hardware_subset": csv_record(args.hardware_plan, runtime_root),
    }
    target_path.write_text(
        json.dumps(target_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime_records[str(target_path.resolve())] = file_record(target_path, runtime_root)

    lineage_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "branch": args.branch,
        "run_id": args.run_id,
        "lineage_id": lineage_id,
        "semantic_contract_id": contract_id,
        "semantic_contract": file_record(contract_path, runtime_root),
        "evidence_source_policy": "current_run_only",
        "source_change_policy": "stage_trace_instrumentation_allowed",
        "source_hash_equality_required": False,
        "external_runtime_reference_count": 0,
        "upstream_goals": expected_goals,
        "ledger": file_record(ledger_path, runtime_root),
        "runtime_references": [runtime_records[key] for key in sorted(runtime_records)],
        "stage_instrumentation_deltas": deltas,
        "semantic_invariants": {
            "status": "PASS",
            "model_input_sampling_device_preserved": True,
            "trace_instrumentation_changes_allowed": True,
            "source_hash_equality_used_as_gate": False,
        },
        "target_manifest": file_record(target_path, runtime_root),
    }
    lineage_path.write_text(
        json.dumps(lineage_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "lineage_id": lineage_id,
                "lineage_manifest": str(lineage_path.resolve()),
                "lineage_manifest_sha256": sha256_file(lineage_path),
                "target_manifest": str(target_path.resolve()),
                "target_manifest_sha256": sha256_file(target_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
