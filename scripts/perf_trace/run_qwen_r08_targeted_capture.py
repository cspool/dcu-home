#!/usr/bin/env python3
"""Run every fresh R08 batch/mode serially on physical DCU 1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CaptureError(RuntimeError):
    """Fail-closed capture error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise CaptureError(f"expected non-empty JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "size_bytes": path.stat().st_size,
    }


def require_record(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise CaptureError(f"{label} is missing or changed: {path}")
    return path


def require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CaptureError(f"{label} is outside the R08 artifact root: {path}") from exc


def validate_completed_capture(
    record: dict[str, Any],
    planned: dict[str, Any],
    artifact_root: Path,
) -> None:
    """Revalidate every durable accepted capture before a recovery resume."""
    identity_fields = ("capture_id", "capture_batch_id", "selection_rank", "mode")
    mismatches = {
        field: {"observed": record.get(field), "expected": planned.get(field)}
        for field in identity_fields
        if record.get(field) != planned.get(field)
    }
    if record.get("status") != "complete" or mismatches:
        raise CaptureError(
            f"completed capture identity drift for {planned.get('capture_id')}: {mismatches}"
        )
    for label in (
        "preflight",
        "launcher_preflight",
        "metadata",
        "runtime_events",
        "raw_db",
        "raw_pmc",
        "provenance",
        "driver_log",
        "trace_summary",
        "pmc_summary",
        "hardware_kernel_metrics",
        "discarded_superset_matches",
        "analysis_compaction_manifest",
    ):
        path = require_record(record.get(label, {}), f"completed capture {label}")
        require_under(path, artifact_root, f"completed capture {label}")
    summary = load_json(Path(record["pmc_summary"]["path"]))
    if (
        summary.get("status") != "PASS"
        or summary.get("capture_batch_id") != planned["capture_batch_id"]
        or summary.get("kind") != planned["mode"]
        or float(summary.get("name_order_match_rate", 0.0)) < 0.99
        or int(summary.get("ambiguous_pair_count", -1)) != 0
        or float(summary.get("selected_exact_attribution_rate", 0.0)) != 1.0
    ):
        raise CaptureError(
            f"completed capture no longer passes strict attribution: {planned['capture_id']}"
        )


def artifact_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def compact_replay_derivable_tables(analysis_dir: Path) -> Path:
    """Losslessly gzip large normalized tables after strict PMC attribution."""
    records: list[dict[str, Any]] = []
    for name in (
        "kernels.csv",
        "pmc_blocks.csv",
        "pmc_name_order_matches.csv",
        "discarded_superset_matches.csv",
    ):
        source = analysis_dir / name
        if not source.is_file():
            raise CaptureError(f"expected analysis table is missing: {source}")
        compressed = source.with_suffix(source.suffix + ".gz")
        original_sha = sha256_file(source)
        with source.open("rb") as input_handle, compressed.open("xb") as output_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=output_handle,
                mtime=0,
            ) as gzip_handle:
                shutil.copyfileobj(input_handle, gzip_handle)
        digest = hashlib.sha256()
        with gzip.open(compressed, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != original_sha:
            raise CaptureError(f"lossless analysis compression failed: {source}")
        source.unlink()
        records.append(
            {
                "original_path": str(source.resolve()),
                "original_sha256": original_sha,
                "compressed_path": str(compressed.resolve()),
                "compressed_sha256": sha256_file(compressed),
                "compressed_size_bytes": compressed.stat().st_size,
                "lossless": True,
                "reconstructable_from_raw_db_or_pmc": True,
            }
        )
    manifest_path = analysis_dir / "analysis_compaction_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "method": "deterministic_gzip_mtime_zero",
            "raw_db_and_raw_pmc_retained": True,
            "tables": records,
        },
    )
    return manifest_path


def live_preflight(expected_unique_id: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "/opt/hyhal/bin/hy-smi",
            "-d",
            "1",
            "--showuniqueid",
            "--showproductname",
            "--showserial",
            "--showuse",
            "--showmemuse",
            "--json",
        ],
        text=True,
    )
    payload = json.loads(output)
    card = payload.get("card1", {})
    if (
        card.get("Unique ID") != expected_unique_id
        or card.get("Card Series") != "BW"
    ):
        raise CaptureError(f"physical DCU 1 identity drift: {card}")
    use = float(card.get("HCU use (%)", "nan"))
    memory_use = float(card.get("HCU memory use (%)", "nan"))
    if use != 0.0 or memory_use != 0.0:
        raise CaptureError(
            f"concurrent or residual DCU 1 use detected: use={use}, memory={memory_use}"
        )
    return {
        "schema_version": 1,
        "status": "idle_verified",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "physical_device_id": 1,
        "card": card,
        "raw_stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def run_with_heartbeat(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    capture_id: str,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while True:
            return_code = process.poll()
            elapsed = time.monotonic() - started
            if return_code is not None:
                return return_code, elapsed
            print(
                json.dumps(
                    {
                        "status": "capture_running",
                        "capture_id": capture_id,
                        "elapsed_seconds": round(elapsed, 1),
                    }
                ),
                flush=True,
            )
            time.sleep(20)


def exact_capture_metadata(
    metadata: dict[str, Any],
    contract: dict[str, Any],
    capture: dict[str, Any],
    lineage_id: str,
) -> None:
    target_range = require_record(
        capture["process_range_targets"], "process range targets"
    ).read_text(encoding="utf-8").strip()
    target_event = require_record(
        capture["process_targets"], "process targets"
    ).read_text(encoding="utf-8").strip()
    checks = {
        "lineage_id": (metadata.get("lineage_id"), lineage_id),
        "contract_id": (metadata.get("contract_id"), contract["contract_id"]),
        "contract_sha256": (
            metadata.get("contract_sha256"),
            contract["contract_sha256"],
        ),
        "process_profile": (metadata.get("process_profile"), "on"),
        "max_new_tokens": (
            int(metadata.get("max_new_tokens", -1)),
            int(contract["expected_max_new_tokens"]),
        ),
        "warmup_iters": (
            int(metadata.get("warmup_iters", -1)),
            int(contract["expected_warmup_iters"]),
        ),
        "same_input": (metadata.get("same_input"), contract["expected_same_input"]),
        "sampling": (metadata.get("sampling"), contract["expected_sampling"]),
        "measured_result": (
            metadata.get("measured_result"),
            contract["expected_measured_result"],
        ),
        "physical HIP visibility": (
            metadata.get("runtime", {}).get("HIP_VISIBLE_DEVICES"),
            "1",
        ),
        "physical CUDA visibility": (
            metadata.get("runtime", {}).get("CUDA_VISIBLE_DEVICES"),
            "1",
        ),
        "profile kind": (
            metadata.get("profiler_session_control", {}).get("profile_kind"),
            capture["mode"],
        ),
        "replay latency flag": (
            metadata.get("request_synchronized_latency_is_replay_distorted"),
            True,
        ),
        "exact range filter": (
            metadata.get("exact_process_range_filter_enabled"),
            True,
        ),
        "exact range count": (
            int(metadata.get("expected_process_range_count", -1)),
            1,
        ),
        "exact range targets": (
            metadata.get("exact_process_range_targets"),
            [target_range],
        ),
        "emitted process ranges": (
            metadata.get("emitted_process_ranges"),
            [target_range],
        ),
        "process targets": (metadata.get("process_targets"), [target_event]),
    }
    mismatches = {
        label: {"observed": observed, "expected": expected}
        for label, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise CaptureError(f"capture metadata semantic drift: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fresh serial R08 captures.")
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a failed-closed run from a separately frozen recovery contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_contract_path = args.run_contract.resolve()
    capture_manifest_path = args.capture_manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    contract = load_json(run_contract_path)
    manifest = load_json(capture_manifest_path)
    if (
        contract.get("runtime_goal") != "R08"
        or contract.get("status") != "ready"
        or manifest.get("status") != "ready"
        or manifest.get("lineage_id") != contract.get("lineage_id")
        or manifest.get("physical_device_id") != 1
        or manifest.get("serial_gpu_collection_required") is not True
    ):
        raise CaptureError("invalid R08 run contract or capture manifest")
    if file_record(capture_manifest_path)["sha256"] != contract["capture_manifest"]["sha256"]:
        raise CaptureError("capture manifest changed after contract freeze")
    source_lineage_path = require_record(contract["source_lineage"], "source lineage")
    source_lineage = load_json(source_lineage_path)
    for role, record in source_lineage["tools"].items():
        require_record(record, f"frozen tool {role}")

    lineage_path = require_record(
        source_lineage["inputs"]["r06_lineage"], "R06 lineage"
    )
    contract_path = Path(contract["contract_path"]).resolve()
    launcher = Path(source_lineage["tools"]["capture_launcher"]["path"])
    trace_analyzer = Path(source_lineage["tools"]["trace_analyzer"]["path"])
    pmc_analyzer = Path(source_lineage["tools"]["pmc_analyzer"]["path"])
    progress_path = artifact_root / "capture_progress.json"
    summary_path = artifact_root / "CAPTURE_RUN_SUMMARY.json"
    if args.resume:
        if summary_path.exists() or not progress_path.is_file():
            raise CaptureError("recovery requires failed progress and no completion summary")
        progress = load_json(progress_path)
        completed = progress.get("captures", [])
        completed_count = len(completed)
        if (
            progress.get("status") != "failed"
            or progress.get("lineage_id") != contract["lineage_id"]
            or int(progress.get("completed_capture_count", -1)) != completed_count
            or int(progress.get("capture_count", -1)) != len(manifest["captures"])
            or not 0 <= completed_count < len(manifest["captures"])
            or not contract.get("recovery_id")
        ):
            raise CaptureError("invalid failed progress for frozen R08 recovery")
        for accepted, planned in zip(
            completed,
            manifest["captures"][:completed_count],
            strict=True,
        ):
            validate_completed_capture(accepted, planned, artifact_root)
        expected_failed = manifest.get("superseded_failed_capture_id")
        if expected_failed != progress.get("failed_capture_id"):
            raise CaptureError("recovery manifest does not bind the failed capture")
        progress.setdefault("failed_attempts", []).append(
            {
                "capture_id": progress.get("failed_capture_id"),
                "failure_reason": progress.get("failure_reason"),
                "profiling_wall_seconds_inclusive_at_failure": progress.get(
                    "profiling_wall_seconds"
                ),
                "recovery_id": contract["recovery_id"],
                "recovery_evidence": contract.get("recovery_evidence"),
            }
        )
        progress["status"] = "running"
        progress["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        progress["active_run_contract"] = file_record(run_contract_path)
        progress["active_capture_manifest"] = file_record(capture_manifest_path)
        progress.pop("failed_capture_id", None)
        progress.pop("failure_reason", None)
        first_pending_index = completed_count
        atomic_json(progress_path, progress)
    else:
        if summary_path.exists() or progress_path.exists():
            raise CaptureError("refusing pre-existing R08 capture state")
        progress = {
            "schema_version": 1,
            "status": "running",
            "lineage_id": contract["lineage_id"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "capture_count": len(manifest["captures"]),
            "completed_capture_count": 0,
            "profiling_wall_seconds": 0.0,
            "captures": [],
        }
        first_pending_index = 0
        atomic_json(progress_path, progress)

    for index, capture in enumerate(manifest["captures"], 1):
        if index <= first_pending_index:
            continue
        for role, record in source_lineage["tools"].items():
            require_record(record, f"frozen tool {role}")
        if progress["profiling_wall_seconds"] >= float(
            contract["maximum_profiling_wall_time_seconds"]
        ):
            raise CaptureError("profiling wall-time cap reached before next capture")
        current_bytes = artifact_bytes(artifact_root)
        if current_bytes >= int(contract["maximum_trace_bundle_bytes"]):
            raise CaptureError("artifact byte cap reached before next capture")

        output_dir = Path(capture["output_dir"]).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise CaptureError(f"refusing non-empty fresh capture root: {output_dir}")
        preflight = live_preflight(contract["device_unique_id"])
        preflight_path = (
            artifact_root
            / "preflight"
            / (
                f"{index:03d}_{capture['capture_batch_id']}_{capture['mode']}"
                f"_attempt-{int(capture.get('attempt', 1)):03d}.json"
            )
        )
        if preflight_path.exists():
            raise CaptureError(f"refusing existing fresh preflight: {preflight_path}")
        atomic_json(preflight_path, preflight)

        selection_path = require_record(capture["selection_plan"], "selection plan")
        event_target_path = require_record(capture["process_targets"], "event targets")
        range_target_path = require_record(
            capture["process_range_targets"], "range targets"
        )
        inventory_path = require_record(capture["process_inventory"], "process inventory")
        env = dict(os.environ)
        env.update(
            {
                "ROOT_DIR": contract["source_root"],
                "RUNTIME_ARTIFACT_ROOT": str(artifact_root),
                "CONTRACT_PATH": str(contract_path),
                "LINEAGE_MANIFEST": str(lineage_path),
                "WORKFLOW05_R08_FRESH_LINEAGE_REQUIRED": "1",
                "MODEL_ROOT": str(contract["model_root"]),
                "SERVED_MODEL_NAME": str(contract["served_model_name"]),
                "DCU_DEVICE": "1",
                "PRA_BACKEND_PERF_PROCESS_TARGETS": "",
                "PRA_BACKEND_PERF_PROCESS_TARGETS_FILE": str(event_target_path),
                "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS": "",
                "PRA_BACKEND_PERF_PROCESS_RANGE_TARGETS_FILE": str(range_target_path),
                "PRA_BACKEND_TUNABLEOP_PROFILE_SHA256_OVERRIDE": contract[
                    "current_tunable_profile"
                ]["sha256"],
                "WORKFLOW05_EXACT_PROCESS_FILTER_REQUIRED": "1",
                "WORKFLOW05_PMC_COLLECTION_POLICY": (
                    "bounded_family_superset_exact_post_attribution"
                ),
                "PRA_HIPPROF_KERNEL_NAME_FILTER": capture[
                    "kernel_name_filter_literal"
                ],
                "WORKFLOW05_PMC_CAPTURE_BATCH_ID": capture["capture_batch_id"],
                "WORKFLOW05_TARGET_SELECTION_PLAN": str(selection_path),
            }
        )
        log_path = (
            artifact_root
            / "logs"
            / (
                f"{index:03d}_{capture['tag']}"
                f"_attempt-{int(capture.get('attempt', 1)):03d}.driver.log"
            )
        )
        print(
            json.dumps(
                {
                    "status": "capture_start",
                    "index": index,
                    "total": len(manifest["captures"]),
                    "capture_id": capture["capture_id"],
                    "artifact_bytes_before": current_bytes,
                }
            ),
            flush=True,
        )
        return_code, wall_seconds = run_with_heartbeat(
            [str(launcher), capture["mode"], str(output_dir), capture["tag"]],
            env=env,
            log_path=log_path,
            capture_id=capture["capture_id"],
        )
        progress["profiling_wall_seconds"] += wall_seconds
        if return_code != 0:
            progress["status"] = "failed"
            progress["failed_capture_id"] = capture["capture_id"]
            progress["failure_reason"] = f"launcher exit code {return_code}"
            atomic_json(progress_path, progress)
            raise CaptureError(progress["failure_reason"])

        metadata_path = output_dir / f"{capture['tag']}.json"
        events_path = output_dir / f"{capture['tag']}.layer_events.runtime.jsonl"
        provenance_path = output_dir / "tool_provenance.txt"
        raw_dbs = sorted((output_dir / "raw").glob("*.db"))
        raw_metrics = sorted((output_dir / "raw").glob("*.txt"))
        if (
            len(raw_dbs) != 1
            or len(raw_metrics) != 1
            or not metadata_path.is_file()
            or not events_path.is_file()
            or not provenance_path.is_file()
        ):
            raise CaptureError(f"capture output set is incomplete: {capture['capture_id']}")
        metadata = load_json(metadata_path)
        exact_capture_metadata(metadata, contract, capture, contract["lineage_id"])
        launcher_preflight = load_json(output_dir / "device_preflight.json")
        card = launcher_preflight.get("card1", {})
        if (
            card.get("Unique ID") != contract["device_unique_id"]
            or card.get("Card Series") != "BW"
            or float(card.get("HCU use (%)", "nan")) != 0.0
            or float(card.get("HCU memory use (%)", "nan")) != 0.0
        ):
            raise CaptureError("launcher-immediate device preflight is not idle DCU 1")

        analysis_dir = output_dir / "analysis"
        trace_command = [
            sys.executable,
            str(trace_analyzer),
            "--db",
            str(raw_dbs[0]),
            "--runtime-events",
            str(events_path),
            "--inventory",
            str(inventory_path),
            "--output-dir",
            str(analysis_dir),
            "--contract-id",
            contract["contract_id"],
            "--contract-sha256",
            contract["contract_sha256"],
            "--capture-mode",
            capture["mode"],
            "--expected-device",
            "1",
            "--selection-plan",
            str(selection_path),
            "--process-overlap-mode",
            "none",
            "--bounded-kernel-name-filter",
            capture["kernel_name_filter_literal"],
        ]
        subprocess.run(trace_command, check=True, stdout=subprocess.DEVNULL)
        pmc_command = [
            sys.executable,
            str(pmc_analyzer),
            "--analysis-dir",
            str(analysis_dir),
            "--metrics-file",
            str(raw_metrics[0]),
            "--selection-plan",
            str(selection_path),
            "--kind",
            capture["mode"],
            "--minimum-match-rate",
            str(contract["minimum_name_order_match_rate"]),
            "--collection-policy",
            "bounded-family-superset",
            "--kernel-name-filter",
            capture["kernel_name_filter_literal"],
            "--capture-batch-id",
            capture["capture_batch_id"],
        ]
        subprocess.run(pmc_command, check=True, stdout=subprocess.DEVNULL)
        trace_summary_path = analysis_dir / "process_trace_summary.json"
        pmc_summary_path = analysis_dir / "hardware_metric_summary.json"
        trace_summary = load_json(trace_summary_path)
        pmc_summary = load_json(pmc_summary_path)
        if (
            trace_summary.get("status") != "PASS"
            or pmc_summary.get("status") != "PASS"
            or float(pmc_summary.get("name_order_match_rate", 0.0))
            < float(contract["minimum_name_order_match_rate"])
            or int(pmc_summary.get("ambiguous_pair_count", -1)) != 0
            or int(pmc_summary.get("missing_selected_kernel_count", -1)) != 0
            or int(pmc_summary.get("covered_selected_family_count", -1)) != 1
            or pmc_summary.get("final_process_family_attribution_exact") is not True
            or pmc_summary.get("pmc_is_latency_evidence") is not False
        ):
            raise CaptureError(f"strict capture analysis failed: {capture['capture_id']}")
        compaction_manifest_path = compact_replay_derivable_tables(analysis_dir)
        discarded_superset_path = analysis_dir / "discarded_superset_matches.csv.gz"

        capture_record = {
            "capture_id": capture["capture_id"],
            "capture_batch_id": capture["capture_batch_id"],
            "selection_rank": capture["selection_rank"],
            "mode": capture["mode"],
            "status": "complete",
            "profiling_wall_seconds": wall_seconds,
            "preflight": file_record(preflight_path),
            "launcher_preflight": file_record(output_dir / "device_preflight.json"),
            "metadata": file_record(metadata_path),
            "runtime_events": file_record(events_path),
            "raw_db": file_record(raw_dbs[0]),
            "raw_pmc": file_record(raw_metrics[0]),
            "provenance": file_record(provenance_path),
            "driver_log": file_record(log_path),
            "trace_summary": file_record(trace_summary_path),
            "pmc_summary": file_record(pmc_summary_path),
            "hardware_kernel_metrics": file_record(
                analysis_dir / "hardware_kernel_metrics.csv"
            ),
            "discarded_superset_matches": file_record(
                discarded_superset_path
            ),
            "analysis_compaction_manifest": file_record(
                compaction_manifest_path
            ),
            "name_order_match_rate": pmc_summary["name_order_match_rate"],
            "selected_exact_attribution_rate": pmc_summary[
                "selected_exact_attribution_rate"
            ],
            "discarded_superset_match_count": pmc_summary[
                "discarded_superset_match_count"
            ],
        }
        progress["captures"].append(capture_record)
        progress["completed_capture_count"] = len(progress["captures"])
        progress["artifact_bytes"] = artifact_bytes(artifact_root)
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "status": "capture_complete",
                    "index": index,
                    "total": len(manifest["captures"]),
                    "capture_id": capture["capture_id"],
                    "wall_seconds": round(wall_seconds, 1),
                    "name_order_match_rate": pmc_summary["name_order_match_rate"],
                    "artifact_bytes": progress["artifact_bytes"],
                }
            ),
            flush=True,
        )

    progress["status"] = "complete"
    progress["completed_utc"] = datetime.now(timezone.utc).isoformat()
    progress["artifact_bytes"] = artifact_bytes(artifact_root)
    if progress["profiling_wall_seconds"] > float(
        contract["maximum_profiling_wall_time_seconds"]
    ):
        raise CaptureError("completed captures exceed profiling wall-time cap")
    if progress["artifact_bytes"] > int(contract["maximum_trace_bundle_bytes"]):
        raise CaptureError("completed captures exceed artifact byte cap")
    atomic_json(progress_path, progress)
    atomic_json(summary_path, progress)
    print(
        json.dumps(
            {
                "status": "complete",
                "capture_count": progress["completed_capture_count"],
                "profiling_wall_seconds": progress["profiling_wall_seconds"],
                "artifact_bytes": progress["artifact_bytes"],
                "summary": str(summary_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
