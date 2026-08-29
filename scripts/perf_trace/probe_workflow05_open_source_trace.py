#!/usr/bin/env python3
"""Attempt official Perfetto interfaces before selecting a custom fallback."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


QUERY = """
SELECT 'slice' AS table_name, COUNT(*) AS row_count FROM slice
UNION ALL
SELECT 'counter' AS table_name, COUNT(*) AS row_count FROM counter
UNION ALL
SELECT 'track' AS table_name, COUNT(*) AS row_count FROM track
ORDER BY table_name
""".strip()

SEMANTIC_QUERY = """
SELECT printf('WF05|category|%s|%d',
              replace(COALESCE(category, ''), '|', '/'), COUNT(*)) AS record
FROM slice GROUP BY category
UNION ALL
SELECT printf('WF05|flow|all|%d', COUNT(*)) AS record FROM flow
UNION ALL
SELECT printf('WF05|track|%s|1',
              replace(COALESCE(name, ''), '|', '/')) AS record
FROM track WHERE name IS NOT NULL
UNION ALL
SELECT printf('WF05|arg_key|%s|%d',
              replace(COALESCE(key, ''), '|', '/'), COUNT(*)) AS record
FROM args GROUP BY key
UNION ALL
SELECT printf('WF05|hiptx_name|%s|%d',
              replace(COALESCE(name, ''), '|', '/'), COUNT(*)) AS record
FROM slice WHERE category = 'HIPTX' GROUP BY name
ORDER BY record
""".strip()

SEMANTIC_RECORD_RE = re.compile(
    r"WF05\|(category|flow|track|arg_key|hiptx_name)\|([^|\r\n]*)\|(\d+)"
)


class ProbeError(RuntimeError):
    """Fail-closed open-source trace-interface probe error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excerpt(value: str, limit: int = 4_000) -> str:
    return value[-limit:]


def run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def resolve_cli(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise ProbeError(f"explicit Trace Processor is missing: {path}")
        return path
    for name in ("trace_processor", "trace_processor_shell"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None


def managed_download_wrapper(path: Path) -> bool:
    """Detect official convenience wrappers that fetch a binary on invocation."""
    try:
        prefix = path.read_bytes()[:256_000].decode("utf-8", errors="ignore")
    except OSError:
        return False
    signatures = (
        "download_or_get_cached",
        "TRACE_PROCESSOR_SHELL_MANIFEST",
        "perfetto-luci-artifacts",
    )
    return sum(signature in prefix for signature in signatures) >= 2


def checked_ui_url(value: str) -> str:
    parsed = urlparse(value)
    loopback = {"localhost", "127.0.0.1", "::1"}
    if not parsed.netloc or not parsed.hostname:
        raise ProbeError("Perfetto UI URL must contain an origin")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in loopback
    ):
        raise ProbeError("Perfetto UI URL must use HTTPS or loopback HTTP")
    return value


def chrome_json_check(trace: Path) -> dict[str, Any]:
    try:
        payload = json.loads(trace.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "fail", "reason": f"invalid Chrome JSON: {exc}"}
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return {"status": "fail", "reason": "traceEvents is not a list"}
    phase_counts: dict[str, int] = {}
    malformed = 0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("ph"), str):
            malformed += 1
            continue
        phase = event["ph"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    interval_count = phase_counts.get("X", 0)
    if malformed or interval_count == 0:
        return {
            "status": "fail",
            "reason": "Chrome JSON has malformed events or no X intervals",
            "malformed_event_count": malformed,
            "phase_counts": phase_counts,
        }
    return {
        "status": "pass",
        "reason": "structurally valid Perfetto-supported Chrome JSON candidate",
        "event_count": len(events),
        "malformed_event_count": 0,
        "phase_counts": phase_counts,
    }


def load_semantic_expectations(
    manifest_path: Path | None,
    trace: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if manifest_path is None:
        return None, None
    path = manifest_path.resolve()
    if not path.is_file():
        raise ProbeError(f"native export manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), list):
        raise ProbeError("native export manifest lacks windows")
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for window in payload["windows"]:
        if not isinstance(window, dict):
            continue
        for attempt in window.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            for trace_record in attempt.get("trace_files", []):
                if not isinstance(trace_record, dict):
                    continue
                candidate = Path(str(trace_record.get("path", ""))).resolve()
                if candidate == trace:
                    matches.append((window, attempt, trace_record))
    if len(matches) != 1:
        raise ProbeError(
            f"expected exactly one native-export trace match, found {len(matches)}"
        )
    window, attempt, trace_record = matches[0]
    if trace_record.get("sha256") != sha256_file(trace):
        raise ProbeError("native export manifest trace SHA-256 mismatch")
    expectations = trace_record.get("semantic_expectations") or attempt.get(
        "semantic_expectations"
    )
    if not isinstance(expectations, dict):
        raise ProbeError("matching native export attempt lacks semantic expectations")
    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "selection_rank": window.get("selection_rank"),
        "stable_key": window.get("stable_key"),
        "format": attempt.get("format"),
        "attempt_status": attempt.get("status"),
    }
    return expectations, provenance


def semantic_records(values: list[str] | str) -> dict[str, dict[str, int]]:
    text = "\n".join(values) if isinstance(values, list) else values
    records: dict[str, dict[str, int]] = {
        "category": {},
        "flow": {},
        "track": {},
        "arg_key": {},
        "hiptx_name": {},
    }
    for record_type, name, count in SEMANTIC_RECORD_RE.findall(text):
        previous = records[record_type].get(name)
        value = int(count)
        if previous is not None and previous != value:
            raise ProbeError(
                f"conflicting semantic query rows for {record_type}/{name}"
            )
        records[record_type][name] = value
    return records


def validate_semantics(
    records: dict[str, dict[str, int]], expectations: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    categories = expectations.get("required_category_min_counts")
    if not isinstance(categories, dict) or not categories:
        raise ProbeError("semantic expectations lack category minimums")
    for category, minimum in categories.items():
        checks[f"category:{category}"] = records["category"].get(category, 0) >= int(
            minimum
        )
    for name in expectations.get("required_slice_names", []):
        checks[f"hiptx_name:{name}"] = records["hiptx_name"].get(str(name), 0) >= 1
    track_names = records["track"]
    for substring in expectations.get("required_track_name_substrings", []):
        checks[f"track_substring:{substring}"] = any(
            str(substring) in name for name in track_names
        )
    arg_keys = records["arg_key"]
    for suffix in expectations.get("required_arg_key_suffixes", []):
        checks[f"arg_suffix:{suffix}"] = any(
            name.endswith(str(suffix)) for name in arg_keys
        )
    minimum_flow = int(expectations.get("minimum_flow_count", 0))
    checks["minimum_flow_count"] = records["flow"].get("all", 0) >= minimum_flow
    status = "pass" if checks and all(checks.values()) else "fail"
    return {
        "status": status,
        "checks": checks,
        "records": records,
        "expectations": expectations,
    }


def python_attempt(
    trace: Path,
    cli: Path | None,
    allow_managed_download: bool,
    expectations: dict[str, Any] | None,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "backend": "perfetto_trace_processor_python_api",
        "interface": (
            "perfetto.trace_processor.TraceProcessor(trace=..., "
            "config=TraceProcessorConfig(...)); query(PerfettoSQL)"
        ),
        "discovered": False,
        "invoked": False,
        "status": "unavailable",
    }
    try:
        spec = importlib.util.find_spec("perfetto.trace_processor")
    except (ModuleNotFoundError, ValueError) as exc:
        attempt["reason"] = f"module discovery failed: {exc}"
        return attempt
    if spec is None:
        attempt["reason"] = "perfetto.trace_processor module is not installed"
        return attempt
    attempt["discovered"] = True
    attempt["module"] = spec.origin
    try:
        attempt["package_version"] = importlib.metadata.version("perfetto")
    except importlib.metadata.PackageNotFoundError:
        attempt["package_version"] = "unresolved"
    if cli is None and not allow_managed_download:
        attempt["status"] = "skipped"
        attempt["reason"] = (
            "module exists but no local Trace Processor binary is resolved; "
            "managed download is disabled"
        )
        return attempt
    if cli is not None and managed_download_wrapper(cli) and not allow_managed_download:
        attempt["status"] = "skipped"
        attempt["reason"] = (
            "resolved Trace Processor is a managed-download wrapper; "
            "network download is disabled"
        )
        attempt["bin_path"] = str(cli)
        attempt["bin_sha256"] = sha256_file(cli)
        return attempt
    attempt["invoked"] = True
    attempt["managed_download_allowed"] = allow_managed_download
    attempt["bin_path"] = str(cli) if cli else None
    try:
        api = importlib.import_module("perfetto.trace_processor")
        config = api.TraceProcessorConfig(
            bin_path=str(cli) if cli else None,
            fetch_latest_trace_processor=False,
            load_timeout=15,
        )
        rows: list[dict[str, Any]] = []
        semantic_validation = None
        with api.TraceProcessor(trace=str(trace), config=config) as processor:
            for row in processor.query(QUERY):
                rows.append(
                    {
                        "table_name": str(row.table_name),
                        "row_count": int(row.row_count),
                    }
                )
            if {row["table_name"] for row in rows} != {
                "slice",
                "counter",
                "track",
            }:
                raise ProbeError(f"unexpected PerfettoSQL result: {rows}")
            if expectations is not None:
                semantic_values = [
                    str(row.record) for row in processor.query(SEMANTIC_QUERY)
                ]
                semantic_validation = validate_semantics(
                    semantic_records(semantic_values), expectations
                )
                if semantic_validation["status"] != "pass":
                    raise ProbeError(
                        f"Perfetto semantic validation failed: {semantic_validation['checks']}"
                    )
        attempt.update(
            {
                "status": "pass",
                "reason": "official Python API parsed and queried the trace",
                "query": QUERY,
                "query_rows": rows,
                "semantic_query": SEMANTIC_QUERY if expectations is not None else None,
                "semantic_validation": semantic_validation,
            }
        )
    except Exception as exc:  # Official API errors vary by release.
        attempt.update(
            {
                "status": "fail",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return attempt


def cli_attempt(
    trace: Path,
    cli: Path | None,
    expectations: dict[str, Any] | None,
    allow_managed_download: bool,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "backend": "perfetto_trace_processor_cli",
        "interface": "trace_processor query TRACE_FILE PERFETTOSQL",
        "discovered": cli is not None,
        "invoked": False,
        "status": "unavailable",
    }
    if cli is None:
        attempt["reason"] = "no trace_processor or trace_processor_shell on PATH"
        return attempt
    attempt["path"] = str(cli)
    attempt["sha256"] = sha256_file(cli)
    if managed_download_wrapper(cli) and not allow_managed_download:
        attempt.update(
            status="skipped",
            reason=(
                "resolved Trace Processor is a managed-download wrapper; "
                "network download is disabled and the wrapper was not invoked"
            ),
            managed_download_wrapper=True,
        )
        return attempt
    try:
        help_result = run([str(cli), "--help"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempt.update(status="fail", reason=f"help failed: {exc}")
        return attempt
    help_text = help_result.stdout + help_result.stderr
    attempt["help_exit_status"] = help_result.returncode
    attempt["help_excerpt"] = excerpt(help_text)
    if help_result.returncode != 0:
        attempt.update(status="fail", reason="Trace Processor --help failed")
        return attempt
    if "Commands:" not in help_text or "query" not in help_text:
        attempt.update(
            status="skipped",
            reason=(
                "resolved binary does not advertise the official query "
                "subcommand; no legacy invocation was guessed"
            ),
        )
        return attempt
    command = [str(cli), "query", str(trace), QUERY]
    attempt["invoked"] = True
    attempt["command"] = command
    try:
        result = run(command)
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempt.update(status="fail", reason=f"query failed: {exc}")
        return attempt
    attempt["exit_status"] = result.returncode
    attempt["stdout_excerpt"] = excerpt(result.stdout)
    attempt["stderr_excerpt"] = excerpt(result.stderr)
    if result.returncode != 0:
        attempt.update(status="fail", reason="official CLI query returned nonzero")
        return attempt
    if not all(name in result.stdout for name in ("slice", "counter", "track")):
        attempt.update(status="fail", reason="query output lacks required tables")
        return attempt
    semantic_validation = None
    semantic_command = None
    if expectations is not None:
        semantic_command = [str(cli), "query", str(trace), SEMANTIC_QUERY]
        try:
            semantic_result = run(semantic_command)
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempt.update(status="fail", reason=f"semantic query failed: {exc}")
            return attempt
        attempt["semantic_exit_status"] = semantic_result.returncode
        attempt["semantic_stdout_excerpt"] = excerpt(semantic_result.stdout)
        attempt["semantic_stderr_excerpt"] = excerpt(semantic_result.stderr)
        if semantic_result.returncode != 0:
            attempt.update(status="fail", reason="official CLI semantic query failed")
            return attempt
        semantic_validation = validate_semantics(
            semantic_records(semantic_result.stdout), expectations
        )
        if semantic_validation["status"] != "pass":
            attempt.update(
                status="fail",
                reason="official CLI parsed the trace but semantic checks failed",
                semantic_validation=semantic_validation,
            )
            return attempt
    attempt.update(
        status="pass",
        reason="official Trace Processor CLI parsed and queried the trace",
        query=QUERY,
        semantic_command=semantic_command,
        semantic_validation=semantic_validation,
    )
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Try official Perfetto Python/CLI interfaces, record every "
            "attempt, and label any custom timeline fallback."
        )
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--expected-format",
        choices=("auto", "chrome-json", "perfetto-proto"),
        default="auto",
    )
    parser.add_argument("--trace-processor-bin", type=Path)
    parser.add_argument("--perfetto-python-root", type=Path)
    parser.add_argument("--native-export-manifest", type=Path)
    parser.add_argument(
        "--perfetto-ui-url",
        default="https://ui.perfetto.dev/#!/?mode=embedded",
    )
    parser.add_argument("--allow-python-managed-download", action="store_true")
    args = parser.parse_args()

    trace = args.trace.resolve()
    output = args.output_manifest.resolve()
    if not trace.is_file() or trace.stat().st_size == 0:
        raise ProbeError(f"trace is missing or empty: {trace}")
    if output.exists():
        raise ProbeError(f"refusing to overwrite manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.perfetto_python_root is not None:
        python_root = args.perfetto_python_root.resolve()
        if not python_root.is_dir():
            raise ProbeError(f"Perfetto Python root is missing: {python_root}")
        sys.path.insert(0, str(python_root))
    expectations, expectation_provenance = load_semantic_expectations(
        args.native_export_manifest,
        trace,
    )

    detected_format = args.expected_format
    if detected_format == "auto":
        detected_format = "chrome-json" if trace.suffix.lower() == ".json" else "perfetto-proto"
    structural = (
        chrome_json_check(trace)
        if detected_format == "chrome-json"
        else {
            "status": "unavailable",
            "reason": "protobuf structure requires official Perfetto parser",
        }
    )
    cli = resolve_cli(args.trace_processor_bin)
    perfetto_ui_url = checked_ui_url(args.perfetto_ui_url)
    attempts: list[dict[str, Any]] = []
    python_result = python_attempt(
        trace,
        cli,
        args.allow_python_managed_download,
        expectations,
    )
    attempts.append(python_result)
    if python_result["status"] == "pass":
        attempts.append(
            {
                "backend": "perfetto_trace_processor_cli",
                "discovered": cli is not None,
                "invoked": False,
                "status": "not_needed_after_preferred_pass",
                "reason": "official Python API already passed",
            }
        )
    else:
        attempts.append(
            cli_attempt(
                trace,
                cli,
                expectations,
                args.allow_python_managed_download,
            )
        )
    passed = next((item for item in attempts if item["status"] == "pass"), None)
    ui_candidate = {
        "backend": "perfetto_ui_local_file",
        "official_interface": "Open trace file or drag-and-drop local trace",
        "ui_url": perfetto_ui_url,
        "version": "unresolved_until_runtime_ui_inspection",
        "discovered": False,
        "trace_uploaded_by_probe": False,
        "invoked": False,
        "status": (
            "ready_after_processor_validation"
            if passed
            else "candidate_unvalidated"
        ),
        "reason": (
            "Trace Processor parse/query passed"
            if passed
            else "no official local parser completed; retain trace for later UI attempt"
        ),
    }
    attempts.append(ui_candidate)
    fallback_required = passed is None
    selected_backend = (
        passed["backend"] if passed else "custom_plotly_timeline_fallback"
    )
    manifest = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "pass" if passed else "degraded_fallback_selected",
        "policy": "open_source_first_with_labeled_custom_fallback",
        "official_interface_references": {
            "python_api": (
                "https://perfetto.dev/docs/analysis/trace-processor-python"
            ),
            "trace_analysis": (
                "https://perfetto.dev/docs/quickstart/trace-analysis"
            ),
            "ui_embedding": (
                "https://perfetto.dev/docs/visualization/embedding-the-ui"
            ),
        },
        "preferred_backend_order": [
            "perfetto_trace_processor_python_api",
            "perfetto_trace_processor_cli",
            "perfetto_ui_local_file",
            "custom_plotly_timeline_fallback",
        ],
        "network_download_allowed": args.allow_python_managed_download,
        "trace": {
            "path": str(trace),
            "sha256": sha256_file(trace),
            "size_bytes": trace.stat().st_size,
            "format": detected_format,
            "structural_check": structural,
            "semantic_expectations": expectations,
            "semantic_expectation_provenance": expectation_provenance,
        },
        "attempts": attempts,
        "selected_backend": selected_backend,
        "open_source_processor_reused": passed is not None,
        "custom_timeline_fallback_required": fallback_required,
        "fallback_label_required": fallback_required,
        "fallback_reason": (
            None
            if passed
            else "no official Perfetto parser completed successfully"
        ),
        "analytical_values_recomputed": False,
        "gpu_or_model_activity": False,
    }
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
