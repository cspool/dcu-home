#!/usr/bin/env python3
"""Project Qwen SAME_INPUT layer/component latency onto FX process stages.

The current Qwen/DCU binding uses the complete R01 layer trace as the canonical
layer denominator. R01 deliberately contains layer totals only, so the matching
R02 process-instrumented trace supplies measured attention and MLP component
envelopes through strict HIPTX -> HIP runtime _Index -> HIPOPS ownership. Those
component envelopes are normalized to the R01 denominator before any FX-stage
allocation. The generated process rows are therefore attribution estimates, not
direct process timings.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SUCCESS_STATUSES = {"ok", "pass", "passed", "success", "complete"}

# The first eleven entries are the portable Skill contract. The final three are
# live Qwen3.5 stage names whose FX targets explicitly remain inside attention.
PROCESS_BUCKET = {
    "inputs": "metadata",
    "input_rmsnorm": "outer",
    "qkv_projection": "attn",
    "rope": "attn",
    "attention_scores": "attn",
    "attention_output": "attn",
    "visual_process": "attn",
    "output_projection": "attn",
    "post_attention_rmsnorm": "outer",
    "mlp": "mlp",
    "layer_output": "metadata",
    "gdn_recurrent_core": "attn",
    "gdn_gated_rmsnorm": "attn",
    "kv_cache_attention": "attn",
}

# R02 contains one intentionally combined residual-add/RMSNorm transition. It
# is excluded from measured attn/mlp and consequently remains in the residual
# outer bucket. This preserves the upstream ambiguity without double counting.
INSTRUMENTED_COMPONENT_BUCKET = {
    **PROCESS_BUCKET,
    "output_projection__post_attention_rmsnorm_fused": "outer",
}

SEMANTIC_MULTIPLIER = {
    "inputs": 0.0,
    "input_rmsnorm": 1.0,
    "qkv_projection": 3.0,
    "rope": 1.2,
    "attention_scores": 4.0,
    "attention_output": 1.3,
    "visual_process": 1.0,
    "output_projection": 2.5,
    "post_attention_rmsnorm": 1.0,
    "mlp": 4.0,
    "layer_output": 0.0,
    "gdn_recurrent_core": 4.0,
    "gdn_gated_rmsnorm": 1.3,
    "kv_cache_attention": 4.0,
}

FAMILY_AFFINITY = {
    "input_rmsnorm": {"norm": 5.0, "elementwise": 1.0, "copy": 0.5},
    "qkv_projection": {"gemm": 5.0, "copy": 1.0, "norm": 0.5},
    "rope": {"rope": 6.0, "elementwise": 2.0, "copy": 1.0},
    "attention_scores": {"attention": 6.0, "reduction": 2.0},
    "attention_output": {"attention": 2.0, "elementwise": 3.0},
    "visual_process": {"elementwise": 2.0, "copy": 1.0},
    "output_projection": {"gemm": 5.0, "copy": 1.0},
    "post_attention_rmsnorm": {
        "norm": 5.0,
        "elementwise": 1.0,
        "copy": 0.5,
    },
    "mlp": {"gemm": 6.0, "elementwise": 2.0, "reduction": 1.0},
    "gdn_recurrent_core": {"attention": 5.0, "gdn": 7.0},
    "gdn_gated_rmsnorm": {
        "norm": 4.0,
        "elementwise": 2.0,
        "copy": 0.5,
    },
    "kv_cache_attention": {"attention": 7.0, "copy": 1.0},
}

REQUIRED_HIPTX_COLUMNS = {
    "_Index",
    "BeginNs",
    "EndNs",
    "message",
    "begin_Index",
    "end_Index",
}
REQUIRED_HIP_COLUMNS = {"_Index", "BeginNs"}
REQUIRED_HIPOPS_COLUMNS = {
    "_Index",
    "BeginNs",
    "EndNs",
    "DurationNs",
    "dev_id",
    "Name",
}

DETAIL_FIELDS = [
    "variant",
    "display_name",
    "fx_event_id",
    "layer",
    "phase",
    "process",
    "title",
    "allocated_cupti_kernel_ms",
    "allocated_nvtx_cpu_ms",
    "fx_q_len",
    "fx_kv_len",
    "match",
    "perf_q_len",
    "perf_kv_len",
    "nodes",
    "bucket",
    "source_bucket_cupti_kernel_ms",
    "source_bucket_nvtx_cpu_ms",
    "source_total_cupti_kernel_ms",
    "source_total_nvtx_cpu_ms",
    "source_forward_id",
    "source_occurrence",
    "source_occurrence_key",
    "component_source_event_id",
    "component_kernel_normalization_factor",
    "component_cpu_normalization_factor",
    "process_weight_mode",
    "kernel_split_mode_requested",
    "kernel_split_mode_effective",
    "source_kernel_families_used",
]


class ProjectionError(RuntimeError):
    """A fail-closed projection or evidence validation error."""


@dataclass(frozen=True)
class VariantSpec:
    slug: str
    display_name: str
    layer_breakdown: Path
    layer_events: Path
    contract_id: str
    contract_sha256: str


@dataclass
class LayerOccurrence:
    contract_id: str
    contract_sha256: str
    forward_id: int
    layer_idx: int
    occurrence: int
    occurrence_key: str
    phase: str
    q_len: int
    past_len: int
    kv_len: int
    workload_type: str
    range_name: str
    total_kernel_ms: float
    total_cpu_ms: float
    family_ms: dict[str, float]
    component_source_event_id: str = ""
    raw_attn_kernel_ms: float = 0.0
    raw_mlp_kernel_ms: float = 0.0
    raw_attn_cpu_ms: float = 0.0
    raw_mlp_cpu_ms: float = 0.0
    attn_kernel_ms: float = 0.0
    mlp_kernel_ms: float = 0.0
    outer_kernel_ms: float = 0.0
    attn_cpu_ms: float = 0.0
    mlp_cpu_ms: float = 0.0
    outer_cpu_ms: float = 0.0
    kernel_normalization_factor: float = 1.0
    cpu_normalization_factor: float = 1.0


@dataclass(frozen=True)
class FxStage:
    process: str
    title: str
    nodes: int


@dataclass(frozen=True)
class FxEvent:
    event_id: str
    layer_idx: int
    phase: str
    q_len: int
    kv_len: int
    workload_type: str
    stages: tuple[FxStage, ...]
    reconstruction_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ProjectionError(f"{field} must be finite, got {value!r}")
    return result


def as_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"{field} must be an integer, got {value!r}") from exc


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def require_file(path: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ProjectionError(f"missing {role}: {resolved}")
    return resolved


def require_dir(path: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ProjectionError(f"missing {role}: {resolved}")
    return resolved


def require_under(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ProjectionError(
            f"{role} escapes runtime artifact root: {resolved}"
        ) from exc
    return resolved


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProjectionError(f"CSV has no header: {path}")
        return list(reader)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def csv_number(value: float) -> str:
    return f"{value:.12f}"


def markdown_number(value: float) -> str:
    return f"{value:.6f}"


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ProjectionError(f"variant cannot be converted to a slug: {value!r}")
    return slug


def normalize_component_pair(
    total_ms: float,
    attn_ms: float,
    mlp_ms: float,
    *,
    role: str,
    max_overage_fraction: float | None = 0.10,
) -> tuple[float, float, float, float]:
    if total_ms < 0 or attn_ms < 0 or mlp_ms < 0:
        raise ProjectionError(f"negative {role} component duration")
    component_sum = attn_ms + mlp_ms
    if (
        max_overage_fraction is not None
        and component_sum > total_ms * (1.0 + max_overage_fraction) + 1e-12
    ):
        raise ProjectionError(
            f"{role} attn+mlp exceeds the R01 total by more than "
            f"{100.0 * max_overage_fraction:g}%: "
            f"total={total_ms}, attn={attn_ms}, mlp={mlp_ms}"
        )
    if component_sum > total_ms and component_sum > 0:
        scale = total_ms / component_sum
        # Construct the second normalized component as an exact residual. This
        # keeps every bucket non-negative without turning a floating-point
        # roundoff residual into a small negative outer duration.
        normalized_attn = min(
            total_ms, max(0.0, total_ms * attn_ms / component_sum)
        )
        normalized_mlp = total_ms - normalized_attn
        outer = 0.0
    else:
        scale = 1.0
        normalized_attn = attn_ms
        normalized_mlp = mlp_ms
        outer = total_ms - component_sum
    return normalized_attn, normalized_mlp, outer, scale


def canonical_family(name: str) -> str:
    value = name.strip().lower()
    if not value or value in {"other", "unknown", "none"}:
        return "other"
    if "gdn" in value or "delta" in value or "recurrent" in value:
        return "gdn"
    if "rope" in value or "rotary" in value or "mrope" in value:
        return "rope"
    if "attention" in value or "flash" in value or "kv_cache" in value:
        return "attention"
    if "gemm" in value or "matmul" in value or "mmac" in value:
        return "gemm"
    if "rms" in value or "norm" in value:
        return "norm"
    if "copy" in value or "cache" in value:
        return "copy"
    if "reduce" in value or "softmax" in value:
        return "reduction"
    if "element" in value or "silu" in value or "sigmoid" in value:
        return "elementwise"
    return "other"


def load_layer_occurrences(spec: VariantSpec) -> dict[tuple[Any, ...], LayerOccurrence]:
    event_rows = read_csv(spec.layer_events)
    breakdown_rows = read_csv(spec.layer_breakdown)
    event_required = {
        "contract_id",
        "contract_sha256",
        "forward_id",
        "layer_idx",
        "occurrence",
        "occurrence_key",
        "phase",
        "q_len",
        "past_len",
        "kv_len",
        "workload_type",
        "range_name",
        "hiptx_host_range_duration_ms",
        "hipprof_launch_owned_kernel_sum_ms",
    }
    breakdown_required = {
        "contract_id",
        "forward_id",
        "layer_idx",
        "occurrence",
        "kernel_family",
        "launch_owned_kernel_duration_ms",
        "layer_launch_owned_kernel_sum_ms",
    }
    if not event_rows or not event_required.issubset(event_rows[0]):
        missing = sorted(event_required - set(event_rows[0] if event_rows else []))
        raise ProjectionError(f"layer-events schema is missing: {missing}")
    if not breakdown_rows or not breakdown_required.issubset(breakdown_rows[0]):
        missing = sorted(
            breakdown_required - set(breakdown_rows[0] if breakdown_rows else [])
        )
        raise ProjectionError(f"layer-breakdown schema is missing: {missing}")

    occurrences: dict[tuple[Any, ...], LayerOccurrence] = {}
    for row in event_rows:
        if row["contract_id"] != spec.contract_id:
            continue
        if row["contract_sha256"] != spec.contract_sha256:
            raise ProjectionError(
                f"{spec.slug}: contract SHA mismatch in layer events"
            )
        key = (
            row["contract_id"],
            as_int(row["forward_id"], field="forward_id"),
            as_int(row["layer_idx"], field="layer_idx"),
            as_int(row["occurrence"], field="occurrence"),
        )
        if key in occurrences:
            raise ProjectionError(f"duplicate layer occurrence: {key}")
        occurrences[key] = LayerOccurrence(
            contract_id=row["contract_id"],
            contract_sha256=row["contract_sha256"],
            forward_id=key[1],
            layer_idx=key[2],
            occurrence=key[3],
            occurrence_key=row["occurrence_key"],
            phase=row["phase"],
            q_len=as_int(row["q_len"], field="q_len"),
            past_len=as_int(row["past_len"], field="past_len"),
            kv_len=as_int(row["kv_len"], field="kv_len"),
            workload_type=row["workload_type"],
            range_name=row["range_name"],
            total_kernel_ms=as_float(
                row["hipprof_launch_owned_kernel_sum_ms"],
                field="hipprof_launch_owned_kernel_sum_ms",
            ),
            total_cpu_ms=as_float(
                row["hiptx_host_range_duration_ms"],
                field="hiptx_host_range_duration_ms",
            ),
            family_ms={},
        )
    if not occurrences:
        raise ProjectionError(f"{spec.slug}: no rows for contract {spec.contract_id}")

    declared_totals: dict[tuple[Any, ...], set[float]] = collections.defaultdict(set)
    for row in breakdown_rows:
        if row["contract_id"] != spec.contract_id:
            continue
        key = (
            row["contract_id"],
            as_int(row["forward_id"], field="forward_id"),
            as_int(row["layer_idx"], field="layer_idx"),
            as_int(row["occurrence"], field="occurrence"),
        )
        occurrence = occurrences.get(key)
        if occurrence is None:
            raise ProjectionError(f"breakdown row lacks layer event: {key}")
        family = row["kernel_family"].strip() or "unknown"
        duration = as_float(
            row["launch_owned_kernel_duration_ms"],
            field="launch_owned_kernel_duration_ms",
        )
        occurrence.family_ms[family] = occurrence.family_ms.get(family, 0.0) + duration
        declared_totals[key].add(
            as_float(
                row["layer_launch_owned_kernel_sum_ms"],
                field="layer_launch_owned_kernel_sum_ms",
            )
        )

    if set(declared_totals) != set(occurrences):
        missing = sorted(set(occurrences) - set(declared_totals))[:5]
        raise ProjectionError(f"layer breakdown misses occurrences: {missing}")
    for key, occurrence in occurrences.items():
        if len(declared_totals[key]) != 1:
            raise ProjectionError(f"ambiguous layer total in breakdown: {key}")
        breakdown_total = next(iter(declared_totals[key]))
        family_total = sum(occurrence.family_ms.values())
        if not math.isclose(
            breakdown_total, occurrence.total_kernel_ms, rel_tol=0, abs_tol=1e-9
        ):
            raise ProjectionError(f"layer total mismatch for {key}")
        if not math.isclose(
            family_total, occurrence.total_kernel_ms, rel_tol=0, abs_tol=1e-8
        ):
            raise ProjectionError(f"kernel-family conservation mismatch for {key}")
    return occurrences


def load_selected_runtime_events(
    path: Path,
    *,
    contract_id: str,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProjectionError(
                    f"invalid runtime event JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ProjectionError("runtime event row must be an object")
            expected = row.get("expected_process_range_names", [])
            if not expected:
                continue
            if row.get("contract_id") != contract_id:
                raise ProjectionError("R02 selected event parent contract mismatch")
            if row.get("contract_sha256") != contract_sha256:
                raise ProjectionError("R02 selected event contract SHA mismatch")
            event_id = str(row.get("event_id", ""))
            if not event_id or event_id in seen_events:
                raise ProjectionError(f"invalid or duplicate selected event: {event_id}")
            if not isinstance(expected, list) or not all(
                isinstance(value, str) and value for value in expected
            ):
                raise ProjectionError(f"invalid expected process names for {event_id}")
            seen_events.add(event_id)
            rows.append(row)
    if not rows:
        raise ProjectionError("no R02 selected process-profile events")
    return rows


def load_process_inventory(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    required = {
        "process_id",
        "process_title",
        "fragment_id",
        "aggregation_key",
        "nvtx_range_name",
        "status",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else []))
        raise ProjectionError(f"process inventory schema is missing: {missing}")
    inventory: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["nvtx_range_name"]
        if not name or name in inventory:
            raise ProjectionError(f"duplicate or empty inventory range: {name!r}")
        status = row["status"].strip().lower()
        if status in {"unresolved", "missing", "unsupported"}:
            raise ProjectionError(f"unresolved process inventory row: {name}")
        process_id = row["process_id"]
        if process_id not in INSTRUMENTED_COMPONENT_BUCKET:
            raise ProjectionError(
                f"process inventory stage has no component bucket: {process_id}"
            )
        inventory[name] = row
    return inventory


def one_table(tables: Sequence[str], pattern: re.Pattern[str], role: str) -> str:
    matches = sorted(table for table in tables if pattern.fullmatch(table))
    if len(matches) != 1:
        raise ProjectionError(
            f"expected exactly one {role} table, found {matches}"
        )
    return matches[0]


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    quoted = table.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{quoted}")')
    }


def strict_process_measurements(
    db_path: Path,
    runtime_events: Sequence[dict[str, Any]],
    inventory: dict[str, dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    marker_to_event: dict[str, str] = {}
    for event in runtime_events:
        event_id = str(event["event_id"])
        for name in event["expected_process_range_names"]:
            if name in marker_to_event:
                raise ProjectionError(f"process marker expected twice: {name}")
            marker_to_event[name] = event_id
    expected_markers = set(marker_to_event)
    if expected_markers != set(inventory):
        missing = sorted(expected_markers - set(inventory))[:5]
        extra = sorted(set(inventory) - expected_markers)[:5]
        raise ProjectionError(
            f"runtime/inventory marker mismatch; missing={missing}, extra={extra}"
        )

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        hiptx = one_table(tables, re.compile(r"HIPTX_.+"), "HIPTX")
        hip = one_table(tables, re.compile(r"HIP_.+"), "HIP runtime")
        hipops = one_table(tables, re.compile(r"HIPOPS_.+"), "HIPOPS")
        schema_checks = (
            (hiptx, REQUIRED_HIPTX_COLUMNS),
            (hip, REQUIRED_HIP_COLUMNS),
            (hipops, REQUIRED_HIPOPS_COLUMNS),
        )
        for table, required in schema_checks:
            missing = sorted(required - table_columns(connection, table))
            if missing:
                raise ProjectionError(f"{table} is missing columns: {missing}")

        query = f"""
            SELECT
                x._Index AS marker_index,
                x.message AS marker_name,
                x.BeginNs AS marker_begin_ns,
                x.EndNs AS marker_end_ns,
                x.begin_Index AS marker_begin_index,
                x.end_Index AS marker_end_index,
                h._Index AS runtime_index,
                o.BeginNs AS kernel_begin_ns,
                o.EndNs AS kernel_end_ns,
                o.DurationNs AS kernel_duration_ns,
                o.dev_id AS device_id,
                o.Name AS kernel_name_id
            FROM "{hiptx}" AS x
            LEFT JOIN "{hip}" AS h
              ON h.BeginNs >= x.BeginNs
             AND h.BeginNs < x.EndNs
             AND h._Index BETWEEN x.begin_Index AND x.end_Index
            LEFT JOIN "{hipops}" AS o
              ON o._Index = h._Index
            WHERE x.message LIKE 'pra.fx_process.%'
            ORDER BY x.BeginNs, h.BeginNs, o.BeginNs
        """
        raw_rows = list(connection.execute(query))
    finally:
        connection.close()
    if not raw_rows:
        raise ProjectionError("R02 DB contains no process HIPTX ranges")

    grouped: dict[int, dict[str, Any]] = {}
    kernel_owner: dict[int, str] = {}
    owned_devices: set[int] = set()
    for row in raw_rows:
        marker_index = as_int(row["marker_index"], field="marker_index")
        marker_name = str(row["marker_name"])
        if marker_name not in expected_markers:
            raise ProjectionError(f"unexpected process marker in DB: {marker_name}")
        record = grouped.setdefault(
            marker_index,
            {
                "marker_name": marker_name,
                "marker_begin_ns": as_int(
                    row["marker_begin_ns"], field="marker_begin_ns"
                ),
                "marker_end_ns": as_int(
                    row["marker_end_ns"], field="marker_end_ns"
                ),
                "marker_begin_index": as_int(
                    row["marker_begin_index"], field="marker_begin_index"
                ),
                "marker_end_index": as_int(
                    row["marker_end_index"], field="marker_end_index"
                ),
                "kernels": {},
            },
        )
        if record["marker_name"] != marker_name:
            raise ProjectionError(f"marker index reused by two names: {marker_index}")
        runtime_index_value = row["runtime_index"]
        duration_value = row["kernel_duration_ns"]
        if runtime_index_value is None or duration_value is None:
            continue
        runtime_index = as_int(runtime_index_value, field="runtime_index")
        previous_owner = kernel_owner.get(runtime_index)
        if previous_owner is not None and previous_owner != marker_name:
            raise ProjectionError(
                f"runtime _Index {runtime_index} has multiple process owners"
            )
        kernel_owner[runtime_index] = marker_name
        kernel = {
            "duration_ns": as_int(duration_value, field="kernel_duration_ns"),
            "device_id": as_int(row["device_id"], field="device_id"),
            "kernel_name_id": str(row["kernel_name_id"]),
        }
        previous_kernel = record["kernels"].get(runtime_index)
        if previous_kernel is not None and previous_kernel != kernel:
            raise ProjectionError(
                f"runtime _Index {runtime_index} maps to multiple HIPOPS rows"
            )
        record["kernels"][runtime_index] = kernel
        owned_devices.add(kernel["device_id"])

    if len(grouped) != len(expected_markers):
        observed = {record["marker_name"] for record in grouped.values()}
        missing = sorted(expected_markers - observed)[:5]
        raise ProjectionError(f"process DB is missing expected markers: {missing}")
    if len({record["marker_name"] for record in grouped.values()}) != len(grouped):
        raise ProjectionError("duplicate process marker messages in DB")

    measurement_rows: list[dict[str, Any]] = []
    components: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {
            "attn_kernel_ms": 0.0,
            "mlp_kernel_ms": 0.0,
            "attn_cpu_ms": 0.0,
            "mlp_cpu_ms": 0.0,
        }
    )
    for marker_index, record in sorted(
        grouped.items(), key=lambda item: item[1]["marker_begin_ns"]
    ):
        marker_name = record["marker_name"]
        event_id = marker_to_event[marker_name]
        inventory_row = inventory[marker_name]
        process_id = inventory_row["process_id"]
        bucket = INSTRUMENTED_COMPONENT_BUCKET[process_id]
        cpu_ms = (
            record["marker_end_ns"] - record["marker_begin_ns"]
        ) / 1e6
        kernels = record["kernels"]
        kernel_ms = sum(
            kernel["duration_ns"] for kernel in kernels.values()
        ) / 1e6
        if bucket in {"attn", "mlp"}:
            components[event_id][f"{bucket}_kernel_ms"] += kernel_ms
            components[event_id][f"{bucket}_cpu_ms"] += cpu_ms
        measurement_rows.append(
            {
                "event_id": event_id,
                "process_id": process_id,
                "process_title": inventory_row["process_title"],
                "fragment_id": inventory_row["fragment_id"],
                "aggregation_key": inventory_row["aggregation_key"],
                "status": inventory_row["status"],
                "nvtx_range_name": marker_name,
                "component_bucket_role": bucket,
                "hiptx_cpu_ms": csv_number(cpu_ms),
                "strict_owned_kernel_count": len(kernels),
                "strict_owned_hipops_ms": csv_number(kernel_ms),
                "runtime_index_examples": ";".join(
                    str(value) for value in sorted(kernels)[:3]
                ),
                "device_ids": ";".join(
                    str(value)
                    for value in sorted(
                        {kernel["device_id"] for kernel in kernels.values()}
                    )
                ),
                "ownership_method": (
                    "process HIPTX range -> HIP Runtime BeginNs inside range and "
                    "runtime _Index inside marker bounds -> HIPOPS identical _Index"
                ),
                "marker_index": marker_index,
            }
        )

    audit = {
        "hiptx_table": hiptx,
        "hip_runtime_table": hip,
        "hipops_table": hipops,
        "correlation_identity": "_Index",
        "process_range_count": len(grouped),
        "expected_process_range_count": len(expected_markers),
        "strict_owned_kernel_count": sum(
            int(row["strict_owned_kernel_count"]) for row in measurement_rows
        ),
        "unique_strict_owned_runtime_indices": len(kernel_owner),
        "multiply_owned_runtime_indices": 0,
        "strict_owned_device_ids": sorted(owned_devices),
        "ownership_rule": (
            "process HIPTX range -> HIP Runtime BeginNs inside range and runtime "
            "_Index inside marker bounds -> HIPOPS identical _Index"
        ),
    }
    if audit["strict_owned_kernel_count"] != len(kernel_owner):
        raise ProjectionError("strict process ownership is not one-to-one")
    return measurement_rows, dict(components), audit


def attach_component_evidence(
    occurrences: dict[tuple[Any, ...], LayerOccurrence],
    runtime_events: Sequence[dict[str, Any]],
    components: dict[str, dict[str, float]],
) -> tuple[list[LayerOccurrence], list[dict[str, Any]]]:
    selected: list[LayerOccurrence] = []
    normalization_rows: list[dict[str, Any]] = []
    for event in runtime_events:
        event_id = str(event["event_id"])
        key = (
            str(event["contract_id"]),
            as_int(event["forward_id"], field="forward_id"),
            as_int(event["layer_idx"], field="layer_idx"),
            as_int(event["occurrence"], field="occurrence"),
        )
        occurrence = occurrences.get(key)
        if occurrence is None:
            raise ProjectionError(f"R02 selected event lacks R01 layer row: {event_id}")
        comparisons = {
            "phase": occurrence.phase,
            "q_len": occurrence.q_len,
            "past_len": occurrence.past_len,
            "kv_len": occurrence.kv_len,
            "workload_type": occurrence.workload_type,
            "range_name": occurrence.range_name,
        }
        for field, expected in comparisons.items():
            observed = event.get(field)
            if isinstance(expected, int):
                observed = as_int(observed, field=field)
            if observed != expected:
                raise ProjectionError(
                    f"R01/R02 selected event mismatch for {event_id}: "
                    f"{field}={observed!r} != {expected!r}"
                )
        measured = components.get(event_id)
        if measured is None:
            raise ProjectionError(f"R02 component evidence missing for {event_id}")
        occurrence.component_source_event_id = event_id
        occurrence.raw_attn_kernel_ms = measured["attn_kernel_ms"]
        occurrence.raw_mlp_kernel_ms = measured["mlp_kernel_ms"]
        occurrence.raw_attn_cpu_ms = measured["attn_cpu_ms"]
        occurrence.raw_mlp_cpu_ms = measured["mlp_cpu_ms"]
        (
            occurrence.attn_kernel_ms,
            occurrence.mlp_kernel_ms,
            occurrence.outer_kernel_ms,
            occurrence.kernel_normalization_factor,
        ) = normalize_component_pair(
            occurrence.total_kernel_ms,
            occurrence.raw_attn_kernel_ms,
            occurrence.raw_mlp_kernel_ms,
            role=f"{event_id} kernel",
        )
        (
            occurrence.attn_cpu_ms,
            occurrence.mlp_cpu_ms,
            occurrence.outer_cpu_ms,
            occurrence.cpu_normalization_factor,
        ) = normalize_component_pair(
            occurrence.total_cpu_ms,
            occurrence.raw_attn_cpu_ms,
            occurrence.raw_mlp_cpu_ms,
            role=f"{event_id} CPU",
            # R02 HIPTX child ranges come from a separately instrumented
            # capture whose absolute host duration is not comparable to R01.
            # Consume only their attn/MLP ratio and rescale it to the R01 host
            # denominator. The output remains an attribution estimate.
            max_overage_fraction=None,
        )
        normalization_rows.append(
            {
                "event_id": event_id,
                "forward_id": occurrence.forward_id,
                "layer": occurrence.layer_idx,
                "phase": occurrence.phase,
                "q_len": occurrence.q_len,
                "kv_len": occurrence.kv_len,
                "r01_total_cupti_kernel_ms": csv_number(
                    occurrence.total_kernel_ms
                ),
                "r02_raw_attn_cupti_kernel_ms": csv_number(
                    occurrence.raw_attn_kernel_ms
                ),
                "r02_raw_mlp_cupti_kernel_ms": csv_number(
                    occurrence.raw_mlp_kernel_ms
                ),
                "kernel_normalization_factor": csv_number(
                    occurrence.kernel_normalization_factor
                ),
                "normalized_attn_cupti_kernel_ms": csv_number(
                    occurrence.attn_kernel_ms
                ),
                "normalized_mlp_cupti_kernel_ms": csv_number(
                    occurrence.mlp_kernel_ms
                ),
                "outer_cupti_kernel_ms": csv_number(
                    occurrence.outer_kernel_ms
                ),
                "r01_total_nvtx_cpu_ms": csv_number(occurrence.total_cpu_ms),
                "r02_raw_attn_nvtx_cpu_ms": csv_number(
                    occurrence.raw_attn_cpu_ms
                ),
                "r02_raw_mlp_nvtx_cpu_ms": csv_number(
                    occurrence.raw_mlp_cpu_ms
                ),
                "cpu_normalization_factor": csv_number(
                    occurrence.cpu_normalization_factor
                ),
                "normalized_attn_nvtx_cpu_ms": csv_number(
                    occurrence.attn_cpu_ms
                ),
                "normalized_mlp_nvtx_cpu_ms": csv_number(
                    occurrence.mlp_cpu_ms
                ),
                "outer_nvtx_cpu_ms": csv_number(occurrence.outer_cpu_ms),
            }
        )
        selected.append(occurrence)
    return selected, normalization_rows


def load_fx_events(fx_root: Path) -> list[FxEvent]:
    event_csv = fx_root / "fx_layer_events.csv"
    rows = read_csv(event_csv)
    required = {
        "event_id",
        "layer_id",
        "phase",
        "q_len",
        "kv_len",
        "layer_type",
        "fx_traced",
        "fx_trace_status",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else []))
        raise ProjectionError(f"FX event schema is missing: {missing}")
    events: list[FxEvent] = []
    seen: set[str] = set()
    for row in rows:
        if not truthy(row["fx_traced"]):
            continue
        if row["fx_trace_status"].strip().lower() not in SUCCESS_STATUSES:
            continue
        event_id = row["event_id"]
        if not event_id or event_id in seen:
            raise ProjectionError(f"duplicate or empty filtered FX event: {event_id!r}")
        reconstruction_path = require_file(
            fx_root / event_id / "fx_process_reconstruction.json",
            role=f"{event_id} FX process reconstruction",
        )
        payload = json.loads(reconstruction_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ProjectionError(f"{reconstruction_path} must contain an object")
        guards = payload.get("evidence_guards", {})
        if guards.get("measured_latency_reported") is not False:
            raise ProjectionError(
                f"{event_id} FX artifact does not preserve the structural-only guard"
            )
        identity = payload.get("event_identity", {})
        layer_idx = as_int(row["layer_id"], field="layer_id")
        q_len = as_int(row["q_len"], field="q_len")
        kv_len = as_int(row["kv_len"], field="kv_len")
        identity_checks = {
            "event_id": event_id,
            "layer_idx": layer_idx,
            "phase": row["phase"],
            "q_len": q_len,
            "kv_len": kv_len,
        }
        for field, expected in identity_checks.items():
            if identity.get(field) != expected:
                raise ProjectionError(
                    f"{event_id} reconstruction identity mismatch: {field}"
                )
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ProjectionError(f"{event_id} has no reconstructed stages")
        if as_int(payload.get("stage_count"), field="stage_count") != len(raw_stages):
            raise ProjectionError(f"{event_id} stage_count mismatch")
        stages: list[FxStage] = []
        stage_names: set[str] = set()
        for raw_stage in raw_stages:
            if not isinstance(raw_stage, dict):
                raise ProjectionError(f"{event_id} stage must be an object")
            process = str(raw_stage.get("stage", ""))
            if not process or process in stage_names:
                raise ProjectionError(
                    f"{event_id} has duplicate or empty process stage: {process!r}"
                )
            if process not in PROCESS_BUCKET:
                raise ProjectionError(
                    f"{event_id} process has no required bucket mapping: {process}"
                )
            nodes = as_int(raw_stage.get("node_count"), field="node_count")
            if nodes <= 0:
                raise ProjectionError(
                    f"{event_id}/{process} must contain at least one FX node"
                )
            stage_names.add(process)
            stages.append(
                FxStage(
                    process=process,
                    title=str(raw_stage.get("title", process)),
                    nodes=nodes,
                )
            )
        seen.add(event_id)
        events.append(
            FxEvent(
                event_id=event_id,
                layer_idx=layer_idx,
                phase=row["phase"],
                q_len=q_len,
                kv_len=kv_len,
                workload_type=row["layer_type"],
                stages=tuple(stages),
                reconstruction_path=reconstruction_path,
            )
        )
    if not events:
        raise ProjectionError("no successful fx_traced=True FX events")
    return events


def match_event(
    event: FxEvent,
    sources: Sequence[LayerOccurrence],
    *,
    match_mode: str,
) -> tuple[LayerOccurrence | None, str | None]:
    candidates = [
        source
        for source in sources
        if source.layer_idx == event.layer_idx and source.phase == event.phase
    ]
    exact = [
        source
        for source in candidates
        if source.q_len == event.q_len and source.kv_len == event.kv_len
    ]
    if len(exact) > 1:
        raise ProjectionError(
            f"{event.event_id} has ambiguous exact performance matches"
        )
    if exact:
        return exact[0], "exact"
    if match_mode == "exact":
        return None, None
    if not candidates:
        return None, None
    scored = sorted(
        (
            (
                abs(source.q_len - event.q_len)
                + abs(source.kv_len - event.kv_len),
                abs(source.q_len - event.q_len),
                abs(source.kv_len - event.kv_len),
            ),
            source,
        )
        for source in candidates
    )
    best_score = scored[0][0]
    best = [source for score, source in scored if score == best_score]
    if len(best) != 1:
        raise ProjectionError(
            f"{event.event_id} has ambiguous nearest-shape performance matches"
        )
    return best[0], "nearest_shape"


def process_base_weight(stage: FxStage, mode: str) -> float:
    if PROCESS_BUCKET[stage.process] == "metadata":
        return 0.0
    if mode == "node-count":
        return float(stage.nodes)
    multiplier = SEMANTIC_MULTIPLIER.get(stage.process)
    if multiplier is None or multiplier <= 0:
        raise ProjectionError(
            f"missing semantic multiplier for {stage.process}"
        )
    return float(stage.nodes) * multiplier


def family_weight_adjustments(
    stages: Sequence[FxStage],
    family_ms: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    canonical_totals: dict[str, float] = collections.defaultdict(float)
    for family, duration in family_ms.items():
        canonical_totals[canonical_family(family)] += duration
    informative = {
        family: duration
        for family, duration in canonical_totals.items()
        if family != "other" and duration > 0
    }
    total = sum(informative.values())
    if total <= 0:
        return {stage.process: 1.0 for stage in stages}, []
    shares = {family: duration / total for family, duration in informative.items()}
    adjustments: dict[str, float] = {}
    for stage in stages:
        affinity = FAMILY_AFFINITY.get(stage.process, {})
        adjustments[stage.process] = 1.0 + sum(
            shares[family] * affinity.get(family, 0.0)
            for family in shares
        )
    return adjustments, sorted(informative)


def exact_allocate(total: float, weighted: Sequence[tuple[str, float]]) -> dict[str, float]:
    if total < 0:
        raise ProjectionError("cannot allocate a negative duration")
    positive = [(name, weight) for name, weight in weighted if weight > 0]
    if not positive:
        if abs(total) <= 1e-12:
            return {name: 0.0 for name, _ in weighted}
        raise ProjectionError("positive source bucket has no weighted process")
    denominator = sum(weight for _, weight in positive)
    result = {name: 0.0 for name, _ in weighted}
    running = 0.0
    for name, weight in positive[:-1]:
        value = total * weight / denominator
        result[name] = value
        running += value
    result[positive[-1][0]] = total - running
    return result


def allocate_event(
    *,
    variant: VariantSpec,
    event: FxEvent,
    source: LayerOccurrence,
    match: str,
    process_weight_mode: str,
    kernel_split_mode: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[FxStage]] = collections.defaultdict(list)
    for stage in event.stages:
        groups[PROCESS_BUCKET[stage.process]].append(stage)
    for bucket in ("attn", "mlp", "outer"):
        if not groups[bucket]:
            raise ProjectionError(
                f"{event.event_id} has no FX process for source bucket {bucket}"
            )

    family_adjustment = {
        stage.process: 1.0 for stage in event.stages
    }
    family_fields_used: list[str] = []
    effective_mode = kernel_split_mode
    if kernel_split_mode == "family-aware":
        family_adjustment, family_fields_used = family_weight_adjustments(
            event.stages, source.family_ms
        )
        if not family_fields_used:
            effective_mode = "component-total-fallback"

    bucket_kernel = {
        "attn": source.attn_kernel_ms,
        "mlp": source.mlp_kernel_ms,
        "outer": source.outer_kernel_ms,
        "metadata": 0.0,
    }
    bucket_cpu = {
        "attn": source.attn_cpu_ms,
        "mlp": source.mlp_cpu_ms,
        "outer": source.outer_cpu_ms,
        "metadata": 0.0,
    }
    kernel_allocations: dict[str, float] = {}
    cpu_allocations: dict[str, float] = {}
    base_weights = {
        stage.process: process_base_weight(stage, process_weight_mode)
        for stage in event.stages
    }
    for bucket, stages in groups.items():
        if bucket == "metadata":
            for stage in stages:
                kernel_allocations[stage.process] = 0.0
                cpu_allocations[stage.process] = 0.0
            continue
        kernel_allocations.update(
            exact_allocate(
                bucket_kernel[bucket],
                [
                    (
                        stage.process,
                        base_weights[stage.process]
                        * family_adjustment[stage.process],
                    )
                    for stage in stages
                ],
            )
        )
        cpu_allocations.update(
            exact_allocate(
                bucket_cpu[bucket],
                [
                    (stage.process, base_weights[stage.process])
                    for stage in stages
                ],
            )
        )

    rows: list[dict[str, Any]] = []
    for stage in event.stages:
        bucket = PROCESS_BUCKET[stage.process]
        rows.append(
            {
                "variant": variant.slug,
                "display_name": variant.display_name,
                "fx_event_id": event.event_id,
                "layer": source.layer_idx,
                "phase": source.phase,
                "process": stage.process,
                "title": stage.title,
                "allocated_cupti_kernel_ms": csv_number(
                    kernel_allocations[stage.process]
                ),
                "allocated_nvtx_cpu_ms": csv_number(
                    cpu_allocations[stage.process]
                ),
                "fx_q_len": event.q_len,
                "fx_kv_len": event.kv_len,
                "match": match,
                "perf_q_len": source.q_len,
                "perf_kv_len": source.kv_len,
                "nodes": stage.nodes,
                "bucket": bucket,
                "source_bucket_cupti_kernel_ms": csv_number(
                    bucket_kernel[bucket]
                ),
                "source_bucket_nvtx_cpu_ms": csv_number(bucket_cpu[bucket]),
                "source_total_cupti_kernel_ms": csv_number(
                    source.total_kernel_ms
                ),
                "source_total_nvtx_cpu_ms": csv_number(source.total_cpu_ms),
                "source_forward_id": source.forward_id,
                "source_occurrence": source.occurrence,
                "source_occurrence_key": source.occurrence_key,
                "component_source_event_id": source.component_source_event_id,
                "component_kernel_normalization_factor": csv_number(
                    source.kernel_normalization_factor
                ),
                "component_cpu_normalization_factor": csv_number(
                    source.cpu_normalization_factor
                ),
                "process_weight_mode": process_weight_mode,
                "kernel_split_mode_requested": kernel_split_mode,
                "kernel_split_mode_effective": effective_mode,
                "source_kernel_families_used": ";".join(family_fields_used),
            }
        )
    allocated_total = sum(
        float(row["allocated_cupti_kernel_ms"]) for row in rows
    )
    if not math.isclose(
        allocated_total, source.total_kernel_ms, rel_tol=0, abs_tol=2e-10
    ):
        raise ProjectionError(
            f"kernel allocation does not conserve {event.event_id}: "
            f"{allocated_total} != {source.total_kernel_ms}"
        )
    return rows


def source_rows_for_report(
    event: FxEvent,
    source: LayerOccurrence,
    match: str,
) -> list[dict[str, Any]]:
    evidence_suffix = (
        f"R02 strict component; kernel scale "
        f"{source.kernel_normalization_factor:.9f}; CPU scale "
        f"{source.cpu_normalization_factor:.9f}"
    )
    return [
        {
            "fx_event": event.event_id,
            "layer": source.layer_idx,
            "phase": source.phase,
            "component": "total",
            "cupti_kernel_ms": source.total_kernel_ms,
            "nvtx_cpu_ms": source.total_cpu_ms,
            "fx_qkv": f"{event.q_len}/{event.kv_len}",
            "match": match,
            "perf_qkv": f"{source.q_len}/{source.kv_len}",
            "evidence": "R01 launch-owned layer denominator",
        },
        {
            "fx_event": event.event_id,
            "layer": source.layer_idx,
            "phase": source.phase,
            "component": "attn",
            "cupti_kernel_ms": source.attn_kernel_ms,
            "nvtx_cpu_ms": source.attn_cpu_ms,
            "fx_qkv": f"{event.q_len}/{event.kv_len}",
            "match": match,
            "perf_qkv": f"{source.q_len}/{source.kv_len}",
            "evidence": evidence_suffix,
        },
        {
            "fx_event": event.event_id,
            "layer": source.layer_idx,
            "phase": source.phase,
            "component": "mlp",
            "cupti_kernel_ms": source.mlp_kernel_ms,
            "nvtx_cpu_ms": source.mlp_cpu_ms,
            "fx_qkv": f"{event.q_len}/{event.kv_len}",
            "match": match,
            "perf_qkv": f"{source.q_len}/{source.kv_len}",
            "evidence": evidence_suffix,
        },
    ]


def render_report(
    *,
    spec: VariantSpec,
    source_rows: Sequence[dict[str, Any]],
    detail_rows: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    fx_root: Path,
    process_db: Path,
    process_inventory: Path,
    process_weight_mode: str,
    kernel_split_mode: str,
) -> str:
    lines = [
        f"# {spec.display_name} SAME_INPUT Process-wise Performance Report",
        "",
        "## Sources",
        "",
        (
            f"- R01 layer kernel denominator: `{spec.layer_breakdown}` "
            f"(SHA-256 `{sha256_file(spec.layer_breakdown)}`)."
        ),
        (
            f"- R01 HIPTX layer host duration: `{spec.layer_events}` "
            f"(SHA-256 `{sha256_file(spec.layer_events)}`)."
        ),
        (
            f"- R02 measured component supplement: `{process_db}` "
            f"(SHA-256 `{sha256_file(process_db)}`)."
        ),
        (
            f"- R02 process inventory: `{process_inventory}` "
            f"(SHA-256 `{sha256_file(process_inventory)}`)."
        ),
        (
            f"- Structural FX root: `{fx_root}`. FX artifacts supply stages and "
            "node counts only; their timing is not consumed."
        ),
        (
            "- Compatibility vocabulary: **CUPTI kernel ms** means the hipprof "
            "HIPOPS launch-owned kernel sum; **NVTX CPU ms** means HIPTX host "
            "range duration."
        ),
        "",
        "## Coverage",
        "",
        (
            "| total FX events | exact | nearest_shape | unmatched | "
            "source rows | process rows |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {coverage['total']} | {coverage['exact']} | "
            f"{coverage['nearest_shape']} | {coverage['unmatched']} | "
            f"{len(source_rows)} | {len(detail_rows)} |"
        ),
        "",
        "Nearest-shape rows are exploratory attribution and are not strict evidence.",
        "",
        "## Layer/component Source Latency",
        "",
        (
            "| fx_event | layer | phase | component | **CUPTI kernel ms** | "
            "**NVTX CPU ms** | fx q/kv | match | perf q/kv | evidence |"
        ),
        "|---|---:|---|---|---:|---:|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(row["fx_event"]),
                    str(row["layer"]),
                    markdown_cell(row["phase"]),
                    markdown_cell(row["component"]),
                    markdown_number(float(row["cupti_kernel_ms"])),
                    markdown_number(float(row["nvtx_cpu_ms"])),
                    markdown_cell(row["fx_qkv"]),
                    markdown_cell(row["match"]),
                    markdown_cell(row["perf_qkv"]),
                    markdown_cell(row["evidence"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## FX Process Latency Attribution",
            "",
            (
                "| fx_event | layer | phase | process | title | "
                "**CUPTI kernel ms** | **NVTX CPU ms** | fx q/kv | match | "
                "perf q/kv | nodes | bucket | source bucket kernel ms | "
                "source bucket CPU ms |"
            ),
            "|---|---:|---|---|---|---:|---:|---|---|---|---:|---|---:|---:|",
        ]
    )
    for row in detail_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(row["fx_event_id"]),
                    str(row["layer"]),
                    markdown_cell(row["phase"]),
                    markdown_cell(row["process"]),
                    markdown_cell(row["title"]),
                    markdown_number(float(row["allocated_cupti_kernel_ms"])),
                    markdown_number(float(row["allocated_nvtx_cpu_ms"])),
                    f"{row['fx_q_len']}/{row['fx_kv_len']}",
                    markdown_cell(row["match"]),
                    f"{row['perf_q_len']}/{row['perf_kv_len']}",
                    str(row["nodes"]),
                    markdown_cell(row["bucket"]),
                    markdown_number(
                        float(row["source_bucket_cupti_kernel_ms"])
                    ),
                    markdown_number(
                        float(row["source_bucket_nvtx_cpu_ms"])
                    ),
                ]
            )
            + " |"
        )
    effective_modes = sorted(
        {str(row["kernel_split_mode_effective"]) for row in detail_rows}
    )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            (
                "- This is process-wise attribution, not direct per-process "
                "timing. R02 direct process measurements are used only to form "
                "the measured attn/mlp source buckets."
            ),
            (
                "- `outer = R01 total - normalized R02 attn - normalized R02 "
                "mlp`; the ambiguous shared residual-add/RMSNorm transition is "
                "therefore counted once in outer."
            ),
            (
                "- R02 HIPTX component CPU durations come from the separately "
                "instrumented process capture. Only their attn/MLP ratio is "
                "used, and it is always normalized to the R01 layer HIPTX CPU "
                "denominator; it is not cross-run absolute timing."
            ),
            (
                f"- Process weighting mode: `{process_weight_mode}`. Requested "
                f"kernel split mode: `{kernel_split_mode}`; effective modes: "
                f"`{', '.join(effective_modes)}`."
            ),
            (
                "- `match=exact` refers to layer/phase/q_len/kv_len equality. "
                "`match=nearest_shape` is an explicitly exploratory fallback."
            ),
            (
                "- Source bucket columns are pre-allocation measured/normalized "
                "bucket totals; timing columns are the allocated estimates."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def summary_rows(detail_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    phase_totals: dict[tuple[str, str], float] = collections.defaultdict(float)
    for row in detail_rows:
        key = (str(row["variant"]), str(row["phase"]), str(row["process"]))
        record = grouped.setdefault(
            key,
            {
                "variant": key[0],
                "phase": key[1],
                "process": key[2],
                "title": row["title"],
                "matched_layer_occurrences": 0,
                "allocated_cupti_kernel_ms": 0.0,
                "allocated_nvtx_cpu_ms": 0.0,
                "exact_rows": 0,
                "nearest_shape_rows": 0,
            },
        )
        record["matched_layer_occurrences"] += 1
        record["allocated_cupti_kernel_ms"] += float(
            row["allocated_cupti_kernel_ms"]
        )
        record["allocated_nvtx_cpu_ms"] += float(
            row["allocated_nvtx_cpu_ms"]
        )
        record[f"{row['match']}_rows"] += 1
        phase_totals[(key[0], key[1])] += float(
            row["allocated_cupti_kernel_ms"]
        )
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        record = grouped[key]
        phase_total = phase_totals[(key[0], key[1])]
        kernel_ms = float(record["allocated_cupti_kernel_ms"])
        result.append(
            {
                **record,
                "allocated_cupti_kernel_ms": csv_number(kernel_ms),
                "allocated_nvtx_cpu_ms": csv_number(
                    float(record["allocated_nvtx_cpu_ms"])
                ),
                "pct_of_variant_phase_kernel": csv_number(
                    100.0 * kernel_ms / phase_total if phase_total else 0.0
                ),
            }
        )
    return result


def render_aggregate_report(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# SAME_INPUT Process-wise Performance Breakdown",
        "",
        (
            "Secondary aggregate view. Per-variant reports remain the primary "
            "process-attribution artifacts."
        ),
        "",
        (
            "| variant | phase | process | title | matched occurrences | "
            "CUPTI kernel ms | NVTX CPU ms | phase kernel % |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(row["variant"]),
                    markdown_cell(row["phase"]),
                    markdown_cell(row["process"]),
                    markdown_cell(row["title"]),
                    str(row["matched_layer_occurrences"]),
                    markdown_number(float(row["allocated_cupti_kernel_ms"])),
                    markdown_number(float(row["allocated_nvtx_cpu_ms"])),
                    markdown_number(float(row["pct_of_variant_phase_kernel"])),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def infer_single_env(name: str) -> list[str]:
    value = os.environ.get(name)
    return [value] if value else []


def aligned_values(
    values: Sequence[str] | None,
    *,
    env_name: str,
    count: int | None = None,
) -> list[str]:
    resolved = list(values or infer_single_env(env_name))
    if count is not None and len(resolved) != count:
        raise ProjectionError(
            f"{env_name}/CLI count {len(resolved)} does not match variants {count}"
        )
    return resolved


def build_variant_specs(args: argparse.Namespace) -> list[VariantSpec]:
    variants = aligned_values(args.variants, env_name="VARIANT")
    if not variants:
        raise ProjectionError("provide --variant or VARIANT")
    count = len(variants)
    displays = aligned_values(
        args.display_names, env_name="DISPLAY_NAME", count=count
    )
    breakdowns = aligned_values(
        args.layer_breakdowns, env_name="LAYER_BREAKDOWN", count=count
    )
    layer_events = aligned_values(
        args.layer_events, env_name="LAYER_EVENTS", count=count
    )
    contract_ids = aligned_values(
        args.contract_ids, env_name="CONTRACT_ID", count=count
    )
    contract_hashes = aligned_values(
        args.contract_hashes, env_name="CONTRACT_SHA256", count=count
    )
    specs: list[VariantSpec] = []
    seen_slugs: set[str] = set()
    for values in zip(
        variants,
        displays,
        breakdowns,
        layer_events,
        contract_ids,
        contract_hashes,
        strict=True,
    ):
        variant, display, breakdown, events, contract_id, contract_hash = values
        slug = sanitize_slug(variant)
        if slug in seen_slugs:
            raise ProjectionError(f"duplicate variant slug: {slug}")
        if not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
            raise ProjectionError(f"invalid contract SHA-256 for {slug}")
        seen_slugs.add(slug)
        specs.append(
            VariantSpec(
                slug=slug,
                display_name=display,
                layer_breakdown=require_file(
                    Path(breakdown), role=f"{slug} layer breakdown"
                ),
                layer_events=require_file(
                    Path(events), role=f"{slug} layer events"
                ),
                contract_id=contract_id,
                contract_sha256=contract_hash,
            )
        )
    return specs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Project SAME_INPUT Qwen layer/component latency onto successful "
            "FX process reconstructions."
        )
    )
    result.add_argument("--variant", dest="variants", action="append")
    result.add_argument("--display-name", dest="display_names", action="append")
    result.add_argument(
        "--layer-breakdown", dest="layer_breakdowns", action="append"
    )
    result.add_argument("--layer-events", dest="layer_events", action="append")
    result.add_argument("--contract-id", dest="contract_ids", action="append")
    result.add_argument(
        "--contract-sha256", dest="contract_hashes", action="append"
    )
    result.add_argument("--fx-root")
    result.add_argument("--process-db")
    result.add_argument("--process-runtime-events")
    result.add_argument("--process-inventory")
    result.add_argument("--report-root")
    result.add_argument("--output-root")
    result.add_argument("--runtime-artifact-root")
    result.add_argument(
        "--match-mode",
        choices=("exact", "exact-or-nearest"),
        default=os.environ.get("MATCH_MODE", "exact-or-nearest"),
    )
    result.add_argument(
        "--process-weight-mode",
        choices=("semantic-cost", "node-count"),
        default=os.environ.get("PROCESS_WEIGHT_MODE", "semantic-cost"),
    )
    result.add_argument(
        "--kernel-split-mode",
        choices=("family-aware", "component-total"),
        default=os.environ.get("KERNEL_SPLIT_MODE", "family-aware"),
    )
    result.add_argument("--write-aggregate-report", action="store_true")
    return result


def arg_or_env(value: str | None, env_name: str) -> str:
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise ProjectionError(f"provide CLI value or {env_name}")
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = build_variant_specs(args)
    parent_contracts = {
        (spec.contract_id, spec.contract_sha256) for spec in specs
    }
    if len(parent_contracts) != 1:
        raise ProjectionError(
            "one process component supplement cannot bind multiple contracts"
        )
    parent_contract_id, parent_contract_hash = next(iter(parent_contracts))

    runtime_artifact_root = require_dir(
        Path(arg_or_env(args.runtime_artifact_root, "RUNTIME_ARTIFACT_ROOT")),
        role="runtime artifact root",
    )
    fx_root = require_dir(
        Path(arg_or_env(args.fx_root, "FX_ROOT")), role="FX root"
    )
    process_db = require_file(
        Path(arg_or_env(args.process_db, "PROCESS_DB")), role="R02 process DB"
    )
    process_runtime_events = require_file(
        Path(
            arg_or_env(
                args.process_runtime_events, "PROCESS_RUNTIME_EVENTS"
            )
        ),
        role="R02 runtime layer events",
    )
    process_inventory = require_file(
        Path(arg_or_env(args.process_inventory, "PROCESS_INVENTORY")),
        role="R02 process inventory",
    )
    report_root = require_under(
        Path(arg_or_env(args.report_root, "REPORT_ROOT")),
        runtime_artifact_root,
        role="report root",
    )
    output_root = require_under(
        Path(arg_or_env(args.output_root, "OUTPUT_ROOT")),
        runtime_artifact_root,
        role="CSV output root",
    )
    report_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_events = load_selected_runtime_events(
        process_runtime_events,
        contract_id=parent_contract_id,
        contract_sha256=parent_contract_hash,
    )
    inventory = load_process_inventory(process_inventory)
    (
        measurement_rows,
        components,
        strict_ownership_audit,
    ) = strict_process_measurements(
        process_db,
        runtime_events,
        inventory,
    )
    fx_events = load_fx_events(fx_root)

    measurement_fields = [
        "event_id",
        "process_id",
        "process_title",
        "fragment_id",
        "aggregation_key",
        "status",
        "nvtx_range_name",
        "component_bucket_role",
        "hiptx_cpu_ms",
        "strict_owned_kernel_count",
        "strict_owned_hipops_ms",
        "runtime_index_examples",
        "device_ids",
        "ownership_method",
        "marker_index",
    ]
    process_measurement_csv = output_root / "r02_process_component_measurements.csv"
    atomic_write_csv(
        process_measurement_csv, measurement_rows, measurement_fields
    )

    all_detail_rows: list[dict[str, Any]] = []
    all_source_csv_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    normalization_rows_all: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    variant_audits: dict[str, Any] = {}
    per_variant_csv_paths: list[Path] = []

    for spec in specs:
        occurrences = load_layer_occurrences(spec)
        selected_sources, normalization_rows = attach_component_evidence(
            occurrences, runtime_events, components
        )
        for row in normalization_rows:
            row["variant"] = spec.slug
        normalization_rows_all.extend(normalization_rows)

        detail_rows: list[dict[str, Any]] = []
        report_source_rows: list[dict[str, Any]] = []
        unmatched_ids: list[str] = []
        exact_count = 0
        nearest_count = 0
        for event in fx_events:
            source, match = match_event(
                event,
                selected_sources,
                match_mode=args.match_mode,
            )
            if source is None or match is None:
                unmatched_ids.append(event.event_id)
                continue
            if match == "exact":
                exact_count += 1
            else:
                nearest_count += 1
            event_detail = allocate_event(
                variant=spec,
                event=event,
                source=source,
                match=match,
                process_weight_mode=args.process_weight_mode,
                kernel_split_mode=args.kernel_split_mode,
            )
            detail_rows.extend(event_detail)
            report_source_rows.extend(source_rows_for_report(event, source, match))

        coverage = {
            "variant": spec.slug,
            "total": len(fx_events),
            "exact": exact_count,
            "nearest_shape": nearest_count,
            "unmatched": len(unmatched_ids),
            "matched": exact_count + nearest_count,
            "unmatched_event_ids": ";".join(unmatched_ids),
        }
        coverage_rows.append(coverage)
        expected_source_rows = 3 * coverage["matched"]
        expected_process_rows = sum(
            len(event.stages)
            for event in fx_events
            if event.event_id
            in {row["fx_event_id"] for row in detail_rows}
        )
        if len(report_source_rows) != expected_source_rows:
            raise ProjectionError(f"{spec.slug}: source report row count mismatch")
        if len(detail_rows) != expected_process_rows:
            raise ProjectionError(f"{spec.slug}: process report row count mismatch")

        report_path = (
            report_root
            / f"SAME_INPUT_{spec.slug.upper()}_PROCESS_WISE_PERFORMANCE_REPORT.md"
        )
        atomic_write_text(
            report_path,
            render_report(
                spec=spec,
                source_rows=report_source_rows,
                detail_rows=detail_rows,
                coverage=coverage,
                fx_root=fx_root,
                process_db=process_db,
                process_inventory=process_inventory,
                process_weight_mode=args.process_weight_mode,
                kernel_split_mode=args.kernel_split_mode,
            ),
        )
        report_paths.append(report_path)

        variant_csv = (
            output_root / f"same_input_{spec.slug}_process_attribution.csv"
        )
        atomic_write_csv(variant_csv, detail_rows, DETAIL_FIELDS)
        per_variant_csv_paths.append(variant_csv)

        source_csv_rows = [
            {
                "variant": spec.slug,
                **row,
                "cupti_kernel_ms": csv_number(float(row["cupti_kernel_ms"])),
                "nvtx_cpu_ms": csv_number(float(row["nvtx_cpu_ms"])),
            }
            for row in report_source_rows
        ]
        all_source_csv_rows.extend(source_csv_rows)
        all_detail_rows.extend(detail_rows)

        conservation: dict[str, float] = collections.defaultdict(float)
        source_totals: dict[str, float] = {}
        for row in detail_rows:
            event_id = str(row["fx_event_id"])
            conservation[event_id] += float(row["allocated_cupti_kernel_ms"])
            source_totals[event_id] = float(
                row["source_total_cupti_kernel_ms"]
            )
        max_error = max(
            (
                abs(value - source_totals[event_id])
                for event_id, value in conservation.items()
            ),
            default=0.0,
        )
        if max_error > 2e-10:
            raise ProjectionError(
                f"{spec.slug}: conservation error {max_error} exceeds tolerance"
            )
        variant_audits[spec.slug] = {
            "coverage": coverage,
            "eligible_component_source_occurrences": len(selected_sources),
            "source_report_rows": len(report_source_rows),
            "process_report_rows": len(detail_rows),
            "expected_source_report_rows": expected_source_rows,
            "expected_process_report_rows": expected_process_rows,
            "max_kernel_conservation_error_ms": max_error,
            "normalization_events": [
                row["event_id"]
                for row in normalization_rows
                if float(row["kernel_normalization_factor"]) < 1.0
                or float(row["cpu_normalization_factor"]) < 1.0
            ],
        }

    aggregate_detail_csv = output_root / "same_input_process_attribution.csv"
    atomic_write_csv(aggregate_detail_csv, all_detail_rows, DETAIL_FIELDS)
    summaries = summary_rows(all_detail_rows)
    summary_fields = [
        "variant",
        "phase",
        "process",
        "title",
        "matched_layer_occurrences",
        "allocated_cupti_kernel_ms",
        "allocated_nvtx_cpu_ms",
        "exact_rows",
        "nearest_shape_rows",
        "pct_of_variant_phase_kernel",
    ]
    summary_csv = output_root / "same_input_process_summary.csv"
    atomic_write_csv(summary_csv, summaries, summary_fields)
    coverage_fields = [
        "variant",
        "total",
        "matched",
        "exact",
        "nearest_shape",
        "unmatched",
        "unmatched_event_ids",
    ]
    coverage_csv = output_root / "same_input_process_coverage.csv"
    atomic_write_csv(coverage_csv, coverage_rows, coverage_fields)
    source_fields = [
        "variant",
        "fx_event",
        "layer",
        "phase",
        "component",
        "cupti_kernel_ms",
        "nvtx_cpu_ms",
        "fx_qkv",
        "match",
        "perf_qkv",
        "evidence",
    ]
    source_csv = output_root / "same_input_layer_component_source.csv"
    atomic_write_csv(source_csv, all_source_csv_rows, source_fields)
    normalization_fields = [
        "variant",
        "event_id",
        "forward_id",
        "layer",
        "phase",
        "q_len",
        "kv_len",
        "r01_total_cupti_kernel_ms",
        "r02_raw_attn_cupti_kernel_ms",
        "r02_raw_mlp_cupti_kernel_ms",
        "kernel_normalization_factor",
        "normalized_attn_cupti_kernel_ms",
        "normalized_mlp_cupti_kernel_ms",
        "outer_cupti_kernel_ms",
        "r01_total_nvtx_cpu_ms",
        "r02_raw_attn_nvtx_cpu_ms",
        "r02_raw_mlp_nvtx_cpu_ms",
        "cpu_normalization_factor",
        "normalized_attn_nvtx_cpu_ms",
        "normalized_mlp_nvtx_cpu_ms",
        "outer_nvtx_cpu_ms",
    ]
    normalization_csv = output_root / "same_input_component_normalization.csv"
    atomic_write_csv(
        normalization_csv, normalization_rows_all, normalization_fields
    )

    aggregate_report_path: Path | None = None
    if args.write_aggregate_report:
        aggregate_report_path = (
            report_root / "SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md"
        )
        atomic_write_text(
            aggregate_report_path, render_aggregate_report(summaries)
        )

    output_paths = (
        report_paths
        + per_variant_csv_paths
        + [
            aggregate_detail_csv,
            summary_csv,
            coverage_csv,
            source_csv,
            normalization_csv,
            process_measurement_csv,
        ]
    )
    if aggregate_report_path is not None:
        output_paths.append(aggregate_report_path)
    output_index = {
        str(path): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in output_paths
    }
    audit = {
        "schema_version": 1,
        "status": "pass",
        "generated_utc": utc_now(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "options": {
            "match_mode": args.match_mode,
            "process_weight_mode": args.process_weight_mode,
            "kernel_split_mode": args.kernel_split_mode,
            "write_aggregate_report": args.write_aggregate_report,
        },
        "inputs": {
            "fx_root": str(fx_root),
            "fx_layer_events_sha256": sha256_file(
                fx_root / "fx_layer_events.csv"
            ),
            "process_db": str(process_db),
            "process_db_sha256": sha256_file(process_db),
            "process_runtime_events": str(process_runtime_events),
            "process_runtime_events_sha256": sha256_file(
                process_runtime_events
            ),
            "process_inventory": str(process_inventory),
            "process_inventory_sha256": sha256_file(process_inventory),
            "variants": {
                spec.slug: {
                    "display_name": spec.display_name,
                    "contract_id": spec.contract_id,
                    "contract_sha256": spec.contract_sha256,
                    "layer_breakdown": str(spec.layer_breakdown),
                    "layer_breakdown_sha256": sha256_file(
                        spec.layer_breakdown
                    ),
                    "layer_events": str(spec.layer_events),
                    "layer_events_sha256": sha256_file(spec.layer_events),
                }
                for spec in specs
            },
        },
        "strict_process_component_ownership": strict_ownership_audit,
        "filtered_fx_event_count": len(fx_events),
        "filtered_fx_process_stage_count": sum(
            len(event.stages) for event in fx_events
        ),
        "variant_validation": variant_audits,
        "aggregate_detail_rows": len(all_detail_rows),
        "aggregate_summary_rows": len(summaries),
        "coverage_rows": len(coverage_rows),
        "component_measurement_rows": len(measurement_rows),
        "component_normalization_rows": len(normalization_rows_all),
        "component_normalization_policy": {
            "kernel": (
                "normalize R02 attn/MLP to the R01 launch-owned kernel "
                "denominator; reject raw component sums more than 10% above "
                "the denominator"
            ),
            "cpu": (
                "use the separately instrumented R02 HIPTX attn/MLP ratio and "
                "always normalize it to the R01 layer HIPTX denominator"
            ),
        },
        "outputs": output_index,
        "evidence_boundary": {
            "output_is_process_attribution": True,
            "output_is_direct_process_timing": False,
            "r01_role": "canonical complete layer denominator",
            "r02_role": "strict measured attn/mlp component supplement only",
            "fx_role": "structural stages and weights only",
            "nearest_shape_is_strict_evidence": False,
        },
    }
    audit_path = output_root / "R03_GENERATION_AUDIT.json"
    atomic_write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "reports": [str(path) for path in report_paths],
                "audit": str(audit_path),
                "coverage": coverage_rows,
                "detail_rows": len(all_detail_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return audit


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except ProjectionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
