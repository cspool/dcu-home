#!/usr/bin/env python3
"""Collect fine-grained, original-run DCU SE active-CU snapshots.

This sidecar intentionally does not call ``hy-smi --show*util``: those values
are documented and observed as previous-second aggregates.  The installed
gfx936 driver exposes ``rsmi_dev_se_util_get`` instead.  It returns one active
CU percentage per shader engine and can be polled without starting a HIP
workload.  Every sample records both CLOCK_MONOTONIC and CLOCK_REALTIME bounds
so a consumer can align it to the same-run hipprof request and retain the alignment
uncertainty.

The requested poll period is a target, not a claim.  The summary reports the
empirical cadence and call latency.  Downstream code must reject a process
window that does not contain enough real samples.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


MAX_SE_COUNT = 8
RSMI_STATUS_SUCCESS = 0


class SeUsage(ctypes.Structure):
    _fields_ = [("percent", ctypes.c_float * MAX_SE_COUNT)]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def load_rsmi(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.rsmi_init.argtypes = [ctypes.c_uint64]
    library.rsmi_init.restype = ctypes.c_int
    library.rsmi_shut_down.argtypes = []
    library.rsmi_shut_down.restype = ctypes.c_int
    library.rsmi_num_monitor_devices.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    library.rsmi_num_monitor_devices.restype = ctypes.c_int
    library.rsmi_dev_se_util_get.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(SeUsage),
    ]
    library.rsmi_dev_se_util_get.restype = ctypes.c_int
    library.rsmi_dev_cu_num_get.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.rsmi_dev_cu_num_get.restype = ctypes.c_int
    return library


def wait_for_arm(arm_file: Path, stop_file: Path, maximum_wait_s: float) -> bool:
    deadline = time.monotonic() + maximum_wait_s
    while not arm_file.exists():
        if stop_file.exists() or time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll current-run DCU shader-engine active-CU utilization."
    )
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--interval-us", type=int, default=500)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("/opt/hyhal/lib/librocm_smi64.so"),
    )
    parser.add_argument("--maximum-arm-wait-seconds", type=float, default=1800.0)
    parser.add_argument("--maximum-sample-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.interval_us < 100:
        parser.error("--interval-us must be at least 100")
    if args.maximum_arm_wait_seconds <= 0 or args.maximum_sample_seconds <= 0:
        parser.error("maximum durations must be positive")
    return args


def main() -> int:
    args = parse_args()
    for path in (
        args.output_jsonl,
        args.summary_json,
        args.ready_file,
        args.arm_file,
        args.stop_file,
    ):
        path = path.resolve()
        if path.exists():
            raise RuntimeError(f"refusing an existing collector path: {path}")
    if not args.library.is_file():
        raise RuntimeError(f"ROCm SMI library is missing: {args.library}")

    terminating = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal terminating
        terminating = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    library = load_rsmi(args.library.resolve())
    init_status = library.rsmi_init(0)
    if init_status != RSMI_STATUS_SUCCESS:
        raise RuntimeError(f"rsmi_init failed with status {init_status}")

    output_handle = None
    status = "failed"
    failure: str | None = None
    samples: list[dict[str, Any]] = []
    armed = False
    try:
        device_count = ctypes.c_uint32()
        count_status = library.rsmi_num_monitor_devices(ctypes.byref(device_count))
        if count_status != RSMI_STATUS_SUCCESS:
            raise RuntimeError(
                f"rsmi_num_monitor_devices failed with status {count_status}"
            )
        if args.device >= device_count.value:
            raise RuntimeError(
                f"device {args.device} is outside [0, {device_count.value})"
            )

        cu_count = ctypes.c_int()
        cu_status = library.rsmi_dev_cu_num_get(
            args.device, ctypes.byref(cu_count)
        )
        probe = SeUsage()
        probe_start = time.monotonic_ns()
        probe_status = library.rsmi_dev_se_util_get(
            args.device, ctypes.byref(probe)
        )
        probe_end = time.monotonic_ns()
        if probe_status != RSMI_STATUS_SUCCESS:
            raise RuntimeError(
                f"rsmi_dev_se_util_get failed with status {probe_status}"
            )

        ready_payload = {
            "schema_version": 1,
            "status": "ready_waiting_for_arm",
            "collector_pid": os.getpid(),
            "physical_device_index": args.device,
            "monitor_device_count": device_count.value,
            "cu_count": cu_count.value if cu_status == 0 else None,
            "cu_count_status": cu_status,
            "metric": "se_active_cu_pct",
            "metric_semantics": (
                "instantaneous ratio of active CUs for each shader engine"
            ),
            "evidence_class": "observed_in_same_run_request",
            "requested_interval_us": args.interval_us,
            "probe_call_latency_ns": probe_end - probe_start,
            "library": str(args.library.resolve()),
            "clock_alignment": (
                "sample carries realtime and monotonic call bounds; hipprof "
                "alignment uses realtime midpoint with half-call uncertainty"
            ),
            "not_equivalent_to": [
                "hy-smi previous-second HCU utilization",
                "hy-smi previous-second CU utilization",
                "hy-smi previous-second wave utilization",
                "HBM bandwidth",
            ],
        }
        atomic_write_json(args.ready_file, ready_payload)
        armed = wait_for_arm(
            args.arm_file,
            args.stop_file,
            args.maximum_arm_wait_seconds,
        )
        if not armed:
            status = "stopped_before_arm"
            return 3

        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output_jsonl.open("x", encoding="utf-8")
        interval_ns = args.interval_us * 1000
        sample_start_monotonic_ns = time.monotonic_ns()
        deadline_ns = sample_start_monotonic_ns
        maximum_end_ns = sample_start_monotonic_ns + int(
            args.maximum_sample_seconds * 1_000_000_000
        )
        sequence = 0
        while not terminating and not args.stop_file.exists():
            now_ns = time.monotonic_ns()
            if now_ns >= maximum_end_ns:
                status = "maximum_sample_duration_reached"
                break
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1_000_000_000)

            before_monotonic_ns = time.monotonic_ns()
            before_realtime_ns = time.time_ns()
            usage = SeUsage()
            read_status = library.rsmi_dev_se_util_get(
                args.device, ctypes.byref(usage)
            )
            after_realtime_ns = time.time_ns()
            after_monotonic_ns = time.monotonic_ns()
            values = [float(value) for value in usage.percent]
            finite_values = [value for value in values if math.isfinite(value)]
            sample = {
                "schema_version": 1,
                "sequence": sequence,
                "physical_device_index": args.device,
                "metric": "se_active_cu_pct",
                "unit": "percent",
                "read_status": read_status,
                "monotonic_begin_ns": before_monotonic_ns,
                "monotonic_end_ns": after_monotonic_ns,
                "monotonic_midpoint_ns": (
                    before_monotonic_ns + after_monotonic_ns
                )
                // 2,
                "realtime_begin_ns": before_realtime_ns,
                "realtime_end_ns": after_realtime_ns,
                "realtime_midpoint_ns": (
                    before_realtime_ns + after_realtime_ns
                )
                // 2,
                "alignment_uncertainty_ns": max(
                    after_monotonic_ns - before_monotonic_ns,
                    after_realtime_ns - before_realtime_ns,
                )
                // 2,
                "call_latency_ns": after_monotonic_ns - before_monotonic_ns,
                "se_active_cu_pct": values,
                "mean_se_active_cu_pct": (
                    sum(finite_values) / len(finite_values)
                    if finite_values
                    else None
                ),
                "max_se_active_cu_pct": (
                    max(finite_values) if finite_values else None
                ),
                "evidence_class": (
                    "observed_in_same_run_request"
                    if read_status == RSMI_STATUS_SUCCESS
                    else "unavailable"
                ),
            }
            output_handle.write(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            if sequence % 256 == 0:
                output_handle.flush()
            samples.append(sample)
            sequence += 1
            deadline_ns += interval_ns
            if deadline_ns <= after_monotonic_ns:
                missed = (after_monotonic_ns - deadline_ns) // interval_ns + 1
                deadline_ns += missed * interval_ns
        else:
            status = "complete"
        if args.stop_file.exists() and status == "failed":
            status = "complete"
        if terminating and status == "failed":
            status = "terminated"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if output_handle is not None:
            output_handle.flush()
            output_handle.close()
        shutdown_status = library.rsmi_shut_down()
        midpoints = [sample["realtime_midpoint_ns"] for sample in samples]
        cadence = [
            later - earlier for earlier, later in zip(midpoints, midpoints[1:])
        ]
        call_latencies = [sample["call_latency_ns"] for sample in samples]
        successful = [sample for sample in samples if sample["read_status"] == 0]
        nonzero = [
            sample
            for sample in successful
            if (sample["mean_se_active_cu_pct"] or 0.0) > 0.0
        ]
        summary = {
            "schema_version": 1,
            "status": status,
            "failure": failure,
            "armed": armed,
            "collector_pid": os.getpid(),
            "physical_device_index": args.device,
            "metric": "se_active_cu_pct",
            "metric_semantics": (
                "instantaneous ratio of active CUs for each shader engine"
            ),
            "evidence_class": "observed_in_same_run_request",
            "requested_interval_us": args.interval_us,
            "sample_count": len(samples),
            "successful_sample_count": len(successful),
            "nonzero_sample_count": len(nonzero),
            "read_error_count": len(samples) - len(successful),
            "cadence_ns": {
                "minimum": min(cadence) if cadence else None,
                "p50": percentile(cadence, 0.50),
                "p95": percentile(cadence, 0.95),
                "maximum": max(cadence) if cadence else None,
            },
            "call_latency_ns": {
                "minimum": min(call_latencies) if call_latencies else None,
                "p50": percentile(call_latencies, 0.50),
                "p95": percentile(call_latencies, 0.95),
                "maximum": max(call_latencies) if call_latencies else None,
            },
            "empirical_sub_millisecond_cadence": {
                "p50": (
                    percentile(cadence, 0.50) < 1_000_000 if cadence else False
                ),
                "p95": (
                    percentile(cadence, 0.95) < 1_000_000 if cadence else False
                ),
            },
            "clock_alignment": {
                "hipprof_axis": "CLOCK_REALTIME nanoseconds",
                "sample_timestamp": "realtime_midpoint_ns",
                "per_sample_uncertainty_field": "alignment_uncertainty_ns",
            },
            "output_jsonl": str(args.output_jsonl.resolve()),
            "library": str(args.library.resolve()),
            "shutdown_status": shutdown_status,
            "limitations": [
                "not HCU, wave-residency, or memory-bandwidth utilization",
                "empirical cadence must be checked per capture",
                "a process metric requires real samples inside its exact window",
            ],
        }
        atomic_write_json(args.summary_json, summary)
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"live utilization collector failed: {exc}", file=sys.stderr)
        raise
