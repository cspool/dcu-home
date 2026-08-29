#!/usr/bin/env python3
"""Freeze the live gfx936 resource limits used by Workflow05."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe physical DCU resource limits.")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rocminfo", type=Path, default=Path("/opt/dtk/bin/rocminfo"))
    parser.add_argument("--hy-smi", type=Path, default=Path("/opt/hyhal/bin/hy-smi"))
    parser.add_argument(
        "--rsmi-library",
        type=Path,
        default=Path("/opt/hyhal/lib/librocm_smi64.so"),
    )
    parser.add_argument(
        "--hipprof",
        type=Path,
        default=Path("/opt/dtk/bin/hipprof"),
    )
    parser.add_argument(
        "--hipprof-llvm",
        type=Path,
        default=Path("/opt/dtk-26.04-DCC2602-0317/dcc/lib"),
    )
    parser.add_argument("--consolidator", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    ).stdout


def first_int(pattern: str, text: str, name: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not resolve {name}")
    return int(match.group(1))


def main() -> int:
    args = parse_args()
    if args.device != 1:
        raise RuntimeError("reviewed Workflow05 policy requires physical DCU 1")
    for path in (
        args.rocminfo,
        args.hy_smi,
        args.rsmi_library,
        args.hipprof,
        args.hipprof_llvm,
        args.consolidator,
    ):
        if not path.is_file():
            if path != args.hipprof_llvm or not path.is_dir():
                raise RuntimeError(f"required capability source is missing: {path}")
    if args.output.exists():
        raise RuntimeError(f"refusing existing output: {args.output}")

    rocminfo = run([str(args.rocminfo.resolve())])
    gpu_suffix = rocminfo[rocminfo.find("Name:                    gfx936") :]
    if not gpu_suffix:
        raise RuntimeError("rocminfo does not expose gfx936")
    wave_size = first_int(r"Wavefront Size:\s+(\d+)", gpu_suffix, "wave size")
    wave_limit = first_int(
        r"Max Waves Per CU:\s+(\d+)", gpu_suffix, "wave limit"
    )
    product = run(
        [
            str(args.hy_smi.resolve()),
            "--showproductname",
            "--showuniqueid",
            "--showdriverversion",
            "-d",
            str(args.device),
        ]
    )
    if "HCU[1]" not in product or "Card Series:" not in product:
        raise RuntimeError("hy-smi did not bind physical HCU 1")

    hipprof_env = dict(os.environ)
    prior_ld_library_path = hipprof_env.get("LD_LIBRARY_PATH", "")
    hipprof_env["LD_LIBRARY_PATH"] = str(args.hipprof_llvm.resolve()) + (
        f":{prior_ld_library_path}" if prior_ld_library_path else ""
    )
    hipprof_help = run(
        [str(args.hipprof.resolve()), "-h"], env=hipprof_env
    )
    required_pmc_flags = (
        "--pmc",
        "--pmc-read",
        "--pmc-write",
        "--pmc-type",
        "--kernel-name",
    )
    missing_pmc_flags = [
        flag for flag in required_pmc_flags if flag not in hipprof_help
    ]
    if missing_pmc_flags:
        raise RuntimeError(
            f"hipprof lacks required bounded PMC flags: {missing_pmc_flags}"
        )

    library = ctypes.CDLL(str(args.rsmi_library.resolve()))
    library.rsmi_init.argtypes = [ctypes.c_uint64]
    library.rsmi_init.restype = ctypes.c_int
    library.rsmi_shut_down.argtypes = []
    library.rsmi_shut_down.restype = ctypes.c_int
    library.rsmi_dev_cu_num_get.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.rsmi_dev_cu_num_get.restype = ctypes.c_int
    if library.rsmi_init(0) != 0:
        raise RuntimeError("rsmi_init failed")
    try:
        cu_count = ctypes.c_int()
        status = library.rsmi_dev_cu_num_get(args.device, ctypes.byref(cu_count))
        if status != 0 or cu_count.value <= 0:
            raise RuntimeError(f"rsmi_dev_cu_num_get failed with status {status}")
    finally:
        shutdown_status = library.rsmi_shut_down()
    if shutdown_status != 0:
        raise RuntimeError(f"rsmi_shut_down failed with status {shutdown_status}")

    consolidator_text = args.consolidator.read_text(encoding="utf-8")
    thread_limit = first_int(
        r"by_thread\s*=\s*(\d+)\s*//", consolidator_text, "thread limit"
    )
    vgpr_resource = first_int(
        r"by_vgpr\s*=\s*(\d+)\s*//", consolidator_text, "VGPR resource"
    )
    shared_memory_bytes = first_int(
        r"else\s+(\d+)\s*//\s*max\(1,\s*int\(shared\)\)",
        consolidator_text,
        "shared-memory resource",
    )
    if thread_limit != wave_size * wave_limit:
        raise RuntimeError("thread limit does not equal wave_size * wave_limit")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified",
        "physical_device_id": args.device,
        "architecture": "gfx936",
        "cu_count": cu_count.value,
        "wave_size": wave_size,
        "wave_limit": wave_limit,
        "thread_limit": thread_limit,
        "vgpr_resource": vgpr_resource,
        "shared_memory_bytes": shared_memory_bytes,
        "resource_semantics": (
            "limits used for a theoretical occupancy/coexistence upper bound; "
            "not achieved occupancy"
        ),
        "counter_availability": {
            "status": "verified_collector_modes",
            "verification_scope": (
                "live hipprof CLI mode/filter availability; per-kernel field "
                "presence is revalidated from each fresh capture"
            ),
            "compute_cache_stall": {
                "available": True,
                "collector_flag": "--pmc",
                "expected_fields": [
                    "processed_alu_instructions",
                    "l2_cache_hit_rate",
                    "l1_cache_unit_is_stalled",
                    "l2_cache_write_unit_is_stalled",
                    "shared_memory_bank_conflict",
                    "work_group_size",
                    "vgpr_count",
                    "sgpr_count",
                    "shared_memory_size",
                ],
            },
            "l2_read": {
                "available": True,
                "collector_flag": "--pmc-read",
                "expected_fields": ["size_of_l2_cache_read"],
            },
            "l2_write": {
                "available": True,
                "collector_flag": "--pmc-write",
                "expected_fields": ["size_of_l2_cache_write"],
            },
            "literal_kernel_filter": {
                "available": True,
                "collector_flag": "--kernel-name",
            },
        },
        "unavailable_quantities": {
            "hbm_or_dram_bytes": (
                "no verified HBM/DRAM traffic counter in the selected hipprof modes"
            ),
            "hbm_or_dram_bandwidth": (
                "must not be inferred from logical FX tensor bytes"
            ),
            "achieved_occupancy_pct": (
                "only a theoretical gfx936 resource upper bound is modeled"
            ),
        },
        "sources": [
            {
                "role": "live_architecture_wave_probe",
                "path": str(args.rocminfo.resolve()),
                "sha256": sha256_file(args.rocminfo.resolve()),
                "stdout_sha256": hashlib.sha256(rocminfo.encode()).hexdigest(),
            },
            {
                "role": "live_physical_device_and_driver_probe",
                "path": str(args.hy_smi.resolve()),
                "sha256": sha256_file(args.hy_smi.resolve()),
                "stdout_sha256": hashlib.sha256(product.encode()).hexdigest(),
            },
            {
                "role": "live_cu_count_api",
                "path": str(args.rsmi_library.resolve()),
                "sha256": sha256_file(args.rsmi_library.resolve()),
                "function": "rsmi_dev_cu_num_get",
            },
            {
                "role": "audited_current_resource_formula",
                "path": str(args.consolidator.resolve()),
                "sha256": sha256_file(args.consolidator.resolve()),
            },
            {
                "role": "live_pmc_mode_and_literal_filter_probe",
                "path": str(args.hipprof.resolve()),
                "sha256": sha256_file(args.hipprof.resolve()),
                "help_stdout_sha256": hashlib.sha256(
                    hipprof_help.encode()
                ).hexdigest(),
                "verified_flags": list(required_pmc_flags),
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
