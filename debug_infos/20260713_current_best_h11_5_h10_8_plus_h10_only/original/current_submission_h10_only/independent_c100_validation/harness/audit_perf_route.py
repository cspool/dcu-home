#!/usr/bin/env python3
"""Audit C100 performance-service identity and route markers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.service_dir / "service_state.json").read_text())
    log = (args.service_dir / "server.log").read_text(errors="replace")
    env = (args.service_dir / "feature_env.txt").read_text()
    command = (args.service_dir / "start.command.txt").read_text()
    checks = {
        "state_c100": state.get("state") == "C100",
        "features_h10_only": state.get("features")
        == {"h10_confirmed": 1, "s32": 0, "hg3": 0},
        "candidate_wheel": state.get("wheel_sha256")
        == "f877d08fdf2380a87298006c915d14077ca947225e50e5bcf56e028fc9075d80",
        "h10_init_ready": "VLLM_ROCM_TUNABLEOP_INIT status=ready" in log,
        "h10_profile_sha": "file_sha256=41742b4c5d071fdf9085c46ad4ec1743d7e4f410431c05ff39b0e0f293548a0b" in log,
        "h10_init_controls": "enabled=1 tuning=0 record_untuned=0" in log,
        "h10_pre_capture_ready": "VLLM_ROCM_TUNABLEOP_PRE_CAPTURE status=ready" in log,
        "graph_finished": "Graph capturing finished" in log,
        "s32_ablation": "CSCC_ATTN_GQA6_S32_ABLATION" in log,
        "s32_fallback": "CSCC_ATTN_GQA6_S32_FALLBACK" in log,
        "s32_no_hit": "CSCC_ATTN_GQA6_S32_HIT" not in log,
        "hg3_fallback": "CSCC_GDN_HG3_FALLBACK" in log,
        "hg3_no_hit": "CSCC_GDN_HG3_HIT" not in log,
        "verbose_unset_recorded": "PYTORCH_TUNABLEOP_VERBOSE=<unset>" in env
        and "-u PYTORCH_TUNABLEOP_VERBOSE" in command,
        "tuning_off_recorded": "PYTORCH_TUNABLEOP_TUNING=0" in env,
        "record_off_recorded": "PYTORCH_TUNABLEOP_RECORD_UNTUNED=0" in env,
        "no_verbose_result_entry": "ResultEntry found" not in log,
        "no_tuning_search": "Finding fastest" not in log,
        "no_traceback": "Traceback" not in log,
    }
    failures = sorted(key for key, value in checks.items() if not value)
    payload = {
        "schema": "p9-c100-performance-route-audit-v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "result_entry_log_count": log.count("ResultEntry found"),
        "finding_fastest_log_count": log.count("Finding fastest"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
