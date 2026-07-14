#!/usr/bin/env python3
"""Launch the current production GDN-prefill core kernels once each."""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path


REPO = Path("/public/home/tangyu408/vllm_cscc")
SOURCE_PROBE = Path(
    "/public/home/tangyu408/testdata/goal_runs/"
    "20260712_gdn_mfma_config_probe/gdn_mfma_config_probe.py"
)


def main() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=0.2):
            raise SystemExit("refuse: vLLM service is active on port 8001")
    except OSError:
        pass

    spec = importlib.util.spec_from_file_location("gdn_existing_probe", SOURCE_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    runtime = module.load_runtime(REPO)
    torch = runtime.torch
    torch.cuda.set_device(0)
    case = module.build_case(runtime, 512, 20260712, torch.device("cuda", 0))
    logical_names = (
        "chunk_scaled_dot_kkt",
        "solve_tril64",
        "recompute_w_u",
        "chunk_delta_h_stateful",
        "chunk_delta_h_no_state",
        "chunk_fwd_o",
    )
    print("torch", torch.__version__)
    print("arch", torch.cuda.get_device_properties(0).gcnArchName)
    for logical_name in logical_names:
        outputs = module.allocate_outputs(torch, logical_name, case)
        launch, holder = module.make_launcher(
            runtime, logical_name, case, outputs, candidate=None
        )
        compiled = launch()
        torch.cuda.synchronize()
        print(logical_name, module.compiled_kernel_metadata(holder.get("compiled") or compiled))


if __name__ == "__main__":
    main()
