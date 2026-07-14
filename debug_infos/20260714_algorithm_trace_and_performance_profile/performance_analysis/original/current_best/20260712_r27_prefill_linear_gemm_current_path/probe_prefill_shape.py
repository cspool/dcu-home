#!/usr/bin/env python3
"""One-dispatch BF16 F.linear probe for rocBLAS trace or PMC collection.

This script deliberately contains no timing loop and refuses to run alongside the
vLLM service.  Use the profiler's kernel substring filter to exclude allocation
and fill kernels.
"""

from __future__ import annotations

import argparse
import os
import socket

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=34816)
    parser.add_argument("--in-features", type=int, default=5120)
    args = parser.parse_args()

    if os.environ.get("PREFILL_GEMM_GPU_AUTHORIZED") != "1":
        raise SystemExit(
            "GPU mode is locked; set PREFILL_GEMM_GPU_AUTHORIZED=1 only after "
            "the device owner confirms the GPU is idle"
        )

    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=0.2):
            raise SystemExit("refuse: vLLM service is active")
    except OSError:
        pass

    torch.cuda.set_device(0)
    x = torch.ones(
        (args.tokens, args.in_features), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.ones(
        (args.out_features, args.in_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = F.linear(x, weight)
    torch.cuda.synchronize()
    print(
        "shape",
        tuple(x.shape),
        tuple(weight.shape),
        tuple(output.shape),
        "dtype",
        output.dtype,
        "sample",
        float(output[0, 0]),
    )


if __name__ == "__main__":
    main()
