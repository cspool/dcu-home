# SPDX-License-Identifier: Apache-2.0
import os
from functools import cache
from pathlib import Path

import torch

from .gemv import K5120Provider


def _registered_provider() -> K5120Provider | None:
    namespace = getattr(torch.ops, "qwen35_rocm_opt", None)
    return getattr(namespace, "k5120_gemv", None) if namespace is not None else None


@cache
def load_k5120_provider(verbose: bool = False) -> K5120Provider:
    """Load the standalone HIP provider, compiling it once when necessary.

    Set ``QWEN35_ROCM_OPT_NATIVE_LIBRARY`` to a prebuilt shared library for a
    closed-book deployment that must not JIT-build at service startup.
    """
    if provider := _registered_provider():
        return provider
    if library := os.getenv("QWEN35_ROCM_OPT_NATIVE_LIBRARY"):
        torch.ops.load_library(library)
    else:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        load(
            name="qwen35_rocm_opt_native",
            sources=[
                str(root / "csrc" / "binding.cpp"),
                str(root / "csrc" / "k5120_gemv.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    provider = _registered_provider()
    if provider is None:
        raise RuntimeError("standalone K=5120 GEMV provider was not registered")
    return provider
