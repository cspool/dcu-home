# SPDX-License-Identifier: Apache-2.0
"""vLLM provider adapter for portable Qwen3.5 GEMV dispatch."""

import torch

from qwen35_rocm_opt.gemv import qwen35_gemv as portable_qwen35_gemv
from qwen35_rocm_opt.gemv import requires_k5120_provider


def qwen35_gemv(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor | None:
    provider = torch.ops._rocm_C.LLMM1 if requires_k5120_provider(weight, x) else None
    return portable_qwen35_gemv(weight, x, provider)
