# SPDX-License-Identifier: Apache-2.0
"""Narrow gfx936 fast paths shared by Qwen3.5 layers."""

from functools import cache

import torch

_K5120_ROWS_PER_BLOCK = {
    (96, 5120): 4,
    (14336, 5120): 2,
    (16384, 5120): 2,
    (34816, 5120): 2,
    (248320, 5120): 2,
}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    """Return whether *device* is a gfx936 accelerator."""
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def qwen35_k5120_gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    *,
    fuse_silu: bool = False,
) -> torch.Tensor | None:
    """Run a Qwen3.5 BF16 K=5120 GEMV, or decline with ``None``."""
    output_features, input_features = weight.shape
    supported_input = (
        x.numel() == input_features
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and x.stride(-1) == 1
        and x.is_cuda
        and is_gfx936(x.device)
    )
    if not supported_input:
        return None

    if fuse_silu:
        rows_per_block = -2 if weight.shape == (34816, 5120) else None
    else:
        rows_per_block = _K5120_ROWS_PER_BLOCK.get(weight.shape)
    if rows_per_block is None:
        return None

    output = torch.ops._rocm_C.LLMM1(
        weight,
        x.reshape(1, input_features),
        rows_per_block,
    )
    final_features = output_features // 2 if fuse_silu else output_features
    return output.reshape(*x.shape[:-1], final_features)
