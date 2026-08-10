# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

_K5120_OUTPUT_FEATURES = {96, 14336, 16384, 34816, 248320}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def qwen35_k5120_gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    *,
    fuse_silu: bool = False,
) -> torch.Tensor | None:
    output_features, input_features = weight.shape
    supported_input = (
        input_features == 5120
        and output_features in _K5120_OUTPUT_FEATURES
        and x.numel() == input_features
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and x.stride(-1) == 1
        and x.is_cuda
        and is_gfx936(x.device)
    )
    if not supported_input:
        return None

    if fuse_silu:
        rows_per_block = -2 if output_features == 34816 else None
    else:
        rows_per_block = 4 if output_features == 96 else 2
    if rows_per_block is None:
        return None

    output = torch.ops._rocm_C.LLMM1(
        weight,
        x.reshape(1, input_features),
        rows_per_block,
    )
    final_features = output_features // 2 if fuse_silu else output_features
    return output.reshape(*x.shape[:-1], final_features)
