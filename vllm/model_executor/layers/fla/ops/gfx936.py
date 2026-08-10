# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

_K5120_OUTPUT_FEATURES = {96, 14336, 16384, 34816, 248320}
_CHUNK_O_SCHEDULES = {
    16: ({"BK": 32, "BV": 32}, 2, 2),
    32: ({"BK": 32, "BV": 32}, 2, 3),
    64: ({"BK": 32, "BV": 64}, 4, 2),
    4096: ({"BK": 128, "BV": 128}, 4, 1),
}
_SINGLE_STAGE_OPTIONS = {
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and is_gfx936(tensor.device)


def gdn_kernel(kernel, tensor: torch.Tensor, schedule, **metadata):
    shape = tensor.shape
    target_shape = shape[:1] == (1,) and shape[2:] in {
        (16, 128),
        (48, 128),
        (48, 64),
    }
    if schedule is None or not target_shape or not use_gfx936(tensor):
        return kernel, {}

    config, num_warps, num_stages = schedule
    options = (
        config
        | metadata
        | {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    if num_stages == 1:
        options |= _SINGLE_STAGE_OPTIONS
    return kernel.fn.fn, options


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
