# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

from vllm.triton_utils import tl, triton

_K5120_OUTPUT_FEATURES = {96, 14336, 16384, 34816, 248320}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and is_gfx936(tensor.device)


@triton.jit
def _gdn_rmsnorm_silu_gate(x, gate, weight, output, eps):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    column = tl.arange(0, 128)
    offset = row[:, None] * 128 + column[None, :]
    gate_offset = row[:, None] // 48 * 16384
    gate_offset += row[:, None] % 48 * 128 + column[None, :]

    value = tl.load(x + offset).to(tl.float32)
    gate_value = tl.load(gate + gate_offset).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=1) / 128 + eps)
    result = value * inverse_rms[:, None]
    result *= tl.load(weight + column)[None, :].to(tl.float32)
    result *= gate_value * tl.sigmoid(gate_value)
    tl.store(output + offset, result)


def qwen35_gdn_rmsnorm(norm, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    target = (
        x.shape[1:] == (48, 128)
        and x.dtype == gate.dtype == norm.weight.dtype == torch.bfloat16
        and gate.stride() == (16384, 128, 1)
        and use_gfx936(x)
    )
    if not target:
        return norm(x.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(x)

    output = torch.empty_like(x)
    _gdn_rmsnorm_silu_gate[(x.shape[0] * 3,)](
        x,
        gate,
        norm.weight,
        output,
        norm.eps,
        num_warps=4,
    )
    return output


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
