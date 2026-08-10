# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import torch

from vllm.triton_utils import tl, triton

# GDN call chain: Qwen3_5GatedDeltaNet -> qwen35_gdn_rmsnorm -> Triton norm+SiLU.
# Decode call chain: ROCm linear/Qwen3NextMLP -> qwen35_k5120_gemv -> LLMM1.
# Role: centralize gfx936 detection and the Python gates for GDN and K5120.
# Work logic: exact target inputs launch the fused kernel/native op; other GDN
# inputs run the supplied official norm, while other GEMV inputs return None so
# their caller can continue the official GEMM or MLP implementation.

_K5120_OUTPUT_FEATURES = {96, 14336, 16384, 34816, 248320}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


@triton.jit
def _gdn_rmsnorm_silu_gate(x, gate, weight, output, eps):
    # Optimization: fuse RMSNorm and SiLU(z) for the fixed [T, 48, 128] GDN output.
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + columns[None, :]
    gate_offsets = rows[:, None] // 48 * 16384
    gate_offsets += rows[:, None] % 48 * 128 + columns[None, :]
    values = tl.load(x + offsets).to(tl.float32)
    gate_values = tl.load(gate + gate_offsets).to(tl.float32)
    # Correctness: keep RMSNorm and SiLU arithmetic in FP32 until the store.
    result = values * tl.rsqrt(tl.sum(values * values, axis=1) / 128 + eps)[:, None]
    result *= tl.load(weight + columns)[None, :].to(tl.float32)
    result *= gate_values * tl.sigmoid(gate_values)
    tl.store(output + offsets, result)


def qwen35_gdn_rmsnorm(norm, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # Correctness: only the exact model layout enters the fused Triton formula.
    target_layout = x.shape[1:] == (48, 128) and gate.stride() == (16384, 128, 1)
    if not (target_layout and x.is_cuda and is_gfx936(x.device)):
        return norm(x.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(x)
    output = torch.empty_like(x)
    _gdn_rmsnorm_silu_gate[(x.shape[0] * 3,)](
        x, gate, norm.weight, output, norm.eps, num_warps=4
    )
    return output


def qwen35_k5120_gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    fuse_silu: bool = False,
) -> torch.Tensor | None:
    # Optimization: dispatch Qwen QKV/GDN/MLP/LM-head single-token K=5120 shapes
    # through the existing _rocm_C.LLMM1 ABI; fuse_silu selects GateUp+SwiGLU.
    output_features, input_features = weight.shape
    supported_input = (
        input_features == 5120
        and output_features in _K5120_OUTPUT_FEATURES
        and (not fuse_silu or output_features == 34816)
        and x.numel() == input_features
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and x.stride(-1) == 1
        and x.is_cuda
        and is_gfx936(x.device)
    )
    # Correctness: unsupported shapes stay on the official GEMM/MLP path.
    if not supported_input:
        return None

    rows_per_block = -2 if fuse_silu else (4 if output_features == 96 else 2)
    output = torch.ops._rocm_C.LLMM1(
        weight, x.reshape(1, input_features), rows_per_block
    )
    final_features = output_features // 2 if fuse_silu else output_features
    return output.reshape(*x.shape[:-1], final_features)
