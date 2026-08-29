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

_K5120_OUTPUT_FEATURES = {
    # TP=1 local output sizes.
    96,
    14336,
    16384,
    34816,
    248320,
    # TP=2 local output sizes.
    48,
    7168,
    8192,
    17408,
    124160,
}
_K5120_FUSED_SILU_FEATURES = {17408, 34816}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


@triton.jit
def _gdn_rmsnorm_silu_gate(
    x,
    gate,
    weight,
    output,
    eps,
    num_rows,
    num_heads,
    gate_token_stride,
    BLOCK_ROWS: tl.constexpr,
):
    # Optimization: fuse RMSNorm and SiLU(z) for TP=1/2 GDN outputs.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + columns[None, :]
    valid_rows = rows < num_rows
    gate_offsets = rows[:, None] // num_heads * gate_token_stride
    gate_offsets += rows[:, None] % num_heads * 128 + columns[None, :]
    values = tl.load(x + offsets, mask=valid_rows[:, None], other=0.0).to(
        tl.float32
    )
    gate_values = tl.load(
        gate + gate_offsets, mask=valid_rows[:, None], other=0.0
    ).to(tl.float32)
    # Correctness: keep RMSNorm and SiLU arithmetic in FP32 until the store.
    result = values * tl.rsqrt(tl.sum(values * values, axis=1) / 128 + eps)[:, None]
    result *= tl.load(weight + columns)[None, :].to(tl.float32)
    result *= gate_values * tl.sigmoid(gate_values)
    tl.store(output + offsets, result, mask=valid_rows[:, None])


def qwen35_gdn_rmsnorm(norm, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # Correctness: only the exact TP=1/2 model layouts enter the fused formula.
    target_layout = (
        x.ndim == 3
        and x.shape == gate.shape
        and x.shape[1] in (24, 48)
        and x.shape[2] == 128
        and x.is_contiguous()
        and gate.stride()[1:] == (128, 1)
    )
    if not (target_layout and x.is_cuda and is_gfx936(x.device)):
        return norm(x.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(x)
    output = torch.empty_like(x)
    num_rows = x.shape[0] * x.shape[1]
    _gdn_rmsnorm_silu_gate[(triton.cdiv(num_rows, 16),)](
        x,
        gate,
        norm.weight,
        output,
        norm.eps,
        num_rows,
        x.shape[1],
        gate.stride(0),
        BLOCK_ROWS=16,
        num_warps=4,
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
        and (not fuse_silu or output_features in _K5120_FUSED_SILU_FEATURES)
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

    rows_per_block = -2 if fuse_silu else (4 if output_features in (48, 96) else 2)
    output = torch.ops._rocm_C.LLMM1(
        weight, x.reshape(1, input_features), rows_per_block
    )
    final_features = output_features // 2 if fuse_silu else output_features
    return output.reshape(*x.shape[:-1], final_features)
