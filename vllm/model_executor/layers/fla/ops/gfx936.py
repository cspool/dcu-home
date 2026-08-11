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
    48,
    96,
    7168,
    8192,
    14336,
    16384,
    17408,
    34816,
    124160,
    248320,
}
_ROW_GEMV_SHAPES = {(5120, 3072), (5120, 8704)}


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
    TOTAL_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    GATE_TOKEN_STRIDE: tl.constexpr,
):
    # Fuse the TP1/TP2 [T, H, 128] RMSNorm and SiLU(z) epilogue.
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    columns = tl.arange(0, 128)
    valid = rows < TOTAL_ROWS
    offsets = rows[:, None] * 128 + columns[None, :]
    gate_offsets = rows[:, None] // HEADS * GATE_TOKEN_STRIDE
    gate_offsets += rows[:, None] % HEADS * 128 + columns[None, :]
    values = tl.load(x + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    gate_values = tl.load(
        gate + gate_offsets, mask=valid[:, None], other=0.0
    ).to(tl.float32)
    # Correctness: keep RMSNorm and SiLU arithmetic in FP32 until the store.
    result = values * tl.rsqrt(tl.sum(values * values, axis=1) / 128 + eps)[:, None]
    result *= tl.load(weight + columns)[None, :].to(tl.float32)
    result *= gate_values * tl.sigmoid(gate_values)
    tl.store(output + offsets, result, mask=valid[:, None])


def qwen35_gdn_rmsnorm(norm, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # Correctness: only the exact model layout enters the fused Triton formula.
    heads = x.shape[1] if x.ndim == 3 else 0
    gate_token_stride = {24: 8192, 48: 16384}.get(heads)
    target_layout = x.shape[1:] == (heads, 128)
    target_layout &= gate_token_stride is not None
    target_layout &= gate.stride() == (gate_token_stride, 128, 1)
    if not (target_layout and x.is_cuda and is_gfx936(x.device)):
        return norm(x.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(x)
    output = torch.empty_like(x)
    total_rows = x.shape[0] * heads
    _gdn_rmsnorm_silu_gate[(triton.cdiv(total_rows, 8),)](
        x,
        gate,
        norm.weight,
        output,
        norm.eps,
        TOTAL_ROWS=total_rows,
        HEADS=heads,
        GATE_TOKEN_STRIDE=gate_token_stride,
        num_warps=1 if x.shape[0] < 128 else 2,
        num_stages=1,
    )
    return output


def qwen35_gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    fuse_silu: bool = False,
) -> torch.Tensor | None:
    # Optimization: dispatch Qwen QKV/GDN/MLP/LM-head single-token K=5120 shapes
    # through the existing _rocm_C.LLMM1 ABI; fuse_silu selects GateUp+SwiGLU.
    output_features, input_features = weight.shape
    supported_shape = (
        input_features == 5120 and output_features in _K5120_OUTPUT_FEATURES
    ) or (output_features, input_features) in _ROW_GEMV_SHAPES
    supported_input = (
        supported_shape
        and (
            not fuse_silu
            or (output_features, input_features)
            in {(17408, 5120), (34816, 5120)}
        )
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

    rows_per_block = -2 if fuse_silu else (4 if output_features <= 96 else 2)
    if input_features == 3072:
        rows_per_block = 4
    output = torch.ops._rocm_C.LLMM1(
        weight, x.reshape(1, input_features), rows_per_block
    )
    final_features = output_features // 2 if fuse_silu else output_features
    return output.reshape(*x.shape[:-1], final_features)
