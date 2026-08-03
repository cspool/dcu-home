# SPDX-License-Identifier: Apache-2.0
import torch; from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.gfx936 import _is_gfx936_device
@triton.jit
def _qwen35_gdn_strided_z_rmsnorm_kernel(x, z, weight, output, num_rows, x_row_stride, z_token_stride, z_head_stride, out_row_stride, eps, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    token = row // NUM_HEADS
    head = row - token * NUM_HEADS
    cols = tl.arange(0, BLOCK_N)
    mask = (row[:, None] < num_rows) & (cols[None, :] < HEAD_DIM)
    x_offsets = row[:, None] * x_row_stride + cols[None, :]
    z_offsets = token[:, None] * z_token_stride + head[:, None] * z_head_stride + cols[None, :]
    values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=1) / HEAD_DIM
    rstd = tl.rsqrt(variance + eps)
    norm_weight = tl.load(weight + cols, mask=cols < HEAD_DIM, other=0.0).to(tl.float32)
    gate = tl.load(z + z_offsets, mask=mask, other=0.0).to(tl.float32)
    result = values * rstd[:, None] * norm_weight[None, :]
    result *= gate * tl.sigmoid(gate)
    out_offsets = row[:, None] * out_row_stride + cols[None, :]
    tl.store(output + out_offsets, result, mask=mask)
def _qwen35_gdn_strided_z_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(x); num_rows = x.shape[0] * x.shape[1]; rows_per_block = 16
    _qwen35_gdn_strided_z_rmsnorm_kernel[triton.cdiv(num_rows, rows_per_block),](x, z, weight, output, num_rows, x.stride(1), z.stride(0), z.stride(1), output.stride(1), eps, NUM_HEADS=x.shape[1], HEAD_DIM=x.shape[2], BLOCK_N=triton.next_power_of_2(x.shape[2]), ROWS_PER_BLOCK=rows_per_block, num_warps=4)
    return output
def _can_use_qwen35_gdn_strided_z_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> bool:
    return x.device.type == 'cuda' and x.device.index is not None and _is_gfx936_device(x.device.index) and (x.ndim == z.ndim == 3) and (x.shape[1:] == z.shape[1:] == (48, 128)) and (weight.shape == (128,)) and (x.dtype == z.dtype == weight.dtype == torch.bfloat16) and x.is_contiguous() and (z.stride() == (16384, 128, 1)) and weight.is_contiguous()
