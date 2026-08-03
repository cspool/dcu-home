# SPDX-License-Identifier: Apache-2.0
import torch; from vllm.triton_utils import tl, triton
_OUTPUT_FEATURES, _INPUT_FEATURES, _BLOCK_K = 5120, 17408, 2048
@triton.jit
def _qwen35_output_gemv_kernel(weight_ptr, input_ptr, output_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + offsets
        x = tl.load(input_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        acc += weight * x
    tl.store(output_ptr + row, tl.sum(acc))
def qwen35_output_gemv(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((1, _OUTPUT_FEATURES), dtype=x.dtype, device=x.device)
    _qwen35_output_gemv_kernel[_OUTPUT_FEATURES,](weight, x, output, K=_INPUT_FEATURES, BLOCK_K=_BLOCK_K, num_warps=16, num_stages=1, waves_per_eu=1)
    return output
