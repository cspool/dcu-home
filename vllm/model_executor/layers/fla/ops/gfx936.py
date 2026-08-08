# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

from vllm.triton_utils import tl, triton

_CHUNK_O = {t: ({"BK": 32, "BV": 32}, 2, t // 16 + 1) for t in (16, 32)} | {
    64: ({"BK": 32, "BV": 64}, 4, 2),
    4096: ({"BK": 128, "BV": 128}, 4, 1),
}
_COMPILER = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and is_gfx936(tensor.device)


def gdn_kernel(kernel, tensor, schedule, **meta):
    shape = tensor.shape
    if schedule is None or shape[:1] != (1,) or shape[2:] not in (
        (16, 128), (48, 128), (48, 64)
    ) or not use_gfx936(tensor):
        return kernel, {}
    config, warps, stages = schedule
    options = config | meta | {"num_warps": warps, "num_stages": stages}
    return kernel.fn.fn, options | (_COMPILER if stages == 1 else {})


@triton.jit
def _gdn_rmsnorm(x, z, weight, output, eps):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    column = tl.arange(0, 128)
    offset = row[:, None] * 128 + column[None, :]
    z_offset = row[:, None] // 48 * 16384
    z_offset += row[:, None] % 48 * 128 + column[None, :]
    value = tl.load(x + offset).to(tl.float32)
    gate = tl.load(z + z_offset).to(tl.float32)
    result = value * tl.rsqrt(tl.sum(value * value, axis=1) / 128 + eps)[:, None]
    result *= tl.load(weight + column)[None, :].to(tl.float32)
    result *= gate * tl.sigmoid(gate)
    tl.store(output + offset, result)


def qwen35_gdn_rmsnorm(norm, x, z):
    target = (x.shape[1:], x.dtype, z.dtype, norm.weight.dtype, z.stride())
    expected = ((48, 128), *((torch.bfloat16,) * 3), (16384, 128, 1))
    if not use_gfx936(x) or target != expected:
        return norm(x.reshape(-1, 128), z.reshape(-1, 128)).reshape_as(x)
    output = torch.empty_like(x)
    _gdn_rmsnorm[(x.shape[0] * 3,)](x, z, norm.weight, output, norm.eps, num_warps=4)
    return output


def qwen35_gemv(weight, x, silu=False):
    m, k = weight.shape
    target = (x.numel(), x.dtype, weight.dtype, weight.is_contiguous() and x.stride(-1) == 1)
    if not use_gfx936(x) or target != (k, torch.bfloat16, torch.bfloat16, True) or (silu and (m, k) != (34816, 5120)):
        return None
    if (m, k) == (5120, 17408):
        output = torch.ops._rocm_C.LLMM1(weight, x.reshape(1, k), 1)
    elif k == 5120 and m in (96, 14336, 16384, 34816, 248320):
        output = (torch.ops._rocm_C.LLMM1(weight, x.reshape(1, k), -2) if silu
                  else torch.ops._rocm_C.LLMM1(weight, x.reshape(1, k), 4 if m == 96 else 2))
    else:
        return None
    return output.reshape(*x.shape[:-1], m // (2 if silu else 1))
