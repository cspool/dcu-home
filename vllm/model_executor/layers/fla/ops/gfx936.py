# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

from vllm.triton_utils import tl, triton

from . import fused_recurrent as recurrent

_DECODE_CONFIG = dict(
    H=16,
    HV=48,
    K=128,
    V=128,
    BK=128,
    BV=32,
    SOFTPLUS_THRESHOLD=20.0,
    num_warps=4,
    num_stages=1,
)


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and is_gfx936(tensor.device)


def use_packed_decode(tensor: torch.Tensor) -> bool:
    return tensor.shape[0] <= 3 and use_gfx936(tensor)


def qwen35_packed_decode(**args):
    state = args["initial_state"]
    names = ("mixed_qkv", "a", "b", "A_log", "dt_bias", "out")
    tensors = tuple(args[n] for n in names) + (state, state, args["ssm_state_indices"])
    grid = (4, tensors[0].shape[0] * 48)
    recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        *tensors,
        args["scale"],
        *(tensor.stride(0) for tensor in (*tensors[:3], *tensors[-3:])),
        USE_QK_L2NORM_IN_KERNEL=args["use_qk_l2norm_in_kernel"],
        **_DECODE_CONFIG,
    )
    return args["out"], state


@triton.jit
def _gdn_rmsnorm(x, z, weight, output, eps):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    column = tl.arange(0, 128)
    offset = row[:, None] * 128 + column[None, :]
    z_offset = row[:, None] // 48 * 16384
    z_offset += row[:, None] % 48 * 128 + column[None, :]
    value = tl.load(x + offset).to(tl.float32)
    gate = tl.load(z + z_offset).to(tl.float32)
    variance = tl.sum(value * value, axis=1) / 128
    result = value * tl.rsqrt(variance + eps)[:, None]
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
