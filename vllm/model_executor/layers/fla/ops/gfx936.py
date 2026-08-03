# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch

from vllm.triton_utils import tl, triton

from . import fused_recurrent as recurrent

_CHUNK_O = {t: ({"BK": 32, "BV": 32}, 2, t // 16 + 1) for t in (16, 32)} | {
    64: ({"BK": 32, "BV": 64}, 4, 2),
    4096: ({"BK": 128, "BV": 128}, 4, 1),
}
_COMPILER = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
_GEMV_CONFIG = {"num_warps": 16, "num_stages": 1, "waves_per_eu": 1}
_DECODE_CONFIG = dict(
    H=16, HV=48, K=128, V=128, BK=128, BV=32, SOFTPLUS_THRESHOLD=20.0,
    num_warps=4, num_stages=1,
)


@cache
def is_gfx936(device: int | torch.device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and is_gfx936(tensor.device)


def gdn_pruner(configs, args, **kwargs):
    args = {**args, **kwargs}
    name = next(name for name in ("q", "v", "k", "A") if name in args)
    if name == "q":
        schedule = _CHUNK_O.get(args["T"])
        shape = (1, args["T"], 16, 128)
    else:
        shape = (1, 4096, *((48, 64) if name == "A" else (16, 128)))
        schedule = ({"BK": 128} if name == "k" else {}, 4 if name == "k" else 2, 1)
    if schedule is not None and args[name].shape == shape and use_gfx936(args[name]):
        meta, warps, stages = schedule
        config = meta | (_COMPILER if stages == 1 else {})
        return [triton.Config(config, num_warps=warps, num_stages=stages)]
    return configs


def qwen35_packed_decode(**args):
    state = args["initial_state"]
    indices = args["ssm_state_indices"]
    names = ("mixed_qkv", "a", "b", "A_log", "dt_bias", "out")
    tensors = tuple(args[name] for name in names) + (state, state, indices)
    recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel[(4, 48)](
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


@triton.jit
def _output_gemv(weight, x, output):
    row = tl.program_id(0)
    offsets = tl.arange(0, 2048)
    acc = tl.zeros((2048,), dtype=tl.float32)
    for start in range(0, 17408, 2048):
        columns = start + offsets
        mask = columns < 17408
        acc += tl.load(x + columns, mask=mask, other=0.0).to(
            tl.float32
        ) * tl.load(
            weight + row * 17408 + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(output + row, tl.sum(acc))


def qwen35_gemv(weight, x):
    m, k = weight.shape
    layout = weight.is_contiguous() and x.stride(-1) == 1
    target = (x.numel(), x.dtype, weight.dtype, layout)
    if not use_gfx936(x) or target != (k, torch.bfloat16, torch.bfloat16, True):
        return None
    if (m, k) == (5120, 17408):
        output = torch.empty((1, m), dtype=x.dtype, device=x.device)
        _output_gemv[(m,)](weight, x.reshape(1, k), output, **_GEMV_CONFIG)
    elif k == 5120 and m in (96, 14336, 16384, 34816, 248320):
        output = torch.ops._rocm_C.LLMM1(weight, x.reshape(1, k), 4 if m == 96 else 2)
    else:
        return None
    return output.reshape(*x.shape[:-1], m)
