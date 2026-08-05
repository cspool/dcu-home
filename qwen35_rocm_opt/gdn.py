# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import triton
import triton.language as tl

from .target import use_gfx936

_CHUNK_O = {t: ({"BK": 32, "BV": 32}, 2, t // 16 + 1) for t in (16, 32)} | {
    64: ({"BK": 32, "BV": 64}, 4, 2),
    4096: ({"BK": 128, "BV": 128}, 4, 1),
}
_COMPILER = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
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


def gdn_pruner(configs, args, **kwargs):
    """Triton autotune pruner for the frozen Qwen3.5 GDN shapes."""
    args = {**args, **kwargs}
    name = next(name for name in ("q", "v", "k", "A") if name in args)
    if name == "q":
        schedule = _CHUNK_O.get(4096 if args["T"] == 4096 else args["BT"])
        shape = (1, args["T"], 16, 128)
    else:
        shape = (1, 4096, *((48, 64) if name == "A" else (16, 128)))
        schedule = ({"BK": 128} if name == "k" else {}, 4 if name == "k" else 2, 1)
    if schedule is not None and args[name].shape == shape and use_gfx936(args[name]):
        meta, warps, stages = schedule
        config = meta | (_COMPILER if stages == 1 else {})
        return [triton.Config(config, num_warps=warps, num_stages=stages)]
    return configs


def launch_packed_decode(kernel: Any, **args):
    """Launch a host-provided packed recurrent kernel with the frozen config.

    ``kernel`` is deliberately injected by the framework adapter because the
    recurrent implementation belongs to vLLM/FLA or the corresponding SGLang
    backend, not to this portable package.
    """
    state = args["initial_state"]
    names = ("mixed_qkv", "a", "b", "A_log", "dt_bias", "out")
    tensors = tuple(args[name] for name in names) + (
        state,
        state,
        args["ssm_state_indices"],
    )
    grid = (4, tensors[0].shape[0] * 48)
    kernel[grid](
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


def qwen35_gdn_rmsnorm(norm, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Fused GDN RMSNorm and SiLU gate with a generic host fallback."""
    target = (x.shape[1:], x.dtype, z.dtype, norm.weight.dtype, z.stride())
    expected = ((48, 128), *((torch.bfloat16,) * 3), (16384, 128, 1))
    if not use_gfx936(x) or target != expected:
        return norm(x.reshape(-1, 128), z.reshape(-1, 128)).reshape_as(x)
    output = torch.empty_like(x)
    _gdn_rmsnorm[(x.shape[0] * 3,)](
        x,
        z,
        norm.weight,
        output,
        norm.eps,
        num_warps=4,
    )
    return output
