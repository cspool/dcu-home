# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"

SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

# Compiler options validated by the integrated GDN T=4096 probe on gfx936.
# Keep num_warps and kernel meta-parameters at each call site because they are
# part of the per-kernel evidence rather than common compiler policy.
GFX936_GDN_T4096_COMPILER_OPTIONS = {
    "num_stages": 1,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}


def unwrap_triton_jit(kernel: Any) -> Any:
    """Return the JITFunction underneath Triton heuristic/autotune wrappers."""
    current = kernel
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if type(current).__name__ == "JITFunction" or (
            hasattr(current, "run") and hasattr(current, "cache_key")
        ):
            return current
        if not hasattr(current, "fn"):
            break
        current = current.fn
    raise TypeError(f"Could not unwrap Triton JITFunction from {kernel!r}")


@functools.cache
def _is_gfx936_device(device_index: int) -> bool:
    try:
        properties = torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError):
        return False
    return getattr(properties, "gcnArchName", "").split(":", 1)[0] == "gfx936"


def use_gfx936_gdn_chunk_o_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Gate the shape-generic gfx936 Qwen3.5 ``chunk_fwd_o`` configs.

    ``chunk_fwd_kernel_o`` does not specialize on ``T`` and its only
    shape-dependent autotune key is ``BT``.  The target model can therefore
    use one validated config for each of BT=16/32/64 instead of autotuning a
    new 36-config set when a short residual prefill first appears.
    """
    if q.device.type != "cuda" or q.device.index is None:
        return False
    if not _is_gfx936_device(q.device.index):
        return False
    if scale != 128**-0.5 or chunk_size != 64:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    B, T, Hg, K = q.shape
    if B != 1 or T < 1 or (Hg, K) != (16, 128):
        return False
    if v.shape != (B, T, 48, 128):
        return False
    if g is None or g.shape != (B, T, 48):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if v.dtype != torch.bfloat16 or g.dtype != torch.float32:
        return False
    if cu_seqlens is None:
        return False
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
    ):
        return False
    tensors = (q, k, v, g, cu_seqlens)
    if any(tensor.device != q.device for tensor in tensors):
        return False
    if any(not tensor.is_contiguous() for tensor in tensors):
        return False
    return True


def use_gfx936_gdn_t4096_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Exact, synchronization-free gate for the validated GDN bundle."""
    if not use_gfx936_gdn_chunk_o_config(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    ):
        return False
    if not output_final_state:
        return False
    if q.shape != (1, 4096, 16, 128) or k.shape != q.shape:
        return False
    if v.shape != (1, 4096, 48, 128):
        return False
    if g.shape != (1, 4096, 48) or beta.shape != g.shape:
        return False
    if beta.dtype != torch.bfloat16:
        return False
    if cu_seqlens.numel() != 2:
        return False
    if beta.device != q.device:
        return False
    if not beta.is_contiguous():
        return False
    if initial_state is not None:
        if (
            initial_state.shape != (1, 48, 128, 128)
            or initial_state.dtype != torch.float32
            or initial_state.device != q.device
            or not initial_state.is_contiguous()
        ):
            return False
    return True


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.accelerator.device_index(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        return "cpu"


@functools.cache
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    mapping = {
        "cuda": "nvidia",
        "hip": "amd",
        "xpu": "intel",
    }
    # return the mapped value, or the original if not found
    return mapping.get(device, device)


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = "cuda" if current_platform.is_cuda_alike() else get_available_device()
device_torch_lib = getattr(torch, device, None)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"
is_gather_supported = hasattr(triton.language, "gather")
is_tma_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9) and (
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
