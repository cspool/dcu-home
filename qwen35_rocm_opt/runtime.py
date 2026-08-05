# SPDX-License-Identifier: Apache-2.0
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import torch


def fixed_mrope_width(
    max_num_tokens: int,
    is_rocm: bool,
    compile_sizes: Sequence[int],
) -> int:
    """Select the persistent contiguous width used by a fixed ROCm graph."""
    fixed = is_rocm and list(compile_sizes) == [max_num_tokens]
    return max_num_tokens if fixed else max_num_tokens + 1


def get_mrope_positions(
    gpu_positions: torch.Tensor,
    scratch: torch.Tensor,
    num_tokens: int,
    max_num_tokens: int,
) -> torch.Tensor:
    """Return contiguous positions without allocating in the hot path."""
    source = gpu_positions[:, :num_tokens]
    if source.stride(0) == max_num_tokens and not source.is_contiguous():
        target = scratch[: source.numel()].view_as(source)
        return target.copy_(source)
    return source


def mrope_copy_tokens(
    gpu_positions: torch.Tensor,
    max_num_tokens: int,
    scheduled_tokens: int,
) -> int:
    """Return the H2D copy width while keeping buffer ownership in the host."""
    if gpu_positions.shape[1] == max_num_tokens:
        return max_num_tokens
    return scheduled_tokens


def tunable_profile_path(name: str) -> Path:
    """Resolve a bundled TunableOp profile independent of the host package."""
    return Path(str(files("qwen35_rocm_opt.profiles").joinpath(f"{name}.csv")))


def load_tunable_profile(name: str, device: int | torch.device) -> Path:
    """Load and validate a frozen PyTorch TunableOp profile for *device*."""
    torch.empty(0, device=device)
    path = tunable_profile_path(name)
    torch.cuda.tunable.set_filename(str(path))
    if not torch.cuda.tunable.get_results():
        raise RuntimeError(f"empty TunableOp profile: {path}")
    return path
