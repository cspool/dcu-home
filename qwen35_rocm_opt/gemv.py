# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from .target import use_gfx936

K5120Provider = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
_GEMV_CONFIG = {"num_warps": 16, "num_stages": 1, "waves_per_eu": 1}
_K5120_SHAPES = frozenset((96, 14336, 16384, 34816, 248320))


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


def qwen35_gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    k5120_provider: K5120Provider | None = None,
) -> torch.Tensor | None:
    """Dispatch the two validated single-token GEMV implementations.

    The K=17408 path is fully portable Triton. K=5120 deliberately remains a
    host-injected HIP provider because all recorded replacement layouts lost
    to its exact 640-thread pair reduction.
    """
    m, k = weight.shape
    layout = weight.is_contiguous() and x.stride(-1) == 1
    target = (x.numel(), x.dtype, weight.dtype, layout)
    if not use_gfx936(x) or target != (k, torch.bfloat16, torch.bfloat16, True):
        return None
    if (m, k) == (5120, 17408):
        output = torch.empty((1, m), dtype=x.dtype, device=x.device)
        _output_gemv[(m,)](weight, x.reshape(1, k), output, **_GEMV_CONFIG)
    elif k == 5120 and m in _K5120_SHAPES and k5120_provider is not None:
        output = k5120_provider(weight, x.reshape(1, k), 4 if m == 96 else 2)
    else:
        return None
    return output.reshape(*x.shape[:-1], m)


def requires_k5120_provider(weight: torch.Tensor, x: torch.Tensor) -> bool:
    """Allow adapters and readiness checks to fail closed before serving."""
    return (
        weight.ndim == 2
        and weight.shape[1] == 5120
        and weight.shape[0] in _K5120_SHAPES
        and x.numel() == 5120
        and x.dtype == weight.dtype == torch.bfloat16
    )
