# SPDX-License-Identifier: Apache-2.0
"""Thin vLLM adapter for the portable Qwen3.5 gfx936 package."""

from qwen35_rocm_opt.gdn import (
    gdn_pruner,
    launch_packed_decode,
    qwen35_gdn_rmsnorm,
)
from qwen35_rocm_opt.target import is_gfx936, use_gfx936

from . import fused_recurrent as recurrent

__all__ = [
    "gdn_pruner",
    "is_gfx936",
    "qwen35_gdn_rmsnorm",
    "qwen35_packed_decode",
    "use_gfx936",
]


def qwen35_packed_decode(**args):
    kernel = recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    return launch_packed_decode(kernel, **args)
