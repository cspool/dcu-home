#!/usr/bin/env python3
"""Launch the current packed single-token GDN decode core once."""

from __future__ import annotations

import socket
import sys
from pathlib import Path


REPO = Path("/public/home/tangyu408/vllm_cscc")


def main() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=0.2):
            raise SystemExit("refuse: vLLM service is active on port 8001")
    except OSError:
        pass
    sys.path.insert(0, str(REPO))

    import torch
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    torch.manual_seed(20260712)
    # Qwen3.5 GDN uses 16 Q/K heads and 48 value/state heads.
    batch, heads, value_heads, key_dim, value_dim = 1, 16, 48, 128, 128
    packed_dim = 2 * heads * key_dim + value_heads * value_dim
    mixed_qkv = torch.randn(
        (batch, packed_dim), device="cuda", dtype=torch.bfloat16
    )
    a = torch.randn((batch, value_heads), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.zeros(value_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    state = torch.zeros(
        (1, value_heads, value_dim, key_dim), device="cuda", dtype=torch.float32
    )
    output = torch.empty(
        (batch, 1, value_heads, value_dim), device="cuda", dtype=torch.bfloat16
    )
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        key_dim**-0.5,
        state,
        output,
        state_indices,
        use_qk_l2norm_in_kernel=True,
        validate=True,
    )
    torch.cuda.synchronize()
    print("torch", torch.__version__)
    print("arch", torch.cuda.get_device_properties(0).gcnArchName)
    print("mode=gdn_decode", "finite", bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    main()
