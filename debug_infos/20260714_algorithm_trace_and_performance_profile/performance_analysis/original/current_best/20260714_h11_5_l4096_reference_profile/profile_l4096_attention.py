#!/usr/bin/env python3
"""Synthetic-only L4096 resource trace for LA-RP/r1.

This harness never loads a model or consumes an official request.  It runs
exactly one of the two attention consumers already screened by the 600-second
microbenchmark, using the production Qwen3-Next V stride of 14336 elements.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch


SEQ_LEN = 4096
PAGE_SIZE = 784
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_SIZE = 256
QKV_ROW_WIDTH = 14336
SOFTMAX_SCALE = HEAD_SIZE**-0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("h11", "direct"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "1":
        raise RuntimeError("resource trace must be pinned to physical HCU1")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")

    # Match the service/validated microbenchmark import order on this DTK stack.
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    from triton.runtime import driver as triton_driver

    triton_driver.active.get_current_device()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible ROCm device")
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)

    import flash_attn_2_cuda
    from vllm.v1.attention.ops.rocm_aiter_unified_attention_gqa6 import (
        unified_attention_gqa6_prefill,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(
        (SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE), device=device, dtype=dtype
    ).contiguous()
    k = torch.randn(
        (SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE), device=device, dtype=dtype
    ).contiguous()
    v_storage = torch.randn(
        (SEQ_LEN, QKV_ROW_WIDTH), device=device, dtype=dtype
    )
    v = v_storage[:, -NUM_KV_HEADS * HEAD_SIZE :].view(
        SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE
    )
    output = torch.empty_like(q)
    cu = torch.tensor([0, SEQ_LEN], device=device, dtype=torch.int32)

    num_pages = math.ceil(SEQ_LEN / PAGE_SIZE)
    k_cache = torch.empty(
        (num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_SIZE),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[:SEQ_LEN].copy_(k)
    v_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[:SEQ_LEN].copy_(v)
    seq_lens = torch.tensor([SEQ_LEN], device=device, dtype=torch.int32)
    block_table = torch.arange(
        num_pages, device=device, dtype=torch.int32
    )[None, :]
    scales = torch.ones(
        (1, NUM_KV_HEADS), device=device, dtype=torch.float32
    )

    def h11() -> None:
        unified_attention_gqa6_prefill(
            q=q,
            k=k_cache,
            v=v_cache,
            out=output,
            cu_seqlens_q=cu,
            max_seqlen_q=SEQ_LEN,
            seqused_k=seq_lens,
            max_seqlen_k=SEQ_LEN,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=scales,
            v_descale=scales,
            alibi_slopes=None,
        )

    def direct() -> None:
        result = flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            output,
            cu,
            cu,
            None,
            None,
            None,
            None,
            SEQ_LEN,
            SEQ_LEN,
            0.0,
            SOFTMAX_SCALE,
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            None,
            None,
            None,
            None,
            None,
        )
        if result[0].data_ptr() != output.data_ptr():
            raise RuntimeError("FA2 did not honor the preallocated output")

    operation = h11 if args.implementation == "h11" else direct
    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    h11_source = Path(inspect.getsourcefile(unified_attention_gqa6_prefill) or "")
    fa2_object = Path(flash_attn_2_cuda.__file__ or "")
    payload = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "implementation": args.implementation,
        "physical_hcu": 1,
        "logical_device_count": torch.cuda.device_count(),
        "model_loaded": False,
        "official_requests_used": False,
        "shape": {
            "seq_len": SEQ_LEN,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "page_size": PAGE_SIZE,
        },
        "strides_elements": {
            "q": list(q.stride()),
            "k": list(k.stride()),
            "v": list(v.stride()),
            "output": list(output.stride()),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "event_ms": elapsed_ms,
        "event_ms_median": statistics.median(elapsed_ms),
        "event_ms_mean": statistics.fmean(elapsed_ms),
        "event_ms_min": min(elapsed_ms),
        "event_ms_max": max(elapsed_ms),
        "output_abs_sum": float(output.float().abs().sum().item()),
        "sources": {
            "harness": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "h11": {
                "path": str(h11_source.resolve()),
                "sha256": sha256_file(h11_source),
            },
            "flash_attn_2_cuda": {
                "path": str(fa2_object.resolve()),
                "sha256": sha256_file(fa2_object),
            },
        },
        "argv": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
