#!/usr/bin/env python3
"""Small same-input and fallback checks for the focused Qwen3.5 optimizations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from flash_attn import flash_attn_varlen_func

PAGE_SIZE = 784
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_SIZE = 256
ATTENTION_SCALE = HEAD_SIZE**-0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checks",
        nargs="+",
        choices=("gqa6", "page784", "gdn", "k5120", "swiglu", "all"),
    )
    return parser.parse_args()


def require_gfx936() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx936" not in architecture:
        raise RuntimeError(f"gfx936 is required, got {architecture}")
    return architecture


def make_attention_case(
    query_len: int, context_len: int, *, interleaved_cache: bool = False
) -> SimpleNamespace:
    total_len = query_len + context_len
    page_count = (total_len + PAGE_SIZE - 1) // PAGE_SIZE
    generator = torch.Generator(device="cuda")
    generator.manual_seed(936_784 + query_len * 100_000 + context_len)
    query = torch.randn(
        (query_len, QUERY_HEADS, HEAD_SIZE),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical_key = torch.randn(
        (total_len, KV_HEADS, HEAD_SIZE),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical_value = torch.randn(
        logical_key.shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache_shape = (page_count, PAGE_SIZE, KV_HEADS, HEAD_SIZE)
    if interleaved_cache:
        cache_storage = torch.zeros(
            (page_count, 2, PAGE_SIZE, KV_HEADS, HEAD_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_cache, value_cache = cache_storage[:, 0], cache_storage[:, 1]
    else:
        key_cache = torch.zeros(cache_shape, device="cuda", dtype=torch.bfloat16)
        value_cache = torch.zeros_like(key_cache)
    for page in range(page_count):
        begin = page * PAGE_SIZE
        end = min(begin + PAGE_SIZE, total_len)
        key_cache[page, : end - begin].copy_(logical_key[begin:end])
        value_cache[page, : end - begin].copy_(logical_value[begin:end])

    current_key = logical_key[context_len:]
    current_value = logical_value[context_len:]
    if interleaved_cache:
        current_storage = torch.empty(
            (query_len, 2, KV_HEADS, HEAD_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        current_key, current_value = current_storage[:, 0], current_storage[:, 1]
        current_key.copy_(logical_key[context_len:])
        current_value.copy_(logical_value[context_len:])

    query_starts = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        max_query_len=query_len,
        max_seq_len=total_len,
        query_start_loc=query_starts,
        seq_lens=torch.tensor([total_len], device="cuda", dtype=torch.int32),
        block_table=torch.arange(page_count, device="cuda", dtype=torch.int32)[None],
        num_actual_tokens=query_len,
    )
    return SimpleNamespace(
        query=query,
        current_key=current_key,
        current_value=current_value,
        logical_key=logical_key,
        logical_value=logical_value,
        key_cache=key_cache,
        value_cache=value_cache,
        metadata=metadata,
    )


def attention_reference(case: SimpleNamespace) -> torch.Tensor:
    total_len = len(case.logical_key)
    key_starts = torch.tensor([0, total_len], device="cuda", dtype=torch.int32)
    result = flash_attn_varlen_func(
        case.query,
        case.logical_key,
        case.logical_value,
        case.metadata.query_start_loc,
        key_starts,
        case.metadata.max_query_len,
        total_len,
        softmax_scale=ATTENTION_SCALE,
        causal=True,
    )
    if not isinstance(result, torch.Tensor):
        raise AssertionError("official FlashAttention returned an unexpected result")
    return result


def error_record(name: str, result: torch.Tensor, reference: torch.Tensor) -> dict:
    difference = result.float() - reference.float()
    torch.testing.assert_close(result, reference, atol=0.04, rtol=0.04)
    return {
        "name": name,
        "finite": bool(torch.isfinite(result).all().item()),
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
    }


def check_gqa6() -> dict:
    from vllm.v1.attention.ops import rocm_aiter_unified_attention_gqa6 as gqa6

    records = []
    for query_len, context_len in ((16, 32), (32, 128), (64, 784), (128, 512)):
        case = make_attention_case(query_len, context_len)
        output = torch.empty_like(case.query)
        gqa6.prefill(
            case.query,
            case.key_cache,
            case.value_cache,
            output,
            case.metadata,
            ATTENTION_SCALE,
        )
        records.append(
            error_record(
                f"q{query_len}_context{context_len}",
                output,
                attention_reference(case),
            )
        )
    case = make_attention_case(128, 512, interleaved_cache=True)
    output = torch.empty_like(case.query)
    gqa6.prefill(
        case.query,
        case.key_cache,
        case.value_cache,
        output,
        case.metadata,
        ATTENTION_SCALE,
    )
    records.append(
        error_record(
            "q128_context512_interleaved_cache", output, attention_reference(case)
        )
    )
    return {"records": records}


def rejected_page_case(query_len: int, context_len: int) -> SimpleNamespace:
    query = torch.zeros(
        (1, QUERY_HEADS, HEAD_SIZE), device="cuda", dtype=torch.bfloat16
    )
    cache = torch.zeros(
        (1, PAGE_SIZE, KV_HEADS, HEAD_SIZE), device="cuda", dtype=torch.bfloat16
    )
    metadata = SimpleNamespace(
        max_query_len=query_len,
        max_seq_len=query_len + context_len,
        query_start_loc=torch.tensor([0, query_len], device="cuda", dtype=torch.int32),
        block_table=torch.zeros((1, 1), device="cuda", dtype=torch.int32),
        num_actual_tokens=query_len,
    )
    return SimpleNamespace(
        query=query,
        current_key=torch.zeros(
            (1, KV_HEADS, HEAD_SIZE), device="cuda", dtype=torch.bfloat16
        ),
        current_value=torch.zeros(
            (1, KV_HEADS, HEAD_SIZE), device="cuda", dtype=torch.bfloat16
        ),
        key_cache=cache,
        value_cache=cache.clone(),
        metadata=metadata,
    )


def check_page784() -> dict:
    from vllm.v1.attention.ops import (
        rocm_aiter_unified_attention_page784 as page784,
    )

    records = []
    for query_len, context_len in ((128, 784), (128, 800)):
        case = make_attention_case(query_len, context_len)
        output = torch.empty_like(case.query)
        accepted = page784.prefill(
            case.query,
            case.current_key,
            case.current_value,
            case.key_cache,
            case.value_cache,
            output,
            case.metadata,
            ATTENTION_SCALE,
        )
        if not accepted:
            raise AssertionError(
                f"page784 rejected q={query_len}, context={context_len}"
            )
        record = error_record(
            f"hit_q{query_len}_context{context_len}",
            output,
            attention_reference(case),
        )
        record["accepted"] = True
        records.append(record)

    case = make_attention_case(128, 800, interleaved_cache=True)
    output = torch.empty_like(case.query)
    accepted = page784.prefill(
        case.query,
        case.current_key,
        case.current_value,
        case.key_cache,
        case.value_cache,
        output,
        case.metadata,
        ATTENTION_SCALE,
    )
    if not accepted:
        raise AssertionError("page784 rejected interleaved cache")
    record = error_record(
        "hit_q128_context800_interleaved_cache", output, attention_reference(case)
    )
    record["accepted"] = True
    records.append(record)

    reject_shapes = (
        (127, 784, "query_below_128"),
        (128, 783, "context_below_784"),
        (4097, 784, "query_above_4096"),
        (4096, 301_840, "packed_pages_above_160"),
    )
    for query_len, context_len, name in reject_shapes:
        case = rejected_page_case(query_len, context_len)
        output = torch.full_like(case.query, 7.0)
        before = output.clone()
        accepted = page784.prefill(
            case.query,
            case.current_key,
            case.current_value,
            case.key_cache,
            case.value_cache,
            output,
            case.metadata,
            ATTENTION_SCALE,
        )
        if accepted or not torch.equal(output, before):
            raise AssertionError(f"page784 rejection contract failed: {name}")
        records.append({"name": name, "accepted": False, "output_unchanged": True})
    return {"records": records}


def make_norm(device: torch.device):
    from vllm.model_executor.layers.fla.ops.layernorm_guard import RMSNormGated

    return RMSNormGated(
        128,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        device=device,
        dtype=torch.bfloat16,
    )


def check_gdn() -> dict:
    from vllm.model_executor.layers.fla.ops.gfx936 import qwen35_gdn_rmsnorm

    records = []
    norm = make_norm(torch.device("cuda"))
    for tokens in (16, 32, 64, 128, 4096):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(936_128 + tokens)
        core = torch.randn(
            (tokens, 48, 128),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_storage = torch.randn(
            (tokens, 16384),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = gate_storage.as_strided((tokens, 48, 128), (16384, 128, 1))
        reference = norm(core.reshape(-1, 128), gate.reshape(-1, 128)).reshape_as(core)
        result = qwen35_gdn_rmsnorm(norm, core, gate)
        difference = result.float() - reference.float()
        torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
        records.append(
            {
                "name": f"tokens_{tokens}",
                "max_abs": difference.abs().max().item(),
                "mean_abs": difference.abs().mean().item(),
            }
        )
        del core, gate_storage, gate, reference, result

    core = torch.randn((16, 48, 128), device="cuda", dtype=torch.bfloat16)
    contiguous_gate = torch.randn_like(core)
    reference = norm(
        core.reshape(-1, 128), contiguous_gate.reshape(-1, 128)
    ).reshape_as(core)
    result = qwen35_gdn_rmsnorm(norm, core, contiguous_gate)
    torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
    records.append({"name": "non_target_stride_fallback", "passed": True})
    return {"records": records}


def native_library_record() -> dict:
    import vllm._rocm_C as rocm_extension

    library = Path(rocm_extension.__file__).resolve()
    return {
        "path": str(library),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def check_k5120(include_swiglu: bool) -> dict:
    import vllm._rocm_C  # noqa: F401, PLC0415

    from vllm.model_executor.layers.fla.ops.gfx936 import qwen35_k5120_gemv

    records = []
    for output_features in (96, 14336, 16384, 34816, 248320):
        torch.manual_seed(936_5120 + output_features)
        vector = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            (output_features, 5120), device="cuda", dtype=torch.bfloat16
        ).mul_(0.02)
        result = qwen35_k5120_gemv(weight, vector)
        if result is None:
            raise AssertionError(f"K5120 rejected M={output_features}")
        reference = functional.linear(vector, weight)
        difference = result.float() - reference.float()
        torch.testing.assert_close(result, reference, atol=0.75, rtol=0.03)
        records.append(
            {
                "name": f"m{output_features}",
                "output_shape": list(result.shape),
                "max_abs": difference.abs().max().item(),
                "mean_abs": difference.abs().mean().item(),
            }
        )
        del vector, weight, result, reference
        torch.cuda.empty_cache()

    if include_swiglu:
        torch.manual_seed(936_34816)
        vector = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((34816, 5120), device="cuda", dtype=torch.bfloat16).mul_(
            0.02
        )
        result = qwen35_k5120_gemv(weight, vector, fuse_silu=True)
        if result is None:
            raise AssertionError("fused GateUp/SwiGLU rejected its target shape")
        gate, up = functional.linear(vector, weight).chunk(2, dim=-1)
        reference = functional.silu(gate.float()).mul(up.float()).to(torch.bfloat16)
        difference = result.float() - reference.float()
        torch.testing.assert_close(result, reference, atol=0.75, rtol=0.03)
        records.append(
            {
                "name": "gate_up_swiglu",
                "output_shape": list(result.shape),
                "max_abs": difference.abs().max().item(),
                "mean_abs": difference.abs().mean().item(),
            }
        )

    vector = torch.zeros((1, 5120), device="cuda", dtype=torch.bfloat16)
    unsupported = torch.zeros((128, 5120), device="cuda", dtype=torch.bfloat16)
    transposed = torch.zeros((5120, 96), device="cuda", dtype=torch.bfloat16).T
    fallback_cases = {
        "unsupported_m": qwen35_k5120_gemv(unsupported, vector),
        "non_contiguous_weight": qwen35_k5120_gemv(transposed, vector),
        "multiple_input_rows": qwen35_k5120_gemv(
            torch.zeros((96, 5120), device="cuda", dtype=torch.bfloat16),
            vector.expand(2, -1),
        ),
        "fp16": qwen35_k5120_gemv(
            torch.zeros((96, 5120), device="cuda", dtype=torch.float16),
            vector.to(torch.float16),
        ),
        "bad_fused_shape": qwen35_k5120_gemv(
            torch.zeros((14336, 5120), device="cuda", dtype=torch.bfloat16),
            vector,
            fuse_silu=True,
        ),
    }
    if any(value is not None for value in fallback_cases.values()):
        raise AssertionError("a K5120 rejection case unexpectedly used the fast path")
    records.append({"name": "fallback_matrix", "passed": True})
    return {"library": native_library_record(), "records": records}


def main() -> int:
    args = parse_args()
    requested = set(args.checks)
    if "all" in requested:
        requested = {"gqa6", "page784", "gdn", "k5120", "swiglu"}
    architecture = require_gfx936()
    results = {}
    if "gqa6" in requested:
        results["gqa6"] = check_gqa6()
    if "page784" in requested:
        results["page784"] = check_page784()
    if "gdn" in requested:
        results["gdn"] = check_gdn()
    if "k5120" in requested or "swiglu" in requested:
        results["k5120"] = check_k5120(include_swiglu="swiglu" in requested)
    print(
        json.dumps(
            {
        "schema": "qwen35-focused-small-matrix-v3",
                "device": architecture,
                "results": results,
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
