#!/usr/bin/env python3
import statistics

import torch
from flash_attn.flash_attn_interface import varlen_fwd_unified
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops import rocm_aiter_unified_attention_gqa6 as gqa6


@triton.jit
def merge2(
    out,
    a,
    la,
    b,
    lb,
    n,
    os0: tl.constexpr,
    os1: tl.constexpr,
    as0: tl.constexpr,
    as1: tl.constexpr,
    bs0: tl.constexpr,
    bs1: tl.constexpr,
    las0: tl.constexpr,
    las1: tl.constexpr,
    lbs0: tl.constexpr,
    lbs1: tl.constexpr,
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    mask = rows < n
    token, head = rows // 24, rows % 24
    dim = tl.arange(0, 256)
    x = tl.load(la + head * las0 + token * las1, mask=mask, other=-float("inf"))
    y = tl.load(lb + head * lbs0 + token * lbs1, mask=mask, other=-float("inf"))
    maximum = tl.maximum(x, y)
    wx, wy = tl.exp(x - maximum), tl.exp(y - maximum)
    va = tl.load(
        a + token[:, None] * as0 + head[:, None] * as1 + dim[None, :],
        mask=mask[:, None],
    ).to(tl.float32)
    vb = tl.load(
        b + token[:, None] * bs0 + head[:, None] * bs1 + dim[None, :],
        mask=mask[:, None],
    ).to(tl.float32)
    target = out + token[:, None] * os0 + head[:, None] * os1 + dim[None, :]
    tl.store(target, (va * wx[:, None] + vb * wy[:, None]) / (wx + wy)[:, None], mask=mask[:, None])


def run(context: int, qlen: int = 4096) -> None:
    torch.manual_seed(784 + context)
    total = context + qlen
    pages = (total + 783) // 784
    flat_k = torch.randn((pages * 784, 4, 256), device="cuda", dtype=torch.bfloat16)
    flat_v = torch.randn_like(flat_k)
    key_cache = flat_k.view(pages, 784, 4, 256)
    value_cache = flat_v.view_as(key_cache)
    key = flat_k[context:total].contiguous()
    value = flat_v[context:total].contiguous()
    query = torch.randn((qlen, 24, 256), device="cuda", dtype=torch.bfloat16)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    cu = torch.tensor([0, qlen], device="cuda", dtype=torch.int32)
    seq = torch.tensor([total], device="cuda", dtype=torch.int32)
    base, candidate = torch.empty_like(query), torch.empty_like(query)

    full, boundary = divmod(context, 784)
    tails = full * 16
    old = tails + boundary
    residual = old + qlen
    residual_pages = (residual + 63) // 64
    packed_k = torch.empty(
        (residual_pages, 64, 4, 256), device="cuda", dtype=torch.bfloat16
    )
    packed_v = torch.empty_like(packed_k)
    packed_flat_k, packed_flat_v = packed_k.view(-1, 4, 256), packed_v.view(-1, 4, 256)
    main_out, residual_out = torch.empty_like(query), torch.empty_like(query)
    main_len = torch.tensor([full * 768], device="cuda", dtype=torch.int32)
    residual_len = torch.tensor([residual], device="cuda", dtype=torch.int32)
    residual_table = torch.arange(residual_pages, device="cuda", dtype=torch.int32)[None]
    page_ids = table[0, :full]

    def baseline() -> None:
        gqa6.prefill(
            q=query,
            k=key_cache,
            v=value_cache,
            out=base,
            block_table=table,
            seqused_k=seq,
            cu_seqlens_q=cu,
            softmax_scale=256**-0.5,
            max_seqlen_q=qlen,
        )

    def proposed() -> None:
        torch.index_select(
            key_cache[:, 768:], 0, page_ids, out=packed_flat_k[:tails].view(full, 16, 4, 256)
        )
        torch.index_select(
            value_cache[:, 768:], 0, page_ids, out=packed_flat_v[:tails].view(full, 16, 4, 256)
        )
        if boundary:
            packed_flat_k[tails:old].copy_(key_cache[table[0, full], :boundary])
            packed_flat_v[tails:old].copy_(value_cache[table[0, full], :boundary])
        packed_flat_k[old:residual].copy_(key)
        packed_flat_v[old:residual].copy_(value)
        a, la = varlen_fwd_unified(
            query,
            key_cache[:, :768],
            value_cache[:, :768],
            cu,
            main_len,
            table[:, :full],
            qlen,
            full * 768,
            softmax_scale=256**-0.5,
            causal=False,
            window_size=(-1, -1),
            out=main_out,
            return_softmax_lse=True,
        )
        b, lb = varlen_fwd_unified(
            query,
            packed_k,
            packed_v,
            cu,
            residual_len,
            residual_table,
            qlen,
            residual,
            softmax_scale=256**-0.5,
            causal=True,
            window_size=(-1, -1),
            out=residual_out,
            return_softmax_lse=True,
        )
        la = la[0] if la.ndim == 3 else la
        lb = lb[0] if lb.ndim == 3 else lb
        merge2[(triton.cdiv(qlen * 24, 4),)](
            candidate,
            a,
            la,
            b,
            lb,
            qlen * 24,
            candidate.stride(0),
            candidate.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            la.stride(0),
            la.stride(1),
            lb.stride(0),
            lb.stride(1),
            num_warps=4,
        )

    baseline()
    proposed()
    torch.cuda.synchronize()
    delta = candidate.float() - base.float()
    samples = {"base": [], "candidate": []}
    for repeat in range(7):
        order = (("base", baseline), ("candidate", proposed))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    base_us = statistics.median(samples["base"])
    candidate_us = statistics.median(samples["candidate"])
    print(
        dict(
            context=context,
            qlen=qlen,
            base_us=base_us,
            candidate_us=candidate_us,
            reduction_percent=100 * (base_us - candidate_us) / base_us,
            max_abs=delta.abs().max().item(),
            mean_abs=delta.abs().mean().item(),
            rmse=delta.square().mean().sqrt().item(),
            finite=bool(torch.isfinite(candidate).all()),
        )
    )


if __name__ == "__main__":
    run(8192)
    run(12288)
