#!/usr/bin/env python3
import statistics
from types import SimpleNamespace

import torch
from vllm.v1.attention.ops import rocm_aiter_unified_attention_gqa6 as gqa6


def run(context: int, qlen: int = 4096) -> None:
    torch.manual_seed(784 + context)
    total = context + qlen
    pages = (total + 783) // 784
    flat_k = torch.randn((pages * 784, 4, 256), device="cuda", dtype=torch.bfloat16)
    flat_v = torch.randn_like(flat_k)
    key_cache = flat_k.view(pages, 784, 4, 256)
    value_cache = flat_v.view_as(key_cache)
    query = torch.randn((qlen, 24, 256), device="cuda", dtype=torch.bfloat16)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    cu = torch.tensor([0, qlen], device="cuda", dtype=torch.int32)
    seq = torch.tensor([total], device="cuda", dtype=torch.int32)
    base, candidate = torch.empty_like(query), torch.empty_like(query)
    meta = SimpleNamespace(max_query_len=qlen, max_seq_len=total, query_start_loc=cu,
                           block_table=table, num_actual_tokens=qlen)

    def baseline() -> None:
        gqa6.prefill(q=query, k=key_cache, v=value_cache, out=base, block_table=table,
                     seqused_k=seq, cu_seqlens_q=cu, softmax_scale=256**-0.5,
                     max_seqlen_q=qlen)

    def proposed() -> None:
        assert gqa6.page784_prefill(query, flat_k[context:total], flat_v[context:total],
                                    key_cache, value_cache, candidate, meta, 256**-0.5)

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
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    base_us, candidate_us = (statistics.median(samples[name]) for name in ("base", "candidate"))
    print(dict(context=context, qlen=qlen, base_us=base_us, candidate_us=candidate_us,
               reduction_percent=100 * (base_us - candidate_us) / base_us,
               max_abs=delta.abs().max().item(), mean_abs=delta.abs().mean().item(),
               rmse=delta.square().mean().sqrt().item(), finite=bool(torch.isfinite(candidate).all())))


if __name__ == "__main__":
    run(8192)
    run(12288)
