#!/usr/bin/env python3
import statistics

import torch
from vllm.v1.attention.ops import rocm_aiter_unified_attention_gqa6 as gqa6


torch.manual_seed(784)
pages, full, boundary = 32, 15, 123
storage_k = torch.randn((pages, 784, 4, 272), device="cuda", dtype=torch.bfloat16)
storage_v = torch.randn_like(storage_k)
key_cache, value_cache = storage_k[..., :256], storage_v[..., :256]
table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None]
tails, residual = full * 16, full * 16 + boundary
expected_k = torch.empty((residual, 4, 256), device="cuda", dtype=torch.bfloat16)
expected_v = torch.empty_like(expected_k)
actual_k, actual_v = torch.empty_like(expected_k), torch.empty_like(expected_v)


def reference() -> None:
    torch.index_select(key_cache[:, 768:], 0, table[0, :full], out=expected_k[:tails].view(full, 16, 4, 256))
    torch.index_select(value_cache[:, 768:], 0, table[0, :full], out=expected_v[:tails].view(full, 16, 4, 256))
    expected_k[tails:].copy_(key_cache[table[0, full], :boundary])
    expected_v[tails:].copy_(value_cache[table[0, full], :boundary])


def candidate() -> None:
    gqa6._pack_page784[(residual, 4)](key_cache, value_cache, table, actual_k, actual_v,
                                     tails, *key_cache.stride(), num_warps=4)


reference()
candidate()
torch.cuda.synchronize()
samples = {"reference": [], "candidate": []}
for repeat in range(11):
    for name, function in (("reference", reference), ("candidate", candidate)):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples[name].append(start.elapsed_time(end) * 1000)
print(dict(stride=key_cache.stride(), k_bitwise=bool(torch.equal(actual_k, expected_k)),
           v_bitwise=bool(torch.equal(actual_v, expected_v)),
           reference_us=statistics.median(samples["reference"]),
           candidate_us=statistics.median(samples["candidate"])))
