# SPDX-License-Identifier: Apache-2.0
import flash_attn_2_cuda, torch
from flash_attn.flash_attn_interface import varlen_fwd_unified

from vllm.triton_utils import tl, triton

_PAGE_WORKSPACE, _PAGE_META = {}, {}


@triton.jit
def _pack_page784(k, v, table, packed_k, packed_v, tails, s0, s1, s2, s3):
    token, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, 256)
    is_tail = token < tails
    logical_page = tl.where(is_tail, token // 16, tails // 16)
    position = tl.where(is_tail, 768 + token % 16, token - tails)
    source = tl.load(table + logical_page) * s0 + position * s1 + head * s2 + dims * s3
    target = token * 1024 + head * 256 + dims
    tl.store(packed_k + target, tl.load(k + source))
    tl.store(packed_v + target, tl.load(v + source))


@triton.jit
def _merge_page784(out, a, la, b, lb, c, lc, rows, query_len):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    mask = row < rows
    token, head = row // 24, row % 24
    offset = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    x = tl.load(la + head * query_len + token, mask=mask, other=-float("inf"))
    y = tl.load(lb + head * query_len + token, mask=mask, other=-float("inf"))
    z = tl.load(lc + head * query_len + token, mask=mask, other=-float("inf"))
    maximum = tl.maximum(x, tl.maximum(y, z))
    wx, wy, wz = tl.exp(x - maximum), tl.exp(y - maximum), tl.exp(z - maximum)
    denominator = wx + wy + wz
    wx, wy, wz = wx / denominator, wy / denominator, wz / denominator
    va = tl.load(a + offset, mask=mask[:, None]).to(tl.float32)
    vb = tl.load(b + offset, mask=mask[:, None]).to(tl.float32)
    vc = tl.load(c + offset, mask=mask[:, None]).to(tl.float32)
    tl.store(out + offset, va * wx[:, None] + vb * wy[:, None] + vc * wz[:, None], mask=mask[:, None])


def _page_workspace(query, pages):
    key = (query.device, query.dtype)
    if key not in _PAGE_WORKSPACE:
        options = dict(device=query.device, dtype=query.dtype)
        _PAGE_WORKSPACE[key] = (
            *(torch.empty((4096, 24, 256), **options) for _ in range(2)),
            *(torch.empty((96, 64, 4, 256), **options) for _ in range(2)),
        )
    a, b, k, v = _PAGE_WORKSPACE[key]
    return a[: len(query)], b[: len(query)], k[:pages], v[:pages]


def page784_prefill(query, current_k, current_v, key_cache, value_cache, output, meta, scale):
    query_len = meta.max_query_len
    context = meta.max_seq_len - query_len
    if current_k is None or query_len < 128 or context < 784 or meta.query_start_loc.numel() != 2:
        return False
    full, boundary = divmod(context, 784)
    tails, residual = full * 16, full * 16 + boundary
    pages = (residual + 63) // 64
    if query_len > 4096 or pages > 96 or meta.num_actual_tokens != query_len:
        return False
    query, current_k, current_v, output = (
        tensor[:query_len] for tensor in (query, current_k, current_v, output)
    )
    main_out, residual_out, packed_k, packed_v = _page_workspace(query, pages)
    flat_k, flat_v = packed_k.view(-1, 4, 256), packed_v.view(-1, 4, 256)
    table, cu = meta.block_table, meta.query_start_loc
    _pack_page784[(residual, 4)](key_cache, value_cache, table, flat_k, flat_v, tails, *key_cache.stride(), num_warps=4)
    meta_key = (query.device, full, residual, pages)
    if meta_key not in _PAGE_META:
        _PAGE_META[meta_key] = (torch.tensor([full * 768], dtype=torch.int32, device=query.device), torch.tensor([residual], dtype=torch.int32, device=query.device), torch.arange(pages, dtype=torch.int32, device=query.device)[None])
    main_len, residual_len, residual_table = _PAGE_META[meta_key]
    common = dict(softmax_scale=scale, window_size=(-1, -1), return_softmax_lse=True)
    a, la = varlen_fwd_unified(query, key_cache[:, :768], value_cache[:, :768], cu, main_len, table[:, :full], query_len, full * 768, causal=False, out=main_out, **common)
    b, lb = varlen_fwd_unified(query, packed_k, packed_v, cu, residual_len, residual_table, query_len, residual, causal=False, out=residual_out, **common)
    current = flash_attn_2_cuda.varlen_fwd(
        query, current_k, current_v, output, cu, cu, None, None, None, None, query_len, query_len,
        0.0, scale, False, True, -1, -1, 0.0, False, None, None, None, None, None)
    c, lc = current[0], current[5]
    la, lb, lc = (state[0] if state.ndim == 3 else state for state in (la, lb, lc))
    _merge_page784[(triton.cdiv(query_len * 24, 4),)](output, a, la, b, lb, c, lc, query_len * 24, query_len, num_warps=4)
    return True


@triton.jit
def _gqa6(
    output,
    query,
    key_cache,
    value_cache,
    table,
    seq_lens,
    query_starts,
    scale, s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // 3
    head_group = head % 3
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 256)
    block_q: tl.constexpr = BLOCK_M // 2
    block_size: tl.constexpr = BLOCK_M
    query_start = tl.load(query_starts)
    query_len = tl.load(query_starts + 1) - query_start
    if query_block * block_q >= query_len:
        return
    local_query_pos = query_block * block_q + rows // 2
    query_pos = query_start + local_query_pos
    query_head = kv_head * 6 + head_group * 2 + rows % 2
    query_offset = query_pos[:, None] * 6144
    query_offset += query_head[:, None] * 256 + dims[None, :]
    query_mask = local_query_pos < query_len
    q = tl.load(query + query_offset, mask=query_mask[:, None], other=0.0)
    maximum = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    denominator = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, 256], dtype=tl.float32)
    context_len = tl.load(seq_lens) - query_len
    query_stop = tl.minimum((query_block + 1) * block_q, query_len)
    num_blocks = (context_len + query_stop + block_size - 1) // block_size
    width: tl.constexpr = 32 if block_size == 64 else block_size
    for block in range(0, num_blocks):
        for subtile in tl.static_range(0, block_size // width):
            columns = tl.arange(0, width)
            start = block * block_size + subtile * width
            logical_page = start // 784
            first_page = tl.load(table + logical_page)
            first_offset = start % 784
            boundary = 784 - first_offset
            second_page = tl.load(
                table + logical_page + 1,
                mask=boundary < width,
                other=first_page,
            )
            on_first_page = columns < boundary
            page = tl.where(on_first_page, first_page, second_page)
            offset = tl.where(
                on_first_page, first_offset + columns, columns - boundary
            )
            k_base = page * s0 + offset * s1 + kv_head * s2
            v_base = k_base
            k = tl.load(key_cache + k_base[None, :] + dims[:, None] * s3)
            v = tl.load(value_cache + v_base[:, None] + dims[None, :] * s3)
            causal = (
                start + columns[None, :]
                < context_len + local_query_pos[:, None] + 1
            )
            scores = scale * tl.dot(q, k)
            scores = tl.where(query_mask[:, None] & causal, scores, float("-inf"))
            block_max = tl.maximum(maximum, tl.max(scores, axis=1))
            block_max = tl.where(block_max > float("-inf"), block_max, 0.0)
            probabilities = tl.exp(scores - block_max[:, None])
            correction = tl.exp(maximum - block_max)
            accumulator *= correction[:, None]
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            maximum = block_max
            accumulator += tl.dot(probabilities.to(v.dtype), v)
    result = accumulator / denominator[:, None]
    output_offset = query_pos[:, None] * 6144
    output_offset += query_head[:, None] * 256 + dims[None, :]
    tl.store(output + output_offset, result, mask=query_mask[:, None])


def prefill(**kwargs):
    query = kwargs["q"]
    block_m = 64 if query.shape[0] >= 128 else 16
    config = {
        "num_warps": 4 if block_m == 64 or query.shape[0] < 128 else 2,
        "num_stages": 2 if query.shape[0] < 128 else 1,
        "waves_per_eu": 1,
    }
    if query.shape[0] >= 128:
        config |= {"matrix_instr_nonkdim": 16, "kpack": 2}
    _gqa6[(triton.cdiv(kwargs["max_seqlen_q"], block_m // 2), 12, 1)](
        kwargs["out"], kwargs["q"], kwargs["k"], kwargs["v"], kwargs["block_table"], kwargs["seqused_k"], kwargs["cu_seqlens_q"], kwargs["softmax_scale"], *kwargs["k"].stride(),
        BLOCK_M=block_m,
        **config,
    )
