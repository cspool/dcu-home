# SPDX-License-Identifier: Apache-2.0
import functools, torch; from typing import Any
GFX936_GDN_T4096_COMPILER_OPTIONS = {'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}
def unwrap_triton_jit(kernel: Any) -> Any:
    visited = set()
    while id(kernel) not in visited:
        visited.add(id(kernel))
        if type(kernel).__name__ == 'JITFunction' or all((hasattr(kernel, x) for x in ('run', 'cache_key'))):
            return kernel
        kernel = getattr(kernel, 'fn', None)
        if kernel is None:
            break
    raise TypeError(f'Could not unwrap Triton JITFunction from {kernel!r}')
@functools.cache
def _is_gfx936_device(device_index: int) -> bool:
    try: properties = torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError): return False
    return getattr(properties, 'gcnArchName', '').split(':', 1)[0] == 'gfx936'
def use_gfx936_gdn_chunk_o_config(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None, scale: float, chunk_size: int, cu_seqlens: torch.Tensor | None) -> bool:
    tensors = (q, k, v, g, cu_seqlens)
    return q.device.type == 'cuda' and q.device.index is not None and _is_gfx936_device(q.device.index) and (scale == 128 ** (-0.5)) and (chunk_size == 64) and (q.ndim == v.ndim == 4) and (k.shape == q.shape) and (q.shape[0] == 1) and (q.shape[1] >= 1) and (q.shape[2:] == (16, 128)) and (v.shape == (1, q.shape[1], 48, 128)) and (g is not None) and (g.shape == (1, q.shape[1], 48)) and (q.dtype == k.dtype == v.dtype == torch.bfloat16) and (g.dtype == torch.float32) and (cu_seqlens is not None) and (cu_seqlens.dtype == torch.int32) and (cu_seqlens.ndim == 1) and (cu_seqlens.numel() >= 2) and all((t.device == q.device and t.is_contiguous() for t in tensors))
def use_gfx936_gdn_t4096_config(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, chunk_size: int, initial_state: torch.Tensor | None, output_final_state: bool, cu_seqlens: torch.Tensor | None) -> bool:
    return use_gfx936_gdn_chunk_o_config(q, k, v, g, scale, chunk_size, cu_seqlens) and output_final_state and (q.shape == (1, 4096, 16, 128)) and (beta.shape == g.shape) and (beta.dtype == torch.bfloat16) and (beta.device == q.device) and beta.is_contiguous() and (cu_seqlens.numel() == 2) and (initial_state is None or (initial_state.shape == (1, 48, 128, 128) and initial_state.dtype == torch.float32 and (initial_state.device == q.device) and initial_state.is_contiguous()))
