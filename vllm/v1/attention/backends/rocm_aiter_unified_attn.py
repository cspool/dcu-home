# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from functools import cache

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadataBuilder,
)

logger = init_logger(__name__)


@cache
def _is_gfx936() -> bool:
    """Resolve the exact H11.3 target without depending on other candidates."""
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "gfx936" in getattr(properties, "gcnArchName", "")


class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_UNIFIED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]:
        return RocmAiterUnifiedAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """RocmAiterUnifiedAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )


class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using aiter unified attention for RocmAiterUnifiedAttentionImpl"
        )
        from aiter.ops.triton.unified_attention import unified_attention

        self.unified_attention = unified_attention
        self._h11_3_gqa6_prefill = None
        self._page784_split_prefill = None
        self._gqa6_decode_80cu = None
        if (
            _is_gfx936()
            and num_heads == 24
            and num_kv_heads == 4
            and head_size == 256
            and kv_cache_dtype == "auto"
            and alibi_slopes is None
            and sliding_window is None
            and logits_soft_cap in (None, 0, 0.0)
            and sinks is None
            and attn_type == AttentionType.DECODER
        ):
            from vllm.v1.attention.ops.rocm_aiter_unified_attention_gqa6 import (
                unified_attention_gqa6_prefill,
            )
            from vllm.v1.attention.ops.rocm_aiter_decode_attention_gqa6 import (
                unified_attention_gqa6_decode_80cu,
            )
            from vllm.v1.attention.ops.rocm_page784_split_attention import (
                page784_split_prefill,
            )

            self._h11_3_gqa6_prefill = unified_attention_gqa6_prefill
            self._page784_split_prefill = page784_split_prefill
            self._gqa6_decode_80cu = unified_attention_gqa6_decode_80cu
            logger.info_once(
                "H11.3 mapping + H11.4 compiler layout + H11.5 wide causal "
                "tiles, aligned page784 later-Prefill, and 80-CU decode "
                "segment mapping enabled for gfx936 BF16 head256 GQA6; "
                "non-target calls keep the original AITER path"
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,
            key.shape[1] if key is not None else self.num_kv_heads,
        )

        # H11.3 fixes AITER's BLOCK_M=16/GQA6 overlap only for the exact
        # Qwen3.5 full-attention prefill shape.  All checks are CPU metadata,
        # dtype, or shape checks and do not introduce a device synchronization.
        use_h11_3_gqa6_prefill = (
            self._h11_3_gqa6_prefill is not None
            and max_seqlen_q > 1
            and query.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and query.ndim == 3
            and query.shape[1:] == (24, 256)
            and key_cache.ndim == 4
            and key_cache.shape[2:] == (4, 256)
            and value_cache.ndim == 4
            and value_cache.shape[2:] == (4, 256)
        )
        use_page784_split_prefill = (
            use_h11_3_gqa6_prefill
            and self._page784_split_prefill is not None
            and cu_seqlens_q.numel() == 2
            and num_actual_tokens == max_seqlen_q
            and max_seqlen_q >= 128
            and max_seqlen_k > max_seqlen_q
            and max_seqlen_k - max_seqlen_q >= 784
            and key is not None
            and value is not None
            and key.ndim == 3
            and value.ndim == 3
            and key.shape[0] >= num_actual_tokens
            and value.shape[0] >= num_actual_tokens
            and key.shape[1:] == (4, 256)
            and value.shape[1:] == (4, 256)
            and key_cache.shape[1] == 784
            and value_cache.shape[1] == 784
            and block_table.ndim == 2
            and block_table.shape[0] == 1
        )
        if use_page784_split_prefill:
            self._page784_split_prefill(
                query=query[:num_actual_tokens],
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                block_table=block_table,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
            )
            return output
        use_gqa6_decode_80cu = (
            self._gqa6_decode_80cu is not None
            and max_seqlen_q == 1
            and num_actual_tokens == 1
            and query.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and query.ndim == 3
            and query.shape[1:] == (24, 256)
            and output.ndim == 3
            and output.shape == query.shape
            and key_cache.ndim == 4
            and key_cache.shape[1:] == (784, 4, 256)
            and value_cache.shape == key_cache.shape
            and cu_seqlens_q.numel() == 2
            and seqused_k.numel() == 1
            and block_table.ndim == 2
            and block_table.shape[0] == 1
        )
        attention_fn = (
            self._gqa6_decode_80cu
            if use_gqa6_decode_80cu
            else (
                self._h11_3_gqa6_prefill
                if use_h11_3_gqa6_prefill
                else self.unified_attention
            )
        )
        attention_fn(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = kv_cache.unbind(0)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
