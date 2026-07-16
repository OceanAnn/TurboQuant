#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""TurboQuant attention backend impl for Ascend NPU (v1).

Faithful implementation of Algorithm 1 (TurboQuant_mse) from the paper:
  - 4-bit Lloyd-Max optimal scalar quantization after random rotation
  - Random rotation matrix (QR decomposition of Gaussian matrix)
  - Both K and V quantized with TurboQuant_mse
  - Decode: dequant to float16, then FIA on dense K/V
  - Prefill: FIA on raw float K/V, then quantize+store

Per Theorem 1, at b=4 the MSE distortion is ≈ 0.009 (negligible).
Per Figure 1, at b=4 TurboQuant_mse has near-zero inner product bias.

Subclasses AscendAttentionBackendImpl to handle TQ's combined
single-tensor paged uint8 cache.
"""

from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import (
    SWA_INT_MAX,
    AscendAttentionBackendImpl,
    AscendAttentionState,
)
from vllm_ascend.quantization.methods.turboquant_ops import (
    build_random_rotation,
    compute_midpoints,
    dequant_paged_kv,
    store_turboquant_kv,
)


class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention impl for TurboQuant compressed KV cache on Ascend.

    The KV cache is a single 4D uint8 tensor:
        (num_blocks, block_size, num_kv_heads, slot_size)

    Each slot packs TurboQuant_mse quantized K and V:
      K: [4-bit MSE indices | ||k|| fp16]
      V: [4-bit MSE indices | ||v|| fp16]
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
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
            **kwargs,
        )
        self._tq_initialized = True

    def _ensure_tq_init(self):
        """Lazy init for TQ-specific fields when class is swapped via __class__.

        When _apply_turboquant_class_surgery changes __class__ without
        calling __init__, this method initializes the missing fields.
        """
        if getattr(self, "_tq_initialized", False):
            return
        self._tq_initialized = True

    def _ensure_on_device(self, layer: Any, device: torch.device) -> None:
        """One-time derivation of TQ buffers.

        Creates the random rotation matrix Π (via QR decomposition),
        shared across all layers. Also computes midpoints from centroids.

        Paper Algorithm 1 line 2:
          - Generate a random rotation matrix Π
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Random rotation matrix (paper: QR decomposition of Gaussian matrix)
            layer._tq_Pi = build_random_rotation(D, device)

            # Compute midpoints from centroids
            c = layer._tq_centroids.to(device=device, dtype=torch.float32)
            layer._tq_midpoints = compute_midpoints(c)
            layer._tq_cached = True

    @staticmethod
    def _get_tq_cache(kv_cache) -> torch.Tensor | None:
        """Extract the single TQ cache tensor from kv_cache argument."""
        if kv_cache is None:
            return None
        if isinstance(kv_cache, (list, tuple)):
            return kv_cache[0]
        return kv_cache

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TurboQuant KV cache.

        Args:
            query: (num_tokens, num_heads, head_size)
            key: (num_tokens, num_kv_heads, head_size)
            value: (num_tokens, num_kv_heads, head_size)
            kv_cache: single (num_blocks, block_size, Hk, slot_size) uint8
        """
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output.fill_(0)

        self._ensure_tq_init()

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        device = query.device
        self._ensure_on_device(layer, device)

        tq_cache = self._get_tq_cache(kv_cache)

        # Store K/V into TQ cache (Algorithm 1: Quant_mse)
        if key is not None and value is not None and tq_cache is not None:
            store_turboquant_kv(
                key=key[:N].view(N, self.num_kv_heads, self.head_size),
                value=value[:N].view(N, self.num_kv_heads, self.head_size),
                kv_cache=tq_cache,
                slot_mapping=attn_metadata.slot_mapping[:N],
                Pi=layer._tq_Pi,
                centroids=layer._tq_centroids,
                midpoints=layer._tq_midpoints,
            )

        # Pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            return self._forward_encoder_attention(
                query[:N], key[:N], value[:N], attn_metadata, output
            )

        # Attention dispatch
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            attn_output = self._forward_tq_prefill(
                query[:N], key[:N], value[:N], attn_metadata, output
            )
        else:
            attn_output = self._forward_tq_dequant_attn(
                query[:N], tq_cache, attn_metadata, output, layer
            )

        output[:N] = attn_output[:N]
        return output

    def _forward_tq_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """First-chunk prefill: FIA on raw float K/V (TND layout)."""
        num_tokens = query.shape[0]

        if self.sliding_window is not None:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                input_layout="TND",
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                pre_tokens=self.sliding_window,
                next_tokens=0,
                sparse_mode=4,
            )
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                input_layout="TND",
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )

        if output.ndim == 3:
            return attn_output.view(num_tokens, self.num_heads, self.head_size)
        return attn_output.view(num_tokens, self.num_heads * self.head_size)

    def _forward_tq_dequant_attn(
        self,
        query: torch.Tensor,
        tq_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        """Decode / chunked-prefill: dequant TQ cache, FIA on dense K/V.

        Dequants the full paged TQ cache to dense float16 K/V buffers
        (Algorithm 2: DeQuant_prod), then runs FIA with BNSD layout.
        """
        B = attn_metadata.seq_lens.shape[0]
        Hq = self.num_heads
        Hk = self.num_kv_heads
        D = self.head_size
        num_tokens = query.shape[0]

        # Dequant paged TQ cache to dense float16 K/V
        k_dense, v_dense, alloc_len = dequant_paged_kv(
            kv_cache=tq_cache,
            block_table=attn_metadata.block_tables,
            seq_lens=attn_metadata.seq_lens,
            Pi=layer._tq_Pi,
            centroids=layer._tq_centroids,
            k_buf=getattr(layer, "_tq_k_dequant_buf", None),
            v_buf=getattr(layer, "_tq_v_dequant_buf", None),
        )
        layer._tq_k_dequant_buf = k_dense
        layer._tq_v_dequant_buf = v_dense

        # Build query in BNSD layout
        qsl = attn_metadata.query_start_loc  # (B + 1,) tensor
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            q_bnsd = query[:B].view(B, Hq, 1, D).to(torch.float16)
            actual_seq_qlen = [1] * B
        else:
            q_lens = (qsl[1:] - qsl[:-1]).tolist()
            max_q_len = max(q_lens) if q_lens else 1
            q_bnsd = torch.zeros(
                B, Hq, max_q_len, D, dtype=torch.float16, device=query.device
            )
            for i, ql in enumerate(q_lens):
                q_start = int(qsl[i].item())
                q_bnsd[i, :, :ql, :] = query[q_start : q_start + ql]
            actual_seq_qlen = q_lens

        # FIA call with BNSD layout, dense float K/V
        if self.sliding_window is not None:
            sparse_mode = 4
            pre_tokens = self.sliding_window
        else:
            sparse_mode = 3
            pre_tokens = SWA_INT_MAX

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q_bnsd,
            key=k_dense,
            value=v_dense,
            input_layout="BNSD",
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_key_value_heads=Hk,
            num_heads=Hq,
            scale=self.scale,
            pre_tokens=pre_tokens,
            next_tokens=0,
            sparse_mode=sparse_mode,
        )

        # Convert BNSD output back to flat (num_tokens, ...)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            flat_output = attn_output.view(B, Hq * D)
        else:
            q_lens = (qsl[1:] - qsl[:-1]).tolist()
            flat_output = torch.empty(
                num_tokens,
                Hq * D,
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            idx = 0
            for i, ql in enumerate(q_lens):
                flat_output[idx : idx + ql] = attn_output[i, :, :ql, :].reshape(
                    ql, Hq * D
                )
                idx += ql

        if output.ndim == 3:
            return flat_output.view(num_tokens, Hq, D)
        return flat_output
