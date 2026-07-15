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
"""TurboQuant KV-cache quantization scheme for Ascend NPU.

Faithful implementation of the TurboQuant paper (Zandieh et al.):
  - Algorithm 2 (TurboQuant_prod): 3-bit MSE + 1-bit QJL = INT4
  - Random rotation matrix (QR decomposition) — not Hadamard
  - Both K and V quantized with the same TurboQuant algorithm
  - QJL residual quantization for unbiased inner product estimation

This scheme performs class surgery on the attention impl and patches
get_kv_cache_spec to use the paper-faithful slot size.

The TQ centroids (3-bit Lloyd-Max) are already registered by vllm's
``_init_turboquant_buffers`` when using ``turboquant_3bit_nc`` as
``kv_cache_dtype``.
"""

import torch

from vllm_ascend.quantization.methods.base import AscendAttentionScheme
from vllm_ascend.quantization.methods.registry import register_scheme
from vllm_ascend.quantization.methods.turboquant_ops import compute_paper_slot_size


@register_scheme("turboquant", "attention")
class AscendTurboQuantAttentionMethod(AscendAttentionScheme):
    """TurboQuant KV-cache quantization scheme (paper-faithful INT4).

    Registered as quant_type="turboquant", layer_type="attention".
    Triggered when the attention layer has ``_tq_config`` (set by vllm's
    ``_init_turboquant_buffers`` when ``kv_cache_dtype`` starts with
    "turboquant_").
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.uint8

        # Ensure 4-bit centroids (for TurboQuant_mse INT4)
        head_dim = layer.head_size
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )

        existing_centroids = getattr(layer, "_tq_centroids", None)
        if existing_centroids is None or existing_centroids.shape[0] != 16:
            layer._tq_centroids = get_centroids(head_dim, 4)

        # Patch get_kv_cache_spec to return the paper-faithful slot size.
        # The slot size from vllm's TurboQuantConfig does not match our
        # layout (3-bit MSE + 1-bit QJL + 2 fp16 norms, for both K and V).
        paper_slot_size = compute_paper_slot_size(head_dim)
        num_kv_heads = layer.num_kv_heads
        kv_cache_torch_dtype = layer.kv_cache_torch_dtype
        head_size = layer.head_size

        def patched_get_kv_cache_spec(vllm_config):
            from vllm.v1.kv_cache_interface import TQFullAttentionSpec

            block_size = vllm_config.cache_config.block_size
            return TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                head_size_v=head_size,
                dtype=kv_cache_torch_dtype,
                tq_slot_size=paper_slot_size,
            )

        layer.get_kv_cache_spec = patched_get_kv_cache_spec

        # Class surgery: upgrade impl to TQ-specific subclass
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_turboquant_v1 import (
                AscendTurboQuantAttentionBackendImpl,
            )

            layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "[vllm-ascend/TQ] AscendTurboQuantAttentionMethod.apply should "
            "not be called. TurboQuant KV cache quantization is handled by "
            "the attention backend."
        )
