## ADDED Requirements

### Requirement: TurboQuant scheme registration

The system SHALL register a TurboQuant KV cache quantization scheme via `@register_scheme("turboquant", "attention")` on a class implementing `AscendAttentionScheme`. The scheme SHALL be instantiated by `modelslim_config.py` when the KV cache dtype string starts with `"turboquant"`.

#### Scenario: Scheme is registered in the scheme registry
- **WHEN** `get_scheme_class("turboquant", "attention")` is called
- **THEN** the system SHALL return the `AscendTurboQuantAttentionMethod` class

#### Scenario: Scheme is selected for turboquant kv_cache_dtype
- **WHEN** `modelslim_config.get_quant_method()` is called for an `AttentionLayerBase` layer and the kv cache dtype string starts with `"turboquant_"`
- **THEN** the system SHALL return an `AscendKVCacheMethod` wrapping an `AscendTurboQuantAttentionMethod` instance

#### Scenario: Scheme is not selected for non-turboquant kv_cache_dtype
- **WHEN** `modelslim_config.get_quant_method()` is called for an `AttentionLayerBase` layer and the kv cache dtype string does not start with `"turboquant_"`
- **THEN** the system SHALL NOT instantiate `AscendTurboQuantAttentionMethod`

### Requirement: TurboQuant config and preset parsing

The system SHALL parse the `kv_cache_dtype` string to determine the TurboQuant preset configuration using vllm's `TurboQuantConfig.from_cache_dtype()`. The system SHALL support all four presets: `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and `turboquant_3bit_nc`.

#### Scenario: Preset turboquant_4bit_nc is parsed correctly
- **WHEN** the scheme is created with `kv_cache_dtype = "turboquant_4bit_nc"` and `head_dim = 128`
- **THEN** the `TurboQuantConfig` SHALL have `key_quant_bits = 4`, `value_quant_bits = 4`, `norm_correction = True`

#### Scenario: Preset turboquant_k3v4_nc is parsed correctly
- **WHEN** the scheme is created with `kv_cache_dtype = "turboquant_k3v4_nc"` and `head_dim = 128`
- **THEN** the `TurboQuantConfig` SHALL have `key_quant_bits = 3`, `value_quant_bits = 4`, `norm_correction = True`

#### Scenario: Invalid preset raises an error
- **WHEN** the scheme is created with `kv_cache_dtype = "turboquant_invalid"`
- **THEN** the system SHALL raise a `ValueError` listing valid presets

### Requirement: Weight creation and class surgery

The scheme's `create_weights` method SHALL create TurboQuant parameters on the attention layer and swap the layer's attention backend implementation to `AscendTurboQuantAttentionBackendImpl` via class surgery (`layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl`).

#### Scenario: Class surgery swaps the impl class
- **WHEN** `create_weights(layer)` is called on a TurboQuant scheme instance
- **THEN** `layer.impl.__class__` SHALL be `AscendTurboQuantAttentionBackendImpl`

#### Scenario: TQ parameters are created on the layer
- **WHEN** `create_weights(layer)` is called on a TurboQuant scheme instance with `head_dim = 128` and `kv_cache_dtype = "turboquant_4bit_nc"`
- **THEN** the layer SHALL have `_tq_PiT` (rotation matrix, shape `(128, 128)`, float32), `_tq_Pi` (inverse rotation, same as PiT for Hadamard), `_tq_centroids` (shape `(16,)`, float32), and `_tq_midpoints` (shape `(15,)`, float32) attributes

#### Scenario: kv_cache_torch_dtype is set to uint8
- **WHEN** `create_weights(layer)` is called on a TurboQuant scheme instance
- **THEN** `layer.kv_cache_torch_dtype` SHALL be `torch.uint8`

### Requirement: Hadamard rotation matrix

The system SHALL precompute an orthonormal Hadamard matrix of size `(head_dim, head_dim)` using Sylvester construction. The matrix SHALL be symmetric (`Pi = PiT`) and normalized so that `Pi @ PiT = I`.

#### Scenario: Hadamard matrix is orthonormal
- **WHEN** the Hadamard matrix `H` is constructed for `head_dim = 128`
- **THEN** `torch.allclose(H @ H.T, torch.eye(128), atol=1e-5)` SHALL be `True`

#### Scenario: Hadamard matrix is symmetric
- **WHEN** the Hadamard matrix `H` is constructed for any supported head_dim
- **THEN** `torch.allclose(H, H.T)` SHALL be `True`

### Requirement: Lloyd-Max centroid precomputation

The system SHALL precompute Lloyd-Max optimal centroids for the `N(0, 1/d)` distribution using vllm's `centroids.py` solver. Centroids SHALL be cached per `(head_dim, bits)` pair via `@lru_cache`. Midpoints (decision boundaries) SHALL be derived as the average of adjacent sorted centroids.

#### Scenario: Centroids are computed for 4-bit quantization
- **WHEN** centroids are computed for `head_dim = 128` and `bits = 4`
- **THEN** the centroids tensor SHALL have shape `(16,)` and be sorted in ascending order

#### Scenario: Midpoints are derived from centroids
- **WHEN** midpoints are derived from sorted centroids of shape `(n,)`
- **THEN** the midpoints tensor SHALL have shape `(n-1,)` where each element is the average of two adjacent centroids

### Requirement: KV cache store — quantize and pack

The system SHALL quantize key and value tensors and store them in the packed uint8 KV cache. The store path SHALL: (1) normalize keys and apply Hadamard rotation, (2) bucketize rotated keys using Lloyd-Max midpoints to obtain MSE indices, (3) pack MSE indices and vec_norm into key_packed bytes, (4) uniform-quantize values and pack into value_packed bytes, (5) scatter packed bytes to the correct cache slots using slot_mapping.

#### Scenario: MSE key quantization and packing (4-bit)
- **WHEN** a key vector of shape `(N, H, D)` is stored with `key_quant_bits = 4`
- **THEN** each key element SHALL be mapped to an index in `[0, 15]` via bucketize against 15 midpoints, and pairs of indices SHALL be packed into single uint8 bytes (two 4-bit indices per byte)

#### Scenario: MSE key quantization and packing (3-bit)
- **WHEN** a key vector of shape `(N, H, D)` is stored with `key_quant_bits = 3`
- **THEN** each key element SHALL be mapped to an index in `[0, 7]` via bucketize against 7 midpoints, and groups of 8 indices SHALL be packed into 3 uint8 bytes (24 bits per group)

#### Scenario: Key normalization and rotation
- **WHEN** a key vector `k` of shape `(N, H, D)` is stored
- **THEN** the system SHALL compute `norms = ||k||` per (token, head), `x_hat = k / norms`, and `y = x_hat @ PiT`, and the vec_norm (fp16) SHALL be stored alongside the packed MSE indices

#### Scenario: Value uniform quantization and packing (4-bit)
- **WHEN** a value vector of shape `(N, H, D)` is stored with `value_quant_bits = 4`
- **THEN** the system SHALL compute `val_min` and `val_max` per (token, head), quantize to 16 levels via `round((v - val_min) / scale)` where `scale = (val_max - val_min) / 15`, pack pairs into uint8 bytes, and store scale and zero as fp16 bytes

#### Scenario: Packed bytes are scattered to correct cache slots
- **WHEN** packed KV bytes are stored for token `i` with `slot_mapping[i] = s`
- **THEN** the packed bytes SHALL be written to `kv_cache[s // block_size, s % block_size, head_idx, :]` for each head

### Requirement: KV cache decode — dequant and attention

The system SHALL dequantize KV from the packed uint8 cache and compute attention using `torch_npu.npu_fused_infer_attention_score` with float KV. The decode path SHALL: (1) gather packed bytes from paged cache for the request's sequence, (2) unpack MSE indices and lookup centroids to reconstruct rotated keys, (3) apply norm correction if enabled, (4) apply inverse rotation and multiply by vec_norm, (5) unpack values and dequantize using stored scale/zero, (6) call FIA with float KV.

#### Scenario: Key dequantization with norm correction
- **WHEN** keys are dequantized from cache with `norm_correction = True`
- **THEN** the system SHALL compute `k_rotated = centroids[idx]`, normalize to unit norm, multiply by vec_norm, and apply inverse rotation: `k = (k_rotated / ||k_rotated|| * vec_norm) @ Pi`

#### Scenario: Key dequantization without norm correction
- **WHEN** keys are dequantized from cache with `norm_correction = False`
- **THEN** the system SHALL compute `k = (centroids[idx] * vec_norm) @ Pi` without normalization

#### Scenario: Value dequantization
- **WHEN** values are dequantized from cache
- **THEN** the system SHALL unpack quantized indices, compute `v = packed_idx * scale + zero` using the stored fp16 scale and zero, and return float values

#### Scenario: Decode attention via FIA with float KV
- **WHEN** decode attention is computed for a batch of decode requests
- **THEN** the system SHALL call `torch_npu.npu_fused_infer_attention_score` with dequantized float K and V tensors, `input_layout="TND"`, and `block_table=None` (dense KV after dequant)

### Requirement: Prefill attention path

The system SHALL compute prefill attention using raw float K/V when available (first-chunk prefill) and dequantized float K/V from cache for continuation prefill. After attention computation, all new tokens SHALL be quantized and stored to the TQ cache.

#### Scenario: First-chunk prefill uses raw float K/V
- **WHEN** a prefill request has `query_len == seq_len` (no prior cached KV)
- **THEN** the system SHALL call FIA with the raw float key and value tensors directly, without reading from the TQ cache

#### Scenario: Continuation prefill dequants cached KV
- **WHEN** a prefill request has `query_len < seq_len` (some KV already cached)
- **THEN** the system SHALL dequantize the cached KV from the TQ cache, concatenate with the current chunk's raw float K/V, and call FIA with the combined float KV

#### Scenario: Prefill stores new tokens to TQ cache
- **WHEN** prefill attention is completed for a request
- **THEN** all new tokens from the current chunk SHALL be quantized and stored to the TQ cache via the store path

### Requirement: ChunkedPrefill path

The system SHALL handle ChunkedPrefill (mixed decode + prefill batches) by splitting the batch: decode tokens use the decode path (dequant + FIA), prefill tokens use the prefill path (raw or dequanted float K/V + FIA).

#### Scenario: Mixed batch with decodes and prefills
- **WHEN** a ChunkedPrefill batch contains `num_decode_tokens` decode tokens followed by prefill tokens
- **THEN** the system SHALL process decode tokens via the decode path and prefill tokens via the prefill path, writing results to the correct positions in the output tensor

### Requirement: KV cache shape override

The system SHALL override the KV cache shape to return a 4D packed layout: `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` where `slot_size_aligned` is computed from `TurboQuantConfig.slot_size_aligned`.

#### Scenario: get_kv_cache_shape returns 4D packed shape
- **WHEN** `get_kv_cache_shape(num_blocks=100, block_size=16, num_kv_heads=8, head_size=128, cache_dtype_str="turboquant_4bit_nc")` is called
- **THEN** the returned shape SHALL be `(100, 16, 8, slot_size_aligned)` where `slot_size_aligned` = `key_packed_size + value_packed_size` rounded up to even, and the dtype SHALL be `torch.uint8`

#### Scenario: Different presets produce different slot sizes
- **WHEN** `get_kv_cache_shape` is called with `turboquant_4bit_nc` (head_dim=128) vs `turboquant_3bit_nc` (head_dim=128)
- **THEN** the `slot_size_aligned` values SHALL differ, reflecting the different key/value packed sizes for 4-bit vs 3-bit quantization

### Requirement: Bit packing and unpacking

The system SHALL implement bit packing and unpacking for 3-bit and 4-bit quantization using PyTorch tensor operations (reshape, bitwise_and, bitwise_left_shift, bitwise_right_shift). The pack and unpack operations SHALL be exact inverses.

#### Scenario: 4-bit pack then unpack is identity
- **WHEN** a tensor of random indices in `[0, 15]` with shape `(N, D)` is packed to uint8 bytes and then unpacked
- **THEN** the unpacked indices SHALL exactly match the original indices

#### Scenario: 3-bit pack then unpack is identity
- **WHEN** a tensor of random indices in `[0, 7]` with shape `(N, D)` is packed to uint8 bytes and then unpacked
- **THEN** the unpacked indices SHALL exactly match the original indices

#### Scenario: Pack produces correct byte count
- **WHEN** `D = 128` elements are packed at 4-bit
- **THEN** the packed output SHALL have `64` bytes (D/2)
- **WHEN** `D = 128` elements are packed at 3-bit
- **THEN** the packed output SHALL have `48` bytes (ceil(D*3/8))

### Requirement: apply method raises RuntimeError

The scheme's `apply` method SHALL raise `RuntimeError` when called, because TurboQuant attention is handled entirely by the attention backend impl (activated via class surgery), not by the scheme's apply method. This follows the same pattern as C8.

#### Scenario: apply raises RuntimeError
- **WHEN** `scheme.apply(layer, query, key, value, kv_cache, attn_metadata, attn_type, scale, output)` is called
- **THEN** the system SHALL raise `RuntimeError` with a message indicating TurboQuant attention is handled by the attention backend

### Requirement: Process weights after loading

The scheme's `process_weights_after_loading` method SHALL ensure TQ parameters (centroids, midpoints, rotation matrix) are on the correct device and have the correct dtype after model weights are loaded.

#### Scenario: TQ parameters are moved to device after loading
- **WHEN** `process_weights_after_loading(layer)` is called
- **THEN** `_tq_PiT`, `_tq_Pi`, `_tq_centroids`, and `_tq_midpoints` SHALL be on the same device as the layer's other parameters, with `_tq_PiT` and `_tq_Pi` as float32 and `_tq_centroids` and `_tq_midpoints` as float32

### Requirement: FP8 key mode fallback

For the `turboquant_k8v4` preset (which specifies FP8 keys in vllm), the system SHALL fall back to MSE quantization at 4-bit for keys, since Ascend 910B4 FP8 support differs from CUDA. The system SHALL log a warning when this fallback is activated.

#### Scenario: turboquant_k8v4 uses MSE fallback
- **WHEN** the scheme is created with `kv_cache_dtype = "turboquant_k8v4"` on Ascend
- **THEN** the system SHALL use `key_quant_bits = 4` (MSE) instead of FP8, and SHALL log a warning message about the FP8 fallback
