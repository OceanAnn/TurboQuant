## 1. Scheme Module and Registration

- [ ] 1.1 Create `vllm_ascend/quantization/methods/turboquant.py` with `AscendTurboQuantAttentionMethod(AscendAttentionScheme)` class, decorated with `@register_scheme("turboquant", "attention")`
- [ ] 1.2 Implement `__init__`: parse `kv_cache_dtype` string via `TurboQuantConfig.from_cache_dtype()`, store config, handle FP8 fallback (log warning, set key_quant_bits=4)
- [ ] 1.3 Implement `create_weights`: create TQ parameters (centroids via `get_centroids(head_dim, mse_bits)`, Hadamard via `_build_hadamard`, midpoints from sorted centroids), set `layer.kv_cache_torch_dtype = torch.uint8`, perform class surgery (`layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl`)
- [ ] 1.4 Implement `process_weights_after_loading`: ensure TQ parameters are on correct device with float32 dtype
- [ ] 1.5 Implement `apply`: raise `RuntimeError` (attention handled by backend impl)
- [ ] 1.6 Add dispatch branch in `modelslim_config.py` (`get_quant_method`) for `kv_cache_type == "turboquant"` or `kv_cache_dtype` starts with `"turboquant_"`, returning `AscendKVCacheMethod(AscendTurboQuantAttentionMethod(...))`
- [ ] 1.7 Add `_build_hadamard` function (Sylvester construction, cached per (d, device)) to the module or import from vllm if accessible

## 2. Bit Packing and Unpacking Utilities

- [ ] 2.1 Implement `pack_4bit(indices: torch.Tensor) -> torch.Tensor`: reshape to pairs, pack two 4-bit indices per uint8 byte via bitwise ops
- [ ] 2.2 Implement `unpack_4bit(packed: torch.Tensor) -> torch.Tensor`: extract low and high nibbles, interleave to original shape
- [ ] 2.3 Implement `pack_3bit(indices: torch.Tensor) -> torch.Tensor`: reshape to groups of 8, pack into 3 uint8 bytes per group via bitwise ops
- [ ] 2.4 Implement `unpack_3bit(packed: torch.Tensor) -> torch.Tensor`: extract 3-bit fields from 3 bytes per group of 8
- [ ] 2.5 Write unit tests verifying pack→unpack is identity for both 3-bit and 4-bit with random indices
- [ ] 2.6 Write unit tests verifying correct byte count output (D/2 for 4-bit, ceil(D*3/8) for 3-bit)

## 3. KV Cache Store Path

- [ ] 3.1 Implement `turboquant_store(key, value, kv_cache, slot_mapping, layer, tq_config)`: main entry point for quantize+pack+scatter
- [ ] 3.2 Implement key normalization and rotation: `norms = k_flat.norm(dim=1, keepdim=True)`, `x_hat = k_flat / (norms + 1e-8)`, `y = x_hat @ PiT`
- [ ] 3.3 Implement key bucketize: `idx = torch.bucketize(y, midpoints)` to map rotated coordinates to MSE indices
- [ ] 3.4 Implement key packing: pack MSE indices via `pack_4bit` or `pack_3bit` based on `key_mse_bits`, store vec_norm as fp16 bytes
- [ ] 3.5 Implement value uniform quantization: compute per-(token,head) min/max, quantize to `2^value_quant_bits - 1` levels, pack via `pack_4bit` or `pack_3bit`, store scale and zero as fp16 bytes
- [ ] 3.6 Implement scatter to cache: compute slot byte offsets from `slot_mapping`, write packed key+value bytes to `kv_cache` at correct positions
- [ ] 3.7 Write unit tests verifying store correctness: store known K/V, manually verify packed bytes match expected quantization output

## 4. KV Cache Decode Path

- [ ] 4.1 Implement `turboquant_decode_dequant(kv_cache, block_table, seq_lens, layer, tq_config)`: gather packed bytes from paged cache and dequantize to float K/V
- [ ] 4.2 Implement key dequant: unpack MSE indices, lookup centroids via `centroids[idx]`, apply norm correction if enabled (normalize to unit norm), multiply by vec_norm, apply inverse rotation `@ Pi`
- [ ] 4.3 Implement value dequant: unpack quantized indices, multiply by scale and add zero using stored fp16 scale/zero
- [ ] 4.4 Implement paged cache gathering: use block_table and seq_lens to gather correct slots, handle variable-length sequences
- [ ] 4.5 Write unit tests verifying decode dequant correctness: store K/V, dequant, compare with original within tolerance

## 5. Attention Backend Implementation

- [ ] 5.1 Create `AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl)` class in `attention_v1.py` (or new file `attention_turboquant_v1.py`)
- [ ] 5.2 Implement `forward()`: dispatch to store path (quantize+scatter), then prefill or decode attention based on `attn_metadata.attn_state`
- [ ] 5.3 Implement decode forward: call `turboquant_decode_dequant` to get float K/V, then call `torch_npu.npu_fused_infer_attention_score` with float KV in TND layout
- [ ] 5.4 Implement first-chunk prefill forward: call FIA with raw float K/V directly (no cache read)
- [ ] 5.5 Implement continuation prefill forward: dequant cached KV, concat with new chunk float K/V, call FIA with combined float KV
- [ ] 5.6 Implement ChunkedPrefill forward: split decode/prefill tokens, process each via respective path, write to output tensor
- [ ] 5.7 Implement `get_kv_cache_shape` override: return `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` as uint8
- [ ] 5.8 Handle `reshape_and_cache` integration: ensure store path is called correctly within the forward flow (determine whether to use `do_kv_cache_update` or inline store)

## 6. Integration and Configuration

- [ ] 6.1 Wire `get_kv_cache_shape` override into the attention layer so the model runner allocates the correct 4D uint8 cache
- [ ] 6.2 Verify `KVCacheSpec` propagation: ensure the custom cache shape and dtype reach the model runner's cache allocation
- [ ] 6.3 Add TQ scheme import to `vllm_ascend/quantization/methods/__init__.py` if needed for registration side effects
- [ ] 6.4 Resolve open question: determine whether vllm-ascend reads `cache_dtype` from `CacheConfig.cache_dtype` or from `quant_description`, and wire dispatch accordingly
- [ ] 6.5 Resolve open question: determine whether TQ should use `do_kv_cache_update` (separate store op) or inline store in `forward`, matching vllm-ascend's model runner expectations

## 7. Testing

- [ ] 7.1 Write unit test for scheme registration: verify `get_scheme_class("turboquant", "attention")` returns the correct class
- [ ] 7.2 Write unit test for `create_weights`: verify class surgery, parameter creation, and `kv_cache_torch_dtype` setting
- [ ] 7.3 Write unit test for Hadamard matrix: verify orthonormality and symmetry
- [ ] 7.4 Write unit test for end-to-end store+dequant: store K/V, dequant, verify reconstruction within tolerance for each preset
- [ ] 7.5 Write unit test for norm correction: verify dequant with `norm_correction=True` produces different (better) results than without
- [ ] 7.6 Write unit test for `get_kv_cache_shape`: verify 4D shape and correct slot_size_aligned for each preset
- [ ] 7.7 Write integration test: run a small model with `kv_cache_dtype="turboquant_4bit_nc"` on Ascend NPU, verify output is generated without errors
- [ ] 7.8 Write accuracy comparison test: compare attention output between TQ-quantized and unquantized KV cache, verify acceptable degradation

## 8. Documentation and Polish

- [ ] 8.1 Add docstrings to all public functions and classes following vllm-ascend conventions
- [ ] 8.2 Add log messages for TQ initialization (preset, key/value bits, norm correction, FP8 fallback) using `logger.info_once`
- [ ] 8.3 Verify code follows vllm-ascend style guidelines (naming, imports, no magic numbers, no `tensor.item()` in hot paths)
- [ ] 8.4 Run `ruff check` and `ruff format` on all new files
