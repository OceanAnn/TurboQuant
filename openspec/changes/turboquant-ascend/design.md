## Context

vllm ships a TurboQuant KV cache quantization implementation that relies on Triton fused kernels (CUDA-only). vllm-ascend targets the Huawei Ascend 910B4 NPU, which has no Triton support. The existing vllm-ascend KV cache quantization scheme (C8) uses static per-channel INT8 with `torch_npu.npu_fused_infer_attention_score` (FIA) for decode. TurboQuant requires a fundamentally different approach: Hadamard rotation + Lloyd-Max scalar quantization with packed-bit storage, which FIA cannot consume natively.

Key reference implementations:
- **vllm TurboQuant** (`vllm/v1/attention/backends/turboquant_attn.py`, `triton_turboquant_store.py`, `triton_turboquant_decode.py`): Full Triton-based implementation with fused store + decode kernels. Cache layout is packed uint8 bytes per head per position.
- **vllm-ascend C8** (`vllm_ascend/quantization/methods/kv_c8.py`, `vllm_ascend/attention/attention_v1.py:AscendC8AttentionBackendImpl`): INT8 KV cache with class-surgery activation and FIA-based decode using native paged INT8 antiquant.
- **vllm centroids** (`vllm/model_executor/layers/quantization/turboquant/centroids.py`): Pure-Python Lloyd-Max solver with `@lru_cache`, no CUDA dependency. Reusable as-is.

Constraints:
- Ascend 910B4 has no Triton; all kernels must be PyTorch ops or `torch_npu` ops.
- `npu_fused_infer_attention_score` accepts INT8 paged KV with per-channel antiquant scales/offsets, or float KV. It does NOT accept TurboQuant's packed-bit format.
- The Hadamard rotation matrix and Lloyd-Max centroids are data-oblivious (precomputed once, shared across all layers).
- vllm's TurboQuant uses the same packed byte cache layout for both FP8 and MSE key modes.

## Goals / Non-Goals

**Goals:**
- Implement TurboQuant KV cache quantization on Ascend 910B4 using only PyTorch and `torch_npu` ops
- Support all four vllm TQ presets: `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`
- Integrate via vllm-ascend's existing scheme registration and class-surgery mechanisms
- Achieve functional correctness: quantize → store → dequant → attention produces results matching vllm's TurboQuant within numerical tolerance
- Reuse vllm's `centroids.py` Lloyd-Max solver and `TurboQuantConfig` unchanged

**Non-Goals:**
- Implementing the paper's Algorithm 2 (QJL two-stage with residual quantization) — vllm's TurboQuant also omits this; community consensus is QJL hurts attention quality
- Matching vllm's Triton kernel performance — the PyTorch-op approach will have more kernel launches; performance optimization is deferred
- FP8 key storage (`turboquant_k8v4` FP8 path) — Ascend 910B4 FP8 support differs from CUDA; FP8 key mode will initially fall back to MSE quantization at the same bit-width, with true FP8 support as a future enhancement
- Modifying upstream vllm code — all changes are in vllm-ascend

## Decisions

### D1: Cache layout — reuse vllm's packed uint8 byte format

**Decision:** Use the same packed byte cache layout as vllm TurboQuant: `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` where each slot is `[key_packed | value_packed]` stored as uint8.

**Rationale:** This ensures bit-exact compatibility with vllm's cache format, enabling potential cache sharing/migration between CUDA and Ascend. The `TurboQuantConfig` class already computes `slot_size`, `slot_size_aligned`, `key_packed_size`, `value_packed_size` — reusing these avoids duplicating layout logic.

**Alternatives considered:**
- Storing dequantized FP16 K/V in cache (like C8 stores INT8): Would allow direct FIA consumption but defeats the purpose of quantization (no memory savings). Rejected.
- Using a separate INT8 cache with rotation applied at store time and dequant at decode time: Would lose the sub-byte packing benefit (3-bit/4-bit). Rejected.

### D2: Store path — PyTorch ops for rotation, bucketize, pack, scatter

**Decision:** Implement the store path as a sequence of PyTorch/torch_npu operations:

1. **Rotation:** `k_flat = key.float().reshape(NH, D)` → `norms = k_flat.norm(dim=1, keepdim=True)` → `x_hat = k_flat / (norms + 1e-8)` → `y = x_hat @ PiT` (matmul). Same as vllm's external GEMM approach.
2. **Bucketize:** `idx = torch.bucketize(y, midpoints)` — replaces Triton binary search. `torch.bucketize` is a standard PyTorch op available on NPU.
3. **Pack:** For 4-bit: reshape to pairs, pack two 4-bit indices per byte via bitwise ops. For 3-bit: reshape to groups of 8, pack three bytes per group. Implemented via `torch.bitwise_left_shift`, `torch.bitwise_or`, `torch.sum` on reshaped tensors.
4. **Value quantization:** Uniform min-max quantization to `value_quant_bits` levels, pack same as keys, store scale/zero as fp16 bytes.
5. **Scatter:** Write packed bytes to kv_cache via `kv_cache.view(-1)[slot_offsets] = packed_bytes` using computed slot offsets from `slot_mapping`.

**Rationale:** `torch.bucketize` is the natural PyTorch replacement for Triton's binary search loop. Bit packing via tensor reshape + bitwise ops is vectorized and avoids per-element Python loops. The matmul for rotation leverages NPU's GEMM hardware.

**Alternatives considered:**
- Custom NPU C++ kernel for fused store: Maximum performance but requires CANN development,大幅 increases complexity. Deferred to future optimization.
- Using `torch.searchsorted` instead of `torch.bucketize`: Functionally equivalent; `bucketize` is more idiomatic for this use case.

### D3: Decode path — dequant to float, then FIA with float KV

**Decision:** For decode, gather packed KV from paged cache, dequantize to float (BF16/FP16), then call `torch_npu.npu_fused_infer_attention_score` with float KV in TND layout (no block_table, no antiquant).

Dequant steps per cached token:
1. **Key dequant:** Unpack MSE indices from packed bytes → lookup centroids via `centroids[idx]` → multiply by vec_norm → inverse rotation via `k = (centroids[idx] * vec_norm) @ Pi` (matmul). If `norm_correction`, normalize centroid vectors before inverse rotation.
2. **Value dequant:** Unpack quantized values from packed bytes → `v = packed * scale + zero` (scale/zero stored as fp16 in cache).

**Rationale:** FIA's INT8 antiquant path cannot handle TurboQuant's sub-byte packed format with rotation. Dequanting to float and using FIA's float KV path is the simplest correct approach. The dequant cost is O(seq_len * head_dim) per request, which is acceptable for decode (batch_size=1, seq_len grows slowly).

**Alternatives considered:**
- Custom attention kernel (no FIA): Would avoid dequant-to-float overhead but requires implementing softmax + scaled dot-product attention in PyTorch ops, likely slower than FIA's optimized kernel. Rejected.
- Storing INT8 rotated keys (no sub-byte packing): Would allow FIA INT8 antiquant path but loses 3-bit/4-bit compression benefit. Rejected.
- Full dequant + `torch.nn.functional.scaled_dot_product_attention`: Available but FIA is NPU-optimized and handles paged cache gathering. Preferred to use FIA.

### D4: Prefill path — FIA on raw float K/V, then quantize+store

**Decision:** Follow vllm's prefill strategy:
1. For first-chunk prefill (no prior cached KV): run FIA with raw float K/V directly.
2. For continuation prefill (some KV already cached): dequant cached KV to float, concat with new chunk's float K/V, run FIA on combined float KV.
3. After attention computation, quantize+store all new tokens to TQ cache.

**Rationale:** Prefill has large query lengths, so attention quality matters. Using raw float K/V for the current chunk avoids quantization error during the compute-heavy prefill phase. The store happens after attention, so the cache is ready for future decode/continuation.

**Alternatives considered:**
- Always dequant from cache for prefill (like C8 does): Adds unnecessary dequant overhead for first-chunk prefills where all K/V are available as float. Rejected.
- Use TQ decode kernel for continuation prefill (like vllm does for small continuations): The Triton decode kernel is fused and efficient; the PyTorch-op equivalent would be slower. Use full dequant + FIA for all continuations instead. Simpler and likely faster on NPU.

### D5: Attention backend — new AscendTurboQuantAttentionBackendImpl via class surgery

**Decision:** Create `AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl)` in `attention_v1.py` (or a new file). The scheme's `create_weights` performs class surgery: `layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl`, exactly following the C8 pattern.

The new impl overrides `forward()` to implement:
- Store path (quantize + scatter to cache)
- Prefill path (FIA on float KV)
- Decode path (dequant from cache + FIA on float KV)
- ChunkedPrefill path (split decode/prefill, handle both)

**Rationale:** This is the proven integration pattern in vllm-ascend. C8 uses the exact same mechanism. The base `AscendAttentionBackendImpl` provides shared infrastructure (reshape_and_cache, FIA helpers, metadata handling).

**Alternatives considered:**
- Separate attention backend class (not subclass of AscendAttentionBackendImpl): Would duplicate shared logic (reshape_and_cache, scale preparation, metadata handling). Rejected.
- Not using class surgery, instead using scheme.apply(): C8's apply() raises RuntimeError because the backend handles everything. Follow the same pattern — the backend impl is the single point of control.

### D6: Scheme registration — @register_scheme + modelslim_config integration

**Decision:** Register the scheme via `@register_scheme("turboquant", "attention")` on a new `AscendTurboQuantAttentionMethod(AscendAttentionScheme)` class in a new file `vllm_ascend/quantization/methods/turboquant.py`. Add a branch in `modelslim_config.py` that instantiates this scheme when `kv_cache_type == "turboquant"` (or when the cache_dtype starts with `"turboquant_"`).

The scheme's `create_weights` will:
1. Parse the TQ preset from `kv_cache_dtype` string
2. Create TQ parameters: centroids (precomputed), rotation matrix (Hadamard), midpoints (derived from centroids)
3. Set `layer.kv_cache_torch_dtype = torch.uint8` (packed byte cache)
4. Perform class surgery to swap impl to `AscendTurboQuantAttentionBackendImpl`
5. Override `get_kv_cache_shape` to return TQ's 4D shape (no leading 2)

**Rationale:** This follows the exact pattern of C8 registration. The `modelslim_config.py` is the central dispatch point for quantization methods in vllm-ascend. Using `@register_scheme` also makes the scheme available via `get_scheme_class()` for programmatic access.

**Alternatives considered:**
- Extending C8's scheme with TQ support: C8 and TQ have fundamentally different cache layouts, quantization logic, and decode paths. Mixing them would violate separation of concerns. Rejected.
- Creating a new config class (like vllm's `TurboQuantConfig`): vllm already has this class; we reuse it directly via `TurboQuantConfig.from_cache_dtype()`.

### D7: Hadamard rotation — reuse vllm's Sylvester construction

**Decision:** Reuse vllm's `_build_hadamard` function (Sylvester construction, cached per (d, device)). The function is pure PyTorch — `torch.cat` and `torch.tensor` only, no CUDA-specific ops. Copy it into the new module or import from vllm if accessible.

**Rationale:** The Hadamard matrix is orthonormal and symmetric (H = H^T), so the same matrix serves as both `Pi` (inverse rotation) and `PiT` (forward rotation). This is identical to vllm's approach. The construction is O(D^2 * log D) in Python but runs once and is cached.

**Alternatives considered:**
- Random orthogonal matrix (as the paper suggests): The paper uses random orthogonal rotation to induce Beta distribution. vllm found that Hadamard (a specific orthogonal matrix) works equally well because Lloyd-Max quantization is symmetric. Follow vllm's choice for compatibility.
- NPU-accelerated Hadamard construction: The construction runs once at model load; Python/PyTorch is fast enough. Not worth the complexity.

### D8: Bit packing/unpacking — PyTorch tensor reshape + bitwise ops

**Decision:** Implement pack/unpack for 3-bit and 4-bit using tensor reshape and bitwise operations:

**4-bit pack:** `idx` shape `(NH, D)` → reshape `(NH, D//2, 2)` → `packed = (idx[..., 0] & 0xF) | ((idx[..., 1] & 0xF) << 4)` → cast to uint8.

**4-bit unpack:** `packed` uint8 shape `(NH, D//2)` → `low = packed & 0xF` → `high = (packed >> 4) & 0xF` → stack/interleave to `(NH, D)`.

**3-bit pack:** `idx` shape `(NH, D)` → reshape `(NH, D//8, 8)` → compute `packed_24 = sum(idx_group << shifts)` where `shifts = [0, 3, 6, 9, 12, 15, 18, 21]` → split into 3 bytes per group.

**3-bit unpack:** Reverse: extract 3-bit fields from 3 bytes per group of 8 elements.

**Rationale:** These are standard bit manipulation operations expressible as PyTorch tensor ops. They work on any device (CPU, CUDA, NPU) since they only use `torch.bitwise_and`, `torch.bitwise_left_shift`, `torch.bitwise_right_shift`, `torch.sum`, and `torch.reshape`.

**Alternatives considered:**
- Using `torch_npu` custom ops for bit packing: No known NPU-specific bit packing op exists. Standard PyTorch ops are sufficient.
- Storing indices as int8 (one byte per index, no packing): Would simplify code but loses 50-75% memory savings (the entire point of 3-bit/4-bit quantization). Rejected.

### D9: KV cache shape override — 4D packed layout

**Decision:** Override `get_kv_cache_shape` to return `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` — a 4D uint8 tensor with no leading "2" dimension (unlike C8's `(2, num_blocks, block_size, num_kv_heads, head_dim)`).

The `slot_size_aligned` is computed from `TurboQuantConfig.slot_size_aligned`, which accounts for `key_packed_size + value_packed_size` rounded up to even.

**Rationale:** TurboQuant packs K+V into a single interleaved slot per head per position. This is the same layout vllm uses. The even-alignment ensures `effective_head_size = slot_size_aligned // 2` is integral, which vllm's cache allocation logic expects.

### D10: Norm correction — optional centroid renormalization

**Decision:** When `norm_correction=True` (presets `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`), after dequantizing keys (centroids lookup + vec_norm multiply), normalize each key vector to unit norm before inverse rotation: `k_hat = k_dequant / (k_dequant.norm(dim=-1, keepdim=True) + 1e-8)`, then `k = k_hat * vec_norm @ Pi`.

**Rationale:** Quantization distorts the norm of the rotated vector. Renormalizing before inverse rotation corrects this distortion, improving PPL by ~0.8% at 4-bit (per vllm's measurements). This matches vllm's implementation exactly.

## Risks / Trade-offs

**[Performance: multiple kernel launches vs fused Triton]** → The PyTorch-op store path requires ~8-10 separate kernel launches (norm, divide, matmul, bucketize, pack, scatter) vs vllm's single fused Triton kernel. Mitigation: batch all tokens/heads in single ops; profile and optimize hot spots. The decode path's dequant is the primary bottleneck — consider caching dequanted KV for repeated decode steps.

**[Performance: decode dequant overhead]** → Each decode step dequants the entire KV cache for the request. For long sequences (32k+ tokens), this is O(seq_len * head_dim * num_kv_heads) per layer. Mitigation: use FIA's paged KV access to avoid dequanting non-attended blocks; consider chunked dequant. This is the fundamental cost of not having a fused decode kernel.

**[Correctness: torch.bucketize boundary behavior]** → `torch.bucketize` with `right=False` (default) returns indices such that `input[i]` belongs in `boundaries[idx-1:idx]`. Must verify this matches vllm's Triton binary search semantics (which uses `>=` comparison). Mitigation: unit test against known centroid/boundary values; compare quantized indices with vllm's output for identical inputs.

**[Compatibility: FP8 key mode (turboquant_k8v4)]** → Ascend 910B4's FP8 support differs from CUDA (different FP8 formats). The initial implementation will use MSE quantization at 4-bit for the k8v4 preset instead of true FP8. Mitigation: document this deviation; add true FP8 support as a follow-up when Ascend FP8 ops are validated.

**[Memory: uint8 cache allocation]** → The packed uint8 cache has a different shape than standard KV caches. vllm's cache allocation infrastructure must handle this via `get_kv_cache_shape` override. Mitigation: verify that `KVCacheSpec` and model runner correctly propagate the custom shape; test with multiple models.

**[Integration: modelslim_config dispatch]** → The TQ scheme must be activated when `kv_cache_dtype` starts with `"turboquant_"`, but `modelslim_config.py` currently dispatches based on `quant_description.get("kv_cache_type")`. Need to bridge between vllm's `cache_dtype` string and vllm-ascend's `quant_description` dict. Mitigation: add a branch that checks for `"turboquant"` prefix in `kv_cache_type`; fall back to checking `cache_config.cache_dtype` if `quant_description` doesn't contain TQ info.

## Open Questions

1. **How does vllm-ascend's model runner discover the custom `get_kv_cache_shape`?** The C8 path uses standard `(2, ...)` shape. TurboQuant needs a 4D shape. Need to trace how `KVCacheSpec` propagates the shape from the attention backend to the model runner's cache allocation. Does `Attention.get_kv_cache_spec()` need modification, or does the class-surgery impl handle it?

2. **Should TQ use `do_kv_cache_update` (separate store op) or inline store in `forward`?** vllm's TurboQuant uses `do_kv_cache_update` called before `forward`. vllm-ascend's C8 stores inline in `forward` (via `reshape_and_cache`). Need to determine which pattern vllm-ascend's model runner expects.

3. **What `kv_cache_dtype` string triggers TQ in vllm-ascend?** In vllm, `--kv-cache-dtype turboquant_4bit_nc` selects TQ. In vllm-ascend, `modelslim_config.py` checks `quant_description.get("kv_cache_type")`. Need to determine whether vllm-ascend reads the `cache_dtype` from `CacheConfig` or from its own `quant_description`, and wire the dispatch accordingly.
