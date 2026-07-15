## Why

vllm-ascend (targeting Huawei Ascend 910B4 NPU) currently lacks a TurboQuant KV cache quantization scheme. vllm ships a TurboQuant implementation, but it relies on Triton kernels (CUDA-only) and uses a simplified algorithm — uniform quantization for values instead of Lloyd-Max, and simple L2 renormalization instead of the paper's two-stage QJL approach. The Ascend NPU has no Triton support, so a native implementation using torch_npu ops is needed to bring paper-faithful TurboQuant quantization to the Ascend platform.

## What Changes

- Add a new `turboquant` KV cache quantization scheme to vllm-ascend, registered via `@register_scheme("turboquant", "attention")`
- Implement the paper's Algorithm 1: random orthogonal Hadamard rotation + per-coordinate Lloyd-Max scalar quantization for both keys and values (vllm's version only uses Lloyd-Max for keys; values use uniform quantization)
- Replace Triton fused kernels with torch_npu / PyTorch ops suitable for Ascend 910B4 (Hadamard rotation via torch ops, Lloyd-Max bucketization via tensor indexing, INT8 packing/unpacking via torch ops)
- Integrate with vllm-ascend's existing class-surgery attention backend mechanism (`AscendC8AttentionBackendImpl` pattern) to swap attention impl at runtime
- Support configurable bit-widths matching vllm's TQ presets: 8-bit/4-bit/3-bit for keys, 4-bit for values, with optional norm correction
- Precompute and cache Lloyd-Max centroids and boundaries for supported bit-widths (reuse vllm's `centroids.py` solver)

## Capabilities

### New Capabilities
- `turboquant-kv-cache`: TurboQuant KV cache quantization scheme for Ascend NPU — covers scheme registration, orthogonal rotation, Lloyd-Max quantization/dequantization of K and V, attention backend integration via class surgery, and configurable bit-width presets

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- **New code**: New quantization method module in `vllm_ascend/quantization/methods/` (e.g., `turboquant.py`) implementing the scheme and parameter creation
- **Attention backend**: New or extended attention backend impl class (following `AscendC8AttentionBackendImpl` pattern) to handle TurboQuant store/decode paths without Triton
- **Existing code**: May extend `vllm_ascend/quantization/methods/kv_c8.py` registration infrastructure or create a parallel registration path
- **Dependencies**: Relies on vllm's `turboquant/centroids.py` for Lloyd-Max centroid precomputation; uses `torch_npu` ops for NPU execution
- **Config**: Maps vllm's `turboquant_*` cache dtypes (e.g., `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`) to the new Ascend scheme
