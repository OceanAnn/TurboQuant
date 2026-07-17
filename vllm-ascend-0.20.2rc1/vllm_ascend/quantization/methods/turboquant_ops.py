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
"""TurboQuant KV-cache quantization ops for Ascend NPU.

Faithful implementation of Algorithm 1 (TurboQuant_mse) from the paper
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al.).

For INT4 (b=4 bits per coordinate):
  1. Normalize input vector to unit sphere, store ||x||
  2. Apply random rotation: y = Π · x̂  (QR decomposition of Gaussian matrix)
  3. Lloyd-Max optimal scalar quantization: idx_j = argmin_k |y_j - c_k|  (4-bit)
  4. Store: packed 4-bit indices + vec_norm (fp16)

Dequant (optimized):
  1. Gather pre-rotated centroids: ỹ = Pi_centroids[idx]  (precomputed: centroids @ Π)
  2. Rescale: x = ỹ · ||x||

Both K and V are quantized with the same TurboQuant_mse algorithm.

Per Theorem 1, at b=4 the MSE distortion is ≈ 0.009, which is negligible.
Per Figure 1, at b=4 TurboQuant_mse has near-zero inner product bias.
"""

import math

import torch

# ── Bit-width for INT4 TurboQuant_mse ──────────────────────────────────
MSE_BITS = 4

# Fixed seed for reproducible random rotation matrix (shared across all layers)
ROTATION_SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Random matrix construction
# ═══════════════════════════════════════════════════════════════════════


def build_random_rotation(d: int, device: torch.device, seed: int = ROTATION_SEED) -> torch.Tensor:
    """Generate random rotation matrix via QR decomposition.

    Paper Section 3.1: "We can generate Π by applying QR decomposition on
    a random matrix with i.i.d Normal entries."
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diag(R))
    Q = Q * sign.unsqueeze(0)
    return Q.to(device)


def compute_midpoints(centroids: torch.Tensor) -> torch.Tensor:
    """Compute decision boundaries from sorted centroids."""
    c_sorted, _ = centroids.sort()
    return (c_sorted[:-1] + c_sorted[1:]) / 2


def compute_pi_centroids(centroids: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Precompute inverse-rotated centroids: Pi_centroids = centroids @ Pi.

    This eliminates the D×D matmul during dequant — instead of:
        c = centroids[idx]        # (M, D) gather
        x = c @ Pi                # (M, D) matmul — O(M * D^2)

    We precompute once:
        pi_c = centroids @ Pi     # (n_centroids, D) — O(n_centroids * D^2)

    Then dequant is just:
        x = pi_c[idx]             # (M, D) gather — O(M * D)
    """
    return (centroids.to(torch.float32) @ Pi).to(torch.float16)


def compute_paper_slot_size(head_dim: int) -> int:
    """Compute slot size for TurboQuant_mse INT4.

    Per-vector layout:
      [mse_indices (4-bit packed) | vec_norm (fp16)]

    Combined K+V slot = per_vector * 2, aligned to even.
    """
    mse_bytes = math.ceil(head_dim * MSE_BITS / 8)
    norm_bytes = 2  # vec_norm (fp16)
    per_vector = mse_bytes + norm_bytes
    slot = per_vector * 2  # K + V
    return slot + (slot % 2)  # align to even


# ═══════════════════════════════════════════════════════════════════════
# Bit packing / unpacking
# ═══════════════════════════════════════════════════════════════════════


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack int indices (0-15) into 4-bit packed bytes."""
    pairs = indices.reshape(*indices.shape[:-1], -1, 2)
    packed = (pairs[..., 0] & 0xF) | ((pairs[..., 1] & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack 4-bit indices from bytes."""
    low = (packed & 0xF).to(torch.int32)
    high = ((packed >> 4) & 0xF).to(torch.int32)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], D)


# ═══════════════════════════════════════════════════════════════════════
# fp16 <-> bytes conversion (little-endian)
# ═══════════════════════════════════════════════════════════════════════


def fp16_to_bytes(t: torch.Tensor) -> torch.Tensor:
    """Convert float16 tensor to uint8 bytes (LSB-first)."""
    i16 = t.contiguous().view(torch.int16)
    b0 = (i16 & 0xFF).to(torch.uint8)
    b1 = ((i16 >> 8) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1], dim=-1)


def bytes_to_fp16(b: torch.Tensor) -> torch.Tensor:
    """Convert uint8 bytes (LSB-first) to float16."""
    i16 = b[..., 0].to(torch.int16) | (b[..., 1].to(torch.int16) << 8)
    return i16.view(torch.float16)


# ═══════════════════════════════════════════════════════════════════════
# Store: Algorithm 1 Quant_mse for both K and V
# ═══════════════════════════════════════════════════════════════════════


def _quantize_single_vector(
    x: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    midpoints: torch.Tensor,
    D: int,
) -> torch.Tensor:
    """Quantize a batch of vectors using TurboQuant_mse (Algorithm 1).

    Paper Algorithm 1:
      y ← Π · x           (random rotation)
      idx_j ← argmin_k |y_j - c_k|  (Lloyd-Max nearest centroid)
    """
    # ── Step 1: Normalize to unit sphere ───────────────────────────
    norms = x.norm(dim=1, keepdim=True)  # (NH, 1)
    x_hat = x / (norms + 1e-8)

    # ── Step 2: Random rotation: y = Π · x̂ ─────────────────────────
    y = x_hat @ Pi.T  # (NH, D)

    # ── Step 3: Lloyd-Max quantize (4-bit) ─────────────────────────
    indices = torch.bucketize(y, midpoints)  # (NH, D)
    indices = indices.clamp(0, centroids.shape[0] - 1).to(torch.int32)

    # ── Step 4: Pack 4-bit indices and vec_norm ────────────────────
    packed_mse = pack_4bit(indices)  # (NH, D//2)
    norm_bytes = fp16_to_bytes(norms.squeeze(1).to(torch.float16))  # (NH, 2)

    slot_data = torch.cat([packed_mse, norm_bytes], dim=1).to(torch.uint8)
    return slot_data


def store_turboquant_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    midpoints: torch.Tensor,
) -> None:
    """Quantize K and V via TurboQuant_mse and store into combined paged cache."""
    N, Hk, D = key.shape
    block_size = kv_cache.shape[1]
    device = key.device

    # Flatten to (NH, D)
    k_flat = key.float().reshape(N * Hk, D)
    v_flat = value.float().reshape(N * Hk, D)

    # Quantize K and V
    k_slot = _quantize_single_vector(k_flat, Pi, centroids, midpoints, D)
    v_slot = _quantize_single_vector(v_flat, Pi, centroids, midpoints, D)

    # Combine K and V into a single slot
    slot_data = torch.cat([k_slot, v_slot], dim=1).to(torch.uint8)  # (NH, slot_size)

    # Scatter into paged cache
    slots = slot_mapping.unsqueeze(1).expand(-1, Hk).reshape(-1)  # (NH,)
    head_idx = torch.arange(Hk, device=device).unsqueeze(0).expand(N, -1).reshape(-1)

    valid = slots >= 0
    blk = (slots[valid] // block_size).long()
    off = (slots[valid] % block_size).long()
    hid = head_idx[valid].long()
    data = slot_data[valid]

    kv_cache[blk, off, hid, :] = data


# ═══════════════════════════════════════════════════════════════════════
# Dequant: Algorithm 1 DeQuant_mse for both K and V (optimized)
# ═══════════════════════════════════════════════════════════════════════


def _dequant_single_vector(
    slot_flat: torch.Tensor,
    pi_centroids: torch.Tensor,
    D: int,
    offset: int,
) -> torch.Tensor:
    """Dequant a batch of vectors using pre-rotated centroids.

    Optimized: pi_centroids = centroids @ Pi (precomputed once).
    Dequant is pure gather — no matmul.

    Paper Algorithm 1 DeQuant:
      ỹ_j ← c_{idx_j}     (centroid lookup)
      x̂ ← Π^T · ỹ         (inverse rotation)

    With precomputation:
      x̂_unit = (centroids @ Π)[idx]  = pi_centroids[idx]  (gather only)
    """
    mse_bytes = math.ceil(D * MSE_BITS / 8)

    base = offset

    # ── Unpack MSE indices (4-bit) ─────────────────────────────────
    packed_mse = slot_flat[:, base : base + mse_bytes].to(torch.uint8)
    indices = unpack_4bit(packed_mse, D)  # (M, D)

    # ── Unpack vec_norm (fp16) ─────────────────────────────────────
    base += mse_bytes
    norm_bytes = slot_flat[:, base : base + 2].to(torch.uint8)
    vec_norm = bytes_to_fp16(norm_bytes)  # (M,) float16

    # ── Gather pre-rotated centroids (no matmul!) ──────────────────
    x_hat_unit = pi_centroids[indices]  # (M, D) float16 — pure gather

    # ── Rescale by ||x|| ───────────────────────────────────────────
    x_hat = x_hat_unit * vec_norm.unsqueeze(-1)  # (M, D) float16

    return x_hat


def dequant_paged_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    pi_centroids: torch.Tensor,
    k_buf: torch.Tensor | None = None,
    v_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Full dequant of paged TQ cache to float16 K/V buffers (optimized).

    Key optimizations:
    1. Uses pre-rotated centroids (pi_centroids) — no D×D matmul during dequant
    2. Buffer reuse with >= check + slice — no realloc when B shrinks
    3. Direct float16 output — no intermediate float32
    """
    B = block_table.shape[0]
    Hk = kv_cache.shape[2]
    D = pi_centroids.shape[1]
    block_size = kv_cache.shape[1]
    device = kv_cache.device

    max_seq_len = int(seq_lens.max().item())
    alloc_len = math.ceil(max_seq_len / block_size) * block_size

    # Buffer reuse: only realloc if too small (>= check, slice handles shrink)
    if k_buf is None or k_buf.shape[0] < B or k_buf.shape[2] < alloc_len:
        k_buf = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)
    if v_buf is None or v_buf.shape[0] < B or v_buf.shape[2] < alloc_len:
        v_buf = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)

    # Slice to current B (buffer may be larger — zero-cost view)
    k_buf = k_buf[:B, :, :alloc_len, :]
    v_buf = v_buf[:B, :, :alloc_len, :]

    mse_bytes = math.ceil(D * MSE_BITS / 8)
    per_vector = mse_bytes + 2
    slot_size = kv_cache.shape[3]

    # Gather slot data: (B, alloc_len, Hk, slot_size)
    pos = torch.arange(alloc_len, device=device).unsqueeze(0).expand(B, -1)
    page_idx = pos // block_size
    page_off = pos % block_size
    max_page = block_table.shape[1] - 1
    block_nums = block_table.gather(1, page_idx.clamp(max=max_page))

    slot_data = kv_cache[block_nums.long(), page_off.long()]
    # Flatten to (B * alloc_len * Hk, slot_size) for batch dequant
    slot_flat = slot_data.reshape(-1, slot_size)

    # ── Dequant K (offset 0) ───────────────────────────────────────
    packed_mse_k = slot_flat[:, :mse_bytes].to(torch.uint8)
    indices_k = unpack_4bit(packed_mse_k, D)  # (M, D) int32
    norm_bytes_k = slot_flat[:, mse_bytes : mse_bytes + 2].to(torch.uint8)
    vec_norm_k = bytes_to_fp16(norm_bytes_k)  # (M,) float16
    k_recon = pi_centroids[indices_k] * vec_norm_k.unsqueeze(-1)  # (M, D) float16

    # ── Dequant V (offset per_vector) ──────────────────────────────
    v_start = per_vector
    packed_mse_v = slot_flat[:, v_start : v_start + mse_bytes].to(torch.uint8)
    indices_v = unpack_4bit(packed_mse_v, D)  # (M, D) int32
    norm_bytes_v = slot_flat[:, v_start + mse_bytes : v_start + mse_bytes + 2].to(torch.uint8)
    vec_norm_v = bytes_to_fp16(norm_bytes_v)  # (M,) float16
    v_recon = pi_centroids[indices_v] * vec_norm_v.unsqueeze(-1)  # (M, D) float16

    # Reshape: (B, alloc_len, Hk, D) -> (B, Hk, alloc_len, D)
    k_buf[:] = k_recon.reshape(B, alloc_len, Hk, D).permute(0, 2, 1, 3)
    v_buf[:] = v_recon.reshape(B, alloc_len, Hk, D).permute(0, 2, 1, 3)

    return k_buf, v_buf, alloc_len
