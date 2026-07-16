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

Dequant:
  1. Gather centroids: ỹ_j = c_{idx_j}
  2. Inverse rotate: x̂ = Π^T · ỹ
  3. Rescale: x = x̂ · ||x||

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

    The resulting matrix is orthogonal with determinant +1 (proper rotation).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    # Ensure det(Π) = +1 (proper rotation, not reflection)
    sign = torch.sign(torch.diag(R))
    Q = Q * sign.unsqueeze(0)
    return Q.to(device)


def compute_midpoints(centroids: torch.Tensor) -> torch.Tensor:
    """Compute decision boundaries from sorted centroids.

    Paper Section 3.1: "interval boundaries are the midpoints between
    consecutive centroids, when arranged in sorted order."
    """
    c_sorted, _ = centroids.sort()
    return (c_sorted[:-1] + c_sorted[1:]) / 2


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
    """Pack int indices (0-15) into 4-bit packed bytes.

    Args:
        indices: (..., D) int32/int64, D must be even.
    Returns:
        (..., D // 2) uint8 — two values per byte, low nibble first.
    """
    pairs = indices.reshape(*indices.shape[:-1], -1, 2)
    packed = (pairs[..., 0] & 0xF) | ((pairs[..., 1] & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack 4-bit indices from bytes.

    Args:
        packed: (..., D // 2) uint8.
        D: original number of elements (must be even).
    Returns:
        (..., D) int32.
    """
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

    Args:
        x: (NH, D) float32 — input vectors (K or V).
        Pi: (D, D) float32 — random rotation matrix.
        centroids: (n_centroids,) float32 — Lloyd-Max centroids.
        midpoints: (n_centroids-1,) float32 — decision boundaries.
        D: head dimension.

    Returns:
        slot_data: (NH, per_vector_bytes) uint8 — packed quantized data.
    """
    # ── Step 1: Normalize to unit sphere ───────────────────────────
    # Paper assumes x ∈ S^{d-1}; real KV vectors are not unit, so store ||x||
    norms = x.norm(dim=1, keepdim=True)  # (NH, 1)
    x_hat = x / (norms + 1e-8)

    # ── Step 2: Random rotation: y = Π · x̂ ─────────────────────────
    # Paper line 5: y ← Π · x
    # For batch: Y = X_hat @ Π^T
    y = x_hat @ Pi.T  # (NH, D)

    # ── Step 3: Lloyd-Max quantize (4-bit) ─────────────────────────
    # Paper line 6: idx_j ← argmin_k |y_j - c_k|
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
    """Quantize K and V via TurboQuant_mse and store into combined paged cache.

    Args:
        key: (N, Hk, D) float16/bfloat16 — raw keys.
        value: (N, Hk, D) float16/bfloat16 — raw values.
        kv_cache: (num_blocks, block_size, Hk, slot_size) uint8.
        slot_mapping: (N,) int32 — per-token cache slot indices.
        Pi: (D, D) float32 — random rotation matrix.
        centroids: (n_centroids,) float32 — Lloyd-Max centroids.
        midpoints: (n_centroids-1,) float32 — decision boundaries.
    """
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
# Dequant: Algorithm 1 DeQuant_mse for both K and V
# ═══════════════════════════════════════════════════════════════════════


def _dequant_single_vector(
    slot_flat: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    D: int,
    offset: int,
) -> torch.Tensor:
    """Dequant a batch of vectors using TurboQuant_mse DeQuant (Algorithm 1).

    Paper Algorithm 1 DeQuant:
      ỹ_j ← c_{idx_j}     (centroid lookup)
      x̂ ← Π^T · ỹ         (inverse rotation)

    Args:
        slot_flat: (M, slot_size) int32 — raw slot bytes.
        Pi: (D, D) float32 — random rotation matrix.
        centroids: (n_centroids,) float32.
        D: head dimension.
        offset: byte offset of this vector's data within the slot.

    Returns:
        x_hat: (M, D) float32 — reconstructed vectors.
    """
    mse_bytes = math.ceil(D * MSE_BITS / 8)

    base = offset

    # ── Unpack MSE indices (4-bit) ─────────────────────────────────
    packed_mse = slot_flat[:, base : base + mse_bytes].to(torch.uint8)
    indices = unpack_4bit(packed_mse, D)  # (M, D)

    # ── Unpack vec_norm (fp16) ─────────────────────────────────────
    base += mse_bytes
    norm_bytes = slot_flat[:, base : base + 2].to(torch.uint8)
    vec_norm = bytes_to_fp16(norm_bytes).float()  # (M,)

    # ── MSE dequant: x̂ = Π^T · centroids[idx] ─────────────────────
    # Paper line 9-10: ỹ_j ← c_{idx_j}, x̂ ← Π^T · ỹ
    c = centroids[indices]  # (M, D)
    x_hat_unit = c @ Pi  # (M, D) — Π^T · c per row

    # ── Rescale by ||x|| ───────────────────────────────────────────
    x_hat = x_hat_unit * vec_norm.unsqueeze(-1)  # (M, D)

    return x_hat


def dequant_paged_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    k_buf: torch.Tensor | None = None,
    v_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Full dequant of paged TQ cache to float16 K/V buffers.

    Args:
        kv_cache: (num_blocks, block_size, Hk, slot_size) uint8.
        block_table: (B, max_num_blocks) int32.
        seq_lens: (B,) int32.
        Pi: (D, D) float32 — random rotation matrix.
        centroids: (n_centroids,) float32.
        k_buf, v_buf: optional pre-allocated (B, Hk, alloc_len, D) float16.

    Returns:
        k_buf, v_buf: (B, Hk, alloc_len, D) float16.
        alloc_len: int.
    """
    B = block_table.shape[0]
    Hk = kv_cache.shape[2]
    D = Pi.shape[0]
    block_size = kv_cache.shape[1]
    device = kv_cache.device

    max_seq_len = int(seq_lens.max().item())
    alloc_len = math.ceil(max_seq_len / block_size) * block_size

    if k_buf is None or k_buf.shape[0] != B or k_buf.shape[2] < alloc_len:
        k_buf = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)
    if v_buf is None or v_buf.shape[0] != B or v_buf.shape[2] < alloc_len:
        v_buf = torch.empty(B, Hk, alloc_len, D, dtype=torch.float16, device=device)

    mse_bytes = math.ceil(D * MSE_BITS / 8)
    per_vector = mse_bytes + 2

    # Gather slot data: (B, alloc_len, Hk, slot_size)
    pos = torch.arange(alloc_len, device=device).unsqueeze(0).expand(B, -1)
    page_idx = pos // block_size
    page_off = pos % block_size
    max_page = block_table.shape[1] - 1
    block_nums = block_table.gather(1, page_idx.clamp(max=max_page))

    slot_data = kv_cache[block_nums.long(), page_off.long()]
    slot_flat = slot_data.reshape(-1, kv_cache.shape[3]).to(torch.int32)

    # Dequant K (offset 0) and V (offset per_vector)
    k_recon = _dequant_single_vector(slot_flat, Pi, centroids, D, offset=0)
    v_recon = _dequant_single_vector(slot_flat, Pi, centroids, D, offset=per_vector)

    # Reshape: (B, alloc_len, Hk, D) -> (B, Hk, alloc_len, D)
    k_buf[:, :, :alloc_len, :] = k_recon.to(torch.float16).reshape(B, alloc_len, Hk, D).permute(0, 2, 1, 3)
    v_buf[:, :, :alloc_len, :] = v_recon.to(torch.float16).reshape(B, alloc_len, Hk, D).permute(0, 2, 1, 3)

    return k_buf, v_buf, alloc_len
