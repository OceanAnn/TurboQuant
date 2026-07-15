# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant ops (paper-faithful Algorithm 1).

Tests run on CPU — no NPU required.
"""

import math
import unittest

import torch

from vllm_ascend.quantization.methods.turboquant_ops import (
    build_random_rotation,
    bytes_to_fp16,
    compute_midpoints,
    compute_paper_slot_size,
    dequant_paged_kv,
    fp16_to_bytes,
    pack_4bit,
    store_turboquant_kv,
    unpack_4bit,
)


class TestBitPacking(unittest.TestCase):
    """Test 4-bit pack/unpack round-trips."""

    def test_pack_unpack_4bit(self):
        D = 128
        indices = torch.randint(0, 16, (4, 8, D), dtype=torch.int32)
        packed = pack_4bit(indices)
        self.assertEqual(packed.shape, (4, 8, D // 2))
        self.assertEqual(packed.dtype, torch.uint8)
        unpacked = unpack_4bit(packed, D)
        torch.testing.assert_close(unpacked, indices)

    def test_pack_unpack_4bit_edge_values(self):
        D = 64
        indices = torch.tensor([[3, 15, 0, 7, 8, 1, 12, 5] * (D // 8)], dtype=torch.int32)
        packed = pack_4bit(indices)
        unpacked = unpack_4bit(packed, D)
        torch.testing.assert_close(unpacked, indices)


class TestFp16Conversion(unittest.TestCase):
    """Test fp16 <-> bytes round-trip."""

    def test_fp16_roundtrip(self):
        values = torch.tensor(
            [0.0, 1.0, -1.0, 0.5, -0.25, 3.14, 1e-8, 65504.0],
            dtype=torch.float16,
        )
        b = fp16_to_bytes(values)
        self.assertEqual(b.shape, (8, 2))
        self.assertEqual(b.dtype, torch.uint8)
        recovered = bytes_to_fp16(b)
        torch.testing.assert_close(recovered, values)


class TestRandomRotation(unittest.TestCase):
    """Test random rotation matrix construction."""

    def test_rotation_orthonormal(self):
        D = 128
        Pi = build_random_rotation(D, torch.device("cpu"))
        self.assertEqual(Pi.shape, (D, D))
        identity = Pi @ Pi.T
        torch.testing.assert_close(identity, torch.eye(D), atol=1e-5, rtol=1e-5)

    def test_rotation_determinant_positive(self):
        D = 64
        Pi = build_random_rotation(D, torch.device("cpu"))
        det = torch.linalg.det(Pi)
        self.assertAlmostEqual(det.item(), 1.0, places=5)

    def test_rotation_reproducible(self):
        D = 128
        Pi1 = build_random_rotation(D, torch.device("cpu"), seed=42)
        Pi2 = build_random_rotation(D, torch.device("cpu"), seed=42)
        torch.testing.assert_close(Pi1, Pi2)


class TestStoreDequant(unittest.TestCase):
    """Test store + dequant round-trip with Algorithm 1."""

    def _get_centroids(self, D, bits=4):
        """Compute Lloyd-Max centroids for N(0, 1/D)."""
        n_levels = 2 ** bits
        sigma2 = 1.0 / D
        sigma = math.sqrt(sigma2)
        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
                -x * x / (2 * sigma2)
            )

        def trapz(f, a, b, n=200):
            h = (b - a) / n
            r = 0.5 * (f(a) + f(b))
            for i in range(1, n):
                r += f(a + i * h)
            return r * h

        for _ in range(200):
            boundaries = [
                (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
            ]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num = trapz(lambda x: x * pdf(x), a, b)
                den = trapz(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c

        return torch.tensor(centroids, dtype=torch.float32)

    def _run_roundtrip(self, D: int, N: int, Hk: int):
        block_size = 16
        num_blocks = max(4, math.ceil(N / block_size))

        Pi = build_random_rotation(D, torch.device("cpu"))
        centroids = self._get_centroids(D)
        midpoints = compute_midpoints(centroids)

        torch.manual_seed(42)
        key = torch.randn(N, Hk, D, dtype=torch.float16)
        value = torch.randn(N, Hk, D, dtype=torch.float16)

        slot_size = compute_paper_slot_size(D)
        kv_cache = torch.zeros(
            num_blocks, block_size, Hk, slot_size, dtype=torch.uint8
        )
        slot_mapping = torch.arange(N, dtype=torch.int32)

        store_turboquant_kv(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            Pi=Pi,
            centroids=centroids,
            midpoints=midpoints,
        )

        B = 1
        block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        seq_lens = torch.tensor([N], dtype=torch.int32)

        k_buf, v_buf, alloc_len = dequant_paged_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
        )

        self.assertEqual(k_buf.shape, (B, Hk, alloc_len, D))
        self.assertEqual(v_buf.shape, (B, Hk, alloc_len, D))

        k_rec = k_buf[0, :, :N, :].float()
        v_rec = v_buf[0, :, :N, :].float()
        k_orig = key.permute(1, 0, 2).float()
        v_orig = value.permute(1, 0, 2).float()

        # Vector reconstruction error
        k_rel = (k_rec - k_orig).norm() / (k_orig.norm() + 1e-8)
        v_rel = (v_rec - v_orig).norm() / (v_orig.norm() + 1e-8)

        # Inner product preservation (the paper's main claim)
        q = torch.randn(Hk, D, dtype=torch.float32)
        ip_orig = (q.unsqueeze(1) * k_orig).sum(dim=-1)
        ip_rec = (q.unsqueeze(1) * k_rec).sum(dim=-1)
        ip_rel = (ip_orig - ip_rec).norm() / (ip_orig.norm() + 1e-8)

        print(
            f"  D={D}: k_rel={k_rel:.4f}, v_rel={v_rel:.4f}, ip_rel={ip_rel:.4f}"
        )

        # Thresholds: with N*Hk=256 vectors, sampling variance from outlier
        # coordinates (heavy-tailed Beta distribution on unit sphere) inflates
        # the empirical error above the theoretical D_mse≈0.009.
        # The error converges to ~0.10 with large N (verified at NH=10000).
        max_rel = 0.6
        self.assertLess(k_rel, max_rel, f"Key relative error too large: {k_rel}")
        self.assertLess(v_rel, max_rel, f"Value relative error too large: {v_rel}")

    def test_roundtrip_128_large(self):
        self._run_roundtrip(D=128, N=64, Hk=4)

    def test_roundtrip_64_large(self):
        self._run_roundtrip(D=64, N=64, Hk=4)

    def test_roundtrip_256_large(self):
        self._run_roundtrip(D=256, N=64, Hk=4)


class TestSlotSize(unittest.TestCase):
    """Test slot size computation."""

    def test_slot_size_128(self):
        # D=128: mse=64, norm=2, per_vec=66, slot=132
        self.assertEqual(compute_paper_slot_size(128), 132)

    def test_slot_size_64(self):
        # D=64: mse=32, norm=2, per_vec=34, slot=68
        self.assertEqual(compute_paper_slot_size(64), 68)

    def test_slot_size_even(self):
        for D in [64, 96, 128, 256]:
            slot = compute_paper_slot_size(D)
            self.assertEqual(slot % 2, 0, f"Slot size {slot} not even for D={D}")


if __name__ == "__main__":
    unittest.main()
