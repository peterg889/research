"""Direct tests for _build_rope_correction."""

import math
import torch
import pytest
from lib.kv_cache import _build_rope_correction


class TestBuildRopeCorrection:
    def test_offset_zero_gives_cos1_sin0(self):
        cos, sin = _build_rope_correction(offset=0, head_dim=16, rope_theta=10000.0)
        torch.testing.assert_close(cos, torch.ones(16))
        torch.testing.assert_close(sin, torch.zeros(16), atol=1e-7, rtol=0)

    def test_output_shape(self):
        for hd in [8, 16, 64, 128]:
            cos, sin = _build_rope_correction(offset=5, head_dim=hd, rope_theta=10000.0)
            assert cos.shape == (hd,)
            assert sin.shape == (hd,)

    def test_symmetry_cos(self):
        """cos(-offset) == cos(offset)"""
        cos_pos, _ = _build_rope_correction(offset=7, head_dim=32, rope_theta=10000.0)
        cos_neg, _ = _build_rope_correction(offset=-7, head_dim=32, rope_theta=10000.0)
        torch.testing.assert_close(cos_pos, cos_neg)

    def test_antisymmetry_sin(self):
        """sin(-offset) == -sin(offset)"""
        _, sin_pos = _build_rope_correction(offset=7, head_dim=32, rope_theta=10000.0)
        _, sin_neg = _build_rope_correction(offset=-7, head_dim=32, rope_theta=10000.0)
        torch.testing.assert_close(sin_neg, -sin_pos, atol=1e-6, rtol=1e-6)

    def test_large_offset_no_nan(self):
        cos, sin = _build_rope_correction(offset=100000, head_dim=128, rope_theta=10000.0)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()
        assert not torch.isinf(cos).any()
        assert not torch.isinf(sin).any()

    def test_manual_computation_small_head_dim(self):
        """For head_dim=4, manually verify the frequency and angle computation."""
        head_dim = 4
        rope_theta = 10000.0
        offset = 3

        # inv_freq: indices [0, 2] => 10000^(-0/4), 10000^(-2/4)
        inv_freq_0 = 1.0 / (rope_theta ** (0.0 / head_dim))  # 1.0
        inv_freq_1 = 1.0 / (rope_theta ** (2.0 / head_dim))  # 1/100

        angle_0 = -offset * inv_freq_0  # -3.0
        angle_1 = -offset * inv_freq_1  # -0.03

        expected_cos = torch.tensor([
            math.cos(angle_0), math.cos(angle_1),
            math.cos(angle_0), math.cos(angle_1),  # duplicated
        ])
        expected_sin = torch.tensor([
            math.sin(angle_0), math.sin(angle_1),
            math.sin(angle_0), math.sin(angle_1),
        ])

        cos, sin = _build_rope_correction(offset=offset, head_dim=head_dim, rope_theta=rope_theta)
        torch.testing.assert_close(cos, expected_cos, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sin, expected_sin, atol=1e-5, rtol=1e-5)
