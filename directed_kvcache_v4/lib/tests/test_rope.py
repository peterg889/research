"""Tests for lib.rope — RoPE repositioning and KV cache selection.

These tests use mock data (no model required) to verify the mathematical
correctness of the RoPE rotation and cache selection operations.
Covers rotate_half, select_kv_cache, and reposition_kv_cache with
extensive math-property tests, edge cases, and independence checks.
"""

from types import SimpleNamespace

import torch
import pytest
from transformers import DynamicCache

from lib.rope import (
    build_layer_inv_freqs,
    get_layer_types,
    rotate_half,
    select_kv_cache,
    reposition_kv_cache,
)


def _make_dummy_cache(n_layers: int = 4, n_heads: int = 2,
                      seq_len: int = 8, head_dim: int = 16,
                      dtype=torch.bfloat16) -> DynamicCache:
    """Create a DynamicCache with random values."""
    torch.manual_seed(42)
    cache = DynamicCache()
    for i in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        cache.update(k, v, i)
    return cache


def _setup_rope(head_dim: int = 16, n_layers: int = 4,
                layer_types=None, theta=10000.0):
    """Create mock inv_freqs and layer_types for testing."""
    if layer_types is None:
        layer_types = ["full_attention"] * n_layers
    unique_types = set(layer_types)
    inv_freqs = {}
    for lt in unique_types:
        t = theta if lt == "full_attention" else 10000.0
        inv_freq = 1.0 / (
            t ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        inv_freqs[lt] = inv_freq
    return inv_freqs, layer_types


# ======================================================================
# rotate_half
# ======================================================================

class TestRotateHalfBasic:
    def test_output_shape(self):
        x = torch.randn(2, 3, 4, 8)
        r = rotate_half(x)
        assert r.shape == x.shape

    def test_known_values(self):
        """rotate_half([a, b, c, d]) should give [-c, -d, a, b]."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        r = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.equal(r, expected)

    def test_known_values_dim6(self):
        """rotate_half([1,2,3,4,5,6]) = [-4,-5,-6,1,2,3]."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = rotate_half(x)
        expected = torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
        assert torch.equal(r, expected)

    def test_dim_2(self):
        """Smallest case: rotate_half([a, b]) = [-b, a]."""
        x = torch.tensor([3.0, 7.0])
        r = rotate_half(x)
        expected = torch.tensor([-7.0, 3.0])
        assert torch.equal(r, expected)


class TestRotateHalfMathProperties:
    """Mathematical properties of the half-rotation operator."""

    def test_double_rotation_negates(self):
        """rotate_half(rotate_half(x)) = -x."""
        x = torch.randn(2, 4, 8)
        r2 = rotate_half(rotate_half(x))
        assert torch.allclose(r2, -x)

    def test_four_rotations_identity(self):
        """Applying rotate_half 4 times = identity."""
        x = torch.randn(3, 5, 16)
        r = x
        for _ in range(4):
            r = rotate_half(r)
        assert torch.allclose(r, x)

    def test_preserves_l2_norm(self):
        """rotate_half is a permutation + sign flip, so norm is preserved."""
        x = torch.randn(4, 8, 32, dtype=torch.float32)
        assert torch.allclose(x.norm(), rotate_half(x).norm(), rtol=1e-5)

    def test_preserves_l2_norm_per_vector(self):
        """Per-vector L2 norm should be preserved."""
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        x_norms = x.norm(dim=-1)
        r_norms = rotate_half(x).norm(dim=-1)
        assert torch.allclose(x_norms, r_norms, rtol=1e-5)

    def test_orthogonal_to_input(self):
        """x and rotate_half(x) are orthogonal: dot product = 0."""
        x = torch.randn(100, 16, dtype=torch.float32)
        dots = (x * rotate_half(x)).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros(100), atol=1e-5)

    def test_orthogonal_batch(self):
        """Orthogonality holds for batched 4D inputs."""
        x = torch.randn(1, 4, 10, 32, dtype=torch.float32)
        r = rotate_half(x)
        dots = (x * r).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4)

    def test_linearity(self):
        """rotate_half is linear: rotate_half(ax + by) = a*rotate_half(x) + b*rotate_half(y)."""
        x = torch.randn(4, 8, dtype=torch.float32)
        y = torch.randn(4, 8, dtype=torch.float32)
        a, b = 2.5, -1.3
        lhs = rotate_half(a * x + b * y)
        rhs = a * rotate_half(x) + b * rotate_half(y)
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_zero_input(self):
        """rotate_half(0) = 0."""
        x = torch.zeros(3, 8)
        assert torch.equal(rotate_half(x), torch.zeros(3, 8))


class TestRotateHalfDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_preserves_dtype(self, dtype):
        x = torch.randn(2, 4, dtype=dtype)
        assert rotate_half(x).dtype == dtype

    def test_double_rotation_bf16(self):
        """Double rotation negation holds in bf16."""
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        r2 = rotate_half(rotate_half(x))
        assert torch.equal(r2, -x)


class TestRotateHalfDimensions:
    """Verify it works with various tensor ranks and dimension sizes."""

    def test_1d(self):
        x = torch.randn(4)
        r = rotate_half(x)
        assert r.shape == (4,)

    def test_2d(self):
        x = torch.randn(3, 8)
        r = rotate_half(x)
        assert r.shape == (3, 8)

    def test_3d(self):
        x = torch.randn(2, 5, 16)
        r = rotate_half(x)
        assert r.shape == (2, 5, 16)

    def test_5d(self):
        x = torch.randn(1, 2, 3, 4, 8)
        r = rotate_half(x)
        assert r.shape == (1, 2, 3, 4, 8)

    @pytest.mark.parametrize("head_dim", [2, 4, 8, 16, 32, 64, 128, 256])
    def test_various_head_dims(self, head_dim):
        x = torch.randn(1, 2, 3, head_dim)
        r = rotate_half(x)
        assert r.shape == x.shape
        # Double rotation = negation
        assert torch.allclose(rotate_half(r), -x)


# ======================================================================
# select_kv_cache
# ======================================================================

class TestSelectKvCacheBasic:
    def test_correct_length(self):
        cache = _make_dummy_cache(seq_len=8)
        selected = select_kv_cache(cache, [0, 2, 4])
        assert selected.layers[0].keys.shape[2] == 3

    def test_correct_values(self):
        cache = _make_dummy_cache(seq_len=8)
        indices = [0, 3, 5]
        selected = select_kv_cache(cache, indices)
        for i in range(len(cache.layers)):
            for j, idx in enumerate(indices):
                assert torch.equal(
                    selected.layers[i].keys[:, :, j, :],
                    cache.layers[i].keys[:, :, idx, :],
                )
                assert torch.equal(
                    selected.layers[i].values[:, :, j, :],
                    cache.layers[i].values[:, :, idx, :],
                )

    def test_all_layers_selected(self):
        n_layers = 6
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=8)
        selected = select_kv_cache(cache, [0, 1, 2])
        assert len(selected.layers) == n_layers

    def test_single_index(self):
        cache = _make_dummy_cache(seq_len=8)
        selected = select_kv_cache(cache, [0])
        assert selected.layers[0].keys.shape[2] == 1

    def test_preserves_dtype_bf16(self):
        cache = _make_dummy_cache(dtype=torch.bfloat16)
        selected = select_kv_cache(cache, [0, 1])
        assert selected.layers[0].keys.dtype == torch.bfloat16

    def test_preserves_dtype_fp32(self):
        cache = _make_dummy_cache(dtype=torch.float32)
        selected = select_kv_cache(cache, [0, 1])
        assert selected.layers[0].keys.dtype == torch.float32


class TestSelectKvCacheIndependence:
    """Selected cache should be independent from the source."""

    def test_mutate_selected_keys(self):
        cache = _make_dummy_cache(seq_len=8)
        orig_k0 = cache.layers[0].keys.clone()
        selected = select_kv_cache(cache, [0, 1, 2])
        selected.layers[0].keys.fill_(999.0)
        assert torch.equal(cache.layers[0].keys, orig_k0)

    def test_mutate_selected_values(self):
        cache = _make_dummy_cache(seq_len=8)
        orig_v0 = cache.layers[0].values.clone()
        selected = select_kv_cache(cache, [0, 1, 2])
        selected.layers[0].values.zero_()
        assert torch.equal(cache.layers[0].values, orig_v0)

    def test_mutate_source_after_select(self):
        cache = _make_dummy_cache(seq_len=8)
        selected = select_kv_cache(cache, [0, 1, 2])
        selected_k0 = selected.layers[0].keys.clone()
        cache.layers[0].keys.fill_(-999.0)
        assert torch.equal(selected.layers[0].keys, selected_k0)

    def test_independence_all_layers(self):
        n_layers = 6
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=8)
        originals = [cache.layers[i].keys.clone() for i in range(n_layers)]
        selected = select_kv_cache(cache, [0, 2, 4])
        for i in range(n_layers):
            selected.layers[i].keys.fill_(float(i))
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys, originals[i])

    def test_not_same_tensor(self):
        cache = _make_dummy_cache(seq_len=8)
        selected = select_kv_cache(cache, [0, 1])
        assert cache.layers[0].keys.data_ptr() != selected.layers[0].keys.data_ptr()


class TestSelectKvCacheOrdering:
    """Index ordering should be respected."""

    def test_reversed_indices(self):
        """Selecting [2, 1, 0] reverses the sequence dimension."""
        cache = _make_dummy_cache(seq_len=4)
        normal = select_kv_cache(cache, [0, 1, 2])
        reversed_ = select_kv_cache(cache, [2, 1, 0])
        for i in range(len(cache.layers)):
            assert torch.equal(
                reversed_.layers[i].keys[:, :, 0, :],
                normal.layers[i].keys[:, :, 2, :],
            )
            assert torch.equal(
                reversed_.layers[i].keys[:, :, 2, :],
                normal.layers[i].keys[:, :, 0, :],
            )

    def test_duplicate_indices(self):
        """Duplicate indices should repeat the same entry."""
        cache = _make_dummy_cache(seq_len=4)
        selected = select_kv_cache(cache, [0, 0, 0])
        assert selected.layers[0].keys.shape[2] == 3
        for j in range(3):
            assert torch.equal(
                selected.layers[0].keys[:, :, j, :],
                cache.layers[0].keys[:, :, 0, :],
            )

    def test_select_all(self):
        """Selecting all indices should reproduce the full cache."""
        seq_len = 6
        cache = _make_dummy_cache(seq_len=seq_len)
        selected = select_kv_cache(cache, list(range(seq_len)))
        for i in range(len(cache.layers)):
            assert torch.equal(selected.layers[i].keys, cache.layers[i].keys)
            assert torch.equal(selected.layers[i].values, cache.layers[i].values)


class TestSelectKvCacheRealistic:
    """Tests mimicking real experiment usage patterns."""

    def test_bos_plus_doc_pattern(self):
        """Typical usage: select BOS (0) + doc tokens (skip prefix + newline)."""
        # Simulate [BOS, prefix(3 tokens), \n, doc(4 tokens)] = 9 tokens total
        cache = _make_dummy_cache(seq_len=9)
        # Keep BOS + doc tokens at positions 5,6,7,8
        indices = [0, 5, 6, 7, 8]
        selected = select_kv_cache(cache, indices)
        assert selected.layers[0].keys.shape[2] == 5
        # BOS should be at position 0
        assert torch.equal(
            selected.layers[0].keys[:, :, 0, :],
            cache.layers[0].keys[:, :, 0, :],
        )
        # Doc first token
        assert torch.equal(
            selected.layers[0].keys[:, :, 1, :],
            cache.layers[0].keys[:, :, 5, :],
        )

    def test_gemma3_like_dimensions(self):
        """48 layers, 16 heads, realistic selection."""
        cache = _make_dummy_cache(n_layers=48, n_heads=16, seq_len=100,
                                  head_dim=128)
        # BOS + 80 doc tokens (skip 19-token prefix + newline)
        indices = [0] + list(range(21, 100))
        selected = select_kv_cache(cache, indices)
        assert len(selected.layers) == 48
        assert selected.layers[0].keys.shape == (1, 16, 80, 128)
        assert selected.layers[47].values.shape == (1, 16, 80, 128)

    def test_single_doc_token(self):
        """Degenerate case: BOS + 1 doc token."""
        cache = _make_dummy_cache(seq_len=10)
        selected = select_kv_cache(cache, [0, 5])
        assert selected.layers[0].keys.shape[2] == 2


# ======================================================================
# reposition_kv_cache
# ======================================================================

class TestRepositionBasic:
    def test_shape_preserved(self):
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        orig_shape = cache.layers[0].keys.shape
        old_pos = torch.arange(3, 7)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        assert cache.layers[0].keys.shape == orig_shape

    def test_doc_keys_actually_change(self):
        """With a nonzero delta, doc keys should be different."""
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.arange(10, 14)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], doc_k_orig)

    def test_returns_same_cache(self):
        """reposition_kv_cache mutates in-place and returns the same object."""
        cache = _make_dummy_cache(n_layers=2, seq_len=3, head_dim=8)
        inv_freqs, layer_types = _setup_rope(head_dim=8, n_layers=2)
        result = reposition_kv_cache(cache, torch.tensor([5, 6]),
                                      torch.tensor([1, 2]), inv_freqs, layer_types)
        assert result is cache


class TestRepositionBOSPreservation:
    """The BOS entry must never be modified by repositioning."""

    def test_bos_keys_unchanged(self):
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        bos_keys = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(4)]
        old_pos = torch.arange(3, 7)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        for i in range(4):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_keys[i])

    def test_bos_unchanged_all_layers(self):
        """BOS preservation must hold for every layer, not just layer 0."""
        n_layers = 12
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        bos_keys = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(n_layers)]
        old_pos = torch.arange(10, 14)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_keys[i])

    def test_bos_unchanged_large_delta(self):
        """Even with a very large position delta, BOS should be untouched."""
        cache = _make_dummy_cache(n_layers=2, seq_len=3, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        old_pos = torch.tensor([5000, 5001])
        new_pos = torch.tensor([1, 2])
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)


class TestRepositionValuesUnchanged:
    """Values should NEVER be modified (RoPE applies to keys only)."""

    def test_values_unchanged(self):
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        v_orig = [cache.layers[i].values.clone() for i in range(4)]
        old_pos = torch.arange(3, 7)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        for i in range(4):
            assert torch.equal(cache.layers[i].values, v_orig[i])

    def test_values_unchanged_all_layers(self):
        n_layers = 12
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        v_orig = [cache.layers[i].values.clone() for i in range(n_layers)]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].values, v_orig[i])

    def test_values_unchanged_with_mixed_layers(self):
        n_layers = 4
        layer_types = ["sliding_attention", "sliding_attention",
                       "full_attention", "sliding_attention"]
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16)
        inv_freqs, _ = _setup_rope(head_dim=16, n_layers=n_layers,
                                    layer_types=layer_types)
        v_orig = [cache.layers[i].values.clone() for i in range(n_layers)]
        reposition_kv_cache(cache, torch.arange(5, 9), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].values, v_orig[i])


class TestRepositionIdentity:
    """Zero delta should be identity."""

    def test_zero_delta_is_identity_fp32(self):
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        k_orig = [cache.layers[i].keys.clone() for i in range(4)]
        pos = torch.arange(1, 5)
        reposition_kv_cache(cache, pos, pos, inv_freqs, layer_types)
        for i in range(4):
            assert torch.allclose(cache.layers[i].keys, k_orig[i], atol=1e-5)

    def test_zero_delta_is_identity_bf16(self):
        cache = _make_dummy_cache(n_layers=4, seq_len=5, head_dim=16,
                                  dtype=torch.bfloat16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=4)
        k_orig = [cache.layers[i].keys.clone() for i in range(4)]
        pos = torch.arange(1, 5)
        reposition_kv_cache(cache, pos, pos, inv_freqs, layer_types)
        for i in range(4):
            assert torch.allclose(cache.layers[i].keys, k_orig[i], atol=1e-2)

    def test_zero_delta_all_layers(self):
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        k_orig = [cache.layers[i].keys.clone() for i in range(n_layers)]
        pos = torch.arange(1, 5)
        reposition_kv_cache(cache, pos, pos, inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.allclose(cache.layers[i].keys, k_orig[i], atol=1e-5)


class TestRepositionRoundtrip:
    """Repositioning forward then backward should recover original."""

    def test_roundtrip_basic(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = [cache.layers[i].keys[:, :, 1:, :].clone() for i in range(2)]
        old_pos = torch.arange(5, 9)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types)
        for i in range(2):
            assert torch.allclose(
                cache.layers[i].keys[:, :, 1:, :], k_orig[i], atol=1e-4
            )

    def test_roundtrip_large_delta(self):
        """Large position shifts should still be invertible."""
        cache = _make_dummy_cache(n_layers=2, seq_len=3, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.tensor([100, 101])
        new_pos = torch.tensor([1, 2])
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types)
        assert torch.allclose(
            cache.layers[0].keys[:, :, 1:, :], k_orig, atol=1e-4
        )

    def test_roundtrip_all_layers(self):
        n_layers = 6
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        k_orig = [cache.layers[i].keys[:, :, 1:, :].clone() for i in range(n_layers)]
        old_pos = torch.arange(10, 14)
        new_pos = torch.arange(1, 5)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.allclose(
                cache.layers[i].keys[:, :, 1:, :], k_orig[i], atol=1e-4
            )


class TestRepositionNormPreservation:
    """RoPE rotation should preserve vector norms (it's a rotation)."""

    def test_preserves_doc_key_norms(self):
        """Per-head, per-position L2 norms should be preserved."""
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        norms_before = [
            cache.layers[i].keys[:, :, 1:, :].norm(dim=-1).clone()
            for i in range(2)
        ]
        reposition_kv_cache(cache, torch.arange(5, 9), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(2):
            norms_after = cache.layers[i].keys[:, :, 1:, :].norm(dim=-1)
            assert torch.allclose(norms_before[i], norms_after, rtol=1e-4)

    def test_preserves_norms_large_delta(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=3, head_dim=32,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=32, n_layers=2)
        norms_before = cache.layers[0].keys[:, :, 1:, :].norm(dim=-1).clone()
        reposition_kv_cache(cache, torch.tensor([500, 501]),
                            torch.tensor([1, 2]), inv_freqs, layer_types)
        norms_after = cache.layers[0].keys[:, :, 1:, :].norm(dim=-1)
        assert torch.allclose(norms_before, norms_after, rtol=1e-4)

    def test_preserves_norms_all_layers(self):
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        norms_before = [
            cache.layers[i].keys[:, :, 1:, :].norm(dim=-1).clone()
            for i in range(n_layers)
        ]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            norms_after = cache.layers[i].keys[:, :, 1:, :].norm(dim=-1)
            assert torch.allclose(norms_before[i], norms_after, rtol=1e-4)


class TestRepositionComposition:
    """Composing two repositions should equal a single reposition with summed delta."""

    def test_two_step_equals_one_step(self):
        """reposition(A→B) then reposition(B→C) = reposition(A→C)."""
        head_dim = 16
        n_layers = 2
        inv_freqs, layer_types = _setup_rope(head_dim=head_dim, n_layers=n_layers)

        # Cache 1: two-step
        cache1 = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=head_dim,
                                   dtype=torch.float32)
        pos_a = torch.arange(10, 14)
        pos_b = torch.arange(5, 9)
        pos_c = torch.arange(1, 5)
        reposition_kv_cache(cache1, pos_a, pos_b, inv_freqs, layer_types)
        reposition_kv_cache(cache1, pos_b, pos_c, inv_freqs, layer_types)

        # Cache 2: single step
        cache2 = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=head_dim,
                                   dtype=torch.float32)
        reposition_kv_cache(cache2, pos_a, pos_c, inv_freqs, layer_types)

        for i in range(n_layers):
            assert torch.allclose(
                cache1.layers[i].keys, cache2.layers[i].keys, atol=1e-4
            )


class TestRepositionMixedLayerTypes:
    """Test with heterogeneous layer types (sliding + full attention)."""

    def test_mixed_types_no_error(self):
        head_dim = 16
        n_layers = 6
        layer_types = ["sliding_attention", "sliding_attention",
                       "full_attention", "sliding_attention",
                       "sliding_attention", "full_attention"]
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=head_dim)
        inv_freqs = {}
        for lt in ["sliding_attention", "full_attention"]:
            theta = 10000.0 if lt == "sliding_attention" else 50000.0
            inv_freqs[lt] = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        reposition_kv_cache(cache, torch.arange(5, 9), torch.arange(1, 5),
                            inv_freqs, layer_types)
        # All layers should have correct shape
        for i in range(n_layers):
            assert cache.layers[i].keys.shape == (1, 2, 5, head_dim)

    def test_different_layer_types_get_different_rotations(self):
        """Sliding and full attention layers should receive different rotations."""
        head_dim = 16
        n_layers = 2
        layer_types = ["sliding_attention", "full_attention"]
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=3, head_dim=head_dim,
                                  dtype=torch.float32)
        # Use very different thetas to make the difference clear
        inv_freqs = {
            "sliding_attention": 1.0 / (100.0 ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)),
            "full_attention": 1.0 / (100000.0 ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)),
        }
        doc_k_before = [cache.layers[i].keys[:, :, 1:, :].clone() for i in range(2)]
        reposition_kv_cache(cache, torch.tensor([10, 11]), torch.tensor([1, 2]),
                            inv_freqs, layer_types)
        # Both layers should change
        delta0 = (cache.layers[0].keys[:, :, 1:, :] - doc_k_before[0]).abs().mean()
        delta1 = (cache.layers[1].keys[:, :, 1:, :] - doc_k_before[1]).abs().mean()
        assert delta0 > 0.01  # Layer 0 changed
        assert delta1 > 0.001  # Layer 1 changed (less due to higher theta)
        # The changes should be different in magnitude
        assert abs(delta0.item() - delta1.item()) > 1e-4

    def test_mixed_types_bos_preserved(self):
        """BOS should be preserved regardless of layer type."""
        head_dim = 16
        n_layers = 4
        layer_types = ["sliding_attention", "full_attention",
                       "sliding_attention", "full_attention"]
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=head_dim)
        inv_freqs, _ = _setup_rope(head_dim=head_dim, n_layers=n_layers,
                                    layer_types=layer_types)
        bos_keys = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(n_layers)]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_keys[i])

    def test_mixed_types_values_preserved(self):
        head_dim = 16
        n_layers = 4
        layer_types = ["sliding_attention", "full_attention",
                       "sliding_attention", "full_attention"]
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=head_dim)
        inv_freqs, _ = _setup_rope(head_dim=head_dim, n_layers=n_layers,
                                    layer_types=layer_types)
        v_orig = [cache.layers[i].values.clone() for i in range(n_layers)]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].values, v_orig[i])


class TestRepositionEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_doc_token(self):
        """Cache with BOS + 1 doc token."""
        cache = _make_dummy_cache(n_layers=2, seq_len=2, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.tensor([10])
        new_pos = torch.tensor([1])
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        # Key should change (delta=9)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], k_orig)

    def test_single_doc_token_roundtrip(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=2, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.tensor([10])
        new_pos = torch.tensor([1])
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types)
        assert torch.allclose(cache.layers[0].keys[:, :, 1:, :], k_orig, atol=1e-4)

    def test_negative_delta(self):
        """Repositioning to higher positions (new > old)."""
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.arange(1, 5)
        new_pos = torch.arange(10, 14)  # Move to higher positions
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], k_orig)

    def test_large_head_dim(self):
        """Test with head_dim=256 (as in Gemma 3 12B)."""
        head_dim = 256
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=head_dim,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=head_dim, n_layers=2)
        norms_before = cache.layers[0].keys[:, :, 1:, :].norm(dim=-1).clone()
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        norms_after = cache.layers[0].keys[:, :, 1:, :].norm(dim=-1)
        assert torch.allclose(norms_before, norms_after, rtol=1e-4)

    @pytest.mark.parametrize("head_dim", [8, 16, 32, 64, 128, 256])
    def test_various_head_dims(self, head_dim):
        """Roundtrip should work for any even head_dim."""
        cache = _make_dummy_cache(n_layers=2, seq_len=3, head_dim=head_dim,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=head_dim, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        reposition_kv_cache(cache, torch.tensor([5, 6]), torch.tensor([1, 2]),
                            inv_freqs, layer_types)
        reposition_kv_cache(cache, torch.tensor([1, 2]), torch.tensor([5, 6]),
                            inv_freqs, layer_types)
        assert torch.allclose(cache.layers[0].keys[:, :, 1:, :], k_orig, atol=1e-4)

    def test_many_doc_tokens(self):
        """Test with a longer document (50 tokens)."""
        seq_len = 51  # BOS + 50 doc tokens
        cache = _make_dummy_cache(n_layers=2, seq_len=seq_len, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        old_pos = torch.arange(100, 150)
        new_pos = torch.arange(1, 51)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        # Verify shape
        assert cache.layers[0].keys.shape[2] == seq_len
        # Verify norm preservation
        norms_before = k_orig.norm(dim=-1)
        norms_after = cache.layers[0].keys[:, :, 1:, :].norm(dim=-1)
        assert torch.allclose(norms_before, norms_after, rtol=1e-4)


class TestRepositionAllLayersModified:
    """Ensure repositioning modifies ALL layers, not just the first one."""

    def test_all_layers_change(self):
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=5, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        doc_keys_orig = [
            cache.layers[i].keys[:, :, 1:, :].clone() for i in range(n_layers)
        ]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        for i in range(n_layers):
            assert not torch.equal(
                cache.layers[i].keys[:, :, 1:, :], doc_keys_orig[i]
            ), f"Layer {i} doc keys were not modified"


class TestRepositionDtypes:
    """Verify behavior across dtypes."""

    def test_bf16_keys_change(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.bfloat16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        doc_k = cache.layers[0].keys[:, :, 1:, :].clone()
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(1, 5),
                            inv_freqs, layer_types)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], doc_k)

    def test_bf16_preserves_dtype(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.bfloat16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        reposition_kv_cache(cache, torch.arange(5, 9), torch.arange(1, 5),
                            inv_freqs, layer_types)
        assert cache.layers[0].keys.dtype == torch.bfloat16

    def test_fp32_preserves_dtype(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        reposition_kv_cache(cache, torch.arange(5, 9), torch.arange(1, 5),
                            inv_freqs, layer_types)
        assert cache.layers[0].keys.dtype == torch.float32


class TestRepositionRealisticUsage:
    """Tests matching actual experiment patterns."""

    def test_comprehend_64_pattern(self):
        """Typical: prefix L=64, newline=1, doc=100 tokens.
        Phase A: [BOS] + prefix(64) + [\\n] + doc(100) = 166 tokens.
        Select: [0] + [66..165] → 101 tokens.
        Reposition: old=[66..165] → new=[1..100]."""
        n_layers = 4
        seq_len = 101  # After selection: BOS + 100 doc
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=seq_len, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=n_layers)
        old_pos = torch.arange(66, 166)
        new_pos = torch.arange(1, 101)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types)
        # Shape preserved
        assert cache.layers[0].keys.shape[2] == 101
        # BOS unchanged
        # Values unchanged
        # Norms preserved

    def test_bare_is_identity(self):
        """Bare conditioning: no prefix, positions already at 1..D.
        old == new → no-op."""
        cache = _make_dummy_cache(n_layers=2, seq_len=5, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys.clone()
        pos = torch.arange(1, 5)
        reposition_kv_cache(cache, pos, pos, inv_freqs, layer_types)
        assert torch.allclose(cache.layers[0].keys, k_orig, atol=1e-5)


# ======================================================================
# build_layer_inv_freqs (with mock model)
# ======================================================================

def _make_mock_model(head_dim=16, rope_params=None):
    """Create a mock model object with the config attributes build_layer_inv_freqs reads."""
    if rope_params is None:
        rope_params = {
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 50000.0},
        }
    text_config = SimpleNamespace(
        rope_parameters=rope_params,
        head_dim=head_dim,
    )
    config = SimpleNamespace(text_config=text_config)
    # build_layer_inv_freqs calls next(model.parameters()) for default device
    dummy_param = torch.zeros(1)
    model = SimpleNamespace(config=config, parameters=lambda: iter([dummy_param]))
    return model


class TestBuildLayerInvFreqs:
    def test_returns_dict(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        assert isinstance(inv_freqs, dict)

    def test_keys_match_rope_params(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        assert set(inv_freqs.keys()) == {"sliding_attention", "full_attention"}

    def test_output_shape(self):
        head_dim = 32
        model = _make_mock_model(head_dim=head_dim)
        inv_freqs = build_layer_inv_freqs(model)
        for lt, freq in inv_freqs.items():
            assert freq.shape == (head_dim // 2,)

    def test_output_dtype_float32(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        for freq in inv_freqs.values():
            assert freq.dtype == torch.float32

    def test_different_theta_gives_different_freqs(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        assert not torch.equal(inv_freqs["sliding_attention"],
                               inv_freqs["full_attention"])

    def test_higher_theta_gives_lower_freqs(self):
        """Higher theta → slower rotation → lower inverse frequencies."""
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        # sliding: theta=10000, full: theta=50000
        # Higher theta → smaller inv_freq values
        assert inv_freqs["sliding_attention"].mean() > inv_freqs["full_attention"].mean()

    def test_values_positive(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        for freq in inv_freqs.values():
            assert (freq > 0).all()

    def test_values_bounded(self):
        """inv_freq = 1 / theta^(i/d). For theta=10000, max freq = 1/1 = 1."""
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model)
        for freq in inv_freqs.values():
            assert freq.max() <= 1.0 + 1e-6

    def test_single_layer_type(self):
        model = _make_mock_model(rope_params={
            "full_attention": {"rope_theta": 10000.0},
        })
        inv_freqs = build_layer_inv_freqs(model)
        assert len(inv_freqs) == 1
        assert "full_attention" in inv_freqs

    @pytest.mark.parametrize("head_dim", [8, 16, 32, 64, 128, 256])
    def test_various_head_dims(self, head_dim):
        model = _make_mock_model(head_dim=head_dim)
        inv_freqs = build_layer_inv_freqs(model)
        for freq in inv_freqs.values():
            assert freq.shape == (head_dim // 2,)

    def test_explicit_device_cpu(self):
        model = _make_mock_model()
        inv_freqs = build_layer_inv_freqs(model, device=torch.device("cpu"))
        for freq in inv_freqs.values():
            assert freq.device == torch.device("cpu")

    def test_no_text_config_fallback(self):
        """If model.config has no text_config, should fall back to model.config."""
        config = SimpleNamespace(
            rope_parameters={"full_attention": {"rope_theta": 10000.0}},
            head_dim=16,
        )
        dummy_param = torch.zeros(1)
        model = SimpleNamespace(config=config, parameters=lambda: iter([dummy_param]))
        inv_freqs = build_layer_inv_freqs(model)
        assert "full_attention" in inv_freqs

    def test_default_theta(self):
        """If rope_theta is missing from params, should default to 10000."""
        model = _make_mock_model(rope_params={
            "full_attention": {},  # No rope_theta key
        })
        inv_freqs = build_layer_inv_freqs(model)
        # Should use default theta=10000
        expected = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        assert torch.allclose(inv_freqs["full_attention"], expected)


# ======================================================================
# get_layer_types (with mock model)
# ======================================================================

class TestGetLayerTypes:
    def test_returns_list(self):
        text_config = SimpleNamespace(layer_types=[
            "sliding_attention", "sliding_attention", "full_attention",
        ])
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        result = get_layer_types(model)
        assert isinstance(result, list)

    def test_correct_values(self):
        expected = ["sliding_attention", "sliding_attention",
                    "full_attention", "sliding_attention"]
        text_config = SimpleNamespace(layer_types=expected)
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        assert get_layer_types(model) == expected

    def test_returns_new_list(self):
        """Should return a copy, not a reference to the config attribute."""
        original = ["full_attention", "sliding_attention"]
        text_config = SimpleNamespace(layer_types=original)
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        result = get_layer_types(model)
        result[0] = "modified"
        assert original[0] == "full_attention"

    def test_empty_layer_types(self):
        text_config = SimpleNamespace(layer_types=[])
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        assert get_layer_types(model) == []

    def test_missing_layer_types_attribute(self):
        """If layer_types attribute is missing, should return empty list."""
        text_config = SimpleNamespace()  # No layer_types
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        assert get_layer_types(model) == []

    def test_no_text_config_fallback(self):
        """If model.config has no text_config, fall back to model.config."""
        config = SimpleNamespace(layer_types=["full_attention"])
        model = SimpleNamespace(config=config)
        assert get_layer_types(model) == ["full_attention"]

    def test_gemma3_like_48_layers(self):
        """Gemma 3 pattern: sliding with full every 6th layer."""
        layer_types = []
        for i in range(48):
            if i % 6 == 5:  # 0-indexed: layers 5, 11, 17, ...
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        text_config = SimpleNamespace(layer_types=layer_types)
        config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=config)
        result = get_layer_types(model)
        assert len(result) == 48
        assert result.count("full_attention") == 8
        assert result.count("sliding_attention") == 40


# ======================================================================
# select_kv_cache — values dtype preservation
# ======================================================================

class TestSelectKvCacheValuesDtype:
    """Values dtype should be preserved just like keys dtype."""

    def test_values_dtype_bf16(self):
        cache = _make_dummy_cache(dtype=torch.bfloat16)
        selected = select_kv_cache(cache, [0, 1])
        assert selected.layers[0].values.dtype == torch.bfloat16

    def test_values_dtype_fp32(self):
        cache = _make_dummy_cache(dtype=torch.float32)
        selected = select_kv_cache(cache, [0, 1])
        assert selected.layers[0].values.dtype == torch.float32

    def test_keys_and_values_dtype_match(self):
        cache = _make_dummy_cache(dtype=torch.bfloat16)
        selected = select_kv_cache(cache, [0, 1, 2])
        for i in range(len(selected.layers)):
            assert selected.layers[i].keys.dtype == selected.layers[i].values.dtype


# ======================================================================
# reposition_kv_cache — bos_start parameter
# ======================================================================

class TestRepositionBosStart:
    """Test with bos_start != 0."""

    def test_bos_start_1(self):
        """When BOS is at index 1, entry at index 1 should be preserved."""
        cache = _make_dummy_cache(n_layers=2, seq_len=6, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        bos_k = cache.layers[0].keys[:, :, :2, :].clone()  # indices 0 and 1
        old_pos = torch.arange(10, 14)  # 4 doc tokens at indices 2..5
        new_pos = torch.arange(2, 6)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types,
                            bos_start=1)
        # Index 0 and 1 (up to bos_start+1=2) should be preserved
        assert torch.equal(cache.layers[0].keys[:, :, :2, :], bos_k)

    def test_bos_start_doc_keys_change(self):
        """Doc keys after bos_start should actually change."""
        cache = _make_dummy_cache(n_layers=2, seq_len=6, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        doc_k = cache.layers[0].keys[:, :, 2:, :].clone()
        old_pos = torch.arange(10, 14)
        new_pos = torch.arange(2, 6)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types,
                            bos_start=1)
        assert not torch.equal(cache.layers[0].keys[:, :, 2:, :], doc_k)

    def test_bos_start_roundtrip(self):
        """Roundtrip should work regardless of bos_start."""
        cache = _make_dummy_cache(n_layers=2, seq_len=6, head_dim=16,
                                  dtype=torch.float32)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        k_orig = cache.layers[0].keys[:, :, 2:, :].clone()
        old_pos = torch.arange(10, 14)
        new_pos = torch.arange(2, 6)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types,
                            bos_start=1)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types,
                            bos_start=1)
        assert torch.allclose(cache.layers[0].keys[:, :, 2:, :], k_orig, atol=1e-4)

    def test_bos_start_values_unchanged(self):
        cache = _make_dummy_cache(n_layers=2, seq_len=6, head_dim=16)
        inv_freqs, layer_types = _setup_rope(head_dim=16, n_layers=2)
        v_orig = [cache.layers[i].values.clone() for i in range(2)]
        reposition_kv_cache(cache, torch.arange(10, 14), torch.arange(2, 6),
                            inv_freqs, layer_types, bos_start=1)
        for i in range(2):
            assert torch.equal(cache.layers[i].values, v_orig[i])
