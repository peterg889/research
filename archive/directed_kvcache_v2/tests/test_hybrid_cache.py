"""Thorough tests for all hybrid/manipulation cache functions.

Tests cover: roundtrip/identity, swap-and-swap-back, independence/no-aliasing,
shape/layer consistency, value correctness, cross-doc specific, edge cases,
and composition tests.
"""

import sys
import os

import torch
import pytest
from transformers import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.kv_cache import (
    build_hybrid_cache,
    replace_values_at_layers,
    interpolate_values,
    replace_values_at_positions,
    build_cross_doc_cache,
    deepcopy_cache,
    _get_cache_keys,
    _get_cache_values,
    _set_cache_keys,
    _set_cache_values,
    _ensure_dynamic_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def caches_equal(a, b, atol=0.0):
    """Check if two caches are element-wise equal (or close)."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        ka, kb = _get_cache_keys(a, i), _get_cache_keys(b, i)
        va, vb = _get_cache_values(a, i), _get_cache_values(b, i)
        if ka.shape != kb.shape or va.shape != vb.shape:
            return False
        if atol == 0.0:
            if not torch.equal(ka, kb) or not torch.equal(va, vb):
                return False
        else:
            if not torch.allclose(ka, kb, atol=atol) or not torch.allclose(va, vb, atol=atol):
                return False
    return True


def make_deterministic_cache(n_layers=4, seq_len=10, n_heads=4, head_dim=16, seed=0):
    """Create a cache with deterministic random values for reproducible tests."""
    g = torch.Generator().manual_seed(seed)
    cache = DynamicCache()
    for layer_idx in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, generator=g)
        v = torch.randn(1, n_heads, seq_len, head_dim, generator=g)
        cache.update(k, v, layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Roundtrip / Identity Tests
# ---------------------------------------------------------------------------

class TestRoundtripIdentity:
    """Tests where the operation should produce a known identity result."""

    def test_hybrid_same_cache_is_identity(self, make_cache):
        """hybrid(A_keys, A_values) == deepcopy(A)"""
        a = make_cache(num_layers=4, seq_len=10)
        result = build_hybrid_cache(a, a)
        expected = deepcopy_cache(a)
        assert caches_equal(result, expected)

    def test_replace_all_layers_equals_hybrid(self, make_cache):
        """replace_values_at_layers(A, B, all_layers) == build_hybrid_cache(A, B)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        all_layers = list(range(4))
        result = replace_values_at_layers(a, b, all_layers)
        expected = build_hybrid_cache(a, b)
        assert caches_equal(result, expected)

    def test_replace_empty_layers_is_identity(self, make_cache):
        """replace_values_at_layers(A, B, []) == deepcopy(A) (both keys AND values)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        result = replace_values_at_layers(a, b, [])
        expected = deepcopy_cache(a)
        assert caches_equal(result, expected)

    def test_replace_full_position_range_equals_hybrid(self, make_cache):
        """replace_values_at_positions(A, B, 0, seq_len) == build_hybrid_cache(A, B)"""
        seq_len = 10
        a = make_cache(num_layers=4, seq_len=seq_len)
        b = make_cache(num_layers=4, seq_len=seq_len)
        result = replace_values_at_positions(a, b, 0, seq_len)
        expected = build_hybrid_cache(a, b)
        assert caches_equal(result, expected)

    def test_replace_empty_position_range_is_identity(self, make_cache):
        """replace_values_at_positions(A, B, 5, 5) == deepcopy(A)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        result = replace_values_at_positions(a, b, 5, 5)
        expected = deepcopy_cache(a)
        assert caches_equal(result, expected)

    def test_interpolate_alpha_zero_gives_bare(self, make_cache):
        """interpolate_values(A, B, 0.0) == build_hybrid_cache(A_keys, A_values)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        result = interpolate_values(a, b, 0.0)
        expected = build_hybrid_cache(a, a)
        assert caches_equal(result, expected)

    def test_interpolate_alpha_one_gives_primed(self, make_cache):
        """interpolate_values(A, B, 1.0) == build_hybrid_cache(A_keys, B_values)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        result = interpolate_values(a, b, 1.0)
        expected = build_hybrid_cache(a, b)
        assert caches_equal(result, expected)

    def test_interpolate_same_cache_is_identity(self, make_cache):
        """interpolate_values(A, A, 0.5) == deepcopy(A)"""
        a = make_cache(num_layers=4, seq_len=10)
        result = interpolate_values(a, a, 0.5)
        expected = deepcopy_cache(a)
        assert caches_equal(result, expected)


# ---------------------------------------------------------------------------
# Swap-and-Swap-Back Tests
# ---------------------------------------------------------------------------

class TestSwapAndSwapBack:
    """Tests verifying operations are reversible."""

    def test_hybrid_swap_back_recovers_original(self, make_cache):
        """Swap values A->B then B->A recovers original."""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        # First swap: keys from A, values from B
        hybrid = build_hybrid_cache(a, b)
        # Swap back: keys from hybrid (=A keys), values from A
        recovered = build_hybrid_cache(hybrid, a)
        expected = deepcopy_cache(a)
        assert caches_equal(recovered, expected)

    def test_layer_replacement_is_reversible(self, make_cache):
        """replace(replace(A, B, [1,3]), A, [1,3]) == deepcopy(A)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        step1 = replace_values_at_layers(a, b, [1, 3])
        step2 = replace_values_at_layers(step1, a, [1, 3])
        expected = deepcopy_cache(a)
        assert caches_equal(step2, expected)

    def test_positional_replacement_is_reversible(self, make_cache):
        """replace_pos(replace_pos(A, B, 2, 5), A, 2, 5) == deepcopy(A)"""
        a = make_cache(num_layers=4, seq_len=10)
        b = make_cache(num_layers=4, seq_len=10)
        step1 = replace_values_at_positions(a, b, 2, 5)
        step2 = replace_values_at_positions(step1, a, 2, 5)
        expected = deepcopy_cache(a)
        assert caches_equal(step2, expected)


# ---------------------------------------------------------------------------
# Independence / No-Aliasing Tests
# ---------------------------------------------------------------------------

class TestIndependence:
    """Tests that modifying returned caches doesn't modify sources and vice versa."""

    def _modify_cache(self, cache):
        """Mutate a cache in-place."""
        k = _get_cache_keys(cache, 0)
        k.fill_(999.0)
        v = _get_cache_values(cache, 0)
        v.fill_(999.0)

    def test_hybrid_result_independent_of_sources(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        a_copy = deepcopy_cache(a)
        b_copy = deepcopy_cache(b)
        result = build_hybrid_cache(a, b)
        self._modify_cache(result)
        assert caches_equal(a, a_copy)
        assert caches_equal(b, b_copy)

    def test_hybrid_source_modification_doesnt_affect_result(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        result = build_hybrid_cache(a, b)
        result_copy = deepcopy_cache(result)
        self._modify_cache(a)
        self._modify_cache(b)
        assert caches_equal(result, result_copy)

    def test_layer_replace_result_independent(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        a_copy = deepcopy_cache(a)
        result = replace_values_at_layers(a, b, [0])
        self._modify_cache(result)
        assert caches_equal(a, a_copy)

    def test_position_replace_result_independent(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        a_copy = deepcopy_cache(a)
        result = replace_values_at_positions(a, b, 1, 3)
        self._modify_cache(result)
        assert caches_equal(a, a_copy)

    def test_interpolate_result_independent(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        a_copy = deepcopy_cache(a)
        result = interpolate_values(a, b, 0.5)
        self._modify_cache(result)
        assert caches_equal(a, a_copy)

    def test_cross_doc_result_independent(self, make_cache):
        a = make_cache(num_layers=2, seq_len=6)  # BOS + 5 doc tokens
        b = make_cache(num_layers=2, seq_len=6)
        a_copy = deepcopy_cache(a)
        result = build_cross_doc_cache(a, b, target_doc_len=5)
        self._modify_cache(result)
        assert caches_equal(a, a_copy)

    def test_interpolate_source_modification_doesnt_affect_result(self, make_cache):
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        result = interpolate_values(a, b, 0.5)
        result_copy = deepcopy_cache(result)
        self._modify_cache(a)
        self._modify_cache(b)
        assert caches_equal(result, result_copy)


# ---------------------------------------------------------------------------
# Shape / Layer Consistency Tests
# ---------------------------------------------------------------------------

class TestShapeConsistency:
    """Tests that all functions preserve layer count, tensor shapes, dtype, device."""

    @pytest.fixture
    def cache_pair(self, make_cache):
        a = make_cache(num_layers=4, seq_len=8, num_heads=4, head_dim=16)
        b = make_cache(num_layers=4, seq_len=8, num_heads=4, head_dim=16)
        return a, b

    def _check_shape(self, result, n_layers, expected_shape):
        assert len(result) == n_layers
        for i in range(n_layers):
            assert _get_cache_keys(result, i).shape == expected_shape
            assert _get_cache_values(result, i).shape == expected_shape

    def _check_dtype_device(self, result, expected_dtype, expected_device):
        for i in range(len(result)):
            assert _get_cache_keys(result, i).dtype == expected_dtype
            assert _get_cache_values(result, i).dtype == expected_dtype
            assert _get_cache_keys(result, i).device == expected_device
            assert _get_cache_values(result, i).device == expected_device

    def test_hybrid_preserves_shape(self, cache_pair):
        a, b = cache_pair
        result = build_hybrid_cache(a, b)
        self._check_shape(result, 4, (1, 4, 8, 16))

    def test_layer_replace_preserves_shape(self, cache_pair):
        a, b = cache_pair
        result = replace_values_at_layers(a, b, [0, 2])
        self._check_shape(result, 4, (1, 4, 8, 16))

    def test_position_replace_preserves_shape(self, cache_pair):
        a, b = cache_pair
        result = replace_values_at_positions(a, b, 2, 6)
        self._check_shape(result, 4, (1, 4, 8, 16))

    def test_interpolate_preserves_shape(self, cache_pair):
        a, b = cache_pair
        result = interpolate_values(a, b, 0.5)
        self._check_shape(result, 4, (1, 4, 8, 16))

    def test_cross_doc_preserves_shape(self, make_cache):
        a = make_cache(num_layers=4, seq_len=11, num_heads=4, head_dim=16)  # BOS + 10
        b = make_cache(num_layers=4, seq_len=11, num_heads=4, head_dim=16)
        result = build_cross_doc_cache(a, b, target_doc_len=10)
        self._check_shape(result, 4, (1, 4, 11, 16))

    def test_all_preserve_dtype_device(self, cache_pair):
        a, b = cache_pair
        expected_dtype = _get_cache_keys(a, 0).dtype
        expected_device = _get_cache_keys(a, 0).device
        for fn_result in [
            build_hybrid_cache(a, b),
            replace_values_at_layers(a, b, [0]),
            replace_values_at_positions(a, b, 2, 5),
            interpolate_values(a, b, 0.5),
        ]:
            self._check_dtype_device(fn_result, expected_dtype, expected_device)


# ---------------------------------------------------------------------------
# Value Correctness Tests
# ---------------------------------------------------------------------------

class TestValueCorrectness:
    """Per-element checks that the right values end up in the right places."""

    def test_hybrid_keys_from_a_values_from_b(self, make_cache):
        """build_hybrid_cache: keys match A, values match B per element."""
        a = make_cache(num_layers=3, seq_len=8)
        b = make_cache(num_layers=3, seq_len=8)
        result = build_hybrid_cache(a, b)
        for i in range(3):
            assert torch.equal(_get_cache_keys(result, i), _get_cache_keys(a, i))
            assert torch.equal(_get_cache_values(result, i), _get_cache_values(b, i))

    def test_layer_replace_correct_layers(self, make_cache):
        """Replaced layers have source values, unreplaced have target values, ALL keys from target."""
        a = make_cache(num_layers=4, seq_len=8)
        b = make_cache(num_layers=4, seq_len=8)
        result = replace_values_at_layers(a, b, [1, 3])
        for i in range(4):
            assert torch.equal(_get_cache_keys(result, i), _get_cache_keys(a, i)), f"Keys at layer {i} should be from A"
            if i in [1, 3]:
                assert torch.equal(_get_cache_values(result, i), _get_cache_values(b, i)), f"Values at replaced layer {i} should be from B"
            else:
                assert torch.equal(_get_cache_values(result, i), _get_cache_values(a, i)), f"Values at unreplaced layer {i} should be from A"

    def test_position_replace_correct_positions(self, make_cache):
        """Positions outside range have bare values, inside have primed, keys all bare."""
        a = make_cache(num_layers=2, seq_len=10)
        b = make_cache(num_layers=2, seq_len=10)
        result = replace_values_at_positions(a, b, 3, 7)
        for i in range(2):
            assert torch.equal(_get_cache_keys(result, i), _get_cache_keys(a, i)), "Keys should all be from bare"
            v_result = _get_cache_values(result, i)
            v_a = _get_cache_values(a, i)
            v_b = _get_cache_values(b, i)
            # Outside range
            assert torch.equal(v_result[:, :, :3, :], v_a[:, :, :3, :])
            assert torch.equal(v_result[:, :, 7:, :], v_a[:, :, 7:, :])
            # Inside range
            assert torch.equal(v_result[:, :, 3:7, :], v_b[:, :, 3:7, :])

    def test_interpolate_half_is_mean(self, make_cache):
        """interpolate at alpha=0.5: values == mean(bare, primed) exactly."""
        a = make_cache(num_layers=2, seq_len=6)
        b = make_cache(num_layers=2, seq_len=6)
        result = interpolate_values(a, b, 0.5)
        for i in range(2):
            expected_v = 0.5 * _get_cache_values(b, i) + 0.5 * _get_cache_values(a, i)
            assert torch.equal(_get_cache_values(result, i), expected_v)

    def test_interpolate_keys_always_from_bare(self, make_cache):
        """Regardless of alpha, keys are from bare."""
        a = make_cache(num_layers=2, seq_len=6)
        b = make_cache(num_layers=2, seq_len=6)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = interpolate_values(a, b, alpha)
            for i in range(2):
                assert torch.equal(_get_cache_keys(result, i), _get_cache_keys(a, i))

    def test_cross_doc_bos_always_from_bare(self, make_cache):
        """BOS position always from bare."""
        a = make_cache(num_layers=2, seq_len=6)  # BOS + 5 doc tokens
        b = make_cache(num_layers=2, seq_len=6)
        result = build_cross_doc_cache(a, b, target_doc_len=5)
        for i in range(2):
            # BOS (position 0)
            assert torch.equal(
                _get_cache_values(result, i)[:, :, :1, :],
                _get_cache_values(a, i)[:, :, :1, :]
            )
            # Keys always from bare
            assert torch.equal(_get_cache_keys(result, i), _get_cache_keys(a, i))


# ---------------------------------------------------------------------------
# Cross-Document Specific Tests
# ---------------------------------------------------------------------------

class TestCrossDoc:
    """Tests specific to build_cross_doc_cache."""

    def test_matching_lengths(self, make_cache):
        """With matching lengths, doc positions all from donor (BOS from bare)."""
        doc_len = 8
        a = make_cache(num_layers=2, seq_len=1 + doc_len)  # BOS + 8
        b = make_cache(num_layers=2, seq_len=1 + doc_len)
        result = build_cross_doc_cache(a, b, target_doc_len=doc_len)
        for i in range(2):
            # BOS from bare
            assert torch.equal(
                _get_cache_values(result, i)[:, :, :1, :],
                _get_cache_values(a, i)[:, :, :1, :]
            )
            # Doc positions from donor
            assert torch.equal(
                _get_cache_values(result, i)[:, :, 1:, :],
                _get_cache_values(b, i)[:, :, 1:, :]
            )

    def test_donor_shorter(self, make_cache):
        """When donor is shorter: first N from donor, rest from bare."""
        target_doc_len = 8
        donor_doc_len = 5
        a = make_cache(num_layers=2, seq_len=1 + target_doc_len)  # BOS + 8
        b = make_cache(num_layers=2, seq_len=1 + donor_doc_len)   # BOS + 5
        result = build_cross_doc_cache(a, b, target_doc_len=target_doc_len)
        for i in range(2):
            v_result = _get_cache_values(result, i)
            # First 5 doc positions from donor
            assert torch.equal(
                v_result[:, :, 1:1 + donor_doc_len, :],
                _get_cache_values(b, i)[:, :, 1:1 + donor_doc_len, :]
            )
            # Remaining 3 doc positions from bare
            assert torch.equal(
                v_result[:, :, 1 + donor_doc_len:, :],
                _get_cache_values(a, i)[:, :, 1 + donor_doc_len:, :]
            )

    def test_donor_longer(self, make_cache):
        """When donor is longer: all target positions filled from donor (sliced)."""
        target_doc_len = 5
        donor_doc_len = 8
        a = make_cache(num_layers=2, seq_len=1 + target_doc_len)  # BOS + 5
        b = make_cache(num_layers=2, seq_len=1 + donor_doc_len)   # BOS + 8
        result = build_cross_doc_cache(a, b, target_doc_len=target_doc_len)
        for i in range(2):
            # All doc positions from donor (first target_doc_len of donor)
            assert torch.equal(
                _get_cache_values(result, i)[:, :, 1:, :],
                _get_cache_values(b, i)[:, :, 1:1 + target_doc_len, :]
            )

    def test_output_seq_len(self, make_cache):
        """Output seq_len always == 1 + target_doc_len."""
        for target_doc_len in [3, 5, 10]:
            a = make_cache(num_layers=2, seq_len=1 + target_doc_len)
            b = make_cache(num_layers=2, seq_len=1 + 7)  # Different donor length
            result = build_cross_doc_cache(a, b, target_doc_len=target_doc_len)
            assert _get_cache_keys(result, 0).shape[2] == 1 + target_doc_len


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_single_layer_cache(self, make_cache):
        """All functions work with n_layers=1."""
        a = make_cache(num_layers=1, seq_len=5)
        b = make_cache(num_layers=1, seq_len=5)

        assert len(build_hybrid_cache(a, b)) == 1
        assert len(replace_values_at_layers(a, b, [0])) == 1
        assert len(replace_values_at_positions(a, b, 1, 3)) == 1
        assert len(interpolate_values(a, b, 0.5)) == 1
        assert len(build_cross_doc_cache(a, b, target_doc_len=4)) == 1

    def test_single_position_cache_positional(self, make_cache):
        """Single-position cache (seq_len=1, just BOS) for positional."""
        a = make_cache(num_layers=2, seq_len=1)
        b = make_cache(num_layers=2, seq_len=1)
        result = replace_values_at_positions(a, b, 0, 1)
        assert _get_cache_keys(result, 0).shape[2] == 1

    def test_single_position_cache_cross_doc(self, make_cache):
        """Single-position cache for cross-doc (BOS only, target_doc_len=0)."""
        a = make_cache(num_layers=2, seq_len=1)  # Just BOS
        b = make_cache(num_layers=2, seq_len=1)
        result = build_cross_doc_cache(a, b, target_doc_len=0)
        assert _get_cache_keys(result, 0).shape[2] == 1

    def test_alpha_boundary_zero(self, make_cache):
        """alpha=0.0 for interpolation."""
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        result = interpolate_values(a, b, 0.0)
        for i in range(2):
            assert torch.equal(_get_cache_values(result, i), _get_cache_values(a, i))

    def test_alpha_boundary_one(self, make_cache):
        """alpha=1.0 for interpolation."""
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        result = interpolate_values(a, b, 1.0)
        for i in range(2):
            assert torch.equal(_get_cache_values(result, i), _get_cache_values(b, i))

    def test_layer_index_boundaries(self, make_cache):
        """Layer 0 only and last layer only."""
        a = make_cache(num_layers=4, seq_len=5)
        b = make_cache(num_layers=4, seq_len=5)

        # Only layer 0
        r0 = replace_values_at_layers(a, b, [0])
        assert torch.equal(_get_cache_values(r0, 0), _get_cache_values(b, 0))
        assert torch.equal(_get_cache_values(r0, 1), _get_cache_values(a, 1))

        # Only last layer
        r3 = replace_values_at_layers(a, b, [3])
        assert torch.equal(_get_cache_values(r3, 3), _get_cache_values(b, 3))
        assert torch.equal(_get_cache_values(r3, 2), _get_cache_values(a, 2))

    def test_duplicate_layer_indices(self, make_cache):
        """Duplicate indices in replace_values_at_layers should work (no double-apply)."""
        a = make_cache(num_layers=4, seq_len=5)
        b = make_cache(num_layers=4, seq_len=5)
        result = replace_values_at_layers(a, b, [1, 1, 1])
        # Layer 1 should have values from b
        assert torch.equal(_get_cache_values(result, 1), _get_cache_values(b, 1))
        # Other layers from a
        assert torch.equal(_get_cache_values(result, 0), _get_cache_values(a, 0))

    def test_interpolate_alpha_out_of_range(self, make_cache):
        """alpha outside [0, 1] should raise."""
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        with pytest.raises(AssertionError):
            interpolate_values(a, b, -0.1)
        with pytest.raises(AssertionError):
            interpolate_values(a, b, 1.1)

    def test_position_replace_invalid_range(self, make_cache):
        """Invalid position ranges should raise."""
        a = make_cache(num_layers=2, seq_len=5)
        b = make_cache(num_layers=2, seq_len=5)
        with pytest.raises(AssertionError):
            replace_values_at_positions(a, b, 3, 2)  # start > end
        with pytest.raises(AssertionError):
            replace_values_at_positions(a, b, 0, 6)  # end > seq_len
        with pytest.raises(AssertionError):
            replace_values_at_positions(a, b, -1, 3)  # negative start


# ---------------------------------------------------------------------------
# Composition Tests
# ---------------------------------------------------------------------------

class TestComposition:
    """Tests for composing multiple operations."""

    def test_interpolate_roundtrip_numerical_stability(self, make_cache):
        """interpolate(interpolate(A, B, 0.3), A, alpha) approximately recovers A.

        If we interpolate A,B at alpha=0.3 to get C = 0.3*B + 0.7*A,
        then interpolate C,A at alpha=1.0 gives A exactly.
        """
        a = make_cache(num_layers=2, seq_len=6)
        b = make_cache(num_layers=2, seq_len=6)
        c = interpolate_values(a, b, 0.3)
        # Interpolating C with A at alpha=1.0 means: 1.0*A + 0.0*C = A
        recovered = interpolate_values(c, a, 1.0)
        assert caches_equal(recovered, deepcopy_cache(a))

    def test_layer_groups_cover_all_equals_hybrid(self, make_cache):
        """Union of all layer groups == build_hybrid_cache."""
        n_layers = 8
        a = make_cache(num_layers=n_layers, seq_len=5)
        b = make_cache(num_layers=n_layers, seq_len=5)

        # Build in two groups
        group1 = list(range(0, 4))
        group2 = list(range(4, 8))
        r1 = replace_values_at_layers(a, b, group1)
        # Apply second group on top of first result
        r2 = replace_values_at_layers(r1, b, group2)

        expected = build_hybrid_cache(a, b)
        assert caches_equal(r2, expected)
