"""Tests for lib.cache — KV cache manipulation utilities.

Covers deep_copy_cache and make_prefix with extensive edge cases,
independence verification, and realistic usage patterns.
"""

import torch
import pytest
from transformers import DynamicCache

from lib.cache import deep_copy_cache, make_prefix, scramble_prefix


def _make_dummy_cache(n_layers: int = 4, n_heads: int = 2,
                      seq_len: int = 5, head_dim: int = 8,
                      dtype=torch.bfloat16) -> DynamicCache:
    """Create a simple DynamicCache with known values for testing."""
    cache = DynamicCache()
    for i in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        cache.update(k, v, i)
    return cache


# ======================================================================
# deep_copy_cache
# ======================================================================

class TestDeepCopyCacheBasic:
    def test_same_shape(self):
        cache = _make_dummy_cache()
        copy = deep_copy_cache(cache)
        assert len(copy.layers) == len(cache.layers)
        for i in range(len(cache.layers)):
            assert copy.layers[i].keys.shape == cache.layers[i].keys.shape
            assert copy.layers[i].values.shape == cache.layers[i].values.shape

    def test_same_values(self):
        cache = _make_dummy_cache()
        copy = deep_copy_cache(cache)
        for i in range(len(cache.layers)):
            assert torch.equal(copy.layers[i].keys, cache.layers[i].keys)
            assert torch.equal(copy.layers[i].values, cache.layers[i].values)

    def test_preserves_dtype_bf16(self):
        cache = _make_dummy_cache(dtype=torch.bfloat16)
        copy = deep_copy_cache(cache)
        assert copy.layers[0].keys.dtype == torch.bfloat16

    def test_preserves_dtype_fp32(self):
        cache = _make_dummy_cache(dtype=torch.float32)
        copy = deep_copy_cache(cache)
        assert copy.layers[0].keys.dtype == torch.float32

    def test_empty_cache(self):
        cache = DynamicCache()
        copy = deep_copy_cache(cache)
        assert len(copy.layers) == 0


class TestDeepCopyCacheIndependence:
    """The core guarantee: mutations to the copy must not affect the original."""

    def test_mutate_copy_keys(self):
        cache = _make_dummy_cache()
        original_k0 = cache.layers[0].keys.clone()
        copy = deep_copy_cache(cache)
        copy.layers[0].keys.fill_(999.0)
        assert torch.equal(cache.layers[0].keys, original_k0)

    def test_mutate_copy_values(self):
        cache = _make_dummy_cache()
        original_v0 = cache.layers[0].values.clone()
        copy = deep_copy_cache(cache)
        copy.layers[0].values.zero_()
        assert torch.equal(cache.layers[0].values, original_v0)

    def test_mutate_original_after_copy(self):
        """Mutating the original should not affect the copy."""
        cache = _make_dummy_cache()
        copy = deep_copy_cache(cache)
        copy_k0 = copy.layers[0].keys.clone()
        cache.layers[0].keys.fill_(-999.0)
        assert torch.equal(copy.layers[0].keys, copy_k0)

    def test_independence_all_layers(self):
        """Test independence for EVERY layer, not just layer 0."""
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers)
        originals = [cache.layers[i].keys.clone() for i in range(n_layers)]
        copy = deep_copy_cache(cache)
        for i in range(n_layers):
            copy.layers[i].keys.fill_(float(i))
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys, originals[i])

    def test_not_same_tensor_object(self):
        """Copy tensors should be different objects in memory."""
        cache = _make_dummy_cache()
        copy = deep_copy_cache(cache)
        assert cache.layers[0].keys.data_ptr() != copy.layers[0].keys.data_ptr()
        assert cache.layers[0].values.data_ptr() != copy.layers[0].values.data_ptr()

    def test_in_place_add_does_not_leak(self):
        """In-place operations on copy should not affect original."""
        cache = _make_dummy_cache(dtype=torch.float32)
        orig_sum = cache.layers[0].keys.sum().item()
        copy = deep_copy_cache(cache)
        copy.layers[0].keys.add_(100.0)
        assert abs(cache.layers[0].keys.sum().item() - orig_sum) < 1e-5


class TestDeepCopyCacheRealisticSizes:
    """Test with sizes matching real Gemma 3 12B-IT architecture."""

    def test_gemma3_like_cache(self):
        """48 layers, 16 heads, 256 head_dim, 100 seq entries."""
        cache = _make_dummy_cache(n_layers=48, n_heads=16,
                                  seq_len=100, head_dim=256)
        copy = deep_copy_cache(cache)
        assert len(copy.layers) == 48
        assert copy.layers[0].keys.shape == (1, 16, 100, 256)
        # Values should match
        assert torch.equal(copy.layers[47].values, cache.layers[47].values)

    def test_single_layer_cache(self):
        cache = _make_dummy_cache(n_layers=1, seq_len=3)
        copy = deep_copy_cache(cache)
        assert len(copy.layers) == 1
        assert torch.equal(copy.layers[0].keys, cache.layers[0].keys)

    def test_single_token_cache(self):
        """BOS-only cache (seq_len=1)."""
        cache = _make_dummy_cache(seq_len=1)
        copy = deep_copy_cache(cache)
        assert copy.layers[0].keys.shape[2] == 1

    def test_large_seq_len(self):
        cache = _make_dummy_cache(seq_len=1024, n_layers=2, n_heads=2, head_dim=8)
        copy = deep_copy_cache(cache)
        assert copy.layers[0].keys.shape[2] == 1024


# ======================================================================
# make_prefix
# ======================================================================

class TestMakePrefixBasic:
    def test_exact_length(self):
        assert make_prefix([1, 2, 3], 3) == [1, 2, 3]

    def test_truncate(self):
        assert make_prefix([1, 2, 3, 4, 5], 3) == [1, 2, 3]

    def test_pad_by_tiling(self):
        assert make_prefix([10, 20], 5) == [10, 20, 10, 20, 10]

    def test_pad_single_token(self):
        assert make_prefix([42], 4) == [42, 42, 42, 42]

    def test_longer_pad(self):
        assert make_prefix([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]

    def test_length_one_output(self):
        assert make_prefix([5, 6, 7], 1) == [5]

    def test_empty_input_length_zero(self):
        assert make_prefix([], 0) == []


class TestMakePrefixLengthGuarantee:
    """The most critical property: output is ALWAYS exactly the requested length."""

    @pytest.mark.parametrize("input_len", [1, 2, 3, 5, 10, 50])
    @pytest.mark.parametrize("target_len", [1, 2, 3, 7, 16, 32, 64, 128, 256])
    def test_exact_output_length(self, input_len, target_len):
        ids = list(range(1, input_len + 1))
        result = make_prefix(ids, target_len)
        assert len(result) == target_len

    def test_length_zero_with_nonempty_input(self):
        assert make_prefix([1, 2, 3], 0) == []

    def test_very_long_output(self):
        result = make_prefix([1, 2], 1000)
        assert len(result) == 1000


class TestMakePrefixContentCorrectness:
    def test_tiling_pattern(self):
        """Output should be a cyclic repetition of the input."""
        ids = [10, 20, 30]
        result = make_prefix(ids, 10)
        for i, val in enumerate(result):
            assert val == ids[i % len(ids)]

    def test_truncation_preserves_prefix(self):
        """Truncation should keep the FIRST L tokens."""
        ids = [10, 20, 30, 40, 50]
        for length in range(1, 6):
            result = make_prefix(ids, length)
            assert result == ids[:length]

    def test_single_token_repeated(self):
        result = make_prefix([99], 100)
        assert all(x == 99 for x in result)

    def test_two_tokens_alternating(self):
        result = make_prefix([1, 2], 6)
        assert result == [1, 2, 1, 2, 1, 2]

    def test_preserves_token_values(self):
        """Token IDs should be preserved exactly, not modified."""
        ids = [0, 65535, 32000, 1, 2]
        result = make_prefix(ids, 3)
        assert result == [0, 65535, 32000]


class TestMakePrefixInputTypes:
    def test_tuple_input(self):
        result = make_prefix((1, 2, 3), 5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_generator_like(self):
        result = make_prefix(range(5), 3)
        # range is a Sequence
        assert result == [0, 1, 2]

    def test_returns_new_list(self):
        """Output should be a new list, not a reference to the input."""
        ids = [1, 2, 3]
        result = make_prefix(ids, 3)
        result[0] = 999
        assert ids[0] == 1  # Original unchanged


class TestMakePrefixRealisticUsage:
    """Tests mimicking actual experiment usage patterns."""

    def test_comprehend_instruction_to_l64(self):
        """Typical: 15-token instruction padded to L=64."""
        instruction_ids = list(range(100, 115))  # 15 tokens
        result = make_prefix(instruction_ids, 64)
        assert len(result) == 64
        assert result[:15] == instruction_ids

    def test_long_instruction_truncated(self):
        """Very long instruction truncated to L=64."""
        long_ids = list(range(200))
        result = make_prefix(long_ids, 64)
        assert len(result) == 64
        assert result == list(range(64))

    @pytest.mark.parametrize("L", [32, 64, 128, 256])
    def test_standard_prefix_lengths(self, L):
        """All standard prefix lengths from the experiments."""
        ids = list(range(10, 25))  # 15-token instruction
        result = make_prefix(ids, L)
        assert len(result) == L


# ======================================================================
# scramble_prefix
# ======================================================================

class TestScramblePrefixBasic:
    def test_same_length(self):
        ids = [10, 20, 30, 40, 50]
        result = scramble_prefix(ids, seed=42)
        assert len(result) == len(ids)

    def test_same_elements(self):
        """Output must be a permutation (same multiset of tokens)."""
        ids = [10, 20, 30, 40, 50]
        result = scramble_prefix(ids, seed=42)
        assert sorted(result) == sorted(ids)

    def test_same_elements_with_duplicates(self):
        ids = [5, 5, 5, 10, 10]
        result = scramble_prefix(ids, seed=99)
        assert sorted(result) == sorted(ids)

    def test_single_element(self):
        result = scramble_prefix([42], seed=0)
        assert result == [42]

    def test_empty_input(self):
        result = scramble_prefix([], seed=0)
        assert result == []


class TestScramblePrefixDeterminism:
    def test_same_seed_same_output(self):
        ids = list(range(100))
        r1 = scramble_prefix(ids, seed=42)
        r2 = scramble_prefix(ids, seed=42)
        assert r1 == r2

    def test_different_seeds_different_output(self):
        ids = list(range(20))
        r1 = scramble_prefix(ids, seed=1)
        r2 = scramble_prefix(ids, seed=2)
        assert r1 != r2

    def test_seed_isolation(self):
        """Calling with one seed should not affect subsequent calls."""
        ids = [10, 20, 30, 40, 50, 60, 70, 80]
        expected = scramble_prefix(ids, seed=99)
        # Call with different seed in between
        scramble_prefix(ids, seed=0)
        scramble_prefix(ids, seed=1234)
        actual = scramble_prefix(ids, seed=99)
        assert actual == expected


class TestScramblePrefixInputSafety:
    def test_does_not_mutate_input(self):
        ids = [10, 20, 30, 40, 50]
        original = ids.copy()
        scramble_prefix(ids, seed=42)
        assert ids == original

    def test_returns_new_list(self):
        ids = [10, 20, 30]
        result = scramble_prefix(ids, seed=42)
        result[0] = 999
        assert ids[0] == 10

    def test_tuple_input(self):
        result = scramble_prefix((1, 2, 3, 4), seed=42)
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3, 4]

    def test_realistic_l64_prefix(self):
        """64-token prefix like the actual experiments."""
        ids = list(range(100, 164))
        result = scramble_prefix(ids, seed=42)
        assert len(result) == 64
        assert sorted(result) == sorted(ids)
        assert result != ids  # Very unlikely to be identical for 64 elements
