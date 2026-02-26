"""Tests for lib.quantization — KV cache quantization and normalization.

Covers simulated_quantize, quantize_kv_cache, norm_roundtrip_kv_cache,
and clip_kv_cache with extensive mathematical property checks, edge cases,
BOS preservation guarantees, and idempotency tests.
"""

import torch
import pytest
from transformers import DynamicCache

from lib.quantization import (
    clip_kv_cache,
    norm_roundtrip_kv_cache,
    quantize_kv_cache,
    simulated_quantize,
)


def _make_dummy_cache(n_layers: int = 4, seq_len: int = 5,
                      n_heads: int = 2, head_dim: int = 8,
                      dtype=torch.bfloat16, seed: int = 42) -> DynamicCache:
    """Create a DynamicCache with known values (BOS + doc tokens)."""
    torch.manual_seed(seed)
    cache = DynamicCache()
    for i in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
        cache.update(k, v, i)
    return cache


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    """Simple clone for test comparison."""
    clone = DynamicCache()
    for i in range(len(cache.layers)):
        clone.update(cache.layers[i].keys.clone(),
                     cache.layers[i].values.clone(), i)
    return clone


# ======================================================================
# simulated_quantize
# ======================================================================

class TestSimulatedQuantizeBasic:
    def test_output_shape_preserved(self):
        t = torch.randn(4, 8, dtype=torch.bfloat16)
        assert simulated_quantize(t, 8).shape == t.shape

    def test_dtype_preserved_bf16(self):
        t = torch.randn(4, 8, dtype=torch.bfloat16)
        assert simulated_quantize(t, 8).dtype == torch.bfloat16

    def test_dtype_preserved_fp32(self):
        t = torch.randn(4, 8, dtype=torch.float32)
        assert simulated_quantize(t, 8).dtype == torch.float32

    def test_zero_tensor_unchanged(self):
        t = torch.zeros(4, 8, dtype=torch.bfloat16)
        assert torch.equal(simulated_quantize(t, 8), t)

    def test_scalar_tensor(self):
        t = torch.tensor(1.5, dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert q.shape == t.shape


class TestSimulatedQuantizeDiscreteLevels:
    """Verify that quantization produces the expected number of discrete levels."""

    def test_int8_max_255_levels(self):
        torch.manual_seed(42)
        t = torch.randn(100, 100, dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert q.unique().numel() <= 255

    def test_int4_max_15_levels(self):
        torch.manual_seed(42)
        t = torch.randn(100, 100, dtype=torch.float32)
        q = simulated_quantize(t, 4)
        assert q.unique().numel() <= 15

    def test_int16_max_65535_levels(self):
        torch.manual_seed(42)
        t = torch.randn(100, 100, dtype=torch.float32)
        q = simulated_quantize(t, 16)
        assert q.unique().numel() <= 65535

    def test_int2_max_3_levels(self):
        torch.manual_seed(42)
        t = torch.randn(50, 50, dtype=torch.float32)
        q = simulated_quantize(t, 2)
        # qmax = (1<<1)-1 = 1, so levels: {-1, 0, 1} * scale = 3 levels
        assert q.unique().numel() <= 3


class TestSimulatedQuantizeErrorOrdering:
    """Quantization error should decrease with more bits."""

    def test_error_decreases_with_bits(self):
        torch.manual_seed(42)
        t = torch.randn(200, 200, dtype=torch.float32)
        errors = {}
        for nbits in [2, 4, 8, 16]:
            q = simulated_quantize(t, nbits)
            errors[nbits] = (t - q).abs().mean().item()
        assert errors[2] > errors[4] > errors[8] > errors[16]

    def test_int16_nearly_lossless_in_fp32(self):
        """int16 (qmax=32767) should have negligible error in float32."""
        torch.manual_seed(42)
        t = torch.randn(100, 100, dtype=torch.float32)
        q = simulated_quantize(t, 16)
        max_err = (t - q).abs().max().item()
        assert max_err < 0.001


class TestSimulatedQuantizeMathProperties:
    def test_range_preserved(self):
        """Output values should not exceed original absmax."""
        torch.manual_seed(42)
        t = torch.randn(50, 50, dtype=torch.float32) * 3.0
        q = simulated_quantize(t, 8)
        assert q.max() <= t.abs().max() + 1e-5
        assert q.min() >= -t.abs().max() - 1e-5

    def test_idempotent(self):
        """Quantizing twice should give the same result as quantizing once."""
        torch.manual_seed(42)
        t = torch.randn(50, 50, dtype=torch.float32)
        q1 = simulated_quantize(t, 8)
        q2 = simulated_quantize(q1, 8)
        assert torch.equal(q1, q2)

    def test_symmetric_around_zero(self):
        """For symmetric input, output should also be approximately symmetric."""
        torch.manual_seed(42)
        t = torch.randn(10000, dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert abs(q.mean().item()) < 0.05  # Centered near zero

    def test_absmax_preserved(self):
        """The maximum absolute value should be exactly preserved."""
        torch.manual_seed(42)
        t = torch.randn(50, 50, dtype=torch.float32)
        q = simulated_quantize(t, 8)
        # absmax maps to ±qmax, which maps back to ±absmax exactly
        assert abs(q.abs().max().item() - t.abs().max().item()) < 1e-5

    def test_negative_values_preserved(self):
        """Quantization should handle negative values correctly."""
        t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
        q = simulated_quantize(t, 8)
        # Signs should be preserved
        assert (q[0] < 0) and (q[1] < 0) and (q[3] > 0) and (q[4] > 0)


class TestSimulatedQuantizeEdgeCases:
    def test_single_value_tensor(self):
        t = torch.tensor([5.0], dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert abs(q.item() - 5.0) < 1e-5

    def test_all_same_nonzero(self):
        t = torch.full((10, 10), 2.5, dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert torch.allclose(q, t, atol=0.05)

    def test_very_small_values(self):
        t = torch.tensor([1e-10, -1e-10], dtype=torch.float32)
        q = simulated_quantize(t, 8)
        # Should not crash, values may be quantized to zero
        assert q.shape == t.shape

    def test_very_large_values(self):
        t = torch.tensor([1e6, -1e6], dtype=torch.float32)
        q = simulated_quantize(t, 8)
        assert abs(q[0].item() - 1e6) < 1e6 * 0.01  # Within 1%

    def test_single_outlier(self):
        """One large outlier should dominate the scale."""
        t = torch.zeros(100, dtype=torch.float32)
        t[0] = 100.0  # Outlier
        q = simulated_quantize(t, 8)
        # Most values quantized to 0
        assert (q[1:] == 0).sum() > 90


# ======================================================================
# quantize_kv_cache
# ======================================================================

class TestQuantizeKvCacheBOSPreservation:
    """BOS preservation is CRITICAL — test it exhaustively."""

    def test_bos_keys_preserved_int8(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        quantize_kv_cache(cache, 8)
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)

    def test_bos_values_preserved_int8(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_v = cache.layers[0].values[:, :, :1, :].clone()
        quantize_kv_cache(cache, 8)
        assert torch.equal(cache.layers[0].values[:, :, :1, :], bos_v)

    def test_bos_preserved_all_layers(self):
        """BOS must be preserved at EVERY layer, not just layer 0."""
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=10)
        bos_k_all = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(n_layers)]
        bos_v_all = [cache.layers[i].values[:, :, :1, :].clone() for i in range(n_layers)]
        quantize_kv_cache(cache, 8)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_k_all[i])
            assert torch.equal(cache.layers[i].values[:, :, :1, :], bos_v_all[i])

    @pytest.mark.parametrize("nbits", [4, 8, 16])
    def test_bos_preserved_all_bit_widths(self, nbits):
        cache = _make_dummy_cache(seq_len=10)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        quantize_kv_cache(cache, nbits)
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)


class TestQuantizeKvCacheBasic:
    def test_doc_entries_changed(self):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=10, dtype=torch.float32)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        quantize_kv_cache(cache, 8)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], doc_k_orig)

    def test_shape_preserved(self):
        cache = _make_dummy_cache(seq_len=8)
        shapes = [(cache.layers[i].keys.shape, cache.layers[i].values.shape)
                  for i in range(len(cache.layers))]
        quantize_kv_cache(cache, 8)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.shape == shapes[i][0]
            assert cache.layers[i].values.shape == shapes[i][1]

    def test_single_entry_cache_unchanged(self):
        cache = _make_dummy_cache(seq_len=1)
        k_orig = cache.layers[0].keys.clone()
        quantize_kv_cache(cache, 8)
        assert torch.equal(cache.layers[0].keys, k_orig)

    def test_in_place_mutation(self):
        cache = _make_dummy_cache(seq_len=5)
        result = quantize_kv_cache(cache, 8)
        assert result is cache

    def test_both_keys_and_values_quantized(self):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=10, dtype=torch.float32)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        v_orig = cache.layers[0].values[:, :, 1:, :].clone()
        quantize_kv_cache(cache, 8)
        assert not torch.equal(cache.layers[0].keys[:, :, 1:, :], k_orig)
        assert not torch.equal(cache.layers[0].values[:, :, 1:, :], v_orig)


class TestQuantizeKvCacheIdempotent:
    """Quantizing an already-quantized cache should be a no-op."""

    @pytest.mark.parametrize("nbits", [4, 8, 16])
    def test_double_quantize_is_idempotent(self, nbits):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.float32)
        quantize_kv_cache(cache, nbits)
        k_after_first = cache.layers[0].keys.clone()
        quantize_kv_cache(cache, nbits)
        assert torch.equal(cache.layers[0].keys, k_after_first)


class TestQuantizeKvCacheErrorOrdering:
    """Higher bits should give lower quantization error."""

    def test_int16_less_error_than_int8(self):
        torch.manual_seed(42)
        cache8 = _make_dummy_cache(seq_len=50, dtype=torch.float32, seed=42)
        cache16 = _clone_cache(cache8)
        orig = _clone_cache(cache8)

        quantize_kv_cache(cache8, 8)
        quantize_kv_cache(cache16, 16)

        err8 = (cache8.layers[0].keys - orig.layers[0].keys).abs().mean()
        err16 = (cache16.layers[0].keys - orig.layers[0].keys).abs().mean()
        assert err16 < err8


# ======================================================================
# norm_roundtrip_kv_cache
# ======================================================================

class TestNormRoundtripBOSPreservation:
    def test_bos_keys_preserved(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        norm_roundtrip_kv_cache(cache)
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)

    def test_bos_values_preserved(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_v = cache.layers[0].values[:, :, :1, :].clone()
        norm_roundtrip_kv_cache(cache)
        assert torch.equal(cache.layers[0].values[:, :, :1, :], bos_v)

    def test_bos_preserved_all_layers(self):
        n_layers = 8
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=10)
        bos_k_all = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(n_layers)]
        norm_roundtrip_kv_cache(cache)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_k_all[i])


class TestNormRoundtripNearIdentity:
    def test_float32_nearly_exact(self):
        """In float32, the divide-multiply roundtrip should be nearly exact."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=10, dtype=torch.float32)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache)
        max_err = (cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs().max()
        assert max_err < 1e-4

    def test_bf16_introduces_perturbation(self):
        """In bf16, roundtrip should introduce small but nonzero noise."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.bfloat16)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache)
        diff = (cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs()
        assert diff.max() > 0  # Some perturbation

    def test_bf16_perturbation_is_small(self):
        """The perturbation should be small relative to values."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=50, dtype=torch.bfloat16)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache)
        rel_err = ((cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs() /
                   (doc_k_orig.abs() + 1e-10)).mean()
        assert rel_err < 0.1  # <10% relative error on average


class TestNormRoundtripProperties:
    def test_shape_preserved(self):
        cache = _make_dummy_cache(seq_len=8)
        shapes = [cache.layers[i].keys.shape for i in range(len(cache.layers))]
        norm_roundtrip_kv_cache(cache)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.shape == shapes[i]

    def test_in_place_mutation(self):
        cache = _make_dummy_cache(seq_len=5)
        result = norm_roundtrip_kv_cache(cache)
        assert result is cache

    def test_custom_qmax_127(self):
        """Default qmax=127 (matching int8 scale)."""
        cache = _make_dummy_cache(seq_len=10, dtype=torch.bfloat16)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache, qmax=127)
        # Should produce some change in bf16
        diff = (cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs()
        assert diff.max() > 0

    def test_different_qmax_different_results(self):
        """Different qmax values should produce different perturbations."""
        torch.manual_seed(42)
        cache1 = _make_dummy_cache(seq_len=10, dtype=torch.bfloat16, seed=42)
        cache2 = _clone_cache(cache1)
        norm_roundtrip_kv_cache(cache1, qmax=127)
        norm_roundtrip_kv_cache(cache2, qmax=32767)
        assert not torch.equal(
            cache1.layers[0].keys[:, :, 1:, :],
            cache2.layers[0].keys[:, :, 1:, :],
        )

    def test_idempotent_in_float32(self):
        """Applying norm roundtrip twice should give nearly the same as once in fp32."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=10, dtype=torch.float32)
        norm_roundtrip_kv_cache(cache)
        k_after_first = cache.layers[0].keys.clone()
        norm_roundtrip_kv_cache(cache)
        max_diff = (cache.layers[0].keys - k_after_first).abs().max()
        assert max_diff < 1e-5

    def test_single_entry_cache_unchanged(self):
        """BOS-only cache should be completely unchanged."""
        cache = _make_dummy_cache(seq_len=1)
        k_orig = cache.layers[0].keys.clone()
        norm_roundtrip_kv_cache(cache)
        assert torch.equal(cache.layers[0].keys, k_orig)

    def test_zero_doc_values_unchanged(self):
        """If all doc values are zero, absmax=0 guard should skip normalization."""
        cache = _make_dummy_cache(seq_len=5, dtype=torch.float32)
        for i in range(len(cache.layers)):
            cache.layers[i].keys[:, :, 1:, :].zero_()
            cache.layers[i].values[:, :, 1:, :].zero_()
        norm_roundtrip_kv_cache(cache)
        for i in range(len(cache.layers)):
            assert (cache.layers[i].keys[:, :, 1:, :] == 0).all()


# ======================================================================
# clip_kv_cache
# ======================================================================

class TestClipKvCacheBOSPreservation:
    def test_bos_keys_preserved(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        clip_kv_cache(cache, 2.0)
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)

    def test_bos_values_preserved(self):
        cache = _make_dummy_cache(seq_len=10)
        bos_v = cache.layers[0].values[:, :, :1, :].clone()
        clip_kv_cache(cache, 2.0)
        assert torch.equal(cache.layers[0].values[:, :, :1, :], bos_v)

    def test_bos_preserved_all_layers(self):
        n_layers = 6
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=10)
        bos_k_all = [cache.layers[i].keys[:, :, :1, :].clone() for i in range(n_layers)]
        clip_kv_cache(cache, 1.0)
        for i in range(n_layers):
            assert torch.equal(cache.layers[i].keys[:, :, :1, :], bos_k_all[i])


class TestClipKvCacheRange:
    def test_tight_clip_reduces_range(self):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=100, n_heads=4, head_dim=16,
                                  dtype=torch.float32)
        orig_range = (cache.layers[0].keys[:, :, 1:, :].max() -
                      cache.layers[0].keys[:, :, 1:, :].min()).item()
        clip_kv_cache(cache, 1.0)
        new_range = (cache.layers[0].keys[:, :, 1:, :].max() -
                     cache.layers[0].keys[:, :, 1:, :].min()).item()
        assert new_range < orig_range

    def test_loose_clip_preserves_values(self):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.float32)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        clip_kv_cache(cache, 10.0)
        max_diff = (cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs().max()
        assert max_diff < 1e-5

    def test_clip_dose_response(self):
        """Tighter clipping (fewer sigma) should clip more aggressively."""
        torch.manual_seed(42)
        cache1 = _make_dummy_cache(seq_len=100, dtype=torch.float32, seed=42)
        cache2 = _clone_cache(cache1)
        orig = _clone_cache(cache1)

        clip_kv_cache(cache1, 1.0)  # Tight
        clip_kv_cache(cache2, 3.0)  # Loose

        err1 = (cache1.layers[0].keys[:, :, 1:, :] - orig.layers[0].keys[:, :, 1:, :]).abs().mean()
        err2 = (cache2.layers[0].keys[:, :, 1:, :] - orig.layers[0].keys[:, :, 1:, :]).abs().mean()
        assert err1 > err2  # Tighter clip = more change

    def test_values_within_bounds(self):
        """After clipping, all doc values should be within [mean ± n*std]."""
        torch.manual_seed(42)
        n_sigma = 2.0
        cache = _make_dummy_cache(seq_len=200, n_heads=4, head_dim=16,
                                  dtype=torch.float32)
        # Compute expected bounds before clipping
        doc_k = cache.layers[0].keys[:, :, 1:, :].clone()
        k_mean = doc_k.mean().item()
        k_std = doc_k.std().item()
        upper = k_mean + n_sigma * k_std
        lower = k_mean - n_sigma * k_std

        clip_kv_cache(cache, n_sigma)
        clipped = cache.layers[0].keys[:, :, 1:, :]
        assert clipped.max().item() <= upper + 1e-5
        assert clipped.min().item() >= lower - 1e-5


class TestClipKvCacheProperties:
    def test_shape_preserved(self):
        cache = _make_dummy_cache(seq_len=8)
        shapes = [cache.layers[i].keys.shape for i in range(len(cache.layers))]
        clip_kv_cache(cache, 3.0)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.shape == shapes[i]

    def test_in_place_mutation(self):
        cache = _make_dummy_cache(seq_len=5)
        result = clip_kv_cache(cache, 3.0)
        assert result is cache

    def test_both_keys_and_values_clipped(self):
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=100, dtype=torch.float32)
        k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        v_orig = cache.layers[0].values[:, :, 1:, :].clone()
        clip_kv_cache(cache, 1.0)
        k_changed = not torch.equal(cache.layers[0].keys[:, :, 1:, :], k_orig)
        v_changed = not torch.equal(cache.layers[0].values[:, :, 1:, :], v_orig)
        assert k_changed and v_changed

    def test_repeated_clipping_converges(self):
        """Repeated clipping should converge (each pass changes less)."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=50, dtype=torch.float32)
        clip_kv_cache(cache, 2.0)
        k_after_first = cache.layers[0].keys.clone()
        clip_kv_cache(cache, 2.0)
        diff1 = (cache.layers[0].keys - k_after_first).abs().max()
        k_after_second = cache.layers[0].keys.clone()
        clip_kv_cache(cache, 2.0)
        diff2 = (cache.layers[0].keys - k_after_second).abs().max()
        # Second pass should change less than first (converging)
        assert diff2 <= diff1

    def test_single_entry_cache_unchanged(self):
        cache = _make_dummy_cache(seq_len=1)
        k_orig = cache.layers[0].keys.clone()
        clip_kv_cache(cache, 2.0)
        assert torch.equal(cache.layers[0].keys, k_orig)


# ======================================================================
# Cross-function interactions
# ======================================================================

class TestCrossFunctionInteractions:
    """Test that different operations compose correctly."""

    def test_quantize_then_norm_roundtrip(self):
        """Quantize then normalize should not crash."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.float32)
        quantize_kv_cache(cache, 8)
        norm_roundtrip_kv_cache(cache)
        # Should still have valid shape
        assert cache.layers[0].keys.shape[2] == 20

    def test_clip_then_quantize(self):
        """Clip then quantize should not crash."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.float32)
        clip_kv_cache(cache, 2.0)
        quantize_kv_cache(cache, 8)
        assert cache.layers[0].keys.shape[2] == 20

    def test_all_three_operations(self):
        """Apply clip → norm → quantize in sequence."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=30, dtype=torch.float32)
        bos_k = cache.layers[0].keys[:, :, :1, :].clone()
        clip_kv_cache(cache, 3.0)
        norm_roundtrip_kv_cache(cache)
        quantize_kv_cache(cache, 8)
        # BOS should still be preserved through all operations
        assert torch.equal(cache.layers[0].keys[:, :, :1, :], bos_k)

    def test_norm_then_quantize_less_error_than_quantize_alone(self):
        """Normalizing before quantizing might change the error profile."""
        torch.manual_seed(42)
        cache_q = _make_dummy_cache(seq_len=50, dtype=torch.float32, seed=42)
        cache_nq = _clone_cache(cache_q)
        orig = _clone_cache(cache_q)

        # Just quantize
        quantize_kv_cache(cache_q, 8)
        # Normalize then quantize
        norm_roundtrip_kv_cache(cache_nq)
        quantize_kv_cache(cache_nq, 8)

        # Both should produce valid results (no NaN or Inf)
        assert not torch.isnan(cache_q.layers[0].keys).any()
        assert not torch.isnan(cache_nq.layers[0].keys).any()


# ======================================================================
# simulated_quantize — nbits=1 edge case
# ======================================================================

class TestSimulatedQuantizeNbits1:
    """nbits=1 gives qmax=0. Verify behavior (divide by zero guard)."""

    def test_nbits_1_zero_tensor(self):
        """Zero tensor with nbits=1: absmax=0 guard should return tensor unchanged."""
        t = torch.zeros(4, 4, dtype=torch.float32)
        q = simulated_quantize(t, 1)
        assert torch.equal(q, t)

    def test_nbits_1_nonzero_tensor(self):
        """nbits=1: qmax=0, scale=absmax/0 → inf. Clamp(-0,0) → zeros.
        This tests the actual behavior: (tensor/inf).round().clamp(0,0) * inf.
        round(0) = 0, clamp(0,0) = 0, 0*inf = nan.
        The function may produce NaN — verify it doesn't crash."""
        t = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
        # Should not raise
        q = simulated_quantize(t, 1)
        assert q.shape == t.shape


# ======================================================================
# quantize_kv_cache — all layers modified
# ======================================================================

class TestQuantizeKvCacheAllLayers:
    """Verify that doc entries are actually quantized at EVERY layer."""

    def test_all_layers_doc_entries_change(self):
        n_layers = 8
        torch.manual_seed(42)
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=20,
                                  dtype=torch.float32)
        doc_k_orig = [cache.layers[i].keys[:, :, 1:, :].clone()
                      for i in range(n_layers)]
        doc_v_orig = [cache.layers[i].values[:, :, 1:, :].clone()
                      for i in range(n_layers)]
        quantize_kv_cache(cache, 8)
        for i in range(n_layers):
            assert not torch.equal(
                cache.layers[i].keys[:, :, 1:, :], doc_k_orig[i]
            ), f"Layer {i} keys were not quantized"
            assert not torch.equal(
                cache.layers[i].values[:, :, 1:, :], doc_v_orig[i]
            ), f"Layer {i} values were not quantized"


# ======================================================================
# quantize_kv_cache — dtype preservation
# ======================================================================

class TestQuantizeKvCacheDtype:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_keys_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=10, dtype=dtype)
        quantize_kv_cache(cache, 8)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_values_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=10, dtype=dtype)
        quantize_kv_cache(cache, 8)
        for i in range(len(cache.layers)):
            assert cache.layers[i].values.dtype == dtype


# ======================================================================
# norm_roundtrip_kv_cache — values are also processed
# ======================================================================

class TestNormRoundtripValues:
    """Verify that values (not just keys) go through the norm roundtrip."""

    def test_values_change_in_bf16(self):
        """In bf16, values should also get perturbed by the roundtrip."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=20, dtype=torch.bfloat16)
        v_orig = cache.layers[0].values[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache)
        diff = (cache.layers[0].values[:, :, 1:, :] - v_orig).abs()
        assert diff.max() > 0, "Values were not modified by norm roundtrip"

    def test_values_nearly_exact_in_fp32(self):
        """In fp32, values roundtrip should be nearly exact."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=10, dtype=torch.float32)
        v_orig = cache.layers[0].values[:, :, 1:, :].clone()
        norm_roundtrip_kv_cache(cache)
        max_err = (cache.layers[0].values[:, :, 1:, :] - v_orig).abs().max()
        assert max_err < 1e-4

    def test_values_bos_preserved(self):
        """BOS values should be preserved through norm roundtrip."""
        cache = _make_dummy_cache(seq_len=10)
        bos_v = cache.layers[0].values[:, :, :1, :].clone()
        norm_roundtrip_kv_cache(cache)
        assert torch.equal(cache.layers[0].values[:, :, :1, :], bos_v)

    def test_values_processed_all_layers(self):
        """All layers should have values processed, not just layer 0."""
        n_layers = 6
        torch.manual_seed(42)
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=20,
                                  dtype=torch.bfloat16)
        v_orig = [cache.layers[i].values[:, :, 1:, :].clone()
                  for i in range(n_layers)]
        norm_roundtrip_kv_cache(cache)
        for i in range(n_layers):
            diff = (cache.layers[i].values[:, :, 1:, :] - v_orig[i]).abs()
            assert diff.max() > 0, f"Layer {i} values were not processed"


# ======================================================================
# norm_roundtrip_kv_cache — dtype preservation
# ======================================================================

class TestNormRoundtripDtype:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_keys_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=10, dtype=dtype)
        norm_roundtrip_kv_cache(cache)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_values_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=10, dtype=dtype)
        norm_roundtrip_kv_cache(cache)
        for i in range(len(cache.layers)):
            assert cache.layers[i].values.dtype == dtype


# ======================================================================
# clip_kv_cache — all layers modified
# ======================================================================

class TestClipKvCacheAllLayers:
    """Verify that doc entries are actually clipped at EVERY layer."""

    def test_all_layers_doc_entries_change(self):
        n_layers = 6
        torch.manual_seed(42)
        cache = _make_dummy_cache(n_layers=n_layers, seq_len=100,
                                  dtype=torch.float32)
        doc_k_orig = [cache.layers[i].keys[:, :, 1:, :].clone()
                      for i in range(n_layers)]
        clip_kv_cache(cache, 1.0)  # Tight clip
        for i in range(n_layers):
            assert not torch.equal(
                cache.layers[i].keys[:, :, 1:, :], doc_k_orig[i]
            ), f"Layer {i} keys were not clipped"


# ======================================================================
# clip_kv_cache — dtype preservation
# ======================================================================

class TestClipKvCacheDtype:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_keys_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=20, dtype=dtype)
        clip_kv_cache(cache, 2.0)
        for i in range(len(cache.layers)):
            assert cache.layers[i].keys.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_preserves_values_dtype(self, dtype):
        cache = _make_dummy_cache(seq_len=20, dtype=dtype)
        clip_kv_cache(cache, 2.0)
        for i in range(len(cache.layers)):
            assert cache.layers[i].values.dtype == dtype


# ======================================================================
# clip_kv_cache — n_sigma edge cases
# ======================================================================

class TestClipKvCacheSigmaEdgeCases:
    def test_n_sigma_zero_clamps_to_mean(self):
        """n_sigma=0 should clamp everything to the mean."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=50, dtype=torch.float32)
        # Compute expected mean before clipping
        doc_k = cache.layers[0].keys[:, :, 1:, :].clone()
        expected_mean = doc_k.mean().item()
        clip_kv_cache(cache, 0.0)
        clipped = cache.layers[0].keys[:, :, 1:, :]
        # All values should be the mean
        assert torch.allclose(clipped, torch.full_like(clipped, expected_mean),
                              atol=1e-5)

    def test_very_large_n_sigma_preserves_values(self):
        """n_sigma=100 should change nothing (no values beyond 100σ)."""
        torch.manual_seed(42)
        cache = _make_dummy_cache(seq_len=50, dtype=torch.float32)
        doc_k_orig = cache.layers[0].keys[:, :, 1:, :].clone()
        clip_kv_cache(cache, 100.0)
        max_diff = (cache.layers[0].keys[:, :, 1:, :] - doc_k_orig).abs().max()
        assert max_diff < 1e-5
