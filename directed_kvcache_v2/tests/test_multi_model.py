"""Tests for multi-model support (Gemma 3, Mistral, etc.).

Validates:
- _get_head_dim() uses explicit head_dim when available
- _get_rope_theta_for_layer() returns per-layer theta for Gemma 3
- correct_rope_positions* applies different corrections per layer type
- Backward compatibility with Mistral configs
"""

import types
import math
import torch
import pytest
from transformers import DynamicCache

from lib.kv_cache import (
    _get_head_dim,
    _get_text_config,
    _get_rope_theta,
    _get_rope_theta_for_layer,
    _build_rope_correction,
    correct_rope_positions,
    correct_rope_positions_with_bos,
    _get_cache_keys,
)
from lib.model_utils import _resolve_dtype
from lib.config import ExperimentConfig
from tests.conftest import get_keys


# =============================================================================
# TestGetHeadDim
# =============================================================================

class TestGetTextConfig:
    def test_composite_config(self, fake_gemma3_model):
        """Composite config resolves to text_config."""
        tc = _get_text_config(fake_gemma3_model.config)
        assert hasattr(tc, 'hidden_size')
        assert tc.hidden_size == 2560

    def test_plain_config(self, fake_model):
        """Plain config returns itself."""
        tc = _get_text_config(fake_model.config)
        assert tc is fake_model.config


class TestGetHeadDim:
    def test_gemma3_explicit(self, fake_gemma3_model):
        """Gemma 3 has explicit head_dim=256, not hidden_size/num_heads=320."""
        assert _get_head_dim(fake_gemma3_model.config) == 256

    def test_mistral_computed(self, fake_model):
        """Mistral has no explicit head_dim; falls back to 4096/32=128."""
        assert _get_head_dim(fake_model.config) == 128

    def test_no_explicit_field(self):
        """Config without head_dim attr falls back to computation."""
        cfg = types.SimpleNamespace(hidden_size=512, num_attention_heads=8)
        assert _get_head_dim(cfg) == 64

    def test_explicit_overrides_computation(self):
        """When head_dim is set, it takes precedence even if it disagrees."""
        cfg = types.SimpleNamespace(
            hidden_size=512, num_attention_heads=8, head_dim=100
        )
        assert _get_head_dim(cfg) == 100


# =============================================================================
# TestGetRopeThetaForLayer
# =============================================================================

class TestGetRopeThetaForLayer:
    def test_sliding_layer(self, fake_gemma3_model):
        """Sliding attention layers use theta=10000."""
        # Layer 0 is sliding_attention
        assert _get_rope_theta_for_layer(fake_gemma3_model.config, 0) == 10000.0

    def test_full_attention_layer(self, fake_gemma3_model):
        """Full attention layers use theta=1000000."""
        # Layer 5 is full_attention (first full layer in the pattern)
        assert _get_rope_theta_for_layer(fake_gemma3_model.config, 5) == 1000000.0

    def test_all_layers_correct(self, fake_gemma3_model):
        """Every layer maps to the correct theta based on its type."""
        config = fake_gemma3_model.config
        text_config = _get_text_config(config)
        for i, layer_type in enumerate(text_config.layer_types):
            theta = _get_rope_theta_for_layer(config, i)
            expected = 10000.0 if layer_type == 'sliding_attention' else 1000000.0
            assert theta == expected, f"Layer {i} ({layer_type}): expected {expected}, got {theta}"

    def test_no_layer_types_fallback(self, fake_model):
        """Mistral-like config without layer_types falls back to uniform theta."""
        theta = _get_rope_theta_for_layer(fake_model.config, 0)
        assert theta == 10000.0

    def test_no_layer_types_fallback_all_layers(self, fake_model):
        """All layers get the same theta for uniform-theta models."""
        for i in range(32):  # Mistral has 32 layers
            assert _get_rope_theta_for_layer(fake_model.config, i) == 10000.0


# =============================================================================
# TestBuildRopeCorrection
# =============================================================================

class TestBuildRopeCorrectionHeadDim256:
    def test_shape(self):
        """Correct shape (256,) for Gemma 3 head_dim."""
        cos, sin = _build_rope_correction(offset=5, head_dim=256, rope_theta=10000.0)
        assert cos.shape == (256,)
        assert sin.shape == (256,)

    def test_different_theta_different_result(self):
        """Different theta values produce different corrections."""
        cos_10k, sin_10k = _build_rope_correction(offset=5, head_dim=256, rope_theta=10000.0)
        cos_1m, sin_1m = _build_rope_correction(offset=5, head_dim=256, rope_theta=1000000.0)
        # They should NOT be equal (different base frequencies)
        assert not torch.allclose(cos_10k, cos_1m)
        assert not torch.allclose(sin_10k, sin_1m)

    def test_higher_theta_smaller_angles(self):
        """Higher theta → slower rotation → cos closer to 1, sin closer to 0."""
        cos_10k, sin_10k = _build_rope_correction(offset=5, head_dim=16, rope_theta=10000.0)
        cos_1m, sin_1m = _build_rope_correction(offset=5, head_dim=16, rope_theta=1000000.0)
        # Higher theta means smaller angles, so cos should be closer to 1
        assert (cos_1m - 1.0).abs().mean() < (cos_10k - 1.0).abs().mean()
        # And sin should be closer to 0
        assert sin_1m.abs().mean() < sin_10k.abs().mean()


# =============================================================================
# TestCorrectRopePerLayerTheta
# =============================================================================

class TestCorrectRopePerLayerTheta:
    def test_different_corrections_applied(self, small_fake_gemma3_model):
        """Sliding and full layers get different corrections.

        Layer 2 is full_attention (theta=1M), others are sliding (theta=10k).
        After correction, layer 2 keys should differ from sliding layers.
        """
        head_dim = 16
        seq_len = 5
        num_layers = 4
        offset = 10

        # Build cache with identical keys across all layers
        cache = DynamicCache()
        identical_keys = torch.randn(1, 2, seq_len, head_dim)
        for i in range(num_layers):
            cache.update(identical_keys.clone(), torch.zeros(1, 2, seq_len, head_dim), i)

        correct_rope_positions(cache, offset=offset, model=small_fake_gemma3_model)

        # Sliding layers (0, 1, 3) should all be equal to each other
        keys_0 = get_keys(cache, 0)
        keys_1 = get_keys(cache, 1)
        keys_3 = get_keys(cache, 3)
        torch.testing.assert_close(keys_0, keys_1)
        torch.testing.assert_close(keys_0, keys_3)

        # Full attention layer (2) should differ from sliding layers
        keys_2 = get_keys(cache, 2)
        assert not torch.allclose(keys_0, keys_2, atol=1e-4)

    def test_bos_preserved_gemma3(self, small_fake_gemma3_model):
        """BOS position untouched in correct_rope_positions_with_bos for Gemma 3."""
        head_dim = 16
        seq_len = 6  # 1 BOS + 5 doc tokens
        num_layers = 4

        cache = DynamicCache()
        for i in range(num_layers):
            keys = torch.randn(1, 2, seq_len, head_dim)
            cache.update(keys.clone(), torch.zeros(1, 2, seq_len, head_dim), i)

        bos_before = [get_keys(cache, i)[:, :, :1, :].clone() for i in range(num_layers)]

        correct_rope_positions_with_bos(cache, offset=10, model=small_fake_gemma3_model)

        for i in range(num_layers):
            bos_after = get_keys(cache, i)[:, :, :1, :]
            torch.testing.assert_close(bos_after, bos_before[i])

    def test_full_vs_sliding_with_bos(self, small_fake_gemma3_model):
        """With BOS, full and sliding layers still get different corrections on doc tokens."""
        head_dim = 16
        seq_len = 6  # 1 BOS + 5 doc tokens
        num_layers = 4
        offset = 10

        # Build cache with identical doc keys across all layers
        cache = DynamicCache()
        bos_key = torch.randn(1, 2, 1, head_dim)
        doc_keys = torch.randn(1, 2, 5, head_dim)
        for i in range(num_layers):
            full_keys = torch.cat([bos_key.clone(), doc_keys.clone()], dim=2)
            cache.update(full_keys, torch.zeros(1, 2, seq_len, head_dim), i)

        correct_rope_positions_with_bos(cache, offset=offset, model=small_fake_gemma3_model)

        # Doc tokens at sliding layers should be equal
        doc_0 = get_keys(cache, 0)[:, :, 1:, :]
        doc_1 = get_keys(cache, 1)[:, :, 1:, :]
        torch.testing.assert_close(doc_0, doc_1)

        # Doc tokens at full attention layer (2) should differ
        doc_2 = get_keys(cache, 2)[:, :, 1:, :]
        assert not torch.allclose(doc_0, doc_2, atol=1e-4)


# =============================================================================
# TestCorrectRopeBackwardCompat
# =============================================================================

class TestCorrectRopeBackwardCompat:
    def test_mistral_identical_to_old_behavior(self, small_fake_model):
        """Mistral config produces identical results — no per-layer theta branching."""
        head_dim = 16
        seq_len = 6
        offset = 7

        # Cache A: corrected with new code
        cache_a = DynamicCache()
        keys = torch.randn(1, 4, seq_len, head_dim)
        cache_a.update(keys.clone(), torch.zeros(1, 4, seq_len, head_dim), 0)
        cache_a.update(keys.clone(), torch.zeros(1, 4, seq_len, head_dim), 1)

        correct_rope_positions(cache_a, offset=offset, model=small_fake_model)

        # Cache B: manual single-theta correction (old behavior)
        cache_b = DynamicCache()
        cache_b.update(keys.clone(), torch.zeros(1, 4, seq_len, head_dim), 0)
        cache_b.update(keys.clone(), torch.zeros(1, 4, seq_len, head_dim), 1)

        cos_a, sin_a = _build_rope_correction(offset, head_dim, 10000.0)
        from lib.kv_cache import _rotate_half
        for layer_idx in range(2):
            k = get_keys(cache_b, layer_idx)
            new_k = k * cos_a + _rotate_half(k) * sin_a
            if hasattr(cache_b, 'key_cache'):
                cache_b.key_cache[layer_idx] = new_k
            else:
                cache_b.layers[layer_idx].keys = new_k

        for i in range(2):
            torch.testing.assert_close(get_keys(cache_a, i), get_keys(cache_b, i))

    def test_offset_zero_noop(self, small_fake_model):
        """offset=0 returns cache unchanged for Mistral."""
        head_dim = 16
        cache = DynamicCache()
        keys = torch.randn(1, 4, 5, head_dim)
        cache.update(keys.clone(), torch.zeros(1, 4, 5, head_dim), 0)
        original = keys.clone()

        correct_rope_positions(cache, offset=0, model=small_fake_model)
        torch.testing.assert_close(get_keys(cache, 0), original)

    def test_offset_zero_noop_gemma3(self, small_fake_gemma3_model):
        """offset=0 returns cache unchanged for Gemma 3."""
        head_dim = 16
        cache = DynamicCache()
        keys = torch.randn(1, 2, 5, head_dim)
        cache.update(keys.clone(), torch.zeros(1, 2, 5, head_dim), 0)
        original = keys.clone()

        correct_rope_positions(cache, offset=0, model=small_fake_gemma3_model)
        torch.testing.assert_close(get_keys(cache, 0), original)

    def test_round_trip_gemma3(self, small_fake_gemma3_model):
        """Applying +S then -S recovers original keys for Gemma 3."""
        head_dim = 16
        num_layers = 4

        cache = DynamicCache()
        original_keys = []
        for i in range(num_layers):
            k = torch.randn(1, 2, 5, head_dim)
            original_keys.append(k.clone())
            cache.update(k, torch.zeros(1, 2, 5, head_dim), i)

        correct_rope_positions(cache, offset=-7, model=small_fake_gemma3_model)
        correct_rope_positions(cache, offset=7, model=small_fake_gemma3_model)

        for i in range(num_layers):
            torch.testing.assert_close(
                get_keys(cache, i), original_keys[i], atol=1e-5, rtol=1e-5
            )

    def test_values_not_modified_gemma3(self, small_fake_gemma3_model):
        """Value cache should remain untouched for Gemma 3."""
        head_dim = 16
        num_layers = 4

        cache = DynamicCache()
        original_values = []
        for i in range(num_layers):
            k = torch.randn(1, 2, 5, head_dim)
            v = torch.randn(1, 2, 5, head_dim)
            original_values.append(v.clone())
            cache.update(k, v, i)

        correct_rope_positions(cache, offset=10, model=small_fake_gemma3_model)

        from tests.conftest import get_values
        for i in range(num_layers):
            torch.testing.assert_close(get_values(cache, i), original_values[i])


# =============================================================================
# TestLoadModelConfig
# =============================================================================

class TestLoadModelConfig:
    def test_gemma3_bfloat16(self):
        """gemma3 model type resolves to bfloat16."""
        config = ExperimentConfig(model_type="gemma3")
        assert _resolve_dtype(config) == torch.bfloat16

    def test_mistral_float16(self):
        """mistral model type resolves to float16."""
        config = ExperimentConfig(model_type="mistral")
        assert _resolve_dtype(config) == torch.float16

    def test_explicit_override(self):
        """Explicit compute_dtype overrides auto."""
        config = ExperimentConfig(model_type="mistral", compute_dtype="bfloat16")
        assert _resolve_dtype(config) == torch.bfloat16

    def test_auto_default(self):
        """Default compute_dtype is 'auto'."""
        config = ExperimentConfig()
        assert config.compute_dtype == "auto"


# =============================================================================
# Slow integration tests (require HF auth + GPU)
# =============================================================================

@pytest.mark.slow
class TestGemma3Integration:
    """Integration tests that load the actual Gemma 3 4B model.

    Run with: pytest tests/test_multi_model.py -m slow -v
    Requires: HF auth token with Gemma 3 license accepted.
    """

    @pytest.fixture(scope="class")
    def gemma3_model_and_tokenizer(self):
        """Load Gemma 3 4B once for all tests in this class."""
        from lib.model_utils import load_model
        config = ExperimentConfig(
            model_name="google/gemma-3-4b-it",
            model_type="gemma3",
            use_4bit=False,  # Pure bfloat16 for reliable testing
            compute_dtype="bfloat16",
        )
        model, tokenizer = load_model(config)
        return model, tokenizer, config

    def test_bare_cache_finite(self, gemma3_model_and_tokenizer):
        """Build bare cache and verify keys/values are finite."""
        model, tokenizer, config = gemma3_model_and_tokenizer
        from lib.kv_cache import build_kv_cache
        ctx_len, cache = build_kv_cache(
            "The capital of France is Paris.", model, tokenizer, config
        )
        assert ctx_len > 0
        for i in range(len(cache)):
            k = _get_cache_keys(cache, i)
            assert torch.isfinite(k).all(), f"Non-finite keys at layer {i}"

    def test_truncated_cache_finite(self, gemma3_model_and_tokenizer):
        """Build truncated+corrected cache and verify finite NLL."""
        model, tokenizer, config = gemma3_model_and_tokenizer
        from lib.kv_cache import build_truncated_kv_cache_corrected
        keep_len, cache = build_truncated_kv_cache_corrected(
            "capital of France",
            "Paris is the capital and largest city of France.",
            model, tokenizer, config,
        )
        assert keep_len > 0
        for i in range(len(cache)):
            k = _get_cache_keys(cache, i)
            assert torch.isfinite(k).all(), f"Non-finite keys at layer {i}"

    def test_per_layer_correction_differs(self, gemma3_model_and_tokenizer):
        """Verify sliding and full attention layers get different corrections."""
        model, tokenizer, config = gemma3_model_and_tokenizer
        from lib.kv_cache import build_truncated_kv_cache_corrected

        # Build two truncated caches to compare: one to inspect
        keep_len, cache = build_truncated_kv_cache_corrected(
            "what is machine learning",
            "Machine learning is a subset of artificial intelligence.",
            model, tokenizer, config,
        )

        # Check that full_attention layers (every 6th starting at 5) have
        # different correction magnitude than sliding layers
        layer_types = _get_text_config(model.config).layer_types
        sliding_idx = next(i for i, t in enumerate(layer_types) if t == 'sliding_attention')
        full_idx = next(i for i, t in enumerate(layer_types) if t == 'full_attention')

        k_sliding = _get_cache_keys(cache, sliding_idx)
        k_full = _get_cache_keys(cache, full_idx)

        # They should have the same shape but different values
        assert k_sliding.shape == k_full.shape
        # Not identical (different theta corrections were applied)
        assert not torch.allclose(k_sliding, k_full, atol=1e-3)
