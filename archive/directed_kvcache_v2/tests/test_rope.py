"""Tests for correct_rope_positions — the most critical function.

Includes both self-consistency tests and tests that validate against
the actual HuggingFace rotate_half / apply_rotary_pos_emb implementation
used by Mistral.
"""

import types
import torch
import pytest
from transformers import DynamicCache
from lib.kv_cache import correct_rope_positions, correct_rope_positions_with_bos
from tests.conftest import get_keys, get_values, set_keys, set_values, num_layers


# =============================================================================
# Reference HF implementation (copied from transformers.models.mistral)
# =============================================================================

def _rotate_half(x):
    """HuggingFace's rotate_half: split first/second half, not even/odd."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_ref(keys, positions, head_dim, rope_theta=10000.0):
    """Apply RoPE to keys at given positions using HF convention.

    Args:
        keys: (batch, heads, seq_len, head_dim)
        positions: (seq_len,) integer positions
        head_dim: dimension per head
        rope_theta: RoPE base frequency

    Returns:
        Rotated keys with same shape
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # freqs: (seq_len, head_dim // 2)
    freqs = torch.outer(positions.float(), inv_freq)
    # emb: (seq_len, head_dim) — duplicate for both halves
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=keys.dtype, device=keys.device)
    sin = emb.sin().to(dtype=keys.dtype, device=keys.device)
    # Broadcast: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (keys * cos) + (_rotate_half(keys) * sin)


# =============================================================================
# Self-consistency tests
# =============================================================================

def test_offset_zero_is_identity(make_cache, small_fake_model):
    """Offset 0 triggers early return; keys must be unchanged."""
    cache = make_cache(num_layers=2, seq_len=5, num_heads=4, head_dim=16)
    original_keys = [get_keys(cache, i).clone() for i in range(2)]

    result = correct_rope_positions(cache, offset=0, model=small_fake_model)

    for i in range(2):
        torch.testing.assert_close(get_keys(result, i), original_keys[i])


def test_round_trip_identity(make_cache, small_fake_model):
    """Applying +S then -S should recover the original keys."""
    cache = make_cache(num_layers=2, seq_len=6, num_heads=4, head_dim=16)
    original_keys = [get_keys(cache, i).clone() for i in range(2)]

    correct_rope_positions(cache, offset=-7, model=small_fake_model)
    correct_rope_positions(cache, offset=7, model=small_fake_model)

    for i in range(2):
        torch.testing.assert_close(
            get_keys(cache, i), original_keys[i], atol=1e-5, rtol=1e-5
        )


def test_rotation_additivity(make_cache, small_fake_model):
    """offset A then B should equal offset A+B in one step."""
    cache_a = make_cache(num_layers=2, seq_len=6, num_heads=4, head_dim=16)
    cache_b = DynamicCache()
    for i in range(2):
        cache_b.update(get_keys(cache_a, i).clone(), get_values(cache_a, i).clone(), i)

    correct_rope_positions(cache_a, offset=3, model=small_fake_model)
    correct_rope_positions(cache_a, offset=5, model=small_fake_model)

    correct_rope_positions(cache_b, offset=8, model=small_fake_model)

    for i in range(2):
        torch.testing.assert_close(
            get_keys(cache_a, i), get_keys(cache_b, i), atol=1e-5, rtol=1e-5
        )


def test_values_not_modified(make_cache, small_fake_model):
    """Value cache should remain untouched."""
    cache = make_cache(num_layers=2, seq_len=5, num_heads=4, head_dim=16)
    original_values = [get_values(cache, i).clone() for i in range(2)]

    correct_rope_positions(cache, offset=10, model=small_fake_model)

    for i in range(2):
        torch.testing.assert_close(get_values(cache, i), original_values[i])


def test_modifies_in_place(make_cache, small_fake_model):
    """Should return the same DynamicCache object."""
    cache = make_cache(num_layers=1, seq_len=3, num_heads=4, head_dim=16)
    result = correct_rope_positions(cache, offset=5, model=small_fake_model)
    assert result is cache


def test_preserves_key_norm(make_cache, small_fake_model):
    """RoPE is unitary; L2 norms of key vectors should be preserved."""
    cache = make_cache(num_layers=2, seq_len=8, num_heads=4, head_dim=16)
    norms_before = [get_keys(cache, i).norm(dim=-1).clone() for i in range(2)]

    correct_rope_positions(cache, offset=13, model=small_fake_model)

    for i in range(2):
        norms_after = get_keys(cache, i).norm(dim=-1)
        torch.testing.assert_close(norms_after, norms_before[i], atol=1e-5, rtol=1e-5)


def test_float16_round_trip(make_cache, small_fake_model):
    """Half-precision round trip with looser tolerance."""
    cache = make_cache(num_layers=1, seq_len=4, num_heads=4, head_dim=16)
    n = num_layers(cache)
    for i in range(n):
        set_keys(cache, i, get_keys(cache, i).half())
        set_values(cache, i, get_values(cache, i).half())

    original_keys = [get_keys(cache, i).clone() for i in range(n)]

    correct_rope_positions(cache, offset=-5, model=small_fake_model)
    correct_rope_positions(cache, offset=5, model=small_fake_model)

    for i in range(n):
        torch.testing.assert_close(
            get_keys(cache, i), original_keys[i], atol=5e-3, rtol=5e-3
        )


def test_large_offset_no_nan(make_cache, small_fake_model):
    """Large offset (4096) should not produce NaN or Inf."""
    cache = make_cache(num_layers=2, seq_len=4, num_heads=4, head_dim=16)

    correct_rope_positions(cache, offset=4096, model=small_fake_model)

    for i in range(2):
        assert not torch.isnan(get_keys(cache, i)).any()
        assert not torch.isinf(get_keys(cache, i)).any()


# =============================================================================
# Model-validated tests: compare against actual HF RoPE implementation
# =============================================================================

class TestAgainstHFImplementation:
    """Tests that validate our inverse RoPE against HF's actual forward RoPE."""

    @pytest.fixture
    def model_config(self):
        """Mistral-7B-like config."""
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                hidden_size=128,
                num_attention_heads=4,
                rope_theta=10000.0,
            )
        )

    def test_inverse_recovers_pre_rope_keys(self, model_config):
        """Apply HF forward RoPE at positions S..S+D-1, then our inverse with
        offset=S. Result should match HF forward RoPE at positions 0..D-1."""
        head_dim = 32  # 128 / 4
        batch, heads, seq_len = 1, 4, 10
        offset = 7

        # Random "pre-RoPE" keys
        k_raw = torch.randn(batch, heads, seq_len, head_dim)

        # Forward RoPE at offset positions (simulating surrogate prefix)
        positions_with_offset = torch.arange(offset, offset + seq_len)
        k_at_offset = _apply_rotary_pos_emb_ref(k_raw, positions_with_offset, head_dim)

        # Forward RoPE at correct positions (what we want to recover)
        positions_correct = torch.arange(0, seq_len)
        k_at_correct = _apply_rotary_pos_emb_ref(k_raw, positions_correct, head_dim)

        # Put the offset-rotated keys into a cache and apply our correction
        cache = DynamicCache()
        cache.update(k_at_offset.clone(), torch.zeros_like(k_at_offset), 0)
        correct_rope_positions(cache, offset=offset, model=model_config)

        corrected = get_keys(cache, 0)
        torch.testing.assert_close(corrected, k_at_correct, atol=1e-5, rtol=1e-5)

    def test_inverse_single_position(self, model_config):
        """Single token: apply HF RoPE at position P, invert with offset P,
        should recover RoPE at position 0."""
        head_dim = 32
        k_raw = torch.randn(1, 4, 1, head_dim)

        for pos in [1, 5, 20, 100]:
            k_at_pos = _apply_rotary_pos_emb_ref(k_raw, torch.tensor([pos]), head_dim)
            k_at_zero = _apply_rotary_pos_emb_ref(k_raw, torch.tensor([0]), head_dim)

            cache = DynamicCache()
            cache.update(k_at_pos.clone(), torch.zeros_like(k_at_pos), 0)
            correct_rope_positions(cache, offset=pos, model=model_config)

            torch.testing.assert_close(
                get_keys(cache, 0), k_at_zero, atol=1e-5, rtol=1e-5
            )

    def test_inverse_various_offsets(self, model_config):
        """Test several different offsets including large ones."""
        head_dim = 32
        batch, heads, seq_len = 1, 4, 5
        k_raw = torch.randn(batch, heads, seq_len, head_dim)
        positions_correct = torch.arange(0, seq_len)
        k_correct = _apply_rotary_pos_emb_ref(k_raw, positions_correct, head_dim)

        for offset in [1, 10, 50, 200, 1024]:
            positions_offset = torch.arange(offset, offset + seq_len)
            k_offset = _apply_rotary_pos_emb_ref(k_raw, positions_offset, head_dim)

            cache = DynamicCache()
            cache.update(k_offset.clone(), torch.zeros_like(k_offset), 0)
            correct_rope_positions(cache, offset=offset, model=model_config)

            # Larger offsets accumulate more floating point error
            tol = 1e-5 if offset <= 50 else 1e-4
            torch.testing.assert_close(
                get_keys(cache, 0), k_correct, atol=tol, rtol=tol,
                msg=f"Failed for offset={offset}"
            )

    def test_inverse_multi_layer(self, model_config):
        """Verify correction works across multiple layers."""
        head_dim = 32
        batch, heads, seq_len = 1, 4, 8
        offset = 15
        num_layers = 3

        k_raws = [torch.randn(batch, heads, seq_len, head_dim) for _ in range(num_layers)]
        positions_correct = torch.arange(0, seq_len)
        positions_offset = torch.arange(offset, offset + seq_len)

        cache = DynamicCache()
        for i, k_raw in enumerate(k_raws):
            k_offset = _apply_rotary_pos_emb_ref(k_raw, positions_offset, head_dim)
            cache.update(k_offset.clone(), torch.zeros(batch, heads, seq_len, head_dim), i)

        correct_rope_positions(cache, offset=offset, model=model_config)

        for i, k_raw in enumerate(k_raws):
            k_correct = _apply_rotary_pos_emb_ref(k_raw, positions_correct, head_dim)
            torch.testing.assert_close(
                get_keys(cache, i), k_correct, atol=1e-5, rtol=1e-5,
                msg=f"Failed for layer {i}"
            )

    def test_inverse_float16(self, model_config):
        """Test with float16 precision (matching quantized model usage)."""
        head_dim = 32
        batch, heads, seq_len = 1, 4, 6
        offset = 20

        k_raw = torch.randn(batch, heads, seq_len, head_dim)
        positions_correct = torch.arange(0, seq_len)
        positions_offset = torch.arange(offset, offset + seq_len)

        k_correct = _apply_rotary_pos_emb_ref(k_raw, positions_correct, head_dim).half()
        k_offset = _apply_rotary_pos_emb_ref(k_raw, positions_offset, head_dim).half()

        cache = DynamicCache()
        cache.update(k_offset.clone(), torch.zeros_like(k_offset), 0)
        correct_rope_positions(cache, offset=offset, model=model_config)

        torch.testing.assert_close(
            get_keys(cache, 0), k_correct, atol=5e-2, rtol=5e-2
        )

    def test_known_rotation_half_split(self):
        """Hand-computed rotation for head_dim=4, offset=3, verifying half-split."""
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                hidden_size=4, num_attention_heads=1, rope_theta=10000.0
            )
        )
        head_dim = 4
        # Key at position 3, want to shift to position 0
        k_raw = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)

        # Apply HF forward RoPE at position 3
        k_at_3 = _apply_rotary_pos_emb_ref(k_raw, torch.tensor([3]), head_dim, rope_theta=10000.0)

        # Apply HF forward RoPE at position 0
        k_at_0 = _apply_rotary_pos_emb_ref(k_raw, torch.tensor([0]), head_dim, rope_theta=10000.0)

        # Our inverse should convert position 3 -> position 0
        cache = DynamicCache()
        cache.update(k_at_3.clone(), torch.zeros_like(k_at_3), 0)
        correct_rope_positions(cache, offset=3, model=model)

        torch.testing.assert_close(get_keys(cache, 0), k_at_0, atol=1e-6, rtol=1e-6)


class TestCorrectRopePositionsWithBos:
    """Tests for correct_rope_positions_with_bos which skips BOS at index 0."""

    @pytest.fixture
    def model_config(self):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                hidden_size=128,
                num_attention_heads=4,
                rope_theta=10000.0,
            )
        )

    def test_bos_untouched(self, model_config):
        """BOS token (index 0) should not be modified."""
        head_dim = 32
        batch, heads, seq_len = 1, 4, 6  # 1 BOS + 5 doc tokens
        cache = DynamicCache()
        keys = torch.randn(batch, heads, seq_len, head_dim)
        cache.update(keys.clone(), torch.zeros_like(keys), 0)

        bos_before = get_keys(cache, 0)[:, :, :1, :].clone()
        correct_rope_positions_with_bos(cache, offset=10, model=model_config)
        bos_after = get_keys(cache, 0)[:, :, :1, :]

        torch.testing.assert_close(bos_after, bos_before)

    def test_doc_tokens_corrected(self, model_config):
        """Document tokens (index 1+) should be corrected same as correct_rope_positions."""
        head_dim = 32
        batch, heads, doc_len = 1, 4, 5
        offset = 12

        k_raw = torch.randn(batch, heads, doc_len, head_dim)
        positions_offset = torch.arange(offset, offset + doc_len)
        positions_correct = torch.arange(0, doc_len)

        k_at_offset = _apply_rotary_pos_emb_ref(k_raw, positions_offset, head_dim)
        k_at_correct = _apply_rotary_pos_emb_ref(k_raw, positions_correct, head_dim)

        # Build cache with [BOS_placeholder, doc_tokens_at_offset]
        bos_key = torch.randn(batch, heads, 1, head_dim)
        full_keys = torch.cat([bos_key, k_at_offset], dim=2)
        cache = DynamicCache()
        cache.update(full_keys.clone(), torch.zeros_like(full_keys), 0)

        correct_rope_positions_with_bos(cache, offset=offset, model=model_config)

        corrected_doc = get_keys(cache, 0)[:, :, 1:, :]
        torch.testing.assert_close(corrected_doc, k_at_correct, atol=1e-5, rtol=1e-5)

    def test_offset_zero_identity(self, model_config):
        """Offset 0 should leave everything unchanged."""
        head_dim = 32
        cache = DynamicCache()
        keys = torch.randn(1, 4, 6, head_dim)
        cache.update(keys.clone(), torch.zeros_like(keys), 0)
        original = keys.clone()

        correct_rope_positions_with_bos(cache, offset=0, model=model_config)
        torch.testing.assert_close(get_keys(cache, 0), original)
