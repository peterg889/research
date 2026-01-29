"""Tests for correct_rope_positions — the most critical function."""

import types
import torch
import pytest
from transformers import DynamicCache
from lib.kv_cache import correct_rope_positions
from tests.conftest import get_keys, get_values, set_keys, set_values, num_layers


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


def test_known_rotation():
    """Hand-computed rotation for head_dim=4, offset=3."""
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            hidden_size=4, num_attention_heads=1, rope_theta=10000.0
        )
    )
    inv_freq = torch.tensor([1.0, 0.01])
    angles = -3 * inv_freq
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    key = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])  # (1,1,1,4)

    cache = DynamicCache()
    cache.update(key.clone(), torch.zeros_like(key), 0)

    correct_rope_positions(cache, offset=3, model=model)

    expected = torch.tensor([
        [cos_a[0].item(), sin_a[0].item(), cos_a[1].item(), sin_a[1].item()]
    ]).reshape(1, 1, 1, 4)

    torch.testing.assert_close(get_keys(cache, 0), expected, atol=1e-6, rtol=1e-6)


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
