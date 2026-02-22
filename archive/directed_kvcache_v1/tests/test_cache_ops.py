"""Tests for extract_and_truncate_cache."""

import torch
from transformers import DynamicCache
from lib.kv_cache import extract_and_truncate_cache
from tests.conftest import get_keys, get_values, num_layers


def test_truncate_keeps_last_n(make_cache):
    cache = make_cache(num_layers=2, seq_len=10, num_heads=4, head_dim=16)
    original_keys = [get_keys(cache, i).clone() for i in range(2)]

    result = extract_and_truncate_cache(cache, keep_last_n=3)

    for layer_idx in range(2):
        assert get_keys(result, layer_idx).shape[2] == 3
        assert get_values(result, layer_idx).shape[2] == 3
        torch.testing.assert_close(
            get_keys(result, layer_idx),
            original_keys[layer_idx][:, :, -3:, :],
        )


def test_keep_all_is_identity(make_cache):
    cache = make_cache(num_layers=2, seq_len=5)
    original_keys = [get_keys(cache, i).clone() for i in range(2)]

    result = extract_and_truncate_cache(cache, keep_last_n=5)

    for layer_idx in range(2):
        torch.testing.assert_close(get_keys(result, layer_idx), original_keys[layer_idx])


def test_keep_one(make_cache):
    cache = make_cache(num_layers=2, seq_len=8)
    result = extract_and_truncate_cache(cache, keep_last_n=1)

    for layer_idx in range(2):
        assert get_keys(result, layer_idx).shape[2] == 1
        assert get_values(result, layer_idx).shape[2] == 1


def test_preserves_layer_count(make_cache):
    cache = make_cache(num_layers=5, seq_len=10)
    result = extract_and_truncate_cache(cache, keep_last_n=4)
    assert num_layers(result) == 5


def test_legacy_tuple_format():
    """extract_and_truncate_cache should accept plain tuple-of-tuples format."""
    n_layers, seq_len, n_heads, head_dim = 2, 8, 4, 16
    legacy = tuple(
        (torch.randn(1, n_heads, seq_len, head_dim),
         torch.randn(1, n_heads, seq_len, head_dim))
        for _ in range(n_layers)
    )

    result = extract_and_truncate_cache(legacy, keep_last_n=3)
    assert num_layers(result) == n_layers
    assert get_keys(result, 0).shape[2] == 3
