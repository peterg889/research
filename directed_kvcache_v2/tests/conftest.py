"""Shared fixtures for directed KV cache tests."""

import types
import torch
import pytest
from transformers import DynamicCache


def get_keys(cache, layer_idx):
    """Get key tensor from cache, compatible with both old and new transformers API."""
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx]
    return cache.layers[layer_idx].keys


def set_keys(cache, layer_idx, value):
    """Set key tensor in cache, compatible with both old and new transformers API."""
    if hasattr(cache, 'key_cache'):
        cache.key_cache[layer_idx] = value
    else:
        cache.layers[layer_idx].keys = value


def get_values(cache, layer_idx):
    """Get value tensor from cache, compatible with both old and new transformers API."""
    if hasattr(cache, 'value_cache'):
        return cache.value_cache[layer_idx]
    return cache.layers[layer_idx].values


def set_values(cache, layer_idx, value):
    """Set value tensor in cache, compatible with both old and new transformers API."""
    if hasattr(cache, 'value_cache'):
        cache.value_cache[layer_idx] = value
    else:
        cache.layers[layer_idx].values = value


def num_layers(cache):
    """Get number of layers in cache."""
    if hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    return len(cache.layers)


@pytest.fixture
def fake_model():
    """Namespace mimicking model.config for Mistral-7B defaults."""
    cfg = types.SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        rope_theta=10000.0,
    )
    model = types.SimpleNamespace(config=cfg)
    return model


@pytest.fixture
def small_fake_model():
    """Small model config for fast math tests."""
    cfg = types.SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        rope_theta=10000.0,
    )
    model = types.SimpleNamespace(config=cfg)
    return model


@pytest.fixture
def fake_gemma3_model():
    """Namespace mimicking Gemma 3 4B config with per-layer RoPE.

    Gemma 3 4B has 34 layers: repeating pattern of 5 sliding_attention + 1 full_attention.
    sliding_attention uses rope_theta=10000, full_attention uses rope_theta=1000000.
    head_dim is explicitly 256 (not hidden_size/num_heads = 2560/8 = 320).

    The real model is Gemma3ForConditionalGeneration (multimodal), so text attributes
    live on config.text_config. We mimic this composite structure.
    """
    # 5 full cycles of (5 sliding + 1 full) = 30, + 4 more sliding = 34 total
    layer_types = (['sliding_attention'] * 5 + ['full_attention']) * 5 + ['sliding_attention'] * 4
    text_cfg = types.SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        layer_types=layer_types,
        rope_parameters={
            'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
            'full_attention': {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1000000.0},
        },
    )
    cfg = types.SimpleNamespace(text_config=text_cfg)
    return types.SimpleNamespace(config=cfg)


@pytest.fixture
def small_fake_gemma3_model():
    """Small Gemma 3-like config for fast math tests (4 layers).

    Mimics the composite config structure (text_config) of the real model.
    """
    layer_types = ['sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention']
    text_cfg = types.SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=layer_types,
        rope_parameters={
            'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
            'full_attention': {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1000000.0},
        },
    )
    cfg = types.SimpleNamespace(text_config=text_cfg)
    return types.SimpleNamespace(config=cfg)


@pytest.fixture
def make_cache():
    """Factory that creates a DynamicCache with random tensors.

    Returns a callable: make_cache(num_layers, seq_len, num_heads=4, head_dim=16, batch=1)
    """

    def _make(num_layers=2, seq_len=10, num_heads=4, head_dim=16, batch=1):
        cache = DynamicCache()
        for layer_idx in range(num_layers):
            k = torch.randn(batch, num_heads, seq_len, head_dim)
            v = torch.randn(batch, num_heads, seq_len, head_dim)
            cache.update(k, v, layer_idx)
        return cache

    return _make
