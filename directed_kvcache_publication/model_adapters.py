"""Multi-model adapters for RoPE parameter extraction.

The core lib/rope.py functions (rotate_half, select_kv_cache, reposition_kv_cache)
are model-agnostic — they accept inv_freq dicts and layer_type lists as arguments.
This module provides the model-family-specific logic to EXTRACT those arguments
from each model's config.

Supported model families:
    - Gemma 3 (gemma3/gemma3_text): Hybrid sliding/full attention, per-layer-type
      rope_theta. Models ≥4B use rope_type="linear" with factor=8.0 on full_attention.
    - Gemma 3N (gemma3n_text): Hybrid sliding/full attention, default rope_type.
    - Llama 3.x (llama): Full attention only, single rope_theta
    - Qwen 2.x (qwen2): Full attention only, single rope_theta
    - Mistral (mistral): Full attention only, single rope_theta
    - Ministral (ministral): Hybrid sliding/full attention, single rope_theta for all layers
    - DeepSeek R1 Distill (qwen2): Qwen2 architecture, standard rope_theta=10000

Usage::

    from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info

    info = get_model_info(model)
    inv_freqs = build_layer_inv_freqs(model)
    layer_types = get_layer_types(model)
    # Then pass these to lib.rope.reposition_kv_cache() as usual
"""

from __future__ import annotations

from typing import Dict, List

import torch


def _get_text_config(model):
    """Get the text config, handling Gemma 3's nested text_config."""
    return getattr(model.config, "text_config", model.config)


def _get_model_type(model) -> str:
    """Determine the model family from config."""
    cfg = _get_text_config(model)
    return getattr(cfg, "model_type", "unknown")


def _get_head_dim(model) -> int:
    """Get the head dimension, handling different config conventions."""
    cfg = _get_text_config(model)
    # Explicit head_dim attribute (Gemma 3, Llama 3.1)
    if hasattr(cfg, "head_dim") and cfg.head_dim is not None:
        return cfg.head_dim
    # Computed from hidden_size / num_attention_heads (Qwen 2.5)
    return cfg.hidden_size // cfg.num_attention_heads


def _get_num_layers(model) -> int:
    """Get the number of layers."""
    cfg = _get_text_config(model)
    return getattr(cfg, "num_hidden_layers", 0)


def get_model_info(model) -> dict:
    """Get key model properties for display/logging.

    Returns:
        Dict with model_type, head_dim, num_layers, num_kv_heads, has_sliding,
        rope_thetas, and max_cache_len.
    """
    cfg = _get_text_config(model)
    model_type = _get_model_type(model)
    head_dim = _get_head_dim(model)
    num_layers = _get_num_layers(model)
    num_kv_heads = getattr(cfg, "num_key_value_heads",
                           getattr(cfg, "num_attention_heads", 0))

    has_sliding = hasattr(cfg, "layer_types") and "sliding_attention" in getattr(cfg, "layer_types", [])

    # Extract rope_theta values
    rope_params = getattr(cfg, "rope_parameters", {})
    if has_sliding:
        # Gemma 3: dict of dicts
        thetas = {lt: params.get("rope_theta", 10000.0)
                  for lt, params in rope_params.items()
                  if isinstance(params, dict)}
    else:
        # Llama/Qwen: flat dict
        thetas = {"all": rope_params.get("rope_theta", 10000.0)}

    # Max cache length (sliding window constraint for Gemma)
    if has_sliding:
        sliding_window = getattr(cfg, "sliding_window", 4096)
        max_cache = sliding_window - 1
    else:
        max_cache = getattr(cfg, "max_position_embeddings", 8192)

    return {
        "model_type": model_type,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "has_sliding": has_sliding,
        "rope_thetas": thetas,
        "max_cache_len": max_cache,
    }


def build_layer_inv_freqs(
    model,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Build per-layer-type inverse frequency tensors for RoPE rotation.

    Works across model families by detecting the config structure:
    - Gemma 3: per-layer-type rope_parameters (different theta per type)
    - Llama/Qwen: single rope_parameters (one theta for all layers)

    Args:
        model: Any HuggingFace causal LM model.
        device: Device for output tensors.

    Returns:
        Dict mapping layer type name to inv_freq tensor of shape (head_dim // 2,).
        For Gemma 3: keys are "sliding_attention", "full_attention".
        For Llama/Qwen: single key "all".
    """
    if device is None:
        device = next(model.parameters()).device

    cfg = _get_text_config(model)
    head_dim = _get_head_dim(model)
    rope_params = getattr(cfg, "rope_parameters", {})

    inv_freqs: Dict[str, torch.Tensor] = {}

    # Detect config style by checking if rope_parameters is a dict-of-dicts
    # (Gemma 3: {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}})
    # vs a flat dict (Llama/Qwen/Mistral: {"rope_theta": ..., "rope_type": ...})
    is_per_layer_type = any(isinstance(v, dict) for v in rope_params.values())

    if is_per_layer_type:
        # Gemma 3 / Gemma 3N: per-layer-type rope_parameters
        for lt, params in rope_params.items():
            if not isinstance(params, dict):
                continue
            theta = params.get("rope_theta", 10000.0)
            inv_freq = 1.0 / (
                theta ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                    / head_dim
                )
            )
            # Apply linear scaling if present (e.g. Gemma 3 4B/12B/27B full_attention
            # uses rope_type="linear" with factor=8.0, which divides inv_freq by factor)
            rope_type = params.get("rope_type", "default")
            if rope_type == "linear":
                factor = params.get("factor", 1.0)
                inv_freq = inv_freq / factor
            inv_freqs[lt] = inv_freq
    else:
        # Flat dict: single rope_theta for all layers
        theta = rope_params.get("rope_theta", 10000.0)
        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        # Apply linear scaling if present
        rope_type = rope_params.get("rope_type", "default")
        if rope_type == "linear":
            factor = rope_params.get("factor", 1.0)
            inv_freq = inv_freq / factor
        # Map to whatever layer_type names the config uses
        layer_type_names = set(getattr(cfg, "layer_types", None) or [])
        if layer_type_names:
            for lt in layer_type_names:
                inv_freqs[lt] = inv_freq
        else:
            inv_freqs["all"] = inv_freq

    return inv_freqs


def get_layer_types(model) -> List[str]:
    """Return per-layer attention type list.

    For Gemma 3: returns the actual layer_types from config (mix of
    "sliding_attention" and "full_attention").
    For Llama/Qwen: returns ["all"] * num_layers (uniform full attention).

    Args:
        model: Any HuggingFace causal LM model.

    Returns:
        List of layer type strings, one per model layer.
    """
    cfg = _get_text_config(model)
    if hasattr(cfg, "layer_types") and cfg.layer_types:
        return list(cfg.layer_types)
    # Uniform full attention
    n_layers = getattr(cfg, "num_hidden_layers", 0)
    return ["all"] * n_layers


def get_sliding_cache_limit(model) -> int | None:
    """Get the maximum cache entries for sliding attention layers.

    Returns None if the model has no sliding attention layers.
    For Gemma 3, returns sliding_window - 1 (e.g. 1023 for window=1024).
    """
    cfg = _get_text_config(model)
    if not hasattr(cfg, "layer_types"):
        return None
    layer_types = getattr(cfg, "layer_types", None) or []
    if "sliding_attention" not in layer_types:
        return None
    sliding_window = getattr(cfg, "sliding_window", 4096)
    return sliding_window - 1
