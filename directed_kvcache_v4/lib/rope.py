"""RoPE repositioning and KV cache selection utilities.

Model-agnostic functions for two-phase KV cache scoring:
    - **select_kv_cache**: Keep only BOS + doc entries (discard prefix + newline).
    - **reposition_kv_cache**: Rotate doc keys via RoPE delta to target positions.
    - **rotate_half**: The ``[-x2, x1]`` half-rotation used in RoPE.

These functions accept ``inv_freq`` dicts and ``layer_type`` lists as arguments
and work across all model families. For extracting those arguments from each
model's config, use ``directed_kvcache_publication.model_adapters``.

Note:
    ``build_layer_inv_freqs`` and ``get_layer_types`` in this module are
    **deprecated** — they only handle Gemma 3 and miss linear RoPE scaling.
    Use ``model_adapters.build_layer_inv_freqs`` and
    ``model_adapters.get_layer_types`` for multi-model support.

Warning:
    ``reposition_kv_cache`` only rotates **keys** (not values), because
    all supported model families apply RoPE to keys and queries only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from transformers import DynamicCache


def build_layer_inv_freqs(
    model,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Build per-layer-type inverse frequency tensors for RoPE rotation.

    .. deprecated::
        This version only handles Gemma 3 models with per-layer-type
        ``rope_parameters`` and does NOT support ``rope_type="linear"``
        scaling (needed for Gemma 3 4B/12B/27B full_attention layers).
        For multi-model support, use
        ``directed_kvcache_publication.model_adapters.build_layer_inv_freqs``
        instead, which handles all model families and linear scaling.

    Args:
        model: A Gemma 3 model (``Gemma3ForConditionalGeneration`` or similar)
            with a ``config.text_config`` attribute.
        device: Device for the output tensors.  Defaults to the device of the
            model's first parameter.

    Returns:
        Dict mapping layer type name to a 1-D ``float32`` tensor of shape
        ``(head_dim // 2,)`` containing the inverse frequencies.
    """
    if device is None:
        device = next(model.parameters()).device
    text_cfg = getattr(model.config, "text_config", model.config)
    rope_params = getattr(text_cfg, "rope_parameters", {})
    head_dim = text_cfg.head_dim

    inv_freqs: Dict[str, torch.Tensor] = {}
    for lt, params in rope_params.items():
        theta = params.get("rope_theta", 10000.0)
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        inv_freqs[lt] = inv_freq
    return inv_freqs


def get_layer_types(model) -> List[str]:
    """Return the per-layer attention type list from a Gemma 3 model.

    .. deprecated::
        This version only handles models with an explicit ``layer_types``
        config attribute. For multi-model support, use
        ``directed_kvcache_publication.model_adapters.get_layer_types``
        instead, which returns ``["all"] * num_layers`` for models
        without per-layer types (Qwen, Mistral, DeepSeek).

    Args:
        model: A Gemma 3 model with ``config.text_config.layer_types``.

    Returns:
        List of strings like ``["sliding_attention", ..., "full_attention", ...]``.
    """
    text_cfg = getattr(model.config, "text_config", model.config)
    return list(getattr(text_cfg, "layer_types", []))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the RoPE half-rotation: ``[-x2, x1]``.

    Splits the last dimension in half and swaps the two halves with a sign
    flip, implementing the imaginary-unit multiplication step of rotary
    position embeddings.

    Args:
        x: Tensor of shape ``(..., D)`` where ``D`` is even.

    Returns:
        Tensor of the same shape with halves swapped and negated.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def select_kv_cache(
    cache: DynamicCache,
    indices: Sequence[int],
    device: torch.device | None = None,
) -> DynamicCache:
    """Select specific token positions from a KV cache.

    Creates a new :class:`DynamicCache` containing only the entries at the
    given sequence-position indices.  This is used after Phase A to keep
    only the BOS token and document tokens (discarding prefix and newline
    tokens).

    Args:
        cache: Source KV cache from a model forward pass.
        indices: Sequence-position indices to keep (e.g. ``[0, 5, 6, 7, 8]``
            to keep BOS at 0 and doc tokens at 5–8).
        device: Device for the index tensor.  Defaults to the device of the
            first key tensor in the cache.

    Returns:
        A new :class:`DynamicCache` with only the selected positions.

    Warning:
        For models with sliding-window attention, the total number of
        selected indices must not exceed ``sliding_window - 1`` (1023 for
        Gemma 3 with ``sliding_window=1024``).
    """
    if device is None:
        device = cache.layers[0].keys.device
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    selected = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def reposition_kv_cache(
    cache: DynamicCache,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    layer_inv_freqs: Dict[str, torch.Tensor],
    layer_types: List[str],
    bos_start: int = 0,
) -> DynamicCache:
    """Reposition document keys in a KV cache via RoPE delta rotation.

    After ``select_kv_cache`` keeps ``[BOS, doc_0, ..., doc_{D-1}]``, the
    doc keys still carry RoPE embeddings for their original positions
    ``(1+P+NL) ... (1+P+NL+D-1)``.  This function rotates them to new
    target positions (typically ``1 ... D``).

    The rotation applies a **delta** of ``(new - old)`` to each doc key
    using the standard RoPE formula::

        key_new = key * cos(delta) + rotate_half(key) * sin(delta)

    The BOS entry at ``cache[:, :, bos_start, :]`` is left untouched.

    Args:
        cache: KV cache with shape ``(1, n_heads, 1+D, head_dim)`` per layer.
        old_positions: 1-D tensor of original positions for the D doc tokens.
        new_positions: 1-D tensor of desired positions for the D doc tokens.
        layer_inv_freqs: Output of :func:`build_layer_inv_freqs`.
        layer_types: Output of :func:`get_layer_types`.
        bos_start: Index of the BOS token in the cache (default 0).

    Returns:
        The same ``cache`` object, mutated in-place, with doc keys rotated.

    Warning:
        Only **keys** are rotated.  Gemma 3 applies RoPE to keys in the
        attention computation; values are position-independent.
    """
    delta = new_positions - old_positions
    for L in range(len(cache.layers)):
        lt = layer_types[L]
        inv_freq = layer_inv_freqs[lt]
        k = cache.layers[L].keys
        doc_keys = k[:, :, bos_start + 1 :, :]
        freqs = torch.einsum("i,j->ij", delta.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_delta = emb.cos().to(k.dtype).unsqueeze(0).unsqueeze(0)
        sin_delta = emb.sin().to(k.dtype).unsqueeze(0).unsqueeze(0)
        doc_keys_new = doc_keys * cos_delta + rotate_half(doc_keys) * sin_delta
        cache.layers[L].keys = torch.cat(
            [k[:, :, : bos_start + 1, :], doc_keys_new], dim=2
        )
    return cache
