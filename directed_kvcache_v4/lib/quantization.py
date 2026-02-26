"""KV cache quantization and normalization utilities.

Provides simulated quantization (int8/int16/int4), scale normalization
(the norm-roundtrip that explains the "quantization benefit"), and outlier
clipping for KV cache tensors.

All cache-level functions preserve the BOS entry (index 0) unmodified,
operating only on document entries at indices 1+.

Key finding (Exp 07-09): Simulated int8 quantization paradoxically
*improves* NLL because the per-tensor ``absmax/qmax`` normalization step
corrects pathological scale drift in two-phase KV caches.  The
``norm_roundtrip_kv_cache`` function isolates this normalization effect
without any rounding or clamping.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def simulated_quantize(tensor: torch.Tensor, nbits: int) -> torch.Tensor:
    """Per-tensor symmetric quantization round-trip (simulated in bf16).

    Quantizes a tensor to ``nbits`` precision and immediately dequantizes
    back, simulating the effect of integer quantization without actually
    changing the dtype.  Uses per-tensor symmetric scaling::

        scale = absmax / qmax
        quantized = round(tensor / scale).clamp(-qmax, qmax)
        result = quantized * scale

    Args:
        tensor: Input tensor (any shape, typically bf16).
        nbits: Number of bits for quantization (e.g. 4, 8, 16).

    Returns:
        Tensor of the same shape and dtype with simulated quantization noise.

    Example::

        >>> t = torch.randn(4, 4, dtype=torch.bfloat16)
        >>> q = simulated_quantize(t, 8)  # 256-level quantization
        >>> q.dtype == t.dtype
        True
    """
    qmax = (1 << (nbits - 1)) - 1
    absmax = tensor.abs().max()
    if absmax == 0:
        return tensor
    scale = absmax / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(tensor.dtype)


def quantize_kv_cache(cache: DynamicCache, nbits: int) -> DynamicCache:
    """Apply simulated quantization to all K/V tensors in-place.

    Quantizes keys and values at all layers using :func:`simulated_quantize`.
    The BOS entry (index 0 along the sequence dimension) is preserved
    unquantized, since it serves as a critical anchor for the attention
    computation.

    Args:
        cache: KV cache to quantize (mutated in-place).
        nbits: Number of bits for quantization (e.g. 4, 8, 16).

    Returns:
        The same ``cache`` object, mutated in-place.

    Warning:
        This modifies the cache in-place.  Use :func:`lib.cache.deep_copy_cache`
        first if you need to preserve the original.
    """
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = simulated_quantize(k[:, :, 1:, :], nbits)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)
            v_bos = v[:, :, :1, :]
            v_doc = simulated_quantize(v[:, :, 1:, :], nbits)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def norm_roundtrip_kv_cache(
    cache: DynamicCache,
    qmax: int = 127,
) -> DynamicCache:
    """Apply scale normalization round-trip without rounding or clamping.

    Performs ``x → (x / scale) * scale`` where ``scale = absmax / qmax``.
    In exact arithmetic this is the identity function, but in bf16 the
    divide-then-multiply introduces tiny perturbations that correct
    pathological scale drift in two-phase KV caches.

    This is the mechanism behind the "quantization benefit" discovered in
    Exp 07-09: the normalization step alone captures 77-100% of the int8
    improvement across all tested datasets.

    BOS entry (index 0) is preserved unmodified.

    Args:
        cache: KV cache to normalize (mutated in-place).
        qmax: Normalization denominator (default 127, matching int8 scale).

    Returns:
        The same ``cache`` object, mutated in-place.
    """
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            k_absmax = k_doc.abs().max()
            if k_absmax > 0:
                k_scale = k_absmax / qmax
                k_doc = ((k_doc / k_scale) * k_scale).to(k.dtype)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)

            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]
            v_absmax = v_doc.abs().max()
            if v_absmax > 0:
                v_scale = v_absmax / qmax
                v_doc = ((v_doc / v_scale) * v_scale).to(v.dtype)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def clip_kv_cache(
    cache: DynamicCache,
    n_sigma: float,
) -> DynamicCache:
    """Clamp document K/V entries to within n standard deviations.

    For each layer, computes the per-tensor mean and standard deviation
    of the document entries (excluding BOS) and clamps values to
    ``[mean - n_sigma * std, mean + n_sigma * std]``.

    This tests the outlier-suppression hypothesis (H_B from Exp 08):
    whether clipping extreme values in the KV cache improves NLL.

    BOS entry (index 0) is preserved unmodified.

    Args:
        cache: KV cache to clip (mutated in-place).
        n_sigma: Number of standard deviations for the clamp range
            (e.g. 2.0 or 3.0).

    Returns:
        The same ``cache`` object, mutated in-place.
    """
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :].float()
            k_mean = k_doc.mean()
            k_std = k_doc.std()
            k_clipped = k_doc.clamp(
                k_mean - n_sigma * k_std, k_mean + n_sigma * k_std
            )
            cache.layers[i].keys = torch.cat(
                [k_bos, k_clipped.to(k.dtype)], dim=2
            )

            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :].float()
            v_mean = v_doc.mean()
            v_std = v_doc.std()
            v_clipped = v_doc.clamp(
                v_mean - n_sigma * v_std, v_mean + n_sigma * v_std
            )
            cache.layers[i].values = torch.cat(
                [v_bos, v_clipped.to(v.dtype)], dim=2
            )
    return cache
