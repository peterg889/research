"""KV cache manipulation utilities.

Provides functions for deep-copying and constructing prefix token sequences
for the two-phase KV cache scoring pipeline used across decoder-only
experiments.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import torch
from transformers import DynamicCache


def deep_copy_cache(cache: DynamicCache) -> DynamicCache:
    """Create a deep copy of a :class:`DynamicCache`.

    Clones every key and value tensor so that in-place mutations (e.g.
    quantization, normalization) on the copy do not affect the original.

    Args:
        cache: The source KV cache.

    Returns:
        A new :class:`DynamicCache` with independently allocated tensors.

    Example::

        original = encode_phase_a(doc_text)[0]
        copy = deep_copy_cache(original)
        quantize_kv_cache(copy, nbits=8)  # original is unchanged
    """
    cloned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys.clone()
        v = cache.layers[i].values.clone()
        cloned.update(k, v, i)
    return cloned


def make_prefix(token_ids: Sequence[int], length: int) -> List[int]:
    """Pad or truncate a token sequence to exactly ``length`` tokens.

    If the input is shorter than ``length``, it is repeated (tiled) until
    it reaches the desired length, then truncated.  If it is already at
    least ``length`` tokens, it is simply truncated.

    This is used to create fixed-length prefix sequences for Phase A
    conditioning (e.g. instruction tokens padded to L=64).

    Args:
        token_ids: Source token IDs (e.g. from tokenizing an instruction).
        length: Desired output length.

    Returns:
        A list of exactly ``length`` token IDs.

    Example::

        >>> make_prefix([10, 20, 30], 7)
        [10, 20, 30, 10, 20, 30, 10]
        >>> make_prefix([10, 20, 30, 40, 50], 3)
        [10, 20, 30]
    """
    ids = list(token_ids)
    if len(ids) >= length:
        return ids[:length]
    # Tile and truncate
    padded = ids * ((length // max(len(ids), 1)) + 1)
    return padded[:length]


def scramble_prefix(prefix_ids: Sequence[int], seed: int) -> List[int]:
    """Randomly shuffle a prefix token sequence deterministically.

    Uses a seeded RNG so that the same ``(prefix_ids, seed)`` pair always
    produces the same output.  This is used to create the
    ``scrambled_comprehend`` condition, which preserves the vocabulary of
    an instruction prefix while destroying its word order — enabling the
    three-level decomposition of structural / vocabulary / meaning effects.

    Args:
        prefix_ids: Source token IDs to shuffle.
        seed: Random seed for deterministic shuffling.

    Returns:
        A new list with the same tokens in a (seeded) random order.

    Example::

        >>> scramble_prefix([10, 20, 30, 40], seed=42)
        [20, 10, 40, 30]
    """
    rng = random.Random(seed)
    shuffled = list(prefix_ids)
    rng.shuffle(shuffled)
    return shuffled
