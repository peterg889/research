"""Dataset loading and text utilities for KV cache experiments."""

from __future__ import annotations


def count_words(text: str) -> int:
    """Count whitespace-delimited words in a text string.

    Args:
        text: Input text.

    Returns:
        Number of words.  Returns 0 for empty or whitespace-only strings.

    Example::

        >>> count_words("The quick brown fox")
        4
        >>> count_words("")
        0
    """
    return len(text.split())
