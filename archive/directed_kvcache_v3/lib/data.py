"""Dataset utilities for v3 experiments."""


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())
