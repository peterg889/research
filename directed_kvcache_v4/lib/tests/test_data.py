"""Tests for lib.data — text utilities."""

import pytest

from lib.data import count_words


class TestCountWordsBasic:
    def test_normal_sentence(self):
        assert count_words("The quick brown fox") == 4

    def test_empty_string(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   ") == 0

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_multiple_spaces(self):
        assert count_words("hello   world") == 2

    def test_tabs_and_newlines(self):
        assert count_words("hello\tworld\nfoo") == 3

    def test_leading_trailing_whitespace(self):
        assert count_words("  hello world  ") == 2


class TestCountWordsEdgeCases:
    def test_newlines_only(self):
        assert count_words("\n\n\n") == 0

    def test_mixed_whitespace_only(self):
        assert count_words(" \t \n \r ") == 0

    def test_punctuation_attached(self):
        """Punctuation is NOT separated — 'hello,' is one word."""
        assert count_words("hello, world!") == 2

    def test_hyphenated_word(self):
        """Hyphenated words count as one word."""
        assert count_words("well-known fact") == 2

    def test_very_long_text(self):
        text = " ".join(["word"] * 10000)
        assert count_words(text) == 10000

    def test_unicode(self):
        assert count_words("café résumé naïve") == 3

    def test_numbers_count_as_words(self):
        assert count_words("I have 42 cats") == 4

    def test_single_character_words(self):
        assert count_words("I a m") == 3


class TestCountWordsRealisticDocuments:
    """Test with patterns from actual experiment datasets."""

    def test_short_answer(self):
        """Typical short MS MARCO answer."""
        assert count_words("New York") == 2

    def test_medium_passage(self):
        """Typical ~100 token passage."""
        passage = (
            "The directed KV cache approach uses a surrogate query to "
            "condition document representations during the first phase "
            "of inference. This allows the model to attend to document "
            "tokens that are most relevant to the likely query, even "
            "before the actual query is known."
        )
        wc = count_words(passage)
        assert 30 < wc < 60

    def test_gsm8k_answer(self):
        """GSM8K-style answer with numbers."""
        answer = "The total cost is 42 + 18 = 60 dollars."
        assert count_words(answer) == 10
