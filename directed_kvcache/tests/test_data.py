"""Tests for count_words utility."""

from lib.data import count_words


def test_count_words_simple():
    assert count_words("hello world foo") == 3


def test_count_words_empty():
    assert count_words("") == 0


def test_count_words_extra_spaces():
    assert count_words("  hello   world  ") == 2


def test_count_words_newlines():
    assert count_words("hello\nworld\nfoo") == 3
