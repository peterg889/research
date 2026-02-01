"""Tests for load_evaluation_samples with mock data."""

import types
import pytest
import numpy as np
from unittest.mock import patch

from lib.data import load_evaluation_samples, count_words
from lib.config import ExperimentConfig


def _make_item(passage_text, query, is_selected=1, answers=None,
               well_formed=None, word_count_override=None):
    """Helper to build a mock dataset item."""
    return {
        'passages': {
            'passage_text': [passage_text],
            'is_selected': [is_selected],
        },
        'query': query,
        'answers': answers or [],
        'wellFormedAnswers': well_formed or [],
    }


class TestWordCountFiltering:
    def test_passage_within_range_accepted(self):
        config = ExperimentConfig(min_passage_words=5, max_passage_words=20, num_samples=10)
        passage = "This is a passage with exactly eight words here."
        items = [_make_item(passage, "query", answers=["answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 1

    def test_passage_too_short_rejected(self):
        config = ExperimentConfig(min_passage_words=10, max_passage_words=100, num_samples=10)
        passage = "Too short."
        items = [_make_item(passage, "query", answers=["answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 0

    def test_passage_too_long_rejected(self):
        config = ExperimentConfig(min_passage_words=5, max_passage_words=10, num_samples=10)
        passage = " ".join(["word"] * 15)
        items = [_make_item(passage, "query", answers=["answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 0


class TestIsSelectedFiltering:
    def test_selected_passage_accepted(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", is_selected=1, answers=["answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 1

    def test_unselected_passage_rejected(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", is_selected=0, answers=["answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 0


class TestAnswerExtraction:
    def test_well_formed_preferred_over_answers(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", answers=["plain answer"],
                           well_formed=["well formed answer"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 1
        assert result[0]['answer'] == "well formed answer"

    def test_falls_back_to_answers(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", answers=["plain answer"],
                           well_formed=[])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 1
        assert result[0]['answer'] == "plain answer"

    def test_no_answer_present_excluded(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", answers=["No Answer Present."],
                           well_formed=[])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        assert len(result) == 0

    def test_well_formed_bracket_excluded(self):
        """wellFormedAnswers == ['[]'] should be skipped."""
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", answers=["real answer"],
                           well_formed=["[]"])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=True)

        # wellFormedAnswers[0] == '[]' is excluded, falls back to answers
        assert len(result) == 1
        assert result[0]['answer'] == "real answer"


class TestRequireAnswerFalse:
    def test_no_answer_still_accepted(self):
        config = ExperimentConfig(min_passage_words=3, max_passage_words=100, num_samples=10)
        passage = "This is a valid passage with enough words."
        items = [_make_item(passage, "query", answers=[], well_formed=[])]

        with patch('lib.data.tqdm', side_effect=lambda x, **kw: x):
            result = load_evaluation_samples(items, config, require_answer=False)

        assert len(result) == 1
        assert 'answer' not in result[0]
