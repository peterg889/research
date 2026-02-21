"""Tests for surrogate generation slicing, stripping, and similarity."""

import types
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from lib.surrogate import (
    generate_surrogate_with_template,
    generate_all_5_surrogates,
    compute_similarity,
    TOP_5_SURROGATE_TEMPLATES,
)
from lib.config import ExperimentConfig


class _TokenizerOutput:
    """Mimics HF tokenizer output with both dict and attribute access, plus .to()."""
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self._data = {'input_ids': input_ids}

    def __getitem__(self, key):
        return self._data[key]

    def to(self, device):
        return self

    def keys(self):
        return self._data.keys()


def _make_surrogate_mocks(decode_return="surrogate query"):
    config = ExperimentConfig(device="cpu", surrogate_max_tokens=20, surrogate_temperature=0.0)

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    tok_output = _TokenizerOutput(input_ids)
    tokenizer.return_value = tok_output
    tokenizer.eos_token_id = 2
    tokenizer.decode.return_value = decode_return

    model = MagicMock()
    # generate returns input + new tokens
    model.generate.return_value = torch.tensor([[10, 11, 12, 13, 14, 50, 51, 52]])

    return config, model, tokenizer


class TestGenerateSurrogateSlicing:
    def test_output_slicing_removes_input(self):
        config, model, tokenizer = _make_surrogate_mocks('  "Hello World"\n extra ')

        result = generate_surrogate_with_template("doc text", "template prompt", model, tokenizer, config)

        decode_args = tokenizer.decode.call_args
        sliced = decode_args[0][0]
        torch.testing.assert_close(sliced, torch.tensor([50, 51, 52]))

    def test_quote_stripping(self):
        config, model, tokenizer = _make_surrogate_mocks('"Hello World"')
        result = generate_surrogate_with_template("doc", "tmpl", model, tokenizer, config)
        assert result == "Hello World"

    def test_newline_stripping(self):
        config, model, tokenizer = _make_surrogate_mocks("first line\nsecond line")
        result = generate_surrogate_with_template("doc", "tmpl", model, tokenizer, config)
        assert result == "first line"


class TestGenerateAll5Surrogates:
    def test_returns_all_5_keys(self):
        config, model, tokenizer = _make_surrogate_mocks("surrogate query")
        result = generate_all_5_surrogates("doc text", model, tokenizer, config)

        assert set(result.keys()) == set(TOP_5_SURROGATE_TEMPLATES.keys())
        assert len(result) == 5
        for v in result.values():
            assert isinstance(v, str)


class TestComputeSimilarity:
    def test_identical_texts_similarity_1(self):
        embed_model = MagicMock()
        embed_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sim = compute_similarity("text", "text", embed_model)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_texts_similarity_0(self):
        embed_model = MagicMock()
        embed_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_similarity("a", "b", embed_model)
        assert abs(sim) < 1e-6

    def test_returns_float(self):
        embed_model = MagicMock()
        embed_model.encode.return_value = np.array([[0.5, 0.5], [0.3, 0.7]])
        sim = compute_similarity("a", "b", embed_model)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0
