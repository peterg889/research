"""Tests for build_truncated_kv_cache_corrected template formatting and length computation."""

import types
import torch
import pytest
from unittest.mock import MagicMock, patch
from transformers import DynamicCache

from lib.kv_cache import build_truncated_kv_cache_corrected


class _DictNamespace(dict):
    """Dict that also supports attribute access, mimicking HF tokenizer output."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


@pytest.fixture
def mock_env():
    """Set up mocked model, tokenizer, and config."""
    from lib.config import ExperimentConfig
    config = ExperimentConfig(device="cpu")

    tokenizer_calls = []

    def tokenize(text, return_tensors="pt", add_special_tokens=True,
                 padding=False, truncation=False):
        tokenizer_calls.append({'text': text, 'add_special_tokens': add_special_tokens})
        BOS = 1
        tokens = []
        if add_special_tokens:
            tokens.append(BOS)
        # 1 token per character
        tokens.extend(list(range(10, 10 + len(text))))
        ids = torch.tensor([tokens])
        return _DictNamespace(input_ids=ids)

    tokenizer = MagicMock(side_effect=tokenize)

    def model_call(input_ids, attention_mask, use_cache, return_dict):
        seq_len = input_ids.shape[1]
        cache = DynamicCache()
        cache.update(torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16), 0)
        return types.SimpleNamespace(past_key_values=cache)

    model = MagicMock()
    model.__call__ = model_call
    model.config = types.SimpleNamespace(
        hidden_size=64, num_attention_heads=4, rope_theta=10000.0,
    )

    return model, tokenizer, config, tokenizer_calls


class TestDefaultTemplateFormatting:
    def test_default_templates_used(self, mock_env):
        model, tokenizer, config, calls = mock_env
        build_truncated_kv_cache_corrected("test surrogate", "test document", model, tokenizer, config)

        assert "This document may be relevant to queries like: test surrogate\n\n" in calls[0]['text']
        assert calls[0]['add_special_tokens'] is True
        full = calls[1]['text']
        assert full.startswith("This document may be relevant to queries like: test surrogate\n\n")
        assert "Document:\ntest document" in full


class TestCustomTemplateFormatting:
    def test_custom_surrogate_template(self, mock_env):
        model, tokenizer, config, calls = mock_env
        build_truncated_kv_cache_corrected(
            "test surrogate", "test document", model, tokenizer, config,
            surrogate_prefix_template="Query: {surrogate}\n\n",
        )
        assert calls[0]['text'] == "Query: test surrogate\n\n"

    def test_custom_document_template(self, mock_env):
        model, tokenizer, config, calls = mock_env
        build_truncated_kv_cache_corrected(
            "test surrogate", "test document", model, tokenizer, config,
            document_template="Passage: {document}",
        )
        full = calls[1]['text']
        assert "Passage: test document" in full

    def test_both_custom_templates(self, mock_env):
        model, tokenizer, config, calls = mock_env
        build_truncated_kv_cache_corrected(
            "my query", "my doc", model, tokenizer, config,
            surrogate_prefix_template="Q: {surrogate} | ",
            document_template="D: {document}",
        )
        assert calls[0]['text'] == "Q: my query | "
        assert calls[1]['text'] == "Q: my query | D: my doc"


class TestDocLenComputation:
    def test_doc_len_is_full_minus_prefix(self, mock_env):
        model, tokenizer, config, calls = mock_env
        keep_len, cache = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config
        )
        prefix_text = "This document may be relevant to queries like: surr\n\n"
        doc_text = "Document:\ndoc"
        expected_doc_len = len(doc_text)
        expected_keep_len = 1 + expected_doc_len
        assert keep_len == expected_keep_len


class TestSurrogateOffset:
    def test_surrogate_offset_is_prefix_len_minus_1(self, mock_env):
        model, tokenizer, config, calls = mock_env
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected("surr", "doc", model, tokenizer, config)
            prefix_text = "This document may be relevant to queries like: surr\n\n"
            expected_prefix_len = 1 + len(prefix_text)  # BOS + chars
            expected_offset = expected_prefix_len - 1
            mock_correct.assert_called_once()
            actual_offset = mock_correct.call_args[0][1]
            assert actual_offset == expected_offset


class TestCallOrder:
    def test_truncate_then_correct(self, mock_env):
        model, tokenizer, config, calls = mock_env
        call_order = []

        with patch('lib.kv_cache.extract_and_truncate_cache_with_bos') as mock_trunc, \
             patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:

            mock_trunc.side_effect = lambda *a, **kw: (call_order.append('truncate'), DynamicCache())[1]
            mock_correct.side_effect = lambda *a, **kw: (call_order.append('correct'), a[0])[1]

            build_truncated_kv_cache_corrected("surr", "doc", model, tokenizer, config)
            assert call_order == ['truncate', 'correct']
