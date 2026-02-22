"""Mock-based tests for score_answer_with_cache."""

import types
import torch
import pytest
from transformers import DynamicCache

from lib.kv_cache import score_answer_with_cache
from lib.config import ExperimentConfig


class _DictNS(dict):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.__dict__.update(kw)


@pytest.fixture
def config():
    return ExperimentConfig(device="cpu")


def _make_model_and_tokenizer(query_ids, answer_ids, logits):
    """Build mock model and tokenizer.

    Args:
        query_ids: 1D list of token ids for query
        answer_ids: 1D list of token ids for answer
        logits: tensor of shape (1, answer_len, vocab_size) for answer pass
    """
    call_count = [0]
    def tokenize(text, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _DictNS(input_ids=torch.tensor([query_ids]))
        else:
            return _DictNS(input_ids=torch.tensor([answer_ids]))

    # Build model as a callable class to avoid MagicMock interception issues
    query_cache = DynamicCache()
    total_ctx = 5 + len(query_ids)
    query_cache.update(torch.randn(1, 1, total_ctx, 4), torch.randn(1, 1, total_ctx, 4), 0)

    model_calls = []

    class FakeModel:
        def __init__(self):
            self.config = None  # not needed for scoring

        def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
            model_calls.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'use_cache': use_cache,
            })
            if len(model_calls) == 1:  # query pass
                return types.SimpleNamespace(past_key_values=query_cache)
            else:  # answer pass
                return types.SimpleNamespace(logits=logits)

    return FakeModel(), tokenize, model_calls, query_cache


class TestScoreAnswerSingleToken:
    def test_single_token_answer_returns_zero(self, config):
        model, tokenize, _, _ = _make_model_and_tokenizer(
            query_ids=[10, 11, 12],
            answer_ids=[20],  # single token
            logits=torch.randn(1, 1, 100),
        )
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

        result = score_answer_with_cache(cache, 5, "q", "x", model, tokenize, config)
        assert result == 0.0


class TestScoreAnswerMultiToken:
    def test_multi_token_returns_positive_finite(self, config):
        model, tokenize, _, _ = _make_model_and_tokenizer(
            query_ids=[10, 11],
            answer_ids=[20, 21, 22],
            logits=torch.randn(1, 3, 100),
        )
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

        result = score_answer_with_cache(cache, 5, "q", "a b c", model, tokenize, config)
        assert isinstance(result, float)
        assert result > 0
        assert result == result  # not NaN


class TestScoreAnswerAttentionMask:
    def test_attention_mask_sizes(self, config):
        model, tokenize, model_calls, _ = _make_model_and_tokenizer(
            query_ids=[10, 11],     # 2 tokens
            answer_ids=[20, 21, 22],  # 3 tokens
            logits=torch.randn(1, 3, 100),
        )
        context_len = 5
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

        score_answer_with_cache(cache, context_len, "q w", "a b c", model, tokenize, config)

        # Query pass: (1, 5+2=7)
        assert model_calls[0]['attention_mask'].shape == (1, 7)
        # Answer pass: (1, 5+2+3=10)
        assert model_calls[1]['attention_mask'].shape == (1, 10)


class TestScoreAnswerCacheMutation:
    def test_cache_extended_after_query_pass(self, config):
        model, tokenize, model_calls, query_cache = _make_model_and_tokenizer(
            query_ids=[10, 11],
            answer_ids=[20, 21, 22],
            logits=torch.randn(1, 3, 100),
        )
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

        score_answer_with_cache(cache, 5, "q", "a b", model, tokenize, config)

        # Query pass uses use_cache=True
        assert model_calls[0]['use_cache'] is True
        # Answer pass uses use_cache=False, receives extended cache
        assert model_calls[1]['use_cache'] is False
        assert model_calls[1]['past_key_values'] is query_cache


class TestScoreAnswerLogitShift:
    def test_shift_logits_and_labels(self, config):
        vocab_size = 100
        logits = torch.full((1, 4, vocab_size), -10.0)
        # shift_logits[0] predicts token 21, [1]->22, [2]->23
        logits[0, 0, 21] = 10.0
        logits[0, 1, 22] = 10.0
        logits[0, 2, 23] = 10.0

        model, tokenize, _, _ = _make_model_and_tokenizer(
            query_ids=[10, 11],
            answer_ids=[20, 21, 22, 23],
            logits=logits,
        )
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

        nll = score_answer_with_cache(cache, 5, "q", "a b c d", model, tokenize, config)

        # num_scored = 4-1 = 3, with high logits => small NLL
        assert nll > 0
        assert nll < 1.0
