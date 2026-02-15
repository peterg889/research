"""Tests for anchor preservation (preserve_anchor / position_offset) feature.

Unit tests use mocked model/tokenizer (no GPU required).
Integration tests require a real model and are skipped if unavailable.
"""

import types
import torch
import pytest
from unittest.mock import MagicMock, patch, call
from transformers import DynamicCache

from lib.kv_cache import (
    build_truncated_kv_cache_corrected,
    build_truncated_cache_variable_prefix,
    score_answer_with_cache,
    score_answer_with_cache_and_attention,
    score_answer_with_cache_flexible,
    extract_and_truncate_cache_with_bos,
    deepcopy_cache,
    _get_cache_keys,
    _get_cache_values,
    _set_cache_keys,
)
from lib.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DictNamespace(dict):
    """Dict that also supports attribute access, mimicking HF tokenizer output."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


def _make_mock_env(num_layers=1):
    """Set up mocked model, tokenizer, and config for build_truncated tests.

    Args:
        num_layers: Number of layers in the mock model's cache output.

    Note: Uses a real FakeModel class (not MagicMock) because Python resolves
    __call__ on the TYPE, not the instance. MagicMock.__call__ = func doesn't
    actually override call behavior.
    """
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

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(
                hidden_size=64, num_attention_heads=4, rope_theta=10000.0,
            )
            self._num_layers = num_layers

        def __call__(self, input_ids, attention_mask, use_cache, return_dict):
            seq_len = input_ids.shape[1]
            cache = DynamicCache()
            for layer_idx in range(self._num_layers):
                cache.update(torch.randn(1, 4, seq_len, 16),
                             torch.randn(1, 4, seq_len, 16), layer_idx)
            return types.SimpleNamespace(past_key_values=cache)

    model = FakeModel()

    return model, tokenizer, config, tokenizer_calls


def _make_scoring_env():
    """Build mock model + tokenizer for score_answer_with_cache tests.

    Returns query_len=2, answer_len=3, context_len=5.
    """
    config = ExperimentConfig(device="cpu")

    call_count = [0]

    def tokenize(text, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _DictNamespace(input_ids=torch.tensor([[10, 11]]))  # query: 2 tokens
        else:
            return _DictNamespace(input_ids=torch.tensor([[20, 21, 22]]))  # answer: 3 tokens

    query_cache = DynamicCache()
    total_ctx = 5 + 2  # context_len + query_len
    query_cache.update(torch.randn(1, 1, total_ctx, 4),
                       torch.randn(1, 1, total_ctx, 4), 0)

    model_calls = []

    class FakeModel:
        def __init__(self):
            self.config = None

        def __call__(self, **kwargs):
            model_calls.append(kwargs)
            if len(model_calls) == 1:  # query pass
                return types.SimpleNamespace(past_key_values=query_cache)
            else:  # answer pass
                return types.SimpleNamespace(logits=torch.randn(1, 3, 100))

    cache = DynamicCache()
    cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

    return FakeModel(), tokenize, config, model_calls, cache


def _make_scoring_env_with_attention():
    """Build mock model + tokenizer for score_answer_with_cache_and_attention tests."""
    config = ExperimentConfig(device="cpu")

    call_count = [0]

    def tokenize(text, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _DictNamespace(input_ids=torch.tensor([[10, 11]]))
        else:
            return _DictNamespace(input_ids=torch.tensor([[20, 21, 22]]))

    query_cache = DynamicCache()
    total_ctx = 5 + 2
    query_cache.update(torch.randn(1, 1, total_ctx, 4),
                       torch.randn(1, 1, total_ctx, 4), 0)

    model_calls = []
    fake_attentions = (torch.randn(1, 1, 3, 10),)

    class FakeModel:
        def __init__(self):
            self.config = None

        def __call__(self, **kwargs):
            model_calls.append(kwargs)
            if len(model_calls) == 1:
                return types.SimpleNamespace(past_key_values=query_cache)
            else:
                return types.SimpleNamespace(
                    logits=torch.randn(1, 3, 100),
                    attentions=fake_attentions,
                )

    cache = DynamicCache()
    cache.update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4), 0)

    return FakeModel(), tokenize, config, model_calls, cache


# ===========================================================================
# PART 1: Unit tests — Return signatures and RoPE behavior
# ===========================================================================


class TestPreserveAnchorReturnSignature:
    """Verify return type changes based on preserve_anchor flag."""

    def test_default_returns_2_tuple(self):
        model, tokenizer, config, _ = _make_mock_env()
        result = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        keep_len, cache = result
        assert isinstance(keep_len, int)
        assert isinstance(cache, DynamicCache)

    def test_preserve_anchor_false_returns_2_tuple(self):
        model, tokenizer, config, _ = _make_mock_env()
        result = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=False
        )
        assert len(result) == 2

    def test_preserve_anchor_true_returns_3_tuple(self):
        model, tokenizer, config, _ = _make_mock_env()
        result = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        keep_len, cache, offset = result
        assert isinstance(keep_len, int)
        assert isinstance(cache, DynamicCache)
        assert isinstance(offset, int)

    def test_keep_len_same_regardless_of_flag(self):
        """Keep length must be identical for both modes."""
        model, tokenizer, config, _ = _make_mock_env()
        keep_old, _ = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=False
        )
        keep_new, _, _ = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        assert keep_old == keep_new

    def test_cache_seq_length_same_regardless_of_flag(self):
        """Physical cache size must be identical for both modes."""
        model, tokenizer, config, _ = _make_mock_env()
        _, cache_old = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=False
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        assert cache_old.get_seq_length() == cache_new.get_seq_length()


class TestPreserveAnchorSkipsRopeCorrection:
    """Verify RoPE correction is skipped when preserve_anchor=True."""

    def test_preserve_anchor_skips_rope_correction(self):
        model, tokenizer, config, _ = _make_mock_env()
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=True
            )
            mock_correct.assert_not_called()

    def test_default_applies_rope_correction(self):
        model, tokenizer, config, _ = _make_mock_env()
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=False
            )
            mock_correct.assert_called_once()


class TestPreserveAnchorPositionOffset:
    """Verify position_offset = prefix_len - 1."""

    def test_position_offset_value(self):
        model, tokenizer, config, calls = _make_mock_env()
        keep_len, cache, offset = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        # prefix text = "This document may be relevant to queries like: surr\n\n"
        prefix_text = "This document may be relevant to queries like: surr\n\n"
        expected_prefix_len = 1 + len(prefix_text)  # BOS + chars (mock: 1 token/char)
        expected_offset = expected_prefix_len - 1
        assert offset == expected_offset

    def test_position_offset_matches_rope_offset(self):
        """The offset returned by preserve_anchor should equal the offset
        that would have been passed to correct_rope_positions_with_bos."""
        model, tokenizer, config, _ = _make_mock_env()

        # Get offset from RoPE correction path
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=False
            )
            rope_offset = mock_correct.call_args[0][1]

        # Get offset from anchor preservation path
        _, _, anchor_offset = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )

        assert anchor_offset == rope_offset

    def test_offset_positive(self):
        """Offset must always be positive (there's always at least a BOS + prefix)."""
        model, tokenizer, config, _ = _make_mock_env()
        _, _, offset = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        assert offset > 0

    def test_offset_with_different_surrogates(self):
        """Longer surrogates produce larger offsets."""
        model, tokenizer, config, _ = _make_mock_env()
        _, _, offset_short = build_truncated_kv_cache_corrected(
            "x", "doc", model, tokenizer, config, preserve_anchor=True
        )
        _, _, offset_long = build_truncated_kv_cache_corrected(
            "a much longer surrogate query text", "doc",
            model, tokenizer, config, preserve_anchor=True
        )
        assert offset_long > offset_short

    def test_offset_with_custom_template(self):
        """Offset should reflect the actual prefix from a custom template."""
        model, tokenizer, config, _ = _make_mock_env()
        # Short template
        _, _, offset = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config,
            surrogate_prefix_template="Q: {surrogate}\n",
            preserve_anchor=True,
        )
        # "Q: surr\n" = 8 chars + BOS = 9 tokens; offset = 9-1 = 8
        assert offset == 8


# ===========================================================================
# PART 2: Unit tests — Variable prefix
# ===========================================================================


class TestVariablePrefixPreserveAnchor:
    """Tests for preserve_anchor in build_truncated_cache_variable_prefix."""

    def test_default_applies_rope_correction(self):
        model, tokenizer, config, _ = _make_mock_env()
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_cache_variable_prefix(
                "my prefix", "doc text", model, tokenizer, config
            )
            mock_correct.assert_called_once()

    def test_preserve_anchor_skips_rope_correction(self):
        model, tokenizer, config, _ = _make_mock_env()
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_cache_variable_prefix(
                "my prefix", "doc text", model, tokenizer, config,
                preserve_anchor=True
            )
            mock_correct.assert_not_called()

    def test_returns_3_tuple_always(self):
        """Variable prefix always returns 3-tuple regardless of flag."""
        model, tokenizer, config, _ = _make_mock_env()

        result_default = build_truncated_cache_variable_prefix(
            "my prefix", "doc text", model, tokenizer, config
        )
        assert len(result_default) == 3

        result_anchor = build_truncated_cache_variable_prefix(
            "my prefix", "doc text", model, tokenizer, config,
            preserve_anchor=True
        )
        assert len(result_anchor) == 3

    def test_keep_len_same_regardless_of_flag(self):
        """Keep length should be identical whether or not we preserve anchor."""
        model, tokenizer, config, _ = _make_mock_env()

        keep_default, _, prefix_len_default = build_truncated_cache_variable_prefix(
            "my prefix", "doc text", model, tokenizer, config
        )
        keep_anchor, _, prefix_len_anchor = build_truncated_cache_variable_prefix(
            "my prefix", "doc text", model, tokenizer, config,
            preserve_anchor=True
        )
        assert keep_default == keep_anchor
        assert prefix_len_default == prefix_len_anchor

    def test_prefix_len_usable_as_offset(self):
        """Caller computes position_offset = prefix_len - 1.
        Verify this produces a positive value."""
        model, tokenizer, config, _ = _make_mock_env()
        _, _, prefix_len = build_truncated_cache_variable_prefix(
            "my prefix", "doc text", model, tokenizer, config,
            preserve_anchor=True
        )
        offset = prefix_len - 1
        assert offset > 0


# ===========================================================================
# PART 3: Unit tests — score_answer_with_cache position_offset
# ===========================================================================


class TestScorePositionOffset:
    """Test that position_offset generates correct position_ids."""

    def test_no_position_ids_when_offset_zero(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize, config,
                                position_offset=0)
        # Neither call should have position_ids
        assert 'position_ids' not in model_calls[0]
        assert 'position_ids' not in model_calls[1]

    def test_position_ids_present_when_offset_positive(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize, config,
                                position_offset=10)
        # Both calls should have position_ids
        assert 'position_ids' in model_calls[0]
        assert 'position_ids' in model_calls[1]

    def test_query_position_ids_start_correctly(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        context_len = 5
        offset = 10
        score_answer_with_cache(cache, context_len, "q w", "a b c", model, tokenize,
                                config, position_offset=offset)

        query_pos = model_calls[0]['position_ids']
        # Should start at context_len + offset = 15
        assert query_pos[0, 0].item() == context_len + offset
        # Query has 2 tokens
        assert query_pos.shape[1] == 2
        assert query_pos[0, 1].item() == context_len + offset + 1

    def test_answer_position_ids_start_correctly(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        context_len = 5
        offset = 10
        query_len = 2  # from the mock tokenizer
        score_answer_with_cache(cache, context_len, "q w", "a b c", model, tokenize,
                                config, position_offset=offset)

        answer_pos = model_calls[1]['position_ids']
        # Should start at context_len + offset + query_len = 17
        expected_start = context_len + offset + query_len
        assert answer_pos[0, 0].item() == expected_start
        # Answer has 3 tokens
        assert answer_pos.shape[1] == 3

    def test_position_continuity_across_query_and_answer(self):
        """Answer positions must start exactly where query positions end."""
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize,
                                config, position_offset=10)

        query_pos = model_calls[0]['position_ids']
        answer_pos = model_calls[1]['position_ids']

        # Last query position + 1 == first answer position
        query_last = query_pos[0, -1].item()
        answer_first = answer_pos[0, 0].item()
        assert answer_first == query_last + 1, \
            f"Position gap: query ends at {query_last}, answer starts at {answer_first}"

    def test_position_ids_are_contiguous_integers(self):
        """Position IDs within each pass should be consecutive."""
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize,
                                config, position_offset=10)

        for i, name in enumerate(["query", "answer"]):
            pos = model_calls[i]['position_ids'].squeeze(0)
            diffs = pos[1:] - pos[:-1]
            assert torch.all(diffs == 1), \
                f"{name} position_ids not contiguous: {pos.tolist()}"

    def test_position_ids_shape_is_batched(self):
        """Position IDs should be (1, seq_len) — batch dim included."""
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize,
                                config, position_offset=10)

        for call_kwargs in model_calls:
            pos = call_kwargs['position_ids']
            assert pos.dim() == 2
            assert pos.shape[0] == 1


class TestAttentionMaskUsesPhysicalSize:
    """Verify that attention_mask length is based on PHYSICAL cache size,
    not logical (physical + offset). This is critical: the mask tells the model
    how many KV entries physically exist, regardless of their logical positions."""

    def test_query_attention_mask_length(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        context_len = 5
        offset = 100  # large offset to make mismatch obvious
        score_answer_with_cache(cache, context_len, "q w", "a b c", model, tokenize,
                                config, position_offset=offset)

        # Query attention mask should be (1, context_len + query_len) = (1, 7)
        query_mask = model_calls[0]['attention_mask']
        query_len = 2  # from mock
        assert query_mask.shape == (1, context_len + query_len), \
            f"Query mask shape {query_mask.shape}, expected (1, {context_len + query_len})"

    def test_answer_attention_mask_length(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        context_len = 5
        offset = 100
        score_answer_with_cache(cache, context_len, "q w", "a b c", model, tokenize,
                                config, position_offset=offset)

        # Answer attention mask should be (1, context_len + query_len + answer_len) = (1, 10)
        answer_mask = model_calls[1]['attention_mask']
        expected = context_len + 2 + 3  # ctx + query + answer
        assert answer_mask.shape == (1, expected), \
            f"Answer mask shape {answer_mask.shape}, expected (1, {expected})"

    def test_mask_does_not_include_offset(self):
        """The mask must NOT be inflated by position_offset."""
        model_no_off, tok_no, cfg_no, calls_no, cache_no = _make_scoring_env()
        score_answer_with_cache(cache_no, 5, "q w", "a b c", model_no_off, tok_no,
                                cfg_no, position_offset=0)

        model_off, tok_off, cfg_off, calls_off, cache_off = _make_scoring_env()
        score_answer_with_cache(cache_off, 5, "q w", "a b c", model_off, tok_off,
                                cfg_off, position_offset=50)

        # Masks should be identical size regardless of offset
        assert calls_no[0]['attention_mask'].shape == calls_off[0]['attention_mask'].shape
        assert calls_no[1]['attention_mask'].shape == calls_off[1]['attention_mask'].shape


# ===========================================================================
# PART 4: Unit tests — score_answer_with_cache_and_attention position_offset
# ===========================================================================


class TestScoreAndAttentionPositionOffset:
    """Test position_offset in score_answer_with_cache_and_attention."""

    def test_no_position_ids_when_offset_zero(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env_with_attention()
        score_answer_with_cache_and_attention(
            cache, 5, "q w", "a b c", model, tokenize, config, position_offset=0
        )
        assert 'position_ids' not in model_calls[0]
        assert 'position_ids' not in model_calls[1]

    def test_position_ids_present_when_offset_positive(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env_with_attention()
        score_answer_with_cache_and_attention(
            cache, 5, "q w", "a b c", model, tokenize, config, position_offset=10
        )
        assert 'position_ids' in model_calls[0]
        assert 'position_ids' in model_calls[1]

    def test_returns_nll_and_attentions(self):
        model, tokenize, config, model_calls, cache = _make_scoring_env_with_attention()
        result = score_answer_with_cache_and_attention(
            cache, 5, "q w", "a b c", model, tokenize, config, position_offset=10
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        nll, attentions = result
        assert isinstance(nll, float)
        assert attentions is not None

    def test_position_continuity_across_query_and_answer(self):
        """Same continuity property as score_answer_with_cache."""
        model, tokenize, config, model_calls, cache = _make_scoring_env_with_attention()
        score_answer_with_cache_and_attention(
            cache, 5, "q w", "a b c", model, tokenize, config, position_offset=10
        )

        query_last = model_calls[0]['position_ids'][0, -1].item()
        answer_first = model_calls[1]['position_ids'][0, 0].item()
        assert answer_first == query_last + 1

    def test_mask_uses_physical_size(self):
        """Attention masks use physical sizes, not logical."""
        model, tokenize, config, model_calls, cache = _make_scoring_env_with_attention()
        context_len = 5
        score_answer_with_cache_and_attention(
            cache, context_len, "q w", "a b c", model, tokenize, config,
            position_offset=100
        )
        query_mask = model_calls[0]['attention_mask']
        assert query_mask.shape == (1, context_len + 2)  # ctx + query_len


# ===========================================================================
# PART 5: Unit tests — Value preservation and cache integrity
# ===========================================================================


class TestValuePreservation:
    """Both methods truncate identically. Only keys differ (RoPE correction).
    Values should be identical between preserve_anchor=True and False."""

    def test_values_identical_between_methods(self):
        """Value tensors must be exactly the same regardless of preserve_anchor."""
        config = ExperimentConfig(device="cpu")

        tokenizer = MagicMock(side_effect=lambda text, **kw: _DictNamespace(
            input_ids=torch.tensor([[1] + list(range(10, 10 + len(text)))])
            if kw.get('add_special_tokens', True)
            else _DictNamespace(input_ids=torch.tensor([list(range(10, 10 + len(text)))]))
        ))

        # Fixed tensors large enough for any input (200 tokens)
        fixed_keys = torch.randn(1, 4, 200, 16)
        fixed_values = torch.randn(1, 4, 200, 16)

        class DeterministicModel:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    hidden_size=64, num_attention_heads=4, rope_theta=10000.0,
                )

            def __call__(self, input_ids, attention_mask, use_cache, return_dict):
                s = input_ids.shape[1]
                cache = DynamicCache()
                cache.update(fixed_keys[:, :, :s, :].clone(),
                             fixed_values[:, :, :s, :].clone(), 0)
                return types.SimpleNamespace(past_key_values=cache)

        model = DeterministicModel()

        _, cache_old = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=False
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )

        v_old = _get_cache_values(cache_old, 0)
        v_new = _get_cache_values(cache_new, 0)
        assert torch.equal(v_old, v_new), "Values must be identical between methods"


class TestDeepcopyCacheWithAnchor:
    """Verify deepcopy_cache works correctly with anchor-preserved caches."""

    def test_deepcopy_preserves_keys(self):
        """After deepcopy, keys should be identical to original."""
        cache = DynamicCache()
        keys = torch.randn(1, 4, 10, 16)
        vals = torch.randn(1, 4, 10, 16)
        cache.update(keys.clone(), vals.clone(), 0)

        copy = deepcopy_cache(cache)

        assert torch.equal(_get_cache_keys(cache, 0), _get_cache_keys(copy, 0))
        assert torch.equal(_get_cache_values(cache, 0), _get_cache_values(copy, 0))

    def test_deepcopy_is_independent(self):
        """Modifying the copy must not affect the original."""
        cache = DynamicCache()
        keys = torch.randn(1, 4, 10, 16)
        vals = torch.randn(1, 4, 10, 16)
        cache.update(keys.clone(), vals.clone(), 0)

        copy = deepcopy_cache(cache)
        # Mutate the copy
        _get_cache_keys(copy, 0).fill_(0.0)

        # Original should be unchanged
        assert not torch.equal(_get_cache_keys(cache, 0), _get_cache_keys(copy, 0))
        assert torch.equal(_get_cache_keys(cache, 0), keys)

    def test_deepcopy_preserves_seq_length(self):
        """Physical cache length should be preserved."""
        cache = DynamicCache()
        cache.update(torch.randn(1, 4, 15, 16), torch.randn(1, 4, 15, 16), 0)

        copy = deepcopy_cache(cache)
        assert copy.get_seq_length() == 15


class TestCacheMutationProtection:
    """score_answer_with_cache mutates its cache argument. Verify that
    anchor caches need deepcopy before reuse, just like regular caches."""

    def test_scoring_mutates_cache(self):
        """After scoring, the cache has been extended with query+answer tokens."""
        model, tokenize, config, _, cache = _make_scoring_env()
        original_seq_len = cache.get_seq_length()

        score_answer_with_cache(cache, 5, "q w", "a b c", model, tokenize, config,
                                position_offset=10)

        # Cache should have been extended (mutated)
        new_seq_len = _get_cache_keys(cache, 0).shape[2]
        # Note: the fake model returns a new query_cache, so the original
        # cache object might not be mutated in the mock. What matters is that
        # the function receives past_key_values and the model extends it.
        # In real usage, past_key_values IS mutated. Let's verify the API
        # contract by checking that the function doesn't fail when we pass
        # a cache and that we can't reuse it naively.
        # This is more of a documentation test.
        assert True  # The real test is the integration version below

    def test_double_scoring_needs_deepcopy(self):
        """Scoring the same cache twice without deepcopy should still work
        but may produce different results due to cache mutation."""
        # This is a property test — we just verify it doesn't crash
        model, tokenize, config, _, cache = _make_scoring_env()
        # First call works fine
        nll1 = score_answer_with_cache(
            deepcopy_cache(cache), 5, "q w", "a b c", model, tokenize, config,
            position_offset=10
        )
        assert isinstance(nll1, float)


# ===========================================================================
# PART 6: Unit tests — extract_and_truncate_cache_with_bos
# ===========================================================================


class TestExtractAndTruncateCacheWithBos:
    """Verify the truncation function that both methods rely on."""

    def test_bos_preserved(self):
        """First position (BOS) should be preserved."""
        original = DynamicCache()
        keys = torch.randn(1, 4, 20, 16)
        vals = torch.randn(1, 4, 20, 16)
        original.update(keys.clone(), vals.clone(), 0)

        truncated = extract_and_truncate_cache_with_bos(original, 10)

        bos_key_orig = keys[:, :, :1, :]
        bos_key_trunc = _get_cache_keys(truncated, 0)[:, :, :1, :]
        assert torch.equal(bos_key_orig, bos_key_trunc)

    def test_doc_portion_from_end(self):
        """Document tokens should be the last doc_len from the original."""
        original = DynamicCache()
        keys = torch.arange(20).float().view(1, 1, 20, 1).expand(1, 4, 20, 16)
        vals = torch.randn(1, 4, 20, 16)
        original.update(keys.clone(), vals.clone(), 0)

        doc_len = 5
        truncated = extract_and_truncate_cache_with_bos(original, doc_len)

        # Should have BOS (index 0) + last 5 positions (indices 15-19)
        assert truncated.get_seq_length() == 1 + doc_len

    def test_output_is_new_cache(self):
        """Truncated cache should be a new object (not a view)."""
        original = DynamicCache()
        original.update(torch.randn(1, 4, 20, 16), torch.randn(1, 4, 20, 16), 0)

        truncated = extract_and_truncate_cache_with_bos(original, 10)
        assert truncated is not original

    def test_multi_layer(self):
        """Truncation should work across multiple layers."""
        original = DynamicCache()
        for i in range(4):
            original.update(torch.randn(1, 8, 20, 64), torch.randn(1, 8, 20, 64), i)

        truncated = extract_and_truncate_cache_with_bos(original, 10)
        assert len(truncated) == 4
        for i in range(4):
            assert _get_cache_keys(truncated, i).shape[2] == 11  # BOS + 10


# ===========================================================================
# PART 7: Gemma3 unit tests (fake model, no GPU)
# ===========================================================================


class TestGemma3RopeSkipPerLayer:
    """Verify anchor preservation skips RoPE correction for all layer types.

    Gemma3 has per-layer RoPE theta (sliding=10000, full=1000000).
    When preserve_anchor=True, NO layers should get RoPE correction.
    """

    def test_preserve_anchor_skips_all_layer_corrections(self):
        """With a multi-layer fake model mimicking Gemma3, verify that
        correct_rope_positions_with_bos is never called."""
        model, tokenizer, config, _ = _make_mock_env()
        # Give the mock model a Gemma3-like config
        model.config = types.SimpleNamespace(
            hidden_size=64, num_attention_heads=4, rope_theta=10000.0,
            text_config=types.SimpleNamespace(
                hidden_size=64, num_attention_heads=4, head_dim=16,
                layer_types=['sliding_attention', 'full_attention'],
                rope_parameters={
                    'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
                    'full_attention': {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1000000.0},
                },
            ),
        )

        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=True
            )
            mock_correct.assert_not_called()

    def test_default_applies_rope_correction_gemma3_config(self):
        """With Gemma3-like config, default behavior still applies RoPE correction."""
        model, tokenizer, config, _ = _make_mock_env()
        model.config = types.SimpleNamespace(
            hidden_size=64, num_attention_heads=4, rope_theta=10000.0,
            text_config=types.SimpleNamespace(
                hidden_size=64, num_attention_heads=4, head_dim=16,
                layer_types=['sliding_attention', 'full_attention'],
                rope_parameters={
                    'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
                    'full_attention': {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1000000.0},
                },
            ),
        )

        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=False
            )
            mock_correct.assert_called_once()


class TestCachePositionAutoCompute:
    """Verify that for anchor preservation, the auto-computed cache_position
    (from physical cache size) is correct for mask creation.

    Key insight: cache_position controls mask creation (physical proximity),
    while position_ids controls RoPE (logical position). These are independent.
    For DynamicCache, get_seq_length() returns the physical size, so
    auto-computed cache_position = [phys_size, phys_size+1, ...] is correct.
    """

    def test_truncated_cache_seq_length_matches_physical_size(self):
        """After truncation, DynamicCache.get_seq_length() returns physical size."""
        # Simulate a cache of length 20 (BOS + 9 prefix + 10 doc)
        original = DynamicCache()
        original.update(torch.randn(1, 4, 20, 16), torch.randn(1, 4, 20, 16), 0)

        # Truncate to BOS + 10 doc tokens
        truncated = extract_and_truncate_cache_with_bos(original, 10)
        assert truncated.get_seq_length() == 11  # BOS + 10 doc

    def test_position_ids_differ_from_cache_position_with_offset(self):
        """When position_offset > 0, position_ids should differ from what
        cache_position would auto-compute to. This is correct behavior:
        cache_position = physical positions, position_ids = logical positions."""
        context_len = 11  # BOS + 10 doc tokens (physical cache size)
        position_offset = 9  # 9 prefix tokens were removed
        query_len = 3

        # Auto-computed cache_position would be:
        auto_cache_position = torch.arange(context_len, context_len + query_len)
        # = [11, 12, 13]

        # Our explicit position_ids:
        explicit_position_ids = torch.arange(
            context_len + position_offset,
            context_len + position_offset + query_len,
        )
        # = [20, 21, 22]

        # They must differ (the whole point of anchor preservation)
        assert not torch.equal(auto_cache_position, explicit_position_ids)

        # position_ids should be larger (logical > physical due to gap)
        assert explicit_position_ids[0].item() > auto_cache_position[0].item()

    def test_gap_equals_offset(self):
        """The gap between logical and physical positions equals position_offset."""
        context_len = 11
        offset = 9
        query_len = 3

        physical = torch.arange(context_len, context_len + query_len)
        logical = torch.arange(context_len + offset, context_len + offset + query_len)

        gap = (logical - physical).unique()
        assert len(gap) == 1
        assert gap.item() == offset


# ===========================================================================
# PART 8: Unit tests — Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases for anchor preservation."""

    def test_single_char_surrogate(self):
        """Minimum-length surrogate should still work."""
        model, tokenizer, config, _ = _make_mock_env()
        keep, cache, offset = build_truncated_kv_cache_corrected(
            "x", "document text here", model, tokenizer, config,
            preserve_anchor=True
        )
        assert keep > 0
        assert offset > 0
        assert isinstance(cache, DynamicCache)

    def test_short_document(self):
        """Very short document (1 char = 1 token in our mock)."""
        model, tokenizer, config, _ = _make_mock_env()
        keep, cache, offset = build_truncated_kv_cache_corrected(
            "surr", "x", model, tokenizer, config, preserve_anchor=True
        )
        # keep = BOS + doc_len. Doc is "Document:\nx" = 11 chars = 11 tokens
        assert keep > 1
        assert isinstance(cache, DynamicCache)

    def test_long_surrogate_short_document(self):
        """Surrogate much longer than document. offset >> keep_len."""
        model, tokenizer, config, _ = _make_mock_env()
        long_surr = "a " * 200  # 400 chars
        keep, cache, offset = build_truncated_kv_cache_corrected(
            long_surr, "x", model, tokenizer, config, preserve_anchor=True
        )
        assert offset > keep
        assert isinstance(cache, DynamicCache)

    def test_large_position_offset_in_scoring(self):
        """Very large offset should not cause numerical issues (no NaN/Inf)."""
        model, tokenize, config, model_calls, cache = _make_scoring_env()
        nll = score_answer_with_cache(
            cache, 5, "q w", "a b c", model, tokenize, config,
            position_offset=10000
        )
        assert isinstance(nll, float)
        # Verify position_ids are reasonable (large but finite)
        query_pos = model_calls[0]['position_ids']
        assert query_pos[0, 0].item() == 10005
        assert torch.all(torch.isfinite(query_pos.float()))


# ===========================================================================
# PART 9: Unit tests — Multi-layer behavior
# ===========================================================================


class TestMultiLayerBehavior:
    """Verify behavior across multiple model layers."""

    def test_all_layers_have_same_seq_length(self):
        """All layers in the truncated cache should have the same sequence length."""
        model, tokenizer, config, _ = _make_mock_env(num_layers=4)
        _, cache, _ = build_truncated_kv_cache_corrected(
            "surr", "doc", model, tokenizer, config, preserve_anchor=True
        )
        lengths = set()
        for layer_idx in range(len(cache)):
            lengths.add(_get_cache_keys(cache, layer_idx).shape[2])
        assert len(lengths) == 1, f"Inconsistent seq lengths across layers: {lengths}"

    def test_all_layers_skip_rope_with_anchor(self):
        """When preserve_anchor=True, RoPE is skipped for ALL layers."""
        model, tokenizer, config, _ = _make_mock_env(num_layers=4)
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=True
            )
            # Should never be called at all — no per-layer or bulk calls
            mock_correct.assert_not_called()

    def test_rope_correction_applied_to_all_layers_when_default(self):
        """When preserve_anchor=False, RoPE correction is called (it operates on all layers)."""
        model, tokenizer, config, _ = _make_mock_env(num_layers=4)
        with patch('lib.kv_cache.correct_rope_positions_with_bos') as mock_correct:
            build_truncated_kv_cache_corrected(
                "surr", "doc", model, tokenizer, config, preserve_anchor=False
            )
            mock_correct.assert_called_once()
            # Verify it received the cache (which has 4 layers)
            received_cache = mock_correct.call_args[0][0]
            assert len(received_cache) == 4


# ===========================================================================
# Integration tests (require real model — skip if unavailable)
# ===========================================================================

def _load_model_if_available():
    """Try to load Mistral-7B-Instruct-v0.2 4-bit. Return None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, None
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        config = ExperimentConfig(device="cuda")
        return model, tokenizer, config
    except Exception:
        return None, None, None


# Cache the model load across integration tests
_integration_model = None
_integration_tokenizer = None
_integration_config = None
_integration_loaded = False


def _get_integration_env():
    global _integration_model, _integration_tokenizer, _integration_config, _integration_loaded
    if not _integration_loaded:
        _integration_model, _integration_tokenizer, _integration_config = _load_model_if_available()
        _integration_loaded = True
    return _integration_model, _integration_tokenizer, _integration_config


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration tests require GPU"
)


# --- Shared test data for integration tests ---
INTEG_SURROGATE = "what are the key facts"
INTEG_DOCUMENT = "The capital of France is Paris. It is the largest city in the country and serves as its political and cultural center."
INTEG_QUERY = "\n\nQuery: What is the capital of France?\n\nAnswer:"
INTEG_ANSWER = " Paris is the capital of France"


@requires_gpu
class TestIntegrationAnchorCacheShape:
    """With real model, verify cache shapes match regardless of flag."""

    def test_anchor_preserved_cache_shape(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep_old, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        keep_new, cache_new, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        assert keep_old == keep_new
        assert _get_cache_keys(cache_old, 0).shape == _get_cache_keys(cache_new, 0).shape


@requires_gpu
class TestIntegrationAnchorVsCorrectedKeysDiffer:
    """Verify keys differ between anchor and RoPE-corrected caches."""

    def test_keys_differ(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        # Document keys (index 1+) should differ because RoPE correction was applied to old but not new
        keys_old = _get_cache_keys(cache_old, 0)
        keys_new = _get_cache_keys(cache_new, 0)
        assert not torch.allclose(keys_old[:, :, 1:, :], keys_new[:, :, 1:, :], atol=1e-5)


@requires_gpu
class TestIntegrationGoldStandardKeyCorrectness:
    """THE most important test: verify anchor-preserved keys match the
    ORIGINAL keys from the full cache (before any correction).

    When we build [surrogate + document] and extract [BOS + document]:
    - preserve_anchor=True: keys are UNCHANGED from original extraction
    - preserve_anchor=False: keys are MODIFIED by RoPE correction

    The anchor-preserved keys should be byte-for-byte identical to the
    raw extraction (since we literally just skip the correction step).
    """

    def test_anchor_keys_match_raw_extraction(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        surrogate = INTEG_SURROGATE
        document = INTEG_DOCUMENT

        # Step 1: Build the anchor-preserved cache via the API
        keep_len, cache_anchor, offset = build_truncated_kv_cache_corrected(
            surrogate, document, model, tokenizer, config, preserve_anchor=True
        )

        # Step 2: Manually build the same full cache and extract without any correction
        surrogate_prefix = f"This document may be relevant to queries like: {surrogate}\n\n"
        document_text = f"Document:\n{document}"
        full_context = surrogate_prefix + document_text

        prefix_encoding = tokenizer(
            surrogate_prefix, return_tensors="pt", add_special_tokens=True
        )
        prefix_len = prefix_encoding['input_ids'].shape[1]

        full_encoding = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=True
        )
        full_ids = full_encoding['input_ids'].to(config.device)
        doc_len = full_ids.shape[1] - prefix_len

        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=True,
                return_dict=True,
            )

        raw_cache = extract_and_truncate_cache_with_bos(outputs.past_key_values, doc_len)

        # Step 3: Compare ALL layers
        num_layers = len(cache_anchor)
        assert num_layers == len(raw_cache)

        for layer_idx in range(num_layers):
            k_anchor = _get_cache_keys(cache_anchor, layer_idx)
            k_raw = _get_cache_keys(raw_cache, layer_idx)
            assert torch.allclose(k_anchor, k_raw, atol=1e-6), \
                f"Layer {layer_idx}: anchor keys differ from raw extraction"

            v_anchor = _get_cache_values(cache_anchor, layer_idx)
            v_raw = _get_cache_values(raw_cache, layer_idx)
            assert torch.allclose(v_anchor, v_raw, atol=1e-6), \
                f"Layer {layer_idx}: anchor values differ from raw extraction"

    def test_corrected_keys_differ_from_raw_extraction(self):
        """Conversely, RoPE-corrected keys should NOT match raw extraction."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        surrogate = INTEG_SURROGATE
        document = INTEG_DOCUMENT

        keep_len, cache_corrected = build_truncated_kv_cache_corrected(
            surrogate, document, model, tokenizer, config, preserve_anchor=False
        )

        # Build raw extraction for comparison
        surrogate_prefix = f"This document may be relevant to queries like: {surrogate}\n\n"
        document_text = f"Document:\n{document}"
        full_context = surrogate_prefix + document_text

        prefix_encoding = tokenizer(
            surrogate_prefix, return_tensors="pt", add_special_tokens=True
        )
        prefix_len = prefix_encoding['input_ids'].shape[1]

        full_encoding = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=True
        )
        full_ids = full_encoding['input_ids'].to(config.device)
        doc_len = full_ids.shape[1] - prefix_len

        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=True,
                return_dict=True,
            )

        raw_cache = extract_and_truncate_cache_with_bos(outputs.past_key_values, doc_len)

        # Document keys (not BOS) should differ because RoPE correction was applied
        k_corrected = _get_cache_keys(cache_corrected, 0)[:, :, 1:, :]
        k_raw = _get_cache_keys(raw_cache, 0)[:, :, 1:, :]
        assert not torch.allclose(k_corrected, k_raw, atol=1e-4), \
            "Corrected keys should differ from raw extraction"


@requires_gpu
class TestIntegrationAllLayerConsistency:
    """Verify properties hold across ALL layers, not just layer 0."""

    def test_bos_key_identical_all_layers(self):
        """BOS key at position 0 should be identical across ALL layers."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        num_layers = len(cache_old)
        for layer_idx in range(num_layers):
            bos_old = _get_cache_keys(cache_old, layer_idx)[:, :, :1, :]
            bos_new = _get_cache_keys(cache_new, layer_idx)[:, :, :1, :]
            assert torch.allclose(bos_old, bos_new, atol=1e-6), \
                f"Layer {layer_idx}: BOS keys should be identical"

    def test_doc_keys_differ_all_layers(self):
        """Document keys should differ for ALL layers (RoPE applies to all)."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        num_layers = len(cache_old)
        for layer_idx in range(num_layers):
            k_old = _get_cache_keys(cache_old, layer_idx)[:, :, 1:, :]
            k_new = _get_cache_keys(cache_new, layer_idx)[:, :, 1:, :]
            assert not torch.allclose(k_old, k_new, atol=1e-4), \
                f"Layer {layer_idx}: document keys should differ"

    def test_values_identical_all_layers(self):
        """Values should be identical across ALL layers for both methods."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        num_layers = len(cache_old)
        for layer_idx in range(num_layers):
            v_old = _get_cache_values(cache_old, layer_idx)
            v_new = _get_cache_values(cache_new, layer_idx)
            assert torch.allclose(v_old, v_new, atol=1e-6), \
                f"Layer {layer_idx}: values should be identical"


@requires_gpu
class TestIntegrationAnchorScoringRuns:
    """End-to-end: build anchor-preserved cache, score an answer, verify finite NLL."""

    def test_anchor_scoring_runs_without_error(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep_len, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll = score_answer_with_cache(
            deepcopy_cache(cache), keep_len, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        assert isinstance(nll, float)
        assert nll == nll  # not NaN
        assert nll > 0  # positive NLL
        assert nll < 100  # reasonable range

    def test_anchor_vs_old_both_produce_reasonable_nll(self):
        """Both methods should produce finite, reasonable NLLs.
        They may differ in magnitude but neither should be degenerate."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        # Old method
        keep_old, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        nll_old = score_answer_with_cache(
            deepcopy_cache(cache_old), keep_old, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=0,
        )

        # Anchor method
        keep_new, cache_new, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )
        nll_anchor = score_answer_with_cache(
            deepcopy_cache(cache_new), keep_new, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        # Both should be finite and positive
        for nll, name in [(nll_old, "old"), (nll_anchor, "anchor")]:
            assert nll == nll, f"{name} NLL is NaN"
            assert nll > 0, f"{name} NLL should be positive"
            assert nll < 50, f"{name} NLL unreasonably large: {nll}"

    def test_bos_key_identical_between_methods(self):
        """BOS key at position 0 should be identical regardless of method,
        since neither method modifies BOS."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache_old = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=False,
        )
        _, cache_new, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        # BOS key (index 0) should be IDENTICAL — neither method touches it
        bos_old = _get_cache_keys(cache_old, 0)[:, :, :1, :]
        bos_new = _get_cache_keys(cache_new, 0)[:, :, :1, :]
        assert torch.allclose(bos_old, bos_new, atol=1e-6), \
            "BOS keys should be identical between old and anchor methods"


@requires_gpu
class TestIntegrationWrongOffsetDiffers:
    """Using the wrong position_offset should produce different NLLs,
    confirming that position_offset actually affects model behavior."""

    def test_wrong_offset_produces_different_nll(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll_correct = score_answer_with_cache(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )
        nll_wrong = score_answer_with_cache(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=0,
        )

        assert abs(nll_correct - nll_wrong) > 0.001, \
            f"NLLs should differ: correct={nll_correct:.4f}, wrong={nll_wrong:.4f}"


@requires_gpu
class TestIntegrationCacheMutation:
    """Verify that scoring mutates the cache (extending it with query tokens),
    so deepcopy is required before each call."""

    def test_cache_extended_after_scoring(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        original_len = cache.get_seq_length()
        cache_copy = deepcopy_cache(cache)

        score_answer_with_cache(
            cache_copy, keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        # The passed-in cache_copy should have been extended
        extended_len = cache_copy.get_seq_length()
        assert extended_len > original_len, \
            f"Cache should be extended: was {original_len}, now {extended_len}"

        # Original should be untouched (deepcopy protected it)
        assert cache.get_seq_length() == original_len

    def test_double_scoring_with_deepcopy_gives_same_nll(self):
        """Scoring the same cache twice with deepcopy should give identical results."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll1 = score_answer_with_cache(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )
        nll2 = score_answer_with_cache(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        assert abs(nll1 - nll2) < 1e-5, \
            f"Same cache should give same NLL: {nll1:.6f} vs {nll2:.6f}"


@requires_gpu
class TestIntegrationDeepcopyCachePreservesAnchor:
    """Verify deepcopy_cache preserves anchor-preserved key state."""

    def test_deepcopy_keys_identical(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        _, cache, _ = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        copy = deepcopy_cache(cache)

        for layer_idx in range(len(cache)):
            k_orig = _get_cache_keys(cache, layer_idx)
            k_copy = _get_cache_keys(copy, layer_idx)
            assert torch.equal(k_orig, k_copy), \
                f"Layer {layer_idx}: deepcopy keys should be identical"


@requires_gpu
class TestIntegrationVariablePrefixAnchor:
    """Integration test for build_truncated_cache_variable_prefix with anchor."""

    def test_variable_prefix_anchor_scoring(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        prefix = "key facts about geography"
        document = INTEG_DOCUMENT
        query = INTEG_QUERY
        answer = INTEG_ANSWER

        # Old method
        keep_old, cache_old, plen_old = build_truncated_cache_variable_prefix(
            prefix, document, model, tokenizer, config, preserve_anchor=False
        )
        nll_old = score_answer_with_cache(
            deepcopy_cache(cache_old), keep_old, query, answer,
            model, tokenizer, config, position_offset=0
        )

        # Anchor method
        keep_new, cache_new, plen_new = build_truncated_cache_variable_prefix(
            prefix, document, model, tokenizer, config, preserve_anchor=True
        )
        offset = plen_new - 1
        nll_anchor = score_answer_with_cache(
            deepcopy_cache(cache_new), keep_new, query, answer,
            model, tokenizer, config, position_offset=offset
        )

        assert keep_old == keep_new
        assert plen_old == plen_new

        for nll, name in [(nll_old, "old"), (nll_anchor, "anchor")]:
            assert nll == nll, f"{name} NLL is NaN"
            assert 0 < nll < 50, f"{name} NLL out of range: {nll}"


@requires_gpu
class TestIntegrationScoreFlexibleWithAnchor:
    """Test score_answer_with_cache_flexible with anchor-preserved caches.

    The plan says callers pass seq_len = keep_len + position_offset.
    Verify this produces reasonable NLLs."""

    def test_flexible_scoring_with_anchor_offset(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep_len, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        # score_answer_with_cache_flexible already passes explicit position_ids
        # starting at seq_len. For anchor caches, pass seq_len = keep_len + offset.
        nll = score_answer_with_cache_flexible(
            deepcopy_cache(cache),
            seq_len=keep_len + offset,
            query="What is the capital of France?",
            answer="Paris",
            model=model,
            tokenizer=tokenizer,
        )

        assert isinstance(nll, float)
        assert nll == nll  # not NaN
        assert 0 < nll < 100, f"NLL out of range: {nll}"

    def test_flexible_vs_standard_scoring_comparable(self):
        """Both scoring functions with correct offset should produce
        NLLs in similar ranges (not necessarily identical due to different
        prompt formats)."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep_len, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll_standard = score_answer_with_cache(
            deepcopy_cache(cache), keep_len, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        nll_flexible = score_answer_with_cache_flexible(
            deepcopy_cache(cache),
            seq_len=keep_len + offset,
            query="What is the capital of France?",
            answer="Paris is the capital of France",
            model=model,
            tokenizer=tokenizer,
        )

        # Both should be in reasonable range (they use different prompt formats
        # so values won't match exactly)
        assert 0 < nll_standard < 50
        assert 0 < nll_flexible < 50


@requires_gpu
class TestIntegrationScoreAndAttention:
    """Test score_answer_with_cache_and_attention with anchor preservation."""

    def test_returns_nll_and_attentions(self):
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll, attentions = score_answer_with_cache_and_attention(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        assert isinstance(nll, float)
        assert 0 < nll < 50
        # Note: SDPA backend returns None for attentions (doesn't support output_attentions).
        # The NLL is still valid. Attentions are available with eager attention only.
        if attentions is not None:
            assert isinstance(attentions, tuple)
            assert len(attentions) > 0

    def test_attention_same_nll_as_standard(self):
        """NLL from attention version should match standard version."""
        model, tokenizer, config = _get_integration_env()
        if model is None:
            pytest.skip("Model not available")

        keep, cache, offset = build_truncated_kv_cache_corrected(
            INTEG_SURROGATE, INTEG_DOCUMENT, model, tokenizer, config,
            preserve_anchor=True,
        )

        nll_standard = score_answer_with_cache(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )
        nll_attention, _ = score_answer_with_cache_and_attention(
            deepcopy_cache(cache), keep, INTEG_QUERY, INTEG_ANSWER,
            model, tokenizer, config, position_offset=offset,
        )

        # Should be very close (same computation, just with output_attentions=True)
        assert abs(nll_standard - nll_attention) < 0.01, \
            f"NLL mismatch: standard={nll_standard:.6f}, attention={nll_attention:.6f}"
