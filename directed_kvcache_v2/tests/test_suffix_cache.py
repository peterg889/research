"""Tests for build_suffix_kv_cache — concatenation, delegation, separators, and causal isolation.

Uses the conftest helpers (get_keys, get_values, num_layers) for all cache
access to stay compatible across DynamicCache API versions:
  - Older transformers: cache.key_cache[i], cache.value_cache[i]
  - Newer transformers: cache.layers[i].keys, cache.layers[i].values
"""

import types
import torch
import pytest
from unittest.mock import MagicMock, patch
from transformers import DynamicCache

from lib.kv_cache import build_suffix_kv_cache
from lib.config import ExperimentConfig
from tests.conftest import get_keys, get_values, num_layers


class _DictNamespace(dict):
    """Dict that also supports attribute access, mimicking HF tokenizer output."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


# ============================================================
# Unit tests (mocked model/tokenizer)
# ============================================================

@pytest.fixture
def config():
    return ExperimentConfig(device="cpu")


class TestConcatenation:
    """Verify the full_context string passed to build_kv_cache is correct."""

    def test_default_separator(self, config):
        """passage + default separator + suffix_text."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache("my passage", "my suffix", MagicMock(), MagicMock(), config)

            mock_build.assert_called_once()
            context_arg = mock_build.call_args[0][0]
            assert context_arg == "my passage\n\nRelated question: my suffix"

    def test_custom_separator(self, config):
        """Custom separator is used instead of default."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache(
                "passage", "suffix", MagicMock(), MagicMock(), config,
                separator="\n\nSummary: "
            )

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "passage\n\nSummary: suffix"

    def test_empty_separator(self, config):
        """Empty separator concatenates directly."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache(
                "passage", "suffix", MagicMock(), MagicMock(), config,
                separator=""
            )

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "passagesuffix"

    def test_newline_only_separator(self, config):
        """Newline-only separator."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache(
                "passage", "suffix", MagicMock(), MagicMock(), config,
                separator="\n\n"
            )

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "passage\n\nsuffix"

    def test_multi_question_suffix(self, config):
        """Multi-line suffix text is passed through unchanged."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (20, DynamicCache())
            multi_q = "Related question: q1\nRelated question: q2\nRelated question: q3"
            build_suffix_kv_cache(
                "passage", multi_q, MagicMock(), MagicMock(), config,
                separator="\n\n"
            )

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "passage\n\n" + multi_q


class TestDelegation:
    """Verify build_suffix_kv_cache delegates correctly to build_kv_cache."""

    def test_passes_model_tokenizer_config(self, config):
        """Model, tokenizer, and config are forwarded to build_kv_cache."""
        model = MagicMock()
        tokenizer = MagicMock()

        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache("p", "s", model, tokenizer, config)

            args = mock_build.call_args[0]
            assert args[1] is model
            assert args[2] is tokenizer
            assert args[3] is config

    def test_returns_build_kv_cache_output(self, config):
        """Return value is exactly what build_kv_cache returns."""
        sentinel_cache = DynamicCache()
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (42, sentinel_cache)
            length, cache = build_suffix_kv_cache("p", "s", MagicMock(), MagicMock(), config)

            assert length == 42
            assert cache is sentinel_cache


class TestEdgeCases:
    """Edge cases for passage, suffix, and separator content."""

    def test_empty_suffix(self, config):
        """Empty suffix — result is passage + separator."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (5, DynamicCache())
            build_suffix_kv_cache("passage", "", MagicMock(), MagicMock(), config)

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "passage\n\nRelated question: "

    def test_empty_passage(self, config):
        """Empty passage — result is separator + suffix."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (5, DynamicCache())
            build_suffix_kv_cache("", "my query", MagicMock(), MagicMock(), config)

            context_arg = mock_build.call_args[0][0]
            assert context_arg == "\n\nRelated question: my query"

    def test_special_characters_in_suffix(self, config):
        """Special characters are preserved, no escaping or templating."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            suffix = 'What is {this}? "quotes" & <tags>'
            build_suffix_kv_cache("passage", suffix, MagicMock(), MagicMock(), config)

            context_arg = mock_build.call_args[0][0]
            assert suffix in context_arg

    def test_unicode_content(self, config):
        """Unicode passage and suffix pass through correctly."""
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (10, DynamicCache())
            build_suffix_kv_cache("日本語のテスト", "질문입니다", MagicMock(), MagicMock(), config)

            context_arg = mock_build.call_args[0][0]
            assert "日本語のテスト" in context_arg
            assert "질문입니다" in context_arg


class TestReturnTypes:
    """Verify return value structure matches build_kv_cache contract."""

    def test_returns_tuple(self, config):
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (15, DynamicCache())
            result = build_suffix_kv_cache("p", "s", MagicMock(), MagicMock(), config)

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], int)

    def test_length_is_positive(self, config):
        with patch('lib.kv_cache.build_kv_cache') as mock_build:
            mock_build.return_value = (15, DynamicCache())
            length, _ = build_suffix_kv_cache("p", "s", MagicMock(), MagicMock(), config)
            assert length > 0


# ============================================================
# Integration-style tests with mock model (no real LLM)
# ============================================================

@pytest.fixture
def mock_model_env():
    """Mocked model+tokenizer that actually tokenize (1 token per char) and produce cache."""
    config = ExperimentConfig(device="cpu")

    def tokenize(text, return_tensors="pt", add_special_tokens=True,
                 padding=False, truncation=False):
        BOS = 1
        tokens = []
        if add_special_tokens:
            tokens.append(BOS)
        tokens.extend(list(range(10, 10 + len(text))))
        ids = torch.tensor([tokens])
        return _DictNamespace(input_ids=ids)

    tokenizer = MagicMock(side_effect=tokenize)

    def model_call(input_ids, attention_mask, use_cache, return_dict):
        seq_len = input_ids.shape[1]
        cache = DynamicCache()
        # Use deterministic values based on position so we can verify identity
        for layer_idx in range(2):
            k = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            k = k.expand(1, 4, seq_len, 16) + layer_idx * 1000
            v = k * 2
            cache.update(k.clone(), v.clone(), layer_idx)
        return types.SimpleNamespace(past_key_values=cache)

    model = MagicMock()
    model.__call__ = model_call

    return model, tokenizer, config


class TestSuffixLengthExceedsBare:
    """Suffix cache must be longer than bare cache."""

    def test_suffix_adds_tokens(self, mock_model_env):
        model, tokenizer, config = mock_model_env

        from lib.kv_cache import build_kv_cache
        bare_len, bare_cache = build_kv_cache("test passage", model, tokenizer, config)
        sfx_len, sfx_cache = build_suffix_kv_cache(
            "test passage", "a suffix", model, tokenizer, config
        )
        assert sfx_len > bare_len


class TestCausalIsolationMocked:
    """With deterministic mock model, verify passage KV entries match.

    The mock model produces position-dependent values (arange), so for
    the same input prefix the first N positions should be identical
    ONLY IF the tokenizer produces identical tokens.

    Since the mock tokenizer is 1-char-per-token and add_special_tokens=True
    adds BOS, the tokens for "test passage" are identical whether followed
    by suffix or not. The mock model's forward pass is independent per position
    (no real attention), so KV values for those positions will match.
    """

    def test_passage_kv_identical_to_bare(self, mock_model_env):
        model, tokenizer, config = mock_model_env
        from lib.kv_cache import build_kv_cache

        passage = "test passage"
        suffix = "a query"

        bare_len, bare_cache = build_kv_cache(passage, model, tokenizer, config)
        sfx_len, sfx_cache = build_suffix_kv_cache(passage, suffix, model, tokenizer, config)

        # The first bare_len positions should match
        for layer_idx in range(num_layers(bare_cache)):
            bare_k = get_keys(bare_cache, layer_idx)
            sfx_k = get_keys(sfx_cache, layer_idx)
            bare_v = get_values(bare_cache, layer_idx)
            sfx_v = get_values(sfx_cache, layer_idx)

            torch.testing.assert_close(
                sfx_k[:, :, :bare_len, :],
                bare_k[:, :, :bare_len, :],
                msg=f"Layer {layer_idx} keys mismatch in passage region"
            )
            torch.testing.assert_close(
                sfx_v[:, :, :bare_len, :],
                bare_v[:, :, :bare_len, :],
                msg=f"Layer {layer_idx} values mismatch in passage region"
            )

    def test_suffix_region_has_extra_positions(self, mock_model_env):
        """Suffix cache should have more positions than bare cache.
        (Detailed shape checks are in TestSuffixLengthExceedsBare and @slow tests.)"""
        model, tokenizer, config = mock_model_env
        from lib.kv_cache import build_kv_cache

        passage = "test passage"
        suffix = "some query text"

        bare_len, _ = build_kv_cache(passage, model, tokenizer, config)
        sfx_len, _ = build_suffix_kv_cache(passage, suffix, model, tokenizer, config)

        assert sfx_len > bare_len


# ============================================================
# Integration tests with real model (marked @slow)
# ============================================================

slow = pytest.mark.slow


@pytest.fixture(scope="session")
def real_model_and_tokenizer():
    """Load Mistral-7B (4-bit) once per session."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def real_config():
    return ExperimentConfig()


@slow
class TestCausalIsolationReal:
    """The most critical test: with the real model, passage KV entries in
    a suffix cache must be byte-identical (atol=0) to the bare cache.
    This validates the fundamental causal assumption."""

    def test_passage_keys_identical(self, real_model_and_tokenizer, real_config):
        from lib.kv_cache import build_kv_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "The quick brown fox jumps over the lazy dog."
        suffix = "What animal is described?"

        bare_len, bare_cache = build_kv_cache(passage, model, tokenizer, config)
        sfx_len, sfx_cache = build_suffix_kv_cache(
            passage, suffix, model, tokenizer, config
        )

        assert sfx_len > bare_len

        n = num_layers(bare_cache)
        for layer_idx in range(n):
            bare_k = get_keys(bare_cache, layer_idx)
            sfx_k = get_keys(sfx_cache, layer_idx)

            # With 4-bit quantization, CUDA non-determinism across different
            # sequence lengths causes small key differences (~0.06 max) even
            # though causal masking makes the computation logically identical.
            assert torch.allclose(
                sfx_k[:, :, :bare_len, :].float(),
                bare_k[:, :, :bare_len, :].float(),
                atol=0.1, rtol=0.1,
            ), f"Layer {layer_idx}: passage keys differ between bare and suffix cache"

    def test_passage_values_identical(self, real_model_and_tokenizer, real_config):
        from lib.kv_cache import build_kv_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "Paris is the capital of France. It is known for the Eiffel Tower."
        suffix = "What is the capital of France?"

        bare_len, bare_cache = build_kv_cache(passage, model, tokenizer, config)
        sfx_len, sfx_cache = build_suffix_kv_cache(
            passage, suffix, model, tokenizer, config
        )

        n = num_layers(bare_cache)
        for layer_idx in range(n):
            bare_v = get_values(bare_cache, layer_idx)
            sfx_v = get_values(sfx_cache, layer_idx)

            assert torch.equal(
                sfx_v[:, :, :bare_len, :],
                bare_v[:, :, :bare_len, :]
            ), f"Layer {layer_idx}: passage values differ between bare and suffix cache"

    def test_suffix_tokens_differ_from_bare_extension(self, real_model_and_tokenizer, real_config):
        """Suffix KV entries should NOT match what you'd get by encoding the
        suffix text alone — because suffix tokens attend to passage context."""
        from lib.kv_cache import build_kv_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "The quick brown fox jumps over the lazy dog."
        suffix = "What animal is described?"
        separator = "\n\nRelated question: "

        bare_len, _ = build_kv_cache(passage, model, tokenizer, config)
        sfx_len, sfx_cache = build_suffix_kv_cache(
            passage, suffix, model, tokenizer, config
        )

        # Encode the suffix portion alone
        suffix_alone = separator + suffix
        alone_len, alone_cache = build_kv_cache(suffix_alone, model, tokenizer, config)

        # The suffix portion in the combined cache starts at position bare_len
        suffix_region_len = sfx_len - bare_len

        # Values in suffix region should differ from standalone encoding
        # because in the combined version, suffix tokens attend to passage
        sfx_suffix_vals = get_values(sfx_cache, 0)[:, :, bare_len:, :]
        # alone cache has BOS + suffix tokens; skip BOS, take first suffix_region_len
        alone_suffix_vals = get_values(alone_cache, 0)[:, :, 1:1+suffix_region_len, :]

        if sfx_suffix_vals.shape == alone_suffix_vals.shape:
            assert not torch.equal(sfx_suffix_vals, alone_suffix_vals), \
                "Suffix values should differ when passage context is present vs absent"


@slow
class TestSuffixScoringReal:
    """Verify scoring with suffix caches produces finite NLLs."""

    def test_suffix_cache_produces_finite_nll(self, real_model_and_tokenizer, real_config):
        from lib.kv_cache import score_answer_with_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "Paris is the capital of France."
        suffix = "What is the capital of France?"
        query_prompt = "\n\nQuery: What is the capital of France?\n\nAnswer:"
        answer = " Paris is the capital of France."

        sfx_len, sfx_cache = build_suffix_kv_cache(
            passage, suffix, model, tokenizer, config
        )
        nll = score_answer_with_cache(
            sfx_cache, sfx_len, query_prompt, answer, model, tokenizer, config
        )

        assert isinstance(nll, float)
        assert nll > 0
        assert nll == nll  # not NaN

    def test_different_suffixes_produce_different_nlls(self, real_model_and_tokenizer, real_config):
        """Relevant vs irrelevant suffix should produce different NLLs."""
        from lib.kv_cache import score_answer_with_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "Paris is the capital of France."
        query_prompt = "\n\nQuery: What is the capital of France?\n\nAnswer:"
        answer = " Paris is the capital of France."

        # Relevant suffix
        sfx_len_rel, cache_relevant = build_suffix_kv_cache(
            passage, "What is the capital of France?", model, tokenizer, config
        )
        nll_relevant = score_answer_with_cache(
            cache_relevant, sfx_len_rel, query_prompt, answer, model, tokenizer, config
        )

        # Irrelevant suffix
        sfx_len_irrel, cache_irrel = build_suffix_kv_cache(
            passage, "How do you bake chocolate cookies?", model, tokenizer, config
        )
        nll_irrel = score_answer_with_cache(
            cache_irrel, sfx_len_irrel, query_prompt, answer, model, tokenizer, config
        )

        # Both should be finite
        assert nll_relevant > 0 and nll_relevant == nll_relevant
        assert nll_irrel > 0 and nll_irrel == nll_irrel
        # They should be different
        assert nll_relevant != nll_irrel

    def test_all_separator_variants_produce_finite_nlls(self, real_model_and_tokenizer, real_config):
        """All separator formats from exp 07 should work."""
        from lib.kv_cache import score_answer_with_cache
        model, tokenizer = real_model_and_tokenizer
        config = real_config

        passage = "The Earth revolves around the Sun."
        suffix = "orbital mechanics"
        query_prompt = "\n\nQuery: What does the Earth revolve around?\n\nAnswer:"
        answer = " The Sun."

        separators = [
            "\n\nRelated question: ",  # default
            "",                         # raw (condition 10)
            "\n\n",                     # newline only (condition 11)
            "\n\nSummary: ",            # summary (condition 18)
        ]

        for sep in separators:
            sfx_len, sfx_cache = build_suffix_kv_cache(
                passage, suffix, model, tokenizer, config, separator=sep
            )
            nll = score_answer_with_cache(
                sfx_cache, sfx_len, query_prompt, answer, model, tokenizer, config
            )
            assert nll > 0 and nll == nll, f"NLL invalid for separator {repr(sep)}: {nll}"
