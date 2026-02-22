"""Integration tests requiring Mistral-7B. Marked @pytest.mark.slow."""

import pytest
import torch
from tests.conftest import get_keys

slow = pytest.mark.slow


@pytest.fixture(scope="session")
def model_and_tokenizer():
    """Load Mistral-7B (4-bit) and tokenizer once per session."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def config():
    from lib.config import ExperimentConfig
    return ExperimentConfig()


@slow
def test_build_kv_cache_shape(model_and_tokenizer, config):
    from lib.kv_cache import build_kv_cache

    model, tokenizer = model_and_tokenizer
    context = "Document:\nThe quick brown fox jumps over the lazy dog."
    ctx_len, cache = build_kv_cache(context, model, tokenizer, config)

    assert ctx_len > 0
    # Check shapes: (batch, num_heads, seq_len, head_dim)
    key = get_keys(cache, 0)
    assert key.shape[0] == 1  # batch
    assert key.shape[2] == ctx_len  # seq_len matches


@slow
def test_corrected_cache_matches_direct(model_and_tokenizer, config):
    """Build cache from doc alone vs corrected truncated cache.

    Keys should be positionally correct (similar but not identical due to
    different attention patterns during the forward pass).
    """
    from lib.kv_cache import build_kv_cache, build_truncated_kv_cache_corrected

    model, tokenizer = model_and_tokenizer
    document = "The quick brown fox jumps over the lazy dog."

    # Direct cache from document alone
    doc_context = f"Document:\n{document}"
    direct_len, direct_cache = build_kv_cache(doc_context, model, tokenizer, config)

    # Corrected truncated cache
    surrogate = "fox jumping animals"
    corrected_len, corrected_cache = build_truncated_kv_cache_corrected(
        surrogate, document, model, tokenizer, config
    )

    # build_kv_cache includes BOS token (add_special_tokens=True) so direct_len
    # is 1 more than corrected_len (which counts only doc tokens without BOS).
    # Compare the document portion only.
    assert corrected_len == direct_len - 1  # BOS difference

    # Keys won't be identical (different attention context) but should have
    # same head_dim and no NaN
    dk = get_keys(direct_cache, 0)
    ck = get_keys(corrected_cache, 0)
    assert dk.shape[0] == ck.shape[0]  # batch
    assert dk.shape[1] == ck.shape[1]  # num_heads
    assert dk.shape[3] == ck.shape[3]  # head_dim
    assert not torch.isnan(ck).any()


@slow
def test_score_answer_produces_finite(model_and_tokenizer, config):
    from lib.kv_cache import build_kv_cache, score_answer_with_cache

    model, tokenizer = model_and_tokenizer
    context = "Document:\nParis is the capital of France."
    ctx_len, cache = build_kv_cache(context, model, tokenizer, config)

    nll = score_answer_with_cache(
        cache, ctx_len,
        "\n\nQuery: What is the capital of France?\n\nAnswer:",
        " Paris is the capital of France.",
        model, tokenizer, config,
    )
    assert isinstance(nll, float)
    assert nll > 0
    assert not (nll != nll)  # not NaN


@slow
def test_corrected_vs_uncorrected_nll(model_and_tokenizer, config):
    """Corrected cache should give different NLL than uncorrected (broken positions)."""
    from lib.kv_cache import (
        build_truncated_kv_cache,
        build_truncated_kv_cache_corrected,
        score_answer_with_cache,
    )

    model, tokenizer = model_and_tokenizer
    document = "Paris is the capital of France."
    surrogate = "capital cities Europe"
    query = "\n\nQuery: What is the capital of France?\n\nAnswer:"
    # Use a multi-token answer so score_answer_with_cache computes a
    # non-trivial NLL (single-token answers yield num_scored=0 → 0.0).
    answer = " Paris is the capital of France."

    uncorrected_len, uncorrected_cache = build_truncated_kv_cache(
        surrogate, document, model, tokenizer, config
    )
    corrected_len, corrected_cache = build_truncated_kv_cache_corrected(
        surrogate, document, model, tokenizer, config
    )

    nll_uncorrected = score_answer_with_cache(
        uncorrected_cache, uncorrected_len, query, answer, model, tokenizer, config
    )
    nll_corrected = score_answer_with_cache(
        corrected_cache, corrected_len, query, answer, model, tokenizer, config
    )

    # Both should be finite positive
    assert nll_uncorrected > 0
    assert nll_corrected > 0
    # They should differ (correction changes key positions)
    assert nll_uncorrected != nll_corrected


@slow
def test_corrected_cache_with_custom_templates(model_and_tokenizer, config):
    """build_truncated_kv_cache_corrected with custom templates produces valid cache."""
    from lib.kv_cache import build_truncated_kv_cache_corrected

    model, tokenizer = model_and_tokenizer
    keep_len, cache = build_truncated_kv_cache_corrected(
        "fox jumping animals",
        "The quick brown fox jumps over the lazy dog.",
        model, tokenizer, config,
        surrogate_prefix_template="Query: {surrogate}\n\n",
        document_template="Passage: {document}",
    )

    assert keep_len > 0
    key = get_keys(cache, 0)
    assert not torch.isnan(key).any()
    assert key.shape[2] == keep_len


@slow
def test_score_query_nll_deep_copy_pattern(model_and_tokenizer, config):
    """Scoring mutates the cache via the query pass (use_cache=True).
    Using deep_copy_cache before scoring preserves the original."""
    from lib.kv_cache import build_kv_cache, score_answer_with_cache
    from copy import deepcopy

    model, tokenizer = model_and_tokenizer
    context = "Document:\nParis is the capital of France."
    ctx_len, cache = build_kv_cache(context, model, tokenizer, config)

    # Deep copy before scoring
    cache_copy = deepcopy(cache)

    _ = score_answer_with_cache(
        cache_copy, ctx_len,
        "\n\nQuery: What is the capital of France?\n\nAnswer:",
        " Paris is the capital.",
        model, tokenizer, config,
    )

    # Original cache should be unchanged
    orig_key = get_keys(cache, 0)
    assert orig_key.shape[2] == ctx_len  # not extended


@slow
def test_bare_vs_corrected_different_nlls(model_and_tokenizer, config):
    """Bare (no surrogate) vs truncated+corrected produce different NLLs."""
    from lib.kv_cache import build_kv_cache, build_truncated_kv_cache_corrected, score_answer_with_cache
    from copy import deepcopy

    model, tokenizer = model_and_tokenizer
    document = "Paris is the capital of France."
    query = "\n\nQuery: What is the capital of France?\n\nAnswer:"
    answer = " Paris is the capital of France."

    # Bare: just the document, no framing
    bare_len, bare_cache = build_kv_cache(document, model, tokenizer, config)
    nll_bare = score_answer_with_cache(
        deepcopy(bare_cache), bare_len, query, answer, model, tokenizer, config
    )

    # Truncated+corrected
    surrogate = "capital cities Europe"
    corrected_len, corrected_cache = build_truncated_kv_cache_corrected(
        surrogate, document, model, tokenizer, config
    )
    nll_corrected = score_answer_with_cache(
        deepcopy(corrected_cache), corrected_len, query, answer, model, tokenizer, config
    )

    assert nll_bare > 0
    assert nll_corrected > 0
    assert nll_bare != nll_corrected


@slow
def test_corrected_cache_empty_surrogate(model_and_tokenizer, config):
    """Empty surrogate string should still produce valid cache."""
    from lib.kv_cache import build_truncated_kv_cache_corrected

    model, tokenizer = model_and_tokenizer
    keep_len, cache = build_truncated_kv_cache_corrected(
        "", "The quick brown fox.", model, tokenizer, config
    )

    assert keep_len > 0
    key = get_keys(cache, 0)
    assert not torch.isnan(key).any()


@slow
def test_keep_len_matches_cache_seq_dim(model_and_tokenizer, config):
    """keep_len = 1 + doc_len should match actual cache sequence dimension."""
    from lib.kv_cache import build_truncated_kv_cache_corrected

    model, tokenizer = model_and_tokenizer
    keep_len, cache = build_truncated_kv_cache_corrected(
        "test query", "A short document with several words in it.",
        model, tokenizer, config
    )

    key = get_keys(cache, 0)
    assert key.shape[2] == keep_len
