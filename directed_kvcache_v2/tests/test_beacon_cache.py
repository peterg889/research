"""Tests for periodic beacon cache functions (Experiment 18).

Unit tests use mocked model/tokenizer (no GPU required).
Integration tests require a real model and are skipped if unavailable.
"""

import types
import torch
import pytest
from unittest.mock import MagicMock
from transformers import DynamicCache

from lib.kv_cache import (
    build_beacon_cache_sequential,
    build_beacon_cache_batch,
    extract_cache_at_indices,
    correct_rope_positions_chunked,
    deepcopy_cache,
    _get_cache_keys,
    _get_cache_values,
    _ensure_dynamic_cache,
)
from lib.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 4
HEAD_DIM = 16


class _DictNamespace(dict):
    """Dict that also supports attribute access, mimicking HF tokenizer output."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


def _make_fake_tokenizer(bos_token_id=1, vocab_size=1000):
    """Build a fake tokenizer that maps each character to a unique token ID."""
    def encode(text, return_tensors=None, add_special_tokens=True, **kwargs):
        BOS = bos_token_id
        tokens = []
        if add_special_tokens:
            tokens.append(BOS)
        # 1 token per character (IDs 10, 11, 12, ...)
        tokens.extend(list(range(10, 10 + len(text))))
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens

    tok = MagicMock()
    tok.encode = encode
    tok.bos_token_id = bos_token_id
    tok.vocab_size = vocab_size
    return tok


def _make_fake_model(num_layers=2):
    """Build a fake model that returns deterministic caches based on input_ids."""
    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(
                hidden_size=NUM_HEADS * HEAD_DIM,
                num_attention_heads=NUM_HEADS,
                rope_theta=10000.0,
            )
            self.device = torch.device('cpu')
            self._num_layers = num_layers

        def __call__(self, input_ids, attention_mask=None, use_cache=True,
                     return_dict=True, position_ids=None, **kwargs):
            seq_len = input_ids.shape[1]
            cache = DynamicCache()
            for layer_idx in range(self._num_layers):
                # Use input_ids to seed keys deterministically
                torch.manual_seed(layer_idx * 1000 + seq_len)
                k = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
                v = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
                cache.update(k, v, layer_idx)
            return types.SimpleNamespace(past_key_values=cache)

    return FakeModel()


def _make_env(num_layers=2):
    """Set up fake model, tokenizer, config for beacon cache tests."""
    model = _make_fake_model(num_layers)
    tokenizer = _make_fake_tokenizer()
    config = ExperimentConfig(device='cpu')
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Unit Tests — Sequential Cache
# ---------------------------------------------------------------------------

class TestSequentialInterleaveShape:
    """Verify output sequence length and structure."""

    def test_basic_shape(self):
        model, tokenizer, config = _make_env()
        doc = "A" * 20  # 20 tokens
        beacon_ids = [100, 101, 102]  # 3-token beacon
        chunk_size = 8

        cache, seq_len, doc_ids, beacon_pos, doc_pos = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        n_chunks = -(-len(doc_ids) // chunk_size)  # ceil division
        expected_len = 1 + sum(len(beacon_ids) + min(chunk_size, len(doc_ids) - i * chunk_size)
                               for i in range(n_chunks))
        assert seq_len == expected_len
        assert _get_cache_keys(cache, 0).shape[2] == seq_len

    def test_single_chunk(self):
        """Doc shorter than chunk_size → single chunk with one beacon."""
        model, tokenizer, config = _make_env()
        doc = "Hi"  # 2 tokens
        beacon_ids = [100, 101]
        chunk_size = 256

        cache, seq_len, doc_ids, beacon_pos, doc_pos = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        # [BOS][beacon(2)][doc(2)] = 5
        assert seq_len == 1 + len(beacon_ids) + len(doc_ids)
        assert len(beacon_pos) == len(beacon_ids)
        assert len(doc_pos) == len(doc_ids)


class TestSequentialPositionTracking:
    """Verify beacon_positions and doc_positions are correct."""

    def test_disjoint_and_complete(self):
        model, tokenizer, config = _make_env()
        doc = "A" * 30
        beacon_ids = [100, 101]
        chunk_size = 10

        _, seq_len, _, beacon_pos, doc_pos = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        # Disjoint
        assert set(beacon_pos).isdisjoint(set(doc_pos))
        # Cover all non-BOS positions
        assert set(beacon_pos) | set(doc_pos) == set(range(1, seq_len))

    def test_positions_ordered(self):
        model, tokenizer, config = _make_env()
        doc = "ABCDEFGHIJ" * 5  # 50 chars = 50 tokens
        beacon_ids = [100, 101, 102]
        chunk_size = 15

        _, _, _, beacon_pos, doc_pos = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        assert beacon_pos == sorted(beacon_pos)
        assert doc_pos == sorted(doc_pos)


class TestSequentialShortDoc:
    """Doc shorter than chunk_size produces one chunk."""

    def test_short_doc_single_chunk(self):
        model, tokenizer, config = _make_env()
        doc = "X" * 5
        beacon_ids = [200, 201, 202, 203]
        chunk_size = 512

        _, seq_len, doc_ids, beacon_pos, doc_pos = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        assert len(doc_ids) == 5
        # Only one beacon group
        assert len(beacon_pos) == 4
        assert len(doc_pos) == 5


class TestMatchedDocTokens:
    """doc_token_ids from sequential match bare tokenization."""

    def test_tokens_match(self):
        model, tokenizer, config = _make_env()
        doc = "Hello world test"
        beacon_ids = [100]
        chunk_size = 256

        _, _, doc_ids, _, _ = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)

        bare_ids = tokenizer.encode(doc, add_special_tokens=False)
        assert doc_ids == bare_ids


# ---------------------------------------------------------------------------
# Unit Tests — Extract Cache at Indices
# ---------------------------------------------------------------------------

class TestExtractCacheAtIndices:
    """Test extract_cache_at_indices utility."""

    def test_shape_and_values(self, make_cache):
        cache = make_cache(num_layers=2, seq_len=10, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        indices = [0, 3, 7]

        extracted = extract_cache_at_indices(cache, indices)

        assert _get_cache_keys(extracted, 0).shape[2] == 3
        for li in range(2):
            orig_k = _get_cache_keys(cache, li)
            ext_k = _get_cache_keys(extracted, li)
            for i, idx in enumerate(indices):
                assert torch.allclose(orig_k[:, :, idx, :], ext_k[:, :, i, :])

    def test_preserves_order(self, make_cache):
        cache = make_cache(num_layers=1, seq_len=8, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        indices = [1, 4, 6]

        extracted = extract_cache_at_indices(cache, indices)

        k_orig = _get_cache_keys(cache, 0)
        k_ext = _get_cache_keys(extracted, 0)
        # Position 1 in extracted should match position 1 in original
        assert torch.allclose(k_ext[:, :, 0, :], k_orig[:, :, 1, :])
        # Position 4 in original → position 1 in extracted
        assert torch.allclose(k_ext[:, :, 1, :], k_orig[:, :, 4, :])

    def test_single_index(self, make_cache):
        cache = make_cache(num_layers=2, seq_len=5, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        extracted = extract_cache_at_indices(cache, [2])
        assert _get_cache_keys(extracted, 0).shape[2] == 1

    def test_all_indices(self, make_cache):
        cache = make_cache(num_layers=1, seq_len=6, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        indices = list(range(6))
        extracted = extract_cache_at_indices(cache, indices)
        k_orig = _get_cache_keys(cache, 0)
        k_ext = _get_cache_keys(extracted, 0)
        assert torch.allclose(k_orig, k_ext)


# ---------------------------------------------------------------------------
# Unit Tests — Chunked RoPE Correction
# ---------------------------------------------------------------------------

class TestChunkedRopeCorrection:
    """Test correct_rope_positions_chunked."""

    def test_offsets_modify_keys(self, make_cache, small_fake_model):
        """Verify that non-zero offsets actually change keys."""
        cache = make_cache(num_layers=1, seq_len=11, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        original_keys = _get_cache_keys(cache, 0).clone()

        # 2 chunks of 5 after BOS
        chunk_boundaries = [1, 6]
        offsets = [3, 6]  # beacon_len * (chunk_idx + 1)

        correct_rope_positions_chunked(cache, chunk_boundaries, offsets, small_fake_model)

        corrected_keys = _get_cache_keys(cache, 0)
        # BOS should be unchanged
        assert torch.allclose(original_keys[:, :, :1, :], corrected_keys[:, :, :1, :])
        # Doc keys should be different
        assert not torch.allclose(original_keys[:, :, 1:, :], corrected_keys[:, :, 1:, :])

    def test_zero_offset_noop(self, make_cache, small_fake_model):
        """Zero offset for a chunk should leave it unchanged."""
        cache = make_cache(num_layers=1, seq_len=6, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        original_keys = _get_cache_keys(cache, 0).clone()

        # Single chunk with offset 0
        correct_rope_positions_chunked(cache, [1], [0], small_fake_model)

        assert torch.allclose(original_keys, _get_cache_keys(cache, 0))

    def test_increasing_offsets(self, make_cache, small_fake_model):
        """Each successive chunk gets a larger offset correction."""
        beacon_len = 3
        chunk_size = 4
        n_chunks = 3
        total = 1 + n_chunks * chunk_size  # BOS + 3*4 = 13

        cache = make_cache(num_layers=1, seq_len=total, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        boundaries = [1 + i * chunk_size for i in range(n_chunks)]
        offsets = [(i + 1) * beacon_len for i in range(n_chunks)]

        assert offsets == [3, 6, 9]
        assert boundaries == [1, 5, 9]

        original = _get_cache_keys(cache, 0).clone()
        correct_rope_positions_chunked(cache, boundaries, offsets, small_fake_model)

        # Each chunk should have different corrections applied
        for i in range(n_chunks):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < n_chunks else total
            chunk_orig = original[:, :, start:end, :]
            chunk_corr = _get_cache_keys(cache, 0)[:, :, start:end, :]
            if offsets[i] != 0:
                assert not torch.allclose(chunk_orig, chunk_corr)


# ---------------------------------------------------------------------------
# Unit Tests — Batch Cache
# ---------------------------------------------------------------------------

class TestBatchCache:
    """Test build_beacon_cache_batch."""

    def test_strips_bos_from_non_first(self):
        """Flattened cache length should be first_full + sum(non_first - 1)."""
        model, tokenizer, config = _make_env()
        doc = "A" * 20
        beacon_ids = [100, 101]
        chunk_size = 8

        cache, total_len, doc_ids = \
            build_beacon_cache_batch(doc, beacon_ids, chunk_size, model, tokenizer, config)

        n_chunks = -(-len(doc_ids) // chunk_size)
        beacon_len = len(beacon_ids)
        # First chunk: [BOS][beacon][chunk] = 1 + beacon_len + min(chunk_size, len)
        first_len = 1 + beacon_len + min(chunk_size, len(doc_ids))
        # Subsequent chunks: [beacon][chunk] (BOS stripped)
        remaining_len = sum(
            beacon_len + min(chunk_size, len(doc_ids) - i * chunk_size)
            for i in range(1, n_chunks)
        )
        expected = first_len + remaining_len
        assert total_len == expected

    def test_matches_sequential_structure(self):
        """Batch and sequential produce same total length for same input."""
        model, tokenizer, config = _make_env()
        doc = "A" * 30
        beacon_ids = [100, 101, 102]
        chunk_size = 10

        _, seq_len, _, _, _ = \
            build_beacon_cache_sequential(doc, beacon_ids, chunk_size, model, tokenizer, config)
        _, batch_len, _ = \
            build_beacon_cache_batch(doc, beacon_ids, chunk_size, model, tokenizer, config)

        assert seq_len == batch_len


class TestRandomBeaconLength:
    """Random beacon has same token count as static beacon."""

    def test_length_matches(self):
        tokenizer = _make_fake_tokenizer()
        beacon_text = "What are the key facts I need to know?"
        beacon_ids = tokenizer.encode(beacon_text, add_special_tokens=False)
        beacon_len = len(beacon_ids)

        # Random beacon of same length
        random_ids = torch.randint(100, 900, (beacon_len,)).tolist()
        assert len(random_ids) == beacon_len


# ---------------------------------------------------------------------------
# Integration Tests (require real model)
# ---------------------------------------------------------------------------

def _load_real_model():
    """Try to load the Mistral model. Returns (model, tokenizer) or raises."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map='auto'
    )
    return model, tokenizer


@pytest.fixture(scope='module')
def real_model():
    """Load real model once per module, skip if unavailable."""
    try:
        model, tokenizer = _load_real_model()
        config = ExperimentConfig(device='cuda')
        return model, tokenizer, config
    except Exception as e:
        pytest.skip(f'Real model unavailable: {e}')


SAMPLE_DOC = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
    "Locally nicknamed La dame de fer, it was constructed from 1887 to 1889 as the centerpiece "
    "of the 1889 World's Fair. Although initially criticised by some of France's leading artists "
    "and intellectuals for its design, it has since become a global cultural icon of France."
)
SAMPLE_QUERY = "\nQuery: Where is the Eiffel Tower located?\nAnswer:"
SAMPLE_ANSWER = " Paris, France"
BEACON_TEXT = "What are the key facts I need to know?"


class TestIntegrationSequentialFiniteNLL:
    """Build beacon cache, score answer, NLL should be finite."""

    def test_finite_nll(self, real_model):
        model, tokenizer, config = real_model
        beacon_ids = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)

        cache, seq_len, _, _, _ = build_beacon_cache_sequential(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )
        from lib.kv_cache import score_answer_with_cache
        nll = score_answer_with_cache(
            deepcopy_cache(cache), seq_len, SAMPLE_QUERY, SAMPLE_ANSWER,
            model, tokenizer, config
        )
        assert torch.isfinite(torch.tensor(nll))
        assert nll > 0


class TestIntegrationBatchFiniteNLL:
    """Build batch beacon cache, score answer, NLL should be finite."""

    def test_finite_nll(self, real_model):
        model, tokenizer, config = real_model
        beacon_ids = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)

        cache, seq_len, _ = build_beacon_cache_batch(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )
        from lib.kv_cache import score_answer_with_cache
        nll = score_answer_with_cache(
            deepcopy_cache(cache), seq_len, SAMPLE_QUERY, SAMPLE_ANSWER,
            model, tokenizer, config
        )
        assert torch.isfinite(torch.tensor(nll))
        assert nll > 0


class TestIntegrationTruncatedBeaconFiniteNLL:
    """Extract doc positions + RoPE correct + score → finite NLL."""

    def test_finite_nll(self, real_model):
        model, tokenizer, config = real_model
        beacon_ids = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)
        beacon_len = len(beacon_ids)

        cache, seq_len, doc_ids, beacon_pos, doc_pos = build_beacon_cache_sequential(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )

        # Extract [BOS] + doc positions
        extract_indices = [0] + doc_pos
        extracted = extract_cache_at_indices(cache, extract_indices)

        # Compute chunk boundaries and offsets for RoPE correction
        # After extraction: [BOS][all doc tokens], but doc tokens came from
        # different chunks each preceded by beacon_len beacon tokens.
        n_doc_ids = len(doc_ids)
        chunk_size = 256
        n_chunks = -(-n_doc_ids // chunk_size)
        boundaries = []
        offsets = []
        pos = 1  # after BOS
        for k in range(n_chunks):
            boundaries.append(pos)
            offsets.append((k + 1) * beacon_len)
            actual_chunk = min(chunk_size, n_doc_ids - k * chunk_size)
            pos += actual_chunk

        correct_rope_positions_chunked(extracted, boundaries, offsets, model)

        keep_len = 1 + n_doc_ids
        from lib.kv_cache import score_answer_with_cache
        nll = score_answer_with_cache(
            deepcopy_cache(extracted), keep_len, SAMPLE_QUERY, SAMPLE_ANSWER,
            model, tokenizer, config
        )
        assert torch.isfinite(torch.tensor(nll))
        assert nll > 0


class TestIntegrationSequentialVsBareDifferentKeys:
    """Beacon cache keys should differ from bare cache keys (contamination happened)."""

    def test_keys_differ(self, real_model):
        model, tokenizer, config = real_model
        beacon_ids = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)

        # Beacon cache
        beacon_cache, _, doc_ids, _, doc_pos = build_beacon_cache_sequential(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )

        # Bare cache: [BOS][doc]
        bos_id = tokenizer.bos_token_id
        bare_ids = torch.tensor([[bos_id] + doc_ids], device=config.device)
        with torch.no_grad():
            bare_out = model(
                input_ids=bare_ids,
                attention_mask=torch.ones_like(bare_ids),
                use_cache=True, return_dict=True,
            )
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)

        # Compare doc token keys: they should differ because beacon context
        # was present during the forward pass
        beacon_doc_keys = _get_cache_keys(beacon_cache, 0)[:, :, doc_pos, :]
        bare_doc_keys = _get_cache_keys(bare_cache, 0)[:, :, 1:, :]  # skip BOS

        # Shapes might differ if doc_ids length != len(doc_pos), but they should match
        assert beacon_doc_keys.shape == bare_doc_keys.shape
        assert not torch.allclose(beacon_doc_keys, bare_doc_keys, atol=1e-3)


class TestIntegrationBatchVsSequentialSameShape:
    """Both batch and sequential produce same cache shape for same input."""

    def test_same_shape(self, real_model):
        model, tokenizer, config = real_model
        beacon_ids = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)

        _, seq_len, _ = build_beacon_cache_batch(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )
        _, seq_len_s, _, _, _ = build_beacon_cache_sequential(
            SAMPLE_DOC, beacon_ids, 256, model, tokenizer, config
        )

        assert seq_len == seq_len_s
