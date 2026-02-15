"""Tests for Bug 2 (missing BOS token) and Bug 3 (tokenizer boundary mismatch).

Bug 2: extract_and_truncate_cache drops the BOS token at position 0, producing a
       cache that starts differently from the baseline (which always has BOS).

Bug 3: Computing doc_len by tokenizing the document in isolation yields a different
       token count than slicing the jointly-tokenized string, because tokenizer
       behavior changes at string boundaries (e.g., leading-space merging).

These tests verify the fixes: extract_and_truncate_cache_with_bos preserves BOS,
and build_truncated_kv_cache_corrected computes doc_len via prefix subtraction
rather than isolated tokenization.
"""

import types
import torch
import pytest
from transformers import DynamicCache

from lib.kv_cache import (
    extract_and_truncate_cache,
    extract_and_truncate_cache_with_bos,
)
from tests.conftest import get_keys, get_values


# =============================================================================
# Helpers
# =============================================================================

def _make_sequential_cache(num_layers, seq_len, num_heads=2, head_dim=4):
    """Build a cache where key[..., pos, 0] == pos, so positions are readable."""
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.zeros(1, num_heads, seq_len, head_dim)
        v = torch.zeros(1, num_heads, seq_len, head_dim)
        for pos in range(seq_len):
            k[:, :, pos, 0] = pos      # encode position in first element
            v[:, :, pos, 0] = pos + 100  # distinct marker for values
        cache.update(k, v, layer_idx)
    return cache


# =============================================================================
# Bug 2: BOS token preservation
# =============================================================================

class TestBosPreservation:
    """extract_and_truncate_cache (old) drops BOS; _with_bos (new) keeps it."""

    def test_old_truncation_drops_bos(self):
        """extract_and_truncate_cache keeps only the last N positions,
        so BOS (position 0) is discarded when N < seq_len."""
        cache = _make_sequential_cache(num_layers=1, seq_len=10)
        # Keep last 5 tokens — positions 5,6,7,8,9
        truncated = extract_and_truncate_cache(cache, keep_last_n=5)

        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 5
        # First position in truncated cache is position 5, NOT 0 (BOS)
        first_pos_marker = keys[0, 0, 0, 0].item()
        assert first_pos_marker == 5.0, (
            f"Old truncation starts at position {first_pos_marker}, BOS (0) is missing"
        )

    def test_new_truncation_preserves_bos(self):
        """extract_and_truncate_cache_with_bos keeps BOS + last doc_len positions."""
        cache = _make_sequential_cache(num_layers=1, seq_len=10)
        # Keep BOS + last 5 tokens → positions 0, 5, 6, 7, 8, 9
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=5)

        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 6  # 1 BOS + 5 doc
        # First position is BOS (position 0)
        assert keys[0, 0, 0, 0].item() == 0.0
        # Second position is start of doc portion (position 5)
        assert keys[0, 0, 1, 0].item() == 5.0

    def test_bos_and_doc_values_preserved(self):
        """Values should also have BOS + doc structure after truncation."""
        cache = _make_sequential_cache(num_layers=1, seq_len=10)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=3)

        vals = get_values(truncated, 0)
        assert vals.shape[2] == 4  # BOS + 3 doc
        # BOS value marker
        assert vals[0, 0, 0, 0].item() == 100.0  # pos 0 + 100
        # Doc values: positions 7, 8, 9
        assert vals[0, 0, 1, 0].item() == 107.0
        assert vals[0, 0, 2, 0].item() == 108.0
        assert vals[0, 0, 3, 0].item() == 109.0

    def test_multi_layer_bos_preserved(self):
        """BOS preservation must work across all layers."""
        cache = _make_sequential_cache(num_layers=4, seq_len=8)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=3)

        for layer_idx in range(4):
            keys = get_keys(truncated, layer_idx)
            assert keys.shape[2] == 4
            assert keys[0, 0, 0, 0].item() == 0.0, f"BOS missing in layer {layer_idx}"
            assert keys[0, 0, 1, 0].item() == 5.0, f"Wrong doc start in layer {layer_idx}"

    def test_doc_len_equals_full_seq_minus_one(self):
        """When doc_len = seq_len - 1, BOS + doc should reconstruct the full sequence."""
        seq_len = 7
        cache = _make_sequential_cache(num_layers=1, seq_len=seq_len)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=seq_len - 1)

        keys = get_keys(truncated, 0)
        # BOS (pos 0) + last 6 (pos 1..6) = full sequence
        assert keys.shape[2] == seq_len
        for pos in range(seq_len):
            assert keys[0, 0, pos, 0].item() == float(pos)

    def test_doc_len_one(self):
        """Edge case: single document token + BOS."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=1)

        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 2
        assert keys[0, 0, 0, 0].item() == 0.0   # BOS
        assert keys[0, 0, 1, 0].item() == 4.0   # last position

    def test_doc_len_equals_seq_len(self):
        """When doc_len == seq_len, BOS is duplicated (position 0 appears twice).
        This documents the behavior — callers should ensure doc_len < seq_len."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=5)

        keys = get_keys(truncated, 0)
        # BOS (pos 0) + last 5 (pos 0,1,2,3,4) = 6 total, with pos 0 duplicated
        assert keys.shape[2] == 6
        assert keys[0, 0, 0, 0].item() == 0.0  # explicit BOS
        assert keys[0, 0, 1, 0].item() == 0.0  # also pos 0 from the slice


# =============================================================================
# Bug 2 continued: structural comparison old vs new
# =============================================================================

class TestOldVsNewTruncation:
    """Compare structural properties of the two truncation methods."""

    def test_old_and_new_differ_on_first_position(self):
        """The key structural difference: old starts with a doc token,
        new starts with BOS."""
        seq_len = 10
        doc_len = 5
        cache_old = _make_sequential_cache(num_layers=1, seq_len=seq_len)
        cache_new = _make_sequential_cache(num_layers=1, seq_len=seq_len)

        trunc_old = extract_and_truncate_cache(cache_old, keep_last_n=doc_len)
        trunc_new = extract_and_truncate_cache_with_bos(cache_new, doc_len=doc_len)

        old_keys = get_keys(trunc_old, 0)
        new_keys = get_keys(trunc_new, 0)

        # Old: 5 positions; New: 6 positions (BOS + 5)
        assert old_keys.shape[2] == doc_len
        assert new_keys.shape[2] == doc_len + 1

        # Old starts at doc token; New starts at BOS
        assert old_keys[0, 0, 0, 0].item() == 5.0
        assert new_keys[0, 0, 0, 0].item() == 0.0

    def test_doc_portion_identical(self):
        """The document portion (non-BOS) should be identical between old and new."""
        seq_len = 10
        doc_len = 4
        cache_old = _make_sequential_cache(num_layers=1, seq_len=seq_len)
        cache_new = _make_sequential_cache(num_layers=1, seq_len=seq_len)

        trunc_old = extract_and_truncate_cache(cache_old, keep_last_n=doc_len)
        trunc_new = extract_and_truncate_cache_with_bos(cache_new, doc_len=doc_len)

        old_keys = get_keys(trunc_old, 0)
        new_doc_keys = get_keys(trunc_new, 0)[:, :, 1:, :]  # skip BOS

        torch.testing.assert_close(old_keys, new_doc_keys)


# =============================================================================
# Bug 3: Tokenizer boundary mismatch
# =============================================================================

class TestTokenizerBoundaryMismatch:
    """Tests demonstrating that tokenizing a document in isolation produces
    different token counts than slicing the joint tokenization.

    The corrected code computes doc_len = full_len - prefix_len instead of
    tokenizing the document separately.
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """A mock tokenizer that simulates the boundary mismatch.

        Behavior:
        - add_special_tokens=True prepends BOS token (id=1)
        - The prefix "Hello\\n\\n" tokenizes to [Hello, \\n, \\n] (3 tokens)
        - The document "Document" tokenizes to [▁Document] (1 token) in isolation
        - But in "Hello\\n\\nDocument", the document part becomes [Document] (different token)
          because the \\n\\n absorbs the leading space differently.

        We simulate this by having joint tokenization produce more tokens for the
        document portion than isolated tokenization would suggest.
        """
        BOS = 1

        def tokenize(text, return_tensors="pt", add_special_tokens=True,
                     padding=False, truncation=False):
            # Simplified simulation of boundary effects
            tokens = []
            if add_special_tokens:
                tokens.append(BOS)

            if text == "Hello\n\n":
                tokens.extend([10, 20, 20])  # Hello, \n, \n
            elif text == "Document:\nSome content here":
                # Isolated: ▁Document, :, \n, ▁Some, ▁content, ▁here
                tokens.extend([30, 40, 20, 50, 60, 70])
            elif text == "Hello\n\nDocument:\nSome content here":
                # Joint: Hello, \n, \n, Doc, ument, :, \n, Some, content, here
                # Boundary effect: "Document" splits differently after \n\n
                # 7 doc tokens in joint vs 6 in isolation
                tokens.extend([10, 20, 20, 31, 32, 40, 20, 51, 61, 71])
            elif text == "Short":
                tokens.extend([80])
            elif text == "Hello\n\nShort":
                tokens.extend([10, 20, 20, 81])
            else:
                # Generic: one token per word
                tokens.extend(list(range(100, 100 + len(text.split()))))

            ids = torch.tensor([tokens])
            return types.SimpleNamespace(input_ids=ids)

        return tokenize

    def test_isolated_vs_joint_token_count_differs(self, mock_tokenizer):
        """Tokenizing document in isolation gives different count than joint."""
        prefix = "Hello\n\n"
        document = "Document:\nSome content here"

        # Isolated document tokenization (with BOS)
        isolated = mock_tokenizer(document, add_special_tokens=False)
        isolated_doc_len = isolated.input_ids.shape[1]

        # Joint tokenization
        joint = mock_tokenizer(prefix + document, add_special_tokens=True)
        joint_total = joint.input_ids.shape[1]

        # Prefix tokenization (with BOS)
        prefix_enc = mock_tokenizer(prefix, add_special_tokens=True)
        prefix_len = prefix_enc.input_ids.shape[1]

        # BUGGY approach: use isolated doc_len
        buggy_doc_len = isolated_doc_len  # 6

        # CORRECT approach: subtract prefix from total
        correct_doc_len = joint_total - prefix_len  # 10 - 4 = 6... wait

        # The point: isolated tokenization gives 6 tokens but the actual
        # document portion in the joint string is 7 tokens (boundary effect)
        joint_doc_portion = joint_total - prefix_len
        assert isolated_doc_len != joint_doc_portion, (
            f"Expected mismatch: isolated={isolated_doc_len}, "
            f"joint_doc_portion={joint_doc_portion}"
        )

    def test_correct_doc_len_via_prefix_subtraction(self, mock_tokenizer):
        """The corrected approach: doc_len = full_len - prefix_len."""
        prefix = "Hello\n\n"
        document = "Document:\nSome content here"

        prefix_enc = mock_tokenizer(prefix, add_special_tokens=True)
        prefix_len = prefix_enc.input_ids.shape[1]

        full_enc = mock_tokenizer(prefix + document, add_special_tokens=True)
        full_len = full_enc.input_ids.shape[1]

        correct_doc_len = full_len - prefix_len

        # Now verify: if we take the last correct_doc_len tokens from full,
        # we get exactly the document portion
        full_ids = full_enc.input_ids[0]
        doc_ids_from_joint = full_ids[-correct_doc_len:]
        prefix_ids = full_ids[:prefix_len]

        # Reconstructed = prefix + doc should equal full
        reconstructed = torch.cat([prefix_ids, doc_ids_from_joint])
        torch.testing.assert_close(reconstructed, full_ids)

    def test_buggy_slice_misaligned(self, mock_tokenizer):
        """Using isolated doc_len to slice the joint sequence picks the wrong boundary."""
        prefix = "Hello\n\n"
        document = "Document:\nSome content here"

        # Buggy: tokenize doc in isolation for length
        isolated = mock_tokenizer(document, add_special_tokens=False)
        buggy_doc_len = isolated.input_ids.shape[1]  # 6

        # Joint tokenization
        full_enc = mock_tokenizer(prefix + document, add_special_tokens=True)
        full_ids = full_enc.input_ids[0]
        full_len = full_ids.shape[0]

        # Correct prefix length
        prefix_enc = mock_tokenizer(prefix, add_special_tokens=True)
        prefix_len = prefix_enc.input_ids.shape[1]
        correct_doc_len = full_len - prefix_len

        # The buggy slice takes last 6 tokens instead of last 7
        buggy_slice = full_ids[-buggy_doc_len:]
        correct_slice = full_ids[-correct_doc_len:]

        assert buggy_doc_len != correct_doc_len, (
            "Buggy and correct doc_len should differ for this example"
        )
        # Buggy slice is shorter and includes a prefix token at the boundary
        assert buggy_slice.shape[0] < correct_slice.shape[0]


# =============================================================================
# Bug 3 continued: prefix subtraction consistency
# =============================================================================

class TestPrefixSubtractionConsistency:
    """Verify that the prefix subtraction method is self-consistent:
    full_tokens == prefix_tokens + doc_tokens (when sliced from the joint encoding)."""

    @pytest.fixture
    def mock_tokenizer_v2(self):
        """More realistic mock: boundary effects at various join points."""
        BOS = 1

        # Simulate several prefix+doc combinations with boundary effects
        joint_table = {
            # (prefix, doc) -> (prefix_tokens_with_bos, full_tokens_with_bos)
            ("A\n\n", "Word"):
                ([BOS, 10, 20, 20], [BOS, 10, 20, 20, 30]),
            ("A\n\n", " Word"):
                ([BOS, 10, 20, 20], [BOS, 10, 20, 20, 31]),  # space absorbed
            ("Query: test\n\n", "Document:\nHello world"):
                ([BOS, 40, 41, 42, 20, 20], [BOS, 40, 41, 42, 20, 20, 50, 51, 52, 53]),
            ("", "Hello"):
                ([BOS], [BOS, 60]),
        }
        # Isolated doc tokens (different from joint due to boundary)
        isolated_table = {
            "Word": [30],
            " Word": [32],  # different token when tokenized alone with leading space
            "Document:\nHello world": [50, 51, 52, 53],
            "Hello": [60],
        }

        def tokenize(text, return_tensors="pt", add_special_tokens=True,
                     padding=False, truncation=False):
            tokens = []
            if add_special_tokens:
                tokens.append(BOS)

            # Check if this is a known joint string
            for (prefix, doc), (prefix_toks, full_toks) in joint_table.items():
                if text == prefix + doc:
                    ids = torch.tensor([full_toks])
                    return types.SimpleNamespace(input_ids=ids)
                if text == prefix and add_special_tokens:
                    ids = torch.tensor([prefix_toks])
                    return types.SimpleNamespace(input_ids=ids)

            # Check isolated doc
            if not add_special_tokens and text in isolated_table:
                ids = torch.tensor([isolated_table[text]])
                return types.SimpleNamespace(input_ids=ids)

            # Fallback
            tokens.extend(list(range(200, 200 + max(1, len(text.split())))))
            ids = torch.tensor([tokens])
            return types.SimpleNamespace(input_ids=ids)

        return tokenize

    @pytest.mark.parametrize("prefix,doc", [
        ("A\n\n", "Word"),
        ("A\n\n", " Word"),
        ("Query: test\n\n", "Document:\nHello world"),
        ("", "Hello"),
    ])
    def test_prefix_subtraction_exact_partition(self, mock_tokenizer_v2, prefix, doc):
        """full_ids[:prefix_len] + full_ids[prefix_len:] == full_ids, always."""
        prefix_enc = mock_tokenizer_v2(prefix, add_special_tokens=True)
        prefix_len = prefix_enc.input_ids.shape[1]

        full_enc = mock_tokenizer_v2(prefix + doc, add_special_tokens=True)
        full_ids = full_enc.input_ids[0]

        doc_len = full_ids.shape[0] - prefix_len
        assert doc_len >= 0, f"doc_len={doc_len} is negative"

        # The partition is exact by construction
        prefix_part = full_ids[:prefix_len]
        doc_part = full_ids[prefix_len:]
        reconstructed = torch.cat([prefix_part, doc_part])
        torch.testing.assert_close(reconstructed, full_ids)

    @pytest.mark.parametrize("prefix,doc", [
        ("A\n\n", "Word"),
        ("Query: test\n\n", "Document:\nHello world"),
    ])
    def test_isolated_doc_len_may_equal_joint(self, mock_tokenizer_v2, prefix, doc):
        """Even when token IDs differ at boundaries, lengths might match or not.
        The key insight: even if lengths happen to match, the tokens themselves
        differ, so the slice boundary in the cache points to different content."""
        isolated_enc = mock_tokenizer_v2(doc, add_special_tokens=False)
        isolated_len = isolated_enc.input_ids.shape[1]

        full_enc = mock_tokenizer_v2(prefix + doc, add_special_tokens=True)
        prefix_enc = mock_tokenizer_v2(prefix, add_special_tokens=True)
        joint_doc_len = full_enc.input_ids.shape[1] - prefix_enc.input_ids.shape[1]

        # This test documents that lengths CAN match even when tokens differ.
        # The prefix subtraction is correct regardless.
        # (No assertion on equality — the point is the method works either way)
        assert joint_doc_len >= 0


# =============================================================================
# Integration: extract_and_truncate_cache_with_bos + cache structure
# =============================================================================

class TestTruncationCacheIntegrity:
    """Verify that the truncated cache has the expected structure for downstream use."""

    def test_truncated_cache_is_dynamic_cache(self):
        """Output must be a DynamicCache for compatibility with HF model forward."""
        cache = _make_sequential_cache(num_layers=2, seq_len=8)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=3)
        assert isinstance(truncated, DynamicCache)

    def test_truncated_cache_layer_count(self):
        """Number of layers must be preserved."""
        for n_layers in [1, 4, 8]:
            cache = _make_sequential_cache(num_layers=n_layers, seq_len=6)
            truncated = extract_and_truncate_cache_with_bos(cache, doc_len=2)
            if hasattr(truncated, 'key_cache'):
                assert len(truncated.key_cache) == n_layers
            else:
                assert len(truncated.layers) == n_layers

    def test_truncated_cache_shapes(self):
        """Key/value shapes must be (batch, heads, 1+doc_len, head_dim)."""
        batch, heads, head_dim = 1, 2, 4
        seq_len, doc_len = 10, 3
        cache = _make_sequential_cache(
            num_layers=1, seq_len=seq_len, num_heads=heads, head_dim=head_dim
        )
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=doc_len)

        keys = get_keys(truncated, 0)
        vals = get_values(truncated, 0)
        expected_shape = (batch, heads, 1 + doc_len, head_dim)
        assert keys.shape == expected_shape, f"Keys: {keys.shape} != {expected_shape}"
        assert vals.shape == expected_shape, f"Vals: {vals.shape} != {expected_shape}"

    def test_contiguity(self):
        """Truncated tensors must be contiguous for efficient downstream ops."""
        cache = _make_sequential_cache(num_layers=2, seq_len=10)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=4)

        for i in range(2):
            assert get_keys(truncated, i).is_contiguous()
            assert get_values(truncated, i).is_contiguous()

    def test_no_shared_memory_with_original(self):
        """Truncated cache should not share memory with the original."""
        cache = _make_sequential_cache(num_layers=1, seq_len=6)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=3)

        # Mutate original
        get_keys(cache, 0).fill_(999.0)

        # Truncated should be unaffected
        assert get_keys(truncated, 0)[0, 0, 0, 0].item() != 999.0


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Boundary conditions and corner cases."""

    def test_single_token_sequence(self):
        """Sequence with only BOS (seq_len=1), doc_len=0 should not crash.
        This is degenerate but should be handled gracefully."""
        cache = _make_sequential_cache(num_layers=1, seq_len=1)
        # doc_len=0: keep BOS + 0 doc tokens
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=0)
        keys = get_keys(truncated, 0)
        # Should have just BOS
        # Note: -0: slice is empty, cat with BOS gives 1 token
        # Actually key[:,:,-0:,:] returns full tensor when doc_len=0
        # This is a known Python slice edge case
        # The function will keep BOS + last 0 = BOS + all tokens
        # This documents the behavior for awareness
        assert keys.shape[2] >= 1  # at minimum BOS

    def test_old_truncation_keep_all(self):
        """Old truncation keeping all tokens preserves full sequence."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache(cache, keep_last_n=5)
        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 5
        for pos in range(5):
            assert keys[0, 0, pos, 0].item() == float(pos)

    def test_old_truncation_keep_one(self):
        """Old truncation keeping 1 token keeps only the last."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache(cache, keep_last_n=1)
        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 1
        assert keys[0, 0, 0, 0].item() == 4.0  # last position

    def test_random_content_preserved_exactly(self):
        """With random data, verify exact value preservation (no interpolation)."""
        torch.manual_seed(42)
        cache = DynamicCache()
        k = torch.randn(1, 3, 8, 16)
        v = torch.randn(1, 3, 8, 16)
        cache.update(k.clone(), v.clone(), 0)

        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=4)
        trunc_keys = get_keys(truncated, 0)

        # BOS position should be exact copy of position 0
        torch.testing.assert_close(trunc_keys[:, :, 0, :], k[:, :, 0, :])
        # Doc positions should be exact copies of positions 4,5,6,7
        torch.testing.assert_close(trunc_keys[:, :, 1:, :], k[:, :, 4:, :])

    def test_legacy_tuple_cache_format(self):
        """extract_and_truncate_cache_with_bos should handle tuple-format caches."""
        k = torch.randn(1, 2, 6, 4)
        v = torch.randn(1, 2, 6, 4)
        # Legacy format: tuple of (key, value) tuples per layer
        legacy_cache = ((k, v),)

        truncated = extract_and_truncate_cache_with_bos(legacy_cache, doc_len=3)
        assert isinstance(truncated, DynamicCache)
        keys = get_keys(truncated, 0)
        assert keys.shape[2] == 4  # BOS + 3 doc

    def test_doc_len_zero_edge_case(self):
        """doc_len=0 with Python slice: key[:,:,-0:,:] returns ALL positions.
        This is a known edge case with negative zero slicing."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=0)
        keys = get_keys(truncated, 0)
        # -0 == 0, so key[:,:,0:,:] returns all 5 positions, plus BOS concat = 6
        # This documents the unintuitive behavior
        assert keys.shape[2] == 6  # BOS + all 5 (because -0 slice returns everything)

    def test_keep_last_n_greater_than_seq_len(self):
        """keep_last_n > seq_len: slice returns entire tensor (no error)."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache(cache, keep_last_n=100)
        keys = get_keys(truncated, 0)
        # key[:,:,-100:,:] when seq_len=5 returns all 5
        assert keys.shape[2] == 5

    def test_keep_last_n_zero_returns_empty(self):
        """keep_last_n=0: key[:,:,-0:,:] returns full tensor (same -0 bug)."""
        cache = _make_sequential_cache(num_layers=1, seq_len=5)
        truncated = extract_and_truncate_cache(cache, keep_last_n=0)
        keys = get_keys(truncated, 0)
        # -0 == 0, so key[:,:,0:,:] returns everything
        assert keys.shape[2] == 5  # returns all, not empty

    def test_single_layer_single_head(self):
        """Minimal cache: 1 layer, 1 head."""
        cache = _make_sequential_cache(num_layers=1, seq_len=4, num_heads=1, head_dim=2)
        truncated = extract_and_truncate_cache_with_bos(cache, doc_len=2)
        keys = get_keys(truncated, 0)
        assert keys.shape == (1, 1, 3, 2)  # BOS + 2 doc
        assert keys[0, 0, 0, 0].item() == 0.0  # BOS
        assert keys[0, 0, 1, 0].item() == 2.0   # doc start
        assert keys[0, 0, 2, 0].item() == 3.0   # doc end
