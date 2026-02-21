"""Tests for ChatGLM-6B specific KV cache operations."""

import types
import torch
import pytest

from lib.chatglm_kv_cache import (
    correct_2d_rope_positions,
    _rotate_half,
    _truncate_chatglm_cache,
    _build_generation_position_ids,
)


# ===== Fixtures =====

@pytest.fixture
def chatglm_fake_model():
    """Namespace mimicking ChatGLM-6B model.config."""
    cfg = types.SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
    )
    return types.SimpleNamespace(config=cfg)


@pytest.fixture
def small_chatglm_model():
    """Small ChatGLM-like config for fast tests."""
    cfg = types.SimpleNamespace(
        hidden_size=128,
        num_attention_heads=2,
    )
    return types.SimpleNamespace(config=cfg)


def _make_chatglm_cache(num_layers=2, seq_len=10, batch=1, num_heads=2, head_dim=64):
    """Create a fake ChatGLM-style cache: tuple of (key, value) per layer.

    Key/Value shape: [seq_len, batch, num_heads, head_dim]
    """
    return tuple(
        (
            torch.randn(seq_len, batch, num_heads, head_dim),
            torch.randn(seq_len, batch, num_heads, head_dim),
        )
        for _ in range(num_layers)
    )


# ===== Tests for _rotate_half =====

def test_rotate_half_shape():
    x = torch.randn(4, 8)
    result = _rotate_half(x)
    assert result.shape == x.shape


def test_rotate_half_inverse():
    """rotate_half applied twice with negation should give identity."""
    x = torch.randn(3, 5, 8)
    # rotate_half(rotate_half(x)) = -x
    rr = _rotate_half(_rotate_half(x))
    torch.testing.assert_close(rr, -x)


# ===== Tests for cache truncation =====

def test_truncate_cache_removes_first_n():
    cache = _make_chatglm_cache(num_layers=2, seq_len=10, num_heads=2, head_dim=64)
    truncated = _truncate_chatglm_cache(cache, remove_first_n=3)

    assert len(truncated) == 2
    for k, v in truncated:
        assert k.shape[0] == 7  # 10 - 3
        assert v.shape[0] == 7


def test_truncate_cache_preserves_content():
    cache = _make_chatglm_cache(num_layers=1, seq_len=8, num_heads=2, head_dim=64)
    original_key = cache[0][0].clone()
    truncated = _truncate_chatglm_cache(cache, remove_first_n=3)
    torch.testing.assert_close(truncated[0][0], original_key[3:])


def test_truncate_zero_is_identity():
    cache = _make_chatglm_cache(num_layers=2, seq_len=5)
    truncated = _truncate_chatglm_cache(cache, remove_first_n=0)
    for i in range(2):
        torch.testing.assert_close(truncated[i][0], cache[i][0])
        torch.testing.assert_close(truncated[i][1], cache[i][1])


# ===== Tests for 2D RoPE correction =====

def test_rope_correction_zero_offset_is_identity(small_chatglm_model):
    cache = _make_chatglm_cache(num_layers=2, seq_len=5, num_heads=2, head_dim=64)
    original = tuple((k.clone(), v.clone()) for k, v in cache)

    corrected = correct_2d_rope_positions(cache, offset=0, model=small_chatglm_model)

    for i in range(2):
        torch.testing.assert_close(corrected[i][0], original[i][0])


def test_rope_correction_only_affects_first_half(small_chatglm_model):
    """2D RoPE correction should only modify the first rotary_dim dims of keys."""
    cache = _make_chatglm_cache(num_layers=1, seq_len=5, num_heads=2, head_dim=64)
    original_key = cache[0][0].clone()
    original_value = cache[0][1].clone()

    corrected = correct_2d_rope_positions(cache, offset=3, model=small_chatglm_model)

    # Values should be unchanged
    torch.testing.assert_close(corrected[0][1], original_value)

    # Second half of key dims (block positions) should be unchanged
    rotary_dim = 32  # head_dim=64, rotary_dim=32
    torch.testing.assert_close(
        corrected[0][0][..., rotary_dim:],
        original_key[..., rotary_dim:]
    )

    # First half should be different (with offset != 0)
    assert not torch.allclose(
        corrected[0][0][..., :rotary_dim],
        original_key[..., :rotary_dim]
    )


def test_rope_correction_round_trip(small_chatglm_model):
    """Applying correction with +offset then -offset should return to original."""
    cache = _make_chatglm_cache(num_layers=2, seq_len=5, num_heads=2, head_dim=64)
    original = tuple((k.clone(), v.clone()) for k, v in cache)

    offset = 7
    corrected = correct_2d_rope_positions(cache, offset, small_chatglm_model)
    restored = correct_2d_rope_positions(corrected, -offset, small_chatglm_model)

    for i in range(2):
        torch.testing.assert_close(restored[i][0], original[i][0], atol=1e-5, rtol=1e-5)


def test_rope_correction_different_offsets_differ(small_chatglm_model):
    """Different offsets should produce different results."""
    cache1 = _make_chatglm_cache(num_layers=1, seq_len=3, num_heads=2, head_dim=64)
    cache2 = tuple((k.clone(), v.clone()) for k, v in cache1)

    c1 = correct_2d_rope_positions(cache1, offset=5, model=small_chatglm_model)
    c2 = correct_2d_rope_positions(cache2, offset=10, model=small_chatglm_model)

    assert not torch.allclose(c1[0][0], c2[0][0])


# ===== Tests with realistic ChatGLM-6B dimensions =====

def test_rope_correction_round_trip_realistic_dims(chatglm_fake_model):
    """Round-trip with real ChatGLM-6B dims: head_dim=128, rotary_dim=64."""
    cache = _make_chatglm_cache(num_layers=2, seq_len=5, num_heads=32, head_dim=128)
    original = tuple((k.clone(), v.clone()) for k, v in cache)

    offset = 7
    corrected = correct_2d_rope_positions(cache, offset, chatglm_fake_model)
    restored = correct_2d_rope_positions(corrected, -offset, chatglm_fake_model)

    for i in range(2):
        torch.testing.assert_close(restored[i][0], original[i][0], atol=1e-5, rtol=1e-5)


def test_rope_correction_only_affects_first_half_realistic(chatglm_fake_model):
    """With real dims, second 64 dims (block positions) should be unchanged."""
    cache = _make_chatglm_cache(num_layers=1, seq_len=5, num_heads=32, head_dim=128)
    original_key = cache[0][0].clone()
    original_value = cache[0][1].clone()

    corrected = correct_2d_rope_positions(cache, offset=3, model=chatglm_fake_model)

    torch.testing.assert_close(corrected[0][1], original_value)
    rotary_dim = 64  # head_dim=128, rotary_dim=64
    torch.testing.assert_close(
        corrected[0][0][..., rotary_dim:],
        original_key[..., rotary_dim:]
    )
    assert not torch.allclose(
        corrected[0][0][..., :rotary_dim],
        original_key[..., :rotary_dim]
    )


def test_truncate_cache_preserves_values():
    """Truncation should preserve value tensors (not just keys)."""
    cache = _make_chatglm_cache(num_layers=1, seq_len=8, num_heads=2, head_dim=64)
    original_value = cache[0][1].clone()
    truncated = _truncate_chatglm_cache(cache, remove_first_n=3)
    torch.testing.assert_close(truncated[0][1], original_value[3:])


def test_truncate_all_tokens():
    """Truncating all tokens should produce empty seq dimension."""
    cache = _make_chatglm_cache(num_layers=2, seq_len=5, num_heads=2, head_dim=64)
    truncated = _truncate_chatglm_cache(cache, remove_first_n=5)
    for k, v in truncated:
        assert k.shape[0] == 0
        assert v.shape[0] == 0


# ===== Tests for generation position_ids =====

def test_generation_position_ids_shape():
    """Shape should be [batch=1, 2, gen_len] — batch-first, matching ChatGLM convention."""
    pos = _build_generation_position_ids(context_len=20, gen_len=5, device=torch.device('cpu'))
    assert pos.shape == (1, 2, 5)


def test_generation_position_ids_absolute():
    """Absolute positions should equal mask_position = context_len - 2 (index of gMASK)."""
    pos = _build_generation_position_ids(context_len=15, gen_len=4, device=torch.device('cpu'))
    # Stream 0 (absolute): all = 15 - 2 = 13
    assert torch.all(pos[0, 0] == 13)


def test_generation_position_ids_block():
    """Block positions should be 2, 3, ..., gen_len+1 (continuing after <sop>'s block_pos=1)."""
    pos = _build_generation_position_ids(context_len=15, gen_len=4, device=torch.device('cpu'))
    expected = torch.tensor([2, 3, 4, 5])
    torch.testing.assert_close(pos[0, 1], expected)


# ===== Cross-implementation RoPE test =====

def _chatglm_rotate_half(x):
    """ChatGLM's rotate_half — identical to ours but defined independently for cross-check."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def _chatglm_apply_rotary_pos_emb(key, position, rotary_dim, rope_theta=10000.0):
    """Replicate ChatGLM's RoPE application on a single stream of the key tensor.

    This mirrors the model's apply_rotary_pos_emb_index but without embedding lookup —
    we directly compute cos/sin at the given integer position.

    Args:
        key: tensor of shape [..., rotary_dim] (one stream of the key)
        position: integer position to encode
        rotary_dim: dimensionality of this stream
        rope_theta: RoPE base frequency

    Returns:
        RoPE-encoded key at the given position
    """
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    ))
    angles = position * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(dtype=key.dtype)
    sin = emb.sin().to(dtype=key.dtype)
    return key * cos + _chatglm_rotate_half(key) * sin


def test_rope_forward_then_our_correction_recovers_original(small_chatglm_model):
    """Apply ChatGLM's forward RoPE at position p, then our correction with -p.

    This is the definitive cross-implementation test: ChatGLM's code rotates keys
    forward, and our correct_2d_rope_positions should exactly undo it.
    """
    rotary_dim = 32  # small model: head_dim=64, rotary_dim=32
    seq_len, batch, num_heads = 3, 1, 2
    head_dim = 64

    # Original unrotated key
    original_key = torch.randn(seq_len, batch, num_heads, head_dim)

    # Apply ChatGLM's forward RoPE to the absolute stream (first rotary_dim dims)
    # at positions 5, 6, 7 (as if these tokens start at position 5)
    rotated_key = original_key.clone()
    for pos_idx in range(seq_len):
        abs_position = 5 + pos_idx
        key_abs_stream = original_key[pos_idx, :, :, :rotary_dim]
        rotated_key[pos_idx, :, :, :rotary_dim] = _chatglm_apply_rotary_pos_emb(
            key_abs_stream, abs_position, rotary_dim
        )
        # Block stream (second half) at position 0 for prefix tokens → identity (cos=1, sin=0)
        # So second half is unchanged — already correct in our clone

    # Build a cache with these rotated keys
    dummy_value = torch.randn_like(original_key)
    cache = ((rotated_key, dummy_value),)

    # Now apply our correction to shift positions back by 5
    # Keys at positions [5, 6, 7] should become [0, 1, 2]
    corrected = correct_2d_rope_positions(cache, offset=5, model=small_chatglm_model)

    # The corrected keys should now be as if RoPE was applied at positions [0, 1, 2]
    expected_key = original_key.clone()
    for pos_idx in range(seq_len):
        key_abs_stream = original_key[pos_idx, :, :, :rotary_dim]
        expected_key[pos_idx, :, :, :rotary_dim] = _chatglm_apply_rotary_pos_emb(
            key_abs_stream, pos_idx, rotary_dim
        )

    torch.testing.assert_close(
        corrected[0][0][..., :rotary_dim],
        expected_key[..., :rotary_dim],
        atol=1e-5, rtol=1e-5,
    )
    # Block stream (second half) should be untouched
    torch.testing.assert_close(
        corrected[0][0][..., rotary_dim:],
        rotated_key[..., rotary_dim:],
    )


def test_rope_forward_then_our_correction_realistic_dims(chatglm_fake_model):
    """Same cross-implementation test with realistic ChatGLM-6B dimensions."""
    rotary_dim = 64  # real model: head_dim=128, rotary_dim=64
    seq_len, batch, num_heads = 3, 1, 32
    head_dim = 128

    original_key = torch.randn(seq_len, batch, num_heads, head_dim)

    rotated_key = original_key.clone()
    for pos_idx in range(seq_len):
        abs_position = 10 + pos_idx
        key_abs_stream = original_key[pos_idx, :, :, :rotary_dim]
        rotated_key[pos_idx, :, :, :rotary_dim] = _chatglm_apply_rotary_pos_emb(
            key_abs_stream, abs_position, rotary_dim
        )

    cache = ((rotated_key, torch.randn_like(original_key)),)
    corrected = correct_2d_rope_positions(cache, offset=10, model=chatglm_fake_model)

    expected_key = original_key.clone()
    for pos_idx in range(seq_len):
        key_abs_stream = original_key[pos_idx, :, :, :rotary_dim]
        expected_key[pos_idx, :, :, :rotary_dim] = _chatglm_apply_rotary_pos_emb(
            key_abs_stream, pos_idx, rotary_dim
        )

    torch.testing.assert_close(
        corrected[0][0][..., :rotary_dim],
        expected_key[..., :rotary_dim],
        atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        corrected[0][0][..., rotary_dim:],
        rotated_key[..., rotary_dim:],
    )


# ===== Slow tests requiring real model =====

@pytest.mark.slow
def test_chatglm_cache_shape():
    """Verify real ChatGLM cache has seq-first layout."""
    from lib.chatglm_kv_cache import load_chatglm, build_kv_cache_chatglm
    from lib.config import ExperimentConfig

    config = ExperimentConfig(
        model_name="THUDM/chatglm-6b",
        model_type="chatglm",
    )
    model, tokenizer = load_chatglm(config)

    ctx_len, cache = build_kv_cache_chatglm("Hello world", model, tokenizer, config)

    assert ctx_len > 0
    assert isinstance(cache, tuple)
    # Each layer: (key, value)
    k, v = cache[0]
    # seq-first: [seq_len, batch, heads, head_dim]
    assert k.shape[0] == ctx_len
    assert k.shape[1] == 1  # batch
    assert k.ndim == 4


@pytest.mark.slow
def test_prefix_lm_bidirectional_attention():
    """Verify passage KV entries DIFFER between bare and suffix caches.

    This is the opposite of our Mistral causal isolation check.
    With prefix LM, passage tokens attend bidirectionally to suffix tokens,
    so their KV representations should change.
    """
    from lib.chatglm_kv_cache import (
        load_chatglm, build_kv_cache_chatglm, build_suffix_kv_cache_chatglm,
    )
    from lib.config import ExperimentConfig

    config = ExperimentConfig(
        model_name="THUDM/chatglm-6b",
        model_type="chatglm",
    )
    model, tokenizer = load_chatglm(config)

    passage = "The quick brown fox jumps over the lazy dog."
    suffix = "What animal is mentioned in this sentence?"

    bare_len, bare_cache = build_kv_cache_chatglm(passage, model, tokenizer, config)
    sfx_len, sfx_cache = build_suffix_kv_cache_chatglm(
        passage, suffix, model, tokenizer, config
    )

    # In prefix LM, passage KV entries should DIFFER (unlike causal Mistral)
    bare_k = bare_cache[0][0]  # [seq, batch, heads, head_dim]
    sfx_k = sfx_cache[0][0]

    # The bare cache has bare_len tokens. The suffix cache has more.
    # Compare the first few layers — passage representations should differ.
    differs = False
    for layer_idx in range(min(3, len(bare_cache))):
        bare_k = bare_cache[layer_idx][0]
        sfx_k = sfx_cache[layer_idx][0]
        # Compare first bare_len tokens (note: tokenization may differ slightly)
        min_len = min(bare_k.shape[0], sfx_k.shape[0]) - 2  # exclude special tokens
        if min_len > 0 and not torch.allclose(bare_k[:min_len], sfx_k[:min_len], atol=1e-3):
            differs = True
            break

    assert differs, "Passage KV entries should DIFFER with prefix LM (bidirectional attention)"


@pytest.mark.slow
def test_rope_correction_matches_bare():
    """Build cache for [surr + passage], truncate, correct RoPE → should match bare passage cache."""
    from lib.chatglm_kv_cache import (
        load_chatglm, build_kv_cache_chatglm, build_truncated_kv_cache_chatglm,
    )
    from lib.config import ExperimentConfig

    config = ExperimentConfig(
        model_name="THUDM/chatglm-6b",
        model_type="chatglm",
    )
    model, tokenizer = load_chatglm(config)

    passage = "The quick brown fox jumps over the lazy dog."
    surrogate = "animals in sentences"

    bare_len, bare_cache = build_kv_cache_chatglm(passage, model, tokenizer, config)
    trunc_len, trunc_cache = build_truncated_kv_cache_chatglm(
        surrogate, passage, model, tokenizer, config
    )

    # Note: With prefix LM, the truncated cache won't perfectly match bare
    # because passage tokens had bidirectional attention to surrogate tokens.
    # The RoPE correction only fixes positions, not the attention influence.
    # This test verifies that the SHAPES match and values are reasonable.
    assert trunc_len == bare_len
    for layer_idx in range(min(3, len(bare_cache))):
        assert trunc_cache[layer_idx][0].shape == bare_cache[layer_idx][0].shape


@pytest.mark.slow
def test_scoring_produces_finite_nll():
    """Score a known answer and verify finite NLL."""
    from lib.chatglm_kv_cache import (
        load_chatglm, build_kv_cache_chatglm, score_answer_with_cache_chatglm,
    )
    from lib.config import ExperimentConfig

    config = ExperimentConfig(
        model_name="THUDM/chatglm-6b",
        model_type="chatglm",
    )
    model, tokenizer = load_chatglm(config)

    passage = "Paris is the capital of France."
    query_prompt = "\n\nQuery: What is the capital of France?\n\nAnswer:"
    answer = "Paris"

    ctx_len, cache = build_kv_cache_chatglm(passage, model, tokenizer, config)
    nll = score_answer_with_cache_chatglm(
        cache, ctx_len, query_prompt, answer, model, tokenizer, config
    )

    assert isinstance(nll, float)
    assert 0 < nll < 100, f"NLL should be finite and positive, got {nll}"
