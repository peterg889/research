"""Pipeline correctness tests for multi-model two-phase KV cache scoring.

Tests the fundamental invariants that must hold for the two-phase pipeline
to produce valid results:

1. RoPE round-trip: rotate then unrotate should recover original keys
2. Bare two-phase ≈ single-pass: without prefix/reposition, cache split
   should reproduce single-pass NLL (within bf16 tolerance)
3. Cache shape: select_kv_cache produces correct shapes
4. Position alignment: Phase B position_ids match expected values

These tests require GPU and model access. Run with:
    cd /home/jupyter/research/directed_kvcache_publication
    pytest tests/test_pipeline_correctness.py -v -s

Mark slow tests with @pytest.mark.slow. Use --runslow to include them.
"""

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../../directed_kvcache_v4")

import pytest
import torch
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv

from lib.rope import rotate_half, select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

# Skip all tests if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for model tests"
)


# =====================================================================
# Fixtures
# =====================================================================

def _load_model(model_name, loader_name):
    """Load a model and tokenizer, return (model, tokenizer, device)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    if loader_name == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0"
        ).eval()
    elif loader_name == "Gemma3nForConditionalGeneration":
        from transformers import Gemma3nForConditionalGeneration
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0"
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0"
        ).eval()

    device = next(model.parameters()).device
    return model, tokenizer, device


# Model configurations for parametrized tests
MODEL_CONFIGS = [
    pytest.param(
        "google/gemma-3-12b-it", "Gemma3ForConditionalGeneration",
        id="gemma3_12b"
    ),
    pytest.param(
        "google/gemma-3n-e4b-it", "Gemma3nForConditionalGeneration",
        id="gemma3n_e4b"
    ),
    pytest.param(
        "mistralai/Mistral-7B-Instruct-v0.3", "AutoModelForCausalLM",
        id="mistral_7b"
    ),
    pytest.param(
        "Qwen/Qwen2.5-7B-Instruct", "AutoModelForCausalLM",
        id="qwen25_7b"
    ),
]

# Test data
TEST_DOC = "The capital of France is Paris. It is located along the Seine river."
TEST_QUERY = "What is the capital of France?"
TEST_ANSWER = "Paris"
PREFIX_L = 64
COMPREHEND_TEXT = "Read and comprehend this text carefully."


# =====================================================================
# Pure math tests (no model required)
# =====================================================================

class TestRoPERoundTripMath:
    """Test that RoPE rotation is exactly reversible in fp32."""

    def test_roundtrip_fp32_identity(self):
        """Forward + reverse rotation in fp32 should be near-identity."""
        torch.manual_seed(42)
        key = torch.randn(1, 8, 10, 128)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))

        delta = torch.arange(1, 11, dtype=torch.float32)  # different delta per position
        freqs = torch.einsum("i,j->ij", delta, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_d = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_d = emb.sin().unsqueeze(0).unsqueeze(0)

        # Forward
        rotated = key * cos_d + rotate_half(key) * sin_d
        # Reverse (cos is even, sin is odd)
        unrotated = rotated * cos_d + rotate_half(rotated) * (-sin_d)

        err = (key - unrotated).abs().max().item()
        assert err < 1e-5, f"fp32 round-trip error {err:.2e} exceeds 1e-5"

    def test_roundtrip_bf16_bounded(self):
        """bf16 round-trip error should be bounded (not zero, but small)."""
        torch.manual_seed(42)
        key = torch.randn(1, 8, 10, 128, dtype=torch.bfloat16)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))

        delta = torch.arange(1, 11, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", delta, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_d = emb.cos().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        sin_d = emb.sin().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

        rotated = key * cos_d + rotate_half(key) * sin_d
        unrotated = rotated * cos_d + rotate_half(rotated) * (-sin_d)

        err = (key - unrotated).abs().max().item()
        # bf16 has ~0.8% relative precision, so round-trip error up to ~5% is expected
        assert err < 0.5, f"bf16 round-trip error {err:.2e} is unreasonably large"
        print(f"  bf16 round-trip error: {err:.4f} (expected < 0.5)")

    def test_zero_delta_is_identity(self):
        """Rotation with delta=0 should be exact identity in any precision."""
        torch.manual_seed(42)
        for dtype in [torch.float32, torch.bfloat16]:
            key = torch.randn(1, 8, 5, 128, dtype=dtype)
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))

            delta = torch.zeros(5, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", delta, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_d = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
            sin_d = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)

            rotated = key * cos_d + rotate_half(key) * sin_d
            err = (key - rotated).abs().max().item()
            assert err == 0.0, f"Zero-delta rotation not identity in {dtype}: err={err}"

    def test_different_theta_values(self):
        """Round-trip should work for different rope_theta values."""
        torch.manual_seed(42)
        key = torch.randn(1, 4, 5, 128)

        for theta in [10000.0, 500000.0, 1000000.0]:
            inv_freq = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
            delta = torch.tensor([50.0] * 5)
            freqs = torch.einsum("i,j->ij", delta, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_d = emb.cos().unsqueeze(0).unsqueeze(0)
            sin_d = emb.sin().unsqueeze(0).unsqueeze(0)

            rotated = key * cos_d + rotate_half(key) * sin_d
            unrotated = rotated * cos_d + rotate_half(rotated) * (-sin_d)

            err = (key - unrotated).abs().max().item()
            assert err < 1e-5, f"fp32 round-trip with theta={theta}: err={err:.2e}"

    def test_head_dim_256(self):
        """Round-trip should work for Gemma's head_dim=256."""
        torch.manual_seed(42)
        key = torch.randn(1, 8, 5, 256)
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, 256, 2, dtype=torch.float32) / 256))

        delta = torch.tensor([100.0] * 5)
        freqs = torch.einsum("i,j->ij", delta, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_d = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_d = emb.sin().unsqueeze(0).unsqueeze(0)

        rotated = key * cos_d + rotate_half(key) * sin_d
        unrotated = rotated * cos_d + rotate_half(rotated) * (-sin_d)

        err = (key - unrotated).abs().max().item()
        assert err < 1e-5, f"fp32 round-trip with head_dim=256: err={err:.2e}"


# =====================================================================
# Model adapter tests (no GPU, use mock configs)
# =====================================================================

class TestModelAdapters:
    """Test that model_adapters.py correctly extracts RoPE params from configs."""

    def test_gemma3_style_config(self):
        """Gemma 3 has per-layer-type rope_parameters (dict of dicts)."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types

        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="gemma3",
                head_dim=256,
                num_hidden_layers=4,
                layer_types=["sliding_attention", "sliding_attention",
                             "full_attention", "sliding_attention"],
                rope_parameters={
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                },
            )
        )
        model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1)]))

        inv_freqs = build_layer_inv_freqs(model, device="cpu")
        layer_types = get_layer_types(model)

        assert "sliding_attention" in inv_freqs
        assert "full_attention" in inv_freqs
        assert inv_freqs["sliding_attention"].shape == (128,)
        assert layer_types == ["sliding_attention", "sliding_attention",
                               "full_attention", "sliding_attention"]
        # Different theta → different inv_freq
        assert not torch.allclose(inv_freqs["sliding_attention"],
                                  inv_freqs["full_attention"])

    def test_flat_config_with_layer_types(self):
        """Qwen has flat rope_parameters BUT config.layer_types exists."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types

        cfg = SimpleNamespace(
            model_type="qwen2",
            head_dim=128,
            hidden_size=3584,
            num_attention_heads=28,
            num_hidden_layers=4,
            layer_types=["full_attention"] * 4,
            rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        )
        model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1)]))

        inv_freqs = build_layer_inv_freqs(model, device="cpu")
        layer_types = get_layer_types(model)

        # Should map the flat theta to the layer_type name
        assert "full_attention" in inv_freqs, f"Keys: {list(inv_freqs.keys())}"
        assert inv_freqs["full_attention"].shape == (64,)
        assert layer_types == ["full_attention"] * 4

        # Every layer type should have a matching inv_freq
        for lt in layer_types:
            assert lt in inv_freqs, f"Layer type '{lt}' not in inv_freqs"

    def test_flat_config_no_layer_types(self):
        """Llama/Mistral have flat rope_parameters and no layer_types."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types

        cfg = SimpleNamespace(
            model_type="llama",
            head_dim=128,
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=4,
            rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        )
        model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1)]))

        inv_freqs = build_layer_inv_freqs(model, device="cpu")
        layer_types = get_layer_types(model)

        assert "all" in inv_freqs
        assert inv_freqs["all"].shape == (64,)
        assert layer_types == ["all"] * 4

    def test_inv_freq_layer_type_consistency(self):
        """Every entry in layer_types must have a matching inv_freq key."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types

        # Test all config styles
        configs = [
            # Gemma style
            SimpleNamespace(
                text_config=SimpleNamespace(
                    model_type="gemma3", head_dim=256, num_hidden_layers=6,
                    layer_types=["sliding_attention"] * 5 + ["full_attention"],
                    rope_parameters={
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                        "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                    },
                )
            ),
            # Qwen style (flat rope but has layer_types)
            SimpleNamespace(
                model_type="qwen2", head_dim=128, hidden_size=3584,
                num_attention_heads=28, num_hidden_layers=4,
                layer_types=["full_attention"] * 4,
                rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
            ),
            # Mistral style (flat, no layer_types)
            SimpleNamespace(
                model_type="mistral", head_dim=128, hidden_size=4096,
                num_attention_heads=32, num_hidden_layers=4,
                rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
            ),
        ]

        for cfg in configs:
            model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1)]))
            inv_freqs = build_layer_inv_freqs(model, device="cpu")
            layer_types = get_layer_types(model)

            for L, lt in enumerate(layer_types):
                assert lt in inv_freqs, (
                    f"Layer {L} type '{lt}' not in inv_freqs {list(inv_freqs.keys())} "
                    f"for config type {getattr(getattr(cfg, 'text_config', cfg), 'model_type', '?')}"
                )


# =====================================================================
# Integration tests (require GPU + model)
# =====================================================================

@pytest.mark.slow
class TestUseCacheFalseConsistency:
    """Verify use_cache=False and use_cache=True produce identical logits.

    Gemma 3N has a known bug where use_cache=False produces garbage output.
    We must use use_cache=True for single-pass scoring on that model.
    """

    @pytest.mark.parametrize("model_name,loader", MODEL_CONFIGS)
    def test_use_cache_flag_consistency(self, model_name, loader):
        """use_cache=False and use_cache=True should produce same logits."""
        model, tokenizer, device = _load_model(model_name, loader)
        try:
            bos_id = tokenizer.bos_token_id
            has_bos = bos_id is not None
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
            query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
            answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
            bos_offset = 1 if has_bos else 0
            D = len(doc_ids)

            sp_ids = (([bos_id] if has_bos else []) + doc_ids +
                      nl_ids + query_ids + nl_ids + answer_ids)
            sp_t = torch.tensor([sp_ids], device=device)
            targets = torch.tensor(answer_ids, device=device)
            ans_start = bos_offset + D + len(nl_ids) + len(query_ids) + len(nl_ids)

            with torch.no_grad():
                out_f = model(input_ids=sp_t, use_cache=False)
                nll_f = torch.nn.functional.cross_entropy(
                    out_f.logits[0, ans_start-1:ans_start-1+len(answer_ids)],
                    targets).item()

                out_t = model(input_ids=sp_t, use_cache=True)
                nll_t = torch.nn.functional.cross_entropy(
                    out_t.logits[0, ans_start-1:ans_start-1+len(answer_ids)],
                    targets).item()

            diff = abs(nll_f - nll_t)
            is_broken = diff > 1.0
            print(f"\n  {model_name}: cache=F NLL={nll_f:.4f}, cache=T NLL={nll_t:.4f}, "
                  f"diff={diff:.4f}, broken={is_broken}")
            if is_broken:
                # Verify the workaround: use_cache=True should match two-phase
                with torch.no_grad():
                    pa_ids = ([bos_id] if has_bos else []) + doc_ids
                    pa_out = model(input_ids=torch.tensor([pa_ids], device=device),
                                   use_cache=True)
                    from lib.cache import deep_copy_cache as _dcc
                    pb_ids = nl_ids + query_ids + nl_ids + answer_ids
                    n_b = len(pb_ids)
                    pos_b = torch.arange(bos_offset + D, bos_offset + D + n_b,
                                         device=device).unsqueeze(0)
                    cc = _dcc(pa_out.past_key_values)
                    pb_out = model(input_ids=torch.tensor([pb_ids], device=device),
                                   position_ids=pos_b, past_key_values=cc,
                                   use_cache=False)
                    b_start = len(nl_ids) + len(query_ids) + len(nl_ids)
                    nll_2p = torch.nn.functional.cross_entropy(
                        pb_out.logits[0, b_start-1:b_start-1+len(answer_ids)],
                        targets).item()
                workaround_diff = abs(nll_t - nll_2p)
                print(f"  Workaround: use_cache=True SP={nll_t:.4f} vs 2P={nll_2p:.4f}, "
                      f"diff={workaround_diff:.4f}")
                assert workaround_diff < 0.15, (
                    f"use_cache=True workaround also broken: diff={workaround_diff:.4f}"
                )
                pytest.skip(
                    f"{model_name}: use_cache=False is broken (diff={diff:.1f}). "
                    f"Workaround (use_cache=True) verified OK (diff={workaround_diff:.4f})."
                )
            assert diff < 0.01, f"use_cache flag produces different NLL: {diff:.4f}"
        finally:
            del model
            torch.cuda.empty_cache()


@pytest.mark.slow
class TestBareTwoPhaseMatchesSinglePass:
    """Bare two-phase (no prefix, no reposition) should match single-pass NLL.

    This is the most fundamental correctness test. Uses use_cache=True for
    single-pass to avoid the Gemma 3N use_cache=False bug.
    """

    @pytest.mark.parametrize("model_name,loader", MODEL_CONFIGS)
    def test_bare_matches_single_pass(self, model_name, loader):
        """NLL from bare two-phase should match single-pass within bf16 tolerance."""
        model, tokenizer, device = _load_model(model_name, loader)
        try:
            bos_id = tokenizer.bos_token_id
            has_bos = bos_id is not None
            bos_offset = 1 if has_bos else 0
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)

            doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
            query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
            answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
            D = len(doc_ids)

            with torch.no_grad():
                # Single-pass (use_cache=True to avoid Gemma 3N bug)
                sp_ids = (([bos_id] if has_bos else []) + doc_ids +
                          nl_ids + query_ids + nl_ids + answer_ids)
                sp_out = model(
                    input_ids=torch.tensor([sp_ids], device=device),
                    use_cache=True
                )
                sp_logits = sp_out.logits[0]
                ans_start = bos_offset + D + len(nl_ids) + len(query_ids) + len(nl_ids)
                sp_ans = sp_logits[ans_start - 1 : ans_start - 1 + len(answer_ids)]
                targets = torch.tensor(answer_ids, device=device)
                nll_sp = torch.nn.functional.cross_entropy(sp_ans, targets).item()

                # Two-phase bare
                pa_ids = ([bos_id] if has_bos else []) + doc_ids
                pa_out = model(
                    input_ids=torch.tensor([pa_ids], device=device),
                    use_cache=True
                )
                cache = pa_out.past_key_values

                pb_ids = nl_ids + query_ids + nl_ids + answer_ids
                n_b = len(pb_ids)
                pos_b = torch.arange(
                    bos_offset + D, bos_offset + D + n_b, device=device
                ).unsqueeze(0)

                cache_copy = deep_copy_cache(cache)
                pb_out = model(
                    input_ids=torch.tensor([pb_ids], device=device),
                    position_ids=pos_b,
                    past_key_values=cache_copy,
                    use_cache=False
                )
                pb_logits = pb_out.logits[0]
                b_ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
                pb_ans = pb_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
                nll_2p = torch.nn.functional.cross_entropy(pb_ans, targets).item()

            diff = abs(nll_sp - nll_2p)
            # bf16 tolerance: allow up to 5% relative difference or 0.15 absolute
            tol = max(0.05 * abs(nll_sp), 0.15)
            print(f"\n  {model_name}: SP={nll_sp:.4f}, 2P={nll_2p:.4f}, "
                  f"diff={diff:.4f}, tol={tol:.4f}")
            assert diff < tol, (
                f"Bare two-phase ({nll_2p:.4f}) != single-pass ({nll_sp:.4f}), "
                f"diff={diff:.4f} > tol={tol:.4f}. "
                f"Pipeline is broken for {model_name}!"
            )
        finally:
            del model
            torch.cuda.empty_cache()


@pytest.mark.slow
class TestRoPERoundTripOnModel:
    """Test RoPE round-trip using actual model inv_freqs."""

    @pytest.mark.parametrize("model_name,loader", MODEL_CONFIGS)
    def test_roundtrip_fp32(self, model_name, loader):
        """Round-trip in fp32 should be near-exact with real model inv_freqs."""
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types

        model, tokenizer, device = _load_model(model_name, loader)
        try:
            inv_freqs = build_layer_inv_freqs(model, device=device)
            layer_types = get_layer_types(model)
            info_cfg = getattr(model.config, "text_config", model.config)
            head_dim = getattr(info_cfg, "head_dim",
                               info_cfg.hidden_size // info_cfg.num_attention_heads)
            n_kv = getattr(info_cfg, "num_key_value_heads",
                           info_cfg.num_attention_heads)
            n_layers = len(layer_types)

            # Create fp32 dummy cache
            torch.manual_seed(42)
            cache = DynamicCache()
            D = 10
            for L in range(n_layers):
                k = torch.randn(1, n_kv, D + 1, head_dim, dtype=torch.float32,
                                device=device)
                v = torch.randn(1, n_kv, D + 1, head_dim, dtype=torch.float32,
                                device=device)
                cache.update(k, v, L)

            orig_keys = [cache.layers[L].keys.clone() for L in range(n_layers)]

            # Forward reposition
            old_pos = torch.arange(100, 100 + D, device=device)
            new_pos = torch.arange(1, 1 + D, device=device)
            cache = reposition_kv_cache(cache, old_pos, new_pos,
                                        inv_freqs, layer_types, bos_start=0)

            # Reverse reposition
            cache = reposition_kv_cache(cache, new_pos, old_pos,
                                        inv_freqs, layer_types, bos_start=0)

            max_err = 0.0
            for L in range(n_layers):
                doc_orig = orig_keys[L][:, :, 1:, :]
                doc_rt = cache.layers[L].keys[:, :, 1:, :]
                err = (doc_orig - doc_rt).abs().max().item()
                max_err = max(max_err, err)

            print(f"\n  {model_name}: fp32 round-trip max error = {max_err:.2e}")
            assert max_err < 1e-4, (
                f"fp32 round-trip error {max_err:.2e} > 1e-4 for {model_name}. "
                f"inv_freqs or layer_types may be incorrect."
            )
        finally:
            del model
            torch.cuda.empty_cache()


@pytest.mark.slow
class TestFullPrefixPipeline:
    """Test the complete prefix conditioning pipeline end-to-end.

    Verifies that prefix + select + reposition produces a valid cache
    that can score answers. The NLL should be finite and in a reasonable
    range (not garbage from misaligned positions or wrong RoPE).
    """

    @pytest.mark.parametrize("model_name,loader", MODEL_CONFIGS)
    def test_prefix_produces_finite_nll(self, model_name, loader):
        """Full pipeline: prefix → select → reposition → normalize → score.
        NLL should be finite and < 20 (not garbage)."""
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types, get_sliding_cache_limit

        model, tokenizer, device = _load_model(model_name, loader)
        try:
            bos_id = tokenizer.bos_token_id
            has_bos = bos_id is not None
            bos_offset = 1 if has_bos else 0
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            inv_freqs = build_layer_inv_freqs(model, device=device)
            layer_types = get_layer_types(model)
            sliding_limit = get_sliding_cache_limit(model)

            doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
            query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
            answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
            comprehend_ids = tokenizer.encode(COMPREHEND_TEXT, add_special_tokens=False)
            prefix = make_prefix(comprehend_ids, PREFIX_L)
            D = len(doc_ids)

            # Check we don't exceed sliding window
            total_cache = bos_offset + D
            if sliding_limit is not None:
                assert total_cache <= sliding_limit, (
                    f"Cache {total_cache} > sliding limit {sliding_limit}")

            with torch.no_grad():
                # Phase A with prefix
                pa_ids = (([bos_id] if has_bos else []) +
                          prefix + nl_ids + doc_ids)
                pa_out = model(
                    input_ids=torch.tensor([pa_ids], device=device),
                    use_cache=True
                )
                cache = pa_out.past_key_values

                # Verify cache length
                expected_len = bos_offset + len(prefix) + len(nl_ids) + D
                actual_len = cache.get_seq_length()
                assert actual_len == expected_len, (
                    f"Phase A cache: {actual_len} != expected {expected_len}")

                # Select BOS + doc
                doc_start = bos_offset + len(prefix) + len(nl_ids)
                if has_bos:
                    keep = [0] + list(range(doc_start, doc_start + D))
                else:
                    keep = list(range(doc_start, doc_start + D))
                cache = select_kv_cache(cache, keep, device=device)

                selected_len = cache.get_seq_length()
                assert selected_len == bos_offset + D, (
                    f"After select: {selected_len} != {bos_offset + D}")

                # Reposition
                old_pos = torch.arange(doc_start, doc_start + D, device=device)
                new_pos = torch.arange(bos_offset, bos_offset + D, device=device)
                cache = reposition_kv_cache(
                    cache, old_pos, new_pos, inv_freqs, layer_types,
                    bos_start=0 if has_bos else -1)

                # Normalize
                cache = norm_roundtrip_kv_cache(cache)

                # Score Phase B
                pb_ids = nl_ids + query_ids + nl_ids + answer_ids
                n_b = len(pb_ids)
                pos_b = torch.arange(
                    bos_offset + D, bos_offset + D + n_b, device=device
                ).unsqueeze(0)
                cache_copy = deep_copy_cache(cache)
                pb_out = model(
                    input_ids=torch.tensor([pb_ids], device=device),
                    position_ids=pos_b,
                    past_key_values=cache_copy,
                    use_cache=False
                )
                pb_logits = pb_out.logits[0]
                b_ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
                pb_ans = pb_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
                targets = torch.tensor(answer_ids, device=device)
                nll = torch.nn.functional.cross_entropy(pb_ans, targets).item()

            print(f"\n  {model_name}: prefix pipeline NLL = {nll:.4f}")
            assert np.isfinite(nll), f"NLL is not finite: {nll}"
            assert nll < 20.0, f"NLL={nll:.1f} is unreasonably high (garbage output)"
            assert nll > 0.0, f"NLL={nll:.4f} is non-positive"
        finally:
            del model
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("model_name,loader", MODEL_CONFIGS)
    def test_normalization_does_not_corrupt(self, model_name, loader):
        """Normalization should change NLL only slightly, not catastrophically."""
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs, get_layer_types, get_sliding_cache_limit

        model, tokenizer, device = _load_model(model_name, loader)
        try:
            bos_id = tokenizer.bos_token_id
            has_bos = bos_id is not None
            bos_offset = 1 if has_bos else 0
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)

            doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
            query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
            answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
            D = len(doc_ids)

            with torch.no_grad():
                # Build bare cache
                pa_ids = ([bos_id] if has_bos else []) + doc_ids
                pa_out = model(
                    input_ids=torch.tensor([pa_ids], device=device),
                    use_cache=True
                )

                # Score without normalization
                pb_ids = nl_ids + query_ids + nl_ids + answer_ids
                n_b = len(pb_ids)
                pos_b = torch.arange(
                    bos_offset + D, bos_offset + D + n_b, device=device
                ).unsqueeze(0)
                targets = torch.tensor(answer_ids, device=device)

                cache_no_norm = deep_copy_cache(pa_out.past_key_values)
                out_nn = model(input_ids=torch.tensor([pb_ids], device=device),
                               position_ids=pos_b, past_key_values=cache_no_norm,
                               use_cache=False)
                b_start = len(nl_ids) + len(query_ids) + len(nl_ids)
                nll_nn = torch.nn.functional.cross_entropy(
                    out_nn.logits[0, b_start-1:b_start-1+len(answer_ids)],
                    targets).item()

                # Score with normalization
                cache_normed = norm_roundtrip_kv_cache(
                    deep_copy_cache(pa_out.past_key_values))
                out_n = model(input_ids=torch.tensor([pb_ids], device=device),
                              position_ids=pos_b,
                              past_key_values=deep_copy_cache(cache_normed),
                              use_cache=False)
                nll_n = torch.nn.functional.cross_entropy(
                    out_n.logits[0, b_start-1:b_start-1+len(answer_ids)],
                    targets).item()

            diff = abs(nll_nn - nll_n)
            print(f"\n  {model_name}: no_norm={nll_nn:.4f}, normed={nll_n:.4f}, "
                  f"diff={diff:.4f}")
            # Normalization should not change NLL by more than 50% or 5.0 absolute
            assert diff < max(0.5 * abs(nll_nn), 5.0), (
                f"Normalization changed NLL by {diff:.2f} on {model_name}. "
                f"This suggests it corrupts the cache for this model."
            )
        finally:
            del model
            torch.cuda.empty_cache()


from transformers import DynamicCache


# =====================================================================
# Unit tests for core pipeline operations (no GPU needed)
# =====================================================================

def _make_dummy_cache(n_layers=4, n_heads=8, seq_len=20, head_dim=128,
                      dtype=torch.float32, device='cpu'):
    """Create a deterministic dummy KV cache for testing."""
    torch.manual_seed(42)
    cache = DynamicCache()
    for L in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        v = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        cache.update(k, v, L)
    return cache


class TestSelectKvCacheCorrectness:
    """Verify select_kv_cache produces correct outputs."""

    def test_output_count_matches_indices(self):
        """Selected cache should have exactly len(indices) entries."""
        cache = _make_dummy_cache(n_layers=4, seq_len=100)
        for indices in [[0], [0, 50, 99], list(range(100)), [0, 1, 2, 3, 4]]:
            selected = select_kv_cache(cache, indices, device='cpu')
            assert selected.get_seq_length() == len(indices), (
                f"Expected {len(indices)} entries, got {selected.get_seq_length()}")

    def test_all_layers_same_length(self):
        """All layers should have the same sequence length after selection."""
        cache = _make_dummy_cache(n_layers=8, seq_len=50)
        selected = select_kv_cache(cache, [0, 10, 20, 30, 40], device='cpu')
        lengths = [selected.layers[L].keys.shape[2] for L in range(len(selected.layers))]
        assert len(set(lengths)) == 1, f"Inconsistent layer lengths: {lengths}"

    def test_selected_values_match_originals(self):
        """Selected entries should have the exact same values as originals."""
        cache = _make_dummy_cache(n_layers=2, seq_len=20)
        indices = [0, 5, 10, 15, 19]
        selected = select_kv_cache(cache, indices, device='cpu')
        for L in range(2):
            for i, idx in enumerate(indices):
                assert torch.equal(selected.layers[L].keys[0, :, i, :],
                                   cache.layers[L].keys[0, :, idx, :]), (
                    f"Layer {L}, position {i} (orig idx {idx}): keys don't match")
                assert torch.equal(selected.layers[L].values[0, :, i, :],
                                   cache.layers[L].values[0, :, idx, :]), (
                    f"Layer {L}, position {i} (orig idx {idx}): values don't match")

    def test_bos_preservation(self):
        """Index 0 (BOS) should be preserved exactly in selection."""
        cache = _make_dummy_cache(n_layers=4, seq_len=50)
        bos_keys_orig = [cache.layers[L].keys[:, :, 0:1, :].clone()
                         for L in range(4)]
        selected = select_kv_cache(cache, [0, 10, 20, 30], device='cpu')
        for L in range(4):
            assert torch.equal(selected.layers[L].keys[:, :, 0:1, :],
                               bos_keys_orig[L])


class TestRepositionCorrectness:
    """Verify reposition_kv_cache modifies keys correctly."""

    def _make_test_setup(self, head_dim=128):
        inv_freqs = {"all": 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))}
        layer_types = ["all"] * 4
        return inv_freqs, layer_types

    def test_reposition_actually_changes_keys(self):
        """Repositioning should measurably change key values."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        keys_before = cache.layers[0].keys[:, :, 1:, :].clone()

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        keys_after = cache.layers[0].keys[:, :, 1:, :]
        diff = (keys_before - keys_after).abs().max().item()
        assert diff > 0.01, f"Keys were barely changed (diff={diff:.4f}) — reposition may be a no-op"

    def test_reposition_does_not_modify_values(self):
        """Values should be completely untouched by repositioning."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        values_before = [cache.layers[L].values.clone() for L in range(4)]

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(4):
            assert torch.equal(cache.layers[L].values, values_before[L]), (
                f"Layer {L} values were modified by reposition!")

    def test_bos_not_rotated_with_bos_start_0(self):
        """BOS entry at index 0 should not be modified when bos_start=0."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        bos_before = [cache.layers[L].keys[:, :, 0:1, :].clone() for L in range(4)]

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(4):
            assert torch.equal(cache.layers[L].keys[:, :, 0:1, :], bos_before[L]), (
                f"Layer {L} BOS was modified by reposition!")

    def test_all_entries_rotated_with_bos_start_neg1(self):
        """With bos_start=-1, ALL entries including first should be rotated."""
        cache = _make_dummy_cache(n_layers=4, seq_len=10, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        first_key_before = cache.layers[0].keys[:, :, 0:1, :].clone()

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(0, 10)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=-1)

        first_key_after = cache.layers[0].keys[:, :, 0:1, :]
        assert not torch.equal(first_key_after, first_key_before), (
            "First entry should be rotated with bos_start=-1")

    def test_zero_delta_is_identity(self):
        """Repositioning with old_pos == new_pos should not change keys."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        keys_before = [cache.layers[L].keys.clone() for L in range(4)]

        same_pos = torch.arange(1, 11)
        reposition_kv_cache(cache, same_pos, same_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(4):
            diff = (cache.layers[L].keys - keys_before[L]).abs().max().item()
            assert diff == 0.0, f"Layer {L} keys changed with zero delta: diff={diff}"

    def test_roundtrip_fp32_exact(self):
        """Forward then reverse reposition should recover original keys in fp32."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        keys_orig = [cache.layers[L].keys[:, :, 1:, :].clone() for L in range(4)]

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(4):
            diff = (cache.layers[L].keys[:, :, 1:, :] - keys_orig[L]).abs().max().item()
            assert diff < 1e-5, f"Layer {L} fp32 round-trip error: {diff:.2e}"


class TestNormalizationCorrectness:
    """Verify normalization behaves correctly."""

    def test_idempotent_fp32(self):
        """Applying normalization twice should give same result as once in fp32."""
        cache = _make_dummy_cache(n_layers=4, seq_len=20, dtype=torch.float32)
        norm_roundtrip_kv_cache(cache)
        keys_after_first = [cache.layers[L].keys.clone() for L in range(4)]
        norm_roundtrip_kv_cache(cache)
        for L in range(4):
            diff = (cache.layers[L].keys - keys_after_first[L]).abs().max().item()
            assert diff < 1e-5, f"Layer {L} not idempotent: diff={diff:.2e}"

    def test_preserves_shape(self):
        """Normalization should not change cache shapes."""
        cache = _make_dummy_cache(n_layers=4, seq_len=20)
        shapes_before = [(cache.layers[L].keys.shape, cache.layers[L].values.shape)
                         for L in range(4)]
        norm_roundtrip_kv_cache(cache)
        for L in range(4):
            assert cache.layers[L].keys.shape == shapes_before[L][0]
            assert cache.layers[L].values.shape == shapes_before[L][1]

    def test_preserves_bos(self):
        """BOS entry (index 0) should be preserved by normalization."""
        cache = _make_dummy_cache(n_layers=4, seq_len=20)
        bos_before = [cache.layers[L].keys[:, :, 0:1, :].clone() for L in range(4)]
        norm_roundtrip_kv_cache(cache)
        for L in range(4):
            assert torch.equal(cache.layers[L].keys[:, :, 0:1, :], bos_before[L])


class TestRotateHalfIdentities:
    """Mathematical identities of rotate_half that RoPE depends on."""

    def test_double_rotate_negates(self):
        """rotate_half(rotate_half(x)) == -x. This is the i*i = -1 identity."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        result = rotate_half(rotate_half(x))
        assert torch.allclose(result, -x, atol=1e-7), (
            f"rotate_half applied twice should negate, max diff: "
            f"{(result + x).abs().max().item():.2e}")

    def test_four_rotations_identity(self):
        """rotate_half applied 4 times should return original (i^4 = 1)."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        result = rotate_half(rotate_half(rotate_half(rotate_half(x))))
        assert torch.allclose(result, x, atol=1e-7)

    def test_preserves_l2_norm(self):
        """rotate_half should preserve L2 norm (orthogonal transformation)."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        norm_before = x.norm(dim=-1)
        norm_after = rotate_half(x).norm(dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-6), (
            f"Norm changed: max diff {(norm_before - norm_after).abs().max().item():.2e}")

    def test_orthogonal_to_input(self):
        """rotate_half(x) should be orthogonal to x (dot product = 0)."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        dot = (x * rotate_half(x)).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5), (
            f"Not orthogonal: max dot product {dot.abs().max().item():.2e}")


class TestRoPERotationIdentities:
    """Mathematical identities of the full RoPE rotation formula."""

    def _make_rotation(self, head_dim=128, theta=10000.0, positions=None):
        """Create cos/sin for given positions."""
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if positions is None:
            positions = torch.arange(5, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _apply_rope(self, x, cos, sin):
        return x * cos + rotate_half(x) * sin

    def test_rotation_preserves_norm(self):
        """RoPE rotation is norm-preserving (it's a rotation matrix)."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        cos, sin = self._make_rotation()
        x_rot = self._apply_rope(x, cos, sin)
        norm_before = x.norm(dim=-1)
        norm_after = x_rot.norm(dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5), (
            f"Rotation changed norm: max diff {(norm_before - norm_after).abs().max().item():.2e}")

    def test_rotation_at_pos_0_is_identity(self):
        """At position 0, cos=1 and sin=0, so rotation is identity."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 1, 128)
        cos, sin = self._make_rotation(positions=torch.tensor([0.0]))
        x_rot = self._apply_rope(x, cos, sin)
        assert torch.allclose(x_rot, x, atol=1e-7)

    def test_rotation_then_inverse_is_identity_fp32(self):
        """x * cos(θ) + rot(x) * sin(θ) then * cos(θ) + rot(*) * (-sin(θ)) = x."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        cos, sin = self._make_rotation()
        rotated = self._apply_rope(x, cos, sin)
        unrotated = self._apply_rope(rotated, cos, -sin)
        assert torch.allclose(x, unrotated, atol=1e-5), (
            f"Round-trip error: {(x - unrotated).abs().max().item():.2e}")

    def test_different_theta_gives_different_rotation(self):
        """Gemma uses theta=10K for sliding and theta=1M for full layers.
        These must produce DIFFERENT rotations for the same position."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        cos_10k, sin_10k = self._make_rotation(theta=10000.0)
        cos_1m, sin_1m = self._make_rotation(theta=1000000.0)
        rot_10k = self._apply_rope(x, cos_10k, sin_10k)
        rot_1m = self._apply_rope(x, cos_1m, sin_1m)
        diff = (rot_10k - rot_1m).abs().max().item()
        assert diff > 0.01, f"Different thetas should give different rotations, diff={diff:.4f}"

    def test_composition_of_rotations(self):
        """Rotating by pos A then by delta B should equal rotating by pos A+B.
        This is the key property that makes RoPE repositioning work."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128)
        # Rotate by position 10
        cos_10, sin_10 = self._make_rotation(positions=torch.arange(10, 15, dtype=torch.float32))
        # Rotate by position 3 (delta from 10 to 13)
        cos_3, sin_3 = self._make_rotation(positions=torch.tensor([3.0] * 5))
        # Rotate by position 13 directly
        cos_13, sin_13 = self._make_rotation(positions=torch.arange(13, 18, dtype=torch.float32))

        # Method 1: rotate to pos 10, then apply delta +3
        x_at_10 = self._apply_rope(x, cos_10, sin_10)
        x_at_13_via_delta = self._apply_rope(x_at_10, cos_3, sin_3)

        # Method 2: rotate to pos 13 directly
        x_at_13_direct = self._apply_rope(x, cos_13, sin_13)

        assert torch.allclose(x_at_13_via_delta, x_at_13_direct, atol=1e-4), (
            f"Rotation composition failed: max diff "
            f"{(x_at_13_via_delta - x_at_13_direct).abs().max().item():.2e}")

    def test_bf16_roundtrip_bounded_error(self):
        """bf16 round-trip should have bounded error (not zero, but small and consistent)."""
        torch.manual_seed(42)
        x = torch.randn(1, 8, 5, 128, dtype=torch.bfloat16)
        cos, sin = self._make_rotation()
        cos_bf16 = cos.to(torch.bfloat16)
        sin_bf16 = sin.to(torch.bfloat16)
        rotated = self._apply_rope(x, cos_bf16, sin_bf16)
        unrotated = self._apply_rope(rotated, cos_bf16, -sin_bf16)
        err = (x - unrotated).abs().max().item()
        # bf16 round-trip error should be bounded — not zero but not catastrophic
        assert err < 0.1, f"bf16 round-trip error too large: {err:.4f}"
        assert err > 0.0, "bf16 should have SOME round-trip error (not exact)"


class TestDeepCopyIndependence:
    """Verify deep_copy_cache produces truly independent copies."""

    def test_modify_copy_does_not_affect_original(self):
        """Mutating the copy should leave the original unchanged."""
        cache = _make_dummy_cache(n_layers=4, seq_len=10)
        original_keys = [cache.layers[L].keys.clone() for L in range(4)]
        copy = deep_copy_cache(cache)
        # Mutate the copy
        for L in range(4):
            copy.layers[L].keys[:] = 0.0
        # Original should be unchanged
        for L in range(4):
            assert torch.equal(cache.layers[L].keys, original_keys[L])

    def test_modify_original_does_not_affect_copy(self):
        """Mutating the original should leave the copy unchanged."""
        cache = _make_dummy_cache(n_layers=4, seq_len=10)
        copy = deep_copy_cache(cache)
        copy_keys = [copy.layers[L].keys.clone() for L in range(4)]
        # Mutate the original
        for L in range(4):
            cache.layers[L].keys[:] = 0.0
        # Copy should be unchanged
        for L in range(4):
            assert torch.equal(copy.layers[L].keys, copy_keys[L])


class TestEndToEndPipelineIdentities:
    """End-to-end identity tests on the full pipeline composition."""

    def _make_test_setup(self, head_dim=128):
        inv_freqs = {"all": 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))}
        layer_types = ["all"] * 4
        return inv_freqs, layer_types

    def test_select_all_then_identity_reposition_is_noop(self):
        """Selecting all entries + repositioning with zero delta = original cache."""
        cache = _make_dummy_cache(n_layers=4, seq_len=11, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()
        keys_orig = [cache.layers[L].keys.clone() for L in range(4)]

        # Select all entries
        all_indices = list(range(11))
        selected = select_kv_cache(cache, all_indices, device='cpu')

        # Reposition with zero delta (identity)
        same_pos = torch.arange(1, 11)
        reposition_kv_cache(selected, same_pos, same_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(4):
            diff = (selected.layers[L].keys - keys_orig[L]).abs().max().item()
            assert diff == 0.0, f"Layer {L}: select_all + identity reposition changed keys: {diff}"

    def test_select_reposition_roundtrip_recovers_values(self):
        """select → reposition(A→B) → reposition(B→A) preserves doc key values in fp32."""
        cache = _make_dummy_cache(n_layers=4, seq_len=21, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()

        # Simulate: BOS at 0, prefix at 1..10, doc at 11..20
        doc_indices = [0] + list(range(11, 21))  # BOS + doc
        doc_keys_orig = [cache.layers[L].keys[:, :, 11:21, :].clone() for L in range(4)]

        # Select BOS + doc
        selected = select_kv_cache(cache, doc_indices, device='cpu')
        assert selected.get_seq_length() == 11  # BOS + 10 doc

        # Reposition doc from positions 11..20 to 1..10
        old_pos = torch.arange(11, 21)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(selected, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        # Reposition BACK from 1..10 to 11..20
        reposition_kv_cache(selected, new_pos, old_pos, inv_freqs, layer_types, bos_start=0)

        # Doc keys should match originals
        for L in range(4):
            recovered = selected.layers[L].keys[:, :, 1:, :]
            diff = (recovered - doc_keys_orig[L]).abs().max().item()
            assert diff < 1e-5, f"Layer {L} round-trip through select+reposition: err={diff:.2e}"

    def test_normalize_after_reposition_roundtrip_is_idempotent(self):
        """select → reposition → normalize → normalize should equal
        select → reposition → normalize (normalization idempotent after pipeline)."""
        cache = _make_dummy_cache(n_layers=4, seq_len=21, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()

        doc_indices = [0] + list(range(11, 21))
        selected = select_kv_cache(cache, doc_indices, device='cpu')
        old_pos = torch.arange(11, 21)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(selected, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        # First normalization
        norm_roundtrip_kv_cache(selected)
        keys_after_first_norm = [selected.layers[L].keys.clone() for L in range(4)]

        # Second normalization
        norm_roundtrip_kv_cache(selected)

        for L in range(4):
            diff = (selected.layers[L].keys - keys_after_first_norm[L]).abs().max().item()
            assert diff < 1e-5, f"Layer {L} norm not idempotent after pipeline: {diff:.2e}"

    def test_cache_length_consistent_through_pipeline(self):
        """Cache length should be correct at every pipeline stage."""
        cache = _make_dummy_cache(n_layers=4, seq_len=30, dtype=torch.float32)
        inv_freqs, layer_types = self._make_test_setup()

        # Stage 0: original cache
        assert cache.get_seq_length() == 30

        # Stage 1: select BOS + 10 doc tokens (from positions 15..24)
        keep = [0] + list(range(15, 25))
        selected = select_kv_cache(cache, keep, device='cpu')
        assert selected.get_seq_length() == 11, f"After select: {selected.get_seq_length()}"

        # Verify all layers consistent
        for L in range(4):
            assert selected.layers[L].keys.shape[2] == 11
            assert selected.layers[L].values.shape[2] == 11

        # Stage 2: reposition (doesn't change length)
        old_pos = torch.arange(15, 25)
        new_pos = torch.arange(1, 11)
        reposition_kv_cache(selected, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)
        assert selected.get_seq_length() == 11, f"After reposition: {selected.get_seq_length()}"

        # Stage 3: normalize (doesn't change length)
        norm_roundtrip_kv_cache(selected)
        assert selected.get_seq_length() == 11, f"After normalize: {selected.get_seq_length()}"

        # Stage 4: deep copy (doesn't change length)
        copy = deep_copy_cache(selected)
        assert copy.get_seq_length() == 11, f"After deep copy: {copy.get_seq_length()}"

    def test_reposition_with_different_inv_freqs_gives_different_results(self):
        """Using Gemma sliding (theta=10K) vs full (theta=1M) inv_freqs
        should produce measurably different repositioned keys. This verifies
        that the per-layer-type dispatch in model_adapters is actually working."""
        cache = _make_dummy_cache(n_layers=2, seq_len=11, dtype=torch.float32, head_dim=128)
        inv_freq_10k = 1.0 / (10000.0 ** (
            torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        inv_freq_1m = 1.0 / (1000000.0 ** (
            torch.arange(0, 128, 2, dtype=torch.float32) / 128))

        # Same cache, same positions, different inv_freqs
        cache_a = deep_copy_cache(cache)
        cache_b = deep_copy_cache(cache)

        old_pos = torch.arange(100, 110)
        new_pos = torch.arange(1, 11)

        reposition_kv_cache(cache_a, old_pos, new_pos,
                            {"all": inv_freq_10k}, ["all"] * 2, bos_start=0)
        reposition_kv_cache(cache_b, old_pos, new_pos,
                            {"all": inv_freq_1m}, ["all"] * 2, bos_start=0)

        diff = (cache_a.layers[0].keys[:, :, 1:, :] -
                cache_b.layers[0].keys[:, :, 1:, :]).abs().max().item()
        assert diff > 0.01, (
            f"Different theta values produced same result (diff={diff:.4f}). "
            f"Per-layer-type dispatch may be broken.")

    def test_reposition_large_delta_roundtrip(self):
        """Round-trip should work even with very large position deltas."""
        cache = _make_dummy_cache(n_layers=2, seq_len=6, dtype=torch.float32)
        inv_freqs = {"all": 1.0 / (10000.0 ** (
            torch.arange(0, 128, 2, dtype=torch.float32) / 128))}
        layer_types = ["all"] * 2
        keys_orig = [cache.layers[L].keys[:, :, 1:, :].clone() for L in range(2)]

        # Large delta: shift by 10000 positions
        old_pos = torch.arange(10000, 10005)
        new_pos = torch.arange(1, 6)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)
        reposition_kv_cache(cache, new_pos, old_pos, inv_freqs, layer_types, bos_start=0)

        for L in range(2):
            diff = (cache.layers[L].keys[:, :, 1:, :] - keys_orig[L]).abs().max().item()
            assert diff < 1e-4, f"Large-delta round-trip error: {diff:.2e}"

    def test_normalization_preserves_relative_magnitudes(self):
        """After normalization, the relative ordering of key magnitudes
        should be preserved (largest stays largest, etc.)."""
        cache = _make_dummy_cache(n_layers=2, seq_len=10, dtype=torch.float32)
        # Get per-entry norms before normalization
        norms_before = cache.layers[0].keys[0, 0, 1:, :].norm(dim=-1)
        order_before = norms_before.argsort()

        norm_roundtrip_kv_cache(cache)

        norms_after = cache.layers[0].keys[0, 0, 1:, :].norm(dim=-1)
        order_after = norms_after.argsort()

        assert torch.equal(order_before, order_after), (
            "Normalization changed the relative magnitude ordering of keys")


class TestSlidingWindowConstraints:
    """Verify sliding window limits are respected for hybrid attention models."""

    def test_gemma3_sliding_limit(self):
        """Gemma 3 (window=1024): max cache entries = 1023."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import get_sliding_cache_limit

        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="gemma3", sliding_window=1024,
                layer_types=["sliding_attention", "full_attention"],
            ))
        model = SimpleNamespace(config=cfg)
        limit = get_sliding_cache_limit(model)
        assert limit == 1023

    def test_gemma3n_sliding_limit(self):
        """Gemma 3N (window=512): max cache entries = 511."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import get_sliding_cache_limit

        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="gemma3n", sliding_window=512,
                layer_types=["sliding_attention", "full_attention"],
            ))
        model = SimpleNamespace(config=cfg)
        limit = get_sliding_cache_limit(model)
        assert limit == 511

    def test_no_sliding_returns_none(self):
        """Models without sliding attention should return None."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import get_sliding_cache_limit

        for model_type in ["llama", "mistral", "qwen2"]:
            cfg = SimpleNamespace(model_type=model_type, num_hidden_layers=32,
                                  rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})
            model = SimpleNamespace(config=cfg)
            assert get_sliding_cache_limit(model) is None, f"{model_type} should have no sliding limit"

    def test_max_doc_respects_sliding_limit(self):
        """With L=64 prefix and 1-token NL, BOS+doc must fit in sliding limit."""
        # Gemma 3: limit=1023, BOS=1, so max doc = 1023 - 1 = 1022
        # With prefix: need BOS + prefix + NL + doc in Phase A,
        # then select BOS + doc = 1 + D <= 1023, so D <= 1022
        # max_doc for prefix conditions = 1023 - 1 - 64 - 1 = 957
        sliding_limit = 1023
        prefix_l = 64
        nl_len = 1
        max_doc = sliding_limit - 1 - prefix_l - nl_len  # BOS + prefix + NL + doc
        selected_entries = 1 + max_doc  # BOS + doc after selection
        assert selected_entries <= sliding_limit, (
            f"Selected {selected_entries} > limit {sliding_limit}")

        # Gemma 3N: limit=511, tighter constraint
        sliding_limit_3n = 511
        max_doc_3n = sliding_limit_3n - 1 - prefix_l - nl_len
        assert max_doc_3n == 445
        assert 1 + max_doc_3n <= sliding_limit_3n

    def test_multi_layer_type_inv_freqs(self):
        """Gemma models have different inv_freqs for sliding vs full attention.
        Both must be present and different."""
        from types import SimpleNamespace
        sys.path.insert(0, "/home/jupyter/research/directed_kvcache_publication")
        from model_adapters import build_layer_inv_freqs

        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="gemma3", head_dim=256, num_hidden_layers=6,
                layer_types=["sliding_attention"] * 5 + ["full_attention"],
                rope_parameters={
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                },
            ))
        model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1)]))
        inv_freqs = build_layer_inv_freqs(model, device="cpu")

        assert "sliding_attention" in inv_freqs
        assert "full_attention" in inv_freqs
        # Different theta should produce different inv_freqs
        assert not torch.allclose(inv_freqs["sliding_attention"],
                                  inv_freqs["full_attention"]), (
            "Sliding and full attention should have different inv_freqs "
            "due to different rope_theta values")
        # Sliding (theta=10K) should have larger inv_freq values than full (theta=1M)
        # at non-zero indices (index 0 is always 1.0 for both)
        assert inv_freqs["sliding_attention"][10] > inv_freqs["full_attention"][10], (
            "Sliding (theta=10K) should have larger inv_freq than full (theta=1M) at mid-freq")

    def test_reposition_uses_correct_inv_freq_per_layer(self):
        """In a mixed cache, sliding and full layers must use their respective inv_freqs."""
        # Create a cache with 4 layers: 3 sliding + 1 full (like Gemma)
        head_dim = 128
        inv_freq_sliding = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        inv_freq_full = 1.0 / (1000000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        inv_freqs = {
            "sliding_attention": inv_freq_sliding,
            "full_attention": inv_freq_full,
        }
        layer_types = ["sliding_attention", "sliding_attention",
                       "sliding_attention", "full_attention"]

        cache = _make_dummy_cache(n_layers=4, seq_len=6, head_dim=head_dim,
                                  dtype=torch.float32)
        keys_before = [cache.layers[L].keys[:, :, 1:, :].clone() for L in range(4)]

        old_pos = torch.arange(100, 105)
        new_pos = torch.arange(1, 6)
        reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

        # Sliding layers (0-2) should all change by the same amount (same inv_freq)
        diff_sliding_0 = (cache.layers[0].keys[:, :, 1:, :] - keys_before[0]).abs().max().item()
        diff_sliding_1 = (cache.layers[1].keys[:, :, 1:, :] - keys_before[1]).abs().max().item()
        # Full layer (3) should change by a DIFFERENT amount
        diff_full = (cache.layers[3].keys[:, :, 1:, :] - keys_before[3]).abs().max().item()

        assert diff_sliding_0 > 0.01, "Sliding layer 0 should be modified"
        assert diff_full > 0.01, "Full layer should be modified"
        # Full layer uses theta=1M (much slower rotation), so change should be smaller
        assert diff_full < diff_sliding_0, (
            f"Full attention (theta=1M) should have smaller change than sliding (theta=10K): "
            f"full={diff_full:.4f}, sliding={diff_sliding_0:.4f}")


class TestScoringLogic:
    """Verify NLL scoring computation is correct."""

    def test_cross_entropy_manual_match(self):
        """PyTorch cross_entropy should match manual computation."""
        logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        nll_pt = torch.nn.functional.cross_entropy(logits, targets).item()
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_manual = -log_softmax[torch.arange(2), targets].mean().item()
        assert abs(nll_pt - nll_manual) < 1e-5, f"PT={nll_pt}, manual={nll_manual}"

    def test_answer_token_indexing(self):
        """Verify the answer token slicing logic used in scoring functions."""
        # Simulate: [NL, query_0, query_1, NL, answer_0, answer_1]
        # logits shape: (6, vocab_size)
        # To predict answer_0, we use logits at position 3 (NL before answer)
        # To predict answer_1, we use logits at position 4 (answer_0)
        nl_len = 1
        query_len = 2
        answer_len = 2
        ans_start = nl_len + query_len + nl_len  # = 4

        # The slice logits[ans_start - 1 : ans_start - 1 + answer_len]
        # = logits[3:5] — positions 3 and 4
        # These predict tokens AT positions 4 and 5 (answer_0 and answer_1)
        assert ans_start - 1 == 3
        assert ans_start - 1 + answer_len == 5

        # Verify with multi-token newline (like Mistral: NL = 2 tokens)
        nl_len_multi = 2
        ans_start_multi = nl_len_multi + query_len + nl_len_multi  # = 6
        assert ans_start_multi - 1 == 5
