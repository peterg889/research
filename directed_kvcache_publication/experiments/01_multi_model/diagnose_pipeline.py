"""Diagnose the two-phase pipeline on each model.

Tests:
1. RoPE round-trip: reposition A→B then B→A, keys should be identical
2. Bare two-phase vs single-pass: NLLs should match exactly (no prefix = no reposition)
3. Prefix two-phase vs single-pass: quantify the gap
4. Normalization ablation: with/without norm_roundtrip
5. BOS verification: correct cache length and position alignment
"""
import os, sys, json, time
os.umask(0o000)
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv, find_dotenv

from lib.rope import rotate_half, select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = {
    'gemma3_12b': {
        'name': 'google/gemma-3-12b-it',
        'loader': 'Gemma3ForConditionalGeneration',
    },
    'gemma3n_e4b': {
        'name': 'google/gemma-3n-e4b-it',
        'loader': 'Gemma3nForConditionalGeneration',
    },
    'mistral_7b': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'loader': 'AutoModelForCausalLM',
    },
    'qwen25_7b': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
}

# Simple test document and query
TEST_DOC = "The capital of France is Paris. It is located in northern France along the Seine river."
TEST_QUERY = "What is the capital of France?"
TEST_ANSWER = "Paris"
COMPREHEND_TEXT = "Read and comprehend this text carefully."
PREFIX_L = 64
SEED = 42

import gc

for model_key, spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# DIAGNOSING: {model_key} ({spec['name']})")
    print(f"{'#'*70}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(spec['name'], token=HF_TOKEN)

    loader = spec.get('loader', 'AutoModelForCausalLM')
    if loader == 'Gemma3ForConditionalGeneration':
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0'
        ).eval()
    elif loader == 'Gemma3nForConditionalGeneration':
        from transformers import Gemma3nForConditionalGeneration
        model = Gemma3nForConditionalGeneration.from_pretrained(
            spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0'
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0'
        ).eval()

    device = next(model.parameters()).device
    info = get_model_info(model)
    inv_freqs = build_layer_inv_freqs(model, device=device)
    layer_types = get_layer_types(model)
    sliding_limit = get_sliding_cache_limit(model)

    bos_id = tokenizer.bos_token_id
    has_bos = bos_id is not None
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    bos_offset = 1 if has_bos else 0

    print(f"  Model type: {info['model_type']}")
    print(f"  Layers: {info['num_layers']}, head_dim: {info['head_dim']}, kv_heads: {info['num_kv_heads']}")
    print(f"  BOS: {bos_id}, has_bos: {has_bos}")
    print(f"  NL tokens: {nl_ids} (len={len(nl_ids)})")
    print(f"  Sliding: {info['has_sliding']}, limit: {sliding_limit}")
    print(f"  inv_freq keys: {list(inv_freqs.keys())}")
    print(f"  layer_types unique: {set(layer_types)}")
    print(f"  rope_thetas: {info['rope_thetas']}")

    # Tokenize test data
    doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
    query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
    answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
    comprehend_ids = tokenizer.encode(COMPREHEND_TEXT, add_special_tokens=False)
    comprehend_prefix = make_prefix(comprehend_ids, PREFIX_L)

    print(f"  Doc: {len(doc_ids)} tokens, Query: {len(query_ids)}, Answer: {len(answer_ids)}")

    # ==========================================
    # TEST 1: RoPE round-trip
    # ==========================================
    print(f"\n  --- TEST 1: RoPE round-trip ---")
    with torch.no_grad():
        # Build bare cache
        bare_ids = ([bos_id] if has_bos else []) + doc_ids
        input_t = torch.tensor([bare_ids], dtype=torch.long, device=device)
        out = model(input_ids=input_t, use_cache=True)
        cache = out.past_key_values

        # Save original keys
        orig_keys = [cache.layers[L].keys.clone() for L in range(len(cache.layers))]

        # Select doc entries (skip BOS if present)
        D = len(doc_ids)
        if has_bos:
            keep = [0] + list(range(1, 1 + D))
        else:
            keep = list(range(0, D))
        cache = select_kv_cache(cache, keep, device=device)

        # Reposition forward: old positions → new positions
        old_pos = torch.arange(bos_offset, bos_offset + D, device=device)
        new_pos = torch.arange(bos_offset + 100, bos_offset + 100 + D, device=device)  # shift by 100
        cache = reposition_kv_cache(
            cache, old_pos, new_pos, inv_freqs, layer_types,
            bos_start=0 if has_bos else -1
        )

        # Reposition backward: reverse the delta
        cache = reposition_kv_cache(
            cache, new_pos, old_pos, inv_freqs, layer_types,
            bos_start=0 if has_bos else -1
        )

        # Compare keys
        max_err = 0.0
        for L in range(len(cache.layers)):
            if has_bos:
                orig_doc = orig_keys[L][:, :, 1:, :]
                rt_doc = cache.layers[L].keys[:, :, 1:, :]
            else:
                orig_doc = orig_keys[L]
                rt_doc = cache.layers[L].keys
            err = (orig_doc - rt_doc).abs().max().item()
            if err > max_err:
                max_err = err

        status = "PASS" if max_err < 1e-2 else "FAIL"
        print(f"  Round-trip max error: {max_err:.6e} [{status}]")
        if max_err > 1e-2:
            print(f"  WARNING: Large round-trip error suggests RoPE implementation mismatch!")
        del cache, orig_keys

    # ==========================================
    # TEST 2: Bare two-phase vs single-pass NLL
    # ==========================================
    print(f"\n  --- TEST 2: Bare two-phase vs single-pass ---")
    with torch.no_grad():
        # Single-pass: BOS + doc + NL + query + NL + answer
        sp_ids = ([bos_id] if has_bos else []) + doc_ids + nl_ids + query_ids + nl_ids + answer_ids
        sp_t = torch.tensor([sp_ids], dtype=torch.long, device=device)
        sp_out = model(input_ids=sp_t, use_cache=False)
        sp_logits = sp_out.logits[0]
        D = len(doc_ids)
        ans_start = bos_offset + D + len(nl_ids) + len(query_ids) + len(nl_ids)
        sp_ans_logits = sp_logits[ans_start - 1 : ans_start - 1 + len(answer_ids)]
        ans_targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
        nll_single = torch.nn.functional.cross_entropy(sp_ans_logits, ans_targets).item()

        # Two-phase bare (no prefix, no reposition, no normalization)
        bare_ids = ([bos_id] if has_bos else []) + doc_ids
        bare_t = torch.tensor([bare_ids], dtype=torch.long, device=device)
        bare_out = model(input_ids=bare_t, use_cache=True)
        cache_bare = bare_out.past_key_values

        phase_b_ids = nl_ids + query_ids + nl_ids + answer_ids
        phase_b_t = torch.tensor([phase_b_ids], dtype=torch.long, device=device)
        n_b = len(phase_b_ids)
        pos_b = torch.arange(bos_offset + D, bos_offset + D + n_b, device=device).unsqueeze(0)

        cache_copy = deep_copy_cache(cache_bare)
        b_out = model(input_ids=phase_b_t, position_ids=pos_b,
                       past_key_values=cache_copy, use_cache=False)
        b_logits = b_out.logits[0]
        b_ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
        b_ans_logits = b_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
        nll_bare_2phase = torch.nn.functional.cross_entropy(b_ans_logits, ans_targets).item()

        diff = abs(nll_single - nll_bare_2phase)
        status = "PASS" if diff < 0.01 else "FAIL"
        print(f"  Single-pass NLL: {nll_single:.6f}")
        print(f"  Bare two-phase NLL: {nll_bare_2phase:.6f}")
        print(f"  Difference: {diff:.6f} [{status}]")
        if diff > 0.01:
            print(f"  WARNING: Bare two-phase should match single-pass exactly!")
            print(f"  This indicates a bug in cache position handling.")
        del cache_bare, cache_copy

    # ==========================================
    # TEST 3: Prefix two-phase (with/without norm)
    # ==========================================
    print(f"\n  --- TEST 3: Prefix two-phase ---")
    with torch.no_grad():
        # Build cache with comprehend prefix
        prefix_ids = ([bos_id] if has_bos else []) + comprehend_prefix + nl_ids + doc_ids
        prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        p_out = model(input_ids=prefix_t, use_cache=True)
        cache_p = p_out.past_key_values

        # Select BOS + doc (skip prefix + NL)
        P = len(comprehend_prefix)
        NL = len(nl_ids)
        doc_start = bos_offset + P + NL
        if has_bos:
            keep = [0] + list(range(doc_start, doc_start + D))
        else:
            keep = list(range(doc_start, doc_start + D))
        cache_p = select_kv_cache(cache_p, keep, device=device)

        # Verify cache shape
        cache_len = cache_p.get_seq_length()
        expected_len = bos_offset + D
        print(f"  Cache length after select: {cache_len} (expected {expected_len})")

        # Reposition
        old_pos_p = torch.arange(doc_start, doc_start + D, device=device)
        new_pos_p = torch.arange(bos_offset, bos_offset + D, device=device)
        cache_p = reposition_kv_cache(
            cache_p, old_pos_p, new_pos_p, inv_freqs, layer_types,
            bos_start=0 if has_bos else -1
        )

        # Score WITHOUT normalization
        cache_no_norm = deep_copy_cache(cache_p)
        b_out_nn = model(input_ids=phase_b_t, position_ids=pos_b,
                          past_key_values=cache_no_norm, use_cache=False)
        nn_logits = b_out_nn.logits[0]
        nn_ans_logits = nn_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
        nll_prefix_no_norm = torch.nn.functional.cross_entropy(nn_ans_logits, ans_targets).item()

        # Score WITH normalization
        cache_normed = norm_roundtrip_kv_cache(cache_p)
        cache_normed_copy = deep_copy_cache(cache_normed)
        b_out_n = model(input_ids=phase_b_t, position_ids=pos_b,
                         past_key_values=cache_normed_copy, use_cache=False)
        n_logits = b_out_n.logits[0]
        n_ans_logits = n_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
        nll_prefix_normed = torch.nn.functional.cross_entropy(n_ans_logits, ans_targets).item()

        print(f"  Single-pass NLL:          {nll_single:.6f}")
        print(f"  Bare two-phase NLL:       {nll_bare_2phase:.6f}")
        print(f"  Prefix (no norm) NLL:     {nll_prefix_no_norm:.6f}")
        print(f"  Prefix (with norm) NLL:   {nll_prefix_normed:.6f}")
        print(f"  Norm effect:              {nll_prefix_no_norm - nll_prefix_normed:+.6f} (positive = norm helps)")
        del cache_p, cache_no_norm, cache_normed, cache_normed_copy

    # ==========================================
    # TEST 4: BOS and position verification
    # ==========================================
    print(f"\n  --- TEST 4: Position verification ---")
    with torch.no_grad():
        # Check that cache.get_seq_length() matches what we expect
        bare_ids = ([bos_id] if has_bos else []) + doc_ids
        bare_t = torch.tensor([bare_ids], dtype=torch.long, device=device)
        bare_out = model(input_ids=bare_t, use_cache=True)
        cache_check = bare_out.past_key_values
        full_len = cache_check.get_seq_length()
        expected_full = bos_offset + D
        print(f"  Full cache length: {full_len} (expected {expected_full}) {'PASS' if full_len == expected_full else 'FAIL'}")

        # Check layer 0 key shape
        k0 = cache_check.layers[0].keys
        print(f"  Layer 0 key shape: {k0.shape}")
        print(f"  Expected: (1, {info['num_kv_heads']}, {expected_full}, {info['head_dim']})")
        del cache_check

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Model unloaded.\n")

print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
