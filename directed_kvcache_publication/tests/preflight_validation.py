#!/usr/bin/env python3
"""Pre-flight validation for all new models before kicking off the expanded sweep.

Loads each model one at a time and runs a compact set of checks:
1. Tokenizer: BOS, PAD, NL encoding
2. Config: build_layer_inv_freqs, get_layer_types, get_sliding_cache_limit
3. inv_freq match: compare our computed inv_freq vs model's rotary_emb buffers
4. Cache shape: verify all layers have (1, n_kv_heads, seq_len, head_dim) after Phase A
5. Bare two-phase ≈ single-pass (critical invariant)
6. Full prefix pipeline → finite, reasonable NLL

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    PYTHONPATH="../directed_kvcache_v4:." python3 tests/preflight_validation.py
"""

import os
import sys
import gc
import time
import shutil
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../directed_kvcache_v4"))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv, find_dotenv

from lib.rope import select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def purge_hf_cache(model_name):
    """Delete cached model weights from disk to free space."""
    slug = "models--" + model_name.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, slug)
    if os.path.isdir(cache_path):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(cache_path) for f in fns
        ) / 1e9
        shutil.rmtree(cache_path)
        print(f"  Purged {cache_path} ({size_gb:.1f} GB)")


# ── Test data ──────────────────────────────────────────────────────────
TEST_DOC = "The capital of France is Paris. It is located along the Seine river."
TEST_QUERY = "What is the capital of France?"
TEST_ANSWER = "Paris"
PREFIX_TEXT = "Read and comprehend this text carefully."
PREFIX_L = 16  # short prefix to stay within sliding_window=512

# ── Models to validate (all NEW models, ordered by VRAM) ──────────────
NEW_MODELS = [
    ("qwen25_0_5b",         "Qwen/Qwen2.5-0.5B-Instruct",              "AutoModelForCausalLM"),
    ("gemma3_4b_base",      "google/gemma-3-4b-pt",                     "Gemma3ForConditionalGeneration"),
    ("qwen25_7b_base",      "Qwen/Qwen2.5-7B",                         "AutoModelForCausalLM"),
    ("qwen25_32b",          "Qwen/Qwen2.5-32B-Instruct",               "AutoModelForCausalLM"),
]


def load_model(model_name, loader_name):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN,
                                               trust_remote_code=True)
    if loader_name == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader_name == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader_name == "Gemma3nForConditionalGeneration":
        from transformers import Gemma3nForConditionalGeneration
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def check_tokenizer(key, tokenizer):
    """Check 1: Tokenizer properties."""
    errors = []
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    # BOS or PAD must exist
    if bos is None and pad is None:
        errors.append("FATAL: both bos_token_id and pad_token_id are None — no attention sink")
    effective_bos = bos if bos is not None else pad

    # NL must encode to at least 1 token
    if len(nl_ids) == 0:
        errors.append("FATAL: newline encodes to 0 tokens")

    # Verify encode/decode roundtrip
    test = "hello world"
    ids = tokenizer.encode(test, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    if test not in decoded:
        errors.append(f"encode/decode roundtrip failed: '{test}' → {ids} → '{decoded}'")

    print(f"  BOS={bos}, PAD={pad}, effective_BOS={effective_bos}, "
          f"NL_ids={nl_ids} ({len(nl_ids)} tok), vocab={tokenizer.vocab_size}")
    return errors


def check_config(key, model, device):
    """Check 2: Config extraction via model_adapters."""
    errors = []
    try:
        inv_freqs = build_layer_inv_freqs(model, device=device)
        layer_types = get_layer_types(model)
        sliding_limit = get_sliding_cache_limit(model)
        info = get_model_info(model)

        # Every layer type must have a matching inv_freq
        for L, lt in enumerate(layer_types):
            if lt not in inv_freqs:
                errors.append(f"Layer {L} type '{lt}' has no inv_freq entry")

        # inv_freq shape
        head_dim = info["head_dim"]
        for lt, freq in inv_freqs.items():
            if freq.shape != (head_dim // 2,):
                errors.append(f"inv_freq['{lt}'] shape {freq.shape} != ({head_dim // 2},)")

        print(f"  layers={info['num_layers']}, head_dim={head_dim}, "
              f"kv_heads={info['num_kv_heads']}, sliding={sliding_limit}, "
              f"layer_types={list(inv_freqs.keys())}")
    except Exception as e:
        errors.append(f"Config extraction failed: {e}")
    return errors


def check_inv_freq_match(key, model, device):
    """Check 3: Our inv_freq vs model's actual rotary embedding buffers."""
    errors = []
    inv_freqs = build_layer_inv_freqs(model, device=device)
    layer_types = get_layer_types(model)
    unique_types = sorted(set(layer_types))

    # Find rotary embedding
    rotary_emb = None
    for name, module in model.named_modules():
        if "rotary_emb" in name.lower():
            rotary_emb = module
            break

    if rotary_emb is None:
        errors.append("Could not find rotary_emb module in model")
        return errors

    for lt in unique_types:
        buffer_name = f"{lt}_inv_freq"
        if hasattr(rotary_emb, buffer_name):
            actual = getattr(rotary_emb, buffer_name).float().cpu()
        elif hasattr(rotary_emb, "inv_freq"):
            actual = rotary_emb.inv_freq.float().cpu()
        else:
            print(f"  {lt}: no inv_freq buffer found (skip)")
            continue

        ours = inv_freqs[lt].float().cpu()
        if actual.shape != ours.shape:
            errors.append(f"{lt}: shape mismatch model={actual.shape} vs ours={ours.shape}")
            continue

        ratio = ours / actual
        max_err = (ratio - 1.0).abs().max().item()
        if max_err > 1e-4:
            errors.append(f"{lt}: inv_freq MISMATCH! max ratio err={max_err:.6f} "
                          f"(model[0]={actual[0]:.4e}, ours[0]={ours[0]:.4e})")
        else:
            print(f"  {lt}: inv_freq OK (max ratio err={max_err:.2e})")

    return errors


def check_cache_shape(key, model, tokenizer, device):
    """Check 4: Cache shape after Phase A — all layers must be (1, n_kv_heads, seq_len, head_dim)."""
    errors = []
    info = get_model_info(model)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id

    doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)[:50]
    input_ids = [bos_id] + doc_ids
    expected_seq = len(input_ids)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([input_ids], device=device), use_cache=True)
    cache = out.past_key_values

    n_layers = len(cache.layers)
    if n_layers != info["num_layers"]:
        errors.append(f"Cache has {n_layers} layers but model has {info['num_layers']}")

    shapes_seen = set()
    for L in range(n_layers):
        k = cache.layers[L].keys
        v = cache.layers[L].values
        shapes_seen.add(k.shape)

        # Check batch=1, head_dim matches
        if k.shape[0] != 1:
            errors.append(f"Layer {L}: batch dim = {k.shape[0]} (expected 1)")
        if k.shape[3] != info["head_dim"]:
            errors.append(f"Layer {L}: head_dim = {k.shape[3]} (expected {info['head_dim']})")
        # Check keys and values have same shape
        if k.shape != v.shape:
            errors.append(f"Layer {L}: K shape {k.shape} != V shape {v.shape}")

    # Check sequence length: all layers should have same seq_len
    # (for short sequences within any sliding window)
    seq_lengths = set(cache.layers[L].keys.shape[2] for L in range(n_layers))
    if len(seq_lengths) > 1:
        errors.append(f"Inconsistent seq lengths across layers: {seq_lengths}")

    actual_seq = list(seq_lengths)[0] if len(seq_lengths) == 1 else "mixed"
    print(f"  cache: {n_layers} layers, seq_len={actual_seq} (expected {expected_seq}), "
          f"shapes={shapes_seen}")

    if isinstance(actual_seq, int) and actual_seq != expected_seq:
        errors.append(f"Cache seq_len={actual_seq} != expected {expected_seq}")

    # Check that select_kv_cache works
    try:
        indices = [0, 1, 2]
        selected = select_kv_cache(cache, indices, device=device)
        if selected.get_seq_length() != len(indices):
            errors.append(f"select_kv_cache: got {selected.get_seq_length()} != {len(indices)}")
    except Exception as e:
        errors.append(f"select_kv_cache failed: {e}")

    # Check that deep_copy_cache works
    try:
        copy = deep_copy_cache(cache)
        if copy.get_seq_length() != cache.get_seq_length():
            errors.append("deep_copy_cache: length mismatch")
    except Exception as e:
        errors.append(f"deep_copy_cache failed: {e}")

    del cache, out
    return errors


def check_bare_two_phase(key, model, tokenizer, device):
    """Check 5: Bare two-phase must match single-pass NLL (critical invariant)."""
    errors = []
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    has_bos = True  # we always use a BOS (real or PAD)
    bos_offset = 1
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)[:50]
    query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
    answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)
    D = len(doc_ids)

    with torch.no_grad():
        # Single-pass (use_cache=True to avoid Gemma 3N-style bugs)
        sp_ids = [bos_id] + doc_ids + nl_ids + query_ids + nl_ids + answer_ids
        sp_out = model(input_ids=torch.tensor([sp_ids], device=device), use_cache=True)
        sp_logits = sp_out.logits[0]
        ans_start = bos_offset + D + len(nl_ids) + len(query_ids) + len(nl_ids)
        targets = torch.tensor(answer_ids, device=device)
        sp_ans = sp_logits[ans_start - 1 : ans_start - 1 + len(answer_ids)]
        nll_sp = torch.nn.functional.cross_entropy(sp_ans, targets).item()
        del sp_out

        # Two-phase bare (no prefix, no reposition)
        pa_ids = [bos_id] + doc_ids
        pa_out = model(input_ids=torch.tensor([pa_ids], device=device), use_cache=True)
        cache = pa_out.past_key_values
        del pa_out

        pb_ids = nl_ids + query_ids + nl_ids + answer_ids
        n_b = len(pb_ids)
        pos_b = torch.arange(bos_offset + D, bos_offset + D + n_b, device=device).unsqueeze(0)
        cache_copy = deep_copy_cache(cache)
        pb_out = model(input_ids=torch.tensor([pb_ids], device=device),
                       position_ids=pos_b, past_key_values=cache_copy, use_cache=False)
        pb_logits = pb_out.logits[0]
        b_ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
        pb_ans = pb_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
        nll_2p = torch.nn.functional.cross_entropy(pb_ans, targets).item()
        del pb_out, cache

    diff = abs(nll_sp - nll_2p)
    tol = max(0.05 * abs(nll_sp), 0.15)
    status = "OK" if diff < tol else "FAIL"
    print(f"  SP={nll_sp:.4f}, 2P={nll_2p:.4f}, diff={diff:.4f}, tol={tol:.4f} → {status}")

    if diff >= tol:
        errors.append(f"CRITICAL: bare two-phase ({nll_2p:.4f}) != single-pass ({nll_sp:.4f}), "
                      f"diff={diff:.4f} > tol={tol:.4f}")
    return errors


def check_prefix_pipeline(key, model, tokenizer, device):
    """Check 6: Full prefix pipeline → select → reposition → normalize → score."""
    errors = []
    inv_freqs = build_layer_inv_freqs(model, device=device)
    layer_types = get_layer_types(model)
    sliding_limit = get_sliding_cache_limit(model)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    bos_offset = 1
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    doc_ids = tokenizer.encode(TEST_DOC, add_special_tokens=False)
    query_ids = tokenizer.encode(TEST_QUERY, add_special_tokens=False)
    answer_ids = tokenizer.encode(TEST_ANSWER, add_special_tokens=False)

    # Use short prefix (L=16) to stay within sliding_window=512
    prefix_tok_ids = tokenizer.encode(PREFIX_TEXT, add_special_tokens=False)
    prefix = make_prefix(prefix_tok_ids, PREFIX_L)
    P = len(prefix)
    NL = len(nl_ids)

    # Truncate doc if needed for sliding window
    max_doc = 200
    if sliding_limit is not None:
        max_doc = min(max_doc, sliding_limit - 1 - P - NL)
    doc_ids = doc_ids[:max_doc]
    D = len(doc_ids)

    total_cache_entries = 1 + D  # BOS + doc (after select)
    if sliding_limit is not None and total_cache_entries > sliding_limit:
        errors.append(f"Cache overflow: {total_cache_entries} > {sliding_limit}")
        return errors

    with torch.no_grad():
        # Phase A with prefix
        pa_ids = [bos_id] + prefix + nl_ids + doc_ids
        pa_out = model(input_ids=torch.tensor([pa_ids], device=device), use_cache=True)
        cache = pa_out.past_key_values
        del pa_out

        # Verify cache length
        expected_len = 1 + P + NL + D
        actual_len = cache.get_seq_length()
        if actual_len != expected_len:
            errors.append(f"Phase A cache: {actual_len} != expected {expected_len}")

        # Select BOS + doc
        doc_start = 1 + P + NL
        keep = [0] + list(range(doc_start, doc_start + D))
        cache = select_kv_cache(cache, keep, device=device)

        # Reposition
        old_pos = torch.arange(doc_start, doc_start + D, device=device)
        new_pos = torch.arange(1, 1 + D, device=device)
        cache = reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types,
                                     bos_start=0)

        # Normalize
        cache = norm_roundtrip_kv_cache(cache)

        # Phase B: score answer
        pb_ids = nl_ids + query_ids + nl_ids + answer_ids
        n_b = len(pb_ids)
        pos_b = torch.arange(bos_offset + D, bos_offset + D + n_b, device=device).unsqueeze(0)
        cache_copy = deep_copy_cache(cache)
        pb_out = model(input_ids=torch.tensor([pb_ids], device=device),
                       position_ids=pos_b, past_key_values=cache_copy, use_cache=False)
        pb_logits = pb_out.logits[0]
        b_ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
        pb_ans = pb_logits[b_ans_start - 1 : b_ans_start - 1 + len(answer_ids)]
        targets = torch.tensor(answer_ids, device=device)
        nll = torch.nn.functional.cross_entropy(pb_ans, targets).item()
        del pb_out, cache

    is_ok = np.isfinite(nll) and 0 < nll < 20
    status = "OK" if is_ok else "FAIL"
    print(f"  prefix pipeline NLL={nll:.4f} → {status}")

    if not np.isfinite(nll):
        errors.append(f"NLL is not finite: {nll}")
    elif nll <= 0:
        errors.append(f"NLL is non-positive: {nll}")
    elif nll >= 20:
        errors.append(f"NLL={nll:.1f} is unreasonably high (garbage output)")

    return errors


def check_position_shift_reposition(key, model, tokenizer, device):
    """Check 7: Position-shift-only reposition should recover original keys.

    This isolates RoPE correctness from attention effects:
    1. Encode [BOS, doc] at natural positions [0, 1..D]
    2. Encode the SAME [BOS, doc] at shifted positions [0, S+1..S+D] (no actual prefix)
    3. Reposition shifted keys from [S+1..S+D] back to [1..D]
    4. Compare repositioned keys with step 1 — should be close in fp32

    Note: the keys from step 2 differ from step 1 ONLY by RoPE position encoding
    (same tokens, same attention pattern, just different position_ids). So after
    perfect RoPE correction, they should match exactly in fp32 and closely in bf16.
    """
    errors = []
    inv_freqs = build_layer_inv_freqs(model, device=device)
    layer_types = get_layer_types(model)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    doc_ids = tokenizer.encode("The quick brown fox.", add_special_tokens=False)[:20]
    D = len(doc_ids)
    SHIFT = 32  # shift doc positions by 32

    with torch.no_grad():
        # Step 1: encode at natural positions [0, 1..D]
        ids = [bos_id] + doc_ids
        ids_tensor = torch.tensor([ids], device=device)
        natural_out = model(input_ids=ids_tensor, use_cache=True)
        natural_cache = natural_out.past_key_values
        del natural_out

        # Step 2: encode SAME tokens at shifted positions [0, SHIFT+1..SHIFT+D]
        shifted_pos = torch.cat([
            torch.tensor([0], device=device),
            torch.arange(SHIFT + 1, SHIFT + 1 + D, device=device)
        ]).unsqueeze(0)
        shifted_out = model(input_ids=ids_tensor, position_ids=shifted_pos, use_cache=True)
        shifted_cache = shifted_out.past_key_values
        del shifted_out

        # Step 3: reposition shifted keys back to natural positions
        old_pos = torch.arange(SHIFT + 1, SHIFT + 1 + D, device=device)
        new_pos = torch.arange(1, 1 + D, device=device)
        repositioned = reposition_kv_cache(shifted_cache, old_pos, new_pos,
                                            inv_freqs, layer_types, bos_start=0)

    # Step 4: compare
    # Only layer 0 has the exact "keys differ only by RoPE" property, because
    # deeper layers receive different hidden states (earlier layers' attention
    # outputs differ due to position-dependent relative distances). Layer 0
    # keys come directly from embeddings, so repositioning should recover them.
    layer0_nat = natural_cache.layers[0].keys[:, :, 1:1+D, :].float()
    layer0_rep = repositioned.layers[0].keys[:, :, 1:1+D, :].float()
    layer0_err = (layer0_nat - layer0_rep).abs().max().item()

    # Also report worst-layer error for diagnostics (not asserted)
    max_err = layer0_err
    worst_layer = 0
    for L in range(1, len(layer_types)):
        nat_k = natural_cache.layers[L].keys[:, :, 1:1+D, :].float()
        rep_k = repositioned.layers[L].keys[:, :, 1:1+D, :].float()
        err = (nat_k - rep_k).abs().max().item()
        if err > max_err:
            max_err = err
            worst_layer = L

    # bf16 reposition introduces error from cos/sin precision loss.
    # Models with low rope_theta (e.g. DeepSeek at 10K) have higher error
    # because high-frequency components lose more precision in bf16.
    # Typical: 0.03-0.25 for theta≥100K, up to 1.0 for theta=10K.
    status = "OK" if layer0_err < 1.5 else "FAIL"
    lt = layer_types[worst_layer] if worst_layer >= 0 else "?"
    print(f"  layer 0 reposition err={layer0_err:.4f}, "
          f"worst layer {worst_layer} [{lt}] err={max_err:.4f} → {status}")

    if layer0_err >= 1.5:
        errors.append(f"Layer 0 position-shift reposition error too large: {layer0_err:.4f}. "
                      f"RoPE inv_freq may be wrong.")

    del natural_cache, repositioned
    return errors


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"PRE-FLIGHT VALIDATION — {len(NEW_MODELS)} NEW MODELS")
    print("=" * 70)

    results = {}
    total_errors = 0

    for key, model_name, loader in NEW_MODELS:
        print(f"\n{'─' * 70}")
        print(f"  {key}: {model_name}")
        print(f"{'─' * 70}")
        t0 = time.time()
        model_errors = []

        try:
            # Load
            print(f"  Loading model...")
            model, tokenizer, device = load_model(model_name, loader)
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Loaded in {time.time() - t0:.0f}s, VRAM: {vram_gb:.1f} GB")

            # Check 1: Tokenizer
            print(f"\n  [1/7] Tokenizer")
            model_errors.extend(check_tokenizer(key, tokenizer))

            # Check 2: Config
            print(f"\n  [2/7] Config extraction")
            model_errors.extend(check_config(key, model, device))

            # Check 3: inv_freq match
            print(f"\n  [3/7] inv_freq vs model buffers")
            model_errors.extend(check_inv_freq_match(key, model, device))

            # Check 4: Cache shape
            print(f"\n  [4/7] Cache shape after Phase A")
            model_errors.extend(check_cache_shape(key, model, tokenizer, device))

            # Check 5: Bare two-phase
            print(f"\n  [5/7] Bare two-phase ≈ single-pass")
            model_errors.extend(check_bare_two_phase(key, model, tokenizer, device))

            # Check 6: Full prefix pipeline
            print(f"\n  [6/7] Full prefix pipeline")
            model_errors.extend(check_prefix_pipeline(key, model, tokenizer, device))

            # Check 7: Position-shift-only reposition (isolates RoPE from attention)
            print(f"\n  [7/7] Position-shift reposition correctness")
            model_errors.extend(check_position_shift_reposition(key, model, tokenizer, device))

        except Exception as e:
            model_errors.append(f"UNHANDLED EXCEPTION: {e}\n{traceback.format_exc()}")
        finally:
            # Unload model from GPU and purge weights from disk
            try:
                del model, tokenizer
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            purge_hf_cache(model_name)

        elapsed = time.time() - t0
        n_err = len(model_errors)
        total_errors += n_err
        status = "PASS" if n_err == 0 else f"FAIL ({n_err} errors)"
        results[key] = {"status": status, "errors": model_errors, "time": elapsed}

        print(f"\n  ── {key}: {status} ({elapsed:.0f}s) ──")
        for err in model_errors:
            print(f"    ERROR: {err}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for key, res in results.items():
        print(f"  {key:<25s} {res['status']:<20s} ({res['time']:.0f}s)")
    print(f"\n  Total: {len(results)} models, {total_errors} errors")
    if total_errors == 0:
        print("  ALL MODELS PASSED — ready to launch expanded sweep")
    else:
        print("  *** ERRORS FOUND — fix before launching sweep ***")
    print(f"{'=' * 70}")

    return total_errors


if __name__ == "__main__":
    sys.exit(main())
