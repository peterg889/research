#!/usr/bin/env python3
"""Comprehensive tests for prefix LM Phase A/B attention masks.

Tests both pure math properties (no model) and model-based equivalences.

Usage:
    python test_attention_masks.py              # Pure math tests only
    python test_attention_masks.py --model      # All tests (loads Gemma 3 12B)
"""

import argparse
import torch
import torch.nn.functional as F
import gc
import os
import sys
import time

sys.path.insert(0, "../../..")

DTYPE = torch.bfloat16


# ====================================================================
# Mask functions (same as notebook Cell 3)
# ====================================================================

def make_phase_a_mask(n_s, n_d, mode="prefix_lm", dtype=DTYPE):
    # Phase A mask for prefix [BOS, surrogate, doc].
    # Returns (1, 1, n_prefix, n_prefix).
    n_prefix = 1 + n_s + n_d
    min_val = torch.finfo(dtype).min
    if mode == "prefix_lm":
        mask = torch.zeros((n_prefix, n_prefix), dtype=dtype)
    else:
        mask = torch.triu(torch.full((n_prefix, n_prefix), min_val, dtype=dtype),
                          diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_phase_b_mask(n_s, n_d, n_q, n_a, truncate=True, dtype=DTYPE):
    # Phase B mask for continuation [query, answer] attending to cached prefix.
    # Returns (1, 1, n_cont, n_prefix + n_cont).
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min

    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)
    mask[:, :n_prefix] = 0.0

    if truncate and n_s > 0:
        mask[:, 1:1 + n_s] = min_val

    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )

    return mask.unsqueeze(0).unsqueeze(0)


def make_single_pass_mask(n_s, n_d, n_q, n_a, mode="prefix_lm", truncate=True,
                          dtype=DTYPE):
    # Single-pass mask for [BOS, surr, doc, query, answer] (for equivalence tests).
    # Returns (1, 1, L, L).
    L = 1 + n_s + n_d + n_q + n_a
    n_prefix = 1 + n_s + n_d
    min_val = torch.finfo(dtype).min

    mask = torch.triu(torch.full((L, L), min_val, dtype=dtype), diagonal=1)

    if mode == "prefix_lm":
        mask[:n_prefix, :n_prefix] = 0.0

    if truncate and n_s > 0:
        mask[n_prefix:, 1:1 + n_s] = min_val

    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


# ====================================================================
# PART 1: PURE MATH TESTS
# ====================================================================

def run_math_tests():
    print("=" * 70)
    print("PART 1: PURE MATH MASK PROPERTY VERIFICATION")
    print("=" * 70)

    min_val = torch.finfo(DTYPE).min

    # --- Test 1: Phase A mask properties ---
    print("\n--- Test 1: Phase A mask properties ---")
    n_pass = n_fail = 0

    test_configs_a = [
        (0, 5, "prefix_lm", "plm_bare"),
        (0, 5, "causal",    "causal_bare"),
        (3, 5, "prefix_lm", "plm_with_surr"),
        (3, 5, "causal",    "causal_with_surr"),
        (0, 1, "prefix_lm", "plm_single_doc"),
        (0, 1, "causal",    "causal_single_doc"),
    ]

    for n_s, n_d, mode, label in test_configs_a:
        mask = make_phase_a_mask(n_s, n_d, mode)
        m = mask[0, 0]
        n_prefix = 1 + n_s + n_d
        assert m.shape == (n_prefix, n_prefix), f"Wrong shape: {m.shape}"
        errors = []

        for i in range(n_prefix):
            for j in range(n_prefix):
                if mode == "prefix_lm":
                    expected_attend = True
                else:
                    expected_attend = (j <= i)

                actual_attend = (m[i, j] == 0.0)
                if actual_attend == expected_attend:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(errors) < 3:
                        errors.append(
                            f"    [{i},{j}]: got {'attend' if actual_attend else 'masked'}, "
                            f"want {'attend' if expected_attend else 'masked'}")

        status = "PASS" if not errors else "FAIL"
        print(f"  {status}: {label} ({n_prefix}x{n_prefix})")
        for e in errors:
            print(e)

    print(f"  Total: {n_pass} passed, {n_fail} failed")
    assert n_fail == 0, f"Phase A tests FAILED ({n_fail} cells wrong)"

    # --- Test 2: Phase B mask properties ---
    print("\n--- Test 2: Phase B mask properties ---")
    n_pass = n_fail = 0

    test_configs_b = [
        (0, 5, 3, 2, False, "bare_q"),
        (3, 5, 3, 2, True,  "oracle_trunc_q"),
        (3, 5, 3, 2, False, "oracle_notrunc_q"),
        (0, 5, 0, 2, False, "bare_nq"),
        (3, 5, 0, 2, True,  "oracle_trunc_nq"),
        (3, 5, 0, 4, True,  "oracle_trunc_nq_long_answer"),
    ]

    for n_s, n_d, n_q, n_a, truncate, label in test_configs_b:
        mask = make_phase_b_mask(n_s, n_d, n_q, n_a, truncate)
        m = mask[0, 0]
        n_prefix = 1 + n_s + n_d
        n_cont = n_q + n_a
        assert m.shape == (n_cont, n_prefix + n_cont), f"Wrong shape: {m.shape}"
        errors = []

        for i in range(n_cont):
            for j in range(n_prefix + n_cont):
                if j < n_prefix:
                    # Cached prefix positions
                    j_is_surr = (1 <= j < 1 + n_s) if n_s > 0 else False
                    if truncate and j_is_surr:
                        expected_attend = False
                    else:
                        expected_attend = True
                else:
                    # Continuation positions: causal
                    j_cont = j - n_prefix
                    expected_attend = (j_cont <= i)

                actual_attend = (m[i, j] == 0.0)
                if actual_attend == expected_attend:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(errors) < 3:
                        errors.append(
                            f"    [{i},{j}]: got {'attend' if actual_attend else 'masked'}, "
                            f"want {'attend' if expected_attend else 'masked'}")

        status = "PASS" if not errors else "FAIL"
        print(f"  {status}: {label} ({n_cont}x{n_prefix + n_cont})")
        for e in errors:
            print(e)

    print(f"  Total: {n_pass} passed, {n_fail} failed")
    assert n_fail == 0, f"Phase B tests FAILED ({n_fail} cells wrong)"

    # --- Test 3: Phase A + Phase B compose to single-pass mask ---
    print("\n--- Test 3: Two-phase mask composes to single-pass mask ---")
    n_pass = n_fail = 0

    compose_configs = [
        (0, 5, 3, 2, "prefix_lm", False, "plm_bare"),
        (3, 5, 3, 2, "prefix_lm", True,  "plm_oracle_trunc"),
        (0, 5, 3, 2, "causal",    False, "causal_bare"),
        (3, 5, 3, 2, "causal",    True,  "causal_oracle_trunc"),
        (0, 5, 0, 2, "causal",    False, "causal_bare_nq"),
        (3, 5, 0, 2, "prefix_lm", True,  "plm_oracle_trunc_nq"),
    ]

    for n_s, n_d, n_q, n_a, mode, truncate, label in compose_configs:
        n_prefix = 1 + n_s + n_d
        n_cont = n_q + n_a

        # Single-pass mask
        sp = make_single_pass_mask(n_s, n_d, n_q, n_a, mode, truncate)[0, 0]

        # Phase A mask: prefix-to-prefix block
        pa = make_phase_a_mask(n_s, n_d, mode)[0, 0]

        # Phase B mask: continuation rows
        pb = make_phase_b_mask(n_s, n_d, n_q, n_a, truncate)[0, 0]

        errors = []

        # Check 1: Phase A mask == single-pass prefix-to-prefix block
        for i in range(n_prefix):
            for j in range(n_prefix):
                a_attend = (pa[i, j] == 0.0)
                sp_attend = (sp[i, j] == 0.0)
                if a_attend == sp_attend:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(errors) < 3:
                        errors.append(f"    PhaseA[{i},{j}] vs SP[{i},{j}]")

        # Check 2: Phase B mask == single-pass continuation rows
        # Phase B columns: [prefix (n_prefix)] [continuation (n_cont)]
        # Single-pass row n_prefix+i: columns [0..n_prefix-1] then [n_prefix..L-1]
        for i in range(n_cont):
            sp_row = n_prefix + i
            for j in range(n_prefix + n_cont):
                b_attend = (pb[i, j] == 0.0)
                sp_attend = (sp[sp_row, j] == 0.0)
                if b_attend == sp_attend:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(errors) < 3:
                        errors.append(
                            f"    PhaseB[{i},{j}] vs SP[{sp_row},{j}]: "
                            f"B={'attend' if b_attend else 'mask'} "
                            f"SP={'attend' if sp_attend else 'mask'}")

        # Check 3: Single-pass prefix-to-continuation block is all masked
        # (prefix cannot see continuation)
        for i in range(n_prefix):
            for j in range(n_prefix, n_prefix + n_cont):
                sp_attend = (sp[i, j] == 0.0)
                if not sp_attend:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(errors) < 3:
                        errors.append(f"    SP[{i},{j}] prefix sees continuation!")

        status = "PASS" if not errors else "FAIL"
        print(f"  {status}: {label}")
        for e in errors:
            print(e)

    print(f"  Total: {n_pass} passed, {n_fail} failed")
    assert n_fail == 0, f"Composition tests FAILED ({n_fail} cells wrong)"

    # --- Test 4: Edge cases ---
    print("\n--- Test 4: Edge cases ---")

    # Single answer token, no query
    mask = make_phase_b_mask(0, 5, 0, 1, truncate=False)
    assert mask.shape == (1, 1, 1, 7), f"Wrong shape for n_a=1: {mask.shape}"
    # Should attend to all 6 prefix positions, and to itself
    assert (mask[0, 0, 0, :6] == 0.0).all(), "n_a=1: should attend to all prefix"
    assert mask[0, 0, 0, 6] == 0.0, "n_a=1: should attend to itself"
    print(f"  PASS: Single answer token (n_a=1, n_q=0)")

    # Bare prefix (n_s=0) with truncate=True should be same as truncate=False
    m_t = make_phase_b_mask(0, 5, 3, 2, truncate=True)
    m_f = make_phase_b_mask(0, 5, 3, 2, truncate=False)
    assert torch.equal(m_t, m_f), "Truncation should have no effect when n_s=0"
    print(f"  PASS: truncate=True with n_s=0 is no-op")

    # Large surrogate
    mask = make_phase_b_mask(20, 5, 3, 2, truncate=True)
    n_prefix = 1 + 20 + 5
    # First row: BOS (pos 0) attend, surr (1-20) masked, doc (21-25) attend
    assert mask[0, 0, 0, 0] == 0.0, "Should attend to BOS"
    assert (mask[0, 0, 0, 1:21] != 0.0).all(), "Should mask all 20 surrogate tokens"
    assert (mask[0, 0, 0, 21:26] == 0.0).all(), "Should attend to all 5 doc tokens"
    print(f"  PASS: Large surrogate (n_s=20)")

    print("\n  All pure math tests PASSED.")


# ====================================================================
# VISUAL MASK DISPLAYS
# ====================================================================

def show_visual():
    print("\n" + "=" * 70)
    print("VISUAL MASK DISPLAYS")
    print("=" * 70)

    # Phase A: prefix_lm with surrogate
    print("\n--- Phase A: prefix_lm, n_s=3, n_d=4 ---")
    mask_a = make_phase_a_mask(3, 4, "prefix_lm")
    m = mask_a[0, 0]
    labels = ['B'] + ['S']*3 + ['D']*4
    print("     " + " ".join(f"{l}" for l in labels))
    for i in range(m.shape[0]):
        row = ""
        for j in range(m.shape[1]):
            row += " ." if m[i, j] == 0.0 else " x"
        print(f"  {labels[i]} {row}")
    print("  (. = attend, x = masked)")

    # Phase A: causal with surrogate
    print("\n--- Phase A: causal, n_s=3, n_d=4 ---")
    mask_a_c = make_phase_a_mask(3, 4, "causal")
    mc = mask_a_c[0, 0]
    print("     " + " ".join(f"{l}" for l in labels))
    for i in range(mc.shape[0]):
        row = ""
        for j in range(mc.shape[1]):
            row += " ." if mc[i, j] == 0.0 else " x"
        print(f"  {labels[i]} {row}")
    print("  (. = attend, x = masked)")
    print("  Note: under causal, doc tokens CAN see surrogate (doc comes after surr).")

    # Phase B: with query, truncated
    print("\n--- Phase B: n_s=3, n_d=4, n_q=2, n_a=2, truncated ---")
    mask_b = make_phase_b_mask(3, 4, 2, 2, truncate=True)
    mb = mask_b[0, 0]
    col_labels = ['B'] + ['S']*3 + ['D']*4 + ['Q']*2 + ['A']*2
    row_labels = ['Q']*2 + ['A']*2
    print("     " + " ".join(f"{l}" for l in col_labels))
    for i in range(mb.shape[0]):
        row = ""
        for j in range(mb.shape[1]):
            row += " ." if mb[i, j] == 0.0 else " x"
        print(f"  {row_labels[i]} {row}")
    print("  (. = attend, x = masked)")
    print("  Columns 0-7: cached prefix (BOS, surr, doc). Columns 8-11: continuation.")

    # Phase B: no query, truncated
    print("\n--- Phase B: n_s=3, n_d=4, n_q=0, n_a=3, truncated ---")
    mask_b_nq = make_phase_b_mask(3, 4, 0, 3, truncate=True)
    mbnq = mask_b_nq[0, 0]
    col_labels_nq = ['B'] + ['S']*3 + ['D']*4 + ['A']*3
    row_labels_nq = ['A']*3
    print("     " + " ".join(f"{l}" for l in col_labels_nq))
    for i in range(mbnq.shape[0]):
        row = ""
        for j in range(mbnq.shape[1]):
            row += " ." if mbnq[i, j] == 0.0 else " x"
        print(f"  {row_labels_nq[i]} {row}")
    print("  (. = attend, x = masked)")

    # Single-pass equivalent for comparison
    print("\n--- Single-pass: prefix_lm, n_s=3, n_d=4, n_q=2, n_a=2, truncated ---")
    mask_sp = make_single_pass_mask(3, 4, 2, 2, "prefix_lm", True)
    msp = mask_sp[0, 0]
    all_labels = ['B'] + ['S']*3 + ['D']*4 + ['Q']*2 + ['A']*2
    print("     " + " ".join(f"{l}" for l in all_labels))
    for i in range(msp.shape[0]):
        row = ""
        for j in range(msp.shape[1]):
            row += " ." if msp[i, j] == 0.0 else " x"
        print(f"  {all_labels[i]} {row}")
    print("  (. = attend, x = masked)")
    print("  This should match Phase A (upper-left) + Phase B (bottom rows).")


# ====================================================================
# PART 2: MODEL-BASED TESTS
# ====================================================================

def run_model_tests():
    print("\n" + "=" * 70)
    print("PART 2: MODEL-BASED MASK TESTS")
    print("=" * 70)

    os.umask(0o000)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    HF_TOKEN = os.environ.get("HF_TOKEN")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "google/gemma-3-12b-it"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers

    print(f"\ntransformers version: {transformers.__version__}")
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")

    # Build test sequence
    test_surr = "hello world test"
    test_doc = "The quick brown fox jumps over the lazy dog near the river bank."
    test_query = "What animal jumps?"
    test_answer = "A brown fox."

    surr_ids = tokenizer(test_surr, add_special_tokens=False).input_ids
    doc_ids_t = tokenizer(test_doc, add_special_tokens=False).input_ids
    query_ids_t = tokenizer(test_query, add_special_tokens=False).input_ids
    answer_ids_t = tokenizer(test_answer, add_special_tokens=False).input_ids
    bos_id = tokenizer.bos_token_id

    n_s_t = len(surr_ids)
    n_d_t = len(doc_ids_t)
    n_q_t = len(query_ids_t)
    n_a_t = len(answer_ids_t)
    n_prefix_t = 1 + n_s_t + n_d_t

    all_ids_t = [bos_id] + surr_ids + doc_ids_t + query_ids_t + answer_ids_t
    L_t = len(all_ids_t)
    input_full = torch.tensor([all_ids_t], dtype=torch.long, device=DEVICE)

    print(f"\nTest sequence: n_s={n_s_t}, n_d={n_d_t}, n_q={n_q_t}, n_a={n_a_t}, "
          f"L={L_t}, n_prefix={n_prefix_t}")

    # --- Test A: Custom causal mask == default forward ---
    print("\n--- Test A: Custom causal mask matches default forward ---")
    simple_text = "The quick brown fox jumps over the lazy dog."
    simple_ids = tokenizer(simple_text, return_tensors="pt",
                           add_special_tokens=True).input_ids.to(DEVICE)
    Lt = simple_ids.shape[1]

    with torch.no_grad():
        out_default = model(input_ids=simple_ids)

    causal_m = make_phase_a_mask(0, Lt - 1, "causal")
    mask_d = make_mask_dict(causal_m.to(DEVICE))
    pos_t = torch.arange(Lt, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_custom = model(input_ids=simple_ids, attention_mask=mask_d,
                           position_ids=pos_t)

    max_diff_a = (out_default.logits - out_custom.logits).abs().max().item()
    mean_diff_a = (out_default.logits - out_custom.logits).abs().mean().item()
    print(f"  Max logit diff:  {max_diff_a:.6f}")
    print(f"  Mean logit diff: {mean_diff_a:.6f}")
    assert max_diff_a < 0.1, (
        f"FAIL: Custom causal mask doesn't match default (max_diff={max_diff_a:.4f})")
    print(f"  PASS: Dict-based mask API works correctly.")

    del out_default, out_custom
    gc.collect(); torch.cuda.empty_cache()

    # --- Test B: Prefix independence (prefix logits unaffected by continuation) ---
    print("\n--- Test B: Prefix independence ---")
    print("  If prefix logits are identical with/without continuation,")
    print("  then prefix KVs are independent -> two-pass is valid.")

    # Forward 1: prefix only [BOS, surr, doc] with bidirectional mask
    prefix_input = torch.tensor([[bos_id] + surr_ids + doc_ids_t],
                                 dtype=torch.long, device=DEVICE)
    prefix_mask = make_phase_a_mask(n_s_t, n_d_t, "prefix_lm")
    prefix_dict = make_mask_dict(prefix_mask.to(DEVICE))
    prefix_pos = torch.arange(n_prefix_t, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_prefix_only = model(input_ids=prefix_input, attention_mask=prefix_dict,
                                position_ids=prefix_pos)
    logits_prefix_only = out_prefix_only.logits[0]

    # Forward 2: full sequence with single-pass prefix_lm mask
    full_mask = make_single_pass_mask(n_s_t, n_d_t, n_q_t, n_a_t,
                                      "prefix_lm", truncate=True)
    full_dict = make_mask_dict(full_mask.to(DEVICE))
    full_pos = torch.arange(L_t, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_full = model(input_ids=input_full, attention_mask=full_dict,
                         position_ids=full_pos)
    logits_full_at_prefix = out_full.logits[0, :n_prefix_t]

    max_diff_b = (logits_prefix_only - logits_full_at_prefix).abs().max().item()
    print(f"  Max logit diff at prefix positions: {max_diff_b:.6f}")
    assert max_diff_b < 0.01, (
        f"FAIL: Prefix reps depend on continuation (max_diff={max_diff_b:.6f})")
    print(f"  PASS: Prefix representations are independent of continuation.")

    # Save single-pass answer logits for Test C
    answer_start_t = L_t - n_a_t
    logits_single_answer = out_full.logits[0, answer_start_t - 1 : L_t - 1].clone()

    del out_prefix_only, out_full
    gc.collect(); torch.cuda.empty_cache()

    # --- Test C: Two-pass KV cache == single-pass ---
    print("\n--- Test C: Two-pass KV cache == single-pass ---")
    print("  Phase A: [BOS, surr, doc] with bidirectional mask -> KV cache")
    print("  Phase B: [query, answer] with cached KVs + truncation mask")

    # Phase A
    with torch.no_grad():
        out_a = model(input_ids=prefix_input, attention_mask=prefix_dict,
                      position_ids=prefix_pos, use_cache=True)
    past_kv = out_a.past_key_values
    print(f"  Phase A: cached {past_kv.get_seq_length()} positions")

    # Phase B
    cont_input = torch.tensor([query_ids_t + answer_ids_t],
                               dtype=torch.long, device=DEVICE)
    n_cont = n_q_t + n_a_t
    cont_pos = torch.arange(n_prefix_t, n_prefix_t + n_cont, device=DEVICE).unsqueeze(0)

    phase_b_mask = make_phase_b_mask(n_s_t, n_d_t, n_q_t, n_a_t, truncate=True)
    phase_b_dict = make_mask_dict(phase_b_mask.to(DEVICE))

    with torch.no_grad():
        out_b = model(input_ids=cont_input, attention_mask=phase_b_dict,
                      position_ids=cont_pos, past_key_values=past_kv)

    # Answer logits from Phase B
    logits_twophase_answer = out_b.logits[0, n_q_t - 1 : n_q_t + n_a_t - 1]

    max_diff_c = (logits_single_answer - logits_twophase_answer).abs().max().item()
    mean_diff_c = (logits_single_answer - logits_twophase_answer).abs().mean().item()
    print(f"  Max answer logit diff (single vs two-pass): {max_diff_c:.6f}")
    print(f"  Mean answer logit diff:                     {mean_diff_c:.6f}")
    assert max_diff_c < 0.01, (
        f"FAIL: Two-pass doesn't match single-pass (max_diff={max_diff_c:.6f})")
    print(f"  PASS: Two-pass KV cache produces identical answer logits.")

    del out_a, out_b, past_kv, logits_single_answer, logits_twophase_answer
    gc.collect(); torch.cuda.empty_cache()

    # --- Test D: Truncation has measurable effect ---
    print("\n--- Test D: Truncation has measurable effect ---")

    # Run two-pass with truncation
    with torch.no_grad():
        out_a_t = model(input_ids=prefix_input, attention_mask=prefix_dict,
                        position_ids=prefix_pos, use_cache=True)
    past_kv_t = out_a_t.past_key_values

    mask_trunc = make_phase_b_mask(n_s_t, n_d_t, n_q_t, n_a_t, truncate=True)
    dict_trunc = make_mask_dict(mask_trunc.to(DEVICE))

    with torch.no_grad():
        out_b_t = model(input_ids=cont_input, attention_mask=dict_trunc,
                        position_ids=cont_pos, past_key_values=past_kv_t)
    logits_t = out_b_t.logits.clone()

    del out_a_t, out_b_t, past_kv_t
    gc.collect(); torch.cuda.empty_cache()

    # Run two-pass without truncation
    with torch.no_grad():
        out_a_nt = model(input_ids=prefix_input, attention_mask=prefix_dict,
                         position_ids=prefix_pos, use_cache=True)
    past_kv_nt = out_a_nt.past_key_values

    mask_notrunc = make_phase_b_mask(n_s_t, n_d_t, n_q_t, n_a_t, truncate=False)
    dict_notrunc = make_mask_dict(mask_notrunc.to(DEVICE))

    with torch.no_grad():
        out_b_nt = model(input_ids=cont_input, attention_mask=dict_notrunc,
                         position_ids=cont_pos, past_key_values=past_kv_nt)
    logits_nt = out_b_nt.logits.clone()

    del out_a_nt, out_b_nt, past_kv_nt
    gc.collect(); torch.cuda.empty_cache()

    cont_diff = (logits_t - logits_nt).abs().max().item()
    print(f"  Continuation logit diff (trunc vs no-trunc): {cont_diff:.6f}")
    assert cont_diff > 0.01, (
        f"FAIL: Truncation had no effect on continuation (diff={cont_diff})")
    print(f"  PASS: Truncation correctly affects Phase B outputs.")

    del logits_t, logits_nt

    # --- Test E: prefix_lm vs causal produces different representations ---
    print("\n--- Test E: Prefix LM != causal (bidirectionality matters) ---")

    # Phase A with prefix_lm
    with torch.no_grad():
        out_plm = model(input_ids=prefix_input, attention_mask=prefix_dict,
                        position_ids=prefix_pos, use_cache=True)
    past_kv_plm = out_plm.past_key_values

    mask_b = make_phase_b_mask(n_s_t, n_d_t, n_q_t, n_a_t, truncate=True)
    dict_b = make_mask_dict(mask_b.to(DEVICE))

    with torch.no_grad():
        out_b_plm = model(input_ids=cont_input, attention_mask=dict_b,
                          position_ids=cont_pos, past_key_values=past_kv_plm)
    logits_plm = out_b_plm.logits.clone()

    del out_plm, out_b_plm, past_kv_plm
    gc.collect(); torch.cuda.empty_cache()

    # Phase A with causal
    causal_a = make_phase_a_mask(n_s_t, n_d_t, "causal")
    causal_a_dict = make_mask_dict(causal_a.to(DEVICE))

    with torch.no_grad():
        out_csl = model(input_ids=prefix_input, attention_mask=causal_a_dict,
                        position_ids=prefix_pos, use_cache=True)
    past_kv_csl = out_csl.past_key_values

    with torch.no_grad():
        out_b_csl = model(input_ids=cont_input, attention_mask=dict_b,
                          position_ids=cont_pos, past_key_values=past_kv_csl)
    logits_csl = out_b_csl.logits.clone()

    del out_csl, out_b_csl, past_kv_csl
    gc.collect(); torch.cuda.empty_cache()

    mode_diff = (logits_plm - logits_csl).abs().max().item()
    print(f"  Continuation logit diff (plm vs causal): {mode_diff:.6f}")
    assert mode_diff > 0.01, (
        f"FAIL: prefix_lm and causal produce identical outputs")
    print(f"  PASS: Bidirectional attention produces different representations than causal.")

    del logits_plm, logits_csl
    gc.collect(); torch.cuda.empty_cache()

    # --- Test F: _nq condition NLL computation ---
    print("\n--- Test F: _nq condition (no query) NLL consistency ---")
    print("  Verify Phase A last logit + Phase B logits match single-pass answer logits.")

    # Single-pass: [BOS, surr, doc, answer] with prefix_lm mask
    nq_ids = [bos_id] + surr_ids + doc_ids_t + answer_ids_t
    nq_input = torch.tensor([nq_ids], dtype=torch.long, device=DEVICE)
    L_nq = len(nq_ids)
    nq_mask = make_single_pass_mask(n_s_t, n_d_t, 0, n_a_t, "prefix_lm", True)
    nq_dict = make_mask_dict(nq_mask.to(DEVICE))
    nq_pos = torch.arange(L_nq, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_nq_sp = model(input_ids=nq_input, attention_mask=nq_dict,
                          position_ids=nq_pos)
    # Answer logits from single-pass: last position of prefix predicts a0,
    # then a0 position predicts a1, etc.
    sp_answer_logits = out_nq_sp.logits[0, n_prefix_t - 1 : n_prefix_t + n_a_t - 1]

    del out_nq_sp

    # Two-pass: Phase A -> Phase B with [answer] only
    with torch.no_grad():
        out_a_nq = model(input_ids=prefix_input, attention_mask=prefix_dict,
                         position_ids=prefix_pos, use_cache=True)
    past_kv_nq = out_a_nq.past_key_values
    logit_first = out_a_nq.logits[0, -1:, :]

    ans_input = torch.tensor([answer_ids_t], dtype=torch.long, device=DEVICE)
    ans_mask = make_phase_b_mask(n_s_t, n_d_t, 0, n_a_t, truncate=True)
    ans_dict = make_mask_dict(ans_mask.to(DEVICE))
    ans_pos = torch.arange(n_prefix_t, n_prefix_t + n_a_t, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_b_nq = model(input_ids=ans_input, attention_mask=ans_dict,
                         position_ids=ans_pos, past_key_values=past_kv_nq)

    if n_a_t > 1:
        logit_rest = out_b_nq.logits[0, :n_a_t - 1, :]
        tp_answer_logits = torch.cat([logit_first, logit_rest], dim=0)
    else:
        tp_answer_logits = logit_first

    max_diff_f = (sp_answer_logits - tp_answer_logits).abs().max().item()
    print(f"  Max answer logit diff (single-pass vs two-pass _nq): {max_diff_f:.6f}")
    assert max_diff_f < 0.01, (
        f"FAIL: _nq two-pass doesn't match single-pass (max_diff={max_diff_f:.6f})")
    print(f"  PASS: _nq two-pass NLL matches single-pass.")

    del out_a_nq, out_b_nq, past_kv_nq
    gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("ALL MODEL TESTS PASSED")
    print("=" * 70)


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test prefix LM attention masks")
    parser.add_argument("--model", action="store_true",
                        help="Run model-based tests (loads Gemma 3 12B)")
    args = parser.parse_args()

    run_math_tests()
    show_visual()

    if args.model:
        run_model_tests()
    else:
        print("\n  (Skipping model tests. Use --model to run them.)")

    print("\nDone.")
