#!/usr/bin/env python3
"""Diagnose the two-phase vs single-pass validation mismatch on Gemma 3 12B."""

import os, sys, json, gc
os.umask(0o000)

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "../../..")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-3-12b-it"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()
DEVICE = next(model.parameters()).device
print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Model config: {model.config.architectures}")
print(f"Attn implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
text_cfg = getattr(model.config, 'text_config', model.config)
print(f"Layers: {text_cfg.num_hidden_layers}, KV heads: {text_cfg.num_key_value_heads}")

# Load the same samples used in the experiment
from lib.data import count_words
from datasets import load_dataset

SEED = 42
N_SAMPLES = 400
np.random.seed(SEED)

ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
all_candidates = []
for item in ds:
    if len(all_candidates) >= 3 * N_SAMPLES:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    if not answer:
        continue
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            all_candidates.append({'passage': pt, 'query': query, 'answer': answer})
            break

indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
BOS_ID = tokenizer.bos_token_id

# ============================================================
# Test 1: Two-phase (no prefix) vs single-pass — BF16
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Two-phase (bare, no prefix) vs single-pass — BF16")
print("=" * 70)

N_TEST = 20

for i in range(N_TEST):
    s = samples[i]
    doc_ids = tokenizer(s['passage'], add_special_tokens=True,
                        truncation=True, max_length=2048).input_ids
    query_part_ids = tokenizer("\n" + s['query'] + "\n",
                               add_special_tokens=False).input_ids
    answer_ids = tokenizer(s['answer'], add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        continue

    # --- Single pass ---
    all_ids = doc_ids + query_part_ids + answer_ids
    input_tensor = torch.tensor([all_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out_full = model(input_ids=input_tensor, use_cache=False)

    n_doc = len(doc_ids)
    n_query = len(query_part_ids)
    n_answer = len(answer_ids)
    start = n_doc + n_query - 1
    logits_full = out_full.logits[0, start:start + n_answer, :]
    targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)
    lp_full = F.log_softmax(logits_full, dim=-1)
    nll_full = -lp_full.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()

    del out_full

    # --- Two-phase (bare) ---
    cond_tensor = torch.tensor([doc_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        phase_a = model(input_ids=cond_tensor, use_cache=True)
    cache = phase_a.past_key_values
    del phase_a

    phase_b_ids = query_part_ids + answer_ids
    phase_b_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=DEVICE)
    total_cond_len = len(doc_ids)
    pos_ids = torch.arange(total_cond_len, total_cond_len + len(phase_b_ids),
                           device=DEVICE).unsqueeze(0)
    cache_position = torch.arange(total_cond_len, total_cond_len + len(phase_b_ids),
                                  device=DEVICE)

    with torch.no_grad():
        phase_b = model(
            input_ids=phase_b_tensor,
            past_key_values=cache,
            position_ids=pos_ids,
            cache_position=cache_position,
            use_cache=False,
        )

    logits_cached = phase_b.logits[0, n_query - 1:n_query - 1 + n_answer, :]
    lp_cached = F.log_softmax(logits_cached, dim=-1)
    nll_cached = -lp_cached.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()

    diff = abs(nll_cached - nll_full)

    # Also compare raw logits for the first answer token
    logit_diff_max = (logits_full[0] - logits_cached[0]).abs().max().item()
    logit_diff_mean = (logits_full[0] - logits_cached[0]).abs().mean().item()

    del cache, phase_b
    gc.collect()
    torch.cuda.empty_cache()

    status = "OK" if diff < 0.001 else "MISMATCH"
    print(f"  Sample {i:>2}: cached={nll_cached:.6f} full={nll_full:.6f} "
          f"nll_diff={diff:.8f} logit_max_diff={logit_diff_max:.6f} "
          f"logit_mean_diff={logit_diff_mean:.6f} [{status}]")


# ============================================================
# Test 2: Check if the issue is systematic (bias direction)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Bias direction analysis")
print("=" * 70)

diffs = []
for i in range(min(50, N_SAMPLES)):
    s = samples[i]
    doc_ids = tokenizer(s['passage'], add_special_tokens=True,
                        truncation=True, max_length=2048).input_ids
    query_part_ids = tokenizer("\n" + s['query'] + "\n",
                               add_special_tokens=False).input_ids
    answer_ids = tokenizer(s['answer'], add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        continue

    # Single pass
    all_ids = doc_ids + query_part_ids + answer_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([all_ids], device=DEVICE), use_cache=False)
    start = len(doc_ids) + len(query_part_ids) - 1
    logits = out.logits[0, start:start + len(answer_ids), :]
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll_full = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out

    # Two-phase
    with torch.no_grad():
        pa = model(input_ids=torch.tensor([doc_ids], device=DEVICE), use_cache=True)
    cache = pa.past_key_values
    del pa

    pb_ids = query_part_ids + answer_ids
    pos = torch.arange(len(doc_ids), len(doc_ids) + len(pb_ids), device=DEVICE)
    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=False,
        )
    logits2 = pb.logits[0, len(query_part_ids)-1:len(query_part_ids)-1+len(answer_ids), :]
    nll_cached = -F.log_softmax(logits2, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del cache, pb

    diffs.append(nll_cached - nll_full)  # positive = cached is higher NLL
    gc.collect()
    torch.cuda.empty_cache()

diffs = np.array(diffs)
print(f"  N = {len(diffs)}")
print(f"  Mean diff (cached - full): {diffs.mean():+.6f}")
print(f"  Std diff:                  {diffs.std():.6f}")
print(f"  Max abs diff:              {np.abs(diffs).max():.6f}")
print(f"  Median diff:               {np.median(diffs):+.6f}")
print(f"  % cached > full:           {100*np.mean(diffs > 0):.1f}%")
print(f"  % cached < full:           {100*np.mean(diffs < 0):.1f}%")
print(f"  % exactly equal:           {100*np.mean(diffs == 0):.1f}%")

from scipy import stats
t, p = stats.ttest_1samp(diffs, 0)
print(f"  t-test vs 0: t={t:.3f}, p={p:.4f}")

if p < 0.05:
    print(f"  --> SYSTEMATIC BIAS detected (mean={diffs.mean():+.6f})")
else:
    print(f"  --> No systematic bias (random rounding noise)")

# ============================================================
# Test 3: Same test with float32 (is it a bfloat16 issue?)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Repeat with float32 computation")
print("=" * 70)
print("Testing if mismatch disappears with higher precision...")

# We can't easily reload in fp32 (too much VRAM), but we can cast
# the logits to fp32 before computing NLL
for i in range(5):
    s = samples[i]
    doc_ids = tokenizer(s['passage'], add_special_tokens=True,
                        truncation=True, max_length=2048).input_ids
    query_part_ids = tokenizer("\n" + s['query'] + "\n",
                               add_special_tokens=False).input_ids
    answer_ids = tokenizer(s['answer'], add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        continue

    # Single pass — fp32 NLL computation
    all_ids = doc_ids + query_part_ids + answer_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([all_ids], device=DEVICE), use_cache=False)
    start = len(doc_ids) + len(query_part_ids) - 1
    logits = out.logits[0, start:start + len(answer_ids), :].float()  # cast to fp32
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll_full_f32 = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out

    # Two-phase — fp32 NLL computation
    with torch.no_grad():
        pa = model(input_ids=torch.tensor([doc_ids], device=DEVICE), use_cache=True)
    cache = pa.past_key_values
    del pa

    pb_ids = query_part_ids + answer_ids
    pos = torch.arange(len(doc_ids), len(doc_ids) + len(pb_ids), device=DEVICE)
    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=False,
        )
    logits2 = pb.logits[0, len(query_part_ids)-1:len(query_part_ids)-1+len(answer_ids), :].float()
    nll_cached_f32 = -F.log_softmax(logits2, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del cache, pb

    diff_f32 = abs(nll_cached_f32 - nll_full_f32)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Sample {i}: cached_f32={nll_cached_f32:.8f} full_f32={nll_full_f32:.8f} "
          f"diff={diff_f32:.10f}")

print("\nDone!")
