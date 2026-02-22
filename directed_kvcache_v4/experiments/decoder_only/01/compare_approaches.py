#!/usr/bin/env python3
"""Quick comparison of 3 scoring approaches on Gemma 3 12B.

Tests bare vs oracle on 20 samples each to see which approach
gives sensible results (oracle should help, not hurt).

Approach 1: Single-pass with pretrained model (gemma-3-12b-pt)
Approach 2: Single-pass with IT model + chat template
Approach 3: Two-phase KV cache with IT model + float32 logits
"""

import os, sys, json, gc
os.umask(0o000)

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d
from lib.data import count_words

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---- Load data (same as main experiment) ----
SEED = 42
N_TEST = 20
np.random.seed(SEED)

print("Loading MS MARCO...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
all_candidates = []
for item in ds:
    if len(all_candidates) >= 1200:
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
samples = [all_candidates[i] for i in indices[:400]]
del ds, all_candidates
gc.collect()
print(f"Using first {N_TEST} of {len(samples)} samples\n")


def run_test(approach_name, score_fn, samples, n=N_TEST):
    """Run bare vs oracle on n samples, print summary."""
    print(f"\n{'=' * 70}")
    print(f"APPROACH: {approach_name}")
    print(f"{'=' * 70}")

    diffs = []
    for i in range(n):
        s = samples[i]
        nll_bare = score_fn(s['passage'], s['query'], s['answer'], prefix_text=None)
        nll_oracle = score_fn(s['passage'], s['query'], s['answer'], prefix_text=s['query'])
        diff = nll_bare - nll_oracle  # positive = oracle better
        diffs.append(diff)
        direction = "oracle better" if diff > 0 else "bare better"
        print(f"  {i:>2}: bare={nll_bare:.4f} oracle={nll_oracle:.4f} "
              f"delta={diff:+.4f} ({direction})")
        gc.collect()
        torch.cuda.empty_cache()

    diffs = np.array(diffs)
    d = cohens_d(diffs)
    _, p = stats.ttest_1samp(diffs, 0)
    win = 100 * np.mean(diffs > 0)

    print(f"\n  Summary (N={n}):")
    print(f"    Mean delta (bare - oracle): {diffs.mean():+.4f}")
    print(f"    Cohen's d: {d:+.3f}")
    print(f"    Oracle win%: {win:.0f}%")
    print(f"    p-value: {p:.4f}")
    if d > 0.05:
        print(f"    --> ORACLE HELPS (d={d:+.3f})")
    elif d < -0.05:
        print(f"    --> ORACLE HURTS (d={d:+.3f})")
    else:
        print(f"    --> NO EFFECT (d={d:+.3f})")
    return diffs


# ============================================================
# APPROACH 1: Pretrained model, single-pass
# ============================================================
print("\nLoading google/gemma-3-12b-pt...")
tokenizer1 = AutoTokenizer.from_pretrained("google/gemma-3-12b-pt", token=HF_TOKEN)
model1 = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-pt", device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model1.eval()
DEVICE = next(model1.parameters()).device
print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def score_pt_single_pass(doc_text, query_text, answer_text, prefix_text=None):
    if prefix_text:
        prefix_ids = tokenizer1(prefix_text, add_special_tokens=False,
                                truncation=True, max_length=512).input_ids
        sep_ids = tokenizer1("\n", add_special_tokens=False).input_ids
        doc_ids = tokenizer1(doc_text, add_special_tokens=False,
                             truncation=True, max_length=1536).input_ids
        pre = [tokenizer1.bos_token_id] + prefix_ids + sep_ids + doc_ids
    else:
        doc_ids = tokenizer1(doc_text, add_special_tokens=False,
                             truncation=True, max_length=2048).input_ids
        pre = [tokenizer1.bos_token_id] + doc_ids

    query_ids = tokenizer1("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer1(answer_text, add_special_tokens=False,
                            truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    all_ids = pre + query_ids + answer_ids
    with torch.no_grad():
        out = model1(input_ids=torch.tensor([all_ids], device=DEVICE), use_cache=False)

    n_before = len(pre) + len(query_ids)
    logits = out.logits[0, n_before - 1:n_before - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out
    return nll


diffs1 = run_test("1. Pretrained (gemma-3-12b-pt) — single-pass", score_pt_single_pass, samples)

# Free model 1
del model1, tokenizer1
gc.collect()
torch.cuda.empty_cache()
gc.collect()


# ============================================================
# APPROACH 2: IT model + chat template, single-pass
# ============================================================
print("\nLoading google/gemma-3-12b-it...")
tokenizer2 = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", token=HF_TOKEN)
model2 = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it", device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model2.eval()
DEVICE = next(model2.parameters()).device
print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def score_it_chat_template(doc_text, query_text, answer_text, prefix_text=None):
    # Wrap in proper chat template:
    # User: [prefix\n]doc\nquery
    # Model: answer
    if prefix_text:
        user_content = f"{prefix_text}\n\n{doc_text}\n\n{query_text}"
    else:
        user_content = f"{doc_text}\n\n{query_text}"

    messages = [{"role": "user", "content": user_content}]
    chat_text = tokenizer2.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Tokenize the chat prefix (everything before the answer)
    chat_ids = tokenizer2(chat_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer2(answer_text, add_special_tokens=False,
                            truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    all_ids = chat_ids + answer_ids
    with torch.no_grad():
        out = model2(input_ids=torch.tensor([all_ids], device=DEVICE), use_cache=False)

    n_chat = len(chat_ids)
    logits = out.logits[0, n_chat - 1:n_chat - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out
    return nll


diffs2 = run_test("2. IT (gemma-3-12b-it) — chat template", score_it_chat_template, samples)


# ============================================================
# APPROACH 3: IT model + two-phase KV cache + float32 logits
# ============================================================
# Reuse model2/tokenizer2

NEWLINE_IDS = tokenizer2("\n", add_special_tokens=False).input_ids
BOS_ID = tokenizer2.bos_token_id


def slice_kv_cache(cache, start_idx):
    from transformers import DynamicCache
    if isinstance(cache, DynamicCache):
        sliced = DynamicCache()
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys[:, :, start_idx:, :]
            v = cache.layers[i].values[:, :, start_idx:, :]
            sliced.update(k, v, i)
        return sliced
    else:
        return tuple(
            (k[:, :, start_idx:, :], v[:, :, start_idx:, :])
            for k, v in cache
        )


def score_it_two_phase_f32(doc_text, query_text, answer_text, prefix_text=None):
    doc_ids = tokenizer2(doc_text, add_special_tokens=False,
                         truncation=True, max_length=1536).input_ids

    if prefix_text:
        prefix_ids = tokenizer2(prefix_text, add_special_tokens=False,
                                truncation=True, max_length=512).input_ids
        cond_ids = [BOS_ID] + prefix_ids + NEWLINE_IDS + doc_ids
        slice_start = 1 + len(prefix_ids) + len(NEWLINE_IDS)
    else:
        cond_ids = [BOS_ID] + doc_ids
        slice_start = 0

    phase_b_start = len(cond_ids)

    with torch.no_grad():
        pa = model2(input_ids=torch.tensor([cond_ids], device=DEVICE), use_cache=True)
    cache = pa.past_key_values
    del pa

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

    query_ids = tokenizer2("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer2(answer_text, add_special_tokens=False,
                            truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    with torch.no_grad():
        pb = model2(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=False,
        )

    n_q = len(query_ids)
    # Cast to float32 before softmax
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del cache, pb
    return nll


diffs3 = run_test("3. IT (gemma-3-12b-it) — two-phase KV cache + f32", score_it_two_phase_f32, samples)

# Free model 2
del model2, tokenizer2
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Oracle benefit (bare - oracle) across approaches")
print("=" * 70)

for name, diffs in [
    ("1. PT single-pass", diffs1),
    ("2. IT chat-template", diffs2),
    ("3. IT two-phase+f32", diffs3),
]:
    d = cohens_d(diffs)
    _, p = stats.ttest_1samp(diffs, 0)
    win = 100 * np.mean(diffs > 0)
    verdict = "HELPS" if d > 0.05 else "HURTS" if d < -0.05 else "NEUTRAL"
    print(f"  {name:<30} d={d:+.3f}  win={win:>4.0f}%  p={p:.4f}  --> {verdict}")

print("\nDone!")
