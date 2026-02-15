#!/usr/bin/env python3
"""Build script for 15_nll_ensemble_ranking.ipynb

Exp 15: NLL Ensemble Ranking — Can Diverse Priming Improve Ranking?

Exp 14 found combined (bare + primed NLL) improves ranking, but cross-
validated ΔMRR was only +0.006 (ns). Per-query oracle alpha (+0.050)
suggests latent signal exists. This experiment tests whether DIVERSE
priming caches can produce a significant ranking ensemble.

5 Scoring Signals:
  1. bare         — standard bare cache NLL
  2. rescore      — bare cache, alt prompt template (non-priming control)
  3. primed_sf    — static_fact prefix, truncated
  4. primed_rand  — random text prefix, truncated
  5. primed_intent — intent prefix, truncated

Ensemble Conditions (equal-weight average of NLLs):
  - ens_2_sf:      bare + sf
  - ens_2_rand:    bare + rand
  - ens_2_rescore: bare + rescore (NON-PRIMING CONTROL)
  - ens_3:         bare + sf + rand
  - ens_4:         bare + sf + rand + intent
  - ens_5_all:     all 5 signals

Key questions:
  1. Does ensembling significantly improve ranking? (Wilcoxon, no tuning)
  2. Does improvement scale with ensemble size?
  3. Is priming special? (compare ens_2_sf vs ens_2_rescore)
  4. What's the oracle gap? (CV-optimized weights vs equal weights)

Compute: ~300 queries × ~8 passages × (4 forward + 5 scoring) ≈ 2-3h on L4
"""

import json


def make_cell(cell_type, source, cell_id=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split('\n')
    }
    if cell_id:
        cell["id"] = cell_id
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def s(text):
    """Convert multi-line string to notebook source lines."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    if result and result[-1] == '':
        result = result[:-1]
    return result


cells = []

# ========== Cell 0: Overview ==========
cells.append(make_cell("markdown", s("""\
# Exp 15: NLL Ensemble Ranking

## Motivation

Exp 14 found that combining bare + primed NLL marginally improves ranking
(global ΔMRR=+0.008), but cross-validated improvement was only +0.006 (p=0.21, ns).
The per-query oracle alpha gap (+0.050) suggests latent signal exists but a single
primed cache can't reliably extract it.

**Core hypothesis:** Diverse priming caches produce NLL estimates with partially
independent errors. Ensembling (averaging) these estimates reduces ranking noise,
just as averaging multiple measurements improves precision.

## Design

**5 Scoring Signals** (each produces a per-passage NLL):

| # | Signal | Cache | Scoring Prompt | Purpose |
|---|--------|-------|---------------|---------|
| 1 | `bare` | Bare cache | Standard prompt | Baseline |
| 2 | `rescore` | Bare cache | Alt prompt | **Control**: diversity without priming |
| 3 | `sf` | Static fact prefix | Standard prompt | Replicate Exp 14 |
| 4 | `rand` | Random text prefix | Standard prompt | Is prefix content irrelevant? |
| 5 | `intent` | Intent prefix | Standard prompt | Different semantic angle |

**Ensemble Conditions** (equal-weight NLL average, no tuning):

| Ensemble | Members | Tests |
|----------|---------|-------|
| `ens_2_sf` | bare + sf | Replicates Exp 14 |
| `ens_2_rand` | bare + rand | Random prefix ensemble |
| `ens_2_rescore` | bare + rescore | **Non-priming control** |
| `ens_3` | bare + sf + rand | 3-member ensemble |
| `ens_4` | bare + sf + rand + intent | 4-member ensemble |
| `ens_5_all` | all 5 signals | Maximum diversity |

**Critical comparison:** `ens_2_sf` vs `ens_2_rescore`. If the control matches
priming, then priming isn't special — any scoring diversity helps.""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup
import os
os.umask(0o000)

import sys
import json
import time
import numpy as np
import torch
import gc
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp15")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"

print(f"SEED: {SEED}")
print(f"Results directory: {RESULTS_DIR}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\
""")))

# ========== Cell 2: Load model ==========
cells.append(make_cell("code", s("""\
# Cell 2: Load model (Mistral-7B 4-bit)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading {MODEL_NAME} (4-bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print(f"Model loaded. dtype={model.dtype}, device={model.device}")\
""")))

# ========== Cell 3: Config + lib imports ==========
cells.append(make_cell("code", s("""\
# Cell 3: Config and library imports
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    _get_cache_keys,
    _get_cache_values,
    _set_cache_keys,
    _set_cache_values,
    _ensure_dynamic_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    score_answer_with_cache,
    deepcopy_cache,
    build_hybrid_cache,
)
from lib.analysis import compute_ranking_metrics
from lib.surrogate import STATIC_SURROGATE_QUERIES
from lib.data import count_words
from scipy import stats
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    seed=SEED,
)

# Templates
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Alternative prompt for rescore control
ALT_QUERY_TEMPLATE = "\\nQuestion: {query}\\nResponse:"

# Prefix texts
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']
RANDOM_PREFIX_TEXT = "The purple elephant danced gracefully on the frozen lake during twilight"
INTENT_PREFIX_TEXT = "What is this passage about?"

# Experiment parameters
MAX_QUERIES = 300
MAX_PASSAGE_WORDS = 300
MIN_PASSAGES_PER_QUERY = 2
CHECKPOINT_EVERY = 25

SIGNAL_NAMES = ['bare', 'rescore', 'sf', 'rand', 'intent']

print("Config ready")
print(f"  MAX_QUERIES: {MAX_QUERIES}")
print(f"  Prefixes:")
print(f"    sf:     '{STATIC_FACT}'")
print(f"    rand:   '{RANDOM_PREFIX_TEXT}'")
print(f"    intent: '{INTENT_PREFIX_TEXT}'")
print(f"  Alt prompt: '{ALT_QUERY_TEMPLATE.format(query='...')}'")
print(f"  Signals: {SIGNAL_NAMES}")\
""")))

# ========== Cell 4: Load dataset ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO v1.1 (same filtering as Exp 14)
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 — ALL PASSAGES PER QUERY")
print("=" * 70)

dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation",
                        trust_remote_code=True)
print(f"Total items in validation: {len(dataset)}")

queries = []
np.random.seed(SEED)

for item in tqdm(dataset, desc="Filtering"):
    passages_info = item.get('passages', {})
    passage_texts = passages_info.get('passage_text', [])
    is_selected = passages_info.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])

    if not passage_texts or not query:
        continue
    if len(passage_texts) < MIN_PASSAGES_PER_QUERY:
        continue
    if not is_selected or sum(is_selected) == 0:
        continue

    word_counts = [count_words(p) for p in passage_texts]
    if any(wc > MAX_PASSAGE_WORDS for wc in word_counts):
        continue

    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    else:
        continue

    passage_list = []
    for i, (ptext, sel) in enumerate(zip(passage_texts, is_selected)):
        passage_list.append({
            'passage': ptext,
            'is_relevant': bool(sel == 1),
            'word_count': word_counts[i],
            'passage_idx': i,
        })

    queries.append({
        'query': query,
        'answer': answer,
        'passages': passage_list,
        'n_passages': len(passage_list),
        'n_relevant': sum(1 for p in passage_list if p['is_relevant']),
    })

    if len(queries) >= MAX_QUERIES * 3:
        break

np.random.shuffle(queries)
queries = queries[:MAX_QUERIES]
N = len(queries)

n_passages_list = [q['n_passages'] for q in queries]
total_passages = sum(n_passages_list)

print(f"\\nSelected {N} queries ({total_passages} total passages)")
print(f"Passages per query: mean={np.mean(n_passages_list):.1f}, "
      f"min={min(n_passages_list)}, max={max(n_passages_list)}")
print(f"Word counts: mean={np.mean([p['word_count'] for q in queries for p in q['passages']]):.0f}")

del dataset
gc.collect()\
""")))

# ========== Cell 5: Tokenize prefixes + explain conditions ==========
cells.append(make_cell("code", s("""\
# Cell 5: Tokenize prefixes and verify BPE boundaries

print("=" * 70)
print("EXPERIMENTAL CONDITIONS — NLL ENSEMBLE RANKING")
print("=" * 70)

# Tokenize each prefix
sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
rand_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=RANDOM_PREFIX_TEXT)
intent_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=INTENT_PREFIX_TEXT)

sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(config.device)
rand_ids = tokenizer(rand_str, return_tensors="pt",
                      add_special_tokens=False)['input_ids'].to(config.device)
intent_ids = tokenizer(intent_str, return_tensors="pt",
                        add_special_tokens=False)['input_ids'].to(config.device)

PREFIX_CONFIGS = [
    ('sf', STATIC_FACT, sf_str, sf_ids),
    ('rand', RANDOM_PREFIX_TEXT, rand_str, rand_ids),
    ('intent', INTENT_PREFIX_TEXT, intent_str, intent_ids),
]

print("\\nPREFIX TOKEN LENGTHS:")
for name, text, full_str, ids in PREFIX_CONFIGS:
    print(f"  {name:<8} {ids.shape[1]:>3} tokens | '{text}'")

# Verify BPE boundary consistency across prefixes
print("\\nBPE BOUNDARY CHECK (first passage):")
example_doc = queries[0]['passages'][0]['passage']
for name, text, full_str, ids in PREFIX_CONFIGS:
    concat = full_str + DOCUMENT_TEMPLATE.format(document=example_doc)
    concat_enc = tokenizer(concat, add_special_tokens=True)['input_ids']
    prefix_enc = tokenizer(full_str, add_special_tokens=True)['input_ids']
    doc_ids_from_concat = concat_enc[len(prefix_enc):]

    bare_doc_enc = tokenizer(DOCUMENT_TEMPLATE.format(document=example_doc),
                              add_special_tokens=False)['input_ids']
    match = sum(1 for a, b in zip(doc_ids_from_concat, bare_doc_enc) if a == b)
    total = max(len(bare_doc_enc), 1)
    print(f"  {name}: {match}/{total} tokens match ({100*match/total:.1f}%)")

# Explain conditions
print("\\n" + "=" * 70)
print("CONDITION DETAILS")
print("=" * 70)

conditions_detail = [
    ("bare", "Standard bare cache scored with standard prompt",
     "Baseline. All other conditions are compared to this."),
    ("rescore", "Bare cache scored with alt prompt ('Question:...Response:')",
     "NON-PRIMING CONTROL. Same cache, different prompt. Tests if scoring "
     "diversity alone improves ensembles, without any cache modification."),
    ("sf (static_fact)", f"Prefix: '{STATIC_FACT}'",
     "Replicates Exp 14's primed_1x. Bare keys + primed values (truncated)."),
    ("rand (random)", f"Prefix: '{RANDOM_PREFIX_TEXT}'",
     "Semantically unrelated prefix. Tests if ANY prefix content works or "
     "if semantic relevance matters."),
    ("intent", f"Prefix: '{INTENT_PREFIX_TEXT}'",
     "Different semantic angle than static_fact. Tests prefix diversity."),
]

for name, detail, purpose in conditions_detail:
    print(f"\\n### {name} ###")
    print(f"  {detail}")
    print(f"  Purpose: {purpose}")

print("\\n" + "=" * 70)
print("ENSEMBLE CONDITIONS (equal-weight NLL average)")
print("=" * 70)
print("  ens_2_sf:      bare + sf           (replicate Exp 14)")
print("  ens_2_rand:    bare + rand         (random prefix)")
print("  ens_2_rescore: bare + rescore      (NON-PRIMING CONTROL)")
print("  ens_3:         bare + sf + rand    (3-member)")
print("  ens_4:         bare + sf + rand + intent  (4-member)")
print("  ens_5_all:     all 5 signals       (maximum diversity)")\
""")))

# ========== Cell 6: Main loop ==========
cells.append(make_cell("code", s("""\
# Cell 6: Main loop — score all passages under all conditions

print("=" * 70)
print(f"MAIN EVALUATION ({N} queries, ~{total_passages} passages)")
print("=" * 70)

# Checkpoint resume
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in queries]
    if ckpt_queries == current_queries:
        all_results = ckpt['results']
        start_idx = len(all_results)
        print(f"Resuming from checkpoint: {start_idx}/{N}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

print(f"Evaluating queries {start_idx} to {N-1}")
print(f"Per passage: 4 forward passes (bare + 3 primed) + 5 scoring passes")

t_start = time.time()

for qidx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Queries"):
    query_data = queries[qidx]
    query = query_data['query']
    answer = query_data['answer']
    query_prompt = QUERY_TEMPLATE.format(query=query)
    alt_query_prompt = ALT_QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    passage_results = []

    for pidx, pinfo in enumerate(query_data['passages']):
        passage = pinfo['passage']
        document_text = DOCUMENT_TEMPLATE.format(document=passage)

        # --- Matched tokenization (using sf prefix) ---
        full_text = sf_str + document_text
        full_enc = tokenizer(full_text, return_tensors="pt",
                              add_special_tokens=True, padding=False, truncation=False)
        full_ids = full_enc['input_ids'].to(config.device)

        sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                                   add_special_tokens=True, padding=False, truncation=False)
        sf_prefix_len_matched = sf_prefix_enc['input_ids'].shape[1]

        bos_id = full_ids[:, :1]
        doc_ids = full_ids[:, sf_prefix_len_matched:]
        doc_len = doc_ids.shape[1]
        context_len = 1 + doc_len  # BOS + doc

        del full_enc, full_ids, sf_prefix_enc

        # === 1. Build bare cache ===
        bare_input = torch.cat([bos_id, doc_ids], dim=1)
        with torch.no_grad():
            bare_out = model(input_ids=bare_input,
                             attention_mask=torch.ones_like(bare_input),
                             use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
        del bare_out, bare_input

        # === 2. Score rescore (deepcopy bare, alt prompt) ===
        bare_copy = deepcopy_cache(bare_cache)
        rescore_nll = score_answer_with_cache(
            bare_copy, context_len, alt_query_prompt, answer_text,
            model, tokenizer, config)
        del bare_copy

        # === 3-5. For each priming prefix: build, truncate, hybrid, score ===
        primed_nlls = {}
        for p_name, p_text, p_str, p_ids in PREFIX_CONFIGS:
            primed_input = torch.cat([bos_id, p_ids, doc_ids], dim=1)
            with torch.no_grad():
                primed_out = model(input_ids=primed_input,
                                   attention_mask=torch.ones_like(primed_input),
                                   use_cache=True, return_dict=True)
            primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
            del primed_out, primed_input

            # Truncate + RoPE correct
            primed_trunc = extract_and_truncate_cache_with_bos(primed_full, doc_len)
            correct_rope_positions_with_bos(primed_trunc, p_ids.shape[1], model)
            del primed_full

            # Hybrid: bare keys + primed values (pure value contamination)
            hybrid = build_hybrid_cache(bare_cache, primed_trunc)
            del primed_trunc

            primed_nlls[p_name] = score_answer_with_cache(
                hybrid, context_len, query_prompt, answer_text,
                model, tokenizer, config)
            del hybrid

        # === 6. Score bare LAST (mutates cache) ===
        bare_nll = score_answer_with_cache(
            bare_cache, context_len, query_prompt, answer_text,
            model, tokenizer, config)
        del bare_cache

        gc.collect()
        torch.cuda.empty_cache()

        passage_results.append({
            'passage_idx': pinfo['passage_idx'],
            'is_relevant': pinfo['is_relevant'],
            'word_count': pinfo['word_count'],
            'bare_nll': bare_nll,
            'rescore_nll': rescore_nll,
            'sf_nll': primed_nlls['sf'],
            'rand_nll': primed_nlls['rand'],
            'intent_nll': primed_nlls['intent'],
        })

    all_results.append({
        'query_idx': qidx,
        'query': query,
        'n_passages': len(passage_results),
        'n_relevant': query_data['n_relevant'],
        'passage_data': passage_results,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N - 1:
        ckpt_data = {
            'results': all_results,
            'query_texts': [q['query'] for q in queries],
            'completed': len(all_results),
            'total': N,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nEvaluation complete: {len(all_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 7: Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 7: Analysis — individual signals, ensembles, significance, scaling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS")
print("=" * 70)

N_VALID = len(all_results)
print(f"Valid queries: {N_VALID}")

# --- Helper functions ---
def mrr_for_signal(results, sig_name):
    \"\"\"Compute per-query MRR ranking by a single NLL signal.\"\"\"
    mrrs = []
    for r in results:
        pd = r['passage_data']
        scores = {i: pd[i][f'{sig_name}_nll'] for i in range(len(pd))}
        rel_idx = next(i for i, p in enumerate(pd) if p['is_relevant'])
        m = compute_ranking_metrics(scores, relevant_idx=rel_idx)
        mrrs.append(m['mrr'])
    return np.array(mrrs)


def mrr_for_ensemble(results, sig_names):
    \"\"\"Compute per-query MRR ranking by equal-weight NLL average.\"\"\"
    mrrs = []
    for r in results:
        pd = r['passage_data']
        scores = {}
        for i in range(len(pd)):
            scores[i] = np.mean([pd[i][f'{s}_nll'] for s in sig_names])
        rel_idx = next(i for i, p in enumerate(pd) if p['is_relevant'])
        m = compute_ranking_metrics(scores, relevant_idx=rel_idx)
        mrrs.append(m['mrr'])
    return np.array(mrrs)


def sig_test(mrrs_a, mrrs_b):
    \"\"\"Wilcoxon signed-rank test, returns (delta, p, sig_str).\"\"\"
    delta = float(np.mean(mrrs_a) - np.mean(mrrs_b))
    nonzero = int(np.sum(mrrs_a != mrrs_b))
    if nonzero > 10:
        _, p = wilcoxon(mrrs_a, mrrs_b)
    else:
        p = 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return delta, float(p), sig, nonzero


# === 1. Individual signal MRR ===
print("\\n" + "=" * 70)
print("INDIVIDUAL SIGNAL RANKING")
print("=" * 70)

individual_mrrs = {}
for sig in SIGNAL_NAMES:
    individual_mrrs[sig] = mrr_for_signal(all_results, sig)

bare_mrrs = individual_mrrs['bare']
print(f"\\n{'Signal':<12} {'MRR':>8} {'ΔMRR':>8} {'p':>12} {'Sig':>5} {'Changed':>8}")
print("-" * 60)

individual_stats = {}
for sig in SIGNAL_NAMES:
    mrrs = individual_mrrs[sig]
    if sig == 'bare':
        print(f"{sig:<12} {np.mean(mrrs):>8.4f} {'--':>8} {'--':>12} {'--':>5} {'--':>8}")
    else:
        d, p, s_str, n_changed = sig_test(mrrs, bare_mrrs)
        print(f"{sig:<12} {np.mean(mrrs):>8.4f} {d:>+8.4f} {p:>12.3e} {s_str:>5} {n_changed:>8}")
        individual_stats[sig] = {'delta_mrr': d, 'p_value': p, 'significant': bool(p < 0.05)}


# === 2. Ensemble MRR ===
print("\\n" + "=" * 70)
print("ENSEMBLE RANKING (EQUAL-WEIGHT NLL AVERAGE)")
print("=" * 70)

ENSEMBLE_CONFIGS = {
    'ens_2_sf':      ['bare', 'sf'],
    'ens_2_rand':    ['bare', 'rand'],
    'ens_2_intent':  ['bare', 'intent'],
    'ens_2_rescore': ['bare', 'rescore'],
    'ens_3':         ['bare', 'sf', 'rand'],
    'ens_4':         ['bare', 'sf', 'rand', 'intent'],
    'ens_5_all':     ['bare', 'sf', 'rand', 'intent', 'rescore'],
}

ensemble_mrrs = {}
ensemble_stats = {}
print(f"\\n{'Ensemble':<20} {'Members':>4} {'MRR':>8} {'ΔMRR':>8} {'p':>12} {'Sig':>5} {'Changed':>8}")
print("-" * 72)

for ens_name, members in ENSEMBLE_CONFIGS.items():
    mrrs = mrr_for_ensemble(all_results, members)
    ensemble_mrrs[ens_name] = mrrs
    d, p, s_str, n_changed = sig_test(mrrs, bare_mrrs)
    print(f"{ens_name:<20} {len(members):>4} {np.mean(mrrs):>8.4f} {d:>+8.4f} "
          f"{p:>12.3e} {s_str:>5} {n_changed:>8}")
    ensemble_stats[ens_name] = {
        'members': members,
        'mrr_mean': float(np.mean(mrrs)),
        'delta_mrr': d,
        'p_value': p,
        'significant': bool(p < 0.05),
        'n_changed': n_changed,
    }


# === 3. Critical comparison: priming vs non-priming control ===
print("\\n" + "=" * 70)
print("CRITICAL: PRIMING vs NON-PRIMING CONTROL")
print("=" * 70)

sf_mrr = float(np.mean(ensemble_mrrs['ens_2_sf']))
rescore_mrr = float(np.mean(ensemble_mrrs['ens_2_rescore']))
d_sf_res, p_sf_res, s_sf_res, n_sf_res = sig_test(
    ensemble_mrrs['ens_2_sf'], ensemble_mrrs['ens_2_rescore'])

print(f"  ens_2_sf (priming):      MRR={sf_mrr:.4f}")
print(f"  ens_2_rescore (control): MRR={rescore_mrr:.4f}")
print(f"  Difference:              {d_sf_res:+.4f}  (p={p_sf_res:.3e}, {s_sf_res})")
if sf_mrr > rescore_mrr + 0.001:
    print("  => Priming adds value BEYOND prompt diversity")
elif rescore_mrr > sf_mrr + 0.001:
    print("  => Prompt diversity alone BEATS priming")
else:
    print("  => Priming and prompt diversity are equivalent")


# === 4. Greedy forward selection (scaling curve) ===
print("\\n" + "=" * 70)
print("GREEDY SCALING CURVE: best member to add at each step")
print("=" * 70)

available = ['rescore', 'sf', 'rand', 'intent']
selected = ['bare']
greedy_results = [{'members': list(selected), 'mrr': float(np.mean(bare_mrrs))}]

for step in range(len(available)):
    best_next = None
    best_mrr = -1
    for candidate in available:
        trial = selected + [candidate]
        trial_mrrs = mrr_for_ensemble(all_results, trial)
        mean_mrr = float(np.mean(trial_mrrs))
        if mean_mrr > best_mrr:
            best_mrr = mean_mrr
            best_next = candidate
    selected.append(best_next)
    available.remove(best_next)
    greedy_results.append({'members': list(selected), 'mrr': best_mrr})

print(f"\\n{'K':<4} {'Added':>10} {'Ensemble':<35} {'MRR':>8} {'ΔMRR':>8}")
print("-" * 70)
for i, gr in enumerate(greedy_results):
    added = gr['members'][-1] if i > 0 else '--'
    delta = gr['mrr'] - greedy_results[0]['mrr']
    members_str = '+'.join(gr['members'])
    print(f"{i+1:<4} {added:>10} {members_str:<35} {gr['mrr']:>8.4f} {delta:>+8.4f}")


# === 5. NLL correlation matrix ===
print("\\n" + "=" * 70)
print("NLL CORRELATION MATRIX (Pearson, across all passages)")
print("=" * 70)

all_nlls = {sig: [] for sig in SIGNAL_NAMES}
for r in all_results:
    for p in r['passage_data']:
        for sig in SIGNAL_NAMES:
            all_nlls[sig].append(p[f'{sig}_nll'])

all_nlls = {sig: np.array(vals) for sig, vals in all_nlls.items()}

print(f"\\n{'':>12}", end='')
for sig in SIGNAL_NAMES:
    print(f" {sig:>10}", end='')
print()

corr_matrix = {}
for sig_a in SIGNAL_NAMES:
    print(f"{sig_a:>12}", end='')
    for sig_b in SIGNAL_NAMES:
        r, _ = stats.pearsonr(all_nlls[sig_a], all_nlls[sig_b])
        corr_matrix[f'{sig_a}_{sig_b}'] = float(r)
        print(f" {r:>10.4f}", end='')
    print()

print("\\nNote: Lower correlation = more diversity = better ensembles")\
""")))

# ========== Cell 8: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 8: Plots (4-panel figure)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors_ens = {
    'ens_2_sf': '#d62728',
    'ens_2_rand': '#ff7f0e',
    'ens_2_intent': '#9467bd',
    'ens_2_rescore': '#2ca02c',
    'ens_3': '#1f77b4',
    'ens_4': '#e377c2',
    'ens_5_all': '#17becf',
}

# --- Plot 1: Ensemble MRR bar chart ---
ax = axes[0, 0]
names = ['bare'] + list(ENSEMBLE_CONFIGS.keys())
mrr_vals = [float(np.mean(bare_mrrs))] + [float(np.mean(ensemble_mrrs[e])) for e in ENSEMBLE_CONFIGS]
bar_colors = ['#7f7f7f'] + [colors_ens.get(e, '#333') for e in ENSEMBLE_CONFIGS]
bars = ax.bar(range(len(names)), mrr_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, (n, m) in enumerate(zip(names, mrr_vals)):
    ax.text(i, m + 0.002, f"{m:.4f}", ha='center', fontsize=7, rotation=45)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel("MRR")
ax.set_title("MRR by Condition")
ax.axhline(y=float(np.mean(bare_mrrs)), color='gray', linestyle='--', alpha=0.5, label='bare')
ax.legend(fontsize=8)

# --- Plot 2: Scaling curve ---
ax = axes[0, 1]
k_vals = list(range(1, len(greedy_results) + 1))
mrr_curve = [gr['mrr'] for gr in greedy_results]
ax.plot(k_vals, mrr_curve, 'o-', color='#1f77b4', linewidth=2, markersize=8)
for i, gr in enumerate(greedy_results):
    label = gr['members'][-1] if i > 0 else 'bare'
    ax.annotate(f"+{label}" if i > 0 else label,
                (k_vals[i], mrr_curve[i]),
                textcoords="offset points", xytext=(5, 8), fontsize=8)
ax.axhline(y=float(np.mean(bare_mrrs)), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Ensemble Size K")
ax.set_ylabel("MRR")
ax.set_title("Greedy Scaling Curve")
ax.set_xticks(k_vals)

# --- Plot 3: Correlation heatmap ---
ax = axes[1, 0]
corr_data = np.zeros((len(SIGNAL_NAMES), len(SIGNAL_NAMES)))
for i, sa in enumerate(SIGNAL_NAMES):
    for j, sb in enumerate(SIGNAL_NAMES):
        corr_data[i, j] = corr_matrix[f'{sa}_{sb}']
im = ax.imshow(corr_data, vmin=0.9, vmax=1.0, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(SIGNAL_NAMES)))
ax.set_xticklabels(SIGNAL_NAMES, fontsize=8)
ax.set_yticks(range(len(SIGNAL_NAMES)))
ax.set_yticklabels(SIGNAL_NAMES, fontsize=8)
for i in range(len(SIGNAL_NAMES)):
    for j in range(len(SIGNAL_NAMES)):
        ax.text(j, i, f"{corr_data[i,j]:.3f}", ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("NLL Correlation Matrix")

# --- Plot 4: Per-query ΔMRR distributions ---
ax = axes[1, 1]
for ens_name in ['ens_2_sf', 'ens_2_rescore', 'ens_4']:
    deltas = ensemble_mrrs[ens_name] - bare_mrrs
    ax.hist(deltas, bins=30, alpha=0.5, label=ens_name,
            color=colors_ens.get(ens_name, 'gray'))
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("ΔMRR (ensemble - bare)")
ax.set_ylabel("Count")
ax.set_title("Per-Query MRR Change Distribution")
ax.legend(fontsize=8)

plt.suptitle('Exp 15: NLL Ensemble Ranking', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 9: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 9: Save results JSON
final = {
    'experiment': 'exp15_nll_ensemble_ranking',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_queries': N,
        'n_valid': N_VALID,
        'max_passage_words': MAX_PASSAGE_WORDS,
        'min_passages_per_query': MIN_PASSAGES_PER_QUERY,
        'dataset': 'MS MARCO v1.1 validation',
        'prefixes': {
            'sf': STATIC_FACT,
            'rand': RANDOM_PREFIX_TEXT,
            'intent': INTENT_PREFIX_TEXT,
        },
        'alt_query_template': ALT_QUERY_TEMPLATE,
    },
    'signal_names': SIGNAL_NAMES,
    'individual_mrrs': {sig: float(np.mean(individual_mrrs[sig])) for sig in SIGNAL_NAMES},
    'individual_stats': individual_stats,
    'ensemble_configs': {k: v for k, v in ENSEMBLE_CONFIGS.items()},
    'ensemble_stats': ensemble_stats,
    'priming_vs_control': {
        'ens_2_sf_mrr': sf_mrr,
        'ens_2_rescore_mrr': rescore_mrr,
        'difference': float(d_sf_res),
        'p_value': float(p_sf_res),
        'priming_is_special': bool(sf_mrr > rescore_mrr + 0.001),
    },
    'greedy_scaling': greedy_results,
    'correlation_matrix': corr_matrix,
    'per_query_results': [
        {k: v for k, v in r.items() if k != 'passage_data'}
        for r in all_results
    ],
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Print summary
print("\\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Bare MRR:           {float(np.mean(bare_mrrs)):.4f}")
best_ens = max(ensemble_stats.items(), key=lambda x: x[1]['mrr_mean'])
print(f"Best ensemble:      {best_ens[0]} (MRR={best_ens[1]['mrr_mean']:.4f}, "
      f"ΔMRR={best_ens[1]['delta_mrr']:+.4f}, p={best_ens[1]['p_value']:.3e})")
print(f"Priming vs control: {d_sf_res:+.4f} (p={p_sf_res:.3e})")
print(f"Scaling saturates:  K={len(greedy_results[-1]['members'])} members, "
      f"MRR={greedy_results[-1]['mrr']:.4f}")
print("\\nDone!")\
""")))


# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "cells": cells
}

output_path = "/home/jupyter/research/directed_kvcache_v2/15_nll_ensemble_ranking.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
