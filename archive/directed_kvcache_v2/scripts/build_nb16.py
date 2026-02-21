#!/usr/bin/env python3
"""Build script for 16_cross_model_gemma3.ipynb

Exp 16: Cross-Model Priming Replication (Gemma 3 4B)

All 15 experiments used Mistral-7B exclusively. The critical open question:
is value contamination via priming a universal transformer mechanism, or
Mistral-specific? Gemma 3 4B is an ideal comparison — different architecture
(34 layers, head_dim=256, per-layer RoPE, 4 KV heads) but all lib functions
already support it.

5 conditions on MS MARCO (where priming works on Mistral), N=300 queries:

| Condition        | Description                                        |
|------------------|----------------------------------------------------|
| bare             | Baseline                                           |
| static_fact_trunc| "What are the key facts?" prefix, truncated+RoPE   |
| random_trunc     | Random text prefix, truncated+RoPE                 |
| oracle_trunc     | Actual query as prefix, truncated+RoPE             |
| values_only      | Bare keys + sf primed values (hybrid cache)        |

Compute: ~300 queries × ~8 passages × 4 forward passes + 5 scoring ≈ 1.5-2.5h on L4
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

# ========== Cell 0: Markdown overview ==========
cells.append(make_cell("markdown", s("""\
# Exp 16: Cross-Model Priming Replication (Gemma 3 4B)

## Motivation

All 15 prior experiments used Mistral-7B-Instruct-v0.2 exclusively. The critical open
question: **is value contamination via priming a universal transformer mechanism, or
Mistral-specific?**

Gemma 3 4B is an ideal comparison model:
- Different architecture: 34 layers (vs 32), head_dim=256 (vs 128), 4 KV heads (vs 8)
- Per-layer RoPE: sliding_attention=10k theta, full_attention=1M theta (vs uniform 1M)
- GQA with different grouping ratio
- bfloat16 required (float16 produces garbage)
- All lib functions already support it

## Architecture Comparison

| Property | Mistral-7B | Gemma 3 4B |
|----------|-----------|------------|
| Parameters | 7B | 4B |
| Layers | 32 | 34 |
| Hidden size | 4096 | 2560 |
| Attention heads | 32 | 8 |
| KV heads | 8 | 4 |
| Head dim | 128 | 256 |
| RoPE theta | 1M (uniform) | 10k/1M (per-layer) |
| Dtype | float16 | bfloat16 |

## Design

**5 conditions on MS MARCO v1.1 (where priming works on Mistral), N=300 queries:**

| # | Condition | Description | Tests |
|---|-----------|-------------|-------|
| 1 | bare | Baseline — no prefix | — |
| 2 | static_fact_trunc | "What are the key facts?" prefix, truncated+RoPE | Best Mistral condition on Gemma? |
| 3 | random_trunc | Random text prefix, truncated+RoPE | Does any prefix help? |
| 4 | oracle_trunc | Actual query as prefix, truncated+RoPE | Is random > oracle on Gemma too? |
| 5 | values_only | Bare keys + sf primed values (hybrid cache) | Is it value contamination on Gemma? |

## Mistral Reference Values (from Exp 01 + 08)

| Metric | Mistral Value | Source |
|--------|--------------|--------|
| static_fact_trunc d | +0.472 | Exp 07 |
| random_trunc d | +0.091 | Exp 01 |
| oracle_trunc d | +0.023 (ns) | Exp 01 |
| values_only fraction | 108% | Exp 08 |
| keys_only fraction | -4% | Exp 08 |""")))

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

RESULTS_DIR = Path("results/exp16")
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

# ========== Cell 2: Load Gemma 3 4B ==========
cells.append(make_cell("code", s("""\
# Cell 2: Load Gemma 3 4B via load_model()
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.model_utils import load_model

MODEL_NAME = "google/gemma-3-4b-it"

exp_config = ExperimentConfig(
    model_name=MODEL_NAME,
    model_type="gemma3",
    compute_dtype="auto",  # resolves to bfloat16 for gemma3
    use_4bit=True,
    num_samples=2000,
    seed=SEED,
)

print(f"Loading {MODEL_NAME} (4-bit, bfloat16)...")
model, tokenizer = load_model(exp_config)

# Architecture diagnostics
from lib.kv_cache import _get_text_config, _get_head_dim, _get_rope_theta_for_layer, _get_cache_keys, _ensure_dynamic_cache

text_config = _get_text_config(model.config)
print(f"\\nModel loaded successfully.")
print(f"  Model class: {type(model).__name__}")
print(f"  Text config class: {type(text_config).__name__}")
print(f"  Hidden size: {text_config.hidden_size}")
print(f"  Num layers: {text_config.num_hidden_layers}")
print(f"  Num attention heads: {text_config.num_attention_heads}")
print(f"  Num KV heads: {text_config.num_key_value_heads}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  BOS token ID: {tokenizer.bos_token_id}")
print(f"  EOS token ID: {tokenizer.eos_token_id}")

# Per-layer RoPE diagnostics
thetas = set()
for layer_idx in range(text_config.num_hidden_layers):
    thetas.add(_get_rope_theta_for_layer(model.config, layer_idx))
print(f"  Unique RoPE thetas: {sorted(thetas)}")

# Verify dtype
sample_ids = tokenizer("test", return_tensors="pt")['input_ids'].to(exp_config.device)
with torch.no_grad():
    out = model(sample_ids, use_cache=True)
    cache_check = _ensure_dynamic_cache(out.past_key_values)
    k0 = _get_cache_keys(cache_check, 0)
    print(f"  Cache key dtype: {k0.dtype}")
    print(f"  Cache key shape: {k0.shape}  (batch, kv_heads, seq, head_dim)")
del out, sample_ids
torch.cuda.empty_cache()\
""")))

# ========== Cell 3: Lib imports + templates + prefix definitions ==========
cells.append(make_cell("code", s("""\
# Cell 3: Config and library imports
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
from lib.analysis import cohens_d
from lib.data import count_words
from scipy import stats
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Prefix texts
from lib.surrogate import STATIC_SURROGATE_QUERIES
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']
RANDOM_PREFIX_TEXT = "The purple elephant danced gracefully on the frozen lake during twilight"

# Experiment parameters
MAX_QUERIES = 300
MAX_PASSAGE_WORDS = 300
MIN_PASSAGES_PER_QUERY = 2
CHECKPOINT_EVERY = 25

CONDITION_NAMES = ['bare', 'static_fact_trunc', 'random_trunc', 'oracle_trunc', 'values_only']

# Mistral reference values (from Exp 01 + 08)
MISTRAL_REF = {
    'random_trunc_d': 0.091,
    'oracle_trunc_d': 0.023,
    'static_fact_trunc_d': 0.472,
    'values_fraction': 1.083,
    'keys_fraction': -0.036,
    'd_full_trunc': 0.254,
    'd_values_only': 0.275,
    'd_keys_only': -0.009,
}

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  MAX_QUERIES: {MAX_QUERIES}")
print(f"  Conditions: {CONDITION_NAMES}")
print(f"  Prefixes:")
print(f"    static_fact: '{STATIC_FACT}'")
print(f"    random:      '{RANDOM_PREFIX_TEXT}'")
print(f"    oracle:      (actual query text per sample)")
print(f"  Mistral reference d values:")
for k, v in MISTRAL_REF.items():
    print(f"    {k}: {v:+.3f}")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO v1.1 (same filtering as previous ranking experiments)
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

# ========== Cell 5: Tokenize prefixes + BPE boundary check ==========
cells.append(make_cell("code", s("""\
# Cell 5: Tokenize prefixes and verify BPE boundaries

print("=" * 70)
print("PREFIX TOKENIZATION — GEMMA 3 4B")
print("=" * 70)

# Tokenize each prefix
sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
rand_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=RANDOM_PREFIX_TEXT)

sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(exp_config.device)
rand_ids = tokenizer(rand_str, return_tensors="pt",
                      add_special_tokens=False)['input_ids'].to(exp_config.device)

# Example oracle prefix
example_oracle_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=queries[0]['query'])
oracle_example_ids = tokenizer(example_oracle_str, return_tensors="pt",
                                add_special_tokens=False)['input_ids'].to(exp_config.device)

PREFIX_CONFIGS = [
    ('static_fact', STATIC_FACT, sf_str, sf_ids),
    ('random', RANDOM_PREFIX_TEXT, rand_str, rand_ids),
    ('oracle (ex)', queries[0]['query'], example_oracle_str, oracle_example_ids),
]

print("\\nPREFIX TOKEN LENGTHS (Gemma tokenizer):")
for name, text, full_str, ids in PREFIX_CONFIGS:
    print(f"  {name:<15} {ids.shape[1]:>3} tokens | '{text}'")

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
    print(f"  {name:<15}: {match}/{total} tokens match ({100*match/total:.1f}%)")

# Condition summary
print("\\n" + "=" * 70)
print("CONDITION DETAILS")
print("=" * 70)

conditions_detail = [
    ("bare",
     "Standard bare cache: [BOS][doc]",
     "Baseline. All other conditions compared to this."),
    ("static_fact_trunc",
     f"Prefix: '{STATIC_FACT}' → truncate + RoPE correct",
     "Best single Mistral condition (d=+0.472). Does it replicate on Gemma?"),
    ("random_trunc",
     f"Prefix: '{RANDOM_PREFIX_TEXT}' → truncate + RoPE correct",
     "Random prefix (d=+0.091 on Mistral). Does ANY prefix help on Gemma?"),
    ("oracle_trunc",
     "Prefix: actual query text → truncate + RoPE correct",
     "Oracle prefix (d=+0.023 ns on Mistral). Is random > oracle on Gemma too?"),
    ("values_only",
     "Bare keys + static_fact primed values (hybrid cache)",
     "Tests value contamination mechanism. On Mistral: values=108%, keys=-4%."),
]

for name, detail, purpose in conditions_detail:
    print(f"\\n### {name} ###")
    print(f"  {detail}")
    print(f"  Purpose: {purpose}")\
""")))

# ========== Cell 6: Main loop ==========
cells.append(make_cell("code", s("""\
# Cell 6: Main loop — score all passages under all 5 conditions

print("=" * 70)
print(f"MAIN EVALUATION ({N} queries, ~{total_passages} passages)")
print("Model: Gemma 3 4B")
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
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    passage_results = []

    for pidx, pinfo in enumerate(query_data['passages']):
        passage = pinfo['passage']
        document_text = DOCUMENT_TEMPLATE.format(document=passage)

        # --- Matched tokenization (using sf prefix as reference) ---
        full_text = sf_str + document_text
        full_enc = tokenizer(full_text, return_tensors="pt",
                              add_special_tokens=True, padding=False, truncation=False)
        full_ids = full_enc['input_ids'].to(exp_config.device)

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

        # === 2-4. For each priming prefix: build primed cache, truncate + RoPE ===
        # Prefix configs for this passage
        oracle_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=query)
        oracle_ids_local = tokenizer(oracle_str, return_tensors="pt",
                                      add_special_tokens=False)['input_ids'].to(exp_config.device)

        prefix_configs_local = [
            ('static_fact_trunc', sf_ids),
            ('random_trunc', rand_ids),
            ('oracle_trunc', oracle_ids_local),
        ]

        primed_caches = {}  # Store truncated+corrected caches

        for p_name, p_ids in prefix_configs_local:
            primed_input = torch.cat([bos_id, p_ids, doc_ids], dim=1)
            with torch.no_grad():
                primed_out = model(input_ids=primed_input,
                                   attention_mask=torch.ones_like(primed_input),
                                   use_cache=True, return_dict=True)
            primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
            del primed_out, primed_input

            # Truncate: keep [BOS] + [last doc_len positions]
            primed_trunc = extract_and_truncate_cache_with_bos(primed_full, doc_len)
            # RoPE correct: shift doc positions back by prefix_len
            correct_rope_positions_with_bos(primed_trunc, p_ids.shape[1], model)
            del primed_full

            primed_caches[p_name] = primed_trunc

        del oracle_ids_local

        # === 5. Build values_only hybrid cache (bare keys + sf primed values) ===
        hybrid_values = build_hybrid_cache(
            keys_source=bare_cache,
            values_source=primed_caches['static_fact_trunc'],
        )

        # === Score all conditions ===
        # Score hybrid and primed conditions first (need deepcopy for each)
        nll_values_only = score_answer_with_cache(
            deepcopy_cache(hybrid_values), context_len, query_prompt, answer_text,
            model, tokenizer, exp_config)
        del hybrid_values

        nll_sf = score_answer_with_cache(
            deepcopy_cache(primed_caches['static_fact_trunc']), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)

        nll_rand = score_answer_with_cache(
            deepcopy_cache(primed_caches['random_trunc']), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)

        nll_oracle = score_answer_with_cache(
            deepcopy_cache(primed_caches['oracle_trunc']), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)

        del primed_caches

        # Score bare LAST (mutates cache)
        nll_bare = score_answer_with_cache(
            bare_cache, context_len, query_prompt, answer_text,
            model, tokenizer, exp_config)
        del bare_cache

        gc.collect()
        torch.cuda.empty_cache()

        passage_results.append({
            'passage_idx': pinfo['passage_idx'],
            'is_relevant': pinfo['is_relevant'],
            'word_count': pinfo['word_count'],
            'bare_nll': nll_bare,
            'static_fact_trunc_nll': nll_sf,
            'random_trunc_nll': nll_rand,
            'oracle_trunc_nll': nll_oracle,
            'values_only_nll': nll_values_only,
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
# Cell 7: Analysis — per-condition stats, Cohen's d, Wilcoxon, cross-model comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — CROSS-MODEL PRIMING REPLICATION")
print("=" * 70)

N_VALID = len(all_results)
print(f"Valid queries: {N_VALID}")

# --- Collect per-passage NLLs ---
cond_nlls = {cn: [] for cn in CONDITION_NAMES}
for r in all_results:
    for p in r['passage_data']:
        for cn in CONDITION_NAMES:
            cond_nlls[cn].append(p[f'{cn}_nll'])

cond_arrays = {cn: np.array(vals) for cn, vals in cond_nlls.items()}

# Filter zero NLLs
valid = np.ones(len(cond_arrays['bare']), dtype=bool)
for cn in CONDITION_NAMES:
    valid &= (cond_arrays[cn] != 0)
n_passages_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total passages: {len(valid)}, Valid: {n_passages_valid}, Excluded: {n_excluded}")

c = {}
for cn in CONDITION_NAMES:
    c[cn] = cond_arrays[cn][valid]

# === 1. NLL Summary Table ===
print("\\n" + "=" * 70)
print("NLL SUMMARY (per-passage, Gemma 3 4B)")
print("=" * 70)

print(f"\\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10} {'Win%':>8}")
print("-" * 68)

gemma_ds = {}
for cn in CONDITION_NAMES:
    mean_nll = np.mean(c[cn])
    std_nll = np.std(c[cn])
    if cn == 'bare':
        print(f"{cn:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {'—':>10} {'—':>8}")
    else:
        delta = c['bare'] - c[cn]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        gemma_ds[cn] = d
        print(f"{cn:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d:>+10.3f} {win:>7.1f}%")

# === 2. Statistical Tests ===
print("\\n" + "=" * 70)
print("STATISTICAL TESTS (paired t-test, per-passage)")
print("=" * 70)

print(f"\\n{'Condition':<25} {'Mean ΔNLL':>10} {'d':>8} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 70)

stat_results = {}
for cn in CONDITION_NAMES:
    if cn == 'bare':
        continue
    delta = c['bare'] - c[cn]
    d = cohens_d(delta)
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{cn:<25} {np.mean(delta):>10.4f} {d:>+8.3f} {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    stat_results[cn] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
        'win_rate': float(np.mean(delta > 0)),
    }

# === 3. Mechanism Decomposition ===
print("\\n" + "=" * 70)
print("MECHANISM DECOMPOSITION (Gemma 3 4B)")
print("=" * 70)

d_sf = gemma_ds.get('static_fact_trunc', 0)
d_vo = gemma_ds.get('values_only', 0)

values_fraction = d_vo / d_sf if d_sf != 0 else float('nan')

print(f"\\n  Full static_fact_trunc (keys+values): d = {d_sf:+.3f}")
print(f"  Values-only (bare keys + sf values):   d = {d_vo:+.3f}")
if d_sf != 0:
    print(f"  Values fraction of full effect:        {values_fraction:.1%}")

if d_vo > 0 and values_fraction > 0.8:
    print(f"\\n  => VALUE CONTAMINATION confirmed on Gemma 3 4B")
    print(f"     (values carry {values_fraction:.0%} of the effect, same as Mistral's {MISTRAL_REF['values_fraction']:.0%})")
elif d_vo > 0:
    print(f"\\n  => Values contribute ({values_fraction:.0%}) but keys also matter on Gemma")
else:
    print(f"\\n  => Values-only does NOT help on Gemma — different mechanism?")

# === 4. Cross-Model Comparison Table ===
print("\\n" + "=" * 70)
print("CROSS-MODEL COMPARISON: Gemma 3 4B vs Mistral 7B")
print("=" * 70)

mistral_ref_ds = {
    'static_fact_trunc': MISTRAL_REF['static_fact_trunc_d'],
    'random_trunc': MISTRAL_REF['random_trunc_d'],
    'oracle_trunc': MISTRAL_REF['oracle_trunc_d'],
}

print(f"\\n{'Condition':<25} {'Mistral d':>10} {'Gemma d':>10} {'Ratio':>8} {'Same Sign?':>12}")
print("-" * 70)

for cn in ['static_fact_trunc', 'random_trunc', 'oracle_trunc']:
    m_d = mistral_ref_ds.get(cn, 0)
    g_d = gemma_ds.get(cn, 0)
    ratio = g_d / m_d if m_d != 0 else float('nan')
    same_sign = "Yes" if (m_d > 0 and g_d > 0) or (m_d < 0 and g_d < 0) or (m_d == 0 and g_d == 0) else "NO"
    print(f"{cn:<25} {m_d:>+10.3f} {g_d:>+10.3f} {ratio:>8.2f} {same_sign:>12}")

print(f"\\n{'Mechanism':<25} {'Mistral':>10} {'Gemma':>10}")
print("-" * 50)
print(f"{'Values fraction':<25} {MISTRAL_REF['values_fraction']:>10.1%} {values_fraction:>10.1%}")

# Verdict
print("\\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

sf_replicates = gemma_ds.get('static_fact_trunc', 0) > 0 and stat_results.get('static_fact_trunc', {}).get('significant', False)
rand_helps = gemma_ds.get('random_trunc', 0) > 0
oracle_worse_than_rand = gemma_ds.get('random_trunc', 0) > gemma_ds.get('oracle_trunc', 0)
values_mechanism = d_vo > 0 and values_fraction > 0.5

if sf_replicates and values_mechanism:
    print("  UNIVERSAL MECHANISM: Priming via value contamination replicates on Gemma 3 4B.")
    print(f"  static_fact_trunc: d={gemma_ds.get('static_fact_trunc', 0):+.3f} (Mistral: d={MISTRAL_REF['static_fact_trunc_d']:+.3f})")
    print(f"  Value contamination: {values_fraction:.0%} (Mistral: {MISTRAL_REF['values_fraction']:.0%})")
elif sf_replicates:
    print("  PARTIAL REPLICATION: Priming helps but mechanism may differ.")
else:
    print("  MISTRAL-SPECIFIC: Priming does NOT replicate on Gemma 3 4B.")

if rand_helps:
    print(f"  Random prefix helps: d={gemma_ds.get('random_trunc', 0):+.3f}")
if oracle_worse_than_rand:
    print(f"  Random > Oracle replicates: rand d={gemma_ds.get('random_trunc', 0):+.3f} > oracle d={gemma_ds.get('oracle_trunc', 0):+.3f}")

# === 5. Hardness Interaction ===
print("\\n" + "=" * 70)
print("HARDNESS INTERACTION")
print("=" * 70)

bare_all = c['bare']
quintile_boundaries = np.percentile(bare_all, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_all])

hardness_conds = ['static_fact_trunc', 'random_trunc', 'oracle_trunc', 'values_only']

header = f"{'Condition':<25}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (25 + 14 * 6))

hardness_breakdown = {}
for cn in hardness_conds:
    row = f"{cn:<25}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            delta = bare_all[mask_q] - c[cn][mask_q]
            d = cohens_d(delta)
            row += f"{d:>+14.3f}"
            quintile_ds.append(float(d))
    d_all = cohens_d(bare_all - c[cn])
    row += f"{d_all:>+14.3f}"
    print(row)
    hardness_breakdown[cn] = {'quintile_ds': quintile_ds, 'overall_d': float(d_all)}

# Hardness correlation
print("\\nHardness correlation (bare NLL vs delta):")
for cn in hardness_conds:
    delta = bare_all - c[cn]
    r, p = stats.pearsonr(bare_all, delta)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {cn:<25} r={r:+.3f}  p={p:.2e}  {sig}")\
""")))

# ========== Cell 8: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 8: Plots — 4-panel figure

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Color scheme
colors = {
    'bare': '#7f7f7f',
    'static_fact_trunc': '#d62728',
    'random_trunc': '#ff7f0e',
    'oracle_trunc': '#2ca02c',
    'values_only': '#1f77b4',
}

# --- Plot 1: Cohen's d bar chart with Mistral reference ---
ax = axes[0, 0]
conds_plot = ['static_fact_trunc', 'random_trunc', 'oracle_trunc', 'values_only']
gemma_d_vals = [gemma_ds.get(cn, 0) for cn in conds_plot]
mistral_d_vals = [mistral_ref_ds.get(cn, MISTRAL_REF.get(f'd_{cn}', 0)) for cn in conds_plot]

# For values_only, use Mistral d_values_only
mistral_d_vals[3] = MISTRAL_REF['d_values_only']

x = np.arange(len(conds_plot))
width = 0.35
bars_g = ax.bar(x - width/2, gemma_d_vals, width, label='Gemma 3 4B',
                color=[colors[cn] for cn in conds_plot], edgecolor='black', linewidth=0.5)
bars_m = ax.bar(x + width/2, mistral_d_vals, width, label='Mistral 7B (ref)',
                color=[colors[cn] for cn in conds_plot], edgecolor='black', linewidth=0.5,
                alpha=0.4, hatch='//')

for i, (gd, md) in enumerate(zip(gemma_d_vals, mistral_d_vals)):
    ax.text(i - width/2, gd + 0.01, f"{gd:+.3f}", ha='center', fontsize=7, rotation=45)
    ax.text(i + width/2, md + 0.01, f"{md:+.3f}", ha='center', fontsize=7, rotation=45, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels([cn.replace('_trunc', '').replace('_', '\\n') for cn in conds_plot],
                    fontsize=8)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Cross-Model: Gemma vs Mistral Effect Sizes")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# --- Plot 2: Hardness scatter (static_fact_trunc) ---
ax = axes[0, 1]
delta_sf = c['bare'] - c['static_fact_trunc']
ax.scatter(c['bare'], delta_sf, alpha=0.15, s=8, color=colors['static_fact_trunc'])
# Fit line
z = np.polyfit(c['bare'], delta_sf, 1)
x_fit = np.linspace(c['bare'].min(), c['bare'].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), color='black', linewidth=2, linestyle='--')
r_sf, p_sf = stats.pearsonr(c['bare'], delta_sf)
ax.set_xlabel("Bare NLL (difficulty)")
ax.set_ylabel("ΔNLL (bare - static_fact)")
ax.set_title(f"Hardness Interaction (r={r_sf:+.3f}, p={p_sf:.1e})")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# --- Plot 3: Mechanism decomposition ---
ax = axes[1, 0]
decomp_labels = ['Full\\n(K+V)', 'Values\\nonly']
gemma_decomp = [d_sf, d_vo]
mistral_decomp = [MISTRAL_REF['d_full_trunc'], MISTRAL_REF['d_values_only']]

x_dec = np.arange(len(decomp_labels))
width_dec = 0.35
ax.bar(x_dec - width_dec/2, gemma_decomp, width_dec, label='Gemma 3 4B',
       color=['#d62728', '#1f77b4'], edgecolor='black', linewidth=0.5)
ax.bar(x_dec + width_dec/2, mistral_decomp, width_dec, label='Mistral 7B (ref)',
       color=['#d62728', '#1f77b4'], edgecolor='black', linewidth=0.5,
       alpha=0.4, hatch='//')

for i, (gd, md) in enumerate(zip(gemma_decomp, mistral_decomp)):
    ax.text(i - width_dec/2, gd + 0.005, f"{gd:+.3f}", ha='center', va='bottom', fontsize=9)
    ax.text(i + width_dec/2, md + 0.005, f"{md:+.3f}", ha='center', va='bottom', fontsize=9, alpha=0.6)

ax.set_xticks(x_dec)
ax.set_xticklabels(decomp_labels, fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Mechanism: Value Contamination Decomposition")
ax.legend(fontsize=8)

# --- Plot 4: NLL distributions (all conditions) ---
ax = axes[1, 1]
for cn in CONDITION_NAMES:
    ax.hist(c[cn], bins=50, alpha=0.4, label=cn, color=colors.get(cn, 'gray'))
ax.set_xlabel("NLL")
ax.set_ylabel("Count")
ax.set_title("NLL Distributions (all conditions)")
ax.legend(fontsize=7)

plt.suptitle('Exp 16: Cross-Model Priming — Gemma 3 4B vs Mistral 7B', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 9: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 9: Save results JSON
final = {
    'experiment': 'exp16_cross_model_gemma3',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'n_queries': N,
        'n_valid': N_VALID,
        'n_passages_valid': n_passages_valid,
        'n_passages_excluded': n_excluded,
        'max_passage_words': MAX_PASSAGE_WORDS,
        'min_passages_per_query': MIN_PASSAGES_PER_QUERY,
        'dataset': 'MS MARCO v1.1 validation',
        'prefixes': {
            'static_fact': STATIC_FACT,
            'random': RANDOM_PREFIX_TEXT,
            'oracle': '(actual query per sample)',
        },
    },
    'gemma_architecture': {
        'hidden_size': text_config.hidden_size,
        'num_layers': text_config.num_hidden_layers,
        'num_attention_heads': text_config.num_attention_heads,
        'num_kv_heads': text_config.num_key_value_heads,
        'head_dim': _get_head_dim(model.config),
        'rope_thetas': sorted(list(thetas)),
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': {
        cn: {
            'mean': float(np.mean(c[cn])),
            'std': float(np.std(c[cn])),
            'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cn])) if cn != 'bare' else 0.0,
        }
        for cn in CONDITION_NAMES
    },
    'statistical_tests': stat_results,
    'mechanism_decomposition': {
        'd_full_sf': float(d_sf),
        'd_values_only': float(d_vo),
        'values_fraction': float(values_fraction) if not np.isnan(values_fraction) else None,
    },
    'cross_model_comparison': {
        'mistral_reference': MISTRAL_REF,
        'gemma_ds': {cn: float(gemma_ds[cn]) for cn in gemma_ds},
    },
    'hardness_breakdown': hardness_breakdown,
    'per_query_results': all_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Print summary
print("\\n" + "=" * 70)
print("SUMMARY — Exp 16: Cross-Model Priming Replication")
print("=" * 70)
print(f"Model: Gemma 3 4B (34 layers, head_dim=256, bfloat16)")
print(f"Dataset: MS MARCO v1.1 ({N} queries, {n_passages_valid} passages)")
print(f"\\nEffect sizes (Cohen's d vs bare):")
for cn in ['static_fact_trunc', 'random_trunc', 'oracle_trunc', 'values_only']:
    g_d = gemma_ds.get(cn, 0)
    sig = stat_results.get(cn, {}).get('significant', False)
    sig_str = "(sig)" if sig else "(ns)"
    print(f"  {cn:<25} d={g_d:>+.3f}  {sig_str}")
print(f"\\nMechanism: values carry {values_fraction:.0%} of the effect")
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

output_path = "/home/jupyter/research/directed_kvcache_v2/16_cross_model_gemma3.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
