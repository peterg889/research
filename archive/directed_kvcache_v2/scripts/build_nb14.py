#!/usr/bin/env python3
"""Build script for 14_ranking_aware_priming.ipynb

Exp 14: Ranking-Aware Priming — Does Priming Improve Ad Ranking?

All prior experiments measured priming benefit as per-document NLL deltas.
But for ad serving, what matters is RANKING: does priming help you rank
the relevant document higher among candidates?

MS MARCO v1.1 has ~10 candidate passages per query with is_selected
relevance labels — a natural ad-ranking simulation.

Key Hypotheses:
  1. Differential Effect: Priming reduces NLL more for relevant passages
     than irrelevant ones
  2. Ranking Improvement: Even if average NLL worsens, the differential
     improves MRR/Hit@k
  3. Delta-as-Signal: The NLL delta (bare - primed) is a relevance predictor
  4. Gating Improvement: Selective priming beats always-prime for ranking

Conditions:
  1. bare           — bare NLL ranking
  2. primed_1x      — static_fact truncated NLL ranking
  3. primed_amp2x   — L0-15 2x amplified NLL ranking
  4. oracle_gated   — primed if bare_nll > median, else bare
  5. delta_signal   — bare_nll - primed_nll as ranking score
  6. combined       — alpha*bare + (1-alpha)*(-delta), alpha tuned
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
# Exp 14: Ranking-Aware Priming — Does Priming Improve Ad Ranking?

## Background & Motivation

All prior experiments (Exps 01-13) measured priming benefit as per-document NLL deltas.
But for ad serving, what matters is **ranking**: does priming help you rank the relevant
document higher among candidates?

Exp 13's per-example analysis found a "sweet spot" (short, hard, high-overlap docs) and
showed that priming response correlates with query-doc overlap (r=+0.10). This suggests
a **differential effect**: relevant passages may benefit more from priming than irrelevant
ones, which would improve ranking even if average NLL worsens.

MS MARCO v1.1 has ~10 candidate passages per query with `is_selected` relevance labels —
a natural ad-ranking simulation.

## Key Hypotheses

1. **Differential Effect**: Priming reduces NLL more for relevant vs irrelevant passages
2. **Ranking Improvement**: Even if average NLL worsens, the differential improves MRR/Hit@k
3. **Delta-as-Signal**: The NLL delta itself (bare - primed) is a relevance predictor (AUC > 0.5)
4. **Gating Improvement**: Selective priming (only hard passages) beats always-prime

## Conditions

| # | Condition | Ranking Signal | Tests |
|---|-----------|---------------|-------|
| 1 | `bare` | bare NLL | Baseline |
| 2 | `primed_1x` | static_fact truncated NLL | Does uniform priming help ranking? |
| 3 | `primed_amp2x` | L0-15 2x amplified NLL | Does stronger contamination help? |
| 4 | `oracle_gated` | primed if bare_nll > median, else bare | Selective priming |
| 5 | `delta_signal` | bare_nll - primed_nll (as score) | Is priming response a relevance signal? |
| 6 | `combined` | alpha*bare + (1-alpha)*(-delta), alpha tuned | Are NLL and delta complementary? |

## Compute Budget
~300 queries × ~8 passages × 2 forward passes = ~4800 forward passes + ~4800 scoring passes.
Estimated: ~60-90 min on L4.""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup
import os
os.umask(0o000)

import sys
import json
import time
import math
import numpy as np
import torch
import gc
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp14")
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
from lib.analysis import cohens_d, compute_ranking_metrics, compute_token_overlap
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

SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

MAX_QUERIES = 300
MAX_PASSAGE_WORDS = 300
MIN_PASSAGES_PER_QUERY = 2
CHECKPOINT_EVERY = 25

print("Config ready")
print(f"  MAX_QUERIES: {MAX_QUERIES}")
print(f"  MAX_PASSAGE_WORDS: {MAX_PASSAGE_WORDS}")
print(f"  MIN_PASSAGES_PER_QUERY: {MIN_PASSAGES_PER_QUERY}")
print(f"  static_fact: '{STATIC_FACT}'")\
""")))

# ========== Cell 4: Load ALL passages per query ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load ALL passages per query from MS MARCO v1.1
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 — ALL PASSAGES PER QUERY")
print("=" * 70)

dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation",
                        trust_remote_code=True)
print(f"Total items in validation: {len(dataset)}")

# Filter: >=2 passages, >=1 selected, valid answer, all passages <300 words
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

    # Need at least MIN_PASSAGES_PER_QUERY passages
    if len(passage_texts) < MIN_PASSAGES_PER_QUERY:
        continue

    # Need at least 1 selected passage
    if not is_selected or sum(is_selected) == 0:
        continue

    # All passages must be <= MAX_PASSAGE_WORDS
    word_counts = [count_words(p) for p in passage_texts]
    if any(wc > MAX_PASSAGE_WORDS for wc in word_counts):
        continue

    # Need a valid answer
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    else:
        continue

    # Build passage list with relevance labels
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

# Shuffle and take MAX_QUERIES
np.random.shuffle(queries)
queries = queries[:MAX_QUERIES]
N = len(queries)

# Statistics
n_passages_list = [q['n_passages'] for q in queries]
n_relevant_list = [q['n_relevant'] for q in queries]
total_passages = sum(n_passages_list)

print(f"\\nSelected {N} queries ({total_passages} total passages)")
print(f"Passages per query: mean={np.mean(n_passages_list):.1f}, "
      f"min={min(n_passages_list)}, max={max(n_passages_list)}")
print(f"Relevant per query: mean={np.mean(n_relevant_list):.1f}, "
      f"min={min(n_relevant_list)}, max={max(n_relevant_list)}")
print(f"Word counts: mean={np.mean([p['word_count'] for q in queries for p in q['passages']]):.0f}")

del dataset
gc.collect()\
""")))

# ========== Cell 5: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 5: Explain experimental conditions with concrete example

print("=" * 70)
print("EXPERIMENTAL CONDITIONS — RANKING-AWARE PRIMING")
print("=" * 70)

sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_ids = tokenizer(sf_str, add_special_tokens=False)['input_ids']
sf_tok_len = len(sf_ids)

print(f"\\nStatic fact prefix: '{STATIC_FACT}' ({sf_tok_len} tokens)")
print(f"\\nFor each query with N candidate passages, we:")
print(f"  1. Build bare cache for each passage (N forward passes)")
print(f"  2. Build primed cache for each passage (N forward passes)")
print(f"  3. Score answer with each cache (2N scoring passes)")
print(f"  4. Compute rankings under 6 conditions (no extra passes)")

conditions_explained = [
    ("1. bare",
     "Rank passages by bare NLL (lower = more relevant). Standard LM ranking baseline.",
     "Does the model rank relevant passages lower-NLL without any help?"),
    ("2. primed_1x",
     "Rank passages by primed NLL using static_fact_trunc prefix. "
     "Every passage gets the same prefix.",
     "Does uniform priming improve ranking? If relevant passages benefit more "
     "from priming than irrelevant ones, ranking improves even without query info."),
    ("3. primed_amp2x",
     "Rank passages by NLL from an amplified priming cache (2x delta at layers 0-15). "
     "Amplification boosts the contamination signal.",
     "Does stronger contamination amplify the differential effect?"),
    ("4. oracle_gated",
     "Use primed NLL only for 'hard' passages (bare NLL > per-query median). "
     "For 'easy' passages, use bare NLL. Threshold is per-query.",
     "Does selective priming (only hard passages) beat always-prime?"),
    ("5. delta_signal",
     "Rank by the DELTA itself: bare_nll - primed_nll. Passages that benefit "
     "more from priming are ranked higher. Ignores absolute NLL entirely.",
     "Is the priming response itself a relevance signal? If relevant passages "
     "respond more to priming, delta alone predicts relevance."),
    ("6. combined",
     "Rank by alpha * bare_nll + (1-alpha) * (-delta), with alpha tuned via grid search. "
     "Tests whether bare NLL and delta carry complementary information.",
     "Are NLL and delta complementary signals? Best alpha = 1 means delta is useless; "
     "best alpha = 0 means bare NLL is useless."),
]

for name, detail, test in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  {detail}")
    print(f"  Test: {test}")

print("\\n" + "=" * 70)
print("CONCRETE EXAMPLE")
print("=" * 70)
print("\\nQuery: 'What is the capital of France?'")
print("Passage A (relevant): 'Paris is the capital...' -> bare NLL=0.8, primed NLL=0.6")
print("Passage B (irrelevant): 'Berlin is in Germany...' -> bare NLL=1.5, primed NLL=1.4")
print()
print("  bare ranking:      A(0.8) > B(1.5)   => A ranked #1 (correct)")
print("  primed ranking:    A(0.6) > B(1.4)   => A ranked #1 (correct)")
print("  delta ranking:     A(0.2) > B(0.1)   => A ranked #1 (delta predicts relevance)")
print("  If priming helps A more than B, ranking is preserved/improved.")\
""")))

# ========== Cell 6: Helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 6: Helper functions for ranking experiment

def compute_delta(bare_cache, primed_cache, layers=None):
    \"\"\"Compute per-position value delta between primed and bare caches.
    Returns dict mapping layer_idx -> delta tensor.
    \"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    primed_cache = _ensure_dynamic_cache(primed_cache)
    n_layers = len(bare_cache)
    deltas = {}
    layer_range = layers if layers is not None else range(n_layers)
    for li in layer_range:
        v_bare = _get_cache_values(bare_cache, li)
        v_primed = _get_cache_values(primed_cache, li)
        deltas[li] = v_primed - v_bare
    return deltas


def apply_delta(bare_cache, deltas, scale=1.0):
    \"\"\"Apply scaled value delta to bare cache. Returns new DynamicCache.\"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    n_layers = len(bare_cache)
    new_cache = DynamicCache()
    for li in range(n_layers):
        k = _get_cache_keys(bare_cache, li)
        v = _get_cache_values(bare_cache, li).clone()
        if li in deltas:
            v = v + deltas[li] * scale
        new_cache.update(k, v, li)
    return new_cache


def score_all_passages_for_query(query_data, sf_prefix_ids, model, tokenizer, config):
    \"\"\"Score all passages for a single query under bare and primed conditions.

    Returns list of dicts, one per passage, with bare_nll, primed_nll, amp2x_nll.
    Memory: builds and scores one passage at a time, freeing between passages.
    \"\"\"
    query = query_data['query']
    answer = query_data['answer']
    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    n_layers = model.config.num_hidden_layers

    # Pre-tokenize the oracle prefix (for matched tokenization)
    oracle_prefix = SURROGATE_PREFIX_TEMPLATE.format(surrogate=query)

    passage_results = []

    for pidx, pinfo in enumerate(query_data['passages']):
        passage = pinfo['passage']

        # --- Matched tokenization ---
        document_text = DOCUMENT_TEMPLATE.format(document=passage)
        full_oracle_text = oracle_prefix + document_text

        full_oracle_enc = tokenizer(full_oracle_text, return_tensors="pt",
                                    add_special_tokens=True, padding=False, truncation=False)
        full_oracle_ids = full_oracle_enc['input_ids'].to(config.device)

        oracle_prefix_enc = tokenizer(oracle_prefix, return_tensors="pt",
                                      add_special_tokens=True, padding=False, truncation=False)
        oracle_prefix_len = oracle_prefix_enc['input_ids'].shape[1]

        bos_id = full_oracle_ids[:, :1]
        doc_ids = full_oracle_ids[:, oracle_prefix_len:]
        doc_len = doc_ids.shape[1]
        context_len = 1 + doc_len  # BOS + doc

        # === Build bare cache ===
        bare_ids = torch.cat([bos_id, doc_ids], dim=1)
        with torch.no_grad():
            bare_out = model(input_ids=bare_ids,
                             attention_mask=torch.ones_like(bare_ids),
                             use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
        del bare_out

        # === Build primed cache (static_fact prefix) ===
        primed_ids = torch.cat([bos_id, sf_prefix_ids, doc_ids], dim=1)
        sf_prefix_len = sf_prefix_ids.shape[1]
        with torch.no_grad():
            primed_out = model(input_ids=primed_ids,
                               attention_mask=torch.ones_like(primed_ids),
                               use_cache=True, return_dict=True)
        primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
        del primed_out

        # Truncate + RoPE correct
        primed_trunc = extract_and_truncate_cache_with_bos(primed_full, doc_len)
        correct_rope_positions_with_bos(primed_trunc, sf_prefix_len, model)
        del primed_full

        del bare_ids, primed_ids
        gc.collect()
        torch.cuda.empty_cache()

        # === Score primed_1x first (uses deepcopy) ===
        primed_for_score = build_hybrid_cache(bare_cache, primed_trunc)
        nll_primed = score_answer_with_cache(
            primed_for_score, context_len, query_prompt, answer_text, model, tokenizer, config)
        del primed_for_score

        # === Build amp2x cache (layers 0-15, 2x delta) ===
        deltas_0_15 = compute_delta(bare_cache, primed_trunc, layers=range(16))
        amp_deltas = {li: d * 2.0 for li, d in deltas_0_15.items()}
        amp_cache = apply_delta(bare_cache, amp_deltas)
        nll_amp2x = score_answer_with_cache(
            amp_cache, context_len, query_prompt, answer_text, model, tokenizer, config)
        del amp_cache, amp_deltas, deltas_0_15

        del primed_trunc

        # === Score bare LAST (mutates cache) ===
        nll_bare = score_answer_with_cache(
            bare_cache, context_len, query_prompt, answer_text, model, tokenizer, config)
        del bare_cache

        gc.collect()
        torch.cuda.empty_cache()

        passage_results.append({
            'passage_idx': pinfo['passage_idx'],
            'is_relevant': pinfo['is_relevant'],
            'word_count': pinfo['word_count'],
            'bare_nll': nll_bare,
            'primed_nll': nll_primed,
            'amp2x_nll': nll_amp2x,
            'delta': nll_bare - nll_primed,
            'delta_amp2x': nll_bare - nll_amp2x,
        })

    return passage_results


def compute_rankings_for_query(passage_results):
    \"\"\"Compute rankings under all 6 conditions for a single query.

    Returns dict with MRR, Hit@1, Hit@3 for each condition.
    \"\"\"
    n = len(passage_results)
    if n < 2:
        return None

    # Find relevant passage indices
    relevant_indices = [i for i, p in enumerate(passage_results) if p['is_relevant']]
    if not relevant_indices:
        return None

    bare_nlls = [p['bare_nll'] for p in passage_results]
    primed_nlls = [p['primed_nll'] for p in passage_results]
    amp2x_nlls = [p['amp2x_nll'] for p in passage_results]
    deltas = [p['delta'] for p in passage_results]

    # Per-query median of bare NLLs for gating
    bare_median = np.median(bare_nlls)

    # Gated: use primed if bare > median, else bare
    gated_nlls = []
    for i in range(n):
        if bare_nlls[i] > bare_median:
            gated_nlls.append(primed_nlls[i])
        else:
            gated_nlls.append(bare_nlls[i])

    # For each condition, build scores dict and compute metrics
    # For all conditions: lower score = ranked higher (like NLL)
    # Exception: delta_signal — higher delta = more relevant, so negate
    conditions = {
        'bare': {i: bare_nlls[i] for i in range(n)},
        'primed_1x': {i: primed_nlls[i] for i in range(n)},
        'primed_amp2x': {i: amp2x_nlls[i] for i in range(n)},
        'oracle_gated': {i: gated_nlls[i] for i in range(n)},
        'delta_signal': {i: -deltas[i] for i in range(n)},  # negate: higher delta = lower score = ranked higher
    }

    # Use first relevant index for compute_ranking_metrics
    rel_idx = relevant_indices[0]

    result = {'n_passages': n, 'n_relevant': len(relevant_indices)}

    for cond_name, scores in conditions.items():
        metrics = compute_ranking_metrics(scores, relevant_idx=rel_idx)
        result[cond_name] = metrics

    # Combined condition: sweep alpha in [0, 1]
    # score = alpha * bare_nll + (1-alpha) * (-delta)
    # = alpha * bare_nll - (1-alpha) * delta
    best_alpha = None
    best_mrr = -1
    for alpha in np.arange(0.0, 1.05, 0.05):
        combined_scores = {}
        for i in range(n):
            combined_scores[i] = alpha * bare_nlls[i] + (1 - alpha) * (-deltas[i])
        m = compute_ranking_metrics(combined_scores, relevant_idx=rel_idx)
        if m['mrr'] > best_mrr:
            best_mrr = m['mrr']
            best_alpha = alpha

    # Compute final combined with best alpha
    combined_scores = {i: best_alpha * bare_nlls[i] + (1 - best_alpha) * (-deltas[i])
                       for i in range(n)}
    result['combined'] = compute_ranking_metrics(combined_scores, relevant_idx=rel_idx)
    result['combined_best_alpha'] = best_alpha

    # Store per-passage data for differential analysis
    result['passage_data'] = passage_results

    return result


print("Helper functions defined:")
print("  compute_delta(bare, primed, layers=None) -> dict")
print("  apply_delta(bare, deltas, scale) -> cache")
print("  score_all_passages_for_query(query_data, sf_prefix_ids, ...) -> list")
print("  compute_rankings_for_query(passage_results) -> dict")\
""")))

# ========== Cell 7: Main loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main loop — iterate queries, score all passages, compute rankings

print("=" * 70)
print(f"MAIN EVALUATION ({N} queries)")
print("=" * 70)

# Pre-tokenize prefix
sf_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_prefix_enc = tokenizer(sf_prefix_str, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
sf_prefix_ids = sf_prefix_enc['input_ids'].to(config.device)
sf_prefix_len = sf_prefix_ids.shape[1]

print(f"Static fact prefix: '{STATIC_FACT}' ({sf_prefix_len} tokens)")

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

t_start = time.time()

for qidx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Queries"):
    query_data = queries[qidx]

    # Score all passages for this query
    passage_results = score_all_passages_for_query(
        query_data, sf_prefix_ids, model, tokenizer, config)

    # Compute rankings under all conditions
    rankings = compute_rankings_for_query(passage_results)

    if rankings is not None:
        rankings['query_idx'] = qidx
        rankings['query'] = query_data['query']

        # Compute query-doc overlap for each passage
        for pr in rankings['passage_data']:
            pidx_in_list = pr['passage_idx']
            passage_text = query_data['passages'][
                next(i for i, p in enumerate(query_data['passages'])
                     if p['passage_idx'] == pidx_in_list)
            ]['passage']
            pr['query_doc_overlap'] = compute_token_overlap(
                query_data['query'], passage_text, tokenizer)

        all_results.append(rankings)

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

# ========== Cell 8: Ranking analysis ==========
cells.append(make_cell("code", s("""\
# Cell 8: Ranking analysis — aggregate MRR/Hit@k, significance tests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("RANKING ANALYSIS")
print("=" * 70)

CONDITION_NAMES = ['bare', 'primed_1x', 'primed_amp2x', 'oracle_gated', 'delta_signal', 'combined']
N_VALID = len(all_results)
print(f"Valid queries: {N_VALID}")

# Aggregate metrics per condition
ranking_summary = {}
for cond in CONDITION_NAMES:
    mrrs = [r[cond]['mrr'] for r in all_results]
    hit1s = [r[cond]['hit_at_1'] for r in all_results]
    hit3s = [r[cond]['hit_at_3'] for r in all_results]
    ranks = [r[cond]['relevant_rank'] for r in all_results]
    ranking_summary[cond] = {
        'mrr_mean': float(np.mean(mrrs)),
        'mrr_std': float(np.std(mrrs)),
        'mrr_values': mrrs,
        'hit_at_1': float(np.mean(hit1s)),
        'hit_at_3': float(np.mean(hit3s)),
        'rank_mean': float(np.mean(ranks)),
        'rank_median': float(np.median(ranks)),
    }

# Print results table
print(f"\\n{'Condition':<18} {'MRR':>8} {'Hit@1':>8} {'Hit@3':>8} {'Rank':>8} {'p vs bare':>12} {'Sig':>5}")
print("-" * 75)

significance_results = {}
bare_mrrs = np.array(ranking_summary['bare']['mrr_values'])

for cond in CONDITION_NAMES:
    s_data = ranking_summary[cond]
    mrrs_arr = np.array(s_data['mrr_values'])

    if cond == 'bare':
        print(f"{cond:<18} {s_data['mrr_mean']:>8.3f} {s_data['hit_at_1']:>8.3f} "
              f"{s_data['hit_at_3']:>8.3f} {s_data['rank_mean']:>8.2f} {'--':>12} {'--':>5}")
    else:
        # Wilcoxon signed-rank test (paired, non-parametric)
        diff = mrrs_arr - bare_mrrs
        nonzero = np.sum(diff != 0)
        if nonzero > 10:
            stat, p_val = wilcoxon(mrrs_arr, bare_mrrs)
        else:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        delta_mrr = s_data['mrr_mean'] - ranking_summary['bare']['mrr_mean']
        print(f"{cond:<18} {s_data['mrr_mean']:>8.3f} {s_data['hit_at_1']:>8.3f} "
              f"{s_data['hit_at_3']:>8.3f} {s_data['rank_mean']:>8.2f} {p_val:>11.2e} {sig:>5}"
              f"  ΔMRR={delta_mrr:+.3f}")

        significance_results[f'{cond} vs bare'] = {
            'delta_mrr': float(delta_mrr),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
        }

# Best condition
best_cond = max(CONDITION_NAMES, key=lambda c: ranking_summary[c]['mrr_mean'])
print(f"\\nBest condition: {best_cond} (MRR={ranking_summary[best_cond]['mrr_mean']:.3f})")

# Alpha distribution for combined condition
alphas = [r['combined_best_alpha'] for r in all_results]
print(f"\\nCombined condition alpha: mean={np.mean(alphas):.2f}, "
      f"median={np.median(alphas):.2f}, std={np.std(alphas):.2f}")
print(f"  alpha=1 (bare only): {sum(1 for a in alphas if a >= 0.95)}/{N_VALID}")
print(f"  alpha=0 (delta only): {sum(1 for a in alphas if a <= 0.05)}/{N_VALID}")\
""")))

# ========== Cell 9: Differential effect analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Differential effect — relevant vs irrelevant deltas, AUC, ROC
from sklearn.metrics import roc_auc_score, roc_curve

print("=" * 70)
print("DIFFERENTIAL EFFECT ANALYSIS")
print("=" * 70)

# Collect all passage-level data
all_passages = []
for r in all_results:
    for p in r['passage_data']:
        all_passages.append(p)

is_relevant = np.array([p['is_relevant'] for p in all_passages])
bare_nlls = np.array([p['bare_nll'] for p in all_passages])
primed_nlls = np.array([p['primed_nll'] for p in all_passages])
amp2x_nlls = np.array([p['amp2x_nll'] for p in all_passages])
deltas = np.array([p['delta'] for p in all_passages])
deltas_amp2x = np.array([p['delta_amp2x'] for p in all_passages])
overlaps = np.array([p.get('query_doc_overlap', 0) for p in all_passages])

n_rel = int(np.sum(is_relevant))
n_irr = int(np.sum(~is_relevant))
print(f"\\nTotal passages: {len(all_passages)} (relevant: {n_rel}, irrelevant: {n_irr})")

# === Differential effect: delta for relevant vs irrelevant ===
delta_rel = deltas[is_relevant]
delta_irr = deltas[~is_relevant]
delta_amp_rel = deltas_amp2x[is_relevant]
delta_amp_irr = deltas_amp2x[~is_relevant]

print(f"\\n{'Metric':<25} {'Relevant':>12} {'Irrelevant':>12} {'Diff':>10} {'p':>12}")
print("-" * 75)

# Delta (primed_1x)
t_stat, p_val = stats.ttest_ind(delta_rel, delta_irr)
d = cohens_d(np.concatenate([delta_rel - np.mean(delta_irr), np.zeros(0)]))
print(f"{'Delta (1x)':25} {np.mean(delta_rel):>+12.4f} {np.mean(delta_irr):>+12.4f} "
      f"{np.mean(delta_rel) - np.mean(delta_irr):>+10.4f} {p_val:>11.2e}")

# Delta (amp2x)
t_stat2, p_val2 = stats.ttest_ind(delta_amp_rel, delta_amp_irr)
print(f"{'Delta (amp2x)':25} {np.mean(delta_amp_rel):>+12.4f} {np.mean(delta_amp_irr):>+12.4f} "
      f"{np.mean(delta_amp_rel) - np.mean(delta_amp_irr):>+10.4f} {p_val2:>11.2e}")

# Bare NLL
bare_rel = bare_nlls[is_relevant]
bare_irr = bare_nlls[~is_relevant]
t_stat3, p_val3 = stats.ttest_ind(bare_rel, bare_irr)
print(f"{'Bare NLL':25} {np.mean(bare_rel):>12.4f} {np.mean(bare_irr):>12.4f} "
      f"{np.mean(bare_rel) - np.mean(bare_irr):>+10.4f} {p_val3:>11.2e}")

# Overlap
overlap_rel = overlaps[is_relevant]
overlap_irr = overlaps[~is_relevant]
t_stat4, p_val4 = stats.ttest_ind(overlap_rel, overlap_irr)
print(f"{'Query-doc overlap':25} {np.mean(overlap_rel):>12.4f} {np.mean(overlap_irr):>12.4f} "
      f"{np.mean(overlap_rel) - np.mean(overlap_irr):>+10.4f} {p_val4:>11.2e}")

differential_results = {
    'delta_1x_relevant_mean': float(np.mean(delta_rel)),
    'delta_1x_irrelevant_mean': float(np.mean(delta_irr)),
    'delta_1x_diff': float(np.mean(delta_rel) - np.mean(delta_irr)),
    'delta_1x_p_value': float(p_val),
    'delta_amp2x_relevant_mean': float(np.mean(delta_amp_rel)),
    'delta_amp2x_irrelevant_mean': float(np.mean(delta_amp_irr)),
    'delta_amp2x_diff': float(np.mean(delta_amp_rel) - np.mean(delta_amp_irr)),
    'delta_amp2x_p_value': float(p_val2),
    'bare_nll_relevant_mean': float(np.mean(bare_rel)),
    'bare_nll_irrelevant_mean': float(np.mean(bare_irr)),
}

# === AUC: delta as relevance predictor ===
print(f"\\n{'='*70}")
print("DELTA AS RELEVANCE PREDICTOR (AUC)")
print(f"{'='*70}")

# For AUC: higher score should predict relevant=True
# bare NLL: relevant passages should have LOWER NLL -> negate for AUC
# delta: relevant passages should have HIGHER delta -> use as-is
# amp2x delta: same

predictors = {
    'bare_nll': -bare_nlls,       # negate: lower NLL = more relevant
    'primed_nll': -primed_nlls,
    'delta_1x': deltas,            # higher delta = more relevant (hypothesis)
    'delta_amp2x': deltas_amp2x,
    'overlap': overlaps,
}

auc_results = {}
roc_data = {}
print(f"\\n{'Predictor':<18} {'AUC':>8} {'Direction':>12}")
print("-" * 40)
for pred_name, pred_vals in predictors.items():
    try:
        auc = roc_auc_score(is_relevant.astype(int), pred_vals)
        fpr, tpr, thresholds = roc_curve(is_relevant.astype(int), pred_vals)
        roc_data[pred_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        direction = "correct" if auc > 0.5 else "INVERTED"
        print(f"{pred_name:<18} {auc:>8.3f} {direction:>12}")
        auc_results[pred_name] = float(auc)
    except Exception as e:
        print(f"{pred_name:<18} ERROR: {e}")
        auc_results[pred_name] = None

# Point-biserial correlation
print(f"\\n{'='*70}")
print("POINT-BISERIAL CORRELATIONS WITH RELEVANCE")
print(f"{'='*70}")

corr_results = {}
for pred_name, pred_vals in predictors.items():
    r, p = stats.pointbiserialr(is_relevant.astype(int), pred_vals)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {pred_name:<18} r={r:+.4f}  p={p:.2e}  {sig}")
    corr_results[pred_name] = {'r': float(r), 'p': float(p)}\
""")))

# ========== Cell 10: Stratified analysis + alpha sweep ==========
cells.append(make_cell("code", s("""\
# Cell 10: Stratified analysis and alpha sweep

print("=" * 70)
print("STRATIFIED ANALYSIS")
print("=" * 70)

# === By query difficulty ===
# Difficulty = mean bare NLL across all passages for the query
query_difficulties = []
for r in all_results:
    bare_mean = np.mean([p['bare_nll'] for p in r['passage_data']])
    query_difficulties.append(bare_mean)

query_difficulties = np.array(query_difficulties)
tercile_edges = np.percentile(query_difficulties, [33.3, 66.7])
difficulty_labels = np.array(['easy' if d <= tercile_edges[0] else
                               'hard' if d >= tercile_edges[1] else 'medium'
                               for d in query_difficulties])

print(f"\\nQuery difficulty terciles:")
print(f"  Easy (NLL <= {tercile_edges[0]:.2f}): n={np.sum(difficulty_labels=='easy')}")
print(f"  Medium: n={np.sum(difficulty_labels=='medium')}")
print(f"  Hard (NLL >= {tercile_edges[1]:.2f}): n={np.sum(difficulty_labels=='hard')}")

stratified_results = {}
for stratum in ['easy', 'medium', 'hard']:
    mask = difficulty_labels == stratum
    n_stratum = int(np.sum(mask))
    stratum_results_list = [r for r, m in zip(all_results, mask) if m]
    stratified_results[stratum] = {}
    print(f"\\n  {stratum} (n={n_stratum}):")
    for cond in CONDITION_NAMES:
        mrrs = [r[cond]['mrr'] for r in stratum_results_list]
        stratified_results[stratum][cond] = float(np.mean(mrrs))
    print(f"    {'Condition':<18} {'MRR':>8}")
    print(f"    {'-'*28}")
    for cond in CONDITION_NAMES:
        mrr = stratified_results[stratum][cond]
        delta_mrr = mrr - stratified_results[stratum]['bare']
        marker = " <-- best" if cond == max(CONDITION_NAMES, key=lambda c: stratified_results[stratum][c]) else ""
        print(f"    {cond:<18} {mrr:>8.3f}  (ΔMRR={delta_mrr:+.3f}){marker}")

# === By N candidates ===
print(f"\\n{'='*70}")
print("BY NUMBER OF CANDIDATES")
print(f"{'='*70}")

n_cands = np.array([r['n_passages'] for r in all_results])
for n_bin_lo, n_bin_hi in [(2, 5), (5, 8), (8, 11)]:
    mask = (n_cands >= n_bin_lo) & (n_cands < n_bin_hi)
    n_in_bin = int(np.sum(mask))
    if n_in_bin < 10:
        continue
    bin_results = [r for r, m in zip(all_results, mask) if m]
    print(f"\\n  {n_bin_lo}-{n_bin_hi-1} candidates (n={n_in_bin}):")
    for cond in ['bare', 'primed_1x', 'delta_signal', 'combined']:
        mrrs = [r[cond]['mrr'] for r in bin_results]
        print(f"    {cond:<18} MRR={np.mean(mrrs):.3f}")

# === Global alpha sweep (MRR) ===
print(f"\\n{'='*70}")
print("GLOBAL ALPHA SWEEP: score = alpha * bare_nll + (1-alpha) * (-delta)")
print(f"{'='*70}")

alpha_sweep_results = {}
for alpha in np.arange(0.0, 1.05, 0.05):
    alpha_val = round(alpha, 2)
    mrrs = []
    for r in all_results:
        n = len(r['passage_data'])
        scores = {}
        for i, p in enumerate(r['passage_data']):
            scores[i] = alpha_val * p['bare_nll'] + (1 - alpha_val) * (-p['delta'])
        rel_idx = next(i for i, p in enumerate(r['passage_data']) if p['is_relevant'])
        m = compute_ranking_metrics(scores, relevant_idx=rel_idx)
        mrrs.append(m['mrr'])
    alpha_sweep_results[alpha_val] = float(np.mean(mrrs))

best_global_alpha = max(alpha_sweep_results, key=alpha_sweep_results.get)
print(f"\\n  {'Alpha':>8} {'MRR':>8}")
print(f"  {'-'*18}")
for alpha_val in sorted(alpha_sweep_results.keys()):
    marker = " <-- best" if alpha_val == best_global_alpha else ""
    print(f"  {alpha_val:>8.2f} {alpha_sweep_results[alpha_val]:>8.3f}{marker}")

print(f"\\nBest global alpha: {best_global_alpha:.2f} "
      f"(MRR={alpha_sweep_results[best_global_alpha]:.3f})")
print(f"Bare-only (alpha=1.0): MRR={alpha_sweep_results.get(1.0, 0):.3f}")
print(f"Delta-only (alpha=0.0): MRR={alpha_sweep_results.get(0.0, 0):.3f}")\
""")))

# ========== Cell 11: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 11: Plots (6-panel figure)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

colors = {
    'bare': '#7f7f7f',
    'primed_1x': '#d62728',
    'primed_amp2x': '#ff7f0e',
    'oracle_gated': '#2ca02c',
    'delta_signal': '#1f77b4',
    'combined': '#9467bd',
}

# --- Plot 1: MRR comparison bar chart ---
ax = axes[0, 0]
conds = CONDITION_NAMES
mrrs = [ranking_summary[c]['mrr_mean'] for c in conds]
bar_colors = [colors[c] for c in conds]
bars = ax.bar(range(len(conds)), mrrs, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds, rotation=30, ha='right', fontsize=8)
for i, (c, m) in enumerate(zip(conds, mrrs)):
    ax.text(i, m + 0.005, f"{m:.3f}", ha='center', fontsize=8)
ax.set_ylabel("MRR")
ax.set_title("MRR by Condition")
ax.set_ylim(0, max(mrrs) * 1.15)

# --- Plot 2: Delta distributions — relevant vs irrelevant ---
ax = axes[0, 1]
bins = np.linspace(min(deltas.min(), -1), max(deltas.max(), 1), 50)
ax.hist(delta_rel, bins=bins, alpha=0.6, color='green', label=f'Relevant (n={n_rel})', density=True)
ax.hist(delta_irr, bins=bins, alpha=0.6, color='red', label=f'Irrelevant (n={n_irr})', density=True)
ax.axvline(np.mean(delta_rel), color='green', linestyle='--', linewidth=2)
ax.axvline(np.mean(delta_irr), color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Delta (bare - primed NLL)")
ax.set_ylabel("Density")
ax.set_title("Delta Distribution: Relevant vs Irrelevant")
ax.legend(fontsize=8)

# --- Plot 3: Query difficulty vs ΔMRR ---
ax = axes[0, 2]
delta_mrrs_primed = np.array([r['primed_1x']['mrr'] - r['bare']['mrr'] for r in all_results])
ax.scatter(query_difficulties, delta_mrrs_primed, alpha=0.3, s=15, color=colors['primed_1x'],
           label='primed_1x')
# Trend line via binning
n_trend_bins = 8
trend_edges = np.percentile(query_difficulties, np.linspace(0, 100, n_trend_bins + 1))
for k in range(n_trend_bins):
    mask_k = (query_difficulties >= trend_edges[k]) & (query_difficulties < trend_edges[k+1])
    if np.sum(mask_k) > 5:
        center = (trend_edges[k] + trend_edges[k+1]) / 2
        ax.scatter(center, np.mean(delta_mrrs_primed[mask_k]), s=80, color=colors['primed_1x'],
                  edgecolor='black', linewidth=1, zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Query Difficulty (mean bare NLL)")
ax.set_ylabel("ΔMRR (primed - bare)")
ax.set_title("Query Difficulty vs Ranking Improvement")
ax.legend(fontsize=8)

# --- Plot 4: ROC curves ---
ax = axes[1, 0]
roc_colors = {'bare_nll': '#7f7f7f', 'delta_1x': '#1f77b4', 'delta_amp2x': '#ff7f0e',
              'overlap': '#2ca02c'}
for pred_name in ['bare_nll', 'delta_1x', 'delta_amp2x', 'overlap']:
    if pred_name in roc_data:
        fpr = roc_data[pred_name]['fpr']
        tpr = roc_data[pred_name]['tpr']
        auc_val = auc_results.get(pred_name, 0)
        ax.plot(fpr, tpr, color=roc_colors.get(pred_name, 'gray'),
                label=f"{pred_name} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Delta as Relevance Predictor")
ax.legend(fontsize=8)

# --- Plot 5: Heatmap — condition × difficulty stratum MRR ---
ax = axes[1, 1]
strata = ['easy', 'medium', 'hard']
heatmap_data = np.zeros((len(CONDITION_NAMES), len(strata)))
for i, cond in enumerate(CONDITION_NAMES):
    for j, stratum in enumerate(strata):
        heatmap_data[i, j] = stratified_results.get(stratum, {}).get(cond, 0)
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(strata)))
ax.set_xticklabels(strata)
ax.set_yticks(range(len(CONDITION_NAMES)))
ax.set_yticklabels(CONDITION_NAMES, fontsize=8)
for i in range(len(CONDITION_NAMES)):
    for j in range(len(strata)):
        ax.text(j, i, f"{heatmap_data[i,j]:.3f}", ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, label="MRR")
ax.set_title("MRR by Condition × Difficulty")

# --- Plot 6: Per-query MRR change distribution ---
ax = axes[1, 2]
for cond in ['primed_1x', 'delta_signal', 'combined']:
    delta_mrrs = [r[cond]['mrr'] - r['bare']['mrr'] for r in all_results]
    ax.hist(delta_mrrs, bins=30, alpha=0.5, label=cond, color=colors[cond])
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("ΔMRR (condition - bare)")
ax.set_ylabel("Count")
ax.set_title("Per-Query MRR Change Distribution")
ax.legend(fontsize=8)

plt.suptitle('Exp 14: Ranking-Aware Priming', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 12: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 12: Save results JSON
final = {
    'experiment': 'exp14_ranking_aware_priming',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_queries': N,
        'n_valid': N_VALID,
        'max_passage_words': MAX_PASSAGE_WORDS,
        'min_passages_per_query': MIN_PASSAGES_PER_QUERY,
        'static_fact': STATIC_FACT,
        'dataset': 'MS MARCO v1.1 validation',
    },
    'condition_names': CONDITION_NAMES,
    'ranking_summary': {c: {k: v for k, v in d.items() if k != 'mrr_values'}
                        for c, d in ranking_summary.items()},
    'significance_results': significance_results,
    'differential_effect': differential_results,
    'auc_results': auc_results,
    'correlation_results': corr_results,
    'stratified_results': stratified_results,
    'alpha_sweep': alpha_sweep_results,
    'best_global_alpha': float(best_global_alpha),
    'per_query_results': [{k: v for k, v in r.items() if k != 'passage_data'}
                          for r in all_results],
    'per_passage_summary': {
        'n_total': len(all_passages),
        'n_relevant': n_rel,
        'n_irrelevant': n_irr,
        'mean_delta_relevant': float(np.mean(delta_rel)),
        'mean_delta_irrelevant': float(np.mean(delta_irr)),
    },
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
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

output_path = "/home/jupyter/research/directed_kvcache_v2/14_ranking_aware_priming.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
