#!/usr/bin/env python3
"""Build script for 08_mechanism_and_amplification.ipynb"""

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
    """Convert multi-line string to notebook source lines (preserving newlines)."""
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
# Exp 08: Mechanism Isolation + Prefix Amplification

## Motivation

Exp 06 found that LLM surrogates improve KV cache quality via value contamination (truncated
prefix alters document value vectors) and suffix attention (separator provides new attention
targets). But which component of the KV cache carries the signal — keys, values, or both?
And does repeating the prefix amplify the effect?

## Core Questions

1. **Key vs Value isolation**: Is the priming effect carried by keys, values, or both?
2. **Prefix amplification**: Does repeating the surrogate prefix K times amplify the effect?

## Self-contained. N=1000 samples (smaller, focused probe).

## 10 Conditions

| # | Condition | Construction | Tests |
|---|-----------|-------------|-------|
| 1 | Bare | `[BOS][doc]` | Baseline |
| 2 | LLM-keyword-trunc | Standard truncated + RoPE | Reference |
| 3 | LLM-keyword-suffix | Suffix mode | Reference |
| 4 | LLM-keyword+sep | Trunc + suffix | Best method reference |
| 5 | Primed-values-only | Keys from bare, values from LLM-trunc | Key vs value |
| 6 | Primed-keys-only | Values from bare, keys from LLM-trunc | Key vs value |
| 7 | Prefix-1x | `[BOS][kw\\n][doc]` → truncate | Baseline repetition |
| 8 | Prefix-3x | `[BOS][kw\\n kw\\n kw\\n][doc]` → truncate | 3× amplification |
| 9 | Prefix-5x | `[BOS][kw\\n kw\\n kw\\n kw\\n kw\\n][doc]` → truncate | 5× amplification |
| 10 | Separator-only | `[BOS][doc][\\n\\nRelated question: ]` | Control |

## 6 Primary Comparisons (Bonferroni alpha = 0.0083)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | Primed-values-only vs Bare | Is the effect in values? |
| C2 | Primed-keys-only vs Bare | Is the effect in keys? |
| C3 | Primed-values-only vs LLM-keyword-trunc | How much do values capture? |
| C4 | Prefix-3x vs Prefix-1x | Does 3× amplify? |
| C5 | Prefix-5x vs Prefix-1x | Does 5× amplify? |
| C6 | Prefix-5x vs Prefix-3x | Diminishing returns? |""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("results/exp08")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_DIR = RESULTS_DIR / "surrogates"
SURROGATES_DIR.mkdir(parents=True, exist_ok=True)

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# ========== Cell 3: Imports + config ==========
cells.append(make_cell("code", s("""\
# Cell 3: Imports + config + templates + helpers
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    build_kv_cache,
    build_suffix_kv_cache,
    score_answer_with_cache,
    deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    build_hybrid_cache,
    _get_cache_keys,
    _get_cache_values,
)
from lib.data import load_ms_marco, load_evaluation_samples
from lib.analysis import cohens_d
from lib.surrogate import generate_all_5_surrogates, TOP_5_SURROGATE_TEMPLATES
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    min_passage_words=20,
    max_passage_words=500,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"
SUFFIX_SEPARATOR = "\\n\\nRelated question: "
CHECKPOINT_EVERY = 50
N_COMPARISONS = 6
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
N_EVAL = 1000  # Smaller focused probe

print("Config ready")
print(f"  num_samples pool: {config.num_samples}")
print(f"  eval samples: {N_EVAL}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: 10")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (1000 samples)
dataset = load_ms_marco(config)

np.random.seed(SEED)
all_samples = load_evaluation_samples(dataset, config, require_answer=True)

samples = all_samples[:N_EVAL]
N = len(samples)
print(f"Loaded {len(all_samples)} candidates, using first {N} for evaluation")
print(f"Example passage ({len(samples[0]['passage'].split())} words): {samples[0]['passage'][:100]}...")
print(f"Example query: {samples[0]['query']}")
print(f"Example answer: {samples[0]['answer']}")\
""")))

# ========== Cell 5: Generate LLM keyword surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 5: Generate LLM keyword surrogates (only keyword_query template needed)
# Using generate_all_5_surrogates for consistency but only using keyword_query

print("=" * 70)
print("PHASE 1: LLM SURROGATE GENERATION (keyword only)")
print("=" * 70)

surrogates_path = SURROGATES_DIR / "keyword_surrogates.json"

if surrogates_path.exists():
    with open(surrogates_path, 'r') as f:
        surrogates_data = json.load(f)
    keyword_surrogates = surrogates_data['surrogates']
    print(f"Loaded {len(keyword_surrogates)} keyword surrogates from cache")
else:
    keyword_surrogates = []

start_idx_gen = len(keyword_surrogates)
if start_idx_gen < N:
    print(f"Generating keyword surrogates for samples {start_idx_gen} to {N-1}...")
    t_start = time.time()
    for idx in tqdm(range(start_idx_gen, N), initial=start_idx_gen, total=N,
                     desc="Keyword surrogates"):
        passage = samples[idx]['passage']
        try:
            s5 = generate_all_5_surrogates(passage, model, tokenizer, config)
            kw = s5.get('keyword_query', '')
        except Exception as e:
            print(f"  WARNING: Generation failed for sample {idx}: {e}")
            kw = ""
        keyword_surrogates.append(kw)

        if (idx + 1) % 100 == 0 or idx == N - 1:
            with open(surrogates_path, 'w') as f:
                json.dump({'surrogates': keyword_surrogates}, f)
            elapsed = time.time() - t_start
            rate = (idx - start_idx_gen + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"  Saved {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

    with open(surrogates_path, 'w') as f:
        json.dump({'surrogates': keyword_surrogates}, f)
    print(f"Keyword surrogates complete: {len(keyword_surrogates)} samples")
else:
    print(f"All keyword surrogates already cached ({len(keyword_surrogates)} samples)")

n_empty = sum(1 for s in keyword_surrogates if not s.strip())
print(f"Empty surrogates: {n_empty}/{N}")
print(f"Example: '{keyword_surrogates[0]}'")\
""")))

# ========== Cell 6: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 6: Condition explanation with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

ex_kw = keyword_surrogates[0]

conditions_explained = [
    ("1. Bare",
     "[BOS][doc]",
     "No prefix — baseline"),
    ("2. LLM-keyword-trunc",
     "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     f"Standard truncated prefix: '{ex_kw}'"),
    ("3. LLM-keyword-suffix",
     "[BOS][doc][\\\\n\\\\nRelated question: llm_kw]",
     f"Suffix mode: '{ex_kw}'"),
    ("4. LLM-keyword+sep",
     "[BOS][llm_kw\\\\n][doc][\\\\n\\\\nRelated question: ] (prefix+suffix)",
     "Stacking: truncated prefix + suffix separator"),
    ("5. Primed-values-only",
     "Keys from bare cache, values from LLM-trunc cache",
     "Tests: is the effect carried by values?"),
    ("6. Primed-keys-only",
     "Values from bare cache, keys from LLM-trunc cache",
     "Tests: is the effect carried by keys?"),
    ("7. Prefix-1x",
     "[BOS][kw\\\\n][doc] → truncate (same as #2)",
     "1× prefix — baseline for repetition"),
    ("8. Prefix-3x",
     "[BOS][kw\\\\n kw\\\\n kw\\\\n][doc] → truncate + RoPE(3×offset)",
     "3× repeated prefix — amplification test"),
    ("9. Prefix-5x",
     "[BOS][kw\\\\n kw\\\\n kw\\\\n kw\\\\n kw\\\\n][doc] → truncate + RoPE(5×offset)",
     "5× repeated prefix — maximum amplification"),
    ("10. Separator-only",
     "[BOS][doc][\\\\n\\\\nRelated question: ]",
     "Suffix framing only — structural control"),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")\
""")))

# ========== Cell 7: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main eval loop — 10 conditions × 1000 samples
print("=" * 70)
print("PHASE 2: MAIN EVALUATION (10 conditions × 1000 samples)")
print("=" * 70)

CONDITION_NAMES = [
    'bare', 'llm_kw_trunc', 'llm_kw_suffix', 'llm_kw_sep',
    'primed_values_only', 'primed_keys_only',
    'prefix_1x', 'prefix_3x', 'prefix_5x',
    'separator_only',
]

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in samples]
    if ckpt_queries == current_queries:
        results = ckpt['results']
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{N}")
    else:
        print("Checkpoint sample mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

print(f"Evaluating samples {start_idx} to {N-1}")
print(f"Conditions: {len(CONDITION_NAMES)}")

t_start = time.time()

for idx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Evaluating"):
    sample = samples[idx]
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']

    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)
    llm_kw_text = keyword_surrogates[idx]

    # --- Matched tokenization ---
    oracle_prefix = SURROGATE_PREFIX_TEMPLATE.format(surrogate=query)
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

    # Prefix IDs for LLM keyword (used for all truncated conditions)
    kw_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=llm_kw_text)
    kw_prefix_enc = tokenizer(kw_prefix_str, return_tensors="pt",
                              add_special_tokens=False, padding=False, truncation=False)
    kw_prefix_ids = kw_prefix_enc['input_ids'].to(config.device)
    kw_prefix_token_len_1x = kw_prefix_ids.shape[1]  # without BOS

    # === Condition 1: BARE ===
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = bare_out.past_key_values
    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_cache), bare_ids.shape[1],
        query_prompt, answer_text, model, tokenizer, config)

    # === Condition 2: LLM-KEYWORD-TRUNC (also prefix-1x) ===
    full_1x_ids = torch.cat([bos_id, kw_prefix_ids, doc_ids], dim=1)
    prefix_token_len_1x = 1 + kw_prefix_token_len_1x

    with torch.no_grad():
        out_1x = model(input_ids=full_1x_ids,
                       attention_mask=torch.ones_like(full_1x_ids),
                       use_cache=True, return_dict=True)
    trunc_cache_1x = extract_and_truncate_cache_with_bos(out_1x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_cache_1x, prefix_token_len_1x - 1, model)
    nll_llm_kw_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache_1x), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del out_1x

    # prefix_1x is same as llm_kw_trunc
    nll_prefix_1x = nll_llm_kw_trunc

    # === Condition 3: LLM-KEYWORD-SUFFIX ===
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, llm_kw_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_llm_kw_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # === Condition 4: LLM-KEYWORD+SEP (stacking) ===
    suffix_enc = tokenizer(SUFFIX_SEPARATOR, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
    suffix_ids = suffix_enc['input_ids'].to(config.device)
    suffix_len_tok = suffix_ids.shape[1]
    cache_len_before_suffix = 1 + doc_len

    suffix_position_ids = torch.arange(
        cache_len_before_suffix, cache_len_before_suffix + suffix_len_tok,
        device=config.device
    ).unsqueeze(0)

    with torch.no_grad():
        suffix_out = model(
            input_ids=suffix_ids,
            attention_mask=torch.ones(1, cache_len_before_suffix + suffix_len_tok,
                                      device=config.device, dtype=torch.long),
            position_ids=suffix_position_ids,
            past_key_values=deepcopy_cache(trunc_cache_1x),
            use_cache=True,
            return_dict=True,
        )
    combo_cache = suffix_out.past_key_values
    combo_len = cache_len_before_suffix + suffix_len_tok
    nll_llm_kw_sep = score_answer_with_cache(
        deepcopy_cache(combo_cache), combo_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suffix_out, combo_cache

    # === Condition 5: PRIMED-VALUES-ONLY ===
    # Keys from bare cache, values from LLM-trunc cache
    hybrid_values = build_hybrid_cache(
        keys_source=bare_cache,
        values_source=trunc_cache_1x,
    )
    nll_primed_values = score_answer_with_cache(
        deepcopy_cache(hybrid_values), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del hybrid_values

    # === Condition 6: PRIMED-KEYS-ONLY ===
    # Values from bare cache, keys from LLM-trunc cache
    hybrid_keys = build_hybrid_cache(
        keys_source=trunc_cache_1x,
        values_source=bare_cache,
    )
    nll_primed_keys = score_answer_with_cache(
        deepcopy_cache(hybrid_keys), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del hybrid_keys

    del bare_cache, trunc_cache_1x  # Free memory

    # === Condition 8: PREFIX-3x ===
    # [BOS][kw\\n kw\\n kw\\n][doc] → truncate + RoPE
    prefix_3x_ids = kw_prefix_ids.repeat(1, 3)
    full_3x_ids = torch.cat([bos_id, prefix_3x_ids, doc_ids], dim=1)
    prefix_token_len_3x = 1 + prefix_3x_ids.shape[1]

    with torch.no_grad():
        out_3x = model(input_ids=full_3x_ids,
                       attention_mask=torch.ones_like(full_3x_ids),
                       use_cache=True, return_dict=True)
    trunc_cache_3x = extract_and_truncate_cache_with_bos(out_3x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_cache_3x, prefix_token_len_3x - 1, model)
    nll_prefix_3x = score_answer_with_cache(
        deepcopy_cache(trunc_cache_3x), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del out_3x, trunc_cache_3x

    # === Condition 9: PREFIX-5x ===
    prefix_5x_ids = kw_prefix_ids.repeat(1, 5)
    full_5x_ids = torch.cat([bos_id, prefix_5x_ids, doc_ids], dim=1)
    prefix_token_len_5x = 1 + prefix_5x_ids.shape[1]

    with torch.no_grad():
        out_5x = model(input_ids=full_5x_ids,
                       attention_mask=torch.ones_like(full_5x_ids),
                       use_cache=True, return_dict=True)
    trunc_cache_5x = extract_and_truncate_cache_with_bos(out_5x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_cache_5x, prefix_token_len_5x - 1, model)
    nll_prefix_5x = score_answer_with_cache(
        deepcopy_cache(trunc_cache_5x), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del out_5x, trunc_cache_5x

    # === Condition 10: SEPARATOR-ONLY ===
    sep_len, sep_cache = build_suffix_kv_cache(
        passage, "", model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_separator = score_answer_with_cache(
        deepcopy_cache(sep_cache), sep_len,
        query_prompt, answer_text, model, tokenizer, config)
    del sep_cache

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len': doc_len,
        'passage_word_count': len(passage.split()),
        'bare': nll_bare,
        'llm_kw_trunc': nll_llm_kw_trunc,
        'llm_kw_suffix': nll_llm_kw_suffix,
        'llm_kw_sep': nll_llm_kw_sep,
        'primed_values_only': nll_primed_values,
        'primed_keys_only': nll_primed_keys,
        'prefix_1x': nll_prefix_1x,
        'prefix_3x': nll_prefix_3x,
        'prefix_5x': nll_prefix_5x,
        'separator_only': nll_separator,
    }
    results.append(result)

    torch.cuda.empty_cache()

    if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == N - 1:
        ckpt_data = {
            'results': results,
            'sample_queries': [s['query'] for s in samples],
            'completed': len(results),
            'total': N,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        rate = (idx - start_idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (N - idx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nEvaluation complete: {len(results)} samples in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 8: Primary analysis ==========
cells.append(make_cell("code", s("""\
# Cell 8: Primary analysis — 6 comparisons + NLL summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — MECHANISM ISOLATION + AMPLIFICATION")
print("=" * 70)

# Extract arrays and filter zero NLLs
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

valid = np.ones(len(results), dtype=bool)
for cname in CONDITION_NAMES:
    valid &= (cond_arrays[cname] != 0)
n_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total: {len(results)}, Valid: {n_valid}, Excluded: {n_excluded}")

c = {}
for cname in CONDITION_NAMES:
    c[cname] = cond_arrays[cname][valid]

# NLL summary table
print(f"\\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10}")
print("-" * 60)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        d_str = "—"
    else:
        d = cohens_d(c['bare'] - c[cname])
        d_str = f"{d:+.3f}"
    print(f"{cname:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d_str:>10}")

# 6 primary comparisons
print(f"\\n{'='*85}")
print(f"6 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*85}")

comparisons = [
    ('C1: Values-only vs Bare',
     c['bare'] - c['primed_values_only'],
     'Is the effect in values?'),
    ('C2: Keys-only vs Bare',
     c['bare'] - c['primed_keys_only'],
     'Is the effect in keys?'),
    ('C3: Values-only vs LLM-trunc',
     c['llm_kw_trunc'] - c['primed_values_only'],
     'How much do values capture?'),
    ('C4: Prefix-3x vs Prefix-1x',
     c['prefix_1x'] - c['prefix_3x'],
     'Does 3x amplify?'),
    ('C5: Prefix-5x vs Prefix-1x',
     c['prefix_1x'] - c['prefix_5x'],
     'Does 5x amplify?'),
    ('C6: Prefix-5x vs Prefix-3x',
     c['prefix_3x'] - c['prefix_5x'],
     'Diminishing returns?'),
]

print(f"\\n{'Comparison':<35} {'Mean Δ':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 90)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<35} {np.mean(delta):>8.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        'question': question,
    }

# All vs Bare
print(f"\\n{'='*85}")
print("ALL CONDITIONS vs BARE")
print(f"{'='*85}")
print(f"\\n{'Condition':<25} {'d vs Bare':>10} {'Win%':>7} {'p':>12}")
print("-" * 60)
all_vs_bare = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname:<25} {d:>10.3f} {win:>6.1f}% {p_val:>11.2e} {sig:>5}")
    all_vs_bare[cname] = {'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val)}\
""")))

# ========== Cell 9: Mechanism isolation analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Mechanism isolation deep-dive + amplification analysis

print("=" * 70)
print("MECHANISM ISOLATION DEEP-DIVE")
print("=" * 70)

d_values_only = cohens_d(c['bare'] - c['primed_values_only'])
d_keys_only = cohens_d(c['bare'] - c['primed_keys_only'])
d_full_trunc = cohens_d(c['bare'] - c['llm_kw_trunc'])

print(f"\\nKey/Value Decomposition:")
print(f"  Full LLM-trunc (keys+values): d = {d_full_trunc:+.3f}")
print(f"  Values-only (bare keys):      d = {d_values_only:+.3f} ({d_values_only/d_full_trunc*100:.0f}% of full)")
print(f"  Keys-only (bare values):       d = {d_keys_only:+.3f} ({d_keys_only/d_full_trunc*100:.0f}% of full)")
print(f"  Sum of parts:                  d = {d_values_only + d_keys_only:+.3f} (vs full: {d_full_trunc:+.3f})")

if d_values_only > d_keys_only and d_values_only > 0:
    if d_keys_only < 0.05:
        print(f"\\n→ VALUES carry the priming signal. Keys contribute negligibly.")
    else:
        print(f"\\n→ BOTH contribute, but values carry more ({d_values_only/d_full_trunc*100:.0f}% vs {d_keys_only/d_full_trunc*100:.0f}%).")
elif d_keys_only > d_values_only and d_keys_only > 0:
    print(f"\\n→ KEYS carry the priming signal (unexpected — RoPE correction may be key).")
else:
    print(f"\\n→ NEITHER component alone matches the combined effect — synergy required.")

# Amplification curve
print(f"\\n{'='*70}")
print("PREFIX AMPLIFICATION CURVE")
print(f"{'='*70}")

d_1x = cohens_d(c['bare'] - c['prefix_1x'])
d_3x = cohens_d(c['bare'] - c['prefix_3x'])
d_5x = cohens_d(c['bare'] - c['prefix_5x'])

print(f"\\nAmplification curve (d vs bare):")
print(f"  1× prefix: d = {d_1x:+.3f}")
print(f"  3× prefix: d = {d_3x:+.3f} (Δd from 1x = {d_3x - d_1x:+.3f})")
print(f"  5× prefix: d = {d_5x:+.3f} (Δd from 1x = {d_5x - d_1x:+.3f})")
print(f"  3x→5x marginal: Δd = {d_5x - d_3x:+.3f}")

if d_5x > d_3x > d_1x:
    if (d_5x - d_3x) < (d_3x - d_1x) * 0.5:
        print(f"\\n→ Amplification works but with DIMINISHING RETURNS.")
    else:
        print(f"\\n→ Amplification works with roughly LINEAR scaling.")
elif d_3x > d_1x and d_5x <= d_3x:
    print(f"\\n→ Amplification helps up to 3× but SATURATES at 5×.")
elif d_3x <= d_1x:
    print(f"\\n→ No amplification effect — repetition does NOT help.")

# Hardness quintile for mechanism conditions
print(f"\\n{'='*70}")
print("HARDNESS QUINTILE BREAKDOWN")
print(f"{'='*70}")

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

mechanism_conds = ['llm_kw_trunc', 'primed_values_only', 'primed_keys_only',
                   'prefix_1x', 'prefix_3x', 'prefix_5x', 'llm_kw_sep', 'separator_only']

header = f"{'Condition':<25}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (25 + 14 * 6))

hardness_breakdown = {}
for cname in mechanism_conds:
    row = f"{cname:<25}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            d = cohens_d(delta)
            row += f"{d:>+14.3f}"
            quintile_ds.append(float(d))
    d_all = cohens_d(bare_valid - c[cname])
    row += f"{d_all:>+14.3f}"
    print(row)
    hardness_breakdown[cname] = {'quintile_ds': quintile_ds, 'overall_d': float(d_all)}\
""")))

# ========== Cell 10: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 10: Plots

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Plot 1: All conditions bar chart ---
ax = axes[0, 0]
cnames_sorted = sorted(
    [cn for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda cn: cohens_d(c['bare'] - c[cn]),
    reverse=True
)
ds_bar = [cohens_d(c['bare'] - c[cn]) for cn in cnames_sorted]
color_map = {
    'llm_kw_trunc': 'forestgreen', 'llm_kw_suffix': 'limegreen',
    'llm_kw_sep': 'gold', 'primed_values_only': 'steelblue',
    'primed_keys_only': 'cornflowerblue', 'prefix_1x': 'mediumpurple',
    'prefix_3x': 'darkorchid', 'prefix_5x': 'purple',
    'separator_only': 'salmon',
}
colors_bar = [color_map.get(cn, 'lightgray') for cn in cnames_sorted]
ax.barh(range(len(cnames_sorted)), ds_bar, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(cnames_sorted)))
ax.set_yticklabels(cnames_sorted, fontsize=8)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare')
ax.invert_yaxis()

# --- Plot 2: Key/Value decomposition ---
ax = axes[0, 1]
decomp_labels = ['Full\\n(K+V)', 'Values\\nonly', 'Keys\\nonly', 'Sum\\n(V+K)']
decomp_vals = [d_full_trunc, d_values_only, d_keys_only, d_values_only + d_keys_only]
decomp_colors = ['forestgreen', 'steelblue', 'cornflowerblue', 'lightgray']
bars = ax.bar(range(len(decomp_labels)), decomp_vals, color=decomp_colors,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(decomp_labels)))
ax.set_xticklabels(decomp_labels, fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Key vs Value Decomposition')
for i, v in enumerate(decomp_vals):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=9)

# --- Plot 3: Amplification curve ---
ax = axes[1, 0]
reps = [1, 3, 5]
amp_ds = [d_1x, d_3x, d_5x]
ax.plot(reps, amp_ds, 'o-', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Number of Prefix Repetitions')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Prefix Amplification Curve')
ax.set_xticks(reps)
ax.grid(True, alpha=0.3)
for x, y in zip(reps, amp_ds):
    ax.annotate(f"d={y:+.3f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

# --- Plot 4: Hardness × mechanism heatmap ---
ax = axes[1, 1]
hm_data = []
for cname in mechanism_conds:
    row = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row.append(0)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            row.append(cohens_d(delta))
    hm_data.append(row)
hm_data = np.array(hm_data)
im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.5, vmax=0.7, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=7)
ax.set_yticks(range(len(mechanism_conds)))
ax.set_yticklabels(mechanism_conds, fontsize=7)
for i in range(len(mechanism_conds)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness × Mechanism')

plt.suptitle('Exp 08: Mechanism Isolation + Prefix Amplification', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 11: Save results JSON ==========
cells.append(make_cell("code", s("""\
# Cell 11: Save comprehensive results JSON

final = {
    'experiment': 'exp08_mechanism_and_amplification',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_eval': N,
        'n_valid': n_valid,
        'n_excluded': n_excluded,
        'min_passage_words': config.min_passage_words,
        'max_passage_words': config.max_passage_words,
        'n_conditions': len(CONDITION_NAMES),
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': {
        cname: {
            'mean': float(np.mean(c[cname])),
            'std': float(np.std(c[cname])),
            'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cname])) if cname != 'bare' else 0.0,
        }
        for cname in CONDITION_NAMES
    },
    'mechanism_decomposition': {
        'd_full_trunc': float(d_full_trunc),
        'd_values_only': float(d_values_only),
        'd_keys_only': float(d_keys_only),
        'd_sum_parts': float(d_values_only + d_keys_only),
        'values_fraction': float(d_values_only / d_full_trunc) if d_full_trunc != 0 else 0,
        'keys_fraction': float(d_keys_only / d_full_trunc) if d_full_trunc != 0 else 0,
    },
    'amplification': {
        'd_1x': float(d_1x),
        'd_3x': float(d_3x),
        'd_5x': float(d_5x),
        'delta_1x_to_3x': float(d_3x - d_1x),
        'delta_3x_to_5x': float(d_5x - d_3x),
    },
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'hardness_breakdown': hardness_breakdown,
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 12: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 12: GPU cleanup
import gc

print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9

del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()
gc.collect()

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Cleanup complete.")\
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

output_path = "/home/jupyter/research/directed_kvcache_v2/08_mechanism_and_amplification.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
