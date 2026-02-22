#!/usr/bin/env python3
"""Build script for 07_static_surrogates_and_routing.ipynb"""

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
# Exp 07: Static Surrogates, Dual-Mode Priming, Intent Routing

## Motivation

Exp 06 revealed that LLM surrogates significantly improve KV cache quality (d=0.23-0.30 vs
bare), but the mechanism is NOT token overlap (r=-0.024). Separator-only (a suffix with no
content) nearly matches LLM-keyword (d=0.231 vs 0.234). The hardness gradient is massive:
LLM-keyword+sep goes from d=-0.226 (Q1 easiest) to d=+0.630 (Q5 hardest).

## Core Question

Can cheap static surrogates match LLM performance? Does the optimal strategy depend on query
intent, answer length, or passage length?

## Self-contained

This experiment generates its own LLM surrogates and computes its own bare NLLs. No reuse of
results across experiments.

## 21 Conditions

| # | Condition | Type |
|---|-----------|------|
| 1 | Bare | Baseline |
| 2 | Random-truncated | Control |
| 3 | Separator-only | Control |
| 4-8 | Static-{definitional,procedural,quantitative,factual,problem}-trunc | Static prefix |
| 9-13 | Static-{definitional,procedural,quantitative,factual,problem}-suffix | Static suffix |
| 14-17 | LLM-{keyword,symptom,question,messy}-suffix | LLM suffix |
| 18 | LLM-keyword-trunc | LLM prefix |
| 19 | LLM-keyword+sep | Stacking |
| 20 | LLM-keyword-full-context | Full context |
| 21 | Novel-generic-trunc | Novel static prefix |

## 10 Primary Comparisons (Bonferroni alpha = 0.005)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | Best-static-trunc vs Bare | Do static prefixes help? |
| C2 | Best-static-suffix vs Bare | Do static suffixes help? |
| C3 | Best-static-trunc vs Best-static-suffix | Which mode? |
| C4 | Best-static vs LLM-keyword-trunc | Can statics match LLM? |
| C5 | Oracle-routed-static-K5 vs Best-single-static | Routing help? |
| C6 | Oracle-routed-static-K5 vs LLM-keyword-suffix | Routed statics vs LLM? |
| C7 | LLM-keyword-suffix vs LLM-keyword-trunc | Suffix vs truncation? |
| C8 | LLM-keyword-full-context vs LLM-keyword-trunc | Full-ctx vs truncated? |
| C9 | LLM-keyword+sep vs LLM-keyword-suffix | Stacking replicates? |
| C10 | Embedding-routed-LLM-K4 vs Oracle-routed-LLM-K4 | Practical routing? |

## Analysis Dimensions

1. Query intent (7 categories)
2. Answer length: short (<5 tokens), medium (5-15), long (>15)
3. Passage length: short (<80 words), medium (80-200), long (>200)
4. Difficulty quintile (bare NLL)""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import re
import numpy as np
import torch
from pathlib import Path
from collections import Counter

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("results/exp07")
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

# ========== Cell 3: Imports + config + templates ==========
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
    _set_cache_keys,
    _set_cache_values,
)
from lib.data import load_ms_marco, load_evaluation_samples
from lib.analysis import cohens_d, compute_token_overlap
from lib.surrogate import (
    generate_all_5_surrogates,
    STATIC_SURROGATE_QUERIES,
    TOP_5_SURROGATE_TEMPLATES,
)
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=4000,
    min_passage_words=20,
    max_passage_words=500,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing (hurts NLL, d=-0.45)
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"
SUFFIX_SEPARATOR = "\\n\\nRelated question: "
CHECKPOINT_EVERY = 50
N_COMPARISONS = 10
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
N_EVAL = 2000

# Static surrogate phrases (from lib)
STATIC_PHRASES = {
    'definitional': STATIC_SURROGATE_QUERIES['static_definitional']['query'],
    'procedural': STATIC_SURROGATE_QUERIES['static_procedural']['query'],
    'quantitative': STATIC_SURROGATE_QUERIES['static_quantitative']['query'],
    'factual': STATIC_SURROGATE_QUERIES['static_factual']['query'],
    'problem': STATIC_SURROGATE_QUERIES['static_problem']['query'],
}

# Novel generic phrase for condition 21
NOVEL_GENERIC_PHRASE = "What is this page about?"


def generate_random_prefix_text(target_text, tokenizer, seed):
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_len = len(target_ids)
    if target_len == 0:
        return ""
    rng = np.random.RandomState(seed)
    vocab_size = len(tokenizer)
    min_id = 3
    random_ids = rng.randint(min_id, vocab_size, size=target_len)
    random_text = tokenizer.decode(random_ids.tolist(), skip_special_tokens=True)
    reencoded = tokenizer.encode(random_text, add_special_tokens=False)
    if len(reencoded) != target_len:
        if len(reencoded) > target_len:
            random_text = tokenizer.decode(reencoded[:target_len], skip_special_tokens=True)
        else:
            extra_needed = target_len - len(reencoded)
            extra_ids = rng.randint(min_id, vocab_size, size=extra_needed)
            extra_text = tokenizer.decode(extra_ids.tolist(), skip_special_tokens=True)
            random_text = random_text + extra_text
            reencoded2 = tokenizer.encode(random_text, add_special_tokens=False)
            if len(reencoded2) > target_len:
                random_text = tokenizer.decode(reencoded2[:target_len], skip_special_tokens=True)
    return random_text


# Intent classification (rule-based)
INTENT_RULES = {
    'definitional': ['what is', 'define', 'meaning of', 'explain', 'what does', 'what are'],
    'procedural': ['how to', 'how do', 'steps', 'tutorial', 'guide', 'instructions'],
    'transactional': ['buy', 'cost', 'price', 'cheap', 'deal', 'order', 'purchase'],
    'comparison': ['best', 'top', 'vs', 'compare', 'review', 'difference between'],
    'factual': ['when did', 'where is', 'who', 'how many', 'how much', 'how long'],
    'medical': ['symptoms', 'treatment', 'diagnosis', 'causes', 'cure', 'side effects'],
}


def classify_intent(query):
    q = query.lower().strip()
    for intent, patterns in INTENT_RULES.items():
        for pattern in patterns:
            if pattern in q:
                return intent
    return 'other'


print("Config ready")
print(f"  num_samples pool: {config.num_samples}")
print(f"  eval samples: {N_EVAL}")
print(f"  passage words: {config.min_passage_words}-{config.max_passage_words}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: 21")
print(f"  static phrases: {list(STATIC_PHRASES.keys())}")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (2000 samples, full distribution)
dataset = load_ms_marco(config)

np.random.seed(SEED)
all_samples = load_evaluation_samples(dataset, config, require_answer=True)

samples = all_samples[:N_EVAL]
N = len(samples)
print(f"Loaded {len(all_samples)} candidates, using first {N} for evaluation")
print(f"Example passage ({len(samples[0]['passage'].split())} words): {samples[0]['passage'][:100]}...")
print(f"Example query: {samples[0]['query']}")
print(f"Example answer: {samples[0]['answer']}")

# Classify all queries by intent
intents = [classify_intent(s['query']) for s in samples]
intent_counts = Counter(intents)
print(f"\\nIntent distribution:")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  {intent}: {count} ({count/N*100:.1f}%)")\
""")))

# ========== Cell 5: Generate LLM surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 5: Generate ALL LLM surrogates (5 templates via generate_all_5_surrogates())
# We need: keyword_query, target_question, symptom_scenario, messy_realworld
# (misconception_negative not used in Exp 07 but generated anyway for consistency)

print("=" * 70)
print("PHASE 1: LLM SURROGATE GENERATION")
print("=" * 70)

surrogates_5_path = SURROGATES_DIR / "all_5_surrogates.json"

if surrogates_5_path.exists():
    with open(surrogates_5_path, 'r') as f:
        surrogates_5_data = json.load(f)
    surrogates_5 = surrogates_5_data['surrogates']
    print(f"Loaded {len(surrogates_5)} sets of 5-template surrogates from cache")
else:
    surrogates_5 = []

start_5 = len(surrogates_5)
if start_5 < N:
    print(f"Generating 5-template surrogates for samples {start_5} to {N-1}...")
    t_start = time.time()
    for idx in tqdm(range(start_5, N), initial=start_5, total=N,
                     desc="5-template surrogates"):
        passage = samples[idx]['passage']
        try:
            s5 = generate_all_5_surrogates(passage, model, tokenizer, config)
        except Exception as e:
            print(f"  WARNING: 5-template generation failed for sample {idx}: {e}")
            s5 = {k: "" for k in TOP_5_SURROGATE_TEMPLATES.keys()}
        surrogates_5.append(s5)

        if (idx + 1) % 100 == 0 or idx == N - 1:
            with open(surrogates_5_path, 'w') as f:
                json.dump({'surrogates': surrogates_5}, f)
            elapsed = time.time() - t_start
            rate = (idx - start_5 + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"  Saved {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

    with open(surrogates_5_path, 'w') as f:
        json.dump({'surrogates': surrogates_5}, f)
    print(f"5-template surrogates complete: {len(surrogates_5)} samples")
else:
    print(f"All 5-template surrogates already cached ({len(surrogates_5)} samples)")

# Validate
n_empty_kw = sum(1 for s in surrogates_5 if not s.get('keyword_query', '').strip())
n_empty_q = sum(1 for s in surrogates_5 if not s.get('target_question', '').strip())
n_empty_symp = sum(1 for s in surrogates_5 if not s.get('symptom_scenario', '').strip())
n_empty_messy = sum(1 for s in surrogates_5 if not s.get('messy_realworld', '').strip())
print(f"\\nValidation:")
print(f"  Empty keyword: {n_empty_kw}/{N}")
print(f"  Empty question: {n_empty_q}/{N}")
print(f"  Empty symptom: {n_empty_symp}/{N}")
print(f"  Empty messy: {n_empty_messy}/{N}")

print(f"\\nExamples (sample 0):")
print(f"  Passage: {samples[0]['passage'][:80]}...")
for k, v in surrogates_5[0].items():
    print(f"  {k}: {v}")\
""")))

# ========== Cell 6: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 6: Condition explanation with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

ex_i = 0
ex_passage = samples[ex_i]['passage']
ex_query = samples[ex_i]['query']
ex_llm_kw = surrogates_5[ex_i].get('keyword_query', '')

conditions_explained = [
    ("1. Bare", "[BOS][doc]", "No prefix — baseline"),
    ("2. Random-truncated", "[BOS][random\\\\n][doc] → truncate + RoPE",
     f"Random text matching query token length"),
    ("3. Separator-only", "[BOS][doc][\\\\n\\\\nRelated question: ]",
     "Suffix framing only — no content after separator"),
]
for name, phrase in STATIC_PHRASES.items():
    i = list(STATIC_PHRASES.keys()).index(name)
    conditions_explained.append(
        (f"{4+i}. Static-{name}-trunc", f"[BOS]['{phrase}'\\\\n][doc] → truncate + RoPE",
         f"Static phrase as prefix")
    )
for name, phrase in STATIC_PHRASES.items():
    i = list(STATIC_PHRASES.keys()).index(name)
    conditions_explained.append(
        (f"{9+i}. Static-{name}-suffix", f"[BOS][doc][\\\\n\\\\nRelated question: {phrase}]",
         f"Static phrase as suffix")
    )
conditions_explained += [
    ("14. LLM-keyword-suffix", "[BOS][doc][\\\\n\\\\nRelated question: llm_kw]",
     f"LLM keyword: '{ex_llm_kw[:60]}'"),
    ("15. LLM-symptom-suffix", "[BOS][doc][\\\\n\\\\nRelated question: llm_symp]",
     "LLM symptom scenario as suffix"),
    ("16. LLM-question-suffix", "[BOS][doc][\\\\n\\\\nRelated question: llm_q]",
     "LLM question as suffix"),
    ("17. LLM-messy-suffix", "[BOS][doc][\\\\n\\\\nRelated question: llm_messy]",
     "LLM messy/informal as suffix"),
    ("18. LLM-keyword-trunc", "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     "LLM keyword as truncated prefix"),
    ("19. LLM-keyword+sep", "[BOS][llm_kw\\\\n][doc][\\\\n\\\\nRelated question: ] (prefix+suffix)",
     "Stacking: truncated prefix + suffix separator (Exp 06 best)"),
    ("20. LLM-keyword-full-context", "[BOS][llm_kw\\\\n][doc] — NOT truncated",
     "Full context: query sees prefix + doc"),
    ("21. Novel-generic-trunc", f"[BOS]['{NOVEL_GENERIC_PHRASE}'\\\\n][doc] → truncate + RoPE",
     "Ultra-cheap novel generic prefix"),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")\
""")))

# ========== Cell 7: Main eval loop (21 conditions × 2000 samples) ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main eval loop — 21 conditions × 2000 samples
print("=" * 70)
print("PHASE 2: MAIN EVALUATION (21 conditions × 2000 samples)")
print("=" * 70)

CONDITION_NAMES = [
    'bare', 'random_trunc', 'separator_only',
    'static_def_trunc', 'static_proc_trunc', 'static_quant_trunc',
    'static_fact_trunc', 'static_prob_trunc',
    'static_def_suffix', 'static_proc_suffix', 'static_quant_suffix',
    'static_fact_suffix', 'static_prob_suffix',
    'llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix', 'llm_messy_suffix',
    'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx',
    'novel_generic_trunc',
]

STATIC_TRUNC_CONDS = ['static_def_trunc', 'static_proc_trunc', 'static_quant_trunc',
                       'static_fact_trunc', 'static_prob_trunc']
STATIC_SUFFIX_CONDS = ['static_def_suffix', 'static_proc_suffix', 'static_quant_suffix',
                        'static_fact_suffix', 'static_prob_suffix']
LLM_SUFFIX_CONDS = ['llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix',
                      'llm_messy_suffix']

static_phrase_list = list(STATIC_PHRASES.values())
static_name_list = list(STATIC_PHRASES.keys())

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

    # --- Matched tokenization (same doc_ids for ALL truncated conditions) ---
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

    # Pre-compute LLM surrogate texts
    llm_kw_text = surrogates_5[idx].get('keyword_query', '')
    llm_symp_text = surrogates_5[idx].get('symptom_scenario', '')
    llm_q_text = surrogates_5[idx].get('target_question', '')
    llm_messy_text = surrogates_5[idx].get('messy_realworld', '')

    # Helper: build truncated cache from prefix text
    def build_trunc(prefix_text):
        prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=prefix_text)
        prefix_enc = tokenizer(prefix_str, return_tensors="pt",
                               add_special_tokens=False, padding=False, truncation=False)
        prefix_ids = prefix_enc['input_ids'].to(config.device)
        full_ids = torch.cat([bos_id, prefix_ids, doc_ids], dim=1)
        prefix_token_len = 1 + prefix_ids.shape[1]

        with torch.no_grad():
            out = model(input_ids=full_ids,
                        attention_mask=torch.ones_like(full_ids),
                        use_cache=True, return_dict=True)
        cache = extract_and_truncate_cache_with_bos(out.past_key_values, doc_len)
        correct_rope_positions_with_bos(cache, prefix_token_len - 1, model)
        nll = score_answer_with_cache(
            deepcopy_cache(cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del out, cache
        return nll

    # === Condition 1: BARE ===
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_out.past_key_values), bare_ids.shape[1],
        query_prompt, answer_text, model, tokenizer, config)
    del bare_out

    # === Condition 2: RANDOM-TRUNCATED ===
    random_text = generate_random_prefix_text(query, tokenizer, seed=SEED + idx)
    nll_random = build_trunc(random_text)

    # === Condition 3: SEPARATOR-ONLY ===
    sep_len, sep_cache = build_suffix_kv_cache(
        passage, "", model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_separator = score_answer_with_cache(
        deepcopy_cache(sep_cache), sep_len,
        query_prompt, answer_text, model, tokenizer, config)
    del sep_cache

    # === Conditions 4-8: STATIC TRUNCATED ===
    nll_static_trunc = {}
    for sname, sphrase in STATIC_PHRASES.items():
        nll_static_trunc[sname] = build_trunc(sphrase)

    # === Conditions 9-13: STATIC SUFFIX ===
    nll_static_suffix = {}
    for sname, sphrase in STATIC_PHRASES.items():
        suf_len, suf_cache = build_suffix_kv_cache(
            passage, sphrase, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
        nll_static_suffix[sname] = score_answer_with_cache(
            deepcopy_cache(suf_cache), suf_len,
            query_prompt, answer_text, model, tokenizer, config)
        del suf_cache

    # === Conditions 14-17: LLM SUFFIX ===
    llm_suffix_texts = {
        'keyword': llm_kw_text,
        'symptom': llm_symp_text,
        'question': llm_q_text,
        'messy': llm_messy_text,
    }
    nll_llm_suffix = {}
    for lname, ltext in llm_suffix_texts.items():
        ls_len, ls_cache = build_suffix_kv_cache(
            passage, ltext, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
        nll_llm_suffix[lname] = score_answer_with_cache(
            deepcopy_cache(ls_cache), ls_len,
            query_prompt, answer_text, model, tokenizer, config)
        del ls_cache

    # === Condition 18: LLM-KEYWORD-TRUNC ===
    nll_llm_kw_trunc = build_trunc(llm_kw_text)

    # === Condition 19: LLM-KEYWORD+SEP (stacking) ===
    # Step 1: Build truncated cache with LLM keyword prefix
    kw_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=llm_kw_text)
    kw_prefix_enc = tokenizer(kw_prefix_str, return_tensors="pt",
                              add_special_tokens=False, padding=False, truncation=False)
    kw_prefix_ids = kw_prefix_enc['input_ids'].to(config.device)
    kw_full_ids = torch.cat([bos_id, kw_prefix_ids, doc_ids], dim=1)
    kw_prefix_token_len = 1 + kw_prefix_ids.shape[1]

    with torch.no_grad():
        kw_out = model(input_ids=kw_full_ids,
                       attention_mask=torch.ones_like(kw_full_ids),
                       use_cache=True, return_dict=True)
    kw_trunc_cache = extract_and_truncate_cache_with_bos(kw_out.past_key_values, doc_len)
    correct_rope_positions_with_bos(kw_trunc_cache, kw_prefix_token_len - 1, model)
    del kw_out

    # Step 2: Extend with suffix separator
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
            past_key_values=deepcopy_cache(kw_trunc_cache),
            use_cache=True,
            return_dict=True,
        )
    combo_cache = suffix_out.past_key_values
    combo_len = cache_len_before_suffix + suffix_len_tok
    nll_llm_kw_sep = score_answer_with_cache(
        deepcopy_cache(combo_cache), combo_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suffix_out, combo_cache, kw_trunc_cache

    # === Condition 20: LLM-KEYWORD-FULL-CONTEXT (NOT truncated) ===
    kw_full_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=llm_kw_text)
    kw_full_text = kw_full_prefix_str + document_text
    kw_full_enc = tokenizer(kw_full_text, return_tensors="pt",
                            add_special_tokens=True, padding=False, truncation=False)
    kw_full_context_ids = kw_full_enc['input_ids'].to(config.device)
    kw_full_context_len = kw_full_context_ids.shape[1]

    with torch.no_grad():
        kw_full_out = model(input_ids=kw_full_context_ids,
                            attention_mask=torch.ones_like(kw_full_context_ids),
                            use_cache=True, return_dict=True)
    nll_llm_kw_full_ctx = score_answer_with_cache(
        deepcopy_cache(kw_full_out.past_key_values), kw_full_context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del kw_full_out

    # === Condition 21: NOVEL-GENERIC-TRUNC ===
    nll_novel_generic = build_trunc(NOVEL_GENERIC_PHRASE)

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len': doc_len,
        'passage_word_count': len(passage.split()),
        'answer_token_count': len(tokenizer.encode(answer, add_special_tokens=False)),
        'intent': intents[idx],
        'bare': nll_bare,
        'random_trunc': nll_random,
        'separator_only': nll_separator,
        'static_def_trunc': nll_static_trunc['definitional'],
        'static_proc_trunc': nll_static_trunc['procedural'],
        'static_quant_trunc': nll_static_trunc['quantitative'],
        'static_fact_trunc': nll_static_trunc['factual'],
        'static_prob_trunc': nll_static_trunc['problem'],
        'static_def_suffix': nll_static_suffix['definitional'],
        'static_proc_suffix': nll_static_suffix['procedural'],
        'static_quant_suffix': nll_static_suffix['quantitative'],
        'static_fact_suffix': nll_static_suffix['factual'],
        'static_prob_suffix': nll_static_suffix['problem'],
        'llm_keyword_suffix': nll_llm_suffix['keyword'],
        'llm_symptom_suffix': nll_llm_suffix['symptom'],
        'llm_question_suffix': nll_llm_suffix['question'],
        'llm_messy_suffix': nll_llm_suffix['messy'],
        'llm_keyword_trunc': nll_llm_kw_trunc,
        'llm_keyword_sep': nll_llm_kw_sep,
        'llm_keyword_full_ctx': nll_llm_kw_full_ctx,
        'novel_generic_trunc': nll_novel_generic,
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
# Cell 8: Primary analysis — filter, NLL summary, 10 comparisons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — PRIMARY COMPARISONS")
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

# Metadata arrays (filtered)
intents_valid = np.array(intents)[valid]
answer_tokens_valid = np.array([r['answer_token_count'] for r in results])[valid]
passage_words_valid = np.array([r['passage_word_count'] for r in results])[valid]

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

# Identify best statics
best_static_trunc_d = -999
best_static_trunc_name = ''
for cname in STATIC_TRUNC_CONDS:
    d = cohens_d(c['bare'] - c[cname])
    if d > best_static_trunc_d:
        best_static_trunc_d = d
        best_static_trunc_name = cname

best_static_suffix_d = -999
best_static_suffix_name = ''
for cname in STATIC_SUFFIX_CONDS:
    d = cohens_d(c['bare'] - c[cname])
    if d > best_static_suffix_d:
        best_static_suffix_d = d
        best_static_suffix_name = cname

best_static_overall_d = max(best_static_trunc_d, best_static_suffix_d)
best_static_overall_name = best_static_trunc_name if best_static_trunc_d > best_static_suffix_d else best_static_suffix_name

print(f"\\nBest static trunc: {best_static_trunc_name} (d={best_static_trunc_d:+.3f})")
print(f"Best static suffix: {best_static_suffix_name} (d={best_static_suffix_d:+.3f})")
print(f"Best static overall: {best_static_overall_name} (d={best_static_overall_d:+.3f})")

# Oracle-routed statics (per-sample min NLL across 5 statics)
oracle_routed_static_trunc = np.minimum.reduce([c[cn] for cn in STATIC_TRUNC_CONDS])
oracle_routed_static_suffix = np.minimum.reduce([c[cn] for cn in STATIC_SUFFIX_CONDS])
oracle_routed_static_k5 = np.minimum(oracle_routed_static_trunc, oracle_routed_static_suffix)

# Oracle-routed LLM (per-sample min NLL across 4 LLM suffix)
oracle_routed_llm_k4 = np.minimum.reduce([c[cn] for cn in LLM_SUFFIX_CONDS])

print(f"\\nOracle-routed-static-K5 d vs bare: {cohens_d(c['bare'] - oracle_routed_static_k5):+.3f}")
print(f"Oracle-routed-LLM-K4 d vs bare: {cohens_d(c['bare'] - oracle_routed_llm_k4):+.3f}")

# 10 primary comparisons
print(f"\\n{'='*85}")
print(f"10 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*85}")

comparisons = [
    ('C1: Best-static-trunc vs Bare',
     c['bare'] - c[best_static_trunc_name],
     'Do static prefixes help at all?'),
    ('C2: Best-static-suffix vs Bare',
     c['bare'] - c[best_static_suffix_name],
     'Do static suffixes help at all?'),
    ('C3: Best-static-trunc vs Best-static-suffix',
     c[best_static_suffix_name] - c[best_static_trunc_name],
     'Which mode is better?'),
    ('C4: Best-static vs LLM-kw-trunc',
     c[best_static_overall_name] - c['llm_keyword_trunc'],
     'Can statics match LLM?'),
    ('C5: Oracle-static-K5 vs Best-static',
     c[best_static_overall_name] - oracle_routed_static_k5,
     'Does routing help?'),
    ('C6: Oracle-static-K5 vs LLM-kw-suf',
     oracle_routed_static_k5 - c['llm_keyword_suffix'],
     'Routed statics vs single LLM?'),
    ('C7: LLM-kw-suffix vs LLM-kw-trunc',
     c['llm_keyword_trunc'] - c['llm_keyword_suffix'],
     'Suffix vs truncation for LLM?'),
    ('C8: LLM-kw-full vs LLM-kw-trunc',
     c['llm_keyword_trunc'] - c['llm_keyword_full_ctx'],
     'Full-ctx vs truncated?'),
    ('C9: LLM-kw+sep vs LLM-kw-suffix',
     c['llm_keyword_suffix'] - c['llm_keyword_sep'],
     'Does stacking replicate?'),
    ('C10: Embed-routed-LLM vs Oracle-LLM',
     None,  # Computed in routing analysis cell
     'Is practical routing viable?'),
]

print(f"\\n{'Comparison':<40} {'Mean Δ':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 95)

comparison_results = {}
for name, delta, question in comparisons:
    if delta is None:
        print(f"{name:<40} {'(computed later)':>50}")
        continue
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<40} {np.mean(delta):>8.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
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

# ========== Cell 9: Hardness quintile breakdown ==========
cells.append(make_cell("code", s("""\
# Cell 9: Hardness quintile breakdown (all conditions)

print("=" * 70)
print("HARDNESS QUINTILE BREAKDOWN")
print("=" * 70)

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

conditions_to_show = [
    'random_trunc', 'separator_only',
    'static_def_trunc', 'static_proc_trunc', 'static_fact_trunc',
    'static_def_suffix', 'static_proc_suffix', 'static_fact_suffix',
    'llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix',
    'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx',
    'novel_generic_trunc',
]

header = f"{'Condition':<25}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (25 + 14 * 6))

hardness_breakdown = {}
for cname in conditions_to_show:
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
    hardness_breakdown[cname] = {
        'quintile_ds': quintile_ds,
        'overall_d': float(d_all),
    }

# Hardness interaction correlations
print(f"\\nHardness interaction (r between bare NLL and benefit):")
for cname in conditions_to_show:
    delta = bare_valid - c[cname]
    r, p = stats.pearsonr(bare_valid, delta)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {cname:<25}: r={r:+.3f}, p={p:.2e} {sig}")\
""")))

# ========== Cell 10: Stratification by intent, answer length, passage length ==========
cells.append(make_cell("code", s("""\
# Cell 10: Stratification by intent, answer length, passage length

print("=" * 70)
print("STRATIFICATION ANALYSIS")
print("=" * 70)

# --- 1. Intent stratification ---
print("\\n--- Intent Stratification ---")
intent_categories = sorted(set(intents_valid))
key_conditions = ['bare', 'separator_only', best_static_trunc_name, best_static_suffix_name,
                   'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']

header = f"{'Intent':<15} {'N':>5}"
for cn in key_conditions:
    if cn == 'bare':
        header += f"{'bare NLL':>12}"
    else:
        header += f"{cn[:10]:>12}"
print(header)
print("-" * (20 + 12 * len(key_conditions)))

intent_analysis = {}
for intent in intent_categories:
    mask = intents_valid == intent
    n_intent = int(np.sum(mask))
    if n_intent < 10:
        continue
    row = f"{intent:<15} {n_intent:>5}"
    intent_ds = {}
    for cn in key_conditions:
        if cn == 'bare':
            row += f"{np.mean(c['bare'][mask]):>12.3f}"
        else:
            d = cohens_d(c['bare'][mask] - c[cn][mask])
            row += f"{d:>+12.3f}"
            intent_ds[cn] = float(d)
    print(row)
    intent_analysis[intent] = {'n': n_intent, 'ds': intent_ds}

# Best surrogate per intent
print(f"\\nBest surrogate per intent:")
all_statics_for_intent = STATIC_TRUNC_CONDS + STATIC_SUFFIX_CONDS
for intent in intent_categories:
    mask = intents_valid == intent
    if int(np.sum(mask)) < 10:
        continue
    best_d = -999
    best_cn = ''
    for cn in all_statics_for_intent:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        if d > best_d:
            best_d = d
            best_cn = cn
    print(f"  {intent:<15}: {best_cn} (d={best_d:+.3f})")

# --- 2. Answer length stratification ---
print("\\n--- Answer Length Stratification ---")
ans_short = answer_tokens_valid < 5
ans_medium = (answer_tokens_valid >= 5) & (answer_tokens_valid <= 15)
ans_long = answer_tokens_valid > 15

print(f"Answer length bins: short(<5)={int(np.sum(ans_short))}, "
      f"medium(5-15)={int(np.sum(ans_medium))}, long(>15)={int(np.sum(ans_long))}")

for label, mask in [('short', ans_short), ('medium', ans_medium), ('long', ans_long)]:
    if int(np.sum(mask)) < 10:
        continue
    row = f"  {label:<10} N={int(np.sum(mask)):>4}"
    for cn in ['separator_only', 'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        row += f"  {cn[:12]}={d:+.3f}"
    print(row)

# --- 3. Passage length stratification ---
print("\\n--- Passage Length Stratification ---")
psg_short = passage_words_valid < 80
psg_medium = (passage_words_valid >= 80) & (passage_words_valid <= 200)
psg_long = passage_words_valid > 200

print(f"Passage length bins: short(<80)={int(np.sum(psg_short))}, "
      f"medium(80-200)={int(np.sum(psg_medium))}, long(>200)={int(np.sum(psg_long))}")

for label, mask in [('short', psg_short), ('medium', psg_medium), ('long', psg_long)]:
    if int(np.sum(mask)) < 10:
        continue
    row = f"  {label:<10} N={int(np.sum(mask)):>4}"
    for cn in ['separator_only', 'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        row += f"  {cn[:12]}={d:+.3f}"
    print(row)\
""")))

# ========== Cell 11: Routing analysis ==========
cells.append(make_cell("code", s("""\
# Cell 11: Routing analysis — oracle K-curve, embedding routing, intent-matched, complementarity

print("=" * 70)
print("ROUTING ANALYSIS")
print("=" * 70)

# --- 1. Oracle K-curve (best-of-K for statics and LLMs) ---
print("\\n--- Oracle K-Curve ---")
all_static_conds = STATIC_TRUNC_CONDS + STATIC_SUFFIX_CONDS

# For statics: sort by overall d, pick top K
static_ds = [(cn, cohens_d(c['bare'] - c[cn])) for cn in all_static_conds]
static_ds.sort(key=lambda x: -x[1])

print(f"\\nStatic surrogates ranked by d vs bare:")
for cn, d in static_ds:
    print(f"  {cn}: d={d:+.3f}")

print(f"\\nOracle best-of-K (static):")
for K in range(1, 6):
    top_k_conds = [cn for cn, _ in static_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k_conds])
    d = cohens_d(c['bare'] - best_of_k)
    print(f"  K={K}: d={d:+.3f} (conditions: {', '.join(top_k_conds)})")

# LLM K-curve
llm_ds = [(cn, cohens_d(c['bare'] - c[cn])) for cn in LLM_SUFFIX_CONDS]
llm_ds.sort(key=lambda x: -x[1])
print(f"\\nOracle best-of-K (LLM suffix):")
for K in range(1, 5):
    top_k_conds = [cn for cn, _ in llm_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k_conds])
    d = cohens_d(c['bare'] - best_of_k)
    print(f"  K={K}: d={d:+.3f}")

# --- 2. Embedding routing ---
print("\\n--- Embedding Routing ---")
print("Loading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all queries (valid subset)
valid_indices = np.where(valid)[0]
valid_queries = [samples[i]['query'] for i in valid_indices]
print(f"Embedding {len(valid_queries)} queries...")
query_embeddings = embed_model.encode(valid_queries, show_progress_bar=True)

# Embed LLM surrogates for each sample
llm_surrogate_keys = ['keyword', 'symptom', 'question', 'messy']
llm_surrogate_cond_map = {
    'keyword': 'llm_keyword_suffix',
    'symptom': 'llm_symptom_suffix',
    'question': 'llm_question_suffix',
    'messy': 'llm_messy_suffix',
}

print("Embedding LLM surrogates...")
# For each sample, embed 4 surrogates and route by cosine sim to query
embed_routed_nlls = np.zeros(n_valid)
embed_routed_choices = []
oracle_routed_choices = []

for vi in tqdm(range(n_valid), desc="Embedding routing"):
    orig_idx = valid_indices[vi]
    q_emb = query_embeddings[vi:vi+1]

    # Get surrogate texts
    surr_texts = {
        'keyword': surrogates_5[orig_idx].get('keyword_query', ''),
        'symptom': surrogates_5[orig_idx].get('symptom_scenario', ''),
        'question': surrogates_5[orig_idx].get('target_question', ''),
        'messy': surrogates_5[orig_idx].get('messy_realworld', ''),
    }

    # Embed surrogates
    surr_embs = embed_model.encode([surr_texts[k] for k in llm_surrogate_keys])
    sims = cos_sim(q_emb, surr_embs)[0]

    # Route by max cosine sim
    best_k_idx = np.argmax(sims)
    best_k = llm_surrogate_keys[best_k_idx]
    embed_routed_nlls[vi] = c[llm_surrogate_cond_map[best_k]][vi]
    embed_routed_choices.append(best_k)

    # Oracle: min NLL
    nlls_4 = {k: c[llm_surrogate_cond_map[k]][vi] for k in llm_surrogate_keys}
    oracle_k = min(nlls_4, key=nlls_4.get)
    oracle_routed_choices.append(oracle_k)

# Report
d_embed_routed = cohens_d(c['bare'] - embed_routed_nlls)
d_oracle_routed = cohens_d(c['bare'] - oracle_routed_llm_k4)
oracle_accuracy = np.mean([e == o for e, o in zip(embed_routed_choices, oracle_routed_choices)])

print(f"\\nEmbedding-routed-LLM-K4: d={d_embed_routed:+.3f}")
print(f"Oracle-routed-LLM-K4: d={d_oracle_routed:+.3f}")
print(f"Embedding routing accuracy vs oracle: {oracle_accuracy*100:.1f}%")

# C10 comparison
delta_c10 = embed_routed_nlls - oracle_routed_llm_k4
d_c10 = cohens_d(delta_c10)
t_c10, p_c10 = stats.ttest_1samp(delta_c10, 0)
sig_c10 = "***" if p_c10 < 0.001 else "**" if p_c10 < BONFERRONI_ALPHA else "*" if p_c10 < 0.05 else "ns"
print(f"\\nC10: Embed-routed vs Oracle-routed: d={d_c10:+.3f}, p={p_c10:.2e} {sig_c10}")
comparison_results['C10: Embed-routed-LLM vs Oracle-LLM'] = {
    'mean_delta': float(np.mean(delta_c10)),
    'cohens_d': float(d_c10),
    'win_rate': float(np.mean(delta_c10 > 0)),
    't_stat': float(t_c10),
    'p_value': float(p_c10),
    'bonferroni_significant': bool(p_c10 < BONFERRONI_ALPHA),
    'question': 'Is practical routing viable?',
}

# Routing choice distribution
print(f"\\nEmbedding routing choice distribution:")
for k in llm_surrogate_keys:
    n_chosen = sum(1 for ch in embed_routed_choices if ch == k)
    print(f"  {k}: {n_chosen} ({n_chosen/n_valid*100:.1f}%)")

print(f"\\nOracle routing choice distribution:")
for k in llm_surrogate_keys:
    n_chosen = sum(1 for ch in oracle_routed_choices if ch == k)
    print(f"  {k}: {n_chosen} ({n_chosen/n_valid*100:.1f}%)")

# --- 3. Intent-matched static routing ---
print("\\n--- Intent-Matched Static Routing ---")
intent_to_static = {
    'definitional': 'static_def_suffix',
    'procedural': 'static_proc_suffix',
    'transactional': 'static_quant_suffix',
    'comparison': 'static_fact_suffix',
    'factual': 'static_fact_suffix',
    'medical': 'static_prob_suffix',
    'other': best_static_suffix_name,
}

intent_matched_nlls = np.zeros(n_valid)
for vi in range(n_valid):
    intent = intents_valid[vi]
    matched_cond = intent_to_static.get(intent, best_static_suffix_name)
    intent_matched_nlls[vi] = c[matched_cond][vi]

d_intent_matched = cohens_d(c['bare'] - intent_matched_nlls)
print(f"Intent-matched-static d vs bare: {d_intent_matched:+.3f}")
print(f"Best-single-static d vs bare: {best_static_overall_d:+.3f}")
print(f"Improvement from intent matching: {d_intent_matched - best_static_overall_d:+.3f}")

# --- 4. Complementarity matrix ---
print("\\n--- Complementarity Matrix (LLM suffix, fraction where row wins over column) ---")
llm_conds_for_comp = LLM_SUFFIX_CONDS
header = f"{'':>20}" + "".join(f"{cn[:12]:>14}" for cn in llm_conds_for_comp)
print(header)
for cn_a in llm_conds_for_comp:
    row = f"{cn_a[:20]:<20}"
    for cn_b in llm_conds_for_comp:
        if cn_a == cn_b:
            row += f"{'—':>14}"
        else:
            frac = np.mean(c[cn_a] < c[cn_b])
            row += f"{frac:>14.3f}"
    print(row)

# --- 5. Hardness-gated routing ---
print("\\n--- Hardness-Gated Routing ---")
print("Prime only if bare_NLL > threshold. Sweep threshold.")
thresholds = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
for thresh in thresholds:
    mask_prime = bare_valid > thresh
    n_primed = int(np.sum(mask_prime))
    if n_primed < 10:
        continue
    # For primed: use LLM-keyword-sep; for unprimed: use bare
    gated_nlls = np.where(mask_prime, c['llm_keyword_sep'], c['bare'])
    d_gated = cohens_d(c['bare'] - gated_nlls)
    frac_primed = n_primed / n_valid * 100
    print(f"  threshold={thresh:.1f}: prime {frac_primed:.0f}% samples, d={d_gated:+.3f}")

del embed_model  # Free memory\
""")))

# ========== Cell 12: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 12: Plots

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# --- Plot 1: All conditions bar chart (d vs bare) ---
ax = axes[0, 0]
cnames_sorted = sorted(
    [cn for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda cn: cohens_d(c['bare'] - c[cn]),
    reverse=True
)
ds_bar = [cohens_d(c['bare'] - c[cn]) for cn in cnames_sorted]
color_map = {}
for cn in STATIC_TRUNC_CONDS:
    color_map[cn] = 'steelblue'
for cn in STATIC_SUFFIX_CONDS:
    color_map[cn] = 'cornflowerblue'
for cn in LLM_SUFFIX_CONDS:
    color_map[cn] = 'forestgreen'
color_map['llm_keyword_trunc'] = 'darkgreen'
color_map['llm_keyword_sep'] = 'gold'
color_map['llm_keyword_full_ctx'] = 'orange'
color_map['novel_generic_trunc'] = 'mediumpurple'
color_map['random_trunc'] = 'gray'
color_map['separator_only'] = 'salmon'
colors_bar = [color_map.get(cn, 'lightgray') for cn in cnames_sorted]

bars = ax.barh(range(len(cnames_sorted)), ds_bar, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(cnames_sorted)))
ax.set_yticklabels(cnames_sorted, fontsize=7)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare')
ax.invert_yaxis()

# --- Plot 2: Static trunc vs suffix comparison ---
ax = axes[0, 1]
static_names = list(STATIC_PHRASES.keys())
abbrev_map = {'definitional': 'def', 'procedural': 'proc', 'quantitative': 'quant', 'factual': 'fact', 'problem': 'prob'}
trunc_ds = [cohens_d(c['bare'] - c[f'static_{abbrev_map[n]}_trunc']) for n in static_names]
suffix_ds = [cohens_d(c['bare'] - c[f'static_{abbrev_map[n]}_suffix']) for n in static_names]
x = np.arange(len(static_names))
width = 0.35
ax.bar(x - width/2, trunc_ds, width, label='Truncated prefix', color='steelblue', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, suffix_ds, width, label='Suffix', color='cornflowerblue', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([n[:6] for n in static_names], fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Static: Prefix vs Suffix Mode')
ax.legend(fontsize=8)

# --- Plot 3: Hardness × condition heatmap ---
ax = axes[0, 2]
hm_conditions = ['separator_only', best_static_trunc_name, best_static_suffix_name,
                  'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx']
hm_data = []
for cname in hm_conditions:
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
ax.set_yticks(range(len(hm_conditions)))
ax.set_yticklabels(hm_conditions, fontsize=7)
for i in range(len(hm_conditions)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness × Condition')

# --- Plot 4: Oracle K-curve ---
ax = axes[1, 0]
static_k_ds = []
for K in range(1, 6):
    top_k = [cn for cn, _ in static_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k])
    static_k_ds.append(cohens_d(c['bare'] - best_of_k))
llm_k_ds = []
for K in range(1, 5):
    top_k = [cn for cn, _ in llm_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k])
    llm_k_ds.append(cohens_d(c['bare'] - best_of_k))

ax.plot(range(1, 6), static_k_ds, 'o-', color='steelblue', label='Static (10 total)')
ax.plot(range(1, 5), llm_k_ds, 's-', color='forestgreen', label='LLM suffix (4 total)')
ax.set_xlabel('K (best-of-K)')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Oracle K-Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 5: Intent stratification ---
ax = axes[1, 1]
intent_cats = sorted(intent_analysis.keys(), key=lambda x: -intent_analysis[x]['n'])
intent_cats = [ic for ic in intent_cats if intent_analysis[ic]['n'] >= 10]
x_int = np.arange(len(intent_cats))
width_int = 0.25
for j, cn in enumerate(['llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']):
    ds_int = [intent_analysis[ic]['ds'].get(cn, 0) for ic in intent_cats]
    ax.bar(x_int + j * width_int - width_int, ds_int, width_int, label=cn[:15],
           edgecolor='black', linewidth=0.5)
ax.set_xticks(x_int)
ax.set_xticklabels(intent_cats, fontsize=7, rotation=45)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Intent Stratification')
ax.legend(fontsize=6)

# --- Plot 6: Hardness-gated routing curve ---
ax = axes[1, 2]
thresh_vals = np.arange(0, 4.1, 0.2)
gated_ds = []
frac_primed_vals = []
for thresh in thresh_vals:
    mask_prime = bare_valid > thresh
    n_primed = int(np.sum(mask_prime))
    if n_primed < 5:
        gated_ds.append(np.nan)
        frac_primed_vals.append(0)
        continue
    gated_nlls = np.where(mask_prime, c['llm_keyword_sep'], c['bare'])
    gated_ds.append(cohens_d(c['bare'] - gated_nlls))
    frac_primed_vals.append(n_primed / n_valid * 100)

ax2 = ax.twinx()
ax.plot(thresh_vals, gated_ds, 'o-', color='forestgreen', markersize=3, label='d vs bare')
ax2.plot(thresh_vals, frac_primed_vals, 's--', color='gray', markersize=3, alpha=0.5, label='% primed')
ax.set_xlabel('Bare NLL Threshold')
ax.set_ylabel("Cohen's d vs bare (gated)", color='forestgreen')
ax2.set_ylabel('% Samples Primed', color='gray')
ax.set_title('Hardness-Gated Routing')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=7)
ax2.legend(loc='upper right', fontsize=7)

plt.suptitle('Exp 07: Static Surrogates, Dual-Mode Priming, Intent Routing', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 13: Save results JSON ==========
cells.append(make_cell("code", s("""\
# Cell 13: Save comprehensive results JSON

final = {
    'experiment': 'exp07_static_surrogates_and_routing',
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
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'hardness_breakdown': hardness_breakdown,
    'intent_analysis': intent_analysis,
    'routing_analysis': {
        'oracle_static_k5_d': float(cohens_d(c['bare'] - oracle_routed_static_k5)),
        'oracle_llm_k4_d': float(d_oracle_routed),
        'embed_routed_llm_k4_d': float(d_embed_routed),
        'embed_routing_accuracy': float(oracle_accuracy),
        'intent_matched_static_d': float(d_intent_matched),
        'best_static_trunc': best_static_trunc_name,
        'best_static_suffix': best_static_suffix_name,
        'best_static_overall': best_static_overall_name,
    },
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 14: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 14: GPU cleanup
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

output_path = "/home/jupyter/research/directed_kvcache_v2/07_static_surrogates_and_routing.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
