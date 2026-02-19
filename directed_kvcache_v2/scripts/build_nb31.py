#!/usr/bin/env python3
"""Build script for 31_ad_benchmark_and_generation.ipynb

Exp 31: Ad-Content Benchmark & Generation Quality (Gemma 3 4B)

Tests whether hero layer priming helps on ad-like short content and whether
lower NLL translates to better generated answers.

Phase 1: NLL evaluation on 3 short-passage datasets (900 samples x 4 conditions)
  - Amazon SubjQA (electronics + grocery): Commercial product review QA
  - MS MARCO short (<250 words): Known positive control
  - SQuAD short (<250 words): Clean extractive QA

Phase 2: Generation quality on hard subset (bare vs hero)
  - Greedy generation with bare and hero caches
  - Metrics: Exact Match, Token F1, ROUGE-L, generation confidence

Conditions (same 4 as Exp 30):
  1. bare: BOS + doc standard causal (baseline)
  2. sf_trunc: Standard priming (truncate + RoPE correct)
  3. values_early: Bare keys + primed values layers 0-15
  4. values_hero: Bare keys + primed values at hero layers {10,12,14,15,20}

Model: Gemma 3 4B (4-bit, bfloat16)
N = 300 per dataset (900 total for Phase 1)
"""

import json
import ast
import sys


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
# Exp 31: Ad-Content Benchmark & Generation Quality (Gemma 3 4B)

## Motivation

Prior experiments showed hero layer value contamination helps on short factoid QA
(MS MARCO d=+0.472, NQ d=+0.213) but fails on long documents and non-retrieval tasks.
For ad-serving, the content is typically short (product descriptions, review snippets)
and the task is generation (producing relevant responses), not just NLL scoring.

This experiment tests two questions:
1. **Does hero layer priming help on ad-like short content?** (NLL benchmark)
2. **Does lower NLL translate to better generated answers?** (Generation quality)

## Datasets

| Dataset | Content Type | Why Chosen | N |
|---------|-------------|-----------|---|
| **Amazon SubjQA** | Product reviews (electronics + grocery) | Commercial content closest to ads | 300 |
| **MS MARCO** | Web passages (<250 words) | Known positive control for priming | 300 |
| **SQuAD** | Wikipedia paragraphs (<250 words) | Clean extractive QA, new short-passage data point | 300 |

All passages filtered to 30-250 words to stay in the "short content" regime.

## Two Phases

**Phase 1: NLL Evaluation** (900 samples x 4 conditions)
- bare, sf_trunc, values_early, values_hero
- Same methodology as Exp 30

**Phase 2: Generation Quality** (hard subset x 2 conditions)
- Greedy generation with bare vs hero caches
- Metrics: Exact Match (substring), Token F1, ROUGE-L, generation confidence""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup
import os
os.umask(0o000)

import sys
import json
import time
import re
import gc
import csv
import string
import numpy as np
import torch
from pathlib import Path
from collections import Counter

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp31")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
GEN_CHECKPOINT_PATH = RESULTS_DIR / "gen_checkpoint.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"
CSV_PATH = RESULTS_DIR / "results.csv"
GEN_CSV_PATH = RESULTS_DIR / "gen_results.csv"

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
    compute_dtype="auto",  # resolves to bfloat16 for Gemma
    use_4bit=True,
    num_samples=2000,
    seed=SEED,
)

print(f"Loading {MODEL_NAME} (4-bit, bfloat16)...")
model, tokenizer = load_model(exp_config)

from lib.kv_cache import _get_text_config, _get_head_dim, _ensure_dynamic_cache, _get_cache_keys

text_config = _get_text_config(model.config)
N_LAYERS = text_config.num_hidden_layers
print(f"\\nModel loaded successfully.")
print(f"  Num layers: {N_LAYERS}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  Model dtype: {model.dtype}")
print(f"  Sliding window: {getattr(text_config, 'sliding_window', 'N/A')}")

# Verify with test forward pass
sample_ids = tokenizer("test", return_tensors="pt")['input_ids'].to(exp_config.device)
with torch.no_grad():
    out = model(sample_ids, use_cache=True)
    cache_check = _ensure_dynamic_cache(out.past_key_values)
    k0 = _get_cache_keys(cache_check, 0)
    print(f"  Cache key dtype: {k0.dtype}")
    print(f"  Cache key shape: {k0.shape}")
del out, sample_ids, cache_check
torch.cuda.empty_cache()\
""")))

# ========== Cell 3: Lib imports + constants ==========
cells.append(make_cell("code", s("""\
# Cell 3: Lib imports + constants
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
    replace_values_at_layers,
)
from lib.analysis import cohens_d
from lib.data import count_words
from scipy import stats
from tqdm.auto import tqdm

# Templates -- bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuestion: {question}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Prefix text
from lib.surrogate import STATIC_SURROGATE_QUERIES
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Experiment parameters
N_PER_DATASET = 300
MAX_DOC_TOKENS = 900
MAX_PASSAGE_WORDS = 250
MIN_PASSAGE_WORDS = 30
CHECKPOINT_EVERY = 25
MAX_GEN_TOKENS = 50

# Conditions (same 4 as Exp 30)
CONDITION_NAMES = ['bare', 'sf_trunc', 'values_early', 'values_hero']

# Layer-selective conditions from Exps 19/21/24
EARLY_LAYER_CUTOFF = 16  # layers 0-15
HERO_LAYERS = [10, 12, 14, 15, 20]  # from Exp 24 single-layer scan

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  N per dataset: {N_PER_DATASET}")
print(f"  MAX_DOC_TOKENS: {MAX_DOC_TOKENS} (sliding window constraint)")
print(f"  MAX_PASSAGE_WORDS: {MAX_PASSAGE_WORDS}")
print(f"  MAX_GEN_TOKENS: {MAX_GEN_TOKENS}")
print(f"  N_LAYERS: {N_LAYERS}")
print(f"  EARLY_LAYER_CUTOFF: {EARLY_LAYER_CUTOFF}")
print(f"  HERO_LAYERS: {HERO_LAYERS}")
print(f"  Conditions: {CONDITION_NAMES}")
print(f"  Static fact prefix: '{STATIC_FACT}'")\
""")))

# ========== Cell 4: Load Amazon SubjQA ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load Amazon SubjQA (electronics + grocery)
from datasets import load_dataset

print("=" * 70)
print("LOADING AMAZON SubjQA (electronics + grocery)")
print("=" * 70)
print("Product review QA. Commercial content closest to ad-serving use case.")

SUBJQA_CACHE = RESULTS_DIR / "subjqa_samples.json"

if SUBJQA_CACHE.exists():
    with open(SUBJQA_CACHE, 'r') as f:
        subjqa_samples = json.load(f)
    print(f"Loaded {len(subjqa_samples)} cached SubjQA samples")
else:
    subjqa_samples = []
    try:
        # datasets v4.5+ dropped loading scripts; use parquet branch
        ds = load_dataset("megagonlabs/subjqa", revision="refs/convert/parquet")
        print(f"  Loaded SubjQA from parquet branch ({list(ds.keys())} splits)")
    except Exception as e:
        print(f"  FAILED to load SubjQA: {e}")
        ds = None

    if ds is not None:
        for domain in ['electronics', 'grocery']:
            n_domain = 0
            for split_name in ds:
                for item in ds[split_name]:
                    if item.get('domain', '') != domain:
                        continue
                    context = item.get('context', '')
                    question = item.get('question', '')
                    answers = item.get('answers', {})
                    answer_texts = answers.get('text', [])
                    if not answer_texts or not question or not context:
                        continue
                    answer_text = answer_texts[0]
                    if not answer_text.strip():
                        continue
                    wc = count_words(context)
                    if MIN_PASSAGE_WORDS <= wc <= MAX_PASSAGE_WORDS:
                        subjqa_samples.append({
                            'passage': context,
                            'query': question,
                            'answer': answer_text,
                            'word_count': wc,
                            'dataset': 'subjqa',
                            'domain': domain,
                        })
                        n_domain += 1
            print(f"  {domain}: {n_domain} samples")

    # Deduplicate by question text
    seen_queries = set()
    unique_samples = []
    for sample in subjqa_samples:
        if sample['query'] not in seen_queries:
            seen_queries.add(sample['query'])
            unique_samples.append(sample)
    subjqa_samples = unique_samples
    print(f"  After dedup: {len(subjqa_samples)} unique samples")

    np.random.seed(SEED)
    np.random.shuffle(subjqa_samples)
    subjqa_samples = subjqa_samples[:N_PER_DATASET]

    with open(SUBJQA_CACHE, 'w') as f:
        json.dump(subjqa_samples, f)
    print(f"Cached {len(subjqa_samples)} samples")

print(f"\\nSubjQA samples: {len(subjqa_samples)}")
if subjqa_samples:
    wcs = [s['word_count'] for s in subjqa_samples]
    domains = [s.get('domain', '?') for s in subjqa_samples]
    domain_counts = Counter(domains)
    print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
    print(f"  Domain distribution: {dict(domain_counts)}")
    ans_lens = [len(s['answer'].split()) for s in subjqa_samples]
    print(f"  Answer word lengths: mean={np.mean(ans_lens):.1f}, max={max(ans_lens)}")
    for i in range(min(3, len(subjqa_samples))):
        print(f"  Example {i+1}:")
        print(f"    Q: {subjqa_samples[i]['query']}")
        print(f"    A: {subjqa_samples[i]['answer']}")
        print(f"    Context (first 120 chars): {subjqa_samples[i]['passage'][:120]}...")
else:
    print("  WARNING: No SubjQA samples loaded! Check dataset availability.")\
""")))

# ========== Cell 5: Load MS MARCO short ==========
cells.append(make_cell("code", s("""\
# Cell 5: Load MS MARCO short passages (<250 words)
print("=" * 70)
print("LOADING MS MARCO (validation, short passages only)")
print("=" * 70)
print(f"Short web passages (<{MAX_PASSAGE_WORDS} words). Known positive control for priming.")

MARCO_CACHE = RESULTS_DIR / "marco_samples.json"

if MARCO_CACHE.exists():
    with open(MARCO_CACHE, 'r') as f:
        marco_samples = json.load(f)
    print(f"Loaded {len(marco_samples)} cached MS MARCO samples")
else:
    marco_ds = load_dataset("ms_marco", "v1.1", split="validation", trust_remote_code=True)
    print(f"MS MARCO validation size: {len(marco_ds)}")

    marco_samples = []
    np.random.seed(SEED)

    for item in tqdm(marco_ds, desc="Filtering MARCO"):
        passages = item.get('passages', {})
        passage_texts = passages.get('passage_text', [])
        is_selected = passages.get('is_selected', [])
        query = item.get('query', '')
        answers = item.get('answers', [])
        well_formed = item.get('wellFormedAnswers', [])

        if not passage_texts or not query:
            continue

        # Get best answer
        answer = None
        if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
            answer = well_formed[0]
        elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
            answer = answers[0]
        else:
            continue

        # Find short selected passage
        for i, passage in enumerate(passage_texts):
            wc = count_words(passage)
            if MIN_PASSAGE_WORDS <= wc <= MAX_PASSAGE_WORDS:
                if is_selected and i < len(is_selected) and is_selected[i] == 1:
                    marco_samples.append({
                        'passage': passage,
                        'query': query,
                        'answer': answer,
                        'word_count': wc,
                        'dataset': 'marco',
                    })
                    break

        if len(marco_samples) >= N_PER_DATASET * 3:
            break

    np.random.shuffle(marco_samples)
    marco_samples = marco_samples[:N_PER_DATASET]

    with open(MARCO_CACHE, 'w') as f:
        json.dump(marco_samples, f)
    print(f"Cached {len(marco_samples)} samples")

    del marco_ds
    gc.collect()

print(f"\\nMS MARCO samples: {len(marco_samples)}")
if marco_samples:
    wcs = [s['word_count'] for s in marco_samples]
    ans_lens = [len(s['answer'].split()) for s in marco_samples]
    print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
    print(f"  Answer word lengths: mean={np.mean(ans_lens):.1f}, max={max(ans_lens)}")
    for i in range(min(3, len(marco_samples))):
        print(f"  Example {i+1}:")
        print(f"    Q: {marco_samples[i]['query']}")
        print(f"    A: {marco_samples[i]['answer']}")
        print(f"    Passage (first 120 chars): {marco_samples[i]['passage'][:120]}...")\
""")))

# ========== Cell 6: Load SQuAD short ==========
cells.append(make_cell("code", s("""\
# Cell 6: Load SQuAD short passages (<250 words)
print("=" * 70)
print("LOADING SQuAD (validation, short passages only)")
print("=" * 70)
print(f"Short Wikipedia paragraphs (<{MAX_PASSAGE_WORDS} words). Clean extractive QA baseline.")

SQUAD_CACHE = RESULTS_DIR / "squad_samples.json"

if SQUAD_CACHE.exists():
    with open(SQUAD_CACHE, 'r') as f:
        squad_samples = json.load(f)
    print(f"Loaded {len(squad_samples)} cached SQuAD samples")
else:
    squad_ds = load_dataset("squad", split="validation")
    print(f"SQuAD validation size: {len(squad_ds)}")

    squad_samples = []
    np.random.seed(SEED)

    for item in tqdm(squad_ds, desc="Filtering SQuAD"):
        context = item.get('context', '')
        question = item.get('question', '')
        answers = item.get('answers', {})
        answer_texts = answers.get('text', [])

        if not answer_texts or not question or not context:
            continue

        answer_text = answer_texts[0]
        if not answer_text.strip():
            continue

        wc = count_words(context)
        if MIN_PASSAGE_WORDS <= wc <= MAX_PASSAGE_WORDS:
            squad_samples.append({
                'passage': context,
                'query': question,
                'answer': answer_text,
                'word_count': wc,
                'dataset': 'squad',
            })

    np.random.shuffle(squad_samples)
    squad_samples = squad_samples[:N_PER_DATASET]

    with open(SQUAD_CACHE, 'w') as f:
        json.dump(squad_samples, f)
    print(f"Cached {len(squad_samples)} samples")

    del squad_ds
    gc.collect()

print(f"\\nSQuAD samples: {len(squad_samples)}")
if squad_samples:
    wcs = [s['word_count'] for s in squad_samples]
    ans_lens = [len(s['answer'].split()) for s in squad_samples]
    print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
    print(f"  Answer word lengths: mean={np.mean(ans_lens):.1f}, max={max(ans_lens)}")
    for i in range(min(3, len(squad_samples))):
        print(f"  Example {i+1}:")
        print(f"    Q: {squad_samples[i]['query']}")
        print(f"    A: {squad_samples[i]['answer']}")
        print(f"    Passage (first 120 chars): {squad_samples[i]['passage'][:120]}...")\
""")))

# ========== Cell 7: Unified pool + pre-screening ==========
cells.append(make_cell("code", s("""\
# Cell 7: Unified sample pool + tokenization + pre-screening
print("=" * 70)
print("UNIFIED SAMPLE POOL")
print("=" * 70)

all_samples = []
for ds_name, ds_samples in [("subjqa", subjqa_samples),
                              ("marco", marco_samples),
                              ("squad", squad_samples)]:
    for sample in ds_samples:
        sample['dataset'] = ds_name
    all_samples.extend(ds_samples)

print(f"Total samples: {len(all_samples)}")
for ds_name in ['subjqa', 'marco', 'squad']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
    if not ds_s:
        print(f"  {ds_name}: n=0 (MISSING)")
        continue
    wcs = [s['word_count'] for s in ds_s]
    print(f"  {ds_name}: n={len(ds_s)}, mean_words={np.mean(wcs):.0f}, "
          f"range=[{min(wcs)}, {max(wcs)}]")

# Tokenize prefix
sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(exp_config.device)
PREFIX_TOKEN_LEN = sf_ids.shape[1]

print(f"\\nPrefix: '{STATIC_FACT}'")
print(f"  Token length (no BOS): {PREFIX_TOKEN_LEN}")

# Verify sliding window safety
max_primed_seq = 1 + PREFIX_TOKEN_LEN + MAX_DOC_TOKENS
print(f"  Max primed sequence: 1 + {PREFIX_TOKEN_LEN} + {MAX_DOC_TOKENS} = {max_primed_seq}")
print(f"  Sliding window: 1024")
assert max_primed_seq < 1024, f"UNSAFE: {max_primed_seq} >= 1024"
print(f"  SAFE: {max_primed_seq} < 1024")

# Tokenize doc lengths
print(f"\\nTokenizing documents to measure token lengths...")
n_truncated = 0
for sample in tqdm(all_samples, desc="Tokenizing"):
    tok_len = len(tokenizer.encode(sample['passage'], add_special_tokens=False))
    if tok_len > MAX_DOC_TOKENS:
        n_truncated += 1
    sample['doc_token_len'] = min(tok_len, MAX_DOC_TOKENS)
    sample['answer_token_len'] = len(tokenizer.encode(sample['answer'], add_special_tokens=False))

print(f"  Documents truncated to {MAX_DOC_TOKENS}: {n_truncated}/{len(all_samples)} "
      f"({100*n_truncated/len(all_samples):.0f}%)")

for ds_name in ['subjqa', 'marco', 'squad']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
    if not ds_s:
        continue
    tls = [s['doc_token_len'] for s in ds_s]
    atls = [s['answer_token_len'] for s in ds_s]
    n_trunc = sum(1 for s in ds_s if s['doc_token_len'] == MAX_DOC_TOKENS)
    print(f"  {ds_name}: mean_tok={np.mean(tls):.0f}, median={np.median(tls):.0f}, "
          f"truncated={n_trunc}/{len(ds_s)} ({100*n_trunc/len(ds_s):.0f}%), "
          f"mean_ans_tok={np.mean(atls):.1f}")

# === PRE-SCREENING: Bare NLL check ===
print(f"\\n{'='*70}")
print("PRE-SCREENING: Bare NLL distribution check (20 samples/dataset)")
print("If median bare NLL < 0.05, ceiling effects may dominate.")
print("=" * 70)

for ds_name in ['subjqa', 'marco', 'squad']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name][:20]
    if not ds_s:
        continue
    bare_nlls = []
    for sample in ds_s:
        passage = sample['passage']
        question = sample['query']
        answer = sample['answer']

        document_text = DOCUMENT_TEMPLATE.format(document=passage)
        query_prompt = QUERY_TEMPLATE.format(question=question)
        answer_text = ANSWER_TEMPLATE.format(answer=answer)

        doc_ids = tokenizer(document_text, return_tensors="pt",
                            add_special_tokens=False)['input_ids'].to(exp_config.device)
        if doc_ids.shape[1] > MAX_DOC_TOKENS:
            doc_ids = doc_ids[:, :MAX_DOC_TOKENS]

        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.encode("", add_special_tokens=True)[0]
        bos_tensor = torch.tensor([[bos_id]], device=exp_config.device)
        bare_input = torch.cat([bos_tensor, doc_ids], dim=1)
        context_len = bare_input.shape[1]

        with torch.no_grad():
            bare_out = model(input_ids=bare_input,
                             attention_mask=torch.ones_like(bare_input),
                             use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
        del bare_out

        nll = score_answer_with_cache(
            deepcopy_cache(bare_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        bare_nlls.append(nll)
        del bare_cache, bare_input, doc_ids
        torch.cuda.empty_cache()

    bare_arr = np.array(bare_nlls)
    pct_floor = 100 * np.mean(bare_arr < 0.01)
    median = np.median(bare_arr)
    mean = np.mean(bare_arr)
    status = "WARNING: CEILING" if pct_floor > 50 else "OK" if pct_floor < 30 else "MARGINAL"
    print(f"  {ds_name:15s}: median={median:.3f}, mean={mean:.3f}, "
          f"pct_floor(<0.01)={pct_floor:.0f}% -> {status}")

print("\\nPre-screening complete. Proceeding with full experiment.")

# Condition explanation
print("\\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS (Gemma 3 4B) -- 4 conditions")
print("=" * 70)

print("\\n### 1. bare ###")
print("  Forward: [BOS][doc]")
print("  Baseline. Standard causal attention.")

print("\\n### 2. sf_trunc (standard priming) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc]")
print("  Standard causal, truncate + RoPE. Keys carry negative interference on Gemma.")

print("\\n### 3. values_early (layers 0-15 only) ###")
print("  Bare keys + primed values from layers 0-15 only.")
print("  Expected: d ~ +0.211 (Exp 19 on MARCO). Late layers carry interference.")

print("\\n### 4. values_hero (layers {10,12,14,15,20}) ###")
print("  Bare keys + primed values from 5 hero layers identified in Exp 24.")
print("  NQ: d=+0.213 (Exp 27b). DROP: d=-0.152 (Exp 29).")\
""")))

# ========== Cell 8: NLL helper function ==========
cells.append(make_cell("code", s("""\
# Cell 8: Helper function -- run_single_sample_4cond()

def run_single_sample_4cond(sample, model, tokenizer, exp_config, sf_ids, sf_str,
                             PREFIX_TOKEN_LEN, N_LAYERS, EARLY_LAYER_CUTOFF, HERO_LAYERS):
    \"\"\"Run 4 conditions for a single sample. Returns dict of NLLs + metadata.

    Conditions:
      1. bare: [BOS][doc] standard causal
      2. sf_trunc: [BOS][prefix][doc] truncate + RoPE correct
      3. values_early: bare keys + primed values layers 0-15
      4. values_hero: bare keys + primed values at hero layers
    \"\"\"
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    ds_name = sample['dataset']

    query_prompt = QUERY_TEMPLATE.format(question=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # === Matched tokenization ===
    full_text = sf_str + document_text
    full_enc = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=True, padding=False, truncation=False)
    full_ids = full_enc['input_ids'].to(exp_config.device)

    sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                               add_special_tokens=True, padding=False, truncation=False)
    sf_prefix_len_with_bos = sf_prefix_enc['input_ids'].shape[1]

    bos_id = full_ids[:, :1]
    doc_ids = full_ids[:, sf_prefix_len_with_bos:]

    # Truncate long docs
    if doc_ids.shape[1] > MAX_DOC_TOKENS:
        doc_ids = doc_ids[:, :MAX_DOC_TOKENS]

    doc_len = doc_ids.shape[1]
    context_len = 1 + doc_len  # BOS + doc

    del full_enc, full_ids, sf_prefix_enc

    # === 1. BARE ===
    bare_input = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_input,
                         attention_mask=torch.ones_like(bare_input),
                         use_cache=True, return_dict=True)
    bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
    del bare_out

    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    # === 2. sf_trunc (standard priming) ===
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    prefix_offset = sf_ids.shape[1]

    with torch.no_grad():
        primed_out = model(input_ids=primed_input,
                           attention_mask=torch.ones_like(primed_input),
                           use_cache=True, return_dict=True)
    primed_full_std = _ensure_dynamic_cache(primed_out.past_key_values)
    del primed_out

    trunc_raw = extract_and_truncate_cache_with_bos(primed_full_std, doc_len)
    del primed_full_std

    sf_trunc_cache = deepcopy_cache(trunc_raw)
    correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
    del trunc_raw

    sf_trunc_nll = score_answer_with_cache(
        deepcopy_cache(sf_trunc_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    # === 3. values_early (layers 0 to EARLY_LAYER_CUTOFF-1) ===
    early_layers = list(range(EARLY_LAYER_CUTOFF))
    values_early_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, early_layers)

    values_early_nll = score_answer_with_cache(
        deepcopy_cache(values_early_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del values_early_cache

    # === 4. values_hero (hero layers only) ===
    values_hero_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, HERO_LAYERS)

    values_hero_nll = score_answer_with_cache(
        deepcopy_cache(values_hero_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del values_hero_cache

    del bare_cache, sf_trunc_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        'dataset': ds_name,
        'query': query,
        'answer': answer,
        'word_count': sample['word_count'],
        'doc_token_len': doc_len,
        'answer_token_len': sample.get('answer_token_len', 0),
        'bare': bare_nll,
        'sf_trunc': sf_trunc_nll,
        'values_early': values_early_nll,
        'values_hero': values_hero_nll,
    }
    if 'domain' in sample:
        result['domain'] = sample['domain']
    return result


print("Helper function defined: run_single_sample_4cond()")
print("  Conditions: bare, sf_trunc, values_early, values_hero")\
""")))

# ========== Cell 9: Main NLL experiment loop ==========
cells.append(make_cell("code", s("""\
# Cell 9: Main NLL experiment loop (Phase 1)

print("=" * 70)
print(f"PHASE 1: NLL EVALUATION -- {len(all_samples)} samples, {len(CONDITION_NAMES)} conditions")
print(f"Model: Gemma 3 4B, MAX_DOC_TOKENS: {MAX_DOC_TOKENS}")
print(f"Datasets: SubjQA, MS MARCO, SQuAD")
print("=" * 70)

# Checkpoint resume
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in all_samples]
    if ckpt_queries == current_queries:
        all_results = ckpt['results']
        start_idx = len(all_results)
        print(f"Resuming from checkpoint: {start_idx}/{len(all_samples)}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

t_start = time.time()
N_TOTAL = len(all_samples)

for qidx in tqdm(range(start_idx, N_TOTAL), initial=start_idx, total=N_TOTAL,
                  desc="Exp 31 Phase 1"):
    sample = all_samples[qidx]

    result = run_single_sample_4cond(
        sample, model, tokenizer, exp_config,
        sf_ids, sf_str, PREFIX_TOKEN_LEN, N_LAYERS,
        EARLY_LAYER_CUTOFF, HERO_LAYERS)
    result['query_idx'] = qidx
    all_results.append(result)

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_TOTAL - 1:
        ckpt_data = {
            'results': all_results,
            'sample_queries': [s['query'] for s in all_samples],
            'completed': len(all_results),
            'total': N_TOTAL,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_TOTAL - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_TOTAL} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nPhase 1 complete: {len(all_results)} samples in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 10: Per-dataset NLL analysis ==========
cells.append(make_cell("code", s("""\
# Cell 10: Per-dataset NLL results table (Phase 1 analysis)
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("PHASE 1 ANALYSIS: PER-DATASET NLL RESULTS (Gemma 3 4B)")
print("=" * 70)

dataset_names = ['subjqa', 'marco', 'squad']
analysis = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    n_ds = len(ds_results)
    if n_ds == 0:
        print(f"\\n  {ds_name}: NO RESULTS")
        continue

    bare_arr = np.array([r['bare'] for r in ds_results])

    # Filter invalid
    valid = np.isfinite(bare_arr)
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        c_arr = np.array([r[cname] for r in ds_results])
        valid &= np.isfinite(c_arr)

    n_valid = int(np.sum(valid))

    print(f"\\n{'='*70}")
    pct_floor = 100 * np.mean(bare_arr < 0.01)
    print(f"DATASET: {ds_name.upper()} (n={n_valid}/{n_ds}, "
          f"median bare NLL={np.median(bare_arr):.3f}, "
          f"pct_floor={pct_floor:.0f}%)")
    print(f"{'='*70}")

    print(f"\\n{'Condition':<20} {'Mean Bare':>10} {'Mean Cond':>10} "
          f"{'Mean D':>10} {'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
    print("-" * 90)

    ds_analysis = {}
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        c_arr = np.array([r[cname] for r in ds_results])
        delta = bare_arr[valid] - c_arr[valid]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        t_stat, p_val = stats.ttest_1samp(delta, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"{cname:<20} {np.mean(bare_arr[valid]):>10.4f} {np.mean(c_arr[valid]):>10.4f} "
              f"{np.mean(delta):>+10.4f} {d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
        ds_analysis[cname] = {
            'n_valid': n_valid,
            'mean_bare': float(np.mean(bare_arr[valid])),
            'mean_cond': float(np.mean(c_arr[valid])),
            'mean_delta': float(np.mean(delta)),
            'cohens_d': float(d),
            'win_pct': float(win),
            't_stat': float(t_stat),
            'p_value': float(p_val),
        }

    analysis[ds_name] = ds_analysis

# Cross-dataset summary table
print(f"\\n\\n{'='*90}")
print("CROSS-DATASET SUMMARY: Cohen's d vs bare (Gemma 3 4B)")
print(f"{'='*90}")
print(f"\\n{'Condition':<20}", end='')
for ds in dataset_names:
    print(f"{'  ' + ds:>16}", end='')
print()
print("-" * 68)
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    print(f"{cname:<20}", end='')
    for ds in dataset_names:
        if ds in analysis and cname in analysis[ds]:
            d = analysis[ds][cname]['cohens_d']
            p = analysis[ds][cname]['p_value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"{d:>+12.3f}{sig:>4}", end='')
        else:
            print(f"{'n/a':>16}", end='')
    print()

# Bare NLL distributions
print(f"\\n\\nBARE NLL DISTRIBUTIONS (ceiling effect check):")
for ds in dataset_names:
    ds_r = [r for r in all_results if r['dataset'] == ds]
    if not ds_r:
        continue
    bare = [r['bare'] for r in ds_r]
    pct_zero = 100 * np.mean(np.array(bare) < 0.01)
    iqr = np.percentile(bare, 75) - np.percentile(bare, 25)
    print(f"  {ds:15s}: mean={np.mean(bare):.3f}, median={np.median(bare):.3f}, "
          f"IQR={iqr:.3f}, pct_floor={pct_zero:.0f}%")

# Comparison with prior experiments
print(f"\\n\\n{'='*90}")
print("COMPARISON WITH PRIOR EXPERIMENTS")
print(f"{'='*90}")
print("\\nExp 07 (Mistral, MS MARCO): static_fact_trunc d=+0.472***")
print("Exp 27b (Gemma, NQ): values_hero d=+0.213***")
print("Exp 30 (Gemma, NQ): values_hero d=+0.213***")
print("Exp 29 (Gemma, DROP): values_hero d=-0.152**")
print("Exp 24 (Gemma, MARCO): values_cutoff_16 d=+0.211***")\
""")))

# ========== Cell 11: Generation helpers + metrics ==========
cells.append(make_cell("code", s("""\
# Cell 11: Generation helper functions + evaluation metrics

def normalize_answer(s):
    \"\"\"Lower text, remove punctuation, articles, and extra whitespace.\"\"\"
    def remove_articles(text):
        return re.sub(r'\\b(a|an|the)\\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, gold):
    \"\"\"Check if normalized gold appears in normalized prediction (substring match).\"\"\"
    return normalize_answer(gold) in normalize_answer(prediction)


def compute_token_f1(prediction, gold):
    \"\"\"Token-level F1 between prediction and gold (SQuAD-style).\"\"\"
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction, gold):
    \"\"\"ROUGE-L F-score based on longest common subsequence.\"\"\"
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def generate_with_cache(cache, context_len, query_prompt, model, tokenizer, config,
                         max_new_tokens=50):
    \"\"\"Generate text autoregressively using a pre-built KV cache.

    WARNING: Mutates the cache. Always deepcopy before calling.

    Args:
        cache: Pre-built KV cache (DynamicCache)
        context_len: Number of tokens in the cache
        query_prompt: Query string to feed before generating
        model: The language model
        tokenizer: The tokenizer
        config: ExperimentConfig
        max_new_tokens: Max tokens to generate

    Returns:
        Tuple of (generated_text, mean_log_prob)
    \"\"\"
    query_ids = tokenizer(query_prompt, return_tensors="pt",
                          add_special_tokens=False)['input_ids'].to(config.device)
    query_len = query_ids.shape[1]

    # Feed query through model to extend cache
    combined_len = context_len + query_len
    attention_mask = torch.ones((1, combined_len), device=config.device)

    with torch.no_grad():
        query_out = model(input_ids=query_ids,
                          attention_mask=attention_mask,
                          past_key_values=cache,
                          use_cache=True, return_dict=True)
    gen_cache = query_out.past_key_values
    next_logits = query_out.logits[:, -1:, :]  # (1, 1, vocab)
    cur_len = combined_len

    generated_ids = []
    log_probs = []

    for step in range(max_new_tokens):
        log_p = torch.log_softmax(next_logits[:, 0, :], dim=-1)
        next_id = torch.argmax(log_p, dim=-1)  # (1,)
        token_log_prob = log_p[0, next_id[0]].item()

        generated_ids.append(next_id[0].item())
        log_probs.append(token_log_prob)

        # Stop at EOS
        if next_id[0].item() == tokenizer.eos_token_id:
            break

        cur_len += 1
        attention_mask = torch.ones((1, cur_len), device=config.device)
        next_token = next_id.unsqueeze(0)  # (1, 1)

        with torch.no_grad():
            out = model(input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=gen_cache,
                        use_cache=True, return_dict=True)
        gen_cache = out.past_key_values
        next_logits = out.logits

    del gen_cache
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    mean_log_prob = float(np.mean(log_probs)) if log_probs else float('-inf')

    return generated_text, mean_log_prob


# Quick test
print("Generation helpers defined:")
print("  normalize_answer(), compute_exact_match(), compute_token_f1(), compute_rouge_l()")
print("  generate_with_cache(cache, context_len, query, model, tokenizer, config)")
print()
print("Test metrics:")
print(f"  EM('Paris is the capital', 'Paris'): {compute_exact_match('Paris is the capital', 'Paris')}")
print(f"  F1('paris is capital city', 'paris capital'): {compute_token_f1('paris is capital city', 'paris capital'):.3f}")
print(f"  ROUGE-L('the cat sat', 'the cat'): {compute_rouge_l('the cat sat', 'the cat'):.3f}")\
""")))

# ========== Cell 12: Generation experiment ==========
cells.append(make_cell("code", s("""\
# Cell 12: Generation experiment on hard subset (Phase 2)

print("=" * 70)
print("PHASE 2: GENERATION QUALITY EVALUATION")
print("=" * 70)
print("Generate with bare and hero caches on hard samples (top 50% by bare NLL).")
print(f"Max generation tokens: {MAX_GEN_TOKENS}")

# Select hard samples: top 50% by bare NLL per dataset
gen_sample_indices = []
for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if not ds_results:
        continue
    bare_nlls = np.array([r['bare'] for r in ds_results])
    threshold = np.median(bare_nlls)
    hard_indices = [r['query_idx'] for r in ds_results if r['bare'] >= threshold]
    gen_sample_indices.extend(hard_indices)
    print(f"  {ds_name}: {len(hard_indices)} hard samples (threshold={threshold:.3f})")

# Map query_idx -> Phase 1 result for later analysis
results_by_idx = {r['query_idx']: r for r in all_results}

print(f"\\nTotal generation samples: {len(gen_sample_indices)}")

# Checkpoint resume
gen_results = []
gen_start_idx = 0

if GEN_CHECKPOINT_PATH.exists():
    with open(GEN_CHECKPOINT_PATH, 'r') as f:
        gen_ckpt = json.load(f)
    if gen_ckpt.get('gen_indices') == gen_sample_indices:
        gen_results = gen_ckpt['gen_results']
        gen_start_idx = len(gen_results)
        print(f"Resuming generation from checkpoint: {gen_start_idx}/{len(gen_sample_indices)}")
    else:
        print("Generation checkpoint mismatch. Starting fresh.")

t_gen_start = time.time()

for gidx in tqdm(range(gen_start_idx, len(gen_sample_indices)),
                  initial=gen_start_idx, total=len(gen_sample_indices),
                  desc="Exp 31 Phase 2"):
    qidx = gen_sample_indices[gidx]
    sample = all_samples[qidx]
    phase1 = results_by_idx[qidx]

    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    query_prompt = QUERY_TEMPLATE.format(question=query)
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # Matched tokenization (same as Phase 1)
    full_text = sf_str + document_text
    full_enc = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=True, padding=False, truncation=False)
    full_ids = full_enc['input_ids'].to(exp_config.device)

    sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                               add_special_tokens=True, padding=False, truncation=False)
    sf_prefix_len_with_bos = sf_prefix_enc['input_ids'].shape[1]

    bos_id = full_ids[:, :1]
    doc_ids = full_ids[:, sf_prefix_len_with_bos:]
    if doc_ids.shape[1] > MAX_DOC_TOKENS:
        doc_ids = doc_ids[:, :MAX_DOC_TOKENS]
    doc_len = doc_ids.shape[1]
    context_len = 1 + doc_len

    del full_enc, full_ids, sf_prefix_enc

    # Build bare cache
    bare_input = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_input,
                         attention_mask=torch.ones_like(bare_input),
                         use_cache=True, return_dict=True)
    bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
    del bare_out

    # Build primed cache -> truncate -> RoPE -> hero
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    prefix_offset = sf_ids.shape[1]
    with torch.no_grad():
        primed_out = model(input_ids=primed_input,
                           attention_mask=torch.ones_like(primed_input),
                           use_cache=True, return_dict=True)
    primed_cache = _ensure_dynamic_cache(primed_out.past_key_values)
    del primed_out

    trunc_raw = extract_and_truncate_cache_with_bos(primed_cache, doc_len)
    del primed_cache
    sf_trunc_cache = deepcopy_cache(trunc_raw)
    correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
    del trunc_raw

    hero_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, HERO_LAYERS)
    del sf_trunc_cache

    # Generate with bare
    bare_text, bare_lp = generate_with_cache(
        deepcopy_cache(bare_cache), context_len, query_prompt,
        model, tokenizer, exp_config, MAX_GEN_TOKENS)

    # Generate with hero
    hero_text, hero_lp = generate_with_cache(
        deepcopy_cache(hero_cache), context_len, query_prompt,
        model, tokenizer, exp_config, MAX_GEN_TOKENS)

    del bare_cache, hero_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate
    gen_result = {
        'query_idx': qidx,
        'dataset': sample['dataset'],
        'query': query,
        'answer': answer,
        'word_count': sample['word_count'],
        'bare_nll': phase1['bare'],
        'hero_nll': phase1['values_hero'],
        'bare_text': bare_text,
        'hero_text': hero_text,
        'bare_log_prob': bare_lp,
        'hero_log_prob': hero_lp,
        'bare_em': compute_exact_match(bare_text, answer),
        'hero_em': compute_exact_match(hero_text, answer),
        'bare_f1': compute_token_f1(bare_text, answer),
        'hero_f1': compute_token_f1(hero_text, answer),
        'bare_rouge_l': compute_rouge_l(bare_text, answer),
        'hero_rouge_l': compute_rouge_l(hero_text, answer),
        'texts_differ': bare_text != hero_text,
    }
    gen_results.append(gen_result)

    # Checkpoint every 25
    if (gidx + 1) % CHECKPOINT_EVERY == 0 or gidx == len(gen_sample_indices) - 1:
        gen_ckpt_data = {
            'gen_results': gen_results,
            'gen_indices': gen_sample_indices,
            'completed': len(gen_results),
            'total': len(gen_sample_indices),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(GEN_CHECKPOINT_PATH, 'w') as f:
            json.dump(gen_ckpt_data, f)
        elapsed = time.time() - t_gen_start
        n_done = gidx - gen_start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (len(gen_sample_indices) - gidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Gen checkpoint {gidx+1}/{len(gen_sample_indices)} | "
                   f"{n_done} done in {elapsed/60:.1f}m | ETA: {remaining/60:.1f} min")

elapsed_gen = time.time() - t_gen_start
print(f"\\nPhase 2 complete: {len(gen_results)} generations in {elapsed_gen/60:.1f} min")\
""")))

# ========== Cell 13: Generation quality analysis ==========
cells.append(make_cell("code", s("""\
# Cell 13: Generation quality analysis

print("=" * 70)
print("PHASE 2 ANALYSIS: GENERATION QUALITY (bare vs hero)")
print("=" * 70)

# Overall metrics
n_gen = len(gen_results)
n_differ = sum(1 for r in gen_results if r['texts_differ'])
print(f"\\nGeneration samples: {n_gen}")
print(f"Texts differ (bare vs hero): {n_differ}/{n_gen} ({100*n_differ/n_gen:.0f}%)")

# Overall comparison
print(f"\\n{'Metric':<20} {'Bare':>10} {'Hero':>10} {'Delta':>10} {'p':>12} {'sig':>5}")
print("-" * 72)

gen_analysis = {}

for metric_name, bare_key, hero_key in [
    ('Exact Match', 'bare_em', 'hero_em'),
    ('Token F1', 'bare_f1', 'hero_f1'),
    ('ROUGE-L', 'bare_rouge_l', 'hero_rouge_l'),
    ('Log Prob', 'bare_log_prob', 'hero_log_prob'),
]:
    bare_vals = np.array([float(r[bare_key]) for r in gen_results])
    hero_vals = np.array([float(r[hero_key]) for r in gen_results])
    delta = hero_vals - bare_vals  # positive = hero better

    # Filter inf for log_prob
    valid = np.isfinite(bare_vals) & np.isfinite(hero_vals)
    if np.sum(valid) < 10:
        print(f"{metric_name:<20} {'n/a (too few valid)':>40}")
        continue

    bare_mean = np.mean(bare_vals[valid])
    hero_mean = np.mean(hero_vals[valid])
    delta_mean = np.mean(delta[valid])

    if np.std(delta[valid]) > 0:
        t_stat, p_val = stats.ttest_1samp(delta[valid], 0)
    else:
        t_stat, p_val = 0.0, 1.0

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{metric_name:<20} {bare_mean:>10.3f} {hero_mean:>10.3f} "
          f"{delta_mean:>+10.3f} {p_val:>12.2e} {sig:>5}")

    gen_analysis[metric_name] = {
        'bare_mean': float(bare_mean),
        'hero_mean': float(hero_mean),
        'delta_mean': float(delta_mean),
        'p_value': float(p_val),
        'n_valid': int(np.sum(valid)),
    }

# Per-dataset generation metrics
print(f"\\n\\n{'='*90}")
print("PER-DATASET GENERATION QUALITY")
print(f"{'='*90}")

gen_per_dataset = {}
for ds_name in dataset_names:
    ds_gen = [r for r in gen_results if r['dataset'] == ds_name]
    if not ds_gen:
        continue
    n = len(ds_gen)
    n_diff = sum(1 for r in ds_gen if r['texts_differ'])

    print(f"\\n--- {ds_name.upper()} (n={n}, {n_diff} differ) ---")
    print(f"{'Metric':<20} {'Bare':>10} {'Hero':>10} {'Delta':>10} {'p':>12}")
    print("-" * 65)

    ds_gen_analysis = {}
    for metric_name, bare_key, hero_key in [
        ('Exact Match', 'bare_em', 'hero_em'),
        ('Token F1', 'bare_f1', 'hero_f1'),
        ('ROUGE-L', 'bare_rouge_l', 'hero_rouge_l'),
    ]:
        bare_vals = np.array([float(r[bare_key]) for r in ds_gen])
        hero_vals = np.array([float(r[hero_key]) for r in ds_gen])
        delta = hero_vals - bare_vals
        valid = np.isfinite(delta)

        if np.sum(valid) < 5:
            continue

        bare_m = np.mean(bare_vals[valid])
        hero_m = np.mean(hero_vals[valid])
        delta_m = np.mean(delta[valid])
        if np.std(delta[valid]) > 0:
            _, p = stats.ttest_1samp(delta[valid], 0)
        else:
            p = 1.0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{metric_name:<20} {bare_m:>10.3f} {hero_m:>10.3f} "
              f"{delta_m:>+10.3f} {p:>12.2e} {sig}")

        ds_gen_analysis[metric_name] = {
            'bare_mean': float(bare_m),
            'hero_mean': float(hero_m),
            'delta_mean': float(delta_m),
            'p_value': float(p),
        }
    gen_per_dataset[ds_name] = ds_gen_analysis

# NLL vs Generation quality correlation
print(f"\\n\\n{'='*90}")
print("NLL IMPROVEMENT vs GENERATION QUALITY IMPROVEMENT")
print(f"{'='*90}")
print("Does lower hero NLL predict better hero generation?")

nll_deltas = []
f1_deltas = []
for r in gen_results:
    nll_delta = r['bare_nll'] - r['hero_nll']  # positive = hero lower NLL
    f1_delta = r['hero_f1'] - r['bare_f1']  # positive = hero better F1
    if np.isfinite(nll_delta) and np.isfinite(f1_delta):
        nll_deltas.append(nll_delta)
        f1_deltas.append(f1_delta)

if len(nll_deltas) >= 10:
    rho, p_rho = stats.spearmanr(nll_deltas, f1_deltas)
    print(f"  Spearman rho (NLL delta vs F1 delta): {rho:+.3f}, p={p_rho:.3f}")
    if p_rho < 0.05:
        print("  -> SIGNIFICANT: NLL improvement predicts generation quality improvement")
    else:
        print("  -> Not significant: NLL improvement does NOT predict generation quality")
else:
    rho = float('nan')
    print("  Too few valid samples for correlation")

# Example outputs: 10 most improved samples
print(f"\\n\\n{'='*90}")
print("EXAMPLE GENERATIONS: Top 10 Most Improved (hero F1 > bare F1)")
print(f"{'='*90}")
improved = sorted(gen_results,
                  key=lambda r: r['hero_f1'] - r['bare_f1'],
                  reverse=True)[:10]

for i, r in enumerate(improved):
    f1_gain = r['hero_f1'] - r['bare_f1']
    print(f"\\n--- Example {i+1} ({r['dataset']}, F1 gain: {f1_gain:+.3f}) ---")
    print(f"  Q: {r['query']}")
    print(f"  Gold: {r['answer']}")
    print(f"  Bare: {r['bare_text'][:100]}{'...' if len(r['bare_text']) > 100 else ''}")
    print(f"  Hero: {r['hero_text'][:100]}{'...' if len(r['hero_text']) > 100 else ''}")
    print(f"  Bare EM={r['bare_em']}, F1={r['bare_f1']:.3f} | "
          f"Hero EM={r['hero_em']}, F1={r['hero_f1']:.3f}")\
""")))

# ========== Cell 14: Multi-panel plots ==========
cells.append(make_cell("code", s("""\
# Cell 14: Multi-panel figure (2x3)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

colors_ds = {'subjqa': '#ff7f0e', 'marco': '#2ca02c', 'squad': '#1f77b4'}

# ---- Panel (a): Hero d by dataset (NLL) ----
ax = axes[0, 0]
ds_labels = []
ds_hero_ds = []
ds_colors = []
for ds in dataset_names:
    if ds not in analysis or 'values_hero' not in analysis[ds]:
        continue
    d_val = analysis[ds]['values_hero']['cohens_d']
    p_val = analysis[ds]['values_hero']['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    ds_labels.append(f"{ds.upper()}\\n{sig}")
    ds_hero_ds.append(d_val)
    ds_colors.append(colors_ds.get(ds, '#7f7f7f'))

if ds_hero_ds:
    bars = ax.bar(range(len(ds_hero_ds)), ds_hero_ds, color=ds_colors,
                  edgecolor='black', linewidth=0.5)
    for i, d_val in enumerate(ds_hero_ds):
        ax.text(i, d_val + (0.01 if d_val >= 0 else -0.03),
                f"{d_val:+.3f}", ha='center',
                va='bottom' if d_val >= 0 else 'top', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(ds_labels)))
    ax.set_xticklabels(ds_labels)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("(a) Hero Layer NLL Effect by Dataset")

# ---- Panel (b): All conditions comparison ----
ax = axes[0, 1]
cond_labels = ['sf_trunc', 'values_early', 'values_hero']
x_pos = np.arange(len(dataset_names))
width = 0.25
for ci, cname in enumerate(cond_labels):
    ds_vals = []
    for ds in dataset_names:
        if ds in analysis and cname in analysis[ds]:
            ds_vals.append(analysis[ds][cname]['cohens_d'])
        else:
            ds_vals.append(0)
    ax.bar(x_pos + ci * width, ds_vals, width, label=cname, alpha=0.8)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([ds.upper() for ds in dataset_names])
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d")
ax.set_title("(b) All Conditions by Dataset")
ax.legend(fontsize=8)

# ---- Panel (c): Bare NLL distributions ----
ax = axes[0, 2]
bare_by_ds = []
ds_labels_box = []
for ds in dataset_names:
    ds_r = [r for r in all_results if r['dataset'] == ds]
    if ds_r:
        bare_by_ds.append([r['bare'] for r in ds_r])
        pct_f = 100 * np.mean(np.array([r['bare'] for r in ds_r]) < 0.01)
        ds_labels_box.append(f"{ds.upper()}\\n({pct_f:.0f}% floor)")
if bare_by_ds:
    bp = ax.boxplot(bare_by_ds, labels=ds_labels_box, showfliers=False, patch_artist=True,
                    medianprops={'color': 'red', 'linewidth': 2})
    for patch, ds in zip(bp['boxes'], dataset_names):
        patch.set_facecolor(colors_ds.get(ds, '#8ecae6'))
        patch.set_alpha(0.7)
ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, label='Floor (0.01)')
ax.set_ylabel("Bare NLL")
ax.set_title("(c) Bare NLL Distributions")
ax.legend(fontsize=7)

# ---- Panel (d): Generation EM by dataset ----
ax = axes[1, 0]
if gen_results:
    gen_ds_labels = []
    bare_ems = []
    hero_ems = []
    for ds in dataset_names:
        ds_gen = [r for r in gen_results if r['dataset'] == ds]
        if ds_gen:
            gen_ds_labels.append(ds.upper())
            bare_ems.append(100 * np.mean([r['bare_em'] for r in ds_gen]))
            hero_ems.append(100 * np.mean([r['hero_em'] for r in ds_gen]))

    x = np.arange(len(gen_ds_labels))
    ax.bar(x - 0.15, bare_ems, 0.3, label='Bare', color='#7f7f7f', alpha=0.8)
    ax.bar(x + 0.15, hero_ems, 0.3, label='Hero', color='#d62728', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gen_ds_labels)
    ax.legend()
ax.set_ylabel("Exact Match (%)")
ax.set_title("(d) Generation Exact Match Rate")

# ---- Panel (e): Generation F1 by dataset ----
ax = axes[1, 1]
if gen_results:
    gen_ds_labels = []
    bare_f1s = []
    hero_f1s = []
    for ds in dataset_names:
        ds_gen = [r for r in gen_results if r['dataset'] == ds]
        if ds_gen:
            gen_ds_labels.append(ds.upper())
            bare_f1s.append(np.mean([r['bare_f1'] for r in ds_gen]))
            hero_f1s.append(np.mean([r['hero_f1'] for r in ds_gen]))

    x = np.arange(len(gen_ds_labels))
    ax.bar(x - 0.15, bare_f1s, 0.3, label='Bare', color='#7f7f7f', alpha=0.8)
    ax.bar(x + 0.15, hero_f1s, 0.3, label='Hero', color='#d62728', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gen_ds_labels)
    ax.legend()
ax.set_ylabel("Token F1")
ax.set_title("(e) Generation Token F1")

# ---- Panel (f): NLL delta vs F1 delta scatter ----
ax = axes[1, 2]
if gen_results:
    for ds in dataset_names:
        ds_gen = [r for r in gen_results if r['dataset'] == ds]
        if ds_gen:
            nll_d = [r['bare_nll'] - r['hero_nll'] for r in ds_gen]
            f1_d = [r['hero_f1'] - r['bare_f1'] for r in ds_gen]
            ax.scatter(nll_d, f1_d, alpha=0.3, s=15, label=ds.upper(),
                       color=colors_ds.get(ds, '#7f7f7f'))
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("NLL Delta (bare - hero, positive = hero lower)")
    ax.set_ylabel("F1 Delta (hero - bare, positive = hero better)")
    ax.set_title("(f) NLL Improvement vs F1 Improvement")
    ax.legend(fontsize=8)

plt.suptitle('Exp 31: Ad-Content Benchmark & Generation Quality (Gemma 3 4B)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 15: Save results + verdict ==========
cells.append(make_cell("code", s("""\
# Cell 15: Save results.json + CSV + verdict

# --- CSV (Phase 1) ---
with open(CSV_PATH, 'w', newline='') as f:
    fieldnames = ['query_idx', 'dataset', 'query', 'answer', 'word_count',
                  'doc_token_len', 'answer_token_len', 'domain',
                  'bare', 'sf_trunc', 'values_early', 'values_hero']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in all_results:
        writer.writerow({k: r.get(k, '') for k in fieldnames})
print(f"NLL CSV saved: {CSV_PATH}")

# --- CSV (Phase 2: Generation) ---
with open(GEN_CSV_PATH, 'w', newline='') as f:
    fieldnames = ['query_idx', 'dataset', 'query', 'answer', 'word_count',
                  'bare_nll', 'hero_nll', 'bare_text', 'hero_text',
                  'bare_em', 'hero_em', 'bare_f1', 'hero_f1',
                  'bare_rouge_l', 'hero_rouge_l', 'bare_log_prob', 'hero_log_prob',
                  'texts_differ']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in gen_results:
        writer.writerow({k: r.get(k, '') for k in fieldnames})
print(f"Generation CSV saved: {GEN_CSV_PATH}")

# --- Compute verdict inputs ---

# Hero d per dataset (Phase 1)
hero_ds = {}
for ds in dataset_names:
    if ds in analysis and 'values_hero' in analysis[ds]:
        hero_ds[ds] = analysis[ds]['values_hero']['cohens_d']

# Generation metrics (Phase 2)
gen_f1_delta = None
gen_em_delta = None
if gen_results:
    bare_f1s = [r['bare_f1'] for r in gen_results]
    hero_f1s = [r['hero_f1'] for r in gen_results]
    gen_f1_delta = float(np.mean(hero_f1s) - np.mean(bare_f1s))

    bare_ems = [float(r['bare_em']) for r in gen_results]
    hero_ems = [float(r['hero_em']) for r in gen_results]
    gen_em_delta = float(np.mean(hero_ems) - np.mean(bare_ems))

print(f"\\nVerdict inputs:")
for ds, d in hero_ds.items():
    p = analysis[ds]['values_hero']['p_value']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {ds} hero d: {d:+.3f} {sig}")
if gen_f1_delta is not None:
    print(f"  Generation F1 delta (hero - bare): {gen_f1_delta:+.3f}")
    print(f"  Generation EM delta (hero - bare): {gen_em_delta:+.3f}")

# --- Verdict ---
n_positive = sum(1 for d in hero_ds.values() if d > 0.1)
n_significant = sum(1 for ds, d in hero_ds.items()
                    if analysis[ds]['values_hero']['p_value'] < 0.05 and d > 0)

if n_positive >= 2 and gen_f1_delta is not None and gen_f1_delta > 0:
    verdict = (f"AD-CONTENT BENEFIT: Hero layers help NLL on {n_positive}/3 datasets "
               f"and generation F1 improves by {gen_f1_delta:+.3f}. "
               f"Operationalization for short ad-content is viable.")
elif n_positive >= 1 and gen_f1_delta is not None and gen_f1_delta > 0:
    verdict = (f"PARTIAL BENEFIT: Hero layers help NLL on {n_positive}/3 datasets "
               f"and generation F1 improves by {gen_f1_delta:+.3f}. "
               f"Benefits are dataset-specific.")
elif n_positive >= 1 and (gen_f1_delta is None or gen_f1_delta <= 0):
    verdict = (f"NLL-ONLY BENEFIT: Hero layers help NLL on {n_positive}/3 datasets "
               f"but generation quality does NOT improve "
               f"(F1 delta: {gen_f1_delta:+.3f if gen_f1_delta is not None else 'n/a'}). "
               f"NLL improvement does not translate to better answers.")
else:
    verdict = (f"NO BENEFIT: Hero layers do not help NLL on ad-content "
               f"({n_positive}/3 datasets with d>0.1). "
               f"Priming is not viable for this content type.")

print(f"\\n{'='*70}")
print(f"VERDICT: {verdict}")
print(f"{'='*70}")

# Updated hero scorecard
print(f"\\nUpdated hero scorecard (all experiments + this one):")
print("  MARCO (Mistral): d=+0.472*** (Exp 07)")
for ds, d in hero_ds.items():
    p = analysis[ds]['values_hero']['p_value']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {ds} (Exp 31): d={d:+.3f} {sig}")
print("  NQ (Exp 27b/30): d=+0.213***")
print("  TriviaQA (Exp 27b): d=+0.000 (ceiling)")
print("  DROP (Exp 29/30): d=-0.152** (hurts)")

# --- results.json ---
final = {
    'experiment': 'exp31_ad_benchmark_and_generation',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'n_per_dataset': N_PER_DATASET,
        'max_doc_tokens': MAX_DOC_TOKENS,
        'max_passage_words': MAX_PASSAGE_WORDS,
        'max_gen_tokens': MAX_GEN_TOKENS,
        'conditions': CONDITION_NAMES,
        'early_layer_cutoff': EARLY_LAYER_CUTOFF,
        'hero_layers': HERO_LAYERS,
        'prefix': STATIC_FACT,
        'prefix_token_len': PREFIX_TOKEN_LEN,
        'datasets': dataset_names,
    },
    'phase1_nll': {
        'per_dataset_analysis': analysis,
        'hero_ds': hero_ds,
    },
    'phase2_generation': {
        'overall': gen_analysis,
        'per_dataset': gen_per_dataset,
        'n_gen_samples': len(gen_results),
        'n_texts_differ': sum(1 for r in gen_results if r['texts_differ']),
        'f1_delta': gen_f1_delta,
        'em_delta': gen_em_delta,
        'nll_f1_correlation': {
            'spearman_rho': float(rho) if np.isfinite(rho) else None,
        },
    },
    'ceiling_status': {
        ds: float(np.mean(np.array([r['bare'] for r in all_results if r['dataset'] == ds]) < 0.01) * 100)
        for ds in dataset_names if any(r['dataset'] == ds for r in all_results)
    },
    'verdict': verdict,
    'per_sample_nll_results': all_results,
    'per_sample_gen_results': gen_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print(f"\\nDone!")\
""")))

# ========== Cell 16: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 16: GPU cleanup
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


# ========== Validate all code cells parse ==========
print("Validating code cells...")
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
            print(f"  Cell {i}: OK")
        except SyntaxError as e:
            print(f"  Cell {i}: SYNTAX ERROR - {e}")
            sys.exit(1)

print(f"\nAll {len(cells)} cells validated.")

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

output_path = "/home/jupyter/research/directed_kvcache_v2/31_ad_benchmark_and_generation.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
