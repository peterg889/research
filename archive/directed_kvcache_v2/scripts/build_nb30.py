#!/usr/bin/env python3
"""Build script for 30_retrieval_vs_reasoning.ipynb

Exp 30: Retrieval vs Reasoning Task-Type Dissociation (Gemma 3 4B)

Exp 29 showed hero layers significantly hurt on DROP (d=-0.152, p=0.009) but were
neutral on AdversarialQA/CoQA. The hypothesis is that priming helps *retrieval*
but not *reasoning/computation*. However, this is based on a single dataset per type.

This experiment runs three datasets to test whether task type predicts hero layer
effect beyond difficulty:

  1. NQ — Retrieval (factoid), known positive control (hero d=+0.213 in Exp 27b)
  2. DROP — Mixed (computation + extraction), known negative (hero d=-0.152 in Exp 29)
     Split by answer type (number vs span) for within-dataset comparison
  3. BoolQ — Retrieval (binary judgment), new data point

Conditions (reduced to 4 for efficiency):
  1. bare: BOS + doc standard causal (baseline)
  2. sf_trunc: Standard priming (truncate + RoPE correct)
  3. values_early: Bare keys + primed values layers 0-15
  4. values_hero: Bare keys + primed values at hero layers {10,12,14,15,20}

Model: Gemma 3 4B (4-bit, bfloat16)
N = 300 per dataset (900 total)
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
# Exp 30: Retrieval vs Reasoning Task-Type Dissociation (Gemma 3 4B)

## Motivation

Exp 29 showed hero layers significantly **hurt** on DROP (d=-0.152, p=0.009) but were neutral on
AdversarialQA/CoQA. The hypothesis is that priming helps *retrieval* but not *reasoning/computation*.
However, this conclusion rests on a single dataset per type.

This experiment provides a clean test: **does task type predict hero layer effect beyond difficulty?**

## Dataset Taxonomy

| Dataset | Task Type | Why Chosen | N |
|---------|-----------|-----------|---|
| **NQ** | Retrieval (factoid) | Known positive control — hero d=+0.213 in Exp 27b | 300 |
| **DROP** | Mixed (computation + extraction) | Known negative — hero d=-0.152 in Exp 29 | 300 |
| **BoolQ** | Retrieval (binary judgment) | New — pure passage-based yes/no, second retrieval data point | 300 |

DROP gets split by answer type (number vs span) for within-dataset comparison:
- **DROP-number**: Computational answers (counting, arithmetic) → tagged "computation"
- **DROP-span**: Extractive answers → tagged "retrieval"

## Conditions (reduced to 4)

| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | Baseline |
| 2 | sf_trunc | Standard priming (truncate + RoPE correct) |
| 3 | values_early | Bare keys + primed values layers 0-15 |
| 4 | values_hero | Bare keys + primed values at hero layers {10,12,14,15,20} |

Dropped sf_trunc_bias2 (always hurts on these datasets) and values_only (dominated by values_early).

## Key Question

Does task type predict hero layer effect **beyond** difficulty?""")))

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
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp30")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"
CSV_PATH = RESULTS_DIR / "results.csv"

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
CHECKPOINT_EVERY = 25

# Conditions (reduced to 4 -- dropped sf_trunc_bias2, values_only)
CONDITION_NAMES = ['bare', 'sf_trunc', 'values_early', 'values_hero']

# Layer-selective conditions from Exps 19/21/24
EARLY_LAYER_CUTOFF = 16  # layers 0-15
HERO_LAYERS = [10, 12, 14, 15, 20]  # from Exp 24 single-layer scan

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  N per dataset: {N_PER_DATASET}")
print(f"  MAX_DOC_TOKENS: {MAX_DOC_TOKENS} (sliding window constraint)")
print(f"  N_LAYERS: {N_LAYERS}")
print(f"  EARLY_LAYER_CUTOFF: {EARLY_LAYER_CUTOFF}")
print(f"  HERO_LAYERS: {HERO_LAYERS}")
print(f"  Conditions: {CONDITION_NAMES}")
print(f"  Static fact prefix: '{STATIC_FACT}'")\
""")))

# ========== Cell 4: Load NQ ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load Natural Questions (streaming, same approach as Exp 27b)
from datasets import load_dataset

print("=" * 70)
print("LOADING NATURAL QUESTIONS (validation, streaming)")
print("=" * 70)
print("Factoid retrieval QA. Known positive control for hero layers (d=+0.213 in Exp 27b).")

NQ_CACHE = RESULTS_DIR / "nq_samples.json"

if NQ_CACHE.exists():
    with open(NQ_CACHE, 'r') as f:
        nq_samples = json.load(f)
    print(f"Loaded {len(nq_samples)} cached NQ samples")
else:
    nq_ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True,
    )

    nq_samples = []
    n_processed = 0

    for example in tqdm(nq_ds, desc="Processing NQ"):
        n_processed += 1

        doc_tokens = example['document']['tokens']
        if isinstance(doc_tokens, dict):
            token_strs = doc_tokens['token']
            is_html_flags = doc_tokens['is_html']
            clean_tokens = [t for t, h in zip(token_strs, is_html_flags) if not h]
        else:
            clean_tokens = [t['token'] for t in doc_tokens if not t['is_html']]

        doc_text = ' '.join(clean_tokens)
        wc = count_words(doc_text)

        if wc < 50 or wc > 4000:
            continue

        annotations = example['annotations']
        short_answers_list = annotations['short_answers']

        answer_text = None
        for annotator_sa in short_answers_list:
            if not annotator_sa:
                continue
            texts = annotator_sa.get('text', [])
            if texts:
                answer_text = texts[0]
                break
            starts = annotator_sa.get('start_token', [])
            ends = annotator_sa.get('end_token', [])
            if not starts or not ends:
                continue
            start_tok = starts[0] if isinstance(starts, list) else starts
            end_tok = ends[0] if isinstance(ends, list) else ends
            if start_tok >= 0 and end_tok > start_tok:
                if isinstance(doc_tokens, dict):
                    ans_tokens = [
                        doc_tokens['token'][i]
                        for i in range(start_tok, min(end_tok, len(doc_tokens['token'])))
                        if not doc_tokens['is_html'][i]
                    ]
                else:
                    ans_tokens = [
                        doc_tokens[i]['token']
                        for i in range(start_tok, min(end_tok, len(doc_tokens)))
                        if not doc_tokens[i]['is_html']
                    ]
                if ans_tokens:
                    answer_text = ' '.join(ans_tokens)
                    break

        if not answer_text or len(answer_text.strip()) == 0:
            continue
        if len(answer_text.split()) > 20:
            continue

        question = example['question']
        if isinstance(question, dict):
            query = question.get('text', '')
        else:
            query = str(question)
        if not query.strip():
            continue

        nq_samples.append({
            'passage': doc_text,
            'query': query,
            'answer': answer_text,
            'word_count': wc,
            'dataset': 'nq',
        })

        if len(nq_samples) >= N_PER_DATASET * 3:
            break

    np.random.seed(SEED)
    np.random.shuffle(nq_samples)
    nq_samples = nq_samples[:N_PER_DATASET]

    with open(NQ_CACHE, 'w') as f:
        json.dump(nq_samples, f)
    print(f"Cached {len(nq_samples)} samples (processed {n_processed})")

print(f"NQ samples: {len(nq_samples)}")
wcs = [s['word_count'] for s in nq_samples]
print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
if nq_samples:
    for i in range(min(3, len(nq_samples))):
        print(f"  Example {i+1}:")
        print(f"    Q: {nq_samples[i]['query']}")
        print(f"    A: {nq_samples[i]['answer']}")\
""")))

# ========== Cell 5: Load DROP ==========
cells.append(make_cell("code", s("""\
# Cell 5: Load DROP dataset (numerical/discrete reasoning + extraction)
print("=" * 70)
print("LOADING DROP (validation)")
print("=" * 70)
print("Mixed computation + extraction. Known negative for hero layers (d=-0.152 in Exp 29).")
print("Each sample tagged with answer_type: 'number' or 'span'.")

DROP_CACHE = RESULTS_DIR / "drop_samples.json"

# Regex for number answers: integers, decimals, comma-separated numbers
NUMBER_PATTERN = re.compile(r'^\\d[\\d,\\.]*$')

if DROP_CACHE.exists():
    with open(DROP_CACHE, 'r') as f:
        drop_samples = json.load(f)
    print(f"Loaded {len(drop_samples)} cached DROP samples")
else:
    drop_ds = load_dataset("drop", split="validation")
    print(f"DROP validation size: {len(drop_ds)}")

    drop_samples = []
    np.random.seed(SEED)

    for item in tqdm(drop_ds, desc="Filtering DROP"):
        passage = item.get('passage', '')
        question = item.get('question', '')
        answers_info = item.get('answers_spans', {})

        spans = answers_info.get('spans', [])
        if not spans:
            continue
        answer_text = spans[0]

        if not question or not answer_text or not passage:
            continue
        if len(answer_text.strip()) == 0:
            continue

        wc = count_words(passage)
        if wc < 30 or wc > 2000:
            continue

        # Tag answer type
        answer_type = 'number' if NUMBER_PATTERN.match(answer_text.strip()) else 'span'

        drop_samples.append({
            'passage': passage,
            'query': question,
            'answer': answer_text,
            'word_count': wc,
            'dataset': 'drop',
            'answer_type': answer_type,
            'all_answers': spans,
        })

        if len(drop_samples) >= N_PER_DATASET * 3:
            break

    np.random.shuffle(drop_samples)
    drop_samples = drop_samples[:N_PER_DATASET]

    with open(DROP_CACHE, 'w') as f:
        json.dump(drop_samples, f)
    print(f"Cached {len(drop_samples)} samples")

    del drop_ds
    gc.collect()

# Print answer type distribution
n_number = sum(1 for s in drop_samples if s.get('answer_type') == 'number')
n_span = sum(1 for s in drop_samples if s.get('answer_type') == 'span')
print(f"DROP samples: {len(drop_samples)}")
print(f"  Answer type distribution: number={n_number}, span={n_span}")
wcs = [s['word_count'] for s in drop_samples]
ans_lens = [len(s['answer'].split()) for s in drop_samples]
print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
print(f"  Answer word lengths: mean={np.mean(ans_lens):.1f}, min={min(ans_lens)}, max={max(ans_lens)}")

# Show borderline cases (answers that look numeric but don't match regex)
borderline = [s for s in drop_samples
              if s.get('answer_type') == 'span' and any(c.isdigit() for c in s['answer'])]
if borderline:
    print(f"\\n  Borderline cases (span with digits): {len(borderline)}")
    for b in borderline[:5]:
        print(f"    '{b['answer']}' -> tagged as '{b['answer_type']}'")

if drop_samples:
    for i in range(min(3, len(drop_samples))):
        print(f"  Example {i+1} ({drop_samples[i].get('answer_type', '?')}):")
        print(f"    Q: {drop_samples[i]['query']}")
        print(f"    A: {drop_samples[i]['answer']}")
        print(f"    Passage (first 120 chars): {drop_samples[i]['passage'][:120]}...")\
""")))

# ========== Cell 6: Load BoolQ ==========
cells.append(make_cell("code", s("""\
# Cell 6: Load BoolQ dataset (binary judgment retrieval)
print("=" * 70)
print("LOADING BOOLQ (validation)")
print("=" * 70)
print("Pure passage-based yes/no questions. New retrieval data point.")

BOOLQ_CACHE = RESULTS_DIR / "boolq_samples.json"

if BOOLQ_CACHE.exists():
    with open(BOOLQ_CACHE, 'r') as f:
        boolq_samples = json.load(f)
    print(f"Loaded {len(boolq_samples)} cached BoolQ samples")
else:
    boolq_ds = load_dataset("google/boolq", split="validation")
    print(f"BoolQ validation size: {len(boolq_ds)}")

    boolq_samples = []
    np.random.seed(SEED)

    for item in tqdm(boolq_ds, desc="Filtering BoolQ"):
        passage = item.get('passage', '')
        question = item.get('question', '')
        answer_bool = item.get('answer', None)

        if not question or not passage or answer_bool is None:
            continue

        # Map boolean to text answer
        answer_text = "Yes" if answer_bool else "No"

        wc = count_words(passage)
        if wc < 30 or wc > 2000:
            continue

        boolq_samples.append({
            'passage': passage,
            'query': question,
            'answer': answer_text,
            'word_count': wc,
            'dataset': 'boolq',
        })

    np.random.shuffle(boolq_samples)
    boolq_samples = boolq_samples[:N_PER_DATASET]

    with open(BOOLQ_CACHE, 'w') as f:
        json.dump(boolq_samples, f)
    print(f"Cached {len(boolq_samples)} samples")

    del boolq_ds
    gc.collect()

print(f"BoolQ samples: {len(boolq_samples)}")
wcs = [s['word_count'] for s in boolq_samples]
n_yes = sum(1 for s in boolq_samples if s['answer'] == 'Yes')
n_no = sum(1 for s in boolq_samples if s['answer'] == 'No')
print(f"  Answer distribution: Yes={n_yes}, No={n_no}")
print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
if boolq_samples:
    for i in range(min(3, len(boolq_samples))):
        print(f"  Example {i+1} ({boolq_samples[i]['answer']}):")
        print(f"    Q: {boolq_samples[i]['query']}")
        print(f"    Passage (first 120 chars): {boolq_samples[i]['passage'][:120]}...")\
""")))

# ========== Cell 7: Unified pool + pre-screening ==========
cells.append(make_cell("code", s("""\
# Cell 7: Unified sample pool + tokenization + pre-screening
print("=" * 70)
print("UNIFIED SAMPLE POOL")
print("=" * 70)

all_samples = []
for ds_name, ds_samples in [("nq", nq_samples),
                              ("drop", drop_samples),
                              ("boolq", boolq_samples)]:
    for sample in ds_samples:
        sample['dataset'] = ds_name
    all_samples.extend(ds_samples)

print(f"Total samples: {len(all_samples)}")
for ds_name in ['nq', 'drop', 'boolq']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
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

for ds_name in ['nq', 'drop', 'boolq']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
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

for ds_name in ['nq', 'drop', 'boolq']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name][:20]
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

# ========== Cell 8: Helper function ==========
cells.append(make_cell("code", s("""\
# Cell 8: Helper function — run_single_sample_4cond()

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
    # Carry forward answer_type for DROP
    if 'answer_type' in sample:
        result['answer_type'] = sample['answer_type']
    return result


print("Helper function defined: run_single_sample_4cond()")
print("  Conditions: bare, sf_trunc, values_early, values_hero")
print("  No bias mask needed (sf_trunc_bias2 dropped)")
print("  No values_only (dominated by values_early)")\
""")))

# ========== Cell 9: Main experiment loop ==========
cells.append(make_cell("code", s("""\
# Cell 9: Main experiment loop

print("=" * 70)
print(f"EXPERIMENT 30: {len(all_samples)} samples, {len(CONDITION_NAMES)} conditions")
print(f"Model: Gemma 3 4B, MAX_DOC_TOKENS: {MAX_DOC_TOKENS}")
print(f"Datasets: NQ, DROP, BoolQ")
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
                  desc="Exp 30"):
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
print(f"\\nExperiment complete: {len(all_results)} samples in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 10: Per-dataset analysis ==========
cells.append(make_cell("code", s("""\
# Cell 10: Per-dataset results table
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("ANALYSIS: PER-DATASET RESULTS (Gemma 3 4B)")
print("=" * 70)

dataset_names = ['nq', 'drop', 'boolq']
analysis = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    n_ds = len(ds_results)
    if n_ds == 0:
        continue

    bare_arr = np.array([r['bare'] for r in ds_results])

    # Filter invalid (keep zeros -- valid for some datasets)
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
    bare = [r['bare'] for r in ds_r]
    pct_zero = 100 * np.mean(np.array(bare) < 0.01)
    iqr = np.percentile(bare, 75) - np.percentile(bare, 25)
    print(f"  {ds:15s}: mean={np.mean(bare):.3f}, median={np.median(bare):.3f}, "
          f"IQR={iqr:.3f}, pct_floor={pct_zero:.0f}%")

# Reference from prior exps
print(f"\\n\\n{'='*90}")
print("COMPARISON WITH PRIOR EXPERIMENTS")
print(f"{'='*90}")
print("\\nExp 27b (Gemma, NQ/TriviaQA/HotpotQA):")
print("  NQ:       values_hero d=+0.213***")
print("  TriviaQA: values_hero d=+0.000 (77% at floor)")
print("  HotpotQA: values_hero d=-0.069 (56% at floor)")
print("\\nExp 29 (Gemma, DROP/AdvQA/CoQA):")
print("  DROP:     values_hero d=-0.152**")
print("  AdvQA:    values_hero d=+0.026 (72% at floor)")
print("  CoQA:     values_hero d=+0.070 (65% at floor)")\
""")))

# ========== Cell 11: Within-DROP split ==========
cells.append(make_cell("code", s("""\
# Cell 11: Within-DROP split by answer type (THE KEY TEST)
print("=" * 70)
print("WITHIN-DROP SPLIT: Number vs Span Answer Types")
print("=" * 70)
print("This is the most important analysis: same dataset, same passages,")
print("different answer types. Number = computation, Span = retrieval.")

drop_results = [r for r in all_results if r['dataset'] == 'drop']

# Split by answer_type
drop_number = [r for r in drop_results if r.get('answer_type') == 'number']
drop_span = [r for r in drop_results if r.get('answer_type') == 'span']

print(f"\\nDROP-number: n={len(drop_number)}")
print(f"DROP-span:   n={len(drop_span)}")

drop_split_analysis = {}

for split_name, split_results in [('drop_number', drop_number), ('drop_span', drop_span)]:
    if len(split_results) < 20:
        print(f"\\n  {split_name}: too few samples ({len(split_results)}), skipping")
        drop_split_analysis[split_name] = {'n': len(split_results), 'skipped': True}
        continue

    bare_arr = np.array([r['bare'] for r in split_results])
    pct_floor = 100 * np.mean(bare_arr < 0.01)

    print(f"\\n{'='*70}")
    print(f"{split_name.upper()} (n={len(split_results)}, "
          f"median bare NLL={np.median(bare_arr):.3f}, pct_floor={pct_floor:.0f}%)")
    print(f"{'='*70}")

    print(f"\\n{'Condition':<20} {'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
    print("-" * 60)

    split_data = {'n': len(split_results), 'pct_floor': float(pct_floor)}
    for cname in ['sf_trunc', 'values_early', 'values_hero']:
        c_arr = np.array([r[cname] for r in split_results])
        delta = bare_arr - c_arr
        valid = np.isfinite(delta)
        delta = delta[valid]
        if len(delta) < 10:
            print(f"{cname:<20} {'n/a (too few valid)':>40}")
            continue
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        t_stat, p_val = stats.ttest_1samp(delta, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"{cname:<20} {d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
        split_data[cname] = {
            'cohens_d': float(d),
            'win_pct': float(win),
            'p_value': float(p_val),
        }

    drop_split_analysis[split_name] = split_data

# Bare NLL distributions by type
print(f"\\n\\nBARE NLL DISTRIBUTIONS BY ANSWER TYPE:")
for split_name, split_results in [('drop_number', drop_number), ('drop_span', drop_span)]:
    if not split_results:
        continue
    bare = np.array([r['bare'] for r in split_results])
    pct_floor = 100 * np.mean(bare < 0.01)
    print(f"  {split_name:15s}: mean={np.mean(bare):.3f}, median={np.median(bare):.3f}, "
          f"IQR={np.percentile(bare,75)-np.percentile(bare,25):.3f}, "
          f"pct_floor={pct_floor:.0f}%")

# Difficulty-matched within DROP
print(f"\\n\\nDIFFICULTY-MATCHED WITHIN DROP (bare > 0.5):")
for split_name, split_results in [('drop_number', drop_number), ('drop_span', drop_span)]:
    hard = [r for r in split_results if r['bare'] > 0.5]
    if len(hard) < 10:
        print(f"  {split_name}: n_hard={len(hard)} (too few)")
        continue
    bare_h = np.array([r['bare'] for r in hard])
    hero_h = np.array([r['values_hero'] for r in hard])
    delta_h = bare_h - hero_h
    d_h = cohens_d(delta_h)
    win_h = np.mean(delta_h > 0) * 100
    _, p_h = stats.ttest_1samp(delta_h, 0)
    sig_h = '***' if p_h < 0.001 else '**' if p_h < 0.01 else '*' if p_h < 0.05 else 'ns'
    print(f"  {split_name} (n_hard={len(hard)}): hero d={d_h:+.3f}, "
          f"win={win_h:.0f}%, p={p_h:.2e} {sig_h}")

# Key interpretation
print(f"\\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
hero_num = drop_split_analysis.get('drop_number', {}).get('values_hero', {})
hero_span = drop_split_analysis.get('drop_span', {}).get('values_hero', {})
d_num = hero_num.get('cohens_d', float('nan'))
d_span = hero_span.get('cohens_d', float('nan'))
print(f"  DROP-number hero d: {d_num:+.3f}")
print(f"  DROP-span hero d:   {d_span:+.3f}")
if d_span > d_num + 0.05:
    print("  -> CONSISTENT with task-type hypothesis: span (retrieval) > number (computation)")
elif abs(d_span - d_num) < 0.05:
    print("  -> INCONCLUSIVE: number and span show similar effects")
else:
    print("  -> INCONSISTENT: number benefits MORE than span (unexpected)")\
""")))

# ========== Cell 12: Difficulty-matched cross-dataset comparison ==========
cells.append(make_cell("code", s("""\
# Cell 12: Difficulty-matched cross-dataset comparison
print("=" * 70)
print("DIFFICULTY-MATCHED CROSS-DATASET COMPARISON")
print("=" * 70)
print("Filter to hard samples (bare > 0.5) across all datasets.")
print("Compare hero d for retrieval vs computation subsets.")

# Collect hard samples by subset
hard_subsets = {}
for ds_name in ['nq', 'drop', 'boolq']:
    ds_r = [r for r in all_results if r['dataset'] == ds_name]
    hard = [r for r in ds_r if r['bare'] > 0.5]
    hard_subsets[ds_name] = hard
    print(f"  {ds_name}: {len(hard)}/{len(ds_r)} hard samples (bare > 0.5)")

# Also split DROP hard by type
drop_hard = hard_subsets.get('drop', [])
drop_hard_number = [r for r in drop_hard if r.get('answer_type') == 'number']
drop_hard_span = [r for r in drop_hard if r.get('answer_type') == 'span']
hard_subsets['drop_number'] = drop_hard_number
hard_subsets['drop_span'] = drop_hard_span
print(f"  drop_number: {len(drop_hard_number)} hard")
print(f"  drop_span:   {len(drop_hard_span)} hard")

# Hero d for each hard subset
print(f"\\n{'='*70}")
print("HERO d ON HARD SAMPLES (bare > 0.5)")
print(f"{'='*70}")
print(f"\\n{'Subset':<20} {'N':>5} {'hero d':>8} {'95% CI':>20} {'p':>12} {'sig':>5}")
print("-" * 75)

hard_hero_ds = {}
for subset_name in ['nq', 'drop_number', 'drop_span', 'boolq']:
    subset = hard_subsets.get(subset_name, [])
    if len(subset) < 20:
        print(f"{subset_name:<20} {len(subset):>5} {'(n<20, skip)':>40}")
        hard_hero_ds[subset_name] = {'n': len(subset), 'skipped': True}
        continue

    bare_h = np.array([r['bare'] for r in subset])
    hero_h = np.array([r['values_hero'] for r in subset])
    delta_h = bare_h - hero_h

    d_h = cohens_d(delta_h)
    _, p_h = stats.ttest_1samp(delta_h, 0)
    sig_h = '***' if p_h < 0.001 else '**' if p_h < 0.01 else '*' if p_h < 0.05 else 'ns'

    # Bootstrap 95% CI for hero d
    np.random.seed(SEED)
    boot_ds = []
    for _ in range(2000):
        idx = np.random.choice(len(delta_h), len(delta_h), replace=True)
        boot_ds.append(cohens_d(delta_h[idx]))
    ci_lo = np.percentile(boot_ds, 2.5)
    ci_hi = np.percentile(boot_ds, 97.5)

    print(f"{subset_name:<20} {len(subset):>5} {d_h:>+8.3f} "
          f"[{ci_lo:>+8.3f}, {ci_hi:>+8.3f}] {p_h:>12.2e} {sig_h:>5}")
    hard_hero_ds[subset_name] = {
        'n': len(subset),
        'cohens_d': float(d_h),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'p_value': float(p_h),
    }

# Welch's t-test: retrieval vs computation hard-subset deltas
print(f"\\n\\nWELCH'S T-TEST: Retrieval vs Computation (hard samples)")
print("-" * 70)

retrieval_deltas = []
computation_deltas = []

# Retrieval: NQ-hard + DROP-span-hard + BoolQ-hard
for subset_name in ['nq', 'drop_span', 'boolq']:
    subset = hard_subsets.get(subset_name, [])
    for r in subset:
        retrieval_deltas.append(r['bare'] - r['values_hero'])

# Computation: DROP-number-hard
for r in hard_subsets.get('drop_number', []):
    computation_deltas.append(r['bare'] - r['values_hero'])

retrieval_deltas = np.array(retrieval_deltas)
computation_deltas = np.array(computation_deltas)

print(f"  Retrieval hard samples: n={len(retrieval_deltas)}")
print(f"  Computation hard samples: n={len(computation_deltas)}")

if len(retrieval_deltas) >= 10 and len(computation_deltas) >= 10:
    t_stat, p_welch = stats.ttest_ind(retrieval_deltas, computation_deltas, equal_var=False)
    sig_w = '***' if p_welch < 0.001 else '**' if p_welch < 0.01 else '*' if p_welch < 0.05 else 'ns'
    print(f"  Retrieval mean delta: {np.mean(retrieval_deltas):+.4f}")
    print(f"  Computation mean delta: {np.mean(computation_deltas):+.4f}")
    print(f"  Welch's t={t_stat:.3f}, p={p_welch:.2e} {sig_w}")
    if p_welch < 0.05:
        print("  -> SIGNIFICANT difference between retrieval and computation hard-sample hero effects")
    else:
        print("  -> No significant difference (may be underpowered or confounded)")
else:
    p_welch = float('nan')
    print("  -> Too few samples for Welch's t-test")\
""")))

# ========== Cell 13: Task-type regression ==========
cells.append(make_cell("code", s("""\
# Cell 13: Task-type regression
print("=" * 70)
print("TASK-TYPE REGRESSION")
print("=" * 70)
print("Linear regression: delta_i = b0 + b1*bare_i + b2*is_retrieval_i")
print("Tests whether task type predicts hero effect BEYOND difficulty.")

# Tag all samples by task type
# Retrieval: NQ, BoolQ, DROP-span
# Computation: DROP-number
task_type_map = {}
for r in all_results:
    if r['dataset'] == 'nq':
        task_type_map[r['query_idx']] = 'retrieval'
    elif r['dataset'] == 'boolq':
        task_type_map[r['query_idx']] = 'retrieval'
    elif r['dataset'] == 'drop':
        if r.get('answer_type') == 'number':
            task_type_map[r['query_idx']] = 'computation'
        else:
            task_type_map[r['query_idx']] = 'retrieval'

# Build regression data
bare_vals = []
delta_vals = []
is_retrieval = []
for r in all_results:
    if r['query_idx'] not in task_type_map:
        continue
    b = r['bare']
    h = r['values_hero']
    if not (np.isfinite(b) and np.isfinite(h)):
        continue
    bare_vals.append(b)
    delta_vals.append(b - h)
    is_retrieval.append(1 if task_type_map[r['query_idx']] == 'retrieval' else 0)

bare_vals = np.array(bare_vals)
delta_vals = np.array(delta_vals)
is_retrieval = np.array(is_retrieval)

print(f"\\nRegression samples: {len(bare_vals)}")
print(f"  Retrieval: {np.sum(is_retrieval)}")
print(f"  Computation: {np.sum(1 - is_retrieval)}")

# OLS regression: delta = b0 + b1*bare + b2*is_retrieval
from numpy.linalg import lstsq

X = np.column_stack([np.ones(len(bare_vals)), bare_vals, is_retrieval])
beta, residuals, rank, sv = lstsq(X, delta_vals, rcond=None)

b0, b1, b2 = beta
y_hat = X @ beta
ss_res = np.sum((delta_vals - y_hat) ** 2)
ss_tot = np.sum((delta_vals - np.mean(delta_vals)) ** 2)
r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

# Standard errors
n = len(delta_vals)
p_params = X.shape[1]
mse = ss_res / (n - p_params)
var_beta = mse * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(var_beta))
t_stats = beta / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p_params))

print(f"\\nOLS Regression: delta_hero = b0 + b1*bare + b2*is_retrieval")
print(f"  R-squared: {r_squared:.4f}")
print(f"\\n{'Parameter':<15} {'Estimate':>10} {'SE':>10} {'t':>8} {'p':>12} {'sig':>5}")
print("-" * 65)
param_names = ['intercept', 'bare_nll', 'is_retrieval']
for i, pname in enumerate(param_names):
    sig = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else 'ns'
    print(f"{pname:<15} {beta[i]:>+10.4f} {se_beta[i]:>10.4f} "
          f"{t_stats[i]:>8.3f} {p_values[i]:>12.2e} {sig:>5}")

print(f"\\nKEY RESULT: beta_2 (is_retrieval) = {b2:+.4f}, p = {p_values[2]:.2e}")
if p_values[2] < 0.05:
    direction = "MORE" if b2 > 0 else "LESS"
    print(f"  -> SIGNIFICANT: Retrieval tasks benefit {direction} from hero layers, "
          f"controlling for difficulty")
else:
    print("  -> NOT SIGNIFICANT: Task type does not predict hero effect beyond difficulty")

# Spearman correlation at dataset-subset level
print(f"\\n\\nSPEARMAN CORRELATION: hero_d vs pct_floor (dataset level)")
subset_ds = []
subset_floors = []
for subset_name in ['nq', 'drop_number', 'drop_span', 'boolq']:
    if subset_name == 'drop_number':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'number']
    elif subset_name == 'drop_span':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'span']
    else:
        sr = [r for r in all_results if r['dataset'] == subset_name]
    if len(sr) < 20:
        continue
    bare_s = np.array([r['bare'] for r in sr])
    hero_s = np.array([r['values_hero'] for r in sr])
    delta_s = bare_s - hero_s
    d_s = cohens_d(delta_s)
    floor_s = float(np.mean(bare_s < 0.01) * 100)
    subset_ds.append(d_s)
    subset_floors.append(floor_s)
    print(f"  {subset_name:15s}: hero d={d_s:+.3f}, pct_floor={floor_s:.0f}%")

if len(subset_ds) >= 3:
    rho, p_spear = stats.spearmanr(subset_floors, subset_ds)
    print(f"  Spearman rho={rho:+.3f}, p={p_spear:.3f} (n={len(subset_ds)} subsets)")
    if p_spear < 0.05:
        print("  -> Ceiling effects significantly predict hero d at dataset level")
    else:
        print("  -> Ceiling effects do NOT significantly predict hero d (but n is small)")
else:
    rho, p_spear = float('nan'), float('nan')
    print("  -> Too few subsets for Spearman")\
""")))

# ========== Cell 14: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 14: Multi-panel figure (2x2)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

colors_task = {'retrieval': '#2ca02c', 'computation': '#d62728', 'mixed': '#7f7f7f'}

# ---- Panel (a): Hero d by dataset-subset, colored by task type ----
ax = axes[0, 0]

subset_info = [
    ('NQ', 'nq', 'retrieval'),
    ('DROP-num', 'drop_number', 'computation'),
    ('DROP-span', 'drop_span', 'retrieval'),
    ('BoolQ', 'boolq', 'retrieval'),
]

x_pos = np.arange(len(subset_info))
bar_ds = []
bar_colors = []
bar_labels = []

for label, subset_key, task_type in subset_info:
    if subset_key == 'drop_number':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'number']
    elif subset_key == 'drop_span':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'span']
    else:
        sr = [r for r in all_results if r['dataset'] == subset_key]

    if len(sr) >= 10:
        bare_s = np.array([r['bare'] for r in sr])
        hero_s = np.array([r['values_hero'] for r in sr])
        delta_s = bare_s - hero_s
        d_s = cohens_d(delta_s)
        _, p_s = stats.ttest_1samp(delta_s, 0)
        sig_s = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else ''
    else:
        d_s = 0
        sig_s = 'n/a'
    bar_ds.append(d_s)
    bar_colors.append(colors_task[task_type])
    bar_labels.append(f"{label}\\n({task_type})")

bars = ax.bar(x_pos, bar_ds, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, (d_val, sig_val) in enumerate(zip(bar_ds, [
    '***' if subset_info[j][1] in ['nq'] else sig_s for j in range(len(subset_info))
])):
    ax.text(i, d_val + (0.01 if d_val >= 0 else -0.03),
            f"{d_val:+.3f}", ha='center',
            va='bottom' if d_val >= 0 else 'top', fontsize=9, fontweight='bold')

# Re-compute significance for each bar
for i, (label, subset_key, task_type) in enumerate(subset_info):
    if subset_key == 'drop_number':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'number']
    elif subset_key == 'drop_span':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'span']
    else:
        sr = [r for r in all_results if r['dataset'] == subset_key]
    if len(sr) >= 10:
        bare_s = np.array([r['bare'] for r in sr])
        hero_s = np.array([r['values_hero'] for r in sr])
        delta_s = bare_s - hero_s
        _, p_s = stats.ttest_1samp(delta_s, 0)
        sig_s = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else ''
        if sig_s:
            ax.text(i, bar_ds[i] + (0.035 if bar_ds[i] >= 0 else -0.055),
                    sig_s, ha='center', va='bottom' if bar_ds[i] >= 0 else 'top', fontsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("(a) Hero Layer Effect by Dataset-Subset\\n(colored by task type)")

# Legend for task types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_task['retrieval'], label='Retrieval'),
                   Patch(facecolor=colors_task['computation'], label='Computation')]
ax.legend(handles=legend_elements, fontsize=9)

# ---- Panel (b): Hard-sample delta distributions (violin/box) ----
ax = axes[0, 1]

hard_data = []
hard_labels = []
hard_colors_list = []
for label, subset_key, task_type in subset_info:
    if subset_key == 'drop_number':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'number']
    elif subset_key == 'drop_span':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'span']
    else:
        sr = [r for r in all_results if r['dataset'] == subset_key]
    hard = [r['bare'] - r['values_hero'] for r in sr if r['bare'] > 0.5]
    if len(hard) >= 10:
        hard_data.append(hard)
        hard_labels.append(f"{label}\\n(n={len(hard)})")
        hard_colors_list.append(colors_task[task_type])

if hard_data:
    bp = ax.boxplot(hard_data, labels=hard_labels, showfliers=True, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2},
                    flierprops={'markersize': 3, 'alpha': 0.5})
    for patch, color in zip(bp['boxes'], hard_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Delta (bare - hero, positive = hero helps)")
ax.set_title("(b) Hard-Sample Delta Distributions\\n(bare > 0.5)")

# ---- Panel (c): Hardness gradient: hero d by quintile for NQ vs DROP-number ----
ax = axes[1, 0]
quintile_labels = ['Q1\\n(easy)', 'Q2', 'Q3', 'Q4', 'Q5\\n(hard)']

for ds_label, ds_filter, color, marker in [
    ('NQ', lambda r: r['dataset'] == 'nq', '#2ca02c', 'o'),
    ('DROP-number', lambda r: r['dataset'] == 'drop' and r.get('answer_type') == 'number', '#d62728', 's'),
    ('DROP-span', lambda r: r['dataset'] == 'drop' and r.get('answer_type') == 'span', '#ff7f0e', '^'),
    ('BoolQ', lambda r: r['dataset'] == 'boolq', '#1f77b4', 'D'),
]:
    ds_r = [r for r in all_results if ds_filter(r)]
    if len(ds_r) < 50:
        continue
    bare_arr = np.array([r['bare'] for r in ds_r])
    hero_arr = np.array([r['values_hero'] for r in ds_r])
    delta_arr = bare_arr - hero_arr
    quintile_boundaries = np.percentile(bare_arr, [20, 40, 60, 80])

    q_ds = []
    q_ns = []
    for q in range(5):
        if q < 4:
            lo = quintile_boundaries[q-1] if q > 0 else -np.inf
            hi = quintile_boundaries[q]
        else:
            lo = quintile_boundaries[3]
            hi = np.inf
        mask = (bare_arr > lo) & (bare_arr <= hi)
        if q == 0:
            mask = bare_arr <= quintile_boundaries[0]
        n_q = int(np.sum(mask))
        if n_q >= 5:
            q_ds.append(cohens_d(delta_arr[mask]))
        else:
            q_ds.append(np.nan)
        q_ns.append(n_q)

    valid_q = [(i, d) for i, d in enumerate(q_ds) if not np.isnan(d)]
    if valid_q:
        xs, ys = zip(*valid_q)
        ax.plot(xs, ys, marker=marker, linewidth=2, markersize=7, label=ds_label, color=color)

ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d (hero vs bare)")
ax.set_xlabel("Bare NLL Quintile")
ax.set_title("(c) Hardness Gradient: Hero Effect by Quintile")
ax.legend(fontsize=8)

# ---- Panel (d): Bare NLL distributions by dataset (ceiling check) ----
ax = axes[1, 1]
bare_by_subset = []
subset_labels_plot = []
for label, subset_key, task_type in subset_info:
    if subset_key == 'drop_number':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'number']
    elif subset_key == 'drop_span':
        sr = [r for r in all_results if r['dataset'] == 'drop' and r.get('answer_type') == 'span']
    else:
        sr = [r for r in all_results if r['dataset'] == subset_key]
    if sr:
        bare_by_subset.append([r['bare'] for r in sr])
        pct_f = 100 * np.mean(np.array([r['bare'] for r in sr]) < 0.01)
        subset_labels_plot.append(f"{label}\\n({pct_f:.0f}% floor)")

if bare_by_subset:
    bp = ax.boxplot(bare_by_subset, labels=subset_labels_plot, showfliers=False, patch_artist=True,
                    medianprops={'color': 'red', 'linewidth': 2})
    for patch in bp['boxes']:
        patch.set_facecolor('#8ecae6')
        patch.set_alpha(0.7)

ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, label='Floor threshold (0.01)')
ax.set_ylabel("Bare NLL")
ax.set_title("(d) Bare NLL Distributions (ceiling check)")
ax.legend(fontsize=7)

plt.suptitle('Exp 30: Retrieval vs Reasoning Task-Type Dissociation (Gemma 3 4B)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 15: Save results + verdict ==========
cells.append(make_cell("code", s("""\
# Cell 15: Save results.json + CSV + verdict

# --- CSV ---
with open(CSV_PATH, 'w', newline='') as f:
    fieldnames = ['query_idx', 'dataset', 'query', 'answer', 'word_count',
                  'doc_token_len', 'answer_token_len', 'answer_type',
                  'bare', 'sf_trunc', 'values_early', 'values_hero']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in all_results:
        writer.writerow({k: r.get(k, '') for k in fieldnames})
print(f"CSV saved: {CSV_PATH}")

# --- Compute verdict inputs ---

# Hero d on hard retrieval samples
retrieval_hard = []
for r in all_results:
    task_type = task_type_map.get(r['query_idx'])
    if task_type == 'retrieval' and r['bare'] > 0.5:
        retrieval_hard.append(r['bare'] - r['values_hero'])
retrieval_hard_d = cohens_d(np.array(retrieval_hard)) if len(retrieval_hard) >= 10 else float('nan')

# Hero d on hard computation samples
computation_hard = []
for r in all_results:
    task_type = task_type_map.get(r['query_idx'])
    if task_type == 'computation' and r['bare'] > 0.5:
        computation_hard.append(r['bare'] - r['values_hero'])
computation_hard_d = cohens_d(np.array(computation_hard)) if len(computation_hard) >= 10 else float('nan')

# Regression significance
regression_sig = p_values[2] < 0.05 if len(p_values) > 2 else False

print(f"\\nVerdict inputs:")
print(f"  Retrieval hard d: {retrieval_hard_d:+.3f} (n={len(retrieval_hard)})")
print(f"  Computation hard d: {computation_hard_d:+.3f} (n={len(computation_hard)})")
print(f"  Regression beta_2 p: {p_values[2]:.2e}")
print(f"  Regression significant: {regression_sig}")

# --- Verdict ---
if retrieval_hard_d > 0.15 and computation_hard_d < 0 and regression_sig:
    verdict = ("SUPPORTED: Hero layers selectively help retrieval. "
               f"Retrieval hard d={retrieval_hard_d:+.3f}, "
               f"computation hard d={computation_hard_d:+.3f}, "
               f"beta_2 p={p_values[2]:.2e}")
elif np.mean(np.array([r['bare'] for r in all_results]) < 0.01) > 0.5 and not regression_sig:
    verdict = ("CONFOUNDED: Cannot separate task type from ceiling effects. "
               f"Overall {100*np.mean(np.array([r['bare'] for r in all_results]) < 0.01):.0f}% at floor.")
else:
    verdict = ("INCONCLUSIVE: " +
               f"Retrieval hard d={retrieval_hard_d:+.3f}, "
               f"computation hard d={computation_hard_d:+.3f}, "
               f"regression p={p_values[2]:.2e}")

# Hero scorecard
hero_scorecard = {}
for ds_name in dataset_names:
    if ds_name in analysis and 'values_hero' in analysis[ds_name]:
        hero_scorecard[ds_name] = analysis[ds_name]['values_hero']['cohens_d']

print(f"\\n{'='*70}")
print(f"VERDICT: {verdict}")
print(f"{'='*70}")
print(f"\\nHero scorecard (this experiment):")
for ds, d in hero_scorecard.items():
    p = analysis[ds]['values_hero']['p_value']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {ds:15s}: d={d:+.3f} {sig}")

# Reference table
print(f"\\nUpdated hero scorecard (all experiments):")
print("  MARCO:    d=+0.472*** (Exp 07, Mistral)")
print("  NQ:       d=+0.213*** (Exp 27b, Gemma)")
hero_nq = hero_scorecard.get('nq', '?')
if isinstance(hero_nq, float):
    print(f"  NQ (30):  d={hero_nq:+.3f} (this experiment)")
print("  TriviaQA: d=+0.000 (Exp 27b, ceiling)")
print("  HotpotQA: d=-0.069 (Exp 27b)")
print("  AdvQA:    d=+0.026 (Exp 29, ceiling)")
print("  CoQA:     d=+0.070 (Exp 29, ceiling)")
print("  DROP:     d=-0.152** (Exp 29)")
hero_drop = hero_scorecard.get('drop', '?')
if isinstance(hero_drop, float):
    print(f"  DROP (30): d={hero_drop:+.3f} (this experiment)")
hero_boolq = hero_scorecard.get('boolq', '?')
if isinstance(hero_boolq, float):
    print(f"  BoolQ:    d={hero_boolq:+.3f} (this experiment, NEW)")

# --- results.json ---
final = {
    'experiment': 'exp30_retrieval_vs_reasoning',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'n_per_dataset': N_PER_DATASET,
        'max_doc_tokens': MAX_DOC_TOKENS,
        'conditions': CONDITION_NAMES,
        'early_layer_cutoff': EARLY_LAYER_CUTOFF,
        'hero_layers': HERO_LAYERS,
        'prefix': STATIC_FACT,
        'prefix_token_len': PREFIX_TOKEN_LEN,
        'datasets': dataset_names,
    },
    'per_dataset_analysis': analysis,
    'drop_split_analysis': drop_split_analysis,
    'hard_hero_analysis': hard_hero_ds,
    'regression': {
        'beta_0': float(b0),
        'beta_1_bare': float(b1),
        'beta_2_is_retrieval': float(b2),
        'se_beta': [float(s) for s in se_beta],
        'p_values': [float(p) for p in p_values],
        'r_squared': float(r_squared),
    },
    'ceiling_status': {
        ds: float(np.mean(np.array([r['bare'] for r in all_results if r['dataset'] == ds]) < 0.01) * 100)
        for ds in dataset_names
    },
    'verdict': verdict,
    'hero_scorecard': hero_scorecard,
    'per_sample_results': all_results,
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

output_path = "/home/jupyter/research/directed_kvcache_v2/30_retrieval_vs_reasoning.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
