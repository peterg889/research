#!/usr/bin/env python3
"""Build script for 27b_cross_dataset_gemma.ipynb

Exp 27b: Cross-Dataset Generalization on Gemma 3 4B

Exp 27 (Mistral) showed attention forcing does NOT generalize beyond MS MARCO.
Key finding: the target datasets (TriviaQA, NQ, HotpotQA) are at ceiling for
Mistral — median bare NLL = 0.000 on TriviaQA/HotpotQA, leaving no room for
improvement.

This experiment tests the same datasets on Gemma 3 4B, using the Gemma-specific
toolkit (values-early-layers, hero layers) established in Exps 19/21/24.

Constraints:
  - Gemma sliding window = 1024 → MAX_DOC_TOKENS = 900 (1 + prefix + 900 < 1024)
  - Attention forcing included as novel test (never tried on Gemma)
  - values_early_layers (cutoff=16) is the expected best condition

Conditions (6 per dataset):
  1. bare: BOS + doc standard causal (baseline)
  2. sf_trunc: Standard priming (truncate + RoPE correct, expected to fail)
  3. sf_trunc_bias2: +2.0 logit bias attention forcing (novel on Gemma)
  4. values_only: Bare keys + all primed values (expected weak positive, d~+0.05)
  5. values_early_layers: Bare keys + primed values layers 0-15 only (expected best)
  6. values_hero: Bare keys + primed values at hero layers {10,12,14,15,20} only

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
# Exp 27b: Cross-Dataset Generalization on Gemma 3 4B

## Motivation

Exp 27 (Mistral) showed attention forcing does NOT generalize beyond MS MARCO. The target
datasets are at ceiling for Mistral (median bare NLL = 0 on TriviaQA/HotpotQA). This experiment
tests whether Gemma 3 4B shows the same ceiling effects, and whether the Gemma-specific toolkit
(values-early-layers from Exps 19/21/24) can extract signal where Mistral could not.

## Key Differences from Exp 27

| Parameter | Exp 27 (Mistral) | Exp 27b (Gemma) |
|-----------|-----------------|-----------------|
| Model | Mistral 7B | Gemma 3 4B |
| MAX_DOC_TOKENS | 4096 | 900 (sliding window constraint) |
| Attention forcing | bias=0, 2.0, 4.0 | bias=0, 2.0 only |
| Values manipulation | values_only (all layers) | values_only + values_early (0-15) + values_hero |
| Hero layers | N/A | {10, 12, 14, 15, 20} from Exp 24 |

## Conditions

| # | Condition | Description | Expected (from prior exps) |
|---|-----------|-------------|---------------------------|
| 1 | bare | Baseline | -- |
| 2 | sf_trunc | Standard priming | d ~ -0.03 (Exp 16) |
| 3 | sf_trunc_bias2 | +2.0 bias forcing | Novel on Gemma |
| 4 | values_only | All-layer value swap | d ~ +0.05 (Exp 16) |
| 5 | values_early | Layers 0-15 value swap | d ~ +0.21 (Exp 19) |
| 6 | values_hero | Layers {10,12,14,15,20} | d ~ +0.24 (Exp 24 single-layer peak) |

## Sliding Window Constraint

Gemma 3 has 29 sliding-window layers (SW=1024) + 5 full-attention layers.
MAX_DOC_TOKENS = 900 ensures total sequence 1 + prefix(~10) + doc(900) = 911 < 1024,
so no positions are evicted from sliding-window layer caches during the primed forward pass.""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup
import os
os.umask(0o000)

import sys
import json
import time
import gc
import csv
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp27b")
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
# Gemma sliding window = 1024: total seq must be < 1024
# Primed pass: 1(BOS) + ~10(prefix) + doc_len < 1024 → doc_len < ~1013
# Cap at 900 for safety (matching Exp 21)
MAX_DOC_TOKENS = 900
CHECKPOINT_EVERY = 25

# Conditions
CONDITION_NAMES = ['bare', 'sf_trunc', 'sf_trunc_bias2', 'values_only',
                   'values_early', 'values_hero']

# Layer-selective conditions from Exps 19/21/24
EARLY_LAYER_CUTOFF = 16  # layers 0-15
HERO_LAYERS = [10, 12, 14, 15, 20]  # from Exp 24 single-layer scan

# Length bins for stratified analysis (token count)
LENGTH_BINS = [
    ('<256', 0, 256),
    ('256-512', 256, 512),
    ('512-900', 512, 901),
]

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

# ========== Cell 4: Load TriviaQA ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load TriviaQA dataset
from datasets import load_dataset

print("=" * 70)
print("LOADING TRIVIAQA (rc.wikipedia, validation)")
print("=" * 70)

TQA_CACHE = RESULTS_DIR / "tqa_samples.json"

if TQA_CACHE.exists():
    with open(TQA_CACHE, 'r') as f:
        tqa_samples = json.load(f)
    print(f"Loaded {len(tqa_samples)} cached TriviaQA samples")
else:
    tqa_ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                           trust_remote_code=True)
    print(f"TriviaQA validation size: {len(tqa_ds)}")

    tqa_samples = []
    np.random.seed(SEED)

    for item in tqdm(tqa_ds, desc="Filtering TriviaQA"):
        entity_pages = item.get('entity_pages', {})
        wiki_contexts = entity_pages.get('wiki_context', [])
        if not wiki_contexts:
            continue
        doc_text = wiki_contexts[0]

        question = item.get('question', '')
        answer_data = item.get('answer', {})
        answer_text = answer_data.get('value', '') if isinstance(answer_data, dict) else str(answer_data)

        if not question or not answer_text or not doc_text:
            continue

        wc = count_words(doc_text)
        if wc < 50 or wc > 5000:
            continue
        if answer_text.lower() not in doc_text.lower():
            continue

        tqa_samples.append({
            'passage': doc_text,
            'query': question,
            'answer': answer_text,
            'word_count': wc,
            'dataset': 'triviaqa',
        })

        if len(tqa_samples) >= N_PER_DATASET * 3:
            break

    np.random.shuffle(tqa_samples)
    tqa_samples = tqa_samples[:N_PER_DATASET]

    with open(TQA_CACHE, 'w') as f:
        json.dump(tqa_samples, f)
    print(f"Cached {len(tqa_samples)} samples")

    del tqa_ds
    gc.collect()

print(f"TriviaQA samples: {len(tqa_samples)}")
wcs = [s['word_count'] for s in tqa_samples]
print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
if tqa_samples:
    print(f"  Example Q: {tqa_samples[0]['query']}")
    print(f"  Example A: {tqa_samples[0]['answer']}")\
""")))

# ========== Cell 5: Load Natural Questions ==========
cells.append(make_cell("code", s("""\
# Cell 5: Load Natural Questions dataset (streaming)
print("=" * 70)
print("LOADING NATURAL QUESTIONS (validation, streaming)")
print("=" * 70)

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
    print(f"  Example Q: {nq_samples[0]['query']}")
    print(f"  Example A: {nq_samples[0]['answer']}")\
""")))

# ========== Cell 6: Load HotpotQA ==========
cells.append(make_cell("code", s("""\
# Cell 6: Load HotpotQA dataset
print("=" * 70)
print("LOADING HOTPOTQA (distractor, validation)")
print("=" * 70)

HQA_CACHE = RESULTS_DIR / "hqa_samples.json"

if HQA_CACHE.exists():
    with open(HQA_CACHE, 'r') as f:
        hqa_samples = json.load(f)
    print(f"Loaded {len(hqa_samples)} cached HotpotQA samples")
else:
    hqa_ds = load_dataset("hotpot_qa", "distractor", split="validation",
                           trust_remote_code=True)
    print(f"HotpotQA validation size: {len(hqa_ds)}")

    hqa_samples = []
    np.random.seed(SEED)

    for item in tqdm(hqa_ds, desc="Filtering HotpotQA"):
        question = item.get('question', '')
        answer_text = item.get('answer', '')

        if not question or not answer_text:
            continue

        context = item.get('context', {})
        titles = context.get('title', [])
        sentences_list = context.get('sentences', [])

        paragraphs = []
        for title, sents in zip(titles, sentences_list):
            para_text = ''.join(sents)
            paragraphs.append(para_text)

        doc_text = '\\n\\n'.join(paragraphs)
        wc = count_words(doc_text)

        if wc < 100 or wc > 3000:
            continue

        hqa_samples.append({
            'passage': doc_text,
            'query': question,
            'answer': answer_text,
            'word_count': wc,
            'dataset': 'hotpotqa',
        })

        if len(hqa_samples) >= N_PER_DATASET * 3:
            break

    np.random.shuffle(hqa_samples)
    hqa_samples = hqa_samples[:N_PER_DATASET]

    with open(HQA_CACHE, 'w') as f:
        json.dump(hqa_samples, f)
    print(f"Cached {len(hqa_samples)} samples")

    del hqa_ds
    gc.collect()

print(f"HotpotQA samples: {len(hqa_samples)}")
wcs = [s['word_count'] for s in hqa_samples]
print(f"  Word counts: mean={np.mean(wcs):.0f}, min={min(wcs)}, max={max(wcs)}")
if hqa_samples:
    print(f"  Example Q: {hqa_samples[0]['query']}")
    print(f"  Example A: {hqa_samples[0]['answer']}")\
""")))

# ========== Cell 7: Unified sample pool ==========
cells.append(make_cell("code", s("""\
# Cell 7: Unified sample pool + tokenize prefix + condition explanation
print("=" * 70)
print("UNIFIED SAMPLE POOL")
print("=" * 70)

all_samples = []
for ds_name, ds_samples in [("triviaqa", tqa_samples), ("nq", nq_samples), ("hotpotqa", hqa_samples)]:
    for sample in ds_samples:
        sample['dataset'] = ds_name
    all_samples.extend(ds_samples)

print(f"Total samples: {len(all_samples)}")
for ds_name in ['triviaqa', 'nq', 'hotpotqa']:
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

for ds_name in ['triviaqa', 'nq', 'hotpotqa']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
    tls = [s['doc_token_len'] for s in ds_s]
    n_trunc = sum(1 for s in ds_s if s['doc_token_len'] == MAX_DOC_TOKENS)
    print(f"  {ds_name}: mean_tok={np.mean(tls):.0f}, median={np.median(tls):.0f}, "
          f"truncated={n_trunc}/{len(ds_s)} ({100*n_trunc/len(ds_s):.0f}%)")

# Condition explanation
print("\\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS (Gemma 3 4B)")
print("=" * 70)

print("\\n### 1. bare ###")
print("  Forward: [BOS][doc]")
print("  Baseline. Standard causal attention.")

print("\\n### 2. sf_trunc (standard priming) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc]")
print("  Standard causal, truncate + RoPE. Expected d ~ -0.03 (Exp 16 on MARCO).")
print("  Keys carry negative interference on Gemma.")

print("\\n### 3. sf_trunc_bias2 (attention forcing, bias=+2.0) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc] with +2.0 bias")
print("  Novel on Gemma. May amplify key interference.")

print("\\n### 4. values_only (all layers) ###")
print("  Bare keys + all primed values from sf_trunc cache.")
print("  Expected d ~ +0.056 (Exp 16). Bypasses key interference.")

print("\\n### 5. values_early (layers 0-15 only) ###")
print("  Bare keys + primed values from layers 0-15 only.")
print("  Expected best: d ~ +0.211 (Exp 19). Late layers carry interference.")

print("\\n### 6. values_hero (layers {10,12,14,15,20}) ###")
print("  Bare keys + primed values from 5 hero layers identified in Exp 24.")
print("  Cherry-picking may outperform blanket cutoff=16.")\
""")))

# ========== Cell 8: Helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 8: Helper functions

def build_biased_causal_mask(total_len, prefix_start, prefix_end, bias_value, dtype, device):
    \"\"\"Build a 4D causal attention mask with logit bias on doc->prefix attention.\"\"\"
    mask = torch.zeros((total_len, total_len), dtype=dtype, device=device)
    causal = torch.triu(
        torch.ones(total_len, total_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    mask.masked_fill_(causal, float('-inf'))

    if bias_value != 0.0:
        doc_start = prefix_end
        mask[doc_start:, prefix_start:prefix_end] += bias_value

    return mask.unsqueeze(0).unsqueeze(0)


def run_single_sample(sample, model, tokenizer, exp_config, sf_ids, sf_str,
                      PREFIX_TOKEN_LEN, N_LAYERS, EARLY_LAYER_CUTOFF, HERO_LAYERS):
    \"\"\"Run all 6 conditions for a single sample. Returns dict of NLLs + metadata.\"\"\"
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

    # === 2. sf_trunc (standard priming, bias=0) ===
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    total_seq_len = primed_input.shape[1]
    prefix_start = 1
    prefix_end = 1 + sf_ids.shape[1]
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

    # === 3. sf_trunc_bias2 (attention forcing, bias=+2.0) ===
    mask_4d = build_biased_causal_mask(
        total_seq_len, prefix_start, prefix_end,
        2.0, model.dtype, exp_config.device)

    with torch.no_grad():
        primed_out = model(input_ids=primed_input,
                           attention_mask=mask_4d,
                           use_cache=True, return_dict=True)
    primed_full_b2 = _ensure_dynamic_cache(primed_out.past_key_values)
    del primed_out, mask_4d

    trunc_raw = extract_and_truncate_cache_with_bos(primed_full_b2, doc_len)
    del primed_full_b2

    bias2_cache = deepcopy_cache(trunc_raw)
    correct_rope_positions_with_bos(bias2_cache, prefix_offset, model)
    del trunc_raw

    bias2_nll = score_answer_with_cache(
        deepcopy_cache(bias2_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del bias2_cache

    # === 4. values_only (all layers) ===
    values_all_cache = deepcopy_cache(bare_cache)
    for layer_idx in range(N_LAYERS):
        primed_vals = _get_cache_values(sf_trunc_cache, layer_idx)
        _set_cache_values(values_all_cache, layer_idx, primed_vals.clone())

    values_only_nll = score_answer_with_cache(
        deepcopy_cache(values_all_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del values_all_cache

    # === 5. values_early (layers 0 to EARLY_LAYER_CUTOFF-1) ===
    early_layers = list(range(EARLY_LAYER_CUTOFF))
    values_early_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, early_layers)

    values_early_nll = score_answer_with_cache(
        deepcopy_cache(values_early_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del values_early_cache

    # === 6. values_hero (hero layers only) ===
    values_hero_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, HERO_LAYERS)

    values_hero_nll = score_answer_with_cache(
        deepcopy_cache(values_hero_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del values_hero_cache

    del bare_cache, sf_trunc_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'dataset': ds_name,
        'query': query,
        'answer': answer,
        'word_count': sample['word_count'],
        'doc_token_len': doc_len,
        'answer_token_len': sample.get('answer_token_len', 0),
        'bare': bare_nll,
        'sf_trunc': sf_trunc_nll,
        'sf_trunc_bias2': bias2_nll,
        'values_only': values_only_nll,
        'values_early': values_early_nll,
        'values_hero': values_hero_nll,
    }


# Verify mask for a toy example
print("Mask verification (toy: BOS + 3 prefix + 5 doc = 9 total):")
toy_mask = build_biased_causal_mask(9, 1, 4, 2.0, model.dtype, 'cpu')
m = toy_mask.squeeze()
print(f"  Shape: {toy_mask.shape}")
print(f"  Doc->Prefix bias (row 4, col 1): {m[4, 1].item():.1f} (expect +2.0)")
print(f"  Causal mask (row 3, col 5): {m[3, 5].item()} (expect -inf)")
del toy_mask, m
print("OK")\
""")))

# ========== Cell 9: Main experiment loop ==========
cells.append(make_cell("code", s("""\
# Cell 9: Main experiment loop

print("=" * 70)
print(f"EXPERIMENT: {len(all_samples)} samples, {len(CONDITION_NAMES)} conditions")
print(f"Model: Gemma 3 4B, MAX_DOC_TOKENS: {MAX_DOC_TOKENS}")
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
                  desc="Exp 27b"):
    sample = all_samples[qidx]

    result = run_single_sample(
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
# Cell 10: Per-dataset analysis
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("ANALYSIS: PER-DATASET RESULTS (Gemma 3 4B)")
print("=" * 70)

dataset_names = ['triviaqa', 'nq', 'hotpotqa']
analysis = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    n_ds = len(ds_results)
    if n_ds == 0:
        continue

    bare_arr = np.array([r['bare'] for r in ds_results])

    # Filter invalid
    valid = np.isfinite(bare_arr) & (bare_arr != 0)
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        c_arr = np.array([r[cname] for r in ds_results])
        valid &= np.isfinite(c_arr) & (c_arr != 0)

    n_valid = int(np.sum(valid))

    print(f"\\n{'='*70}")
    print(f"DATASET: {ds_name.upper()} (n={n_valid}/{n_ds}, "
          f"median bare NLL={np.median(bare_arr):.3f})")
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
    print(f"{'  ' + ds:>14}", end='')
print()
print("-" * 62)
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    print(f"{cname:<20}", end='')
    for ds in dataset_names:
        if ds in analysis and cname in analysis[ds]:
            d = analysis[ds][cname]['cohens_d']
            p = analysis[ds][cname]['p_value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"{d:>+10.3f}{sig:>4}", end='')
        else:
            print(f"{'n/a':>14}", end='')
    print()

# Compare with Exp 27 (Mistral) reference
print(f"\\n\\n{'='*90}")
print("COMPARISON: Gemma vs Mistral (Exp 27)")
print(f"{'='*90}")
print("\\nMistral Exp 27 results (for reference):")
print("  TriviaQA: sf_trunc d=-0.201, bias2 d=-0.162, values_only d=-0.126")
print("  NQ:       sf_trunc d=+0.010, bias2 d=+0.048, values_only d=+0.032")
print("  HotpotQA: sf_trunc d=-0.111, bias2 d=-0.061, values_only d=-0.200")

# Check bare NLL distributions
print(f"\\n\\nBARE NLL DISTRIBUTIONS (ceiling effect check):")
for ds in dataset_names:
    ds_r = [r for r in all_results if r['dataset'] == ds]
    bare = [r['bare'] for r in ds_r]
    pct_zero = 100 * np.mean(np.array(bare) < 0.01)
    print(f"  {ds:10s}: mean={np.mean(bare):.3f}, median={np.median(bare):.3f}, "
          f"pct_near_zero={pct_zero:.0f}%")\
""")))

# ========== Cell 11: Length and hardness analysis ==========
cells.append(make_cell("code", s("""\
# Cell 11: Length stratification + hardness interaction

print("=" * 70)
print("LENGTH STRATIFICATION (Gemma 3 4B)")
print("=" * 70)

length_strat = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if not ds_results:
        continue

    print(f"\\n--- {ds_name.upper()} ---")
    ds_length_data = {}

    for cname in ['sf_trunc', 'sf_trunc_bias2', 'values_only', 'values_early', 'values_hero']:
        print(f"  {cname}:")
        bin_ds = []
        for bin_label, bin_min, bin_max in LENGTH_BINS:
            bin_results = [r for r in ds_results
                          if bin_min <= r['doc_token_len'] < bin_max]
            n_bin = len(bin_results)
            if n_bin < 10:
                print(f"    {bin_label}: n={n_bin} (too few)")
                bin_ds.append({'label': bin_label, 'n': n_bin, 'd': None})
                continue
            bare = np.array([r['bare'] for r in bin_results])
            cond = np.array([r[cname] for r in bin_results])
            delta = bare - cond
            d = cohens_d(delta)
            _, p_val = stats.ttest_1samp(delta, 0)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"    {bin_label}: n={n_bin}, d={d:+.3f}, p={p_val:.2e} {sig}")
            bin_ds.append({'label': bin_label, 'n': n_bin, 'd': float(d), 'p': float(p_val)})
        ds_length_data[cname] = bin_ds

    length_strat[ds_name] = ds_length_data

# === HARDNESS QUINTILE INTERACTION ===
print(f"\\n\\n{'='*70}")
print("HARDNESS QUINTILE INTERACTION (Gemma 3 4B)")
print(f"{'='*70}")

hardness_data = {}
quintile_labels = ['Q1(easy)', 'Q2', 'Q3', 'Q4', 'Q5(hard)']

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if len(ds_results) < 50:
        continue

    bare_arr = np.array([r['bare'] for r in ds_results])
    quintile_boundaries = np.percentile(bare_arr, [20, 40, 60, 80])
    print(f"\\n--- {ds_name.upper()} (boundaries: {[f'{b:.3f}' for b in quintile_boundaries]}) ---")

    def get_quintile(nll):
        for i, b in enumerate(quintile_boundaries):
            if nll <= b:
                return i
        return 4

    quintiles = np.array([get_quintile(r['bare']) for r in ds_results])

    ds_hardness = {}
    for cname in ['sf_trunc_bias2', 'values_only', 'values_early', 'values_hero']:
        cond_arr = np.array([r[cname] for r in ds_results])
        delta = bare_arr - cond_arr
        row = f"  {cname:<20}"
        q_ds = []
        for q in range(5):
            mask_q = quintiles == q
            n_q = int(np.sum(mask_q))
            if n_q < 5:
                row += f"{'n/a':>12}"
                q_ds.append(None)
            else:
                d_q = cohens_d(delta[mask_q])
                row += f"{d_q:>+12.3f}"
                q_ds.append(float(d_q))
        d_all = cohens_d(delta)
        row += f"{d_all:>+12.3f}"
        print(f"  {'':20}" + "".join(f"{ql:>12}" for ql in quintile_labels) + f"{'Overall':>12}")
        print(row)
        ds_hardness[cname] = q_ds

    hardness_data[ds_name] = ds_hardness\
""")))

# ========== Cell 12: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 12: Multi-panel figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

colors = {
    'sf_trunc': '#1f77b4',
    'sf_trunc_bias2': '#d62728',
    'values_only': '#7f7f7f',
    'values_early': '#2ca02c',
    'values_hero': '#ff7f0e',
}

# ---- Panel (a): Cohen's d by dataset x condition ----
ax = axes[0, 0]
x = np.arange(len(dataset_names))
width = 0.15
conds_plot = ['sf_trunc', 'sf_trunc_bias2', 'values_only', 'values_early', 'values_hero']
for i, cname in enumerate(conds_plot):
    ds_vals = []
    for ds in dataset_names:
        if ds in analysis and cname in analysis[ds]:
            ds_vals.append(analysis[ds][cname]['cohens_d'])
        else:
            ds_vals.append(0)
    offset = (i - 2) * width
    bars = ax.bar(x + offset, ds_vals, width, label=cname, color=colors[cname],
                  edgecolor='black', linewidth=0.5)
    for j, val in enumerate(ds_vals):
        ax.text(x[j] + offset, val + (0.01 if val >= 0 else -0.03),
                f"{val:+.2f}", ha='center', va='bottom' if val >= 0 else 'top', fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels([ds.upper() for ds in dataset_names])
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("(a) Gemma 3 4B: Effect Size by Dataset x Condition")
ax.legend(fontsize=6, loc='best')

# ---- Panel (b): Length stratification for values_early across datasets ----
ax = axes[0, 1]
for ds_name in dataset_names:
    if ds_name not in length_strat:
        continue
    cname = 'values_early'
    if cname not in length_strat[ds_name]:
        continue
    bins_data = length_strat[ds_name][cname]
    valid_idx = [i for i, b in enumerate(bins_data) if b['d'] is not None]
    if valid_idx:
        x_vals = valid_idx
        y_vals = [bins_data[i]['d'] for i in valid_idx]
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6, label=ds_name)

bin_labels_all = [b[0] for b in LENGTH_BINS]
ax.set_xticks(range(len(bin_labels_all)))
ax.set_xticklabels(bin_labels_all, rotation=30, ha='right', fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d")
ax.set_xlabel("Document Token Length Bin")
ax.set_title("(b) values_early (layers 0-15) by Length")
ax.legend(fontsize=8)

# ---- Panel (c): Hardness heatmap for values_early ----
ax = axes[1, 0]
hm_rows = []
hm_ylabels = []
for ds_name in dataset_names:
    if ds_name in hardness_data and 'values_early' in hardness_data[ds_name]:
        row = hardness_data[ds_name]['values_early']
        hm_rows.append([v if v is not None else 0 for v in row])
        hm_ylabels.append(ds_name.upper())

if hm_rows:
    hm_arr = np.array(hm_rows)
    im = ax.imshow(hm_arr, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(quintile_labels, fontsize=8)
    ax.set_yticks(range(len(hm_ylabels)))
    ax.set_yticklabels(hm_ylabels)
    for i in range(len(hm_ylabels)):
        for j in range(5):
            val = hm_arr[i, j]
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    fontsize=9, color='white' if abs(val) > 0.25 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")
ax.set_title("(c) values_early: Hardness x Dataset")

# ---- Panel (d): Gemma conditions comparison (best per dataset) ----
ax = axes[1, 1]
conds_compare = ['sf_trunc', 'values_only', 'values_early', 'values_hero']
x = np.arange(len(conds_compare))
width = 0.25
for i, ds_name in enumerate(dataset_names):
    ds_vals = []
    for cname in conds_compare:
        if ds_name in analysis and cname in analysis[ds_name]:
            ds_vals.append(analysis[ds_name][cname]['cohens_d'])
        else:
            ds_vals.append(0)
    offset = (i - 1) * width
    ax.bar(x + offset, ds_vals, width, label=ds_name, alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([c.replace('values_', 'val_') for c in conds_compare], fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
# Reference lines from Gemma Exp 19 (MARCO)
ax.axhline(y=0.211, color='green', linestyle=':', alpha=0.5,
           label='Exp 19 MARCO val_early d=+0.211')
ax.axhline(y=0.056, color='gray', linestyle=':', alpha=0.5,
           label='Exp 16 MARCO val_only d=+0.056')
ax.set_ylabel("Cohen's d")
ax.set_title("(d) Gemma Toolkit Comparison by Dataset")
ax.legend(fontsize=6)

plt.suptitle('Exp 27b: Cross-Dataset Generalization (Gemma 3 4B)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 13: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 13: Save results.json + CSV

# --- CSV ---
with open(CSV_PATH, 'w', newline='') as f:
    fieldnames = ['query_idx', 'dataset', 'query', 'answer', 'word_count',
                  'doc_token_len', 'answer_token_len',
                  'bare', 'sf_trunc', 'sf_trunc_bias2',
                  'values_only', 'values_early', 'values_hero']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_results:
        writer.writerow({k: r.get(k, '') for k in fieldnames})
print(f"CSV saved: {CSV_PATH}")

# --- Verdict ---
best_ds = None
best_cond = None
best_d = -999
for ds_name in dataset_names:
    if ds_name not in analysis:
        continue
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        if cname in analysis[ds_name]:
            d = analysis[ds_name][cname]['cohens_d']
            if d > best_d:
                best_d = d
                best_ds = ds_name
                best_cond = cname

if best_d > 0.15:
    verdict = (f"SUCCESS: Gemma toolkit generalizes! Best: {best_ds}/{best_cond} "
               f"d={best_d:+.3f}")
elif best_d > 0.05:
    verdict = (f"PARTIAL: Weak generalization. Best: {best_ds}/{best_cond} "
               f"d={best_d:+.3f}")
else:
    verdict = (f"FAILURE: Gemma toolkit does NOT generalize beyond MARCO. "
               f"Best: {best_ds}/{best_cond} d={best_d:+.3f}")

# Check values_early on each dataset
vel_results = {}
for ds_name in dataset_names:
    if ds_name in analysis and 'values_early' in analysis[ds_name]:
        vel_results[ds_name] = analysis[ds_name]['values_early']['cohens_d']
vel_verdict = "values_early results: " + ", ".join(
    f"{ds}={d:+.3f}" for ds, d in vel_results.items())

print(f"\\nVERDICT: {verdict}")
print(f"VEL: {vel_verdict}")

# --- results.json ---
final = {
    'experiment': 'exp27b_cross_dataset_gemma',
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
        'length_bins': LENGTH_BINS,
    },
    'per_dataset_analysis': analysis,
    'length_stratification': length_strat,
    'hardness_data': hardness_data,
    'verdict': verdict,
    'vel_verdict': vel_verdict,
    'per_sample_results': all_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Final summary
print("\\n" + "=" * 70)
print("SUMMARY -- Exp 27b: Cross-Dataset (Gemma 3 4B)")
print("=" * 70)
for ds_name in dataset_names:
    if ds_name not in analysis:
        continue
    print(f"\\n  {ds_name.upper()}:")
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        if cname in analysis[ds_name]:
            a = analysis[ds_name][cname]
            sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
            print(f"    {cname:<20} d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  {sig}")

print(f"\\nVERDICT: {verdict}")
print(f"VEL: {vel_verdict}")
print(f"\\nDone!")\
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

output_path = "/home/jupyter/research/directed_kvcache_v2/27b_cross_dataset_gemma.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
