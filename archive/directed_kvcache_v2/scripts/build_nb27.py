#!/usr/bin/env python3
"""Build script for 27_cross_dataset_attention_forcing.ipynb

Exp 27: Cross-Dataset Generalization with Attention Forcing

All 26 experiments primarily tested on MS MARCO (short web snippets, ~130 tokens).
The Exp 26 breakthrough (attention forcing recovers d=+0.291 at 1024 tok on padded MARCO)
has never been tested on other datasets or on naturally long documents.

Core question: Does attention forcing generalize beyond MS MARCO?
Does value contamination work on other QA datasets when we apply our full toolkit?

Datasets:
  - TriviaQA (rc.wikipedia): factoid QA over Wikipedia articles (500-5000 tok)
  - Natural Questions: factoid QA over full Wikipedia (150-6000 tok)
  - HotpotQA (distractor): multi-hop reasoning (800-2000 tok)

Conditions (5 per dataset):
  1. bare: BOS + doc standard causal (baseline)
  2. sf_trunc: Standard priming (bias=0, truncate + RoPE correct)
  3. sf_trunc_bias2: Attention forcing with +2.0 logit bias (optimal from Exp 26)
  4. sf_trunc_bias4: Attention forcing with +4.0 logit bias (may be needed for longer docs)
  5. values_only: Bare keys + primed values from sf_trunc cache

Model: Mistral-7B-Instruct-v0.2 (4-bit, float16)
N = 300 per dataset (900 total queries)
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
# Exp 27: Cross-Dataset Generalization with Attention Forcing

## Motivation

All 26 experiments have been primarily tested on MS MARCO (short web snippets, ~130 tokens).
The few cross-dataset tests (Exp 11 on NQ, Exp 19v1 on 6 datasets, Exp 24 Part 2 on SQuAD)
used only **basic priming** -- no attention forcing, no values-only isolation on other datasets.
The Exp 26 breakthrough (attention forcing recovers d=+0.291 at 1024 tok on padded MARCO)
has **never been tested on other datasets or on naturally long documents**.

## Core Question

Does attention forcing generalize beyond MS MARCO? Does the mechanism (value contamination)
work on other QA datasets when we apply our full toolkit?

## Datasets

| Dataset | Doc Length | QA Type | Prior Result (basic priming) |
|---------|------------|---------|------------------------------|
| **TriviaQA** | 500-5000 tok | Factoid (closest to MARCO) | Never tested |
| **Natural Questions** | 150-6000 tok | Factoid, long Wikipedia | d=-0.019 (Exp 11) |
| **HotpotQA** | 800-2000 tok | Multi-hop reasoning | d=-0.35 (Exp 19v1) |

## Conditions

| # | Condition | Description |
|---|-----------|-------------|
| 1 | `bare` | BOS + doc standard causal (baseline) |
| 2 | `sf_trunc` | Standard priming (bias=0) + truncate + RoPE correct |
| 3 | `sf_trunc_bias2` | +2.0 logit bias on doc->prefix attention (optimal from Exp 26) |
| 4 | `sf_trunc_bias4` | +4.0 logit bias (may help with longer docs) |
| 5 | `values_only` | Bare keys + primed values from sf_trunc cache |

## Success Criteria

1. Does attention forcing (bias=2.0 or 4.0) show d > 0 on any non-MARCO dataset?
2. Does values_only show positive signal universally (confirming value contamination mechanism)?
3. Does the length x bias interaction hold: optimal bias increases with document length?""")))

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

RESULTS_DIR = Path("results/exp27")
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

# ========== Cell 2: Load Mistral 7B ==========
cells.append(make_cell("code", s("""\
# Cell 2: Load Mistral 7B via load_model()
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.model_utils import load_model

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

exp_config = ExperimentConfig(
    model_name=MODEL_NAME,
    model_type="mistral",
    compute_dtype="auto",
    use_4bit=True,
    num_samples=2000,
    seed=SEED,
)

print(f"Loading {MODEL_NAME} (4-bit, float16)...")
model, tokenizer = load_model(exp_config)

from lib.kv_cache import _get_text_config, _get_head_dim, _ensure_dynamic_cache, _get_cache_keys

text_config = _get_text_config(model.config)
N_LAYERS = text_config.num_hidden_layers
print(f"\\nModel loaded successfully.")
print(f"  Num layers: {N_LAYERS}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  Model dtype: {model.dtype}")

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
MAX_DOC_TOKENS = 4096
CHECKPOINT_EVERY = 25

# Conditions
CONDITION_NAMES = ['bare', 'sf_trunc', 'sf_trunc_bias2', 'sf_trunc_bias4', 'values_only']
BIAS_MAP = {
    'sf_trunc': 0.0,
    'sf_trunc_bias2': 2.0,
    'sf_trunc_bias4': 4.0,
}

# Length bins for stratified analysis (token count)
LENGTH_BINS = [
    ('<256', 0, 256),
    ('256-512', 256, 512),
    ('512-1024', 512, 1024),
    ('1024-2048', 1024, 2048),
    ('>2048', 2048, 999999),
]

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  N per dataset: {N_PER_DATASET}")
print(f"  MAX_DOC_TOKENS: {MAX_DOC_TOKENS}")
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
        # Extract document from entity_pages
        entity_pages = item.get('entity_pages', {})
        wiki_contexts = entity_pages.get('wiki_context', [])
        if not wiki_contexts:
            continue
        doc_text = wiki_contexts[0]  # first Wikipedia evidence article

        # Get question and answer
        question = item.get('question', '')
        answer_data = item.get('answer', {})
        answer_text = answer_data.get('value', '') if isinstance(answer_data, dict) else str(answer_data)

        if not question or not answer_text or not doc_text:
            continue

        # Filter: answer must appear in document, doc 200-5000 words
        wc = count_words(doc_text)
        if wc < 200 or wc > 5000:
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
    print(f"  Example A: {tqa_samples[0]['answer']}")
    print(f"  Doc preview: {tqa_samples[0]['passage'][:150]}...")\
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

        # Extract clean document text (non-HTML tokens)
        doc_tokens = example['document']['tokens']
        if isinstance(doc_tokens, dict):
            token_strs = doc_tokens['token']
            is_html_flags = doc_tokens['is_html']
            clean_tokens = [t for t, h in zip(token_strs, is_html_flags) if not h]
        else:
            clean_tokens = [t['token'] for t in doc_tokens if not t['is_html']]

        doc_text = ' '.join(clean_tokens)
        wc = count_words(doc_text)

        if wc < 100 or wc > 4000:
            continue

        # Extract short answer
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

        # Extract query
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
    print(f"  Example A: {nq_samples[0]['answer']}")
    print(f"  Doc preview: {nq_samples[0]['passage'][:150]}...")\
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

        # Concatenate context paragraphs (including distractors)
        context = item.get('context', {})
        titles = context.get('title', [])
        sentences_list = context.get('sentences', [])

        paragraphs = []
        for title, sents in zip(titles, sentences_list):
            para_text = ''.join(sents)
            paragraphs.append(para_text)

        doc_text = '\\n\\n'.join(paragraphs)
        wc = count_words(doc_text)

        if wc < 200 or wc > 3000:
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
    print(f"  Example A: {hqa_samples[0]['answer']}")
    print(f"  Doc preview: {hqa_samples[0]['passage'][:150]}...")\
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

# Tokenize doc lengths for all samples
print(f"\\nTokenizing documents to measure token lengths...")
for sample in tqdm(all_samples, desc="Tokenizing"):
    tok_len = len(tokenizer.encode(sample['passage'], add_special_tokens=False))
    sample['doc_token_len'] = min(tok_len, MAX_DOC_TOKENS)
    sample['answer_token_len'] = len(tokenizer.encode(sample['answer'], add_special_tokens=False))

# Token length summary
for ds_name in ['triviaqa', 'nq', 'hotpotqa']:
    ds_s = [s for s in all_samples if s['dataset'] == ds_name]
    tls = [s['doc_token_len'] for s in ds_s]
    print(f"  {ds_name} tokens: mean={np.mean(tls):.0f}, median={np.median(tls):.0f}, "
          f"range=[{min(tls)}, {max(tls)}]")

# Condition explanation
print("\\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

print("\\n### 1. bare ###")
print("  Forward: [BOS][doc]")
print("  Standard causal attention, no prefix. Baseline for all comparisons.")

print("\\n### 2. sf_trunc (bias=0.0) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc]")
print("  Standard priming: standard causal mask, then truncate prefix + RoPE correct.")
print("  This is the 'classic' priming that works on short MARCO but fails on long docs.")

print("\\n### 3. sf_trunc_bias2 (bias=+2.0) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc] with +2.0 logit bias")
print("  Every doc token gets +2.0 added to pre-softmax attention for prefix positions.")
print("  Optimal bias from Exp 26 (d=+0.291 at 1024 tok on padded MARCO).")

print("\\n### 4. sf_trunc_bias4 (bias=+4.0) ###")
print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc] with +4.0 logit bias")
print("  Stronger forcing. May be needed for docs longer than 1024 tokens.")

print("\\n### 5. values_only ###")
print("  Bare cache keys + primed cache values (from sf_trunc, bias=0).")
print("  Tests pure value contamination without key interference.")\
""")))

# ========== Cell 8: Helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 8: Helper functions

def build_biased_causal_mask(total_len, prefix_start, prefix_end, bias_value, dtype, device):
    \"\"\"Build a 4D causal attention mask with logit bias on doc->prefix attention.

    Args:
        total_len: Total sequence length [BOS + prefix + doc]
        prefix_start: Start index of prefix tokens (typically 1, after BOS)
        prefix_end: End index of prefix tokens (exclusive)
        bias_value: Positive float to add to doc->prefix attention scores.
        dtype: Model dtype
        device: Model device

    Returns:
        Tensor of shape (1, 1, total_len, total_len)
    \"\"\"
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


def run_single_sample(sample, model, tokenizer, exp_config, sf_ids, sf_str, PREFIX_TOKEN_LEN, N_LAYERS):
    \"\"\"Run all 5 conditions for a single sample. Returns dict of NLLs + metadata.\"\"\"
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

    # === 2-4. PRIMED with bias=0, 2.0, 4.0 ===
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    total_seq_len = primed_input.shape[1]
    prefix_start = 1
    prefix_end = 1 + sf_ids.shape[1]
    prefix_offset = sf_ids.shape[1]

    primed_nlls = {}
    sf_trunc_cache_for_values = None  # save bias=0 cache for values_only

    for cond_name, bias_value in BIAS_MAP.items():
        mask_4d = build_biased_causal_mask(
            total_seq_len, prefix_start, prefix_end,
            bias_value, model.dtype, exp_config.device)

        with torch.no_grad():
            primed_out = model(input_ids=primed_input,
                               attention_mask=mask_4d,
                               use_cache=True, return_dict=True)
        primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
        del primed_out, mask_4d

        trunc_raw = extract_and_truncate_cache_with_bos(primed_full, doc_len)
        del primed_full

        sf_trunc_cache = deepcopy_cache(trunc_raw)
        correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
        del trunc_raw

        nll = score_answer_with_cache(
            deepcopy_cache(sf_trunc_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        primed_nlls[cond_name] = nll

        if cond_name == 'sf_trunc':
            sf_trunc_cache_for_values = sf_trunc_cache
        else:
            del sf_trunc_cache

    # === 5. VALUES_ONLY ===
    # Bare keys + primed values from sf_trunc (bias=0) cache
    values_cache = deepcopy_cache(bare_cache)
    for layer_idx in range(N_LAYERS):
        primed_vals = _get_cache_values(sf_trunc_cache_for_values, layer_idx)
        _set_cache_values(values_cache, layer_idx, primed_vals.clone())

    values_nll = score_answer_with_cache(
        deepcopy_cache(values_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    del bare_cache, values_cache, sf_trunc_cache_for_values
    del bare_input, primed_input
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
        'sf_trunc': primed_nlls['sf_trunc'],
        'sf_trunc_bias2': primed_nlls['sf_trunc_bias2'],
        'sf_trunc_bias4': primed_nlls['sf_trunc_bias4'],
        'values_only': values_nll,
    }


# Verify mask for a toy example
print("Mask verification (toy: BOS + 3 prefix + 5 doc = 9 total):")
toy_mask = build_biased_causal_mask(9, 1, 4, 2.0, model.dtype, 'cpu')
m = toy_mask.squeeze()
print(f"  Shape: {toy_mask.shape}")
print(f"  Doc->Prefix bias (row 4, col 1): {m[4, 1].item():.1f} (expect +2.0)")
print(f"  Doc->Doc (row 5, col 4): {m[5, 4].item():.1f} (expect 0.0)")
print(f"  Causal mask (row 3, col 5): {m[3, 5].item()} (expect -inf)")
del toy_mask, m
print("OK")\
""")))

# ========== Cell 9: Main experiment loop ==========
cells.append(make_cell("code", s("""\
# Cell 9: Main experiment loop

print("=" * 70)
print(f"EXPERIMENT: {len(all_samples)} samples, {len(CONDITION_NAMES)} conditions")
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
                  desc="Exp 27"):
    sample = all_samples[qidx]

    result = run_single_sample(
        sample, model, tokenizer, exp_config,
        sf_ids, sf_str, PREFIX_TOKEN_LEN, N_LAYERS)
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
# Cell 10: Per-dataset analysis — Cohen's d, win rates, p-values
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("ANALYSIS: PER-DATASET RESULTS")
print("=" * 70)

dataset_names = ['triviaqa', 'nq', 'hotpotqa']
analysis = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    n_ds = len(ds_results)
    if n_ds == 0:
        print(f"\\n{ds_name}: NO RESULTS")
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
    print(f"DATASET: {ds_name.upper()} (n={n_valid}/{n_ds})")
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

# Cross-dataset summary
print(f"\\n\\n{'='*70}")
print("CROSS-DATASET SUMMARY (Cohen's d vs bare)")
print(f"{'='*70}")
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
    print()\
""")))

# ========== Cell 11: Length stratification + hardness analysis ==========
cells.append(make_cell("code", s("""\
# Cell 11: Length stratification + hardness + answer length interaction

print("=" * 70)
print("LENGTH STRATIFICATION ANALYSIS")
print("=" * 70)

length_strat = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if not ds_results:
        continue

    print(f"\\n--- {ds_name.upper()} ---")
    ds_length_data = {}

    for cname in ['sf_trunc', 'sf_trunc_bias2', 'sf_trunc_bias4', 'values_only']:
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
print("HARDNESS QUINTILE INTERACTION")
print(f"{'='*70}")

hardness_data = {}

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if len(ds_results) < 50:
        continue

    bare_arr = np.array([r['bare'] for r in ds_results])
    quintile_boundaries = np.percentile(bare_arr, [20, 40, 60, 80])
    quintile_labels = ['Q1(easy)', 'Q2', 'Q3', 'Q4', 'Q5(hard)']

    def get_quintile(nll):
        for i, b in enumerate(quintile_boundaries):
            if nll <= b:
                return i
        return 4

    quintiles = np.array([get_quintile(r['bare']) for r in ds_results])

    print(f"\\n--- {ds_name.upper()} ---")
    ds_hardness = {}

    for cname in ['sf_trunc', 'sf_trunc_bias2', 'values_only']:
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

    hardness_data[ds_name] = ds_hardness

# === ANSWER LENGTH INTERACTION ===
print(f"\\n\\n{'='*70}")
print("ANSWER LENGTH INTERACTION")
print(f"{'='*70}")

answer_len_data = {}
answer_bins = [('short(<5)', 0, 5), ('medium(5-15)', 5, 15), ('long(>15)', 15, 9999)]

for ds_name in dataset_names:
    ds_results = [r for r in all_results if r['dataset'] == ds_name]
    if len(ds_results) < 50:
        continue

    print(f"\\n--- {ds_name.upper()} ---")
    ds_ans_data = {}

    for cname in ['sf_trunc_bias2', 'values_only']:
        bare_arr = np.array([r['bare'] for r in ds_results])
        cond_arr = np.array([r[cname] for r in ds_results])
        delta = bare_arr - cond_arr
        a_lens = np.array([r['answer_token_len'] for r in ds_results])

        row_parts = []
        for a_label, a_min, a_max in answer_bins:
            mask_a = (a_lens >= a_min) & (a_lens < a_max)
            n_a = int(np.sum(mask_a))
            if n_a < 10:
                row_parts.append(f"  {a_label}: n={n_a} (too few)")
            else:
                d_a = cohens_d(delta[mask_a])
                row_parts.append(f"  {a_label}: n={n_a}, d={d_a:+.3f}")
        print(f"  {cname}: " + ", ".join(row_parts))
        ds_ans_data[cname] = row_parts

    answer_len_data[ds_name] = ds_ans_data\
""")))

# ========== Cell 12: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 12: Multi-panel figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

colors = {
    'sf_trunc': '#1f77b4',
    'sf_trunc_bias2': '#d62728',
    'sf_trunc_bias4': '#ff7f0e',
    'values_only': '#2ca02c',
}

# ---- Panel (a): Cohen's d by dataset x condition ----
ax = axes[0, 0]
x = np.arange(len(dataset_names))
width = 0.18
for i, cname in enumerate(['sf_trunc', 'sf_trunc_bias2', 'sf_trunc_bias4', 'values_only']):
    ds_vals = []
    for ds in dataset_names:
        if ds in analysis and cname in analysis[ds]:
            ds_vals.append(analysis[ds][cname]['cohens_d'])
        else:
            ds_vals.append(0)
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, ds_vals, width, label=cname, color=colors[cname],
                  edgecolor='black', linewidth=0.5)
    for j, val in enumerate(ds_vals):
        ax.text(x[j] + offset, val + (0.01 if val >= 0 else -0.03),
                f"{val:+.2f}", ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([ds.upper() for ds in dataset_names])
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("(a) Effect Size by Dataset x Condition")
ax.legend(fontsize=7, loc='best')

# ---- Panel (b): Length stratification for bias2 across datasets ----
ax = axes[0, 1]
for ds_idx, ds_name in enumerate(dataset_names):
    if ds_name not in length_strat:
        continue
    cname = 'sf_trunc_bias2'
    if cname not in length_strat[ds_name]:
        continue
    bins_data = length_strat[ds_name][cname]
    bin_labels = [b['label'] for b in bins_data]
    bin_ds = [b['d'] if b['d'] is not None else 0 for b in bins_data]
    bin_ns = [b['n'] for b in bins_data]
    # Only plot bins with enough data
    valid_idx = [i for i, b in enumerate(bins_data) if b['d'] is not None]
    if valid_idx:
        x_vals = [i for i in valid_idx]
        y_vals = [bin_ds[i] for i in valid_idx]
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                label=ds_name)

bin_labels_all = [b[0] for b in LENGTH_BINS]
ax.set_xticks(range(len(bin_labels_all)))
ax.set_xticklabels(bin_labels_all, rotation=30, ha='right', fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d")
ax.set_xlabel("Document Token Length Bin")
ax.set_title("(b) Attention Forcing (bias=2.0) by Length")
ax.legend(fontsize=8)

# ---- Panel (c): Hardness heatmap for bias2 ----
ax = axes[1, 0]
quintile_labels = ['Q1\\n(easy)', 'Q2', 'Q3', 'Q4', 'Q5\\n(hard)']
hm_rows = []
hm_ylabels = []
for ds_name in dataset_names:
    if ds_name in hardness_data and 'sf_trunc_bias2' in hardness_data[ds_name]:
        row = hardness_data[ds_name]['sf_trunc_bias2']
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
ax.set_title("(c) Hardness x Dataset (bias=2.0)")

# ---- Panel (d): Bias tuning per dataset ----
ax = axes[1, 1]
bias_vals_plot = [0.0, 2.0, 4.0]
bias_labels_plot = ['bias=0', 'bias=2', 'bias=4']
cond_for_bias = ['sf_trunc', 'sf_trunc_bias2', 'sf_trunc_bias4']

for ds_name in dataset_names:
    if ds_name not in analysis:
        continue
    d_vals = []
    for cname in cond_for_bias:
        if cname in analysis[ds_name]:
            d_vals.append(analysis[ds_name][cname]['cohens_d'])
        else:
            d_vals.append(0)
    ax.plot(bias_vals_plot, d_vals, marker='o', linewidth=2, markersize=8,
            label=ds_name)
    for i, (bv, dv) in enumerate(zip(bias_vals_plot, d_vals)):
        ax.annotate(f"{dv:+.2f}", (bv, dv), textcoords='offset points',
                    xytext=(5, 8), fontsize=7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Attention Bias Value')
ax.set_ylabel("Cohen's d")
ax.set_title("(d) Bias Tuning by Dataset")
ax.legend(fontsize=8)

plt.suptitle('Exp 27: Cross-Dataset Generalization with Attention Forcing', fontsize=14, y=1.02)
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
                  'bare', 'sf_trunc', 'sf_trunc_bias2', 'sf_trunc_bias4', 'values_only']
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
    for cname in ['sf_trunc_bias2', 'sf_trunc_bias4']:
        if cname in analysis[ds_name]:
            d = analysis[ds_name][cname]['cohens_d']
            if d > best_d:
                best_d = d
                best_ds = ds_name
                best_cond = cname

if best_d > 0.15:
    verdict = (f"SUCCESS: Attention forcing generalizes! Best: {best_ds}/{best_cond} "
               f"d={best_d:+.3f}")
elif best_d > 0.05:
    verdict = (f"PARTIAL: Weak generalization. Best: {best_ds}/{best_cond} "
               f"d={best_d:+.3f}")
else:
    verdict = (f"FAILURE: Attention forcing does NOT generalize beyond MS MARCO. "
               f"Best: {best_ds}/{best_cond} d={best_d:+.3f}")

# Check values_only universality
values_positive = []
for ds_name in dataset_names:
    if ds_name in analysis and 'values_only' in analysis[ds_name]:
        d = analysis[ds_name]['values_only']['cohens_d']
        values_positive.append(d > 0)
values_verdict = ("VALUES_ONLY universally positive" if all(values_positive)
                   else "VALUES_ONLY NOT universally positive")

print(f"\\nVERDICT: {verdict}")
print(f"VALUES: {values_verdict}")

# --- results.json ---
final = {
    'experiment': 'exp27_cross_dataset_attention_forcing',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'mistral',
        'seed': SEED,
        'n_per_dataset': N_PER_DATASET,
        'max_doc_tokens': MAX_DOC_TOKENS,
        'conditions': CONDITION_NAMES,
        'bias_map': {k: float(v) for k, v in BIAS_MAP.items()},
        'prefix': STATIC_FACT,
        'prefix_token_len': PREFIX_TOKEN_LEN,
        'datasets': dataset_names,
        'length_bins': LENGTH_BINS,
    },
    'per_dataset_analysis': analysis,
    'length_stratification': length_strat,
    'hardness_data': hardness_data,
    'verdict': verdict,
    'values_verdict': values_verdict,
    'per_sample_results': all_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Final summary
print("\\n" + "=" * 70)
print("SUMMARY -- Exp 27: Cross-Dataset Generalization")
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
print(f"VALUES: {values_verdict}")
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

output_path = "/home/jupyter/research/directed_kvcache_v2/27_cross_dataset_attention_forcing.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
