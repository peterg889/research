#!/usr/bin/env python3
"""Build script for 26_attention_forcing.ipynb

Exp 26: Attention Forcing for Long Documents

Experiments 11 and 20 proved that value contamination fails on documents longer
than 256 tokens. The prefix signal is diluted because attention weights for the
prefix become infinitesimally small as document length grows.

Core Question: If we bypass the model's natural attention decay and mathematically
force document tokens to pay attention to the prefix during cache generation, can
we recover the priming benefit for long documents?

Method: During the forward pass that builds the primed cache, pass a custom float
attention mask that adds a positive logit bias to the doc->prefix attention scores
(pre-softmax). This forces every document token to mix more prefix semantics into
its value vector. Then truncate + RoPE correct as normal.

Conditions (on 1024-token padded MS MARCO):
  - bare: BOS + doc cache (control)
  - bias_0.0: Standard priming, no bias (expected d ~ 0 at 1024 tokens)
  - bias_2.0: +2.0 logit bias on doc->prefix attention
  - bias_5.0: +5.0 logit bias on doc->prefix attention
  - bias_10.0: +10.0 logit bias on doc->prefix attention

Model: Mistral 7B (4-bit, float16) -- matches Exp 20 baseline.
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
# Exp 26: Attention Forcing for Long Documents

## Motivation

Experiments 11 and 20 proved that **value contamination fails on documents longer than 256 tokens**.
The prefix signal is diluted because attention weights for the prefix become infinitesimally small
as document length grows. Experiment 12 attempted to fix this by repeating the prefix 20 times,
but it failed — the model's attention mechanism naturally learns to ignore redundant tokens.

## Core Question

If we bypass the model's natural attention decay and **mathematically force** document tokens to
pay attention to the prefix during cache generation, can we recover the priming benefit?

## Theoretical Mechanism

In SDPA, the attention scores (pre-softmax) determine how much information flows from previous tokens
into the current token's representation. Normally, we pass a boolean causal mask (0 for visible,
-inf for masked). Instead, we pass a **float-based attention mask** with a positive logit bias
(e.g., +5.0) exclusively at the intersection of document token queries and prefix token keys.

This forces every document token to aggressively mix prefix semantics into its value vector,
artificially counteracting long-document dilution.

## Conditions (on 1024-token padded MS MARCO)

| Condition | Description |
|-----------|-------------|
| `bare` | BOS + doc cache (control) |
| `bias_0.0` | Standard priming, no bias (failure baseline, expected d ≈ 0) |
| `bias_2.0` | +2.0 logit bias on doc→prefix attention |
| `bias_5.0` | +5.0 logit bias on doc→prefix attention |
| `bias_10.0` | +10.0 logit bias on doc→prefix attention |

## Reference Values

| Source | Condition | d |
|--------|-----------|---|
| Exp 20 (Mistral) | full priming @ original (~130 tok) | +0.303 |
| Exp 20 (Mistral) | full priming @ 256 tok | +0.114 (ns) |
| Exp 20 (Mistral) | full priming @ 1024 tok | -0.043 (ns) |

## Success Criteria

1. Does any bias level recover a positive Cohen's d (> +0.20) at 1024 tokens?
2. At what bias does the document representation corrupt (NLL explodes)?
3. What is the optimal bias on the tuning curve?""")))

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

RESULTS_DIR = Path("results/exp26")
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
    compute_dtype="auto",  # resolves to float16 for Mistral
    use_4bit=True,
    num_samples=2000,
    seed=SEED,
)

print(f"Loading {MODEL_NAME} (4-bit, float16)...")
model, tokenizer = load_model(exp_config)

from lib.kv_cache import _get_text_config, _get_head_dim, _ensure_dynamic_cache, _get_cache_keys

text_config = _get_text_config(model.config)
print(f"\\nModel loaded successfully.")
print(f"  Model class: {type(model).__name__}")
print(f"  Hidden size: {text_config.hidden_size}")
print(f"  Num layers: {text_config.num_hidden_layers}")
print(f"  Num attention heads: {text_config.num_attention_heads}")
print(f"  Num KV heads: {text_config.num_key_value_heads}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  Model dtype: {model.dtype}")

# Verify dtype with a test forward pass
sample_ids = tokenizer("test", return_tensors="pt")['input_ids'].to(exp_config.device)
with torch.no_grad():
    out = model(sample_ids, use_cache=True)
    cache_check = _ensure_dynamic_cache(out.past_key_values)
    k0 = _get_cache_keys(cache_check, 0)
    print(f"  Cache key dtype: {k0.dtype}")
    print(f"  Cache key shape: {k0.shape}  (batch, kv_heads, seq, head_dim)")
del out, sample_ids, cache_check
torch.cuda.empty_cache()\
""")))

# ========== Cell 3: Lib imports + constants ==========
cells.append(make_cell("code", s("""\
# Cell 3: Lib imports + templates + constants
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
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Prefix text
from lib.surrogate import STATIC_SURROGATE_QUERIES
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Experiment parameters
N_QUERIES = 500
MAX_PASSAGE_WORDS = 300
PAD_TARGET = 1024  # pad all documents to this token length
CHECKPOINT_EVERY = 25

# Bias values to sweep (0.0 = standard priming, no bias)
BIAS_VALUES = [0.0, 2.0, 5.0, 10.0]

# Reference values from Exp 20 (Mistral)
EXP20_REF = {
    'original_d': 0.303,
    '256_d': 0.114,
    '1024_d': -0.043,
}

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  N_QUERIES: {N_QUERIES}")
print(f"  PAD_TARGET: {PAD_TARGET} tokens")
print(f"  BIAS_VALUES: {BIAS_VALUES}")
print(f"  Static fact prefix: '{STATIC_FACT}'")
print(f"\\nExp 20 reference (Mistral, standard priming):")
for k, v in EXP20_REF.items():
    print(f"    {k}: {v:+.3f}")\
""")))

# ========== Cell 4: Load MS MARCO + padding pool ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO v1.1, filter positive passages, build padding pool
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 -- POSITIVE PASSAGES ONLY")
print("=" * 70)

dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation",
                        trust_remote_code=True)
print(f"Total items in validation: {len(dataset)}")

queries = []
padding_passages = []
eval_passage_set = set()
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

    # Get best answer
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    else:
        # Still collect non-eval passages for padding pool
        for p in passage_texts:
            if count_words(p) <= MAX_PASSAGE_WORDS:
                padding_passages.append(p)
        continue

    # Find positive passage(s)
    for i, (ptext, sel) in enumerate(zip(passage_texts, is_selected)):
        if sel == 1 and count_words(ptext) <= MAX_PASSAGE_WORDS:
            if len(queries) < N_QUERIES * 3:  # collect 3x for shuffling
                queries.append({
                    'query': query,
                    'answer': answer,
                    'passage': ptext,
                    'word_count': count_words(ptext),
                })
                eval_passage_set.add(ptext)
                break

    # Collect non-eval passages for padding pool
    for p in passage_texts:
        if p not in eval_passage_set and count_words(p) <= MAX_PASSAGE_WORDS:
            padding_passages.append(p)

np.random.shuffle(queries)
queries = queries[:N_QUERIES]
N = len(queries)

print(f"\\nSelected {N} queries with positive passages")
print(f"Word counts: mean={np.mean([q['word_count'] for q in queries]):.0f}, "
      f"min={min(q['word_count'] for q in queries)}, "
      f"max={max(q['word_count'] for q in queries)}")

# Build padding pool (pre-tokenize)
print(f"\\nPadding pool passages: {len(padding_passages):,}")
padding_text = ' '.join(padding_passages)
padding_ids = tokenizer.encode(padding_text, add_special_tokens=False)
print(f"Padding pool tokens: {len(padding_ids):,}")

max_needed = PAD_TARGET * N_QUERIES
print(f"Max tokens needed: {max_needed:,}")
assert len(padding_ids) > max_needed, (
    f"Padding pool too small: {len(padding_ids):,} < {max_needed:,}"
)
print(f"Pool is {len(padding_ids) / max_needed:.1f}x the max needed. OK.")

del dataset
gc.collect()\
""")))

# ========== Cell 5: Tokenize prefix + condition explanations ==========
cells.append(make_cell("code", s("""\
# Cell 5: Tokenize prefix and explain experimental conditions

print("=" * 70)
print("PREFIX TOKENIZATION")
print("=" * 70)

sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)

sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(exp_config.device)
PREFIX_TOKEN_LEN = sf_ids.shape[1]

print(f"\\nStatic fact prefix: '{STATIC_FACT}'")
print(f"  Formatted: '{sf_str.strip()}'")
print(f"  Token length (no BOS): {PREFIX_TOKEN_LEN}")

# Verify BPE boundary consistency
print("\\nBPE BOUNDARY CHECK (first passage):")
example_doc = queries[0]['passage']
concat = sf_str + DOCUMENT_TEMPLATE.format(document=example_doc)
concat_enc = tokenizer(concat, add_special_tokens=True)['input_ids']
prefix_enc = tokenizer(sf_str, add_special_tokens=True)['input_ids']
doc_ids_from_concat = concat_enc[len(prefix_enc):]

bare_doc_enc = tokenizer(DOCUMENT_TEMPLATE.format(document=example_doc),
                          add_special_tokens=False)['input_ids']
match = sum(1 for a, b in zip(doc_ids_from_concat, bare_doc_enc) if a == b)
total = max(len(bare_doc_enc), 1)
print(f"  Token match: {match}/{total} ({100*match/total:.1f}%)")

# Condition explanations
print("\\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

print("\\nAll conditions use 1024-token padded documents from MS MARCO.")
print("Prefix: static_fact_trunc ('What are the key facts I need to know?')")
print(f"Prefix tokens: {PREFIX_TOKEN_LEN} (plus BOS = {PREFIX_TOKEN_LEN + 1} total prefix positions)")
print(f"Document tokens: {PAD_TARGET}")
print(f"Total sequence for primed passes: 1 + {PREFIX_TOKEN_LEN} + {PAD_TARGET} = {1 + PREFIX_TOKEN_LEN + PAD_TARGET}")

print("\\n### bare ###")
print("  Forward: [BOS][doc_1024]")
print("  Cache:   Standard causal attention, no prefix")
print("  Score:   Standard scoring against bare cache")

for bias in BIAS_VALUES:
    label = f"bias_{bias:.1f}"
    print(f"\\n### {label} ###")
    print(f"  Forward: [BOS][prefix_{PREFIX_TOKEN_LEN}][doc_{PAD_TARGET}]")
    if bias == 0.0:
        print("  Mask:    Standard causal (no bias)")
        print("  Note:    This is standard priming -- the failure baseline at 1024 tokens")
    else:
        print(f"  Mask:    Causal + {bias:+.1f} logit boost on doc->prefix attention")
        print(f"           Every doc token gets +{bias:.1f} added to its pre-softmax")
        print(f"           attention scores for the {PREFIX_TOKEN_LEN} prefix positions")
    print("  Post:    Truncate prefix -> RoPE correct -> score")\
""")))

# ========== Cell 6: Biased mask helper ==========
cells.append(make_cell("code", s("""\
# Cell 6: Helper function for building biased attention masks

def build_biased_causal_mask(total_len, prefix_start, prefix_end, bias_value, dtype, device):
    \"\"\"Build a 4D causal attention mask with logit bias on doc->prefix attention.

    Creates a standard causal mask (lower-triangular = 0, upper-triangular = -inf),
    then adds a positive bias to the attention scores at the intersection of
    document token queries (rows) and prefix token keys (columns).

    Args:
        total_len: Total sequence length [BOS + prefix + doc]
        prefix_start: Start index of prefix tokens (typically 1, after BOS)
        prefix_end: End index of prefix tokens (exclusive)
        bias_value: Positive float to add to doc->prefix attention scores.
            0.0 = standard causal mask (no bias).
        dtype: Model dtype (e.g., torch.float16)
        device: Model device

    Returns:
        Tensor of shape (1, 1, total_len, total_len)
    \"\"\"
    # Standard causal mask: 0 for attend, -inf for future positions
    mask = torch.zeros((total_len, total_len), dtype=dtype, device=device)
    causal = torch.triu(
        torch.ones(total_len, total_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    mask.masked_fill_(causal, float('-inf'))

    # Apply positive bias to doc->prefix attention
    # Doc tokens start at prefix_end, prefix tokens at [prefix_start, prefix_end)
    if bias_value != 0.0:
        doc_start = prefix_end
        mask[doc_start:, prefix_start:prefix_end] += bias_value

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len)


# Verify mask shape and values for a toy example
print("=" * 70)
print("MASK VERIFICATION (toy example)")
print("=" * 70)

# Toy: BOS + 3 prefix tokens + 5 doc tokens = 9 total
toy_mask = build_biased_causal_mask(
    total_len=9, prefix_start=1, prefix_end=4,
    bias_value=5.0, dtype=model.dtype, device='cpu'
)
m = toy_mask.squeeze()  # (9, 9)

print(f"\\nMask shape: {toy_mask.shape}")
print(f"Dtype: {toy_mask.dtype}")
print("\\nPositions: [BOS=0, P1=1, P2=2, P3=3, D1=4, D2=5, D3=6, D4=7, D5=8]")
print("\\nMask values (rows=queries, cols=keys):")
print("         BOS   P1    P2    P3    D1    D2    D3    D4    D5")
labels = ['BOS', 'P1 ', 'P2 ', 'P3 ', 'D1 ', 'D2 ', 'D3 ', 'D4 ', 'D5 ']
for i, label in enumerate(labels):
    row_vals = []
    for j in range(9):
        v = m[i, j].item()
        if v == float('-inf'):
            row_vals.append(' -inf')
        elif v == 0.0:
            row_vals.append('  0.0')
        else:
            row_vals.append(f'{v:+5.1f}')
    print(f"  {label}: {'  '.join(row_vals)}")

print("\\nKey observations:")
print("  - BOS (row 0) can only attend to itself (0.0)")
print("  - Prefix tokens (rows 1-3) attend causally to BOS + prior prefix (0.0)")
print("  - Doc tokens (rows 4-8) attend to prefix with +5.0 bias")
print("  - Doc tokens attend to BOS and other doc tokens normally (0.0)")
print("  - Upper triangle is -inf (causal masking)")

del toy_mask, m\
""")))

# ========== Cell 7: Main experiment loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main experiment loop

print("=" * 70)
print(f"EXPERIMENT: {N_QUERIES} queries, {len(BIAS_VALUES)} bias levels + bare")
print(f"Document length: {PAD_TARGET} tokens (padded)")
print("=" * 70)

# Checkpoint resume
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in queries[:N_QUERIES]]
    if ckpt_queries == current_queries:
        all_results = ckpt['results']
        start_idx = len(all_results)
        print(f"Resuming from checkpoint: {start_idx}/{N_QUERIES}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

t_start = time.time()

for qidx in tqdm(range(start_idx, N_QUERIES), initial=start_idx, total=N_QUERIES,
                  desc="Exp 26"):
    qdata = queries[qidx]
    query_prompt = QUERY_TEMPLATE.format(query=qdata['query'])
    answer_text = ANSWER_TEMPLATE.format(answer=qdata['answer'])
    passage = qdata['passage']
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # === Matched tokenization ===
    # Tokenize concatenated prefix+doc to get matched BPE boundaries
    full_text = sf_str + document_text
    full_enc = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=True, padding=False, truncation=False)
    full_ids = full_enc['input_ids'].to(exp_config.device)

    sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                               add_special_tokens=True, padding=False, truncation=False)
    sf_prefix_len_with_bos = sf_prefix_enc['input_ids'].shape[1]

    bos_id = full_ids[:, :1]
    base_doc_ids = full_ids[:, sf_prefix_len_with_bos:]
    base_doc_len = base_doc_ids.shape[1]

    del full_enc, full_ids, sf_prefix_enc

    # === Pad doc to PAD_TARGET tokens ===
    if base_doc_len < PAD_TARGET:
        pad_needed = PAD_TARGET - base_doc_len
        max_start = len(padding_ids) - pad_needed
        start = np.random.randint(0, max_start)
        pad_tensor = torch.tensor([padding_ids[start:start + pad_needed]],
                                   device=exp_config.device)
        doc_ids = torch.cat([base_doc_ids, pad_tensor], dim=1)
    else:
        doc_ids = base_doc_ids[:, :PAD_TARGET]

    doc_len = doc_ids.shape[1]
    context_len = 1 + doc_len  # BOS + doc

    # === Forward pass: BARE ===
    bare_input = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_input,
                         attention_mask=torch.ones_like(bare_input),
                         use_cache=True, return_dict=True)
    bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
    del bare_out

    # Score bare
    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    # === Forward passes: BIASED (one per bias level) ===
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    total_seq_len = primed_input.shape[1]
    prefix_start = 1  # prefix starts after BOS
    prefix_end = 1 + sf_ids.shape[1]  # prefix ends before doc
    prefix_offset = sf_ids.shape[1]  # for RoPE correction

    query_rows = []

    for bias_value in BIAS_VALUES:
        # Build 4D attention mask with bias
        mask_4d = build_biased_causal_mask(
            total_seq_len, prefix_start, prefix_end,
            bias_value, model.dtype, exp_config.device)

        # Forward pass with custom mask
        with torch.no_grad():
            primed_out = model(input_ids=primed_input,
                               attention_mask=mask_4d,
                               use_cache=True, return_dict=True)
        primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
        del primed_out, mask_4d

        # Truncate: keep [BOS] + [last doc_len positions]
        trunc_raw = extract_and_truncate_cache_with_bos(primed_full, doc_len)
        del primed_full

        # RoPE correct
        sf_trunc_cache = deepcopy_cache(trunc_raw)
        correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
        del trunc_raw

        # Score
        biased_nll = score_answer_with_cache(
            deepcopy_cache(sf_trunc_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del sf_trunc_cache

        query_rows.append({
            'query_idx': qidx,
            'bias_value': bias_value,
            'bias_label': f"bias_{bias_value:.1f}",
            'actual_doc_len': doc_len,
            'bare_nll': bare_nll,
            'primed_nll': biased_nll,
            'delta_nll': bare_nll - biased_nll,
        })

    del bare_cache, bare_input, primed_input
    if base_doc_len < PAD_TARGET:
        del pad_tensor
    gc.collect()
    torch.cuda.empty_cache()

    all_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'base_doc_len': base_doc_len,
        'padded_doc_len': doc_len,
        'bare_nll': bare_nll,
        'rows': query_rows,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_QUERIES - 1:
        ckpt_data = {
            'results': all_results,
            'query_texts': [q['query'] for q in queries[:N_QUERIES]],
            'completed': len(all_results),
            'total': N_QUERIES,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_QUERIES - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_QUERIES} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nExperiment complete: {len(all_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 8: Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 8: Analysis -- Cohen's d, statistical tests, perplexity check

import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("ANALYSIS: ATTENTION FORCING RESULTS")
print("=" * 70)

# Collect per-sample deltas by bias level
bias_deltas = {}
bias_bare = {}
bias_primed = {}
for bv in BIAS_VALUES:
    label = f"bias_{bv:.1f}"
    bias_deltas[label] = []
    bias_bare[label] = []
    bias_primed[label] = []

for r in all_results:
    for row in r['rows']:
        label = row['bias_label']
        if label in bias_deltas:
            bias_deltas[label].append(row['delta_nll'])
            bias_bare[label].append(row['bare_nll'])
            bias_primed[label].append(row['primed_nll'])

# Convert to arrays and filter invalid values
bias_arrays = {}
for label in bias_deltas:
    bare = np.array(bias_bare[label])
    primed = np.array(bias_primed[label])
    delta = np.array(bias_deltas[label])
    valid = (bare != 0) & (primed != 0) & np.isfinite(bare) & np.isfinite(primed)
    bias_arrays[label] = {
        'bare': bare[valid],
        'primed': primed[valid],
        'delta': delta[valid],
        'n_valid': int(np.sum(valid)),
    }

# Summary table
print(f"\\n{'Condition':<14} {'N':>5} {'Mean Bare':>10} {'Mean Primed':>12} "
      f"{'Mean D':>10} {'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
print("-" * 95)

analysis = {}
for bv in BIAS_VALUES:
    label = f"bias_{bv:.1f}"
    a = bias_arrays[label]
    d = cohens_d(a['delta'])
    win = np.mean(a['delta'] > 0) * 100
    t_stat, p_val = stats.ttest_1samp(a['delta'], 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{label:<14} {a['n_valid']:>5} {np.mean(a['bare']):>10.4f} "
          f"{np.mean(a['primed']):>12.4f} {np.mean(a['delta']):>+10.4f} "
          f"{d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
    analysis[label] = {
        'bias_value': bv,
        'n_valid': a['n_valid'],
        'mean_bare': float(np.mean(a['bare'])),
        'mean_primed': float(np.mean(a['primed'])),
        'mean_delta': float(np.mean(a['delta'])),
        'std_delta': float(np.std(a['delta'])),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }

# Perplexity check
print("\\n" + "=" * 70)
print("PERPLEXITY CHECK")
print("=" * 70)
print("\\nDoes attention forcing corrupt the document representation?")
print("If mean primed NLL is much higher than mean bare NLL, the bias is too aggressive.")
print()

bare_mean = np.mean(bias_arrays['bias_0.0']['bare'])
for bv in BIAS_VALUES:
    label = f"bias_{bv:.1f}"
    a = bias_arrays[label]
    primed_mean = np.mean(a['primed'])
    ratio = primed_mean / bare_mean if bare_mean > 0 else float('inf')
    corruption = "OK" if ratio < 1.5 else "WARNING" if ratio < 3.0 else "CORRUPTED"
    print(f"  {label}: bare={bare_mean:.4f}, primed={primed_mean:.4f}, "
          f"ratio={ratio:.2f}x  [{corruption}]")

# Exp 20 comparison
print("\\n" + "=" * 70)
print("COMPARISON WITH EXP 20 (standard priming)")
print("=" * 70)

d_standard = analysis['bias_0.0']['cohens_d']
print(f"\\nExp 20 standard priming @ 1024 tok: d={EXP20_REF['1024_d']:+.3f}")
print(f"This exp bias_0.0 (standard priming):  d={d_standard:+.3f}")

best_label = max(analysis, key=lambda k: analysis[k]['cohens_d'])
best_d = analysis[best_label]['cohens_d']
best_bias = analysis[best_label]['bias_value']

print(f"\\nBest bias: {best_label} (d={best_d:+.3f})")
if best_d > 0.20:
    verdict = (f"SUCCESS: Attention forcing recovers a meaningful effect (d={best_d:+.3f}) "
               f"at 1024 tokens with bias={best_bias:.1f}")
elif best_d > 0.05:
    verdict = (f"PARTIAL: Some recovery (d={best_d:+.3f}) but below the +0.20 target. "
               f"Best bias={best_bias:.1f}")
elif best_d > d_standard + 0.05:
    verdict = (f"MARGINAL: Bias helps vs standard (d={best_d:+.3f} vs {d_standard:+.3f}) "
               f"but effect is small")
else:
    verdict = (f"FAILURE: Attention forcing does not recover priming at 1024 tokens. "
               f"Best d={best_d:+.3f}, standard d={d_standard:+.3f}")

print(f"\\nVERDICT: {verdict}")

# Hardness interaction
print("\\n" + "=" * 70)
print("HARDNESS INTERACTION (quintiles by bare NLL)")
print("=" * 70)

all_bare_nlls = np.array([r['bare_nll'] for r in all_results])
quintile_boundaries = np.percentile(all_bare_nlls, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in all_bare_nlls])

header = f"{'Condition':<14}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (14 + 14 * 6))

hardness_data = {}
for bv in BIAS_VALUES:
    label = f"bias_{bv:.1f}"
    delta = np.array(bias_deltas[label])
    row_str = f"{label:<14}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 5:
            row_str += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            d_q = cohens_d(delta[mask_q])
            row_str += f"{d_q:>+14.3f}"
            quintile_ds.append(float(d_q))
    d_all = cohens_d(delta)
    row_str += f"{d_all:>+14.3f}"
    print(row_str)
    hardness_data[label] = quintile_ds\
""")))

# ========== Cell 9: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 9: Plots -- 4-panel figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- Panel 1 (top-left): Bias Tuning Curve (d vs bias) ----
ax = axes[0, 0]

bias_vals_plot = BIAS_VALUES
d_vals = [analysis[f"bias_{bv:.1f}"]['cohens_d'] for bv in bias_vals_plot]

# Bootstrap 95% CI
np.random.seed(SEED)
ci_lo, ci_hi = [], []
for bv in bias_vals_plot:
    label = f"bias_{bv:.1f}"
    delta = bias_arrays[label]['delta']
    boot_ds = []
    for _ in range(2000):
        idx_boot = np.random.randint(0, len(delta), size=len(delta))
        boot_ds.append(cohens_d(delta[idx_boot]))
    boot_ds = np.array(boot_ds)
    ci_lo.append(np.percentile(boot_ds, 2.5))
    ci_hi.append(np.percentile(boot_ds, 97.5))
ci_lo = np.array(ci_lo)
ci_hi = np.array(ci_hi)

ax.errorbar(bias_vals_plot, d_vals,
            yerr=[np.array(d_vals) - ci_lo, ci_hi - np.array(d_vals)],
            marker='o', markersize=8, linewidth=2, capsize=5,
            color='#1f77b4', ecolor='#aec7e8')

# Reference lines
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axhline(y=EXP20_REF['1024_d'], color='#d62728', linestyle='--', linewidth=1.5,
           label=f"Exp 20 standard @ 1024tok (d={EXP20_REF['1024_d']:+.3f})")
ax.axhline(y=EXP20_REF['original_d'], color='#2ca02c', linestyle=':', linewidth=1.5,
           label=f"Exp 20 standard @ original (d={EXP20_REF['original_d']:+.3f})")

for i, bv in enumerate(bias_vals_plot):
    label = f"bias_{bv:.1f}"
    p_val = analysis[label]['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.annotate(f'd={d_vals[i]:+.3f} {sig}',
                (bv, d_vals[i]),
                textcoords='offset points', xytext=(0, 18),
                ha='center', fontsize=8)

ax.set_xlabel('Attention Bias Value')
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("Bias Tuning Curve: d vs Attention Bias")
ax.legend(fontsize=7)

# ---- Panel 2 (top-right): Mean NLL by condition (perplexity check) ----
ax = axes[0, 1]

cond_labels = ['bare'] + [f"bias_{bv:.1f}" for bv in BIAS_VALUES]
bare_mean_nll = float(np.mean(all_bare_nlls))
mean_nlls = [bare_mean_nll] + [analysis[f"bias_{bv:.1f}"]['mean_primed'] for bv in BIAS_VALUES]

colors_bar = ['#7f7f7f'] + ['#1f77b4' if bv < 10 else '#ff7f0e' for bv in BIAS_VALUES]
bars = ax.bar(range(len(cond_labels)), mean_nlls, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(cond_labels)))
ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Mean NLL')
ax.set_title('Perplexity Check: Mean NLL by Condition')

for i, v in enumerate(mean_nlls):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# ---- Panel 3 (bottom-left): Win Rate vs Bias ----
ax = axes[1, 0]

win_rates = [analysis[f"bias_{bv:.1f}"]['win_pct'] for bv in BIAS_VALUES]
ax.plot(BIAS_VALUES, win_rates, marker='s', markersize=8, linewidth=2, color='#2ca02c')
ax.axhline(y=50, color='black', linestyle='--', linewidth=0.8, label='Chance (50%)')

for i, bv in enumerate(BIAS_VALUES):
    ax.annotate(f'{win_rates[i]:.1f}%',
                (bv, win_rates[i]),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=9)

ax.set_xlabel('Attention Bias Value')
ax.set_ylabel('Win Rate (% where primed < bare)')
ax.set_title('Win Rate vs Bias Level')
ax.legend(fontsize=8)

# ---- Panel 4 (bottom-right): Hardness x Bias heatmap ----
ax = axes[1, 1]

heatmap_data = np.zeros((len(BIAS_VALUES), 5))
for i, bv in enumerate(BIAS_VALUES):
    label = f"bias_{bv:.1f}"
    for q in range(5):
        val = hardness_data[label][q]
        heatmap_data[i, q] = val if val is not None else np.nan

im = ax.imshow(heatmap_data, cmap='RdBu', aspect='auto',
               vmin=-0.5, vmax=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=8)
ax.set_yticks(range(len(BIAS_VALUES)))
ax.set_yticklabels([f"bias_{bv:.1f}" for bv in BIAS_VALUES])
ax.set_xlabel('Difficulty Quintile')
ax.set_ylabel('Bias Level')
ax.set_title("Hardness x Bias Interaction (Cohen's d)")

for i in range(len(BIAS_VALUES)):
    for j in range(5):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    fontsize=8, color='white' if abs(val) > 0.25 else 'black')

fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")

plt.suptitle('Exp 26: Attention Forcing for Long Documents (1024 tok)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 10: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 10: Save results.json + CSV
import csv

# --- CSV ---
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'bias_value', 'bias_label', 'actual_doc_len',
        'bare_nll', 'primed_nll', 'delta_nll'])
    writer.writeheader()
    for r in all_results:
        for row in r['rows']:
            writer.writerow(row)
print(f"CSV saved: {CSV_PATH}")

# --- results.json ---
final = {
    'experiment': 'exp26_attention_forcing',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'mistral',
        'seed': SEED,
        'dataset': 'MS MARCO v1.1 validation',
        'max_passage_words': MAX_PASSAGE_WORDS,
        'n_queries': N_QUERIES,
        'pad_target': PAD_TARGET,
        'bias_values': BIAS_VALUES,
        'prefix': STATIC_FACT,
        'prefix_token_len': PREFIX_TOKEN_LEN,
    },
    'analysis': analysis,
    'verdict': verdict,
    'hardness_data': hardness_data,
    'reference_values': {
        'exp20_mistral': EXP20_REF,
    },
    'per_query': all_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Final summary
print("\\n" + "=" * 70)
print("SUMMARY -- Exp 26: Attention Forcing for Long Documents")
print("=" * 70)
print(f"Model: Mistral 7B (4-bit, float16)")
print(f"Document length: {PAD_TARGET} tokens (padded)")
print(f"Prefix: static_fact_trunc ({PREFIX_TOKEN_LEN} tokens)")
print()
for bv in BIAS_VALUES:
    label = f"bias_{bv:.1f}"
    a = analysis[label]
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    marker = " <-- BEST" if label == best_label else ""
    print(f"  {label:<14} d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  "
          f"NLL={a['mean_primed']:.4f}  {sig}{marker}")
print(f"\\nExp 20 standard @ 1024 tok: d={EXP20_REF['1024_d']:+.3f}")
print(f"Exp 20 standard @ original:  d={EXP20_REF['original_d']:+.3f}")
print(f"\\nVERDICT: {verdict}")
print(f"\\nDone!")\
""")))

# ========== Cell 11: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 11: GPU cleanup
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

output_path = "/home/jupyter/research/directed_kvcache_v2/26_attention_forcing.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
