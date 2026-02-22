#!/usr/bin/env python3
"""Build script for 12_long_doc_priming_diagnostic.ipynb

Exp 12: Why Does Priming Fail on Long Documents? — Diagnostic Battery

Exp 11 showed that priming (static_fact_trunc d=+0.472 on MS MARCO) collapses
to d=-0.019 on Natural Questions long documents, with oracle HURTING (d=-0.188).

This experiment tests three competing hypotheses for the failure:

  A. SIGNAL DILUTION — 7 prefix tokens contaminate ~4000 doc values (0.2% dose
     vs 8% on MARCO). Fix: repeat prefix or amplify value delta.
  B. ATTENTION REDISTRIBUTION — On long docs, each value contributes a tiny
     fraction of output. Fix: amplify the contamination delta (alpha > 1).
  C. POSITIONAL INTERFERENCE — RoPE correction introduces phase noise on long
     sequences. Fix: suffix mode (no RoPE needed) or skip RoPE correction.

Design:
  - Dataset: Natural Questions (same bins as exp 11)
  - 9 conditions testing all three hypotheses
  - 400 samples (100 per length bin)
  - Per-bin analysis to see how each fix interacts with document length
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
# Exp 12: Why Does Priming Fail on Long Documents? — Diagnostic Battery

## Background

Exp 11 showed that static_fact_trunc (d=+0.472 on MS MARCO) collapses to d=-0.019
on Natural Questions (100-4000 word documents). Oracle priming actively HURTS
(d=-0.188, p<0.001). The failure is a step function at ~300 words, not gradual.

## Three Hypotheses

### A. Signal Dilution
On MARCO (~90 tokens), 7 prefix tokens contaminate ~83 doc values — an 8% "dose."
On NQ (~4000 tokens), the same 7 tokens contaminate ~4000 values — a 0.2% dose.
The contamination signal gets drowned by sheer volume.

**Prediction:** Proportionally increasing the prefix (via repetition) should restore
the effect. Value amplification (boosting the contamination delta) should also help.

### B. Attention Redistribution
On long docs, query attention is spread across thousands of positions. Even if values
are contaminated, each contributes a tiny fraction of the output.

**Prediction:** Amplifying the contamination delta (alpha > 1) should help, since
the direction is correct but the magnitude is too small.

### C. Positional Interference
RoPE correction shifts positions by ~7 on a 4000-token sequence. While the relative
error is tiny, the absolute correction interacts differently at position 3000+ than
at position 50+. The correction may introduce phase noise.

**Prediction:** Suffix mode (no RoPE correction needed) should work better than
truncated prefix on long docs. Removing RoPE correction should be informative.

## 9 Experimental Conditions

| # | Condition | Build | Tests |
|---|-----------|-------|-------|
| 1 | bare | [BOS][doc] | Baseline |
| 2 | prefix_1x | [BOS][sf\\n][doc] → trunc+RoPE | Confirms exp 11 failure |
| 3 | prefix_5x | [BOS][sf\\n ×5][doc] → trunc+RoPE | Hyp A: 5x dose |
| 4 | prefix_20x | [BOS][sf\\n ×20][doc] → trunc+RoPE | Hyp A: 20x dose |
| 5 | amplify_2x | bare keys + 2x boosted values | Hyp A+B: amplify delta |
| 6 | amplify_5x | bare keys + 5x boosted values | Hyp A+B: stronger boost |
| 7 | layers_0_15 | primed values only at layers 0-15 | Signal localization |
| 8 | suffix | [BOS][doc][sep][sf] (full context) | Hyp C: no RoPE needed |
| 9 | no_rope | [BOS][sf\\n][doc] → trunc, NO RoPE | Hyp C: direct test |

Where sf = "What are the key facts?" (best static surrogate from exp 07).

## Key Analysis

For each condition, compute Cohen's d vs bare across 4 length bins.
The critical question: **which condition recovers the effect on long docs?**""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import math
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp12")
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

# ========== Cell 3: Config, constants, helpers ==========
cells.append(make_cell("code", s("""\
# Cell 3: Config, constants, and library imports
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
    replace_values_at_layers,
)
from lib.analysis import cohens_d
from lib.surrogate import STATIC_SURROGATE_QUERIES
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

N_EVAL = 400  # total target (100 per bin)
N_CONDITIONS = 9
N_COMPARISONS = 8  # each non-bare condition vs bare
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
CHECKPOINT_EVERY = 25

STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Length bins (word count)
LENGTH_BINS = [
    ('short',     100,  300),
    ('medium',    300,  800),
    ('long',      800,  2000),
    ('very_long', 2000, 4000),
]
SAMPLES_PER_BIN = 100
MAX_DOC_WORDS = 4000

CONDITION_NAMES = [
    'bare',
    'prefix_1x',
    'prefix_5x',
    'prefix_20x',
    'amplify_2x',
    'amplify_5x',
    'layers_0_15',
    'suffix',
    'no_rope',
]

# Suffix separator
SUFFIX_SEPARATOR = "\\n\\nRelated question: "

# Repetition counts
REP_COUNTS = {'prefix_5x': 5, 'prefix_20x': 20}

# Amplification alphas (extrapolation beyond 1.0)
AMP_ALPHAS = {'amplify_2x': 2.0, 'amplify_5x': 5.0}

print("Config ready")
print(f"  N_EVAL: {N_EVAL}")
print(f"  SAMPLES_PER_BIN: {SAMPLES_PER_BIN}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: {len(CONDITION_NAMES)}")
print(f"  static_fact: '{STATIC_FACT}'")
print(f"  length_bins: {LENGTH_BINS}")
print(f"  max_doc_words: {MAX_DOC_WORDS}")\
""")))

# ========== Cell 4: Load Natural Questions ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load Natural Questions — stratified by document length
from datasets import load_dataset

print("=" * 70)
print("LOADING NATURAL QUESTIONS (validation split)")
print("=" * 70)

# Check for cached samples (can reuse exp 11 samples with subsampling)
SAMPLES_CACHE_PATH = RESULTS_DIR / "nq_samples.json"
EXP11_SAMPLES_PATH = Path("results/exp11/nq_samples.json")

if SAMPLES_CACHE_PATH.exists():
    with open(SAMPLES_CACHE_PATH, 'r') as f:
        cached = json.load(f)
    samples = cached['samples']
    print(f"Loaded {len(samples)} cached NQ samples from {SAMPLES_CACHE_PATH}")
elif EXP11_SAMPLES_PATH.exists():
    # Reuse exp 11 samples — subsample to 100 per bin
    print("Reusing exp 11 NQ samples (subsampling to 100 per bin)...")
    with open(EXP11_SAMPLES_PATH, 'r') as f:
        exp11_data = json.load(f)
    all_exp11 = exp11_data['samples']

    samples = []
    for bin_name, _, _ in LENGTH_BINS:
        bin_s = [s for s in all_exp11 if s['length_bin'] == bin_name]
        samples.extend(bin_s[:SAMPLES_PER_BIN])
        print(f"  {bin_name}: {min(len(bin_s), SAMPLES_PER_BIN)} samples (from {len(bin_s)} available)")

    with open(SAMPLES_CACHE_PATH, 'w') as f:
        json.dump({'samples': samples, 'source': 'exp11_subsampled'}, f)
    print(f"Cached {len(samples)} samples to {SAMPLES_CACHE_PATH}")
else:
    print("Loading NQ dataset from scratch (streaming mode)...")
    nq = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True,
    )

    bin_samples = {name: [] for name, _, _ in LENGTH_BINS}
    n_processed = 0

    for example in tqdm(nq, desc="Processing NQ"):
        n_processed += 1

        doc_tokens = example['document']['tokens']
        if isinstance(doc_tokens, dict):
            token_strs = doc_tokens['token']
            is_html_flags = doc_tokens['is_html']
            clean_tokens = [t for t, h in zip(token_strs, is_html_flags) if not h]
        else:
            clean_tokens = [t['token'] for t in doc_tokens if not t['is_html']]

        doc_text = ' '.join(clean_tokens)
        word_count = len(doc_text.split())

        if word_count < LENGTH_BINS[0][1]:
            continue
        if word_count > MAX_DOC_WORDS:
            words = doc_text.split()
            doc_text = ' '.join(words[:MAX_DOC_WORDS])
            word_count = MAX_DOC_WORDS

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

        for bin_name, bin_min, bin_max in LENGTH_BINS:
            if bin_min <= word_count < bin_max:
                if len(bin_samples[bin_name]) < SAMPLES_PER_BIN:
                    bin_samples[bin_name].append({
                        'passage': doc_text,
                        'query': query,
                        'answer': answer_text,
                        'word_count': word_count,
                        'length_bin': bin_name,
                    })
                break

        all_full = all(len(bin_samples[name]) >= SAMPLES_PER_BIN for name, _, _ in LENGTH_BINS)
        if all_full:
            print(f"All bins full after processing {n_processed} examples.")
            break

    samples = []
    for bin_name, _, _ in LENGTH_BINS:
        bin_s = bin_samples[bin_name]
        np.random.seed(SEED)
        np.random.shuffle(bin_s)
        samples.extend(bin_s)
        print(f"  {bin_name}: {len(bin_s)} samples")

    with open(SAMPLES_CACHE_PATH, 'w') as f:
        json.dump({'samples': samples, 'n_processed': n_processed}, f)
    print(f"Cached to {SAMPLES_CACHE_PATH}")

N = len(samples)

print(f"\\n{'='*70}")
print(f"SAMPLE SUMMARY")
print(f"{'='*70}")
for bin_name, bin_min, bin_max in LENGTH_BINS:
    bin_s = [s for s in samples if s['length_bin'] == bin_name]
    if bin_s:
        wcs = [s['word_count'] for s in bin_s]
        print(f"  {bin_name} ({bin_min}-{bin_max}w): n={len(bin_s)}, "
              f"mean={np.mean(wcs):.0f}w, range=[{min(wcs)}, {max(wcs)}]")\
""")))

# ========== Cell 5: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 5: Explain experimental conditions with concrete examples

print("=" * 70)
print("EXPERIMENTAL CONDITIONS — DIAGNOSTIC BATTERY")
print("=" * 70)

# Pre-tokenize the static fact prefix to show token counts
sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_ids = tokenizer(sf_str, add_special_tokens=False)['input_ids']
sf_tok_len = len(sf_ids)

conditions_explained = [
    ("1. bare",
     "[BOS][doc]",
     "Baseline — no prefix. Document encoded in isolation.",
     "—"),
    ("2. prefix_1x",
     f"[BOS][sf\\\\n][doc] → trunc + RoPE",
     f"Standard 1x static_fact prefix ({sf_tok_len} tokens). Should replicate exp 11 failure.",
     "Hypothesis: NONE (baseline failure)"),
    ("3. prefix_5x",
     f"[BOS][sf\\\\n ×5][doc] → trunc + RoPE  ({5*sf_tok_len} prefix tokens)",
     "5x repeated prefix with block-diagonal attention mask (reps can't see each other).",
     "Hypothesis A: If d(5x) > d(1x), dilution is the issue"),
    ("4. prefix_20x",
     f"[BOS][sf\\\\n ×20][doc] → trunc + RoPE  ({20*sf_tok_len} prefix tokens)",
     "20x repeated prefix (strongest dilution test). ~2.7% dose on 4000w doc.",
     "Hypothesis A: If d(20x) >> d(5x), more dose = more signal"),
    ("5. amplify_2x",
     "bare keys + v_bare + 2.0 * (v_primed - v_bare)",
     "Value amplification: double the contamination delta. Keys from bare (correct positions).",
     "Hypothesis A+B: Tests if contamination direction is correct but too weak"),
    ("6. amplify_5x",
     "bare keys + v_bare + 5.0 * (v_primed - v_bare)",
     "5x value amplification. If 2x helps but 5x hurts, contamination is partially noise.",
     "Hypothesis A+B: Finds the signal-vs-noise boundary"),
    ("7. layers_0_15",
     "bare everywhere + primed values at layers 0-15 only",
     "Exp 09 found signal in layers 0-15 on MARCO. Late layers may add noise on long docs.",
     "Signal localization: If d(L0-15) > d(1x), late layers hurt"),
    ("8. suffix",
     "[BOS][doc][sep][sf] (full context, no truncation)",
     f"Suffix mode: doc can't attend backward to suffix. No RoPE correction needed. Sep='{SUFFIX_SEPARATOR.strip()}'",
     "Hypothesis C: If d(suffix) > d(1x), RoPE correction is the problem"),
    ("9. no_rope",
     "[BOS][sf\\\\n][doc] → truncate only, NO RoPE correction",
     "Same as prefix_1x but without RoPE position correction. Keys keep original positions.",
     "Hypothesis C: Direct test — is RoPE correction helping or hurting on long docs?"),
]

for name, pattern, detail, hypothesis in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")
    print(f"  Tests: {hypothesis}")

# Show dose ratios
print(f"\\n{'='*70}")
print("PREFIX-TO-DOCUMENT DOSE RATIOS")
print(f"{'='*70}")
print(f"  MS MARCO (exp 07):  {sf_tok_len} prefix / ~90 doc tokens = {sf_tok_len/90*100:.1f}%")
for bin_name, bin_min, bin_max in LENGTH_BINS:
    mid_tokens = int((bin_min + bin_max) / 2 * 1.5)
    for rep_name, n_reps in [('1x', 1), ('5x', 5), ('20x', 20)]:
        dose = n_reps * sf_tok_len / mid_tokens * 100
        print(f"  NQ {bin_name} ({rep_name}): {n_reps*sf_tok_len} prefix / ~{mid_tokens} doc tokens = {dose:.1f}%")\
""")))

# ========== Cell 6: Helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 6: Helper functions for repeated prefix and value amplification


def build_repeated_prefix_mask(prefix_len_single, n_reps, doc_len, dtype, device):
    \"\"\"Build block-diagonal attention mask for [BOS][rep1][rep2]...[repN][doc].

    Pattern:
    - BOS (pos 0): visible to everything, sees only itself
    - Each rep block: causal within block, can see BOS, CANNOT see other reps
    - Doc tokens: causal, can see BOS and all reps (standard)

    Returns: (1, 1, total_len, total_len) additive mask (0=attend, -inf=block)
    \"\"\"
    total_prefix = n_reps * prefix_len_single
    total_len = 1 + total_prefix + doc_len  # BOS + reps + doc

    # Start with standard causal mask (lower triangle = 0, upper = -inf)
    mask = torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device)
    mask = torch.tril(mask, diagonal=0)  # This gives -inf everywhere; wrong approach
    # Correct: start fresh
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)
    # Upper triangle = -inf (no future attention)
    for i in range(total_len):
        for j in range(i + 1, total_len):
            mask[i, j] = float('-inf')

    # Block cross-repetition attention (rep i can't see rep j for j != i)
    for rep_i in range(n_reps):
        start_i = 1 + rep_i * prefix_len_single
        end_i = start_i + prefix_len_single
        for rep_j in range(n_reps):
            if rep_j == rep_i:
                continue
            start_j = 1 + rep_j * prefix_len_single
            end_j = start_j + prefix_len_single
            # Block rep_i rows from seeing rep_j columns
            # Only block where rep_i could causally see rep_j (j < i)
            if rep_j < rep_i:
                mask[start_i:end_i, start_j:end_j] = float('-inf')

    return mask.unsqueeze(0).unsqueeze(0)


def build_repeated_prefix_mask_fast(prefix_len_single, n_reps, doc_len, dtype, device):
    \"\"\"Vectorized version of build_repeated_prefix_mask for large sequences.

    Same semantics as build_repeated_prefix_mask but uses tensor ops
    instead of Python loops for the causal structure.
    \"\"\"
    total_prefix = n_reps * prefix_len_single
    total_len = 1 + total_prefix + doc_len

    # Start with causal mask using triu
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)
    mask += torch.triu(
        torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

    # Block cross-rep attention: rep i rows cannot see rep j columns (j < i)
    for rep_i in range(1, n_reps):
        start_i = 1 + rep_i * prefix_len_single
        end_i = start_i + prefix_len_single
        for rep_j in range(rep_i):
            start_j = 1 + rep_j * prefix_len_single
            end_j = start_j + prefix_len_single
            mask[start_i:end_i, start_j:end_j] = float('-inf')

    return mask.unsqueeze(0).unsqueeze(0)


def amplify_values(bare_cache, primed_cache, alpha):
    \"\"\"Create cache with amplified value contamination.

    v_amplified = v_bare + alpha * (v_primed - v_bare)
               = (1 - alpha) * v_bare + alpha * v_primed

    When alpha > 1.0, this EXTRAPOLATES beyond the primed values,
    amplifying the contamination signal.

    Keys are taken from bare_cache (correct position encoding).

    Args:
        bare_cache: Cache with uncontaminated values and correct keys
        primed_cache: Cache with contaminated values (from truncated prefix)
        alpha: Amplification factor (1.0 = primed, 2.0 = 2x amplification)

    Returns:
        New DynamicCache with bare keys and amplified values
    \"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    primed_cache = _ensure_dynamic_cache(primed_cache)

    n_layers = len(bare_cache)
    new_cache = DynamicCache()

    for li in range(n_layers):
        k = _get_cache_keys(bare_cache, li).clone()
        v_bare = _get_cache_values(bare_cache, li)
        v_primed = _get_cache_values(primed_cache, li)
        v_amp = v_bare + alpha * (v_primed - v_bare)
        new_cache.update(k, v_amp.clone(), li)

    return new_cache


# Quick validation of the mask
print("Validating block-diagonal mask...")
test_mask = build_repeated_prefix_mask_fast(3, 2, 2, torch.float32, 'cpu')
test_mask_2d = test_mask.squeeze()
# Shape should be (1+6+2, 1+6+2) = (9, 9)
# BOS=0, R1=[1,2,3], R2=[4,5,6], Doc=[7,8]
assert test_mask_2d.shape == (9, 9), f"Expected (9,9), got {test_mask_2d.shape}"
# R2 (row 4) should NOT see R1 (cols 1,2,3)
assert test_mask_2d[4, 1] == float('-inf'), "R2 should not see R1"
assert test_mask_2d[4, 2] == float('-inf'), "R2 should not see R1"
# R2 (row 4) SHOULD see BOS (col 0) and itself (col 4)
assert test_mask_2d[4, 0] == 0.0, "R2 should see BOS"
assert test_mask_2d[4, 4] == 0.0, "R2 should see itself"
# Doc (row 7) should see everything before it
assert test_mask_2d[7, 0] == 0.0, "Doc should see BOS"
assert test_mask_2d[7, 1] == 0.0, "Doc should see R1"
assert test_mask_2d[7, 4] == 0.0, "Doc should see R2"
assert test_mask_2d[7, 7] == 0.0, "Doc should see itself"
print("  Block-diagonal mask validated OK")

print("\\nHelper functions ready.")\
""")))

# ========== Cell 7: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main eval loop — 9 conditions x N samples
import gc

print("=" * 70)
print(f"PHASE: MAIN EVALUATION ({N_CONDITIONS} conditions x {N} samples)")
print("=" * 70)

# Pre-tokenize fixed strings (reused for every sample)
sf_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_prefix_enc = tokenizer(sf_prefix_str, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
sf_prefix_ids = sf_prefix_enc['input_ids'].to(config.device)
sf_prefix_len = sf_prefix_ids.shape[1]  # tokens per single prefix rep

suffix_sep_enc = tokenizer(SUFFIX_SEPARATOR, return_tensors="pt",
                            add_special_tokens=False, padding=False, truncation=False)
suffix_sep_ids = suffix_sep_enc['input_ids'].to(config.device)

suffix_text_enc = tokenizer(STATIC_FACT, return_tensors="pt",
                             add_special_tokens=False, padding=False, truncation=False)
suffix_text_ids = suffix_text_enc['input_ids'].to(config.device)

print(f"Static fact prefix: '{STATIC_FACT}' ({sf_prefix_len} tokens per rep)")
print(f"5x prefix: {5*sf_prefix_len} tokens, 20x prefix: {20*sf_prefix_len} tokens")
print(f"Suffix separator: {suffix_sep_ids.shape[1]} tokens")
print(f"Suffix text: {suffix_text_ids.shape[1]} tokens")

# Checkpoint resume
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
n_layers = model.config.num_hidden_layers

t_start = time.time()

for idx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Evaluating"):
    sample = samples[idx]
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    word_count = sample['word_count']
    length_bin = sample['length_bin']

    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    # --- Matched tokenization (from exp 11) ---
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
    context_len = 1 + doc_len  # BOS + doc

    # ===================================================================
    # PHASE 1: BUILD bare + 1x PREFIX (shared forward passes)
    # ===================================================================
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = bare_out.past_key_values
    del bare_out

    primed_1x_ids = torch.cat([bos_id, sf_prefix_ids, doc_ids], dim=1)
    with torch.no_grad():
        out_1x = model(input_ids=primed_1x_ids,
                       attention_mask=torch.ones_like(primed_1x_ids),
                       use_cache=True, return_dict=True)
    trunc_1x = extract_and_truncate_cache_with_bos(out_1x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_1x, sf_prefix_len, model)
    trunc_no_rope = extract_and_truncate_cache_with_bos(out_1x.past_key_values, doc_len)
    del out_1x

    # ===================================================================
    # PHASE 2: DERIVED CACHES (need bare + 1x, no forward passes)
    # ===================================================================
    amp_2x_cache = amplify_values(bare_cache, trunc_1x, 2.0)
    amp_5x_cache = amplify_values(bare_cache, trunc_1x, 5.0)
    layers_cache = replace_values_at_layers(bare_cache, trunc_1x, list(range(16)))

    # ===================================================================
    # PHASE 3: SCORE + FREE all caches built so far
    # No deepcopy needed — each cache is scored exactly once then freed.
    # ===================================================================
    nll_bare = score_answer_with_cache(
        bare_cache, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del bare_cache

    nll_1x = score_answer_with_cache(
        trunc_1x, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_1x

    nll_no_rope = score_answer_with_cache(
        trunc_no_rope, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_no_rope

    nll_amp2 = score_answer_with_cache(
        amp_2x_cache, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del amp_2x_cache

    nll_amp5 = score_answer_with_cache(
        amp_5x_cache, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del amp_5x_cache

    nll_layers = score_answer_with_cache(
        layers_cache, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del layers_cache

    # Free memory before heavy build passes
    del bare_ids, primed_1x_ids
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # PHASE 4: BUILD + SCORE 5x PREFIX (isolated to limit peak memory)
    # ===================================================================
    rep5_ids = sf_prefix_ids.repeat(1, 5)
    rep5_full_ids = torch.cat([bos_id, rep5_ids, doc_ids], dim=1)
    mask_5x = build_repeated_prefix_mask_fast(
        sf_prefix_len, 5, doc_len, model.dtype, model.device)
    with torch.no_grad():
        out_5x = model(input_ids=rep5_full_ids, attention_mask=mask_5x,
                       use_cache=True, return_dict=True)
    trunc_5x = extract_and_truncate_cache_with_bos(out_5x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_5x, 5 * sf_prefix_len, model)
    del out_5x, mask_5x, rep5_ids, rep5_full_ids

    nll_5x = score_answer_with_cache(
        trunc_5x, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_5x
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # PHASE 5: BUILD + SCORE 20x PREFIX (heaviest — isolated)
    # ===================================================================
    rep20_ids = sf_prefix_ids.repeat(1, 20)
    rep20_full_ids = torch.cat([bos_id, rep20_ids, doc_ids], dim=1)
    mask_20x = build_repeated_prefix_mask_fast(
        sf_prefix_len, 20, doc_len, model.dtype, model.device)
    with torch.no_grad():
        out_20x = model(input_ids=rep20_full_ids, attention_mask=mask_20x,
                        use_cache=True, return_dict=True)
    trunc_20x = extract_and_truncate_cache_with_bos(out_20x.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_20x, 20 * sf_prefix_len, model)
    del out_20x, mask_20x, rep20_ids, rep20_full_ids

    nll_20x = score_answer_with_cache(
        trunc_20x, context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_20x
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # PHASE 6: BUILD + SCORE SUFFIX
    # ===================================================================
    suffix_full_ids = torch.cat([bos_id, doc_ids, suffix_sep_ids, suffix_text_ids], dim=1)
    with torch.no_grad():
        out_suffix = model(input_ids=suffix_full_ids,
                           attention_mask=torch.ones_like(suffix_full_ids),
                           use_cache=True, return_dict=True)
    suffix_cache = out_suffix.past_key_values
    suffix_context_len = suffix_full_ids.shape[1]
    del out_suffix, suffix_full_ids

    nll_suffix = score_answer_with_cache(
        suffix_cache, suffix_context_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suffix_cache
    gc.collect()
    torch.cuda.empty_cache()

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len_tokens': doc_len,
        'word_count': word_count,
        'length_bin': length_bin,
        'bare': nll_bare,
        'prefix_1x': nll_1x,
        'prefix_5x': nll_5x,
        'prefix_20x': nll_20x,
        'amplify_2x': nll_amp2,
        'amplify_5x': nll_amp5,
        'layers_0_15': nll_layers,
        'suffix': nll_suffix,
        'no_rope': nll_no_rope,
    }
    results.append(result)

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

# ========== Cell 8: Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 8: Analysis — overall + per length bin + hypothesis testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — LONG-DOCUMENT PRIMING DIAGNOSTIC")
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

length_bins_arr = np.array([r['length_bin'] for r in results])[valid]
word_counts_arr = np.array([r['word_count'] for r in results])[valid]
doc_lens_arr = np.array([r['doc_len_tokens'] for r in results])[valid]

# ===== OVERALL NLL SUMMARY =====
print(f"\\n{'Condition':<20} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10} {'Win%':>7}")
print("-" * 62)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        print(f"{cname:<20} {mean_nll:>10.4f} {std_nll:>10.4f} {'--':>10} {'--':>7}")
    else:
        delta = c['bare'] - c[cname]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        _, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
        print(f"{cname:<20} {mean_nll:>10.4f} {std_nll:>10.4f} {d:>+10.3f} {win:>5.1f}% {sig}")

# ===== 8 PRIMARY COMPARISONS =====
print(f"\\n{'='*90}")
print(f"8 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*90}")

primary_conditions = [cn for cn in CONDITION_NAMES if cn != 'bare']
comparisons = []
for cname in primary_conditions:
    delta = c['bare'] - c[cname]
    comparisons.append((f"{cname} vs bare", delta, cname))

print(f"\\n{'Comparison':<25} {'Mean delta':>10} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 80)

comparison_results = {}
for name, delta, cname in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<25} {np.mean(delta):>10.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
    }

# ===== HYPOTHESIS-SPECIFIC COMPARISONS =====
print(f"\\n{'='*90}")
print("HYPOTHESIS TESTS — Between-condition comparisons")
print(f"{'='*90}")

hyp_comparisons = [
    ("Hyp A: 5x vs 1x (repetition helps?)",
     c['prefix_1x'] - c['prefix_5x']),
    ("Hyp A: 20x vs 1x (strong repetition?)",
     c['prefix_1x'] - c['prefix_20x']),
    ("Hyp A: 20x vs 5x (more = better?)",
     c['prefix_5x'] - c['prefix_20x']),
    ("Hyp A+B: amplify_2x vs 1x",
     c['prefix_1x'] - c['amplify_2x']),
    ("Hyp A+B: amplify_5x vs 1x",
     c['prefix_1x'] - c['amplify_5x']),
    ("Hyp C: suffix vs 1x (RoPE issue?)",
     c['prefix_1x'] - c['suffix']),
    ("Hyp C: no_rope vs 1x (correction helps?)",
     c['prefix_1x'] - c['no_rope']),
    ("Signal: layers_0_15 vs 1x",
     c['prefix_1x'] - c['layers_0_15']),
]

print(f"\\n{'Comparison':<40} {'Mean delta':>10} {'d':>8} {'Win%':>7} {'p':>12}")
print("-" * 82)
hyp_results = {}
for name, delta in hyp_comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{name:<40} {np.mean(delta):>10.4f} {d:>8.3f} {win:>6.1f}% {p_val:>11.2e} {sig}")
    hyp_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        'p_value': float(p_val),
    }

# ===== PER LENGTH BIN ANALYSIS =====
print(f"\\n{'='*90}")
print("PER LENGTH BIN — Does any condition recover the effect on long docs?")
print(f"{'='*90}")

bin_names_ordered = [name for name, _, _ in LENGTH_BINS]
per_bin_results = {}

for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    print(f"\\n  {cname}:")
    bin_ds = []
    bin_wins = []
    bin_ns = []
    for bin_name in bin_names_ordered:
        mask = length_bins_arr == bin_name
        n_bin = int(np.sum(mask))
        if n_bin < 10:
            print(f"    {bin_name}: n={n_bin} (too few)")
            bin_ds.append(None)
            bin_wins.append(None)
            bin_ns.append(n_bin)
            continue
        delta = c['bare'][mask] - c[cname][mask]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        _, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
        print(f"    {bin_name}: n={n_bin}, d={d:+.3f}, win={win:.1f}%, p={p_val:.2e} {sig}")
        bin_ds.append(float(d))
        bin_wins.append(float(win))
        bin_ns.append(n_bin)

    per_bin_results[cname] = {
        'bin_names': bin_names_ordered,
        'bin_ds': bin_ds,
        'bin_wins': bin_wins,
        'bin_ns': bin_ns,
    }

# ===== DOSE-RESPONSE: 1x vs 5x vs 20x across bins =====
print(f"\\n{'='*90}")
print("DOSE-RESPONSE: Does increasing prefix repetitions help progressively?")
print(f"{'='*90}")

dose_response = {}
for bin_name in bin_names_ordered:
    mask = length_bins_arr == bin_name
    n_bin = int(np.sum(mask))
    if n_bin < 10:
        continue
    d_1x = cohens_d(c['bare'][mask] - c['prefix_1x'][mask])
    d_5x = cohens_d(c['bare'][mask] - c['prefix_5x'][mask])
    d_20x = cohens_d(c['bare'][mask] - c['prefix_20x'][mask])
    print(f"  {bin_name} (n={n_bin}): 1x d={d_1x:+.3f}, 5x d={d_5x:+.3f}, 20x d={d_20x:+.3f}")
    trend = "INCREASING" if d_20x > d_5x > d_1x else "NON-MONOTONIC" if d_20x > d_1x else "DECREASING"
    print(f"    Trend: {trend}")
    dose_response[bin_name] = {'d_1x': d_1x, 'd_5x': d_5x, 'd_20x': d_20x, 'trend': trend}

# ===== AMPLIFICATION RESPONSE =====
print(f"\\n{'='*90}")
print("AMPLIFICATION RESPONSE: Does boosting the value delta help?")
print(f"{'='*90}")

amp_response = {}
for bin_name in bin_names_ordered:
    mask = length_bins_arr == bin_name
    n_bin = int(np.sum(mask))
    if n_bin < 10:
        continue
    d_1x = cohens_d(c['bare'][mask] - c['prefix_1x'][mask])
    d_a2 = cohens_d(c['bare'][mask] - c['amplify_2x'][mask])
    d_a5 = cohens_d(c['bare'][mask] - c['amplify_5x'][mask])
    print(f"  {bin_name} (n={n_bin}): 1x d={d_1x:+.3f}, amp2x d={d_a2:+.3f}, amp5x d={d_a5:+.3f}")
    amp_response[bin_name] = {'d_1x': d_1x, 'd_amp2x': d_a2, 'd_amp5x': d_a5}

# ===== LENGTH INTERACTION CORRELATIONS =====
print(f"\\n{'='*90}")
print("LENGTH INTERACTION — Correlation between doc length and benefit")
print(f"{'='*90}")

from scipy.stats import spearmanr

interaction_results = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    r_spear, p_spear = spearmanr(word_counts_arr, delta)
    print(f"  {cname}: Spearman r={r_spear:+.3f} (p={p_spear:.3f})")
    interaction_results[cname] = {
        'spearman_r': float(r_spear), 'spearman_p': float(p_spear),
    }

# ===== HYPOTHESIS VERDICT =====
print(f"\\n{'='*90}")
print("HYPOTHESIS VERDICT")
print(f"{'='*90}")

# Overall d for key conditions
d_1x = cohens_d(c['bare'] - c['prefix_1x'])
d_5x = cohens_d(c['bare'] - c['prefix_5x'])
d_20x = cohens_d(c['bare'] - c['prefix_20x'])
d_amp2 = cohens_d(c['bare'] - c['amplify_2x'])
d_amp5 = cohens_d(c['bare'] - c['amplify_5x'])
d_suffix = cohens_d(c['bare'] - c['suffix'])
d_no_rope = cohens_d(c['bare'] - c['no_rope'])
d_layers = cohens_d(c['bare'] - c['layers_0_15'])

print(f"\\n  Hypothesis A (Signal Dilution):")
print(f"    1x d={d_1x:+.3f} → 5x d={d_5x:+.3f} → 20x d={d_20x:+.3f}")
if d_20x > d_1x + 0.05:
    print(f"    SUPPORTED: More repetition helps (+{d_20x - d_1x:.3f})")
elif d_20x < d_1x - 0.05:
    print(f"    REFUTED: More repetition HURTS ({d_20x - d_1x:.3f})")
else:
    print(f"    INCONCLUSIVE: Repetition has negligible effect")

print(f"\\n  Hypothesis A+B (Amplification):")
print(f"    1x d={d_1x:+.3f} → amp2x d={d_amp2:+.3f} → amp5x d={d_amp5:+.3f}")
if d_amp2 > d_1x + 0.05:
    print(f"    SUPPORTED: Amplification helps (direction is correct, magnitude too small)")
elif d_amp5 < d_1x - 0.1:
    print(f"    REFUTED: Amplification hurts (contamination is noise, not signal)")
else:
    print(f"    INCONCLUSIVE")

print(f"\\n  Hypothesis C (RoPE Interference):")
print(f"    1x d={d_1x:+.3f} vs suffix d={d_suffix:+.3f} vs no_rope d={d_no_rope:+.3f}")
if d_suffix > d_1x + 0.05 or d_no_rope > d_1x + 0.05:
    print(f"    SUPPORTED: RoPE correction is part of the problem")
else:
    print(f"    REFUTED: RoPE is not the issue")

print(f"\\n  Signal Localization:")
print(f"    1x d={d_1x:+.3f} vs layers_0_15 d={d_layers:+.3f}")
if d_layers > d_1x + 0.05:
    print(f"    SUPPORTED: Late layers add noise on long docs")
else:
    print(f"    Layers 16+ do not hurt on long docs")\
""")))

# ========== Cell 9: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 9: Plots (2x2 grid + dose-response subplot)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

colors = {
    'prefix_1x': '#d62728',
    'prefix_5x': '#ff7f0e',
    'prefix_20x': '#e377c2',
    'amplify_2x': '#2ca02c',
    'amplify_5x': '#17becf',
    'layers_0_15': '#9467bd',
    'suffix': '#1f77b4',
    'no_rope': '#7f7f7f',
}

# --- Plot 1: Per-bin Cohen's d for ALL conditions ---
ax = axes[0, 0]
x = np.arange(len(bin_names_ordered))
width = 0.09
conds_to_plot = [cn for cn in CONDITION_NAMES if cn != 'bare']
for i, cname in enumerate(conds_to_plot):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - len(conds_to_plot)/2 + 0.5) * width
    ax.bar(x + offset, ds_clean, width, label=cname,
           color=colors[cname], edgecolor='black', linewidth=0.3, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_xlabel("Document Length Bin")
ax.set_title("All Conditions by Length Bin")
ax.legend(fontsize=6, ncol=2)

# --- Plot 2: Dose-response (1x, 5x, 20x) per bin ---
ax = axes[0, 1]
x = np.arange(len(bin_names_ordered))
width = 0.25
for i, (cname, label) in enumerate([('prefix_1x', '1x'), ('prefix_5x', '5x'), ('prefix_20x', '20x')]):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, ds_clean, width, label=label,
                  color=colors[cname], edgecolor='black', linewidth=0.5, alpha=0.85)
    for j, d_val in enumerate(ds):
        if d_val is not None:
            ax.text(x[j] + offset, d_val + 0.005, f"{d_val:+.2f}", ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Dose-Response: Prefix Repetitions")
ax.legend()

# --- Plot 3: Amplification response per bin ---
ax = axes[0, 2]
for i, (cname, label) in enumerate([('prefix_1x', '1x (baseline)'), ('amplify_2x', 'amplify 2x'), ('amplify_5x', 'amplify 5x')]):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, ds_clean, width, label=label,
                  color=colors[cname], edgecolor='black', linewidth=0.5, alpha=0.85)
    for j, d_val in enumerate(ds):
        if d_val is not None:
            ax.text(x[j] + offset, d_val + 0.005, f"{d_val:+.2f}", ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Amplification: Boosting Value Delta")
ax.legend()

# --- Plot 4: Overall bar chart (all conditions, sorted) ---
ax = axes[1, 0]
conds_sorted = sorted(
    [(cn, cohens_d(c['bare'] - c[cn])) for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda x: x[1], reverse=True
)
names_sorted = [x[0] for x in conds_sorted]
ds_sorted = [x[1] for x in conds_sorted]
bar_colors = [colors.get(cn, 'gray') for cn in names_sorted]
bars = ax.barh(range(len(names_sorted)), ds_sorted, color=bar_colors,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=9)
for i, (name, d_val) in enumerate(conds_sorted):
    ax.text(d_val + 0.005, i, f"d={d_val:+.3f}", va='center', fontsize=8)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title("Overall Effect (All Bins Combined)")
ax.invert_yaxis()
# Reference line from MS MARCO
ax.axvline(x=0.472, color='red', linestyle=':', alpha=0.5, label='Exp07 MARCO static_fact')
ax.legend(fontsize=7)

# --- Plot 5: RoPE hypothesis (1x vs suffix vs no_rope) per bin ---
ax = axes[1, 1]
for i, (cname, label) in enumerate([('prefix_1x', '1x (with RoPE)'), ('suffix', 'suffix (no RoPE)'), ('no_rope', 'no_rope (skip correction)')]):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, ds_clean, width, label=label,
                  color=colors[cname], edgecolor='black', linewidth=0.5, alpha=0.85)
    for j, d_val in enumerate(ds):
        if d_val is not None:
            ax.text(x[j] + offset, d_val + 0.005, f"{d_val:+.2f}", ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Hypothesis C: RoPE Interference")
ax.legend()

# --- Plot 6: Scatter — word count vs benefit for key conditions ---
ax = axes[1, 2]
for cname in ['prefix_1x', 'prefix_20x', 'amplify_2x']:
    delta = c['bare'] - c[cname]
    ax.scatter(word_counts_arr, delta, alpha=0.15, s=8, color=colors[cname], label=cname)
    # Binned trend
    n_trend_bins = 15
    edges = np.linspace(word_counts_arr.min(), word_counts_arr.max(), n_trend_bins + 1)
    for k in range(n_trend_bins):
        mask_k = (word_counts_arr >= edges[k]) & (word_counts_arr < edges[k+1])
        if np.sum(mask_k) > 5:
            ax.scatter((edges[k] + edges[k+1])/2, np.mean(delta[mask_k]),
                      s=40, color=colors[cname], edgecolor='black', linewidth=0.5, zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Document Word Count")
ax.set_ylabel("NLL Reduction (bare - primed)")
ax.set_title("NLL Reduction vs Document Length")
ax.legend(fontsize=8)

plt.suptitle('Exp 12: Why Does Priming Fail on Long Documents?', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 10: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 10: Save comprehensive results JSON

nll_summary = {}
for cname in CONDITION_NAMES:
    nll_summary[cname] = {
        'mean': float(np.mean(c[cname])),
        'std': float(np.std(c[cname])),
        'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cname])) if cname != 'bare' else 0.0,
    }

final = {
    'experiment': 'exp12_long_doc_priming_diagnostic',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_eval': N,
        'n_valid': n_valid,
        'n_excluded': n_excluded,
        'n_conditions': N_CONDITIONS,
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
        'dataset': 'google-research-datasets/natural_questions',
        'dataset_split': 'validation',
        'length_bins': LENGTH_BINS,
        'max_doc_words': MAX_DOC_WORDS,
        'static_fact': STATIC_FACT,
        'sf_prefix_tokens': sf_prefix_len,
        'rep_counts': REP_COUNTS,
        'amp_alphas': {k: v for k, v in AMP_ALPHAS.items()},
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': nll_summary,
    'primary_comparisons': comparison_results,
    'hypothesis_comparisons': hyp_results,
    'per_bin_results': per_bin_results,
    'dose_response': dose_response,
    'amplification_response': amp_response,
    'interaction_results': interaction_results,
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
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

output_path = "/home/jupyter/research/directed_kvcache_v2/12_long_doc_priming_diagnostic.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
