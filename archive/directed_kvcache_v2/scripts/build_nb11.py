#!/usr/bin/env python3
"""Build script for 11_long_document_priming.ipynb

Exp 11: Long-Document Priming — Does KV Cache Priming Scale to Longer Documents?

All v2 experiments used MS MARCO v1.1 passages (avg ~60 words, max 300 words).
In production, documents are much longer. This experiment tests whether our
key findings (especially static_fact_trunc dominance) hold on longer documents
using Natural Questions, which pairs real Google queries with full Wikipedia articles.

Design:
  - Dataset: Natural Questions (validation split)
  - Length bins: 100-300w, 300-800w, 800-2000w, 2000-4000w
  - 5 conditions: bare, static_fact_trunc, random_trunc, llm_kw_trunc, oracle_trunc
  - ~125 samples per length bin (500 total)
  - 5 primary comparisons + per-bin interaction analysis
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
# Exp 11: Long-Document Priming — Does It Scale?

## Motivation

All v2 experiments (01-10) used MS MARCO v1.1, where passages average ~60 words
(max 300 words). In production ad-serving, documents will be much longer — full
web pages, articles, product descriptions. The key question: **does our best
priming approach (static_fact_trunc, d=+0.438 on MS MARCO) still work on longer
documents?**

Prior evidence (v1 Exp 19) showed priming hurts on datasets with longer documents
(CNN/DailyMail d=-1.31, NarrativeQA d=-0.35), but that used full-context mode
with known bugs (BPE mismatch, Document:\\n framing). This experiment re-tests
with clean v2 methodology (truncated prefix, matched tokenization, no framing).

## Dataset: Natural Questions

Real Google search queries over full Wikipedia articles:
- **Queries**: Short factoid Google queries (closest to production ad queries)
- **Documents**: Full Wikipedia article text (100-10000+ words)
- **Answers**: Short extractive answer spans (entity names, dates, numbers)

## Design

| Parameter | Value |
|-----------|-------|
| Dataset | Natural Questions (validation) |
| Length bins | 100-300w, 300-800w, 800-2000w, 2000-4000w |
| Samples per bin | ~125 (500 total) |
| Conditions | 5: bare, static_fact_trunc, random_trunc, llm_kw_trunc, oracle_trunc |
| Forward passes | 5 × 500 = 2500 |
| Estimated runtime | 4-6 hours |

## 5 Primary Comparisons (Bonferroni alpha = 0.01)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | static_fact_trunc vs bare | Does static_fact help overall? |
| C2 | random_trunc vs bare | Does ANY prefix help overall? |
| C3 | llm_kw_trunc vs bare | Do LLM keywords help overall? |
| C4 | oracle_trunc vs bare | Does the perfect query help? |
| C5 | static_fact_trunc vs random_trunc | Is content better than noise? |

## Key Interaction Analysis

For each comparison, test per length bin. If static_fact benefit decreases with
document length, we expect a negative slope in the length × d regression.""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import re
from collections import Counter
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp11")
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

# ========== Cell 3: Config, constants, helpers ==========
cells.append(make_cell("code", s("""\
# Cell 3: Config, constants, and helper functions
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    score_answer_with_cache,
)
from lib.analysis import cohens_d
from lib.surrogate import STATIC_SURROGATE_QUERIES, generate_surrogate_with_template
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,  # pool to draw from
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

N_EVAL = 500  # total target (125 per bin)
N_COMPARISONS = 5
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
CHECKPOINT_EVERY = 25

STATIC_FACTUAL_PHRASE = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Length bins (word count)
LENGTH_BINS = [
    ('short',     100,  300),   # MS MARCO-like
    ('medium',    300,  800),   # Moderate web page
    ('long',      800,  2000),  # Full article section
    ('very_long', 2000, 4000),  # Full article
]
SAMPLES_PER_BIN = 125
MAX_DOC_WORDS = 4000  # hard cap (context window safety)

CONDITION_NAMES = [
    'bare',
    'static_fact_trunc',
    'random_trunc',
    'llm_kw_trunc',
    'oracle_trunc',
]

# LLM keyword generation prompt (same as previous experiments)
LLM_KW_PROMPT = (
    "You are helping index a document for search. Write a search query the way "
    "real users type into Google: just keywords, no complete sentences, no question marks. "
    "Think of someone quickly typing a few relevant words. "
    "Output only the keyword query (3-6 words), nothing else.\\n\\n"
    "Document:"
)

# Max words of document to show for LLM keyword generation (truncate long docs)
LLM_KW_MAX_DOC_WORDS = 500


def build_primed_and_truncated(prefix_text, bos_id, doc_ids, doc_len, model, tokenizer, config):
    \"\"\"Build a primed cache: tokenize prefix, concat [BOS][prefix][doc], forward, truncate+RoPE.

    Returns:
        (trunc_cache, prefix_token_len)
    \"\"\"
    prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=prefix_text)
    prefix_enc = tokenizer(prefix_str, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
    prefix_ids = prefix_enc['input_ids'].to(config.device)
    prefix_token_len = 1 + prefix_ids.shape[1]  # BOS + prefix tokens

    full_ids = torch.cat([bos_id, prefix_ids, doc_ids], dim=1)

    with torch.no_grad():
        out = model(input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=True, return_dict=True)

    trunc_cache = extract_and_truncate_cache_with_bos(out.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_cache, prefix_token_len - 1, model)

    del out
    return trunc_cache, prefix_token_len


print("Config ready")
print(f"  N_EVAL: {N_EVAL}")
print(f"  SAMPLES_PER_BIN: {SAMPLES_PER_BIN}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: {len(CONDITION_NAMES)}")
print(f"  static_factual_phrase: '{STATIC_FACTUAL_PHRASE}'")
print(f"  length_bins: {LENGTH_BINS}")
print(f"  max_doc_words: {MAX_DOC_WORDS}")\
""")))

# ========== Cell 4: Load Natural Questions ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load Natural Questions — extract clean text + short answers, stratify by length
from datasets import load_dataset

print("=" * 70)
print("LOADING NATURAL QUESTIONS (validation split)")
print("=" * 70)

# Check for cached samples
SAMPLES_CACHE_PATH = RESULTS_DIR / "nq_samples.json"

if SAMPLES_CACHE_PATH.exists():
    with open(SAMPLES_CACHE_PATH, 'r') as f:
        cached = json.load(f)
    samples = cached['samples']
    print(f"Loaded {len(samples)} cached NQ samples from {SAMPLES_CACHE_PATH}")
else:
    print("Loading NQ dataset (streaming mode)...")
    nq = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True,
    )

    # Collect samples into length bins
    bin_samples = {name: [] for name, _, _ in LENGTH_BINS}
    n_processed = 0
    n_no_answer = 0
    n_too_short = 0
    n_too_long = 0

    for example in tqdm(nq, desc="Processing NQ"):
        n_processed += 1

        # Extract clean document text (non-HTML tokens)
        doc_tokens = example['document']['tokens']
        if isinstance(doc_tokens, dict):
            # HF may return dict of lists instead of list of dicts
            token_strs = doc_tokens['token']
            is_html_flags = doc_tokens['is_html']
            clean_tokens = [t for t, h in zip(token_strs, is_html_flags) if not h]
        else:
            clean_tokens = [t['token'] for t in doc_tokens if not t['is_html']]

        doc_text = ' '.join(clean_tokens)
        word_count = len(doc_text.split())

        # Skip if outside our range
        if word_count < LENGTH_BINS[0][1]:  # below minimum
            n_too_short += 1
            continue
        if word_count > MAX_DOC_WORDS:
            # Truncate to MAX_DOC_WORDS
            words = doc_text.split()
            doc_text = ' '.join(words[:MAX_DOC_WORDS])
            word_count = MAX_DOC_WORDS

        # Extract short answer
        annotations = example['annotations']
        short_answers_list = annotations['short_answers']

        answer_text = None
        # NQ short_answers is a list of dicts (one per annotator).
        # Each dict has 'start_token', 'end_token', 'text' as lists (one entry per span).
        # Use 'text' directly when available; fall back to token reconstruction.
        for annotator_sa in short_answers_list:
            if not annotator_sa:
                continue
            # Use pre-extracted text field if available
            texts = annotator_sa.get('text', [])
            if texts:
                answer_text = texts[0]
                break
            # Fallback: reconstruct from document tokens
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
            n_no_answer += 1
            continue

        # Skip very long answers (>20 words) — we want factoid answers
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

        # Assign to length bin
        assigned = False
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
                assigned = True
                break

        # Check if all bins are full
        all_full = all(len(bin_samples[name]) >= SAMPLES_PER_BIN for name, _, _ in LENGTH_BINS)
        if all_full:
            print(f"All bins full after processing {n_processed} examples.")
            break

    # Combine all bins
    samples = []
    for bin_name, _, _ in LENGTH_BINS:
        bin_s = bin_samples[bin_name]
        np.random.seed(SEED)
        np.random.shuffle(bin_s)
        samples.extend(bin_s)
        print(f"  {bin_name}: {len(bin_s)} samples")

    print(f"\\nTotal samples: {len(samples)}")
    print(f"Processed: {n_processed}, No answer: {n_no_answer}, Too short: {n_too_short}")

    # Cache for fast reload
    with open(SAMPLES_CACHE_PATH, 'w') as f:
        json.dump({'samples': samples, 'n_processed': n_processed}, f)
    print(f"Cached to {SAMPLES_CACHE_PATH}")

N = len(samples)

# Summary statistics
print(f"\\n{'='*70}")
print(f"SAMPLE SUMMARY")
print(f"{'='*70}")
for bin_name, bin_min, bin_max in LENGTH_BINS:
    bin_s = [s for s in samples if s['length_bin'] == bin_name]
    if bin_s:
        wcs = [s['word_count'] for s in bin_s]
        print(f"  {bin_name} ({bin_min}-{bin_max}w): n={len(bin_s)}, "
              f"mean={np.mean(wcs):.0f}w, range=[{min(wcs)}, {max(wcs)}]")

print(f"\\nExample (short):")
ex_short = [s for s in samples if s['length_bin'] == 'short']
if ex_short:
    print(f"  Q: {ex_short[0]['query']}")
    print(f"  A: {ex_short[0]['answer']}")
    print(f"  Doc ({ex_short[0]['word_count']}w): {ex_short[0]['passage'][:150]}...")

print(f"\\nExample (very_long):")
ex_long = [s for s in samples if s['length_bin'] == 'very_long']
if ex_long:
    print(f"  Q: {ex_long[0]['query']}")
    print(f"  A: {ex_long[0]['answer']}")
    print(f"  Doc ({ex_long[0]['word_count']}w): {ex_long[0]['passage'][:150]}...")\
""")))

# ========== Cell 5: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 5: Condition explanation
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

conditions_explained = [
    ("1. bare",
     "[BOS][doc]",
     "No prefix — baseline. Identical to bare caches used in Exps 01-10."),
    ("2. static_fact_trunc",
     "[BOS][static_fact\\\\n][doc] → truncate + RoPE",
     f"Best condition from Exp 07/10 (d=+0.438 on MS MARCO). Phrase: '{STATIC_FACTUAL_PHRASE}'"),
    ("3. random_trunc",
     "[BOS][random_tokens\\\\n][doc] → truncate + RoPE",
     "Structural control — random vocabulary tokens. Isolates non-semantic value contamination."),
    ("4. llm_kw_trunc",
     "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     "LLM-generated keyword query from first 500 words of doc. Tests doc-specific surrogates."),
    ("5. oracle_trunc",
     "[BOS][oracle_query\\\\n][doc] → truncate + RoPE",
     "Oracle (actual NQ query) as prefix. Upper bound for query-specific priming."),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")

print(f"\\n{'='*70}")
print("KEY QUESTION: How does each condition's d vs bare change across length bins?")
print(f"{'='*70}")

# Show token counts for different length docs
print(f"\\nExpected token counts per bin (approx):")
for bin_name, bin_min, bin_max in LENGTH_BINS:
    mid_words = (bin_min + bin_max) // 2
    approx_tokens = int(mid_words * 1.5)  # rough word-to-token ratio
    print(f"  {bin_name} ({bin_min}-{bin_max}w): ~{approx_tokens} tokens per doc")\
""")))

# ========== Cell 6: LLM keyword generation ==========
cells.append(make_cell("code", s("""\
# Cell 6: Generate LLM keyword surrogates
print("=" * 70)
print("PHASE 1: LLM KEYWORD GENERATION")
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
    print(f"(Using first {LLM_KW_MAX_DOC_WORDS} words of each doc for generation)")
    t_start = time.time()

    for idx in tqdm(range(start_idx_gen, N), initial=start_idx_gen, total=N,
                     desc="Keyword surrogates"):
        passage = samples[idx]['passage']
        # Truncate to first LLM_KW_MAX_DOC_WORDS for generation efficiency
        words = passage.split()
        if len(words) > LLM_KW_MAX_DOC_WORDS:
            passage_for_gen = ' '.join(words[:LLM_KW_MAX_DOC_WORDS])
        else:
            passage_for_gen = passage

        try:
            kw = generate_surrogate_with_template(
                passage_for_gen, LLM_KW_PROMPT, model, tokenizer, config)
        except Exception as e:
            print(f"  WARNING: Generation failed for sample {idx}: {e}")
            kw = ""
        keyword_surrogates.append(kw)

        if (idx + 1) % 50 == 0 or idx == N - 1:
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
if keyword_surrogates:
    print(f"Example: '{keyword_surrogates[0]}'")\
""")))

# ========== Cell 7: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main eval loop — 5 conditions × N samples
print("=" * 70)
print(f"PHASE 2: MAIN EVALUATION (5 conditions × {N} samples)")
print("=" * 70)

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
    word_count = sample['word_count']
    length_bin = sample['length_bin']

    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

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

    # ===== 1. BARE =====
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = bare_out.past_key_values
    del bare_out

    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_cache), bare_ids.shape[1],
        query_prompt, answer_text, model, tokenizer, config)

    # ===== 2. static_fact_trunc =====
    trunc_cache, _ = build_primed_and_truncated(
        STATIC_FACTUAL_PHRASE, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_static = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # ===== 3. random_trunc =====
    llm_kw_text = keyword_surrogates[idx] if idx < len(keyword_surrogates) else ""
    n_random_tokens = max(5, len(tokenizer.encode(
        llm_kw_text if llm_kw_text else STATIC_FACTUAL_PHRASE,
        add_special_tokens=False)))
    random_ids = torch.randint(100, tokenizer.vocab_size - 100, (n_random_tokens,), device='cpu')
    random_text = tokenizer.decode(random_ids, skip_special_tokens=True)

    trunc_cache, _ = build_primed_and_truncated(
        random_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_random = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # ===== 4. llm_kw_trunc =====
    if llm_kw_text.strip():
        trunc_cache, _ = build_primed_and_truncated(
            llm_kw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
        nll_llm_kw = score_answer_with_cache(
            deepcopy_cache(trunc_cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del trunc_cache
    else:
        nll_llm_kw = 0.0  # empty surrogate → exclude from analysis

    # ===== 5. oracle_trunc =====
    trunc_cache, _ = build_primed_and_truncated(
        query, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_oracle = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    del bare_cache, bare_ids
    torch.cuda.empty_cache()

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len_tokens': doc_len,
        'word_count': word_count,
        'length_bin': length_bin,
        'bare': nll_bare,
        'static_fact_trunc': nll_static,
        'random_trunc': nll_random,
        'llm_kw_trunc': nll_llm_kw,
        'oracle_trunc': nll_oracle,
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
# Cell 8: Analysis — overall + per length bin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — LONG-DOCUMENT PRIMING")
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
print(f"\\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10} {'Win%':>7}")
print("-" * 67)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        print(f"{cname:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {'—':>10} {'—':>7}")
    else:
        delta = c['bare'] - c[cname]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        _, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
        print(f"{cname:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d:>+10.3f} {win:>5.1f}% {sig}")

# ===== 5 PRIMARY COMPARISONS =====
print(f"\\n{'='*90}")
print(f"5 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*90}")

comparisons = [
    ('C1: static_fact vs bare',
     c['bare'] - c['static_fact_trunc'],
     'Does static_fact help overall?'),
    ('C2: random vs bare',
     c['bare'] - c['random_trunc'],
     'Does ANY prefix help overall?'),
    ('C3: llm_kw vs bare',
     c['bare'] - c['llm_kw_trunc'],
     'Do LLM keywords help overall?'),
    ('C4: oracle vs bare',
     c['bare'] - c['oracle_trunc'],
     'Does the perfect query help?'),
    ('C5: static_fact vs random',
     c['random_trunc'] - c['static_fact_trunc'],
     'Is content better than noise?'),
]

print(f"\\n{'Comparison':<30} {'Mean delta':>10} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 85)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<30} {np.mean(delta):>10.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        'question': question,
    }

# ===== PER LENGTH BIN ANALYSIS (KEY RESULT) =====
print(f"\\n{'='*90}")
print("PER LENGTH BIN ANALYSIS — Does priming effect change with document length?")
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

# ===== LENGTH INTERACTION: does d decrease with length? =====
print(f"\\n{'='*90}")
print("LENGTH INTERACTION — Correlation between document length and priming effect")
print(f"{'='*90}")

from scipy.stats import spearmanr, pearsonr

interaction_results = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    r_spear, p_spear = spearmanr(word_counts_arr, delta)
    r_pears, p_pears = pearsonr(word_counts_arr, delta)
    print(f"  {cname}: Spearman r={r_spear:+.3f} (p={p_spear:.3f}), Pearson r={r_pears:+.3f} (p={p_pears:.3f})")
    interaction_results[cname] = {
        'spearman_r': float(r_spear), 'spearman_p': float(p_spear),
        'pearson_r': float(r_pears), 'pearson_p': float(p_pears),
    }

# ===== HARDNESS QUINTILE (WITHIN EACH BIN) =====
print(f"\\n{'='*90}")
print("HARDNESS QUINTILE WITHIN EACH LENGTH BIN")
print(f"{'='*90}")

hardness_x_length = {}
for bin_name in bin_names_ordered:
    mask_bin = length_bins_arr == bin_name
    n_bin = int(np.sum(mask_bin))
    if n_bin < 30:
        continue
    bare_bin = c['bare'][mask_bin]
    median_nll = np.median(bare_bin)
    hard_mask_within = bare_bin >= median_nll
    easy_mask_within = bare_bin < median_nll

    print(f"\\n  {bin_name} (n={n_bin}, median bare NLL={median_nll:.3f}):")
    bin_results = {}
    for cname in ['static_fact_trunc', 'random_trunc', 'oracle_trunc']:
        cond_bin = c[cname][mask_bin]
        delta_easy = bare_bin[easy_mask_within] - cond_bin[easy_mask_within]
        delta_hard = bare_bin[hard_mask_within] - cond_bin[hard_mask_within]
        d_easy = cohens_d(delta_easy) if len(delta_easy) > 5 else float('nan')
        d_hard = cohens_d(delta_hard) if len(delta_hard) > 5 else float('nan')
        print(f"    {cname}: easy d={d_easy:+.3f}, hard d={d_hard:+.3f}")
        bin_results[cname] = {'easy_d': float(d_easy), 'hard_d': float(d_hard)}
    hardness_x_length[bin_name] = bin_results\
""")))

# ========== Cell 9: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 9: Plots (2x2 grid)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

colors = {
    'static_fact_trunc': '#d62728',
    'random_trunc': '#7f7f7f',
    'llm_kw_trunc': '#2ca02c',
    'oracle_trunc': '#1f77b4',
}

# --- Plot 1: Per-bin Cohen's d for each condition ---
ax = axes[0, 0]
x = np.arange(len(bin_names_ordered))
width = 0.18
for i, cname in enumerate(['static_fact_trunc', 'random_trunc', 'llm_kw_trunc', 'oracle_trunc']):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, ds_clean, width, label=cname.replace('_trunc', ''),
                  color=colors[cname], edgecolor='black', linewidth=0.5, alpha=0.85)
    for j, (d_val, bar) in enumerate(zip(ds, bars)):
        if d_val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{d_val:+.2f}", ha='center', va='bottom', fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_xlabel("Document Length Bin")
ax.set_title("Priming Effect by Document Length")
ax.legend(fontsize=8)

# --- Plot 2: Scatter — word count vs per-sample NLL reduction ---
ax = axes[0, 1]
for cname in ['static_fact_trunc', 'oracle_trunc']:
    delta = c['bare'] - c[cname]
    ax.scatter(word_counts_arr, delta, alpha=0.15, s=8, color=colors[cname], label=cname.replace('_trunc', ''))
    # Trend line (binned means)
    n_trend_bins = 20
    edges = np.linspace(word_counts_arr.min(), word_counts_arr.max(), n_trend_bins + 1)
    for k in range(n_trend_bins):
        mask_k = (word_counts_arr >= edges[k]) & (word_counts_arr < edges[k+1])
        if np.sum(mask_k) > 5:
            ax.scatter((edges[k] + edges[k+1])/2, np.mean(delta[mask_k]),
                      s=40, color=colors[cname], edgecolor='black', linewidth=0.5, zorder=5)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel("Document Word Count")
ax.set_ylabel("NLL Reduction (bare - primed)")
ax.set_title("NLL Reduction vs Document Length")
ax.legend(fontsize=8)

# --- Plot 3: Overall bar chart (all conditions) ---
ax = axes[1, 0]
conds_sorted = sorted(
    [(cn, cohens_d(c['bare'] - c[cn])) for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda x: x[1], reverse=True
)
names_sorted = [x[0] for x in conds_sorted]
ds_sorted = [x[1] for x in conds_sorted]
bar_colors = [colors.get(cn, 'gray') for cn in names_sorted]
bars = ax.barh(range(len(names_sorted)), ds_sorted, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels([n.replace('_trunc', '') for n in names_sorted], fontsize=9)
for i, (name, d_val) in enumerate(conds_sorted):
    ax.text(d_val + 0.005, i, f"d={d_val:+.3f}", va='center', fontsize=8)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title("Overall Priming Effect (All Length Bins)")
ax.invert_yaxis()

# Reference line from MS MARCO
ax.axvline(x=0.438, color='red', linestyle=':', alpha=0.5, label='Exp10 MS MARCO static_fact')
ax.legend(fontsize=7)

# --- Plot 4: Hardness × Length heatmap for static_fact_trunc ---
ax = axes[1, 1]
hm_data = []
hm_labels_y = []
for bin_name in bin_names_ordered:
    if bin_name in hardness_x_length:
        hm_data.append([
            hardness_x_length[bin_name]['static_fact_trunc']['easy_d'],
            hardness_x_length[bin_name]['static_fact_trunc']['hard_d'],
        ])
        hm_labels_y.append(bin_name)
if hm_data:
    hm_data = np.array(hm_data)
    im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Easy (below median)', 'Hard (above median)'])
    ax.set_yticks(range(len(hm_labels_y)))
    ax.set_yticklabels(hm_labels_y)
    for i in range(len(hm_labels_y)):
        for j in range(2):
            ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
    ax.set_title("static_fact_trunc: Hardness × Length")

plt.suptitle('Exp 11: Long-Document Priming', fontsize=14, y=1.02)
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
    'experiment': 'exp11_long_document_priming',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_eval': N,
        'n_valid': n_valid,
        'n_excluded': n_excluded,
        'n_conditions': len(CONDITION_NAMES),
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
        'dataset': 'google-research-datasets/natural_questions',
        'dataset_split': 'validation',
        'length_bins': LENGTH_BINS,
        'max_doc_words': MAX_DOC_WORDS,
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': nll_summary,
    'primary_comparisons': comparison_results,
    'per_bin_results': per_bin_results,
    'interaction_results': interaction_results,
    'hardness_x_length': hardness_x_length,
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

output_path = "/home/jupyter/research/directed_kvcache_v2/11_long_document_priming.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
