#!/usr/bin/env python3
"""Build script for 21_gemma_robustness_tuning.ipynb

Exp 21: Gemma Mechanism Robustness & Tuning

Exp 19 discovered that layer-selective value contamination (layers 0-16 only)
amplifies Gemma's priming signal from d=+0.056 to d=+0.211 (p=3.7e-15), a 4x
improvement. This works because late-layer values (17-33) carry interference
that cancels the early-layer benefit.

Two critical questions remain:
1. Does this survive document length scaling? Exp 20 showed Mistral's full
   priming vanishes by ~256 tokens. Does Gemma's layer-selective approach also
   fail at long lengths, or does removing late-layer interference make it more
   robust?
2. Is cutoff=17 optimal? Exp 19 only tested one boundary (layers 0-16).
   Sweeping cutoffs may find a better split point.

Part 1: Length Generalization (N=500, 4 pad lengths, cutoff=16)
Part 2: Layer Boundary Sweep (N=200, 5 cutoffs, no padding)
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
# Exp 21: Gemma Mechanism Robustness & Tuning

## Motivation

Exp 19 discovered that **layer-selective value contamination** (layers 0-16 only)
amplifies Gemma's priming signal from d=+0.056 to d=**+0.211** (p=3.7e-15). This works
because late-layer values (17-33) carry interference that cancels the early-layer benefit.

Two critical questions remain:

### Q1: Does this survive document length scaling?
Exp 20 showed Mistral's full priming vanishes by ~256 tokens. Does Gemma's layer-selective
approach also fail at long lengths, or does removing late-layer interference make it more robust?

### Q2: Is cutoff=17 optimal?
Exp 19 only tested one boundary (layers 0-16). Sweeping cutoffs may find a better split point.

## Design

### Part 1: Length Generalization "Kryptonite" Test
- **N=500 queries** from MS MARCO v1.1 (positive passages only)
- **Pad lengths**: `[None, 512, 1024, 2048]`
- **Fixed cutoff**: layers 0-15 (`list(range(16))`)
- **Method**: `replace_values_at_layers(bare_cache, sf_trunc_cache, list(range(16)))`

### Part 2: Layer Boundary Sweep
- **N=200 queries** (first 200 from Part 1)
- **Cutoffs**: `[8, 12, 16, 20, 24]`
- **No padding** (original passage length)

## Reference Values

| Source | Condition | d |
|--------|-----------|---|
| Exp 19 (Gemma) | values_only (all layers) | +0.056 |
| Exp 19 (Gemma) | values_early_layers (0-16) | +0.211 |
| Exp 20 (Mistral) | full priming @ original | +0.303 |
| Exp 20 (Mistral) | full priming @ 256 tok | +0.114 (ns) |
| Exp 20 (Mistral) | full priming @ 512 tok | +0.034 (ns) |""")))

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

RESULTS_DIR = Path("results/exp21")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_P1_PATH = RESULTS_DIR / "checkpoint_part1.json"
CHECKPOINT_P2_PATH = RESULTS_DIR / "checkpoint_part2.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"
CSV_P1_PATH = RESULTS_DIR / "part1_results.csv"
CSV_P2_PATH = RESULTS_DIR / "part2_results.csv"

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
    compute_dtype="auto",  # resolves to bfloat16 for gemma3
    use_4bit=True,
    num_samples=2000,
    seed=SEED,
)

print(f"Loading {MODEL_NAME} (4-bit, bfloat16)...")
model, tokenizer = load_model(exp_config)

# Architecture diagnostics
from lib.kv_cache import _get_text_config, _get_head_dim, _get_rope_theta_for_layer, _get_cache_keys, _ensure_dynamic_cache

text_config = _get_text_config(model.config)
print(f"\\nModel loaded successfully.")
print(f"  Model class: {type(model).__name__}")
print(f"  Text config class: {type(text_config).__name__}")
print(f"  Hidden size: {text_config.hidden_size}")
print(f"  Num layers: {text_config.num_hidden_layers}")
print(f"  Num attention heads: {text_config.num_attention_heads}")
print(f"  Num KV heads: {text_config.num_key_value_heads}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  BOS token ID: {tokenizer.bos_token_id}")

# Per-layer RoPE diagnostics
thetas = set()
for layer_idx in range(text_config.num_hidden_layers):
    thetas.add(_get_rope_theta_for_layer(model.config, layer_idx))
print(f"  Unique RoPE thetas: {sorted(thetas)}")

# Verify dtype
sample_ids = tokenizer("test", return_tensors="pt")['input_ids'].to(exp_config.device)
with torch.no_grad():
    out = model(sample_ids, use_cache=True)
    cache_check = _ensure_dynamic_cache(out.past_key_values)
    k0 = _get_cache_keys(cache_check, 0)
    print(f"  Cache key dtype: {k0.dtype}")
    print(f"  Cache key shape: {k0.shape}  (batch, kv_heads, seq, head_dim)")
del out, sample_ids
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
    replace_values_at_layers,
    _get_text_config,
)
from lib.analysis import cohens_d
from lib.data import count_words
from scipy import stats
from tqdm.auto import tqdm

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Prefix text
from lib.surrogate import STATIC_SURROGATE_QUERIES
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Experiment parameters
N_PART1 = 500
N_PART2 = 200  # first 200 from Part 1
MAX_PASSAGE_WORDS = 300
CHECKPOINT_EVERY = 25

# Part 1: Length generalization
# Gemma 3 sliding_window=1024 → max safe cache = 1023 positions (incl. BOS).
# Beyond this, sliding attention layers truncate their cache, breaking our
# cross-cache value substitution. Cap at 900 tokens (+ BOS = 901 < 1024).
PAD_LENGTHS = [None, 256, 512, 900]

# Part 2: Layer boundary sweep
CUTOFFS = [8, 12, 16, 20, 24]

# Fixed cutoff for Part 1 (layers 0-15 = 16 layers)
FIXED_CUTOFF = 16

# Reference values
EXP19_REF = {
    'values_only_d': 0.056,
    'values_early_layers_d': 0.211,  # layers 0-16 (17 layers)
}
EXP20_REF = {
    'original_d': 0.303,
    '256_d': 0.114,
    '512_d': 0.034,
}

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  Part 1: N={N_PART1}, pad_lengths={PAD_LENGTHS}, cutoff={FIXED_CUTOFF}")
print(f"  Part 2: N={N_PART2}, cutoffs={CUTOFFS}, no padding")
print(f"  Static fact prefix: '{STATIC_FACT}'")
print(f"\\nExp 19 Gemma reference:")
for k, v in EXP19_REF.items():
    print(f"    {k}: {v:+.3f}")
print(f"\\nExp 20 Mistral reference:")
for k, v in EXP20_REF.items():
    print(f"    {k}: {v:+.3f}")\
""")))

# ========== Cell 4: Load MS MARCO + padding pool ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO v1.1, filter positive passages, build padding pool
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 — POSITIVE PASSAGES ONLY")
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
    found_positive = False
    for i, (ptext, sel) in enumerate(zip(passage_texts, is_selected)):
        if sel == 1 and count_words(ptext) <= MAX_PASSAGE_WORDS:
            if len(queries) < N_PART1 * 3:  # collect 3x for shuffling
                queries.append({
                    'query': query,
                    'answer': answer,
                    'passage': ptext,
                    'word_count': count_words(ptext),
                })
                eval_passage_set.add(ptext)
                found_positive = True
                break

    # Collect non-eval passages for padding pool
    for p in passage_texts:
        if p not in eval_passage_set and count_words(p) <= MAX_PASSAGE_WORDS:
            padding_passages.append(p)

np.random.shuffle(queries)
queries = queries[:N_PART1]
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

max_needed = max(tl for tl in PAD_LENGTHS if tl is not None) * N_PART1
print(f"Max tokens needed: {max_needed:,}")
assert len(padding_ids) > max_needed, (
    f"Padding pool too small: {len(padding_ids):,} < {max_needed:,}"
)
print(f"Pool is {len(padding_ids) / max_needed:.1f}x the max needed. OK.")

del dataset
gc.collect()\
""")))

# ========== Cell 5: Tokenize prefix ==========
cells.append(make_cell("code", s("""\
# Cell 5: Tokenize prefix and verify BPE boundaries

print("=" * 70)
print("PREFIX TOKENIZATION — GEMMA 3 4B")
print("=" * 70)

sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)

sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(exp_config.device)

print(f"\\nStatic fact prefix: '{STATIC_FACT}'")
print(f"  Formatted: '{sf_str.strip()}'")
print(f"  Token length: {sf_ids.shape[1]}")

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
print(f"  static_fact: {match}/{total} tokens match ({100*match/total:.1f}%)")

# Condition explanations
print("\\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

print("\\n### Part 1: Length Generalization ###")
print(f"  Fixed cutoff: {FIXED_CUTOFF} layers (layers 0-{FIXED_CUTOFF-1})")
print(f"  Pad lengths: {PAD_LENGTHS}")
print("  Per sample per length:")
print("    1. Pad doc_ids to target length from pre-tokenized padding pool")
print("    2. Forward pass 1 (bare):    [BOS][padded_doc_ids]")
print("    3. Forward pass 2 (primed):  [BOS][sf_prefix_ids][padded_doc_ids]")
print("       -> truncate prefix -> RoPE correct")
print(f"    4. values_early_layers = replace_values_at_layers(bare, sf_trunc, range({FIXED_CUTOFF}))")
print("    5. Score bare + values_early_layers with deepcopy_cache()")

print("\\n### Part 2: Layer Boundary Sweep ###")
print(f"  Cutoffs: {CUTOFFS}")
print("  No padding (original passage length)")
print("  Per sample:")
print("    1. Forward pass (bare):    [BOS][doc_ids]")
print("    2. Forward pass (primed):  [BOS][sf_prefix_ids][doc_ids]")
print("       -> truncate prefix -> RoPE correct")
print("    3. Score bare once")
print(f"    4. For each cutoff in {CUTOFFS}:")
print("       vel_cache = replace_values_at_layers(bare, sf_trunc, range(cutoff))")
print("       Score vel_cache")\
""")))

# ========== Cell 6: Part 1 loop (Length Generalization) ==========
cells.append(make_cell("code", s("""\
# Cell 6: Part 1 — Length Generalization (N=500, 4 pad lengths, cutoff=16)

print("=" * 70)
print(f"PART 1: LENGTH GENERALIZATION ({N_PART1} queries, {len(PAD_LENGTHS)} lengths)")
print(f"Fixed cutoff: {FIXED_CUTOFF} layers")
print("=" * 70)

# Checkpoint resume
p1_results = []
p1_start_idx = 0

if CHECKPOINT_P1_PATH.exists():
    with open(CHECKPOINT_P1_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in queries[:N_PART1]]
    if ckpt_queries == current_queries:
        p1_results = ckpt['results']
        p1_start_idx = len(p1_results)
        print(f"Resuming from checkpoint: {p1_start_idx}/{N_PART1}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

layer_indices = list(range(FIXED_CUTOFF))
t_start = time.time()

for qidx in tqdm(range(p1_start_idx, N_PART1), initial=p1_start_idx, total=N_PART1,
                  desc="Part 1"):
    qdata = queries[qidx]
    query_prompt = QUERY_TEMPLATE.format(query=qdata['query'])
    answer_text = ANSWER_TEMPLATE.format(answer=qdata['answer'])
    passage = qdata['passage']
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # Matched tokenization: tokenize concatenated then split
    full_text = sf_str + document_text
    full_enc = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=True, padding=False, truncation=False)
    full_ids = full_enc['input_ids'].to(exp_config.device)

    sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                               add_special_tokens=True, padding=False, truncation=False)
    sf_prefix_len_matched = sf_prefix_enc['input_ids'].shape[1]

    bos_id = full_ids[:, :1]
    base_doc_ids = full_ids[:, sf_prefix_len_matched:]
    base_doc_len = base_doc_ids.shape[1]

    del full_enc, full_ids, sf_prefix_enc

    query_rows = []

    for pad_length in PAD_LENGTHS:
        # Pad doc_ids at token level if needed
        if pad_length is not None and base_doc_len < pad_length:
            pad_needed = pad_length - base_doc_len
            max_start = len(padding_ids) - pad_needed
            start = np.random.randint(0, max_start)
            pad_tensor = torch.tensor([padding_ids[start:start + pad_needed]],
                                       device=exp_config.device)
            doc_ids = torch.cat([base_doc_ids, pad_tensor], dim=1)
        else:
            doc_ids = base_doc_ids

        doc_len = doc_ids.shape[1]
        context_len = 1 + doc_len  # BOS + doc

        # Forward pass 1: BARE
        bare_input = torch.cat([bos_id, doc_ids], dim=1)
        with torch.no_grad():
            bare_out = model(input_ids=bare_input,
                             attention_mask=torch.ones_like(bare_input),
                             use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
        del bare_out

        # Forward pass 2: PRIMED (static_fact prefix)
        primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
        with torch.no_grad():
            primed_out = model(input_ids=primed_input,
                               attention_mask=torch.ones_like(primed_input),
                               use_cache=True, return_dict=True)
        primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
        del primed_out

        # Truncate: keep [BOS] + [last doc_len positions]
        trunc_raw = extract_and_truncate_cache_with_bos(primed_full, doc_len)
        prefix_offset = sf_ids.shape[1]
        del primed_full

        # RoPE correct
        sf_trunc_cache = deepcopy_cache(trunc_raw)
        correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
        del trunc_raw

        # Build values_early_layers: bare keys + primed values at layers 0-15
        vel_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, layer_indices)

        # Score bare
        bare_nll = score_answer_with_cache(
            deepcopy_cache(bare_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)

        # Score values_early_layers
        vel_nll = score_answer_with_cache(
            deepcopy_cache(vel_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)

        del bare_cache, sf_trunc_cache, vel_cache, bare_input, primed_input
        if pad_length is not None and base_doc_len < pad_length:
            del pad_tensor
        torch.cuda.empty_cache()

        pad_label = "original" if pad_length is None else str(pad_length)
        query_rows.append({
            'query_idx': qidx,
            'layer_cutoff': FIXED_CUTOFF,
            'pad_length': pad_label,
            'actual_doc_len': doc_len,
            'unprimed_nll': bare_nll,
            'primed_nll': vel_nll,
            'delta_nll': bare_nll - vel_nll,
        })

    p1_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'base_doc_len': base_doc_len,
        'rows': query_rows,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_PART1 - 1:
        ckpt_data = {
            'results': p1_results,
            'query_texts': [q['query'] for q in queries[:N_PART1]],
            'completed': len(p1_results),
            'total': N_PART1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_P1_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - p1_start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_PART1 - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_PART1} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nPart 1 complete: {len(p1_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 7: Part 2 loop (Layer Boundary Sweep) ==========
cells.append(make_cell("code", s("""\
# Cell 7: Part 2 — Layer Boundary Sweep (N=200, 5 cutoffs, no padding)

print("=" * 70)
print(f"PART 2: LAYER BOUNDARY SWEEP ({N_PART2} queries, cutoffs={CUTOFFS})")
print("=" * 70)

# Checkpoint resume
p2_results = []
p2_start_idx = 0

if CHECKPOINT_P2_PATH.exists():
    with open(CHECKPOINT_P2_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in queries[:N_PART2]]
    if ckpt_queries == current_queries:
        p2_results = ckpt['results']
        p2_start_idx = len(p2_results)
        print(f"Resuming from checkpoint: {p2_start_idx}/{N_PART2}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

t_start = time.time()

for qidx in tqdm(range(p2_start_idx, N_PART2), initial=p2_start_idx, total=N_PART2,
                  desc="Part 2"):
    qdata = queries[qidx]
    query_prompt = QUERY_TEMPLATE.format(query=qdata['query'])
    answer_text = ANSWER_TEMPLATE.format(answer=qdata['answer'])
    passage = qdata['passage']
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # Matched tokenization
    full_text = sf_str + document_text
    full_enc = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=True, padding=False, truncation=False)
    full_ids = full_enc['input_ids'].to(exp_config.device)

    sf_prefix_enc = tokenizer(sf_str, return_tensors="pt",
                               add_special_tokens=True, padding=False, truncation=False)
    sf_prefix_len_matched = sf_prefix_enc['input_ids'].shape[1]

    bos_id = full_ids[:, :1]
    doc_ids = full_ids[:, sf_prefix_len_matched:]
    doc_len = doc_ids.shape[1]
    context_len = 1 + doc_len

    del full_enc, full_ids, sf_prefix_enc

    # Forward pass 1: BARE
    bare_input = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_input,
                         attention_mask=torch.ones_like(bare_input),
                         use_cache=True, return_dict=True)
    bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
    del bare_out

    # Forward pass 2: PRIMED
    primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
    with torch.no_grad():
        primed_out = model(input_ids=primed_input,
                           attention_mask=torch.ones_like(primed_input),
                           use_cache=True, return_dict=True)
    primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
    del primed_out

    # Truncate + RoPE correct
    trunc_raw = extract_and_truncate_cache_with_bos(primed_full, doc_len)
    prefix_offset = sf_ids.shape[1]
    del primed_full

    sf_trunc_cache = deepcopy_cache(trunc_raw)
    correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)
    del trunc_raw

    # Score bare once
    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    query_rows = []

    # Score each cutoff
    for cutoff in CUTOFFS:
        vel_cache = replace_values_at_layers(
            bare_cache, sf_trunc_cache, list(range(cutoff)))
        vel_nll = score_answer_with_cache(
            deepcopy_cache(vel_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del vel_cache

        query_rows.append({
            'query_idx': qidx,
            'layer_cutoff': cutoff,
            'pad_length': 'none',
            'actual_doc_len': doc_len,
            'unprimed_nll': bare_nll,
            'primed_nll': vel_nll,
            'delta_nll': bare_nll - vel_nll,
        })

    del bare_cache, sf_trunc_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    p2_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'doc_len': doc_len,
        'bare_nll': bare_nll,
        'rows': query_rows,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_PART2 - 1:
        ckpt_data = {
            'results': p2_results,
            'query_texts': [q['query'] for q in queries[:N_PART2]],
            'completed': len(p2_results),
            'total': N_PART2,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_P2_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - p2_start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_PART2 - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_PART2} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nPart 2 complete: {len(p2_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 8: Part 1 Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 8: Part 1 Analysis — d vs length, statistical tests

import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("PART 1 ANALYSIS: LENGTH GENERALIZATION")
print("=" * 70)

# Collect per-sample deltas by pad_length
p1_deltas = {}
p1_bare = {}
p1_primed = {}
for pl in ['original', '256', '512', '900']:
    p1_deltas[pl] = []
    p1_bare[pl] = []
    p1_primed[pl] = []

for r in p1_results:
    for row in r['rows']:
        pl = row['pad_length']
        if pl in p1_deltas:
            p1_deltas[pl].append(row['delta_nll'])
            p1_bare[pl].append(row['unprimed_nll'])
            p1_primed[pl].append(row['primed_nll'])

# Convert to arrays and filter zeros
p1_arrays = {}
for pl in p1_deltas:
    bare = np.array(p1_bare[pl])
    primed = np.array(p1_primed[pl])
    delta = np.array(p1_deltas[pl])
    valid = (bare != 0) & (primed != 0) & np.isfinite(bare) & np.isfinite(primed)
    p1_arrays[pl] = {
        'bare': bare[valid],
        'primed': primed[valid],
        'delta': delta[valid],
        'n_valid': int(np.sum(valid)),
    }

# Summary table
print(f"\\n{'Pad Length':<12} {'N':>5} {'Mean Bare':>10} {'Mean Primed':>12} "
      f"{'Mean D':>10} {'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
print("-" * 90)

p1_analysis = {}
for pl in ['original', '256', '512', '900']:
    a = p1_arrays[pl]
    d = cohens_d(a['delta'])
    win = np.mean(a['delta'] > 0) * 100
    t_stat, p_val = stats.ttest_1samp(a['delta'], 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{pl:<12} {a['n_valid']:>5} {np.mean(a['bare']):>10.4f} "
          f"{np.mean(a['primed']):>12.4f} {np.mean(a['delta']):>+10.4f} "
          f"{d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
    p1_analysis[pl] = {
        'n_valid': a['n_valid'],
        'mean_bare': float(np.mean(a['bare'])),
        'mean_primed': float(np.mean(a['primed'])),
        'mean_delta': float(np.mean(a['delta'])),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }

# Exp 20 comparison
print("\\n" + "=" * 70)
print("COMPARISON: Gemma layer-selective vs Mistral full priming (Exp 20)")
print("=" * 70)

exp20_lengths = ['original', '256', '512']
exp20_ds = [EXP20_REF.get(f'{pl}_d', EXP20_REF.get('original_d', 0))
            for pl in exp20_lengths]

print(f"\\n{'Length':<12} {'Gemma VEL d':>12} {'Mistral d':>10} {'Diff':>8}")
print("-" * 48)
for pl in exp20_lengths:
    if pl in p1_analysis:
        gemma_d = p1_analysis[pl]['cohens_d']
        mistral_d = EXP20_REF.get(f'{pl}_d', EXP20_REF.get('original_d', 0))
        print(f"{pl:<12} {gemma_d:>+12.3f} {mistral_d:>+10.3f} {gemma_d - mistral_d:>+8.3f}")

# Verdict (compare original to 900, the max safe length for Gemma's sliding window)
d_orig = p1_analysis['original']['cohens_d']
d_max = p1_analysis['900']['cohens_d']
if d_orig > 0.1 and d_max < 0.05:
    p1_verdict = ("Layer-selective values ARE NOT robust to length. "
                  f"d decays from {d_orig:+.3f} at original to {d_max:+.3f} at 900 tokens.")
elif d_max > 0.1:
    p1_verdict = ("Layer-selective values ARE robust to length! "
                  f"d remains {d_max:+.3f} even at 900 tokens (vs {d_orig:+.3f} at original).")
else:
    p1_verdict = (f"Mixed: d goes from {d_orig:+.3f} (original) to {d_max:+.3f} (900). "
                  "See detailed numbers above.")

print(f"\\nVERDICT: {p1_verdict}")\
""")))

# ========== Cell 9: Part 2 Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Part 2 Analysis — d vs cutoff, optimal cutoff

print("=" * 70)
print("PART 2 ANALYSIS: LAYER BOUNDARY SWEEP")
print("=" * 70)

# Collect per-sample deltas by cutoff
p2_deltas = {}
p2_bare_all = []
for cutoff in CUTOFFS:
    p2_deltas[cutoff] = []

for r in p2_results:
    p2_bare_all.append(r['bare_nll'])
    for row in r['rows']:
        cutoff = row['layer_cutoff']
        if cutoff in p2_deltas:
            p2_deltas[cutoff].append(row['delta_nll'])

p2_bare_arr = np.array(p2_bare_all)

# Summary table
print(f"\\n{'Cutoff':<10} {'Layers':<15} {'N':>5} {'Mean D':>10} "
      f"{'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
print("-" * 80)

p2_analysis = {}
best_cutoff = None
best_d = -999

for cutoff in CUTOFFS:
    delta = np.array(p2_deltas[cutoff])
    valid = np.isfinite(delta)
    delta = delta[valid]
    n_valid = len(delta)
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    layer_desc = f"0-{cutoff-1}"
    print(f"{cutoff:<10} {layer_desc:<15} {n_valid:>5} {np.mean(delta):>+10.4f} "
          f"{d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
    p2_analysis[cutoff] = {
        'n_valid': n_valid,
        'layers': layer_desc,
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }
    if d > best_d:
        best_d = d
        best_cutoff = cutoff

# Exp 19 comparison
print(f"\\nOptimal cutoff: {best_cutoff} layers (d={best_d:+.3f})")
print(f"Exp 19 reference: layers 0-16 (17 layers), d={EXP19_REF['values_early_layers_d']:+.3f}")
print(f"Exp 19 values_only (all layers): d={EXP19_REF['values_only_d']:+.3f}")

# Hardness interaction for Part 2
print("\\n" + "=" * 70)
print("HARDNESS × CUTOFF INTERACTION (Part 2)")
print("=" * 70)

# Compute quintile boundaries from bare NLL
quintile_boundaries = np.percentile(p2_bare_arr, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in p2_bare_arr])

header = f"{'Cutoff':<10}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (10 + 14 * 6))

hardness_data = {}
for cutoff in CUTOFFS:
    delta = np.array(p2_deltas[cutoff])
    row_str = f"{cutoff:<10}"
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
    hardness_data[cutoff] = quintile_ds\
""")))

# ========== Cell 10: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 10: Plots — 4-panel figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- Panel 1 (top-left): Part 1 — d vs Document Length ----
ax = axes[0, 0]

length_labels = ['original', '256', '512', '900']
gemma_ds = [p1_analysis[pl]['cohens_d'] for pl in length_labels]
x_lengths = []
for pl in length_labels:
    if pl == 'original':
        # Mean original doc length
        x_lengths.append(np.mean([r['base_doc_len'] for r in p1_results]))
    else:
        x_lengths.append(int(pl))

# Bootstrap 95% CI
np.random.seed(SEED)
ci_lo, ci_hi = [], []
for pl in length_labels:
    delta = p1_arrays[pl]['delta']
    boot_ds = []
    for _ in range(2000):
        idx_boot = np.random.randint(0, len(delta), size=len(delta))
        boot_ds.append(cohens_d(delta[idx_boot]))
    boot_ds = np.array(boot_ds)
    ci_lo.append(np.percentile(boot_ds, 2.5))
    ci_hi.append(np.percentile(boot_ds, 97.5))
ci_lo = np.array(ci_lo)
ci_hi = np.array(ci_hi)

ax.errorbar(x_lengths, gemma_ds,
            yerr=[np.array(gemma_ds) - ci_lo, ci_hi - np.array(gemma_ds)],
            marker='o', markersize=8, linewidth=2, capsize=5,
            color='#9467bd', ecolor='#c5b0d5', label='Gemma VEL (this exp)')

# Exp 20 Mistral reference (only lengths we overlap with)
exp20_x = [130, 256, 512]
exp20_y = [EXP20_REF['original_d'], EXP20_REF['256_d'], EXP20_REF['512_d']]
ax.plot(exp20_x, exp20_y, marker='s', markersize=6, linewidth=1.5, linestyle='--',
        color='#d62728', alpha=0.7, label='Mistral full priming (Exp 20)')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel('Document Token Length')
ax.set_ylabel("Cohen's d (positive = helps)")
ax.set_title("Part 1: d vs Document Length")
ax.legend(fontsize=8)

for i, pl in enumerate(length_labels):
    p_val = p1_analysis[pl]['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.annotate(f'{pl}\\nd={gemma_ds[i]:+.3f} {sig}',
                (x_lengths[i], gemma_ds[i]),
                textcoords='offset points', xytext=(0, 18),
                ha='center', fontsize=7)

# ---- Panel 2 (top-right): Part 1 — Win Rate vs Length ----
ax = axes[0, 1]

wins = [p1_analysis[pl]['win_pct'] for pl in length_labels]
ax.plot(x_lengths, wins, marker='s', markersize=8, linewidth=2, color='#2ca02c')
ax.axhline(y=50, color='black', linestyle='--', linewidth=0.8, label='Chance (50%)')

for i, pl in enumerate(length_labels):
    ax.annotate(f'{pl}\\n{wins[i]:.1f}%',
                (x_lengths[i], wins[i]),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=8)

ax.set_xscale('log')
ax.set_xlabel('Document Token Length')
ax.set_ylabel('Win Rate (%)')
ax.set_title("Part 1: Win Rate vs Length")
ax.legend(fontsize=8)

# ---- Panel 3 (bottom-left): Part 2 — d vs Layer Cutoff ----
ax = axes[1, 0]

cutoff_ds = [p2_analysis[c]['cohens_d'] for c in CUTOFFS]
x_pos = range(len(CUTOFFS))
colors_bar = ['#1f77b4' if c != best_cutoff else '#ff7f0e' for c in CUTOFFS]
bars = ax.bar(x_pos, cutoff_ds, color=colors_bar, edgecolor='black', linewidth=0.5)

# Exp 19 reference
ax.axhline(y=EXP19_REF['values_early_layers_d'], color='#9467bd', linestyle='--',
           linewidth=1.5, label=f"Exp 19 (17 layers) d={EXP19_REF['values_early_layers_d']:+.3f}")
ax.axhline(y=EXP19_REF['values_only_d'], color='#7f7f7f', linestyle=':',
           linewidth=1.5, label=f"Exp 19 values_only d={EXP19_REF['values_only_d']:+.3f}")
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xticks(x_pos)
ax.set_xticklabels([str(c) for c in CUTOFFS])
ax.set_xlabel('Layer Cutoff (layers 0 to cutoff-1)')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title(f"Part 2: d vs Layer Cutoff (best={best_cutoff})")
ax.legend(fontsize=7)

for i, d_val in enumerate(cutoff_ds):
    ax.text(i, d_val + 0.003 if d_val >= 0 else d_val - 0.012,
            f"{d_val:+.3f}", ha='center',
            va='bottom' if d_val >= 0 else 'top', fontsize=9)

# ---- Panel 4 (bottom-right): Part 2 — Hardness x Cutoff heatmap ----
ax = axes[1, 1]

heatmap_data = np.zeros((len(CUTOFFS), 5))
for i, cutoff in enumerate(CUTOFFS):
    for q in range(5):
        val = hardness_data[cutoff][q]
        heatmap_data[i, q] = val if val is not None else np.nan

im = ax.imshow(heatmap_data, cmap='RdBu', aspect='auto',
               vmin=-0.5, vmax=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=8)
ax.set_yticks(range(len(CUTOFFS)))
ax.set_yticklabels([str(c) for c in CUTOFFS])
ax.set_xlabel('Difficulty Quintile')
ax.set_ylabel('Layer Cutoff')
ax.set_title("Part 2: Hardness x Cutoff (Cohen's d)")

# Add text annotations
for i in range(len(CUTOFFS)):
    for j in range(5):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    fontsize=7, color='white' if abs(val) > 0.25 else 'black')

fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")

plt.suptitle('Exp 21: Gemma Mechanism Robustness & Tuning', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 11: Save results + CSVs ==========
cells.append(make_cell("code", s("""\
# Cell 11: Save results.json + CSVs
import csv

# --- Part 1 CSV ---
with open(CSV_P1_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'layer_cutoff', 'pad_length', 'actual_doc_len',
        'unprimed_nll', 'primed_nll', 'delta_nll'])
    writer.writeheader()
    for r in p1_results:
        for row in r['rows']:
            writer.writerow(row)
print(f"Part 1 CSV saved: {CSV_P1_PATH}")

# --- Part 2 CSV ---
with open(CSV_P2_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'layer_cutoff', 'pad_length', 'actual_doc_len',
        'unprimed_nll', 'primed_nll', 'delta_nll'])
    writer.writeheader()
    for r in p2_results:
        for row in r['rows']:
            writer.writerow(row)
print(f"Part 2 CSV saved: {CSV_P2_PATH}")

# --- Combined results.json ---
final = {
    'experiment': 'exp21_gemma_robustness_tuning',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'dataset': 'MS MARCO v1.1 validation',
        'max_passage_words': MAX_PASSAGE_WORDS,
        'part1': {
            'n_queries': N_PART1,
            'pad_lengths': [str(pl) if pl else 'original' for pl in PAD_LENGTHS],
            'fixed_cutoff': FIXED_CUTOFF,
        },
        'part2': {
            'n_queries': N_PART2,
            'cutoffs': CUTOFFS,
        },
    },
    'gemma_architecture': {
        'hidden_size': text_config.hidden_size,
        'num_layers': text_config.num_hidden_layers,
        'num_attention_heads': text_config.num_attention_heads,
        'num_kv_heads': text_config.num_key_value_heads,
        'head_dim': _get_head_dim(model.config),
        'rope_thetas': sorted(list(thetas)),
    },
    'part1_analysis': p1_analysis,
    'part1_verdict': p1_verdict,
    'part2_analysis': {str(k): v for k, v in p2_analysis.items()},
    'part2_best_cutoff': best_cutoff,
    'part2_best_d': float(best_d),
    'part2_hardness_data': {str(k): v for k, v in hardness_data.items()},
    'reference_values': {
        'exp19_gemma': EXP19_REF,
        'exp20_mistral': EXP20_REF,
    },
    'part1_per_query': p1_results,
    'part2_per_query': p2_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Final summary
print("\\n" + "=" * 70)
print("SUMMARY — Exp 21: Gemma Mechanism Robustness & Tuning")
print("=" * 70)
print(f"Model: Gemma 3 4B (34 layers, head_dim=256, bfloat16)")
print(f"\\nPart 1: Length Generalization (cutoff={FIXED_CUTOFF})")
for pl in ['original', '256', '512', '900']:
    a = p1_analysis[pl]
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  {pl:<12} d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  {sig}")
print(f"  Verdict: {p1_verdict}")
print(f"\\nPart 2: Layer Boundary Sweep")
for cutoff in CUTOFFS:
    a = p2_analysis[cutoff]
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    marker = " <-- BEST" if cutoff == best_cutoff else ""
    print(f"  cutoff={cutoff:<4} d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  {sig}{marker}")
print(f"  Optimal cutoff: {best_cutoff} layers (d={best_d:+.3f})")
print(f"\\nDone!")\
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

output_path = "/home/jupyter/research/directed_kvcache_v2/21_gemma_robustness_tuning.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
