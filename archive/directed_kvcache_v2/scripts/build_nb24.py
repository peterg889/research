#!/usr/bin/env python3
"""Build script for 24_gemma_layer_mechanism.ipynb

Exp 24: Gemma Layer-Selective Mechanism Deep Dive

Exp 21 confirmed layer-selective value contamination (layers 0-15, cutoff=16) produces
d=+0.227 on Gemma 3 4B with MS MARCO. Four open questions remain:

1. Which individual layers carry the signal? Smooth gradient or concentrated?
2. Does this generalize to SQuAD v2? (short extractive QA, median 114 words)
3. Does prefix content matter under layer selectivity? (static_fact vs random vs oracle)
4. What makes early-layer values different? (L2 norms, cosine sim, delta magnitudes)

Part 1+4: Individual Layer Contribution Map + Value Features (MS MARCO, N=300)
Part 2:   Cross-Dataset — SQuAD v2 (N=400)
Part 3:   Prefix Content × Layer Selectivity (MS MARCO, N=300)
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
# Exp 24: Gemma Layer-Selective Mechanism Deep Dive

## Motivation

Exp 21 confirmed that **layer-selective value contamination** (layers 0-15, cutoff=16) produces
d=**+0.227** on Gemma 3 4B with MS MARCO, and is more length-robust than Mistral's full priming
(still d=+0.173 at 512 tokens, vs Mistral dropping to +0.034). The layer boundary sweep confirmed
cutoff=16 is optimal.

**Four open questions:**

### Q1: Which individual layers carry the signal?
We know layers 0-15 collectively help and 16-33 collectively hurt, but which specific layers
matter most? Is it a smooth gradient or concentrated in a few layers?

### Q2: Does this generalize to SQuAD v2?
MS MARCO-specific effects are a known risk. SQuAD v2 has short extractive QA passages
(median 114 words, 92% under 200 words) — ideal for Gemma's sliding window constraint.

### Q3: Does prefix content matter under layer selectivity?
Exp 16 showed static_fact, random, and oracle all fail with full-cache replacement on Gemma.
But with layer-selective values (the method that works), does prefix content make a difference?

### Q4: What makes early-layer values different?
L2 norms, cosine similarity between bare/primed values, and delta magnitudes per layer can
reveal why early layers carry the useful signal.

## Design

| Part | Data | N | Conditions | Fwd/q | Score/q |
|------|------|---|------------|-------|---------|
| 1+4 | MS MARCO | 300 | 34 single-layer + bare + features | 2 | 35 |
| 2 | SQuAD v2 | 400 | bare, values_all, values_cutoff_16 | 2 | 3 |
| 3 | MS MARCO | 300 | bare + 3 prefix types at cutoff=16 | 4 | 4 |

## Reference Values

| Source | Condition | d |
|--------|-----------|---|
| Exp 19 | values_only (all 34 layers) | +0.056 |
| Exp 19 | values_early_layers (0-16) | +0.211 |
| Exp 21 | values_early_layers (0-15) @ original | +0.227 |
| Exp 21 | cutoff sweep best (cutoff=16) | +0.161 |
| Exp 16 | static_fact full-cache | -0.031 (ns) |
| Exp 16 | random full-cache | -0.109 (***) |""")))

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

RESULTS_DIR = Path("results/exp24")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_P1_PATH = RESULTS_DIR / "checkpoint_part1.json"
CHECKPOINT_P2_PATH = RESULTS_DIR / "checkpoint_part2.json"
CHECKPOINT_P3_PATH = RESULTS_DIR / "checkpoint_part3.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"
CSV_P1_PATH = RESULTS_DIR / "part1_layer_map.csv"
CSV_P2_PATH = RESULTS_DIR / "part2_squad.csv"
CSV_P3_PATH = RESULTS_DIR / "part3_prefix_content.csv"
CSV_P4_PATH = RESULTS_DIR / "part4_value_features.csv"

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
NUM_LAYERS = text_config.num_hidden_layers
print(f"\\nModel loaded successfully.")
print(f"  Model class: {type(model).__name__}")
print(f"  Text config class: {type(text_config).__name__}")
print(f"  Hidden size: {text_config.hidden_size}")
print(f"  Num layers: {NUM_LAYERS}")
print(f"  Num attention heads: {text_config.num_attention_heads}")
print(f"  Num KV heads: {text_config.num_key_value_heads}")
print(f"  Head dim: {_get_head_dim(model.config)}")
print(f"  BOS token ID: {tokenizer.bos_token_id}")

# Per-layer RoPE diagnostics
thetas = set()
for layer_idx in range(NUM_LAYERS):
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
    build_hybrid_cache,
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
N_PART1 = 300  # Part 1+4: individual layer map + features
N_PART2 = 400  # Part 2: SQuAD v2
N_PART3 = 300  # Part 3: prefix content (same queries as Part 1)
MAX_PASSAGE_WORDS = 300
CHECKPOINT_EVERY = 25
CUTOFF = 16  # layers 0-15

# Reference values
EXP19_REF = {
    'values_only_d': 0.056,
    'values_early_layers_d': 0.211,  # layers 0-16 (17 layers)
}
EXP21_REF = {
    'values_early_layers_d': 0.227,  # layers 0-15 (16 layers)
    'cutoff_sweep_best_d': 0.161,    # cutoff=16 on N=200
}
EXP16_REF = {
    'sf_trunc_full_d': -0.031,
    'random_trunc_full_d': -0.109,
}

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  Num layers: {NUM_LAYERS}")
print(f"  Part 1+4: N={N_PART1}, individual layer map + features")
print(f"  Part 2: N={N_PART2}, SQuAD v2 cross-dataset")
print(f"  Part 3: N={N_PART3}, prefix content x layer selectivity (cutoff={CUTOFF})")
print(f"  Static fact prefix: '{STATIC_FACT}'")
print(f"\\nReference values:")
for ref_name, ref_dict in [('Exp 19', EXP19_REF), ('Exp 21', EXP21_REF), ('Exp 16', EXP16_REF)]:
    for k, v in ref_dict.items():
        print(f"  {ref_name} {k}: {v:+.3f}")\
""")))

# ========== Cell 4: Load MS MARCO + SQuAD v2 ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (positive, <=300w) + SQuAD v2 (has-answer, <=300w)
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 — POSITIVE PASSAGES ONLY")
print("=" * 70)

dataset_marco = load_dataset("microsoft/ms_marco", "v1.1", split="validation",
                              trust_remote_code=True)
print(f"Total items in validation: {len(dataset_marco)}")

marco_queries = []
random_passages = []  # non-eval passages for Part 3 random prefix
eval_passage_set = set()
np.random.seed(SEED)

for item in tqdm(dataset_marco, desc="Filtering MARCO"):
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
        # Collect non-eval passages for random pool
        for p in passage_texts:
            if count_words(p) <= MAX_PASSAGE_WORDS:
                random_passages.append(p)
        continue

    # Find positive passage(s)
    for i, (ptext, sel) in enumerate(zip(passage_texts, is_selected)):
        if sel == 1 and count_words(ptext) <= MAX_PASSAGE_WORDS:
            if len(marco_queries) < N_PART1 * 3:
                marco_queries.append({
                    'query': query,
                    'answer': answer,
                    'passage': ptext,
                    'word_count': count_words(ptext),
                })
                eval_passage_set.add(ptext)
                break

    # Collect non-eval passages for random pool
    for p in passage_texts:
        if p not in eval_passage_set and count_words(p) <= MAX_PASSAGE_WORDS:
            random_passages.append(p)

np.random.shuffle(marco_queries)
marco_queries = marco_queries[:N_PART1]
N_MARCO = len(marco_queries)

print(f"\\nSelected {N_MARCO} MS MARCO queries with positive passages")
print(f"Word counts: mean={np.mean([q['word_count'] for q in marco_queries]):.0f}, "
      f"min={min(q['word_count'] for q in marco_queries)}, "
      f"max={max(q['word_count'] for q in marco_queries)}")
print(f"Random passage pool: {len(random_passages):,}")

del dataset_marco
gc.collect()

# === SQuAD v2 ===
print("\\n" + "=" * 70)
print("LOADING SQuAD v2 — HAS-ANSWER, <=300 WORDS")
print("=" * 70)

ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")
print(f"Total items in SQuAD v2 validation: {len(ds_squad)}")

squad_samples = []
for item in ds_squad:
    if len(item['answers']['text']) > 0 and count_words(item['context']) <= MAX_PASSAGE_WORDS:
        squad_samples.append({
            'query': item['question'],
            'answer': item['answers']['text'][0],
            'passage': item['context'],
            'word_count': count_words(item['context']),
        })

print(f"SQuAD v2 samples with answers & <=300 words: {len(squad_samples)}")

np.random.shuffle(squad_samples)
squad_samples = squad_samples[:N_PART2]
N_SQUAD = len(squad_samples)

print(f"Selected {N_SQUAD} SQuAD v2 samples")
print(f"Word counts: mean={np.mean([q['word_count'] for q in squad_samples]):.0f}, "
      f"median={np.median([q['word_count'] for q in squad_samples]):.0f}, "
      f"min={min(q['word_count'] for q in squad_samples)}, "
      f"max={max(q['word_count'] for q in squad_samples)}")

del ds_squad
gc.collect()\
""")))

# ========== Cell 5: Tokenize prefixes ==========
cells.append(make_cell("code", s("""\
# Cell 5: Tokenize prefixes (static_fact, random pool) + condition explanations

print("=" * 70)
print("PREFIX TOKENIZATION — GEMMA 3 4B")
print("=" * 70)

# Static fact prefix
sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_ids = tokenizer(sf_str, return_tensors="pt",
                    add_special_tokens=False)['input_ids'].to(exp_config.device)

print(f"\\nStatic fact prefix: '{STATIC_FACT}'")
print(f"  Formatted: '{sf_str.strip()}'")
print(f"  Token length: {sf_ids.shape[1]}")

# Pre-select random passages for Part 3 (one per query)
np.random.seed(SEED + 1)
random_prefix_passages = []
rand_idx_pool = np.random.permutation(len(random_passages))
for i in range(N_PART3):
    rp = random_passages[rand_idx_pool[i % len(rand_idx_pool)]]
    random_prefix_passages.append(rp)
print(f"\\nRandom prefix passages pre-selected: {len(random_prefix_passages)}")
print(f"  Example (first 80 chars): '{random_prefix_passages[0][:80]}...'")

# BPE boundary check
print("\\nBPE BOUNDARY CHECK (first passage):")
example_doc = marco_queries[0]['passage']
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

print("\\n### Part 1+4: Individual Layer Contribution Map + Value Features ###")
print(f"  Data: {N_PART1} MS MARCO queries")
print("  Per query: 2 fwd passes (bare + sf_trunc primed)")
print(f"  Then for each of {NUM_LAYERS} layers individually:")
print("    - Replace ONLY that layer's values from primed into bare -> score")
print("    - Collect value features: L2 norms, cosine sim, delta norm")
print(f"  Output: {N_PART1} x {1 + NUM_LAYERS} NLL scores + {N_PART1} x {NUM_LAYERS} x 4 features")

print(f"\\n### Part 2: Cross-Dataset — SQuAD v2 ###")
print(f"  Data: {N_PART2} SQuAD v2 queries")
print("  Conditions: bare, values_all (34 layers), values_cutoff_16 (layers 0-15)")
print("  Per query: 2 fwd passes -> score 3 conditions")

print(f"\\n### Part 3: Prefix Content x Layer Selectivity ###")
print(f"  Data: {N_PART3} MS MARCO queries (same as Part 1)")
print(f"  Prefix types: static_fact, random (random MARCO passage), oracle (query text)")
print(f"  All at cutoff={CUTOFF} (layers 0-{CUTOFF-1})")
print("  Per query: 4 fwd passes (bare + 3 prefix types) -> truncate -> RoPE -> replace values -> score 4 conditions")\
""")))

# ========== Cell 6: Part 1+4 loop ==========
cells.append(make_cell("code", s("""\
# Cell 6: Part 1+4 — Individual Layer Map + Value Features (300 queries, 34 layers)

print("=" * 70)
print(f"PART 1+4: INDIVIDUAL LAYER MAP + VALUE FEATURES ({N_PART1} queries, {NUM_LAYERS} layers)")
print("=" * 70)

# Checkpoint resume
p1_results = []
p4_features = []
p1_start_idx = 0

if CHECKPOINT_P1_PATH.exists():
    with open(CHECKPOINT_P1_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in marco_queries[:N_PART1]]
    if ckpt_queries == current_queries:
        p1_results = ckpt['results']
        p4_features = ckpt.get('features', [])
        p1_start_idx = len(p1_results)
        print(f"Resuming from checkpoint: {p1_start_idx}/{N_PART1}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

t_start = time.time()

for qidx in tqdm(range(p1_start_idx, N_PART1), initial=p1_start_idx, total=N_PART1,
                  desc="Part 1+4"):
    qdata = marco_queries[qidx]
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

    # Forward pass 2: PRIMED (static_fact)
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

    # Score bare
    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    # Part 4: Value features (cheap tensor ops on existing caches)
    query_features = []
    for l in range(NUM_LAYERS):
        bare_v = _get_cache_values(bare_cache, l)[:, :, 1:, :]  # skip BOS
        primed_v = _get_cache_values(sf_trunc_cache, l)[:, :, 1:, :]

        # L2 norms (mean over positions and heads)
        bare_l2 = bare_v.float().norm(dim=-1).mean().item()
        primed_l2 = primed_v.float().norm(dim=-1).mean().item()

        # Delta
        delta_v = (primed_v.float() - bare_v.float())
        delta_norm = delta_v.norm(dim=-1).mean().item()

        # Cosine similarity (flatten heads and positions, compute per-vector)
        bare_flat = bare_v.float().reshape(-1, bare_v.shape[-1])
        primed_flat = primed_v.float().reshape(-1, primed_v.shape[-1])
        cos_sim = torch.nn.functional.cosine_similarity(bare_flat, primed_flat, dim=-1).mean().item()

        query_features.append({
            'layer': l,
            'bare_l2': bare_l2,
            'primed_l2': primed_l2,
            'delta_norm': delta_norm,
            'cosine_sim': cos_sim,
        })

    # Part 1: Individual layer scoring
    layer_nlls = {}
    for l in range(NUM_LAYERS):
        vel_cache = replace_values_at_layers(bare_cache, sf_trunc_cache, [l])
        vel_nll = score_answer_with_cache(
            deepcopy_cache(vel_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        layer_nlls[l] = vel_nll
        del vel_cache

    del bare_cache, sf_trunc_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    p1_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'doc_len': doc_len,
        'bare_nll': bare_nll,
        'layer_nlls': layer_nlls,
    })
    p4_features.append({
        'query_idx': qidx,
        'features': query_features,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_PART1 - 1:
        ckpt_data = {
            'results': p1_results,
            'features': p4_features,
            'query_texts': [q['query'] for q in marco_queries[:N_PART1]],
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
print(f"\\nPart 1+4 complete: {len(p1_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 7: Part 2 loop (SQuAD v2) ==========
cells.append(make_cell("code", s("""\
# Cell 7: Part 2 — Cross-Dataset SQuAD v2 (400 queries, 3 conditions)

print("=" * 70)
print(f"PART 2: CROSS-DATASET SQuAD v2 ({N_SQUAD} queries, 3 conditions)")
print("=" * 70)

# Checkpoint resume
p2_results = []
p2_start_idx = 0

if CHECKPOINT_P2_PATH.exists():
    with open(CHECKPOINT_P2_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in squad_samples[:N_SQUAD]]
    if ckpt_queries == current_queries:
        p2_results = ckpt['results']
        p2_start_idx = len(p2_results)
        print(f"Resuming from checkpoint: {p2_start_idx}/{N_SQUAD}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

layer_indices_all = list(range(NUM_LAYERS))
layer_indices_cutoff = list(range(CUTOFF))

t_start = time.time()

for qidx in tqdm(range(p2_start_idx, N_SQUAD), initial=p2_start_idx, total=N_SQUAD,
                  desc="Part 2"):
    qdata = squad_samples[qidx]
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

    # Forward pass 2: PRIMED (static_fact)
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

    # Score bare
    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)

    # values_all: bare keys + primed values at ALL layers
    val_all_cache = build_hybrid_cache(
        keys_source=bare_cache,
        values_source=sf_trunc_cache,
    )
    val_all_nll = score_answer_with_cache(
        deepcopy_cache(val_all_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del val_all_cache

    # values_cutoff_16: bare keys + primed values at layers 0-15
    val_cutoff_cache = replace_values_at_layers(
        bare_cache, sf_trunc_cache, layer_indices_cutoff)
    val_cutoff_nll = score_answer_with_cache(
        deepcopy_cache(val_cutoff_cache), context_len,
        query_prompt, answer_text, model, tokenizer, exp_config)
    del val_cutoff_cache

    del bare_cache, sf_trunc_cache, bare_input, primed_input
    gc.collect()
    torch.cuda.empty_cache()

    p2_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'doc_len': doc_len,
        'word_count': qdata['word_count'],
        'bare_nll': bare_nll,
        'values_all_nll': val_all_nll,
        'values_cutoff_16_nll': val_cutoff_nll,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_SQUAD - 1:
        ckpt_data = {
            'results': p2_results,
            'query_texts': [q['query'] for q in squad_samples[:N_SQUAD]],
            'completed': len(p2_results),
            'total': N_SQUAD,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_P2_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - p2_start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_SQUAD - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_SQUAD} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nPart 2 complete: {len(p2_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 8: Part 3 loop (Prefix Content × Layer Selectivity) ==========
cells.append(make_cell("code", s("""\
# Cell 8: Part 3 — Prefix Content x Layer Selectivity (300 queries, 3 prefix types at cutoff=16)

print("=" * 70)
print(f"PART 3: PREFIX CONTENT x LAYER SELECTIVITY ({N_PART3} queries, cutoff={CUTOFF})")
print("=" * 70)

# Checkpoint resume
p3_results = []
p3_start_idx = 0

if CHECKPOINT_P3_PATH.exists():
    with open(CHECKPOINT_P3_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in marco_queries[:N_PART3]]
    if ckpt_queries == current_queries:
        p3_results = ckpt['results']
        p3_start_idx = len(p3_results)
        print(f"Resuming from checkpoint: {p3_start_idx}/{N_PART3}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

layer_indices_cutoff = list(range(CUTOFF))

t_start = time.time()

for qidx in tqdm(range(p3_start_idx, N_PART3), initial=p3_start_idx, total=N_PART3,
                  desc="Part 3"):
    qdata = marco_queries[qidx]
    query_prompt = QUERY_TEMPLATE.format(query=qdata['query'])
    answer_text = ANSWER_TEMPLATE.format(answer=qdata['answer'])
    passage = qdata['passage']
    document_text = DOCUMENT_TEMPLATE.format(document=passage)

    # === Build 3 prefix strings ===
    # 1. static_fact
    prefix_sf = sf_str
    # 2. random (random MS MARCO passage)
    prefix_random = SURROGATE_PREFIX_TEMPLATE.format(surrogate=random_prefix_passages[qidx])
    # 3. oracle (query text)
    prefix_oracle = SURROGATE_PREFIX_TEMPLATE.format(surrogate=qdata['query'])

    prefixes = {
        'static_fact': prefix_sf,
        'random': prefix_random,
        'oracle': prefix_oracle,
    }

    # Matched tokenization for each prefix type
    prefix_caches = {}
    prefix_offsets = {}

    for pname, pstr in prefixes.items():
        full_text = pstr + document_text
        full_enc = tokenizer(full_text, return_tensors="pt",
                              add_special_tokens=True, padding=False, truncation=False)
        full_ids_p = full_enc['input_ids'].to(exp_config.device)

        p_prefix_enc = tokenizer(pstr, return_tensors="pt",
                                  add_special_tokens=True, padding=False, truncation=False)
        p_prefix_len = p_prefix_enc['input_ids'].shape[1]
        p_ids_only = tokenizer(pstr, return_tensors="pt",
                                add_special_tokens=False)['input_ids']
        p_offset = p_ids_only.shape[1]

        bos_id = full_ids_p[:, :1]
        doc_ids_p = full_ids_p[:, p_prefix_len:]
        doc_len_p = doc_ids_p.shape[1]

        # Forward pass: PRIMED
        p_prefix_ids = tokenizer(pstr, return_tensors="pt",
                                  add_special_tokens=False)['input_ids'].to(exp_config.device)
        primed_input_p = torch.cat([bos_id, p_prefix_ids, doc_ids_p], dim=1)
        with torch.no_grad():
            primed_out_p = model(input_ids=primed_input_p,
                                 attention_mask=torch.ones_like(primed_input_p),
                                 use_cache=True, return_dict=True)
        primed_full_p = _ensure_dynamic_cache(primed_out_p.past_key_values)
        del primed_out_p

        # Truncate + RoPE correct
        trunc_raw_p = extract_and_truncate_cache_with_bos(primed_full_p, doc_len_p)
        del primed_full_p

        trunc_cache_p = deepcopy_cache(trunc_raw_p)
        correct_rope_positions_with_bos(trunc_cache_p, p_offset, model)
        del trunc_raw_p

        prefix_caches[pname] = trunc_cache_p
        prefix_offsets[pname] = p_offset

        del full_enc, full_ids_p, p_prefix_enc, p_ids_only, primed_input_p, p_prefix_ids

    # Use static_fact's matched tokenization for bare cache
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

    # Forward pass: BARE
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

    # Score each prefix type with layer-selective values
    prefix_nlls = {}
    for pname in prefixes:
        vel_cache = replace_values_at_layers(
            bare_cache, prefix_caches[pname], layer_indices_cutoff)
        vel_nll = score_answer_with_cache(
            deepcopy_cache(vel_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        prefix_nlls[pname] = vel_nll
        del vel_cache

    del bare_cache, bare_input, prefix_caches
    gc.collect()
    torch.cuda.empty_cache()

    p3_results.append({
        'query_idx': qidx,
        'query': qdata['query'],
        'doc_len': doc_len,
        'bare_nll': bare_nll,
        'static_fact_nll': prefix_nlls['static_fact'],
        'random_nll': prefix_nlls['random'],
        'oracle_nll': prefix_nlls['oracle'],
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N_PART3 - 1:
        ckpt_data = {
            'results': p3_results,
            'query_texts': [q['query'] for q in marco_queries[:N_PART3]],
            'completed': len(p3_results),
            'total': N_PART3,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_P3_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - p3_start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N_PART3 - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N_PART3} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nPart 3 complete: {len(p3_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 9: Part 1 Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Part 1 Analysis — per-layer d table, cumulative d, critical layers

import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("PART 1 ANALYSIS: INDIVIDUAL LAYER CONTRIBUTION MAP")
print("=" * 70)

# Collect per-layer deltas
layer_deltas = {l: [] for l in range(NUM_LAYERS)}
bare_nlls_p1 = []

for r in p1_results:
    bare_nll = r['bare_nll']
    bare_nlls_p1.append(bare_nll)
    for l in range(NUM_LAYERS):
        delta = bare_nll - r['layer_nlls'][str(l)]
        layer_deltas[l].append(delta)

# Per-layer Cohen's d
print(f"\\n{'Layer':<8} {'Mean D':>10} {'d':>8} {'Win%':>7} {'p':>12} {'sig':>5}")
print("-" * 55)

layer_analysis = {}
for l in range(NUM_LAYERS):
    delta = np.array(layer_deltas[l])
    valid = np.isfinite(delta)
    delta = delta[valid]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{l:<8} {np.mean(delta):>+10.4f} {d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
    layer_analysis[l] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }

# Rank layers by effect size
layer_ranking = sorted(layer_analysis.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
print(f"\\nTop 10 layers by Cohen's d:")
for rank, (l, a) in enumerate(layer_ranking[:10], 1):
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  #{rank}: layer {l:>2}  d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  {sig}")

print(f"\\nBottom 5 layers (most harmful):")
for rank, (l, a) in enumerate(layer_ranking[-5:], 1):
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  #{rank}: layer {l:>2}  d={a['cohens_d']:>+.3f}  win={a['win_pct']:.0f}%  {sig}")

# Cumulative d: add layers in order of individual d (greedy)
print(f"\\n{'Cum Layers':<12} {'Layers Added':>15} {'d':>8}")
print("-" * 40)

ranked_layers = [l for l, _ in layer_ranking]
cum_deltas = np.zeros(len(p1_results))
cum_analysis = []

for step, l in enumerate(ranked_layers):
    # Can't actually compute cumulative scoring without re-running model
    # Instead, approximate: sum of individual deltas as proxy
    cum_deltas = cum_deltas + np.array(layer_deltas[l])
    cum_d = cohens_d(cum_deltas)
    cum_analysis.append({
        'n_layers': step + 1,
        'layer_added': l,
        'cumulative_d_approx': float(cum_d),
    })
    if step < 20 or step == NUM_LAYERS - 1:
        print(f"{step+1:<12} layer {l:>3}         {cum_d:>+8.3f}")

# Early vs late summary
early_ds = [layer_analysis[l]['cohens_d'] for l in range(CUTOFF)]
late_ds = [layer_analysis[l]['cohens_d'] for l in range(CUTOFF, NUM_LAYERS)]
print(f"\\nEarly layers (0-{CUTOFF-1}): mean d = {np.mean(early_ds):+.3f}, "
      f"positive = {sum(1 for d in early_ds if d > 0)}/{len(early_ds)}")
print(f"Late layers ({CUTOFF}-{NUM_LAYERS-1}):  mean d = {np.mean(late_ds):+.3f}, "
      f"positive = {sum(1 for d in late_ds if d > 0)}/{len(late_ds)}")\
""")))

# ========== Cell 10: Part 2+3 Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 10: Part 2+3 Analysis — cross-dataset + prefix content results

print("=" * 70)
print("PART 2 ANALYSIS: CROSS-DATASET (SQuAD v2)")
print("=" * 70)

# SQuAD v2 results
p2_bare = np.array([r['bare_nll'] for r in p2_results])
p2_val_all = np.array([r['values_all_nll'] for r in p2_results])
p2_val_cutoff = np.array([r['values_cutoff_16_nll'] for r in p2_results])

valid_p2 = (p2_bare != 0) & np.isfinite(p2_bare) & (p2_val_all != 0) & (p2_val_cutoff != 0)
p2_b = p2_bare[valid_p2]
p2_va = p2_val_all[valid_p2]
p2_vc = p2_val_cutoff[valid_p2]

p2_analysis = {}
for name, arr in [('values_all', p2_va), ('values_cutoff_16', p2_vc)]:
    delta = p2_b - arr
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    p2_analysis[name] = {
        'n_valid': int(np.sum(valid_p2)),
        'mean_bare': float(np.mean(p2_b)),
        'mean_nll': float(np.mean(arr)),
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }
    print(f"  {name:<20} d={d:>+.3f}  win={win:.1f}%  p={p_val:.2e}  {sig}")

# MS MARCO comparison (from Part 1: values at all layers vs cutoff)
# We can compute values_all and values_cutoff from Part 1 individual layer results
# by summing all layer deltas (approx)
p1_bare_arr = np.array(bare_nlls_p1)
print(f"\\nCross-dataset comparison:")
print(f"  {'Dataset':<15} {'Condition':<20} {'d':>8} {'Win%':>7} {'p':>12}")
print("  " + "-" * 68)

# Exp 21 MS MARCO reference for comparison
print(f"  {'MS MARCO':<15} {'values_cutoff_16':<20} {'(Exp 21):':>8} {'+0.227':>7} {'—':>12}")
for name, a in p2_analysis.items():
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  {'SQuAD v2':<15} {name:<20} {a['cohens_d']:>+8.3f} {a['win_pct']:>6.1f}% {a['p_value']:>12.2e}  {sig}")

# Part 3 Analysis
print("\\n" + "=" * 70)
print("PART 3 ANALYSIS: PREFIX CONTENT x LAYER SELECTIVITY")
print("=" * 70)

p3_bare = np.array([r['bare_nll'] for r in p3_results])
p3_sf = np.array([r['static_fact_nll'] for r in p3_results])
p3_rand = np.array([r['random_nll'] for r in p3_results])
p3_oracle = np.array([r['oracle_nll'] for r in p3_results])

valid_p3 = (p3_bare != 0) & np.isfinite(p3_bare) & (p3_sf != 0) & (p3_rand != 0) & (p3_oracle != 0)

p3_analysis = {}
print(f"\\n{'Prefix':<15} {'Mean NLL':>10} {'d vs bare':>10} {'Win%':>7} {'p':>12} {'sig':>5}")
print("-" * 62)

for pname, parr in [('static_fact', p3_sf), ('random', p3_rand), ('oracle', p3_oracle)]:
    b = p3_bare[valid_p3]
    a = parr[valid_p3]
    delta = b - a
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{pname:<15} {np.mean(a):>10.4f} {d:>+10.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}")
    p3_analysis[pname] = {
        'n_valid': int(np.sum(valid_p3)),
        'mean_nll': float(np.mean(a)),
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_pct': float(win),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    }

# Pairwise comparisons
print(f"\\nPairwise prefix comparisons (at cutoff={CUTOFF}):")
for n1, a1, n2, a2 in [
    ('static_fact', p3_sf, 'random', p3_rand),
    ('static_fact', p3_sf, 'oracle', p3_oracle),
    ('oracle', p3_oracle, 'random', p3_rand),
]:
    delta = a2[valid_p3] - a1[valid_p3]  # positive = n1 better
    d = cohens_d(delta)
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"  {n1} vs {n2}: d={d:+.3f}, p={p_val:.2e} {sig}")

# Exp 16 comparison (full-cache)
print(f"\\nComparison with Exp 16 (full-cache replacement on Gemma):")
vel_header = f'VEL@{CUTOFF} d'
print(f"  {'Prefix':<15} {'Full-cache d':>13} {vel_header:>13} {'Gain':>8}")
print("  " + "-" * 55)
exp16_map = {'static_fact': -0.031, 'random': -0.109}
for pname in ['static_fact', 'random']:
    if pname in exp16_map:
        full_d = exp16_map[pname]
        vel_d = p3_analysis[pname]['cohens_d']
        print(f"  {pname:<15} {full_d:>+13.3f} {vel_d:>+13.3f} {vel_d - full_d:>+8.3f}")\
""")))

# ========== Cell 11: Part 4 Analysis (Value Features) ==========
cells.append(make_cell("code", s("""\
# Cell 11: Part 4 Analysis — Value features + correlation with Part 1 d

print("=" * 70)
print("PART 4 ANALYSIS: VALUE FEATURE ANALYSIS")
print("=" * 70)

# Aggregate features per layer
feat_agg = {l: {'bare_l2': [], 'primed_l2': [], 'delta_norm': [], 'cosine_sim': []}
            for l in range(NUM_LAYERS)}

for qf in p4_features:
    for feat in qf['features']:
        l = feat['layer']
        feat_agg[l]['bare_l2'].append(feat['bare_l2'])
        feat_agg[l]['primed_l2'].append(feat['primed_l2'])
        feat_agg[l]['delta_norm'].append(feat['delta_norm'])
        feat_agg[l]['cosine_sim'].append(feat['cosine_sim'])

# Summary table
print(f"\\n{'Layer':<8} {'Bare L2':>10} {'Primed L2':>10} {'Delta Norm':>12} {'Cosine Sim':>12} {'d (Part1)':>10}")
print("-" * 68)

feat_summary = {}
layer_ds = []
delta_norms = []
cosine_sims = []

for l in range(NUM_LAYERS):
    mean_bare_l2 = np.mean(feat_agg[l]['bare_l2'])
    mean_primed_l2 = np.mean(feat_agg[l]['primed_l2'])
    mean_delta_norm = np.mean(feat_agg[l]['delta_norm'])
    mean_cosine_sim = np.mean(feat_agg[l]['cosine_sim'])
    d_val = layer_analysis[l]['cohens_d']
    print(f"{l:<8} {mean_bare_l2:>10.3f} {mean_primed_l2:>10.3f} {mean_delta_norm:>12.4f} "
          f"{mean_cosine_sim:>12.4f} {d_val:>+10.3f}")
    feat_summary[l] = {
        'bare_l2': float(mean_bare_l2),
        'primed_l2': float(mean_primed_l2),
        'delta_norm': float(mean_delta_norm),
        'cosine_sim': float(mean_cosine_sim),
    }
    layer_ds.append(d_val)
    delta_norms.append(mean_delta_norm)
    cosine_sims.append(mean_cosine_sim)

layer_ds = np.array(layer_ds)
delta_norms = np.array(delta_norms)
cosine_sims = np.array(cosine_sims)

# Correlations
print(f"\\nCorrelation: per-layer d vs delta_norm")
r_dn, p_dn = stats.pearsonr(layer_ds, delta_norms)
print(f"  Pearson r={r_dn:+.3f}, p={p_dn:.2e}")

print(f"\\nCorrelation: per-layer d vs cosine_sim")
r_cs, p_cs = stats.pearsonr(layer_ds, cosine_sims)
print(f"  Pearson r={r_cs:+.3f}, p={p_cs:.2e}")

# Early vs late feature comparison
print(f"\\nEarly (0-{CUTOFF-1}) vs Late ({CUTOFF}-{NUM_LAYERS-1}) features:")
early_delta = np.mean([feat_summary[l]['delta_norm'] for l in range(CUTOFF)])
late_delta = np.mean([feat_summary[l]['delta_norm'] for l in range(CUTOFF, NUM_LAYERS)])
early_cos = np.mean([feat_summary[l]['cosine_sim'] for l in range(CUTOFF)])
late_cos = np.mean([feat_summary[l]['cosine_sim'] for l in range(CUTOFF, NUM_LAYERS)])
print(f"  Mean delta_norm: early={early_delta:.4f}, late={late_delta:.4f}, ratio={early_delta/late_delta:.2f}" if late_delta > 0 else f"  Mean delta_norm: early={early_delta:.4f}, late={late_delta:.4f}")
print(f"  Mean cosine_sim: early={early_cos:.4f}, late={late_cos:.4f}")\
""")))

# ========== Cell 12: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 12: Plots — 6-panel summary figure (2x3)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# ---- Panel 1 (top-left): Per-layer d bar chart ----
ax = axes[0, 0]
ds_per_layer = [layer_analysis[l]['cohens_d'] for l in range(NUM_LAYERS)]
colors_layer = []
for l in range(NUM_LAYERS):
    p_val = layer_analysis[l]['p_value']
    d_val = layer_analysis[l]['cohens_d']
    if p_val < 0.001:
        colors_layer.append('#2ca02c' if d_val > 0 else '#d62728')
    elif p_val < 0.05:
        colors_layer.append('#98df8a' if d_val > 0 else '#ff9896')
    else:
        colors_layer.append('#cccccc')

ax.bar(range(NUM_LAYERS), ds_per_layer, color=colors_layer, edgecolor='black', linewidth=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=CUTOFF - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Cutoff={CUTOFF}')
ax.set_xlabel('Layer')
ax.set_ylabel("Cohen's d (single layer)")
ax.set_title("Part 1: Per-Layer Effect Size")
ax.legend(fontsize=8)

# ---- Panel 2 (top-center): Cumulative d ----
ax = axes[0, 1]
cum_ds = [ca['cumulative_d_approx'] for ca in cum_analysis]
cum_labels = [ca['layer_added'] for ca in cum_analysis]
ax.plot(range(1, NUM_LAYERS + 1), cum_ds, marker='.', markersize=3, linewidth=1.5, color='#1f77b4')
ax.axhline(y=EXP21_REF['values_early_layers_d'], color='#9467bd', linestyle='--', linewidth=1,
           label=f"Exp 21 VEL d={EXP21_REF['values_early_layers_d']:+.3f}")
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=CUTOFF, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Annotate peak
peak_idx = int(np.argmax(cum_ds))
ax.annotate(f"Peak: {cum_ds[peak_idx]:+.3f}\\n({peak_idx+1} layers)",
            (peak_idx + 1, cum_ds[peak_idx]),
            textcoords='offset points', xytext=(15, -10),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('Number of layers added (ranked by d)')
ax.set_ylabel("Cumulative d (approx)")
ax.set_title("Part 1: Cumulative d (greedy)")
ax.legend(fontsize=8)

# ---- Panel 3 (top-right): Cosine similarity + delta norm by layer ----
ax = axes[0, 2]
ax2 = ax.twinx()

x = range(NUM_LAYERS)
ln1 = ax.plot(x, cosine_sims, 'b-', linewidth=1.5, label='Cosine sim')
ln2 = ax2.plot(x, delta_norms, 'r-', linewidth=1.5, label='Delta norm')

ax.set_xlabel('Layer')
ax.set_ylabel('Cosine similarity', color='b')
ax2.set_ylabel('Delta norm', color='r')
ax.axvline(x=CUTOFF - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title("Part 4: Value Features by Layer")

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize=8, loc='center right')

# ---- Panel 4 (bottom-left): SQuAD v2 vs MS MARCO ----
ax = axes[1, 0]

# Grouped bars: dataset x condition
conditions = ['values_all', 'values_cutoff_16']
datasets = ['MS MARCO', 'SQuAD v2']

# MS MARCO reference values
marco_ds = [EXP19_REF['values_only_d'], EXP21_REF['values_early_layers_d']]
squad_ds = [p2_analysis['values_all']['cohens_d'], p2_analysis['values_cutoff_16']['cohens_d']]

x_pos = np.arange(len(conditions))
w = 0.35
bars1 = ax.bar(x_pos - w/2, marco_ds, w, color='#1f77b4', edgecolor='black', linewidth=0.5,
               label='MS MARCO')
bars2 = ax.bar(x_pos + w/2, squad_ds, w, color='#ff7f0e', edgecolor='black', linewidth=0.5,
               label='SQuAD v2')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xticks(x_pos)
ax.set_xticklabels(['values_all\\n(34 layers)', f'values_cutoff_{CUTOFF}\\n(layers 0-{CUTOFF-1})'])
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Part 2: Cross-Dataset Comparison")
ax.legend(fontsize=8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005 if h >= 0 else h - 0.015,
                f"{h:+.3f}", ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

# ---- Panel 5 (bottom-center): Prefix content bars ----
ax = axes[1, 1]

prefix_names = ['static_fact', 'random', 'oracle']
prefix_ds = [p3_analysis[pn]['cohens_d'] for pn in prefix_names]
prefix_colors = ['#2ca02c', '#d62728', '#9467bd']

bars = ax.bar(range(len(prefix_names)), prefix_ds, color=prefix_colors,
              edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xticks(range(len(prefix_names)))
ax.set_xticklabels(prefix_names)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title(f"Part 3: Prefix Content (cutoff={CUTOFF})")

for i, (d_val, pn) in enumerate(zip(prefix_ds, prefix_names)):
    p_val = p3_analysis[pn]['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.text(i, d_val + 0.005 if d_val >= 0 else d_val - 0.015,
            f"{d_val:+.3f} {sig}", ha='center',
            va='bottom' if d_val >= 0 else 'top', fontsize=9)

# ---- Panel 6 (bottom-right): Scatter per-layer d vs delta norm ----
ax = axes[1, 2]

ax.scatter(delta_norms, layer_ds, c=range(NUM_LAYERS), cmap='viridis', s=60,
           edgecolors='black', linewidths=0.5, zorder=3)

# Color bar for layer index
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, NUM_LAYERS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Layer index')

# Fit line
z = np.polyfit(delta_norms, layer_ds, 1)
x_fit = np.linspace(delta_norms.min(), delta_norms.max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Mean delta norm (primed - bare values)')
ax.set_ylabel("Cohen's d (single layer)")
ax.set_title(f"Part 4: d vs Delta Norm (r={r_dn:+.3f})")
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Annotate a few key layers
for l in [0, CUTOFF-1, CUTOFF, NUM_LAYERS-1]:
    ax.annotate(f"L{l}", (delta_norms[l], layer_ds[l]),
                textcoords='offset points', xytext=(5, 5), fontsize=7)

plt.suptitle('Exp 24: Gemma Layer-Selective Mechanism Deep Dive', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 13: Save results.json + CSVs ==========
cells.append(make_cell("code", s("""\
# Cell 13: Save results.json + CSVs
import csv

# --- Part 1 CSV (per query x per layer) ---
with open(CSV_P1_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'layer', 'bare_nll', 'layer_nll', 'delta_nll', 'cohens_d'])
    writer.writeheader()
    for r in p1_results:
        for l in range(NUM_LAYERS):
            layer_nll = r['layer_nlls'][str(l)]
            writer.writerow({
                'query_idx': r['query_idx'],
                'layer': l,
                'bare_nll': r['bare_nll'],
                'layer_nll': layer_nll,
                'delta_nll': r['bare_nll'] - layer_nll,
                'cohens_d': layer_analysis[l]['cohens_d'],
            })
print(f"Part 1 CSV saved: {CSV_P1_PATH}")

# --- Part 2 CSV ---
with open(CSV_P2_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'word_count', 'bare_nll', 'values_all_nll', 'values_cutoff_16_nll'])
    writer.writeheader()
    for r in p2_results:
        writer.writerow({
            'query_idx': r['query_idx'],
            'word_count': r['word_count'],
            'bare_nll': r['bare_nll'],
            'values_all_nll': r['values_all_nll'],
            'values_cutoff_16_nll': r['values_cutoff_16_nll'],
        })
print(f"Part 2 CSV saved: {CSV_P2_PATH}")

# --- Part 3 CSV ---
with open(CSV_P3_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'bare_nll', 'static_fact_nll', 'random_nll', 'oracle_nll'])
    writer.writeheader()
    for r in p3_results:
        writer.writerow({
            'query_idx': r['query_idx'],
            'bare_nll': r['bare_nll'],
            'static_fact_nll': r['static_fact_nll'],
            'random_nll': r['random_nll'],
            'oracle_nll': r['oracle_nll'],
        })
print(f"Part 3 CSV saved: {CSV_P3_PATH}")

# --- Part 4 CSV ---
with open(CSV_P4_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query_idx', 'layer', 'bare_l2', 'primed_l2', 'delta_norm', 'cosine_sim'])
    writer.writeheader()
    for qf in p4_features:
        for feat in qf['features']:
            writer.writerow({
                'query_idx': qf['query_idx'],
                'layer': feat['layer'],
                'bare_l2': feat['bare_l2'],
                'primed_l2': feat['primed_l2'],
                'delta_norm': feat['delta_norm'],
                'cosine_sim': feat['cosine_sim'],
            })
print(f"Part 4 CSV saved: {CSV_P4_PATH}")

# --- Combined results.json ---
final = {
    'experiment': 'exp24_gemma_layer_mechanism_deep_dive',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'dataset_marco': 'MS MARCO v1.1 validation',
        'dataset_squad': 'SQuAD v2 validation',
        'max_passage_words': MAX_PASSAGE_WORDS,
        'part1': {'n_queries': N_PART1, 'n_layers': NUM_LAYERS},
        'part2': {'n_queries': N_SQUAD, 'conditions': ['bare', 'values_all', 'values_cutoff_16']},
        'part3': {'n_queries': N_PART3, 'cutoff': CUTOFF, 'prefix_types': ['static_fact', 'random', 'oracle']},
        'part4': {'n_queries': N_PART1, 'features': ['bare_l2', 'primed_l2', 'delta_norm', 'cosine_sim']},
    },
    'gemma_architecture': {
        'hidden_size': text_config.hidden_size,
        'num_layers': NUM_LAYERS,
        'num_attention_heads': text_config.num_attention_heads,
        'num_kv_heads': text_config.num_key_value_heads,
        'head_dim': _get_head_dim(model.config),
        'rope_thetas': sorted(list(thetas)),
    },
    'part1_layer_analysis': {str(k): v for k, v in layer_analysis.items()},
    'part1_layer_ranking': [{'layer': l, **a} for l, a in layer_ranking],
    'part1_cumulative': cum_analysis,
    'part1_early_vs_late': {
        'early_mean_d': float(np.mean(early_ds)),
        'early_positive_count': sum(1 for d in early_ds if d > 0),
        'late_mean_d': float(np.mean(late_ds)),
        'late_positive_count': sum(1 for d in late_ds if d > 0),
    },
    'part2_analysis': p2_analysis,
    'part3_analysis': p3_analysis,
    'part4_feature_summary': {str(k): v for k, v in feat_summary.items()},
    'part4_correlations': {
        'd_vs_delta_norm': {'r': float(r_dn), 'p': float(p_dn)},
        'd_vs_cosine_sim': {'r': float(r_cs), 'p': float(p_cs)},
    },
    'reference_values': {
        'exp19_gemma': EXP19_REF,
        'exp21_gemma': EXP21_REF,
        'exp16_gemma': EXP16_REF,
    },
    'part1_per_query': p1_results,
    'part2_per_query': p2_results,
    'part3_per_query': p3_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"\\nResults saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Final summary
print("\\n" + "=" * 70)
print("SUMMARY — Exp 24: Gemma Layer-Selective Mechanism Deep Dive")
print("=" * 70)
print(f"Model: Gemma 3 4B ({NUM_LAYERS} layers, head_dim={_get_head_dim(model.config)}, bfloat16)")

print(f"\\nPart 1: Individual Layer Map ({N_PART1} queries)")
top5_str = ', '.join(f'L{l}(d={a["cohens_d"]:+.3f})' for l, a in layer_ranking[:5])
print(f"  Top 5 layers: {top5_str}")
print(f"  Early (0-{CUTOFF-1}) mean d: {np.mean(early_ds):+.3f}")
print(f"  Late ({CUTOFF}-{NUM_LAYERS-1}) mean d: {np.mean(late_ds):+.3f}")

print(f"\\nPart 2: SQuAD v2 Cross-Dataset ({N_SQUAD} queries)")
for name, a in p2_analysis.items():
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  {name}: d={a['cohens_d']:+.3f}, win={a['win_pct']:.0f}%, {sig}")

print(f"\\nPart 3: Prefix Content (cutoff={CUTOFF}, {N_PART3} queries)")
for pn in ['static_fact', 'random', 'oracle']:
    a = p3_analysis[pn]
    sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
    print(f"  {pn}: d={a['cohens_d']:+.3f}, win={a['win_pct']:.0f}%, {sig}")

print(f"\\nPart 4: Value Features")
print(f"  d vs delta_norm: r={r_dn:+.3f} (p={p_dn:.2e})")
print(f"  d vs cosine_sim: r={r_cs:+.3f} (p={p_cs:.2e})")

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

output_path = "/home/jupyter/research/directed_kvcache_v2/24_gemma_layer_mechanism.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
