#!/usr/bin/env python3
"""Build script for 09_values_deep_dive.ipynb"""

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
# Exp 09: Values Deep Dive

## Motivation

Exp 08 showed value vectors carry 100% of the priming effect (d=+0.275) while keys
contribute nothing (d=-0.009). This experiment investigates the values-only mechanism
across 5 directions:

1. **Layer-wise isolation** — Which layers carry the signal?
2. **Prefix type variation** — Does the prefix content matter in values-only mode?
3. **Interpolation** — How does blending bare/primed values at different ratios affect quality?
4. **Cross-document transfer** — Do primed values generalize across documents?
5. **Positional isolation** — Which token positions carry the signal?

## 17 Experimental Conditions

| # | Name | Category | Construction |
|---|------|----------|-------------|
| 1 | `bare` | Baseline | `[BOS][doc]` |
| 2 | `full_llm_kw` | Baseline | LLM-kw truncated (keys+values) |
| 3 | `values_only_llm_kw` | Baseline | Values from LLM-kw, keys from bare |
| 4 | `values_layers_0_7` | Layer-wise | Primed values in layers 0-7 only |
| 5 | `values_layers_8_15` | Layer-wise | Primed values in layers 8-15 only |
| 6 | `values_layers_16_23` | Layer-wise | Primed values in layers 16-23 only |
| 7 | `values_layers_24_31` | Layer-wise | Primed values in layers 24-31 only |
| 8 | `values_only_static_fact` | Prefix type | Values from "What are the key facts?" prefix |
| 9 | `values_only_oracle` | Prefix type | Values from actual query prefix |
| 10 | `values_only_random` | Prefix type | Values from random token prefix |
| 11 | `values_interp_025` | Interpolation | 25% primed + 75% bare values |
| 12 | `values_interp_050` | Interpolation | 50/50 blend |
| 13 | `values_interp_075` | Interpolation | 75% primed + 25% bare values |
| 14 | `values_cross_doc` | Cross-doc | Primed values from previous sample's document |
| 15 | `values_first_quarter` | Positional | Primed values in first 25% of doc positions |
| 16 | `values_last_quarter` | Positional | Primed values in last 25% of doc positions |
| 17 | `values_middle_half` | Positional | Primed values in middle 50% of doc positions |

## 10 Primary Comparisons (Bonferroni alpha = 0.005)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | values_layers_0_7 vs bare | Early layers carry signal? |
| C2 | values_layers_8_15 vs bare | Early-mid layers? |
| C3 | values_layers_16_23 vs bare | Mid-late layers? |
| C4 | values_layers_24_31 vs bare | Late layers? |
| C5 | values_only_static_fact vs values_only_llm_kw | Static advantage in values? |
| C6 | values_only_random vs bare | Random-primed values help? |
| C7 | values_interp_050 vs bare | 50% blend helps? |
| C8 | values_cross_doc vs bare | Wrong-doc values help? |
| C9 | values_cross_doc vs values_only_llm_kw | Same-doc vs cross-doc? |
| C10 | values_first_quarter vs values_last_quarter | Beginning vs end positions? |""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp09")
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

# ========== Cell 3: Imports + config + constants ==========
cells.append(make_cell("code", s("""\
# Cell 3: Imports + config + constants
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    build_hybrid_cache,
    replace_values_at_layers,
    interpolate_values,
    replace_values_at_positions,
    build_cross_doc_cache,
    deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    score_answer_with_cache,
    _get_cache_keys,
    _get_cache_values,
    _set_cache_values,
    _ensure_dynamic_cache,
)
from lib.data import load_ms_marco, load_evaluation_samples
from lib.analysis import cohens_d
from lib.surrogate import generate_all_5_surrogates, STATIC_SURROGATE_QUERIES
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    min_passage_words=20,
    max_passage_words=500,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

N_EVAL = 1000
N_COMPARISONS = 10
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
CHECKPOINT_EVERY = 50

STATIC_FACTUAL_PHRASE = STATIC_SURROGATE_QUERIES['static_factual']['query']

LAYER_GROUPS = {
    'layers_0_7': list(range(0, 8)),
    'layers_8_15': list(range(8, 16)),
    'layers_16_23': list(range(16, 24)),
    'layers_24_31': list(range(24, 32)),
}

INTERP_ALPHAS = [0.25, 0.50, 0.75]

CONDITION_NAMES = [
    'bare', 'full_llm_kw', 'values_only_llm_kw',
    'values_layers_0_7', 'values_layers_8_15', 'values_layers_16_23', 'values_layers_24_31',
    'values_only_static_fact', 'values_only_oracle', 'values_only_random',
    'values_interp_025', 'values_interp_050', 'values_interp_075',
    'values_cross_doc',
    'values_first_quarter', 'values_last_quarter', 'values_middle_half',
]

print("Config ready")
print(f"  num_samples pool: {config.num_samples}")
print(f"  eval samples: {N_EVAL}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: {len(CONDITION_NAMES)}")
print(f"  static_factual_phrase: '{STATIC_FACTUAL_PHRASE}'")
print(f"  layer_groups: {list(LAYER_GROUPS.keys())}")
print(f"  interp_alphas: {INTERP_ALPHAS}")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (1000 samples)
dataset = load_ms_marco(config)

np.random.seed(SEED)
all_samples = load_evaluation_samples(dataset, config, require_answer=True)

samples = all_samples[:N_EVAL]
N = len(samples)
print(f"Loaded {len(all_samples)} candidates, using first {N} for evaluation")
print(f"Example passage ({len(samples[0]['passage'].split())} words): {samples[0]['passage'][:100]}...")
print(f"Example query: {samples[0]['query']}")
print(f"Example answer: {samples[0]['answer']}")\
""")))

# ========== Cell 5: Generate LLM keyword surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 5: Generate LLM keyword surrogates (fresh, independent from Exp 08)
print("=" * 70)
print("PHASE 1: LLM SURROGATE GENERATION (keyword only)")
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
    t_start = time.time()
    for idx in tqdm(range(start_idx_gen, N), initial=start_idx_gen, total=N,
                     desc="Keyword surrogates"):
        passage = samples[idx]['passage']
        try:
            s5 = generate_all_5_surrogates(passage, model, tokenizer, config)
            kw = s5.get('keyword_query', '')
        except Exception as e:
            print(f"  WARNING: Generation failed for sample {idx}: {e}")
            kw = ""
        keyword_surrogates.append(kw)

        if (idx + 1) % 100 == 0 or idx == N - 1:
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
print(f"Example: '{keyword_surrogates[0]}'")\
""")))

# ========== Cell 6: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 6: Condition explanation with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

ex_kw = keyword_surrogates[0]
ex_query = samples[0]['query']

conditions_explained = [
    ("1. bare",
     "[BOS][doc]",
     "No prefix — baseline"),
    ("2. full_llm_kw",
     "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     f"Standard truncated prefix (keys+values): '{ex_kw}'"),
    ("3. values_only_llm_kw",
     "Keys from bare, values from full_llm_kw cache",
     "Values-only baseline (replicates Exp 08 finding)"),
    ("4-7. values_layers_X_Y",
     "Keys from bare, values from primed cache at specified layers only",
     "Layer-wise isolation: which layer groups carry the signal?"),
    ("8. values_only_static_fact",
     f"Prefix: '{STATIC_FACTUAL_PHRASE}' → truncate → values-only",
     "Static factual prefix — same for every document"),
    ("9. values_only_oracle",
     f"Prefix: '{ex_query}' → truncate → values-only",
     "Oracle (actual query) prefix — best possible semantic prefix"),
    ("10. values_only_random",
     "Prefix: random tokens → truncate → values-only",
     "Random token prefix — structural control for values"),
    ("11-13. values_interp_XXX",
     "v = alpha * v_primed + (1-alpha) * v_bare",
     "Value interpolation at alpha = 0.25, 0.50, 0.75"),
    ("14. values_cross_doc",
     "Values from PREVIOUS sample's primed cache",
     "Cross-document transfer — wrong-doc values"),
    ("15-17. values_first/last/middle",
     "Primed values at specific token positions only",
     "Positional isolation: first 25%, last 25%, middle 50%"),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")

print(f"\\n{'='*70}")
print("5 FORWARD PASSES PER SAMPLE:")
print("  1. Bare: [BOS][doc] → bare_cache")
print("  2. LLM-kw: [BOS][kw\\\\n][doc] → truncate+RoPE → primed_llm_cache")
print("  3. Static fact: [BOS][static_fact\\\\n][doc] → truncate+RoPE")
print("  4. Oracle: [BOS][query\\\\n][doc] → truncate+RoPE")
print("  5. Random: [BOS][random_tokens\\\\n][doc] → truncate+RoPE")
print("  Cross-doc reuses previous sample's primed_llm_cache — no extra pass")\
""")))

# ========== Cell 7: Inline helper ==========
cells.append(make_cell("code", s("""\
# Cell 7: Convenience wrapper for building primed+truncated caches

def build_primed_and_truncated(prefix_text, bos_id, doc_ids, doc_len, model, tokenizer, config):
    \"\"\"Build a primed cache: tokenize prefix, concat [BOS][prefix][doc], forward, truncate+RoPE.

    Args:
        prefix_text: Raw prefix text (will be formatted with SURROGATE_PREFIX_TEMPLATE)
        bos_id: BOS token id tensor (1, 1)
        doc_ids: Document token ids (1, doc_len)
        doc_len: Number of document tokens
        model: The language model
        tokenizer: The tokenizer
        config: ExperimentConfig

    Returns:
        (trunc_cache, prefix_token_len) where prefix_token_len includes BOS
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

print("Helper defined: build_primed_and_truncated()")\
""")))

# ========== Cell 8: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 8: Main eval loop — 17 conditions × 1000 samples
print("=" * 70)
print("PHASE 2: MAIN EVALUATION (17 conditions × 1000 samples)")
print("=" * 70)

results = []
start_idx = 0
prev_primed_cache = None
prev_doc_len = None

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in samples]
    if ckpt_queries == current_queries:
        results = ckpt['results']
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{N}")

        # Rebuild prev_primed_cache from sample start_idx - 1
        if start_idx > 0:
            print(f"Rebuilding prev_primed_cache from sample {start_idx - 1}...")
            prev_sample = samples[start_idx - 1]
            prev_passage = prev_sample['passage']
            prev_query = prev_sample['query']
            prev_kw = keyword_surrogates[start_idx - 1]

            oracle_prefix = SURROGATE_PREFIX_TEMPLATE.format(surrogate=prev_query)
            document_text = DOCUMENT_TEMPLATE.format(document=prev_passage)
            full_text = oracle_prefix + document_text

            full_enc = tokenizer(full_text, return_tensors="pt",
                                 add_special_tokens=True, padding=False, truncation=False)
            full_ids_prev = full_enc['input_ids'].to(config.device)

            oracle_prefix_enc = tokenizer(oracle_prefix, return_tensors="pt",
                                          add_special_tokens=True, padding=False, truncation=False)
            prev_prefix_len = oracle_prefix_enc['input_ids'].shape[1]

            prev_bos_id = full_ids_prev[:, :1]
            prev_doc_ids = full_ids_prev[:, prev_prefix_len:]
            prev_doc_len_rebuild = prev_doc_ids.shape[1]

            prev_primed_cache, _ = build_primed_and_truncated(
                prev_kw, prev_bos_id, prev_doc_ids, prev_doc_len_rebuild,
                model, tokenizer, config)
            prev_doc_len = prev_doc_len_rebuild
            print(f"Rebuilt prev_primed_cache (doc_len={prev_doc_len})")
            del full_ids_prev
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
    llm_kw_text = keyword_surrogates[idx]

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

    # === Forward pass 1: BARE ===
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = bare_out.past_key_values
    del bare_out

    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_cache), bare_ids.shape[1],
        query_prompt, answer_text, model, tokenizer, config)

    # === Forward pass 2: LLM-KW (primed) ===
    primed_llm_cache, llm_prefix_token_len = build_primed_and_truncated(
        llm_kw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)

    nll_full_llm_kw = score_answer_with_cache(
        deepcopy_cache(primed_llm_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)

    # === Condition 3: VALUES-ONLY LLM-KW ===
    hybrid_vals = build_hybrid_cache(bare_cache, primed_llm_cache)
    nll_values_only_llm_kw = score_answer_with_cache(
        deepcopy_cache(hybrid_vals), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del hybrid_vals

    # === Conditions 4-7: LAYER-WISE ===
    nll_layers = {}
    for group_name, layer_indices in LAYER_GROUPS.items():
        layer_cache = replace_values_at_layers(bare_cache, primed_llm_cache, layer_indices)
        nll_layers[f'values_{group_name}'] = score_answer_with_cache(
            deepcopy_cache(layer_cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del layer_cache

    # === Conditions 11-13: INTERPOLATION ===
    nll_interp = {}
    for alpha in INTERP_ALPHAS:
        interp_cache = interpolate_values(bare_cache, primed_llm_cache, alpha)
        key_name = f'values_interp_{int(alpha*100):03d}'
        nll_interp[key_name] = score_answer_with_cache(
            deepcopy_cache(interp_cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del interp_cache

    # === Conditions 15-17: POSITIONAL ===
    # Positions are within the cache: BOS=0, doc starts at 1
    first_q_end = 1 + max(1, doc_len // 4)
    last_q_start = 1 + doc_len - max(1, doc_len // 4)
    mid_start = 1 + max(1, doc_len // 4)
    mid_end = 1 + doc_len - max(1, doc_len // 4)

    pos_first = replace_values_at_positions(bare_cache, primed_llm_cache, 1, first_q_end)
    nll_first_quarter = score_answer_with_cache(
        deepcopy_cache(pos_first), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del pos_first

    pos_last = replace_values_at_positions(bare_cache, primed_llm_cache, last_q_start, 1 + doc_len)
    nll_last_quarter = score_answer_with_cache(
        deepcopy_cache(pos_last), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del pos_last

    pos_mid = replace_values_at_positions(bare_cache, primed_llm_cache, mid_start, mid_end)
    nll_middle_half = score_answer_with_cache(
        deepcopy_cache(pos_mid), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del pos_mid

    # === Condition 14: CROSS-DOC ===
    if prev_primed_cache is not None and prev_doc_len is not None:
        cross_cache = build_cross_doc_cache(bare_cache, prev_primed_cache, doc_len)
        nll_cross_doc = score_answer_with_cache(
            deepcopy_cache(cross_cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del cross_cache
    else:
        nll_cross_doc = 0.0  # First sample — no previous cache

    # Save prev_primed_cache for next sample
    prev_primed_cache = deepcopy_cache(primed_llm_cache)
    prev_doc_len = doc_len

    # === Forward pass 3: STATIC FACT ===
    static_cache, _ = build_primed_and_truncated(
        STATIC_FACTUAL_PHRASE, bos_id, doc_ids, doc_len, model, tokenizer, config)
    hybrid_static = build_hybrid_cache(bare_cache, static_cache)
    nll_static_fact = score_answer_with_cache(
        deepcopy_cache(hybrid_static), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del static_cache, hybrid_static

    # === Forward pass 4: ORACLE ===
    oracle_cache, _ = build_primed_and_truncated(
        query, bos_id, doc_ids, doc_len, model, tokenizer, config)
    hybrid_oracle = build_hybrid_cache(bare_cache, oracle_cache)
    nll_oracle = score_answer_with_cache(
        deepcopy_cache(hybrid_oracle), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del oracle_cache, hybrid_oracle

    # === Forward pass 5: RANDOM ===
    # Generate random token prefix of similar length to keyword surrogate
    n_random_tokens = max(5, len(tokenizer.encode(llm_kw_text, add_special_tokens=False)))
    random_ids = torch.randint(100, tokenizer.vocab_size - 100,
                               (n_random_tokens,), device='cpu')
    random_text = tokenizer.decode(random_ids, skip_special_tokens=True)
    random_cache, _ = build_primed_and_truncated(
        random_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    hybrid_random = build_hybrid_cache(bare_cache, random_cache)
    nll_random = score_answer_with_cache(
        deepcopy_cache(hybrid_random), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del random_cache, hybrid_random

    # --- Cleanup ---
    del bare_cache, primed_llm_cache
    torch.cuda.empty_cache()

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len': doc_len,
        'prev_doc_len': int(prev_doc_len) if prev_doc_len is not None else None,
        'passage_word_count': len(passage.split()),
        'bare': nll_bare,
        'full_llm_kw': nll_full_llm_kw,
        'values_only_llm_kw': nll_values_only_llm_kw,
        **nll_layers,
        'values_only_static_fact': nll_static_fact,
        'values_only_oracle': nll_oracle,
        'values_only_random': nll_random,
        **nll_interp,
        'values_cross_doc': nll_cross_doc,
        'values_first_quarter': nll_first_quarter,
        'values_last_quarter': nll_last_quarter,
        'values_middle_half': nll_middle_half,
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

# ========== Cell 9: Primary analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Primary analysis — NLL summary + 10 comparisons + all vs bare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — VALUES DEEP DIVE")
print("=" * 70)

# Extract arrays and filter zero NLLs (cross-doc sample 0)
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

valid = np.ones(len(results), dtype=bool)
for cname in CONDITION_NAMES:
    valid &= (cond_arrays[cname] != 0)
n_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total: {len(results)}, Valid: {n_valid}, Excluded (cross-doc sample 0): {n_excluded}")

c = {}
for cname in CONDITION_NAMES:
    c[cname] = cond_arrays[cname][valid]

# NLL summary table
print(f"\\n{'Condition':<30} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10}")
print("-" * 65)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        d_str = "—"
    else:
        d = cohens_d(c['bare'] - c[cname])
        d_str = f"{d:+.3f}"
    print(f"{cname:<30} {mean_nll:>10.4f} {std_nll:>10.4f} {d_str:>10}")

# 10 primary comparisons
print(f"\\n{'='*90}")
print(f"10 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*90}")

comparisons = [
    ('C1: layers_0_7 vs bare',
     c['bare'] - c['values_layers_0_7'],
     'Early layers carry signal?'),
    ('C2: layers_8_15 vs bare',
     c['bare'] - c['values_layers_8_15'],
     'Early-mid layers?'),
    ('C3: layers_16_23 vs bare',
     c['bare'] - c['values_layers_16_23'],
     'Mid-late layers?'),
    ('C4: layers_24_31 vs bare',
     c['bare'] - c['values_layers_24_31'],
     'Late layers?'),
    ('C5: static_fact vs llm_kw',
     c['values_only_llm_kw'] - c['values_only_static_fact'],
     'Static advantage in values?'),
    ('C6: random vs bare',
     c['bare'] - c['values_only_random'],
     'Random-primed values help?'),
    ('C7: interp_050 vs bare',
     c['bare'] - c['values_interp_050'],
     '50% blend helps?'),
    ('C8: cross_doc vs bare',
     c['bare'] - c['values_cross_doc'],
     'Wrong-doc values help?'),
    ('C9: cross_doc vs llm_kw',
     c['values_only_llm_kw'] - c['values_cross_doc'],
     'Same-doc vs cross-doc?'),
    ('C10: first_q vs last_q',
     c['values_last_quarter'] - c['values_first_quarter'],
     'Beginning vs end positions?'),
]

print(f"\\n{'Comparison':<35} {'Mean Δ':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 90)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<35} {np.mean(delta):>8.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
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
print(f"\\n{'='*90}")
print("ALL CONDITIONS vs BARE")
print(f"{'='*90}")
print(f"\\n{'Condition':<30} {'d vs Bare':>10} {'Win%':>7} {'p':>12}")
print("-" * 65)
all_vs_bare = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname:<30} {d:>10.3f} {win:>6.1f}% {p_val:>11.2e} {sig:>5}")
    all_vs_bare[cname] = {'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val)}\
""")))

# ========== Cell 10: Direction-specific deep-dive ==========
cells.append(make_cell("code", s("""\
# Cell 10: Direction-specific deep-dive

# --- LAYERS ---
print("=" * 70)
print("DIRECTION 1: LAYER-WISE DECOMPOSITION")
print("=" * 70)

layer_ds = {}
for group_name in LAYER_GROUPS:
    cname = f'values_{group_name}'
    d = cohens_d(c['bare'] - c[cname])
    layer_ds[group_name] = d
    print(f"  {group_name}: d = {d:+.3f}")

d_all_values = cohens_d(c['bare'] - c['values_only_llm_kw'])
d_sum = sum(layer_ds.values())
print(f"\\n  Sum of 4 groups: d = {d_sum:+.3f}")
print(f"  All layers (values_only_llm_kw): d = {d_all_values:+.3f}")
print(f"  Additivity ratio: {d_sum/d_all_values:.2f}" if d_all_values != 0 else "  All-layers d = 0")

ranked = sorted(layer_ds.items(), key=lambda x: x[1], reverse=True)
print(f"  Ranked: {' > '.join(f'{n}({d:+.3f})' for n, d in ranked)}")

# --- PREFIX TYPES ---
print(f"\\n{'='*70}")
print("DIRECTION 2: PREFIX TYPE COMPARISON (VALUES-ONLY MODE)")
print(f"{'='*70}")

prefix_conds = {
    'values_only_llm_kw': 'LLM keyword',
    'values_only_static_fact': 'Static factual',
    'values_only_oracle': 'Oracle (actual query)',
    'values_only_random': 'Random tokens',
}
for cname, label in prefix_conds.items():
    d = cohens_d(c['bare'] - c[cname])
    print(f"  {label:<25} d = {d:+.3f}")

# --- INTERPOLATION ---
print(f"\\n{'='*70}")
print("DIRECTION 3: INTERPOLATION CURVE")
print(f"{'='*70}")

alphas = [0.0] + INTERP_ALPHAS + [1.0]
interp_ds = []
for alpha in alphas:
    if alpha == 0.0:
        d = 0.0  # bare vs bare
    elif alpha == 1.0:
        d = cohens_d(c['bare'] - c['values_only_llm_kw'])
    else:
        cname = f'values_interp_{int(alpha*100):03d}'
        d = cohens_d(c['bare'] - c[cname])
    interp_ds.append(d)
    print(f"  alpha={alpha:.2f}: d = {d:+.3f}")

# Fit linear to 0.25, 0.50, 0.75
from numpy.polynomial import polynomial as P
xs = np.array(INTERP_ALPHAS)
ys = np.array([interp_ds[1], interp_ds[2], interp_ds[3]])
slope, intercept = np.polyfit(xs, ys, 1)
y_pred_lin = slope * xs + intercept
ss_res = np.sum((ys - y_pred_lin) ** 2)
ss_tot = np.sum((ys - np.mean(ys)) ** 2)
r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
print(f"\\n  Linear fit: d = {slope:.3f} * alpha + {intercept:.3f}, R² = {r2_lin:.4f}")

# --- CROSS-DOC ---
print(f"\\n{'='*70}")
print("DIRECTION 4: CROSS-DOCUMENT TRANSFER")
print(f"{'='*70}")

d_cross = cohens_d(c['bare'] - c['values_cross_doc'])
d_same = cohens_d(c['bare'] - c['values_only_llm_kw'])
print(f"  Cross-doc vs bare: d = {d_cross:+.3f}")
print(f"  Same-doc vs bare:  d = {d_same:+.3f}")
print(f"  Cross-doc retains {d_cross/d_same*100:.0f}% of same-doc effect" if d_same != 0 else "")

# Length-match robustness: filter pairs where doc lengths are within 20%
doc_lens = np.array([r['doc_len'] for r in results])[valid]
prev_doc_lens = np.array([r.get('prev_doc_len', 0) or 0 for r in results])[valid]
length_ratio = np.where(prev_doc_lens > 0, doc_lens / prev_doc_lens, 0)
length_matched = (length_ratio > 0.8) & (length_ratio < 1.2)
n_matched = int(np.sum(length_matched))
if n_matched > 30:
    d_cross_matched = cohens_d(c['bare'][length_matched] - c['values_cross_doc'][length_matched])
    print(f"  Length-matched pairs ({n_matched}): d = {d_cross_matched:+.3f}")

# --- POSITIONAL ---
print(f"\\n{'='*70}")
print("DIRECTION 5: POSITIONAL ISOLATION")
print(f"{'='*70}")

pos_conds = {
    'values_first_quarter': ('First 25%', 0.25),
    'values_middle_half': ('Middle 50%', 0.50),
    'values_last_quarter': ('Last 25%', 0.25),
}
for cname, (label, frac) in pos_conds.items():
    d = cohens_d(c['bare'] - c[cname])
    d_per_frac = d / frac if frac > 0 else 0
    print(f"  {label:<15} d = {d:+.3f} (d per fraction = {d_per_frac:+.3f})")

d_all = cohens_d(c['bare'] - c['values_only_llm_kw'])
print(f"  All positions: d = {d_all:+.3f}")

# --- HARDNESS QUINTILE BREAKDOWN ---
print(f"\\n{'='*70}")
print("HARDNESS QUINTILE BREAKDOWN (key conditions)")
print(f"{'='*70}")

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

key_conds = ['values_only_llm_kw', 'values_layers_0_7', 'values_layers_24_31',
             'values_only_oracle', 'values_only_random', 'values_interp_050',
             'values_cross_doc', 'values_first_quarter', 'values_last_quarter']

header = f"{'Condition':<30}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (30 + 14 * 6))

hardness_breakdown = {}
for cname in key_conds:
    row = f"{cname:<30}"
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
    hardness_breakdown[cname] = {'quintile_ds': quintile_ds, 'overall_d': float(d_all)}\
""")))

# ========== Cell 11: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 11: Plots (2x3 grid)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Color-coding by direction
direction_colors = {
    'full_llm_kw': 'forestgreen', 'values_only_llm_kw': 'forestgreen',
    'values_layers_0_7': '#1f77b4', 'values_layers_8_15': '#2ca02c',
    'values_layers_16_23': '#ff7f0e', 'values_layers_24_31': '#d62728',
    'values_only_static_fact': 'goldenrod', 'values_only_oracle': 'gold',
    'values_only_random': 'khaki',
    'values_interp_025': 'mediumpurple', 'values_interp_050': 'darkorchid',
    'values_interp_075': 'purple',
    'values_cross_doc': 'coral',
    'values_first_quarter': 'lightblue', 'values_last_quarter': 'steelblue',
    'values_middle_half': 'cornflowerblue',
}

# --- Plot 1: All conditions bar chart ---
ax = axes[0, 0]
cnames_sorted = sorted(
    [cn for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda cn: cohens_d(c['bare'] - c[cn]),
    reverse=True
)
ds_bar = [cohens_d(c['bare'] - c[cn]) for cn in cnames_sorted]
colors_bar = [direction_colors.get(cn, 'lightgray') for cn in cnames_sorted]
ax.barh(range(len(cnames_sorted)), ds_bar, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(cnames_sorted)))
ax.set_yticklabels(cnames_sorted, fontsize=7)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare')
ax.invert_yaxis()

# --- Plot 2: Layer-wise decomposition ---
ax = axes[0, 1]
layer_names = list(LAYER_GROUPS.keys())
layer_d_vals = [cohens_d(c['bare'] - c[f'values_{gn}']) for gn in layer_names]
layer_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
bars = ax.bar(range(len(layer_names)), layer_d_vals, color=layer_colors,
              edgecolor='black', linewidth=0.5)
# Add all-layers reference
ax.axhline(y=d_all_values, color='forestgreen', linestyle='--', label=f'All layers (d={d_all_values:+.3f})')
ax.set_xticks(range(len(layer_names)))
ax.set_xticklabels([n.replace('layers_', 'L') for n in layer_names], fontsize=9)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Layer-wise Decomposition')
ax.legend(fontsize=8)
for i, v in enumerate(layer_d_vals):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=8)

# --- Plot 3: Prefix type comparison ---
ax = axes[0, 2]
prefix_names = list(prefix_conds.keys())
prefix_labels = list(prefix_conds.values())
prefix_ds = [cohens_d(c['bare'] - c[cn]) for cn in prefix_names]
prefix_bar_colors = ['forestgreen', 'goldenrod', 'gold', 'khaki']
bars = ax.bar(range(len(prefix_names)), prefix_ds, color=prefix_bar_colors,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(prefix_names)))
ax.set_xticklabels(prefix_labels, fontsize=8, rotation=15)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Prefix Type (Values-Only)')
for i, v in enumerate(prefix_ds):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=8)

# --- Plot 4: Interpolation curve ---
ax = axes[1, 0]
ax.plot(alphas, interp_ds, 'o-', color='purple', linewidth=2, markersize=8)
# Linear fit overlay
xs_fit = np.linspace(0, 1, 50)
ys_fit = slope * xs_fit + intercept
ax.plot(xs_fit, ys_fit, '--', color='gray', alpha=0.5, label=f'Linear fit (R²={r2_lin:.3f})')
ax.set_xlabel('Alpha (primed fraction)')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Value Interpolation Curve')
ax.set_xticks(alphas)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
for x, y in zip(alphas, interp_ds):
    ax.annotate(f"d={y:+.3f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8)

# --- Plot 5: Positional breakdown ---
ax = axes[1, 1]
pos_names = ['first_quarter', 'middle_half', 'last_quarter']
pos_labels = ['First 25%', 'Middle 50%', 'Last 25%']
pos_ds = [cohens_d(c['bare'] - c[f'values_{pn}']) for pn in pos_names]
pos_colors = ['lightblue', 'cornflowerblue', 'steelblue']
bars = ax.bar(range(len(pos_names)), pos_ds, color=pos_colors,
              edgecolor='black', linewidth=0.5)
# All-positions reference
ax.axhline(y=d_all_values, color='forestgreen', linestyle='--',
           label=f'All positions (d={d_all_values:+.3f})')
ax.set_xticks(range(len(pos_names)))
ax.set_xticklabels(pos_labels, fontsize=9)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Positional Breakdown')
ax.legend(fontsize=8)
for i, v in enumerate(pos_ds):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=8)

# --- Plot 6: Hardness × condition heatmap ---
ax = axes[1, 2]
hm_data = []
for cname in key_conds:
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
ax.set_yticks(range(len(key_conds)))
ax.set_yticklabels([cn.replace('values_', '') for cn in key_conds], fontsize=7)
for i in range(len(key_conds)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness × Condition')

plt.suptitle('Exp 09: Values Deep Dive', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 12: Save results JSON ==========
cells.append(make_cell("code", s("""\
# Cell 12: Save comprehensive results JSON

# Direction-specific summaries
layer_summary = {}
for group_name in LAYER_GROUPS:
    cname = f'values_{group_name}'
    layer_summary[group_name] = {
        'cohens_d': float(cohens_d(c['bare'] - c[cname])),
        'layers': LAYER_GROUPS[group_name],
    }

prefix_summary = {}
for cname, label in prefix_conds.items():
    prefix_summary[label] = {
        'cohens_d': float(cohens_d(c['bare'] - c[cname])),
    }

interp_summary = {}
for alpha, d_val in zip(alphas, interp_ds):
    interp_summary[f'alpha_{alpha:.2f}'] = float(d_val)
interp_summary['linear_slope'] = float(slope)
interp_summary['linear_intercept'] = float(intercept)
interp_summary['linear_r2'] = float(r2_lin)

cross_doc_summary = {
    'd_cross_doc_vs_bare': float(d_cross),
    'd_same_doc_vs_bare': float(d_same),
    'retention_fraction': float(d_cross / d_same) if d_same != 0 else 0,
}

positional_summary = {}
for cname, (label, frac) in pos_conds.items():
    d = cohens_d(c['bare'] - c[cname])
    positional_summary[label] = {
        'cohens_d': float(d),
        'fraction': frac,
        'd_per_fraction': float(d / frac) if frac > 0 else 0,
    }

final = {
    'experiment': 'exp09_values_deep_dive',
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
    'direction_summaries': {
        'layer_wise': layer_summary,
        'prefix_types': prefix_summary,
        'interpolation': interp_summary,
        'cross_document': cross_doc_summary,
        'positional': positional_summary,
    },
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'hardness_breakdown': hardness_breakdown,
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 13: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 13: GPU cleanup
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

output_path = "/home/jupyter/research/directed_kvcache_v2/09_values_deep_dive.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
