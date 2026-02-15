#!/usr/bin/env python3
"""Build script for 19_gemma_precision_and_selectivity.ipynb

Exp 19: Gemma Priming — Precision Fix & Selective Value Contamination

Exp 16 showed priming FAILS on Gemma 3 4B (static_fact d=-0.031 ns), but
values_only works (d=+0.056, p=0.009). The gap reveals -0.087 of key interference.

Two hypotheses:
- H1 (Precision): RoPE correction in bfloat16 (7-bit mantissa) with head_dim=256
  introduces ~8.6x more noise than Mistral's float16/128-dim. Float32 may recover it.
- H2 (Selectivity): On Mistral, value contamination signal lives in layers 0-15 and
  first 25% of positions (Exp 09). Targeting these on Gemma may amplify d=+0.056.

9 conditions on MS MARCO (N=300 queries):

| # | Condition           | Tests                              |
|---|---------------------|------------------------------------|
| 1 | bare                | Baseline                           |
| 2 | sf_trunc            | Standard bfloat16 RoPE correction  |
| 3 | sf_trunc_fp32       | Float32 RoPE correction (H1)       |
| 4 | sf_trunc_nocorr     | No RoPE correction                 |
| 5 | values_only         | Bare keys + sf primed values       |
| 6 | values_early_layers | Values layers 0-16 only (H2)       |
| 7 | values_early_pos    | Values first 25% positions (H2)    |
| 8 | values_alpha_25     | 25% primed / 75% bare blend (H2)  |
| 9 | rope_roundtrip      | Bare + RoPE roundtrip noise        |

Compute: 2 forward passes per passage + 9 scoring ≈ ~2-3h on L4
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
# Exp 19: Gemma Priming — Precision Fix & Selective Value Contamination

## Motivation

Exp 16 showed priming **FAILS** on Gemma 3 4B (`static_fact_trunc` d=-0.031, ns), but
**`values_only` works** (d=+0.056, p=0.009). The gap reveals **-0.087 of key interference**.

Two hypotheses explain this:

### H1: Precision Hypothesis
RoPE correction in bfloat16 (7-bit mantissa) with head_dim=256 introduces ~8.6x more
quantization noise than Mistral's float16 (10-bit mantissa) / 128-dim. Computing the
correction in float32 may recover the effect.

### H2: Selectivity Hypothesis
On Mistral (Exp 09), value contamination signal lives in layers 0-15 (88% of effect) and
first 25% of positions (dominant). Targeting these on Gemma — and reducing the "dose" —
may amplify the weak d=+0.056.

## Exp 16 Reference Values

| Condition | Gemma d | Mistral d | Gap |
|-----------|---------|-----------|-----|
| static_fact_trunc | -0.031 (ns) | +0.472 | -0.503 |
| random_trunc | -0.109 (***) | +0.091 | -0.200 |
| values_only | +0.056 (**) | +0.275 | -0.219 |
| Key interference | -0.087 | ~0 | — |

## Design: 9 Conditions, N=300 queries (MS MARCO)

| # | Condition | Description | Tests |
|---|-----------|-------------|-------|
| 1 | `bare` | Baseline | — |
| 2 | `sf_trunc` | Standard truncated + bfloat16 RoPE correction | Replicate Exp 16 reference |
| 3 | `sf_trunc_fp32` | Truncated + **float32** RoPE correction | **H1: precision hypothesis** |
| 4 | `sf_trunc_nocorr` | Truncated, NO RoPE correction | Is correction worse than mismatch? |
| 5 | `values_only` | Bare keys + sf primed values (all layers) | Replicate Exp 16 d=+0.056 |
| 6 | `values_early_layers` | Values_only, layers 0-16 only | **H2: layer selectivity** |
| 7 | `values_early_pos` | Values_only, first 25% of doc positions only | **H2: position selectivity** |
| 8 | `values_alpha_25` | 25% primed / 75% bare value blend | **H2: dose reduction** |
| 9 | `rope_roundtrip` | Bare cache + RoPE roundtrip noise on keys | **Control: noise vs content** |

## 7 Primary Comparisons (Bonferroni α = 0.05/7 = 0.00714)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | sf_trunc_fp32 vs sf_trunc | Does fp32 precision help? |
| C2 | sf_trunc_nocorr vs bare | Does uncorrected truncation hurt? |
| C3 | sf_trunc_nocorr vs sf_trunc | Is correction better than no correction? |
| C4 | values_only vs bare | Replicate Exp 16 d=+0.056 |
| C5 | values_early_layers vs values_only | Does layer selectivity help? |
| C6 | values_early_pos vs values_only | Does position selectivity help? |
| C7 | values_alpha_25 vs values_only | Does reduced dose help? |""")))

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

RESULTS_DIR = Path("results/exp19")
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
print(f"  EOS token ID: {tokenizer.eos_token_id}")

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

# ========== Cell 3: Lib imports + custom correct_rope_fp32 + templates ==========
cells.append(make_cell("code", s("""\
# Cell 3: Lib imports + custom correct_rope_fp32() + templates + constants
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
    build_hybrid_cache,
    replace_values_at_layers,
    replace_values_at_positions,
    interpolate_values,
    apply_rope_roundtrip_noise,
    _build_rope_correction,
    _get_rope_theta_for_layer,
    _get_head_dim,
    _rotate_half,
    _get_text_config,
)
from lib.analysis import cohens_d
from lib.data import count_words
from scipy import stats
from tqdm.auto import tqdm


# === Custom function: RoPE correction in float32 ===
def correct_rope_fp32(cache, offset, model):
    \"\"\"RoPE correction in float32 (not bfloat16) for Gemma precision test.

    Upcasts keys to float32 before applying the correction, then downcasts
    back to the original dtype. This isolates the precision hypothesis: if
    bfloat16 quantization during RoPE correction is the bottleneck, fp32
    should recover the priming effect.
    \"\"\"
    if offset == 0:
        return cache
    config = model.config
    head_dim = _get_head_dim(config)
    text_cfg = _get_text_config(config)
    n_layers = text_cfg.num_hidden_layers

    # Pre-compute corrections per unique theta (in float32)
    corrections = {}
    for layer_idx in range(n_layers):
        theta = _get_rope_theta_for_layer(config, layer_idx)
        if theta not in corrections:
            corrections[theta] = _build_rope_correction(offset, head_dim, theta)

    for layer_idx in range(n_layers):
        theta = _get_rope_theta_for_layer(config, layer_idx)
        cos_a, sin_a = corrections[theta]
        keys = _get_cache_keys(cache, layer_idx)
        device = keys.device
        orig_dtype = keys.dtype

        # Keep cos/sin in float32, upcast keys to float32
        c = cos_a.to(device=device)       # stays float32
        s_val = sin_a.to(device=device)    # stays float32

        # Skip BOS at position 0, correct doc keys at positions 1:
        doc_keys = keys[:, :, 1:, :].float()  # upcast bfloat16 -> float32
        corrected = doc_keys * c + _rotate_half(doc_keys) * s_val
        corrected = corrected.to(orig_dtype)   # downcast back

        _set_cache_keys(cache, layer_idx,
                       torch.cat([keys[:, :, :1, :], corrected], dim=2))
    return cache


# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

# Prefix text
from lib.surrogate import STATIC_SURROGATE_QUERIES
STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

# Experiment parameters
MAX_QUERIES = 300
MAX_PASSAGE_WORDS = 300
MIN_PASSAGES_PER_QUERY = 2
CHECKPOINT_EVERY = 25

CONDITION_NAMES = [
    'bare', 'sf_trunc', 'sf_trunc_fp32', 'sf_trunc_nocorr',
    'values_only', 'values_early_layers', 'values_early_pos',
    'values_alpha_25', 'rope_roundtrip',
]

N_COMPARISONS = 7
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS

# Exp 16 reference values (Gemma)
EXP16_REF = {
    'sf_trunc_d': -0.031,
    'random_trunc_d': -0.109,
    'values_only_d': 0.056,
}

# Exp 09 reference values (Mistral layer/position selectivity)
EXP09_REF = {
    'layers_0_7_d': 0.172,   # ~63% of values-only d=0.275
    'layers_8_15_d': 0.069,  # ~25%
    'layers_0_15_pct': 0.88, # 88% of total signal in first 16 layers
    'first_quarter_pct': 'dominant',
    'alpha_025_pct': 0.86,   # 86% of full values-only effect
}

# Mistral reference values
MISTRAL_REF = {
    'static_fact_trunc_d': 0.472,
    'values_only_d': 0.275,
}

print("Config ready")
print(f"  Model: {MODEL_NAME}")
print(f"  MAX_QUERIES: {MAX_QUERIES}")
print(f"  Conditions: {CONDITION_NAMES}")
print(f"  N_COMPARISONS: {N_COMPARISONS}, Bonferroni alpha: {BONFERRONI_ALPHA:.4f}")
print(f"  Static fact prefix: '{STATIC_FACT}'")
print(f"\\nExp 16 Gemma reference:")
for k, v in EXP16_REF.items():
    print(f"    {k}: {v:+.3f}")
print(f"\\nExp 09 Mistral selectivity reference:")
for k, v in EXP09_REF.items():
    print(f"    {k}: {v}")

# Verify correct_rope_fp32 works on a small test
print("\\nVerifying correct_rope_fp32()...")
test_ids = tokenizer("Hello world", return_tensors="pt", add_special_tokens=True)['input_ids'].to(exp_config.device)
with torch.no_grad():
    test_out = model(test_ids, use_cache=True, return_dict=True)
test_cache = _ensure_dynamic_cache(test_out.past_key_values)
test_cache_copy = deepcopy_cache(test_cache)
correct_rope_fp32(test_cache_copy, 5, model)
k_before = _get_cache_keys(test_cache, 0)[:, :, 1:, :]
k_after = _get_cache_keys(test_cache_copy, 0)[:, :, 1:, :]
diff = (k_before.float() - k_after.float()).abs().mean().item()
print(f"  Mean key difference after fp32 correction (offset=5): {diff:.6f}")
print(f"  BOS preserved: {torch.equal(_get_cache_keys(test_cache, 0)[:,:,:1,:], _get_cache_keys(test_cache_copy, 0)[:,:,:1,:])}")
del test_out, test_cache, test_cache_copy, test_ids
torch.cuda.empty_cache()
print("  correct_rope_fp32() verified OK")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO v1.1, filter ≤300 words, ≥2 passages, limit 300 queries
from datasets import load_dataset

print("=" * 70)
print("LOADING MS MARCO v1.1 — ALL PASSAGES PER QUERY")
print("=" * 70)

dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation",
                        trust_remote_code=True)
print(f"Total items in validation: {len(dataset)}")

queries = []
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
    if len(passage_texts) < MIN_PASSAGES_PER_QUERY:
        continue
    if not is_selected or sum(is_selected) == 0:
        continue

    word_counts = [count_words(p) for p in passage_texts]
    if any(wc > MAX_PASSAGE_WORDS for wc in word_counts):
        continue

    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    else:
        continue

    passage_list = []
    for i, (ptext, sel) in enumerate(zip(passage_texts, is_selected)):
        passage_list.append({
            'passage': ptext,
            'is_relevant': bool(sel == 1),
            'word_count': word_counts[i],
            'passage_idx': i,
        })

    queries.append({
        'query': query,
        'answer': answer,
        'passages': passage_list,
        'n_passages': len(passage_list),
        'n_relevant': sum(1 for p in passage_list if p['is_relevant']),
    })

    if len(queries) >= MAX_QUERIES * 3:
        break

np.random.shuffle(queries)
queries = queries[:MAX_QUERIES]
N = len(queries)

n_passages_list = [q['n_passages'] for q in queries]
total_passages = sum(n_passages_list)

print(f"\\nSelected {N} queries ({total_passages} total passages)")
print(f"Passages per query: mean={np.mean(n_passages_list):.1f}, "
      f"min={min(n_passages_list)}, max={max(n_passages_list)}")
print(f"Word counts: mean={np.mean([p['word_count'] for q in queries for p in q['passages']]):.0f}")

del dataset
gc.collect()\
""")))

# ========== Cell 5: Tokenize prefix + BPE boundary check ==========
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
example_doc = queries[0]['passages'][0]['passage']
concat = sf_str + DOCUMENT_TEMPLATE.format(document=example_doc)
concat_enc = tokenizer(concat, add_special_tokens=True)['input_ids']
prefix_enc = tokenizer(sf_str, add_special_tokens=True)['input_ids']
doc_ids_from_concat = concat_enc[len(prefix_enc):]

bare_doc_enc = tokenizer(DOCUMENT_TEMPLATE.format(document=example_doc),
                          add_special_tokens=False)['input_ids']
match = sum(1 for a, b in zip(doc_ids_from_concat, bare_doc_enc) if a == b)
total = max(len(bare_doc_enc), 1)
print(f"  static_fact: {match}/{total} tokens match ({100*match/total:.1f}%)")

# Condition explanation
print("\\n" + "=" * 70)
print("CONDITION DETAILS")
print("=" * 70)

conditions_detail = [
    ("1. bare",
     "[BOS][doc]",
     "Baseline. All other conditions compared to this."),
    ("2. sf_trunc",
     f"[BOS]['{STATIC_FACT}'\\n][doc] -> truncate + RoPE correct (bfloat16)",
     "Standard Exp 16 condition. Expected d ~ -0.031 (ns)."),
    ("3. sf_trunc_fp32",
     f"Same as sf_trunc but RoPE correction computed in float32",
     "H1 TEST: Does float32 precision recover the priming effect?"),
    ("4. sf_trunc_nocorr",
     f"[BOS]['{STATIC_FACT}'\\n][doc] -> truncate, NO RoPE correction",
     "Control: Is correction worse than position mismatch?"),
    ("5. values_only",
     "Bare keys + sf primed values (all layers, all positions)",
     "Replicate Exp 16 d=+0.056. Isolates value contamination."),
    ("6. values_early_layers",
     "values_only but ONLY layers 0-16 (first 50% of 34 layers)",
     "H2 TEST: Layer selectivity. On Mistral layers 0-15 = 88%."),
    ("7. values_early_pos",
     "values_only but ONLY first 25% of document positions",
     "H2 TEST: Position selectivity. On Mistral first 25% = dominant."),
    ("8. values_alpha_25",
     "25% primed values + 75% bare values (linear blend)",
     "H2 TEST: Dose reduction. On Mistral alpha=0.25 retains 86%."),
    ("9. rope_roundtrip",
     "Bare cache with RoPE(+offset) then RoPE(-offset) applied to keys",
     "CONTROL: Pure bfloat16 quantization noise, no content signal."),
]

for name, detail, purpose in conditions_detail:
    print(f"\\n### {name} ###")
    print(f"  Cache: {detail}")
    print(f"  Purpose: {purpose}")\
""")))

# ========== Cell 6: Main loop ==========
cells.append(make_cell("code", s("""\
# Cell 6: Main loop — 2 fwd passes per passage, build 9 cache variants, score all

print("=" * 70)
print(f"MAIN EVALUATION ({N} queries, ~{total_passages} passages)")
print("Model: Gemma 3 4B | 9 conditions")
print("=" * 70)

# Checkpoint resume
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('query_texts', [])
    current_queries = [q['query'] for q in queries]
    if ckpt_queries == current_queries:
        all_results = ckpt['results']
        start_idx = len(all_results)
        print(f"Resuming from checkpoint: {start_idx}/{N}")
    else:
        print("Checkpoint query mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

print(f"Evaluating queries {start_idx} to {N-1}")
print(f"Per passage: 2 forward passes + 9 scoring passes")

t_start = time.time()

for qidx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Queries"):
    query_data = queries[qidx]
    query = query_data['query']
    answer = query_data['answer']
    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    passage_results = []

    for pidx, pinfo in enumerate(query_data['passages']):
        passage = pinfo['passage']
        document_text = DOCUMENT_TEMPLATE.format(document=passage)

        # --- Matched tokenization (using sf prefix as reference) ---
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
        context_len = 1 + doc_len  # BOS + doc

        del full_enc, full_ids, sf_prefix_enc

        # === Forward pass 1: BARE cache ===
        bare_input = torch.cat([bos_id, doc_ids], dim=1)
        with torch.no_grad():
            bare_out = model(input_ids=bare_input,
                             attention_mask=torch.ones_like(bare_input),
                             use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
        del bare_out, bare_input

        # === Forward pass 2: PRIMED cache (static_fact prefix) ===
        primed_input = torch.cat([bos_id, sf_ids, doc_ids], dim=1)
        with torch.no_grad():
            primed_out = model(input_ids=primed_input,
                               attention_mask=torch.ones_like(primed_input),
                               use_cache=True, return_dict=True)
        primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
        del primed_out, primed_input

        # === Truncate once: keep [BOS] + [last doc_len positions] ===
        trunc_raw = extract_and_truncate_cache_with_bos(primed_full, doc_len)
        prefix_offset = sf_ids.shape[1]
        del primed_full

        # === Build 9 scoring caches ===

        # (1) bare — use bare_cache directly
        # Score bare LAST since score_answer_with_cache mutates

        # (2) sf_trunc — standard bfloat16 correction
        sf_trunc_cache = deepcopy_cache(trunc_raw)
        correct_rope_positions_with_bos(sf_trunc_cache, prefix_offset, model)

        # (3) sf_trunc_fp32 — float32 precision correction
        sf_trunc_fp32_cache = deepcopy_cache(trunc_raw)
        correct_rope_fp32(sf_trunc_fp32_cache, prefix_offset, model)

        # (4) sf_trunc_nocorr — no correction at all
        sf_trunc_nocorr_cache = deepcopy_cache(trunc_raw)
        # no correction applied

        # (5) values_only — bare keys + sf primed values (from corrected cache)
        values_only_cache = build_hybrid_cache(
            keys_source=bare_cache,
            values_source=sf_trunc_cache,
        )

        # (6) values_early_layers — values from layers 0-16 only
        values_early_layers_cache = replace_values_at_layers(
            bare_cache, sf_trunc_cache, list(range(17))
        )

        # (7) values_early_pos — values from first 25% of doc positions only
        pos_end = 1 + max(1, doc_len // 4)
        values_early_pos_cache = replace_values_at_positions(
            bare_cache, sf_trunc_cache, 1, pos_end
        )

        # (8) values_alpha_25 — 25% primed / 75% bare value blend
        values_alpha_25_cache = interpolate_values(
            bare_cache, sf_trunc_cache, 0.25
        )

        # (9) rope_roundtrip — bare cache + RoPE roundtrip noise
        rope_roundtrip_cache = deepcopy_cache(bare_cache)
        apply_rope_roundtrip_noise(rope_roundtrip_cache, prefix_offset, model)

        del trunc_raw

        # === Score all 9 conditions (deepcopy before each except bare which is last) ===
        nll_sf_trunc = score_answer_with_cache(
            deepcopy_cache(sf_trunc_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del sf_trunc_cache

        nll_sf_trunc_fp32 = score_answer_with_cache(
            deepcopy_cache(sf_trunc_fp32_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del sf_trunc_fp32_cache

        nll_sf_trunc_nocorr = score_answer_with_cache(
            deepcopy_cache(sf_trunc_nocorr_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del sf_trunc_nocorr_cache

        nll_values_only = score_answer_with_cache(
            deepcopy_cache(values_only_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del values_only_cache

        nll_values_early_layers = score_answer_with_cache(
            deepcopy_cache(values_early_layers_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del values_early_layers_cache

        nll_values_early_pos = score_answer_with_cache(
            deepcopy_cache(values_early_pos_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del values_early_pos_cache

        nll_values_alpha_25 = score_answer_with_cache(
            deepcopy_cache(values_alpha_25_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del values_alpha_25_cache

        nll_rope_roundtrip = score_answer_with_cache(
            deepcopy_cache(rope_roundtrip_cache), context_len,
            query_prompt, answer_text, model, tokenizer, exp_config)
        del rope_roundtrip_cache

        # Score bare LAST (mutates cache)
        nll_bare = score_answer_with_cache(
            bare_cache, context_len, query_prompt, answer_text,
            model, tokenizer, exp_config)
        del bare_cache

        gc.collect()
        torch.cuda.empty_cache()

        passage_results.append({
            'passage_idx': pinfo['passage_idx'],
            'is_relevant': pinfo['is_relevant'],
            'word_count': pinfo['word_count'],
            'doc_len': doc_len,
            'bare_nll': nll_bare,
            'sf_trunc_nll': nll_sf_trunc,
            'sf_trunc_fp32_nll': nll_sf_trunc_fp32,
            'sf_trunc_nocorr_nll': nll_sf_trunc_nocorr,
            'values_only_nll': nll_values_only,
            'values_early_layers_nll': nll_values_early_layers,
            'values_early_pos_nll': nll_values_early_pos,
            'values_alpha_25_nll': nll_values_alpha_25,
            'rope_roundtrip_nll': nll_rope_roundtrip,
        })

    all_results.append({
        'query_idx': qidx,
        'query': query,
        'n_passages': len(passage_results),
        'n_relevant': query_data['n_relevant'],
        'passage_data': passage_results,
    })

    # Checkpoint
    if (qidx + 1) % CHECKPOINT_EVERY == 0 or qidx == N - 1:
        ckpt_data = {
            'results': all_results,
            'query_texts': [q['query'] for q in queries],
            'completed': len(all_results),
            'total': N,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        n_done = qidx - start_idx + 1
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (N - qidx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {qidx+1}/{N} | {n_done} done in {elapsed/60:.1f}m | "
                   f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nEvaluation complete: {len(all_results)} queries in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 7: Analysis ==========
cells.append(make_cell("code", s("""\
# Cell 7: Analysis — H1 verdict, H2 verdict, key interference decomposition
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — GEMMA PRECISION FIX & SELECTIVE VALUE CONTAMINATION")
print("=" * 70)

N_VALID = len(all_results)
print(f"Valid queries: {N_VALID}")

# --- Collect per-passage NLLs ---
cond_nlls = {cn: [] for cn in CONDITION_NAMES}
for r in all_results:
    for p in r['passage_data']:
        for cn in CONDITION_NAMES:
            cond_nlls[cn].append(p[f'{cn}_nll'])

cond_arrays = {cn: np.array(vals) for cn, vals in cond_nlls.items()}

# Filter zero NLLs
valid = np.ones(len(cond_arrays['bare']), dtype=bool)
for cn in CONDITION_NAMES:
    valid &= (cond_arrays[cn] != 0)
n_passages_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total passages: {len(valid)}, Valid: {n_passages_valid}, Excluded: {n_excluded}")

c = {}
for cn in CONDITION_NAMES:
    c[cn] = cond_arrays[cn][valid]

# === 1. NLL Summary Table ===
print("\\n" + "=" * 70)
print("NLL SUMMARY (per-passage, Gemma 3 4B)")
print("=" * 70)

print(f"\\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10} {'Win%':>8}")
print("-" * 68)

gemma_ds = {}
for cn in CONDITION_NAMES:
    mean_nll = np.mean(c[cn])
    std_nll = np.std(c[cn])
    if cn == 'bare':
        print(f"{cn:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {'---':>10} {'---':>8}")
    else:
        delta = c['bare'] - c[cn]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        gemma_ds[cn] = d
        print(f"{cn:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d:>+10.3f} {win:>7.1f}%")

# === 2. Statistical Tests (7 primary comparisons) ===
print("\\n" + "=" * 70)
print(f"7 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print("=" * 70)

comparisons = [
    ('C1: fp32 vs bf16',
     c['sf_trunc'] - c['sf_trunc_fp32'],
     'Does fp32 precision help?'),
    ('C2: nocorr vs bare',
     c['bare'] - c['sf_trunc_nocorr'],
     'Does uncorrected truncation hurt?'),
    ('C3: nocorr vs bf16',
     c['sf_trunc'] - c['sf_trunc_nocorr'],
     'Is correction better than no correction?'),
    ('C4: values_only vs bare',
     c['bare'] - c['values_only'],
     'Replicate Exp 16 d=+0.056'),
    ('C5: early_layers vs values_only',
     c['values_only'] - c['values_early_layers'],
     'Does layer selectivity help?'),
    ('C6: early_pos vs values_only',
     c['values_only'] - c['values_early_pos'],
     'Does position selectivity help?'),
    ('C7: alpha_25 vs values_only',
     c['values_only'] - c['values_alpha_25'],
     'Does reduced dose help?'),
]

print(f"\\n{'Comparison':<35} {'Mean D':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 90)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    bonf_sig = p_val < BONFERRONI_ALPHA
    print(f"{name:<35} {np.mean(delta):>8.4f} {d:>+8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(bonf_sig),
        'question': question,
    }

# === 3. Key Derived Metrics ===
print("\\n" + "=" * 70)
print("KEY DERIVED METRICS")
print("=" * 70)

d_sf = gemma_ds.get('sf_trunc', 0)
d_fp32 = gemma_ds.get('sf_trunc_fp32', 0)
d_nocorr = gemma_ds.get('sf_trunc_nocorr', 0)
d_vo = gemma_ds.get('values_only', 0)
d_el = gemma_ds.get('values_early_layers', 0)
d_ep = gemma_ds.get('values_early_pos', 0)
d_a25 = gemma_ds.get('values_alpha_25', 0)
d_rt = gemma_ds.get('rope_roundtrip', 0)

precision_gain = d_fp32 - d_sf
key_interference_bf16 = d_vo - d_sf
key_interference_fp32 = d_vo - d_fp32
noise_baseline = d_rt
best_selective = max(d_el, d_ep, d_a25)
best_selective_name = ['values_early_layers', 'values_early_pos', 'values_alpha_25'][
    [d_el, d_ep, d_a25].index(best_selective)
]

print(f"\\n  Precision gain (fp32 - bf16):    {precision_gain:+.3f}")
print(f"    sf_trunc (bf16): d = {d_sf:+.3f}")
print(f"    sf_trunc_fp32:   d = {d_fp32:+.3f}")
print(f"")
print(f"  Key interference (bf16):          {key_interference_bf16:+.3f}")
print(f"    = d(values_only) - d(sf_trunc)")
print(f"    = {d_vo:+.3f} - ({d_sf:+.3f})")
print(f"")
print(f"  Key interference (fp32):          {key_interference_fp32:+.3f}")
print(f"    = d(values_only) - d(sf_trunc_fp32)")
print(f"    = {d_vo:+.3f} - ({d_fp32:+.3f})")
print(f"    (If smaller than bf16 interference, precision matters)")
print(f"")
print(f"  Noise baseline (rope_roundtrip):  {noise_baseline:+.3f}")
print(f"    (Pure bfloat16 roundtrip noise cost, no content)")
print(f"")
print(f"  Best selective condition:          {best_selective_name} d={best_selective:+.3f}")

# === 4. H1 VERDICT ===
print("\\n" + "=" * 70)
print("H1 VERDICT: PRECISION HYPOTHESIS")
print("=" * 70)

c1_result = comparison_results.get('C1: fp32 vs bf16', {})
c1_sig = c1_result.get('bonferroni_significant', False)
c1_d = c1_result.get('cohens_d', 0)

if c1_sig and precision_gain > 0.05:
    h1_verdict = "SUPPORTED"
    h1_msg = f"fp32 correction RECOVERS priming on Gemma (gain = {precision_gain:+.3f})"
elif precision_gain > 0 and not c1_sig:
    h1_verdict = "TREND"
    h1_msg = f"fp32 shows trend toward recovery ({precision_gain:+.3f}) but not significant"
else:
    h1_verdict = "REJECTED"
    h1_msg = f"fp32 correction DOES NOT RECOVER priming on Gemma (gain = {precision_gain:+.3f})"

print(f"\\n  >>> H1 {h1_verdict}: {h1_msg}")
print(f"")
print(f"  Details:")
print(f"    sf_trunc (bf16):     d = {d_sf:+.3f}")
print(f"    sf_trunc_fp32:       d = {d_fp32:+.3f}")
print(f"    precision gain:      {precision_gain:+.3f}")
print(f"    C1 test p-value:     {c1_result.get('p_value', 1):.2e}")
print(f"    noise baseline:      d = {d_rt:+.3f}")
print(f"")
print(f"  Cross-reference:")
print(f"    Mistral static_fact: d = {MISTRAL_REF['static_fact_trunc_d']:+.3f}")
print(f"    Gemma sf_trunc_fp32: d = {d_fp32:+.3f}")
print(f"    Recovery fraction:   {d_fp32 / MISTRAL_REF['static_fact_trunc_d'] * 100:.1f}%" if MISTRAL_REF['static_fact_trunc_d'] != 0 else "")

# === 5. H2 VERDICT ===
print("\\n" + "=" * 70)
print("H2 VERDICT: SELECTIVE CONTAMINATION")
print("=" * 70)

any_selective_better = (d_el > d_vo + 0.01) or (d_ep > d_vo + 0.01) or (d_a25 > d_vo + 0.01)
c5_sig = comparison_results.get('C5: early_layers vs values_only', {}).get('bonferroni_significant', False)
c6_sig = comparison_results.get('C6: early_pos vs values_only', {}).get('bonferroni_significant', False)
c7_sig = comparison_results.get('C7: alpha_25 vs values_only', {}).get('bonferroni_significant', False)

if any_selective_better and (c5_sig or c6_sig or c7_sig):
    h2_verdict = "SUPPORTED"
    h2_msg = "Selective contamination AMPLIFIES the value signal"
elif any_selective_better:
    h2_verdict = "TREND"
    h2_msg = f"Selective conditions show trends but not significant"
else:
    h2_verdict = "REJECTED"
    h2_msg = "Selective contamination DOES NOT AMPLIFY the value signal"

print(f"\\n  >>> H2 {h2_verdict}: {h2_msg}")
print(f"")
print(f"  Details:")
print(f"    values_only (all):    d = {d_vo:+.3f}")
print(f"    values_early_layers:  d = {d_el:+.3f} ({'*' if c5_sig else 'ns'})")
print(f"    values_early_pos:     d = {d_ep:+.3f} ({'*' if c6_sig else 'ns'})")
print(f"    values_alpha_25:      d = {d_a25:+.3f} ({'*' if c7_sig else 'ns'})")
print(f"")
print(f"  Exp 09 Mistral comparison:")
print(f"    Mistral layers 0-15:  ~88% of full signal")
print(f"    Gemma layers 0-16:    {d_el/d_vo*100:.0f}% of values_only" if d_vo != 0 else "    Gemma: d_vo = 0")
print(f"    Mistral alpha=0.25:   ~86% of full signal")
print(f"    Gemma alpha=0.25:     {d_a25/d_vo*100:.0f}% of values_only" if d_vo != 0 else "    Gemma: d_vo = 0")

# === 6. Key Interference Decomposition ===
print("\\n" + "=" * 70)
print("KEY INTERFERENCE DECOMPOSITION")
print("=" * 70)

print(f"\\n  bf16 pipeline:  values d={d_vo:+.3f} + keys d={d_sf - d_vo:+.3f} = total d={d_sf:+.3f}")
print(f"  fp32 pipeline:  values d={d_vo:+.3f} + keys d={d_fp32 - d_vo:+.3f} = total d={d_fp32:+.3f}")
print(f"")
print(f"  Key interference reduced by fp32: {key_interference_bf16 - key_interference_fp32:+.3f}")
print(f"    bf16 interference: {-key_interference_bf16:+.3f}")
print(f"    fp32 interference: {-key_interference_fp32:+.3f}")

# === 7. Hardness Interaction ===
print("\\n" + "=" * 70)
print("HARDNESS INTERACTION")
print("=" * 70)

bare_all = c['bare']
quintile_boundaries = np.percentile(bare_all, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_all])

hardness_conds = ['sf_trunc', 'sf_trunc_fp32', 'values_only',
                  'values_early_layers', 'values_alpha_25']

header = f"{'Condition':<25}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (25 + 14 * 6))

hardness_breakdown = {}
for cn in hardness_conds:
    row = f"{cn:<25}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            delta = bare_all[mask_q] - c[cn][mask_q]
            d_q = cohens_d(delta)
            row += f"{d_q:>+14.3f}"
            quintile_ds.append(float(d_q))
    d_all = cohens_d(bare_all - c[cn])
    row += f"{d_all:>+14.3f}"
    print(row)
    hardness_breakdown[cn] = {'quintile_ds': quintile_ds, 'overall_d': float(d_all)}

# Hardness correlation
print("\\nHardness correlation (bare NLL vs delta):")
for cn in ['sf_trunc', 'sf_trunc_fp32', 'values_only']:
    delta = bare_all - c[cn]
    r, p = stats.pearsonr(bare_all, delta)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {cn:<25} r={r:+.3f}  p={p:.2e}  {sig}")\
""")))

# ========== Cell 8: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 8: Plots — 4-panel figure

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Color scheme
colors_h1 = {
    'sf_trunc': '#d62728',
    'sf_trunc_fp32': '#2ca02c',
    'sf_trunc_nocorr': '#ff7f0e',
    'rope_roundtrip': '#7f7f7f',
}
colors_h2 = {
    'values_only': '#1f77b4',
    'values_early_layers': '#9467bd',
    'values_early_pos': '#8c564b',
    'values_alpha_25': '#e377c2',
}

# --- Panel 1 (top-left): H1 — RoPE Precision ---
ax = axes[0, 0]
h1_conds = ['sf_trunc', 'sf_trunc_fp32', 'sf_trunc_nocorr', 'rope_roundtrip']
h1_ds = [gemma_ds.get(cn, 0) for cn in h1_conds]
h1_colors = [colors_h1[cn] for cn in h1_conds]
h1_labels = ['sf_trunc\\n(bf16)', 'sf_trunc\\n(fp32)', 'sf_trunc\\n(no corr)', 'rope\\nroundtrip']

bars = ax.bar(range(len(h1_conds)), h1_ds, color=h1_colors, edgecolor='black', linewidth=0.5)
# values_only reference line
ax.axhline(y=d_vo, color='#1f77b4', linestyle='--', linewidth=1.5,
           label=f'values_only (d={d_vo:+.3f})')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xticks(range(len(h1_conds)))
ax.set_xticklabels(h1_labels, fontsize=8)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("H1: RoPE Precision")
ax.legend(fontsize=8)
for i, v in enumerate(h1_ds):
    ax.text(i, v + 0.003 if v >= 0 else v - 0.012, f"{v:+.3f}",
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

# --- Panel 2 (top-right): H2 — Selective Contamination ---
ax = axes[0, 1]
h2_conds = ['values_only', 'values_early_layers', 'values_early_pos', 'values_alpha_25']
h2_ds = [gemma_ds.get(cn, 0) for cn in h2_conds]
h2_colors = [colors_h2[cn] for cn in h2_conds]
h2_labels = ['values\\nonly (all)', 'early\\nlayers', 'early\\npositions', 'alpha\\n0.25']

bars = ax.bar(range(len(h2_conds)), h2_ds, color=h2_colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xticks(range(len(h2_conds)))
ax.set_xticklabels(h2_labels, fontsize=8)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("H2: Selective Contamination")
for i, v in enumerate(h2_ds):
    ax.text(i, v + 0.003 if v >= 0 else v - 0.012, f"{v:+.3f}",
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

# --- Panel 3 (bottom-left): Key Interference Decomposition (waterfall) ---
ax = axes[1, 0]

# Waterfall: show values contribution + key interference for bf16 and fp32
categories = ['Values\\n(shared)', 'Keys\\n(bf16)', 'Total\\n(bf16)',
              'Values\\n(shared)', 'Keys\\n(fp32)', 'Total\\n(fp32)']
values_contrib = d_vo
keys_bf16 = d_sf - d_vo
total_bf16 = d_sf
keys_fp32 = d_fp32 - d_vo
total_fp32 = d_fp32

# Draw as grouped bars
x_pos = [0, 1, 2, 3.5, 4.5, 5.5]
bar_vals = [values_contrib, keys_bf16, total_bf16, values_contrib, keys_fp32, total_fp32]
bar_colors = ['#1f77b4', '#d62728', '#2c2c2c', '#1f77b4', '#2ca02c', '#2c2c2c']

bars = ax.bar(x_pos, bar_vals, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=7)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.set_ylabel("Cohen's d component")
ax.set_title("Key Interference Decomposition")

# Add labels
for i, (xp, v) in enumerate(zip(x_pos, bar_vals)):
    ax.text(xp, v + 0.003 if v >= 0 else v - 0.012, f"{v:+.3f}",
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

# Add bf16/fp32 section labels
ax.text(1, ax.get_ylim()[1] * 0.9, 'bfloat16', ha='center', fontsize=10, fontstyle='italic')
ax.text(4.5, ax.get_ylim()[1] * 0.9, 'float32', ha='center', fontsize=10, fontstyle='italic')

# --- Panel 4 (bottom-right): Hardness Interaction scatter ---
ax = axes[1, 1]

# Scatter for sf_trunc_fp32 and values_only
delta_fp32 = c['bare'] - c['sf_trunc_fp32']
delta_vo = c['bare'] - c['values_only']

ax.scatter(c['bare'], delta_fp32, alpha=0.1, s=6, color='#2ca02c', label='sf_trunc_fp32')
ax.scatter(c['bare'], delta_vo, alpha=0.1, s=6, color='#1f77b4', label='values_only')

# Fit lines
z_fp32 = np.polyfit(c['bare'], delta_fp32, 1)
z_vo = np.polyfit(c['bare'], delta_vo, 1)
x_fit = np.linspace(c['bare'].min(), c['bare'].max(), 100)
ax.plot(x_fit, np.polyval(z_fp32, x_fit), color='#2ca02c', linewidth=2, linestyle='--')
ax.plot(x_fit, np.polyval(z_vo, x_fit), color='#1f77b4', linewidth=2, linestyle='--')

r_fp32, p_fp32 = stats.pearsonr(c['bare'], delta_fp32)
r_vo, p_vo = stats.pearsonr(c['bare'], delta_vo)

ax.set_xlabel("Bare NLL (difficulty)")
ax.set_ylabel("ΔNLL (bare - condition)")
ax.set_title(f"Hardness: fp32 r={r_fp32:+.3f}, values r={r_vo:+.3f}")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=8, markerscale=3)

plt.suptitle('Exp 19: Gemma Precision Fix & Selective Value Contamination', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 9: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 9: Save results JSON
final = {
    'experiment': 'exp19_gemma_precision_and_selectivity',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME,
        'model_type': 'gemma3',
        'seed': SEED,
        'n_queries': N,
        'n_valid': N_VALID,
        'n_passages_valid': n_passages_valid,
        'n_passages_excluded': n_excluded,
        'max_passage_words': MAX_PASSAGE_WORDS,
        'min_passages_per_query': MIN_PASSAGES_PER_QUERY,
        'dataset': 'MS MARCO v1.1 validation',
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
    },
    'gemma_architecture': {
        'hidden_size': text_config.hidden_size,
        'num_layers': text_config.num_hidden_layers,
        'num_attention_heads': text_config.num_attention_heads,
        'num_kv_heads': text_config.num_key_value_heads,
        'head_dim': _get_head_dim(model.config),
        'rope_thetas': sorted(list(thetas)),
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': {
        cn: {
            'mean': float(np.mean(c[cn])),
            'std': float(np.std(c[cn])),
            'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cn])) if cn != 'bare' else 0.0,
        }
        for cn in CONDITION_NAMES
    },
    'primary_comparisons': comparison_results,
    'derived_metrics': {
        'precision_gain': float(precision_gain),
        'key_interference_bf16': float(key_interference_bf16),
        'key_interference_fp32': float(key_interference_fp32),
        'noise_baseline': float(noise_baseline),
        'best_selective_condition': best_selective_name,
        'best_selective_d': float(best_selective),
    },
    'verdicts': {
        'h1_precision': h1_verdict,
        'h1_message': h1_msg,
        'h2_selectivity': h2_verdict,
        'h2_message': h2_msg,
    },
    'reference_values': {
        'exp16_gemma': EXP16_REF,
        'exp09_mistral_selectivity': {k: str(v) for k, v in EXP09_REF.items()},
        'mistral': MISTRAL_REF,
    },
    'hardness_breakdown': hardness_breakdown,
    'per_query_results': all_results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")

# Print final summary
print("\\n" + "=" * 70)
print("SUMMARY — Exp 19: Gemma Precision & Selectivity")
print("=" * 70)
print(f"Model: Gemma 3 4B (34 layers, head_dim=256, bfloat16)")
print(f"Dataset: MS MARCO v1.1 ({N} queries, {n_passages_valid} passages)")
print(f"\\nEffect sizes (Cohen's d vs bare):")
for cn in CONDITION_NAMES[1:]:
    d_val = gemma_ds.get(cn, 0)
    print(f"  {cn:<25} d={d_val:>+.3f}")
print(f"\\nH1 (Precision):    {h1_verdict} — {h1_msg}")
print(f"H2 (Selectivity):  {h2_verdict} — {h2_msg}")
print(f"\\nDone!")\
""")))

# ========== Cell 10: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 10: GPU cleanup
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

output_path = "/home/jupyter/research/directed_kvcache_v2/19_gemma_precision_and_selectivity.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
