#!/usr/bin/env python3
"""Build script for 13_position_aware_priming.ipynb

Exp 13: Position-Aware Value Contamination for Long Documents

KEY INSIGHT FROM DIAGNOSTIC ANALYSIS (exp 11/12):
  - 80% of NQ answers are in the first 25% of the document
  - Contamination is STRONGEST at early positions (exponential decay)
  - When answers are early (high contamination): priming HURTS (d=-0.052)
  - When answers are late (low contamination): priming HELPS (d=+0.145)
  - Hurt-tail has extreme kurtosis (73-227): a few catastrophic outliers
    drag the mean negative despite 65% win rate
  - layers_0_15 and amplify_2x are r=0.982 correlated (doing same thing)

HYPOTHESIS: The contamination mechanism IS working on long docs, but it
DISRUPTS early-position answer regions. If we modulate contamination by
position (reduce at beginning, boost at end), we can recover the effect.

Design:
  - Dataset: Same 315 NQ samples from exp 12
  - 10 conditions: all derived from bare + 1x primed caches (2 forward passes)
  - Position-variant conditions manipulate the delta without extra forward passes
  - Delta forensics diagnostic on a subset

Conditions:
  1. bare                — baseline
  2. standard_1x         — static_fact_trunc (replicate exp 12)
  3. layers_0_15_amp2x   — combine layer targeting + 2x amplification
  4. layers_0_15_amp3x   — combine layer targeting + 3x amplification
  5. pos_normalized      — normalize delta to constant per-position L2 norm
  6. attenuate_first_25  — scale delta ×0.25 at first 25% of positions
  7. skip_first_25       — zero delta at first 25% of positions
  8. last_50_only        — contaminate only positions 50-100%
  9. window_25_75        — contaminate only positions 25-75%
  10. pos_norm_L0_15     — position normalization + layer targeting
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

# ========== Cell 0: Overview ==========
cells.append(make_cell("markdown", s("""\
# Exp 13: Position-Aware Value Contamination for Long Documents

## Background & Motivation

Exp 12 tested three hypotheses for why priming fails on long documents:
- Signal dilution (repetition) → REFUTED (more reps = worse)
- Amplification → Partially promising (amplify_2x d=+0.090, best overall)
- RoPE interference → REFUTED (suffix/no_rope both hurt)
- Layer targeting (0-15) → Second best (d=+0.083)

**But diagnostic analysis of the per-sample data revealed something deeper:**

### The Answer Position Finding

| Answer Location | % of Samples | static_fact d | Priming Effect |
|-----------------|-------------|---------------|----------------|
| First 25% of doc | **80%** | **-0.052** | **HURTS** |
| Later 75% of doc | 20% | **+0.145** | **HELPS** |

The contamination mechanism IS working — it helps when answers are far from
the heavily-contaminated early positions. The problem is that **80% of NQ answers
sit in the first 25% of the document, exactly where contamination is strongest.**

### The Asymmetry Finding

The delta distribution has extreme kurtosis (73-227):
- Most samples: tiny positive or negative delta (near zero)
- A few outliers: catastrophically harmed (Δ = -0.5 to -1.5)
- Win rate is 65% but mean is negative because hurt magnitude is 2.16x help magnitude

For `layers_0_15`: hurt magnitude is only 0.42x of help magnitude (best ratio).
For `amplify_2x`: hurt magnitude is 0.55x of help magnitude.

### The Correlation Finding

`layers_0_15` and `amplify_2x` have r=0.982 per-sample correlation — they are
essentially doing the same thing. This makes sense because the delta at layers
16-31 is ~0 (Exp 09), so both operations effectively "keep delta at layers 0-15."

## This Experiment: Position-Selective Contamination

**Core idea:** Instead of uniform contamination, modulate the contamination delta
by position — reduce or eliminate it at early positions (where answers live) and
optionally boost it at later positions (where it helps).

10 conditions, all derived from just 2 forward passes (bare + primed) per sample.""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup
import os
os.umask(0o000)

import sys
import json
import time
import math
import numpy as np
import torch
import gc
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp13")
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

# ========== Cell 3: Config ==========
cells.append(make_cell("code", s("""\
# Cell 3: Config and library imports
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
    deepcopy_cache,
    replace_values_at_layers,
    replace_values_at_positions,
    interpolate_values,
    build_hybrid_cache,
)
from lib.analysis import cohens_d
from lib.surrogate import STATIC_SURROGATE_QUERIES
from scipy import stats
from scipy.stats import spearmanr
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    seed=SEED,
)

SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

N_CONDITIONS = 10
N_COMPARISONS = 9  # each non-bare vs bare
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
CHECKPOINT_EVERY = 25
DELTA_FORENSICS_EVERY = 50  # log delta diagnostics for every Nth sample

STATIC_FACT = STATIC_SURROGATE_QUERIES['static_factual']['query']

LENGTH_BINS = [
    ('short',     100,  300),
    ('medium',    300,  800),
    ('long',      800,  2000),
    ('very_long', 2000, 4000),
]

CONDITION_NAMES = [
    'bare',
    'standard_1x',
    'layers_0_15_amp2x',
    'layers_0_15_amp3x',
    'pos_normalized',
    'attenuate_first_25',
    'skip_first_25',
    'last_50_only',
    'window_25_75',
    'pos_norm_L0_15',
]

print("Config ready")
print(f"  N_CONDITIONS: {N_CONDITIONS}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  static_fact: '{STATIC_FACT}'")\
""")))

# ========== Cell 4: Load samples ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load NQ samples (reuse exp 12's cached samples)
print("=" * 70)
print("LOADING NATURAL QUESTIONS SAMPLES")
print("=" * 70)

EXP12_SAMPLES_PATH = Path("results/exp12/nq_samples.json")

if not EXP12_SAMPLES_PATH.exists():
    raise FileNotFoundError(
        f"Exp 12 samples not found at {EXP12_SAMPLES_PATH}. "
        "Run exp 12 first."
    )

with open(EXP12_SAMPLES_PATH, 'r') as f:
    cached = json.load(f)
samples = cached['samples']
N = len(samples)

print(f"Loaded {N} NQ samples from {EXP12_SAMPLES_PATH}")
print(f"\\nSample distribution:")
for bin_name, bin_min, bin_max in LENGTH_BINS:
    bin_s = [s for s in samples if s['length_bin'] == bin_name]
    if bin_s:
        wcs = [s['word_count'] for s in bin_s]
        print(f"  {bin_name} ({bin_min}-{bin_max}w): n={len(bin_s)}, "
              f"mean={np.mean(wcs):.0f}w")\
""")))

# ========== Cell 5: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 5: Explain experimental conditions

print("=" * 70)
print("EXPERIMENTAL CONDITIONS — POSITION-AWARE CONTAMINATION")
print("=" * 70)

sf_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_ids = tokenizer(sf_str, add_special_tokens=False)['input_ids']
sf_tok_len = len(sf_ids)

print(f"\\nAll conditions use static_fact prefix ({sf_tok_len} tokens).")
print("All use bare keys. Only VALUES are modified (Exp 08: values carry 100% of signal).")
print("All are derived from 2 forward passes: bare + standard 1x primed.")
print(f"\\nDiagram of a 1000-word NQ document (~1500 tokens):")
print()
print("Position:     0     25%      50%      75%    100%")
print("              |------|--------|--------|-------|")
print("standard_1x:  ██████████░░░░░░░░░░░░░░░░░░░░░░  (decay from left)")
print("pos_norm:     ████████████████████████████████████ (uniform)")
print("atten_25:     ░░████████░░░░░░░░░░░░░░░░░░░░░░  (reduce first 25%)")
print("skip_25:      ··████████░░░░░░░░░░░░░░░░░░░░░░  (zero first 25%)")
print("last_50:      ··········████████████████████████  (only last 50%)")
print("window_25_75: ··████████████████████████··      (middle 50%)")
print()
print("█ = full delta  ░ = reduced delta  · = zero delta")

conditions_explained = [
    ("1. bare",
     "Baseline — no prefix",
     "—"),
    ("2. standard_1x",
     "Standard static_fact_trunc. Natural decay: first tokens get ~85% contamination "
     "from prefix attention, last tokens get <1%. Replicates exp 12 prefix_1x.",
     "Reference for position-modulated conditions"),
    ("3. layers_0_15_amp2x",
     "Amplify delta by 2x, but ONLY at layers 0-15 (layers 16-31 get bare values). "
     "Combines the two best Exp 12 approaches.",
     "Tests: Do amplification + layer targeting have independent benefits? "
     "(r=0.982 in Exp 12 suggests they're nearly identical)"),
    ("4. layers_0_15_amp3x",
     "Same as above but 3x amplification. Pushes further to find the sweet spot.",
     "Tests: Is 2x undershoot? Does 3x over-amplify?"),
    ("5. pos_normalized",
     "Normalize the delta at each position to the MEDIAN per-position L2 norm. "
     "Early positions (over-contaminated) get reduced. Late positions (under-contaminated) "
     "get amplified. Net contamination 'dose' stays the same but distributed evenly.",
     "KEY TEST: If contamination DIRECTION is correct everywhere but MAGNITUDE is wrong, "
     "normalization should fix it. This is the cleanest test of 'position-dependent dose.'"),
    ("6. attenuate_first_25",
     "Scale delta by 0.25 at first 25% of positions. Keep full delta elsewhere. "
     "A gentler version of 'skip' — we know the first 25% has the answer.",
     "Tests: Does REDUCING (not eliminating) early contamination help?"),
    ("7. skip_first_25",
     "Set delta to ZERO at first 25% of positions. Full delta elsewhere. "
     "Since 80% of answers are in the first 25%, this protects the answer region.",
     "KEY TEST: If early contamination hurts, zeroing it should flip the effect positive."),
    ("8. last_50_only",
     "Delta only at positions 50-100% (last half). First half gets bare values. "
     "Most aggressive answer-region protection.",
     "Tests: Can contamination help even with zero dose in the answer region?"),
    ("9. window_25_75",
     "Delta only at positions 25-75% (middle half). Protects BOTH early (answer) "
     "and late (possibly low-quality at extremes) positions.",
     "Tests: Is there an optimal position window for contamination?"),
    ("10. pos_norm_L0_15",
     "Position-normalized + layers 0-15 only. The 'kitchen sink' — combines the "
     "best layer targeting with position normalization.",
     "Tests: Do position + layer targeting compound?"),
]

for name, detail, test in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  {detail}")
    print(f"  Test: {test}")\
""")))

# ========== Cell 6: Helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 6: Position-aware cache manipulation functions

def compute_delta(bare_cache, primed_cache, layers=None):
    \"\"\"Compute per-position value delta between primed and bare caches.

    Returns dict mapping layer_idx -> delta tensor (same shape as values).
    If layers is specified, only compute for those layers.
    \"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    primed_cache = _ensure_dynamic_cache(primed_cache)
    n_layers = len(bare_cache)

    deltas = {}
    layer_range = layers if layers is not None else range(n_layers)
    for li in layer_range:
        v_bare = _get_cache_values(bare_cache, li)
        v_primed = _get_cache_values(primed_cache, li)
        deltas[li] = v_primed - v_bare

    return deltas


def apply_delta(bare_cache, deltas, scale=1.0, pos_start=None, pos_end=None):
    \"\"\"Apply value delta to bare cache with optional scaling and position range.

    Args:
        bare_cache: Cache with correct keys and uncontaminated values
        deltas: Dict mapping layer_idx -> delta tensor
        scale: Scalar or per-position tensor to multiply delta by
        pos_start: First position to modify (inclusive, 0-indexed in BOS+doc)
        pos_end: Last position to modify (exclusive)

    Returns:
        New DynamicCache with bare keys and modified values
    \"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    n_layers = len(bare_cache)
    new_cache = DynamicCache()

    for li in range(n_layers):
        k = _get_cache_keys(bare_cache, li)  # keys never modified — share, don't clone
        v = _get_cache_values(bare_cache, li).clone()

        if li in deltas:
            delta = deltas[li]
            if isinstance(scale, torch.Tensor):
                # Per-position scaling: scale shape should be (1, 1, seq_len, 1)
                # or broadcastable to delta shape
                scaled_delta = delta * scale
            else:
                scaled_delta = delta * scale

            if pos_start is not None or pos_end is not None:
                ps = pos_start if pos_start is not None else 0
                pe = pos_end if pos_end is not None else v.shape[2]
                v[:, :, ps:pe, :] += scaled_delta[:, :, ps:pe, :]
            else:
                v += scaled_delta

        new_cache.update(k, v, li)

    return new_cache


def compute_position_norms(deltas, start_pos=1):
    \"\"\"Compute L2 norm of delta at each position (averaged across layers).

    Args:
        deltas: Dict mapping layer_idx -> delta tensor (1, n_heads, seq_len, head_dim)
        start_pos: Skip BOS (position 0)

    Returns:
        Tensor of shape (seq_len,) with per-position L2 norms averaged across layers
    \"\"\"
    norms_list = []
    for li, delta in deltas.items():
        # L2 norm across head_dim, averaged across heads
        # delta shape: (1, n_heads, seq_len, head_dim)
        pos_norms = torch.norm(delta[0], dim=-1).mean(dim=0)  # (seq_len,)
        norms_list.append(pos_norms)

    avg_norms = torch.stack(norms_list).mean(dim=0)  # (seq_len,)
    return avg_norms


def position_normalize_delta(deltas, target_norm=None, start_pos=1, eps=1e-8):
    \"\"\"Normalize delta to constant per-position L2 norm.

    For each position, scale the delta so its L2 norm equals target_norm.
    Default target_norm = median of non-BOS position norms.

    Args:
        deltas: Dict mapping layer_idx -> delta tensor
        target_norm: Target L2 norm per position (default: median)
        start_pos: Position to start normalization (skip BOS)
        eps: Small constant to avoid division by zero

    Returns:
        New deltas dict with normalized values, plus the target_norm used
    \"\"\"
    # Compute per-position norms for each layer
    normalized = {}

    # First pass: compute all position norms per layer
    layer_norms = {}
    for li, delta in deltas.items():
        # (1, n_heads, seq_len, head_dim) -> per-position norm
        # Norm across head_dim for each head, then average across heads
        pos_norms = torch.norm(delta[0], dim=-1).mean(dim=0)  # (seq_len,)
        layer_norms[li] = pos_norms

    # Compute target from average across layers
    avg_norms = torch.stack(list(layer_norms.values())).mean(dim=0)
    if target_norm is None:
        # Use median of doc positions (skip BOS at 0)
        target_norm = float(torch.median(avg_norms[start_pos:]).item())

    # Second pass: normalize each layer's delta
    for li, delta in deltas.items():
        pos_norms = layer_norms[li]  # (seq_len,)
        # Scale factor per position: target / current_norm
        scale = torch.ones_like(pos_norms)
        # Only normalize doc positions (not BOS)
        for p in range(start_pos, len(pos_norms)):
            if pos_norms[p] > eps:
                scale[p] = target_norm / pos_norms[p]
            else:
                scale[p] = 0.0  # Zero delta stays zero

        # Reshape for broadcasting: (1, 1, seq_len, 1)
        scale_4d = scale.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        normalized[li] = delta * scale_4d.to(delta.device, dtype=delta.dtype)

    return normalized, target_norm


def build_position_scaled_cache(bare_cache, deltas, scale_fn, n_layers_model):
    \"\"\"Build cache with position-dependent scaling applied to deltas.

    Args:
        bare_cache: Uncontaminated cache
        deltas: Dict mapping layer_idx -> delta tensor
        scale_fn: Function(position_idx, seq_len) -> scale factor
        n_layers_model: Total number of model layers

    Returns:
        New DynamicCache
    \"\"\"
    bare_cache = _ensure_dynamic_cache(bare_cache)
    new_cache = DynamicCache()

    # Pre-compute scale factors
    sample_delta = next(iter(deltas.values()))
    seq_len = sample_delta.shape[2]
    scales = torch.tensor([scale_fn(p, seq_len) for p in range(seq_len)],
                          dtype=sample_delta.dtype, device=sample_delta.device)
    scales_4d = scales.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, seq_len, 1)

    for li in range(n_layers_model):
        k = _get_cache_keys(bare_cache, li)  # keys never modified — share, don't clone
        v = _get_cache_values(bare_cache, li).clone()

        if li in deltas:
            v = v + deltas[li] * scales_4d

        new_cache.update(k, v, li)

    return new_cache


print("Helper functions defined:")
print("  compute_delta(bare, primed, layers=None) -> dict")
print("  apply_delta(bare, deltas, scale, pos_start, pos_end) -> cache")
print("  compute_position_norms(deltas) -> tensor")
print("  position_normalize_delta(deltas, target_norm) -> (norm_deltas, target)")
print("  build_position_scaled_cache(bare, deltas, scale_fn, n_layers) -> cache")\
""")))

# ========== Cell 7: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 7: Main evaluation loop — 10 conditions, 2 forward passes per sample

print("=" * 70)
print(f"MAIN EVALUATION ({N_CONDITIONS} conditions x {N} samples)")
print("=" * 70)

# Pre-tokenize prefix
sf_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=STATIC_FACT)
sf_prefix_enc = tokenizer(sf_prefix_str, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
sf_prefix_ids = sf_prefix_enc['input_ids'].to(config.device)
sf_prefix_len = sf_prefix_ids.shape[1]

print(f"Static fact prefix: '{STATIC_FACT}' ({sf_prefix_len} tokens)")
print(f"Forward passes per sample: 2 (bare + primed)")
print(f"All other conditions derived via delta manipulation (no extra forward passes)")

# Checkpoint resume
results = []
delta_forensics = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in samples]
    if ckpt_queries == current_queries:
        results = ckpt['results']
        delta_forensics = ckpt.get('delta_forensics', [])
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
    context_len = 1 + doc_len  # BOS + doc

    # =================================================================
    # PHASE 1: Build bare and primed caches (2 forward passes)
    # =================================================================
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)
    del bare_out

    # Use static_fact prefix (matched tokenization: build from sf_prefix + same doc_ids)
    primed_ids = torch.cat([bos_id, sf_prefix_ids, doc_ids], dim=1)
    with torch.no_grad():
        primed_out = model(input_ids=primed_ids,
                           attention_mask=torch.ones_like(primed_ids),
                           use_cache=True, return_dict=True)
    primed_full = _ensure_dynamic_cache(primed_out.past_key_values)
    del primed_out

    # Truncate + RoPE correct -> standard primed cache
    primed_trunc = extract_and_truncate_cache_with_bos(primed_full, doc_len)
    correct_rope_positions_with_bos(primed_trunc, sf_prefix_len, model)
    del primed_full

    del bare_ids, primed_ids
    gc.collect()
    torch.cuda.empty_cache()

    # =================================================================
    # PHASE 2: Compute deltas (all layers, then layer-specific)
    # =================================================================
    deltas_all = compute_delta(bare_cache, primed_trunc)
    deltas_0_15 = {li: deltas_all[li] for li in range(16)}

    # =================================================================
    # PHASE 3: Build all position-variant caches from deltas
    # =================================================================
    # Positions: BOS is at 0, doc tokens at 1..doc_len
    # "first 25%" = positions 1..floor(doc_len*0.25)
    p25 = 1 + max(1, int(doc_len * 0.25))  # position index for 25% boundary
    p50 = 1 + max(1, int(doc_len * 0.50))
    p75 = 1 + max(1, int(doc_len * 0.75))
    p_end = 1 + doc_len

    # =================================================================
    # PHASE 3+4: Build, score, and free each condition one at a time
    # (Avoids holding all 9 derived caches in GPU memory simultaneously)
    # bare_cache must stay unmutated until all derived caches are scored,
    # so we score it LAST.
    # =================================================================

    # 2. standard_1x
    cache = build_hybrid_cache(bare_cache, primed_trunc)
    nll_standard = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache, primed_trunc

    # 3. layers_0_15_amp2x
    amp_deltas = {li: d * 2.0 for li, d in deltas_0_15.items()}
    cache = apply_delta(bare_cache, amp_deltas)
    nll_l15_a2 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache, amp_deltas

    # 4. layers_0_15_amp3x
    amp_deltas = {li: d * 3.0 for li, d in deltas_0_15.items()}
    cache = apply_delta(bare_cache, amp_deltas)
    nll_l15_a3 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache, amp_deltas

    # 5. pos_normalized (compute norm_deltas just-in-time, free immediately after)
    norm_deltas, target_norm = position_normalize_delta(deltas_all)
    cache = apply_delta(bare_cache, norm_deltas)
    nll_pos_norm = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache, norm_deltas
    gc.collect()
    torch.cuda.empty_cache()

    # 6. attenuate_first_25
    def atten_25_scale(pos, seq_len):
        if pos == 0:
            return 0.0
        if pos < p25:
            return 0.25
        return 1.0
    cache = build_position_scaled_cache(bare_cache, deltas_all, atten_25_scale, n_layers)
    nll_atten_25 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache

    # 7. skip_first_25
    def skip_25_scale(pos, seq_len):
        if pos < p25:
            return 0.0
        return 1.0
    cache = build_position_scaled_cache(bare_cache, deltas_all, skip_25_scale, n_layers)
    nll_skip_25 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache

    # 8. last_50_only
    def last_50_scale(pos, seq_len):
        if pos < p50:
            return 0.0
        return 1.0
    cache = build_position_scaled_cache(bare_cache, deltas_all, last_50_scale, n_layers)
    nll_last_50 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache

    # 9. window_25_75
    def window_scale(pos, seq_len):
        if pos < p25 or pos >= p75:
            return 0.0
        return 1.0
    cache = build_position_scaled_cache(bare_cache, deltas_all, window_scale, n_layers)
    nll_window = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache

    # 10. pos_norm_L0_15
    norm_deltas_0_15, _ = position_normalize_delta(deltas_0_15, target_norm=target_norm)
    cache = apply_delta(bare_cache, norm_deltas_0_15)
    nll_pnl15 = score_answer_with_cache(
        cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del cache, norm_deltas_0_15

    # =================================================================
    # Delta forensics (every Nth sample) — reuse existing deltas_all
    # =================================================================
    if idx % DELTA_FORENSICS_EVERY == 0:
        pos_norms = compute_position_norms(deltas_all, start_pos=1)

        n_positions = min(20, doc_len)
        sample_positions = [1 + int(i * doc_len / n_positions) for i in range(n_positions)]
        sample_norms = [float(pos_norms[p].item()) for p in sample_positions if p < len(pos_norms)]
        pct_positions = [float((p-1)/doc_len) for p in sample_positions if p < len(pos_norms)]

        layer_group_norms = {}
        for group_name, layers in [("L0-7", range(8)), ("L8-15", range(8,16)),
                                    ("L16-23", range(16,24)), ("L24-31", range(24,32))]:
            group_deltas = {li: deltas_all[li] for li in layers if li in deltas_all}
            if group_deltas:
                gn = compute_position_norms(group_deltas, start_pos=1)
                layer_group_norms[group_name] = {
                    'mean_norm': float(gn[1:].mean().item()),
                    'first_25_norm': float(gn[1:p25].mean().item()) if p25 > 1 else 0,
                    'last_25_norm': float(gn[p75:].mean().item()) if p75 < len(gn) else 0,
                }

        forensic_entry = {
            'idx': idx,
            'doc_len': doc_len,
            'word_count': word_count,
            'length_bin': length_bin,
            'target_norm': float(target_norm),
            'position_norms': sample_norms,
            'position_pcts': pct_positions,
            'decay_ratio': float(sample_norms[-1] / max(sample_norms[0], 1e-10))
                if sample_norms else 0,
            'layer_group_norms': layer_group_norms,
        }
        delta_forensics.append(forensic_entry)

    # Clean up deltas, then score bare last (scoring mutates the cache)
    del deltas_all, deltas_0_15
    nll_bare = score_answer_with_cache(
        bare_cache, context_len, query_prompt, answer_text, model, tokenizer, config)
    del bare_cache

    gc.collect()
    torch.cuda.empty_cache()

    # --- Store result ---
    # Find answer position for per-sample analysis
    ans_pos = None
    ans_lower = answer.lower()
    pass_lower = passage.lower()
    char_idx = pass_lower.find(ans_lower)
    if char_idx >= 0:
        ans_pos = char_idx / max(len(passage), 1)

    result = {
        'idx': idx,
        'doc_len_tokens': doc_len,
        'word_count': word_count,
        'length_bin': length_bin,
        'answer_position': ans_pos,
        'bare': nll_bare,
        'standard_1x': nll_standard,
        'layers_0_15_amp2x': nll_l15_a2,
        'layers_0_15_amp3x': nll_l15_a3,
        'pos_normalized': nll_pos_norm,
        'attenuate_first_25': nll_atten_25,
        'skip_first_25': nll_skip_25,
        'last_50_only': nll_last_50,
        'window_25_75': nll_window,
        'pos_norm_L0_15': nll_pnl15,
    }
    results.append(result)

    if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == N - 1:
        ckpt_data = {
            'results': results,
            'delta_forensics': delta_forensics,
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
# Cell 8: Analysis — overall + per-bin + answer position interaction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — POSITION-AWARE VALUE CONTAMINATION")
print("=" * 70)

# Arrays
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

# Filter zero NLLs
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
answer_pos_arr = np.array([r.get('answer_position', None) for r in results], dtype=object)[valid]

# ===== OVERALL RESULTS =====
print(f"\\n{'='*90}")
print(f"OVERALL RESULTS (N={n_valid})")
print(f"{'='*90}")
print(f"\\n{'Condition':<22} {'Mean NLL':>10} {'d vs Bare':>10} {'Win%':>7} {'p':>12} {'Sig':>5}")
print("-" * 70)

nll_summary = {}
comparison_results = {}

for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])

    if cname == 'bare':
        print(f"{cname:<22} {mean_nll:>10.4f} {'--':>10} {'--':>7} {'--':>12}")
        nll_summary[cname] = {'mean': float(mean_nll), 'std': float(std_nll), 'cohens_d': 0.0}
    else:
        delta = c['bare'] - c[cname]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        _, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
        print(f"{cname:<22} {mean_nll:>10.4f} {d:>+10.3f} {win:>5.1f}% {p_val:>11.2e} {sig:>5}")
        nll_summary[cname] = {'mean': float(mean_nll), 'std': float(std_nll), 'cohens_d': float(d)}
        comparison_results[f"{cname} vs bare"] = {
            'mean_delta': float(np.mean(delta)),
            'cohens_d': float(d),
            'win_rate': float(win / 100),
            'p_value': float(p_val),
            'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        }

# Highlight best
best_cond = max(comparison_results.items(), key=lambda x: x[1]['cohens_d'])
print(f"\\nBest condition: {best_cond[0]} (d={best_cond[1]['cohens_d']:+.3f})")

# ===== PER LENGTH BIN =====
print(f"\\n{'='*90}")
print("PER LENGTH BIN")
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
        if n_bin < 5:
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
        'bin_names': bin_names_ordered, 'bin_ds': bin_ds, 'bin_wins': bin_wins, 'bin_ns': bin_ns
    }

# ===== ANSWER POSITION INTERACTION =====
print(f"\\n{'='*90}")
print("ANSWER POSITION INTERACTION")
print(f"{'='*90}")

# Split by answer position
ans_pos_valid = np.array([
    float(ap) if ap is not None else np.nan for ap in answer_pos_arr
])
has_ans_pos = ~np.isnan(ans_pos_valid)

if np.sum(has_ans_pos) > 20:
    print(f"\\nSamples with answer position: {np.sum(has_ans_pos)}")
    early_mask = has_ans_pos & (ans_pos_valid < 0.25)
    late_mask = has_ans_pos & (ans_pos_valid >= 0.25)
    print(f"  Early answers (<25%): {np.sum(early_mask)}")
    print(f"  Late answers (>=25%): {np.sum(late_mask)}")

    answer_pos_results = {}
    print(f"\\n{'Condition':<22} {'Early d':>10} {'Late d':>10} {'Diff':>10}")
    print("-" * 55)
    for cname in CONDITION_NAMES:
        if cname == 'bare':
            continue
        delta = c['bare'] - c[cname]
        early_d = cohens_d(delta[early_mask]) if np.sum(early_mask) > 5 else float('nan')
        late_d = cohens_d(delta[late_mask]) if np.sum(late_mask) > 5 else float('nan')
        diff = late_d - early_d if not (np.isnan(early_d) or np.isnan(late_d)) else float('nan')
        if not np.isnan(early_d):
            print(f"{cname:<22} {early_d:>+10.3f} {late_d:>+10.3f} {diff:>+10.3f}")
        answer_pos_results[cname] = {
            'early_d': float(early_d) if not np.isnan(early_d) else None,
            'late_d': float(late_d) if not np.isnan(late_d) else None,
        }

# ===== ASYMMETRY ANALYSIS =====
print(f"\\n{'='*90}")
print("ASYMMETRY ANALYSIS — Hurt:Help Magnitude Ratio")
print(f"{'='*90}")
print(f"\\n{'Condition':<22} {'Helped':>8} {'Hurt':>8} {'Help mag':>10} {'Hurt mag':>10} {'Ratio':>7}")
print("-" * 70)

asymmetry_results = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    helped = delta > 0
    hurt = delta < 0
    help_mag = np.mean(delta[helped]) if np.any(helped) else 0
    hurt_mag = np.mean(delta[hurt]) if np.any(hurt) else 0
    ratio = abs(hurt_mag / help_mag) if help_mag != 0 else float('inf')
    print(f"{cname:<22} {np.sum(helped):>8d} {np.sum(hurt):>8d} "
          f"{help_mag:>+10.4f} {hurt_mag:>+10.4f} {ratio:>6.2f}x")
    asymmetry_results[cname] = {
        'n_helped': int(np.sum(helped)),
        'n_hurt': int(np.sum(hurt)),
        'help_magnitude': float(help_mag),
        'hurt_magnitude': float(hurt_mag),
        'hurt_help_ratio': float(ratio),
    }

# ===== KEY COMPARISONS: Position conditions vs standard_1x =====
print(f"\\n{'='*90}")
print("KEY COMPARISONS: Position-aware vs Standard")
print(f"{'='*90}")

position_conditions = ['pos_normalized', 'attenuate_first_25', 'skip_first_25',
                        'last_50_only', 'window_25_75', 'pos_norm_L0_15']

position_comparisons = {}
print(f"\\n{'Comparison':<35} {'d':>8} {'Win%':>7} {'p':>12}")
print("-" * 65)
for cname in position_conditions:
    delta = c[cname] - c['standard_1x']  # negative = position version better
    d = cohens_d(-delta)  # positive = position version wins
    win = np.mean(delta < 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname + ' vs standard_1x':<35} {d:>+8.3f} {win:>5.1f}% {p_val:>11.2e} {sig}")
    position_comparisons[f"{cname} vs standard_1x"] = {
        'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val),
    }\
""")))

# ========== Cell 9: Delta forensics ==========
cells.append(make_cell("code", s("""\
# Cell 9: Delta Forensics — How does contamination decay with position?

if delta_forensics:
    print("=" * 70)
    print("DELTA FORENSICS — Contamination Profile")
    print("=" * 70)

    # Group by length bin
    for bin_name in bin_names_ordered:
        bin_forensics = [f for f in delta_forensics if f['length_bin'] == bin_name]
        if len(bin_forensics) < 2:
            continue

        print(f"\\n  {bin_name} (n={len(bin_forensics)}):")

        # Average decay ratio
        decay_ratios = [f['decay_ratio'] for f in bin_forensics]
        print(f"    Decay ratio (last/first norm): {np.mean(decay_ratios):.4f} "
              f"(range: {min(decay_ratios):.4f} - {max(decay_ratios):.4f})")

        # Average target norm
        target_norms = [f['target_norm'] for f in bin_forensics]
        print(f"    Target norm (median position): {np.mean(target_norms):.6f}")

        # Layer group norms
        for group in ["L0-7", "L8-15", "L16-23", "L24-31"]:
            group_means = [f['layer_group_norms'].get(group, {}).get('mean_norm', 0)
                          for f in bin_forensics]
            first_25_means = [f['layer_group_norms'].get(group, {}).get('first_25_norm', 0)
                             for f in bin_forensics]
            last_25_means = [f['layer_group_norms'].get(group, {}).get('last_25_norm', 0)
                            for f in bin_forensics]
            if any(g > 0 for g in group_means):
                print(f"    {group}: mean={np.mean(group_means):.6f}, "
                      f"first_25%={np.mean(first_25_means):.6f}, "
                      f"last_25%={np.mean(last_25_means):.6f}, "
                      f"ratio={np.mean(last_25_means)/max(np.mean(first_25_means), 1e-10):.4f}")

    # Average contamination profile across all samples
    print(f"\\n  Average contamination profile (all samples, n={len(delta_forensics)}):")
    # Normalize to percentage positions
    n_bins = 20
    pct_bins = np.linspace(0, 1, n_bins + 1)
    avg_profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for f in delta_forensics:
        for pct, norm in zip(f['position_pcts'], f['position_norms']):
            bin_idx = min(int(pct * n_bins), n_bins - 1)
            avg_profile[bin_idx] += norm
            counts[bin_idx] += 1

    avg_profile = np.divide(avg_profile, counts, where=counts > 0)
    for i in range(n_bins):
        pct_lo = pct_bins[i] * 100
        pct_hi = pct_bins[i+1] * 100
        bar = "█" * int(avg_profile[i] / max(avg_profile.max(), 1e-10) * 40)
        print(f"    {pct_lo:5.0f}-{pct_hi:3.0f}%: {avg_profile[i]:.6f}  {bar}")
else:
    print("No delta forensics data collected.")\
""")))

# ========== Cell 10: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 10: Plots

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

colors = {
    'standard_1x': '#d62728',
    'layers_0_15_amp2x': '#ff7f0e',
    'layers_0_15_amp3x': '#e377c2',
    'pos_normalized': '#2ca02c',
    'attenuate_first_25': '#17becf',
    'skip_first_25': '#1f77b4',
    'last_50_only': '#9467bd',
    'window_25_75': '#8c564b',
    'pos_norm_L0_15': '#bcbd22',
}

# --- Plot 1: Overall bar chart sorted by d ---
ax = axes[0, 0]
conds_sorted = sorted(
    [(cn, cohens_d(c['bare'] - c[cn])) for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda x: x[1], reverse=True
)
names_s = [x[0] for x in conds_sorted]
ds_s = [x[1] for x in conds_sorted]
bar_colors = [colors.get(cn, 'gray') for cn in names_s]
bars = ax.barh(range(len(names_s)), ds_s, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=8)
for i, (name, dv) in enumerate(conds_sorted):
    ax.text(dv + 0.003, i, f"d={dv:+.3f}", va='center', fontsize=7)
ax.axvline(x=0, color='gray', linestyle='--')
ax.axvline(x=0.472, color='red', linestyle=':', alpha=0.4, label='MARCO static_fact')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title("Overall Effect (All Bins)")
ax.invert_yaxis()
ax.legend(fontsize=7)

# --- Plot 2: Per-bin for position conditions ---
ax = axes[0, 1]
x = np.arange(len(bin_names_ordered))
width = 0.12
plot_conds = ['standard_1x', 'pos_normalized', 'skip_first_25', 'last_50_only', 'pos_norm_L0_15']
for i, cname in enumerate(plot_conds):
    ds = per_bin_results[cname]['bin_ds']
    ds_clean = [d if d is not None else 0 for d in ds]
    offset = (i - len(plot_conds)/2 + 0.5) * width
    ax.bar(x + offset, ds_clean, width, label=cname, color=colors.get(cname, 'gray'),
           edgecolor='black', linewidth=0.3, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title("Position Conditions by Length Bin")
ax.legend(fontsize=6)

# --- Plot 3: Answer position interaction ---
ax = axes[0, 2]
if np.sum(has_ans_pos) > 20:
    for cname in ['standard_1x', 'pos_normalized', 'skip_first_25', 'layers_0_15_amp2x']:
        delta = c['bare'] - c[cname]
        valid_ap = has_ans_pos
        ax.scatter(ans_pos_valid[valid_ap], delta[valid_ap],
                  alpha=0.15, s=8, color=colors.get(cname, 'gray'), label=cname)
        # Trend line
        bins_ap = np.linspace(0, 1, 8)
        for k in range(len(bins_ap)-1):
            mask_k = valid_ap & (ans_pos_valid >= bins_ap[k]) & (ans_pos_valid < bins_ap[k+1])
            if np.sum(mask_k) > 5:
                ax.scatter((bins_ap[k]+bins_ap[k+1])/2, np.mean(delta[mask_k]),
                          s=50, color=colors.get(cname, 'gray'), edgecolor='black', linewidth=0.5, zorder=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Answer Position (0=start, 1=end)")
    ax.set_ylabel("NLL Reduction (positive = helps)")
    ax.set_title("Answer Position vs Priming Benefit")
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'Insufficient answer position data', ha='center', transform=ax.transAxes)

# --- Plot 4: Asymmetry comparison ---
ax = axes[1, 0]
asymmetry_conds = sorted(
    [(cn, asymmetry_results[cn]['hurt_help_ratio']) for cn in asymmetry_results],
    key=lambda x: x[1]
)
a_names = [x[0] for x in asymmetry_conds]
a_ratios = [x[1] for x in asymmetry_conds]
a_colors = [colors.get(cn, 'gray') for cn in a_names]
ax.barh(range(len(a_names)), a_ratios, color=a_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(a_names)))
ax.set_yticklabels(a_names, fontsize=8)
for i, (name, ratio) in enumerate(asymmetry_conds):
    ax.text(ratio + 0.02, i, f"{ratio:.2f}x", va='center', fontsize=7)
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Equal hurt/help')
ax.set_xlabel("Hurt:Help Magnitude Ratio (lower = better)")
ax.set_title("Asymmetry: How Bad is the Hurt Tail?")
ax.invert_yaxis()
ax.legend(fontsize=7)

# --- Plot 5: Contamination decay profile ---
ax = axes[1, 1]
if delta_forensics:
    for bin_name in bin_names_ordered:
        bin_f = [f for f in delta_forensics if f['length_bin'] == bin_name]
        if len(bin_f) < 2:
            continue
        # Average profile
        all_pcts = []
        all_norms = []
        for f in bin_f:
            all_pcts.extend(f['position_pcts'])
            all_norms.extend(f['position_norms'])
        if all_pcts:
            # Bin into 20 segments
            n_seg = 20
            seg_edges = np.linspace(0, 1, n_seg + 1)
            seg_means = []
            seg_centers = []
            for k in range(n_seg):
                mask_k = [(p >= seg_edges[k]) and (p < seg_edges[k+1])
                         for p in all_pcts]
                vals = [n for n, m in zip(all_norms, mask_k) if m]
                if vals:
                    seg_means.append(np.mean(vals))
                    seg_centers.append((seg_edges[k] + seg_edges[k+1])/2)
            ax.plot(seg_centers, seg_means, marker='o', markersize=3, label=bin_name)
    ax.set_xlabel("Position in Document (%)")
    ax.set_ylabel("Delta L2 Norm (contamination strength)")
    ax.set_title("Contamination Decay by Document Length")
    ax.legend(fontsize=8)
    ax.set_yscale('log')
else:
    ax.text(0.5, 0.5, 'No forensics data', ha='center', transform=ax.transAxes)

# --- Plot 6: Key condition comparison per bin (heatmap style) ---
ax = axes[1, 2]
all_conds = [cn for cn in CONDITION_NAMES if cn != 'bare']
heatmap_data = []
for cname in all_conds:
    row = []
    for bin_name in bin_names_ordered:
        d = None
        if cname in per_bin_results:
            bin_idx = bin_names_ordered.index(bin_name)
            d = per_bin_results[cname]['bin_ds'][bin_idx]
        row.append(d if d is not None else 0)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=-0.3, vmax=0.3)
ax.set_xticks(range(len(bin_names_ordered)))
ax.set_xticklabels(bin_names_ordered)
ax.set_yticks(range(len(all_conds)))
ax.set_yticklabels(all_conds, fontsize=8)
for i in range(len(all_conds)):
    for j in range(len(bin_names_ordered)):
        ax.text(j, i, f"{heatmap_data[i,j]:+.2f}", ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, label="Cohen's d vs Bare")
ax.set_title("Effect by Condition × Length Bin")

plt.suptitle('Exp 13: Position-Aware Value Contamination', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 11: Save results ==========
cells.append(make_cell("code", s("""\
# Cell 11: Save results
final = {
    'experiment': 'exp13_position_aware_priming',
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
        'dataset': 'Natural Questions (from exp 12 samples)',
        'length_bins': LENGTH_BINS,
        'static_fact': STATIC_FACT,
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': nll_summary,
    'primary_comparisons': comparison_results,
    'position_comparisons': position_comparisons,
    'per_bin_results': per_bin_results,
    'answer_position_results': answer_pos_results if 'answer_pos_results' in dir() else {},
    'asymmetry_results': asymmetry_results,
    'delta_forensics_summary': {
        'n_samples': len(delta_forensics),
        'samples': delta_forensics,
    },
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 12: Cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 12: Cleanup
import gc

print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9

del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()
gc.collect()

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")\
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

output_path = "/home/jupyter/research/directed_kvcache_v2/13_position_aware_priming.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
