#!/usr/bin/env python3
"""Build Exp 04: Position Sweep + Cache Surgery.

Two questions from Exp 03:
  1. Position offset: Why does S=4 >> S=20? Where's the peak?
  2. BOS exploitation: Can we prune more attention sinks? Does BOS
     matter in Phase A (shaping doc) or Phase B (distracting query)?

14 conditions, N=400, MS MARCO v1.1, Gemma 3 4B-PT.
"""

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/04", exist_ok=True)

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 04: Position Sweep + Cache Surgery

## Motivation

Exp 03 revealed that the "structural effect" was actually BOS removal (87%) +
position offset (53%) minus attention-enrichment harm (-40%). But two mysteries remain:

1. **Position sweet spot**: `pos_offset_4` (d=+0.78) vastly outperformed `pos_offset_20`
   (d=+0.43) and `bare_no_bos` (d=+0.46). Why? Where exactly is the peak?

2. **BOS mechanism**: Removing BOS from the Phase B cache was the biggest single factor.
   Can we exploit this further? Does BOS matter during Phase A (shaping doc representations)
   or only during Phase B (distracting query attention)?

## Conditions (14 total)

### Baselines
| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | Standard causal, BOS in cache |
| 2 | oracle | Real query as prefix — upper bound |

### Position sweep (BOS removed from cache, no prefix)
| # | S | Doc RoPE positions | BOS-to-doc distance |
|---|---|-------------------|-------------------|
| 3 | 1 | 1..D (= bare_no_bos) | 1 |
| 4 | 2 | 2..2+D | 2 |
| 5 | 4 | 4..4+D | 4 |
| 6 | 8 | 8..8+D | 8 |
| 7 | 16 | 16..16+D | 16 |
| 8 | 32 | 32..32+D | 32 |

### BOS mechanism isolation
| # | Condition | Phase A BOS? | Phase B BOS? | Doc positions |
|---|-----------|-------------|-------------|--------------|
| 9 | no_bos_at_all | NO | NO | 0..D |
| 10 | keep_bos_offset_4 | yes | **YES** | 4..4+D |

### Cache pruning (natural positions, selective removal)
| # | Condition | What's removed from cache | Purpose |
|---|-----------|--------------------------|---------|
| 11 | prune_first_1 | BOS + doc[0] | Are early doc tokens sinks too? |
| 12 | prune_first_3 | BOS + doc[0:3] | Deeper pruning |
| 13 | prune_first_5 | BOS + doc[0:5] | How far can we go? |
| 14 | prune_last_3 | BOS + doc[-3:] | Control (end tokens shouldn't be sinks) |

## Key comparisons

**Position sweep**: S=1 → 2 → 4 → 8 → 16 → 32 traces the full curve.

**BOS in Phase A vs Phase B**:
- bare vs bare_no_bos → Phase B BOS effect (removing from cache)
- bare_no_bos vs no_bos_at_all → Phase A BOS effect (removing from input)
- pos_offset_4 vs keep_bos_offset_4 → Phase B BOS effect at optimal offset

**Cache pruning depth**:
- prune_first_{1,3,5} vs bare_no_bos → do early doc tokens act as sinks?
- prune_last_3 → control (removing informative tokens should hurt)""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup and model loading
import os
os.umask(0o000)

import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 400
MODEL_NAME = "google/gemma-3-4b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp04")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device

print(f"Exp 04: Position Sweep + Cache Surgery")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
print(f"Vocab size: {getattr(text_cfg, 'vocab_size', 'N/A')}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
""")


# ===== Cell 3: Scoring functions =====
code(r"""# Cell 3: KV cache helpers and scoring functions

def slice_kv_cache(cache, start_idx):
    # Remove first start_idx entries from KV cache.
    from transformers import DynamicCache
    if isinstance(cache, DynamicCache):
        sliced = DynamicCache()
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys[:, :, start_idx:, :]
            v = cache.layers[i].values[:, :, start_idx:, :]
            sliced.update(k, v, i)
        return sliced
    else:
        return tuple(
            (k[:, :, start_idx:, :], v[:, :, start_idx:, :])
            for k, v in cache
        )


def prune_kv_cache_end(cache, n):
    # Remove last n entries from KV cache.
    from transformers import DynamicCache
    if isinstance(cache, DynamicCache):
        pruned = DynamicCache()
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys[:, :, :-n, :]
            v = cache.layers[i].values[:, :, :-n, :]
            pruned.update(k, v, i)
        return pruned
    else:
        return tuple(
            (k[:, :, :-n, :], v[:, :, :-n, :])
            for k, v in cache
        )


def score(doc_text, query_text, answer_text, prefix_text=None,
          position_offset=0, remove_bos=False,
          keep_bos_in_offset=False, no_bos_input=False,
          prune_first=0, prune_last=0):
    # Score NLL of answer tokens using two-phase KV cache.
    #
    # Modes:
    #   prefix_text: [BOS + prefix + \n + doc], prefix sliced from cache
    #   position_offset > 0: BOS at pos 0, doc at offset..offset+D
    #     BOS removed from cache unless keep_bos_in_offset=True
    #   no_bos_input: forward doc WITHOUT BOS token, no BOS in cache
    #   remove_bos: bare but BOS removed from cache
    #   prune_first/prune_last: additional doc token pruning from cache

    # --- Phase A: Conditioning ---
    if prefix_text is not None:
        prefix_ids = tokenizer(prefix_text + "\n", add_special_tokens=True,
                               truncation=True, max_length=512).input_ids
        doc_ids = tokenizer(doc_text, add_special_tokens=False,
                            truncation=True, max_length=1536).input_ids
        cond_ids = prefix_ids + doc_ids
        slice_start = len(prefix_ids)
        custom_pos = None
        phase_b_start = len(cond_ids)

    elif no_bos_input:
        # No BOS in input at all
        cond_ids = tokenizer(doc_text, add_special_tokens=False,
                             truncation=True, max_length=2048).input_ids
        slice_start = 0
        custom_pos = None
        phase_b_start = len(cond_ids)

    elif position_offset > 0:
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        n_doc = len(cond_ids) - 1
        pos_list = [0] + list(range(position_offset, position_offset + n_doc))
        custom_pos = torch.tensor([pos_list], dtype=torch.long, device=DEVICE)
        if keep_bos_in_offset:
            slice_start = 0
        else:
            slice_start = 1
        phase_b_start = position_offset + n_doc

    else:
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        slice_start = 1 if remove_bos else 0
        custom_pos = None
        phase_b_start = len(cond_ids)

    cond_tensor = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)

    fwd_kwargs = {'input_ids': cond_tensor, 'use_cache': True}
    if custom_pos is not None:
        fwd_kwargs['position_ids'] = custom_pos

    with torch.no_grad():
        phase_a = model(**fwd_kwargs)

    cache = phase_a.past_key_values
    del phase_a

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

    # Additional pruning
    if prune_first > 0:
        cache = slice_kv_cache(cache, prune_first)
    if prune_last > 0:
        cache = prune_kv_cache_end(cache, prune_last)

    # --- Phase B: Inference ---
    query_part_ids = tokenizer("\n" + query_text + "\n",
                               add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids

    if not answer_ids:
        del cache
        return 0.0

    phase_b_ids = query_part_ids + answer_ids
    phase_b_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=DEVICE)

    pos_ids = torch.arange(phase_b_start, phase_b_start + len(phase_b_ids),
                           device=DEVICE).unsqueeze(0)
    cache_position = torch.arange(phase_b_start, phase_b_start + len(phase_b_ids),
                                  device=DEVICE)

    with torch.no_grad():
        phase_b = model(
            input_ids=phase_b_tensor,
            past_key_values=cache,
            position_ids=pos_ids,
            cache_position=cache_position,
            use_cache=False,
        )

    logits = phase_b.logits
    n_query_part = len(query_part_ids)
    n_answer = len(answer_ids)

    answer_logits = logits[0, n_query_part - 1 : n_query_part - 1 + n_answer, :]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del cache, phase_b, logits, log_probs
    return mean_nll


print("Scoring function defined with extended modes.")
print(f"  position_offset: RoPE shift for doc tokens")
print(f"  keep_bos_in_offset: keep BOS in Phase B cache")
print(f"  no_bos_input: skip BOS token entirely")
print(f"  prune_first/prune_last: remove doc tokens from cache")
""")


# ===== Cell 4: Load data =====
code(r"""# Cell 4: Load MS MARCO data
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
for item in ds:
    if len(all_candidates) >= 3 * N_SAMPLES:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    if not answer:
        continue
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            all_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

print(f"Total candidates: {len(all_candidates)}")
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([len(s['query'].split()) for s in samples]):.1f}")
mean_doc_tokens = np.mean([len(tokenizer(s['passage'], add_special_tokens=False).input_ids)
                           for s in samples[:50]])
print(f"Mean passage tokens (first 50): {mean_doc_tokens:.0f}")
""")


# ===== Cell 5: Validation =====
code(r"""# Cell 5: Validate all scoring modes
print("=" * 70)
print("VALIDATION")
print("=" * 70)

s = samples[0]

# All modes run without error
print(f"\n--- All modes on sample 0 ---")
modes = [
    ("bare",              dict()),
    ("oracle",            dict(prefix_text=s['query'])),
    ("S=1 (bare_no_bos)", dict(position_offset=1)),
    ("S=4",               dict(position_offset=4)),
    ("S=8",               dict(position_offset=8)),
    ("S=32",              dict(position_offset=32)),
    ("no_bos_at_all",     dict(no_bos_input=True)),
    ("keep_bos_S=4",      dict(position_offset=4, keep_bos_in_offset=True)),
    ("prune_first_3",     dict(remove_bos=True, prune_first=3)),
    ("prune_last_3",      dict(remove_bos=True, prune_last=3)),
]

for name, kwargs in modes:
    nll = score(s['passage'], s['query'], s['answer'], **kwargs)
    delta = score(s['passage'], s['query'], s['answer']) - nll
    print(f"  {name:<20} NLL = {nll:.4f}  delta vs bare = {delta:+.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 14 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle',
    'pos_1', 'pos_2', 'pos_4', 'pos_8', 'pos_16', 'pos_32',
    'no_bos_at_all', 'keep_bos_offset_4',
    'prune_first_1', 'prune_first_3', 'prune_first_5', 'prune_last_3',
]

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            results = ckpt['results']
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
        'query_words': len(query.split()),
    }

    # Baselines
    result['nll_bare'] = score(passage, query, answer)
    result['nll_oracle'] = score(passage, query, answer, prefix_text=query)

    # Position sweep (BOS removed)
    for S in [1, 2, 4, 8, 16, 32]:
        result[f'nll_pos_{S}'] = score(passage, query, answer, position_offset=S)

    # BOS mechanism
    result['nll_no_bos_at_all'] = score(passage, query, answer, no_bos_input=True)
    result['nll_keep_bos_offset_4'] = score(passage, query, answer,
                                             position_offset=4,
                                             keep_bos_in_offset=True)

    # Cache pruning (natural positions, BOS removed)
    result['nll_prune_first_1'] = score(passage, query, answer,
                                         remove_bos=True, prune_first=1)
    result['nll_prune_first_3'] = score(passage, query, answer,
                                         remove_bos=True, prune_first=3)
    result['nll_prune_first_5'] = score(passage, query, answer,
                                         remove_bos=True, prune_first=5)
    result['nll_prune_last_3'] = score(passage, query, answer,
                                        remove_bos=True, prune_last=3)

    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = i - start_idx + 1
        eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 7: Results table =====
code(r"""# Cell 7: Results table
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

arrays = {}
for name in COND_NAMES:
    arrays[name] = np.array([r[f'nll_{name}'] for r in results])

bare = arrays['bare']
oracle = arrays['oracle']
oracle_delta_mean = (bare - oracle).mean()

print(f"\n  {'Condition':<22} {'NLL':>8} {'vs bare':>10} {'d':>8} "
      f"{'Win%':>8} {'p':>12} {'sig':>5} {'Recovery':>10}")
print(f"  {'-'*92}")

analysis = {}
for name in COND_NAMES:
    nlls = arrays[name]
    mean_nll = nlls.mean()

    if name == 'bare':
        print(f"  {name:<22} {mean_nll:>8.4f} {'--':>10} {'--':>8} "
              f"{'--':>8} {'--':>12} {'--':>5} {'--':>10}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        rec = diff.mean() / oracle_delta_mean * 100 if oracle_delta_mean > 0 else 0
        rec_str = f"{rec:>9.1f}%"

        print(f"  {name:<22} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {rec_str:>10}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            'recovery': float(rec),
        }
""")


# ===== Cell 8: Position sweep analysis =====
code(r"""# Cell 8: Position sweep analysis
print("=" * 70)
print("POSITION SWEEP")
print("=" * 70)

print(f"\n  All conditions: BOS removed from cache, no prefix tokens.")
print(f"  S = position offset (BOS at 0, doc at S..S+D)")

print(f"\n  {'S':>4} {'NLL':>8} {'d vs bare':>10} {'recovery':>10} "
      f"{'d vs S=1':>10} {'p vs S=1':>12}")
print(f"  {'-'*62}")

sweep_s = [1, 2, 4, 8, 16, 32]
sweep_d = []
for S in sweep_s:
    name = f'pos_{S}'
    nlls = arrays[name]
    d_b = cohens_d(bare - nlls)
    rec = (bare - nlls).mean() / oracle_delta_mean * 100
    diff_vs_1 = arrays['pos_1'] - nlls
    d_1 = cohens_d(diff_vs_1)
    _, p_1 = stats.ttest_1samp(diff_vs_1, 0)
    sig = '***' if p_1 < 0.001 else '**' if p_1 < 0.01 else '*' if p_1 < 0.05 else 'ns'
    print(f"  {S:>4} {nlls.mean():>8.4f} {d_b:>+10.4f} {rec:>9.1f}% "
          f"{d_1:>+10.4f} {p_1:>12.2e} {sig}")
    sweep_d.append(d_b)

# Find peak
peak_idx = np.argmax(sweep_d)
peak_s = sweep_s[peak_idx]
print(f"\n  Peak: S={peak_s} (d = {sweep_d[peak_idx]:+.4f})")

# Shape characterization
print(f"\n  Curve shape:")
if peak_idx == 0:
    print(f"  -> MONOTONICALLY DECREASING from S=1. Offset always hurts.")
elif peak_idx == len(sweep_s) - 1:
    print(f"  -> MONOTONICALLY INCREASING to S=32. Larger offset always better.")
else:
    print(f"  -> INVERTED-U with peak at S={peak_s}.")
    print(f"  -> Below peak: benefit increases with offset")
    print(f"  -> Above peak: benefit decreases (over-shifting)")

# Adjacent pairwise significance
print(f"\n  Adjacent pairwise tests (does increasing S help?):")
for i in range(len(sweep_s) - 1):
    s_a, s_b = sweep_s[i], sweep_s[i + 1]
    diff = arrays[f'pos_{s_a}'] - arrays[f'pos_{s_b}']
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    direction = "better" if d > 0 else "worse" if d < 0 else "same"
    print(f"  S={s_a:>2} → S={s_b:>2}: d = {d:+.4f} ({sig}) — S={s_b} is {direction}")
""")


# ===== Cell 9: BOS mechanism + cache surgery =====
code(r"""# Cell 9: BOS mechanism isolation and cache surgery
print("=" * 70)
print("BOS MECHANISM ISOLATION")
print("=" * 70)

# --- Phase A vs Phase B BOS effects ---
print(f"\n--- Where does BOS matter? ---\n")

# Phase B effect: bare → bare_no_bos (pos_1)
f_phase_b = bare - arrays['pos_1']
d_pb = cohens_d(f_phase_b)
_, p_pb = stats.ttest_1samp(f_phase_b, 0)
print(f"  Phase B (removing BOS from cache):")
print(f"    bare → pos_1 (bare_no_bos): d = {d_pb:+.4f}, p = {p_pb:.2e}")
print(f"    NLL: {bare.mean():.4f} → {arrays['pos_1'].mean():.4f}")

# Phase A effect: bare_no_bos → no_bos_at_all
f_phase_a = arrays['pos_1'] - arrays['no_bos_at_all']
d_pa = cohens_d(f_phase_a)
_, p_pa = stats.ttest_1samp(f_phase_a, 0)
sig_pa = '***' if p_pa < 0.001 else '**' if p_pa < 0.01 else '*' if p_pa < 0.05 else 'ns'
print(f"\n  Phase A (removing BOS from input):")
print(f"    pos_1 → no_bos_at_all: d = {d_pa:+.4f} ({sig_pa}), p = {p_pa:.2e}")
print(f"    NLL: {arrays['pos_1'].mean():.4f} → {arrays['no_bos_at_all'].mean():.4f}")
if d_pa > 0.05:
    print(f"    → BOS in Phase A HURTS doc representations")
elif d_pa < -0.05:
    print(f"    → BOS in Phase A HELPS doc representations")
else:
    print(f"    → BOS in Phase A has minimal effect on doc representations")

# Phase B BOS effect at optimal offset
print(f"\n  Phase B BOS effect at S=4:")
diff_keep = arrays['keep_bos_offset_4'] - arrays['pos_4']
d_keep = cohens_d(diff_keep)
_, p_keep = stats.ttest_1samp(diff_keep, 0)
sig_k = '***' if p_keep < 0.001 else '**' if p_keep < 0.01 else '*' if p_keep < 0.05 else 'ns'
print(f"    keep_bos_offset_4 NLL = {arrays['keep_bos_offset_4'].mean():.4f}")
print(f"    pos_4 (BOS removed) NLL = {arrays['pos_4'].mean():.4f}")
print(f"    Keeping BOS: d = {d_keep:+.4f} ({sig_k})")
if d_keep > 0.05:
    print(f"    → Even at S=4, BOS in cache HURTS Phase B")
elif d_keep < -0.05:
    print(f"    → At S=4, BOS in cache actually helps Phase B")
else:
    print(f"    → At S=4, BOS in cache makes no difference")

# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"CACHE PRUNING")
print(f"{'='*70}")

print(f"\n  All conditions: BOS removed, natural positions, doc tokens pruned.")
print(f"\n  {'Condition':<22} {'NLL':>8} {'d vs bare':>10} {'d vs pos_1':>12} {'p':>12}")
print(f"  {'-'*70}")
for name in ['pos_1', 'prune_first_1', 'prune_first_3', 'prune_first_5', 'prune_last_3']:
    nlls = arrays[name]
    d_b = cohens_d(bare - nlls)
    diff_vs_1 = arrays['pos_1'] - nlls
    d_1 = cohens_d(diff_vs_1)
    _, p_1 = stats.ttest_1samp(diff_vs_1, 0)
    sig = '***' if p_1 < 0.001 else '**' if p_1 < 0.01 else '*' if p_1 < 0.05 else 'ns'
    print(f"  {name:<22} {nlls.mean():>8.4f} {d_b:>+10.4f} {d_1:>+12.4f} {p_1:>12.2e} {sig}")

# Interpretation
d_pf1 = cohens_d(arrays['pos_1'] - arrays['prune_first_1'])
d_pf3 = cohens_d(arrays['pos_1'] - arrays['prune_first_3'])
d_pf5 = cohens_d(arrays['pos_1'] - arrays['prune_first_5'])
d_pl3 = cohens_d(arrays['pos_1'] - arrays['prune_last_3'])

print(f"\n  Interpretation:")
if d_pf1 > 0.05:
    print(f"  -> Removing doc[0] helps (d = {d_pf1:+.4f}): early tokens ARE sinks")
elif d_pf1 < -0.05:
    print(f"  -> Removing doc[0] hurts (d = {d_pf1:+.4f}): doc[0] is informative")
else:
    print(f"  -> Removing doc[0] neutral (d = {d_pf1:+.4f})")

if d_pl3 < -0.05:
    print(f"  -> Removing last 3 hurts (d = {d_pl3:+.4f}): confirms end tokens are informative")
elif d_pl3 > 0.05:
    print(f"  -> Removing last 3 helps?! (d = {d_pl3:+.4f}): unexpected")
else:
    print(f"  -> Removing last 3 neutral (d = {d_pl3:+.4f})")

# Best overall condition
print(f"\n  Best non-oracle conditions:")
non_oracle = {k: v for k, v in analysis.items()
              if k not in ('bare', 'oracle') and 'd' in v}
sorted_conds = sorted(non_oracle.items(), key=lambda x: x[1]['d'], reverse=True)
for name, info in sorted_conds[:5]:
    print(f"    {name:<22} d = {info['d']:+.4f} ({info['recovery']:.0f}% recovery)")
""")


# ===== Cell 10: Verdict + save =====
code(r"""# Cell 10: Verdict
print("=" * 70)
print("VERDICT — Exp 04: Position Sweep + Cache Surgery")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")

# Best condition
non_oracle = {k: v for k, v in analysis.items()
              if k not in ('bare', 'oracle') and 'd' in v}
best_name = max(non_oracle, key=lambda k: non_oracle[k]['d'])
best_d = non_oracle[best_name]['d']
best_rec = non_oracle[best_name]['recovery']

print(f"\n--- Key findings ---")
print(f"  Best condition: {best_name} (d = {best_d:+.4f}, {best_rec:.0f}% of oracle)")
print(f"  Oracle: d = {cohens_d(bare - oracle):+.4f}")

# Position peak
sweep_ds = [cohens_d(bare - arrays[f'pos_{S}']) for S in [1, 2, 4, 8, 16, 32]]
peak_s = [1, 2, 4, 8, 16, 32][np.argmax(sweep_ds)]
print(f"  Position peak: S={peak_s}")

# BOS summary
d_phase_b = cohens_d(bare - arrays['pos_1'])
d_phase_a = cohens_d(arrays['pos_1'] - arrays['no_bos_at_all'])
print(f"\n  BOS effects:")
print(f"    Phase B (cache removal): d = {d_phase_b:+.4f}")
print(f"    Phase A (input removal): d = {d_phase_a:+.4f}")

# All conditions ordered
print(f"\n--- All conditions (ranked by d) ---")
all_ranked = sorted(analysis.items(),
                    key=lambda x: x[1].get('d', -999), reverse=True)
for name, info in all_ranked:
    if name == 'bare':
        print(f"  {name:<22} NLL = {info['mean_nll']:.4f}  (baseline)")
    else:
        print(f"  {name:<22} NLL = {info['mean_nll']:.4f}  d = {info['d']:+.4f}")

# Save
final_results = {
    'experiment': 'v4_decoder_only_exp04_position_sweep_cache_surgery',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {k: v for k, v in analysis.items()},
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/04/04_position_sweep_cache_surgery.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
