#!/usr/bin/env python3
# Build Exp 10: Compression Robustness After Normalization Deconfounding.
#
# Exp 07 tested whether prefix-conditioned KV caches are more robust to
# compression (quantization + eviction).  Exps 08-09 revealed a confound:
# int8 quantization improves NLL because the absmax/qmax normalization step
# corrects pathological scale drift.  norm_roundtrip alone captures ~97%
# of the int8 benefit.
#
# This experiment separates normalization from compression: apply norm first
# (to all caches equally), THEN quantize.  The damage from norm->intX vs
# norm alone is the true compression cost, cleaned of the norm confound.
#
# Factorial: 3 Conditioning x 6 Treatment + 1 single-pass = 19 per sample
# Conditioning: bare, random_64, comprehend_64
# Treatment: bf16, norm, int8, int4, norm_int8, norm_int4
# + single_pass (ground truth)
#
# 4 datasets x 160 hard samples = 640 total
# Datasets: MS MARCO, SQuAD v2, DROP, HotpotQA
#
# SEED=42, SCORING_KEY='bos_retained_norm_deconfound_v10'

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/10", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 10: Compression Robustness After Normalization Deconfounding

## Motivation

Exp 07 tested whether prefix-conditioned KV caches are more robust to compression
(quantization + eviction). However, Exps 08-09 revealed a critical confound: int8
quantization **improves** NLL because the `absmax/qmax` normalization step corrects
pathological scale drift in two-phase caches. The `norm_roundtrip` alone captures ~97%
of the int8 benefit. This means Exp 07's "shielding" results conflate normalization
benefit with actual compression damage.

**Goal**: Separate normalization from compression and re-assess whether
prefix-conditioned caches are genuinely more robust to precision loss.

**Key idea**: Apply normalization first (to all caches equally), THEN quantize.
The damage from `norm -> intX` vs `norm` alone is the **true compression cost**,
cleaned of the normalization confound.

## Design

### Factorial: 3 Conditioning x 6 Treatment + 1 single-pass = 19 per sample

**Factor A — Conditioning (3 levels):**

| Level | Description |
|-------|-------------|
| `bare` | `[BOS] + doc` -> cache (no prefix) |
| `random_64` | Random tokens L=64 -> select + reposition (structural control) |
| `comprehend_64` | Comprehend instruction L=64 -> select + reposition |

**Factor B — Treatment (6 levels):**

| Level | Description | Purpose |
|-------|-------------|---------|
| `bf16` | No treatment (baseline) | Reference |
| `norm` | `norm_roundtrip_kv_cache(qmax=127)` only | Normalization benefit alone |
| `int8` | `quantize_kv_cache(8)` directly (replicates Exp 07) | Validation |
| `int4` | `quantize_kv_cache(4)` directly (replicates Exp 07) | Validation |
| `norm_int8` | `norm_roundtrip` then `quantize_kv_cache(8)` | Norm + compression |
| `norm_int4` | `norm_roundtrip` then `quantize_kv_cache(4)` | Norm + compression |

**Plus**: `single_pass` — full `[BOS]+doc+query+answer` in one forward pass (ground truth).

### Key Diagnostic Comparisons

1. **True compression damage**: `NLL(norm_intX) - NLL(norm)` per conditioning level
2. **Deconfounded shielding**: `damage(bare) - damage(prefixed)` — positive = prefix protects
3. **Normalization benefit**: `NLL(bf16) - NLL(norm)` per conditioning (replicates Exp 09)
4. **Exp 07 replication**: `bf16`, `int8`, `int4` should match Exp 07 within tolerance
5. **Residual structure**: Does `int8 ~ norm`? Does `norm_int8 ~ int8`?
6. **int4 asymmetry**: Does pre-normalization make bare tolerate int4?

### Datasets & Samples

4 datasets x 160 hard samples (same as Exp 07 for direct comparison):
MS MARCO, SQuAD v2, DROP, HotpotQA""")


# ===== Cell 2: Setup + Model + All Functions =====
code(r"""# Cell 2: Setup, model loading, and all functions
import os
os.umask(0o000)
import sys, json, time, gc
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d, win_rate, paired_ttest
from lib.data import count_words
from lib.cache import deep_copy_cache, make_prefix
from lib.rope import build_layer_inv_freqs, get_layer_types, select_kv_cache, reposition_kv_cache
from lib.quantization import simulated_quantize, quantize_kv_cache, norm_roundtrip_kv_cache

SEED = 42
N_SAMPLES = 400
HARD_FRAC = 0.40
N_HARD = int(N_SAMPLES * HARD_FRAC)  # 160
PREFIX_L = 64
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP07_DIR = Path("../../../results/decoder_only/exp07")

DATASETS = ['ms_marco', 'squad_v2', 'drop', 'hotpotqa']

CONDITIONING_LEVELS = ['bare', 'random_64', 'comprehend_64']
TREATMENT_LEVELS = ['bf16', 'norm', 'int8', 'int4', 'norm_int8', 'norm_int4']

DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
}

SCORING_KEY = 'bos_retained_norm_deconfound_v10'

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = model.get_input_embeddings().num_embeddings
cfg_vocab = getattr(text_cfg, 'vocab_size', None)
if cfg_vocab != VOCAB_SIZE:
    print(f"WARNING: config vocab_size={cfg_vocab} != embedding size={VOCAB_SIZE}")
    print(f"Using embedding size {VOCAB_SIZE} for random token generation")
N_LAYERS = len(get_layer_types(model))
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # 1023 for Gemma 3

NL = len(NEWLINE_IDS)
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL  # 765 (same as Exp 05/07)

special_ids = set(tokenizer.all_special_ids)

# Build RoPE helpers from lib
LAYER_INV_FREQS = build_layer_inv_freqs(model, DEVICE)
LAYER_TYPES = get_layer_types(model)

print(f"Exp 10: Compression Robustness After Normalization Deconfounding")
print(f"N_SAMPLES: {N_SAMPLES}, HARD_FRAC: {HARD_FRAC}, N_HARD: {N_HARD}")
print(f"PREFIX_L: {PREFIX_L}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}, N_LAYERS: {N_LAYERS}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print(f"Conditioning: {CONDITIONING_LEVELS}")
print(f"Treatments: {TREATMENT_LEVELS}")
print(f"Datasets: {DATASETS}")

# --- Instruction definitions ---
INSTRUCTIONS = {
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
}
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")


# ===================================================================
# EXPERIMENT-SPECIFIC FUNCTIONS
# ===================================================================

def encode_phase_a(doc_text, prefix_token_ids=None):
    # Phase A: encode document with optional prefix, return cache + metadata.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]

    if prefix_token_ids is not None:
        P = len(prefix_token_ids)
        _NL = len(NEWLINE_IDS)
        max_doc = SLIDING_CACHE_LIMIT - 1 - P - _NL
        if len(doc_ids) > max_doc:
            doc_ids = doc_ids[:max_doc]
        D = len(doc_ids)
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True, output_attentions=False)
        cache = pa.past_key_values
        del pa
        keep_indices = [0] + list(range(1 + P + _NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)
        old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos,
                                    LAYER_INV_FREQS, LAYER_TYPES, bos_start=0)
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True, output_attentions=False)
        cache = pa.past_key_values
        del pa

    return cache, D, doc_ids


def score_phase_b(cache, D_effective, query_text, answer_text):
    # Phase B: score query+answer against a pre-built cache.
    phase_b_start = D_effective + 1
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            use_cache=False,
        )

    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del pb
    return nll


def score_single_pass(doc_text, query_text, answer_text):
    # Full single-pass forward. Ground truth NLL.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]
    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    full_ids = [BOS_ID] + doc_ids + query_ids + answer_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([full_ids], device=DEVICE))

    n_ctx = 1 + len(doc_ids) + len(query_ids)
    logits = out.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out
    return nll


def treat_and_score(cache, D, treatment, query_text, answer_text):
    # Dispatcher: deep-copy cache, apply treatment, score Phase B.
    if treatment == 'bf16':
        return score_phase_b(cache, D, query_text, answer_text)

    c = deep_copy_cache(cache)

    if treatment == 'norm':
        norm_roundtrip_kv_cache(c, qmax=127)
    elif treatment == 'int8':
        quantize_kv_cache(c, 8)
    elif treatment == 'int4':
        quantize_kv_cache(c, 4)
    elif treatment == 'norm_int8':
        norm_roundtrip_kv_cache(c, qmax=127)
        quantize_kv_cache(c, 8)
    elif treatment == 'norm_int4':
        norm_roundtrip_kv_cache(c, qmax=127)
        quantize_kv_cache(c, 4)
    else:
        del c
        raise ValueError(f"Unknown treatment: {treatment}")

    nll = score_phase_b(c, D, query_text, answer_text)
    del c
    return nll


print(f"\nSetup complete.")
print(f"Functions: encode_phase_a, score_phase_b, score_single_pass, treat_and_score")
""")


# ===== Cell 3: Load Datasets + Hard Samples =====
code(r"""# Cell 3: Load 4 datasets + hard samples (same as Exp 07) + Exp 07 checkpoints
from datasets import load_dataset

print("=" * 70)
print("LOADING 4 DATASETS + HARD SAMPLES")
print("=" * 70)

hard_samples = {}   # ds_name -> list of hard sample dicts
all_samples = {}    # ds_name -> list of all N_SAMPLES sample dicts

# ================================================================
# MS MARCO (from Exp 02)
# ================================================================
print("\n--- MS MARCO ---")
assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES

msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])

print("  Reloading MS MARCO v1.1 for passage text...")
ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES:
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
            msmarco_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES]
msmarco_all = [msmarco_candidates[i] for i in indices]
del ds_msmarco, msmarco_candidates
gc.collect()

for i in range(min(20, N_SAMPLES)):
    assert msmarco_all[i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"
print("  MS MARCO alignment verified")

hs_msmarco = []
for idx in msmarco_hard_idx:
    s = dict(msmarco_all[idx])
    s['nll_bare'] = float(msmarco_bare[idx])
    s['original_idx'] = int(idx)
    hs_msmarco.append(s)
hard_samples['ms_marco'] = hs_msmarco
all_samples['ms_marco'] = msmarco_all
print(f"  MS MARCO: {N_HARD} hard samples")

del exp02_ckpt, exp02_results
gc.collect()

# ================================================================
# SQuAD 2.0
# ================================================================
print("\n--- SQuAD 2.0 ---")
ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")
squad_candidates = []
for item in ds_squad:
    answers = item.get('answers', {})
    answer_texts = answers.get('text', [])
    if not answer_texts:
        continue
    passage = item['context']
    query = item['question']
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        squad_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['squad_v2'])
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
all_samples['squad_v2'] = [squad_candidates[i] for i in sq_indices]
del ds_squad, squad_candidates
gc.collect()

# ================================================================
# DROP
# ================================================================
print("\n--- DROP ---")
ds_drop = load_dataset("ucinlp/drop", split="validation")
drop_candidates = []
for item in ds_drop:
    passage = item['passage']
    question = item['question']
    answers_spans = item.get('answers_spans', {})
    spans = answers_spans.get('spans', [])
    if not spans or not spans[0]:
        continue
    answer = spans[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        drop_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['drop'])
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES]
all_samples['drop'] = [drop_candidates[i] for i in drop_indices]
del ds_drop, drop_candidates
gc.collect()

# ================================================================
# HotpotQA
# ================================================================
print("\n--- HotpotQA ---")
ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
hotpot_candidates = []
for item in ds_hotpot:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    sf_titles = sf.get('title', [])
    sf_sent_ids = sf.get('sent_id', [])
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
    passage_parts = []
    for title, sid in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents and sid < len(title_to_sents[title]):
            passage_parts.append(title_to_sents[title][sid])
    if not passage_parts:
        continue
    passage = ' '.join(passage_parts)
    query = item['question']
    answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        hotpot_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['hotpotqa'])
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# ================================================================
# Load bare NLLs for hard selection (non-MS-MARCO datasets)
# ================================================================
print("\n--- Loading bare NLLs for hard selection ---")
for ds_name in ['squad_v2', 'drop', 'hotpotqa']:
    samples_ds = all_samples[ds_name]

    if ds_name in ('squad_v2', 'hotpotqa'):
        bare_path = EXP03_DIR / f"bare_{ds_name}.json"
    else:
        bare_path = EXP05_DIR / f"bare_{ds_name}.json"

    bare_ckpt = json.loads(bare_path.read_text())
    bare_nlls_all = bare_ckpt['bare_nlls']

    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in samples_ds[:len(saved_queries)]]
    assert saved_queries == current_queries, \
        f"{ds_name}: query alignment mismatch"

    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])

    hs = []
    for idx in h_idx:
        s = dict(samples_ds[idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        hs.append(s)
    hard_samples[ds_name] = hs
    print(f"  {ds_name}: {N_HARD} hard, mean bare NLL: {bare_arr[h_idx].mean():.4f}")

del bare_ckpt
gc.collect()

# ================================================================
# Load Exp 07 checkpoints for validation (Cell 5)
# ================================================================
print("\n--- Loading Exp 07 checkpoints for validation ---")
exp07_results = {}
for ds_name in DATASETS:
    ckpt_path = EXP07_DIR / f"checkpoint_{ds_name}.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        exp07_results[ds_name] = ckpt['results']
        print(f"  {ds_name}: loaded {len(ckpt['results'])} Exp 07 results")
        del ckpt
    else:
        print(f"  {ds_name}: Exp 07 checkpoint not found at {ckpt_path}")
gc.collect()

# Summary
print("\n" + "=" * 70)
print("Dataset loading summary:")
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    mean_bare = np.mean([s['nll_bare'] for s in hard_samples[ds_name]])
    print(f"  {ds_name}: {n_h} hard samples, mean bare NLL: {mean_bare:.3f}")
""")


# ===== Cell 4: Main Scoring Loop =====
code(r"""# Cell 4: Main scoring loop — 3 conditioning x 6 treatment + single_pass x 4 datasets x 160 samples
print("=" * 70)
print("MAIN SCORING LOOP")
n_per_sample = len(CONDITIONING_LEVELS) * len(TREATMENT_LEVELS) + 1  # 18 + 1 single_pass
print(f"  {len(DATASETS)} datasets x {N_HARD} samples x "
      f"({len(CONDITIONING_LEVELS)} cond x {len(TREATMENT_LEVELS)} treatments + 1 single_pass)")
print(f"  = {len(DATASETS) * N_HARD} samples, {n_per_sample} Phase B + {len(CONDITIONING_LEVELS)} Phase A per sample")
print("=" * 70)

# Pre-build prefix token IDs
comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

# Quick validation
print("\n--- Validation: bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"

nll_single_t = score_single_pass(doc_text_t, query_text_t, answer_text_t)
cache_t, D_t, _ = encode_phase_a(doc_text_t)
nll_twophase_t = score_phase_b(cache_t, D_t, query_text_t, answer_text_t)
del cache_t
diff_pct = abs(nll_single_t - nll_twophase_t) / nll_single_t * 100
print(f"  Single-pass: {nll_single_t:.6f}, Two-phase: {nll_twophase_t:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"

# Validate all treatments dispatch correctly
cache_v, D_v, _ = encode_phase_a(doc_text_t)
for trt in TREATMENT_LEVELS:
    nll_v = treat_and_score(cache_v, D_v, trt, query_text_t, answer_text_t)
    print(f"  {trt}: {nll_v:.4f}")
del cache_v
print("  PASSED all validation checks\n")

gc.collect()
torch.cuda.empty_cache()

# ================================================================
# Main loop
# ================================================================
all_results = {}

for ds_name in DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"

    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name} ({n_hard} hard samples)")
    print(f"{'='*70}")

    ds_results = []
    start_idx = 0

    # Resume from checkpoint
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if (ckpt.get('dataset') == ds_name and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('n_hard') == n_hard):
            saved_queries = [r['query'][:50] for r in ckpt.get('results', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                ds_results = ckpt['results']
                start_idx = len(ds_results)
                print(f"  Resuming from checkpoint: {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()

        # Reset RNG for reproducible random prefixes (same seed as Exp 07)
        np.random.seed(DS_SEEDS.get(ds_name, SEED) + 7000)
        pyrandom.seed(DS_SEEDS.get(ds_name, SEED) + 7000)

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Exp10 {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
            }

            # Generate random prefix for this sample (same RNG as Exp 07)
            rand_ids = []
            while len(rand_ids) < PREFIX_L:
                tid = np.random.randint(0, VOCAB_SIZE)
                if tid not in special_ids:
                    rand_ids.append(int(tid))
            rand_prefix = rand_ids[:PREFIX_L]

            # === Phase A: 3 conditioning passes ===
            conditioning_caches = {}
            conditioning_D = {}

            for cond in CONDITIONING_LEVELS:
                if cond == 'bare':
                    prefix_ids = None
                elif cond == 'random_64':
                    prefix_ids = rand_prefix
                elif cond == 'comprehend_64':
                    prefix_ids = comprehend_prefix
                else:
                    raise ValueError(f"Unknown conditioning: {cond}")

                cache, D, doc_ids = encode_phase_a(
                    s['passage'], prefix_token_ids=prefix_ids)

                conditioning_caches[cond] = cache
                conditioning_D[cond] = D
                result[f'D_{cond}'] = D

            # === Phase B: 6 treatment levels per conditioning ===
            for cond in CONDITIONING_LEVELS:
                cache = conditioning_caches[cond]
                D = conditioning_D[cond]

                for trt in TREATMENT_LEVELS:
                    key = f'nll_{cond}_{trt}'
                    nll = treat_and_score(
                        cache, D, trt,
                        s['query'], s['answer'])
                    result[key] = nll

            # === Single pass (ground truth) ===
            result['nll_single_pass'] = score_single_pass(
                s['passage'], s['query'], s['answer'])

            # Cleanup caches
            for cond in CONDITIONING_LEVELS:
                del conditioning_caches[cond]
            del conditioning_caches, conditioning_D
            gc.collect()
            torch.cuda.empty_cache()

            ds_results.append(result)

            # Checkpoint every 20 samples
            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'conditioning_levels': CONDITIONING_LEVELS,
                    'treatment_levels': TREATMENT_LEVELS,
                    'prefix_l': PREFIX_L,
                    'common_max_doc': COMMON_MAX_DOC,
                    'results': ds_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Scoring complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    all_results[ds_name] = ds_results

gc.collect()
torch.cuda.empty_cache()
print(f"\nAll scoring complete for {len(DATASETS)} datasets.")
""")


# ===== Cell 5: Validation Against Exp 07 =====
code(r"""# Cell 5: Validation against Exp 07 — bf16, int8, int4 should match within tolerance
print("=" * 70)
print("VALIDATION AGAINST EXP 07")
print("=" * 70)

SHARED_TREATMENTS = ['bf16', 'int8', 'int4']

for ds_name in DATASETS:
    if ds_name not in exp07_results:
        print(f"\n  {ds_name}: no Exp 07 data available, skipping")
        continue

    results_10 = all_results[ds_name]
    results_07 = exp07_results[ds_name]

    # Match samples by query text (both should be in same order)
    n = min(len(results_10), len(results_07))
    print(f"\n  {ds_name} ({n} samples):")

    # Verify query alignment
    mismatches = 0
    for j in range(n):
        if results_10[j]['query'][:50] != results_07[j]['query'][:50]:
            mismatches += 1
    if mismatches > 0:
        print(f"    WARNING: {mismatches} query mismatches!")
    else:
        print(f"    Query alignment: PASSED")

    for cond in CONDITIONING_LEVELS:
        for trt in SHARED_TREATMENTS:
            key_10 = f'nll_{cond}_{trt}'
            # Exp 07 used COMPRESSION_LEVELS naming: same key format
            key_07 = f'nll_{cond}_{trt}'

            nll_10 = np.array([r[key_10] for r in results_10[:n]])
            nll_07 = np.array([r[key_07] for r in results_07[:n]])
            diff = np.abs(nll_10 - nll_07)

            max_diff = diff.max()
            mean_diff = diff.mean()
            status = "PASSED" if max_diff < 0.01 else "WARNING"
            print(f"    {cond}_{trt}: max_diff={max_diff:.6f}, "
                  f"mean_diff={mean_diff:.6f} [{status}]")
""")


# ===== Cell 6: Per-Dataset Analysis =====
code(r"""# Cell 6: Per-dataset analysis
print("=" * 70)
print("PER-DATASET ANALYSIS")
print("=" * 70)

treatments_only = [t for t in TREATMENT_LEVELS if t != 'bf16']

# Build master NLL arrays
master_nll = {}
for ds_name in DATASETS:
    results = all_results[ds_name]
    master_nll[ds_name] = {}
    for cond in CONDITIONING_LEVELS:
        for trt in TREATMENT_LEVELS:
            key = f'nll_{cond}_{trt}'
            master_nll[ds_name][(cond, trt)] = np.array([r[key] for r in results])
    master_nll[ds_name]['single_pass'] = np.array([r['nll_single_pass'] for r in results])

per_dataset_analysis = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    # --- 3x6 Mean NLL table ---
    print(f"\n  Mean NLL (3 conditioning x 6 treatment):")
    header = f"  {'Conditioning':<18}"
    for trt in TREATMENT_LEVELS:
        header += f" {trt:>10}"
    print(header)
    print(f"  {'-'*(18 + 11 * len(TREATMENT_LEVELS))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for trt in TREATMENT_LEVELS:
            row += f" {master_nll[ds_name][(cond, trt)].mean():>10.4f}"
        print(row)
    sp = master_nll[ds_name]['single_pass']
    print(f"  {'single_pass':<18} {sp.mean():>10.4f}")

    # --- Delta NLL vs bf16 ---
    print(f"\n  Delta NLL (treatment - bf16; negative = improvement):")
    header = f"  {'Conditioning':<18}"
    for trt in treatments_only:
        header += f" {trt:>10}"
    print(header)
    print(f"  {'-'*(18 + 11 * len(treatments_only))}")

    delta_arrays = {}
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for trt in treatments_only:
            delta = master_nll[ds_name][(cond, trt)] - bf16
            delta_arrays[(cond, trt)] = delta
            row += f" {delta.mean():>+10.4f}"
        print(row)

    # --- Win rate and Cohen's d ---
    print(f"\n  Win rate (% where treatment < bf16) | Cohen's d:")
    header = f"  {'Conditioning':<18}"
    for trt in treatments_only:
        header += f" {trt:>10}"
    print(header)
    print(f"  {'-'*(18 + 11 * len(treatments_only))}")

    for cond in CONDITIONING_LEVELS:
        row_w = f"  {cond:<18}"
        row_d = f"  {'':18}"
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for trt in treatments_only:
            delta = master_nll[ds_name][(cond, trt)] - bf16
            w = np.mean(delta < 0)
            d = cohens_d(delta)
            row_w += f" {w:>9.1%}"
            row_d += f" {d:>+10.3f}"
        print(row_w)
        print(row_d)

    # --- True compression damage: NLL(norm_intX) - NLL(norm) ---
    print(f"\n  True compression damage (NLL(norm_intX) - NLL(norm)):")
    print(f"  {'Conditioning':<18} {'norm_int8':>10} {'norm_int4':>10}")
    print(f"  {'-'*40}")

    ds_analysis = {'damage': {}, 'shielding': {}}
    for cond in CONDITIONING_LEVELS:
        norm_arr = master_nll[ds_name][(cond, 'norm')]
        dmg_int8 = master_nll[ds_name][(cond, 'norm_int8')] - norm_arr
        dmg_int4 = master_nll[ds_name][(cond, 'norm_int4')] - norm_arr
        print(f"  {cond:<18} {dmg_int8.mean():>+10.4f} {dmg_int4.mean():>+10.4f}")
        ds_analysis['damage'][(cond, 'int8')] = dmg_int8
        ds_analysis['damage'][(cond, 'int4')] = dmg_int4

    # --- Deconfounded shielding ---
    print(f"\n  Deconfounded shielding (damage_bare - damage_prefixed; positive = prefix shields):")
    print(f"  {'Prefix':<18} {'int8_shield':>12} {'d':>7} {'p':>10} "
          f"{'int4_shield':>12} {'d':>7} {'p':>10}")
    print(f"  {'-'*78}")

    for cond in ['random_64', 'comprehend_64']:
        bare_dmg_int8 = ds_analysis['damage'][('bare', 'int8')]
        cond_dmg_int8 = ds_analysis['damage'][(cond, 'int8')]
        shield_int8 = bare_dmg_int8 - cond_dmg_int8
        d_int8 = cohens_d(shield_int8)
        _, p_int8 = paired_ttest(shield_int8)

        bare_dmg_int4 = ds_analysis['damage'][('bare', 'int4')]
        cond_dmg_int4 = ds_analysis['damage'][(cond, 'int4')]
        shield_int4 = bare_dmg_int4 - cond_dmg_int4
        d_int4 = cohens_d(shield_int4)
        _, p_int4 = paired_ttest(shield_int4)

        print(f"  {cond:<18} {shield_int8.mean():>+12.4f} {d_int8:>+7.3f} {p_int8:>10.2e} "
              f"{shield_int4.mean():>+12.4f} {d_int4:>+7.3f} {p_int4:>10.2e}")

        ds_analysis['shielding'][(cond, 'int8')] = {
            'mean': float(shield_int8.mean()), 'd': float(d_int8), 'p': float(p_int8),
        }
        ds_analysis['shielding'][(cond, 'int4')] = {
            'mean': float(shield_int4.mean()), 'd': float(d_int4), 'p': float(p_int4),
        }

    per_dataset_analysis[ds_name] = ds_analysis

gc.collect()
""")


# ===== Cell 7: Cross-Dataset Meta-Analysis =====
code(r"""# Cell 7: Cross-dataset meta-analysis
print("=" * 70)
print("CROSS-DATASET META-ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Grand mean NLL by conditioning x treatment
# ================================================================
print("\n--- PART 1: Grand Mean NLL (pooled across 4 datasets) ---")
header = f"  {'Conditioning':<18}"
for trt in TREATMENT_LEVELS:
    header += f" {trt:>10}"
header += f" {'single_pass':>12}"
print(header)
print(f"  {'-'*(18 + 11 * len(TREATMENT_LEVELS) + 13)}")

for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    for trt in TREATMENT_LEVELS:
        means = [master_nll[ds][(cond, trt)].mean() for ds in DATASETS]
        row += f" {np.mean(means):>10.4f}"
    sp_means = [master_nll[ds]['single_pass'].mean() for ds in DATASETS]
    row += f" {np.mean(sp_means):>12.4f}"
    print(row)

# ================================================================
# PART 2: Normalization benefit (bf16 - norm)
# ================================================================
print(f"\n--- PART 2: Normalization Benefit (bf16 - norm; positive = norm helps) ---")
print(f"  {'Dataset':<12}", end="")
for cond in CONDITIONING_LEVELS:
    print(f" {cond:>15}", end="")
print()
print(f"  {'-'*(12 + 16 * len(CONDITIONING_LEVELS))}")

for ds_name in DATASETS:
    row = f"  {ds_name:<12}"
    for cond in CONDITIONING_LEVELS:
        benefit = (master_nll[ds_name][(cond, 'bf16')] - master_nll[ds_name][(cond, 'norm')]).mean()
        row += f" {benefit:>+15.4f}"
    print(row)

row = f"  {'GRAND MEAN':<12}"
for cond in CONDITIONING_LEVELS:
    benefits = [(master_nll[ds][(cond, 'bf16')] - master_nll[ds][(cond, 'norm')]).mean() for ds in DATASETS]
    row += f" {np.mean(benefits):>+15.4f}"
print(row)

# ================================================================
# PART 3: True compression damage (pooled)
# ================================================================
print(f"\n--- PART 3: True Compression Damage (NLL(norm_intX) - NLL(norm), pooled) ---")
print(f"  {'Conditioning':<18} {'int8_damage':>12} {'int4_damage':>12}")
print(f"  {'-'*44}")

for cond in CONDITIONING_LEVELS:
    dmg_int8_all = []
    dmg_int4_all = []
    for ds_name in DATASETS:
        norm = master_nll[ds_name][(cond, 'norm')]
        dmg_int8_all.append((master_nll[ds_name][(cond, 'norm_int8')] - norm).mean())
        dmg_int4_all.append((master_nll[ds_name][(cond, 'norm_int4')] - norm).mean())
    print(f"  {cond:<18} {np.mean(dmg_int8_all):>+12.4f} {np.mean(dmg_int4_all):>+12.4f}")

# ================================================================
# PART 4: Deconfounded shielding meta-analysis
# ================================================================
print(f"\n--- PART 4: Deconfounded Shielding Meta-Analysis ---")
print(f"  damage = NLL(norm_intX) - NLL(norm)")
print(f"  shielding = damage(bare) - damage(prefixed)")
print()

meta_shielding = {}
for nbits_label in ['int8', 'int4']:
    print(f"  {nbits_label}:")
    print(f"  {'Prefix':<18} {'pooled_d':>9} {'SE':>8} {'z':>8} "
          f"{'p':>10} {'95% CI':>16} {'sig':>4}")
    print(f"  {'-'*76}")

    for cond in ['random_64', 'comprehend_64']:
        ds_effects = []
        for ds_name in DATASETS:
            norm = master_nll[ds_name][('bare', 'norm')]
            norm_intX_bare = master_nll[ds_name][('bare', f'norm_{nbits_label}')]
            dmg_bare = norm_intX_bare - norm

            norm_p = master_nll[ds_name][(cond, 'norm')]
            norm_intX_p = master_nll[ds_name][(cond, f'norm_{nbits_label}')]
            dmg_pref = norm_intX_p - norm_p

            shield = dmg_bare - dmg_pref
            n_s = len(shield)
            d = cohens_d(shield)
            se = np.sqrt(1.0 / n_s + d ** 2 / (2.0 * n_s))
            ds_effects.append((d, se, n_s, ds_name))

        weights = [1.0 / (se ** 2) for _, se, _, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        z = pooled_d / pooled_se if pooled_se > 0 else 0.0
        p = 2 * stats.norm.sf(abs(z))
        ci_lo = pooled_d - 1.96 * pooled_se
        ci_hi = pooled_d + 1.96 * pooled_se
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        print(f"  {cond:<18} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
              f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4}")

        meta_shielding[f'{cond}_{nbits_label}'] = {
            'pooled_d': float(pooled_d), 'se': float(pooled_se),
            'z': float(z), 'p': float(p),
            'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
            'per_dataset': {ds: float(d) for d, _, _, ds in ds_effects},
        }
    print()

# ================================================================
# PART 5: Structural vs semantic decomposition
# ================================================================
print(f"--- PART 5: Structural vs Semantic Shield (deconfounded) ---")
print(f"  {'Quant':<8} {'struct_d':>10} {'semantic_d':>11} "
      f"{'ratio':>7} {'interpretation':>20}")
print(f"  {'-'*62}")

for nbits_label in ['int8', 'int4']:
    struct_d = meta_shielding.get(f'random_64_{nbits_label}', {}).get('pooled_d', 0)
    semantic_d = meta_shielding.get(f'comprehend_64_{nbits_label}', {}).get('pooled_d', 0)
    ratio = struct_d / semantic_d if abs(semantic_d) > 0.001 else float('inf')

    if struct_d > 0.1 and semantic_d > 0.1:
        interp = "structural" if ratio > 0.8 else ("mixed" if ratio > 0.3 else "semantic")
    elif semantic_d > 0.1:
        interp = "semantic"
    elif struct_d > 0.1:
        interp = "structural"
    else:
        interp = "none"

    print(f"  {nbits_label:<8} {struct_d:>+10.4f} {semantic_d:>+11.4f} "
          f"{ratio:>7.2f} {interp:>20}")

# ================================================================
# PART 6: int4 asymmetry analysis
# ================================================================
print(f"\n--- PART 6: int4 Asymmetry — Does pre-normalization rescue bare? ---")
print(f"  Compare NLL(bare, int4) vs NLL(bare, norm_int4):")
print(f"  If norm_int4 < int4: pre-normalization makes int4 more tolerable")
print(f"  If norm_int4 > int4: int4 already subsumes the normalization benefit")
print()
print(f"  {'Dataset':<12} {'bare_int4':>10} {'bare_norm_int4':>15} {'diff':>8} {'interp':>15}")
print(f"  {'-'*65}")

for ds_name in DATASETS:
    int4 = master_nll[ds_name][('bare', 'int4')].mean()
    norm_int4 = master_nll[ds_name][('bare', 'norm_int4')].mean()
    diff = norm_int4 - int4
    interp = "norm helps" if diff < -0.001 else ("redundant" if abs(diff) <= 0.001 else "norm hurts")
    print(f"  {ds_name:<12} {int4:>10.4f} {norm_int4:>15.4f} {diff:>+8.4f} {interp:>15}")

# Same for comprehend_64
print(f"\n  Compare NLL(comprehend_64, int4) vs NLL(comprehend_64, norm_int4):")
print(f"  {'Dataset':<12} {'comp_int4':>10} {'comp_norm_int4':>15} {'diff':>8}")
print(f"  {'-'*48}")

for ds_name in DATASETS:
    int4 = master_nll[ds_name][('comprehend_64', 'int4')].mean()
    norm_int4 = master_nll[ds_name][('comprehend_64', 'norm_int4')].mean()
    diff = norm_int4 - int4
    print(f"  {ds_name:<12} {int4:>10.4f} {norm_int4:>15.4f} {diff:>+8.4f}")

# ================================================================
# PART 7: Residual structure — does int8 ≈ norm?
# ================================================================
print(f"\n--- PART 7: Residual Structure ---")
print(f"  Does int8 ≈ norm? (expected from Exp 09: ~97% overlap)")
print(f"  Does norm_int8 ≈ int8? (if norm captures most of it, combined ≈ int8 alone)")
print()
print(f"  {'Dataset':<12} {'bare_int8':>10} {'bare_norm':>10} {'bare_norm_i8':>13} "
      f"{'int8≈norm':>10} {'ni8≈int8':>10}")
print(f"  {'-'*70}")

for ds_name in DATASETS:
    int8 = master_nll[ds_name][('bare', 'int8')].mean()
    norm = master_nll[ds_name][('bare', 'norm')].mean()
    norm_int8 = master_nll[ds_name][('bare', 'norm_int8')].mean()
    print(f"  {ds_name:<12} {int8:>10.4f} {norm:>10.4f} {norm_int8:>13.4f} "
          f"{abs(int8 - norm):>10.4f} {abs(norm_int8 - int8):>10.4f}")
""")


# ===== Cell 8: Normalization x Compression Interaction =====
code(r"""# Cell 8: Normalization x compression interaction — 2x2 factorial + gap recovery
print("=" * 70)
print("NORMALIZATION x COMPRESSION INTERACTION")
print("=" * 70)

# ================================================================
# PART 1: 2x2 factorial for each quantization level
# ================================================================
# Factors: norm (yes/no) x quantize (yes/no)
# Cells: bf16 (no/no), norm (yes/no), intX (no/yes), norm_intX (yes/yes)
# Interaction = (norm_intX - bf16) - (norm - bf16) - (intX - bf16)
#             = norm_intX - norm - intX + bf16

interaction_results = {}

for nbits_label, nbits_trt, norm_nbits_trt in [('int8', 'int8', 'norm_int8'),
                                                 ('int4', 'int4', 'norm_int4')]:
    print(f"\n--- 2x2 Factorial: norm x {nbits_label} ---")

    for ds_name in DATASETS:
        print(f"\n  {ds_name.upper()}:")

        ds_interaction = {}
        for cond in CONDITIONING_LEVELS:
            bf16 = master_nll[ds_name][(cond, 'bf16')]
            norm = master_nll[ds_name][(cond, 'norm')]
            intX = master_nll[ds_name][(cond, nbits_trt)]
            norm_intX = master_nll[ds_name][(cond, norm_nbits_trt)]

            norm_effect = (bf16 - norm).mean()     # improvement from norm alone
            quant_effect = (bf16 - intX).mean()    # improvement from quant alone
            combined = (bf16 - norm_intX).mean()   # improvement from both
            interaction = combined - norm_effect - quant_effect  # interaction term

            # Interpretation
            if abs(interaction) < 0.005:
                interp = "additive"
            elif interaction > 0:
                interp = "synergistic"
            else:
                interp = "redundant"

            print(f"    {cond:<18} norm={norm_effect:+.4f}  quant={quant_effect:+.4f}  "
                  f"combined={combined:+.4f}  interaction={interaction:+.4f}  [{interp}]")

            if norm_intX.mean() > norm.mean():
                extra_dmg = (norm_intX - norm).mean()
                print(f"    {'':18} -> quantization after norm causes {extra_dmg:+.4f} extra damage")

            ds_interaction[(cond, nbits_label)] = {
                'norm_effect': float(norm_effect),
                'quant_effect': float(quant_effect),
                'combined': float(combined),
                'interaction': float(interaction),
                'interp': interp,
            }

        interaction_results[ds_name] = interaction_results.get(ds_name, {})
        interaction_results[ds_name].update(ds_interaction)

# ================================================================
# PART 2: Grand interaction summary
# ================================================================
print(f"\n{'='*70}")
print("GRAND INTERACTION SUMMARY (mean across 4 datasets)")
print(f"{'='*70}")

for nbits_label, nbits_trt, norm_nbits_trt in [('int8', 'int8', 'norm_int8'),
                                                 ('int4', 'int4', 'norm_int4')]:
    print(f"\n  {nbits_label}:")
    print(f"  {'Conditioning':<18} {'norm':>8} {'quant':>8} {'combined':>10} {'interaction':>12}")
    print(f"  {'-'*60}")

    for cond in CONDITIONING_LEVELS:
        norm_effs = []
        quant_effs = []
        combined_effs = []
        interactions = []
        for ds_name in DATASETS:
            bf16 = master_nll[ds_name][(cond, 'bf16')]
            norm = master_nll[ds_name][(cond, 'norm')]
            intX = master_nll[ds_name][(cond, nbits_trt)]
            norm_intX = master_nll[ds_name][(cond, norm_nbits_trt)]
            norm_effs.append((bf16 - norm).mean())
            quant_effs.append((bf16 - intX).mean())
            combined_effs.append((bf16 - norm_intX).mean())
            interactions.append((bf16 - norm_intX).mean() - (bf16 - norm).mean() - (bf16 - intX).mean())
        print(f"  {cond:<18} {np.mean(norm_effs):>+8.4f} {np.mean(quant_effs):>+8.4f} "
              f"{np.mean(combined_effs):>+10.4f} {np.mean(interactions):>+12.4f}")

# ================================================================
# PART 3: Gap recovery — fraction of bf16-single_pass gap closed
# ================================================================
print(f"\n{'='*70}")
print("GAP RECOVERY: % of (bf16 - single_pass) gap closed by each treatment")
print(f"{'='*70}")

treatments_only = [t for t in TREATMENT_LEVELS if t != 'bf16']

# Per-dataset
for ds_name in DATASETS:
    print(f"\n  {ds_name.upper()}:")
    sp = master_nll[ds_name]['single_pass']

    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        gap = (bf16 - sp).mean()
        row = f"    {cond:<18} gap={gap:+.4f} |"
        for trt in treatments_only:
            corrected = master_nll[ds_name][(cond, trt)]
            recovery = (bf16 - corrected).mean()
            pct = recovery / (gap + 1e-10) * 100
            row += f" {trt}={pct:+.0f}%"
        print(row)

# Grand summary
print(f"\n  GRAND MEAN:")
for cond in CONDITIONING_LEVELS:
    row = f"    {cond:<18} |"
    for trt in treatments_only:
        pcts = []
        for ds_name in DATASETS:
            bf16 = master_nll[ds_name][(cond, 'bf16')]
            sp = master_nll[ds_name]['single_pass']
            gap = (bf16 - sp).mean()
            corrected = master_nll[ds_name][(cond, trt)]
            recovery = (bf16 - corrected).mean()
            pcts.append(recovery / (gap + 1e-10) * 100)
        row += f" {trt}={np.mean(pcts):+.0f}%"
    print(row)
""")


# ===== Cell 9: Save results.json =====
code(r"""# Cell 9: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Per-dataset summary statistics
per_ds_summary = {}
treatments_only = [t for t in TREATMENT_LEVELS if t != 'bf16']

for ds_name in DATASETS:
    ds_data = {
        'n': len(all_results[ds_name]),
        'mean_nll': {},
        'delta_nll': {},
        'win_rate': {},
        'cohens_d': {},
        'true_damage': {},
        'deconfounded_shielding': {},
    }

    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for trt in TREATMENT_LEVELS:
            key = f'{cond}_{trt}'
            arr = master_nll[ds_name][(cond, trt)]
            ds_data['mean_nll'][key] = float(arr.mean())
            if trt != 'bf16':
                delta = arr - bf16
                ds_data['delta_nll'][key] = float(delta.mean())
                ds_data['win_rate'][key] = float(np.mean(delta < 0))
                ds_data['cohens_d'][key] = float(cohens_d(delta))

    ds_data['mean_nll']['single_pass'] = float(master_nll[ds_name]['single_pass'].mean())

    # True compression damage
    for cond in CONDITIONING_LEVELS:
        norm = master_nll[ds_name][(cond, 'norm')]
        for nbits in ['int8', 'int4']:
            dmg = master_nll[ds_name][(cond, f'norm_{nbits}')] - norm
            ds_data['true_damage'][f'{cond}_{nbits}'] = float(dmg.mean())

    # Deconfounded shielding
    for cond in ['random_64', 'comprehend_64']:
        for nbits in ['int8', 'int4']:
            bare_dmg = master_nll[ds_name][('bare', f'norm_{nbits}')] - master_nll[ds_name][('bare', 'norm')]
            pref_dmg = master_nll[ds_name][(cond, f'norm_{nbits}')] - master_nll[ds_name][(cond, 'norm')]
            shield = bare_dmg - pref_dmg
            d = cohens_d(shield)
            _, p = paired_ttest(shield)
            ds_data['deconfounded_shielding'][f'{cond}_{nbits}'] = {
                'mean': float(shield.mean()), 'd': float(d), 'p': float(p),
            }

    # Interaction data
    ds_data['interaction'] = {}
    for key, val in interaction_results.get(ds_name, {}).items():
        ds_data['interaction'][f'{key[0]}_{key[1]}'] = val

    per_ds_summary[ds_name] = ds_data

final_results = {
    'experiment': 'exp10_norm_deconfound',
    'model': MODEL_NAME,
    'scoring': SCORING_KEY,
    'seed': SEED,
    'n_samples_per_dataset': N_SAMPLES,
    'n_hard_per_dataset': N_HARD,
    'hard_fraction': HARD_FRAC,
    'prefix_length': PREFIX_L,
    'common_max_doc': COMMON_MAX_DOC,
    'conditioning_levels': CONDITIONING_LEVELS,
    'treatment_levels': TREATMENT_LEVELS,
    'datasets': DATASETS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'per_dataset_summary': per_ds_summary,
    'meta_shielding': meta_shielding,
    'per_sample_results': {ds: all_results[ds] for ds in DATASETS},
}

results_path = RESULTS_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {results_path}")
print(f"File size: {results_path.stat().st_size / 1024:.1f} KB")

# Also save compact summary without per-sample data
summary = {k: v for k, v in final_results.items() if k != 'per_sample_results'}
summary_path = RESULTS_DIR / 'summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {summary_path}")

print("\nDone!")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/10/10_norm_deconfound.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
