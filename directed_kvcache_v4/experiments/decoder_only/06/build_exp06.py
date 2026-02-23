#!/usr/bin/env python3
# Build Exp 06: Hero Run — Seven New Reasoning-Heavy Datasets (Expanded).
#
# Exp 05 found DROP (discrete reasoning) as a standout dataset with d=+0.914 at
# L=64 — the largest effect in any decoder-only experiment. BoolQ (yes/no answers)
# showed the opposite: ALL prefixes hurt. We hypothesized that complex reasoning
# tasks benefit most from prefix priming.
#
# Goal: Test 7 new reasoning-heavy datasets to find more DROP-like standout
# performers and build a 14-dataset meta-analysis combining these with the 7
# existing datasets from Exp 05.
#
# New datasets: GSM8K, QuALITY, MultiRC, ROPES, Quoref, ReCoRD, RACE-middle
#
# Expanded design vs original plan:
#   - 7 new datasets (was 5: added ReCoRD, RACE-middle)
#   - N_SAMPLES=500, N_HARD=200 (was 400/160)
#   - PREFIX_LENGTHS=[8,16,32,64,128,256] (was [16,32,64,128,256]: added L=8)
#   - 4 instructions (was 2: added classify, extract_claims from Exp 04)
#
# SEED=42, SCORING_KEY='bos_retained_token_matched_v06'

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/06", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 06: Hero Run — Seven New Reasoning-Heavy Datasets

## Motivation

Exp 05 found DROP (discrete reasoning) as a standout dataset with d=+0.914 at L=64 —
the largest effect in any decoder-only experiment. BoolQ (yes/no answers) showed the
opposite: ALL prefixes hurt. We hypothesized that complex reasoning tasks benefit most
from prefix priming.

**Goal:** Test 7 new reasoning-heavy datasets to find more DROP-like standout performers
and build a 14-dataset meta-analysis combining these with the 7 existing datasets from Exp 05.

## New Datasets

| Dataset | HF ID | Split | Passage | Query | Answer | Mean ans words |
|---------|-------|-------|---------|-------|--------|---------------|
| **GSM8K** | `openai/gsm8k`, `main` | test | `question` (math problem) | `"What is the answer?"` (fixed) | Number after `####` | 1.0 |
| **QuALITY** | `tasksource/QuALITY` | validation | `article` (~4106w, truncated) | `question` | `options[gold_label-1]` | MC 4-opt |
| **MultiRC** | `aps/super_glue`, `multirc` | validation | `paragraph` (~255w) | `question` | First `answer` with `label=1` | ~6 |
| **ROPES** | `allenai/ropes` | validation | `background`+`\n`+`situation` | `question` | `answers['text'][0]` | 1.4 |
| **Quoref** | `nc33/multispan_quoref` | validation | `context` (~332w) | `question` | `answers['text'][0]` | 1.7 |
| **ReCoRD** | `aps/super_glue`, `record` | validation | `passage` | query (fill @placeholder) | `answers[0]` | ~2 |
| **RACE-mid** | `race`, `middle` | test | `article` | `question` | `options[correct_idx]` | MC 4-opt |

## Risk Notes

- GSM8K: answers ALL single numbers — high BoolQ-like risk, but complex reasoning like DROP
- ROPES: very short answers (mean 1.4w) — similar BoolQ concern
- QuALITY: articles truncated from ~4100w to ~765 tokens (~20%) — relative comparison still valid
- MultiRC: `gold_label` 1-indexed; group by `(paragraph, question)`; some pairs have no positive label
- Quoref: `allenai/quoref` deprecated, using `nc33/multispan_quoref`
- ReCoRD: query contains `@placeholder` — use raw query as-is for NLL scoring
- RACE-middle: same MC structure as RACE-high; tests whether easier difficulty level shows same pattern

## Design

- **Model**: google/gemma-3-12b-it, BF16, A100-SXM4-40GB
- **SCORING\_KEY**: `'bos_retained_token_matched_v06'`
- **PREFIX\_LENGTHS**: `[8, 16, 32, 64, 128, 256]` (adds L=8 vs Exp 05)
- **COMMON\_MAX\_DOC**: 765 (= 1023 - 1 - 256 - 1, same as Exp 05)
- **N\_SAMPLES**: 500 per dataset, **N\_HARD**: 200 (top 40% by bare NLL)
- **Instructions**: `comprehend`, `extract_general`, `classify`, `extract_claims` (4 instructions from Exp 04)
- **Conditions**: bare, random\_tokens, comprehend, extract\_general, classify, extract\_claims, scrambled\_comprehend
- **QuALITY MC**: Score all 4 options under all conditions; accuracy = argmin(NLL) == correct\_idx

## Key Hypotheses

1. **MultiRC** (multi-sentence reasoning, ~6-word answers) is the strongest DROP-like candidate
2. **GSM8K** — the critical test: complex reasoning (like DROP) but single-number answers (like BoolQ)
3. **ROPES** — causal reasoning but very short answers (1.4w). Likely BoolQ-like.
4. **QuALITY** — despite severe truncation, MC accuracy should improve with prefix
5. **Quoref** — coreference resolution may benefit from "comprehend" priming
6. **ReCoRD** — entity-level comprehension may benefit differently from extractive QA
7. **RACE-middle** — easier exam items may show weaker prefix effect than RACE-high
8. **classify/extract\_claims** — Exp 04 showed classify was decent (non-extraction), extract\_claims adds data-focused priming

## Scoring Budget

| Phase | Scorings | Est. time |
|-------|----------|-----------|
| Bare scoring (7 x 500) | 3,500 | ~47 min |
| Phase A Q-matched (7 x 200 x 6) | 8,400 | ~112 min |
| Phase B fixed-length (7 x 200 x 37) | 51,800 | ~690 min |
| QuALITY MC (200 x 4 x 46) | 36,800 | ~491 min |
| **Total** | **100,500** | **~22 hours** |""")


# ===== Cell 2: Setup + model loading + scoring functions =====
code(r"""# Cell 2: Setup, model loading, and scoring functions
import os
os.umask(0o000)
import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d
from lib.data import count_words

SEED = 42
N_SAMPLES = 500      # per dataset (expanded from 400)
HARD_FRAC = 0.40     # top 40% by bare NLL
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp06")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP05_DIR = Path("../../../results/decoder_only/exp05")

# 7 new datasets for this experiment
NEW_DATASETS = ['gsm8k', 'quality', 'multirc', 'ropes', 'quoref', 'record', 'race_middle']
# 7 existing datasets from Exp 05
OLD_DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'race_high']
# All 14 datasets for meta-analysis
ALL_DATASETS = OLD_DATASETS + NEW_DATASETS

# Phase B prefix lengths — adds L=8 vs Exp 05
PREFIX_LENGTHS = [8, 16, 32, 64, 128, 256]

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
rope_params = getattr(text_cfg, 'rope_parameters', {})
layer_types = getattr(text_cfg, 'layer_types', [])
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # 1023 for Gemma 3

# Common max doc length for Phase B: use L=256 (max prefix) to ensure all fit
NL = len(NEWLINE_IDS)
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - max(PREFIX_LENGTHS) - NL  # 765 for Gemma 3

N_HARD = int(N_SAMPLES * HARD_FRAC)

print(f"Exp 06: Hero Run — Seven New Reasoning-Heavy Datasets (Expanded)")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"N_SAMPLES: {N_SAMPLES} per dataset, HARD_FRAC: {HARD_FRAC}, N_HARD: {N_HARD}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens (for Phase B)")
print(f"Prefix lengths: {PREFIX_LENGTHS}")

# --- RoPE repositioning helpers ---
def build_layer_inv_freqs():
    inv_freqs = {}
    for lt, params in rope_params.items():
        theta = params.get('rope_theta', 10000.0)
        dim = text_cfg.head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
        inv_freqs[lt] = inv_freq
    return inv_freqs

LAYER_INV_FREQS = build_layer_inv_freqs()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def select_kv_cache(cache, indices):
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def reposition_kv_cache(cache, old_positions, new_positions, bos_start=0):
    delta = new_positions - old_positions
    for L in range(len(cache.layers)):
        lt = layer_types[L]
        inv_freq = LAYER_INV_FREQS[lt]
        k = cache.layers[L].keys
        doc_keys = k[:, :, bos_start + 1:, :]
        freqs = torch.einsum('i,j->ij', delta.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_delta = emb.cos().to(k.dtype).unsqueeze(0).unsqueeze(0)
        sin_delta = emb.sin().to(k.dtype).unsqueeze(0).unsqueeze(0)
        doc_keys_new = doc_keys * cos_delta + rotate_half(doc_keys) * sin_delta
        cache.layers[L].keys = torch.cat([
            k[:, :, :bos_start + 1, :],
            doc_keys_new,
        ], dim=2)
    return cache


def score(doc_text, query_text, answer_text, prefix_token_ids=None,
          max_doc_override=None):
    # BOS-retained repositioning.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids

    # Apply max_doc_override for Phase B consistent truncation
    if max_doc_override is not None and len(doc_ids) > max_doc_override:
        doc_ids = doc_ids[:max_doc_override]

    if prefix_token_ids is not None:
        P = len(prefix_token_ids)
        _NL = len(NEWLINE_IDS)
        max_doc = SLIDING_CACHE_LIMIT - 1 - P - _NL  # 1 for BOS
        if len(doc_ids) > max_doc:
            doc_ids = doc_ids[:max_doc]
        D = len(doc_ids)
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa
        keep_indices = [0] + list(range(1 + P + _NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)
        old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

    phase_b_start = D + 1
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
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
    del cache, pb
    return nll


def make_prefix(token_ids, L):
    # Pad/truncate instruction to exactly L tokens.
    if len(token_ids) >= L:
        return token_ids[:L]
    padded = token_ids * ((L // max(len(token_ids), 1)) + 1)
    return padded[:L]


def scramble_prefix(prefix_ids, seed):
    rng = pyrandom.Random(seed)
    shuffled = list(prefix_ids)
    rng.shuffle(shuffled)
    return shuffled


# --- Instruction definitions (4 instructions from Exp 04) ---
INSTRUCTIONS = {
    'extract_general': "Extract all key data points, facts, entities, and specific attributes from the following text.",
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
    'classify': "Determine the subject matter, text type, writing style, and intended audience of this passage.",
    'extract_claims': "Extract all factual claims, statistics, numerical data, and specific assertions made in this passage.",
}

# Pre-tokenize instructions
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")

# Phase A Q-matched conditions (for new datasets)
PHASE_A_PREFIX_CONDS = ['random_tokens', 'comprehend', 'extract_general',
                        'classify', 'extract_claims', 'scrambled_comprehend']

# Phase B fixed-length conditions (same set)
PHASE_B_PREFIX_CONDS = ['random_tokens', 'comprehend', 'extract_general',
                        'classify', 'extract_claims', 'scrambled_comprehend']

# Per-dataset seeds (extending Exp 05 pattern)
DS_SEEDS = {
    'gsm8k': SEED + 700,
    'quality': SEED + 800,
    'multirc': SEED + 900,
    'ropes': SEED + 1000,
    'quoref': SEED + 1100,
    'record': SEED + 1200,
    'race_middle': SEED + 1300,
}

SCORING_KEY = 'bos_retained_token_matched_v06'

special_ids = set(tokenizer.all_special_ids)

print(f"\nSetup complete. Functions: score, make_prefix, scramble_prefix")
print(f"Phase A conditions: bare + {PHASE_A_PREFIX_CONDS}")
print(f"Phase B conditions: bare_trunc + {len(PREFIX_LENGTHS)} lengths x {len(PHASE_B_PREFIX_CONDS)} prefixes")
print(f"Phase B scorings per sample: 1 + {len(PREFIX_LENGTHS)} x {len(PHASE_B_PREFIX_CONDS)} = "
      f"{1 + len(PREFIX_LENGTHS) * len(PHASE_B_PREFIX_CONDS)}")
""")


# ===== Cell 3: Load 7 new datasets =====
code(r"""# Cell 3: Load 7 new reasoning-heavy datasets
from datasets import load_dataset

print("=" * 70)
print("LOADING 7 NEW REASONING-HEAVY DATASETS")
print("=" * 70)

all_samples = {}  # ds_name -> list of N_SAMPLES sample dicts

# ================================================================
# GSM8K — math word problems, single-number answers
# ================================================================
print("\n--- GSM8K (openai/gsm8k, main, test) ---")
ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test")

gsm8k_candidates = []
for item in ds_gsm8k:
    passage = item['question']
    # Answer is the number after ####
    raw_answer = item['answer']
    if '####' not in raw_answer:
        continue
    answer = raw_answer.split('####')[-1].strip()
    if not answer:
        continue
    query = "What is the answer?"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        gsm8k_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })

print(f"  GSM8K candidates: {len(gsm8k_candidates)}")
np.random.seed(DS_SEEDS['gsm8k'])
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:N_SAMPLES]
all_samples['gsm8k'] = [gsm8k_candidates[i] for i in gsm8k_indices]
del ds_gsm8k, gsm8k_candidates
gc.collect()

# ================================================================
# QuALITY — long-form reading comprehension with MC options
# ================================================================
print("\n--- QuALITY (tasksource/QuALITY, validation) ---")
ds_quality = load_dataset("tasksource/QuALITY", split="validation")

quality_candidates = []
for item in ds_quality:
    passage = item['article']
    question = item['question']
    options = item['options']
    gold_label = item['gold_label']
    # gold_label is 1-indexed
    correct_idx = gold_label - 1
    if correct_idx < 0 or correct_idx >= len(options):
        continue
    answer = options[correct_idx]
    wc = count_words(passage)
    # QuALITY articles are very long (~4100w) — we accept all and truncate at tokenization
    if wc >= 30 and count_words(answer) >= 1:
        quality_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
            'all_options': options,
            'correct_idx': correct_idx,
        })

print(f"  QuALITY candidates: {len(quality_candidates)}")
np.random.seed(DS_SEEDS['quality'])
quality_indices = np.random.permutation(len(quality_candidates))[:N_SAMPLES]
all_samples['quality'] = [quality_candidates[i] for i in quality_indices]
del ds_quality, quality_candidates
gc.collect()

# ================================================================
# MultiRC — multi-sentence reasoning with multi-label answers
# ================================================================
print("\n--- MultiRC (aps/super_glue, multirc, validation) ---")
ds_multirc = load_dataset("aps/super_glue", "multirc", split="validation")

# Group by (paragraph_idx, question_idx) and pick first label=1 answer
multirc_groups = defaultdict(list)
for item in ds_multirc:
    idx = item['idx']
    p_idx = idx['paragraph']
    q_idx = idx['question']
    multirc_groups[(p_idx, q_idx)].append(item)

multirc_candidates = []
for (p_idx, q_idx), items in multirc_groups.items():
    # Find first answer with label=1
    positive_answer = None
    for it in items:
        if it['label'] == 1:
            positive_answer = it['answer']
            break
    if positive_answer is None:
        continue

    paragraph = items[0]['paragraph']
    question = items[0]['question']
    wc = count_words(paragraph)
    if 30 <= wc <= 500 and count_words(positive_answer) >= 1:
        multirc_candidates.append({
            'passage': paragraph, 'query': question, 'answer': positive_answer,
            'word_count': wc,
        })

print(f"  MultiRC candidates: {len(multirc_candidates)}")
np.random.seed(DS_SEEDS['multirc'])
multirc_indices = np.random.permutation(len(multirc_candidates))[:N_SAMPLES]
all_samples['multirc'] = [multirc_candidates[i] for i in multirc_indices]
del ds_multirc, multirc_groups, multirc_candidates
gc.collect()

# ================================================================
# ROPES — Reasoning Over Paragraph Effects in Situations
# ================================================================
print("\n--- ROPES (allenai/ropes, validation) ---")
ds_ropes = load_dataset("allenai/ropes", split="validation")

ropes_candidates = []
for item in ds_ropes:
    background = item['background']
    situation = item['situation']
    passage = background + "\n" + situation
    question = item['question']
    answers = item['answers']
    answer_texts = answers.get('text', [])
    if not answer_texts or not answer_texts[0]:
        continue
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        ropes_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })

print(f"  ROPES candidates: {len(ropes_candidates)}")
np.random.seed(DS_SEEDS['ropes'])
ropes_indices = np.random.permutation(len(ropes_candidates))[:N_SAMPLES]
all_samples['ropes'] = [ropes_candidates[i] for i in ropes_indices]
del ds_ropes, ropes_candidates
gc.collect()

# ================================================================
# Quoref — coreference-heavy reading comprehension
# ================================================================
print("\n--- Quoref (nc33/multispan_quoref, validation) ---")
ds_quoref = load_dataset("nc33/multispan_quoref", split="validation")

quoref_candidates = []
for item in ds_quoref:
    passage = item['context']
    question = item['question']
    answers = item['answers']
    answer_texts = answers.get('text', [])
    if not answer_texts or not answer_texts[0]:
        continue
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        quoref_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })

print(f"  Quoref candidates: {len(quoref_candidates)}")
np.random.seed(DS_SEEDS['quoref'])
quoref_indices = np.random.permutation(len(quoref_candidates))[:N_SAMPLES]
all_samples['quoref'] = [quoref_candidates[i] for i in quoref_indices]
del ds_quoref, quoref_candidates
gc.collect()

# ================================================================
# ReCoRD — Reading Comprehension with Commonsense Reasoning Dataset
# ================================================================
print("\n--- ReCoRD (aps/super_glue, record, validation) ---")
ds_record = load_dataset("aps/super_glue", "record", split="validation")

record_candidates = []
for item in ds_record:
    passage = item['passage']
    query = item['query']  # contains @placeholder
    answers = item.get('answers', [])
    if not answers:
        continue
    # Use first answer (entity that fills @placeholder)
    answer = answers[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        record_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })

print(f"  ReCoRD candidates: {len(record_candidates)}")
np.random.seed(DS_SEEDS['record'])
record_indices = np.random.permutation(len(record_candidates))[:N_SAMPLES]
all_samples['record'] = [record_candidates[i] for i in record_indices]
del ds_record, record_candidates
gc.collect()

# ================================================================
# RACE-middle — exam comprehension (easier difficulty level)
# ================================================================
print("\n--- RACE-middle (race, middle, test) ---")
ds_race_mid = load_dataset("race", "middle", split="test")

race_mid_candidates = []
for item in ds_race_mid:
    passage = item['article']
    question = item['question']
    correct_idx = ord(item['answer']) - ord('A')
    options = item['options']
    if correct_idx < 0 or correct_idx >= len(options):
        continue
    answer = options[correct_idx]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        race_mid_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
            'all_options': options,
            'correct_idx': correct_idx,
        })

print(f"  RACE-middle candidates: {len(race_mid_candidates)}")
np.random.seed(DS_SEEDS['race_middle'])
race_mid_indices = np.random.permutation(len(race_mid_candidates))[:N_SAMPLES]
all_samples['race_middle'] = [race_mid_candidates[i] for i in race_mid_indices]
del ds_race_mid, race_mid_candidates
gc.collect()

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("New dataset loading summary:")
for ds_name in NEW_DATASETS:
    samps = all_samples[ds_name]
    print(f"\n  {ds_name}: {len(samps)} samples")
    print(f"    Mean passage words: {np.mean([s['word_count'] for s in samps]):.0f}")
    print(f"    Mean answer words: {np.mean([count_words(s['answer']) for s in samps]):.1f}")
    print(f"    Example query: {samps[0]['query'][:70]}...")
    print(f"    Example answer: {samps[0]['answer'][:70]}...")
    if ds_name in ('quality', 'race_middle'):
        print(f"    Example options: {[o[:30] for o in samps[0]['all_options']]}")
        print(f"    Correct idx: {samps[0]['correct_idx']}")
""")


# ===== Cell 4: Bare scoring + hard selection =====
code(r"""# Cell 4: Bare NLL scoring for 7 new datasets + hard 40% selection
print("=" * 70)
print(f"BARE SCORING — {len(NEW_DATASETS)} new datasets x {N_SAMPLES} samples")
print("=" * 70)

hard_nlls = {}      # ds_name -> {cond_name: np.array}
hard_metadata = {}  # ds_name -> dict
hard_samples = {}   # ds_name -> list of hard sample dicts

for ds_name in NEW_DATASETS:
    print(f"\n--- {ds_name} ({N_SAMPLES} samples) ---")
    samples = all_samples[ds_name]
    bare_ckpt_path = RESULTS_DIR / f"bare_{ds_name}.json"

    bare_nlls = []
    start_idx = 0

    if bare_ckpt_path.exists():
        ckpt = json.loads(bare_ckpt_path.read_text())
        if (ckpt.get('n_total') == N_SAMPLES and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('dataset') == ds_name):
            saved_queries = ckpt.get('queries_first50', [])
            current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
            if saved_queries == current_queries:
                bare_nlls = ckpt['bare_nlls']
                start_idx = len(bare_nlls)
                print(f"  Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

    if start_idx < N_SAMPLES:
        t0 = time.time()
        for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx,
                      total=N_SAMPLES, desc=f"Bare {ds_name}"):
            s = samples[i]
            nll = score(s['passage'], s['query'], s['answer'])
            bare_nlls.append(nll)

            if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_total': N_SAMPLES,
                    'scoring': SCORING_KEY,
                    'bare_nlls': bare_nlls,
                    'queries_first50': [s['query'][:50]
                                        for s in samples[:len(bare_nlls)]],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                bare_ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Bare scoring complete in {elapsed/60:.1f} min")

    bare_arr = np.array(bare_nlls)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])

    hs = []
    for idx in h_idx:
        s = dict(samples[idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        # Carry over MC fields for QuALITY and RACE-middle
        if ds_name in ('quality', 'race_middle'):
            orig = samples[idx]
            s['all_options'] = orig['all_options']
            s['correct_idx'] = orig['correct_idx']
        hs.append(s)
    hard_samples[ds_name] = hs

    hard_nlls[ds_name] = {'bare': bare_arr[h_idx]}

    hard_metadata[ds_name] = {
        'n_total': N_SAMPLES, 'n_hard': N_HARD, 'source': 'scored',
        'mean_passage_words': float(np.mean([s['word_count'] for s in hs])),
        'mean_answer_words': float(np.mean([count_words(s['answer']) for s in hs])),
    }

    print(f"  Hard cutoff: {bare_arr[h_idx].min():.4f}")
    print(f"  Hard mean bare NLL: {bare_arr[h_idx].mean():.4f}")

gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("Hard sample selection complete:")
for ds_name in NEW_DATASETS:
    n_h = len(hard_samples[ds_name])
    print(f"  {ds_name}: {n_h} hard samples (mean bare NLL: "
          f"{hard_nlls[ds_name]['bare'].mean():.4f})")
""")


# ===== Cell 5: Phase A — Q-matched scoring =====
code(r"""# Cell 5: Phase A — Q-matched scoring for 7 new datasets
# Conditions: random_tokens, comprehend, extract_general, classify,
#             extract_claims, scrambled_comprehend
print("=" * 70)
print(f"PHASE A: Q-MATCHED SCORING — {len(NEW_DATASETS)} datasets x "
      f"~{N_HARD} hard x {len(PHASE_A_PREFIX_CONDS)} prefix conditions")
print("=" * 70)

# --- Validation tests ---
print("\n--- Validation: bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"
doc_ids_t = tokenizer(doc_text_t, add_special_tokens=False).input_ids
D_t = len(doc_ids_t)
query_ids_t = tokenizer("\n" + query_text_t + "\n", add_special_tokens=False).input_ids
answer_ids_t = tokenizer(answer_text_t, add_special_tokens=False).input_ids

full_ids = [BOS_ID] + doc_ids_t + query_ids_t + answer_ids_t
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D_t + len(query_ids_t)
logits_full = out_full.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids_t), :].float()
targets_t = torch.tensor(answer_ids_t, device=DEVICE)
nll_single = -F.log_softmax(logits_full, dim=-1).gather(
    1, targets_t.unsqueeze(1)).squeeze(1).mean().item()
del out_full

nll_bare = score(doc_text_t, query_text_t, answer_text_t)
diff_pct = abs(nll_single - nll_bare) / nll_single * 100
print(f"  Single-pass: {nll_single:.6f}, Two-phase: {nll_bare:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"
print("  PASSED")

# --- Score Phase A ---
for ds_name in NEW_DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"phaseA_{ds_name}.json"

    print(f"\n--- Phase A: {ds_name} ({n_hard} hard x {len(PHASE_A_PREFIX_CONDS)} conditions) ---")

    ds_results = []
    start_idx = 0

    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if (ckpt.get('dataset') == ds_name and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('phase') == 'A' and
            ckpt.get('n_hard') == n_hard):
            saved_queries = [r['query'][:50] for r in ckpt.get('results', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                ds_results = ckpt['results']
                start_idx = len(ds_results)
                print(f"  Resuming from checkpoint: {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()

        # Reset RNG for prefix generation
        np.random.seed(DS_SEEDS[ds_name] + 1000)
        pyrandom.seed(DS_SEEDS[ds_name] + 1000)

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"PhaseA {ds_name}"):
            s = hs[i]
            q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
            Q = len(q_ids)

            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'Q': Q,
            }

            # random_tokens: Q random IDs
            rand_ids = []
            while len(rand_ids) < Q:
                tid = np.random.randint(0, VOCAB_SIZE)
                if tid not in special_ids:
                    rand_ids.append(int(tid))
            result['nll_random_tokens'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=rand_ids[:Q])

            # comprehend: Q tokens
            comp_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], Q)
            result['nll_comprehend'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=comp_prefix)

            # extract_general: Q tokens
            ext_prefix = make_prefix(INSTRUCTION_IDS['extract_general'], Q)
            result['nll_extract_general'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=ext_prefix)

            # classify: Q tokens
            cls_prefix = make_prefix(INSTRUCTION_IDS['classify'], Q)
            result['nll_classify'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=cls_prefix)

            # extract_claims: Q tokens
            ecl_prefix = make_prefix(INSTRUCTION_IDS['extract_claims'], Q)
            result['nll_extract_claims'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=ecl_prefix)

            # scrambled_comprehend: Q tokens
            scr_seed = hash('scrambled_comprehend') % (2**31) + i
            scr_prefix = scramble_prefix(comp_prefix, scr_seed)
            result['nll_scrambled_comprehend'] = score(
                s['passage'], s['query'], s['answer'],
                prefix_token_ids=scr_prefix)

            ds_results.append(result)

            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'phase': 'A',
                    'results': ds_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Phase A scoring complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    # Populate hard_nlls
    for cond in PHASE_A_PREFIX_CONDS:
        hard_nlls[ds_name][cond] = np.array(
            [r[f'nll_{cond}'] for r in ds_results])

gc.collect()
torch.cuda.empty_cache()

# Quick Phase A summary
print("\n" + "=" * 70)
print("Phase A summary (Q-matched, new datasets):")
for ds_name in NEW_DATASETS:
    bare = hard_nlls[ds_name]['bare']
    rand = hard_nlls[ds_name]['random_tokens']
    comp = hard_nlls[ds_name]['comprehend']
    ext = hard_nlls[ds_name]['extract_general']
    cls = hard_nlls[ds_name]['classify']
    ecl = hard_nlls[ds_name]['extract_claims']
    print(f"\n  {ds_name}:")
    print(f"    bare:            {bare.mean():.4f}")
    print(f"    random_tokens:   {rand.mean():.4f}  d_bare={cohens_d(bare - rand):+.3f}")
    print(f"    comprehend:      {comp.mean():.4f}  sem_d={cohens_d(rand - comp):+.3f}")
    print(f"    extract_general: {ext.mean():.4f}  sem_d={cohens_d(rand - ext):+.3f}")
    print(f"    classify:        {cls.mean():.4f}  sem_d={cohens_d(rand - cls):+.3f}")
    print(f"    extract_claims:  {ecl.mean():.4f}  sem_d={cohens_d(rand - ecl):+.3f}")
""")


# ===== Cell 6: Phase B + QuALITY MC =====
code(r"""# Cell 6: Phase B — Fixed-length prefix scoring + QuALITY MC accuracy
# L = 8, 16, 32, 64, 128, 256; conditions: bare_trunc + 6 prefix conds at each L
print("=" * 70)
print("PHASE B: FIXED-LENGTH PREFIX SCORING")
print(f"  Lengths: {PREFIX_LENGTHS}")
print(f"  Datasets: {NEW_DATASETS}")
print(f"  COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print(f"  Conditions per sample: 1 + {len(PREFIX_LENGTHS)} x {len(PHASE_B_PREFIX_CONDS)} = "
      f"{1 + len(PREFIX_LENGTHS) * len(PHASE_B_PREFIX_CONDS)}")
print("=" * 70)

# Storage for Phase B NLLs
phase_b_nlls = {ds: {} for ds in NEW_DATASETS}

for ds_name in NEW_DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"phaseB_{ds_name}.json"

    print(f"\n{'='*70}")
    print(f"Phase B: {ds_name} ({n_hard} hard samples)")

    ds_results = []
    start_idx = 0

    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if (ckpt.get('dataset') == ds_name and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('phase') == 'B' and
            ckpt.get('n_hard') == n_hard):
            saved_queries = [r['query'][:50] for r in ckpt.get('results', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                ds_results = ckpt['results']
                start_idx = len(ds_results)
                print(f"  Resuming from checkpoint: {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()

        # Reset RNG for this dataset's Phase B
        np.random.seed(DS_SEEDS[ds_name] + 2000)
        pyrandom.seed(DS_SEEDS[ds_name] + 2000)

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"PhaseB {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
            }

            # bare_trunc: bare with COMMON_MAX_DOC truncation
            result['nll_bare_trunc'] = score(
                s['passage'], s['query'], s['answer'],
                max_doc_override=COMMON_MAX_DOC)

            # For each prefix length
            for L in PREFIX_LENGTHS:
                # random_tokens_L
                rand_ids = []
                while len(rand_ids) < L:
                    tid = np.random.randint(0, VOCAB_SIZE)
                    if tid not in special_ids:
                        rand_ids.append(int(tid))
                result[f'nll_random_tokens_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=rand_ids[:L],
                    max_doc_override=COMMON_MAX_DOC)

                # comprehend_L
                comp_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], L)
                result[f'nll_comprehend_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=comp_prefix,
                    max_doc_override=COMMON_MAX_DOC)

                # extract_general_L
                ext_prefix = make_prefix(INSTRUCTION_IDS['extract_general'], L)
                result[f'nll_extract_general_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=ext_prefix,
                    max_doc_override=COMMON_MAX_DOC)

                # classify_L
                cls_prefix = make_prefix(INSTRUCTION_IDS['classify'], L)
                result[f'nll_classify_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=cls_prefix,
                    max_doc_override=COMMON_MAX_DOC)

                # extract_claims_L
                ecl_prefix = make_prefix(INSTRUCTION_IDS['extract_claims'], L)
                result[f'nll_extract_claims_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=ecl_prefix,
                    max_doc_override=COMMON_MAX_DOC)

                # scrambled_comprehend_L
                scr_seed = hash(f'scrambled_comprehend_{L}') % (2**31) + i
                scr_prefix = scramble_prefix(comp_prefix, scr_seed)
                result[f'nll_scrambled_comprehend_{L}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=scr_prefix,
                    max_doc_override=COMMON_MAX_DOC)

            ds_results.append(result)

            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'phase': 'B',
                    'common_max_doc': COMMON_MAX_DOC,
                    'prefix_lengths': PREFIX_LENGTHS,
                    'results': ds_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Phase B scoring complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    # Populate phase_b_nlls
    phase_b_nlls[ds_name]['bare_trunc'] = np.array(
        [r['nll_bare_trunc'] for r in ds_results])
    for L in PREFIX_LENGTHS:
        for cond in PHASE_B_PREFIX_CONDS:
            key = f'{cond}_{L}'
            phase_b_nlls[ds_name][key] = np.array(
                [r[f'nll_{key}'] for r in ds_results])

gc.collect()
torch.cuda.empty_cache()

# ================================================================
# QuALITY MC accuracy scoring
# ================================================================
print("\n" + "=" * 70)
print("QuALITY MC ACCURACY SCORING")
print("=" * 70)

quality_mc_results = []
hs_quality = hard_samples['quality']
n_hard_quality = len(hs_quality)
quality_mc_ckpt_path = RESULTS_DIR / "quality_mc.json"

start_idx_mc = 0
if quality_mc_ckpt_path.exists():
    mc_ckpt = json.loads(quality_mc_ckpt_path.read_text())
    if (mc_ckpt.get('scoring') == SCORING_KEY and
        mc_ckpt.get('n_hard') == n_hard_quality):
        saved_queries = [r['query'][:50] for r in mc_ckpt.get('results', [])]
        current_queries = [s['query'][:50] for s in hs_quality[:len(saved_queries)]]
        if saved_queries == current_queries:
            quality_mc_results = mc_ckpt['results']
            start_idx_mc = len(quality_mc_results)
            print(f"  Resuming from checkpoint: {start_idx_mc}/{n_hard_quality}")

if start_idx_mc < n_hard_quality:
    t0 = time.time()

    np.random.seed(DS_SEEDS['quality'] + 3000)
    pyrandom.seed(DS_SEEDS['quality'] + 3000)

    for i in tqdm(range(start_idx_mc, n_hard_quality), initial=start_idx_mc,
                  total=n_hard_quality, desc="QuALITY MC"):
        s = hs_quality[i]
        options = s['all_options']
        correct_idx = s['correct_idx']
        q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
        Q = len(q_ids)

        result = {
            'query': s['query'],
            'correct_idx': correct_idx,
            'n_options': len(options),
        }

        # Score all 4 options under each condition
        for opt_i, opt_text in enumerate(options):
            # bare
            result[f'nll_bare_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text)

            # bare_trunc
            result[f'nll_bare_trunc_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                max_doc_override=COMMON_MAX_DOC)

            # Q-matched conditions
            rand_ids = []
            while len(rand_ids) < Q:
                tid = np.random.randint(0, VOCAB_SIZE)
                if tid not in special_ids:
                    rand_ids.append(int(tid))

            result[f'nll_qmatched_random_tokens_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=rand_ids[:Q])

            comp_prefix_q = make_prefix(INSTRUCTION_IDS['comprehend'], Q)
            result[f'nll_qmatched_comprehend_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=comp_prefix_q)

            ext_prefix_q = make_prefix(INSTRUCTION_IDS['extract_general'], Q)
            result[f'nll_qmatched_extract_general_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=ext_prefix_q)

            cls_prefix_q = make_prefix(INSTRUCTION_IDS['classify'], Q)
            result[f'nll_qmatched_classify_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=cls_prefix_q)

            ecl_prefix_q = make_prefix(INSTRUCTION_IDS['extract_claims'], Q)
            result[f'nll_qmatched_extract_claims_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=ecl_prefix_q)

            scr_seed_q = hash('scrambled_comprehend') % (2**31) + i
            scr_prefix_q = scramble_prefix(comp_prefix_q, scr_seed_q)
            result[f'nll_qmatched_scrambled_comprehend_opt{opt_i}'] = score(
                s['passage'], s['query'], opt_text,
                prefix_token_ids=scr_prefix_q)

            # Fixed-length conditions
            for L in PREFIX_LENGTHS:
                rand_ids_L = []
                while len(rand_ids_L) < L:
                    tid = np.random.randint(0, VOCAB_SIZE)
                    if tid not in special_ids:
                        rand_ids_L.append(int(tid))
                result[f'nll_random_tokens_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=rand_ids_L[:L],
                    max_doc_override=COMMON_MAX_DOC)

                comp_prefix_L = make_prefix(INSTRUCTION_IDS['comprehend'], L)
                result[f'nll_comprehend_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=comp_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

                ext_prefix_L = make_prefix(INSTRUCTION_IDS['extract_general'], L)
                result[f'nll_extract_general_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=ext_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

                cls_prefix_L = make_prefix(INSTRUCTION_IDS['classify'], L)
                result[f'nll_classify_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=cls_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

                ecl_prefix_L = make_prefix(INSTRUCTION_IDS['extract_claims'], L)
                result[f'nll_extract_claims_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=ecl_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

                scr_seed_L = hash(f'scrambled_comprehend_{L}') % (2**31) + i
                scr_prefix_L = scramble_prefix(comp_prefix_L, scr_seed_L)
                result[f'nll_scrambled_comprehend_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=scr_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

        quality_mc_results.append(result)

        if (i + 1) % 20 == 0 or i == n_hard_quality - 1:
            mc_ckpt = {
                'scoring': SCORING_KEY,
                'n_hard': n_hard_quality,
                'results': quality_mc_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            quality_mc_ckpt_path.write_text(json.dumps(mc_ckpt))
            elapsed = time.time() - t0
            done = i - start_idx_mc + 1
            eta = (n_hard_quality - i - 1) * elapsed / done if done > 0 else 0
            tqdm.write(f"  MC Checkpoint {i+1}/{n_hard_quality} | "
                       f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  QuALITY MC scoring complete in {elapsed/60:.1f} min")
else:
    print(f"  Loaded {len(quality_mc_results)} cached MC results")

gc.collect()
torch.cuda.empty_cache()
print(f"\nPhase B + QuALITY MC scoring complete for all datasets.")
""")


# ===== Cell 7: Load Exp 05 results for 7 existing datasets =====
code(r"""# Cell 7: Load Exp 05 results for 7 existing datasets
print("=" * 70)
print("LOADING EXP 05 RESULTS FOR 7 EXISTING DATASETS")
print("=" * 70)

assert EXP05_DIR.exists(), f"Exp 05 results not found at {EXP05_DIR}"

# Load the Exp 05 summary results
exp05_results = json.loads((EXP05_DIR / "results.json").read_text())

# ================================================================
# Load Phase B raw NLLs for each old dataset
# ================================================================
exp05_phase_b_nlls = {}
exp05_hard_nlls = {}

for ds_name in OLD_DATASETS:
    # Phase B checkpoints
    pb_path = EXP05_DIR / f"phaseB_{ds_name}.json"
    assert pb_path.exists(), f"Phase B checkpoint not found: {pb_path}"
    pb_ckpt = json.loads(pb_path.read_text())
    pb_results = pb_ckpt['results']
    n_hard_old = len(pb_results)

    exp05_phase_b_nlls[ds_name] = {}
    exp05_phase_b_nlls[ds_name]['bare_trunc'] = np.array(
        [r['nll_bare_trunc'] for r in pb_results])

    # Exp 05 used PREFIX_LENGTHS = [32, 64, 128, 256] (no L=8 or L=16)
    # and conditions: random_tokens, comprehend, extract_general, scrambled_comprehend
    # (no classify or extract_claims)
    exp05_lengths = [32, 64, 128, 256]
    exp05_conds = ['random_tokens', 'comprehend', 'extract_general', 'scrambled_comprehend']
    for L in exp05_lengths:
        for cond in exp05_conds:
            key = f'{cond}_{L}'
            exp05_phase_b_nlls[ds_name][key] = np.array(
                [r[f'nll_{key}'] for r in pb_results])

    print(f"  {ds_name}: loaded Phase B ({n_hard_old} hard, L={exp05_lengths})")

# Load Q-matched NLLs for old datasets from their Phase A / bare checkpoints
for ds_name in OLD_DATASETS:
    exp05_hard_nlls[ds_name] = {}

    # Bare NLLs
    bare_path = EXP05_DIR / f"bare_{ds_name}.json"
    if bare_path.exists():
        bare_ckpt = json.loads(bare_path.read_text())
        bare_arr = np.array(bare_ckpt['bare_nlls'])
        exp05_n_samples = len(bare_arr)
        exp05_n_hard = int(exp05_n_samples * HARD_FRAC)
        sorted_idx = np.argsort(bare_arr)[::-1]
        h_idx = np.sort(sorted_idx[:exp05_n_hard])
        exp05_hard_nlls[ds_name]['bare'] = bare_arr[h_idx]
    else:
        # Fallback: use Phase B bare_trunc
        exp05_hard_nlls[ds_name]['bare'] = exp05_phase_b_nlls[ds_name]['bare_trunc']

    # Phase A for datasets scored in Exp 05 (drop, boolq, race_high)
    pa_path = EXP05_DIR / f"phaseA_{ds_name}.json"
    if pa_path.exists():
        pa_ckpt = json.loads(pa_path.read_text())
        pa_results = pa_ckpt['results']
        # Exp 05 Phase A conditions: random_tokens, comprehend, extract_general, scrambled_comprehend
        for cond in ['random_tokens', 'comprehend', 'extract_general', 'scrambled_comprehend']:
            key = f'nll_{cond}'
            if pa_results and key in pa_results[0]:
                exp05_hard_nlls[ds_name][cond] = np.array(
                    [r[key] for r in pa_results])

# Load RACE MC results from Exp 05
race_mc_path = EXP05_DIR / "race_mc.json"
exp05_race_mc_results = []
if race_mc_path.exists():
    race_mc_ckpt = json.loads(race_mc_path.read_text())
    exp05_race_mc_results = race_mc_ckpt.get('results', [])
    print(f"  RACE-high MC: loaded {len(exp05_race_mc_results)} results")

# Load per-dataset analysis from results.json
exp05_per_dataset = exp05_results.get('per_dataset', {})

print(f"\nLoaded Exp 05 results for {len(OLD_DATASETS)} datasets")
print(f"  Phase B lengths available: [32, 64, 128, 256]")
print(f"  Phase B conditions available: random_tokens, comprehend, extract_general, scrambled_comprehend")
print(f"  Note: L=8, L=16, classify, extract_claims only available for {len(NEW_DATASETS)} new datasets")
""")


# ===== Cell 8: Per-dataset analysis tables (all 14 datasets) =====
code(r"""# Cell 8: Per-dataset results — Phase A + Phase B tables
print("=" * 70)
print(f"PER-DATASET RESULTS ({len(ALL_DATASETS)} DATASETS)")
print("=" * 70)

per_dataset_analysis = {}

# ================================================================
# PART 1: Phase A — Q-matched results (7 new datasets)
# ================================================================
print("\n" + "=" * 70)
print(f"PHASE A: Q-MATCHED RESULTS ({len(NEW_DATASETS)} new datasets)")
print("=" * 70)

PHASE_A_CONDS = ['bare', 'random_tokens', 'comprehend', 'extract_general',
                 'classify', 'extract_claims', 'scrambled_comprehend']

for ds_name in NEW_DATASETS:
    nlls = hard_nlls[ds_name]
    n_hard = len(nlls['bare'])
    bare = nlls['bare']
    rand = nlls.get('random_tokens')

    print(f"\n--- {ds_name.upper()} ({n_hard} hard samples, Q-matched) ---")
    print(f"  {'Condition':<28} {'NLL':>7} {'d_bare':>8} {'sem_d':>8} "
          f"{'win%':>6} {'p':>10} {'sig':>4}")
    print(f"  {'-'*76}")

    analysis = {}
    for cond in PHASE_A_CONDS:
        if cond not in nlls:
            continue
        c = nlls[cond]
        mean_nll = c.mean()

        if cond == 'bare':
            print(f"  {cond:<28} {mean_nll:>7.3f} {'--':>8} {'--':>8} "
                  f"{'--':>6} {'--':>10} {'--':>4}")
            analysis[cond] = {'mean_nll': float(mean_nll)}
            continue

        diff_bare = bare - c
        d_bare = cohens_d(diff_bare)
        _, p_bare = stats.ttest_1samp(diff_bare, 0)

        if cond == 'random_tokens':
            win_pct = 100 * np.mean(diff_bare > 0)
            sig = ('***' if p_bare < 0.001 else '**' if p_bare < 0.01
                   else '*' if p_bare < 0.05 else 'ns')
            print(f"  {cond:<28} {mean_nll:>7.3f} {d_bare:>+8.3f} {'(ref)':>8} "
                  f"{win_pct:>5.1f}% {p_bare:>10.2e} {sig:>4}")
            analysis[cond] = {
                'mean_nll': float(mean_nll), 'd_bare': float(d_bare),
                'semantic_delta_d': 0.0, 'p_bare': float(p_bare),
            }
        else:
            sem_delta = rand - c
            d_sem = cohens_d(sem_delta)
            _, p_sem = stats.ttest_1samp(sem_delta, 0)
            win_pct = 100 * np.mean(sem_delta > 0)
            sig = ('***' if p_sem < 0.001 else '**' if p_sem < 0.01
                   else '*' if p_sem < 0.05 else 'ns')
            print(f"  {cond:<28} {mean_nll:>7.3f} {d_bare:>+8.3f} {d_sem:>+8.3f} "
                  f"{win_pct:>5.1f}% {p_sem:>10.2e} {sig:>4}")
            analysis[cond] = {
                'mean_nll': float(mean_nll), 'd_bare': float(d_bare),
                'semantic_delta_d': float(d_sem), 'p_semantic': float(p_sem),
            }

    # Three-level decomposition (Q-matched)
    if 'random_tokens' in nlls and 'comprehend' in nlls and 'scrambled_comprehend' in nlls:
        structural = (bare - rand).mean()
        vocab = (rand - nlls['scrambled_comprehend']).mean()
        meaning = (nlls['scrambled_comprehend'] - nlls['comprehend']).mean()
        total = structural + vocab + meaning
        print(f"\n  Q-matched decomposition (comprehend):")
        print(f"    Structural: {structural:+.4f}, Vocabulary: {vocab:+.4f}, "
              f"Meaning: {meaning:+.4f}, Total: {total:+.4f}")
        analysis['decomposition_qmatched'] = {
            'structural': float(structural), 'vocabulary': float(vocab),
            'meaning': float(meaning), 'total': float(total),
        }

    per_dataset_analysis[ds_name] = analysis

# ================================================================
# PART 2: Phase B — Fixed-length results (7 new datasets)
# ================================================================
print("\n" + "=" * 70)
print(f"PHASE B: FIXED-LENGTH RESULTS ({len(NEW_DATASETS)} new datasets)")
print("=" * 70)

for ds_name in NEW_DATASETS:
    pb = phase_b_nlls[ds_name]
    bare_t = pb['bare_trunc']
    n_hard = len(bare_t)

    print(f"\n--- {ds_name.upper()} ({n_hard} hard samples, fixed-length) ---")
    print(f"  bare_trunc NLL: {bare_t.mean():.4f}")

    print(f"\n  {'L':>5} {'rand':>9} {'comp':>9} {'ext':>9} {'cls':>9} "
          f"{'ecl':>9} {'scr':>9} {'d_comp':>8} {'d_ext':>8}")
    print(f"  {'-'*84}")

    phase_b_analysis = {}
    for L in PREFIX_LENGTHS:
        rand_L = pb[f'random_tokens_{L}']
        comp_L = pb[f'comprehend_{L}']
        ext_L = pb[f'extract_general_{L}']
        cls_L = pb[f'classify_{L}']
        ecl_L = pb[f'extract_claims_{L}']
        scr_L = pb[f'scrambled_comprehend_{L}']

        d_comp = cohens_d(rand_L - comp_L)
        d_ext = cohens_d(rand_L - ext_L)
        d_cls = cohens_d(rand_L - cls_L)
        d_ecl = cohens_d(rand_L - ecl_L)
        d_scr = cohens_d(rand_L - scr_L)

        print(f"  {L:>5} {rand_L.mean():>9.4f} {comp_L.mean():>9.4f} "
              f"{ext_L.mean():>9.4f} {cls_L.mean():>9.4f} "
              f"{ecl_L.mean():>9.4f} {scr_L.mean():>9.4f} "
              f"{d_comp:>+8.3f} {d_ext:>+8.3f}")

        # Three-level decomposition at this length
        structural = (bare_t - rand_L).mean()
        vocab = (rand_L - scr_L).mean()
        meaning = (scr_L - comp_L).mean()
        total = structural + vocab + meaning

        phase_b_analysis[L] = {
            'structural': float(structural), 'vocabulary': float(vocab),
            'meaning': float(meaning), 'total': float(total),
            'd_comprehend': float(d_comp), 'd_extract_general': float(d_ext),
            'd_classify': float(d_cls), 'd_extract_claims': float(d_ecl),
            'd_scrambled_comprehend': float(d_scr),
            'mean_nll': {
                'bare_trunc': float(bare_t.mean()),
                'random_tokens': float(rand_L.mean()),
                'comprehend': float(comp_L.mean()),
                'extract_general': float(ext_L.mean()),
                'classify': float(cls_L.mean()),
                'extract_claims': float(ecl_L.mean()),
                'scrambled_comprehend': float(scr_L.mean()),
            }
        }

    per_dataset_analysis[ds_name]['phase_b'] = phase_b_analysis

    # Decomposition table
    print(f"\n  Three-Level Decomposition x Length:")
    print(f"  {'L':>5} {'Structural':>12} {'Vocabulary':>12} "
          f"{'Meaning':>12} {'Total':>12}")
    print(f"  {'-'*56}")
    for L in PREFIX_LENGTHS:
        d = phase_b_analysis[L]
        print(f"  {L:>5} {d['structural']:>+12.4f} {d['vocabulary']:>+12.4f} "
              f"{d['meaning']:>+12.4f} {d['total']:>+12.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 9: Cross-dataset meta-analysis (all 14 datasets) =====
code(r"""# Cell 9: Cross-dataset meta-analysis (14 datasets)
print("=" * 70)
print(f"CROSS-DATASET META-ANALYSIS ({len(ALL_DATASETS)} DATASETS)")
print("=" * 70)

# Merge Exp 05 Phase B NLLs with new Phase B NLLs
all_phase_b_nlls = {}
for ds_name in NEW_DATASETS:
    all_phase_b_nlls[ds_name] = phase_b_nlls[ds_name]
for ds_name in OLD_DATASETS:
    all_phase_b_nlls[ds_name] = exp05_phase_b_nlls[ds_name]

# Merge Q-matched NLLs
all_hard_nlls = {}
for ds_name in NEW_DATASETS:
    all_hard_nlls[ds_name] = hard_nlls[ds_name]
for ds_name in OLD_DATASETS:
    all_hard_nlls[ds_name] = exp05_hard_nlls[ds_name]

# ================================================================
# PART 1: Phase A — Q-matched meta-analysis
# ================================================================
print("\n--- PART 1: Phase A Q-Matched Meta-Analysis ---")

# Use comprehend and extract_general (available in both Exp 05 and 06)
PHASE_A_META_CONDS = ['comprehend', 'extract_general', 'scrambled_comprehend']

print(f"\n  {'Condition':<28} {'pooled_d':>9} {'SE':>8} {'z':>8} "
      f"{'p':>10} {'95% CI':>16} {'sig':>4} {'n_ds':>5}")
print(f"  {'-'*92}")

meta_results_A = {}
for cond in PHASE_A_META_CONDS:
    ds_effects = []
    for ds_name in ALL_DATASETS:
        nlls = all_hard_nlls.get(ds_name, {})
        if cond not in nlls or 'random_tokens' not in nlls:
            continue
        sem_delta = nlls['random_tokens'] - nlls[cond]
        n = len(sem_delta)
        d = cohens_d(sem_delta)
        se = np.sqrt(1.0/n + d**2 / (2.0*n))
        ds_effects.append((d, se, n, ds_name))

    if not ds_effects:
        continue

    weights = [1.0 / (se**2) for _, se, _, _ in ds_effects]
    w_sum = sum(weights)
    pooled_d = sum(w * d for (d, _, _, _), w in zip(ds_effects, weights)) / w_sum
    pooled_se = 1.0 / np.sqrt(w_sum)
    z = pooled_d / pooled_se if pooled_se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    ci_lo = pooled_d - 1.96 * pooled_se
    ci_hi = pooled_d + 1.96 * pooled_se
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')

    print(f"  {cond:<28} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
          f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4} {len(ds_effects):>5}")
    meta_results_A[cond] = {
        'pooled_d': float(pooled_d), 'se': float(pooled_se),
        'z': float(z), 'p': float(p),
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
        'n_datasets': len(ds_effects),
        'per_dataset': {name: float(d) for d, _, _, name in ds_effects},
    }

# Expanded meta-analysis for classify and extract_claims (7 new datasets only)
print(f"\n  Expanded conditions (7 new datasets only):")
print(f"  {'Condition':<28} {'pooled_d':>9} {'SE':>8} {'z':>8} "
      f"{'p':>10} {'sig':>4} {'n_ds':>5}")
print(f"  {'-'*72}")

for cond in ['classify', 'extract_claims']:
    ds_effects = []
    for ds_name in NEW_DATASETS:
        nlls = hard_nlls.get(ds_name, {})
        if cond not in nlls or 'random_tokens' not in nlls:
            continue
        sem_delta = nlls['random_tokens'] - nlls[cond]
        n = len(sem_delta)
        d = cohens_d(sem_delta)
        se = np.sqrt(1.0/n + d**2 / (2.0*n))
        ds_effects.append((d, se, n, ds_name))

    if not ds_effects:
        continue

    weights = [1.0 / (se**2) for _, se, _, _ in ds_effects]
    w_sum = sum(weights)
    pooled_d = sum(w * d for (d, _, _, _), w in zip(ds_effects, weights)) / w_sum
    pooled_se = 1.0 / np.sqrt(w_sum)
    z = pooled_d / pooled_se if pooled_se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')

    print(f"  {cond:<28} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
          f"{p:>10.2e} {sig:>4} {len(ds_effects):>5}")
    meta_results_A[cond] = {
        'pooled_d': float(pooled_d), 'se': float(pooled_se),
        'z': float(z), 'p': float(p),
        'n_datasets': len(ds_effects),
        'per_dataset': {name: float(d) for d, _, _, name in ds_effects},
    }

# Per-dataset d table for comprehend
print(f"\n  Per-dataset sem_d (comprehend):")
print(f"  {'Dataset':<16} {'d':>8} {'n':>5}")
print(f"  {'-'*32}")
for ds_name in ALL_DATASETS:
    nlls = all_hard_nlls.get(ds_name, {})
    if 'comprehend' in nlls and 'random_tokens' in nlls:
        d = cohens_d(nlls['random_tokens'] - nlls['comprehend'])
        n = len(nlls['comprehend'])
        print(f"  {ds_name:<16} {d:>+8.3f} {n:>5}")
    else:
        print(f"  {ds_name:<16} {'N/A':>8}")

# ================================================================
# PART 2: Phase B — Length Scaling Curves
# ================================================================
print(f"\n--- PART 2: Length Scaling Curves ---")

# Common conditions between Exp 05 and 06
COMMON_CONDS = ['comprehend', 'extract_general', 'scrambled_comprehend']

scaling_results = {}
for cond_base in COMMON_CONDS:
    print(f"\n  {cond_base}:")
    print(f"  {'L':>5} {'pooled_d':>9} {'SE':>8} {'z':>8} "
          f"{'p':>10} {'sig':>4} {'n_ds':>5}")
    print(f"  {'-'*54}")

    scaling_results[cond_base] = {}
    for L in PREFIX_LENGTHS:
        ds_effects = []
        for ds_name in ALL_DATASETS:
            pb = all_phase_b_nlls.get(ds_name, {})
            rand_key = f'random_tokens_{L}'
            cond_key = f'{cond_base}_{L}'
            if rand_key not in pb or cond_key not in pb:
                continue
            sem_delta = pb[rand_key] - pb[cond_key]
            n = len(sem_delta)
            d = cohens_d(sem_delta)
            se = np.sqrt(1.0/n + d**2 / (2.0*n))
            ds_effects.append((d, se, n))

        if not ds_effects:
            continue

        weights = [1.0 / (se**2) for _, se, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        z = pooled_d / pooled_se if pooled_se > 0 else 0.0
        p = 2 * stats.norm.sf(abs(z))
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')

        print(f"  {L:>5} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
              f"{p:>10.2e} {sig:>4} {len(ds_effects):>5}")
        scaling_results[cond_base][L] = {
            'pooled_d': float(pooled_d), 'se': float(pooled_se),
            'z': float(z), 'p': float(p),
            'n_datasets': len(ds_effects),
        }

# Expanded scaling for classify and extract_claims (7 new datasets)
print(f"\n  Expanded conditions (7 new datasets only):")
for cond_base in ['classify', 'extract_claims']:
    print(f"\n  {cond_base}:")
    print(f"  {'L':>5} {'pooled_d':>9} {'SE':>8} {'z':>8} "
          f"{'p':>10} {'sig':>4} {'n_ds':>5}")
    print(f"  {'-'*54}")

    scaling_results[cond_base] = {}
    for L in PREFIX_LENGTHS:
        ds_effects = []
        for ds_name in NEW_DATASETS:
            pb = phase_b_nlls.get(ds_name, {})
            rand_key = f'random_tokens_{L}'
            cond_key = f'{cond_base}_{L}'
            if rand_key not in pb or cond_key not in pb:
                continue
            sem_delta = pb[rand_key] - pb[cond_key]
            n = len(sem_delta)
            d = cohens_d(sem_delta)
            se = np.sqrt(1.0/n + d**2 / (2.0*n))
            ds_effects.append((d, se, n))

        if not ds_effects:
            continue

        weights = [1.0 / (se**2) for _, se, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        z = pooled_d / pooled_se if pooled_se > 0 else 0.0
        p = 2 * stats.norm.sf(abs(z))
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')

        print(f"  {L:>5} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
              f"{p:>10.2e} {sig:>4} {len(ds_effects):>5}")
        scaling_results[cond_base][L] = {
            'pooled_d': float(pooled_d), 'se': float(pooled_se),
            'z': float(z), 'p': float(p),
            'n_datasets': len(ds_effects),
        }

# ================================================================
# PART 3: Three-Level Decomposition x Length (pooled)
# ================================================================
print(f"\n--- PART 3: Three-Level Decomposition x Length (pooled) ---")
print(f"  {'L':>5} {'Structural':>12} {'Vocabulary':>12} "
      f"{'Meaning':>12} {'Total':>12} {'n_ds':>5}")
print(f"  {'-'*62}")

pooled_decomp_by_length = {}
for L in PREFIX_LENGTHS:
    structs, vocabs, meanings, totals = [], [], [], []
    n_ds = 0
    for ds_name in ALL_DATASETS:
        pb = all_phase_b_nlls.get(ds_name, {})
        bare_key = 'bare_trunc'
        rand_key = f'random_tokens_{L}'
        scr_key = f'scrambled_comprehend_{L}'
        comp_key = f'comprehend_{L}'
        if bare_key not in pb or rand_key not in pb or scr_key not in pb or comp_key not in pb:
            continue

        structs.append((pb[bare_key] - pb[rand_key]).mean())
        vocabs.append((pb[rand_key] - pb[scr_key]).mean())
        meanings.append((pb[scr_key] - pb[comp_key]).mean())
        totals.append((pb[bare_key] - pb[comp_key]).mean())
        n_ds += 1

    if not structs:
        continue

    s_m, v_m, m_m, t_m = np.mean(structs), np.mean(vocabs), np.mean(meanings), np.mean(totals)
    print(f"  {L:>5} {s_m:>+12.4f} {v_m:>+12.4f} {m_m:>+12.4f} {t_m:>+12.4f} {n_ds:>5}")
    pooled_decomp_by_length[L] = {
        'structural': float(s_m), 'vocabulary': float(v_m),
        'meaning': float(m_m), 'total': float(t_m),
        'n_datasets': n_ds,
    }

# Percentage breakdown
print(f"\n  Percentage breakdown:")
print(f"  {'L':>5} {'Struct%':>9} {'Vocab%':>9} {'Meaning%':>9} {'n_ds':>5}")
print(f"  {'-'*42}")
for L in PREFIX_LENGTHS:
    if L not in pooled_decomp_by_length:
        continue
    d = pooled_decomp_by_length[L]
    t = d['total']
    if abs(t) > 0.001:
        print(f"  {L:>5} {d['structural']/t*100:>8.0f}% "
              f"{d['vocabulary']/t*100:>8.0f}% {d['meaning']/t*100:>8.0f}% "
              f"{d['n_datasets']:>5}")

# ================================================================
# PART 4: Task x Length Interaction
# ================================================================
print(f"\n--- PART 4: Task x Length Interaction (comprehend sem_d) ---")
print(f"  {'Dataset':<16}", end="")
for L in PREFIX_LENGTHS:
    print(f" {'L='+str(L):>8}", end="")
print(f"  {'trend':>8}")
print(f"  {'-'*82}")

task_length = {}
for ds_name in ALL_DATASETS:
    pb = all_phase_b_nlls.get(ds_name, {})
    row = f"  {ds_name:<16}"
    ds_vals = []
    ds_lengths = []
    for L in PREFIX_LENGTHS:
        rand_key = f'random_tokens_{L}'
        comp_key = f'comprehend_{L}'
        if rand_key in pb and comp_key in pb:
            d = cohens_d(pb[rand_key] - pb[comp_key])
            row += f" {d:>+8.3f}"
            ds_vals.append(d)
            ds_lengths.append(L)
        else:
            row += f" {'N/A':>8}"
    # Simple trend
    if len(ds_vals) >= 2:
        rho, _ = stats.spearmanr(ds_lengths, ds_vals)
        trend = "UP" if rho > 0.5 else "DOWN" if rho < -0.5 else "FLAT"
    else:
        trend = "N/A"
    row += f"  {trend:>8}"
    print(row)
    task_length[ds_name] = {L: float(d) for L, d in zip(ds_lengths, ds_vals)}

# ================================================================
# PART 5: QuALITY MC Accuracy
# ================================================================
print(f"\n--- PART 5: QuALITY MC Accuracy ---")

def compute_accuracy(results, cond_prefix, n_options=4):
    correct = 0
    total = 0
    for r in results:
        correct_idx = r['correct_idx']
        nlls = []
        for opt_i in range(n_options):
            key = f'nll_{cond_prefix}_opt{opt_i}'
            if key not in r:
                break
            nlls.append(r[key])
        if len(nlls) == n_options:
            pred = np.argmin(nlls)
            if pred == correct_idx:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

print(f"\n  {'Condition':<36} {'Accuracy':>8} {'N':>5}")
print(f"  {'-'*52}")

quality_accuracy = {}
# Bare
acc_bare = compute_accuracy(quality_mc_results, 'bare')
print(f"  {'bare':<36} {acc_bare:>7.1%} {len(quality_mc_results):>5}")
quality_accuracy['bare'] = float(acc_bare)

# Bare truncated
acc_bare_t = compute_accuracy(quality_mc_results, 'bare_trunc')
print(f"  {'bare_trunc':<36} {acc_bare_t:>7.1%} {len(quality_mc_results):>5}")
quality_accuracy['bare_trunc'] = float(acc_bare_t)

# Q-matched
for cond in PHASE_A_PREFIX_CONDS:
    acc = compute_accuracy(quality_mc_results, f'qmatched_{cond}')
    print(f"  {'qmatched_' + cond:<36} {acc:>7.1%} {len(quality_mc_results):>5}")
    quality_accuracy[f'qmatched_{cond}'] = float(acc)

# Fixed-length
for L in PREFIX_LENGTHS:
    for cond in PHASE_B_PREFIX_CONDS:
        acc = compute_accuracy(quality_mc_results, f'{cond}_{L}')
        print(f"  {f'{cond}_{L}':<36} {acc:>7.1%} {len(quality_mc_results):>5}")
        quality_accuracy[f'{cond}_{L}'] = float(acc)

# ================================================================
# PART 6: Cross-Benchmark Ranking (best condition x length)
# ================================================================
print(f"\n--- PART 6: Cross-Benchmark Ranking (best prefix x length) ---")

SHARED_LENGTHS = [32, 64, 128, 256]  # Available in both Exp 05 and 06

all_combos = []
for cond_base in ['comprehend', 'extract_general']:
    for L in SHARED_LENGTHS:
        ds_effects = []
        for ds_name in ALL_DATASETS:
            pb = all_phase_b_nlls.get(ds_name, {})
            rand_key = f'random_tokens_{L}'
            cond_key = f'{cond_base}_{L}'
            if rand_key not in pb or cond_key not in pb:
                continue
            sem_delta = pb[rand_key] - pb[cond_key]
            n = len(sem_delta)
            d = cohens_d(sem_delta)
            se = np.sqrt(1.0/n + d**2 / (2.0*n))
            ds_effects.append((d, se, n))
        if not ds_effects:
            continue
        weights = [1.0 / (se**2) for _, se, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        all_combos.append((cond_base, L, pooled_d, pooled_se, len(ds_effects)))

all_combos.sort(key=lambda x: x[2], reverse=True)
print(f"\n  {'Rank':>4} {'Condition':<28} {'L':>5} {'pooled_d':>9} {'SE':>8} {'n_ds':>5}")
print(f"  {'-'*64}")
for rank, (cond, L, d, se, n_ds) in enumerate(all_combos, 1):
    print(f"  {rank:>4} {cond:<28} {L:>5} {d:>+9.4f} {se:>8.4f} {n_ds:>5}")

# ================================================================
# PART 7: Comprehend Meaning Growth
# ================================================================
print(f"\n--- PART 7: Comprehend Meaning Effect at Each Length ---")
print(f"  {'L':>5} {'pooled_d':>9} {'SE':>8} {'z':>8} {'p':>10} {'sig':>4} {'n_ds':>5}")
print(f"  {'-'*54}")

meaning_by_length = {}
for L in PREFIX_LENGTHS:
    ds_diffs = []
    for ds_name in ALL_DATASETS:
        pb = all_phase_b_nlls.get(ds_name, {})
        scr_key = f'scrambled_comprehend_{L}'
        comp_key = f'comprehend_{L}'
        if scr_key not in pb or comp_key not in pb:
            continue
        diff = pb[scr_key] - pb[comp_key]
        ds_diffs.append(diff)
    if not ds_diffs:
        continue
    pooled_diff = np.concatenate(ds_diffs)
    d = cohens_d(pooled_diff)
    _, p = stats.ttest_1samp(pooled_diff, 0)
    se = np.sqrt(1.0/len(pooled_diff) + d**2/(2.0*len(pooled_diff)))
    z = d / se if se > 0 else 0.0
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')
    print(f"  {L:>5} {d:>+9.4f} {se:>8.4f} {z:>+8.2f} {p:>10.2e} {sig:>4} {len(ds_diffs):>5}")
    meaning_by_length[L] = {
        'd': float(d), 'p': float(p), 'se': float(se),
        'n_datasets': len(ds_diffs),
    }

# ================================================================
# PART 8: Ceiling Analysis
# ================================================================
print(f"\n--- PART 8: Ceiling Analysis ---")
print(f"  Maximum achievable total effect (bare_trunc - comprehend_L):")
print(f"  {'L':>5} {'pooled mean delta':>18} {'max ds delta':>14} "
      f"{'as % of bare':>14} {'n_ds':>5}")
print(f"  {'-'*62}")

ceiling = {}
for L in PREFIX_LENGTHS:
    ds_deltas = []
    ds_pcts = []
    for ds_name in ALL_DATASETS:
        pb = all_phase_b_nlls.get(ds_name, {})
        bare_key = 'bare_trunc'
        comp_key = f'comprehend_{L}'
        if bare_key not in pb or comp_key not in pb:
            continue
        delta = (pb[bare_key] - pb[comp_key]).mean()
        bare_mean = pb[bare_key].mean()
        pct = delta / bare_mean * 100 if bare_mean > 0 else 0
        ds_deltas.append(delta)
        ds_pcts.append(pct)
    if not ds_deltas:
        continue
    mean_delta = np.mean(ds_deltas)
    max_delta = np.max(ds_deltas)
    mean_pct = np.mean(ds_pcts)
    print(f"  {L:>5} {mean_delta:>+18.4f} {max_delta:>+14.4f} "
          f"{mean_pct:>13.1f}% {len(ds_deltas):>5}")
    ceiling[L] = {
        'mean_delta': float(mean_delta), 'max_delta': float(max_delta),
        'mean_pct': float(mean_pct), 'n_datasets': len(ds_deltas),
    }

# ================================================================
# PART 9: Dataset Ranking at L=64
# ================================================================
print(f"\n--- PART 9: Dataset Ranking at L=64 (comprehend sem_d) ---")
print(f"  {'Rank':>4} {'Dataset':<16} {'d(comp)':>9} {'d(ext)':>9} "
      f"{'mean_ans_w':>11} {'mean_pass_w':>12}")
print(f"  {'-'*66}")

ranking_at_64 = []
for ds_name in ALL_DATASETS:
    pb = all_phase_b_nlls.get(ds_name, {})
    if 'random_tokens_64' not in pb or 'comprehend_64' not in pb:
        continue
    d_comp = cohens_d(pb['random_tokens_64'] - pb['comprehend_64'])
    d_ext_key = 'extract_general_64'
    d_ext = cohens_d(pb['random_tokens_64'] - pb[d_ext_key]) if d_ext_key in pb else 0.0

    # Get answer word stats
    if ds_name in hard_metadata:
        mean_ans_w = hard_metadata[ds_name].get('mean_answer_words', 0)
        mean_pass_w = hard_metadata[ds_name].get('mean_passage_words', 0)
    elif ds_name in exp05_results.get('hard_metadata', {}):
        hm = exp05_results['hard_metadata'][ds_name]
        mean_ans_w = hm.get('mean_answer_words', 0)
        mean_pass_w = hm.get('mean_passage_words', 0)
    else:
        mean_ans_w = 0
        mean_pass_w = 0

    ranking_at_64.append((ds_name, d_comp, d_ext, mean_ans_w, mean_pass_w))

ranking_at_64.sort(key=lambda x: x[1], reverse=True)
for rank, (name, d_c, d_e, maw, mpw) in enumerate(ranking_at_64, 1):
    print(f"  {rank:>4} {name:<16} {d_c:>+9.3f} {d_e:>+9.3f} "
          f"{maw:>11.1f} {mpw:>12.0f}")

# ================================================================
# PART 10: Answer-Length Analysis
# ================================================================
print(f"\n--- PART 10: Answer-Length Analysis ---")
print(f"  Does mean answer word count predict effect direction?")

ans_length_data = []
for name, d_c, d_e, maw, mpw in ranking_at_64:
    if maw > 0:
        ans_length_data.append((maw, d_c))

if len(ans_length_data) >= 4:
    maw_arr = np.array([x[0] for x in ans_length_data])
    d_arr = np.array([x[1] for x in ans_length_data])
    rho, p_rho = stats.spearmanr(maw_arr, d_arr)
    print(f"  Spearman rho(mean_ans_words, d_comp@L=64) = {rho:+.3f}, p = {p_rho:.4f}")
    print(f"  N datasets with answer length data: {len(ans_length_data)}")
    sig = ('***' if p_rho < 0.001 else '**' if p_rho < 0.01
           else '*' if p_rho < 0.05 else 'ns')
    print(f"  Significance: {sig}")
else:
    rho, p_rho = 0.0, 1.0
    print(f"  Insufficient data for correlation (n={len(ans_length_data)})")

# ================================================================
# PART 11: Instruction Comparison (7 new datasets, all 4 instructions)
# ================================================================
print(f"\n--- PART 11: Instruction Comparison at L=64 (7 new datasets) ---")
print(f"  {'Instruction':<20}", end="")
for ds_name in NEW_DATASETS:
    print(f" {ds_name[:8]:>8}", end="")
print(f"  {'mean':>8}")
print(f"  {'-'*90}")

instruction_comparison = {}
for instr in ['comprehend', 'extract_general', 'classify', 'extract_claims']:
    row = f"  {instr:<20}"
    ds_vals = []
    for ds_name in NEW_DATASETS:
        pb = phase_b_nlls.get(ds_name, {})
        rand_key = 'random_tokens_64'
        cond_key = f'{instr}_64'
        if rand_key in pb and cond_key in pb:
            d = cohens_d(pb[rand_key] - pb[cond_key])
            row += f" {d:>+8.3f}"
            ds_vals.append(d)
        else:
            row += f" {'N/A':>8}"
    if ds_vals:
        row += f"  {np.mean(ds_vals):>+8.3f}"
    print(row)
    instruction_comparison[instr] = {
        ds: float(d) for ds, d in zip(
            [n for n in NEW_DATASETS if f'{instr}_64' in phase_b_nlls.get(n, {})],
            ds_vals)
    }

# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 70)
print("VERDICT — Exp 06: Hero Run — Seven New Reasoning-Heavy Datasets")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"New datasets: {len(NEW_DATASETS)} ({', '.join(NEW_DATASETS)})")
print(f"Total datasets in meta-analysis: {len(ALL_DATASETS)}")
print(f"N_SAMPLES: {N_SAMPLES}, N_HARD: {N_HARD}")
print(f"Prefix lengths: {PREFIX_LENGTHS}")
print(f"Instructions: {list(INSTRUCTIONS.keys())}")

print(f"\n--- Key Findings ---")

# 1. Phase A meta-analysis
print(f"\n  1. Phase A meta-analysis ({len(ALL_DATASETS)} datasets, Q-matched):")
for cond in ['comprehend', 'extract_general', 'classify', 'extract_claims', 'scrambled_comprehend']:
    if cond in meta_results_A:
        m = meta_results_A[cond]
        sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
               else '*' if m['p'] < 0.05 else 'ns')
        print(f"     {cond:<28} pooled_d={m['pooled_d']:+.4f} ({sig}) "
              f"across {m['n_datasets']} datasets")

# 2. Best length
print(f"\n  2. Optimal prefix length (comprehend):")
if scaling_results.get('comprehend'):
    for L in PREFIX_LENGTHS:
        if L in scaling_results['comprehend']:
            sr = scaling_results['comprehend'][L]
            sig = ('***' if sr['p'] < 0.001 else '**' if sr['p'] < 0.01
                   else '*' if sr['p'] < 0.05 else 'ns')
            print(f"     L={L}: pooled_d={sr['pooled_d']:+.4f} ({sig}, "
                  f"n={sr['n_datasets']})")

# 3. Meaning growth
print(f"\n  3. Comprehend meaning effect by length:")
for L in PREFIX_LENGTHS:
    if L in meaning_by_length:
        m = meaning_by_length[L]
        sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
               else '*' if m['p'] < 0.05 else 'ns')
        print(f"     L={L}: d={m['d']:+.4f} ({sig}, n={m['n_datasets']})")

# 4. QuALITY MC accuracy
print(f"\n  4. QuALITY MC accuracy:")
print(f"     bare: {quality_accuracy.get('bare', 0):.1%}")
if quality_accuracy:
    best_qa = max(quality_accuracy.items(), key=lambda x: x[1])
    print(f"     best: {best_qa[0]} = {best_qa[1]:.1%}")

# 5. Dataset ranking at L=64
print(f"\n  5. Dataset ranking at L=64 (comprehend sem_d):")
for rank, (name, d_c, d_e, maw, mpw) in enumerate(ranking_at_64[:5], 1):
    print(f"     {rank}. {name}: d={d_c:+.3f}")
if len(ranking_at_64) > 5:
    name, d_c = ranking_at_64[-1][0], ranking_at_64[-1][1]
    print(f"     ...{len(ranking_at_64)}. {name}: d={d_c:+.3f}")

# 6. Answer-length prediction
print(f"\n  6. Answer-length prediction:")
print(f"     rho(mean_ans_words, d_comp@L=64) = {rho:+.3f}, p = {p_rho:.4f}")

# 7. Instruction ranking
print(f"\n  7. Instruction ranking (mean d@L=64 across 7 new datasets):")
instr_means = []
for instr in ['comprehend', 'extract_general', 'classify', 'extract_claims']:
    vals = list(instruction_comparison.get(instr, {}).values())
    if vals:
        instr_means.append((instr, np.mean(vals)))
instr_means.sort(key=lambda x: x[1], reverse=True)
for instr, m_d in instr_means:
    print(f"     {instr:<20} mean_d={m_d:+.3f}")

# 8. Hypothesis outcomes
print(f"\n  8. Hypothesis outcomes:")
for ds_name, hypothesis in [
    ('multirc', 'H1: MultiRC is DROP-like standout'),
    ('gsm8k', 'H2: GSM8K — complex reasoning vs short answers'),
    ('ropes', 'H3: ROPES — causal reasoning + short answers (BoolQ-like?)'),
    ('quality', 'H4: QuALITY MC accuracy improves with prefix'),
    ('quoref', 'H5: Quoref benefits from comprehend priming'),
    ('record', 'H6: ReCoRD entity comprehension benefits differently'),
    ('race_middle', 'H7: RACE-middle shows weaker effect than RACE-high'),
]:
    if ds_name in task_length and 64 in task_length[ds_name]:
        d_64 = task_length[ds_name][64]
        outcome = "SUPPORTED" if d_64 > 0.1 else "REFUTED" if d_64 < -0.1 else "MIXED"
    else:
        outcome = "N/A"
        d_64 = float('nan')
    print(f"     {hypothesis}")
    if outcome != "N/A":
        print(f"       -> {outcome} (d@L=64={d_64:+.3f})")
    else:
        print(f"       -> {outcome}")
""")


# ===== Cell 10: Save results =====
code(r"""# Cell 10: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'v4_exp06_hero_run',
    'model': MODEL_NAME,
    'scoring': 'bos_retained_repositioning_token_matched',
    'scoring_key': SCORING_KEY,
    'hard_fraction': HARD_FRAC,
    'new_datasets': NEW_DATASETS,
    'old_datasets': OLD_DATASETS,
    'all_datasets': ALL_DATASETS,
    'n_samples_per_dataset': N_SAMPLES,
    'n_hard_per_dataset': N_HARD,
    'prefix_lengths': PREFIX_LENGTHS,
    'common_max_doc': COMMON_MAX_DOC,
    'seed': SEED,
    'ds_seeds': DS_SEEDS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'instructions': {name: text for name, text in INSTRUCTIONS.items()},
    'phase_a_meta_analysis': meta_results_A,
    'phase_b_scaling': scaling_results,
    'pooled_decomposition_by_length': pooled_decomp_by_length,
    'task_length_interaction': task_length,
    'meaning_by_length': meaning_by_length,
    'ceiling_analysis': ceiling,
    'quality_accuracy': quality_accuracy,
    'cross_benchmark_ranking': [
        {'condition': c, 'length': l, 'pooled_d': float(d), 'se': float(se),
         'n_datasets': n_ds}
        for c, l, d, se, n_ds in all_combos
    ],
    'dataset_ranking_at_64': [
        {'dataset': name, 'd_comprehend': float(d_c), 'd_extract_general': float(d_e),
         'mean_answer_words': float(maw), 'mean_passage_words': float(mpw)}
        for name, d_c, d_e, maw, mpw in ranking_at_64
    ],
    'answer_length_correlation': {
        'spearman_rho': float(rho), 'p_value': float(p_rho),
        'n_datasets': len(ans_length_data),
    },
    'instruction_comparison': instruction_comparison,
    'per_dataset': per_dataset_analysis,
    'hard_metadata': {ds: hard_metadata[ds] for ds in NEW_DATASETS},
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'results.json'}")

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
out_path = "experiments/decoder_only/06/06_hero_run.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
