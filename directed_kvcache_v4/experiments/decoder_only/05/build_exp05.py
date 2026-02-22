#!/usr/bin/env python3
# Build Exp 05: Scaling the Prefix Effect — New Benchmarks + Length Optimization.
#
# Phase A: Q-matched scoring on 3 new benchmarks (DROP, BoolQ, RACE-high) with
# top Exp 04 conditions (bare, random_tokens, comprehend, extract_general,
# scrambled_comprehend) to test task generalization.
#
# Phase B: Fixed-length prefix scaling (L=32,64,128,256) across all 7 datasets
# with 4 prefix conditions + bare_trunc, testing how the prefix effect scales
# with length. Three-level decomposition at each length.
#
# RACE includes MC accuracy scoring (all 4 options scored, correct = lowest NLL).
#
# Loads Q-matched baselines from Exp 03/04 for existing 4 datasets.
# Scoring: BOS-retained repositioning + token-level prefix matching on
# Gemma 3 12B-IT (identical to Exps 02-04).
#
# SEED=42, N=400 per dataset, hard 40% = 160.

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/05", exist_ok=True)

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
md(r"""# Experiment 05: Scaling the Prefix Effect — New Benchmarks + Length Optimization

## Motivation

Exp 04 decomposed the extractor\_matched effect into three components:
- **Structural** (any tokens in cache): 39% of total effect
- **Vocabulary** (instruction tokens vs random): **73%** of total — the dominant factor
- **Meaning** (coherent order vs scrambled): **-11%** — meaning is negligible or slightly harmful

Key Exp 04 findings:
- `comprehend` is the best prefix (pooled d=+0.470 \*\*\*), beating extract\_general (d=+0.357)
- `comprehend` is the ONLY instruction with a significant positive meaning effect (d=+0.235)
- Scrambled instructions often match or beat coherent ones
- Extraction framing is NOT specifically better than other framings (H1 rejected)
- Longer instruction > shorter repeated (H5 refuted, d=-0.176)

**What we don't know:**
1. All experiments use Q-matched prefix length (mean 6-19 tokens) — prefix length has never been varied independently
2. All experiments test extractive QA only — does the effect generalize to reasoning, boolean QA, exam comprehension?
3. Does lower NLL translate to better generated answers?

## Design

### Two Experimental Phases

**Phase A: Q-matched scoring on new benchmarks** (comparison with Exp 04)
- Score 3 new datasets (DROP, BoolQ, RACE-high) with 5 conditions
- Tests: Does the vocabulary-dominated effect generalize to reasoning tasks?

**Phase B: Fixed-length prefix scaling across all 7 datasets**
- Fixed prefix lengths: L = 32, 64, 128, 256 tokens
- Conditions at each L: bare\_trunc, random\_tokens\_L, comprehend\_L, extract\_general\_L, scrambled\_comprehend\_L
- All docs truncated to common max\_doc for consistent cross-length comparison

### Conditions Summary

**Phase A (Q-matched, 3 new datasets only):** 5 conditions
1. `bare` — no prefix
2. `random_tokens_Q` — Q random tokens
3. `comprehend_Q` — comprehend instruction at Q tokens
4. `extract_general_Q` — extract instruction at Q tokens
5. `scrambled_comprehend_Q` — scrambled comprehend at Q tokens

**Phase B (fixed-length, all 7 datasets):** 4 lengths x 4 conditions + 1 bare = 17 per sample
- `bare_trunc` — bare on truncated doc (scored once)
- At each L in {32, 64, 128, 256}: random\_tokens\_L, comprehend\_L, extract\_general\_L, scrambled\_comprehend\_L

**Loaded from Exp 03/04 (existing 4 datasets):** Q-matched bare, random\_tokens, comprehend, extract\_general, scrambled\_comprehend

### New Benchmarks

| Dataset | Task type | Answer format | Why it's harder |
|---------|-----------|---------------|-----------------|
| **DROP** | Discrete reasoning | Numbers, dates, spans | Counting, arithmetic, sorting over text |
| **BoolQ** | Boolean reasoning | "Yes" / "No" | Requires inference, not just extraction |
| **RACE-high** | Exam comprehension | MC text (4 options) | English exam questions, deeper comprehension |

### Three-Level Decomposition at Each Length

For each length L:
- **Structural\_L** = NLL(bare\_trunc) - NLL(random\_L)
- **Vocabulary\_L** = NLL(random\_L) - NLL(scrambled\_comprehend\_L)
- **Meaning\_L** = NLL(scrambled\_comprehend\_L) - NLL(comprehend\_L)""")


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
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d
from lib.data import count_words

SEED = 42
N_SAMPLES = 400      # per dataset
HARD_FRAC = 0.40     # top 40% by bare NLL
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp05")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP04_DIR = Path("../../../results/decoder_only/exp04")

# All 7 datasets
OLD_DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa']
NEW_DATASETS = ['drop', 'boolq', 'race_high']
ALL_DATASETS = OLD_DATASETS + NEW_DATASETS

# Phase B prefix lengths
PREFIX_LENGTHS = [32, 64, 128, 256]

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

print(f"Exp 05: Scaling the Prefix Effect — New Benchmarks + Length Optimization")
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


# --- Instruction definitions (from Exp 04) ---
INSTRUCTIONS = {
    'extract_general': "Extract all key data points, facts, entities, and specific attributes from the following text.",
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
}

# Pre-tokenize instructions
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")

# Phase A Q-matched conditions (for new datasets)
PHASE_A_PREFIX_CONDS = ['random_tokens', 'comprehend', 'extract_general',
                        'scrambled_comprehend']

# Phase B fixed-length conditions (for all datasets)
PHASE_B_PREFIX_CONDS = ['random_tokens', 'comprehend', 'extract_general',
                        'scrambled_comprehend']

# Per-dataset seeds (extending Exp 03 pattern)
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
    'boolq': SEED + 500,
    'race_high': SEED + 600,
}

SCORING_KEY = 'bos_retained_token_matched_v05'

special_ids = set(tokenizer.all_special_ids)

print(f"\nSetup complete. Functions: score, make_prefix, scramble_prefix")
print(f"Phase A conditions: bare + {PHASE_A_PREFIX_CONDS}")
print(f"Phase B conditions: bare_trunc + {len(PREFIX_LENGTHS)} lengths x {len(PHASE_B_PREFIX_CONDS)} prefixes")
""")


# ===== Cell 3: Load existing datasets + Exp 03/04 baselines =====
code(r"""# Cell 3: Load existing 4 datasets + Exp 03/04 Q-matched baselines
from datasets import load_dataset

print("=" * 70)
print("LOADING EXISTING 4 DATASETS + EXP 03/04 BASELINES")
print("=" * 70)

hard_nlls = {}      # ds_name -> {cond_name: np.array}
hard_metadata = {}  # ds_name -> dict
hard_samples = {}   # ds_name -> list of hard sample dicts
all_samples = {}    # ds_name -> list of N_SAMPLES sample dicts

# ================================================================
# PART 1: Load MS MARCO from Exp 02
# ================================================================
print("\n--- MS MARCO from Exp 02 + Exp 03 baselines ---")
assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES

msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])

# Load passage text
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

# Verify alignment
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

# Load baselines: bare, random_tokens from Exp 03
hard_nlls['ms_marco'] = {}
hard_nlls['ms_marco']['bare'] = msmarco_bare[msmarco_hard_idx]
for cond in ['random_tokens']:
    hard_nlls['ms_marco'][cond] = np.array(
        [exp02_results[i][f'nll_{cond}'] for i in msmarco_hard_idx])
# extract_general = extractor_matched from Exp 03
hard_nlls['ms_marco']['extract_general'] = np.array(
    [exp02_results[i]['nll_extractor_matched'] for i in msmarco_hard_idx])

# Load comprehend + scrambled_comprehend from Exp 04
exp04_ckpt_msmarco = json.loads(
    (EXP04_DIR / "checkpoint_ms_marco.json").read_text())
exp04_results_msmarco = exp04_ckpt_msmarco['results']
hard_nlls['ms_marco']['comprehend'] = np.array(
    [r['nll_comprehend'] for r in exp04_results_msmarco])
hard_nlls['ms_marco']['scrambled_comprehend'] = np.array(
    [r['nll_scrambled_comprehend'] for r in exp04_results_msmarco])
hard_nlls['ms_marco']['scrambled_extract_general'] = np.array(
    [r['nll_scrambled_extract_general'] for r in exp04_results_msmarco])

hard_metadata['ms_marco'] = {
    'n_total': N_SAMPLES, 'n_hard': N_HARD, 'source': 'exp02_reuse',
    'mean_passage_words': float(np.mean([s['word_count'] for s in hs_msmarco])),
}
print(f"  MS MARCO: {N_HARD} hard, loaded bare/random_tokens/extract_general/"
      f"comprehend/scrambled_comprehend")

del exp02_ckpt, exp02_results, exp04_ckpt_msmarco, exp04_results_msmarco
gc.collect()

# ================================================================
# PART 2: Load 3 existing datasets (SQuAD, TriviaQA, HotpotQA)
# ================================================================
# --- SQuAD 2.0 ---
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

# --- TriviaQA ---
print("--- TriviaQA ---")
ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
trivia_candidates = []
for item in ds_trivia:
    entity_pages = item.get('entity_pages', {})
    wiki_contexts = entity_pages.get('wiki_context', [])
    if not wiki_contexts or not wiki_contexts[0]:
        continue
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
    passage_lower = passage.lower()
    found = answer_val.lower() in passage_lower
    if not found:
        for alias in aliases:
            if alias.lower() in passage_lower:
                found = True
                break
    if not found:
        continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        trivia_candidates.append({
            'passage': passage, 'query': query, 'answer': answer_val,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['triviaqa'])
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
all_samples['triviaqa'] = [trivia_candidates[i] for i in tr_indices]
del ds_trivia, trivia_candidates
gc.collect()

# --- HotpotQA ---
print("--- HotpotQA ---")
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
# PART 3: Load Exp 03/04 baselines for SQuAD, TriviaQA, HotpotQA
# ================================================================
print("\n--- Loading Exp 03/04 baselines ---")
for ds_name in ['squad_v2', 'triviaqa', 'hotpotqa']:
    samples_ds = all_samples[ds_name]

    # Bare NLLs from Exp 03
    bare_path = EXP03_DIR / f"bare_{ds_name}.json"
    bare_ckpt = json.loads(bare_path.read_text())
    assert bare_ckpt.get('n_total') == N_SAMPLES
    bare_nlls_all = bare_ckpt['bare_nlls']

    # Verify alignment
    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in samples_ds[:len(saved_queries)]]
    assert saved_queries == current_queries, \
        f"{ds_name}: query alignment mismatch with Exp 03"

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

    # Exp 03 baselines
    ckpt03 = json.loads((EXP03_DIR / f"checkpoint_{ds_name}.json").read_text())
    exp03_results = ckpt03['results']
    assert len(exp03_results) == N_HARD

    hard_nlls[ds_name] = {}
    hard_nlls[ds_name]['bare'] = bare_arr[h_idx]
    hard_nlls[ds_name]['random_tokens'] = np.array(
        [r['nll_random_tokens'] for r in exp03_results])
    hard_nlls[ds_name]['extract_general'] = np.array(
        [r['nll_extractor_matched'] for r in exp03_results])

    # Exp 04 baselines (comprehend + scrambled_comprehend)
    ckpt04 = json.loads((EXP04_DIR / f"checkpoint_{ds_name}.json").read_text())
    exp04_results = ckpt04['results']
    assert len(exp04_results) == N_HARD
    hard_nlls[ds_name]['comprehend'] = np.array(
        [r['nll_comprehend'] for r in exp04_results])
    hard_nlls[ds_name]['scrambled_comprehend'] = np.array(
        [r['nll_scrambled_comprehend'] for r in exp04_results])
    hard_nlls[ds_name]['scrambled_extract_general'] = np.array(
        [r['nll_scrambled_extract_general'] for r in exp04_results])

    hard_metadata[ds_name] = {
        'n_total': N_SAMPLES, 'n_hard': N_HARD, 'source': 'exp03_04_reuse',
        'mean_passage_words': float(np.mean([s['word_count'] for s in hs])),
    }

    print(f"  {ds_name}: {N_HARD} hard, loaded 6 Q-matched baselines")

del bare_ckpt, ckpt03, exp03_results, ckpt04, exp04_results
gc.collect()

# Summary
print("\n" + "=" * 70)
print("Existing dataset loading summary:")
for ds_name in OLD_DATASETS:
    n_h = len(hard_samples[ds_name])
    conds = sorted(hard_nlls[ds_name].keys())
    print(f"  {ds_name}: {n_h} hard, conditions: {conds}")
""")


# ===== Cell 4: Load 3 new datasets (DROP, BoolQ, RACE) =====
code(r"""# Cell 4: Load DROP, BoolQ, RACE-high
from datasets import load_dataset

print("=" * 70)
print("LOADING 3 NEW DATASETS")
print("=" * 70)

# ---- DROP ----
print("\n--- DROP (ucinlp/drop, validation) ---")
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

print(f"  DROP candidates: {len(drop_candidates)}")
np.random.seed(DS_SEEDS['drop'])
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES]
all_samples['drop'] = [drop_candidates[i] for i in drop_indices]
del ds_drop, drop_candidates
gc.collect()

# ---- BoolQ ----
print("\n--- BoolQ (google/boolq, validation) ---")
ds_boolq = load_dataset("google/boolq", split="validation")

boolq_candidates = []
for item in ds_boolq:
    passage = item['passage']
    question = item['question']
    answer = "Yes" if item['answer'] else "No"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        boolq_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })

print(f"  BoolQ candidates: {len(boolq_candidates)}")
np.random.seed(DS_SEEDS['boolq'])
boolq_indices = np.random.permutation(len(boolq_candidates))[:N_SAMPLES]
all_samples['boolq'] = [boolq_candidates[i] for i in boolq_indices]
del ds_boolq, boolq_candidates
gc.collect()

# ---- RACE-high ----
print("\n--- RACE (race, high, test) ---")
ds_race = load_dataset("race", "high", split="test")

race_candidates = []
for item in ds_race:
    passage = item['article']
    question = item['question']
    correct_idx = ord(item['answer']) - ord('A')
    options = item['options']
    if correct_idx < 0 or correct_idx >= len(options):
        continue
    answer = options[correct_idx]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        race_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
            'all_options': options,
            'correct_idx': correct_idx,
        })

print(f"  RACE-high candidates: {len(race_candidates)}")
np.random.seed(DS_SEEDS['race_high'])
race_indices = np.random.permutation(len(race_candidates))[:N_SAMPLES]
all_samples['race_high'] = [race_candidates[i] for i in race_indices]
del ds_race, race_candidates
gc.collect()

# Summary
print("\n" + "=" * 70)
print("New dataset loading summary:")
for ds_name in NEW_DATASETS:
    samps = all_samples[ds_name]
    print(f"\n  {ds_name}: {len(samps)} samples")
    print(f"    Mean passage words: {np.mean([s['word_count'] for s in samps]):.0f}")
    print(f"    Mean answer words: {np.mean([count_words(s['answer']) for s in samps]):.0f}")
    print(f"    Example query: {samps[0]['query'][:70]}...")
    print(f"    Example answer: {samps[0]['answer'][:70]}...")
    if ds_name == 'race_high':
        print(f"    Example options: {[o[:30] for o in samps[0]['all_options']]}")
        print(f"    Correct idx: {samps[0]['correct_idx']}")
""")


# ===== Cell 5: Bare scoring + hard selection for new datasets =====
code(r"""# Cell 5: Bare NLL scoring for 3 new datasets + hard 40% selection
print("=" * 70)
print("BARE SCORING — 3 new datasets x 400 samples")
print("=" * 70)

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


# ===== Cell 6: Phase A — Q-matched scoring for new datasets =====
code(r"""# Cell 6: Phase A — Q-matched scoring for 3 new datasets
# Conditions: random_tokens, comprehend, extract_general, scrambled_comprehend
print("=" * 70)
print("PHASE A: Q-MATCHED SCORING — 3 new datasets x ~160 hard x 4 prefix conditions")
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

    print(f"\n--- Phase A: {ds_name} ({n_hard} hard x 4 conditions) ---")

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
    print(f"\n  {ds_name}:")
    print(f"    bare:            {bare.mean():.4f}")
    print(f"    random_tokens:   {rand.mean():.4f}  d_bare={cohens_d(bare - rand):+.3f}")
    print(f"    comprehend:      {comp.mean():.4f}  sem_d={cohens_d(rand - comp):+.3f}")
    print(f"    extract_general: {ext.mean():.4f}  sem_d={cohens_d(rand - ext):+.3f}")
""")


# ===== Cell 7: Phase B — Fixed-length prefix scoring =====
code(r"""# Cell 7: Phase B — Fixed-length prefix scoring across all 7 datasets
# L = 32, 64, 128, 256; conditions: bare_trunc, random_tokens, comprehend,
# extract_general, scrambled_comprehend at each L
print("=" * 70)
print("PHASE B: FIXED-LENGTH PREFIX SCORING")
print(f"  Lengths: {PREFIX_LENGTHS}")
print(f"  Datasets: {ALL_DATASETS}")
print(f"  COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print("=" * 70)

# Storage for Phase B NLLs
# phase_b_nlls[ds_name][cond_key] = np.array of N_HARD values
# cond_key examples: 'bare_trunc', 'random_tokens_32', 'comprehend_128', etc.
phase_b_nlls = {ds: {} for ds in ALL_DATASETS}

for ds_name in ALL_DATASETS:
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
        np.random.seed(DS_SEEDS.get(ds_name, SEED) + 2000)
        pyrandom.seed(DS_SEEDS.get(ds_name, SEED) + 2000)

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
# RACE MC accuracy scoring
# ================================================================
print("\n" + "=" * 70)
print("RACE MC ACCURACY SCORING")
print("=" * 70)

race_mc_results = []  # list of dicts with NLLs for all 4 options under each condition

hs_race = hard_samples['race_high']
n_hard_race = len(hs_race)
race_mc_ckpt_path = RESULTS_DIR / "race_mc.json"

start_idx_mc = 0
if race_mc_ckpt_path.exists():
    mc_ckpt = json.loads(race_mc_ckpt_path.read_text())
    if (mc_ckpt.get('scoring') == SCORING_KEY and
        mc_ckpt.get('n_hard') == n_hard_race):
        saved_queries = [r['query'][:50] for r in mc_ckpt.get('results', [])]
        current_queries = [s['query'][:50] for s in hs_race[:len(saved_queries)]]
        if saved_queries == current_queries:
            race_mc_results = mc_ckpt['results']
            start_idx_mc = len(race_mc_results)
            print(f"  Resuming from checkpoint: {start_idx_mc}/{n_hard_race}")

if start_idx_mc < n_hard_race:
    t0 = time.time()

    # Define conditions to score MC for
    # Q-matched + Phase B fixed-length
    mc_conditions = ['bare']  # bare (no prefix, no truncation)
    mc_conditions += ['bare_trunc']  # bare with truncation
    for cond in PHASE_A_PREFIX_CONDS:
        mc_conditions.append(f'qmatched_{cond}')
    for L in PREFIX_LENGTHS:
        for cond in PHASE_B_PREFIX_CONDS:
            mc_conditions.append(f'{cond}_{L}')

    np.random.seed(DS_SEEDS['race_high'] + 3000)
    pyrandom.seed(DS_SEEDS['race_high'] + 3000)

    for i in tqdm(range(start_idx_mc, n_hard_race), initial=start_idx_mc,
                  total=n_hard_race, desc="RACE MC"):
        s = hs_race[i]
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

                scr_seed_L = hash(f'scrambled_comprehend_{L}') % (2**31) + i
                scr_prefix_L = scramble_prefix(comp_prefix_L, scr_seed_L)
                result[f'nll_scrambled_comprehend_{L}_opt{opt_i}'] = score(
                    s['passage'], s['query'], opt_text,
                    prefix_token_ids=scr_prefix_L,
                    max_doc_override=COMMON_MAX_DOC)

        race_mc_results.append(result)

        if (i + 1) % 20 == 0 or i == n_hard_race - 1:
            mc_ckpt = {
                'scoring': SCORING_KEY,
                'n_hard': n_hard_race,
                'results': race_mc_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            race_mc_ckpt_path.write_text(json.dumps(mc_ckpt))
            elapsed = time.time() - t0
            done = i - start_idx_mc + 1
            eta = (n_hard_race - i - 1) * elapsed / done if done > 0 else 0
            tqdm.write(f"  MC Checkpoint {i+1}/{n_hard_race} | "
                       f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  RACE MC scoring complete in {elapsed/60:.1f} min")
else:
    print(f"  Loaded {len(race_mc_results)} cached MC results")

gc.collect()
torch.cuda.empty_cache()
print(f"\nPhase B + RACE MC scoring complete for all datasets.")
""")


# ===== Cell 8: Per-dataset results =====
code(r"""# Cell 8: Per-dataset results — Phase A + Phase B tables
print("=" * 70)
print("PER-DATASET RESULTS")
print("=" * 70)

per_dataset_analysis = {}

# ================================================================
# PART 1: Phase A — Q-matched results (all 7 datasets)
# ================================================================
print("\n" + "=" * 70)
print("PHASE A: Q-MATCHED RESULTS (all 7 datasets)")
print("=" * 70)

PHASE_A_CONDS = ['bare', 'random_tokens', 'comprehend', 'extract_general',
                 'scrambled_comprehend']

for ds_name in ALL_DATASETS:
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
# PART 2: Phase B — Fixed-length results (all 7 datasets)
# ================================================================
print("\n" + "=" * 70)
print("PHASE B: FIXED-LENGTH RESULTS (all 7 datasets)")
print("=" * 70)

for ds_name in ALL_DATASETS:
    pb = phase_b_nlls[ds_name]
    bare_t = pb['bare_trunc']
    n_hard = len(bare_t)

    print(f"\n--- {ds_name.upper()} ({n_hard} hard samples, fixed-length) ---")
    print(f"  bare_trunc NLL: {bare_t.mean():.4f}")

    print(f"\n  {'L':>5} {'rand NLL':>9} {'comp NLL':>9} {'ext NLL':>9} "
          f"{'scr NLL':>9} {'d_comp':>8} {'d_ext':>8} {'d_scr':>8}")
    print(f"  {'-'*72}")

    phase_b_analysis = {}
    for L in PREFIX_LENGTHS:
        rand_L = pb[f'random_tokens_{L}']
        comp_L = pb[f'comprehend_{L}']
        ext_L = pb[f'extract_general_{L}']
        scr_L = pb[f'scrambled_comprehend_{L}']

        d_comp = cohens_d(rand_L - comp_L)
        d_ext = cohens_d(rand_L - ext_L)
        d_scr = cohens_d(rand_L - scr_L)

        print(f"  {L:>5} {rand_L.mean():>9.4f} {comp_L.mean():>9.4f} "
              f"{ext_L.mean():>9.4f} {scr_L.mean():>9.4f} "
              f"{d_comp:>+8.3f} {d_ext:>+8.3f} {d_scr:>+8.3f}")

        # Three-level decomposition at this length
        structural = (bare_t - rand_L).mean()
        vocab = (rand_L - scr_L).mean()
        meaning = (scr_L - comp_L).mean()
        total = structural + vocab + meaning

        phase_b_analysis[L] = {
            'structural': float(structural), 'vocabulary': float(vocab),
            'meaning': float(meaning), 'total': float(total),
            'd_comprehend': float(d_comp), 'd_extract_general': float(d_ext),
            'd_scrambled_comprehend': float(d_scr),
            'mean_nll': {
                'bare_trunc': float(bare_t.mean()),
                'random_tokens': float(rand_L.mean()),
                'comprehend': float(comp_L.mean()),
                'extract_general': float(ext_L.mean()),
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


# ===== Cell 9: Cross-dataset analysis =====
code(r"""# Cell 9: Cross-dataset analysis — scaling curves, decomposition, RACE accuracy
print("=" * 70)
print("CROSS-DATASET ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Phase A — Q-matched meta-analysis (all 7 datasets)
# ================================================================
print("\n--- PART 1: Phase A Q-Matched Meta-Analysis ---")

PHASE_A_META_CONDS = ['comprehend', 'extract_general', 'scrambled_comprehend']

print(f"\n  {'Condition':<28} {'pooled_d':>9} {'SE':>8} {'z':>8} "
      f"{'p':>10} {'95% CI':>16} {'sig':>4}")
print(f"  {'-'*86}")

meta_results_A = {}
for cond in PHASE_A_META_CONDS:
    ds_effects = []
    for ds_name in ALL_DATASETS:
        nlls = hard_nlls[ds_name]
        if cond not in nlls or 'random_tokens' not in nlls:
            continue
        sem_delta = nlls['random_tokens'] - nlls[cond]
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
    ci_lo = pooled_d - 1.96 * pooled_se
    ci_hi = pooled_d + 1.96 * pooled_se
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')

    print(f"  {cond:<28} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
          f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4}")
    meta_results_A[cond] = {
        'pooled_d': float(pooled_d), 'se': float(pooled_se),
        'z': float(z), 'p': float(p),
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
        'n_datasets': len(ds_effects),
    }

# Per-dataset d for cross-dataset consistency
print(f"\n  {'Condition':<28}", end="")
for ds_name in ALL_DATASETS:
    print(f" {ds_name[:8]:>8}", end="")
print(f"  {'mean':>8}")
print(f"  {'-'*100}")

for cond in PHASE_A_META_CONDS:
    row = f"  {cond:<28}"
    ds_vals = []
    for ds_name in ALL_DATASETS:
        nlls = hard_nlls[ds_name]
        if cond not in nlls or 'random_tokens' not in nlls:
            row += f" {'N/A':>8}"
            continue
        d = cohens_d(nlls['random_tokens'] - nlls[cond])
        row += f" {d:>+8.3f}"
        ds_vals.append(d)
    if ds_vals:
        row += f"  {np.mean(ds_vals):>+8.3f}"
    print(row)

# ================================================================
# PART 2: Phase B — Length Scaling Curves
# ================================================================
print(f"\n--- PART 2: Length Scaling Curves (pooled across datasets) ---")

scaling_results = {}
for cond_base in ['comprehend', 'extract_general', 'scrambled_comprehend']:
    print(f"\n  {cond_base}:")
    print(f"  {'L':>5} {'pooled_d':>9} {'SE':>8} {'z':>8} "
          f"{'p':>10} {'sig':>4}")
    print(f"  {'-'*48}")

    scaling_results[cond_base] = {}
    for L in PREFIX_LENGTHS:
        ds_effects = []
        for ds_name in ALL_DATASETS:
            pb = phase_b_nlls[ds_name]
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
              f"{p:>10.2e} {sig:>4}")
        scaling_results[cond_base][L] = {
            'pooled_d': float(pooled_d), 'se': float(pooled_se),
            'z': float(z), 'p': float(p),
        }

# ================================================================
# PART 3: Three-Level Decomposition x Length (pooled)
# ================================================================
print(f"\n--- PART 3: Three-Level Decomposition x Length (pooled) ---")
print(f"  {'L':>5} {'Structural':>12} {'Vocabulary':>12} "
      f"{'Meaning':>12} {'Total':>12}")
print(f"  {'-'*56}")

pooled_decomp_by_length = {}
for L in PREFIX_LENGTHS:
    structs, vocabs, meanings, totals = [], [], [], []
    for ds_name in ALL_DATASETS:
        pb = phase_b_nlls[ds_name]
        bare_t = pb['bare_trunc']
        rand_L = pb[f'random_tokens_{L}']
        scr_L = pb[f'scrambled_comprehend_{L}']
        comp_L = pb[f'comprehend_{L}']

        structs.append((bare_t - rand_L).mean())
        vocabs.append((rand_L - scr_L).mean())
        meanings.append((scr_L - comp_L).mean())
        totals.append((bare_t - comp_L).mean())

    s_m, v_m, m_m, t_m = np.mean(structs), np.mean(vocabs), np.mean(meanings), np.mean(totals)
    print(f"  {L:>5} {s_m:>+12.4f} {v_m:>+12.4f} {m_m:>+12.4f} {t_m:>+12.4f}")
    pooled_decomp_by_length[L] = {
        'structural': float(s_m), 'vocabulary': float(v_m),
        'meaning': float(m_m), 'total': float(t_m),
    }

# Percentage breakdown
print(f"\n  Percentage breakdown:")
print(f"  {'L':>5} {'Struct%':>9} {'Vocab%':>9} {'Meaning%':>9}")
print(f"  {'-'*36}")
for L in PREFIX_LENGTHS:
    d = pooled_decomp_by_length[L]
    t = d['total']
    if abs(t) > 0.001:
        print(f"  {L:>5} {d['structural']/t*100:>8.0f}% "
              f"{d['vocabulary']/t*100:>8.0f}% {d['meaning']/t*100:>8.0f}%")

# ================================================================
# PART 4: Task x Length Interaction
# ================================================================
print(f"\n--- PART 4: Task x Length Interaction (comprehend sem_d) ---")
print(f"  {'Dataset':<16}", end="")
for L in PREFIX_LENGTHS:
    print(f" {'L='+str(L):>8}", end="")
print(f"  {'trend':>8}")
print(f"  {'-'*60}")

task_length = {}
for ds_name in ALL_DATASETS:
    pb = phase_b_nlls[ds_name]
    row = f"  {ds_name:<16}"
    ds_vals = []
    for L in PREFIX_LENGTHS:
        d = cohens_d(pb[f'random_tokens_{L}'] - pb[f'comprehend_{L}'])
        row += f" {d:>+8.3f}"
        ds_vals.append(d)
    # Simple trend: increasing or decreasing?
    if len(ds_vals) >= 2:
        rho, _ = stats.spearmanr(PREFIX_LENGTHS, ds_vals)
        trend = "UP" if rho > 0.5 else "DOWN" if rho < -0.5 else "FLAT"
    else:
        trend = "N/A"
    row += f"  {trend:>8}"
    print(row)
    task_length[ds_name] = {L: float(d) for L, d in zip(PREFIX_LENGTHS, ds_vals)}

# ================================================================
# PART 5: RACE MC Accuracy
# ================================================================
print(f"\n--- PART 5: RACE MC Accuracy ---")

def compute_accuracy(results, cond_prefix, n_options=4):
    correct = 0
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
    return correct / len(results) if results else 0.0

print(f"\n  {'Condition':<36} {'Accuracy':>8} {'N':>5}")
print(f"  {'-'*52}")

race_accuracy = {}
# Bare
acc_bare = compute_accuracy(race_mc_results, 'bare')
print(f"  {'bare':<36} {acc_bare:>7.1%} {len(race_mc_results):>5}")
race_accuracy['bare'] = float(acc_bare)

# Bare truncated
acc_bare_t = compute_accuracy(race_mc_results, 'bare_trunc')
print(f"  {'bare_trunc':<36} {acc_bare_t:>7.1%} {len(race_mc_results):>5}")
race_accuracy['bare_trunc'] = float(acc_bare_t)

# Q-matched
for cond in PHASE_A_PREFIX_CONDS:
    acc = compute_accuracy(race_mc_results, f'qmatched_{cond}')
    print(f"  {'qmatched_' + cond:<36} {acc:>7.1%} {len(race_mc_results):>5}")
    race_accuracy[f'qmatched_{cond}'] = float(acc)

# Fixed-length
for L in PREFIX_LENGTHS:
    for cond in PHASE_B_PREFIX_CONDS:
        acc = compute_accuracy(race_mc_results, f'{cond}_{L}')
        print(f"  {f'{cond}_{L}':<36} {acc:>7.1%} {len(race_mc_results):>5}")
        race_accuracy[f'{cond}_{L}'] = float(acc)

# ================================================================
# PART 6: Cross-Benchmark Ranking
# ================================================================
print(f"\n--- PART 6: Cross-Benchmark Ranking (best prefix x length) ---")

# Find the condition+length with highest pooled semantic delta d
all_combos = []
for cond_base in ['comprehend', 'extract_general']:
    for L in PREFIX_LENGTHS:
        ds_effects = []
        for ds_name in ALL_DATASETS:
            pb = phase_b_nlls[ds_name]
            sem_delta = pb[f'random_tokens_{L}'] - pb[f'{cond_base}_{L}']
            n = len(sem_delta)
            d = cohens_d(sem_delta)
            se = np.sqrt(1.0/n + d**2 / (2.0*n))
            ds_effects.append((d, se, n))
        weights = [1.0 / (se**2) for _, se, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        all_combos.append((cond_base, L, pooled_d, pooled_se))

all_combos.sort(key=lambda x: x[2], reverse=True)
print(f"\n  {'Rank':>4} {'Condition':<28} {'L':>5} {'pooled_d':>9} {'SE':>8}")
print(f"  {'-'*58}")
for rank, (cond, L, d, se) in enumerate(all_combos, 1):
    print(f"  {rank:>4} {cond:<28} {L:>5} {d:>+9.4f} {se:>8.4f}")

# ================================================================
# PART 7: Comprehend Meaning at Length
# ================================================================
print(f"\n--- PART 7: Comprehend Meaning Effect at Each Length ---")
print(f"  {'L':>5} {'pooled_d':>9} {'SE':>8} {'z':>8} {'p':>10} {'sig':>4}")
print(f"  {'-'*48}")

meaning_by_length = {}
for L in PREFIX_LENGTHS:
    ds_diffs = []
    for ds_name in ALL_DATASETS:
        pb = phase_b_nlls[ds_name]
        diff = pb[f'scrambled_comprehend_{L}'] - pb[f'comprehend_{L}']
        ds_diffs.append(diff)
    pooled_diff = np.concatenate(ds_diffs)
    d = cohens_d(pooled_diff)
    _, p = stats.ttest_1samp(pooled_diff, 0)
    se = np.sqrt(1.0/len(pooled_diff) + d**2/(2.0*len(pooled_diff)))
    z = d / se if se > 0 else 0.0
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')
    print(f"  {L:>5} {d:>+9.4f} {se:>8.4f} {z:>+8.2f} {p:>10.2e} {sig:>4}")
    meaning_by_length[L] = {'d': float(d), 'p': float(p), 'se': float(se)}

# ================================================================
# PART 8: Ceiling Analysis
# ================================================================
print(f"\n--- PART 8: Ceiling Analysis ---")
print(f"  Maximum achievable total effect (bare_trunc - comprehend_L):")
print(f"  {'L':>5} {'pooled mean delta':>18} {'max ds delta':>14} {'as % of bare':>14}")
print(f"  {'-'*56}")

ceiling = {}
for L in PREFIX_LENGTHS:
    ds_deltas = []
    ds_pcts = []
    for ds_name in ALL_DATASETS:
        pb = phase_b_nlls[ds_name]
        delta = (pb['bare_trunc'] - pb[f'comprehend_{L}']).mean()
        bare_mean = pb['bare_trunc'].mean()
        pct = delta / bare_mean * 100 if bare_mean > 0 else 0
        ds_deltas.append(delta)
        ds_pcts.append(pct)
    mean_delta = np.mean(ds_deltas)
    max_delta = np.max(ds_deltas)
    mean_pct = np.mean(ds_pcts)
    print(f"  {L:>5} {mean_delta:>+18.4f} {max_delta:>+14.4f} {mean_pct:>13.1f}%")
    ceiling[L] = {
        'mean_delta': float(mean_delta), 'max_delta': float(max_delta),
        'mean_pct': float(mean_pct),
    }

# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 70)
print("VERDICT — Exp 05: Scaling the Prefix Effect")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"Datasets: {len(ALL_DATASETS)} ({', '.join(ALL_DATASETS)})")
print(f"Hard selection: top {HARD_FRAC*100:.0f}% by bare NLL")

print(f"\n--- Key Findings ---")

# 1. Task generalization
print(f"\n  1. Task generalization (Phase A, Q-matched):")
for cond in PHASE_A_META_CONDS:
    if cond in meta_results_A:
        m = meta_results_A[cond]
        sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
               else '*' if m['p'] < 0.05 else 'ns')
        print(f"     {cond:<28} pooled_d={m['pooled_d']:+.4f} ({sig}) "
              f"across {m['n_datasets']} datasets")

# 2. Optimal length
print(f"\n  2. Optimal prefix length (Phase B, comprehend):")
if scaling_results.get('comprehend'):
    best_L = max(scaling_results['comprehend'].items(),
                 key=lambda x: x[1]['pooled_d'])
    print(f"     Best L={best_L[0]}: pooled_d={best_L[1]['pooled_d']:+.4f}")

# 3. Meaning growth
print(f"\n  3. Comprehend meaning effect by length:")
for L in PREFIX_LENGTHS:
    if L in meaning_by_length:
        m = meaning_by_length[L]
        sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
               else '*' if m['p'] < 0.05 else 'ns')
        print(f"     L={L}: d={m['d']:+.4f} ({sig})")

# 4. RACE accuracy
print(f"\n  4. RACE MC accuracy:")
print(f"     bare: {race_accuracy.get('bare', 0):.1%}")
best_race = max(race_accuracy.items(), key=lambda x: x[1])
print(f"     best: {best_race[0]} = {best_race[1]:.1%}")

# 5. Best overall combo
if all_combos:
    best = all_combos[0]
    print(f"\n  5. Best overall prefix x length:")
    print(f"     {best[0]} @ L={best[1]}: pooled_d={best[2]:+.4f}")
""")


# ===== Cell 10: Save results =====
code(r"""# Cell 10: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'v4_exp05_prefix_scaling',
    'model': MODEL_NAME,
    'scoring': 'bos_retained_repositioning_token_matched',
    'hard_fraction': HARD_FRAC,
    'datasets': ALL_DATASETS,
    'old_datasets': OLD_DATASETS,
    'new_datasets': NEW_DATASETS,
    'n_samples_per_dataset': N_SAMPLES,
    'n_hard_per_dataset': N_HARD,
    'prefix_lengths': PREFIX_LENGTHS,
    'common_max_doc': COMMON_MAX_DOC,
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'instructions': {name: text for name, text in INSTRUCTIONS.items()},
    'phase_a_meta_analysis': meta_results_A,
    'phase_b_scaling': scaling_results,
    'pooled_decomposition_by_length': pooled_decomp_by_length,
    'task_length_interaction': task_length,
    'meaning_by_length': meaning_by_length,
    'ceiling_analysis': ceiling,
    'race_accuracy': race_accuracy,
    'cross_benchmark_ranking': [
        {'condition': c, 'length': l, 'pooled_d': float(d), 'se': float(se)}
        for c, l, d, se in all_combos
    ],
    'per_dataset': per_dataset_analysis,
    'hard_metadata': {ds: hard_metadata[ds] for ds in ALL_DATASETS},
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
out_path = "experiments/decoder_only/05/05_prefix_scaling.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
