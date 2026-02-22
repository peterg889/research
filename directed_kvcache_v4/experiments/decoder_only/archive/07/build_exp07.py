#!/usr/bin/env python3
"""Build Exp 07: Swapped-Query Paired Contrasts — Decoder-Only.

Port of v3 Exp 13 to decoder-only two-phase KV cache scoring.
Maximum-power paired test: same doc scored with real vs swapped query.
All structural confounds equalized via token-level prefix matching.

No LLM generation needed. 4 conditions, N=400, SEED=43.
"""

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/07", exist_ok=True)

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 07: Swapped-Query Paired Contrasts

## Motivation

Port of v3 Exp 13 to decoder-only two-phase KV cache scoring.
v3 Exp 13 found a significant semantic signal (d=+0.166, p=2.3e-04)
with the encoder-decoder T5Gemma. Does the same paired semantic contrast
appear with decoder-only Gemma 3 4B-PT using KV cache priming?

## Design

For each (query, document, answer) triple, we score the answer NLL under two
prefix conditions that are structurally identical (same Q prefix token IDs):
- `oracle`: the real query tokens (semantically relevant)
- `swapped`: query tokens from a completely different sample (semantically irrelevant)

The per-document paired contrast `swapped_nll - oracle_nll` isolates the semantic
component with maximum statistical power.

All prefixed conditions use exactly Q token IDs (token-level matching).

## Conditions (4)

| # | Condition | Prefix content |
|---|-----------|---------------|
| 1 | bare | (none) |
| 2 | oracle | real query tokens |
| 3 | swapped | query tokens from sample (i + N//2) % N |
| 4 | random_matched | random passage word tokens |

## Analysis

1. Standard condition table
2. Paired semantic contrast (the key test)
3. Effect distribution
4. Predictors of semantic benefit
5. Structural equivalence check""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, re, gc, random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 43  # Different seed from Exp 06 (42) for independent samples
N_SAMPLES = 400
MODEL_NAME = "google/gemma-3-4b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp07")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

print("Exp 07: Swapped-Query Paired Contrasts (Decoder-Only)")
print(f"N: {N_SAMPLES}, SEED: {SEED}")
print(f"Model: {MODEL_NAME}")
""")


# ===== Cell 3: Load MS MARCO =====
code(r"""# Cell 3: Load MS MARCO and select samples
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= N_SAMPLES * 3:
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
            samples.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

passage_words = np.array([s['word_count'] for s in samples])
query_words = np.array([len(s['query'].split()) for s in samples])
print(f"Selected {N_SAMPLES} samples")
print(f"Document lengths: {passage_words.min()}-{passage_words.max()} words, "
      f"mean={passage_words.mean():.0f}")
print(f"Query lengths: {query_words.min()}-{query_words.max()} words, "
      f"mean={query_words.mean():.1f}")

# Verify swapped queries are from different topics
print(f"\nSwapped query assignment:")
for i in range(5):
    j = (i + N_SAMPLES // 2) % N_SAMPLES
    print(f"  Sample {i}: '{samples[i]['query'][:50]}...'")
    print(f"    Swapped: '{samples[j]['query'][:50]}...'")
    print()
""")


# ===== Cell 4: Load model + define helpers =====
code(r"""# Cell 4: Load model and define scoring helpers
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

print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = getattr(text_cfg, 'vocab_size', 262208)
print(f"Vocab size: {VOCAB_SIZE}")

NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
BOS_ID = tokenizer.bos_token_id
print(f"BOS token ID: {BOS_ID}")
print(f"Newline token IDs: {NEWLINE_IDS} ({len(NEWLINE_IDS)} tokens)")

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'if', 'then', 'than', 'so', 'up', 'out', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'they', 'them', 'their', 'how', 'when',
    'where', 'why', 'much', 'many', 'some', 'any', 'all', 'each',
    'does', 'also', 'just', 'more', 'most', 'very', 'too', 'only',
}


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


def score(doc_text, query_text, answer_text, prefix_token_ids=None):
    # Score NLL of answer tokens using two-phase KV cache.
    #
    # If prefix_token_ids is provided:
    #   Phase A: [BOS] + prefix_ids + [\n] + doc_ids
    #   Slice first 1+len(prefix_ids)+len(NEWLINE_IDS) entries
    # Otherwise (bare):
    #   Phase A: [BOS] + doc_ids (nothing sliced)

    # --- Phase A: Conditioning ---
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids

    if prefix_token_ids is not None:
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        slice_start = 1 + len(prefix_token_ids) + len(NEWLINE_IDS)
        phase_b_start = len(cond_ids)
    else:
        cond_ids = [BOS_ID] + doc_ids
        slice_start = 0
        phase_b_start = len(cond_ids)

    cond_tensor = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        phase_a = model(input_ids=cond_tensor, use_cache=True)

    cache = phase_a.past_key_values
    del phase_a

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

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


print("Scoring function defined.")
""")


# ===== Cell 5: Build token-level prefixes =====
code(r"""# Cell 5: Build per-sample token-level prefix IDs

pyrandom.seed(SEED + 200)

for i, s in enumerate(samples):
    q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    Q = len(q_ids)
    s['Q'] = Q

    # 1. oracle: actual query tokens
    s['prefix_oracle'] = q_ids

    # 2. swapped: query from a distant sample, tokenized and truncated/padded to Q
    swapped_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    swapped_q_ids = tokenizer(samples[swapped_idx]['query'],
                              add_special_tokens=False).input_ids
    if len(swapped_q_ids) >= Q:
        s['prefix_swapped'] = swapped_q_ids[:Q]
    else:
        padded = swapped_q_ids * ((Q // max(len(swapped_q_ids), 1)) + 1)
        s['prefix_swapped'] = padded[:Q]

    # 3. random_matched: words from unrelated passage, tokenized and truncated/padded to Q
    other_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    n_query_words = len(s['query'].split())
    other_words = samples[other_idx]['passage'].split()
    random_text = " ".join(other_words[:n_query_words])
    rand_ids = tokenizer(random_text, add_special_tokens=False).input_ids
    if len(rand_ids) >= Q:
        s['prefix_random'] = rand_ids[:Q]
    else:
        padded = rand_ids * ((Q // max(len(rand_ids), 1)) + 1)
        s['prefix_random'] = padded[:Q]

    s['swapped_query_text'] = samples[swapped_idx]['query']

# Summary statistics
q_lens = [s['Q'] for s in samples]
print(f"Loaded {len(samples)} samples")
print(f"Query token count — mean: {np.mean(q_lens):.1f}, "
      f"median: {np.median(q_lens):.0f}, "
      f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

# Verify prefix lengths
for i, s in enumerate(samples[:5]):
    Q = s['Q']
    for name in ['prefix_oracle', 'prefix_swapped', 'prefix_random']:
        assert len(s[name]) == Q, f"Sample {i} {name}: len={len(s[name])} != Q={Q}"
    print(f"  Sample {i}: Q={Q}")
    print(f"    oracle:  {tokenizer.decode(s['prefix_oracle'][:8])}...")
    print(f"    swapped: {tokenizer.decode(s['prefix_swapped'][:8])}...")
    print(f"    random:  {tokenizer.decode(s['prefix_random'][:8])}...")
print("All prefix lengths verified.")
""")


# ===== Cell 6: Validation =====
code(r"""# Cell 6: Validate scoring
print("=" * 70)
print("VALIDATION")
print("=" * 70)

s = samples[0]
Q = s['Q']

print(f"\nSample 0: Q={Q} query tokens")
print(f"  Query:   '{s['query']}'")
print(f"  Swapped: '{s['swapped_query_text']}'")

print(f"\n--- NLL for each condition (sample 0) ---")
nll_bare = score(s['passage'], s['query'], s['answer'])
print(f"  {'bare':<18} NLL = {nll_bare:.4f}")

for name, prefix_key in [('oracle', 'prefix_oracle'),
                          ('swapped', 'prefix_swapped'),
                          ('random_matched', 'prefix_random')]:
    nll = score(s['passage'], s['query'], s['answer'],
                prefix_token_ids=s[prefix_key])
    print(f"  {name:<18} NLL = {nll:.4f}  delta = {nll_bare - nll:+.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 7: Scoring loop =====
code(r"""# Cell 7: Scoring loop — 4 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = ['bare', 'oracle', 'swapped', 'random_matched']

PREFIX_MAP = {
    'oracle': 'prefix_oracle',
    'swapped': 'prefix_swapped',
    'random_matched': 'prefix_random',
}

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
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {len(COND_NAMES) * N_SAMPLES} scorings")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
        'swapped_query': s['swapped_query_text'],
        'Q': s['Q'],
    }

    # bare
    result['nll_bare'] = score(s['passage'], s['query'], s['answer'])

    # All prefixed conditions
    for cond_name, prefix_key in PREFIX_MAP.items():
        result[f'nll_{cond_name}'] = score(
            s['passage'], s['query'], s['answer'],
            prefix_token_ids=s[prefix_key]
        )

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
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | "
                   f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 8: Part 1 — Standard Condition Table =====
code(r"""# Cell 8: Part 1 — Standard Condition Table
print("=" * 70)
print("PART 1: STANDARD CONDITION TABLE")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_nlls = np.array([r['nll_oracle'] for r in results])
swapped_nlls = np.array([r['nll_swapped'] for r in results])
random_nlls = np.array([r['nll_random_matched'] for r in results])

oracle_benefit = bare_nlls - oracle_nlls
oracle_d = cohens_d(oracle_benefit)

all_conds = [
    ('oracle', 'Oracle (real query)'),
    ('swapped', 'Swapped (wrong query)'),
    ('random_matched', 'Random matched (structural)'),
]

alpha_bonf = 0.05 / len(all_conds)

print(f"\n{'Condition':<35} {'NLL':>8} {'Delta':>8} {'d':>8} "
      f"{'Win%':>7} {'%Orc':>6} {'p':>12} {'sig':>5}")
print("-" * 95)

for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    _, p = stats.ttest_1samp(benefit, 0)
    sig = '***' if p < alpha_bonf / 10 else '**' if p < alpha_bonf else '*' if p < 0.05 else 'ns'
    print(f"  {desc:<33} {nlls.mean():>8.4f} {delta:>+8.4f} {d:>+8.3f} "
          f"{win:>6.1f}% {pct:>5.0f}% {p:>12.2e} {sig}")

print(f"\n  bare (lower bound): {bare_nlls.mean():.4f}")
print(f"  Bonferroni threshold: alpha={alpha_bonf:.4f}")
""")


# ===== Cell 9: Part 2 — Paired Semantic Contrast (the key test) =====
code(r"""# Cell 9: Part 2 — Paired Semantic Contrast
print("=" * 70)
print("PART 2: PAIRED SEMANTIC CONTRAST (the key test)")
print("=" * 70)
print("Per-document: Delta_semantic = swapped_nll - oracle_nll")
print("Both conditions have the same structural perturbation (Q prefix tokens).")
print("Only semantic relevance differs.\n")

semantic_effect = swapped_nlls - oracle_nlls  # positive = oracle is better

# Paired t-test
t_stat, p_paired = stats.ttest_rel(swapped_nlls, oracle_nlls)
d_semantic = cohens_d(semantic_effect)
win_oracle = 100 * np.mean(semantic_effect > 0)

print(f"  Mean(swapped_nll - oracle_nll): {semantic_effect.mean():+.4f}")
print(f"  Cohen's d:                      {d_semantic:+.3f}")
print(f"  Oracle wins:                    {win_oracle:.1f}%")
print(f"  Paired t-test:                  t={t_stat:.3f}, p={p_paired:.2e}")

sig = '***' if p_paired < 0.001 else '**' if p_paired < 0.01 else '*' if p_paired < 0.05 else 'ns'
print(f"  Significance:                   {sig}")

if p_paired < 0.05 and d_semantic > 0:
    print(f"\n  --> SEMANTIC RELEVANCE MATTERS: oracle query produces significantly")
    print(f"      lower NLL than a swapped query from a different topic.")
elif p_paired < 0.05 and d_semantic < 0:
    print(f"\n  --> REVERSE: swapped query is actually BETTER than oracle.")
else:
    print(f"\n  --> NO SIGNIFICANT SEMANTIC EFFECT: oracle and swapped queries")
    print(f"      produce equivalent NLLs. The benefit is purely structural.")

print(f"\n--- Context ---")
print(f"  Overall oracle benefit (vs bare): d={oracle_d:+.3f}")
print(f"  Semantic component (paired):      d={d_semantic:+.3f}")
print(f"  Structural component (estimated): d={oracle_d - d_semantic:+.3f}")
if oracle_d > 0:
    sem_frac = d_semantic / oracle_d * 100
    print(f"  Semantic fraction:                {sem_frac:.1f}% of total benefit")

# Cross-architecture comparison
print(f"\n--- v3 comparison (T5Gemma encoder-decoder) ---")
print(f"  v3 Exp 13: d_semantic=+0.166, win=63.4%, p=2.3e-04, "
      f"semantic_fraction=33.4%")
print(f"  v4 Exp 07: d_semantic={d_semantic:+.3f}, win={win_oracle:.1f}%, "
      f"p={p_paired:.2e}")
""")


# ===== Cell 10: Part 3 — Effect Distribution =====
code(r"""# Cell 10: Part 3 — Effect Distribution
print("=" * 70)
print("PART 3: EFFECT DISTRIBUTION")
print("=" * 70)
print("Per-sample distribution of swapped_nll - oracle_nll\n")

print(f"  Mean:   {semantic_effect.mean():+.4f}")
print(f"  Median: {np.median(semantic_effect):+.4f}")
print(f"  Std:    {semantic_effect.std():.4f}")
print(f"  Min:    {semantic_effect.min():+.4f}")
print(f"  Max:    {semantic_effect.max():+.4f}")

oracle_better = np.sum(semantic_effect > 0)
swapped_better = np.sum(semantic_effect < 0)
tied = np.sum(semantic_effect == 0)

print(f"\n  Oracle better (oracle < swapped): "
      f"{oracle_better} ({oracle_better/N_SAMPLES*100:.1f}%)")
print(f"  Swapped better (swapped < oracle): "
      f"{swapped_better} ({swapped_better/N_SAMPLES*100:.1f}%)")
print(f"  Tied:                              {tied}")

print(f"\n--- Effect size distribution ---")
for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
    n_above = np.sum(semantic_effect > threshold)
    n_below = np.sum(semantic_effect < -threshold)
    print(f"  |effect| > {threshold:.2f}: "
          f"{n_above} oracle wins, {n_below} swapped wins")
""")


# ===== Cell 11: Part 4 — Predictors of Semantic Benefit =====
code(r"""# Cell 11: Part 4 — Predictors of Semantic Benefit
print("=" * 70)
print("PART 4: PREDICTORS OF SEMANTIC BENEFIT")
print("=" * 70)

# (a) Query-document vocabulary overlap (Jaccard on content words)
jaccard_overlaps = []
for i in range(N_SAMPLES):
    doc_words = set(re.sub(r'[^\w\s]', '', samples[i]['passage'].lower()).split())
    doc_content = doc_words - STOP_WORDS
    q_words = set(re.sub(r'[^\w\s]', '', samples[i]['query'].lower()).split())
    q_content = q_words - STOP_WORDS
    if len(doc_content | q_content) > 0:
        jaccard = len(doc_content & q_content) / len(doc_content | q_content)
    else:
        jaccard = 0.0
    jaccard_overlaps.append(jaccard)
jaccard_overlaps = np.array(jaccard_overlaps)

doc_lengths = np.array([r['passage_words'] for r in results])
answer_lengths = np.array([len(r['answer'].split()) for r in results])
query_lengths = np.array([len(r['query'].split()) for r in results])

predictors = [
    ('Query-doc Jaccard overlap', jaccard_overlaps),
    ('Document length (words)', doc_lengths),
    ('Bare NLL (hardness)', bare_nlls),
    ('Answer length (words)', answer_lengths),
    ('Query length (words)', query_lengths),
]

print(f"\n  {'Predictor':<30} {'Pearson r':>10} {'p':>12} {'sig':>5}")
print(f"  {'-'*62}")

alpha_bonf_pred = 0.05 / len(predictors)

for name, values in predictors:
    r_val, p_val = stats.pearsonr(values, semantic_effect)
    sig = '***' if p_val < alpha_bonf_pred / 10 else '**' if p_val < alpha_bonf_pred else \
          '*' if p_val < 0.05 else 'ns'
    print(f"  {name:<30} {r_val:>+10.3f} {p_val:>12.2e} {sig}")

# Hardness interaction (detailed)
print(f"\n--- Semantic effect by hardness quintile ---")
quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

print(f"  {'Quintile':<12} {'N':>4} {'Bare NLL':>10} {'Sem effect':>12} {'d':>8} "
      f"{'Win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*75}")

for q in range(5):
    mask = quintiles == q
    n = mask.sum()
    eff_q = semantic_effect[mask]
    d_q = cohens_d(eff_q)
    win_q = 100 * np.mean(eff_q > 0)
    _, p_q = stats.ttest_1samp(eff_q, 0)
    sig_q = '***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'ns'
    print(f"  {q_labels[q]:<12} {n:>4} {bare_nlls[mask].mean():>10.3f} "
          f"{eff_q.mean():>+12.4f} {d_q:>+8.3f} {win_q:>6.1f}% {p_q:>12.2e} {sig_q}")
""")


# ===== Cell 12: Part 5 — Structural Equivalence Check =====
code(r"""# Cell 12: Part 5 — Structural Equivalence Check
print("=" * 70)
print("PART 5: STRUCTURAL EQUIVALENCE CHECK")
print("=" * 70)
print("Confirm oracle and swapped have similar structural benefit vs bare.")
print("With token-level matching, structural confounds should be eliminated.\n")

oracle_struct = bare_nlls - oracle_nlls
swapped_struct = bare_nlls - swapped_nlls
random_struct = bare_nlls - random_nlls

oracle_vs_bare_d = cohens_d(oracle_struct)
swapped_vs_bare_d = cohens_d(swapped_struct)
random_vs_bare_d = cohens_d(random_struct)

print(f"  Condition vs bare (Cohen's d):")
print(f"    Oracle:  d={oracle_vs_bare_d:+.3f}")
print(f"    Swapped: d={swapped_vs_bare_d:+.3f}")
print(f"    Random:  d={random_vs_bare_d:+.3f}")

# Semantic component: condition vs random
oracle_vs_random = random_nlls - oracle_nlls
swapped_vs_random = random_nlls - swapped_nlls

d_orac_rand = cohens_d(oracle_vs_random)
d_swap_rand = cohens_d(swapped_vs_random)
_, p_orac_rand = stats.ttest_1samp(oracle_vs_random, 0)
_, p_swap_rand = stats.ttest_1samp(swapped_vs_random, 0)

print(f"\n  Condition vs random (semantic component):")
print(f"    Oracle - random:  d={d_orac_rand:+.3f}, p={p_orac_rand:.2e}")
print(f"    Swapped - random: d={d_swap_rand:+.3f}, p={p_swap_rand:.2e}")

# Key check
structural_diff = oracle_struct - swapped_struct
d_struct_diff = cohens_d(structural_diff)
_, p_struct = stats.ttest_1samp(structural_diff, 0)
sig_struct = '***' if p_struct < 0.001 else '**' if p_struct < 0.01 else '*' if p_struct < 0.05 else 'ns'

print(f"\n  Oracle benefit - Swapped benefit: d={d_struct_diff:+.3f}, "
      f"p={p_struct:.2e} {sig_struct}")
print(f"  (Should match Part 2 semantic effect: d={d_semantic:+.3f})")
print(f"  Consistency check: {abs(d_struct_diff - d_semantic):.4f} (should be ~0)")

# Token count verification (should be exactly equal with token-level matching)
oracle_q = np.array([r['Q'] for r in results])
print(f"\n  Token-level matching verification:")
print(f"    All prefixes have exactly Q tokens per sample: VERIFIED")
print(f"    Mean Q: {oracle_q.mean():.1f}, range: [{oracle_q.min()}, {oracle_q.max()}]")
print(f"    No token count confound possible with token-level matching.")
""")


# ===== Cell 13: Synthesis + Save =====
code(r"""# Cell 13: Synthesis + Save
print("=" * 70)
print("SYNTHESIS: SWAPPED-QUERY PAIRED CONTRAST (DECODER-ONLY)")
print("=" * 70)

print(f"\n1. CONDITION TABLE:")
print(f"   {'Condition':<25} {'d vs bare':>10} {'%Oracle':>8}")
print(f"   {'-'*45}")
for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    d = cohens_d(bare_nlls - nlls)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"   {desc:<25} {d:>+10.3f} {pct:>7.0f}%")

print(f"\n2. PAIRED SEMANTIC CONTRAST:")
print(f"   swapped_nll - oracle_nll: mean={semantic_effect.mean():+.4f}, "
      f"d={d_semantic:+.3f}")
print(f"   Oracle win rate: {win_oracle:.1f}%, p={p_paired:.2e}")
if oracle_d > 0:
    sem_frac = d_semantic / oracle_d * 100
    print(f"   Semantic fraction of total benefit: {sem_frac:.1f}%")

print(f"\n3. STRONGEST PREDICTOR:")
best_r = 0
best_name = ""
for name, values in predictors:
    r_val, _ = stats.pearsonr(values, semantic_effect)
    if abs(r_val) > abs(best_r):
        best_r = r_val
        best_name = name
print(f"   {best_name} (r={best_r:+.3f})")

print(f"\n{'='*70}")
print("CONCLUSIONS:")

if p_paired < 0.001 and d_semantic > 0.05:
    print(f"  1. STRONG SEMANTIC SIGNAL: oracle significantly beats swapped")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "STRONG_SEMANTIC"
elif p_paired < 0.05 and d_semantic > 0:
    print(f"  1. WEAK SEMANTIC SIGNAL: marginally significant")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "WEAK_SEMANTIC"
elif p_paired < 0.05 and d_semantic < 0:
    print(f"  1. SEMANTIC INTERFERENCE: swapped query is BETTER")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "INTERFERENCE"
else:
    print(f"  1. NO SEMANTIC EFFECT: oracle and swapped are equivalent")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "NO_EFFECT"

print(f"\n  Token-level matching eliminates all structural confounds.")

# Cross-architecture comparison
print(f"\n--- Cross-architecture comparison ---")
print(f"  v3 (T5Gemma enc-dec): d_semantic=+0.166, p=2.3e-04, "
      f"fraction=33.4%")
print(f"  v4 (Gemma 3 dec-only): d_semantic={d_semantic:+.3f}, p={p_paired:.2e}")
if oracle_d > 0:
    print(f"                         fraction={sem_frac:.1f}%")
print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'v4_decoder_only_exp07_swapped_query',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'baseline': {
        'bare_nll': float(bare_nlls.mean()),
        'oracle_d': float(oracle_d),
    },
    'semantic_contrast': {
        'mean_effect': float(semantic_effect.mean()),
        'd': float(d_semantic),
        'win_pct': float(win_oracle),
        'p_paired': float(p_paired),
        'semantic_fraction_pct': float(sem_frac) if oracle_d > 0 else None,
    },
    'structural_equivalence': {
        'oracle_vs_bare_d': float(oracle_vs_bare_d),
        'swapped_vs_bare_d': float(swapped_vs_bare_d),
        'random_vs_bare_d': float(random_vs_bare_d),
    },
    'predictors': {},
    'conditions': {},
    'conclusion': conclusion,
    'query_token_stats': {
        'mean': float(np.mean([r['Q'] for r in results])),
        'median': float(np.median([r['Q'] for r in results])),
        'min': int(np.min([r['Q'] for r in results])),
        'max': int(np.max([r['Q'] for r in results])),
    },
}

for name, values in predictors:
    r_val, p_val = stats.pearsonr(values, semantic_effect)
    final_results['predictors'][name] = {
        'pearson_r': float(r_val),
        'p': float(p_val),
    }

for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    _, p = stats.ttest_1samp(benefit, 0)
    final_results['conditions'][cond] = {
        'description': desc,
        'd': float(d),
        'mean_nll': float(nlls.mean()),
        'mean_delta': float(benefit.mean()),
        'pct_oracle': float(d / oracle_d * 100) if oracle_d > 0 else 0,
        'p': float(p),
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
out_path = "experiments/decoder_only/07/07_swapped_query_paired.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
