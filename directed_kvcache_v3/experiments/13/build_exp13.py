#!/usr/bin/env python3
# Build Exp 13 notebook: Swapped-Query Paired Contrasts.
#
# Maximum-power paired test for semantic relevance: same doc scored with
# real query vs swapped query from a different sample. No LLM generation needed.
#
# 4 conditions, 500 samples = 2,000 scoring passes. ~20 min.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 13: Swapped-Query Paired Contrasts

## Motivation

Exp 2B showed ~85% of the encoder priming benefit is structural, ~10% semantic.
But measuring the semantic component precisely is confounded by vocabulary and
length differences between conditions. This experiment provides the **purest
possible test** of semantic relevance by comparing oracle vs swapped query on the
**same document** — same structural perturbation, same prefix format, only semantic
relevance differs.

## Design

For each (query, document, answer) triple, we score the answer NLL under two
prefix conditions:
- `oracle_trunc`: the real query (semantically relevant)
- `swapped_trunc`: a query from a completely different sample (semantically irrelevant)

The per-document paired contrast `swapped_nll - oracle_nll` isolates the semantic
component with maximum statistical power.

## Conditions (4)

| # | Condition | Prefix |
|---|-----------|--------|
| 1 | `bare` | (none) |
| 2 | `oracle_trunc` | real query |
| 3 | `swapped_trunc` | query from sample (i + N//2) % N |
| 4 | `random_matched_trunc` | words from random passage |

## Analysis

1. Standard condition table
2. Paired semantic contrast (the key test)
3. Effect distribution — per-sample histogram
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

sys.path.insert(0, "../..")
from lib.analysis import cohens_d

SEED = 43  # Different seed from Exp 12 (42) for independent samples
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../results/exp13")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

print("Exp 13: Swapped-Query Paired Contrasts")
print(f"N: {N_SAMPLES}, SEED: {SEED}")
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

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)
    if ans_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            labels=ans_ids,
        )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    # Count how many tokens the prefix occupies in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


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

print("Helpers defined.")
""")


# ===== Cell 5: Generate conditions per sample =====
code(r"""# Cell 5: Generate all 4 scoring conditions per sample

for i, s in enumerate(samples):
    query = s['query']
    passage = s['passage']
    query_words_list = query.split()
    n_query_words = len(query_words_list)

    # Swapped query: query from a distant sample (guaranteed different topic)
    swapped_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    s['swapped'] = samples[swapped_idx]['query']

    # Random matched: words from unrelated passage, same word count as oracle
    other_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    other_words = samples[other_idx]['passage'].split()
    s['random_matched'] = " ".join(other_words[:n_query_words])

    # Oracle (just the query)
    s['oracle'] = query

COND_NAMES = [
    'bare',
    'oracle_trunc',
    'swapped_trunc',
    'random_matched_trunc',
]

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query:   {ex['query']}")
print(f"  Answer:  {ex['answer'][:80]}")
print(f"  Passage: {ex['passage'][:80]}...")
for c in COND_NAMES:
    if c == 'bare':
        print(f"  {c:<28}: [document only]")
    else:
        key = c.replace('_trunc', '')
        text = ex[key]
        ptoks = count_prefix_tokens(text, ex['passage'])
        print(f"  {c:<28} ({ptoks:>3} toks): {str(text)[:55]}")
""")


# ===== Cell 6: Scoring loop with checkpointing =====
code(r"""# Cell 6: Scoring loop with checkpointing

print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

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
        'swapped_query': s['swapped'],
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['passage'], s['answer'])
            result['nll_bare'] = nll
        else:
            key = cond.replace('_trunc', '')
            prefix = s[key]
            enc_text = prefix + "\n" + s['passage']
            ptoks = count_prefix_tokens(prefix, s['passage'])
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result[f'nll_{cond}'] = nll
            result[f'ptoks_{cond}'] = ptoks

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


# ===== Cell 7: Part 1 — Standard Condition Table =====
code(r"""# Cell 7: Part 1 — Standard Condition Table

print("=" * 70)
print("PART 1: STANDARD CONDITION TABLE")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_nlls = np.array([r['nll_oracle_trunc'] for r in results])
swapped_nlls = np.array([r['nll_swapped_trunc'] for r in results])
random_nlls = np.array([r['nll_random_matched_trunc'] for r in results])

oracle_benefit = bare_nlls - oracle_nlls
oracle_d = cohens_d(oracle_benefit)

all_conds = [
    ('oracle_trunc', 'Oracle (real query)'),
    ('swapped_trunc', 'Swapped (wrong query)'),
    ('random_matched_trunc', 'Random matched (structural)'),
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


# ===== Cell 8: Part 2 — Paired Semantic Contrast (the key test) =====
code(r"""# Cell 8: Part 2 — Paired Semantic Contrast

print("=" * 70)
print("PART 2: PAIRED SEMANTIC CONTRAST (the key test)")
print("=" * 70)
print("Per-document: Delta_semantic = swapped_nll - oracle_nll")
print("Both conditions have the same structural perturbation (a real query prefix).")
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
    print(f"      The semantic component is d={d_semantic:+.3f} in paired comparison.")
elif p_paired < 0.05 and d_semantic < 0:
    print(f"\n  --> REVERSE: swapped query is actually BETTER than oracle.")
    print(f"      This would suggest semantic interference from the real query.")
else:
    print(f"\n  --> NO SIGNIFICANT SEMANTIC EFFECT: oracle and swapped queries")
    print(f"      produce equivalent NLLs. The benefit is purely structural.")

# Context: how does this compare to overall benefit?
print(f"\n--- Context ---")
print(f"  Overall oracle benefit (vs bare): d={oracle_d:+.3f}")
print(f"  Semantic component (paired):      d={d_semantic:+.3f}")
print(f"  Structural component (estimated): d={oracle_d - d_semantic:+.3f}")
if oracle_d > 0:
    sem_frac = d_semantic / oracle_d * 100
    print(f"  Semantic fraction:                {sem_frac:.1f}% of total benefit")
""")


# ===== Cell 9: Part 3 — Effect Distribution =====
code(r"""# Cell 9: Part 3 — Effect Distribution

print("=" * 70)
print("PART 3: EFFECT DISTRIBUTION")
print("=" * 70)
print("Per-sample distribution of swapped_nll - oracle_nll\n")

# Distribution statistics
print(f"  Mean:   {semantic_effect.mean():+.4f}")
print(f"  Median: {np.median(semantic_effect):+.4f}")
print(f"  Std:    {semantic_effect.std():.4f}")
print(f"  Min:    {semantic_effect.min():+.4f}")
print(f"  Max:    {semantic_effect.max():+.4f}")

# Fraction showing semantic benefit
oracle_better = np.sum(semantic_effect > 0)
swapped_better = np.sum(semantic_effect < 0)
tied = np.sum(semantic_effect == 0)

print(f"\n  Oracle better (oracle < swapped): {oracle_better} ({oracle_better/N_SAMPLES*100:.1f}%)")
print(f"  Swapped better (swapped < oracle): {swapped_better} ({swapped_better/N_SAMPLES*100:.1f}%)")
print(f"  Tied:                              {tied}")

# Effect size distribution
print(f"\n--- Effect size distribution ---")
for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
    n_above = np.sum(semantic_effect > threshold)
    n_below = np.sum(semantic_effect < -threshold)
    print(f"  |effect| > {threshold:.2f}: {n_above} oracle wins, {n_below} swapped wins")

# Quintile breakdown of per-sample semantic effect
print(f"\n--- Per-sample semantic effect by quintile ---")
eff_quintile_bounds = np.percentile(semantic_effect, [20, 40, 60, 80])
eff_quintiles = np.digitize(semantic_effect, eff_quintile_bounds)
for q in range(5):
    mask = eff_quintiles == q
    eff_q = semantic_effect[mask]
    print(f"  Q{q+1}: mean={eff_q.mean():+.4f}, range=[{eff_q.min():+.4f}, {eff_q.max():+.4f}], "
          f"N={mask.sum()}")
""")


# ===== Cell 10: Part 4 — Predictors of Semantic Benefit =====
code(r"""# Cell 10: Part 4 — Predictors of Semantic Benefit

print("=" * 70)
print("PART 4: PREDICTORS OF SEMANTIC BENEFIT")
print("=" * 70)
print("What sample characteristics predict whether oracle > swapped?\n")

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

# (b) Document length (word count)
doc_lengths = np.array([r['passage_words'] for r in results])

# (c) Bare NLL (hardness)
# bare_nlls already defined

# (d) Answer length
answer_lengths = np.array([len(r['answer'].split()) for r in results])

# (e) Query length
query_lengths = np.array([len(r['query'].split()) for r in results])

predictors = [
    ('Query-doc Jaccard overlap', jaccard_overlaps),
    ('Document length (words)', doc_lengths),
    ('Bare NLL (hardness)', bare_nlls),
    ('Answer length (words)', answer_lengths),
    ('Query length (words)', query_lengths),
]

print(f"  {'Predictor':<30} {'Pearson r':>10} {'p':>12} {'sig':>5}")
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

# Jaccard interaction (detailed)
print(f"\n--- Semantic effect by query-doc overlap quintile ---")
jacc_bounds = np.percentile(jaccard_overlaps, [20, 40, 60, 80])
jacc_quints = np.digitize(jaccard_overlaps, jacc_bounds)

for q in range(5):
    mask = jacc_quints == q
    n = mask.sum()
    eff_q = semantic_effect[mask]
    d_q = cohens_d(eff_q)
    jacc_q = jaccard_overlaps[mask].mean()
    _, p_q = stats.ttest_1samp(eff_q, 0)
    sig_q = '***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'ns'
    print(f"  Q{q+1} (Jaccard={jacc_q:.3f}, N={n}): effect={eff_q.mean():+.4f}, "
          f"d={d_q:+.3f}, p={p_q:.2e} {sig_q}")
""")


# ===== Cell 11: Part 5 — Structural Equivalence Check =====
code(r"""# Cell 11: Part 5 — Structural Equivalence Check

print("=" * 70)
print("PART 5: STRUCTURAL EQUIVALENCE CHECK")
print("=" * 70)
print("Confirm oracle and swapped have similar structural benefit vs bare.")
print("If structural effects differ, the 'semantic' contrast is confounded.\n")

# Structural benefit: condition vs bare
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

# Both should have similar structural benefit relative to random
oracle_vs_random = random_nlls - oracle_nlls  # semantic component
swapped_vs_random = random_nlls - swapped_nlls  # should be ~0 or small

d_orac_rand = cohens_d(oracle_vs_random)
d_swap_rand = cohens_d(swapped_vs_random)
_, p_orac_rand = stats.ttest_1samp(oracle_vs_random, 0)
_, p_swap_rand = stats.ttest_1samp(swapped_vs_random, 0)

print(f"\n  Condition vs random (semantic component):")
print(f"    Oracle - random:  d={d_orac_rand:+.3f}, p={p_orac_rand:.2e}")
print(f"    Swapped - random: d={d_swap_rand:+.3f}, p={p_swap_rand:.2e}")

# The key check: oracle and swapped should have similar benefit over random
# IF the semantic component is real, oracle should beat random more than swapped
structural_diff = oracle_struct - swapped_struct
d_struct_diff = cohens_d(structural_diff)
_, p_struct = stats.ttest_1samp(structural_diff, 0)
sig_struct = '***' if p_struct < 0.001 else '**' if p_struct < 0.01 else '*' if p_struct < 0.05 else 'ns'

print(f"\n  Oracle benefit - Swapped benefit: d={d_struct_diff:+.3f}, p={p_struct:.2e} {sig_struct}")
print(f"  (This should equal the semantic effect from Part 2: d={d_semantic:+.3f})")
print(f"  Consistency check: {abs(d_struct_diff - d_semantic):.4f} (should be ~0)")

# Prefix token count comparison
oracle_ptoks = np.array([r['ptoks_oracle_trunc'] for r in results])
swapped_ptoks = np.array([r['ptoks_swapped_trunc'] for r in results])
random_ptoks = np.array([r['ptoks_random_matched_trunc'] for r in results])

print(f"\n  Prefix token counts:")
print(f"    Oracle:  mean={oracle_ptoks.mean():.1f}, std={oracle_ptoks.std():.1f}")
print(f"    Swapped: mean={swapped_ptoks.mean():.1f}, std={swapped_ptoks.std():.1f}")
print(f"    Random:  mean={random_ptoks.mean():.1f}, std={random_ptoks.std():.1f}")

# Are token counts correlated with semantic effect? (potential confound)
ptok_diff = swapped_ptoks - oracle_ptoks
r_ptok, p_ptok = stats.pearsonr(ptok_diff.astype(float), semantic_effect)
sig_ptok = '***' if p_ptok < 0.001 else '**' if p_ptok < 0.01 else '*' if p_ptok < 0.05 else 'ns'
print(f"\n  Token count diff vs semantic effect: r={r_ptok:+.3f}, p={p_ptok:.2e} {sig_ptok}")

if abs(r_ptok) > 0.1 and p_ptok < 0.05:
    print(f"  WARNING: token count difference correlates with semantic effect.")
    print(f"  The 'semantic' contrast may be partially confounded by prefix length.")
else:
    print(f"  CLEAN: no confound from prefix length differences.")
""")


# ===== Cell 12: Synthesis + Save =====
code(r"""# Cell 12: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: SWAPPED-QUERY PAIRED CONTRAST RESULTS")
print("=" * 70)

# 1. Summary
print(f"\n1. CONDITION TABLE:")
print(f"   {'Condition':<25} {'d vs bare':>10} {'%Oracle':>8}")
print(f"   {'-'*45}")
for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    d = cohens_d(bare_nlls - nlls)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"   {desc:<25} {d:>+10.3f} {pct:>7.0f}%")

# 2. The key result
print(f"\n2. PAIRED SEMANTIC CONTRAST:")
print(f"   swapped_nll - oracle_nll: mean={semantic_effect.mean():+.4f}, d={d_semantic:+.3f}")
print(f"   Oracle win rate: {win_oracle:.1f}%, p={p_paired:.2e}")
if oracle_d > 0:
    sem_frac = d_semantic / oracle_d * 100
    print(f"   Semantic fraction of total benefit: {sem_frac:.1f}%")

# 3. Strongest predictor
print(f"\n3. PREDICTORS OF SEMANTIC BENEFIT:")
best_r = 0
best_name = ""
for name, values in predictors:
    r_val, _ = stats.pearsonr(values, semantic_effect)
    if abs(r_val) > abs(best_r):
        best_r = r_val
        best_name = name
print(f"   Strongest: {best_name} (r={best_r:+.3f})")

# 4. Structural equivalence
print(f"\n4. STRUCTURAL EQUIVALENCE:")
print(f"   Oracle vs bare:  d={oracle_vs_bare_d:+.3f}")
print(f"   Swapped vs bare: d={swapped_vs_bare_d:+.3f}")
print(f"   Token count confound: r={r_ptok:+.3f} ({'CLEAN' if abs(r_ptok) < 0.1 else 'WARNING'})")

# 5. Conclusions
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
    print(f"  1. SEMANTIC INTERFERENCE: swapped query is actually BETTER")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "INTERFERENCE"
else:
    print(f"  1. NO SEMANTIC EFFECT: oracle and swapped are equivalent")
    print(f"     (d={d_semantic:+.3f}, p={p_paired:.2e})")
    conclusion = "NO_EFFECT"

if abs(r_ptok) < 0.1:
    print(f"  2. Result is CLEAN: no confound from prefix length")
else:
    print(f"  2. Result is CONFOUNDED: token count correlates with effect (r={r_ptok:+.3f})")

# Cross-reference with Exp 12
print(f"\n  Cross-reference with Exp 12:")
print(f"  Exp 12 tests the GRADIENT; this experiment confirms/denies the binary signal.")
print(f"  If d_semantic > 0 here, the gradient in Exp 12 should be monotonic.")
print(f"  If d_semantic ~ 0 here, any gradient in Exp 12 is likely noise.")
print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp13_swapped_query',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
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
        'token_count_confound_r': float(r_ptok),
        'token_count_confound_p': float(p_ptok),
    },
    'predictors': {},
    'conditions': {},
    'conclusion': conclusion,
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
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/13/13_swapped_query_paired.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
