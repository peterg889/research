#!/usr/bin/env python3
"""Build Exp 3D notebook: Cross-Dataset Content Ablation (Long Queries).

Tests whether the 85/6/10 structure/vocabulary/semantics split from Exp 2B
(MS MARCO, 6-word queries) holds on a dataset with longer queries (~18 words).

Dataset: neural-bridge/rag-dataset-12000 (filtered to q>=15 words, a>=5 words).
  - 3,384 eligible samples
  - Mean query: 17.7 words (vs 6.0 for MS MARCO)
  - Mean document: 603 words
  - Mean answer: 42.7 words

Pre-screen results:
  - Ceiling check: PASS (mean bare NLL=0.98, 0% at floor)
  - Oracle headroom: STRONG (+0.255 nats mean, all 5 positive)

Design: Same ablation framework as Exp 2B, adapted for new dataset.
  Part 1: Baseline characterization (bare, oracle, headroom distribution)
  Part 2: Content ablation (structure/vocabulary/semantics decomposition)
  Part 3: Token diversity (repeat_the, repeat_kw, diverse random)
  Part 4: Query length stratification (within-dataset, exploit the spread)
  Part 5: Comparison with Exp 2B (MS MARCO) results
"""

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 3D: Cross-Dataset Content Ablation (Long Queries)

## Motivation

Exp 2B on MS MARCO found that **85% of the oracle headroom is pure structure** —
any prefix triggers a mode shift in the encoder. But MS MARCO queries are only
~6 words long. For such short queries:
- **Scrambled oracle** preserves bag-of-words, which IS most of the semantics
- **Vocabulary test** (random → scrambled) is weak because 6 random words cover
  a lot of vocabulary space relative to a 6-word query
- **Word order** barely matters in 6 words

This experiment tests the same decomposition on a dataset with **3x longer queries**
(~18 words) and **5x longer documents** (~600 words). If the split changes, the
MS MARCO finding was an artifact of short queries. If it holds, the structural
mechanism is genuine.

## Three-Point Framework

| Label | Encoder input | Cost | Role |
|-------|--------------|------|------|
| **Upper bound** | [real_query + document], mask query from decoder | O(Q×D) | Ideal but too expensive |
| **Lower bound** | [document] only | O(1) | Current worst case |
| **Middle ground** | [surrogate + document], mask surrogate from decoder | O(1) | Our hypothesis |

## Dataset: neural-bridge/rag-dataset-12000

- Synthetic QA pairs with retrieved context passages
- Filtered to: query ≥ 15 words, answer ≥ 5 words → ~3,384 samples
- Mean query: 17.7 words (3x MS MARCO), document: 603 words, answer: 42.7 words
- Ceiling pre-screen: PASS (mean bare NLL = 0.98, strong oracle headroom +0.255)

## Design

**Conditions** (all with truncation):
1. `bare` — document only (lower bound)
2. `oracle_trunc` — real query + document (upper bound)
3. `random_matched_trunc` — N random words (N = query word count)
4. `scrambled_oracle_trunc` — query words in random order
5. `surr_template_trunc` — "What is [top_keyword]?"
6. `repeat_the_trunc` — "the" repeated N times (N = query word count)
7. `repeat_kw_trunc` — top doc keyword repeated N times

**Content ablation decomposition**:
- Structure = bare → random_matched
- Vocabulary = random_matched → scrambled_oracle
- Semantics = scrambled_oracle → oracle

**Key prediction**: With 18-word queries, scrambling should lose more information
than with 6-word queries. We expect the semantic component to grow from 10% to
potentially 20-30%+.""")


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

sys.path.insert(0, ".")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("results/exp03d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

print("Exp 3D: Cross-Dataset Content Ablation (Long Queries)")
print(f"Dataset: neural-bridge/rag-dataset-12000")
print(f"N: {N_SAMPLES}")
""")


# ===== Cell 3: Load and prepare dataset =====
code(r"""# Cell 3: Load neural-bridge/rag-dataset-12000 and prepare samples
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
print(f"Total samples: {len(ds)}")

# Filter to long queries with real answers
all_candidates = []
for row in ds:
    q = row.get("question", "")
    doc = row.get("context", "")
    answer = row.get("answer", "")
    if not q or not doc or not answer:
        continue
    q_words = len(q.split())
    a_words = len(answer.split())
    if q_words >= 15 and a_words >= 5:
        all_candidates.append({
            "query": q,
            "document": doc,
            "answer": answer,
            "query_words": q_words,
            "doc_words": len(doc.split()),
            "answer_words": a_words,
        })

print(f"Candidates (q>=15w, a>=5w): {len(all_candidates)}")

# Shuffle and take N_SAMPLES
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]

# Dataset statistics
q_lens = np.array([s["query_words"] for s in samples])
d_lens = np.array([s["doc_words"] for s in samples])
a_lens = np.array([s["answer_words"] for s in samples])

print(f"\nSample statistics (N={N_SAMPLES}):")
print(f"  Query length:  mean={q_lens.mean():.1f}, median={np.median(q_lens):.0f}, "
      f"range=[{q_lens.min()}, {q_lens.max()}]")
print(f"  Doc length:    mean={d_lens.mean():.1f}, median={np.median(d_lens):.0f}, "
      f"range=[{d_lens.min()}, {d_lens.max()}]")
print(f"  Answer length: mean={a_lens.mean():.1f}, median={np.median(a_lens):.0f}, "
      f"range=[{a_lens.min()}, {a_lens.max()}]")

# Show 3 examples
for i in range(3):
    s = samples[i]
    print(f"\nExample {i+1}:")
    print(f"  Q ({s['query_words']}w): {s['query'][:120]}...")
    print(f"  D ({s['doc_words']}w): {s['document'][:100]}...")
    print(f"  A ({s['answer_words']}w): {s['answer'][:100]}...")

# Compare with MS MARCO distribution
print(f"\n--- Comparison with MS MARCO ---")
print(f"  MS MARCO: mean query = 6.0w, mean doc = ~60w, mean answer = ~20w")
print(f"  This dataset: mean query = {q_lens.mean():.1f}w, "
      f"mean doc = {d_lens.mean():.0f}w, mean answer = {a_lens.mean():.0f}w")
print(f"  Query ratio: {q_lens.mean()/6.0:.1f}x longer")
print(f"  Doc ratio: {d_lens.mean()/60:.1f}x longer")

del ds
gc.collect()
""")


# ===== Cell 4: Load model + define helpers =====
code(r"""# Cell 4: Load model and define scoring helpers
from dotenv import load_dotenv
load_dotenv()
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


# ===== Cell 5: Generate conditions for each sample =====
code(r"""# Cell 5: Generate surrogate conditions for each sample

# Build a pool of "other" words from unrelated documents for random conditions
other_words_pool = []
for i, s in enumerate(samples):
    other_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    other_doc = samples[other_idx]['document']
    other_words_pool.append(other_doc.split())

for i, s in enumerate(samples):
    query_words = s['query'].split()
    n_query_words = len(query_words)
    other_words = other_words_pool[i]

    # --- Content ablation (length-matched to oracle) ---

    # scrambled_oracle: same words as query, random order
    rng = np.random.RandomState(SEED + i)
    shuffled = list(query_words)
    rng.shuffle(shuffled)
    s['scrambled_oracle'] = " ".join(shuffled)

    # random_matched: N random words from unrelated doc (same count as query)
    if len(other_words) >= n_query_words:
        s['random_matched'] = " ".join(other_words[:n_query_words])
    else:
        # Pad with repeated words if other doc is too short
        padded = other_words * ((n_query_words // len(other_words)) + 1)
        s['random_matched'] = " ".join(padded[:n_query_words])

    # --- Token diversity (length-matched to oracle word count) ---

    # repeat_the: "the" repeated N times
    s['repeat_the'] = " ".join(["the"] * n_query_words)

    # repeat_kw: top document keyword repeated N times
    doc_words_clean = re.sub(r'[^\w\s]', '', s['document'].lower()).split()
    content = [w for w in doc_words_clean if w not in STOP_WORDS and len(w) > 2]
    if content:
        counts = Counter(content)
        top_word = counts.most_common(1)[0][0]
    else:
        top_word = "information"
    s['repeat_kw'] = " ".join([top_word] * n_query_words)

    # --- Practical surrogate ---

    # surr_template: "What is [top_keyword]?"
    if content:
        kw = counts.most_common(1)[0][0]
    else:
        kw = "information"
    s['surr_template'] = f"What is {kw}?"

# Define all conditions to score
COND_NAMES = [
    'bare',
    'oracle_trunc',
    'random_matched_trunc',
    'scrambled_oracle_trunc',
    'surr_template_trunc',
    'repeat_the_trunc',
    'repeat_kw_trunc',
]

print(f"Conditions to score: {len(COND_NAMES)}")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query ({ex['query_words']}w): {ex['query'][:120]}")
print(f"  Document ({ex['doc_words']}w): {ex['document'][:80]}...")
print(f"  Answer ({ex['answer_words']}w): {ex['answer'][:80]}...")
print()

for c in COND_NAMES:
    if c == 'bare':
        print(f"  {c:<28}: [document only]")
    elif c == 'oracle_trunc':
        ptoks = count_prefix_tokens(ex['query'], ex['document'])
        print(f"  {c:<28} ({ptoks:>3} prefix toks): {ex['query'][:60]}...")
    else:
        key = c.replace('_trunc', '')
        text = ex[key]
        ptoks = count_prefix_tokens(text, ex['document'])
        print(f"  {c:<28} ({ptoks:>3} prefix toks): {str(text)[:60]}")

# Report prefix token stats across first 50 samples
print(f"\nPrefix token counts (first 50 samples):")
for c in COND_NAMES:
    if c == 'bare':
        continue
    if c == 'oracle_trunc':
        toks = [count_prefix_tokens(s['query'], s['document']) for s in samples[:50]]
    else:
        key = c.replace('_trunc', '')
        toks = [count_prefix_tokens(s[key], s['document']) for s in samples[:50]]
    print(f"  {c:<28} mean={np.mean(toks):.1f}, range=[{min(toks)}, {max(toks)}]")
""")


# ===== Cell 6: Run scoring =====
code(r"""# Cell 6: Run scoring with checkpointing

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
        'query_words': s['query_words'],
        'doc_words': s['doc_words'],
        'answer_words': s['answer_words'],
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['document'], s['answer'])
            result['nll_bare'] = nll
        elif cond == 'oracle_trunc':
            enc_text = s['query'] + "\n" + s['document']
            ptoks = count_prefix_tokens(s['query'], s['document'])
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result['nll_oracle_trunc'] = nll
            result['ptoks_oracle'] = ptoks
        else:
            key = cond.replace('_trunc', '')
            surr_text = s[key]
            enc_text = surr_text + "\n" + s['document']
            ptoks = count_prefix_tokens(surr_text, s['document'])
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


# ===== Cell 7: Part 1 — Baseline characterization =====
code(r"""# Cell 7: Part 1 — Baseline Characterization

print("=" * 70)
print("PART 1: BASELINE CHARACTERIZATION")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_nlls = np.array([r['nll_oracle_trunc'] for r in results])
oracle_benefit = bare_nlls - oracle_nlls

print(f"\nBaseline NLL (document only):  mean={bare_nlls.mean():.4f}, std={bare_nlls.std():.4f}")
print(f"Oracle NLL (query + document): mean={oracle_nlls.mean():.4f}, std={oracle_nlls.std():.4f}")
print(f"Oracle headroom: delta={oracle_benefit.mean():+.4f}, d={cohens_d(oracle_benefit):+.3f}")

# Win rate and significance
win_rate = np.mean(oracle_benefit > 0) * 100
_, p_oracle = stats.ttest_1samp(oracle_benefit, 0)
print(f"Oracle win rate: {win_rate:.1f}% (p={p_oracle:.2e})")

if oracle_benefit.mean() < 0.05:
    print("\nWARNING: Oracle headroom is very small. Results may not be meaningful.")
elif p_oracle > 0.05:
    print("\nWARNING: Oracle headroom is not statistically significant.")
else:
    print(f"\nGood: Oracle headroom is significant (d={cohens_d(oracle_benefit):+.3f}, "
          f"p={p_oracle:.2e})")

# Headroom distribution
pcts = np.percentile(oracle_benefit, [10, 25, 50, 75, 90])
print(f"\nOracle benefit distribution:")
print(f"  10th pctile: {pcts[0]:+.4f}")
print(f"  25th pctile: {pcts[1]:+.4f}")
print(f"  Median:      {pcts[2]:+.4f}")
print(f"  75th pctile: {pcts[3]:+.4f}")
print(f"  90th pctile: {pcts[4]:+.4f}")
print(f"  % negative:  {np.mean(oracle_benefit < 0)*100:.1f}%")

# Comparison with MS MARCO
print(f"\n--- Comparison with MS MARCO Exp 02 ---")
print(f"  MS MARCO: bare NLL=3.68, oracle NLL=2.99, headroom=+0.68, d=+0.376")
print(f"  This:     bare NLL={bare_nlls.mean():.2f}, oracle NLL={oracle_nlls.mean():.2f}, "
      f"headroom={oracle_benefit.mean():+.3f}, d={cohens_d(oracle_benefit):+.3f}")
""")


# ===== Cell 8: Part 2 — Content Ablation =====
code(r"""# Cell 8: Part 2 — Content Ablation Decomposition

print("=" * 70)
print("PART 2: CONTENT ABLATION")
print("=" * 70)
print("Decompose: bare -> random_matched -> scrambled_oracle -> oracle")
print("  Structure:  bare -> random_matched (any prefix helps)")
print("  Vocabulary: random_matched -> scrambled (right words, wrong order)")
print("  Semantics:  scrambled -> oracle (right word order, full meaning)\n")

randmatch_nlls = np.array([r['nll_random_matched_trunc'] for r in results])
scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in results])

# Component benefits
struct_comp = bare_nlls - randmatch_nlls
vocab_comp = randmatch_nlls - scrambled_nlls
sem_comp = scrambled_nlls - oracle_nlls
total_comp = bare_nlls - oracle_nlls

total_mean = total_comp.mean()

# Table: condition NLLs and incremental gains
print(f"{'Step':<32} {'Mean NLL':>10} {'Delta':>10} {'% total':>9} "
      f"{'d':>8} {'p':>12} {'sig':>5}")
print("-" * 90)

print(f"  {'bare (baseline)':<30} {bare_nlls.mean():>10.4f}")

for name, nlls, component, label in [
    ('random_matched_trunc', randmatch_nlls, struct_comp, '+ Structure'),
    ('scrambled_oracle_trunc', scrambled_nlls, vocab_comp, '+ Vocabulary'),
    ('oracle_trunc', oracle_nlls, sem_comp, '+ Semantics'),
]:
    mu = component.mean()
    pct = mu / total_mean * 100 if total_mean != 0 else 0
    d = cohens_d(component)
    _, p = stats.ttest_1samp(component, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<30} {nlls.mean():>10.4f} {mu:>+10.4f} {pct:>8.1f}% "
          f"{d:>+8.3f} {p:>12.2e} {sig}")

print(f"  {'TOTAL':<30} {'':>10} {total_mean:>+10.4f} {'100.0%':>9}")

# Decomposition residual
residual = total_mean - (struct_comp.mean() + vocab_comp.mean() + sem_comp.mean())
print(f"\n  Decomposition residual: {residual:.6f} (should be ~0)")

# Percentages
struct_pct = struct_comp.mean() / total_mean * 100 if total_mean != 0 else 0
vocab_pct = vocab_comp.mean() / total_mean * 100 if total_mean != 0 else 0
sem_pct = sem_comp.mean() / total_mean * 100 if total_mean != 0 else 0

print(f"\n  SUMMARY: Structure={struct_pct:.1f}%, Vocabulary={vocab_pct:.1f}%, "
      f"Semantics={sem_pct:.1f}%")

# Compare with MS MARCO Exp 2B
print(f"\n--- Comparison with Exp 2B (MS MARCO, 6-word queries) ---")
print(f"  MS MARCO:  Structure=84.7%, Vocabulary=5.5% (ns), Semantics=9.7% (***)")
print(f"  This data: Structure={struct_pct:.1f}%, Vocabulary={vocab_pct:.1f}%, "
      f"Semantics={sem_pct:.1f}%")

delta_struct = struct_pct - 84.7
delta_sem = sem_pct - 9.7
print(f"  Structure shift: {delta_struct:+.1f} percentage points")
print(f"  Semantics shift: {delta_sem:+.1f} percentage points")

if sem_pct > 20:
    print(f"\n  --> SEMANTIC COMPONENT GREW with longer queries!")
    print(f"      This suggests the MS MARCO finding was partly an artifact of short queries.")
elif sem_pct > 15:
    print(f"\n  --> MODERATE semantic growth. Content matters somewhat more with longer queries.")
else:
    print(f"\n  --> Semantic component is similar to MS MARCO.")
    print(f"      The 85% structural finding appears to be a genuine property of the mechanism.")

# Token count verification
print(f"\n--- Length matching verification ---")
oracle_toks = np.array([r['ptoks_oracle'] for r in results])
scrambled_toks = np.array([r['ptoks_scrambled_oracle_trunc'] for r in results])
randmatch_toks = np.array([r['ptoks_random_matched_trunc'] for r in results])
print(f"  Oracle:     mean={oracle_toks.mean():.1f} toks")
print(f"  Scrambled:  mean={scrambled_toks.mean():.1f} toks")
print(f"  RandMatch:  mean={randmatch_toks.mean():.1f} toks")
tok_diff = np.abs(oracle_toks - scrambled_toks)
print(f"  Oracle-Scrambled token diff: mean={tok_diff.mean():.1f}, max={tok_diff.max()}")
""")


# ===== Cell 9: Part 3 — Token Diversity =====
code(r"""# Cell 9: Part 3 — Token Diversity

print("=" * 70)
print("PART 3: TOKEN DIVERSITY")
print("=" * 70)
print("All conditions length-matched to query word count. Does content matter?\n")

repeat_the_nlls = np.array([r['nll_repeat_the_trunc'] for r in results])
repeat_kw_nlls = np.array([r['nll_repeat_kw_trunc'] for r in results])
surr_tmpl_nlls = np.array([r['nll_surr_template_trunc'] for r in results])

oracle_d = cohens_d(oracle_benefit)

all_conds = [
    ('oracle_trunc', oracle_nlls, 'Real query (upper bound)'),
    ('scrambled_oracle_trunc', scrambled_nlls, 'Query words, shuffled'),
    ('random_matched_trunc', randmatch_nlls, 'Random words, length-matched'),
    ('surr_template_trunc', surr_tmpl_nlls, '"What is [keyword]?"'),
    ('repeat_kw_trunc', repeat_kw_nlls, 'Doc keyword repeated N times'),
    ('repeat_the_trunc', repeat_the_nlls, '"the" repeated N times'),
]

print(f"{'Description':<42} {'NLL':>8} {'Delta':>10} {'d':>8} {'Win%':>7} {'%Orc':>6}")
print("-" * 85)

for name, nlls, desc in all_conds:
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"  {desc:<40} {nlls.mean():>8.4f} {delta:>+10.4f} {d:>+8.3f} {win:>6.1f}% {pct:>5.0f}%")

print(f"  {'bare (lower bound)':<40} {bare_nlls.mean():>8.4f}")

# Pairwise comparisons
print(f"\n--- Pairwise head-to-head ---")
pairs = [
    ('oracle', oracle_nlls, 'scrambled', scrambled_nlls,
     "Does word ORDER matter?"),
    ('scrambled', scrambled_nlls, 'random_matched', randmatch_nlls,
     "Do the right WORDS matter?"),
    ('oracle', oracle_nlls, 'random_matched', randmatch_nlls,
     "Does ANY content matter (oracle vs random)?"),
    ('repeat_the', repeat_the_nlls, 'random_matched', randmatch_nlls,
     "Does diversity help (uniform vs diverse)?"),
    ('repeat_kw', repeat_kw_nlls, 'repeat_the', repeat_the_nlls,
     "Does keyword content help vs pure filler?"),
    ('surr_template', surr_tmpl_nlls, 'random_matched', randmatch_nlls,
     "Does template surrogate beat random?"),
]

for name_a, nlls_a, name_b, nlls_b, question in pairs:
    diff = nlls_b - nlls_a  # positive = A is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = name_a if d > 0 else name_b
    print(f"  {question}")
    print(f"    {name_a} vs {name_b}: d={d:+.3f}, win={win:.1f}%, p={p:.2e} {sig} [{winner}]")
""")


# ===== Cell 10: Part 4 — Query Length Stratification =====
code(r"""# Cell 10: Part 4 — Query Length Stratification (within-dataset)

print("=" * 70)
print("PART 4: QUERY LENGTH STRATIFICATION")
print("=" * 70)
print("Does the structural/semantic balance shift with query length?\n")

q_lens = np.array([r['query_words'] for r in results])

# Use terciles for reasonable bin sizes
tercile_bounds = np.percentile(q_lens, [33, 67])
q_bins = np.digitize(q_lens, tercile_bounds)
bin_labels = []
for b in range(3):
    mask = q_bins == b
    bmin, bmax = q_lens[mask].min(), q_lens[mask].max()
    bmean = q_lens[mask].mean()
    label = f"{'Short' if b==0 else 'Medium' if b==1 else 'Long'} ({bmin}-{bmax}w, mean={bmean:.0f})"
    bin_labels.append(label)

print(f"{'Query bin':<35} {'N':>4} {'Struct%':>9} {'Vocab%':>8} {'Sem%':>7} "
      f"{'Oracle d':>10} {'Rand d':>9}")
print("-" * 90)

bin_struct_pcts = []
bin_sem_pcts = []
bin_q_means = []

for b in range(3):
    mask = q_bins == b
    n = mask.sum()

    # Compute components for this bin
    b_struct = (bare_nlls[mask] - randmatch_nlls[mask]).mean()
    b_vocab = (randmatch_nlls[mask] - scrambled_nlls[mask]).mean()
    b_sem = (scrambled_nlls[mask] - oracle_nlls[mask]).mean()
    b_total = (bare_nlls[mask] - oracle_nlls[mask]).mean()

    if b_total > 0:
        s_pct = b_struct / b_total * 100
        v_pct = b_vocab / b_total * 100
        sm_pct = b_sem / b_total * 100
    else:
        s_pct = v_pct = sm_pct = 0

    o_d = cohens_d(bare_nlls[mask] - oracle_nlls[mask])
    r_d = cohens_d(bare_nlls[mask] - randmatch_nlls[mask])

    bin_struct_pcts.append(s_pct)
    bin_sem_pcts.append(sm_pct)
    bin_q_means.append(q_lens[mask].mean())

    print(f"  {bin_labels[b]:<33} {n:>4} {s_pct:>8.1f}% {v_pct:>7.1f}% {sm_pct:>6.1f}% "
          f"{o_d:>+10.3f} {r_d:>+9.3f}")

# Correlation: query length vs each component
print(f"\n--- Correlations ---")
r_struct_q, p_struct_q = stats.pearsonr(q_lens, struct_comp)
r_vocab_q, p_vocab_q = stats.pearsonr(q_lens, vocab_comp)
r_sem_q, p_sem_q = stats.pearsonr(q_lens, sem_comp)
r_total_q, p_total_q = stats.pearsonr(q_lens, total_comp)
print(f"  query_len vs structure:  r={r_struct_q:+.3f} (p={p_struct_q:.3f})")
print(f"  query_len vs vocabulary: r={r_vocab_q:+.3f} (p={p_vocab_q:.3f})")
print(f"  query_len vs semantics:  r={r_sem_q:+.3f} (p={p_sem_q:.3f})")
print(f"  query_len vs total:      r={r_total_q:+.3f} (p={p_total_q:.3f})")

# Semantic gap: does oracle beat random more for longer queries?
semantic_gap = oracle_benefit - (bare_nlls - randmatch_nlls)
r_gap_q, p_gap_q = stats.pearsonr(q_lens, semantic_gap)
print(f"  query_len vs semantic_gap (oracle-random): r={r_gap_q:+.3f} (p={p_gap_q:.3f})")

if r_gap_q > 0.1 and p_gap_q < 0.05:
    print(f"\n  --> Semantic gap GROWS with query length!")
    print(f"      Content matters MORE for longer queries, as hypothesized.")
elif r_gap_q < -0.1 and p_gap_q < 0.05:
    print(f"\n  --> Semantic gap SHRINKS with query length.")
    print(f"      Structure becomes MORE dominant for longer queries.")
else:
    print(f"\n  --> Semantic gap is STABLE across query lengths (within this dataset).")
""")


# ===== Cell 11: Part 5 — Hardness Stratification =====
code(r"""# Cell 11: Part 5 — Hardness Stratification

print("=" * 70)
print("PART 5: HARDNESS STRATIFICATION")
print("=" * 70)
print("Does the decomposition change for harder documents?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"{'Quintile':<12} {'N':>4} {'Bare NLL':>10} {'Struct%':>9} {'Vocab%':>8} "
      f"{'Sem%':>7} {'Oracle d':>10} {'Sem gap':>10}")
print("-" * 80)

for q in range(5):
    mask = quintiles == q
    n = mask.sum()
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    bare_q = bare_nlls[mask].mean()

    b_struct = (bare_nlls[mask] - randmatch_nlls[mask]).mean()
    b_vocab = (randmatch_nlls[mask] - scrambled_nlls[mask]).mean()
    b_sem = (scrambled_nlls[mask] - oracle_nlls[mask]).mean()
    b_total = (bare_nlls[mask] - oracle_nlls[mask]).mean()

    if b_total > 0:
        s_pct = b_struct / b_total * 100
        v_pct = b_vocab / b_total * 100
        sm_pct = b_sem / b_total * 100
    else:
        s_pct = v_pct = sm_pct = 0

    o_d = cohens_d(bare_nlls[mask] - oracle_nlls[mask])
    gap = (oracle_benefit[mask] - (bare_nlls[mask] - randmatch_nlls[mask])).mean()

    print(f"  {qlabel:<10} {n:>4} {bare_q:>10.3f} {s_pct:>8.1f}% {v_pct:>7.1f}% "
          f"{sm_pct:>6.1f}% {o_d:>+10.3f} {gap:>+10.4f}")

# Correlation: hardness vs each component
print(f"\n--- Correlations ---")
r_s, p_s = stats.pearsonr(bare_nlls, struct_comp)
r_v, p_v = stats.pearsonr(bare_nlls, vocab_comp)
r_sm, p_sm = stats.pearsonr(bare_nlls, sem_comp)
print(f"  hardness vs structure:  r={r_s:+.3f} (p={p_s:.2e})")
print(f"  hardness vs vocabulary: r={r_v:+.3f} (p={p_v:.2e})")
print(f"  hardness vs semantics:  r={r_sm:+.3f} (p={p_sm:.2e})")

# Semantic component x hardness: does content matter more for hard samples?
# (This was true on MS MARCO: Q1 gap=+0.013 -> Q5 gap=+0.397)
print(f"\n--- Semantic gap by hardness ---")
print(f"  MS MARCO Exp 2B: Q1 gap=+0.013, Q5 gap=+0.397")
for q in range(5):
    mask = quintiles == q
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    gap = (oracle_benefit[mask] - (bare_nlls[mask] - randmatch_nlls[mask])).mean()
    print(f"  This data {qlabel}: gap={gap:+.4f}")
""")


# ===== Cell 12: Synthesis + Save =====
code(r"""# Cell 12: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: CROSS-DATASET CONTENT ABLATION RESULTS")
print("=" * 70)

# Key numbers
oracle_d = cohens_d(oracle_benefit)
struct_d = cohens_d(struct_comp)
vocab_d = cohens_d(vocab_comp)
sem_d = cohens_d(sem_comp)

struct_pct = struct_comp.mean() / total_comp.mean() * 100 if total_comp.mean() != 0 else 0
vocab_pct = vocab_comp.mean() / total_comp.mean() * 100 if total_comp.mean() != 0 else 0
sem_pct = sem_comp.mean() / total_comp.mean() * 100 if total_comp.mean() != 0 else 0

repeat_the_d = cohens_d(bare_nlls - repeat_the_nlls)
repeat_kw_d = cohens_d(bare_nlls - repeat_kw_nlls)
randmatch_d = cohens_d(bare_nlls - randmatch_nlls)
surr_tmpl_d = cohens_d(bare_nlls - surr_tmpl_nlls)

# All condition NLLs for the summary
print(f"\n1. DATASET: neural-bridge/rag-dataset-12000")
print(f"   Query length: {q_lens.mean():.1f}w (vs 6.0w MS MARCO)")
print(f"   Doc length: {np.array([r['doc_words'] for r in results]).mean():.0f}w")
print(f"   N: {N_SAMPLES}")

print(f"\n2. ORACLE HEADROOM:")
print(f"   Cohen's d: {oracle_d:+.3f}")
print(f"   NLL improvement: {oracle_benefit.mean():+.4f}")
_, p_oracle = stats.ttest_1samp(oracle_benefit, 0)
print(f"   Win rate: {np.mean(oracle_benefit > 0)*100:.1f}% (p={p_oracle:.2e})")

print(f"\n3. CONTENT ABLATION:")
print(f"   {'Component':<15} {'This dataset':>15} {'MS MARCO':>15} {'Change':>10}")
print(f"   {'-'*60}")
print(f"   {'Structure':<15} {struct_pct:>14.1f}% {'84.7%':>15} {struct_pct-84.7:>+9.1f}pp")
print(f"   {'Vocabulary':<15} {vocab_pct:>14.1f}% {'5.5%':>15} {vocab_pct-5.5:>+9.1f}pp")
print(f"   {'Semantics':<15} {sem_pct:>14.1f}% {'9.7%':>15} {sem_pct-9.7:>+9.1f}pp")

print(f"\n4. TOKEN DIVERSITY:")
print(f"   repeat_the: d={repeat_the_d:+.3f} ({repeat_the_d/oracle_d*100:.0f}% oracle)")
print(f"   repeat_kw:  d={repeat_kw_d:+.3f} ({repeat_kw_d/oracle_d*100:.0f}% oracle)")
print(f"   rand_match: d={randmatch_d:+.3f} ({randmatch_d/oracle_d*100:.0f}% oracle)")
print(f"   surr_tmpl:  d={surr_tmpl_d:+.3f} ({surr_tmpl_d/oracle_d*100:.0f}% oracle)")

print(f"\n5. QUERY LENGTH EFFECT:")
for b in range(3):
    mask = q_bins == b
    b_total = (bare_nlls[mask] - oracle_nlls[mask]).mean()
    if b_total > 0:
        b_sem_pct = (scrambled_nlls[mask] - oracle_nlls[mask]).mean() / b_total * 100
    else:
        b_sem_pct = 0
    print(f"   {bin_labels[b]}: semantic={b_sem_pct:.1f}%")

# Determine overall conclusion
if struct_pct > 70:
    mechanism = "STRUCTURAL"
elif struct_pct > 50:
    mechanism = "MIXED"
else:
    mechanism = "SEMANTIC"

if abs(struct_pct - 84.7) < 10:
    cross_dataset = "CONSISTENT"
elif struct_pct < 74.7:
    cross_dataset = "SEMANTIC_GREW"
else:
    cross_dataset = "STRUCTURAL_GREW"

print(f"\n{'='*70}")
print(f"CONCLUSION:")
print(f"  Mechanism: {mechanism}")
print(f"  Cross-dataset consistency: {cross_dataset}")
if mechanism == "STRUCTURAL":
    print(f"  The 85% structural finding holds with 3x longer queries on a different dataset.")
    print(f"  This is a genuine property of the T5Gemma bidirectional encoder mechanism,")
    print(f"  not an artifact of MS MARCO's short queries.")
elif mechanism == "MIXED":
    print(f"  With longer queries, the semantic component grew meaningfully.")
    print(f"  The mechanism is still primarily structural, but content matters more")
    print(f"  when queries carry more semantic structure to preserve.")
else:
    print(f"  With longer queries, the semantic component dominates.")
    print(f"  The MS MARCO finding WAS an artifact of short queries.")
    print(f"  Content-aware surrogates are worthwhile for long-query use cases.")
print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp03d_cross_dataset_content_ablation',
    'model': MODEL_NAME,
    'dataset': 'neural-bridge/rag-dataset-12000',
    'n_samples': N_SAMPLES,
    'mean_query_words': float(q_lens.mean()),
    'mean_doc_words': float(np.array([r['doc_words'] for r in results]).mean()),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'baseline': {
        'bare_nll': float(bare_nlls.mean()),
        'oracle_nll': float(oracle_nlls.mean()),
        'oracle_d': float(oracle_d),
        'oracle_headroom': float(oracle_benefit.mean()),
        'oracle_p': float(p_oracle),
    },
    'ablation': {
        'structure_pct': float(struct_pct),
        'vocabulary_pct': float(vocab_pct),
        'semantics_pct': float(sem_pct),
        'structure_d': float(struct_d),
        'vocabulary_d': float(vocab_d),
        'semantics_d': float(sem_d),
    },
    'conditions': {
        'oracle_trunc_d': float(oracle_d),
        'scrambled_oracle_trunc_d': float(cohens_d(bare_nlls - scrambled_nlls)),
        'random_matched_trunc_d': float(randmatch_d),
        'surr_template_trunc_d': float(surr_tmpl_d),
        'repeat_the_trunc_d': float(repeat_the_d),
        'repeat_kw_trunc_d': float(repeat_kw_d),
    },
    'comparison_with_msmarco': {
        'struct_pct_delta': float(struct_pct - 84.7),
        'vocab_pct_delta': float(vocab_pct - 5.5),
        'sem_pct_delta': float(sem_pct - 9.7),
    },
    'conclusion': {
        'mechanism': mechanism,
        'cross_dataset': cross_dataset,
    },
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
out_path = "03d_cross_dataset_ablation.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
