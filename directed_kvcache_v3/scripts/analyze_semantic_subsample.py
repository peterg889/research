#!/usr/bin/env python3
"""Analyze the semantic subsample: which MS MARCO samples benefit from query CONTENT
(not just structural prefix presence) in the directed KV cache?

Uses per-sample NLLs from Exp 02, 02B, and 05 to define, characterize, and predict
the "semantic subsample" -- samples where oracle > random (query content helps).
"""

import json
import re
import string
import numpy as np
from scipy import stats
from collections import Counter

# ============================================================
# Load data
# ============================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

exp02 = json.load(open("results/exp02/checkpoint.json"))["results"]
exp02b = json.load(open("results/exp02b/checkpoint.json"))["results"]
exp05 = json.load(open("results/exp05/checkpoint.json"))["results"]

N = len(exp02)
assert len(exp02b) == N and len(exp05) == N, "Sample count mismatch"
assert all(exp02[i]["query"] == exp02b[i]["query"] == exp05[i]["query"] for i in range(N)), \
    "Query mismatch across checkpoints"

print(f"Loaded {N} matched samples from exp02, exp02b, exp05")

# Reload passages from MS MARCO (not stored in checkpoints)
print("\nReloading passages from MS MARCO v1.1 validation...")
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Reconstruct samples with same selection logic and seed as build_exp02.py
SEED = 42
raw_samples = []
for item in ds:
    if len(raw_samples) >= N * 3:
        break
    passages = item.get("passages", {})
    ptexts = passages.get("passage_text", [])
    is_sel = passages.get("is_selected", [])
    query = item.get("query", "")
    answers = item.get("answers", [])
    well_formed = item.get("wellFormedAnswers", [])

    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ("[]", ""):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != "No Answer Present.":
        answer = answers[0]
    if not answer:
        continue

    for pt, sel in zip(ptexts, is_sel):
        wc = len(pt.split())
        if sel == 1 and 30 <= wc <= 300:
            raw_samples.append({"passage": pt, "query": query, "answer": answer, "word_count": wc})
            break

np.random.seed(SEED)
np.random.shuffle(raw_samples)
raw_samples = raw_samples[:N]

# Verify alignment
for i in range(N):
    assert raw_samples[i]["query"] == exp02[i]["query"], f"Mismatch at {i}: {raw_samples[i]['query'][:40]} vs {exp02[i]['query'][:40]}"
print(f"Passages aligned with checkpoint data (verified all {N} queries match)")

passages = [s["passage"] for s in raw_samples]
del ds, raw_samples
import gc; gc.collect()

# ============================================================
# STOP_WORDS and helpers
# ============================================================
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

def content_words(text):
    """Extract content words: lowercase, strip punctuation, remove stop words."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]

def count_sentences(text):
    """Count sentences in text (rough proxy)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if len(s.strip()) > 5])

def vocab_overlap(query, passage):
    """Fraction of query content words that appear in passage."""
    q_words = set(content_words(query))
    p_words = set(content_words(passage))
    if len(q_words) == 0:
        return 0.0
    return len(q_words & p_words) / len(q_words)

cohens_d = lambda diff: float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0

# ============================================================
# PART 1: Define the "semantic subsample"
# ============================================================
print("\n" + "=" * 80)
print("PART 1: DEFINING THE SEMANTIC SUBSAMPLE")
print("=" * 80)

nll_bare = np.array([r["nll_bare"] for r in exp02])
nll_oracle = np.array([r["nll_oracle_trunc"] for r in exp02])
nll_random = np.array([r["nll_random_trunc"] for r in exp02])

semantic_gap = nll_random - nll_oracle  # positive = oracle better than random
total_benefit = nll_bare - nll_oracle   # positive = oracle better than bare
structural_benefit = nll_bare - nll_random  # positive = random better than bare

print(f"\nsemantic_gap = nll_random_trunc - nll_oracle_trunc")
print(f"  (positive means query content helps beyond structure)")
print(f"\nDistribution of semantic_gap (N={N}):")
print(f"  Mean:       {np.mean(semantic_gap):+.4f}")
print(f"  Median:     {np.median(semantic_gap):+.4f}")
print(f"  Std:        {np.std(semantic_gap):.4f}")
print(f"  Min:        {np.min(semantic_gap):+.4f}")
print(f"  Max:        {np.max(semantic_gap):+.4f}")
print(f"  % positive: {100 * np.mean(semantic_gap > 0):.1f}%")
print(f"  % > 0.1:    {100 * np.mean(semantic_gap > 0.1):.1f}%")
print(f"  % > 0.5:    {100 * np.mean(semantic_gap > 0.5):.1f}%")
print(f"  % > 1.0:    {100 * np.mean(semantic_gap > 1.0):.1f}%")
print(f"  % > 2.0:    {100 * np.mean(semantic_gap > 2.0):.1f}%")

# Define semantic subsample: semantic_gap > median AND > 0
median_gap = np.median(semantic_gap)
semantic_mask = (semantic_gap > median_gap) & (semantic_gap > 0)
nonsemantic_mask = ~semantic_mask

n_semantic = semantic_mask.sum()
n_nonsemantic = nonsemantic_mask.sum()

print(f"\nSemantic subsample definition:")
print(f"  semantic_gap > median ({median_gap:+.4f}) AND semantic_gap > 0")
print(f"  N semantic:     {n_semantic} ({100*n_semantic/N:.1f}%)")
print(f"  N non-semantic: {n_nonsemantic} ({100*n_nonsemantic/N:.1f}%)")

# ============================================================
# PART 2: Characterize the semantic subsample
# ============================================================
print("\n" + "=" * 80)
print("PART 2: CHARACTERIZING THE SEMANTIC SUBSAMPLE")
print("=" * 80)

passage_words = np.array([r["passage_words"] for r in exp02])
query_words = np.array([len(r["query"].split()) for r in exp02])
answer_words = np.array([len(r["answer"].split()) for r in exp02])

def subsample_stats(label, mask):
    n = mask.sum()
    print(f"\n  [{label}] (N={n}):")
    print(f"    Mean bare NLL:            {nll_bare[mask].mean():.4f}")
    print(f"    Mean passage words:       {passage_words[mask].mean():.1f}")
    print(f"    Mean query words:         {query_words[mask].mean():.1f}")
    print(f"    Mean answer words:        {answer_words[mask].mean():.1f}")
    print(f"    Structural benefit (d):   {cohens_d(structural_benefit[mask]):+.3f}")
    print(f"    Total benefit (d):        {cohens_d(total_benefit[mask]):+.3f}")
    print(f"    Mean semantic_gap:        {semantic_gap[mask].mean():+.4f}")

    # Semantic percentage
    tb = total_benefit[mask]
    sg = semantic_gap[mask]
    # Per-sample percentage (only where total_benefit > 0)
    valid = tb > 0.01  # avoid division by near-zero
    if valid.sum() > 0:
        pct_semantic = 100 * sg[valid] / tb[valid]
        print(f"    Semantic % of total:      {np.mean(pct_semantic):.1f}% (mean per-sample, {valid.sum()} valid)")
        print(f"    Semantic % (median):      {np.median(pct_semantic):.1f}%")
    else:
        print(f"    Semantic % of total:      N/A (no valid samples)")

print("Comparison: semantic vs non-semantic subsamples")
subsample_stats("SEMANTIC", semantic_mask)
subsample_stats("NON-SEMANTIC", nonsemantic_mask)

# Statistical tests between groups
print(f"\n  --- Mann-Whitney U tests (semantic vs non-semantic) ---")
for label, arr in [("bare NLL", nll_bare), ("passage_words", passage_words),
                   ("query_words", query_words), ("answer_words", answer_words),
                   ("structural_benefit", structural_benefit), ("total_benefit", total_benefit)]:
    u, p = stats.mannwhitneyu(arr[semantic_mask], arr[nonsemantic_mask], alternative="two-sided")
    d_btwn = cohens_d(arr[semantic_mask]) - cohens_d(arr[nonsemantic_mask])  # rough
    # Better: compute actual mean diff / pooled std
    m1, m2 = arr[semantic_mask].mean(), arr[nonsemantic_mask].mean()
    s1, s2 = arr[semantic_mask].std(), arr[nonsemantic_mask].std()
    n1, n2 = semantic_mask.sum(), nonsemantic_mask.sum()
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)) if (n1+n2-2) > 0 else 1.0
    d_between = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {label:<25} sem={m1:.3f} non={m2:.3f} d={d_between:+.3f} p={p:.2e} {sig}")

# ============================================================
# PART 3: Content analysis
# ============================================================
print("\n" + "=" * 80)
print("PART 3: WHAT MAKES SEMANTIC SAMPLES SPECIAL? CONTENT ANALYSIS")
print("=" * 80)

# Sort by semantic_gap
sorted_idx = np.argsort(semantic_gap)  # ascending
top50_idx = sorted_idx[-50:][::-1]  # highest semantic_gap
bot50_idx = sorted_idx[:50]          # lowest semantic_gap (most negative)

print(f"\n--- TOP 50 samples (highest semantic_gap: oracle >> random) ---")
print(f"{'#':>3} {'sem_gap':>8} {'query':<50} {'answer':<30} {'passage':<50}")
print("-" * 145)
for rank, idx in enumerate(top50_idx[:10]):
    r = exp02[idx]
    print(f"{rank+1:>3} {semantic_gap[idx]:>+8.3f} {r['query'][:50]:<50} "
          f"{r['answer'][:30]:<30} {passages[idx][:50]}")

print(f"\n--- BOTTOM 50 samples (lowest semantic_gap: random >= oracle) ---")
print(f"{'#':>3} {'sem_gap':>8} {'query':<50} {'answer':<30} {'passage':<50}")
print("-" * 145)
for rank, idx in enumerate(bot50_idx[:10]):
    r = exp02[idx]
    print(f"{rank+1:>3} {semantic_gap[idx]:>+8.3f} {r['query'][:50]:<50} "
          f"{r['answer'][:30]:<30} {passages[idx][:50]}")

# Query-passage vocabulary overlap
all_overlap = np.array([vocab_overlap(exp02[i]["query"], passages[i]) for i in range(N)])
top50_overlap = np.array([vocab_overlap(exp02[idx]["query"], passages[idx]) for idx in top50_idx])
bot50_overlap = np.array([vocab_overlap(exp02[idx]["query"], passages[idx]) for idx in bot50_idx])

print(f"\n--- Query-passage vocabulary overlap (fraction of query content words in passage) ---")
print(f"  Top 50 (semantic):       {top50_overlap.mean():.3f} (std={top50_overlap.std():.3f})")
print(f"  Bottom 50 (non-semantic):{bot50_overlap.mean():.3f} (std={bot50_overlap.std():.3f})")
print(f"  Full sample:             {all_overlap.mean():.3f} (std={all_overlap.std():.3f})")

# Answer length
top50_anslen = np.array([len(exp02[idx]["answer"].split()) for idx in top50_idx])
bot50_anslen = np.array([len(exp02[idx]["answer"].split()) for idx in bot50_idx])
print(f"\n--- Answer length (words) ---")
print(f"  Top 50 (semantic):       {top50_anslen.mean():.1f} (std={top50_anslen.std():.1f})")
print(f"  Bottom 50 (non-semantic):{bot50_anslen.mean():.1f} (std={bot50_anslen.std():.1f})")

# Sentence count (proxy for answer density)
top50_sents = np.array([count_sentences(passages[idx]) for idx in top50_idx])
bot50_sents = np.array([count_sentences(passages[idx]) for idx in bot50_idx])
print(f"\n--- Passage sentence count (proxy for # plausible answers) ---")
print(f"  Top 50 (semantic):       {top50_sents.mean():.1f} (std={top50_sents.std():.1f})")
print(f"  Bottom 50 (non-semantic):{bot50_sents.mean():.1f} (std={bot50_sents.std():.1f})")

# Passage word count for top/bottom
top50_words = np.array([exp02[idx]["passage_words"] for idx in top50_idx])
bot50_words = np.array([exp02[idx]["passage_words"] for idx in bot50_idx])
print(f"\n--- Passage word count ---")
print(f"  Top 50 (semantic):       {top50_words.mean():.1f} (std={top50_words.std():.1f})")
print(f"  Bottom 50 (non-semantic):{bot50_words.mean():.1f} (std={bot50_words.std():.1f})")

# Query word count for top/bottom
top50_qwords = np.array([len(exp02[idx]["query"].split()) for idx in top50_idx])
bot50_qwords = np.array([len(exp02[idx]["query"].split()) for idx in bot50_idx])
print(f"\n--- Query word count ---")
print(f"  Top 50 (semantic):       {top50_qwords.mean():.1f} (std={top50_qwords.std():.1f})")
print(f"  Bottom 50 (non-semantic):{bot50_qwords.mean():.1f} (std={bot50_qwords.std():.1f})")

# Query type analysis
def classify_query(q):
    q_lower = q.lower().strip()
    if q_lower.startswith("what"):
        return "what"
    elif q_lower.startswith("how"):
        return "how"
    elif q_lower.startswith("where"):
        return "where"
    elif q_lower.startswith("when"):
        return "when"
    elif q_lower.startswith("who"):
        return "who"
    elif q_lower.startswith("why"):
        return "why"
    elif q_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ")):
        return "yes/no"
    elif q_lower.startswith("define") or "definition" in q_lower or "meaning of" in q_lower:
        return "definition"
    else:
        return "other"

all_qtypes = [classify_query(exp02[i]["query"]) for i in range(N)]
sem_qtypes = [classify_query(exp02[i]["query"]) for i in range(N) if semantic_mask[i]]
nonsem_qtypes = [classify_query(exp02[i]["query"]) for i in range(N) if nonsemantic_mask[i]]

print(f"\n--- Query type distribution ---")
print(f"  {'Type':<15} {'All':>8} {'Semantic':>10} {'Non-sem':>10} {'Sem %':>8}")
print(f"  {'-'*53}")
for qtype in sorted(set(all_qtypes)):
    n_all = all_qtypes.count(qtype)
    n_sem = sem_qtypes.count(qtype)
    n_non = nonsem_qtypes.count(qtype)
    pct_sem = 100 * n_sem / (n_sem + n_non) if (n_sem + n_non) > 0 else 0
    print(f"  {qtype:<15} {n_all:>8} {n_sem:>10} {n_non:>10} {pct_sem:>7.1f}%")

# ============================================================
# PART 4: Exp 05 LLM conditions on the semantic subsample
# ============================================================
print("\n" + "=" * 80)
print("PART 4: EXP 05 LLM CONDITIONS ON THE SEMANTIC SUBSAMPLE")
print("=" * 80)

nll_bare_05 = np.array([r["nll_bare"] for r in exp05])

conditions_05 = {
    "oracle_x1":        "nll_oracle_x1_trunc",
    "random_x1":        "nll_random_x1_trunc",
    "surr_template_x1": "nll_surr_template_x1_trunc",
    "llm_need_x1":      "nll_llm_need_x1_trunc",
    "llm_question_x1":  "nll_llm_question_x1_trunc",
    "llm_keywords_x1":  "nll_llm_keywords_x1_trunc",
}

print(f"\n--- Cohen's d vs bare (full sample vs semantic subsample) ---")
print(f"  {'Condition':<25} {'Full d':>10} {'Semantic d':>12} {'Non-sem d':>12} {'Sem uplift':>12}")
print(f"  {'-'*73}")

for cond_name, field in conditions_05.items():
    nlls = np.array([r[field] for r in exp05])
    diff_full = nll_bare_05 - nlls
    diff_sem = diff_full[semantic_mask]
    diff_non = diff_full[nonsemantic_mask]

    d_full = cohens_d(diff_full)
    d_sem = cohens_d(diff_sem)
    d_non = cohens_d(diff_non)
    uplift = d_sem - d_non

    print(f"  {cond_name:<25} {d_full:>+10.3f} {d_sem:>+12.3f} {d_non:>+12.3f} {uplift:>+12.3f}")

# Head-to-head: LLM need vs template on semantic subsample
print(f"\n--- Head-to-head: LLM need x1 vs surr_template x1 ---")
nll_need = np.array([r["nll_llm_need_x1_trunc"] for r in exp05])
nll_template = np.array([r["nll_surr_template_x1_trunc"] for r in exp05])

for label, mask in [("Full sample", np.ones(N, dtype=bool)),
                    ("Semantic", semantic_mask),
                    ("Non-semantic", nonsemantic_mask)]:
    diff = nll_template[mask] - nll_need[mask]  # positive = need is better (lower NLL)
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    if len(diff) > 1:
        t, p = stats.ttest_1samp(diff, 0)
    else:
        p = 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    winner = "llm_need" if d > 0 else "template"
    print(f"  {label:<20} d={d:>+.3f}  win%={win:>5.1f}%  p={p:.2e} {sig}  [{winner}]")

# Also check: does LLM need outperform random on the semantic subsample?
print(f"\n--- Head-to-head: LLM need x1 vs random x1 (on semantic subsample) ---")
nll_random_05 = np.array([r["nll_random_x1_trunc"] for r in exp05])

for label, mask in [("Full sample", np.ones(N, dtype=bool)),
                    ("Semantic", semantic_mask),
                    ("Non-semantic", nonsemantic_mask)]:
    diff = nll_random_05[mask] - nll_need[mask]  # positive = need is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    if len(diff) > 1:
        t, p = stats.ttest_1samp(diff, 0)
    else:
        p = 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    winner = "llm_need" if d > 0 else "random"
    print(f"  {label:<20} d={d:>+.3f}  win%={win:>5.1f}%  p={p:.2e} {sig}  [{winner}]")

# Semantic decomposition for exp05 conditions on semantic subsample
print(f"\n--- Mechanism decomposition on semantic subsample (exp05 data) ---")
nll_oracle_05 = np.array([r["nll_oracle_x1_trunc"] for r in exp05])
struct_05 = nll_bare_05 - nll_random_05
semantic_05 = nll_random_05 - nll_oracle_05
total_05 = nll_bare_05 - nll_oracle_05

for label, mask in [("Full sample", np.ones(N, dtype=bool)),
                    ("Semantic", semantic_mask),
                    ("Non-semantic", nonsemantic_mask)]:
    t = total_05[mask]
    s = struct_05[mask]
    sem = semantic_05[mask]
    valid = t > 0.01
    if valid.sum() > 0:
        struct_pct = 100 * np.mean(s[valid] / t[valid])
        sem_pct = 100 * np.mean(sem[valid] / t[valid])
    else:
        struct_pct = sem_pct = 0
    print(f"  {label:<20} structural={struct_pct:.1f}%  semantic={sem_pct:.1f}%  "
          f"mean_total={t.mean():+.3f}  mean_semantic_gap={sem.mean():+.3f}")

# ============================================================
# PART 5: Predicting semantic samples
# ============================================================
print("\n" + "=" * 80)
print("PART 5: CAN WE PREDICT WHICH SAMPLES ARE 'SEMANTIC'?")
print("=" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Features
X = np.column_stack([
    nll_bare,
    passage_words,
    query_words,
    answer_words,
    all_overlap,  # query-passage vocabulary overlap
])
feature_names = ["bare_nll", "passage_words", "query_words", "answer_words", "vocab_overlap"]
y = semantic_mask.astype(int)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression with cross-validation
lr = LogisticRegression(max_iter=1000, random_state=SEED)
cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")

print(f"\nLogistic Regression (5-fold CV):")
print(f"  Features: {feature_names}")
print(f"  AUC per fold: {', '.join(f'{s:.3f}' for s in cv_scores)}")
print(f"  Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Fit on full data to get coefficients
lr.fit(X_scaled, y)
print(f"\n  Logistic regression coefficients (standardized):")
for name, coef in sorted(zip(feature_names, lr.coef_[0]), key=lambda x: -abs(x[1])):
    direction = "+" if coef > 0 else "-"
    print(f"    {name:<20} {coef:>+.3f}  (more {name} -> {'MORE' if coef > 0 else 'LESS'} likely semantic)")

# Individual feature AUCs
print(f"\n  Individual feature AUCs (univariate):")
for i, name in enumerate(feature_names):
    xi = X_scaled[:, i].reshape(-1, 1)
    try:
        auc_i = roc_auc_score(y, xi)
        # Flip if < 0.5 (negative correlation)
        if auc_i < 0.5:
            auc_i = 1 - auc_i
            direction = "negative"
        else:
            direction = "positive"
        print(f"    {name:<20} AUC={auc_i:.3f} ({direction} relationship)")
    except:
        print(f"    {name:<20} AUC=N/A")

# Correlation: semantic_gap vs vocabulary overlap
r_overlap, p_overlap = stats.pearsonr(semantic_gap, all_overlap)
r_overlap_s, p_overlap_s = stats.spearmanr(semantic_gap, all_overlap)
print(f"\n--- Correlation: semantic_gap vs vocabulary overlap ---")
print(f"  Pearson r  = {r_overlap:+.3f} (p={p_overlap:.2e})")
print(f"  Spearman r = {r_overlap_s:+.3f} (p={p_overlap_s:.2e})")

# Correlation: semantic_gap vs other features
print(f"\n--- Correlations: semantic_gap vs features ---")
for name, arr in [("bare_nll", nll_bare), ("passage_words", passage_words),
                  ("query_words", query_words), ("answer_words", answer_words),
                  ("vocab_overlap", all_overlap), ("structural_benefit", structural_benefit)]:
    r_val, p_val = stats.pearsonr(semantic_gap, arr)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {name:<25} r={r_val:+.3f} (p={p_val:.2e}) {sig}")

# Additional: try a simple threshold classifier
print(f"\n--- Simple threshold classifiers ---")
for name, arr in [("bare_nll", nll_bare), ("passage_words", passage_words),
                  ("vocab_overlap", all_overlap), ("answer_words", answer_words)]:
    # Find optimal threshold
    best_acc = 0
    best_thresh = 0
    for pctile in range(5, 96, 5):
        thresh = np.percentile(arr, pctile)
        pred_pos = arr > thresh
        acc = np.mean(pred_pos == semantic_mask)
        pred_neg = arr <= thresh
        acc_neg = np.mean(pred_neg == semantic_mask)
        if acc_neg > acc:
            acc = acc_neg
            direction = "<="
        else:
            direction = ">"
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_dir = direction
    print(f"  {name:<20} best_acc={best_acc:.3f} at {best_dir} {best_thresh:.2f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY AND INTERPRETATION")
print("=" * 80)

print(f"""
1. SEMANTIC SUBSAMPLE DEFINITION
   - semantic_gap = nll_random - nll_oracle (query content benefit beyond structure)
   - Mean gap: {np.mean(semantic_gap):+.4f}, {100*np.mean(semantic_gap > 0):.1f}% of samples positive
   - Semantic subsample: N={n_semantic} samples where gap > median ({median_gap:+.4f}) AND > 0

2. CHARACTERIZATION
   - Semantic samples have {'HIGHER' if nll_bare[semantic_mask].mean() > nll_bare[nonsemantic_mask].mean() else 'LOWER'} bare NLL ({nll_bare[semantic_mask].mean():.2f} vs {nll_bare[nonsemantic_mask].mean():.2f}) -- {'harder' if nll_bare[semantic_mask].mean() > nll_bare[nonsemantic_mask].mean() else 'easier'} samples
   - Passage words: {passage_words[semantic_mask].mean():.0f} vs {passage_words[nonsemantic_mask].mean():.0f} ({'longer' if passage_words[semantic_mask].mean() > passage_words[nonsemantic_mask].mean() else 'shorter'} for semantic)
   - Query words: {query_words[semantic_mask].mean():.1f} vs {query_words[nonsemantic_mask].mean():.1f}
   - Answer words: {answer_words[semantic_mask].mean():.1f} vs {answer_words[nonsemantic_mask].mean():.1f}

3. CONTENT ANALYSIS (Top 50 vs Bottom 50)
   - Vocabulary overlap: {top50_overlap.mean():.3f} (semantic) vs {bot50_overlap.mean():.3f} (non-semantic)
   - Passage sentences: {top50_sents.mean():.1f} vs {bot50_sents.mean():.1f}
   - Answer length: {top50_anslen.mean():.1f} vs {bot50_anslen.mean():.1f} words

4. LLM CONDITIONS (Exp 05) ON SEMANTIC SUBSAMPLE
   - LLM need vs template: see head-to-head results above
   - Semantic decomposition: semantic fraction is {'larger' if semantic_05[semantic_mask].mean() > semantic_05[nonsemantic_mask].mean() else 'similar or smaller'} on semantic subsample

5. PREDICTABILITY
   - Logistic regression AUC: {cv_scores.mean():.3f} (5-fold CV)
   - Most predictive feature: {feature_names[np.argmax(np.abs(lr.coef_[0]))]}
   - semantic_gap vs vocab_overlap: r={r_overlap:+.3f} (p={p_overlap:.2e})
""")

print("Done.")
