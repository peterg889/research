#!/usr/bin/env python3
"""Diagnostic analysis of Exp 11/12 data — no GPU needed.

Analyzes:
1. Answer position vs. priming benefit (does contamination hurt answers near the start?)
2. Hurt-tail characterization (what makes the fat tail of harmed samples?)
3. Cross-condition consistency (are the same samples hurt across conditions?)
4. Contamination benefit by document structure
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

os.umask(0o000)

# ─── Load data ───────────────────────────────────────────────────────────────

print("=" * 70)
print("DIAGNOSTIC ANALYSIS OF EXP 11/12 DATA")
print("=" * 70)

# Exp 11: 5 conditions x 390 samples
with open("results/exp11/results.json") as f:
    exp11 = json.load(f)

with open("results/exp11/nq_samples.json") as f:
    nq11 = json.load(f)

# Exp 12: 9 conditions x 315 samples (subset of exp 11)
with open("results/exp12/results.json") as f:
    exp12 = json.load(f)

with open("results/exp12/nq_samples.json") as f:
    nq12 = json.load(f)

samples_11 = nq11["samples"]
results_11 = exp11["per_sample_results"]
samples_12 = nq12["samples"]
results_12 = exp12["per_sample_results"]

print(f"\nExp 11: {len(results_11)} samples, {len(exp11['condition_names'])} conditions")
print(f"Exp 12: {len(results_12)} samples, {len(exp12['condition_names'])} conditions")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Answer Position vs. Priming Benefit
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ANALYSIS 1: ANSWER POSITION vs PRIMING BENEFIT")
print("=" * 70)

# Find answer positions via text matching
def find_answer_position(passage, answer):
    """Find answer character position in passage. Returns fractional position 0-1."""
    if not answer or not passage:
        return None
    # Try exact match
    pos = passage.lower().find(answer.lower())
    if pos >= 0:
        return pos / max(len(passage), 1)
    # Try word-level match (handles minor formatting diffs)
    answer_words = answer.lower().split()
    passage_words = passage.lower().split()
    if len(answer_words) == 0:
        return None
    for i in range(len(passage_words) - len(answer_words) + 1):
        if passage_words[i:i + len(answer_words)] == answer_words:
            # Convert word position to approximate fractional position
            return i / max(len(passage_words), 1)
    return None


# Compute answer positions for exp 11 samples
answer_positions = []
matched_count = 0
for i, sample in enumerate(samples_11):
    pos = find_answer_position(sample["passage"], sample["answer"])
    answer_positions.append(pos)
    if pos is not None:
        matched_count += 1

print(f"\nAnswer position matching: {matched_count}/{len(samples_11)} "
      f"({100*matched_count/len(samples_11):.1f}%) matched")

# Distribution of answer positions
matched_positions = [p for p in answer_positions if p is not None]
if matched_positions:
    quartiles = [0, 0.25, 0.5, 0.75, 1.0]
    print("\nAnswer position distribution:")
    for i in range(4):
        count = sum(1 for p in matched_positions if quartiles[i] <= p < quartiles[i+1])
        pct = 100 * count / len(matched_positions)
        print(f"  {int(quartiles[i]*100):3d}-{int(quartiles[i+1]*100):3d}% of document: "
              f"{count:4d} samples ({pct:.1f}%)")

# Correlate answer position with priming benefit (exp 11)
print("\nAnswer position × priming benefit correlation (Exp 11):")
print("-" * 65)

conditions_11 = ["static_fact_trunc", "random_trunc", "llm_kw_trunc", "oracle_trunc"]

for cond in conditions_11:
    positions = []
    deltas = []
    for i, r in enumerate(results_11):
        if answer_positions[i] is not None and r["bare"] != 0:
            positions.append(answer_positions[i])
            deltas.append(r["bare"] - r[cond])  # positive = priming helps

    positions = np.array(positions)
    deltas = np.array(deltas)

    if len(positions) > 10:
        r = np.corrcoef(positions, deltas)[0, 1]
        # Split into answer-early vs answer-late
        early_mask = positions < 0.25
        late_mask = positions >= 0.25
        early_d = np.mean(deltas[early_mask]) / max(np.std(deltas[early_mask]), 1e-10)
        late_d = np.mean(deltas[late_mask]) / max(np.std(deltas[late_mask]), 1e-10)
        early_mean = np.mean(deltas[early_mask])
        late_mean = np.mean(deltas[late_mask])

        print(f"\n  {cond}:")
        print(f"    Pearson r(answer_pos, delta): {r:+.3f}")
        print(f"    Answer in first 25%:  mean Δ = {early_mean:+.4f}  d = {early_d:+.3f}  "
              f"(n={np.sum(early_mask)})")
        print(f"    Answer in later 75%:  mean Δ = {late_mean:+.4f}  d = {late_d:+.3f}  "
              f"(n={np.sum(late_mask)})")

        # Finer breakdown by answer position quartile
        for q_lo, q_hi, label in [(0, 0.10, "0-10%"), (0.10, 0.25, "10-25%"),
                                   (0.25, 0.50, "25-50%"), (0.50, 1.0, "50-100%")]:
            mask = (positions >= q_lo) & (positions < q_hi)
            if np.sum(mask) >= 5:
                qmean = np.mean(deltas[mask])
                qd = qmean / max(np.std(deltas[mask]), 1e-10)
                print(f"      Answer at {label}: Δ={qmean:+.4f}  d={qd:+.3f}  (n={np.sum(mask)})")

# Answer position × length bin interaction
print("\n\nAnswer position × length bin × priming benefit:")
print("-" * 65)

for length_bin in ["short", "medium", "long", "very_long"]:
    bin_mask = [r["length_bin"] == length_bin for r in results_11]
    bin_indices = [i for i, m in enumerate(bin_mask) if m]

    if len(bin_indices) < 5:
        continue

    print(f"\n  {length_bin} (n={len(bin_indices)}):")

    # Answer position distribution for this bin
    bin_positions = [answer_positions[i] for i in bin_indices if answer_positions[i] is not None]
    if bin_positions:
        mean_pos = np.mean(bin_positions)
        median_pos = np.median(bin_positions)
        print(f"    Answer position: mean={mean_pos:.2f}, median={median_pos:.2f}")

    # Priming benefit for early vs late answers
    for cond in ["static_fact_trunc", "oracle_trunc"]:
        early_deltas = []
        late_deltas = []
        for i in bin_indices:
            if answer_positions[i] is not None and results_11[i]["bare"] != 0:
                delta = results_11[i]["bare"] - results_11[i][cond]
                if answer_positions[i] < 0.25:
                    early_deltas.append(delta)
                else:
                    late_deltas.append(delta)

        if len(early_deltas) >= 3 and len(late_deltas) >= 3:
            e_mean = np.mean(early_deltas)
            l_mean = np.mean(late_deltas)
            print(f"    {cond}: early_ans Δ={e_mean:+.4f} (n={len(early_deltas)})  "
                  f"late_ans Δ={l_mean:+.4f} (n={len(late_deltas)})")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Hurt-Tail Characterization
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 2: HURT-TAIL CHARACTERIZATION")
print("=" * 70)

# For exp 12's best condition (amplify_2x), identify most-hurt samples
print("\nIdentifying samples most HURT by each condition (Exp 12):")
print("-" * 65)

conditions_12 = ["prefix_1x", "amplify_2x", "amplify_5x", "layers_0_15"]

for cond in conditions_12:
    deltas = []
    for r in results_12:
        delta = r["bare"] - r[cond]  # positive = priming helps
        deltas.append(delta)

    deltas = np.array(deltas)

    # Stats
    helped = np.sum(deltas > 0)
    hurt = np.sum(deltas < 0)
    neutral = np.sum(deltas == 0)

    # Magnitude asymmetry
    helped_magnitude = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    hurt_magnitude = np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0

    print(f"\n  {cond}:")
    print(f"    Helped: {helped} ({100*helped/len(deltas):.0f}%)  "
          f"avg magnitude: {helped_magnitude:+.4f}")
    print(f"    Hurt:   {hurt} ({100*hurt/len(deltas):.0f}%)  "
          f"avg magnitude: {hurt_magnitude:+.4f}")
    print(f"    Ratio (hurt_mag / help_mag): {abs(hurt_magnitude/helped_magnitude):.2f}x")

    # Top 10 most hurt
    sorted_indices = np.argsort(deltas)  # most hurt first
    print(f"    Top 10 most HURT samples:")
    for rank, idx in enumerate(sorted_indices[:10]):
        r = results_12[idx]
        s = samples_12[idx]
        apos = find_answer_position(s["passage"], s["answer"])
        apos_str = f"{apos:.2f}" if apos is not None else "N/A"
        print(f"      #{rank+1}: idx={r['idx']}  Δ={deltas[idx]:+.4f}  "
              f"bare={r['bare']:.3f}  bin={r['length_bin']:>10s}  "
              f"words={r['word_count']:5d}  ans_pos={apos_str}")

# Characterize the hurt tail
print("\n\nHurt-tail characteristics (Exp 12, prefix_1x):")
print("-" * 65)

deltas_1x = np.array([r["bare"] - r["prefix_1x"] for r in results_12])

# Split into hurt and helped
for label, mask_fn in [("HURT (bottom 20%)", lambda d: d < np.percentile(d, 20)),
                        ("NEUTRAL (20-80%)", lambda d: (d >= np.percentile(d, 20)) & (d <= np.percentile(d, 80))),
                        ("HELPED (top 20%)", lambda d: d > np.percentile(d, 80))]:
    mask = mask_fn(deltas_1x)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        continue

    # Characteristics
    word_counts = [results_12[i]["word_count"] for i in indices]
    bare_nlls = [results_12[i]["bare"] for i in indices]
    bins = [results_12[i]["length_bin"] for i in indices]
    ans_positions = []
    for i in indices:
        p = find_answer_position(samples_12[i]["passage"], samples_12[i]["answer"])
        if p is not None:
            ans_positions.append(p)

    bin_counts = defaultdict(int)
    for b in bins:
        bin_counts[b] += 1

    print(f"\n  {label} (n={len(indices)}):")
    print(f"    Word count:  mean={np.mean(word_counts):.0f}  "
          f"median={np.median(word_counts):.0f}")
    print(f"    Bare NLL:    mean={np.mean(bare_nlls):.3f}  "
          f"median={np.median(bare_nlls):.3f}")
    if ans_positions:
        print(f"    Answer pos:  mean={np.mean(ans_positions):.3f}  "
              f"median={np.median(ans_positions):.3f}")
    print(f"    Length bins:  " + "  ".join(
        f"{b}={bin_counts[b]}" for b in ["short", "medium", "long", "very_long"]))

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Cross-Condition Consistency
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 3: CROSS-CONDITION CONSISTENCY")
print("=" * 70)
print("Are the same samples hurt across different conditions?")
print("-" * 65)

# Compute deltas for all conditions
all_deltas = {}
for cond in conditions_12:
    all_deltas[cond] = np.array([r["bare"] - r[cond] for r in results_12])

# Pairwise correlation of deltas
print("\nPearson correlation of per-sample deltas between conditions:")
for i, c1 in enumerate(conditions_12):
    for j, c2 in enumerate(conditions_12):
        if j > i:
            r = np.corrcoef(all_deltas[c1], all_deltas[c2])[0, 1]
            print(f"  {c1:20s} × {c2:20s}: r = {r:+.3f}")

# Overlap of hurt samples (bottom 20%)
print("\nOverlap of most-hurt samples (bottom 20%) across conditions:")
hurt_sets = {}
n_bottom = max(1, len(results_12) // 5)
for cond in conditions_12:
    sorted_idx = np.argsort(all_deltas[cond])
    hurt_sets[cond] = set(sorted_idx[:n_bottom])

for i, c1 in enumerate(conditions_12):
    for j, c2 in enumerate(conditions_12):
        if j > i:
            overlap = len(hurt_sets[c1] & hurt_sets[c2])
            pct = 100 * overlap / n_bottom
            print(f"  {c1:20s} ∩ {c2:20s}: {overlap}/{n_bottom} ({pct:.0f}%)")

# "Universally hurt" samples (hurt by ALL conditions)
universal_hurt = hurt_sets[conditions_12[0]]
for cond in conditions_12[1:]:
    universal_hurt = universal_hurt & hurt_sets[cond]
print(f"\n  Hurt by ALL 4 conditions: {len(universal_hurt)} samples")

if universal_hurt:
    print("  Characteristics of universally hurt samples:")
    for idx in sorted(universal_hurt):
        r = results_12[idx]
        s = samples_12[idx]
        apos = find_answer_position(s["passage"], s["answer"])
        apos_str = f"{apos:.2f}" if apos is not None else "N/A"
        print(f"    idx={r['idx']}  bare={r['bare']:.3f}  words={r['word_count']:5d}  "
              f"bin={r['length_bin']:>10s}  ans_pos={apos_str}")
        # Show delta for each condition
        for cond in conditions_12:
            d = r["bare"] - r[cond]
            print(f"      {cond:20s}: Δ={d:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Document Structure Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 4: DOCUMENT STRUCTURE ANALYSIS")
print("=" * 70)

# Analyze passage features that might predict priming benefit
print("\nPassage features correlated with priming benefit (static_fact, Exp 11):")
print("-" * 65)

static_deltas_11 = []
features = {"word_count": [], "answer_position": [], "bare_nll": [],
            "unique_word_ratio": [], "avg_word_length": [], "sentence_count": [],
            "answer_length_words": []}

for i, (r, s) in enumerate(zip(results_11, samples_11)):
    delta = r["bare"] - r["static_fact_trunc"]
    static_deltas_11.append(delta)

    words = s["passage"].split()
    unique_words = set(w.lower() for w in words)
    sentences = s["passage"].count('.') + s["passage"].count('?') + s["passage"].count('!')

    features["word_count"].append(len(words))
    features["answer_position"].append(answer_positions[i] if answer_positions[i] is not None else np.nan)
    features["bare_nll"].append(r["bare"])
    features["unique_word_ratio"].append(len(unique_words) / max(len(words), 1))
    features["avg_word_length"].append(np.mean([len(w) for w in words]) if words else 0)
    features["sentence_count"].append(max(sentences, 1))
    features["answer_length_words"].append(len(s["answer"].split()))

static_deltas_11 = np.array(static_deltas_11)

for feat_name, feat_values in features.items():
    feat_arr = np.array(feat_values)
    valid = ~np.isnan(feat_arr)
    if np.sum(valid) > 10:
        r = np.corrcoef(feat_arr[valid], static_deltas_11[valid])[0, 1]
        print(f"  {feat_name:25s}: r = {r:+.3f}  (n={np.sum(valid)})")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Amplification Sweet Spot Analysis (Exp 12)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 5: AMPLIFICATION SWEET SPOT")
print("=" * 70)
print("Per-sample: Is there an optimal amplification that depends on doc length?")
print("-" * 65)

# For each sample, find which amplification level is best
for length_bin in ["short", "medium", "long", "very_long"]:
    bin_results = [r for r in results_12 if r["length_bin"] == length_bin]
    if len(bin_results) < 5:
        continue

    # Count how often each condition wins
    wins = defaultdict(int)
    for r in bin_results:
        best_cond = None
        best_nll = float('inf')
        for cond in ["bare", "prefix_1x", "amplify_2x", "amplify_5x", "layers_0_15"]:
            if r[cond] < best_nll:
                best_nll = r[cond]
                best_cond = cond
        wins[best_cond] += 1

    total = len(bin_results)
    print(f"\n  {length_bin} (n={total}): best condition per sample:")
    for cond in ["bare", "prefix_1x", "amplify_2x", "amplify_5x", "layers_0_15"]:
        pct = 100 * wins[cond] / total
        bar = "█" * int(pct / 2)
        print(f"    {cond:20s}: {wins[cond]:3d} ({pct:4.1f}%) {bar}")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Win Rate vs Mean Delta Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 6: WIN RATE vs MEAN — ASYMMETRY ANALYSIS")
print("=" * 70)
print("Why do win rates look good (65%) while mean delta is negative?")
print("-" * 65)

for cond in ["prefix_1x", "amplify_2x", "layers_0_15"]:
    deltas = np.array([r["bare"] - r[cond] for r in results_12])

    helped = deltas > 0
    hurt = deltas < 0

    # Compute skewness of delta distribution
    from scipy.stats import skew, kurtosis
    sk = skew(deltas)
    kurt = kurtosis(deltas)

    # Percentile analysis
    p5 = np.percentile(deltas, 5)
    p25 = np.percentile(deltas, 25)
    p50 = np.percentile(deltas, 50)
    p75 = np.percentile(deltas, 75)
    p95 = np.percentile(deltas, 95)

    print(f"\n  {cond}:")
    print(f"    Mean: {np.mean(deltas):+.4f}  Median: {p50:+.4f}")
    print(f"    Skewness: {sk:+.3f}  Kurtosis: {kurt:+.3f}")
    print(f"    P5={p5:+.4f}  P25={p25:+.4f}  P50={p50:+.4f}  P75={p75:+.4f}  P95={p95:+.4f}")
    print(f"    Win rate: {100*np.mean(helped):.1f}%")
    print(f"    Mean when helped: {np.mean(deltas[helped]):+.4f}")
    print(f"    Mean when hurt:   {np.mean(deltas[hurt]):+.4f}")
    print(f"    Magnitude ratio (hurt/helped): {abs(np.mean(deltas[hurt])/np.mean(deltas[helped])):.2f}x")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: What's Special About the "Long" Bin Where Amplify Works?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("ANALYSIS 7: WHY DOES AMPLIFY WORK ON 'LONG' BUT NOT 'VERY_LONG'?")
print("=" * 70)

for length_bin in ["long", "very_long"]:
    bin_results = [r for r in results_12 if r["length_bin"] == length_bin]
    bin_samples = [s for s, r in zip(samples_12, results_12) if r["length_bin"] == length_bin]

    if len(bin_results) < 5:
        continue

    print(f"\n  {length_bin} (n={len(bin_results)}):")

    # Compare amplify_2x helped vs hurt in this bin
    deltas = np.array([r["bare"] - r["amplify_2x"] for r in bin_results])
    helped_idx = np.where(deltas > 0)[0]
    hurt_idx = np.where(deltas < 0)[0]

    print(f"    amplify_2x: {len(helped_idx)} helped, {len(hurt_idx)} hurt")

    # Characteristics of helped vs hurt
    for label, idx_set in [("HELPED", helped_idx), ("HURT", hurt_idx)]:
        if len(idx_set) < 3:
            continue
        wc = [bin_results[i]["word_count"] for i in idx_set]
        bare = [bin_results[i]["bare"] for i in idx_set]
        positions = []
        for i in idx_set:
            p = find_answer_position(bin_samples[i]["passage"], bin_samples[i]["answer"])
            if p is not None:
                positions.append(p)

        print(f"    {label}: word_count={np.mean(wc):.0f}±{np.std(wc):.0f}  "
              f"bare_nll={np.mean(bare):.3f}±{np.std(bare):.3f}  "
              f"ans_pos={np.mean(positions):.3f}" if positions else f"    {label}: word_count={np.mean(wc):.0f}  bare_nll={np.mean(bare):.3f}")

print("\n\n" + "=" * 70)
print("DIAGNOSTIC ANALYSIS COMPLETE")
print("=" * 70)
