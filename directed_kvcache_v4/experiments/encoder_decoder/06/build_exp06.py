#!/usr/bin/env python3
# Build Exp 06 notebook: Factoid vs Long-Answer Split (v4 Replication).
#
# v3 Exp 06 showed two populations:
#   - Factoid (<=5 word answers): 76% structural, 15% vocabulary, 9% semantics
#   - Long answers: >100% structural, negative semantics
#
# v4 Exp 01 has per-sample NLLs for all 8 conditions with answer text.
# This notebook loads that checkpoint and stratifies by answer length.
#
# Key question: does the oracle's remaining edge over surrogates (18% in Exp 04)
# concentrate in factoid QA? If so, content optimization has higher ROI for factoid.
#
# NO GPU needed — pure analysis of existing data.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 06: Factoid vs Long-Answer Split

## Motivation

v3 Exp 06 showed a two-population structure in the co-encoding benefit:
- **Factoid answers** (≤5 words): 76% structural, 15% vocabulary, 9% semantics
- **Long answers** (>5 words): >100% structural, negative semantics

In v4, the structural component collapsed from 85% to 35% on short documents. This
split may be dramatically different — factoid QA might show the strongest
content-dependent (oracle > surrogate) signal, since v3 showed factoid was the only
subpopulation where semantics mattered.

v4 Exp 04 showed that oracle retains an 18% edge over the best surrogate (kw10).
Does this edge concentrate in factoid answers?

## Method

Reuses Exp 01 checkpoint data (500 samples, 8 conditions). No new scoring needed.
Analysis stratified by answer word count:
- Factoid: ≤5 words
- Long: >5 words

Also explores finer bins (1-3w, 4-5w, 6-10w, 11-20w, 20+w) for gradient analysis.""")


# ===== Cell 2: Load Exp 01 data =====
code(r"""# Cell 2: Load Exp 01 checkpoint
import os
os.umask(0o000)

import json
import numpy as np
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

RESULTS_DIR = Path("../../../results/exp06")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Exp 01 checkpoint
exp01_ckpt = json.loads(Path("../../../results/exp01/checkpoint.json").read_text())
results = exp01_ckpt['results']
N = len(results)
print(f"Loaded {N} samples from Exp 01 checkpoint")

# Compute answer word counts
for r in results:
    r['answer_words'] = len(r['answer'].split())

answer_lens = np.array([r['answer_words'] for r in results])
print(f"\nAnswer length distribution:")
print(f"  mean={answer_lens.mean():.1f}, median={np.median(answer_lens):.0f}")
print(f"  min={answer_lens.min()}, max={answer_lens.max()}")

# Factoid vs long split
n_factoid = np.sum(answer_lens <= 5)
n_long = np.sum(answer_lens > 5)
print(f"\n  Factoid (≤5 words): {n_factoid} ({100*n_factoid/N:.0f}%)")
print(f"  Long (>5 words):    {n_long} ({100*n_long/N:.0f}%)")

# Finer bins
bins = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 999)]
print(f"\nFiner bins:")
for lo, hi in bins:
    n = np.sum((answer_lens >= lo) & (answer_lens <= hi))
    label = f"{lo}-{hi}w" if hi < 999 else f"{lo}+w"
    print(f"  {label:>8}: {n:>4} ({100*n/N:.0f}%)")
""")


# ===== Cell 3: Stratified results =====
code(r"""# Cell 3: Stratified results — factoid vs long
print("=" * 70)
print("FACTOID vs LONG-ANSWER ANALYSIS")
print("=" * 70)

# Extract NLL arrays
bare = np.array([r['nll_bare'] for r in results])
oracle_trunc = np.array([r['nll_oracle_trunc'] for r in results])
oracle_full = np.array([r['nll_oracle_full'] for r in results])
surr_template = np.array([r['nll_surr_template_trunc'] for r in results])
surr_doc = np.array([r['nll_surr_doc_trunc'] for r in results])
random_trunc = np.array([r['nll_random_trunc'] for r in results])
bare_nq = np.array([r['nll_bare_nq'] for r in results])
oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])
answer_lens = np.array([r['answer_words'] for r in results])

def analyze_subgroup(mask, label):
    n = mask.sum()
    if n < 10:
        return None
    b = bare[mask]
    ot = oracle_trunc[mask]
    sd = surr_doc[mask]
    rt = random_trunc[mask]
    bnq = bare_nq[mask]
    onq = oracle_nq[mask]

    d_oracle = cohens_d(b - ot)
    d_surr = cohens_d(b - sd)
    d_random = cohens_d(b - rt)
    d_oracle_nq = cohens_d(bnq - onq)

    _, p_oracle = stats.ttest_1samp(b - ot, 0)
    _, p_surr = stats.ttest_1samp(b - sd, 0)
    _, p_random = stats.ttest_1samp(b - rt, 0)

    sig_oracle = '***' if p_oracle < 0.001 else '**' if p_oracle < 0.01 else '*' if p_oracle < 0.05 else 'ns'
    sig_surr = '***' if p_surr < 0.001 else '**' if p_surr < 0.01 else '*' if p_surr < 0.05 else 'ns'
    sig_random = '***' if p_random < 0.001 else '**' if p_random < 0.01 else '*' if p_random < 0.05 else 'ns'

    struct_frac = d_random / d_oracle * 100 if d_oracle > 0 else float('inf')
    surr_pct = d_surr / d_oracle * 100 if d_oracle > 0 else float('inf')
    v4v3 = d_oracle / d_oracle_nq * 100 if d_oracle_nq > 0 else float('inf')

    return {
        'label': label, 'n': int(n),
        'd_oracle': d_oracle, 'sig_oracle': sig_oracle,
        'd_surr': d_surr, 'sig_surr': sig_surr,
        'd_random': d_random, 'sig_random': sig_random,
        'd_oracle_nq': d_oracle_nq,
        'struct_frac': struct_frac, 'surr_pct': surr_pct, 'v4v3': v4v3,
        'mean_bare_nll': float(b.mean()),
    }

# Binary split
mask_factoid = answer_lens <= 5
mask_long = answer_lens > 5

print(f"\n--- Binary split ---")
print(f"\n  {'Group':<12} {'N':>4} {'d_oracle':>10} {'sig':>5} {'d_surr':>10} {'sig':>5} "
      f"{'d_random':>10} {'sig':>5} {'Struct%':>8} {'Surr%':>6} {'v4/v3%':>7}")
print(f"  {'-'*92}")

all_results = {}
for label, mask in [('ALL', np.ones(N, dtype=bool)),
                     ('Factoid', mask_factoid),
                     ('Long', mask_long)]:
    res = analyze_subgroup(mask, label)
    if res:
        all_results[label] = res
        print(f"  {label:<12} {res['n']:>4} {res['d_oracle']:>+10.3f} {res['sig_oracle']:>5} "
              f"{res['d_surr']:>+10.3f} {res['sig_surr']:>5} "
              f"{res['d_random']:>+10.3f} {res['sig_random']:>5} "
              f"{res['struct_frac']:>7.0f}% {res['surr_pct']:>5.0f}% {res['v4v3']:>6.0f}%")

# Direct test: is enrichment DIFFERENT between factoid and long?
enrichment_factoid = (bare - oracle_trunc)[mask_factoid]
enrichment_long = (bare - oracle_trunc)[mask_long]
_, p_diff = stats.ttest_ind(enrichment_factoid, enrichment_long)
d_diff = (enrichment_factoid.mean() - enrichment_long.mean()) / np.sqrt(
    (enrichment_factoid.var() + enrichment_long.var()) / 2)
print(f"\n  Factoid vs Long enrichment difference: d={d_diff:+.3f}, p={p_diff:.3e}")
""")


# ===== Cell 4: Finer bins =====
code(r"""# Cell 4: Finer answer-length bins
print("=" * 70)
print("FINER ANSWER-LENGTH BINS")
print("=" * 70)

bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
bin_labels = ['1-2w', '3-5w', '6-10w', '11-20w', '21+w']

print(f"\n  {'Bin':<8} {'N':>4} {'bare NLL':>10} {'d_oracle':>10} {'d_surr':>10} "
      f"{'d_random':>10} {'Struct%':>8} {'v4/v3%':>7}")
print(f"  {'-'*72}")

bin_data = []
for (lo, hi), label in zip(bins, bin_labels):
    mask = (answer_lens >= lo) & (answer_lens <= hi)
    res = analyze_subgroup(mask, label)
    if res:
        bin_data.append(res)
        sf = f"{res['struct_frac']:.0f}%" if abs(res['struct_frac']) < 500 else "N/A"
        v4v3 = f"{res['v4v3']:.0f}%" if abs(res['v4v3']) < 500 else "N/A"
        print(f"  {label:<8} {res['n']:>4} {res['mean_bare_nll']:>10.3f} "
              f"{res['d_oracle']:>+10.3f} {res['d_surr']:>+10.3f} "
              f"{res['d_random']:>+10.3f} {sf:>8} {v4v3:>7}")

# Is there a gradient?
print(f"\n--- Gradient analysis ---")
if len(bin_data) >= 3:
    mid_lens = [1.5, 4, 8, 15, 30][:len(bin_data)]
    d_oracles = [b['d_oracle'] for b in bin_data]
    struct_fracs = [b['struct_frac'] for b in bin_data]

    r_d, p_d = stats.spearmanr(mid_lens, d_oracles)
    print(f"  Spearman(answer_len vs d_oracle): rho={r_d:+.3f}, p={p_d:.3e}")

    valid_sf = [(m, s) for m, s in zip(mid_lens, struct_fracs) if abs(s) < 500]
    if len(valid_sf) >= 3:
        r_sf, p_sf = stats.spearmanr([x[0] for x in valid_sf], [x[1] for x in valid_sf])
        print(f"  Spearman(answer_len vs struct%): rho={r_sf:+.3f}, p={p_sf:.3e}")
""")


# ===== Cell 5: Oracle vs surrogate gap by subgroup =====
code(r"""# Cell 5: Does the oracle-vs-surrogate gap concentrate in factoid QA?
print("=" * 70)
print("ORACLE vs SURROGATE GAP BY ANSWER TYPE")
print("=" * 70)

# The key question: oracle retains 18% edge over kw5 in aggregate.
# Does this edge come from factoid answers?

print(f"\n--- Oracle vs surr_doc (kw5) by answer type ---")
for label, mask in [('ALL', np.ones(N, dtype=bool)),
                     ('Factoid (<=5w)', mask_factoid),
                     ('Long (>5w)', mask_long)]:
    n = mask.sum()
    d_orc = cohens_d((bare - oracle_trunc)[mask])
    d_surr = cohens_d((bare - surr_doc)[mask])
    d_rand = cohens_d((bare - random_trunc)[mask])

    # Pairwise: oracle vs surr_doc
    diff_os = (surr_doc - oracle_trunc)[mask]  # positive = oracle is better
    d_os = cohens_d(diff_os)
    _, p_os = stats.ttest_1samp(diff_os, 0)
    sig_os = '***' if p_os < 0.001 else '**' if p_os < 0.01 else '*' if p_os < 0.05 else 'ns'

    # Pairwise: surr_doc vs random
    diff_sr = (random_trunc - surr_doc)[mask]  # positive = surr is better
    d_sr = cohens_d(diff_sr)
    _, p_sr = stats.ttest_1samp(diff_sr, 0)
    sig_sr = '***' if p_sr < 0.001 else '**' if p_sr < 0.01 else '*' if p_sr < 0.05 else 'ns'

    print(f"\n  {label} (N={n}):")
    print(f"    oracle d={d_orc:+.3f}, surr_doc d={d_surr:+.3f}, random d={d_rand:+.3f}")
    print(f"    oracle vs surr_doc: d={d_os:+.3f} ({sig_os})")
    print(f"    surr_doc vs random: d={d_sr:+.3f} ({sig_sr})")

    if d_orc > 0:
        content_pct = (d_orc - d_rand) / d_orc * 100
        surr_capture = d_surr / d_orc * 100
        print(f"    Content component: {content_pct:.0f}% of oracle")
        print(f"    Surrogate captures: {surr_capture:.0f}% of oracle")

# Template surrogate by subgroup (failed in aggregate, maybe works for factoid?)
print(f"\n--- Template surrogate by answer type ---")
for label, mask in [('Factoid (<=5w)', mask_factoid),
                     ('Long (>5w)', mask_long)]:
    d_tmpl = cohens_d((bare - surr_template)[mask])
    _, p_tmpl = stats.ttest_1samp((bare - surr_template)[mask], 0)
    sig_tmpl = '***' if p_tmpl < 0.001 else '**' if p_tmpl < 0.01 else '*' if p_tmpl < 0.05 else 'ns'
    print(f"  {label}: template d={d_tmpl:+.3f} ({sig_tmpl})")
""")


# ===== Cell 6: v3 comparison + verdict + save =====
code(r"""# Cell 6: Comparison with v3 Exp 06 + verdict + save
print("=" * 70)
print("COMPARISON WITH v3 Exp 06")
print("=" * 70)

# v3 Exp 06 reference values
print(f"\n  v3 Exp 06 findings (no query in decoder):")
print(f"    Factoid: struct=76%, vocab=15%, semantics=9%, oracle d=0.767")
print(f"    Long:    struct=>100%, semantics<0, oracle d=0.376")

# v4 values
res_f = all_results.get('Factoid', {})
res_l = all_results.get('Long', {})

print(f"\n  v4 (query in decoder):")
if res_f:
    print(f"    Factoid: oracle d={res_f['d_oracle']:+.3f}, struct={res_f['struct_frac']:.0f}%, "
          f"surr={res_f['surr_pct']:.0f}%, v4/v3={res_f['v4v3']:.0f}%")
if res_l:
    sf = f"{res_l['struct_frac']:.0f}%" if abs(res_l['struct_frac']) < 500 else "N/A"
    print(f"    Long:    oracle d={res_l['d_oracle']:+.3f}, struct={sf}, "
          f"surr={res_l['surr_pct']:.0f}%, v4/v3={res_l['v4v3']:.0f}%")

print(f"\n--- Verdict ---")
if res_f and res_l:
    if res_f['d_oracle'] > res_l['d_oracle'] + 0.05:
        print(f"  Factoid QA shows STRONGER enrichment than long answers.")
        print(f"  This is consistent with v3 (factoid had 2x oracle headroom).")
    elif res_l['d_oracle'] > res_f['d_oracle'] + 0.05:
        print(f"  Long answers show STRONGER enrichment than factoid.")
        print(f"  This reverses the v3 pattern.")
    else:
        print(f"  Similar enrichment across answer types.")

    if res_f.get('struct_frac', 0) < res_l.get('struct_frac', 0):
        print(f"  Factoid has LOWER structural fraction — content matters more for factoid.")
    else:
        print(f"  Factoid has HIGHER structural fraction — same pattern as v3.")

# Save results
final_results = {
    'experiment': 'v4_exp06_factoid_split',
    'source': 'exp01_checkpoint (reanalysis, no new scoring)',
    'n_samples': N,
    'subgroups': {k: v for k, v in all_results.items()},
    'bin_data': bin_data,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/encoder_decoder/06/06_factoid_split.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
