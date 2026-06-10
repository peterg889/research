#!/usr/bin/env python3
"""Analyze exp14b ablations. The three audit questions:
 T1 (active ingredient): is dist_clean > tfidf_plain? (contrast vs plain keywords)
 T2 (cacheability):      does dist_clean retain dist_corpus's win over generic?
 T3 (hybrid):            does "Extract the key facts about: {terms}" beat everything,
                          and crucially does anything beat BARE?
Paired bootstrap 95% CIs on MRR/R@1 for all key contrasts."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp14b_ablations"
MODELS = ["gemma3_4b", "gemma3_12b", "qwen25_7b", "gemma3_4b_base"]
CONDS = ["bare", "generic", "tfidf_plain", "dist_corpus", "dist_clean", "hybrid"]
RNG = np.random.RandomState(0)

def ranks(nl, ri): return 1 + sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])
def met(S, c):
    rr = np.array([1.0/ranks(r[f"{c}__q"], r["relevant_idx"]) for r in S])
    r1 = np.array([float(ranks(r[f"{c}__q"], r["relevant_idx"]) == 1) for r in S])
    return rr, r1
def boot(d, n=4000):
    d = np.asarray(d, float); idx = RNG.randint(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5]); return d.mean(), lo, hi
def fmt(d, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{d:+.3f}[{lo:+.3f},{hi:+.3f}]{sig}"

def main():
    print("EXP14B ABLATIONS — audit questions T1 (contrast vs plain), T2 (cacheable), T3 (hybrid/vs-bare)\n")
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m}: no data\n"); continue
        S = json.loads(f.read_text())["samples"]
        have = [c for c in CONDS if f"{c}__q" in S[0]]
        if len(S) < 30: print(f"{m}: only {len(S)}\n"); continue
        M = {c: met(S, c) for c in have}
        print(f"{m}  (n={len(S)})")
        for c in have:
            print(f"  {c:12s} MRR={M[c][0].mean():.3f}  R@1={M[c][1].mean():.3f}")
        pairs = [
            ("T1 contrast ingredient", "dist_clean", "tfidf_plain"),
            ("T2 cacheable win      ", "dist_clean", "generic"),
            ("T2 leak size          ", "dist_corpus", "dist_clean"),
            ("T3 hybrid vs generic  ", "hybrid", "generic"),
            ("T3 hybrid vs dist_clean", "hybrid", "dist_clean"),
            ("KEY  dist_clean vs BARE", "dist_clean", "bare"),
            ("KEY  hybrid vs BARE    ", "hybrid", "bare"),
            ("KEY  tfidf_plain vs BARE", "tfidf_plain", "bare"),
        ]
        print("  -- contrasts (ΔMRR) --")
        for label, a, b in pairs:
            if a in M and b in M:
                d, lo, hi = boot(M[a][0] - M[b][0])
                print(f"   {label}: {fmt(d, lo, hi)}")
        print()

if __name__ == "__main__":
    main()
