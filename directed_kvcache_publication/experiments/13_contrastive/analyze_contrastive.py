#!/usr/bin/env python3
"""Analyze contrastive/distinctive reranking. KEY question: does distinctive priming
beat BOTH bare and generic, especially on the CAPABLE models (Qwen 7B, Gemma 12B)
where generic priming HURT in exp13? Paired bootstrap CIs on MRR / R@1."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp14_contrastive"
MODELS = ["qwen25_1_5b", "qwen25_7b", "gemma3_12b", "gemma3_27b", "mistral_7b", "qwen25_14b"]
CONDS = ["bare", "generic", "distinctive_corpus", "distinctive_cand"]
RNG = np.random.RandomState(0)

def ranks(nlls, ri):
    return 1 + sum(1 for j, x in enumerate(nlls) if j != ri and x < nlls[ri])

def metrics(S, cond):
    rr, r1 = [], []
    for r in S:
        ri = r["relevant_idx"]; rk = ranks(r[f"{cond}__q"], ri)
        rr.append(1.0/rk); r1.append(int(rk == 1))
    return np.array(rr), np.array(r1)

def boot(diff, n=4000):
    diff = np.asarray(diff, float)
    idx = RNG.randint(0, len(diff), (n, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5]); return diff.mean(), lo, hi

def main():
    print("CONTRASTIVE RERANKING — does distinctive priming break the non-selectivity failure?")
    print("Paired Δ vs bare AND vs generic; bootstrap 95% CI. * = excludes 0.\n")
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m}: no data\n"); continue
        S = json.loads(f.read_text())["samples"]
        if len(S) < 20: print(f"{m}: only {len(S)}\n"); continue
        M = {c: metrics(S, c) for c in CONDS}
        print(f"{m}  (n={len(S)})")
        for c in CONDS:
            rr, r1 = M[c]; print(f"  {c:20s} MRR={rr.mean():.3f}  R@1={r1.mean():.3f}")
        print("  -- Δ vs BARE --")
        for c in CONDS[1:]:
            for nm, i in [("MRR", 0), ("R@1", 1)]:
                d, lo, hi = boot(M[c][i] - M["bare"][i]); sig = "*" if (lo > 0 or hi < 0) else " "
                print(f"     {c:20s} Δ{nm}={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print("  -- Δ vs GENERIC (does distinctiveness add value over non-selective priming?) --")
        for c in ["distinctive_corpus", "distinctive_cand"]:
            for nm, i in [("MRR", 0), ("R@1", 1)]:
                d, lo, hi = boot(M[c][i] - M["generic"][i]); sig = "*" if (lo > 0 or hi < 0) else " "
                print(f"     {c:20s} Δ{nm}={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print()

if __name__ == "__main__":
    main()
