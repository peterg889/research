#!/usr/bin/env python3
"""WS2: does the contrastive-priming win replicate on HotpotQA (2nd benchmark)?
distinctive_corpus MRR/R@1 vs bare AND vs generic; paired bootstrap 95% CI."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp16_hotpot"
MODELS = ["gemma3_4b", "gemma3_12b", "gemma3_27b", "qwen25_7b"]
RNG = np.random.RandomState(0)

def ranks(nl, ri): return 1 + sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])
def met(S, c):
    rr = np.array([1.0/ranks(r[f"{c}__q"], r["relevant_idx"]) for r in S])
    r1 = np.array([int(ranks(r[f"{c}__q"], r["relevant_idx"]) == 1) for r in S])
    return rr, r1
def boot(d, n=4000):
    d = np.asarray(d, float); idx = RNG.randint(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5]); return d.mean(), lo, hi

def main():
    print("HOTPOTQA RERANKING (2nd benchmark) — distinctive_corpus (cacheable) effect\n")
    print(f"{'model':14s} {'bare':>6s} {'gen':>6s} {'dcorp':>6s} | {'ΔMRR vs bare':>18s} {'ΔMRR vs generic':>18s}")
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m:14s} (no data)"); continue
        S = json.loads(f.read_text())["samples"]
        if len(S) < 30: print(f"{m:14s} (only {len(S)})"); continue
        b, _ = met(S, "bare"); g, _ = met(S, "generic"); d, _ = met(S, "distinctive_corpus")
        db, lb, hb = boot(d - b); dg, lg, hg = boot(d - g)
        sb = "*" if (lb > 0 or hb < 0) else " "; sg = "*" if (lg > 0 or hg < 0) else " "
        print(f"{m:14s} {b.mean():>6.3f} {g.mean():>6.3f} {d.mean():>6.3f} | "
              f"{db:+.3f}[{lb:+.3f},{hb:+.3f}]{sb} {dg:+.3f}[{lg:+.3f},{hg:+.3f}]{sg}  (n={len(S)})")
    print("\nPrediction: dcorp beats generic* on Gemma (primable); qwen25_7b control shows none.")

if __name__ == "__main__":
    main()
