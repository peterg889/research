#!/usr/bin/env python3
"""Analyze needle positional rescue. KEY questions:
 (1) Does bare show lost-in-the-middle (NLL worse at mid positions)?
 (2) Does extract-priming reduce the needle's answer NLL, and does the benefit
     CONCENTRATE at the hard middle positions (= positional rescue)?
Lower NLL = better. Paired Δ(extract-bare) per position; bootstrap 95% CI."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp15_needle"
MODELS = ["qwen25_1_5b", "qwen25_7b", "gemma3_12b"]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
RNG = np.random.RandomState(0)

def boot(diff, n=4000):
    diff = np.asarray(diff, float)
    idx = RNG.randint(0, len(diff), (n, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5]); return diff.mean(), lo, hi

def main():
    print("NEEDLE POSITIONAL RESCUE — needle answer NLL by position (lower=better)")
    print("Δ=extract-bare (negative=priming helps). * = CI excludes 0.\n")
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m}: no data\n"); continue
        S = json.loads(f.read_text())["samples"]
        if len(S) < 10: print(f"{m}: only {len(S)}\n"); continue
        print(f"{m}  (n={len(S)})")
        bare_by_p, ext_by_p = {}, {}
        for p in POSITIONS:
            b = np.array([r[f"p{p}__bare"] for r in S]); e = np.array([r[f"p{p}__extract"] for r in S])
            bare_by_p[p] = b; ext_by_p[p] = e
            d, lo, hi = boot(e - b); sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  p={p:<4} bare={b.mean():.3f}  extract={e.mean():.3f}  Δ={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        # lost-in-the-middle: is bare worse at mid (0.5) than at ends?
        ends = np.concatenate([bare_by_p[0.0], bare_by_p[1.0]]).mean()
        mid = bare_by_p[0.5].mean()
        print(f"  [bare lost-in-middle] ends(p0,p1)={ends:.3f}  mid(p0.5)={mid:.3f}  penalty={mid-ends:+.3f}")
        # does priming rescue the middle more than the ends?
        d_mid = (ext_by_p[0.5] - bare_by_p[0.5]).mean()
        d_ends = np.concatenate([ext_by_p[0.0]-bare_by_p[0.0], ext_by_p[1.0]-bare_by_p[1.0]]).mean()
        print(f"  [rescue concentration] Δmid={d_mid:+.3f}  Δends={d_ends:+.3f}  "
              f"(more-negative-at-mid => positional rescue)\n")

if __name__ == "__main__":
    main()
