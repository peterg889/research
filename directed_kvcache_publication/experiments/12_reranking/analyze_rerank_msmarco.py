#!/usr/bin/env python3
"""Analyze MS MARCO reranking: does cache priming improve a real RAG-rerank metric,
and does the selective recipe work here (where a serve-time confidence signal —
the candidate score gap — actually exists)?"""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp13_rerank_msmarco"
MODELS = ["qwen25_1_5b", "qwen25_7b", "gemma3_12b"]
RNG = np.random.RandomState(0)

def ranks(nlls, ri):
    return 1 + sum(1 for j, x in enumerate(nlls) if j != ri and x < nlls[ri])

def metrics(S, cond):
    rr, r1, r3 = [], [], []
    for r in S:
        ri = r["relevant_idx"]; rk = ranks(r[f"{cond}__q"], ri)
        rr.append(1.0/rk); r1.append(int(rk == 1)); r3.append(int(rk <= 3))
    return np.array(rr), np.array(r1), np.array(r3)

def boot(diff, n=4000):
    diff = np.asarray(diff, float)
    idx = RNG.randint(0, len(diff), (n, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5]); return diff.mean(), lo, hi

def serve_conf(nlls, ri):
    """Serve-time confidence = gap between best and 2nd-best candidate (NLL), no labels."""
    srt = sorted(nlls); return srt[1] - srt[0]

def main():
    print("MS MARCO RERANKING — paired Δ(extract-bare), bootstrap 95% CI. * excludes 0.\n")
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m}: no data"); continue
        S = json.loads(f.read_text())["samples"]
        if len(S) < 30: print(f"{m}: only {len(S)}"); continue
        brr, br1, br3 = metrics(S, "bare"); err, er1, er3 = metrics(S, "extract")
        print(f"{m}  (n={len(S)})")
        print(f"  bare:    MRR={brr.mean():.3f}  R@1={br1.mean():.3f}  R@3={br3.mean():.3f}")
        print(f"  extract: MRR={err.mean():.3f}  R@1={er1.mean():.3f}  R@3={er3.mean():.3f}")
        for name, b, e in [("ΔMRR", brr, err), ("ΔR@1", br1, er1), ("ΔR@3", br3, er3)]:
            d, lo, hi = boot(e - b); sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {name}={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        # boundary concentration with a DEPLOYABLE serve-time signal (candidate score gap)
        conf = np.array([serve_conf(r["bare__q"], r["relevant_idx"]) for r in S])
        dr1 = er1 - br1
        q = np.quantile(conf, [1/3, 2/3])
        print(f"    ΔR@1 by serve-confidence (candidate gap): uncertain={dr1[conf<=q[0]].mean():+.3f}  "
              f"mid={dr1[(conf>q[0])&(conf<=q[1])].mean():+.3f}  confident={dr1[conf>q[2-1]].mean():+.3f}")
        # selective deployment by serve-confidence (prime only low-gap/uncertain queries)
        order = np.argsort(conf)  # uncertain first
        n = len(S); best = (br1.mean(), 0)
        for K in range(0, 101, 10):
            k = int(K/100*n); prime = set(order[:k].tolist())
            acc = np.mean([er1[i] if i in prime else br1[i] for i in range(n)])
            if acc > best[0]: best = (acc, K)
        print(f"    selective R@1 (prime uncertain bottom-K%): {best[0]:.3f} @ {best[1]}% "
              f"(+{best[0]-br1.mean():.3f} vs bare; prime-all +{er1.mean()-br1.mean():+.3f})")
        print()

if __name__ == "__main__":
    main()
