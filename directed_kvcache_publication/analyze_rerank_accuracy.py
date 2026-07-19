#!/usr/bin/env python3
"""Reranking through an ACCURACY lens (top-1 / hit@k), not just MRR. The §7 reranking result is
+0.036 MRR for keyword priming on Gemma-12B/27B; here we ask whether that translates to a hard
ranking-accuracy gain (does the relevant passage actually reach rank 1?). Computed from stored
per-passage query-NLLs: bare/generic/tfidf_plain (exp14c_highN, N=900) and tfidf_shuffled
(exp34_rerank_shuffle), sample-paired via relevant_idx.
Run: cd <repo> && PYTHONPATH="../directed_kvcache_v4:." python3 analyze_rerank_accuracy.py"""
import json, numpy as np
from pathlib import Path
RES = Path("results")

def rank_of_relevant(nlls, ri):
    """1 = relevant passage has the lowest NLL (best)."""
    return 1 + sum(1 for j, x in enumerate(nlls) if j != ri and x < nlls[ri])

def metrics(samples, cond):
    r = np.array([rank_of_relevant(s[f"{cond}__q"], s["relevant_idx"]) for s in samples])
    return {"top1": (r == 1).astype(float), "hit3": (r <= 3).astype(float), "mrr": 1.0 / r}

def boot_diff(a, b, n=4000, seed=0):
    d = a - b; rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
def f(t): return f"{100*t[0]:+.2f}[{100*t[1]:+.2f},{100*t[2]:+.2f}]" + ("*" if (t[1] > 0 or t[2] < 0) else " ")

print("="*92)
print("RERANKING ACCURACY (top-1 / hit@3), vs MRR. keyword priming (tfidf) vs bare, MS MARCO N=900.")
print("="*92)
for m in ["gemma3_4b", "gemma3_12b", "gemma3_27b", "qwen25_7b", "qwen25_14b", "mistral_7b"]:
    base = RES / f"exp14c_highN/{m}/results.json"
    if not base.exists(): continue
    B = json.loads(base.read_text())["samples"]
    sh = RES / f"exp34_rerank_shuffle/{m}/results.json"
    S = json.loads(sh.read_text())["samples"] if sh.exists() else None
    if S: n = min(len(B), len(S)); B, S = B[:n], S[:n]
    M = {c: metrics(B, c) for c in ["bare", "generic", "tfidf_plain"]}
    if S: M["tfidf_shuffled"] = metrics(S, "tfidf_shuffled")
    print(f"\n## {m}  (n={len(B)})")
    for met in ["top1", "hit3", "mrr"]:
        row = "  ".join(f"{c.split('_')[-1] if '_' in c else c}={100*M[c][met].mean():.1f}" for c in M)
        print(f"  {met.upper():5s}: {row}")
    print(f"    keyword−bare:  top1={f(boot_diff(M['tfidf_plain']['top1'], M['bare']['top1']))}  "
          f"hit3={f(boot_diff(M['tfidf_plain']['hit3'], M['bare']['hit3']))}  "
          f"mrr={f(boot_diff(M['tfidf_plain']['mrr'], M['bare']['mrr']))}")
    if S:
        print(f"    shuffled−bare: top1={f(boot_diff(M['tfidf_shuffled']['top1'], M['bare']['top1']))}  "
              f"(keyword benefit survives token shuffling → token presence, §6.5)")
print("\n" + "="*92)
