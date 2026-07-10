#!/usr/bin/env python3
"""Analyze the reranking shuffle test (§7 hardening): does Gemma's keyword-priming MRR benefit
survive TOKEN-ORDER shuffling of the keywords? tfidf_shuffled (exp34) is paired sample-for-sample
against bare/tfidf_plain in exp14c_highN (same script, same load_msmarco, same N -> aligned;
verified via relevant_idx). If shuffled ≈ plain (both beat bare), the rerank benefit is TOKEN
PRESENCE (consistent with §6.5); if shuffled -> bare, it was word order.
Run: cd <repo> && PYTHONPATH="../directed_kvcache_v4:." python3 analyze_rerank_shuffle.py"""
import json, numpy as np
from pathlib import Path
RES = Path("results")

def mrr_vec(samples, cond):
    out = []
    for r in samples:
        ri = r["relevant_idx"]; nl = r[f"{cond}__q"]
        out.append(1.0 / (1 + sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])))
    return np.array(out)

def boot_diff(a, b, n=4000, seed=0):
    d = a - b; rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi

def f(t): return f"{t[0]:+.4f}[{t[1]:+.4f},{t[2]:+.4f}]" + ("*" if (t[1] > 0 or t[2] < 0) else " ")

print("="*88)
print("RERANK SHUFFLE (§7): does the keyword-priming MRR benefit survive token-order shuffling?")
print("  plain = passage tfidf keywords; shuffled = same keyword TOKENS, order scrambled")
print("="*88)
for m in ["gemma3_4b", "gemma3_12b", "gemma3_27b"]:
    base = RES / f"exp14c_highN/{m}/results.json"
    shuf = RES / f"exp34_rerank_shuffle/{m}/results.json"
    if not (base.exists() and shuf.exists()):
        print(f"  {m:11s} (pending: base={base.exists()} shuf={shuf.exists()})"); continue
    B = json.loads(base.read_text())["samples"]; S = json.loads(shuf.read_text())["samples"]
    n = min(len(B), len(S)); B, S = B[:n], S[:n]
    # alignment check via relevant_idx
    mism = sum(1 for i in range(n) if B[i]["relevant_idx"] != S[i]["relevant_idx"])
    bare = mrr_vec(B, "bare"); plain = mrr_vec(B, "tfidf_plain"); shufl = mrr_vec(S, "tfidf_shuffled")
    print(f"\n  {m}  (n={n}, alignment mismatches={mism}{'  !! MISALIGNED' if mism else ''})")
    print(f"    MRR: bare={bare.mean():.4f}  plain={plain.mean():.4f}  shuffled={shufl.mean():.4f}")
    print(f"    plain  - bare     = {f(boot_diff(plain, bare))}   (known keyword benefit)")
    print(f"    shuffled - bare   = {f(boot_diff(shufl, bare))}   (does shuffle preserve it?)")
    print(f"    shuffled - plain  = {f(boot_diff(shufl, plain))}   (~0 => TOKEN PRESENCE; <0 => order mattered)")
print("\n" + "="*88)
