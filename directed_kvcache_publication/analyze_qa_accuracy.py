#!/usr/bin/env python3
"""exp35 downstream QA-accuracy analysis (§7.2). Generation-based EM/F1 is confounded by answer
verbosity (interventions change generation length); we therefore also report a verbosity-robust
'answer-contained' recall (does the generation contain the gold span?) plus mean generation length.
Correctness conclusions rest on ANSWER-CONTAINED (robust); EM/F1 are reported for completeness.
Run: cd <repo> && PYTHONPATH="../directed_kvcache_v4:." python3 analyze_qa_accuracy.py"""
import json, re, string, numpy as np
from pathlib import Path
RES = Path("results/exp35_qa_accuracy")

def norm(x):
    x = x.lower(); x = "".join(c for c in x if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", x).split())
def contains(pred, golds): return max(int(len(norm(g)) > 0 and norm(g) in norm(pred)) for g in golds)
def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float); idx = rng.randint(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5]); return d.mean(), lo, hi
def f(t): return f"{100*t[0]:+.1f}[{100*t[1]:+.1f},{100*t[2]:+.1f}]" + ("*" if (t[1] > 0 or t[2] < 0) else " ")
CONDS = ["bare", "prime_full", "sel_plain", "sel_primed"]

print("="*94)
print("exp35 DOWNSTREAM QA-ACCURACY (SQuAD, N=200). CONTAINS = verbosity-robust answer recall (primary).")
print("  conditioning = prime_full - bare ; selection = sel_plain - bare ; COND|sel = sel_primed - sel_plain")
print("="*94)
for m in ["gemma3_12b", "qwen25_7b", "qwen25_1_5b"]:
    p = RES / m / "results.json"
    if not p.exists(): print(f"\n{m}: (missing)"); continue
    S = json.loads(p.read_text())["samples"]
    for s in S:
        for c in CONDS:
            s[f"{c}__contains"] = contains(s[f"{c}__pred"], s["golds"])
            s[f"{c}__len"] = len(s[f"{c}__pred"].split())
    A = lambda c, k: np.mean([s[f"{c}__{k}"] for s in S])
    print(f"\n## {m} (n={len(S)})")
    print(f"  gen length (words):  " + "  ".join(f"{c}={A(c,'len'):.1f}" for c in CONDS))
    for k in ["contains", "f1", "em"]:
        tag = " <- primary (correctness)" if k == "contains" else ""
        print(f"  {k.upper():9s} %:  " + "  ".join(f"{c}={100*A(c,k):.1f}" for c in CONDS) + tag)
        print(f"      conditioning={f(boot([s[f'prime_full__{k}']-s[f'bare__{k}'] for s in S]))}  "
              f"selection={f(boot([s[f'sel_plain__{k}']-s[f'bare__{k}'] for s in S]))}  "
              f"COND|sel={f(boot([s[f'sel_primed__{k}']-s[f'sel_plain__{k}'] for s in S]))}")
print("\n" + "="*94)
print("Takeaway: SELECTION hurts answer correctness (CONTAINS) — large on Qwen (-24..-36pt), sig on Gemma")
print("(-8.5pt). CONDITIONING does NOT change correctness (CONTAINS n.s. all models); its EM/F1 shift is")
print("a verbosity/calibration effect, not whether the answer is found.")
