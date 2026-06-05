#!/usr/bin/env python3
"""Rigorous analysis of the BoolQ sharpening experiment.

From the stored per-sample class logits, computes for each condition and the paired
Δ(cond - bare) with bootstrap 95% CIs:
  - accuracy, balanced accuracy
  - calibration: ECE (15 bins), Brier
  - selective prediction: accuracy@50% coverage, AURC (area under risk-coverage)
  - logit margin = class_logit(gold) - class_logit(other)
  - PRIOR-SHIFT control: Δmargin split by gold class (yes vs no). Both positive =>
    real discrimination sharpening; one up / one down => mere label-prior shift.
  - contextual-calibration-adjusted accuracy (divide out the content-free prior).
"""
import json, math
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp10_boolq"
MODELS = ["qwen25_1_5b", "qwen25_7b", "gemma3_12b", "ministral_8b"]
CONDS = ["extract", "comprehend", "keywords"]
RNG = np.random.RandomState(0)


def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def load(m):
    f = RES / m / "results.json"
    if not f.exists(): return None, None
    d = json.loads(f.read_text())
    return d["samples"], d.get("prior", {})

def pvec(S, cond):
    """Return (p_yes array, gold array) for a condition."""
    p = np.array([sigmoid(s[f"{cond}__yes"] - s[f"{cond}__no"]) for s in S if f"{cond}__yes" in s])
    g = np.array([s["gold"] for s in S if f"{cond}__yes" in s], dtype=float)
    return p, g

def margins(S, cond):
    """gold-class margin = logit(gold) - logit(other), and gold labels."""
    mg = np.array([(s[f"{cond}__yes"] - s[f"{cond}__no"]) if s["gold"] else (s[f"{cond}__no"] - s[f"{cond}__yes"])
                   for s in S if f"{cond}__yes" in s])
    g = np.array([s["gold"] for s in S if f"{cond}__yes" in s], dtype=float)
    return mg, g

def accuracy(p, g): return float(np.mean((p > 0.5) == (g > 0.5)))
def balanced_acc(p, g):
    pred = (p > 0.5).astype(float)
    tpr = np.mean(pred[g == 1] == 1) if (g == 1).any() else 0
    tnr = np.mean(pred[g == 0] == 0) if (g == 0).any() else 0
    return float((tpr + tnr) / 2)
def brier(p, g): return float(np.mean((p - g) ** 2))
def ece(p, g, nb=15):
    conf = np.maximum(p, 1 - p); correct = ((p > 0.5) == (g > 0.5)).astype(float)
    e = 0.0; N = len(p)
    for b in range(nb):
        lo, hi = b/nb, (b+1)/nb
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum()/N
    return float(e)
def aurc(p, g):
    """Area under risk-coverage curve (lower=better). Risk = error on most-confident fraction."""
    conf = np.maximum(p, 1 - p); correct = ((p > 0.5) == (g > 0.5)).astype(float)
    order = np.argsort(-conf); c = correct[order]; risks = []
    cum = 0
    for i in range(len(c)):
        cum += c[i]; risks.append(1 - cum/(i+1))
    return float(np.mean(risks))
def acc_at_cov(p, g, cov=0.5):
    conf = np.maximum(p, 1 - p); correct = ((p > 0.5) == (g > 0.5)).astype(float)
    order = np.argsort(-conf); k = max(1, int(cov*len(p)))
    return float(correct[order][:k].mean())

def boot_diff(fn, Sb, Sc, n=3000):
    """Bootstrap CI for fn(cond)-fn(bare), paired by sample index."""
    pb, gb = Sb; pc, gc = Sc
    base = fn(pc, gc) - fn(pb, gb)
    N = len(pb); vals = []
    for _ in range(n):
        idx = RNG.randint(0, N, N)
        vals.append(fn(pc[idx], gc[idx]) - fn(pb[idx], gb[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, lo, hi

def main():
    print("=" * 96)
    print("BoolQ SHARPENING ANALYSIS — paired Δ(cond - bare), bootstrap 95% CI. * = excludes 0.")
    print("=" * 96)
    for m in MODELS:
        S, prior = load(m)
        if not S or len(S) < 50:
            print(f"\n{m}: incomplete ({0 if not S else len(S)})"); continue
        pb, gb = pvec(S, "bare")
        print(f"\n{'='*96}\n{m}  (n={len(pb)})  prior P(yes)={ {k:round(v,3) for k,v in prior.items()} }")
        print(f"  bare: acc={accuracy(pb,gb):.3f}  bal_acc={balanced_acc(pb,gb):.3f}  "
              f"ECE={ece(pb,gb):.3f}  Brier={brier(pb,gb):.3f}  acc@50%={acc_at_cov(pb,gb):.3f}  AURC={aurc(pb,gb):.3f}")
        print(f"  {'cond':<11s} | {'Δacc':>16s} | {'Δbal_acc':>16s} | {'Δacc@50%':>16s} | {'ΔECE':>14s} | {'ΔBrier':>14s}")
        for c in CONDS:
            pc, gc = pvec(S, c)
            if len(pc) != len(pb): continue
            row = f"  {c:<11s}"
            for fn, lo_better in [(accuracy, False), (balanced_acc, False), (lambda p,g: acc_at_cov(p,g), False)]:
                b, lo, hi = boot_diff(fn, (pb, gb), (pc, gc)); sig = "*" if (lo > 0 or hi < 0) else " "
                row += f" | {b:+.3f}[{lo:+.2f},{hi:+.2f}]{sig}"
            for fn in [ece, brier]:
                b, lo, hi = boot_diff(fn, (pb, gb), (pc, gc)); sig = "*" if (lo > 0 or hi < 0) else " "
                row += f" | {b:+.3f}[{lo:+.2f},{hi:+.2f}]{sig}"
            print(row)
        # PRIOR-SHIFT control: Δmargin split by gold class
        print(f"  prior-shift control — Δmargin (cond-bare) by gold class (both>0 = real sharpening):")
        mb, gmb = margins(S, "bare")
        for c in CONDS:
            mc, _ = margins(S, c)
            if len(mc) != len(mb): continue
            dy = (mc - mb)[gmb == 1].mean(); dn = (mc - mb)[gmb == 0].mean()
            verdict = "REAL SHARPENING" if (dy > 0 and dn > 0) else ("prior shift" if dy*dn < 0 else "mixed")
            print(f"    {c:<11s} gold=yes Δmgn={dy:+.3f}  gold=no Δmgn={dn:+.3f}   -> {verdict}")
        # contextual-calibration-adjusted accuracy
        if prior:
            print(f"  contextual-calibration-adjusted accuracy:")
            for c in ["bare"] + CONDS:
                pc, gc = pvec(S, c)
                pcf = prior.get(c, prior.get("bare", 0.5)) if c != "keywords" else prior.get("bare", 0.5)
                # divide out prior, renormalize
                num_y = pc / max(pcf, 1e-3); num_n = (1 - pc) / max(1 - pcf, 1e-3)
                p_cal = num_y / (num_y + num_n)
                print(f"    {c:<11s} raw_acc={accuracy(pc,gc):.3f}  cal_acc={accuracy(p_cal,gc):.3f}")


if __name__ == "__main__":
    main()
