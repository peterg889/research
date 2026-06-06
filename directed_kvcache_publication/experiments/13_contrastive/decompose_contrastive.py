#!/usr/bin/env python3
"""WHY does distinctive priming beat generic on Gemma 12B? Mechanism = SELECTIVE
amplification: distinctive priming should lower the RELEVANT passage's query-NLL
more than it lowers the NEGATIVES' query-NLL (so the relevant rises in rank). Generic
priming lowers both ~equally (non-selective -> no rank gain, even hurts).

For each condition we report Δ(query-NLL vs bare), split by relevant vs negative.
Selectivity = Δ_neg - Δ_rel  (how much MORE the relevant drops than the negatives;
positive = selective toward the relevant passage). Lower query-NLL = better rank."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp14_contrastive"
MODELS = ["qwen25_1_5b", "qwen25_7b", "gemma3_12b", "gemma3_27b", "mistral_7b", "qwen25_14b"]
CONDS = ["generic", "distinctive_corpus", "distinctive_cand"]
RNG = np.random.RandomState(0)

def boot(x, n=4000):
    x = np.asarray(x, float); idx = RNG.randint(0, len(x), (n, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5]); return x.mean(), lo, hi

def main():
    print("SELECTIVITY DECOMPOSITION — Δquery-NLL vs bare (negative = passage predicts query better)")
    print("selectivity = Δneg - Δrel  (>0 means relevant drops MORE than negatives = selective)\n")
    rows = []
    for m in MODELS:
        f = RES / m / "results.json"
        if not f.exists(): print(f"{m}: no data\n"); continue
        S = json.loads(f.read_text())["samples"]
        print(f"{m}  (n={len(S)})")
        # primability index = mean |Δquery-NLL| caused by GENERIC priming (how much
        # priming moves anything at all on this model, regardless of direction)
        prim = []
        for r in S:
            b = r["bare__q"]; g = r["generic__q"]
            prim += [abs(g[j]-b[j]) for j in range(len(b)) if np.isfinite(b[j]) and np.isfinite(g[j])]
        primability = float(np.mean(prim))
        bare = [(r["relevant_idx"], r["bare__q"]) for r in S]
        sel_dc = None
        for c in CONDS:
            drel, dneg, sel = [], [], []
            for r in S:
                ri = r["relevant_idx"]; b = r["bare__q"]; x = r[f"{c}__q"]
                dr = x[ri] - b[ri]
                negs = [x[j] - b[j] for j in range(len(b)) if j != ri and np.isfinite(b[j]) and np.isfinite(x[j])]
                if not negs or not np.isfinite(dr): continue
                dn = float(np.mean(negs)); drel.append(dr); dneg.append(dn); sel.append(dn - dr)
            mr, _, _ = boot(drel); mn, _, _ = boot(dneg); ms, lo, hi = boot(sel)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {c:20s} Δrel={mr:+.3f}  Δneg={mn:+.3f}  selectivity={ms:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            if c == "distinctive_corpus": sel_dc = ms
        print(f"  >>> primability(|Δ| from generic)={primability:.3f}   distinctive_corpus selectivity={sel_dc:+.3f}")
        rows.append((m, primability, sel_dc))
        print()
    print("PRIMABILITY x SELECTIVITY SUMMARY (does contrastive selectivity require a primable model?)")
    print(f"  {'model':14s} {'primability':>12s} {'dist_corp_selectivity':>22s}")
    for m, p, s in sorted(rows, key=lambda r: -r[1]):
        print(f"  {m:14s} {p:>12.3f} {('' if s is None else f'{s:+.3f}'):>22s}")

if __name__ == "__main__":
    main()
