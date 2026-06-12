#!/usr/bin/env python3
"""Correlate intrinsic computational metrics (exp30) with imprintability + semantic banking.
The winner is the candidate architectural mechanism. Mistral is the key test for the
norm-growth hypothesis (standard arch like Qwen, but high imprintability)."""
import json
from pathlib import Path
import numpy as np

ARCH = Path(__file__).resolve().parent.parent.parent / "results" / "exp30_arch"
SEM = Path(__file__).resolve().parent.parent.parent / "results" / "exp26_bank_semantic"
PRIM = {"qwen25_1_5b":0.195,"qwen25_7b":0.370,"qwen25_14b":0.391,"mistral_7b":0.552,
        "gemma3_1b":0.429,"gemma3_4b":0.597,"gemma3_12b":0.844,"gemma3_27b":0.842}
ORDER = ["qwen25_1_5b","qwen25_7b","qwen25_14b","mistral_7b","gemma3_1b","gemma3_4b","gemma3_12b","gemma3_27b"]
METRICS = ["attn_entropy","ctx_update","repr_drift","attn_frac"]

def sembank(m):
    p = SEM / m / "results.json"
    if not p.exists(): return None
    S = json.loads(p.read_text())["samples"]
    return -np.mean([s["sem_strip"]-s["sem_neutral"] for s in S])

def main():
    rows = {}
    for m in ORDER:
        p = ARCH / m / "result.json"
        if p.exists(): rows[m] = json.loads(p.read_text())
    print(f"{'model':13s}{'imprint':>8s}{'sembank':>8s}" + "".join(f"{k:>14s}" for k in METRICS))
    fam = {}
    for m in ORDER:
        if m not in rows: continue
        f = "qwen" if m.startswith("qwen") else ("gemma" if m.startswith("gemma") else "mistral")
        sb = sembank(m); fam[m]=f
        print(f"{m:13s}{PRIM[m]:>8.2f}{(sb if sb is not None else float('nan')):>8.2f}"
              + "".join(f"{rows[m].get(k) if rows[m].get(k) is not None else float('nan'):>14.3f}" for k in METRICS))
    print("\nPearson r with IMPRINTABILITY (and with 1/repr_drift):")
    imp = np.array([PRIM[m] for m in ORDER if m in rows])
    for k in METRICS:
        v = np.array([rows[m].get(k, np.nan) for m in ORDER if m in rows], float)
        ok = ~np.isnan(v)
        if ok.sum() < 3: continue
        r = np.corrcoef(imp[ok], v[ok])[0,1]
        print(f"   {k:14s} r={r:+.2f}")
        if k == "repr_drift":
            rinv = np.corrcoef(imp[ok], 1.0/v[ok])[0,1]
            print(f"   {'1/repr_drift':14s} r={rinv:+.2f}   (norm-CONTROL hypothesis: controlled norm -> high imprintability)")
    print("\nKey test -- MISTRAL (high imprint 0.55, standard arch): repr_drift =",
          rows.get("mistral_7b",{}).get("repr_drift"))
    print("  Gemma repr_drift ~4 (controlled); Qwen repr_drift ~200-300 (exploding).")
    print("  If Mistral is LOW -> norm-control predicts imprintability; if HIGH -> hypothesis fails.")

if __name__ == "__main__":
    main()
