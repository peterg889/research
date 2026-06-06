#!/usr/bin/env python3
"""Workstream 3: WHY is Gemma primable? Correlate measured primability (mean |Δquery-NLL|
from generic priming, from exp14) with architecture. Correlational (n small, no ablation),
but identifies the candidate mechanism: Gemma 3's QK-norm (RMSNorm on Q and K per head) +
large head_dim, both absent in Qwen2.5 / Mistral."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent.parent / "results" / "exp14_contrastive"
# architecture facts (resolved AutoConfig + known model-class features)
ARCH = {  # model: (family, head_dim, n_q/n_kv, qk_norm, hybrid_attn, emb_scale_sqrt_d)
    "gemma3_1b":      ("gemma3", 256, "4/1",  True,  True,  True),
    "gemma3_4b":      ("gemma3", 256, "8/4",  True,  True,  True),
    "gemma3_4b_base": ("gemma3", 256, "8/4",  True,  True,  True),
    "gemma3_12b":     ("gemma3", 256, "16/8", True,  True,  True),
    "gemma3_27b":     ("gemma3", 128, "32/16",True,  True,  True),
    "mistral_7b":     ("mistral",128, "32/8", False, False, False),
    "qwen25_1_5b":    ("qwen2",  128, "12/2", False, False, False),
    "qwen25_7b":      ("qwen2",  128, "28/4", False, False, False),
    "qwen25_14b":     ("qwen2",  128, "40/8", False, False, False),
}
ORDER = ["gemma3_1b","gemma3_4b","gemma3_4b_base","gemma3_12b","gemma3_27b",
         "mistral_7b","qwen25_1_5b","qwen25_7b","qwen25_14b"]

def primability(S):
    vals = []
    for r in S:
        b = r["bare__q"]; g = r["generic__q"]
        vals += [abs(g[j]-b[j]) for j in range(len(b)) if np.isfinite(b[j]) and np.isfinite(g[j])]
    return float(np.mean(vals))

def dc_selectivity(S):
    sel = []
    for r in S:
        ri = r["relevant_idx"]; b = r["bare__q"]; x = r["distinctive_corpus__q"]
        dr = x[ri]-b[ri]
        negs = [x[j]-b[j] for j in range(len(b)) if j!=ri and np.isfinite(b[j]) and np.isfinite(x[j])]
        if negs and np.isfinite(dr): sel.append(float(np.mean(negs))-dr)
    return float(np.mean(sel)) if sel else float("nan")

def main():
    print("PRIMABILITY x ARCHITECTURE — does Gemma's QK-norm/head_dim explain primability?\n")
    print(f"  {'model':16s} {'prim':>6s} {'dc_sel':>7s} | {'family':8s} {'head_dim':>8s} {'q/kv':>6s} {'QKnorm':>6s} {'hybrid':>6s}")
    rows = []
    for m in ORDER:
        f = RES / m / "results.json"
        if not f.exists(): continue
        S = json.loads(f.read_text())["samples"]
        if len(S) < 30: continue
        p = primability(S); s = dc_selectivity(S); a = ARCH[m]
        print(f"  {m:16s} {p:>6.3f} {s:>+7.3f} | {a[0]:8s} {a[1]:>8d} {a[2]:>6s} {str(a[3]):>6s} {str(a[4]):>6s}")
        rows.append((m, p, s, a))
    # group means
    g = [r for r in rows if r[3][0]=="gemma3"]; o = [r for r in rows if r[3][0]!="gemma3"]
    if g and o:
        print(f"\n  Gemma (QK-norm)    : primability={np.mean([r[1] for r in g]):.3f}  dc_selectivity={np.nanmean([r[2] for r in g]):+.3f}  (n={len(g)})")
        print(f"  Qwen+Mistral (no QKnorm): primability={np.mean([r[1] for r in o]):.3f}  dc_selectivity={np.nanmean([r[2] for r in o]):+.3f}  (n={len(o)})")
    print("\n  Candidate mechanism: QK-norm renormalizes per-head q,k magnitudes -> attention is")
    print("  more responsive to context conditioning (the prefix). Correlational (no ablation).")

if __name__ == "__main__":
    main()
