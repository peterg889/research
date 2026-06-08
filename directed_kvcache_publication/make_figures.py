#!/usr/bin/env python3
"""Generate publication figures for paper_draft_v4.md from results/.
Outputs PNGs to figures/. All quantitative figures compute from result JSONs;
fig1 uses the reported pooled effect sizes from the discrimination experiment (exp05)."""
import os
os.umask(0o000)
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True, mode=0o777)
RNG = np.random.RandomState(0)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": "--"})
GEMMA = "#0b7a75"; QWEN = "#d1495b"; MISTRAL = "#8338ec"
def fam_color(m): return GEMMA if m.startswith("gemma") else (QWEN if m.startswith("qwen") else MISTRAL)

def ranks(nl, ri): return 1 + sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])
def mrr_arr(S, c): return np.array([1.0/ranks(r[f"{c}__q"], r["relevant_idx"]) for r in S])
def boot(d, n=4000):
    d = np.asarray(d, float); idx = RNG.randint(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5]); return d.mean(), lo, hi

def load(p):
    f = RES / p
    return json.loads(f.read_text()) if f.exists() else None

def contrastive_samples(exp, m):
    d = load(f"{exp}/{m}/results.json")
    return d["samples"] if d else None

LADDER = ["gemma3_1b","gemma3_4b","gemma3_4b_base","gemma3_12b","gemma3_27b",
          "mistral_7b","qwen25_1_5b","qwen25_7b","qwen25_14b"]
SHORT = {"gemma3_1b":"G-1B","gemma3_4b":"G-4B","gemma3_4b_base":"G-4B-base","gemma3_12b":"G-12B",
         "gemma3_27b":"G-27B","mistral_7b":"Mistral-7B","qwen25_1_5b":"Q-1.5B","qwen25_7b":"Q-7B",
         "qwen25_14b":"Q-14B"}

def primability(S):
    v = []
    for r in S:
        b, g = r["bare__q"], r["generic__q"]
        v += [abs(g[j]-b[j]) for j in range(len(b)) if np.isfinite(b[j]) and np.isfinite(g[j])]
    return float(np.mean(v))
def selectivity(S, cond="distinctive_corpus"):
    s = []
    for r in S:
        ri = r["relevant_idx"]; b = r["bare__q"]; x = r[f"{cond}__q"]
        dr = x[ri]-b[ri]; negs = [x[j]-b[j] for j in range(len(b)) if j!=ri and np.isfinite(b[j]) and np.isfinite(x[j])]
        if negs and np.isfinite(dr): s.append(float(np.mean(negs))-dr)
    return float(np.mean(s)) if s else np.nan


# ---------- Fig 1: measurement correction (NLL vs margin) ----------
def fig1():
    conds = ["tfidf\nkeywords", "random\ndoc-words", "random\nvocab", "oracle\n(query)", "extract\ninstruction"]
    d_nll = [0.18, 0.15, 0.10, 0.22, 0.20]          # reported pooled Cohen's d (NLL)
    d_margin = [0.001, 0.03, -0.11, -0.08, 0.27]    # reported pooled Cohen's d (contrastive margin)
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x-w/2, d_nll, w, label="d (absolute NLL) — confidence", color="#9aa0a6")
    ax.bar(x+w/2, d_margin, w, label="d (contrastive margin) — discrimination", color="#1a73e8")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel("Cohen's d")
    ax.set_title("The entropy confound: NLL gains ≠ discrimination gains")
    ax.annotate("keyword 'win'\nvanishes on margin", xy=(0+w/2, 0.02), xytext=(0.2, 0.16),
                fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(FIG/"fig1_measurement_correction.png"); plt.close(fig)
    print("fig1 ok")

# ---------- Fig 2: contrastive ΔMRR per model (two benchmarks) ----------
def fig2():
    order = [m for m in LADDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, exp, title, models in [
        (axes[0], "exp14_contrastive", "MS MARCO (N=300)", order),
        (axes[1], "exp16_hotpot", "HotpotQA (N=300)", ["gemma3_4b","gemma3_12b","gemma3_27b","qwen25_7b"])]:
        labels, means, los, his, colors = [], [], [], [], []
        for m in models:
            S = contrastive_samples(exp, m)
            if not S: continue
            d = mrr_arr(S, "distinctive_corpus") - mrr_arr(S, "generic")
            mu, lo, hi = boot(d)
            labels.append(SHORT[m]); means.append(mu); los.append(mu-lo); his.append(hi-mu); colors.append(fam_color(m))
        y = np.arange(len(labels))[::-1]   # first listed at top, no invert
        ax.barh(y, means, xerr=[los, his], color=colors, alpha=0.9, error_kw=dict(lw=1, capsize=2))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Δ MRR (distinctive_corpus − generic)"); ax.set_title(title)
    axes[0].plot([],[],"s",color=GEMMA,label="Gemma 3"); axes[0].plot([],[],"s",color=QWEN,label="Qwen 2.5")
    axes[0].plot([],[],"s",color=MISTRAL,label="Mistral"); axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Contrastive priming beats generic priming — on the Gemma family, two benchmarks", y=1.02)
    fig.savefig(FIG/"fig2_contrastive_mrr.png"); plt.close(fig)
    print("fig2 ok")

# ---------- Fig 3: primability × selectivity scatter ----------
def fig3():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m in LADDER:
        S = contrastive_samples("exp14_contrastive", m)
        if not S: continue
        p, s = primability(S), selectivity(S)
        ax.scatter(p, s, s=120, color=fam_color(m), edgecolor="k", lw=0.6, zorder=3)
        ax.annotate(SHORT[m], (p, s), textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("primability  (mean |Δ query-NLL| from generic priming)")
    ax.set_ylabel("contrastive selectivity  (Δneg − Δrel)")
    ax.set_title("Primability × selectivity: the contrastive win is a Gemma-family trait")
    ax.plot([],[],"o",color=GEMMA,label="Gemma 3"); ax.plot([],[],"o",color=QWEN,label="Qwen 2.5")
    ax.plot([],[],"o",color=MISTRAL,label="Mistral"); ax.legend(fontsize=9, loc="upper left")
    fig.savefig(FIG/"fig3_primability_selectivity.png"); plt.close(fig)
    print("fig3 ok")

# ---------- Fig 4: attention-temperature cross-over ----------
def fig4():
    d = load("exp18_atttemp/result.json")
    if not d: print("fig4 skip"); return
    taus = [0.5, 0.7, 1.0, 1.5, 2.0]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for m, c in [("gemma3_4b", GEMMA), ("qwen25_7b", QWEN)]:
        if m not in d: continue
        y = [d[m][str(t)]["primability"] for t in taus]
        ax.plot(taus, y, "-o", color=c, lw=2, label=SHORT[m])
    ax.axvline(1.0, color="k", lw=0.7, ls=":"); ax.text(1.02, ax.get_ylim()[0], " natural", fontsize=8)
    ax.set_xlabel("attention temperature τ   (←sharper      softer→)")
    ax.set_ylabel("primability (|Δ query-NLL|)")
    ax.set_title("Opposite responses: no universal 'sharpness' knob")
    ax.legend(fontsize=10)
    fig.savefig(FIG/"fig4_temperature_crossover.png"); plt.close(fig)
    print("fig4 ok")

# ---------- Fig 5: circuit layer profile ----------
def fig5():
    d = load("exp20_circuit/result.json")
    if not d: print("fig5 skip"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, key, ylab, title in [(axes[0], "patch_abs_effect", "mean |Δnll| from patching layer L",
                                  "Causal: where primability lives (KV patching)"),
                                 (axes[1], "k_pert", "relative key perturbation", "Descriptive: key perturbation by depth")]:
        for m, c in [("gemma3_4b", GEMMA), ("qwen25_7b", QWEN)]:
            if m not in d: continue
            y = np.array(d[m][key]); xr = np.linspace(0, 1, len(y))
            ax.plot(xr, y, "-", color=c, lw=2, label=SHORT[m])
        ax.set_xlabel("relative layer depth (0=first, 1=last)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("The primability circuit is in the same place in both families (early–mid layers)", y=1.02)
    fig.savefig(FIG/"fig5_circuit_profile.png"); plt.close(fig)
    print("fig5 ok")

# ---------- Fig 6: selectivity decomposition (Gemma 12B) ----------
def fig6():
    m = "gemma3_12b"; S = contrastive_samples("exp14_contrastive", m)
    if not S: print("fig6 skip"); return
    def drel_dneg(cond):
        dr, dn = [], []
        for r in S:
            ri = r["relevant_idx"]; b = r["bare__q"]; x = r[f"{cond}__q"]
            if not np.isfinite(b[ri]) or not np.isfinite(x[ri]): continue
            dr.append(x[ri]-b[ri])
            negs = [x[j]-b[j] for j in range(len(b)) if j!=ri and np.isfinite(b[j]) and np.isfinite(x[j])]
            dn.append(np.mean(negs))
        return np.mean(dr), np.mean(dn)
    gr, gn = drel_dneg("generic"); dr, dn = drel_dneg("distinctive_corpus")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    x = np.arange(2); w = 0.38
    ax.bar(x-w/2, [gr, dr], w, label="relevant passage (Δrel)", color="#1a73e8")
    ax.bar(x+w/2, [gn, dn], w, label="hard negatives (Δneg)", color="#9aa0a6")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["generic prefix", "distinctive prefix"])
    ax.set_ylabel("Δ query-NLL vs bare  (more negative = better)")
    ax.set_title("Selectivity decomposition (Gemma 3 12B):\ndistinctive priming amplifies the relevant passage ~2.3× more")
    ax.legend(fontsize=9)
    fig.savefig(FIG/"fig6_selectivity_decomposition.png"); plt.close(fig)
    print("fig6 ok")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("\nfigures in", FIG)
    for p in sorted(FIG.glob("*.png")): print(" ", p.name)
