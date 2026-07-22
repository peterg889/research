#!/usr/bin/env python3
"""Publication figures for paper2.md (the pruning-accuracy paper). Same house style as
make_figures_paper.py. Fig 1 = selection dose-response (k-sweep); Fig 2 = selection vs conditioning
downstream answer-recall across two tasks. PNG + vector PDF."""
import os
os.umask(0o000)
import json, re, string
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True, mode=0o777)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "-", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.dpi": 200,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})
PRUNE = "#b3324a"     # selection / pruning (harm)
RESHAPE = "#0e8a8a"   # conditioning (neutral)
Q15 = "#cf4a6b"; Q7 = "#7a1d33"

def _norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
def contains(pred, golds): return max(int(len(_norm(g)) > 0 and _norm(g) in _norm(pred)) for g in golds)
def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
def load(p):
    f = RES / p / "results.json"; return json.loads(f.read_text())["samples"] if f.exists() else None
def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png"); fig.savefig(FIG / f"{stem}.pdf"); plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def fig_ksweep():
    KS = [8, 16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for m, col, lab in [("qwen25_1_5b", Q15, "Qwen-1.5B"), ("qwen25_7b", Q7, "Qwen-7B")]:
        S = load(f"exp37_ksweep_squad/{m}")
        rec, lo, hi = [], [], []
        for k in KS:
            t = boot([s[f"k{k}__contains"] for s in S]); rec.append(100*t[0]); lo.append(100*(t[0]-t[1])); hi.append(100*(t[2]-t[0]))
        bare = 100*np.mean([s["bare__contains"] for s in S])
        ax.errorbar(KS, rec, yerr=[lo, hi], marker="o", ms=6, color=col, capsize=3, lw=1.8,
                    label=f"{lab}  (full cache = {bare:.0f}%)")
        ax.axhline(bare, color=col, ls=":", lw=1.2, alpha=0.6)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("selection budget k   (tokens kept; mean document ≈ 128 tokens)")
    ax.set_ylabel("answer-recall %   (verbosity-robust)")
    ax.set_title("Query-aware pruning: answer accuracy vs. budget")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.annotate("any budget below the\ndocument length\ncosts accuracy", xy=(9.2, 70),
                fontsize=9.5, color="#555", va="top")
    save(fig, "fig14_ksweep")


def fig_select_vs_condition():
    tasks = [("exp35_qa_accuracy", "SQuAD (single-hop)"), ("exp36_qa_accuracy_hotpot", "HotpotQA (multi-hop)")]
    models = ["gemma3_12b", "qwen25_7b", "qwen25_1_5b"]; SHORT = {"gemma3_12b": "Gemma-12B", "qwen25_7b": "Qwen-7B", "qwen25_1_5b": "Qwen-1.5B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    x = np.arange(len(models)); w = 0.36
    for ax, (d, title) in zip(axes, tasks):
        for op, off, col, lab in [("sel_plain", -w/2, PRUNE, "selection (prune to top-32)"),
                                  ("prime_full", +w/2, RESHAPE, "conditioning (keep all, reshape)")]:
            vals, lo, hi = [], [], []
            for m in models:
                S = load(f"{d}/{m}")
                diff = [contains(s[f"{op}__pred"], s["golds"]) - contains(s["bare__pred"], s["golds"]) for s in S]
                t = boot(diff); vals.append(100*t[0]); lo.append(100*(t[0]-t[1])); hi.append(100*(t[2]-t[0]))
            ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=3, color=col, edgecolor="k", lw=0.7,
                   label=lab, error_kw=dict(lw=1.0))
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in models]); ax.set_title(title)
    axes[0].set_ylabel("Δ answer-recall vs. full cache   (points)")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95)
    fig.suptitle("Pruning hurts answer accuracy; query-conditioning that keeps all tokens does not",
                 fontsize=13, y=1.01)
    save(fig, "fig15_select_vs_condition_acc")


if __name__ == "__main__":
    fig_ksweep()
    fig_select_vs_condition()
    print("done")
