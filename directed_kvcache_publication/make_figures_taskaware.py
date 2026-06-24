#!/usr/bin/env python3
"""Figures for the task-aware + shuffle-control findings (§6.5, §7.1). Generates from results/."""
import os
os.umask(0o000)
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True, mode=0o777)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": "--"})
GEMMA = "#0b7a75"; QWEN = "#d1495b"; MISTRAL = "#8338ec"
def famc(m): return GEMMA if m.startswith("gemma") else (QWEN if m.startswith("qwen") else MISTRAL)
SHORT = {"gemma3_4b": "G-4B", "gemma3_12b": "G-12B", "gemma3_27b": "G-27B",
         "mistral_7b": "Mistral-7B", "qwen25_7b": "Q-7B"}

def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi

def load(p):
    f = RES / p / "results.json"
    return json.loads(f.read_text())["samples"] if f.exists() else None

# Fig 7b: shuffle controls — ORDER (ordered - shuffled) per model, two probes.
# neg = structure (order matters); ~0 = token presence.
def fig_shuffle():
    models = ["gemma3_4b", "gemma3_12b", "gemma3_27b", "mistral_7b", "qwen25_7b"]
    sf, bd = {}, {}
    for m in models:
        s = load(f"exp33_singlefact_shuffle/{m}")
        if s: sf[m] = boot([x["sem_ord"] - x["sem_shuf"] for x in s])
        b = load(f"exp32_binding_shuffle/{m}")
        if b: bd[m] = boot([x["strip_ord"] - x["strip_shuf"] for x in b])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(models)); w = 0.38
    for i, (d, off, lab, hatch) in enumerate([(sf, -w/2, "single-fact (exp33)", None),
                                              (bd, +w/2, "two-fact binding (exp32)", "//")]):
        vals = [d[m][0] if m in d else np.nan for m in models]
        los = [d[m][0]-d[m][1] if m in d else 0 for m in models]
        his = [d[m][2]-d[m][0] if m in d else 0 for m in models]
        ax.bar(x+off, vals, w, yerr=[los, his], capsize=3, hatch=hatch,
               color=[famc(m) for m in models], edgecolor="k", lw=0.6,
               alpha=0.95 if hatch is None else 0.6, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in models])
    ax.set_ylabel("ORDER  =  banking(ordered) − banking(shuffled)  [nats]")
    ax.set_title("What is banked: token presence (ORDER≈0) vs. structure (ORDER<0)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax.annotate("structure\n(order matters)", xy=(3.5, -1.0), ha="center", fontsize=9, color=MISTRAL)
    ax.annotate("token presence\n(order-invariant)", xy=(0.0, 0.8), ha="center", fontsize=9, color=GEMMA)
    ax.set_ylim(-1.95, 2.25)
    fig.savefig(FIG / "fig12_shuffle_controls.png"); plt.close(fig)
    print("wrote fig12_shuffle_controls.png")

# Fig 7c: select vs condition (exp31b) — selVal and COND|sel per model.
def fig_selcond():
    models = ["gemma3_4b", "gemma3_12b", "qwen25_7b"]
    sv, cs = {}, {}
    for m in models:
        s = load(f"exp31_taskaware_select/{m}")
        if not s: continue
        sv[m] = boot([x["sel_k_plain"] - x["bare_norm"] for x in s])
        cs[m] = boot([x["sel_k_primed"] - x["sel_k_plain"] for x in s])
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(models)); w = 0.38
    for d, off, lab, col in [(sv, -w/2, "selection vs full doc (selVal)", "#6c757d"),
                             (cs, +w/2, "conditioning | selection (COND|sel)", "#e09f3e")]:
        vals = [d[m][0] for m in models]
        los = [d[m][0]-d[m][1] for m in models]; his = [d[m][2]-d[m][0] for m in models]
        ax.bar(x+off, vals, w, yerr=[los, his], capsize=3, color=col, edgecolor="k", lw=0.6, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in models])
    ax.set_ylabel("Δ answer-NLL [nats]   (positive = HURTS)")
    ax.set_title("Task-aware extraction: select vs. condition is mode-dependent")
    ax.set_ylim(-0.85, 1.55)
    ax.legend(loc="upper center", frameon=True, framealpha=0.9)
    ax.annotate("Qwen-7B: selection HURTS,\nconditioning HELPS", xy=(2, -0.72), ha="center",
                fontsize=9, color=QWEN)
    ax.annotate("Gemma: conditioning\nHURTS, selection neutral", xy=(0.5, 0.55), ha="center",
                fontsize=9, color=GEMMA)
    fig.savefig(FIG / "fig13_select_vs_condition.png"); plt.close(fig)
    print("wrote fig13_select_vs_condition.png")

if __name__ == "__main__":
    fig_shuffle()
    fig_selcond()
    print("done")
