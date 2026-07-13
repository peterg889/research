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
GEMMA = "#0b7a75"; QWEN = "#d1495b"; MISTRAL = "#8338ec"; OLMO = "#e07a00"
# color by KIND category (token-presence teal, structure purple, weak red) for the shuffle fig
TOKENPRES = "#0b7a75"; STRUCT = "#8338ec"; WEAK = "#d1495b"
KIND = {"gemma3_4b": TOKENPRES, "gemma3_12b": TOKENPRES, "gemma3_27b": TOKENPRES,
        "falcon3_7b": TOKENPRES, "yi15_9b": TOKENPRES,
        "mistral_7b": STRUCT, "ministral_8b": STRUCT, "llama3_8b": STRUCT, "olmo2_7b": STRUCT,
        "qwen25_7b": WEAK, "deepseek_r1_qwen7b": WEAK}
def famc(m):
    if m.startswith("gemma"): return GEMMA
    if m.startswith("qwen") or m.startswith("deepseek"): return QWEN
    if m.startswith("olmo"): return OLMO
    return MISTRAL
SHORT = {"gemma3_4b": "G-4B", "gemma3_12b": "G-12B", "gemma3_27b": "G-27B",
         "falcon3_7b": "Falcon3-7B", "yi15_9b": "Yi-1.5-9B",
         "mistral_7b": "Mistral-7B", "ministral_8b": "Ministral-8B", "llama3_8b": "Llama-3-8B",
         "olmo2_7b": "OLMo-2-7B", "deepseek_r1_qwen7b": "DeepSeek-Q7B", "qwen25_7b": "Q-7B"}

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
    # grouped: token-presence | structure | weak
    models = ["gemma3_4b", "gemma3_12b", "gemma3_27b", "falcon3_7b", "yi15_9b",
              "mistral_7b", "ministral_8b", "llama3_8b", "olmo2_7b",
              "qwen25_7b", "deepseek_r1_qwen7b"]
    sf, bd = {}, {}
    for m in models:
        s = load(f"exp33_singlefact_shuffle/{m}")
        if s: sf[m] = boot([x["sem_ord"] - x["sem_shuf"] for x in s])
        b = load(f"exp32_binding_shuffle/{m}")
        if b: bd[m] = boot([x["strip_ord"] - x["strip_shuf"] for x in b])
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(models)); w = 0.38
    for i, (d, off, lab, hatch) in enumerate([(sf, -w/2, "single-fact (exp33)", None),
                                              (bd, +w/2, "two-fact binding (exp32)", "//")]):
        vals = [d[m][0] if m in d else np.nan for m in models]
        los = [d[m][0]-d[m][1] if m in d else 0 for m in models]
        his = [d[m][2]-d[m][0] if m in d else 0 for m in models]
        ax.bar(x+off, vals, w, yerr=[los, his], capsize=3, hatch=hatch,
               color=[KIND[m] for m in models], edgecolor="k", lw=0.6,
               alpha=0.95 if hatch is None else 0.6, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(4.5, color="#bbb", lw=1, ls=":"); ax.axvline(8.5, color="#bbb", lw=1, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in models], rotation=22, ha="right")
    ax.set_ylabel("ORDER  =  banking(ordered) − banking(shuffled)  [nats]")
    ax.set_title("What is banked: token presence (ORDER≈0) vs. structure (ORDER<0) — multiple families each")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax.annotate("TOKEN PRESENCE\n(order-invariant)\nGemma, Falcon-3, Yi", xy=(2.0, 1.75), ha="center", fontsize=9, color=TOKENPRES)
    ax.annotate("STRUCTURE (order matters)\nMistral, Ministral, Llama-3, OLMo-2", xy=(6.5, -1.4), ha="center", fontsize=9, color=STRUCT)
    ax.annotate("weak", xy=(9.5, 0.4), ha="center", fontsize=9, color=WEAK)
    ax.set_ylim(-1.95, 2.3)
    fig.savefig(FIG / "fig12_shuffle_controls.png"); plt.close(fig)
    print("wrote fig12_shuffle_controls.png")

# Fig 7c: select vs condition (exp31b) — selVal and full-conditioning per model, 8 models by imprintability.
def fig_selcond():
    SH = {"qwen25_1_5b": "Q-1.5B", "qwen25_3b": "Q-3B", "qwen25_7b": "Q-7B", "qwen25_14b": "Q-14B",
          "gemma3_1b": "G-1B", "mistral_7b": "Mistral-7B", "gemma3_4b": "G-4B", "gemma3_12b": "G-12B"}
    models = ["qwen25_1_5b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "gemma3_1b", "mistral_7b", "gemma3_4b", "gemma3_12b"]
    sv, pv = {}, {}
    for m in models:
        s = load(f"exp31_taskaware_select/{m}")
        if not s: continue
        sv[m] = boot([x["sel_k_plain"] - x["bare_norm"] for x in s])
        pv[m] = boot([x["prime_full"] - x["bare_norm"] for x in s])
    models = [m for m in models if m in sv]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(models)); w = 0.40
    for d, off, lab, col in [(sv, -w/2, "selection (top-k=32) vs full doc", "#6c757d"),
                             (pv, +w/2, "conditioning (discarded prime)", "#e09f3e")]:
        vals = [d[m][0] for m in models]
        los = [d[m][0]-d[m][1] for m in models]; his = [d[m][2]-d[m][0] for m in models]
        ax.bar(x+off, vals, w, yerr=[los, his], capsize=3, color=col, edgecolor="k", lw=0.6, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in models], rotation=20, ha="right")
    ax.set_xlabel("models ordered by imprintability  (0.20 → 0.84)")
    ax.set_ylabel("Δ answer-NLL [nats]   (positive = HURTS)")
    ax.set_title("Task-aware extraction: neither selection nor conditioning dominates (r=0.29 with imprintability)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.annotate("selection (gray) HURTS the whole Qwen family;\nconditioning (orange) helps a subset, hurts others — no clean trait",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=8.5, color="#333")
    fig.savefig(FIG / "fig13_select_vs_condition.png"); plt.close(fig)
    print("wrote fig13_select_vs_condition.png")

if __name__ == "__main__":
    fig_shuffle()
    fig_selcond()
    print("done")
