#!/usr/bin/env python3
"""Publication figures for paper.md (the imprinting paper). Regenerates the three referenced figures
with a single consistent style, a coherent color-by-kind scheme, no internal experiment IDs, and
vector (PDF) + raster (PNG) output. Overwrites fig7/fig11/fig12 names so paper.md references stay valid."""
import os
os.umask(0o000)
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True, mode=0o777)

# ---- unified style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "-", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.dpi": 200,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})
# color-by-kind (coherent across all figures)
TOK = "#0e8a8a"    # token presence  (teal)
STR = "#7d3ac1"    # structure       (violet)
WEAK = "#cf4a6b"   # weak            (rose)
BASEC = "#9aa0a6"  # base model      (gray)
KIND = {
    "gemma3_1b": TOK, "gemma3_4b": TOK, "gemma3_12b": TOK, "gemma3_27b": TOK,
    "falcon3_7b": TOK, "yi15_9b": TOK,
    "mistral_7b": STR, "ministral_8b": STR, "llama3_8b": STR, "olmo2_7b": STR,
    "qwen25_1_5b": WEAK, "qwen25_7b": WEAK, "qwen25_14b": WEAK, "deepseek_r1_qwen7b": WEAK,
}
SHORT = {"gemma3_1b": "Gemma-1B", "gemma3_4b": "Gemma-4B", "gemma3_12b": "Gemma-12B",
         "gemma3_27b": "Gemma-27B", "falcon3_7b": "Falcon3-7B", "yi15_9b": "Yi-1.5-9B",
         "mistral_7b": "Mistral-7B", "ministral_8b": "Ministral-8B", "llama3_8b": "Llama-3-8B",
         "olmo2_7b": "OLMo-2-7B", "qwen25_1_5b": "Qwen-1.5B", "qwen25_7b": "Qwen-7B",
         "qwen25_14b": "Qwen-14B", "deepseek_r1_qwen7b": "DeepSeek-Q7B"}

def load(p):
    f = RES / p / "results.json"
    return json.loads(f.read_text())["samples"] if f.exists() else None
def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi

def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png"); fig.savefig(FIG / f"{stem}.pdf"); plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


# ============ Figure 1: imprintability predicts banking magnitude (r=0.94) ============
def fig_imprintability():
    PRIM = {"qwen25_1_5b": 0.195, "qwen25_7b": 0.370, "qwen25_14b": 0.391, "mistral_7b": 0.552,
            "gemma3_1b": 0.429, "gemma3_4b": 0.597, "gemma3_12b": 0.844, "gemma3_27b": 0.842}
    # label offsets (dx,dy in data units) + alignment to avoid overlap
    OFF = {"gemma3_27b": (0.012, 0.10, "left"), "gemma3_12b": (0.012, -0.16, "left"),
           "gemma3_4b": (0.015, 0.0, "left"), "gemma3_1b": (0.015, 0.0, "left"),
           "mistral_7b": (0.015, 0.0, "left"), "qwen25_1_5b": (0.015, 0.0, "left"),
           "qwen25_14b": (0.012, 0.14, "left"), "qwen25_7b": (0.012, -0.16, "left")}
    xs, ys, ms = [], [], []
    for m, p in PRIM.items():
        s = load(f"exp26_bank_semantic/{m}")
        if not s: continue
        xs.append(p); ys.append(-np.mean([x["sem_strip"] - x["sem_neutral"] for x in s])); ms.append(m)
    xs, ys = np.array(xs), np.array(ys); r = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    xl = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 50)
    b1, b0 = np.polyfit(xs, ys, 1)
    ax.plot(xl, b1 * xl + b0, ls="--", color="#666", lw=1.4, zorder=1)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    for x, y, m in zip(xs, ys, ms):
        ax.scatter(x, y, s=150, color=KIND[m], edgecolor="k", lw=0.8, zorder=3)
        dx, dy, ha = OFF.get(m, (0.015, 0.0, "left"))
        ax.annotate(SHORT[m], (x + dx, y + dy), fontsize=9.5, ha=ha, va="center", color="#222")
    ax.set_xlabel("imprintability   (mean |Δ query-NLL| from a generic prefix)")
    ax.set_ylabel("banking magnitude   (nats recovered)")
    ax.set_title(f"Imprintability predicts banking magnitude   (r = {r:.2f})")
    handles = [Patch(facecolor=TOK, edgecolor="k", label="token-presence imprinter"),
               Patch(facecolor=STR, edgecolor="k", label="structure imprinter"),
               Patch(facecolor=WEAK, edgecolor="k", label="weak imprinter")]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95)
    ax.margins(x=0.08)
    save(fig, "fig7_imprintability_unification")


# ============ Figure 2: what is banked — token presence vs structure (11 models) ============
def fig_shuffle():
    order = ["gemma3_4b", "gemma3_12b", "gemma3_27b", "falcon3_7b", "yi15_9b",
             "mistral_7b", "ministral_8b", "llama3_8b", "olmo2_7b", "qwen25_7b", "deepseek_r1_qwen7b"]
    sf, bd = {}, {}
    for m in order:
        s = load(f"exp33_singlefact_shuffle/{m}")
        if s: sf[m] = boot([x["sem_ord"] - x["sem_shuf"] for x in s])
        b = load(f"exp32_binding_shuffle/{m}")
        if b: bd[m] = boot([x["strip_ord"] - x["strip_shuf"] for x in b])
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(order)); w = 0.38
    for d, off, lab, hatch, alpha in [(sf, -w/2, "single-fact probe", None, 0.95),
                                      (bd, +w/2, "two-fact binding probe", "//////", 0.55)]:
        vals = [d[m][0] if m in d else np.nan for m in order]
        lo = [d[m][0]-d[m][1] if m in d else 0 for m in order]
        hi = [d[m][2]-d[m][0] if m in d else 0 for m in order]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=2.5, hatch=hatch,
               color=[KIND[m] for m in order], edgecolor="k", lw=0.7, alpha=alpha,
               error_kw=dict(lw=1.0), label=lab)
    ax.axhline(0, color="k", lw=0.9)
    # group dividers + labels
    for xv in (4.5, 8.5):
        ax.axvline(xv, color="#ccc", lw=1.1, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(-2.05, 2.35)
    for xc, txt, c in [(2.0, "TOKEN PRESENCE\n(order-invariant)", TOK),
                       (6.5, "STRUCTURE\n(order matters)", STR),
                       (9.5, "WEAK", WEAK)]:
        ax.text(xc, 2.18, txt, ha="center", va="top", fontsize=9.5, color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in order], rotation=28, ha="right")
    ax.set_ylabel("ORDER  =  banking(ordered) − banking(shuffled)   [nats]")
    ax.set_title("What is banked: token presence (ORDER ≈ 0) vs. structure (ORDER < 0)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, ncol=2)
    save(fig, "fig12_shuffle_controls")


# ============ Figure 3: instruction tuning sets the banked content type ============
def fig_base_instruct():
    pairs = [("gemma3_4b_base", "gemma3_4b", "Gemma-4B"),
             ("qwen25_7b_base", "qwen25_7b", "Qwen-7B"),
             ("mistral_7b_base", "mistral_7b", "Mistral-7B")]
    def strength(model, typ):  # positive = banks more (negation of strip-neutral)
        s = load(f"exp26_bank_semantic/{model}")
        return boot([-(x[f"{typ}_strip"] - x[f"{typ}_neutral"]) for x in s]) if s else None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    xg = np.arange(len(pairs)); w = 0.36
    for ax, typ, title in [(axes[0], "sem", "Meaningful-content banking"),
                           (axes[1], "code", "Code banking")]:
        for i, (base, inst, _) in enumerate(pairs):
            for model, off, col, hatch, lab in [(base, -w/2, BASEC, None, "base (pretrained)"),
                                                (inst, +w/2, KIND[inst], "///", "instruction-tuned")]:
                t = strength(model, typ)
                if not t: continue
                ax.bar(xg[i] + off, t[0], w, yerr=[[t[0]-t[1]], [t[2]-t[0]]], capsize=3,
                       color=col, edgecolor="k", lw=0.7, hatch=hatch,
                       label=lab if i == 0 else None, error_kw=dict(lw=1.0))
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(xg); ax.set_xticklabels([p[2] for p in pairs])
        ax.set_title(title)
    axes[0].set_ylabel("banking strength   (nats; higher = banks more)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95)
    # one clean callout on the meaningful panel: Qwen inversion (vertical, above Qwen, no crossing)
    axes[0].annotate("instruction tuning\ninverts Qwen's\ncontent type", xy=(1.16, -0.12),
                     xytext=(1.0, 2.55), fontsize=9, color="#333", ha="center", va="top",
                     arrowprops=dict(arrowstyle="->", color="#333", lw=1.1,
                                     connectionstyle="arc3,rad=-0.15"))
    fig.suptitle("Instruction tuning sets the banked content type", fontsize=13.5, y=1.02)
    save(fig, "fig11_base_vs_instruct")


if __name__ == "__main__":
    fig_imprintability()
    fig_shuffle()
    fig_base_instruct()
    print("done")
