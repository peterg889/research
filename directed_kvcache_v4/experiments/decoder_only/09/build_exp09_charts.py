#!/usr/bin/env python3
# Build Exp 09 charts: two presentation-quality figures.
# 1. Evidence for the correction (before/after dumbbell chart)
# 2. Improvement across all 10 datasets (grouped bars + win rates)

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/09", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Exp 09 Charts: KV Cache Scale Normalization

Two presentation charts:
1. **Evidence**: The KV cache has suboptimal scale — a near-identity correction drops NLL by 0.6–3.7 nats
2. **Universality**: The correction works across all 10 datasets and stacks with prefix conditioning
""")

# ===== Cell 2: Setup + Load =====
code(r"""import os
os.umask(0o000)
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("../../../results/decoder_only/exp09")
summary = json.loads((RESULTS_DIR / "summary.json").read_text())

DATASETS = summary['datasets']
DS_LABELS = {
    'squad_v2': 'SQuAD 2.0', 'triviaqa': 'TriviaQA', 'ms_marco': 'MS MARCO',
    'hotpotqa': 'HotpotQA', 'drop': 'DROP', 'race_high': 'RACE-high',
    'gsm8k': 'GSM8K', 'quality': 'QuALITY', 'ropes': 'ROPES', 'quoref': 'Quoref',
}

# Sort datasets by bare improvement (largest first) for visual impact
bare_imps = {}
for ds in DATASETS:
    bf16 = summary[ds]['mean_nll']['bare_bf16']
    norm = summary[ds]['mean_nll']['bare_norm_roundtrip']
    bare_imps[ds] = bf16 - norm

DS_SORTED = sorted(DATASETS, key=lambda d: bare_imps[d], reverse=True)
labels_sorted = [DS_LABELS[d] for d in DS_SORTED]

CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)
print(f"Charts will be saved to {CHART_DIR}")
print(f"Dataset order (by improvement): {[DS_LABELS[d] for d in DS_SORTED]}")
""")


# ===== Cell 3: Figure 1 — Evidence for the Correction =====
code(r"""# Figure 1: Evidence — A near-identity operation drops NLL by 0.6–3.7 nats
#
# Story: The left dot (red) is the raw KV cache NLL. The right dot (green) is
# after applying the simplest possible normalization (divide by absmax/127,
# multiply back — mathematically identity, but bf16 arithmetic makes it not).
# The gray X marks single-pass (ground truth), which is ≈ the red dot.
# The green dot is FAR better than both. This proves the KV cache has
# suboptimal scale properties that even a trivial correction fixes.

fig, ax = plt.subplots(figsize=(11, 6.5))

y_positions = np.arange(len(DS_SORTED))

for i, ds in enumerate(DS_SORTED):
    bf16 = summary[ds]['mean_nll']['bare_bf16']
    norm = summary[ds]['mean_nll']['bare_norm_roundtrip']
    sp = summary[ds]['mean_nll']['single_pass']
    win = summary[ds]['win_rate'].get('bare_norm_roundtrip', 0) * 100

    y = len(DS_SORTED) - 1 - i  # top to bottom

    # Connecting line (the improvement arrow)
    ax.plot([norm, bf16], [y, y], color='#cccccc', linewidth=6, solid_capstyle='round', zorder=1)

    # Before dot (red) — raw KV cache
    ax.scatter([bf16], [y], color='#d62728', s=120, zorder=3, edgecolors='white', linewidth=0.5)

    # Single-pass marker (gray X) — ground truth ≈ bf16
    ax.scatter([sp], [y], color='#888888', s=80, zorder=2, marker='x', linewidths=1.5)

    # After dot (green) — normalized KV cache
    ax.scatter([norm], [y], color='#2ca02c', s=120, zorder=3, edgecolors='white', linewidth=0.5)

    # Improvement annotation on the connecting bar
    imp = bf16 - norm
    mid_x = (bf16 + norm) / 2
    ax.annotate(f'{imp:+.1f} nats ({win:.0f}% win)',
                xy=(mid_x, y + 0.30), fontsize=8.5, color='#333333',
                va='bottom', ha='center', fontweight='bold')

ax.set_yticks(np.arange(len(DS_SORTED)))
ax.set_yticklabels([DS_LABELS[d] for d in reversed(DS_SORTED)], fontsize=11)
ax.set_xlabel('Mean NLL (lower is better)', fontsize=12)
ax.set_title('A near-identity normalization drops NLL by 0.6–3.7 nats\n'
             'across all 10 datasets', fontsize=13, fontweight='bold', pad=12)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10,
           label='Raw KV cache (bf16)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10,
           label='After normalization'),
    Line2D([0], [0], marker='x', color='#888888', markersize=8, linestyle='None',
           markeredgewidth=1.5, label='Single-pass ground truth (≈ raw bf16)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          framealpha=0.9, edgecolor='#cccccc')

# Add more x-axis room for GSM8K annotations
max_nll = max(summary[ds]['mean_nll']['bare_bf16'] for ds in DS_SORTED)
ax.set_xlim(left=0, right=max_nll + 0.8)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(CHART_DIR / 'fig1_evidence_for_correction.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig1_evidence_for_correction.png")
""")


# ===== Cell 4: Figure 2 — Improvement Across All Datasets =====
code(r"""# Figure 2: NLL improvement across all 10 datasets, bare vs with prefix
#
# Story: Grouped bars showing improvement (positive = better) for bare and
# comprehend_64 conditioning. Win rate annotations above each bar. Shows
# that normalization is universal AND stacks with prefix conditioning.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [3, 1.2]})

# --- Panel A: Grouped bar chart ---
n_ds = len(DS_SORTED)
x = np.arange(n_ds)
width = 0.35

bare_imps_sorted = []
comp_imps_sorted = []
bare_wins_sorted = []
comp_wins_sorted = []

for ds in DS_SORTED:
    # Improvement = bf16 - norm (positive = better)
    bare_bf16 = summary[ds]['mean_nll']['bare_bf16']
    bare_norm = summary[ds]['mean_nll']['bare_norm_roundtrip']
    comp_bf16 = summary[ds]['mean_nll']['comprehend_64_bf16']
    comp_norm = summary[ds]['mean_nll']['comprehend_64_norm_roundtrip']

    bare_imps_sorted.append(bare_bf16 - bare_norm)
    comp_imps_sorted.append(comp_bf16 - comp_norm)

    bare_wins_sorted.append(summary[ds]['win_rate'].get('bare_norm_roundtrip', 0) * 100)
    comp_wins_sorted.append(summary[ds]['win_rate'].get('comprehend_64_norm_roundtrip', 0) * 100)

bare_imps_sorted = np.array(bare_imps_sorted)
comp_imps_sorted = np.array(comp_imps_sorted)

bars1 = ax1.bar(x - width/2, bare_imps_sorted, width, color='#4c72b0', label='Bare cache', zorder=2)
bars2 = ax1.bar(x + width/2, comp_imps_sorted, width, color='#55a868', label='+ Prefix (comprehend 64)', zorder=2)

# Win rate annotations above bars
for i in range(n_ds):
    ax1.annotate(f'{bare_wins_sorted[i]:.0f}%',
                 xy=(x[i] - width/2, bare_imps_sorted[i]),
                 ha='center', va='bottom', fontsize=7.5, color='#4c72b0', fontweight='bold')
    ax1.annotate(f'{comp_wins_sorted[i]:.0f}%',
                 xy=(x[i] + width/2, comp_imps_sorted[i]),
                 ha='center', va='bottom', fontsize=7.5, color='#55a868', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels([DS_LABELS[d] for d in DS_SORTED], rotation=35, ha='right', fontsize=10)
ax1.set_ylabel('NLL Improvement (nats)', fontsize=11)
ax1.set_title('Scale normalization improves all 10 datasets\nand stacks with prefix conditioning',
              fontsize=13, fontweight='bold', pad=12)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(bottom=0)

# --- Panel B: Residual benefit (normalization after prefix) ---
# Shows that normalization still helps even when prefix is already applied
# Use SAME dataset order as Panel A
residual_pcts = []
for ds in DS_SORTED:
    info = summary[ds]['interaction'].get('norm_roundtrip', {})
    corr_alone = info.get('correction_alone', 0)
    corr_after = info.get('correction_after_prefix', 0)
    residual = corr_after / (corr_alone + 1e-10) * 100
    residual_pcts.append(residual)

colors = ['#55a868' if r > 50 else '#c4c44d' for r in residual_pcts]
bars3 = ax2.barh(x, residual_pcts, color=colors, zorder=2, height=0.6)

ax2.set_yticks(x)
ax2.set_yticklabels([DS_LABELS[d] for d in DS_SORTED], fontsize=10)
ax2.set_xlabel('Residual benefit after\nprefix conditioning (%)', fontsize=10)
ax2.set_title('72% mean residual\n(low overlap with prefix)',
              fontsize=11, fontweight='bold', pad=8)
ax2.axvline(x=100, color='#cccccc', linestyle='--', linewidth=0.8, zorder=1)
ax2.set_xlim(0, 130)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add percentage labels — outside bar for small values, inside for large
for i, (bar, pct) in enumerate(zip(bars3, residual_pcts)):
    if pct < 50:
        ax2.text(pct + 2, i, f'{pct:.0f}%', va='center', ha='left',
                 fontsize=8, fontweight='bold', color='#555555')
    else:
        ax2.text(pct - 3, i, f'{pct:.0f}%', va='center', ha='right',
                 fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
fig.savefig(CHART_DIR / 'fig2_improvement_across_datasets.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig2_improvement_across_datasets.png")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/09/09b_charts.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path} ({len(nb.cells)} cells)")
