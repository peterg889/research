#!/usr/bin/env python3
"""Build Exp 14 presentation charts notebook.

Generates 14_presentation_charts.ipynb — lightweight (no model), loads JSON results,
produces 7 figures covering soft prompt methodology, rankings, and cross-dataset transfer.

Usage:
    cd /home/jupyter/research/directed_kvcache_v4
    python3 experiments/decoder_only/14/build_presentation_charts.py
    cd experiments/decoder_only/14
    papermill 14_presentation_charts.ipynb 14_presentation_charts_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/14", exist_ok=True)

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


# =====================================================================
# Cell 0: Title
# =====================================================================
md(r"""# Exp 14 Presentation Charts

Soft prompt tuning for KV cache conditioning: 7 datasets, 4 initializations,
per-dataset vs universal, cross-dataset transfer evaluation.

**Figures:**
1. Methodology — Soft prompt training pipeline
2. Training curves — val NLL vs epoch (MS MARCO, 4 inits)
3. Full ranking — 21 soft prompt + 13 Exp 13 conditions
4. Dataset x condition heatmap
5. Init comparison — 7 datasets x 4 inits
6. Cross-dataset transfer matrix
7. Per-dataset vs universal specialization""")


# =====================================================================
# Cell 1: Setup
# =====================================================================
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../../..")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from scipy import stats

from lib.analysis import cohens_d, win_rate, paired_ttest

# --- Seaborn theme ---
sns.set_theme(style='whitegrid', context='notebook', font_scale=1.05,
              rc={
                  'figure.dpi': 150,
                  'savefig.dpi': 150,
                  'savefig.bbox': 'tight',
                  'font.family': 'sans-serif',
                  'axes.edgecolor': '.8',
                  'axes.linewidth': 0.6,
                  'grid.color': '.92',
                  'grid.linewidth': 0.5,
                  'grid.alpha': 0.7,
                  'patch.edgecolor': 'white',
                  'patch.linewidth': 0.6,
                  'xtick.major.width': 0.6,
                  'ytick.major.width': 0.6,
                  'xtick.color': '.4',
                  'ytick.color': '.4',
                  'axes.titleweight': 'medium',
                  'axes.labelweight': 'normal',
                  'axes.titlesize': 13,
                  'axes.labelsize': 11,
                  'axes.titlepad': 10,
                  'legend.framealpha': 0.9,
                  'legend.edgecolor': '.85',
                  'legend.fontsize': 10,
              })

RESULTS_DIR = Path("../../../results/decoder_only/exp14")
CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True, mode=0o777)

# --- Load data ---
summary = json.loads((RESULTS_DIR / "summary.json").read_text())
results = json.loads((RESULTS_DIR / "results.json").read_text())

transfer_path = RESULTS_DIR / "transfer_matrix.json"
has_transfer = transfer_path.exists()
if has_transfer:
    transfer = json.loads(transfer_path.read_text())
    print(f"Transfer matrix loaded: {len(transfer['per_pair'])} pairs")
else:
    transfer = None
    print("WARNING: transfer_matrix.json not found — Fig 6 will be skipped")

print(f"Summary loaded: {len(summary['rankings'])} ranked conditions")

# --- Shared constants ---
DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'gsm8k']
DS_LABELS = {
    'ms_marco': 'MS MARCO', 'squad_v2': 'SQuAD 2.0', 'triviaqa': 'TriviaQA',
    'hotpotqa': 'HotpotQA', 'drop': 'DROP', 'boolq': 'BoolQ', 'gsm8k': 'GSM8K',
}
INIT_NAMES = ['warm_comprehend', 'warm_extract', 'warm_classify', 'rand']
INIT_LABELS = {
    'warm_comprehend': 'Warm: comprehend',
    'warm_extract': 'Warm: extract',
    'warm_classify': 'Warm: classify',
    'rand': 'Random init',
}

# --- Color palette ---
# Exp 13 condition groups
GROUP_COLORS = {
    'structural':  {'table_bg': '#F0F1F3', 'bar': '#D9DCE0', 'accent': '#A0A7B1'},
    'keywords':    {'table_bg': '#FBF5EC', 'bar': '#EDE0CA', 'accent': '#C8AE82'},
    'instruction': {'table_bg': '#EDF2FA', 'bar': '#CADAEE', 'accent': '#8DB3D0'},
    'query':       {'table_bg': '#EAF5F2', 'bar': '#C4E5DC', 'accent': '#84C4B6'},
    # Exp 14 new groups
    'soft':        {'table_bg': '#F3E8FF', 'bar': '#D8B4FE', 'accent': '#A855F7'},
    'universal':   {'table_bg': '#FEF3C7', 'bar': '#FDE68A', 'accent': '#D97706'},
}

# Map Exp 13 conditions to groups
EXP13_COND_GROUP = {
    'repeat_token': 'structural', 'random': 'structural',
    'unrelated': 'structural', 'adversarial': 'structural',
    'tfidf': 'keywords', 'scrambled_comprehend': 'keywords',
    'extract': 'instruction', 'classify': 'instruction', 'comprehend': 'instruction',
    'oracle': 'query', 'llm_question': 'query',
    'ood_query': 'query', 'misleading_query': 'query',
}

# Init colors for training curves
INIT_COLORS = {
    'warm_comprehend': '#3B82F6',
    'warm_extract': '#10B981',
    'warm_classify': '#F59E0B',
    'rand': '#A855F7',
}

def stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''

# Build per-sample NLL arrays from results.json
per_sample = results['per_sample_results']
nll_arrays = {}  # ds_name -> {nll_key -> np.array}
for ds_name in DATASETS:
    nll_arrays[ds_name] = {}
    samples = per_sample[ds_name]
    for key in samples[0]:
        if key.startswith('nll_'):
            vals = [s[key] for s in samples if key in s]
            if len(vals) == len(samples):
                nll_arrays[ds_name][key] = np.array(vals)

print(f"Charts will be saved to {CHART_DIR}")
""")


# =====================================================================
# Cell 2: Fig 1 — Methodology diagram
# =====================================================================
md(r"""## Fig 1: Soft Prompt Training Pipeline

Two-panel diagram showing Phase A with learned soft prompt embeddings and
gradient flow through the KV cache back to the soft prompt parameters.""")

code(r"""# Fig 1: Methodology — Soft prompt training pipeline
with plt.style.context('default'):
    fig, axes = plt.subplots(2, 1, figsize=(15, 11),
                              gridspec_kw={'height_ratios': [4.5, 2.5], 'hspace': 0.12})

    def draw_block(ax, x, y, w, h, label, color, fontsize=9, text_color='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color)

    # === Phase A: Build and Edit ===
    ax = axes[0]
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1.5, 8.5)
    ax.axis('off')
    ax.set_title('Phase A: Build and Edit the KV Cache (with Soft Prompt)',
                 fontsize=15, fontweight='bold', pad=10, loc='left', color='#1E293B')

    # Step 1: Full concatenated input
    y = 7.0
    ax.text(-0.8, y + 0.85, 'Step 1: Concatenate', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 3.8, 0.7, 'soft_prompt  (P = 64)', '#A855F7')
    draw_block(ax, 5.2, y, 0.7, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 6.1, y, 7.0, 0.7, 'document tokens  (D)', '#3B82F6')
    ax.text(13.4, y + 0.35, 'inputs_embeds\n(not token IDs)', fontsize=9,
            color='#A855F7', style='italic', fontweight='bold')

    # Step 2: Select KV entries
    y = 5.0
    ax.text(-0.8, y + 0.85, 'Step 2: Select KV entries', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    ax.add_patch(FancyBboxPatch((1.2, y), 3.8, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#F3E8FF', edgecolor='#A855F7', linewidth=2.5, alpha=0.5, linestyle='--'))
    ax.text(3.1, y + 0.35, 'soft_prompt  (REMOVED)', ha='center', fontsize=9,
            color='#A855F7', fontweight='bold')
    ax.add_patch(FancyBboxPatch((5.2, y), 0.7, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#FECACA', edgecolor='#EF4444', linewidth=2.5, alpha=0.5, linestyle='--'))
    ax.text(5.55, y + 0.35, '\\n', ha='center', fontsize=9, color='#DC2626')
    draw_block(ax, 6.1, y, 7.0, 0.7, 'document tokens  (D)', '#3B82F6')
    ax.annotate('', xy=(6.1, y - 0.2), xytext=(1.0, y - 0.2),
                arrowprops=dict(arrowstyle='->', color='#A855F7', lw=2))
    ax.text(3.55, y - 0.45, 'skip KV indices 1 .. 65', fontsize=9, color='#A855F7', ha='center')

    # Step 3: RoPE reposition
    y = 3.0
    ax.text(-0.8, y + 0.85, 'Step 3: RoPE correct', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 7.0, 0.7, 'document tokens  (D)', '#3B82F6')
    ax.text(0.5, y - 0.3, 'pos 0', fontsize=9, color='#6366F1', ha='center', fontweight='bold')
    ax.text(4.7, y - 0.3, 'pos 1 .. D', fontsize=9, color='#3B82F6', ha='center', fontweight='bold')
    ax.annotate('rotate keys:\nold pos 66..66+D \u2192 new pos 1..D',
                xy=(8.4, y + 0.35), xytext=(9.5, y + 0.35),
                fontsize=9, color='#7C3AED', ha='left', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=1.5))

    # Step 4: Normalize
    y = 1.0
    ax.text(-0.8, y + 1.15, 'Step 4: Normalize', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 7.0, 0.7, 'document tokens  (D)', '#22C55E')
    ax.add_patch(FancyBboxPatch((-0.15, y - 0.15), 8.5, 1.0, boxstyle="round,pad=0.08",
                 facecolor='none', edgecolor='#22C55E', linewidth=2.5, linestyle='--'))
    ax.text(8.6, y + 0.55, 'enriched cache', fontsize=10, color='#22C55E',
            fontweight='bold', va='center')
    ax.text(8.6, y + 0.15, 'per-tensor:  x / (absmax/127) * (absmax/127)',
            fontsize=9, color='#475569', style='italic', va='center')

    # === Phase B + Gradient Flow ===
    ax = axes[1]
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2.5, 4.0)
    ax.axis('off')
    ax.set_title('Phase B: Score + Gradient Flow', fontsize=15,
                 fontweight='bold', pad=15, loc='left', color='#1E293B')

    # Cached doc + new input
    y = 2.0
    ax.add_patch(FancyBboxPatch((0, y), 1.0, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#6366F1', edgecolor='white', linewidth=1.5, alpha=0.9))
    ax.text(0.5, y + 0.35, 'BOS', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.add_patch(FancyBboxPatch((1.2, y), 4.5, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#22C55E', edgecolor='white', linewidth=1.5, alpha=0.65))
    ax.text(3.45, y + 0.35, 'cached doc  (pos 0..D)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(6.0, y + 0.35, '+', fontsize=16, color='#64748B', ha='center', fontweight='bold')
    draw_block(ax, 6.5, y, 0.6, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 7.3, y, 3.0, 0.7, 'query tokens', '#F59E0B')
    draw_block(ax, 10.5, y, 0.6, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 11.3, y, 3.5, 0.7, 'answer tokens', '#EF4444')
    ax.annotate('NLL computed here only', xy=(13.0, y + 0.75), xytext=(13.0, y + 1.35),
                fontsize=10, color='#EF4444', fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#EF4444', lw=2))

    # Gradient flow annotation
    y = -0.2
    ax.annotate('', xy=(-0.5, y + 0.3), xytext=(14.5, y + 0.3),
                arrowprops=dict(arrowstyle='<-', color='#A855F7', lw=2.5,
                                linestyle='--'))
    ax.text(7.0, y + 0.7, 'gradient flow', fontsize=11, color='#A855F7',
            fontweight='bold', ha='center')

    grad_steps = [
        (11.0, 'loss'),
        (8.0, 'cache grads'),
        (4.5, 'recompute Phase A'),
        (0.5, 'soft_prompt.grad'),
    ]
    for x, label in grad_steps:
        ax.text(x, y - 0.3, label, fontsize=9, color='#7C3AED', ha='center',
                fontweight='medium', style='italic')

    y = -1.5
    ax.text(7.0, y, 'Memory strategy: detach cache, backward through Phase B,\n'
            'then recompute Phase A forward + backward with saved KV gradients.',
            fontsize=9, color='#64748B', ha='center', va='center', style='italic')

    fig.savefig(CHART_DIR / 'fig_methodology.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig_methodology.png")
""")


# =====================================================================
# Cell 3: Fig 2 — Training Curves
# =====================================================================
md(r"""## Fig 2: Training Curves

Validation NLL vs epoch for MS MARCO, comparing 4 initialization strategies.
Random init starts high but drops furthest, outperforming all warm starts.""")

code(r"""# Fig 2: Training curves — MS MARCO, 4 inits
training = summary['training_summary']
ds_name = 'ms_marco'

rows = []
for init_name in INIT_NAMES:
    ts = training[ds_name][init_name]
    for epoch, val_nll in enumerate(ts['val_losses']):
        rows.append({
            'Epoch': epoch,
            'Val NLL': val_nll,
            'Init': INIT_LABELS[init_name],
            'init_key': init_name,
        })

curve_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 5))
for init_name in INIT_NAMES:
    sub = curve_df[curve_df['init_key'] == init_name]
    ax.plot(sub['Epoch'], sub['Val NLL'], marker='o', markersize=5,
            linewidth=1.8, color=INIT_COLORS[init_name],
            label=INIT_LABELS[init_name])

    # Mark best epoch
    best_ep = training[ds_name][init_name]['best_epoch']
    best_val = training[ds_name][init_name]['best_val_nll']
    ax.scatter([best_ep], [best_val], s=80, color=INIT_COLORS[init_name],
               zorder=5, edgecolor='white', linewidth=1.5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation NLL')
ax.set_title(f'Training curves: MS MARCO (4 initializations)', pad=12)
ax.legend(fontsize=10, framealpha=0.9)
sns.despine()

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig1_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig1_training_curves.png")
""")


# =====================================================================
# Cell 4: Fig 3 — Full Ranking
# =====================================================================
md(r"""## Fig 3: Full Condition Ranking

All conditions ranked by pooled Cohen's d across 7 datasets. Soft prompt (purple)
and universal (amber) conditions compared against Exp 13 static prefixes.

Only nonorm variants shown (norm delta < 0.002 — normalization has negligible
effect on learned soft prompts).""")

code(r"""# Fig 3: Full ranking — soft prompts + Exp 13 conditions
rankings = summary['rankings']

# Filter: keep nonorm variants of soft/univ (drop norm versions since delta < 0.002)
# Keep all Exp 13 conditions
filtered = []
seen_base = set()
for r in rankings:
    cond = r['condition']
    if r['source'] == 'Exp13':
        filtered.append(r)
    elif cond.endswith('_nonorm'):
        filtered.append(r)
        seen_base.add(cond.replace('_nonorm', ''))
    elif cond not in seen_base and not cond.endswith('_nonorm'):
        # Keep norm version only if nonorm version doesn't exist
        nonorm_exists = any(r2['condition'] == cond + '_nonorm' for r2 in rankings)
        if not nonorm_exists:
            filtered.append(r)

# Sort by pooled_d
filtered.sort(key=lambda r: r['pooled_d'])

# Display labels
COND_LABELS_FULL = {
    'comprehend': 'Comprehend', 'extract': 'Extract', 'classify': 'Classify',
    'llm_question': 'LLM question', 'oracle': 'Oracle (query)',
    'tfidf': 'TF-IDF keywords', 'scrambled_comprehend': 'Scrambled comprehend',
    'unrelated': 'Unrelated text', 'adversarial': 'Adversarial',
    'random': 'Random tokens', 'repeat_token': 'Repeat token',
    'ood_query': 'OOD query', 'misleading_query': 'Misleading query',
    'soft_rand_nonorm': 'Soft: random init',
    'soft_warm_comprehend_nonorm': 'Soft: warm comprehend',
    'soft_warm_extract_nonorm': 'Soft: warm extract',
    'soft_warm_classify_nonorm': 'Soft: warm classify',
    'univ_rand_nonorm': 'Univ: random init',
    'univ_warm_comprehend_nonorm': 'Univ: warm comprehend',
    'univ_warm_extract_nonorm': 'Univ: warm extract',
    'univ_warm_classify_nonorm': 'Univ: warm classify',
    # norm versions (fallback)
    'soft_rand': 'Soft: random init',
    'soft_warm_comprehend': 'Soft: warm comprehend',
    'soft_warm_extract': 'Soft: warm extract',
    'soft_warm_classify': 'Soft: warm classify',
    'univ_rand': 'Univ: random init',
    'univ_warm_comprehend': 'Univ: warm comprehend',
    'univ_warm_extract': 'Univ: warm extract',
    'univ_warm_classify': 'Univ: warm classify',
}

def get_bar_color(r):
    if r['source'] == 'Soft':
        return GROUP_COLORS['soft']['bar']
    elif r['source'] == 'Univ':
        return GROUP_COLORS['universal']['bar']
    elif r['source'] == 'Exp13':
        cond = r['condition']
        grp = EXP13_COND_GROUP.get(cond, 'structural')
        return GROUP_COLORS[grp]['bar']
    return '#D9DCE0'

labels = [COND_LABELS_FULL.get(r['condition'], r['condition']) for r in filtered]
colors = [get_bar_color(r) for r in filtered]
d_vals = [r['pooled_d'] for r in filtered]
win_vals = [r['pooled_win'] for r in filtered]

fig, ax = plt.subplots(figsize=(11, 9))
y_pos = np.arange(len(filtered))
ax.barh(y_pos, d_vals, color=colors, edgecolor='white', linewidth=0.6, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color='.6', linewidth=0.5, zorder=0)
sns.despine(left=True, bottom=True)

for i, r in enumerate(filtered):
    d_val = r['pooled_d']
    w = r['pooled_win']
    x_text = max(d_val, 0) + 0.01
    ax.text(x_text, i, f"d = {d_val:+.2f}   {w:.0%}",
            va='center', ha='left', fontsize=8, color='.3')

# Legend
legend_elements = [
    Line2D([0], [0], color=GROUP_COLORS['soft']['bar'], lw=8, label='Per-dataset soft prompt'),
    Line2D([0], [0], color=GROUP_COLORS['universal']['bar'], lw=8, label='Universal soft prompt'),
    Line2D([0], [0], color=GROUP_COLORS['instruction']['bar'], lw=8, label='Exp 13: instruction'),
    Line2D([0], [0], color=GROUP_COLORS['query']['bar'], lw=8, label='Exp 13: query'),
    Line2D([0], [0], color=GROUP_COLORS['keywords']['bar'], lw=8, label='Exp 13: keywords'),
    Line2D([0], [0], color=GROUP_COLORS['structural']['bar'], lw=8, label='Exp 13: structural'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

ax.set_xlabel("Pooled Cohen's d  (positive = condition helps)")
ax.set_ylabel('')
ax.set_title('Full condition ranking: soft prompts vs static prefixes  (N = 1120)', pad=12)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig2_full_ranking.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig2_full_ranking.png")
""")


# =====================================================================
# Cell 5: Fig 4 — Heatmap
# =====================================================================
md(r"""## Fig 4: Dataset x Condition Heatmap

Cohen's d for key conditions across 7 datasets. Includes best Exp 13 conditions
plus all soft prompt variants.""")

code(r"""# Fig 4: Heatmap — datasets x conditions
# Select key conditions to show
hm_conditions = [
    # Exp 13 top conditions
    ('comprehend', 'nll_comprehend', 'Exp13'),
    ('adversarial', 'nll_adversarial', 'Exp13'),
    ('extract', 'nll_extract', 'Exp13'),
    ('tfidf', 'nll_tfidf', 'Exp13'),
    ('random', 'nll_random', 'Exp13'),
    # Per-dataset soft prompts (nonorm)
    ('soft_rand_nonorm', 'nll_soft_rand_nonorm', 'Soft'),
    ('soft_warm_comprehend_nonorm', 'nll_soft_warm_comprehend_nonorm', 'Soft'),
    # Universal soft prompts (nonorm)
    ('univ_rand_nonorm', 'nll_univ_rand_nonorm', 'Univ'),
    ('univ_warm_comprehend_nonorm', 'nll_univ_warm_comprehend_nonorm', 'Univ'),
]

hm_labels = {
    'comprehend': 'Comprehend',
    'adversarial': 'Adversarial',
    'extract': 'Extract',
    'tfidf': 'TF-IDF',
    'random': 'Random tokens',
    'soft_rand_nonorm': 'Soft: rand',
    'soft_warm_comprehend_nonorm': 'Soft: comprehend',
    'univ_rand_nonorm': 'Univ: rand',
    'univ_warm_comprehend_nonorm': 'Univ: comprehend',
}

# Compute d for each (ds, condition) pair
hm_data = []
hm_annot = []
for ds_name in DATASETS:
    row_d = []
    row_a = []
    for cond_name, nll_key, source in hm_conditions:
        if nll_key in nll_arrays[ds_name] and 'nll_bare' in nll_arrays[ds_name]:
            diff = nll_arrays[ds_name]['nll_bare'] - nll_arrays[ds_name][nll_key]
            d = cohens_d(diff)
            _, p = paired_ttest(diff)
            row_d.append(d)
            row_a.append(f"{d:+.2f}{stars(p)}")
        else:
            row_d.append(0.0)
            row_a.append("N/A")
    hm_data.append(row_d)
    hm_annot.append(row_a)

col_labels = [hm_labels[c[0]] for c in hm_conditions]
row_labels = [DS_LABELS[ds] for ds in DATASETS]

matrix = pd.DataFrame(hm_data, index=row_labels, columns=col_labels)
annot_df = pd.DataFrame(hm_annot, index=row_labels, columns=col_labels)

vabs = max(abs(matrix.values.min()), abs(matrix.values.max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-vabs - 0.1, vmax=vabs + 0.1)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(matrix, annot=annot_df, fmt='', cmap='RdBu_r', norm=norm,
            linewidths=2, linecolor='white', ax=ax, square=False,
            cbar_kws={'shrink': 0.75, 'label': "Cohen's d  (positive = helps)"},
            annot_kws={'fontsize': 8, 'fontweight': 'normal'})

ax.set_title("Task specificity: Cohen's d by dataset and condition", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig3_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig3_heatmap.png")
""")


# =====================================================================
# Cell 6: Fig 5 — Init Comparison
# =====================================================================
md(r"""## Fig 5: Initialization Comparison

Grouped horizontal bars: 7 datasets x 4 initialization strategies (per-dataset scope).
Random initialization wins on 6/7 datasets despite starting from scratch.""")

code(r"""# Fig 5: Init comparison — 7 datasets x 4 inits (per-dataset soft prompts)
init_rows = []
for ds_name in DATASETS:
    for init_name in INIT_NAMES:
        nll_key = f'nll_soft_{init_name}_nonorm'
        if nll_key in nll_arrays[ds_name] and 'nll_bare' in nll_arrays[ds_name]:
            diff = nll_arrays[ds_name]['nll_bare'] - nll_arrays[ds_name][nll_key]
            d = cohens_d(diff)
            w = win_rate(diff)
        else:
            d = 0.0
            w = 0.5
        init_rows.append({
            'Dataset': DS_LABELS[ds_name],
            'Init': INIT_LABELS[init_name],
            'init_key': init_name,
            "Cohen's d": d,
            'win': w,
        })

init_df = pd.DataFrame(init_rows)

fig, ax = plt.subplots(figsize=(11, 7))
init_palette = {INIT_LABELS[k]: v for k, v in INIT_COLORS.items()}
sns.barplot(data=init_df, y='Dataset', x="Cohen's d", hue='Init',
            palette=init_palette, ax=ax, saturation=0.85,
            edgecolor='white', linewidth=0.6)

ax.axvline(0, color='.6', linewidth=0.5, zorder=0)
sns.despine(left=True, bottom=True)

ax.set_xlabel("Cohen's d  (positive = condition helps)")
ax.set_ylabel('')
ax.set_title('Per-dataset soft prompts: initialization comparison', pad=12)
ax.legend(title='Initialization', fontsize=9, title_fontsize=10, framealpha=0.9,
          loc='lower right')

# Count wins for each init
wins_by_init = {}
for ds_name in DATASETS:
    best_init = None
    best_d = -999
    for init_name in INIT_NAMES:
        sub = init_df[(init_df['Dataset'] == DS_LABELS[ds_name]) &
                      (init_df['init_key'] == init_name)]
        if len(sub) > 0:
            d_val = sub["Cohen's d"].values[0]
            if d_val > best_d:
                best_d = d_val
                best_init = init_name
    wins_by_init[best_init] = wins_by_init.get(best_init, 0) + 1

wins_text = ', '.join(f'{INIT_LABELS[k]}: {v}' for k, v in sorted(wins_by_init.items(), key=lambda x: -x[1]))
ax.text(0.02, 0.02, f'Best init wins: {wins_text}',
        transform=ax.transAxes, fontsize=9, color='.5', style='italic')

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig4_init_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig4_init_comparison.png")
""")


# =====================================================================
# Cell 7: Fig 6 — Transfer Matrix
# =====================================================================
md(r"""## Fig 6: Cross-Dataset Transfer Matrix

How well do soft prompts trained on one dataset generalize to others?
Rows = source dataset (where the soft prompt was trained), columns = target dataset.
Diagonal = same-dataset performance (expected best).""")

code(r"""# Fig 6: Cross-dataset transfer matrix (7x7 + universal row)
if not has_transfer:
    print("SKIPPED: transfer_matrix.json not found")
else:
    SOURCE_NAMES = DATASETS + ['universal']
    SOURCE_LABELS = {**DS_LABELS, 'universal': 'Universal'}

    # Build matrix
    tm_data = []
    tm_annot = []
    for source in SOURCE_NAMES:
        row_d = []
        row_a = []
        for target in DATASETS:
            d = transfer['matrix_d'][source][target]
            p = transfer['matrix_p'][source][target]
            row_d.append(d)
            row_a.append(f"{d:+.2f}{stars(p)}")
        tm_data.append(row_d)
        tm_annot.append(row_a)

    row_labels = [SOURCE_LABELS[s] for s in SOURCE_NAMES]
    col_labels = [DS_LABELS[ds] for ds in DATASETS]

    tm_matrix = pd.DataFrame(tm_data, index=row_labels, columns=col_labels)
    tm_annot_df = pd.DataFrame(tm_annot, index=row_labels, columns=col_labels)

    vabs = max(abs(tm_matrix.values.min()), abs(tm_matrix.values.max()))
    tm_norm = TwoSlopeNorm(vcenter=0, vmin=-vabs - 0.1, vmax=vabs + 0.1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(tm_matrix, annot=tm_annot_df, fmt='', cmap='RdBu_r', norm=tm_norm,
                linewidths=2.5, linecolor='white', ax=ax, square=True,
                cbar_kws={'shrink': 0.7, 'label': "Cohen's d vs bare"},
                annot_kws={'fontsize': 9, 'fontweight': 'normal'})

    ax.set_title("Cross-dataset transfer: source (row) \u2192 target (col)", pad=12)
    ax.set_xlabel('Target dataset')
    ax.set_ylabel('Source prompt')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

    # Highlight diagonal with thicker border
    for i in range(len(DATASETS)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#1E293B',
                                   linewidth=2.5, clip_on=False))

    fig.tight_layout()
    fig.savefig(CHART_DIR / 'fig5_transfer_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved fig5_transfer_matrix.png")

    # Print summary stats
    diag_vals = [tm_matrix.values[i, i] for i in range(len(DATASETS))]
    off_diag_vals = [tm_matrix.values[i, j] for i in range(len(DATASETS))
                     for j in range(len(DATASETS)) if i != j]
    univ_vals = tm_matrix.values[-1, :].tolist()
    print(f"  Diagonal (same-ds):     mean d = {np.mean(diag_vals):+.3f}")
    print(f"  Off-diagonal (transfer): mean d = {np.mean(off_diag_vals):+.3f}")
    print(f"  Universal:              mean d = {np.mean(univ_vals):+.3f}")
    print(f"  Specialization gap:     {np.mean(diag_vals) - np.mean(off_diag_vals):+.3f}")
""")


# =====================================================================
# Cell 8: Fig 7 — Per-dataset vs Universal
# =====================================================================
md(r"""## Fig 7: Per-Dataset vs Universal Specialization

For each dataset, compare the per-dataset rand soft prompt (purple) against the
universal rand soft prompt (amber). Shows where specialization helps and where
the universal prompt suffices.""")

code(r"""# Fig 7: Per-dataset vs universal — dumbbell chart
spec_rows = []
for ds_name in DATASETS:
    # Per-dataset rand (nonorm)
    per_key = 'nll_soft_rand_nonorm'
    univ_key = 'nll_univ_rand_nonorm'

    if per_key in nll_arrays[ds_name] and univ_key in nll_arrays[ds_name] and 'nll_bare' in nll_arrays[ds_name]:
        diff_per = nll_arrays[ds_name]['nll_bare'] - nll_arrays[ds_name][per_key]
        diff_univ = nll_arrays[ds_name]['nll_bare'] - nll_arrays[ds_name][univ_key]
        d_per = cohens_d(diff_per)
        d_univ = cohens_d(diff_univ)
        w_per = win_rate(diff_per)
        w_univ = win_rate(diff_univ)
    else:
        d_per = 0.0
        d_univ = 0.0
        w_per = 0.5
        w_univ = 0.5

    spec_rows.append({
        'Dataset': DS_LABELS[ds_name],
        'ds_key': ds_name,
        'd_per': d_per,
        'd_univ': d_univ,
        'delta': d_per - d_univ,
    })

spec_df = pd.DataFrame(spec_rows)

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(spec_df))

# Dumbbell: line connecting per-dataset and universal, dots at each end
for i, row in spec_df.iterrows():
    ax.plot([row['d_per'], row['d_univ']], [i, i], color='.75', linewidth=1.5, zorder=1)

ax.scatter(spec_df['d_per'], y_pos, s=80, color=GROUP_COLORS['soft']['accent'],
           zorder=3, label='Per-dataset', edgecolor='white', linewidth=1)
ax.scatter(spec_df['d_univ'], y_pos, s=80, color=GROUP_COLORS['universal']['accent'],
           zorder=3, label='Universal', edgecolor='white', linewidth=1)

# Annotations
for i, row in spec_df.iterrows():
    d_max = max(row['d_per'], row['d_univ'])
    delta = row['delta']
    winner = 'per-ds' if delta > 0.01 else ('univ' if delta < -0.01 else 'tie')
    color = GROUP_COLORS['soft']['accent'] if winner == 'per-ds' else (
        GROUP_COLORS['universal']['accent'] if winner == 'univ' else '.5')
    ax.text(d_max + 0.03, i, f"\u0394={delta:+.2f}", va='center', ha='left',
            fontsize=9, color=color, fontweight='bold' if abs(delta) > 0.05 else 'normal')

ax.set_yticks(y_pos)
ax.set_yticklabels(spec_df['Dataset'])
ax.axvline(0, color='.6', linewidth=0.5, zorder=0)
sns.despine(left=True, bottom=True)

ax.set_xlabel("Cohen's d  (positive = condition helps)")
ax.set_ylabel('')
ax.set_title('Per-dataset specialization vs universal soft prompt  (rand init)', pad=12)
ax.legend(fontsize=10, framealpha=0.9, loc='lower right')

# Summary
n_per_wins = sum(1 for _, r in spec_df.iterrows() if r['delta'] > 0.01)
n_univ_wins = sum(1 for _, r in spec_df.iterrows() if r['delta'] < -0.01)
n_ties = len(spec_df) - n_per_wins - n_univ_wins
ax.text(0.02, 0.02, f'Per-dataset wins: {n_per_wins},  Universal wins: {n_univ_wins},  Ties: {n_ties}',
        transform=ax.transAxes, fontsize=9, color='.5', style='italic')

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig6_specialization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig6_specialization.png")
""")


# =====================================================================
# Cell 9: Summary
# =====================================================================
code(r"""# Summary
print("=" * 70)
print("ALL CHARTS GENERATED")
print("=" * 70)

import os
chart_files = sorted(os.listdir(CHART_DIR))
for f in chart_files:
    size = (CHART_DIR / f).stat().st_size / 1024
    print(f"  {f:<40} {size:>6.0f} KB")
print(f"\nTotal: {len(chart_files)} files in {CHART_DIR}")
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/decoder_only/14/14_presentation_charts.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
