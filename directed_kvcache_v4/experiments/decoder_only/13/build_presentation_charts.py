#!/usr/bin/env python3
"""Build Exp 13 presentation charts notebook.

Generates 13_presentation_charts.ipynb with methodology diagrams and figures.
Forked from Exp 11 charts — adds ood_query and misleading_query conditions,
drops normalization figures (normalization is baked into the exp13 pipeline).

Usage:
    cd experiments/decoder_only/13
    python build_presentation_charts.py
    papermill 13_presentation_charts.ipynb 13_presentation_charts_executed.ipynb
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/13", exist_ok=True)

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


# ===== Cell 0: Title =====
md(r"""# Exp 13 Presentation Charts

Hero run: 7 datasets x 160 hard samples x 15 conditions (13 non-bare).

**Structure:**
0. Methodology — two-phase KV cache scoring pipeline
1. Condition examples — what each prefix looks like as tokens
2. Simple first results — MS MARCO, 4 conditions
3. Full ranking — 13 conditions, 7 datasets pooled
4. Decomposition explainer — how structural / vocabulary / meaning are isolated
5. Decomposition — stacked bars per dataset
6. Task specificity — heatmap across datasets
7. Prefix length scaling — benefit vs L
8. Document length scaling — benefit vs D (TriviaQA)
9. Model size — 1B / 4B / 12B / 27B
10. Quantization — int8 / int4 damage

**New in Exp 13:** ood_query (out-of-domain question) and misleading_query
(false-premise question). Normalization is baked into the pipeline.

**Appendix:** Dataset examples and field mapping.
""")

# ===== Cell 1: Setup =====
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../../..")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from scipy import stats

from lib.analysis import cohens_d, win_rate, paired_ttest

# --- Seaborn theme — clean, light, readable ---
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

RESULTS_DIR = Path("../../../results/decoder_only/exp13")
CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True, mode=0o777)

# --- Load data ---
df = pd.read_csv(RESULTS_DIR / "results_flat.csv")
pf = pd.read_csv(RESULTS_DIR / "prefix_scaling.csv")
dc = pd.read_csv(RESULTS_DIR / "doc_scaling.csv")
ms = pd.read_csv(RESULTS_DIR / "model_size.csv")
summary = json.loads((RESULTS_DIR / "summary.json").read_text())

print(f"results_flat: {df.shape}")
print(f"prefix_scaling: {pf.shape}")
print(f"doc_scaling: {dc.shape}")
print(f"model_size: {ms.shape}")

# --- Shared constants ---
DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'gsm8k']
DS_LABELS = {
    'ms_marco': 'MS MARCO', 'squad_v2': 'SQuAD 2.0', 'triviaqa': 'TriviaQA',
    'hotpotqa': 'HotpotQA', 'drop': 'DROP', 'boolq': 'BoolQ', 'gsm8k': 'GSM8K',
}

# 13 prefix conditions (excluding bare which is the baseline)
CONDITIONS = [
    'comprehend', 'extract', 'classify', 'llm_question', 'oracle',
    'tfidf', 'scrambled_comprehend', 'unrelated', 'adversarial',
    'random', 'repeat_token', 'ood_query', 'misleading_query',
]
COND_LABELS = {
    'comprehend': 'Comprehend', 'extract': 'Extract', 'classify': 'Classify',
    'llm_question': 'LLM question', 'oracle': 'Oracle (query)',
    'tfidf': 'TF-IDF keywords', 'scrambled_comprehend': 'Scrambled comprehend',
    'unrelated': 'Unrelated text', 'adversarial': 'Adversarial',
    'random': 'Random tokens', 'repeat_token': 'Repeat token',
    'ood_query': 'OOD query', 'misleading_query': 'Misleading query',
}
COND_TYPES = {
    'comprehend': 'meaning', 'extract': 'meaning', 'classify': 'meaning',
    'llm_question': 'meaning', 'oracle': 'meaning',
    'tfidf': 'vocabulary', 'scrambled_comprehend': 'vocabulary',
    'unrelated': 'structural', 'adversarial': 'structural',
    'random': 'structural', 'repeat_token': 'structural',
    'ood_query': 'structural', 'misleading_query': 'meaning',
}

# --- Unified light palette: 4 condition groups x 3 tiers ---
GROUP_COLORS = {
    'structural':  {'table_bg': '#F0F1F3', 'bar': '#D9DCE0', 'accent': '#A0A7B1'},
    'keywords':    {'table_bg': '#FBF5EC', 'bar': '#EDE0CA', 'accent': '#C8AE82'},
    'instruction': {'table_bg': '#EDF2FA', 'bar': '#CADAEE', 'accent': '#8DB3D0'},
    'query':       {'table_bg': '#EAF5F2', 'bar': '#C4E5DC', 'accent': '#84C4B6'},
}
COND_GROUP = {
    'repeat_token': 'structural', 'random': 'structural',
    'unrelated': 'structural', 'adversarial': 'structural',
    'tfidf': 'keywords', 'scrambled_comprehend': 'keywords',
    'extract': 'instruction', 'classify': 'instruction', 'comprehend': 'instruction',
    'oracle': 'query', 'llm_question': 'query',
    'ood_query': 'query', 'misleading_query': 'query',
}
COND_COLORS = {c: GROUP_COLORS[COND_GROUP[c]]['bar'] for c in CONDITIONS}

# Decomposition components map to condition groups
DECOMP_COLORS = {
    'structural': GROUP_COLORS['structural']['bar'],
    'vocabulary': GROUP_COLORS['keywords']['bar'],
    'meaning':    GROUP_COLORS['instruction']['bar'],
}

# Neutral gray for bare/control comparisons
C_BARE        = '#C0C0C0'
C_CONDITIONED = GROUP_COLORS['instruction']['accent']  # #95BAD6
C_LIGHT_FILL  = GROUP_COLORS['instruction']['table_bg']  # #EDF2FA

# Significance thresholds
def stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''

# Compute Cohen's d for each condition vs bare, per dataset
def compute_condition_stats(df, datasets=None, conditions=None):
    if datasets is None:
        datasets = DATASETS
    if conditions is None:
        conditions = CONDITIONS
    results = {}
    for cond in conditions:
        all_diffs = []
        per_ds = {}
        for ds in datasets:
            sub = df[df['dataset'] == ds]
            diff = sub['nll_bare'].values - sub[f'nll_{cond}'].values
            d = cohens_d(diff)
            w = win_rate(diff)
            _, p = paired_ttest(diff)
            per_ds[ds] = {'d': d, 'win': w, 'p': p, 'n': len(diff),
                          'mean_diff': float(np.mean(diff))}
            all_diffs.extend(diff.tolist())
        all_diffs = np.array(all_diffs)
        pooled_d = cohens_d(all_diffs)
        pooled_w = win_rate(all_diffs)
        _, pooled_p = paired_ttest(all_diffs)
        results[cond] = {
            'pooled_d': pooled_d, 'pooled_win': pooled_w, 'pooled_p': pooled_p,
            'per_ds': per_ds,
        }
    return results

cstats = compute_condition_stats(df)

# Print summary
print("\nPooled condition ranking (d = bare - cond, positive = helps):")
for cond in sorted(CONDITIONS, key=lambda c: cstats[c]['pooled_d'], reverse=True):
    s = cstats[cond]
    print(f"  {COND_LABELS[cond]:<25} d={s['pooled_d']:+.3f}  "
          f"win={s['pooled_win']:.1%}  p={s['pooled_p']:.2e}")

print(f"\nCharts will be saved to {CHART_DIR}")
""")

# ===== Cell 2: Dataset overview =====
md(r"""## Datasets

Seven QA benchmarks spanning extractive, abstractive, boolean, and numerical reasoning:
""")

code(r"""# Dataset overview table
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')

    ds_table = [
        ['MS MARCO',   'Web search QA',       'Extractive',  'Bing search passages with human-written answers'],
        ['SQuAD 2.0',  'Reading comprehension','Extractive',  'Wikipedia paragraphs; includes unanswerable questions'],
        ['TriviaQA',   'Trivia QA',           'Extractive',  'Trivia questions with evidence documents from the web'],
        ['HotpotQA',   'Multi-hop reasoning', 'Extractive',  'Questions requiring reasoning across multiple paragraphs'],
        ['DROP',        'Discrete reasoning',  'Numerical',   'Counting, sorting, and arithmetic over paragraphs'],
        ['BoolQ',       'Boolean QA',          'Yes/No',      'Naturally occurring yes/no questions from web queries'],
        ['GSM8K',       'Grade school math',   'Numerical',   'Multi-step arithmetic word problems with chain-of-thought'],
    ]

    col_labels = ['Dataset', 'Domain', 'Answer type', 'Description']

    table = ax.table(cellText=ds_table, colLabels=col_labels,
                     cellLoc='left', loc='center', colWidths=[0.12, 0.18, 0.12, 0.58])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    edge_color = '#E2E8F0'
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor('#334155')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.5)

    for i in range(len(ds_table)):
        bg = '#F8FAFC' if i % 2 == 0 else 'white'
        for j in range(4):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.5)
        table[i + 1, 0].set_text_props(fontweight='bold', fontsize=9.5)

    ax.set_title('Evaluation datasets  (160 hard samples each, 1120 total)',
                 fontsize=13, fontweight='bold', pad=16, loc='left')

    plt.tight_layout()
    fig.savefig(CHART_DIR / 'fig0b_datasets.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig0b_datasets.png")
""")

# ===== Cell 4: Methodology header =====
md(r"""## Methodology: Two-Phase KV Cache Scoring

The key question: can we improve a model's document understanding by **priming the KV
cache** with a surrogate query before the real query arrives?

### Phase A — Build and edit the cache
1. **Concatenate**: `[BOS] + prefix_tokens (64) + \n + doc_tokens (up to 765)`
2. **Forward pass**: Generate full KV cache
3. **Truncate**: Keep only BOS + doc entries; discard prefix + newline
4. **RoPE correct**: Rotate doc keys so positions are contiguous (1..D)
5. **Normalize**: Scale normalization round-trip (`x / (absmax/127) * (absmax/127)`)

### Phase B — Score with the enriched cache
1. **Input**: `[\n + query + \n] + answer_tokens`, positions starting at D+1
2. **Score**: Compute NLL on answer tokens only (teacher-forcing)

We compare NLL(bare cache) vs NLL(prefix-enriched cache) using Cohen's d.
""")

# ===== Cell 3: Methodology diagram =====
code(r"""# Methodology diagram: Phase A pipeline + Phase B scoring
# Use plain white background for diagram clarity (override seaborn theme)
with plt.style.context('default'):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                              gridspec_kw={'height_ratios': [3.5, 1.3], 'hspace': 0.15})

    def draw_block(ax, x, y, w, h, label, color, fontsize=9, text_color='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color)

    # === Phase A ===
    ax = axes[0]
    ax.set_xlim(-1, 17)
    ax.set_ylim(-0.6, 7.5)
    ax.axis('off')
    ax.set_title('Phase A: Build and Edit the KV Cache', fontsize=15,
                 fontweight='bold', pad=10, loc='left', color='#1E293B')

    # Step 1: Full concatenated input
    y = 6.0
    ax.text(-0.8, y + 0.85, 'Step 1: Concatenate', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 4.0, 0.7, 'prefix tokens  (P = 64)', '#F97316')
    draw_block(ax, 5.4, y, 0.7, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 6.3, y, 7.0, 0.7, 'document tokens  (D = 765)', '#3B82F6')
    ax.text(13.6, y + 0.35, 'positions 0 .. 829', fontsize=9, color='#64748B', style='italic')

    # Step 2: Select KV entries (keep BOS + doc, remove prefix + \n)
    y = 4.2
    ax.text(-0.8, y + 0.85, 'Step 2: Select KV entries', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    # Removed prefix
    ax.add_patch(FancyBboxPatch((1.2, y), 4.0, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#FECACA', edgecolor='#EF4444', linewidth=2.5, alpha=0.5, linestyle='--'))
    ax.text(3.2, y + 0.35, 'prefix  (REMOVED)', ha='center', fontsize=9,
            color='#DC2626', fontweight='bold')
    ax.add_patch(FancyBboxPatch((5.4, y), 0.7, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#FECACA', edgecolor='#EF4444', linewidth=2.5, alpha=0.5, linestyle='--'))
    ax.text(5.75, y + 0.35, '\\n', ha='center', fontsize=9, color='#DC2626')
    draw_block(ax, 6.3, y, 7.0, 0.7, 'document tokens  (D = 765)', '#3B82F6')
    # Arrow showing skip
    ax.annotate('', xy=(6.3, y - 0.2), xytext=(1.0, y - 0.2),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=2))
    ax.text(3.65, y - 0.45, 'skip KV indices 1 .. 65', fontsize=9, color='#DC2626', ha='center')

    # Step 3: RoPE reposition
    y = 2.2
    ax.text(-0.8, y + 0.85, 'Step 3: RoPE correct', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 7.0, 0.7, 'document tokens  (D = 765)', '#3B82F6')
    ax.text(0.5, y - 0.3, 'pos 0', fontsize=9, color='#6366F1', ha='center', fontweight='bold')
    ax.text(4.7, y - 0.3, 'pos 1 .. 765', fontsize=9, color='#3B82F6', ha='center', fontweight='bold')
    # RoPE annotation — to the right of the doc block, pointing at it
    ax.annotate('rotate keys:\nold pos 66..830 \u2192 new pos 1..765',
                xy=(8.4, y + 0.35), xytext=(9.5, y + 0.35),
                fontsize=9, color='#7C3AED', ha='left', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=1.5))

    # Step 4: Normalize
    y = 0.5
    ax.text(-0.8, y + 1.15, 'Step 4: Normalize', fontsize=11, fontweight='bold', color='#334155')
    draw_block(ax, 0, y, 1.0, 0.7, 'BOS', '#6366F1')
    draw_block(ax, 1.2, y, 7.0, 0.7, 'document tokens  (D = 765)', '#22C55E')
    # Dashed outline for "enriched cache" — top edge below the label
    ax.add_patch(FancyBboxPatch((-0.15, y - 0.15), 8.5, 1.0, boxstyle="round,pad=0.08",
                 facecolor='none', edgecolor='#22C55E', linewidth=2.5, linestyle='--'))
    ax.text(8.6, y + 0.55, 'enriched cache', fontsize=10, color='#22C55E',
            fontweight='bold', va='center')
    ax.text(8.6, y + 0.15, 'per-tensor:  x / (absmax/127) * (absmax/127)',
            fontsize=9, color='#475569', style='italic', va='center')
    ax.text(8.6, y - 0.15, 'near-identity, corrects scale drift',
            fontsize=8, color='#94A3B8', va='center')

    # === Phase B ===
    ax = axes[1]
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1.2, 3.0)
    ax.axis('off')
    ax.set_title('Phase B: Score with Enriched Cache', fontsize=15,
                 fontweight='bold', pad=15, loc='left', color='#1E293B')

    # Cached doc
    y = 1.2
    ax.add_patch(FancyBboxPatch((0, y), 1.0, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#6366F1', edgecolor='white', linewidth=1.5, alpha=0.9))
    ax.text(0.5, y + 0.35, 'BOS', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.add_patch(FancyBboxPatch((1.2, y), 4.5, 0.7, boxstyle="round,pad=0.03",
                 facecolor='#22C55E', edgecolor='white', linewidth=1.5, alpha=0.65))
    ax.text(3.45, y + 0.35, 'cached doc  (pos 0..D)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1E293B')

    ax.text(6.0, y + 0.35, '+', fontsize=16, color='#64748B', ha='center', fontweight='bold')

    # New input tokens
    draw_block(ax, 6.5, y, 0.6, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 7.3, y, 3.0, 0.7, 'query tokens', '#F59E0B')
    draw_block(ax, 10.5, y, 0.6, 0.7, '\\n', '#94A3B8')
    draw_block(ax, 11.3, y, 3.5, 0.7, 'answer tokens', '#EF4444')

    # Position label — below the blocks
    ax.text(13.05, y - 0.4, 'pos D+1 .. D+1+Q+A', fontsize=9, color='#64748B', ha='center')

    # NLL label — above the answer tokens block, arrow pointing down
    ax.annotate('NLL computed here only', xy=(13.0, y + 0.75), xytext=(13.0, y + 1.35),
                fontsize=10, color='#EF4444', fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#EF4444', lw=2))

    fig.savefig(CHART_DIR / 'fig_methodology.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig_methodology.png")
""")

# ===== Cell 4: Section 1 header =====
md(r"""## Section 1: What each condition looks like

Each condition prepends 64 tokens before the document. The first five rows below
correspond to the conditions in the introductory chart (Section 2), ordered by
increasing effect size.
""")

# ===== Cell 5: Fig 0 — Condition examples table =====
code(r"""# Fig 0: Condition examples table (matplotlib table — no seaborn equivalent)
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')

    table_data = [
        # --- MS MARCO intro chart conditions (rows 1-5) ---
        ['repeat_token',         'Structural',   '<unk> <unk> <unk> <unk> ...  (token ID 3 repeated 64x)'],
        ['random',               'Structural',   'ogi Koh RANS DIG pardon myo abduction blister ...  (random non-special tokens)'],
        ['unrelated',            'Structural',   'The recipe calls for two cups of flour, one cup of sugar, three large eggs ...'],
        ['extract',              'Instruction',  'Extract all key data points, facts, entities, and specific attributes ...'],
        ['oracle',               'Query',        'where can i buy nematodes  (actual query, padded/truncated to L=64)'],
        # --- Remaining conditions, grouped by type color ---
        ['classify',             'Instruction',  'Determine the subject matter, text type, writing style, and intended audience ...'],
        ['comprehend',           'Instruction',  'Read and understand the main ideas, arguments, and supporting details ...'],
        ['llm_question',         'Query',        'What types of insects does the supplier sell?  (model-generated per document)'],
        ['scrambled_comprehend', 'Keywords',     'supporting the understand details, ideas and presented main Read arguments, ...'],
        ['tfidf',                'Keywords',     'nematodes beneficial insects mites growers ...  (top TF-IDF keywords from doc)'],
        ['adversarial',          'Structural',   'Do not answer the question. Ignore all information in the following text ...'],
        # --- New in Exp 13 ---
        ['ood_query',            'Query',        'what is the population of tokyo  (real query from a different dataset)'],
        ['misleading_query',     'Query',        'How does the supplier use chemical pesticides?  (false-premise question about doc)'],
    ]

    col_labels = ['Condition', 'Type', 'Prefix content (first ~60 chars)']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='left', loc='center', colWidths=[0.18, 0.12, 0.70])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    edge_color = '#E2E8F0'
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#334155')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.5)

    # Row backgrounds from GROUP_COLORS
    type_to_group = {
        'Structural': 'structural', 'Keywords': 'keywords',
        'Instruction': 'instruction', 'Query': 'query',
    }

    for i, row in enumerate(table_data):
        grp = type_to_group.get(row[1], 'structural')
        color = GROUP_COLORS[grp]['table_bg']
        for j in range(3):
            cell = table[i + 1, j]
            cell.set_facecolor(color)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.5)
        table[i + 1, 0].set_text_props(fontweight='bold', fontfamily='monospace', fontsize=9)

    ax.set_title('Exp 13 Prefix Conditions', fontsize=13, fontweight='bold', pad=16, loc='left')

    plt.tight_layout()
    fig.savefig(CHART_DIR / 'fig0_condition_examples.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig0_condition_examples.png")
""")

# ===== Cell 6: Section 2 header =====
md(r"""## Section 2: Simple first results — MS MARCO, 4 conditions

Start with just one dataset and five conditions to build intuition:
- **repeat_token**: single token repeated — minimal structural signal
- **random**: random tokens — pure structural diversity
- **unrelated**: coherent but off-topic text — structure + grammar
- **extract**: a meaningful instruction — does semantic content help?
- **oracle**: the actual query — the ceiling for prefix quality

""")

# ===== Cell 7: Fig 1 — MS MARCO simple bars (seaborn) =====
code(r"""# Fig 1: MS MARCO — 4 conditions (seaborn barplot)
simple_conds = ['repeat_token', 'random', 'unrelated', 'extract', 'oracle']
ms_sub = df[df['dataset'] == 'ms_marco']

ds_vals = []
for cond in simple_conds:
    diff = ms_sub['nll_bare'].values - ms_sub[f'nll_{cond}'].values
    d = cohens_d(diff)
    w = win_rate(diff)
    _, p = paired_ttest(diff)
    ds_vals.append({'Condition': COND_LABELS[cond], 'cond': cond,
                    "Cohen's d": d, 'win': w, 'p': p})

plot_df = pd.DataFrame(ds_vals).sort_values("Cohen's d").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 4))
palette = [COND_COLORS[row['cond']] for _, row in plot_df.iterrows()]
sns.barplot(data=plot_df, y='Condition', x="Cohen's d", palette=palette,
            orient='h', ax=ax, saturation=0.85, edgecolor='white', linewidth=0.6)
sns.despine(left=True, bottom=True)

ax.axvline(0, color='.6', linewidth=0.5, zorder=0)

for i, (_, row) in enumerate(plot_df.iterrows()):
    s = stars(row['p'])
    d_val = row["Cohen's d"]
    x_text = max(d_val, 0) + 0.015
    ax.text(x_text, i, f"d = {d_val:+.2f}   {row['win']:.0%} win{s}",
            va='center', ha='left', fontsize=10, color='.3')

ax.set_xlabel("Cohen's d  (positive = condition helps)")
ax.set_ylabel('')
ax.set_title('MS MARCO: Does the prefix content matter?', pad=12)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig1_msmarco_simple.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig1_msmarco_simple.png")
""")

# ===== Cell 8: Section 3 header =====
md(r"""## Section 3: Full condition ranking — all 13 conditions, 7 datasets pooled

All 13 prefix conditions ranked by pooled Cohen's d across 7 datasets (N = 1120).
""")

# ===== Cell 9: Fig 2 — Full ranking (seaborn) =====
code(r"""# Fig 2: Full 13-condition ranking — pooled (seaborn barplot)
sorted_conds = sorted(CONDITIONS, key=lambda c: cstats[c]['pooled_d'])
rank_df = pd.DataFrame([{
    'Condition': COND_LABELS[c],
    'cond': c,
    "Cohen's d": cstats[c]['pooled_d'],
    'win': cstats[c]['pooled_win'],
    'p': cstats[c]['pooled_p'],
} for c in sorted_conds])

fig, ax = plt.subplots(figsize=(11, 6.5))
palette = [COND_COLORS[c] for c in sorted_conds]
sns.barplot(data=rank_df, y='Condition', x="Cohen's d", palette=palette,
            orient='h', ax=ax, saturation=0.85, edgecolor='white', linewidth=0.6)
sns.despine(left=True, bottom=True)

ax.axvline(0, color='.6', linewidth=0.5, zorder=0)

for i, (_, row) in enumerate(rank_df.iterrows()):
    s = stars(row['p'])
    d_val = row["Cohen's d"]
    x_text = max(d_val, 0) + 0.008
    ax.text(x_text, i, f"d = {d_val:+.2f}   {row['win']:.0%}{s}",
            va='center', ha='left', fontsize=9, color='.3')

ax.set_ylabel('')
ax.set_xlabel("Pooled Cohen's d  (positive = condition helps)")
ax.set_title('Full condition ranking across 7 datasets  (N = 1120)', pad=12)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig2_full_ranking.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig2_full_ranking.png")
""")

# ===== Cell 10: Decomposition explainer diagram =====
md(r"""## How the decomposition works

We isolate **three levels of prefix benefit** by comparing conditions that
progressively add more information:

| Level | Proxy condition | What it measures |
|-------|----------------|-----------------|
| **Structural** | `random` tokens | Benefit from *any* prefix occupying the cache |
| **+ Vocabulary** | `scrambled_comprehend` | Added benefit from *relevant words* (minus structural) |
| **+ Meaning** | `comprehend` | Added benefit from *coherent meaning* (minus structural + vocabulary) |
""")

code(r"""# Decomposition explainer diagram
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    # Three stacked boxes showing the nesting
    box_specs = [
        (0, 0.2, 3.2, 1.2, DECOMP_COLORS['structural'], 'Structural\n(random tokens)', '.3'),
        (3.5, 0.2, 3.2, 1.2, DECOMP_COLORS['vocabulary'], 'Vocabulary\n(scrambled comprehend\n\u2212 random)', '.3'),
        (7.0, 0.2, 3.2, 1.2, DECOMP_COLORS['meaning'], 'Meaning\n(comprehend\n\u2212 scrambled)', '.3'),
    ]
    for x, y, w, h, color, label, tc in box_specs:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=10, color=tc, fontweight='medium')

    # Arrows between boxes
    for x_start, x_end in [(3.2, 3.5), (6.7, 7.0)]:
        ax.annotate('', xy=(x_end, 0.8), xytext=(x_start, 0.8),
                    arrowprops=dict(arrowstyle='->', color='.5', lw=1.2))

    # Bottom: condition examples
    examples = [
        (1.6, -0.25, 'random, unrelated,\nadversarial, repeat_token,\nood_query', DECOMP_COLORS['structural']),
        (5.1, -0.25, 'tfidf,\nscrambled_comprehend', DECOMP_COLORS['vocabulary']),
        (8.6, -0.25, 'comprehend, extract,\nclassify, llm_question, oracle,\nmisleading_query', DECOMP_COLORS['meaning']),
    ]
    for x, y, label, color in examples:
        ax.text(x, y, label, ha='center', va='top', fontsize=8,
                color=color, style='italic', fontfamily='monospace')

    # Top: formula
    ax.text(5.25, 3.2, "total benefit (comprehend)  =  structural  +  vocabulary  +  meaning",
            ha='center', va='center', fontsize=11, color='.3')

    # Curly-brace-style underlines showing the subtraction logic
    ax.text(1.6, 1.7, "d(random)", ha='center', fontsize=9, color='.3')
    ax.text(5.1, 1.7, "d(scrambled) \u2212 d(random)", ha='center', fontsize=9, color='.3')
    ax.text(8.6, 1.7, "d(comprehend) \u2212 d(scrambled)", ha='center', fontsize=9, color='.3')

    ax.set_title('Three-level decomposition of prefix benefit', pad=12,
                 fontsize=13, fontweight='medium', color='.2')
    fig.tight_layout()
    fig.savefig(CHART_DIR / 'fig2b_decomposition_explainer.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig2b_decomposition_explainer.png")
""")

# ===== Cell 12: Fig 3 — Decomposition (horizontal stacked bars) =====
code(r"""# Fig 3: Three-level decomposition — structural / vocabulary / meaning
# Horizontal stacked bars: each dataset is a row, segments stacked left->right
# (No direct seaborn equivalent for stacked bars — matplotlib with seaborn theme)
struct_d, vocab_d, meaning_d = [], [], []
for ds in DATASETS:
    struct_d.append(np.mean([cstats[c]['per_ds'][ds]['d']
                             for c in CONDITIONS if COND_TYPES[c] == 'structural']))
    vocab_d.append(np.mean([cstats[c]['per_ds'][ds]['d']
                            for c in CONDITIONS if COND_TYPES[c] == 'vocabulary']))
    meaning_d.append(np.mean([cstats[c]['per_ds'][ds]['d']
                              for c in CONDITIONS if COND_TYPES[c] == 'meaning']))

struct_d = np.array(struct_d)
vocab_d = np.array(vocab_d)
meaning_d = np.array(meaning_d)
totals = struct_d + vocab_d + meaning_d
ds_labels = [DS_LABELS[ds] for ds in DATASETS]

fig, ax = plt.subplots(figsize=(12, 5))
y = np.arange(len(DATASETS))
bar_h = 0.6

# Stack: structural first, then vocabulary on top, then meaning on top of that
# Handle mixed positive/negative by stacking positive from 0 rightward, negative leftward
bars1 = ax.barh(y, struct_d, height=bar_h, label='Structural',
                color=DECOMP_COLORS['structural'], edgecolor='white', linewidth=0.6, alpha=0.85)
bars2 = ax.barh(y, vocab_d, height=bar_h, left=struct_d, label='Vocabulary',
                color=DECOMP_COLORS['vocabulary'], edgecolor='white', linewidth=0.6, alpha=0.85)
bars3 = ax.barh(y, meaning_d, height=bar_h, left=struct_d + vocab_d, label='Meaning',
                color=DECOMP_COLORS['meaning'], edgecolor='white', linewidth=0.6, alpha=0.85)

# Total annotations at end of each bar
for i in range(len(DATASETS)):
    total = totals[i]
    x_pos = total + 0.03 if total >= 0 else total - 0.03
    ha = 'left' if total >= 0 else 'right'
    ax.text(x_pos, y[i], f'{total:+.2f}', va='center', ha=ha,
            fontsize=9, color='.3')

ax.axvline(0, color='.6', linewidth=0.5, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(ds_labels)
ax.invert_yaxis()
# Pad x-axis so annotations aren't clipped
x_min = min(totals) - 0.4
x_max = max(totals) + 1.0
ax.set_xlim(x_min, x_max)
ax.set_xlabel("Mean Cohen's d  (positive = helps)")
ax.set_title('Decomposition: Where does the benefit come from?', pad=12)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
sns.despine(left=True, bottom=True)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig3_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig3_decomposition.png")
""")

# ===== Cell 11: Section 4 header =====
md(r"""## Section 4: Task specificity

Not all conditions work equally well on all datasets. The heatmap below shows
Cohen's d for each dataset x condition pair.
""")

# ===== Cell 12: Fig 4 — Heatmap (seaborn) =====
code(r"""# Fig 4: Heatmap — 7 datasets x 13 conditions (sns.heatmap)
hm_conds = sorted(CONDITIONS, key=lambda c: cstats[c]['pooled_d'], reverse=True)
hm_labels = [COND_LABELS[c] for c in hm_conds]

matrix = pd.DataFrame(
    [[cstats[c]['per_ds'][ds]['d'] for c in hm_conds] for ds in DATASETS],
    index=[DS_LABELS[d] for d in DATASETS],
    columns=hm_labels,
)

annot = pd.DataFrame(
    [[f"{cstats[c]['per_ds'][ds]['d']:+.2f}{stars(cstats[c]['per_ds'][ds]['p'])}"
      for c in hm_conds] for ds in DATASETS],
    index=matrix.index, columns=matrix.columns,
)

vabs = max(abs(matrix.values.min()), abs(matrix.values.max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-vabs - 0.1, vmax=vabs + 0.1)

fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(matrix, annot=annot, fmt='', cmap='RdBu_r', norm=norm,
            linewidths=2, linecolor='white', ax=ax, square=False,
            cbar_kws={'shrink': 0.75, 'label': "Cohen's d  (positive = helps)"},
            annot_kws={'fontsize': 8, 'fontweight': 'normal'})

ax.set_title("Task specificity: Cohen's d by dataset and condition", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig4_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig4_heatmap.png")
""")

# ===== Cell 13: Section 5 header =====
md(r"""## Section 5: Prefix length scaling

How does the benefit change with prefix length? L = 16, 32, 64, 128, 256 on 4 datasets.
""")

# ===== Cell 14: Fig 5 — Prefix length scaling (seaborn) =====
code(r"""# Fig 5: Prefix length scaling — (A) component lines, (B) stacked area
PF_LENGTHS = sorted(pf['prefix_length'].unique())

pf_structural, pf_vocab, pf_meaning, pf_total = [], [], [], []
for L in PF_LENGTHS:
    sub = pf[pf['prefix_length'] == L]
    d_random = cohens_d(sub['nll_bare'].values - sub['nll_random'].values)
    d_scrambled = cohens_d(sub['nll_bare'].values - sub['nll_scrambled_comprehend'].values)
    d_comprehend = cohens_d(sub['nll_bare'].values - sub['nll_comprehend'].values)
    pf_structural.append(d_random)
    pf_vocab.append(d_scrambled - d_random)
    pf_meaning.append(d_comprehend - d_scrambled)
    pf_total.append(d_comprehend)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: seaborn lineplot
line_rows = []
for i, L in enumerate(PF_LENGTHS):
    line_rows.append({'Prefix length': L, "Cohen's d": pf_total[i], 'Component': 'Total (comprehend)'})
    line_rows.append({'Prefix length': L, "Cohen's d": pf_structural[i], 'Component': 'Structural (random)'})
    line_rows.append({'Prefix length': L, "Cohen's d": pf_structural[i] + pf_vocab[i], 'Component': '+ Vocabulary (scrambled)'})

line_df = pd.DataFrame(line_rows)
comp_pal = {'Total (comprehend)': GROUP_COLORS['instruction']['accent'],
            'Structural (random)': GROUP_COLORS['structural']['accent'],
            '+ Vocabulary (scrambled)': GROUP_COLORS['keywords']['accent']}
sns.lineplot(data=line_df, x='Prefix length', y="Cohen's d", hue='Component',
             style='Component', markers=True, dashes=False, palette=comp_pal, ax=ax1,
             linewidth=1.8, markersize=5)

ax1.axvline(64, color='.85', linestyle=':', linewidth=1.0, zorder=0)
ax1.text(72, min(pf_structural) * 0.92, 'L = 64\n(default)',
         fontsize=9, color='#94A3B8', va='top')
ax1.set_title('(A) Component benefit vs prefix length')
ax1.set_xscale('log', base=2)
ax1.set_xticks(PF_LENGTHS)
ax1.set_xticklabels([str(L) for L in PF_LENGTHS])
h, l = ax1.get_legend_handles_labels()
# Reorder legend to match visual top-to-bottom: Total, + Vocabulary, Structural
order = {name: i for i, name in enumerate(l)}
desired = ['Total (comprehend)', '+ Vocabulary (scrambled)', 'Structural (random)']
idx = [order[name] for name in desired]
ax1.legend([h[i] for i in idx], [l[i] for i in idx], fontsize=9, framealpha=0.9)
sns.despine(ax=ax1)

# Panel B: Stacked area (no seaborn equivalent — matplotlib with seaborn style)
pf_s, pf_v, pf_m = np.array(pf_structural), np.array(pf_vocab), np.array(pf_meaning)
abs_total = np.abs(pf_s) + np.abs(pf_v) + np.abs(pf_m)
abs_total = np.where(abs_total == 0, 1, abs_total)
pct_s = np.abs(pf_s) / abs_total * 100
pct_v = np.abs(pf_v) / abs_total * 100
pct_m = np.abs(pf_m) / abs_total * 100

ax2.stackplot(PF_LENGTHS, pct_s, pct_v, pct_m,
              colors=[DECOMP_COLORS['structural'], DECOMP_COLORS['vocabulary'],
                      DECOMP_COLORS['meaning']],
              labels=['Structural', 'Vocabulary', 'Meaning'], alpha=0.65)
ax2.axvline(64, color='white', linestyle=':', linewidth=1.0, zorder=3)
ax2.set_xlabel('Prefix length (tokens)')
ax2.set_ylabel('Share of total effect (%)')
ax2.set_title('(B) Composition vs prefix length')
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h2[::-1], l2[::-1], loc='upper left', fontsize=9, framealpha=0.9)
ax2.set_ylim(0, 100)
ax2.set_xscale('log', base=2)
ax2.set_xticks(PF_LENGTHS)
ax2.set_xticklabels([str(L) for L in PF_LENGTHS])
sns.despine(ax=ax2)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig5_prefix_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig5_prefix_scaling.png")
""")

# ===== Cell 15: Fig 6 — Doc length scaling (seaborn) =====
code(r"""# Fig 6: Document length scaling — TriviaQA only (seaborn lineplot)
DC_LENGTHS = sorted(dc['doc_length'].unique())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: Mean NLL vs D — tidy DataFrame
nll_rows = []
for cond, label in [('bare', 'Bare'), ('comprehend', 'Comprehend'), ('random', 'Random')]:
    for D in DC_LENGTHS:
        mean_nll = dc[dc['doc_length'] == D][f'nll_{cond}'].mean()
        nll_rows.append({'Doc length': D, 'Mean NLL': mean_nll, 'Condition': label})

nll_df = pd.DataFrame(nll_rows)
cond_pal = {'Bare': C_BARE, 'Comprehend': C_CONDITIONED, 'Random': GROUP_COLORS['structural']['accent']}
sns.lineplot(data=nll_df, x='Doc length', y='Mean NLL', hue='Condition', style='Condition',
             markers=True, dashes=False, palette=cond_pal, ax=ax1,
             linewidth=1.8, markersize=5)

ax1.set_xlabel('Document length (tokens)')
ax1.set_title('(A) NLL vs document length (TriviaQA)')
ax1.legend(fontsize=10, framealpha=0.9)
sns.despine(ax=ax1)

# Panel B: Cohen's d vs D — seaborn lineplot
d_by_D = []
for D in DC_LENGTHS:
    sub = dc[dc['doc_length'] == D]
    diff = sub['nll_bare'].values - sub['nll_comprehend'].values
    d_by_D.append({'Doc length': D, "Cohen's d": cohens_d(diff) if len(diff) >= 2 else 0})

d_df = pd.DataFrame(d_by_D)
sns.lineplot(data=d_df, x='Doc length', y="Cohen's d", marker='o',
             color=C_CONDITIONED, linewidth=1.8, markersize=5, ax=ax2)
ax2.fill_between(DC_LENGTHS, 0, [r["Cohen's d"] for r in d_by_D],
                 alpha=0.15, color=C_LIGHT_FILL)
ax2.axhline(0, color='.6', linewidth=0.5)
ax2.set_xlabel('Document length (tokens)')
ax2.set_title('(B) Comprehend benefit vs document length')
sns.despine(ax=ax2)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig6_doc_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig6_doc_scaling.png")
""")

# ===== Cell 16: Fig 7 — Model size (seaborn) =====
code(r"""# Fig 7: Model size — grouped bars (seaborn barplot)
MS_DATASETS = sorted(ms['dataset'].unique())
MODELS = ['google/gemma-3-1b-it', 'google/gemma-3-4b-it',
          'google/gemma-3-12b-it', 'google/gemma-3-27b-it']
MODEL_LABELS = {m: m.split('-')[2].upper() for m in MODELS}

rows = []
for model in MODELS:
    for ds in MS_DATASETS:
        sub = ms[(ms['dataset'] == ds) & (ms['model'] == model)]
        diff = sub['nll_bare'].values - sub['nll_comprehend'].values
        rows.append({'Model': MODEL_LABELS[model], 'Dataset': DS_LABELS[ds],
                     "Cohen's d": cohens_d(diff)})
mdf = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11, 5.5))
model_pal = dict(zip([MODEL_LABELS[m] for m in MODELS],
                     sns.light_palette(GROUP_COLORS['instruction']['accent'], 6, reverse=True)[1:5]))
sns.barplot(data=mdf, x='Dataset', y="Cohen's d", hue='Model', palette=model_pal,
            ax=ax, saturation=0.85, edgecolor='white', linewidth=0.6)
sns.despine(left=True, bottom=True)

ax.axhline(0, color='.6', linewidth=0.5)
ax.set_title('Model size: Comprehend benefit across Gemma 3 scales', pad=12)
ax.legend(title='Model', fontsize=10, title_fontsize=10, framealpha=0.9)

fig.tight_layout()
fig.savefig(CHART_DIR / 'fig7_model_size.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig7_model_size.png")
""")

# ===== Cell 19: Section 9 header =====
md(r"""## Section 9: Quantization damage

How much does int8 and int4 quantization hurt, and does prefix conditioning shield
against it? (Normalization is baked into the exp13 pipeline.)
""")

# ===== Cell 20: Quantization explainer diagram =====
code(r"""# Quantization damage explainer diagram
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1.5, 5.5)
    ax.axis('off')

    # --- Top row: the two pipelines ---
    ax.text(7.5, 5.0, 'How quantization damage is measured', ha='center',
            fontsize=13, fontweight='medium', color='.2')

    # Pipeline 1: Bare
    y1 = 3.5
    ax.text(-0.3, y1 + 0.35, 'Bare:', fontsize=10, color='.3', fontweight='medium', va='center')
    bare_steps = [
        (1.0, 2.5, 'bare\ncache', GROUP_COLORS['structural']['bar'], '.3'),
        (4.0, 1.8, 'normalize', '#B8EBC8', '.3'),
        (6.3, 1.8, 'quantize\n(int8/int4)', '#EBB8B6', '.3'),
        (8.6, 1.5, 'NLL', '.3', 'white'),
    ]
    for x, w, label, color, tc in bare_steps:
        box = FancyBboxPatch((x, y1 - 0.05), w, 0.8, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y1 + 0.35, label, ha='center', va='center',
                fontsize=9, color=tc, fontweight='medium')
    # Arrows between steps
    for x1, w1, x2 in [(1.0, 2.5, 4.0), (4.0, 1.8, 6.3), (6.3, 1.8, 8.6)]:
        ax.annotate('', xy=(x2, y1 + 0.35), xytext=(x1 + w1, y1 + 0.35),
                    arrowprops=dict(arrowstyle='->', color='.6', lw=1.0))

    # Reference: bare (no quantization)
    ax.annotate('', xy=(10.6, y1 + 0.35), xytext=(10.1, y1 + 0.35),
                arrowprops=dict(arrowstyle='-', color='.75', lw=1.0, linestyle='--'))
    ax.text(12.8, y1 + 0.35, 'reference:  bare (no quant)', fontsize=8,
            color='.5', style='italic', ha='center', va='center')

    # Pipeline 2: Comprehend
    y2 = 1.8
    ax.text(-0.3, y2 + 0.35, 'Comp:', fontsize=10, color='.3', fontweight='medium', va='center')
    comp_steps = [
        (1.0, 2.5, 'comprehend\ncache', GROUP_COLORS['instruction']['bar'], '.3'),
        (4.0, 1.8, 'normalize', '#B8EBC8', '.3'),
        (6.3, 1.8, 'quantize\n(int8/int4)', '#EBB8B6', '.3'),
        (8.6, 1.5, 'NLL', '.3', 'white'),
    ]
    for x, w, label, color, tc in comp_steps:
        box = FancyBboxPatch((x, y2 - 0.05), w, 0.8, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y2 + 0.35, label, ha='center', va='center',
                fontsize=9, color=tc, fontweight='medium')
    for x1, w1, x2 in [(1.0, 2.5, 4.0), (4.0, 1.8, 6.3), (6.3, 1.8, 8.6)]:
        ax.annotate('', xy=(x2, y2 + 0.35), xytext=(x1 + w1, y2 + 0.35),
                    arrowprops=dict(arrowstyle='->', color='.6', lw=1.0))

    ax.annotate('', xy=(10.6, y2 + 0.35), xytext=(10.1, y2 + 0.35),
                arrowprops=dict(arrowstyle='-', color='.75', lw=1.0, linestyle='--'))
    ax.text(12.8, y2 + 0.35, 'reference:  comprehend (no quant)', fontsize=8,
            color='.5', style='italic', ha='center', va='center')

    # --- Bottom: formula box ---
    formula_box = FancyBboxPatch((1.5, -1.2), 13, 1.1, boxstyle="round,pad=0.1",
                                  facecolor='#f7f7f7', edgecolor='.8', linewidth=0.8)
    ax.add_patch(formula_box)
    ax.text(8.0, -0.4, 'damage  =  NLL(cache + quant)  \u2212  NLL(cache)',
            ha='center', va='center', fontsize=10, color='.3', fontfamily='monospace')
    ax.text(8.0, -0.9, 'shielding ratio  =  bare damage / comprehend damage',
            ha='center', va='center', fontsize=10, color='.3', fontfamily='monospace')

    fig.tight_layout()
    fig.savefig(CHART_DIR / 'fig8b_quantization_explainer.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig8b_quantization_explainer.png")
""")

# ===== Cell 21: Fig 9 — Quantization damage (seaborn) =====
code(r"""# Fig 9: Quantization damage — (A) int8, (B) int4 (seaborn barplot)
# Exp 13 has normalization baked in: columns are nll_bare_int8 (not nll_bare_norm_int8)
QUANT_DATASETS = sorted(df.dropna(subset=['nll_bare_int8'])['dataset'].unique())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, col_suffix, panel_label in [(ax1, 'int8', 'A'), (ax2, 'int4', 'B')]:
    quant_rows = []
    bare_damage, comp_damage = [], []
    for ds in QUANT_DATASETS:
        sub = df[df['dataset'] == ds]
        bd = sub[f'nll_bare_{col_suffix}'].mean() - sub['nll_bare'].mean()
        cd = sub[f'nll_comprehend_{col_suffix}'].mean() - sub['nll_comprehend'].mean()
        bare_damage.append(bd)
        comp_damage.append(cd)
        quant_rows.append({'Dataset': DS_LABELS[ds], 'Cache': 'Bare', 'NLL damage': bd})
        quant_rows.append({'Dataset': DS_LABELS[ds], 'Cache': 'Comprehend', 'NLL damage': cd})

    qdf = pd.DataFrame(quant_rows)
    q_pal = {'Bare': C_BARE, 'Comprehend': C_CONDITIONED}
    sns.barplot(data=qdf, x='Dataset', y='NLL damage', hue='Cache', palette=q_pal,
                ax=ax, saturation=0.85, edgecolor='white', linewidth=0.6)
    sns.despine(ax=ax, left=True, bottom=True)

    # Shielding annotations
    for i in range(len(QUANT_DATASETS)):
        bd, cd = bare_damage[i], comp_damage[i]
        if bd > 0.01 and cd > 0.01:
            ratio = bd / cd
            y_top = max(bd, cd) + 0.08
            ax.text(i, y_top, f'{ratio:.0f}x', ha='center', va='bottom',
                    fontsize=9, color='.3')
        elif bd > 0.01 and cd <= 0.01:
            ax.text(i, bd + 0.08, 'full\nshield', ha='center', va='bottom',
                    fontsize=9, color='.3')

    ax.axhline(0, color='.6', linewidth=0.5, zorder=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.set_title(f'({panel_label})  {col_suffix} quantization damage')
    if panel_label == 'B':
        if ax.get_legend():
            ax.get_legend().remove()

ax1.set_ylabel('NLL damage  (positive = worse)')
ax1.legend(fontsize=10, framealpha=0.9)

fig.suptitle('Prefix conditioning shields against quantization damage',
             y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(CHART_DIR / 'fig9_quantization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig9_quantization.png")
""")

# ===== Appendix: Dataset Examples & Experimental Mapping =====
md(r"""## Appendix: Dataset Examples & Experimental Mapping

How each dataset's native structure maps to our doc/query/answer framework.
""")

code(r"""# Appendix table 1: Dataset field mapping
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')

    mapping_table = [
        ['MS MARCO',  'passage_text (selected)',        'query',     'wellFormedAnswers[0]',  'Passages with is_selected=1'],
        ['SQuAD 2.0', 'context',                        'question',  'answers.text[0]',       'First answer span'],
        ['TriviaQA',  'wiki_context (500 words)',        'question',  'answer.value',          'Truncated wiki article'],
        ['HotpotQA',  'Concat. supporting facts',       'question',  'answer',                'Multi-hop reasoning'],
        ['DROP',       'passage',                        'question',  'answers_spans.spans[0]','Numerical/discrete reasoning'],
        ['BoolQ',      'passage',                        'question',  'bool -> "Yes"/"No"',    'Boolean conversion'],
        ['GSM8K',      'question (problem text)',        'Fixed: "What is the answer?"', 'Number after ####',     'Doc is the problem statement'],
    ]

    col_labels = ['Dataset', 'Source -> Document', 'Source -> Query', 'Source -> Answer', 'Notes']

    table = ax.table(cellText=mapping_table, colLabels=col_labels,
                     cellLoc='left', loc='center',
                     colWidths=[0.10, 0.22, 0.22, 0.20, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)

    edge_color = '#E2E8F0'
    for j in range(5):
        cell = table[0, j]
        cell.set_facecolor('#334155')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.5)

    for i in range(len(mapping_table)):
        bg = '#F8FAFC' if i % 2 == 0 else 'white'
        for j in range(5):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.5)
        table[i + 1, 0].set_text_props(fontweight='bold', fontsize=9)

    ax.set_title('Dataset field mapping: source fields -> experimental framework',
                 fontsize=13, fontweight='bold', pad=16, loc='left')

    plt.tight_layout()
    fig.savefig(CHART_DIR / 'fig_appendix_mapping.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig_appendix_mapping.png")
""")

code(r"""# Appendix table 2: Representative sample from each dataset
# Passage text extracted by extract_dataset_examples.py (replays HF loading pipeline)
examples_path = RESULTS_DIR / "dataset_examples.json"
assert examples_path.exists(), (
    f"Run extract_dataset_examples.py first to generate {examples_path}")
ds_examples = json.loads(examples_path.read_text())

TRUNC = 55  # max chars per cell before truncation

def trunc(s, n=TRUNC):
    return s[:n] + ('...' if len(s) > n else '')

example_table = []
for ds_name, ds_label in [('ms_marco', 'MS MARCO'), ('squad_v2', 'SQuAD 2.0'),
                           ('triviaqa', 'TriviaQA'), ('hotpotqa', 'HotpotQA'),
                           ('drop', 'DROP'), ('boolq', 'BoolQ'), ('gsm8k', 'GSM8K')]:
    ex = ds_examples[ds_name]
    example_table.append([
        ds_label,
        trunc(ex['passage']),
        trunc(ex['query']),
        trunc(ex['answer']),
    ])

with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis('off')

    col_labels = ['Dataset', 'Document (first ~55 chars)', 'Query', 'Answer']

    table = ax.table(cellText=example_table, colLabels=col_labels,
                     cellLoc='left', loc='center',
                     colWidths=[0.10, 0.35, 0.32, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    edge_color = '#E2E8F0'
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor('#334155')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.5)

    for i in range(len(example_table)):
        bg = '#F8FAFC' if i % 2 == 0 else 'white'
        for j in range(4):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.5)
        table[i + 1, 0].set_text_props(fontweight='bold', fontsize=9)

    ax.set_title('Representative sample from each dataset  (first hard sample)',
                 fontsize=13, fontweight='bold', pad=16, loc='left')

    plt.tight_layout()
    fig.savefig(CHART_DIR / 'fig_appendix_examples.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Saved fig_appendix_examples.png")
""")

# ===== Per-dataset example slides =====
md(r"""## Per-dataset slides

One slide per dataset showing the document, query, and answer text with
source field mapping annotations.
""")

code(r"""# Per-dataset example slides — one figure per dataset
import textwrap

DS_FIELD_MAP = {
    'ms_marco': {
        'doc_field': 'passage_text  (is_selected=1)',
        'query_field': 'query',
        'answer_field': 'wellFormedAnswers[0]',
        'domain': 'Web search QA',
        'answer_type': 'Extractive',
        'note': 'Bing search passages; only selected passage used as document',
    },
    'squad_v2': {
        'doc_field': 'context',
        'query_field': 'question',
        'answer_field': 'answers.text[0]',
        'domain': 'Reading comprehension',
        'answer_type': 'Extractive span',
        'note': 'Wikipedia paragraphs; first answer span selected',
    },
    'triviaqa': {
        'doc_field': 'entity_pages.wiki_context[0]  (500 words)',
        'query_field': 'question',
        'answer_field': 'answer.value',
        'domain': 'Trivia QA',
        'answer_type': 'Extractive',
        'note': 'Wiki article truncated to 500 words; answer must appear in text',
    },
    'hotpotqa': {
        'doc_field': 'supporting_facts sentences  (concatenated)',
        'query_field': 'question',
        'answer_field': 'answer',
        'domain': 'Multi-hop reasoning',
        'answer_type': 'Extractive',
        'note': 'Only gold supporting fact sentences, not distractor paragraphs',
    },
    'drop': {
        'doc_field': 'passage',
        'query_field': 'question',
        'answer_field': 'answers_spans.spans[0]',
        'domain': 'Discrete reasoning',
        'answer_type': 'Numerical / discrete',
        'note': 'Counting, sorting, arithmetic over paragraphs',
    },
    'boolq': {
        'doc_field': 'passage',
        'query_field': 'question',
        'answer_field': 'answer  (bool -> "Yes"/"No")',
        'domain': 'Boolean QA',
        'answer_type': 'Yes / No',
        'note': 'Boolean label converted to text; naturally-occurring web queries',
    },
    'gsm8k': {
        'doc_field': 'question  (problem text used as document)',
        'query_field': 'Fixed: "What is the answer?"',
        'answer_field': 'answer.split("####")[-1]',
        'domain': 'Grade school math',
        'answer_type': 'Numerical',
        'note': 'Problem statement is the document; query is fixed across all samples',
    },
}

DOC_MAX_CHARS = 600
ANSWER_MAX_CHARS = 200
WRAP_W = 85

for ds_name in DATASETS:
    ds_label = DS_LABELS[ds_name]
    ex = ds_examples[ds_name]
    fm = DS_FIELD_MAP[ds_name]

    # Prepare text
    doc_text = ex['passage'].strip()
    if len(doc_text) > DOC_MAX_CHARS:
        doc_text = doc_text[:DOC_MAX_CHARS].rsplit(' ', 1)[0] + ' ...'
    query_text = ex['query']
    answer_text = ex['answer']
    if len(answer_text) > ANSWER_MAX_CHARS:
        answer_text = answer_text[:ANSWER_MAX_CHARS].rsplit(' ', 1)[0] + ' ...'

    doc_wrapped = textwrap.fill(doc_text, width=WRAP_W)
    doc_lines = doc_wrapped.count('\n') + 1
    ans_wrapped = textwrap.fill(answer_text, width=WRAP_W)
    ans_lines = ans_wrapped.count('\n') + 1

    # Row heights — tight, matching table style from fig0b
    FW = 11
    def _rh(n):
        return 0.42 + 0.22 * n
    header_h = 0.50
    doc_rh = _rh(doc_lines)
    query_rh = _rh(1)
    ans_rh = _rh(ans_lines)
    note_h = 0.30
    fig_h = header_h + doc_rh + query_rh + ans_rh + note_h

    ec = '#E2E8F0'
    LM = 0.25
    RM = FW - 0.25

    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(FW, fig_h))
        ax.set_xlim(0, FW)
        ax.set_ylim(0, fig_h)
        ax.axis('off')

        y = fig_h  # top-down cursor

        # --- Header (matching table header: #334155 with white bold text) ---
        y -= header_h
        ax.add_patch(FancyBboxPatch((0, y), FW, header_h,
                     boxstyle='square,pad=0', facecolor='#334155', edgecolor='none'))
        ax.text(LM, y + header_h / 2, ds_label,
                fontsize=13, fontweight='bold', color='white', va='center')
        sub = f"{fm['domain']}  \u00b7  {fm['answer_type']}  \u00b7  {ex['word_count']} words"
        ax.text(RM, y + header_h / 2, sub,
                fontsize=8.5, color='#94A3B8', ha='right', va='center')

        # --- Helper: draw one table row ---
        def _draw_row(y_cur, h, label, src, text, bg, fs=9.5):
            y_cur -= h
            ax.add_patch(FancyBboxPatch((0, y_cur), FW, h,
                         boxstyle='square,pad=0', facecolor=bg,
                         edgecolor=ec, linewidth=0.5))
            ly = y_cur + h - 0.15
            ax.text(LM, ly, label, fontsize=9.5, fontweight='bold',
                    color='#334155', va='center')
            ax.text(RM, ly, src, fontsize=7.5, color='#94A3B8',
                    ha='right', va='center')
            ax.text(LM, ly - 0.20, text, fontsize=fs, color='#1E293B',
                    va='top', linespacing=1.35)
            return y_cur

        # --- Document (alternating bg: light gray) ---
        doc_src = fm['doc_field']
        if 'words' not in doc_src.lower():
            doc_src += f'  \u00b7  {ex["word_count"]} words'
        y = _draw_row(y, doc_rh, 'Document', doc_src, doc_wrapped, '#F8FAFC', fs=9)

        # --- Query (alternating bg: white) ---
        y = _draw_row(y, query_rh, 'Query', fm['query_field'], query_text, 'white', fs=10)

        # --- Answer (alternating bg: light gray) ---
        y = _draw_row(y, ans_rh, 'Answer', fm['answer_field'], ans_wrapped, '#F8FAFC', fs=10)

        # --- Note (alternating bg: white) ---
        y -= note_h
        ax.add_patch(FancyBboxPatch((0, y), FW, note_h,
                     boxstyle='square,pad=0', facecolor='white',
                     edgecolor=ec, linewidth=0.5))
        ax.text(FW / 2, y + note_h / 2, fm['note'],
                fontsize=7.5, color='#64748B', style='italic',
                ha='center', va='center')

        fig.savefig(CHART_DIR / f'fig_slide_{ds_name}.png', dpi=150,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print(f'Saved fig_slide_{ds_name}.png')
""")

# ===== Cell 21: Summary =====
md(r"""## Summary

| Finding | Evidence |
|---------|----------|
| **Content matters** | Comprehend >> random >> repeat_token |
| **Three levels** | Structural + vocabulary + meaning decomposition per dataset |
| **Task-specific** | BoolQ reverses (negative d); GSM8K sees largest benefit |
| **Scales with L** | Benefit grows with prefix length; meaning share increases |
| **Model-robust** | Effect present from 1B to 27B; 12B = production sweet spot |
| **OOD query** | Real but unrelated question — tests structural vs semantic query benefit |
| **Misleading query** | False-premise question — tests robustness to misleading content |
| **Quantization** | int8 ~ free; int4 damages bare caches, comprehend shields |
""")

# ===== Cell 22: Summary print =====
code(r"""# Summary: list generated charts
import glob as _glob
charts = sorted(_glob.glob(str(CHART_DIR / '*.png')))
print(f"\nGenerated {len(charts)} charts:")
for c in charts:
    print(f"  {Path(c).name}")
print(f"\nAll charts saved to {CHART_DIR}")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/13/13_presentation_charts.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}")
print(f"  {len(nb.cells)} cells ({sum(1 for c in nb.cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown)")
