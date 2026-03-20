#!/usr/bin/env python3
"""Build Exp 15 KV cache routing notebook.

Generates 15_kv_cache_routing.ipynb — CPU-only, loads pre-computed NLLs from Exp 14,
evaluates 10 routing strategies across 4 cache pools using 5-fold stratified CV.

Usage:
    cd /home/jupyter/research/directed_kvcache_v4
    python3 experiments/decoder_only/15/build_exp15.py
    cd experiments/decoder_only/15
    papermill 15_kv_cache_routing.ipynb 15_kv_cache_routing_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/15", exist_ok=True)

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
# Cell 0: Title & design overview
# =====================================================================
md(r"""# Exp 15 — KV Cache Routing

Given K pre-conditioned KV caches for a document, route incoming queries to the cache
that minimizes answer NLL. All analysis is CPU-only, using pre-computed NLLs from Exp 14.

**Research question:** What is the headroom for routing (oracle vs best-single), and can
practical routers close the gap?

**Method:**
- 4 cache pools (3-7 conditions each) representing different deployment scenarios
- 10 routing strategies (2 baselines, 2 heuristic, 6 ML) evaluated via 5-fold stratified CV
- 1120 hard samples (160 × 7 datasets) from Exp 14

**Figures:** 9 charts covering oracle headroom, condition dominance, strategy comparison,
regret, win rates, per-dataset breakdown, feature importance, cost-benefit, and confusion.""")


# =====================================================================
# Cell 1: Setup — imports, seaborn theme, load data
# =====================================================================
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../../..")

import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from lib.analysis import cohens_d, win_rate, paired_ttest

warnings.filterwarnings('ignore', category=FutureWarning)

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

RESULTS14_DIR = Path("../../../results/decoder_only/exp14")
RESULTS_DIR = Path("../../../results/decoder_only/exp15")
RESULTS_DIR.mkdir(exist_ok=True, mode=0o777)
CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True, mode=0o777)

# --- Load data ---
results = json.loads((RESULTS14_DIR / "results.json").read_text())
transfer = json.loads((RESULTS14_DIR / "transfer_matrix.json").read_text())

print(f"Loaded Exp 14 results: {len(results['rankings'])} conditions")
print(f"Loaded transfer matrix: {len(transfer['per_pair'])} pairs")

# --- Shared constants ---
DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'gsm8k']
DS_LABELS = {
    'ms_marco': 'MS MARCO', 'squad_v2': 'SQuAD 2.0', 'triviaqa': 'TriviaQA',
    'hotpotqa': 'HotpotQA', 'drop': 'DROP', 'boolq': 'BoolQ', 'gsm8k': 'GSM8K',
}
N_HARD = 160
print(f"Datasets: {len(DATASETS)}, samples per dataset: {N_HARD}, total: {len(DATASETS) * N_HARD}")
""")


# =====================================================================
# Cell 2: Build master DataFrame
# =====================================================================
code(r"""# Build master DataFrame: 1120 rows x (metadata + NLL columns)
rows = []
for ds in DATASETS:
    samples = results['per_sample_results'][ds]
    assert len(samples) == N_HARD, f"{ds}: expected {N_HARD}, got {len(samples)}"
    for i, s in enumerate(samples):
        row = {
            'dataset': ds,
            'sample_idx': i,
            'query': s['query'],
            'answer': s['answer'],
            'passage_words': s['passage_words'],
            'original_idx': s['original_idx'],
        }
        # Add all NLL columns
        for k, v in s.items():
            if k.startswith('nll_'):
                row[k] = v
        rows.append(row)

df = pd.DataFrame(rows)
nll_cols = [c for c in df.columns if c.startswith('nll_')]
print(f"Master DataFrame: {df.shape[0]} rows, {len(nll_cols)} NLL columns")
print(f"NLL columns: {sorted(nll_cols)[:5]} ... ({len(nll_cols)} total)")
print(f"\nDataset counts:\n{df['dataset'].value_counts().sort_index().to_string()}")
""")


# =====================================================================
# Cell 3: Cache pool definitions (markdown)
# =====================================================================
md(r"""## Cache Pool Definitions

| Pool | K | Conditions | Scenario |
|------|---|-----------|----------|
| **A** | 3 | comprehend, extract, classify | Task-type caches |
| **B** | 5 | comprehend, extract, tfidf, random, adversarial | Diverse methodology mix |
| **C** | 4 | soft_rand_nonorm, univ_rand_nonorm, comprehend, random | Best learned + best static |
| **D** | 7 | 7 per-dataset rand soft prompts | Domain-specialist caches (transfer eval) |

**Pool D** uses cross-dataset transfer NLLs: for each sample in dataset X, the NLL
from applying dataset Y's soft prompt. The diagonal (Y=X) comes from `nll_soft_rand_nonorm`
in results.json; off-diagonal entries come from `transfer_matrix.json`.""")


# =====================================================================
# Cell 4: Define 4 cache pools + Pool D from transfer data
# =====================================================================
code(r"""# --- Cache pools A-C: directly from results.json NLL columns ---
POOLS = {
    'A': {
        'description': 'Task-type caches',
        'conditions': ['comprehend', 'extract', 'classify'],
    },
    'B': {
        'description': 'Diverse methodology mix',
        'conditions': ['comprehend', 'extract', 'tfidf', 'random', 'adversarial'],
    },
    'C': {
        'description': 'Best learned + best static',
        'conditions': ['soft_rand_nonorm', 'univ_rand_nonorm', 'comprehend', 'random'],
    },
}

# Verify all conditions exist as nll_ columns
for pool_name, pool in POOLS.items():
    for cond in pool['conditions']:
        col = f'nll_{cond}'
        assert col in df.columns, f"Pool {pool_name}: missing column {col}"
    print(f"Pool {pool_name} ({pool['description']}): {len(pool['conditions'])} conditions — {pool['conditions']}")

# --- Pool D: per-dataset specialist soft prompts (from transfer eval) ---
# For a sample in target dataset T, the NLL under source-dataset S's soft prompt:
#   - If S == T (diagonal): use nll_soft_rand_nonorm from results.json
#   - If S != T (off-diagonal): use transfer_matrix per_pair[S_to_T]['nlls'][sample_idx]

pool_d_sources = DATASETS  # 7 source datasets
pool_d_nll_cols = []

for source_ds in pool_d_sources:
    col_name = f'nll_transfer_{source_ds}'
    pool_d_nll_cols.append(col_name)

    nlls = []
    for target_ds in DATASETS:
        if source_ds == target_ds:
            # Diagonal: use per-dataset soft prompt NLL from results.json
            nlls.extend(df.loc[df['dataset'] == target_ds, 'nll_soft_rand_nonorm'].values)
        else:
            # Off-diagonal: use transfer matrix
            pair_key = f'{source_ds}_to_{target_ds}'
            pair_data = transfer['per_pair'][pair_key]
            nlls.extend(pair_data['nlls'])

    assert len(nlls) == len(df), f"Pool D {source_ds}: expected {len(df)}, got {len(nlls)}"
    df[col_name] = nlls

POOLS['D'] = {
    'description': 'Domain-specialist caches (transfer)',
    'conditions': [f'transfer_{ds}' for ds in pool_d_sources],
}

print(f"\nPool D (Domain-specialist caches): {len(pool_d_sources)} conditions")
print(f"  Sources: {pool_d_sources}")
print(f"  New columns: {pool_d_nll_cols}")

# Summary
for pool_name in sorted(POOLS.keys()):
    pool = POOLS[pool_name]
    K = len(pool['conditions'])
    print(f"\nPool {pool_name} (K={K}): {pool['description']}")
    for cond in pool['conditions']:
        col = f'nll_{cond}'
        print(f"  {cond}: mean NLL = {df[col].mean():.4f}")
""")


# =====================================================================
# Cell 5: Feature engineering — text features
# =====================================================================
code(r"""# --- Feature engineering: query-only + document metadata features ---
# These are realistic features available at routing time (no answer leakage)

WH_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose'}
YESNO_WORDS = {'is', 'does', 'do', 'did', 'was', 'were', 'are', 'can', 'could',
               'would', 'should', 'will', 'has', 'have', 'had'}
COMPARISON_WORDS = {'more', 'less', 'better', 'worse', 'larger', 'smaller',
                    'higher', 'lower', 'most', 'least', 'compare', 'difference', 'between'}
COUNTING_WORDS = {'how many', 'how much', 'number of', 'count', 'total'}
TEMPORAL_WORDS = {'when', 'before', 'after', 'during', 'since', 'until', 'year', 'date', 'time'}
CAUSAL_WORDS = {'why', 'cause', 'because', 'reason', 'result', 'effect', 'lead to', 'due to'}


def extract_text_features(df_in):
    # Extract hand-crafted text features from query and document metadata.
    feats = pd.DataFrame(index=df_in.index)

    queries_lower = df_in['query'].str.lower().str.strip()

    # Question word type (one-hot)
    first_words = queries_lower.str.split().str[0].fillna('')
    feats['is_wh'] = first_words.isin(WH_WORDS).astype(float)
    feats['is_yesno'] = first_words.isin(YESNO_WORDS).astype(float)
    feats['is_other_qtype'] = (~feats['is_wh'].astype(bool) & ~feats['is_yesno'].astype(bool)).astype(float)

    # Length features
    feats['query_words'] = df_in['query'].str.split().str.len().fillna(0).astype(float)
    feats['query_chars'] = df_in['query'].str.len().fillna(0).astype(float)
    feats['passage_words'] = df_in['passage_words'].astype(float)
    feats['query_doc_ratio'] = feats['query_words'] / feats['passage_words'].clip(lower=1)

    # Complexity signals
    feats['has_comparison'] = queries_lower.apply(
        lambda q: float(any(w in q for w in COMPARISON_WORDS)))
    feats['has_counting'] = queries_lower.apply(
        lambda q: float(any(w in q for w in COUNTING_WORDS)))
    feats['has_temporal'] = queries_lower.apply(
        lambda q: float(any(w in q for w in TEMPORAL_WORDS)))
    feats['has_causal'] = queries_lower.apply(
        lambda q: float(any(w in q for w in CAUSAL_WORDS)))

    # Dataset identity one-hot
    for ds in DATASETS:
        feats[f'ds_{ds}'] = (df_in['dataset'] == ds).astype(float)

    return feats


text_features = extract_text_features(df)
print(f"Text features: {text_features.shape}")
print(f"Columns: {list(text_features.columns)}")
print(f"\nQuestion type distribution:")
print(f"  wh-word: {text_features['is_wh'].sum():.0f}")
print(f"  yes/no:  {text_features['is_yesno'].sum():.0f}")
print(f"  other:   {text_features['is_other_qtype'].sum():.0f}")
""")


# =====================================================================
# Cell 6: Sentence embeddings + TF-IDF + caching
# =====================================================================
code(r"""# --- Sentence embeddings (all-MiniLM-L6-v2) + TF-IDF ---
EMBED_CACHE = RESULTS_DIR / "embeddings_cache.npz"

if EMBED_CACHE.exists():
    cache_data = np.load(EMBED_CACHE)
    query_embeddings = cache_data['embeddings']
    print(f"Loaded cached embeddings: {query_embeddings.shape}")
else:
    print("Computing sentence embeddings (first run only)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embeddings = model.encode(df['query'].tolist(), show_progress_bar=True,
                                     batch_size=64)
    np.savez(EMBED_CACHE, embeddings=query_embeddings)
    print(f"Computed and cached embeddings: {query_embeddings.shape}")

assert query_embeddings.shape == (len(df), 384), f"Unexpected shape: {query_embeddings.shape}"

# TF-IDF features
tfidf = TfidfVectorizer(max_features=200, stop_words='english')
tfidf_features = tfidf.fit_transform(df['query']).toarray()
print(f"TF-IDF features: {tfidf_features.shape}")

# PCA of embeddings for tree-based models
pca = PCA(n_components=20, random_state=42)
embed_pca = pca.fit_transform(query_embeddings)
print(f"PCA embeddings: {embed_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
""")


# =====================================================================
# Cell 7: Cross-validation setup
# =====================================================================
code(r"""# --- 5-fold stratified cross-validation ---
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(df, df['dataset']))

print(f"Cross-validation: {N_FOLDS} folds, stratified by dataset")
for i, (train_idx, test_idx) in enumerate(folds):
    train_ds_counts = df.iloc[train_idx]['dataset'].value_counts()
    test_ds_counts = df.iloc[test_idx]['dataset'].value_counts()
    print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)}, "
          f"test datasets: {dict(test_ds_counts)}")
""")


# =====================================================================
# Cell 8: Routing strategies definition (markdown)
# =====================================================================
md(r"""## Routing Strategies

| # | Strategy | Type | Description |
|---|----------|------|-------------|
| 1 | Oracle | Ceiling | argmin NLL per sample (uses ground truth) |
| 2 | Random | Floor | Uniform random selection |
| 3 | Best-single | Baseline | Always use globally best condition (learned on train fold) |
| 4 | Best-per-dataset | Lookup | Best condition per dataset (uses dataset identity only) |
| 5 | Question-word heuristic | Rule | Route by first word type (wh-word, yes/no, other) |
| 6 | Sentence embedding kNN | ML | k=15 nearest neighbors on query embeddings |
| 7 | Logistic regression (features) | ML | Multinomial LR on hand-crafted text features |
| 8 | Logistic regression (embeddings) | ML | Multinomial LR on 384-dim sentence embeddings |
| 9 | Gradient-boosted classifier | ML | GBT on features + PCA(20) of embeddings |
| 10 | NLL regression (pick min) | ML | Ridge regression predicts NLL per condition, routes to min |

All ML strategies train on oracle labels (argmin NLL) using 5-fold stratified CV.""")


# =====================================================================
# Cell 9: Implement routing strategies + evaluation harness
# =====================================================================
code(r"""def get_pool_nll_matrix(df_in, pool):
    # Return (N, K) NLL matrix and condition names for a pool.
    conds = pool['conditions']
    cols = [f'nll_{c}' for c in conds]
    return df_in[cols].values, conds


def oracle_route(nll_matrix, **kwargs):
    # Oracle: pick condition with minimum NLL per sample.
    return np.argmin(nll_matrix, axis=1)


def random_route(nll_matrix, rng=None, **kwargs):
    # Random: uniform random selection.
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, nll_matrix.shape[1], size=nll_matrix.shape[0])


def best_single_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # Best-single: always use globally best condition (learned on train fold).
    if train_idx is not None:
        # Learn best condition on training fold
        train_mean = nll_matrix[train_idx].mean(axis=0)
        best = np.argmin(train_mean)
    else:
        best = np.argmin(nll_matrix.mean(axis=0))
    return np.full(nll_matrix.shape[0], best)


def best_per_dataset_route(nll_matrix, df_in=None, train_idx=None, test_idx=None, **kwargs):
    # Best-per-dataset: best condition per dataset (learned on train fold).
    preds = np.zeros(nll_matrix.shape[0], dtype=int)
    for ds in DATASETS:
        ds_mask = (df_in['dataset'] == ds).values
        if train_idx is not None:
            # Learn best condition on training fold for this dataset
            train_ds_mask = ds_mask.copy()
            train_ds_mask[test_idx] = False
            if train_ds_mask.any():
                ds_mean = nll_matrix[train_ds_mask].mean(axis=0)
                best = np.argmin(ds_mean)
            else:
                best = np.argmin(nll_matrix[ds_mask].mean(axis=0))
        else:
            best = np.argmin(nll_matrix[ds_mask].mean(axis=0))
        preds[ds_mask] = best
    return preds


def question_word_route(nll_matrix, df_in=None, train_idx=None, test_idx=None, **kwargs):
    # Route by question word type: wh-word, yes/no, other.
    queries_lower = df_in['query'].str.lower().str.strip()
    first_words = queries_lower.str.split().str[0].fillna('')

    # Classify question type
    qtypes = np.where(
        first_words.isin(WH_WORDS), 'wh',
        np.where(first_words.isin(YESNO_WORDS), 'yesno', 'other')
    )

    preds = np.zeros(nll_matrix.shape[0], dtype=int)
    for qtype in ['wh', 'yesno', 'other']:
        mask = (qtypes == qtype)
        if train_idx is not None:
            train_mask = mask.copy()
            train_mask[test_idx] = False
            if train_mask.any():
                best = np.argmin(nll_matrix[train_mask].mean(axis=0))
            else:
                best = np.argmin(nll_matrix[mask].mean(axis=0))
        else:
            best = np.argmin(nll_matrix[mask].mean(axis=0))
        preds[mask] = best
    return preds


def knn_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # kNN on sentence embeddings.
    oracle = np.argmin(nll_matrix, axis=1)
    X = query_embeddings
    clf = KNeighborsClassifier(n_neighbors=15, metric='cosine', weights='distance')
    clf.fit(X[train_idx], oracle[train_idx])
    preds = np.full(nll_matrix.shape[0], -1, dtype=int)
    preds[test_idx] = clf.predict(X[test_idx])
    preds[train_idx] = oracle[train_idx]  # train fold gets oracle (not evaluated)
    return preds


def lr_features_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # Logistic regression on hand-crafted text features.
    oracle = np.argmin(nll_matrix, axis=1)
    X = text_features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_scaled[train_idx], oracle[train_idx])
    preds = np.full(nll_matrix.shape[0], -1, dtype=int)
    preds[test_idx] = clf.predict(X_scaled[test_idx])
    preds[train_idx] = oracle[train_idx]
    return preds


def lr_embeddings_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # Logistic regression on 384-dim sentence embeddings.
    oracle = np.argmin(nll_matrix, axis=1)
    X = query_embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_scaled[train_idx], oracle[train_idx])
    preds = np.full(nll_matrix.shape[0], -1, dtype=int)
    preds[test_idx] = clf.predict(X_scaled[test_idx])
    preds[train_idx] = oracle[train_idx]
    return preds


def gbt_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # Gradient-boosted classifier on features + PCA(20) of embeddings.
    oracle = np.argmin(nll_matrix, axis=1)
    X = np.hstack([text_features.values, embed_pca])
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42, subsample=0.8,
    )
    clf.fit(X[train_idx], oracle[train_idx])
    preds = np.full(nll_matrix.shape[0], -1, dtype=int)
    preds[test_idx] = clf.predict(X[test_idx])
    preds[train_idx] = oracle[train_idx]
    return preds


def nll_regression_route(nll_matrix, train_idx=None, test_idx=None, **kwargs):
    # Ridge regression predicts NLL per condition, route to predicted min.
    K = nll_matrix.shape[1]
    X = np.hstack([text_features.values, embed_pca])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train one Ridge regressor per condition
    pred_nlls = np.zeros((nll_matrix.shape[0], K))
    for k in range(K):
        reg = Ridge(alpha=1.0)
        reg.fit(X_scaled[train_idx], nll_matrix[train_idx, k])
        pred_nlls[:, k] = reg.predict(X_scaled)

    preds = np.full(nll_matrix.shape[0], -1, dtype=int)
    preds[test_idx] = np.argmin(pred_nlls[test_idx], axis=1)
    preds[train_idx] = np.argmin(nll_matrix[train_idx], axis=1)  # oracle on train
    return preds


STRATEGIES = [
    ('Oracle', oracle_route, 'ceiling'),
    ('Random', random_route, 'floor'),
    ('Best-single', best_single_route, 'baseline'),
    ('Best-per-dataset', best_per_dataset_route, 'baseline'),
    ('Question-word', question_word_route, 'heuristic'),
    ('kNN (embeddings)', knn_route, 'ml'),
    ('LR (features)', lr_features_route, 'ml'),
    ('LR (embeddings)', lr_embeddings_route, 'ml'),
    ('GBT (feat+emb)', gbt_route, 'ml'),
    ('NLL regression', nll_regression_route, 'ml'),
]


def evaluate_router(nll_matrix, route_fn, df_in, folds, strategy_type):
    # Evaluate a routing strategy using cross-validation.
    # Returns dict with: routed NLLs (per-sample), hit rate, metrics vs bare/oracle/best-single.
    N = nll_matrix.shape[0]
    routed_nlls = np.full(N, np.nan)
    predicted_choices = np.full(N, -1, dtype=int)
    oracle_choices = np.argmin(nll_matrix, axis=1)

    if strategy_type == 'ceiling':
        # Oracle: no CV needed
        predicted_choices = oracle_choices.copy()
        routed_nlls = nll_matrix[np.arange(N), oracle_choices]
    elif strategy_type == 'floor':
        # Random: no CV needed
        rng = np.random.default_rng(42)
        predicted_choices = rng.integers(0, nll_matrix.shape[1], size=N)
        routed_nlls = nll_matrix[np.arange(N), predicted_choices]
    else:
        # CV-based strategies
        for fold_i, (train_idx, test_idx) in enumerate(folds):
            choices = route_fn(
                nll_matrix,
                df_in=df_in,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            predicted_choices[test_idx] = choices[test_idx]
            routed_nlls[test_idx] = nll_matrix[test_idx, choices[test_idx]]

    assert not np.isnan(routed_nlls).any(), "Some samples have NaN routed NLLs"

    # Compute metrics
    bare_nlls = df_in['nll_bare'].values

    # Best-single baseline NLLs (using overall best condition in pool)
    best_cond_idx = np.argmin(nll_matrix.mean(axis=0))
    best_single_nlls = nll_matrix[:, best_cond_idx]

    oracle_nlls = nll_matrix[np.arange(N), oracle_choices]

    diff_vs_bare = bare_nlls - routed_nlls
    diff_vs_best_single = best_single_nlls - routed_nlls

    return {
        'routed_nlls': routed_nlls,
        'predicted_choices': predicted_choices,
        'oracle_choices': oracle_choices,
        'mean_nll': float(np.mean(routed_nlls)),
        'd_vs_bare': cohens_d(diff_vs_bare),
        'win_vs_bare': win_rate(diff_vs_bare),
        'p_vs_bare': paired_ttest(diff_vs_bare)[1],
        'd_vs_best_single': cohens_d(diff_vs_best_single),
        'win_vs_best_single': win_rate(diff_vs_best_single),
        'regret': float(np.mean(routed_nlls - oracle_nlls)),
        'hit_rate': float(np.mean(predicted_choices == oracle_choices)),
    }


print(f"Defined {len(STRATEGIES)} routing strategies")
print("Evaluation harness ready")
""")


# =====================================================================
# Cell 10: Run all strategies × all pools
# =====================================================================
code(r"""# Run all strategies × all pools
all_results = {}

for pool_name in sorted(POOLS.keys()):
    pool = POOLS[pool_name]
    nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
    K = len(cond_names)
    print(f"\n{'='*60}")
    print(f"Pool {pool_name} (K={K}): {pool['description']}")
    print(f"  Conditions: {cond_names}")
    print(f"{'='*60}")

    for strat_name, strat_fn, strat_type in STRATEGIES:
        result = evaluate_router(nll_matrix, strat_fn, df, folds, strat_type)
        result['pool'] = pool_name
        result['strategy'] = strat_name
        result['strategy_type'] = strat_type
        result['K'] = K
        key = (pool_name, strat_name)
        all_results[key] = result

        print(f"  {strat_name:25s}: d_bare={result['d_vs_bare']:+.3f}, "
              f"d_best={result['d_vs_best_single']:+.3f}, "
              f"regret={result['regret']:.4f}, "
              f"hit={result['hit_rate']:.3f}")

print(f"\nTotal evaluations: {len(all_results)}")
""")


# =====================================================================
# Cell 11: Results DataFrame + summary table
# =====================================================================
code(r"""# Build results DataFrame
result_rows = []
for (pool_name, strat_name), r in all_results.items():
    result_rows.append({
        'pool': pool_name,
        'strategy': strat_name,
        'strategy_type': r['strategy_type'],
        'K': r['K'],
        'mean_nll': r['mean_nll'],
        'd_vs_bare': r['d_vs_bare'],
        'win_vs_bare': r['win_vs_bare'],
        'p_vs_bare': r['p_vs_bare'],
        'd_vs_best_single': r['d_vs_best_single'],
        'win_vs_best_single': r['win_vs_best_single'],
        'regret': r['regret'],
        'hit_rate': r['hit_rate'],
    })

results_df = pd.DataFrame(result_rows)
results_df = results_df.sort_values(['pool', 'd_vs_bare'], ascending=[True, False])

# Display summary table
print("=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)
for pool_name in sorted(POOLS.keys()):
    pool_df = results_df[results_df['pool'] == pool_name].copy()
    print(f"\nPool {pool_name} (K={POOLS[pool_name]['conditions'].__len__()}): "
          f"{POOLS[pool_name]['description']}")
    print("-" * 95)
    print(f"{'Strategy':25s} {'d(bare)':>8s} {'win(bare)':>9s} {'d(best)':>8s} "
          f"{'win(best)':>9s} {'regret':>8s} {'hit%':>6s}")
    print("-" * 95)
    for _, row in pool_df.iterrows():
        print(f"{row['strategy']:25s} {row['d_vs_bare']:+8.3f} {row['win_vs_bare']:9.3f} "
              f"{row['d_vs_best_single']:+8.3f} {row['win_vs_best_single']:9.3f} "
              f"{row['regret']:8.4f} {row['hit_rate']:6.1%}")
""")


# =====================================================================
# Cell 12: Analysis section header
# =====================================================================
md(r"""## Analysis

Nine figures examining routing headroom, strategy performance, and feature importance.""")


# =====================================================================
# Cell 13: Fig 1 — Oracle headroom
# =====================================================================
code(r"""# Fig 1: Oracle headroom — best-single vs oracle gap per pool/dataset
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

for ax_i, pool_name in enumerate(sorted(POOLS.keys())):
    ax = axes[ax_i]
    pool = POOLS[pool_name]
    nll_matrix, cond_names = get_pool_nll_matrix(df, pool)

    # Per-dataset comparison
    best_single_idx = np.argmin(nll_matrix.mean(axis=0))
    best_single_nlls = nll_matrix[:, best_single_idx]
    oracle_nlls = np.min(nll_matrix, axis=1)

    ds_data = []
    for ds in DATASETS:
        mask = (df['dataset'] == ds).values
        bs_mean = best_single_nlls[mask].mean()
        or_mean = oracle_nlls[mask].mean()
        gap = bs_mean - or_mean
        ds_data.append({'dataset': DS_LABELS[ds], 'best_single': bs_mean,
                        'oracle': or_mean, 'gap': gap})

    ds_df = pd.DataFrame(ds_data)

    x = np.arange(len(DATASETS))
    width = 0.35
    bars1 = ax.bar(x - width/2, ds_df['best_single'], width, label='Best-single',
                   color=sns.color_palette('Set2')[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, ds_df['oracle'], width, label='Oracle',
                   color=sns.color_palette('Set2')[1], alpha=0.85)

    ax.set_title(f'Pool {pool_name} (K={len(cond_names)})', fontsize=12, fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], rotation=45, ha='right', fontsize=8)
    if ax_i == 0:
        ax.set_ylabel('Mean NLL')
        ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Oracle Headroom: Best-Single vs Oracle per Pool', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(CHART_DIR / "fig1_oracle_headroom.png", bbox_inches='tight')
plt.show()
print("Saved fig1_oracle_headroom.png")
""")


# =====================================================================
# Cell 14: Fig 2 — Condition dominance pie charts
# =====================================================================
code(r"""# Fig 2: Condition dominance — pie charts showing oracle winner distribution
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

pool_colors = {}
for pool_name in sorted(POOLS.keys()):
    conds = POOLS[pool_name]['conditions']
    pool_colors[pool_name] = dict(zip(conds, sns.color_palette('Set2', len(conds))))

for ax_i, pool_name in enumerate(sorted(POOLS.keys())):
    ax = axes[ax_i]
    pool = POOLS[pool_name]
    nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
    oracle_choices = np.argmin(nll_matrix, axis=1)

    counts = Counter(oracle_choices)
    labels = [cond_names[k] for k in sorted(counts.keys())]
    sizes = [counts[k] for k in sorted(counts.keys())]
    colors = [pool_colors[pool_name][cond_names[k]] for k in sorted(counts.keys())]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.0f%%', colors=colors,
        pctdistance=0.75, startangle=90, textprops={'fontsize': 8}
    )
    ax.set_title(f'Pool {pool_name} (K={len(cond_names)})', fontsize=11)

    # Short legend
    short_labels = []
    for l, s in zip(labels, sizes):
        short = l.replace('transfer_', 'T:').replace('_nonorm', '').replace('_rand', '_r')
        short_labels.append(f'{short} ({s})')
    ax.legend(short_labels, fontsize=6.5, loc='center left', bbox_to_anchor=(-0.2, 0.5))

fig.suptitle('Oracle Condition Winner Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(CHART_DIR / "fig2_condition_dominance.png", bbox_inches='tight')
plt.show()
print("Saved fig2_condition_dominance.png")
""")


# =====================================================================
# Cell 15: Fig 3 — Strategy comparison grouped bars
# =====================================================================
code(r"""# Fig 3: Strategy comparison — grouped bars, d vs bare (main result)
strategy_order = [s[0] for s in STRATEGIES]
pool_names = sorted(POOLS.keys())
n_strats = len(strategy_order)
n_pools = len(pool_names)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(n_strats)
width = 0.18
pool_palette = sns.color_palette('Set2', n_pools)

for pi, pool_name in enumerate(pool_names):
    vals = []
    for strat_name in strategy_order:
        key = (pool_name, strat_name)
        vals.append(all_results[key]['d_vs_bare'])
    ax.bar(x + pi * width - (n_pools - 1) * width / 2, vals, width,
           label=f'Pool {pool_name}', color=pool_palette[pi], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(strategy_order, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Cohen's d vs bare", fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.legend(fontsize=9, ncol=n_pools)
ax.set_title("Routing Strategy Comparison: Cohen's d vs Bare Baseline", fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(CHART_DIR / "fig3_strategy_comparison.png", bbox_inches='tight')
plt.show()
print("Saved fig3_strategy_comparison.png")
""")


# =====================================================================
# Cell 16: Fig 4 — Regret analysis
# =====================================================================
code(r"""# Fig 4: Regret analysis — line chart, gap from oracle
fig, ax = plt.subplots(figsize=(12, 5))

strategy_order_no_oracle = [s[0] for s in STRATEGIES if s[0] != 'Oracle']
pool_palette = sns.color_palette('Set2', len(pool_names))
markers = ['o', 's', 'D', '^']

for pi, pool_name in enumerate(pool_names):
    regrets = []
    for strat_name in strategy_order_no_oracle:
        key = (pool_name, strat_name)
        regrets.append(all_results[key]['regret'])
    ax.plot(range(len(strategy_order_no_oracle)), regrets,
            marker=markers[pi], label=f'Pool {pool_name}',
            color=pool_palette[pi], linewidth=1.5, markersize=6)

ax.set_xticks(range(len(strategy_order_no_oracle)))
ax.set_xticklabels(strategy_order_no_oracle, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Regret (NLL above oracle)', fontsize=11)
ax.set_title('Routing Regret: NLL Gap from Oracle', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

plt.tight_layout()
fig.savefig(CHART_DIR / "fig4_regret_analysis.png", bbox_inches='tight')
plt.show()
print("Saved fig4_regret_analysis.png")
""")


# =====================================================================
# Cell 17: Fig 5 — Win rate vs best-single heatmap
# =====================================================================
code(r"""# Fig 5: Win rate vs best-single — heatmap
strategy_order_filtered = [s[0] for s in STRATEGIES if s[0] not in ('Oracle', 'Random')]

heatmap_data = np.zeros((len(strategy_order_filtered), len(pool_names)))
for si, strat_name in enumerate(strategy_order_filtered):
    for pi, pool_name in enumerate(pool_names):
        key = (pool_name, strat_name)
        heatmap_data[si, pi] = all_results[key]['win_vs_best_single']

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(heatmap_data, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0.5, vmin=0.3, vmax=0.7,
            xticklabels=[f'Pool {p}' for p in pool_names],
            yticklabels=strategy_order_filtered,
            cbar_kws={'label': 'Win rate vs best-single'})
ax.set_title('Win Rate vs Best-Single Baseline', fontsize=13, fontweight='bold')
ax.set_xlabel('')

plt.tight_layout()
fig.savefig(CHART_DIR / "fig5_win_vs_best_single.png", bbox_inches='tight')
plt.show()
print("Saved fig5_win_vs_best_single.png")
""")


# =====================================================================
# Cell 18: Fig 6 — Per-dataset heatmap for best ML router
# =====================================================================
code(r"""# Fig 6: Per-dataset heatmap for best ML router
# Find best ML router (highest mean d_vs_bare across pools)
ml_strats = [s[0] for s in STRATEGIES if s[2] == 'ml']
ml_mean_d = {}
for strat_name in ml_strats:
    ds = [all_results[(p, strat_name)]['d_vs_bare'] for p in pool_names]
    ml_mean_d[strat_name] = np.mean(ds)
best_ml = max(ml_mean_d, key=ml_mean_d.get)
print(f"Best ML router: {best_ml} (mean d_vs_bare = {ml_mean_d[best_ml]:.3f})")

fig, axes = plt.subplots(1, len(pool_names), figsize=(16, 5), sharey=True)

for pi, pool_name in enumerate(pool_names):
    ax = axes[pi]
    pool = POOLS[pool_name]
    nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
    key = (pool_name, best_ml)
    r = all_results[key]

    # Per-dataset metrics
    ds_metrics = []
    for ds in DATASETS:
        mask = (df['dataset'] == ds).values
        bare_ds = df.loc[mask, 'nll_bare'].values
        routed_ds = r['routed_nlls'][mask]
        diff = bare_ds - routed_ds
        ds_metrics.append({
            'dataset': DS_LABELS[ds],
            'd_vs_bare': cohens_d(diff),
            'win_vs_bare': win_rate(diff),
        })

    ds_df = pd.DataFrame(ds_metrics)
    data = ds_df[['d_vs_bare', 'win_vs_bare']].values.T

    sns.heatmap(data, ax=ax, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0.0,
                xticklabels=[DS_LABELS[ds] for ds in DATASETS],
                yticklabels=["d vs bare", "win rate"] if pi == 0 else ["", ""],
                cbar=pi == len(pool_names) - 1)
    ax.set_title(f'Pool {pool_name}', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

fig.suptitle(f'Per-Dataset Performance: {best_ml}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(CHART_DIR / "fig6_per_dataset_best_ml.png", bbox_inches='tight')
plt.show()
print("Saved fig6_per_dataset_best_ml.png")
""")


# =====================================================================
# Cell 19: Fig 7 — Feature importance from GBT
# =====================================================================
code(r"""# Fig 7: Feature importance — top-20 from GBT (trained on full data, Pool C)
# Retrain GBT on all data for feature importance analysis
pool = POOLS['C']
nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
oracle_choices = np.argmin(nll_matrix, axis=1)

X_full = np.hstack([text_features.values, embed_pca])
feature_names = list(text_features.columns) + [f'emb_pca_{i}' for i in range(embed_pca.shape[1])]

gbt_full = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, subsample=0.8,
)
gbt_full.fit(X_full, oracle_choices)

importances = gbt_full.feature_importances_
sorted_idx = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(20)
ax.barh(y_pos, importances[sorted_idx][::-1], color=sns.color_palette('viridis', 20), alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('GBT Feature Importance (Pool C, Top 20)', fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(CHART_DIR / "fig7_feature_importance.png", bbox_inches='tight')
plt.show()
print("Saved fig7_feature_importance.png")

# Print top features
print("\nTop 10 features:")
for rank, idx in enumerate(sorted_idx[:10]):
    print(f"  {rank+1}. {feature_names[idx]:25s} importance={importances[idx]:.4f}")
""")


# =====================================================================
# Cell 20: Fig 8 — Cost-benefit K curve
# =====================================================================
code(r"""# Fig 8: Cost-benefit K curve — diminishing returns of more caches
# For each K from 1..max, find the best subset of K conditions and compute oracle d
# Use Pool B (K=5) and Pool D (K=7) to show progression

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_i, pool_name in enumerate(['B', 'D']):
    ax = axes[ax_i]
    pool = POOLS[pool_name]
    nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
    K_max = len(cond_names)
    bare_nlls = df['nll_bare'].values

    # For each K, try all subsets of size K and pick best by oracle d
    from itertools import combinations

    k_values = list(range(1, K_max + 1))
    oracle_ds = []
    best_single_ds = []

    for k in k_values:
        best_d = -np.inf
        best_bs_d = -np.inf
        for combo in combinations(range(K_max), k):
            sub_matrix = nll_matrix[:, combo]
            oracle_nlls = np.min(sub_matrix, axis=1)
            d = cohens_d(bare_nlls - oracle_nlls)
            if d > best_d:
                best_d = d

            # Also track best-single for this subset
            for ci in combo:
                single_d = cohens_d(bare_nlls - nll_matrix[:, ci])
                if single_d > best_bs_d:
                    best_bs_d = single_d

        oracle_ds.append(best_d)
        best_single_ds.append(best_bs_d)

    ax.plot(k_values, oracle_ds, 'o-', label='Oracle (best subset)', color=sns.color_palette('Set2')[1],
            linewidth=2, markersize=7)
    ax.plot(k_values, best_single_ds, 's--', label='Best-single (best subset)', color=sns.color_palette('Set2')[0],
            linewidth=1.5, markersize=6)
    ax.fill_between(k_values, best_single_ds, oracle_ds, alpha=0.15, color=sns.color_palette('Set2')[1])

    ax.set_xlabel('Number of caches (K)', fontsize=11)
    ax.set_ylabel("Cohen's d vs bare", fontsize=11)
    ax.set_title(f'Pool {pool_name}: Diminishing Returns', fontsize=12, fontweight='medium')
    ax.set_xticks(k_values)
    ax.legend(fontsize=9)

fig.suptitle('Cost-Benefit: More Caches vs Routing Gain', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(CHART_DIR / "fig8_cost_benefit_k.png", bbox_inches='tight')
plt.show()
print("Saved fig8_cost_benefit_k.png")
""")


# =====================================================================
# Cell 21: Fig 9 — Confusion matrix for best router
# =====================================================================
code(r"""# Fig 9: Confusion matrix — oracle vs predicted for best ML router (Pool C)
pool_name = 'C'
pool = POOLS[pool_name]
nll_matrix, cond_names = get_pool_nll_matrix(df, pool)
key = (pool_name, best_ml)
r = all_results[key]

oracle_choices = r['oracle_choices']
predicted_choices = r['predicted_choices']

# Compute confusion matrix
labels = list(range(len(cond_names)))
cm = confusion_matrix(oracle_choices, predicted_choices, labels=labels)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw counts
short_names = [c.replace('_nonorm', '').replace('_rand', '_r') for c in cond_names]
sns.heatmap(cm, ax=axes[0], annot=True, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Oracle')
axes[0].set_title('Confusion Matrix (counts)', fontsize=12)

# Normalized (row-wise)
sns.heatmap(cm_normalized, ax=axes[1], annot=True, fmt='.2f', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            vmin=0, vmax=1)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Oracle')
axes[1].set_title('Confusion Matrix (normalized)', fontsize=12)

fig.suptitle(f'Router Confusion: {best_ml} on Pool {pool_name}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(CHART_DIR / "fig9_confusion_matrix.png", bbox_inches='tight')
plt.show()
print("Saved fig9_confusion_matrix.png")
""")


# =====================================================================
# Cell 22: Save results
# =====================================================================
code(r"""# Save results
import datetime

# Summary JSON
summary_out = {
    'experiment': 'exp15_kv_cache_routing',
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_samples': len(df),
    'n_folds': N_FOLDS,
    'pools': {k: {'K': len(v['conditions']), 'conditions': v['conditions'],
                   'description': v['description']} for k, v in POOLS.items()},
    'strategies': [s[0] for s in STRATEGIES],
    'results': {},
}

for (pool_name, strat_name), r in all_results.items():
    key_str = f"{pool_name}_{strat_name}"
    summary_out['results'][key_str] = {
        'pool': pool_name,
        'strategy': strat_name,
        'strategy_type': r['strategy_type'],
        'mean_nll': r['mean_nll'],
        'd_vs_bare': r['d_vs_bare'],
        'win_vs_bare': r['win_vs_bare'],
        'p_vs_bare': r['p_vs_bare'],
        'd_vs_best_single': r['d_vs_best_single'],
        'win_vs_best_single': r['win_vs_best_single'],
        'regret': r['regret'],
        'hit_rate': r['hit_rate'],
    }

(RESULTS_DIR / "summary.json").write_text(json.dumps(summary_out, indent=2))
print(f"Saved summary.json ({len(summary_out['results'])} entries)")

# Results CSV
results_df.to_csv(RESULTS_DIR / "results.csv", index=False)
print(f"Saved results.csv ({len(results_df)} rows)")

# Set permissions
for f in RESULTS_DIR.rglob('*'):
    try:
        f.chmod(0o777)
    except Exception:
        pass

print(f"\nAll results saved to {RESULTS_DIR}")
""")


# =====================================================================
# Cell 23: Final summary table
# =====================================================================
code(r"""# Final summary: key findings
print("=" * 80)
print("EXP 15 — KV CACHE ROUTING: KEY FINDINGS")
print("=" * 80)

for pool_name in sorted(POOLS.keys()):
    pool = POOLS[pool_name]
    K = len(pool['conditions'])
    print(f"\n--- Pool {pool_name} (K={K}): {pool['description']} ---")

    # Oracle headroom
    oracle_r = all_results[(pool_name, 'Oracle')]
    best_single_r = all_results[(pool_name, 'Best-single')]
    headroom = oracle_r['d_vs_bare'] - best_single_r['d_vs_bare']
    print(f"  Oracle d={oracle_r['d_vs_bare']:.3f}, Best-single d={best_single_r['d_vs_bare']:.3f}, "
          f"Headroom={headroom:.3f}")

    # Best ML router
    ml_results = {s: all_results[(pool_name, s)] for s in ml_strats}
    best_ml_name = max(ml_results, key=lambda s: ml_results[s]['d_vs_bare'])
    best_ml_r = ml_results[best_ml_name]
    gap_closed = (best_ml_r['d_vs_bare'] - best_single_r['d_vs_bare']) / max(headroom, 1e-6)
    print(f"  Best ML: {best_ml_name}, d={best_ml_r['d_vs_bare']:.3f}, "
          f"gap closed={gap_closed:.1%}")

    # Does any ML router beat best-single?
    ml_beats_best = {s: r['d_vs_best_single'] for s, r in ml_results.items() if r['d_vs_best_single'] > 0}
    if ml_beats_best:
        print(f"  ML routers beating best-single: {ml_beats_best}")
    else:
        print(f"  No ML router beats best-single")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/decoder_only/15/15_kv_cache_routing.ipynb"
nbf.write(nb, out_path)
print(f"\nNotebook written to {out_path}")
print(f"Total cells: {len(nb.cells)}")
