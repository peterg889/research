#!/usr/bin/env python3
"""Generate the Exp 32 notebook: Ad-Serving Pipeline."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")
                  if isinstance(source, str) else source})

def code(source):
    lines = source.split("\n") if isinstance(source, str) else source
    # Ensure each line ends with \n except the last
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": formatted,
                  "outputs": [], "execution_count": None})

# ============================================================
# Cell 1: Markdown overview
# ============================================================
md("""# Experiment 32: Ad-Serving Pipeline
## Signal Fusion, Generation Quality, and Commercial Domain Evaluation

Three-part experiment combining our strongest findings into an ad-serving pipeline.

### Part A -- AL+QL Signal Fusion for Ranking
Exp 31 showed answer-likelihood (AL) and query-likelihood (QL) are nearly uncorrelated
(r=0.111). We test whether combining these orthogonal signals improves ranking beyond
either alone. **Uses existing Exp 31 data -- no new model inference needed.**

### Part B -- Generation Quality with Primed Caches
We've measured NLL improvement obsessively but never measured what it means for generated
text. Does d=+0.35 NLL improvement translate to measurably better generated answers?
If priming improves generation quality, the research has immediate ad-serving value.

### Part C -- Commercial Domain (Amazon ESCI / MS MARCO Commercial)
MS MARCO is an information retrieval dataset. Real ad serving involves commercial queries
and product descriptions. We test whether our findings transfer to commercial domains
where product descriptions (~30-150 words) fall squarely in the priming sweet spot.

### Success Criteria
- **Part A**: Fusion AUC > 0.841 (PMI-AL alone). Even +0.005 is meaningful.
- **Part B**: Primed generation has higher Token F1 / contains-answer rate than bare.
- **Part C**: QL performs better on diverse commercial pools than on MS MARCO (AUC > 0.60).
""")

# ============================================================
# Cell 2: Imports and setup
# ============================================================
code("""# Cell 2: Imports and setup
import os
os.umask(0o000)

import sys
import json
import time
import re
import gc
import csv
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, ".")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp32")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results dir: {RESULTS_DIR}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {gpu.total_memory / 1e9:.1f} GB")""")

# ============================================================
# Cell 3: Load model
# ============================================================
code("""# Cell 3: Load Gemma 3 4B
from lib.config import ExperimentConfig
from lib.model_utils import load_model
from lib.kv_cache import (
    _get_text_config, _get_head_dim,
    _get_cache_keys, _get_cache_values,
    _set_cache_keys, _set_cache_values,
    _ensure_dynamic_cache, deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    replace_values_at_layers,
    score_answer_with_cache,
)
from lib.analysis import cohens_d
from lib.data import count_words

MODEL_NAME = "google/gemma-3-4b-it"
config = ExperimentConfig(
    model_name=MODEL_NAME, model_type="gemma3",
    compute_dtype="auto", use_4bit=True, seed=SEED,
)

print(f"\\nLoading Gemma 3 4B...")
model, tokenizer = load_model(config)

text_config = _get_text_config(model.config)
NUM_LAYERS = text_config.num_hidden_layers
HIDDEN_SIZE = text_config.hidden_size
DEVICE = config.device

print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")""")

# ============================================================
# Cell 4: Load soft prefix + define all helpers
# ============================================================
code(r"""# Cell 4: Load soft prefix and define helper functions
CUTOFF = 16  # layers 0-15
QUERY_TEMPLATE = "\nQuery: {query}\nAnswer:"
ANSWER_TEMPLATE = " {answer}"
QL_NEWLINE_SEP = "\n"
QL_SEARCH_SEP = "\nSearch query: "

# --- Load soft prefix from Exp 25 ---
SOFT_PREFIX_PATH = Path("results/exp25/soft_prefix_fact.pt")
USE_SOFT = False

if SOFT_PREFIX_PATH.exists():
    soft_prefix_raw = torch.load(SOFT_PREFIX_PATH, map_location=DEVICE, weights_only=True)
    if soft_prefix_raw.dim() == 3:
        soft_prefix_embeds = soft_prefix_raw.squeeze(0)  # (11, 2560)
    else:
        soft_prefix_embeds = soft_prefix_raw
    PREFIX_LEN = soft_prefix_embeds.shape[0]
    USE_SOFT = True
    print(f"Loaded soft prefix: shape={soft_prefix_embeds.shape}, dtype={soft_prefix_embeds.dtype}")
else:
    print("Soft prefix not found -- using discrete 'static_fact' prefix")
    PREFIX_TEXT = "What are the key facts I need to know?\n"
    prefix_ids = tokenizer(PREFIX_TEXT, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(DEVICE)
    embed_fn = model.get_input_embeddings()
    with torch.no_grad():
        soft_prefix_embeds = embed_fn(prefix_ids).squeeze(0).float()
    PREFIX_LEN = soft_prefix_embeds.shape[0]
    print(f"Discrete prefix: '{PREFIX_TEXT.strip()}' -> {PREFIX_LEN} tokens")

print(f"Prefix length: {PREFIX_LEN}, Cutoff: {CUTOFF} (layers 0-{CUTOFF-1})")


# ================================================================
# Helper: Build bare cache
# ================================================================
def build_bare_cache(passage_text):
    '''Build bare KV cache from passage text.'''
    ids = tokenizer(passage_text, return_tensors="pt",
                    add_special_tokens=True, padding=False, truncation=False
                    ).input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=True, return_dict=True)
    cache = _ensure_dynamic_cache(out.past_key_values)
    length = ids.shape[1]
    del out, ids
    return cache, length


# ================================================================
# Helper: Build primed cache (layer-selective values)
# ================================================================
def build_primed_cache(passage_text, prefix_embeds=None):
    '''Build hybrid cache: bare keys + primed values at layers 0-CUTOFF.

    Works for both soft (learned) and discrete prefixes by operating in
    embedding space, avoiding BPE boundary issues.
    '''
    if prefix_embeds is None:
        prefix_embeds = soft_prefix_embeds

    # Passage embeddings
    passage_ids = tokenizer(passage_text, return_tensors="pt",
                            add_special_tokens=True, padding=False, truncation=False
                            ).input_ids.to(DEVICE)
    passage_len = passage_ids.shape[1]
    embed_layer = model.get_input_embeddings()

    with torch.no_grad():
        passage_embs = embed_layer(passage_ids)
        pf = prefix_embeds.unsqueeze(0).to(dtype=passage_embs.dtype, device=DEVICE)

        # [prefix_embeds, BOS + passage_embeds]
        full_embeds = torch.cat([pf, passage_embs], dim=1)
        full_mask = torch.ones(1, full_embeds.shape[1], device=DEVICE)

        primed_out = model(inputs_embeds=full_embeds, attention_mask=full_mask,
                           use_cache=True, return_dict=True)
        primed_cache = _ensure_dynamic_cache(primed_out.past_key_values)

        # Bare cache
        bare_out = model(input_ids=passage_ids, attention_mask=torch.ones_like(passage_ids),
                         use_cache=True, return_dict=True)
        bare_cache = _ensure_dynamic_cache(bare_out.past_key_values)

        pf_len = prefix_embeds.shape[0]
        # Splice: bare keys + primed values (passage portion) at early layers
        for layer_idx in range(min(CUTOFF, NUM_LAYERS)):
            primed_v = _get_cache_values(primed_cache, layer_idx)
            # Extract passage portion: positions pf_len to pf_len+passage_len
            _set_cache_values(bare_cache, layer_idx,
                              primed_v[:, :, pf_len:pf_len + passage_len, :])

    del primed_cache, primed_out, bare_out, full_embeds, passage_embs, pf
    return bare_cache, passage_len


# ================================================================
# Helper: Generate text with a pre-built cache
# ================================================================
def generate_with_cache(cache, context_len, prompt, max_new_tokens=64):
    '''Greedy decode from a pre-built cache + prompt.'''
    cache = deepcopy_cache(cache)
    prompt_ids = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(DEVICE)

    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache,
                    use_cache=True, return_dict=True)

    generated = []
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = out.past_key_values

    for _ in range(max_new_tokens):
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated.append(tok_id)
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=cache,
                        use_cache=True, return_dict=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = out.past_key_values

    text = tokenizer.decode(generated, skip_special_tokens=True)
    del cache
    return text.strip()


# ================================================================
# Quality metrics for generation
# ================================================================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction, ground_truth):
    return float(normalize_text(prediction) == normalize_text(ground_truth))

def contains_answer(prediction, ground_truth):
    return float(normalize_text(ground_truth) in normalize_text(prediction))

def passage_grounding(prediction, passage):
    '''Fraction of prediction tokens found in the passage.'''
    pred_tokens = set(normalize_text(prediction).split())
    pass_tokens = set(normalize_text(passage).split())
    if not pred_tokens:
        return 0.0
    return len(pred_tokens & pass_tokens) / len(pred_tokens)

print("All helpers defined.")
print(f"  build_bare_cache(), build_primed_cache()")
print(f"  generate_with_cache(), token_f1(), contains_answer(), passage_grounding()")
print(f"  Using {'SOFT' if USE_SOFT else 'DISCRETE'} prefix ({PREFIX_LEN} vectors)")""")

# ============================================================
# Cell 5: Explain experimental conditions
# ============================================================
code(r"""# Cell 5: Explain experimental conditions
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

print('''
======================================================================
PART A -- AL+QL SIGNAL FUSION (analysis of Exp 31 data)
======================================================================
Exp 31 scored 1692 passages across 200 queries with 6 methods:
  - Raw AL:  NLL(answer | passage + query_template)     AUC=0.828
  - PMI AL:  Raw AL - NLL(answer | BOS + query_template) AUC=0.841
  - Raw QL:  NLL(query | passage + "\\n")                 AUC=0.578
  - PMI QL:  Raw QL - NLL(query | BOS + "\\n")            AUC=0.561
  - Raw QL-s: NLL(query | passage + "\\nSearch query: ")  AUC=0.593
  - PMI QL-s: Raw QL-s - baseline                        AUC=0.568

Key finding: AL-QL correlation r=0.111 (nearly orthogonal).
Hypothesis: Combining orthogonal signals should beat either alone.

Methods:
  1. Linear: score = alpha * PMI_AL + (1-alpha) * PMI_QL, grid-search alpha
  2. LogReg: sklearn LogisticRegression on all 6 features, 5-fold CV by query
  3. Rank fusion: average rank positions from AL and QL rankings

======================================================================
PART B -- GENERATION QUALITY (MS MARCO, N=100 queries)
======================================================================
For each query with a relevant passage:
  Bare cache:   [BOS][passage]  -> score NLL + generate answer
  Primed cache: [soft_prefix values at L0-15][passage] -> score NLL + generate answer

Compare generated text quality:
  - Token F1 vs ground-truth answer
  - Contains-answer rate (ground truth substring in generation)
  - Passage grounding (fraction of generated tokens from passage)
  - NLL of ground-truth answer (reference)

Key hypothesis: NLL improvement -> better passage-grounded generation.

======================================================================
PART C -- COMMERCIAL DOMAIN
======================================================================
Test 1: MS MARCO commercial vs informational query split
  - Filter queries by commercial-intent keywords
  - Compare scoring/ranking performance on each subset
  - Tests whether the commercial domain is easier/harder for our methods

Test 2: Amazon ESCI (Shopping Queries Dataset) if available
  - Product descriptions as passages, search queries as queries
  - Score: NLL(query | product_desc) [QL] and PMI versions
  - Hypothesis: QL works better on diverse pools than MS MARCOs
    topically homogeneous retrieval set (Exp 31: AUC=0.59)
''')""")

# ============================================================
# Cell 6: Part A -- Signal Fusion
# ============================================================
code(r"""# Cell 6: PART A -- AL+QL Signal Fusion
print("=" * 70)
print("PART A: AL+QL SIGNAL FUSION FOR RANKING")
print("=" * 70)

# Load Exp 31 data
exp31_csv = Path("results/exp31/passage_scores.csv")
if not exp31_csv.exists():
    print("ERROR: Exp 31 passage_scores.csv not found. Skipping Part A.")
    fusion_results = None
else:
    df = pd.read_csv(exp31_csv)
    print(f"Loaded {len(df)} passages from Exp 31")
    print(f"Columns: {list(df.columns)}")
    print(f"Queries: {df['query_idx'].nunique()}")
    print(f"Relevant: {df['is_relevant'].sum()} ({100*df['is_relevant'].mean():.1f}%)")

    # --- Method 1: Linear combination grid search ---
    from sklearn.metrics import roc_auc_score

    query_ids = df['query_idx'].unique()
    n_queries = len(query_ids)

    # Cross-validated AUC for a given scoring function
    def cv_auc(score_fn, n_folds=5):
        '''5-fold CV by query, return mean AUC.'''
        np.random.seed(SEED)
        fold_ids = np.random.permutation(n_queries) % n_folds
        aucs = []
        for fold in range(n_folds):
            test_queries = query_ids[fold_ids == fold]
            test_mask = df['query_idx'].isin(test_queries)
            test_df = df[test_mask]
            if test_df['is_relevant'].sum() == 0 or test_df['is_relevant'].sum() == len(test_df):
                continue
            scores = score_fn(test_df)
            # Lower score = more relevant for NLL-based scoring
            aucs.append(roc_auc_score(test_df['is_relevant'], -scores))
        return np.mean(aucs) if aucs else 0.0

    # Baselines
    auc_pmi_al = cv_auc(lambda d: d['pmi_al'].values)
    auc_pmi_ql = cv_auc(lambda d: d['pmi_ql'].values)
    auc_raw_al = cv_auc(lambda d: d['nll_al'].values)
    auc_raw_ql = cv_auc(lambda d: d['nll_ql'].values)
    auc_pmi_qls = cv_auc(lambda d: d['pmi_ql_search'].values)

    print(f"\n--- Baseline AUCs (5-fold CV) ---")
    print(f"  PMI AL:         {auc_pmi_al:.4f}")
    print(f"  Raw AL:         {auc_raw_al:.4f}")
    print(f"  PMI QL:         {auc_pmi_ql:.4f}")
    print(f"  Raw QL:         {auc_raw_ql:.4f}")
    print(f"  PMI QL-search:  {auc_pmi_qls:.4f}")

    # Grid search: alpha * PMI_AL + (1-alpha) * PMI_QL
    alphas = np.arange(0.0, 1.01, 0.05)
    fusion_aucs = []
    for alpha in alphas:
        auc = cv_auc(lambda d, a=alpha: a * d['pmi_al'].values + (1-a) * d['pmi_ql'].values)
        fusion_aucs.append(auc)

    best_alpha_idx = np.argmax(fusion_aucs)
    best_alpha = alphas[best_alpha_idx]
    best_fusion_auc = fusion_aucs[best_alpha_idx]

    print(f"\n--- Linear Fusion: alpha * PMI_AL + (1-alpha) * PMI_QL ---")
    print(f"  Best alpha: {best_alpha:.2f}")
    print(f"  Best AUC:   {best_fusion_auc:.4f} (vs PMI_AL alone: {auc_pmi_al:.4f})")
    print(f"  Delta:      {best_fusion_auc - auc_pmi_al:+.4f}")

    # Also try with QL-search
    fusion_aucs_s = []
    for alpha in alphas:
        auc = cv_auc(lambda d, a=alpha: a * d['pmi_al'].values + (1-a) * d['pmi_ql_search'].values)
        fusion_aucs_s.append(auc)
    best_alpha_s = alphas[np.argmax(fusion_aucs_s)]
    best_fusion_auc_s = max(fusion_aucs_s)

    print(f"\n--- Linear Fusion: alpha * PMI_AL + (1-alpha) * PMI_QL_search ---")
    print(f"  Best alpha: {best_alpha_s:.2f}")
    print(f"  Best AUC:   {best_fusion_auc_s:.4f} (vs PMI_AL alone: {auc_pmi_al:.4f})")
    print(f"  Delta:      {best_fusion_auc_s - auc_pmi_al:+.4f}")

    # --- Method 2: Logistic Regression (all 6 features) ---
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold

        feature_cols = ['nll_al', 'nll_ql', 'nll_ql_search', 'pmi_al', 'pmi_ql', 'pmi_ql_search']
        X = df[feature_cols].values
        y = df['is_relevant'].values
        groups = df['query_idx'].values

        gkf = GroupKFold(n_splits=5)
        logreg_aucs = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if y_test.sum() == 0 or y_test.sum() == len(y_test):
                continue
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X_train_s, y_train)
            proba = clf.predict_proba(X_test_s)[:, 1]
            logreg_aucs.append(roc_auc_score(y_test, proba))

        logreg_auc = np.mean(logreg_aucs)

        # Feature importance from full model
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf_full = LogisticRegression(max_iter=1000, random_state=SEED)
        clf_full.fit(X_s, y)
        coefs = dict(zip(feature_cols, clf_full.coef_[0]))

        print(f"\n--- Logistic Regression (6 features, 5-fold GroupKFold) ---")
        print(f"  AUC: {logreg_auc:.4f} (vs PMI_AL alone: {auc_pmi_al:.4f})")
        print(f"  Delta: {logreg_auc - auc_pmi_al:+.4f}")
        print(f"  Feature coefficients (standardized):")
        for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {feat:<18} {coef:+.4f}")
    except ImportError:
        logreg_auc = None
        print("\nsklearn not available -- skipping logistic regression")

    # --- Method 3: Rank fusion ---
    def rank_fusion_auc():
        '''Average rank from AL and QL within each query.'''
        aucs = []
        for qid in query_ids:
            qdf = df[df['query_idx'] == qid].copy()
            if qdf['is_relevant'].sum() == 0 or qdf['is_relevant'].sum() == len(qdf):
                continue
            qdf['rank_al'] = qdf['pmi_al'].rank()
            qdf['rank_ql'] = qdf['pmi_ql'].rank()
            qdf['rank_fusion'] = (qdf['rank_al'] + qdf['rank_ql']) / 2
            aucs.append(roc_auc_score(qdf['is_relevant'], -qdf['rank_fusion']))
        return np.mean(aucs) if aucs else 0.0

    rank_fusion = rank_fusion_auc()
    print(f"\n--- Rank Fusion (avg rank from PMI_AL + PMI_QL) ---")
    print(f"  AUC: {rank_fusion:.4f} (vs PMI_AL alone: {auc_pmi_al:.4f})")
    print(f"  Delta: {rank_fusion - auc_pmi_al:+.4f}")

    # --- MRR@10 comparison ---
    def compute_mrr(score_col, negate=True):
        '''Compute MRR@10 across queries.'''
        mrrs = []
        for qid in query_ids:
            qdf = df[df['query_idx'] == qid].copy()
            scores = qdf[score_col].values if isinstance(score_col, str) else score_col(qdf)
            if negate:
                order = np.argsort(scores)  # lower NLL = better
            else:
                order = np.argsort(-scores)  # higher = better
            relevant = qdf['is_relevant'].values
            for rank, idx in enumerate(order[:10], 1):
                if relevant[idx]:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)
        return np.mean(mrrs)

    mrr_pmi_al = compute_mrr('pmi_al')
    mrr_pmi_ql = compute_mrr('pmi_ql')
    mrr_fusion = compute_mrr(
        lambda d: best_alpha * d['pmi_al'].values + (1-best_alpha) * d['pmi_ql'].values,
        negate=True)

    print(f"\n--- MRR@10 ---")
    print(f"  PMI AL:               {mrr_pmi_al:.4f}")
    print(f"  PMI QL:               {mrr_pmi_ql:.4f}")
    print(f"  Linear fusion (a={best_alpha:.2f}): {mrr_fusion:.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"PART A SUMMARY -- Signal Fusion Results")
    print(f"{'='*70}")
    print(f"{'Method':<40} {'AUC':>8} {'vs PMI-AL':>10}")
    print(f"{'-'*58}")
    print(f"{'PMI AL (baseline)':<40} {auc_pmi_al:>8.4f} {'--':>10}")
    print(f"{'Raw AL':<40} {auc_raw_al:>8.4f} {auc_raw_al-auc_pmi_al:>+10.4f}")
    print(f"{'PMI QL':<40} {auc_pmi_ql:>8.4f} {auc_pmi_ql-auc_pmi_al:>+10.4f}")
    print(f"{'Linear fusion (AL+QL, a={:.2f})'.format(best_alpha):<40} {best_fusion_auc:>8.4f} {best_fusion_auc-auc_pmi_al:>+10.4f}")
    print(f"{'Linear fusion (AL+QL-s, a={:.2f})'.format(best_alpha_s):<40} {best_fusion_auc_s:>8.4f} {best_fusion_auc_s-auc_pmi_al:>+10.4f}")
    if logreg_auc is not None:
        print(f"{'Logistic regression (6 features)':<40} {logreg_auc:>8.4f} {logreg_auc-auc_pmi_al:>+10.4f}")
    print(f"{'Rank fusion (AL+QL avg rank)':<40} {rank_fusion:>8.4f} {rank_fusion-auc_pmi_al:>+10.4f}")

    fusion_results = {
        'auc_pmi_al': auc_pmi_al, 'auc_raw_al': auc_raw_al,
        'auc_pmi_ql': auc_pmi_ql, 'auc_raw_ql': auc_raw_ql,
        'best_linear_alpha': float(best_alpha),
        'best_linear_auc': float(best_fusion_auc),
        'best_linear_s_alpha': float(best_alpha_s),
        'best_linear_s_auc': float(best_fusion_auc_s),
        'logreg_auc': float(logreg_auc) if logreg_auc is not None else None,
        'rank_fusion_auc': float(rank_fusion),
        'mrr_pmi_al': mrr_pmi_al, 'mrr_pmi_ql': mrr_pmi_ql, 'mrr_fusion': mrr_fusion,
        'fusion_curve': [{'alpha': float(a), 'auc': float(v)} for a, v in zip(alphas, fusion_aucs)],
    }
    print(f"\nPrimary: Fusion AUC > 0.841?  {'YES' if best_fusion_auc > 0.841 else 'NO'} (best={best_fusion_auc:.4f})")""")

# ============================================================
# Cell 7: Part B -- Generation Quality
# ============================================================
code(r"""# Cell 7: PART B -- Generation Quality
print("=" * 70)
print("PART B: GENERATION QUALITY WITH PRIMED CACHES")
print("=" * 70)

# Load MS MARCO validation (multi-passage format for relevant passages with answers)
from datasets import load_dataset

N_GEN = 100  # queries for generation eval
MAX_NEW_TOKENS = 64
CHECKPOINT_GEN = RESULTS_DIR / "checkpoint_gen.json"

print(f"Loading MS MARCO v1.1 validation (single-passage, relevant only)...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
gen_samples = []

for item in ds:
    if len(gen_samples) >= N_GEN * 3:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])

    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    if not answer:
        continue

    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            gen_samples.append({'passage': pt, 'query': query, 'answer': answer,
                                'word_count': wc})
            break

np.random.seed(SEED + 100)
np.random.shuffle(gen_samples)
gen_samples = gen_samples[:N_GEN]
del ds
gc.collect()

print(f"Selected {len(gen_samples)} queries with relevant passages")
print(f"Word counts: mean={np.mean([s['word_count'] for s in gen_samples]):.0f}")

# Resume from checkpoint
gen_results = []
gen_start = 0
if CHECKPOINT_GEN.exists():
    ckpt = json.loads(CHECKPOINT_GEN.read_text())
    if ckpt.get('n_total') == N_GEN and len(ckpt.get('results', [])) > 0:
        # Verify query match
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in gen_samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            gen_results = ckpt['results']
            gen_start = len(gen_results)
            print(f"Resuming from checkpoint: {gen_start}/{N_GEN}")

print(f"\nRunning generation evaluation ({gen_start}/{N_GEN} done)...")
t0 = time.time()

for i in tqdm(range(gen_start, N_GEN), initial=gen_start, total=N_GEN, desc="GenQual"):
    sample = gen_samples[i]
    passage, query, answer = sample['passage'], sample['query'], sample['answer']
    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    # Build caches
    bare_cache, bare_len = build_bare_cache(passage)
    primed_cache, primed_len = build_primed_cache(passage)

    # Score NLL
    bare_nll = score_answer_with_cache(
        deepcopy_cache(bare_cache), bare_len,
        query_prompt, answer_text, model, tokenizer, config)
    primed_nll = score_answer_with_cache(
        deepcopy_cache(primed_cache), primed_len,
        query_prompt, answer_text, model, tokenizer, config)

    # Generate answers
    bare_gen = generate_with_cache(bare_cache, bare_len, query_prompt, MAX_NEW_TOKENS)
    primed_gen = generate_with_cache(primed_cache, primed_len, query_prompt, MAX_NEW_TOKENS)

    # Quality metrics
    result = {
        'query': query, 'answer': answer, 'passage_words': sample['word_count'],
        'bare_nll': bare_nll, 'primed_nll': primed_nll,
        'bare_gen': bare_gen, 'primed_gen': primed_gen,
        'bare_f1': token_f1(bare_gen, answer),
        'primed_f1': token_f1(primed_gen, answer),
        'bare_em': exact_match(bare_gen, answer),
        'primed_em': exact_match(primed_gen, answer),
        'bare_contains': contains_answer(bare_gen, answer),
        'primed_contains': contains_answer(primed_gen, answer),
        'bare_grounding': passage_grounding(bare_gen, passage),
        'primed_grounding': passage_grounding(primed_gen, passage),
        'bare_gen_len': len(bare_gen.split()),
        'primed_gen_len': len(primed_gen.split()),
    }
    gen_results.append(result)

    del bare_cache, primed_cache
    gc.collect()
    torch.cuda.empty_cache()

    # Checkpoint every 10
    if (i + 1) % 10 == 0 or i == N_GEN - 1:
        ckpt = {'n_total': N_GEN, 'results': gen_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        CHECKPOINT_GEN.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = i - gen_start + 1
        eta = (N_GEN - i - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {i+1}/{N_GEN} | {elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m")

elapsed = time.time() - t0
print(f"\nGeneration eval complete: {len(gen_results)} queries in {elapsed/60:.1f} min")""")

# ============================================================
# Cell 8: Part B -- Generation Results
# ============================================================
code(r"""# Cell 8: PART B -- Generation Quality Results
print("=" * 70)
print(f"PART B RESULTS -- Generation Quality (N={len(gen_results)})")
print("=" * 70)

# Aggregate metrics
metrics = {
    'NLL': ('bare_nll', 'primed_nll'),
    'Token F1': ('bare_f1', 'primed_f1'),
    'Exact Match': ('bare_em', 'primed_em'),
    'Contains Answer': ('bare_contains', 'primed_contains'),
    'Passage Grounding': ('bare_grounding', 'primed_grounding'),
    'Gen Length (words)': ('bare_gen_len', 'primed_gen_len'),
}

print(f"\n{'Metric':<22} {'Bare':>8} {'Primed':>8} {'Delta':>8} {'p':>12} {'sig':>5}")
print("-" * 65)

gen_analysis = {}
for metric_name, (bare_key, primed_key) in metrics.items():
    bare_vals = np.array([r[bare_key] for r in gen_results])
    primed_vals = np.array([r[primed_key] for r in gen_results])

    bare_mean = np.mean(bare_vals)
    primed_mean = np.mean(primed_vals)
    delta = primed_mean - bare_mean

    # For NLL, lower is better; for others, higher is better
    if metric_name == 'NLL':
        diff = bare_vals - primed_vals  # positive = primed better
    elif metric_name == 'Gen Length (words)':
        diff = primed_vals - bare_vals  # just show direction
    else:
        diff = primed_vals - bare_vals  # positive = primed better

    if np.std(diff) > 0:
        t_stat, p_val = stats.ttest_1samp(diff, 0)
    else:
        t_stat, p_val = 0.0, 1.0
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"{metric_name:<22} {bare_mean:>8.4f} {primed_mean:>8.4f} {delta:>+8.4f} {p_val:>12.2e} {sig:>5}")
    gen_analysis[metric_name] = {
        'bare_mean': float(bare_mean), 'primed_mean': float(primed_mean),
        'delta': float(delta), 'p_value': float(p_val),
    }

# Hardness interaction
bare_nlls = np.array([r['bare_nll'] for r in gen_results])
quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"\n--- Hardness Gradient (by bare NLL quintile) ---")
print(f"{'Quintile':<12} {'N':>4} {'Bare F1':>10} {'Primed F1':>10} {'Delta F1':>10} {'NLL Delta':>10}")
print("-" * 58)

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    bf1 = np.mean([gen_results[j]['bare_f1'] for j in range(len(gen_results)) if mask[j]])
    pf1 = np.mean([gen_results[j]['primed_f1'] for j in range(len(gen_results)) if mask[j]])
    bnll = np.mean([gen_results[j]['bare_nll'] for j in range(len(gen_results)) if mask[j]])
    pnll = np.mean([gen_results[j]['primed_nll'] for j in range(len(gen_results)) if mask[j]])
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    print(f"{qlabel:<12} {n_q:>4} {bf1:>10.4f} {pf1:>10.4f} {pf1-bf1:>+10.4f} {bnll-pnll:>+10.4f}")

# Show example generations
print(f"\n--- Example Generations (3 samples) ---")
for i in [0, len(gen_results)//2, len(gen_results)-1]:
    r = gen_results[i]
    print(f"\nQuery: {r['query'][:80]}")
    print(f"Truth: {r['answer'][:80]}")
    print(f"Bare:  {r['bare_gen'][:80]}  (F1={r['bare_f1']:.3f})")
    print(f"Prime: {r['primed_gen'][:80]}  (F1={r['primed_f1']:.3f})")
    print(f"NLL:   bare={r['bare_nll']:.3f}, primed={r['primed_nll']:.3f}")""")

# ============================================================
# Cell 9: Part C -- Load Commercial Data
# ============================================================
code(r"""# Cell 9: PART C -- Load Commercial Data
print("=" * 70)
print("PART C: COMMERCIAL DOMAIN EVALUATION")
print("=" * 70)

N_COMMERCIAL = 100  # queries for commercial eval
CHECKPOINT_COM = RESULTS_DIR / "checkpoint_commercial.json"

# ================================================================
# Test 1: MS MARCO commercial vs informational split
# ================================================================
COMMERCIAL_KEYWORDS = {
    'buy', 'price', 'cost', 'product', 'review', 'best', 'shop', 'order',
    'cheap', 'deal', 'discount', 'sale', 'recommend', 'brand', 'store',
    'purchase', 'compare', 'worth', 'quality', 'rating', 'how much',
}

def is_commercial(query):
    q = query.lower()
    return any(kw in q for kw in COMMERCIAL_KEYWORDS)

# Load multi-passage format (like Exp 31)
print("Loading MS MARCO v1.1 validation (multi-passage format)...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

commercial_queries = []
informational_queries = []

for item in ds:
    if len(commercial_queries) >= N_COMMERCIAL and len(informational_queries) >= N_COMMERCIAL:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')

    if not ptexts or len(ptexts) < 3:
        continue

    # Filter passages by length
    valid_passages = []
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if 20 <= wc <= 300:
            valid_passages.append({'text': pt, 'is_relevant': int(sel == 1), 'words': wc})

    if len(valid_passages) < 3 or not any(p['is_relevant'] for p in valid_passages):
        continue

    entry = {'query': query, 'passages': valid_passages}

    if is_commercial(query) and len(commercial_queries) < N_COMMERCIAL:
        commercial_queries.append(entry)
    elif not is_commercial(query) and len(informational_queries) < N_COMMERCIAL:
        informational_queries.append(entry)

del ds
gc.collect()

print(f"Commercial queries: {len(commercial_queries)}")
print(f"Informational queries: {len(informational_queries)}")
print(f"Example commercial: {commercial_queries[0]['query'][:60] if commercial_queries else 'N/A'}")
print(f"Example informational: {informational_queries[0]['query'][:60] if informational_queries else 'N/A'}")

# ================================================================
# Test 2: Try loading Amazon ESCI
# ================================================================
esci_queries = []
esci_loaded = False

for source in ["amazon-science/esci-data", "smhavens/esci-s", "tasksource/esci"]:
    try:
        print(f"\nTrying to load ESCI from '{source}'...")
        esci_ds = load_dataset(source, trust_remote_code=True)
        # Inspect structure
        if isinstance(esci_ds, dict):
            print(f"  Splits: {list(esci_ds.keys())}")
            split_name = list(esci_ds.keys())[0]
            esci_split = esci_ds[split_name]
        else:
            esci_split = esci_ds

        print(f"  Columns: {esci_split.column_names}")
        print(f"  Rows: {len(esci_split)}")

        # Try to extract query-product pairs
        if 'query' in esci_split.column_names and 'product_title' in esci_split.column_names:
            # Filter for US locale if available
            if 'product_locale' in esci_split.column_names:
                esci_split = esci_split.filter(lambda x: x.get('product_locale') == 'us')
                print(f"  US-only rows: {len(esci_split)}")

            # Group by query
            from collections import defaultdict
            query_products = defaultdict(list)
            for row in esci_split:
                q = row['query']
                label = row.get('esci_label', 'I')
                title = row.get('product_title', '')
                desc = row.get('product_description', '') or ''
                bullets = row.get('product_bullet_point', '') or ''

                # Build passage
                passage = title
                if bullets:
                    passage += "\n" + bullets[:500]
                elif desc:
                    passage += "\n" + desc[:500]

                wc = count_words(passage)
                if 5 <= wc <= 300 and title:
                    is_rel = 1 if label in ('E', 'S') else 0
                    query_products[q].append({
                        'text': passage, 'is_relevant': is_rel,
                        'words': wc, 'label': label, 'title': title
                    })

            # Filter queries with enough products and at least one relevant
            for q, prods in query_products.items():
                if len(prods) >= 4 and any(p['is_relevant'] for p in prods):
                    esci_queries.append({'query': q, 'passages': prods[:15]})
                if len(esci_queries) >= N_COMMERCIAL:
                    break

            if esci_queries:
                esci_loaded = True
                print(f"  ESCI loaded: {len(esci_queries)} queries")
                print(f"  Total products: {sum(len(q['passages']) for q in esci_queries)}")
                print(f"  Relevant: {sum(sum(p['is_relevant'] for p in q['passages']) for q in esci_queries)}")
                break

        del esci_ds
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if not esci_loaded:
    print("\nESCI not available -- Part C will use MS MARCO commercial/informational split only.")

gc.collect()""")

# ============================================================
# Cell 10: Part C -- Scoring
# ============================================================
code(r"""# Cell 10: PART C -- Score commercial data
print("=" * 70)
print("PART C: SCORING")
print("=" * 70)

def score_query_set(query_set, set_name, do_priming=False, max_queries=None):
    '''Score a set of queries with their passages using QL and AL scoring.'''
    if max_queries:
        query_set = query_set[:max_queries]

    all_scores = []
    t0 = time.time()

    for qi, qdata in enumerate(tqdm(query_set, desc=set_name)):
        query = qdata['query']

        # Baselines (once per query)
        # BL for QL: NLL(query | BOS + "\n")
        bos_cache, bos_len = build_bare_cache("")
        bl_ql = score_answer_with_cache(
            deepcopy_cache(bos_cache), bos_len,
            QL_NEWLINE_SEP, query, model, tokenizer, config)
        bl_ql_s = score_answer_with_cache(
            deepcopy_cache(bos_cache), bos_len,
            QL_SEARCH_SEP, query, model, tokenizer, config)
        del bos_cache

        for pi, pdata in enumerate(qdata['passages']):
            passage = pdata['text']

            # Bare cache
            bare_cache, bare_len = build_bare_cache(passage)

            # QL scores
            nll_ql = score_answer_with_cache(
                deepcopy_cache(bare_cache), bare_len,
                QL_NEWLINE_SEP, query, model, tokenizer, config)
            nll_ql_s = score_answer_with_cache(
                deepcopy_cache(bare_cache), bare_len,
                QL_SEARCH_SEP, query, model, tokenizer, config)

            result = {
                'query_idx': qi, 'passage_idx': pi,
                'is_relevant': pdata['is_relevant'],
                'word_count': pdata.get('words', count_words(passage)),
                'nll_ql': nll_ql, 'nll_ql_search': nll_ql_s,
                'bl_ql': bl_ql, 'bl_ql_search': bl_ql_s,
                'pmi_ql': nll_ql - bl_ql, 'pmi_ql_search': nll_ql_s - bl_ql_s,
            }

            # Priming comparison (subset only)
            if do_priming and qi < 50:
                primed_cache, primed_len = build_primed_cache(passage)
                primed_ql = score_answer_with_cache(
                    deepcopy_cache(primed_cache), primed_len,
                    QL_NEWLINE_SEP, query, model, tokenizer, config)
                result['primed_ql'] = primed_ql
                result['primed_pmi_ql'] = primed_ql - bl_ql
                del primed_cache

            del bare_cache
            all_scores.append(result)

        gc.collect()
        torch.cuda.empty_cache()

        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = (len(query_set) - qi - 1) * elapsed / (qi + 1)
            tqdm.write(f"  {set_name} {qi+1}/{len(query_set)} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"  {set_name}: {len(all_scores)} passages scored in {elapsed/60:.1f} min")
    return all_scores


# Score MS MARCO commercial queries
print("\n--- Scoring MS MARCO Commercial Queries ---")
com_scores = score_query_set(commercial_queries, "Commercial", do_priming=True)

# Score MS MARCO informational queries
print("\n--- Scoring MS MARCO Informational Queries ---")
info_scores = score_query_set(informational_queries, "Informational", do_priming=True)

# Score ESCI (if available)
esci_scores = None
if esci_loaded and esci_queries:
    print("\n--- Scoring Amazon ESCI ---")
    esci_scores = score_query_set(esci_queries, "ESCI", do_priming=True)

# Save checkpoint
com_checkpoint = {
    'commercial_scores': com_scores,
    'informational_scores': info_scores,
    'esci_scores': esci_scores,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}
CHECKPOINT_COM.write_text(json.dumps(com_checkpoint))
print(f"\nCheckpoint saved to {CHECKPOINT_COM}")""")

# ============================================================
# Cell 11: Part C -- Results
# ============================================================
code(r"""# Cell 11: PART C -- Commercial Domain Results
from sklearn.metrics import roc_auc_score

print("=" * 70)
print("PART C RESULTS -- Commercial Domain")
print("=" * 70)

def analyze_scores(scores, set_name):
    '''Compute AUC and MRR for a set of passage scores.'''
    df = pd.DataFrame(scores)
    n_passages = len(df)
    n_queries = df['query_idx'].nunique()
    n_relevant = df['is_relevant'].sum()

    print(f"\n{'='*50}")
    print(f"{set_name}: {n_queries} queries, {n_passages} passages, {n_relevant} relevant ({100*n_relevant/n_passages:.1f}%)")
    print(f"{'='*50}")

    results = {}

    # AUC for each scoring method
    score_cols = [c for c in ['nll_ql', 'nll_ql_search', 'pmi_ql', 'pmi_ql_search'] if c in df.columns]

    print(f"\n{'Method':<25} {'AUC':>8} {'MRR@10':>8}")
    print("-" * 43)

    for col in score_cols:
        try:
            # Lower NLL = more relevant, so negate for AUC
            auc = roc_auc_score(df['is_relevant'], -df[col])
        except ValueError:
            auc = 0.5

        # MRR@10
        mrrs = []
        for qid in df['query_idx'].unique():
            qdf = df[df['query_idx'] == qid]
            order = np.argsort(qdf[col].values)  # lower = better
            relevant = qdf['is_relevant'].values
            for rank, idx in enumerate(order[:10], 1):
                if relevant[idx]:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)
        mrr = np.mean(mrrs)

        print(f"{col:<25} {auc:>8.4f} {mrr:>8.4f}")
        results[col] = {'auc': float(auc), 'mrr': float(mrr)}

    # Priming effect (if available)
    if 'primed_ql' in df.columns:
        primed_df = df.dropna(subset=['primed_ql'])
        if len(primed_df) > 0:
            bare_ql = primed_df['nll_ql'].values
            primed_ql = primed_df['primed_ql'].values
            delta = bare_ql - primed_ql
            d = cohens_d(delta) if np.std(delta) > 0 else 0.0
            win_pct = 100 * np.mean(delta > 0)
            t_stat, p_val = stats.ttest_1samp(delta, 0) if np.std(delta) > 0 else (0, 1)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"\n  Priming effect (NLL QL): d={d:+.3f}, win%={win_pct:.0f}%, p={p_val:.2e} {sig}")
            print(f"  Mean bare QL:   {np.mean(bare_ql):.4f}")
            print(f"  Mean primed QL: {np.mean(primed_ql):.4f}")
            results['priming'] = {'d': float(d), 'win_pct': float(win_pct),
                                  'p_value': float(p_val)}

    # Differential NLL
    rel_mask = df['is_relevant'] == 1
    irr_mask = df['is_relevant'] == 0
    for col in ['nll_ql', 'pmi_ql']:
        if col in df.columns:
            rel_mean = df.loc[rel_mask, col].mean()
            irr_mean = df.loc[irr_mask, col].mean()
            gap = irr_mean - rel_mean  # positive = relevant has lower NLL = good
            print(f"  {col} gap: relevant={rel_mean:.3f}, irrelevant={irr_mean:.3f}, gap={gap:+.3f}")

    return results


# Analyze each set
com_analysis = analyze_scores(com_scores, "MS MARCO -- Commercial")
info_analysis = analyze_scores(info_scores, "MS MARCO -- Informational")

esci_analysis = None
if esci_scores:
    esci_analysis = analyze_scores(esci_scores, "Amazon ESCI")

# Comparison table
print(f"\n{'='*70}")
print(f"PART C SUMMARY -- Cross-Domain Comparison")
print(f"{'='*70}")
print(f"{'Dataset':<30} {'QL AUC':>8} {'PMI QL AUC':>10} {'Exp31 ref':>10}")
print(f"{'-'*60}")

ql_ref = 0.578  # from Exp 31
pmi_ql_ref = 0.561

for name, analysis in [("MS MARCO Commercial", com_analysis),
                        ("MS MARCO Informational", info_analysis),
                        ("Amazon ESCI", esci_analysis)]:
    if analysis is None:
        continue
    ql_auc = analysis.get('nll_ql', {}).get('auc', 0)
    pmi_auc = analysis.get('pmi_ql', {}).get('auc', 0)
    print(f"{name:<30} {ql_auc:>8.4f} {pmi_auc:>10.4f} {ql_ref:>10.3f}")

print(f"\nExp 31 references (full MS MARCO):")
print(f"  Raw QL AUC:  {ql_ref}")
print(f"  PMI QL AUC:  {pmi_ql_ref}")
print(f"  PMI AL AUC:  0.841")

commercial_results = {
    'commercial': com_analysis, 'informational': info_analysis,
    'esci': esci_analysis, 'esci_loaded': esci_loaded,
}""")

# ============================================================
# Cell 12: Final Verdict + Save
# ============================================================
code(r"""# Cell 12: Final Verdict and Save
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Save plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Fusion alpha curve
if fusion_results and 'fusion_curve' in fusion_results:
    ax = axes[0, 0]
    alphas_plot = [p['alpha'] for p in fusion_results['fusion_curve']]
    aucs_plot = [p['auc'] for p in fusion_results['fusion_curve']]
    ax.plot(alphas_plot, aucs_plot, 'b-o', markersize=3, label='Linear fusion')
    ax.axhline(y=fusion_results['auc_pmi_al'], color='r', linestyle='--',
               label=f"PMI-AL alone ({fusion_results['auc_pmi_al']:.4f})")
    ax.set_xlabel('Alpha (weight on PMI-AL)')
    ax.set_ylabel('AUC-ROC (5-fold CV)')
    ax.set_title('Part A: Signal Fusion')
    ax.legend(fontsize=8)

# Panel 2: Generation quality comparison
ax = axes[0, 1]
if gen_results:
    bare_f1s = [r['bare_f1'] for r in gen_results]
    primed_f1s = [r['primed_f1'] for r in gen_results]
    ax.scatter(bare_f1s, primed_f1s, alpha=0.4, s=20, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Bare Token F1')
    ax.set_ylabel('Primed Token F1')
    ax.set_title(f'Part B: Generation Quality (N={len(gen_results)})')
    wins = sum(1 for b, p in zip(bare_f1s, primed_f1s) if p > b)
    losses = sum(1 for b, p in zip(bare_f1s, primed_f1s) if p < b)
    ax.text(0.05, 0.95, f"Primed wins: {wins}\nBare wins: {losses}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='wheat', alpha=0.5))

# Panel 3: QL AUC across domains
ax = axes[1, 0]
domains = []
ql_aucs = []
for name, analysis in [("MARCO\n(Exp31)", {'nll_ql': {'auc': 0.578}}),
                        ("Commercial", com_analysis),
                        ("Informational", info_analysis)]:
    if analysis:
        domains.append(name)
        ql_aucs.append(analysis.get('nll_ql', {}).get('auc', 0))
if esci_analysis:
    domains.append("ESCI")
    ql_aucs.append(esci_analysis.get('nll_ql', {}).get('auc', 0))

colors = ['gray'] + ['#2ca02c' if a > 0.65 else '#ff7f0e' if a > 0.55 else '#d62728'
                      for a in ql_aucs[1:]]
ax.bar(range(len(domains)), ql_aucs, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(domains)))
ax.set_xticklabels(domains, fontsize=9)
ax.set_ylabel('AUC-ROC')
ax.set_title('Part C: QL Ranking Across Domains')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
ax.legend(fontsize=8)

# Panel 4: NLL improvement from priming (generation)
ax = axes[1, 1]
if gen_results:
    bare_nlls = [r['bare_nll'] for r in gen_results]
    primed_nlls = [r['primed_nll'] for r in gen_results]
    deltas = [b - p for b, p in zip(bare_nlls, primed_nlls)]
    ax.hist(deltas, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--')
    ax.axvline(x=np.mean(deltas), color='green', linestyle='-',
               label=f'Mean={np.mean(deltas):.3f}')
    ax.set_xlabel('NLL Delta (bare - primed, positive = primed helps)')
    ax.set_ylabel('Count')
    ax.set_title('Part B: NLL Improvement Distribution')
    ax.legend(fontsize=8)

plt.suptitle('Experiment 32: Ad-Serving Pipeline', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'exp32_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plots to {RESULTS_DIR / 'exp32_plots.png'}")

# --- Save all results ---
final_results = {
    'experiment': 'exp32_ad_serving_pipeline',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_NAME,
    'part_a_fusion': fusion_results,
    'part_b_generation': {
        'n_queries': len(gen_results),
        'analysis': gen_analysis,
        'use_soft_prefix': USE_SOFT,
        'prefix_len': PREFIX_LEN,
        'cutoff': CUTOFF,
    },
    'part_c_commercial': commercial_results,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'results.json'}")

# --- CSV for generation results ---
gen_csv = RESULTS_DIR / 'generation_scores.csv'
with open(gen_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'query', 'bare_nll', 'primed_nll', 'bare_f1', 'primed_f1',
        'bare_contains', 'primed_contains', 'bare_grounding', 'primed_grounding'])
    writer.writeheader()
    for r in gen_results:
        writer.writerow({k: r[k] for k in writer.fieldnames})
print(f"Generation CSV saved: {gen_csv}")

# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 70)
print("FINAL VERDICT -- Exp 32: Ad-Serving Pipeline")
print("=" * 70)

print(f"\nModel: Gemma 3 4B | Prefix: {'soft (Exp25)' if USE_SOFT else 'discrete'} ({PREFIX_LEN} vectors)")

# Part A
print(f"\n--- Part A: Signal Fusion ---")
if fusion_results:
    best_auc = max(fusion_results.get('best_linear_auc', 0),
                   fusion_results.get('logreg_auc', 0) or 0,
                   fusion_results.get('rank_fusion_auc', 0))
    ref_auc = fusion_results['auc_pmi_al']
    verdict_a = "YES" if best_auc > ref_auc + 0.003 else "MARGINAL" if best_auc > ref_auc else "NO"
    print(f"  Fusion beats PMI-AL?  {verdict_a} (best={best_auc:.4f} vs ref={ref_auc:.4f})")
else:
    print(f"  SKIPPED (Exp 31 data not available)")

# Part B
print(f"\n--- Part B: Generation Quality ---")
if gen_analysis:
    f1_delta = gen_analysis.get('Token F1', {}).get('delta', 0)
    contains_delta = gen_analysis.get('Contains Answer', {}).get('delta', 0)
    nll_delta = gen_analysis.get('NLL', {}).get('delta', 0)
    f1_p = gen_analysis.get('Token F1', {}).get('p_value', 1)
    verdict_b = "YES" if f1_delta > 0 and f1_p < 0.05 else "TRENDING" if f1_delta > 0 else "NO"
    print(f"  Priming improves generation?  {verdict_b}")
    print(f"  Token F1 delta: {f1_delta:+.4f} (p={f1_p:.3e})")
    print(f"  Contains-answer delta: {contains_delta:+.4f}")
    print(f"  NLL delta: {nll_delta:+.4f}")

# Part C
print(f"\n--- Part C: Commercial Domain ---")
if com_analysis:
    com_ql = com_analysis.get('nll_ql', {}).get('auc', 0)
    info_ql = info_analysis.get('nll_ql', {}).get('auc', 0)
    print(f"  Commercial QL AUC:     {com_ql:.4f}")
    print(f"  Informational QL AUC:  {info_ql:.4f}")
    print(f"  Exp 31 full QL AUC:    0.578")
    if esci_analysis:
        esci_ql = esci_analysis.get('nll_ql', {}).get('auc', 0)
        print(f"  ESCI QL AUC:           {esci_ql:.4f}")
        print(f"  QL better on diverse pool? {'YES' if esci_ql > 0.65 else 'NO'} (ESCI={esci_ql:.3f} vs MARCO=0.578)")

print(f"\nDone!")""")

# ============================================================
# Cell 13: Cleanup
# ============================================================
code("""# Cell 13: Cleanup
print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"\\nGPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Cleanup complete.")""")

# ============================================================
# Build notebook JSON
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

outpath = "32_ad_serving_pipeline.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {outpath}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
