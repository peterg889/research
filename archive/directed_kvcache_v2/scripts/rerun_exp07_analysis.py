#!/usr/bin/env python3
"""Re-run Exp 07 analysis and plotting from checkpoint data.

This script loads the checkpoint and surrogate data, then runs the
analysis (Cells 8-12) and saves plots + results.json without needing
the model or re-running the evaluation loop.
"""
import os
os.umask(0o000)

import json
import time
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# Constants (matching build_nb07.py)
# ============================================================
RESULTS_DIR = Path('/home/jupyter/research/directed_kvcache_v2/results/exp07')
SURROGATES_DIR = RESULTS_DIR / 'surrogates'
FINAL_RESULTS_PATH = RESULTS_DIR / 'results.json'
SEED = 42
N = 2000

STATIC_PHRASES = {
    'definitional': "What is this and what does it mean?",
    'procedural': "How do I do this step by step?",
    'quantitative': "How much does this cost or how long does it take?",
    'factual': "What are the key facts I need to know?",
    'problem': "What problem does this solve?",
}

SUFFIX_SEPARATOR = "\n\nRelated question: "
N_COMPARISONS = 10
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS

CONDITION_NAMES = [
    'bare', 'random_trunc', 'separator_only',
    'static_def_trunc', 'static_proc_trunc', 'static_quant_trunc',
    'static_fact_trunc', 'static_prob_trunc',
    'static_def_suffix', 'static_proc_suffix', 'static_quant_suffix',
    'static_fact_suffix', 'static_prob_suffix',
    'llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix', 'llm_messy_suffix',
    'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx',
    'novel_generic_trunc',
]

STATIC_TRUNC_CONDS = ['static_def_trunc', 'static_proc_trunc', 'static_quant_trunc',
                       'static_fact_trunc', 'static_prob_trunc']
STATIC_SUFFIX_CONDS = ['static_def_suffix', 'static_proc_suffix', 'static_quant_suffix',
                        'static_fact_suffix', 'static_prob_suffix']
LLM_SUFFIX_CONDS = ['llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix',
                      'llm_messy_suffix']

def cohens_d(diff):
    return np.mean(diff) / np.std(diff, ddof=1)

def classify_intent(query):
    q = query.lower()
    if any(w in q for w in ['what is', 'define', 'meaning', 'explain']):
        return 'definitional'
    if any(w in q for w in ['how to', 'how do', 'steps', 'tutorial', 'guide']):
        return 'procedural'
    if any(w in q for w in ['buy', 'cost', 'price', 'cheap', 'deal', 'order', 'purchase']):
        return 'transactional'
    if any(w in q for w in ['best', 'top', 'vs', 'compare', 'review', 'difference']):
        return 'comparison'
    if any(w in q for w in ['when did', 'where is', 'who', 'how many', 'how much', 'how long']):
        return 'factual'
    if any(w in q for w in ['symptoms', 'treatment', 'diagnosis', 'causes', 'cure']):
        return 'medical'
    return 'other'

# ============================================================
# Load data
# ============================================================
print("Loading checkpoint...")
ckpt = json.load(open(RESULTS_DIR / 'checkpoint.json'))
results = ckpt['results']
print(f"Loaded {len(results)} results")

print("Loading surrogates...")
surrogates_5 = json.load(open(SURROGATES_DIR / 'all_5_surrogates.json'))['surrogates']
print(f"Loaded {len(surrogates_5)} surrogate sets")

# Get queries from checkpoint (for embedding routing + intent classification)
sample_queries = ckpt['sample_queries']
print(f"Loaded {len(sample_queries)} sample queries from checkpoint")

# Get intents
intents = [classify_intent(q) for q in sample_queries]

# ============================================================
# Cell 8: Comparisons & Summary (from build_nb07.py)
# ============================================================
print("=" * 70)
print("CELL 8: SUMMARY & KEY COMPARISONS")
print("=" * 70)

# Extract arrays and filter zero NLLs
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

valid = np.ones(len(results), dtype=bool)
for cname in CONDITION_NAMES:
    valid &= (cond_arrays[cname] != 0)
n_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total: {len(results)}, Valid: {n_valid}, Excluded: {n_excluded}")

c = {}
for cname in CONDITION_NAMES:
    c[cname] = cond_arrays[cname][valid]

# Metadata arrays (filtered)
intents_valid = np.array(intents)[valid]
answer_tokens_valid = np.array([r['answer_token_count'] for r in results])[valid]
passage_words_valid = np.array([r['passage_word_count'] for r in results])[valid]

# NLL summary table
print(f"\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10}")
print("-" * 60)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        d_str = "—"
    else:
        d = cohens_d(c['bare'] - c[cname])
        d_str = f"{d:+.3f}"
    print(f"{cname:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d_str:>10}")

# Identify best statics
best_static_trunc_d = -999
best_static_trunc_name = ''
for cname in STATIC_TRUNC_CONDS:
    d = cohens_d(c['bare'] - c[cname])
    if d > best_static_trunc_d:
        best_static_trunc_d = d
        best_static_trunc_name = cname

best_static_suffix_d = -999
best_static_suffix_name = ''
for cname in STATIC_SUFFIX_CONDS:
    d = cohens_d(c['bare'] - c[cname])
    if d > best_static_suffix_d:
        best_static_suffix_d = d
        best_static_suffix_name = cname

best_static_overall_d = max(best_static_trunc_d, best_static_suffix_d)
best_static_overall_name = best_static_trunc_name if best_static_trunc_d > best_static_suffix_d else best_static_suffix_name

print(f"\nBest static trunc: {best_static_trunc_name} (d={best_static_trunc_d:+.3f})")
print(f"Best static suffix: {best_static_suffix_name} (d={best_static_suffix_d:+.3f})")
print(f"Best static overall: {best_static_overall_name} (d={best_static_overall_d:+.3f})")

# Oracle-routed
oracle_routed_static_trunc = np.minimum.reduce([c[cn] for cn in STATIC_TRUNC_CONDS])
oracle_routed_static_suffix = np.minimum.reduce([c[cn] for cn in STATIC_SUFFIX_CONDS])
oracle_routed_static_k5 = np.minimum(oracle_routed_static_trunc, oracle_routed_static_suffix)
oracle_routed_llm_k4 = np.minimum.reduce([c[cn] for cn in LLM_SUFFIX_CONDS])

print(f"\nOracle-routed-static-K5 d vs bare: {cohens_d(c['bare'] - oracle_routed_static_k5):+.3f}")
print(f"Oracle-routed-LLM-K4 d vs bare: {cohens_d(c['bare'] - oracle_routed_llm_k4):+.3f}")

# 10 primary comparisons
print(f"\n{'='*85}")
print(f"10 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*85}")

comparisons = [
    ('C1: Best-static-trunc vs Bare',
     c['bare'] - c[best_static_trunc_name],
     'Do static prefixes help at all?'),
    ('C2: Best-static-suffix vs Bare',
     c['bare'] - c[best_static_suffix_name],
     'Do static suffixes help at all?'),
    ('C3: Best-static-trunc vs Best-static-suffix',
     c[best_static_suffix_name] - c[best_static_trunc_name],
     'Which mode is better?'),
    ('C4: Best-static vs LLM-kw-trunc',
     c[best_static_overall_name] - c['llm_keyword_trunc'],
     'Can statics match LLM?'),
    ('C5: Oracle-static-K5 vs Best-static',
     c[best_static_overall_name] - oracle_routed_static_k5,
     'Does routing help?'),
    ('C6: Oracle-static-K5 vs LLM-kw-suf',
     oracle_routed_static_k5 - c['llm_keyword_suffix'],
     'Routed statics vs single LLM?'),
    ('C7: LLM-kw-suffix vs LLM-kw-trunc',
     c['llm_keyword_trunc'] - c['llm_keyword_suffix'],
     'Suffix vs truncation for LLM?'),
    ('C8: LLM-kw-full vs LLM-kw-trunc',
     c['llm_keyword_trunc'] - c['llm_keyword_full_ctx'],
     'Full-ctx vs truncated?'),
    ('C9: LLM-kw+sep vs LLM-kw-suffix',
     c['llm_keyword_suffix'] - c['llm_keyword_sep'],
     'Does stacking replicate?'),
    ('C10: Embed-routed-LLM vs Oracle-LLM',
     None,
     'Is practical routing viable?'),
]

print(f"\n{'Comparison':<40} {'Mean D':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 95)

comparison_results = {}
for name, delta, question in comparisons:
    if delta is None:
        print(f"{name:<40} {'(computed later)':>50}")
        continue
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<40} {np.mean(delta):>8.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        'question': question,
    }

# All vs Bare
print(f"\n{'='*85}")
print("ALL CONDITIONS vs BARE")
print(f"{'='*85}")
print(f"\n{'Condition':<25} {'d vs Bare':>10} {'Win%':>7} {'p':>12}")
print("-" * 60)
all_vs_bare = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname:<25} {d:>10.3f} {win:>6.1f}% {p_val:>11.2e} {sig:>5}")
    all_vs_bare[cname] = {'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val)}

# ============================================================
# Cell 9: Hardness quintile breakdown
# ============================================================
print("\n" + "=" * 70)
print("CELL 9: HARDNESS QUINTILE BREAKDOWN")
print("=" * 70)

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

conditions_to_show = [
    'random_trunc', 'separator_only',
    'static_def_trunc', 'static_proc_trunc', 'static_fact_trunc',
    'static_def_suffix', 'static_proc_suffix', 'static_fact_suffix',
    'llm_keyword_suffix', 'llm_symptom_suffix', 'llm_question_suffix',
    'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx',
    'novel_generic_trunc',
]

header = f"{'Condition':<25}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\n{header}")
print("-" * (25 + 14 * 6))

hardness_breakdown = {}
for cname in conditions_to_show:
    row = f"{cname:<25}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            d = cohens_d(delta)
            row += f"{d:>+14.3f}"
            quintile_ds.append(float(d))
    d_all = cohens_d(bare_valid - c[cname])
    row += f"{d_all:>+14.3f}"
    print(row)
    hardness_breakdown[cname] = {
        'quintile_ds': quintile_ds,
        'overall_d': float(d_all),
    }

# Hardness interaction correlations
print(f"\nHardness interaction (r between bare NLL and benefit):")
for cname in conditions_to_show:
    delta = bare_valid - c[cname]
    r, p = stats.pearsonr(bare_valid, delta)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {cname:<25}: r={r:+.3f}, p={p:.2e} {sig}")

# ============================================================
# Cell 10: Stratification
# ============================================================
print("\n" + "=" * 70)
print("CELL 10: STRATIFICATION ANALYSIS")
print("=" * 70)

# Intent stratification
print("\n--- Intent Stratification ---")
intent_categories = sorted(set(intents_valid))
key_conditions = ['bare', 'separator_only', best_static_trunc_name, best_static_suffix_name,
                   'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']

header = f"{'Intent':<15} {'N':>5}"
for cn in key_conditions:
    if cn == 'bare':
        header += f"{'bare NLL':>12}"
    else:
        header += f"{cn[:10]:>12}"
print(header)
print("-" * (20 + 12 * len(key_conditions)))

intent_analysis = {}
for intent in intent_categories:
    mask = intents_valid == intent
    n_intent = int(np.sum(mask))
    if n_intent < 10:
        continue
    row = f"{intent:<15} {n_intent:>5}"
    intent_ds = {}
    for cn in key_conditions:
        if cn == 'bare':
            row += f"{np.mean(c['bare'][mask]):>12.3f}"
        else:
            d = cohens_d(c['bare'][mask] - c[cn][mask])
            row += f"{d:>+12.3f}"
            intent_ds[cn] = float(d)
    print(row)
    intent_analysis[intent] = {'n': n_intent, 'ds': intent_ds}

# Best surrogate per intent
print(f"\nBest surrogate per intent:")
all_statics_for_intent = STATIC_TRUNC_CONDS + STATIC_SUFFIX_CONDS
for intent in intent_categories:
    mask = intents_valid == intent
    if int(np.sum(mask)) < 10:
        continue
    best_d = -999
    best_cn = ''
    for cn in all_statics_for_intent:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        if d > best_d:
            best_d = d
            best_cn = cn
    print(f"  {intent:<15}: {best_cn} (d={best_d:+.3f})")

# Answer length stratification
print("\n--- Answer Length Stratification ---")
ans_short = answer_tokens_valid < 5
ans_medium = (answer_tokens_valid >= 5) & (answer_tokens_valid <= 15)
ans_long = answer_tokens_valid > 15

print(f"Answer length bins: short(<5)={int(np.sum(ans_short))}, "
      f"medium(5-15)={int(np.sum(ans_medium))}, long(>15)={int(np.sum(ans_long))}")

for label, mask in [('short', ans_short), ('medium', ans_medium), ('long', ans_long)]:
    if int(np.sum(mask)) < 10:
        continue
    row = f"  {label:<10} N={int(np.sum(mask)):>4}"
    for cn in ['separator_only', 'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        row += f"  {cn[:12]}={d:+.3f}"
    print(row)

# Passage length stratification
print("\n--- Passage Length Stratification ---")
psg_short = passage_words_valid < 80
psg_medium = (passage_words_valid >= 80) & (passage_words_valid <= 200)
psg_long = passage_words_valid > 200

print(f"Passage length bins: short(<80)={int(np.sum(psg_short))}, "
      f"medium(80-200)={int(np.sum(psg_medium))}, long(>200)={int(np.sum(psg_long))}")

for label, mask in [('short', psg_short), ('medium', psg_medium), ('long', psg_long)]:
    if int(np.sum(mask)) < 10:
        continue
    row = f"  {label:<10} N={int(np.sum(mask)):>4}"
    for cn in ['separator_only', 'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']:
        d = cohens_d(c['bare'][mask] - c[cn][mask])
        row += f"  {cn[:12]}={d:+.3f}"
    print(row)

# ============================================================
# Cell 11: Routing analysis
# ============================================================
print("\n" + "=" * 70)
print("CELL 11: ROUTING ANALYSIS")
print("=" * 70)

# Oracle K-curve
print("\n--- Oracle K-Curve ---")
all_static_conds = STATIC_TRUNC_CONDS + STATIC_SUFFIX_CONDS
static_ds = [(cn, cohens_d(c['bare'] - c[cn])) for cn in all_static_conds]
static_ds.sort(key=lambda x: -x[1])

print(f"\nStatic surrogates ranked by d vs bare:")
for cn, d in static_ds:
    print(f"  {cn}: d={d:+.3f}")

print(f"\nOracle best-of-K (static):")
for K in range(1, 6):
    top_k_conds = [cn for cn, _ in static_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k_conds])
    d = cohens_d(c['bare'] - best_of_k)
    print(f"  K={K}: d={d:+.3f} (conditions: {', '.join(top_k_conds)})")

llm_ds = [(cn, cohens_d(c['bare'] - c[cn])) for cn in LLM_SUFFIX_CONDS]
llm_ds.sort(key=lambda x: -x[1])
print(f"\nOracle best-of-K (LLM suffix):")
for K in range(1, 5):
    top_k_conds = [cn for cn, _ in llm_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k_conds])
    d = cohens_d(c['bare'] - best_of_k)
    print(f"  K={K}: d={d:+.3f}")

# Embedding routing
print("\n--- Embedding Routing ---")
print("Loading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

valid_indices = np.where(valid)[0]
valid_queries = [sample_queries[i] for i in valid_indices]
print(f"Embedding {len(valid_queries)} queries...")
query_embeddings = embed_model.encode(valid_queries, show_progress_bar=True)

llm_surrogate_keys = ['keyword', 'symptom', 'question', 'messy']
llm_surrogate_cond_map = {
    'keyword': 'llm_keyword_suffix',
    'symptom': 'llm_symptom_suffix',
    'question': 'llm_question_suffix',
    'messy': 'llm_messy_suffix',
}

print("Embedding LLM surrogates...")
embed_routed_nlls = np.zeros(n_valid)
embed_routed_choices = []
oracle_routed_choices = []

for vi in tqdm(range(n_valid), desc="Embedding routing"):
    orig_idx = valid_indices[vi]
    q_emb = query_embeddings[vi:vi+1]

    surr_texts = {
        'keyword': surrogates_5[orig_idx].get('keyword_query', ''),
        'symptom': surrogates_5[orig_idx].get('symptom_scenario', ''),
        'question': surrogates_5[orig_idx].get('target_question', ''),
        'messy': surrogates_5[orig_idx].get('messy_realworld', ''),
    }

    surr_embs = embed_model.encode([surr_texts[k] for k in llm_surrogate_keys])
    sims = cos_sim(q_emb, surr_embs)[0]

    best_k_idx = np.argmax(sims)
    best_k = llm_surrogate_keys[best_k_idx]
    embed_routed_nlls[vi] = c[llm_surrogate_cond_map[best_k]][vi]
    embed_routed_choices.append(best_k)

    nlls_4 = {k: c[llm_surrogate_cond_map[k]][vi] for k in llm_surrogate_keys}
    oracle_k = min(nlls_4, key=nlls_4.get)
    oracle_routed_choices.append(oracle_k)

d_embed_routed = cohens_d(c['bare'] - embed_routed_nlls)
d_oracle_routed = cohens_d(c['bare'] - oracle_routed_llm_k4)
oracle_accuracy = np.mean([e == o for e, o in zip(embed_routed_choices, oracle_routed_choices)])

print(f"\nEmbedding-routed-LLM-K4: d={d_embed_routed:+.3f}")
print(f"Oracle-routed-LLM-K4: d={d_oracle_routed:+.3f}")
print(f"Embedding routing accuracy vs oracle: {oracle_accuracy*100:.1f}%")

# C10 comparison
delta_c10 = embed_routed_nlls - oracle_routed_llm_k4
d_c10 = cohens_d(delta_c10)
t_c10, p_c10 = stats.ttest_1samp(delta_c10, 0)
sig_c10 = "***" if p_c10 < 0.001 else "**" if p_c10 < BONFERRONI_ALPHA else "*" if p_c10 < 0.05 else "ns"
print(f"\nC10: Embed-routed vs Oracle-routed: d={d_c10:+.3f}, p={p_c10:.2e} {sig_c10}")
comparison_results['C10: Embed-routed-LLM vs Oracle-LLM'] = {
    'mean_delta': float(np.mean(delta_c10)),
    'cohens_d': float(d_c10),
    'win_rate': float(np.mean(delta_c10 > 0)),
    't_stat': float(t_c10),
    'p_value': float(p_c10),
    'bonferroni_significant': bool(p_c10 < BONFERRONI_ALPHA),
    'question': 'Is practical routing viable?',
}

# Routing choice distribution
print(f"\nEmbedding routing choice distribution:")
for k in llm_surrogate_keys:
    n_chosen = sum(1 for ch in embed_routed_choices if ch == k)
    print(f"  {k}: {n_chosen} ({n_chosen/n_valid*100:.1f}%)")

print(f"\nOracle routing choice distribution:")
for k in llm_surrogate_keys:
    n_chosen = sum(1 for ch in oracle_routed_choices if ch == k)
    print(f"  {k}: {n_chosen} ({n_chosen/n_valid*100:.1f}%)")

# Intent-matched static routing
print("\n--- Intent-Matched Static Routing ---")
intent_to_static = {
    'definitional': 'static_def_suffix',
    'procedural': 'static_proc_suffix',
    'transactional': 'static_quant_suffix',
    'comparison': 'static_fact_suffix',
    'factual': 'static_fact_suffix',
    'medical': 'static_prob_suffix',
    'other': best_static_suffix_name,
}

intent_matched_nlls = np.zeros(n_valid)
for vi in range(n_valid):
    intent = intents_valid[vi]
    matched_cond = intent_to_static.get(intent, best_static_suffix_name)
    intent_matched_nlls[vi] = c[matched_cond][vi]

d_intent_matched = cohens_d(c['bare'] - intent_matched_nlls)
print(f"Intent-matched-static d vs bare: {d_intent_matched:+.3f}")
print(f"Best-single-static d vs bare: {best_static_overall_d:+.3f}")
print(f"Improvement from intent matching: {d_intent_matched - best_static_overall_d:+.3f}")

# Complementarity matrix
print("\n--- Complementarity Matrix (LLM suffix, fraction where row wins over column) ---")
llm_conds_for_comp = LLM_SUFFIX_CONDS
header = f"{'':>20}" + "".join(f"{cn[:12]:>14}" for cn in llm_conds_for_comp)
print(header)
for cn_a in llm_conds_for_comp:
    row = f"{cn_a[:20]:<20}"
    for cn_b in llm_conds_for_comp:
        if cn_a == cn_b:
            row += f"{'---':>14}"
        else:
            frac = np.mean(c[cn_a] < c[cn_b])
            row += f"{frac:>14.3f}"
    print(row)

# Hardness-gated routing
print("\n--- Hardness-Gated Routing ---")
print("Prime only if bare_NLL > threshold. Sweep threshold.")
thresholds = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
for thresh in thresholds:
    mask_prime = bare_valid > thresh
    n_primed = int(np.sum(mask_prime))
    if n_primed < 10:
        continue
    gated_nlls = np.where(mask_prime, c['llm_keyword_sep'], c['bare'])
    d_gated = cohens_d(c['bare'] - gated_nlls)
    frac_primed = n_primed / n_valid * 100
    print(f"  threshold={thresh:.1f}: prime {frac_primed:.0f}% samples, d={d_gated:+.3f}")

del embed_model

# ============================================================
# Cell 12: Plots (with bug fix)
# ============================================================
print("\n" + "=" * 70)
print("CELL 12: GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# --- Plot 1: All conditions bar chart (d vs bare) ---
ax = axes[0, 0]
cnames_sorted = sorted(
    [cn for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda cn: cohens_d(c['bare'] - c[cn]),
    reverse=True
)
ds_bar = [cohens_d(c['bare'] - c[cn]) for cn in cnames_sorted]
color_map = {}
for cn in STATIC_TRUNC_CONDS:
    color_map[cn] = 'steelblue'
for cn in STATIC_SUFFIX_CONDS:
    color_map[cn] = 'cornflowerblue'
for cn in LLM_SUFFIX_CONDS:
    color_map[cn] = 'forestgreen'
color_map['llm_keyword_trunc'] = 'darkgreen'
color_map['llm_keyword_sep'] = 'gold'
color_map['llm_keyword_full_ctx'] = 'orange'
color_map['novel_generic_trunc'] = 'mediumpurple'
color_map['random_trunc'] = 'gray'
color_map['separator_only'] = 'salmon'
colors_bar = [color_map.get(cn, 'lightgray') for cn in cnames_sorted]

bars = ax.barh(range(len(cnames_sorted)), ds_bar, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(cnames_sorted)))
ax.set_yticklabels(cnames_sorted, fontsize=7)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare')
ax.invert_yaxis()

# --- Plot 2: Static trunc vs suffix comparison ---
ax = axes[0, 1]
static_names = list(STATIC_PHRASES.keys())
abbrev_map = {'definitional': 'def', 'procedural': 'proc', 'quantitative': 'quant', 'factual': 'fact', 'problem': 'prob'}
trunc_ds = [cohens_d(c['bare'] - c[f'static_{abbrev_map[n]}_trunc']) for n in static_names]
suffix_ds = [cohens_d(c['bare'] - c[f'static_{abbrev_map[n]}_suffix']) for n in static_names]
x = np.arange(len(static_names))
width = 0.35
ax.bar(x - width/2, trunc_ds, width, label='Truncated prefix', color='steelblue', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, suffix_ds, width, label='Suffix', color='cornflowerblue', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([n[:6] for n in static_names], fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Static: Prefix vs Suffix Mode')
ax.legend(fontsize=8)

# --- Plot 3: Hardness x condition heatmap ---
ax = axes[0, 2]
hm_conditions = ['separator_only', best_static_trunc_name, best_static_suffix_name,
                  'llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep', 'llm_keyword_full_ctx']
hm_data = []
for cname in hm_conditions:
    row = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row.append(0)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            row.append(cohens_d(delta))
    hm_data.append(row)
hm_data = np.array(hm_data)
im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.5, vmax=0.7, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=7)
ax.set_yticks(range(len(hm_conditions)))
ax.set_yticklabels(hm_conditions, fontsize=7)
for i in range(len(hm_conditions)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness x Condition')

# --- Plot 4: Oracle K-curve ---
ax = axes[1, 0]
static_k_ds = []
for K in range(1, 6):
    top_k = [cn for cn, _ in static_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k])
    static_k_ds.append(cohens_d(c['bare'] - best_of_k))
llm_k_ds = []
for K in range(1, 5):
    top_k = [cn for cn, _ in llm_ds[:K]]
    best_of_k = np.minimum.reduce([c[cn] for cn in top_k])
    llm_k_ds.append(cohens_d(c['bare'] - best_of_k))

ax.plot(range(1, 6), static_k_ds, 'o-', color='steelblue', label='Static (10 total)')
ax.plot(range(1, 5), llm_k_ds, 's-', color='forestgreen', label='LLM suffix (4 total)')
ax.set_xlabel('K (best-of-K)')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Oracle K-Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 5: Intent stratification ---
ax = axes[1, 1]
intent_cats = sorted(intent_analysis.keys(), key=lambda x: -intent_analysis[x]['n'])
intent_cats = [ic for ic in intent_cats if intent_analysis[ic]['n'] >= 10]
x_int = np.arange(len(intent_cats))
width_int = 0.25
for j, cn in enumerate(['llm_keyword_suffix', 'llm_keyword_trunc', 'llm_keyword_sep']):
    ds_int = [intent_analysis[ic]['ds'].get(cn, 0) for ic in intent_cats]
    ax.bar(x_int + j * width_int - width_int, ds_int, width_int, label=cn[:15],
           edgecolor='black', linewidth=0.5)
ax.set_xticks(x_int)
ax.set_xticklabels(intent_cats, fontsize=7, rotation=45)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Intent Stratification')
ax.legend(fontsize=6)

# --- Plot 6: Hardness-gated routing curve ---
ax = axes[1, 2]
thresh_vals = np.arange(0, 4.1, 0.2)
gated_ds = []
frac_primed_vals = []
for thresh in thresh_vals:
    mask_prime = bare_valid > thresh
    n_primed = int(np.sum(mask_prime))
    if n_primed < 5:
        gated_ds.append(np.nan)
        frac_primed_vals.append(0)
        continue
    gated_nlls = np.where(mask_prime, c['llm_keyword_sep'], c['bare'])
    gated_ds.append(cohens_d(c['bare'] - gated_nlls))
    frac_primed_vals.append(n_primed / n_valid * 100)

ax2 = ax.twinx()
ax.plot(thresh_vals, gated_ds, 'o-', color='forestgreen', markersize=3, label='d vs bare')
ax2.plot(thresh_vals, frac_primed_vals, 's--', color='gray', markersize=3, alpha=0.5, label='% primed')
ax.set_xlabel('Bare NLL Threshold')
ax.set_ylabel("Cohen's d vs bare (gated)", color='forestgreen')
ax2.set_ylabel('% Samples Primed', color='gray')
ax.set_title('Hardness-Gated Routing')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=7)
ax2.legend(loc='upper right', fontsize=7)

plt.suptitle('Exp 07: Static Surrogates, Dual-Mode Priming, Intent Routing', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")

# ============================================================
# Cell 13: Save comprehensive results JSON
# ============================================================
print("\n" + "=" * 70)
print("CELL 13: SAVING RESULTS JSON")
print("=" * 70)

final = {
    'experiment': 'exp07_static_surrogates_and_routing',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'seed': SEED,
        'n_eval': N,
        'n_valid': n_valid,
        'n_excluded': n_excluded,
        'n_conditions': len(CONDITION_NAMES),
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': {
        cname: {
            'mean': float(np.mean(c[cname])),
            'std': float(np.std(c[cname])),
            'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cname])) if cname != 'bare' else 0.0,
        }
        for cname in CONDITION_NAMES
    },
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'hardness_breakdown': hardness_breakdown,
    'intent_analysis': intent_analysis,
    'routing_analysis': {
        'oracle_static_k5_d': float(cohens_d(c['bare'] - oracle_routed_static_k5)),
        'oracle_llm_k4_d': float(d_oracle_routed),
        'embed_routed_llm_k4_d': float(d_embed_routed),
        'embed_routing_accuracy': float(oracle_accuracy),
        'intent_matched_static_d': float(d_intent_matched),
        'best_static_trunc': best_static_trunc_name,
        'best_static_suffix': best_static_suffix_name,
        'best_static_overall': best_static_overall_name,
    },
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\nDone!")
