#!/usr/bin/env python3
"""Generate Exp 11 notebook: Delta-as-Feature and Prefix Diversity for Ranking.

Two-phase experiment building on Exp 04A results:

Phase A (no GPU): Reuse Exp 04A per-passage NLL scores to test whether the NLL
change from bare to primed (delta) works as a ranking feature. Tests delta alone,
linear combination with bare NLL, and ensemble of multiple conditions.

Phase B (GPU): Score each passage with K=10 different random prefixes (with
truncation). Test whether NLL variance across prefixes discriminates relevant
from irrelevant passages.

Key insight from Exp 04A: Structural surrogates (static_fact d=+0.153, random
d=+0.117) create POSITIVE differential -- they help relevant passages MORE than
irrelevant ones. But oracle shows ZERO differential (d=-0.007).
"""
import json

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")
                  if isinstance(source, str) else source})


def code(source):
    lines = source.split("\n") if isinstance(source, str) else source
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": formatted,
                  "outputs": [], "execution_count": None})


# ============================================================
# Cell 1: Markdown title
# ============================================================
md(r"""# Experiment 11: Delta-as-Feature and Prefix Diversity for Ranking
## Can we extract better ranking signals from the structural mechanism?

### Context
Exp 04A found a paradox: **structural surrogates improve ranking but oracle doesn't**.
- bare AUC = 0.845
- oracle\_trunc AUC = 0.853 (ns, d=-0.007 differential)
- surr\_doc\_trunc AUC = 0.867 (\*\*, d=+0.053 differential)
- static\_fact\_trunc AUC = 0.860 (\*\*, d=+0.153 differential)
- random\_trunc AUC = 0.866 (\*\*, d=+0.117 differential)

The structural surrogates create **differential signal**: they help relevant passages
MORE than irrelevant ones. This experiment explores two ways to exploit this:

### Phase A: Delta-as-Feature (from existing 04A data, no GPU needed)
For each passage, define: `delta = NLL_bare - NLL_primed` (positive = priming helped).
Since delta is larger for relevant passages (positive differential from 04A):
1. Rank by delta alone -- does higher delta = more relevant?
2. Linear combination: `score = (1-lambda)*NLL_bare + lambda*NLL_primed` -- optimal weighting
3. Ensemble: average NLL across multiple primed conditions

### Phase B: Prefix Diversity (new GPU scoring, ~4.4 hours)
Score each passage with K=10 DIFFERENT random prefixes (all with truncation).
Compute per-passage NLL mean and std across prefixes.

**Hypothesis**: Relevant passages respond more consistently to structural perturbation
(lower NLL variance), while irrelevant passages show more erratic responses.

### N=400 queries (same as Exp 04A), K=10 random prefixes, Bonferroni=8
""")

# ============================================================
# Cell 2: Setup
# ============================================================
code("""# Cell 2: Setup
import os
os.umask(0o000)

import sys
import json
import time
import re
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import wilcoxon, pearsonr, spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, "../..")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../../results/exp11")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 400
MODEL_NAME = "google/t5gemma-2-4b-4b"
K_PREFIXES = 10
N_BONFERRONI = 8

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 11: Delta-as-Feature and Prefix Diversity for Ranking")
print(f"N queries: {N_SAMPLES}")
print(f"K random prefixes: {K_PREFIXES}")
print(f"Bonferroni comparisons: {N_BONFERRONI}")""")

# ============================================================
# Cell 3: Load 04A checkpoint + compute deltas
# ============================================================
code(r"""# Cell 3: Load Exp 04A checkpoint and compute delta features
from lib.analysis import cohens_d

print("=" * 70)
print("LOADING EXP 04A DATA")
print("=" * 70)

EXP04A_CKPT = Path("../../results/exp04a/checkpoint.json")
assert EXP04A_CKPT.exists(), f"Exp 04A checkpoint not found at {EXP04A_CKPT}"

with open(EXP04A_CKPT) as f:
    exp04a = json.load(f)

results_04a = exp04a['results']
print(f"Loaded {len(results_04a)} queries from Exp 04A")

COND_NAMES_04A = ['bare', 'oracle_trunc', 'surr_template_trunc',
                  'surr_doc_trunc', 'random_trunc', 'static_fact_trunc']
PRIMED_CONDITIONS = COND_NAMES_04A[1:]

# Ranking metrics (reused throughout)
def compute_auc(nlls, relevant_idx):
    rel_nll = nlls[relevant_idx]
    irrel_nlls = [nlls[i] for i in range(len(nlls)) if i != relevant_idx]
    if len(irrel_nlls) == 0:
        return 0.5
    wins = sum(1 for nll in irrel_nlls if nll > rel_nll)
    ties = sum(1 for nll in irrel_nlls if nll == rel_nll)
    return (wins + 0.5 * ties) / len(irrel_nlls)

def compute_mrr_at_k(nlls, relevant_idx, k=3):
    ranked = list(np.argsort(nlls))
    for rank, idx in enumerate(ranked[:k], 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0

def compute_hit_at_k(nlls, relevant_idx, k=1):
    ranked = set(np.argsort(nlls)[:k].tolist())
    return 1.0 if relevant_idx in ranked else 0.0

# Compute delta features: delta = NLL_bare - NLL_primed
print(f"\nComputing delta features for {len(PRIMED_CONDITIONS)} conditions...")
for r in results_04a:
    bare_nlls = np.array(r['scores']['bare'])
    r['deltas'] = {}
    for cond in PRIMED_CONDITIONS:
        primed_nlls = np.array(r['scores'][cond])
        r['deltas'][cond] = (bare_nlls - primed_nlls).tolist()

# Bare AUC reference
bare_aucs_ref = np.array([
    compute_auc(np.array(r['scores']['bare']), r['relevant_idx'])
    for r in results_04a
])
print(f"Bare AUC reference: {bare_aucs_ref.mean():.3f}")

# Correlation: delta vs bare NLL
print(f"\n--- Delta vs bare NLL correlation ---")
for cond in PRIMED_CONDITIONS:
    all_bare, all_delta = [], []
    for r in results_04a:
        all_bare.extend(r['scores']['bare'])
        all_delta.extend(r['deltas'][cond])
    r_s, p_s = spearmanr(all_bare, all_delta)
    print(f"  {cond:<22s}: Spearman rho={r_s:+.3f} (p={p_s:.2e})")

# Delta by relevance
print(f"\n--- Mean delta by relevance ---")
print(f"  {'Condition':<22s} {'delta_rel':>10} {'delta_irrel':>12} {'diff':>8}")
for cond in PRIMED_CONDITIONS:
    delta_rels, delta_irrels = [], []
    for r in results_04a:
        rel_idx = r['relevant_idx']
        deltas = r['deltas'][cond]
        delta_rels.append(deltas[rel_idx])
        for i, d in enumerate(deltas):
            if i != rel_idx:
                delta_irrels.append(d)
    print(f"  {cond:<22s} {np.mean(delta_rels):>+10.4f} {np.mean(delta_irrels):>+12.4f} "
          f"{np.mean(delta_rels) - np.mean(delta_irrels):>+8.4f}")""")

# ============================================================
# Cell 4: Phase A — Delta ranking + lambda sweep + ensemble
# ============================================================
code(r"""# Cell 4: Phase A -- Delta-only ranking, lambda sweep, ensemble
print("=" * 70)
print("PHASE A: DELTA-AS-FEATURE RANKING")
print("=" * 70)

# === Test 1: Rank by delta alone ===
print(f"\n--- Test 1: Rank by delta alone (higher delta = more relevant) ---")
print(f"  {'Condition':<22s} {'AUC':>7} {'MRR@3':>7} {'Hit@1':>7} {'Hit@3':>7}")
print(f"  {'-'*55}")

delta_only_aucs = {}
for cond in PRIMED_CONDITIONS:
    aucs, mrr3s, hit1s, hit3s = [], [], [], []
    for r in results_04a:
        rel_idx = r['relevant_idx']
        neg_deltas = [-d for d in r['deltas'][cond]]
        aucs.append(compute_auc(neg_deltas, rel_idx))
        mrr3s.append(compute_mrr_at_k(neg_deltas, rel_idx, k=3))
        hit1s.append(compute_hit_at_k(neg_deltas, rel_idx, k=1))
        hit3s.append(compute_hit_at_k(neg_deltas, rel_idx, k=3))
    delta_only_aucs[cond] = np.array(aucs)
    print(f"  {cond:<22s} {np.mean(aucs):>7.3f} {np.mean(mrr3s):>7.3f} "
          f"{np.mean(hit1s):>7.3f} {np.mean(hit3s):>7.3f}")

print(f"  {'bare NLL (ref)':<22s} {bare_aucs_ref.mean():>7.3f}")

# Test delta-only vs chance
print(f"\n  Delta-only vs chance (AUC=0.5):")
for cond in PRIMED_CONDITIONS:
    d = cohens_d(delta_only_aucs[cond] - 0.5)
    nonzero = delta_only_aucs[cond] - 0.5
    nonzero = nonzero[nonzero != 0]
    _, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
    sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
           else '*' if p < 0.05/N_BONFERRONI else 'ns')
    print(f"    {cond:<22s}: AUC={delta_only_aucs[cond].mean():.3f}, d={d:+.3f}, p={p:.2e} {sig}")

# === Test 2: Lambda sweep ===
print(f"\n--- Test 2: Lambda sweep: score = (1-lam)*NLL_bare + lam*NLL_primed ---")
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]

lambda_results = {}
for cond in PRIMED_CONDITIONS:
    best_auc = 0
    best_lambda = 0
    lambda_aucs = {}

    for lam in LAMBDAS:
        aucs = []
        for r in results_04a:
            bare_nlls = np.array(r['scores']['bare'])
            primed_nlls = np.array(r['scores'][cond])
            combined = (1 - lam) * bare_nlls + lam * primed_nlls
            aucs.append(compute_auc(combined, r['relevant_idx']))
        mean_auc = np.mean(aucs)
        lambda_aucs[lam] = np.array(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lambda = lam

    lambda_results[cond] = {
        'best_lambda': best_lambda,
        'best_auc': best_auc,
        'bare_auc': float(lambda_aucs[0.0].mean()),
        'primed_auc': float(lambda_aucs[1.0].mean()),
        'aucs_by_lambda': {str(l): float(a.mean()) for l, a in lambda_aucs.items()},
        'best_lambda_aucs': lambda_aucs[best_lambda],
    }

print(f"\n  {'Condition':<22s} {'Best lam':>8} {'Best AUC':>9} {'bare':>7} {'Gain':>7}")
print(f"  {'-'*58}")
for cond in PRIMED_CONDITIONS:
    lr = lambda_results[cond]
    gain = lr['best_auc'] - lr['bare_auc']
    print(f"  {cond:<22s} {lr['best_lambda']:>8.1f} {lr['best_auc']:>9.3f} "
          f"{lr['bare_auc']:>7.3f} {gain:>+7.3f}")

# Test best lambda vs bare
print(f"\n  Best lambda vs bare (Wilcoxon):")
for cond in PRIMED_CONDITIONS:
    lr = lambda_results[cond]
    diff = lr['best_lambda_aucs'] - bare_aucs_ref
    d = cohens_d(diff)
    nonzero = diff[diff != 0]
    _, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
    sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
           else '*' if p < 0.05/N_BONFERRONI else 'ns')
    print(f"    {cond:<22s}: lam={lr['best_lambda']:.1f}, AUC={lr['best_auc']:.3f}, "
          f"d={d:+.3f}, p={p:.2e} {sig}")

# === Test 3: Ensemble ===
print(f"\n--- Test 3: Ensemble -- average NLL across conditions ---")

structural_conds = ['surr_template_trunc', 'surr_doc_trunc', 'random_trunc', 'static_fact_trunc']
struct_ensemble_aucs = []
for r in results_04a:
    mean_nlls = np.zeros(r['n_passages'])
    for cond in structural_conds:
        mean_nlls += np.array(r['scores'][cond])
    mean_nlls /= len(structural_conds)
    struct_ensemble_aucs.append(compute_auc(mean_nlls, r['relevant_idx']))
struct_ensemble_aucs = np.array(struct_ensemble_aucs)

diff = struct_ensemble_aucs - bare_aucs_ref
d = cohens_d(diff)
nonzero = diff[diff != 0]
_, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
       else '*' if p < 0.05/N_BONFERRONI else 'ns')
print(f"  Structural (4 conds): AUC={struct_ensemble_aucs.mean():.3f}, "
      f"d={d:+.3f}, p={p:.2e} {sig}")

all5_aucs = []
for r in results_04a:
    mean_nlls = np.zeros(r['n_passages'])
    for cond in PRIMED_CONDITIONS:
        mean_nlls += np.array(r['scores'][cond])
    mean_nlls /= len(PRIMED_CONDITIONS)
    all5_aucs.append(compute_auc(mean_nlls, r['relevant_idx']))
all5_aucs = np.array(all5_aucs)

diff = all5_aucs - bare_aucs_ref
d = cohens_d(diff)
nonzero = diff[diff != 0]
_, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
       else '*' if p < 0.05/N_BONFERRONI else 'ns')
print(f"  All 5 primed:         AUC={all5_aucs.mean():.3f}, "
      f"d={d:+.3f}, p={p:.2e} {sig}")

print(f"\n--- Phase A Summary ---")
print(f"  Bare AUC: {bare_aucs_ref.mean():.3f}")
best_combo_cond = max(lambda_results, key=lambda c: lambda_results[c]['best_auc'])
best_combo = lambda_results[best_combo_cond]
print(f"  Best lambda: {best_combo_cond} lam={best_combo['best_lambda']:.1f}, "
      f"AUC={best_combo['best_auc']:.3f}")
print(f"  Best ensemble: structural 4-cond AUC={struct_ensemble_aucs.mean():.3f}")""")

# ============================================================
# Cell 5: Phase A visualization
# ============================================================
code(r"""# Cell 5: Phase A visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Lambda sweep
ax = axes[0]
for cond in PRIMED_CONDITIONS:
    lr = lambda_results[cond]
    lams = sorted(lr['aucs_by_lambda'].keys(), key=float)
    aucs = [lr['aucs_by_lambda'][l] for l in lams]
    ax.plot([float(l) for l in lams], aucs, '-o', label=cond.replace('_trunc', ''), markersize=4)
ax.axhline(y=bare_aucs_ref.mean(), color='black', linestyle='--', alpha=0.5, label='bare')
ax.set_xlabel('Lambda (0=bare, 1=primed)')
ax.set_ylabel('Mean AUC')
ax.set_title('Lambda Sweep: (1-lam)*bare + lam*primed')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Delta-only ranking
ax = axes[1]
cond_short = [c.replace('_trunc', '') for c in PRIMED_CONDITIONS]
delta_vals = [delta_only_aucs[c].mean() for c in PRIMED_CONDITIONS]
colors = ['C0' if v > 0.5 else 'gray' for v in delta_vals]
ax.bar(range(len(PRIMED_CONDITIONS)), delta_vals, color=colors, alpha=0.7)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='chance')
ax.axhline(y=bare_aucs_ref.mean(), color='black', linestyle='--', alpha=0.5, label='bare NLL')
ax.set_xticks(range(len(PRIMED_CONDITIONS)))
ax.set_xticklabels(cond_short, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('AUC')
ax.set_title('Delta-Only Ranking (higher delta = more relevant)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = RESULTS_DIR / 'phase_a_results.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")""")

# ============================================================
# Cell 6: Load model + data for Phase B
# ============================================================
code(r"""# Cell 6: Load model and rebuild data for Phase B
from lib.data import count_words
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print("=" * 70)
print("PHASE B SETUP: Loading model and rebuilding data")
print("=" * 70)

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Scoring helpers
def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=8192).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids=enc_ids, attention_mask=enc_mask)
    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)
    if ans_ids.shape[1] == 0:
        return 0.0
    with torch.no_grad():
        outputs = model(encoder_outputs=encoder_outputs, attention_mask=cross_attn_mask, labels=ans_ids)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()
    del encoder_outputs, outputs, logits, log_probs
    return mean_nll

def count_prefix_tokens(prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)

# Rebuild query pools (same as 04A with SEED=42)
print(f"\nRebuilding MS MARCO query pools (SEED={SEED})...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

queries = []
all_passage_texts = []

for item in ds:
    passages_data = item.get('passages', {})
    ptexts = passages_data.get('passage_text', [])
    is_sel = passages_data.get('is_selected', [])
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

    word_counts = [count_words(pt) for pt in ptexts]
    if not all(30 <= wc <= 300 for wc in word_counts):
        continue

    n_selected = sum(is_sel)
    n_not_selected = len(is_sel) - n_selected
    if n_selected != 1 or n_not_selected < 2:
        continue

    relevant_idx = is_sel.index(1)
    passages = []
    for p_idx, (pt, sel) in enumerate(zip(ptexts, is_sel)):
        passages.append({'text': pt, 'is_selected': sel})

    queries.append({
        'query': query, 'answer': answer, 'passages': passages,
        'relevant_idx': relevant_idx, 'n_passages': len(passages),
    })
    all_passage_texts.append(ptexts[0])

    if len(queries) >= N_SAMPLES * 3:
        break

del ds
gc.collect()

np.random.seed(SEED)
np.random.shuffle(queries)
queries = queries[:N_SAMPLES]

# Verify alignment with 04A
mismatches = sum(1 for q, r in zip(queries, results_04a) if q['query'][:50] != r['query'][:50])
assert mismatches == 0, f"{mismatches} query mismatches with 04A checkpoint"
print(f"Data alignment verified: all {N_SAMPLES} queries match 04A")

# Generate K=10 fixed random prefixes (SEED+100 for independence)
np.random.seed(SEED + 100)
prefix_indices = np.random.choice(len(all_passage_texts), K_PREFIXES, replace=False)
RANDOM_PREFIXES = [" ".join(all_passage_texts[idx].split()[:20]) for idx in prefix_indices]

print(f"\nGenerated {K_PREFIXES} fixed random prefixes:")
for k, pref in enumerate(RANDOM_PREFIXES):
    print(f"  Prefix {k}: '{pref[:60]}...'")

total_calls = sum(q['n_passages'] for q in queries) * K_PREFIXES
print(f"\nTotal scoring calls: {total_calls}")
print(f"Estimated runtime: ~{total_calls * 0.4 / 3600:.1f} hours")""")

# ============================================================
# Cell 7: Phase B scoring loop
# ============================================================
code(r"""# Cell 7: Score with K=10 random prefixes (with checkpointing)
print("=" * 70)
print("PHASE B: SCORING WITH K=%d RANDOM PREFIXES" % K_PREFIXES)
print("=" * 70)

DIVERSITY_CKPT = RESULTS_DIR / "diversity_checkpoint.json"

diversity_results = []
start_idx = 0
if DIVERSITY_CKPT.exists():
    saved = json.loads(DIVERSITY_CKPT.read_text())
    if saved.get('n_total') == N_SAMPLES and saved.get('k_prefixes') == K_PREFIXES:
        saved_results = saved.get('results', [])
        saved_queries = [r['query'][:50] for r in saved_results]
        current_queries = [q['query'][:50] for q in queries[:len(saved_results)]]
        if saved_queries == current_queries:
            diversity_results = saved_results
            start_idx = len(diversity_results)
            print(f"Resumed from checkpoint: {start_idx}/{N_SAMPLES} queries")

t0 = time.time()

for q_idx in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
                  desc="Prefix diversity"):
    q = queries[q_idx]
    answer = q['answer']

    query_result = {
        'query_idx': q_idx,
        'query': q['query'],
        'n_passages': q['n_passages'],
        'relevant_idx': q['relevant_idx'],
        'diversity_scores': [],
    }

    for p_idx, passage_data in enumerate(q['passages']):
        passage_nlls = []
        for k in range(K_PREFIXES):
            prefix = RANDOM_PREFIXES[k]
            enc_text = prefix + "\n" + passage_data['text']
            prefix_count = count_prefix_tokens(prefix, passage_data['text'])
            nll = score_nll(enc_text, answer, prefix_count, truncate=True)
            passage_nlls.append(nll)
        query_result['diversity_scores'].append(passage_nlls)

    diversity_results.append(query_result)

    if (q_idx + 1) % 20 == 0 or q_idx == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'k_prefixes': K_PREFIXES,
            'results': diversity_results,
            'completed': len(diversity_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        DIVERSITY_CKPT.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = q_idx - start_idx + 1
        eta = (N_SAMPLES - q_idx - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {q_idx+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed_total = time.time() - t0
print(f"\nDiversity scoring complete: {len(diversity_results)} queries in {elapsed_total/60:.1f} min")""")

# ============================================================
# Cell 8: Phase B analysis
# ============================================================
code(r"""# Cell 8: Phase B -- Diversity analysis and ranking
print("=" * 70)
print("PHASE B: PREFIX DIVERSITY ANALYSIS")
print("=" * 70)

# Compute per-passage diversity stats
for dr in diversity_results:
    dr['passage_stats'] = []
    for p_scores in dr['diversity_scores']:
        arr = np.array(p_scores)
        dr['passage_stats'].append({
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'range': float(arr.max() - arr.min()),
            'cv': float(arr.std() / arr.mean()) if arr.mean() > 0 else 0,
        })

# Relevant vs irrelevant diversity
print(f"\n--- NLL diversity by relevance ---")
for stat_name in ['mean', 'std', 'range', 'cv']:
    rel_vals, irrel_vals = [], []
    for dr in diversity_results:
        rel_idx = dr['relevant_idx']
        for p_idx, ps in enumerate(dr['passage_stats']):
            if p_idx == rel_idx:
                rel_vals.append(ps[stat_name])
            else:
                irrel_vals.append(ps[stat_name])
    rel_mean = np.mean(rel_vals)
    irrel_mean = np.mean(irrel_vals)
    diff = rel_mean - irrel_mean
    pooled_std = np.sqrt((np.var(rel_vals) * len(rel_vals) + np.var(irrel_vals) * len(irrel_vals))
                          / (len(rel_vals) + len(irrel_vals)))
    d = diff / pooled_std if pooled_std > 0 else 0
    print(f"  {stat_name:>5s}: relevant={rel_mean:.4f}, irrelevant={irrel_mean:.4f}, "
          f"diff={diff:+.4f}, d={d:+.3f}")

# === Ranking by NLL_mean (ensemble of K random prefixes) ===
print(f"\n--- Rank by NLL_mean (ensemble of K={K_PREFIXES} random prefixes) ---")
div_mean_aucs = []
div_mean_mrr3s = []
div_mean_hit1s = []
for dr in diversity_results:
    rel_idx = dr['relevant_idx']
    mean_nlls = np.array([ps['mean'] for ps in dr['passage_stats']])
    div_mean_aucs.append(compute_auc(mean_nlls, rel_idx))
    div_mean_mrr3s.append(compute_mrr_at_k(mean_nlls, rel_idx, k=3))
    div_mean_hit1s.append(compute_hit_at_k(mean_nlls, rel_idx, k=1))
div_mean_aucs = np.array(div_mean_aucs)

diff = div_mean_aucs - bare_aucs_ref
d = cohens_d(diff)
nonzero = diff[diff != 0]
_, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
       else '*' if p < 0.05/N_BONFERRONI else 'ns')
print(f"  AUC={div_mean_aucs.mean():.3f} (vs bare {bare_aucs_ref.mean():.3f}), "
      f"d={d:+.3f}, p={p:.2e} {sig}")
print(f"  MRR@3={np.mean(div_mean_mrr3s):.3f}, Hit@1={np.mean(div_mean_hit1s):.3f}")

# === Ranking by NLL_std (test both directions) ===
print(f"\n--- Rank by NLL_std ---")
for direction, label in [(1, 'lower std = better'), (-1, 'higher std = better')]:
    std_aucs = []
    for dr in diversity_results:
        rel_idx = dr['relevant_idx']
        std_vals = np.array([ps['std'] for ps in dr['passage_stats']])
        if direction == -1:
            std_vals = -std_vals
        std_aucs.append(compute_auc(std_vals, rel_idx))
    std_aucs = np.array(std_aucs)
    d_vs_chance = cohens_d(std_aucs - 0.5)
    print(f"  {label}: AUC={std_aucs.mean():.3f}, d vs 0.5={d_vs_chance:+.3f}")

# Store better direction
std_aucs_lower = []
std_aucs_higher = []
for dr in diversity_results:
    rel_idx = dr['relevant_idx']
    std_vals = np.array([ps['std'] for ps in dr['passage_stats']])
    std_aucs_lower.append(compute_auc(std_vals, rel_idx))
    std_aucs_higher.append(compute_auc(-std_vals, rel_idx))
std_aucs_lower = np.array(std_aucs_lower)
std_aucs_higher = np.array(std_aucs_higher)
if std_aucs_higher.mean() > std_aucs_lower.mean():
    std_best_aucs = std_aucs_higher
    std_direction = "higher std = more relevant"
else:
    std_best_aucs = std_aucs_lower
    std_direction = "lower std = more relevant"
print(f"  Best direction: {std_direction}, AUC={std_best_aucs.mean():.3f}")

# === Lambda sweep: bare + diversity_mean ===
print(f"\n--- Lambda sweep: (1-lam)*bare + lam*diversity_mean ---")
LAMBDAS_B = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
best_div_auc = 0
best_div_lambda = 0
div_lambda_aucs = {}

for lam in LAMBDAS_B:
    aucs = []
    for i, dr in enumerate(diversity_results):
        rel_idx = dr['relevant_idx']
        bare_nlls = np.array(results_04a[i]['scores']['bare'])
        div_nlls = np.array([ps['mean'] for ps in dr['passage_stats']])
        combined = (1 - lam) * bare_nlls + lam * div_nlls
        aucs.append(compute_auc(combined, rel_idx))
    mean_auc = np.mean(aucs)
    div_lambda_aucs[lam] = np.array(aucs)
    if mean_auc > best_div_auc:
        best_div_auc = mean_auc
        best_div_lambda = lam

print(f"  Best lambda={best_div_lambda:.1f}, AUC={best_div_auc:.3f}")
for lam in LAMBDAS_B:
    print(f"    lam={lam:.1f}: AUC={div_lambda_aucs[lam].mean():.3f}")""")

# ============================================================
# Cell 9: Phase C — Combined multi-feature
# ============================================================
code(r"""# Cell 9: Phase C -- Combined multi-feature ranking
print("=" * 70)
print("PHASE C: COMBINED MULTI-FEATURE RANKING")
print("=" * 70)

best_single_cond = max(PRIMED_CONDITIONS, key=lambda c: lambda_results[c]['best_auc'])
print(f"Best single primed: {best_single_cond} (AUC={lambda_results[best_single_cond]['best_auc']:.3f})")

# Grid search: w_bare + w_primed + w_div = 1.0 (step 0.1)
print(f"\n--- Grid search: w_bare*bare + w_primed*{best_single_cond.replace('_trunc','')} + w_div*div_mean ---")
grid_results = []

for w_bare_pct in range(0, 11):
    for w_primed_pct in range(0, 11 - w_bare_pct):
        w_div_pct = 10 - w_bare_pct - w_primed_pct
        w_bare = w_bare_pct / 10
        w_primed = w_primed_pct / 10
        w_div = w_div_pct / 10

        aucs = []
        for i, dr in enumerate(diversity_results):
            rel_idx = dr['relevant_idx']
            bare_nlls = np.array(results_04a[i]['scores']['bare'])
            primed_nlls = np.array(results_04a[i]['scores'][best_single_cond])
            div_nlls = np.array([ps['mean'] for ps in dr['passage_stats']])
            combined = w_bare * bare_nlls + w_primed * primed_nlls + w_div * div_nlls
            aucs.append(compute_auc(combined, rel_idx))

        grid_results.append({
            'w_bare': w_bare, 'w_primed': w_primed, 'w_div': w_div,
            'auc': np.mean(aucs), 'aucs': np.array(aucs),
        })

grid_results.sort(key=lambda x: x['auc'], reverse=True)

print(f"\n  Top 10 weight combinations:")
print(f"  {'w_bare':>7} {'w_primed':>9} {'w_div':>6} {'AUC':>7}")
print(f"  {'-'*33}")
for gr in grid_results[:10]:
    print(f"  {gr['w_bare']:>7.1f} {gr['w_primed']:>9.1f} {gr['w_div']:>6.1f} {gr['auc']:>7.3f}")

best_grid = grid_results[0]
diff = best_grid['aucs'] - bare_aucs_ref
d = cohens_d(diff)
nonzero = diff[diff != 0]
_, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
       else '*' if p < 0.05/N_BONFERRONI else 'ns')
print(f"\n  Best: w=({best_grid['w_bare']:.1f}, {best_grid['w_primed']:.1f}, {best_grid['w_div']:.1f})")
print(f"  AUC={best_grid['auc']:.3f} vs bare {bare_aucs_ref.mean():.3f}, d={d:+.3f}, p={p:.2e} {sig}")""")

# ============================================================
# Cell 10: Grand comparison + verdict + save
# ============================================================
code(r"""# Cell 10: Grand comparison + verdict + save
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("GRAND COMPARISON: ALL RANKING METHODS")
print("=" * 70)

methods = {'bare NLL': bare_aucs_ref}

for cond in PRIMED_CONDITIONS:
    lr = lambda_results[cond]
    name = f"{cond.replace('_trunc','')} (lam={lr['best_lambda']:.1f})"
    methods[name] = lr['best_lambda_aucs']

methods['ensemble_structural_4'] = struct_ensemble_aucs
methods[f'diversity_mean (K={K_PREFIXES})'] = div_mean_aucs
methods[f'div_combo (lam={best_div_lambda:.1f})'] = div_lambda_aucs[best_div_lambda]
methods[f'combined ({best_grid["w_bare"]:.1f}/{best_grid["w_primed"]:.1f}/{best_grid["w_div"]:.1f})'] = best_grid['aucs']

sorted_methods = sorted(methods.items(), key=lambda x: x[1].mean(), reverse=True)

print(f"\n  {'#':>3} {'Method':<48} {'AUC':>7} {'vs bare':>8} {'d':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*95}")
for rank, (name, aucs) in enumerate(sorted_methods, 1):
    mean_auc = aucs.mean()
    if name == 'bare NLL':
        print(f"  {rank:>3} {name:<48} {mean_auc:>7.3f} {'--':>8} {'--':>7} {'--':>12} {'--':>5}")
    else:
        diff = aucs - bare_aucs_ref
        d = cohens_d(diff)
        nonzero = diff[diff != 0]
        _, p = wilcoxon(nonzero) if len(nonzero) >= 10 else (0, 1.0)
        sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
               else '*' if p < 0.05/N_BONFERRONI else 'ns')
        delta_str = f"{mean_auc - bare_aucs_ref.mean():+.3f}"
        print(f"  {rank:>3} {name:<48} {mean_auc:>7.3f} {delta_str:>8} {d:>+7.3f} {p:>12.2e} {sig:>5}")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
names = [n for n, _ in sorted_methods]
aucs_vals = [a.mean() for _, a in sorted_methods]
colors = ['gray' if n == 'bare NLL' else 'C0' for n in names]
ax.barh(range(len(names)), aucs_vals, color=colors, alpha=0.7)
ax.axvline(x=bare_aucs_ref.mean(), color='red', linestyle='--', alpha=0.5, label='bare')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Mean AUC')
ax.set_title('Exp 11: All Ranking Methods Compared')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plot_path = RESULTS_DIR / 'grand_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {plot_path}")

# === VERDICT ===
print(f"\n{'='*70}")
print(f"VERDICT -- Exp 11: Delta-as-Feature and Prefix Diversity")
print(f"{'='*70}")

best_delta_cond = max(PRIMED_CONDITIONS, key=lambda c: delta_only_aucs[c].mean())
print(f"\n--- Phase A: Delta-as-Feature ---")
print(f"  Best delta-only: {best_delta_cond.replace('_trunc','')}, AUC={delta_only_aucs[best_delta_cond].mean():.3f}")
if delta_only_aucs[best_delta_cond].mean() > 0.55:
    print(f"  -> Delta carries ranking information above chance")
else:
    print(f"  -> Delta alone is near chance (not useful standalone)")
print(f"  Best lambda: {best_combo_cond} lam={best_combo['best_lambda']:.1f}, "
      f"AUC={best_combo['best_auc']:.3f}")
print(f"  Best ensemble: structural 4-cond AUC={struct_ensemble_aucs.mean():.3f}")

print(f"\n--- Phase B: Prefix Diversity ---")
print(f"  NLL_mean (K={K_PREFIXES}): AUC={div_mean_aucs.mean():.3f}")
print(f"  NLL_std direction: {std_direction}")
print(f"  Best diversity combo: lam={best_div_lambda:.1f}, AUC={best_div_auc:.3f}")

print(f"\n--- Phase C: Combined ---")
print(f"  Best multi-feature: w=({best_grid['w_bare']:.1f}, {best_grid['w_primed']:.1f}, "
      f"{best_grid['w_div']:.1f}), AUC={best_grid['auc']:.3f}")

overall_best_name, overall_best_aucs = sorted_methods[0]
gain = overall_best_aucs.mean() - bare_aucs_ref.mean()
print(f"\n--- OVERALL ---")
print(f"  Best method: {overall_best_name}")
print(f"  AUC: {overall_best_aucs.mean():.3f} (bare: {bare_aucs_ref.mean():.3f}, gain: {gain:+.3f})")

if gain > 0.02:
    print(f"\n  >>> PRACTICAL IMPROVEMENT: +{gain:.3f} AUC")
    print(f"  >>> Structural perturbation signals ARE useful for ranking")
elif gain > 0.01:
    print(f"\n  >>> MODERATE IMPROVEMENT: +{gain:.3f} AUC")
else:
    print(f"\n  >>> MINIMAL IMPROVEMENT over single-condition ranking from 04A")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp11_delta_and_diversity',
    'model': MODEL_NAME,
    'n_queries': N_SAMPLES,
    'k_prefixes': K_PREFIXES,
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'phase_a': {
        'delta_only_aucs': {c: float(delta_only_aucs[c].mean()) for c in PRIMED_CONDITIONS},
        'lambda_results': {c: {
            'best_lambda': lr['best_lambda'],
            'best_auc': lr['best_auc'],
            'aucs_by_lambda': lr['aucs_by_lambda'],
        } for c, lr in lambda_results.items()},
        'ensemble_structural_4_auc': float(struct_ensemble_aucs.mean()),
        'ensemble_all_5_auc': float(all5_aucs.mean()),
    },
    'phase_b': {
        'diversity_mean_auc': float(div_mean_aucs.mean()),
        'std_direction': std_direction,
        'std_best_auc': float(std_best_aucs.mean()),
        'best_diversity_lambda': float(best_div_lambda),
        'best_diversity_auc': float(best_div_auc),
    },
    'phase_c': {
        'best_weights': {
            'w_bare': best_grid['w_bare'],
            'w_primed': best_grid['w_primed'],
            'w_div': best_grid['w_div'],
        },
        'best_combined_auc': float(best_grid['auc']),
    },
    'bare_auc': float(bare_aucs_ref.mean()),
    'grand_comparison': [
        {'method': name, 'auc': float(aucs.mean())} for name, aucs in sorted_methods
    ],
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
# Cell 11: Cleanup
# ============================================================
code("""# Cell 11: Cleanup
print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")""")

# ============================================================
# Write notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

outpath = "experiments/11/11_delta_and_diversity.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
