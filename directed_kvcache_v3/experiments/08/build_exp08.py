#!/usr/bin/env python3
# Generate Exp 08 notebook: Contrastive & Differential Ranking Analysis.
#
# Exp 04A showed zero oracle differential (d=-0.007). The ~85% structural component
# helps all passages equally. But ~15% is semantic -- can we isolate it for ranking?
#
# Key addition: oracle_cross_trunc (score each pool with a DIFFERENT query's text).
# Since structural benefit is length-dependent (not content-dependent), oracle and
# oracle_cross should give ~same structural benefit. The DIFFERENCE isolates the
# semantic interaction between query and document.
#
# Six analysis strategies all computed from the same NLL data.
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
md(r"""# Experiment 08: Contrastive & Differential Ranking Analysis
## Can we isolate the semantic residual for ranking?

### Motivation
Exp 04A showed zero oracle differential (d=-0.007): priming helps relevant and
irrelevant passages equally. The ~85% structural component is document-independent.
But ~15% IS semantic. Can we isolate it?

### Key idea: oracle_cross_trunc
Each query's passage pool is scored with a **different** query's oracle text.
Since the structural benefit depends on prefix LENGTH (not content), oracle and
oracle_cross should provide ~identical structural lift. The **difference** between
NLL(oracle) and NLL(oracle_cross) isolates the **semantic interaction** between the
correct query and each passage.

### Conditions (7)
| # | Condition | Prefix | Notes |
|---|-----------|--------|-------|
| 1 | bare | (none) | lower bound |
| 2 | oracle\_trunc | real query | upper bound |
| 3 | oracle\_cross\_trunc | DIFFERENT query | **NEW** -- structural control |
| 4 | surr\_template\_trunc | "What is [kw]?" per passage | doc-derived |
| 5 | surr\_doc\_trunc | TF keywords per passage | doc-derived |
| 6 | random\_trunc | unrelated text | structural control |
| 7 | static\_fact\_trunc | "What are the key facts?" | content-agnostic |

### Six Analysis Strategies (all from the same NLL data)

**A. Standard ranking**: rank by raw NLL (replicates 04A)
**B. Delta-NLL ranking**: rank by NLL(bare) - NLL(condition)
**C. Relative delta**: subtract pool-mean delta, rank by deviation
**D. Per-doc template contrastive**: variation across per-passage templates
**E. Oracle cross-contrastive**: rank by NLL(oracle) - NLL(oracle_cross) -- KEY
**F. Vocabulary overlap correlation**: correlate delta with Jaccard(query, passage)

### N=400 queries (~8.2 passages/query)
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
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

sys.path.insert(0, "../..")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../../results/exp08")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 400   # queries
MODEL_NAME = "google/t5gemma-2-4b-4b"
N_BONFERRONI = 6  # 6 non-bare conditions

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 08: Contrastive & Differential Ranking Analysis")
print(f"Model: {MODEL_NAME}")
print(f"N queries: {N_SAMPLES}")
print(f"Bonferroni comparisons: {N_BONFERRONI}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

# ============================================================
# Cell 3: Load model
# ============================================================
code("""# Cell 3: Load model
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

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
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")""")

# ============================================================
# Cell 4: Scoring helpers
# ============================================================
code(r"""# Cell 4: Scoring and ranking helpers

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer tokens with optional truncation.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=8192).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

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
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            labels=ans_ids,
        )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    # Count prefix tokens in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)


# === Surrogate generation ===
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'if', 'then', 'than', 'so', 'up', 'out', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'they', 'them', 'their', 'how', 'when',
    'where', 'why', 'much', 'many', 'some', 'any', 'all', 'each',
    'does', 'also', 'just', 'more', 'most', 'very', 'too', 'only',
}

def extract_keywords(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def make_surrogate_doc_kw(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"

STATIC_FACT = "What are the key facts I need to know?"


# === Ranking metrics ===
def compute_auc(nlls, relevant_idx):
    # AUC: lower NLL = more relevant. Exactly one relevant passage.
    rel_nll = nlls[relevant_idx]
    irrel_nlls = [nlls[i] for i in range(len(nlls)) if i != relevant_idx]
    n_irrel = len(irrel_nlls)
    if n_irrel == 0:
        return 0.5
    wins = sum(1 for nll in irrel_nlls if nll > rel_nll)
    ties = sum(1 for nll in irrel_nlls if nll == rel_nll)
    return (wins + 0.5 * ties) / n_irrel

def compute_auc_higher_better(scores, relevant_idx):
    # AUC when higher score = more relevant.
    rel_score = scores[relevant_idx]
    irrel_scores = [scores[i] for i in range(len(scores)) if i != relevant_idx]
    n_irrel = len(irrel_scores)
    if n_irrel == 0:
        return 0.5
    wins = sum(1 for s in irrel_scores if s < rel_score)
    ties = sum(1 for s in irrel_scores if s == rel_score)
    return (wins + 0.5 * ties) / n_irrel

def compute_mrr_at_k(nlls, relevant_idx, k=3):
    # MRR@k: rank by ascending NLL (lower = more relevant).
    ranked_indices = list(np.argsort(nlls))
    for rank, idx in enumerate(ranked_indices[:k], 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0

def compute_hit_at_k(nlls, relevant_idx, k=1):
    # Hit@k: 1 if relevant in top-k by ascending NLL.
    ranked_indices = set(np.argsort(nlls)[:k].tolist())
    return 1.0 if relevant_idx in ranked_indices else 0.0

def compute_vocab_overlap(query, passage):
    # Jaccard overlap between query and passage content words.
    q_words = set(extract_keywords(query))
    p_words = set(extract_keywords(passage))
    if not q_words or not p_words:
        return 0.0
    return len(q_words & p_words) / len(q_words | p_words)


print("Helpers defined.")
print("  Scoring: score_nll (answer-likelihood)")
print("  Surrogates: doc_kw, template, static_fact")
print("  Ranking: AUC, AUC_higher_better, MRR@k, Hit@k")
print("  Analysis: compute_vocab_overlap (Jaccard)")""")

# ============================================================
# Cell 5: Load data + assign cross-queries
# ============================================================
code(r"""# Cell 5: Load MS MARCO ranking data + assign cross-queries
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

queries = []

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
        passages.append({
            'text': pt,
            'is_selected': sel,
            'word_count': word_counts[p_idx],
            'surr_doc_kw': make_surrogate_doc_kw(pt),
            'surr_template': make_surrogate_template(pt),
        })

    queries.append({
        'query': query,
        'answer': answer,
        'passages': passages,
        'relevant_idx': relevant_idx,
        'n_passages': len(passages),
    })

    if len(queries) >= N_SAMPLES * 3:
        break

del ds
gc.collect()

# Shuffle and select
np.random.seed(SEED)
np.random.shuffle(queries)
queries = queries[:N_SAMPLES]

# Assign cross-queries: each query i gets a different query's text
# Use circular offset to ensure no self-assignment
for i, q in enumerate(queries):
    cross_idx = (i + 1) % len(queries)
    q['cross_query'] = queries[cross_idx]['query']

# Generate per-query random surrogates
for i, q in enumerate(queries):
    other_idx = (i + N_SAMPLES // 2) % len(queries)
    other_passage = queries[other_idx]['passages'][queries[other_idx]['relevant_idx']]['text']
    q['surr_random'] = " ".join(other_passage.split()[:20])

# Stats
n_passages_list = [q['n_passages'] for q in queries]
print(f"Selected {len(queries)} queries for ranking")
print(f"Passages per query: mean={np.mean(n_passages_list):.1f}, "
      f"median={np.median(n_passages_list):.0f}, "
      f"min={np.min(n_passages_list)}, max={np.max(n_passages_list)}")
total_scorings = sum(n_passages_list) * 7  # 7 conditions
print(f"Total passage scorings: {total_scorings}")
print(f"Estimated runtime: ~{total_scorings * 0.4 / 3600:.1f} hours")

# Verify cross-query lengths are similar to oracle
oracle_lens = [len(q['query'].split()) for q in queries]
cross_lens = [len(q['cross_query'].split()) for q in queries]
print(f"\nQuery lengths: oracle mean={np.mean(oracle_lens):.1f}, "
      f"cross mean={np.mean(cross_lens):.1f}")
print(f"Length correlation: r={np.corrcoef(oracle_lens, cross_lens)[0,1]:.3f}")""")

# ============================================================
# Cell 6: Condition examples
# ============================================================
code(r"""# Cell 6: Condition examples (including oracle_cross)
print("=" * 70)
print("CONDITION EXAMPLES")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'oracle_cross_trunc',
              'surr_template_trunc', 'surr_doc_trunc',
              'random_trunc', 'static_fact_trunc']

ex = queries[0]
print(f"\nQuery: {ex['query']}")
print(f"Cross-query: {ex['cross_query']}")
print(f"Answer: {ex['answer']}")
print(f"Passages: {ex['n_passages']} ({ex['n_passages']-1} irrelevant, "
      f"1 relevant at idx {ex['relevant_idx']})")

rel_p = ex['passages'][ex['relevant_idx']]
print(f"\nRelevant passage (idx {ex['relevant_idx']}):")
print(f"  Text: {rel_p['text'][:120]}...")
print(f"  surr_template: {rel_p['surr_template']}")
print(f"  surr_doc_kw: {rel_p['surr_doc_kw']}")

print(f"\nWhat the encoder sees for the relevant passage:")
for cond in COND_NAMES:
    if cond == 'bare':
        enc = rel_p['text'][:70] + "..."
    elif cond == 'oracle_trunc':
        enc = ex['query'][:30] + "... | " + rel_p['text'][:30] + "..."
    elif cond == 'oracle_cross_trunc':
        enc = ex['cross_query'][:30] + "... | " + rel_p['text'][:30] + "..."
    elif cond == 'surr_template_trunc':
        enc = rel_p['surr_template'] + " | " + rel_p['text'][:30] + "..."
    elif cond == 'surr_doc_trunc':
        enc = rel_p['surr_doc_kw'] + " | " + rel_p['text'][:30] + "..."
    elif cond == 'random_trunc':
        enc = ex['surr_random'][:30] + "... | " + rel_p['text'][:30] + "..."
    elif cond == 'static_fact_trunc':
        enc = STATIC_FACT[:30] + "... | " + rel_p['text'][:30] + "..."
    print(f"  {cond:<24s}: {enc}")

print(f"\nKey insight: oracle_cross_trunc uses query '{ex['cross_query'][:50]}...'")
print(f"  This provides the SAME structural benefit as oracle_trunc")
print(f"  (similar prefix length) but WRONG semantic content.")
print(f"  Difference isolates the semantic signal.")""")

# ============================================================
# Cell 7: Scoring loop
# ============================================================
code(r"""# Cell 7: Scoring loop (7 conditions x all passages)
print("=" * 70)
print("RUNNING CONTRASTIVE RANKING EXPERIMENT")
print("=" * 70)

def build_condition_input(cond_name, passage_data, query_data):
    # Return (encoder_text, prefix_token_count, truncate) for a condition.
    passage_text = passage_data['text']

    if cond_name == 'bare':
        return passage_text, 0, False
    elif cond_name == 'oracle_trunc':
        surr = query_data['query']
    elif cond_name == 'oracle_cross_trunc':
        surr = query_data['cross_query']
    elif cond_name == 'surr_template_trunc':
        surr = passage_data['surr_template']
    elif cond_name == 'surr_doc_trunc':
        surr = passage_data['surr_doc_kw']
    elif cond_name == 'random_trunc':
        surr = query_data['surr_random']
    elif cond_name == 'static_fact_trunc':
        surr = STATIC_FACT
    else:
        raise ValueError(f"Unknown condition: {cond_name}")

    enc_text = surr + "\n" + passage_text
    prefix_count = count_prefix_tokens(surr, passage_text)
    return enc_text, prefix_count, True


# Resume from checkpoint
results = []
start_idx = 0
if CHECKPOINT_PATH.exists():
    saved = json.loads(CHECKPOINT_PATH.read_text())
    if saved.get('n_total') == N_SAMPLES:
        saved_results = saved.get('results', [])
        saved_queries = [r['query'][:50] for r in saved_results]
        current_queries = [q['query'][:50] for q in queries[:len(saved_results)]]
        if saved_queries == current_queries:
            results = saved_results
            start_idx = len(results)
            print(f"Resumed from checkpoint: {start_idx}/{N_SAMPLES} queries")

t0 = time.time()

for q_idx in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
                  desc="Queries"):
    q = queries[q_idx]
    answer = q['answer']

    query_result = {
        'query_idx': q_idx,
        'query': q['query'],
        'cross_query': q['cross_query'],
        'answer': answer,
        'n_passages': q['n_passages'],
        'relevant_idx': q['relevant_idx'],
        'is_selected': [p['is_selected'] for p in q['passages']],
        'scores': {},
    }

    for cond_name in COND_NAMES:
        cond_nlls = []
        for p_idx, passage_data in enumerate(q['passages']):
            enc_text, prefix_count, truncate = build_condition_input(
                cond_name, passage_data, q)
            nll = score_nll(enc_text, answer, prefix_count, truncate)
            cond_nlls.append(nll)
        query_result['scores'][cond_name] = cond_nlls

    results.append(query_result)

    if (q_idx + 1) % 20 == 0 or q_idx == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': results,
            'completed': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = q_idx - start_idx + 1
        eta = (N_SAMPLES - q_idx - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {q_idx+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed_total = time.time() - t0
print(f"\nScoring complete: {len(results)} queries in {elapsed_total/60:.1f} min")""")

# ============================================================
# Cell 8: Strategy A -- Standard ranking
# ============================================================
code(r"""# Cell 8: Strategy A -- Standard ranking (replicates Exp 04A)
from lib.analysis import cohens_d

print("=" * 70)
print("STRATEGY A: STANDARD RANKING (raw NLL)")
print("=" * 70)
print("Rank by raw NLL. Should replicate Exp 04A for shared conditions.\n")

metrics_a = {cond: {'auc': [], 'mrr3': [], 'hit1': [], 'hit3': []}
             for cond in COND_NAMES}

for r in results:
    rel_idx = r['relevant_idx']
    for cond in COND_NAMES:
        nlls = np.array(r['scores'][cond])
        metrics_a[cond]['auc'].append(compute_auc(nlls, rel_idx))
        metrics_a[cond]['mrr3'].append(compute_mrr_at_k(nlls, rel_idx, k=3))
        metrics_a[cond]['hit1'].append(compute_hit_at_k(nlls, rel_idx, k=1))
        metrics_a[cond]['hit3'].append(compute_hit_at_k(nlls, rel_idx, k=3))

for cond in COND_NAMES:
    for m in metrics_a[cond]:
        metrics_a[cond][m] = np.array(metrics_a[cond][m])

# Results table
print(f"{'Condition':<24} {'AUC':>8} {'MRR@3':>8} {'Hit@1':>8} {'Hit@3':>8} "
      f"{'AUC d':>8} {'AUC p':>12} {'sig':>5}")
print("-" * 90)

bare_aucs = metrics_a['bare']['auc']
strategy_a = {}

for cond in COND_NAMES:
    auc_mean = metrics_a[cond]['auc'].mean()
    mrr_mean = metrics_a[cond]['mrr3'].mean()
    hit1_mean = metrics_a[cond]['hit1'].mean()
    hit3_mean = metrics_a[cond]['hit3'].mean()

    if cond == 'bare':
        print(f"{cond:<24} {auc_mean:>8.3f} {mrr_mean:>8.3f} {hit1_mean:>8.3f} "
              f"{hit3_mean:>8.3f} {'--':>8} {'--':>12} {'--':>5}")
        strategy_a[cond] = {'auc': float(auc_mean)}
    else:
        diff = metrics_a[cond]['auc'] - bare_aucs
        d = cohens_d(diff)
        nonzero = diff[diff != 0]
        if len(nonzero) >= 10:
            try:
                _, p_val = wilcoxon(nonzero)
            except ValueError:
                p_val = 1.0
        else:
            p_val = 1.0
        sig = ('***' if p_val < 0.001/N_BONFERRONI else
               '**' if p_val < 0.01/N_BONFERRONI else
               '*' if p_val < 0.05/N_BONFERRONI else 'ns')
        print(f"{cond:<24} {auc_mean:>8.3f} {mrr_mean:>8.3f} {hit1_mean:>8.3f} "
              f"{hit3_mean:>8.3f} {d:>+8.3f} {p_val:>12.2e} {sig:>5}")
        strategy_a[cond] = {
            'auc': float(auc_mean), 'd': float(d), 'p': float(p_val),
        }

# Cross-reference with Exp 04A
print(f"\nExp 04A reference (6 shared conditions):")
print(f"  bare AUC ~ 0.845, oracle AUC ~ 0.853 (ns)")
print(f"  oracle differential d = -0.007 (ns)")

# New condition: oracle_cross_trunc
cross_auc = strategy_a.get('oracle_cross_trunc', {}).get('auc', 0)
oracle_auc = strategy_a.get('oracle_trunc', {}).get('auc', 0)
print(f"\nNEW: oracle_cross_trunc AUC = {cross_auc:.3f}")
print(f"     oracle_trunc AUC = {oracle_auc:.3f}")
print(f"     Difference = {oracle_auc - cross_auc:+.3f}")""")

# ============================================================
# Cell 9: Strategy B+C -- Delta-NLL and relative delta ranking
# ============================================================
code(r"""# Cell 9: Strategies B+C -- Delta-NLL and Relative Delta ranking
print("=" * 70)
print("STRATEGY B: DELTA-NLL RANKING")
print("=" * 70)
print("Rank by delta = NLL(bare) - NLL(condition).")
print("Higher delta = more improved by priming = predicted relevant.\n")

strategy_b = {}
strategy_c = {}

for cond in COND_NAMES[1:]:  # skip bare
    aucs_b = []
    aucs_c = []

    for r in results:
        rel_idx = r['relevant_idx']
        bare_nlls_q = np.array(r['scores']['bare'])
        cond_nlls_q = np.array(r['scores'][cond])

        # Strategy B: rank by delta (higher = better)
        deltas = bare_nlls_q - cond_nlls_q
        aucs_b.append(compute_auc_higher_better(deltas, rel_idx))

        # Strategy C: rank by relative delta (subtract pool mean)
        pool_mean_delta = deltas.mean()
        relative_deltas = deltas - pool_mean_delta
        aucs_c.append(compute_auc_higher_better(relative_deltas, rel_idx))

    aucs_b = np.array(aucs_b)
    aucs_c = np.array(aucs_c)

    # Compare to Strategy A and chance
    d_b = cohens_d(aucs_b - 0.5)  # vs chance
    d_c = cohens_d(aucs_c - 0.5)

    _, p_b = stats.ttest_1samp(aucs_b, 0.5)
    _, p_c = stats.ttest_1samp(aucs_c, 0.5)

    sig_b = ('***' if p_b < 0.001 else '**' if p_b < 0.01 else
             '*' if p_b < 0.05 else 'ns')
    sig_c = ('***' if p_c < 0.001 else '**' if p_c < 0.01 else
             '*' if p_c < 0.05 else 'ns')

    strategy_b[cond] = {'auc': float(aucs_b.mean()), 'd_vs_chance': float(d_b),
                         'p': float(p_b)}
    strategy_c[cond] = {'auc': float(aucs_c.mean()), 'd_vs_chance': float(d_c),
                         'p': float(p_c)}

print(f"{'Condition':<24} {'B: delta AUC':>14} {'d vs 0.5':>10} {'sig':>5} "
      f"{'C: rel delta':>14} {'d vs 0.5':>10} {'sig':>5}")
print("-" * 90)

for cond in COND_NAMES[1:]:
    b = strategy_b[cond]
    c = strategy_c[cond]
    sig_b = ('***' if b['p'] < 0.001 else '**' if b['p'] < 0.01 else
             '*' if b['p'] < 0.05 else 'ns')
    sig_c = ('***' if c['p'] < 0.001 else '**' if c['p'] < 0.01 else
             '*' if c['p'] < 0.05 else 'ns')
    print(f"  {cond:<22} {b['auc']:>14.3f} {b['d_vs_chance']:>+10.3f} {sig_b:>5} "
          f"{c['auc']:>14.3f} {c['d_vs_chance']:>+10.3f} {sig_c:>5}")

print(f"\nInterpretation:")
print(f"  Strategy B: higher delta = more improved -> predicted relevant")
print(f"  Strategy C: remove pool-level shift, rank by deviation from mean")
print(f"  AUC > 0.5 with p<0.05 = ranking signal detected")""")

# ============================================================
# Cell 10: Strategy D -- Per-doc template contrastive
# ============================================================
code(r"""# Cell 10: Strategy D -- Per-doc template contrastive
print("=" * 70)
print("STRATEGY D: PER-DOC TEMPLATE CONTRASTIVE")
print("=" * 70)
print("surr_template is per-passage (different keyword per doc).")
print("Structural component is ~constant -> variation is semantic match.\n")

# For each query, compute the template delta for each passage
# Then check if relevant passage has higher template delta
strategy_d = {'auc': [], 'delta_rel': [], 'delta_irrel': []}

for r in results:
    rel_idx = r['relevant_idx']
    bare_nlls_q = np.array(r['scores']['bare'])
    template_nlls_q = np.array(r['scores']['surr_template_trunc'])
    random_nlls_q = np.array(r['scores']['random_trunc'])

    # Template delta minus random delta: isolates per-passage keyword match
    template_delta = bare_nlls_q - template_nlls_q
    random_delta = bare_nlls_q - random_nlls_q
    contrastive_delta = template_delta - random_delta

    # Rank by contrastive_delta (higher = template helped more than random)
    strategy_d['auc'].append(compute_auc_higher_better(contrastive_delta, rel_idx))
    strategy_d['delta_rel'].append(contrastive_delta[rel_idx])
    irrel_deltas = [contrastive_delta[i] for i in range(len(contrastive_delta))
                    if i != rel_idx]
    strategy_d['delta_irrel'].append(np.mean(irrel_deltas))

for k in strategy_d:
    strategy_d[k] = np.array(strategy_d[k])

auc_mean = strategy_d['auc'].mean()
d_vs_chance = cohens_d(strategy_d['auc'] - 0.5)
_, p_val = stats.ttest_1samp(strategy_d['auc'], 0.5)
sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
       '*' if p_val < 0.05 else 'ns')

print(f"Template contrastive AUC: {auc_mean:.3f} (d vs 0.5: {d_vs_chance:+.3f}, "
      f"p={p_val:.2e} {sig})")
print(f"  Relevant passage template uplift:   {strategy_d['delta_rel'].mean():+.4f}")
print(f"  Irrelevant passage template uplift: {strategy_d['delta_irrel'].mean():+.4f}")
print(f"  Differential (rel - irrel):          "
      f"{(strategy_d['delta_rel'] - strategy_d['delta_irrel']).mean():+.4f}")

diff_rel_irrel = strategy_d['delta_rel'] - strategy_d['delta_irrel']
d_diff = cohens_d(diff_rel_irrel)
_, p_diff = stats.ttest_1samp(diff_rel_irrel, 0)
sig_diff = ('***' if p_diff < 0.001 else '**' if p_diff < 0.01 else
            '*' if p_diff < 0.05 else 'ns')
print(f"  Differential d={d_diff:+.3f}, p={p_diff:.2e} {sig_diff}")

if auc_mean > 0.52 and p_val < 0.05:
    print(f"\n  >>> TEMPLATE CONTRASTIVE detects ranking signal!")
    print(f"      Per-passage keywords DO create differential benefit.")
else:
    print(f"\n  >>> No ranking signal from per-doc template variation.")

strategy_d_summary = {
    'auc': float(auc_mean), 'd_vs_chance': float(d_vs_chance),
    'p': float(p_val), 'differential_d': float(d_diff),
}""")

# ============================================================
# Cell 11: Strategy E -- Oracle cross-contrastive (KEY)
# ============================================================
code(r"""# Cell 11: Strategy E -- Oracle cross-contrastive ranking (KEY NOVEL ANALYSIS)
print("=" * 70)
print("STRATEGY E: ORACLE CROSS-CONTRASTIVE RANKING")
print("=" * 70)
print("Rank by NLL(oracle) - NLL(oracle_cross).")
print("Lower = oracle helps more than wrong query = more relevant.")
print("Structural benefit cancels. Residual = semantic query-doc match.\n")

strategy_e = {'auc': [], 'contrastive_rel': [], 'contrastive_irrel': [],
              'mrr3': [], 'hit1': []}

for r in results:
    rel_idx = r['relevant_idx']
    oracle_nlls_q = np.array(r['scores']['oracle_trunc'])
    cross_nlls_q = np.array(r['scores']['oracle_cross_trunc'])

    # Contrastive score = NLL(oracle) - NLL(oracle_cross)
    # More negative = oracle helps more than cross = more relevant
    contrastive = oracle_nlls_q - cross_nlls_q

    # Rank by contrastive (lower = more relevant)
    strategy_e['auc'].append(compute_auc(contrastive, rel_idx))

    # Also compute MRR@3 and Hit@1
    ranked = np.argsort(contrastive)
    for rank, idx in enumerate(ranked[:3], 1):
        if idx == rel_idx:
            strategy_e['mrr3'].append(1.0 / rank)
            break
    else:
        strategy_e['mrr3'].append(0.0)
    strategy_e['hit1'].append(1.0 if ranked[0] == rel_idx else 0.0)

    # Track contrastive values
    strategy_e['contrastive_rel'].append(contrastive[rel_idx])
    irrel_vals = [contrastive[i] for i in range(len(contrastive)) if i != rel_idx]
    strategy_e['contrastive_irrel'].append(np.mean(irrel_vals))

for k in strategy_e:
    strategy_e[k] = np.array(strategy_e[k])

auc_mean = strategy_e['auc'].mean()
mrr_mean = strategy_e['mrr3'].mean()
hit1_mean = strategy_e['hit1'].mean()
d_vs_chance = cohens_d(strategy_e['auc'] - 0.5)
_, p_val = stats.ttest_1samp(strategy_e['auc'], 0.5)
sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
       '*' if p_val < 0.05 else 'ns')

print(f"Oracle cross-contrastive ranking:")
print(f"  AUC:   {auc_mean:.3f} (d vs 0.5: {d_vs_chance:+.3f}, p={p_val:.2e} {sig})")
print(f"  MRR@3: {mrr_mean:.3f}")
print(f"  Hit@1: {hit1_mean:.3f}")

# Contrastive value analysis
print(f"\n--- Contrastive values ---")
print(f"  Relevant mean:   {strategy_e['contrastive_rel'].mean():+.4f}")
print(f"  Irrelevant mean: {strategy_e['contrastive_irrel'].mean():+.4f}")

diff_re = strategy_e['contrastive_rel'] - strategy_e['contrastive_irrel']
d_diff = cohens_d(diff_re)
_, p_diff = stats.ttest_1samp(diff_re, 0)
sig_diff = ('***' if p_diff < 0.001 else '**' if p_diff < 0.01 else
            '*' if p_diff < 0.05 else 'ns')
print(f"  Differential: {diff_re.mean():+.4f} (d={d_diff:+.3f}, p={p_diff:.2e} {sig_diff})")
print(f"  % queries where relevant more negative: "
      f"{100*np.mean(diff_re < 0):.1f}%")

# Structural benefit comparison: oracle vs oracle_cross
print(f"\n--- Structural benefit check ---")
print(f"  Oracle mean NLL drop: "
      f"{np.mean([np.mean(np.array(r['scores']['bare']) - np.array(r['scores']['oracle_trunc'])) for r in results]):+.4f}")
print(f"  Cross mean NLL drop:  "
      f"{np.mean([np.mean(np.array(r['scores']['bare']) - np.array(r['scores']['oracle_cross_trunc'])) for r in results]):+.4f}")
print(f"  (Should be similar if structural benefit is length-dependent)")

if auc_mean > 0.52 and p_val < 0.05:
    print(f"\n  >>> CROSS-CONTRASTIVE RANKING WORKS!")
    print(f"      The semantic residual CAN be isolated for ranking.")
    print(f"      This is the signal that v2 and Exp 04A could never detect.")
elif auc_mean > 0.51:
    print(f"\n  >>> MARGINAL signal: {auc_mean:.3f} (needs larger N to confirm)")
else:
    print(f"\n  >>> No cross-contrastive signal: semantic residual too small for ranking")

strategy_e_summary = {
    'auc': float(auc_mean), 'mrr3': float(mrr_mean), 'hit1': float(hit1_mean),
    'd_vs_chance': float(d_vs_chance), 'p': float(p_val),
    'contrastive_rel_mean': float(strategy_e['contrastive_rel'].mean()),
    'contrastive_irrel_mean': float(strategy_e['contrastive_irrel'].mean()),
    'differential_d': float(d_diff),
}""")

# ============================================================
# Cell 12: Strategy F -- Vocabulary overlap correlation
# ============================================================
code(r"""# Cell 12: Strategy F -- Vocabulary overlap correlation
print("=" * 70)
print("STRATEGY F: VOCABULARY OVERLAP CORRELATION")
print("=" * 70)
print("Correlate delta(bare, oracle) with Jaccard overlap between query and passage.\n")

# For each passage, compute vocab overlap and oracle delta
per_passage_data = []
for r in results:
    query_text = r['query']
    bare_nlls_q = np.array(r['scores']['bare'])
    oracle_nlls_q = np.array(r['scores']['oracle_trunc'])
    rel_idx = r['relevant_idx']

    q_idx = r['query_idx']
    for p_idx in range(r['n_passages']):
        p_text = queries[q_idx]['passages'][p_idx]['text']
        overlap = compute_vocab_overlap(query_text, p_text)
        delta = bare_nlls_q[p_idx] - oracle_nlls_q[p_idx]
        per_passage_data.append({
            'overlap': overlap,
            'delta': delta,
            'is_relevant': 1 if p_idx == rel_idx else 0,
        })

overlaps = np.array([d['overlap'] for d in per_passage_data])
deltas = np.array([d['delta'] for d in per_passage_data])
is_rel = np.array([d['is_relevant'] for d in per_passage_data])

# Overall correlation
r_all, p_all = stats.pearsonr(overlaps, deltas)
print(f"Overall correlation (all passages):")
print(f"  Pearson r(overlap, delta) = {r_all:+.3f} (p={p_all:.2e})")

# Relevant vs irrelevant
r_rel, p_rel = stats.pearsonr(overlaps[is_rel == 1], deltas[is_rel == 1])
r_irrel, p_irrel = stats.pearsonr(overlaps[is_rel == 0], deltas[is_rel == 0])
print(f"  Relevant only:   r = {r_rel:+.3f} (p={p_rel:.2e})")
print(f"  Irrelevant only: r = {r_irrel:+.3f} (p={p_irrel:.2e})")

# Overlap difference: relevant vs irrelevant
rel_overlaps = overlaps[is_rel == 1]
irrel_overlaps = overlaps[is_rel == 0]
print(f"\nVocabulary overlap stats:")
print(f"  Relevant passages:   mean={rel_overlaps.mean():.3f}")
print(f"  Irrelevant passages: mean={irrel_overlaps.mean():.3f}")

# Can overlap alone predict relevance?
overlap_auc_per_query = []
for r in results:
    q_text = r['query']
    q_idx = r['query_idx']
    rel_idx = r['relevant_idx']
    passage_overlaps = []
    for p_idx in range(r['n_passages']):
        p_text = queries[q_idx]['passages'][p_idx]['text']
        passage_overlaps.append(-compute_vocab_overlap(q_text, p_text))
    passage_overlaps = np.array(passage_overlaps)
    overlap_auc_per_query.append(compute_auc(passage_overlaps, rel_idx))

overlap_auc = np.mean(overlap_auc_per_query)
d_overlap = cohens_d(np.array(overlap_auc_per_query) - 0.5)
print(f"\nOverlap-based ranking AUC: {overlap_auc:.3f} (d vs 0.5: {d_overlap:+.3f})")

if abs(r_all) > 0.1 and p_all < 0.05:
    print(f"\n  >>> Vocabulary overlap DOES predict oracle benefit.")
    print(f"      Higher overlap passages get more help from the oracle query.")
else:
    print(f"\n  >>> No correlation: oracle benefit is independent of vocab overlap.")
    print(f"      Consistent with ~85% structural mechanism.")

strategy_f_summary = {
    'r_overall': float(r_all), 'p_overall': float(p_all),
    'r_relevant': float(r_rel), 'r_irrelevant': float(r_irrel),
    'overlap_ranking_auc': float(overlap_auc),
}""")

# ============================================================
# Cell 13: Summary table
# ============================================================
code(r"""# Cell 13: Summary table comparing all 6 strategies
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("SUMMARY: AUC ACROSS ALL 6 STRATEGIES")
print("=" * 70)

# Collect best AUC per strategy
summary = {
    'A: Standard (raw NLL)': {
        'best_cond': max([(c, strategy_a[c]['auc']) for c in COND_NAMES[1:]],
                         key=lambda x: x[1]),
        'oracle_auc': strategy_a.get('oracle_trunc', {}).get('auc', 0.5),
    },
    'B: Delta-NLL': {
        'best_cond': max([(c, strategy_b[c]['auc']) for c in strategy_b],
                         key=lambda x: x[1]),
        'oracle_auc': strategy_b.get('oracle_trunc', {}).get('auc', 0.5),
    },
    'C: Relative delta': {
        'best_cond': max([(c, strategy_c[c]['auc']) for c in strategy_c],
                         key=lambda x: x[1]),
        'oracle_auc': strategy_c.get('oracle_trunc', {}).get('auc', 0.5),
    },
    'D: Template contrastive': {
        'best_cond': ('template-random', strategy_d_summary['auc']),
        'oracle_auc': strategy_d_summary['auc'],
    },
    'E: Oracle cross-contrastive': {
        'best_cond': ('oracle-cross', strategy_e_summary['auc']),
        'oracle_auc': strategy_e_summary['auc'],
    },
    'F: Vocab overlap': {
        'best_cond': ('overlap', strategy_f_summary['overlap_ranking_auc']),
        'oracle_auc': strategy_f_summary['overlap_ranking_auc'],
    },
}

print(f"\n{'Strategy':<32} {'Best AUC':>10} {'Best condition':>25} {'vs 0.5':>8}")
print("-" * 80)

strategy_aucs = []
strategy_names = []
for name, data in summary.items():
    best_cond, best_auc = data['best_cond']
    delta = best_auc - 0.5
    print(f"  {name:<30} {best_auc:>10.3f} {best_cond:>25} {delta:>+8.3f}")
    strategy_aucs.append(best_auc)
    strategy_names.append(name.split(':')[0])

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(strategy_aucs))
colors = ['steelblue' if a > 0.52 else 'gray' if a > 0.5 else 'lightcoral'
          for a in strategy_aucs]
ax.bar(x, strategy_aucs, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
ax.set_ylabel('AUC')
ax.set_title('Ranking AUC Across All 6 Strategies')
ax.set_xticks(x)
ax.set_xticklabels(strategy_names, rotation=30, ha='right')
ax.legend()
ax.set_ylim(0.45, max(strategy_aucs) + 0.05)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = RESULTS_DIR / 'strategy_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")

# Best strategy?
best_idx = np.argmax(strategy_aucs)
best_name = list(summary.keys())[best_idx]
best_auc = strategy_aucs[best_idx]
print(f"\nBest strategy: {best_name} (AUC={best_auc:.3f})")
if best_auc > 0.52:
    print(f"  >>> Ranking signal detected! Best approach: {best_name}")
else:
    print(f"  >>> No strategy achieves meaningful ranking. Confirmed: ranking is dead.")""")

# ============================================================
# Cell 14: Verdict + save + cleanup
# ============================================================
code(r"""# Cell 14: Verdict + save + cleanup
print("=" * 70)
print("VERDICT -- Exp 08: Contrastive & Differential Ranking")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N queries: {N_SAMPLES}")
print(f"Mean passages per query: {np.mean([r['n_passages'] for r in results]):.1f}")

# Key results by strategy
print(f"\n--- Strategy results ---")
print(f"  A (standard):          oracle AUC = {strategy_a.get('oracle_trunc', {}).get('auc', 0):.3f}")
print(f"  B (delta-NLL):         oracle AUC = {strategy_b.get('oracle_trunc', {}).get('auc', 0):.3f}")
print(f"  C (relative delta):    oracle AUC = {strategy_c.get('oracle_trunc', {}).get('auc', 0):.3f}")
print(f"  D (template contrast): AUC = {strategy_d_summary['auc']:.3f}")
print(f"  E (cross-contrastive): AUC = {strategy_e_summary['auc']:.3f} *** KEY ***")
print(f"  F (vocab overlap):     AUC = {strategy_f_summary['overlap_ranking_auc']:.3f}")

# Strategy E deep dive
print(f"\n--- Strategy E: Cross-contrastive deep dive ---")
print(f"  Contrastive = NLL(oracle) - NLL(oracle_cross)")
print(f"  Relevant:   {strategy_e_summary['contrastive_rel_mean']:+.4f}")
print(f"  Irrelevant: {strategy_e_summary['contrastive_irrel_mean']:+.4f}")
print(f"  Differential d: {strategy_e_summary['differential_d']:+.3f}")

# Overall verdict
print(f"\n--- OVERALL VERDICT ---")
any_signal = any(a > 0.52 for a in strategy_aucs)
cross_works = strategy_e_summary['auc'] > 0.52 and strategy_e_summary['p'] < 0.05

if cross_works:
    print(f"  CROSS-CONTRASTIVE RANKING WORKS (AUC={strategy_e_summary['auc']:.3f})")
    print(f"  The semantic residual (~15%) CAN be isolated by subtracting structural")
    print(f"  benefit using a wrong query. This is the first positive ranking result")
    print(f"  across 8 experiments (v2) and 3 experiments (v3).")
elif any_signal:
    best_idx = np.argmax(strategy_aucs)
    best_name = list(summary.keys())[best_idx]
    print(f"  MARGINAL signal via {best_name} (AUC={strategy_aucs[best_idx]:.3f})")
    print(f"  Cross-contrastive did NOT achieve clear signal.")
else:
    print(f"  NO strategy produces ranking signal.")
    print(f"  Confirmed: the structural mechanism is document-independent,")
    print(f"  and the semantic residual is too small or noisy for ranking.")
    print(f"  This closes the ranking investigation definitively.")

# Exp 04A comparison
print(f"\n--- Exp 04A comparison ---")
print(f"  04A oracle AUC: ~0.853 (Strategy A only)")
print(f"  08 oracle AUC:  {strategy_a.get('oracle_trunc', {}).get('auc', 0):.3f} (Strategy A)")
print(f"  Replication: {'YES' if abs(strategy_a.get('oracle_trunc', {}).get('auc', 0) - 0.853) < 0.02 else 'CLOSE' if abs(strategy_a.get('oracle_trunc', {}).get('auc', 0) - 0.853) < 0.05 else 'DIFFERENT'}")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp08_contrastive_ranking',
    'model': MODEL_NAME,
    'n_queries': N_SAMPLES,
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'strategy_a': strategy_a,
    'strategy_b': strategy_b,
    'strategy_c': strategy_c,
    'strategy_d': strategy_d_summary,
    'strategy_e': strategy_e_summary,
    'strategy_f': strategy_f_summary,
    'pool_stats': {
        'mean_passages_per_query': float(np.mean([r['n_passages'] for r in results])),
        'total_passages': int(sum(r['n_passages'] for r in results)),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
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

outpath = "experiments/08/08_contrastive_ranking.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
