#!/usr/bin/env python3
"""Generate Exp 04A notebook: MS MARCO Ranking with Surrogate Co-Encoding.

Exps 01-03B established that surrogate co-encoding genuinely improves document
representations (d~+0.35-0.45, no decay to 6144 tokens, 85% structural mechanism).
But ALL of that was single-passage NLL scoring. The critical deployment question is
RANKING: does the surrogate help identify the RIGHT document from a pool?

Ranking NEVER worked in v2 -- 6 experiments (14, 15, 22, 23, 28, 31) all failed
because decoder-only value contamination was document-independent. v3's bidirectional
encoder creates document-SPECIFIC representations, which could create differential
ranking signal.

This experiment: MS MARCO answer-likelihood ranking. Score NLL(answer | encode([cond + passage]))
for every candidate passage per query, rank by NLL, measure AUC/MRR@3/Hit@1/Hit@3.
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
md(r"""# Experiment 04A: MS MARCO Ranking with Surrogate Co-Encoding
## Does surrogate priming create differential ranking signal?

### The critical question
Exps 01-03B proved that co-encoding a surrogate with a document improves the document's
representation (measured by NLL on a gold answer). But NLL improvement alone doesn't
guarantee ranking improvement. The question:

> Does priming help the **relevant** passage MORE than the **irrelevant** passages?

If yes, NLL-based ranking improves. If no (priming helps all passages equally), the
absolute NLL drops but relative ordering doesn't change.

### v2 ranking history (all failed)
| Exp | Method | Result |
|-----|--------|--------|
| 14 | Causal priming + ranking | AUC flat |
| 15 | Values-only priming + ranking | AUC flat |
| 22 | T5Gemma answer-likelihood | AUC=0.828, priming +0.001 |
| 23 | Contrastive ranking | Failed |
| 28 | Hinge loss ranking | Failed |
| 31 | Query-likelihood ranking | QL AUC=0.578 (near chance) |

**Why v2 failed**: Decoder-only value contamination was document-INDEPENDENT. The KV cache
modification lowered NLL equally for relevant and irrelevant passages.

**Why v3 might succeed**: Bidirectional encoder creates document-SPECIFIC representations.
The surrogate tokens interact with document tokens differently depending on document content.

### Scoring
`NLL(answer | encode([condition + passage]))` -- same as Exps 01-03, applied to ALL
candidate passages per query.

### Conditions (6)
| # | Condition | Encoder input | Purpose |
|---|-----------|--------------|---------|
| 1 | bare | passage only | Lower bound |
| 2 | oracle\_trunc | real query + passage | Upper bound |
| 3 | surr\_template\_trunc | "What is [keyword]?" + passage | Best doc-derived (Exp 02) |
| 4 | surr\_doc\_trunc | TF keywords + passage | Doc-derived control |
| 5 | random\_trunc | unrelated text + passage | Structural control |
| 6 | static\_fact\_trunc | "What are the key facts?" + passage | Content-agnostic |

### Metrics
- **AUC** (binary: selected vs not-selected) per query
- **MRR@3** (reciprocal rank of relevant passage in top-3)
- **Hit@1**, **Hit@3** per query
- **Differential signal**: delta\_relevant vs delta\_irrelevant

### Statistical testing
- Wilcoxon signed-rank for paired metric differences (condition vs bare)
- Cohen's d on per-query metric differences
- Bonferroni: 5 comparisons (5 non-bare conditions)

### N=400 queries (~4000 passage scorings per condition)
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

RESULTS_DIR = Path("../../results/exp04a")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 400   # queries
MODEL_NAME = "google/t5gemma-2-4b-4b"
N_BONFERRONI = 5  # 5 non-bare conditions

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 04A: MS MARCO Ranking with Surrogate Co-Encoding")
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
    '''Score NLL of answer tokens with optional truncation.'''
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
    '''Count how many tokens the prefix occupies in the concatenated encoding.'''
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
    '''AUC when exactly one passage is relevant. Lower NLL = more relevant.'''
    rel_nll = nlls[relevant_idx]
    irrel_nlls = [nlls[i] for i in range(len(nlls)) if i != relevant_idx]
    n_irrel = len(irrel_nlls)
    if n_irrel == 0:
        return 0.5
    wins = sum(1 for nll in irrel_nlls if nll > rel_nll)
    ties = sum(1 for nll in irrel_nlls if nll == rel_nll)
    return (wins + 0.5 * ties) / n_irrel

def compute_mrr_at_k(nlls, relevant_idx, k=3):
    '''MRR@k: reciprocal rank of relevant passage in top-k by ascending NLL.'''
    ranked_indices = list(np.argsort(nlls))
    for rank, idx in enumerate(ranked_indices[:k], 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0

def compute_hit_at_k(nlls, relevant_idx, k=1):
    '''Hit@k: 1 if relevant passage is in top-k by ascending NLL.'''
    ranked_indices = set(np.argsort(nlls)[:k].tolist())
    return 1.0 if relevant_idx in ranked_indices else 0.0

print("Helpers defined.")
print("  Scoring: score_nll (answer-likelihood)")
print("  Surrogates: doc_kw, template, static_fact, random")
print("  Ranking metrics: AUC, MRR@k, Hit@k")""")

# ============================================================
# Cell 5: Load data
# ============================================================
code("""# Cell 5: Load MS MARCO ranking data
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Collect queries with full passage pools for ranking
queries = []

for item in ds:
    passages_data = item.get('passages', {})
    ptexts = passages_data.get('passage_text', [])
    is_sel = passages_data.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])

    # Get best answer
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]

    if not answer:
        continue

    # Check passage pool: all passages 30-300 words
    word_counts = [count_words(pt) for pt in ptexts]
    if not all(30 <= wc <= 300 for wc in word_counts):
        continue

    # Exactly 1 selected, 2+ non-selected
    n_selected = sum(is_sel)
    n_not_selected = len(is_sel) - n_selected
    if n_selected != 1 or n_not_selected < 2:
        continue

    # Find relevant passage index
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

# Shuffle and select N_SAMPLES
np.random.seed(SEED)
np.random.shuffle(queries)
queries = queries[:N_SAMPLES]

# Generate per-query random surrogates (from another query's relevant passage)
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
print(f"Total passage scorings per condition: {sum(n_passages_list)}")
total_calls = sum(n_passages_list) * 6  # 6 conditions
print(f"Total scoring calls: {total_calls}")
print(f"Estimated runtime: ~{total_calls * 0.4 / 3600:.1f} hours")""")

# ============================================================
# Cell 6: Explain conditions
# ============================================================
code(r"""# Cell 6: Explain conditions with concrete example
print("=" * 70)
print("CONDITION EXAMPLES")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'surr_template_trunc',
              'surr_doc_trunc', 'random_trunc', 'static_fact_trunc']

ex = queries[0]
print(f"\nQuery: {ex['query']}")
print(f"Answer: {ex['answer']}")
print(f"Passages: {ex['n_passages']} ({ex['n_passages']-1} irrelevant, 1 relevant at idx {ex['relevant_idx']})")

# Show relevant and one irrelevant passage
rel_p = ex['passages'][ex['relevant_idx']]
irr_p = ex['passages'][0 if ex['relevant_idx'] != 0 else 1]

print(f"\n--- Relevant passage (idx {ex['relevant_idx']}) ---")
print(f"  Text: {rel_p['text'][:120]}...")
print(f"  surr_doc_kw: {rel_p['surr_doc_kw']}")
print(f"  surr_template: {rel_p['surr_template']}")

irr_idx = 0 if ex['relevant_idx'] != 0 else 1
print(f"\n--- Irrelevant passage (idx {irr_idx}) ---")
print(f"  Text: {irr_p['text'][:120]}...")
print(f"  surr_doc_kw: {irr_p['surr_doc_kw']}")
print(f"  surr_template: {irr_p['surr_template']}")

print(f"\n--- What the encoder sees for the relevant passage ---")
for cond in COND_NAMES:
    if cond == 'bare':
        enc = rel_p['text'][:80] + "..."
    elif cond == 'oracle_trunc':
        enc = ex['query'] + " | " + rel_p['text'][:60] + "..."
    elif cond == 'surr_template_trunc':
        enc = rel_p['surr_template'] + " | " + rel_p['text'][:60] + "..."
    elif cond == 'surr_doc_trunc':
        enc = rel_p['surr_doc_kw'] + " | " + rel_p['text'][:60] + "..."
    elif cond == 'random_trunc':
        enc = ex['surr_random'][:40] + "... | " + rel_p['text'][:40] + "..."
    elif cond == 'static_fact_trunc':
        enc = STATIC_FACT + " | " + rel_p['text'][:60] + "..."
    print(f"  {cond:<22s}: {enc}")

print(f"\n--- Key insight ---")
print(f"  Oracle surrogate = real query (same for all passages of this query)")
print(f"  Doc-derived surrogates (template, doc_kw) are PER-PASSAGE (different content)")
print(f"  Random and static_fact are query-level (same for all passages)")
print(f"  If oracle helps relevant MORE than irrelevant -> ranking improves")""")

# ============================================================
# Cell 7: Run scoring
# ============================================================
code(r"""# Cell 7: Run scoring -- outer loop over queries
print("=" * 70)
print("RUNNING RANKING EXPERIMENT")
print("=" * 70)

def build_condition_input(cond_name, passage_data, query_data):
    '''Return (encoder_text, prefix_token_count, truncate) for a condition.'''
    passage_text = passage_data['text']

    if cond_name == 'bare':
        return passage_text, 0, False
    elif cond_name == 'oracle_trunc':
        surr = query_data['query']
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
        # Validate alignment
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
print(f"\nScoring complete: {len(results)} queries in {elapsed_total/60:.1f} min")

# Quick peek at bare AUC
bare_aucs = []
for r in results:
    nlls = np.array(r['scores']['bare'])
    bare_aucs.append(compute_auc(nlls, r['relevant_idx']))
print(f"Bare AUC: mean={np.mean(bare_aucs):.3f}, median={np.median(bare_aucs):.3f}")""")

# ============================================================
# Cell 8: Compute ranking metrics
# ============================================================
code(r"""# Cell 8: Compute ranking metrics for all conditions
print("=" * 70)
print("COMPUTING RANKING METRICS")
print("=" * 70)

# For each query x condition, compute AUC, MRR@3, Hit@1, Hit@3
metrics = {cond: {'auc': [], 'mrr3': [], 'hit1': [], 'hit3': []}
           for cond in COND_NAMES}

for r in results:
    rel_idx = r['relevant_idx']
    for cond in COND_NAMES:
        nlls = np.array(r['scores'][cond])
        metrics[cond]['auc'].append(compute_auc(nlls, rel_idx))
        metrics[cond]['mrr3'].append(compute_mrr_at_k(nlls, rel_idx, k=3))
        metrics[cond]['hit1'].append(compute_hit_at_k(nlls, rel_idx, k=1))
        metrics[cond]['hit3'].append(compute_hit_at_k(nlls, rel_idx, k=3))

# Convert to arrays
for cond in COND_NAMES:
    for m in metrics[cond]:
        metrics[cond][m] = np.array(metrics[cond][m])

# Quick summary
for cond in COND_NAMES:
    print(f"  {cond:<22s}: AUC={metrics[cond]['auc'].mean():.3f}  "
          f"MRR@3={metrics[cond]['mrr3'].mean():.3f}  "
          f"Hit@1={metrics[cond]['hit1'].mean():.3f}  "
          f"Hit@3={metrics[cond]['hit3'].mean():.3f}")""")

# ============================================================
# Cell 9: Results table with statistical tests
# ============================================================
code(r"""# Cell 9: Results table with statistical tests
from lib.analysis import cohens_d

print("=" * 70)
print("RESULTS: Ranking Metrics per Condition (N=%d queries)" % N_SAMPLES)
print("=" * 70)

METRIC_NAMES = ['auc', 'mrr3', 'hit1', 'hit3']
METRIC_LABELS = {'auc': 'AUC', 'mrr3': 'MRR@3', 'hit1': 'Hit@1', 'hit3': 'Hit@3'}

analysis = {}

for metric_name in METRIC_NAMES:
    print(f"\n--- {METRIC_LABELS[metric_name]} ---")
    print(f"  {'Condition':<22} {'Mean':>8} {'vs Bare':>10} {'d':>8} {'p':>12} {'sig':>5}")
    print(f"  {'-'*70}")

    bare_vals = metrics['bare'][metric_name]
    analysis[metric_name] = {}

    for cond in COND_NAMES:
        vals = metrics[cond][metric_name]
        mean_val = vals.mean()

        if cond == 'bare':
            print(f"  {cond:<22} {mean_val:>8.3f} {'--':>10} {'--':>8} {'--':>12} {'--':>5}")
            analysis[metric_name][cond] = {'mean': float(mean_val)}
        else:
            diff = vals - bare_vals
            d = cohens_d(diff)

            # Wilcoxon signed-rank test (non-parametric, appropriate for ranking metrics)
            nonzero = diff[diff != 0]
            if len(nonzero) >= 10:
                try:
                    stat, p_val = wilcoxon(nonzero)
                except ValueError:
                    p_val = 1.0
            else:
                p_val = 1.0

            sig = ('***' if p_val < 0.001/N_BONFERRONI else
                   '**' if p_val < 0.01/N_BONFERRONI else
                   '*' if p_val < 0.05/N_BONFERRONI else 'ns')

            print(f"  {cond:<22} {mean_val:>8.3f} {diff.mean():>+10.4f} {d:>+8.3f} {p_val:>12.2e} {sig:>5}")
            analysis[metric_name][cond] = {
                'mean': float(mean_val), 'delta': float(diff.mean()),
                'd': float(d), 'p': float(p_val),
            }

# Headline result
oracle_auc = analysis['auc']['oracle_trunc']['mean']
bare_auc = analysis['auc']['bare']['mean']
auc_gain = oracle_auc - bare_auc
print(f"\n{'='*70}")
print(f"HEADLINE: oracle_trunc AUC = {oracle_auc:.3f} vs bare AUC = {bare_auc:.3f} (gain = {auc_gain:+.3f})")
if auc_gain > 0.01:
    print(f"  >>> BREAKTHROUGH: Ranking signal detected! v2 never achieved this.")
elif auc_gain > 0.005:
    print(f"  >>> Marginal ranking signal (v2 Exp 22 got +0.001)")
else:
    print(f"  >>> No ranking signal -- same as v2")
print(f"v2 Exp 22 reference: bare AUC = 0.828, primed AUC = 0.829")
print(f"{'='*70}")""")

# ============================================================
# Cell 10: Differential signal analysis
# ============================================================
code(r"""# Cell 10: Differential signal analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("DIFFERENTIAL SIGNAL ANALYSIS")
print("=" * 70)
print("Core test: does priming help the RELEVANT passage MORE than IRRELEVANT ones?")

# Compute per-query differential: delta_rel vs mean(delta_irrel)
diff_analysis = {}
for cond in COND_NAMES[1:]:  # skip bare
    delta_rels = []
    delta_irrels = []

    for r in results:
        rel_idx = r['relevant_idx']
        bare_nlls = r['scores']['bare']
        cond_nlls = r['scores'][cond]

        # Delta for relevant passage (positive = priming helped)
        delta_rel = bare_nlls[rel_idx] - cond_nlls[rel_idx]

        # Mean delta for irrelevant passages
        irrel_deltas = [bare_nlls[i] - cond_nlls[i]
                        for i in range(len(bare_nlls)) if i != rel_idx]
        delta_irrel = np.mean(irrel_deltas)

        delta_rels.append(delta_rel)
        delta_irrels.append(delta_irrel)

    delta_rels = np.array(delta_rels)
    delta_irrels = np.array(delta_irrels)
    differential = delta_rels - delta_irrels  # positive = helps ranking

    # Test: is differential > 0?
    d = cohens_d(differential)
    nonzero = differential[differential != 0]
    if len(nonzero) >= 10:
        try:
            stat, p_val = wilcoxon(nonzero)
            # One-sided: we care about positive differential
            p_val_onesided = p_val / 2 if np.mean(differential) > 0 else 1 - p_val / 2
        except ValueError:
            p_val_onesided = 1.0
    else:
        p_val_onesided = 1.0

    diff_analysis[cond] = {
        'delta_rel_mean': float(delta_rels.mean()),
        'delta_irrel_mean': float(delta_irrels.mean()),
        'differential_mean': float(differential.mean()),
        'd': float(d),
        'p_onesided': float(p_val_onesided),
        'pct_positive': float(100 * np.mean(differential > 0)),
    }

    sig = ('***' if p_val_onesided < 0.001/N_BONFERRONI else
           '**' if p_val_onesided < 0.01/N_BONFERRONI else
           '*' if p_val_onesided < 0.05/N_BONFERRONI else 'ns')

    print(f"\n  {cond}:")
    print(f"    delta_rel  (mean NLL drop for relevant):   {delta_rels.mean():+.4f}")
    print(f"    delta_irrel (mean NLL drop for irrelevant): {delta_irrels.mean():+.4f}")
    print(f"    differential (rel - irrel):                 {differential.mean():+.4f}  "
          f"d={d:+.3f}  p={p_val_onesided:.2e}  {sig}")
    print(f"    % queries where relevant helped MORE:       {100*np.mean(differential > 0):.1f}%")

# Plot: delta_rel vs delta_irrel scatter for oracle
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_idx, cond in enumerate(['oracle_trunc', 'surr_template_trunc', 'random_trunc']):
    ax = axes[ax_idx]
    delta_rels = []
    delta_irrels = []
    for r in results:
        rel_idx = r['relevant_idx']
        bare_nlls = r['scores']['bare']
        cond_nlls = r['scores'][cond]
        delta_rels.append(bare_nlls[rel_idx] - cond_nlls[rel_idx])
        irrel_deltas = [bare_nlls[i] - cond_nlls[i]
                        for i in range(len(bare_nlls)) if i != rel_idx]
        delta_irrels.append(np.mean(irrel_deltas))

    ax.scatter(delta_irrels, delta_rels, alpha=0.3, s=10)
    lims = [min(min(delta_irrels), min(delta_rels)) - 0.5,
            max(max(delta_irrels), max(delta_rels)) + 0.5]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='equal help')
    ax.set_xlabel('delta_irrelevant (mean NLL drop)')
    ax.set_ylabel('delta_relevant (NLL drop)')
    ax.set_title(f'{cond.replace("_trunc", "")}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Differential Signal: Points ABOVE red line = ranking improves', fontsize=12)
plt.tight_layout()
plot_path = RESULTS_DIR / 'differential_signal.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {plot_path}")""")

# ============================================================
# Cell 11: Hardness stratification
# ============================================================
code(r"""# Cell 11: Hardness stratification by bare AUC
print("=" * 70)
print("HARDNESS STRATIFICATION")
print("=" * 70)
print("Split queries by bare AUC quintiles, check if priming helps more for hard queries")

bare_aucs = metrics['bare']['auc']

# Quintile boundaries
quintile_boundaries = np.percentile(bare_aucs, [20, 40, 60, 80])
quintile_labels = ['Q1 (hardest)', 'Q2', 'Q3', 'Q4', 'Q5 (easiest)']

def get_quintile(auc):
    for q, bound in enumerate(quintile_boundaries):
        if auc <= bound:
            return q
    return 4

quintile_assignments = np.array([get_quintile(a) for a in bare_aucs])

print(f"\nQuintile boundaries (bare AUC): {quintile_boundaries}")
for q in range(5):
    mask = quintile_assignments == q
    n = mask.sum()
    mean_bare_auc = bare_aucs[mask].mean()
    print(f"  {quintile_labels[q]}: N={n}, mean bare AUC={mean_bare_auc:.3f}")

# Per-quintile metric gains
print(f"\n--- AUC gain by hardness quintile ---")
header = f"  {'Quintile':<16}"
for cond in COND_NAMES[1:]:
    short = cond.replace('_trunc', '')
    header += f" {short:>14}"
print(header)
print(f"  {'-'*(16 + 15 * len(COND_NAMES[1:]))}")

hardness_analysis = {}
for q in range(5):
    mask = quintile_assignments == q
    row = f"  {quintile_labels[q]:<16}"
    hardness_analysis[quintile_labels[q]] = {}

    for cond in COND_NAMES[1:]:
        cond_aucs = metrics[cond]['auc'][mask]
        bare_q_aucs = bare_aucs[mask]
        gain = (cond_aucs - bare_q_aucs).mean()
        d = cohens_d(cond_aucs - bare_q_aucs) if mask.sum() > 1 else 0
        row += f" {gain:>+7.3f} d={d:>+5.2f}"
        hardness_analysis[quintile_labels[q]][cond] = {
            'auc_gain': float(gain), 'd': float(d)}

    print(row)

# Plot: oracle gain vs hardness quintile
fig, ax = plt.subplots(figsize=(8, 5))
for cond in ['oracle_trunc', 'surr_template_trunc', 'random_trunc']:
    gains = [hardness_analysis[quintile_labels[q]].get(cond, {}).get('auc_gain', 0)
             for q in range(5)]
    ax.plot(range(5), gains, '-o', label=cond.replace('_trunc', ''), markersize=8)

ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, rotation=15)
ax.set_ylabel('AUC gain vs bare')
ax.set_title('Ranking Gain by Query Hardness (bare AUC quintiles)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = RESULTS_DIR / 'hardness_stratification.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")""")

# ============================================================
# Cell 12: Verdict + save
# ============================================================
code(r"""# Cell 12: Verdict and save results
print("=" * 70)
print("VERDICT -- Exp 04A: MS MARCO Ranking")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N queries: {N_SAMPLES}")
print(f"Mean passages per query: {np.mean([r['n_passages'] for r in results]):.1f}")

# Key results
print(f"\n--- Ranking metrics summary ---")
for metric_name in METRIC_NAMES:
    print(f"\n  {METRIC_LABELS[metric_name]}:")
    bare_val = analysis[metric_name]['bare']['mean']
    print(f"    bare:           {bare_val:.3f}")
    for cond in COND_NAMES[1:]:
        a = analysis[metric_name].get(cond, {})
        mean = a.get('mean', 0)
        delta = a.get('delta', 0)
        d = a.get('d', 0)
        p = a.get('p', 1)
        sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
               else '*' if p < 0.05/N_BONFERRONI else 'ns')
        print(f"    {cond:<22s}: {mean:.3f} ({delta:+.4f}, d={d:+.3f}) {sig}")

# v2 comparison
print(f"\n--- v2 comparison ---")
print(f"  v2 Exp 22 (T5Gemma, same model, no truncation):")
print(f"    bare AUC = 0.828, primed AUC = 0.829 (gain = +0.001)")
print(f"  v3 Exp 04A (with truncation + surrogate co-encoding):")
oracle_auc = analysis['auc']['oracle_trunc']['mean']
bare_auc = analysis['auc']['bare']['mean']
print(f"    bare AUC = {bare_auc:.3f}, oracle AUC = {oracle_auc:.3f} (gain = {oracle_auc - bare_auc:+.3f})")

# Differential verdict
print(f"\n--- Differential signal verdict ---")
for cond in COND_NAMES[1:]:
    da = diff_analysis.get(cond, {})
    diff_mean = da.get('differential_mean', 0)
    d = da.get('d', 0)
    p = da.get('p_onesided', 1)
    pct = da.get('pct_positive', 0)
    sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
           else '*' if p < 0.05/N_BONFERRONI else 'ns')
    print(f"  {cond:<22s}: differential={diff_mean:+.4f} d={d:+.3f} {pct:.0f}% positive {sig}")

# Overall verdict
print(f"\n--- OVERALL VERDICT ---")
oracle_auc_d = analysis['auc'].get('oracle_trunc', {}).get('d', 0)
oracle_auc_p = analysis['auc'].get('oracle_trunc', {}).get('p', 1)
oracle_diff_d = diff_analysis.get('oracle_trunc', {}).get('d', 0)
oracle_diff_p = diff_analysis.get('oracle_trunc', {}).get('p_onesided', 1)

if oracle_auc_p < 0.05/N_BONFERRONI and oracle_auc_d > 0:
    print(f"  RANKING SIGNAL DETECTED (oracle AUC d={oracle_auc_d:+.3f}, p={oracle_auc_p:.2e})")
    if oracle_diff_p < 0.05/N_BONFERRONI:
        print(f"  DIFFERENTIAL CONFIRMED: priming helps relevant passages MORE (d={oracle_diff_d:+.3f})")
        print(f"  >>> This is the v3 breakthrough that v2 could never achieve")
    else:
        print(f"  BUT differential is NS -- benefit may be uniform across passages")
else:
    print(f"  NO ranking signal (oracle AUC d={oracle_auc_d:+.3f}, p={oracle_auc_p:.2e})")
    if oracle_diff_p < 0.05/N_BONFERRONI and oracle_diff_d > 0:
        print(f"  BUT differential IS significant -- effect exists but too small for AUC")
    else:
        print(f"  Consistent with v2: benefit is document-independent, no differential")

# Check surrogates
print(f"\n--- Surrogate ranking performance ---")
for cond in ['surr_template_trunc', 'surr_doc_trunc']:
    cond_auc = analysis['auc'].get(cond, {}).get('mean', 0)
    cond_p = analysis['auc'].get(cond, {}).get('p', 1)
    if cond_p < 0.05/N_BONFERRONI and cond_auc > bare_auc:
        ratio = (cond_auc - bare_auc) / (oracle_auc - bare_auc) * 100 if oracle_auc > bare_auc else 0
        print(f"  {cond}: AUC={cond_auc:.3f} ({ratio:.0f}% of oracle ranking gain)")
    else:
        print(f"  {cond}: AUC={cond_auc:.3f} (ns)")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp04a_msmarco_ranking',
    'model': MODEL_NAME,
    'n_queries': N_SAMPLES,
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'diff_analysis': diff_analysis,
    'hardness_analysis': hardness_analysis,
    'v2_comparison': {
        'exp22_bare_auc': 0.828,
        'exp22_primed_auc': 0.829,
    },
    'pool_stats': {
        'mean_passages_per_query': float(np.mean([r['n_passages'] for r in results])),
        'total_passages': int(sum(r['n_passages'] for r in results)),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
# Cell 13: Cleanup
# ============================================================
code("""# Cell 13: Cleanup
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

outpath = "experiments/04a/04a_msmarco_ranking.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
