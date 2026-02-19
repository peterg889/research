#!/usr/bin/env python3
"""Generate Exp 04B notebook: Amazon ESCI Query-Likelihood Ranking.

Part B of the ranking experiments. Key differences from Part A (MS MARCO):
1. Query-likelihood scoring: NLL(query | encode([condition + product_text]))
   No gold answer text -- the decoder scores the query itself.
2. Graded relevance: E=3, S=2, C=1, I=0 -> NDCG metrics.
3. New surrogate: surr_title_trunc (product title as prefix -- natural for ads).
4. Pre-screen: abort if bare QL AUC < 0.55 on first 20 queries.
5. Larger pools (10-40 candidates per query) -> NDCG@5 is meaningful.

This is the real ad-serving use case: "which product best explains this query?"
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
md(r"""# Experiment 04B: Amazon ESCI Query-Likelihood Ranking
## The real ad-serving use case: which product best explains the user's query?

### Context
Exp 04A tested answer-likelihood ranking on MS MARCO (same scoring as Exps 01-03).
This experiment tests **query-likelihood** ranking on a commercial product search dataset.

### Key difference: Query-Likelihood Scoring
No gold answer text in ESCI. Instead, the decoder scores the query:

```
NLL(query | encode([condition + product_text]))
```

"Given this product representation, how likely is the user's query?"
This IS the ad-serving question.

### Dataset: Amazon ESCI
- Real product search with graded relevance: Exact (3), Substitute (2), Complement (1), Irrelevant (0)
- 130K+ queries, up to 40 candidates per query
- Product text = title + bullet\_points + description
- Truly irrelevant products (unlike MS MARCO where all candidates are topically related)

### Pre-Screen
v2 Exp 31 got QL AUC=0.578 on MS MARCO (barely above chance) because all candidates
are topically similar. ESCI has truly Irrelevant products -> expect higher AUC.
**Abort if bare QL AUC < 0.55 on first 20 queries.**

### Conditions (6)
| # | Condition | Encoder input | Note |
|---|-----------|--------------|------|
| 1 | bare | product\_text only | Lower bound |
| 2 | oracle\_trunc | real query + product\_text | Upper bound |
| 3 | surr\_title\_trunc | product\_title + product\_text | Natural for ads! |
| 4 | surr\_doc\_trunc | TF keywords from product\_text | Doc-derived |
| 5 | surr\_template\_trunc | "What is [keyword]?" + product\_text | Templated |
| 6 | random\_trunc | unrelated product text | Structural control |

### Metrics
- **NDCG@1**, **NDCG@3**, **NDCG@5** (graded: E=3, S=2, C=1, I=0)
- **AUC** (binary: E+S vs C+I)
- **MRR@3** (binary: E+S relevant)
- **Hit@1** (binary: E+S relevant)

### N=400 queries, Bonferroni=5
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

sys.path.insert(0, ".")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp04b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 400   # queries
N_PRESCREEN = 20  # queries for pre-screen
MODEL_NAME = "google/t5gemma-2-4b-4b"
N_BONFERRONI = 5  # 5 non-bare conditions

# Graded relevance mapping
ESCI_RELEVANCE = {'Exact': 3, 'Substitute': 2, 'Complement': 1, 'Irrelevant': 0}
ESCI_BINARY = {'Exact': 1, 'Substitute': 1, 'Complement': 0, 'Irrelevant': 0}

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 04B: Amazon ESCI Query-Likelihood Ranking")
print(f"Model: {MODEL_NAME}")
print(f"N queries: {N_SAMPLES} (pre-screen: {N_PRESCREEN})")
print(f"Relevance: {ESCI_RELEVANCE}")
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

def score_nll(encoder_text, target_text, prefix_token_count=0, truncate=False):
    '''Score NLL of target tokens with optional truncation.
    For ESCI: target_text is the QUERY (query-likelihood scoring).'''
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

    target_ids = tokenizer(target_text, return_tensors="pt",
                           add_special_tokens=False, truncation=True,
                           max_length=256).input_ids.to(DEVICE)

    if target_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            labels=target_ids,
        )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, target_ids[0].unsqueeze(1)).squeeze(1)
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

def make_surrogate_doc_kw(text):
    content_words = extract_keywords(text)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

def make_surrogate_template(text):
    content_words = extract_keywords(text)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"


# === Ranking metrics ===
def ndcg_at_k(relevance_scores_ranked, k):
    '''NDCG@k. relevance_scores_ranked: ALL relevance scores in predicted rank order.'''
    rel = np.asarray(relevance_scores_ranked[:k], dtype=float)
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    all_rel = np.asarray(relevance_scores_ranked, dtype=float)
    ideal = np.sort(all_rel)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_auc_binary(nlls, binary_labels):
    '''AUC for binary relevance (E+S=1 vs C+I=0). Lower NLL = more relevant.'''
    pos_nlls = [nlls[i] for i in range(len(nlls)) if binary_labels[i] == 1]
    neg_nlls = [nlls[i] for i in range(len(nlls)) if binary_labels[i] == 0]
    if len(pos_nlls) == 0 or len(neg_nlls) == 0:
        return 0.5
    wins = sum(1 for p in pos_nlls for n in neg_nlls if n > p)
    ties = sum(1 for p in pos_nlls for n in neg_nlls if n == p)
    return (wins + 0.5 * ties) / (len(pos_nlls) * len(neg_nlls))

def compute_mrr_at_k(nlls, binary_labels, k=3):
    '''MRR@k: reciprocal rank of first relevant item in top-k by ascending NLL.'''
    ranked_indices = list(np.argsort(nlls))
    for rank, idx in enumerate(ranked_indices[:k], 1):
        if binary_labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def compute_hit_at_k(nlls, binary_labels, k=1):
    '''Hit@k: 1 if any relevant item in top-k by ascending NLL.'''
    ranked_indices = np.argsort(nlls)[:k]
    return 1.0 if any(binary_labels[idx] == 1 for idx in ranked_indices) else 0.0

print("Helpers defined.")
print("  Scoring: score_nll (query-likelihood)")
print("  Surrogates: title, doc_kw, template, random")
print("  Ranking metrics: NDCG@k, AUC, MRR@k, Hit@k")""")

# ============================================================
# Cell 5: Load ESCI data
# ============================================================
code(r"""# Cell 5: Load Amazon ESCI data and build ranking pools
from datasets import load_dataset

print("Loading Amazon ESCI dataset...")
print("  Source: tasksource/esci (pre-joined, ~2M rows)")
ds = load_dataset("tasksource/esci", split="train")
print(f"  Loaded {len(ds)} rows")
print(f"  Columns: {ds.column_names}")

# Filter to US locale and build product text
print("\nFiltering to US locale and building query pools...")

def safe_str(val):
    '''Handle None/NaN values in product fields.'''
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ('nan', 'none', ''):
        return ""
    return s

def build_product_text(row):
    '''Use pre-built product_text if available, else concatenate fields.'''
    pt = safe_str(row.get('product_text', ''))
    if pt:
        return pt
    parts = []
    for field in ['product_title', 'product_bullet_point', 'product_description']:
        val = safe_str(row.get(field, ''))
        if val:
            parts.append(val)
    return " ".join(parts)

# Group by query
query_pools = defaultdict(list)

for row in tqdm(ds, desc="Filtering"):
    if safe_str(row.get('product_locale', '')) != 'us':
        continue

    label = safe_str(row.get('esci_label', ''))
    if label not in ESCI_RELEVANCE:
        continue

    product_text = build_product_text(row)
    if len(product_text.split()) < 5:
        continue

    query = safe_str(row.get('query', ''))
    if not query:
        continue

    product_title = safe_str(row.get('product_title', ''))

    query_pools[query].append({
        'product_text': product_text,
        'product_title': product_title if product_title else product_text.split()[0],
        'label': label,
        'relevance': ESCI_RELEVANCE[label],
        'binary_label': ESCI_BINARY[label],
    })

del ds
gc.collect()

print(f"US queries with products: {len(query_pools)}")

# Filter: 4+ candidates, at least 1 relevant and 1 irrelevant
qualifying_queries = []
for query_text, products in query_pools.items():
    if len(products) < 4:
        continue
    has_relevant = any(p['binary_label'] == 1 for p in products)
    has_irrelevant = any(p['binary_label'] == 0 for p in products)
    if not (has_relevant and has_irrelevant):
        continue
    qualifying_queries.append({
        'query': query_text,
        'products': products,
        'n_products': len(products),
    })

del query_pools
gc.collect()

print(f"Qualifying queries (4+ candidates, rel + irrel): {len(qualifying_queries)}")

if len(qualifying_queries) < N_SAMPLES + 50:
    print(f"WARNING: Only {len(qualifying_queries)} qualifying queries (need {N_SAMPLES}+50)")
    print(f"Reducing N_SAMPLES to {max(0, len(qualifying_queries) - 50)}")
    N_SAMPLES = max(0, len(qualifying_queries) - 50)

# Shuffle and select
np.random.seed(SEED)
np.random.shuffle(qualifying_queries)

# We need N_SAMPLES + some buffer; keep extra for pre-screen
queries = qualifying_queries[:N_SAMPLES + 50]
del qualifying_queries
gc.collect()

# Generate surrogates for each product in each query
for i, q in enumerate(queries):
    for p in q['products']:
        p['surr_doc_kw'] = make_surrogate_doc_kw(p['product_text'])
        p['surr_template'] = make_surrogate_template(p['product_text'])

    # Random: use product text from another query (circular offset)
    other_idx = (i + len(queries) // 2) % len(queries)
    other_product = queries[other_idx]['products'][0]
    q['surr_random'] = " ".join(other_product['product_text'].split()[:20])

# Stats
n_products = [q['n_products'] for q in queries[:N_SAMPLES]]
print(f"\nSelected {min(len(queries), N_SAMPLES)} queries")
print(f"Products per query: mean={np.mean(n_products):.1f}, "
      f"median={np.median(n_products):.0f}, "
      f"min={np.min(n_products)}, max={np.max(n_products)}")

# Label distribution
all_labels = [p['label'] for q in queries[:N_SAMPLES] for p in q['products']]
for label in ['Exact', 'Substitute', 'Complement', 'Irrelevant']:
    count = sum(1 for l in all_labels if l == label)
    print(f"  {label}: {count} ({100*count/len(all_labels):.1f}%)")

total_calls = sum(n_products) * 6  # 6 conditions
print(f"Total scoring calls: {total_calls}")
print(f"Estimated runtime: ~{total_calls * 0.5 / 3600:.1f} hours")""")

# ============================================================
# Cell 6: Pre-screen
# ============================================================
code(r"""# Cell 6: Pre-screen -- bare QL AUC on first 20 queries
print("=" * 70)
print("PRE-SCREEN: Bare Query-Likelihood AUC")
print("=" * 70)
print(f"Scoring {N_PRESCREEN} queries with bare condition only...")
print(f"v2 Exp 31 got QL AUC=0.578 on MS MARCO (near chance)")
print(f"ESCI has truly Irrelevant products -> expect higher AUC")
print(f"Abort threshold: AUC < 0.55")

prescreen_aucs = []
t0 = time.time()

for q_idx in tqdm(range(N_PRESCREEN), desc="Pre-screen"):
    q = queries[q_idx]
    query_text = q['query']
    nlls = []

    for p in q['products']:
        nll = score_nll(p['product_text'], query_text, 0, False)
        nlls.append(nll)

    binary_labels = [p['binary_label'] for p in q['products']]
    auc = compute_auc_binary(nlls, binary_labels)
    prescreen_aucs.append(auc)

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
mean_auc = np.mean(prescreen_aucs)
print(f"\nPre-screen complete: {elapsed/60:.1f} min")
print(f"Bare QL AUC: mean={mean_auc:.3f}, median={np.median(prescreen_aucs):.3f}")
print(f"Range: [{np.min(prescreen_aucs):.3f}, {np.max(prescreen_aucs):.3f}]")

PRESCREEN_PASSED = mean_auc >= 0.55
if PRESCREEN_PASSED:
    print(f"\n>>> PRE-SCREEN PASSED (AUC={mean_auc:.3f} >= 0.55)")
    print(f">>> Proceeding with full experiment")
else:
    print(f"\n>>> PRE-SCREEN FAILED (AUC={mean_auc:.3f} < 0.55)")
    print(f">>> QL scoring cannot distinguish relevant from irrelevant on ESCI")
    print(f">>> ABORTING -- no point running 400 queries")""")

# ============================================================
# Cell 7: Explain conditions
# ============================================================
code(r"""# Cell 7: Explain conditions with concrete example
if not PRESCREEN_PASSED:
    print("SKIPPED (pre-screen failed)")
else:
    print("=" * 70)
    print("CONDITION EXAMPLES")
    print("=" * 70)

    COND_NAMES = ['bare', 'oracle_trunc', 'surr_title_trunc',
                  'surr_doc_trunc', 'surr_template_trunc', 'random_trunc']

    ex = queries[0]
    print(f"\nQuery: {ex['query']}")
    print(f"Products: {ex['n_products']}")

    # Show one relevant and one irrelevant product
    rel_p = next(p for p in ex['products'] if p['binary_label'] == 1)
    irr_p = next(p for p in ex['products'] if p['binary_label'] == 0)

    print(f"\n--- Relevant product (label={rel_p['label']}) ---")
    print(f"  Title: {rel_p['product_title'][:100]}")
    print(f"  Text: {rel_p['product_text'][:150]}...")

    print(f"\n--- Irrelevant product (label={irr_p['label']}) ---")
    print(f"  Title: {irr_p['product_title'][:100]}")
    print(f"  Text: {irr_p['product_text'][:150]}...")

    print(f"\n--- Encoder input for relevant product ---")
    for cond in COND_NAMES:
        if cond == 'bare':
            enc = rel_p['product_text'][:80] + "..."
        elif cond == 'oracle_trunc':
            enc = ex['query'] + " | " + rel_p['product_text'][:60] + "..."
        elif cond == 'surr_title_trunc':
            enc = rel_p['product_title'][:40] + " | " + rel_p['product_text'][:40] + "..."
        elif cond == 'surr_doc_trunc':
            enc = rel_p['surr_doc_kw'] + " | " + rel_p['product_text'][:50] + "..."
        elif cond == 'surr_template_trunc':
            enc = rel_p['surr_template'] + " | " + rel_p['product_text'][:50] + "..."
        elif cond == 'random_trunc':
            enc = ex['surr_random'][:40] + "... | " + rel_p['product_text'][:40] + "..."
        print(f"  {cond:<22s}: {enc}")

    print(f"\n  Decoder target: '{ex['query']}' (query-likelihood)")
    print(f"\n--- Note: surr_title_trunc uses the product's OWN title as prefix ---")
    print(f"  This is the most natural ad-serving surrogate (title always available)")""")

# ============================================================
# Cell 8: Run scoring
# ============================================================
code(r"""# Cell 8: Run scoring -- outer loop over queries
if not PRESCREEN_PASSED:
    print("SKIPPED (pre-screen failed)")
    results = []
else:
    print("=" * 70)
    print("RUNNING RANKING EXPERIMENT")
    print("=" * 70)

    def build_condition_input(cond_name, product_data, query_data):
        '''Return (encoder_text, prefix_token_count, truncate) for a condition.'''
        product_text = product_data['product_text']

        if cond_name == 'bare':
            return product_text, 0, False
        elif cond_name == 'oracle_trunc':
            surr = query_data['query']
        elif cond_name == 'surr_title_trunc':
            surr = product_data['product_title']
        elif cond_name == 'surr_doc_trunc':
            surr = product_data['surr_doc_kw']
        elif cond_name == 'surr_template_trunc':
            surr = product_data['surr_template']
        elif cond_name == 'random_trunc':
            surr = query_data['surr_random']
        else:
            raise ValueError(f"Unknown condition: {cond_name}")

        enc_text = surr + "\n" + product_text
        prefix_count = count_prefix_tokens(surr, product_text)
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
        query_text = q['query']

        query_result = {
            'query_idx': q_idx,
            'query': query_text,
            'n_products': q['n_products'],
            'labels': [p['label'] for p in q['products']],
            'relevances': [p['relevance'] for p in q['products']],
            'binary_labels': [p['binary_label'] for p in q['products']],
            'scores': {},
        }

        for cond_name in COND_NAMES:
            cond_nlls = []
            for p_idx, product_data in enumerate(q['products']):
                enc_text, prefix_count, truncate = build_condition_input(
                    cond_name, product_data, q)
                nll = score_nll(enc_text, query_text, prefix_count, truncate)
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

    # Quick peek
    bare_aucs = []
    for r in results:
        nlls = np.array(r['scores']['bare'])
        bare_aucs.append(compute_auc_binary(nlls, r['binary_labels']))
    print(f"Bare QL AUC: mean={np.mean(bare_aucs):.3f}")""")

# ============================================================
# Cell 9: Compute ranking metrics
# ============================================================
code(r"""# Cell 9: Compute ranking metrics
if not PRESCREEN_PASSED or len(results) == 0:
    print("SKIPPED (pre-screen failed or no results)")
    metrics = {}
else:
    print("=" * 70)
    print("COMPUTING RANKING METRICS")
    print("=" * 70)

    # For each query x condition: NDCG@1/3/5, AUC, MRR@3, Hit@1
    metrics = {cond: {'ndcg1': [], 'ndcg3': [], 'ndcg5': [],
                      'auc': [], 'mrr3': [], 'hit1': []}
               for cond in COND_NAMES}

    for r in results:
        binary_labels = r['binary_labels']
        relevances = r['relevances']

        for cond in COND_NAMES:
            nlls = np.array(r['scores'][cond])
            ranked_indices = np.argsort(nlls)  # ascending NLL = best first
            ranked_relevances = [relevances[i] for i in ranked_indices]

            metrics[cond]['ndcg1'].append(ndcg_at_k(ranked_relevances, k=1))
            metrics[cond]['ndcg3'].append(ndcg_at_k(ranked_relevances, k=3))
            metrics[cond]['ndcg5'].append(ndcg_at_k(ranked_relevances, k=5))
            metrics[cond]['auc'].append(compute_auc_binary(nlls, binary_labels))
            metrics[cond]['mrr3'].append(compute_mrr_at_k(nlls, binary_labels, k=3))
            metrics[cond]['hit1'].append(compute_hit_at_k(nlls, binary_labels, k=1))

    # Convert to arrays
    for cond in COND_NAMES:
        for m in metrics[cond]:
            metrics[cond][m] = np.array(metrics[cond][m])

    # Quick summary
    for cond in COND_NAMES:
        print(f"  {cond:<22s}: AUC={metrics[cond]['auc'].mean():.3f}  "
              f"NDCG@3={metrics[cond]['ndcg3'].mean():.3f}  "
              f"MRR@3={metrics[cond]['mrr3'].mean():.3f}  "
              f"Hit@1={metrics[cond]['hit1'].mean():.3f}")""")

# ============================================================
# Cell 10: Results table
# ============================================================
code(r"""# Cell 10: Results table with statistical tests
if not PRESCREEN_PASSED or len(metrics) == 0:
    print("SKIPPED (pre-screen failed)")
    analysis = {}
else:
    from lib.analysis import cohens_d

    print("=" * 70)
    print("RESULTS: ESCI Query-Likelihood Ranking (N=%d queries)" % len(results))
    print("=" * 70)

    METRIC_NAMES = ['ndcg1', 'ndcg3', 'ndcg5', 'auc', 'mrr3', 'hit1']
    METRIC_LABELS = {'ndcg1': 'NDCG@1', 'ndcg3': 'NDCG@3', 'ndcg5': 'NDCG@5',
                     'auc': 'AUC', 'mrr3': 'MRR@3', 'hit1': 'Hit@1'}

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

                print(f"  {cond:<22} {mean_val:>8.3f} {diff.mean():>+10.4f} "
                      f"{d:>+8.3f} {p_val:>12.2e} {sig:>5}")
                analysis[metric_name][cond] = {
                    'mean': float(mean_val), 'delta': float(diff.mean()),
                    'd': float(d), 'p': float(p_val),
                }

    # Headline
    oracle_auc = analysis['auc']['oracle_trunc']['mean']
    bare_auc = analysis['auc']['bare']['mean']
    title_auc = analysis['auc'].get('surr_title_trunc', {}).get('mean', 0)
    print(f"\n{'='*70}")
    print(f"HEADLINE:")
    print(f"  bare AUC = {bare_auc:.3f}")
    print(f"  oracle AUC = {oracle_auc:.3f} (gain = {oracle_auc - bare_auc:+.3f})")
    print(f"  surr_title AUC = {title_auc:.3f} (gain = {title_auc - bare_auc:+.3f})")
    print(f"{'='*70}")""")

# ============================================================
# Cell 11: Differential analysis
# ============================================================
code(r"""# Cell 11: Differential signal analysis
if not PRESCREEN_PASSED or len(results) == 0:
    print("SKIPPED (pre-screen failed)")
    diff_analysis = {}
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lib.analysis import cohens_d

    print("=" * 70)
    print("DIFFERENTIAL SIGNAL ANALYSIS")
    print("=" * 70)
    print("Does priming help RELEVANT products MORE than IRRELEVANT ones?")

    diff_analysis = {}
    for cond in COND_NAMES[1:]:
        delta_rels = []
        delta_irrels = []

        for r in results:
            bare_nlls = np.array(r['scores']['bare'])
            cond_nlls = np.array(r['scores'][cond])
            binary = r['binary_labels']

            rel_mask = [i for i, b in enumerate(binary) if b == 1]
            irrel_mask = [i for i, b in enumerate(binary) if b == 0]

            if len(rel_mask) == 0 or len(irrel_mask) == 0:
                continue

            delta_rel = np.mean([bare_nlls[i] - cond_nlls[i] for i in rel_mask])
            delta_irrel = np.mean([bare_nlls[i] - cond_nlls[i] for i in irrel_mask])

            delta_rels.append(delta_rel)
            delta_irrels.append(delta_irrel)

        delta_rels = np.array(delta_rels)
        delta_irrels = np.array(delta_irrels)
        differential = delta_rels - delta_irrels

        d = cohens_d(differential)
        nonzero = differential[differential != 0]
        if len(nonzero) >= 10:
            try:
                stat, p_val = wilcoxon(nonzero)
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

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, cond in enumerate(['oracle_trunc', 'surr_title_trunc', 'random_trunc']):
        ax = axes[ax_idx]
        delta_rels_plot = []
        delta_irrels_plot = []
        for r in results:
            bare_nlls = np.array(r['scores']['bare'])
            cond_nlls = np.array(r['scores'][cond])
            binary = r['binary_labels']
            rel_mask = [i for i, b in enumerate(binary) if b == 1]
            irrel_mask = [i for i, b in enumerate(binary) if b == 0]
            if len(rel_mask) == 0 or len(irrel_mask) == 0:
                continue
            delta_rels_plot.append(np.mean([bare_nlls[i] - cond_nlls[i] for i in rel_mask]))
            delta_irrels_plot.append(np.mean([bare_nlls[i] - cond_nlls[i] for i in irrel_mask]))

        ax.scatter(delta_irrels_plot, delta_rels_plot, alpha=0.3, s=10)
        lims = [min(min(delta_irrels_plot), min(delta_rels_plot)) - 0.5,
                max(max(delta_irrels_plot), max(delta_rels_plot)) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='equal help')
        ax.set_xlabel('delta_irrelevant')
        ax.set_ylabel('delta_relevant')
        ax.set_title(f'{cond.replace("_trunc", "")}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ESCI Differential Signal: Points ABOVE red line = ranking improves', fontsize=12)
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'differential_signal.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to {plot_path}")""")

# ============================================================
# Cell 12: Hardness stratification
# ============================================================
code(r"""# Cell 12: Hardness stratification
if not PRESCREEN_PASSED or len(metrics) == 0:
    print("SKIPPED (pre-screen failed)")
    hardness_analysis = {}
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lib.analysis import cohens_d

    print("=" * 70)
    print("HARDNESS STRATIFICATION (by bare AUC quintiles)")
    print("=" * 70)

    bare_aucs = metrics['bare']['auc']
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
        print(f"  {quintile_labels[q]}: N={mask.sum()}, mean bare AUC={bare_aucs[mask].mean():.3f}")

    # Per-quintile NDCG@3 and AUC gains
    hardness_analysis = {}
    print(f"\n--- AUC gain by hardness quintile ---")
    header = f"  {'Quintile':<16}"
    for cond in COND_NAMES[1:]:
        short = cond.replace('_trunc', '')
        header += f" {short:>14}"
    print(header)
    print(f"  {'-'*(16 + 15 * len(COND_NAMES[1:]))}")

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

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ['oracle_trunc', 'surr_title_trunc', 'random_trunc']:
        gains = [hardness_analysis[quintile_labels[q]].get(cond, {}).get('auc_gain', 0)
                 for q in range(5)]
        ax.plot(range(5), gains, '-o', label=cond.replace('_trunc', ''), markersize=8)

    ax.set_xticks(range(5))
    ax.set_xticklabels(quintile_labels, rotation=15)
    ax.set_ylabel('AUC gain vs bare')
    ax.set_title('ESCI Ranking Gain by Query Hardness')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'hardness_stratification.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {plot_path}")""")

# ============================================================
# Cell 13: Verdict + save
# ============================================================
code(r"""# Cell 13: Verdict and save results
print("=" * 70)
print("VERDICT -- Exp 04B: Amazon ESCI Query-Likelihood Ranking")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N queries: {len(results) if results else 0}")
print(f"Pre-screen: {'PASSED' if PRESCREEN_PASSED else 'FAILED'}")
print(f"Pre-screen bare AUC: {np.mean(prescreen_aucs):.3f}")

if not PRESCREEN_PASSED:
    print(f"\n--- EXPERIMENT ABORTED ---")
    print(f"Query-likelihood scoring cannot discriminate relevant from irrelevant")
    print(f"products on ESCI with bare encoding.")
    print(f"This matches v2 Exp 31 finding (QL AUC near chance on MS MARCO).")

    final_results = {
        'experiment': 'exp04b_esci_ranking',
        'model': MODEL_NAME,
        'status': 'ABORTED_PRESCREEN',
        'prescreen_bare_auc': float(np.mean(prescreen_aucs)),
        'prescreen_n': N_PRESCREEN,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
else:
    # Full results
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

    # Differential verdict
    print(f"\n--- Differential signal ---")
    for cond in COND_NAMES[1:]:
        da = diff_analysis.get(cond, {})
        diff_mean = da.get('differential_mean', 0)
        d = da.get('d', 0)
        p = da.get('p_onesided', 1)
        pct = da.get('pct_positive', 0)
        sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
               else '*' if p < 0.05/N_BONFERRONI else 'ns')
        print(f"  {cond:<22s}: diff={diff_mean:+.4f} d={d:+.3f} {pct:.0f}% positive {sig}")

    # surr_title spotlight
    print(f"\n--- surr_title_trunc (the ad-serving surrogate) ---")
    title_auc = analysis['auc'].get('surr_title_trunc', {}).get('mean', 0)
    bare_auc = analysis['auc']['bare']['mean']
    oracle_auc = analysis['auc']['oracle_trunc']['mean']
    if oracle_auc > bare_auc:
        title_ratio = (title_auc - bare_auc) / (oracle_auc - bare_auc) * 100
    else:
        title_ratio = 0
    print(f"  AUC: {title_auc:.3f} ({title_ratio:.0f}% of oracle gain)")
    title_ndcg3 = analysis['ndcg3'].get('surr_title_trunc', {}).get('mean', 0)
    bare_ndcg3 = analysis['ndcg3']['bare']['mean']
    oracle_ndcg3 = analysis['ndcg3']['oracle_trunc']['mean']
    print(f"  NDCG@3: {title_ndcg3:.3f} (bare={bare_ndcg3:.3f}, oracle={oracle_ndcg3:.3f})")

    # Overall verdict
    print(f"\n--- OVERALL VERDICT ---")
    oracle_auc_p = analysis['auc'].get('oracle_trunc', {}).get('p', 1)
    oracle_auc_d = analysis['auc'].get('oracle_trunc', {}).get('d', 0)
    if oracle_auc_p < 0.05/N_BONFERRONI and oracle_auc_d > 0:
        print(f"  RANKING SIGNAL DETECTED on ESCI")
        title_p = analysis['auc'].get('surr_title_trunc', {}).get('p', 1)
        if title_p < 0.05/N_BONFERRONI:
            print(f"  surr_title_trunc also significant -- product title is a viable surrogate!")
            print(f"  >>> This validates the ad-serving use case")
        else:
            print(f"  surr_title_trunc is NS -- need oracle query for ranking benefit")
    else:
        print(f"  NO ranking signal on ESCI (same outcome as v2)")
        print(f"  Query-likelihood may not be the right scoring approach")

    final_results = {
        'experiment': 'exp04b_esci_ranking',
        'model': MODEL_NAME,
        'status': 'COMPLETED',
        'n_queries': len(results),
        'n_bonferroni': N_BONFERRONI,
        'prescreen_bare_auc': float(np.mean(prescreen_aucs)),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': analysis,
        'diff_analysis': diff_analysis,
        'hardness_analysis': hardness_analysis,
        'pool_stats': {
            'mean_products_per_query': float(np.mean([r['n_products'] for r in results])),
            'total_products': int(sum(r['n_products'] for r in results)),
        },
    }

print(f"\n{'='*70}")

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
# Cell 14: Cleanup
# ============================================================
code("""# Cell 14: Cleanup
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

outpath = "04b_esci_ranking.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
