#!/usr/bin/env python3
# Build Prefix LM Exp 04 notebook: Causal Ordering Test.
#
# The simplest possible test of whether token ORDER matters in causal LMs.
# Single forward pass, standard causal attention, no custom masks, no two-phase.
#
# Three conditions (same answer tokens, same total information, different order):
#   doc_only:  [BOS, doc, answer]          -- no query at all (baseline)
#   doc_query: [BOS, doc, query, answer]   -- standard reading order
#   query_doc: [BOS, query, doc, answer]   -- reversed: doc is "enriched" by query
#
# In causal attention:
#   doc_query: doc reps are "pure" (no query context), query reps see doc
#   query_doc: doc reps are "enriched" (attend to query), query reps are "pure"
#
# Both give answer tokens the same information (full causal access to both).
# The ONLY difference is internal representation quality.
#
# N=500 MS MARCO samples, SEED=42.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04: Causal Ordering Test

## Motivation

Exp 01-03 showed: surrogate enrichment works (d~+0.45) but structural fraction is 105% --
random tokens help as much as oracle. The semantic signal is small (d=+0.255) and
instruction content barely matters (Exp 03).

**Fundamental question**: Does token ORDER have any effect at all in causal LMs?

## Design

The simplest possible test. Single forward pass, native causal attention, no custom masks.

| # | Condition | Input sequence | What doc sees | What query sees |
|---|-----------|---------------|---------------|-----------------|
| 1 | `doc_only` | `[BOS, doc, answer]` | preceding doc | n/a |
| 2 | `doc_query` | `[BOS, doc, query, answer]` | preceding doc | doc |
| 3 | `query_doc` | `[BOS, query, doc, answer]` | **query + preceding doc** | nothing |

In all cases, answer tokens see EVERYTHING before them (full causal access to both
doc and query). The ONLY difference is the internal representations:
- `doc_query`: doc representations are "pure" (encoded without query context)
- `query_doc`: doc representations are "enriched" (each doc token attended to query)

**Key comparison**: `doc_query` vs `query_doc`
- If `query_doc` is better: enrichment is real -- doc reps conditioned on query are more useful
- If equal: order doesn't matter -- answer tokens extract info regardless
- If `doc_query` is better: standard order is better (query seeing doc > doc seeing query)

## Connection to Two-Phase Experiments

The two-phase surrogate design (Exp 01-03) is a more constrained version:
- Phase A: `[BOS, surrogate, doc]` cached → equivalent to prefix of `query_doc`
- Phase B: `[query, answer]` with surrogate truncated → answer sees doc + query but NOT prefix

This single-pass test removes all that complexity. If order doesn't matter here,
the two-phase enrichment effect must come from something else (position shifts, etc.).""")


# ===== Cell 1: Setup =====
code(r"""# Cell 1: Setup
import os
os.umask(0o000)

import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500

MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/prefix_lm_exp04")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = ["doc_only", "doc_query", "query_doc"]

print(f"Prefix LM Exp 04: Causal Ordering Test")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nConditions:")
for cn in CONDITIONS:
    print(f"  {cn}")
print(f"\nNote: Single forward pass, native causal attention, NO custom masks.")
""")


# ===== Cell 2: Load model =====
code(r"""# Cell 2: Load model + tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

print(f"transformers version: {transformers.__version__}")

print(f"Loading {MODEL_NAME}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    token=HF_TOKEN,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"Loaded: {n_params:.1f}B params, {gpu_mem:.1f} GB GPU, {time.time()-t0:.0f}s")
print(f"BOS token id: {tokenizer.bos_token_id}")
""")


# ===== Cell 3: Load data =====
code(r"""# Cell 3: Load MS MARCO data (same pipeline as Exp 01-03)
from lib.data import count_words
from datasets import load_dataset

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

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
for item in ds:
    if len(all_candidates) >= 3 * N_SAMPLES:
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
            all_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

print(f"Total candidates: {len(all_candidates)}")
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

# Query-document overlap for post-hoc analysis
for i, s in enumerate(samples):
    q_words = set(re.sub(r'[^\w\s]', '', s['query'].lower()).split()) - STOP_WORDS
    d_words = set(re.sub(r'[^\w\s]', '', s['passage'].lower()).split()) - STOP_WORDS
    union = q_words | d_words
    s['query_doc_overlap'] = len(q_words & d_words) / len(union) if len(union) > 0 else 0.0

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
""")


# ===== Cell 4: Scoring function =====
code(r"""# Cell 4: score_sample() -- single forward pass, native causal attention
#
# No custom masks, no two-phase, no truncation.
# Just build the input sequence in different orders and compute NLL on answer tokens.

def score_sample(model, tokenizer, sample, device):
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        return None

    n_a = len(answer_ids)
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)

    # Three input orderings -- answer always last
    sequences = {
        "doc_only":  [bos_id] + doc_ids + answer_ids,
        "doc_query": [bos_id] + doc_ids + query_ids + answer_ids,
        "query_doc": [bos_id] + query_ids + doc_ids + answer_ids,
    }

    result = {
        'n_doc': len(doc_ids),
        'n_query': len(query_ids),
    }

    for name, seq in sequences.items():
        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        n_before = len(seq) - n_a

        with torch.no_grad():
            out = model(input_ids=input_tensor)

        # Logit at position n_before-1 predicts first answer token
        answer_logits = out.logits[0, n_before - 1 : n_before + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        result[f'nll_{name}'] = token_nlls.mean().item()

        del out, input_tensor, answer_logits, log_probs, token_nlls

    return result


print("Scoring function defined (single-pass, 3 conditions per sample).")
""")


# ===== Cell 5: Main loop =====
code(r"""# Cell 5: Main scoring loop
from lib.data import count_words as _cw

print("=" * 70)
print("MAIN SCORING LOOP")
print("=" * 70)

CKPT_PATH = RESULTS_DIR / "checkpoint.json"

# Resume from checkpoint
all_results = []
start_idx = 0
if CKPT_PATH.exists():
    ckpt = json.loads(CKPT_PATH.read_text())
    if len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            all_results = ckpt['results']
            start_idx = len(all_results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {N_SAMPLES} samples x {len(CONDITIONS)} conditions")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    try:
        result = score_sample(model, tokenizer, s, DEVICE)
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        result = None

    if result is None:
        continue

    result['query'] = s['query'][:50]
    result['query_doc_overlap'] = s['query_doc_overlap']
    result['answer_wc'] = _cw(s['answer'])
    result['doc_wc'] = s['word_count']
    all_results.append(result)

    if (i + 1) % 25 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'model': MODEL_NAME,
            'n_total': N_SAMPLES,
            'n_conditions': len(CONDITIONS),
            'condition_names': CONDITIONS,
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 100 == 0:
        gc.collect()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {len(all_results)} samples in {elapsed/60:.1f} min")
print(f"\nQuick summary:")
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}'] for r in all_results]
    print(f"  {cn:<12} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 6: Analysis =====
code(r"""# Cell 6: Analysis
print("=" * 70)
print("RESULTS: CAUSAL ORDERING TEST")
print("=" * 70)

# Extract NLL arrays
nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# --- Mean NLL table ---
print(f"\n--- Mean NLL ({N} samples) ---\n")
print(f"  {'Condition':<12} {'Mean NLL':>10} {'Std':>8}")
print(f"  {'-'*32}")
for cn in CONDITIONS:
    print(f"  {cn:<12} {nll[cn].mean():>10.4f} {nll[cn].std():>8.4f}")

# --- Key comparisons ---
print(f"\n--- Key Comparisons ---\n")
print(f"  {'Comparison':<50} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*85}")

comparisons = [
    # THE core test: does order matter?
    ("ORDERING: query_doc vs doc_query",
     nll['doc_query'] - nll['query_doc']),

    # Query benefit in standard order
    ("QUERY BENEFIT (standard): doc_only vs doc_query",
     nll['doc_only'] - nll['doc_query']),

    # Query benefit in reversed order
    ("QUERY BENEFIT (reversed): doc_only vs query_doc",
     nll['doc_only'] - nll['query_doc']),
]

for label, diff in comparisons:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<50} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# --- Per-sample ordering effect distribution ---
ordering_effect = nll['doc_query'] - nll['query_doc']
print(f"\n--- Ordering Effect Distribution ---\n")
print(f"  Per-sample NLL(doc_query) - NLL(query_doc):")
print(f"    Mean:   {ordering_effect.mean():+.4f}")
print(f"    Std:    {ordering_effect.std():.4f}")
print(f"    Median: {np.median(ordering_effect):+.4f}")
print(f"    % where query_doc wins: {(ordering_effect > 0).mean()*100:.1f}%")
pcts = np.percentile(ordering_effect, [5, 25, 75, 95])
print(f"    5th/25th/75th/95th: {pcts[0]:+.3f} / {pcts[1]:+.3f} / {pcts[2]:+.3f} / {pcts[3]:+.3f}")

# --- Per-sample heterogeneity ---
print(f"\n--- Per-Sample Heterogeneity ---\n")

overlap = np.array([r['query_doc_overlap'] for r in all_results])
answer_wc = np.array([r['answer_wc'] for r in all_results])
doc_wc = np.array([r['doc_wc'] for r in all_results])
n_query = np.array([r['n_query'] for r in all_results])

print(f"  Correlations with ordering effect (positive = query_doc better):")
print(f"  {'Covariate':<20} {'r':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*48}")

for cov_name, cov_vals in [("query_doc_overlap", overlap), ("answer_wc", answer_wc),
                            ("doc_wc", doc_wc), ("n_query_tokens", n_query)]:
    r, p = stats.pearsonr(ordering_effect, cov_vals)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cov_name:<20} {r:>+8.3f} {p:>12.2e} {sig:>5}")

# --- Answer length subpopulation ---
print(f"\n  Answer length split:")
short = answer_wc <= 5
long = ~short
print(f"  {'Group':<15} {'N':>5} {'d_ordering':>12} {'d_q_benefit_std':>16} {'d_q_benefit_rev':>16}")
print(f"  {'-'*68}")
for label, mask in [("Short (<=5w)", short), ("Long (>5w)", long)]:
    d_ord = cohens_d(ordering_effect[mask])
    d_std = cohens_d((nll['doc_only'] - nll['doc_query'])[mask])
    d_rev = cohens_d((nll['doc_only'] - nll['query_doc'])[mask])
    print(f"  {label:<15} {mask.sum():>5} {d_ord:>+12.3f} {d_std:>+16.3f} {d_rev:>+16.3f}")

# --- Query length subpopulation ---
print(f"\n  Query length split:")
med_nq = np.median(n_query)
short_q = n_query <= med_nq
long_q = ~short_q
print(f"  {'Group':<20} {'N':>5} {'d_ordering':>12}")
print(f"  {'-'*40}")
for label, mask in [(f"Short q (<={med_nq:.0f} tok)", short_q),
                     (f"Long q (>{med_nq:.0f} tok)", long_q)]:
    d_ord = cohens_d(ordering_effect[mask])
    print(f"  {label:<20} {mask.sum():>5} {d_ord:>+12.3f}")
""")


# ===== Cell 7: Save results =====
code(r"""# Cell 7: Save results and verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 04: Causal Ordering Test")
print("=" * 70)

d_ordering = cohens_d(nll['doc_query'] - nll['query_doc'])
_, p_ordering = stats.ttest_1samp(nll['doc_query'] - nll['query_doc'], 0)
d_q_std = cohens_d(nll['doc_only'] - nll['doc_query'])
d_q_rev = cohens_d(nll['doc_only'] - nll['query_doc'])

print(f"\n  d_ordering (query_doc vs doc_query):  {d_ordering:+.3f} (p={p_ordering:.2e})")
print(f"  d_query_benefit (standard order):     {d_q_std:+.3f}")
print(f"  d_query_benefit (reversed order):     {d_q_rev:+.3f}")

print(f"\n  VERDICT:")
if p_ordering < 0.05 and d_ordering > 0:
    print(f"  ORDER MATTERS: query_doc > doc_query (d={d_ordering:+.3f}, ***).")
    print(f"  Doc representations enriched by query ARE more useful.")
    print(f"  -> Enrichment is real, even in single-pass causal LM.")
    gap = d_q_rev - d_q_std
    print(f"  -> Ordering bonus: {gap:+.3f} additional d from query-first order.")
elif p_ordering < 0.05 and d_ordering < 0:
    print(f"  ORDER MATTERS but REVERSED: doc_query > query_doc (d={d_ordering:+.3f}, ***).")
    print(f"  Standard reading order is better -- query seeing doc matters more")
    print(f"  than doc seeing query.")
    print(f"  -> The two-phase enrichment effect is NOT about doc representation quality.")
else:
    print(f"  ORDER DOES NOT MATTER (d={d_ordering:+.3f}, ns).")
    print(f"  Answer tokens extract the same information regardless of internal rep quality.")
    print(f"  -> The two-phase enrichment (d~+0.45) comes from something else entirely:")
    print(f"     position shifts, attention pattern changes, or RoPE effects.")

# Connection to Exp 01-03
print(f"\n  Connection to Exp 01-03 (two-phase enrichment d~+0.45):")
print(f"  In two-phase: Phase A caches [BOS,surr,doc], Phase B uses [query,answer].")
print(f"  Surrogate positions are MASKED from Phase B (truncation).")
print(f"  The enrichment thus operates ONLY through modified doc representations")
print(f"  (indirect channel) plus position ID shifts.")

summary = {
    'n_samples': N,
    'model': MODEL_NAME,
    'nll_doc_only': float(nll['doc_only'].mean()),
    'nll_doc_query': float(nll['doc_query'].mean()),
    'nll_query_doc': float(nll['query_doc'].mean()),
    'd_ordering': float(d_ordering),
    'd_ordering_p': float(p_ordering),
    'd_query_benefit_standard': float(d_q_std),
    'd_query_benefit_reversed': float(d_q_rev),
}

final_results = {
    'experiment': 'prefix_lm_exp04',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
    'exp01_03_references': {
        'd_causal_oracle_trunc': 0.452,
        'd_causal_random_trunc': 0.475,
        'structural_fraction': 1.051,
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04_causal_ordering.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
