#!/usr/bin/env python3
# Build examples notebook for Exp 01: Production-Realistic KV Cache.
# Shows actual text for each condition using real data. No GPU needed.

import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell(
    "# Experiment 01: Production-Realistic KV Cache — Condition Examples\n\n"
    "This notebook shows the actual text for each experimental condition "
    "using real data from the dataset. No GPU needed.\n\n"
    "**Key difference from v3**: The decoder now receives the query as a prefix "
    "before the answer tokens, modeling a production encoder-decoder system."
))

nb.cells.append(nbf.v4.new_code_cell(r"""import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, "../../..")
from lib.data import count_words

SEED = 42

# ---- Load MS MARCO (same reconstruction as all SEED=42 experiments) ----
from datasets import load_dataset
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
for item in ds:
    if len(all_candidates) >= 1500:
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

np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:500]]
del ds, all_candidates

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

def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"

def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

# Verify against checkpoint
ckpt_path = Path("../../../results/exp01/checkpoint.json")
if ckpt_path.exists():
    ckpt = json.loads(ckpt_path.read_text())
    results = ckpt.get('results', [])
    if results and results[0].get('query', '')[:50] == samples[0]['query'][:50]:
        print(f"Checkpoint verification: MATCH")
    elif results:
        print(f"Checkpoint verification: MISMATCH")
else:
    print("No checkpoint found yet")

print(f"Loaded {len(samples)} MS MARCO samples (SEED={SEED})")

# ---- Generate surrogates for sample 0 ----
ex = samples[0]
surr_template = make_surrogate_template(ex['passage'])
surr_doc_kw = make_surrogate_from_doc(ex['passage'])
other_idx = (0 + 250) % len(samples)
other_words = samples[other_idx]['passage'].split()
query_word_count = len(ex['query'].split())
random_prefix = " ".join(other_words[:query_word_count])
doc_short = ex['passage'][:80]

print()
print("=" * 80)
print("SAMPLE")
print("=" * 80)
print(f"  Query:      {ex['query']}")
print(f"  Answer:     {ex['answer']}")
print(f"  Document:   {doc_short}...")
print(f"  Doc words:  {ex['word_count']}")

print()
print("=" * 80)
print("HOW THIS EXPERIMENT WORKS")
print("=" * 80)
print()
print("  The T5Gemma encoder-decoder has two stages:")
print()
print("    1. ENCODER: reads text with bidirectional attention (sees everything)")
print("    2. DECODER: generates text, cross-attending to encoder output")
print()
print("  v3 setup (decoder has NO query):")
print("    - Decoder input: [BOS] + answer tokens")
print("    - Measured how well the model predicts the answer from encoder states alone")
print("    - Found d=+0.408 benefit from co-encoding query with document")
print()
print("  v4 setup (decoder HAS the query -- PRODUCTION REALISTIC):")
print("    - Decoder input: [BOS] + query tokens + answer tokens")
print("    - NLL computed only on answer token positions")
print("    - Models real production: query arrives at inference, decoder sees it")
print()
print("  KEY QUESTION: Does co-encoding still help when the decoder already")
print("  has the query? If not, the v3 benefit was just the decoder reading")
print("  the query from encoder output (trivial).")

print()
print("=" * 80)
print("CONDITIONS (8 total)")
print("=" * 80)

print()
print("  === WITH QUERY IN DECODER (production-realistic) ===")

print()
print("--- CONDITION 1: bare ---")
print()
print(f"  Encoder input:      [document]")
print(f"  Decoder cross-attn: all encoder tokens (= document)")
print(f"  Decoder input:      [BOS] + query + answer")
print(f"  NLL measured on:    answer tokens only")
print()
print(f"  Baseline. The decoder has the query but encoder only has the document.")
print()
print(f"  Encoder sees: \"{doc_short}...\"")
print(f"  Decoder sees: \"{ex['query']}\" -> \"{ex['answer']}\"")

print()
print("--- CONDITION 2: oracle_trunc  *** THE KEY CONDITION ***")
print()
print(f"  Encoder prefix:     \"{ex['query']}\"")
print(f"  Encoder input:      [query + document]  (full bidirectional attention)")
print(f"  Decoder cross-attn: document tokens ONLY  (query tokens MASKED)")
print(f"  Decoder input:      [BOS] + query + answer")
print(f"  NLL measured on:    answer tokens only")
print()
print(f"  The encoder co-encodes query + document, but the decoder cannot read")
print(f"  the query from encoder output. If this beats bare, the document")
print(f"  representations are genuinely improved -- and it still matters even")
print(f"  though the decoder already knows the query!")

print()
print("--- CONDITION 3: oracle_full ---")
print()
print(f"  Encoder prefix:     \"{ex['query']}\"")
print(f"  Encoder input:      [query + document]")
print(f"  Decoder cross-attn: ALL encoder tokens  (query visible)")
print(f"  Decoder input:      [BOS] + query + answer")
print()
print(f"  Decoder can read the query from BOTH its own input AND the encoder.")
print(f"  Comparison with oracle_trunc shows if reading query from encoder adds")
print(f"  anything beyond having it in the decoder input.")

print()
print("--- CONDITION 4: surr_template_trunc ---")
print()
print(f"  Encoder prefix:     \"{surr_template}\"  ('What is [top keyword]?')")
print(f"  Encoder input:      [template + document]")
print(f"  Decoder cross-attn: document tokens ONLY")
print(f"  Decoder input:      [BOS] + query + answer")
print()
print(f"  Production-realistic surrogate: cheap to generate offline, no query needed.")

print()
print("--- CONDITION 5: surr_doc_trunc ---")
print()
print(f"  Encoder prefix:     \"{surr_doc_kw}\"  (top-5 TF keywords from document)")
print(f"  Encoder input:      [keywords + document]")
print(f"  Decoder cross-attn: document tokens ONLY")
print(f"  Decoder input:      [BOS] + query + answer")

print()
print("--- CONDITION 6: random_trunc ---")
print()
print(f"  Encoder prefix:     \"{random_prefix[:60]}...\"")
print(f"                      ({query_word_count} words from unrelated passage)")
print(f"  Encoder input:      [random + document]")
print(f"  Decoder cross-attn: document tokens ONLY")
print(f"  Decoder input:      [BOS] + query + answer")
print()
print(f"  Structural control: tests if ANY prefix helps via attention redistribution,")
print(f"  even when the decoder already has the query.")

print()
print("  === WITHOUT QUERY IN DECODER (v3 replication) ===")

print()
print("--- CONDITION 7: bare_nq ---")
print()
print(f"  Encoder input:      [document]")
print(f"  Decoder cross-attn: all encoder tokens")
print(f"  Decoder input:      [BOS] + answer  (NO query)")
print(f"  NLL measured on:    answer tokens")
print()
print(f"  v3 baseline. Decoder does NOT see the query at all.")

print()
print("--- CONDITION 8: oracle_trunc_nq ---")
print()
print(f"  Encoder prefix:     \"{ex['query']}\"")
print(f"  Encoder input:      [query + document]")
print(f"  Decoder cross-attn: document tokens ONLY")
print(f"  Decoder input:      [BOS] + answer  (NO query)")
print()
print(f"  Replicates v3 Exp 01 (expected d~+0.4). Provides the reference:")
print(f"  how much does enrichment help when the decoder does NOT have the query?")

print()
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()
print(f"  {'#':<3} {'Condition':<25} {'Enc prefix':<22} {'Trunc':>6} {'Dec query':>10}")
print(f"  {'-'*70}")
print(f"  {'1':<3} {'bare':<25} {'(none)':<22} {'no':>6} {'yes':>10}")
print(f"  {'2':<3} {'oracle_trunc':<25} {'real query':<22} {'yes':>6} {'yes':>10}")
print(f"  {'3':<3} {'oracle_full':<25} {'real query':<22} {'no':>6} {'yes':>10}")
print(f"  {'4':<3} {'surr_template_trunc':<25} {surr_template:<22} {'yes':>6} {'yes':>10}")
print(f"  {'5':<3} {'surr_doc_trunc':<25} {surr_doc_kw[:20]:<22} {'yes':>6} {'yes':>10}")
print(f"  {'6':<3} {'random_trunc':<25} {'(unrelated words)':<22} {'yes':>6} {'yes':>10}")
print(f"  {'7':<3} {'bare_nq':<25} {'(none)':<22} {'no':>6} {'no':>10}")
print(f"  {'8':<3} {'oracle_trunc_nq':<25} {'real query':<22} {'yes':>6} {'no':>10}")

print()
print("=" * 80)
print("WHAT TO LOOK FOR IN RESULTS")
print("=" * 80)
print()
print("  1. oracle_trunc vs bare (both with query in decoder):")
print("     THE key question. If d > 0: enrichment helps even in production.")
print()
print("  2. oracle_trunc_nq vs bare_nq (without query in decoder):")
print("     Should replicate v3 Exp 01 (d ~ +0.4). Reference for comparison.")
print()
print("  3. Ratio: (oracle_trunc vs bare) / (oracle_trunc_nq vs bare_nq):")
print("     What fraction of enrichment survives when decoder has the query?")
print("     >80%: almost fully preserved. <30%: mostly redundant.")
print()
print("  4. oracle_full vs oracle_trunc (with query in decoder):")
print("     Does full cross-attention add anything beyond enriched doc reps")
print("     when the decoder already has the query in its own input?")
print()
print("  5. surr_doc_trunc vs bare:")
print("     Can a cheap doc-keyword surrogate provide production value?")
print()
print("  6. random_trunc vs bare:")
print("     Does structural attention redistribution still help when")
print("     the decoder has the query? (In v3 it explained 85% of benefit.)")
"""))

out_path = "experiments/encoder_decoder/01/01_examples.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
