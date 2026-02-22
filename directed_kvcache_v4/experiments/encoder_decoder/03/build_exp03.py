#!/usr/bin/env python3
# Build Exp 03 notebook: Cross-Dataset Validation on Neural-Bridge (Long Documents).
#
# Exp 01 (MS MARCO, ~98 tokens) showed v4 enrichment d=+0.228.
# Exp 02 showed the benefit GROWS with padded document length (d=+0.43 at 4096 tok).
# But padding with unrelated text is artificial.
#
# Neural-bridge/rag-dataset-12000 has naturally long documents (~600 words, ~800-1000
# tokens) and longer queries (~18 words). This tests whether the v4 production-
# realistic enrichment holds on real long documents from a different dataset.
#
# v3 Exp 3D on this dataset (no query in decoder) found:
#   - Structure = 84.3%, matching MS MARCO's 84.7%
#   - ALL surrogates beat oracle (semantic interference)
#
# Key question: does the v4 mechanism shift (structural→content) hold cross-dataset?
#
# 6 conditions, N=500.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 03: Cross-Dataset Validation — Neural-Bridge (Long Documents)

## Motivation

Exp 01 (MS MARCO, ~98 token docs) showed the v4 enrichment benefit survives with query
in decoder (d=+0.228). Exp 02 showed this benefit *grows* at longer padded documents
(d=+0.43 at 4096 tokens). But Exp 02 used artificial padding with unrelated text.

Neural-bridge/rag-dataset-12000 provides a natural test:
- Documents are ~600 words (~800-1000 tokens) — genuinely long
- Queries are ~18 words (3x MS MARCO)
- Different domain and generation process

v3 Exp 3D on this dataset (no query in decoder) found:
- Structure = 84.3% (matched MS MARCO's 84.7%)
- ALL surrogates beat oracle — the real query creates semantic interference
- Oracle d was modest because of this interference

**Key questions**:
1. Does the v4 enrichment benefit hold on naturally long documents?
2. Does the v4 mechanism shift (structural collapse, content dominance) replicate?
3. Does the "surrogates beat oracle" phenomenon from v3 persist when the decoder has the query?

## Conditions (6 total)

### With query in decoder (production-realistic):

| # | Condition | Encoder input | Cross-attn | Purpose |
|---|-----------|--------------|------------|---------|
| 1 | bare | [document] | all | Baseline |
| 2 | oracle_trunc | [query + doc] | doc only | Upper bound |
| 3 | surr_doc_trunc | [top-5 kw + doc] | doc only | Production surrogate |
| 4 | random_trunc | [random words + doc] | doc only | Structural control |

### Without query in decoder (v3 replication):

| # | Condition | Encoder input | Cross-attn | Purpose |
|---|-----------|--------------|------------|---------|
| 5 | bare_nq | [document] | all | v3 baseline |
| 6 | oracle_trunc_nq | [query + doc] | doc only | v3 enrichment reference |""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup and model loading
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
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../../results/exp03")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = getattr(model.config, 'decoder_start_token_id', None) or tokenizer.bos_token_id

print(f"Exp 03: Cross-Dataset Validation — Neural-Bridge")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
""")


# ===== Cell 3: Scoring helpers =====
code(r"""# Cell 3: Scoring helpers

def count_prefix_tokens(prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # No query in decoder — used for _nq conditions (v3 replication).
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
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
        del encoder_outputs
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


def score_nll_query_prefix(encoder_text, query_text, answer_text,
                           prefix_token_count=0, truncate=False):
    # Query as decoder prefix — production-realistic.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
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

    query_ids = tokenizer(query_text, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        del encoder_outputs
        return 0.0

    dec_ids = [BOS_ID] + query_ids + answer_ids
    dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)

    n_query = len(query_ids)
    n_answer = len(answer_ids)

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=dec_tensor,
        )

    logits = outputs.logits
    answer_logits = logits[0, n_query:n_query + n_answer, :]

    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


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

def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

print("Scoring functions defined.")
""")


# ===== Cell 4: Load neural-bridge data =====
code(r"""# Cell 4: Load neural-bridge/rag-dataset-12000
from datasets import load_dataset

print("Loading neural-bridge/rag-dataset-12000...")
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
print(f"Total samples: {len(ds)}")

# Filter to long queries with real answers (same filter as v3 Exp 3D)
all_candidates = []
for row in ds:
    q = row.get("question", "")
    doc = row.get("context", "")
    answer = row.get("answer", "")
    if not q or not doc or not answer:
        continue
    q_words = len(q.split())
    a_words = len(answer.split())
    if q_words >= 15 and a_words >= 5:
        all_candidates.append({
            "query": q,
            "passage": doc,
            "answer": answer,
            "query_words": q_words,
            "doc_words": len(doc.split()),
            "answer_words": a_words,
        })

print(f"Candidates (q>=15w, a>=5w): {len(all_candidates)}")

np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

# Generate surrogates
for i, s in enumerate(samples):
    s['surr_doc'] = make_surrogate_from_doc(s['passage'])

    # Random prefix: words from unrelated document, matched to query word count
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    query_word_count = len(s['query'].split())
    s['random_prefix'] = " ".join(other_words[:query_word_count])

    # Count prefix tokens
    s['n_prefix_oracle'] = count_prefix_tokens(s['query'], s['passage'])
    s['n_prefix_doc'] = count_prefix_tokens(s['surr_doc'], s['passage'])
    s['n_prefix_random'] = count_prefix_tokens(s['random_prefix'], s['passage'])

# Dataset statistics
q_lens = np.array([s['query_words'] for s in samples])
d_lens = np.array([s['doc_words'] for s in samples])
a_lens = np.array([s['answer_words'] for s in samples])
doc_tok_counts = [len(tokenizer(s['passage'], add_special_tokens=True).input_ids)
                  for s in samples]

print(f"\nSample statistics (N={N_SAMPLES}):")
print(f"  Query:    mean={q_lens.mean():.1f}w, median={np.median(q_lens):.0f}w")
print(f"  Document: mean={d_lens.mean():.0f}w, median={np.median(d_lens):.0f}w")
print(f"  Doc toks: mean={np.mean(doc_tok_counts):.0f}, median={np.median(doc_tok_counts):.0f}, "
      f"max={np.max(doc_tok_counts)}")
print(f"  Answer:   mean={a_lens.mean():.1f}w, median={np.median(a_lens):.0f}w")

print(f"\nComparison with MS MARCO (Exp 01):")
print(f"  MS MARCO: query=6.0w, doc=~60w (~98 tok), answer=~20w")
print(f"  This:     query={q_lens.mean():.1f}w, doc={d_lens.mean():.0f}w "
      f"(~{np.mean(doc_tok_counts):.0f} tok), answer={a_lens.mean():.0f}w")

print(f"\nFirst sample:")
print(f"  Query:   {samples[0]['query'][:100]}...")
print(f"  Answer:  {samples[0]['answer'][:100]}...")
print(f"  Doc:     {samples[0]['passage'][:100]}...")
print(f"  Surr:    {samples[0]['surr_doc']}")
""")


# ===== Cell 5: Show example conditions =====
code(r"""# Cell 5: Show example conditions
print("=" * 70)
print("EXAMPLE CONDITIONS (sample 0)")
print("=" * 70)

ex = samples[0]
print(f"\nQuery ({ex['query_words']}w):  {ex['query'][:100]}...")
print(f"Answer ({ex['answer_words']}w): {ex['answer'][:100]}...")
print(f"Doc ({ex['doc_words']}w):    {ex['passage'][:100]}...")

print(f"\n  {'Condition':<20} {'Enc prefix':<25} {'Trunc':>6} {'Dec query':>10} {'Pfx tok':>8}")
print(f"  {'-'*75}")
for name, prefix, trunc, has_q, n_pfx in [
    ('bare',             '(none)',                'no',  'yes', 0),
    ('oracle_trunc',     'real query',            'yes', 'yes', ex['n_prefix_oracle']),
    ('surr_doc_trunc',   ex['surr_doc'][:20],     'yes', 'yes', ex['n_prefix_doc']),
    ('random_trunc',     '(unrelated)',           'yes', 'yes', ex['n_prefix_random']),
    ('bare_nq',          '(none)',                'no',  'no',  0),
    ('oracle_trunc_nq',  'real query',            'yes', 'no',  ex['n_prefix_oracle']),
]:
    print(f"  {name:<20} {prefix:<25} {trunc:>6} {has_q:>10} {n_pfx:>8}")

# Sanity check
print(f"\nSanity check...")
nll_bare = score_nll_query_prefix(ex['passage'], ex['query'], ex['answer'])
nll_oracle = score_nll_query_prefix(
    ex['query'] + "\n" + ex['passage'], ex['query'], ex['answer'],
    prefix_token_count=ex['n_prefix_oracle'], truncate=True)
print(f"  bare:          {nll_bare:.4f}")
print(f"  oracle_trunc:  {nll_oracle:.4f}")
print(f"  delta:         {nll_bare - nll_oracle:+.4f}")
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 6 conditions x 500 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'surr_doc_trunc', 'random_trunc',
    'bare_nq', 'oracle_trunc_nq',
]

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            results = ckpt['results']
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    result = {
        'query': query,
        'answer': answer,
        'query_words': s['query_words'],
        'doc_words': s['doc_words'],
        'answer_words': s['answer_words'],
    }

    # --- With query in decoder ---
    result['nll_bare'] = score_nll_query_prefix(passage, query, answer)

    result['nll_oracle_trunc'] = score_nll_query_prefix(
        query + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_oracle'], truncate=True)

    result['nll_surr_doc_trunc'] = score_nll_query_prefix(
        s['surr_doc'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_doc'], truncate=True)

    result['nll_random_trunc'] = score_nll_query_prefix(
        s['random_prefix'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_random'], truncate=True)

    # --- Without query in decoder (v3 replication) ---
    result['nll_bare_nq'] = score_nll(passage, answer)

    result['nll_oracle_trunc_nq'] = score_nll(
        query + "\n" + passage, answer,
        prefix_token_count=s['n_prefix_oracle'], truncate=True)

    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = i - start_idx + 1
        eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 7: Results table =====
code(r"""# Cell 7: Results table
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

bare = np.array([r['nll_bare'] for r in results])
oracle_trunc = np.array([r['nll_oracle_trunc'] for r in results])
surr_doc = np.array([r['nll_surr_doc_trunc'] for r in results])
random_trunc = np.array([r['nll_random_trunc'] for r in results])
bare_nq = np.array([r['nll_bare_nq'] for r in results])
oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])

# Bonferroni: 3 query-prefix + 1 nq = 4
N_BONF = 4

print(f"\n--- With query in decoder (production-realistic) ---")
print(f"  {'Condition':<20} {'NLL':>8} {'vs bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*74}")

analysis = {}
for name, nlls in [('bare', bare), ('oracle_trunc', oracle_trunc),
                    ('surr_doc_trunc', surr_doc), ('random_trunc', random_trunc)]:
    mean_nll = nlls.mean()
    if name == 'bare':
        print(f"  {name:<20} {mean_nll:>8.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001/N_BONF else '**' if p_val < 0.01/N_BONF else '*' if p_val < 0.05/N_BONF else 'ns'
        print(f"  {name:<20} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# No-query conditions
diff_nq = bare_nq - oracle_nq
d_nq = cohens_d(diff_nq)
win_nq = 100 * np.mean(diff_nq > 0)
_, p_nq = stats.ttest_1samp(diff_nq, 0)
sig_nq = '***' if p_nq < 0.001/N_BONF else '**' if p_nq < 0.01/N_BONF else '*' if p_nq < 0.05/N_BONF else 'ns'

print(f"\n--- Without query in decoder (v3 replication) ---")
print(f"  {'bare_nq':<20} {bare_nq.mean():>8.4f}")
print(f"  {'oracle_trunc_nq':<20} {oracle_nq.mean():>8.4f} {diff_nq.mean():>+10.4f} {d_nq:>+8.3f} {win_nq:>7.1f}% {p_nq:>12.2e} {sig_nq:>5}")

analysis['bare_nq'] = {'mean_nll': float(bare_nq.mean())}
analysis['oracle_trunc_nq'] = {
    'mean_nll': float(oracle_nq.mean()), 'delta': float(diff_nq.mean()),
    'd': float(d_nq), 'win_pct': float(win_nq), 'p': float(p_nq),
}
""")


# ===== Cell 8: Cross-dataset comparison =====
code(r"""# Cell 8: Cross-dataset comparison with Exp 01 (MS MARCO)
print("=" * 70)
print("CROSS-DATASET COMPARISON")
print("=" * 70)

d_oracle = cohens_d(bare - oracle_trunc)
d_surr = cohens_d(bare - surr_doc)
d_random = cohens_d(bare - random_trunc)
d_oracle_nq = cohens_d(bare_nq - oracle_nq)

# Exp 01 reference values
exp01 = {
    'oracle_trunc': 0.228, 'surr_doc_trunc': 0.148,
    'random_trunc': 0.080, 'oracle_trunc_nq': 0.376,
}

print(f"\n  {'Condition':<20} {'MS MARCO (Exp01)':>18} {'Neural-Bridge':>15} {'Ratio':>8}")
print(f"  {'-'*65}")
for name, exp01_d in exp01.items():
    if name == 'oracle_trunc':
        this_d = d_oracle
    elif name == 'surr_doc_trunc':
        this_d = d_surr
    elif name == 'random_trunc':
        this_d = d_random
    else:
        this_d = d_oracle_nq
    ratio = this_d / exp01_d if exp01_d != 0 else 0
    print(f"  {name:<20} {exp01_d:>+18.3f} {this_d:>+15.3f} {ratio:>7.1f}x")

# Structural fraction comparison
struct_marco = 0.080 / 0.228 * 100 if 0.228 > 0 else 0
struct_nb = d_random / d_oracle * 100 if d_oracle > 0 else 0

print(f"\n--- Structural Fraction (random/oracle) ---")
print(f"  MS MARCO Exp 01: {struct_marco:.0f}%")
print(f"  Neural-Bridge:   {struct_nb:.0f}%")

# Surrogate efficiency
surr_pct_marco = 0.148 / 0.228 * 100 if 0.228 > 0 else 0
surr_pct_nb = d_surr / d_oracle * 100 if d_oracle > 0 else 0

print(f"\n--- Surrogate Efficiency (surr_doc/oracle) ---")
print(f"  MS MARCO Exp 01: {surr_pct_marco:.0f}%")
print(f"  Neural-Bridge:   {surr_pct_nb:.0f}%")

# v4/v3 ratio
ratio_marco = 0.228 / 0.376 * 100
ratio_nb = d_oracle / d_oracle_nq * 100 if d_oracle_nq > 0 else 0

print(f"\n--- v4/v3 Enrichment Ratio (how much survives with query in decoder) ---")
print(f"  MS MARCO:        {ratio_marco:.0f}%")
print(f"  Neural-Bridge:   {ratio_nb:.0f}%")

# v3 Exp 3D comparison (surrogates beat oracle)
print(f"\n--- v3 Exp 3D Comparison (surrogates beat oracle phenomenon) ---")
print(f"  v3 Exp 3D: ALL surrogates beat oracle (150%+ of oracle d)")
print(f"  v3 explanation: real query creates semantic interference in encoder")
if d_surr > d_oracle:
    print(f"  v4: surr_doc ({d_surr:+.3f}) STILL beats oracle ({d_oracle:+.3f})")
    print(f"       -> Semantic interference persists even with query in decoder")
else:
    print(f"  v4: oracle ({d_oracle:+.3f}) beats surr_doc ({d_surr:+.3f})")
    print(f"       -> Decoder query resolves the interference")
""")


# ===== Cell 9: Key comparison =====
code(r"""# Cell 9: Key comparison — enrichment with query vs without query
print("=" * 70)
print("KEY COMPARISON: Is enrichment redundant when decoder has the query?")
print("=" * 70)

enrichment_with_q = bare - oracle_trunc
enrichment_no_q = bare_nq - oracle_nq
d_with_q = cohens_d(enrichment_with_q)
d_no_q = cohens_d(enrichment_no_q)

ratio = d_with_q / d_no_q * 100 if d_no_q > 0 else 0

print(f"\n  Enrichment with query in decoder:    d={d_with_q:+.3f}")
print(f"  Enrichment without query (v3 repl):  d={d_no_q:+.3f}")
print(f"  Ratio: {ratio:.0f}%")

# Per-sample correlation
r_corr, p_corr = stats.pearsonr(enrichment_with_q, enrichment_no_q)
print(f"  Per-sample correlation: r={r_corr:.3f} (p={p_corr:.2e})")

# Hardness gradient
print(f"\n--- Hardness gradient (with query in decoder) ---")
quintile_bounds = np.percentile(bare, [20, 40, 60, 80])
quintiles = np.digitize(bare, quintile_bounds)

print(f"  {'Quintile':<12} {'N':>4} {'bare':>8} {'oracle':>8} {'delta':>8} {'d':>8}")
print(f"  {'-'*50}")
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 5:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare[mask].mean()
    o = oracle_trunc[mask].mean()
    delta = (bare[mask] - oracle_trunc[mask]).mean()
    d = cohens_d(bare[mask] - oracle_trunc[mask])
    print(f"  {qlabel:<12} {n_q:>4} {b:>8.4f} {o:>8.4f} {delta:>+8.4f} {d:>+8.3f}")
""")


# ===== Cell 10: Verdict + save =====
code(r"""# Cell 10: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 03: Neural-Bridge Cross-Dataset Validation")
print("=" * 70)

d_oracle = cohens_d(bare - oracle_trunc)
d_surr = cohens_d(bare - surr_doc)
d_random = cohens_d(bare - random_trunc)
d_oracle_nq = cohens_d(bare_nq - oracle_nq)
ratio = d_oracle / d_oracle_nq * 100 if d_oracle_nq > 0 else 0
struct_frac = d_random / d_oracle * 100 if d_oracle > 0 else 0

print(f"\nModel: {MODEL_NAME}")
print(f"Dataset: neural-bridge/rag-dataset-12000")
print(f"N: {len(results)}, mean doc: {np.mean([r['doc_words'] for r in results]):.0f} words")

print(f"\n--- Key results ---")
print(f"  Oracle enrichment (query in decoder):  d={d_oracle:+.3f}")
print(f"  Oracle enrichment (no query, v3 repl): d={d_oracle_nq:+.3f}")
print(f"  Ratio (v4/v3):                         {ratio:.0f}%")
print(f"  Structural fraction (random/oracle):   {struct_frac:.0f}%")
print(f"  Surrogate doc efficiency:              {d_surr/d_oracle*100:.0f}% of oracle" if d_oracle > 0 else "")

_, p_oracle = stats.ttest_1samp(bare - oracle_trunc, 0)
_, p_surr = stats.ttest_1samp(bare - surr_doc, 0)
_, p_rand = stats.ttest_1samp(bare - random_trunc, 0)

print(f"\n--- Significance ---")
print(f"  oracle_trunc: p={p_oracle:.2e}")
print(f"  surr_doc:     p={p_surr:.2e}")
print(f"  random:       p={p_rand:.2e}")

print(f"\n--- Cross-dataset consistency ---")
print(f"  MS MARCO (Exp 01): oracle d=+0.228, surr_doc d=+0.148, random d=+0.080")
print(f"  Neural-Bridge:     oracle d={d_oracle:+.3f}, surr_doc d={d_surr:+.3f}, random d={d_random:+.3f}")

if d_oracle > 0.1:
    print(f"\n  CONCLUSION: v4 enrichment benefit REPLICATES on naturally long documents.")
else:
    print(f"\n  CONCLUSION: v4 enrichment benefit does NOT replicate on this dataset.")

# Save
final_results = {
    'experiment': 'v4_exp03_neural_bridge',
    'model': MODEL_NAME,
    'dataset': 'neural-bridge/rag-dataset-12000',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_stats': {
        'mean_query_words': float(np.mean([r['query_words'] for r in results])),
        'mean_doc_words': float(np.mean([r['doc_words'] for r in results])),
        'mean_answer_words': float(np.mean([r['answer_words'] for r in results])),
    },
    'key_result': {
        'enrichment_with_query_d': float(d_oracle),
        'enrichment_no_query_d': float(d_oracle_nq),
        'ratio_pct': float(ratio),
        'structural_fraction_pct': float(struct_frac),
    },
    'conditions': analysis,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/encoder_decoder/03/03_neural_bridge.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
