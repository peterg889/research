#!/usr/bin/env python3
# Build Exp 02 notebook: Length Scaling with Query in Decoder.
#
# v3 Exp 03/03B showed NO decay of enrichment up to 6144 tokens — but that was
# with the structural mechanism dominant (85%). In v4 Exp 01, the structural
# component collapsed to ~35% when the decoder has the query, and the content-
# dependent component dominates. Content-dependent enrichment may scale
# differently with document length.
#
# This experiment pads the same MS MARCO passages to controlled lengths with
# unrelated text. Same questions, same answers — only document length changes.
# Within-subject design for maximum statistical power.
#
# 6 conditions x 6 length bins x 500 samples.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 02: Length Scaling with Query in Decoder

## Motivation

v3 Exp 03/03B showed NO decay of enrichment benefit up to 6144 tokens. But the v3
decoder never had the query — the mechanism was 85% structural (any prefix worked).

v4 Exp 01 showed that when the decoder has the query:
- Structural component collapsed: 85% → ~35% (random d=+0.080, ns)
- Content-dependent component now dominant: surr_doc d=+0.148 (65% of oracle)
- Overall enrichment preserved at 61%: oracle d=+0.228 vs v3's d=+0.376

**Key question**: Does the v4 content-dependent enrichment decay with document length?
The structural mechanism was length-invariant (v3 Exp 03), but the content-dependent
component might dilute as documents get longer — the prefix's semantic influence may
not reach distant tokens as effectively.

## Method

Pad the same short MS MARCO passages to controlled token lengths using unrelated
passages. Same questions, same answers — only document length varies. Within-subject
design: every sample appears at every length.

## Conditions (6 per length bin)

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
| 6 | oracle_trunc_nq | [query + doc] | doc only | v3 enrichment reference |

## Length bins

| Bin | Target tokens | v4 Exp 01 | v3 Exp 03 |
|-----|--------------|-----------|-----------|
| original | ~98 tok (no padding) | d=+0.228 | d=+0.41 |
| 256 | padded | ? | d=+0.42 |
| 512 | padded | ? | d=+0.38 |
| 1024 | padded | ? | d=+0.45 |
| 2048 | padded | ? | d=+0.42 |
| 4096 | padded | ? | d=+0.40 (03B) |

## What to look for

1. **d(oracle_trunc vs bare) at each length**: Does it decay? (v3: flat)
2. **d(surr_doc_trunc vs bare)**: Does the production surrogate decay faster?
3. **d(random_trunc vs bare)**: Does the structural component stay marginal?
4. **d(oracle_trunc_nq vs bare_nq)**: v3 replication (should be flat)
5. **Ratio: v4/v3 enrichment at each length**: Does the gap widen or narrow?""")


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
LENGTH_BINS = ["original", "256", "512", "1024", "2048", "4096"]
TARGET_LENGTHS = {
    "original": None,
    "256": 256,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
    "4096": 4096,
}

RESULTS_DIR = Path("../../../results/exp02")
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

print(f"Exp 02: Length Scaling with Query in Decoder")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"Length bins: {LENGTH_BINS}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Decoder start token ID (BOS): {BOS_ID}")
""")


# ===== Cell 3: Scoring helpers =====
code(r"""# Cell 3: Scoring helpers
# NOTE: max_length=4608 for encoder (4096 doc + prefix tokens)

def count_prefix_tokens(prefix_text, document_text):
    # BPE-aware token count of prefix in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=4608).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=4608).input_ids
    return len(full_ids) - len(doc_ids)


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer tokens — decoder receives ONLY answer (no query).
    # Used for _nq (no-query) conditions that replicate v3.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=4608).input_ids.to(DEVICE)
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
    # Score NLL of answer tokens with query as decoder prefix.
    # Production-realistic: decoder_input_ids = [BOS] + query_ids + answer_ids.
    # NLL is computed only on answer tokens.

    # 1. Encode
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=4608).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # 2. Cross-attention mask
    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    # 3. Tokenize query and answer for decoder
    query_ids = tokenizer(query_text, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        del encoder_outputs
        return 0.0

    # 4. Build decoder_input_ids = [BOS] + query_ids + answer_ids
    dec_ids = [BOS_ID] + query_ids + answer_ids
    dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)

    n_query = len(query_ids)
    n_answer = len(answer_ids)

    # 5. Forward pass
    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=dec_tensor,
        )

    # 6. Extract answer logits
    # logits[0, t, :] predicts the token at position t+1
    # Position n_query predicts the first answer token
    logits = outputs.logits
    answer_logits = logits[0, n_query:n_query + n_answer, :]

    # 7. Compute NLL
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

print("Scoring functions defined:")
print("  score_nll(encoder_text, answer_text, prefix_token_count, truncate)")
print("  score_nll_query_prefix(encoder_text, query_text, answer_text, prefix_token_count, truncate)")
print(f"  Encoder max_length: 4608 (supports 4096 doc + prefix)")
""")


# ===== Cell 4: Load data + build padding pool =====
code(r"""# Cell 4: Load MS MARCO data and build padding pool
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
padding_pool = []

for item in ds:
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

    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300 and answer:
            if len(all_candidates) < 3 * N_SAMPLES:
                all_candidates.append({
                    'passage': pt, 'query': query, 'answer': answer,
                    'word_count': wc,
                })
        elif sel == 0 and 20 <= wc <= 200:
            padding_pool.append(pt)

    if len(all_candidates) >= 3 * N_SAMPLES and len(padding_pool) >= 10000:
        break

print(f"Total candidates: {len(all_candidates)}")
print(f"Padding pool: {len(padding_pool)} unrelated passages")

np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates

# Shuffle padding pool for random padding
np.random.shuffle(padding_pool)

# Generate surrogates for each sample
for i, s in enumerate(samples):
    s['surr_doc'] = make_surrogate_from_doc(s['passage'])

    # Random prefix: words from unrelated passage, matched to query word count
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    query_word_count = len(s['query'].split())
    s['random_prefix'] = " ".join(other_words[:query_word_count])

gc.collect()

print(f"\nLoaded {len(samples)} samples (SEED={SEED})")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")

# Show token counts for original passages
orig_tok_counts = [len(tokenizer(s['passage'], add_special_tokens=True).input_ids)
                   for s in samples]
print(f"Original passage tokens: mean={np.mean(orig_tok_counts):.0f}, "
      f"median={np.median(orig_tok_counts):.0f}, "
      f"min={np.min(orig_tok_counts)}, max={np.max(orig_tok_counts)}")

print(f"\nFirst sample:")
print(f"  Query:    {samples[0]['query'][:70]}...")
print(f"  Answer:   {samples[0]['answer'][:70]}...")
print(f"  Passage:  {samples[0]['passage'][:70]}...")
print(f"  Surr doc: {samples[0]['surr_doc']}")
print(f"  Random:   {samples[0]['random_prefix'][:60]}...")
""")


# ===== Cell 5: Build padded documents =====
code(r"""# Cell 5: Build padded documents at each length bin
print("=" * 70)
print("BUILDING PADDED DOCUMENTS")
print("=" * 70)

def pad_passage_to_length(passage, target_tokens, pool, pool_offset):
    # Pad a passage to target_tokens by appending unrelated passages.
    if target_tokens is None:
        toks = tokenizer(passage, add_special_tokens=True).input_ids
        return passage, len(toks), 0

    current_ids = tokenizer(passage, add_special_tokens=True).input_ids
    if len(current_ids) >= target_tokens:
        return passage, len(current_ids), 0

    padded = passage
    n_used = 0
    idx = pool_offset

    while True:
        if idx >= len(pool):
            idx = 0
        candidate = padded + "\n\n" + pool[idx]
        candidate_ids = tokenizer(candidate, add_special_tokens=True).input_ids
        if len(candidate_ids) >= target_tokens:
            # Add words from this passage until we hit target
            pad_words = pool[idx].split()
            for w_end in range(1, len(pad_words) + 1):
                partial = padded + "\n\n" + " ".join(pad_words[:w_end])
                partial_ids = tokenizer(partial, add_special_tokens=True).input_ids
                if len(partial_ids) >= target_tokens:
                    padded = partial
                    break
            else:
                padded = candidate
            n_used += 1
            break
        padded = candidate
        n_used += 1
        idx += 1

    final_ids = tokenizer(padded, add_special_tokens=True).input_ids
    return padded, len(final_ids), n_used


# Build padded versions for each sample at each length
padded_docs = {}
padded_stats = {}

for length_bin, target_tokens in TARGET_LENGTHS.items():
    padded_docs[length_bin] = []
    tok_counts = []

    for i, s in enumerate(samples):
        pool_offset = i * 50
        padded_text, actual_tokens, n_pad = pad_passage_to_length(
            s['passage'], target_tokens, padding_pool, pool_offset
        )
        padded_docs[length_bin].append(padded_text)
        tok_counts.append(actual_tokens)

    padded_stats[length_bin] = {
        'mean': float(np.mean(tok_counts)),
        'min': int(np.min(tok_counts)),
        'max': int(np.max(tok_counts)),
        'median': float(np.median(tok_counts)),
    }

    print(f"  {length_bin:>8s}: mean={padded_stats[length_bin]['mean']:.0f} tokens "
          f"(min={padded_stats[length_bin]['min']}, max={padded_stats[length_bin]['max']}, "
          f"median={padded_stats[length_bin]['median']:.0f})")

# Preview sample 0
print(f"\n--- Sample 0 preview ---")
print(f"  Query:  {samples[0]['query'][:80]}")
print(f"  Answer: {samples[0]['answer'][:80]}")
for lb in LENGTH_BINS:
    preview = padded_docs[lb][0]
    tok_count = len(tokenizer(preview, add_special_tokens=True).input_ids)
    print(f"  {lb:>8s}: {tok_count} tokens, starts='{preview[:60]}...', ends='...{preview[-40:]}'")
""")


# ===== Cell 6: Show example conditions =====
code(r"""# Cell 6: Show example conditions for sample 0 at original length
print("=" * 70)
print("EXAMPLE CONDITIONS (sample 0, original length)")
print("=" * 70)

ex = samples[0]
doc = padded_docs["original"][0]

# Count prefix tokens
n_prefix_oracle = count_prefix_tokens(ex['query'], doc)
n_prefix_doc = count_prefix_tokens(ex['surr_doc'], doc)
n_prefix_random = count_prefix_tokens(ex['random_prefix'], doc)

print(f"\nQuery:          {ex['query']}")
print(f"Answer:         {ex['answer']}")
print(f"Surr doc kw:    {ex['surr_doc']}")
print(f"Random prefix:  {ex['random_prefix'][:60]}...")

print(f"\n  {'Condition':<20} {'Enc prefix':<25} {'Trunc':>6} {'Dec query':>10} {'Pfx tok':>8}")
print(f"  {'-'*75}")
for name, prefix, trunc, has_q, n_pfx in [
    ('bare',             '(none)',           'no',  'yes', 0),
    ('oracle_trunc',     'real query',       'yes', 'yes', n_prefix_oracle),
    ('surr_doc_trunc',   ex['surr_doc'][:20],'yes', 'yes', n_prefix_doc),
    ('random_trunc',     '(unrelated)',      'yes', 'yes', n_prefix_random),
    ('bare_nq',          '(none)',           'no',  'no',  0),
    ('oracle_trunc_nq',  'real query',       'yes', 'no',  n_prefix_oracle),
]:
    print(f"  {name:<20} {prefix:<25} {trunc:>6} {has_q:>10} {n_pfx:>8}")

# Decoder input structure
q_ids = tokenizer(ex['query'], add_special_tokens=False).input_ids
a_ids = tokenizer(ex['answer'], add_special_tokens=False).input_ids
print(f"\nDecoder input (query-prefix): [BOS] + query ({len(q_ids)} tok) + "
      f"answer ({len(a_ids)} tok) = {1 + len(q_ids) + len(a_ids)} tok total")
print(f"NLL computed on last {len(a_ids)} positions (answer only)")

# Quick sanity at original length
print(f"\nSanity check (original length)...")
nll_bare = score_nll_query_prefix(doc, ex['query'], ex['answer'])
nll_oracle = score_nll_query_prefix(
    ex['query'] + "\n" + doc, ex['query'], ex['answer'],
    prefix_token_count=n_prefix_oracle, truncate=True)
print(f"  bare:          {nll_bare:.4f}")
print(f"  oracle_trunc:  {nll_oracle:.4f}")
print(f"  delta:         {nll_bare - nll_oracle:+.4f}")
""")


# ===== Cell 7: Scoring loop =====
code(r"""# Cell 7: Scoring loop — 6 conditions x 6 length bins x 500 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'surr_doc_trunc', 'random_trunc',
    'bare_nq', 'oracle_trunc_nq',
]

# Checkpoint format: {bins: {length_bin: {results: [...]}}, n_total: N}
all_checkpoint = {}
if CHECKPOINT_PATH.exists():
    saved = json.loads(CHECKPOINT_PATH.read_text())
    if saved.get('n_total') == N_SAMPLES:
        all_checkpoint = saved.get('bins', {})
        summary = ', '.join(f'{k}={len(v.get("results",[]))}' for k, v in all_checkpoint.items())
        print(f"Loaded checkpoint: {summary}")

t0_total = time.time()

for length_bin in LENGTH_BINS:
    print(f"\n{'='*70}")
    print(f"LENGTH BIN: {length_bin} "
          f"(target={TARGET_LENGTHS[length_bin] or 'no padding'})")
    print(f"{'='*70}")

    # Check for existing results
    bin_results = []
    start_idx = 0
    if length_bin in all_checkpoint:
        bin_data = all_checkpoint[length_bin]
        saved_results = bin_data.get('results', [])
        saved_queries = [r['query'][:50] for r in saved_results]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            bin_results = saved_results
            start_idx = len(bin_results)
            print(f"  Resuming from sample {start_idx}/{N_SAMPLES}")

    if start_idx >= N_SAMPLES:
        print(f"  Already complete ({len(bin_results)} results)")
        all_checkpoint[length_bin] = {"results": bin_results}
        continue

    t0_bin = time.time()

    for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
                  desc=f"  {length_bin}"):
        s = samples[i]
        padded_passage = padded_docs[length_bin][i]
        query = s['query']
        answer = s['answer']

        # Count prefix tokens for this (padded) passage
        n_pfx_oracle = count_prefix_tokens(query, padded_passage)
        n_pfx_doc = count_prefix_tokens(s['surr_doc'], padded_passage)
        n_pfx_random = count_prefix_tokens(s['random_prefix'], padded_passage)

        result = {
            'query': query,
            'answer': answer,
            'passage_words': s['word_count'],
            'padded_tokens': len(tokenizer(padded_passage, add_special_tokens=True).input_ids),
        }

        # --- Conditions 1-4: query in decoder (production-realistic) ---

        # 1. bare: encoder=[doc], decoder=[query]->answer
        result['nll_bare'] = score_nll_query_prefix(
            padded_passage, query, answer)

        # 2. oracle_trunc: encoder=[query+doc], decoder=[query]->answer, mask prefix
        result['nll_oracle_trunc'] = score_nll_query_prefix(
            query + "\n" + padded_passage, query, answer,
            prefix_token_count=n_pfx_oracle, truncate=True)

        # 3. surr_doc_trunc: encoder=[kw+doc], decoder=[query]->answer, mask prefix
        result['nll_surr_doc_trunc'] = score_nll_query_prefix(
            s['surr_doc'] + "\n" + padded_passage, query, answer,
            prefix_token_count=n_pfx_doc, truncate=True)

        # 4. random_trunc: encoder=[random+doc], decoder=[query]->answer, mask prefix
        result['nll_random_trunc'] = score_nll_query_prefix(
            s['random_prefix'] + "\n" + padded_passage, query, answer,
            prefix_token_count=n_pfx_random, truncate=True)

        # --- Conditions 5-6: no query in decoder (v3 replication) ---

        # 5. bare_nq: encoder=[doc], decoder=answer only
        result['nll_bare_nq'] = score_nll(padded_passage, answer)

        # 6. oracle_trunc_nq: encoder=[query+doc], decoder=answer only, mask prefix
        result['nll_oracle_trunc_nq'] = score_nll(
            query + "\n" + padded_passage, answer,
            prefix_token_count=n_pfx_oracle, truncate=True)

        bin_results.append(result)

        if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
            all_checkpoint[length_bin] = {"results": bin_results}
            ckpt = {
                'n_total': N_SAMPLES,
                'bins': all_checkpoint,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            CHECKPOINT_PATH.write_text(json.dumps(ckpt))
            elapsed_bin = time.time() - t0_bin
            done = i - start_idx + 1
            eta = (N_SAMPLES - i - 1) * elapsed_bin / done if done > 0 else 0
            tqdm.write(f"    Checkpoint {i+1}/{N_SAMPLES} | "
                       f"{elapsed_bin/60:.1f}m | ETA {eta/60:.1f}m")

        gc.collect()
        torch.cuda.empty_cache()

    elapsed_bin = time.time() - t0_bin
    print(f"  {length_bin} complete: {len(bin_results)} samples in {elapsed_bin/60:.1f} min")

    # Quick peek
    bare_arr = np.array([r['nll_bare'] for r in bin_results])
    orc_arr = np.array([r['nll_oracle_trunc'] for r in bin_results])
    d_peek = cohens_d(bare_arr - orc_arr)
    print(f"  Quick peek: oracle_trunc d={d_peek:+.3f}")

elapsed_total = time.time() - t0_total
print(f"\n{'='*70}")
print(f"ALL BINS COMPLETE: {elapsed_total/60:.1f} min total")
print(f"{'='*70}")
""")


# ===== Cell 8: Results table per length =====
code(r"""# Cell 8: Results table — all conditions at all lengths
print("=" * 70)
print(f"RESULTS (N={N_SAMPLES})")
print("=" * 70)

# N_BONFERRONI: 3 query-prefix conditions x 6 lengths + 1 nq condition x 6 lengths = 24
N_BONF = 24

results_by_bin = {}
analysis = {}

for length_bin in LENGTH_BINS:
    bin_results = all_checkpoint[length_bin]['results']
    results_by_bin[length_bin] = bin_results
    n = len(bin_results)
    mean_tokens = np.mean([r['padded_tokens'] for r in bin_results])

    # Extract NLL arrays
    bare = np.array([r['nll_bare'] for r in bin_results])
    oracle_trunc = np.array([r['nll_oracle_trunc'] for r in bin_results])
    surr_doc = np.array([r['nll_surr_doc_trunc'] for r in bin_results])
    random_trunc = np.array([r['nll_random_trunc'] for r in bin_results])
    bare_nq = np.array([r['nll_bare_nq'] for r in bin_results])
    oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in bin_results])

    print(f"\n--- {length_bin} (mean {mean_tokens:.0f} tokens, N={n}) ---")

    # Query-in-decoder conditions
    print(f"  With query in decoder:")
    print(f"  {'Condition':<20} {'NLL':>8} {'vs bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
    print(f"  {'-'*74}")

    analysis[length_bin] = {'mean_tokens': float(mean_tokens)}
    for name, nlls in [('bare', bare), ('oracle_trunc', oracle_trunc),
                        ('surr_doc_trunc', surr_doc), ('random_trunc', random_trunc)]:
        mean_nll = nlls.mean()
        if name == 'bare':
            print(f"  {name:<20} {mean_nll:>8.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
            analysis[length_bin][name] = {'mean_nll': float(mean_nll)}
        else:
            diff = bare - nlls
            d = cohens_d(diff)
            win_pct = 100 * np.mean(diff > 0)
            _, p_val = stats.ttest_1samp(diff, 0)
            sig = '***' if p_val < 0.001/N_BONF else '**' if p_val < 0.01/N_BONF else '*' if p_val < 0.05/N_BONF else 'ns'
            print(f"  {name:<20} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
            analysis[length_bin][name] = {
                'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
                'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            }

    # No-query conditions
    diff_nq = bare_nq - oracle_nq
    d_nq = cohens_d(diff_nq)
    win_nq = 100 * np.mean(diff_nq > 0)
    _, p_nq = stats.ttest_1samp(diff_nq, 0)
    sig_nq = '***' if p_nq < 0.001/N_BONF else '**' if p_nq < 0.01/N_BONF else '*' if p_nq < 0.05/N_BONF else 'ns'

    print(f"\n  Without query in decoder (v3 replication):")
    print(f"  {'bare_nq':<20} {bare_nq.mean():>8.4f}")
    print(f"  {'oracle_trunc_nq':<20} {oracle_nq.mean():>8.4f} {diff_nq.mean():>+10.4f} {d_nq:>+8.3f} {win_nq:>7.1f}% {p_nq:>12.2e} {sig_nq:>5}")

    analysis[length_bin]['bare_nq'] = {'mean_nll': float(bare_nq.mean())}
    analysis[length_bin]['oracle_trunc_nq'] = {
        'mean_nll': float(oracle_nq.mean()), 'delta': float(diff_nq.mean()),
        'd': float(d_nq), 'win_pct': float(win_nq), 'p': float(p_nq),
    }
""")


# ===== Cell 9: Decay curve analysis =====
code(r"""# Cell 9: Decay curve analysis
print("=" * 70)
print("DECAY CURVE ANALYSIS")
print("=" * 70)

# v3 Exp 03 reference data (from EXPERIMENT_PLAN.md)
v3_oracle_d = {
    "original": 0.41, "256": 0.42, "512": 0.38,
    "1024": 0.45, "2048": 0.42, "4096": 0.40,
}

# v4 Exp 01 reference
v4_exp01_oracle_d = 0.228  # at original length

# Collect d values
print(f"\n--- Cohen's d vs Length (with query in decoder) ---")
print(f"  {'Length':<10} {'oracle':>10} {'surr_doc':>10} {'random':>10} {'tokens':>8} {'v3 oracle':>10} {'v4/v3':>8}")
print(f"  {'-'*72}")

decay_data = {
    'length_bin': [], 'mean_tokens': [],
    'd_oracle': [], 'd_surr_doc': [], 'd_random': [],
    'd_oracle_nq': [], 'd_v3_oracle': [],
}

for lb in LENGTH_BINS:
    mean_tok = analysis[lb]['mean_tokens']
    d_orc = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    d_doc = analysis[lb].get('surr_doc_trunc', {}).get('d', 0)
    d_rnd = analysis[lb].get('random_trunc', {}).get('d', 0)
    d_nq = analysis[lb].get('oracle_trunc_nq', {}).get('d', 0)
    v3_d = v3_oracle_d.get(lb, 0)

    decay_data['length_bin'].append(lb)
    decay_data['mean_tokens'].append(mean_tok)
    decay_data['d_oracle'].append(d_orc)
    decay_data['d_surr_doc'].append(d_doc)
    decay_data['d_random'].append(d_rnd)
    decay_data['d_oracle_nq'].append(d_nq)
    decay_data['d_v3_oracle'].append(v3_d)

    ratio_v3 = d_orc / v3_d * 100 if v3_d > 0 else 0
    print(f"  {lb:<10} {d_orc:>+10.3f} {d_doc:>+10.3f} {d_rnd:>+10.3f} {mean_tok:>7.0f} {v3_d:>+10.3f} {ratio_v3:>7.0f}%")

# Decay rate
print(f"\n--- Retention vs Original Length ---")
orig_d = analysis["original"].get('oracle_trunc', {}).get('d', 0)
for cond in ['oracle_trunc', 'surr_doc_trunc', 'random_trunc']:
    orig_c = analysis["original"].get(cond, {}).get('d', 0)
    if orig_c == 0:
        continue
    print(f"\n  {cond}:")
    for lb in LENGTH_BINS:
        d = analysis[lb].get(cond, {}).get('d', 0)
        p = analysis[lb].get(cond, {}).get('p', 1)
        retention = d / orig_c * 100 if orig_c > 0 else 0
        sig = '***' if p < 0.001/N_BONF else '**' if p < 0.01/N_BONF else '*' if p < 0.05/N_BONF else 'ns'
        print(f"    {lb:>8s}: d={d:+.3f} ({retention:5.1f}% of original) {sig}")

# v3 replication: should show flat curve
print(f"\n--- v3 Replication (no query in decoder) ---")
print(f"  {'Length':<10} {'d (nq)':>10} {'v3 ref':>10} {'ratio':>8}")
print(f"  {'-'*42}")
for i, lb in enumerate(LENGTH_BINS):
    d_nq = decay_data['d_oracle_nq'][i]
    v3_d = decay_data['d_v3_oracle'][i]
    ratio = d_nq / v3_d * 100 if v3_d > 0 else 0
    print(f"  {lb:<10} {d_nq:>+10.3f} {v3_d:>+10.3f} {ratio:>7.0f}%")

# Structural fraction at each length
print(f"\n--- Structural Fraction (random/oracle) at Each Length ---")
print(f"  {'Length':<10} {'oracle':>10} {'random':>10} {'struct%':>10}")
print(f"  {'-'*44}")
for i, lb in enumerate(LENGTH_BINS):
    d_orc = decay_data['d_oracle'][i]
    d_rnd = decay_data['d_random'][i]
    frac = d_rnd / d_orc * 100 if d_orc > 0 else 0
    print(f"  {lb:<10} {d_orc:>+10.3f} {d_rnd:>+10.3f} {frac:>9.0f}%")
""")


# ===== Cell 10: Plot =====
code(r"""# Cell 10: Decay curve plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x_tokens = decay_data['mean_tokens']

# --- Panel 1: v4 decay curves (with query in decoder) ---
ax = axes[0]
for cond, d_key, color, marker in [
    ('oracle_trunc', 'd_oracle', 'tab:red', 'o'),
    ('surr_doc_trunc', 'd_surr_doc', 'tab:blue', 's'),
    ('random_trunc', 'd_random', 'tab:gray', '^'),
]:
    d_vals = decay_data[d_key]
    ax.plot(x_tokens, d_vals, f'-{marker}', color=color, label=cond, markersize=8)
    for j, (x, d) in enumerate(zip(x_tokens, d_vals)):
        p = analysis[LENGTH_BINS[j]].get(cond, {}).get('p', 1)
        if p < 0.05 / N_BONF:
            ax.annotate('*', (x, d), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=14, color=color)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('v4: Effect Size vs Length\n(query in decoder)')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Panel 2: v4 oracle vs v3 oracle ---
ax = axes[1]
ax.plot(x_tokens, decay_data['d_oracle'], '-o', color='tab:red',
        label='v4 oracle (query in decoder)', markersize=8)
ax.plot(x_tokens, decay_data['d_oracle_nq'], '-s', color='tab:orange',
        label='v4 oracle_nq (v3 replication)', markersize=8)
ax.plot(x_tokens, decay_data['d_v3_oracle'], '--^', color='tab:purple',
        label='v3 Exp 03 oracle (reference)', markersize=8, alpha=0.7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('Oracle Enrichment: v4 vs v3\nacross lengths')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Panel 3: v4/v3 ratio across lengths ---
ax = axes[2]
ratios = [d4/d3 * 100 if d3 > 0 else 0
          for d4, d3 in zip(decay_data['d_oracle'], decay_data['d_v3_oracle'])]
ax.plot(x_tokens, ratios, '-o', color='tab:green', markersize=8)
ax.axhline(y=60.6, color='tab:red', linestyle='--', alpha=0.5,
           label=f'Exp 01 ratio: 60.6%')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel('v4/v3 enrichment ratio (%)')
ax.set_title('Enrichment Preservation\n(v4 oracle / v3 oracle)')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.set_ylim(0, 120)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / 'decay_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")
""")


# ===== Cell 11: Verdict + save =====
code(r"""# Cell 11: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 02: Length Scaling with Query in Decoder")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {N_SAMPLES} samples per length bin")
print(f"Length bins: {LENGTH_BINS}")
print(f"Bonferroni correction: {N_BONF} comparisons")

# Key question: does the v4 benefit decay?
print(f"\n--- Oracle Enrichment Decay (with query in decoder) ---")
orig_d = analysis["original"].get('oracle_trunc', {}).get('d', 0)
last_sig = "none"
for lb in LENGTH_BINS:
    d = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    p = analysis[lb].get('oracle_trunc', {}).get('p', 1)
    sig = p < 0.05 / N_BONF
    retention = d / orig_d * 100 if orig_d > 0 else 0
    sig_str = "SIG" if sig else "ns"
    print(f"  {lb:>8s}: d={d:+.3f} ({retention:5.1f}% retained) [{sig_str}]")
    if sig:
        last_sig = lb

print(f"\n  Last significant bin: {last_sig}")

# Compare patterns
print(f"\n--- Pattern Summary ---")
nq_flat = all(
    analysis[lb].get('oracle_trunc_nq', {}).get('p', 1) < 0.05 / N_BONF
    for lb in LENGTH_BINS
)
v4_decays = not all(
    analysis[lb].get('oracle_trunc', {}).get('p', 1) < 0.05 / N_BONF
    for lb in LENGTH_BINS
)

if nq_flat and not v4_decays:
    print("  v3 replication: FLAT (all lengths significant) -- matches v3 Exp 03")
    print("  v4 enrichment:  FLAT (all lengths significant)")
    print("  -> Content-dependent enrichment is ALSO length-invariant.")
    print("     The mechanism works at all scales.")
elif nq_flat and v4_decays:
    print("  v3 replication: FLAT (all lengths significant) -- matches v3 Exp 03")
    print("  v4 enrichment:  DECAYS (some lengths non-significant)")
    print(f"  -> Content-dependent enrichment FADES at longer documents.")
    print(f"     Last significant: {last_sig}. This limits production applicability.")
else:
    print("  v3 replication: UNEXPECTED (check results)")

# Surrogate performance across lengths
print(f"\n--- Surrogate vs Oracle at Each Length ---")
print(f"  {'Length':<10} {'oracle d':>10} {'surr_doc d':>10} {'surr/oracle':>12}")
print(f"  {'-'*46}")
for lb in LENGTH_BINS:
    od = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    sd = analysis[lb].get('surr_doc_trunc', {}).get('d', 0)
    ratio = sd / od * 100 if od > 0 else 0
    print(f"  {lb:<10} {od:>+10.3f} {sd:>+10.3f} {ratio:>11.0f}%")

# Structural fraction trend
print(f"\n--- Structural Fraction Trend ---")
struct_fracs = []
for lb in LENGTH_BINS:
    od = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    rd = analysis[lb].get('random_trunc', {}).get('d', 0)
    frac = rd / od * 100 if od > 0 else 0
    struct_fracs.append(frac)
    print(f"  {lb:<10}: {frac:.0f}%")

mean_struct = np.mean(struct_fracs)
print(f"\n  Mean structural fraction: {mean_struct:.0f}%")
if mean_struct < 50:
    print(f"  Content-dependent mechanism dominates at all lengths.")
else:
    print(f"  Structural component grows with length (decoder query less effective at long range).")

# Save
final_results = {
    'experiment': 'v4_exp02_length_scaling',
    'model': MODEL_NAME,
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'length_bins': LENGTH_BINS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'bonferroni': N_BONF,
    'analysis': analysis,
    'decay_data': decay_data,
    'padded_stats': padded_stats,
    'v3_reference': v3_oracle_d,
    'v4_exp01_reference': v4_exp01_oracle_d,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
""")


# ===== Cell 12: Cleanup =====
code(r"""# Cell 12: Cleanup
print("Cleaning up GPU memory...")
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
out_path = "experiments/encoder_decoder/02/02_length_scaling.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
