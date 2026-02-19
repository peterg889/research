#!/usr/bin/env python3
# Build Exp 01 notebook: Production-Realistic KV Cache Experiment.
#
# v3 proved co-encoding [surrogate + document] enriches doc representations (d=+0.408).
# But the decoder never saw the query. In production:
#   1. Offline: encode [surrogate + document] -> cache encoder states
#   2. Online: query arrives -> decoder receives query, cross-attends to cache
#
# This experiment: does surrogate enrichment still help when decoder has the query?
# If yes: real production value. If no: enrichment is redundant.
#
# 8 conditions: 6 with query in decoder, 2 without (replicates v3 Exp 01).
# N=500, MS MARCO v1.1 validation.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 01: Production-Realistic KV Cache

## Motivation

v3 Experiment 01 proved that co-encoding [surrogate + document] enriches document
representations (d=+0.408). But the decoder never saw the query — it only scored
answer NLL from encoder states alone. This doesn't model any real production system.

In production:
1. **Offline**: Encode [surrogate + document] → cache encoder hidden states
2. **Online**: Query arrives → decoder receives query as input, cross-attends to cached encoder states

**The key question**: Does surrogate-enriched encoder caching still help when the
decoder already has the query? If the decoder knowing the query makes enrichment
redundant, the approach has no production value.

## Method

`T5Gemma2ForConditionalGeneration.forward()` accepts explicit `decoder_input_ids`
alongside `encoder_outputs`. We build `decoder_input_ids = [BOS] + query_tokens + answer_tokens`
and compute NLL only on the answer token positions.

## Conditions (8 total)

### With query in decoder (production-realistic):

| # | Condition | Encoder input | Cross-attn | Decoder input |
|---|-----------|--------------|------------|---------------|
| 1 | bare | [document] | all | [query] → answer |
| 2 | oracle_trunc | [query + document] | doc only | [query] → answer |
| 3 | oracle_full | [query + document] | all | [query] → answer |
| 4 | surr_template_trunc | ["What is [kw]?" + doc] | doc only | [query] → answer |
| 5 | surr_doc_trunc | [top-5 kw + document] | doc only | [query] → answer |
| 6 | random_trunc | [random words + doc] | doc only | [query] → answer |

### Without query in decoder (replicates v3 Exp 01):

| # | Condition | Encoder input | Cross-attn | Decoder input |
|---|-----------|--------------|------------|---------------|
| 7 | bare_nq | [document] | all | answer only |
| 8 | oracle_trunc_nq | [query + document] | doc only | answer only |

## Key comparisons

- **(2) vs (1)**: Does enrichment help when decoder already has query? (**THE** question)
- **(8) vs (7)**: Replicates v3 Exp 01 finding (expected d≈+0.4)
- **(2)−(1) vs (8)−(7)**: Is enrichment redundant once decoder has query?
- **(4) vs (1)**: Production-realistic surrogate benefit with query in decoder
- **(3) vs (2)**: Does full cross-attention add value beyond enriched doc reps?""")


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

sys.path.insert(0, "../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../results/exp01")
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

print(f"Exp 01: Production-Realistic KV Cache")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Decoder start token ID (BOS): {BOS_ID}")
""")


# ===== Cell 3: Scoring helpers =====
code(r"""# Cell 3: Scoring helpers

def count_prefix_tokens(prefix_text, document_text):
    # BPE-aware token count of prefix in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer tokens — decoder receives ONLY answer (no query).
    # Used for _nq (no-query) conditions that replicate v3 Exp 01.
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
    # Score NLL of answer tokens with query as decoder prefix.
    # Production-realistic: decoder_input_ids = [BOS] + query_ids + answer_ids.
    # NLL is computed only on answer tokens.
    #
    # Args:
    #   encoder_text: Text for encoder (e.g., "[prefix]\n[document]" or "[document]")
    #   query_text: Query text fed to decoder as prefix
    #   answer_text: Answer text whose NLL we measure
    #   prefix_token_count: Number of encoder prefix tokens to potentially mask
    #   truncate: If True, mask prefix tokens from decoder cross-attention

    # 1. Encode
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
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

    # 5. Forward pass (no labels — we compute NLL manually)
    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=dec_tensor,
        )

    # 6. Extract answer logits
    # logits[0, t, :] predicts the token at position t+1
    # Positions: [BOS=0, q1=1, ..., qK=K, a1=K+1, ..., aM=K+M]
    # To predict a1 at position K+1, use logits[0, K, :]
    # To predict aM at position K+M, use logits[0, K+M-1, :]
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

print("Scoring functions defined:")
print("  score_nll(encoder_text, answer_text, prefix_token_count, truncate)")
print("  score_nll_query_prefix(encoder_text, query_text, answer_text, prefix_token_count, truncate)")

# Sanity check: verify BOS_ID and decoder behavior
test_q_ids = tokenizer("What is Python?", add_special_tokens=False).input_ids
test_a_ids = tokenizer("A programming language.", add_special_tokens=False).input_ids
print(f"\nSanity check:")
print(f"  BOS_ID: {BOS_ID} (token: '{tokenizer.decode([BOS_ID])}')")
print(f"  Query 'What is Python?' -> {len(test_q_ids)} tokens")
print(f"  Answer 'A programming language.' -> {len(test_a_ids)} tokens")
print(f"  Decoder input length: 1 + {len(test_q_ids)} + {len(test_a_ids)} = {1 + len(test_q_ids) + len(test_a_ids)}")
""")


# ===== Cell 4: Load data + surrogates =====
code(r"""# Cell 4: Load MS MARCO data and generate surrogates
from lib.data import count_words
from datasets import load_dataset

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

# Generate surrogates for each sample
for i, s in enumerate(samples):
    s['surr_template'] = make_surrogate_template(s['passage'])
    s['surr_doc'] = make_surrogate_from_doc(s['passage'])

    # Random prefix: words from unrelated passage, matched to query word count
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    query_word_count = len(s['query'].split())
    s['random_prefix'] = " ".join(other_words[:query_word_count])

    # Count prefix tokens for each condition
    s['n_prefix_oracle'] = count_prefix_tokens(s['query'], s['passage'])
    s['n_prefix_template'] = count_prefix_tokens(s['surr_template'], s['passage'])
    s['n_prefix_doc'] = count_prefix_tokens(s['surr_doc'], s['passage'])
    s['n_prefix_random'] = count_prefix_tokens(s['random_prefix'], s['passage'])

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")

print(f"\nFirst sample:")
print(f"  Query:  {samples[0]['query'][:70]}...")
print(f"  Answer: {samples[0]['answer'][:70]}...")
print(f"  Passage ({samples[0]['word_count']}w): {samples[0]['passage'][:70]}...")
""")


# ===== Cell 5: Show example conditions =====
code(r"""# Cell 5: Show example conditions for sample 0
print("=" * 70)
print("EXAMPLE CONDITIONS (sample 0)")
print("=" * 70)

ex = samples[0]
print(f"\nQuery:          {ex['query']}")
print(f"Answer:         {ex['answer']}")
print(f"Passage:        {ex['passage'][:100]}...")
print(f"Surr template:  {ex['surr_template']}")
print(f"Surr doc kw:    {ex['surr_doc']}")
print(f"Random prefix:  {ex['random_prefix'][:60]}...")

print(f"\n  {'Condition':<25} {'Enc prefix':<30} {'Trunc':>6} {'Dec query':>10}")
print(f"  {'-'*75}")

cond_display = [
    ('bare',                '(none)',               'no',  'yes'),
    ('oracle_trunc',        'real query',           'yes', 'yes'),
    ('oracle_full',         'real query',           'no',  'yes'),
    ('surr_template_trunc', ex['surr_template'],    'yes', 'yes'),
    ('surr_doc_trunc',      ex['surr_doc'][:28],    'yes', 'yes'),
    ('random_trunc',        ex['random_prefix'][:28], 'yes', 'yes'),
    ('bare_nq',             '(none)',               'no',  'no'),
    ('oracle_trunc_nq',     'real query',           'yes', 'no'),
]

for name, prefix, trunc, has_q in cond_display:
    print(f"  {name:<25} {prefix:<30} {trunc:>6} {has_q:>10}")

# Show decoder input structure
q_ids = tokenizer(ex['query'], add_special_tokens=False).input_ids
a_ids = tokenizer(ex['answer'], add_special_tokens=False).input_ids
print(f"\nDecoder input (query-prefix conditions):")
print(f"  [BOS] + query ({len(q_ids)} tok) + answer ({len(a_ids)} tok) "
      f"= {1 + len(q_ids) + len(a_ids)} tok total")
print(f"  NLL computed on last {len(a_ids)} positions (answer only)")

print(f"\nDecoder input (no-query conditions):")
print(f"  Model creates [BOS, a1, ..., a_{{M-1}}] internally from labels")
print(f"  NLL computed on all {len(a_ids)} answer token positions")

# Quick sanity: compare query-prefix NLL vs no-query NLL for bare
print(f"\nSanity check: bare NLL with and without query in decoder...")
nll_bare_q = score_nll_query_prefix(ex['passage'], ex['query'], ex['answer'])
nll_bare_nq = score_nll(ex['passage'], ex['answer'])
print(f"  bare (with query in decoder):    {nll_bare_q:.6f}")
print(f"  bare_nq (no query in decoder):   {nll_bare_nq:.6f}")
print(f"  Difference: {nll_bare_nq - nll_bare_q:+.6f}")
print(f"  (Expected: query prefix should lower NLL substantially)")
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 8 conditions x 500 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'oracle_full',
    'surr_template_trunc', 'surr_doc_trunc', 'random_trunc',
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
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {len(COND_NAMES) * N_SAMPLES} forward passes")

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
        'passage_words': s['word_count'],
    }

    # --- Conditions 1-6: query in decoder (production-realistic) ---

    # 1. bare: encoder=[doc], cross-attn=all, decoder=[query]->answer
    result['nll_bare'] = score_nll_query_prefix(
        passage, query, answer)

    # 2. oracle_trunc: encoder=[query+doc], cross-attn=doc only, decoder=[query]->answer
    result['nll_oracle_trunc'] = score_nll_query_prefix(
        query + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_oracle'], truncate=True)

    # 3. oracle_full: encoder=[query+doc], cross-attn=all, decoder=[query]->answer
    result['nll_oracle_full'] = score_nll_query_prefix(
        query + "\n" + passage, query, answer)

    # 4. surr_template_trunc: encoder=["What is [kw]?"+doc], decoder=[query]->answer
    result['nll_surr_template_trunc'] = score_nll_query_prefix(
        s['surr_template'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_template'], truncate=True)

    # 5. surr_doc_trunc: encoder=[top5kw+doc], decoder=[query]->answer
    result['nll_surr_doc_trunc'] = score_nll_query_prefix(
        s['surr_doc'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_doc'], truncate=True)

    # 6. random_trunc: encoder=[random+doc], decoder=[query]->answer
    result['nll_random_trunc'] = score_nll_query_prefix(
        s['random_prefix'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_prefix_random'], truncate=True)

    # --- Conditions 7-8: no query in decoder (replicates v3 Exp 01) ---

    # 7. bare_nq: encoder=[doc], decoder=answer only
    result['nll_bare_nq'] = score_nll(passage, answer)

    # 8. oracle_trunc_nq: encoder=[query+doc], decoder=answer only (masked)
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
code(r"""# Cell 7: Results table — all conditions
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

# Extract NLL arrays — query-prefix conditions
bare = np.array([r['nll_bare'] for r in results])
oracle_trunc = np.array([r['nll_oracle_trunc'] for r in results])
oracle_full = np.array([r['nll_oracle_full'] for r in results])
surr_template = np.array([r['nll_surr_template_trunc'] for r in results])
surr_doc = np.array([r['nll_surr_doc_trunc'] for r in results])
random_trunc = np.array([r['nll_random_trunc'] for r in results])

# No-query conditions
bare_nq = np.array([r['nll_bare_nq'] for r in results])
oracle_trunc_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])

print(f"\n--- Query in decoder (production-realistic) ---")
print(f"  Baseline: bare (decoder has query, encoder has document only)")
print(f"\n  {'Condition':<25} {'NLL':>8} {'vs bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*78}")

analysis = {}
for name, nlls in [
    ('bare', bare),
    ('oracle_trunc', oracle_trunc),
    ('oracle_full', oracle_full),
    ('surr_template_trunc', surr_template),
    ('surr_doc_trunc', surr_doc),
    ('random_trunc', random_trunc),
]:
    mean_nll = nlls.mean()
    if name == 'bare':
        print(f"  {name:<25} {mean_nll:>8.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls  # positive = condition is better (lower NLL)
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {name:<25} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

print(f"\n--- No query in decoder (v3 Exp 01 replication) ---")
print(f"  Baseline: bare_nq (decoder has answer only)")
print(f"\n  {'Condition':<25} {'NLL':>8} {'vs bare_nq':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*78}")

diff_nq = bare_nq - oracle_trunc_nq
d_nq = cohens_d(diff_nq)
win_nq = 100 * np.mean(diff_nq > 0)
_, p_nq = stats.ttest_1samp(diff_nq, 0)
sig_nq = '***' if p_nq < 0.001 else '**' if p_nq < 0.01 else '*' if p_nq < 0.05 else 'ns'

print(f"  {'bare_nq':<25} {bare_nq.mean():>8.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
print(f"  {'oracle_trunc_nq':<25} {oracle_trunc_nq.mean():>8.4f} {diff_nq.mean():>+10.4f} {d_nq:>+8.3f} {win_nq:>7.1f}% {p_nq:>12.2e} {sig_nq:>5}")

analysis['bare_nq'] = {'mean_nll': float(bare_nq.mean())}
analysis['oracle_trunc_nq'] = {
    'mean_nll': float(oracle_trunc_nq.mean()), 'delta': float(diff_nq.mean()),
    'd': float(d_nq), 'win_pct': float(win_nq), 'p': float(p_nq),
}

print(f"\nExpected: oracle_trunc_nq vs bare_nq should replicate d~+0.4 from v3 Exp 01")
print(f"Actual:   d={d_nq:+.3f} ({sig_nq})")
""")


# ===== Cell 8: Key comparison =====
code(r"""# Cell 8: Key comparison — enrichment with query vs without query
print("=" * 70)
print("KEY COMPARISON: Is enrichment redundant when decoder has the query?")
print("=" * 70)

# Enrichment benefit WITH query in decoder
enrichment_with_q = bare - oracle_trunc  # positive = enrichment helps
d_with_q = cohens_d(enrichment_with_q)
_, p_with_q = stats.ttest_1samp(enrichment_with_q, 0)
sig_with_q = '***' if p_with_q < 0.001 else '**' if p_with_q < 0.01 else '*' if p_with_q < 0.05 else 'ns'

# Enrichment benefit WITHOUT query in decoder (v3 replication)
enrichment_no_q = bare_nq - oracle_trunc_nq  # positive = enrichment helps
d_no_q = cohens_d(enrichment_no_q)
_, p_no_q = stats.ttest_1samp(enrichment_no_q, 0)
sig_no_q = '***' if p_no_q < 0.001 else '**' if p_no_q < 0.01 else '*' if p_no_q < 0.05 else 'ns'

print(f"\n  Enrichment benefit (oracle_trunc vs respective bare):")
print(f"    With query in decoder:    d={d_with_q:+.4f} ({sig_with_q}), "
      f"mean delta={enrichment_with_q.mean():+.4f}")
print(f"    Without query in decoder: d={d_no_q:+.4f} ({sig_no_q}), "
      f"mean delta={enrichment_no_q.mean():+.4f}")

# Are they different?
diff_of_diffs = enrichment_with_q - enrichment_no_q
d_diff = cohens_d(diff_of_diffs)
_, p_diff = stats.ttest_1samp(diff_of_diffs, 0)
sig_diff = '***' if p_diff < 0.001 else '**' if p_diff < 0.01 else '*' if p_diff < 0.05 else 'ns'

# Ratio
if d_no_q > 0:
    ratio = d_with_q / d_no_q * 100
else:
    ratio = float('inf')

print(f"\n  Ratio: with_q / without_q = {ratio:.1f}%")
print(f"  Difference of differences: d={d_diff:+.4f} ({sig_diff})")

print(f"\n  Interpretation:")
if d_with_q > 0.1 and ratio > 50:
    print(f"    ENRICHMENT STILL HELPS even when decoder has the query.")
    print(f"    The approach has genuine production value.")
    if ratio > 80:
        print(f"    Enrichment benefit is {ratio:.0f}% of the no-query benefit — mostly preserved.")
    else:
        print(f"    Enrichment benefit is {ratio:.0f}% of the no-query benefit — partially redundant.")
elif d_with_q > 0.05:
    print(f"    WEAK enrichment benefit with query. The decoder knowing the query")
    print(f"    makes most of the enrichment redundant.")
else:
    print(f"    ENRICHMENT IS REDUNDANT when the decoder has the query.")
    print(f"    The whole approach has no production value.")

# Full cross-attention vs truncation (with query)
print(f"\n--- Full vs truncated cross-attention (with query in decoder) ---")
diff_full_trunc = oracle_trunc - oracle_full  # negative = full is better
d_ft = cohens_d(diff_full_trunc)
_, p_ft = stats.ttest_1samp(diff_full_trunc, 0)
sig_ft = '***' if p_ft < 0.001 else '**' if p_ft < 0.01 else '*' if p_ft < 0.05 else 'ns'

print(f"  oracle_trunc NLL: {oracle_trunc.mean():.4f}")
print(f"  oracle_full NLL:  {oracle_full.mean():.4f}")
print(f"  full vs trunc: d={d_ft:+.4f} ({sig_ft})")
if abs(d_ft) < 0.05:
    print(f"  -> Full cross-attention adds minimal value beyond enriched doc reps.")
elif d_ft < -0.05:
    print(f"  -> Full cross-attention substantially better — decoder benefits from")
    print(f"     reading query directly from encoder output as well.")
else:
    print(f"  -> Truncation is actually BETTER — same pattern as v3.")

# Surrogate conditions vs bare (with query)
print(f"\n--- Surrogate benefit with query in decoder ---")
for name, nlls in [('surr_template_trunc', surr_template),
                     ('surr_doc_trunc', surr_doc),
                     ('random_trunc', random_trunc)]:
    diff = bare - nlls
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    if d_with_q > 0:
        pct_oracle = d / d_with_q * 100
    else:
        pct_oracle = 0
    print(f"  {name:<25} d={d:+.4f} ({sig}), {pct_oracle:.0f}% of oracle")

# Per-sample correlation: does the same enrichment help the same samples?
r_corr, p_corr = stats.pearsonr(enrichment_with_q, enrichment_no_q)
print(f"\nPer-sample correlation (enrichment with q vs without q): r={r_corr:.3f} (p={p_corr:.2e})")
print(f"  (High r = same samples benefit from enrichment regardless of decoder query)")
""")


# ===== Cell 9: Hardness gradient =====
code(r"""# Cell 9: Hardness gradient — does enrichment benefit vary with difficulty?
print("=" * 70)
print("HARDNESS GRADIENT")
print("=" * 70)

# Use bare NLL as hardness proxy
quintile_bounds = np.percentile(bare, [20, 40, 60, 80])
quintiles = np.digitize(bare, quintile_bounds)

print(f"\n--- Enrichment benefit by hardness (with query in decoder) ---")
print(f"  {'Quintile':<12} {'N':>4} {'bare':>8} {'orc_trunc':>10} {'delta':>8} {'d':>8}")
print(f"  {'-'*55}")

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 5:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare[mask].mean()
    ot = oracle_trunc[mask].mean()
    delta = (bare[mask] - oracle_trunc[mask]).mean()
    d = cohens_d(bare[mask] - oracle_trunc[mask])
    print(f"  {qlabel:<12} {n_q:>4} {b:>8.4f} {ot:>10.4f} {delta:>+8.4f} {d:>+8.3f}")

# Same for no-query
print(f"\n--- Enrichment benefit by hardness (no query in decoder) ---")
quintile_bounds_nq = np.percentile(bare_nq, [20, 40, 60, 80])
quintiles_nq = np.digitize(bare_nq, quintile_bounds_nq)

print(f"  {'Quintile':<12} {'N':>4} {'bare_nq':>8} {'orc_tr_nq':>10} {'delta':>8} {'d':>8}")
print(f"  {'-'*55}")

for q in range(5):
    mask = quintiles_nq == q
    n_q = mask.sum()
    if n_q < 5:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare_nq[mask].mean()
    ot = oracle_trunc_nq[mask].mean()
    delta = (bare_nq[mask] - oracle_trunc_nq[mask]).mean()
    d = cohens_d(bare_nq[mask] - oracle_trunc_nq[mask])
    print(f"  {qlabel:<12} {n_q:>4} {b:>8.4f} {ot:>10.4f} {delta:>+8.4f} {d:>+8.3f}")

# Spearman correlations with hardness
r_hard, p_hard = stats.spearmanr(bare, bare - oracle_trunc)
print(f"\nSpearman correlation (hardness vs enrichment with query): "
      f"rho={r_hard:.3f} (p={p_hard:.2e})")
r_hard_nq, p_hard_nq = stats.spearmanr(bare_nq, bare_nq - oracle_trunc_nq)
print(f"Spearman correlation (hardness vs enrichment no query):   "
      f"rho={r_hard_nq:.3f} (p={p_hard_nq:.2e})")
""")


# ===== Cell 10: Verdict + save =====
code(r"""# Cell 10: Verdict and save results
print("=" * 70)
print("VERDICT -- Exp 01: Production-Realistic KV Cache")
print("=" * 70)

d_with_q = cohens_d(bare - oracle_trunc)
d_no_q = cohens_d(bare_nq - oracle_trunc_nq)
ratio = d_with_q / d_no_q * 100 if d_no_q > 0 else float('inf')

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")

print(f"\n--- THE key result ---")
print(f"  Enrichment without query in decoder (v3 baseline): d={d_no_q:+.3f}")
print(f"  Enrichment WITH query in decoder (production):     d={d_with_q:+.3f}")
print(f"  Ratio: {ratio:.0f}%")

if d_with_q > 0.1:
    print(f"\n  CONCLUSION: Encoder enrichment provides genuine value")
    print(f"  even in production where the decoder already has the query.")
    if ratio > 80:
        print(f"  The enrichment benefit is almost fully preserved ({ratio:.0f}%).")
    elif ratio > 50:
        print(f"  About {ratio:.0f}% of the enrichment benefit survives.")
    else:
        print(f"  Only {ratio:.0f}% survives -- most benefit was query-reading.")
elif d_with_q > 0.05:
    print(f"\n  CONCLUSION: Marginal benefit. The decoder knowing the query")
    print(f"  makes most enrichment redundant. Production value is limited.")
else:
    print(f"\n  CONCLUSION: No production value. The enrichment benefit")
    print(f"  vanishes when the decoder already has the query.")
    print(f"  The v3 benefit was primarily from the decoder reading the query")
    print(f"  from encoder output, not from improved document representations.")

# Surrogate summary
print(f"\n--- Surrogate summary (with query in decoder) ---")
for name in ['surr_template_trunc', 'surr_doc_trunc', 'random_trunc']:
    nlls = np.array([r[f'nll_{name}'] for r in results])
    d = cohens_d(bare - nlls)
    _, p = stats.ttest_1samp(bare - nlls, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<25} d={d:+.4f} ({sig})")

# Save
final_results = {
    'experiment': 'v4_exp01_production_kv_cache',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'key_result': {
        'enrichment_with_query_d': float(d_with_q),
        'enrichment_no_query_d': float(d_no_q),
        'ratio_pct': float(ratio),
    },
    'conditions': {k: v for k, v in analysis.items()},
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
out_path = "experiments/01/01_production_kv_cache.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
