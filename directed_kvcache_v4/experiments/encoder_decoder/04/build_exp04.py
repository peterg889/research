#!/usr/bin/env python3
# Build Exp 04 notebook: Prefix Content Optimization.
#
# Exp 01 showed that on short MS MARCO docs (~98 tokens), prefix CONTENT matters:
#   - oracle d=+0.228, surr_kw5 d=+0.148 (65% oracle), random d=+0.080 (35%)
# The structural fraction collapsed from 85% (v3) to 35% (v4 with query in decoder).
#
# But Exp 02/03 showed that on longer documents (256+ tokens), the structural
# mechanism regains dominance (surrogates beat oracle on neural-bridge).
#
# This experiment focuses on the SHORT-DOC regime (MS MARCO ~98 tokens) where
# content optimization has the most potential impact. We sweep different prefix
# content types to find what maximizes enrichment.
#
# Key questions:
#   1. Does more keyword density help? (kw5 -> kw10 -> kw20)
#   2. Does natural text (first sentence) beat keyword bags?
#   3. Do document-SPECIFIC keywords matter, or do any keywords work?
#   4. Can any surrogate close the gap to oracle?
#
# 10 conditions (8 with query + 2 without), N=500.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 04: Prefix Content Optimization

## Motivation

Exp 01 (MS MARCO, ~98 token docs) showed that when the decoder already has the query,
the **content** of the encoder prefix matters:

| Condition | d vs bare | % of oracle |
|-----------|-----------|-------------|
| oracle_trunc (real query) | +0.228 *** | 100% |
| surr_doc_kw5 (top-5 keywords) | +0.148 ** | 65% |
| random_trunc (unrelated words) | +0.080 ns | 35% |

The structural fraction collapsed from 85% (v3, no query in decoder) to 35% (v4,
query in decoder). This means 65% of the benefit comes from prefix **content** — and
there's headroom between surr_kw5 (65%) and oracle (100%).

However, Exp 02/03 showed that on longer documents (256+ tokens), the structural
mechanism regains dominance and even random prefixes become highly effective. So this
content optimization is most relevant for **short documents**.

## Questions

1. Does more keyword density help? (kw5 → kw10 → kw20)
2. Does natural text (first sentence) beat keyword bags?
3. Do document-SPECIFIC keywords matter, or do any keywords work? (random_kw5 control)
4. Can any surrogate close the gap between kw5 (65%) and oracle (100%)?

## Conditions (10 total)

### With query in decoder (production-realistic):

| # | Condition | Encoder prefix | Rationale |
|---|-----------|---------------|-----------|
| 1 | bare | (none) | Baseline |
| 2 | oracle_trunc | real query | Upper bound |
| 3 | random_trunc | random unrelated words | Structural-only control |
| 4 | surr_kw5 | top-5 TF keywords | Exp 01 baseline surrogate |
| 5 | surr_kw10 | top-10 TF keywords | More keywords |
| 6 | surr_kw20 | top-20 TF keywords | Maximum keyword density |
| 7 | surr_first_sent | first sentence of doc | Natural text, high density |
| 8 | surr_random_kw5 | top-5 kw from WRONG doc | Vocabulary control |

### Without query in decoder (v3 replication):

| # | Condition | Purpose |
|---|-----------|---------|
| 9 | bare_nq | v3 baseline |
| 10 | oracle_trunc_nq | v3 enrichment reference |""")


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

RESULTS_DIR = Path("../../../results/exp04")
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

print(f"Exp 04: Prefix Content Optimization")
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

def make_kw_surrogate(passage, n_keywords):
    # Extract top-N TF keywords from passage.
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(n_keywords))

def get_first_sentence(text):
    # Extract the first sentence by splitting on sentence-ending punctuation.
    # Handle common abbreviations minimally.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if parts:
        return parts[0]
    return text[:100]

print("Scoring functions defined.")
print("Surrogate types: kw5, kw10, kw20, first_sent, random_kw5")
""")


# ===== Cell 4: Load data + generate surrogates =====
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

# Generate all surrogate types
for i, s in enumerate(samples):
    passage = s['passage']

    # Keyword surrogates at different densities
    s['surr_kw5'] = make_kw_surrogate(passage, 5)
    s['surr_kw10'] = make_kw_surrogate(passage, 10)
    s['surr_kw20'] = make_kw_surrogate(passage, 20)

    # First sentence
    s['surr_first_sent'] = get_first_sentence(passage)

    # Random words from unrelated passage (structural control, length-matched to query)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    query_word_count = len(s['query'].split())
    s['random_prefix'] = " ".join(other_words[:query_word_count])

    # Keywords from WRONG document (vocabulary control)
    wrong_idx = (i + 1) % len(samples)
    s['surr_random_kw5'] = make_kw_surrogate(samples[wrong_idx]['passage'], 5)

    # Count prefix tokens for each condition
    s['n_pfx_oracle'] = count_prefix_tokens(s['query'], passage)
    s['n_pfx_kw5'] = count_prefix_tokens(s['surr_kw5'], passage)
    s['n_pfx_kw10'] = count_prefix_tokens(s['surr_kw10'], passage)
    s['n_pfx_kw20'] = count_prefix_tokens(s['surr_kw20'], passage)
    s['n_pfx_first_sent'] = count_prefix_tokens(s['surr_first_sent'], passage)
    s['n_pfx_random'] = count_prefix_tokens(s['random_prefix'], passage)
    s['n_pfx_random_kw5'] = count_prefix_tokens(s['surr_random_kw5'], passage)

# Statistics
print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")

# Prefix token statistics
print(f"\nPrefix token counts (mean):")
for key, label in [('n_pfx_oracle', 'oracle (query)'),
                    ('n_pfx_kw5', 'kw5'), ('n_pfx_kw10', 'kw10'),
                    ('n_pfx_kw20', 'kw20'), ('n_pfx_first_sent', 'first_sent'),
                    ('n_pfx_random', 'random'), ('n_pfx_random_kw5', 'random_kw5')]:
    vals = [s[key] for s in samples]
    print(f"  {label:<18}: mean={np.mean(vals):.1f}, min={np.min(vals)}, max={np.max(vals)}")

# How many unique content keywords do passages typically have?
kw_counts = [len(set(extract_keywords(s['passage']))) for s in samples]
print(f"\nUnique content keywords per passage:")
print(f"  mean={np.mean(kw_counts):.1f}, median={np.median(kw_counts):.0f}, "
      f"min={np.min(kw_counts)}, max={np.max(kw_counts)}")

print(f"\nFirst sample:")
print(f"  Query:       {samples[0]['query'][:80]}...")
print(f"  Answer:      {samples[0]['answer'][:80]}...")
print(f"  Passage:     {samples[0]['passage'][:80]}...")
print(f"  surr_kw5:    {samples[0]['surr_kw5']}")
print(f"  surr_kw10:   {samples[0]['surr_kw10']}")
print(f"  surr_kw20:   {samples[0]['surr_kw20']}")
print(f"  first_sent:  {samples[0]['surr_first_sent'][:80]}")
print(f"  random_kw5:  {samples[0]['surr_random_kw5']}")
""")


# ===== Cell 5: Show example conditions =====
code(r"""# Cell 5: Show example conditions for sample 0
print("=" * 70)
print("EXAMPLE CONDITIONS (sample 0)")
print("=" * 70)

ex = samples[0]
print(f"\nQuery:        {ex['query']}")
print(f"Answer:       {ex['answer']}")
print(f"Passage:      {ex['passage'][:120]}...")

print(f"\n  {'Condition':<18} {'Prefix':<35} {'Tokens':>6} {'Trunc':>6} {'Dec Q':>6}")
print(f"  {'-'*75}")

conditions_display = [
    ('bare',           '(none)',                         0,                   'no',  'yes'),
    ('oracle_trunc',   ex['query'][:32],                 ex['n_pfx_oracle'],  'yes', 'yes'),
    ('random_trunc',   ex['random_prefix'][:32],         ex['n_pfx_random'],  'yes', 'yes'),
    ('surr_kw5',       ex['surr_kw5'][:32],              ex['n_pfx_kw5'],     'yes', 'yes'),
    ('surr_kw10',      ex['surr_kw10'][:32],             ex['n_pfx_kw10'],    'yes', 'yes'),
    ('surr_kw20',      ex['surr_kw20'][:32],             ex['n_pfx_kw20'],    'yes', 'yes'),
    ('surr_first_sent', ex['surr_first_sent'][:32],      ex['n_pfx_first_sent'], 'yes', 'yes'),
    ('surr_random_kw5', ex['surr_random_kw5'][:32],      ex['n_pfx_random_kw5'], 'yes', 'yes'),
    ('bare_nq',        '(none)',                         0,                   'no',  'no'),
    ('oracle_trunc_nq', ex['query'][:32],                ex['n_pfx_oracle'],  'yes', 'no'),
]

for name, prefix, n_tok, trunc, has_q in conditions_display:
    print(f"  {name:<18} {prefix:<35} {n_tok:>6} {trunc:>6} {has_q:>6}")

# Sanity check
print(f"\nSanity check...")
nll_bare = score_nll_query_prefix(ex['passage'], ex['query'], ex['answer'])
nll_oracle = score_nll_query_prefix(
    ex['query'] + "\n" + ex['passage'], ex['query'], ex['answer'],
    prefix_token_count=ex['n_pfx_oracle'], truncate=True)
nll_kw5 = score_nll_query_prefix(
    ex['surr_kw5'] + "\n" + ex['passage'], ex['query'], ex['answer'],
    prefix_token_count=ex['n_pfx_kw5'], truncate=True)
print(f"  bare:         {nll_bare:.4f}")
print(f"  oracle_trunc: {nll_oracle:.4f} (delta: {nll_bare - nll_oracle:+.4f})")
print(f"  surr_kw5:     {nll_kw5:.4f} (delta: {nll_bare - nll_kw5:+.4f})")
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 10 conditions x 500 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'random_trunc',
    'surr_kw5', 'surr_kw10', 'surr_kw20',
    'surr_first_sent', 'surr_random_kw5',
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

    # --- With query in decoder (production-realistic) ---

    # 1. bare: encoder=[doc], no prefix
    result['nll_bare'] = score_nll_query_prefix(passage, query, answer)

    # 2. oracle_trunc: encoder=[query+doc], mask query
    result['nll_oracle_trunc'] = score_nll_query_prefix(
        query + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_oracle'], truncate=True)

    # 3. random_trunc: encoder=[random+doc], mask random
    result['nll_random_trunc'] = score_nll_query_prefix(
        s['random_prefix'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_random'], truncate=True)

    # 4. surr_kw5: top-5 TF keywords
    result['nll_surr_kw5'] = score_nll_query_prefix(
        s['surr_kw5'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_kw5'], truncate=True)

    # 5. surr_kw10: top-10 TF keywords
    result['nll_surr_kw10'] = score_nll_query_prefix(
        s['surr_kw10'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_kw10'], truncate=True)

    # 6. surr_kw20: top-20 TF keywords
    result['nll_surr_kw20'] = score_nll_query_prefix(
        s['surr_kw20'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_kw20'], truncate=True)

    # 7. surr_first_sent: first sentence of document
    result['nll_surr_first_sent'] = score_nll_query_prefix(
        s['surr_first_sent'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_first_sent'], truncate=True)

    # 8. surr_random_kw5: top-5 kw from WRONG document
    result['nll_surr_random_kw5'] = score_nll_query_prefix(
        s['surr_random_kw5'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_random_kw5'], truncate=True)

    # --- Without query in decoder (v3 replication) ---

    # 9. bare_nq
    result['nll_bare_nq'] = score_nll(passage, answer)

    # 10. oracle_trunc_nq
    result['nll_oracle_trunc_nq'] = score_nll(
        query + "\n" + passage, answer,
        prefix_token_count=s['n_pfx_oracle'], truncate=True)

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

# Extract NLL arrays
bare = np.array([r['nll_bare'] for r in results])
oracle_trunc = np.array([r['nll_oracle_trunc'] for r in results])
random_trunc = np.array([r['nll_random_trunc'] for r in results])
surr_kw5 = np.array([r['nll_surr_kw5'] for r in results])
surr_kw10 = np.array([r['nll_surr_kw10'] for r in results])
surr_kw20 = np.array([r['nll_surr_kw20'] for r in results])
surr_first_sent = np.array([r['nll_surr_first_sent'] for r in results])
surr_random_kw5 = np.array([r['nll_surr_random_kw5'] for r in results])
bare_nq = np.array([r['nll_bare_nq'] for r in results])
oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])

# Bonferroni: 7 comparisons vs bare (all with-query conditions except bare)
N_BONF = 7

print(f"\n--- With query in decoder (production-realistic) ---")
print(f"  Bonferroni correction: {N_BONF} comparisons")
print(f"\n  {'Condition':<18} {'NLL':>8} {'delta':>8} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5} {'%orc':>6}")
print(f"  {'-'*82}")

analysis = {}
d_oracle = None
for name, nlls in [
    ('bare', bare),
    ('oracle_trunc', oracle_trunc),
    ('surr_kw20', surr_kw20),
    ('surr_kw10', surr_kw10),
    ('surr_kw5', surr_kw5),
    ('surr_first_sent', surr_first_sent),
    ('surr_random_kw5', surr_random_kw5),
    ('random_trunc', random_trunc),
]:
    mean_nll = nlls.mean()
    if name == 'bare':
        print(f"  {name:<18} {mean_nll:>8.4f} {'--':>8} {'--':>8} {'--':>8} {'--':>12} {'--':>5} {'--':>6}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001/N_BONF else '**' if p_val < 0.01/N_BONF else '*' if p_val < 0.05/N_BONF else 'ns'
        if name == 'oracle_trunc':
            d_oracle = d
        pct_orc = f"{d/d_oracle*100:.0f}%" if d_oracle and d_oracle > 0 else "--"
        print(f"  {name:<18} {mean_nll:>8.4f} {diff.mean():>+8.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {pct_orc:>6}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# No-query conditions
diff_nq = bare_nq - oracle_nq
d_nq = cohens_d(diff_nq)
win_nq = 100 * np.mean(diff_nq > 0)
_, p_nq = stats.ttest_1samp(diff_nq, 0)
sig_nq = '***' if p_nq < 0.001 else '**' if p_nq < 0.01 else '*' if p_nq < 0.05 else 'ns'

print(f"\n--- Without query in decoder (v3 replication) ---")
print(f"  {'bare_nq':<18} {bare_nq.mean():>8.4f}")
print(f"  {'oracle_trunc_nq':<18} {oracle_nq.mean():>8.4f} {diff_nq.mean():>+8.4f} {d_nq:>+8.3f} {win_nq:>7.1f}% {p_nq:>12.2e} {sig_nq:>5}")

analysis['bare_nq'] = {'mean_nll': float(bare_nq.mean())}
analysis['oracle_trunc_nq'] = {
    'mean_nll': float(oracle_nq.mean()), 'delta': float(diff_nq.mean()),
    'd': float(d_nq), 'win_pct': float(win_nq), 'p': float(p_nq),
}
""")


# ===== Cell 8: Content type analysis =====
code(r"""# Cell 8: Content type analysis — what type of prefix works best?
print("=" * 70)
print("CONTENT TYPE ANALYSIS")
print("=" * 70)

# Rank conditions by effect size
condition_list = [
    ('oracle_trunc', oracle_trunc),
    ('surr_kw20', surr_kw20),
    ('surr_kw10', surr_kw10),
    ('surr_kw5', surr_kw5),
    ('surr_first_sent', surr_first_sent),
    ('surr_random_kw5', surr_random_kw5),
    ('random_trunc', random_trunc),
]

ranked = []
for name, nlls in condition_list:
    d = cohens_d(bare - nlls)
    ranked.append((name, d))
ranked.sort(key=lambda x: -x[1])

print(f"\n--- Conditions ranked by effect size ---")
for rank, (name, d) in enumerate(ranked, 1):
    bar = '#' * max(0, int(d * 50))
    pct = d / d_oracle * 100 if d_oracle and d_oracle > 0 else 0
    print(f"  {rank}. {name:<18} d={d:+.3f} ({pct:5.1f}% oracle) {bar}")

# Q1: Does more keyword density help?
print(f"\n--- Q1: Keyword density sweep (kw5 -> kw10 -> kw20) ---")
d_kw5 = cohens_d(bare - surr_kw5)
d_kw10 = cohens_d(bare - surr_kw10)
d_kw20 = cohens_d(bare - surr_kw20)
print(f"  surr_kw5:  d={d_kw5:+.3f}")
print(f"  surr_kw10: d={d_kw10:+.3f}")
print(f"  surr_kw20: d={d_kw20:+.3f}")

# Direct pairwise: kw10 vs kw5
diff_10v5 = surr_kw5 - surr_kw10  # positive = kw10 is better
d_10v5 = cohens_d(diff_10v5)
_, p_10v5 = stats.ttest_1samp(diff_10v5, 0)
sig_10v5 = '***' if p_10v5 < 0.001 else '**' if p_10v5 < 0.01 else '*' if p_10v5 < 0.05 else 'ns'
print(f"  kw10 vs kw5:  d={d_10v5:+.3f} ({sig_10v5})")

# kw20 vs kw10
diff_20v10 = surr_kw10 - surr_kw20  # positive = kw20 is better
d_20v10 = cohens_d(diff_20v10)
_, p_20v10 = stats.ttest_1samp(diff_20v10, 0)
sig_20v10 = '***' if p_20v10 < 0.001 else '**' if p_20v10 < 0.01 else '*' if p_20v10 < 0.05 else 'ns'
print(f"  kw20 vs kw10: d={d_20v10:+.3f} ({sig_20v10})")

if d_kw20 > d_kw10 > d_kw5:
    print(f"  -> More keywords = better. Monotonic increase.")
elif abs(d_kw10 - d_kw5) < 0.03 and abs(d_kw20 - d_kw10) < 0.03:
    print(f"  -> Keyword count doesn't matter much. Plateau after kw5.")
else:
    print(f"  -> Non-monotonic or saturating pattern.")

# Q2: Natural text vs keyword bags
print(f"\n--- Q2: Natural text vs keyword bags ---")
d_first = cohens_d(bare - surr_first_sent)
print(f"  surr_first_sent: d={d_first:+.3f} ({d_first/d_oracle*100:.0f}% oracle)" if d_oracle else "")

diff_sent_kw5 = surr_kw5 - surr_first_sent  # positive = first_sent is better
d_sv5 = cohens_d(diff_sent_kw5)
_, p_sv5 = stats.ttest_1samp(diff_sent_kw5, 0)
sig_sv5 = '***' if p_sv5 < 0.001 else '**' if p_sv5 < 0.01 else '*' if p_sv5 < 0.05 else 'ns'
print(f"  first_sent vs kw5: d={d_sv5:+.3f} ({sig_sv5})")

if d_sv5 > 0.05:
    print(f"  -> Natural text beats keyword bags!")
elif d_sv5 < -0.05:
    print(f"  -> Keywords beat natural text.")
else:
    print(f"  -> No significant difference between natural text and keywords.")

# Q3: Document-specific vs generic keywords
print(f"\n--- Q3: Document-specific vs wrong-document keywords ---")
d_rnd_kw5 = cohens_d(bare - surr_random_kw5)
print(f"  surr_kw5 (this doc):    d={d_kw5:+.3f}")
print(f"  surr_random_kw5 (wrong): d={d_rnd_kw5:+.3f}")

diff_spec = surr_random_kw5 - surr_kw5  # positive = specific kw is better
d_spec = cohens_d(diff_spec)
_, p_spec = stats.ttest_1samp(diff_spec, 0)
sig_spec = '***' if p_spec < 0.001 else '**' if p_spec < 0.01 else '*' if p_spec < 0.05 else 'ns'
print(f"  specific vs wrong-doc kw: d={d_spec:+.3f} ({sig_spec})")

if d_spec > 0.05:
    print(f"  -> Document-specific keywords matter! Content-matching is important.")
elif d_spec < -0.05:
    print(f"  -> Wrong-doc keywords work just as well — it's about keyword-like tokens, not specificity.")
else:
    print(f"  -> No significant specificity effect — any keywords work similarly.")

# Compare random_kw5 vs random_trunc
d_random = cohens_d(bare - random_trunc)
print(f"\n  Decomposition (random_kw5 sits between random_trunc and surr_kw5):")
print(f"    random_trunc (no doc signal):   d={d_random:+.3f}")
print(f"    random_kw5 (wrong-doc keywords): d={d_rnd_kw5:+.3f}")
print(f"    surr_kw5 (this-doc keywords):   d={d_kw5:+.3f}")
vocab_component = d_rnd_kw5 - d_random
semantic_component = d_kw5 - d_rnd_kw5
total = d_kw5 - d_random
print(f"    Vocabulary component: {vocab_component:+.3f} ({vocab_component/total*100:.0f}% of kw5-random gap)" if total > 0 else "")
print(f"    Semantic component:  {semantic_component:+.3f} ({semantic_component/total*100:.0f}% of kw5-random gap)" if total > 0 else "")
""")


# ===== Cell 9: Prefix token count analysis =====
code(r"""# Cell 9: Does prefix LENGTH explain the differences, or is it content?
print("=" * 70)
print("PREFIX LENGTH ANALYSIS")
print("=" * 70)

# Collect prefix token counts and effect sizes per sample
prefix_data = {}
for name, nlls, pfx_key in [
    ('oracle_trunc', oracle_trunc, 'n_pfx_oracle'),
    ('surr_kw5', surr_kw5, 'n_pfx_kw5'),
    ('surr_kw10', surr_kw10, 'n_pfx_kw10'),
    ('surr_kw20', surr_kw20, 'n_pfx_kw20'),
    ('surr_first_sent', surr_first_sent, 'n_pfx_first_sent'),
    ('surr_random_kw5', surr_random_kw5, 'n_pfx_random_kw5'),
    ('random_trunc', random_trunc, 'n_pfx_random'),
]:
    pfx_tokens = np.array([r.get(pfx_key, s[pfx_key]) for r, s in zip(results, samples)])
    d = cohens_d(bare - nlls)
    prefix_data[name] = {
        'mean_pfx_tok': float(pfx_tokens.mean()),
        'd': float(d),
    }
    print(f"  {name:<18}: mean prefix tokens = {pfx_tokens.mean():.1f}, d = {d:+.3f}")

# Is there a correlation between prefix length and benefit across conditions?
lengths = [v['mean_pfx_tok'] for v in prefix_data.values()]
ds = [v['d'] for v in prefix_data.values()]
r_len, p_len = stats.pearsonr(lengths, ds)
print(f"\nCorrelation (mean prefix tokens vs d) across conditions:")
print(f"  r={r_len:.3f}, p={p_len:.3e}")

if abs(r_len) < 0.3:
    print(f"  -> Prefix length does NOT explain condition differences. Content matters.")
elif r_len > 0.3:
    print(f"  -> Longer prefixes tend to help more. Could be length or content confound.")
else:
    print(f"  -> Longer prefixes are WORSE. Content quality > quantity.")

# Within surr_kw5: does per-sample prefix token count predict benefit?
kw5_pfx = np.array([s['n_pfx_kw5'] for s in samples])
kw5_benefit = bare - surr_kw5
r_within, p_within = stats.pearsonr(kw5_pfx, kw5_benefit)
print(f"\nWithin surr_kw5: correlation of prefix tokens with per-sample benefit:")
print(f"  r={r_within:.3f}, p={p_within:.3e}")
""")


# ===== Cell 10: Comparison with Exp 01 + save =====
code(r"""# Cell 10: Comparison with Exp 01 + verdict + save
print("=" * 70)
print("VERDICT — Exp 04: Prefix Content Optimization")
print("=" * 70)

# Exp 01 reference
exp01_ref = {
    'oracle_trunc': 0.228, 'surr_doc_trunc (=kw5)': 0.148,
    'random_trunc': 0.080,
}

print(f"\n--- Replication check vs Exp 01 ---")
print(f"  {'Condition':<25} {'Exp 01 d':>10} {'Exp 04 d':>10}")
print(f"  {'-'*48}")
for name, ref_d in exp01_ref.items():
    if 'kw5' in name:
        this_d = cohens_d(bare - surr_kw5)
    elif 'oracle' in name:
        this_d = cohens_d(bare - oracle_trunc)
    else:
        this_d = cohens_d(bare - random_trunc)
    print(f"  {name:<25} {ref_d:>+10.3f} {this_d:>+10.3f}")

# Summary
print(f"\n--- Summary ---")
d_oracle_val = cohens_d(bare - oracle_trunc)
best_surr_name, best_surr_d = None, -999
for name, nlls in [('surr_kw5', surr_kw5), ('surr_kw10', surr_kw10),
                     ('surr_kw20', surr_kw20), ('surr_first_sent', surr_first_sent),
                     ('surr_random_kw5', surr_random_kw5)]:
    d = cohens_d(bare - nlls)
    if d > best_surr_d:
        best_surr_d = d
        best_surr_name = name

d_random_val = cohens_d(bare - random_trunc)

print(f"  Oracle (ceiling):         d={d_oracle_val:+.3f}")
print(f"  Best surrogate:           {best_surr_name} d={best_surr_d:+.3f} ({best_surr_d/d_oracle_val*100:.0f}% oracle)")
print(f"  Random (structural floor): d={d_random_val:+.3f} ({d_random_val/d_oracle_val*100:.0f}% oracle)")
print(f"  Content headroom closed:  {(best_surr_d - d_random_val)/(d_oracle_val - d_random_val)*100:.0f}% "
      f"(from random {d_random_val:+.3f} toward oracle {d_oracle_val:+.3f})"
      if d_oracle_val > d_random_val else "")

# Save
final_results = {
    'experiment': 'v4_exp04_prefix_optimization',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'bonferroni': N_BONF,
    'conditions': analysis,
    'ranking': [{'name': n, 'd': d} for n, d in ranked],
    'best_surrogate': {'name': best_surr_name, 'd': float(best_surr_d)},
    'prefix_length_analysis': prefix_data,
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
out_path = "experiments/encoder_decoder/04/04_prefix_optimization.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
