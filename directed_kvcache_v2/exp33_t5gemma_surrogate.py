#!/usr/bin/env python3
"""Experiment 33: Encoder-Decoder Surrogate Query Transfer.

Tests the core hypothesis with T5Gemma 2 4B-4B (encoder-decoder):
  Does encoding [surrogate_query + document] produce better encoder
  representations for answering a real user query than encoding
  [document] alone?

Key difference from decoder-only experiments (Exps 1-32):
  - Encoder uses BIDIRECTIONAL attention (query and document tokens
    mutually influence each other)
  - Encoder representations are the "KV cache" equivalent
  - Decoder cross-attends to encoder output at inference time

Conditions (encoder input varies, decoder input is always the same):
  1. bare:       encoder("[document]")
  2. oracle:     encoder("[query]\n[document]")     -- same query encode + decode
  3. static:     encoder("What are the key facts?\n[document]")
  4. surrogate:  encoder("[surrogate]\n[document]")  -- related but different query
  5. doc_kw:     encoder("[doc_keywords]\n[document]") -- keywords from doc itself

Decoder (same for all): score NLL of answer tokens given "[query] Answer: [answer]"
"""
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
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, ".")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp33")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 200
MODEL_NAME = "google/t5gemma-2-4b-4b"

print(f"Experiment 33: Encoder-Decoder Surrogate Query Transfer")
print(f"Model: {MODEL_NAME}")
print(f"N samples: {N_SAMPLES}")
print(f"Results dir: {RESULTS_DIR}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================================================
# Load model
# ================================================================
print(f"\nLoading {MODEL_NAME}...")
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {model.config.vocab_size}")

# ================================================================
# Helper: Score answer NLL given encoder input and decoder target
# ================================================================
def score_answer_nll(encoder_text, query, answer, encoder_outputs_cache=None):
    """Score NLL of answer tokens.

    Encoder input: encoder_text
    Decoder target: "[query] Answer: [answer]"
    Returns: mean NLL over answer tokens only.
    """
    # Encode
    if encoder_outputs_cache is not None:
        encoder_outputs = encoder_outputs_cache
        enc_ids = tokenizer(encoder_text, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(DEVICE)
        enc_mask = torch.ones_like(enc_ids)
    else:
        enc_ids = tokenizer(encoder_text, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(DEVICE)
        enc_mask = torch.ones_like(enc_ids)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(
                input_ids=enc_ids, attention_mask=enc_mask
            )

    # Build decoder target: "[query] Answer: [answer]"
    full_target = f"{query} Answer: {answer}"
    query_prefix = f"{query} Answer: "

    full_ids = tokenizer(full_target, return_tensors="pt",
                         add_special_tokens=False).input_ids.to(DEVICE)
    prefix_ids = tokenizer(query_prefix, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(DEVICE)

    prefix_len = prefix_ids.shape[1]
    total_len = full_ids.shape[1]
    answer_len = total_len - prefix_len

    if answer_len <= 0:
        return 0.0

    # Forward pass with labels
    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            labels=full_ids,
        )

    # Extract per-token loss for answer portion only
    logits = outputs.logits  # (1, seq_len, vocab)
    # T5 shifts labels internally: logits[t] predicts labels[t+1]
    # Actually, for T5-style models, the shift is:
    # decoder_input_ids = shift_right(labels)
    # logits[t] predicts labels[t]
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log-prob of each target token
    # logits shape: (1, total_len, vocab)
    # labels shape: (1, total_len)
    token_log_probs = log_probs[0].gather(1, full_ids[0].unsqueeze(1)).squeeze(1)

    # Sum only over answer tokens (after prefix)
    answer_log_probs = token_log_probs[prefix_len:]
    mean_nll = -answer_log_probs.mean().item()

    return mean_nll


def encode_text(text):
    """Get encoder outputs for a text."""
    enc_ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=True).input_ids.to(DEVICE)
    enc_mask = torch.ones_like(enc_ids)
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )
    return encoder_outputs, enc_mask


# ================================================================
# Surrogate query generation
# ================================================================
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


def extract_query_keywords(query):
    """Extract content keywords from query."""
    words = re.sub(r'[^\w\s]', '', query.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def make_surrogate_paraphrase(query):
    """Create a paraphrased surrogate: reorder keywords + simple template.

    'what is the capital of france' -> 'france capital information'
    """
    keywords = extract_query_keywords(query)
    if not keywords:
        return query
    # Reverse keyword order + add generic framing
    reversed_kw = keywords[::-1]
    return " ".join(reversed_kw)


def make_surrogate_from_doc(passage):
    """Create a surrogate query from document keywords.

    Most realistic scenario: you only have the document, no query.
    Extract top keywords by frequency weighting.
    """
    words = re.sub(r'[^\w\s]', '', passage.lower()).split()
    content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    if not content_words:
        return "what is this about"

    # Top keywords by frequency
    counts = Counter(content_words)
    top_kw = [w for w, _ in counts.most_common(5)]
    return " ".join(top_kw)


STATIC_PREFIX = "What are the key facts?"

# ================================================================
# Load MS MARCO data
# ================================================================
from lib.data import count_words
from datasets import load_dataset

print("\nLoading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= N_SAMPLES * 3:
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
            samples.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

print(f"Selected {len(samples)} samples")
print(f"Word counts: mean={np.mean([s['word_count'] for s in samples]):.0f}")

# ================================================================
# Generate surrogates for each sample
# ================================================================
print("\nGenerating surrogate queries...")
for s in samples:
    s['surrogate_para'] = make_surrogate_paraphrase(s['query'])
    s['surrogate_doc_kw'] = make_surrogate_from_doc(s['passage'])

# Show examples
print("\n--- Example surrogates ---")
for i in range(5):
    s = samples[i]
    print(f"\nQuery:     {s['query'][:70]}")
    print(f"Para:      {s['surrogate_para'][:70]}")
    print(f"Doc KW:    {s['surrogate_doc_kw'][:70]}")
    print(f"Answer:    {s['answer'][:70]}")

# ================================================================
# Define conditions
# ================================================================
CONDITIONS = {
    'bare':       lambda s: s['passage'],
    'oracle':     lambda s: s['query'] + "\n" + s['passage'],
    'static':     lambda s: STATIC_PREFIX + "\n" + s['passage'],
    'surr_para':  lambda s: s['surrogate_para'] + "\n" + s['passage'],
    'surr_doc':   lambda s: s['surrogate_doc_kw'] + "\n" + s['passage'],
}

print("\n" + "=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)
for name, fn in CONDITIONS.items():
    ex = fn(samples[0])
    print(f"\n{name}:")
    print(f"  Encoder input (first 100 chars): {ex[:100]}...")
    n_tokens = len(tokenizer(ex, add_special_tokens=True).input_ids)
    print(f"  Token count: {n_tokens}")
print(f"\nDecoder target (same for all): '[query] Answer: [answer]'")

# ================================================================
# Run experiment
# ================================================================
print("\n" + "=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

# Resume from checkpoint
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            all_results = ckpt['results']
            start_idx = len(all_results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES, desc="Scoring"):
    s = samples[i]
    query, answer, passage = s['query'], s['answer'], s['passage']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
        'surrogate_para': s['surrogate_para'],
        'surrogate_doc_kw': s['surrogate_doc_kw'],
    }

    for cond_name, cond_fn in CONDITIONS.items():
        encoder_text = cond_fn(s)
        nll = score_answer_nll(encoder_text, query, answer)
        result[f'nll_{cond_name}'] = nll

    all_results.append(result)

    # Checkpoint every 20
    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES, 'results': all_results,
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
print(f"\nScoring complete: {len(all_results)} samples in {elapsed/60:.1f} min")

# ================================================================
# Analysis
# ================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

cond_names = list(CONDITIONS.keys())


def cohens_d(a, b=None):
    if b is None:
        return np.mean(a) / np.std(a, ddof=1) if np.std(a) > 0 else 0.0
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0


# Per-condition stats
bare_nlls = np.array([r['nll_bare'] for r in all_results])

print(f"\n{'Condition':<15} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print("-" * 73)

analysis = {}
for cond in cond_names:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()
    delta = bare_nlls.mean() - mean_nll  # positive = condition has lower NLL = better
    diff = bare_nlls - nlls
    d = cohens_d(diff)
    win_pct = 100 * np.mean(diff > 0)

    if cond == 'bare':
        print(f"{cond:<15} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
    else:
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"{cond:<15} {mean_nll:>10.4f} {delta:>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")

    analysis[cond] = {
        'mean_nll': float(mean_nll),
        'delta_vs_bare': float(delta),
        'cohens_d': float(d),
        'win_pct': float(win_pct) if cond != 'bare' else None,
    }

# Pairwise comparisons
print(f"\n--- Pairwise comparisons (row vs column, positive = row is better) ---")
print(f"{'':>15}", end='')
for c in cond_names:
    print(f" {c:>12}", end='')
print()

for c1 in cond_names:
    nlls1 = np.array([r[f'nll_{c1}'] for r in all_results])
    print(f"{c1:<15}", end='')
    for c2 in cond_names:
        if c1 == c2:
            print(f" {'--':>12}", end='')
        else:
            nlls2 = np.array([r[f'nll_{c2}'] for r in all_results])
            diff = nlls2 - nlls1  # positive = c1 better (lower NLL)
            d = cohens_d(diff)
            print(f" {d:>+12.3f}", end='')
    print()

# Hardness interaction
print(f"\n--- Hardness gradient (by bare NLL quintile) ---")
quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"{'Quintile':<12} {'N':>4}", end='')
for cond in cond_names:
    print(f" {cond:>12}", end='')
print(f" {'oracle-bare':>12}")
print("-" * (20 + 13 * (len(cond_names) + 1)))

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    print(f"{qlabel:<12} {n_q:>4}", end='')
    for cond in cond_names:
        nlls = np.array([all_results[j][f'nll_{cond}'] for j in range(len(all_results)) if mask[j]])
        print(f" {nlls.mean():>12.4f}", end='')
    # Oracle - bare delta
    bare_q = np.array([all_results[j]['nll_bare'] for j in range(len(all_results)) if mask[j]])
    oracle_q = np.array([all_results[j]['nll_oracle'] for j in range(len(all_results)) if mask[j]])
    print(f" {(bare_q - oracle_q).mean():>+12.4f}")

# Correlation between oracle benefit and surrogate benefit
oracle_delta = bare_nlls - np.array([r['nll_oracle'] for r in all_results])
surr_para_delta = bare_nlls - np.array([r['nll_surr_para'] for r in all_results])
surr_doc_delta = bare_nlls - np.array([r['nll_surr_doc'] for r in all_results])

r_oracle_para, p_op = stats.pearsonr(oracle_delta, surr_para_delta)
r_oracle_doc, p_od = stats.pearsonr(oracle_delta, surr_doc_delta)
r_para_doc, p_pd = stats.pearsonr(surr_para_delta, surr_doc_delta)

print(f"\n--- Correlations between condition benefits ---")
print(f"  oracle-bare vs surr_para-bare: r={r_oracle_para:.3f} (p={p_op:.2e})")
print(f"  oracle-bare vs surr_doc-bare:  r={r_oracle_doc:.3f} (p={p_od:.2e})")
print(f"  surr_para-bare vs surr_doc-bare: r={r_para_doc:.3f} (p={p_pd:.2e})")

# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*70}")
print("VERDICT -- Exp 33: Encoder-Decoder Surrogate Transfer")
print("=" * 70)
print(f"\nModel: {MODEL_NAME} (encoder-decoder, bidirectional encoder)")
print(f"N: {len(all_results)} samples")

oracle_d = analysis['oracle']['cohens_d']
static_d = analysis['static']['cohens_d']
surr_para_d = analysis['surr_para']['cohens_d']
surr_doc_d = analysis['surr_doc']['cohens_d']

print(f"\n1. Does oracle query in encoder help?")
print(f"   oracle d={oracle_d:+.3f} → {'YES' if oracle_d > 0.1 else 'MARGINAL' if oracle_d > 0.03 else 'NO'}")

print(f"\n2. Does static prefix help (like decoder-only value contamination)?")
print(f"   static d={static_d:+.3f} → {'YES' if static_d > 0.1 else 'MARGINAL' if static_d > 0.03 else 'NO'}")

print(f"\n3. Does surrogate query transfer work?")
print(f"   surr_para d={surr_para_d:+.3f} → {'YES' if surr_para_d > 0.1 else 'MARGINAL' if surr_para_d > 0.03 else 'NO'}")
print(f"   surr_doc d={surr_doc_d:+.3f} → {'YES' if surr_doc_d > 0.1 else 'MARGINAL' if surr_doc_d > 0.03 else 'NO'}")

if oracle_d > 0.1:
    transfer_ratio_para = surr_para_d / oracle_d * 100 if oracle_d > 0 else 0
    transfer_ratio_doc = surr_doc_d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"\n4. Transfer efficiency (surrogate / oracle):")
    print(f"   Paraphrase: {transfer_ratio_para:.0f}%")
    print(f"   Doc keywords: {transfer_ratio_doc:.0f}%")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp33_t5gemma_surrogate',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'correlations': {
        'oracle_vs_surr_para': float(r_oracle_para),
        'oracle_vs_surr_doc': float(r_oracle_doc),
        'surr_para_vs_surr_doc': float(r_para_doc),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print("\nCleaning up...")
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("Done!")
