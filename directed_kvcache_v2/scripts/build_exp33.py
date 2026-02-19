#!/usr/bin/env python3
"""Generate the Exp 33 notebook: Encoder-Decoder Surrogate Query Transfer."""
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
# Cell 1: Markdown overview
# ============================================================
md("""# Experiment 33: Encoder-Decoder Surrogate Query Transfer
## Does bidirectional query-document encoding help? Does it transfer to surrogates?

### Motivation
Experiments 1-32 used decoder-only models (Gemma 3 4B, Mistral 7B) with causal attention.
The core idea -- encoding [query + document] should produce better representations than
[document] alone -- was limited by causal attention: query tokens influence document tokens
but NOT vice versa. The query is already presented at inference time, so the one-directional
influence during cache building adds little.

**T5Gemma 2 4B-4B** is an encoder-decoder model with **bidirectional attention** in the encoder.
Query and document tokens mutually influence each other during encoding. This is the natural
architecture for the core hypothesis.

### Core Question
In an ad-serving scenario:
1. **Offline**: Pre-compute encoder representations for each ad/document
2. **Online**: User query arrives, decoder generates answer using pre-computed encoder output

Does encoding [surrogate_query + document] offline help the decoder answer a *different*
real user query at serving time?

### Conditions (encoder input varies, decoder target is always the same)
1. **bare**: encoder("[document]")
2. **oracle**: encoder("[query]\\n[document]") -- same query at encode + decode time
3. **static**: encoder("What are the key facts?\\n[document]") -- content-agnostic prefix
4. **surr_para**: encoder("[paraphrased_query]\\n[document]") -- reordered keywords of real query
5. **surr_doc**: encoder("[doc_keywords]\\n[document]") -- keywords extracted from document

Decoder (same for all): score NLL of answer tokens given "[query] Answer: [answer]"

### Success Criteria
- **Oracle d > +0.1**: Bidirectional encoding with query helps (unlike decoder-only d~0)
- **Surrogate d > 0.05**: Surrogate query transfers partial benefit
- **Transfer ratio > 30%**: surr_para captures >30% of oracle benefit
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

# Load HF token from .env
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Experiment 33: Encoder-Decoder Surrogate Query Transfer")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

# ============================================================
# Cell 3: Load model
# ============================================================
code("""# Cell 3: Load T5Gemma 2 4B-4B
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
# Cell 4: Define scoring and surrogate helpers
# ============================================================
code(r"""# Cell 4: Scoring and surrogate helpers

def score_answer_nll(encoder_text, query, answer):
    '''Score NLL of answer tokens only.

    Encoder: encoder_text (varies by condition)
    Decoder: "[query] Answer: [answer]" -- NLL computed on answer tokens only
    '''
    # Encode
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    enc_mask = torch.ones_like(enc_ids)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # Decoder target
    full_target = f"{query} Answer: {answer}"
    query_prefix = f"{query} Answer: "

    full_ids = tokenizer(full_target, return_tensors="pt",
                         add_special_tokens=False, truncation=True,
                         max_length=512).input_ids.to(DEVICE)
    prefix_ids = tokenizer(query_prefix, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(DEVICE)

    prefix_len = prefix_ids.shape[1]
    total_len = full_ids.shape[1]
    answer_len = total_len - prefix_len

    if answer_len <= 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            labels=full_ids,
        )

    # Per-token log-probs, only over answer portion
    logits = outputs.logits  # (1, total_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, full_ids[0].unsqueeze(1)).squeeze(1)

    answer_log_probs = token_log_probs[prefix_len:]
    mean_nll = -answer_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


# === Surrogate query generation ===
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


def make_surrogate_paraphrase(query):
    '''Paraphrase: reverse keyword order.
    "what is the capital of france" -> "france capital"
    '''
    keywords = extract_keywords(query)
    if not keywords:
        return query
    return " ".join(keywords[::-1])


def make_surrogate_from_doc(passage):
    '''Extract top-5 keywords from document by frequency.
    Most realistic: you only have the document, no query.
    '''
    content_words = extract_keywords(passage)
    if not content_words:
        return "what is this about"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))


STATIC_PREFIX = "What are the key facts?"

print("Helpers defined.")
print("  score_answer_nll(encoder_text, query, answer)")
print("  make_surrogate_paraphrase(query)")
print("  make_surrogate_from_doc(passage)")""")

# ============================================================
# Cell 5: Load data
# ============================================================
code("""# Cell 5: Load MS MARCO data
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
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

# Generate surrogates
for s in samples:
    s['surrogate_para'] = make_surrogate_paraphrase(s['query'])
    s['surrogate_doc_kw'] = make_surrogate_from_doc(s['passage'])

print(f"Selected {len(samples)} samples")
print(f"Word counts: mean={np.mean([s['word_count'] for s in samples]):.0f}")""")

# ============================================================
# Cell 6: Explain conditions
# ============================================================
code(r"""# Cell 6: Explain experimental conditions
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

CONDITIONS = {
    'bare':      lambda s: s['passage'],
    'oracle':    lambda s: s['query'] + "\n" + s['passage'],
    'static':    lambda s: STATIC_PREFIX + "\n" + s['passage'],
    'surr_para': lambda s: s['surrogate_para'] + "\n" + s['passage'],
    'surr_doc':  lambda s: s['surrogate_doc_kw'] + "\n" + s['passage'],
}

ex = samples[0]
for name, fn in CONDITIONS.items():
    enc_input = fn(ex)
    n_tokens = len(tokenizer(enc_input, add_special_tokens=True).input_ids)
    print(f"\n### {name} ###")
    print(f"  Encoder input ({n_tokens} tokens):")
    print(f"    {enc_input[:120]}...")

print(f"\n### Decoder (same for all conditions) ###")
dec_target = f"{ex['query']} Answer: {ex['answer']}"
print(f"  Target: {dec_target[:120]}...")
print(f"  NLL computed on answer tokens only (after 'Answer: ')")

print(f"\n--- Example surrogates (first 5) ---")
for i in range(5):
    s = samples[i]
    print(f"\n  Query:    {s['query'][:60]}")
    print(f"  Para:     {s['surrogate_para'][:60]}")
    print(f"  Doc KW:   {s['surrogate_doc_kw'][:60]}")""")

# ============================================================
# Cell 7: Run scoring
# ============================================================
code(r"""# Cell 7: Run scoring
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

cond_names = list(CONDITIONS.keys())

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
    query, answer = s['query'], s['answer']

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
print(f"\nScoring complete: {len(all_results)} samples in {elapsed/60:.1f} min")""")

# ============================================================
# Cell 8: Results
# ============================================================
code(r"""# Cell 8: Results
from lib.analysis import cohens_d

print("=" * 70)
print(f"RESULTS (N={len(all_results)})")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in all_results])

print(f"\n{'Condition':<15} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print("-" * 73)

analysis = {}
for cond in cond_names:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()
    diff = bare_nlls - nlls  # positive = condition better (lower NLL)
    d = cohens_d(diff)
    win_pct = 100 * np.mean(diff > 0)

    if cond == 'bare':
        print(f"{cond:<15} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"{cond:<15} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta_vs_bare': float(diff.mean()),
            'cohens_d': float(d), 'win_pct': float(win_pct), 'p_value': float(p_val),
        }

# Pairwise d matrix
print(f"\n--- Pairwise Cohen's d (row vs column, positive = row better) ---")
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
            diff = nlls2 - nlls1
            d = cohens_d(diff)
            print(f" {d:>+12.3f}", end='')
    print()""")

# ============================================================
# Cell 9: Hardness and transfer analysis
# ============================================================
code(r"""# Cell 9: Hardness and transfer analysis

# --- Hardness gradient ---
print("=" * 70)
print("HARDNESS GRADIENT (by bare NLL quintile)")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"\n{'Quintile':<12} {'N':>4}", end='')
for cond in cond_names:
    print(f" {cond:>12}", end='')
print(f" {'orc-bare':>10}")
print("-" * (20 + 13 * len(cond_names) + 12))

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    print(f"{qlabel:<12} {n_q:>4}", end='')
    for cond in cond_names:
        vals = np.array([all_results[j][f'nll_{cond}'] for j in range(len(all_results)) if mask[j]])
        print(f" {vals.mean():>12.4f}", end='')
    bare_q = np.array([all_results[j]['nll_bare'] for j in range(len(all_results)) if mask[j]])
    orc_q = np.array([all_results[j]['nll_oracle'] for j in range(len(all_results)) if mask[j]])
    print(f" {(bare_q - orc_q).mean():>+10.4f}")

# --- Surrogate transfer ---
print(f"\n{'='*70}")
print("SURROGATE TRANSFER ANALYSIS")
print("=" * 70)

oracle_delta = bare_nlls - np.array([r['nll_oracle'] for r in all_results])
surr_para_delta = bare_nlls - np.array([r['nll_surr_para'] for r in all_results])
surr_doc_delta = bare_nlls - np.array([r['nll_surr_doc'] for r in all_results])
static_delta = bare_nlls - np.array([r['nll_static'] for r in all_results])

# Correlations
r_op, p_op = stats.pearsonr(oracle_delta, surr_para_delta)
r_od, p_od = stats.pearsonr(oracle_delta, surr_doc_delta)
r_pd, p_pd = stats.pearsonr(surr_para_delta, surr_doc_delta)
r_os, p_os = stats.pearsonr(oracle_delta, static_delta)

print(f"\nCorrelations between condition benefits (positive = same samples helped):")
print(f"  oracle vs surr_para:  r={r_op:.3f} (p={p_op:.2e})")
print(f"  oracle vs surr_doc:   r={r_od:.3f} (p={p_od:.2e})")
print(f"  oracle vs static:     r={r_os:.3f} (p={p_os:.2e})")
print(f"  surr_para vs surr_doc: r={r_pd:.3f} (p={p_pd:.2e})")

# Transfer ratios
oracle_d = analysis.get('oracle', {}).get('cohens_d', 0)
if oracle_d > 0:
    for name, delta in [('static', static_delta), ('surr_para', surr_para_delta), ('surr_doc', surr_doc_delta)]:
        d = cohens_d(delta)
        ratio = d / oracle_d * 100
        print(f"\n  {name} transfer: d={d:+.3f}, ratio={ratio:.0f}% of oracle (d={oracle_d:+.3f})")""")

# ============================================================
# Cell 10: Verdict and save
# ============================================================
code(r"""# Cell 10: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 33: Encoder-Decoder Surrogate Transfer")
print("=" * 70)

print(f"\nModel: {MODEL_NAME} (encoder-decoder, bidirectional encoder)")
print(f"N: {len(all_results)} samples")

oracle_d = analysis.get('oracle', {}).get('cohens_d', 0)
static_d = analysis.get('static', {}).get('cohens_d', 0)
surr_para_d = analysis.get('surr_para', {}).get('cohens_d', 0)
surr_doc_d = analysis.get('surr_doc', {}).get('cohens_d', 0)

print(f"\n1. Does oracle query in encoder help? (target: d > +0.1)")
print(f"   d = {oracle_d:+.3f} -> {'YES' if oracle_d > 0.1 else 'MARGINAL' if oracle_d > 0.03 else 'NO'}")
print(f"   (Decoder-only comparison: d ~ 0.0)")

print(f"\n2. Does static prefix help?")
print(f"   d = {static_d:+.3f} -> {'YES' if static_d > 0.1 else 'MARGINAL' if static_d > 0.03 else 'NO'}")

print(f"\n3. Does surrogate query transfer? (target: d > +0.05)")
print(f"   Paraphrase: d = {surr_para_d:+.3f} -> {'YES' if surr_para_d > 0.05 else 'MARGINAL' if surr_para_d > 0.02 else 'NO'}")
print(f"   Doc keywords: d = {surr_doc_d:+.3f} -> {'YES' if surr_doc_d > 0.05 else 'MARGINAL' if surr_doc_d > 0.02 else 'NO'}")

if oracle_d > 0:
    print(f"\n4. Transfer efficiency:")
    print(f"   Paraphrase:    {surr_para_d/oracle_d*100:.0f}% of oracle")
    print(f"   Doc keywords:  {surr_doc_d/oracle_d*100:.0f}% of oracle")
    print(f"   Static prefix: {static_d/oracle_d*100:.0f}% of oracle")

# Comparison to decoder-only
print(f"\n5. Decoder-only comparison (Gemma 3 4B, Exps 1-32):")
print(f"   Oracle priming (causal, full-context):  d ~ +0.023 (ns)")
print(f"   Oracle priming (causal, values-only):   d ~ +0.211")
print(f"   Static prefix (causal, values-only):    d ~ +0.288")
print(f"   T5Gemma oracle (bidirectional):         d = {oracle_d:+.3f}")

verdict = "CONFIRMED" if oracle_d > 0.1 and surr_para_d > 0.05 else \
          "PARTIAL" if oracle_d > 0.05 or surr_para_d > 0.03 else "NEGATIVE"
print(f"\n{'='*70}")
print(f"OVERALL: {verdict}")
if verdict == "CONFIRMED":
    print("Bidirectional encoding unlocks the query-conditioning benefit")
    print("that causal attention could not provide. Surrogate transfer works.")
elif verdict == "PARTIAL":
    print("Some evidence for bidirectional benefit, but surrogate transfer")
    print("is weaker than hoped.")
else:
    print("Even with bidirectional attention, query-conditioning during")
    print("encoding does not meaningfully help.")
print(f"{'='*70}")

# Save
final_results = {
    'experiment': 'exp33_t5gemma_surrogate',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'correlations': {
        'oracle_vs_surr_para': float(r_op),
        'oracle_vs_surr_doc': float(r_od),
        'oracle_vs_static': float(r_os),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
# Cell 11: Cleanup
# ============================================================
code("""# Cell 11: Cleanup
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
# Build notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

outpath = "33_t5gemma_surrogate_transfer.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {outpath}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
