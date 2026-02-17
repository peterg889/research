#!/usr/bin/env python3
"""Generate the Exp 33b notebook: Corrected Surrogate Transfer Test.

KEY FIX: Query goes in encoder ONLY, not decoder. The decoder scores just
the answer text. This way the encoder representation is the decoder's ONLY
source of query information, which is exactly the ad-serving scenario.

Upper bound: encoder([query + document]) → decoder scores NLL(answer)
Lower bound: encoder([document])         → decoder scores NLL(answer)
Proposal:    encoder([surrogate + doc])   → decoder scores NLL(answer)
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
md("""# Experiment 33b: Corrected Surrogate Transfer Test
## Query in encoder ONLY -- encoder is the sole source of context

### Why Exp 33 was wrong
In Exp 33, the decoder target was "[query] Answer: [answer]". The decoder
already had the query via cross-attention to its own input tokens. Adding
the query to the encoder was redundant -- the decoder didn't NEED the encoder
to know about the query.

### The ad-serving scenario
- **Offline**: Pre-compute encoder representations for each ad/document
- **Online**: User query arrives. We need the encoder representation to CARRY
  the context for answering. The decoder generates an answer based solely on
  what the encoder provides.

### Corrected setup
- **Encoder input**: varies by condition (document ± query/surrogate)
- **Decoder target**: just "[answer]" (NO query in decoder)
- The decoder's ONLY source of information is the encoder output
- Therefore, query-aware encoding SHOULD beat bare encoding

### Conditions
1. **bare**: encoder("[document]") → decoder NLL("[answer]")
2. **oracle**: encoder("[query]\\n[document]") → decoder NLL("[answer]")
3. **static**: encoder("What are the key facts?\\n[document]") → decoder NLL("[answer]")
4. **surr_para**: encoder("[paraphrased_query]\\n[document]") → decoder NLL("[answer]")
5. **surr_doc**: encoder("[doc_keywords]\\n[document]") → decoder NLL("[answer]")

### Expected hierarchy
oracle > surr_para > surr_doc ≈ static > bare

If surrogate captures >30% of the oracle-bare gap, the core idea works.
""")

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

RESULTS_DIR = Path("results/exp33b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 200
MODEL_NAME = "google/t5gemma-2-4b-4b"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Experiment 33b: Corrected Surrogate Transfer")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

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
code(r"""# Cell 4: Scoring helper -- CORRECTED: answer-only in decoder

def score_answer_nll(encoder_text, answer_text):
    '''Score NLL of answer tokens.

    Encoder: encoder_text (contains document, optionally query/surrogate)
    Decoder: answer_text ONLY (no query -- encoder is sole context source)

    This is the corrected version: the decoder's ONLY information about
    what to answer comes from the encoder representation.
    '''
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    enc_mask = torch.ones_like(enc_ids)

    # Decoder target: just the answer
    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)

    if ans_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            labels=ans_ids,
        )

    # Per-token NLL over answer
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

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
    '''Paraphrase: reverse keyword order.'''
    keywords = extract_keywords(query)
    return " ".join(keywords[::-1]) if keywords else query

def make_surrogate_from_doc(passage):
    '''Extract top-5 keywords from document.'''
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

STATIC_PREFIX = "What are the key facts?"

print("Helpers defined.")
print("CRITICAL DIFFERENCE from Exp 33:")
print("  score_answer_nll(encoder_text, answer_text)")
print("  Decoder sees ONLY the answer -- NO query in decoder")
print("  Encoder representation is the SOLE source of context")""")

# ============================================================
code("""# Cell 5: Load data
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

for s in samples:
    s['surrogate_para'] = make_surrogate_paraphrase(s['query'])
    s['surrogate_doc_kw'] = make_surrogate_from_doc(s['passage'])

print(f"Selected {len(samples)} samples")
print(f"Word counts: mean={np.mean([s['word_count'] for s in samples]):.0f}")""")

# ============================================================
code(r"""# Cell 6: Explain experimental conditions
print("=" * 70)
print("EXPERIMENTAL CONDITIONS -- CORRECTED")
print("=" * 70)

CONDITIONS = {
    'bare':      lambda s: s['passage'],
    'oracle':    lambda s: s['query'] + "\n" + s['passage'],
    'static':    lambda s: STATIC_PREFIX + "\n" + s['passage'],
    'surr_para': lambda s: s['surrogate_para'] + "\n" + s['passage'],
    'surr_doc':  lambda s: s['surrogate_doc_kw'] + "\n" + s['passage'],
}

ex = samples[0]
print(f"\nExample query:  {ex['query'][:70]}")
print(f"Example answer: {ex['answer'][:70]}")

for name, fn in CONDITIONS.items():
    enc_input = fn(ex)
    n_tokens = len(tokenizer(enc_input, add_special_tokens=True).input_ids)
    print(f"\n### {name} ###")
    print(f"  Encoder ({n_tokens} tok): {enc_input[:100]}...")

print(f"\n### Decoder (same for ALL conditions) ###")
print(f"  Target: '{ex['answer'][:80]}...'")
print(f"  NO query in decoder -- encoder is the ONLY context source!")
print(f"  This means:")
print(f"    - bare:   decoder must produce answer knowing only the document")
print(f"    - oracle: decoder knows what question to answer (via encoder)")
print(f"    - surr:   decoder has approximate question info (via encoder)")

print(f"\n--- Surrogate examples ---")
for i in range(5):
    s = samples[i]
    print(f"\n  Real query: {s['query'][:55]}")
    print(f"  Paraphrase: {s['surrogate_para'][:55]}")
    print(f"  Doc KW:     {s['surrogate_doc_kw'][:55]}")""")

# ============================================================
code(r"""# Cell 7: Run scoring
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

cond_names = list(CONDITIONS.keys())

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

    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
        'surrogate_para': s['surrogate_para'],
        'surrogate_doc_kw': s['surrogate_doc_kw'],
    }

    for cond_name, cond_fn in CONDITIONS.items():
        encoder_text = cond_fn(s)
        # CORRECTED: decoder gets answer ONLY, no query
        nll = score_answer_nll(encoder_text, s['answer'])
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

# Pairwise
print(f"\n--- Pairwise Cohen's d (row better than column = positive) ---")
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
code(r"""# Cell 9: Transfer analysis
print("=" * 70)
print("TRANSFER ANALYSIS")
print("=" * 70)

oracle_nlls = np.array([r['nll_oracle'] for r in all_results])
surr_para_nlls = np.array([r['nll_surr_para'] for r in all_results])
surr_doc_nlls = np.array([r['nll_surr_doc'] for r in all_results])
static_nlls = np.array([r['nll_static'] for r in all_results])

# Oracle-bare gap = the total benefit of query awareness
oracle_gap = bare_nlls.mean() - oracle_nlls.mean()
print(f"\nOracle-bare NLL gap: {oracle_gap:+.4f}")
print(f"  (positive = oracle better, this is the 'prize' to capture)")

# How much of the gap does each surrogate capture?
for name, nlls in [('static', static_nlls), ('surr_para', surr_para_nlls),
                    ('surr_doc', surr_doc_nlls)]:
    gap = bare_nlls.mean() - nlls.mean()
    if oracle_gap > 0:
        ratio = gap / oracle_gap * 100
    else:
        ratio = float('nan')
    print(f"\n  {name}:")
    print(f"    Gap captured: {gap:+.4f} ({ratio:.0f}% of oracle gap)")

# Per-sample correlations
oracle_delta = bare_nlls - oracle_nlls
surr_para_delta = bare_nlls - surr_para_nlls
surr_doc_delta = bare_nlls - surr_doc_nlls

r_op, p_op = stats.pearsonr(oracle_delta, surr_para_delta)
r_od, p_od = stats.pearsonr(oracle_delta, surr_doc_delta)
r_pd, p_pd = stats.pearsonr(surr_para_delta, surr_doc_delta)

print(f"\n--- Per-sample correlations ---")
print(f"  oracle vs surr_para: r={r_op:.3f} (p={p_op:.2e})")
print(f"  oracle vs surr_doc:  r={r_od:.3f} (p={p_od:.2e})")
print(f"  surr_para vs surr_doc: r={r_pd:.3f} (p={p_pd:.2e})")

# Hardness gradient
print(f"\n--- Hardness gradient (by bare NLL quintile) ---")
quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"{'Quintile':<12} {'N':>4} {'bare':>10} {'oracle':>10} {'surr_para':>10} {'surr_doc':>10} {'orc-bare':>10} {'sp-bare':>10}")
print("-" * 78)

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare_nlls[mask].mean()
    o = oracle_nlls[mask].mean()
    sp = surr_para_nlls[mask].mean()
    sd = surr_doc_nlls[mask].mean()
    print(f"{qlabel:<12} {n_q:>4} {b:>10.4f} {o:>10.4f} {sp:>10.4f} {sd:>10.4f} {b-o:>+10.4f} {b-sp:>+10.4f}")""")

# ============================================================
code(r"""# Cell 10: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 33b: Corrected Surrogate Transfer")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(all_results)} samples")
print(f"Setup: query in encoder ONLY, decoder sees answer ONLY")

oracle_d = analysis.get('oracle', {}).get('cohens_d', 0)
static_d = analysis.get('static', {}).get('cohens_d', 0)
surr_para_d = analysis.get('surr_para', {}).get('cohens_d', 0)
surr_doc_d = analysis.get('surr_doc', {}).get('cohens_d', 0)

oracle_gap = bare_nlls.mean() - oracle_nlls.mean()

print(f"\n--- Core question: does query in encoder help? ---")
print(f"  oracle d = {oracle_d:+.3f}")
if oracle_d > 0.2:
    print(f"  YES -- strong benefit. Query-aware encoding helps the decoder.")
elif oracle_d > 0.05:
    print(f"  MODERATE -- some benefit from query-aware encoding.")
elif oracle_d > 0:
    print(f"  MARGINAL -- barely helps.")
else:
    print(f"  NO -- even with query as sole context source, encoding doesn't help.")

print(f"\n--- Surrogate transfer ---")
if oracle_gap > 0:
    for name, d_val in [('surr_para', surr_para_d), ('surr_doc', surr_doc_d), ('static', static_d)]:
        gap = analysis.get(name, {}).get('delta_vs_bare', 0)
        ratio = gap / oracle_gap * 100 if oracle_gap > 0 else 0
        print(f"  {name}: d={d_val:+.3f}, captures {ratio:.0f}% of oracle gap")
else:
    print(f"  Oracle gap is zero or negative -- no benefit to transfer.")

# Expected hierarchy check
expected = ['oracle', 'surr_para', 'surr_doc', 'static', 'bare']
actual_order = sorted(cond_names, key=lambda c: np.array([r[f'nll_{c}'] for r in all_results]).mean())
print(f"\n--- Expected vs actual ranking (best to worst NLL) ---")
print(f"  Expected: {' > '.join(expected)}")
print(f"  Actual:   {' > '.join(actual_order)}")

# Comparison to Exp 33 (wrong setup)
print(f"\n--- Exp 33 vs 33b comparison ---")
print(f"  Exp 33  (query in decoder): oracle d = -0.175 (HURT)")
print(f"  Exp 33b (query in encoder): oracle d = {oracle_d:+.3f}")
if oracle_d > 0:
    print(f"  CONFIRMED: removing query from decoder reveals the encoder benefit")
else:
    print(f"  Even with corrected setup, encoder query-awareness doesn't help")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp33b_corrected_surrogate',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'setup': 'query in encoder ONLY, decoder scores answer ONLY',
    'analysis': analysis,
    'oracle_bare_gap': float(oracle_gap),
    'correlations': {
        'oracle_vs_surr_para': float(r_op),
        'oracle_vs_surr_doc': float(r_od),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

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
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

outpath = "33b_t5gemma_corrected_surrogate.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
