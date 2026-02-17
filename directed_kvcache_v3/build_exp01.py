#!/usr/bin/env python3
"""Generate Exp 01 notebook: Truncation Test -- Disentangling the Benefit.

Exp 33b (v2) showed oracle d=+0.345 with T5Gemma encoder-decoder.
But the decoder cross-attended to ALL encoder tokens, including query tokens.
Was the benefit from:
  (a) The decoder reading query tokens from encoder output (trivial)
  (b) Document representations improved by bidirectional co-encoding (the real prize)

This experiment adds TRUNCATED conditions: encode [query + document] with full
bidirectional attention, but MASK query tokens from decoder cross-attention.
The decoder can only attend to document positions. If the benefit persists,
it's from improved document representations. If it vanishes, the decoder was
just reading the query.

Critical T5Gemma2 detail: cross-attention keys have NO RoPE, so masking
encoder positions is safe without any position correction.
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
md("""# Experiment 01: Truncation Test -- Disentangling the Benefit
## Does the benefit come from improved document representations or the decoder reading query tokens?

### Background
Exp 33b showed that encoding [query + document] in T5Gemma's bidirectional encoder
dramatically helps answer prediction (oracle d=+0.345, surr_doc captures 96%).
But the decoder cross-attended to ALL encoder tokens, including query/surrogate tokens.

### The key question
Is the decoder just reading the query from the encoder output (trivial), or are the
document representations genuinely improved by bidirectional co-encoding with the query?

### Method: Masking
Encode [prefix + document] with full bidirectional attention (encoder sees everything).
Then MASK the prefix tokens from decoder cross-attention. The decoder can only cross-attend
to document positions.

This is safe because T5Gemma2's cross-attention keys have NO RoPE applied -- there are
no positional embeddings to invalidate when masking.

### Conditions (6 total)
| Condition | Encoder input | Decoder cross-attends to | Tests |
|-----------|---------------|--------------------------|-------|
| bare | [document] | all (=document) | Baseline |
| oracle_full | [query + doc] | all (query + doc) | Upper bound (=Exp 33b) |
| oracle_trunc | [query + doc] | document only | Value contamination |
| surr_para_full | [para + doc] | all (para + doc) | Surrogate upper bound |
| surr_para_trunc | [para + doc] | document only | Surrogate value contamination |
| surr_doc_full | [kw + doc] | all (kw + doc) | Doc-keyword upper bound |
| surr_doc_trunc | [kw + doc] | document only | Doc-keyword value contamination |

### Success criteria
- oracle_full >> bare (replicates 33b, expected d~+0.35)
- oracle_trunc > bare (document reps improved, the real prize)
- oracle_trunc / oracle_full > 30% (significant fraction from doc reps, not just query reading)
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

RESULTS_DIR = Path("results/exp01")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 200
MODEL_NAME = "google/t5gemma-2-4b-4b"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 01: Truncation Test -- Disentangling the Benefit")
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
code(r"""# Cell 4: Scoring helpers with truncation support

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    '''Score NLL of answer tokens with optional truncation.

    Args:
        encoder_text: Full text for encoder (e.g., "[query]\n[document]")
        answer_text: Answer text for decoder (NO query in decoder)
        prefix_token_count: Number of prefix tokens (query/surrogate) to potentially mask
        truncate: If True, mask prefix tokens from decoder cross-attention

    When truncate=True:
        - Encoder processes full [prefix + document] with bidirectional attention
        - But decoder can only cross-attend to document positions
        - Tests whether document representations are improved by co-encoding

    When truncate=False:
        - Decoder cross-attends to all encoder tokens (prefix + document)
        - This is the Exp 33b setup
    '''
    # Tokenize encoder input
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]

    # Full mask for encoder (bidirectional, sees everything)
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    # Run encoder with full bidirectional attention
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # Build cross-attention mask for decoder
    if truncate and prefix_token_count > 0:
        # Mask prefix tokens: decoder can only attend to document positions
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        # Full cross-attention (decoder sees all encoder tokens)
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    # Tokenize answer for decoder
    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)

    if ans_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,  # This controls decoder cross-attention
            labels=ans_ids,
        )

    # Per-token NLL
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    '''Count how many tokens the prefix occupies in the concatenated encoding.

    Tokenizes "[prefix]\n[document]" and "[document]" separately,
    returns the difference in token count.
    '''
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

def make_surrogate_paraphrase(query):
    keywords = extract_keywords(query)
    return " ".join(keywords[::-1]) if keywords else query

def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

print("Helpers defined.")
print("  score_nll(encoder_text, answer, prefix_token_count, truncate)")
print("  Key: truncate=True masks prefix from decoder cross-attention")""")

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

print(f"Selected {len(samples)} samples, mean words={np.mean([s['word_count'] for s in samples]):.0f}")""")

# ============================================================
code(r"""# Cell 6: Explain conditions
print("=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

ex = samples[0]
print(f"\nExample query:  {ex['query'][:70]}")
print(f"Example answer: {ex['answer'][:70]}")

# Count prefix tokens for the example
oracle_prefix_tokens = count_prefix_tokens(ex['query'], ex['passage'])
para_prefix_tokens = count_prefix_tokens(ex['surrogate_para'], ex['passage'])
doc_prefix_tokens = count_prefix_tokens(ex['surrogate_doc_kw'], ex['passage'])

conditions_explained = f'''
CONDITION         ENCODER INPUT              DECODER CROSS-ATTENDS TO    PREFIX TOKENS
-------------------------------------------------------------------------------------
bare              [document]                 all (= document)            0
oracle_full       [query + doc]              all (query + doc)           ~{oracle_prefix_tokens}
oracle_trunc      [query + doc]              document ONLY               ~{oracle_prefix_tokens} (masked)
surr_para_full    [paraphrase + doc]         all (paraphrase + doc)      ~{para_prefix_tokens}
surr_para_trunc   [paraphrase + doc]         document ONLY               ~{para_prefix_tokens} (masked)
surr_doc_full     [doc_keywords + doc]       all (keywords + doc)        ~{doc_prefix_tokens}
surr_doc_trunc    [doc_keywords + doc]       document ONLY               ~{doc_prefix_tokens} (masked)

KEY INSIGHT:
  _full conditions:  decoder reads prefix + gets improved doc reps
  _trunc conditions: decoder gets improved doc reps ONLY (prefix hidden)
  If _trunc ≈ _full:  benefit is from improved document representations
  If _trunc ≈ bare:   benefit was just the decoder reading the prefix
'''
print(conditions_explained)

# Verify masking by checking token counts
print("--- Token count verification ---")
for i in range(3):
    s = samples[i]
    full_text = s['query'] + "\n" + s['passage']
    doc_text = s['passage']
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(doc_text, add_special_tokens=True).input_ids
    prefix_toks = len(full_ids) - len(doc_ids)
    print(f"  Sample {i}: query='{s['query'][:40]}...' prefix_tokens={prefix_toks}, doc_tokens={len(doc_ids)}")""")

# ============================================================
code(r"""# Cell 7: Run scoring
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

# Define all conditions
def make_conditions(sample):
    '''Return dict of {name: (encoder_text, prefix_token_count, truncate)}'''
    query = sample['query']
    passage = sample['passage']
    para = sample['surrogate_para']
    doc_kw = sample['surrogate_doc_kw']

    # Count prefix tokens for each condition
    oracle_prefix = count_prefix_tokens(query, passage)
    para_prefix = count_prefix_tokens(para, passage)
    doc_prefix = count_prefix_tokens(doc_kw, passage)

    return {
        'bare':           (passage,                          0,              False),
        'oracle_full':    (query + "\n" + passage,           0,              False),
        'oracle_trunc':   (query + "\n" + passage,           oracle_prefix,  True),
        'surr_para_full': (para + "\n" + passage,            0,              False),
        'surr_para_trunc':(para + "\n" + passage,            para_prefix,    True),
        'surr_doc_full':  (doc_kw + "\n" + passage,          0,              False),
        'surr_doc_trunc': (doc_kw + "\n" + passage,          doc_prefix,     True),
    }

cond_names = ['bare', 'oracle_full', 'oracle_trunc',
              'surr_para_full', 'surr_para_trunc',
              'surr_doc_full', 'surr_doc_trunc']

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
            print(f"Resuming: {start_idx}/{N_SAMPLES}")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES, desc="Scoring"):
    s = samples[i]
    conditions = make_conditions(s)

    result = {
        'query': s['query'], 'answer': s['answer'],
        'passage_words': s['word_count'],
    }

    for cond_name in cond_names:
        enc_text, prefix_count, trunc = conditions[cond_name]
        nll = score_nll(enc_text, s['answer'], prefix_count, trunc)
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

print(f"\n{'Condition':<20} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print("-" * 78)

analysis = {}
for cond in cond_names:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()
    diff = bare_nlls - nlls
    d = cohens_d(diff)
    win_pct = 100 * np.mean(diff > 0)

    if cond == 'bare':
        print(f"{cond:<20} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"{cond:<20} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# ---- Full vs Truncated comparison ----
print(f"\n{'='*70}")
print("FULL vs TRUNCATED -- The Key Comparison")
print("=" * 70)

for prefix_type in ['oracle', 'surr_para', 'surr_doc']:
    full_nlls = np.array([r[f'nll_{prefix_type}_full'] for r in all_results])
    trunc_nlls = np.array([r[f'nll_{prefix_type}_trunc'] for r in all_results])

    full_gap = bare_nlls.mean() - full_nlls.mean()     # full vs bare
    trunc_gap = bare_nlls.mean() - trunc_nlls.mean()   # trunc vs bare

    if full_gap > 0:
        retention = trunc_gap / full_gap * 100
    else:
        retention = float('nan')

    # Direct full vs trunc comparison
    diff_ft = full_nlls - trunc_nlls  # positive = trunc better
    d_ft = cohens_d(diff_ft)
    t_ft, p_ft = stats.ttest_1samp(diff_ft, 0) if np.std(diff_ft) > 0 else (0, 1)

    print(f"\n  {prefix_type}:")
    print(f"    full  vs bare: NLL gap = {full_gap:+.4f}")
    print(f"    trunc vs bare: NLL gap = {trunc_gap:+.4f}")
    print(f"    Retention: {retention:.0f}% of full benefit survives truncation")
    print(f"    full vs trunc: d={d_ft:+.3f}, p={p_ft:.2e}")
    if retention > 50:
        print(f"    --> DOCUMENT REPRESENTATIONS carry majority of the benefit")
    elif retention > 20:
        print(f"    --> MIXED: both doc reps and direct query reading contribute")
    else:
        print(f"    --> DECODER QUERY READING is the primary mechanism")""")

# ============================================================
code(r"""# Cell 9: Hardness gradient
print("=" * 70)
print("HARDNESS GRADIENT")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

# Show full vs trunc retention by hardness
print(f"\n--- Oracle: full vs trunc by hardness ---")
print(f"{'Quintile':<12} {'N':>4} {'bare':>10} {'orc_full':>10} {'orc_trunc':>10} {'full_gap':>10} {'trunc_gap':>10} {'retain%':>10}")
print("-" * 78)

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare_nlls[mask].mean()
    of = np.array([all_results[j]['nll_oracle_full'] for j in range(len(all_results)) if mask[j]]).mean()
    ot = np.array([all_results[j]['nll_oracle_trunc'] for j in range(len(all_results)) if mask[j]]).mean()
    fg = b - of
    tg = b - ot
    ret = tg / fg * 100 if fg > 0 else float('nan')
    print(f"{qlabel:<12} {n_q:>4} {b:>10.4f} {of:>10.4f} {ot:>10.4f} {fg:>+10.4f} {tg:>+10.4f} {ret:>9.0f}%")

# Correlations
oracle_full_delta = bare_nlls - np.array([r['nll_oracle_full'] for r in all_results])
oracle_trunc_delta = bare_nlls - np.array([r['nll_oracle_trunc'] for r in all_results])
r_ft, p_ft = stats.pearsonr(oracle_full_delta, oracle_trunc_delta)
print(f"\nCorrelation(oracle_full benefit, oracle_trunc benefit): r={r_ft:.3f} (p={p_ft:.2e})")
print("  (high r = same samples helped by both, suggesting shared mechanism)")""")

# ============================================================
code(r"""# Cell 10: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 01: Truncation Test")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(all_results)} samples")

# Key numbers
oracle_full_d = analysis.get('oracle_full', {}).get('d', 0)
oracle_trunc_d = analysis.get('oracle_trunc', {}).get('d', 0)

oracle_full_gap = bare_nlls.mean() - np.array([r['nll_oracle_full'] for r in all_results]).mean()
oracle_trunc_gap = bare_nlls.mean() - np.array([r['nll_oracle_trunc'] for r in all_results]).mean()
oracle_retention = oracle_trunc_gap / oracle_full_gap * 100 if oracle_full_gap > 0 else 0

print(f"\n--- Oracle (the most important comparison) ---")
print(f"  full  d = {oracle_full_d:+.3f} (decoder sees query + improved doc reps)")
print(f"  trunc d = {oracle_trunc_d:+.3f} (decoder sees improved doc reps ONLY)")
print(f"  Retention: {oracle_retention:.0f}%")

if oracle_trunc_d > 0.2:
    print(f"\n  STRONG: Document representations carry substantial query-specific benefit.")
    print(f"  Bidirectional co-encoding with the query genuinely improves document reps.")
    print(f"  This is the REAL mechanism, not just the decoder reading the query.")
elif oracle_trunc_d > 0.05:
    print(f"\n  MODERATE: Some benefit from improved doc reps, but decoder query reading")
    print(f"  also contributes significantly.")
elif oracle_trunc_d > 0:
    print(f"\n  WEAK: Most benefit comes from the decoder reading query tokens directly.")
    print(f"  Document representations are only marginally improved by co-encoding.")
else:
    print(f"\n  NONE: Truncation eliminates all benefit. The decoder was just reading")
    print(f"  the query from encoder output. Document representations are NOT improved")
    print(f"  by bidirectional co-encoding.")

# Surrogate transfer with truncation
print(f"\n--- Surrogate transfer (truncated) ---")
for name in ['surr_para', 'surr_doc']:
    full_d = analysis.get(f'{name}_full', {}).get('d', 0)
    trunc_d = analysis.get(f'{name}_trunc', {}).get('d', 0)
    print(f"  {name}: full d={full_d:+.3f}, trunc d={trunc_d:+.3f}")

# Comparison to decoder-only
print(f"\n--- Cross-architecture comparison ---")
print(f"  Decoder-only (Gemma 3 4B):")
print(f"    Oracle full-context:    d ~ +0.023 (ns)")
print(f"    Oracle values-only:     d ~ +0.211 ***")
print(f"  Encoder-decoder (T5Gemma 2 4B-4B):")
print(f"    Oracle full:            d = {oracle_full_d:+.3f}")
print(f"    Oracle trunc (this exp): d = {oracle_trunc_d:+.3f}")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp01_truncation_test',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'oracle_retention_pct': float(oracle_retention),
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

outpath = "01_truncation_test.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
