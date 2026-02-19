#!/usr/bin/env python3
"""Generate Exp 03 notebook: Length Scaling -- Does the Benefit Survive Longer Documents?

Exp 01 proved that bidirectional co-encoding with a surrogate genuinely improves
document representations (oracle_trunc d=+0.408, surr_doc_trunc d=+0.363 capturing
89% of oracle benefit). But those results were on short MS MARCO passages (~130 tokens).

The #1 limitation from v2 was that value contamination in decoder-only models
diluted at ~200 tokens (v2 Exp 20) — a step function, not gradual decay. T5Gemma's
bidirectional encoder should distribute surrogate influence more uniformly because
every document token attends to the surrogate in BOTH directions.

This experiment pads the same MS MARCO passages to controlled lengths with unrelated
text, isolating length as the only variable. Same questions, same answers — only the
document length changes.
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
md("""# Experiment 03: Length Scaling -- Does the Benefit Survive Longer Documents?
## The #1 limitation from v2: value contamination diluted at ~200 tokens. Does T5Gemma's bidirectional encoder change this?

### Background
Exp 01 showed that bidirectional co-encoding genuinely improves document representations:
- oracle_trunc d=+0.408 (94% win, p=3e-08)
- surr_doc_trunc d=+0.363 (89% of oracle benefit)
- surr_para_trunc d=+0.357 (87% of oracle benefit)

But those passages were short (~130 tokens). In v2 Exp 20 (decoder-only Gemma 3 4B),
the benefit collapsed as documents got longer:

| Length | v2 Exp 20 oracle d |
|--------|-------------------|
| ~130 tok | +0.303*** |
| 256 tok | +0.114 (ns) |
| 512 tok | +0.034 (ns) |
| 1024 tok | -0.043 (ns) |

### Why T5Gemma should be different
In decoder-only models, the surrogate's influence propagates only forward through causal
attention — by ~200 tokens downstream, it's diluted to noise. T5Gemma's bidirectional
encoder lets every document token attend to the surrogate in BOTH directions via global
self-attention. The surrogate influence should be distributed more uniformly.

### Method
Take the same short MS MARCO passages and pad them to controlled lengths with unrelated
MS MARCO text (separated by `\\n\\n`). Same questions, same answers, same passages —
only the document length changes. This is a within-subject design for maximum statistical power.

### Conditions (all with truncation, per Exp 01)
| Condition | Encoder input | Decoder cross-attends to |
|-----------|---------------|--------------------------|
| bare | [document] | all (= document) |
| oracle_trunc | [query + doc] | document only |
| surr_doc_trunc | [doc_kw + doc] | document only |
| surr_para_trunc | [para + doc] | document only |

### Length bins
| Bin | Target tokens | v2 Exp 20 result |
|-----|--------------|------------------|
| original | ~130 tok (no padding) | d=+0.303*** |
| 256 | padded | d=+0.114 (ns) |
| 384 | padded (new, near v2 cliff) | not tested in v2 |
| 512 | padded | d=+0.034 (ns) |
| 1024 | padded | d=-0.043 (ns) |
| 2048 | padded (new) | not tested in v2 |

### Design: N=400 for power at longer lengths
N=200 with Bonferroni can only detect d>=0.25. The decay tail (d~0.15 at 512+) is where
the interesting science is. N=400 gives power to detect d>=0.15 with Bonferroni correction.

### Success criteria
- If d stays significant at 512+ tokens: bidirectional encoder is qualitatively better than causal
- If decay is gradual (not a cliff): surrogate influence distributed more uniformly
- Critical threshold: 512 tokens (where ESCI product descriptions live)
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

sys.path.insert(0, "../..")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../../results/exp03")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 400
MODEL_NAME = "google/t5gemma-2-4b-4b"
LENGTH_BINS = ["original", "256", "384", "512", "1024", "2048"]

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 03: Length Scaling -- Does the Benefit Survive Longer Documents?")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"Length bins: {LENGTH_BINS}")
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
code(r"""# Cell 4: Scoring helpers (reused from Exp 01)

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    '''Score NLL of answer tokens with optional truncation.

    Args:
        encoder_text: Full text for encoder (e.g., "[query]\n[document]")
        answer_text: Answer text for decoder (NO query in decoder)
        prefix_token_count: Number of prefix tokens (query/surrogate) to potentially mask
        truncate: If True, mask prefix tokens from decoder cross-attention
    '''
    # Tokenize encoder input
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=4096).input_ids.to(DEVICE)
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
    '''Count how many tokens the prefix occupies in the concatenated encoding.'''
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)


# === Surrogate generation (from Exp 01) ===
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
print("  NOTE: max_length=4096 (up from 2048 in Exp 01) for longer padded documents")""")

# ============================================================
code("""# Cell 5: Load data and build padding pool
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Collect target samples (short passages with answers)
samples = []
# Collect unrelated passages for padding pool
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
            if len(samples) < N_SAMPLES * 3:
                samples.append({
                    'passage': pt, 'query': query, 'answer': answer,
                    'word_count': wc
                })
        elif sel == 0 and 20 <= wc <= 200:
            # Non-selected passages make good padding material
            padding_pool.append(pt)

    if len(samples) >= N_SAMPLES * 3 and len(padding_pool) >= 5000:
        break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]

# Shuffle padding pool so concatenation order is random
np.random.shuffle(padding_pool)

del ds
gc.collect()

# Generate surrogates
for s in samples:
    s['surrogate_para'] = make_surrogate_paraphrase(s['query'])
    s['surrogate_doc_kw'] = make_surrogate_from_doc(s['passage'])

print(f"Selected {len(samples)} target samples, mean words={np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Padding pool: {len(padding_pool)} unrelated passages")

# Show token counts for target passages
target_tok_counts = []
for s in samples:
    toks = tokenizer(s['passage'], add_special_tokens=True).input_ids
    target_tok_counts.append(len(toks))
print(f"Target passage tokens: mean={np.mean(target_tok_counts):.0f}, "
      f"median={np.median(target_tok_counts):.0f}, "
      f"min={np.min(target_tok_counts)}, max={np.max(target_tok_counts)}")""")

# ============================================================
code(r"""# Cell 6: Build padded documents at each length bin
print("=" * 70)
print("BUILDING PADDED DOCUMENTS")
print("=" * 70)

TARGET_LENGTHS = {
    "original": None,  # No padding
    "256": 256,
    "384": 384,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
}

def pad_passage_to_length(passage, target_tokens, padding_pool, pool_offset):
    '''Pad a passage to target_tokens by appending unrelated passages.

    Args:
        passage: Original passage text
        target_tokens: Target token count (None = no padding)
        padding_pool: List of unrelated passages
        pool_offset: Starting index into padding pool (for reproducibility)

    Returns:
        (padded_text, actual_token_count, n_padding_passages_used)
    '''
    if target_tokens is None:
        toks = tokenizer(passage, add_special_tokens=True).input_ids
        return passage, len(toks), 0

    # Check if passage already exceeds target
    current_ids = tokenizer(passage, add_special_tokens=True).input_ids
    if len(current_ids) >= target_tokens:
        return passage, len(current_ids), 0

    # Append padding passages until we reach target length
    padded = passage
    n_used = 0
    idx = pool_offset

    while True:
        if idx >= len(padding_pool):
            idx = 0  # Wrap around if needed
        candidate = padded + "\n\n" + padding_pool[idx]
        candidate_ids = tokenizer(candidate, add_special_tokens=True).input_ids
        if len(candidate_ids) >= target_tokens:
            # This passage would push us over — truncate it to fit
            # Add words from this padding passage one at a time
            pad_words = padding_pool[idx].split()
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
padded_docs = {}  # {length_bin: [padded_passage_text, ...]}
padded_stats = {}  # {length_bin: {mean_tokens, min_tokens, max_tokens}}

for length_bin, target_tokens in TARGET_LENGTHS.items():
    padded_docs[length_bin] = []
    tok_counts = []

    for i, s in enumerate(samples):
        # Use different pool offset per sample for diversity
        pool_offset = i * 50  # Spread across pool
        padded_text, actual_tokens, n_pad = pad_passage_to_length(
            s['passage'], target_tokens, padding_pool, pool_offset
        )
        padded_docs[length_bin].append(padded_text)
        tok_counts.append(actual_tokens)

    padded_stats[length_bin] = {
        'mean': np.mean(tok_counts),
        'min': int(np.min(tok_counts)),
        'max': int(np.max(tok_counts)),
        'median': np.median(tok_counts),
    }

    print(f"\n  {length_bin:>8s}: mean={padded_stats[length_bin]['mean']:.0f} tokens "
          f"(min={padded_stats[length_bin]['min']}, max={padded_stats[length_bin]['max']}, "
          f"median={padded_stats[length_bin]['median']:.0f})")

# Verify: show first sample at each length
print(f"\n--- Sample 0 preview ---")
print(f"  Query:  {samples[0]['query'][:80]}")
print(f"  Answer: {samples[0]['answer'][:80]}")
for lb in LENGTH_BINS:
    preview = padded_docs[lb][0]
    tok_count = len(tokenizer(preview, add_special_tokens=True).input_ids)
    print(f"  {lb:>8s}: {tok_count} tokens, starts='{preview[:60]}...', ends='...{preview[-40:]}'")

# Show actual surrogate prefix text for each condition
print(f"\n--- Actual prefix text for each condition (sample 0) ---")
surrogates = {
    'oracle (real query)': samples[0]['query'],
    'surr_para (reversed kw)': samples[0]['surrogate_para'],
    'surr_doc (top-5 doc kw)': samples[0]['surrogate_doc_kw'],
}
for name, text in surrogates.items():
    ptoks = count_prefix_tokens(text, padded_docs["original"][0])
    print(f"  {name:<25} ({ptoks:>3} prefix toks): {text[:70]}")""")

# ============================================================
code(r"""# Cell 7: Explain conditions
print("=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

ex = samples[0]
ex_doc = padded_docs["original"][0]

# Count prefix tokens for the example
oracle_prefix = count_prefix_tokens(ex['query'], ex_doc)
para_prefix = count_prefix_tokens(ex['surrogate_para'], ex_doc)
doc_prefix = count_prefix_tokens(ex['surrogate_doc_kw'], ex_doc)

conditions_explained = f'''
CONDITIONS (all with truncation, per Exp 01 findings):

  CONDITION          ENCODER INPUT              DECODER CROSS-ATTENDS TO    PREFIX TOKENS
  ---------------------------------------------------------------------------------
  bare               [document]                 all (= document)            0
  oracle_trunc       [query + doc]              document ONLY               ~{oracle_prefix} (masked)
  surr_doc_trunc     [doc_kw + doc]             document ONLY               ~{doc_prefix} (masked)
  surr_para_trunc    [para + doc]               document ONLY               ~{para_prefix} (masked)

LENGTH BINS: {LENGTH_BINS}

DESIGN:
  - Same {N_SAMPLES} samples at every length (within-subject, matched design)
  - Outer loop = length bin (all samples at one length before moving to next)
  - Padding: unrelated MS MARCO passages appended after target passage
  - Question and answer unchanged — only document length varies
  - Total scoring calls: {N_SAMPLES} samples x {len(LENGTH_BINS)} lengths x 4 conditions = {N_SAMPLES * len(LENGTH_BINS) * 4}
'''
print(conditions_explained)

# Estimate runtime
print(f"Estimated runtime: ~{N_SAMPLES * len(LENGTH_BINS) * 4 * 0.5 / 60:.0f} min "
      f"(assuming ~0.5s per scoring call)")""")

# ============================================================
code(r"""# Cell 8: Run scoring — outer loop over length bins
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'surr_doc_trunc', 'surr_para_trunc']

def make_conditions(sample, padded_passage):
    '''Return dict of {name: (encoder_text, prefix_token_count, truncate)}'''
    query = sample['query']
    para = sample['surrogate_para']
    doc_kw = sample['surrogate_doc_kw']

    # Count prefix tokens for each condition (using padded passage)
    oracle_prefix = count_prefix_tokens(query, padded_passage)
    para_prefix = count_prefix_tokens(para, padded_passage)
    doc_prefix = count_prefix_tokens(doc_kw, padded_passage)

    return {
        'bare':            (padded_passage,                        0,              False),
        'oracle_trunc':    (query + "\n" + padded_passage,         oracle_prefix,  True),
        'surr_doc_trunc':  (doc_kw + "\n" + padded_passage,        doc_prefix,     True),
        'surr_para_trunc': (para + "\n" + padded_passage,          para_prefix,    True),
    }

# Resume from checkpoint
# Checkpoint format: {length_bin: {"results": [...], "completed": N}, ...}
all_checkpoint = {}
if CHECKPOINT_PATH.exists():
    saved = json.loads(CHECKPOINT_PATH.read_text())
    if saved.get('n_total') == N_SAMPLES:
        all_checkpoint = saved.get('bins', {})
        summary = ', '.join(f'{k}={len(v.get("results",[]))}' for k,v in all_checkpoint.items())
        print(f"Loaded checkpoint: {summary}")

t0_total = time.time()

for length_bin in LENGTH_BINS:
    print(f"\n{'='*70}")
    print(f"LENGTH BIN: {length_bin} (target={TARGET_LENGTHS[length_bin] or 'no padding'})")
    print(f"{'='*70}")

    # Check for existing results for this bin
    bin_results = []
    start_idx = 0
    if length_bin in all_checkpoint:
        bin_data = all_checkpoint[length_bin]
        saved_results = bin_data.get('results', [])
        # Verify alignment
        saved_queries = [r['query'][:50] for r in saved_results]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            bin_results = saved_results
            start_idx = len(bin_results)
            print(f"  Resuming from sample {start_idx}/{N_SAMPLES}")

    if start_idx >= N_SAMPLES:
        print(f"  Already complete ({len(bin_results)} results)")
        all_checkpoint[length_bin] = {"results": bin_results, "completed": N_SAMPLES}
        continue

    t0_bin = time.time()

    for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
                  desc=f"  {length_bin}"):
        s = samples[i]
        padded_passage = padded_docs[length_bin][i]
        conditions = make_conditions(s, padded_passage)

        result = {
            'query': s['query'],
            'answer': s['answer'],
            'passage_words': s['word_count'],
            'padded_tokens': len(tokenizer(padded_passage, add_special_tokens=True).input_ids),
        }

        for cond_name in COND_NAMES:
            enc_text, prefix_count, trunc = conditions[cond_name]
            nll = score_nll(enc_text, s['answer'], prefix_count, trunc)
            result[f'nll_{cond_name}'] = nll

        bin_results.append(result)

        if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
            all_checkpoint[length_bin] = {"results": bin_results, "completed": len(bin_results)}
            ckpt = {
                'n_total': N_SAMPLES,
                'bins': all_checkpoint,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            CHECKPOINT_PATH.write_text(json.dumps(ckpt))
            elapsed_bin = time.time() - t0_bin
            done = i - start_idx + 1
            eta = (N_SAMPLES - i - 1) * elapsed_bin / done if done > 0 else 0
            tqdm.write(f"    Checkpoint {i+1}/{N_SAMPLES} | {elapsed_bin/60:.1f}m | ETA {eta/60:.1f}m")

        gc.collect()
        torch.cuda.empty_cache()

    elapsed_bin = time.time() - t0_bin
    print(f"  {length_bin} complete: {len(bin_results)} samples in {elapsed_bin/60:.1f} min")

    # Quick peek at this bin's results
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    oracle_nlls = np.array([r['nll_oracle_trunc'] for r in bin_results])
    diff = bare_nlls - oracle_nlls
    from lib.analysis import cohens_d
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    print(f"  Quick peek: oracle_trunc d={d:+.3f}, win={win:.0f}%")

elapsed_total = time.time() - t0_total
print(f"\n{'='*70}")
print(f"ALL BINS COMPLETE: {elapsed_total/60:.1f} min total")
print(f"{'='*70}")""")

# ============================================================
code(r"""# Cell 9: Results — per-length table
from lib.analysis import cohens_d

print("=" * 70)
print("RESULTS: Per-Length Condition Comparison")
print("=" * 70)

# Collect results from checkpoint
results_by_bin = {}
for length_bin in LENGTH_BINS:
    results_by_bin[length_bin] = all_checkpoint[length_bin]['results']

# Full results table
analysis = {}
for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    n = len(bin_results)
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    mean_tokens = np.mean([r['padded_tokens'] for r in bin_results])

    print(f"\n--- {length_bin} (mean {mean_tokens:.0f} tokens, N={n}) ---")
    print(f"  {'Condition':<20} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
    print(f"  {'-'*78}")

    analysis[length_bin] = {}
    for cond in COND_NAMES:
        nlls = np.array([r[f'nll_{cond}'] for r in bin_results])
        mean_nll = nlls.mean()
        diff = bare_nlls - nlls

        if cond == 'bare':
            print(f"  {cond:<20} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
            analysis[length_bin][cond] = {'mean_nll': float(mean_nll)}
        else:
            d = cohens_d(diff)
            win_pct = 100 * np.mean(diff > 0)
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            # Bonferroni: 3 conditions x 6 lengths = 18 comparisons
            sig = '***' if p_val < 0.001/18 else '**' if p_val < 0.01/18 else '*' if p_val < 0.05/18 else 'ns'
            print(f"  {cond:<20} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
            analysis[length_bin][cond] = {
                'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
                'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            }""")

# ============================================================
code(r"""# Cell 10: Decay curve analysis
print("=" * 70)
print("DECAY CURVE ANALYSIS")
print("=" * 70)

# Collect d values across lengths for each condition
print(f"\n--- Cohen's d vs Length ---")
print(f"  {'Length':<10} {'oracle_trunc':>14} {'surr_doc':>14} {'surr_para':>14} {'mean_tokens':>12}")
print(f"  {'-'*68}")

# v2 Exp 20 results for comparison
v2_oracle_d = {
    "original": 0.303,
    "256": 0.114,
    "384": None,   # Not tested in v2
    "512": 0.034,
    "1024": -0.043,
    "2048": None,  # Not tested in v2
}

decay_data = {'length_bin': [], 'mean_tokens': []}
for cond in ['oracle_trunc', 'surr_doc_trunc', 'surr_para_trunc']:
    decay_data[f'd_{cond}'] = []
    decay_data[f'p_{cond}'] = []

for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    mean_tokens = np.mean([r['padded_tokens'] for r in bin_results])
    decay_data['length_bin'].append(length_bin)
    decay_data['mean_tokens'].append(mean_tokens)

    d_vals = []
    for cond in ['oracle_trunc', 'surr_doc_trunc', 'surr_para_trunc']:
        a = analysis[length_bin].get(cond, {})
        d = a.get('d', 0)
        p = a.get('p', 1)
        decay_data[f'd_{cond}'].append(d)
        decay_data[f'p_{cond}'].append(p)
        d_vals.append(f"{d:+.3f}")

    v2_d = v2_oracle_d.get(length_bin)
    v2_str = f"{v2_d:+.3f}" if v2_d is not None else "  N/A"
    print(f"  {length_bin:<10} {d_vals[0]:>14} {d_vals[1]:>14} {d_vals[2]:>14} {mean_tokens:>11.0f}")

# Compare to v2 decay
print(f"\n--- v3 (T5Gemma) vs v2 (Gemma 3 4B) Oracle Decay ---")
print(f"  {'Length':<10} {'v3 oracle_trunc':>16} {'v2 oracle (Exp20)':>18} {'v3/v2 ratio':>12}")
print(f"  {'-'*60}")

for length_bin in LENGTH_BINS:
    v3_d = analysis[length_bin].get('oracle_trunc', {}).get('d', 0)
    v2_d = v2_oracle_d.get(length_bin)
    if v2_d is not None and v2_d != 0:
        ratio = v3_d / v2_d
        print(f"  {length_bin:<10} {v3_d:>+16.3f} {v2_d:>+18.3f} {ratio:>11.1f}x")
    elif v2_d is not None:
        print(f"  {length_bin:<10} {v3_d:>+16.3f} {v2_d:>+18.3f} {'--':>12}")
    else:
        print(f"  {length_bin:<10} {v3_d:>+16.3f} {'N/A':>18} {'--':>12}")

# Decay rate analysis
print(f"\n--- Decay Rate ---")
orig_d_oracle = analysis["original"].get('oracle_trunc', {}).get('d', 0)
for cond in ['oracle_trunc', 'surr_doc_trunc', 'surr_para_trunc']:
    orig_d = analysis["original"].get(cond, {}).get('d', 0)
    if orig_d == 0:
        continue
    print(f"\n  {cond}:")
    for length_bin in LENGTH_BINS:
        d = analysis[length_bin].get(cond, {}).get('d', 0)
        retention = d / orig_d * 100 if orig_d > 0 else 0
        sig = analysis[length_bin].get(cond, {}).get('p', 1)
        sig_str = '***' if sig < 0.001/18 else '**' if sig < 0.01/18 else '*' if sig < 0.05/18 else 'ns'
        print(f"    {length_bin:>8s}: d={d:+.3f} ({retention:5.1f}% of original) {sig_str}")""")

# ============================================================
code(r"""# Cell 11: Decay curve plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Map length bins to numeric x values for plotting
x_tokens = decay_data['mean_tokens']

# --- Left panel: v3 decay curves for all conditions ---
ax = axes[0]
for cond, color, marker in [('oracle_trunc', 'tab:red', 'o'),
                              ('surr_doc_trunc', 'tab:blue', 's'),
                              ('surr_para_trunc', 'tab:green', '^')]:
    d_vals = decay_data[f'd_{cond}']
    p_vals = decay_data[f'p_{cond}']
    ax.plot(x_tokens, d_vals, f'-{marker}', color=color, label=cond, markersize=8)
    # Mark significant points
    for x, d, p in zip(x_tokens, d_vals, p_vals):
        if p < 0.05 / 15:  # Bonferroni
            ax.annotate('*', (x, d), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=14, color=color)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('v3 T5Gemma: Effect Size vs Document Length')
ax.legend()
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Right panel: v3 oracle vs v2 oracle ---
ax = axes[1]
v3_oracle_d = decay_data['d_oracle_trunc']
ax.plot(x_tokens, v3_oracle_d, '-o', color='tab:red', label='v3 T5Gemma (oracle_trunc)', markersize=8)

# v2 data points (only where available)
v2_lengths = []
v2_d_vals = []
for lb, tok in zip(LENGTH_BINS, x_tokens):
    v2_d = v2_oracle_d.get(lb)
    if v2_d is not None:
        v2_lengths.append(tok)
        v2_d_vals.append(v2_d)

ax.plot(v2_lengths, v2_d_vals, '-s', color='tab:purple', label='v2 Gemma 3 4B (oracle)', markersize=8)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('Cross-Architecture Comparison: Decay Curves')
ax.legend()
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / 'decay_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")""")

# ============================================================
code(r"""# Cell 12: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 03: Length Scaling")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {N_SAMPLES} samples per length bin")
print(f"Length bins: {LENGTH_BINS}")

# Key question: at what length does the benefit become non-significant?
print(f"\n--- Oracle Decay Summary ---")
orig_d = analysis["original"].get('oracle_trunc', {}).get('d', 0)
last_sig_bin = "original"
for length_bin in LENGTH_BINS:
    d = analysis[length_bin].get('oracle_trunc', {}).get('d', 0)
    p = analysis[length_bin].get('oracle_trunc', {}).get('p', 1)
    sig = p < 0.05 / 15  # Bonferroni
    if sig:
        last_sig_bin = length_bin
    retention = d / orig_d * 100 if orig_d > 0 else 0
    sig_str = "SIGNIFICANT" if sig else "ns"
    print(f"  {length_bin:>8s}: d={d:+.3f} ({retention:5.1f}% retained) [{sig_str}]")

print(f"\n  Last significant length bin: {last_sig_bin}")

# Compare architectures
print(f"\n--- Cross-Architecture Verdict ---")
# v2 cliff was at ~200 tokens (256 bin was already ns)
v2_cliff = "~200 tokens (256 bin ns)"

# Determine v3 pattern
v3_pattern = []
for lb in LENGTH_BINS:
    d = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    p = analysis[lb].get('oracle_trunc', {}).get('p', 1)
    v3_pattern.append((lb, d, p < 0.05/18))

sig_bins = [lb for lb, d, sig in v3_pattern if sig]
if len(sig_bins) >= 4:
    print(f"  T5Gemma shows ROBUST length scaling — benefit persists to {sig_bins[-1]} tokens")
    print(f"  This is qualitatively different from v2's cliff at {v2_cliff}")
    print(f"  Bidirectional attention distributes surrogate influence across document length")
elif len(sig_bins) >= 3:
    print(f"  T5Gemma shows MODERATE length scaling — benefit persists to {sig_bins[-1]} tokens")
    print(f"  Better than v2's cliff at {v2_cliff}, but still decays")
elif len(sig_bins) >= 2:
    print(f"  T5Gemma shows LIMITED improvement — benefit extends to {sig_bins[-1]} tokens")
    print(f"  Somewhat better than v2's cliff at {v2_cliff}")
else:
    print(f"  T5Gemma shows NO improvement over v2 — benefit only at original length")
    print(f"  Bidirectional attention does NOT help with length scaling")

# Surrogate comparison across lengths
print(f"\n--- Surrogate Performance Across Lengths ---")
print(f"  {'Length':<10} {'oracle':>10} {'doc_kw':>10} {'para':>10} {'doc_kw/oracle':>14} {'para/oracle':>12}")
print(f"  {'-'*60}")
for lb in LENGTH_BINS:
    od = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    dd = analysis[lb].get('surr_doc_trunc', {}).get('d', 0)
    pd = analysis[lb].get('surr_para_trunc', {}).get('d', 0)
    dr = dd / od * 100 if od > 0 else 0
    pr = pd / od * 100 if od > 0 else 0
    print(f"  {lb:<10} {od:>+10.3f} {dd:>+10.3f} {pd:>+10.3f} {dr:>13.0f}% {pr:>11.0f}%")

# Implications for Exp 04
print(f"\n--- Implications for Exp 04 (Ranking) ---")
d_512 = analysis.get("512", {}).get('oracle_trunc', {}).get('d', 0)
p_512 = analysis.get("512", {}).get('oracle_trunc', {}).get('p', 1)
if p_512 < 0.05:
    print(f"  512-token d={d_512:+.3f} (p={p_512:.2e}) — ESCI product descriptions ({chr(126)}100-500 words) are in range")
    print(f"  Proceed with Exp 04")
else:
    print(f"  512-token d={d_512:+.3f} (ns) — ESCI product descriptions may be too long")
    print(f"  Consider filtering to shorter product texts in Exp 04")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp03_length_scaling',
    'model': MODEL_NAME,
    'n_samples': N_SAMPLES,
    'length_bins': LENGTH_BINS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'decay_data': decay_data,
    'padded_stats': padded_stats,
    'v2_oracle_d': v2_oracle_d,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
code("""# Cell 13: Cleanup
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

outpath = "experiments/03/03_length_scaling.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
