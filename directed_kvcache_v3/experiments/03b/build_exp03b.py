#!/usr/bin/env python3
"""Generate Exp 03B notebook: Extended Length Scaling with All Surrogate Types.

Exp 03 showed NO DECAY up to 2048 tokens with oracle/surr_doc/surr_para. But it
only tested content-specific surrogates. Exp 2B proved 85% of the benefit is
structural (random prefix captures 81% of oracle). The critical question:

  Does the structural switch still fire at very long documents?

This experiment extends to 6144 tokens (3x beyond Exp 03) and tests the full
surrogate hierarchy: random, static, template, doc keywords, and oracle.

Architecture note: T5Gemma encoder has sliding_window=1024 with full attention
every 6th layer. At 3072+ tokens, most encoder layers can't directly connect
distant document tokens to the surrogate. If the effect persists, the global
layers (every 6th) are sufficient for propagating surrogate influence.
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
md("""# Experiment 03B: Extended Length Scaling with All Surrogate Types
## Does the structural switch still fire at very long documents? How far can we push?

### Background
Exp 03 showed NO DECAY up to 2048 tokens for oracle/surr\\_doc/surr\\_para. But:
1. Only content-specific surrogates were tested
2. Exp 2B proved 85% of the benefit is structural (1 random word = 85% of oracle)
3. 2048 tokens is not actually that long for real-world documents

### Architecture constraint
T5Gemma encoder uses sliding\\_window=1024 with full attention every 6th layer (5 out of 34).
At document lengths >> 1024, most encoder layers can only connect nearby tokens to the
surrogate. The global layers must propagate the signal across the full document.

### The questions
1. Does `random\\_trunc` hold up at 4096+ tokens? (structural mechanism at scale)
2. Does the semantic gap (oracle - random) change with length?
3. Where does the encoder architecture actually break down?

### Conditions (6, all with truncation)
| Condition | Source | Type | Exp 02 d |
|-----------|--------|------|----------|
| bare | — | Lower bound | — |
| oracle\\_trunc | Real query | Upper bound | +0.376 |
| scrambled\\_oracle\\_trunc | Query words, random order | Vocabulary control | Exp 2B |
| random\\_matched\\_trunc | Random words, query length | Length-matched structural | Exp 2B |
| random\\_trunc | Unrelated passage | Structural control | +0.303 |
| static\\_fact\\_trunc | "What are the key facts?" | Content-agnostic | +0.372 |
| surr\\_template\\_trunc | "What is [keyword]?" | Doc-derived minimal | +0.336 |
| surr\\_doc\\_trunc | Top-5 TF keywords | Doc-derived | +0.322 |

### Three-way decomposition (at each length)
- **Structure** = bare → random\\_matched (any prefix helps)
- **Vocabulary** = random\\_matched → scrambled\\_oracle (right words, wrong order)
- **Semantics** = scrambled\\_oracle → oracle (right word order)

### Length bins (pushing 3x beyond Exp 03)
| Bin | Target tokens | Exp 03 oracle d | Sliding window coverage |
|-----|--------------|-----------------|------------------------|
| original | ~130 tok | +0.384*** | 100% within window |
| 512 | padded | +0.442*** | 100% within window |
| 1024 | padded | +0.452*** | ~100% (boundary) |
| 2048 | padded | +0.392*** | ~50% in window layers |
| 3072 | padded (new) | — | ~33% in window layers |
| 4096 | padded (new) | — | ~25% in window layers |
| 6144 | padded (new) | — | ~17% in window layers |

### N=400, Bonferroni for 49 comparisons (7 non-bare conditions x 7 lengths)
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

RESULTS_DIR = Path("../../results/exp03b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 400
MODEL_NAME = "google/t5gemma-2-4b-4b"
LENGTH_BINS = ["original", "512", "1024", "2048", "3072", "4096", "6144"]
N_BONFERRONI = 49  # 7 non-bare conditions x 7 lengths

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 03B: Extended Length Scaling with All Surrogate Types")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"Length bins: {LENGTH_BINS}")
print(f"Bonferroni comparisons: {N_BONFERRONI}")
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
code(r"""# Cell 4: Scoring helpers

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    '''Score NLL of answer tokens with optional truncation.'''
    # Tokenize encoder input — no truncation limit (let model handle long sequences)
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=8192).input_ids.to(DEVICE)
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
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
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
            attention_mask=cross_attn_mask,
            labels=ans_ids,
        )

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

def make_surrogate_doc_kw(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"

STATIC_FACT = "What are the key facts I need to know?"

# Build a vocabulary pool for random_matched surrogates
# Use content words from all passages — populated after data loading
VOCAB_POOL = []

def make_surrogate_random_matched(query, vocab_pool, rng):
    '''Random words, same count as query — length-matched structural control.'''
    n_words = len(query.split())
    if len(vocab_pool) == 0:
        return query  # fallback
    words = rng.choice(vocab_pool, size=min(n_words, len(vocab_pool)), replace=True).tolist()
    return " ".join(words)

def make_surrogate_scrambled_oracle(query, rng):
    '''Query words in random order — vocabulary control.'''
    words = query.split()
    if len(words) <= 1:
        return query
    rng.shuffle(words)
    return " ".join(words)

print("Helpers defined.")
print("  Conditions: bare, oracle_trunc, scrambled_oracle_trunc, random_matched_trunc,")
print("              random_trunc, static_fact_trunc, surr_template_trunc, surr_doc_trunc")
print("  Three-way decomposition: Structure | Vocabulary | Semantics")
print(f"  NOTE: max_length=8192 for long padded documents")""")

# ============================================================
code("""# Cell 5: Load data and build padding pool
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Collect target samples AND a large padding pool
samples = []
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
            padding_pool.append(pt)

    # Need a very large pool for 6144-token documents
    if len(samples) >= N_SAMPLES * 3 and len(padding_pool) >= 20000:
        break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
np.random.shuffle(padding_pool)

del ds
gc.collect()

# Build vocabulary pool from all passages for random_matched
all_passage_words = []
for s in samples:
    all_passage_words.extend(extract_keywords(s['passage']))
VOCAB_POOL.extend(list(set(all_passage_words)))
print(f"Vocabulary pool: {len(VOCAB_POOL)} unique content words")

# Generate surrogates
rng = np.random.RandomState(SEED + 1)  # Separate RNG for surrogates
for i, s in enumerate(samples):
    s['surr_doc_kw'] = make_surrogate_doc_kw(s['passage'])
    s['surr_template'] = make_surrogate_template(s['passage'])
    # Random: use passage from a different sample (circular offset)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    s['surr_random'] = " ".join(samples[other_idx]['passage'].split()[:20])
    # Decomposition surrogates
    s['surr_scrambled_oracle'] = make_surrogate_scrambled_oracle(s['query'], rng)
    s['surr_random_matched'] = make_surrogate_random_matched(s['query'], np.array(VOCAB_POOL), rng)

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
    "original": None,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
    "3072": 3072,
    "4096": 4096,
    "6144": 6144,
}

def pad_passage_to_length(passage, target_tokens, padding_pool, pool_offset):
    '''Pad a passage to target_tokens by appending unrelated passages.'''
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
        if idx >= len(padding_pool):
            idx = 0
        candidate = padded + "\n\n" + padding_pool[idx]
        candidate_ids = tokenizer(candidate, add_special_tokens=True).input_ids
        if len(candidate_ids) >= target_tokens:
            # Fine-grained: add words until we hit target
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
padded_docs = {}
padded_stats = {}

for length_bin, target_tokens in TARGET_LENGTHS.items():
    padded_docs[length_bin] = []
    tok_counts = []

    for i, s in enumerate(samples):
        pool_offset = i * 100  # Wider spread for longer docs
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

    print(f"  {length_bin:>8s}: mean={padded_stats[length_bin]['mean']:.0f} tokens "
          f"(min={padded_stats[length_bin]['min']}, max={padded_stats[length_bin]['max']}, "
          f"median={padded_stats[length_bin]['median']:.0f})")

# Preview
print(f"\n--- Sample 0 preview ---")
print(f"  Query:  {samples[0]['query'][:80]}")
print(f"  Answer: {samples[0]['answer'][:80]}")
for lb in LENGTH_BINS:
    preview = padded_docs[lb][0]
    tok_count = len(tokenizer(preview, add_special_tokens=True).input_ids)
    print(f"  {lb:>8s}: {tok_count} tokens")

# Show actual prefix text for each non-bare condition
print(f"\n--- Actual prefix text for each condition (sample 0) ---")
ex = samples[0]
ex_doc = padded_docs["original"][0]
surr_items = {
    'oracle': ex['query'],
    'scrambled_oracle': ex['surr_scrambled_oracle'],
    'random_matched': ex['surr_random_matched'],
    'random (~20w)': ex['surr_random'],
    'static_fact': STATIC_FACT,
    'surr_template': ex['surr_template'],
    'surr_doc (kw)': ex['surr_doc_kw'],
}
for name, text in surr_items.items():
    ptoks = count_prefix_tokens(text, ex_doc)
    print(f"  {name:<22} ({ptoks:>3} prefix toks): {str(text)[:60]}")""")

# ============================================================
code(r"""# Cell 7: Explain conditions and estimate runtime
print("=" * 70)
print("EXPERIMENTAL CONDITIONS")
print("=" * 70)

ex = samples[0]
ex_doc = padded_docs["original"][0]

oracle_prefix = count_prefix_tokens(ex['query'], ex_doc)
scrambled_prefix = count_prefix_tokens(ex['surr_scrambled_oracle'], ex_doc)
matched_prefix = count_prefix_tokens(ex['surr_random_matched'], ex_doc)
random_prefix = count_prefix_tokens(ex['surr_random'], ex_doc)
static_prefix = count_prefix_tokens(STATIC_FACT, ex_doc)
template_prefix = count_prefix_tokens(ex['surr_template'], ex_doc)
doc_prefix = count_prefix_tokens(ex['surr_doc_kw'], ex_doc)

print("CONDITIONS (all with truncation):")
print()
print(f"  {'CONDITION':<26} {'SURROGATE TYPE':<28} {'PREFIX TOKENS':<16}")
print(f"  {'-'*70}")
print(f"  {'bare':<26} {'(none)':<28} {'0':<16}")
print(f"  {'oracle_trunc':<26} {'Real query':<28} {'~' + str(oracle_prefix):<16}")
print(f"  {'scrambled_oracle_trunc':<26} {'Query words, random order':<28} {'~' + str(scrambled_prefix):<16}")
print(f"  {'random_matched_trunc':<26} {'Random words, query length':<28} {'~' + str(matched_prefix):<16}")
print(f"  {'random_trunc':<26} {'Unrelated passage (~20w)':<28} {'~' + str(random_prefix):<16}")
print(f"  {'static_fact_trunc':<26} {'Fixed question':<28} {'~' + str(static_prefix):<16}")
print(f"  {'surr_template_trunc':<26} {'What is [keyword]?':<28} {'~' + str(template_prefix):<16}")
print(f"  {'surr_doc_trunc':<26} {'Top-5 TF keywords':<28} {'~' + str(doc_prefix):<16}")
print()
print("THREE-WAY DECOMPOSITION:")
print("  Structure  = bare -> random_matched       (any prefix helps)")
print("  Vocabulary = random_matched -> scrambled   (right words, wrong order)")
print("  Semantics  = scrambled -> oracle           (right word order)")
print()
print(f"LENGTH BINS: {LENGTH_BINS}")
print(f"ENCODER SLIDING WINDOW: 1024 tokens (full attention every 6th layer)")
N_CONDITIONS = 8  # bare + 7 non-bare
n_calls = N_SAMPLES * len(LENGTH_BINS) * N_CONDITIONS
print(f"Total scoring calls: {N_SAMPLES} x {len(LENGTH_BINS)} x {N_CONDITIONS} = {n_calls}")
print(f"Bonferroni: {N_BONFERRONI} comparisons (7 conditions x 7 lengths)")

# Rough runtime estimate: longer docs take longer
avg_time_per_call = {
    "original": 0.4, "512": 0.6, "1024": 0.9, "2048": 1.5,
    "3072": 2.2, "4096": 3.0, "6144": 5.0,
}
total_est = sum(N_SAMPLES * N_CONDITIONS * avg_time_per_call.get(lb, 3.0) for lb in LENGTH_BINS)
print(f"Estimated runtime: ~{total_est/3600:.1f} hours")""")

# ============================================================
code(r"""# Cell 8: Run scoring — outer loop over length bins
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'scrambled_oracle_trunc', 'random_matched_trunc',
              'random_trunc', 'static_fact_trunc', 'surr_template_trunc', 'surr_doc_trunc']

def make_conditions(sample, padded_passage):
    '''Return dict of {name: (encoder_text, prefix_token_count, truncate)}'''
    query = sample['query']
    surr_map = {
        'oracle':           query,
        'scrambled_oracle':  sample['surr_scrambled_oracle'],
        'random_matched':    sample['surr_random_matched'],
        'random':            sample['surr_random'],
        'static_fact':       STATIC_FACT,
        'surr_template':     sample['surr_template'],
        'surr_doc':          sample['surr_doc_kw'],
    }

    conditions = {'bare': (padded_passage, 0, False)}

    for surr_name, surr_text in surr_map.items():
        cond_name = f'{surr_name}_trunc'
        enc_text = surr_text + "\n" + padded_passage
        prefix_count = count_prefix_tokens(surr_text, padded_passage)
        conditions[cond_name] = (enc_text, prefix_count, True)

    return conditions

# Resume from checkpoint
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

    # Quick peek
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    oracle_nlls = np.array([r['nll_oracle_trunc'] for r in bin_results])
    random_nlls = np.array([r['nll_random_trunc'] for r in bin_results])
    from lib.analysis import cohens_d
    d_oracle = cohens_d(bare_nlls - oracle_nlls)
    d_random = cohens_d(bare_nlls - random_nlls)
    print(f"  Quick peek: oracle d={d_oracle:+.3f}, random d={d_random:+.3f}")

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

results_by_bin = {}
for length_bin in LENGTH_BINS:
    results_by_bin[length_bin] = all_checkpoint[length_bin]['results']

analysis = {}
for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    n = len(bin_results)
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    mean_tokens = np.mean([r['padded_tokens'] for r in bin_results])

    print(f"\n--- {length_bin} (mean {mean_tokens:.0f} tokens, N={n}) ---")
    print(f"  {'Condition':<22} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
    print(f"  {'-'*82}")

    analysis[length_bin] = {}
    for cond in COND_NAMES:
        nlls = np.array([r[f'nll_{cond}'] for r in bin_results])
        mean_nll = nlls.mean()
        diff = bare_nlls - nlls

        if cond == 'bare':
            print(f"  {cond:<22} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
            analysis[length_bin][cond] = {'mean_nll': float(mean_nll)}
        else:
            d = cohens_d(diff)
            win_pct = 100 * np.mean(diff > 0)
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            sig = '***' if p_val < 0.001/N_BONFERRONI else '**' if p_val < 0.01/N_BONFERRONI else '*' if p_val < 0.05/N_BONFERRONI else 'ns'
            print(f"  {cond:<22} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
            analysis[length_bin][cond] = {
                'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
                'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            }""")

# ============================================================
code(r"""# Cell 10: Structural vs Semantic analysis across lengths
print("=" * 70)
print("STRUCTURAL vs SEMANTIC ACROSS LENGTHS")
print("=" * 70)

# Key comparison: random (structural) vs oracle (structural + semantic)
print(f"\n--- Structural floor (random) vs Oracle across lengths ---")
print(f"  {'Length':<10} {'oracle d':>10} {'random d':>10} {'gap':>8} {'random/oracle':>14} {'semantic p':>12}")
print(f"  {'-'*68}")

for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    oracle_nlls = np.array([r['nll_oracle_trunc'] for r in bin_results])
    random_nlls = np.array([r['nll_random_trunc'] for r in bin_results])

    d_oracle = cohens_d(bare_nlls - oracle_nlls)
    d_random = cohens_d(bare_nlls - random_nlls)
    gap = d_oracle - d_random
    ratio = d_random / d_oracle * 100 if d_oracle > 0 else 0

    # Direct oracle vs random test
    diff_or = oracle_nlls - random_nlls  # positive = random has higher NLL (oracle better)
    t_or, p_or = stats.ttest_1samp(diff_or, 0) if np.std(diff_or) > 0 else (0, 1)

    print(f"  {length_bin:<10} {d_oracle:>+10.3f} {d_random:>+10.3f} {gap:>+8.3f} {ratio:>13.0f}% {p_or:>12.2e}")

# All surrogate types across lengths
print(f"\n--- All conditions: Cohen's d vs Length ---")
header = f"  {'Length':<10}"
for cond in COND_NAMES[1:]:  # skip bare
    short = cond.replace('_trunc', '').replace('surr_', '').replace('static_', 's_')
    header += f" {short:>10}"
print(header)
print(f"  {'-'*(10 + 11 * len(COND_NAMES[1:]))}")

for length_bin in LENGTH_BINS:
    row = f"  {length_bin:<10}"
    for cond in COND_NAMES[1:]:
        a = analysis[length_bin].get(cond, {})
        d = a.get('d', 0)
        p = a.get('p', 1)
        sig = '***' if p < 0.001/N_BONFERRONI else ' **' if p < 0.01/N_BONFERRONI else '  *' if p < 0.05/N_BONFERRONI else ' ns'
        row += f" {d:>+6.3f}{sig}"

    print(row)

# Content-specific vs content-agnostic by length
print(f"\n--- Content-specific vs Content-agnostic by length ---")
print(f"  {'Length':<10} {'content-spec':>14} {'content-agn':>14} {'gap':>8}")
print(f"  {'-'*50}")

for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])

    # Content-specific: surr_doc, surr_template
    cs_nlls = np.mean([
        np.array([r['nll_surr_doc_trunc'] for r in bin_results]),
        np.array([r['nll_surr_template_trunc'] for r in bin_results]),
    ], axis=0)
    d_cs = cohens_d(bare_nlls - cs_nlls)

    # Content-agnostic: random, static_fact
    ca_nlls = np.mean([
        np.array([r['nll_random_trunc'] for r in bin_results]),
        np.array([r['nll_static_fact_trunc'] for r in bin_results]),
    ], axis=0)
    d_ca = cohens_d(bare_nlls - ca_nlls)

    gap = d_cs - d_ca
    print(f"  {length_bin:<10} {d_cs:>+14.3f} {d_ca:>+14.3f} {gap:>+8.3f}")

# Three-way decomposition at each length
print(f"\n--- THREE-WAY DECOMPOSITION (Structure / Vocabulary / Semantics) ---")
print(f"  {'Length':<10} {'Structure':>12} {'Vocabulary':>12} {'Semantics':>12} {'Total':>10} {'Struct%':>10} {'Vocab%':>10} {'Sem%':>10}")
print(f"  {'-'*90}")

for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    oracle_nlls = np.array([r['nll_oracle_trunc'] for r in bin_results])
    scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in bin_results])
    matched_nlls = np.array([r['nll_random_matched_trunc'] for r in bin_results])

    # Three-way decomposition (per sample, then average)
    structure = bare_nlls - matched_nlls          # bare -> random_matched
    vocabulary = matched_nlls - scrambled_nlls     # random_matched -> scrambled_oracle
    semantics = scrambled_nlls - oracle_nlls       # scrambled_oracle -> oracle
    total = bare_nlls - oracle_nlls                # bare -> oracle

    s_mean = structure.mean()
    v_mean = vocabulary.mean()
    sem_mean = semantics.mean()
    t_mean = total.mean()

    s_pct = s_mean / t_mean * 100 if t_mean > 0 else 0
    v_pct = v_mean / t_mean * 100 if t_mean > 0 else 0
    sem_pct = sem_mean / t_mean * 100 if t_mean > 0 else 0

    print(f"  {length_bin:<10} {s_mean:>+12.4f} {v_mean:>+12.4f} {sem_mean:>+12.4f} {t_mean:>+10.4f} {s_pct:>9.0f}% {v_pct:>9.0f}% {sem_pct:>9.0f}%")

# Statistical tests for each component
print(f"\n--- Component significance at each length ---")
print(f"  {'Length':<10} {'Structure':>20} {'Vocabulary':>20} {'Semantics':>20}")
print(f"  {'-'*72}")

for length_bin in LENGTH_BINS:
    bin_results = results_by_bin[length_bin]
    bare_nlls = np.array([r['nll_bare'] for r in bin_results])
    oracle_nlls = np.array([r['nll_oracle_trunc'] for r in bin_results])
    scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in bin_results])
    matched_nlls = np.array([r['nll_random_matched_trunc'] for r in bin_results])

    structure = bare_nlls - matched_nlls
    vocabulary = matched_nlls - scrambled_nlls
    semantics = scrambled_nlls - oracle_nlls

    parts = []
    for comp, label in [(structure, 'struct'), (vocabulary, 'vocab'), (semantics, 'sem')]:
        d = cohens_d(comp)
        t, p = stats.ttest_1samp(comp, 0)
        sig = '***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI else '*' if p < 0.05/N_BONFERRONI else 'ns'
        parts.append(f"d={d:+.3f} {sig:>3s}")

    print(f"  {length_bin:<10} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")""")


# ============================================================
code(r"""# Cell 11: Decay curve plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_tokens = [padded_stats[lb]['mean'] for lb in LENGTH_BINS]

# --- Panel 1: All conditions decay curves ---
ax = axes[0, 0]
colors = {
    'oracle_trunc': 'tab:red', 'scrambled_oracle_trunc': 'tab:pink',
    'random_matched_trunc': 'tab:olive', 'random_trunc': 'tab:gray',
    'static_fact_trunc': 'tab:orange', 'surr_template_trunc': 'tab:purple',
    'surr_doc_trunc': 'tab:blue',
}
markers = {
    'oracle_trunc': 'o', 'scrambled_oracle_trunc': 'v',
    'random_matched_trunc': 'P', 'random_trunc': 'x',
    'static_fact_trunc': 's', 'surr_template_trunc': '^',
    'surr_doc_trunc': 'D',
}

for cond in COND_NAMES[1:]:
    d_vals = [analysis[lb].get(cond, {}).get('d', 0) for lb in LENGTH_BINS]
    ax.plot(x_tokens, d_vals, f'-{markers[cond]}', color=colors[cond],
            label=cond.replace('_trunc', ''), markersize=7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1024, color='gray', linestyle=':', alpha=0.3, label='sliding window')
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('All Conditions vs Length')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Panel 2: Structural vs Semantic gap ---
ax = axes[0, 1]
oracle_d = [analysis[lb].get('oracle_trunc', {}).get('d', 0) for lb in LENGTH_BINS]
random_d = [analysis[lb].get('random_trunc', {}).get('d', 0) for lb in LENGTH_BINS]
semantic_gap = [o - r for o, r in zip(oracle_d, random_d)]

ax.plot(x_tokens, oracle_d, '-o', color='tab:red', label='oracle (structural+semantic)', markersize=7)
ax.plot(x_tokens, random_d, '-x', color='tab:gray', label='random (structural only)', markersize=7)
ax.fill_between(x_tokens, random_d, oracle_d, alpha=0.2, color='tab:green', label='semantic gap')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1024, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('Structural vs Semantic Decomposition')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Panel 3: Cross-architecture comparison (oracle only) ---
ax = axes[1, 0]
ax.plot(x_tokens, oracle_d, '-o', color='tab:red', label='v3 T5Gemma oracle_trunc', markersize=7)
ax.plot(x_tokens, random_d, '-x', color='tab:gray', label='v3 T5Gemma random_trunc', markersize=7)

# v2 Exp 20 data (only where lengths overlap)
v2_data = {"original": 0.303, "512": 0.034, "1024": -0.043}
v2_x = [padded_stats[lb]['mean'] for lb in v2_data.keys() if lb in padded_stats]
v2_y = [v2_data[lb] for lb in v2_data.keys() if lb in padded_stats]
ax.plot(v2_x, v2_y, '-s', color='tab:purple', label='v2 Gemma 3 4B oracle', markersize=7)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1024, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Document Length (tokens)')
ax.set_ylabel("Cohen's d (vs bare)")
ax.set_title('Cross-Architecture: v3 vs v2')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks(x_tokens)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.grid(True, alpha=0.3)

# --- Panel 4: Three-way decomposition stacked ---
ax = axes[1, 1]
struct_pcts = []
vocab_pcts = []
sem_pcts = []
for lb in LENGTH_BINS:
    br = results_by_bin[lb]
    bare = np.array([r['nll_bare'] for r in br])
    orc = np.array([r['nll_oracle_trunc'] for r in br])
    scr = np.array([r['nll_scrambled_oracle_trunc'] for r in br])
    mat = np.array([r['nll_random_matched_trunc'] for r in br])
    total = (bare - orc).mean()
    if total > 0:
        struct_pcts.append((bare - mat).mean() / total * 100)
        vocab_pcts.append((mat - scr).mean() / total * 100)
        sem_pcts.append((scr - orc).mean() / total * 100)
    else:
        struct_pcts.append(0)
        vocab_pcts.append(0)
        sem_pcts.append(0)

x_pos = np.arange(len(LENGTH_BINS))
ax.bar(x_pos, struct_pcts, label='Structure', color='tab:gray', alpha=0.8)
ax.bar(x_pos, vocab_pcts, bottom=struct_pcts, label='Vocabulary', color='tab:orange', alpha=0.8)
bottoms = [s + v for s, v in zip(struct_pcts, vocab_pcts)]
ax.bar(x_pos, sem_pcts, bottom=bottoms, label='Semantics', color='tab:red', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(LENGTH_BINS, rotation=45)
ax.set_ylabel('% of Oracle Benefit')
ax.set_title('Three-Way Decomposition vs Length')
ax.legend(fontsize=8)
ax.set_ylim(0, 120)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = RESULTS_DIR / 'decay_curves_extended.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")""")

# ============================================================
code(r"""# Cell 12: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 03B: Extended Length Scaling")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {N_SAMPLES} samples per length bin")
print(f"Length bins: {LENGTH_BINS}")
print(f"Encoder: sliding_window=1024, full attention every 6th layer (5/34 layers)")

# Key question 1: Does random hold up?
print(f"\n--- Q1: Does the structural switch hold at long documents? ---")
for lb in LENGTH_BINS:
    d = analysis[lb].get('random_trunc', {}).get('d', 0)
    p = analysis[lb].get('random_trunc', {}).get('p', 1)
    sig = "SIGNIFICANT" if p < 0.05/N_BONFERRONI else "ns"
    print(f"  {lb:>8s}: random d={d:+.3f} [{sig}]")

# Key question 2: Does semantic gap change with length?
print(f"\n--- Q2: Does semantic gap (oracle - random) change with length? ---")
semantic_gaps = []
for lb in LENGTH_BINS:
    d_oracle = analysis[lb].get('oracle_trunc', {}).get('d', 0)
    d_random = analysis[lb].get('random_trunc', {}).get('d', 0)
    gap = d_oracle - d_random
    semantic_gaps.append(gap)
    print(f"  {lb:>8s}: oracle d={d_oracle:+.3f}, random d={d_random:+.3f}, semantic gap={gap:+.3f}")

# Trend in semantic gap
from scipy.stats import spearmanr
mean_tokens_list = [padded_stats[lb]['mean'] for lb in LENGTH_BINS]
rho, p_rho = spearmanr(mean_tokens_list, semantic_gaps)
print(f"\n  Spearman correlation (length vs semantic gap): rho={rho:+.3f} (p={p_rho:.3f})")
if p_rho < 0.05:
    if rho > 0:
        print(f"  Semantic gap GROWS with length — content matters MORE for long documents")
    else:
        print(f"  Semantic gap SHRINKS with length — structural mechanism dominates at scale")
else:
    print(f"  No significant trend — semantic gap is stable across lengths")

# Key question 3: Where does it break down?
print(f"\n--- Q3: At what length does the encoder break down? ---")
last_sig = {}
for cond in COND_NAMES[1:]:
    for lb in LENGTH_BINS:
        p = analysis[lb].get(cond, {}).get('p', 1)
        if p < 0.05/N_BONFERRONI:
            last_sig[cond] = lb
    sig_lb = last_sig.get(cond, 'none')
    print(f"  {cond:<22s}: significant up to {sig_lb}")

# Overall verdict
print(f"\n--- Overall Verdict ---")
all_sig = all(
    analysis[lb].get('random_trunc', {}).get('p', 1) < 0.05/N_BONFERRONI
    for lb in LENGTH_BINS
)
if all_sig:
    print(f"  STRUCTURAL SWITCH holds at ALL lengths up to {LENGTH_BINS[-1]} tokens")
    print(f"  Random prefix is sufficient — no need for content-specific surrogates")
    print(f"  The encoder's global attention layers (every 6th) propagate the signal")
else:
    first_ns = next(
        (lb for lb in LENGTH_BINS
         if analysis[lb].get('random_trunc', {}).get('p', 1) >= 0.05/N_BONFERRONI),
        None
    )
    print(f"  Structural switch breaks down at {first_ns} tokens")
    print(f"  Beyond this, content-specific surrogates may be needed")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp03b_extended_length_scaling',
    'model': MODEL_NAME,
    'n_samples': N_SAMPLES,
    'length_bins': LENGTH_BINS,
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'padded_stats': padded_stats,
    'encoder_config': {
        'sliding_window': 1024,
        'full_attention_every': 6,
        'num_layers': 34,
        'num_global_layers': 5,
    },
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

outpath = "experiments/03b/03b_extended_length_scaling.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
