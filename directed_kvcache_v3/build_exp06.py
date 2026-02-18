#!/usr/bin/env python3
# Build Exp 06 notebook: Factoid Subsample Validation.
#
# Tests whether the semantic component dominates for short factoid answers.
# Prior finding: the "85% structural" aggregate hides two populations:
#   - Short answers (<=5 words): ~63% semantic, query content matters
#   - Long answers (>9 words): ~113% structural, query content hurts
#
# Design: Fresh 500 MS MARCO samples filtered to answer <= 5 words.
# 8 conditions x 500 = 4,000 scoring passes. ~13 min.
#
# Prediction: structural% drops from 85% to ~40-50%.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 06: Factoid Subsample Validation

## Motivation

The "85% structural" finding from Exp 2B is an aggregate that hides two distinct
populations (discovered in post-hoc analysis of Exp 02/05 data):

| Population | N (in Exp 02) | Structural % | Semantic % | Characteristic |
|---|---|---|---|---|
| Short factoid answers (<=9 words) | ~248 | ~37% | ~63% | Disambiguation required |
| Long descriptive answers (>9 words) | ~252 | ~113% | ~-13% | Broad generation |
| **Weighted average** | 500 | **85%** | **15%** | |

For short precise answers, the model must select ONE specific fact from the passage.
The query tells it which fact. A random prefix fixes the attention sink but cannot
help with disambiguation. For long answers, broad representation suffices.

This experiment validates the finding on a **fresh sample** of 500 MS MARCO examples
filtered to short factoid answers (<=5 words), eliminating look-ahead bias from
the post-hoc analysis.

## Prediction

If the two-population hypothesis is correct:
- Structural% should drop from 85% to **~40-50%**
- Vocabulary and semantics should contribute **~50-60%** combined
- surr_template should significantly beat random (it couldn't on the mixed sample)
- The "directed" in directed KV cache should finally matter

## Conditions (8)

| # | Condition | Prefix | Role |
|---|-----------|--------|------|
| 1 | `bare` | (none) | lower bound |
| 2 | `oracle_x1_trunc` | query x 1 | upper bound |
| 3 | `oracle_x4_trunc` | query x 4 | upper bound + rep |
| 4 | `random_x1_trunc` | random_matched x 1 | structural control |
| 5 | `random_x4_trunc` | random_matched x 4 | structural + rep |
| 6 | `scrambled_oracle_trunc` | shuffled query x 1 | vocabulary control |
| 7 | `surr_template_x1_trunc` | "What is [kw]?" x 1 | heuristic |
| 8 | `surr_template_x4_trunc` | "What is [kw]?" x 4 | heuristic + rep |

## Analysis

1. Baseline characterization (verify short-answer distribution)
2. 3-way decomposition (structure / vocabulary / semantics) — compare to Exp 2B
3. Surrogate comparison: does template beat random for factoid QA?
4. Hardness interaction: does semantic% vary by difficulty within factoid?
5. Direct comparison with Exp 2B full-sample results""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, re, gc, random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, ".")
from lib.analysis import cohens_d

SEED = 43  # Different seed from Exp 02 (42) for a fresh sample
N_SAMPLES = 500
MAX_ANSWER_WORDS = 5
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("results/exp06")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

print("Exp 06: Factoid Subsample Validation")
print(f"N: {N_SAMPLES}, max answer words: {MAX_ANSWER_WORDS}")
""")


# ===== Cell 3: Load MS MARCO + filter to short answers =====
code(r"""# Cell 3: Load MS MARCO and filter to short factoid answers
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Collect ALL eligible samples (not just 3*N), then filter to short answers
all_candidates = []
for item in ds:
    if len(all_candidates) >= 20000:
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
    # Filter: answer must be short (factoid)
    answer_words = count_words(answer)
    if answer_words > MAX_ANSWER_WORDS:
        continue
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            all_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc, 'answer_words': answer_words,
            })
            break

print(f"Total short-answer candidates: {len(all_candidates)}")

# Shuffle with our seed and take N_SAMPLES
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

# Dataset statistics
passage_words = np.array([s['word_count'] for s in samples])
answer_words = np.array([s['answer_words'] for s in samples])
query_words = np.array([len(s['query'].split()) for s in samples])

print(f"\nSample statistics (N={N_SAMPLES}):")
print(f"  Answer length:  mean={answer_words.mean():.1f}, median={np.median(answer_words):.0f}, "
      f"range=[{answer_words.min()}, {answer_words.max()}]")
print(f"  Answer distribution: " + ", ".join(
    f"{w}w={np.sum(answer_words==w)}" for w in range(1, MAX_ANSWER_WORDS + 1)))
print(f"  Passage length: mean={passage_words.mean():.1f}, median={np.median(passage_words):.0f}")
print(f"  Query length:   mean={query_words.mean():.1f}, median={np.median(query_words):.0f}")

# Show 5 examples
for i in range(5):
    s = samples[i]
    print(f"\nExample {i}:")
    print(f"  Q: {s['query']}")
    print(f"  A ({s['answer_words']}w): {s['answer']}")
    print(f"  P ({s['word_count']}w): {s['passage'][:100]}...")

# Compare with Exp 02 distribution
print(f"\n--- Comparison with Exp 02 (mixed answers) ---")
print(f"  Exp 02: mean answer ~14w (range 1-96), mean passage ~74w")
print(f"  This:   mean answer {answer_words.mean():.1f}w (range {answer_words.min()}-{answer_words.max()}), "
      f"mean passage {passage_words.mean():.0f}w")
""")


# ===== Cell 4: Load model + define helpers =====
code(r"""# Cell 4: Load model and define scoring helpers
from dotenv import load_dotenv
load_dotenv()
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
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
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
    # Count how many tokens the prefix occupies in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


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

print("Helpers defined.")
""")


# ===== Cell 5: Generate conditions per sample =====
code(r"""# Cell 5: Generate all 8 scoring conditions per sample

for i, s in enumerate(samples):
    query = s['query']
    passage = s['passage']
    query_words_list = query.split()

    # Random text from unrelated passage
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    random_matched = " ".join(other_words[:len(query_words_list)])

    # Scrambled oracle (same words, random order)
    rng = np.random.RandomState(SEED + i)
    shuffled = list(query_words_list)
    rng.shuffle(shuffled)
    scrambled = " ".join(shuffled)

    # Template surrogate: "What is [keyword]?"
    doc_words_clean = re.sub(r'[^\w\s]', '', passage.lower()).split()
    content = [w for w in doc_words_clean if w not in STOP_WORDS and len(w) > 2]
    if content:
        kw = Counter(content).most_common(1)[0][0]
    else:
        kw = "information"

    s['oracle_x1'] = query
    s['oracle_x4'] = " ".join([query] * 4)
    s['random_x1'] = random_matched
    s['random_x4'] = " ".join([random_matched] * 4)
    s['scrambled_oracle'] = scrambled
    s['surr_template_x1'] = f"What is {kw}?"
    s['surr_template_x4'] = " ".join([f"What is {kw}?"] * 4)

COND_NAMES = [
    'bare',
    'oracle_x1_trunc',
    'oracle_x4_trunc',
    'random_x1_trunc',
    'random_x4_trunc',
    'scrambled_oracle_trunc',
    'surr_template_x1_trunc',
    'surr_template_x4_trunc',
]

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query: {ex['query']}")
print(f"  Answer ({ex['answer_words']}w): {ex['answer']}")
for c in COND_NAMES:
    if c == 'bare':
        print(f"  {c:<30}: [document only]")
    else:
        key = c.replace('_trunc', '')
        text = ex[key]
        ptoks = count_prefix_tokens(text, ex['passage'])
        print(f"  {c:<30} ({ptoks:>3} toks): {str(text)[:55]}")
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop with checkpointing

print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

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
          f"= {len(COND_NAMES) * N_SAMPLES} scorings")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
        'answer_words': s['answer_words'],
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['passage'], s['answer'])
            result['nll_bare'] = nll
        else:
            key = cond.replace('_trunc', '')
            prefix = s[key]
            enc_text = prefix + "\n" + s['passage']
            ptoks = count_prefix_tokens(prefix, s['passage'])
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result[f'nll_{cond}'] = nll
            result[f'ptoks_{cond}'] = ptoks

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


# ===== Cell 7: Part 1 — Baseline characterization =====
code(r"""# Cell 7: Part 1 — Baseline Characterization

print("=" * 70)
print("PART 1: BASELINE CHARACTERIZATION")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_x1_nlls = np.array([r['nll_oracle_x1_trunc'] for r in results])
oracle_benefit = bare_nlls - oracle_x1_nlls
oracle_d = cohens_d(oracle_benefit)

print(f"\nBaseline (bare):  mean NLL = {bare_nlls.mean():.4f}, std = {bare_nlls.std():.4f}")
print(f"Oracle (x1):      mean NLL = {oracle_x1_nlls.mean():.4f}")
print(f"Oracle headroom:  delta = {oracle_benefit.mean():+.4f}, d = {oracle_d:+.3f}")

_, p_oracle = stats.ttest_1samp(oracle_benefit, 0)
win_rate = np.mean(oracle_benefit > 0) * 100
print(f"Oracle win rate:  {win_rate:.1f}% (p={p_oracle:.2e})")

# Answer length distribution
aw = np.array([r['answer_words'] for r in results])
print(f"\nAnswer distribution: " + ", ".join(
    f"{w}w: {np.sum(aw==w)} ({np.sum(aw==w)/len(aw)*100:.0f}%)" for w in range(1, MAX_ANSWER_WORDS + 1)))

# Comparison with Exp 02
print(f"\n--- Comparison with Exp 02 (mixed answers) ---")
print(f"  Exp 02: bare NLL=3.68, oracle d=+0.376, headroom=+0.684")
print(f"  This:   bare NLL={bare_nlls.mean():.2f}, oracle d={oracle_d:+.3f}, "
      f"headroom={oracle_benefit.mean():+.3f}")

# All conditions overview
print(f"\n{'Condition':<30} {'NLL':>8} {'Delta':>8} {'d':>8} {'Win%':>7} {'%Orc':>6}")
print("-" * 75)

all_cond_pairs = [
    ('oracle_x1_trunc', 'Oracle x1'),
    ('oracle_x4_trunc', 'Oracle x4'),
    ('scrambled_oracle_trunc', 'Scrambled oracle'),
    ('surr_template_x1_trunc', 'Template x1'),
    ('surr_template_x4_trunc', 'Template x4'),
    ('random_x1_trunc', 'Random x1'),
    ('random_x4_trunc', 'Random x4'),
]

for cond, desc in all_cond_pairs:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"  {desc:<28} {nlls.mean():>8.4f} {delta:>+8.4f} {d:>+8.3f} {win:>6.1f}% {pct:>5.0f}%")

print(f"  {'bare (lower bound)':<28} {bare_nlls.mean():>8.4f}")
""")


# ===== Cell 8: Part 2 — 3-Way Decomposition =====
code(r"""# Cell 8: Part 2 — 3-Way Decomposition

print("=" * 70)
print("PART 2: 3-WAY DECOMPOSITION (the key test)")
print("=" * 70)
print("Decompose: bare -> random_x1 -> scrambled_oracle -> oracle_x1")
print("  Structure:  bare -> random_x1 (any prefix helps)")
print("  Vocabulary: random_x1 -> scrambled (right words, wrong order)")
print("  Semantics:  scrambled -> oracle (right word order)\n")

random_x1_nlls = np.array([r['nll_random_x1_trunc'] for r in results])
scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in results])

struct_comp = bare_nlls - random_x1_nlls
vocab_comp = random_x1_nlls - scrambled_nlls
sem_comp = scrambled_nlls - oracle_x1_nlls
total_comp = bare_nlls - oracle_x1_nlls

total_mean = total_comp.mean()

print(f"  {'Component':<20} {'Delta':>10} {'%total':>8} {'d':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*65}")

for label, comp in [('Structure', struct_comp), ('Vocabulary', vocab_comp),
                    ('Semantics', sem_comp)]:
    mu = comp.mean()
    pct = mu / total_mean * 100 if total_mean != 0 else 0
    d = cohens_d(comp)
    _, p = stats.ttest_1samp(comp, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<20} {mu:>+10.4f} {pct:>7.1f}% {d:>+8.3f} {p:>12.2e} {sig}")

print(f"  {'TOTAL':<20} {total_mean:>+10.4f} {'100.0%':>8}")
residual = total_mean - (struct_comp.mean() + vocab_comp.mean() + sem_comp.mean())
print(f"\n  Decomposition residual: {residual:.6f}")

struct_pct = struct_comp.mean() / total_mean * 100 if total_mean != 0 else 0
vocab_pct = vocab_comp.mean() / total_mean * 100 if total_mean != 0 else 0
sem_pct = sem_comp.mean() / total_mean * 100 if total_mean != 0 else 0

# THE KEY COMPARISON
print(f"\n{'='*70}")
print(f"  KEY COMPARISON: Exp 2B (mixed) vs Exp 06 (factoid only)")
print(f"  {'Component':<15} {'Exp 2B (mixed)':>18} {'Exp 06 (factoid)':>20} {'Change':>10}")
print(f"  {'-'*65}")
print(f"  {'Structure':<15} {'84.7%':>18} {struct_pct:>19.1f}% {struct_pct-84.7:>+9.1f}pp")
print(f"  {'Vocabulary':<15} {'5.5% (ns)':>18} {vocab_pct:>19.1f}% {vocab_pct-5.5:>+9.1f}pp")
print(f"  {'Semantics':<15} {'9.7% (***)':>18} {sem_pct:>19.1f}% {sem_pct-9.7:>+9.1f}pp")
print(f"{'='*70}")

if struct_pct < 60:
    print(f"\n  --> PREDICTION CONFIRMED: structural% dropped to {struct_pct:.0f}%.")
    print(f"      For short factoid answers, query content matters.")
    print(f"      The 'directed' in directed KV cache IS valuable for this task type.")
elif struct_pct < 75:
    print(f"\n  --> PARTIAL: structural% dropped to {struct_pct:.0f}% (from 85%).")
    print(f"      Meaningful shift but structure still dominates.")
else:
    print(f"\n  --> PREDICTION FAILED: structural% is still {struct_pct:.0f}%.")
    print(f"      Even on factoid QA, the mechanism is primarily structural.")

# Per-sample structural% distribution
per_sample_struct_pct = []
for i in range(N_SAMPLES):
    total_i = total_comp[i]
    if total_i > 0.01:  # avoid division by near-zero
        per_sample_struct_pct.append(struct_comp[i] / total_i * 100)
per_sample_struct_pct = np.array(per_sample_struct_pct)
print(f"\n  Per-sample structural% (N={len(per_sample_struct_pct)} with total>0.01):")
print(f"    Mean: {per_sample_struct_pct.mean():.1f}%, Median: {np.median(per_sample_struct_pct):.1f}%")
print(f"    % samples with structural < 50%: {np.mean(per_sample_struct_pct < 50)*100:.1f}%")
print(f"    % samples with structural < 0% (random hurts): {np.mean(per_sample_struct_pct < 0)*100:.1f}%")
""")


# ===== Cell 9: Part 3 — Surrogate Comparison =====
code(r"""# Cell 9: Part 3 — Surrogate Comparison

print("=" * 70)
print("PART 3: SURROGATE COMPARISON")
print("=" * 70)
print("Does 'What is [keyword]?' beat random for factoid QA?\n")

surr_x1_nlls = np.array([r['nll_surr_template_x1_trunc'] for r in results])
surr_x4_nlls = np.array([r['nll_surr_template_x4_trunc'] for r in results])
random_x4_nlls = np.array([r['nll_random_x4_trunc'] for r in results])
oracle_x4_nlls = np.array([r['nll_oracle_x4_trunc'] for r in results])

# Head-to-head comparisons
pairs = [
    ('oracle_x1', oracle_x1_nlls, 'random_x1', random_x1_nlls,
     "Does oracle beat random? (semantic signal exists?)"),
    ('surr_template_x1', surr_x1_nlls, 'random_x1', random_x1_nlls,
     "Does template beat random? (heuristic captures semantics?)"),
    ('oracle_x1', oracle_x1_nlls, 'surr_template_x1', surr_x1_nlls,
     "Does oracle beat template? (room for improvement?)"),
    ('surr_template_x4', surr_x4_nlls, 'random_x4', random_x4_nlls,
     "Does template x4 beat random x4? (amplified by repetition?)"),
    ('oracle_x4', oracle_x4_nlls, 'surr_template_x4', surr_x4_nlls,
     "Does oracle x4 beat template x4?"),
]

for name_a, nlls_a, name_b, nlls_b, question in pairs:
    diff = nlls_b - nlls_a  # positive = A is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = name_a if d > 0 else name_b
    print(f"  {question}")
    print(f"    {name_a} vs {name_b}: d={d:+.3f}, win={win:.1f}%, p={p:.2e} {sig} [{winner}]")
    print()

# Compare with Exp 02 (mixed sample)
print(f"--- Comparison with Exp 02 ---")
print(f"  Exp 02 oracle vs random: d=+0.080 (ns on mixed sample)")
surr_rand_d = cohens_d(random_x1_nlls - surr_x1_nlls)
oracle_rand_d = cohens_d(random_x1_nlls - oracle_x1_nlls)
print(f"  Exp 06 oracle vs random: d={oracle_rand_d:+.3f} (factoid)")
print(f"  Exp 06 template vs random: d={surr_rand_d:+.3f} (factoid)")
""")


# ===== Cell 10: Part 4 — Hardness Interaction =====
code(r"""# Cell 10: Part 4 — Hardness Interaction

print("=" * 70)
print("PART 4: HARDNESS INTERACTION (within factoid)")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

print(f"\n  {'Quintile':<12} {'N':>4} {'Bare NLL':>10} {'Struct%':>9} {'Vocab%':>8} "
      f"{'Sem%':>7} {'Oracle d':>10} {'Orc vs Rand':>12}")
print(f"  {'-'*80}")

for q in range(5):
    mask = quintiles == q
    n = mask.sum()
    bare_q = bare_nlls[mask].mean()

    s_comp = (bare_nlls[mask] - random_x1_nlls[mask]).mean()
    v_comp = (random_x1_nlls[mask] - scrambled_nlls[mask]).mean()
    sm_comp = (scrambled_nlls[mask] - oracle_x1_nlls[mask]).mean()
    t_comp = (bare_nlls[mask] - oracle_x1_nlls[mask]).mean()

    if t_comp > 0:
        s_pct = s_comp / t_comp * 100
        v_pct = v_comp / t_comp * 100
        sm_pct = sm_comp / t_comp * 100
    else:
        s_pct = v_pct = sm_pct = 0

    o_d = cohens_d(bare_nlls[mask] - oracle_x1_nlls[mask])

    # Oracle vs random head-to-head
    diff_q = random_x1_nlls[mask] - oracle_x1_nlls[mask]
    or_d = cohens_d(diff_q)
    _, or_p = stats.ttest_1samp(diff_q, 0)
    or_sig = '***' if or_p < 0.001 else '**' if or_p < 0.01 else '*' if or_p < 0.05 else 'ns'

    print(f"  {q_labels[q]:<12} {n:>4} {bare_q:>10.3f} {s_pct:>8.1f}% {v_pct:>7.1f}% "
          f"{sm_pct:>6.1f}% {o_d:>+10.3f} {or_d:>+8.3f} {or_sig}")

# Correlations
print(f"\n--- Correlations ---")
r_s, p_s = stats.pearsonr(bare_nlls, struct_comp)
r_v, p_v = stats.pearsonr(bare_nlls, vocab_comp)
r_sm, p_sm = stats.pearsonr(bare_nlls, sem_comp)
print(f"  hardness vs structure:  r={r_s:+.3f} (p={p_s:.2e})")
print(f"  hardness vs vocabulary: r={r_v:+.3f} (p={p_v:.2e})")
print(f"  hardness vs semantics:  r={r_sm:+.3f} (p={p_sm:.2e})")

# By answer length (1w vs 2-3w vs 4-5w)
print(f"\n--- By answer length ---")
for aw_label, aw_min, aw_max in [('1 word', 1, 1), ('2-3 words', 2, 3), ('4-5 words', 4, 5)]:
    aw = np.array([r['answer_words'] for r in results])
    mask = (aw >= aw_min) & (aw <= aw_max)
    n = mask.sum()
    if n < 10:
        continue
    t = (bare_nlls[mask] - oracle_x1_nlls[mask]).mean()
    s = (bare_nlls[mask] - random_x1_nlls[mask]).mean()
    if t > 0:
        s_pct = s / t * 100
        sm_pct = ((scrambled_nlls[mask] - oracle_x1_nlls[mask]).mean()) / t * 100
    else:
        s_pct = sm_pct = 0
    o_d = cohens_d(bare_nlls[mask] - oracle_x1_nlls[mask])
    print(f"  {aw_label} (N={n}): struct={s_pct:.1f}%, sem={sm_pct:.1f}%, oracle d={o_d:+.3f}")
""")


# ===== Cell 11: Part 5 — Synthesis + Save =====
code(r"""# Cell 11: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: FACTOID SUBSAMPLE RESULTS")
print("=" * 70)

# Summary numbers
print(f"\n1. DATASET: MS MARCO v1.1, filtered to answer <= {MAX_ANSWER_WORDS} words")
print(f"   N: {N_SAMPLES}, mean answer: {answer_words.mean():.1f} words")
print(f"   Oracle headroom: d={oracle_d:+.3f}, delta={oracle_benefit.mean():+.3f}")

print(f"\n2. 3-WAY DECOMPOSITION:")
print(f"   {'Component':<15} {'Exp 2B (mixed)':>18} {'Exp 06 (factoid)':>20}")
print(f"   {'-'*55}")
print(f"   {'Structure':<15} {'84.7%':>18} {struct_pct:>19.1f}%")
print(f"   {'Vocabulary':<15} {'5.5%':>18} {vocab_pct:>19.1f}%")
print(f"   {'Semantics':<15} {'9.7%':>18} {sem_pct:>19.1f}%")

print(f"\n3. SURROGATE COMPARISON:")
oracle_rand_d_val = cohens_d(random_x1_nlls - oracle_x1_nlls)
surr_rand_d_val = cohens_d(random_x1_nlls - surr_x1_nlls)
_, p_or = stats.ttest_1samp(random_x1_nlls - oracle_x1_nlls, 0)
_, p_sr = stats.ttest_1samp(random_x1_nlls - surr_x1_nlls, 0)
or_sig = '***' if p_or < 0.001 else '**' if p_or < 0.01 else '*' if p_or < 0.05 else 'ns'
sr_sig = '***' if p_sr < 0.001 else '**' if p_sr < 0.01 else '*' if p_sr < 0.05 else 'ns'
print(f"   Oracle vs random:   d={oracle_rand_d_val:+.3f} ({or_sig})")
print(f"   Template vs random: d={surr_rand_d_val:+.3f} ({sr_sig})")

print(f"\n{'='*70}")
print("CONCLUSIONS:")

if struct_pct < 60:
    mechanism = "SEMANTIC_DOMINANT"
    print(f"  1. PREDICTION CONFIRMED: factoid QA is {struct_pct:.0f}% structural / "
          f"{100-struct_pct:.0f}% content.")
    print(f"     The '85% structural' finding was an artifact of averaging over two populations.")
    print(f"     For short factoid answers, the directed KV cache IS worth directing.")
elif struct_pct < 75:
    mechanism = "MIXED"
    print(f"  1. PARTIAL CONFIRMATION: structural% dropped to {struct_pct:.0f}% (from 85%).")
    print(f"     Content matters more for factoid QA but structure still dominates.")
else:
    mechanism = "STILL_STRUCTURAL"
    print(f"  1. PREDICTION FAILED: structural% is still {struct_pct:.0f}% even for factoid QA.")

if surr_rand_d_val > 0.05 and p_sr < 0.05:
    surrogate_value = "POSITIVE"
    print(f"  2. Template surrogate significantly beats random (d={surr_rand_d_val:+.3f}).")
    print(f"     Heuristic surrogates have positive ROI for factoid QA.")
elif oracle_rand_d_val > 0.05 and p_or < 0.05:
    surrogate_value = "ORACLE_ONLY"
    print(f"  2. Oracle beats random (d={oracle_rand_d_val:+.3f}) but template does not.")
    print(f"     The semantic signal exists but the heuristic doesn't capture it.")
else:
    surrogate_value = "NONE"
    print(f"  2. Even oracle barely beats random on factoid QA.")

print(f"\n  Practical implication: ", end="")
if mechanism == "SEMANTIC_DOMINANT" and surrogate_value == "POSITIVE":
    print("for factoid extraction tasks, invest in content-aware surrogates.")
elif mechanism == "SEMANTIC_DOMINANT":
    print("semantic content matters but better surrogates are needed to capture it.")
else:
    print("the structural mechanism dominates even for factoid QA.")
print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp06_factoid_subsample',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1_short_answers',
    'n_samples': N_SAMPLES,
    'max_answer_words': MAX_ANSWER_WORDS,
    'mean_answer_words': float(answer_words.mean()),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'baseline': {
        'bare_nll': float(bare_nlls.mean()),
        'oracle_d': float(oracle_d),
        'oracle_headroom': float(oracle_benefit.mean()),
    },
    'decomposition': {
        'structure_pct': float(struct_pct),
        'vocabulary_pct': float(vocab_pct),
        'semantics_pct': float(sem_pct),
        'structure_d': float(cohens_d(struct_comp)),
        'vocabulary_d': float(cohens_d(vocab_comp)),
        'semantics_d': float(cohens_d(sem_comp)),
    },
    'surrogate_comparison': {
        'oracle_vs_random_d': float(oracle_rand_d_val),
        'oracle_vs_random_p': float(p_or),
        'template_vs_random_d': float(surr_rand_d_val),
        'template_vs_random_p': float(p_sr),
    },
    'conditions': {},
    'conclusion': {
        'mechanism': mechanism,
        'surrogate_value': surrogate_value,
    },
}

for cond, desc in all_cond_pairs:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    _, p = stats.ttest_1samp(benefit, 0)
    final_results['conditions'][cond] = {
        'description': desc,
        'd': float(d),
        'mean_nll': float(nlls.mean()),
        'mean_delta': float(benefit.mean()),
        'pct_oracle': float(d / oracle_d * 100) if oracle_d > 0 else 0,
        'p': float(p),
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
out_path = "06_factoid_subsample.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
