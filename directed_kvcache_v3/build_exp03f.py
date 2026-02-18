#!/usr/bin/env python3
# Build Exp 3F notebook: Semantic Amplification.
#
# Tests whether repeating the query prefix amplifies the semantic
# component beyond the ~10% baseline from Exp 2B.
# 20 conditions x 500 samples = 10,000 forward passes.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 3F: Semantic Amplification

## Motivation

Exps 2B, 3D, and 3E established that ~85% of the prefix benefit is structural
(attention sink redistribution). The semantic component is ~10% on MS MARCO
but -4% on neural-bridge. This experiment asks: **can we amplify the semantic
component by repeating the prefix?**

From Exp 3E: prefix tokens absorb 13-26% of doc attention. All conditions
produce similar KL redistribution. The structural effect saturates quickly
(1 random word = 85% of oracle per Exp 2B).

**Key insight**: if the structural effect saturates but we keep adding more
semantic tokens, the semantic component's share should grow.

## Design

**Dataset**: MS MARCO v1.1 (same 500 samples as Exp 02/2B).

### 20 Conditions

| # | Group | Condition | Prefix |
|---|-------|-----------|--------|
| 1 | — | `bare` | (none) |
| 2-6 | A | `oracle_x{1,3,5,10,20}_trunc` | query x N |
| 7-11 | A | `random_x{1,3,5,10,20}_trunc` | random_matched x N |
| 12-13 | B | `scrambled_x{5,10}_trunc` | scrambled oracle x N |
| 14-15 | C | `content_x{5,10}_trunc` | content words only x N |
| 16-17 | C | `random_content_x{5,10}_trunc` | random matched to content length x N |
| 18 | D | `the_matched10_trunc` | "the" x M (token-matched to oracle_x10) |
| 19 | D | `bare_short` | (none), doc truncated to 30 words |
| 20 | D | `oracle_short_trunc` | query x 1, doc truncated to 30 words |

`random_xN` repeats the SAME random words N times (not N different sets).

### Analysis

1. **Repetition sweep**: structural(N), semantic(N), semantic_fraction(N) vs N
2. **3-way decomposition** at N=5 and N=10 (structure/vocabulary/semantics)
3. **Content concentration**: content-only words vs full query at N=5,10
4. **Short documents**: prefix/doc ratio effect
5. **Structural saturation**: "the" control vs random_x10
6. **Hardness interaction**: semantic(N) by hardness quintile""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, re, gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, ".")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"
REPS = [1, 3, 5, 10, 20]

RESULTS_DIR = Path("results/exp03f")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
EXP02_CHECKPOINT = Path("results/exp02/checkpoint.json")

np.random.seed(SEED)
torch.manual_seed(SEED)

print("Exp 3F: Semantic Amplification")
print(f"N: {N_SAMPLES}, Repetitions: {REPS}")
""")


# ===== Cell 3: Load MS MARCO + reconstruct samples =====
code(r"""# Cell 3: Load MS MARCO and reconstruct same 500 samples as Exp 02
from lib.data import count_words
from datasets import load_dataset

# Load Exp 02 checkpoint for sample alignment verification
print("Loading Exp 02 checkpoint...")
exp02_ckpt = json.loads(EXP02_CHECKPOINT.read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(exp02_results)}"

# Extract Exp 02 NLLs for verification
exp02_bare = np.array([r['nll_bare'] for r in exp02_results])
exp02_oracle = np.array([r['nll_oracle_trunc'] for r in exp02_results])

# Reconstruct dataset samples
print("Loading MS MARCO to reconstruct samples...")
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
                'word_count': wc,
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

# Verify samples match Exp 02
for i in range(min(20, N_SAMPLES)):
    assert samples[i]['query'] == exp02_results[i]['query'], \
        f"Sample {i} mismatch: {samples[i]['query'][:40]} != {exp02_results[i]['query'][:40]}"
passage_words = np.array([s['word_count'] for s in samples])
print(f"Verified: {N_SAMPLES} samples match Exp 02")
print(f"Document lengths: {passage_words.min()}-{passage_words.max()} words, "
      f"mean={passage_words.mean():.0f}")
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


# ===== Cell 5: Generate all 20 conditions per sample =====
code(r"""# Cell 5: Generate all 20 conditions per sample

COND_NAMES = ['bare']
for N in REPS:
    COND_NAMES.append(f'oracle_x{N}_trunc')
    COND_NAMES.append(f'random_x{N}_trunc')
COND_NAMES.extend([
    'scrambled_x5_trunc', 'scrambled_x10_trunc',
    'content_x5_trunc', 'content_x10_trunc',
    'random_content_x5_trunc', 'random_content_x10_trunc',
    'the_matched10_trunc',
    'bare_short', 'oracle_short_trunc',
])

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

for i, s in enumerate(samples):
    query = s['query']
    passage = s['passage']
    query_words = query.split()

    # Random words from unrelated passage (same set for all reps)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    random_matched = " ".join(other_words[:len(query_words)])

    # Scrambled oracle (same words, random order)
    rng = np.random.RandomState(SEED + i)
    shuffled = list(query_words)
    rng.shuffle(shuffled)
    scrambled = " ".join(shuffled)

    # Content words (stop words removed from query)
    content_words = [w for w in query_words if w.lower() not in STOP_WORDS]
    if not content_words:
        content_words = query_words[:1]  # fallback
    content_text = " ".join(content_words)
    s['n_content_words'] = len(content_words)

    # Random words matched to content-word count
    random_content = " ".join(other_words[:len(content_words)])

    # Group A: oracle and random at each repetition level
    for N in REPS:
        s[f'oracle_x{N}'] = " ".join([query] * N)
        s[f'random_x{N}'] = " ".join([random_matched] * N)

    # Group B: scrambled at x5 and x10
    s['scrambled_x5'] = " ".join([scrambled] * 5)
    s['scrambled_x10'] = " ".join([scrambled] * 10)

    # Group C: content concentration
    s['content_x5'] = " ".join([content_text] * 5)
    s['content_x10'] = " ".join([content_text] * 10)
    s['random_content_x5'] = " ".join([random_content] * 5)
    s['random_content_x10'] = " ".join([random_content] * 10)

    # Group D: "the" matched to oracle_x10 token count
    oracle_x10_toks = count_prefix_tokens(s['oracle_x10'], passage)
    s['the_matched10'] = " ".join(["the"] * oracle_x10_toks)
    s['oracle_x10_toks'] = oracle_x10_toks

    # Group D: short document (first 30 words)
    s['short_doc'] = " ".join(passage.split()[:30])

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query:   {ex['query'][:80]}")
print(f"  Answer:  {ex['answer'][:80]}")
print(f"  Passage: {ex['passage'][:80]}...")
cw = [w for w in ex['query'].split() if w.lower() not in STOP_WORDS]
print(f"  Content words: {' '.join(cw)}")
print(f"  Short doc: {ex['short_doc'][:80]}...")

# Token count summary for first 50 samples
print(f"\nPrefix token counts (first 50 samples):")
for c in COND_NAMES:
    if c in ('bare', 'bare_short'):
        continue
    if c == 'oracle_short_trunc':
        toks = [count_prefix_tokens(s['query'], s['short_doc']) for s in samples[:50]]
    else:
        key = c.replace('_trunc', '')
        toks = [count_prefix_tokens(s[key], s['passage']) for s in samples[:50]]
    print(f"  {c:<30} mean={np.mean(toks):>6.1f}, range=[{min(toks):>3}, {max(toks):>3}]")

# Content word statistics
cw_counts = [s['n_content_words'] for s in samples]
print(f"\nContent words per query: mean={np.mean(cw_counts):.1f}, "
      f"range=[{min(cw_counts)}, {max(cw_counts)}]")
""")


# ===== Cell 6: Scoring loop with checkpointing =====
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
        'n_content_words': s['n_content_words'],
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['passage'], s['answer'])
            result['nll_bare'] = nll
        elif cond == 'bare_short':
            nll = score_nll(s['short_doc'], s['answer'])
            result['nll_bare_short'] = nll
        elif cond == 'oracle_short_trunc':
            prefix = s['query']
            doc = s['short_doc']
            enc_text = prefix + "\n" + doc
            ptoks = count_prefix_tokens(prefix, doc)
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result['nll_oracle_short_trunc'] = nll
            result['ptoks_oracle_short_trunc'] = ptoks
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


# ===== Cell 7: Part 1 - Repetition sweep curves =====
code(r"""# Cell 7: Part 1 -- Repetition Sweep Curves

print("=" * 70)
print("PART 1: REPETITION SWEEP")
print("=" * 70)

# Extract NLLs
bare_nlls = np.array([r['nll_bare'] for r in results])

# Verify against Exp 02
print("--- Verification against Exp 02 ---")
bare_diff = np.abs(bare_nlls - exp02_bare)
print(f"  bare NLL max diff: {bare_diff.max():.6f} (should be ~0)")

oracle_x1_nlls = np.array([r['nll_oracle_x1_trunc'] for r in results])
oracle_diff = np.abs(oracle_x1_nlls - exp02_oracle)
print(f"  oracle_x1 vs Exp02 oracle max diff: {oracle_diff.max():.6f} (should be ~0)")

# Sweep table
print(f"\n{'N':>3} {'Struct delta':>13} {'Sem delta':>11} {'Total delta':>12} "
      f"{'Sem frac':>9} {'Struct d':>9} {'Sem d':>7} {'Total d':>8}")
print("-" * 80)

sweep = {}
for N in REPS:
    o_nlls = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    r_nlls = np.array([r[f'nll_random_x{N}_trunc'] for r in results])

    structural = bare_nlls - r_nlls
    semantic = r_nlls - o_nlls
    total = bare_nlls - o_nlls

    struct_mean = structural.mean()
    sem_mean = semantic.mean()
    total_mean = total.mean()
    sem_frac = sem_mean / total_mean * 100 if total_mean > 0 else 0

    _, p_sem = stats.ttest_1samp(semantic, 0)
    sig = '***' if p_sem < 0.001 else '**' if p_sem < 0.01 else '*' if p_sem < 0.05 else 'ns'

    sweep[N] = {
        'structural': structural, 'semantic': semantic, 'total': total,
        'struct_mean': struct_mean, 'sem_mean': sem_mean, 'total_mean': total_mean,
        'sem_frac': sem_frac, 'p_sem': p_sem,
    }

    print(f"{N:>3} {struct_mean:>+13.4f} {sem_mean:>+11.4f} {total_mean:>+12.4f} "
          f"{sem_frac:>8.1f}% {cohens_d(structural):>+9.3f} "
          f"{cohens_d(semantic):>+7.3f} {cohens_d(total):>+8.3f} {sig}")

# Key test: does semantic fraction grow?
fracs = [sweep[N]['sem_frac'] for N in REPS]
print(f"\nSemantic fraction: {' -> '.join(f'{f:.1f}%' for f in fracs)}")

if fracs[-1] > fracs[0] * 1.5:
    print("  --> AMPLIFICATION: semantic fraction grew with repetition!")
elif fracs[-1] > fracs[0] * 1.1:
    print("  --> MODERATE growth in semantic fraction.")
else:
    print("  --> NO amplification: semantic fraction is stable.")

# Structural saturation check
struct_ds = [cohens_d(sweep[N]['structural']) for N in REPS]
print(f"\nStructural d: {' -> '.join(f'{d:+.3f}' for d in struct_ds)}")
struct_ratio_sweep = struct_ds[-1] / struct_ds[0] if struct_ds[0] > 0 else 0
print(f"  x20/x1 ratio: {struct_ratio_sweep:.2f}")

if struct_ratio_sweep < 1.2:
    print("  --> Structural effect is SATURATED (x1 ~ x20).")
else:
    print(f"  --> Structural effect still growing ({struct_ratio_sweep:.1f}x from x1 to x20).")

# Bonferroni significance check for all non-bare conditions vs bare
print(f"\n--- Significance vs bare (Bonferroni k=19, alpha={0.05/19:.4f}) ---")
alpha_bonf = 0.05 / 19
all_sig = True
for cond in COND_NAMES:
    if cond == 'bare':
        continue
    nlls_c = np.array([r[f'nll_{cond}'] for r in results])
    if cond == 'bare_short':
        continue  # different document, not comparable to bare
    benefit_c = bare_nlls - nlls_c
    _, p_c = stats.ttest_1samp(benefit_c, 0)
    if p_c >= alpha_bonf:
        all_sig = False
        print(f"  {cond:<30} p={p_c:.4e} NOT SIGNIFICANT")
print(f"  All comparable conditions significant after Bonferroni: "
      f"{'YES' if all_sig else 'NO'}")
""")


# ===== Cell 8: Part 2 - 3-Way Decomposition =====
code(r"""# Cell 8: Part 2 -- 3-Way Decomposition at N=5 and N=10

print("=" * 70)
print("PART 2: 3-WAY DECOMPOSITION (structure / vocabulary / semantics)")
print("=" * 70)
print("Decompose: bare -> random_xN -> scrambled_xN -> oracle_xN\n")

for N in [5, 10]:
    o_nlls = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    r_nlls = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    scr_nlls = np.array([r[f'nll_scrambled_x{N}_trunc'] for r in results])

    structure = bare_nlls - r_nlls
    vocabulary = r_nlls - scr_nlls
    semantics = scr_nlls - o_nlls
    total = bare_nlls - o_nlls

    total_mean = total.mean()
    s_pct = structure.mean() / total_mean * 100 if total_mean > 0 else 0
    v_pct = vocabulary.mean() / total_mean * 100 if total_mean > 0 else 0
    sm_pct = semantics.mean() / total_mean * 100 if total_mean > 0 else 0

    print(f"--- N = {N} ---")
    print(f"  {'Component':<15} {'Delta':>10} {'%total':>8} {'d':>8} {'p':>12} {'sig':>4}")
    print(f"  {'-'*60}")

    for label, comp in [('Structure', structure), ('Vocabulary', vocabulary),
                        ('Semantics', semantics)]:
        mu = comp.mean()
        pct = mu / total_mean * 100 if total_mean > 0 else 0
        d = cohens_d(comp)
        _, p = stats.ttest_1samp(comp, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<15} {mu:>+10.4f} {pct:>7.1f}% {d:>+8.3f} {p:>12.2e} {sig}")

    print(f"  {'TOTAL':<15} {total_mean:>+10.4f} {'100.0%':>8}")
    residual = total_mean - (structure.mean() + vocabulary.mean() + semantics.mean())
    print(f"  Residual: {residual:.6f}")
    print()

# Compare to Exp 2B baseline (N=1)
print("--- Comparison with Exp 2B (N=1) ---")
print(f"  Exp 2B (N=1): Structure=84.7%, Vocabulary=5.5% (ns), Semantics=9.7% (***)")
for N in [5, 10]:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    sc = np.array([r[f'nll_scrambled_x{N}_trunc'] for r in results])
    t = (bare_nlls - o).mean()
    s_p = (bare_nlls - rn).mean() / t * 100 if t > 0 else 0
    v_p = (rn - sc).mean() / t * 100 if t > 0 else 0
    sm_p = (sc - o).mean() / t * 100 if t > 0 else 0
    print(f"  N={N}:     Structure={s_p:.1f}%, Vocabulary={v_p:.1f}%, Semantics={sm_p:.1f}%")
""")


# ===== Cell 9: Part 3 - Content Concentration =====
code(r"""# Cell 9: Part 3 -- Content Concentration

print("=" * 70)
print("PART 3: CONTENT CONCENTRATION")
print("=" * 70)
print("Does stripping stop words improve the semantic-to-structural ratio?\n")

for N in [5, 10]:
    # Full query
    o_nlls = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    r_nlls = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    oracle_sem = r_nlls - o_nlls
    oracle_struct = bare_nlls - r_nlls

    # Content-only query
    c_nlls = np.array([r[f'nll_content_x{N}_trunc'] for r in results])
    rc_nlls = np.array([r[f'nll_random_content_x{N}_trunc'] for r in results])
    content_sem = rc_nlls - c_nlls
    content_struct = bare_nlls - rc_nlls

    # Semantic-to-structural ratios
    oracle_ratio = oracle_sem.mean() / oracle_struct.mean() if oracle_struct.mean() > 0 else 0
    content_ratio = content_sem.mean() / content_struct.mean() if content_struct.mean() > 0 else 0

    # Total benefits
    oracle_total = bare_nlls - o_nlls
    content_total = bare_nlls - c_nlls

    print(f"--- N = {N} ---")
    print(f"  {'Metric':<30} {'Full query':>12} {'Content only':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Structural delta':<30} {oracle_struct.mean():>+12.4f} {content_struct.mean():>+14.4f}")
    print(f"  {'Semantic delta':<30} {oracle_sem.mean():>+12.4f} {content_sem.mean():>+14.4f}")
    print(f"  {'Total delta':<30} {oracle_total.mean():>+12.4f} {content_total.mean():>+14.4f}")
    print(f"  {'Semantic / Structural ratio':<30} {oracle_ratio:>12.3f} {content_ratio:>14.3f}")
    print(f"  {'Total d':<30} {cohens_d(oracle_total):>+12.3f} {cohens_d(content_total):>+14.3f}")

    # Significance of content vs oracle semantic gap
    sem_diff = content_sem - oracle_sem
    _, p_diff = stats.ttest_1samp(sem_diff, 0)
    sig = '***' if p_diff < 0.001 else '**' if p_diff < 0.01 else '*' if p_diff < 0.05 else 'ns'
    print(f"  Content-Oracle semantic gap: {sem_diff.mean():+.4f} (p={p_diff:.2e}) {sig}")

    # Token count comparison
    o_toks = np.array([r[f'ptoks_oracle_x{N}_trunc'] for r in results])
    c_toks = np.array([r[f'ptoks_content_x{N}_trunc'] for r in results])
    print(f"  Mean prefix tokens: oracle={o_toks.mean():.0f}, content={c_toks.mean():.0f} "
          f"({c_toks.mean()/o_toks.mean()*100:.0f}%)")
    print()

# Overall interpretation
c5_rc5 = np.array([r['nll_random_content_x5_trunc'] for r in results])
c5 = np.array([r['nll_content_x5_trunc'] for r in results])
r5 = np.array([r['nll_random_x5_trunc'] for r in results])
o5 = np.array([r['nll_oracle_x5_trunc'] for r in results])
c_rat = (c5_rc5 - c5).mean() / (bare_nlls - c5_rc5).mean() if (bare_nlls - c5_rc5).mean() > 0 else 0
o_rat = (r5 - o5).mean() / (bare_nlls - r5).mean() if (bare_nlls - r5).mean() > 0 else 0

if c_rat > o_rat * 1.2:
    print("  --> Concentrating meaning IMPROVES the semantic ratio.")
elif c_rat < o_rat * 0.8:
    print("  --> Concentrating meaning HURTS -- stop words may provide useful structure.")
else:
    print("  --> Concentrating meaning has NO significant effect on semantic ratio.")
""")


# ===== Cell 10: Part 4 - Short Documents =====
code(r"""# Cell 10: Part 4 -- Short Document Analysis

print("=" * 70)
print("PART 4: SHORT DOCUMENTS (prefix/doc ratio effect)")
print("=" * 70)
print("Does the prefix matter more when documents are shorter?\n")

bare_short_nlls = np.array([r['nll_bare_short'] for r in results])
oracle_short_nlls = np.array([r['nll_oracle_short_trunc'] for r in results])

# Benefits
short_benefit = bare_short_nlls - oracle_short_nlls
full_benefit = bare_nlls - oracle_x1_nlls

print(f"  {'Metric':<35} {'Full doc':>12} {'Short doc (30w)':>16}")
print(f"  {'-'*65}")
print(f"  {'Bare NLL':<35} {bare_nlls.mean():>12.4f} {bare_short_nlls.mean():>16.4f}")
print(f"  {'Oracle NLL':<35} {oracle_x1_nlls.mean():>12.4f} {oracle_short_nlls.mean():>16.4f}")
print(f"  {'Total benefit (delta)':<35} {full_benefit.mean():>+12.4f} {short_benefit.mean():>+16.4f}")
print(f"  {'Cohen d':<35} {cohens_d(full_benefit):>+12.3f} {cohens_d(short_benefit):>+16.3f}")
print(f"  {'Win rate':<35} {np.mean(full_benefit>0)*100:>11.1f}% {np.mean(short_benefit>0)*100:>15.1f}%")

# Significance
_, p_full = stats.ttest_1samp(full_benefit, 0)
_, p_short = stats.ttest_1samp(short_benefit, 0)
print(f"  {'p-value':<35} {p_full:>12.2e} {p_short:>16.2e}")

# Prefix/doc token ratio
short_ptoks = np.array([r.get('ptoks_oracle_short_trunc', 0) for r in results])
full_ptoks = np.array([r.get('ptoks_oracle_x1_trunc', 0) for r in results])
short_doc_toks = np.array([len(tokenizer(s['short_doc'], add_special_tokens=True).input_ids)
                           for s in samples])
full_doc_toks = np.array([len(tokenizer(s['passage'], add_special_tokens=True).input_ids)
                          for s in samples])

print(f"\n--- Prefix/doc token ratio ---")
print(f"  Full doc:  prefix={full_ptoks.mean():.0f} toks, doc={full_doc_toks.mean():.0f} toks, "
      f"ratio={full_ptoks.mean()/full_doc_toks.mean():.2f}")
print(f"  Short doc: prefix={short_ptoks.mean():.0f} toks, doc={short_doc_toks.mean():.0f} toks, "
      f"ratio={short_ptoks.mean()/short_doc_toks.mean():.2f}")

d_full = cohens_d(full_benefit)
d_short = cohens_d(short_benefit)
if d_short > d_full * 1.1:
    print(f"\n  --> Prefix benefit is LARGER for shorter documents (d={d_short:+.3f} vs {d_full:+.3f}).")
    print(f"      The semantic signal may be diluted by document length.")
elif d_short < d_full * 0.9:
    print(f"\n  --> Prefix benefit is SMALLER for shorter documents (d={d_short:+.3f} vs {d_full:+.3f}).")
    print(f"      Longer documents benefit MORE from prefix co-encoding.")
else:
    print(f"\n  --> Prefix benefit is similar regardless of document length "
          f"(d={d_short:+.3f} vs {d_full:+.3f}).")
""")


# ===== Cell 11: Part 5 - Structural Saturation + "the" control =====
code(r"""# Cell 11: Part 5 -- Structural Saturation + "the" Control

print("=" * 70)
print("PART 5: STRUCTURAL SATURATION")
print("=" * 70)
print("Is the structural benefit from attention MASS or token DIVERSITY?\n")

the_nlls = np.array([r['nll_the_matched10_trunc'] for r in results])
random_x10_nlls = np.array([r['nll_random_x10_trunc'] for r in results])
oracle_x10_nlls = np.array([r['nll_oracle_x10_trunc'] for r in results])

the_benefit = bare_nlls - the_nlls
random_x10_benefit = bare_nlls - random_x10_nlls
oracle_x10_benefit = bare_nlls - oracle_x10_nlls

oracle_x10_d = cohens_d(oracle_x10_benefit)

print(f"  {'Condition':<30} {'NLL':>8} {'Delta':>10} {'d':>8} {'%Oracle':>9}")
print(f"  {'-'*70}")
for name, benefit, nlls in [
    ('the_matched10 (uniform)', the_benefit, the_nlls),
    ('random_x10 (diverse)', random_x10_benefit, random_x10_nlls),
    ('oracle_x10 (semantic)', oracle_x10_benefit, oracle_x10_nlls),
]:
    d = cohens_d(benefit)
    pct = d / oracle_x10_d * 100 if oracle_x10_d > 0 else 0
    print(f"  {name:<30} {nlls.mean():>8.4f} {benefit.mean():>+10.4f} "
          f"{d:>+8.3f} {pct:>8.1f}%")

# Head-to-head: "the" vs random_x10
diff_the_rand = random_x10_nlls - the_nlls  # positive = "the" is better
d_the_rand = cohens_d(diff_the_rand)
_, p_the_rand = stats.ttest_1samp(diff_the_rand, 0)
sig = '***' if p_the_rand < 0.001 else '**' if p_the_rand < 0.01 else '*' if p_the_rand < 0.05 else 'ns'
winner = '"the"' if d_the_rand > 0 else 'random_x10'
print(f"\n  the vs random_x10: d={d_the_rand:+.3f}, p={p_the_rand:.2e} {sig} [{winner}]")

# Token count comparison
the_toks = np.array([r['ptoks_the_matched10_trunc'] for r in results])
r10_toks = np.array([r['ptoks_random_x10_trunc'] for r in results])
o10_toks = np.array([r['ptoks_oracle_x10_trunc'] for r in results])
print(f"\n  Token counts: the={the_toks.mean():.0f}, random_x10={r10_toks.mean():.0f}, "
      f"oracle_x10={o10_toks.mean():.0f}")

# Structural saturation curve
print(f"\n--- Structural benefit saturation ---")
print(f"  {'N':>3} {'Structural d':>13} {'% of x20':>9}")
struct_ds = []
for N in REPS:
    r_nlls_n = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    d = cohens_d(bare_nlls - r_nlls_n)
    struct_ds.append(d)

max_struct_d = struct_ds[-1]
for N, d in zip(REPS, struct_ds):
    pct = d / max_struct_d * 100 if max_struct_d > 0 else 0
    print(f"  {N:>3} {d:>+13.3f} {pct:>8.1f}%")

if abs(d_the_rand) < 0.05:
    print(f"\n  --> Token diversity does NOT matter. Structure comes from attention MASS.")
elif d_the_rand < 0:
    print(f"\n  --> Diverse tokens are BETTER. Some structural benefit comes from diversity.")
else:
    print(f"\n  --> Uniform 'the' is BETTER than diverse random. Diversity may add noise.")
""")


# ===== Cell 12: Part 6 - Hardness Interaction =====
code(r"""# Cell 12: Part 6 -- Hardness x Repetition Interaction

print("=" * 70)
print("PART 6: HARDNESS x REPETITION INTERACTION")
print("=" * 70)
print("Does repetition amplify the semantic gap for harder samples?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

# Table: semantic gap by quintile and N
header = f"  {'Quintile':<10}"
for N in REPS:
    header += f"  {'x'+str(N)+' sem':>8}"
header += f"  {'x1 frac':>8} {'x20 frac':>9}"
print(header)
print(f"  {'-'*(12 + 10 * len(REPS) + 20)}")

for q in range(5):
    mask = quintiles == q
    row = f"  {q_labels[q]:<10}"
    fracs = []
    for N in REPS:
        o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])[mask]
        rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])[mask]
        sem = (rn - o).mean()
        total = (bare_nlls[mask] - o).mean()
        frac = sem / total * 100 if total > 0 else 0
        fracs.append(frac)
        row += f"  {sem:>+8.4f}"
    row += f"  {fracs[0]:>7.1f}% {fracs[-1]:>8.1f}%"
    print(row)

# Correlation: hardness vs semantic gap at each N
print(f"\n--- Hardness correlations ---")
for N in REPS:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    sem_gap = rn - o
    r_val, p_val = stats.pearsonr(bare_nlls, sem_gap)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"  x{N}: r={r_val:+.3f} (p={p_val:.2e}) {sig}")

# Semantic fraction trajectory for Q1 vs Q5
for qlabel, q_idx in [('Q1 easy', 0), ('Q5 hard', 4)]:
    mask = quintiles == q_idx
    fracs_q = []
    for N in REPS:
        o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])[mask]
        rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])[mask]
        total = (bare_nlls[mask] - o).mean()
        sem = (rn - o).mean()
        fracs_q.append(sem / total * 100 if total > 0 else 0)
    print(f"\n  {qlabel} semantic fraction: {' -> '.join(f'{f:.1f}%' for f in fracs_q)}")
""")


# ===== Cell 13: Synthesis + Save =====
code(r"""# Cell 13: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: SEMANTIC AMPLIFICATION RESULTS")
print("=" * 70)

# 1. Repetition sweep summary
print(f"\n1. REPETITION SWEEP:")
for N in REPS:
    f = sweep[N]['sem_frac']
    sd = cohens_d(sweep[N]['semantic'])
    print(f"   x{N}: semantic fraction = {f:.1f}%, semantic d = {sd:+.3f}")

frac_x1 = sweep[1]['sem_frac']
frac_x20 = sweep[20]['sem_frac']
amplification = frac_x20 / frac_x1 if frac_x1 > 0 else 0
print(f"   Amplification: x{amplification:.1f} (x1={frac_x1:.1f}% -> x20={frac_x20:.1f}%)")

# 2. 3-Way decomposition shifts
print(f"\n2. 3-WAY DECOMPOSITION SHIFT:")
print(f"   Exp 2B (N=1): Structure=84.7%, Vocabulary=5.5%, Semantics=9.7%")
for N in [5, 10]:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    sc = np.array([r[f'nll_scrambled_x{N}_trunc'] for r in results])
    t = (bare_nlls - o).mean()
    sp = (bare_nlls - rn).mean() / t * 100 if t > 0 else 0
    vp = (rn - sc).mean() / t * 100 if t > 0 else 0
    smp = (sc - o).mean() / t * 100 if t > 0 else 0
    print(f"   N={N}: Structure={sp:.1f}%, Vocabulary={vp:.1f}%, Semantics={smp:.1f}%")

# 3. Content concentration
print(f"\n3. CONTENT CONCENTRATION:")
for N in [5, 10]:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    c = np.array([r[f'nll_content_x{N}_trunc'] for r in results])
    rc = np.array([r[f'nll_random_content_x{N}_trunc'] for r in results])
    o_ratio = (rn - o).mean() / (bare_nlls - rn).mean() if (bare_nlls - rn).mean() > 0 else 0
    c_ratio = (rc - c).mean() / (bare_nlls - rc).mean() if (bare_nlls - rc).mean() > 0 else 0
    print(f"   N={N}: oracle sem/struct={o_ratio:.3f}, content sem/struct={c_ratio:.3f}")

# 4. Short documents
d_full = cohens_d(bare_nlls - oracle_x1_nlls)
d_short = cohens_d(bare_short_nlls - oracle_short_nlls)
print(f"\n4. SHORT DOCUMENTS:")
print(f"   Full doc d = {d_full:+.3f}, Short doc d = {d_short:+.3f}")

# 5. Structural saturation
the_d = cohens_d(bare_nlls - the_nlls)
rand10_d = cohens_d(bare_nlls - random_x10_nlls)
print(f"\n5. STRUCTURAL SATURATION:")
print(f"   the_matched10 d = {the_d:+.3f}, random_x10 d = {rand10_d:+.3f}")

# 6. Key conclusions
print(f"\n{'='*70}")
print("CONCLUSIONS:")

if amplification > 2.0:
    print(f"  1. STRONG amplification: semantic fraction grew {amplification:.1f}x with repetition.")
elif amplification > 1.3:
    print(f"  1. MODERATE amplification ({amplification:.1f}x). Some semantic growth with repetition.")
else:
    print(f"  1. NO amplification ({amplification:.1f}x). Semantic fraction is stable across N.")
    print(f"     Both structural and semantic components scale similarly with repetition.")

struct_sat = struct_ds[-1] / struct_ds[0] if struct_ds[0] > 0 else 0
if struct_sat < 1.2:
    print(f"  2. Structural effect is SATURATED (x20/x1 = {struct_sat:.2f}).")
else:
    print(f"  2. Structural effect still GROWING (x20/x1 = {struct_sat:.2f}).")

print(f"  3. Practical implication: ", end="")
if amplification > 1.5 and frac_x20 > 20:
    print("repeating query 5-10x could meaningfully boost semantic component.")
else:
    print("repetition does not meaningfully amplify semantic content.")
    print("     The structural mechanism dominates at all repetition levels.")

print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp03f_semantic_amplification',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'repetition_sweep': {
        str(N): {
            'structural_d': float(cohens_d(sweep[N]['structural'])),
            'semantic_d': float(cohens_d(sweep[N]['semantic'])),
            'total_d': float(cohens_d(sweep[N]['total'])),
            'structural_delta': float(sweep[N]['struct_mean']),
            'semantic_delta': float(sweep[N]['sem_mean']),
            'total_delta': float(sweep[N]['total_mean']),
            'semantic_fraction': float(sweep[N]['sem_frac']),
            'semantic_p': float(sweep[N]['p_sem']),
        }
        for N in REPS
    },
    'decomposition': {},
    'content_concentration': {},
    'structural_saturation': {
        'the_matched10_d': float(the_d),
        'random_x10_d': float(rand10_d),
    },
    'short_documents': {
        'full_doc_d': float(d_full),
        'short_doc_d': float(d_short),
    },
    'amplification_factor': float(amplification),
}

# Add decomposition results
for N in [5, 10]:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    sc = np.array([r[f'nll_scrambled_x{N}_trunc'] for r in results])
    t = (bare_nlls - o).mean()
    final_results['decomposition'][str(N)] = {
        'structure_pct': float((bare_nlls - rn).mean() / t * 100) if t > 0 else 0,
        'vocabulary_pct': float((rn - sc).mean() / t * 100) if t > 0 else 0,
        'semantics_pct': float((sc - o).mean() / t * 100) if t > 0 else 0,
    }

# Add content concentration results
for N in [5, 10]:
    o = np.array([r[f'nll_oracle_x{N}_trunc'] for r in results])
    rn = np.array([r[f'nll_random_x{N}_trunc'] for r in results])
    c = np.array([r[f'nll_content_x{N}_trunc'] for r in results])
    rc = np.array([r[f'nll_random_content_x{N}_trunc'] for r in results])
    final_results['content_concentration'][str(N)] = {
        'oracle_sem_struct_ratio': float(
            (rn - o).mean() / (bare_nlls - rn).mean()) if (bare_nlls - rn).mean() > 0 else 0,
        'content_sem_struct_ratio': float(
            (rc - c).mean() / (bare_nlls - rc).mean()) if (bare_nlls - rc).mean() > 0 else 0,
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
out_path = "03f_semantic_amplification.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
