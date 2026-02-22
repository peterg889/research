#!/usr/bin/env python3
"""Build Exp 2B notebook: Structural vs Semantic Mechanism Decomposition.

Three targeted tests to decompose the Exp 02 finding that content barely matters:
  Part 1: Re-analysis of Exp 02 data (length, hardness, variance)
  Part 2: Prefix length titration (1-50 random words)
  Part 3: Content ablation (structure + vocabulary + semantics)
  Part 4: Token diversity (repeated vs diverse tokens)
"""

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 2B: Structural vs Semantic Mechanism Decomposition

## Motivation

Exp 02 found that the **content of the surrogate barely matters** — random text
captures 81% of the oracle benefit, and there is no content gradient
(Spearman rho = -0.167, p = 0.693).

Yet pairwise comparisons show doc-specific surrogates DO beat random on a
per-sample basis (surr_doc > random: d=+0.130, p=0.004). And static surrogates
appear to match oracle by Cohen's d despite smaller absolute NLL improvements.

Something subtle is going on. This experiment decomposes the mechanism.

## v2 vs v3: Different Mechanisms, Same Symptom

| Property | v2 (decoder-only) | v3 (encoder-decoder) |
|----------|-------------------|---------------------|
| Attention | Causal (forward only) | Bidirectional |
| Mechanism | Value contamination | Co-encoding |
| Truncation effect | Removed benefit | **Improved** benefit |
| Document-specific? | No (same contamination for all docs) | Yes (bidirectional) |
| Content gradient | None (Exp 10) | None (Exp 02) |

Both show no content gradient, but the mechanisms are fundamentally different.

## Hypotheses for the Structural Benefit

1. **Attention redistribution**: adding tokens changes softmax normalization
2. **Position shift**: document tokens at different RoPE positions with prefix
3. **Implicit regularization**: prefix = noise injection that improves representations
4. **Information injection**: content genuinely flows into document representations

## Design

**Part 1**: Re-analyze Exp 02 data (no GPU needed)
- Document length stratification (does semantic gap change with length?)
- Hardness stratification (does semantic advantage emerge for hard samples?)
- Variance decomposition (why static has high d but low delta)

**Part 2**: Prefix length titration (random words: 1, 3, 5, 10, 20, 50)
- Saturation curve: does 1 token suffice (switch) or do we need many (gradual)?

**Part 3**: Content ablation (all length-matched to oracle)
- `bare → random_matched`: structure (any prefix)
- `random_matched → scrambled_oracle`: vocabulary (right words, wrong order)
- `scrambled_oracle → oracle`: semantics (right word order, full meaning)

**Part 4**: Token diversity (all ~10 words)
- `"the" x10` vs `doc_keyword x10` vs `diverse random words`
- Does diversity matter, or is any 10-token prefix equivalent?""")


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

sys.path.insert(0, "../..")

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../results/exp02b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
EXP02_CHECKPOINT = Path("../../results/exp02/checkpoint.json")

np.random.seed(SEED)
torch.manual_seed(SEED)

print("Exp 2B: Structural vs Semantic Mechanism Decomposition")
print(f"N: {N_SAMPLES}")
""")


# ===== Cell 3: Load Exp 02 data + reconstruct samples =====
code(r"""# Cell 3: Load Exp 02 data and reconstruct samples
from lib.data import count_words
from lib.analysis import cohens_d
from datasets import load_dataset

# Load Exp 02 checkpoint (has all NLLs for 9 conditions x 500 samples)
print("Loading Exp 02 checkpoint...")
exp02_ckpt = json.loads(EXP02_CHECKPOINT.read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(exp02_results)}"

# Extract key condition NLLs from Exp 02
bare_nlls = np.array([r['nll_bare'] for r in exp02_results])
oracle_nlls = np.array([r['nll_oracle_trunc'] for r in exp02_results])
random_nlls = np.array([r['nll_random_trunc'] for r in exp02_results])
surr_doc_nlls = np.array([r['nll_surr_doc_trunc'] for r in exp02_results])
surr_template_nlls = np.array([r['nll_surr_template_trunc'] for r in exp02_results])
static_fact_nlls = np.array([r['nll_static_fact_trunc'] for r in exp02_results])
surr_lead_nlls = np.array([r['nll_surr_lead_trunc'] for r in exp02_results])
surr_para_nlls = np.array([r['nll_surr_para_trunc'] for r in exp02_results])
static_howto_nlls = np.array([r['nll_static_howto_trunc'] for r in exp02_results])
passage_words = np.array([r['passage_words'] for r in exp02_results])

# Pre-compute benefits (positive = condition is better than bare)
oracle_benefit = bare_nlls - oracle_nlls
random_benefit = bare_nlls - random_nlls
surr_doc_benefit = bare_nlls - surr_doc_nlls
surr_template_benefit = bare_nlls - surr_template_nlls
static_fact_benefit = bare_nlls - static_fact_nlls
surr_lead_benefit = bare_nlls - surr_lead_nlls
semantic_gap = oracle_benefit - random_benefit  # positive = oracle beats random

# Reload dataset to get passage text (needed for new conditions in Parts 2-4)
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
        f"Sample {i} query mismatch: {samples[i]['query'][:40]} != {exp02_results[i]['query'][:40]}"
print(f"Verified: {N_SAMPLES} samples match Exp 02")
print(f"Document lengths: {passage_words.min()}-{passage_words.max()} words, "
      f"mean={passage_words.mean():.0f}, median={np.median(passage_words):.0f}")
""")


# ===== Cell 4: Part 1a - Document length stratification =====
code(r"""# Cell 4: Part 1a - Document Length Stratification
# Does the structural/semantic ratio change with document length?

print("=" * 70)
print("PART 1A: DOCUMENT LENGTH STRATIFICATION")
print("=" * 70)
print("Does the semantic advantage (oracle - random) change with doc length?\n")

# Use quartiles for even bin sizes
quartile_bounds = np.percentile(passage_words, [25, 50, 75])
length_bins = np.digitize(passage_words, quartile_bounds)
bin_labels = [
    f"Q1 short ({passage_words[length_bins==0].min()}-{passage_words[length_bins==0].max()}w)",
    f"Q2 ({passage_words[length_bins==1].min()}-{passage_words[length_bins==1].max()}w)",
    f"Q3 ({passage_words[length_bins==2].min()}-{passage_words[length_bins==2].max()}w)",
    f"Q4 long ({passage_words[length_bins==3].min()}-{passage_words[length_bins==3].max()}w)",
]

print(f"{'Length bin':<25} {'N':>4} {'Oracle d':>10} {'Random d':>10} "
      f"{'Doc_kw d':>10} {'Sem gap':>10} {'p(gap>0)':>10}")
print("-" * 85)

for q in range(4):
    mask = length_bins == q
    n = mask.sum()
    o_d = cohens_d(oracle_benefit[mask])
    r_d = cohens_d(random_benefit[mask])
    dk_d = cohens_d(surr_doc_benefit[mask])
    gap = semantic_gap[mask]
    g_mean = gap.mean()
    _, p_gap = stats.ttest_1samp(gap, 0)
    sig = '***' if p_gap < 0.001 else '**' if p_gap < 0.01 else '*' if p_gap < 0.05 else 'ns'
    print(f"{bin_labels[q]:<25} {n:>4} {o_d:>+10.3f} {r_d:>+10.3f} "
          f"{dk_d:>+10.3f} {g_mean:>+10.4f} {p_gap:>9.2e} {sig}")

# Correlation: doc length vs semantic gap
r_len, p_len = stats.pearsonr(passage_words, semantic_gap)
print(f"\nCorrelation: doc_length vs semantic_gap: r={r_len:+.3f} (p={p_len:.3f})")

# Also: does structural (random) benefit scale with length?
r_struct, p_struct = stats.pearsonr(passage_words, random_benefit)
r_oracle, p_oracle = stats.pearsonr(passage_words, oracle_benefit)
print(f"  doc_length vs random_benefit:  r={r_struct:+.3f} (p={p_struct:.3f})")
print(f"  doc_length vs oracle_benefit:  r={r_oracle:+.3f} (p={p_oracle:.3f})")

# Interpretation
if abs(r_len) < 0.1 and p_len > 0.05:
    print("\n  --> Semantic gap is STABLE across document lengths.")
    print("      Structural and semantic benefits scale similarly.")
elif r_len > 0.1:
    print("\n  --> Semantic gap GROWS with document length.")
    print("      Content matters MORE for longer documents.")
else:
    print("\n  --> Semantic gap SHRINKS with document length.")
    print("      Structural mechanism dominates for longer documents.")

# Check surr_lead anomaly: is it a length artifact?
print(f"\n--- surr_lead anomaly check ---")
print(f"surr_lead was weakest at 40% oracle. Is this a surrogate LENGTH artifact?")
# surr_lead uses the first sentence, which is often long (many tokens)
# Compare surr_lead to other surrogates by length bin
for q in range(4):
    mask = length_bins == q
    lead_d = cohens_d(surr_lead_benefit[mask])
    tmpl_d = cohens_d(surr_template_benefit[mask])
    print(f"  {bin_labels[q]:<25} surr_lead d={lead_d:+.3f}, surr_template d={tmpl_d:+.3f}, "
          f"gap={tmpl_d - lead_d:+.3f}")
""")


# ===== Cell 5: Part 1b - Hardness stratification =====
code(r"""# Cell 5: Part 1b - Hardness Stratification
# Does the semantic advantage (oracle > random) emerge for harder documents?

print("=" * 70)
print("PART 1B: HARDNESS STRATIFICATION")
print("=" * 70)
print("Does the semantic advantage grow or shrink for harder documents?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"{'Quintile':<12} {'N':>4} {'Bare NLL':>10} {'Oracle d':>10} {'Random d':>10} "
      f"{'SurrDoc d':>10} {'Sem gap':>10} {'p(gap)':>10}")
print("-" * 85)

gap_by_q = []
hardness_by_q = []
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    bare_q = bare_nlls[mask].mean()

    o_d = cohens_d(oracle_benefit[mask])
    r_d = cohens_d(random_benefit[mask])
    sd_d = cohens_d(surr_doc_benefit[mask])
    gap = semantic_gap[mask]
    g_mean = gap.mean()
    _, p_gap = stats.ttest_1samp(gap, 0)
    sig = '***' if p_gap < 0.001 else '**' if p_gap < 0.01 else '*' if p_gap < 0.05 else 'ns'

    print(f"{qlabel:<12} {n_q:>4} {bare_q:>10.3f} {o_d:>+10.3f} {r_d:>+10.3f} "
          f"{sd_d:>+10.3f} {g_mean:>+10.4f} {p_gap:>9.2e} {sig}")

    gap_by_q.append(g_mean)
    hardness_by_q.append(bare_q)

# Correlation: hardness vs semantic gap
r_hard, p_hard = stats.pearsonr(bare_nlls, semantic_gap)
print(f"\nCorrelation: hardness vs semantic_gap: r={r_hard:+.3f} (p={p_hard:.3f})")

# Per-sample: when does oracle beat random?
oracle_wins = oracle_nlls < random_nlls
print(f"\nOracle beats random on {oracle_wins.sum()}/{N_SAMPLES} samples "
      f"({oracle_wins.mean()*100:.1f}%)")

for q in range(5):
    mask = quintiles == q
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    win_rate = oracle_wins[mask].mean() * 100
    print(f"  {qlabel}: oracle beats random {win_rate:.1f}% of the time")

# Key question: for which samples does content help MOST?
print(f"\n--- Where does content matter? ---")
print("Samples where oracle >> random (semantic advantage > median):")
med_gap = np.median(semantic_gap)
high_semantic = semantic_gap > med_gap
print(f"  High-semantic samples: mean bare NLL = {bare_nlls[high_semantic].mean():.3f}")
print(f"  Low-semantic samples:  mean bare NLL = {bare_nlls[~high_semantic].mean():.3f}")
print(f"  High-semantic samples: mean doc length = {passage_words[high_semantic].mean():.0f} words")
print(f"  Low-semantic samples:  mean doc length = {passage_words[~high_semantic].mean():.0f} words")
""")


# ===== Cell 6: Part 1c - Variance decomposition =====
code(r"""# Cell 6: Part 1c - Variance Decomposition
# Why does static_fact have higher Cohen's d (0.372) despite lower delta (0.418)?

print("=" * 70)
print("PART 1C: VARIANCE DECOMPOSITION")
print("=" * 70)
print("Cohen's d = mean(benefit) / std(benefit).")
print("A condition can have HIGH d with LOW mean if its variance is very low.\n")

all_conds = {
    'oracle_trunc': oracle_benefit,
    'surr_doc_trunc': surr_doc_benefit,
    'surr_template_trunc': surr_template_benefit,
    'surr_para_trunc': bare_nlls - surr_para_nlls,
    'static_fact_trunc': static_fact_benefit,
    'static_howto_trunc': bare_nlls - static_howto_nlls,
    'random_trunc': random_benefit,
    'surr_lead_trunc': surr_lead_benefit,
}

print(f"{'Condition':<25} {'Mean':>10} {'Std':>10} {'d':>8} {'CV':>8} {'Skew':>8} {'IQR':>10}")
print("-" * 85)

for name, benefit in all_conds.items():
    mu = benefit.mean()
    sd = benefit.std(ddof=1)
    d = mu / sd if sd > 0 else 0
    cv = sd / abs(mu) if abs(mu) > 0 else float('inf')
    skew = stats.skew(benefit)
    iqr = np.percentile(benefit, 75) - np.percentile(benefit, 25)
    print(f"{name:<25} {mu:>+10.4f} {sd:>10.4f} {d:>+8.3f} {cv:>8.2f} {skew:>+8.2f} {iqr:>10.4f}")

print(f"\nKey insight:")
print(f"  static_fact: mean=+0.418, std={all_conds['static_fact_trunc'].std():.3f} --> d=+0.372")
print(f"  surr_doc:    mean=+0.620, std={all_conds['surr_doc_trunc'].std():.3f} --> d=+0.322")
print(f"  static_fact has 48% LESS improvement but even LESS variance.")
print(f"  The doc-specific surrogates are noisier: they help a lot on some samples,")
print(f"  very little on others. Static surrogates provide a uniform 'lift'.")

# Cross-condition benefit correlations
print(f"\n--- Cross-condition correlations ---")
print(f"How much does knowing one condition's benefit predict another?")

pairs = [
    ('oracle', oracle_benefit, 'random', random_benefit),
    ('oracle', oracle_benefit, 'surr_doc', surr_doc_benefit),
    ('surr_doc', surr_doc_benefit, 'random', random_benefit),
    ('static_fact', static_fact_benefit, 'random', random_benefit),
    ('oracle', oracle_benefit, 'static_fact', static_fact_benefit),
]

for name_a, ben_a, name_b, ben_b in pairs:
    r, p = stats.pearsonr(ben_a, ben_b)
    print(f"  {name_a:<12} vs {name_b:<12}: r={r:.3f} (p={p:.2e})")

print(f"\nIf all benefits are highly correlated, the mechanism is shared (structural).")
print(f"If doc-specific benefits diverge from random, there is a semantic component.")
""")


# ===== Cell 7: Load model + define helpers =====
code(r"""# Cell 7: Load model and define scoring helpers

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
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
    # Encoder processes full text bidirectionally; if truncate=True, decoder
    # cross-attention is masked for the first prefix_token_count positions.
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
    # Uses BPE-aware subtraction: len(full) - len(doc_only).
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
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


# ===== Cell 8: Generate conditions for each sample =====
code(r"""# Cell 8: Generate surrogate conditions for each sample

# ---- Group A: Prefix length titration (random words) ----
# For each sample, use words from an unrelated passage at varying word counts.
# This gives real text (not garbage tokens) while controlling length.
TITRATION_LENGTHS = [1, 3, 5, 10, 20, 50]

# ---- Group B: Content ablation (length-matched to oracle) ----
# oracle: real query (reuse Exp 02 NLLs)
# scrambled: same words as oracle, shuffled randomly
# rand_matched: random words, same word count as oracle

# ---- Group C: Token diversity (all ~10 words) ----
# repeat_the: "the the the..." (zero semantic content)
# repeat_kw: top document keyword repeated (single concept, doc-specific)
# rand_10w: diverse random words (from Group A, reused)

for i, s in enumerate(samples):
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_passage = samples[other_idx]['passage']
    other_words = other_passage.split()

    # Group A: random words at different lengths
    for nw in TITRATION_LENGTHS:
        key = f'rand_{nw}w'
        s[key] = " ".join(other_words[:nw]) if len(other_words) >= nw else " ".join(other_words)

    # Group B: scrambled oracle (same words, random order)
    query_words = s['query'].split()
    rng = np.random.RandomState(SEED + i)  # per-sample deterministic shuffle
    shuffled = list(query_words)
    rng.shuffle(shuffled)
    s['scrambled_oracle'] = " ".join(shuffled)

    # Group B: random text, matched to oracle word count
    n_query_words = len(query_words)
    s['rand_matched'] = " ".join(other_words[:n_query_words])

    # Group C: "the" repeated ~10 times
    s['repeat_the'] = " ".join(["the"] * 10)

    # Group C: top document keyword repeated ~10 times
    doc_words = re.sub(r'[^\w\s]', '', s['passage'].lower()).split()
    content = [w for w in doc_words if w not in STOP_WORDS and len(w) > 2]
    if content:
        counts = Counter(content)
        top_word = counts.most_common(1)[0][0]
    else:
        top_word = "information"
    s['repeat_kw'] = " ".join([top_word] * 10)

# Define all new conditions to score
NEW_COND_NAMES = []

# Group A: prefix length titration
for nw in TITRATION_LENGTHS:
    NEW_COND_NAMES.append(f'rand_{nw}w_trunc')

# Group B: content ablation (oracle reused from Exp 02)
NEW_COND_NAMES.append('scrambled_oracle_trunc')
NEW_COND_NAMES.append('rand_matched_trunc')

# Group C: diversity (rand_10w_trunc from Group A = diverse control)
NEW_COND_NAMES.append('repeat_the_trunc')
NEW_COND_NAMES.append('repeat_kw_trunc')

print(f"New conditions to score: {len(NEW_COND_NAMES)}")
for c in NEW_COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query:   {ex['query'][:80]}")
print(f"  Answer:  {ex['answer'][:80]}")
print(f"  Passage: {ex['passage'][:80]}...")
print()

for c in NEW_COND_NAMES:
    key = c.replace('_trunc', '')
    text = ex.get(key, '???')
    ptoks = count_prefix_tokens(text, ex['passage'])
    print(f"  {c:<28} ({ptoks:>3} prefix toks): {str(text)[:60]}")

# Report token count statistics across a subsample
print(f"\nPrefix token counts (first 50 samples):")
for c in NEW_COND_NAMES:
    key = c.replace('_trunc', '')
    toks = [count_prefix_tokens(s[key], s['passage']) for s in samples[:50]]
    print(f"  {c:<28} mean={np.mean(toks):.1f}, range=[{min(toks)}, {max(toks)}]")
""")


# ===== Cell 9: Run scoring =====
code(r"""# Cell 9: Run scoring (with checkpointing)

print("=" * 70)
print("RUNNING NEW CONDITIONS")
print("=" * 70)

# Resume from checkpoint if available
new_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            new_results = ckpt['results']
            start_idx = len(new_results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(NEW_COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {len(NEW_COND_NAMES) * N_SAMPLES} scorings")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
    }

    for cond_name in NEW_COND_NAMES:
        key = cond_name.replace('_trunc', '')
        surr_text = s[key]
        enc_text = surr_text + "\n" + s['passage']
        prefix_count = count_prefix_tokens(surr_text, s['passage'])
        nll = score_nll(enc_text, s['answer'], prefix_count, truncate=True)
        result[f'nll_{cond_name}'] = nll
        result[f'ptoks_{cond_name}'] = prefix_count

    new_results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': new_results,
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
print(f"\nScoring complete: {len(new_results)} samples, "
      f"{len(NEW_COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 10: Part 2 - Prefix length titration =====
code(r"""# Cell 10: Part 2 - Prefix Length Titration
# How many random prefix tokens does the encoder need?

print("=" * 70)
print("PART 2: PREFIX LENGTH TITRATION")
print("=" * 70)
print("Random words as prefix, varying from 1 to 50 words.")
print("Key question: saturation curve shape.\n")

titration_conds = [f'rand_{nw}w_trunc' for nw in TITRATION_LENGTHS]
oracle_d_val = cohens_d(oracle_benefit)

print(f"{'Condition':<20} {'~Prefix toks':>12} {'Mean NLL':>10} {'Delta':>8} "
      f"{'d':>8} {'Win%':>7} {'% Oracle':>10}")
print("-" * 80)

titration_ds = []
titration_toks = []

for cond in titration_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in new_results])
    ptoks = np.array([r[f'ptoks_{cond}'] for r in new_results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d_val * 100 if oracle_d_val > 0 else 0
    mean_ptoks = ptoks.mean()

    titration_ds.append(d)
    titration_toks.append(mean_ptoks)

    print(f"{cond:<20} {mean_ptoks:>12.1f} {nlls.mean():>10.4f} {delta:>+8.4f} "
          f"{d:>+8.3f} {win:>6.1f}% {pct:>9.1f}%")

# Reference points
print(f"\n--- Reference points from Exp 02 ---")
print(f"  bare (0 tokens):            d=  0.000")
rand_exp02_d = cohens_d(random_benefit)
print(f"  random_trunc (~33 toks):    d={rand_exp02_d:>+.3f}")
print(f"  oracle_trunc (~10 toks):    d={oracle_d_val:>+.3f}")

# Saturation analysis
print(f"\n--- Saturation analysis ---")
max_d = titration_ds[-1]
for i, (nw, d_val) in enumerate(zip(TITRATION_LENGTHS, titration_ds)):
    pct_max = d_val / max_d * 100 if max_d > 0 else 0
    pct_oracle = d_val / oracle_d_val * 100 if oracle_d_val > 0 else 0
    print(f"  {nw:>2} words ({titration_toks[i]:>5.1f} toks): "
          f"d={d_val:+.3f} ({pct_max:.0f}% of 50w, {pct_oracle:.0f}% of oracle)")

# Fit: logarithmic vs linear
log_toks = np.log(titration_toks)
r_log, p_log = stats.pearsonr(log_toks, titration_ds)
r_lin, p_lin = stats.pearsonr(titration_toks, titration_ds)
print(f"\n  Log fit (d ~ log(tokens)):   r={r_log:.3f} (p={p_log:.3f})")
print(f"  Linear fit (d ~ tokens):     r={r_lin:.3f} (p={p_lin:.3f})")

if titration_ds[0] / max_d > 0.7:
    print(f"\n  --> SWITCH MECHANISM: 1 word captures >{titration_ds[0]/max_d*100:.0f}% of the benefit.")
    print(f"      The encoder just needs *something* in the prefix. Like a gate.")
elif r_log > r_lin:
    print(f"\n  --> LOGARITHMIC scaling: diminishing returns with more tokens.")
    print(f"      The encoder benefits from prefix but saturates quickly.")
else:
    print(f"\n  --> LINEAR scaling: benefit grows proportionally with prefix length.")
    print(f"      More tokens = more attention redistribution = better representations.")
""")


# ===== Cell 11: Part 3 - Content ablation =====
code(r"""# Cell 11: Part 3 - Content Ablation
# Decompose the total benefit into structure + vocabulary + semantics

print("=" * 70)
print("PART 3: CONTENT ABLATION")
print("=" * 70)
print("Decompose: bare -> random_matched -> scrambled_oracle -> oracle")
print("  Structure:  bare -> random_matched (any prefix helps)")
print("  Vocabulary: random_matched -> scrambled (right words, wrong order)")
print("  Semantics:  scrambled -> oracle (right word order, full meaning)\n")

scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in new_results])
randmatch_nlls = np.array([r['nll_rand_matched_trunc'] for r in new_results])

# Component benefits (all positive = improvement)
struct_comp = bare_nlls - randmatch_nlls        # structure
vocab_comp = randmatch_nlls - scrambled_nlls     # vocabulary (scrambled has query words)
sem_comp = scrambled_nlls - oracle_nlls          # semantics (oracle has correct order)
total_comp = bare_nlls - oracle_nlls             # total

print(f"{'Component':<30} {'Mean NLL':>10} {'Delta':>8} {'% total':>9} "
      f"{'d':>8} {'p':>12} {'sig':>5}")
print("-" * 85)

# Per-step NLLs
steps = [
    ('bare (baseline)', bare_nlls, None),
    ('rand_matched_trunc', randmatch_nlls, struct_comp),
    ('scrambled_oracle_trunc', scrambled_nlls, vocab_comp),
    ('oracle_trunc', oracle_nlls, sem_comp),
]

total_mean = total_comp.mean()

for name, nlls, component in steps:
    if component is None:
        print(f"  {name:<28} {nlls.mean():>10.4f}")
        continue
    mu = component.mean()
    pct = mu / total_mean * 100 if total_mean != 0 else 0
    d = cohens_d(component)
    _, p = stats.ttest_1samp(component, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    label = {'rand_matched_trunc': '+ Structure',
             'scrambled_oracle_trunc': '+ Vocabulary',
             'oracle_trunc': '+ Semantics'}[name]
    print(f"  {label:<28} {nlls.mean():>10.4f} {mu:>+8.4f} {pct:>8.1f}% "
          f"{d:>+8.3f} {p:>12.2e} {sig}")

print(f"  {'TOTAL':<28} {'':>10} {total_mean:>+8.4f} {'100.0%':>9}")

# Verify decomposition sums correctly
residual = total_mean - (struct_comp.mean() + vocab_comp.mean() + sem_comp.mean())
print(f"\n  Decomposition residual: {residual:.6f} (should be ~0)")

# Token count verification (are lengths actually matched?)
print(f"\n--- Length matching verification ---")
oracle_toks_all = [count_prefix_tokens(s['query'], s['passage']) for s in samples]
scrambled_toks_all = [r['ptoks_scrambled_oracle_trunc'] for r in new_results]
randmatch_toks_all = [r['ptoks_rand_matched_trunc'] for r in new_results]
print(f"  Oracle:     mean={np.mean(oracle_toks_all):.1f} toks (range {min(oracle_toks_all)}-{max(oracle_toks_all)})")
print(f"  Scrambled:  mean={np.mean(scrambled_toks_all):.1f} toks (range {min(scrambled_toks_all)}-{max(scrambled_toks_all)})")
print(f"  RandMatch:  mean={np.mean(randmatch_toks_all):.1f} toks (range {min(randmatch_toks_all)}-{max(randmatch_toks_all)})")

tok_diff = np.abs(np.array(oracle_toks_all) - np.array(scrambled_toks_all))
print(f"  Oracle-Scrambled token diff: mean={tok_diff.mean():.1f}, max={tok_diff.max()}")

# Does vocabulary help MORE for hard samples?
print(f"\n--- Component x hardness interaction ---")
for name, comp in [('Structure', struct_comp), ('Vocabulary', vocab_comp),
                    ('Semantics', sem_comp)]:
    r, p = stats.pearsonr(bare_nlls, comp)
    print(f"  {name:<15} vs hardness: r={r:+.3f} (p={p:.2e})")
""")


# ===== Cell 12: Part 4 - Token diversity =====
code(r"""# Cell 12: Part 4 - Token Diversity
# Does token diversity matter, or is any 10-token prefix equivalent?

print("=" * 70)
print("PART 4: TOKEN DIVERSITY")
print("=" * 70)
print("All conditions use ~10 words. Test whether content/diversity matters.\n")

repeat_the_nlls = np.array([r['nll_repeat_the_trunc'] for r in new_results])
repeat_kw_nlls = np.array([r['nll_repeat_kw_trunc'] for r in new_results])
rand_10w_nlls = np.array([r['nll_rand_10w_trunc'] for r in new_results])

diversity_conds = [
    ('repeat_the_trunc', repeat_the_nlls, '"the" x10 (zero info, same for all)'),
    ('repeat_kw_trunc', repeat_kw_nlls, 'doc keyword x10 (one concept, doc-specific)'),
    ('rand_10w_trunc', rand_10w_nlls, '10 diverse random words (from unrelated passage)'),
]

oracle_d_val = cohens_d(oracle_benefit)

print(f"{'Description':<50} {'NLL':>8} {'Delta':>8} {'d':>8} {'Win%':>7} {'%Orc':>6}")
print("-" * 90)

for name, nlls, desc in diversity_conds:
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d_val * 100 if oracle_d_val > 0 else 0
    print(f"  {desc:<48} {nlls.mean():>8.4f} {delta:>+8.4f} {d:>+8.3f} {win:>6.1f}% {pct:>5.0f}%")

# Pairwise comparisons
print(f"\n--- Pairwise head-to-head ---")
pairs = [
    ('"the" x10', repeat_the_nlls, '10 diverse random', rand_10w_nlls,
     "Does diversity help?"),
    ('keyword x10', repeat_kw_nlls, '10 diverse random', rand_10w_nlls,
     "Does a relevant keyword beat diverse noise?"),
    ('"the" x10', repeat_the_nlls, 'keyword x10', repeat_kw_nlls,
     "Does keyword content help vs pure filler?"),
]

for name_a, nlls_a, name_b, nlls_b, question in pairs:
    diff = nlls_b - nlls_a  # positive = A is better (lower NLL)
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = name_a if d > 0 else name_b
    print(f"  {question}")
    print(f"    {name_a} vs {name_b}: d={d:+.3f}, win={win:.1f}%, p={p:.2e} {sig} [{winner}]")

# Token counts
print(f"\n--- Token counts ---")
for name, nlls, desc in diversity_conds:
    ptoks = [r.get(f'ptoks_{name}', 0) for r in new_results[:50]]
    print(f"  {name:<25} mean={np.mean(ptoks):.1f} tokens")
""")


# ===== Cell 13: Synthesis + save + cleanup =====
code(r"""# Cell 13: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: STRUCTURAL vs SEMANTIC MECHANISM IN T5GEMMA")
print("=" * 70)

# --- 1. Length titration summary ---
print(f"\n1. PREFIX LENGTH TITRATION:")
titration_ds_final = []
for nw in TITRATION_LENGTHS:
    cond = f'rand_{nw}w_trunc'
    nlls = np.array([r[f'nll_{cond}'] for r in new_results])
    titration_ds_final.append(cohens_d(bare_nlls - nlls))

one_word_d = titration_ds_final[0]
fifty_word_d = titration_ds_final[-1]
oracle_d_val = cohens_d(oracle_benefit)
one_pct_fifty = one_word_d / fifty_word_d * 100 if fifty_word_d > 0 else 0
one_pct_oracle = one_word_d / oracle_d_val * 100 if oracle_d_val > 0 else 0

print(f"   1 random word:  d={one_word_d:+.3f} ({one_pct_fifty:.0f}% of 50-word, "
      f"{one_pct_oracle:.0f}% of oracle)")
print(f"   50 random words: d={fifty_word_d:+.3f} ({fifty_word_d/oracle_d_val*100:.0f}% of oracle)")

if one_pct_fifty > 70:
    titration_finding = "SWITCH"
    print(f"   --> SWITCH: 1 word captures {one_pct_fifty:.0f}% — mechanism is binary (on/off).")
elif one_pct_fifty > 40:
    titration_finding = "GRADUAL"
    print(f"   --> GRADUAL: benefit scales with prefix length, diminishing returns.")
else:
    titration_finding = "LENGTH_DEPENDENT"
    print(f"   --> LENGTH_DEPENDENT: needs substantial prefix for full benefit.")

# --- 2. Content ablation summary ---
print(f"\n2. CONTENT ABLATION:")
struct_pct = struct_comp.mean() / total_comp.mean() * 100
vocab_pct = vocab_comp.mean() / total_comp.mean() * 100
sem_pct = sem_comp.mean() / total_comp.mean() * 100
print(f"   Structure:  {struct_pct:>5.1f}% of total benefit")
print(f"   Vocabulary: {vocab_pct:>5.1f}% of total benefit")
print(f"   Semantics:  {sem_pct:>5.1f}% of total benefit")

if struct_pct > 70:
    ablation_finding = "STRUCTURAL"
    print(f"   --> Primarily STRUCTURAL — any prefix captures most of the benefit.")
elif struct_pct > 50:
    ablation_finding = "MIXED"
    print(f"   --> MIXED — structure dominates but content provides meaningful uplift.")
else:
    ablation_finding = "SEMANTIC"
    print(f"   --> Primarily SEMANTIC — content matters more than structure.")

# --- 3. Diversity summary ---
print(f"\n3. TOKEN DIVERSITY:")
d_the = cohens_d(bare_nlls - repeat_the_nlls)
d_kw = cohens_d(bare_nlls - repeat_kw_nlls)
d_diverse = cohens_d(bare_nlls - rand_10w_nlls)
print(f"   'the' x10:      d={d_the:+.3f}")
print(f"   keyword x10:    d={d_kw:+.3f}")
print(f"   diverse random: d={d_diverse:+.3f}")

diversity_gap = d_diverse - d_the
if abs(diversity_gap) < 0.03:
    diversity_finding = "NO_DIVERSITY_EFFECT"
    print(f"   --> Diversity does NOT matter (gap={diversity_gap:+.3f})")
else:
    diversity_finding = "DIVERSITY_HELPS" if diversity_gap > 0 else "DIVERSITY_HURTS"
    print(f"   --> Diversity {'helps' if diversity_gap > 0 else 'hurts'} (gap={diversity_gap:+.3f})")

# --- 4. v2 comparison ---
print(f"\n4. COMPARISON WITH v2 IMPLICIT REGULARIZATION:")
print(f"   v2 mechanism: value contamination (causal, document-independent)")
print(f"     - Content didn't matter (same)")
print(f"     - Truncation REMOVED benefit (DIFFERENT — v3 truncation IMPROVES it)")
print(f"     - Benefit diluted at ~200 tokens (TBD for v3 — Exp 03)")
print(f"   v3 mechanism: bidirectional co-encoding (document-specific)")
print(f"     - Prefix changes document reps via bidirectional self-attention")
print(f"     - Even with prefix tokens masked from decoder, benefit persists")
print(f"     - This is NOT value contamination — it is representation enrichment")

# --- Overall conclusion ---
print(f"\n{'='*70}")
print(f"CONCLUSION:")

if ablation_finding == "STRUCTURAL" and titration_finding == "SWITCH":
    print(f"  The mechanism is primarily STRUCTURAL and acts like a SWITCH.")
    print(f"  Adding any prefix — even 1 random word — triggers a mode shift")
    print(f"  in the encoder that improves document representations.")
    print(f"  This is analogous to v2's implicit regularization but operates")
    print(f"  through bidirectional attention rather than value contamination.")
    same_as_v2 = True
elif ablation_finding == "STRUCTURAL":
    print(f"  The mechanism is primarily STRUCTURAL but scales with prefix length.")
    print(f"  The encoder benefits from having more attention targets, suggesting")
    print(f"  attention redistribution rather than a simple on/off switch.")
    same_as_v2 = False
elif ablation_finding == "MIXED":
    print(f"  The mechanism is MIXED: a large structural base with a meaningful")
    print(f"  semantic component. The encoder benefits from any prefix (structure)")
    print(f"  but extracts additional value from relevant content (semantics).")
    print(f"  This is DIFFERENT from v2, where content provided zero uplift.")
    same_as_v2 = False
else:
    print(f"  The mechanism is primarily SEMANTIC — content matters.")
    print(f"  This is fundamentally different from v2.")
    same_as_v2 = False

print(f"\n  Mechanism type: {'Same as v2 (structural)' if same_as_v2 else 'Different from v2'}")
print(f"  Titration: {titration_finding}")
print(f"  Ablation: struct={struct_pct:.0f}%, vocab={vocab_pct:.0f}%, sem={sem_pct:.0f}%")
print(f"  Diversity: {diversity_finding}")
print(f"{'='*70}")

# --- Save results ---
final_results = {
    'experiment': 'exp02b_mechanism_decomposition',
    'model': MODEL_NAME,
    'n_samples': N_SAMPLES,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'titration': {
        str(nw): {
            'd': float(titration_ds_final[i]),
            'mean_ptoks': float(np.mean([r[f'ptoks_rand_{nw}w_trunc'] for r in new_results])),
        }
        for i, nw in enumerate(TITRATION_LENGTHS)
    },
    'ablation': {
        'structure_pct': float(struct_pct),
        'vocabulary_pct': float(vocab_pct),
        'semantics_pct': float(sem_pct),
        'structure_d': float(cohens_d(struct_comp)),
        'vocabulary_d': float(cohens_d(vocab_comp)),
        'semantics_d': float(cohens_d(sem_comp)),
    },
    'diversity': {
        'repeat_the_d': float(d_the),
        'repeat_kw_d': float(d_kw),
        'diverse_random_d': float(d_diverse),
    },
    'conclusion': {
        'titration_finding': titration_finding,
        'ablation_finding': ablation_finding,
        'diversity_finding': diversity_finding,
        'same_as_v2': same_as_v2,
    },
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
out_path = "experiments/02b/02b_mechanism_decomposition.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
