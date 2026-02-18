#!/usr/bin/env python3
# Build Exp 07 notebook: RoPE Isolation.
#
# Tests whether the structural benefit comes from attention redistribution
# (prefix tokens as new attention targets) or from RoPE position shifts
# (document tokens at different absolute positions).
#
# Since RoPE is purely relative (doc-doc attention depends on i-j not on i),
# shifting all doc positions shouldn't change doc-doc attention. But this
# experiment provides empirical confirmation.
#
# Design: 2x2 factorial — {RoPE shift present/absent} x {prefix in attention ON/OFF}
# 6 conditions x 500 = 3000 forward passes. ~10 min.
#
# Prediction: benefit comes ENTIRELY from attention redistribution, not RoPE.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 07: RoPE Isolation

## Motivation

When we prepend a prefix to the encoder input, two things change simultaneously:

1. **Attention redistribution**: New tokens in the attention pool soak up attention mass,
   reorganizing doc-doc attention patterns (KL up to 2.97 nats, Exp 3E).
2. **RoPE position shift**: Document tokens shift from positions [0..L-1] to [N..N+L-1],
   changing the absolute RoPE embeddings.

Since RoPE is *purely relative* (attention score between positions i and j depends only
on i-j, not on i or j individually), the doc-doc relative positions don't change when
a prefix is prepended. Therefore the RoPE position shift should NOT affect doc-doc
attention patterns. But this hasn't been empirically verified.

## Design: 2x2 Factorial

| | Prefix in attention (ON) | Prefix in attention (OFF) |
|---|---|---|
| **RoPE shift (ON)** | `random_trunc` (standard) | `prefix_encoder_blocked` |
| **RoPE shift (OFF)** | `random_rope_neutralized` | `bare` (baseline) |

Plus `oracle_trunc` as upper bound and `shifted_bare` as pure absolute position test.

### 6 Conditions

| # | Condition | Input | position_ids | Encoder attention | Cross-attn |
|---|-----------|-------|-------------|-------------------|-----------|
| 1 | `bare` | [doc] | [0..L-1] | normal | all |
| 2 | `oracle_trunc` | [oracle+doc] | default | normal | mask prefix |
| 3 | `random_trunc` | [random+doc] | default | normal | mask prefix |
| 4 | `shifted_bare` | [doc] | [6..6+L-1] | normal | all |
| 5 | `prefix_encoder_blocked` | [random+doc] | default | doc→prefix BLOCKED | mask prefix |
| 6 | `random_rope_neutralized` | [random+doc] | [L..L+N, 0..L-1] | normal | mask prefix |

### Predictions

- **4 ≈ 1**: Shifting bare doc positions has no effect (RoPE is relative)
- **5 ≈ 1**: Invisible prefix + position shift has no effect (no attention redistribution)
- **6 ≈ 3**: Benefit comes from attention redistribution, not position shift
- **3 >> 1**: The benefit is from having prefix tokens in the attention pool
- If all predictions hold: **RoPE conclusively ruled out as a contributor**""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, gc, random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, ".")
from lib.analysis import cohens_d

SEED = 42  # Same samples as Exp 02
N_SAMPLES = 500
POSITION_OFFSET = 6  # Typical prefix length in tokens
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("results/exp07")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

print("Exp 07: RoPE Isolation")
print(f"N: {N_SAMPLES}, position offset: {POSITION_OFFSET}")
""")


# ===== Cell 3: Load MS MARCO (same 500 samples as Exp 02) =====
code(r"""# Cell 3: Load MS MARCO — reconstruct same 500 samples as Exp 02
from datasets import load_dataset

print("Loading MS MARCO...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# Reconstruct same candidates as Exp 02
from lib.data import count_words

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

print(f"Loaded {len(samples)} samples")
print(f"First query: {samples[0]['query'][:60]}...")

# Verify alignment with Exp 02
exp02_ckpt = Path("results/exp02/checkpoint.json")
if exp02_ckpt.exists():
    ckpt02 = json.loads(exp02_ckpt.read_text())
    if 'results' in ckpt02 and len(ckpt02['results']) > 0:
        match = all(
            r['query'][:50] == s['query'][:50]
            for r, s in zip(ckpt02['results'][:10], samples[:10])
        )
        print(f"Sample alignment with Exp 02: {'MATCH' if match else 'MISMATCH'}")
""")


# ===== Cell 4: Load model + define scoring functions =====
code(r"""# Cell 4: Load model and define scoring functions
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

# Verify encoder accepts position_ids
import inspect
enc_sig = inspect.signature(model.get_encoder().forward)
assert 'position_ids' in enc_sig.parameters, "Encoder does not accept position_ids!"
print(f"Encoder accepts position_ids: YES")

# Get encoder layer types for attention mask dict
encoder_text_model = model.model.encoder.text_model
layer_types = set()
for layer in encoder_text_model.layers:
    layer_types.add(layer.attention_type)
print(f"Encoder layer types: {layer_types}")


def _decode_nll(encoder_outputs, cross_attn_mask, answer_text):
    # Shared decoder scoring: NLL of answer given encoder output + mask.
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
    del outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    # BPE-aware token count of prefix in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_bare(passage, answer):
    # Condition 1: Standard bare encoding.
    enc_ids = tokenizer(passage, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    enc_mask = torch.ones(1, enc_ids.shape[1], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )
    nll = _decode_nll(encoder_outputs, enc_mask, answer)
    del encoder_outputs
    return nll


def score_prefix_trunc(prefix, passage, answer):
    # Conditions 2-3: Standard prefix + truncation (oracle or random).
    full_text = prefix + "\n" + passage
    enc_ids = tokenizer(full_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_len = enc_ids.shape[1]
    n_prefix = count_prefix_tokens(prefix, passage)
    enc_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # Cross-attention mask: hide prefix from decoder
    cross_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)
    cross_mask[:, :n_prefix] = 0
    nll = _decode_nll(encoder_outputs, cross_mask, answer)
    del encoder_outputs
    return nll


def score_shifted_bare(passage, answer, offset=POSITION_OFFSET):
    # Condition 4: Bare encoding with shifted position_ids.
    # Document at positions [offset..offset+L-1] instead of [0..L-1].
    # Tests pure absolute RoPE shift (no prefix tokens).
    enc_ids = tokenizer(passage, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    seq_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, seq_len, device=DEVICE, dtype=torch.long)
    position_ids = torch.arange(offset, offset + seq_len,
                                device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask,
            position_ids=position_ids,
        )

    nll = _decode_nll(encoder_outputs, enc_mask, answer)
    del encoder_outputs
    return nll


def score_prefix_encoder_blocked(prefix, passage, answer):
    # Condition 5: Prefix in sequence but blocked from encoder attention.
    # Doc tokens cannot attend to prefix tokens. Positions ARE shifted.
    # Tests: does RoPE shift + invisible prefix do anything?
    full_text = prefix + "\n" + passage
    enc_ids = tokenizer(full_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_len = enc_ids.shape[1]
    n_prefix = count_prefix_tokens(prefix, passage)

    # Build 4D additive attention mask: doc tokens cannot attend to prefix
    # Shape: [1, 1, total_len, total_len], dtype matching model
    # 0.0 = allowed, large negative = blocked
    min_val = torch.finfo(torch.bfloat16).min
    mask_4d = torch.zeros(1, 1, total_len, total_len,
                          dtype=torch.bfloat16, device=DEVICE)
    # Block: doc rows [n_prefix:] attending to prefix cols [:n_prefix]
    mask_4d[:, :, n_prefix:, :n_prefix] = min_val

    # Create dict for both layer types (sequences are short enough that
    # sliding window doesn't actually mask anything)
    encoder_mask = {
        "full_attention": mask_4d,
        "sliding_attention": mask_4d.clone(),
    }

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=encoder_mask,
        )

    # Cross-attention: also mask prefix from decoder
    cross_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)
    cross_mask[:, :n_prefix] = 0
    nll = _decode_nll(encoder_outputs, cross_mask, answer)
    del encoder_outputs, mask_4d, encoder_mask
    return nll


def score_prefix_rope_neutralized(prefix, passage, answer):
    # Condition 6: Prefix in attention but doc positions NOT shifted.
    # position_ids: prefix at [L..L+N-1], doc at [0..L-1].
    # Tests: does attention redistribution work without RoPE position shift?
    full_text = prefix + "\n" + passage
    enc_ids = tokenizer(full_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_len = enc_ids.shape[1]
    n_prefix = count_prefix_tokens(prefix, passage)
    doc_len = total_len - n_prefix

    # Custom position_ids: prefix at [doc_len .. doc_len+n_prefix-1],
    # doc at [0 .. doc_len-1]
    prefix_positions = torch.arange(doc_len, doc_len + n_prefix, device=DEVICE)
    doc_positions = torch.arange(0, doc_len, device=DEVICE)
    position_ids = torch.cat([prefix_positions, doc_positions]).unsqueeze(0)

    enc_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask,
            position_ids=position_ids,
        )

    # Cross-attention: mask prefix from decoder
    cross_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)
    cross_mask[:, :n_prefix] = 0
    nll = _decode_nll(encoder_outputs, cross_mask, answer)
    del encoder_outputs
    return nll


print("All scoring functions defined.")

# Quick sanity check on a single sample
s = samples[0]
nll_bare = score_bare(s['passage'], s['answer'])
nll_shift = score_shifted_bare(s['passage'], s['answer'])
print(f"\nSanity check (sample 0):")
print(f"  bare NLL:          {nll_bare:.6f}")
print(f"  shifted_bare NLL:  {nll_shift:.6f}")
print(f"  difference:        {abs(nll_bare - nll_shift):.6f}")
if abs(nll_bare - nll_shift) < 0.01:
    print("  -> Very close (consistent with RoPE being relative)")
else:
    print("  -> DIFFERENT (absolute position matters!)")
""")


# ===== Cell 5: Generate random prefixes =====
code(r"""# Cell 5: Generate random prefixes for each sample
# Use the same random prefix generation as Exp 02/2B

for i, s in enumerate(samples):
    query = s['query']
    query_words_list = query.split()

    # Random prefix from unrelated passage, matched to query length
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    random_matched = " ".join(other_words[:len(query_words_list)])

    s['oracle'] = query
    s['random_matched'] = random_matched

    # Count tokens for reference
    s['n_prefix_oracle'] = count_prefix_tokens(query, s['passage'])
    s['n_prefix_random'] = count_prefix_tokens(random_matched, s['passage'])

COND_NAMES = [
    'bare',
    'oracle_trunc',
    'random_trunc',
    'shifted_bare',
    'prefix_encoder_blocked',
    'random_rope_neutralized',
]

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query: {ex['query']}")
print(f"  Answer: {ex['answer']}")
print(f"  Passage ({ex['word_count']}w): {ex['passage'][:80]}...")
print(f"  Random prefix: {ex['random_matched']}")
print(f"  Oracle prefix tokens: {ex['n_prefix_oracle']}, "
      f"Random prefix tokens: {ex['n_prefix_random']}")
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
    }

    # Condition 1: bare
    result['nll_bare'] = score_bare(s['passage'], s['answer'])

    # Condition 2: oracle_trunc (standard)
    result['nll_oracle_trunc'] = score_prefix_trunc(
        s['oracle'], s['passage'], s['answer'])

    # Condition 3: random_trunc (standard)
    result['nll_random_trunc'] = score_prefix_trunc(
        s['random_matched'], s['passage'], s['answer'])

    # Condition 4: shifted_bare (pure RoPE shift)
    result['nll_shifted_bare'] = score_shifted_bare(
        s['passage'], s['answer'], offset=s['n_prefix_random'])

    # Condition 5: prefix_encoder_blocked (RoPE shift + invisible prefix)
    result['nll_prefix_encoder_blocked'] = score_prefix_encoder_blocked(
        s['random_matched'], s['passage'], s['answer'])

    # Condition 6: random_rope_neutralized (attention redistribution, no RoPE shift)
    result['nll_random_rope_neutralized'] = score_prefix_rope_neutralized(
        s['random_matched'], s['passage'], s['answer'])

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


# ===== Cell 7: Analysis =====
code(r"""# Cell 7: Analysis — RoPE Isolation Results

print("=" * 70)
print("ROPE ISOLATION ANALYSIS")
print("=" * 70)

# Extract arrays
bare = np.array([r['nll_bare'] for r in results])
oracle = np.array([r['nll_oracle_trunc'] for r in results])
random_std = np.array([r['nll_random_trunc'] for r in results])
shifted = np.array([r['nll_shifted_bare'] for r in results])
blocked = np.array([r['nll_prefix_encoder_blocked'] for r in results])
neutralized = np.array([r['nll_random_rope_neutralized'] for r in results])

# Verify Exp 02 alignment
print(f"\nBaseline verification:")
print(f"  bare mean NLL:    {bare.mean():.6f}")
print(f"  oracle mean NLL:  {oracle.mean():.6f}")

# ---- Test 1: shifted_bare vs bare (pure absolute RoPE) ----
print(f"\n{'='*70}")
print(f"TEST 1: Does absolute RoPE position shift matter?")
print(f"  shifted_bare: doc at positions [N..N+L-1] instead of [0..L-1]")
print(f"{'='*70}")

diff_shift = bare - shifted  # positive = shifted is better
d_shift = cohens_d(diff_shift)
_, p_shift = stats.ttest_1samp(diff_shift, 0)
sig_shift = '***' if p_shift < 0.001 else '**' if p_shift < 0.01 else '*' if p_shift < 0.05 else 'ns'

print(f"  bare mean NLL:         {bare.mean():.6f}")
print(f"  shifted_bare mean NLL: {shifted.mean():.6f}")
print(f"  difference:            {diff_shift.mean():+.6f}")
print(f"  Cohen's d:             {d_shift:+.4f} ({sig_shift})")
print(f"  Max abs difference:    {np.abs(diff_shift).max():.6f}")
corr, _ = stats.pearsonr(bare, shifted)
print(f"  Correlation:           r={corr:.6f}")

if abs(d_shift) < 0.05:
    print(f"\n  -> CONFIRMED: Absolute RoPE position shift has NO effect (d={d_shift:+.4f})")
    print(f"     RoPE is purely relative — shifting all positions preserves doc-doc attention.")
else:
    print(f"\n  -> UNEXPECTED: Position shift has an effect (d={d_shift:+.4f})")

# ---- Test 2: prefix_encoder_blocked vs bare (RoPE + invisible prefix) ----
print(f"\n{'='*70}")
print(f"TEST 2: Does invisible prefix + position shift matter?")
print(f"  prefix is in sequence but doc can't attend to it")
print(f"{'='*70}")

diff_blocked = bare - blocked  # positive = blocked is better
d_blocked = cohens_d(diff_blocked)
_, p_blocked = stats.ttest_1samp(diff_blocked, 0)
sig_blocked = '***' if p_blocked < 0.001 else '**' if p_blocked < 0.01 else '*' if p_blocked < 0.05 else 'ns'

print(f"  bare mean NLL:                  {bare.mean():.6f}")
print(f"  prefix_encoder_blocked mean NLL:{blocked.mean():.6f}")
print(f"  difference:                     {diff_blocked.mean():+.6f}")
print(f"  Cohen's d:                      {d_blocked:+.4f} ({sig_blocked})")

# Compare to shifted_bare — should be nearly identical
diff_shift_vs_blocked = shifted - blocked
d_sb = cohens_d(diff_shift_vs_blocked)
print(f"\n  shifted_bare vs blocked: d={d_sb:+.4f} (should be ~0)")

if abs(d_blocked) < 0.05:
    print(f"\n  -> CONFIRMED: Invisible prefix has NO effect beyond RoPE shift.")
    print(f"     Prefix tokens must be attended to for benefit.")
else:
    print(f"\n  -> UNEXPECTED: Invisible prefix has an effect (d={d_blocked:+.4f})")

# ---- Test 3: random_rope_neutralized vs random_trunc (attention only vs attention+RoPE) ----
print(f"\n{'='*70}")
print(f"TEST 3: Does neutralizing RoPE reduce the attention redistribution benefit?")
print(f"  random_trunc: positions shifted (standard)")
print(f"  random_rope_neutralized: doc keeps original positions")
print(f"{'='*70}")

# Each vs bare
diff_random = bare - random_std
d_random = cohens_d(diff_random)
_, p_random = stats.ttest_1samp(diff_random, 0)
sig_random = '***' if p_random < 0.001 else '**' if p_random < 0.01 else '*' if p_random < 0.05 else 'ns'

diff_neutral = bare - neutralized
d_neutral = cohens_d(diff_neutral)
_, p_neutral = stats.ttest_1samp(diff_neutral, 0)
sig_neutral = '***' if p_neutral < 0.001 else '**' if p_neutral < 0.01 else '*' if p_neutral < 0.05 else 'ns'

print(f"  random_trunc vs bare:       d={d_random:+.4f} ({sig_random})")
print(f"  rope_neutralized vs bare:   d={d_neutral:+.4f} ({sig_neutral})")

# Head-to-head
diff_std_vs_neutral = neutralized - random_std  # positive = standard is better
d_svn = cohens_d(diff_std_vs_neutral)
_, p_svn = stats.ttest_1samp(diff_std_vs_neutral, 0)
sig_svn = '***' if p_svn < 0.001 else '**' if p_svn < 0.01 else '*' if p_svn < 0.05 else 'ns'
win_svn = np.mean(diff_std_vs_neutral > 0) * 100

print(f"\n  Head-to-head (standard - neutralized):")
print(f"    d={d_svn:+.4f}, win%={win_svn:.1f}%, p={p_svn:.2e} ({sig_svn})")

ratio = d_neutral / d_random * 100 if d_random != 0 else 0
print(f"\n  Benefit ratio: neutralized/standard = {ratio:.1f}%")

if abs(d_svn) < 0.05:
    print(f"\n  -> CONFIRMED: Neutralizing RoPE has NO effect on the benefit.")
    print(f"     The benefit is entirely from attention redistribution.")
elif d_svn > 0.05:
    print(f"\n  -> RoPE contributes: standard > neutralized by d={d_svn:+.4f}")
else:
    print(f"\n  -> Neutralized is actually BETTER (d={d_svn:+.4f})")

# ---- Summary 2x2 Table ----
print(f"\n{'='*70}")
print(f"2x2 FACTORIAL RESULTS")
print(f"{'='*70}")
print(f"\n  {'':>30} | {'Prefix in attn':>16} | {'Prefix NOT in attn':>18}")
print(f"  {'':>30} | {'(redistribution)':>16} | {'(no redistribution)':>18}")
print(f"  {'-'*70}")

d_oracle = cohens_d(bare - oracle)
print(f"  {'RoPE shift ON':>30} | d={d_random:>+.3f} (***) | d={d_blocked:>+.3f} ({sig_blocked})")
print(f"  {'RoPE shift OFF':>30} | d={d_neutral:>+.3f} ({sig_neutral}) | d=+0.000 (baseline)")

print(f"\n  Oracle reference: d={d_oracle:+.3f}")

# Row effect (prefix in attention matters?)
row_effect = d_random - d_blocked
# Column effect (RoPE shift matters?)
col_effect = d_random - d_neutral

print(f"\n  ROW effect (attention redistribution): {row_effect:+.3f}")
print(f"  COLUMN effect (RoPE position shift):  {col_effect:+.3f}")

if abs(col_effect) < 0.05 and row_effect > 0.1:
    print(f"\n  CONCLUSION: Benefit is ENTIRELY from attention redistribution.")
    print(f"  RoPE position shifts contribute NOTHING to the structural mechanism.")
elif abs(col_effect) > abs(row_effect):
    print(f"\n  CONCLUSION: RoPE is the dominant mechanism (unexpected!).")
else:
    print(f"\n  CONCLUSION: Both contribute, but attention redistribution "
          f"is {row_effect/col_effect:.1f}x larger.")
""")


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save results

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Compute all d values
d_oracle = cohens_d(bare - oracle)
d_random = cohens_d(bare - random_std)
d_shifted = cohens_d(bare - shifted)
d_blocked = cohens_d(bare - blocked)
d_neutralized = cohens_d(bare - neutralized)

print(f"\n  {'Condition':<35} {'d vs bare':>10} {'NLL':>10} {'Interpretation':>30}")
print(f"  {'-'*90}")
print(f"  {'oracle_trunc':<35} {d_oracle:>+10.4f} {oracle.mean():>10.4f} {'upper bound':>30}")
print(f"  {'random_trunc (standard)':<35} {d_random:>+10.4f} {random_std.mean():>10.4f} {'attention + RoPE':>30}")
print(f"  {'random_rope_neutralized':<35} {d_neutralized:>+10.4f} {neutralized.mean():>10.4f} {'attention only':>30}")
print(f"  {'shifted_bare':<35} {d_shifted:>+10.4f} {shifted.mean():>10.4f} {'RoPE only':>30}")
print(f"  {'prefix_encoder_blocked':<35} {d_blocked:>+10.4f} {blocked.mean():>10.4f} {'RoPE + invisible prefix':>30}")
print(f"  {'bare':<35} {'baseline':>10} {bare.mean():>10.4f} {'baseline':>30}")

_, p_shift = stats.ttest_1samp(bare - shifted, 0)
_, p_blocked = stats.ttest_1samp(bare - blocked, 0)
_, p_neutral = stats.ttest_1samp(bare - neutralized, 0)
_, p_random = stats.ttest_1samp(bare - random_std, 0)
diff_svn = neutralized - random_std
_, p_svn = stats.ttest_1samp(diff_svn, 0)
d_svn = cohens_d(diff_svn)

rope_contributes = abs(d_shifted) > 0.05 and p_shift < 0.05
attn_contributes = d_random > 0.1
rope_amplifies = abs(d_svn) > 0.05 and p_svn < 0.05

final_results = {
    'experiment': 'exp07_rope_isolation',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
    'position_offset': POSITION_OFFSET,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {
        'bare': {'d': 0.0, 'mean_nll': float(bare.mean())},
        'oracle_trunc': {
            'd': float(d_oracle), 'mean_nll': float(oracle.mean()),
            'p': float(stats.ttest_1samp(bare - oracle, 0)[1]),
        },
        'random_trunc': {
            'd': float(d_random), 'mean_nll': float(random_std.mean()),
            'p': float(p_random),
            'description': 'Standard: attention redistribution + RoPE shift',
        },
        'shifted_bare': {
            'd': float(d_shifted), 'mean_nll': float(shifted.mean()),
            'p': float(p_shift),
            'description': 'Pure absolute RoPE shift, no prefix',
        },
        'prefix_encoder_blocked': {
            'd': float(d_blocked), 'mean_nll': float(blocked.mean()),
            'p': float(p_blocked),
            'description': 'RoPE shift + invisible prefix in encoder',
        },
        'random_rope_neutralized': {
            'd': float(d_neutralized), 'mean_nll': float(neutralized.mean()),
            'p': float(p_neutral),
            'description': 'Attention redistribution only, doc positions preserved',
        },
    },
    'factorial': {
        'row_effect_attention': float(d_random - d_blocked),
        'col_effect_rope': float(d_random - d_neutralized),
        'standard_vs_neutralized_d': float(d_svn),
        'standard_vs_neutralized_p': float(p_svn),
    },
    'conclusion': {
        'rope_contributes': bool(rope_contributes),
        'attention_contributes': bool(attn_contributes),
        'rope_amplifies_attention': bool(rope_amplifies),
        'primary_mechanism': 'attention_redistribution' if not rope_contributes else 'both',
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

print(f"\n{'='*70}")
if not rope_contributes and attn_contributes:
    print("CONCLUSION: RoPE position shifts are RULED OUT.")
    print("The structural benefit comes ENTIRELY from attention redistribution.")
    print("Prepending a prefix adds new attention targets that reorganize")
    print("doc-doc attention patterns. The specific positions don't matter.")
elif rope_contributes:
    print("CONCLUSION: RoPE position shifts DO contribute to the mechanism.")
    print("This needs further investigation.")
else:
    print("CONCLUSION: Neither mechanism shows strong effects (unexpected).")
print(f"{'='*70}")

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
out_path = "07_rope_isolation.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
