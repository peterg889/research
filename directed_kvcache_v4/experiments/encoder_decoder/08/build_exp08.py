#!/usr/bin/env python3
# Build Exp 08 notebook: Decoder Length Control.
#
# Exp 07 found that decoder query tokens absorb 5.5% of answer-token attention
# and the nq→q NLL improvement is d=+0.309. But the _q conditions have LONGER
# decoder sequences (BOS + query + answer) vs _nq (BOS + answer). The "query
# buffer" effect and NLL improvement could be partly a length artifact — any
# extra tokens might absorb attention and change the budget mechanically.
#
# This experiment adds random_q conditions: decoder = [BOS, random_tokens, answer]
# where random_tokens has the SAME length as query_ids per sample. This separates
# the pure length effect from the query-specific semantic effect.
#
# Design: 2x3 factorial {encoder: bare, oracle_trunc} x {decoder: nq, random_q, q}
# with attention probes at the last decoder layer.
# N=500, MS MARCO v1.1.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 08: Decoder Length Control

## Motivation

Exp 07 found that decoder query tokens absorb 5.5% of answer-token attention and
the nq→q NLL improvement is d=+0.309. But the `_q` conditions have **longer decoder
sequences** (`[BOS, query, answer]` vs `[BOS, answer]`). This means:

1. The 5.5% "query buffer" could be a **length artifact** — any extra tokens would
   absorb some attention mechanically
2. The NLL improvement could partly be due to the decoder having more context positions
   (longer causal window) rather than the query's semantic content

## Design: 2×3 Factorial

Add `random_q` conditions where `decoder_input_ids = [BOS, random_tokens, answer]`
with `len(random_tokens) == len(query_ids)` per sample. Random tokens are sampled
uniformly from the vocabulary (avoiding special tokens).

| # | Condition | Encoder input | Cross-attn mask | Decoder input |
|---|-----------|--------------|-----------------|---------------|
| 1 | bare_nq | [document] | all visible | [BOS, answer] |
| 2 | bare_random_q | [document] | all visible | [BOS, random, answer] |
| 3 | bare_q | [document] | all visible | [BOS, query, answer] |
| 4 | oracle_trunc_nq | [query + doc] | doc only | [BOS, answer] |
| 5 | oracle_trunc_random_q | [query + doc] | doc only | [BOS, random, answer] |
| 6 | oracle_trunc_q | [query + doc] | doc only | [BOS, query, answer] |

## Key comparisons

**NLL decomposition of nq→q improvement:**
- `nq → random_q`: Pure length effect (no semantics)
- `random_q → q`: Query-specific semantic effect (length-controlled)
- If length accounts for most of the improvement, `random_q` ≈ `q`

**Attention decomposition:**
- Does random prefix absorb similar attention as query (~5.5%)?
- Does random prefix steal from the same budget (cross-attention)?
- If yes, the "query buffer" is purely mechanical""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup and model loading (EAGER attention for weight extraction)
import os
os.umask(0o000)

import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../../results/exp08")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME} with attn_implementation='eager'...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
    attn_implementation="eager",
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = getattr(model.config, 'decoder_start_token_id', None) or tokenizer.bos_token_id

# Discover decoder layer count
N_DEC_LAYERS = len(model.model.decoder.layers)
# Probe only 3 key layers: first, middle, last (faster than 6)
PROBE_LAYERS = [0, N_DEC_LAYERS // 2, N_DEC_LAYERS - 1]

VOCAB_SIZE = len(tokenizer)

print(f"Exp 08: Decoder Length Control")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Decoder layers: {N_DEC_LAYERS}, Vocab size: {VOCAB_SIZE}")
print(f"Probe layers: {PROBE_LAYERS}")
""")


# ===== Cell 3: Data loading + random token generation =====
code(r"""# Cell 3: Load MS MARCO data + generate matched-length random tokens
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

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

# Count prefix tokens for oracle conditions
def count_prefix_tokens(prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)

# Pre-tokenize queries and generate matched-length random tokens
for i, s in enumerate(samples):
    s['n_pfx_oracle'] = count_prefix_tokens(s['query'], s['passage'])
    q_ids = tokenizer(s['query'], add_special_tokens=False, truncation=True,
                      max_length=512).input_ids
    s['query_ids'] = q_ids
    # Generate random token IDs of the same length as the query
    # Avoid special tokens (IDs 0-99) and the very end of vocab
    rng = np.random.RandomState(SEED + i + 10000)
    s['random_ids'] = rng.randint(100, VOCAB_SIZE - 100, size=len(q_ids)).tolist()

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query tokens: {np.mean([len(s['query_ids']) for s in samples]):.1f}")
print(f"Mean oracle prefix tokens: {np.mean([s['n_pfx_oracle'] for s in samples]):.1f}")

# Show example random vs query tokens
s0 = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query: {s0['query'][:60]}...")
print(f"  Query tokens ({len(s0['query_ids'])}): {s0['query_ids'][:10]}...")
print(f"  Random tokens ({len(s0['random_ids'])}): {s0['random_ids'][:10]}...")
print(f"  Random decoded: {tokenizer.decode(s0['random_ids'][:10])}...")
""")


# ===== Cell 4: Define probing function =====
code(r"""# Cell 4: Define probing function (same as Exp 07 but with fewer layers)

def forward_with_probes(encoder_outputs, cross_attn_mask, decoder_input_ids,
                        answer_start, answer_len, answer_ids_list):
    # Forward pass with attention extraction.
    # Returns (nll, probes_dict) where probes_dict is keyed by layer index.
    dec_len = decoder_input_ids.shape[1]
    n_prefix = answer_start - 1  # 0 for _nq, len(prefix_ids) for _q/_random_q

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )

    # --- NLL ---
    logits = outputs.logits
    answer_logits = logits[0, n_prefix:n_prefix + answer_len, :]
    targets = torch.tensor(answer_ids_list, dtype=torch.long, device=DEVICE)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    nll = -token_log_probs.mean().item()

    # --- Probes ---
    probes = {}
    for layer_idx in PROBE_LAYERS:
        # Self-attention: [1, heads, dec_len, dec_len]
        sa = outputs.decoder_attentions[layer_idx][0].float().mean(dim=0)
        # Cross-attention: [1, heads, dec_len, enc_len]
        ca = outputs.cross_attentions[layer_idx][0].float().mean(dim=0)

        # Extract answer-token rows
        ans_sa = sa[answer_start:answer_start + answer_len]  # [M, dec_len]
        ans_ca = ca[answer_start:answer_start + answer_len]  # [M, enc_len]

        # Self-attention budget decomposition for answer tokens
        self_to_bos = ans_sa[:, 0].mean().item()

        if n_prefix > 0:
            self_to_prefix = ans_sa[:, 1:1 + n_prefix].sum(dim=1).mean().item()
        else:
            self_to_prefix = 0.0

        # Self-attention to answer positions (causal)
        answer_mask = torch.zeros(answer_len, dec_len, device=DEVICE)
        for t in range(answer_len):
            p = answer_start + t
            answer_mask[t, answer_start:p + 1] = 1.0
        self_to_answer = (ans_sa * answer_mask).sum(dim=1).mean().item()

        # Totals
        self_total = ans_sa.sum(dim=1).mean().item()
        cross_total = ans_ca.sum(dim=1).mean().item()

        # Self-attention entropy
        eps = 1e-10
        positions = torch.arange(dec_len, device=DEVICE)
        abs_positions = torch.arange(answer_start, answer_start + answer_len, device=DEVICE)
        causal = (positions.unsqueeze(0) <= abs_positions.unsqueeze(1)).float()
        masked_sa = ans_sa * causal
        sa_clamped = masked_sa.clamp(min=eps)
        self_entropy = -(masked_sa * sa_clamped.log()).sum(dim=1).mean().item()

        # Cross-attention entropy
        ca_clamped = ans_ca.clamp(min=eps)
        cross_entropy = -(ans_ca * ca_clamped.log()).sum(dim=1).mean().item()

        probes[layer_idx] = {
            'sb': round(self_to_bos, 6),
            'sp': round(self_to_prefix, 6),  # 'sp' = self_to_prefix (query or random)
            'sa': round(self_to_answer, 6),
            'st': round(self_total, 6),
            'ct': round(cross_total, 6),
            'se': round(self_entropy, 4),
            'ce': round(cross_entropy, 4),
        }

    del outputs, logits, log_probs
    return nll, probes


print("Probing function defined. Ready for scoring loop.")
""")


# ===== Cell 5: Probing loop =====
code(r"""# Cell 5: Probing loop — 6 conditions x 500 samples
print("=" * 70)
print("PROBING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = ['bare_nq', 'bare_random_q', 'bare_q',
              'oracle_trunc_nq', 'oracle_trunc_random_q', 'oracle_trunc_q']

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            results = ckpt['results']
            # JSON converts int dict keys to strings — convert back
            for r in results:
                for cond in COND_NAMES:
                    key = f'probes_{cond}'
                    if key in r and isinstance(r[key], dict):
                        r[key] = {int(k): v for k, v in r[key].items()}
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples")
    print(f"Probe layers: {PROBE_LAYERS}")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Probing"):
    s = samples[i]
    passage = s['passage']
    answer = s['answer']
    query_ids = s['query_ids']
    random_ids = s['random_ids']

    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        continue

    n_prefix = len(query_ids)  # same for query and random

    result = {
        'query': s['query'][:50],
        'n_prefix_toks': n_prefix,
        'n_answer_toks': len(answer_ids),
    }

    # Build decoder tensors once
    dec_nq = torch.tensor([[BOS_ID] + answer_ids], dtype=torch.long, device=DEVICE)
    dec_random = torch.tensor([[BOS_ID] + random_ids + answer_ids],
                               dtype=torch.long, device=DEVICE)
    dec_q = torch.tensor([[BOS_ID] + query_ids + answer_ids],
                          dtype=torch.long, device=DEVICE)

    # --- Encoder pass 1: bare document ---
    enc_ids_bare = tokenizer(passage, return_tensors="pt",
                             add_special_tokens=True, truncation=True,
                             max_length=2048).input_ids.to(DEVICE)
    enc_len_bare = enc_ids_bare.shape[1]
    enc_mask_bare = torch.ones(1, enc_len_bare, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        enc_out_bare = model.get_encoder()(
            input_ids=enc_ids_bare, attention_mask=enc_mask_bare
        )

    # Condition 1: bare_nq
    nll, probes = forward_with_probes(
        enc_out_bare, enc_mask_bare, dec_nq,
        answer_start=1, answer_len=len(answer_ids), answer_ids_list=answer_ids)
    result['nll_bare_nq'] = nll
    result['probes_bare_nq'] = probes

    # Condition 2: bare_random_q
    nll, probes = forward_with_probes(
        enc_out_bare, enc_mask_bare, dec_random,
        answer_start=1 + n_prefix, answer_len=len(answer_ids),
        answer_ids_list=answer_ids)
    result['nll_bare_random_q'] = nll
    result['probes_bare_random_q'] = probes

    # Condition 3: bare_q
    nll, probes = forward_with_probes(
        enc_out_bare, enc_mask_bare, dec_q,
        answer_start=1 + n_prefix, answer_len=len(answer_ids),
        answer_ids_list=answer_ids)
    result['nll_bare_q'] = nll
    result['probes_bare_q'] = probes

    del enc_out_bare

    # --- Encoder pass 2: oracle (query + document) ---
    enc_text_oracle = s['query'] + "\n" + passage
    enc_ids_oracle = tokenizer(enc_text_oracle, return_tensors="pt",
                               add_special_tokens=True, truncation=True,
                               max_length=2048).input_ids.to(DEVICE)
    enc_len_oracle = enc_ids_oracle.shape[1]
    enc_mask_oracle = torch.ones(1, enc_len_oracle, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        enc_out_oracle = model.get_encoder()(
            input_ids=enc_ids_oracle, attention_mask=enc_mask_oracle
        )

    # Cross-attention mask: hide prefix
    pfx_count = s['n_pfx_oracle']
    cross_mask_trunc = torch.ones(1, enc_len_oracle, device=DEVICE, dtype=torch.long)
    cross_mask_trunc[:, :pfx_count] = 0

    # Condition 4: oracle_trunc_nq
    nll, probes = forward_with_probes(
        enc_out_oracle, cross_mask_trunc, dec_nq,
        answer_start=1, answer_len=len(answer_ids), answer_ids_list=answer_ids)
    result['nll_oracle_trunc_nq'] = nll
    result['probes_oracle_trunc_nq'] = probes

    # Condition 5: oracle_trunc_random_q
    nll, probes = forward_with_probes(
        enc_out_oracle, cross_mask_trunc, dec_random,
        answer_start=1 + n_prefix, answer_len=len(answer_ids),
        answer_ids_list=answer_ids)
    result['nll_oracle_trunc_random_q'] = nll
    result['probes_oracle_trunc_random_q'] = probes

    # Condition 6: oracle_trunc_q
    nll, probes = forward_with_probes(
        enc_out_oracle, cross_mask_trunc, dec_q,
        answer_start=1 + n_prefix, answer_len=len(answer_ids),
        answer_ids_list=answer_ids)
    result['nll_oracle_trunc_q'] = nll
    result['probes_oracle_trunc_q'] = probes

    del enc_out_oracle
    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'probe_layers': PROBE_LAYERS,
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
print(f"\nProbing complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 6: NLL decomposition =====
code(r"""# Cell 6: NLL decomposition — length effect vs query-specific effect
print("=" * 70)
print("NLL DECOMPOSITION: LENGTH vs QUERY SEMANTICS")
print("=" * 70)

nll = {}
for cond in COND_NAMES:
    nll[cond] = np.array([r[f'nll_{cond}'] for r in results])

print(f"\n  {'Condition':<30} {'Mean NLL':>10} {'d vs bare_*':>12} {'sig':>5}")
print(f"  {'-'*62}")

# Bare encoder conditions
print(f"  {'bare_nq':<30} {nll['bare_nq'].mean():>10.4f} {'—':>12} {'—':>5}")

diff_rq = nll['bare_nq'] - nll['bare_random_q']
d_rq = cohens_d(diff_rq)
_, p_rq = stats.ttest_1samp(diff_rq, 0)
sig_rq = '***' if p_rq < 0.001 else '**' if p_rq < 0.01 else '*' if p_rq < 0.05 else 'ns'
print(f"  {'bare_random_q':<30} {nll['bare_random_q'].mean():>10.4f} {d_rq:>+12.3f} {sig_rq:>5}")

diff_q = nll['bare_nq'] - nll['bare_q']
d_q = cohens_d(diff_q)
_, p_q = stats.ttest_1samp(diff_q, 0)
sig_q = '***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'ns'
print(f"  {'bare_q':<30} {nll['bare_q'].mean():>10.4f} {d_q:>+12.3f} {sig_q:>5}")

# Oracle encoder conditions
print(f"\n  {'oracle_trunc_nq':<30} {nll['oracle_trunc_nq'].mean():>10.4f} {'—':>12} {'—':>5}")

diff_o_rq = nll['oracle_trunc_nq'] - nll['oracle_trunc_random_q']
d_o_rq = cohens_d(diff_o_rq)
_, p_o_rq = stats.ttest_1samp(diff_o_rq, 0)
sig_o_rq = '***' if p_o_rq < 0.001 else '**' if p_o_rq < 0.01 else '*' if p_o_rq < 0.05 else 'ns'
print(f"  {'oracle_trunc_random_q':<30} {nll['oracle_trunc_random_q'].mean():>10.4f} {d_o_rq:>+12.3f} {sig_o_rq:>5}")

diff_o_q = nll['oracle_trunc_nq'] - nll['oracle_trunc_q']
d_o_q = cohens_d(diff_o_q)
_, p_o_q = stats.ttest_1samp(diff_o_q, 0)
sig_o_q = '***' if p_o_q < 0.001 else '**' if p_o_q < 0.01 else '*' if p_o_q < 0.05 else 'ns'
print(f"  {'oracle_trunc_q':<30} {nll['oracle_trunc_q'].mean():>10.4f} {d_o_q:>+12.3f} {sig_o_q:>5}")

# Decomposition
print(f"\n{'='*70}")
print(f"DECOMPOSITION OF nq→q IMPROVEMENT")
print(f"{'='*70}")

# Bare encoder
total_bare = d_q
length_bare = d_rq
semantic_bare_diff = nll['bare_random_q'] - nll['bare_q']
d_semantic_bare = cohens_d(semantic_bare_diff)
_, p_sem_bare = stats.ttest_1samp(semantic_bare_diff, 0)
sig_sem_bare = '***' if p_sem_bare < 0.001 else '**' if p_sem_bare < 0.01 else '*' if p_sem_bare < 0.05 else 'ns'

print(f"\n  Bare encoder:")
print(f"    Total (nq → q):        d={total_bare:+.3f}")
print(f"    Length (nq → random_q): d={length_bare:+.3f}")
print(f"    Semantic (random → q):  d={d_semantic_bare:+.3f} ({sig_sem_bare})")
if total_bare > 0:
    print(f"    Length fraction:    {length_bare / total_bare * 100:.0f}%")
    print(f"    Semantic fraction: {d_semantic_bare / total_bare * 100:.0f}%")

# Oracle encoder
total_oracle = d_o_q
length_oracle = d_o_rq
semantic_oracle_diff = nll['oracle_trunc_random_q'] - nll['oracle_trunc_q']
d_semantic_oracle = cohens_d(semantic_oracle_diff)
_, p_sem_oracle = stats.ttest_1samp(semantic_oracle_diff, 0)
sig_sem_oracle = '***' if p_sem_oracle < 0.001 else '**' if p_sem_oracle < 0.01 else '*' if p_sem_oracle < 0.05 else 'ns'

print(f"\n  Oracle encoder:")
print(f"    Total (nq → q):        d={total_oracle:+.3f}")
print(f"    Length (nq → random_q): d={length_oracle:+.3f}")
print(f"    Semantic (random → q):  d={d_semantic_oracle:+.3f} ({sig_sem_oracle})")
if total_oracle > 0:
    print(f"    Length fraction:    {length_oracle / total_oracle * 100:.0f}%")
    print(f"    Semantic fraction: {d_semantic_oracle / total_oracle * 100:.0f}%")

# Does random prefix HURT? (random tokens might confuse the decoder)
print(f"\n  Does random prefix hurt answer prediction?")
print(f"    bare: nq → random_q NLL change: {(nll['bare_random_q'] - nll['bare_nq']).mean():+.4f}")
print(f"    oracle: nq → random_q NLL change: {(nll['oracle_trunc_random_q'] - nll['oracle_trunc_nq']).mean():+.4f}")
print(f"    (Positive = random hurts, Negative = random helps)")
""")


# ===== Cell 7: Attention budget — length artifact test =====
code(r"""# Cell 7: Attention budget — is the 5.5% query buffer a length artifact?
print("=" * 70)
print("ATTENTION BUDGET: LENGTH ARTIFACT TEST")
print("=" * 70)

last_layer = PROBE_LAYERS[-1]
print(f"Layer {last_layer} — mean over {len(results)} samples")

print(f"\n  {'Condition':<30} {'BOS':>8} {'Prefix':>8} {'Answer':>8} "
      f"{'Self':>8} {'Cross':>8} {'Check':>8}")
print(f"  {'-'*82}")

for cond in COND_NAMES:
    sb = np.mean([r[f'probes_{cond}'][last_layer]['sb'] for r in results])
    sp = np.mean([r[f'probes_{cond}'][last_layer]['sp'] for r in results])
    sa = np.mean([r[f'probes_{cond}'][last_layer]['sa'] for r in results])
    st = np.mean([r[f'probes_{cond}'][last_layer]['st'] for r in results])
    ct = np.mean([r[f'probes_{cond}'][last_layer]['ct'] for r in results])
    check = st + ct
    print(f"  {cond:<30} {sb:>8.4f} {sp:>8.4f} {sa:>8.4f} "
          f"{st:>8.4f} {ct:>8.4f} {check:>8.4f}")

# Key comparison: random prefix vs query prefix attention absorption
sp_random_bare = np.array([r['probes_bare_random_q'][last_layer]['sp'] for r in results])
sp_query_bare = np.array([r['probes_bare_q'][last_layer]['sp'] for r in results])
sp_random_orc = np.array([r['probes_oracle_trunc_random_q'][last_layer]['sp'] for r in results])
sp_query_orc = np.array([r['probes_oracle_trunc_q'][last_layer]['sp'] for r in results])

print(f"\n--- Prefix attention absorption (query vs random, layer {last_layer}) ---")
print(f"  Bare encoder:")
print(f"    Random prefix absorbs: {sp_random_bare.mean()*100:.1f}%")
print(f"    Query prefix absorbs:  {sp_query_bare.mean()*100:.1f}%")
diff_sp_bare = sp_query_bare - sp_random_bare
d_sp_bare = cohens_d(diff_sp_bare)
_, p_sp_bare = stats.ttest_1samp(diff_sp_bare, 0)
sig_sp = '***' if p_sp_bare < 0.001 else '**' if p_sp_bare < 0.01 else '*' if p_sp_bare < 0.05 else 'ns'
print(f"    Difference: {diff_sp_bare.mean()*100:+.1f}pp (d={d_sp_bare:+.3f}, {sig_sp})")

print(f"\n  Oracle encoder:")
print(f"    Random prefix absorbs: {sp_random_orc.mean()*100:.1f}%")
print(f"    Query prefix absorbs:  {sp_query_orc.mean()*100:.1f}%")
diff_sp_orc = sp_query_orc - sp_random_orc
d_sp_orc = cohens_d(diff_sp_orc)
_, p_sp_orc = stats.ttest_1samp(diff_sp_orc, 0)
sig_sp_o = '***' if p_sp_orc < 0.001 else '**' if p_sp_orc < 0.01 else '*' if p_sp_orc < 0.05 else 'ns'
print(f"    Difference: {diff_sp_orc.mean()*100:+.1f}pp (d={d_sp_orc:+.3f}, {sig_sp_o})")

# Where does the prefix steal from? (nq vs random_q, to isolate pure length)
print(f"\n--- Where does random prefix steal from? (bare, layer {last_layer}) ---")
sb_nq = np.mean([r['probes_bare_nq'][last_layer]['sb'] for r in results])
sb_rq = np.mean([r['probes_bare_random_q'][last_layer]['sb'] for r in results])
sa_nq = np.mean([r['probes_bare_nq'][last_layer]['sa'] for r in results])
sa_rq = np.mean([r['probes_bare_random_q'][last_layer]['sa'] for r in results])
ct_nq = np.mean([r['probes_bare_nq'][last_layer]['ct'] for r in results])
ct_rq = np.mean([r['probes_bare_random_q'][last_layer]['ct'] for r in results])

print(f"  BOS change:    {sb_rq - sb_nq:+.4f}")
print(f"  Answer change: {sa_rq - sa_nq:+.4f}")
print(f"  Cross change:  {ct_rq - ct_nq:+.4f}")
print(f"  (Compare to query: BOS {np.mean([r['probes_bare_q'][last_layer]['sb'] for r in results]) - sb_nq:+.4f}, "
      f"Ans {np.mean([r['probes_bare_q'][last_layer]['sa'] for r in results]) - sa_nq:+.4f}, "
      f"Cross {np.mean([r['probes_bare_q'][last_layer]['ct'] for r in results]) - ct_nq:+.4f})")
""")


# ===== Cell 8: Entropy analysis =====
code(r"""# Cell 8: Entropy analysis — does random prefix change attention patterns?
print("=" * 70)
print("ENTROPY ANALYSIS")
print("=" * 70)

last_layer = PROBE_LAYERS[-1]

print(f"\n--- Self-attention entropy (layer {last_layer}) ---")
for cond in COND_NAMES:
    se = np.mean([r[f'probes_{cond}'][last_layer]['se'] for r in results])
    print(f"  {cond:<30} {se:.4f}")

se_nq = np.array([r['probes_bare_nq'][last_layer]['se'] for r in results])
se_rq = np.array([r['probes_bare_random_q'][last_layer]['se'] for r in results])
se_q = np.array([r['probes_bare_q'][last_layer]['se'] for r in results])

d_se_len = cohens_d(se_rq - se_nq)
d_se_sem = cohens_d(se_q - se_rq)
d_se_tot = cohens_d(se_q - se_nq)
print(f"\n  Self-entropy change (bare):")
print(f"    nq → random_q (length):   d={d_se_len:+.3f}")
print(f"    random_q → q (semantic):   d={d_se_sem:+.3f}")
print(f"    nq → q (total):            d={d_se_tot:+.3f}")

print(f"\n--- Cross-attention entropy (layer {last_layer}) ---")
for cond in COND_NAMES:
    ce = np.mean([r[f'probes_{cond}'][last_layer]['ce'] for r in results])
    print(f"  {cond:<30} {ce:.4f}")

ce_nq = np.array([r['probes_bare_nq'][last_layer]['ce'] for r in results])
ce_rq = np.array([r['probes_bare_random_q'][last_layer]['ce'] for r in results])
ce_q = np.array([r['probes_bare_q'][last_layer]['ce'] for r in results])

d_ce_len = cohens_d(ce_rq - ce_nq)
d_ce_sem = cohens_d(ce_q - ce_rq)
d_ce_tot = cohens_d(ce_q - ce_nq)
print(f"\n  Cross-entropy change (bare):")
print(f"    nq → random_q (length):   d={d_ce_len:+.3f}")
print(f"    random_q → q (semantic):   d={d_ce_sem:+.3f}")
print(f"    nq → q (total):            d={d_ce_tot:+.3f}")
""")


# ===== Cell 9: Encoder prefix interaction with length control =====
code(r"""# Cell 9: Does the encoder prefix interaction survive length control?
print("=" * 70)
print("ENCODER PREFIX EFFECT — WITH LENGTH CONTROL")
print("=" * 70)

# The key question: In Exp 07, the 35% redundancy was between encoder prefix
# and decoder query. But how much of that is just decoder LENGTH?

print(f"\n--- Encoder prefix effect across decoder conditions ---")
enc_nq = nll['bare_nq'] - nll['oracle_trunc_nq']
enc_rq = nll['bare_random_q'] - nll['oracle_trunc_random_q']
enc_q = nll['bare_q'] - nll['oracle_trunc_q']

d_enc_nq = cohens_d(enc_nq)
d_enc_rq = cohens_d(enc_rq)
d_enc_q = cohens_d(enc_q)

_, p_enc_nq = stats.ttest_1samp(enc_nq, 0)
_, p_enc_rq = stats.ttest_1samp(enc_rq, 0)
_, p_enc_q = stats.ttest_1samp(enc_q, 0)

print(f"  With nq decoder:        d={d_enc_nq:+.3f} (p={p_enc_nq:.2e})")
print(f"  With random_q decoder:  d={d_enc_rq:+.3f} (p={p_enc_rq:.2e})")
print(f"  With query decoder:     d={d_enc_q:+.3f} (p={p_enc_q:.2e})")

# Interactions
int_length = enc_nq - enc_rq  # nq vs random_q: pure length interaction
int_semantic = enc_rq - enc_q  # random_q vs q: semantic interaction
int_total = enc_nq - enc_q    # total (same as Exp 07)

d_int_length = cohens_d(int_length)
d_int_semantic = cohens_d(int_semantic)
d_int_total = cohens_d(int_total)

_, p_il = stats.ttest_1samp(int_length, 0)
_, p_is = stats.ttest_1samp(int_semantic, 0)
_, p_it = stats.ttest_1samp(int_total, 0)

sig_il = '***' if p_il < 0.001 else '**' if p_il < 0.01 else '*' if p_il < 0.05 else 'ns'
sig_is = '***' if p_is < 0.001 else '**' if p_is < 0.01 else '*' if p_is < 0.05 else 'ns'
sig_it = '***' if p_it < 0.001 else '**' if p_it < 0.01 else '*' if p_it < 0.05 else 'ns'

print(f"\n--- Interaction decomposition ---")
print(f"  Total interaction (nq vs q):        d={d_int_total:+.3f} ({sig_it})")
print(f"  Length interaction (nq vs random_q): d={d_int_length:+.3f} ({sig_il})")
print(f"  Semantic interaction (random vs q):  d={d_int_semantic:+.3f} ({sig_is})")

if d_int_total != 0:
    print(f"\n  Of the total encoder prefix reduction:")
    print(f"    Due to decoder LENGTH:    {d_int_length / d_int_total * 100:.0f}%")
    print(f"    Due to query SEMANTICS:   {d_int_semantic / d_int_total * 100:.0f}%")

# Exp 07 replication check
print(f"\n--- Exp 07 replication ---")
print(f"  Exp 07: encoder prefix d_nq={0.366:+.3f}, d_q={0.238:+.3f}, interaction={0.316:+.3f}")
print(f"  Exp 08: encoder prefix d_nq={d_enc_nq:+.3f}, d_q={d_enc_q:+.3f}, interaction={d_int_total:+.3f}")
""")


# ===== Cell 10: Summary + save =====
code(r"""# Cell 10: Summary and save
print("=" * 70)
print("SUMMARY — Exp 08: Decoder Length Control")
print("=" * 70)

last_layer = PROBE_LAYERS[-1]

# NLL decomposition
total_bare = cohens_d(nll['bare_nq'] - nll['bare_q'])
length_bare = cohens_d(nll['bare_nq'] - nll['bare_random_q'])
semantic_bare = cohens_d(nll['bare_random_q'] - nll['bare_q'])

total_oracle = cohens_d(nll['oracle_trunc_nq'] - nll['oracle_trunc_q'])
length_oracle = cohens_d(nll['oracle_trunc_nq'] - nll['oracle_trunc_random_q'])
semantic_oracle = cohens_d(nll['oracle_trunc_random_q'] - nll['oracle_trunc_q'])

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)}")

print(f"\n--- NLL Decomposition (nq → q) ---")
print(f"  Bare encoder:")
print(f"    Total:    d={total_bare:+.3f}")
print(f"    Length:   d={length_bare:+.3f} ({length_bare/total_bare*100:.0f}%)" if total_bare else "")
print(f"    Semantic: d={semantic_bare:+.3f} ({semantic_bare/total_bare*100:.0f}%)" if total_bare else "")
print(f"  Oracle encoder:")
print(f"    Total:    d={total_oracle:+.3f}")
print(f"    Length:   d={length_oracle:+.3f} ({length_oracle/total_oracle*100:.0f}%)" if total_oracle else "")
print(f"    Semantic: d={semantic_oracle:+.3f} ({semantic_oracle/total_oracle*100:.0f}%)" if total_oracle else "")

# Attention
sp_random = np.mean([r['probes_bare_random_q'][last_layer]['sp'] for r in results])
sp_query = np.mean([r['probes_bare_q'][last_layer]['sp'] for r in results])

print(f"\n--- Attention (layer {last_layer}) ---")
print(f"  Random prefix absorbs: {sp_random*100:.1f}% of answer attention")
print(f"  Query prefix absorbs:  {sp_query*100:.1f}% of answer attention")
if sp_query > 0:
    print(f"  Length accounts for:   {sp_random/sp_query*100:.0f}% of prefix buffer")

# Interaction decomposition
enc_nq = nll['bare_nq'] - nll['oracle_trunc_nq']
enc_rq = nll['bare_random_q'] - nll['oracle_trunc_random_q']
enc_q = nll['bare_q'] - nll['oracle_trunc_q']
d_enc_nq = cohens_d(enc_nq)
d_enc_q = cohens_d(enc_q)
int_total = enc_nq - enc_q
int_length = enc_nq - enc_rq
int_semantic = enc_rq - enc_q
d_int_total = cohens_d(int_total)
d_int_length = cohens_d(int_length)
d_int_semantic = cohens_d(int_semantic)

print(f"\n--- Encoder prefix × decoder interaction ---")
print(f"  Total redundancy (Exp 07 replication): {(1-d_enc_q/d_enc_nq)*100:.0f}%")
if d_int_total != 0:
    print(f"  Due to length: {d_int_length/d_int_total*100:.0f}%")
    print(f"  Due to semantics: {d_int_semantic/d_int_total*100:.0f}%")

# Verdict
print(f"\n--- Verdict ---")
if abs(length_bare) < 0.05:
    print(f"  Length effect is NEGLIGIBLE (d={length_bare:+.3f})")
    print(f"  The nq→q improvement is almost entirely query-semantic.")
elif abs(length_bare) > abs(semantic_bare):
    print(f"  Length effect DOMINATES (d={length_bare:+.3f} vs semantic d={semantic_bare:+.3f})")
    print(f"  The Exp 07 'query buffer' finding is largely a length artifact.")
else:
    print(f"  Both effects contribute: length d={length_bare:+.3f}, semantic d={semantic_bare:+.3f}")
    print(f"  Query provides genuine semantic value beyond mere length.")

# Save results
final_results = {
    'experiment': 'v4_exp08_decoder_length_control',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'probe_layers': PROBE_LAYERS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'nll': {cond: float(nll[cond].mean()) for cond in COND_NAMES},
    'nll_decomposition': {
        'bare': {
            'total': float(total_bare),
            'length': float(length_bare),
            'semantic': float(semantic_bare),
        },
        'oracle': {
            'total': float(total_oracle),
            'length': float(length_oracle),
            'semantic': float(semantic_oracle),
        },
    },
    'prefix_attention': {
        'random_bare': float(sp_random),
        'query_bare': float(sp_query),
    },
    'interaction_decomposition': {
        'd_int_total': float(d_int_total),
        'd_int_length': float(d_int_length),
        'd_int_semantic': float(d_int_semantic),
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
out_path = "experiments/encoder_decoder/08/08_decoder_length_control.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
