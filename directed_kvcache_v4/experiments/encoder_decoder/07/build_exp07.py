#!/usr/bin/env python3
# Build Exp 07 notebook: Decoder Attention Probing.
#
# The v3→v4 structural collapse (85%→35%) happened when we put the query in
# the decoder. Hypothesis: the query tokens in the decoder act as attention
# buffers for the decoder's own self-attention — the SAME mechanism as the
# encoder prefix, but on the decoder side.
#
# If true, the decoder's query provides its own structural redistribution
# through self-attention, making the encoder prefix's structural role
# partially redundant. This would explain WHY the structural fraction
# collapsed.
#
# Design: 2x2 factorial {encoder: bare, oracle_trunc} x {decoder: nq, q}
# with attention weight extraction from all 34 decoder layers.
# N=500, MS MARCO v1.1.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 07: Decoder Attention Probing

## Hypothesis

The v3-to-v4 structural collapse (85% → 35%) occurred when we gave the decoder
the query as input. **Hypothesis: the query tokens act as attention buffers in the
decoder's self-attention** — the same mechanism that the encoder prefix provides
in the encoder's bidirectional attention.

In the decoder, `[BOS, query_tokens, answer_tokens]` is processed with causal
self-attention + merged cross-attention to encoder representations. The decoder's
BOS token likely acts as an attention sink (like the encoder's BOS). The query
tokens may absorb attention from answer tokens, redistributing the answer-token
self-attention budget — exactly the "attention buffer" mechanism we identified
in the encoder (v3 Exp 3E).

If this is correct:
1. The decoder's BOS should be a massive attention sink (like encoder BOS)
2. Query tokens should absorb significant attention from answer tokens
3. This absorption should come at the expense of answer-answer self-attention
4. The encoder prefix's contribution to cross-attention redistribution should
   shrink when the decoder already has query buffers (the interaction effect)

## Design: 2×2 Factorial

| # | Condition | Encoder input | Cross-attn mask | Decoder input |
|---|-----------|--------------|-----------------|---------------|
| 1 | bare_nq | [document] | all visible | [BOS, answer] |
| 2 | bare_q | [document] | all visible | [BOS, query, answer] |
| 3 | oracle_trunc_nq | [query + document] | doc only | [BOS, answer] |
| 4 | oracle_trunc_q | [query + document] | doc only | [BOS, query, answer] |

**Key comparisons:**
- **(2) vs (1)**: Decoder query buffer effect (no encoder prefix)
- **(4) vs (3)**: Decoder query buffer effect (with encoder prefix)
- **(3) vs (1)**: Encoder prefix effect (no decoder query)
- **(4) vs (2)**: Encoder prefix effect (with decoder query)
- **Interaction**: Does having a decoder query reduce the encoder prefix's effect?

## Probes (per decoder layer, per condition)

**Self-attention budget** (from `decoder_attentions`):
- `self_to_bos`: Answer tokens' attention to decoder BOS
- `self_to_query`: Answer tokens' attention to query positions (0 for _nq)
- `self_to_answer`: Answer tokens' attention to other answer positions
- `self_entropy`: Entropy of answer-token self-attention distribution

**Cross-attention budget** (from `cross_attentions`):
- `cross_total`: Total cross-attention mass per answer token
- `cross_entropy`: Entropy of cross-attention distribution over encoder positions

**Budget check**: `self_total + cross_total = 1.0` (merged softmax)""")


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

RESULTS_DIR = Path("../../../results/exp07")
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
# Probe 6 representative layers (evenly spaced)
PROBE_LAYERS = [0, N_DEC_LAYERS // 6, N_DEC_LAYERS // 3,
                N_DEC_LAYERS // 2, 2 * N_DEC_LAYERS // 3, N_DEC_LAYERS - 1]

print(f"Exp 07: Decoder Attention Probing")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Decoder layers: {N_DEC_LAYERS}")
print(f"Probe layers: {PROBE_LAYERS}")
""")


# ===== Cell 3: Data loading =====
code(r"""# Cell 3: Load MS MARCO data (same pipeline as Exp 01-06)
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

for s in samples:
    s['n_pfx_oracle'] = count_prefix_tokens(s['query'], s['passage'])

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean oracle prefix tokens: {np.mean([s['n_pfx_oracle'] for s in samples]):.1f}")
""")


# ===== Cell 4: Verify attention structure + define probing function =====
code(r"""# Cell 4: Verify attention output structure and define probing function

# Test forward pass to verify shapes
print("Verifying attention output structure...")
s0 = samples[0]

# Encode bare document
enc_ids = tokenizer(s0['passage'], return_tensors="pt",
                    add_special_tokens=True, truncation=True,
                    max_length=2048).input_ids.to(DEVICE)
enc_mask = torch.ones(1, enc_ids.shape[1], device=DEVICE, dtype=torch.long)
with torch.no_grad():
    encoder_outputs = model.get_encoder()(input_ids=enc_ids, attention_mask=enc_mask)

# Decoder with query prefix
query_ids = tokenizer(s0['query'], add_special_tokens=False, truncation=True,
                      max_length=512).input_ids
answer_ids = tokenizer(s0['answer'], add_special_tokens=False, truncation=True,
                       max_length=256).input_ids
dec_ids = [BOS_ID] + query_ids + answer_ids
dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)

with torch.no_grad():
    outputs = model(
        encoder_outputs=encoder_outputs,
        attention_mask=enc_mask,
        decoder_input_ids=dec_tensor,
        output_attentions=True,
    )

# Check what we got
dec_len = len(dec_ids)
enc_len = enc_ids.shape[1]
print(f"  Decoder seq len: {dec_len} (1 BOS + {len(query_ids)} query + {len(answer_ids)} answer)")
print(f"  Encoder seq len: {enc_len}")

if outputs.decoder_attentions is not None:
    print(f"  decoder_attentions: {len(outputs.decoder_attentions)} layers")
    print(f"    Shape per layer: {outputs.decoder_attentions[0].shape}")
    # Expected: [1, heads, dec_len, dec_len]
else:
    print("  WARNING: decoder_attentions is None!")

if outputs.cross_attentions is not None:
    print(f"  cross_attentions: {len(outputs.cross_attentions)} layers")
    print(f"    Shape per layer: {outputs.cross_attentions[0].shape}")
    # Expected: [1, heads, dec_len, enc_len]
else:
    print("  WARNING: cross_attentions is None!")

# Verify merged softmax: self + cross should sum to 1.0
sa = outputs.decoder_attentions[0][0].float().mean(dim=0)  # [dec_len, dec_len]
ca = outputs.cross_attentions[0][0].float().mean(dim=0)    # [dec_len, enc_len]
budget_sum = sa.sum(dim=1) + ca.sum(dim=1)  # [dec_len]
print(f"\n  Budget check (self + cross per position):")
print(f"    Min: {budget_sum.min().item():.6f}")
print(f"    Max: {budget_sum.max().item():.6f}")
print(f"    Mean: {budget_sum.mean().item():.6f}")
print(f"    (Should be ~1.0)")

del outputs, encoder_outputs
gc.collect()
torch.cuda.empty_cache()


# === Probing function ===
def forward_with_probes(encoder_outputs, cross_attn_mask, decoder_input_ids,
                        answer_start, answer_len, answer_ids_list):
    # Forward pass with attention extraction.
    # Returns (nll, probes_dict) where probes_dict is keyed by layer index.
    dec_len = decoder_input_ids.shape[1]
    n_query = answer_start - 1  # 0 for _nq, len(query_ids) for _q

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )

    # --- NLL ---
    logits = outputs.logits
    answer_logits = logits[0, n_query:n_query + answer_len, :]
    targets = torch.tensor(answer_ids_list, dtype=torch.long, device=DEVICE)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    nll = -token_log_probs.mean().item()

    # --- Probes ---
    probes = {}
    for layer_idx in PROBE_LAYERS:
        # Self-attention: [1, heads, dec_len, dec_len]
        sa = outputs.decoder_attentions[layer_idx][0].float().mean(dim=0)  # [dec_len, dec_len]
        # Cross-attention: [1, heads, dec_len, enc_len]
        ca = outputs.cross_attentions[layer_idx][0].float().mean(dim=0)  # [dec_len, enc_len]

        # Extract answer-token rows
        ans_sa = sa[answer_start:answer_start + answer_len]  # [M, dec_len]
        ans_ca = ca[answer_start:answer_start + answer_len]  # [M, enc_len]

        # Self-attention budget decomposition for answer tokens
        self_to_bos = ans_sa[:, 0].mean().item()

        if n_query > 0:
            self_to_query = ans_sa[:, 1:1 + n_query].sum(dim=1).mean().item()
        else:
            self_to_query = 0.0

        # Self-attention to answer positions (including self)
        # For answer token t at absolute position p=answer_start+t,
        # attend to positions answer_start..p (causal)
        answer_mask = torch.zeros(answer_len, dec_len, device=DEVICE)
        for t in range(answer_len):
            p = answer_start + t
            answer_mask[t, answer_start:p + 1] = 1.0
        self_to_answer = (ans_sa * answer_mask).sum(dim=1).mean().item()

        # Totals
        self_total = ans_sa.sum(dim=1).mean().item()
        cross_total = ans_ca.sum(dim=1).mean().item()

        # Self-attention entropy (over causal positions 0..p for each answer token)
        eps = 1e-10
        # Build per-token causal mask
        positions = torch.arange(dec_len, device=DEVICE)
        abs_positions = torch.arange(answer_start, answer_start + answer_len, device=DEVICE)
        causal = (positions.unsqueeze(0) <= abs_positions.unsqueeze(1)).float()  # [M, dec_len]
        masked_sa = ans_sa * causal  # zero out non-causal
        sa_clamped = masked_sa.clamp(min=eps)
        self_entropy = -(masked_sa * sa_clamped.log()).sum(dim=1).mean().item()

        # Cross-attention entropy (over all encoder positions)
        ca_clamped = ans_ca.clamp(min=eps)
        cross_entropy = -(ans_ca * ca_clamped.log()).sum(dim=1).mean().item()

        probes[layer_idx] = {
            'sb': round(self_to_bos, 6),
            'sq': round(self_to_query, 6),
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
code(r"""# Cell 5: Probing loop — 4 conditions x 500 samples
print("=" * 70)
print("PROBING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']

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
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        continue

    result = {
        'query': query[:50],
        'n_query_toks': len(query_ids),
        'n_answer_toks': len(answer_ids),
    }

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

    # Condition 1: bare_nq — decoder=[BOS, answer]
    dec_nq = torch.tensor([[BOS_ID] + answer_ids], dtype=torch.long, device=DEVICE)
    nll, probes = forward_with_probes(
        enc_out_bare, enc_mask_bare, dec_nq,
        answer_start=1, answer_len=len(answer_ids), answer_ids_list=answer_ids)
    result['nll_bare_nq'] = nll
    result['probes_bare_nq'] = probes

    # Condition 2: bare_q — decoder=[BOS, query, answer]
    dec_q = torch.tensor([[BOS_ID] + query_ids + answer_ids],
                         dtype=torch.long, device=DEVICE)
    nll, probes = forward_with_probes(
        enc_out_bare, enc_mask_bare, dec_q,
        answer_start=1 + len(query_ids), answer_len=len(answer_ids),
        answer_ids_list=answer_ids)
    result['nll_bare_q'] = nll
    result['probes_bare_q'] = probes

    del enc_out_bare

    # --- Encoder pass 2: oracle (query + document) ---
    enc_text_oracle = query + "\n" + passage
    enc_ids_oracle = tokenizer(enc_text_oracle, return_tensors="pt",
                               add_special_tokens=True, truncation=True,
                               max_length=2048).input_ids.to(DEVICE)
    enc_len_oracle = enc_ids_oracle.shape[1]
    enc_mask_oracle = torch.ones(1, enc_len_oracle, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        enc_out_oracle = model.get_encoder()(
            input_ids=enc_ids_oracle, attention_mask=enc_mask_oracle
        )

    # Cross-attention mask: hide prefix (query + BOS)
    pfx_count = s['n_pfx_oracle']
    cross_mask_trunc = torch.ones(1, enc_len_oracle, device=DEVICE, dtype=torch.long)
    cross_mask_trunc[:, :pfx_count] = 0

    # Condition 3: oracle_trunc_nq — decoder=[BOS, answer]
    nll, probes = forward_with_probes(
        enc_out_oracle, cross_mask_trunc, dec_nq,
        answer_start=1, answer_len=len(answer_ids), answer_ids_list=answer_ids)
    result['nll_oracle_trunc_nq'] = nll
    result['probes_oracle_trunc_nq'] = probes

    # Condition 4: oracle_trunc_q — decoder=[BOS, query, answer]
    nll, probes = forward_with_probes(
        enc_out_oracle, cross_mask_trunc, dec_q,
        answer_start=1 + len(query_ids), answer_len=len(answer_ids),
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


# ===== Cell 6: NLL calibration + budget overview =====
code(r"""# Cell 6: NLL calibration and attention budget overview
print("=" * 70)
print("NLL CALIBRATION")
print("=" * 70)

nll_bare_nq = np.array([r['nll_bare_nq'] for r in results])
nll_bare_q = np.array([r['nll_bare_q'] for r in results])
nll_oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])
nll_oracle_q = np.array([r['nll_oracle_trunc_q'] for r in results])

# Expected from Exp 01: oracle_trunc_q vs bare_q: d~+0.228, oracle_trunc_nq vs bare_nq: d~+0.376
print(f"\n  {'Condition':<25} {'Mean NLL':>10} {'d vs bare':>10} {'sig':>5}")
print(f"  {'-'*55}")

for name, nlls, baseline, bl_name in [
    ('bare_nq', nll_bare_nq, None, None),
    ('oracle_trunc_nq', nll_oracle_nq, nll_bare_nq, 'bare_nq'),
    ('bare_q', nll_bare_q, None, None),
    ('oracle_trunc_q', nll_oracle_q, nll_bare_q, 'bare_q'),
]:
    if baseline is None:
        print(f"  {name:<25} {nlls.mean():>10.4f} {'--':>10} {'--':>5}")
    else:
        diff = baseline - nlls
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {name:<25} {nlls.mean():>10.4f} {d:>+10.3f} {sig:>5}")

# Query-in-decoder effect on bare NLL
diff_q = nll_bare_nq - nll_bare_q
d_q = cohens_d(diff_q)
_, p_q = stats.ttest_1samp(diff_q, 0)
print(f"\n  Query in decoder effect (bare_nq → bare_q): d={d_q:+.3f} (p={p_q:.2e})")
print(f"  (Expected: large positive — query helps predict answer)")

print(f"\n{'='*70}")
print("ATTENTION BUDGET OVERVIEW (last probe layer)")
print("=" * 70)

last_layer = PROBE_LAYERS[-1]
print(f"Layer {last_layer} — mean over {len(results)} samples, averaged over heads and answer tokens")
print(f"\n  {'Condition':<25} {'self_bos':>10} {'self_query':>10} {'self_ans':>10} "
      f"{'self_tot':>10} {'cross_tot':>10} {'check':>8}")
print(f"  {'-'*83}")

for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    sb = np.mean([r[f'probes_{cond}'][last_layer]['sb'] for r in results])
    sq = np.mean([r[f'probes_{cond}'][last_layer]['sq'] for r in results])
    sa = np.mean([r[f'probes_{cond}'][last_layer]['sa'] for r in results])
    st = np.mean([r[f'probes_{cond}'][last_layer]['st'] for r in results])
    ct = np.mean([r[f'probes_{cond}'][last_layer]['ct'] for r in results])
    check = st + ct
    print(f"  {cond:<25} {sb:>10.4f} {sq:>10.4f} {sa:>10.4f} "
          f"{st:>10.4f} {ct:>10.4f} {check:>8.4f}")

print(f"\n  Budget check: self_total + cross_total should = 1.0000")
""")


# ===== Cell 7: Decoder BOS sink + query buffer =====
code(r"""# Cell 7: Probe A — Decoder BOS sink and query as attention buffer
print("=" * 70)
print("PROBE A: DECODER BOS SINK AND QUERY BUFFER")
print("=" * 70)

# Layer-by-layer trajectory of BOS sink mass
print(f"\n--- BOS attention sink (answer tokens → decoder BOS) ---")
print(f"\n  {'Layer':>6} {'bare_nq':>10} {'bare_q':>10} {'orc_nq':>10} {'orc_q':>10}  "
      f"{'q effect':>10} {'p':>10}")
print(f"  {'-'*73}")

for layer in PROBE_LAYERS:
    vals = {}
    for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
        vals[cond] = np.array([r[f'probes_{cond}'][layer]['sb'] for r in results])

    # Query buffer effect on BOS: does adding query reduce BOS attention?
    diff_bos = vals['bare_nq'] - vals['bare_q']
    d_bos = cohens_d(diff_bos)
    _, p_bos = stats.ttest_1samp(diff_bos, 0)
    sig = '***' if p_bos < 0.001 else '**' if p_bos < 0.01 else '*' if p_bos < 0.05 else 'ns'

    print(f"  {layer:>6} {vals['bare_nq'].mean():>10.4f} {vals['bare_q'].mean():>10.4f} "
          f"{vals['oracle_trunc_nq'].mean():>10.4f} {vals['oracle_trunc_q'].mean():>10.4f}  "
          f"{d_bos:>+10.3f} {p_bos:>10.2e} {sig}")

# Query buffer mass: how much attention do answer tokens give to query positions?
print(f"\n--- Query as attention buffer (answer tokens → query positions) ---")
print(f"  (Only nonzero for _q conditions)")
print(f"\n  {'Layer':>6} {'bare_q':>10} {'orc_q':>10} {'diff':>10}")
print(f"  {'-'*40}")

for layer in PROBE_LAYERS:
    sq_bare = np.array([r['probes_bare_q'][layer]['sq'] for r in results])
    sq_orc = np.array([r['probes_oracle_trunc_q'][layer]['sq'] for r in results])
    print(f"  {layer:>6} {sq_bare.mean():>10.4f} {sq_orc.mean():>10.4f} "
          f"{sq_orc.mean() - sq_bare.mean():>+10.4f}")

# Where does the query buffer steal attention FROM?
print(f"\n--- Where does query attention come from? ---")
print(f"  Compare bare_nq vs bare_q (no encoder prefix)")
print(f"\n  {'Layer':>6} {'BOS change':>12} {'Answer chg':>12} {'Cross chg':>12}")
print(f"  {'-'*50}")

for layer in PROBE_LAYERS:
    sb_nq = np.mean([r['probes_bare_nq'][layer]['sb'] for r in results])
    sb_q = np.mean([r['probes_bare_q'][layer]['sb'] for r in results])
    sa_nq = np.mean([r['probes_bare_nq'][layer]['sa'] for r in results])
    sa_q = np.mean([r['probes_bare_q'][layer]['sa'] for r in results])
    ct_nq = np.mean([r['probes_bare_nq'][layer]['ct'] for r in results])
    ct_q = np.mean([r['probes_bare_q'][layer]['ct'] for r in results])

    print(f"  {layer:>6} {sb_q - sb_nq:>+12.4f} {sa_q - sa_nq:>+12.4f} {ct_q - ct_nq:>+12.4f}")

print(f"\n  (Negative = query steals FROM that budget. Positive = that budget grows.)")
print(f"  Query buffer mass at last layer:")
sq_last = np.mean([r['probes_bare_q'][PROBE_LAYERS[-1]]['sq'] for r in results])
print(f"  = {sq_last:.4f} ({sq_last*100:.1f}% of total attention budget)")
""")


# ===== Cell 8: Self vs cross budget + cross-attention entropy =====
code(r"""# Cell 8: Probe B — Self vs cross allocation and cross-attention redistribution
print("=" * 70)
print("PROBE B: SELF VS CROSS ALLOCATION")
print("=" * 70)

# How does the self/cross split change across conditions?
print(f"\n--- Cross-attention total mass (answer tokens → encoder) ---")
print(f"\n  {'Layer':>6} {'bare_nq':>10} {'bare_q':>10} {'orc_nq':>10} {'orc_q':>10}")
print(f"  {'-'*50}")

for layer in PROBE_LAYERS:
    ct = {}
    for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
        ct[cond] = np.mean([r[f'probes_{cond}'][layer]['ct'] for r in results])
    print(f"  {layer:>6} {ct['bare_nq']:>10.4f} {ct['bare_q']:>10.4f} "
          f"{ct['oracle_trunc_nq']:>10.4f} {ct['oracle_trunc_q']:>10.4f}")

# Does encoder prefix change cross-attention mass?
print(f"\n--- Encoder prefix effect on cross-attention mass ---")
print(f"  (oracle_trunc vs bare, for each decoder condition)")
print(f"\n  {'Layer':>6} {'nq: orc-bare':>14} {'q: orc-bare':>14}")
print(f"  {'-'*38}")

for layer in PROBE_LAYERS:
    ct_bare_nq = np.array([r['probes_bare_nq'][layer]['ct'] for r in results])
    ct_orc_nq = np.array([r['probes_oracle_trunc_nq'][layer]['ct'] for r in results])
    ct_bare_q = np.array([r['probes_bare_q'][layer]['ct'] for r in results])
    ct_orc_q = np.array([r['probes_oracle_trunc_q'][layer]['ct'] for r in results])
    print(f"  {layer:>6} {(ct_orc_nq - ct_bare_nq).mean():>+14.4f} "
          f"{(ct_orc_q - ct_bare_q).mean():>+14.4f}")

print(f"\n  (Positive = encoder prefix increases cross-attention mass)")

# Cross-attention entropy
print(f"\n--- Cross-attention entropy (answer → encoder) ---")
print(f"\n  {'Layer':>6} {'bare_nq':>10} {'bare_q':>10} {'orc_nq':>10} {'orc_q':>10}")
print(f"  {'-'*50}")

for layer in PROBE_LAYERS:
    ce = {}
    for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
        ce[cond] = np.mean([r[f'probes_{cond}'][layer]['ce'] for r in results])
    print(f"  {layer:>6} {ce['bare_nq']:>10.4f} {ce['bare_q']:>10.4f} "
          f"{ce['oracle_trunc_nq']:>10.4f} {ce['oracle_trunc_q']:>10.4f}")

# Self-attention entropy
print(f"\n--- Self-attention entropy (answer → self positions) ---")
print(f"\n  {'Layer':>6} {'bare_nq':>10} {'bare_q':>10} {'orc_nq':>10} {'orc_q':>10}")
print(f"  {'-'*50}")

for layer in PROBE_LAYERS:
    se = {}
    for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
        se[cond] = np.mean([r[f'probes_{cond}'][layer]['se'] for r in results])
    print(f"  {layer:>6} {se['bare_nq']:>10.4f} {se['bare_q']:>10.4f} "
          f"{se['oracle_trunc_nq']:>10.4f} {se['oracle_trunc_q']:>10.4f}")
""")


# ===== Cell 9: The 2x2 interaction test =====
code(r"""# Cell 9: The 2x2 interaction test — does decoder query reduce encoder prefix effect?
print("=" * 70)
print("THE 2x2 INTERACTION TEST")
print("=" * 70)

# For each probe metric at the last layer, compute the 2x2 decomposition
last_layer = PROBE_LAYERS[-1]

print(f"\nLayer {last_layer} — 2x2 factorial decomposition")
print(f"  Factors: Encoder prefix (bare vs oracle_trunc) x Decoder query (nq vs q)")

# NLL interaction
print(f"\n--- NLL ---")
nll = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    nll[cond] = np.array([r[f'nll_{cond}'] for r in results])

enc_effect_nq = nll['bare_nq'] - nll['oracle_trunc_nq']  # positive = prefix helps
enc_effect_q = nll['bare_q'] - nll['oracle_trunc_q']
dec_effect_bare = nll['bare_nq'] - nll['bare_q']  # positive = query helps
dec_effect_oracle = nll['oracle_trunc_nq'] - nll['oracle_trunc_q']

d_enc_nq = cohens_d(enc_effect_nq)
d_enc_q = cohens_d(enc_effect_q)
d_dec_bare = cohens_d(dec_effect_bare)
d_dec_oracle = cohens_d(dec_effect_oracle)

print(f"\n  {'':>24} {'No query':>12} {'With query':>12} {'Difference':>12}")
print(f"  {'Bare encoder':<24} {nll['bare_nq'].mean():>12.4f} {nll['bare_q'].mean():>12.4f} "
      f"{dec_effect_bare.mean():>+12.4f}")
print(f"  {'Oracle encoder':<24} {nll['oracle_trunc_nq'].mean():>12.4f} {nll['oracle_trunc_q'].mean():>12.4f} "
      f"{dec_effect_oracle.mean():>+12.4f}")
print(f"  {'Enc prefix effect':<24} {enc_effect_nq.mean():>+12.4f} {enc_effect_q.mean():>+12.4f}")

print(f"\n  Encoder prefix effect:")
print(f"    Without decoder query: d={d_enc_nq:+.3f}")
print(f"    With decoder query:    d={d_enc_q:+.3f}")
print(f"    Reduction: {(1 - d_enc_q/d_enc_nq)*100:.0f}% (from {d_enc_nq:.3f} to {d_enc_q:.3f})")

print(f"\n  Decoder query effect:")
print(f"    Without encoder prefix: d={d_dec_bare:+.3f}")
print(f"    With encoder prefix:    d={d_dec_oracle:+.3f}")

# Interaction test
interaction = enc_effect_nq - enc_effect_q  # positive = query reduces prefix benefit
d_interaction = cohens_d(interaction)
_, p_interaction = stats.ttest_1samp(interaction, 0)
sig_int = '***' if p_interaction < 0.001 else '**' if p_interaction < 0.01 else '*' if p_interaction < 0.05 else 'ns'
print(f"\n  NLL INTERACTION (enc_prefix_benefit_nq - enc_prefix_benefit_q):")
print(f"    d={d_interaction:+.3f} ({sig_int})")
print(f"    Positive = decoder query makes encoder prefix benefit SMALLER")

# Attention budget interaction
print(f"\n--- Cross-attention mass: 2x2 ---")
ct = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    ct[cond] = np.array([r[f'probes_{cond}'][last_layer]['ct'] for r in results])

enc_ct_nq = ct['oracle_trunc_nq'] - ct['bare_nq']
enc_ct_q = ct['oracle_trunc_q'] - ct['bare_q']
dec_ct_bare = ct['bare_q'] - ct['bare_nq']
dec_ct_oracle = ct['oracle_trunc_q'] - ct['oracle_trunc_nq']
ct_interaction = enc_ct_nq - enc_ct_q

print(f"\n  {'':>24} {'No query':>12} {'With query':>12}")
print(f"  {'Bare encoder':<24} {ct['bare_nq'].mean():>12.4f} {ct['bare_q'].mean():>12.4f}")
print(f"  {'Oracle encoder':<24} {ct['oracle_trunc_nq'].mean():>12.4f} {ct['oracle_trunc_q'].mean():>12.4f}")
print(f"\n  Encoder prefix changes cross-attn mass:")
print(f"    Without query: {enc_ct_nq.mean():>+.4f}")
print(f"    With query:    {enc_ct_q.mean():>+.4f}")
print(f"  Interaction: {ct_interaction.mean():>+.4f}")
_, p_ct_int = stats.ttest_1samp(ct_interaction, 0)
sig_ct = '***' if p_ct_int < 0.001 else '**' if p_ct_int < 0.01 else '*' if p_ct_int < 0.05 else 'ns'
print(f"    ({sig_ct}, p={p_ct_int:.2e})")

# Self-attention entropy interaction
print(f"\n--- Self-attention entropy: 2x2 ---")
se = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    se[cond] = np.array([r[f'probes_{cond}'][last_layer]['se'] for r in results])

enc_se_nq = se['oracle_trunc_nq'] - se['bare_nq']
enc_se_q = se['oracle_trunc_q'] - se['bare_q']
dec_se_bare = se['bare_q'] - se['bare_nq']
se_interaction = enc_se_nq - enc_se_q

print(f"  Encoder prefix changes self-attn entropy:")
print(f"    Without query: {enc_se_nq.mean():>+.4f}")
print(f"    With query:    {enc_se_q.mean():>+.4f}")
print(f"  Decoder query changes self-attn entropy:")
print(f"    Without prefix: {dec_se_bare.mean():>+.4f}")
print(f"  Interaction: {se_interaction.mean():>+.4f}")
_, p_se_int = stats.ttest_1samp(se_interaction, 0)
sig_se = '***' if p_se_int < 0.001 else '**' if p_se_int < 0.01 else '*' if p_se_int < 0.05 else 'ns'
print(f"    ({sig_se}, p={p_se_int:.2e})")
""")


# ===== Cell 10: Summary + save =====
code(r"""# Cell 10: Summary and save
print("=" * 70)
print("SUMMARY — Exp 07: Decoder Attention Probing")
print("=" * 70)

last_layer = PROBE_LAYERS[-1]

# Gather key metrics
nll = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    nll[cond] = np.array([r[f'nll_{cond}'] for r in results])

d_enc_nq = cohens_d(nll['bare_nq'] - nll['oracle_trunc_nq'])
d_enc_q = cohens_d(nll['bare_q'] - nll['oracle_trunc_q'])
d_dec_bare = cohens_d(nll['bare_nq'] - nll['bare_q'])
d_dec_oracle = cohens_d(nll['oracle_trunc_nq'] - nll['oracle_trunc_q'])

interaction = (nll['bare_nq'] - nll['oracle_trunc_nq']) - (nll['bare_q'] - nll['oracle_trunc_q'])
d_interaction = cohens_d(interaction)
_, p_interaction = stats.ttest_1samp(interaction, 0)

# Query buffer mass at last layer
sq_bare = np.mean([r['probes_bare_q'][last_layer]['sq'] for r in results])
sq_oracle = np.mean([r['probes_oracle_trunc_q'][last_layer]['sq'] for r in results])

# Cross-attention totals at last layer
ct = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    ct[cond] = np.mean([r[f'probes_{cond}'][last_layer]['ct'] for r in results])

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)}, Decoder layers: {N_DEC_LAYERS}")
print(f"Probe layers: {PROBE_LAYERS}")

print(f"\n--- NLL 2x2 ---")
print(f"  Encoder prefix effect without query: d={d_enc_nq:+.3f}")
print(f"  Encoder prefix effect WITH query:    d={d_enc_q:+.3f}")
print(f"  Decoder query effect without prefix: d={d_dec_bare:+.3f}")
print(f"  Decoder query effect WITH prefix:    d={d_dec_oracle:+.3f}")
sig_int = '***' if p_interaction < 0.001 else 'ns'
print(f"  Interaction: d={d_interaction:+.3f} ({sig_int})")

print(f"\n--- Decoder query as attention buffer (layer {last_layer}) ---")
print(f"  Query buffer absorbs {sq_bare*100:.1f}% of answer-token attention (bare encoder)")
print(f"  Query buffer absorbs {sq_oracle*100:.1f}% of answer-token attention (oracle encoder)")

print(f"\n--- Cross-attention budget (layer {last_layer}) ---")
print(f"  bare_nq:         {ct['bare_nq']*100:.1f}%")
print(f"  bare_q:          {ct['bare_q']*100:.1f}%")
print(f"  oracle_trunc_nq: {ct['oracle_trunc_nq']*100:.1f}%")
print(f"  oracle_trunc_q:  {ct['oracle_trunc_q']*100:.1f}%")

# Hypothesis verdict
print(f"\n--- Hypothesis verdict ---")
if sq_bare > 0.05:
    print(f"  CONFIRMED: Query tokens absorb {sq_bare*100:.1f}% of answer attention budget.")
    print(f"  This is the decoder-side attention buffer mechanism.")
else:
    print(f"  NOT CONFIRMED: Query tokens absorb only {sq_bare*100:.1f}% of attention.")

if d_interaction > 0.05 and p_interaction < 0.05:
    redundancy = (1 - d_enc_q / d_enc_nq) * 100 if d_enc_nq > 0 else 0
    print(f"  Encoder prefix benefit reduced by {redundancy:.0f}% when decoder has query.")
    print(f"  The two buffer mechanisms are PARTIALLY REDUNDANT.")
elif d_interaction < -0.05:
    print(f"  Encoder prefix benefit INCREASES when decoder has query.")
    print(f"  The mechanisms are SYNERGISTIC, not redundant.")
else:
    print(f"  No significant interaction. The mechanisms appear INDEPENDENT.")

# Save results
# Aggregate probe data per (condition, layer, metric)
probe_summary = {}
for cond in ['bare_nq', 'bare_q', 'oracle_trunc_nq', 'oracle_trunc_q']:
    probe_summary[cond] = {}
    for layer in PROBE_LAYERS:
        layer_data = {}
        for metric in ['sb', 'sq', 'sa', 'st', 'ct', 'se', 'ce']:
            vals = [r[f'probes_{cond}'][layer][metric] for r in results]
            layer_data[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
        probe_summary[cond][str(layer)] = layer_data

final_results = {
    'experiment': 'v4_exp07_decoder_attention_probing',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'n_decoder_layers': N_DEC_LAYERS,
    'probe_layers': PROBE_LAYERS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'nll': {
        'bare_nq': float(nll['bare_nq'].mean()),
        'bare_q': float(nll['bare_q'].mean()),
        'oracle_trunc_nq': float(nll['oracle_trunc_nq'].mean()),
        'oracle_trunc_q': float(nll['oracle_trunc_q'].mean()),
    },
    'nll_effects': {
        'd_enc_nq': float(d_enc_nq),
        'd_enc_q': float(d_enc_q),
        'd_dec_bare': float(d_dec_bare),
        'd_dec_oracle': float(d_dec_oracle),
        'd_interaction': float(d_interaction),
        'p_interaction': float(p_interaction),
    },
    'query_buffer_mass': {
        'bare_q': float(sq_bare),
        'oracle_q': float(sq_oracle),
    },
    'probe_summary': probe_summary,
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
out_path = "experiments/encoder_decoder/07/07_decoder_attention_probing.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
