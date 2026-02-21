#!/usr/bin/env python3
"""Build Exp 3E notebook: Attention Mechanism Probing.

Experiments 2B and 3D established that ~85% of the oracle headroom is "structural" --
prepending ANY text (even "the the the...") to the encoder improves document
representations. This experiment extracts encoder attention weights and hidden states
to directly measure what changes.

Three hypotheses:
1. Attention redistribution: prefix tokens absorb attention mass
2. RoPE position shift: document tokens move to later positions
3. Representation regularization: prefix acts as noise injection

Architecture: 34 encoder layers, 5 full-attention (layers 5, 11, 17, 23, 29),
29 sliding window (1024 tokens). 8 attention heads, hidden_size=2560, head_dim=256.
Must use attn_implementation="eager" -- SDPA returns None for attention weights.

Design: N=500, 4 conditions (bare, oracle_trunc, random_matched_trunc, repeat_the_trunc),
6 probes (attention mass, entropy, doc-doc redistribution, shift magnitude, shift
direction, attention sinks).
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 3E: Attention Mechanism Probing

## Motivation

Experiments 2B and 3D established that **~85% of the oracle headroom is "structural"** --
prepending ANY text (even "the the the...") to the encoder improves document
representations. But we don't understand WHY.

**Three hypotheses**:
1. **Attention redistribution**: prefix tokens absorb attention mass, changing how
   document tokens attend to each other
2. **RoPE position shift**: document tokens move to later positions, changing their
   frequency signatures
3. **Representation regularization**: prefix acts as noise injection that produces
   more distributed/robust representations

This experiment extracts encoder attention weights and hidden states to directly
measure what changes.

## Architecture

- 34 encoder layers, 5 full-attention (layers 5, 11, 17, 23, 29), 29 sliding window
- 8 attention heads (GQA: 4 KV heads expanded to 8)
- Hidden size: 2560, head dim: 256
- Sliding window: bidirectional, 513 tokens each direction
- For a 20-token prefix + 600-word doc: only the 5 full-attention layers can directly
  connect prefix to distant document tokens

## Design

**N=500** from neural-bridge/rag-12000 (same samples as Exp 3D).

**4 conditions** (all with truncation mask):
1. `bare` -- document only
2. `oracle_trunc` -- real query + document
3. `random_matched_trunc` -- random words + document
4. `repeat_the_trunc` -- "the"xN + document

## Probes

| Probe | What it measures |
|-------|-----------------|
| **A: Attention mass on prefix** | Fraction of document tokens' attention going to prefix |
| **B: Attention entropy** | Whether prefix increases or decreases entropy of doc token attention |
| **C: Doc-doc redistribution** | How the remaining doc-doc attention changes with prefix |
| **D: Shift magnitude** | L2 distance of doc token representations (bare vs prefixed) |
| **E: Shift direction** | Cosine similarity of shift vectors across conditions (structural vs semantic) |
| **F: Attention sinks** | Whether prefix tokens take over the "sink" role |
""")


# ===== Cell 2: Setup + load dataset =====
code(r"""# Cell 2: Setup + load dataset
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

sys.path.insert(0, "../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../results/exp03e")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

# Load dataset (same as Exp 3D)
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")

all_candidates = []
for row in ds:
    q = row.get("question", "")
    doc = row.get("context", "")
    answer = row.get("answer", "")
    if not q or not doc or not answer:
        continue
    q_words = len(q.split())
    a_words = len(answer.split())
    if q_words >= 15 and a_words >= 5:
        all_candidates.append({
            "query": q,
            "document": doc,
            "answer": answer,
            "query_words": q_words,
            "doc_words": len(doc.split()),
            "answer_words": a_words,
        })

print(f"Candidates (q>=15w, a>=5w): {len(all_candidates)}")

# Same shuffle and selection as Exp 3D
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]

q_lens = np.array([s["query_words"] for s in samples])
d_lens = np.array([s["doc_words"] for s in samples])

print(f"\nSample statistics (N={N_SAMPLES}):")
print(f"  Query length:  mean={q_lens.mean():.1f}, range=[{q_lens.min()}, {q_lens.max()}]")
print(f"  Doc length:    mean={d_lens.mean():.1f}, range=[{d_lens.min()}, {d_lens.max()}]")

del ds
gc.collect()
print("Dataset loaded.")
""")


# ===== Cell 3: Load model with eager attention + define hooks =====
code(r"""# Cell 3: Load model with eager attention + define hooks
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME} with attn_implementation='eager'...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    attn_implementation="eager",
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Get encoder text model reference (T5Gemma2 has multimodal wrapper)
# model.model.encoder = T5Gemma2Encoder (multimodal)
# model.model.encoder.text_model = T5Gemma2TextEncoder (has .layers)
encoder_text = model.model.encoder.text_model
n_layers = len(encoder_text.layers)
print(f"Encoder layers: {n_layers}")

# Verify eager attention is active (SDPA returns None for attention weights)
enc_attn_impl = encoder_text.layers[0].self_attn.config._attn_implementation
print(f"Encoder attn_implementation: {enc_attn_impl}")
assert enc_attn_impl == "eager", (
    f"Expected 'eager' but got '{enc_attn_impl}'. "
    f"SDPA will return None attention weights!"
)

# Identify full-attention vs sliding-window layers
layer_types = []
full_attn_layers = []
for i in range(n_layers):
    lt = encoder_text.layers[i].attention_type
    layer_types.append(lt)
    if lt == "full_attention":
        full_attn_layers.append(i)

print(f"Full-attention layers: {full_attn_layers}")
print(f"Sliding-window layers: {[i for i in range(n_layers) if layer_types[i] != 'full_attention']}")

# Layers to probe: first layer, all full-attention layers, final layer
ATTN_LAYERS = sorted(set([0] + full_attn_layers))
HIDDEN_LAYERS = sorted(set([0] + full_attn_layers + [n_layers - 1]))
print(f"\nAttention probe layers: {ATTN_LAYERS}")
print(f"Hidden state probe layers: {HIDDEN_LAYERS}")

# Hook infrastructure: stores captured tensors per forward pass
captured_attn = {}
captured_hidden = {}
hook_handles = []


def make_attn_hook(layer_idx):
    # Hook on self_attn: captures (attn_output, attn_weights)
    def hook_fn(module, input, output):
        attn_output, attn_weights = output
        if attn_weights is not None:
            # attn_weights: (batch, n_heads, seq_len, seq_len)
            captured_attn[layer_idx] = attn_weights.detach().float()
        else:
            print(f"WARNING: Layer {layer_idx} returned None attention weights!")
    return hook_fn


def make_hidden_hook(layer_idx):
    # Hook on encoder layer: captures hidden_states output
    def hook_fn(module, input, output):
        # T5GemmaEncoderLayer.forward returns just hidden_states (a single tensor)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured_hidden[layer_idx] = h.detach().float()
    return hook_fn


def register_hooks():
    global hook_handles
    remove_hooks()
    for layer_idx in ATTN_LAYERS:
        h = encoder_text.layers[layer_idx].self_attn.register_forward_hook(
            make_attn_hook(layer_idx)
        )
        hook_handles.append(h)
    for layer_idx in HIDDEN_LAYERS:
        h = encoder_text.layers[layer_idx].register_forward_hook(
            make_hidden_hook(layer_idx)
        )
        hook_handles.append(h)


def remove_hooks():
    global hook_handles
    for h in hook_handles:
        h.remove()
    hook_handles = []


def clear_captures():
    captured_attn.clear()
    captured_hidden.clear()


# Token counting helper (same as Exp 3D)
def count_prefix_tokens(prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


print("Hooks and helpers defined.")
""")


# ===== Cell 4: Generate conditions =====
code(r"""# Cell 4: Generate conditions for each sample (same as Exp 3D)

# Build word pool from unrelated documents
other_words_pool = []
for i, s in enumerate(samples):
    other_idx = (i + N_SAMPLES // 2) % N_SAMPLES
    other_doc = samples[other_idx]['document']
    other_words_pool.append(other_doc.split())

for i, s in enumerate(samples):
    query_words = s['query'].split()
    n_query_words = len(query_words)
    other_words = other_words_pool[i]

    # random_matched: N random words from unrelated doc
    if len(other_words) >= n_query_words:
        s['random_matched'] = " ".join(other_words[:n_query_words])
    else:
        padded = other_words * ((n_query_words // len(other_words)) + 1)
        s['random_matched'] = " ".join(padded[:n_query_words])

    # repeat_the: "the" repeated N times
    s['repeat_the'] = " ".join(["the"] * n_query_words)

COND_NAMES = ['bare', 'oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']

# Show prefix token stats
print(f"Conditions: {COND_NAMES}")
print(f"\nPrefix token counts (first 50 samples):")
for c in COND_NAMES:
    if c == 'bare':
        continue
    if c == 'oracle_trunc':
        toks = [count_prefix_tokens(s['query'], s['document']) for s in samples[:50]]
    else:
        key = c.replace('_trunc', '')
        toks = [count_prefix_tokens(s[key], s['document']) for s in samples[:50]]
    print(f"  {c:<28} mean={np.mean(toks):.1f}, range=[{min(toks)}, {max(toks)}]")

# Example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query ({ex['query_words']}w): {ex['query'][:100]}...")
print(f"  Document ({ex['doc_words']}w): {ex['document'][:80]}...")

# Show what each condition's encoder input looks like
print(f"\n--- Encoder input for each condition (sample 0) ---")
cond_examples = {
    'bare': ex['document'],
    'oracle_trunc': ex['query'] + "\n" + ex['document'],
    'random_matched_trunc': ex['random_matched'] + "\n" + ex['document'],
    'repeat_the_trunc': ex['repeat_the'] + "\n" + ex['document'],
}
for cond_name, enc_text in cond_examples.items():
    if cond_name == 'bare':
        ptoks = 0
        prefix_display = "(none)"
    else:
        key = cond_name.replace('_trunc', '')
        prefix_text = ex[key]
        ptoks = count_prefix_tokens(prefix_text, ex['document'])
        prefix_display = prefix_text[:60]
    print(f"  {cond_name:<28} ({ptoks:>3} prefix toks)")
    if cond_name != 'bare':
        print(f"    prefix: {prefix_display}")
    print(f"    enc input: {enc_text[:80]}...")
""")


# ===== Cell 5: Run extraction loop =====
code(r"""# Cell 5: Run extraction loop
# For each sample x condition: run encoder, capture attention + hidden states,
# compute all probe metrics on-the-fly, store only aggregated results.

print("=" * 70)
print("EXTRACTION LOOP")
print("=" * 70)

# Initialize accumulators for all probes
# Indexed by [condition][layer] where appropriate.

# Probe A: mean attention mass on prefix per doc token, per layer/head
# Shape per entry: (n_heads,) -- mean across doc tokens and samples
probe_a_mass = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES if c != 'bare'}
# Also store per-head mass distribution across doc token positions (mean across samples)
probe_a_by_position = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES if c != 'bare'}

# Probe B: attention entropy per doc token, per layer
# Store: mean entropy per layer, for each condition
probe_b_entropy = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES}

# Probe C: doc-doc attention pattern divergence
# KL divergence between bare doc-doc attention and prefixed doc-doc attention
probe_c_kl = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES if c != 'bare'}
# Also: entropy of doc-doc sub-pattern (separate from full entropy)
probe_c_docdoc_entropy = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES}

# Probe D: representation shift magnitude per layer
# Mean L2 distance of doc token reps (bare vs prefixed)
probe_d_shift = {c: {l: [] for l in HIDDEN_LAYERS} for c in COND_NAMES if c != 'bare'}
# Per-position shift (binned into 10 position bins)
N_POS_BINS = 10
probe_d_by_position = {c: {l: [] for l in HIDDEN_LAYERS} for c in COND_NAMES if c != 'bare'}

# Probe E: shift direction similarity
# Cosine similarity between (h_oracle - h_bare) and (h_X - h_bare) per doc token per layer
probe_e_cosine = {c: {l: [] for l in HIDDEN_LAYERS}
                  for c in COND_NAMES if c not in ('bare', 'oracle_trunc')}

# Probe F: attention sinks -- total attention received by each position
# For bare: which doc positions absorb most attention
# For prefixed: do prefix tokens absorb it instead
probe_f_received = {c: {l: [] for l in ATTN_LAYERS} for c in COND_NAMES}
# Separate: total attention received by prefix positions
probe_f_prefix_received = {c: {l: [] for l in ATTN_LAYERS}
                           for c in COND_NAMES if c != 'bare'}

# NLL scores for cross-reference with Exp 3D
nll_scores = {c: [] for c in COND_NAMES}

# Per-sample metadata
sample_meta = []

# ---- Helper: compute attention entropy ----
def attn_entropy(weights, mask=None):
    # weights: (n_heads, seq, seq) -- attention probabilities
    # mask: (seq,) boolean -- True for positions to include
    # Returns: (n_heads, seq) mean entropy per head per query position
    eps = 1e-10
    # Only compute for positions indicated by mask
    log_w = torch.log(weights + eps)
    ent = -(weights * log_w).sum(dim=-1)  # (n_heads, seq)
    return ent

# ---- Helper: encode and extract ----
def encode_and_extract(text):
    # Returns encoder outputs (via hooks on text_model layers) and input_ids
    enc_ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    enc_mask = torch.ones(1, enc_ids.shape[1], device=DEVICE, dtype=torch.long)
    clear_captures()
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )
    return enc_ids, encoder_outputs

# ---- Helper: score NLL ----
def score_nll(encoder_outputs, total_enc_len, answer_text, prefix_token_count=0,
              truncate=False):
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

    del outputs, logits, log_probs
    return mean_nll

# ---- Main loop ----
register_hooks()

start_idx = 0
if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if (ckpt.get('n_total') == N_SAMPLES
            and len(ckpt.get('sample_meta', [])) > 0):
        saved_qs = [m['query'][:50] for m in ckpt['sample_meta']]
        current_qs = [s['query'][:50] for s in samples[:len(saved_qs)]]
        if saved_qs == current_qs:
            start_idx = len(saved_qs)
            # Restore accumulators
            for key in ['probe_a_mass', 'probe_b_entropy', 'probe_c_kl',
                        'probe_c_docdoc_entropy', 'probe_d_shift',
                        'probe_d_by_position', 'probe_e_cosine',
                        'probe_f_received', 'probe_f_prefix_received',
                        'nll_scores', 'sample_meta']:
                saved = ckpt.get(key)
                if saved is not None:
                    local_var = locals()[key]
                    if isinstance(local_var, dict) and isinstance(saved, dict):
                        # Nested dict: convert string keys back to int
                        for ck, cv in saved.items():
                            if isinstance(cv, dict):
                                local_var[ck] = {int(lk): lv for lk, lv in cv.items()}
                            else:
                                local_var[ck] = cv
                    elif isinstance(local_var, list):
                        local_var.clear()
                        local_var.extend(saved)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {N_SAMPLES} samples x {len(COND_NAMES)} conditions "
          f"= {N_SAMPLES * len(COND_NAMES)} forward passes")

t0 = time.time()

for sample_idx in tqdm(range(start_idx, N_SAMPLES), initial=start_idx,
                       total=N_SAMPLES, desc="Extracting"):
    s = samples[sample_idx]

    # ---- Build encoder texts for each condition ----
    cond_texts = {}
    cond_ptoks = {}
    for c in COND_NAMES:
        if c == 'bare':
            cond_texts[c] = s['document']
            cond_ptoks[c] = 0
        elif c == 'oracle_trunc':
            cond_texts[c] = s['query'] + "\n" + s['document']
            cond_ptoks[c] = count_prefix_tokens(s['query'], s['document'])
        else:
            key = c.replace('_trunc', '')
            cond_texts[c] = s[key] + "\n" + s['document']
            cond_ptoks[c] = count_prefix_tokens(s[key], s['document'])

    # ---- Run encoder for each condition ----
    cond_attn = {}    # condition -> layer -> attn_weights
    cond_hidden = {}  # condition -> layer -> hidden_states

    for c in COND_NAMES:
        enc_ids, enc_out = encode_and_extract(cond_texts[c])
        seq_len = enc_ids.shape[1]

        # Copy captured tensors (they'll be overwritten next forward pass)
        cond_attn[c] = {l: captured_attn[l].clone() for l in ATTN_LAYERS
                        if l in captured_attn}
        cond_hidden[c] = {l: captured_hidden[l].clone() for l in HIDDEN_LAYERS
                          if l in captured_hidden}

        # Score NLL for cross-reference
        nll = score_nll(enc_out, seq_len, s['answer'],
                        cond_ptoks[c], truncate=(c != 'bare'))
        nll_scores[c].append(nll)

        del enc_ids, enc_out
        clear_captures()

    # ---- Record metadata ----
    bare_seq_len = list(cond_hidden['bare'].values())[0].shape[1]
    sample_meta.append({
        'query': s['query'],
        'query_words': s['query_words'],
        'doc_words': s['doc_words'],
        'bare_seq_len': bare_seq_len,
        'prefix_tokens': {c: cond_ptoks[c] for c in COND_NAMES},
    })

    # ---- Compute probe metrics ----
    n_doc_bare = bare_seq_len  # all tokens are "document" in bare

    for c in COND_NAMES:
        ptoks = cond_ptoks[c]
        n_doc = (list(cond_hidden[c].values())[0].shape[1]) - ptoks

        for l in ATTN_LAYERS:
            if l not in cond_attn[c]:
                continue
            # attn: (1, n_heads, seq, seq) -> (n_heads, seq, seq)
            attn = cond_attn[c][l][0]
            n_heads = attn.shape[0]
            seq = attn.shape[1]

            if c != 'bare':
                # Probe A: attention mass on prefix from each doc token
                # doc tokens are positions [ptoks:], prefix at [:ptoks]
                doc_to_prefix = attn[:, ptoks:, :ptoks]  # (heads, n_doc, ptoks)
                mass_per_head = doc_to_prefix.sum(dim=-1).mean(dim=-1)  # (heads,)
                probe_a_mass[c][l].append(mass_per_head.cpu().numpy().tolist())

                # Position-dependent: mass on prefix by doc token position (10 bins)
                doc_mass = doc_to_prefix.sum(dim=-1).mean(dim=0)  # (n_doc,)
                if n_doc >= N_POS_BINS:
                    bins = np.array_split(doc_mass.cpu().numpy(), N_POS_BINS)
                    binned = [float(np.mean(b)) for b in bins]
                else:
                    binned = doc_mass.cpu().numpy().tolist()
                probe_a_by_position[c][l].append(binned)

            # Probe B: attention entropy for doc tokens
            if c == 'bare':
                doc_attn_rows = attn[:, :, :]  # all tokens are doc
            else:
                doc_attn_rows = attn[:, ptoks:, :]  # doc token rows
            ent = attn_entropy(doc_attn_rows)  # (heads, n_doc)
            mean_ent = ent.mean(dim=-1).mean(dim=0).item()  # scalar
            probe_b_entropy[c][l].append(mean_ent)

            # Probe C: doc-doc sub-attention
            if c == 'bare':
                docdoc = attn[:, :, :]  # all is doc-doc
            else:
                docdoc = attn[:, ptoks:, ptoks:]  # doc-to-doc submatrix
            # Renormalize doc-doc to sum to 1 per row
            docdoc_sum = docdoc.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            docdoc_norm = docdoc / docdoc_sum
            ent_dd = attn_entropy(docdoc_norm)
            mean_ent_dd = ent_dd.mean(dim=-1).mean(dim=0).item()
            probe_c_docdoc_entropy[c][l].append(mean_ent_dd)

            if c != 'bare':
                # KL(bare_docdoc || prefixed_docdoc) for matching doc positions
                # bare doc-doc: all positions are doc
                bare_attn = cond_attn['bare'][l][0]
                bare_n = bare_attn.shape[1]
                # Use the last min(bare_n, n_doc) positions for alignment
                align_n = min(bare_n, n_doc)
                if align_n > 0:
                    bare_dd = bare_attn[:, -align_n:, -align_n:]
                    pref_dd = docdoc_norm[:, -align_n:, -align_n:]
                    # Renormalize bare to same window
                    bare_dd_sum = bare_dd.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    bare_dd_norm = bare_dd / bare_dd_sum
                    # KL divergence per position per head, then mean
                    eps = 1e-10
                    kl = (bare_dd_norm * (torch.log(bare_dd_norm + eps)
                                          - torch.log(pref_dd + eps)))
                    kl = kl.sum(dim=-1).mean(dim=-1).mean(dim=0).item()
                    probe_c_kl[c][l].append(kl)
                else:
                    probe_c_kl[c][l].append(0.0)

            # Probe F: attention received per position (attention sink)
            # Sum attention each position receives from all other positions
            received = attn.sum(dim=1).mean(dim=0)  # (seq,) mean across heads
            # Normalize by seq_len so it's comparable across conditions
            received = received / seq
            if c == 'bare':
                # Store stats about top-k sinks
                top_vals, top_idxs = received.topk(min(5, seq))
                probe_f_received[c][l].append({
                    'top5_vals': top_vals.cpu().numpy().tolist(),
                    'top5_idxs': top_idxs.cpu().numpy().tolist(),
                    'first_pos_val': received[0].item(),
                    'mean_val': received.mean().item(),
                })
            else:
                prefix_recv = received[:ptoks].mean().item() if ptoks > 0 else 0
                doc_recv = received[ptoks:].mean().item()
                top_vals, top_idxs = received.topk(min(5, seq))
                probe_f_received[c][l].append({
                    'prefix_mean_recv': prefix_recv,
                    'doc_mean_recv': doc_recv,
                    'top5_vals': top_vals.cpu().numpy().tolist(),
                    'top5_idxs': top_idxs.cpu().numpy().tolist(),
                })
                probe_f_prefix_received[c][l].append(prefix_recv)

        # Probe D: representation shift
        if c != 'bare':
            for l in HIDDEN_LAYERS:
                if l not in cond_hidden[c] or l not in cond_hidden['bare']:
                    continue
                h_bare = cond_hidden['bare'][l][0]   # (bare_seq, hidden)
                h_pref = cond_hidden[c][l][0]        # (pref_seq, hidden)

                # Align on last n_doc positions (document tokens)
                align_n = min(h_bare.shape[0], n_doc)
                if align_n > 0:
                    h_b = h_bare[-align_n:]
                    h_p = h_pref[-align_n:]
                    shifts = (h_p - h_b).norm(dim=-1)  # (align_n,)
                    probe_d_shift[c][l].append(shifts.mean().item())

                    # Position-binned shifts
                    if align_n >= N_POS_BINS:
                        bins = np.array_split(shifts.cpu().numpy(), N_POS_BINS)
                        binned = [float(np.mean(b)) for b in bins]
                    else:
                        binned = shifts.cpu().numpy().tolist()
                    probe_d_by_position[c][l].append(binned)

        # Probe E: shift direction cosine similarity
        if c not in ('bare', 'oracle_trunc'):
            for l in HIDDEN_LAYERS:
                if (l not in cond_hidden[c] or l not in cond_hidden['bare']
                        or l not in cond_hidden.get('oracle_trunc', {})):
                    continue
                h_bare = cond_hidden['bare'][l][0]
                h_oracle = cond_hidden['oracle_trunc'][l][0]
                h_other = cond_hidden[c][l][0]

                n_doc_oracle = h_oracle.shape[0] - cond_ptoks['oracle_trunc']
                n_doc_other = h_other.shape[0] - cond_ptoks[c]
                align_n = min(h_bare.shape[0], n_doc_oracle, n_doc_other)

                if align_n > 0:
                    shift_oracle = h_oracle[-align_n:] - h_bare[-align_n:]
                    shift_other = h_other[-align_n:] - h_bare[-align_n:]
                    # Cosine similarity per token
                    cos = F.cosine_similarity(shift_oracle, shift_other, dim=-1)
                    probe_e_cosine[c][l].append(cos.mean().item())

    # Free condition tensors
    del cond_attn, cond_hidden
    gc.collect()
    torch.cuda.empty_cache()

    # Checkpoint every 20 samples
    if (sample_idx + 1) % 20 == 0 or sample_idx == N_SAMPLES - 1:
        # Convert all accumulators to serializable form
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return obj
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        ckpt = {
            'n_total': N_SAMPLES,
            'probe_a_mass': to_serializable(probe_a_mass),
            'probe_b_entropy': to_serializable(probe_b_entropy),
            'probe_c_kl': to_serializable(probe_c_kl),
            'probe_c_docdoc_entropy': to_serializable(probe_c_docdoc_entropy),
            'probe_d_shift': to_serializable(probe_d_shift),
            'probe_d_by_position': to_serializable(probe_d_by_position),
            'probe_e_cosine': to_serializable(probe_e_cosine),
            'probe_f_received': to_serializable(probe_f_received),
            'probe_f_prefix_received': to_serializable(probe_f_prefix_received),
            'nll_scores': nll_scores,
            'sample_meta': sample_meta,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = sample_idx - start_idx + 1
        eta = (N_SAMPLES - sample_idx - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {sample_idx+1}/{N_SAMPLES} | "
                   f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

remove_hooks()

elapsed = time.time() - t0
print(f"\nExtraction complete: {N_SAMPLES} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
print(f"NLL cross-check: bare={np.mean(nll_scores['bare']):.4f}, "
      f"oracle={np.mean(nll_scores['oracle_trunc']):.4f}")
""")


# ===== Cell 6: Probe A — Attention mass on prefix =====
code(r"""# Cell 6: Probe A — Attention mass on prefix

print("=" * 70)
print("PROBE A: ATTENTION MASS ON PREFIX")
print("=" * 70)
print("For prefixed conditions, what fraction of each document token's attention")
print("goes to prefix tokens? By layer and head.\n")

for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    print(f"\n--- {c} ---")
    print(f"{'Layer':>6} {'Type':>10} {'Mean mass':>11} {'Std':>8} {'Min head':>10} {'Max head':>10}")
    print("-" * 60)
    for l in ATTN_LAYERS:
        data = probe_a_mass[c][l]
        if not data:
            continue
        # data is list of (n_heads,) per sample -> (N, n_heads)
        arr = np.array(data)  # (N, n_heads)
        mean_per_head = arr.mean(axis=0)  # (n_heads,)
        overall_mean = mean_per_head.mean()
        overall_std = arr.mean(axis=1).std()
        lt = layer_types[l][:4]
        print(f"  {l:>4}  {lt:>10} {overall_mean:>11.4f} {overall_std:>8.4f} "
              f"{mean_per_head.min():>10.4f} {mean_per_head.max():>10.4f}")

# Compare across conditions at full-attention layers
print(f"\n\n--- Cross-condition comparison (full-attention layers only) ---")
print(f"{'Layer':>6} {'oracle':>10} {'random':>10} {'repeat_the':>12} {'orc-rand':>10}")
print("-" * 60)
for l in full_attn_layers:
    o_mass = np.array(probe_a_mass['oracle_trunc'][l]).mean()
    r_mass = np.array(probe_a_mass['random_matched_trunc'][l]).mean()
    t_mass = np.array(probe_a_mass['repeat_the_trunc'][l]).mean()
    print(f"  {l:>4} {o_mass:>10.4f} {r_mass:>10.4f} {t_mass:>12.4f} "
          f"{o_mass - r_mass:>+10.4f}")

# Position-dependent attention mass (does prefix attract equally from all doc positions?)
print(f"\n\n--- Position-dependent prefix attention mass (oracle, mean across heads) ---")
print(f"Position bin: 0=start of doc, 9=end of doc")
for l in full_attn_layers:
    data = probe_a_by_position['oracle_trunc'][l]
    if not data or not all(len(d) == N_POS_BINS for d in data):
        # Skip if variable length
        continue
    arr = np.array(data)  # (N, N_POS_BINS)
    means = arr.mean(axis=0)
    print(f"  Layer {l}: " + " ".join(f"{m:.3f}" for m in means))

print(f"\nKey question: is mass UNIFORM across doc positions or CONCENTRATED?")
for l in full_attn_layers:
    data = probe_a_by_position['oracle_trunc'][l]
    if not data or not all(len(d) == N_POS_BINS for d in data):
        continue
    arr = np.array(data).mean(axis=0)
    cv = np.std(arr) / np.mean(arr) if np.mean(arr) > 0 else 0
    print(f"  Layer {l}: CV (coeff of variation) = {cv:.3f} "
          f"({'uniform' if cv < 0.3 else 'concentrated'})")
""")


# ===== Cell 7: Probe B — Attention entropy =====
code(r"""# Cell 7: Probe B — Attention entropy

print("=" * 70)
print("PROBE B: ATTENTION ENTROPY")
print("=" * 70)
print("Does the prefix INCREASE entropy (dilution/regularization)")
print("or DECREASE it (focusing)?\n")

print(f"{'Layer':>6} {'Type':>10} {'bare':>10} {'oracle':>10} {'random':>10} "
      f"{'repeat':>10} {'orc-bare':>10} {'rand-bare':>10}")
print("-" * 85)

entropy_changes = {c: [] for c in COND_NAMES if c != 'bare'}

for l in ATTN_LAYERS:
    bare_ent = np.mean(probe_b_entropy['bare'][l])
    lt = layer_types[l][:4]
    vals = [f"  {l:>4}  {lt:>10} {bare_ent:>10.3f}"]
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        c_ent = np.mean(probe_b_entropy[c][l])
        vals.append(f"{c_ent:>10.3f}")
    # Deltas
    for c in ['oracle_trunc', 'random_matched_trunc']:
        c_ent = np.mean(probe_b_entropy[c][l])
        delta = c_ent - bare_ent
        vals.append(f"{delta:>+10.3f}")
        entropy_changes[c].append(delta)
    print(" ".join(vals))

# Statistical test: is entropy change significant?
print(f"\n--- Statistical tests ---")
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    for l in ATTN_LAYERS:
        bare_arr = np.array(probe_b_entropy['bare'][l])
        cond_arr = np.array(probe_b_entropy[c][l])
        if len(bare_arr) == len(cond_arr) and len(bare_arr) > 1:
            diff = cond_arr - bare_arr
            d = cohens_d(diff)
            _, p = stats.ttest_1samp(diff, 0)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            if l in full_attn_layers:
                print(f"  {c} layer {l} (full): d={d:+.3f}, p={p:.2e} {sig}")

# Summary
print(f"\n--- Summary ---")
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    increases = 0
    decreases = 0
    for l in ATTN_LAYERS:
        bare_ent = np.mean(probe_b_entropy['bare'][l])
        cond_ent = np.mean(probe_b_entropy[c][l])
        if cond_ent > bare_ent:
            increases += 1
        else:
            decreases += 1
    direction = "INCREASES" if increases > decreases else "DECREASES"
    print(f"  {c}: entropy {direction} in {increases}/{len(ATTN_LAYERS)} layers")
""")


# ===== Cell 8: Probe C — Doc-doc attention redistribution =====
code(r"""# Cell 8: Probe C — Document-document attention redistribution

print("=" * 70)
print("PROBE C: DOC-DOC ATTENTION REDISTRIBUTION")
print("=" * 70)
print("After removing attention to prefix, how does the remaining doc-doc")
print("attention pattern compare to bare?\n")

# Doc-doc entropy
print(f"{'Layer':>6} {'bare dd':>10} {'oracle dd':>12} {'random dd':>12} "
      f"{'repeat dd':>12} {'orc-bare':>10}")
print("-" * 75)

for l in ATTN_LAYERS:
    bare_dd = np.mean(probe_c_docdoc_entropy['bare'][l])
    vals = [f"  {l:>4} {bare_dd:>10.3f}"]
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        c_dd = np.mean(probe_c_docdoc_entropy[c][l])
        vals.append(f"{c_dd:>12.3f}")
    delta = np.mean(probe_c_docdoc_entropy['oracle_trunc'][l]) - bare_dd
    vals.append(f"{delta:>+10.3f}")
    print(" ".join(vals))

# KL divergence
print(f"\n--- KL divergence: bare doc-doc || prefixed doc-doc ---")
print(f"{'Layer':>6} {'Type':>10} {'oracle KL':>12} {'random KL':>12} {'repeat KL':>12}")
print("-" * 60)

for l in ATTN_LAYERS:
    lt = layer_types[l][:4]
    vals = [f"  {l:>4}  {lt:>10}"]
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        kl = np.mean(probe_c_kl[c][l]) if probe_c_kl[c][l] else 0
        vals.append(f"{kl:>12.4f}")
    print(" ".join(vals))

# Key question: does oracle redistribute MORE than random?
print(f"\n--- Does oracle redistribute more than random? ---")
for l in full_attn_layers:
    o_kl = np.array(probe_c_kl['oracle_trunc'][l])
    r_kl = np.array(probe_c_kl['random_matched_trunc'][l])
    if len(o_kl) > 1 and len(r_kl) > 1:
        diff = o_kl - r_kl
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        winner = "oracle" if d > 0 else "random"
        print(f"  Layer {l}: d={d:+.3f}, p={p:.2e} {sig} [{winner} redistributes more]")

print(f"\nInterpretation:")
print(f"  High KL = prefix causes large change to doc-doc attention pattern")
print(f"  Similar KL across conditions = redistribution is structural")
print(f"  Oracle KL > Random KL = oracle causes content-specific redistribution")
""")


# ===== Cell 9: Probe D — Representation shift magnitude =====
code(r"""# Cell 9: Probe D — Representation shift magnitude

print("=" * 70)
print("PROBE D: REPRESENTATION SHIFT MAGNITUDE")
print("=" * 70)
print("L2 distance between bare and prefixed doc token representations.\n")

print(f"{'Layer':>6} {'oracle L2':>12} {'random L2':>12} {'repeat L2':>12} "
      f"{'orc/rand':>10}")
print("-" * 60)

for l in HIDDEN_LAYERS:
    vals = [f"  {l:>4}"]
    o_shift = np.mean(probe_d_shift['oracle_trunc'][l])
    r_shift = np.mean(probe_d_shift['random_matched_trunc'][l])
    t_shift = np.mean(probe_d_shift['repeat_the_trunc'][l])
    ratio = o_shift / r_shift if r_shift > 0 else 0
    vals.append(f"{o_shift:>12.4f}")
    vals.append(f"{r_shift:>12.4f}")
    vals.append(f"{t_shift:>12.4f}")
    vals.append(f"{ratio:>10.2f}x")
    print(" ".join(vals))

# Does shift grow with layer depth?
print(f"\n--- Shift growth across layers ---")
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    shifts = [np.mean(probe_d_shift[c][l]) for l in HIDDEN_LAYERS
              if probe_d_shift[c][l]]
    if len(shifts) >= 2:
        ratio = shifts[-1] / shifts[0] if shifts[0] > 0 else 0
        print(f"  {c}: first layer={shifts[0]:.4f}, last layer={shifts[-1]:.4f}, "
              f"ratio={ratio:.1f}x")

# Position-dependent shift
print(f"\n--- Position-dependent shift (last hidden layer) ---")
last_l = HIDDEN_LAYERS[-1]
print(f"Position bin: 0=start of doc, 9=end of doc")
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    data = probe_d_by_position[c][last_l]
    if data and all(len(d) == N_POS_BINS for d in data):
        arr = np.array(data).mean(axis=0)
        print(f"  {c}: " + " ".join(f"{v:.3f}" for v in arr))

# Statistical test: does oracle shift MORE than random?
print(f"\n--- Oracle vs Random shift magnitude ---")
for l in HIDDEN_LAYERS:
    o_arr = np.array(probe_d_shift['oracle_trunc'][l])
    r_arr = np.array(probe_d_shift['random_matched_trunc'][l])
    if len(o_arr) > 1 and len(r_arr) > 1:
        diff = o_arr - r_arr
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  Layer {l}: d={d:+.3f}, p={p:.2e} {sig}")

print(f"\nKey question: if oracle and random shift by SIMILAR amounts,")
print(f"  the shift is structural. If oracle shifts MORE, there's a semantic component.")
""")


# ===== Cell 10: Probe E — Representation shift direction =====
code(r"""# Cell 10: Probe E — Representation shift direction

print("=" * 70)
print("PROBE E: REPRESENTATION SHIFT DIRECTION")
print("=" * 70)
print("Cosine similarity between shift vectors: (h_oracle - h_bare) vs (h_X - h_bare)")
print("High cosine = all prefixes push in same direction (structural)")
print("Low cosine = different prefixes push differently (semantic)\n")

print(f"{'Layer':>6} {'rand vs orc':>14} {'repeat vs orc':>16} {'interpretation':>20}")
print("-" * 60)

for l in HIDDEN_LAYERS:
    vals = [f"  {l:>4}"]
    cosines = {}
    for c in ['random_matched_trunc', 'repeat_the_trunc']:
        if probe_e_cosine[c][l]:
            cos_mean = np.mean(probe_e_cosine[c][l])
            cosines[c] = cos_mean
            vals.append(f"{cos_mean:>14.4f}")
        else:
            vals.append(f"{'N/A':>14}")

    # Interpretation
    if cosines:
        avg_cos = np.mean(list(cosines.values()))
        if avg_cos > 0.7:
            interp = "STRUCTURAL"
        elif avg_cos > 0.3:
            interp = "MIXED"
        else:
            interp = "SEMANTIC"
        vals.append(f"{interp:>20}")
    print(" ".join(vals))

# Distribution across samples (for the last full-attention layer)
print(f"\n--- Distribution across samples (layer {full_attn_layers[-1]}) ---")
target_l = full_attn_layers[-1]
for c in ['random_matched_trunc', 'repeat_the_trunc']:
    if probe_e_cosine[c][target_l]:
        arr = np.array(probe_e_cosine[c][target_l])
        print(f"  {c}:")
        print(f"    mean={arr.mean():.4f}, std={arr.std():.4f}")
        pcts = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"    10th={pcts[0]:.3f}, 25th={pcts[1]:.3f}, median={pcts[2]:.3f}, "
              f"75th={pcts[3]:.3f}, 90th={pcts[4]:.3f}")
        print(f"    % > 0.5: {100*np.mean(arr > 0.5):.1f}%")
        print(f"    % > 0.7: {100*np.mean(arr > 0.7):.1f}%")

# Cosine between random and repeat_the shifts (both non-oracle)
print(f"\n--- random vs repeat_the shift direction ---")
for l in HIDDEN_LAYERS:
    r_cos = probe_e_cosine.get('random_matched_trunc', {}).get(l, [])
    t_cos = probe_e_cosine.get('repeat_the_trunc', {}).get(l, [])
    if r_cos and t_cos:
        # Both are vs oracle. If both have high cosine with oracle,
        # they also have high cosine with each other.
        r_mean = np.mean(r_cos)
        t_mean = np.mean(t_cos)
        print(f"  Layer {l}: random-vs-oracle={r_mean:.4f}, "
              f"repeat-vs-oracle={t_mean:.4f}")

print(f"\nSummary:")
overall_cos = []
for c in ['random_matched_trunc', 'repeat_the_trunc']:
    for l in HIDDEN_LAYERS:
        if probe_e_cosine[c][l]:
            overall_cos.append(np.mean(probe_e_cosine[c][l]))
if overall_cos:
    avg = np.mean(overall_cos)
    print(f"  Overall mean cosine: {avg:.4f}")
    if avg > 0.7:
        print(f"  --> Shift is overwhelmingly STRUCTURAL (same direction regardless of prefix)")
    elif avg > 0.3:
        print(f"  --> MIXED: partially structural, partially content-dependent")
    else:
        print(f"  --> Shift is content-DEPENDENT (different prefixes push differently)")
""")


# ===== Cell 11: Probe F — Attention sink analysis =====
code(r"""# Cell 11: Probe F — Attention sink analysis

print("=" * 70)
print("PROBE F: ATTENTION SINK ANALYSIS")
print("=" * 70)
print("Which positions absorb the most attention from other tokens?")
print("Do prefix tokens take over the 'sink' role?\n")

# Bare: which positions are sinks?
print("--- Bare condition: top sink positions ---")
for l in full_attn_layers:
    data = probe_f_received['bare'][l]
    if not data:
        continue
    # Average first-position attention received
    first_vals = [d['first_pos_val'] for d in data]
    mean_vals = [d['mean_val'] for d in data]
    first_mean = np.mean(first_vals)
    avg_mean = np.mean(mean_vals)
    ratio = first_mean / avg_mean if avg_mean > 0 else 0
    print(f"  Layer {l}: first_pos receives {first_mean:.4f} "
          f"(avg pos receives {avg_mean:.4f}), ratio={ratio:.1f}x")

# Prefixed: do prefix tokens absorb attention?
print(f"\n--- Prefixed conditions: prefix vs doc attention received ---")
print(f"{'Layer':>6} {'Condition':>26} {'prefix recv':>14} {'doc recv':>12} "
      f"{'prefix/doc':>12}")
print("-" * 75)

for l in full_attn_layers:
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        data = probe_f_received[c][l]
        if not data:
            continue
        p_recv = np.mean([d['prefix_mean_recv'] for d in data])
        d_recv = np.mean([d['doc_mean_recv'] for d in data])
        ratio = p_recv / d_recv if d_recv > 0 else 0
        print(f"  {l:>4}  {c:>26} {p_recv:>14.4f} {d_recv:>12.4f} {ratio:>12.1f}x")

# Does prefix absorb MORE attention for oracle vs random?
print(f"\n--- Oracle vs Random prefix attention received ---")
for l in full_attn_layers:
    o_data = probe_f_prefix_received['oracle_trunc'][l]
    r_data = probe_f_prefix_received['random_matched_trunc'][l]
    if len(o_data) > 1 and len(r_data) > 1:
        o_arr = np.array(o_data)
        r_arr = np.array(r_data)
        diff = o_arr - r_arr
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        winner = "oracle" if d > 0 else "random"
        print(f"  Layer {l}: d={d:+.3f}, p={p:.2e} {sig} [{winner} receives more]")

# Sink transfer: in bare, position 0 is often the sink.
# With prefix, does position 0 of the prefix take over?
print(f"\n--- Sink transfer hypothesis ---")
for l in full_attn_layers:
    bare_data = probe_f_received['bare'][l]
    pref_data = probe_f_received['oracle_trunc'][l]
    if bare_data and pref_data:
        bare_first = np.mean([d['first_pos_val'] for d in bare_data])
        pref_prefix = np.mean([d['prefix_mean_recv'] for d in pref_data])
        pref_doc_first = np.mean([d['doc_mean_recv'] for d in pref_data])
        print(f"  Layer {l}: bare pos0={bare_first:.4f}, "
              f"prefixed prefix_mean={pref_prefix:.4f}, "
              f"prefixed doc_mean={pref_doc_first:.4f}")
""")


# ===== Cell 12: Synthesis + save =====
code(r"""# Cell 12: Synthesis + save results

print("=" * 70)
print("SYNTHESIS: ATTENTION MECHANISM PROBING RESULTS")
print("=" * 70)

# ---- NLL cross-reference with Exp 3D ----
bare_nlls = np.array(nll_scores['bare'])
oracle_nlls = np.array(nll_scores['oracle_trunc'])
oracle_benefit = bare_nlls - oracle_nlls

print(f"\n1. NLL CROSS-REFERENCE:")
print(f"   bare NLL:   {bare_nlls.mean():.4f}")
print(f"   oracle NLL: {oracle_nlls.mean():.4f}")
print(f"   headroom:   {oracle_benefit.mean():+.4f} (d={cohens_d(oracle_benefit):+.3f})")
print(f"   (Should match Exp 3D results)")

# ---- Probe A summary ----
print(f"\n2. PROBE A — ATTENTION MASS ON PREFIX:")
for l in full_attn_layers:
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        if probe_a_mass[c][l]:
            mass = np.array(probe_a_mass[c][l]).mean()
            cname = c.replace('_trunc', '')
            print(f"   Layer {l} {cname}: {mass:.1%} of doc attention -> prefix")

# ---- Probe B summary ----
print(f"\n3. PROBE B — ENTROPY CHANGE:")
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    increases = sum(1 for l in ATTN_LAYERS
                    if np.mean(probe_b_entropy[c][l]) > np.mean(probe_b_entropy['bare'][l]))
    direction = "INCREASES" if increases > len(ATTN_LAYERS)/2 else "DECREASES"
    print(f"   {c}: entropy {direction} in {increases}/{len(ATTN_LAYERS)} layers")

# ---- Probe C summary ----
print(f"\n4. PROBE C — DOC-DOC REDISTRIBUTION:")
for l in full_attn_layers:
    o_kl = np.mean(probe_c_kl['oracle_trunc'][l]) if probe_c_kl['oracle_trunc'][l] else 0
    r_kl = np.mean(probe_c_kl['random_matched_trunc'][l]) if probe_c_kl['random_matched_trunc'][l] else 0
    print(f"   Layer {l}: oracle KL={o_kl:.4f}, random KL={r_kl:.4f}, "
          f"ratio={o_kl/r_kl:.2f}x" if r_kl > 0 else f"   Layer {l}: oracle KL={o_kl:.4f}")

# ---- Probe D summary ----
print(f"\n5. PROBE D — SHIFT MAGNITUDE:")
last_l = HIDDEN_LAYERS[-1]
for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
    if probe_d_shift[c][last_l]:
        shift = np.mean(probe_d_shift[c][last_l])
        print(f"   Last layer ({last_l}) {c}: mean L2={shift:.4f}")
o_shift = np.mean(probe_d_shift['oracle_trunc'][last_l])
r_shift = np.mean(probe_d_shift['random_matched_trunc'][last_l])
if r_shift > 0:
    print(f"   Oracle/Random ratio: {o_shift/r_shift:.2f}x")

# ---- Probe E summary ----
print(f"\n6. PROBE E — SHIFT DIRECTION (structural vs semantic):")
for l in HIDDEN_LAYERS:
    cosines = []
    for c in ['random_matched_trunc', 'repeat_the_trunc']:
        if probe_e_cosine[c][l]:
            cosines.append(np.mean(probe_e_cosine[c][l]))
    if cosines:
        avg = np.mean(cosines)
        label = "STRUCTURAL" if avg > 0.7 else "MIXED" if avg > 0.3 else "SEMANTIC"
        print(f"   Layer {l}: mean cosine={avg:.4f} [{label}]")

# ---- Probe F summary ----
print(f"\n7. PROBE F — ATTENTION SINKS:")
for l in full_attn_layers:
    if (probe_f_received['bare'][l] and
            probe_f_received['oracle_trunc'][l]):
        bare_first = np.mean([d['first_pos_val']
                              for d in probe_f_received['bare'][l]])
        pref_prefix = np.mean([d['prefix_mean_recv']
                               for d in probe_f_received['oracle_trunc'][l]])
        print(f"   Layer {l}: bare sink={bare_first:.4f}, "
              f"prefix absorbs={pref_prefix:.4f}")

# ---- Overall interpretation ----
print(f"\n{'='*70}")
print(f"INTERPRETATION:")

# Determine dominant mechanism
# Check if shifts are structural (probe E cosine > 0.7)
mean_cosines = []
for c in ['random_matched_trunc', 'repeat_the_trunc']:
    for l in HIDDEN_LAYERS:
        if probe_e_cosine[c][l]:
            mean_cosines.append(np.mean(probe_e_cosine[c][l]))
overall_cos = np.mean(mean_cosines) if mean_cosines else 0

# Check if entropy increases or decreases
ent_increases = 0
ent_total = 0
for l in full_attn_layers:
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        bare_ent = np.mean(probe_b_entropy['bare'][l])
        cond_ent = np.mean(probe_b_entropy[c][l])
        ent_total += 1
        if cond_ent > bare_ent:
            ent_increases += 1
ent_direction = "increases" if ent_increases > ent_total / 2 else "decreases"

# Check prefix attention mass
mean_mass = []
for l in full_attn_layers:
    for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']:
        if probe_a_mass[c][l]:
            mean_mass.append(np.array(probe_a_mass[c][l]).mean())
overall_mass = np.mean(mean_mass) if mean_mass else 0

print(f"\n  Hypothesis 1 (Attention redistribution):")
print(f"    Prefix absorbs {overall_mass:.1%} of doc token attention (full-attn layers)")
print(f"    Entropy {ent_direction} -> {'dilution/regularization' if ent_direction == 'increases' else 'focusing'}")

print(f"\n  Hypothesis 2 (RoPE position shift):")
print(f"    If dominant, shift magnitude would scale with prefix length")
print(f"    and direction would differ by prefix type.")
print(f"    Shift direction cosine: {overall_cos:.4f} "
      f"({'consistent = not RoPE-driven' if overall_cos > 0.5 else 'divergent = possibly RoPE-driven'})")

print(f"\n  Hypothesis 3 (Representation regularization):")
o_last = np.mean(probe_d_shift['oracle_trunc'][last_l])
r_last = np.mean(probe_d_shift['random_matched_trunc'][last_l])
t_last = np.mean(probe_d_shift['repeat_the_trunc'][last_l])
print(f"    Shift magnitudes: oracle={o_last:.4f}, random={r_last:.4f}, "
      f"repeat={t_last:.4f}")
if r_last > 0:
    print(f"    Oracle/Random: {o_last/r_last:.2f}x (1.0 = purely structural)")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp03e_attention_probing',
    'model': MODEL_NAME,
    'dataset': 'neural-bridge/rag-dataset-12000',
    'n_samples': N_SAMPLES,
    'attn_implementation': 'eager',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'architecture': {
        'n_layers': n_layers,
        'full_attn_layers': full_attn_layers,
        'attn_probe_layers': ATTN_LAYERS,
        'hidden_probe_layers': HIDDEN_LAYERS,
    },
    'nll_crossref': {
        'bare_nll': float(bare_nlls.mean()),
        'oracle_nll': float(oracle_nlls.mean()),
        'oracle_d': float(cohens_d(oracle_benefit)),
    },
    'probe_a_prefix_mass': {
        c: {str(l): float(np.array(probe_a_mass[c][l]).mean())
            for l in full_attn_layers if probe_a_mass[c][l]}
        for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']
    },
    'probe_b_entropy': {
        c: {str(l): float(np.mean(probe_b_entropy[c][l]))
            for l in ATTN_LAYERS if probe_b_entropy[c][l]}
        for c in COND_NAMES
    },
    'probe_c_kl': {
        c: {str(l): float(np.mean(probe_c_kl[c][l]))
            for l in ATTN_LAYERS if probe_c_kl[c][l]}
        for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']
    },
    'probe_d_shift': {
        c: {str(l): float(np.mean(probe_d_shift[c][l]))
            for l in HIDDEN_LAYERS if probe_d_shift[c][l]}
        for c in ['oracle_trunc', 'random_matched_trunc', 'repeat_the_trunc']
    },
    'probe_e_cosine': {
        c: {str(l): float(np.mean(probe_e_cosine[c][l]))
            for l in HIDDEN_LAYERS if probe_e_cosine[c][l]}
        for c in ['random_matched_trunc', 'repeat_the_trunc']
    },
    'probe_e_overall_cosine': float(overall_cos),
    'interpretation': {
        'prefix_mass_pct': float(overall_mass),
        'entropy_direction': ent_direction,
        'shift_direction_cosine': float(overall_cos),
        'shift_structural': bool(overall_cos > 0.7),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, processor, tokenizer, encoder_text
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/03e/03e_attention_probing.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
