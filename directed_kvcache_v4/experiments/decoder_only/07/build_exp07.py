#!/usr/bin/env python3
# Build Exp 07: Prefix Shielding Against KV Cache Compression.
#
# Tests whether instruction prefixes (especially comprehend at L=64) produce
# document KV cache representations that are more robust to post-hoc compression
# (simulated quantization and H2O token eviction).
#
# Factorial design: 3 Conditioning x 6 Compression = 18 conditions
# Conditioning: bare, random_64, comprehend_64
# Compression: bf16, int8, int4, evict_80, evict_50, evict_20
#
# 4 datasets x 160 hard samples = 640 total
# Datasets: MS MARCO, SQuAD v2, DROP, HotpotQA
#
# Reuses hard samples from Exp 05 (same SEED=42, same hard selection by bare NLL).
#
# SEED=42, SCORING_KEY='bos_retained_compression_shielding_v07'

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/07", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 07: Prefix Shielding Against KV Cache Compression

## Hypothesis

The prefix acts as a structural attention buffer during encoding, causing document
tokens to settle into representations that are more robust to post-hoc compression.
After truncating the prefix, the remaining document cache should withstand aggressive
quantization and token eviction better than a bare cache.

If confirmed, this has direct production implications: pre-condition, truncate, compress,
and serve at a fraction of the memory cost.

## Design

### Factorial: 3 Conditioning x 6 Compression = 18 conditions

**Factor A — Cache Conditioning:**

| Level | Description |
|-------|-------------|
| `bare` | `[BOS] + doc` -> cache (no prefix) |
| `random_64` | Random token IDs L=64 -> select + reposition (structural-only control) |
| `comprehend_64` | Comprehend instruction L=64 -> select + reposition (best prefix) |

The `random_64` control is critical: if it also shields, the effect is **structural**
(attention sink). If only `comprehend_64` shields, it's **semantic**.

**Factor B — Compression Strategy:**

| Level | Description |
|-------|-------------|
| `bf16` | No compression (baseline) |
| `int8` | Simulated 8-bit per-tensor symmetric quantization of K+V |
| `int4` | Simulated 4-bit per-tensor symmetric quantization of K+V |
| `evict_80` | Keep 80% of doc tokens by H2O attention importance |
| `evict_50` | Keep 50% |
| `evict_20` | Keep 20% |

**Datasets**: 4 datasets x 160 hard samples = 640 total
- MS MARCO (weak prefix effect)
- SQuAD v2 (moderate)
- DROP (strong — discrete reasoning)
- HotpotQA (strong — multi-hop)

Hard samples reused from Exp 05 (same SEED=42, same hard selection by bare NLL).

## Key Metrics

1. **Delta NLL** = NLL(compressed) - NLL(bf16 same conditioning) — degradation from compression
2. **Shielding effect** = Delta\_NLL(bare) - Delta\_NLL(prefixed) — positive means prefix shields
3. **Structural vs semantic shield** = shielding(random\_64) vs shielding(comprehend\_64)
4. **Crossover test**: Does NLL(comprehend\_64, int4) < NLL(bare, bf16)? (production mandate)""")


# ===== Cell 2: Setup + Model + All Functions =====
code(r"""# Cell 2: Setup, model loading, and all functions
import os
os.umask(0o000)
import sys, json, time, gc, copy
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d
from lib.data import count_words

SEED = 42
N_SAMPLES = 400      # per dataset (same as Exp 05)
HARD_FRAC = 0.40     # top 40% by bare NLL
N_HARD = int(N_SAMPLES * HARD_FRAC)  # 160
PREFIX_L = 64
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp07")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")

DATASETS = ['ms_marco', 'squad_v2', 'drop', 'hotpotqa']

CONDITIONING_LEVELS = ['bare', 'random_64', 'comprehend_64']
COMPRESSION_LEVELS = ['bf16', 'int8', 'int4', 'evict_80', 'evict_50', 'evict_20']

# Per-dataset seeds (same as Exp 05)
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
}

SCORING_KEY = 'bos_retained_compression_shielding_v07'

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = model.get_input_embeddings().num_embeddings
cfg_vocab = getattr(text_cfg, 'vocab_size', None)
if cfg_vocab != VOCAB_SIZE:
    print(f"WARNING: config vocab_size={cfg_vocab} != embedding size={VOCAB_SIZE}")
    print(f"Using embedding size {VOCAB_SIZE} for random token generation")
rope_params = getattr(text_cfg, 'rope_parameters', {})
layer_types = getattr(text_cfg, 'layer_types', [])
N_LAYERS = len(layer_types)
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # 1023 for Gemma 3

NL = len(NEWLINE_IDS)
# With PREFIX_L=64: max_doc = 1023 - 1 - 64 - 1 = 957 > COMMON_MAX_DOC=765
# Use COMMON_MAX_DOC from Exp 05 so all conditionings produce the same D
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL  # 765 (same as Exp 05)

special_ids = set(tokenizer.all_special_ids)

print(f"Exp 07: Prefix Shielding Against KV Cache Compression")
print(f"N_SAMPLES: {N_SAMPLES}, HARD_FRAC: {HARD_FRAC}, N_HARD: {N_HARD}")
print(f"PREFIX_L: {PREFIX_L}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}, N_LAYERS: {N_LAYERS}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print(f"Conditioning: {CONDITIONING_LEVELS}")
print(f"Compression: {COMPRESSION_LEVELS}")
print(f"Datasets: {DATASETS}")

# --- RoPE repositioning helpers (reused from Exp 05/06) ---
def build_layer_inv_freqs():
    inv_freqs = {}
    for lt, params in rope_params.items():
        theta = params.get('rope_theta', 10000.0)
        dim = text_cfg.head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
        inv_freqs[lt] = inv_freq
    return inv_freqs

LAYER_INV_FREQS = build_layer_inv_freqs()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def select_kv_cache(cache, indices):
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def reposition_kv_cache(cache, old_positions, new_positions, bos_start=0):
    delta = new_positions - old_positions
    for L in range(len(cache.layers)):
        lt = layer_types[L]
        inv_freq = LAYER_INV_FREQS[lt]
        k = cache.layers[L].keys
        doc_keys = k[:, :, bos_start + 1:, :]
        freqs = torch.einsum('i,j->ij', delta.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_delta = emb.cos().to(k.dtype).unsqueeze(0).unsqueeze(0)
        sin_delta = emb.sin().to(k.dtype).unsqueeze(0).unsqueeze(0)
        doc_keys_new = doc_keys * cos_delta + rotate_half(doc_keys) * sin_delta
        cache.layers[L].keys = torch.cat([
            k[:, :, :bos_start + 1, :],
            doc_keys_new,
        ], dim=2)
    return cache


def make_prefix(token_ids, L):
    if len(token_ids) >= L:
        return token_ids[:L]
    padded = token_ids * ((L // max(len(token_ids), 1)) + 1)
    return padded[:L]


# --- Instruction definitions ---
INSTRUCTIONS = {
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
}
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")


# ===================================================================
# NEW FUNCTIONS FOR EXP 07
# ===================================================================

def deep_copy_cache(cache):
    # Clone all K/V tensors in a DynamicCache (compression is destructive).
    cloned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys.clone()
        v = cache.layers[i].values.clone()
        cloned.update(k, v, i)
    return cloned


def simulated_quantize(tensor, nbits):
    # Per-tensor symmetric quantization round-trip (simulated, returns bf16).
    # scale = absmax / (2^(nbits-1) - 1), round to int grid, clamp, dequantize.
    qmax = (1 << (nbits - 1)) - 1  # 127 for int8, 7 for int4
    absmax = tensor.abs().max()
    if absmax == 0:
        return tensor
    scale = absmax / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(tensor.dtype)


def quantize_kv_cache(cache, nbits):
    # Apply simulated_quantize to all layers K and V tensors in-place.
    # BOS entry (index 0) is preserved unquantized to maintain the attention sink.
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            # Preserve BOS (index 0), quantize doc entries (index 1+)
            k_bos = k[:, :, :1, :]
            k_doc = simulated_quantize(k[:, :, 1:, :], nbits)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)

            v_bos = v[:, :, :1, :]
            v_doc = simulated_quantize(v[:, :, 1:, :], nbits)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def _compute_importance(attentions, doc_start, D):
    # H2O importance: for each doc token j, sum attention it RECEIVES
    # across all layers, heads, and query positions.
    # attentions: tuple of (1, n_heads, seq_len, seq_len), one per layer
    # Returns np.array of shape (D,)
    importance = np.zeros(D, dtype=np.float64)
    for layer_attn in attentions:
        # layer_attn: (1, n_heads, seq_len, seq_len)
        attn = layer_attn[0]  # (n_heads, seq_len, seq_len)
        # Sum attention received by each doc token across all heads, all query positions
        # attn[h, q, k] = attention query q pays to key k
        # We want sum over h and q of attn[h, q, doc_start+j] for each doc token j
        doc_attn_received = attn[:, :, doc_start:doc_start + D]  # (n_heads, seq_len, D)
        importance += doc_attn_received.sum(dim=(0, 1)).cpu().numpy()  # (D,)
    return importance


def evict_kv_cache(cache, D, importance, keep_frac):
    # Keep top-K doc tokens by H2O importance, always keep BOS.
    # cache: DynamicCache with 1+D entries (BOS at 0, doc at 1..D)
    # Returns (new_cache, K) where K = number of retained doc tokens
    K = max(1, int(D * keep_frac))
    # Top-K doc token indices by importance, sorted to preserve document order
    top_k_doc_idx = np.sort(np.argsort(importance)[::-1][:K])
    # Map to cache indices: BOS is 0, doc tokens at 1..D
    keep_indices = [0] + [int(idx) + 1 for idx in top_k_doc_idx]
    new_cache = select_kv_cache(cache, keep_indices)
    # Reposition retained doc tokens to contiguous positions 1..K
    old_pos = torch.tensor([idx + 1 for idx in top_k_doc_idx],
                           dtype=torch.long, device=DEVICE)
    new_pos = torch.arange(1, K + 1, device=DEVICE)
    new_cache = reposition_kv_cache(new_cache, old_pos, new_pos, bos_start=0)
    return new_cache, K


def encode_phase_a(doc_text, prefix_token_ids=None, output_attentions=False):
    # Phase A: encode document with optional prefix, return cache + metadata.
    # Returns (cache, D, doc_ids, attentions, doc_start)
    # cache: DynamicCache with 1+D entries (BOS + doc, prefix truncated)
    # attentions: tuple of attention tensors if output_attentions=True, else None
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]

    if prefix_token_ids is not None:
        P = len(prefix_token_ids)
        _NL = len(NEWLINE_IDS)
        max_doc = SLIDING_CACHE_LIMIT - 1 - P - _NL
        if len(doc_ids) > max_doc:
            doc_ids = doc_ids[:max_doc]
        D = len(doc_ids)
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True, output_attentions=output_attentions)
        cache = pa.past_key_values
        attentions = pa.attentions if output_attentions else None
        del pa
        # Select BOS + doc entries, skip prefix + newline
        keep_indices = [0] + list(range(1 + P + _NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)
        old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
        # Remap attention doc_start for importance computation
        doc_start = 1 + P + _NL  # in the original sequence
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True, output_attentions=output_attentions)
        cache = pa.past_key_values
        attentions = pa.attentions if output_attentions else None
        del pa
        doc_start = 1  # doc starts at position 1 (after BOS)

    return cache, D, doc_ids, attentions, doc_start


def score_phase_b(cache, D_effective, query_text, answer_text):
    # Phase B: score query+answer against a pre-built cache.
    # cache: DynamicCache with 1+D_effective entries
    # Returns mean NLL of answer tokens
    phase_b_start = D_effective + 1
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            use_cache=False,
        )

    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del pb
    return nll


def compress_and_score(cache, D, importance, compression, query_text, answer_text):
    # Deep-copy cache, apply compression, score Phase B, return NLL.
    # cache: original DynamicCache (not modified)
    # importance: np.array of shape (D,) or None (only needed for eviction)
    if compression == 'bf16':
        # No compression needed — score directly on original cache
        return score_phase_b(cache, D, query_text, answer_text)

    # Deep copy for destructive operations
    c = deep_copy_cache(cache)

    if compression == 'int8':
        quantize_kv_cache(c, 8)
        nll = score_phase_b(c, D, query_text, answer_text)
    elif compression == 'int4':
        quantize_kv_cache(c, 4)
        nll = score_phase_b(c, D, query_text, answer_text)
    elif compression.startswith('evict_'):
        frac = int(compression.split('_')[1]) / 100.0
        c, K = evict_kv_cache(c, D, importance, frac)
        nll = score_phase_b(c, K, query_text, answer_text)
    else:
        raise ValueError(f"Unknown compression: {compression}")

    del c
    return nll


print(f"\nSetup complete.")
print(f"Functions: encode_phase_a, score_phase_b, compress_and_score")
print(f"  deep_copy_cache, simulated_quantize, quantize_kv_cache")
print(f"  _compute_importance, evict_kv_cache")
print(f"  select_kv_cache, reposition_kv_cache, make_prefix")
""")


# ===== Cell 3: Load Datasets + Exp 05 Hard Samples =====
code(r"""# Cell 3: Load 4 datasets + Exp 05 hard samples
from datasets import load_dataset

print("=" * 70)
print("LOADING 4 DATASETS + EXP 05 HARD SAMPLES")
print("=" * 70)

hard_samples = {}   # ds_name -> list of hard sample dicts
all_samples = {}    # ds_name -> list of all N_SAMPLES sample dicts
exp05_nlls = {}     # ds_name -> dict of NLL arrays from Exp 05

# ================================================================
# MS MARCO (from Exp 02)
# ================================================================
print("\n--- MS MARCO ---")
assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES

msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])

# Reload passage text
print("  Reloading MS MARCO v1.1 for passage text...")
ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES:
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
            msmarco_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES]
msmarco_all = [msmarco_candidates[i] for i in indices]
del ds_msmarco, msmarco_candidates
gc.collect()

# Verify alignment
for i in range(min(20, N_SAMPLES)):
    assert msmarco_all[i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"
print("  MS MARCO alignment verified")

hs_msmarco = []
for idx in msmarco_hard_idx:
    s = dict(msmarco_all[idx])
    s['nll_bare'] = float(msmarco_bare[idx])
    s['original_idx'] = int(idx)
    hs_msmarco.append(s)
hard_samples['ms_marco'] = hs_msmarco
all_samples['ms_marco'] = msmarco_all
print(f"  MS MARCO: {N_HARD} hard samples")

del exp02_ckpt, exp02_results
gc.collect()

# ================================================================
# SQuAD 2.0
# ================================================================
print("\n--- SQuAD 2.0 ---")
ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")
squad_candidates = []
for item in ds_squad:
    answers = item.get('answers', {})
    answer_texts = answers.get('text', [])
    if not answer_texts:
        continue
    passage = item['context']
    query = item['question']
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        squad_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['squad_v2'])
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
all_samples['squad_v2'] = [squad_candidates[i] for i in sq_indices]
del ds_squad, squad_candidates
gc.collect()

# ================================================================
# DROP
# ================================================================
print("\n--- DROP ---")
ds_drop = load_dataset("ucinlp/drop", split="validation")
drop_candidates = []
for item in ds_drop:
    passage = item['passage']
    question = item['question']
    answers_spans = item.get('answers_spans', {})
    spans = answers_spans.get('spans', [])
    if not spans or not spans[0]:
        continue
    answer = spans[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        drop_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['drop'])
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES]
all_samples['drop'] = [drop_candidates[i] for i in drop_indices]
del ds_drop, drop_candidates
gc.collect()

# ================================================================
# HotpotQA
# ================================================================
print("\n--- HotpotQA ---")
ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
hotpot_candidates = []
for item in ds_hotpot:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    sf_titles = sf.get('title', [])
    sf_sent_ids = sf.get('sent_id', [])
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
    passage_parts = []
    for title, sid in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents and sid < len(title_to_sents[title]):
            passage_parts.append(title_to_sents[title][sid])
    if not passage_parts:
        continue
    passage = ' '.join(passage_parts)
    query = item['question']
    answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        hotpot_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['hotpotqa'])
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# ================================================================
# Load Exp 05 bare NLLs for hard selection (non-MS-MARCO datasets)
# ================================================================
print("\n--- Loading Exp 05 bare NLLs for hard selection ---")
for ds_name in ['squad_v2', 'drop', 'hotpotqa']:
    samples_ds = all_samples[ds_name]
    bare_path = EXP03_DIR / f"bare_{ds_name}.json" if ds_name != 'drop' else \
                EXP05_DIR / f"bare_{ds_name}.json"

    # SQuAD and HotpotQA bare NLLs from Exp 03, DROP from Exp 05
    if ds_name in ('squad_v2', 'hotpotqa'):
        bare_path = EXP03_DIR / f"bare_{ds_name}.json"
    else:
        bare_path = EXP05_DIR / f"bare_{ds_name}.json"

    bare_ckpt = json.loads(bare_path.read_text())
    bare_nlls_all = bare_ckpt['bare_nlls']

    # Verify alignment
    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in samples_ds[:len(saved_queries)]]
    assert saved_queries == current_queries, \
        f"{ds_name}: query alignment mismatch"

    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])

    hs = []
    for idx in h_idx:
        s = dict(samples_ds[idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        hs.append(s)
    hard_samples[ds_name] = hs
    print(f"  {ds_name}: {N_HARD} hard, mean bare NLL: {bare_arr[h_idx].mean():.4f}")

del bare_ckpt
gc.collect()

# ================================================================
# Load Exp 05 Phase B NLLs for validation baselines
# ================================================================
print("\n--- Loading Exp 05 Phase B baselines for validation ---")
for ds_name in DATASETS:
    pb_path = EXP05_DIR / f"phaseB_{ds_name}.json"
    if pb_path.exists():
        pb_ckpt = json.loads(pb_path.read_text())
        pb_results = pb_ckpt['results']
        exp05_nlls[ds_name] = {
            'bare_trunc': np.array([r['nll_bare_trunc'] for r in pb_results]),
            'comprehend_64': np.array([r['nll_comprehend_64'] for r in pb_results]),
        }
        print(f"  {ds_name}: loaded {len(pb_results)} Phase B results")
    else:
        print(f"  {ds_name}: no Phase B results found at {pb_path}")

gc.collect()

# Summary
print("\n" + "=" * 70)
print("Dataset loading summary:")
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    print(f"  {ds_name}: {n_h} hard samples")
""")


# ===== Cell 4: Main Scoring Loop =====
code(r"""# Cell 4: Main scoring loop — 3 conditioning x 6 compression x 4 datasets x 160 samples
print("=" * 70)
print("MAIN SCORING LOOP")
print(f"  {len(DATASETS)} datasets x {N_HARD} samples x "
      f"{len(CONDITIONING_LEVELS)} conditionings x {len(COMPRESSION_LEVELS)} compressions")
print(f"  = {len(DATASETS) * N_HARD * len(CONDITIONING_LEVELS) * len(COMPRESSION_LEVELS)} total scorings")
print("=" * 70)

# Pre-build prefix token IDs
comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

# Validation: quick sanity check on tiny example
print("\n--- Validation: bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"
doc_ids_t = tokenizer(doc_text_t, add_special_tokens=False).input_ids
D_t = len(doc_ids_t)
query_ids_t = tokenizer("\n" + query_text_t + "\n", add_special_tokens=False).input_ids
answer_ids_t = tokenizer(answer_text_t, add_special_tokens=False).input_ids

full_ids = [BOS_ID] + doc_ids_t + query_ids_t + answer_ids_t
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D_t + len(query_ids_t)
logits_full = out_full.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids_t), :].float()
targets_t = torch.tensor(answer_ids_t, device=DEVICE)
nll_single = -F.log_softmax(logits_full, dim=-1).gather(
    1, targets_t.unsqueeze(1)).squeeze(1).mean().item()
del out_full

cache_t, D_t2, _, _, _ = encode_phase_a(doc_text_t)
nll_twophase = score_phase_b(cache_t, D_t2, query_text_t, answer_text_t)
del cache_t
diff_pct = abs(nll_single - nll_twophase) / nll_single * 100
print(f"  Single-pass: {nll_single:.6f}, Two-phase: {nll_twophase:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"

# Validate quantize round-trip doesn't crash
cache_v, D_v, _, _, _ = encode_phase_a(doc_text_t)
c_q = deep_copy_cache(cache_v)
quantize_kv_cache(c_q, 8)
nll_q = score_phase_b(c_q, D_v, query_text_t, answer_text_t)
print(f"  Quantized int8 NLL: {nll_q:.6f} (diff from bf16: {nll_q - nll_twophase:+.6f})")
del c_q

# Validate eviction with dummy importance
cache_v2, D_v2, _, attn_v, ds_v = encode_phase_a(doc_text_t, output_attentions=True)
imp_v = _compute_importance(attn_v, ds_v, D_v2)
del attn_v
c_e = deep_copy_cache(cache_v2)
c_e, K_e = evict_kv_cache(c_e, D_v2, imp_v, 0.5)
nll_e = score_phase_b(c_e, K_e, query_text_t, answer_text_t)
print(f"  Evict 50% NLL: {nll_e:.6f} (kept {K_e}/{D_v2} tokens)")
del c_e, cache_v, cache_v2
gc.collect()
torch.cuda.empty_cache()
print("  PASSED all validation checks\n")

# ================================================================
# Main loop
# ================================================================
all_results = {}  # ds_name -> list of result dicts

for ds_name in DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"

    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name} ({n_hard} hard samples)")
    print(f"{'='*70}")

    ds_results = []
    start_idx = 0

    # Resume from checkpoint
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if (ckpt.get('dataset') == ds_name and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('n_hard') == n_hard):
            saved_queries = [r['query'][:50] for r in ckpt.get('results', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                ds_results = ckpt['results']
                start_idx = len(ds_results)
                print(f"  Resuming from checkpoint: {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()

        # Reset RNG for reproducible random prefixes
        np.random.seed(DS_SEEDS.get(ds_name, SEED) + 7000)
        pyrandom.seed(DS_SEEDS.get(ds_name, SEED) + 7000)

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Exp07 {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
            }

            # Generate random prefix for this sample
            rand_ids = []
            while len(rand_ids) < PREFIX_L:
                tid = np.random.randint(0, VOCAB_SIZE)
                if tid not in special_ids:
                    rand_ids.append(int(tid))
            rand_prefix = rand_ids[:PREFIX_L]

            # === Phase A: 3 conditioning passes (with output_attentions=True) ===
            conditioning_caches = {}
            conditioning_D = {}
            conditioning_importance = {}

            for cond in CONDITIONING_LEVELS:
                if cond == 'bare':
                    prefix_ids = None
                elif cond == 'random_64':
                    prefix_ids = rand_prefix
                elif cond == 'comprehend_64':
                    prefix_ids = comprehend_prefix
                else:
                    raise ValueError(f"Unknown conditioning: {cond}")

                cache, D, doc_ids, attentions, doc_start = encode_phase_a(
                    s['passage'], prefix_token_ids=prefix_ids,
                    output_attentions=True)

                # Compute importance from attentions immediately, then delete
                importance = _compute_importance(attentions, doc_start, D)
                del attentions
                torch.cuda.empty_cache()

                conditioning_caches[cond] = cache
                conditioning_D[cond] = D
                conditioning_importance[cond] = importance
                result[f'D_{cond}'] = D

            # === Phase B: 6 compression levels per conditioning ===
            for cond in CONDITIONING_LEVELS:
                cache = conditioning_caches[cond]
                D = conditioning_D[cond]
                importance = conditioning_importance[cond]

                for comp in COMPRESSION_LEVELS:
                    key = f'nll_{cond}_{comp}'
                    nll = compress_and_score(
                        cache, D, importance, comp,
                        s['query'], s['answer'])
                    result[key] = nll

            # Cleanup caches for this sample
            for cond in CONDITIONING_LEVELS:
                del conditioning_caches[cond]
            del conditioning_caches, conditioning_D, conditioning_importance
            gc.collect()
            torch.cuda.empty_cache()

            ds_results.append(result)

            # Checkpoint every 20 samples
            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'conditioning_levels': CONDITIONING_LEVELS,
                    'compression_levels': COMPRESSION_LEVELS,
                    'prefix_l': PREFIX_L,
                    'common_max_doc': COMMON_MAX_DOC,
                    'results': ds_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Scoring complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    all_results[ds_name] = ds_results

gc.collect()
torch.cuda.empty_cache()
print(f"\nAll scoring complete.")
""")


# ===== Cell 5: Validation Against Exp 05 =====
code(r"""# Cell 5: Validation against Exp 05 baselines
print("=" * 70)
print("VALIDATION AGAINST EXP 05")
print("=" * 70)

for ds_name in DATASETS:
    if ds_name not in exp05_nlls:
        print(f"\n  {ds_name}: no Exp 05 baselines available, skipping")
        continue

    results = all_results[ds_name]
    n = len(results)

    # Compare bare bf16 vs Exp 05 bare_trunc
    exp07_bare = np.array([r['nll_bare_bf16'] for r in results])
    exp05_bare = exp05_nlls[ds_name]['bare_trunc'][:n]
    diff_bare = np.abs(exp07_bare - exp05_bare)
    print(f"\n  {ds_name} — bare_bf16 vs Exp05 bare_trunc:")
    print(f"    Max abs diff: {diff_bare.max():.6f}")
    print(f"    Mean abs diff: {diff_bare.mean():.6f}")
    if diff_bare.max() > 0.001:
        print(f"    WARNING: max diff exceeds 0.001!")
    else:
        print(f"    PASSED (< 0.001)")

    # Compare comprehend_64 bf16 vs Exp 05 comprehend_64
    exp07_comp = np.array([r['nll_comprehend_64_bf16'] for r in results])
    exp05_comp = exp05_nlls[ds_name]['comprehend_64'][:n]
    diff_comp = np.abs(exp07_comp - exp05_comp)
    print(f"  {ds_name} — comprehend_64_bf16 vs Exp05 comprehend_64:")
    print(f"    Max abs diff: {diff_comp.max():.6f}")
    print(f"    Mean abs diff: {diff_comp.mean():.6f}")
    if diff_comp.max() > 0.001:
        print(f"    WARNING: max diff exceeds 0.001!")
    else:
        print(f"    PASSED (< 0.001)")
""")


# ===== Cell 6: Per-Dataset Analysis =====
code(r"""# Cell 6: Per-dataset analysis — NLL tables, Delta NLL, shielding, crossover
print("=" * 70)
print("PER-DATASET ANALYSIS")
print("=" * 70)

per_dataset_analysis = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    # --- 3x6 NLL table ---
    print(f"\n  Mean NLL (3 conditioning x 6 compression):")
    print(f"  {'Conditioning':<18}", end="")
    for comp in COMPRESSION_LEVELS:
        print(f" {comp:>10}", end="")
    print()
    print(f"  {'-'*78}")

    nll_arrays = {}  # (cond, comp) -> np.array
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for comp in COMPRESSION_LEVELS:
            key = f'nll_{cond}_{comp}'
            arr = np.array([r[key] for r in results])
            nll_arrays[(cond, comp)] = arr
            row += f" {arr.mean():>10.4f}"
        print(row)

    # --- Delta NLL table (compression - bf16, same conditioning) ---
    print(f"\n  Delta NLL (= NLL_compressed - NLL_bf16, same conditioning):")
    comp_only = [c for c in COMPRESSION_LEVELS if c != 'bf16']
    print(f"  {'Conditioning':<18}", end="")
    for comp in comp_only:
        print(f" {comp:>10}", end="")
    print()
    print(f"  {'-'*68}")

    delta_arrays = {}
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = nll_arrays[(cond, 'bf16')]
        for comp in comp_only:
            delta = nll_arrays[(cond, comp)] - bf16
            delta_arrays[(cond, comp)] = delta
            row += f" {delta.mean():>+10.4f}"
        print(row)

    # --- Shielding effect table ---
    print(f"\n  Shielding effect (= Delta_bare - Delta_prefixed, positive = prefix shields):")
    print(f"  {'Prefix':<18}", end="")
    for comp in comp_only:
        print(f" {'shield':>8} {'d':>7} {'p':>10}", end="")
    print()
    print(f"  {'-'*100}")

    ds_analysis = {'nll_means': {}, 'delta_means': {}, 'shielding': {}}
    for cond in CONDITIONING_LEVELS:
        for comp in COMPRESSION_LEVELS:
            ds_analysis['nll_means'][f'{cond}_{comp}'] = float(nll_arrays[(cond, comp)].mean())
        for comp in comp_only:
            ds_analysis['delta_means'][f'{cond}_{comp}'] = float(delta_arrays[(cond, comp)].mean())

    for cond in ['random_64', 'comprehend_64']:
        row = f"  {cond:<18}"
        for comp in comp_only:
            shield = delta_arrays[('bare', comp)] - delta_arrays[(cond, comp)]
            d = cohens_d(shield)
            _, p = stats.ttest_1samp(shield, 0)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            row += f" {shield.mean():>+8.4f} {d:>+7.3f} {p:>10.2e}"
            ds_analysis['shielding'][f'{cond}_{comp}'] = {
                'mean': float(shield.mean()), 'd': float(d), 'p': float(p),
            }
        print(row)

    # --- Crossover check ---
    print(f"\n  Crossover test: NLL(comprehend_64, compressed) < NLL(bare, bf16)?")
    bare_bf16 = nll_arrays[('bare', 'bf16')]
    for comp in COMPRESSION_LEVELS:
        comp_nll = nll_arrays[('comprehend_64', comp)]
        wins = np.mean(comp_nll < bare_bf16)
        mean_diff = (comp_nll - bare_bf16).mean()
        print(f"    comprehend_64_{comp} vs bare_bf16: "
              f"win rate {wins:.1%}, mean diff {mean_diff:+.4f}")
        ds_analysis[f'crossover_{comp}'] = {
            'win_rate': float(wins), 'mean_diff': float(mean_diff),
        }

    per_dataset_analysis[ds_name] = ds_analysis

gc.collect()
""")


# ===== Cell 7: Cross-Dataset Pooled Analysis =====
code(r"""# Cell 7: Cross-dataset pooled analysis
print("=" * 70)
print("CROSS-DATASET POOLED ANALYSIS")
print("=" * 70)

comp_only = [c for c in COMPRESSION_LEVELS if c != 'bf16']

# ================================================================
# PART 1: Inverse-variance weighted meta-analysis of shielding d
# ================================================================
print("\n--- PART 1: Shielding Meta-Analysis (4 datasets) ---")

meta_shielding = {}
for cond in ['random_64', 'comprehend_64']:
    print(f"\n  {cond}:")
    print(f"  {'Compression':<12} {'pooled_d':>9} {'SE':>8} {'z':>8} "
          f"{'p':>10} {'95% CI':>16} {'sig':>4}")
    print(f"  {'-'*76}")

    for comp in comp_only:
        ds_effects = []
        for ds_name in DATASETS:
            results = all_results[ds_name]
            bf16_bare = np.array([r['nll_bare_bf16'] for r in results])
            comp_bare = np.array([r[f'nll_bare_{comp}'] for r in results])
            bf16_pref = np.array([r[f'nll_{cond}_bf16'] for r in results])
            comp_pref = np.array([r[f'nll_{cond}_{comp}'] for r in results])

            delta_bare = comp_bare - bf16_bare
            delta_pref = comp_pref - bf16_pref
            shield = delta_bare - delta_pref

            n = len(shield)
            d = cohens_d(shield)
            se = np.sqrt(1.0/n + d**2 / (2.0*n))
            ds_effects.append((d, se, n, ds_name))

        weights = [1.0 / (se**2) for _, se, _, _ in ds_effects]
        w_sum = sum(weights)
        pooled_d = sum(w * d for (d, _, _, _), w in zip(ds_effects, weights)) / w_sum
        pooled_se = 1.0 / np.sqrt(w_sum)
        z = pooled_d / pooled_se if pooled_se > 0 else 0.0
        p = 2 * stats.norm.sf(abs(z))
        ci_lo = pooled_d - 1.96 * pooled_se
        ci_hi = pooled_d + 1.96 * pooled_se
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        print(f"  {comp:<12} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
              f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4}")

        meta_shielding[f'{cond}_{comp}'] = {
            'pooled_d': float(pooled_d), 'se': float(pooled_se),
            'z': float(z), 'p': float(p),
            'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
            'per_dataset': {ds: float(d) for d, _, _, ds in ds_effects},
        }

# ================================================================
# PART 2: Structural vs semantic shield decomposition
# ================================================================
print(f"\n--- PART 2: Structural vs Semantic Shield ---")
print(f"  {'Compression':<12} {'struct_d':>10} {'semantic_d':>11} "
      f"{'ratio':>7} {'interpretation':>20}")
print(f"  {'-'*68}")

struct_vs_semantic = {}
for comp in comp_only:
    struct_d = meta_shielding.get(f'random_64_{comp}', {}).get('pooled_d', 0)
    semantic_d = meta_shielding.get(f'comprehend_64_{comp}', {}).get('pooled_d', 0)
    if abs(semantic_d) > 0.001:
        ratio = struct_d / semantic_d
    else:
        ratio = float('inf')

    if struct_d > 0.1 and semantic_d > 0.1:
        if ratio > 0.8:
            interp = "structural"
        elif ratio > 0.3:
            interp = "mixed"
        else:
            interp = "semantic"
    elif semantic_d > 0.1:
        interp = "semantic"
    elif struct_d > 0.1:
        interp = "structural"
    else:
        interp = "none"

    print(f"  {comp:<12} {struct_d:>+10.4f} {semantic_d:>+11.4f} "
          f"{ratio:>7.2f} {interp:>20}")
    struct_vs_semantic[comp] = {
        'structural_d': float(struct_d), 'semantic_d': float(semantic_d),
        'ratio': float(ratio), 'interpretation': interp,
    }

# ================================================================
# PART 3: Dose-response (shielding vs eviction aggressiveness)
# ================================================================
print(f"\n--- PART 3: Dose-Response (eviction aggressiveness) ---")
evict_comps = ['evict_80', 'evict_50', 'evict_20']
evict_fracs = [0.80, 0.50, 0.20]

print(f"  {'Conditioning':<18}", end="")
for comp in evict_comps:
    print(f" {comp:>10}", end="")
print(f"  {'trend':>8}")
print(f"  {'-'*60}")

for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    delta_means = []
    for comp in evict_comps:
        # Pooled mean delta NLL across datasets
        deltas = []
        for ds_name in DATASETS:
            results = all_results[ds_name]
            bf16 = np.array([r[f'nll_{cond}_bf16'] for r in results])
            comp_arr = np.array([r[f'nll_{cond}_{comp}'] for r in results])
            deltas.append((comp_arr - bf16).mean())
        mean_delta = np.mean(deltas)
        delta_means.append(mean_delta)
        row += f" {mean_delta:>+10.4f}"
    rho, _ = stats.spearmanr([80, 50, 20], delta_means)
    row += f"  {'UP' if rho > 0.5 else 'DOWN' if rho < -0.5 else 'FLAT':>8}"
    print(row)

# ================================================================
# PART 4: Quantization vs eviction comparison
# ================================================================
print(f"\n--- PART 4: Quantization vs Eviction ---")
print(f"  Pooled mean Delta NLL (higher = worse degradation):")
print(f"  {'Conditioning':<18} {'int8':>8} {'int4':>8} {'evict80':>8} "
      f"{'evict50':>8} {'evict20':>8}")
print(f"  {'-'*60}")

quant_vs_evict = {}
for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    cond_data = {}
    for comp in comp_only:
        deltas = []
        for ds_name in DATASETS:
            results = all_results[ds_name]
            bf16 = np.array([r[f'nll_{cond}_bf16'] for r in results])
            comp_arr = np.array([r[f'nll_{cond}_{comp}'] for r in results])
            deltas.append((comp_arr - bf16).mean())
        mean_delta = np.mean(deltas)
        row += f" {mean_delta:>+8.4f}"
        cond_data[comp] = float(mean_delta)
    print(row)
    quant_vs_evict[cond] = cond_data

# ================================================================
# PART 5: Crossover rate
# ================================================================
print(f"\n--- PART 5: Crossover Rate (% where compressed-prefixed < uncompressed-bare) ---")
print(f"  {'Compression':<12}", end="")
for ds_name in DATASETS:
    print(f" {ds_name[:8]:>10}", end="")
print(f"  {'pooled':>8}")
print(f"  {'-'*60}")

crossover_rates = {}
for comp in COMPRESSION_LEVELS:
    row = f"  {comp:<12}"
    ds_rates = []
    for ds_name in DATASETS:
        results = all_results[ds_name]
        bare_bf16 = np.array([r['nll_bare_bf16'] for r in results])
        comp_pref = np.array([r[f'nll_comprehend_64_{comp}'] for r in results])
        rate = float(np.mean(comp_pref < bare_bf16))
        ds_rates.append(rate)
        row += f" {rate:>9.1%}"
    pooled = np.mean(ds_rates)
    row += f"  {pooled:>7.1%}"
    print(row)
    crossover_rates[comp] = {
        'per_dataset': {ds: r for ds, r in zip(DATASETS, ds_rates)},
        'pooled': float(pooled),
    }
""")


# ===== Cell 8: Mechanism Analysis =====
code(r"""# Cell 8: Mechanism analysis — attention distribution and quantization error
print("=" * 70)
print("MECHANISM ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Attention importance distribution analysis
# ================================================================
print("\n--- PART 1: Re-encode subsample for attention analysis ---")
# Re-encode a subsample (first 40 hard samples from DROP, the standout dataset)
analysis_ds = 'drop'
analysis_samples = hard_samples[analysis_ds][:40]
n_analysis = len(analysis_samples)

np.random.seed(SEED + 9000)
pyrandom.seed(SEED + 9000)

importance_data = {cond: [] for cond in CONDITIONING_LEVELS}
quant_error_data = {cond: {'int8': [], 'int4': []} for cond in CONDITIONING_LEVELS}

comprehend_prefix_a = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

for i in tqdm(range(n_analysis), desc="Mechanism analysis"):
    s = analysis_samples[i]

    # Random prefix for this sample
    rand_ids = []
    while len(rand_ids) < PREFIX_L:
        tid = np.random.randint(0, VOCAB_SIZE)
        if tid not in special_ids:
            rand_ids.append(int(tid))
    rand_prefix = rand_ids[:PREFIX_L]

    for cond in CONDITIONING_LEVELS:
        if cond == 'bare':
            prefix_ids = None
        elif cond == 'random_64':
            prefix_ids = rand_prefix
        elif cond == 'comprehend_64':
            prefix_ids = comprehend_prefix_a
        else:
            continue

        cache, D, doc_ids, attentions, doc_start = encode_phase_a(
            s['passage'], prefix_token_ids=prefix_ids,
            output_attentions=True)
        importance = _compute_importance(attentions, doc_start, D)
        del attentions
        torch.cuda.empty_cache()

        # Importance distribution stats
        if D > 0:
            imp_norm = importance / (importance.sum() + 1e-10)
            entropy = -np.sum(imp_norm * np.log(imp_norm + 1e-10))
            max_share = imp_norm.max()
            top10_share = np.sort(imp_norm)[::-1][:max(1, D // 10)].sum()
        else:
            entropy = 0.0
            max_share = 0.0
            top10_share = 0.0
        importance_data[cond].append({
            'entropy': float(entropy),
            'max_share': float(max_share),
            'top10_share': float(top10_share),
            'D': D,
        })

        # Quantization error per layer
        for nbits_label, nbits in [('int8', 8), ('int4', 4)]:
            layer_errors = []
            for layer_idx in range(N_LAYERS):
                k_orig = cache.layers[layer_idx].keys[:, :, 1:, :]  # skip BOS
                v_orig = cache.layers[layer_idx].values[:, :, 1:, :]
                k_quant = simulated_quantize(k_orig, nbits)
                v_quant = simulated_quantize(v_orig, nbits)
                k_err = (k_orig - k_quant).float().norm().item()
                v_err = (v_orig - v_quant).float().norm().item()
                layer_errors.append(k_err + v_err)
            quant_error_data[cond][nbits_label].append(layer_errors)

        del cache
    gc.collect()
    torch.cuda.empty_cache()

# ================================================================
# Print importance distribution stats
# ================================================================
print(f"\n--- Attention Importance Distribution ({analysis_ds}, n={n_analysis}) ---")
print(f"  {'Conditioning':<18} {'entropy':>9} {'max_share':>11} {'top10_share':>12}")
print(f"  {'-'*54}")
for cond in CONDITIONING_LEVELS:
    data = importance_data[cond]
    ent_mean = np.mean([d['entropy'] for d in data])
    max_mean = np.mean([d['max_share'] for d in data])
    top10_mean = np.mean([d['top10_share'] for d in data])
    print(f"  {cond:<18} {ent_mean:>9.3f} {max_mean:>11.4f} {top10_mean:>12.4f}")

# ================================================================
# Print quantization error stats
# ================================================================
print(f"\n--- Quantization Error by Layer Group ({analysis_ds}, n={n_analysis}) ---")
for nbits_label in ['int8', 'int4']:
    print(f"\n  {nbits_label}:")
    for cond in CONDITIONING_LEVELS:
        errors = np.array(quant_error_data[cond][nbits_label])  # (n_analysis, N_LAYERS)
        mean_per_layer = errors.mean(axis=0)
        # Group: first 12, 12-24, 24-36, 36-48
        groups = [(0, 12), (12, 24), (24, 36), (36, 48)]
        group_means = [mean_per_layer[a:b].mean() for a, b in groups]
        total_mean = mean_per_layer.mean()
        print(f"    {cond:<18} layers[0:12]={group_means[0]:.2f} "
              f"[12:24]={group_means[1]:.2f} [24:36]={group_means[2]:.2f} "
              f"[36:48]={group_means[3]:.2f} total={total_mean:.2f}")

# ================================================================
# Token importance overlap
# ================================================================
print(f"\n--- Token Importance Overlap (bare vs comprehend_64) ---")
# Re-run on same samples to get paired importance vectors
np.random.seed(SEED + 9000)
pyrandom.seed(SEED + 9000)
overlaps = []
for i in range(n_analysis):
    s = analysis_samples[i]
    # Bare
    cache_b, D_b, _, attn_b, ds_b = encode_phase_a(
        s['passage'], output_attentions=True)
    imp_b = _compute_importance(attn_b, ds_b, D_b)
    del attn_b, cache_b

    # Skip random prefix RNG to keep aligned
    rand_ids = []
    while len(rand_ids) < PREFIX_L:
        tid = np.random.randint(0, VOCAB_SIZE)
        if tid not in special_ids:
            rand_ids.append(int(tid))

    # Comprehend
    cache_c, D_c, _, attn_c, ds_c = encode_phase_a(
        s['passage'], prefix_token_ids=comprehend_prefix_a,
        output_attentions=True)
    imp_c = _compute_importance(attn_c, ds_c, D_c)
    del attn_c, cache_c
    torch.cuda.empty_cache()

    # Both should have same D (COMMON_MAX_DOC truncation)
    D = min(D_b, D_c)
    if D > 0:
        # Top-20% overlap
        k = max(1, D // 5)
        top_b = set(np.argsort(imp_b[:D])[::-1][:k])
        top_c = set(np.argsort(imp_c[:D])[::-1][:k])
        overlap = len(top_b & top_c) / k
        overlaps.append(overlap)

mean_overlap = np.mean(overlaps) if overlaps else 0
print(f"  Top-20% token overlap: {mean_overlap:.3f} (1.0 = identical, 0.0 = disjoint)")
print(f"  n={len(overlaps)} samples from {analysis_ds}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 9: Save results.json =====
code(r"""# Cell 9: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Collect all per-sample results
all_sample_results = {}
for ds_name in DATASETS:
    all_sample_results[ds_name] = all_results[ds_name]

final_results = {
    'experiment': 'v4_exp07_prefix_shielding_compression',
    'model': MODEL_NAME,
    'scoring': SCORING_KEY,
    'seed': SEED,
    'n_samples_per_dataset': N_SAMPLES,
    'n_hard_per_dataset': N_HARD,
    'hard_fraction': HARD_FRAC,
    'prefix_length': PREFIX_L,
    'common_max_doc': COMMON_MAX_DOC,
    'conditioning_levels': CONDITIONING_LEVELS,
    'compression_levels': COMPRESSION_LEVELS,
    'datasets': DATASETS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'per_dataset_analysis': per_dataset_analysis,
    'meta_shielding': meta_shielding,
    'structural_vs_semantic': struct_vs_semantic,
    'quantization_vs_eviction': quant_vs_evict,
    'crossover_rates': crossover_rates,
    'mechanism_analysis': {
        'importance_distribution': {
            cond: {
                'mean_entropy': float(np.mean([d['entropy'] for d in importance_data[cond]])),
                'mean_max_share': float(np.mean([d['max_share'] for d in importance_data[cond]])),
                'mean_top10_share': float(np.mean([d['top10_share'] for d in importance_data[cond]])),
            } for cond in CONDITIONING_LEVELS
        },
        'token_overlap_top20': float(mean_overlap),
        'quant_error_by_condition': {
            cond: {
                nbits: float(np.array(quant_error_data[cond][nbits]).mean())
                for nbits in ['int8', 'int4']
            } for cond in CONDITIONING_LEVELS
        },
    },
    'per_sample_results': all_sample_results,
}

results_path = RESULTS_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {results_path}")
print(f"File size: {results_path.stat().st_size / 1024:.1f} KB")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/07/07_prefix_shielding.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
