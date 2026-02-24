#!/usr/bin/env python3
# Build Exp 08: Diagnosing the KV Cache Quantization Benefit.
#
# Exp 07 found that simulated int8 quantization IMPROVES NLL (0.91-3.19 pts)
# across all 4 datasets and all 3 conditionings. This experiment diagnoses WHY.
#
# Hypotheses:
#   H_A: Regularization (any noise of the right magnitude helps)
#   H_B: Outlier suppression (per-tensor quantization clips K/V outliers)
#   H_C: Scale normalization (absmax/qmax normalization corrects scale drift)
#   H_D: Structured noise (quantization grid specifically helps vs random noise)
#   H_E: K vs V asymmetry (benefit driven by one tensor type)
#
# Factorial design: 2 Conditioning x 10 Perturbation + 1 single-pass = 21 per sample
# Conditioning: bare, comprehend_64
# Perturbation: bf16, int8, int4, int16, gaussian_matched, clip_2sigma, clip_3sigma,
#               K_only_int8, V_only_int8, per_channel_int8
#
# 2 datasets (DROP, MS MARCO) x 160 hard samples = 320 total
# Budget: 320 x (2 Phase A + 20 Phase B + 1 single-pass) = 7,360 passes
#
# SEED=42, SCORING_KEY='bos_retained_quantization_diagnosis_v08'

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/08", exist_ok=True)

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
md(r"""# Experiment 08: Diagnosing the KV Cache Quantization Benefit

## Exp 07 Findings

Simulated int8 quantization of the KV cache **improves** NLL by 0.91–3.19 points
across all 4 datasets and all 3 conditioning levels:

- **Universal**: 84–100% of samples improve (DROP: 100%)
- **Difficulty-dependent**: 2.4–11× stronger for hard samples (r=0.69–0.92 with bare NLL)
- **Not passage-length dependent** (r ≈ 0.06–0.14)
- **Bare conditioning shows the LARGEST improvement** — rules out RoPE repositioning artifacts
- Prefixed caches have 3× lower quantization error than bare, yet both improve
- int4 helps prefixed caches but hurts bare (precision threshold)

## Candidate Hypotheses

| ID | Hypothesis | Mechanism |
|----|-----------|-----------|
| H_A | Regularization | Any noise of the right magnitude helps |
| H_B | Outlier suppression | Per-tensor quantization clips K/V outlier values that distort attention |
| H_C | Scale normalization | The absmax÷qmax normalization step corrects pathological scale drift |
| H_D | Structured noise | The quantization grid specifically helps (vs random noise) |
| H_E | K vs V asymmetry | The benefit is driven primarily by one tensor type |

## Design

### Factorial: 2 Conditioning × 10 Perturbation + 1 single-pass = 21 per sample

**Factor A — Conditioning (2 levels):**

| Level | Description |
|-------|-------------|
| `bare` | `[BOS] + doc` → cache (no prefix, no repositioning) |
| `comprehend_64` | Comprehend L=64 → select + reposition |

**Factor B — Perturbation (10 levels):**

| Level | Description | Tests |
|-------|-------------|-------|
| `bf16` | No perturbation (baseline) | — |
| `int8` | Per-tensor symmetric 8-bit (replicate Exp 07) | replication |
| `int4` | Per-tensor symmetric 4-bit (replicate Exp 07) | replication |
| `int16` | Per-tensor symmetric 16-bit (qmax=32767) | H_C: normalization with negligible rounding |
| `gaussian_matched` | Gaussian noise σ matched to int8 error std, per layer per tensor | H_A vs H_D |
| `clip_2sigma` | Clamp doc K/V to ±2σ per tensor | H_B: outlier removal |
| `clip_3sigma` | Clamp doc K/V to ±3σ per tensor | H_B: gentler outlier removal |
| `K_only_int8` | Quantize only keys, leave values bf16 | H_E |
| `V_only_int8` | Quantize only values, leave keys bf16 | H_E |
| `per_channel_int8` | Per-head quantization (scale per attention head) | H_C scope |

**Plus**: `single_pass` — full `[BOS]+doc+query+answer` in one forward pass (ground truth).

**Datasets**: DROP (100% improvement, strongest) + MS MARCO (84.4%, weakest). 160 hard each.

### Key Diagnostic Comparisons

1. **Noise vs structure** (H_A vs H_D): `int8` vs `gaussian_matched`
2. **Normalization** (H_C): `int16` vs `bf16` — same normalization, negligible rounding
3. **Outlier removal** (H_B): `clip_2sigma`, `clip_3sigma` vs `bf16` — dose-response
4. **K vs V** (H_E): `K_only_int8` vs `V_only_int8` vs `int8`
5. **Granularity**: `per_channel_int8` vs `int8` — finer normalization scope
6. **Two-phase gap**: `single_pass` vs `bf16` vs `int8`""")


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

RESULTS_DIR = Path("../../../results/decoder_only/exp08")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP07_DIR = Path("../../../results/decoder_only/exp07")

DATASETS = ['drop', 'ms_marco']

CONDITIONING_LEVELS = ['bare', 'comprehend_64']
PERTURBATION_LEVELS = [
    'bf16', 'int8', 'int4', 'int16',
    'gaussian_matched', 'clip_2sigma', 'clip_3sigma',
    'K_only_int8', 'V_only_int8', 'per_channel_int8',
]

# Per-dataset seeds (same as Exp 05/07)
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
}

SCORING_KEY = 'bos_retained_quantization_diagnosis_v08'

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

print(f"Exp 08: Diagnosing the KV Cache Quantization Benefit")
print(f"N_SAMPLES: {N_SAMPLES}, HARD_FRAC: {HARD_FRAC}, N_HARD: {N_HARD}")
print(f"PREFIX_L: {PREFIX_L}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}, N_LAYERS: {N_LAYERS}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print(f"Conditioning: {CONDITIONING_LEVELS}")
print(f"Perturbation: {PERTURBATION_LEVELS}")
print(f"Datasets: {DATASETS}")

# --- RoPE repositioning helpers (reused from Exp 05/06/07) ---
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
# REUSED FUNCTIONS FROM EXP 07
# ===================================================================

def deep_copy_cache(cache):
    # Clone all K/V tensors in a DynamicCache (perturbation is destructive).
    cloned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys.clone()
        v = cache.layers[i].values.clone()
        cloned.update(k, v, i)
    return cloned


def simulated_quantize(tensor, nbits):
    # Per-tensor symmetric quantization round-trip (simulated, returns bf16).
    # scale = absmax / (2^(nbits-1) - 1), round to int grid, clamp, dequantize.
    qmax = (1 << (nbits - 1)) - 1  # 127 for int8, 7 for int4, 32767 for int16
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


def encode_phase_a(doc_text, prefix_token_ids=None):
    # Phase A: encode document with optional prefix, return cache + metadata.
    # Returns (cache, D, doc_ids)
    # cache: DynamicCache with 1+D entries (BOS + doc, prefix truncated)
    # No output_attentions needed for Exp 08 (no eviction).
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
                       use_cache=True, output_attentions=False)
        cache = pa.past_key_values
        del pa
        # Select BOS + doc entries, skip prefix + newline
        keep_indices = [0] + list(range(1 + P + _NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)
        old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True, output_attentions=False)
        cache = pa.past_key_values
        del pa

    return cache, D, doc_ids


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


# ===================================================================
# NEW FUNCTIONS FOR EXP 08
# ===================================================================

def quantize_kv_cache_k_only(cache, nbits):
    # Quantize only keys, leave values at bf16. In-place.
    # BOS preserved unquantized.
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = simulated_quantize(k[:, :, 1:, :], nbits)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)
    return cache


def quantize_kv_cache_v_only(cache, nbits):
    # Quantize only values, leave keys at bf16. In-place.
    # BOS preserved unquantized.
    for i in range(len(cache.layers)):
        v = cache.layers[i].values
        if v.shape[2] > 1:
            v_bos = v[:, :, :1, :]
            v_doc = simulated_quantize(v[:, :, 1:, :], nbits)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def quantize_kv_cache_per_channel(cache, nbits):
    # Per-head (per-channel) symmetric quantization. In-place.
    # Compute absmax per head: shape (1, n_heads, 1, 1).
    # BOS preserved unquantized.
    qmax = (1 << (nbits - 1)) - 1
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            k_absmax = k_doc.abs().amax(dim=(0, 2, 3), keepdim=True)  # (1, n_heads, 1, 1)
            k_absmax = k_absmax.clamp(min=1e-10)
            k_scale = k_absmax / qmax
            k_q = (k_doc / k_scale).round().clamp(-qmax, qmax) * k_scale
            cache.layers[i].keys = torch.cat([k_bos, k_q.to(k.dtype)], dim=2)

            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]
            v_absmax = v_doc.abs().amax(dim=(0, 2, 3), keepdim=True)
            v_absmax = v_absmax.clamp(min=1e-10)
            v_scale = v_absmax / qmax
            v_q = (v_doc / v_scale).round().clamp(-qmax, qmax) * v_scale
            cache.layers[i].values = torch.cat([v_bos, v_q.to(v.dtype)], dim=2)
    return cache


def gaussian_noise_matched(cache, reference_nbits=8):
    # Add Gaussian noise with σ matched to int8 quantization error, per layer per K/V.
    # For each layer: compute actual int8 error std on doc entries, apply N(0, σ) noise.
    # BOS preserved. In-place.
    torch.manual_seed(SEED + 8080)  # fixed seed for reproducibility
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]

            # Measure int8 quantization error std for K
            k_quant = simulated_quantize(k_doc, reference_nbits)
            k_sigma = (k_quant - k_doc).float().std().to(k.dtype)
            k_noise = torch.randn_like(k_doc) * k_sigma
            cache.layers[i].keys = torch.cat([k_bos, k_doc + k_noise], dim=2)

            # Measure int8 quantization error std for V
            v_quant = simulated_quantize(v_doc, reference_nbits)
            v_sigma = (v_quant - v_doc).float().std().to(v.dtype)
            v_noise = torch.randn_like(v_doc) * v_sigma
            cache.layers[i].values = torch.cat([v_bos, v_doc + v_noise], dim=2)
    return cache


def clip_kv_cache(cache, n_sigma):
    # Clamp doc K/V entries to [mean - n*sigma, mean + n*sigma] per tensor. In-place.
    # Pure outlier removal, no quantization grid.
    # BOS preserved.
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :].float()
            k_mean = k_doc.mean()
            k_std = k_doc.std()
            k_clipped = k_doc.clamp(k_mean - n_sigma * k_std, k_mean + n_sigma * k_std)
            cache.layers[i].keys = torch.cat([k_bos, k_clipped.to(k.dtype)], dim=2)

            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :].float()
            v_mean = v_doc.mean()
            v_std = v_doc.std()
            v_clipped = v_doc.clamp(v_mean - n_sigma * v_std, v_mean + n_sigma * v_std)
            cache.layers[i].values = torch.cat([v_bos, v_clipped.to(v.dtype)], dim=2)
    return cache


def score_single_pass(doc_text, query_text, answer_text):
    # Full single-pass forward: [BOS] + doc + \nquery\n + answer. Ground truth NLL.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]
    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    full_ids = [BOS_ID] + doc_ids + query_ids + answer_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([full_ids], device=DEVICE))

    n_ctx = 1 + len(doc_ids) + len(query_ids)
    logits = out.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del out
    return nll


def perturb_and_score(cache, D, perturbation, query_text, answer_text):
    # Dispatcher: deep-copy cache, apply perturbation, score Phase B.
    if perturbation == 'bf16':
        return score_phase_b(cache, D, query_text, answer_text)

    c = deep_copy_cache(cache)

    if perturbation == 'int8':
        quantize_kv_cache(c, 8)
    elif perturbation == 'int4':
        quantize_kv_cache(c, 4)
    elif perturbation == 'int16':
        quantize_kv_cache(c, 16)
    elif perturbation == 'gaussian_matched':
        gaussian_noise_matched(c, reference_nbits=8)
    elif perturbation == 'clip_2sigma':
        clip_kv_cache(c, 2.0)
    elif perturbation == 'clip_3sigma':
        clip_kv_cache(c, 3.0)
    elif perturbation == 'K_only_int8':
        quantize_kv_cache_k_only(c, 8)
    elif perturbation == 'V_only_int8':
        quantize_kv_cache_v_only(c, 8)
    elif perturbation == 'per_channel_int8':
        quantize_kv_cache_per_channel(c, 8)
    else:
        del c
        raise ValueError(f"Unknown perturbation: {perturbation}")

    nll = score_phase_b(c, D, query_text, answer_text)
    del c
    return nll


print(f"\nSetup complete.")
print(f"Functions reused from Exp 07: encode_phase_a, score_phase_b,")
print(f"  deep_copy_cache, simulated_quantize, quantize_kv_cache,")
print(f"  select_kv_cache, reposition_kv_cache, make_prefix")
print(f"New functions: quantize_kv_cache_k_only, quantize_kv_cache_v_only,")
print(f"  quantize_kv_cache_per_channel, gaussian_noise_matched,")
print(f"  clip_kv_cache, score_single_pass, perturb_and_score")
""")


# ===== Cell 3: Load Datasets + Hard Samples =====
code(r"""# Cell 3: Load DROP and MS MARCO + hard samples
from datasets import load_dataset

print("=" * 70)
print("LOADING 2 DATASETS + HARD SAMPLES")
print("=" * 70)

hard_samples = {}   # ds_name -> list of hard sample dicts
all_samples = {}    # ds_name -> list of all N_SAMPLES sample dicts

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
print(f"  MS MARCO: {N_HARD} hard samples, mean bare NLL: {msmarco_bare[msmarco_hard_idx].mean():.4f}")

del exp02_ckpt, exp02_results
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

# Load Exp 05 bare NLLs for hard selection
print("  Loading Exp 05 bare NLLs for DROP hard selection...")
bare_path = EXP05_DIR / "bare_drop.json"
bare_ckpt = json.loads(bare_path.read_text())
bare_nlls_all = bare_ckpt['bare_nlls']

# Verify alignment
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['drop'][:len(saved_queries)]]
assert saved_queries == current_queries, "DROP: query alignment mismatch"

bare_arr = np.array(bare_nlls_all)
sorted_idx = np.argsort(bare_arr)[::-1]
drop_hard_idx = np.sort(sorted_idx[:N_HARD])

hs_drop = []
for idx in drop_hard_idx:
    s = dict(all_samples['drop'][idx])
    s['nll_bare'] = float(bare_arr[idx])
    s['original_idx'] = int(idx)
    hs_drop.append(s)
hard_samples['drop'] = hs_drop
print(f"  DROP: {N_HARD} hard samples, mean bare NLL: {bare_arr[drop_hard_idx].mean():.4f}")

del bare_ckpt
gc.collect()

# ================================================================
# Load Exp 07 checkpoints for validation
# ================================================================
print("\n--- Loading Exp 07 checkpoints for validation ---")
exp07_results = {}
for ds_name in DATASETS:
    ckpt_path = EXP07_DIR / f"checkpoint_{ds_name}.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        exp07_results[ds_name] = ckpt['results']
        print(f"  {ds_name}: loaded {len(ckpt['results'])} Exp 07 results")
    else:
        print(f"  {ds_name}: WARNING — no Exp 07 checkpoint found")

# Summary
print("\n" + "=" * 70)
print("Dataset loading summary:")
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    print(f"  {ds_name}: {n_h} hard samples")
n_total = sum(len(hard_samples[ds]) for ds in DATASETS)
n_passes = n_total * (len(CONDITIONING_LEVELS) + len(CONDITIONING_LEVELS) * len(PERTURBATION_LEVELS) + 1)
print(f"Total: {n_total} samples, ~{n_passes} forward passes")
""")


# ===== Cell 4: Main Scoring Loop =====
code(r"""# Cell 4: Main scoring loop — 2 conditioning x 10 perturbation + single_pass x 2 datasets x 160 samples
print("=" * 70)
print("MAIN SCORING LOOP")
print(f"  {len(DATASETS)} datasets x {N_HARD} samples x "
      f"({len(CONDITIONING_LEVELS)} conditionings x {len(PERTURBATION_LEVELS)} perturbations + 1 single_pass)")
print(f"  = {len(DATASETS) * N_HARD} samples, "
      f"{len(DATASETS) * N_HARD * (len(CONDITIONING_LEVELS) * len(PERTURBATION_LEVELS) + len(CONDITIONING_LEVELS) + 1)} forward passes")
print("=" * 70)

# Pre-build prefix token IDs
comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

# Validation: quick sanity check on tiny example
print("\n--- Validation: bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"

nll_single_t = score_single_pass(doc_text_t, query_text_t, answer_text_t)
cache_t, D_t, _ = encode_phase_a(doc_text_t)
nll_twophase_t = score_phase_b(cache_t, D_t, query_text_t, answer_text_t)
del cache_t
diff_pct = abs(nll_single_t - nll_twophase_t) / nll_single_t * 100
print(f"  Single-pass: {nll_single_t:.6f}, Two-phase: {nll_twophase_t:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"

# Validate perturb_and_score dispatches correctly
cache_v, D_v, _ = encode_phase_a(doc_text_t)
nll_bf16 = perturb_and_score(cache_v, D_v, 'bf16', query_text_t, answer_text_t)
nll_int8 = perturb_and_score(cache_v, D_v, 'int8', query_text_t, answer_text_t)
nll_gauss = perturb_and_score(cache_v, D_v, 'gaussian_matched', query_text_t, answer_text_t)
nll_clip2 = perturb_and_score(cache_v, D_v, 'clip_2sigma', query_text_t, answer_text_t)
nll_konly = perturb_and_score(cache_v, D_v, 'K_only_int8', query_text_t, answer_text_t)
nll_pch = perturb_and_score(cache_v, D_v, 'per_channel_int8', query_text_t, answer_text_t)
del cache_v
print(f"  bf16={nll_bf16:.4f} int8={nll_int8:.4f} gauss={nll_gauss:.4f} "
      f"clip2σ={nll_clip2:.4f} K_only={nll_konly:.4f} per_ch={nll_pch:.4f}")
print("  PASSED all validation checks\n")

gc.collect()
torch.cuda.empty_cache()

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

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Exp08 {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
                'nll_bare_orig': s['nll_bare'],
            }

            # === Phase A: 2 conditioning passes ===
            conditioning_caches = {}
            conditioning_D = {}

            for cond in CONDITIONING_LEVELS:
                if cond == 'bare':
                    prefix_ids = None
                elif cond == 'comprehend_64':
                    prefix_ids = comprehend_prefix
                else:
                    raise ValueError(f"Unknown conditioning: {cond}")

                cache, D, doc_ids = encode_phase_a(
                    s['passage'], prefix_token_ids=prefix_ids)

                conditioning_caches[cond] = cache
                conditioning_D[cond] = D
                result[f'D_{cond}'] = D

            # === Phase B: 10 perturbation levels per conditioning ===
            for cond in CONDITIONING_LEVELS:
                cache = conditioning_caches[cond]
                D = conditioning_D[cond]

                for pert in PERTURBATION_LEVELS:
                    key = f'nll_{cond}_{pert}'
                    nll = perturb_and_score(
                        cache, D, pert,
                        s['query'], s['answer'])
                    result[key] = nll

            # === Single pass (ground truth) ===
            result['nll_single_pass'] = score_single_pass(
                s['passage'], s['query'], s['answer'])

            # Cleanup caches for this sample
            for cond in CONDITIONING_LEVELS:
                del conditioning_caches[cond]
            del conditioning_caches, conditioning_D
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
                    'perturbation_levels': PERTURBATION_LEVELS,
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


# ===== Cell 5: Validation Against Exp 07 =====
code(r"""# Cell 5: Validation against Exp 07 — bf16, int8, int4 must match within 0.001
print("=" * 70)
print("VALIDATION AGAINST EXP 07")
print("=" * 70)

validation_passed = True

for ds_name in DATASETS:
    if ds_name not in exp07_results:
        print(f"\n  {ds_name}: no Exp 07 results available, skipping validation")
        continue

    results_08 = all_results[ds_name]
    results_07 = exp07_results[ds_name]

    # Build lookup from Exp 07 by (query[:50], answer[:50]) for matching
    exp07_lookup = {}
    for r07 in results_07:
        lookup_key = (r07['query'][:50], r07['answer'][:50])
        exp07_lookup[lookup_key] = r07

    print(f"\n  {ds_name} — comparing {len(results_08)} Exp 08 samples against Exp 07:")

    # Check bf16, int8, int4 for bare and comprehend_64
    for cond in CONDITIONING_LEVELS:
        for comp in ['bf16', 'int8', 'int4']:
            key_08 = f'nll_{cond}_{comp}'
            key_07 = f'nll_{cond}_{comp}'
            diffs = []
            matched = 0
            for r08 in results_08:
                lk = (r08['query'][:50], r08['answer'][:50])
                if lk in exp07_lookup:
                    r07 = exp07_lookup[lk]
                    if key_07 in r07 and key_08 in r08:
                        diff = abs(r08[key_08] - r07[key_07])
                        diffs.append(diff)
                        matched += 1

            if diffs:
                max_diff = max(diffs)
                mean_diff = np.mean(diffs)
                status = "PASSED" if max_diff < 0.001 else "FAILED"
                if max_diff >= 0.001:
                    validation_passed = False
                print(f"    {cond}_{comp}: matched={matched}, "
                      f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
            else:
                print(f"    {cond}_{comp}: no matching samples found")

if validation_passed:
    print(f"\n  ALL VALIDATION CHECKS PASSED")
else:
    print(f"\n  WARNING: Some validation checks FAILED (max diff >= 0.001)")
    print(f"  This may indicate sample alignment differences between Exp 07 and Exp 08.")
    print(f"  Check that the same hard samples are being used.")
""")


# ===== Cell 6: Primary Diagnostic Table =====
code(r"""# Cell 6: Primary diagnostic table — 2x10 mean NLL matrix per dataset
print("=" * 70)
print("PRIMARY DIAGNOSTIC TABLE")
print("=" * 70)

pert_only = [p for p in PERTURBATION_LEVELS if p != 'bf16']

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    # --- 2x10 NLL table ---
    print(f"\n  Mean NLL (2 conditioning x 10 perturbation):")
    header = f"  {'Conditioning':<18}"
    for pert in PERTURBATION_LEVELS:
        label = pert[:10]
        header += f" {label:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(PERTURBATION_LEVELS))}")

    nll_arrays = {}  # (cond, pert) -> np.array
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for pert in PERTURBATION_LEVELS:
            key = f'nll_{cond}_{pert}'
            arr = np.array([r[key] for r in results])
            nll_arrays[(cond, pert)] = arr
            row += f" {arr.mean():>11.4f}"
        print(row)

    # Also store single_pass
    sp_arr = np.array([r['nll_single_pass'] for r in results])
    print(f"  {'single_pass':<18} {sp_arr.mean():>11.4f}")

    # --- Delta NLL table ---
    print(f"\n  Delta NLL (= perturbation - bf16, same conditioning; negative = improved):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        label = pert[:10]
        header += f" {label:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    delta_arrays = {}
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = nll_arrays[(cond, 'bf16')]
        for pert in pert_only:
            delta = nll_arrays[(cond, pert)] - bf16
            delta_arrays[(cond, pert)] = delta
            row += f" {delta.mean():>+11.4f}"
        print(row)

    # --- Win rate table ---
    print(f"\n  Win rate (% samples where perturbation < bf16):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        label = pert[:10]
        header += f" {label:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for pert in pert_only:
            wins = np.mean(delta_arrays[(cond, pert)] < 0)
            row += f" {wins:>10.1%}"
        print(row)

    # --- Cohen's d table ---
    print(f"\n  Cohen's d (perturbation vs bf16; negative d = improvement):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        label = pert[:10]
        header += f" {label:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for pert in pert_only:
            d = cohens_d(delta_arrays[(cond, pert)])
            row += f" {d:>+11.3f}"
        print(row)

gc.collect()
""")


# ===== Cell 7: Hypothesis Testing =====
code(r"""# Cell 7: Hypothesis testing — formal paired t-tests for each hypothesis
print("=" * 70)
print("HYPOTHESIS TESTING")
print("=" * 70)

hypothesis_results = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    # Build NLL arrays
    nll = {}
    for cond in CONDITIONING_LEVELS:
        for pert in PERTURBATION_LEVELS:
            nll[(cond, pert)] = np.array([r[f'nll_{cond}_{pert}'] for r in results])
    nll['single_pass'] = np.array([r['nll_single_pass'] for r in results])

    ds_hyp = {}

    # --- H_A vs H_D: Noise vs structure ---
    # Compare: does int8 improve more than gaussian_matched?
    # If gaussian_matched ≈ int8, it's regularization (H_A). If int8 >> gaussian, grid matters (H_D).
    print(f"\n  --- H_A vs H_D: Noise vs Structure ---")
    for cond in CONDITIONING_LEVELS:
        delta_int8 = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        delta_gauss = nll[(cond, 'gaussian_matched')] - nll[(cond, 'bf16')]
        diff = delta_int8 - delta_gauss  # negative = int8 better than gaussian
        t, p = stats.ttest_1samp(diff, 0)
        d = cohens_d(diff)
        print(f"    {cond}: int8 delta={delta_int8.mean():+.4f}, "
              f"gauss delta={delta_gauss.mean():+.4f}, "
              f"diff={diff.mean():+.4f} (d={d:+.3f}, p={p:.2e})")
        interpretation = ("int8 > gauss: grid structure matters (H_D)" if diff.mean() < -0.01
                          else "int8 ≈ gauss: regularization (H_A)" if abs(diff.mean()) < 0.01
                          else "gauss > int8: noise alone helps more")
        print(f"      → {interpretation}")
        ds_hyp[f'H_AD_{cond}'] = {
            'delta_int8': float(delta_int8.mean()),
            'delta_gauss': float(delta_gauss.mean()),
            'diff': float(diff.mean()),
            'd': float(d), 'p': float(p),
            'interpretation': interpretation,
        }

    # --- H_C: Scale normalization ---
    # int16 uses normalization with negligible rounding. If int16 ≈ int8, normalization is the key.
    print(f"\n  --- H_C: Scale Normalization ---")
    for cond in CONDITIONING_LEVELS:
        delta_int16 = nll[(cond, 'int16')] - nll[(cond, 'bf16')]
        delta_int8 = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        t16, p16 = stats.ttest_1samp(delta_int16, 0)
        d16 = cohens_d(delta_int16)
        print(f"    {cond}: int16 delta={delta_int16.mean():+.4f} (d={d16:+.3f}, p={p16:.2e}), "
              f"int8 delta={delta_int8.mean():+.4f}")
        if delta_int16.mean() < -0.01:
            interpretation = "int16 improves: normalization alone is sufficient (H_C supported)"
        elif abs(delta_int16.mean()) < 0.01:
            interpretation = "int16 neutral: normalization alone doesn't help (H_C not supported)"
        else:
            interpretation = "int16 hurts: normalization alone is harmful"
        print(f"      → {interpretation}")
        ds_hyp[f'H_C_{cond}'] = {
            'delta_int16': float(delta_int16.mean()),
            'delta_int8': float(delta_int8.mean()),
            'd_int16': float(d16), 'p_int16': float(p16),
            'interpretation': interpretation,
        }

    # --- H_B: Outlier suppression ---
    # clip_2sigma and clip_3sigma vs bf16. Dose-response: 2σ should help more than 3σ.
    print(f"\n  --- H_B: Outlier Suppression ---")
    for cond in CONDITIONING_LEVELS:
        delta_clip2 = nll[(cond, 'clip_2sigma')] - nll[(cond, 'bf16')]
        delta_clip3 = nll[(cond, 'clip_3sigma')] - nll[(cond, 'bf16')]
        t2, p2 = stats.ttest_1samp(delta_clip2, 0)
        t3, p3 = stats.ttest_1samp(delta_clip3, 0)
        d2 = cohens_d(delta_clip2)
        d3 = cohens_d(delta_clip3)
        print(f"    {cond}: clip_2σ delta={delta_clip2.mean():+.4f} (d={d2:+.3f}, p={p2:.2e})")
        print(f"    {cond}: clip_3σ delta={delta_clip3.mean():+.4f} (d={d3:+.3f}, p={p3:.2e})")
        dose_response = delta_clip2.mean() < delta_clip3.mean()
        print(f"      → Dose-response (2σ < 3σ): {dose_response}")
        if delta_clip2.mean() < -0.01 or delta_clip3.mean() < -0.01:
            interpretation = "Clipping helps: outlier suppression contributes (H_B supported)"
        else:
            interpretation = "Clipping doesn't help: outliers not the main issue (H_B not supported)"
        print(f"      → {interpretation}")
        ds_hyp[f'H_B_{cond}'] = {
            'delta_clip2': float(delta_clip2.mean()),
            'delta_clip3': float(delta_clip3.mean()),
            'd_clip2': float(d2), 'p_clip2': float(p2),
            'd_clip3': float(d3), 'p_clip3': float(p3),
            'dose_response': bool(dose_response),
            'interpretation': interpretation,
        }

    # --- H_E: K vs V asymmetry ---
    # Compare K_only_int8, V_only_int8, int8 (both)
    print(f"\n  --- H_E: K vs V Asymmetry ---")
    for cond in CONDITIONING_LEVELS:
        delta_k = nll[(cond, 'K_only_int8')] - nll[(cond, 'bf16')]
        delta_v = nll[(cond, 'V_only_int8')] - nll[(cond, 'bf16')]
        delta_both = nll[(cond, 'int8')] - nll[(cond, 'bf16')]

        # Paired test: K vs V
        diff_kv = delta_k - delta_v
        t_kv, p_kv = stats.ttest_1samp(diff_kv, 0)
        d_kv = cohens_d(diff_kv)

        # Additivity check: K + V ≈ both?
        sum_kv = delta_k + delta_v
        additivity_ratio = delta_both.mean() / (sum_kv.mean() + 1e-10)

        print(f"    {cond}: K_only delta={delta_k.mean():+.4f}, "
              f"V_only delta={delta_v.mean():+.4f}, "
              f"both delta={delta_both.mean():+.4f}")
        print(f"      K vs V: diff={diff_kv.mean():+.4f} (d={d_kv:+.3f}, p={p_kv:.2e})")
        print(f"      Additivity ratio (both / (K+V)): {additivity_ratio:.2f}")

        if abs(delta_k.mean()) > 2 * abs(delta_v.mean()):
            driver = "K-driven"
        elif abs(delta_v.mean()) > 2 * abs(delta_k.mean()):
            driver = "V-driven"
        else:
            driver = "balanced"
        print(f"      → {driver}")

        ds_hyp[f'H_E_{cond}'] = {
            'delta_k': float(delta_k.mean()),
            'delta_v': float(delta_v.mean()),
            'delta_both': float(delta_both.mean()),
            'd_kv': float(d_kv), 'p_kv': float(p_kv),
            'additivity_ratio': float(additivity_ratio),
            'driver': driver,
        }

    # --- Granularity: per_channel_int8 vs int8 ---
    print(f"\n  --- Granularity: per_channel_int8 vs int8 ---")
    for cond in CONDITIONING_LEVELS:
        delta_pch = nll[(cond, 'per_channel_int8')] - nll[(cond, 'bf16')]
        delta_int8 = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        diff = delta_pch - delta_int8  # positive = per_channel worse than int8
        t_g, p_g = stats.ttest_1samp(diff, 0)
        d_g = cohens_d(diff)
        print(f"    {cond}: per_channel delta={delta_pch.mean():+.4f}, "
              f"int8 delta={delta_int8.mean():+.4f}, "
              f"diff={diff.mean():+.4f} (d={d_g:+.3f}, p={p_g:.2e})")
        if diff.mean() > 0.01:
            interpretation = "Per-tensor better: coarser normalization helps more (supports H_C)"
        elif diff.mean() < -0.01:
            interpretation = "Per-channel better: finer normalization helps more"
        else:
            interpretation = "Similar: granularity doesn't matter much"
        print(f"      → {interpretation}")
        ds_hyp[f'granularity_{cond}'] = {
            'delta_per_channel': float(delta_pch.mean()),
            'delta_int8': float(delta_int8.mean()),
            'diff': float(diff.mean()),
            'd': float(d_g), 'p': float(p_g),
            'interpretation': interpretation,
        }

    hypothesis_results[ds_name] = ds_hyp

# ================================================================
# Hypothesis Decision Table (pooled across datasets)
# ================================================================
print(f"\n{'='*70}")
print("HYPOTHESIS DECISION TABLE (pooled across datasets)")
print(f"{'='*70}")

print(f"\n  {'Hypothesis':<20} {'Evidence':>10} {'Key Metric':>30} {'Verdict':<40}")
print(f"  {'-'*104}")

# H_A vs H_D
int8_deltas = []
gauss_deltas = []
for ds_name in DATASETS:
    for cond in CONDITIONING_LEVELS:
        int8_deltas.append(hypothesis_results[ds_name][f'H_AD_{cond}']['delta_int8'])
        gauss_deltas.append(hypothesis_results[ds_name][f'H_AD_{cond}']['delta_gauss'])
int8_mean = np.mean(int8_deltas)
gauss_mean = np.mean(gauss_deltas)
if abs(int8_mean - gauss_mean) < 0.3:
    ha_verdict = "H_A SUPPORTED: any noise helps"
else:
    ha_verdict = "H_D SUPPORTED: grid structure matters"
print(f"  {'H_A: Regularization':<20} {'int8 vs':>10} {'gauss: ' + f'{int8_mean:+.3f} vs {gauss_mean:+.3f}':>30} {ha_verdict:<40}")

# H_B
clip_deltas = []
for ds_name in DATASETS:
    for cond in CONDITIONING_LEVELS:
        clip_deltas.append(hypothesis_results[ds_name][f'H_B_{cond}']['delta_clip2'])
clip_mean = np.mean(clip_deltas)
if clip_mean < -0.3:
    hb_verdict = "H_B SUPPORTED: outlier clipping helps"
else:
    hb_verdict = "H_B NOT SUPPORTED: clipping doesn't help"
print(f"  {'H_B: Outlier clip':<20} {'clip_2σ':>10} {'mean delta: ' + f'{clip_mean:+.3f}':>30} {hb_verdict:<40}")

# H_C
int16_deltas = []
for ds_name in DATASETS:
    for cond in CONDITIONING_LEVELS:
        int16_deltas.append(hypothesis_results[ds_name][f'H_C_{cond}']['delta_int16'])
int16_mean = np.mean(int16_deltas)
if int16_mean < -0.3:
    hc_verdict = "H_C SUPPORTED: normalization alone helps"
else:
    hc_verdict = "H_C NOT SUPPORTED: normalization alone insufficient"
print(f"  {'H_C: Normalization':<20} {'int16':>10} {'mean delta: ' + f'{int16_mean:+.3f}':>30} {hc_verdict:<40}")

# H_E
k_deltas = []
v_deltas = []
for ds_name in DATASETS:
    for cond in CONDITIONING_LEVELS:
        k_deltas.append(hypothesis_results[ds_name][f'H_E_{cond}']['delta_k'])
        v_deltas.append(hypothesis_results[ds_name][f'H_E_{cond}']['delta_v'])
k_mean = np.mean(k_deltas)
v_mean = np.mean(v_deltas)
if abs(k_mean) > 2 * abs(v_mean):
    he_verdict = "H_E SUPPORTED: K-driven"
elif abs(v_mean) > 2 * abs(k_mean):
    he_verdict = "H_E SUPPORTED: V-driven"
else:
    he_verdict = f"H_E: balanced (K={k_mean:+.3f}, V={v_mean:+.3f})"
print(f"  {'H_E: K vs V':<20} {'K/V':>10} {'K: ' + f'{k_mean:+.3f}' + ' V: ' + f'{v_mean:+.3f}':>30} {he_verdict:<40}")

print()
""")


# ===== Cell 8: Two-Phase Gap + Hardness Interaction =====
code(r"""# Cell 8: Two-phase gap analysis and hardness interaction (5 quintiles)
print("=" * 70)
print("TWO-PHASE GAP + HARDNESS INTERACTION")
print("=" * 70)

gap_analysis = {}
hardness_data = {}

Q_LABELS = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    bf16_bare = np.array([r['nll_bare_bf16'] for r in results])
    int8_bare = np.array([r['nll_bare_int8'] for r in results])
    single_pass = np.array([r['nll_single_pass'] for r in results])

    # --- Part 1: Two-phase gap analysis ---
    print(f"\n  --- Part 1: Two-Phase Gap ---")
    gap = bf16_bare - single_pass          # positive = two-phase worse (expected)
    recovery = bf16_bare - int8_bare       # positive = int8 helps
    overshoot = single_pass - int8_bare    # positive = int8 overshoots single_pass

    recovery_ratio = recovery.mean() / (gap.mean() + 1e-10)

    print(f"    Two-phase gap (bf16_bare - single_pass): {gap.mean():+.4f} +/- {gap.std():.4f}")
    print(f"    Recovery (bf16_bare - int8_bare):        {recovery.mean():+.4f} +/- {recovery.std():.4f}")
    print(f"    Overshoot (single_pass - int8_bare):     {overshoot.mean():+.4f} +/- {overshoot.std():.4f}")
    print(f"    Recovery ratio: {recovery_ratio:.2%}")
    print(f"    Samples where int8 < single_pass (overshoot): {np.mean(int8_bare < single_pass):.1%}")

    if recovery_ratio > 0.8 and recovery_ratio < 1.2:
        gap_interp = "int8 closes the two-phase gap almost exactly"
    elif recovery_ratio > 1.2:
        gap_interp = "int8 overshoots: does MORE than gap correction"
    elif recovery_ratio > 0.5:
        gap_interp = "int8 partially closes the two-phase gap"
    else:
        gap_interp = "int8 recovery is small relative to the gap"
    print(f"    -> {gap_interp}")

    # Also check comprehend_64
    bf16_comp = np.array([r['nll_comprehend_64_bf16'] for r in results])
    int8_comp = np.array([r['nll_comprehend_64_int8'] for r in results])
    gap_comp = bf16_comp - single_pass
    recovery_comp = bf16_comp - int8_comp
    ratio_comp = recovery_comp.mean() / (gap_comp.mean() + 1e-10)
    print(f"\n    comprehend_64:")
    print(f"    Two-phase gap: {gap_comp.mean():+.4f}, Recovery: {recovery_comp.mean():+.4f}, "
          f"Ratio: {ratio_comp:.2%}")

    gap_analysis[ds_name] = {
        'bare': {
            'gap': float(gap.mean()),
            'recovery': float(recovery.mean()),
            'overshoot': float(overshoot.mean()),
            'recovery_ratio': float(recovery_ratio),
            'overshoot_rate': float(np.mean(int8_bare < single_pass)),
            'interpretation': gap_interp,
        },
        'comprehend_64': {
            'gap': float(gap_comp.mean()),
            'recovery': float(recovery_comp.mean()),
            'recovery_ratio': float(ratio_comp),
        },
    }

    # ================================================================
    # Part 2: Hardness Interaction (5 quintiles x conditions)
    # ================================================================
    # Quintile bins by bare bf16 NLL (hardness proxy)
    quintile_bounds = np.percentile(bf16_bare, [20, 40, 60, 80])
    quintiles = np.digitize(bf16_bare, quintile_bounds)

    pert_only = [p for p in PERTURBATION_LEVELS if p != 'bf16']
    ds_hardness = {}

    for cond in CONDITIONING_LEVELS:
        print(f"\n  --- Part 2a: Hardness Interaction ({cond}, 5 quintiles) ---")
        print(f"  Cohen's d of (perturbation - bf16) per quintile (negative d = improvement):")

        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])

        # Header
        header = f"    {'Quintile':<10} {'N':>4} {'bf16':>8}"
        for pert in pert_only:
            header += f" {pert[:8]:>9}"
        print(header)
        print(f"    {'-'*(14 + 8 + 10 * len(pert_only))}")

        cond_hardness = {}
        for q in range(5):
            mask = quintiles == q
            n_q = mask.sum()
            row = f"    {Q_LABELS[q]:<10} {n_q:>4} {bf16_cond[mask].mean():>8.3f}"
            cond_hardness[Q_LABELS[q]] = {'n': int(n_q), 'bf16_mean': float(bf16_cond[mask].mean())}
            for pert in pert_only:
                pert_arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
                delta = pert_arr[mask] - bf16_cond[mask]
                d = cohens_d(delta) if n_q > 1 else 0.0
                row += f" {d:>+9.3f}"
                cond_hardness[Q_LABELS[q]][pert] = float(d)
            print(row)
        ds_hardness[cond] = cond_hardness

    # ================================================================
    # Part 2b: Mean delta NLL by quintile (raw values, not d)
    # ================================================================
    for cond in CONDITIONING_LEVELS:
        print(f"\n  --- Part 2b: Mean Delta NLL ({cond}, 5 quintiles) ---")
        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])

        header = f"    {'Quintile':<10} {'N':>4} {'bf16':>8}"
        for pert in pert_only:
            header += f" {pert[:8]:>9}"
        print(header)
        print(f"    {'-'*(14 + 8 + 10 * len(pert_only))}")

        for q in range(5):
            mask = quintiles == q
            n_q = mask.sum()
            row = f"    {Q_LABELS[q]:<10} {n_q:>4} {bf16_cond[mask].mean():>8.3f}"
            for pert in pert_only:
                pert_arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
                delta = pert_arr[mask] - bf16_cond[mask]
                row += f" {delta.mean():>+9.3f}"
            print(row)

    # ================================================================
    # Part 2c: Hardness-benefit Spearman correlations
    # ================================================================
    print(f"\n  --- Part 2c: Hardness-Benefit Correlations ---")
    print(f"  Spearman rho between bare NLL (hardness) and benefit (bf16 - perturbed):")
    for cond in CONDITIONING_LEVELS:
        print(f"\n    {cond}:")
        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])
        for pert in pert_only:
            pert_arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
            benefit = bf16_cond - pert_arr  # positive = perturbation helps
            rho, p_val = stats.spearmanr(bf16_bare, benefit)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"      {pert:<20} rho={rho:+.3f} (p={p_val:.2e}) {sig}")
        # Also single_pass
        sp_arr = np.array([r['nll_single_pass'] for r in results])
        benefit_sp = bf16_cond - sp_arr
        rho_sp, p_sp = stats.spearmanr(bf16_bare, benefit_sp)
        sig_sp = '***' if p_sp < 0.001 else '**' if p_sp < 0.01 else '*' if p_sp < 0.05 else 'ns'
        print(f"      {'single_pass':<20} rho={rho_sp:+.3f} (p={p_sp:.2e}) {sig_sp}")

    # ================================================================
    # Part 2d: Best perturbation per quintile
    # ================================================================
    print(f"\n  --- Part 2d: Best Perturbation by Quintile ---")
    for cond in CONDITIONING_LEVELS:
        print(f"\n    {cond}:")
        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])
        for q in range(5):
            mask = quintiles == q
            if mask.sum() == 0:
                continue
            best_pert = None
            best_delta = 0
            for pert in pert_only:
                pert_arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
                delta = (pert_arr[mask] - bf16_cond[mask]).mean()
                if delta < best_delta:
                    best_delta = delta
                    best_pert = pert
            print(f"      {Q_LABELS[q]:<10}: {best_pert} ({best_delta:+.4f})")

    # ================================================================
    # Part 2e: Win rate by quintile (% where perturbation < bf16)
    # ================================================================
    print(f"\n  --- Part 2e: Win Rate by Quintile (bare conditioning) ---")
    bf16_cond = np.array([r['nll_bare_bf16'] for r in results])

    header = f"    {'Quintile':<10} {'N':>4}"
    for pert in pert_only:
        header += f" {pert[:8]:>9}"
    print(header)
    print(f"    {'-'*(14 + 10 * len(pert_only))}")

    for q in range(5):
        mask = quintiles == q
        n_q = mask.sum()
        row = f"    {Q_LABELS[q]:<10} {n_q:>4}"
        for pert in pert_only:
            pert_arr = np.array([r[f'nll_bare_{pert}'] for r in results])
            wins = np.mean(pert_arr[mask] < bf16_cond[mask]) if n_q > 0 else 0
            row += f" {wins:>8.0%}"
        print(row)

    hardness_data[ds_name] = ds_hardness

gc.collect()
""")


# ===== Cell 9: Save results.json =====
code(r"""# Cell 9: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'v4_exp08_quantization_diagnosis',
    'model': MODEL_NAME,
    'scoring': SCORING_KEY,
    'seed': SEED,
    'n_hard_per_dataset': N_HARD,
    'hard_fraction': HARD_FRAC,
    'prefix_length': PREFIX_L,
    'common_max_doc': COMMON_MAX_DOC,
    'conditioning_levels': CONDITIONING_LEVELS,
    'perturbation_levels': PERTURBATION_LEVELS,
    'datasets': DATASETS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'hypothesis_results': hypothesis_results,
    'gap_analysis': gap_analysis,
    'hardness_interaction': hardness_data,
    'per_sample_results': {ds: all_results[ds] for ds in DATASETS},
}

results_path = RESULTS_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {results_path}")
print(f"File size: {results_path.stat().st_size / 1024:.1f} KB")

# Summary of hypothesis decisions
print(f"\n--- SUMMARY ---")
for ds_name in DATASETS:
    print(f"\n  {ds_name}:")
    ds_hyp = hypothesis_results[ds_name]
    for key in sorted(ds_hyp.keys()):
        if 'interpretation' in ds_hyp[key]:
            print(f"    {key}: {ds_hyp[key]['interpretation']}")

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
out_path = "experiments/decoder_only/08/08_quantization_diagnosis.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
