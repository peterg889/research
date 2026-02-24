#!/usr/bin/env python3
# Build Exp 09: Scale Normalization as a First-Class KV Cache Correction.
#
# Exp 08 proved that int8 quantization's benefit is driven by the per-tensor
# absmax/qmax normalization step (H_C), not by the rounding grid, noise, or
# outlier removal. int16 quantization (32767 levels — negligible rounding)
# captures 77–100% of the int8 improvement across 6 datasets.
#
# The user's key insight: this "quantization benefit" is really a deficiency
# in two-phase KV cache generation. The normalization step accidentally
# corrects pathological scale properties. We should test whether explicit
# scale normalization should be a standard post-Phase A correction.
#
# This experiment tests scale normalization as a standalone production fix
# across 10 diverse datasets with comprehensive metrics.
#
# Design:
#   3 Conditioning x 5 Corrections + 1 single-pass = 16 Phase B + 3 Phase A + 1
#   Conditioning: bare, comprehend_64, random_64
#   Corrections: bf16, int8, int16, norm_roundtrip, clip_3sigma
#   + single_pass (ground truth, once per sample)
#
#   10 datasets x 200 hard = 2000 samples x 20 passes = 40,000 forward passes
#
# SEED=42, SCORING_KEY='bos_retained_scale_norm_v09'

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/09", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    # Validate Python syntax
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 09: Scale Normalization as a First-Class KV Cache Correction

## Motivation

Exp 08 diagnosed **why** simulated int8 quantization paradoxically improves NLL across all
datasets and conditioning levels. The answer: **scale normalization** (H_C). The `absmax/qmax`
normalization step in quantization corrects pathological scale drift inherent in two-phase
KV cache generation. Key evidence:

- **int16** (32767 levels, negligible rounding) captures 77–100% of the int8 benefit
- Gaussian noise matched to int8 error std does NOT replicate the benefit → not regularization
- Outlier clipping helps only partially → outlier suppression is secondary
- Per-tensor normalization outperforms per-head → coarser scope is better
- K and V contribute roughly equally

**Key insight**: The "quantization benefit" is really a **two-phase caching deficiency**.
The `absmax/qmax` normalization accidentally corrects it. We should test whether explicit
normalization belongs in the standard pipeline.

## Design

### Datasets (10)

| Category | Dataset | Source | N_HARD |
|----------|---------|--------|--------|
| Extractive QA | SQuAD 2.0, TriviaQA, Quoref | Exp 03/06 | 200 each |
| Retrieval QA | MS MARCO | Exp 02 | 200 |
| Reasoning | DROP, HotpotQA, GSM8K, ROPES | Exp 03/05/06 | 200 each |
| Multiple choice | RACE-high, QuALITY | Exp 05/06 | 200 each |

### Factorial: 3 Conditioning × 5 Correction + 1 single-pass = 16 per sample

**Conditioning (3):**

| Level | Description |
|-------|-------------|
| `bare` | `[BOS] + doc` → cache (no prefix) |
| `comprehend_64` | Comprehend instruction, L=64 → select + reposition |
| `random_64` | Random tokens, L=64 → select + reposition (structural control) |

**Correction (5):**

| Level | Description | Tests |
|-------|-------------|-------|
| `bf16` | No correction (baseline) | — |
| `int8` | Full simulated int8 quantization | Exp 07/08 benchmark |
| `int16` | Full simulated int16 quantization | Near-lossless normalization |
| `norm_roundtrip` | `(x / (absmax/127)) * (absmax/127)`, NO round/clamp | Is bf16 arithmetic noise sufficient? |
| `clip_3sigma` | Clamp doc K/V to ±3σ per tensor | Outlier removal only |

**Plus**: `single_pass` — ground truth NLL from full `[BOS]+doc+query+answer` in one pass.

### Budget

10 datasets × 200 hard × (3 Phase A + 15 Phase B + 1 single-pass) = 38,000 forward passes.
Estimated: ~3 hours on A100.

### Key Analyses

1. **Cross-dataset NLL improvement**: Does normalization help universally?
2. **Two-phase gap recovery**: What fraction of the `bf16 − single_pass` gap does each correction close?
3. **Quintile stratification**: Does normalization help hard samples disproportionately?
4. **Prefix × normalization interaction**: Does the benefit compound or overlap?
5. **norm_roundtrip**: Is the bf16 arithmetic round-trip alone sufficient, or is the int grid needed?
""")


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
from collections import defaultdict
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d
from lib.data import count_words

SEED = 42
N_SAMPLES_OLD = 400   # Exp 02/03/05 used 400 samples per dataset
N_SAMPLES_NEW = 500   # Exp 06 used 500 samples per dataset
HARD_FRAC = 0.40
N_HARD = 200          # top 40% of 500 (or top 200 of 400)
PREFIX_L = 64
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp09")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Existing results directories for bare NLL reuse
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP06_DIR = Path("../../../results/decoder_only/exp06")

# 10 datasets spanning extractive QA, retrieval, reasoning, and MC
DATASETS = [
    'squad_v2', 'triviaqa', 'ms_marco', 'hotpotqa', 'drop',
    'race_high', 'gsm8k', 'quality', 'ropes', 'quoref',
]

CONDITIONING_LEVELS = ['bare', 'comprehend_64', 'random_64']
CORRECTION_LEVELS = ['bf16', 'int8', 'int16', 'norm_roundtrip', 'clip_3sigma']

# Per-dataset seeds (same as Exp 05/06)
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
    'boolq': SEED + 500,
    'race_high': SEED + 600,
    'gsm8k': SEED + 700,
    'quality': SEED + 800,
    'multirc': SEED + 900,
    'ropes': SEED + 1000,
    'quoref': SEED + 1100,
    'record': SEED + 1200,
    'race_middle': SEED + 1300,
}

SCORING_KEY = 'bos_retained_scale_norm_v09'

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
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL  # 765 (same as Exp 05/06/08)

special_ids = set(tokenizer.all_special_ids)

print(f"Exp 09: Scale Normalization as a First-Class KV Cache Correction")
print(f"N_HARD: {N_HARD} per dataset")
print(f"PREFIX_L: {PREFIX_L}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}, N_LAYERS: {N_LAYERS}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")
print(f"Conditioning: {CONDITIONING_LEVELS}")
print(f"Corrections: {CORRECTION_LEVELS}")
print(f"Datasets: {DATASETS}")

# --- RoPE repositioning helpers (reused from Exp 05/06/08) ---
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
# CORE SCORING FUNCTIONS (from Exp 08)
# ===================================================================

def deep_copy_cache(cache):
    cloned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys.clone()
        v = cache.layers[i].values.clone()
        cloned.update(k, v, i)
    return cloned


def simulated_quantize(tensor, nbits):
    # Per-tensor symmetric quantization round-trip (simulated, returns bf16).
    qmax = (1 << (nbits - 1)) - 1
    absmax = tensor.abs().max()
    if absmax == 0:
        return tensor
    scale = absmax / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(tensor.dtype)


def quantize_kv_cache(cache, nbits):
    # Apply simulated_quantize to all layers K and V in-place. BOS preserved.
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = simulated_quantize(k[:, :, 1:, :], nbits)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)
            v_bos = v[:, :, :1, :]
            v_doc = simulated_quantize(v[:, :, 1:, :], nbits)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def norm_roundtrip_kv_cache(cache, qmax=127):
    # The normalization step from quantization WITHOUT rounding or clamping.
    # x -> (x / scale) * scale where scale = absmax / qmax.
    # In exact arithmetic this is identity. In bf16, the divide-multiply
    # round-trip introduces tiny perturbations — testing whether this alone
    # is sufficient to explain the quantization benefit.
    # BOS preserved.
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            k_absmax = k_doc.abs().max()
            if k_absmax > 0:
                k_scale = k_absmax / qmax
                k_doc = ((k_doc / k_scale) * k_scale).to(k.dtype)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)

            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]
            v_absmax = v_doc.abs().max()
            if v_absmax > 0:
                v_scale = v_absmax / qmax
                v_doc = ((v_doc / v_scale) * v_scale).to(v.dtype)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def clip_kv_cache(cache, n_sigma):
    # Clamp doc K/V to [mean - n*sigma, mean + n*sigma] per tensor. BOS preserved.
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


def encode_phase_a(doc_text, prefix_token_ids=None):
    # Phase A: encode document with optional prefix, return cache + metadata.
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


def score_single_pass(doc_text, query_text, answer_text):
    # Full single-pass forward. Ground truth NLL.
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


def correct_and_score(cache, D, correction, query_text, answer_text):
    # Dispatcher: deep-copy cache, apply correction, score Phase B.
    if correction == 'bf16':
        return score_phase_b(cache, D, query_text, answer_text)

    c = deep_copy_cache(cache)

    if correction == 'int8':
        quantize_kv_cache(c, 8)
    elif correction == 'int16':
        quantize_kv_cache(c, 16)
    elif correction == 'norm_roundtrip':
        norm_roundtrip_kv_cache(c, qmax=127)
    elif correction == 'clip_3sigma':
        clip_kv_cache(c, 3.0)
    else:
        del c
        raise ValueError(f"Unknown correction: {correction}")

    nll = score_phase_b(c, D, query_text, answer_text)
    del c
    return nll


print(f"\nSetup complete.")
print(f"Core functions: encode_phase_a, score_phase_b, score_single_pass, correct_and_score")
print(f"Correction functions: quantize_kv_cache, norm_roundtrip_kv_cache, clip_kv_cache")
""")


# ===== Cell 3: Load 10 Datasets + Hard Samples =====
code(r"""# Cell 3: Load 10 datasets + hard samples from existing bare checkpoints
from datasets import load_dataset

print("=" * 70)
print("LOADING 10 DATASETS + HARD SAMPLES")
print("=" * 70)

all_samples = {}    # ds_name -> list of all N_SAMPLES sample dicts
hard_samples = {}   # ds_name -> list of N_HARD hard sample dicts


# ================================================================
# HELPER: Load bare NLLs and select hard samples
# ================================================================
def select_hard(samples, bare_nlls, n_hard):
    bare_arr = np.array(bare_nlls)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:n_hard])
    hs = []
    for idx in h_idx:
        s = dict(samples[idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        hs.append(s)
    return hs


# ================================================================
# MS MARCO (from Exp 02)
# ================================================================
print("\n--- MS MARCO (Exp 02) ---")
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']

ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES_OLD:
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
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES_OLD]
all_samples['ms_marco'] = [msmarco_candidates[i] for i in indices]
del ds_msmarco, msmarco_candidates

# Verify alignment
for i in range(min(20, N_SAMPLES_OLD)):
    assert all_samples['ms_marco'][i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"
print("  MS MARCO alignment verified")

msmarco_bare = [r['nll_bare'] for r in exp02_results]
hard_samples['ms_marco'] = select_hard(all_samples['ms_marco'], msmarco_bare, N_HARD)
print(f"  MS MARCO: {N_HARD} hard samples")
del exp02_ckpt, exp02_results
gc.collect()


# ================================================================
# SQuAD 2.0, TriviaQA, HotpotQA (from Exp 03 bare checkpoints)
# ================================================================
# --- SQuAD 2.0 ---
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
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES_OLD]
all_samples['squad_v2'] = [squad_candidates[i] for i in sq_indices]
del ds_squad, squad_candidates
gc.collect()

# --- TriviaQA ---
print("--- TriviaQA ---")
ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
trivia_candidates = []
for item in ds_trivia:
    entity_pages = item.get('entity_pages', {})
    wiki_contexts = entity_pages.get('wiki_context', [])
    if not wiki_contexts or not wiki_contexts[0]:
        continue
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
    passage_lower = passage.lower()
    found = answer_val.lower() in passage_lower
    if not found:
        for alias in aliases:
            if alias.lower() in passage_lower:
                found = True
                break
    if not found:
        continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        trivia_candidates.append({
            'passage': passage, 'query': query, 'answer': answer_val,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['triviaqa'])
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES_OLD]
all_samples['triviaqa'] = [trivia_candidates[i] for i in tr_indices]
del ds_trivia, trivia_candidates
gc.collect()

# --- HotpotQA ---
print("--- HotpotQA ---")
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
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES_OLD]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# Load bare NLLs from Exp 03
for ds_name in ['squad_v2', 'triviaqa', 'hotpotqa']:
    bare_path = EXP03_DIR / f"bare_{ds_name}.json"
    bare_ckpt = json.loads(bare_path.read_text())
    assert bare_ckpt.get('n_total') == N_SAMPLES_OLD
    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in all_samples[ds_name][:len(saved_queries)]]
    assert saved_queries == current_queries, f"{ds_name}: query mismatch with Exp 03"
    hard_samples[ds_name] = select_hard(all_samples[ds_name], bare_ckpt['bare_nlls'], N_HARD)
    print(f"  {ds_name}: {N_HARD} hard samples (from Exp 03 bare)")
    del bare_ckpt
gc.collect()


# ================================================================
# DROP, RACE-high (from Exp 05 bare checkpoints)
# ================================================================
# --- DROP ---
print("\n--- DROP (Exp 05) ---")
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
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES_OLD]
all_samples['drop'] = [drop_candidates[i] for i in drop_indices]
del ds_drop, drop_candidates
gc.collect()

bare_path = EXP05_DIR / "bare_drop.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['drop'][:len(saved_queries)]]
assert saved_queries == current_queries, "DROP: query mismatch with Exp 05"
hard_samples['drop'] = select_hard(all_samples['drop'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  DROP: {N_HARD} hard samples (from Exp 05 bare)")
del bare_ckpt

# --- RACE-high ---
print("--- RACE-high (Exp 05) ---")
ds_race = load_dataset("race", "high", split="test")
race_candidates = []
for item in ds_race:
    passage = item['article']
    question = item['question']
    correct_idx = ord(item['answer']) - ord('A')
    options = item['options']
    if correct_idx < 0 or correct_idx >= len(options):
        continue
    answer = options[correct_idx]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        race_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['race_high'])
race_indices = np.random.permutation(len(race_candidates))[:N_SAMPLES_OLD]
all_samples['race_high'] = [race_candidates[i] for i in race_indices]
del ds_race, race_candidates
gc.collect()

bare_path = EXP05_DIR / "bare_race_high.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['race_high'][:len(saved_queries)]]
assert saved_queries == current_queries, "RACE-high: query mismatch with Exp 05"
hard_samples['race_high'] = select_hard(all_samples['race_high'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  RACE-high: {N_HARD} hard samples (from Exp 05 bare)")
del bare_ckpt
gc.collect()


# ================================================================
# GSM8K, QuALITY, ROPES, Quoref (from Exp 06 bare checkpoints)
# ================================================================
# --- GSM8K ---
print("\n--- GSM8K (Exp 06) ---")
ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_candidates = []
for item in ds_gsm8k:
    passage = item['question']
    raw_answer = item['answer']
    if '####' not in raw_answer:
        continue
    answer = raw_answer.split('####')[-1].strip()
    if not answer:
        continue
    query = "What is the answer?"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        gsm8k_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['gsm8k'])
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:N_SAMPLES_NEW]
all_samples['gsm8k'] = [gsm8k_candidates[i] for i in gsm8k_indices]
del ds_gsm8k, gsm8k_candidates
gc.collect()

bare_path = EXP06_DIR / "bare_gsm8k.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['gsm8k'][:len(saved_queries)]]
assert saved_queries == current_queries, "GSM8K: query mismatch with Exp 06"
hard_samples['gsm8k'] = select_hard(all_samples['gsm8k'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  GSM8K: {N_HARD} hard samples (from Exp 06 bare)")
del bare_ckpt

# --- QuALITY ---
print("--- QuALITY (Exp 06) ---")
ds_quality = load_dataset("tasksource/QuALITY", split="validation")
quality_candidates = []
for item in ds_quality:
    passage = item['article']
    question = item['question']
    options = item['options']
    gold_label = item['gold_label']
    correct_idx = gold_label - 1
    if correct_idx < 0 or correct_idx >= len(options):
        continue
    answer = options[correct_idx]
    wc = count_words(passage)
    if wc >= 30 and count_words(answer) >= 1:
        quality_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['quality'])
quality_indices = np.random.permutation(len(quality_candidates))[:N_SAMPLES_NEW]
all_samples['quality'] = [quality_candidates[i] for i in quality_indices]
del ds_quality, quality_candidates
gc.collect()

bare_path = EXP06_DIR / "bare_quality.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['quality'][:len(saved_queries)]]
assert saved_queries == current_queries, "QuALITY: query mismatch with Exp 06"
hard_samples['quality'] = select_hard(all_samples['quality'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  QuALITY: {N_HARD} hard samples (from Exp 06 bare)")
del bare_ckpt

# --- ROPES ---
print("--- ROPES (Exp 06) ---")
ds_ropes = load_dataset("allenai/ropes", split="validation")
ropes_candidates = []
for item in ds_ropes:
    background = item['background']
    situation = item['situation']
    passage = background + "\n" + situation
    question = item['question']
    answers = item['answers']
    answer_texts = answers.get('text', [])
    if not answer_texts or not answer_texts[0]:
        continue
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        ropes_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['ropes'])
ropes_indices = np.random.permutation(len(ropes_candidates))[:N_SAMPLES_NEW]
all_samples['ropes'] = [ropes_candidates[i] for i in ropes_indices]
del ds_ropes, ropes_candidates
gc.collect()

bare_path = EXP06_DIR / "bare_ropes.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['ropes'][:len(saved_queries)]]
assert saved_queries == current_queries, "ROPES: query mismatch with Exp 06"
hard_samples['ropes'] = select_hard(all_samples['ropes'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  ROPES: {N_HARD} hard samples (from Exp 06 bare)")
del bare_ckpt

# --- Quoref ---
print("--- Quoref (Exp 06) ---")
ds_quoref = load_dataset("nc33/multispan_quoref", split="validation")
quoref_candidates = []
for item in ds_quoref:
    passage = item['context']
    question = item['question']
    answers = item['answers']
    answer_texts = answers.get('text', [])
    if not answer_texts or not answer_texts[0]:
        continue
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        quoref_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['quoref'])
quoref_indices = np.random.permutation(len(quoref_candidates))[:N_SAMPLES_NEW]
all_samples['quoref'] = [quoref_candidates[i] for i in quoref_indices]
del ds_quoref, quoref_candidates
gc.collect()

bare_path = EXP06_DIR / "bare_quoref.json"
bare_ckpt = json.loads(bare_path.read_text())
saved_queries = bare_ckpt.get('queries_first50', [])
current_queries = [s['query'][:50] for s in all_samples['quoref'][:len(saved_queries)]]
assert saved_queries == current_queries, "Quoref: query mismatch with Exp 06"
hard_samples['quoref'] = select_hard(all_samples['quoref'], bare_ckpt['bare_nlls'], N_HARD)
print(f"  Quoref: {N_HARD} hard samples (from Exp 06 bare)")
del bare_ckpt

gc.collect()

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("Dataset loading summary:")
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    mean_bare = np.mean([s['nll_bare'] for s in hard_samples[ds_name]])
    mean_wc = np.mean([s['word_count'] for s in hard_samples[ds_name]])
    print(f"  {ds_name:<12}: {n_h} hard, mean bare NLL={mean_bare:.3f}, mean words={mean_wc:.0f}")

n_total = sum(len(hard_samples[ds]) for ds in DATASETS)
n_passes = n_total * (len(CONDITIONING_LEVELS) + len(CONDITIONING_LEVELS) * len(CORRECTION_LEVELS) + 1)
print(f"\nTotal: {n_total} samples, ~{n_passes} forward passes")
""")


# ===== Cell 4: Main Scoring Loop =====
code(r"""# Cell 4: Main scoring loop — 3 conditioning x 5 corrections + single_pass x 10 datasets x 200 hard
print("=" * 70)
print("MAIN SCORING LOOP")
print(f"  {len(DATASETS)} datasets x {N_HARD} hard x "
      f"({len(CONDITIONING_LEVELS)} cond x {len(CORRECTION_LEVELS)} corrections + 1 single_pass)")
n_passes_per_sample = (len(CONDITIONING_LEVELS) + len(CONDITIONING_LEVELS) * len(CORRECTION_LEVELS) + 1)
print(f"  = {n_passes_per_sample} passes per sample")
print(f"  Total: {len(DATASETS) * N_HARD * n_passes_per_sample} forward passes")
print("=" * 70)

# Pre-build prefix token IDs for the two prefix conditions
comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

np.random.seed(SEED + 9000)
random_prefix = []
while len(random_prefix) < PREFIX_L:
    tid = np.random.randint(0, VOCAB_SIZE)
    if tid not in special_ids:
        random_prefix.append(int(tid))
random_prefix = random_prefix[:PREFIX_L]

# Map conditioning level names to prefix token IDs
COND_PREFIXES = {
    'bare': None,
    'comprehend_64': comprehend_prefix,
    'random_64': random_prefix,
}

# --- Quick validation ---
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

# Validate all corrections dispatch correctly
cache_v, D_v, _ = encode_phase_a(doc_text_t)
for corr in CORRECTION_LEVELS:
    nll_v = correct_and_score(cache_v, D_v, corr, query_text_t, answer_text_t)
    print(f"  {corr}: {nll_v:.4f}")
del cache_v
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
                      total=n_hard, desc=f"Exp09 {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
                'nll_bare_orig': s['nll_bare'],
            }

            # === Phase A: 3 conditioning passes ===
            conditioning_caches = {}
            conditioning_D = {}

            for cond in CONDITIONING_LEVELS:
                prefix_ids = COND_PREFIXES[cond]
                cache, D, doc_ids = encode_phase_a(
                    s['passage'], prefix_token_ids=prefix_ids)
                conditioning_caches[cond] = cache
                conditioning_D[cond] = D
                result[f'D_{cond}'] = D

            # === Phase B: 5 correction levels per conditioning ===
            for cond in CONDITIONING_LEVELS:
                cache = conditioning_caches[cond]
                D = conditioning_D[cond]

                for corr in CORRECTION_LEVELS:
                    key = f'nll_{cond}_{corr}'
                    nll = correct_and_score(
                        cache, D, corr,
                        s['query'], s['answer'])
                    result[key] = nll

            # === Single pass (ground truth) ===
            result['nll_single_pass'] = score_single_pass(
                s['passage'], s['query'], s['answer'])

            # Cleanup caches
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
                    'correction_levels': CORRECTION_LEVELS,
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
print(f"\nAll scoring complete for {len(DATASETS)} datasets.")
""")


# ===== Cell 5: Cross-Dataset Summary =====
code(r"""# Cell 5: Cross-dataset NLL summary — mean NLL, delta, win rate, Cohen's d
print("=" * 70)
print("CROSS-DATASET SUMMARY")
print("=" * 70)

corrections_only = [c for c in CORRECTION_LEVELS if c != 'bf16']

# Build master NLL arrays: ds_name -> (cond, corr) -> np.array
master_nll = {}
for ds_name in DATASETS:
    results = all_results[ds_name]
    master_nll[ds_name] = {}
    for cond in CONDITIONING_LEVELS:
        for corr in CORRECTION_LEVELS:
            key = f'nll_{cond}_{corr}'
            master_nll[ds_name][(cond, corr)] = np.array([r[key] for r in results])
    master_nll[ds_name]['single_pass'] = np.array([r['nll_single_pass'] for r in results])

# --- Per-dataset summary tables ---
for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    # Mean NLL table
    print(f"\n  Mean NLL (conditioning x correction):")
    header = f"  {'Conditioning':<18}"
    for corr in CORRECTION_LEVELS:
        header += f" {corr[:12]:>12}"
    print(header)
    print(f"  {'-'*(18 + 13 * len(CORRECTION_LEVELS))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for corr in CORRECTION_LEVELS:
            row += f" {master_nll[ds_name][(cond, corr)].mean():>12.4f}"
        print(row)
    sp = master_nll[ds_name]['single_pass']
    print(f"  {'single_pass':<18} {sp.mean():>12.4f}")

    # Delta NLL table (correction - bf16, negative = improved)
    print(f"\n  Delta NLL (correction - bf16; negative = improvement):")
    header = f"  {'Conditioning':<18}"
    for corr in corrections_only:
        header += f" {corr[:12]:>12}"
    print(header)
    print(f"  {'-'*(18 + 13 * len(corrections_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for corr in corrections_only:
            delta = master_nll[ds_name][(cond, corr)] - bf16
            row += f" {delta.mean():>+12.4f}"
        print(row)

    # Win rate table
    print(f"\n  Win rate (% where correction < bf16):")
    header = f"  {'Conditioning':<18}"
    for corr in corrections_only:
        header += f" {corr[:12]:>12}"
    print(header)
    print(f"  {'-'*(18 + 13 * len(corrections_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for corr in corrections_only:
            wins = np.mean(master_nll[ds_name][(cond, corr)] < bf16)
            row += f" {wins:>11.1%}"
        print(row)

    # Cohen's d table
    print(f"\n  Cohen's d (correction vs bf16; negative = improvement):")
    header = f"  {'Conditioning':<18}"
    for corr in corrections_only:
        header += f" {corr[:12]:>12}"
    print(header)
    print(f"  {'-'*(18 + 13 * len(corrections_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for corr in corrections_only:
            delta = master_nll[ds_name][(cond, corr)] - bf16
            d = cohens_d(delta)
            row += f" {d:>+12.3f}"
        print(row)

# --- Grand summary: mean delta across all datasets ---
print(f"\n\n{'='*70}")
print("GRAND SUMMARY: Mean NLL improvement across 10 datasets")
print(f"{'='*70}")

print(f"\n  Mean delta NLL (correction - bf16; negative = improvement):")
header = f"  {'Conditioning':<18}"
for corr in corrections_only:
    header += f" {corr[:12]:>12}"
print(header)
print(f"  {'-'*(18 + 13 * len(corrections_only))}")

for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    for corr in corrections_only:
        deltas = []
        for ds_name in DATASETS:
            bf16 = master_nll[ds_name][(cond, 'bf16')]
            delta = master_nll[ds_name][(cond, corr)] - bf16
            deltas.append(delta.mean())
        row += f" {np.mean(deltas):>+12.4f}"
    print(row)

print(f"\n  Mean win rate across 10 datasets:")
header = f"  {'Conditioning':<18}"
for corr in corrections_only:
    header += f" {corr[:12]:>12}"
print(header)
print(f"  {'-'*(18 + 13 * len(corrections_only))}")

for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    for corr in corrections_only:
        wins = []
        for ds_name in DATASETS:
            bf16 = master_nll[ds_name][(cond, 'bf16')]
            w = np.mean(master_nll[ds_name][(cond, corr)] < bf16)
            wins.append(w)
        row += f" {np.mean(wins):>11.1%}"
    print(row)
""")


# ===== Cell 6: Two-Phase Gap Recovery =====
code(r"""# Cell 6: Two-phase gap recovery — how much of the bf16-single_pass gap does each correction close?
print("=" * 70)
print("TWO-PHASE GAP RECOVERY")
print("=" * 70)

gap_results = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    sp = master_nll[ds_name]['single_pass']

    print(f"\n  {ds_name.upper()}:")

    ds_gaps = {}
    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        gap = bf16 - sp  # positive when bf16 is worse (expected)
        mean_gap = gap.mean()

        print(f"    {cond}:")
        print(f"      Two-phase gap (bf16 - single_pass): {mean_gap:+.4f}")

        for corr in corrections_only:
            corrected = master_nll[ds_name][(cond, corr)]
            recovery = bf16 - corrected  # positive when correction helps
            overshoot = corrected - sp   # positive when correction hasn't reached single_pass

            mean_recovery = recovery.mean()
            recovery_ratio = mean_recovery / (mean_gap + 1e-10) * 100

            print(f"      {corr}: recovery={mean_recovery:+.4f} "
                  f"({recovery_ratio:+.1f}% of gap), "
                  f"overshoot={overshoot.mean():+.4f}")

            ds_gaps[(cond, corr)] = {
                'gap': float(mean_gap),
                'recovery': float(mean_recovery),
                'recovery_pct': float(recovery_ratio),
                'overshoot': float(overshoot.mean()),
            }

    gap_results[ds_name] = ds_gaps

# Summary: mean recovery % across datasets
print(f"\n{'='*70}")
print("RECOVERY SUMMARY: Mean % of two-phase gap closed")
print(f"{'='*70}")

header = f"  {'Conditioning':<18}"
for corr in corrections_only:
    header += f" {corr[:12]:>12}"
print(header)
print(f"  {'-'*(18 + 13 * len(corrections_only))}")

for cond in CONDITIONING_LEVELS:
    row = f"  {cond:<18}"
    for corr in corrections_only:
        recoveries = []
        for ds_name in DATASETS:
            if (cond, corr) in gap_results[ds_name]:
                recoveries.append(gap_results[ds_name][(cond, corr)]['recovery_pct'])
        row += f" {np.mean(recoveries):>+11.1f}%"
    print(row)
""")


# ===== Cell 7: Quintile Stratification =====
code(r"""# Cell 7: Quintile stratification — does normalization help harder samples more?
print("=" * 70)
print("QUINTILE STRATIFICATION BY BARE NLL DIFFICULTY")
print("=" * 70)

quintile_results = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    # Use bare bf16 NLL as difficulty measure
    bare_bf16 = master_nll[ds_name][('bare', 'bf16')]
    quintile_edges = np.percentile(bare_bf16, [0, 20, 40, 60, 80, 100])
    quintile_labels = ['Q1 (easiest)', 'Q2', 'Q3', 'Q4', 'Q5 (hardest)']

    print(f"\n  {ds_name.upper()}:")
    print(f"    Quintile edges: {[f'{e:.2f}' for e in quintile_edges]}")

    ds_quintiles = {}

    # For each conditioning x correction, compute mean improvement per quintile
    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for corr in corrections_only:
            corrected = master_nll[ds_name][(cond, corr)]
            improvement = bf16 - corrected  # positive = correction helped

            q_means = []
            for q in range(5):
                low = quintile_edges[q]
                high = quintile_edges[q + 1]
                if q == 4:
                    mask = bare_bf16 >= low
                else:
                    mask = (bare_bf16 >= low) & (bare_bf16 < high)
                if mask.sum() > 0:
                    q_means.append(float(improvement[mask].mean()))
                else:
                    q_means.append(0.0)

            ds_quintiles[(cond, corr)] = q_means

    quintile_results[ds_name] = ds_quintiles

    # Print key rows: bare+int8, bare+int16, bare+norm_roundtrip
    for corr in corrections_only:
        row = f"    bare+{corr:<14}"
        qm = ds_quintiles[('bare', corr)]
        for q in range(5):
            row += f" {qm[q]:>+7.3f}"
        ratio = qm[4] / (qm[0] + 1e-10) if qm[0] != 0 else float('inf')
        row += f"  (Q5/Q1={ratio:.1f}x)"
        print(row)

# Grand summary: pooled quintile analysis
print(f"\n{'='*70}")
print("POOLED QUINTILE ANALYSIS (mean across 10 datasets)")
print(f"{'='*70}")

for cond in ['bare', 'comprehend_64']:
    print(f"\n  {cond}:")
    for corr in corrections_only:
        pooled = np.zeros(5)
        for ds_name in DATASETS:
            qm = quintile_results[ds_name].get((cond, corr), [0]*5)
            pooled += np.array(qm)
        pooled /= len(DATASETS)
        row = f"    {corr:<16}"
        for q in range(5):
            row += f" {pooled[q]:>+7.3f}"
        ratio = pooled[4] / (pooled[0] + 1e-10)
        row += f"  (Q5/Q1={ratio:.1f}x)"
        print(row)
""")


# ===== Cell 8: Prefix x Normalization Interaction =====
code(r"""# Cell 8: Prefix x normalization interaction — do the benefits compound or overlap?
print("=" * 70)
print("PREFIX x NORMALIZATION INTERACTION ANALYSIS")
print("=" * 70)

interaction_results = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    sp = master_nll[ds_name]['single_pass']

    print(f"\n  {ds_name.upper()}:")

    ds_interaction = {}

    for corr in corrections_only:
        # Four conditions for 2x2 analysis:
        bare_bf16 = master_nll[ds_name][('bare', 'bf16')]
        bare_corr = master_nll[ds_name][('bare', corr)]
        comp_bf16 = master_nll[ds_name][('comprehend_64', 'bf16')]
        comp_corr = master_nll[ds_name][('comprehend_64', corr)]

        # Main effects (mean improvement)
        correction_effect = (bare_bf16 - bare_corr).mean()  # correction alone
        prefix_effect = (bare_bf16 - comp_bf16).mean()      # prefix alone
        combined_effect = (bare_bf16 - comp_corr).mean()     # both together
        interaction = combined_effect - correction_effect - prefix_effect  # interaction term

        # Additivity ratio: combined / (correction + prefix)
        additive_sum = correction_effect + prefix_effect
        additivity_ratio = combined_effect / (additive_sum + 1e-10)

        # Overlap: how much of the correction benefit is already captured by prefix?
        # If prefix+bf16 is already close to bare+corr, then they fix the same thing
        correction_after_prefix = (comp_bf16 - comp_corr).mean()
        overlap_pct = (1.0 - correction_after_prefix / (correction_effect + 1e-10)) * 100

        ds_interaction[corr] = {
            'correction_alone': float(correction_effect),
            'prefix_alone': float(prefix_effect),
            'combined': float(combined_effect),
            'interaction': float(interaction),
            'additivity_ratio': float(additivity_ratio),
            'correction_after_prefix': float(correction_after_prefix),
            'overlap_pct': float(overlap_pct),
        }

        if corr in ['int8', 'int16']:
            print(f"    {corr}:")
            print(f"      Correction alone:     {correction_effect:+.4f}")
            print(f"      Prefix alone:         {prefix_effect:+.4f}")
            print(f"      Combined:             {combined_effect:+.4f}")
            print(f"      Interaction:          {interaction:+.4f}")
            print(f"      Additivity ratio:     {additivity_ratio:.2f}")
            print(f"      Correction after prefix: {correction_after_prefix:+.4f}")
            print(f"      Overlap:              {overlap_pct:.1f}%")

    interaction_results[ds_name] = ds_interaction

# --- Structural decomposition: random prefix + normalization ---
print(f"\n{'='*70}")
print("STRUCTURAL DECOMPOSITION: bare -> random_64 -> comprehend_64")
print("  (with and without int8 correction)")
print(f"{'='*70}")

for ds_name in DATASETS:
    bare_bf16 = master_nll[ds_name][('bare', 'bf16')].mean()
    bare_int8 = master_nll[ds_name][('bare', 'int8')].mean()
    rand_bf16 = master_nll[ds_name][('random_64', 'bf16')].mean()
    rand_int8 = master_nll[ds_name][('random_64', 'int8')].mean()
    comp_bf16 = master_nll[ds_name][('comprehend_64', 'bf16')].mean()
    comp_int8 = master_nll[ds_name][('comprehend_64', 'int8')].mean()
    sp = master_nll[ds_name]['single_pass'].mean()

    structural = bare_bf16 - rand_bf16
    semantic = rand_bf16 - comp_bf16
    norm_bare = bare_bf16 - bare_int8
    norm_comp = comp_bf16 - comp_int8

    print(f"\n  {ds_name}:")
    print(f"    bare+bf16={bare_bf16:.3f}  bare+int8={bare_int8:.3f}  "
          f"rand+bf16={rand_bf16:.3f}  comp+bf16={comp_bf16:.3f}  "
          f"comp+int8={comp_int8:.3f}  single={sp:.3f}")
    print(f"    structural={structural:+.3f}  semantic={semantic:+.3f}  "
          f"norm(bare)={norm_bare:+.3f}  norm(comp)={norm_comp:+.3f}")

# Grand summary
print(f"\n{'='*70}")
print("GRAND INTERACTION SUMMARY (mean across 10 datasets)")
print(f"{'='*70}")

for corr in corrections_only:
    overlap_pcts = [interaction_results[ds][corr]['overlap_pct'] for ds in DATASETS]
    additivity = [interaction_results[ds][corr]['additivity_ratio'] for ds in DATASETS]
    corr_alone = [interaction_results[ds][corr]['correction_alone'] for ds in DATASETS]
    corr_after = [interaction_results[ds][corr]['correction_after_prefix'] for ds in DATASETS]
    print(f"\n  {corr}:")
    print(f"    Correction alone:      {np.mean(corr_alone):+.4f}")
    print(f"    Correction after prefix: {np.mean(corr_after):+.4f}")
    print(f"    Mean overlap:          {np.mean(overlap_pcts):.1f}%")
    print(f"    Mean additivity ratio: {np.mean(additivity):.2f}")
""")


# ===== Cell 9: norm_roundtrip Diagnostic =====
code(r"""# Cell 9: norm_roundtrip diagnostic — is the bf16 arithmetic round-trip alone sufficient?
print("=" * 70)
print("NORM_ROUNDTRIP DIAGNOSTIC")
print("Is the bf16 divide/multiply round-trip alone sufficient to explain the benefit?")
print("=" * 70)

# Key comparison: norm_roundtrip vs int8 vs int16
# If norm_roundtrip ≈ int8/int16: normalization (no rounding) is enough
# If norm_roundtrip << int8/int16: rounding/grid snap is needed

for ds_name in DATASETS:
    print(f"\n  {ds_name.upper()}:")

    for cond in ['bare', 'comprehend_64']:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        int8 = master_nll[ds_name][(cond, 'int8')]
        int16 = master_nll[ds_name][(cond, 'int16')]
        norm = master_nll[ds_name][(cond, 'norm_roundtrip')]

        delta_int8 = (bf16 - int8).mean()
        delta_int16 = (bf16 - int16).mean()
        delta_norm = (bf16 - norm).mean()

        # What fraction of the int8 benefit does norm_roundtrip capture?
        norm_frac = delta_norm / (delta_int8 + 1e-10) * 100

        # Paired test: norm_roundtrip vs int8
        diff = norm - int8  # positive means norm is worse
        t, p = stats.ttest_1samp(diff, 0)

        # Win rates
        norm_wins_vs_bf16 = np.mean(norm < bf16) * 100
        int8_wins_vs_bf16 = np.mean(int8 < bf16) * 100

        print(f"    {cond}:")
        print(f"      int8:  delta={delta_int8:+.4f}  win={int8_wins_vs_bf16:.0f}%")
        print(f"      int16: delta={delta_int16:+.4f}")
        print(f"      norm:  delta={delta_norm:+.4f}  win={norm_wins_vs_bf16:.0f}%  "
              f"({norm_frac:.0f}% of int8)")
        print(f"      norm vs int8: diff={diff.mean():+.4f} (p={p:.2e})")

# Grand summary
print(f"\n{'='*70}")
print("NORM_ROUNDTRIP SUMMARY: Is bf16 round-trip enough?")
print(f"{'='*70}")

for cond in ['bare', 'comprehend_64']:
    norm_fracs = []
    norm_wins = []
    for ds_name in DATASETS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        int8 = master_nll[ds_name][(cond, 'int8')]
        norm = master_nll[ds_name][(cond, 'norm_roundtrip')]
        delta_int8 = (bf16 - int8).mean()
        delta_norm = (bf16 - norm).mean()
        norm_fracs.append(delta_norm / (delta_int8 + 1e-10) * 100)
        norm_wins.append(np.mean(norm < bf16) * 100)

    print(f"\n  {cond}:")
    print(f"    Mean norm_roundtrip as % of int8 benefit: {np.mean(norm_fracs):.1f}%")
    print(f"    Mean norm_roundtrip win rate vs bf16:     {np.mean(norm_wins):.1f}%")

    if np.mean(norm_fracs) > 80:
        print(f"    → CONCLUSION: bf16 round-trip IS sufficient. No rounding grid needed.")
        print(f"      The normalization step (divide/multiply) alone corrects the scale drift.")
    elif np.mean(norm_fracs) > 20:
        print(f"    → CONCLUSION: bf16 round-trip captures PARTIAL benefit.")
        print(f"      Both normalization and grid snap contribute.")
    else:
        print(f"    → CONCLUSION: bf16 round-trip NOT sufficient. Rounding grid is essential.")
        print(f"      The benefit requires snapping values to a discrete grid.")
""")


# ===== Cell 10: Save Results =====
code(r"""# Cell 10: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'exp09_scale_normalization',
    'model': MODEL_NAME,
    'scoring': SCORING_KEY,
    'n_hard': N_HARD,
    'prefix_l': PREFIX_L,
    'common_max_doc': COMMON_MAX_DOC,
    'conditioning_levels': CONDITIONING_LEVELS,
    'correction_levels': CORRECTION_LEVELS,
    'datasets': DATASETS,
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

# Per-dataset summary statistics
for ds_name in DATASETS:
    ds_data = {
        'n': len(all_results[ds_name]),
        'mean_nll': {},
        'delta_nll': {},
        'win_rate': {},
        'cohens_d': {},
    }

    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        for corr in CORRECTION_LEVELS:
            key = f'{cond}_{corr}'
            arr = master_nll[ds_name][(cond, corr)]
            ds_data['mean_nll'][key] = float(arr.mean())
            if corr != 'bf16':
                delta = arr - bf16
                ds_data['delta_nll'][key] = float(delta.mean())
                ds_data['win_rate'][key] = float(np.mean(delta < 0))
                ds_data['cohens_d'][key] = float(cohens_d(delta))

    ds_data['mean_nll']['single_pass'] = float(master_nll[ds_name]['single_pass'].mean())

    # Gap recovery
    ds_data['gap_recovery'] = {}
    for cond in CONDITIONING_LEVELS:
        bf16 = master_nll[ds_name][(cond, 'bf16')]
        sp = master_nll[ds_name]['single_pass']
        gap = (bf16 - sp).mean()
        for corr in corrections_only:
            corrected = master_nll[ds_name][(cond, corr)]
            recovery = (bf16 - corrected).mean()
            ds_data['gap_recovery'][f'{cond}_{corr}'] = {
                'gap': float(gap),
                'recovery': float(recovery),
                'recovery_pct': float(recovery / (gap + 1e-10) * 100),
            }

    # Quintile data
    ds_data['quintile'] = quintile_results.get(ds_name, {})
    # Convert tuple keys to strings
    ds_data['quintile'] = {f'{k[0]}_{k[1]}': v
                           for k, v in ds_data['quintile'].items()}

    # Interaction data
    ds_data['interaction'] = interaction_results.get(ds_name, {})

    final_results[ds_name] = ds_data

# Per-sample data (for downstream analysis)
final_results['per_sample'] = {}
for ds_name in DATASETS:
    final_results['per_sample'][ds_name] = all_results[ds_name]

results_path = RESULTS_DIR / "results.json"
results_path.write_text(json.dumps(final_results, default=str))
print(f"Results saved to {results_path}")
print(f"File size: {results_path.stat().st_size / 1024 / 1024:.1f} MB")

# Also save a compact summary without per-sample data
summary = {k: v for k, v in final_results.items() if k != 'per_sample'}
summary_path = RESULTS_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, default=str))
print(f"Summary saved to {summary_path}")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/09/09_scale_normalization.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path} ({len(nb.cells)} cells)")
