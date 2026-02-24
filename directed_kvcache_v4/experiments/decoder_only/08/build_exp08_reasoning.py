#!/usr/bin/env python3
# Build Exp 08b: Quantization Diagnosis on Reasoning Datasets.
#
# Extends Exp 08 to 4 reasoning-heavy datasets:
#   GSM8K (math), HotpotQA (multi-hop), ROPES (causal), RACE-high (exam)
#
# Same 2 Conditioning x 10 Perturbation + single_pass design as Exp 08.
# Reuses all functions from Exp 08. Separate notebook to avoid re-running
# DROP and MS MARCO.
#
# Bare baselines sourced from:
#   GSM8K, ROPES: Exp 06 (N=500, top 40% -> 200 hard)
#   HotpotQA: Exp 03 (N=400, top 40% -> 160 hard)
#   RACE-high: Exp 05 (N=400, top 40% -> 160 hard)

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
md(r"""# Experiment 08b: Quantization Diagnosis — Reasoning Datasets

Extends the Exp 08 quantization diagnosis to 4 reasoning-heavy datasets where the
prefix effect was strongest in Exp 06:

| Dataset | Type | Prefix d (L=64) | Bare baseline |
|---------|------|------------------|---------------|
| **GSM8K** | Multi-step math | +1.33 (largest) | Exp 06, N=500 |
| **HotpotQA** | Multi-hop reasoning | +0.76 | Exp 03, N=400 |
| **ROPES** | Causal reasoning | +0.49 | Exp 06, N=500 |
| **RACE-high** | Exam comprehension | +0.66 | Exp 05, N=400 |

Same 2×10+1 factorial design as Exp 08. All hypotheses (H_A–H_E) tested on each dataset.
Hardness stratification with 5 quintiles.""")


# ===== Cell 2: Setup + Model + All Functions =====
code(r"""# Cell 2: Setup, model loading, and all functions (identical to Exp 08)
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
PREFIX_L = 64
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp08")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Bare baseline sources
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP06_DIR = Path("../../../results/decoder_only/exp06")

# 4 reasoning datasets with their N and bare source
DATASET_CONFIG = {
    'gsm8k':     {'N_SAMPLES': 500, 'bare_dir': EXP06_DIR, 'seed': SEED + 700},
    'hotpotqa':  {'N_SAMPLES': 400, 'bare_dir': EXP03_DIR, 'seed': SEED + 300},
    'ropes':     {'N_SAMPLES': 500, 'bare_dir': EXP06_DIR, 'seed': SEED + 1000},
    'race_high': {'N_SAMPLES': 400, 'bare_dir': EXP05_DIR, 'seed': SEED + 600},
}

DATASETS = list(DATASET_CONFIG.keys())
HARD_FRAC = 0.40

CONDITIONING_LEVELS = ['bare', 'comprehend_64']
PERTURBATION_LEVELS = [
    'bf16', 'int8', 'int4', 'int16',
    'gaussian_matched', 'clip_2sigma', 'clip_3sigma',
    'K_only_int8', 'V_only_int8', 'per_channel_int8',
]

SCORING_KEY = 'bos_retained_quantization_diagnosis_v08b'

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
rope_params = getattr(text_cfg, 'rope_parameters', {})
layer_types = getattr(text_cfg, 'layer_types', [])
N_LAYERS = len(layer_types)
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # 1023 for Gemma 3

NL = len(NEWLINE_IDS)
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL  # 765 (same as Exp 05/08)

special_ids = set(tokenizer.all_special_ids)

print(f"Exp 08b: Quantization Diagnosis — Reasoning Datasets")
print(f"Datasets: {DATASETS}")
for ds, cfg in DATASET_CONFIG.items():
    n_hard = int(cfg['N_SAMPLES'] * HARD_FRAC)
    print(f"  {ds}: N={cfg['N_SAMPLES']}, N_HARD={n_hard}")
print(f"Conditioning: {CONDITIONING_LEVELS}")
print(f"Perturbation: {PERTURBATION_LEVELS}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC} tokens")

# --- RoPE repositioning helpers ---
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


INSTRUCTIONS = {
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
}
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens")


# === Cache + scoring functions (verbatim from Exp 08) ===

def deep_copy_cache(cache):
    cloned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys.clone()
        v = cache.layers[i].values.clone()
        cloned.update(k, v, i)
    return cloned


def simulated_quantize(tensor, nbits):
    qmax = (1 << (nbits - 1)) - 1
    absmax = tensor.abs().max()
    if absmax == 0:
        return tensor
    scale = absmax / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(tensor.dtype)


def quantize_kv_cache(cache, nbits):
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


def encode_phase_a(doc_text, prefix_token_ids=None):
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


# === New Exp 08 perturbation functions ===

def quantize_kv_cache_k_only(cache, nbits):
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = simulated_quantize(k[:, :, 1:, :], nbits)
            cache.layers[i].keys = torch.cat([k_bos, k_doc], dim=2)
    return cache


def quantize_kv_cache_v_only(cache, nbits):
    for i in range(len(cache.layers)):
        v = cache.layers[i].values
        if v.shape[2] > 1:
            v_bos = v[:, :, :1, :]
            v_doc = simulated_quantize(v[:, :, 1:, :], nbits)
            cache.layers[i].values = torch.cat([v_bos, v_doc], dim=2)
    return cache


def quantize_kv_cache_per_channel(cache, nbits):
    qmax = (1 << (nbits - 1)) - 1
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            k_absmax = k_doc.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-10)
            k_scale = k_absmax / qmax
            k_q = (k_doc / k_scale).round().clamp(-qmax, qmax) * k_scale
            cache.layers[i].keys = torch.cat([k_bos, k_q.to(k.dtype)], dim=2)
            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]
            v_absmax = v_doc.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-10)
            v_scale = v_absmax / qmax
            v_q = (v_doc / v_scale).round().clamp(-qmax, qmax) * v_scale
            cache.layers[i].values = torch.cat([v_bos, v_q.to(v.dtype)], dim=2)
    return cache


def gaussian_noise_matched(cache, reference_nbits=8):
    torch.manual_seed(SEED + 8080)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[2] > 1:
            k_bos = k[:, :, :1, :]
            k_doc = k[:, :, 1:, :]
            v_bos = v[:, :, :1, :]
            v_doc = v[:, :, 1:, :]
            k_quant = simulated_quantize(k_doc, reference_nbits)
            k_sigma = (k_quant - k_doc).float().std().to(k.dtype)
            k_noise = torch.randn_like(k_doc) * k_sigma
            cache.layers[i].keys = torch.cat([k_bos, k_doc + k_noise], dim=2)
            v_quant = simulated_quantize(v_doc, reference_nbits)
            v_sigma = (v_quant - v_doc).float().std().to(v.dtype)
            v_noise = torch.randn_like(v_doc) * v_sigma
            cache.layers[i].values = torch.cat([v_bos, v_doc + v_noise], dim=2)
    return cache


def clip_kv_cache(cache, n_sigma):
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


print(f"\nSetup complete. All functions loaded.")
""")


# ===== Cell 3: Load Datasets + Hard Samples =====
code(r"""# Cell 3: Load 4 reasoning datasets + hard samples from bare baselines
from datasets import load_dataset

print("=" * 70)
print("LOADING 4 REASONING DATASETS + HARD SAMPLES")
print("=" * 70)

hard_samples = {}   # ds_name -> list of hard sample dicts
all_samples = {}    # ds_name -> list of all N sample dicts

# ================================================================
# GSM8K — math word problems (from Exp 06, N=500)
# ================================================================
print("\n--- GSM8K ---")
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
print(f"  GSM8K candidates: {len(gsm8k_candidates)}")
np.random.seed(DATASET_CONFIG['gsm8k']['seed'])
gsm8k_N = DATASET_CONFIG['gsm8k']['N_SAMPLES']
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:gsm8k_N]
all_samples['gsm8k'] = [gsm8k_candidates[i] for i in gsm8k_indices]
del ds_gsm8k, gsm8k_candidates
gc.collect()

# ================================================================
# HotpotQA — multi-hop (from Exp 05, N=400)
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
print(f"  HotpotQA candidates: {len(hotpot_candidates)}")
np.random.seed(DATASET_CONFIG['hotpotqa']['seed'])
hotpot_N = DATASET_CONFIG['hotpotqa']['N_SAMPLES']
hp_indices = np.random.permutation(len(hotpot_candidates))[:hotpot_N]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# ================================================================
# ROPES — causal reasoning (from Exp 06, N=500)
# ================================================================
print("\n--- ROPES ---")
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
print(f"  ROPES candidates: {len(ropes_candidates)}")
np.random.seed(DATASET_CONFIG['ropes']['seed'])
ropes_N = DATASET_CONFIG['ropes']['N_SAMPLES']
ropes_indices = np.random.permutation(len(ropes_candidates))[:ropes_N]
all_samples['ropes'] = [ropes_candidates[i] for i in ropes_indices]
del ds_ropes, ropes_candidates
gc.collect()

# ================================================================
# RACE-high — exam comprehension (from Exp 05, N=400)
# ================================================================
print("\n--- RACE-high ---")
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
print(f"  RACE-high candidates: {len(race_candidates)}")
np.random.seed(DATASET_CONFIG['race_high']['seed'])
race_N = DATASET_CONFIG['race_high']['N_SAMPLES']
race_indices = np.random.permutation(len(race_candidates))[:race_N]
all_samples['race_high'] = [race_candidates[i] for i in race_indices]
del ds_race, race_candidates
gc.collect()

# ================================================================
# Load bare NLLs and select hard samples
# ================================================================
print("\n--- Loading bare NLLs for hard selection ---")
for ds_name in DATASETS:
    cfg = DATASET_CONFIG[ds_name]
    n_samples = cfg['N_SAMPLES']
    n_hard = int(n_samples * HARD_FRAC)
    bare_dir = cfg['bare_dir']

    bare_path = bare_dir / f"bare_{ds_name}.json"
    assert bare_path.exists(), f"Bare baseline not found at {bare_path}"
    bare_ckpt = json.loads(bare_path.read_text())
    bare_nlls_all = bare_ckpt['bare_nlls']
    assert len(bare_nlls_all) == n_samples, \
        f"{ds_name}: expected {n_samples} bare NLLs, got {len(bare_nlls_all)}"

    # Verify alignment
    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in all_samples[ds_name][:len(saved_queries)]]
    assert saved_queries == current_queries, \
        f"{ds_name}: query alignment mismatch with bare baseline"

    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:n_hard])

    hs = []
    for idx in h_idx:
        s = dict(all_samples[ds_name][idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        hs.append(s)
    hard_samples[ds_name] = hs
    print(f"  {ds_name}: {n_hard} hard samples, "
          f"mean bare NLL: {bare_arr[h_idx].mean():.4f}, "
          f"range: [{bare_arr[h_idx].min():.4f}, {bare_arr[h_idx].max():.4f}]")

del bare_ckpt
gc.collect()

# Summary
print("\n" + "=" * 70)
print("Dataset loading summary:")
total_passes = 0
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    n_passes = n_h * (len(CONDITIONING_LEVELS) + len(CONDITIONING_LEVELS) * len(PERTURBATION_LEVELS) + 1)
    total_passes += n_passes
    print(f"  {ds_name}: {n_h} hard samples, ~{n_passes} forward passes")
print(f"Total: ~{total_passes} forward passes")
""")


# ===== Cell 4: Main Scoring Loop =====
code(r"""# Cell 4: Main scoring loop
print("=" * 70)
print("MAIN SCORING LOOP — 4 reasoning datasets")
print("=" * 70)

comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)

# Quick validation
print("\n--- Validation ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"
nll_sp = score_single_pass(doc_text_t, query_text_t, answer_text_t)
cache_t, D_t, _ = encode_phase_a(doc_text_t)
nll_tp = score_phase_b(cache_t, D_t, query_text_t, answer_text_t)
del cache_t
diff_pct = abs(nll_sp - nll_tp) / nll_sp * 100
print(f"  Single-pass: {nll_sp:.6f}, Two-phase: {nll_tp:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Validation failed: {diff_pct}%"
print("  PASSED\n")
gc.collect()
torch.cuda.empty_cache()

# Main loop
all_results = {}

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
                      total=n_hard, desc=f"Exp08b {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
                'nll_bare_orig': s['nll_bare'],
            }

            # Phase A: 2 conditioning passes
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

            # Phase B: 10 perturbation levels per conditioning
            for cond in CONDITIONING_LEVELS:
                cache = conditioning_caches[cond]
                D = conditioning_D[cond]
                for pert in PERTURBATION_LEVELS:
                    key = f'nll_{cond}_{pert}'
                    nll = perturb_and_score(
                        cache, D, pert, s['query'], s['answer'])
                    result[key] = nll

            # Single pass
            result['nll_single_pass'] = score_single_pass(
                s['passage'], s['query'], s['answer'])

            for cond in CONDITIONING_LEVELS:
                del conditioning_caches[cond]
            del conditioning_caches, conditioning_D
            gc.collect()
            torch.cuda.empty_cache()

            ds_results.append(result)

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


# ===== Cell 5: Primary Diagnostic Table =====
code(r"""# Cell 5: Primary diagnostic table per dataset
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

    # Mean NLL table
    print(f"\n  Mean NLL (2 conditioning x 10 perturbation):")
    header = f"  {'Conditioning':<18}"
    for pert in PERTURBATION_LEVELS:
        header += f" {pert[:10]:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(PERTURBATION_LEVELS))}")

    nll_arrays = {}
    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        for pert in PERTURBATION_LEVELS:
            arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
            nll_arrays[(cond, pert)] = arr
            row += f" {arr.mean():>11.4f}"
        print(row)
    sp_arr = np.array([r['nll_single_pass'] for r in results])
    print(f"  {'single_pass':<18} {sp_arr.mean():>11.4f}")

    # Delta NLL
    print(f"\n  Delta NLL (negative = improved):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        header += f" {pert[:10]:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = nll_arrays[(cond, 'bf16')]
        for pert in pert_only:
            delta = nll_arrays[(cond, pert)] - bf16
            row += f" {delta.mean():>+11.4f}"
        print(row)

    # Win rate
    print(f"\n  Win rate (% perturbation < bf16):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        header += f" {pert[:10]:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = nll_arrays[(cond, 'bf16')]
        for pert in pert_only:
            wins = np.mean(nll_arrays[(cond, pert)] < bf16)
            row += f" {wins:>10.1%}"
        print(row)

    # Cohen's d
    print(f"\n  Cohen's d (negative = improvement):")
    header = f"  {'Conditioning':<18}"
    for pert in pert_only:
        header += f" {pert[:10]:>11}"
    print(header)
    print(f"  {'-'*(18 + 12 * len(pert_only))}")

    for cond in CONDITIONING_LEVELS:
        row = f"  {cond:<18}"
        bf16 = nll_arrays[(cond, 'bf16')]
        for pert in pert_only:
            delta = nll_arrays[(cond, pert)] - bf16
            d = cohens_d(delta)
            row += f" {d:>+11.3f}"
        print(row)

gc.collect()
""")


# ===== Cell 6: Hypothesis Testing =====
code(r"""# Cell 6: Hypothesis testing per dataset
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

    nll = {}
    for cond in CONDITIONING_LEVELS:
        for pert in PERTURBATION_LEVELS:
            nll[(cond, pert)] = np.array([r[f'nll_{cond}_{pert}'] for r in results])
    nll['single_pass'] = np.array([r['nll_single_pass'] for r in results])

    ds_hyp = {}

    # H_A vs H_D
    print(f"\n  --- H_A vs H_D: Noise vs Structure ---")
    for cond in CONDITIONING_LEVELS:
        delta_int8 = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        delta_gauss = nll[(cond, 'gaussian_matched')] - nll[(cond, 'bf16')]
        diff = delta_int8 - delta_gauss
        t, p = stats.ttest_1samp(diff, 0)
        d = cohens_d(diff)
        print(f"    {cond}: int8={delta_int8.mean():+.4f}, gauss={delta_gauss.mean():+.4f}, "
              f"diff={diff.mean():+.4f} (d={d:+.3f}, p={p:.2e})")
        interp = ("int8 > gauss: grid structure matters (H_D)" if diff.mean() < -0.01
                  else "int8 ~ gauss: regularization (H_A)" if abs(diff.mean()) < 0.01
                  else "gauss > int8: noise alone helps more")
        print(f"      -> {interp}")
        ds_hyp[f'H_AD_{cond}'] = {
            'delta_int8': float(delta_int8.mean()), 'delta_gauss': float(delta_gauss.mean()),
            'diff': float(diff.mean()), 'd': float(d), 'p': float(p), 'interpretation': interp,
        }

    # H_C
    print(f"\n  --- H_C: Scale Normalization ---")
    for cond in CONDITIONING_LEVELS:
        delta_int16 = nll[(cond, 'int16')] - nll[(cond, 'bf16')]
        t16, p16 = stats.ttest_1samp(delta_int16, 0)
        d16 = cohens_d(delta_int16)
        print(f"    {cond}: int16={delta_int16.mean():+.4f} (d={d16:+.3f}, p={p16:.2e})")
        interp = ("int16 improves: normalization alone sufficient (H_C supported)" if delta_int16.mean() < -0.01
                  else "int16 neutral: normalization alone insufficient" if abs(delta_int16.mean()) < 0.01
                  else "int16 hurts")
        print(f"      -> {interp}")
        ds_hyp[f'H_C_{cond}'] = {
            'delta_int16': float(delta_int16.mean()), 'd': float(d16), 'p': float(p16),
            'interpretation': interp,
        }

    # H_B
    print(f"\n  --- H_B: Outlier Suppression ---")
    for cond in CONDITIONING_LEVELS:
        delta_clip2 = nll[(cond, 'clip_2sigma')] - nll[(cond, 'bf16')]
        delta_clip3 = nll[(cond, 'clip_3sigma')] - nll[(cond, 'bf16')]
        t2, p2 = stats.ttest_1samp(delta_clip2, 0)
        d2 = cohens_d(delta_clip2)
        t3, p3 = stats.ttest_1samp(delta_clip3, 0)
        d3 = cohens_d(delta_clip3)
        print(f"    {cond}: clip_2s={delta_clip2.mean():+.4f} (d={d2:+.3f}), "
              f"clip_3s={delta_clip3.mean():+.4f} (d={d3:+.3f})")
        interp = ("Clipping helps (H_B supported)" if delta_clip2.mean() < -0.01 or delta_clip3.mean() < -0.01
                  else "Clipping doesn't help (H_B not supported)")
        print(f"      -> {interp}")
        ds_hyp[f'H_B_{cond}'] = {
            'delta_clip2': float(delta_clip2.mean()), 'delta_clip3': float(delta_clip3.mean()),
            'd_clip2': float(d2), 'd_clip3': float(d3),
            'interpretation': interp,
        }

    # H_E
    print(f"\n  --- H_E: K vs V Asymmetry ---")
    for cond in CONDITIONING_LEVELS:
        delta_k = nll[(cond, 'K_only_int8')] - nll[(cond, 'bf16')]
        delta_v = nll[(cond, 'V_only_int8')] - nll[(cond, 'bf16')]
        delta_both = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        diff_kv = delta_k - delta_v
        t_kv, p_kv = stats.ttest_1samp(diff_kv, 0)
        d_kv = cohens_d(diff_kv)
        additivity = delta_both.mean() / (delta_k.mean() + delta_v.mean() + 1e-10)
        driver = ("K-driven" if abs(delta_k.mean()) > 2 * abs(delta_v.mean())
                  else "V-driven" if abs(delta_v.mean()) > 2 * abs(delta_k.mean())
                  else "balanced")
        print(f"    {cond}: K={delta_k.mean():+.4f}, V={delta_v.mean():+.4f}, "
              f"both={delta_both.mean():+.4f}, additivity={additivity:.2f} -> {driver}")
        ds_hyp[f'H_E_{cond}'] = {
            'delta_k': float(delta_k.mean()), 'delta_v': float(delta_v.mean()),
            'delta_both': float(delta_both.mean()), 'additivity': float(additivity),
            'd_kv': float(d_kv), 'driver': driver,
        }

    # Granularity
    print(f"\n  --- Granularity: per_channel vs per_tensor ---")
    for cond in CONDITIONING_LEVELS:
        delta_pch = nll[(cond, 'per_channel_int8')] - nll[(cond, 'bf16')]
        delta_int8 = nll[(cond, 'int8')] - nll[(cond, 'bf16')]
        diff = delta_pch - delta_int8
        d_g = cohens_d(diff)
        interp = ("Per-tensor better (supports H_C)" if diff.mean() > 0.01
                  else "Per-channel better" if diff.mean() < -0.01
                  else "Similar")
        print(f"    {cond}: per_ch={delta_pch.mean():+.4f}, int8={delta_int8.mean():+.4f}, "
              f"diff={diff.mean():+.4f} -> {interp}")
        ds_hyp[f'granularity_{cond}'] = {
            'delta_per_channel': float(delta_pch.mean()), 'delta_int8': float(delta_int8.mean()),
            'diff': float(diff.mean()), 'd': float(d_g), 'interpretation': interp,
        }

    hypothesis_results[ds_name] = ds_hyp

# Pooled decision table
print(f"\n{'='*70}")
print("HYPOTHESIS DECISION TABLE (pooled across 4 reasoning datasets)")
print(f"{'='*70}")

print(f"\n  {'Hypothesis':<20} {'Key Metric':>30} {'Verdict':<40}")
print(f"  {'-'*94}")

# Pool across datasets x conditionings
int8_d = np.mean([hypothesis_results[ds][f'H_AD_{c}']['delta_int8']
                   for ds in DATASETS for c in CONDITIONING_LEVELS])
gauss_d = np.mean([hypothesis_results[ds][f'H_AD_{c}']['delta_gauss']
                    for ds in DATASETS for c in CONDITIONING_LEVELS])
ha_verdict = "H_A: any noise helps" if abs(int8_d - gauss_d) < 0.3 else "H_D: grid structure matters"
print(f"  {'H_A/D: Noise':<20} {'int8=' + f'{int8_d:+.3f}' + ' gauss=' + f'{gauss_d:+.3f}':>30} {ha_verdict:<40}")

clip_d = np.mean([hypothesis_results[ds][f'H_B_{c}']['delta_clip2']
                   for ds in DATASETS for c in CONDITIONING_LEVELS])
hb_verdict = "H_B SUPPORTED" if clip_d < -0.3 else "H_B NOT SUPPORTED"
print(f"  {'H_B: Outlier clip':<20} {'clip_2s=' + f'{clip_d:+.3f}':>30} {hb_verdict:<40}")

int16_d = np.mean([hypothesis_results[ds][f'H_C_{c}']['delta_int16']
                    for ds in DATASETS for c in CONDITIONING_LEVELS])
hc_verdict = "H_C SUPPORTED: normalization" if int16_d < -0.3 else "H_C NOT SUPPORTED"
print(f"  {'H_C: Normalization':<20} {'int16=' + f'{int16_d:+.3f}':>30} {hc_verdict:<40}")

k_d = np.mean([hypothesis_results[ds][f'H_E_{c}']['delta_k']
                for ds in DATASETS for c in CONDITIONING_LEVELS])
v_d = np.mean([hypothesis_results[ds][f'H_E_{c}']['delta_v']
                for ds in DATASETS for c in CONDITIONING_LEVELS])
he_verdict = f"K={k_d:+.3f}, V={v_d:+.3f} -> " + (
    "K-driven" if abs(k_d) > 2 * abs(v_d) else
    "V-driven" if abs(v_d) > 2 * abs(k_d) else "balanced")
print(f"  {'H_E: K vs V':<20} {'':>30} {he_verdict:<40}")

print()
""")


# ===== Cell 7: Hardness Interaction =====
code(r"""# Cell 7: Hardness interaction (5 quintiles) + two-phase gap
print("=" * 70)
print("HARDNESS INTERACTION + TWO-PHASE GAP")
print("=" * 70)

Q_LABELS = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']
gap_analysis = {}
hardness_data = {}

for ds_name in DATASETS:
    results = all_results[ds_name]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"{ds_name.upper()} ({n} hard samples)")
    print(f"{'='*70}")

    bf16_bare = np.array([r['nll_bare_bf16'] for r in results])
    int8_bare = np.array([r['nll_bare_int8'] for r in results])
    single_pass = np.array([r['nll_single_pass'] for r in results])

    # Two-phase gap
    print(f"\n  --- Two-Phase Gap ---")
    gap = bf16_bare - single_pass
    recovery = bf16_bare - int8_bare
    overshoot = single_pass - int8_bare
    recovery_ratio = recovery.mean() / (gap.mean() + 1e-10)
    print(f"    Gap (bf16-sp): {gap.mean():+.4f}, Recovery (bf16-int8): {recovery.mean():+.4f}")
    print(f"    Overshoot: {overshoot.mean():+.4f}, Overshoot rate: {np.mean(int8_bare < single_pass):.1%}")

    bf16_comp = np.array([r['nll_comprehend_64_bf16'] for r in results])
    int8_comp = np.array([r['nll_comprehend_64_int8'] for r in results])
    gap_comp = bf16_comp - single_pass
    recovery_comp = bf16_comp - int8_comp
    print(f"    comprehend_64: gap={gap_comp.mean():+.4f}, recovery={recovery_comp.mean():+.4f}")

    gap_analysis[ds_name] = {
        'bare': {'gap': float(gap.mean()), 'recovery': float(recovery.mean()),
                 'overshoot_rate': float(np.mean(int8_bare < single_pass))},
        'comprehend_64': {'gap': float(gap_comp.mean()),
                          'recovery': float(recovery_comp.mean())},
    }

    # Quintile stratification
    quintile_bounds = np.percentile(bf16_bare, [20, 40, 60, 80])
    quintiles = np.digitize(bf16_bare, quintile_bounds)
    pert_only = [p for p in PERTURBATION_LEVELS if p != 'bf16']
    ds_hardness = {}

    for cond in CONDITIONING_LEVELS:
        print(f"\n  --- Cohen's d by Quintile ({cond}) ---")
        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])

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

    # Mean delta by quintile
    for cond in CONDITIONING_LEVELS:
        print(f"\n  --- Mean Delta NLL by Quintile ({cond}) ---")
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

    # Hardness-benefit correlations
    print(f"\n  --- Hardness-Benefit Correlations ---")
    for cond in CONDITIONING_LEVELS:
        print(f"    {cond}:")
        bf16_cond = np.array([r[f'nll_{cond}_bf16'] for r in results])
        for pert in pert_only:
            pert_arr = np.array([r[f'nll_{cond}_{pert}'] for r in results])
            benefit = bf16_cond - pert_arr
            rho, p_val = stats.spearmanr(bf16_bare, benefit)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"      {pert:<20} rho={rho:+.3f} (p={p_val:.2e}) {sig}")

    # Win rate by quintile
    print(f"\n  --- Win Rate by Quintile (bare) ---")
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


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'v4_exp08b_quantization_diagnosis_reasoning',
    'model': MODEL_NAME,
    'scoring': SCORING_KEY,
    'seed': SEED,
    'datasets': DATASETS,
    'dataset_config': {ds: {'N_SAMPLES': cfg['N_SAMPLES'],
                            'N_HARD': int(cfg['N_SAMPLES'] * HARD_FRAC),
                            'seed': cfg['seed']}
                       for ds, cfg in DATASET_CONFIG.items()},
    'conditioning_levels': CONDITIONING_LEVELS,
    'perturbation_levels': PERTURBATION_LEVELS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'hypothesis_results': hypothesis_results,
    'gap_analysis': gap_analysis,
    'hardness_interaction': hardness_data,
    'per_sample_results': {ds: all_results[ds] for ds in DATASETS},
}

results_path = RESULTS_DIR / 'results_reasoning.json'
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {results_path}")
print(f"File size: {results_path.stat().st_size / 1024:.1f} KB")

# Summary
print(f"\n--- SUMMARY ---")
for ds_name in DATASETS:
    print(f"\n  {ds_name}:")
    ds_hyp = hypothesis_results[ds_name]
    for key in sorted(ds_hyp.keys()):
        if 'interpretation' in ds_hyp[key]:
            print(f"    {key}: {ds_hyp[key]['interpretation']}")
        elif 'driver' in ds_hyp[key]:
            print(f"    {key}: {ds_hyp[key]['driver']}")

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
out_path = "experiments/decoder_only/08/08b_quantization_diagnosis_reasoning.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
