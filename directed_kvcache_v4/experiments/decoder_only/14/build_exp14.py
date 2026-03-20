#!/usr/bin/env python3
# Build Exp 14: Soft Prompt Tuning for KV Cache Conditioning.
#
# Can we learn continuous embedding vectors (soft prompts) that condition the
# KV cache more effectively than the best static text prefix?
#
# Approach: Learn soft_prompt = nn.Parameter(shape=(P, hidden_size)) where P=64.
# Model frozen, only 245K parameters trained via AdamW.
#
# Experiment structure:
#   Phase 1: Per-dataset soft prompts (7 datasets x 4 inits = 28 runs)
#   Phase 2: Universal soft prompt (4 inits trained on pooled data)
#   Phase 3: Evaluation on 160 hard samples per dataset
#   Phase 4: Analysis vs Exp 13 conditions
#
# Data split (from 400 samples/dataset):
#   160 hard (eval) — same as Exp 13, identified by original_idx
#   200 train + 40 val from remaining 240 "easy" samples
#
# 4 initializations: warm_comprehend, warm_extract, warm_classify, rand
#
# Checkpoint/resume pattern: training logs + .pt files per run.

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/14", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# =====================================================================
# Cell 0: Markdown — Title & Design Overview
# =====================================================================
md(r"""# Experiment 14: Soft Prompt Tuning for KV Cache Conditioning

## Overview

Exps 01-13 established that static text prefixes (especially "comprehend") improve
answer NLL when used to condition the KV cache during Phase A. The prefix influences
how the model encodes the document, producing better KV representations.

**Core question**: Can we learn continuous embedding vectors (soft prompts) that
condition the KV cache more effectively than the best static text prefix?

## Approach

Learn `soft_prompt = nn.Parameter(shape=(P, hidden_size))` where P=64, hidden_size=3840.
Model frozen, only ~245K parameters trained via AdamW. Gradients flow through the
two-phase KV cache pipeline back to the soft prompt embeddings.

## Experiment Design

### Initializations (4 variants)

| Init | Source | Description |
|------|--------|-------------|
| `warm_comprehend` | "comprehend" instruction embeddings | Best static prefix; warm start |
| `warm_extract` | "extract" instruction embeddings | Second-best instruction |
| `warm_classify` | "classify" instruction embeddings | Third instruction variant |
| `rand` | N(0, embed_std) random | Cold start baseline |

### Data split (per dataset, from 400 samples)

- 160 hard samples (eval) — same as Exp 13, by `original_idx`
- 240 remaining → 200 train + 40 validation (deterministic split, seeded)

### Training

| Parameter | Value |
|-----------|-------|
| P (prompt length) | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Gradient clipping | max_norm=1.0 |
| Epochs | 10 |
| Batch size | 1 (online SGD) |
| Early stopping | Patience=3 on validation NLL |

### Evaluation

For each learned soft prompt, evaluate on 160 hard samples in two conditions:
- `soft_{init}`: with normalization (fair comparison with Exp 13)
- `soft_{init}_nonorm`: without normalization (does soft prompt compensate for scale drift?)

## Datasets (7, same as Exp 13)

| Tier | Dataset |
|------|---------|
| High-Reasoning | GSM8K, DROP |
| Mid-Reasoning | HotpotQA, SQuAD v2 |
| Factoid | MS MARCO, TriviaQA |
| Negative Control | BoolQ |""")


# =====================================================================
# Cell 1: Setup + Model + Functions
# =====================================================================
code(r"""# Cell 1: Setup, model loading, and all functions (Exp 14)
import os
os.umask(0o000)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys, json, time, gc, copy
import random as pyrandom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d, win_rate, paired_ttest
from lib.data import count_words
from lib.cache import deep_copy_cache, make_prefix, scramble_prefix
from lib.rope import build_layer_inv_freqs, get_layer_types, select_kv_cache, reposition_kv_cache
from lib.quantization import norm_roundtrip_kv_cache

SEED = 42
N_SAMPLES = 400
HARD_FRAC = 0.40
N_HARD = int(N_SAMPLES * HARD_FRAC)  # 160
PREFIX_L = 64
N_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PATIENCE = 3
N_TRAIN = 200
N_VAL = 40
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp14")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP06_DIR = Path("../../../results/decoder_only/exp06")
EXP13_DIR = Path("../../../results/decoder_only/exp13")

DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'gsm8k']
DATASET_TIERS = {
    'gsm8k': 'high_reasoning', 'drop': 'high_reasoning',
    'hotpotqa': 'mid_reasoning', 'squad_v2': 'mid_reasoning',
    'ms_marco': 'factoid', 'triviaqa': 'factoid',
    'boolq': 'negative_control',
}

INIT_NAMES = ['warm_comprehend', 'warm_extract', 'warm_classify', 'rand']

DS_SEEDS = {
    'ms_marco': SEED,
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
    'boolq': SEED + 500,
    'gsm8k': SEED + 700,
}

SCORING_KEY = 'bos_retained_exp14_soft_prompt'

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
for p in model.parameters():
    p.requires_grad_(False)
# Model frozen — only soft_prompt is trainable (~245K params).
# eval mode disables dropout; requires_grad_(False) prevents gradient storage
# for the 12B model params during backward (saves ~24 GB).

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = model.get_input_embeddings().num_embeddings
N_LAYERS = len(get_layer_types(model))
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # 1023 for Gemma 3
HIDDEN_SIZE = getattr(text_cfg, 'hidden_size', 3840)

NL = len(NEWLINE_IDS)
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL  # 765 (same as Exp 05)
TRAIN_MAX_DOC = 384  # Shorter docs during training to fit backward pass in 40 GB

# Build RoPE helpers from lib
LAYER_INV_FREQS = build_layer_inv_freqs(model, DEVICE)
LAYER_TYPES = get_layer_types(model)

# --- Instruction definitions ---
INSTRUCTIONS = {
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
    'extract': "Extract all key data points, facts, entities, and specific attributes from the following text.",
    'classify': "Determine the subject matter, text type, writing style, and intended audience of this passage.",
}
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")

# Pre-build static prefixes (for warm initialization)
comprehend_prefix = make_prefix(INSTRUCTION_IDS['comprehend'], PREFIX_L)
extract_prefix = make_prefix(INSTRUCTION_IDS['extract'], PREFIX_L)
classify_prefix = make_prefix(INSTRUCTION_IDS['classify'], PREFIX_L)

# Embedding function
embed_fn = model.get_input_embeddings()
EMBED_STD = embed_fn.weight.std().item()

print(f"\nExp 14: Soft Prompt Tuning for KV Cache Conditioning")
print(f"N_SAMPLES: {N_SAMPLES}, N_HARD: {N_HARD}, PREFIX_L: {PREFIX_L}")
print(f"HIDDEN_SIZE: {HIDDEN_SIZE}, soft prompt params: {PREFIX_L * HIDDEN_SIZE:,}")
print(f"N_TRAIN: {N_TRAIN}, N_VAL: {N_VAL}, N_EPOCHS: {N_EPOCHS}")
print(f"LR: {LR}, WEIGHT_DECAY: {WEIGHT_DECAY}, GRAD_CLIP: {GRAD_CLIP}")
print(f"PATIENCE: {PATIENCE}")
print(f"EMBED_STD: {EMBED_STD:.6f}")
print(f"Model: {MODEL_NAME}, DEVICE: {DEVICE}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"N_LAYERS: {N_LAYERS}, SLIDING_CACHE_LIMIT: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC}, TRAIN_MAX_DOC: {TRAIN_MAX_DOC}")
print(f"Memory strategy: reduced doc length during training ({TRAIN_MAX_DOC} tokens)")
print(f"Datasets: {DATASETS}")
print(f"Initializations: {INIT_NAMES}")


# ===================================================================
# EXPERIMENT-SPECIFIC FUNCTIONS
# ===================================================================

def encode_phase_a_soft(doc_text, soft_prompt_embeddings, max_doc_override=None,
                        normalize=False):
    # Phase A with learned soft prompt embeddings (inputs_embeds).
    # NO torch.no_grad() — gradients flow through to soft_prompt.
    # Normalization off during training, applied only at eval time.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if max_doc_override is not None and len(doc_ids) > max_doc_override:
        doc_ids = doc_ids[:max_doc_override]
    elif len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]

    P = soft_prompt_embeddings.shape[0]
    _NL = len(NEWLINE_IDS)
    max_doc = SLIDING_CACHE_LIMIT - 1 - P - _NL
    if len(doc_ids) > max_doc:
        doc_ids = doc_ids[:max_doc]
    D = len(doc_ids)

    # Build embeddings: [BOS, soft_prompt, newline, doc]
    bos_emb = embed_fn(torch.tensor([[BOS_ID]], device=DEVICE))
    nl_emb = embed_fn(torch.tensor([NEWLINE_IDS], device=DEVICE))
    doc_emb = embed_fn(torch.tensor([doc_ids], device=DEVICE))

    inputs_embeds = torch.cat([
        bos_emb,
        soft_prompt_embeddings.unsqueeze(0),
        nl_emb,
        doc_emb,
    ], dim=1)

    pa = model(inputs_embeds=inputs_embeds, use_cache=True, output_attentions=False)
    cache = pa.past_key_values
    del pa

    # Select BOS + doc entries (discard soft prompt + newline)
    keep_indices = [0] + list(range(1 + P + _NL, 1 + P + _NL + D))
    cache = select_kv_cache(cache, keep_indices)

    # RoPE correction: move doc from positions [1+P+NL, ...) to [1, ...)
    old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
    new_pos = torch.arange(1, D + 1, device=DEVICE)
    cache = reposition_kv_cache(cache, old_pos, new_pos,
                                LAYER_INV_FREQS, LAYER_TYPES, bos_start=0)

    if normalize:
        cache = deep_copy_cache(cache)
        norm_roundtrip_kv_cache(cache)

    return cache, D, doc_ids


def encode_phase_a_tokens(doc_text, prefix_token_ids=None, normalize=True):
    # Phase A with token IDs (for evaluation baseline comparison).
    # Same as Exp 13 encode_phase_a.
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
        cache = reposition_kv_cache(cache, old_pos, new_pos,
                                    LAYER_INV_FREQS, LAYER_TYPES, bos_start=0)
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True, output_attentions=False)
        cache = pa.past_key_values
        del pa

    if normalize:
        norm_roundtrip_kv_cache(cache)

    return cache, D, doc_ids


def score_phase_b_train(cache, D_effective, query_text, answer_text):
    # Phase B returning loss tensor for backprop.
    # NO torch.no_grad() — gradients flow through cache to Phase A.
    phase_b_start = D_effective + 1
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    pb = model(
        input_ids=torch.tensor([pb_ids], device=DEVICE),
        past_key_values=cache,
        position_ids=pos.unsqueeze(0),
        use_cache=False,
    )

    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    loss = F.cross_entropy(logits, targets)
    del pb
    return loss


def score_phase_b_eval(cache, D_effective, query_text, answer_text):
    # Phase B returning scalar NLL (no gradients). Same as Exp 13.
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


def init_soft_prompt(init_name):
    # Create initial soft prompt embeddings.
    if init_name == 'rand':
        return nn.Parameter(
            torch.randn(PREFIX_L, HIDDEN_SIZE, device=DEVICE,
                         dtype=torch.bfloat16) * EMBED_STD)
    else:
        # Warm start from instruction tokens
        instr_key = init_name.replace('warm_', '')
        prefix_ids = make_prefix(INSTRUCTION_IDS[instr_key], PREFIX_L)
        with torch.no_grad():
            embs = embed_fn(torch.tensor(prefix_ids, device=DEVICE))
        return nn.Parameter(embs.clone().to(torch.bfloat16))


def train_step(soft_prompt, passage, query, answer):
    # Recompute-at-boundary training step to fit in 40 GB GPU memory.
    #
    # Problem: Phase A + Phase B computation graphs together use ~14 GB of
    # saved activations + 24 GB model = OOM on 40 GB A100.
    #
    # Solution: run Phase A, save cache DATA, free Phase A graph completely,
    # run Phase B forward+backward (small), get cache gradients, then
    # re-run Phase A forward to rebuild its graph and backward through it.
    # Peak = model + max(Phase_A_saved, Phase_B_saved + cache_data).
    from transformers import DynamicCache as _DC

    # --- Step 1: Phase A forward, save cache data, free graph ---
    cache, D, _ = encode_phase_a_soft(passage, soft_prompt, max_doc_override=TRAIN_MAX_DOC)

    saved_kv = []
    layer_types_cache = []
    for lc in cache.layers:
        saved_kv.append((lc.keys.detach().clone(), lc.values.detach().clone()))
        layer_types_cache.append(type(lc))
        if hasattr(lc, 'sliding_window'):
            saved_kv[-1] = (*saved_kv[-1], lc.sliding_window)

    del cache, lc  # lc holds last loop iteration's layer -> keeps entire graph alive
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 2: Phase B forward + backward with detached cache ---
    pb_cache = _DC()
    pb_cache.layers = []
    detached_kv = []
    for i, entry in enumerate(saved_kv):
        k_data, v_data = entry[0], entry[1]
        lc = layer_types_cache[i].__new__(layer_types_cache[i])
        lc.is_initialized = True
        lc.dtype = k_data.dtype
        lc.device = k_data.device
        if len(entry) > 2:
            lc.sliding_window = entry[2]
        k = k_data.requires_grad_(True)
        v = v_data.requires_grad_(True)
        lc.keys = k
        lc.values = v
        detached_kv.extend([k, v])
        pb_cache.layers.append(lc)

    loss = score_phase_b_train(pb_cache, D, query, answer)
    loss_val = loss.item()
    loss.backward()

    kv_grads = [t.grad for t in detached_kv]

    del loss, pb_cache, saved_kv, detached_kv
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 3: Phase A forward AGAIN (recompute) + backward ---
    cache2, D2, _ = encode_phase_a_soft(passage, soft_prompt, max_doc_override=TRAIN_MAX_DOC)

    cache_tensors = []
    for lc in cache2.layers:
        cache_tensors.extend([lc.keys, lc.values])

    torch.autograd.backward(cache_tensors, kv_grads)

    del cache2, cache_tensors, kv_grads
    gc.collect()
    torch.cuda.empty_cache()

    return loss_val


def train_soft_prompt(train_samples, val_samples, init_name, n_epochs=N_EPOCHS,
                      lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
                      patience=PATIENCE, log_path=None, pt_path=None,
                      scope='per_dataset', dataset_label=''):
    # Training loop with validation and early stopping.
    # Returns (soft_prompt, train_log_dict).

    # Check for completed checkpoint
    if log_path and log_path.exists():
        log = json.loads(log_path.read_text())
        if log.get('completed', False):
            print(f"  [{dataset_label}/{init_name}] Already completed, loading...")
            sp = torch.load(pt_path, map_location=DEVICE, weights_only=True)
            return nn.Parameter(sp), log

    # Initialize
    soft_prompt = init_soft_prompt(init_name)
    optimizer = torch.optim.AdamW([soft_prompt], lr=lr, weight_decay=weight_decay)

    best_val_nll = float('inf')
    best_epoch = -1
    best_state = soft_prompt.data.clone()
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Resume from partial checkpoint
    start_epoch = 0
    if log_path and log_path.exists():
        log = json.loads(log_path.read_text())
        if not log.get('completed', False):
            start_epoch = log.get('epoch', 0) + 1
            train_losses = log.get('train_losses', [])
            val_losses = log.get('val_losses', [])
            best_val_nll = log.get('best_val_nll', float('inf'))
            best_epoch = log.get('best_epoch', -1)
            epochs_without_improvement = log.get('epochs_without_improvement', 0)
            # Load current state from pt file
            if pt_path and pt_path.exists():
                sp = torch.load(pt_path, map_location=DEVICE, weights_only=True)
                soft_prompt = nn.Parameter(sp)
                optimizer = torch.optim.AdamW([soft_prompt], lr=lr, weight_decay=weight_decay)
            # Load best state
            best_pt_path = pt_path.parent / pt_path.name.replace('.pt', '_best.pt')
            if best_pt_path.exists():
                best_state = torch.load(best_pt_path, map_location=DEVICE, weights_only=True)
            else:
                best_state = soft_prompt.data.clone()
            print(f"  [{dataset_label}/{init_name}] Resuming from epoch {start_epoch}")

    desc = f"{dataset_label}/{init_name}" if dataset_label else init_name

    for epoch in range(start_epoch, n_epochs):
        # --- Training ---
        epoch_losses = []
        perm = torch.randperm(len(train_samples), generator=torch.Generator().manual_seed(SEED + epoch))

        for idx in tqdm(perm.tolist(), desc=f"Train E{epoch} {desc}", leave=False):
            s = train_samples[idx]
            optimizer.zero_grad()

            loss_val = train_step(soft_prompt, s['passage'], s['query'], s['answer'])

            torch.nn.utils.clip_grad_norm_([soft_prompt], grad_clip)
            optimizer.step()

            epoch_losses.append(loss_val)

        mean_train_loss = np.mean(epoch_losses)
        train_losses.append(float(mean_train_loss))

        # --- Validation ---
        val_nlls = []
        with torch.no_grad():
            for s in val_samples:
                cache, D, _ = encode_phase_a_soft(
                    s['passage'], soft_prompt, max_doc_override=TRAIN_MAX_DOC, normalize=True)
                nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                val_nlls.append(nll)
                del cache
                gc.collect()
                torch.cuda.empty_cache()

        mean_val_nll = np.mean(val_nlls)
        val_losses.append(float(mean_val_nll))

        # Early stopping check
        if mean_val_nll < best_val_nll:
            best_val_nll = mean_val_nll
            best_epoch = epoch
            best_state = soft_prompt.data.clone()
            epochs_without_improvement = 0
            # Save best state
            if pt_path:
                best_pt_path = pt_path.parent / pt_path.name.replace('.pt', '_best.pt')
                torch.save(best_state, best_pt_path)
        else:
            epochs_without_improvement += 1

        print(f"  [{desc}] Epoch {epoch}: train_loss={mean_train_loss:.4f}, "
              f"val_nll={mean_val_nll:.4f}, best={best_val_nll:.4f} (E{best_epoch}), "
              f"patience={epochs_without_improvement}/{patience}")

        # Checkpoint
        if log_path:
            # Save current state for resume
            if pt_path:
                torch.save(soft_prompt.data, pt_path)
            log_dict = {
                'dataset': dataset_label,
                'init': init_name,
                'scope': scope,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_val_nll': float(best_val_nll),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_without_improvement': epochs_without_improvement,
                'soft_prompt_shape': list(soft_prompt.shape),
                'completed': False,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            log_path.write_text(json.dumps(log_dict))

        if epochs_without_improvement >= patience:
            print(f"  [{desc}] Early stopping at epoch {epoch}")
            break

    # Restore best state and save final
    soft_prompt_final = nn.Parameter(best_state)
    if pt_path:
        torch.save(soft_prompt_final.data, pt_path)
        # Clean up _best.pt (now redundant)
        best_pt_path = pt_path.parent / pt_path.name.replace('.pt', '_best.pt')
        if best_pt_path.exists():
            best_pt_path.unlink()

    log_dict = {
        'dataset': dataset_label,
        'init': init_name,
        'scope': scope,
        'epoch': epoch,
        'best_epoch': best_epoch,
        'best_val_nll': float(best_val_nll),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_without_improvement': epochs_without_improvement,
        'soft_prompt_shape': list(soft_prompt_final.shape),
        'completed': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if log_path:
        log_path.write_text(json.dumps(log_dict))

    return soft_prompt_final, log_dict


def evaluate_soft_prompt(hard_samples_list, soft_prompt_param, normalize=True):
    # Evaluate a learned soft prompt on hard samples, returning list of NLLs.
    nlls = []
    with torch.no_grad():
        for s in hard_samples_list:
            cache, D, _ = encode_phase_a_soft(
                s['passage'], soft_prompt_param, normalize=normalize)
            nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
            nlls.append(nll)
            del cache
            gc.collect()
            torch.cuda.empty_cache()
    return nlls


print(f"Functions defined: encode_phase_a_soft, score_phase_b_train, "
      f"score_phase_b_eval, init_soft_prompt, train_soft_prompt, evaluate_soft_prompt")
""")


# =====================================================================
# Cell 2: Dataset Loading + Train/Val/Hard Split
# =====================================================================
code(r"""# Cell 2: Load 7 datasets, identify hard samples, split remaining into train/val
from datasets import load_dataset

print("=" * 70)
print("LOADING 7 DATASETS + TRAIN/VAL/HARD SPLIT")
print("=" * 70)

hard_samples = {}   # ds_name -> list of hard sample dicts
train_samples = {}  # ds_name -> list of train sample dicts
val_samples = {}    # ds_name -> list of val sample dicts
all_samples = {}    # ds_name -> list of all N_SAMPLES sample dicts

# ================================================================
# MS MARCO (bare NLLs from Exp 02)
# ================================================================
print("\n--- MS MARCO ---")
assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES

msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = set(np.sort(sorted_idx[:N_HARD]).tolist())

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

for i in range(min(20, N_SAMPLES)):
    assert msmarco_all[i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"
print("  MS MARCO alignment verified")

hs_msmarco = []
tr_msmarco = []
for idx in range(N_SAMPLES):
    s = dict(msmarco_all[idx])
    s['original_idx'] = idx
    if idx in msmarco_hard_idx:
        s['nll_bare_ref'] = float(msmarco_bare[idx])
        hs_msmarco.append(s)
    else:
        tr_msmarco.append(s)

hard_samples['ms_marco'] = hs_msmarco
all_samples['ms_marco'] = msmarco_all
print(f"  MS MARCO: {len(hs_msmarco)} hard, {len(tr_msmarco)} non-hard")
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
# TriviaQA
# ================================================================
print("\n--- TriviaQA ---")
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
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
all_samples['triviaqa'] = [trivia_candidates[i] for i in tr_indices]
del ds_trivia, trivia_candidates
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
# BoolQ
# ================================================================
print("\n--- BoolQ ---")
ds_boolq = load_dataset("google/boolq", split="validation")
boolq_candidates = []
for item in ds_boolq:
    passage = item['passage']
    question = item['question']
    answer = "Yes" if item['answer'] else "No"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        boolq_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['boolq'])
boolq_indices = np.random.permutation(len(boolq_candidates))[:N_SAMPLES]
all_samples['boolq'] = [boolq_candidates[i] for i in boolq_indices]
del ds_boolq, boolq_candidates
gc.collect()

# ================================================================
# GSM8K
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
np.random.seed(DS_SEEDS['gsm8k'])
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:N_SAMPLES]
all_samples['gsm8k'] = [gsm8k_candidates[i] for i in gsm8k_indices]
del ds_gsm8k, gsm8k_candidates
gc.collect()

# ================================================================
# Load bare NLLs for hard selection (non-MS-MARCO datasets)
# ================================================================
print("\n--- Loading bare NLLs for hard selection ---")
BARE_NLL_SOURCES = {
    'squad_v2': EXP03_DIR / "bare_squad_v2.json",
    'triviaqa': EXP03_DIR / "bare_triviaqa.json",
    'hotpotqa': EXP03_DIR / "bare_hotpotqa.json",
    'drop': EXP05_DIR / "bare_drop.json",
    'boolq': EXP05_DIR / "bare_boolq.json",
    'gsm8k': EXP06_DIR / "bare_gsm8k.json",
}

for ds_name, bare_path in BARE_NLL_SOURCES.items():
    samples_ds = all_samples[ds_name]
    bare_ckpt = json.loads(bare_path.read_text())
    bare_nlls_all = bare_ckpt['bare_nlls'][:N_SAMPLES]

    saved_queries = bare_ckpt.get('queries_first50', [])
    n_check = min(len(saved_queries), len(samples_ds))
    current_queries = [s['query'][:50] for s in samples_ds[:n_check]]
    assert saved_queries[:n_check] == current_queries, f"{ds_name}: query alignment mismatch"

    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = set(np.sort(sorted_idx[:N_HARD]).tolist())

    hs = []
    non_hard = []
    for idx in range(N_SAMPLES):
        s = dict(samples_ds[idx])
        s['original_idx'] = idx
        if idx in h_idx:
            s['nll_bare_ref'] = float(bare_arr[idx])
            hs.append(s)
        else:
            non_hard.append(s)
    hard_samples[ds_name] = hs
    print(f"  {ds_name}: {len(hs)} hard, {len(non_hard)} non-hard, "
          f"mean bare NLL: {bare_arr[list(h_idx)].mean():.4f}")

    # Split non-hard into train/val
    rng = np.random.RandomState(DS_SEEDS[ds_name] + 14000)
    perm = rng.permutation(len(non_hard))
    train_samples[ds_name] = [non_hard[i] for i in perm[:N_TRAIN]]
    val_samples[ds_name] = [non_hard[i] for i in perm[N_TRAIN:N_TRAIN + N_VAL]]
    print(f"    -> {len(train_samples[ds_name])} train, {len(val_samples[ds_name])} val")

del bare_ckpt
gc.collect()

# Split MS MARCO non-hard into train/val
rng_mm = np.random.RandomState(DS_SEEDS['ms_marco'] + 14000)
perm_mm = rng_mm.permutation(len(tr_msmarco))
train_samples['ms_marco'] = [tr_msmarco[i] for i in perm_mm[:N_TRAIN]]
val_samples['ms_marco'] = [tr_msmarco[i] for i in perm_mm[N_TRAIN:N_TRAIN + N_VAL]]
print(f"  ms_marco: -> {len(train_samples['ms_marco'])} train, {len(val_samples['ms_marco'])} val")
del tr_msmarco

# Summary
print("\n" + "=" * 70)
print("Dataset split summary:")
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    n_tr = len(train_samples[ds_name])
    n_va = len(val_samples[ds_name])
    n_all = len(all_samples[ds_name])
    mean_bare = np.mean([s['nll_bare_ref'] for s in hard_samples[ds_name]])
    print(f"  {ds_name:<12}: {n_all} total, {n_h} hard (eval), {n_tr} train, {n_va} val, "
          f"mean bare NLL: {mean_bare:.3f}")
""")


# =====================================================================
# Cell 3: Gradient Validation
# =====================================================================
code(r"""# Cell 3: Validate gradient flow through soft prompt -> cache -> loss
print("=" * 70)
print("GRADIENT VALIDATION")
print("=" * 70)

# Quick test: verify gradients flow from loss back to soft_prompt
test_sp = init_soft_prompt('rand')
assert test_sp.grad is None or (test_sp.grad == 0).all()

s = train_samples[DATASETS[0]][0]
loss_val = train_step(test_sp, s['passage'], s['query'], s['answer'])

assert test_sp.grad is not None, "FAIL: soft_prompt.grad is None after backward"
grad_norm = test_sp.grad.float().norm().item()
assert grad_norm > 0, "FAIL: soft_prompt.grad is zero"
print(f"  Gradient validation PASSED")
print(f"    loss = {loss_val:.4f}")
print(f"    grad norm = {grad_norm:.6f}")
print(f"    grad shape = {test_sp.grad.shape}")
print(f"    grad dtype = {test_sp.grad.dtype}")

# Memory check
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"    Peak GPU memory: {peak_mem:.2f} GB")
assert peak_mem < 40, f"FAIL: Peak memory {peak_mem:.1f} GB exceeds 40 GB limit"
print(f"    Memory check PASSED (< 40 GB)")

del test_sp
gc.collect()
torch.cuda.empty_cache()

# Verify warm initialization matches token embeddings
print("\n--- Warm initialization check ---")
for init_name in ['warm_comprehend', 'warm_extract', 'warm_classify']:
    sp = init_soft_prompt(init_name)
    instr_key = init_name.replace('warm_', '')
    prefix_ids = make_prefix(INSTRUCTION_IDS[instr_key], PREFIX_L)
    with torch.no_grad():
        expected = embed_fn(torch.tensor(prefix_ids, device=DEVICE))
    diff = (sp.data.float() - expected.float()).abs().max().item()
    print(f"  {init_name}: max diff from token embeddings = {diff:.2e}")
    assert diff < 1e-3, f"FAIL: warm init differs too much from token embeddings"
    del sp, expected

print("  Warm initialization PASSED")

# Quick single-sample training sanity check
print("\n--- Single-sample training check ---")
test_sp = init_soft_prompt('warm_comprehend')
test_opt = torch.optim.AdamW([test_sp], lr=LR)

losses_before = []
for _ in range(3):
    test_opt.zero_grad()
    loss_val = train_step(test_sp, s['passage'], s['query'], s['answer'])
    torch.nn.utils.clip_grad_norm_([test_sp], GRAD_CLIP)
    test_opt.step()
    losses_before.append(loss_val)

print(f"  3-step losses: {[f'{l:.4f}' for l in losses_before]}")
if losses_before[-1] < losses_before[0]:
    print(f"  Loss decreased: {losses_before[0]:.4f} -> {losses_before[-1]:.4f} (good)")
else:
    print(f"  WARNING: Loss did not decrease (may need more steps or is already optimal)")

del test_sp, test_opt
gc.collect()
torch.cuda.empty_cache()
print("\nGradient validation complete.")
""")


# =====================================================================
# Cell 4: Markdown — Phase 1: Per-Dataset Training
# =====================================================================
md(r"""## Phase 1: Per-Dataset Soft Prompt Training

Train separate soft prompts for each of 7 datasets with 4 initializations each.

- 200 train samples per dataset
- 40 validation samples for early stopping
- Checkpoint after each epoch for resume capability
- 28 total training runs (7 datasets x 4 inits)""")


# =====================================================================
# Cell 5: Per-Dataset Training Loop
# =====================================================================
code(r"""# Cell 5: Per-dataset soft prompt training (7 datasets x 4 inits = 28 runs)
print("=" * 70)
print("PHASE 1: PER-DATASET SOFT PROMPT TRAINING")
print("=" * 70)

per_dataset_prompts = {}  # (ds_name, init_name) -> soft_prompt parameter
per_dataset_logs = {}     # (ds_name, init_name) -> training log dict

t0_all = time.time()
total_runs = len(DATASETS) * len(INIT_NAMES)
run_count = 0

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} ({len(train_samples[ds_name])} train, "
          f"{len(val_samples[ds_name])} val)")
    print(f"{'='*60}")

    for init_name in INIT_NAMES:
        run_count += 1
        print(f"\n--- Run {run_count}/{total_runs}: {ds_name}/{init_name} ---")

        log_path = RESULTS_DIR / f"training_log_{ds_name}_{init_name}.json"
        pt_path = RESULTS_DIR / f"soft_prompt_{ds_name}_{init_name}.pt"

        sp, log = train_soft_prompt(
            train_samples[ds_name],
            val_samples[ds_name],
            init_name,
            log_path=log_path,
            pt_path=pt_path,
            scope='per_dataset',
            dataset_label=ds_name,
        )

        per_dataset_prompts[(ds_name, init_name)] = sp
        per_dataset_logs[(ds_name, init_name)] = log
        print(f"  Best epoch: {log['best_epoch']}, best val NLL: {log['best_val_nll']:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

elapsed_all = time.time() - t0_all
print(f"\n{'='*60}")
print(f"Per-dataset training complete: {run_count} runs in {elapsed_all/3600:.1f} hours")
print(f"{'='*60}")

# Summary table
print(f"\n{'Init':<20}", end="")
for ds in DATASETS:
    print(f" {ds:>10}", end="")
print()
print("-" * (20 + 11 * len(DATASETS)))
for init_name in INIT_NAMES:
    print(f"{init_name:<20}", end="")
    for ds in DATASETS:
        val = per_dataset_logs[(ds, init_name)]['best_val_nll']
        print(f" {val:>10.4f}", end="")
    print()
""")


# =====================================================================
# Cell 6: Markdown — Phase 2: Universal Training
# =====================================================================
md(r"""## Phase 2: Universal Soft Prompt Training

Train a single soft prompt on pooled data from all 7 datasets.

- 1400 train samples (200 x 7)
- 280 validation samples (40 x 7)
- 4 initializations
- Tests whether cross-dataset generalization is possible""")


# =====================================================================
# Cell 7: Universal Training Loop
# =====================================================================
code(r"""# Cell 7: Universal soft prompt training (pooled across all datasets, 4 inits)
print("=" * 70)
print("PHASE 2: UNIVERSAL SOFT PROMPT TRAINING")
print("=" * 70)

# Pool train and val across all datasets
universal_train = []
universal_val = []
for ds_name in DATASETS:
    universal_train.extend(train_samples[ds_name])
    universal_val.extend(val_samples[ds_name])

print(f"Universal train: {len(universal_train)} samples")
print(f"Universal val: {len(universal_val)} samples")

universal_prompts = {}  # init_name -> soft_prompt parameter
universal_logs = {}     # init_name -> training log dict

t0 = time.time()

for init_name in INIT_NAMES:
    print(f"\n--- Universal/{init_name} ---")

    log_path = RESULTS_DIR / f"training_log_universal_{init_name}.json"
    pt_path = RESULTS_DIR / f"soft_prompt_universal_{init_name}.pt"

    sp, log = train_soft_prompt(
        universal_train,
        universal_val,
        init_name,
        log_path=log_path,
        pt_path=pt_path,
        scope='universal',
        dataset_label='universal',
    )

    universal_prompts[init_name] = sp
    universal_logs[init_name] = log
    print(f"  Best epoch: {log['best_epoch']}, best val NLL: {log['best_val_nll']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nUniversal training complete: 4 runs in {elapsed/3600:.1f} hours")

# Summary
print(f"\n{'Init':<25} {'Best Epoch':>10} {'Best Val NLL':>12}")
print("-" * 50)
for init_name in INIT_NAMES:
    log = universal_logs[init_name]
    print(f"{init_name:<25} {log['best_epoch']:>10} {log['best_val_nll']:>12.4f}")
""")


# =====================================================================
# Cell 8: Markdown — Phase 3: Evaluation
# =====================================================================
md(r"""## Phase 3: Evaluation on Hard Samples

Evaluate all learned soft prompts on the 160 hard samples per dataset.

For each soft prompt, two conditions:
- **With normalization** (`soft_{init}`): fair comparison with Exp 13
- **Without normalization** (`soft_{init}_nonorm`): test if soft prompt compensates for scale drift

Also load all Exp 13 results from checkpoints (not re-run).""")


# =====================================================================
# Cell 9: Evaluation Loop
# =====================================================================
code(r"""# Cell 9: Evaluate all learned soft prompts on hard samples + load Exp 13 results
print("=" * 70)
print("PHASE 3: EVALUATION ON HARD SAMPLES")
print("=" * 70)

eval_results = {}  # ds_name -> list of result dicts (one per hard sample)

# Evaluation conditions for soft prompts
SOFT_CONDS = []
for init_name in INIT_NAMES:
    SOFT_CONDS.append(f'soft_{init_name}')
    SOFT_CONDS.append(f'soft_{init_name}_nonorm')
UNIVERSAL_CONDS = []
for init_name in INIT_NAMES:
    UNIVERSAL_CONDS.append(f'univ_{init_name}')
    UNIVERSAL_CONDS.append(f'univ_{init_name}_nonorm')

ALL_EVAL_CONDS = SOFT_CONDS + UNIVERSAL_CONDS
print(f"Soft prompt conditions: {len(ALL_EVAL_CONDS)}")
print(f"  Per-dataset: {SOFT_CONDS}")
print(f"  Universal: {UNIVERSAL_CONDS}")

# Exp 13 conditions to load from checkpoints
EXP13_CONDITIONS = [
    'bare', 'random', 'repeat_token', 'unrelated', 'adversarial',
    'tfidf', 'oracle', 'comprehend', 'extract', 'classify',
    'scrambled_comprehend', 'llm_question', 'ood_query', 'misleading_query',
]

for ds_name in DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"

    print(f"\n{'='*60}")
    print(f"Evaluating: {ds_name} ({n_hard} hard samples)")
    print(f"{'='*60}")

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

    # Load Exp 13 checkpoint for this dataset
    exp13_ckpt_path = EXP13_DIR / f"checkpoint_{ds_name}.json"
    exp13_results = []
    if exp13_ckpt_path.exists():
        exp13_ckpt = json.loads(exp13_ckpt_path.read_text())
        exp13_results = exp13_ckpt.get('results', [])
        print(f"  Loaded {len(exp13_results)} Exp 13 results")
        del exp13_ckpt
    else:
        print(f"  WARNING: Exp 13 checkpoint not found at {exp13_ckpt_path}")

    # Build Exp 13 lookup by original_idx
    exp13_by_idx = {}
    for r in exp13_results:
        exp13_by_idx[r['original_idx']] = r
    del exp13_results

    if start_idx < n_hard:
        t0 = time.time()

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Eval {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'original_idx': s['original_idx'],
            }

            # Load Exp 13 NLLs for this sample
            exp13_r = exp13_by_idx.get(s['original_idx'], {})
            for cond in EXP13_CONDITIONS:
                key = f'nll_{cond}'
                if key in exp13_r:
                    result[key] = exp13_r[key]
            if 'nll_single_pass' in exp13_r:
                result['nll_single_pass'] = exp13_r['nll_single_pass']

            # Evaluate per-dataset soft prompts
            for init_name in INIT_NAMES:
                sp = per_dataset_prompts.get((ds_name, init_name))
                if sp is None:
                    continue

                # With normalization
                with torch.no_grad():
                    cache, D, _ = encode_phase_a_soft(s['passage'], sp, normalize=True)
                    nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                    result[f'nll_soft_{init_name}'] = nll
                    del cache

                # Without normalization
                with torch.no_grad():
                    cache, D, _ = encode_phase_a_soft(s['passage'], sp, normalize=False)
                    nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                    result[f'nll_soft_{init_name}_nonorm'] = nll
                    del cache

                gc.collect()
                torch.cuda.empty_cache()

            # Evaluate universal soft prompts
            for init_name in INIT_NAMES:
                sp = universal_prompts.get(init_name)
                if sp is None:
                    continue

                with torch.no_grad():
                    cache, D, _ = encode_phase_a_soft(s['passage'], sp, normalize=True)
                    nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                    result[f'nll_univ_{init_name}'] = nll
                    del cache

                with torch.no_grad():
                    cache, D, _ = encode_phase_a_soft(s['passage'], sp, normalize=False)
                    nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                    result[f'nll_univ_{init_name}_nonorm'] = nll
                    del cache

                gc.collect()
                torch.cuda.empty_cache()

            ds_results.append(result)

            # Checkpoint every 20 samples
            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'soft_conds': SOFT_CONDS,
                    'universal_conds': UNIVERSAL_CONDS,
                    'exp13_conds': EXP13_CONDITIONS,
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
        print(f"  Evaluation complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    eval_results[ds_name] = ds_results

    del exp13_by_idx
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nEvaluation complete for {len(DATASETS)} datasets.")
""")


# =====================================================================
# Cell 10: Markdown — Phase 4: Analysis
# =====================================================================
md(r"""## Phase 4: Analysis

Compare learned soft prompts against all Exp 13 static conditions.

Key questions:
1. Do learned soft prompts outperform the best static prefix (comprehend)?
2. Which initialization works best?
3. Does per-dataset specialization help vs universal?
4. Does the soft prompt compensate for scale drift (nonorm vs norm)?""")


# =====================================================================
# Cell 11: Statistical Comparison
# =====================================================================
code(r"""# Cell 11: Statistical comparison — soft prompts vs Exp 13 conditions
import pandas as pd

print("=" * 70)
print("PHASE 4 ANALYSIS: SOFT PROMPTS VS EXP 13")
print("=" * 70)

# Build NLL arrays for all conditions
master_nll = {}
for ds_name in DATASETS:
    master_nll[ds_name] = {}
    results = eval_results[ds_name]
    # Collect all NLL keys present in results
    nll_keys = [k for k in results[0].keys() if k.startswith('nll_')]
    for key in nll_keys:
        vals = [r[key] for r in results if key in r]
        if len(vals) == len(results):
            master_nll[ds_name][key] = np.array(vals)

# All conditions present across all datasets
all_cond_keys = set()
for ds_name in DATASETS:
    all_cond_keys.update(master_nll[ds_name].keys())
all_cond_keys = sorted(all_cond_keys)
print(f"\nAll NLL keys: {len(all_cond_keys)}")

# --- Cohen's d table: soft prompts vs bare ---
print("\n\nCohen's d (bare - condition; positive = condition better than bare):")
header = f"  {'Condition':<30}"
for ds in DATASETS:
    header += f" {ds:>10}"
header += f" {'pooled':>8}"
print(header)
print(f"  {'-'*(30 + 11 * len(DATASETS) + 9)}")

# Exp 13 conditions
for cond in EXP13_CONDITIONS:
    if cond == 'bare':
        continue
    key = f'nll_{cond}'
    row = f"  {cond:<30}"
    all_diffs = []
    for ds in DATASETS:
        if key in master_nll[ds] and 'nll_bare' in master_nll[ds]:
            diff = master_nll[ds]['nll_bare'] - master_nll[ds][key]
            d = cohens_d(diff)
            row += f" {d:>+10.3f}"
            all_diffs.extend(diff.tolist())
        else:
            row += f" {'N/A':>10}"
    if all_diffs:
        pooled_d = cohens_d(all_diffs)
        row += f" {pooled_d:>+8.3f}"
    print(row)

print()

# Soft prompt conditions (per-dataset)
for init_name in INIT_NAMES:
    for suffix in ['', '_nonorm']:
        cond_label = f'soft_{init_name}{suffix}'
        key = f'nll_soft_{init_name}{suffix}'
        row = f"  {cond_label:<30}"
        all_diffs = []
        for ds in DATASETS:
            if key in master_nll[ds] and 'nll_bare' in master_nll[ds]:
                diff = master_nll[ds]['nll_bare'] - master_nll[ds][key]
                d = cohens_d(diff)
                row += f" {d:>+10.3f}"
                all_diffs.extend(diff.tolist())
            else:
                row += f" {'N/A':>10}"
        if all_diffs:
            pooled_d = cohens_d(all_diffs)
            row += f" {pooled_d:>+8.3f}"
        print(row)

print()

# Universal conditions
for init_name in INIT_NAMES:
    for suffix in ['', '_nonorm']:
        cond_label = f'univ_{init_name}{suffix}'
        key = f'nll_univ_{init_name}{suffix}'
        row = f"  {cond_label:<30}"
        all_diffs = []
        for ds in DATASETS:
            if key in master_nll[ds] and 'nll_bare' in master_nll[ds]:
                diff = master_nll[ds]['nll_bare'] - master_nll[ds][key]
                d = cohens_d(diff)
                row += f" {d:>+10.3f}"
                all_diffs.extend(diff.tolist())
            else:
                row += f" {'N/A':>10}"
        if all_diffs:
            pooled_d = cohens_d(all_diffs)
            row += f" {pooled_d:>+8.3f}"
        print(row)
""")


# =====================================================================
# Cell 12: Ranking Chart
# =====================================================================
code(r"""# Cell 12: Ranking chart — all conditions ranked by pooled Cohen's d
print("=" * 70)
print("CONDITION RANKING BY POOLED COHEN'S d (vs bare)")
print("=" * 70)

rankings = []

# Gather all conditions
all_conditions_to_rank = []
# Exp 13 (excluding bare)
for cond in EXP13_CONDITIONS:
    if cond == 'bare':
        continue
    all_conditions_to_rank.append(('Exp13', cond, f'nll_{cond}'))

# Soft prompts (per-dataset, with norm)
for init_name in INIT_NAMES:
    all_conditions_to_rank.append(('Soft', f'soft_{init_name}', f'nll_soft_{init_name}'))
    all_conditions_to_rank.append(('Soft', f'soft_{init_name}_nonorm', f'nll_soft_{init_name}_nonorm'))

# Universal (with norm)
for init_name in INIT_NAMES:
    all_conditions_to_rank.append(('Univ', f'univ_{init_name}', f'nll_univ_{init_name}'))
    all_conditions_to_rank.append(('Univ', f'univ_{init_name}_nonorm', f'nll_univ_{init_name}_nonorm'))

for source, cond_label, nll_key in all_conditions_to_rank:
    all_diffs = []
    per_ds = {}
    for ds in DATASETS:
        if nll_key in master_nll[ds] and 'nll_bare' in master_nll[ds]:
            diff = master_nll[ds]['nll_bare'] - master_nll[ds][nll_key]
            d = cohens_d(diff)
            w = win_rate(diff)
            _, p = paired_ttest(diff)
            per_ds[ds] = {'d': float(d), 'win': float(w), 'p': float(p)}
            all_diffs.extend(diff.tolist())
    if all_diffs:
        pooled_d = cohens_d(all_diffs)
        pooled_w = win_rate(all_diffs)
        rankings.append({
            'source': source,
            'condition': cond_label,
            'pooled_d': float(pooled_d),
            'pooled_win': float(pooled_w),
            'per_dataset': per_ds,
        })

# Sort by pooled Cohen's d (descending = most effective)
rankings.sort(key=lambda x: -x['pooled_d'])

print(f"\n{'Rank':>4} {'Source':>6} {'Condition':<30} {'d':>8} {'Win%':>7}")
print(f"  {'-'*58}")
for rank, r in enumerate(rankings, 1):
    sig = '*' if r['pooled_d'] > 0 else ''
    print(f"{rank:>4} {r['source']:>6} {r['condition']:<30} "
          f"{r['pooled_d']:>+8.4f} {r['pooled_win']:>6.1%}")

# Highlight best soft prompt vs best Exp 13
best_exp13 = next((r for r in rankings if r['source'] == 'Exp13'), None)
best_soft = next((r for r in rankings if r['source'] in ('Soft', 'Univ')), None)
print(f"\n--- Key Comparison ---")
if best_exp13:
    print(f"  Best Exp 13: {best_exp13['condition']} (d={best_exp13['pooled_d']:+.4f})")
if best_soft:
    print(f"  Best Soft:   {best_soft['condition']} (d={best_soft['pooled_d']:+.4f})")
if best_exp13 and best_soft:
    delta = best_soft['pooled_d'] - best_exp13['pooled_d']
    print(f"  Delta:       {delta:+.4f} ({'soft wins' if delta > 0 else 'static wins'})")
""")


# =====================================================================
# Cell 13: Training Curves
# =====================================================================
code(r"""# Cell 13: Training curves — loss vs epoch for each dataset/init
print("=" * 70)
print("TRAINING CURVES")
print("=" * 70)

# Per-dataset training curves
print("\nPer-dataset training curves (val NLL by epoch):")
for ds_name in DATASETS:
    print(f"\n  {ds_name.upper()}:")
    print(f"    {'Init':<25} {'E0':>8} {'E1':>8} {'E2':>8} {'E3':>8} {'E4':>8} "
          f"{'Best':>8} {'@E':>4}")
    print(f"    {'-'*75}")
    for init_name in INIT_NAMES:
        log = per_dataset_logs.get((ds_name, init_name), {})
        val_losses = log.get('val_losses', [])
        row = f"    {init_name:<25}"
        for e in range(min(5, len(val_losses))):
            row += f" {val_losses[e]:>8.4f}"
        for _ in range(5 - min(5, len(val_losses))):
            row += f" {'':>8}"
        best_val = log.get('best_val_nll', float('inf'))
        best_ep = log.get('best_epoch', -1)
        row += f" {best_val:>8.4f} {best_ep:>4}"
        print(row)

# Universal training curves
print(f"\n  UNIVERSAL:")
print(f"    {'Init':<25} {'E0':>8} {'E1':>8} {'E2':>8} {'E3':>8} {'E4':>8} "
      f"{'Best':>8} {'@E':>4}")
print(f"    {'-'*75}")
for init_name in INIT_NAMES:
    log = universal_logs.get(init_name, {})
    val_losses = log.get('val_losses', [])
    row = f"    {init_name:<25}"
    for e in range(min(5, len(val_losses))):
        row += f" {val_losses[e]:>8.4f}"
    for _ in range(5 - min(5, len(val_losses))):
        row += f" {'':>8}"
    best_val = log.get('best_val_nll', float('inf'))
    best_ep = log.get('best_epoch', -1)
    row += f" {best_val:>8.4f} {best_ep:>4}"
    print(row)
""")


# =====================================================================
# Cell 14: Init Comparison
# =====================================================================
code(r"""# Cell 14: Init comparison — which initialization works best?
print("=" * 70)
print("INITIALIZATION COMPARISON")
print("=" * 70)

# Per-dataset: average best val NLL across datasets for each init
print("\nPer-dataset soft prompts — mean best val NLL:")
print(f"  {'Init':<25} {'Mean Val NLL':>12} {'Mean d (vs bare)':>16}")
print(f"  {'-'*55}")
for init_name in INIT_NAMES:
    val_nlls = []
    ds_effects = []
    for ds_name in DATASETS:
        log = per_dataset_logs.get((ds_name, init_name), {})
        val_nlls.append(log.get('best_val_nll', float('inf')))
        key = f'nll_soft_{init_name}'
        if key in master_nll[ds_name] and 'nll_bare' in master_nll[ds_name]:
            diff = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][key]
            ds_effects.extend(diff.tolist())
    mean_val = np.mean(val_nlls)
    mean_d = cohens_d(ds_effects) if ds_effects else 0.0
    print(f"  {init_name:<25} {mean_val:>12.4f} {mean_d:>+16.4f}")

# Universal: compare inits
print("\nUniversal soft prompts — best val NLL and eval d:")
print(f"  {'Init':<25} {'Val NLL':>10} {'d (vs bare)':>12}")
print(f"  {'-'*50}")
for init_name in INIT_NAMES:
    log = universal_logs.get(init_name, {})
    val_nll = log.get('best_val_nll', float('inf'))
    key = f'nll_univ_{init_name}'
    all_diffs = []
    for ds in DATASETS:
        if key in master_nll[ds] and 'nll_bare' in master_nll[ds]:
            diff = master_nll[ds]['nll_bare'] - master_nll[ds][key]
            all_diffs.extend(diff.tolist())
    d = cohens_d(all_diffs) if all_diffs else 0.0
    print(f"  {init_name:<25} {val_nll:>10.4f} {d:>+12.4f}")

# Warm vs random
print("\n--- Warm vs Random Start ---")
for scope, prompts_logs in [('Per-dataset', per_dataset_logs), ('Universal', universal_logs)]:
    if scope == 'Per-dataset':
        rand_ds = []
        warm_ds = []
        for ds_name in DATASETS:
            rand_key = f'nll_soft_rand'
            warm_key = f'nll_soft_warm_comprehend'
            if rand_key in master_nll[ds_name] and warm_key in master_nll[ds_name]:
                rand_ds.extend(master_nll[ds_name][rand_key].tolist())
                warm_ds.extend(master_nll[ds_name][warm_key].tolist())
        if rand_ds and warm_ds:
            diff = np.array(warm_ds) - np.array(rand_ds)
            d = cohens_d(-diff)  # negative because lower NLL is better
            print(f"  {scope}: warm_comprehend vs rand d={d:+.4f} "
                  f"({'warm better' if np.mean(diff) < 0 else 'rand better'})")
    else:
        rand_key = f'nll_univ_rand'
        warm_key = f'nll_univ_warm_comprehend'
        all_diff = []
        for ds in DATASETS:
            if rand_key in master_nll[ds] and warm_key in master_nll[ds]:
                diff = master_nll[ds][warm_key] - master_nll[ds][rand_key]
                all_diff.extend(diff.tolist())
        if all_diff:
            d = cohens_d([-x for x in all_diff])
            print(f"  {scope}: warm_comprehend vs rand d={d:+.4f} "
                  f"({'warm better' if np.mean(all_diff) < 0 else 'rand better'})")
""")


# =====================================================================
# Cell 15: Per-dataset vs Universal
# =====================================================================
code(r"""# Cell 15: Per-dataset vs universal — does specialization help?
print("=" * 70)
print("PER-DATASET vs UNIVERSAL COMPARISON")
print("=" * 70)

print(f"\n{'Dataset':<12} {'Init':<25} {'Per-ds d':>10} {'Univ d':>10} {'Delta':>8} {'Winner':>10}")
print(f"{'-'*78}")

per_wins = 0
univ_wins = 0
ties = 0

for ds_name in DATASETS:
    for init_name in INIT_NAMES:
        per_key = f'nll_soft_{init_name}'
        univ_key = f'nll_univ_{init_name}'

        if per_key not in master_nll[ds_name] or univ_key not in master_nll[ds_name]:
            continue
        if 'nll_bare' not in master_nll[ds_name]:
            continue

        diff_per = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][per_key]
        diff_univ = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][univ_key]
        d_per = cohens_d(diff_per)
        d_univ = cohens_d(diff_univ)
        delta = d_per - d_univ

        if abs(delta) < 0.01:
            winner = 'tie'
            ties += 1
        elif delta > 0:
            winner = 'per-dataset'
            per_wins += 1
        else:
            winner = 'universal'
            univ_wins += 1

        print(f"{ds_name:<12} {init_name:<25} {d_per:>+10.4f} {d_univ:>+10.4f} "
              f"{delta:>+8.4f} {winner:>10}")

print(f"\nSummary: per-dataset wins {per_wins}, universal wins {univ_wins}, ties {ties}")
""")


# =====================================================================
# Cell 16: Normalization Analysis
# =====================================================================
code(r"""# Cell 16: Norm vs no-norm — does the soft prompt compensate for scale drift?
print("=" * 70)
print("NORMALIZATION ANALYSIS")
print("=" * 70)

print(f"\nDoes normalization still help learned soft prompts?")
print(f"{'Dataset':<12} {'Init':<25} {'d(norm)':>10} {'d(nonorm)':>12} {'norm helps':>11}")
print(f"{'-'*72}")

norm_helps_count = 0
total_comparisons = 0

for ds_name in DATASETS:
    for init_name in INIT_NAMES:
        norm_key = f'nll_soft_{init_name}'
        nonorm_key = f'nll_soft_{init_name}_nonorm'
        if norm_key not in master_nll[ds_name] or nonorm_key not in master_nll[ds_name]:
            continue
        if 'nll_bare' not in master_nll[ds_name]:
            continue

        diff_norm = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][norm_key]
        diff_nonorm = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][nonorm_key]
        d_norm = cohens_d(diff_norm)
        d_nonorm = cohens_d(diff_nonorm)

        helps = d_norm > d_nonorm
        if helps:
            norm_helps_count += 1
        total_comparisons += 1

        print(f"{ds_name:<12} {init_name:<25} {d_norm:>+10.4f} {d_nonorm:>+12.4f} "
              f"{'yes' if helps else 'no':>11}")

print(f"\nNormalization helps in {norm_helps_count}/{total_comparisons} cases "
      f"({norm_helps_count/total_comparisons*100:.0f}%)" if total_comparisons > 0 else "")

# Same for universal
print(f"\nUniversal soft prompts:")
print(f"{'Dataset':<12} {'Init':<25} {'d(norm)':>10} {'d(nonorm)':>12} {'norm helps':>11}")
print(f"{'-'*72}")
for ds_name in DATASETS:
    for init_name in INIT_NAMES:
        norm_key = f'nll_univ_{init_name}'
        nonorm_key = f'nll_univ_{init_name}_nonorm'
        if norm_key not in master_nll[ds_name] or nonorm_key not in master_nll[ds_name]:
            continue
        if 'nll_bare' not in master_nll[ds_name]:
            continue

        diff_norm = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][norm_key]
        diff_nonorm = master_nll[ds_name]['nll_bare'] - master_nll[ds_name][nonorm_key]
        d_norm = cohens_d(diff_norm)
        d_nonorm = cohens_d(diff_nonorm)

        helps = d_norm > d_nonorm
        print(f"{ds_name:<12} {init_name:<25} {d_norm:>+10.4f} {d_nonorm:>+12.4f} "
              f"{'yes' if helps else 'no':>11}")
""")


# =====================================================================
# Cell 17: Save Summary
# =====================================================================
code(r"""# Cell 17: Save results and summary
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# ================================================================
# Training logs summary
# ================================================================
training_summary = {}
for ds_name in DATASETS:
    training_summary[ds_name] = {}
    for init_name in INIT_NAMES:
        log = per_dataset_logs.get((ds_name, init_name), {})
        training_summary[ds_name][init_name] = {
            'best_epoch': log.get('best_epoch', -1),
            'best_val_nll': log.get('best_val_nll', float('inf')),
            'n_epochs_run': log.get('epoch', -1) + 1,
            'train_losses': log.get('train_losses', []),
            'val_losses': log.get('val_losses', []),
        }

training_summary['universal'] = {}
for init_name in INIT_NAMES:
    log = universal_logs.get(init_name, {})
    training_summary['universal'][init_name] = {
        'best_epoch': log.get('best_epoch', -1),
        'best_val_nll': log.get('best_val_nll', float('inf')),
        'n_epochs_run': log.get('epoch', -1) + 1,
        'train_losses': log.get('train_losses', []),
        'val_losses': log.get('val_losses', []),
    }

# ================================================================
# Full results.json
# ================================================================
experiment_config = {
    'seed': SEED, 'n_samples': N_SAMPLES, 'n_hard': N_HARD,
    'hard_frac': HARD_FRAC, 'prefix_l': PREFIX_L,
    'hidden_size': HIDDEN_SIZE,
    'soft_prompt_params': PREFIX_L * HIDDEN_SIZE,
    'common_max_doc': COMMON_MAX_DOC, 'model': MODEL_NAME,
    'scoring_key': SCORING_KEY,
    'n_epochs': N_EPOCHS, 'lr': LR,
    'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP,
    'patience': PATIENCE,
    'n_train': N_TRAIN, 'n_val': N_VAL,
    'init_names': INIT_NAMES,
}

dataset_metadata = {}
for ds in DATASETS:
    dataset_metadata[ds] = {
        'tier': DATASET_TIERS[ds],
        'n_hard': len(hard_samples[ds]),
        'n_train': len(train_samples[ds]),
        'n_val': len(val_samples[ds]),
        'n_total': len(all_samples[ds]),
    }

final_results = {
    'experiment': 'exp14_soft_prompt_tuning',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment_config': experiment_config,
    'dataset_metadata': dataset_metadata,
    'training_summary': training_summary,
    'rankings': rankings,
    'per_sample_results': {ds: eval_results[ds] for ds in DATASETS},
}

results_path = RESULTS_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results JSON: {results_path} ({results_path.stat().st_size / 1024:.0f} KB)")

# Compact summary (no per-sample)
summary = {k: v for k, v in final_results.items() if k != 'per_sample_results'}
summary_path = RESULTS_DIR / 'summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON: {summary_path}")

print("\n" + "=" * 70)
print("EXPERIMENT 14 COMPLETE")
print("=" * 70)
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/decoder_only/14/14_soft_prompt_tuning.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
