#!/usr/bin/env python3
"""Build Exp 01: Multi-model evaluation of prefix conditioning.

Tests the prefix conditioning effect across 4 models from 3 families:
  - Gemma 3 12B-IT (hybrid sliding/full attention, 48 layers)
  - Gemma 3N E4B-IT (hybrid sliding/full attention, 35 layers — Gemma 4 family)
  - Mistral 7B-Instruct-v0.3 (full attention, GQA, 32 layers)
  - Qwen 2.5 7B-Instruct (full attention, GQA, 28 layers)

Runs all models sequentially in a single notebook, unloading each model
before loading the next to manage GPU memory.

Design:
    Conditions: bare, random_64, comprehend_64
    Datasets:   MS MARCO, SQuAD v2, DROP, GSM8K
    N:          400 sampled, 160 hard (top 40% by bare NLL)

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/01_multi_model/build_exp01.py
    cd experiments/01_multi_model
    papermill 01_multi_model.ipynb 01_multi_model_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/01_multi_model", exist_ok=True)

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
# Cell 0: Title
# =====================================================================
md(r"""# Exp 01: Multi-Model Prefix Conditioning Evaluation

Validates that KV cache prefix conditioning generalizes across model families.
Tests bare / random / comprehend conditions on 4 datasets across 4 models
(Gemma 3 12B, Gemma 3N E4B, Mistral 7B, Qwen 2.5 7B).

Models are loaded one at a time to fit in GPU memory.""")


# =====================================================================
# Cell 1: Setup
# =====================================================================
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import json
import time
import gc
import random as pyrandom
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv, find_dotenv

from lib.rope import rotate_half, select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from lib.analysis import cohens_d, win_rate, paired_ttest
from lib.data import count_words
from model_adapters import (
    build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit
)

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

SEED = 42
N_SAMPLES = 400
N_HARD = 160
PREFIX_L = 64

MODELS = {
    'gemma3_12b': {
        'name': 'google/gemma-3-12b-it',
        'loader': 'Gemma3ForConditionalGeneration',
    },
    'gemma3n_e4b': {
        'name': 'google/gemma-3n-e4b-it',
        'loader': 'Gemma3nForConditionalGeneration',
    },
    'mistral_7b': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'loader': 'AutoModelForCausalLM',
    },
    'qwen25_7b': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
}

DS_SPECS = {
    'ms_marco': {'path': 'microsoft/ms_marco', 'config': 'v1.1', 'split': 'validation'},
    'squad_v2': {'path': 'rajpurkar/squad_v2', 'config': None, 'split': 'validation'},
    'drop':     {'path': 'ucinlp/drop', 'config': None, 'split': 'validation'},
    'gsm8k':    {'path': 'openai/gsm8k', 'config': 'main', 'split': 'test'},
}

RESULTS_BASE = Path("../../results/exp01_multi_model")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

torch.manual_seed(SEED)
pyrandom.seed(SEED)
np.random.seed(SEED)

print(f"Models: {list(MODELS.keys())}")
print(f"Datasets: {list(DS_SPECS.keys())}")
print(f"N_SAMPLES={N_SAMPLES}, N_HARD={N_HARD}, PREFIX_L={PREFIX_L}")
""")


# =====================================================================
# Cell 2: Load Datasets (once, before model loop)
# =====================================================================
code(r"""# Load all datasets up front (CPU, no model needed)
def load_qa_dataset(ds_key):
    spec = DS_SPECS[ds_key]
    if spec['config']:
        raw = load_dataset(spec['path'], spec['config'], split=spec['split'])
    else:
        raw = load_dataset(spec['path'], split=spec['split'])

    samples = []
    for item in raw:
        if ds_key == 'ms_marco':
            passages = item.get('passages', {}).get('passage_text', [])
            passage = ' '.join(passages) if passages else ''
            query = item.get('query', '')
            answers = item.get('answers', [])
            answer = answers[0] if answers and answers[0] != 'No Answer Present.' else ''
        elif ds_key == 'squad_v2':
            passage = item.get('context', '')
            query = item.get('question', '')
            answers = item.get('answers', {}).get('text', [])
            answer = answers[0] if answers else ''
        elif ds_key == 'drop':
            passage = item.get('passage', '')
            query = item.get('question', '')
            answers_spans = item.get('answers_spans', {}).get('spans', [])
            answer = answers_spans[0] if answers_spans else ''
        elif ds_key == 'gsm8k':
            passage = item.get('question', '')
            query = 'What is the answer?'
            answer = item.get('answer', '')

        if passage and query and answer:
            samples.append({
                'passage': passage,
                'query': query,
                'answer': answer,
                'passage_words': count_words(passage),
            })

    pyrandom.seed(SEED)
    pyrandom.shuffle(samples)
    return samples[:N_SAMPLES]

print("Loading datasets...")
all_samples = {}
for ds_key in DS_SPECS:
    all_samples[ds_key] = load_qa_dataset(ds_key)
    print(f"  {ds_key}: {len(all_samples[ds_key])} samples")
print("Datasets ready.")
""")


# =====================================================================
# Cell 3: Scoring Functions
# =====================================================================
code(r"""# Scoring functions — use module-level globals for model, tokenizer, etc.
# These are set in the model loop below.
_model = None
_tokenizer = None
_device = None
_layer_inv_freqs = None
_layer_types = None
_sliding_limit = None
_bos_id = None
_nl_ids = None

def _encode_phase_a(doc_ids, prefix_ids=None):
    # Build input: [BOS] + (prefix + NL if prefix) + doc
    # For models without BOS (e.g. Qwen), skip BOS and adjust positions
    has_bos = _bos_id is not None
    input_ids = []
    if has_bos:
        input_ids.append(_bos_id)
    if prefix_ids is not None:
        input_ids += list(prefix_ids) + _nl_ids
    input_ids += list(doc_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    outputs = _model(input_ids=input_tensor, use_cache=True)
    cache = outputs.past_key_values

    bos_offset = 1 if has_bos else 0
    if prefix_ids is not None:
        P = len(prefix_ids)
        NL = len(_nl_ids)
        doc_start = bos_offset + P + NL
    else:
        doc_start = bos_offset
    D = len(doc_ids)

    # Keep BOS (if present) + doc entries
    if has_bos:
        keep = [0] + list(range(doc_start, doc_start + D))
    else:
        keep = list(range(doc_start, doc_start + D))

    if _sliding_limit is not None and len(keep) > _sliding_limit:
        raise ValueError(f"Cache overflow: {len(keep)} > sliding limit {_sliding_limit}")

    cache = select_kv_cache(cache, keep, device=_device)
    if prefix_ids is not None:
        old_pos = torch.arange(doc_start, doc_start + D, device=_device)
        new_pos = torch.arange(bos_offset, bos_offset + D, device=_device)
        cache = reposition_kv_cache(cache, old_pos, new_pos,
                                     _layer_inv_freqs, _layer_types,
                                     bos_start=-1 if not has_bos else 0)

    cache = norm_roundtrip_kv_cache(cache)
    return cache, D


def _score_phase_b(cache, D, query_ids, answer_ids):
    has_bos = _bos_id is not None
    bos_offset = 1 if has_bos else 0
    phase_b_ids = _nl_ids + list(query_ids) + _nl_ids + list(answer_ids)
    input_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=_device)
    n_tokens = len(phase_b_ids)
    position_ids = torch.arange(D + bos_offset, D + bos_offset + n_tokens,
                                device=_device).unsqueeze(0)

    cache_copy = deep_copy_cache(cache)
    outputs = _model(input_ids=input_tensor, position_ids=position_ids,
                     past_key_values=cache_copy, use_cache=False)

    logits = outputs.logits[0]
    answer_start = len(_nl_ids) + len(query_ids) + len(_nl_ids)
    answer_logits = logits[answer_start - 1 : -1]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=_device)
    loss = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction='mean')
    return loss.item()


def _score_single_pass(doc_ids, query_ids, answer_ids):
    has_bos = _bos_id is not None
    input_ids = []
    if has_bos:
        input_ids.append(_bos_id)
    input_ids += list(doc_ids) + _nl_ids + list(query_ids) + _nl_ids + list(answer_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    # use_cache=True to work around Gemma 3N use_cache=False bug
    outputs = _model(input_ids=input_tensor, use_cache=True)
    logits = outputs.logits[0]
    D = len(doc_ids)
    bos_offset = 1 if has_bos else 0
    answer_start = bos_offset + D + len(_nl_ids) + len(query_ids) + len(_nl_ids)
    answer_logits = logits[answer_start - 1 : -1]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=_device)
    loss = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction='mean')
    return loss.item()

print("Scoring functions defined.")
""")


# =====================================================================
# Cell 4: Main Model Loop
# =====================================================================
code(r"""# Main loop: for each model, load -> score all datasets -> save -> unload
all_summaries = {}

for model_key, model_spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_key} ({model_spec['name']})")
    print(f"{'#'*70}")

    model_results_dir = RESULTS_BASE / model_key
    model_results_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f'pub_exp01_{model_key}'

    # --- Load model ---
    global _model, _tokenizer, _device, _layer_inv_freqs, _layer_types
    global _sliding_limit, _bos_id, _nl_ids

    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(model_spec['name'], token=HF_TOKEN)

    loader_name = model_spec.get('loader', 'AutoModelForCausalLM')
    if loader_name == 'Gemma3ForConditionalGeneration':
        from transformers import Gemma3ForConditionalGeneration
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN,
            device_map='cuda:0',
        ).eval()
    elif loader_name == 'Gemma3nForConditionalGeneration':
        from transformers import Gemma3nForConditionalGeneration
        _model = Gemma3nForConditionalGeneration.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN,
            device_map='cuda:0',
        ).eval()
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN,
            device_map='cuda:0',
        ).eval()

    _device = next(_model.parameters()).device
    _layer_inv_freqs = build_layer_inv_freqs(_model, device=_device)
    _layer_types = get_layer_types(_model)
    _sliding_limit = get_sliding_cache_limit(_model)
    _bos_id = _tokenizer.bos_token_id
    _nl_ids = _tokenizer.encode("\n", add_special_tokens=False)

    info = get_model_info(_model)
    print(f"Loaded in {time.time()-t0:.0f}s — {info['num_layers']} layers, "
          f"head_dim={info['head_dim']}, kv_heads={info['num_kv_heads']}, "
          f"sliding={'yes' if info['has_sliding'] else 'no'}")

    # --- Build prefixes for this model's tokenizer ---
    comprehend_text = "Read and comprehend this text carefully."
    comprehend_ids = _tokenizer.encode(comprehend_text, add_special_tokens=False)
    comprehend_prefix = make_prefix(comprehend_ids, PREFIX_L)

    rng = pyrandom.Random(SEED)
    vocab_size = _tokenizer.vocab_size
    random_prefix = [rng.randint(100, vocab_size - 1) for _ in range(PREFIX_L)]

    # Max doc length
    if _sliding_limit is not None:
        max_doc = _sliding_limit - 1 - PREFIX_L - len(_nl_ids)
    else:
        max_doc = 765
    print(f"Max doc tokens: {max_doc}, BOS={_bos_id}, NL={_nl_ids}")

    # --- Score all datasets ---
    model_rankings = []
    model_per_dataset = {}

    for ds_key in DS_SPECS:
        print(f"\n  --- {ds_key} ---")
        samples = all_samples[ds_key]
        checkpoint_path = model_results_dir / f"checkpoint_{ds_key}.json"

        # Resume from checkpoint
        scored = []
        if checkpoint_path.exists():
            ckpt = json.loads(checkpoint_path.read_text())
            if ckpt.get('scoring_key') == scoring_key:
                scored = ckpt['samples']
                print(f"  Resumed: {len(scored)} samples")

        for idx in range(len(scored), len(samples)):
            s = samples[idx]
            doc_ids = _tokenizer.encode(s['passage'], add_special_tokens=False)[:max_doc]
            query_ids = _tokenizer.encode(s['query'], add_special_tokens=False)
            answer_ids = _tokenizer.encode(s['answer'], add_special_tokens=False)
            if not answer_ids:
                continue

            with torch.no_grad():
                cache_bare, D = _encode_phase_a(doc_ids)
                nll_bare = _score_phase_b(cache_bare, D, query_ids, answer_ids)
                del cache_bare

                cache_rand, D = _encode_phase_a(doc_ids, prefix_ids=random_prefix)
                nll_random = _score_phase_b(cache_rand, D, query_ids, answer_ids)
                del cache_rand

                cache_comp, D = _encode_phase_a(doc_ids, prefix_ids=comprehend_prefix)
                nll_comprehend = _score_phase_b(cache_comp, D, query_ids, answer_ids)
                del cache_comp

                nll_single = _score_single_pass(doc_ids, query_ids, answer_ids)

            scored.append({
                'query': s['query'][:200],
                'answer': s['answer'][:200],
                'passage_words': s['passage_words'],
                'original_idx': idx,
                'nll_bare': nll_bare,
                'nll_random': nll_random,
                'nll_comprehend': nll_comprehend,
                'nll_single_pass': nll_single,
            })

            if (idx + 1) % 20 == 0:
                checkpoint_path.write_text(json.dumps(
                    {'scoring_key': scoring_key, 'samples': scored}))
                d_c = cohens_d(np.array([x['nll_bare']-x['nll_comprehend'] for x in scored]))
                print(f"    [{idx+1}/{len(samples)}] comprehend d={d_c:+.3f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final checkpoint
        checkpoint_path.write_text(json.dumps(
            {'scoring_key': scoring_key, 'samples': scored}))

        # Hard subset
        scored.sort(key=lambda x: x['nll_bare'], reverse=True)
        hard = scored[:N_HARD]
        model_per_dataset[ds_key] = hard

        for cond in ['random', 'comprehend']:
            diffs = np.array([x['nll_bare'] - x[f'nll_{cond}'] for x in hard])
            d = cohens_d(diffs)
            w = win_rate(diffs)
            print(f"    {cond}: d={d:+.3f}, win={w:.1%}")

    # --- Build summary for this model ---
    summary = {
        'model': model_key,
        'model_name': model_spec['name'],
        'model_info': info,
        'rankings': [],
    }

    for cond in ['random', 'comprehend', 'single_pass']:
        nll_key = f'nll_{cond}'
        all_diffs = []
        per_ds = {}
        for ds_key in DS_SPECS:
            hard = model_per_dataset[ds_key]
            diffs = np.array([x['nll_bare'] - x[nll_key] for x in hard])
            d = cohens_d(diffs)
            w = win_rate(diffs)
            _, p = paired_ttest(diffs)
            all_diffs.extend(diffs.tolist())
            per_ds[ds_key] = {'d': round(d, 4), 'win': round(w, 4), 'p': p}

        pooled = np.array(all_diffs)
        summary['rankings'].append({
            'condition': cond,
            'pooled_d': round(cohens_d(pooled), 4),
            'pooled_win': round(win_rate(pooled), 4),
            'per_dataset': per_ds,
        })

    summary['rankings'].sort(key=lambda r: r['pooled_d'], reverse=True)
    (model_results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    all_summaries[model_key] = summary
    print(f"\n  Summary saved to {model_results_dir / 'summary.json'}")

    # --- Unload model ---
    del _model, _tokenizer
    _model = None
    _tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Model unloaded. GPU memory freed.")

print(f"\n{'='*70}")
print("ALL MODELS COMPLETE")
print(f"{'='*70}")
""")


# =====================================================================
# Cell 5: Cross-Model Comparison
# =====================================================================
code(r"""# Cross-model comparison table
print(f"\n{'='*70}")
print("CROSS-MODEL COMPARISON")
print(f"{'='*70}")

DS_LABELS = {'ms_marco': 'MARCO', 'squad_v2': 'SQuAD', 'drop': 'DROP', 'gsm8k': 'GSM8K'}

# Table: pooled d per model per condition
print(f"\n{'Model':<15} {'Condition':<15} {'Pooled d':>10} {'Win':>8}")
print("-" * 50)
for model_key, summary in all_summaries.items():
    for r in summary['rankings']:
        if r['condition'] == 'single_pass':
            continue
        print(f"{model_key:<15} {r['condition']:<15} {r['pooled_d']:>+10.3f} {r['pooled_win']:>8.1%}")
    print()

# Per-dataset breakdown
print(f"\nPer-dataset Cohen's d:")
header = f"{'Model':<13} {'Cond':<13}"
for ds in DS_SPECS:
    header += f" {DS_LABELS[ds]:>8}"
print(header)
print("-" * len(header))

for model_key, summary in all_summaries.items():
    for r in summary['rankings']:
        if r['condition'] == 'single_pass':
            continue
        row = f"{model_key:<13} {r['condition']:<13}"
        for ds in DS_SPECS:
            d = r['per_dataset'][ds]['d']
            row += f" {d:>+8.3f}"
        print(row)
    print()

# Save combined summary
combined = {
    'experiment': 'pub_exp01_multi_model',
    'models': list(MODELS.keys()),
    'datasets': list(DS_SPECS.keys()),
    'conditions': ['bare', 'random', 'comprehend'],
    'per_model': all_summaries,
}
combined_path = RESULTS_BASE / "combined_summary.json"
combined_path.write_text(json.dumps(combined, indent=2, default=str))
print(f"\nCombined summary saved to {combined_path}")
""")


# =====================================================================
# Cell 6: Key Finding Check
# =====================================================================
code(r"""# Verify the key finding: comprehend > random > bare across all models
print("KEY FINDING CHECK:")
print("Expected ranking: comprehend > random > bare (all models)\n")

all_pass = True
for model_key, summary in all_summaries.items():
    d_comp = next(r['pooled_d'] for r in summary['rankings'] if r['condition'] == 'comprehend')
    d_rand = next(r['pooled_d'] for r in summary['rankings'] if r['condition'] == 'random')

    comp_positive = d_comp > 0
    rand_positive = d_rand > 0
    comp_beats_rand = d_comp > d_rand

    status = 'PASS' if (comp_positive and rand_positive and comp_beats_rand) else 'FAIL'
    if status == 'FAIL':
        all_pass = False

    print(f"  {model_key}: comprehend d={d_comp:+.3f}, random d={d_rand:+.3f} "
          f"-> comp>rand={comp_beats_rand}, both positive={comp_positive and rand_positive} [{status}]")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES — investigate'}")
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/01_multi_model/01_multi_model.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
