#!/usr/bin/env python3
"""Build deep sweep: systematic tests of all hypotheses across all models.

Tests H1-H5 from EXPERIMENT_NOTES.md in a single sweep:
  H1: Scrambled prefix (meaning vs vocabulary)
  H2: Anti-instruction (semantic direction)
  H3: Prefix length curve (L=1, 4, 16, 64)
  H4: Position-only shift (RoPE position effect)
  H5: Qwen with BOS fix

Conditions (14 total):
  bare                    - no prefix
  random_64               - random tokens L=64
  repeat_64               - "the" x 64
  comprehend_64           - coherent instruction L=64
  comprehend_scrambled_64 - scrambled comprehend tokens L=64
  extract_64              - coherent extract instruction L=64
  extract_scrambled_64    - scrambled extract tokens L=64
  anti_instruction_64     - "Ignore this text completely" L=64
  oracle_64               - query as prefix L=64
  comprehend_1            - comprehend instruction L=1
  comprehend_4            - comprehend instruction L=4
  comprehend_16           - comprehend instruction L=16
  random_1                - random tokens L=1
  position_shift_only     - no prefix, but doc at position 65+ (RoPE shift control)

All models get BOS: Qwen uses PAD token as artificial BOS.
Normalization applied to all (standard pipeline).
GSM8K uses number-only answer (fixed).

N=200 per dataset (80 hard). 2 datasets: SQuAD v2 (mid effect) + GSM8K (large effect).

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/01_multi_model/build_deep_sweep.py
    cd experiments/01_multi_model
    papermill 01_deep_sweep.ipynb 01_deep_sweep_executed.ipynb --no-progress-bar
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


md(r"""# Deep Sweep: Hypothesis Testing Across 4 Models

Systematic evaluation of prefix conditioning mechanisms:
- H1: Does word order matter? (scrambled vs coherent)
- H2: Does semantic direction matter? (anti-instruction)
- H3: How does prefix length affect structural vs semantic components?
- H4: Does RoPE position shift alone matter?
- H5: Does BOS fix Qwen?

All models use BOS (Qwen gets PAD as artificial BOS). GSM8K uses number-only answer.""")


code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import json
import time
import gc
import re
import random as pyrandom
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv, find_dotenv

from lib.rope import select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix, scramble_prefix
from lib.quantization import norm_roundtrip_kv_cache
from lib.analysis import cohens_d, win_rate, paired_ttest
from lib.data import count_words
from model_adapters import (
    build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit
)

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

SEED = 42
N_SAMPLES = 200
N_HARD = 80

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
    'squad_v2': {'path': 'rajpurkar/squad_v2', 'config': None, 'split': 'validation'},
    'gsm8k':    {'path': 'openai/gsm8k', 'config': 'main', 'split': 'test'},
}

RESULTS_BASE = Path("../../results/exp01_deep_sweep")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

INSTRUCTIONS = {
    'comprehend': "Read and comprehend this text carefully.",
    'extract': "Extract the key facts from this text.",
    'anti': "Ignore this text completely. Do not read it.",
}

torch.manual_seed(SEED)
pyrandom.seed(SEED)
np.random.seed(SEED)

print(f"Models: {list(MODELS.keys())}")
print(f"Datasets: {list(DS_SPECS.keys())}")
print(f"N_SAMPLES={N_SAMPLES}, N_HARD={N_HARD}")
""")


code(r"""# Load datasets (with GSM8K number-only fix)
def load_qa_dataset(ds_key):
    spec = DS_SPECS[ds_key]
    if spec['config']:
        raw = load_dataset(spec['path'], spec['config'], split=spec['split'])
    else:
        raw = load_dataset(spec['path'], split=spec['split'])
    samples = []
    for item in raw:
        if ds_key == 'squad_v2':
            passage = item.get('context', '')
            query = item.get('question', '')
            answers = item.get('answers', {}).get('text', [])
            answer = answers[0] if answers else ''
        elif ds_key == 'gsm8k':
            passage = item.get('question', '')
            query = 'What is the answer?'
            raw_answer = item.get('answer', '')
            if '####' not in raw_answer:
                continue
            answer = raw_answer.split('####')[-1].strip()
        if passage and query and answer:
            wc = count_words(passage)
            if 10 <= wc <= 500:
                samples.append({'passage': passage, 'query': query, 'answer': answer,
                               'passage_words': wc})
    pyrandom.seed(SEED)
    pyrandom.shuffle(samples)
    return samples[:N_SAMPLES]

print("Loading datasets...")
all_samples = {}
for ds_key in DS_SPECS:
    all_samples[ds_key] = load_qa_dataset(ds_key)
    print(f"  {ds_key}: {len(all_samples[ds_key])} samples")
    print(f"    sample answer: {repr(all_samples[ds_key][0]['answer'][:80])}")
""")


code(r"""# Scoring functions with BOS support for all models
_model = None
_tokenizer = None
_device = None
_layer_inv_freqs = None
_layer_types = None
_sliding_limit = None
_bos_id = None  # actual or artificial BOS
_nl_ids = None

def _encode_phase_a(doc_ids, prefix_ids=None):
    input_ids = [_bos_id]
    if prefix_ids is not None:
        input_ids += list(prefix_ids) + _nl_ids
    input_ids += list(doc_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    outputs = _model(input_ids=input_tensor, use_cache=True)
    cache = outputs.past_key_values

    if prefix_ids is not None:
        P = len(prefix_ids)
        NL = len(_nl_ids)
        doc_start = 1 + P + NL
    else:
        doc_start = 1
    D = len(doc_ids)
    keep = [0] + list(range(doc_start, doc_start + D))

    if _sliding_limit is not None and len(keep) > _sliding_limit:
        raise ValueError(f"Cache overflow: {len(keep)} > sliding limit {_sliding_limit}")

    cache = select_kv_cache(cache, keep, device=_device)
    if prefix_ids is not None:
        old_pos = torch.arange(doc_start, doc_start + D, device=_device)
        new_pos = torch.arange(1, 1 + D, device=_device)
        cache = reposition_kv_cache(cache, old_pos, new_pos,
                                     _layer_inv_freqs, _layer_types, bos_start=0)
    cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def _encode_phase_a_position_shift(doc_ids, shift=64):
    # H4: no prefix, but place doc at shifted positions (as if prefix were removed)
    input_ids = [_bos_id] + list(doc_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    # Use shifted position IDs: BOS at 0, doc at shift+1..shift+D
    D = len(doc_ids)
    pos = torch.cat([
        torch.tensor([0], device=_device),
        torch.arange(shift + 1, shift + 1 + D, device=_device)
    ]).unsqueeze(0)
    outputs = _model(input_ids=input_tensor, position_ids=pos, use_cache=True)
    cache = outputs.past_key_values
    # Reposition doc keys from shift+1..shift+D to 1..D
    old_pos = torch.arange(shift + 1, shift + 1 + D, device=_device)
    new_pos = torch.arange(1, 1 + D, device=_device)
    cache = reposition_kv_cache(cache, old_pos, new_pos,
                                 _layer_inv_freqs, _layer_types, bos_start=0)
    cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def _score_phase_b(cache, D, query_ids, answer_ids):
    phase_b_ids = _nl_ids + list(query_ids) + _nl_ids + list(answer_ids)
    input_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=_device)
    n_tokens = len(phase_b_ids)
    position_ids = torch.arange(D + 1, D + 1 + n_tokens, device=_device).unsqueeze(0)
    cache_copy = deep_copy_cache(cache)
    outputs = _model(input_ids=input_tensor, position_ids=position_ids,
                     past_key_values=cache_copy, use_cache=False)
    logits = outputs.logits[0]
    answer_start = len(_nl_ids) + len(query_ids) + len(_nl_ids)
    answer_logits = logits[answer_start - 1 : answer_start - 1 + len(answer_ids)]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=_device)
    loss = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction='mean')
    return loss.item()

print("Scoring functions defined.")
""")


code(r"""# Main loop
CONDITION_NAMES = [
    'random_64', 'repeat_64',
    'comprehend_64', 'comprehend_scrambled_64',
    'extract_64', 'extract_scrambled_64',
    'anti_64', 'oracle_64',
    'comprehend_16', 'comprehend_4', 'comprehend_1',
    'random_16', 'random_4', 'random_1',
    'position_shift',
]

all_summaries = {}

for model_key, model_spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {model_key} ({model_spec['name']})")
    print(f"{'#'*70}")

    model_dir = RESULTS_BASE / model_key
    model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f'deep_{model_key}'

    global _model, _tokenizer, _device, _layer_inv_freqs, _layer_types
    global _sliding_limit, _bos_id, _nl_ids

    _tokenizer = AutoTokenizer.from_pretrained(model_spec['name'], token=HF_TOKEN)
    loader = model_spec.get('loader', 'AutoModelForCausalLM')
    if loader == 'Gemma3ForConditionalGeneration':
        from transformers import Gemma3ForConditionalGeneration
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()
    elif loader == 'Gemma3nForConditionalGeneration':
        from transformers import Gemma3nForConditionalGeneration
        _model = Gemma3nForConditionalGeneration.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()

    _device = next(_model.parameters()).device
    _layer_inv_freqs = build_layer_inv_freqs(_model, device=_device)
    _layer_types = get_layer_types(_model)
    _sliding_limit = get_sliding_cache_limit(_model)
    _nl_ids = _tokenizer.encode("\n", add_special_tokens=False)

    # BOS: use native if available, otherwise PAD (Qwen fix)
    native_bos = _tokenizer.bos_token_id
    if native_bos is not None:
        _bos_id = native_bos
    else:
        _bos_id = _tokenizer.pad_token_id
        print(f"  Using PAD ({_bos_id}) as artificial BOS")

    info = get_model_info(_model)
    if _sliding_limit is not None:
        max_doc = _sliding_limit - 1 - 64 - len(_nl_ids)
    else:
        max_doc = 765
    print(f"  Loaded: {info['num_layers']} layers, BOS={_bos_id}, max_doc={max_doc}")

    # Build prefixes
    prefixes = {}
    for iname, itext in INSTRUCTIONS.items():
        ids = _tokenizer.encode(itext, add_special_tokens=False)
        for L in [64, 16, 4, 1]:
            prefixes[f'{iname}_{L}'] = make_prefix(ids, L)
        # Scrambled version (L=64 only)
        prefixes[f'{iname}_scrambled_64'] = scramble_prefix(make_prefix(ids, 64), seed=SEED)

    rng = pyrandom.Random(SEED)
    for L in [64, 16, 4, 1]:
        prefixes[f'random_{L}'] = [rng.randint(100, _tokenizer.vocab_size - 1) for _ in range(L)]

    the_id = _tokenizer.encode("the", add_special_tokens=False)[0]
    prefixes['repeat_64'] = [the_id] * 64

    # Score all datasets
    for ds_key in DS_SPECS:
        print(f"\n  --- {ds_key} ---")
        samples = all_samples[ds_key]
        ckpt_path = model_dir / f"checkpoint_{ds_key}.json"

        scored = []
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
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

            result = {
                'query': s['query'][:200],
                'answer': s['answer'][:200],
                'passage_words': s['passage_words'],
            }

            with torch.no_grad():
                # Bare
                cache, D = _encode_phase_a(doc_ids)
                result['nll_bare'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Standard prefixes
                for cname in ['random_64', 'repeat_64', 'comprehend_64',
                              'comprehend_scrambled_64', 'extract_64',
                              'extract_scrambled_64', 'anti_64']:
                    cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes[cname])
                    result[f'nll_{cname}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                    del cache

                # Oracle (query as prefix)
                oracle_prefix = make_prefix(query_ids, 64)
                cache, D = _encode_phase_a(doc_ids, prefix_ids=oracle_prefix)
                result['nll_oracle_64'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Length variants
                for L in [16, 4, 1]:
                    for ptype in ['comprehend', 'random']:
                        cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes[f'{ptype}_{L}'])
                        result[f'nll_{ptype}_{L}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                        del cache

                # H4: Position shift only (no actual prefix)
                cache, D = _encode_phase_a_position_shift(doc_ids, shift=64)
                result['nll_position_shift'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

            scored.append(result)

            if (idx + 1) % 20 == 0:
                ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
                print(f"    [{idx+1}/{len(samples)}]")

            torch.cuda.empty_cache()

        # Final checkpoint
        ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))

    # Build summary
    summary = {'model': model_key, 'model_name': model_spec['name'], 'rankings': []}
    for cname in CONDITION_NAMES:
        all_diffs = []
        per_ds = {}
        for ds_key in DS_SPECS:
            ckpt = json.loads((model_dir / f"checkpoint_{ds_key}.json").read_text())
            samples_all = ckpt['samples']
            samples_all.sort(key=lambda x: x['nll_bare'], reverse=True)
            hard = samples_all[:N_HARD]
            nll_key = f'nll_{cname}'
            diffs = np.array([x['nll_bare'] - x[nll_key] for x in hard])
            all_diffs.extend(diffs.tolist())
            d = cohens_d(diffs)
            w = win_rate(diffs)
            _, p = paired_ttest(diffs)
            per_ds[ds_key] = {'d': round(d, 4), 'win': round(w, 4), 'p': p}
        pooled = np.array(all_diffs)
        summary['rankings'].append({
            'condition': cname,
            'pooled_d': round(cohens_d(pooled), 4),
            'pooled_win': round(win_rate(pooled), 4),
            'per_dataset': per_ds,
        })
    summary['rankings'].sort(key=lambda r: r['pooled_d'], reverse=True)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    all_summaries[model_key] = summary
    print(f"\n  Summary saved.")

    del _model, _tokenizer
    _model = None; _tokenizer = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Model unloaded.")

print(f"\n{'='*70}")
print("ALL MODELS COMPLETE")
""")


code(r"""# Hypothesis analysis
DS_LABELS = {'squad_v2': 'SQuAD', 'gsm8k': 'GSM8K'}

for model_key, summary in all_summaries.items():
    print(f"\n{'='*60}")
    print(f"{model_key}")
    print(f"{'='*60}")
    print(f"{'Condition':<28} {'Pool d':>7} {'Win':>5}", end='')
    for ds in DS_SPECS:
        print(f" {DS_LABELS[ds]:>7}", end='')
    print()
    print("-" * 60)
    for r in summary['rankings']:
        print(f"{r['condition']:<28} {r['pooled_d']:>+7.3f} {r['pooled_win']:>5.0%}", end='')
        for ds in DS_SPECS:
            d = r['per_dataset'][ds]['d']
            print(f" {d:>+7.2f}", end='')
        print()

# Hypothesis tests
print(f"\n{'='*60}")
print("HYPOTHESIS TESTS")
print(f"{'='*60}")

for model_key, summary in all_summaries.items():
    r = {x['condition']: x for x in summary['rankings']}
    print(f"\n{model_key}:")

    # H1: Meaning (word order)
    if 'comprehend_64' in r and 'comprehend_scrambled_64' in r:
        meaning_comp = r['comprehend_64']['pooled_d'] - r['comprehend_scrambled_64']['pooled_d']
        meaning_ext = r['extract_64']['pooled_d'] - r['extract_scrambled_64']['pooled_d']
        print(f"  H1 Meaning (comp coherent - scrambled):    {meaning_comp:+.3f}")
        print(f"  H1 Meaning (extract coherent - scrambled): {meaning_ext:+.3f}")

    # H2: Anti-instruction
    if 'anti_64' in r and 'comprehend_64' in r:
        anti_vs_comp = r['comprehend_64']['pooled_d'] - r['anti_64']['pooled_d']
        print(f"  H2 Semantic direction (comp - anti):       {anti_vs_comp:+.3f}")

    # H3: Length curve
    for ptype in ['comprehend', 'random']:
        lengths = []
        for L in [1, 4, 16, 64]:
            key = f'{ptype}_{L}'
            if key in r:
                lengths.append((L, r[key]['pooled_d']))
        if lengths:
            curve = ', '.join(f'L={L}: {d:+.3f}' for L, d in lengths)
            print(f"  H3 Length curve ({ptype}): {curve}")

    # H4: Position shift
    if 'position_shift' in r:
        print(f"  H4 Position shift only:                    {r['position_shift']['pooled_d']:+.3f}")

# Save combined
combined = {k: v for k, v in all_summaries.items()}
(RESULTS_BASE / "combined_summary.json").write_text(json.dumps(combined, indent=2, default=str))
print(f"\nCombined saved to {RESULTS_BASE / 'combined_summary.json'}")
""")


out_path = "experiments/01_multi_model/01_deep_sweep.ipynb"
nbf.write(nb, out_path)
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
