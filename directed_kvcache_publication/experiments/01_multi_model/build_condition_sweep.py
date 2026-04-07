#!/usr/bin/env python3
"""Build condition sweep: test 8 prefix strategies across all 4 models.

Tests a broad range of prefix types to find what (if anything) helps each model:
  - bare: no prefix (baseline)
  - random: random token IDs (pure structural)
  - repeat_token: repeat "the" 64 times (minimal structural)
  - comprehend: "Read and comprehend this text carefully" (instruction)
  - extract: "Extract the key facts from this text" (instruction)
  - summarize: "Summarize the main points of this text" (instruction)
  - oracle: the actual query as prefix (semantic, known to hurt on Gemma 3)
  - doc_keywords: first 64 tokens of the document itself (vocabulary priming)

Reduced N=200 (80 hard) for faster turnaround. 4 datasets, 4 models.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/01_multi_model/build_condition_sweep.py
    cd experiments/01_multi_model
    papermill 01_condition_sweep.ipynb 01_condition_sweep_executed.ipynb --no-progress-bar
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


md(r"""# Condition Sweep: 8 Prefix Strategies × 4 Models

Broad exploration of what prefix types help each model. Tests structural,
instruction-based, query-based, and vocabulary-based prefixes.""")


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

from lib.rope import select_kv_cache, reposition_kv_cache
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
N_SAMPLES = 200
N_HARD = 80
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

RESULTS_BASE = Path("../../results/exp01_condition_sweep")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

# Instruction texts for prefixes
INSTRUCTIONS = {
    'comprehend': "Read and comprehend this text carefully.",
    'extract': "Extract the key facts from this text.",
    'summarize': "Summarize the main points of this text.",
}

torch.manual_seed(SEED)
pyrandom.seed(SEED)
np.random.seed(SEED)

print(f"Models: {list(MODELS.keys())}")
print(f"Datasets: {list(DS_SPECS.keys())}")
print(f"Conditions: bare, random, repeat_token, comprehend, extract, summarize, oracle, doc_keywords")
print(f"N_SAMPLES={N_SAMPLES}, N_HARD={N_HARD}, PREFIX_L={PREFIX_L}")
""")


code(r"""# Load datasets
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
            samples.append({'passage': passage, 'query': query, 'answer': answer,
                           'passage_words': count_words(passage)})
    pyrandom.seed(SEED)
    pyrandom.shuffle(samples)
    return samples[:N_SAMPLES]

print("Loading datasets...")
all_samples = {}
for ds_key in DS_SPECS:
    all_samples[ds_key] = load_qa_dataset(ds_key)
    print(f"  {ds_key}: {len(all_samples[ds_key])} samples")
""")


code(r"""# Scoring functions (same as Exp 01 with use_cache=True fix)
_model = None
_tokenizer = None
_device = None
_layer_inv_freqs = None
_layer_types = None
_sliding_limit = None
_bos_id = None
_nl_ids = None

def _encode_phase_a(doc_ids, prefix_ids=None):
    has_bos = _bos_id is not None
    bos_offset = 1 if has_bos else 0
    input_ids = []
    if has_bos:
        input_ids.append(_bos_id)
    if prefix_ids is not None:
        input_ids += list(prefix_ids) + _nl_ids
    input_ids += list(doc_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    outputs = _model(input_ids=input_tensor, use_cache=True)
    cache = outputs.past_key_values

    if prefix_ids is not None:
        P = len(prefix_ids)
        NL = len(_nl_ids)
        doc_start = bos_offset + P + NL
    else:
        doc_start = bos_offset
    D = len(doc_ids)

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
    answer_logits = logits[answer_start - 1 : answer_start - 1 + len(answer_ids)]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=_device)
    loss = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction='mean')
    return loss.item()

print("Scoring functions defined.")
""")


code(r"""# Main loop: all models x all conditions
CONDITION_NAMES = ['random', 'repeat_token', 'comprehend', 'extract', 'summarize', 'oracle', 'doc_keywords']
all_summaries = {}

for model_key, model_spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {model_key} ({model_spec['name']})")
    print(f"{'#'*70}")

    model_dir = RESULTS_BASE / model_key
    model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f'sweep_{model_key}'

    # Load model
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
    _bos_id = _tokenizer.bos_token_id
    _nl_ids = _tokenizer.encode("\n", add_special_tokens=False)
    info = get_model_info(_model)

    if _sliding_limit is not None:
        max_doc = _sliding_limit - 1 - PREFIX_L - len(_nl_ids)
    else:
        max_doc = 765

    print(f"  Loaded: {info['num_layers']} layers, head_dim={info['head_dim']}, "
          f"BOS={_bos_id}, NL={_nl_ids}, max_doc={max_doc}")

    # Build instruction prefixes
    instruction_prefixes = {}
    for iname, itext in INSTRUCTIONS.items():
        ids = _tokenizer.encode(itext, add_special_tokens=False)
        instruction_prefixes[iname] = make_prefix(ids, PREFIX_L)

    # Random prefix
    rng = pyrandom.Random(SEED)
    random_prefix = [rng.randint(100, _tokenizer.vocab_size - 1) for _ in range(PREFIX_L)]

    # Repeat token ("the")
    the_id = _tokenizer.encode("the", add_special_tokens=False)[0]
    repeat_prefix = [the_id] * PREFIX_L

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
                'original_idx': idx,
            }

            with torch.no_grad():
                # Bare
                cache_bare, D = _encode_phase_a(doc_ids)
                result['nll_bare'] = _score_phase_b(cache_bare, D, query_ids, answer_ids)
                del cache_bare

                # Random
                cache, D = _encode_phase_a(doc_ids, prefix_ids=random_prefix)
                result['nll_random'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Repeat token
                cache, D = _encode_phase_a(doc_ids, prefix_ids=repeat_prefix)
                result['nll_repeat_token'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Instruction prefixes
                for iname in INSTRUCTIONS:
                    cache, D = _encode_phase_a(doc_ids, prefix_ids=instruction_prefixes[iname])
                    result[f'nll_{iname}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                    del cache

                # Oracle (query as prefix)
                oracle_ids = make_prefix(query_ids, PREFIX_L)
                cache, D = _encode_phase_a(doc_ids, prefix_ids=oracle_ids)
                result['nll_oracle'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Doc keywords (first 64 tokens of doc)
                doc_kw = doc_ids[:PREFIX_L]
                if len(doc_kw) < PREFIX_L:
                    doc_kw = make_prefix(doc_kw, PREFIX_L)
                cache, D = _encode_phase_a(doc_ids, prefix_ids=doc_kw)
                result['nll_doc_keywords'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

            scored.append(result)

            if (idx + 1) % 20 == 0:
                ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
                d_best = max(
                    cohens_d(np.array([x['nll_bare'] - x[f'nll_{c}'] for x in scored]))
                    for c in CONDITION_NAMES
                )
                print(f"    [{idx+1}/{len(samples)}] best_d={d_best:+.3f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final checkpoint
        ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))

        # Hard subset
        scored.sort(key=lambda x: x['nll_bare'], reverse=True)
        hard = scored[:N_HARD]

        # Print per-condition results
        for c in CONDITION_NAMES:
            diffs = np.array([x['nll_bare'] - x[f'nll_{c}'] for x in hard])
            d = cohens_d(diffs)
            w = win_rate(diffs)
            sig = '***' if paired_ttest(diffs)[1] < 0.001 else ('**' if paired_ttest(diffs)[1] < 0.01 else ('*' if paired_ttest(diffs)[1] < 0.05 else ''))
            print(f"    {c:<15} d={d:+.3f}  win={w:.0%}  {sig}")

    # Build summary
    summary = {'model': model_key, 'model_name': model_spec['name'], 'rankings': []}
    for c in CONDITION_NAMES:
        all_diffs = []
        per_ds = {}
        for ds_key in DS_SPECS:
            ckpt = json.loads((model_dir / f"checkpoint_{ds_key}.json").read_text())
            samples_all = ckpt['samples']
            samples_all.sort(key=lambda x: x['nll_bare'], reverse=True)
            hard = samples_all[:N_HARD]
            diffs = np.array([x['nll_bare'] - x[f'nll_{c}'] for x in hard])
            all_diffs.extend(diffs.tolist())
            d = cohens_d(diffs)
            w = win_rate(diffs)
            _, p = paired_ttest(diffs)
            per_ds[ds_key] = {'d': round(d, 4), 'win': round(w, 4), 'p': p}
        pooled = np.array(all_diffs)
        summary['rankings'].append({
            'condition': c,
            'pooled_d': round(cohens_d(pooled), 4),
            'pooled_win': round(win_rate(pooled), 4),
            'per_dataset': per_ds,
        })
    summary['rankings'].sort(key=lambda r: r['pooled_d'], reverse=True)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    all_summaries[model_key] = summary
    print(f"\n  Summary saved.")

    del _model, _tokenizer
    _model = None
    _tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Model unloaded.")

print(f"\n{'='*70}")
print("ALL MODELS COMPLETE")
""")


code(r"""# Cross-model comparison
DS_LABELS = {'ms_marco': 'MARCO', 'squad_v2': 'SQuAD', 'drop': 'DROP', 'gsm8k': 'GSM8K'}

for model_key, summary in all_summaries.items():
    print(f"\n{'='*50}")
    print(f"{model_key} ({summary['model_name']})")
    print(f"{'='*50}")
    print(f"{'Condition':<16} {'Pooled d':>9} {'Win':>6}", end='')
    for ds in DS_SPECS:
        print(f" {DS_LABELS[ds]:>7}", end='')
    print()
    print("-" * 70)
    for r in summary['rankings']:
        print(f"{r['condition']:<16} {r['pooled_d']:>+9.3f} {r['pooled_win']:>6.0%}", end='')
        for ds in DS_SPECS:
            d = r['per_dataset'][ds]['d']
            print(f" {d:>+7.2f}", end='')
        print()

# Find best condition per model
print(f"\n{'='*50}")
print("BEST CONDITION PER MODEL")
print(f"{'='*50}")
for model_key, summary in all_summaries.items():
    best = summary['rankings'][0]
    n_positive = sum(1 for r in summary['rankings'] if r['pooled_d'] > 0)
    print(f"  {model_key}: best={best['condition']} (d={best['pooled_d']:+.3f}), "
          f"{n_positive}/{len(summary['rankings'])} conditions positive")

# Save combined
combined = {k: v for k, v in all_summaries.items()}
(RESULTS_BASE / "combined_summary.json").write_text(json.dumps(combined, indent=2, default=str))
print(f"\nCombined summary saved to {RESULTS_BASE / 'combined_summary.json'}")
""")


out_path = "experiments/01_multi_model/01_condition_sweep.ipynb"
nbf.write(nb, out_path)
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
