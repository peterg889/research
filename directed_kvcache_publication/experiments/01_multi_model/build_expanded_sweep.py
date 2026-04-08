#!/usr/bin/env python3
"""Build the expanded model sweep: 12 models x 18 conditions x 6 datasets x 400 samples.

Extends the final sweep from 4 to 12 models across 5 architecture families.
Fixes linear RoPE scaling bug on Gemma 3 4B/12B/27B full_attention layers.
Ordered by VRAM: small models first for fast failure detection.

Includes:
- 6 datasets spanning 3 reasoning tiers (high, mid, factoid)
- 15 prefix conditions + position-shift control
- Normalization ablation (bare and comprehend, with/without norm)
- N=400 (160 hard) per dataset for strong statistical power
- All models use BOS (Qwen/DeepSeek get PAD), GSM8K number-only answer

Conditions (18 NLL values per sample):
  nll_bare                      bare, with norm
  nll_bare_nonorm               bare, without norm (normalization ablation)
  nll_random_64                 random tokens L=64
  nll_repeat_64                 "the" x 64
  nll_comprehend_64             coherent comprehend instruction L=64
  nll_comprehend_scrambled_64   scrambled comprehend tokens L=64
  nll_comprehend_64_nonorm      comprehend, without norm (normalization ablation)
  nll_extract_64                extract instruction L=64
  nll_extract_scrambled_64      scrambled extract tokens L=64
  nll_anti_64                   "Ignore this text completely" L=64
  nll_oracle_64                 actual query as prefix L=64
  nll_comprehend_16             comprehend L=16
  nll_comprehend_4              comprehend L=4
  nll_comprehend_1              comprehend L=1
  nll_random_16                 random L=16
  nll_random_4                  random L=4
  nll_random_1                  random L=1
  nll_position_shift            no tokens, just RoPE shift by 64

Datasets:
  GSM8K       high-reasoning   (number-only answer after ####)
  DROP        high-reasoning   (first answer span)
  SQuAD v2    mid-reasoning    (first answer text)
  HotpotQA    mid-reasoning    (supporting-facts passage reconstruction)
  TriviaQA    factoid          (wiki context, answer-in-passage filter)
  MS MARCO    factoid          (joined passages, filter "No Answer")

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/01_multi_model/build_expanded_sweep.py
    cd experiments/01_multi_model
    papermill 01_expanded_sweep.ipynb 01_expanded_sweep_executed.ipynb --no-progress-bar
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


md(r"""# Expanded Model Sweep

12 models x 18 conditions x 6 datasets x 400 samples.
Fixes linear RoPE scaling on Gemma 3 4B/12B/27B full_attention layers.
Ordered by VRAM (small→large) for fast failure detection.""")


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
import shutil
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
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def purge_hf_cache(model_name):
    slug = "models--" + model_name.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, slug)
    if os.path.isdir(cache_path):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(cache_path) for f in fns
        ) / 1e9
        shutil.rmtree(cache_path)
        print(f"  Purged cache for {model_name} ({size_gb:.1f} GB)")

SEED = 42
N_SAMPLES = 400
N_HARD = 160

# Ordered by estimated VRAM (small → large) for fast failure detection
MODELS = {
    # ~1.4 GB — Gemma 3 1B (text-only, Gemma3ForCausalLM, sliding_window=512)
    'gemma3_1b': {
        'name': 'google/gemma-3-1b-it',
        'loader': 'Gemma3ForCausalLM',
    },
    # ~2.1 GB — Qwen 2.5 1.5B (no BOS, needs PAD)
    'qwen25_1_5b': {
        'name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
    # ~4.2 GB — Qwen 2.5 3B
    'qwen25_3b': {
        'name': 'Qwen/Qwen2.5-3B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
    # ~4.6 GB — Gemma 3N E4B (has use_cache=False bug, sliding_window=512)
    'gemma3n_e4b': {
        'name': 'google/gemma-3n-e4b-it',
        'loader': 'Gemma3nForConditionalGeneration',
    },
    # ~6.7 GB — Gemma 3 4B (multimodal, linear RoPE on full_attn, sliding_window=1024)
    'gemma3_4b': {
        'name': 'google/gemma-3-4b-it',
        'loader': 'Gemma3ForConditionalGeneration',
    },
    # ~9.7 GB — Qwen 2.5 7B (no BOS, rope_theta=1M)
    'qwen25_7b': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
    # ~9.7 GB — DeepSeek R1 Distill Qwen 7B (Qwen2 arch, rope_theta=10K, no BOS)
    'deepseek_r1_qwen_7b': {
        'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'loader': 'AutoModelForCausalLM',
    },
    # ~13.2 GB — Mistral 7B v0.3 (full attention, rope_theta=1M)
    'mistral_7b': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'loader': 'AutoModelForCausalLM',
    },
    # ~15.6 GB — Ministral 8B (hybrid sliding+full, rope_theta=100M, sliding_window=32768)
    'ministral_8b': {
        'name': 'mistralai/Ministral-8B-Instruct-2410',
        'loader': 'AutoModelForCausalLM',
    },
    # ~19 GB — Gemma 3 12B (linear RoPE on full_attn, sliding_window=1024)
    'gemma3_12b': {
        'name': 'google/gemma-3-12b-it',
        'loader': 'Gemma3ForConditionalGeneration',
    },
    # ~31.8 GB — Qwen 2.5 14B
    'qwen25_14b': {
        'name': 'Qwen/Qwen2.5-14B-Instruct',
        'loader': 'AutoModelForCausalLM',
    },
    # ~45.8 GB — Gemma 3 27B (linear RoPE on full_attn, head_dim=128, needs A100-80GB)
    'gemma3_27b': {
        'name': 'google/gemma-3-27b-it',
        'loader': 'Gemma3ForConditionalGeneration',
    },
}

DATASET_TIERS = {
    'gsm8k': 'high_reasoning', 'drop': 'high_reasoning',
    'squad_v2': 'mid_reasoning', 'hotpotqa': 'mid_reasoning',
    'triviaqa': 'factoid', 'ms_marco': 'factoid',
}

INSTRUCTIONS = {
    'comprehend': "Read and comprehend this text carefully.",
    'extract': "Extract the key facts from this text.",
    'anti': "Ignore this text completely. Do not read it.",
}

RESULTS_BASE = Path("../../results/exp02_model_expansion")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

torch.manual_seed(SEED)
pyrandom.seed(SEED)
np.random.seed(SEED)

print(f"Models: {list(MODELS.keys())}")
print(f"Datasets: {list(DATASET_TIERS.keys())} ({len(DATASET_TIERS)} total)")
print(f"N_SAMPLES={N_SAMPLES}, N_HARD={N_HARD}")
print(f"Conditions: 18 NLL values per sample (15 prefix + position_shift + 2 no-norm)")
print(f"Estimated run order: small→large by VRAM (fast failure detection)")
""")


# =====================================================================
# Cell 2: Load all 6 datasets
# =====================================================================
code(r"""# Load all 6 datasets with v4-validated extraction patterns
print("Loading datasets...")
all_samples = {}

# --- MS MARCO ---
print("  ms_marco...", end="")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
candidates = []
for item in ds:
    passages = item.get('passages', {}).get('passage_text', [])
    passage = ' '.join(passages) if passages else ''
    query = item.get('query', '')
    answers = item.get('answers', [])
    answer = answers[0] if answers and answers[0] != 'No Answer Present.' else ''
    if passage and query and answer:
        wc = count_words(passage)
        if 30 <= wc <= 500:
            candidates.append({'passage': passage, 'query': query, 'answer': answer,
                              'passage_words': wc})
pyrandom.seed(SEED + 100)
pyrandom.shuffle(candidates)
all_samples['ms_marco'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['ms_marco'])} samples")
del ds, candidates; gc.collect()

# --- SQuAD v2 ---
print("  squad_v2...", end="")
ds = load_dataset("rajpurkar/squad_v2", split="validation")
candidates = []
for item in ds:
    passage = item.get('context', '')
    query = item.get('question', '')
    answers = item.get('answers', {}).get('text', [])
    answer = answers[0] if answers else ''
    if passage and query and answer:
        wc = count_words(passage)
        if 30 <= wc <= 500:
            candidates.append({'passage': passage, 'query': query, 'answer': answer,
                              'passage_words': wc})
pyrandom.seed(SEED + 200)
pyrandom.shuffle(candidates)
all_samples['squad_v2'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['squad_v2'])} samples")
del ds, candidates; gc.collect()

# --- TriviaQA (wiki context, answer-in-passage filter) ---
print("  triviaqa...", end="")
ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
candidates = []
for item in ds:
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
        candidates.append({'passage': passage, 'query': query, 'answer': answer_val,
                          'passage_words': wc})
pyrandom.seed(SEED + 300)
pyrandom.shuffle(candidates)
all_samples['triviaqa'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['triviaqa'])} samples")
del ds, candidates; gc.collect()

# --- HotpotQA (supporting-facts passage reconstruction) ---
print("  hotpotqa...", end="")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
candidates = []
for item in ds:
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
        candidates.append({'passage': passage, 'query': query, 'answer': answer,
                          'passage_words': wc})
pyrandom.seed(SEED + 400)
pyrandom.shuffle(candidates)
all_samples['hotpotqa'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['hotpotqa'])} samples")
del ds, candidates; gc.collect()

# --- DROP ---
print("  drop...", end="")
ds = load_dataset("ucinlp/drop", split="validation")
candidates = []
for item in ds:
    passage = item['passage']
    question = item['question']
    answers_spans = item.get('answers_spans', {})
    spans = answers_spans.get('spans', [])
    if not spans or not spans[0]:
        continue
    answer = spans[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        candidates.append({'passage': passage, 'query': question, 'answer': answer,
                          'passage_words': wc})
pyrandom.seed(SEED + 500)
pyrandom.shuffle(candidates)
all_samples['drop'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['drop'])} samples")
del ds, candidates; gc.collect()

# --- GSM8K (number-only answer after ####) ---
print("  gsm8k...", end="")
ds = load_dataset("openai/gsm8k", "main", split="test")
candidates = []
for item in ds:
    passage = item['question']
    raw_answer = item['answer']
    if '####' not in raw_answer:
        continue
    answer = raw_answer.split('####')[-1].strip()
    if not answer:
        continue
    query = "What is the answer?"
    wc = count_words(passage)
    if 10 <= wc <= 500:
        candidates.append({'passage': passage, 'query': query, 'answer': answer,
                          'passage_words': wc})
pyrandom.seed(SEED + 600)
pyrandom.shuffle(candidates)
all_samples['gsm8k'] = candidates[:N_SAMPLES]
print(f" {len(all_samples['gsm8k'])} samples")
del ds, candidates; gc.collect()

print(f"\nAll datasets loaded: {sum(len(v) for v in all_samples.values())} total samples")
for ds_key, tier in DATASET_TIERS.items():
    print(f"  {ds_key} ({tier}): {len(all_samples[ds_key])} samples, "
          f"sample answer: {repr(all_samples[ds_key][0]['answer'][:60])}")
""")


# =====================================================================
# Cell 3: Scoring functions with normalization toggle
# =====================================================================
code(r"""# Scoring functions with normalization ablation support
_model = None
_tokenizer = None
_device = None
_layer_inv_freqs = None
_layer_types = None
_sliding_limit = None
_bos_id = None
_nl_ids = None

def _encode_phase_a(doc_ids, prefix_ids=None, apply_norm=True):
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
    if apply_norm:
        cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def _encode_phase_a_position_shift(doc_ids, shift=64, apply_norm=True):
    input_ids = [_bos_id] + list(doc_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    D = len(doc_ids)
    pos = torch.cat([
        torch.tensor([0], device=_device),
        torch.arange(shift + 1, shift + 1 + D, device=_device)
    ]).unsqueeze(0)
    outputs = _model(input_ids=input_tensor, position_ids=pos, use_cache=True)
    cache = outputs.past_key_values
    old_pos = torch.arange(shift + 1, shift + 1 + D, device=_device)
    new_pos = torch.arange(1, 1 + D, device=_device)
    cache = reposition_kv_cache(cache, old_pos, new_pos,
                                 _layer_inv_freqs, _layer_types, bos_start=0)
    if apply_norm:
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

print("Scoring functions defined (with normalization toggle).")
""")


# =====================================================================
# Cell 4: Main model loop
# =====================================================================
code(r"""# Main loop: all models x all conditions x all datasets
CONDITION_NAMES = [
    'random_64', 'repeat_64',
    'comprehend_64', 'comprehend_scrambled_64', 'comprehend_64_nonorm',
    'extract_64', 'extract_scrambled_64',
    'anti_64', 'oracle_64',
    'comprehend_16', 'comprehend_4', 'comprehend_1',
    'random_16', 'random_4', 'random_1',
    'position_shift',
    'bare_nonorm',
]

all_summaries = {}

for model_key, model_spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {model_key} ({model_spec['name']})")
    print(f"{'#'*70}")

    model_dir = RESULTS_BASE / model_key
    model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f'expanded_sweep_{model_key}'

    global _model, _tokenizer, _device, _layer_inv_freqs, _layer_types
    global _sliding_limit, _bos_id, _nl_ids

    _tokenizer = AutoTokenizer.from_pretrained(model_spec['name'], token=HF_TOKEN)
    loader = model_spec.get('loader', 'AutoModelForCausalLM')
    if loader == 'Gemma3ForConditionalGeneration':
        from transformers import Gemma3ForConditionalGeneration
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            model_spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()
    elif loader == 'Gemma3ForCausalLM':
        from transformers import Gemma3ForCausalLM
        _model = Gemma3ForCausalLM.from_pretrained(
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
    # max_doc accounts for longest prefix (L=64) + NL + BOS
    if _sliding_limit is not None:
        max_doc = _sliding_limit - 1 - 64 - len(_nl_ids)
    else:
        max_doc = 765
    print(f"  Loaded: {info['num_layers']} layers, head_dim={info['head_dim']}, "
          f"BOS={_bos_id}, sliding={'yes (limit=' + str(_sliding_limit) + ')' if _sliding_limit else 'no'}, "
          f"max_doc={max_doc}")

    # Build prefix token sequences for this model's tokenizer
    prefixes = {}
    for iname, itext in INSTRUCTIONS.items():
        ids = _tokenizer.encode(itext, add_special_tokens=False)
        for L in [64, 16, 4, 1]:
            prefixes[f'{iname}_{L}'] = make_prefix(ids, L)
        prefixes[f'{iname}_scrambled_64'] = scramble_prefix(make_prefix(ids, 64), seed=SEED)

    rng = pyrandom.Random(SEED)
    for L in [64, 16, 4, 1]:
        prefixes[f'random_{L}'] = [rng.randint(100, _tokenizer.vocab_size - 1) for _ in range(L)]

    the_id = _tokenizer.encode("the", add_special_tokens=False)[0]
    prefixes['repeat_64'] = [the_id] * 64

    # Score all datasets
    for ds_key in DATASET_TIERS:
        print(f"\n  --- {ds_key} ({DATASET_TIERS[ds_key]}) ---")
        samples = all_samples[ds_key]
        ckpt_path = model_dir / f"checkpoint_{ds_key}.json"

        scored = []
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            if ckpt.get('scoring_key') == scoring_key:
                scored = ckpt['samples']
                print(f"  Resumed: {len(scored)}/{len(samples)} samples")

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
                # Bare with norm (standard baseline)
                cache, D = _encode_phase_a(doc_ids, apply_norm=True)
                result['nll_bare'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Bare WITHOUT norm (normalization ablation)
                cache, D = _encode_phase_a(doc_ids, apply_norm=False)
                result['nll_bare_nonorm'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Standard L=64 prefix conditions (with norm)
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

                # Comprehend WITHOUT norm (normalization ablation)
                cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes['comprehend_64'],
                                            apply_norm=False)
                result['nll_comprehend_64_nonorm'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Length variants
                for L in [16, 4, 1]:
                    for ptype in ['comprehend', 'random']:
                        cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes[f'{ptype}_{L}'])
                        result[f'nll_{ptype}_{L}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                        del cache

                # Position shift only
                cache, D = _encode_phase_a_position_shift(doc_ids, shift=64)
                result['nll_position_shift'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

            scored.append(result)

            if (idx + 1) % 20 == 0:
                ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
                d_comp = cohens_d(np.array([x['nll_bare'] - x['nll_comprehend_64'] for x in scored]))
                print(f"    [{idx+1}/{len(samples)}] comp d={d_comp:+.3f}")

            torch.cuda.empty_cache()

        # Final checkpoint
        ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
        print(f"  {ds_key}: {len(scored)} samples scored")

    # Build summary
    summary = {'model': model_key, 'model_name': model_spec['name'],
               'model_info': info, 'rankings': [], 'normalization': {}}
    for cname in CONDITION_NAMES:
        nll_key = f'nll_{cname}'
        all_diffs = []
        per_ds = {}
        for ds_key in DATASET_TIERS:
            ckpt = json.loads((model_dir / f"checkpoint_{ds_key}.json").read_text())
            samples_all = ckpt['samples']
            samples_all.sort(key=lambda x: x['nll_bare'], reverse=True)
            hard = samples_all[:N_HARD]
            diffs = np.array([x['nll_bare'] - x[nll_key] for x in hard
                             if nll_key in x])
            if len(diffs) == 0:
                per_ds[ds_key] = {'d': 0.0, 'win': 0.5, 'p': 1.0}
                continue
            all_diffs.extend(diffs.tolist())
            per_ds[ds_key] = {
                'd': round(cohens_d(diffs), 4),
                'win': round(win_rate(diffs), 4),
                'p': paired_ttest(diffs)[1],
            }
        pooled = np.array(all_diffs) if all_diffs else np.array([0.0])
        summary['rankings'].append({
            'condition': cname,
            'pooled_d': round(cohens_d(pooled), 4),
            'pooled_win': round(win_rate(pooled), 4),
            'per_dataset': per_ds,
        })

    summary['rankings'].sort(key=lambda r: r['pooled_d'], reverse=True)

    # Normalization ablation
    for ds_key in DATASET_TIERS:
        ckpt = json.loads((model_dir / f"checkpoint_{ds_key}.json").read_text())
        samples_all = ckpt['samples']
        samples_all.sort(key=lambda x: x['nll_bare'], reverse=True)
        hard = samples_all[:N_HARD]
        norm_bare = np.array([x['nll_bare_nonorm'] - x['nll_bare'] for x in hard
                             if 'nll_bare_nonorm' in x])
        norm_comp = np.array([x['nll_comprehend_64_nonorm'] - x['nll_comprehend_64'] for x in hard
                             if 'nll_comprehend_64_nonorm' in x and 'nll_comprehend_64' in x])
        summary['normalization'][ds_key] = {
            'norm_effect_bare_d': round(cohens_d(norm_bare), 4) if len(norm_bare) > 1 else 0,
            'norm_effect_comp_d': round(cohens_d(norm_comp), 4) if len(norm_comp) > 1 else 0,
        }

    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    all_summaries[model_key] = summary
    print(f"\n  Summary saved to {model_dir / 'summary.json'}")

    del _model, _tokenizer
    _model = None; _tokenizer = None
    gc.collect(); torch.cuda.empty_cache()
    purge_hf_cache(model_spec['name'])
    print(f"  Model unloaded.")

print(f"\n{'='*70}")
print("ALL MODELS COMPLETE")
""")


# =====================================================================
# Cell 5: Analysis
# =====================================================================
code(r"""# Cross-model analysis
DS_LABELS = {'ms_marco': 'MARCO', 'squad_v2': 'SQuAD', 'triviaqa': 'Trivia',
             'hotpotqa': 'Hotpot', 'drop': 'DROP', 'gsm8k': 'GSM8K'}
DECOMP_CONDS = ['random_64', 'repeat_64', 'comprehend_64',
                'comprehend_scrambled_64', 'extract_64',
                'extract_scrambled_64', 'anti_64', 'oracle_64',
                'position_shift']

for model_key, summary in all_summaries.items():
    print(f"\n{'='*70}")
    print(f"{model_key}")
    print(f"{'='*70}")
    r = {x['condition']: x for x in summary['rankings']}

    # Condition ranking
    print(f"{'Condition':<28} {'Pool d':>7} {'Win':>5}", end='')
    for ds in DATASET_TIERS:
        print(f" {DS_LABELS[ds]:>6}", end='')
    print()
    print("-" * 80)
    for rank in summary['rankings']:
        if rank['condition'] in ['bare_nonorm', 'comprehend_64_nonorm']:
            continue  # skip ablation rows from main ranking
        print(f"{rank['condition']:<28} {rank['pooled_d']:>+7.3f} {rank['pooled_win']:>5.0%}", end='')
        for ds in DATASET_TIERS:
            d = rank['per_dataset'].get(ds, {}).get('d', 0)
            print(f" {d:>+6.2f}", end='')
        print()

    # Four-level decomposition
    if all(c in r for c in ['position_shift', 'random_64', 'comprehend_scrambled_64', 'comprehend_64']):
        pos = r['position_shift']['pooled_d']
        tok = r['random_64']['pooled_d'] - pos
        vocab = r['comprehend_scrambled_64']['pooled_d'] - r['random_64']['pooled_d']
        order = r['comprehend_64']['pooled_d'] - r['comprehend_scrambled_64']['pooled_d']
        total = r['comprehend_64']['pooled_d']
        print(f"\n  Decomposition:")
        print(f"    Position shift:  {pos:+.3f}")
        print(f"    Token presence:  {tok:+.3f}")
        print(f"    Vocabulary:      {vocab:+.3f}")
        print(f"    Word order:      {order:+.3f}")
        print(f"    = Total:         {total:+.3f}")

    # Normalization ablation
    print(f"\n  Normalization effect (positive = norm helps):")
    for ds_key in DATASET_TIERS:
        na = summary['normalization'].get(ds_key, {})
        nb = na.get('norm_effect_bare_d', 0)
        nc = na.get('norm_effect_comp_d', 0)
        print(f"    {DS_LABELS[ds_key]:>6}: bare d={nb:+.3f}, comp d={nc:+.3f}")

# Save combined
combined = {k: v for k, v in all_summaries.items()}
(RESULTS_BASE / "combined_summary.json").write_text(json.dumps(combined, indent=2, default=str))
print(f"\nCombined summary saved to {RESULTS_BASE / 'combined_summary.json'}")

# Summary counts
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for model_key, summary in all_summaries.items():
    r = {x['condition']: x for x in summary['rankings']}
    n_positive = sum(1 for x in summary['rankings']
                     if x['condition'] not in ['bare_nonorm', 'comprehend_64_nonorm']
                     and x['pooled_d'] > 0)
    n_total = sum(1 for x in summary['rankings']
                  if x['condition'] not in ['bare_nonorm', 'comprehend_64_nonorm'])
    print(f"  {model_key}: {n_positive}/{n_total} conditions positive")
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/01_multi_model/01_expanded_sweep.ipynb"
nbf.write(nb, out_path)
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
