#!/usr/bin/env python3
"""Build the ablation sweep: instruction variants, AI prefixes, oracle fix.

New conditions tested (per sample):
  Fixed instructions (L=64):
    nll_summarize_64          "Summarize the following text."
    nll_question_64           "What are the key facts in this text?"
    nll_index_64              "Index the following information for retrieval."
    nll_declarative_64        "This text contains important information."
  Minimal cues (L=1):
    nll_extract_word_1        single token: "Extract"
    nll_comprehend_word_1     single token: "Comprehend"
    nll_facts_word_1          single token: "Facts"
  Document-derived:
    nll_summary_ai            Gemini-generated one-sentence summary
    nll_instruction_ai        Gemini-generated customized reading instruction
    nll_keywords_tfidf        TF-IDF top-10 keywords
  Oracle fix:
    nll_oracle_natural        query at its natural length (no padding/truncation)
  Baselines (carried from main sweep checkpoint):
    nll_bare                  (read from checkpoint, not re-scored)
    nll_comprehend_64         (read from checkpoint, not re-scored)
    nll_extract_64            (read from checkpoint, not re-scored)

Models: all 16 from the main sweep.
Datasets: same 6, same 400 samples, same seeds.
Results: results/exp03_ablation/{model_key}/checkpoint_{ds}.json

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/02_ablation/build_ablation_sweep.py
    cd experiments/02_ablation
    papermill 02_ablation_sweep.ipynb 02_ablation_sweep_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/02_ablation", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
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


md(r"""# Ablation Sweep

Instruction variants, AI-generated prefixes, TF-IDF keywords, natural-length oracle.
Tests 11 new conditions across 16 models x 6 datasets x 400 samples.""")


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

MODELS = {
    'qwen25_0_5b': {'name': 'Qwen/Qwen2.5-0.5B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'gemma3_1b': {'name': 'google/gemma-3-1b-it', 'loader': 'Gemma3ForCausalLM'},
    'qwen25_1_5b': {'name': 'Qwen/Qwen2.5-1.5B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'qwen25_3b': {'name': 'Qwen/Qwen2.5-3B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'gemma3n_e4b': {'name': 'google/gemma-3n-e4b-it', 'loader': 'Gemma3nForConditionalGeneration'},
    'gemma3_4b': {'name': 'google/gemma-3-4b-it', 'loader': 'Gemma3ForConditionalGeneration'},
    'gemma3_4b_base': {'name': 'google/gemma-3-4b-pt', 'loader': 'Gemma3ForConditionalGeneration'},
    'qwen25_7b': {'name': 'Qwen/Qwen2.5-7B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'qwen25_7b_base': {'name': 'Qwen/Qwen2.5-7B', 'loader': 'AutoModelForCausalLM'},
    'deepseek_r1_qwen_7b': {'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'loader': 'AutoModelForCausalLM'},
    'mistral_7b': {'name': 'mistralai/Mistral-7B-Instruct-v0.3', 'loader': 'AutoModelForCausalLM'},
    'ministral_8b': {'name': 'mistralai/Ministral-8B-Instruct-2410', 'loader': 'AutoModelForCausalLM'},
    'gemma3_12b': {'name': 'google/gemma-3-12b-it', 'loader': 'Gemma3ForConditionalGeneration'},
    'qwen25_14b': {'name': 'Qwen/Qwen2.5-14B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'gemma3_27b': {'name': 'google/gemma-3-27b-it', 'loader': 'Gemma3ForConditionalGeneration'},
    'qwen25_32b': {'name': 'Qwen/Qwen2.5-32B-Instruct', 'loader': 'AutoModelForCausalLM'},
}

DATASET_TIERS = {
    'gsm8k': 'high_reasoning', 'drop': 'high_reasoning',
    'squad_v2': 'mid_reasoning', 'hotpotqa': 'mid_reasoning',
    'triviaqa': 'factoid', 'ms_marco': 'factoid',
}

# Fixed instructions for ablation
INSTRUCTIONS = {
    'comprehend': "Read and comprehend this text carefully.",
    'extract': "Extract the key facts from this text.",
    'summarize': "Summarize the following text.",
    'question': "What are the key facts in this text?",
    'index': "Index the following information for retrieval.",
    'declarative': "This text contains important information.",
}

RESULTS_BASE = Path("../../results/exp03_ablation")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

# Load pre-generated AI prefixes
PREFIXES_PATH = Path("generated_prefixes.json")
assert PREFIXES_PATH.exists(), f"Run generate_prefixes.py first: {PREFIXES_PATH}"
generated = json.loads(PREFIXES_PATH.read_text())
AI_SUMMARIES = generated['ai_summaries']
AI_INSTRUCTIONS = generated['ai_instructions']
TFIDF_KEYWORDS = generated['tfidf_keywords']
print(f"Loaded AI prefixes: {generated['metadata']}")

print(f"\nModels: {len(MODELS)}")
print(f"Datasets: {list(DATASET_TIERS.keys())} ({len(DATASET_TIERS)} total)")
""")


# =====================================================================
# Cell 2: Load datasets (same as main sweep)
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

# --- TriviaQA ---
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

# --- HotpotQA ---
print("  hotpotqa...", end="")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
candidates = []
for item in ds:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
    passage_parts = []
    for title, sid in zip(sf.get('title', []), sf.get('sent_id', [])):
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

# --- GSM8K ---
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
""")


# =====================================================================
# Cell 3: Scoring functions
# =====================================================================
code(r"""# Scoring functions
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


# =====================================================================
# Cell 4: Main loop
# =====================================================================
code(r"""# Main loop: all models x new conditions x all datasets

all_summaries = {}

for model_key, model_spec in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {model_key} ({model_spec['name']})")
    print(f"{'#'*70}")

    model_dir = RESULTS_BASE / model_key
    model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f'ablation_sweep_{model_key}'

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
    print(f"  Loaded: {info['num_layers']} layers, head_dim={info['head_dim']}, max_doc={max_doc}")

    # Build fixed instruction prefixes
    prefixes = {}
    for iname, itext in INSTRUCTIONS.items():
        ids = _tokenizer.encode(itext, add_special_tokens=False)
        prefixes[f'{iname}_64'] = make_prefix(ids, 64)

    # Single-word prefixes
    for word in ['Extract', 'Comprehend', 'Facts']:
        word_id = _tokenizer.encode(word, add_special_tokens=False)
        prefixes[f'{word.lower()}_word_1'] = word_id[:1]

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
                # Bare baseline
                cache, D = _encode_phase_a(doc_ids)
                result['nll_bare'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Fixed instruction prefixes (L=64)
                for cname in ['comprehend_64', 'extract_64', 'summarize_64',
                              'question_64', 'index_64', 'declarative_64']:
                    cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes[cname])
                    result[f'nll_{cname}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                    del cache

                # Single-word prefixes (L=1)
                for cname in ['extract_word_1', 'comprehend_word_1', 'facts_word_1']:
                    cache, D = _encode_phase_a(doc_ids, prefix_ids=prefixes[cname])
                    result[f'nll_{cname}'] = _score_phase_b(cache, D, query_ids, answer_ids)
                    del cache

                # Oracle at natural length (no padding)
                oracle_ids = query_ids  # use query at whatever length it naturally is
                if _sliding_limit is not None:
                    max_oracle = _sliding_limit - 1 - D - len(_nl_ids)
                    oracle_ids = oracle_ids[:max_oracle]
                cache, D = _encode_phase_a(doc_ids, prefix_ids=oracle_ids)
                result['nll_oracle_natural'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # Oracle padded to 64 (same as main sweep for comparison)
                oracle_prefix_64 = make_prefix(query_ids, 64)
                cache, D = _encode_phase_a(doc_ids, prefix_ids=oracle_prefix_64)
                result['nll_oracle_64'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # AI-generated summary prefix
                summary_text = AI_SUMMARIES[ds_key][idx]
                summary_ids = _tokenizer.encode(summary_text, add_special_tokens=False)[:64]
                cache, D = _encode_phase_a(doc_ids, prefix_ids=summary_ids)
                result['nll_summary_ai'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # AI-generated instruction prefix
                instr_text = AI_INSTRUCTIONS[ds_key][idx]
                instr_ids = _tokenizer.encode(instr_text, add_special_tokens=False)[:64]
                cache, D = _encode_phase_a(doc_ids, prefix_ids=instr_ids)
                result['nll_instruction_ai'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

                # TF-IDF keywords prefix
                kw_text = TFIDF_KEYWORDS[ds_key][idx]
                kw_ids = _tokenizer.encode(kw_text, add_special_tokens=False)[:64]
                cache, D = _encode_phase_a(doc_ids, prefix_ids=kw_ids)
                result['nll_keywords_tfidf'] = _score_phase_b(cache, D, query_ids, answer_ids)
                del cache

            scored.append(result)

            if (idx + 1) % 20 == 0:
                ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
                d_ext = cohens_d(np.array([x['nll_bare'] - x['nll_extract_64'] for x in scored]))
                d_sum = cohens_d(np.array([x['nll_bare'] - x['nll_summary_ai'] for x in scored]))
                print(f"    [{idx+1}/{len(samples)}] ext d={d_ext:+.3f}, summary_ai d={d_sum:+.3f}")

            torch.cuda.empty_cache()

        # Final checkpoint
        ckpt_path.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
        print(f"  {ds_key}: {len(scored)} samples scored")

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
code(r"""# Quick analysis
DS_LABELS = {'ms_marco': 'MARCO', 'squad_v2': 'SQuAD', 'triviaqa': 'Trivia',
             'hotpotqa': 'Hotpot', 'drop': 'DROP', 'gsm8k': 'GSM8K'}

CONDITIONS = [
    'comprehend_64', 'extract_64', 'summarize_64', 'question_64',
    'index_64', 'declarative_64',
    'extract_word_1', 'comprehend_word_1', 'facts_word_1',
    'oracle_natural', 'oracle_64',
    'summary_ai', 'instruction_ai', 'keywords_tfidf',
]

print(f"\n{'Model':<22s}", end="")
for c in CONDITIONS:
    short = c.replace('_64', '').replace('_word_1', '/1')[:8]
    print(f" | {short:>8s}", end="")
print()
print("-" * (22 + 11 * len(CONDITIONS)))

for model_key in MODELS:
    model_dir = RESULTS_BASE / model_key
    all_diffs = {c: [] for c in CONDITIONS}
    for ds_key in DATASET_TIERS:
        ckpt = model_dir / f"checkpoint_{ds_key}.json"
        if not ckpt.exists(): continue
        samples = json.loads(ckpt.read_text())['samples']
        samples.sort(key=lambda x: x.get('nll_bare', 0), reverse=True)
        hard = samples[:N_HARD]
        for s in hard:
            b = s.get('nll_bare', 0)
            for c in CONDITIONS:
                nk = f'nll_{c}'
                if nk in s:
                    all_diffs[c].append(b - s[nk])
    row = f"{model_key:<22s}"
    for c in CONDITIONS:
        arr = np.array(all_diffs[c])
        d = cohens_d(arr) if len(arr) > 1 else 0
        row += f" | {d:>+8.2f}"
    print(row)
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/02_ablation/02_ablation_sweep.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
