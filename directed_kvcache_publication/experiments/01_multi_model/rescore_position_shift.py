#!/usr/bin/env python3
"""Re-score only the position_shift condition across all models/datasets.

Uses the fixed roundtrip reposition (encode at natural positions, shift forward
then back) instead of the old confounded version (encode at shifted positions).

This updates nll_position_shift in each checkpoint without re-running the other
17 conditions. ~1/18th the cost of a full re-run.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/01_multi_model/rescore_position_shift.py
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import json
import gc
import shutil
import time
import random as pyrandom
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv

from lib.rope import reposition_kv_cache
from lib.cache import deep_copy_cache
from lib.quantization import norm_roundtrip_kv_cache
from lib.data import count_words
from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
SEED = 42
N_SAMPLES = 400

RESULTS_BASE = Path(__file__).resolve().parent.parent.parent / "results" / "exp02_model_expansion"

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

DS_NAMES = ['gsm8k', 'drop', 'squad_v2', 'hotpotqa', 'triviaqa', 'ms_marco']
SHIFT = 64


def load_all_datasets():
    """Load all 6 datasets with the same seed/shuffle as the sweep notebook."""
    all_samples = {}

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
                candidates.append({'passage': passage, 'query': query, 'answer': answer, 'passage_words': wc})
    pyrandom.seed(SEED + 100); pyrandom.shuffle(candidates)
    all_samples['ms_marco'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    candidates = []
    for item in ds:
        passage = item.get('context', ''); query = item.get('question', '')
        answers = item.get('answers', {}).get('text', [])
        answer = answers[0] if answers else ''
        if passage and query and answer:
            wc = count_words(passage)
            if 30 <= wc <= 500:
                candidates.append({'passage': passage, 'query': query, 'answer': answer, 'passage_words': wc})
    pyrandom.seed(SEED + 200); pyrandom.shuffle(candidates)
    all_samples['squad_v2'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
    candidates = []
    for item in ds:
        entity_pages = item.get('entity_pages', {})
        wiki_contexts = entity_pages.get('wiki_context', [])
        if not wiki_contexts or not wiki_contexts[0]: continue
        passage = ' '.join(wiki_contexts[0].split()[:500])
        query = item['question']; answer_val = item['answer']['value']
        aliases = item['answer'].get('aliases', [])
        passage_lower = passage.lower()
        found = answer_val.lower() in passage_lower
        if not found:
            for alias in aliases:
                if alias.lower() in passage_lower: found = True; break
        if not found: continue
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer_val) >= 1:
            candidates.append({'passage': passage, 'query': query, 'answer': answer_val, 'passage_words': wc})
    pyrandom.seed(SEED + 300); pyrandom.shuffle(candidates)
    all_samples['triviaqa'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    candidates = []
    for item in ds:
        context = item.get('context', {}); sf = item.get('supporting_facts', {})
        title_to_sents = {t: s for t, s in zip(context.get('title', []), context.get('sentences', []))}
        parts = [title_to_sents[t][sid] for t, sid in zip(sf.get('title', []), sf.get('sent_id', []))
                 if t in title_to_sents and sid < len(title_to_sents[t])]
        if not parts: continue
        passage = ' '.join(parts); query = item['question']; answer = item['answer']
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer) >= 1:
            candidates.append({'passage': passage, 'query': query, 'answer': answer, 'passage_words': wc})
    pyrandom.seed(SEED + 400); pyrandom.shuffle(candidates)
    all_samples['hotpotqa'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    ds = load_dataset("ucinlp/drop", split="validation")
    candidates = []
    for item in ds:
        spans = item.get('answers_spans', {}).get('spans', [])
        if not spans or not spans[0]: continue
        passage = item['passage']; answer = spans[0]
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer) >= 1:
            candidates.append({'passage': passage, 'query': item['question'], 'answer': answer, 'passage_words': wc})
    pyrandom.seed(SEED + 500); pyrandom.shuffle(candidates)
    all_samples['drop'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    candidates = []
    for item in ds:
        if '####' not in item['answer']: continue
        answer = item['answer'].split('####')[-1].strip()
        if not answer: continue
        passage = item['question']; wc = count_words(passage)
        if 10 <= wc <= 500:
            candidates.append({'passage': passage, 'query': 'What is the answer?', 'answer': answer, 'passage_words': wc})
    pyrandom.seed(SEED + 600); pyrandom.shuffle(candidates)
    all_samples['gsm8k'] = candidates[:N_SAMPLES]
    del ds, candidates; gc.collect()

    return all_samples


def load_model(model_name, loader_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if loader_name == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader_name == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader_name == "Gemma3nForConditionalGeneration":
        from transformers import Gemma3nForConditionalGeneration
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return model, tokenizer


def purge_hf_cache(model_name):
    slug = "models--" + model_name.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, slug)
    if os.path.isdir(cache_path):
        size_gb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fns in os.walk(cache_path) for f in fns) / 1e9
        shutil.rmtree(cache_path)
        print(f"  Purged cache ({size_gb:.1f} GB)")


def rescore_position_shift(model, tokenizer, device, inv_freqs, layer_types,
                           sliding_limit, bos_id, nl_ids, ckpt_samples, ds_samples):
    """Re-score position_shift for all samples using roundtrip reposition."""
    if sliding_limit is not None:
        max_doc = sliding_limit - 1 - SHIFT - len(nl_ids)
    else:
        max_doc = 765

    for idx, s in enumerate(ckpt_samples):
        # Use the full passage from the dataset (checkpoint only has truncated query/answer)
        raw = ds_samples[idx]
        doc_ids = tokenizer.encode(raw['passage'], add_special_tokens=False)[:max_doc]
        query_ids = tokenizer.encode(raw['query'], add_special_tokens=False)
        answer_ids = tokenizer.encode(raw['answer'], add_special_tokens=False)
        if not answer_ids:
            continue
        D = len(doc_ids)

        with torch.no_grad():
            # Encode at natural positions (identical to bare)
            input_ids = [bos_id] + doc_ids
            outputs = model(input_ids=torch.tensor([input_ids], device=device), use_cache=True)
            cache = outputs.past_key_values
            del outputs

            # Roundtrip reposition: forward then back
            natural_pos = torch.arange(1, 1 + D, device=device)
            shifted_pos = torch.arange(SHIFT + 1, SHIFT + 1 + D, device=device)
            cache = reposition_kv_cache(cache, natural_pos, shifted_pos,
                                         inv_freqs, layer_types, bos_start=0)
            cache = reposition_kv_cache(cache, shifted_pos, natural_pos,
                                         inv_freqs, layer_types, bos_start=0)
            cache = norm_roundtrip_kv_cache(cache)

            # Score Phase B
            phase_b_ids = nl_ids + query_ids + nl_ids + answer_ids
            n_b = len(phase_b_ids)
            pos_b = torch.arange(D + 1, D + 1 + n_b, device=device).unsqueeze(0)
            cache_copy = deep_copy_cache(cache)
            pb_out = model(input_ids=torch.tensor([phase_b_ids], device=device),
                           position_ids=pos_b, past_key_values=cache_copy, use_cache=False)
            ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
            targets = torch.tensor(answer_ids, device=device)
            nll = torch.nn.functional.cross_entropy(
                pb_out.logits[0, ans_start-1:ans_start-1+len(answer_ids)], targets).item()
            del pb_out, cache

        s['nll_position_shift'] = nll
        torch.cuda.empty_cache()

        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{len(ckpt_samples)}]")

    return ckpt_samples


def main():
    print("=" * 70)
    print("RESCORE position_shift (roundtrip reposition control)")
    print("=" * 70)

    print("\nLoading datasets...")
    all_samples = load_all_datasets()
    for ds_key, samples in all_samples.items():
        print(f"  {ds_key}: {len(samples)} samples")

    for model_key, model_spec in MODELS.items():
        model_dir = RESULTS_BASE / model_key
        if not model_dir.exists():
            print(f"\n  {model_key}: no results directory, skipping")
            continue

        # Check if any checkpoints exist
        has_data = any((model_dir / f"checkpoint_{ds}.json").exists() for ds in DS_NAMES)
        if not has_data:
            print(f"\n  {model_key}: no checkpoints, skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"# {model_key} ({model_spec['name']})")
        print(f"{'#'*70}")

        t0 = time.time()
        model, tokenizer = load_model(model_spec['name'], model_spec['loader'])
        device = next(model.parameters()).device
        inv_freqs = build_layer_inv_freqs(model, device=device)
        layer_types = get_layer_types(model)
        sliding_limit = get_sliding_cache_limit(model)

        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.pad_token_id
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        print(f"  Loaded in {time.time()-t0:.0f}s")

        for ds in DS_NAMES:
            ckpt_path = model_dir / f"checkpoint_{ds}.json"
            if not ckpt_path.exists():
                continue

            ckpt = json.loads(ckpt_path.read_text())
            ckpt_samples = ckpt['samples']
            ds_samples = all_samples[ds]
            print(f"  {ds}: re-scoring {len(ckpt_samples)} samples...", end="", flush=True)

            t1 = time.time()
            ckpt_samples = rescore_position_shift(
                model, tokenizer, device, inv_freqs, layer_types,
                sliding_limit, bos_id, nl_ids, ckpt_samples, ds_samples)

            ckpt['samples'] = ckpt_samples
            ckpt_path.write_text(json.dumps(ckpt))
            print(f" done ({time.time()-t1:.0f}s)")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        purge_hf_cache(model_spec['name'])
        print(f"  Total: {time.time()-t0:.0f}s")

    print(f"\n{'='*70}")
    print("ALL MODELS RE-SCORED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
