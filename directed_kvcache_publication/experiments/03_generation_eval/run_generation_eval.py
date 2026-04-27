#!/usr/bin/env python3
"""Generation-based evaluation + inference-time baseline comparison.

For each model × dataset × condition:
1. Build cache (bare or primed)
2. Generate answer via greedy decoding from [NL, query, NL]
3. Compute Exact Match against reference answer
4. Also score NLL for the inference-time baseline (keywords at query time)

Models: Qwen 7B IT, Qwen 1.5B, Gemma 12B, Qwen 14B
Datasets: SQuAD v2, TriviaQA, GSM8K
Conditions: bare, keywords_tfidf, extract_64
+ Inference-time baseline: bare cache + keywords prepended at query time

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/03_generation_eval/run_generation_eval.py
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json
import gc
import re
import shutil
import time
import string
import random as pyrandom
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv

from lib.rope import select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from lib.data import count_words
from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

SEED = 42
N_SAMPLES = 400
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "exp04_generation_eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o777)

# Load generated prefixes for TF-IDF keywords
PREFIXES_PATH = Path(__file__).resolve().parent.parent / "02_ablation" / "generated_prefixes.json"

MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_14b": {"name": "Qwen/Qwen2.5-14B-Instruct", "loader": "AutoModelForCausalLM"},
}

EVAL_DATASETS = ["squad_v2", "triviaqa", "gsm8k"]
MAX_NEW_TOKENS = 50
EXTRACT_INSTRUCTION = "Extract the key facts from this text."


# ── Exact Match / F1 utilities ────────────────────────────────────────

def normalize_answer(s):
    """Lower text and remove punctuation, articles, extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_first_answer(text):
    """Extract just the first line/sentence of generated text, stripping continuations."""
    # Stop at common continuation patterns
    for stop in ["\n", "Human:", "Question:", "Answer:", "Note:", "The answer"]:
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def exact_match(prediction, ground_truth):
    pred = normalize_answer(extract_first_answer(prediction))
    gt = normalize_answer(ground_truth)
    return pred == gt


def contains_match(prediction, ground_truth):
    """Check if the ground truth appears in the first part of the prediction."""
    pred = normalize_answer(extract_first_answer(prediction))
    gt = normalize_answer(ground_truth)
    return gt in pred


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(extract_first_answer(prediction)).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def gsm8k_exact_match(prediction, ground_truth):
    """Extract the first number from prediction and compare."""
    first_part = extract_first_answer(prediction)
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", first_part.replace(",", ""))
    if not numbers:
        return False
    pred_num = numbers[0]
    gt_num = ground_truth.strip().replace(",", "")
    return pred_num == gt_num


# ── Dataset loading ───────────────────────────────────────────────────

def load_eval_datasets():
    all_samples = {}

    print("  squad_v2...", end="", flush=True)
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    candidates = []
    for item in ds:
        passage = item.get("context", ""); query = item.get("question", "")
        answers = item.get("answers", {}).get("text", [])
        answer = answers[0] if answers else ""
        if passage and query and answer:
            wc = count_words(passage)
            if 30 <= wc <= 500:
                candidates.append({"passage": passage, "query": query, "answer": answer,
                                   "passage_words": wc})
    pyrandom.seed(SEED + 200); pyrandom.shuffle(candidates)
    all_samples["squad_v2"] = candidates[:N_SAMPLES]
    print(f" {len(all_samples['squad_v2'])}")
    del ds, candidates; gc.collect()

    print("  triviaqa...", end="", flush=True)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
    candidates = []
    for item in ds:
        entity_pages = item.get("entity_pages", {})
        wiki_contexts = entity_pages.get("wiki_context", [])
        if not wiki_contexts or not wiki_contexts[0]: continue
        passage = " ".join(wiki_contexts[0].split()[:500])
        query = item["question"]; answer_val = item["answer"]["value"]
        aliases = item["answer"].get("aliases", [])
        passage_lower = passage.lower()
        found = answer_val.lower() in passage_lower
        if not found:
            for alias in aliases:
                if alias.lower() in passage_lower: found = True; break
        if not found: continue
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer_val) >= 1:
            candidates.append({"passage": passage, "query": query, "answer": answer_val,
                               "passage_words": wc, "aliases": aliases})
    pyrandom.seed(SEED + 300); pyrandom.shuffle(candidates)
    all_samples["triviaqa"] = candidates[:N_SAMPLES]
    print(f" {len(all_samples['triviaqa'])}")
    del ds, candidates; gc.collect()

    print("  gsm8k...", end="", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    candidates = []
    for item in ds:
        if "####" not in item["answer"]: continue
        answer = item["answer"].split("####")[-1].strip()
        if not answer: continue
        passage = item["question"]; wc = count_words(passage)
        if 10 <= wc <= 500:
            candidates.append({"passage": passage, "query": "What is the answer?",
                               "answer": answer, "passage_words": wc})
    pyrandom.seed(SEED + 600); pyrandom.shuffle(candidates)
    all_samples["gsm8k"] = candidates[:N_SAMPLES]
    print(f" {len(all_samples['gsm8k'])}")
    del ds, candidates; gc.collect()

    return all_samples


# ── Model loading ─────────────────────────────────────────────────────

def load_model(model_name, loader_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if loader_name == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
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


# ── Core evaluation ───────────────────────────────────────────────────

def build_primed_cache(model, tokenizer, device, doc_ids, prefix_ids,
                       inv_freqs, layer_types, sliding_limit, bos_id, nl_ids):
    """Build a primed cache: encode [BOS, prefix, NL, doc], select BOS+doc, reposition."""
    input_ids = [bos_id]
    if prefix_ids is not None:
        input_ids += list(prefix_ids) + nl_ids
    input_ids += list(doc_ids)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([input_ids], device=device), use_cache=True)
    cache = out.past_key_values
    del out

    if prefix_ids is not None:
        P = len(prefix_ids)
        NL = len(nl_ids)
        doc_start = 1 + P + NL
    else:
        doc_start = 1
    D = len(doc_ids)
    keep = [0] + list(range(doc_start, doc_start + D))
    cache = select_kv_cache(cache, keep, device=device)

    if prefix_ids is not None:
        old_pos = torch.arange(doc_start, doc_start + D, device=device)
        new_pos = torch.arange(1, 1 + D, device=device)
        cache = reposition_kv_cache(cache, old_pos, new_pos, inv_freqs, layer_types, bos_start=0)

    cache = norm_roundtrip_kv_cache(cache)
    return cache, D


def generate_answer(model, tokenizer, device, cache, D, query_ids, nl_ids, max_new_tokens):
    """Generate answer text from a primed cache using greedy decoding."""
    prompt_ids = nl_ids + query_ids + nl_ids
    n_prompt = len(prompt_ids)
    pos_start = D + 1

    cache_copy = deep_copy_cache(cache)
    input_ids = torch.tensor([prompt_ids], device=device)
    position_ids = torch.arange(pos_start, pos_start + n_prompt, device=device).unsqueeze(0)

    # Process prompt
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids,
                    past_key_values=cache_copy, use_cache=True)
    cache_gen = out.past_key_values
    next_token = out.logits[0, -1:].argmax(dim=-1)
    generated = [next_token.item()]

    # Autoregressive generation
    for _ in range(max_new_tokens - 1):
        pos = pos_start + n_prompt + len(generated) - 1
        with torch.no_grad():
            out = model(input_ids=next_token.unsqueeze(0),
                        position_ids=torch.tensor([[pos]], device=device),
                        past_key_values=cache_gen, use_cache=True)
        cache_gen = out.past_key_values
        next_token = out.logits[0, -1:].argmax(dim=-1)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated.append(tok_id)

    del cache_gen
    return tokenizer.decode(generated, skip_special_tokens=True)


def score_nll_inference_baseline(model, tokenizer, device, cache, D,
                                  query_ids, answer_ids, keyword_ids, nl_ids):
    """Score NLL with keywords prepended at query time (inference-time baseline).

    Uses a bare cache but prepends keywords before the query in Phase B:
    [NL, keywords, NL, query, NL, answer]
    """
    phase_b_ids = nl_ids + keyword_ids + nl_ids + query_ids + nl_ids + answer_ids
    n_tokens = len(phase_b_ids)
    pos = torch.arange(D + 1, D + 1 + n_tokens, device=device).unsqueeze(0)
    cache_copy = deep_copy_cache(cache)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([phase_b_ids], device=device),
                    position_ids=pos, past_key_values=cache_copy, use_cache=False)

    # NLL on answer tokens only
    ans_start = len(nl_ids) + len(keyword_ids) + len(nl_ids) + len(query_ids) + len(nl_ids)
    targets = torch.tensor(answer_ids, device=device)
    nll = torch.nn.functional.cross_entropy(
        out.logits[0, ans_start-1:ans_start-1+len(answer_ids)], targets).item()
    del out, cache_copy
    return nll


def score_nll_standard(model, device, cache, D, query_ids, answer_ids, nl_ids):
    """Standard Phase B NLL scoring."""
    phase_b_ids = nl_ids + query_ids + nl_ids + answer_ids
    n_tokens = len(phase_b_ids)
    pos = torch.arange(D + 1, D + 1 + n_tokens, device=device).unsqueeze(0)
    cache_copy = deep_copy_cache(cache)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([phase_b_ids], device=device),
                    position_ids=pos, past_key_values=cache_copy, use_cache=False)

    ans_start = len(nl_ids) + len(query_ids) + len(nl_ids)
    targets = torch.tensor(answer_ids, device=device)
    nll = torch.nn.functional.cross_entropy(
        out.logits[0, ans_start-1:ans_start-1+len(answer_ids)], targets).item()
    del out, cache_copy
    return nll


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GENERATION EVAL + INFERENCE-TIME BASELINE")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    all_samples = load_eval_datasets()

    # Load TF-IDF keywords
    print("Loading TF-IDF keywords...")
    generated = json.loads(PREFIXES_PATH.read_text())
    tfidf_keywords = generated["tfidf_keywords"]

    for model_key, model_spec in MODELS.items():
        print(f"\n{'#'*70}")
        print(f"# {model_key} ({model_spec['name']})")
        print(f"{'#'*70}")

        model_dir = RESULTS_DIR / model_key
        model_dir.mkdir(exist_ok=True, mode=0o777)

        t0 = time.time()
        model, tokenizer = load_model(model_spec["name"], model_spec["loader"])
        device = next(model.parameters()).device
        inv_freqs = build_layer_inv_freqs(model, device=device)
        layer_types = get_layer_types(model)
        sliding_limit = get_sliding_cache_limit(model)
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.pad_token_id

        if sliding_limit is not None:
            max_doc = sliding_limit - 1 - 64 - len(nl_ids)
        else:
            max_doc = 765

        # Build extract prefix
        extract_ids = tokenizer.encode(EXTRACT_INSTRUCTION, add_special_tokens=False)
        extract_prefix = make_prefix(extract_ids, 64)

        print(f"  Loaded in {time.time()-t0:.0f}s")

        for ds_key in EVAL_DATASETS:
            print(f"\n  --- {ds_key} ---")
            samples = all_samples[ds_key]
            ckpt_path = model_dir / f"results_{ds_key}.json"

            # Check for existing results
            scored = []
            scoring_key = f"gen_eval_{model_key}"
            if ckpt_path.exists():
                ckpt = json.loads(ckpt_path.read_text())
                if ckpt.get("scoring_key") == scoring_key:
                    scored = ckpt["results"]
                    print(f"  Resumed: {len(scored)}/{len(samples)}")

            for idx in range(len(scored), len(samples)):
                s = samples[idx]
                doc_ids = tokenizer.encode(s["passage"], add_special_tokens=False)[:max_doc]
                query_ids = tokenizer.encode(s["query"], add_special_tokens=False)
                answer_ids = tokenizer.encode(s["answer"], add_special_tokens=False)
                if not answer_ids:
                    continue
                D = len(doc_ids)

                # TF-IDF keywords for this document
                kw_text = tfidf_keywords[ds_key][idx]
                kw_ids = tokenizer.encode(kw_text, add_special_tokens=False)[:64]

                result = {
                    "query": s["query"][:200],
                    "answer": s["answer"][:200],
                }

                with torch.no_grad():
                    # === CONDITION 1: Bare ===
                    cache_bare, D = build_primed_cache(
                        model, tokenizer, device, doc_ids, None,
                        inv_freqs, layer_types, sliding_limit, bos_id, nl_ids)
                    result["nll_bare"] = score_nll_standard(
                        model, device, cache_bare, D, query_ids, answer_ids, nl_ids)
                    result["gen_bare"] = generate_answer(
                        model, tokenizer, device, cache_bare, D, query_ids, nl_ids, MAX_NEW_TOKENS)

                    # === CONDITION 2: Keywords (cache-time priming) ===
                    cache_kw, D = build_primed_cache(
                        model, tokenizer, device, doc_ids, kw_ids,
                        inv_freqs, layer_types, sliding_limit, bos_id, nl_ids)
                    result["nll_keywords"] = score_nll_standard(
                        model, device, cache_kw, D, query_ids, answer_ids, nl_ids)
                    result["gen_keywords"] = generate_answer(
                        model, tokenizer, device, cache_kw, D, query_ids, nl_ids, MAX_NEW_TOKENS)
                    del cache_kw

                    # === CONDITION 3: Extract instruction (cache-time priming) ===
                    cache_ext, D = build_primed_cache(
                        model, tokenizer, device, doc_ids, extract_prefix,
                        inv_freqs, layer_types, sliding_limit, bos_id, nl_ids)
                    result["nll_extract"] = score_nll_standard(
                        model, device, cache_ext, D, query_ids, answer_ids, nl_ids)
                    result["gen_extract"] = generate_answer(
                        model, tokenizer, device, cache_ext, D, query_ids, nl_ids, MAX_NEW_TOKENS)
                    del cache_ext

                    # === CONDITION 4: Inference-time baseline (keywords at query time) ===
                    result["nll_inftime_kw"] = score_nll_inference_baseline(
                        model, tokenizer, device, cache_bare, D,
                        query_ids, answer_ids, kw_ids, nl_ids)
                    result["gen_inftime_kw"] = generate_answer(
                        model, tokenizer, device, cache_bare, D,
                        kw_ids + nl_ids + query_ids, nl_ids, MAX_NEW_TOKENS)

                    del cache_bare

                # Compute EM/F1/Contains
                ref = s["answer"]
                for cond in ["bare", "keywords", "extract", "inftime_kw"]:
                    gen = result[f"gen_{cond}"]
                    if ds_key == "gsm8k":
                        result[f"em_{cond}"] = gsm8k_exact_match(gen, ref)
                        result[f"contains_{cond}"] = normalize_answer(ref) in normalize_answer(extract_first_answer(gen))
                    elif ds_key == "triviaqa":
                        aliases = s.get("aliases", []) + [ref]
                        result[f"em_{cond}"] = any(exact_match(gen, a) for a in aliases)
                        result[f"contains_{cond}"] = any(contains_match(gen, a) for a in aliases)
                        result[f"f1_{cond}"] = max(f1_score(gen, a) for a in aliases)
                    else:
                        result[f"em_{cond}"] = exact_match(gen, ref)
                        result[f"contains_{cond}"] = contains_match(gen, ref)
                        result[f"f1_{cond}"] = f1_score(gen, ref)

                scored.append(result)
                torch.cuda.empty_cache()

                if (idx + 1) % 20 == 0:
                    ckpt_path.write_text(json.dumps({"scoring_key": scoring_key, "results": scored}))
                    em_bare = np.mean([r["em_bare"] for r in scored])
                    em_kw = np.mean([r["em_keywords"] for r in scored])
                    print(f"    [{idx+1}/{len(samples)}] EM: bare={em_bare:.1%}, kw={em_kw:.1%}")

            # Final save
            ckpt_path.write_text(json.dumps({"scoring_key": scoring_key, "results": scored}))

            # Summary
            if scored:
                for cond in ["bare", "keywords", "extract", "inftime_kw"]:
                    em = np.mean([r[f"em_{cond}"] for r in scored])
                    nll_key = f"nll_{cond}" if cond != "inftime_kw" else "nll_inftime_kw"
                    nll = np.mean([r.get(nll_key, 0) for r in scored])
                    print(f"  {cond:<15s}: EM={em:.1%}, mean NLL={nll:.3f}")

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        purge_hf_cache(model_spec["name"])
        print(f"  Total: {time.time()-t0:.0f}s")

    print(f"\n{'='*70}")
    print("ALL MODELS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
