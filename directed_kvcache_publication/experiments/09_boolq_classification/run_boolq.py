#!/usr/bin/env python3
"""Sharpening value test #1: BoolQ document-grounded binary classification.

Hypothesis: cache priming with an extract-style instruction concentrates probability
on the correct answer (real differential signal, d~0.2; verified by decomposition).
This barely moves open-ended-generation argmax, but should help most where the
decision boundary is DENSE and the metric reads the DISTRIBUTION. Binary
classification is the densest-boundary case.

BoolQ: passage + yes/no question. We CACHE the passage (bare vs primed), then read
the verbalizer logits for {yes-variants} vs {no-variants} at the answer position
(no generation needed -> one forward per sample/condition). We store the two class
logits + gold per sample/condition; all metrics are computed in analysis.

Conditions: bare, extract (signal), comprehend (weak), keywords (TF-IDF; entropy ctrl).
Models: Qwen 1.5B, Qwen 7B, Gemma 12B, Ministral 8B.

Rigor / confound controls (computed in analysis from stored logits):
  - accuracy, balanced accuracy (BoolQ is 62% True), ECE (15 bins), Brier,
    selective risk-coverage AUC, logit margin = class_logit(gold)-class_logit(other).
  - PRIOR-SHIFT control: margin gain split by gold class (yes vs no). Real
    discrimination sharpening => BOTH positive; a mere label-prior shift => one up,
    one down.
  - contextual-calibration prior p_cf per condition (content-free passage), stored
    in metadata for prior-corrected accuracy.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/09_boolq_classification/run_boolq.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/09_boolq_classification/run_boolq.py
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time, string, math
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
from model_adapters import build_layer_inv_freqs, get_layer_types, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EVAL = 20 if SMOKE else 3270
L_MATCH = 16
PLACE = "<<<DOCPLACEHOLDER>>>"
EXTRACT = "Extract the key facts from this text."
COMPREHEND = "Read and comprehend this text carefully."
NULL_PASSAGE = "N/A"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp10_boolq"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)

MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "ministral_8b":{"name": "mistralai/Ministral-8B-Instruct-2410", "loader": "AutoModelForCausalLM"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}

YES_VARIANTS = [" Yes", " yes", "Yes"]
NO_VARIANTS  = [" No", " no", "No"]


# ---------- TF-IDF keywords (inline over the eval passages) ----------
_STRIP = ".,;:!?" + chr(34) + "'()[]{}"
def tokenize_simple(t):
    return [w.strip(_STRIP) for w in t.lower().split() if len(w) > 2]

def compute_tfidf(passages, top_k=10):
    df = Counter()
    toks = []
    for p in passages:
        s = set(tokenize_simple(p)); toks.append(s)
        for w in s: df[w] += 1
    N = len(passages); out = []
    for p in passages:
        tf = Counter(tokenize_simple(p)); mx = max(tf.values()) if tf else 1
        sc = {w: (c/mx)*math.log(N/df[w]) for w, c in tf.items() if df[w] >= 2}
        out.append(" ".join(sorted(sc, key=sc.get, reverse=True)[:top_k]))
    return out


# ---------- model ----------
def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)

def chat_pieces(tok, question):
    content = PLACE + f"\n\nQuestion: {question}\nAnswer with a single word, yes or no:"
    msgs = [{"role": "user", "content": content}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre, tail = rendered.split(PLACE)
    return tok.encode(pre, add_special_tokens=False), tok.encode(tail, add_special_tokens=False)


# ---------- two-phase cache (chat prefix kept fixed; priming stripped+repositioned) ----------
def build_cache(model, dev, chat_pre, doc_ids, priming_ids, inv_freqs, lt, nl):
    C = len(chat_pre); D = len(doc_ids)
    if priming_ids is None:
        ids = list(chat_pre) + list(doc_ids)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
        return norm_roundtrip_kv_cache(out.past_key_values), C, D
    P = len(priming_ids); NL = len(nl)
    ids = list(chat_pre) + list(priming_ids) + nl + list(doc_ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; doc_start = C + P + NL
    cache = select_kv_cache(cache, list(range(C)) + list(range(doc_start, doc_start + D)), device=dev)
    cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                torch.arange(C, C + D, device=dev), inv_freqs, lt, bos_start=C - 1)
    return norm_roundtrip_kv_cache(cache), C, D

def class_logits(model, dev, cache, C, D, tail_ids, yes_ids, no_ids):
    """Forward the tail; aggregate verbalizer logits at the final position.
    class logit = logsumexp over variant token logits."""
    pos = torch.arange(C + D, C + D + len(tail_ids), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tail_ids], device=dev), position_ids=pos,
                    past_key_values=deep_copy_cache(cache), use_cache=False)
    logits = out.logits[0, -1].float()  # next-token logits at answer position
    yl = torch.logsumexp(logits[yes_ids], dim=0).item()
    nl_ = torch.logsumexp(logits[no_ids], dim=0).item()
    return yl, nl_


def main():
    print(f"BoolQ classification  SMOKE={SMOKE} N={N_EVAL}")
    ds = load_dataset("google/boolq", split="validation")
    samples = [{"passage": x["passage"], "question": x["question"], "gold": bool(x["answer"])}
               for x in ds if 10 <= count_words(x["passage"]) <= 500][:N_EVAL]
    kws = compute_tfidf([s["passage"] for s in samples])
    print(f"  {len(samples)} samples, frac True={np.mean([s['gold'] for s in samples]):.2f}")

    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); model, tok = load_model(spec["name"], spec["loader"]); dev = next(model.parameters()).device
        inv_freqs = build_layer_inv_freqs(model, device=dev); lt = get_layer_types(model)
        slim = get_sliding_cache_limit(model); nl = tok.encode("\n", add_special_tokens=False)
        max_doc = (slim - 1 - 96 - len(nl)) if slim is not None else 760
        yes_ids = [tok.encode(v, add_special_tokens=False)[0] for v in YES_VARIANTS]
        no_ids  = [tok.encode(v, add_special_tokens=False)[0] for v in NO_VARIANTS]
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        comp = make_prefix(tok.encode(COMPREHEND, add_special_tokens=False), L_MATCH)
        print(f"  loaded in {time.time()-t0:.0f}s; yes_ids={yes_ids} no_ids={no_ids}")

        ckpt = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"boolq_{mk}" + ("_smoke" if SMOKE else "")
        scored = []
        if ckpt.exists():
            prev = json.loads(ckpt.read_text())
            if prev.get("scoring_key") == skey:
                scored = prev["samples"]; print(f"  resumed {len(scored)}")

        # content-free prior per condition (Zhao et al. contextual calibration)
        prior = {}
        if not scored:
            null_doc = tok.encode(NULL_PASSAGE, add_special_tokens=False)
            for cond, pfx in [("bare", None), ("extract", ext), ("comprehend", comp)]:
                ys, ns = [], []
                for s in samples[:40]:
                    cp, tl = chat_pieces(tok, s["question"])
                    cache, C, D = build_cache(model, dev, cp, null_doc, pfx, inv_freqs, lt, nl)
                    yl, nl_ = class_logits(model, dev, cache, C, D, tl, yes_ids, no_ids); del cache
                    p = 1/(1+math.exp(nl_-yl)); ys.append(p); ns.append(1-p)
                prior[cond] = float(np.mean(ys))
            print(f"  content-free prior P(yes): {prior}")

        for idx in range(len(scored), len(samples)):
            s = samples[idx]
            doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
            cp, tl = chat_pieces(tok, s["question"])
            kw = make_prefix(tok.encode(kws[idx], add_special_tokens=False), L_MATCH) if kws[idx] else None
            rec = {"gold": int(s["gold"])}
            for cond, pfx in [("bare", None), ("extract", ext), ("comprehend", comp), ("keywords", kw)]:
                if cond == "keywords" and pfx is None: continue
                cache, C, D = build_cache(model, dev, cp, doc_ids, pfx, inv_freqs, lt, nl)
                yl, nl_ = class_logits(model, dev, cache, C, D, tl, yes_ids, no_ids); del cache
                rec[f"{cond}__yes"] = yl; rec[f"{cond}__no"] = nl_
                torch.cuda.empty_cache()
            scored.append(rec)
            if (idx + 1) % 100 == 0 or SMOKE:
                ckpt.write_text(json.dumps({"scoring_key": skey, "prior": prior, "samples": scored}))
                acc_b = np.mean([(r["bare__yes"] > r["bare__no"]) == bool(r["gold"]) for r in scored])
                acc_e = np.mean([(r["extract__yes"] > r["extract__no"]) == bool(r["gold"]) for r in scored])
                print(f"    [{idx+1}/{len(samples)}] acc bare={acc_b:.3f} extract={acc_e:.3f}")
        ckpt.write_text(json.dumps({"scoring_key": skey, "prior": prior, "samples": scored}))
        print(f"  done {len(scored)}")
        del model, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"])
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
