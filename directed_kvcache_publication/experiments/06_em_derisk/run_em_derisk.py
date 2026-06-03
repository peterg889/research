#!/usr/bin/env python3
"""PHASE 0 de-risk: does the 'extract' intervention move REAL task accuracy (EM/F1)?

Gates the whole research program. We have intervention->margin (exp05/06). We have
NOT shown intervention->accuracy. Prediction: if the contrastive margin is a valid
proxy, extract (margin+) should give EM+ while keywords/comprehend (margin~0) give
EM~0. bare and primed conditions differ ONLY in the cache; identical Phase-B format,
so format noise is a constant offset and the paired Δ(primed-bare) is a clean test.

Conditions (prefix length-matched to L=16, matching exp06):
  bare        no prefix
  extract     "Extract the key facts from this text."   (TEST; margin+ in exp06)
  comprehend  "Read and comprehend this text carefully." (null control)
  keywords    TF-IDF top-10 doc words                    (null control; exp05 margin~0)

Metrics: EM, F1, contains (substring), per condition; analysis reports paired
Δ(primed-bare) with bootstrap CIs.

Models: qwen25_1_5b, qwen25_7b, gemma3_12b, ministral_8b
Datasets: squad_v2, hotpotqa, triviaqa (extractive, clean EM). 300 samples.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/06_em_derisk/run_em_derisk.py        # full
    SMOKE=1 PYTHONPATH=... python3 experiments/06_em_derisk/run_em_derisk.py                        # smoke
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, re, shutil, time, string
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
N_EVAL = 10 if SMOKE else 300
L_MATCH = 16
MAX_NEW = 32
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp07_em_derisk"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
PREFIXES_PATH = Path(__file__).resolve().parent.parent / "02_ablation" / "generated_prefixes.json"
TFIDF = json.loads(PREFIXES_PATH.read_text())["tfidf_keywords"]

INSTRUCTIONS = {
    "extract": "Extract the key facts from this text.",
    "comprehend": "Read and comprehend this text carefully.",
}
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "ministral_8b":{"name": "mistralai/Ministral-8B-Instruct-2410", "loader": "AutoModelForCausalLM"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}
DATASETS = ["squad_v2", "hotpotqa", "triviaqa"]
if SMOKE:
    DATASETS = ["squad_v2"]


# ---------- metrics ----------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def first_line(text):
    for stop in ["\n", "Question:", "Q:", "Context:", "Passage:", "Human:", "Note:"]:
        i = text.find(stop)
        if i > 0:
            text = text[:i]
    return text.strip()

def em(pred, golds):
    p = normalize_answer(first_line(pred))
    return int(any(p == normalize_answer(g) for g in golds))

def contains(pred, golds):
    p = normalize_answer(first_line(pred))
    return int(any(normalize_answer(g) in p for g in golds if g.strip()))

def f1(pred, golds):
    p = normalize_answer(first_line(pred)).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        if not p or not gt:
            continue
        common = Counter(p) & Counter(gt); ns = sum(common.values())
        if ns == 0:
            continue
        prec = ns / len(p); rec = ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ---------- datasets (same seeds/filters as exp05/06) ----------
def load_datasets():
    out = {}
    ds = load_dataset("rajpurkar/squad_v2", split="validation"); cand = []
    for it in ds:
        p = it.get("context", ""); q = it.get("question", "")
        ans = it.get("answers", {}).get("text", [])
        if p and q and ans:
            wc = count_words(p)
            if 30 <= wc <= 500:
                cand.append({"passage": p, "query": q, "golds": list(dict.fromkeys(ans)), "passage_words": wc})
    pyrandom.seed(SEED + 200); pyrandom.shuffle(cand); out["squad_v2"] = cand[:N_EVAL]
    del ds, cand; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation"); cand = []
    for it in ds:
        wc_ctx = it.get("entity_pages", {}).get("wiki_context", [])
        if not wc_ctx or not wc_ctx[0]: continue
        p = " ".join(wc_ctx[0].split()[:500]); q = it["question"]
        val = it["answer"]["value"]; aliases = it["answer"].get("aliases", [])
        pl = p.lower()
        if not (val.lower() in pl or any(a.lower() in pl for a in aliases)): continue
        wc = count_words(p)
        if 30 <= wc <= 500 and count_words(val) >= 1:
            cand.append({"passage": p, "query": q, "golds": [val] + aliases, "passage_words": wc})
    pyrandom.seed(SEED + 300); pyrandom.shuffle(cand); out["triviaqa"] = cand[:N_EVAL]
    del ds, cand; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); cand = []
    for it in ds:
        ctx = it.get("context", {}); sf = it.get("supporting_facts", {})
        t2s = {t: s for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))}
        parts = [t2s[t][sid] for t, sid in zip(sf.get("title", []), sf.get("sent_id", []))
                 if t in t2s and sid < len(t2s[t])]
        if not parts: continue
        p = " ".join(parts); q = it["question"]; a = it["answer"]
        wc = count_words(p)
        if 30 <= wc <= 500 and count_words(a) >= 1:
            cand.append({"passage": p, "query": q, "golds": [a], "passage_words": wc})
    pyrandom.seed(SEED + 400); pyrandom.shuffle(cand); out["hotpotqa"] = cand[:N_EVAL]
    del ds, cand; gc.collect()
    return out


# ---------- model ----------
def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p):
        gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e9
        shutil.rmtree(p); print(f"  purged {name} ({gb:.1f} GB)")


# ---------- pipeline ----------
def build_cache(model, dev, doc_ids, prefix_ids, inv_freqs, lt, slim, bos, nl):
    ids = [bos] + (list(prefix_ids) + nl if prefix_ids is not None else []) + list(doc_ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; del out
    doc_start = (1 + len(prefix_ids) + len(nl)) if prefix_ids is not None else 1
    D = len(doc_ids); keep = [0] + list(range(doc_start, doc_start + D))
    cache = select_kv_cache(cache, keep, device=dev)
    if prefix_ids is not None:
        cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv_freqs, lt, bos_start=0)
    cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def generate(model, tok, dev, cache, D, prompt, nl, max_new=MAX_NEW):
    # prompt: full Phase-B prompt token ids (incl. question/answer elicitation).
    pos0 = D + 1
    cc = deep_copy_cache(cache)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt], device=dev),
                    position_ids=torch.arange(pos0, pos0 + len(prompt), device=dev).unsqueeze(0),
                    past_key_values=cc, use_cache=True)
    cache_g = out.past_key_values
    nxt = out.logits[0, -1:].argmax(-1); gen = [nxt.item()]
    nl_set = set(nl)
    for _ in range(max_new - 1):
        pos = pos0 + len(prompt) + len(gen) - 1
        with torch.no_grad():
            out = model(input_ids=nxt.unsqueeze(0),
                        position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cache_g, use_cache=True)
        cache_g = out.past_key_values
        nxt = out.logits[0, -1:].argmax(-1); t = nxt.item()
        if t == tok.eos_token_id or t in nl_set:
            break
        gen.append(t)
    del cache_g
    return tok.decode(gen, skip_special_tokens=True)


def main():
    print(f"PHASE 0 EM DE-RISK  SMOKE={SMOKE} N_EVAL={N_EVAL}")
    print("Loading datasets..."); data = load_datasets()
    for k in DATASETS: print(f"  {k}: {len(data[k])}")

    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        mdir = RESULTS / mk; mdir.mkdir(exist_ok=True, mode=0o777)
        skey = f"emderisk_{mk}" + ("_smoke" if SMOKE else "")
        t0 = time.time()
        model, tok = load_model(spec["name"], spec["loader"])
        dev = next(model.parameters()).device
        inv_freqs = build_layer_inv_freqs(model, device=dev)
        lt = get_layer_types(model); slim = get_sliding_cache_limit(model)
        nl = tok.encode("\n", add_special_tokens=False)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
        max_doc = (slim - 1 - 64 - len(nl)) if slim is not None else 765
        instr_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in INSTRUCTIONS.items()}
        # Answer-elicitation wrappers: concise extractive answers, identical across conditions.
        q_pre = tok.encode("Question: ", add_special_tokens=False)
        a_pre = tok.encode("Answer:", add_special_tokens=False)
        print(f"  loaded in {time.time()-t0:.0f}s")

        for dk in DATASETS:
            samples = data[dk]; ck = mdir / f"results_{dk}.json"; scored = []
            if ck.exists():
                prev = json.loads(ck.read_text())
                if prev.get("scoring_key") == skey:
                    scored = prev["results"]; print(f"  {dk}: resumed {len(scored)}")
            for idx in range(len(scored), len(samples)):
                s = samples[idx]
                doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
                q_ids = tok.encode(s["query"], add_special_tokens=False)
                prompt_ids = nl + q_pre + q_ids + nl + a_pre  # [\n Question: <q> \n Answer:]
                D = len(doc_ids)
                kw_ids = tok.encode(TFIDF[dk][idx], add_special_tokens=False)
                prefixes = {
                    "bare": None,
                    "extract": make_prefix(instr_ids["extract"], L_MATCH),
                    "comprehend": make_prefix(instr_ids["comprehend"], L_MATCH),
                    "keywords": make_prefix(kw_ids, L_MATCH) if kw_ids else None,
                }
                rec = {"golds": s["golds"][:10]}
                for cond, pfx in prefixes.items():
                    if cond != "bare" and pfx is None:
                        continue
                    cache, D = build_cache(model, dev, doc_ids, pfx, inv_freqs, lt, slim, bos, nl)
                    gen = generate(model, tok, dev, cache, D, prompt_ids, nl)
                    rec[f"{cond}__gen"] = gen[:200]
                    rec[f"{cond}__em"] = em(gen, s["golds"])
                    rec[f"{cond}__f1"] = f1(gen, s["golds"])
                    rec[f"{cond}__contains"] = contains(gen, s["golds"])
                    del cache
                    torch.cuda.empty_cache()
                scored.append(rec)
                if (idx + 1) % 20 == 0 or SMOKE:
                    ck.write_text(json.dumps({"scoring_key": skey, "results": scored}))
                    be = np.mean([r["bare__em"] for r in scored])
                    ee = np.mean([r["extract__em"] for r in scored])
                    print(f"    [{idx+1}/{len(samples)}] EM bare={be:.1%} extract={ee:.1%} (Δ={ee-be:+.1%})")
            ck.write_text(json.dumps({"scoring_key": skey, "results": scored}))
            print(f"  {dk}: {len(scored)} done")
        del model, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"])

    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
