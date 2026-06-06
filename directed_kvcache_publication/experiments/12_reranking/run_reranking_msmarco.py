#!/usr/bin/env python3
"""Proper reranking test: MS MARCO passage reranking via cache priming.

MS MARCO v2.1 gives, per query, 10 BM25-retrieved candidate passages with exactly
one marked relevant (is_selected=1); the other 9 are genuine hard negatives. We
cache each passage (bare vs extract-primed; RAG precomputed-cache premise) and rank
by query-likelihood P(query | passage). Reranking is entropy-invariant (relative)
and has real headroom here (unlike QA data, where the query is derived from its
passage -> ceiling MRR=1). Tests whether content amplification improves a real
RAG-reranking metric (MRR, Recall@1).

Stores per query: relevant_idx and, per condition, the 10 query-NLLs.

Models: Qwen 1.5B, Qwen 7B, Gemma 12B. N=300 queries, 10-way reranking.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/12_reranking/run_reranking_msmarco.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/12_reranking/run_reranking_msmarco.py
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time
import random as pyrandom
from pathlib import Path

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
EXTRACT = "Extract the key facts from this text."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp13_rerank_msmarco"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)

MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}


def load_msmarco():
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1:   # exactly one relevant
            continue
        rel = sel.index(1)
        if not (5 <= count_words(pt[rel]) <= 300):
            continue
        out.append({"query": x["query"], "passages": pt, "relevant_idx": rel})
        if len(out) >= N_EVAL:
            break
    return out


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


_M = {}
def encode_passage(doc_ids, prefix_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    ids = [bos] + (list(prefix_ids) + nl if prefix_ids is not None else []) + list(doc_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; D = len(doc_ids)
    doc_start = (1 + len(prefix_ids) + len(nl)) if prefix_ids is not None else 1
    cache = select_kv_cache(cache, [0] + list(range(doc_start, doc_start + D)), device=dev)
    if prefix_ids is not None:
        cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
    return norm_roundtrip_kv_cache(cache), D

def query_nll(cache, D, q_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids); pos = torch.arange(D+1, D+1+len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(nl); al = out.logits[0][a0-1:a0-1+len(q_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(q_ids, device=dev)).item()


def main():
    print(f"MS MARCO RERANKING  SMOKE={SMOKE} N={N_EVAL}")
    samples = load_msmarco(); print(f"  {len(samples)} queries (exactly-1-relevant, 10-way)")
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        print(f"  loaded in {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"rerankmm_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  resumed {len(scored)}")
        for idx in range(len(scored), len(samples)):
            s = samples[idx]; q_ids = tok.encode(s["query"], add_special_tokens=False)
            rec = {"relevant_idx": s["relevant_idx"]}
            for cond, pfx in [("bare", None), ("extract", ext)]:
                nlls = []
                for ptext in s["passages"]:
                    doc_ids = tok.encode(ptext, add_special_tokens=False)[:max_doc]
                    if not doc_ids: nlls.append(float("inf")); continue
                    cache, D = encode_passage(doc_ids, pfx)
                    nlls.append(query_nll(cache, D, q_ids)); del cache; torch.cuda.empty_cache()
                rec[f"{cond}__q"] = nlls
            scored.append(rec)
            if (idx+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                def metrics(cond):
                    rr = []; r1 = []
                    for r in scored:
                        ri = r["relevant_idx"]; nl = r[f"{cond}__q"]
                        rank = 1 + sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])
                        rr.append(1.0/rank); r1.append(int(rank == 1))
                    return np.mean(rr), np.mean(r1)
                bmrr, br1 = metrics("bare"); emrr, er1 = metrics("extract")
                print(f"    [{idx+1}/{len(samples)}] MRR bare={bmrr:.3f} ext={emrr:.3f} (Δ{emrr-bmrr:+.3f}) | R@1 bare={br1:.3f} ext={er1:.3f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
