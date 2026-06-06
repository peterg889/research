#!/usr/bin/env python3
"""Idea test #1: CONTRASTIVE / DISTINCTIVE priming on MS MARCO reranking.

Generic priming HURTS reranking because it is non-selective: it amplifies content
in the relevant passage AND in the lexically-similar hard negatives. The fix is to
prime each passage with what DISTINGUISHES it from its competitors, so the relevant
passage's discriminating content is amplified, not the shared content.

We test 4 conditions per passage (length-matched to L=16):
  bare                  no prefix
  generic               "Extract the key facts from this text." (non-selective; baseline)
  distinctive_corpus    top terms of this passage MINUS its nearest CORPUS neighbors
                        (query-agnostic -> CACHEABLE, realistic)
  distinctive_cand      top terms of this passage MINUS the OTHER candidates for its
                        query (uses the candidate set -> ORACLE upper bound, not
                        cacheable; tells us if the idea works in principle)

Rerank by query-likelihood P(query|passage). If distinctive_* improves MRR/R@1 where
generic hurt, contrastive priming breaks the non-selectivity failure mode.

Models: Qwen 1.5B, Qwen 7B, Gemma 12B. MS MARCO v2.1, 10-way, N=300.
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

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
TOPK = 10
EXTRACT = "Extract the key facts from this text."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp14_contrastive"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    # generalization set: replicate the Gemma-12B contrastive win within-family at scale
    # (27B), cross-family (Mistral 7B), and test whether larger Qwen becomes primable (14B).
    "gemma3_27b":  {"name": "google/gemma-3-27b-it", "loader": "Gemma3ForConditionalGeneration"},
    "mistral_7b":  {"name": "mistralai/Mistral-7B-Instruct-v0.3", "loader": "AutoModelForCausalLM"},
    "qwen25_14b":  {"name": "Qwen/Qwen2.5-14B-Instruct", "loader": "AutoModelForCausalLM"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY:
    MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


def load_msmarco():
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1: continue
        rel = sel.index(1)
        if not (5 <= count_words(pt[rel]) <= 300): continue
        out.append({"query": x["query"], "passages": pt, "relevant_idx": rel})
        if len(out) >= N_EVAL: break
    return out


def distinctive_terms(texts, idx, vectorizer, matrix, neighbor_idx=None, topk=TOPK):
    """Top terms of texts[idx] minus the mean of a comparison set (neighbors or others)."""
    vec = matrix[idx].toarray().ravel()
    if neighbor_idx is None:  # corpus neighbors: nearest by cosine within matrix
        sims = (matrix @ matrix[idx].T).toarray().ravel(); sims[idx] = -1
        neighbor_idx = np.argsort(-sims)[:10]
    comp = matrix[neighbor_idx].toarray().mean(axis=0) if len(neighbor_idx) else np.zeros_like(vec)
    diff = vec - comp
    feats = vectorizer.get_feature_names_out()
    top = np.argsort(-diff)[:topk]
    return " ".join(feats[t] for t in top if diff[t] > 0)


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
    print(f"CONTRASTIVE RERANK  SMOKE={SMOKE} N={N_EVAL}")
    samples = load_msmarco(); print(f"  {len(samples)} queries")
    # corpus TF-IDF over ALL candidate passages (query-agnostic distinctive_corpus)
    all_passages = []; offsets = []
    for s in samples:
        offsets.append(len(all_passages)); all_passages.extend(s["passages"])
    corpus_vec = TfidfVectorizer(stop_words="english", max_features=50000)
    corpus_mat = corpus_vec.fit_transform(all_passages)
    # precompute distinctive primes (text) per (query, candidate)
    dist_corpus = {}; dist_cand = {}
    for qi, s in enumerate(samples):
        base = offsets[qi]; cand_texts = s["passages"]
        cv = TfidfVectorizer(stop_words="english"); cm = cv.fit_transform(cand_texts)
        for ci in range(len(cand_texts)):
            dist_corpus[(qi, ci)] = distinctive_terms(all_passages, base + ci, corpus_vec, corpus_mat)
            others = [j for j in range(len(cand_texts)) if j != ci]
            dist_cand[(qi, ci)] = distinctive_terms(cand_texts, ci, cv, cm, neighbor_idx=others)
    print(f"  example distinctive_corpus: {dist_corpus[(0,samples[0]['relevant_idx'])][:80]!r}")
    print(f"  example distinctive_cand:   {dist_cand[(0,samples[0]['relevant_idx'])][:80]!r}")

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
        skey = f"contrast_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  resumed {len(scored)}")
        for qi in range(len(scored), len(samples)):
            s = samples[qi]; q_ids = tok.encode(s["query"], add_special_tokens=False)
            rec = {"relevant_idx": s["relevant_idx"]}
            for cond in ["bare", "generic", "distinctive_corpus", "distinctive_cand"]:
                nlls = []
                for ci, ptext in enumerate(s["passages"]):
                    doc_ids = tok.encode(ptext, add_special_tokens=False)[:max_doc]
                    if not doc_ids: nlls.append(float("inf")); continue
                    if cond == "bare": pfx = None
                    elif cond == "generic": pfx = ext
                    else:
                        dt = dist_corpus[(qi, ci)] if cond == "distinctive_corpus" else dist_cand[(qi, ci)]
                        ids = tok.encode(dt, add_special_tokens=False)
                        pfx = make_prefix(ids, L_MATCH) if ids else ext
                    cache, D = encode_passage(doc_ids, pfx)
                    nlls.append(query_nll(cache, D, q_ids)); del cache; torch.cuda.empty_cache()
                rec[f"{cond}__q"] = nlls
            scored.append(rec)
            if (qi+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                def mrr(cond):
                    out = []
                    for r in scored:
                        ri = r["relevant_idx"]; nl = r[f"{cond}__q"]
                        out.append(1.0/(1+sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])))
                    return np.mean(out)
                print(f"    [{qi+1}/{len(samples)}] MRR bare={mrr('bare'):.3f} gen={mrr('generic'):.3f} "
                      f"dist_corp={mrr('distinctive_corpus'):.3f} dist_cand={mrr('distinctive_cand'):.3f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
