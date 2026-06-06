#!/usr/bin/env python3
"""The one bounded application test: passage reranking via cache priming.

Reranking is the setting where (a) the metric is naturally entropy-invariant
(relative ranking), and (b) the boundary signal exists at serve time (candidate
score gaps) — so the content-amplification + boundary-concentration we found could
combine into a usable gain, unlike open generation.

Setup: for each query, candidates = its SOURCE passage (relevant) + K random
distractor passages from the same dataset. Each passage is cached (bare vs
extract-primed; precomputed-cache RAG premise). We rank candidates by:
  query-likelihood  P(query | passage)        [realistic RAG reranking]
  answer-grounding  P(answer | passage, query) [secondary; answer = content]
Content amplification should lower the relevant passage's NLL more than distractors
=> better MRR / Recall@1. A uniform (entropy) shift would not change ranking.

Stores per query, per condition (bare/extract): the per-candidate query-NLL and
answer-NLL with the relevant candidate at index 0. Reranking metrics in analysis.

Models: Qwen 1.5B, Qwen 7B, Gemma 12B. Datasets: squad_v2, hotpotqa, triviaqa.
N=200 queries, K=7 distractor passages (8-way reranking).

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/12_reranking/run_reranking.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/12_reranking/run_reranking.py
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
N_EVAL = 8 if SMOKE else 200
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp12_reranking"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)

MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}
DATASETS = ["squad_v2", "hotpotqa", "triviaqa"]
if SMOKE:
    DATASETS = ["squad_v2"]


def load_datasets():
    out = {}
    ds = load_dataset("rajpurkar/squad_v2", split="validation"); C = []
    for it in ds:
        p, q = it.get("context", ""), it.get("question", ""); a = it.get("answers", {}).get("text", [])
        if p and q and a and 30 <= count_words(p) <= 500:
            C.append({"passage": p, "query": q, "answer": a[0]})
    pyrandom.seed(SEED+200); pyrandom.shuffle(C); out["squad_v2"] = C[:max(N_EVAL*3, 100)]; del ds, C; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation"); C = []
    for it in ds:
        wc = it.get("entity_pages", {}).get("wiki_context", [])
        if not wc or not wc[0]: continue
        p = " ".join(wc[0].split()[:500]); v = it["answer"]["value"]; al = it["answer"].get("aliases", []); pl = p.lower()
        if not (v.lower() in pl or any(x.lower() in pl for x in al)): continue
        if 30 <= count_words(p) <= 500 and count_words(v) >= 1:
            C.append({"passage": p, "query": it["question"], "answer": v})
    pyrandom.seed(SEED+300); pyrandom.shuffle(C); out["triviaqa"] = C[:max(N_EVAL*3, 100)]; del ds, C; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); C = []
    for it in ds:
        ctx = it.get("context", {}); sf = it.get("supporting_facts", {})
        t2s = {t: s for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))}
        parts = [t2s[t][sid] for t, sid in zip(sf.get("title", []), sf.get("sent_id", [])) if t in t2s and sid < len(t2s[t])]
        if not parts: continue
        p = " ".join(parts)
        if 30 <= count_words(p) <= 500 and count_words(it["answer"]) >= 1:
            C.append({"passage": p, "query": it["question"], "answer": it["answer"]})
    pyrandom.seed(SEED+400); pyrandom.shuffle(C); out["hotpotqa"] = C[:max(N_EVAL*3, 100)]; del ds, C; gc.collect()
    return out


def build_hard_negatives(pool):
    """For each query, the K_DISTRACT most lexically-similar OTHER passages (TF-IDF
    over the pool, queried by the question). Hard negatives share query vocabulary,
    so reranking is non-trivial (unlike random distractors -> ceiling MRR=1)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    passages = [p["passage"] for p in pool]
    queries = [p["query"] for p in pool]
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    P = vec.fit_transform(passages)
    Q = vec.transform(queries)
    sims = linear_kernel(Q, P)  # (n_queries, n_passages)
    hard = []
    for i in range(len(pool)):
        order = np.argsort(-sims[i])
        negs = [j for j in order if j != i][:K_DISTRACT]
        hard.append(negs)
    return hard


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

def score_nll(cache, D, prefix_ids, target_ids):
    """NLL of target_ids given the cache, with prefix_ids (e.g. [nl] or [nl,query,nl]) before."""
    m, dev = _M["m"], _M["dev"]
    seq = list(prefix_ids) + list(target_ids); pos = torch.arange(D+1, D+1+len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(prefix_ids)
    al = out.logits[0][a0-1:a0-1+len(target_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(target_ids, device=dev)).item()


def main():
    print(f"RERANKING  SMOKE={SMOKE} N={N_EVAL} K={K_DISTRACT}")
    data = load_datasets()
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        nl = _M["nl"]
        print(f"  loaded in {time.time()-t0:.0f}s")
        for dk in DATASETS:
            pool = data[dk]; samples = pool[:N_EVAL]
            hard_negs = build_hard_negatives(pool)  # TF-IDF hard negatives per query
            ck = RESULTS / mk / f"results_{dk}.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
            skey = f"rerank_{mk}" + ("_smoke" if SMOKE else ""); scored = []
            if ck.exists():
                prev = json.loads(ck.read_text())
                if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  {dk} resumed {len(scored)}")
            for idx in range(len(scored), len(samples)):
                s = samples[idx]
                q_ids = tok.encode(s["query"], add_special_tokens=False)
                a_ids = tok.encode(s["answer"], add_special_tokens=False)
                # candidates: relevant (idx 0) + K hard-negative passages (TF-IDF similar)
                cand_idx = [idx] + hard_negs[idx]
                rec = {"query": s["query"][:120]}
                for cond, pfx in [("bare", None), ("extract", ext)]:
                    qnlls, anlls = [], []
                    for j in cand_idx:
                        doc_ids = tok.encode(pool[j]["passage"], add_special_tokens=False)[:max_doc]
                        cache, D = encode_passage(doc_ids, pfx)
                        qnlls.append(score_nll(cache, D, nl, q_ids))                       # P(query|passage)
                        anlls.append(score_nll(cache, D, nl + list(q_ids) + nl, a_ids))    # P(answer|passage,query)
                        del cache; torch.cuda.empty_cache()
                    rec[f"{cond}__q"] = qnlls   # index 0 = relevant
                    rec[f"{cond}__a"] = anlls
                scored.append(rec)
                if (idx+1) % 20 == 0 or SMOKE:
                    ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                    def rr(key):
                        return np.mean([1.0/(1+sum(1 for x in r[key][1:] if x < r[key][0])) for r in scored])
                    print(f"    [{idx+1}/{len(samples)}] MRR(query) bare={rr('bare__q'):.3f} ext={rr('extract__q'):.3f} | "
                          f"MRR(ans) bare={rr('bare__a'):.3f} ext={rr('extract__a'):.3f}")
            ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  {dk}: {len(scored)} done")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
