#!/usr/bin/env python3
"""exp14b: DOUBLE-DOWN ablations on the contrastive-priming hypothesis (MS MARCO).

The audit of exp14 identified three threats to the key claim, each testable:

 T1  "Contrastive" may not be the active ingredient — exp14 never ran PLAIN passage
     TF-IDF terms (no neighbor subtraction). If tfidf_plain ≈ distinctive, the active
     ingredient is "passage keywords," not the contrast.
       -> condition `tfidf_plain`
 T2  Cacheability leak: exp14's `distinctive_corpus` searched neighbors over the pooled
     candidate corpus WITHOUT excluding same-query candidates (BM25 siblings), so its
     contrast set is partially query-conditioned.
       -> condition `dist_clean` (neighbors exclude same-query candidates)
 T3  Best-of-both never tested: `extract` is the only margin-positive instruction
     (exp05/06); distinctive terms are the selective content. The natural hybrid is
     "Extract the key facts about: {terms}." at the SAME L=16 budget.
       -> condition `hybrid`

Conditions: bare, generic, tfidf_plain, dist_corpus (original, for within-run pairing),
dist_clean, hybrid. All prefixes length-matched (make_prefix tiling, L=16).

Env knobs: ONLY_MODELS, N_EVAL (default 300; use 1200 for the high-power vs-bare run),
CONDS (comma list to subset conditions), SMOKE.
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
N_EVAL = 10 if SMOKE else int(os.environ.get("N_EVAL", "300"))
L_MATCH = 16
TOPK = 10
EXTRACT = "Extract the key facts from this text."
ALL_CONDS = ["bare", "generic", "tfidf_plain", "dist_corpus", "dist_clean", "hybrid"]
CONDS = [c for c in os.environ.get("CONDS", ",".join(ALL_CONDS)).split(",") if c in ALL_CONDS]
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp14b_ablations"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_4b":      {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_12b":     {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":      {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
    "gemma3_4b_base": {"name": "google/gemma-3-4b-pt", "loader": "Gemma3ForConditionalGeneration"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY:
    MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
elif SMOKE:
    MODELS = {"gemma3_4b": MODELS["gemma3_4b"]}


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


def build_prefix_texts(samples):
    """Per (query, candidate): tfidf_plain, dist_corpus (original leaky), dist_clean
    (same-query candidates excluded from the neighbor pool), hybrid (scaffold + clean terms)."""
    all_passages, offsets = [], []
    for s in samples:
        offsets.append(len(all_passages)); all_passages.extend(s["passages"])
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    mat = vec.fit_transform(all_passages)
    feats = vec.get_feature_names_out()

    def top_terms(idx, neighbor_idx, topk=TOPK):
        v = mat[idx].toarray().ravel()
        comp = mat[neighbor_idx].toarray().mean(axis=0) if len(neighbor_idx) else np.zeros_like(v)
        d = v - comp
        top = np.argsort(-d)[:topk]
        return " ".join(feats[t] for t in top if d[t] > 0)

    P = {}
    for qi, s in enumerate(samples):
        base = offsets[qi]; nc = len(s["passages"])
        same_q = set(range(base, base + nc))
        for ci in range(nc):
            gidx = base + ci
            sims = (mat @ mat[gidx].T).toarray().ravel(); sims[gidx] = -1
            order = np.argsort(-sims)
            nb_orig = order[:10]                                        # leaky (exp14 behavior)
            nb_clean = [j for j in order if j not in same_q][:10]       # leakage-free
            plain = top_terms(gidx, np.array([], dtype=int))
            dcorp = top_terms(gidx, nb_orig)
            dclean = top_terms(gidx, np.array(nb_clean))
            P[(qi, ci)] = {"tfidf_plain": plain, "dist_corpus": dcorp, "dist_clean": dclean,
                           "hybrid": f"Extract the key facts about: {dclean}."}
    return P


def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        m = Gemma3ForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
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
    print(f"CONTRASTIVE ABLATIONS (exp14b)  SMOKE={SMOKE} N={N_EVAL} conds={CONDS}")
    samples = load_msmarco(); print(f"  {len(samples)} queries")
    P = build_prefix_texts(samples)
    ex = P[(0, samples[0]["relevant_idx"])]
    for k in ["tfidf_plain", "dist_corpus", "dist_clean", "hybrid"]:
        print(f"  example {k}: {ex[k][:70]!r}")

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
        skey = f"abl_{mk}_n{N_EVAL}_" + "-".join(CONDS) + ("_smoke" if SMOKE else "")
        scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  resumed {len(scored)}")
        for qi in range(len(scored), len(samples)):
            s = samples[qi]; q_ids = tok.encode(s["query"], add_special_tokens=False)
            rec = {"relevant_idx": s["relevant_idx"]}
            for cond in CONDS:
                nlls = []
                for ci, ptext in enumerate(s["passages"]):
                    doc_ids = tok.encode(ptext, add_special_tokens=False)[:max_doc]
                    if not doc_ids: nlls.append(float("inf")); continue
                    if cond == "bare": pfx = None
                    elif cond == "generic": pfx = ext
                    else:
                        txt = P[(qi, ci)][cond]
                        ids = tok.encode(txt, add_special_tokens=False)
                        pfx = make_prefix(ids, L_MATCH) if ids else ext
                    cache, D = encode_passage(doc_ids, pfx)
                    nlls.append(query_nll(cache, D, q_ids)); del cache; torch.cuda.empty_cache()
                rec[f"{cond}__q"] = nlls
            scored.append(rec)
            if (qi+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                def mrr(c):
                    o = []
                    for r in scored:
                        ri = r["relevant_idx"]; nl = r[f"{c}__q"]
                        o.append(1.0/(1+sum(1 for j, x in enumerate(nl) if j != ri and x < nl[ri])))
                    return np.mean(o)
                print(f"    [{qi+1}/{len(samples)}] MRR " + " ".join(f"{c}={mrr(c):.3f}" for c in CONDS))
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache()
        if os.environ.get("KEEP_WEIGHTS", "0") != "1": purge(spec["name"])
        _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
