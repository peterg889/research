#!/usr/bin/env python3
"""WS3 circuit-level: WHERE in the network does the prefix's influence live, and which
layers' primed-KV actually CAUSE the query-NLL change? Two analyses on gemma3_4b vs
qwen25_7b (matched MS MARCO query-passage pairs):

(1) PERTURBATION PROFILE (descriptive): per layer, relative change of the doc's cached
    keys/values between primed and bare encoding -> where the prefix perturbs the cache.

(2) LAYER-WISE KV PATCHING (causal, gold standard): score the query with a hybrid cache
    = bare everywhere EXCEPT layer L gets the primed K/V. effect_L = NLL(hybrid_L) -
    NLL(bare). Sweep L. Localizes which layers' primed-KV drives primability.

Compares the Gemma vs Qwen layer profiles to localize the family difference.
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc
from pathlib import Path
import numpy as np
import torch
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
SMOKE = os.environ.get("SMOKE", "0") == "1"
N = 3 if SMOKE else int(os.environ.get("N_DOCS", "15"))
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp20_circuit"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)


def load_msmarco(n):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1: continue
        rel = sel.index(1)
        if 5 <= count_words(pt[rel]) <= 200:
            out.append({"query": x["query"], "passage": pt[rel]})
        if len(out) >= n: break
    return out


_M = {}
def encode_full(doc_ids, prefix_ids):
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

def query_nll(cache, D, q_ids, copy=True):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids); pos = torch.arange(D+1, D+1+len(seq), device=dev).unsqueeze(0)
    pkv = deep_copy_cache(cache) if copy else cache
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=pkv, use_cache=False)
    a0 = len(nl); al = out.logits[0][a0-1:a0-1+len(q_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(q_ids, device=dev)).item()

def perturbation_profile(cb, cp):
    """per-layer relative L2 change of doc keys & values (positions 1.. , skip BOS@0)."""
    kpert, vpert = [], []
    for i in range(len(cb.layers)):
        kb = cb.layers[i].keys[:, :, 1:].float(); kp = cp.layers[i].keys[:, :, 1:].float()
        vb = cb.layers[i].values[:, :, 1:].float(); vp = cp.layers[i].values[:, :, 1:].float()
        kpert.append(((kp-kb).norm(dim=-1) / kb.norm(dim=-1).clamp_min(1e-6)).mean().item())
        vpert.append(((vp-vb).norm(dim=-1) / vb.norm(dim=-1).clamp_min(1e-6)).mean().item())
    return kpert, vpert

def patch_layer(cb, cp, L):
    h = deep_copy_cache(cb)
    h.layers[L].keys = cp.layers[L].keys.clone()
    h.layers[L].values = cp.layers[L].values.clone()
    return h


def main():
    print(f"CIRCUIT PRIMABILITY  SMOKE={SMOKE}  N={N}")
    samples = load_msmarco(N)
    out = {}
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        tok = AutoTokenizer.from_pretrained(spec["name"], token=HF_TOKEN)
        if spec["loader"] == "Gemma3ForConditionalGeneration":
            from transformers import Gemma3ForConditionalGeneration as Loader
        else:
            Loader = AutoModelForCausalLM
        m = Loader.from_pretrained(spec["name"], dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
        dev = next(m.parameters()).device
        _M.update(m=m, tok=tok, dev=dev, inv=build_layer_inv_freqs(m, device=dev),
                  lt=get_layer_types(m), slim=get_sliding_cache_limit(m),
                  nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        nL = len(m.model.layers) if hasattr(m, "model") and hasattr(m.model, "layers") else None
        kP, vP, effP, fulls = [], [], [], []
        for s in samples:
            q_ids = tok.encode(s["query"], add_special_tokens=False)
            doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
            cb, D = encode_full(doc_ids, None)
            cp, _ = encode_full(doc_ids, ext)
            nL = len(cb.layers)
            base = query_nll(cb, D, q_ids); full = query_nll(cp, D, q_ids)
            kp, vp = perturbation_profile(cb, cp); kP.append(kp); vP.append(vp)
            eff = []
            for L in range(nL):
                h = patch_layer(cb, cp, L)
                eff.append(query_nll(h, D, q_ids, copy=False) - base); del h; torch.cuda.empty_cache()
            effP.append(eff); fulls.append(full - base)
            del cb, cp; torch.cuda.empty_cache()
        kP, vP, effP = np.array(kP), np.array(vP), np.array(effP)
        absE = np.abs(effP).mean(0)  # robust localizer (no cross-doc cancellation)
        out[mk] = {"n_layers": nL, "k_pert": kP.mean(0).tolist(), "v_pert": vP.mean(0).tolist(),
                   "patch_effect": effP.mean(0).tolist(), "patch_abs_effect": absE.tolist(),
                   "full_effect_mean": float(np.mean(fulls)),
                   "full_abs_effect_mean": float(np.mean(np.abs(fulls))),
                   "patch_sum_mean": float(effP.mean(0).sum())}
        e = effP.mean(0); ks = kP.mean(0)
        print(f"  layers={nL}  full|Δnll|={np.mean(np.abs(fulls)):.3f}  (signed {np.mean(fulls):+.3f})")
        print(f"  top-5 causal layers (mean|effect|): {sorted(range(nL), key=lambda L: -absE[L])[:5]}")
        print(f"  k_pert by layer-decile:     " + " ".join(f"{ks[int(q*(nL-1))]:.2f}" for q in np.linspace(0,1,6)))
        print(f"  |patch_effect| by decile:   " + " ".join(f"{absE[int(q*(nL-1))]:.3f}" for q in np.linspace(0,1,6)))
        del m; gc.collect(); torch.cuda.empty_cache(); _M.clear()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print("\n=== CIRCUIT COMPARISON (where does primability live?) ===")
    for mk, r in out.items():
        a = np.array(r["patch_abs_effect"]); nL = r["n_layers"]; tot = a.sum() or 1
        fe, fm, fl = a[:nL//3].sum()/tot, a[nL//3:2*nL//3].sum()/tot, a[2*nL//3:].sum()/tot
        kp = np.array(r["k_pert"]); vp = np.array(r["v_pert"])
        print(f"  {mk:12s} full|Δnll|={r['full_abs_effect_mean']:.3f}  "
              f"|patch| thirds early/mid/late = {fe:.2f}/{fm:.2f}/{fl:.2f}  "
              f"peak_k_pert@L{int(kp.argmax())} peak|effect|@L{int(a.argmax())}")


if __name__ == "__main__":
    main()
