#!/usr/bin/env python3
"""WS3 causal capstone: is the directional cache shift SUFFICIENT to reproduce priming?

If priming's effect = adding a coherent direction to the doc cache, then EXTRACTING that
direction and ADDING it to a bare cache (no prefix) should REPRODUCE the priming effect.
Train/test split (non-circular): extract the per-layer mean value-perturbation from TRAIN
docs (generic prefix), add it to BARE caches of HELD-OUT TEST docs, score the matched query.

reproduction = (steered_nll - bare_nll) / (primed_nll - bare_nll), per test doc.
~1.0 => the extracted direction fully reproduces priming (mechanism is sufficient).
Value channel only (no RoPE -> frame-free). gemma3_4b vs qwen25_7b.
Prediction: Gemma reproduction >> Qwen (content-structured, steerable vs orthogonal).
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
N = 8 if SMOKE else 60          # split half train / half test
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp22_steering"
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

def value_perturbations(cb, cp):
    """per layer: mean over doc tokens (pos>=1) of (V_primed - V_bare), shape [n_kv, hd]."""
    out = []
    for i in range(len(cb.layers)):
        vb = cb.layers[i].values[0, :, 1:].float(); vp = cp.layers[i].values[0, :, 1:].float()
        out.append((vp - vb).mean(1))   # [n_kv, hd]
    return out

def steer(cb, dirs, scale=1.0):
    """add dirs[L] (broadcast over doc positions) to bare cache values."""
    h = deep_copy_cache(cb)
    for L in range(len(h.layers)):
        d = dirs[L].to(h.layers[L].values.dtype)
        h.layers[L].values[0, :, 1:] += scale * d[:, None, :]
    return h


def main():
    print(f"STEERING SUFFICIENCY  SMOKE={SMOKE}  N={N} (half train/half test)")
    samples = load_msmarco(N); ntr = len(samples)//2
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
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        # TRAIN: extract mean per-layer value-perturbation direction
        acc = None
        for s in samples[:ntr]:
            doc = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
            cb, D = encode_full(doc, None); cp, _ = encode_full(doc, ext)
            vp = value_perturbations(cb, cp)
            acc = vp if acc is None else [a + b for a, b in zip(acc, vp)]
            del cb, cp; torch.cuda.empty_cache()
        dirs = [a / ntr for a in acc]   # mean direction per layer
        # TEST: does adding dirs to bare reproduce priming?
        repro, prim_eff, steer_eff = [], [], []
        for s in samples[ntr:]:
            q = tok.encode(s["query"], add_special_tokens=False)
            doc = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
            cb, D = encode_full(doc, None); cp, _ = encode_full(doc, ext)
            base = query_nll(cb, D, q); prim = query_nll(cp, D, q)
            st = query_nll(steer(cb, dirs), D, q, copy=False)
            if abs(prim - base) > 1e-3:
                repro.append((st - base) / (prim - base))
            prim_eff.append(prim - base); steer_eff.append(st - base)
            del cb, cp; torch.cuda.empty_cache()
        out[mk] = {"reproduction_mean": float(np.mean(repro)), "reproduction_median": float(np.median(repro)),
                   "primed_effect_mean": float(np.mean(prim_eff)), "steered_effect_mean": float(np.mean(steer_eff)),
                   "n_test": len(prim_eff)}
        print(f"  primed Δnll(test)={np.mean(prim_eff):+.3f}  steered Δnll={np.mean(steer_eff):+.3f}  "
              f"reproduction mean={np.mean(repro):.2f} median={np.median(repro):.2f}")
        del m; gc.collect(); torch.cuda.empty_cache(); _M.clear()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print("\n=== STEERING SUFFICIENCY (does the extracted direction reproduce priming?) ===")
    for mk, r in out.items():
        print(f"  {mk:12s} primed Δ={r['primed_effect_mean']:+.3f}  steered Δ={r['steered_effect_mean']:+.3f}  "
              f"reproduction={r['reproduction_mean']:.2f} (median {r['reproduction_median']:.2f})")
    print("  Gemma reproduction >> Qwen => directional shift is SUFFICIENT (causal) on Gemma.")


if __name__ == "__main__":
    main()
