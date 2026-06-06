#!/usr/bin/env python3
"""WS3 deeper: is PRIMABILITY governed by attention SHARPNESS? (the QK-norm ablation
showed sharper/saturated attention -> higher primability). Rather than destructively
remove trained Gemma features (embedding*sqrt(d), (1+w)RMSNorm -> catastrophic), use a
GENTLE continuous knob: scale attention logits by 1/tau (tau<1 sharper, tau>1 softer),
which removes no trained component. Sweep tau on a Gemma AND a Qwen model.

Causal reading:
 - If primability rises monotonically as attention sharpens (tau down) on BOTH families,
   primability is a function of attention sharpness (universal axis).
 - Cross-family decomposition: does the Gemma>Qwen primability gap close when we match
   attention sharpness, or is Gemma's whole curve shifted up (=> more than sharpness)?
Also reports each model's NATURAL primability (tau=1) and bare_nll (health).
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, time
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
SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_Q = 4 if SMOKE else 30
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
TAUS = [0.5, 1.0] if SMOKE else [0.5, 0.7, 1.0, 1.5, 2.0]
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp18_atttemp"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)


def load_msmarco(n):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1: continue
        rel = sel.index(1)
        if not (5 <= count_words(pt[rel]) <= 300): continue
        out.append({"query": x["query"], "passages": pt, "relevant_idx": rel})
        if len(out) >= n: break
    return out


def attn_modules(model):
    return [m for m in model.modules()
            if m.__class__.__name__.endswith("Attention") and hasattr(m, "scaling")]

_M = {}
def encode_passage(doc_ids, prefix_ids, want_hidden=False):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    ids = [bos] + (list(prefix_ids) + nl if prefix_ids is not None else []) + list(doc_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True, output_hidden_states=want_hidden)
    cache = out.past_key_values; D = len(doc_ids)
    doc_start = (1 + len(prefix_ids) + len(nl)) if prefix_ids is not None else 1
    hid = out.hidden_states[-1][0, doc_start:doc_start + D].float().cpu() if want_hidden else None
    cache = select_kv_cache(cache, [0] + list(range(doc_start, doc_start + D)), device=dev)
    if prefix_ids is not None:
        cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
    return norm_roundtrip_kv_cache(cache), D, hid

def query_nll(cache, D, q_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids); pos = torch.arange(D+1, D+1+len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(nl); al = out.logits[0][a0-1:a0-1+len(q_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(q_ids, device=dev)).item()

def measure(samples, max_doc, ext):
    tok = _M["tok"]; dnll, bare, repr_d = [], [], []
    for s in samples:
        q_ids = tok.encode(s["query"], add_special_tokens=False)
        for ptext in s["passages"]:
            doc_ids = tok.encode(ptext, add_special_tokens=False)[:max_doc]
            if not doc_ids: continue
            cb, D, hb = encode_passage(doc_ids, None, want_hidden=True)
            nb = query_nll(cb, D, q_ids); del cb; torch.cuda.empty_cache()
            cg, Dg, hg = encode_passage(doc_ids, ext, want_hidden=True)
            ng = query_nll(cg, Dg, q_ids); del cg; torch.cuda.empty_cache()
            if not (np.isfinite(nb) and np.isfinite(ng)): continue
            dnll.append(abs(ng - nb)); bare.append(nb)
            d = (hg - hb).norm(dim=-1); base = hb.norm(dim=-1).clamp_min(1e-6)
            repr_d.append(float((d / base).mean()))
    return float(np.mean(dnll)), float(np.mean(bare)), float(np.mean(repr_d))


def main():
    print(f"ATTENTION-TEMPERATURE SWEEP  SMOKE={SMOKE}  N_Q={N_Q}  taus={TAUS}")
    samples = load_msmarco(N_Q)
    all_out = {}
    for mk, spec in MODELS.items():
        print(f"\n{'#'*56}\n# {mk}\n{'#'*56}")
        tok = AutoTokenizer.from_pretrained(spec["name"], token=HF_TOKEN)
        if spec["loader"] == "Gemma3ForConditionalGeneration":
            from transformers import Gemma3ForConditionalGeneration as L
        else:
            L = AutoModelForCausalLM
        m = L.from_pretrained(spec["name"], dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
        dev = next(m.parameters()).device
        mods = attn_modules(m); orig = [mod.scaling for mod in mods]
        print(f"  {len(mods)} attention modules; base scaling={orig[0]:.5f}")
        _M.update(m=m, tok=tok, dev=dev, inv=build_layer_inv_freqs(m, device=dev),
                  lt=get_layer_types(m), slim=get_sliding_cache_limit(m),
                  nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        rows = {}
        for tau in TAUS:
            for mod, o in zip(mods, orig): mod.scaling = o / tau  # sharper if tau<1
            t0 = time.time(); prim, bare, rprim = measure(samples, max_doc, ext)
            rows[tau] = {"primability": prim, "bare_nll": bare, "repr_primability": rprim}
            print(f"    tau={tau:<4} primability={prim:.3f}  repr={rprim:.3f}  bare_nll={bare:.3f}  ({time.time()-t0:.0f}s)")
        for mod, o in zip(mods, orig): mod.scaling = o  # restore
        all_out[mk] = rows
        del m; gc.collect(); torch.cuda.empty_cache(); _M.clear()
    (RESULTS / "result.json").write_text(json.dumps(all_out, indent=2))
    print("\n=== SUMMARY: primability vs attention temperature (tau<1=sharper) ===")
    for mk, rows in all_out.items():
        s = "  ".join(f"t{t}={rows[t]['primability']:.2f}" for t in TAUS)
        print(f"  {mk:12s} {s}   natural(t1.0)={rows[1.0]['primability']:.3f}")
    print("  Sharper(lower tau) -> higher primability on both => primability ~ attention sharpness.")


if __name__ == "__main__":
    main()
