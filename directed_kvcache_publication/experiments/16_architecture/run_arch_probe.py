#!/usr/bin/env python3
"""exp30 (Phase 1): what INTRINSIC computational property predicts imprintability?

Imprintability (|Δ query-NLL from a generic prefix|, r=0.94 with semantic banking) is a model
trait whose architectural root resisted 4 ablations (QK-norm, sharpness, prefix-salience,
fixed-direction). Here we measure intrinsic properties of a plain document encoding (NO priming)
and correlate each with imprintability across 8 models. The hypothesis: high-imprintability
models CONTEXTUALIZE tokens more -- the residual stream is rewritten more by (cross-token)
attention, so a prepended prefix leaves a bigger imprint.

Measured per model on plain [BOS, doc] encodings (output_hidden_states + output_attentions):
  attn_entropy     mean entropy of doc-token attention distributions (diffuse = more mixing)
  ctx_update       mean ||h_l - h_{l-1}|| / ||h_{l-1}||  over layers (residual rewrite rate)
  repr_drift       mean ||h_last - h_embed|| / ||h_embed||  (total departure from input embedding)
  attn_frac        mean ||attn_sublayer_out|| / ||h_after_attn||  (attention's share of the stream)
Outputs result.json; analyze with analyze_arch.py.
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
from lib.data import count_words

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
SMOKE = os.environ.get("SMOKE", "0") == "1"
N = 3 if SMOKE else 12
DOC_TOK = 48
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp30_arch"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "qwen25_14b":  {"name": "Qwen/Qwen2.5-14B-Instruct",  "loader": "AutoModelForCausalLM"},
    "mistral_7b":  {"name": "mistralai/Mistral-7B-Instruct-v0.3", "loader": "AutoModelForCausalLM"},
    "gemma3_1b":   {"name": "google/gemma-3-1b-it", "loader": "Gemma3ForCausalLM"},
    "gemma3_4b":   {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_27b":  {"name": "google/gemma-3-27b-it", "loader": "Gemma3ForConditionalGeneration"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


def load_docs(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, out = set(), []
    for x in ds:
        c = x["context"]
        if c in seen: continue
        seen.add(c)
        if count_words(c) >= 40: out.append(c)
        if len(out) >= n: break
    return out


def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    kw = dict(dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0", attn_implementation="eager")
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration as L
    elif loader == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM as L
    else:
        L = AutoModelForCausalLM
    return L.from_pretrained(name, **kw).eval(), tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


def find_layers(m):
    for path in ["model.layers", "model.language_model.layers", "language_model.model.layers"]:
        obj = m
        try:
            for a in path.split("."): obj = getattr(obj, a)
            return list(obj)
        except AttributeError:
            continue
    return None


def main():
    print(f"ARCH PROBE  SMOKE={SMOKE}  N={N}")
    docs = load_docs(N)
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
        layers = find_layers(m)
        # hooks: capture self_attn sublayer output norm per layer
        attn_out_norm = {}
        handles = []
        if layers is not None:
            for li, layer in enumerate(layers):
                sa = getattr(layer, "self_attn", None)
                if sa is None: continue
                def mk_hook(idx):
                    def hook(mod, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        attn_out_norm[idx] = o.detach().float().norm(dim=-1)  # [B,T]
                    return hook
                handles.append(sa.register_forward_hook(mk_hook(li)))
        print(f"  loaded {time.time()-t0:.0f}s  ({len(layers) if layers else '?'} layers, {len(handles)} attn hooks)")
        AE, CU, RD, AF = [], [], [], []
        for d in docs:
            ids = [bos] + tok.encode(d, add_special_tokens=False)[:DOC_TOK]
            attn_out_norm.clear()
            with torch.no_grad():
                out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, output_attentions=True)
            hs = out.hidden_states  # tuple [L+1] of [1,T,H]
            T = len(ids); ds = 1  # doc tokens from index 1
            # attn entropy (doc query positions, all layers/heads)
            ents = []
            for att in out.attentions:
                a = att[0].float().clamp_min(1e-9)[:, ds:, :]  # H, ndoc, T
                ents.append((-(a*a.log()).sum(-1)).mean().item())
            AE.append(np.mean(ents))
            # ctx_update: per-layer relative residual change at doc tokens
            ups = []
            for l in range(1, len(hs)):
                a = hs[l][0, ds:].float(); b = hs[l-1][0, ds:].float()
                ups.append(((a-b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean().item())
            CU.append(np.mean(ups))
            # repr drift: last vs embedding (layer 0) at doc tokens
            h0 = hs[0][0, ds:].float(); hL = hs[-1][0, ds:].float()
            RD.append(((hL-h0).norm(dim=-1)/h0.norm(dim=-1).clamp_min(1e-6)).mean().item())
            # attn fraction: ||attn_out|| / ||h_after_attn||  (approx via captured norms vs hidden)
            if attn_out_norm:
                fr = []
                for li in attn_out_norm:
                    an = attn_out_norm[li][0, ds:]
                    hn = hs[min(li+1, len(hs)-1)][0, ds:].float().norm(dim=-1)
                    fr.append((an / hn.clamp_min(1e-6)).mean().item())
                AF.append(np.mean(fr))
            del out; torch.cuda.empty_cache()
        for h in handles: h.remove()
        res = {"attn_entropy": float(np.mean(AE)), "ctx_update": float(np.mean(CU)),
               "repr_drift": float(np.mean(RD)), "attn_frac": float(np.mean(AF)) if AF else None}
        (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        (RESULTS / mk / "result.json").write_text(json.dumps(res, indent=2))
        print(f"  attn_entropy={res['attn_entropy']:.3f}  ctx_update={res['ctx_update']:.3f}  "
              f"repr_drift={res['repr_drift']:.3f}  attn_frac={res['attn_frac']}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"])
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
