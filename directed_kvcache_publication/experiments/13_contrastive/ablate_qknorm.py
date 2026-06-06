#!/usr/bin/env python3
"""WS3 causal test: is Gemma's QK-norm the CAUSE of its high primability?

Monkeypatch a Gemma 3 forward to replace the per-head q_norm/k_norm RMSNorms with
identities, then re-measure primability = mean |Δquery-NLL from generic priming| over
MS MARCO (query, passage) pairs. Compare intact vs ablated on gemma3_4b (primable 0.597).

Confound guard: removing a trained-in normalization degrades the model, which could
lower primability for the wrong reason. So we also report BARE query-NLL (model health).
Causal reading requires primability to collapse toward the no-QK-norm (Qwen/Mistral ~0.2-0.4)
range WITHOUT bare-NLL exploding. We also report a representation-level primability
(mean L2 change of doc-token last-hidden-state from the prefix), which does not depend
on output quality, as a robustness check.
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
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
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
N_Q = 5 if SMOKE else 60          # queries; x10 passages = pairs
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
MODEL = os.environ.get("ABLATE_MODEL", "google/gemma-3-4b-it")
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp17_ablate_qknorm"
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


def count_qknorm(model):
    return sum(1 for n, _ in model.named_modules() if n.endswith("q_norm") or n.endswith("k_norm"))

def disable_qknorm(model):
    """Replace every *.q_norm / *.k_norm submodule with nn.Identity()."""
    patched = 0
    for name, mod in list(model.named_modules()):
        for attr in ("q_norm", "k_norm"):
            if hasattr(mod, attr) and mod.__class__.__name__.endswith("Attention"):
                setattr(mod, attr, nn.Identity()); patched += 1
    return patched


_M = {}
def encode_passage(doc_ids, prefix_ids, want_hidden=False):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    ids = [bos] + (list(prefix_ids) + nl if prefix_ids is not None else []) + list(doc_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True, output_hidden_states=want_hidden)
    cache = out.past_key_values; D = len(doc_ids)
    doc_start = (1 + len(prefix_ids) + len(nl)) if prefix_ids is not None else 1
    hid = None
    if want_hidden:
        hid = out.hidden_states[-1][0, doc_start:doc_start + D].float().cpu()  # doc-token last hidden
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
    """Return (primability, bare_nll_mean, repr_primability) over (query,passage) pairs."""
    tok = _M["tok"]
    dnll, bare_nlls, repr_d = [], [], []
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
            dnll.append(abs(ng - nb)); bare_nlls.append(nb)
            # representation primability: mean L2 of per-token last-hidden change (normalized by bare norm)
            d = (hg - hb).norm(dim=-1); base = hb.norm(dim=-1).clamp_min(1e-6)
            repr_d.append(float((d / base).mean()))
    return float(np.mean(dnll)), float(np.mean(bare_nlls)), float(np.mean(repr_d))


def main():
    print(f"QK-NORM ABLATION  model={MODEL}  SMOKE={SMOKE}  N_Q={N_Q}")
    samples = load_msmarco(N_Q)
    from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    Loader = Gemma3ForCausalLM if "1b" in MODEL else Gemma3ForConditionalGeneration
    out = {}
    for tag in ["intact", "ablated"]:
        t0 = time.time()
        m = Loader.from_pretrained(MODEL, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
        dev = next(m.parameters()).device
        nqk = count_qknorm(m)
        if tag == "ablated":
            patched = disable_qknorm(m)
            print(f"  [{tag}] patched {patched} q_norm/k_norm -> Identity (model had {nqk})")
            assert patched > 0, "no QK-norm modules found to patch!"
        else:
            print(f"  [{tag}] model has {nqk} q_norm/k_norm modules (kept)")
        _M.update(m=m, tok=tok, dev=dev, inv=build_layer_inv_freqs(m, device=dev),
                  lt=get_layer_types(m), slim=get_sliding_cache_limit(m),
                  nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        prim, bare, rprim = measure(samples, max_doc, ext)
        out[tag] = {"primability": prim, "bare_nll": bare, "repr_primability": rprim}
        print(f"  [{tag}] loaded {time.time()-t0:.0f}s | primability(|Δnll|)={prim:.3f}  "
              f"bare_nll={bare:.3f}  repr_primability={rprim:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache(); _M.clear()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print("\n=== QK-NORM CAUSAL TEST ===")
    pi, pa = out["intact"]["primability"], out["ablated"]["primability"]
    bi, ba = out["intact"]["bare_nll"], out["ablated"]["bare_nll"]
    ri, ra = out["intact"]["repr_primability"], out["ablated"]["repr_primability"]
    print(f"  primability:       intact={pi:.3f}  ablated={pa:.3f}  ({(pa-pi)/pi*100:+.0f}%)")
    print(f"  repr_primability:  intact={ri:.3f}  ablated={ra:.3f}  ({(ra-ri)/ri*100:+.0f}%)")
    print(f"  bare_nll (health): intact={bi:.3f}  ablated={ba:.3f}  ({ba-bi:+.3f} nats; large => model broke)")
    print(f"  Gemma-family primability baseline ~0.58; Qwen/Mistral (no QK-norm) ~0.38.")
    print(f"  Causal if primability drops toward ~0.38 WITHOUT bare_nll exploding.")


if __name__ == "__main__":
    main()
