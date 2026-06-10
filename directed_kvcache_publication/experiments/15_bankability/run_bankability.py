#!/usr/bin/env python3
"""THE BANKABILITY TEST — is context value attendable-token-bound or imprint-bankable?

User's insight: context MUST matter (RAG/ICL work), so if priming shows ~nothing either
we under-measure or there's a bug. Reconciliation hypothesis: context delivers value via
ATTENTION TO ITS TOKENS; cache priming keeps the imprint on the document's KV but discards
the tokens -> near-total loss expected.

Decisive test on HotpotQA (multi-hop: the answer needs a SECOND supporting paragraph P2
that adds info beyond the answer paragraph P1). Score answer NLL (lower=better):
  bare        cache=[BOS, P1]                      ; ask Q -> A     (no extra context)
  p2_retain   cache=[BOS, P2, \n, P1]  (KEEP all)  ; ask Q -> A     (P2 attendable)
  p2_strip    cache=[BOS, P2, \n, P1] -> strip P2, reposition P1    ; ask Q -> A
                                                                     (P2 primed into P1's KV, then removed)
  q_aware     cache=[BOS, Q, \n, P1]   (KEEP Q)    ; -> A            (encode P1 knowing Q; ceiling-ish)

Predictions: p2_retain << bare (P2 helps; PIPELINE SANITY); p2_strip ~ bare (imprint not
bankable). bankable_fraction = (bare - p2_strip) / (bare - p2_retain).
Models: gemma3_4b (primable), qwen25_7b (not). N=200.
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

from lib.rope import select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache
from lib.quantization import norm_roundtrip_kv_cache
from lib.data import count_words
from model_adapters import build_layer_inv_freqs, get_layer_types, get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
SMOKE = os.environ.get("SMOKE", "0") == "1"
N = 6 if SMOKE else 200
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp23_bankability"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


def load_hotpot(n):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    out = []
    for x in ds:
        ans = x["answer"].strip()
        if ans.lower() in ("yes", "no") or len(ans) < 2: continue
        titles = x["context"]["title"]; sents = x["context"]["sentences"]
        paras = {t: " ".join(s) for t, s in zip(titles, sents)}
        sup = list(dict.fromkeys(x["supporting_facts"]["title"]))  # unique supporting titles
        if len(sup) != 2: continue
        if sup[0] not in paras or sup[1] not in paras: continue
        pa, pb = paras[sup[0]], paras[sup[1]]
        # P1 = answer-bearing paragraph; P2 = the other supporting paragraph
        if ans.lower() in pa.lower() and ans.lower() not in pb.lower(): P1, P2 = pa, pb
        elif ans.lower() in pb.lower() and ans.lower() not in pa.lower(): P1, P2 = pb, pa
        else: continue
        if not (5 <= count_words(P1) <= 180 and 5 <= count_words(P2) <= 180): continue
        out.append({"question": x["question"], "answer": ans, "P1": P1, "P2": P2})
        if len(out) >= n: break
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
def build_cache(mode, P1_ids, ctx_ids):
    """mode: 'bare' [BOS,P1]; 'retain' [BOS,ctx,nl,P1] kept; 'strip' [BOS,ctx,nl,P1]->P1 only repositioned."""
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare":
        ids = [bos] + list(P1_ids)
    else:
        ids = [bos] + list(ctx_ids) + nl + list(P1_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "strip":
        D = len(P1_ids); p1_start = 1 + len(ctx_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(p1_start, p1_start + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(p1_start, p1_start + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
        cache = norm_roundtrip_kv_cache(cache)
        clen = 1 + D
    elif mode == "retain":
        clen = len(ids)
    else:
        clen = len(ids)
    return cache, clen

def answer_nll(cache, clen, q_ids, a_ids, ask=True):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = (nl + list(q_ids) + nl + list(a_ids)) if ask else (nl + list(a_ids))
    pos = torch.arange(clen, clen + len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(seq) - len(a_ids)
    al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"BANKABILITY TEST  SMOKE={SMOKE}  N={N}")
    data = load_hotpot(N); print(f"  {len(data)} multi-hop items")
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"bank_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  resumed {len(scored)}")
        for i in range(len(scored), len(data)):
            d = data[i]
            q = tok.encode(d["question"], add_special_tokens=False)
            a = tok.encode(" " + d["answer"], add_special_tokens=False)
            P1 = tok.encode(d["P1"], add_special_tokens=False)[:200]
            P2 = tok.encode(d["P2"], add_special_tokens=False)[:200]
            rec = {}
            cb, lb = build_cache("bare", P1, None);  rec["bare"] = answer_nll(cb, lb, q, a); del cb
            cr, lr = build_cache("retain", P1, P2);  rec["p2_retain"] = answer_nll(cr, lr, q, a); del cr
            cs, ls = build_cache("strip", P1, P2);   rec["p2_strip"] = answer_nll(cs, ls, q, a); del cs
            cq, lq = build_cache("retain", P1, q);   rec["q_aware"] = answer_nll(cq, lq, q, a, ask=False); del cq
            torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = {k: np.mean([s[k] for s in scored]) for k in ["bare","p2_retain","p2_strip","q_aware"]}
                bf = (A["bare"]-A["p2_strip"])/(A["bare"]-A["p2_retain"]+1e-9)
                print(f"    [{i+1}/{len(data)}] NLL bare={A['bare']:.3f} p2_retain={A['p2_retain']:.3f} "
                      f"p2_strip={A['p2_strip']:.3f} q_aware={A['q_aware']:.3f} | bankable_frac={bf:.2f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
