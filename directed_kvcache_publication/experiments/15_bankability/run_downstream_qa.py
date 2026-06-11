#!/usr/bin/env python3
"""exp29: DOWNSTREAM VALUE — does semantic imprintability help REAL QA? Prime the answer
passage with the QUESTION (semantic context), strip it, measure answer NLL. If imprintable
Gemma banks the question's semantics into the passage's KV, answering improves even though
the question is also present in Phase B for all conditions.

  bare      [BOS, P]            ; [\n Q \n A]   NLL(A)   (passage alone; Q only in phase B)
  q_primed  [BOS, Q, \n, P]->strip Q ; [\n Q \n A] NLL(A) (P encoded WITH Q, Q banked then removed)
  q_retain  [BOS, Q, \n, P]     ; [\n A]         NLL(A)   (Q retained -- upper bound)

q_primed - bare = the bankable value of question-conditioning the passage. Prediction:
negative (helps) on imprintable models (gemma12b), ~0 on qwen7b. SQuAD, N=300.
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
N = 8 if SMOKE else 300
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp29_downstream_qa"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_4b":  {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


def load_squad(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    out = []
    for x in ds:
        ans = x["answers"]["text"][0] if x["answers"]["text"] else None
        if not ans: continue
        ctx = x["context"]
        if not (30 <= count_words(ctx) <= 160): continue
        if ans.lower() not in ctx.lower(): continue
        out.append({"q": x["question"], "a": ans, "ctx": ctx})
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
def build(mode, P_ids, Q_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare": ids = [bos] + list(P_ids); clen = len(ids)
    elif mode == "retain": ids = [bos] + list(Q_ids) + nl + list(P_ids); clen = len(ids)
    elif mode == "primed": ids = [bos] + list(Q_ids) + nl + list(P_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "primed":
        D = len(P_ids); ds = 1 + len(Q_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(ds, ds + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(ds, ds + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
        cache = norm_roundtrip_kv_cache(cache); clen = 1 + D
    return cache, clen

def answer_nll(cache, clen, q_ids, a_ids, ask=True):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = (nl + list(q_ids) + nl + list(a_ids)) if ask else (nl + list(a_ids))
    pos = torch.arange(clen, clen + len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(seq) - len(a_ids); al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"DOWNSTREAM QA (question-priming)  SMOKE={SMOKE}  N={N}")
    data = load_squad(N); print(f"  {len(data)} squad items")
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-96-len(_M["nl"])) if _M["slim"] is not None else 700
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"dqa2_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), len(data)):
            d = data[i]
            Q = tok.encode(d["q"], add_special_tokens=False)
            A = tok.encode(" " + d["a"], add_special_tokens=False)
            P = tok.encode(d["ctx"], add_special_tokens=False)[:max_doc]
            neutral = (_M["nl"] * len(Q))[:len(Q)]  # length-matched content-free prime
            rec = {}
            c, cl = build("bare", P, None);      rec["bare"]      = answer_nll(c, cl, Q, A); del c
            c, cl = build("primed", P, neutral); rec["q_neutral"] = answer_nll(c, cl, Q, A); del c  # machinery only
            c, cl = build("primed", P, Q);       rec["q_primed"]  = answer_nll(c, cl, Q, A); del c
            torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                Aa = {k: np.mean([s[k] for s in scored]) for k in rec}
                print(f"    [{i+1}/{len(data)}] bare={Aa['bare']:.3f} q_neutral={Aa['q_neutral']:.3f} q_primed={Aa['q_primed']:.3f} "
                      f"| MACH={Aa['q_neutral']-Aa['bare']:+.3f} CONTENT={Aa['q_primed']-Aa['q_neutral']:+.3f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
