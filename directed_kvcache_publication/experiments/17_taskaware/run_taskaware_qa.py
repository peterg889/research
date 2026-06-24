#!/usr/bin/env python3
"""exp31a: TASK-AWARE PRIME SELECTION for extraction (QA).

Turns the descriptive mode-task finding into a PRESCRIPTIVE rule: if we know the task is
extraction, which prime CONTENT should we bank into the passage? We hold the question's
TOKEN SET fixed and vary only order/meaning, isolating the semantic vs surface axis:

  bare      [BOS, P]                 ; [\n Q \n A]   NLL(A)      passage alone (RAG baseline)
  neutral   [BOS, <nl*|Q|>, \n, P]   -> strip ; ...  NLL(A)      machinery-only control
  sem_prime [BOS, Q, \n, P]          -> strip ; ...  NLL(A)      ORDERED question (meaning)
  surf_prime[BOS, shuffle(Q), \n, P] -> strip ; ...  NLL(A)      SAME tokens, scrambled (surface)

Machinery-controlled content effects (pos = HURTS answer):
  sem_content  = sem_prime  - neutral
  surf_content = surf_prime - neutral
  order_value  = sem_prime  - surf_prime   (the word-order/MEANING contribution, token set fixed)

Predictions (mode-task match):
 - Gemma (semantic imprinter): banks meaning -> sem_prime blurs the answer (sem_content > 0, hurts);
   surf_prime has less meaning to bank -> closer to neutral. order_value > 0 (ordered hurts more).
 - Qwen (surface imprinter): banks tokens -> both primes help locate the answer (content < 0);
   order_value ~ 0 (order doesn't matter; token presence does).
Actionable rule: to help extraction, prefer a SURFACE prime (token presence) -- helps surface
imprinters and hurts semantic imprinters less than a meaningful prime.

SQuAD, N=300. Deterministic shuffle (seeded per item). Same RoPE/normalize machinery as exp29.
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time, random
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
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp31_taskaware_qa"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_4b":  {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_14b": {"name": "Qwen/Qwen2.5-14B-Instruct", "loader": "AutoModelForCausalLM"},
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
def build(mode, P_ids, prime_ids):
    """mode bare: passage only. mode primed: [BOS, prime, nl, P] -> strip prime, keep BOS+P."""
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare":
        ids = [bos] + list(P_ids); clen = len(ids)
    else:
        ids = [bos] + list(prime_ids) + nl + list(P_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "primed":
        D = len(P_ids); ds = 1 + len(prime_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(ds, ds + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(ds, ds + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
        cache = norm_roundtrip_kv_cache(cache); clen = 1 + D
    return cache, clen

def answer_nll(cache, clen, q_ids, a_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids) + nl + list(a_ids)
    pos = torch.arange(clen, clen + len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(seq) - len(a_ids); al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"TASK-AWARE QA (sem vs surf prime)  SMOKE={SMOKE}  N={N}")
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
        skey = f"tqa_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), len(data)):
            d = data[i]
            Q = tok.encode(d["q"], add_special_tokens=False)
            A = tok.encode(" " + d["a"], add_special_tokens=False)
            P = tok.encode(d["ctx"], add_special_tokens=False)[:max_doc]
            # surface prime = SAME token multiset as Q, deterministically shuffled (destroys order/meaning)
            Qs = list(Q); random.Random(1234 + i).shuffle(Qs)
            neutral = (_M["nl"] * len(Q))[:len(Q)]  # length-matched content-free prime
            rec = {}
            c, cl = build("bare",   P, None);    rec["bare"]       = answer_nll(c, cl, Q, A); del c
            c, cl = build("primed", P, neutral); rec["neutral"]    = answer_nll(c, cl, Q, A); del c
            c, cl = build("primed", P, Q);       rec["sem_prime"]  = answer_nll(c, cl, Q, A); del c
            c, cl = build("primed", P, Qs);      rec["surf_prime"] = answer_nll(c, cl, Q, A); del c
            torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                Aa = {k: np.mean([s[k] for s in scored]) for k in rec}
                print(f"    [{i+1}/{len(data)}] bare={Aa['bare']:.3f} neut={Aa['neutral']:.3f} "
                      f"sem={Aa['sem_prime']:.3f} surf={Aa['surf_prime']:.3f} | "
                      f"semC={Aa['sem_prime']-Aa['neutral']:+.3f} surfC={Aa['surf_prime']-Aa['neutral']:+.3f} "
                      f"order={Aa['sem_prime']-Aa['surf_prime']:+.3f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
