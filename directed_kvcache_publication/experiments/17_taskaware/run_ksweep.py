#!/usr/bin/env python3
"""exp37: k-budget sweep — where does SnapKV-style selection stop hurting answer correctness?
Per item: generate the answer from (a) bare full-doc cache, and (b) the top-k selected cache for
each k in KLIST (same attention importance ranking, sliced), scoring verbosity-robust answer-recall.
Shows the crossover where pruning recovers the bare accuracy. Lean: bare computed once per item.
SQuAD, N=200. Reuses the exp35/36 RoPE/normalize/generation machinery."""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))
import json, gc, shutil, time, re, string
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
N = 8 if SMOKE else 200
KLIST = [int(x) for x in os.environ.get("KLIST", "8,16,32,64,128,256").split(",")]
MAXNEW = 24
DATASET = os.environ.get("DATASET", "squad")
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / f"exp37_ksweep_{DATASET}"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


def _norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
def contains(pred, golds): return max(int(len(_norm(g)) > 0 and _norm(g) in _norm(pred)) for g in golds)
_MARK = ["Human:", "Assistant:", "User:", "\nHuman", "\nAssistant", "Question:", "\n\n", "\n"]
def _clean(t):
    cut = len(t)
    for mk in _MARK:
        j = t.find(mk)
        if j != -1: cut = min(cut, j)
    return t[:cut].strip()


def load_data(n):
    if DATASET == "squad":
        ds = load_dataset("rajpurkar/squad", split="validation"); out = []
        for x in ds:
            g = [t for t in x["answers"]["text"] if t]
            if not g or not (30 <= count_words(x["context"]) <= 160) or g[0].lower() not in x["context"].lower(): continue
            out.append({"q": x["question"], "golds": g, "ctx": x["context"]})
            if len(out) >= n: break
        return out
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); out = []
    for x in ds:
        ans = x["answer"].strip()
        if ans.lower() in ("yes", "no"): continue
        gt = set(x["supporting_facts"]["title"])
        paras = [" ".join(s) for t, s in zip(x["context"]["title"], x["context"]["sentences"]) if t in gt]
        if len(paras) != 2: continue
        ctx = " ".join(paras)
        if not (60 <= count_words(ctx) <= 300) or ans.lower() not in ctx.lower(): continue
        out.append({"q": x["question"], "golds": [ans], "ctx": ctx})
        if len(out) >= n: break
    return out


def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN,
                                             device_map="cuda:0", attn_implementation="eager").eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


_M = {}
def probe_ranking(P_ids, Q_ids):
    """Full doc-relative index ranking by question->doc attention (descending importance)."""
    m, dev, bos, nl, lt = _M["m"], _M["dev"], _M["bos"], _M["nl"], _M["lt"]
    D = len(P_ids); ids = [bos] + list(P_ids) + nl + list(Q_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), output_attentions=True, use_cache=False)
    fl = [i for i, t in enumerate(lt) if t == "full_attention"] or list(range(len(out.attentions)))
    qpos = list(range(1 + D + len(nl), len(ids)))
    imp = torch.zeros(D, device=dev)
    for li in fl:
        imp += out.attentions[li][0][:, qpos, 1:1+D].mean(dim=(0, 1)).float()
    return torch.argsort(imp, descending=True).tolist()


def build(mode, P_ids, sel_rel=None):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    ids = [bos] + list(P_ids); D = len(P_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "bare":
        return cache, len(ids)
    kept = sorted(sel_rel)
    old = [1 + j for j in kept]
    cache = select_kv_cache(cache, [0] + old, device=dev)
    cache = reposition_kv_cache(cache, torch.tensor(old, device=dev),
                                torch.tensor(list(range(1, 1+len(kept))), device=dev), inv, lt, bos_start=0)
    return norm_roundtrip_kv_cache(cache), 1 + len(kept)


def generate(cache, clen, q_ids):
    m, dev, tok = _M["m"], _M["dev"], _M["tok"]
    stop = set(_M["stop_ids"]); lead = list(_M["lead_pre"]) + list(q_ids) + list(_M["lead_post"])
    c = deep_copy_cache(cache)
    pos = torch.arange(clen, clen + len(lead), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([lead], device=dev), position_ids=pos, past_key_values=c, use_cache=True)
    c = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); gen = []; cp = clen + len(lead)
    for _ in range(MAXNEW):
        if nxt in stop: break
        gen.append(nxt)
        with torch.no_grad():
            out = m(input_ids=torch.tensor([[nxt]], device=dev), position_ids=torch.tensor([[cp]], device=dev),
                    past_key_values=c, use_cache=True)
        c = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); cp += 1
    return _clean(tok.decode(gen, skip_special_tokens=True))


def main():
    print(f"K-SWEEP  DATASET={DATASET}  SMOKE={SMOKE}  N={N}  KLIST={KLIST}")
    data = load_data(N); print(f"  {len(data)} items")
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        nl = tok.encode("\n", add_special_tokens=False)
        stop_ids = [tok.eos_token_id]
        INSTR = "Answer the question with a short span copied from the passage, and nothing else."
        _M.update(m=m, tok=tok, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=nl, stop_ids=stop_ids,
                  lead_pre=tok.encode("\n" + INSTR + "\nQuestion: ", add_special_tokens=False),
                  lead_post=tok.encode("\nAnswer:", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-96-len(nl)) if _M["slim"] is not None else 700
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"ksweep_{DATASET}_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), len(data)):
            d = data[i]
            Q = tok.encode(d["q"], add_special_tokens=False)
            P = tok.encode(d["ctx"], add_special_tokens=False)[:max_doc]
            rank = probe_ranking(P, Q)
            rec = {"D": len(P)}
            c, cl = build("bare", P); rec["bare__contains"] = contains(generate(c, cl, Q), d["golds"]); del c
            for k in KLIST:
                sel = rank[:min(k, len(P))]
                c, cl = build("sel", P, sel); rec[f"k{k}__contains"] = contains(generate(c, cl, Q), d["golds"]); del c
                torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = lambda key: 100*np.mean([s[key] for s in scored])
                print(f"    [{i+1}/{len(data)}] bare={A('bare__contains'):.1f} | " +
                      " ".join(f"k{k}={A(f'k{k}__contains'):.1f}" for k in KLIST))
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
