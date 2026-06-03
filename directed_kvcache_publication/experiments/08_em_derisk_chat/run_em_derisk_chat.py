#!/usr/bin/env python3
"""PHASE 0b: chat-correct + multiple-choice de-risk.

Fixes two weaknesses of the first de-risk (exp07):
  1. Floor effect: Qwen produced ~6% bare EM because the raw [doc][Q][A:] format
     isn't how instruct models expect input. Here we use the model's CHAT TEMPLATE
     (apply_chat_template), which lifts the baseline so ΔEM is measurable.
  2. Format noise: free-gen EM is confounded by verbosity. We add a FORMAT-FREE
     multiple-choice metric: score the correct answer + K type-matched distractors
     as the assistant response and pick the argmax (MC top1 accuracy).

Two-phase cache with a multi-token chat prefix:
  cache = [chat_prefix, doc]  (chat_prefix kept fixed as the sink region)
  priming prefix is inserted between chat_prefix and doc during encoding, then
  stripped and the doc keys repositioned (reposition bos_start = C-1, keeping the
  whole C-token chat prefix).
  Phase B = chat tail (question + assistant generation prompt).

Conditions: bare, extract, comprehend (null ctrl), keywords (null ctrl).
Metrics: gen EM / F1 / contains (chat-correct free generation) + MC top1 / margin.
Models: Qwen 7B, Gemma 12B, Ministral 8B (good-baseline / good instruction-followers).
Datasets: squad_v2, hotpotqa, triviaqa. N=300, K=7 distractors.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/08_em_derisk_chat/run_em_derisk_chat.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/08_em_derisk_chat/run_em_derisk_chat.py
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, re, shutil, time, string
import random as pyrandom
from pathlib import Path
from collections import Counter

import torch
import numpy as np
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
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EVAL = 10 if SMOKE else 300
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16
MAX_NEW = 32
PLACE = "<<<DOCPLACEHOLDER>>>"
EXTRACT = "Extract the key facts from this text."
COMPREHEND = "Read and comprehend this text carefully."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp09_em_derisk_chat"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
TFIDF = json.loads((Path(__file__).resolve().parent.parent / "02_ablation" / "generated_prefixes.json").read_text())["tfidf_keywords"]

MODELS = {
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "ministral_8b":{"name": "mistralai/Ministral-8B-Instruct-2410", "loader": "AutoModelForCausalLM"},
}
if SMOKE:
    MODELS = {"qwen25_7b": MODELS["qwen25_7b"]}
DATASETS = ["squad_v2", "hotpotqa", "triviaqa"]
if SMOKE:
    DATASETS = ["squad_v2"]


# ---------------- metrics ----------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def first_line(t):
    for stop in ["\n", "<end_of_turn>", "<|im_end|>", "</s>"]:
        i = t.find(stop)
        if i >= 0:
            t = t[:i]
    return t.strip()

def em(pred, golds):
    p = normalize_answer(first_line(pred)); return int(any(p == normalize_answer(g) for g in golds))
def contains(pred, golds):
    p = normalize_answer(first_line(pred)); return int(any(normalize_answer(g) in p for g in golds if g.strip()))
def f1(pred, golds):
    p = normalize_answer(first_line(pred)).split(); best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        if not p or not gt: continue
        ns = sum((Counter(p) & Counter(gt)).values())
        if ns == 0: continue
        pr, rc = ns/len(p), ns/len(gt); best = max(best, 2*pr*rc/(pr+rc))
    return best


# ---------------- datasets + distractors ----------------
def load_datasets():
    out = {}
    ds = load_dataset("rajpurkar/squad_v2", split="validation"); cand = []
    for it in ds:
        p, q = it.get("context", ""), it.get("question", "")
        ans = it.get("answers", {}).get("text", [])
        if p and q and ans and 30 <= count_words(p) <= 500:
            cand.append({"passage": p, "query": q, "answer": ans[0], "golds": list(dict.fromkeys(ans))})
    pyrandom.seed(SEED+200); pyrandom.shuffle(cand); out["squad_v2"] = cand[:N_EVAL]; del ds,cand; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa","rc.wikipedia",split="validation"); cand=[]
    for it in ds:
        wc=it.get("entity_pages",{}).get("wiki_context",[])
        if not wc or not wc[0]: continue
        p=" ".join(wc[0].split()[:500]); q=it["question"]; v=it["answer"]["value"]; al=it["answer"].get("aliases",[])
        pl=p.lower()
        if not (v.lower() in pl or any(a.lower() in pl for a in al)): continue
        if 30<=count_words(p)<=500 and count_words(v)>=1:
            cand.append({"passage":p,"query":q,"answer":v,"golds":[v]+al})
    pyrandom.seed(SEED+300); pyrandom.shuffle(cand); out["triviaqa"]=cand[:N_EVAL]; del ds,cand; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa","distractor",split="validation"); cand=[]
    for it in ds:
        ctx=it.get("context",{}); sf=it.get("supporting_facts",{})
        t2s={t:s for t,s in zip(ctx.get("title",[]),ctx.get("sentences",[]))}
        parts=[t2s[t][sid] for t,sid in zip(sf.get("title",[]),sf.get("sent_id",[])) if t in t2s and sid<len(t2s[t])]
        if not parts: continue
        p=" ".join(parts)
        if 30<=count_words(p)<=500 and count_words(it["answer"])>=1:
            cand.append({"passage":p,"query":it["question"],"answer":it["answer"],"golds":[it["answer"]]})
    pyrandom.seed(SEED+400); pyrandom.shuffle(cand); out["hotpotqa"]=cand[:N_EVAL]; del ds,cand; gc.collect()
    return out

def make_distractor_picker(samples):
    pool = [s["answer"] for s in samples]
    def _t(a):
        a=a.strip(); isn=bool(a) and (a[0].isdigit() or (a[0]=="-" and len(a)>1 and a[1].isdigit()))
        nt=len(a.split()); return (isn, 0 if nt<=1 else (1 if nt<=3 else 2))
    tix={}
    for j,a in enumerate(pool): tix.setdefault(_t(a),[]).append(j)
    def pick(idx, correct, aliases=None):
        bad={normalize_answer(correct)}
        if aliases: bad|={normalize_answer(a) for a in aliases}
        bucket=tix.get(_t(correct),[]); cands=bucket if len(bucket)>K_DISTRACT*3 else list(range(len(pool)))
        rng=pyrandom.Random(SEED+7000+idx); order=cands[:]; rng.shuffle(order); out=[]
        for j in order:
            if j==idx: continue
            nc=normalize_answer(pool[j])
            if nc in bad or not nc: continue
            out.append(pool[j]); bad.add(nc)
            if len(out)>=K_DISTRACT: break
        return out
    return pick


# ---------------- model + chat pieces ----------------
def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def chat_pieces(tok, query):
    """Return (chat_prefix_ids, tail_ids) by splitting the rendered template on PLACE."""
    msgs = [{"role": "user", "content": PLACE + f"\n\nQuestion: {query}"}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre, tail = rendered.split(PLACE)
    return tok.encode(pre, add_special_tokens=False), tok.encode(tail, add_special_tokens=False)

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


# ---------------- two-phase cache with chat prefix ----------------
def build_cache(model, dev, chat_pre, doc_ids, priming_ids, inv_freqs, lt, nl):
    C = len(chat_pre); D = len(doc_ids)
    if priming_ids is None:
        ids = list(chat_pre) + list(doc_ids)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
        cache = out.past_key_values; del out
        return norm_roundtrip_kv_cache(cache), C, D
    P = len(priming_ids); NL = len(nl)
    ids = list(chat_pre) + list(priming_ids) + nl + list(doc_ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; del out
    doc_start = C + P + NL
    keep = list(range(C)) + list(range(doc_start, doc_start + D))
    cache = select_kv_cache(cache, keep, device=dev)
    # reposition doc from [doc_start..] back to [C..C+D-1], keeping the C-token chat prefix
    cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                torch.arange(C, C + D, device=dev), inv_freqs, lt, bos_start=C - 1)
    return norm_roundtrip_kv_cache(cache), C, D

def generate(model, tok, dev, cache, C, D, tail_ids, max_new=MAX_NEW):
    pos0 = C + D
    cc = deep_copy_cache(cache)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tail_ids], device=dev),
                    position_ids=torch.arange(pos0, pos0 + len(tail_ids), device=dev).unsqueeze(0),
                    past_key_values=cc, use_cache=True)
    cg = out.past_key_values; nxt = out.logits[0, -1:].argmax(-1); gen = [nxt.item()]
    eot = set(tok.encode("\n", add_special_tokens=False))
    for et in ["<end_of_turn>", "<|im_end|>"]:
        try: eot.add(tok.convert_tokens_to_ids(et))
        except Exception: pass
    for _ in range(max_new - 1):
        pos = pos0 + len(tail_ids) + len(gen) - 1
        with torch.no_grad():
            out = model(input_ids=nxt.unsqueeze(0), position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cg, use_cache=True)
        cg = out.past_key_values; nxt = out.logits[0, -1:].argmax(-1); t = nxt.item()
        if t == tok.eos_token_id or t in eot: break
        gen.append(t)
    del cg
    return tok.decode(gen, skip_special_tokens=True)

def mc_scores(model, tok, dev, cache, C, D, tail_ids, cand_texts):
    """Format-free MC: NLL of each candidate as the assistant answer; return list."""
    pos0 = C + D
    nlls = []
    for txt in cand_texts:
        cand_ids = tok.encode(" " + txt.strip(), add_special_tokens=False)
        if not cand_ids: nlls.append(float("inf")); continue
        seq = list(tail_ids) + cand_ids
        pos = torch.arange(pos0, pos0 + len(seq), device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device=dev), position_ids=pos,
                        past_key_values=deep_copy_cache(cache), use_cache=False)
        a0 = len(tail_ids)
        al = out.logits[0][a0 - 1:a0 - 1 + len(cand_ids)]
        nlls.append(torch.nn.functional.cross_entropy(al, torch.tensor(cand_ids, device=dev)).item())
    return nlls


def main():
    print(f"PHASE 0b chat+MC de-risk  SMOKE={SMOKE} N={N_EVAL} K={K_DISTRACT}")
    data = load_datasets()
    for k in DATASETS: print(f"  {k}: {len(data[k])}")
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); model, tok = load_model(spec["name"], spec["loader"]); dev = next(model.parameters()).device
        inv_freqs = build_layer_inv_freqs(model, device=dev); lt = get_layer_types(model)
        slim = get_sliding_cache_limit(model); nl = tok.encode("\n", add_special_tokens=False)
        max_doc = (slim - 1 - 96 - len(nl)) if slim is not None else 760
        ext_ids = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        comp_ids = make_prefix(tok.encode(COMPREHEND, add_special_tokens=False), L_MATCH)
        print(f"  loaded in {time.time()-t0:.0f}s")
        for dk in DATASETS:
            samples = data[dk]; pick = make_distractor_picker(samples)
            ck = RESULTS / mk / f"results_{dk}.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
            skey = f"chatderisk_{mk}" + ("_smoke" if SMOKE else ""); scored = []
            if ck.exists():
                prev = json.loads(ck.read_text())
                if prev.get("scoring_key") == skey: scored = prev["results"]; print(f"  {dk} resumed {len(scored)}")
            for idx in range(len(scored), len(samples)):
                s = samples[idx]
                doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
                chat_pre, tail_ids = chat_pieces(tok, s["query"])
                kw_ids = make_prefix(tok.encode(TFIDF[dk][idx], add_special_tokens=False), L_MATCH) if TFIDF[dk][idx] else None
                distract = pick(idx, s["answer"], s.get("golds"))
                cands = [s["answer"]] + distract  # index 0 = correct
                prefixes = {"bare": None, "extract": ext_ids, "comprehend": comp_ids, "keywords": kw_ids}
                rec = {"golds": s["golds"][:8]}
                for cond, pfx in prefixes.items():
                    if cond != "bare" and pfx is None: continue
                    cache, C, D = build_cache(model, dev, chat_pre, doc_ids, pfx, inv_freqs, lt, nl)
                    gen = generate(model, tok, dev, cache, C, D, tail_ids)
                    rec[f"{cond}__gen"] = gen[:200]
                    rec[f"{cond}__em"] = em(gen, s["golds"]); rec[f"{cond}__f1"] = f1(gen, s["golds"]); rec[f"{cond}__contains"] = contains(gen, s["golds"])
                    nlls = mc_scores(model, tok, dev, cache, C, D, tail_ids, cands)
                    rec[f"{cond}__mc_top1"] = int(np.argmin(nlls) == 0)
                    rec[f"{cond}__mc_margin"] = float(np.mean(nlls[1:]) - nlls[0])
                    del cache; torch.cuda.empty_cache()
                scored.append(rec)
                if (idx + 1) % 20 == 0 or SMOKE:
                    ck.write_text(json.dumps({"scoring_key": skey, "results": scored}))
                    be, ee = np.mean([r["bare__em"] for r in scored]), np.mean([r["extract__em"] for r in scored])
                    bm, em_ = np.mean([r["bare__mc_top1"] for r in scored]), np.mean([r["extract__mc_top1"] for r in scored])
                    print(f"    [{idx+1}/{len(samples)}] EM bare={be:.0%} ext={ee:.0%} | MC bare={bm:.0%} ext={em_:.0%}")
            ck.write_text(json.dumps({"scoring_key": skey, "results": scored})); print(f"  {dk}: {len(scored)} done")
        del model, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"])
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
