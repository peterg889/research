#!/usr/bin/env python3
"""Mechanism exploitation: task-aligned priming + serve-time selective validation.

The mechanism is document-CONTENT amplification. Exploit it by priming toward the
content TYPE the task's answer needs, instead of generic "extract key facts":
  generic     "Extract the key facts from this text."
  entity      "List the people, places, dates, numbers, and named entities ..."
  qa_directed "Note who did what, when, where, and how many, in this text."

Predictions:
  (1) On extractive/factoid QA, entity/qa_directed margin >= generic (amplifying the
      RIGHT content type helps more).
  (2) Gold-class behaviour stays real-sharpening for content answers.
  (3) A SERVE-TIME confidence proxy (entropy of the bare next-token distribution at
      the answer position, no distractors needed) predicts the priming benefit ->
      validates the selective-deployment recipe with a deployable signal.

Setup mirrors exp05 (BOS + prefix + doc, select, reposition, norm; score correct +
K=7 type-matched distractors). Stores per sample: bare serve-time entropy, and per
condition: nll_correct, individual distractor NLLs (for margin/top1/rescue/break).

Models: Gemma 12B, Qwen 7B, Qwen 1.5B (the responsive set). Datasets: squad_v2,
hotpotqa, triviaqa, gsm8k. N=300, K=7.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/11_task_aligned/run_task_aligned.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/11_task_aligned/run_task_aligned.py
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time, string
import random as pyrandom
from pathlib import Path

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
N_EVAL = 8 if SMOKE else 300
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp11_task_aligned"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)

INSTRUCTIONS = {
    "generic":     "Extract the key facts from this text.",
    "entity":      "List the people, places, dates, numbers, and named entities in this text.",
    "qa_directed": "Note who did what, when, where, and how many, in this text.",
}
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}
DATASETS = ["squad_v2", "hotpotqa", "triviaqa", "gsm8k"]
if SMOKE:
    DATASETS = ["squad_v2"]


def normalize_answer(s):
    s = s.lower(); s = "".join(c for c in s if c not in string.punctuation)
    import re as _re; s = _re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def load_datasets():
    out = {}
    ds = load_dataset("rajpurkar/squad_v2", split="validation"); C = []
    for it in ds:
        p, q = it.get("context", ""), it.get("question", ""); a = it.get("answers", {}).get("text", [])
        if p and q and a and 30 <= count_words(p) <= 500:
            C.append({"passage": p, "query": q, "answer": a[0]})
    pyrandom.seed(SEED+200); pyrandom.shuffle(C); out["squad_v2"] = C[:N_EVAL]; del ds, C; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation"); C = []
    for it in ds:
        wc = it.get("entity_pages", {}).get("wiki_context", [])
        if not wc or not wc[0]: continue
        p = " ".join(wc[0].split()[:500]); v = it["answer"]["value"]; al = it["answer"].get("aliases", []); pl = p.lower()
        if not (v.lower() in pl or any(x.lower() in pl for x in al)): continue
        if 30 <= count_words(p) <= 500 and count_words(v) >= 1:
            C.append({"passage": p, "query": it["question"], "answer": v, "aliases": al})
    pyrandom.seed(SEED+300); pyrandom.shuffle(C); out["triviaqa"] = C[:N_EVAL]; del ds, C; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); C = []
    for it in ds:
        ctx = it.get("context", {}); sf = it.get("supporting_facts", {})
        t2s = {t: s for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))}
        parts = [t2s[t][sid] for t, sid in zip(sf.get("title", []), sf.get("sent_id", [])) if t in t2s and sid < len(t2s[t])]
        if not parts: continue
        p = " ".join(parts)
        if 30 <= count_words(p) <= 500 and count_words(it["answer"]) >= 1:
            C.append({"passage": p, "query": it["question"], "answer": it["answer"]})
    pyrandom.seed(SEED+400); pyrandom.shuffle(C); out["hotpotqa"] = C[:N_EVAL]; del ds, C; gc.collect()

    ds = load_dataset("openai/gsm8k", "main", split="test"); C = []
    for it in ds:
        if "####" not in it["answer"]: continue
        a = it["answer"].split("####")[-1].strip()
        if a and 10 <= count_words(it["question"]) <= 500:
            C.append({"passage": it["question"], "query": "What is the answer?", "answer": a})
    pyrandom.seed(SEED+600); pyrandom.shuffle(C); out["gsm8k"] = C[:N_EVAL]; del ds, C; gc.collect()
    return out


def make_picker(samples):
    pool = [s["answer"] for s in samples]
    def _t(a):
        a = a.strip(); isn = bool(a) and (a[0].isdigit() or (a[0] == "-" and len(a) > 1 and a[1].isdigit()))
        nt = len(a.split()); return (isn, 0 if nt <= 1 else (1 if nt <= 3 else 2))
    tix = {}
    for j, a in enumerate(pool): tix.setdefault(_t(a), []).append(j)
    def pick(idx, correct, aliases=None):
        bad = {normalize_answer(correct)}
        if aliases: bad |= {normalize_answer(a) for a in aliases}
        b = tix.get(_t(correct), []); cands = b if len(b) > K_DISTRACT*3 else list(range(len(pool)))
        rng = pyrandom.Random(SEED+7000+idx); order = cands[:]; rng.shuffle(order); out = []
        for j in order:
            if j == idx: continue
            nc = normalize_answer(pool[j])
            if nc in bad or not nc: continue
            out.append(pool[j]); bad.add(nc)
            if len(out) >= K_DISTRACT: break
        return out
    return pick


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
def encode_phase_a(doc_ids, prefix_ids):
    m, dev, inv, lt, slim, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["slim"], _M["bos"], _M["nl"]
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

def cand_nll(cache, D, query_ids, cand_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    pb = nl + list(query_ids) + nl + list(cand_ids); pos = torch.arange(D+1, D+1+len(pb), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([pb], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(nl) + len(query_ids) + len(nl)
    al = out.logits[0][a0-1:a0-1+len(cand_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(cand_ids, device=dev)).item()

def serve_entropy(cache, D, query_ids):
    """Serve-time uncertainty: entropy of bare next-token dist at the answer position."""
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    pb = nl + list(query_ids) + nl; pos = torch.arange(D+1, D+1+len(pb), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([pb], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return float(-(logp.exp() * logp).sum().item())


def main():
    print(f"TASK-ALIGNED PRIMING  SMOKE={SMOKE} N={N_EVAL} K={K_DISTRACT}")
    data = load_datasets()
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-64-len(_M["nl"])) if _M["slim"] is not None else 765
        prefixes = {k: make_prefix(tok.encode(v, add_special_tokens=False), L_MATCH) for k, v in INSTRUCTIONS.items()}
        print(f"  loaded in {time.time()-t0:.0f}s")
        for dk in DATASETS:
            samples = data[dk]; pick = make_picker(samples)
            ck = RESULTS / mk / f"results_{dk}.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
            skey = f"taskaligned_{mk}" + ("_smoke" if SMOKE else ""); scored = []
            if ck.exists():
                prev = json.loads(ck.read_text())
                if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  {dk} resumed {len(scored)}")
            for idx in range(len(scored), len(samples)):
                s = samples[idx]
                doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
                q_ids = tok.encode(s["query"], add_special_tokens=False)
                c_ids = tok.encode(s["answer"], add_special_tokens=False)
                if not c_ids: continue
                distract = pick(idx, s["answer"], s.get("aliases"))
                d_ids = [x for x in (tok.encode(d, add_special_tokens=False) for d in distract) if x]
                rec = {"answer": s["answer"][:100]}
                with torch.no_grad():
                    bare, D = encode_phase_a(doc_ids, None)
                    rec["serve_entropy"] = serve_entropy(bare, D, q_ids)
                    for cond, pfx in [("bare", None)] + [(k, prefixes[k]) for k in INSTRUCTIONS]:
                        cache, D = (bare, D) if cond == "bare" else encode_phase_a(doc_ids, pfx)
                        rec[f"{cond}__c"] = cand_nll(cache, D, q_ids, c_ids)
                        rec[f"{cond}__d"] = [cand_nll(cache, D, q_ids, di) for di in d_ids]
                        if cond != "bare": del cache
                    del bare
                rec["margin"] = {c: float(np.mean(rec[f"{c}__d"]) - rec[f"{c}__c"]) for c in ["bare"]+list(INSTRUCTIONS)}
                rec["top1"] = {c: int(all(rec[f"{c}__c"] < x for x in rec[f"{c}__d"])) for c in ["bare"]+list(INSTRUCTIONS)}
                scored.append(rec); torch.cuda.empty_cache()
                if (idx+1) % 20 == 0 or SMOKE:
                    ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                    mg = {c: np.mean([r["margin"][c] for r in scored]) for c in ["bare","generic","entity","qa_directed"]}
                    print(f"    [{idx+1}/{len(samples)}] margin bare={mg['bare']:+.2f} gen={mg['generic']:+.2f} ent={mg['entity']:+.2f} qa={mg['qa_directed']:+.2f}")
            ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  {dk}: {len(scored)} done")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
