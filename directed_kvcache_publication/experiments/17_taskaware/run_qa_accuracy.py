#!/usr/bin/env python3
"""exp35: does the task-aware NLL story translate to downstream ACCURACY (exact-match / F1)?

§7/§7.1 measure the effect of conditioning vs selection in ANSWER-NLL. A reviewer will ask whether
that moves task accuracy, not just perplexity. Here we GENERATE the answer (greedy) from the
constructed cache and score SQuAD EM/F1, under the SAME four conditions as exp31b:

  bare         [BOS, full P]                          full doc, no prime
  prime_full   [BOS, Q, \n, P] -> strip Q, keep P     conditioning (discarded question prime)
  sel_plain    [BOS, P]        -> keep BOS + top-k     SnapKV-style selection (k=32)
  sel_primed   [BOS, Q, \n, P] -> keep BOS + top-k     selection + conditioning

For each: build cache, feed "\n{Q}\n", greedy-decode the answer, score EM/F1 vs gold answers.
Contrasts mirror §7.1: prime_full−bare (conditioning), sel_plain−bare (selection),
sel_primed−sel_plain (conditioning | selection). Same RoPE/normalize machinery; eager attn for probe.
N=200 SQuAD. Manual greedy loop with explicit position_ids (avoids the cache_position look-ahead bug).
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time, re, string, collections
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
KSEL = int(os.environ.get("KSEL", "32"))
MAXNEW = 24
# DATASET=squad (exp35, single-hop) | hotpot (exp36, multi-hop 2-paragraph gold context)
DATASET = os.environ.get("DATASET", "squad")
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / (
    "exp35_qa_accuracy" if DATASET == "squad" else "exp36_qa_accuracy_hotpot")
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},   # cond HELPS NLL
    "qwen25_1_5b":{"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"}, # cond HELPS NLL
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},  # cond HURTS NLL
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}


# ---- SQuAD EM/F1 (official normalization) ----
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em_score(pred, golds): return max(int(_norm(pred) == _norm(g)) for g in golds)

def f1_score(pred, golds):
    def _f1(p, g):
        pt, gt = _norm(p).split(), _norm(g).split()
        common = collections.Counter(pt) & collections.Counter(gt); ns = sum(common.values())
        if ns == 0: return 0.0
        prec, rec = ns / max(len(pt), 1), ns / max(len(gt), 1)
        return 2 * prec * rec / (prec + rec)
    return max(_f1(pred, g) for g in golds)


def load_squad(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    out = []
    for x in ds:
        golds = [t for t in x["answers"]["text"] if t]
        if not golds: continue
        ctx = x["context"]
        if not (30 <= count_words(ctx) <= 160): continue
        if golds[0].lower() not in ctx.lower(): continue
        out.append({"q": x["question"], "golds": golds, "ctx": ctx})
        if len(out) >= n: break
    return out


def load_hotpot(n):
    """Multi-hop: doc = the two GOLD supporting paragraphs concatenated (order as given).
    Extractive answers only (skip yes/no); answer must appear in the combined context."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    out = []
    for x in ds:
        ans = x["answer"].strip()
        if ans.lower() in ("yes", "no"): continue
        gold_titles = set(x["supporting_facts"]["title"])
        paras = [" ".join(sents) for t, sents in zip(x["context"]["title"], x["context"]["sentences"])
                 if t in gold_titles]
        if len(paras) != 2: continue
        ctx = " ".join(paras)
        if not (60 <= count_words(ctx) <= 300): continue
        if ans.lower() not in ctx.lower(): continue
        out.append({"q": x["question"], "golds": [ans], "ctx": ctx})
        if len(out) >= n: break
    return out


def load_data(n):
    return load_squad(n) if DATASET == "squad" else load_hotpot(n)


def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    kw = dict(dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0", attn_implementation="eager")
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, **kw).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


_M = {}
def probe_topk(P_ids, Q_ids, k):
    m, dev, bos, nl, lt = _M["m"], _M["dev"], _M["bos"], _M["nl"], _M["lt"]
    D = len(P_ids)
    if k >= D: return list(range(D))
    ids = [bos] + list(P_ids) + nl + list(Q_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), output_attentions=True, use_cache=False)
    attns = out.attentions
    full_layers = [i for i, t in enumerate(lt) if t == "full_attention"] or list(range(len(attns)))
    qpos = list(range(1 + D + len(nl), len(ids))); docpos = slice(1, 1 + D)
    imp = torch.zeros(D, device=dev)
    for li in full_layers:
        imp += attns[li][0][:, qpos, docpos].mean(dim=(0, 1)).float()
    return sorted(torch.topk(imp / max(len(full_layers), 1), k).indices.tolist())


def build(mode, P_ids, prime_ids, sel_rel=None):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    primed = mode in ("prime_full", "sel_primed")
    if primed:
        ids = [bos] + list(prime_ids) + nl + list(P_ids); doc_start = 1 + len(prime_ids) + len(nl)
    else:
        ids = [bos] + list(P_ids); doc_start = 1
    D = len(P_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "bare":
        return cache, len(ids)
    kept = list(range(D)) if (sel_rel is None or mode == "prime_full") else list(sel_rel)
    old_pos = [doc_start + j for j in kept]
    cache = select_kv_cache(cache, [0] + old_pos, device=dev)
    new_pos = list(range(1, 1 + len(kept)))
    cache = reposition_kv_cache(cache, torch.tensor(old_pos, device=dev),
                                torch.tensor(new_pos, device=dev), inv, lt, bos_start=0)
    cache = norm_roundtrip_kv_cache(cache)
    return cache, 1 + len(kept)


# instruct models generate the answer then run on into hallucinated chat turns; truncate at the
# earliest dialogue/turn marker (applied identically to every condition, so no per-condition confound).
_MARKERS = ["Human:", "Assistant:", "User:", "\nHuman", "\nAssistant", "Question:", "\n\n", "\n"]
def _clean(text):
    cut = len(text)
    for mk in _MARKERS:
        j = text.find(mk)
        if j != -1: cut = min(cut, j)
    return text[:cut].strip()

def generate(cache, clen, q_ids):
    """Greedy-decode the answer from the cache. A concise-answer instruction (identical across all
    conditions) is appended so verbose/reasoning models still emit a short span — this isolates
    correctness from generation length."""
    m, dev, nl, tok = _M["m"], _M["dev"], _M["nl"], _M["tok"]
    stop = set(_M["stop_ids"])
    lead = list(_M["lead_pre"]) + list(q_ids) + list(_M["lead_post"])
    c = deep_copy_cache(cache)
    pos = torch.arange(clen, clen + len(lead), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([lead], device=dev), position_ids=pos, past_key_values=c, use_cache=True)
    c = out.past_key_values
    nxt = int(out.logits[0, -1].argmax()); gen = []; curpos = clen + len(lead)
    for _ in range(MAXNEW):
        if nxt in stop: break
        gen.append(nxt)
        with torch.no_grad():
            out = m(input_ids=torch.tensor([[nxt]], device=dev),
                    position_ids=torch.tensor([[curpos]], device=dev), past_key_values=c, use_cache=True)
        c = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); curpos += 1
    return _clean(tok.decode(gen, skip_special_tokens=True))


def main():
    print(f"QA ACCURACY (EM/F1)  SMOKE={SMOKE}  N={N}  k={KSEL}")
    data = load_data(N); print(f"  {len(data)} {DATASET} items")
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        nl = tok.encode("\n", add_special_tokens=False)
        stop_ids = [tok.eos_token_id] + [t for t in [tok.convert_tokens_to_ids("<end_of_turn>")] if isinstance(t, int) and t >= 0]
        INSTR = "Answer the question with a short span copied from the passage, and nothing else."
        _M.update(m=m, tok=tok, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=nl, stop_ids=stop_ids,
                  lead_pre=tok.encode("\n" + INSTR + "\nQuestion: ", add_special_tokens=False),
                  lead_post=tok.encode("\nAnswer:", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-96-len(nl)) if _M["slim"] is not None else 700
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"qaacc_{DATASET}_{mk}_k{KSEL}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        CONDS = ["bare", "prime_full", "sel_plain", "sel_primed"]
        for i in range(len(scored), len(data)):
            d = data[i]
            Q = tok.encode(d["q"], add_special_tokens=False)
            P = tok.encode(d["ctx"], add_special_tokens=False)[:max_doc]
            sel = probe_topk(P, Q, KSEL)
            rec = {"golds": d["golds"]}
            for cond in CONDS:
                if cond == "bare":        c, cl = build("bare", P, None, None)
                elif cond == "prime_full":c, cl = build("prime_full", P, Q, None)
                elif cond == "sel_plain": c, cl = build("sel_plain", P, None, sel)
                else:                     c, cl = build("sel_primed", P, Q, sel)
                pred = generate(c, cl, Q); del c
                rec[f"{cond}__pred"] = pred
                rec[f"{cond}__em"] = em_score(pred, d["golds"])
                rec[f"{cond}__f1"] = f1_score(pred, d["golds"])
                torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                agg = lambda cond, met: 100*np.mean([s[f"{cond}__{met}"] for s in scored])
                print(f"    [{i+1}/{len(data)}] F1  bare={agg('bare','f1'):.1f} primeFull={agg('prime_full','f1'):.1f} "
                      f"selPlain={agg('sel_plain','f1'):.1f} selPrimed={agg('sel_primed','f1'):.1f} | "
                      f"EM bare={agg('bare','em'):.1f} primeFull={agg('prime_full','em'):.1f} "
                      f"selPlain={agg('sel_plain','em'):.1f} selPrimed={agg('sel_primed','em'):.1f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
