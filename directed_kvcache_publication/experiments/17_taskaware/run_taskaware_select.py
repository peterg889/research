#!/usr/bin/env python3
"""exp31b: CONDITIONING vs SELECTION (the make-or-break control vs Beyond RAG / SnapKV).

If we know the task at build time, the obvious baseline (Beyond RAG, SnapKV) is to KEEP the
task-relevant tokens and drop the rest -- task-aware SELECTION. Our method instead keeps the
doc and CONDITIONS it with a discarded prime. This experiment asks, at an ISO-BUDGET retained
set, whether conditioning adds value BEYOND selecting the same tokens.

Per SQuAD item we pick the top-k doc tokens by attention from the question (SnapKV-style probe:
forward [BOS, P, \n, Q], pool attention from Q positions to P positions over full-attention
layers). Then we score the answer under four caches:

  bare         [BOS, full P]                          full budget, no prime   (RAG baseline)
  prime_full   [BOS, Q, \n, P] -> strip Q, keep P     full budget, conditioned (= exp29 q_primed)
  sel_k_plain  [BOS, P]        -> keep BOS + top-k     budget k,    NOT conditioned (SnapKV-ish)
  sel_k_primed [BOS, Q, \n, P] -> keep BOS + top-k     budget k,    conditioned (selection + prime)

The decisive contrast holds the RETAINED SET fixed (same top-k indices):
  cond_value = sel_k_primed - sel_k_plain   (value of conditioning, given the kept tokens)
If cond_value < 0 (helps), conditioning adds signal selection cannot get by keeping tokens alone.
Other contrasts: sel_k_plain - bare (pure selection at budget k), sel_k_primed - bare (best op pt).

SQuAD, N=300, k=KSEL (default 32). Same RoPE/normalize machinery as exp29/31a. Eager attn for the probe.
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
KSEL = int(os.environ.get("KSEL", "32"))
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp31_taskaware_select"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
# Expanded set spans the imprintability range to test whether select-vs-condition tracks a
# CONTINUOUS trait (imprintability) rather than family. Low-imprintability/surface models should
# favor conditioning; high-imprintability ones should favor selection.
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},  # imprint 0.20
    "qwen25_3b":   {"name": "Qwen/Qwen2.5-3B-Instruct",   "loader": "AutoModelForCausalLM"},  # imprint ~0.30
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},  # imprint 0.37
    "qwen25_14b":  {"name": "Qwen/Qwen2.5-14B-Instruct",  "loader": "AutoModelForCausalLM"},  # imprint 0.39
    "gemma3_1b":   {"name": "google/gemma-3-1b-it",       "loader": "Gemma3ForCausalLM"},     # imprint 0.43
    "mistral_7b":  {"name": "mistralai/Mistral-7B-Instruct-v0.3", "loader": "AutoModelForCausalLM"},  # 0.55 structure
    "gemma3_4b":   {"name": "google/gemma-3-4b-it",       "loader": "Gemma3ForConditionalGeneration"},  # 0.60
    "gemma3_12b":  {"name": "google/gemma-3-12b-it",      "loader": "Gemma3ForConditionalGeneration"},  # 0.84
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
    kw = dict(dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0", attn_implementation="eager")
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, **kw).eval()
    elif loader == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        m = Gemma3ForCausalLM.from_pretrained(name, **kw).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return m, tok


def answer_doc_indices(tok, ctx, ans, n_doc):
    """Doc-relative token indices (within the first n_doc tokens) that overlap the answer span,
    via the tokenizer's char offset mapping. Used to test whether selection drops the answer."""
    off = ctx.lower().find(ans.lower())
    if off < 0:
        return []
    try:
        enc = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
    except Exception:
        return []
    s, e = off, off + len(ans)
    return [i for i, (a, b) in enumerate(offs) if i < n_doc and a < e and b > s]

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


_M = {}

def probe_topk(P_ids, Q_ids, k):
    """SnapKV-style: forward [BOS, P, nl, Q], pool attention from Q->P over full-attention
    layers, return the doc-relative indices (0..D-1) of the top-k most-attended doc tokens,
    sorted ascending (reading order)."""
    m, dev, bos, nl, lt = _M["m"], _M["dev"], _M["bos"], _M["nl"], _M["lt"]
    D = len(P_ids)
    if k >= D: return list(range(D))
    ids = [bos] + list(P_ids) + nl + list(Q_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), output_attentions=True, use_cache=False)
    attns = out.attentions  # tuple[n_layers] of [1, heads, seq, seq]
    full_layers = [i for i, t in enumerate(lt) if t == "full_attention"] or list(range(len(attns)))
    qpos = list(range(1 + D + len(nl), len(ids)))
    docpos = slice(1, 1 + D)
    imp = torch.zeros(D, device=dev)
    for li in full_layers:
        a = attns[li][0]                       # [heads, seq, seq]
        sub = a[:, qpos, docpos]               # [heads, |Q|, D]
        imp += sub.mean(dim=(0, 1)).float()    # mean over heads and query positions
    imp /= max(len(full_layers), 1)
    top = torch.topk(imp, k).indices.tolist()
    return sorted(top)


def build(mode, P_ids, prime_ids, sel_rel=None):
    """mode in {bare, prime_full, sel_plain, sel_primed}. sel_rel = doc-relative kept indices."""
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
    kept = list(range(D)) if sel_rel is None or mode in ("prime_full",) else list(sel_rel)
    old_pos = [doc_start + j for j in kept]
    cache = select_kv_cache(cache, [0] + old_pos, device=dev)
    new_pos = list(range(1, 1 + len(kept)))
    cache = reposition_kv_cache(cache, torch.tensor(old_pos, device=dev),
                                torch.tensor(new_pos, device=dev), inv, lt, bos_start=0)
    cache = norm_roundtrip_kv_cache(cache)
    return cache, 1 + len(kept)


def answer_nll(cache, clen, q_ids, a_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids) + nl + list(a_ids)
    pos = torch.arange(clen, clen + len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(seq) - len(a_ids); al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"CONDITIONING vs SELECTION (exp31b)  SMOKE={SMOKE}  N={N}  k={KSEL}")
    data = load_squad(N); print(f"  {len(data)} squad items")
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        max_doc = (_M["slim"]-1-96-len(_M["nl"])) if _M["slim"] is not None else 700
        print(f"  loaded {time.time()-t0:.0f}s  full_attn_layers={sum(1 for t in _M['lt'] if t=='full_attention')}/{len(_M['lt'])}")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"tsel_{mk}_k{KSEL}_as" + ("_smoke" if SMOKE else ""); scored = []  # _as: logs answer-span survival
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), len(data)):
            d = data[i]
            Q = tok.encode(d["q"], add_special_tokens=False)
            A = tok.encode(" " + d["a"], add_special_tokens=False)
            P = tok.encode(d["ctx"], add_special_tokens=False)[:max_doc]
            sel = probe_topk(P, Q, KSEL)
            adoc = answer_doc_indices(tok, d["ctx"], d["a"], len(P))  # answer's doc-token indices in P
            selset = set(sel)
            rec = {"D": len(P), "k": len(sel),
                   "a_span": len(adoc), "a_in_sel": sum(1 for j in adoc if j in selset)}
            # bare      = raw full doc (comparable to exp29).  bare_norm = full doc, repositioned+normalized
            # (the machinery-matched reference for selVal, since norm alone shifts NLL ~0.6-3.7 nats).
            c, cl = build("bare",       P, None, None);          rec["bare"]         = answer_nll(c, cl, Q, A); del c
            c, cl = build("sel_plain",  P, None, list(range(len(P)))); rec["bare_norm"] = answer_nll(c, cl, Q, A); del c
            c, cl = build("prime_full", P, Q,    None);          rec["prime_full"]   = answer_nll(c, cl, Q, A); del c
            c, cl = build("sel_plain",  P, None, sel);           rec["sel_k_plain"]  = answer_nll(c, cl, Q, A); del c
            c, cl = build("sel_primed", P, Q,    sel);           rec["sel_k_primed"] = answer_nll(c, cl, Q, A); del c
            torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                Aa = {x: np.mean([s[x] for s in scored]) for x in ["bare","bare_norm","prime_full","sel_k_plain","sel_k_primed"]}
                tot_span = sum(s["a_span"] for s in scored); tot_in = sum(s["a_in_sel"] for s in scored)
                asurv = (tot_in / tot_span) if tot_span else float("nan")
                print(f"    [{i+1}/{len(data)}] primeFull={Aa['prime_full']:.3f} selPlain={Aa['sel_k_plain']:.3f} "
                      f"selPrimed={Aa['sel_k_primed']:.3f} | selVal={Aa['sel_k_plain']-Aa['bare_norm']:+.3f} "
                      f"COND|sel={Aa['sel_k_primed']-Aa['sel_k_plain']:+.3f} primeVal={Aa['prime_full']-Aa['bare_norm']:+.3f} "
                      f"| ans-span-survival={asurv:.2f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
