#!/usr/bin/env python3
"""Idea test #2: long-context positional rescue (needle-in-a-haystack).

In long contexts, models under-attend to content in the middle (lost-in-the-middle).
Here the "competitor" to the gold content is the model's own POSITIONAL attention
decay, not a semantically-hard negative -- so non-selective content amplification
could genuinely help (nothing competing benefits from the same boost).

Setup: a "needle" fact is inserted at fractional position p in a long filler
"haystack"; we cache the haystack (bare vs extract-primed), then score the needle's
answer NLL given the query. If priming reduces the mid-position NLL penalty, it
rescues lost-in-the-middle.

Conditions: bare, extract. Positions p in {0.0, 0.25, 0.5, 0.75, 1.0}.
Models: Qwen 1.5B, Qwen 7B (full attention -> long contexts; Gemma's sliding window
caps context length so it is excluded). Context ~2000 tokens. N=100 needles/position.
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, shutil, time
import random as pyrandom
from pathlib import Path
import numpy as np
import torch
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
N_EVAL = 6 if SMOKE else 100
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "800" if SMOKE else "2000"))
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp15_needle"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"},
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    # primable model: fair test of positional rescue (Qwen is barely primable). Gemma's
    # sliding window (1024) caps context -> run with CTX_TOKENS=700 to stay in-window.
    "gemma3_12b":  {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY:
    MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
elif SMOKE:
    MODELS = {"qwen25_1_5b": MODELS["qwen25_1_5b"]}

# 100 distinct needles (varied codes/entities) to avoid memorization confounds
CODES = [f"{a}{b}{c}{d}" for a, b, c, d in zip(
    np.random.RandomState(1).randint(1,10,200), np.random.RandomState(2).randint(0,10,200),
    np.random.RandomState(3).randint(0,10,200), np.random.RandomState(4).randint(0,10,200))]
NEEDLE_TMPL = "The secret passcode for the {city} meeting is {code}."
CITIES = ["Lisbon","Tokyo","Cairo","Oslo","Lima","Perth","Accra","Hanoi","Quito","Riga",
          "Sofia","Tunis","Dakar","Minsk","Amman","Doha","Baku","Kyiv","Vienna","Bogota"]


def build_filler(tok, target_tokens):
    """Filler text from unrelated SQuAD context paragraphs (small, reliably cached)."""
    ds = load_dataset("rajpurkar/squad", split="validation")
    sents, seen = [], set()
    for x in ds:
        ctx = x["context"]
        if ctx in seen: continue
        seen.add(ctx)
        for s in ctx.split(". "):
            s = s.strip()
            if 8 <= len(s.split()) <= 40: sents.append(s + ".")
        if len(sents) > 20000: break
    pyrandom.seed(SEED); pyrandom.shuffle(sents)
    return sents


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
def encode_ctx(doc_ids, prefix_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
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

def answer_nll(cache, D, q_ids, a_ids):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids) + nl + list(a_ids); pos = torch.arange(D+1, D+1+len(seq), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(nl) + len(q_ids) + len(nl); al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"NEEDLE  SMOKE={SMOKE} N={N_EVAL} CTX={CTX_TOKENS} positions={POSITIONS}")
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        sents = build_filler(tok, CTX_TOKENS)
        # pre-tokenize a big filler stream
        filler_ids = []
        si = 0
        while len(filler_ids) < CTX_TOKENS * (N_EVAL + 5):
            filler_ids += tok.encode(" " + sents[si % len(sents)], add_special_tokens=False); si += 1
        print(f"  loaded in {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"needle_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]; print(f"  resumed {len(scored)}")
        for idx in range(len(scored), N_EVAL):
            rng = pyrandom.Random(SEED + idx)
            code = CODES[idx % len(CODES)]; city = CITIES[idx % len(CITIES)]
            needle = NEEDLE_TMPL.format(city=city, code=code)
            q = f"What is the secret passcode for the {city} meeting?"
            needle_ids = tok.encode(" " + needle, add_special_tokens=False)
            q_ids = tok.encode(q, add_special_tokens=False); a_ids = tok.encode(" " + code, add_special_tokens=False)
            start = rng.randint(0, len(filler_ids) - CTX_TOKENS - 50)
            base_filler = filler_ids[start:start + CTX_TOKENS]
            rec = {"idx": idx}
            for p in POSITIONS:
                ins = int(p * len(base_filler))
                doc_ids = base_filler[:ins] + needle_ids + base_filler[ins:]
                for cond, pfx in [("bare", None), ("extract", ext)]:
                    cache, D = encode_ctx(doc_ids, pfx)
                    rec[f"p{p}__{cond}"] = answer_nll(cache, D, q_ids, a_ids); del cache; torch.cuda.empty_cache()
            scored.append(rec)
            if (idx+1) % 10 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                msg = " ".join(f"p{p}:b={np.mean([r[f'p{p}__bare'] for r in scored]):.2f}/e={np.mean([r[f'p{p}__extract'] for r in scored]):.2f}" for p in POSITIONS)
                print(f"    [{idx+1}/{N_EVAL}] needle-NLL {msg}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
