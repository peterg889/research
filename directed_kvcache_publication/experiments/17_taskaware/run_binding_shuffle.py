#!/usr/bin/env python3
"""exp32: does zero-retention priming bank MEANING (relational binding) or just TOKEN PRESENCE?

The exp26 "semantic banking" measure used ONE fact per doc, so recalling the single primed topic
word suffices -- token presence alone answers it, and it cannot tell meaning from lexical banking.
This experiment forces BINDING: prime the doc with TWO facts and ask about one.

  F  = "The {c1} office specializes entirely in {t1}. The {c2} office specializes entirely in {t2}."
  Q  = "What does the {c_ask} office specialize entirely in?"   A = t_ask  (t1 or t2, randomized)

Both topics t1,t2 are present in the cache, so a TOKEN-PRESENCE bank cannot tell which goes with
c_ask; only a bank that preserves the city->topic BINDING can. We compare ordered vs token-shuffled
priming (same token multiset, structure destroyed):

  bare        [BOS, D]                       ; recall            (no fact)
  neutral     [BOS, <nl*|F|>, D] -> strip    ; recall            (machinery)
  strip_ord   [BOS, F, D]        -> strip     ; recall           (ordered facts, banked then removed)
  strip_shuf  [BOS, shuf(F), D]  -> strip     ; recall           (same tokens, no structure)
  retain_ord  [BOS, F, D]                     ; recall           (upper bound, facts retained)

  bank_ord  = strip_ord  - neutral     bank_shuf = strip_shuf - neutral
  BINDING   = strip_ord  - strip_shuf  (neg => ordered beats shuffled => genuine relational banking;
                                        ~0 => banking is token-presence, NOT meaning)

If BINDING ~ 0 across models, the headline "semantic axis" is really lexical/topic-token banking.
N=150. Same RoPE/normalize machinery as exp26.
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
N = 6 if SMOKE else 150
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp32_binding_shuffle"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
    "mistral_7b": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "loader": "AutoModelForCausalLM"},
    "gemma3_4b":  {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "gemma3_27b": {"name": "google/gemma-3-27b-it", "loader": "Gemma3ForConditionalGeneration"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}

CITIES = ["Lisbon","Tokyo","Cairo","Oslo","Lima","Perth","Accra","Hanoi","Quito","Riga",
          "Sofia","Tunis","Dakar","Minsk","Amman","Doha","Baku","Kyiv","Vienna","Bogota",
          "Maputo","Tbilisi","Skopje","Nicosia","Tirana","Yerevan","Bishkek","Astana","Vientiane","Gaborone"]
TOPICS = ["robotics","archaeology","cartography","virology","metallurgy","oceanography","linguistics",
          "horticulture","seismology","cryptography","entomology","astronomy","ceramics","forestry",
          "meteorology","numismatics","hydrology","genetics","acoustics","glassblowing","beekeeping",
          "falconry","viticulture","cinematography","taxidermy","origami","calligraphy","spelunking",
          "philately","cartooning"]


def load_filler(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, out = set(), []
    for x in ds:
        c = x["context"]
        if c in seen: continue
        seen.add(c)
        if 30 <= count_words(c) <= 140: out.append(c)
        if len(out) >= n: break
    return out


def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if loader == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        m = Gemma3ForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    elif loader == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        m = Gemma3ForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


_M = {}
def build(mode, D_ids, F_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare":     ids = [bos] + list(D_ids); clen = len(ids)
    elif mode == "retain": ids = [bos] + list(F_ids) + nl + list(D_ids); clen = len(ids)
    else:                  ids = [bos] + list(F_ids) + nl + list(D_ids)   # strip
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "strip":
        D = len(D_ids); d_start = 1 + len(F_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(d_start, d_start + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(d_start, d_start + D, device=dev),
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
    print(f"BINDING SHUFFLE (meaning vs token-presence)  SMOKE={SMOKE}  N={N}")
    fillers = load_filler(N + 5)
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"bind_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), N):
            rng = random.Random(900 + i)
            c1, c2 = CITIES[i % len(CITIES)], CITIES[(i + 7) % len(CITIES)]
            t1, t2 = TOPICS[(i * 3) % len(TOPICS)], TOPICS[(i * 3 + 5) % len(TOPICS)]
            if c2 == c1: c2 = CITIES[(i + 11) % len(CITIES)]
            if t2 == t1: t2 = TOPICS[(i * 3 + 9) % len(TOPICS)]
            F = f"The {c1} office specializes entirely in {t1}. The {c2} office specializes entirely in {t2}."
            ask_first = rng.random() < 0.5
            c_ask, t_ask = (c1, t1) if ask_first else (c2, t2)
            Q = f"What does the {c_ask} office specialize entirely in?"
            D = tok.encode(fillers[i], add_special_tokens=False)[:160]
            Fi = tok.encode(F, add_special_tokens=False)
            Fs = list(Fi); rng.shuffle(Fs)
            Qi = tok.encode(Q, add_special_tokens=False)
            Ai = tok.encode(" " + t_ask, add_special_tokens=False)
            neutral = (_M["nl"] * len(Fi))[:len(Fi)]
            rec = {"ask_first": ask_first}
            c, cl = build("bare",   D, None);    rec["bare"]       = answer_nll(c, cl, Qi, Ai); del c
            c, cl = build("strip",  D, neutral); rec["neutral"]    = answer_nll(c, cl, Qi, Ai); del c
            c, cl = build("strip",  D, Fi);      rec["strip_ord"]  = answer_nll(c, cl, Qi, Ai); del c
            c, cl = build("strip",  D, Fs);      rec["strip_shuf"] = answer_nll(c, cl, Qi, Ai); del c
            c, cl = build("retain", D, Fi);      rec["retain_ord"] = answer_nll(c, cl, Qi, Ai); del c
            torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = {k: np.mean([s[k] for s in scored]) for k in ["bare","neutral","strip_ord","strip_shuf","retain_ord"]}
                print(f"    [{i+1}/{N}] bare={A['bare']:.2f} retain={A['retain_ord']:.2f} | "
                      f"bankOrd={A['strip_ord']-A['neutral']:+.2f} bankShuf={A['strip_shuf']-A['neutral']:+.2f} "
                      f"BINDING(ord-shuf)={A['strip_ord']-A['strip_shuf']:+.2f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
