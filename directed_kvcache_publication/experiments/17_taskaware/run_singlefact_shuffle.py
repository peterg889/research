#!/usr/bin/env python3
"""exp33: is exp26 "semantic banking" MEANING or TOKEN PRESENCE? (single-fact shuffle control)

exp26's headline "semantic axis" primes a filler doc with ONE fact, strips it, and measures recall
of the answer token. With one fact there is only one topic to recall, so token presence of that
topic alone suffices -- the measure cannot separate meaning-banking from lexical banking. Here we
add the missing control: prime with the SAME fact tokens, ORDERED vs SHUFFLED.

  bare      [BOS, D]                       recall
  neutral   [BOS, <nl*|F|>, D] -> strip    recall      (machinery)
  ord       [BOS, F, D]        -> strip    recall      (ordered fact)
  shuf      [BOS, shuf(F), D]  -> strip    recall      (same tokens, no order/meaning)
  retain    [BOS, F, D]                     recall     (upper bound)

  bank_ord = ord - neutral ;  bank_shuf = shuf - neutral
  ORDER    = ord - shuf   (neg => meaning/order helps recall beyond token presence;
                           ~0  => exp26 "semantic banking" is TOKEN PRESENCE, label is overstated)

Run for sem (meaningful topic) and code (literal digits) targets, mirroring exp26. N=150.
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
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp33_singlefact_shuffle"
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
RS = np.random.RandomState(23)
CODES = [f"{RS.randint(1,10)}{RS.randint(0,10)}{RS.randint(0,10)}{RS.randint(0,10)}" for _ in range(400)]


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
    else:                  ids = [bos] + list(F_ids) + nl + list(D_ids)  # strip
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
    print(f"SINGLE-FACT SHUFFLE (semantic = meaning or tokens?)  SMOKE={SMOKE}  N={N}")
    fillers = load_filler(N + 5)
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"sf_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), N):
            rng = random.Random(700 + i)
            city = CITIES[i % len(CITIES)]; code = CODES[i]; topic = TOPICS[(i*7) % len(TOPICS)]
            D = tok.encode(fillers[i], add_special_tokens=False)[:160]
            specs = {
                "code": (f"The access code for the {city} office is {code}.",
                         f"What is the access code for the {city} office?", " " + code),
                "sem":  (f"The {city} office specializes entirely in {topic}.",
                         f"What does the {city} office specialize entirely in?", " " + topic),
            }
            rec = {}
            for typ, (F, Q, A) in specs.items():
                Fi = tok.encode(F, add_special_tokens=False)
                Fs = list(Fi); rng.shuffle(Fs)
                Qi = tok.encode(Q, add_special_tokens=False)
                Ai = tok.encode(A, add_special_tokens=False)
                neutral = (_M["nl"] * len(Fi))[:len(Fi)]
                c, cl = build("bare",  D, None);    rec[f"{typ}_bare"]    = answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("strip", D, neutral); rec[f"{typ}_neutral"] = answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("strip", D, Fi);      rec[f"{typ}_ord"]     = answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("strip", D, Fs);      rec[f"{typ}_shuf"]    = answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("retain",D, Fi);      rec[f"{typ}_retain"]  = answer_nll(c, cl, Qi, Ai); del c
                torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = {k: np.mean([s[k] for s in scored]) for k in rec}
                for typ in ["sem","code"]:
                    print(f"    [{i+1}/{N}] {typ}: bankOrd={A[f'{typ}_ord']-A[f'{typ}_neutral']:+.2f} "
                          f"bankShuf={A[f'{typ}_shuf']-A[f'{typ}_neutral']:+.2f} "
                          f"ORDER(ord-shuf)={A[f'{typ}_ord']-A[f'{typ}_shuf']:+.2f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
