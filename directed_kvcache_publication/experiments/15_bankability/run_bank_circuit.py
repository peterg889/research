#!/usr/bin/env python3
"""exp28: WHERE does semantic banking live? Layer-wise KV patching on semantic recovery.

Prime filler D with a semantic fact F (rare topic), strip F. Score the topic answer A with a
HYBRID cache = bare-D everywhere EXCEPT layer L gets the primed-D K/V. recovery_L =
NLL_bare(A) - NLL_hybrid_L(A)  (positive => layer L's primed K/V carries the banked topic).
Sweep L. gemma3_12b (semantic banker) vs qwen25_7b (not). Localizes the banking circuit.
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
N = 4 if SMOKE else 30
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp28_bank_circuit"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_12b": {"name": "google/gemma-3-12b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}

CITIES = ["Lisbon","Tokyo","Cairo","Oslo","Lima","Perth","Accra","Hanoi","Quito","Riga",
          "Sofia","Tunis","Dakar","Minsk","Amman","Doha","Baku","Kyiv","Vienna","Bogota",
          "Maputo","Tbilisi","Skopje","Nicosia","Tirana","Yerevan","Bishkek","Astana","Vientiane","Gaborone"]
RARE = ["robotics","archaeology","cartography","virology","metallurgy","oceanography","linguistics",
        "horticulture","seismology","cryptography","entomology","astronomy","numismatics","hydrology",
        "acoustics","glassblowing","beekeeping","falconry","viticulture","taxidermy","origami",
        "calligraphy","spelunking","philately","cartooning","meteorology","forestry","genetics","ceramics","cinematography"]


def load_filler(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, out = set(), []
    for x in ds:
        c = x["context"]
        if c in seen: continue
        seen.add(c)
        if 30 <= count_words(c) <= 130: out.append(c)
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
def encode(mode, D_ids, F_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare": ids = [bos] + list(D_ids)
    else: ids = [bos] + list(F_ids) + nl + list(D_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; D = len(D_ids)
    if mode == "primed":
        ds = 1 + len(F_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(ds, ds + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(ds, ds + D, device=dev),
                                    torch.arange(1, 1 + D, device=dev), inv, lt, bos_start=0)
        cache = norm_roundtrip_kv_cache(cache)
    return cache, 1 + D

def patch(cb, cp, L):
    h = deep_copy_cache(cb)
    h.layers[L].keys = cp.layers[L].keys.clone()
    h.layers[L].values = cp.layers[L].values.clone()
    return h

def nll(cache, clen, q_ids, a_ids, copy=True):
    m, dev, nl = _M["m"], _M["dev"], _M["nl"]
    seq = nl + list(q_ids) + nl + list(a_ids)
    pos = torch.arange(clen, clen + len(seq), device=dev).unsqueeze(0)
    pkv = deep_copy_cache(cache) if copy else cache
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), position_ids=pos, past_key_values=pkv, use_cache=False)
    a0 = len(seq) - len(a_ids); al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"BANK CIRCUIT (semantic recovery layer localization)  SMOKE={SMOKE}  N={N}")
    fillers = load_filler(N + 3)
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"bankcirc_{mk}" + ("_smoke" if SMOKE else ""); recovs, fulls = [], []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: recovs = prev["recovs"]; fulls = prev["fulls"]
        for i in range(len(recovs), N):
            c = CITIES[i % len(CITIES)]; topic = RARE[(i*7) % len(RARE)]
            F = f"The {c} office specializes entirely in {topic}."
            Q = f"What does the {c} office specialize entirely in?"
            Fi = tok.encode(F, add_special_tokens=False); Qi = tok.encode(Q, add_special_tokens=False)
            Ai = tok.encode(" " + topic, add_special_tokens=False)
            D = tok.encode(fillers[i], add_special_tokens=False)[:130]
            cb, lb = encode("bare", D, None); cp, lp = encode("primed", D, Fi)
            base = nll(cb, lb, Qi, Ai); full = nll(cp, lp, Qi, Ai)
            nL = len(cb.layers); rec = []
            for L in range(nL):
                h = patch(cb, cp, L)
                rec.append(base - nll(h, lb, Qi, Ai, copy=False)); del h; torch.cuda.empty_cache()
            recovs.append(rec); fulls.append(base - full)
            del cb, cp; torch.cuda.empty_cache()
            if (i+1) % 5 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "recovs": recovs, "fulls": fulls}))
                e = np.array(recovs).mean(0); nL = len(e)
                top = sorted(range(nL), key=lambda L: -e[L])[:5]
                print(f"    [{i+1}/{N}] full_recovery={np.mean(fulls):+.2f}  top-5 layers={top}  "
                      f"profile(deciles)=" + " ".join(f"{e[int(q*(nL-1))]:+.2f}" for q in np.linspace(0,1,6)))
        ck.write_text(json.dumps({"scoring_key": skey, "recovs": recovs, "fulls": fulls})); print(f"  done {len(recovs)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
