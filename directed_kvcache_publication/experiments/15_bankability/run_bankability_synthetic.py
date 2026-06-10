#!/usr/bin/env python3
"""BANKABILITY, clean version: context that is DECISIVE for the answer token.

HotpotQA failed as a probe because the answer is already determined by the cached
paragraph -> extra context doesn't move answer-NLL retained OR stripped. Here the answer
is UNKNOWABLE without the context fact C, so the retained effect is unambiguous.

  C = "The secret access code for the {city} office is {code}."   (decisive fact)
  D = an unrelated filler passage (the 'cached document')
  Q = "What is the secret access code for the {city} office?"   A = " {code}"

  no_ctx (bare):  cache=[BOS, D]                 ; ask Q -> A   (code unknowable -> high NLL)
  retain:         cache=[BOS, C, \n, D]  (KEEP)  ; ask Q -> A   (code attendable -> ~0 NLL)
  strip:          cache=[BOS, C, \n, D] -> strip C, reposition D ; ask Q -> A
                                                                 (C primed into D's KV, removed)
  retain_after:   cache=[BOS, D, \n, C]  (KEEP)  ; ask Q -> A   (C after D, attendable)

PIPELINE SANITY: retain must be << no_ctx (if not, the machinery is broken). BANKABILITY:
bankable_fraction = (no_ctx - strip) / (no_ctx - retain). ~0 => imprint not bankable;
~1 => priming fully banks the context. Models: gemma3_4b, qwen25_7b. N=150.
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
N = 6 if SMOKE else 150
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp24_bank_synth"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}

CITIES = ["Lisbon","Tokyo","Cairo","Oslo","Lima","Perth","Accra","Hanoi","Quito","Riga",
          "Sofia","Tunis","Dakar","Minsk","Amman","Doha","Baku","Kyiv","Vienna","Bogota",
          "Maputo","Tbilisi","Skopje","Nicosia","Tirana","Yerevan","Bishkek","Astana","Vientiane","Gaborone"]
RS = np.random.RandomState(7)
CODES = [f"{RS.randint(1,10)}{RS.randint(0,10)}{RS.randint(0,10)}{RS.randint(0,10)}" for _ in range(400)]


def load_filler(n):
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, out = set(), []
    for x in ds:
        c = x["context"]
        if c in seen: continue
        seen.add(c)
        if 30 <= count_words(c) <= 160: out.append(c)
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
def build_cache(mode, D_ids, C_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "no_ctx":
        ids = [bos] + list(D_ids); clen = len(ids)
    elif mode == "retain":          # [BOS, C, nl, D]
        ids = [bos] + list(C_ids) + nl + list(D_ids); clen = len(ids)
    elif mode == "retain_after":    # [BOS, D, nl, C]
        ids = [bos] + list(D_ids) + nl + list(C_ids); clen = len(ids)
    elif mode == "strip":           # [BOS, C, nl, D] -> keep BOS+D, reposition to 1..D
        ids = [bos] + list(C_ids) + nl + list(D_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "strip":
        D = len(D_ids); d_start = 1 + len(C_ids) + len(nl)
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
    a0 = len(seq) - len(a_ids)
    al = out.logits[0][a0-1:a0-1+len(a_ids)]
    return torch.nn.functional.cross_entropy(al, torch.tensor(a_ids, device=dev)).item()


def main():
    print(f"BANKABILITY SYNTHETIC  SMOKE={SMOKE}  N={N}")
    fillers = load_filler(N + 5)
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"banksyn_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), N):
            city = CITIES[i % len(CITIES)]; code = CODES[i]
            C = f"The secret access code for the {city} office is {code}."
            Q = f"What is the secret access code for the {city} office?"
            C_ids = tok.encode(C, add_special_tokens=False)
            Q_ids = tok.encode(Q, add_special_tokens=False)
            A_ids = tok.encode(" " + code, add_special_tokens=False)
            D_ids = tok.encode(fillers[i], add_special_tokens=False)[:160]
            rec = {}
            for mode in ["no_ctx", "retain", "retain_after", "strip"]:
                c, cl = build_cache(mode, D_ids, C_ids)
                rec[mode] = answer_nll(c, cl, Q_ids, A_ids); del c; torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = {k: np.mean([s[k] for s in scored]) for k in ["no_ctx","retain","retain_after","strip"]}
                bf = (A["no_ctx"]-A["strip"])/(A["no_ctx"]-A["retain"]+1e-9)
                print(f"    [{i+1}/{N}] NLL no_ctx={A['no_ctx']:.2f} retain={A['retain']:.2f} "
                      f"retain_after={A['retain_after']:.2f} strip={A['strip']:.2f} | bankable_frac={bf:+.2f}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
