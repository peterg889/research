#!/usr/bin/env python3
"""exp27: is it MEANING that banks, or just in-vocabulary tokens? Banking across a spectrum
of answer types on a strong semantic-banker (gemma3_12b) vs a literal-banker (qwen25_7b).

  code        "...access code... is 4821."            4821       (digits, meaningless)
  pseudoword  "...codename is Blorvak."               Blorvak    (pronounceable NON-word, meaningless)
  rare_word   "...specializes in robotics."           robotics   (rare, meaningful)
  common_word "...office is painted blue."            blue       (common, meaningful)
  phrase      "...specializes in marine biology."     marine biology (multi-word, meaningful)

If Gemma banks the MEANINGFUL ones (rare/common/phrase) but not the MEANINGLESS ones
(code/pseudoword), 'semantic banking' is the right characterization. If it banks pseudowords
too, the effect is about token-ness, not meaning. content=strip(F)-strip(neutral), machinery-
controlled (neutral=newlines). N=150.
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
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp27_bank_types"
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
RS = np.random.RandomState(31)
CODES = [f"{RS.randint(1,10)}{RS.randint(0,10)}{RS.randint(0,10)}{RS.randint(0,10)}" for _ in range(400)]
PSEUDO = ["Blorvak","Trquenil","Vandethorpe","Quizzane","Morbleck","Frinda","Skelvor","Plonquist",
          "Drannock","Velmiry","Cthonel","Grunswick","Yarbeth","Lomquar","Phenwick","Brizzel",
          "Twanglemore","Crendol","Vexmoor","Snarvic","Gloumby","Pendrath","Whilford","Quobble",
          "Trelmek","Bavorny","Klindash","Murvex","Slindro","Gawkmoth"]
RARE = ["robotics","archaeology","cartography","virology","metallurgy","oceanography","linguistics",
        "horticulture","seismology","cryptography","entomology","astronomy","numismatics","hydrology",
        "acoustics","glassblowing","beekeeping","falconry","viticulture","taxidermy","origami",
        "calligraphy","spelunking","philately","cartooning","meteorology","forestry","genetics","ceramics","cinematography"]
COMMON = ["blue","green","red","yellow","brown","orange","purple","silver","golden","crimson",
          "white","black","grey","pink","teal","violet","amber","ivory","scarlet","maroon"]
PHRASE = ["marine biology","quantum optics","medieval history","tropical agriculture","urban planning",
          "structural engineering","ancient philosophy","molecular gastronomy","arctic geology","renaissance art",
          "industrial design","cognitive science","desert ecology","colonial architecture","forensic chemistry"]


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
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


_M = {}
def build(mode, D_ids, F_ids):
    m, dev, inv, lt, bos, nl = _M["m"], _M["dev"], _M["inv"], _M["lt"], _M["bos"], _M["nl"]
    if mode == "bare": ids = [bos] + list(D_ids); clen = len(ids)
    elif mode == "retain": ids = [bos] + list(F_ids) + nl + list(D_ids); clen = len(ids)
    elif mode == "strip": ids = [bos] + list(F_ids) + nl + list(D_ids)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values
    if mode == "strip":
        D = len(D_ids); ds = 1 + len(F_ids) + len(nl)
        cache = select_kv_cache(cache, [0] + list(range(ds, ds + D)), device=dev)
        cache = reposition_kv_cache(cache, torch.arange(ds, ds + D, device=dev),
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


def specs_for(i):
    c = CITIES[i % len(CITIES)]
    return {
        "code":        (f"The access code for the {c} office is {CODES[i]}.",
                        f"What is the access code for the {c} office?", " " + CODES[i]),
        "pseudoword":  (f"The codename of the {c} office is {PSEUDO[i%len(PSEUDO)]}.",
                        f"What is the codename of the {c} office?", " " + PSEUDO[i%len(PSEUDO)]),
        "rare_word":   (f"The {c} office specializes entirely in {RARE[(i*7)%len(RARE)]}.",
                        f"What does the {c} office specialize entirely in?", " " + RARE[(i*7)%len(RARE)]),
        "common_word": (f"The {c} office is painted entirely {COMMON[(i*3)%len(COMMON)]}.",
                        f"What color is the {c} office painted?", " " + COMMON[(i*3)%len(COMMON)]),
        "phrase":      (f"The {c} office specializes entirely in {PHRASE[(i*5)%len(PHRASE)]}.",
                        f"What does the {c} office specialize entirely in?", " " + PHRASE[(i*5)%len(PHRASE)]),
    }


def main():
    print(f"BANK TYPES (meaning vs token)  SMOKE={SMOKE}  N={N}")
    fillers = load_filler(N + 5)
    TYPES = ["code","pseudoword","rare_word","common_word","phrase"]
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        t0 = time.time(); m, tok = load_model(spec["name"], spec["loader"]); dev = next(m.parameters()).device
        _M.update(m=m, dev=dev, inv=build_layer_inv_freqs(m, device=dev), lt=get_layer_types(m),
                  slim=get_sliding_cache_limit(m), nl=tok.encode("\n", add_special_tokens=False),
                  bos=(tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id))
        print(f"  loaded {time.time()-t0:.0f}s")
        ck = RESULTS / mk / "results.json"; (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        skey = f"banktype_{mk}" + ("_smoke" if SMOKE else ""); scored = []
        if ck.exists():
            prev = json.loads(ck.read_text())
            if prev.get("scoring_key") == skey: scored = prev["samples"]
        for i in range(len(scored), N):
            D = tok.encode(fillers[i], add_special_tokens=False)[:160]
            sp = specs_for(i); rec = {}
            for typ in TYPES:
                F, Q, A = sp[typ]
                Fi = tok.encode(F, add_special_tokens=False); Qi = tok.encode(Q, add_special_tokens=False)
                Ai = tok.encode(A, add_special_tokens=False); neutral = (_M["nl"]*len(Fi))[:len(Fi)]
                c, cl = build("bare", D, None);     rec[f"{typ}_bare"]   = answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("strip", D, neutral); rec[f"{typ}_neutral"]= answer_nll(c, cl, Qi, Ai); del c
                c, cl = build("strip", D, Fi);      rec[f"{typ}_strip"]  = answer_nll(c, cl, Qi, Ai); del c
                torch.cuda.empty_cache()
            scored.append(rec)
            if (i+1) % 20 == 0 or SMOKE:
                ck.write_text(json.dumps({"scoring_key": skey, "samples": scored}))
                A = {k: np.mean([s[k] for s in scored]) for k in rec}
                msg = "  ".join(f"{t}={A[f'{t}_strip']-A[f'{t}_neutral']:+.2f}" for t in TYPES)
                print(f"    [{i+1}/{N}] bank(content): {msg}")
        ck.write_text(json.dumps({"scoring_key": skey, "samples": scored})); print(f"  done {len(scored)}")
        del m, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"]); _M.clear()
    print(f"\n{'='*56}\nDONE\n{'='*56}")


if __name__ == "__main__":
    main()
