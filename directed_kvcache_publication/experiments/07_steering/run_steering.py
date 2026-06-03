#!/usr/bin/env python3
"""PHASE 1: activation-space cache priming (difference-of-means steering vector).

Idea: the 'extract' prefix does TWO things at once — (a) it improves document
representations during encoding, and (b) it biases the model toward verbose answer
generation (which hurts EM, per Phase 0). A steering VECTOR reproduces only (a):
it adds the residual-stream offset the prefix induced, with NO prefix tokens in
context to trigger (b). So steering can capture the discrimination benefit without
the generation-style cost.

Method
------
1. EXTRACT a per-layer priming direction from a TRAIN split, position-controlled:
   v_l = mean over (train docs, doc positions) of [ h_l(extract@shifted) - h_l(bare@shifted) ]
   where both encodings place doc tokens at the SAME shifted positions, so the only
   difference is the presence of the extract prefix in the attention context (the
   position-shift component is differenced out). Residual stream carries no RoPE, so
   the offset is position-agnostic and transfers to natural-position encoding.

2. APPLY via forward hooks: when encoding a NEW doc as bare [BOS, doc] at natural
   positions 1..D, add alpha * v_l to the residual at doc positions after each layer.
   No prefix -> no select, no reposition, no bf16 reposition error, no verbosity bias.

3. SCORE the contrastive margin (correct vs K type-matched distractors, as exp05) for
   bare / extract-prompt / steered(alpha sweep), on HELD-OUT test docs.

Internal validations (must pass in smoke):
  V1  alpha=0 steered margin == bare margin exactly (hook adds zero).
  V2  reconstruction: applying v at alpha=1 to bare@shifted reproduces extract@shifted
      doc residuals on train docs (cosine ~1, small residual).
  V3  the real test: does steered (natural pos) margin >= extract-prompt margin, and
      does it generalize to held-out docs?

v1 scope: Qwen/Mistral/Ministral family (standard CausalLM, model.model.layers).
Gemma3 added after Qwen validates (different module path).

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    SMOKE=1 PYTHONPATH="../directed_kvcache_v4:." python3 experiments/07_steering/run_steering.py
    PYTHONPATH="../directed_kvcache_v4:." python3 experiments/07_steering/run_steering.py
"""

import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc, re, shutil, time, string
import random as pyrandom
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv

from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from lib.data import count_words
from model_adapters import get_sliding_cache_limit

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 20 if SMOKE else 150     # docs used to estimate the steering vector
N_TEST  = 10 if SMOKE else 150     # held-out docs for evaluation
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16
ALPHAS = [0.0, 1.0, 2.0, 4.0] if SMOKE else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
EXTRACT = "Extract the key facts from this text."
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp08_steering"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)

# v1: standard-CausalLM family only (model.model.layers; clean output_hidden_states)
MODELS = {
    "qwen25_7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "loader": "AutoModelForCausalLM"},
    "ministral_8b":{"name": "mistralai/Ministral-8B-Instruct-2410", "loader": "AutoModelForCausalLM"},
}
if SMOKE:
    MODELS = {"qwen25_1_5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "loader": "AutoModelForCausalLM"}}
DATASETS = ["squad_v2", "hotpotqa"] if not SMOKE else ["squad_v2"]


# ---------------- distractor / margin scoring (mirrors exp05) ----------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def cohens_d(a):
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) + 1e-12) if len(a) > 1 else 0.0


def load_split(ds_key, n_total):
    """Load n_total samples for a dataset (same seeds/filters as exp05)."""
    if ds_key == "squad_v2":
        ds = load_dataset("rajpurkar/squad_v2", split="validation"); seed = SEED + 200
        cand = []
        for it in ds:
            p = it.get("context", ""); q = it.get("question", "")
            ans = it.get("answers", {}).get("text", [])
            a = ans[0] if ans else ""
            if p and q and a and 30 <= count_words(p) <= 500:
                cand.append({"passage": p, "query": q, "answer": a})
    elif ds_key == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); seed = SEED + 400
        cand = []
        for it in ds:
            ctx = it.get("context", {}); sf = it.get("supporting_facts", {})
            t2s = {t: s for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))}
            parts = [t2s[t][sid] for t, sid in zip(sf.get("title", []), sf.get("sent_id", []))
                     if t in t2s and sid < len(t2s[t])]
            if not parts: continue
            p = " ".join(parts)
            if 30 <= count_words(p) <= 500 and count_words(it["answer"]) >= 1:
                cand.append({"passage": p, "query": it["question"], "answer": it["answer"]})
    else:
        raise ValueError(ds_key)
    pyrandom.seed(seed); pyrandom.shuffle(cand)
    return cand[:n_total]


def build_distractor_pool(samples):
    pool = [s["answer"] for s in samples]
    def _type(a):
        a = a.strip(); isnum = bool(a) and (a[0].isdigit() or (a[0] == "-" and len(a) > 1 and a[1].isdigit()))
        nt = len(a.split()); lb = 0 if nt <= 1 else (1 if nt <= 3 else 2)
        return (isnum, lb)
    tindex = {}
    for j, a in enumerate(pool):
        tindex.setdefault(_type(a), []).append(j)
    def pick(idx, correct):
        bad = {normalize_answer(correct)}
        bucket = tindex.get(_type(correct), [])
        cands = bucket if len(bucket) > K_DISTRACT * 3 else list(range(len(pool)))
        rng = pyrandom.Random(SEED + 7000 + idx); order = cands[:]; rng.shuffle(order)
        out = []
        for j in order:
            if j == idx: continue
            nc = normalize_answer(pool[j])
            if nc in bad or not nc: continue
            out.append(pool[j]); bad.add(nc)
            if len(out) >= K_DISTRACT: break
        return out
    return pick


# ---------------- model + layer access ----------------
def load_model(name, loader):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
    return m, tok

def get_layers(model):
    # standard CausalLM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("could not locate decoder layers (v1 supports standard CausalLM only)")

def purge(name):
    p = os.path.join(HF_CACHE_DIR, "models--" + name.replace("/", "--"))
    if os.path.isdir(p):
        shutil.rmtree(p)


# ---------------- steering state + hooks ----------------
class Steerer:
    """Adds alpha * v[layer] to residual at given positions, via forward hooks."""
    def __init__(self, model, vecs):
        self.layers = get_layers(model)
        self.vecs = vecs              # tensor [n_layers, hidden], on device, float
        self.alpha = 0.0
        self.positions = None         # 1-D LongTensor of seq positions to steer (doc tokens)
        self.handles = []

    def _mk_hook(self, li):
        def hook(module, inp, out):
            if self.alpha == 0.0 or self.positions is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            v = (self.alpha * self.vecs[li]).to(hs.dtype)
            hs[:, self.positions, :] = hs[:, self.positions, :] + v
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs
        return hook

    def attach(self):
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._mk_hook(li)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------- steering-vector extraction ----------------
def extract_steering_vector(model, tok, dev, train_samples, bos, nl, max_doc, n_layers, hidden):
    """v_l = mean over train docs & doc positions of [h_l(extract@shifted) - h_l(bare@shifted)],
    position-controlled so only the prefix presence differs."""
    extract_ids = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
    P = len(extract_ids); NL = len(nl); shift = P + NL
    acc = torch.zeros(n_layers, hidden, dtype=torch.float32, device=dev)
    count = 0
    for s in train_samples:
        doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
        D = len(doc_ids)
        # extract encoding: [BOS, prefix, NL, doc]; doc at positions (1+shift)..(shift+D)
        ext_ids = [bos] + list(extract_ids) + nl + list(doc_ids)
        # bare@shifted: [BOS, doc] with doc forced to the SAME positions via position_ids
        bare_ids = [bos] + list(doc_ids)
        bare_pos = torch.cat([torch.tensor([0], device=dev),
                              torch.arange(shift + 1, shift + 1 + D, device=dev)]).unsqueeze(0)
        with torch.no_grad():
            ext = model(input_ids=torch.tensor([ext_ids], device=dev),
                        output_hidden_states=True, use_cache=False).hidden_states
            bar = model(input_ids=torch.tensor([bare_ids], device=dev), position_ids=bare_pos,
                        output_hidden_states=True, use_cache=False).hidden_states
        doc_start_ext = 1 + P + NL
        for li in range(n_layers):
            he = ext[li + 1][0, doc_start_ext:doc_start_ext + D, :].float()   # layer li output
            hb = bar[li + 1][0, 1:1 + D, :].float()
            acc[li] += (he - hb).sum(dim=0)
        count += D
    return acc / max(count, 1)


# ---------------- cache build + margin ----------------
def build_bare_cache(model, dev, doc_ids, bos, nl):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[bos] + list(doc_ids)], device=dev), use_cache=True)
    cache = out.past_key_values; del out
    return norm_roundtrip_kv_cache(cache), len(doc_ids)

def build_prefixed_cache(model, dev, doc_ids, prefix_ids, inv_freqs, lt, bos, nl):
    from lib.rope import select_kv_cache, reposition_kv_cache
    ids = [bos] + list(prefix_ids) + nl + list(doc_ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache = out.past_key_values; del out
    doc_start = 1 + len(prefix_ids) + len(nl); D = len(doc_ids)
    cache = select_kv_cache(cache, [0] + list(range(doc_start, doc_start + D)), device=dev)
    cache = reposition_kv_cache(cache, torch.arange(doc_start, doc_start + D, device=dev),
                                torch.arange(1, 1 + D, device=dev), inv_freqs, lt, bos_start=0)
    return norm_roundtrip_kv_cache(cache), D

def score_margin(model, dev, cache, D, query_ids, correct_ids, distractor_ids, nl):
    def cand_nll(cand):
        pb = nl + list(query_ids) + nl + list(cand)
        pos = torch.arange(D + 1, D + 1 + len(pb), device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([pb], device=dev), position_ids=pos,
                        past_key_values=deep_copy_cache(cache), use_cache=False)
        a0 = len(nl) + len(query_ids) + len(nl)
        al = out.logits[0][a0 - 1:a0 - 1 + len(cand)]
        return torch.nn.functional.cross_entropy(al, torch.tensor(cand, device=dev)).item()
    c = cand_nll(correct_ids)
    ds = [cand_nll(d) for d in distractor_ids]
    return float(np.mean(ds) - c), c


def main():
    print(f"PHASE 1 STEERING  SMOKE={SMOKE}  N_TRAIN={N_TRAIN} N_TEST={N_TEST} alphas={ALPHAS}")
    for mk, spec in MODELS.items():
        print(f"\n{'#'*60}\n# {mk}\n{'#'*60}")
        t0 = time.time()
        model, tok = load_model(spec["name"], spec["loader"])
        dev = next(model.parameters()).device
        from model_adapters import build_layer_inv_freqs, get_layer_types
        inv_freqs = build_layer_inv_freqs(model, device=dev); lt = get_layer_types(model)
        slim = get_sliding_cache_limit(model)
        nl = tok.encode("\n", add_special_tokens=False)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
        max_doc = (slim - 1 - 64 - len(nl)) if slim is not None else 765
        layers = get_layers(model); n_layers = len(layers)
        hidden = model.config.hidden_size
        extract_pfx = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        print(f"  loaded {n_layers} layers, hidden={hidden} in {time.time()-t0:.0f}s")

        out_rows = {}
        for dk in DATASETS:
            train = load_split(dk, N_TRAIN + N_TEST)
            train_docs, test_docs = train[:N_TRAIN], train[N_TRAIN:N_TRAIN + N_TEST]
            pick = build_distractor_pool(train)  # pool over all loaded for type-matching

            print(f"\n  [{dk}] extracting steering vector from {len(train_docs)} docs...")
            vecs = extract_steering_vector(model, tok, dev, train_docs, bos, nl, max_doc, n_layers, hidden)
            print(f"    vec per-layer norms: min={vecs.norm(dim=1).min():.2f} "
                  f"max={vecs.norm(dim=1).max():.2f} mean={vecs.norm(dim=1).mean():.2f}")

            steerer = Steerer(model, vecs); steerer.attach()
            rows = []
            for i, s in enumerate(test_docs):
                global_idx = N_TRAIN + i
                doc_ids = tok.encode(s["passage"], add_special_tokens=False)[:max_doc]
                q_ids = tok.encode(s["query"], add_special_tokens=False)
                c_ids = tok.encode(s["answer"], add_special_tokens=False)
                if not c_ids: continue
                D = len(doc_ids)
                distract = pick(global_idx, s["answer"])
                d_ids = [x for x in (tok.encode(d, add_special_tokens=False) for d in distract) if x]
                rec = {}
                # bare (steerer off)
                steerer.alpha = 0.0; steerer.positions = None
                cache, D = build_bare_cache(model, dev, doc_ids, bos, nl)
                rec["bare__margin"], rec["bare__nll"] = score_margin(model, dev, cache, D, q_ids, c_ids, d_ids, nl); del cache
                # extract prompt
                cache, D = build_prefixed_cache(model, dev, doc_ids, extract_pfx, inv_freqs, lt, bos, nl)
                rec["extract__margin"], rec["extract__nll"] = score_margin(model, dev, cache, D, q_ids, c_ids, d_ids, nl); del cache
                # steered: encode bare WITH hook adding alpha*v at doc positions 1..D
                doc_positions = torch.arange(1, 1 + D, device=dev)
                for a in ALPHAS:
                    steerer.alpha = a; steerer.positions = doc_positions
                    with torch.no_grad():
                        o = model(input_ids=torch.tensor([[bos] + list(doc_ids)], device=dev), use_cache=True)
                    steerer.alpha = 0.0; steerer.positions = None  # off for Phase B scoring
                    cache = norm_roundtrip_kv_cache(o.past_key_values); del o
                    m, n = score_margin(model, dev, cache, D, q_ids, c_ids, d_ids, nl); del cache
                    rec[f"steer{a}__margin"], rec[f"steer{a}__nll"] = m, n
                    torch.cuda.empty_cache()
                rows.append(rec)
            steerer.detach()
            out_rows[dk] = rows

            # quick report
            def dmean(key): return np.mean([r[key] for r in rows])
            print(f"    [{dk}] n={len(rows)}  margin: bare={dmean('bare__margin'):+.3f} "
                  f"extract={dmean('extract__margin'):+.3f}")
            for a in ALPHAS:
                dm = [r[f'steer{a}__margin'] - r['bare__margin'] for r in rows]
                print(f"      steer α={a}: margin={dmean(f'steer{a}__margin'):+.3f}  "
                      f"d(vs bare)={cohens_d(dm):+.3f}")
            ed = [r['extract__margin'] - r['bare__margin'] for r in rows]
            print(f"      extract: d(vs bare)={cohens_d(ed):+.3f}")

        (RESULTS / mk).mkdir(exist_ok=True, mode=0o777)
        (RESULTS / mk / "steering_results.json").write_text(json.dumps(out_rows, default=float))
        del model, tok; gc.collect(); torch.cuda.empty_cache(); purge(spec["name"])

    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
