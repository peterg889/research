#!/usr/bin/env python3
"""exp21b: coherence probe v2 — fixes a positional confound and adds the Mistral test.

AUDIT FIX (positional confound): in exp21, bare encodes the doc at positions 1..D while
primed encodes it at 18..17+D, so the measured value-perturbation includes a shared
"BOS is farther away" component that is IDENTICAL across docs and could inflate coherence
R for both models. v2 adds a POSITION-MATCHED control: `shifted` = [BOS, neutral x17, doc]
(17 newline tokens = L_MATCH + len(nl)), so doc positions match the primed encoding.
  content-only perturbation  = V_primed  - V_shifted   (prefix CONTENT effect, position-matched)
  positional-only component  = V_shifted - V_bare      (BOS-distance artifact)
We report R / pairwise-cos / content-alignment for BOTH decompositions.

FALSIFICATION TEST (Mistral): mistral_7b has mid primability (0.55) but NO contrastive
selectivity. The mechanism story (content-structured coherence => steerable => selectivity)
PREDICTS Mistral's content-only perturbation is content-ORTHOGONAL like Qwen's. If Mistral
shows Gemma-like content-structure, the mechanism story breaks.
"""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))

import json, gc
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv
from lib.cache import make_prefix
from lib.data import count_words

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
SMOKE = os.environ.get("SMOKE", "0") == "1"
N = 4 if SMOKE else 40
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
MODELS = {
    "gemma3_4b":  {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b":  {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
    "mistral_7b": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp21b_coherence2"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)


def load_msmarco(n):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1: continue
        rel = sel.index(1)
        if 5 <= count_words(pt[rel]) <= 200:
            out.append(pt[rel])
        if len(out) >= n: break
    return out


def doc_values(m, dev, ids, doc_slice):
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    vals = []
    for layer in out.past_key_values.layers:
        v = layer.values[0, :, doc_slice]
        v = v.permute(1, 0, 2).reshape(v.shape[1], -1)
        vals.append(v.mean(0).float().cpu().numpy())
    return vals


def coherence_stats(deltas, bares):
    """deltas/bares: per-layer lists of [dim] vectors across docs."""
    nL = len(deltas); R, COS, CONT = [], [], []
    for L in range(nL):
        X = np.stack(deltas[L]); B = np.stack(bares[L])
        norms = np.linalg.norm(X, axis=1)
        R.append(float(np.linalg.norm(X.mean(0)) / (norms.mean() + 1e-9)))
        Xn = X / (norms[:, None] + 1e-9)
        G = Xn @ Xn.T; iu = np.triu_indices(len(X), 1)
        COS.append(float(G[iu].mean()))
        bn = np.linalg.norm(B, axis=1)
        CONT.append(float(((X * B).sum(1) / (norms * bn + 1e-9)).mean()))
    return float(np.mean(R)), float(np.mean(COS)), float(np.mean(CONT))


def main():
    print(f"COHERENCE v2 (position-matched + Mistral)  SMOKE={SMOKE}  N={N}")
    passages = load_msmarco(N)
    out = {}
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        tok = AutoTokenizer.from_pretrained(spec["name"], token=HF_TOKEN)
        if spec["loader"] == "Gemma3ForConditionalGeneration":
            from transformers import Gemma3ForConditionalGeneration as Loader
        else:
            Loader = AutoModelForCausalLM
        m = Loader.from_pretrained(spec["name"], dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda:0").eval()
        dev = next(m.parameters()).device
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
        nl = tok.encode("\n", add_special_tokens=False)
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        pad_len = L_MATCH + len(nl)
        neutral = (nl * pad_len)[:pad_len]            # newline filler, same length as prefix+sep
        D_content, D_pos, BARES = None, None, None
        for ptext in passages:
            doc = tok.encode(ptext, add_special_tokens=False)[:200]; D = len(doc)
            vb = doc_values(m, dev, [bos] + doc, slice(1, 1 + D))
            vs = doc_values(m, dev, [bos] + neutral + doc, slice(1 + pad_len, 1 + pad_len + D))
            vp = doc_values(m, dev, [bos] + list(ext) + nl + doc, slice(1 + pad_len, 1 + pad_len + D))
            nL = len(vb)
            if D_content is None:
                D_content = [[] for _ in range(nL)]; D_pos = [[] for _ in range(nL)]; BARES = [[] for _ in range(nL)]
            for L in range(nL):
                D_content[L].append(vp[L] - vs[L])    # prefix-content effect, position-matched
                D_pos[L].append(vs[L] - vb[L])        # BOS-distance artifact
                BARES[L].append(vb[L])
            torch.cuda.empty_cache()
        rc = coherence_stats(D_content, BARES); rp = coherence_stats(D_pos, BARES)
        out[mk] = {"content_only": {"R": rc[0], "pairwise_cos": rc[1], "content_align": rc[2]},
                   "positional_only": {"R": rp[0], "pairwise_cos": rp[1], "content_align": rp[2]},
                   "random_R": float(1/np.sqrt(N))}
        print(f"  CONTENT-only (primed-shifted): R={rc[0]:.3f} cos={rc[1]:+.3f} content_align={rc[2]:+.3f}")
        print(f"  POSITIONAL-only (shifted-bare): R={rp[0]:.3f} cos={rp[1]:+.3f} content_align={rp[2]:+.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== VERDICT (random R~{1/np.sqrt(N):.2f}) ===")
    for mk, r in out.items():
        c = r["content_only"]
        print(f"  {mk:12s} content-only: R={c['R']:.3f} cos={c['pairwise_cos']:+.3f} align={c['content_align']:+.3f}")
    print("  Mechanism predicts: Gemma content-structured (|align| large), Qwen AND Mistral ~orthogonal.")


if __name__ == "__main__":
    main()
