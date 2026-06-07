#!/usr/bin/env python3
"""WS3 deeper: measure DIRECTIONAL COHERENCE directly in representation space.

The patching showed Gemma's prefix-induced KV perturbation has a CONSISTENT signed
effect on query-NLL (steerable) while Qwen's cancels. Test this at the representation
level: do the per-doc perturbation vectors actually point the SAME WAY across documents?

Use VALUE vectors (no RoPE -> frame-free). For each doc d, delta_d^L = mean over doc
tokens of (V_primed - V_bare) at layer L. Across docs:
  coherence R_L = ||mean_d delta_d|| / mean_d ||delta_d||   (1=all aligned, ~1/sqrt(N)=random)
  cos_L = mean pairwise cosine of {delta_d}
Also: content-alignment = mean_d cos(delta_d, bare_value_d) — does the perturbation
amplify the doc's own content direction (steerable) vs a fixed offset?
Prediction: Gemma R/cos >> Qwen (representation-level proof of directional coherence).
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
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp21_coherence"
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
    """Return list over layers of mean doc-token VALUE vector [n_kv*hd] (no RoPE on V)."""
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    vals = []
    for layer in out.past_key_values.layers:
        v = layer.values[0, :, doc_slice]            # [n_kv, D, hd]
        v = v.permute(1, 0, 2).reshape(v.shape[1], -1)  # [D, n_kv*hd]
        vals.append(v.mean(0).float().cpu().numpy())   # [n_kv*hd]
    return vals


def main():
    print(f"COHERENCE PROBE  SMOKE={SMOKE}  N={N}")
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
        deltas, bares = None, None
        for ptext in passages:
            doc = tok.encode(ptext, add_special_tokens=False)[:200]; D = len(doc)
            bare_ids = [bos] + doc
            prim_ids = [bos] + list(ext) + nl + doc
            vb = doc_values(m, dev, bare_ids, slice(1, 1 + D))
            vp = doc_values(m, dev, prim_ids, slice(1 + L_MATCH + len(nl), 1 + L_MATCH + len(nl) + D))
            nLrs = len(vb)
            if deltas is None: deltas = [[] for _ in range(nLrs)]; bares = [[] for _ in range(nLrs)]
            for L in range(nLrs):
                deltas[L].append(vp[L] - vb[L]); bares[L].append(vb[L])
            torch.cuda.empty_cache()
        nLrs = len(deltas)
        R, COS, CONT = [], [], []
        for L in range(nLrs):
            X = np.stack(deltas[L])                    # [N, dim]
            B = np.stack(bares[L])
            norms = np.linalg.norm(X, axis=1)
            R.append(float(np.linalg.norm(X.mean(0)) / (norms.mean() + 1e-9)))
            Xn = X / (norms[:, None] + 1e-9)
            G = Xn @ Xn.T; iu = np.triu_indices(len(X), 1)
            COS.append(float(G[iu].mean()))
            # content alignment: cos(delta_d, bare_d)
            bn = np.linalg.norm(B, axis=1)
            CONT.append(float(((X * B).sum(1) / (norms * bn + 1e-9)).mean()))
        out[mk] = {"n_layers": nLrs, "R": R, "pairwise_cos": COS, "content_align": CONT,
                   "R_mean": float(np.mean(R)), "cos_mean": float(np.mean(COS)),
                   "content_align_mean": float(np.mean(CONT)), "random_baseline_R": float(1/np.sqrt(N))}
        print(f"  layers={nLrs}  R_mean={np.mean(R):.3f}  pairwise_cos_mean={np.mean(COS):.3f}  "
              f"content_align_mean={np.mean(CONT):+.3f}  (random R~{1/np.sqrt(N):.3f})")
        print(f"  R by layer-decile:   " + " ".join(f"{R[int(q*(nLrs-1))]:.2f}" for q in np.linspace(0,1,6)))
        del m; gc.collect(); torch.cuda.empty_cache()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print("\n=== DIRECTIONAL COHERENCE (representation-level) ===")
    rb = 1/np.sqrt(N)
    for mk, r in out.items():
        print(f"  {mk:12s} R={r['R_mean']:.3f} (random~{rb:.2f})  pairwise_cos={r['cos_mean']:+.3f}  "
              f"content_align={r['content_align_mean']:+.3f}")
    print("  Prediction: Gemma R/cos >> Qwen => prefix induces a coherent, steerable direction in Gemma.")


if __name__ == "__main__":
    main()
