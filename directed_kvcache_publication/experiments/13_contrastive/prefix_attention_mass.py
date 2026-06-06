#!/usr/bin/env python3
"""WS3 clincher: WHY is Gemma more primable, and why do the families respond OPPOSITELY
to attention temperature (Gemma: sharper->more primable; Qwen: softer->more primable)?

Hypothesis: in Gemma the prepended prefix is a HIGH-SALIENCE key, so doc tokens attend
to it a lot (peaking attention -> concentrates ONTO prefix -> more primable). In Qwen the
prefix is minor (only diffuse/soft attention picks it up). Test directly: measure the
PREFIX-ATTENTION-MASS = fraction of doc-token attention that lands on the prefix tokens
during Phase-A encoding, vs the uniform baseline (L/total). Also BOS (sink) mass.

Eager attention + output_attentions on ~20 MS MARCO passages, gemma3_4b vs qwen25_7b.
Prediction: Gemma prefix-mass >> uniform and >> Qwen's.
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
N = 4 if SMOKE else 20
L_MATCH = 16
EXTRACT = "Extract the key facts from this text."
MODELS = {
    "gemma3_4b": {"name": "google/gemma-3-4b-it", "loader": "Gemma3ForConditionalGeneration"},
    "qwen25_7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "loader": "AutoModelForCausalLM"},
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: MODELS = {k: MODELS[k] for k in ONLY.split(",") if k in MODELS}
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp19_prefixmass"
RESULTS.mkdir(parents=True, exist_ok=True, mode=0o777)


def load_msmarco(n):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    out = []
    for x in ds:
        pt = x["passages"]["passage_text"]; sel = x["passages"]["is_selected"]
        if sum(sel) != 1: continue
        rel = sel.index(1)
        if 5 <= count_words(pt[rel]) <= 250:
            out.append(pt[rel])
        if len(out) >= n: break
    return out


def main():
    print(f"PREFIX-ATTENTION-MASS  SMOKE={SMOKE}  N={N}")
    passages = load_msmarco(N)
    out = {}
    for mk, spec in MODELS.items():
        print(f"\n# {mk}")
        tok = AutoTokenizer.from_pretrained(spec["name"], token=HF_TOKEN)
        if spec["loader"] == "Gemma3ForConditionalGeneration":
            from transformers import Gemma3ForConditionalGeneration as Loader
        else:
            Loader = AutoModelForCausalLM
        m = Loader.from_pretrained(spec["name"], dtype=torch.bfloat16, token=HF_TOKEN,
                                   device_map="cuda:0", attn_implementation="eager").eval()
        dev = next(m.parameters()).device
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
        nl = tok.encode("\n", add_special_tokens=False)
        ext = make_prefix(tok.encode(EXTRACT, add_special_tokens=False), L_MATCH)
        pmass, bmass, unif, ents = [], [], [], []
        for ptext in passages:
            doc = tok.encode(ptext, add_special_tokens=False)[:200]
            ids = [bos] + list(ext) + nl + list(doc)
            p0, p1 = 1, 1 + L_MATCH                     # prefix key positions
            d0 = 1 + L_MATCH + len(nl)                  # doc query positions start
            with torch.no_grad():
                o = m(input_ids=torch.tensor([ids], device=dev), output_attentions=True, use_cache=False)
            T = len(ids)
            for att in o.attentions:                    # [1, H, T, T] per layer
                a = att[0].float()                      # H,T,T
                docq = a[:, d0:, :]                      # H, n_doc, T  (doc queries)
                pf = docq[:, :, p0:p1].sum(-1)          # mass on prefix: H, n_doc
                bo = docq[:, :, 0]                      # mass on BOS
                pmass.append(pf.mean().item()); bmass.append(bo.mean().item())
                # attention entropy of doc queries (nats), averaged
                pr = docq.clamp_min(1e-9)
                ents.append((-(pr * pr.log()).sum(-1)).mean().item())
            unif.append(L_MATCH / T)
            del o; torch.cuda.empty_cache()
        out[mk] = {"prefix_mass": float(np.mean(pmass)), "bos_mass": float(np.mean(bmass)),
                   "uniform_prefix_mass": float(np.mean(unif)),
                   "prefix_over_uniform": float(np.mean(pmass) / np.mean(unif)),
                   "doc_attn_entropy": float(np.mean(ents))}
        print(f"  prefix_mass={out[mk]['prefix_mass']:.4f}  uniform={out[mk]['uniform_prefix_mass']:.4f}  "
              f"over_uniform={out[mk]['prefix_over_uniform']:.2f}x  bos_mass={out[mk]['bos_mass']:.3f}  "
              f"doc_attn_entropy={out[mk]['doc_attn_entropy']:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    (RESULTS / "result.json").write_text(json.dumps(out, indent=2))
    print("\n=== PREFIX SALIENCE (does Gemma attend to the prefix more?) ===")
    for mk, r in out.items():
        print(f"  {mk:12s} prefix_mass={r['prefix_mass']:.4f} ({r['prefix_over_uniform']:.2f}x uniform)  "
              f"BOS_sink={r['bos_mass']:.3f}  entropy={r['doc_attn_entropy']:.2f}")
    print("  Prediction: Gemma prefix_mass / over_uniform >> Qwen => prefix is salient in Gemma.")


if __name__ == "__main__":
    main()
