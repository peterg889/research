#!/usr/bin/env python3
"""Round 2: validate RoPE adapters for candidate NON-GEMMA families, hunting a token-presence
imprinter (banks with magnitude but order-invariant). Only models that PASS (rel-err<1e-4) are run.
Note: Llama-3.1/3.2 use rope_type='llama3' (scaled freqs) which the adapter does NOT handle -> they
will FAIL validation; we use Llama-3 (plain rope_theta) via an ungated mirror instead."""
import os
os.umask(0o000)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../directed_kvcache_v4"))
import gc, shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv, find_dotenv
from model_adapters import build_layer_inv_freqs, get_layer_types, get_model_info

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

CANDIDATES = {
    "llama3_8b":  "NousResearch/Meta-Llama-3-8B-Instruct",  # Llama-3, plain rope theta=5e5 (no llama3 scaling)
    "yi15_9b":    "01-ai/Yi-1.5-9B-Chat",                   # Llama arch, rope theta=5e6
    "falcon3_7b": "tiiuae/Falcon3-7B-Instruct",             # llama-family arch
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",        # phi3 (partial rotary? -> may fail shape)
    "internlm25_7b": "internlm/internlm2_5-7b-chat",        # internlm2 (may be unsupported)
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: CANDIDATES = {k: CANDIDATES[k] for k in ONLY.split(",") if k in CANDIDATES}


def find_rotary(model):
    out = {}
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb") and hasattr(mod, "inv_freq"):
            out[name] = mod.inv_freq.detach().float().cpu()
    return out

def purge(name):
    p = os.path.join(HF_CACHE, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


for key, hf in CANDIDATES.items():
    print(f"\n{'='*70}\n# {key}  ({hf})")
    try:
        tok = AutoTokenizer.from_pretrained(hf, token=HF_TOKEN, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, token=HF_TOKEN,
                                                 device_map="cuda:0", trust_remote_code=True).eval()
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}"); continue
    try:
        info = get_model_info(m)
        print(f"  model_type={info['model_type']} head_dim={info['head_dim']} layers={info['num_layers']} "
              f"has_sliding={info['has_sliding']} thetas={info['rope_thetas']}")
        adapter = build_layer_inv_freqs(m, device=torch.device("cpu"))
        actual = find_rotary(m)
        print(f"  adapter keys={list(adapter.keys())}  model rotary modules={len(actual)}")
        if not actual:
            print("  !! no rotary_emb.inv_freq found -> cannot validate -> SKIP")
        else:
            ref = list(actual.values())[0]; worst = 0.0
            for lt, iv in adapter.items():
                iv = iv.float().cpu()
                if iv.shape != ref.shape:
                    print(f"  !! shape mismatch {lt}: {tuple(iv.shape)} vs {tuple(ref.shape)}"); worst = float("inf"); continue
                worst = max(worst, ((iv - ref).abs() / (ref.abs() + 1e-9)).max().item())
            v = "PASS" if worst < 1e-4 else ("FAIL" if worst != float("inf") else "SHAPE-FAIL")
            print(f"  max rel-err = {worst:.2e}  -> {v}")
    finally:
        del m, tok; gc.collect(); torch.cuda.empty_cache()
        if os.environ.get("KEEP", "0") != "1": purge(hf)

print(f"\n{'='*70}\nVALIDATION DONE")
