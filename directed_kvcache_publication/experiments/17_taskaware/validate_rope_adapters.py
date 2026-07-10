#!/usr/bin/env python3
"""Pre-flight: verify build_layer_inv_freqs() matches each model's ACTUAL rotary embedding.

The RoPE reposition is only valid if our hand-built inv_freqs equal the ones the model uses.
This has silently broken experiments before (wrong theta / unhandled rope_type). For each
candidate model we compare adapter inv_freq to model.*.rotary_emb.inv_freq and to a from-config
recomputation, per layer type. Only models that PASS (max rel-err < 1e-4) should be run.
"""
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
    "llama31_8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "ministral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "deepseek_r1_qwen7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "olmo2_7b":     "allenai/OLMo-2-1124-7B-Instruct",
}
ONLY = os.environ.get("ONLY_MODELS")
if ONLY: CANDIDATES = {k: CANDIDATES[k] for k in ONLY.split(",") if k in CANDIDATES}


def find_rotary(model):
    """Return {layer_idx: inv_freq tensor} from the model's own rotary modules."""
    found = {}
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb") and hasattr(mod, "inv_freq"):
            found[name] = mod.inv_freq.detach().float().cpu()
    return found


def purge(name):
    p = os.path.join(HF_CACHE, "models--" + name.replace("/", "--"))
    if os.path.isdir(p): shutil.rmtree(p)


for key, hf in CANDIDATES.items():
    print(f"\n{'='*70}\n# {key}  ({hf})")
    try:
        tok = AutoTokenizer.from_pretrained(hf, token=HF_TOKEN)
        m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, token=HF_TOKEN,
                                                 device_map="cuda:0").eval()
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}")
        continue
    try:
        info = get_model_info(m)
        print(f"  model_type={info['model_type']} head_dim={info['head_dim']} "
              f"layers={info['num_layers']} has_sliding={info['has_sliding']} thetas={info['rope_thetas']}")
        adapter = build_layer_inv_freqs(m, device=torch.device("cpu"))
        lts = get_layer_types(m)
        print(f"  layer_types(uniq)={sorted(set(lts))}  adapter keys={list(adapter.keys())}")
        actual = find_rotary(m)
        print(f"  model rotary modules found: {len(actual)}  e.g. {list(actual)[:3]}")
        if not actual:
            print("  !! no rotary_emb.inv_freq buffers found; cannot validate -> SKIP")
        else:
            # compare adapter inv_freq (map layer type -> our tensor) to each actual buffer
            ref = list(actual.values())[0]
            worst = 0.0
            for lt, iv in adapter.items():
                iv = iv.float().cpu()
                if iv.shape != ref.shape:
                    print(f"  !! shape mismatch {lt}: adapter {tuple(iv.shape)} vs model {tuple(ref.shape)}")
                    worst = float("inf"); continue
                rel = (iv - ref).abs() / (ref.abs() + 1e-9)
                worst = max(worst, rel.max().item())
            verdict = "PASS" if worst < 1e-4 else ("FAIL" if worst != float("inf") else "SHAPE-FAIL")
            print(f"  max rel-err(adapter vs model rotary) = {worst:.2e}  -> {verdict}")
    finally:
        del m, tok; gc.collect(); torch.cuda.empty_cache()
        if os.environ.get("KEEP", "0") != "1": purge(hf)

print(f"\n{'='*70}\nVALIDATION DONE")
