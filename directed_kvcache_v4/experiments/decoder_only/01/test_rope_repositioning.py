"""Test RoPE repositioning against native Gemma 3 implementation.

Verifies that our reposition_kv_cache() correctly shifts cached keys to
new RoPE positions, matching what the model would produce natively.

Usage: python test_rope_repositioning.py
"""
import os, sys, gc
os.umask(0o000)
sys.path.insert(0, "../../..")

import torch
import torch.nn.functional as F
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_NAME = "google/gemma-3-12b-it"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = model.config.text_config
head_dim = text_cfg.head_dim
n_layers = text_cfg.num_hidden_layers
global_idx = [i for i, t in enumerate(text_cfg.layer_types)
              if t == 'full_attention'][0]

# ---- Functions under test (copied from notebook) ----

def slice_kv_cache(cache, start_idx):
    sliced = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, start_idx:, :]
        v = cache.layers[i].values[:, :, start_idx:, :]
        sliced.update(k, v, i)
    return sliced

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def build_layer_inv_freqs(config, head_dim, device):
    text_cfg = getattr(config, 'text_config', config)
    layer_types = text_cfg.layer_types
    rope_params = text_cfg.rope_parameters
    type_inv_freqs = {}
    for ltype, params in rope_params.items():
        theta = float(params['rope_theta'])
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim))
        if params.get('rope_type') == 'linear':
            inv_freq = inv_freq / float(params['factor'])
        type_inv_freqs[ltype] = inv_freq
    return [type_inv_freqs[layer_types[i]] for i in range(len(layer_types))]

def reposition_kv_cache(cache, delta, layer_inv_freqs):
    rotation_cache = {}
    repositioned = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        freq_id = id(layer_inv_freqs[i])
        if freq_id not in rotation_cache:
            inv_freq = layer_inv_freqs[i]
            angles = delta * inv_freq
            emb = torch.cat([angles, angles])
            rotation_cache[freq_id] = (
                emb.cos().view(1, 1, 1, -1),
                emb.sin().view(1, 1, 1, -1))
        cos_d, sin_d = rotation_cache[freq_id]
        k_f = k.float()
        k_new = (k_f * cos_d + rotate_half(k_f) * sin_d).to(k.dtype)
        repositioned.update(k_new, v, i)
    return repositioned

def score(doc_text, query_text, answer_text, prefix_text=None):
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)
    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               truncation=True, max_length=512).input_ids
        P = len(prefix_ids)
        NL = len(NEWLINE_IDS)
        cond_ids = [BOS_ID] + prefix_ids + NEWLINE_IDS + doc_ids
        slice_start = 1 + P + NL
        reposition_delta = -(P + NL)
    else:
        cond_ids = [BOS_ID] + doc_ids
        slice_start = 1
        reposition_delta = 0
    phase_b_start = D + 1
    with torch.no_grad():
        pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   use_cache=True)
    cache = pa.past_key_values
    del pa
    cache = slice_kv_cache(cache, slice_start)
    if reposition_delta != 0:
        cache = reposition_kv_cache(cache, reposition_delta, LAYER_INV_FREQS)
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0
    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)
    with torch.no_grad():
        pb = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache, position_ids=pos.unsqueeze(0),
                   cache_position=pos, use_cache=False)
    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del cache, pb
    return nll

LAYER_INV_FREQS = build_layer_inv_freqs(model.config, head_dim, DEVICE)

# ---- Helpers ----

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1

# BF16 relative precision: ~2^-7 = 0.0078, so roundtrip through bf16
# introduces ~0.8% relative error. Tests allow 1% margin.
BF16_REL_TOL = 0.01

# ================================================================
# TEST 1: inv_freq matches model's internal buffers
# ================================================================
print("\n=== Test 1: inv_freq matches model's rotary_emb buffers ===")
rotary = model.model.language_model.rotary_emb
for ltype in ['sliding_attention', 'full_attention']:
    model_inv = getattr(rotary, f'{ltype}_inv_freq').float().to(DEVICE)
    layer_idx = [i for i, t in enumerate(text_cfg.layer_types)
                 if t == ltype][0]
    our_inv = LAYER_INV_FREQS[layer_idx]
    diff = (model_inv - our_inv).abs().max().item()
    check(diff < 1e-6,
          f"{ltype}: inv_freq diff = {diff:.2e}")

# ================================================================
# TEST 2: attention_scaling is 1.0 for both layer types
# ================================================================
print("\n=== Test 2: attention_scaling values ===")
for ltype in ['sliding_attention', 'full_attention']:
    scaling = getattr(rotary, f'{ltype}_attention_scaling')
    check(scaling == 1.0,
          f"{ltype}: attention_scaling = {scaling}")

# ================================================================
# TEST 3: cos/sin composition identity against model's rotary_emb
# ================================================================
# R(a+b) = R(a) @ R(b). Model returns cos/sin as bf16, so expect ~bf16 error.
print("\n=== Test 3: cos/sin composition against model's rotary_emb ===")
dummy = torch.zeros(1, 1, head_dim, device=DEVICE, dtype=torch.bfloat16)
pos_ab = torch.tensor([[5, 22]], device=DEVICE)
for ltype in ['sliding_attention', 'full_attention']:
    cos_model, sin_model = rotary(dummy, pos_ab, layer_type=ltype)
    cos_5, sin_5 = cos_model[0, 0].float(), sin_model[0, 0].float()
    cos_22, sin_22 = cos_model[0, 1].float(), sin_model[0, 1].float()

    inv_f = LAYER_INV_FREQS[
        [i for i, t in enumerate(text_cfg.layer_types) if t == ltype][0]]
    angles = 17.0 * inv_f
    emb = torch.cat([angles, angles])
    our_cos = emb.cos()
    our_sin = emb.sin()

    cos_composed = cos_5 * our_cos - sin_5 * our_sin
    sin_composed = sin_5 * our_cos + cos_5 * our_sin
    cos_err = (cos_22 - cos_composed).abs().max().item()
    sin_err = (sin_22 - sin_composed).abs().max().item()
    # bf16 quantization on model cos/sin introduces ~0.4% error
    check(cos_err < BF16_REL_TOL,
          f"{ltype}: cos composition err = {cos_err:.2e} (bf16 limited)")
    check(sin_err < BF16_REL_TOL,
          f"{ltype}: sin composition err = {sin_err:.2e} (bf16 limited)")

# ================================================================
# TEST 4: Layer-0 repositioning matches native computation
# ================================================================
# At layer 0, pre-RoPE keys depend only on token identity.
# Reposition(key_at_p, delta) should match key_natively_at_{p+delta}
# up to bf16 quantization from cache storage.
print("\n=== Test 4: Layer-0 repositioning vs native keys ===")
test_toks = tokenizer("The quick brown fox jumps over the lazy dog",
                      add_special_tokens=False).input_ids
ids = [BOS_ID] + test_toks
delta = 23

with torch.no_grad():
    out_nat = model(input_ids=torch.tensor([ids], device=DEVICE), use_cache=True)
cache_nat = out_nat.past_key_values
del out_nat

pos_s = torch.arange(delta, delta + len(ids), device=DEVICE)
with torch.no_grad():
    out_s = model(input_ids=torch.tensor([ids], device=DEVICE),
                  position_ids=pos_s.unsqueeze(0),
                  cache_position=pos_s, use_cache=True)
cache_s = out_s.past_key_values
del out_s

for test_layer in [0, global_idx]:
    ltype = text_cfg.layer_types[test_layer]
    k_nat = cache_nat.layers[test_layer].keys
    k_s = cache_s.layers[test_layer].keys

    inv_f = LAYER_INV_FREQS[test_layer]
    angles = delta * inv_f
    emb = torch.cat([angles, angles])
    cos_d = emb.cos().view(1, 1, 1, -1)
    sin_d = emb.sin().view(1, 1, 1, -1)
    k_repo = (k_nat.float() * cos_d + rotate_half(k_nat.float()) * sin_d)

    rel_diff = ((k_repo - k_s.float()).abs().max().item()
                / k_s.float().abs().max().item())

    v_nat = cache_nat.layers[test_layer].values
    v_s = cache_s.layers[test_layer].values
    v_diff = (v_nat.float() - v_s.float()).abs().max().item()
    v_mag = v_nat.float().abs().max().item()

    check(rel_diff < BF16_REL_TOL,
          f"Layer {test_layer} ({ltype[:7]}): key rel diff = {rel_diff:.2e}")
    check(v_diff < 1e-6 or (v_diff / v_mag < BF16_REL_TOL),
          f"Layer {test_layer} ({ltype[:7]}): val diff = {v_diff:.2e} "
          f"(rel={v_diff/v_mag:.2e})")

print("  (Higher layers — expected to diverge due to attention context)")
for test_layer in [1, 2, 3]:
    k_nat = cache_nat.layers[test_layer].keys
    k_s = cache_s.layers[test_layer].keys
    inv_f = LAYER_INV_FREQS[test_layer]
    angles = delta * inv_f
    emb = torch.cat([angles, angles])
    cos_d = emb.cos().view(1, 1, 1, -1)
    sin_d = emb.sin().view(1, 1, 1, -1)
    k_repo = (k_nat.float() * cos_d + rotate_half(k_nat.float()) * sin_d)
    rel_diff = ((k_repo - k_s.float()).abs().max().item()
                / k_s.float().abs().max().item())
    print(f"    Layer {test_layer}: key rel diff = {rel_diff:.2e}")

del cache_nat, cache_s

# ================================================================
# TEST 5: Full pipeline — encode conditioned, slice, reposition
# ================================================================
# At layer 0, repositioned doc keys should match bare doc keys.
# Values at layer 0 may differ slightly due to GPU matmul non-determinism
# across different sequence lengths (NOT a code bug).
print("\n=== Test 5: Full pipeline (cond+slice+reposition) vs bare ===")
doc_text = "Average Temperatures in Montevideo Uruguay"
query_text = "average annual temperature of Uruguay"
doc_ids = tokenizer(doc_text, add_special_tokens=False).input_ids
prefix_ids = tokenizer(query_text, add_special_tokens=False).input_ids
P = len(prefix_ids)
NL = len(NEWLINE_IDS)
D = len(doc_ids)
print(f"  P={P}, NL={NL}, D={D}, delta={-(P+NL)}")

with torch.no_grad():
    out_bare = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                     use_cache=True)
cache_bare = out_bare.past_key_values
del out_bare

cond_ids = [BOS_ID] + prefix_ids + NEWLINE_IDS + doc_ids
with torch.no_grad():
    out_cond = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                     use_cache=True)
cache_cond = out_cond.past_key_values
del out_cond

cache_sliced = slice_kv_cache(cache_cond, 1 + P + NL)
cache_repo = reposition_kv_cache(cache_sliced, -(P + NL), LAYER_INV_FREQS)

# Layer 0: keys match (bf16 precision), values should be close
bare_k0 = cache_bare.layers[0].keys[:, :, 1:, :].float()
repo_k0 = cache_repo.layers[0].keys.float()
key_rd = (bare_k0 - repo_k0).abs().max().item() / bare_k0.abs().max().item()
check(key_rd < BF16_REL_TOL,
      f"Layer 0 keys: rel diff = {key_rd:.2e}")

bare_v0 = cache_bare.layers[0].values[:, :, 1:, :].float()
repo_v0 = cache_repo.layers[0].values.float()
val_rd = (bare_v0 - repo_v0).abs().max().item() / bare_v0.abs().max().item()
check(val_rd < BF16_REL_TOL,
      f"Layer 0 values: rel diff = {val_rd:.2e} "
      f"(GPU matmul non-determinism across seq lengths)")

# Global layer 0
bare_kg = cache_bare.layers[global_idx].keys[:, :, 1:, :].float()
repo_kg = cache_repo.layers[global_idx].keys.float()
key_rdg = (bare_kg - repo_kg).abs().max().item() / bare_kg.abs().max().item()
print(f"  INFO: Layer {global_idx} (global) keys: rel diff = {key_rdg:.2e} "
      f"(diverges due to prefix attention at layers 0-{global_idx-1})")

# Per-layer divergence profile
print(f"\n  Per-layer divergence (conditioned+repositioned vs bare):")
print(f"  {'Layer':>5} {'Type':>4} {'Key RelDiff':>12} {'Val RelDiff':>12}")
for L in range(min(15, n_layers)):
    bare_k = cache_bare.layers[L].keys[:, :, 1:, :].float()
    repo_k = cache_repo.layers[L].keys.float()
    bare_v = cache_bare.layers[L].values[:, :, 1:, :].float()
    repo_v = cache_repo.layers[L].values.float()
    krd = (bare_k - repo_k).abs().max().item() / bare_k.abs().max().item()
    vrd = (bare_v - repo_v).abs().max().item() / (bare_v.abs().max().item() + 1e-10)
    lt = 'G' if text_cfg.layer_types[L] == 'full_attention' else 'L'
    print(f"  {L:>5} {lt:>4} {krd:>12.4e} {vrd:>12.4e}")

del cache_bare, cache_cond, cache_sliced, cache_repo

# ================================================================
# TEST 6: Float32 algebraic round-trip R(+d) @ R(-d) = I
# ================================================================
print("\n=== Test 6: Float32 round-trip R(+17) @ R(-17) ===")
test_k = torch.randn(1, 8, 10, head_dim, device=DEVICE)
for test_layer in [0, global_idx]:
    inv_f = LAYER_INV_FREQS[test_layer]
    angles = 17.0 * inv_f
    emb = torch.cat([angles, angles])
    cos_f = emb.cos().view(1, 1, 1, -1)
    sin_f = emb.sin().view(1, 1, 1, -1)
    k_fwd = test_k * cos_f + rotate_half(test_k) * sin_f
    k_rt = k_fwd * cos_f + rotate_half(k_fwd) * (-sin_f)
    diff = (test_k - k_rt).abs().max().item()
    ltype = text_cfg.layer_types[test_layer]
    check(diff < 1e-5,
          f"Layer {test_layer} ({ltype[:7]}): round-trip diff = {diff:.2e}")

# ================================================================
# TEST 7: Two-phase mechanism correctness
# ================================================================
# Tests whether the two-phase split itself introduces errors,
# separately from BOS removal.
print("\n=== Test 7: Two-phase mechanism correctness ===")
doc_text_7 = "The cat sat on the mat near the door of the house by the lake"
query_text_7 = "Where did the cat sit?"
answer_text_7 = "on the mat"
doc_ids_7 = tokenizer(doc_text_7, add_special_tokens=False).input_ids
D7 = len(doc_ids_7)

query_ids_7 = tokenizer("\n" + query_text_7 + "\n",
                        add_special_tokens=False).input_ids
answer_ids_7 = tokenizer(answer_text_7, add_special_tokens=False,
                         truncation=True, max_length=256).input_ids

# --- Method A: Normal single-pass (ground truth) ---
full_ids_all = [BOS_ID] + doc_ids_7 + query_ids_7 + answer_ids_7
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids_all], device=DEVICE))
n_ctx = 1 + D7 + len(query_ids_7)
logits_full = out_full.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids_7), :].float()
targets_7 = torch.tensor(answer_ids_7, device=DEVICE)
nll_single_pass = -F.log_softmax(logits_full, dim=-1).gather(
    1, targets_7.unsqueeze(1)).squeeze(1).mean().item()
del out_full

# --- Method B: Two-phase WITH BOS in cache (slice_start=0) ---
# Phase A: encode [BOS + doc], keep ALL entries (including BOS)
cond_ids_b = [BOS_ID] + doc_ids_7
with torch.no_grad():
    out_b = model(input_ids=torch.tensor([cond_ids_b], device=DEVICE),
                  use_cache=True)
cache_with_bos = out_b.past_key_values  # all entries: BOS + doc
del out_b

# Phase B: query + answer starting at position 1+D7
pb_ids_b = query_ids_7 + answer_ids_7
phase_b_start = 1 + D7
pos_b = torch.arange(phase_b_start, phase_b_start + len(pb_ids_b), device=DEVICE)
with torch.no_grad():
    out_pb = model(input_ids=torch.tensor([pb_ids_b], device=DEVICE),
                   past_key_values=cache_with_bos, position_ids=pos_b.unsqueeze(0),
                   cache_position=pos_b, use_cache=False)
n_q = len(query_ids_7)
logits_b = out_pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids_7), :].float()
nll_two_phase_with_bos = -F.log_softmax(logits_b, dim=-1).gather(
    1, targets_7.unsqueeze(1)).squeeze(1).mean().item()
del out_pb, cache_with_bos

# --- Method C: Two-phase WITHOUT BOS (our score() function) ---
nll_two_phase_no_bos = score(doc_text_7, query_text_7, answer_text_7)

print(f"  Single-pass NLL:          {nll_single_pass:.6f}")
print(f"  Two-phase WITH BOS NLL:   {nll_two_phase_with_bos:.6f}")
print(f"  Two-phase NO BOS NLL:     {nll_two_phase_no_bos:.6f}")

diff_with_bos = abs(nll_single_pass - nll_two_phase_with_bos)
diff_with_bos_pct = diff_with_bos / nll_single_pass * 100
diff_no_bos = abs(nll_single_pass - nll_two_phase_no_bos)
diff_no_bos_pct = diff_no_bos / nll_single_pass * 100
bos_effect = nll_single_pass - nll_two_phase_no_bos

print(f"  With-BOS vs single-pass:  {diff_with_bos:.6f} ({diff_with_bos_pct:.2f}%)")
print(f"  No-BOS vs single-pass:    {diff_no_bos:.6f} ({diff_no_bos_pct:.2f}%)")
print(f"  BOS removal effect:       {bos_effect:+.6f} "
      f"({'helps' if bos_effect > 0 else 'hurts'})")

# The two-phase mechanism WITH BOS should closely match single-pass
check(diff_with_bos_pct < 1.0,
      f"Two-phase WITH BOS matches single-pass within 1% ({diff_with_bos_pct:.2f}%)")
# BOS removal is expected to change NLL — just report it
print(f"  INFO: BOS removal creates {diff_no_bos_pct:.1f}% NLL change "
      f"(expected from attention sink removal)")

# ================================================================
# TEST 8: End-to-end NLL comparison
# ================================================================
print("\n=== Test 8: End-to-end NLL scoring ===")
nll_a = score("The cat sat on the mat.", "Where did the cat sit?", "on the mat")
nll_b = score("The cat sat on the mat.", "Where did the cat sit?", "on the mat")
check(abs(nll_a - nll_b) < 1e-5,
      f"Bare NLL consistency: {nll_a:.6f} vs {nll_b:.6f}")

nll_bare = score("The cat sat on the mat.", "Where did the cat sit?", "on the mat")
nll_oracle = score("The cat sat on the mat.", "Where did the cat sit?", "on the mat",
                   prefix_text="Where did the cat sit?")
nll_adv = score("The cat sat on the mat.", "Where did the cat sit?", "on the mat",
                prefix_text="Recipe for chocolate cake")
check(0 < nll_bare < 20, f"Bare NLL valid: {nll_bare:.4f}")
check(0 < nll_oracle < 20, f"Oracle NLL valid: {nll_oracle:.4f}")
check(0 < nll_adv < 20, f"Adversarial NLL valid: {nll_adv:.4f}")
print(f"  Info: bare={nll_bare:.4f}, oracle={nll_oracle:.4f}, "
      f"adv={nll_adv:.4f}")
print(f"  Info: oracle delta = {nll_bare - nll_oracle:+.4f}, "
      f"adv delta = {nll_bare - nll_adv:+.4f}")

# ================================================================
# Summary
# ================================================================
gc.collect()
torch.cuda.empty_cache()
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if failed > 0:
    print("SOME TESTS FAILED — review output above")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
