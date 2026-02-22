"""
Diagnose why negative position prefix produces broken results.

Key mystery: Layer 0 doc keys should be IDENTICAL between bare and conditioned
(same token + same RoPE position = same key). But they differ by 0.0625.

This script systematically tests hypotheses:
1. Are negative position_ids handled correctly by Gemma 3's RoPE?
2. Does the causal mask get corrupted by non-monotonic position_ids?
3. Is there an interaction between cache_position and position_ids?
"""
import os, sys, torch
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

text_cfg = getattr(model.config, 'text_config', model.config)
layer_types = getattr(text_cfg, 'layer_types', [])
print(f"Loaded. Layers: {len(layer_types)}, types: {set(layer_types)}")

# Simple test sequence
doc_text = "The cat sat on the mat"
query_text = "Where did the cat sit?"
answer_text = "on the mat"
prefix_text = "Recipe for chocolate cake with butter and eggs"

doc_ids = tokenizer(doc_text, add_special_tokens=False).input_ids
prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
NL_IDS = tokenizer("\n", add_special_tokens=False).input_ids
D = len(doc_ids)
P = len(prefix_ids)
NL = len(NL_IDS)

print(f"\nDoc: {D} tokens, Prefix: {P} tokens, NL: {NL} tokens")

# ================================================================
# TEST A: Layer 0 keys — bare vs conditioned with negative positions
# ================================================================
print("\n" + "=" * 70)
print("TEST A: Layer 0 key comparison — bare vs negative-position conditioned")
print("=" * 70)

# A1: Bare — [BOS, doc], default positions
bare_ids = [BOS_ID] + doc_ids
with torch.no_grad():
    out_bare = model(input_ids=torch.tensor([bare_ids], device=DEVICE), use_cache=True)
cache_bare = out_bare.past_key_values
del out_bare

# A2: Bare with EXPLICIT positions and cache_position (control)
bare_pos = torch.arange(len(bare_ids), device=DEVICE).unsqueeze(0)
bare_cpos = torch.arange(len(bare_ids), device=DEVICE)
with torch.no_grad():
    out_bare_exp = model(input_ids=torch.tensor([bare_ids], device=DEVICE),
                         position_ids=bare_pos, cache_position=bare_cpos,
                         use_cache=True)
cache_bare_exp = out_bare_exp.past_key_values
del out_bare_exp

# A3: Conditioned with negative positions
cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
neg_pos = torch.cat([
    torch.tensor([0], device=DEVICE),               # BOS at 0
    torch.arange(-(P + NL), -NL, device=DEVICE),    # prefix at negative
    torch.arange(-NL, 0, device=DEVICE),             # \n at negative
    torch.arange(1, D + 1, device=DEVICE),           # doc at 1..D
]).unsqueeze(0)
cond_cpos = torch.arange(len(cond_ids), device=DEVICE)

print(f"\nBare positions: {list(range(len(bare_ids)))}")
print(f"Cond positions: {neg_pos[0].tolist()}")
print(f"Cond cache_pos: {cond_cpos.tolist()}")

with torch.no_grad():
    out_cond = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                     position_ids=neg_pos, cache_position=cond_cpos,
                     use_cache=True)
cache_cond = out_cond.past_key_values
del out_cond

# Compare layer 0 doc keys
slice_start = 1 + P + NL
bare_k0_doc = cache_bare.layers[0].keys[:, :, 1:, :].float()  # skip BOS
bare_exp_k0_doc = cache_bare_exp.layers[0].keys[:, :, 1:, :].float()
cond_k0_doc = cache_cond.layers[0].keys[:, :, slice_start:, :].float()

bare_v0_doc = cache_bare.layers[0].values[:, :, 1:, :].float()
bare_exp_v0_doc = cache_bare_exp.layers[0].values[:, :, 1:, :].float()
cond_v0_doc = cache_cond.layers[0].values[:, :, slice_start:, :].float()

print(f"\nLayer 0 key shapes: bare={bare_k0_doc.shape}, cond={cond_k0_doc.shape}")

d1 = (bare_k0_doc - bare_exp_k0_doc).abs().max().item()
d2 = (bare_k0_doc - cond_k0_doc).abs().max().item()
d3 = (bare_exp_k0_doc - cond_k0_doc).abs().max().item()
print(f"\n  Bare (default) vs Bare (explicit pos): {d1:.6e}")
print(f"  Bare (default) vs Conditioned:         {d2:.6e}")
print(f"  Bare (explicit) vs Conditioned:        {d3:.6e}")

d1v = (bare_v0_doc - bare_exp_v0_doc).abs().max().item()
d2v = (bare_v0_doc - cond_v0_doc).abs().max().item()
print(f"\n  VALUES — Bare (default) vs Bare (explicit): {d1v:.6e}")
print(f"  VALUES — Bare (default) vs Conditioned:     {d2v:.6e}")

# ================================================================
# TEST B: Check if the issue is negative positions or non-monotonic positions
# ================================================================
print("\n" + "=" * 70)
print("TEST B: Positive non-standard positions (prefix at 10000+)")
print("=" * 70)

# Put prefix at high positions: prefix at 10000..10000+P-1, doc at 1..D
high_pos = torch.cat([
    torch.tensor([0], device=DEVICE),                      # BOS at 0
    torch.arange(10000, 10000 + P, device=DEVICE),         # prefix at 10000+
    torch.arange(10000 + P, 10000 + P + NL, device=DEVICE),  # \n at 10000+P+
    torch.arange(1, D + 1, device=DEVICE),                 # doc at 1..D
]).unsqueeze(0)

print(f"High positions: {high_pos[0].tolist()}")

with torch.no_grad():
    out_high = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                     position_ids=high_pos, cache_position=cond_cpos,
                     use_cache=True)
cache_high = out_high.past_key_values
del out_high

high_k0_doc = cache_high.layers[0].keys[:, :, slice_start:, :].float()
d_high = (bare_k0_doc - high_k0_doc).abs().max().item()
print(f"\n  Bare vs High-pos conditioned (layer 0 keys): {d_high:.6e}")

# ================================================================
# TEST C: Same tokens, same positions, different sequence lengths
# ================================================================
print("\n" + "=" * 70)
print("TEST C: Single doc token in different contexts")
print("=" * 70)

# Encode just one doc token at position 1 in different contexts
single_doc_id = doc_ids[0]
print(f"Testing single token: {tokenizer.decode([single_doc_id])} (id={single_doc_id})")

# C1: [BOS, token] — positions [0, 1]
with torch.no_grad():
    out_c1 = model(input_ids=torch.tensor([[BOS_ID, single_doc_id]], device=DEVICE),
                   use_cache=True)
k_c1 = out_c1.past_key_values.layers[0].keys[:, :, 1:, :].float()  # token at pos 1
del out_c1

# C2: [BOS, pad, pad, pad, token] — positions [0, 1, 2, 3, 4], token at pos 4
# but with custom position_ids: [0, -3, -2, -1, 1] — token at pos 1
pad_ids = tokenizer("x y z", add_special_tokens=False).input_ids[:3]
c2_ids = [BOS_ID] + pad_ids + [single_doc_id]
c2_pos = torch.tensor([[0, -3, -2, -1, 1]], device=DEVICE)
c2_cpos = torch.arange(len(c2_ids), device=DEVICE)
with torch.no_grad():
    out_c2 = model(input_ids=torch.tensor([c2_ids], device=DEVICE),
                   position_ids=c2_pos, cache_position=c2_cpos,
                   use_cache=True)
k_c2 = out_c2.past_key_values.layers[0].keys[:, :, -1:, :].float()  # last token at pos 1
del out_c2

# C3: [BOS, token] with explicit positions [0, 1] and cache_position [0, 1]
c3_pos = torch.tensor([[0, 1]], device=DEVICE)
c3_cpos = torch.arange(2, device=DEVICE)
with torch.no_grad():
    out_c3 = model(input_ids=torch.tensor([[BOS_ID, single_doc_id]], device=DEVICE),
                   position_ids=c3_pos, cache_position=c3_cpos,
                   use_cache=True)
k_c3 = out_c3.past_key_values.layers[0].keys[:, :, 1:, :].float()
del out_c3

d_c12 = (k_c1 - k_c2).abs().max().item()
d_c13 = (k_c1 - k_c3).abs().max().item()
d_c23 = (k_c2 - k_c3).abs().max().item()
print(f"\n  C1 [BOS,tok] vs C2 [BOS,pad,pad,pad,tok] (both at pos 1): {d_c12:.6e}")
print(f"  C1 [BOS,tok] vs C3 [BOS,tok] explicit pos:                {d_c13:.6e}")
print(f"  C2 [BOS,pad,pad,pad,tok] vs C3 [BOS,tok] explicit pos:    {d_c23:.6e}")

if d_c12 > 1e-6:
    print("  ** Layer 0 keys DIFFER for same token at same position in different contexts!")
    print("  ** This should NOT happen — layer 0 keys depend only on token + position.")
    print("  ** Investigating: is the model doing something unexpected?")

# ================================================================
# TEST D: Check causal mask construction
# ================================================================
print("\n" + "=" * 70)
print("TEST D: Causal mask with negative positions")
print("=" * 70)

# Hook into the model to capture the causal mask
captured_masks = {}

def capture_mask_hook(module, args, kwargs, output):
    # The model's forward method constructs the causal mask
    # We need to check what's passed to the first decoder layer
    pass

# Instead, let's check what _update_causal_mask produces
# by calling it directly if available
if hasattr(model, '_update_causal_mask'):
    print("  Model has _update_causal_mask method")
elif hasattr(model.model, '_update_causal_mask'):
    print("  model.model has _update_causal_mask method")

# Let's look at what the model's forward method does with position_ids
print("\n  Checking model internals...")
print(f"  Model class: {type(model).__name__}")
print(f"  Inner model class: {type(model.model).__name__}")

# Check if there's any position clamping or modification
import inspect
forward_src = inspect.getsource(type(model.model).forward)
if 'position_ids' in forward_src:
    # Look for any clamping, abs(), or modification of position_ids
    lines = forward_src.split('\n')
    pos_lines = [l.strip() for l in lines if 'position_ids' in l and 'cache_position' not in l]
    print(f"\n  Lines mentioning position_ids in model.forward:")
    for l in pos_lines[:10]:
        print(f"    {l}")

# ================================================================
# TEST E: Direct RoPE test with negative positions
# ================================================================
print("\n" + "=" * 70)
print("TEST E: Direct RoPE computation with negative positions")
print("=" * 70)

# Get the rotary embedding module
rotary = model.model.layers[0].self_attn.rotary_emb
print(f"  Rotary class: {type(rotary).__name__}")

# Compute RoPE for position 1 directly
pos_1 = torch.tensor([[1]], device=DEVICE)
pos_neg5 = torch.tensor([[-5]], device=DEVICE)

# Get cos/sin for position 1
cos1, sin1 = rotary(torch.zeros(1, 1, 1, device=DEVICE), pos_1)
cos_neg5, sin_neg5 = rotary(torch.zeros(1, 1, 1, device=DEVICE), pos_neg5)

print(f"  cos(pos=1) first 8: {cos1[0, 0, :8].tolist()}")
print(f"  sin(pos=1) first 8: {sin1[0, 0, :8].tolist()}")
print(f"  cos(pos=-5) first 8: {cos_neg5[0, 0, :8].tolist()}")
print(f"  sin(pos=-5) first 8: {sin_neg5[0, 0, :8].tolist()}")

# cos(-x) = cos(x), sin(-x) = -sin(x)
cos5, sin5 = rotary(torch.zeros(1, 1, 1, device=DEVICE), torch.tensor([[5]], device=DEVICE))
print(f"\n  cos(pos=5) first 8:  {cos5[0, 0, :8].tolist()}")
print(f"  sin(pos=5) first 8:  {sin5[0, 0, :8].tolist()}")
print(f"  cos(-5)==cos(5)? {torch.allclose(cos_neg5, cos5, atol=1e-5)}")
print(f"  sin(-5)==-sin(5)? {torch.allclose(sin_neg5, -sin5, atol=1e-5)}")

# Test with batch of positions including negatives
batch_pos = torch.tensor([[0, -3, -2, -1, 1, 2, 3]], device=DEVICE)
cos_batch, sin_batch = rotary(torch.zeros(1, 7, 1, device=DEVICE), batch_pos)
# Position 1 is at index 4
cos_p1_from_batch = cos_batch[:, 4:5, :]
sin_p1_from_batch = sin_batch[:, 4:5, :]
print(f"\n  cos(pos=1) from single: {cos1[0, 0, :4].tolist()}")
print(f"  cos(pos=1) from batch:  {cos_p1_from_batch[0, 0, :4].tolist()}")
print(f"  Match? {torch.allclose(cos1, cos_p1_from_batch, atol=1e-5)}")

# ================================================================
# TEST F: Phase B NLL comparison
# ================================================================
print("\n" + "=" * 70)
print("TEST F: End-to-end NLL with different approaches")
print("=" * 70)

def slice_kv_cache(cache, start_idx):
    sliced = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, start_idx:, :]
        v = cache.layers[i].values[:, :, start_idx:, :]
        sliced.update(k, v, i)
    return sliced

query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
pb_ids = query_ids + answer_ids

# F1: Single-pass baseline
full_ids = [BOS_ID] + doc_ids + query_ids + answer_ids
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D + len(query_ids)
logits_full = out_full.logits[0, n_ctx-1:n_ctx-1+len(answer_ids), :].float()
targets = torch.tensor(answer_ids, device=DEVICE)
nll_single = -F.log_softmax(logits_full, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_full

# F2: Two-phase bare (BOS removed)
with torch.no_grad():
    out_b = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE), use_cache=True)
cache_b = slice_kv_cache(out_b.past_key_values, 1)  # remove BOS
del out_b

phase_b_start = D + 1
pos_pb = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)
with torch.no_grad():
    out_pb = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache_b, position_ids=pos_pb.unsqueeze(0),
                   cache_position=pos_pb, use_cache=False)
logits_b = out_pb.logits[0, len(query_ids)-1:len(query_ids)-1+len(answer_ids), :].float()
nll_bare = -F.log_softmax(logits_b, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pb, cache_b

# F3: Two-phase conditioned (negative positions, prefix removed)
with torch.no_grad():
    out_c = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                  position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_c = slice_kv_cache(out_c.past_key_values, slice_start)
del out_c

with torch.no_grad():
    out_pc = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache_c, position_ids=pos_pb.unsqueeze(0),
                   cache_position=pos_pb, use_cache=False)
logits_c = out_pc.logits[0, len(query_ids)-1:len(query_ids)-1+len(answer_ids), :].float()
nll_cond = -F.log_softmax(logits_c, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pc, cache_c

# F4: Two-phase conditioned with HIGH positions (instead of negative)
with torch.no_grad():
    out_h = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                  position_ids=high_pos, cache_position=cond_cpos, use_cache=True)
cache_h = slice_kv_cache(out_h.past_key_values, slice_start)
del out_h

with torch.no_grad():
    out_ph = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache_h, position_ids=pos_pb.unsqueeze(0),
                   cache_position=pos_pb, use_cache=False)
logits_h = out_ph.logits[0, len(query_ids)-1:len(query_ids)-1+len(answer_ids), :].float()
nll_high = -F.log_softmax(logits_h, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_ph, cache_h

# F5: Two-phase conditioned with NATURAL positions (prefix at 0..P-1, doc at P+NL..P+NL+D-1)
# Then DON'T slice — just use the full cache and start phase B accordingly
nat_pos = torch.arange(len(cond_ids), device=DEVICE).unsqueeze(0)  # 0, 1, 2, ..., P+NL+D
with torch.no_grad():
    out_n = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                  position_ids=nat_pos, cache_position=cond_cpos, use_cache=True)
# Slice prefix but keep doc with natural positions
cache_n = slice_kv_cache(out_n.past_key_values, slice_start)
del out_n

# Phase B starts at len(cond_ids) for natural positions
nat_pb_start = len(cond_ids)
nat_pos_pb = torch.arange(nat_pb_start, nat_pb_start + len(pb_ids), device=DEVICE)
with torch.no_grad():
    out_pn = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache_n, position_ids=nat_pos_pb.unsqueeze(0),
                   cache_position=nat_pos_pb, use_cache=False)
logits_n = out_pn.logits[0, len(query_ids)-1:len(query_ids)-1+len(answer_ids), :].float()
nll_nat = -F.log_softmax(logits_n, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pn, cache_n

print(f"\n  Single-pass (with BOS):                {nll_single:.6f}")
print(f"  Two-phase bare (BOS removed):          {nll_bare:.6f}")
print(f"  Two-phase neg-pos conditioned:         {nll_cond:.6f}  {'<-- BROKEN?' if nll_cond < 0.01 else ''}")
print(f"  Two-phase high-pos conditioned:        {nll_high:.6f}  {'<-- BROKEN?' if nll_high < 0.01 else ''}")
print(f"  Two-phase natural-pos conditioned:     {nll_nat:.6f}")

# ================================================================
# TEST G: Check _seen_tokens and cache seq length after slicing
# ================================================================
print("\n" + "=" * 70)
print("TEST G: Cache metadata after slicing")
print("=" * 70)

with torch.no_grad():
    out_g = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                  position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_full = out_g.past_key_values
del out_g

print(f"  Full cache: _seen_tokens={cache_full._seen_tokens}, "
      f"seq_length={cache_full.get_seq_length()}")

cache_sliced = slice_kv_cache(cache_full, slice_start)
print(f"  Sliced cache: _seen_tokens={cache_sliced._seen_tokens}, "
      f"seq_length={cache_sliced.get_seq_length()}")
print(f"  Expected seq_length: {D}")
print(f"  Slice start: {slice_start}, Full length: {len(cond_ids)}")

# What does the model think the next cache_position should be?
# The cache has D entries. Phase B uses cache_position starting at D+1.
# But the model might internally compute the causal mask based on _seen_tokens.
print(f"\n  Phase B cache_position starts at: {phase_b_start} (= D+1 = {D+1})")
print(f"  Cache _seen_tokens: {cache_sliced._seen_tokens}")
print(f"  Gap: phase_b_start - _seen_tokens = {phase_b_start - cache_sliced._seen_tokens}")

del cache_full, cache_sliced

# ================================================================
# TEST H: What if we DON'T pass cache_position to Phase A?
# ================================================================
print("\n" + "=" * 70)
print("TEST H: Negative positions WITHOUT explicit cache_position")
print("=" * 70)

# Try passing position_ids but NOT cache_position
with torch.no_grad():
    out_h2 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, use_cache=True)
cache_h2 = out_h2.past_key_values
del out_h2

cache_h2_sliced = slice_kv_cache(cache_h2, slice_start)

with torch.no_grad():
    out_ph2 = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                    past_key_values=cache_h2_sliced,
                    position_ids=pos_pb.unsqueeze(0),
                    cache_position=pos_pb, use_cache=False)
logits_h2 = out_ph2.logits[0, len(query_ids)-1:len(query_ids)-1+len(answer_ids), :].float()
nll_no_cpos = -F.log_softmax(logits_h2, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_ph2, cache_h2, cache_h2_sliced

print(f"  Neg-pos WITHOUT explicit cache_position: {nll_no_cpos:.6f}")
print(f"  Neg-pos WITH explicit cache_position:    {nll_cond:.6f}")

# Also check layer 0 keys
with torch.no_grad():
    out_h3 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, use_cache=True)
k0_no_cpos = out_h3.past_key_values.layers[0].keys[:, :, slice_start:, :].float()
del out_h3

d_cpos = (bare_k0_doc - k0_no_cpos).abs().max().item()
print(f"  Layer 0 key diff (bare vs no-cpos): {d_cpos:.6e}")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Layer 0 key diffs (should be 0.0 if same token+position):")
print(f"    Bare (default) vs Bare (explicit):    {d1:.6e}")
print(f"    Bare vs Neg-pos conditioned:          {d2:.6e}")
print(f"    Bare vs High-pos conditioned:         {d_high:.6e}")
print(f"    Bare vs Neg-pos (no cache_position):  {d_cpos:.6e}")
print(f"    Single-token test (C1 vs C2):         {d_c12:.6e}")

print(f"\n  NLL comparison:")
print(f"    Single-pass:     {nll_single:.6f} (reference)")
print(f"    Bare (no BOS):   {nll_bare:.6f}")
print(f"    Neg-pos cond:    {nll_cond:.6f}")
print(f"    High-pos cond:   {nll_high:.6f}")
print(f"    Natural-pos:     {nll_nat:.6f}")
print(f"    No cache_pos:    {nll_no_cpos:.6f}")

if d2 > 1e-4:
    print(f"\n  ** PROBLEM: Layer 0 keys differ between bare and conditioned!")
    print(f"  ** This means something about the model computation differs even")
    print(f"  ** for the same token at the same RoPE position.")
if nll_cond < 0.1:
    print(f"\n  ** PROBLEM: Conditioned NLL is unrealistically low ({nll_cond:.6f})!")
    print(f"  ** The model is predicting with near-perfect confidence.")
    print(f"  ** This suggests a fundamental bug in the approach.")

print("\nDone.")
