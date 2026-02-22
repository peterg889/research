"""Focused diagnostic: why do layer 0 keys differ with negative positions?"""
import os, sys, warnings, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import logging
logging.disable(logging.WARNING)

MODEL_NAME = "google/gemma-3-12b-it"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN)
model.eval()
DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NL_IDS = tokenizer("\n", add_special_tokens=False).input_ids
NL = len(NL_IDS)

print(f"Model type: {type(model).__name__}", flush=True)
text_cfg = getattr(model.config, 'text_config', model.config)
layer_types = getattr(text_cfg, 'layer_types', [])
print(f"Layers: {len(layer_types)}", flush=True)

doc_text = "The cat sat on the mat"
prefix_text = "Recipe for chocolate cake with butter and eggs"
query_text = "Where did the cat sit?"
answer_text = "on the mat"

doc_ids = tokenizer(doc_text, add_special_tokens=False).input_ids
prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
D = len(doc_ids)
P = len(prefix_ids)
print(f"Doc: {D} tok, Prefix: {P} tok, NL: {NL} tok", flush=True)


def slice_kv_cache(cache, start_idx):
    sliced = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, start_idx:, :]
        v = cache.layers[i].values[:, :, start_idx:, :]
        sliced.update(k, v, i)
    return sliced


def score_nll(doc_ids, query_text, answer_text, cache, phase_b_start):
    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)
    with torch.no_grad():
        pb = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache, position_ids=pos.unsqueeze(0),
                   cache_position=pos, use_cache=False)
    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del pb
    return nll


# ================================================================
print("\n" + "=" * 70)
print("TEST 1: Layer 0 key comparison (same token, same position)")
print("=" * 70)

# A: Bare [BOS, doc] with default positions
with torch.no_grad():
    out_a = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                  use_cache=True)
cache_a = out_a.past_key_values
del out_a

# B: Conditioned [BOS, prefix, \n, doc] with negative prefix positions
cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
slice_start = 1 + P + NL
neg_pos = torch.cat([
    torch.tensor([0], device=DEVICE),
    torch.arange(-(P + NL), -NL, device=DEVICE),
    torch.arange(-NL, 0, device=DEVICE),
    torch.arange(1, D + 1, device=DEVICE),
]).unsqueeze(0)
cond_cpos = torch.arange(len(cond_ids), device=DEVICE)

with torch.no_grad():
    out_b = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                  position_ids=neg_pos, cache_position=cond_cpos,
                  use_cache=True)
cache_b = out_b.past_key_values
del out_b

# Compare doc keys at layer 0
bare_k0 = cache_a.layers[0].keys[:, :, 1:, :].float()   # skip BOS
cond_k0 = cache_b.layers[0].keys[:, :, slice_start:, :].float()
bare_v0 = cache_a.layers[0].values[:, :, 1:, :].float()
cond_v0 = cache_b.layers[0].values[:, :, slice_start:, :].float()

k_maxdiff = (bare_k0 - cond_k0).abs().max().item()
v_maxdiff = (bare_v0 - cond_v0).abs().max().item()
k_meandiff = (bare_k0 - cond_k0).abs().mean().item()
v_meandiff = (bare_v0 - cond_v0).abs().mean().item()

print(f"  Layer 0 key max diff:  {k_maxdiff:.6e}  (mean: {k_meandiff:.6e})")
print(f"  Layer 0 val max diff:  {v_maxdiff:.6e}  (mean: {v_meandiff:.6e})")

# Check per-token key diffs at layer 0
print(f"\n  Per-token layer 0 key diffs (max across heads/dims):")
for t in range(D):
    diff_t = (bare_k0[:, :, t, :] - cond_k0[:, :, t, :]).abs().max().item()
    tok_str = tokenizer.decode([doc_ids[t]])
    print(f"    Token {t} '{tok_str}' (pos={t+1}): {diff_t:.6e}")

# Check a few higher layers
print(f"\n  Per-layer doc key max diff:")
for L in [0, 1, 2, 3, 5, 10, 20, 47]:
    if L >= len(cache_a.layers):
        continue
    bk = cache_a.layers[L].keys[:, :, 1:, :].float()
    ck = cache_b.layers[L].keys[:, :, slice_start:, :].float()
    bv = cache_a.layers[L].values[:, :, 1:, :].float()
    cv = cache_b.layers[L].values[:, :, slice_start:, :].float()
    lt = 'G' if layer_types[L] == 'full_attention' else 'L'
    print(f"    Layer {L:>2} ({lt}): key={( bk - ck).abs().max().item():.4e}  "
          f"val={(bv - cv).abs().max().item():.4e}")

del cache_a, cache_b

# ================================================================
print("\n" + "=" * 70)
print("TEST 2: Single token — isolate whether context length matters")
print("=" * 70)

# Same single token at same RoPE position, but in sequences of different lengths
single_id = doc_ids[0]
tok_str = tokenizer.decode([single_id])
print(f"  Token: '{tok_str}' (id={single_id}), testing at position 1")

# Short: [BOS, token]
with torch.no_grad():
    out_s = model(input_ids=torch.tensor([[BOS_ID, single_id]], device=DEVICE),
                  use_cache=True)
k_short = out_s.past_key_values.layers[0].keys[:, :, 1:, :].float()
del out_s

# Long: [BOS, pad*10, token] with explicit positions [0, -10, -9, ..., -1, 1]
pad_ids = tokenizer("x " * 10, add_special_tokens=False).input_ids[:10]
long_ids = [BOS_ID] + pad_ids + [single_id]
long_pos = torch.cat([
    torch.tensor([0], device=DEVICE),
    torch.arange(-10, 0, device=DEVICE),
    torch.tensor([1], device=DEVICE),
]).unsqueeze(0)
long_cpos = torch.arange(len(long_ids), device=DEVICE)

with torch.no_grad():
    out_l = model(input_ids=torch.tensor([long_ids], device=DEVICE),
                  position_ids=long_pos, cache_position=long_cpos,
                  use_cache=True)
k_long = out_l.past_key_values.layers[0].keys[:, :, -1:, :].float()
del out_l

# Also: [BOS, token] with explicit positions [0, 1]
with torch.no_grad():
    out_e = model(input_ids=torch.tensor([[BOS_ID, single_id]], device=DEVICE),
                  position_ids=torch.tensor([[0, 1]], device=DEVICE),
                  cache_position=torch.arange(2, device=DEVICE),
                  use_cache=True)
k_explicit = out_e.past_key_values.layers[0].keys[:, :, 1:, :].float()
del out_e

d_sl = (k_short - k_long).abs().max().item()
d_se = (k_short - k_explicit).abs().max().item()
d_le = (k_long - k_explicit).abs().max().item()

print(f"  Short [BOS,tok] vs Long [BOS,pad*10,tok] (both pos=1): {d_sl:.6e}")
print(f"  Short [BOS,tok] vs Explicit [BOS,tok]:                 {d_se:.6e}")
print(f"  Long [BOS,pad*10,tok] vs Explicit [BOS,tok]:           {d_le:.6e}")

if d_sl < 1e-6 and d_se < 1e-6:
    print("  All match — layer 0 keys are deterministic for same token+position")
elif d_se < 1e-6 and d_sl > 1e-4:
    print("  ** Context length MATTERS — layer 0 keys differ in longer sequence!")
    print("  ** This suggests CUDA kernel non-determinism based on problem size")

# ================================================================
print("\n" + "=" * 70)
print("TEST 3: End-to-end NLL")
print("=" * 70)

# Bare
with torch.no_grad():
    out_bare = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                     use_cache=True)
cache_bare = slice_kv_cache(out_bare.past_key_values, 1)
del out_bare
nll_bare = score_nll(doc_ids, query_text, answer_text, cache_bare, D + 1)
del cache_bare

# Conditioned with negative positions
cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
with torch.no_grad():
    out_cond = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                     position_ids=neg_pos, cache_position=cond_cpos,
                     use_cache=True)
cache_cond = slice_kv_cache(out_cond.past_key_values, slice_start)
del out_cond
nll_cond = score_nll(doc_ids, query_text, answer_text, cache_cond, D + 1)
del cache_cond

# Conditioned with NATURAL positions (no neg, just sequential)
nat_pos = torch.arange(len(cond_ids), device=DEVICE).unsqueeze(0)
with torch.no_grad():
    out_nat = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                    position_ids=nat_pos, cache_position=cond_cpos,
                    use_cache=True)
cache_nat = slice_kv_cache(out_nat.past_key_values, slice_start)
del out_nat
# For natural positions, doc starts at position slice_start, so phase_b_start = len(cond_ids)
nll_nat = score_nll(doc_ids, query_text, answer_text, cache_nat, len(cond_ids))
del cache_nat

# Single-pass reference
full_ids = [BOS_ID] + doc_ids + tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids + tokenizer(answer_text, add_special_tokens=False).input_ids
query_ids_r = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
answer_ids_r = tokenizer(answer_text, add_special_tokens=False).input_ids
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D + len(query_ids_r)
logits_f = out_full.logits[0, n_ctx-1:n_ctx-1+len(answer_ids_r), :].float()
targets_f = torch.tensor(answer_ids_r, device=DEVICE)
nll_single = -F.log_softmax(logits_f, dim=-1).gather(1, targets_f.unsqueeze(1)).squeeze(1).mean().item()
del out_full

print(f"  Single-pass (with BOS):          {nll_single:.6f}")
print(f"  Two-phase bare (BOS removed):    {nll_bare:.6f}")
print(f"  Two-phase neg-pos conditioned:   {nll_cond:.6f}")
print(f"  Two-phase natural-pos cond:      {nll_nat:.6f}")

if nll_cond < 0.01:
    print(f"\n  ** BROKEN: neg-pos NLL is {nll_cond:.6f} — unrealistically low")
    print(f"  ** Investigating Phase B cache interaction...")

# ================================================================
print("\n" + "=" * 70)
print("TEST 4: Cache _seen_tokens after slicing")
print("=" * 70)

with torch.no_grad():
    out_t4 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, cache_position=cond_cpos,
                   use_cache=True)
cache_full = out_t4.past_key_values
del out_t4

print(f"  Full cache: _seen_tokens={cache_full._seen_tokens}, "
      f"seq_len={cache_full.get_seq_length()}")
print(f"  Full cache layer 0 key shape: {cache_full.layers[0].keys.shape}")

cache_sliced = slice_kv_cache(cache_full, slice_start)
print(f"  Sliced cache: _seen_tokens={cache_sliced._seen_tokens}, "
      f"seq_len={cache_sliced.get_seq_length()}")
print(f"  Sliced cache layer 0 key shape: {cache_sliced.layers[0].keys.shape}")
print(f"  Expected: _seen_tokens={D}, seq_len={D}")

# The issue might be that phase_b uses cache_position starting at D+1,
# but the cache only has D entries (_seen_tokens=D).
# The model might think the cache has seen D tokens, so the next
# cache_position should be D, not D+1.
# Let's try starting phase B at position D instead of D+1.

print(f"\n  Phase B starts at position D+1={D+1}")
print(f"  But cache _seen_tokens={cache_sliced._seen_tokens}")
print(f"  Gap = {D + 1 - cache_sliced._seen_tokens}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 5: Phase B with different start positions")
print("=" * 70)

# Try phase B at D (no gap) vs D+1 (gap of 1)
for pb_start in [D, D + 1, D + 2]:
    with torch.no_grad():
        out_t5 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       position_ids=neg_pos, cache_position=cond_cpos,
                       use_cache=True)
    cache_t5 = slice_kv_cache(out_t5.past_key_values, slice_start)
    del out_t5
    nll_t5 = score_nll(doc_ids, query_text, answer_text, cache_t5, pb_start)
    del cache_t5
    print(f"  Phase B start={pb_start}: NLL={nll_t5:.6f}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 6: Inspect Phase B logits — what is the model predicting?")
print("=" * 70)

with torch.no_grad():
    out_t6 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, cache_position=cond_cpos,
                   use_cache=True)
cache_t6 = slice_kv_cache(out_t6.past_key_values, slice_start)
del out_t6

query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
pb_ids = query_ids + answer_ids
pos_pb = torch.arange(D + 1, D + 1 + len(pb_ids), device=DEVICE)

with torch.no_grad():
    out_pb = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                   past_key_values=cache_t6, position_ids=pos_pb.unsqueeze(0),
                   cache_position=pos_pb, use_cache=False)
del cache_t6

# Check logits at each position
print(f"  Phase B tokens: {[tokenizer.decode([t]) for t in pb_ids]}")
print(f"  Answer tokens: {[tokenizer.decode([t]) for t in answer_ids]}")
n_q = len(query_ids)
for i in range(len(pb_ids)):
    logits_i = out_pb.logits[0, i, :].float()
    probs = F.softmax(logits_i, dim=-1)
    top5 = probs.topk(5)
    top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    top5_probs = top5.values.tolist()

    # What does this position predict?
    if i < len(pb_ids) - 1:
        next_tok = tokenizer.decode([pb_ids[i + 1]])
        next_prob = probs[pb_ids[i + 1]].item()
    else:
        next_tok = "(end)"
        next_prob = 0.0

    marker = " <-- answer logit" if n_q - 1 <= i < n_q - 1 + len(answer_ids) else ""
    print(f"  Pos {i}: '{tokenizer.decode([pb_ids[i]])}' -> "
          f"next='{next_tok}' p={next_prob:.4f} | "
          f"top: {list(zip(top5_toks, [f'{p:.4f}' for p in top5_probs]))}{marker}")

del out_pb

# ================================================================
print("\n" + "=" * 70)
print("TEST 7: Check if bare Phase B looks normal")
print("=" * 70)

with torch.no_grad():
    out_t7 = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                   use_cache=True)
cache_t7 = slice_kv_cache(out_t7.past_key_values, 1)
del out_t7

with torch.no_grad():
    out_pb7 = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                    past_key_values=cache_t7, position_ids=pos_pb.unsqueeze(0),
                    cache_position=pos_pb, use_cache=False)
del cache_t7

print(f"  Bare Phase B logits:")
for i in range(len(pb_ids)):
    logits_i = out_pb7.logits[0, i, :].float()
    probs = F.softmax(logits_i, dim=-1)
    top5 = probs.topk(5)
    top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    top5_probs = top5.values.tolist()

    if i < len(pb_ids) - 1:
        next_tok = tokenizer.decode([pb_ids[i + 1]])
        next_prob = probs[pb_ids[i + 1]].item()
    else:
        next_tok = "(end)"
        next_prob = 0.0

    marker = " <-- answer logit" if n_q - 1 <= i < n_q - 1 + len(answer_ids) else ""
    print(f"  Pos {i}: '{tokenizer.decode([pb_ids[i]])}' -> "
          f"next='{next_tok}' p={next_prob:.4f} | "
          f"top: {list(zip(top5_toks, [f'{p:.4f}' for p in top5_probs]))}{marker}")

del out_pb7

print("\nDone.", flush=True)
