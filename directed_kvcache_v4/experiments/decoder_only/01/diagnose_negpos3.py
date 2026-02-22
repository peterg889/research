"""Focused diagnostic: why does neg-pos give NLL ~0?
Tests cache metadata, Phase B logits, and attention patterns."""
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
print(f"Loaded. Type: {type(model).__name__}", flush=True)

doc_text = "The cat sat on the mat"
prefix_text = "Recipe for chocolate cake with butter and eggs"
query_text = "Where did the cat sit?"
answer_text = "on the mat"

doc_ids = tokenizer(doc_text, add_special_tokens=False).input_ids
prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
D = len(doc_ids)
P = len(prefix_ids)
print(f"Doc: {D} tok, Prefix: {P} tok", flush=True)


def slice_kv_cache(cache, start_idx):
    sliced = DynamicCache()
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, start_idx:, :]
        v = cache.layers[i].values[:, :, start_idx:, :]
        sliced.update(k, v, i)
    return sliced


# ================================================================
print("\n" + "=" * 70)
print("TEST 1: DynamicCache metadata after slicing")
print("=" * 70)

# Check what attributes DynamicCache has
dc = DynamicCache()
print(f"  DynamicCache attributes: {[a for a in dir(dc) if not a.startswith('__')]}")
print(f"  Has _seen_tokens: {hasattr(dc, '_seen_tokens')}")
print(f"  Has get_seq_length: {hasattr(dc, 'get_seq_length')}")

# Build neg-pos cache
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
    out = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_full = out.past_key_values
del out

print(f"\n  Full cache type: {type(cache_full).__name__}")
print(f"  Full cache get_seq_length(): {cache_full.get_seq_length()}")
print(f"  Full cache layer 0 key shape: {cache_full.layers[0].keys.shape}")

cache_sliced = slice_kv_cache(cache_full, slice_start)
print(f"\n  Sliced cache type: {type(cache_sliced).__name__}")
print(f"  Sliced cache get_seq_length(): {cache_sliced.get_seq_length()}")
print(f"  Sliced cache layer 0 key shape: {cache_sliced.layers[0].keys.shape}")
print(f"  Expected: seq_length={D}")

# Check if there's a _seen_tokens or similar attribute
for attr in ['_seen_tokens', 'seen_tokens', '_seq_len']:
    if hasattr(cache_sliced, attr):
        print(f"  {attr}: {getattr(cache_sliced, attr)}")

del cache_full, cache_sliced

# ================================================================
print("\n" + "=" * 70)
print("TEST 2: Compare Phase B logits — bare vs neg-pos")
print("=" * 70)

query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
pb_ids = query_ids + answer_ids
n_q = len(query_ids)

print(f"  Phase B: {len(pb_ids)} tokens ({n_q} query + {len(answer_ids)} answer)")
print(f"  Query tokens: {[tokenizer.decode([t]) for t in query_ids]}")
print(f"  Answer tokens: {[tokenizer.decode([t]) for t in answer_ids]}")

# --- Bare ---
with torch.no_grad():
    out_bare = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                     use_cache=True)
cache_bare = slice_kv_cache(out_bare.past_key_values, 1)
del out_bare

pos_pb = torch.arange(D + 1, D + 1 + len(pb_ids), device=DEVICE)
with torch.no_grad():
    out_pb_bare = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                        past_key_values=cache_bare, position_ids=pos_pb.unsqueeze(0),
                        cache_position=pos_pb, use_cache=False)
del cache_bare

# --- Neg-pos ---
with torch.no_grad():
    out_neg = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                    position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_neg = slice_kv_cache(out_neg.past_key_values, slice_start)
del out_neg

with torch.no_grad():
    out_pb_neg = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                       past_key_values=cache_neg, position_ids=pos_pb.unsqueeze(0),
                       cache_position=pos_pb, use_cache=False)
del cache_neg

# Print logits side by side
print(f"\n  {'Pos':<5} {'Token':<10} {'Next':<10} {'Bare p(next)':<14} {'NegPos p(next)':<14} {'Bare top1':<20} {'NegPos top1':<20}")
print(f"  {'-'*95}")
for i in range(len(pb_ids)):
    # Bare
    logits_b = out_pb_bare.logits[0, i, :].float()
    probs_b = F.softmax(logits_b, dim=-1)
    top1_b_idx = probs_b.argmax().item()
    top1_b_tok = tokenizer.decode([top1_b_idx])
    top1_b_p = probs_b[top1_b_idx].item()

    # Neg-pos
    logits_n = out_pb_neg.logits[0, i, :].float()
    probs_n = F.softmax(logits_n, dim=-1)
    top1_n_idx = probs_n.argmax().item()
    top1_n_tok = tokenizer.decode([top1_n_idx])
    top1_n_p = probs_n[top1_n_idx].item()

    if i < len(pb_ids) - 1:
        next_tok = tokenizer.decode([pb_ids[i + 1]])
        next_p_b = probs_b[pb_ids[i + 1]].item()
        next_p_n = probs_n[pb_ids[i + 1]].item()
    else:
        next_tok = "(end)"
        next_p_b = 0
        next_p_n = 0

    marker = " <--ANS" if n_q - 1 <= i < n_q - 1 + len(answer_ids) else ""
    print(f"  {i:<5} {tokenizer.decode([pb_ids[i]]):<10} {next_tok:<10} "
          f"{next_p_b:<14.6f} {next_p_n:<14.6f} "
          f"'{top1_b_tok}' ({top1_b_p:.4f}){'':<5} "
          f"'{top1_n_tok}' ({top1_n_p:.4f}){marker}")

del out_pb_bare, out_pb_neg

# ================================================================
print("\n" + "=" * 70)
print("TEST 3: Value magnitudes in cache — bare vs neg-pos")
print("=" * 70)

with torch.no_grad():
    out_bare2 = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                      use_cache=True)
cache_b2 = slice_kv_cache(out_bare2.past_key_values, 1)
del out_bare2

with torch.no_grad():
    out_neg2 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                     position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_n2 = slice_kv_cache(out_neg2.past_key_values, slice_start)
del out_neg2

print(f"  {'Layer':<6} {'Type':<5} {'Bare V mag':<12} {'NegP V mag':<12} "
      f"{'Ratio':<8} {'Bare K mag':<12} {'NegP K mag':<12}")
print(f"  {'-'*75}")
for L in range(0, len(cache_b2.layers), 4):
    bv = cache_b2.layers[L].values.float()
    nv = cache_n2.layers[L].values.float()
    bk = cache_b2.layers[L].keys.float()
    nk = cache_n2.layers[L].keys.float()
    lt = 'G' if hasattr(model, 'config') and hasattr(model.config, 'text_config') and \
         model.config.text_config.layer_types[L] == 'full_attention' else 'L'
    print(f"  {L:<6} {lt:<5} {bv.abs().mean().item():<12.4f} {nv.abs().mean().item():<12.4f} "
          f"{nv.abs().mean().item() / (bv.abs().mean().item() + 1e-10):<8.2f} "
          f"{bk.abs().mean().item():<12.4f} {nk.abs().mean().item():<12.4f}")

del cache_b2, cache_n2

# ================================================================
print("\n" + "=" * 70)
print("TEST 4: Neg-pos WITHOUT slicing (full cache, Phase B continues)")
print("=" * 70)

# If we don't slice, Phase B tokens can attend to prefix AND doc
with torch.no_grad():
    out_ns = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_ns = out_ns.past_key_values  # NOT sliced
del out_ns

# Phase B starts at position len(cond_ids)
pb_start_ns = len(cond_ids)
pos_pb_ns = torch.arange(pb_start_ns, pb_start_ns + len(pb_ids), device=DEVICE)

with torch.no_grad():
    out_pb_ns = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                      past_key_values=cache_ns,
                      position_ids=pos_pb_ns.unsqueeze(0),
                      cache_position=pos_pb_ns, use_cache=False)
del cache_ns

logits_ns = out_pb_ns.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
targets = torch.tensor(answer_ids, device=DEVICE)
nll_noslice = -F.log_softmax(logits_ns, dim=-1).gather(
    1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pb_ns

print(f"  Neg-pos WITHOUT slicing: NLL={nll_noslice:.6f}")
print(f"  (Full cache, Phase B can attend to prefix + doc)")

# Compare: what about with RoPE positions for Phase B matching neg-pos doc positions?
# Phase B should start at D+1 (matching the neg-pos doc end position)
with torch.no_grad():
    out_ns2 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                    position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_ns2 = out_ns2.past_key_values
del out_ns2

pos_pb_match = torch.arange(D + 1, D + 1 + len(pb_ids), device=DEVICE)
# But cache_position must continue from len(cond_ids)!
cpos_pb_match = torch.arange(len(cond_ids), len(cond_ids) + len(pb_ids), device=DEVICE)

with torch.no_grad():
    out_pb_ns2 = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                       past_key_values=cache_ns2,
                       position_ids=pos_pb_match.unsqueeze(0),
                       cache_position=cpos_pb_match, use_cache=False)
del cache_ns2

logits_ns2 = out_pb_ns2.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
nll_noslice2 = -F.log_softmax(logits_ns2, dim=-1).gather(
    1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pb_ns2

print(f"  Neg-pos NO slice, position_ids=D+1..., cache_pos=len(cond)+...: NLL={nll_noslice2:.6f}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 5: Bare with manually matching cache_position")
print("=" * 70)

# The bare case: cache has D entries. Phase B uses cache_position D+1,...
# What if we use cache_position D (matching cache length) instead of D+1?
with torch.no_grad():
    out_b5 = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                   use_cache=True)
cache_b5 = slice_kv_cache(out_b5.past_key_values, 1)
del out_b5

print(f"  Cache seq_length after slice: {cache_b5.get_seq_length()}")

# Test: cache_position = D, D+1, ... (continuous with cache)
pos_d = torch.arange(D, D + len(pb_ids), device=DEVICE)
with torch.no_grad():
    out_d = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                  past_key_values=cache_b5,
                  position_ids=pos_pb.unsqueeze(0),  # D+1, D+2, ...
                  cache_position=pos_d,  # D, D+1, ...
                  use_cache=False)
logits_d = out_d.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
nll_d = -F.log_softmax(logits_d, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_d, cache_b5

# Test: without explicit cache_position
with torch.no_grad():
    out_b5b = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                    use_cache=True)
cache_b5b = slice_kv_cache(out_b5b.past_key_values, 1)
del out_b5b

with torch.no_grad():
    out_no_cpos = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                        past_key_values=cache_b5b,
                        position_ids=pos_pb.unsqueeze(0),
                        use_cache=False)  # NO cache_position
logits_nc = out_no_cpos.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
nll_nc = -F.log_softmax(logits_nc, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_no_cpos, cache_b5b

print(f"  Bare, cache_pos=D+1,...:   NLL={0.987140:.6f} (from TEST 3)")
print(f"  Bare, cache_pos=D,...:     NLL={nll_d:.6f}")
print(f"  Bare, NO cache_position:   NLL={nll_nc:.6f}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 6: Neg-pos with NO cache_position in Phase B")
print("=" * 70)

with torch.no_grad():
    out_n6 = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                   position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_n6 = slice_kv_cache(out_n6.past_key_values, slice_start)
del out_n6

# Phase B without explicit cache_position — let the model figure it out
with torch.no_grad():
    out_pb_n6 = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                      past_key_values=cache_n6,
                      position_ids=pos_pb.unsqueeze(0),
                      use_cache=False)  # NO cache_position
del cache_n6

logits_n6 = out_pb_n6.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
nll_n6 = -F.log_softmax(logits_n6, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pb_n6

print(f"  Neg-pos, NO cache_position in Phase B:  NLL={nll_n6:.6f}")
print(f"  Neg-pos, WITH cache_position in Phase B: NLL=0.000035 (from above)")

# Also try with cache_position = D, D+1, ... (matching cache seq_length)
with torch.no_grad():
    out_n6b = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                    position_ids=neg_pos, cache_position=cond_cpos, use_cache=True)
cache_n6b = slice_kv_cache(out_n6b.past_key_values, slice_start)
del out_n6b

cpos_d = torch.arange(D, D + len(pb_ids), device=DEVICE)
with torch.no_grad():
    out_pb_n6b = model(input_ids=torch.tensor([pb_ids], device=DEVICE),
                       past_key_values=cache_n6b,
                       position_ids=pos_pb.unsqueeze(0),
                       cache_position=cpos_d,
                       use_cache=False)
del cache_n6b

logits_n6b = out_pb_n6b.logits[0, n_q-1:n_q-1+len(answer_ids), :].float()
nll_n6b = -F.log_softmax(logits_n6b, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_pb_n6b

print(f"  Neg-pos, cache_position=D,...:           NLL={nll_n6b:.6f}")

print("\nDone.", flush=True)
