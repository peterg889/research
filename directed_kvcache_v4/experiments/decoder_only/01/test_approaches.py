"""
Test multiple scoring approaches with corrected masking (no look-ahead).

Approaches:
A) Full cache (no slicing) — Phase B attends to BOS + prefix + doc
B) Natural positions, BOS retained, RoPE repositioning — isolates doc priming
C) Natural positions, BOS retained, NO repositioning — position shift confound
"""
import os, sys, warnings, gc, torch, torch.nn.functional as F
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from lib.data import count_words
from lib.analysis import cohens_d
import logging
logging.disable(logging.WARNING)

MODEL_NAME = "google/gemma-3-12b-it"
HF_TOKEN = os.environ.get("HF_TOKEN")
SEED = 42
N_TEST = 30

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN)
model.eval()
DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NL_IDS = tokenizer("\n", add_special_tokens=False).input_ids
NL = len(NL_IDS)

text_cfg = getattr(model.config, 'text_config', model.config)
layer_types = getattr(text_cfg, 'layer_types', [])
rope_params = getattr(text_cfg, 'rope_parameters', {})
print(f"Loaded. Layers: {len(layer_types)}", flush=True)

# Build per-layer inv_freqs for repositioning
def build_layer_inv_freqs():
    inv_freqs = {}
    for lt, params in rope_params.items():
        theta = params.get('rope_theta', 10000.0)
        dim = text_cfg.head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
        inv_freqs[lt] = inv_freq
    return inv_freqs

LAYER_INV_FREQS = build_layer_inv_freqs()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def reposition_kv_cache(cache, old_positions, new_positions, bos_start=0):
    """Reposition doc keys in cache from old_positions to new_positions.
    Assumes cache entries at indices bos_start..bos_start+len(old_positions)-1
    are BOS (at bos_start, unchanged) + doc (rest, repositioned).
    Actually: only repositions entries starting at bos_start+1.
    If bos_start=0, entry 0 is BOS (untouched), entries 1..D are doc (repositioned).
    """
    delta = new_positions - old_positions  # position shift per doc token
    for L in range(len(cache.layers)):
        lt = layer_types[L]
        inv_freq = LAYER_INV_FREQS[lt]
        k = cache.layers[L].keys  # [1, n_heads, seq_len, head_dim]

        # Only reposition doc entries (skip BOS at index bos_start)
        doc_keys = k[:, :, bos_start + 1:, :]  # [1, n_heads, D, head_dim]

        # Compute rotation angles for delta positions
        freqs = torch.einsum('i,j->ij', delta.float(), inv_freq)  # [D, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [D, head_dim]
        cos_delta = emb.cos().to(k.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, D, head_dim]
        sin_delta = emb.sin().to(k.dtype).unsqueeze(0).unsqueeze(0)

        # Apply rotation: k_new = k * cos(delta) + rotate_half(k) * sin(delta)
        doc_keys_new = doc_keys * cos_delta + rotate_half(doc_keys) * sin_delta
        cache.layers[L].keys = torch.cat([
            k[:, :, :bos_start + 1, :],  # BOS (unchanged)
            doc_keys_new,
        ], dim=2)

    return cache


def select_kv_cache(cache, indices):
    """Select specific cache indices."""
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def score_phase_b(cache, pb_ids, query_ids, answer_ids, phase_b_start):
    """Run Phase B scoring. cache_position = auto-generated from cache length."""
    n_q = len(query_ids)
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)
    # Let model auto-generate cache_position from cache.get_seq_length()
    # This ensures no look-ahead: cache_position = [seq_len, seq_len+1, ...]
    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            use_cache=False,
        )
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del pb
    return nll


def score_A_full_cache(doc_text, query_text, answer_text, prefix_text=None):
    """Approach A: Full cache (no slicing). Phase B attends to everything."""
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)

    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               truncation=True, max_length=512).input_ids
        cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa
        phase_b_start = len(cond_ids)  # continue from end
    else:
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa
        phase_b_start = 1 + D

    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0

    pb_ids = query_ids + answer_ids
    nll = score_phase_b(cache, pb_ids, query_ids, answer_ids, phase_b_start)
    del cache
    return nll


def score_B_repositioned(doc_text, query_text, answer_text, prefix_text=None):
    """Approach B: Natural positions, BOS retained, RoPE repositioning.
    Doc keys repositioned from P+NL+1..P+NL+D to 1..D.
    BOS stays at position 0. Phase B at D+1.
    """
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)

    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               truncation=True, max_length=512).input_ids
        P = len(prefix_ids)
        cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

        # Select BOS + doc
        keep_indices = [0] + list(range(1 + P + NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)

        # Reposition doc keys from P+NL+1..P+NL+D to 1..D
        old_pos = torch.arange(1 + P + NL, 1 + P + NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
    else:
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

    # Phase B at D+1 (matching doc at 1..D, BOS at 0)
    phase_b_start = D + 1
    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0

    pb_ids = query_ids + answer_ids
    nll = score_phase_b(cache, pb_ids, query_ids, answer_ids, phase_b_start)
    del cache
    return nll


def score_C_natural_pos(doc_text, query_text, answer_text, prefix_text=None):
    """Approach C: Natural positions, BOS retained, NO repositioning.
    Doc stays at positions P+NL+1..P+NL+D. Phase B at P+NL+D+1.
    Has position shift confound but no repositioning artifacts.
    """
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)

    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               truncation=True, max_length=512).input_ids
        P = len(prefix_ids)
        cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

        # Select BOS + doc (no repositioning)
        keep_indices = [0] + list(range(1 + P + NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)

        # Phase B continues from original doc end position
        phase_b_start = 1 + P + NL + D
    else:
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa
        phase_b_start = 1 + D

    query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0

    pb_ids = query_ids + answer_ids
    nll = score_phase_b(cache, pb_ids, query_ids, answer_ids, phase_b_start)
    del cache
    return nll


# ================================================================
# Validation
# ================================================================
print("\n" + "=" * 70)
print("VALIDATION: All approaches match single-pass for bare")
print("=" * 70)

doc = "The cat sat on the mat near the door of the house by the lake"
q = "Where did the cat sit?"
a = "on the mat"

doc_ids_v = tokenizer(doc, add_special_tokens=False).input_ids
Dv = len(doc_ids_v)
query_ids_v = tokenizer("\n" + q + "\n", add_special_tokens=False).input_ids
answer_ids_v = tokenizer(a, add_special_tokens=False).input_ids

full_ids = [BOS_ID] + doc_ids_v + query_ids_v + answer_ids_v
with torch.no_grad():
    out = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + Dv + len(query_ids_v)
logits = out.logits[0, n_ctx-1:n_ctx-1+len(answer_ids_v), :].float()
targets = torch.tensor(answer_ids_v, device=DEVICE)
nll_ref = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out

nll_A = score_A_full_cache(doc, q, a)
nll_B = score_B_repositioned(doc, q, a)
nll_C = score_C_natural_pos(doc, q, a)

print(f"  Single-pass:    {nll_ref:.6f}")
print(f"  Approach A:     {nll_A:.6f} (diff: {abs(nll_A-nll_ref)/nll_ref*100:.2f}%)")
print(f"  Approach B:     {nll_B:.6f} (diff: {abs(nll_B-nll_ref)/nll_ref*100:.2f}%)")
print(f"  Approach C:     {nll_C:.6f} (diff: {abs(nll_C-nll_ref)/nll_ref*100:.2f}%)")
assert abs(nll_A - nll_ref) / nll_ref < 0.01
assert abs(nll_B - nll_ref) / nll_ref < 0.01
assert abs(nll_C - nll_ref) / nll_ref < 0.01
print("  All match within 1%")

# ================================================================
# Quick test with prefix
# ================================================================
print("\n" + "=" * 70)
print("QUICK TEST: Oracle and adversarial (single sample)")
print("=" * 70)

prefix_oracle = q
prefix_adv = "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt."

for label, score_fn in [("A (full cache)", score_A_full_cache),
                         ("B (repositioned)", score_B_repositioned),
                         ("C (natural pos)", score_C_natural_pos)]:
    bare = score_fn(doc, q, a)
    oracle = score_fn(doc, q, a, prefix_text=prefix_oracle)
    adv = score_fn(doc, q, a, prefix_text=prefix_adv)
    print(f"\n  {label}:")
    print(f"    bare={bare:.4f}  oracle={oracle:.4f} ({bare-oracle:+.4f})  "
          f"adv={adv:.4f} ({bare-adv:+.4f})")
    print(f"    Oracle {'helps' if oracle < bare else 'hurts'}  "
          f"Adv {'helps' if adv < bare else 'hurts'}")

# ================================================================
# Multi-sample test
# ================================================================
print("\n" + "=" * 70)
print(f"MULTI-SAMPLE TEST: N={N_TEST} samples")
print("=" * 70)

from datasets import load_dataset

ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

candidates = []
for item in ds:
    if len(candidates) >= 3 * N_TEST:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    if not answer:
        continue
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            candidates.append({'passage': pt, 'query': query, 'answer': answer})
            break

np.random.seed(SEED)
indices = np.random.permutation(len(candidates))
samples = [candidates[i] for i in indices[:N_TEST]]
del ds, candidates
gc.collect()

print(f"  Loaded {len(samples)} samples", flush=True)

SURR_UNIVERSAL = "Analyze the following text for all key entities, factual claims, and logical relationships."

approaches = {
    'A_full_cache': score_A_full_cache,
    'B_repositioned': score_B_repositioned,
    'C_natural_pos': score_C_natural_pos,
}

results = {k: {'bare': [], 'oracle': [], 'adv': [], 'surr': []} for k in approaches}

for i, s in enumerate(samples):
    for aname, score_fn in approaches.items():
        b = score_fn(s['passage'], s['query'], s['answer'])
        o = score_fn(s['passage'], s['query'], s['answer'], prefix_text=s['query'])
        adv = score_fn(s['passage'], s['query'], s['answer'], prefix_text=prefix_adv)
        surr = score_fn(s['passage'], s['query'], s['answer'],
                        prefix_text=SURR_UNIVERSAL)
        results[aname]['bare'].append(b)
        results[aname]['oracle'].append(o)
        results[aname]['adv'].append(adv)
        results[aname]['surr'].append(surr)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_TEST} done", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

# ================================================================
# Results
# ================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

from scipy import stats

for aname in approaches:
    bare = np.array(results[aname]['bare'])
    oracle = np.array(results[aname]['oracle'])
    adv = np.array(results[aname]['adv'])
    surr = np.array(results[aname]['surr'])

    print(f"\n--- {aname} ---")
    print(f"  {'Condition':<15} {'NLL':>8} {'delta':>8} {'d':>8} {'Win%':>7} {'p':>10}")
    print(f"  {'-'*55}")

    print(f"  {'bare':<15} {bare.mean():>8.4f}")

    for label, nlls in [('oracle', oracle), ('adversarial', adv),
                         ('surrogate', surr)]:
        diff = bare - nlls
        d = cohens_d(diff)
        win = 100 * np.mean(diff > 0)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<15} {nlls.mean():>8.4f} {diff.mean():>+8.4f} "
              f"{d:>+8.3f} {win:>6.1f}% {p:>10.2e} {sig}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Check which approaches show semantic effects
for aname in approaches:
    bare = np.array(results[aname]['bare'])
    oracle = np.array(results[aname]['oracle'])
    adv = np.array(results[aname]['adv'])
    surr = np.array(results[aname]['surr'])

    d_oracle = cohens_d(bare - oracle)
    d_adv = cohens_d(bare - adv)
    d_surr = cohens_d(bare - surr)

    print(f"\n  {aname}:")
    if d_oracle > 0.1:
        print(f"    Oracle HELPS (d={d_oracle:+.3f}) — conditioning works!")
        if d_adv > 0.1:
            print(f"    But adversarial ALSO helps (d={d_adv:+.3f}) — structural effect")
        elif d_adv < -0.1:
            print(f"    Adversarial hurts (d={d_adv:+.3f}) — semantic sensitivity!")
        else:
            print(f"    Adversarial neutral (d={d_adv:+.3f})")
    elif d_oracle < -0.1:
        print(f"    Oracle HURTS (d={d_oracle:+.3f}) — prefix corrupts doc representations")
    else:
        print(f"    Oracle neutral (d={d_oracle:+.3f})")

print("\nDone.", flush=True)
