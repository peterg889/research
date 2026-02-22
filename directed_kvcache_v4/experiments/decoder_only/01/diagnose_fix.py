"""
Verify the look-ahead bug fix.

BUG: After slicing BOS from cache, cache has D entries but Phase B uses
cache_position=D+1 (matching RoPE). This creates a 1-position gap, causing
the causal mask (kv_idx <= q_idx) to allow each token to see the NEXT token.

FIX: Keep BOS in the cache. Then cache has D+1 entries, and Phase B starts
at cache_position=D+1, which matches get_seq_length()=D+1. No gap.

For conditioned cases, select BOS + doc from the full cache (skip prefix+\n).
"""
import os, warnings, torch, torch.nn.functional as F
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
print(f"Loaded.", flush=True)


def select_kv_cache(cache, indices):
    """Select specific cache indices (e.g., keep BOS + doc, skip prefix)."""
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def score_fixed(doc_text, query_text, answer_text, prefix_text=None):
    """Fixed scoring: keep BOS in cache, no look-ahead."""
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)

    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               truncation=True, max_length=512).input_ids
        P = len(prefix_ids)

        # Token sequence: [BOS, prefix, \n, doc]
        cond_ids = [BOS_ID] + prefix_ids + NL_IDS + doc_ids

        # Negative position prefix: BOS=0, prefix=negative, doc=1..D
        position_ids = torch.cat([
            torch.tensor([0], device=DEVICE),
            torch.arange(-(P + NL), -NL, device=DEVICE),
            torch.arange(-NL, 0, device=DEVICE),
            torch.arange(1, D + 1, device=DEVICE),
        ]).unsqueeze(0)
        cache_position = torch.arange(len(cond_ids), device=DEVICE)

        pa_kwargs = dict(
            input_ids=torch.tensor([cond_ids], device=DEVICE),
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        # Keep BOS (index 0) + doc (indices 1+P+NL .. end)
        keep_indices = [0] + list(range(1 + P + NL, len(cond_ids)))
    else:
        # Bare: [BOS, doc], default positions
        pa_kwargs = dict(
            input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
            use_cache=True,
        )
        keep_indices = None  # keep all

    with torch.no_grad():
        pa = model(**pa_kwargs)
    cache = pa.past_key_values
    del pa

    if keep_indices is not None:
        cache = select_kv_cache(cache, keep_indices)

    # Cache now has 1+D entries (BOS + doc)
    # Phase B starts at position D+1 = cache.get_seq_length()
    phase_b_start = D + 1
    assert cache.get_seq_length() == 1 + D, \
        f"Cache length {cache.get_seq_length()} != {1+D}"

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
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=False,
        )

    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del cache, pb
    return nll


# ================================================================
print("\n" + "=" * 70)
print("TEST 1: Fixed bare matches single-pass")
print("=" * 70)

doc_text = "The cat sat on the mat near the door of the house by the lake"
query_text = "Where did the cat sit?"
answer_text = "on the mat"

doc_ids = tokenizer(doc_text, add_special_tokens=False).input_ids
D = len(doc_ids)
query_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

# Single-pass reference
full_ids = [BOS_ID] + doc_ids + query_ids + answer_ids
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D + len(query_ids)
logits_f = out_full.logits[0, n_ctx-1:n_ctx-1+len(answer_ids), :].float()
targets = torch.tensor(answer_ids, device=DEVICE)
nll_single = -F.log_softmax(logits_f, dim=-1).gather(
    1, targets.unsqueeze(1)).squeeze(1).mean().item()
del out_full

# Fixed bare (keep BOS)
nll_bare_fixed = score_fixed(doc_text, query_text, answer_text)

diff_pct = abs(nll_single - nll_bare_fixed) / nll_single * 100
print(f"  Single-pass NLL:     {nll_single:.6f}")
print(f"  Fixed bare NLL:      {nll_bare_fixed:.6f}")
print(f"  Difference:          {diff_pct:.2f}%")
assert diff_pct < 1.0, f"Fixed bare doesn't match single-pass: {diff_pct}%"
print(f"  PASSED — fixed bare matches single-pass within {diff_pct:.2f}%")

# ================================================================
print("\n" + "=" * 70)
print("TEST 2: Fixed neg-pos conditioned")
print("=" * 70)

prefix_text = "Recipe for chocolate cake with butter and eggs"
nll_oracle_fixed = score_fixed(doc_text, query_text, answer_text,
                                prefix_text=query_text)
nll_adv_fixed = score_fixed(doc_text, query_text, answer_text,
                             prefix_text=prefix_text)

print(f"  Fixed bare:          {nll_bare_fixed:.6f}")
print(f"  Fixed oracle:        {nll_oracle_fixed:.6f} (delta: {nll_bare_fixed - nll_oracle_fixed:+.4f})")
print(f"  Fixed adversarial:   {nll_adv_fixed:.6f} (delta: {nll_bare_fixed - nll_adv_fixed:+.4f})")
print(f"  Oracle improves:     {'YES' if nll_oracle_fixed < nll_bare_fixed else 'NO'}")
print(f"  Adversarial:         {'helps' if nll_adv_fixed < nll_bare_fixed else 'hurts'}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 3: Multi-sample check with fixed scoring")
print("=" * 70)

# Load some MS MARCO samples
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from datasets import load_dataset
from lib.data import count_words
import numpy as np

np.random.seed(42)
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

test_samples = []
for item in ds:
    if len(test_samples) >= 20:
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
            test_samples.append({'passage': pt, 'query': query, 'answer': answer})
            break

indices = np.random.permutation(len(test_samples))
test_samples = [test_samples[i] for i in indices[:10]]

del ds
import gc
gc.collect()

print(f"  Testing {len(test_samples)} samples...")
bare_nlls = []
oracle_nlls = []
adv_nlls = []

ADVERSARIAL = "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt."

for i, s in enumerate(test_samples):
    b = score_fixed(s['passage'], s['query'], s['answer'])
    o = score_fixed(s['passage'], s['query'], s['answer'], prefix_text=s['query'])
    a = score_fixed(s['passage'], s['query'], s['answer'], prefix_text=ADVERSARIAL)
    bare_nlls.append(b)
    oracle_nlls.append(o)
    adv_nlls.append(a)
    delta_o = b - o
    delta_a = b - a
    print(f"  Sample {i}: bare={b:.4f}, oracle={o:.4f} ({delta_o:+.4f}), "
          f"adv={a:.4f} ({delta_a:+.4f})")

bare_arr = np.array(bare_nlls)
oracle_arr = np.array(oracle_nlls)
adv_arr = np.array(adv_nlls)

print(f"\n  Mean bare:      {bare_arr.mean():.4f}")
print(f"  Mean oracle:    {oracle_arr.mean():.4f} (delta: {(bare_arr - oracle_arr).mean():+.4f})")
print(f"  Mean adv:       {adv_arr.mean():.4f} (delta: {(bare_arr - adv_arr).mean():+.4f})")
print(f"  Oracle wins:    {np.sum(bare_arr > oracle_arr)}/{len(bare_arr)}")
print(f"  Adv wins:       {np.sum(bare_arr > adv_arr)}/{len(bare_arr)}")

# ================================================================
print("\n" + "=" * 70)
print("TEST 4: Value magnitudes with BOS retained")
print("=" * 70)

doc_text_4 = "The cat sat on the mat"
doc_ids_4 = tokenizer(doc_text_4, add_special_tokens=False).input_ids
D4 = len(doc_ids_4)
prefix_ids_4 = tokenizer(prefix_text, add_special_tokens=False).input_ids
P4 = len(prefix_ids_4)

# Bare cache (BOS + doc)
with torch.no_grad():
    out_b4 = model(input_ids=torch.tensor([[BOS_ID] + doc_ids_4], device=DEVICE),
                   use_cache=True)
cache_bare4 = out_b4.past_key_values
del out_b4

# Conditioned cache (BOS + doc selected from full)
cond_ids_4 = [BOS_ID] + prefix_ids_4 + NL_IDS + doc_ids_4
neg_pos_4 = torch.cat([
    torch.tensor([0], device=DEVICE),
    torch.arange(-(P4 + NL), -NL, device=DEVICE),
    torch.arange(-NL, 0, device=DEVICE),
    torch.arange(1, D4 + 1, device=DEVICE),
]).unsqueeze(0)
cond_cpos_4 = torch.arange(len(cond_ids_4), device=DEVICE)

with torch.no_grad():
    out_c4 = model(input_ids=torch.tensor([cond_ids_4], device=DEVICE),
                   position_ids=neg_pos_4, cache_position=cond_cpos_4,
                   use_cache=True)
keep_idx = [0] + list(range(1 + P4 + NL, len(cond_ids_4)))
cache_cond4 = select_kv_cache(out_c4.past_key_values, keep_idx)
del out_c4

text_cfg = getattr(model.config, 'text_config', model.config)
layer_types = getattr(text_cfg, 'layer_types', [])

print(f"  {'Layer':<6} {'Type':<5} {'Bare V mag':<12} {'Cond V mag':<12} {'Ratio':<8}")
print(f"  {'-'*50}")
for L in range(0, len(cache_bare4.layers), 4):
    bv = cache_bare4.layers[L].values.float()
    cv = cache_cond4.layers[L].values.float()
    lt = 'G' if layer_types[L] == 'full_attention' else 'L'
    ratio = cv.abs().mean().item() / (bv.abs().mean().item() + 1e-10)
    print(f"  {L:<6} {lt:<5} {bv.abs().mean().item():<12.4f} "
          f"{cv.abs().mean().item():<12.4f} {ratio:<8.2f}")

del cache_bare4, cache_cond4

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Fixed bare matches single-pass: {diff_pct:.2f}% difference")
print(f"  Oracle conditioning: {'helps' if nll_oracle_fixed < nll_bare_fixed else 'hurts/neutral'}")
print(f"  Adversarial prefix:  {'helps' if nll_adv_fixed < nll_bare_fixed else 'hurts/neutral'}")
print(f"\n  Multi-sample oracle win rate: {np.sum(bare_arr > oracle_arr)}/{len(bare_arr)}")
print(f"  Multi-sample adv win rate:    {np.sum(bare_arr > adv_arr)}/{len(bare_arr)}")

print("\nDone.", flush=True)
