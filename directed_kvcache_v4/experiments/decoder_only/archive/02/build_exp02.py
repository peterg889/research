#!/usr/bin/env python3
# Build Exp 02 notebook: Length vs Content Decomposition.
#
# decoder_only/01 found ALL surrogates beat oracle (recovery >100%), including
# adversarial. This suggests the effect is mostly structural (prefix length).
#
# This experiment disentangles LENGTH from CONTENT using Gemma 3 4B-PT
# (a larger model than exp01's Gemma 2 2B) with 10 carefully designed conditions.
#
# N=400, MS MARCO v1.1 validation.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 02: Length vs Content Decomposition

## Motivation

Exp 01 (Gemma 2 2B) found that **all surrogates beat the oracle** (recovery 115-146%),
including a completely off-topic adversarial prefix (124%). This suggests the effect
is primarily structural — any prefix text enriches document KV representations.

This experiment disentangles **prefix length** from **prefix content** using a larger
model (Gemma 3 4B-PT) with 10 conditions designed as a controlled factorial.

## Conditions (10 total)

### Baselines
| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | No prefix — lower bound |
| 2 | oracle | Real query as prefix — upper bound |

### Content at surrogate length (~15 words)
| # | Condition | Description |
|---|-----------|-------------|
| 3 | surr_reasonant | "Evaluate the underlying arguments..." (best from exp01) |
| 4 | surr_universal | "Analyze the following text..." |
| 5 | adversarial | Off-topic text (~15 words) |

### Length sweep (random words, varying count)
| # | Condition | Description |
|---|-----------|-------------|
| 6 | random_matched | Random words, same count as query (per-sample) |
| 7 | random_15w | 15 random words |
| 8 | random_30w | 30 random words |

### Controls
| # | Condition | Description |
|---|-----------|-------------|
| 9 | repeat_15w | "the" × 15 — minimal content, same length |
| 10 | oracle_padded | Query + "the" padding to 15 words |

## Key comparisons

1. **Pure length**: random_matched → random_15w → random_30w
2. **Content at oracle's length**: oracle vs random_matched
3. **Content at surrogate length**: surr_reasonant vs random_15w vs adversarial
4. **Token diversity**: repeat_15w vs random_15w (both 15 words)
5. **Oracle + length boost**: oracle_padded vs oracle""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup and model loading
import os
os.umask(0o000)

import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 400
MODEL_NAME = "google/gemma-3-4b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp02")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device

print(f"Exp 02: Length vs Content Decomposition")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Config: {type(model.config).__name__}")
text_cfg = getattr(model.config, 'text_config', model.config)
print(f"Vocab size: {getattr(text_cfg, 'vocab_size', 'N/A')}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
""")


# ===== Cell 3: KV cache helpers + scoring =====
code(r"""# Cell 3: KV cache helpers and scoring function

def slice_kv_cache(cache, start_idx):
    # Remove first start_idx entries from KV cache.
    from transformers import DynamicCache

    if isinstance(cache, DynamicCache):
        sliced = DynamicCache()
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys[:, :, start_idx:, :]
            v = cache.layers[i].values[:, :, start_idx:, :]
            sliced.update(k, v, i)
        return sliced
    else:
        return tuple(
            (k[:, :, start_idx:, :], v[:, :, start_idx:, :])
            for k, v in cache
        )


def score(doc_text, query_text, answer_text, prefix_text=None):
    # Score NLL of answer tokens using two-phase KV cache approach.
    #
    # Phase A: Forward [prefix + doc] (or just [doc]) -> KV cache.
    # Phase B: Forward [query + answer] using cached doc KV.
    # If prefix_text is provided, prefix KV entries are sliced off.

    # --- Phase A: Conditioning ---
    if prefix_text:
        prefix_ids = tokenizer(prefix_text + "\n", add_special_tokens=True,
                               truncation=True, max_length=512).input_ids
        doc_ids = tokenizer(doc_text, add_special_tokens=False,
                            truncation=True, max_length=1536).input_ids
        cond_ids = prefix_ids + doc_ids
        slice_start = len(prefix_ids)
    else:
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        slice_start = 0

    cond_tensor = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)
    total_cond_len = len(cond_ids)

    with torch.no_grad():
        phase_a = model(input_ids=cond_tensor, use_cache=True)

    cache = phase_a.past_key_values
    del phase_a

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

    # --- Phase B: Inference with query + answer ---
    query_part_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids

    if not answer_ids:
        del cache
        return 0.0

    phase_b_ids = query_part_ids + answer_ids
    phase_b_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=DEVICE)

    pos_ids = torch.arange(total_cond_len, total_cond_len + len(phase_b_ids),
                           device=DEVICE).unsqueeze(0)

    cache_position = torch.arange(total_cond_len, total_cond_len + len(phase_b_ids),
                                  device=DEVICE)

    with torch.no_grad():
        phase_b = model(
            input_ids=phase_b_tensor,
            past_key_values=cache,
            position_ids=pos_ids,
            cache_position=cache_position,
            use_cache=False,
        )

    logits = phase_b.logits
    n_query_part = len(query_part_ids)
    n_answer = len(answer_ids)

    answer_logits = logits[0, n_query_part - 1 : n_query_part - 1 + n_answer, :]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del cache, phase_b, logits, log_probs
    return mean_nll


def score_full_sequence(doc_text, query_text, answer_text):
    # Single-pass scoring for validation.
    doc_ids = tokenizer(doc_text, add_special_tokens=True,
                        truncation=True, max_length=2048).input_ids
    query_part_ids = tokenizer("\n" + query_text + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids

    if not answer_ids:
        return 0.0

    all_ids = doc_ids + query_part_ids + answer_ids
    input_tensor = torch.tensor([all_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, use_cache=False)

    logits = outputs.logits
    n_doc = len(doc_ids)
    n_query = len(query_part_ids)
    n_answer = len(answer_ids)

    start = n_doc + n_query - 1
    answer_logits = logits[0, start : start + n_answer, :]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del outputs, logits, log_probs
    return mean_nll


# === Surrogate definitions ===
SURR_REASONANT = "Evaluate the underlying arguments, sentiment, and intent of the following passage."
SURR_UNIVERSAL = "Analyze the following text for all key entities, factual claims, and logical relationships."
ADVERSARIAL_15W = "The recipe calls for two cups of flour, one cup of sugar, a pinch of salt, and some butter."

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'if', 'then', 'than', 'so', 'up', 'out', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'they', 'them', 'their', 'how', 'when',
    'where', 'why', 'much', 'many', 'some', 'any', 'all', 'each',
    'does', 'also', 'just', 'more', 'most', 'very', 'too', 'only',
}

def extract_keywords(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

print("Scoring functions defined.")
print(f"\nPrefix token counts:")
for name, text in [('surr_reasonant', SURR_REASONANT),
                    ('surr_universal', SURR_UNIVERSAL),
                    ('adversarial_15w', ADVERSARIAL_15W)]:
    n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
    n_words = len(text.split())
    print(f"  {name:<20} {n_words:>3}w {n_tok:>3}tok: {text[:50]}...")
print(f"  {'repeat_15w':<20} {'15':>3}w {'?':>3}tok: the the the the the ...")
the_15 = " ".join(["the"] * 15)
n_tok_the = len(tokenizer(the_15, add_special_tokens=False).input_ids)
print(f"  {'':>24} -> {n_tok_the} tokens")
""")


# ===== Cell 4: Load data =====
code(r"""# Cell 4: Load MS MARCO data and generate per-sample prefixes
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
for item in ds:
    if len(all_candidates) >= 3 * N_SAMPLES:
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
            all_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

print(f"Total candidates: {len(all_candidates)}")
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

# Build a pool of random English words from OTHER passages for random prefixes.
# Use the second half of the shuffled data as the word pool (no overlap).
word_pool = []
for i in indices[N_SAMPLES:N_SAMPLES + 500]:
    words = samples[0]['passage'].split() if i >= len(samples) else []
    # Use all_candidates — but it's deleted. Rebuild from samples neighbors.
    pass

# Simpler: collect words from all passages, shuffle once
all_words = []
for s in samples:
    all_words.extend(s['passage'].split())
pyrandom.seed(SEED + 99)
pyrandom.shuffle(all_words)
word_pool = all_words  # ~30k words, plenty for random prefixes

# Generate per-sample prefixes
for i, s in enumerate(samples):
    query_wc = len(s['query'].split())

    # random_matched: same word count as query, from word pool
    pool_offset = i * 50  # each sample gets a different slice
    s['random_matched'] = " ".join(word_pool[pool_offset:pool_offset + query_wc])

    # random_15w: 15 words from pool
    s['random_15w'] = " ".join(word_pool[pool_offset + 50:pool_offset + 65])

    # random_30w: 30 words from pool
    s['random_30w'] = " ".join(word_pool[pool_offset + 65:pool_offset + 95])

    # repeat_15w: "the" x 15
    s['repeat_15w'] = " ".join(["the"] * 15)

    # oracle_padded: query + "the" to reach 15 words
    pad_count = max(0, 15 - query_wc)
    s['oracle_padded'] = s['query'] + " " + " ".join(["the"] * pad_count)

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([len(s['query'].split()) for s in samples]):.1f}")

# Show prefix word count stats
print(f"\nPrefix word counts:")
for key in ['random_matched', 'random_15w', 'random_30w', 'repeat_15w', 'oracle_padded']:
    wcs = [len(s[key].split()) for s in samples]
    print(f"  {key:<20} mean={np.mean(wcs):.1f}, range=[{min(wcs)}, {max(wcs)}]")

print(f"\nSample 0:")
print(f"  Query ({len(samples[0]['query'].split())}w): {samples[0]['query']}")
print(f"  random_matched:     {samples[0]['random_matched']}")
print(f"  random_15w:         {samples[0]['random_15w'][:60]}...")
print(f"  random_30w:         {samples[0]['random_30w'][:60]}...")
print(f"  repeat_15w:         {samples[0]['repeat_15w']}")
print(f"  oracle_padded:      {samples[0]['oracle_padded']}")
""")


# ===== Cell 5: Validation =====
code(r"""# Cell 5: Validate two-phase caching vs single-pass
print("=" * 70)
print("VALIDATION: Two-phase caching vs single-pass")
print("=" * 70)

print("\nComparing bare score (cached) vs full-sequence for 5 samples...")
max_diff = 0.0
for i in range(5):
    s = samples[i]
    nll_cached = score(s['passage'], s['query'], s['answer'], prefix_text=None)
    nll_full = score_full_sequence(s['passage'], s['query'], s['answer'])
    diff = abs(nll_cached - nll_full)
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 0.01 else "MISMATCH"
    print(f"  Sample {i}: cached={nll_cached:.6f}, full={nll_full:.6f}, "
          f"diff={diff:.8f} [{status}]")

if max_diff < 0.01:
    print(f"\nVALIDATION PASSED (max diff = {max_diff:.8f})")
else:
    print(f"\nWARNING: max diff = {max_diff:.8f}")

# Quick sanity: conditioned vs bare
print(f"\nQuick sanity check (sample 0):")
nll_bare = score(samples[0]['passage'], samples[0]['query'], samples[0]['answer'])
nll_oracle = score(samples[0]['passage'], samples[0]['query'], samples[0]['answer'],
                   prefix_text=samples[0]['query'])
nll_random = score(samples[0]['passage'], samples[0]['query'], samples[0]['answer'],
                   prefix_text=samples[0]['random_15w'])
print(f"  bare:      {nll_bare:.6f}")
print(f"  oracle:    {nll_oracle:.6f} (delta: {nll_bare - nll_oracle:+.4f})")
print(f"  random_15w:{nll_random:.6f} (delta: {nll_bare - nll_random:+.4f})")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 10 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle',
    'surr_reasonant', 'surr_universal', 'adversarial',
    'random_matched', 'random_15w', 'random_30w',
    'repeat_15w', 'oracle_padded',
]

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            results = ckpt['results']
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
        'query_words': len(query.split()),
    }

    # 1. bare
    result['nll_bare'] = score(passage, query, answer)

    # 2. oracle
    result['nll_oracle'] = score(passage, query, answer, prefix_text=query)

    # 3. surr_reasonant
    result['nll_surr_reasonant'] = score(passage, query, answer,
                                         prefix_text=SURR_REASONANT)

    # 4. surr_universal
    result['nll_surr_universal'] = score(passage, query, answer,
                                          prefix_text=SURR_UNIVERSAL)

    # 5. adversarial
    result['nll_adversarial'] = score(passage, query, answer,
                                      prefix_text=ADVERSARIAL_15W)

    # 6. random_matched (same word count as query)
    result['nll_random_matched'] = score(passage, query, answer,
                                          prefix_text=s['random_matched'])

    # 7. random_15w
    result['nll_random_15w'] = score(passage, query, answer,
                                      prefix_text=s['random_15w'])

    # 8. random_30w
    result['nll_random_30w'] = score(passage, query, answer,
                                      prefix_text=s['random_30w'])

    # 9. repeat_15w ("the" x 15)
    result['nll_repeat_15w'] = score(passage, query, answer,
                                      prefix_text=s['repeat_15w'])

    # 10. oracle_padded (query + "the" to 15 words)
    result['nll_oracle_padded'] = score(passage, query, answer,
                                         prefix_text=s['oracle_padded'])

    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = i - start_idx + 1
        eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 7: Results table =====
code(r"""# Cell 7: Results table
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

# Extract arrays
arrays = {}
for name in ['bare', 'oracle', 'surr_reasonant', 'surr_universal', 'adversarial',
             'random_matched', 'random_15w', 'random_30w', 'repeat_15w', 'oracle_padded']:
    arrays[name] = np.array([r[f'nll_{name}'] for r in results])

bare = arrays['bare']
oracle = arrays['oracle']
oracle_delta_mean = (bare - oracle).mean()

print(f"\n  {'Condition':<20} {'~words':>6} {'NLL':>8} {'vs bare':>10} {'d':>8} "
      f"{'Win%':>8} {'p':>12} {'sig':>5} {'Recovery':>10}")
print(f"  {'-'*98}")

# Approximate word counts for display
approx_words = {
    'bare': 0, 'oracle': 6, 'surr_reasonant': 11, 'surr_universal': 13,
    'adversarial': 15, 'random_matched': 6, 'random_15w': 15, 'random_30w': 30,
    'repeat_15w': 15, 'oracle_padded': 15,
}

analysis = {}
for name in ['bare', 'oracle', 'surr_reasonant', 'surr_universal', 'adversarial',
             'random_matched', 'random_15w', 'random_30w', 'repeat_15w', 'oracle_padded']:
    nlls = arrays[name]
    mean_nll = nlls.mean()
    aw = approx_words[name]

    if name == 'bare':
        print(f"  {name:<20} {aw:>6} {mean_nll:>8.4f} {'--':>10} {'--':>8} "
              f"{'--':>8} {'--':>12} {'--':>5} {'--':>10}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        if oracle_delta_mean > 0:
            recovery = diff.mean() / oracle_delta_mean * 100
            rec_str = f"{recovery:>9.1f}%"
        else:
            recovery = float('nan')
            rec_str = "n/a"

        print(f"  {name:<20} {aw:>6} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {rec_str:>10}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            'recovery': float(recovery) if not np.isnan(recovery) else None,
        }
""")


# ===== Cell 8: Length vs Content decomposition =====
code(r"""# Cell 8: Length vs Content decomposition
print("=" * 70)
print("DECOMPOSITION: LENGTH vs CONTENT")
print("=" * 70)

# --- 1. Pure length effect (random words at different counts) ---
print(f"\n--- 1. PURE LENGTH EFFECT (random words, varying count) ---")
print(f"  {'Condition':<20} {'~words':>6} {'d vs bare':>10} {'p':>12}")
print(f"  {'-'*55}")
for name, nw in [('random_matched', '~6'), ('random_15w', '15'), ('random_30w', '30')]:
    diff = bare - arrays[name]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<20} {nw:>6} {d:>+10.4f} {p:>12.2e} {sig}")

# Length gradient
d_matched = cohens_d(bare - arrays['random_matched'])
d_15 = cohens_d(bare - arrays['random_15w'])
d_30 = cohens_d(bare - arrays['random_30w'])
if d_matched > 0:
    print(f"\n  Length scaling: 15w/matched = {d_15/d_matched:.2f}x, "
          f"30w/matched = {d_30/d_matched:.2f}x")
    if d_30 > d_15 * 1.2:
        print(f"  -> Benefit INCREASES with length (not saturated)")
    elif d_30 > d_15 * 0.9:
        print(f"  -> Benefit roughly FLAT from 15w to 30w (saturated)")
    else:
        print(f"  -> Benefit DECREASES at 30w (diminishing returns)")

# --- 2. Content effect at oracle's length (~6 words) ---
print(f"\n--- 2. CONTENT EFFECT at oracle's length (~6 words) ---")
diff_oracle_vs_random = arrays['random_matched'] - arrays['oracle']
d_content_oracle = cohens_d(diff_oracle_vs_random)
_, p_content_oracle = stats.ttest_1samp(diff_oracle_vs_random, 0)
sig_co = '***' if p_content_oracle < 0.001 else '**' if p_content_oracle < 0.01 else '*' if p_content_oracle < 0.05 else 'ns'

d_oracle = cohens_d(bare - arrays['oracle'])
d_random_m = cohens_d(bare - arrays['random_matched'])

print(f"  oracle        d vs bare = {d_oracle:+.4f}")
print(f"  random_matched d vs bare = {d_random_m:+.4f}")
print(f"  oracle vs random_matched: d = {d_content_oracle:+.4f} ({sig_co})")
if d_oracle > 0 and d_random_m > 0:
    content_pct_oracle = (d_oracle - d_random_m) / d_oracle * 100
    print(f"  -> Content accounts for {content_pct_oracle:.0f}% of oracle benefit at this length")
    print(f"  -> Structure accounts for {100 - content_pct_oracle:.0f}%")

# --- 3. Content effect at surrogate length (~15 words) ---
print(f"\n--- 3. CONTENT EFFECT at surrogate length (~15 words) ---")
print(f"  {'Condition':<20} {'Content':>12} {'d vs bare':>10}")
print(f"  {'-'*48}")
for name, content_type in [('repeat_15w', 'none'),
                            ('random_15w', 'random'),
                            ('adversarial', 'off-topic'),
                            ('surr_universal', 'generic'),
                            ('surr_reasonant', 'targeted')]:
    d = cohens_d(bare - arrays[name])
    print(f"  {name:<20} {content_type:>12} {d:>+10.4f}")

# Pairwise: surr_reasonant vs random_15w
diff_sr = arrays['random_15w'] - arrays['surr_reasonant']
d_sr = cohens_d(diff_sr)
_, p_sr = stats.ttest_1samp(diff_sr, 0)
sig_sr = '***' if p_sr < 0.001 else '**' if p_sr < 0.01 else '*' if p_sr < 0.05 else 'ns'
print(f"\n  surr_reasonant vs random_15w: d = {d_sr:+.4f} ({sig_sr})")
print(f"  -> {'Content matters' if p_sr < 0.05 else 'No significant content effect'} "
      f"at matched length")

# Pairwise: random_15w vs repeat_15w
diff_rr = arrays['repeat_15w'] - arrays['random_15w']
d_rr = cohens_d(diff_rr)
_, p_rr = stats.ttest_1samp(diff_rr, 0)
sig_rr = '***' if p_rr < 0.001 else '**' if p_rr < 0.01 else '*' if p_rr < 0.05 else 'ns'
print(f"  random_15w vs repeat_15w: d = {d_rr:+.4f} ({sig_rr})")
print(f"  -> {'Token diversity matters' if p_rr < 0.05 else 'Token diversity does NOT matter'}")

# --- 4. Oracle + length boost ---
print(f"\n--- 4. ORACLE + LENGTH BOOST ---")
d_oracle_plain = cohens_d(bare - arrays['oracle'])
d_oracle_pad = cohens_d(bare - arrays['oracle_padded'])
diff_pad = arrays['oracle'] - arrays['oracle_padded']
d_pad_boost = cohens_d(diff_pad)
_, p_pad = stats.ttest_1samp(diff_pad, 0)
sig_pad = '***' if p_pad < 0.001 else '**' if p_pad < 0.01 else '*' if p_pad < 0.05 else 'ns'

print(f"  oracle (natural):  d = {d_oracle_plain:+.4f}")
print(f"  oracle_padded(15w): d = {d_oracle_pad:+.4f}")
print(f"  padding boost: d = {d_pad_boost:+.4f} ({sig_pad})")
if d_pad_boost > 0.05:
    print(f"  -> Padding oracle to 15w IMPROVES it — length matters even with real query")
elif d_pad_boost < -0.05:
    print(f"  -> Padding oracle HURTS — the extra 'the' tokens add noise")
else:
    print(f"  -> Padding has no significant effect on oracle")
""")


# ===== Cell 9: Summary decomposition =====
code(r"""# Cell 9: Summary — how much is length vs content?
print("=" * 70)
print("SUMMARY DECOMPOSITION")
print("=" * 70)

d_bare = 0  # reference
d_oracle = cohens_d(bare - arrays['oracle'])
d_random_m = cohens_d(bare - arrays['random_matched'])
d_random_15 = cohens_d(bare - arrays['random_15w'])
d_random_30 = cohens_d(bare - arrays['random_30w'])
d_repeat_15 = cohens_d(bare - arrays['repeat_15w'])
d_surr_r = cohens_d(bare - arrays['surr_reasonant'])
d_surr_u = cohens_d(bare - arrays['surr_universal'])
d_adv = cohens_d(bare - arrays['adversarial'])
d_oracle_pad = cohens_d(bare - arrays['oracle_padded'])

print(f"\n  Effect sizes (Cohen's d vs bare):")
print(f"  {'':>30} {'d':>8}")
print(f"  {'-'*42}")
print(f"  {'bare (reference)':>30} {0:>+8.4f}")
print(f"  {'--- ~6 words ---':>30}")
print(f"  {'random_matched (~6w)':>30} {d_random_m:>+8.4f}")
print(f"  {'oracle (~6w)':>30} {d_oracle:>+8.4f}")
print(f"  {'--- ~15 words ---':>30}")
print(f"  {'repeat_15w':>30} {d_repeat_15:>+8.4f}")
print(f"  {'random_15w':>30} {d_random_15:>+8.4f}")
print(f"  {'adversarial (~15w)':>30} {d_adv:>+8.4f}")
print(f"  {'oracle_padded (15w)':>30} {d_oracle_pad:>+8.4f}")
print(f"  {'surr_universal (~13w)':>30} {d_surr_u:>+8.4f}")
print(f"  {'surr_reasonant (~11w)':>30} {d_surr_r:>+8.4f}")
print(f"  {'--- ~30 words ---':>30}")
print(f"  {'random_30w':>30} {d_random_30:>+8.4f}")

# Decompose the best surrogate's effect
print(f"\n  Decomposition of surr_reasonant (d = {d_surr_r:+.4f}):")
length_component = d_random_15  # effect of having ~15 random words
content_component = d_surr_r - d_random_15
total = d_surr_r
if total > 0:
    print(f"    Length (~15 random words):  {length_component:+.4f} ({length_component/total*100:.0f}%)")
    print(f"    Content (reasonant vs random): {content_component:+.4f} ({content_component/total*100:.0f}%)")

# Decompose oracle's effect
print(f"\n  Decomposition of oracle (d = {d_oracle:+.4f}):")
length_component_o = d_random_m  # effect of having ~6 random words
content_component_o = d_oracle - d_random_m
if d_oracle > 0:
    print(f"    Length (~6 random words):   {length_component_o:+.4f} ({length_component_o/d_oracle*100:.0f}%)")
    print(f"    Content (real query vs random): {content_component_o:+.4f} ({content_component_o/d_oracle*100:.0f}%)")

# Interpret
print(f"\n  INTERPRETATION:")
if d_random_15 > d_surr_r * 0.8:
    print(f"  -> STRUCTURAL DOMINANCE: random words at 15w achieve {d_random_15/d_surr_r*100:.0f}% of best surrogate")
    print(f"     Content adds only marginal value. The effect is primarily about")
    print(f"     giving doc tokens additional causal context to attend to.")
elif d_random_15 > d_surr_r * 0.5:
    print(f"  -> MIXED: length provides {d_random_15/d_surr_r*100:.0f}% of surrogate benefit,")
    print(f"     but content contributes a meaningful additional {(d_surr_r - d_random_15)/d_surr_r*100:.0f}%.")
else:
    print(f"  -> CONTENT DOMINATES: random words at 15w only achieve {d_random_15/d_surr_r*100:.0f}%.")
    print(f"     The semantic content of the prefix is the primary driver.")
""")


# ===== Cell 10: Verdict + save =====
code(r"""# Cell 10: Verdict and save
print("=" * 70)
print("VERDICT — Decoder-Only Exp 02: Length vs Content")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")

d_oracle = cohens_d(bare - arrays['oracle'])
d_surr_r = cohens_d(bare - arrays['surr_reasonant'])
d_random_15 = cohens_d(bare - arrays['random_15w'])
d_random_m = cohens_d(bare - arrays['random_matched'])

print(f"\n--- Key findings ---")
print(f"  1. Oracle (real query): d = {d_oracle:+.4f}")
print(f"  2. Best surrogate:     d = {d_surr_r:+.4f}")
print(f"  3. Random 15w:         d = {d_random_15:+.4f}")
print(f"  4. Random matched:     d = {d_random_m:+.4f}")

# Compare to exp01 (Gemma 2 2B)
print(f"\n--- Model comparison (vs exp01 Gemma 2 2B) ---")
print(f"  Exp01 oracle d = +0.440, surr_reasonant d = +0.647")
print(f"  Exp02 oracle d = {d_oracle:+.4f}, surr_reasonant d = {d_surr_r:+.4f}")
if d_oracle > 0.3:
    print(f"  -> Effect REPLICATES on larger model")
elif d_oracle > 0.1:
    print(f"  -> Weaker but present on larger model")
else:
    print(f"  -> Effect FAILS to replicate on larger model")

# All conditions summary
print(f"\n--- All conditions ---")
for name in ['bare', 'oracle', 'surr_reasonant', 'surr_universal', 'adversarial',
             'random_matched', 'random_15w', 'random_30w', 'repeat_15w', 'oracle_padded']:
    if name == 'bare':
        print(f"  {name:<20} NLL = {arrays[name].mean():.4f}")
    else:
        d = cohens_d(bare - arrays[name])
        _, p = stats.ttest_1samp(bare - arrays[name], 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {name:<20} NLL = {arrays[name].mean():.4f}  d = {d:+.4f} ({sig})")

# Save
final_results = {
    'experiment': 'v4_decoder_only_exp02_length_vs_content',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {k: v for k, v in analysis.items()},
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/decoder_only/02/02_length_vs_content.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
