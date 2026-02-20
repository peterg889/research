#!/usr/bin/env python3
# Build Exp 02 notebook: Decoder-Only Surrogate KV Caching.
#
# Tests whether conditioning a decoder-only model's KV cache with surrogate
# prompts improves answer quality. Uses actual KV cache slicing + position ID
# alignment — the production deployment approach.
#
# 8 conditions: bare, oracle, 4 surrogate types, doc keywords, adversarial.
# N=400, MS MARCO v1.1 validation, Gemma 2 2B.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 02: Decoder-Only Surrogate KV Caching

## Motivation

v4 Exp 01 tested surrogate enrichment with T5Gemma (encoder-decoder).
This experiment tests the same concept with a **decoder-only** model using
actual KV cache manipulation — the production deployment approach.

In a causal (decoder-only) model, document tokens D cannot attend to a query Q
that comes after them. This creates a performance gap vs. bidirectional models.
We address this by using a **surrogate query** during offline cache generation
to "condition" D's KV representations.

## Method

**Phase A — Offline Cache Generation:**
1. Construct `[surrogate + document]`
2. Forward pass with `use_cache=True` → full KV cache
3. Slice: remove surrogate entries, keep only document KV
4. The document KV now encodes features influenced by the surrogate

**Phase B — Online Inference:**
1. Load sliced document KV cache
2. Set `position_ids` for query to start after the document's original positions
3. Forward pass: query + answer attend to conditioned document KV
4. Compute NLL on answer tokens

**Position alignment**: If the surrogate was S tokens and document is D tokens,
the cached document occupies positions S through S+D-1. New query tokens start
at position S+D. This preserves correct RoPE relative distances.

**Key insight from v3**: RoPE-based attention depends on relative positions.
Doc-to-query relative distance is identical across conditions, so the position
offset from the surrogate does NOT confound the comparison.

## Conditions (8 total)

| # | Condition | Prefix | Slice? | Description |
|---|-----------|--------|--------|-------------|
| 1 | bare | (none) | no | Standard causal — lower bound |
| 2 | oracle | real query | yes | Real query conditions doc — upper bound |
| 3 | surr_universal | generic analysis | yes | "Analyze for entities, facts, relationships" |
| 4 | surr_extractor | data extraction | yes | "Examine for data points, dates, attributes" |
| 5 | surr_reasonant | reasoning | yes | "Evaluate arguments, sentiment, intent" |
| 6 | surr_analytic | technical | yes | "Technical breakdown of systems/processes" |
| 7 | surr_doc_kw | doc keywords | yes | Top-5 document keywords (v3's best) |
| 8 | adversarial | off-topic | yes | Off-topic text — negative control |

## Key metrics
- **Recovery rate**: (surrogate − bare) / (oracle − bare) × 100%
- Cohen's d, win%, paired t-test
- Hardness gradient analysis""")


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

RESULTS_DIR = Path("../../../results/decoder_only/exp01")
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

print(f"Exp 02: Decoder-Only Surrogate KV Caching")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
print(f"Vocab size: {getattr(text_cfg, 'vocab_size', 'N/A')}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
""")


# ===== Cell 3: KV cache helpers + scoring =====
code(r"""# Cell 3: KV cache helpers and scoring function

def slice_kv_cache(cache, start_idx):
    # Remove first start_idx entries from KV cache.
    # DynamicCache API: cache.layers[i].keys / .values are tensors.
    from transformers import DynamicCache

    if isinstance(cache, DynamicCache):
        sliced = DynamicCache()
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys[:, :, start_idx:, :]
            v = cache.layers[i].values[:, :, start_idx:, :]
            sliced.update(k, v, i)
        return sliced
    else:
        # Tuple-of-tuples fallback
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
    #
    # Returns: mean NLL over answer tokens.

    # --- Phase A: Conditioning ---
    if prefix_text:
        # Tokenize prefix and doc separately to know exact split point
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

    # Slice prefix from cache
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

    # Position IDs: new tokens start after the original conditioning sequence
    pos_ids = torch.arange(total_cond_len, total_cond_len + len(phase_b_ids),
                           device=DEVICE).unsqueeze(0)

    # Cache position for correct causal mask computation
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

    # NLL on answer tokens only
    logits = phase_b.logits  # [1, n_phase_b, vocab]
    n_query_part = len(query_part_ids)
    n_answer = len(answer_ids)

    # logits[0, t, :] predicts token at position t+1 in the new sequence
    # Answer starts at index n_query_part in phase_b sequence
    # To predict answer[0], need logits at index n_query_part - 1
    answer_logits = logits[0, n_query_part - 1 : n_query_part - 1 + n_answer, :]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del cache, phase_b, logits, log_probs
    return mean_nll


def score_full_sequence(doc_text, query_text, answer_text):
    # Score NLL with a single forward pass: [doc + query + answer].
    # Used for validation against the two-phase approach.
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

    # Answer starts at index n_doc + n_query in the full sequence
    # To predict answer[0], need logits at index n_doc + n_query - 1
    start = n_doc + n_query - 1
    answer_logits = logits[0, start : start + n_answer, :]
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del outputs, logits, log_probs
    return mean_nll


# === Surrogate definitions ===
SURROGATES = {
    'universal': "Analyze the following text for all key entities, factual claims, and logical relationships.",
    'extractor': "Examine this document specifically for data points, dates, numerical values, and specific named attributes.",
    'reasonant': "Evaluate the underlying arguments, sentiment, and intent of the following passage.",
    'analytic': "Provide a technical breakdown of the systems and processes described in this text.",
}

ADVERSARIAL_PREFIX = "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt."

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

def make_doc_keywords(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))


print("Scoring functions defined.")
print(f"\nSurrogate prompts:")
for name, prompt in SURROGATES.items():
    n_tok = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    print(f"  {name:<12} ({n_tok:>2} tok): {prompt[:60]}...")
adv_tok = len(tokenizer(ADVERSARIAL_PREFIX, add_special_tokens=False).input_ids)
print(f"  {'adversarial':<12} ({adv_tok:>2} tok): {ADVERSARIAL_PREFIX[:60]}...")
""")


# ===== Cell 4: Load data =====
code(r"""# Cell 4: Load MS MARCO data and generate surrogates
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

# Generate surrogates
for s in samples:
    s['surr_doc_kw'] = make_doc_keywords(s['passage'])

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"\nFirst sample:")
print(f"  Query:  {samples[0]['query'][:70]}...")
print(f"  Answer: {samples[0]['answer'][:70]}...")
print(f"  Passage ({samples[0]['word_count']}w): {samples[0]['passage'][:70]}...")
print(f"  Doc keywords: {samples[0]['surr_doc_kw']}")
""")


# ===== Cell 5: Validation =====
code(r"""# Cell 5: Validate two-phase caching matches single-pass (bare condition)
print("=" * 70)
print("VALIDATION: Two-phase caching vs single-pass")
print("=" * 70)

print("\nComparing bare score (cached) vs full-sequence score for 5 samples...")
max_diff = 0.0
for i in range(5):
    s = samples[i]
    nll_cached = score(s['passage'], s['query'], s['answer'], prefix_text=None)
    nll_full = score_full_sequence(s['passage'], s['query'], s['answer'])
    diff = abs(nll_cached - nll_full)
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 0.001 else "MISMATCH"
    print(f"  Sample {i}: cached={nll_cached:.6f}, full={nll_full:.6f}, "
          f"diff={diff:.8f} [{status}]")

if max_diff < 0.001:
    print(f"\nVALIDATION PASSED: max diff = {max_diff:.8f}")
else:
    print(f"\nWARNING: max diff = {max_diff:.8f} — investigate before proceeding")

# Quick test: conditioned score runs without error
print(f"\nQuick test: oracle score for sample 0...")
nll_oracle = score(samples[0]['passage'], samples[0]['query'], samples[0]['answer'],
                   prefix_text=samples[0]['query'])
nll_bare = score(samples[0]['passage'], samples[0]['query'], samples[0]['answer'])
print(f"  bare:   {nll_bare:.6f}")
print(f"  oracle: {nll_oracle:.6f}")
print(f"  delta:  {nll_bare - nll_oracle:+.6f} (positive = oracle better)")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 8 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle',
    'surr_universal', 'surr_extractor', 'surr_reasonant', 'surr_analytic',
    'surr_doc_kw', 'adversarial',
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
    }

    # 1. bare — no prefix
    result['nll_bare'] = score(passage, query, answer)

    # 2. oracle — real query as prefix
    result['nll_oracle'] = score(passage, query, answer, prefix_text=query)

    # 3-6. Surrogate prompts
    for surr_name, surr_prompt in SURROGATES.items():
        result[f'nll_surr_{surr_name}'] = score(
            passage, query, answer, prefix_text=surr_prompt)

    # 7. doc keywords
    result['nll_surr_doc_kw'] = score(
        passage, query, answer, prefix_text=s['surr_doc_kw'])

    # 8. adversarial
    result['nll_adversarial'] = score(
        passage, query, answer, prefix_text=ADVERSARIAL_PREFIX)

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

bare = np.array([r['nll_bare'] for r in results])
oracle = np.array([r['nll_oracle'] for r in results])
surr_universal = np.array([r['nll_surr_universal'] for r in results])
surr_extractor = np.array([r['nll_surr_extractor'] for r in results])
surr_reasonant = np.array([r['nll_surr_reasonant'] for r in results])
surr_analytic = np.array([r['nll_surr_analytic'] for r in results])
surr_doc_kw = np.array([r['nll_surr_doc_kw'] for r in results])
adversarial = np.array([r['nll_adversarial'] for r in results])

print(f"\n  {'Condition':<20} {'NLL':>8} {'vs bare':>10} {'d':>8} {'Win%':>8} "
      f"{'p':>12} {'sig':>5} {'Recovery':>10}")
print(f"  {'-'*85}")

# Oracle delta for recovery calculation
oracle_delta_mean = (bare - oracle).mean()
oracle_d = cohens_d(bare - oracle)

all_conds = [
    ('bare', bare),
    ('oracle', oracle),
    ('surr_universal', surr_universal),
    ('surr_extractor', surr_extractor),
    ('surr_reasonant', surr_reasonant),
    ('surr_analytic', surr_analytic),
    ('surr_doc_kw', surr_doc_kw),
    ('adversarial', adversarial),
]

analysis = {}
for name, nlls in all_conds:
    mean_nll = nlls.mean()
    if name == 'bare':
        print(f"  {name:<20} {mean_nll:>8.4f} {'--':>10} {'--':>8} {'--':>8} "
              f"{'--':>12} {'--':>5} {'--':>10}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls  # positive = condition has lower NLL (better)
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

        print(f"  {name:<20} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {rec_str:>10}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            'recovery': float(recovery) if not np.isnan(recovery) else None,
        }
""")


# ===== Cell 8: Key comparisons =====
code(r"""# Cell 8: Key comparisons and hardness gradient
print("=" * 70)
print("KEY COMPARISONS")
print("=" * 70)

# 1. Does conditioning help at all?
d_oracle = cohens_d(bare - oracle)
_, p_oracle = stats.ttest_1samp(bare - oracle, 0)
sig_oracle = '***' if p_oracle < 0.001 else '**' if p_oracle < 0.01 else '*' if p_oracle < 0.05 else 'ns'
print(f"\n1. Oracle conditioning (upper bound):")
print(f"   d={d_oracle:+.4f} ({sig_oracle}), mean delta={bare.mean() - oracle.mean():+.4f}")

# 2. Adversarial vs bare (semantic sensitivity test)
d_adv = cohens_d(bare - adversarial)
_, p_adv = stats.ttest_1samp(bare - adversarial, 0)
sig_adv = '***' if p_adv < 0.001 else '**' if p_adv < 0.01 else '*' if p_adv < 0.05 else 'ns'
print(f"\n2. Adversarial (negative control):")
print(f"   d={d_adv:+.4f} ({sig_adv})")
if d_adv < -0.05:
    print(f"   -> Off-topic prefix HURTS: conditioning is semantically sensitive")
elif d_adv > 0.05:
    print(f"   -> Off-topic prefix helps?! Suggests structural (not semantic) effect")
else:
    print(f"   -> Off-topic prefix neutral: conditioning effect is structural")

# 3. Best surrogate
surr_results = {k: v for k, v in analysis.items()
                if k.startswith('surr_') or k == 'adversarial'}
best_surr = max(surr_results.items(), key=lambda x: x[1].get('d', -999))
print(f"\n3. Best surrogate: {best_surr[0]} (d={best_surr[1]['d']:+.4f})")

# 4. Surrogate type comparison
print(f"\n4. Surrogate type ranking:")
sorted_surrs = sorted(surr_results.items(), key=lambda x: x[1].get('d', -999), reverse=True)
for name, info in sorted_surrs:
    sig = '***' if info['p'] < 0.001 else '**' if info['p'] < 0.01 else '*' if info['p'] < 0.05 else 'ns'
    rec = f"{info['recovery']:.0f}%" if info.get('recovery') is not None else "n/a"
    print(f"   {name:<20} d={info['d']:+.4f} ({sig}) recovery={rec}")

# 5. Hardness gradient
print(f"\n--- Hardness gradient (oracle conditioning by difficulty) ---")
quintile_bounds = np.percentile(bare, [20, 40, 60, 80])
quintiles = np.digitize(bare, quintile_bounds)

print(f"  {'Quintile':<12} {'N':>4} {'bare':>8} {'oracle':>8} {'delta':>8} {'d':>8}")
print(f"  {'-'*52}")
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 5:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    b = bare[mask].mean()
    o = oracle[mask].mean()
    delta = (bare[mask] - oracle[mask]).mean()
    d = cohens_d(bare[mask] - oracle[mask])
    print(f"  {qlabel:<12} {n_q:>4} {b:>8.4f} {o:>8.4f} {delta:>+8.4f} {d:>+8.3f}")

r_hard, p_hard = stats.spearmanr(bare, bare - oracle)
print(f"\n  Spearman (hardness vs oracle benefit): rho={r_hard:.3f} (p={p_hard:.2e})")
""")


# ===== Cell 9: Verdict + save =====
code(r"""# Cell 9: Verdict and save
print("=" * 70)
print("VERDICT — Exp 02: Decoder-Only Surrogate KV Caching")
print("=" * 70)

d_oracle = cohens_d(bare - oracle)
_, p_oracle = stats.ttest_1samp(bare - oracle, 0)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")

print(f"\n--- Key result ---")
print(f"  Oracle conditioning: d={d_oracle:+.4f} "
      f"({'***' if p_oracle < 0.001 else '**' if p_oracle < 0.01 else '*' if p_oracle < 0.05 else 'ns'})")

if d_oracle > 0.1:
    print(f"\n  CONDITIONING WORKS in decoder-only KV cache manipulation.")
    print(f"  Surrogate prompts that saw the document improve answer NLL.")
elif d_oracle > 0.05:
    print(f"\n  WEAK conditioning effect. Some benefit from KV cache manipulation")
    print(f"  but the effect is small.")
else:
    print(f"\n  NO significant conditioning effect detected.")
    print(f"  The causal mask position trick doesn't transfer sufficient")
    print(f"  information through the KV cache to help answer generation.")

# Compare surrogates vs v3 findings
print(f"\n--- Surrogate comparison ---")
for name in ['surr_universal', 'surr_extractor', 'surr_reasonant',
             'surr_analytic', 'surr_doc_kw', 'adversarial']:
    nlls = np.array([r[f'nll_{name}'] for r in results])
    d = cohens_d(bare - nlls)
    _, p = stats.ttest_1samp(bare - nlls, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<20} d={d:+.4f} ({sig})")

# Save
final_results = {
    'experiment': 'v4_exp02_decoder_kv_caching',
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
out_path = "experiments/decoder_only/01/01_decoder_kv_caching.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
