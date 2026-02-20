#!/usr/bin/env python3
"""Build Exp 05: Semantic Priming in Isolation.

All structural confounds (BOS removal, position offset, cache length) are
equalized across conditions by using length-matched token-level prefixes.
The ONLY variable is the semantic content of the prefix.

For each sample, Q = len(tokenizer(query, add_special_tokens=False).input_ids).
Every prefixed condition constructs exactly Q prefix token IDs.
Phase A input: [BOS] + prefix_ids(Q) + [newline] + doc_ids
Slice from cache: first Q+2 entries (BOS + prefix + newline)
Result: only doc KV entries remain, at identical positions across all conditions.

9 conditions, N=400, MS MARCO v1.1, Gemma 3 4B-PT.
"""

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/05", exist_ok=True)

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 05: Semantic Priming in Isolation

## Motivation

Exp 03-04 showed that the "structural effect" is actually BOS removal + position
offset + attention sink pruning. These structural factors are so large that
`prune_first_3` (d=+0.80) and `pos_4` (d=+0.78) beat the oracle (d=+0.64).

But **we cannot conclude that semantic priming is zero**, because in all prior
experiments the oracle differs from controls in BOTH structure AND content
simultaneously. To isolate semantics, we need an experiment where **all structural
confounds are equalized** and only prefix content varies.

## Design principle

For each sample, compute Q = number of query tokens (without BOS). ALL prefixed
conditions (4-9) use a prefix of **exactly Q token IDs**. This ensures:
- Same BOS removal (always sliced)
- Same position offset (doc starts at position Q+2 in all conditions)
- Same number of Phase A attention targets (Q prefix tokens)
- Same cache length (only doc KV entries)

The **only** thing that varies across prefixed conditions is the semantic content
of the Q prefix tokens.

## Conditions (9)

### Reference baselines (different structure, for context)
| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | Standard: [BOS + doc], nothing sliced |
| 2 | best_structural | BOS + first 3 doc tokens removed from cache (exp04 champion) |
| 3 | no_prefix_posmatched | position_offset = per-sample Q+2, BOS removed, no prefix |

### Length-matched semantic gradient (all have exactly Q prefix tokens)
| # | Condition | Prefix content | Semantic level |
|---|-----------|---------------|----------------|
| 4 | repeat_token | token("the") × Q | Zero content variation |
| 5 | random_tokens | Q random IDs from query vocab | Varied embeddings, no meaning |
| 6 | unrelated_query | Different sample's query tokens (truncated/padded to Q) | Coherent text, wrong content |
| 7 | shuffled_query | Query tokens randomly permuted | Right vocabulary, wrong syntax |
| 8 | doc_keywords | First Q non-stopword doc tokens | Doc-relevant, not query-specific |
| 9 | oracle | Actual query tokens | Full semantic match |

## Planned contrasts

| Contrast | Isolates |
|----------|----------|
| no_prefix_posmatched vs repeat_token | Attention enrichment effect |
| repeat_token vs random_tokens | Embedding variation |
| random_tokens vs unrelated_query | Natural language structure |
| unrelated_query vs shuffled_query | Vocabulary match (right words, wrong order) |
| shuffled_query vs oracle | Word order / syntax |
| doc_keywords vs oracle | Query vs document relevance |
| best_structural vs oracle | Pure structure vs semantic priming |""")


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

RESULTS_DIR = Path("../../../results/decoder_only/exp05")
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

print(f"Exp 05: Semantic Priming in Isolation")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = getattr(text_cfg, 'vocab_size', 262208)
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
""")


# ===== Cell 3: Scoring functions =====
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


NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
BOS_ID = tokenizer.bos_token_id
print(f"BOS token ID: {BOS_ID}")
print(f"Newline token IDs: {NEWLINE_IDS} ({len(NEWLINE_IDS)} tokens)")


def score(doc_text, query_text, answer_text,
          prefix_token_ids=None,
          position_offset=0, remove_bos=False,
          prune_first=0):
    # Score NLL of answer tokens using two-phase KV cache.
    #
    # Modes:
    #   prefix_token_ids: [BOS] + prefix_ids + [\n] + doc_ids
    #     Slices first 1+len(prefix_ids)+len(NEWLINE_IDS) entries (BOS+prefix+\n)
    #   position_offset > 0: BOS at pos 0, doc at offset..offset+D, BOS removed
    #   remove_bos + prune_first: bare with BOS + first N doc tokens removed
    #   Default: bare (BOS in cache)

    # --- Phase A: Conditioning ---
    if prefix_token_ids is not None:
        doc_ids = tokenizer(doc_text, add_special_tokens=False,
                            truncation=True, max_length=1536).input_ids
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        slice_start = 1 + len(prefix_token_ids) + len(NEWLINE_IDS)
        custom_pos = None
        phase_b_start = len(cond_ids)

    elif position_offset > 0:
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        n_doc = len(cond_ids) - 1
        pos_list = [0] + list(range(position_offset, position_offset + n_doc))
        custom_pos = torch.tensor([pos_list], dtype=torch.long, device=DEVICE)
        slice_start = 1  # remove BOS
        phase_b_start = position_offset + n_doc

    else:
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        slice_start = 1 if remove_bos else 0
        custom_pos = None
        phase_b_start = len(cond_ids)

    cond_tensor = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)

    fwd_kwargs = {'input_ids': cond_tensor, 'use_cache': True}
    if custom_pos is not None:
        fwd_kwargs['position_ids'] = custom_pos

    with torch.no_grad():
        phase_a = model(**fwd_kwargs)

    cache = phase_a.past_key_values
    del phase_a

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

    if prune_first > 0:
        cache = slice_kv_cache(cache, prune_first)

    # --- Phase B: Inference ---
    query_part_ids = tokenizer("\n" + query_text + "\n",
                               add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids

    if not answer_ids:
        del cache
        return 0.0

    phase_b_ids = query_part_ids + answer_ids
    phase_b_tensor = torch.tensor([phase_b_ids], dtype=torch.long, device=DEVICE)

    pos_ids = torch.arange(phase_b_start, phase_b_start + len(phase_b_ids),
                           device=DEVICE).unsqueeze(0)
    cache_position = torch.arange(phase_b_start, phase_b_start + len(phase_b_ids),
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


print("Scoring function defined with token-level prefix support.")
""")


# ===== Cell 4: Load data + build prefix pools =====
code(r"""# Cell 4: Load MS MARCO data and build per-sample prefix token IDs
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

# --- Build per-sample token-level prefix data ---

# Collect all query token IDs (for random_tokens pool)
all_query_token_ids = []
for s in samples:
    ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    all_query_token_ids.extend(ids)
query_vocab_pool = list(set(all_query_token_ids))
print(f"Query vocabulary pool: {len(query_vocab_pool)} unique token IDs")

# Stopword token IDs for doc_keywords filtering
STOPWORDS = set("the a an is are was were be been being have has had do does did "
                "will would shall should may might can could of in to for on with "
                "at by from as into through during before after above below between "
                "and or but not no nor so yet both either neither each every all any "
                "few more most other some such than too very it its this that these "
                "those i me my we our you your he him his she her they them their "
                "what which who whom whose when where how why if then else".split())

# The token ID for "the" (for repeat_token condition)
THE_TOKEN_ID = tokenizer("the", add_special_tokens=False).input_ids[0]
print(f"Token ID for 'the': {THE_TOKEN_ID}")

pyrandom.seed(SEED + 200)

for i, s in enumerate(samples):
    # Tokenize query
    q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    Q = len(q_ids)
    s['query_token_ids'] = q_ids
    s['Q'] = Q

    # 1. repeat_token: "the" repeated Q times
    s['prefix_repeat'] = [THE_TOKEN_ID] * Q

    # 2. random_tokens: Q random IDs from query vocabulary pool
    s['prefix_random'] = [pyrandom.choice(query_vocab_pool) for _ in range(Q)]

    # 3. unrelated_query: next sample's query tokens, truncated/padded to Q
    other_idx = (i + 1) % len(samples)
    other_q_ids = tokenizer(samples[other_idx]['query'],
                            add_special_tokens=False).input_ids
    if len(other_q_ids) >= Q:
        s['prefix_unrelated'] = other_q_ids[:Q]
    else:
        # Pad by repeating the other query's tokens
        padded = other_q_ids * ((Q // len(other_q_ids)) + 1)
        s['prefix_unrelated'] = padded[:Q]

    # 4. shuffled_query: query tokens permuted
    shuffled = list(q_ids)
    pyrandom.shuffle(shuffled)
    s['prefix_shuffled'] = shuffled

    # 5. doc_keywords: first Q non-stopword tokens from document
    doc_tokens = tokenizer(s['passage'], add_special_tokens=False).input_ids
    doc_words = s['passage'].split()
    keyword_ids = []
    for word in doc_words:
        if word.lower().strip(".,;:!?()[]{}\"'") not in STOPWORDS:
            w_ids = tokenizer(word, add_special_tokens=False).input_ids
            keyword_ids.extend(w_ids)
            if len(keyword_ids) >= Q:
                break
    # Pad with first doc tokens if not enough keywords
    if len(keyword_ids) < Q:
        keyword_ids.extend(doc_tokens[:Q - len(keyword_ids)])
    s['prefix_doc_kw'] = keyword_ids[:Q]

    # 6. oracle: actual query tokens (already have them)
    s['prefix_oracle'] = q_ids

# Summary statistics
q_lens = [s['Q'] for s in samples]
print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([len(s['query'].split()) for s in samples]):.1f}")
print(f"Query token count — mean: {np.mean(q_lens):.1f}, "
      f"median: {np.median(q_lens):.0f}, "
      f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

# Verify all prefixes have exactly Q tokens
for i, s in enumerate(samples[:5]):
    Q = s['Q']
    for name in ['prefix_repeat', 'prefix_random', 'prefix_unrelated',
                  'prefix_shuffled', 'prefix_doc_kw', 'prefix_oracle']:
        assert len(s[name]) == Q, f"Sample {i} {name}: len={len(s[name])} != Q={Q}"
    print(f"  Sample {i}: Q={Q}, query='{s['query'][:50]}...'")
    print(f"    repeat:    {tokenizer.decode(s['prefix_repeat'][:8])}...")
    print(f"    random:    {tokenizer.decode(s['prefix_random'][:8])}...")
    print(f"    unrelated: {tokenizer.decode(s['prefix_unrelated'][:8])}...")
    print(f"    shuffled:  {tokenizer.decode(s['prefix_shuffled'][:8])}...")
    print(f"    doc_kw:    {tokenizer.decode(s['prefix_doc_kw'][:8])}...")
    print(f"    oracle:    {tokenizer.decode(s['prefix_oracle'][:8])}...")
print("All prefix lengths verified.")
""")


# ===== Cell 5: Validation =====
code(r"""# Cell 5: Validate scoring modes
print("=" * 70)
print("VALIDATION")
print("=" * 70)

s = samples[0]
Q = s['Q']

print(f"\nSample 0: Q={Q} query tokens")
print(f"  Query: '{s['query']}'")
print(f"  Position of doc start in all prefixed conditions: {Q + 2}")
print(f"  (BOS=1 + prefix={Q} + newline={len(NEWLINE_IDS)})")

# Verify position matching: all prefixed conditions should put doc at same position
print(f"\n--- Position verification ---")
doc_ids = tokenizer(s['passage'], add_special_tokens=False,
                    truncation=True, max_length=1536).input_ids
for name in ['prefix_repeat', 'prefix_random', 'prefix_unrelated',
             'prefix_shuffled', 'prefix_doc_kw', 'prefix_oracle']:
    prefix_ids = s[name]
    cond_ids = [BOS_ID] + prefix_ids + NEWLINE_IDS + doc_ids
    slice_start = 1 + len(prefix_ids) + len(NEWLINE_IDS)
    doc_start_pos = slice_start  # doc[0] is at this position in the sequence
    print(f"  {name:<20} prefix_len={len(prefix_ids):>3}, "
          f"slice_start={slice_start:>3}, doc_start_pos={doc_start_pos:>3}, "
          f"total_len={len(cond_ids):>4}")

# Verify no_prefix_posmatched uses same offset
pos_offset = Q + 1 + len(NEWLINE_IDS)
print(f"\n  no_prefix_posmatched: position_offset={pos_offset} "
      f"(matching prefixed doc start)")

# Score all modes
print(f"\n--- NLL for each condition (sample 0) ---")
nll_bare = score(s['passage'], s['query'], s['answer'])
print(f"  {'bare':<24} NLL = {nll_bare:.4f}")

nll_struct = score(s['passage'], s['query'], s['answer'],
                   remove_bos=True, prune_first=3)
print(f"  {'best_structural':<24} NLL = {nll_struct:.4f}  "
      f"delta = {nll_bare - nll_struct:+.4f}")

nll_posmatched = score(s['passage'], s['query'], s['answer'],
                       position_offset=pos_offset)
print(f"  {'no_prefix_posmatched':<24} NLL = {nll_posmatched:.4f}  "
      f"delta = {nll_bare - nll_posmatched:+.4f}")

for name, prefix_key in [('repeat_token', 'prefix_repeat'),
                          ('random_tokens', 'prefix_random'),
                          ('unrelated_query', 'prefix_unrelated'),
                          ('shuffled_query', 'prefix_shuffled'),
                          ('doc_keywords', 'prefix_doc_kw'),
                          ('oracle', 'prefix_oracle')]:
    nll = score(s['passage'], s['query'], s['answer'],
                prefix_token_ids=s[prefix_key])
    print(f"  {name:<24} NLL = {nll:.4f}  delta = {nll_bare - nll:+.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 9 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'best_structural', 'no_prefix_posmatched',
    'repeat_token', 'random_tokens', 'unrelated_query',
    'shuffled_query', 'doc_keywords', 'oracle',
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
    Q = s['Q']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
        'query_words': len(query.split()),
        'Q': Q,
    }

    # 1. bare
    result['nll_bare'] = score(passage, query, answer)

    # 2. best_structural (BOS + first 3 doc tokens removed)
    result['nll_best_structural'] = score(passage, query, answer,
                                          remove_bos=True, prune_first=3)

    # 3. no_prefix_posmatched (position offset = Q + 1 + len(NEWLINE_IDS))
    pos_offset = Q + 1 + len(NEWLINE_IDS)
    result['nll_no_prefix_posmatched'] = score(passage, query, answer,
                                                position_offset=pos_offset)

    # 4-9. Length-matched prefixed conditions
    for cond_name, prefix_key in [
        ('repeat_token', 'prefix_repeat'),
        ('random_tokens', 'prefix_random'),
        ('unrelated_query', 'prefix_unrelated'),
        ('shuffled_query', 'prefix_shuffled'),
        ('doc_keywords', 'prefix_doc_kw'),
        ('oracle', 'prefix_oracle'),
    ]:
        result[f'nll_{cond_name}'] = score(passage, query, answer,
                                            prefix_token_ids=s[prefix_key])

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

arrays = {}
for name in COND_NAMES:
    arrays[name] = np.array([r[f'nll_{name}'] for r in results])

bare = arrays['bare']
oracle = arrays['oracle']
oracle_delta_mean = (bare - oracle).mean()

print(f"\n  Oracle delta (bare - oracle): {oracle_delta_mean:+.4f}")

print(f"\n  {'Condition':<24} {'NLL':>8} {'vs bare':>10} {'d':>8} "
      f"{'Win%':>8} {'p':>12} {'sig':>5} {'Recovery':>10}")
print(f"  {'-'*96}")

analysis = {}
for name in COND_NAMES:
    nlls = arrays[name]
    mean_nll = nlls.mean()

    if name == 'bare':
        print(f"  {name:<24} {mean_nll:>8.4f} {'--':>10} {'--':>8} "
              f"{'--':>8} {'--':>12} {'--':>5} {'--':>10}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        rec = diff.mean() / oracle_delta_mean * 100 if oracle_delta_mean > 0 else 0

        print(f"  {name:<24} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {rec:>9.1f}%")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            'recovery': float(rec),
        }
""")


# ===== Cell 8: Semantic gradient analysis =====
code(r"""# Cell 8: Semantic gradient — pairwise contrasts
print("=" * 70)
print("SEMANTIC GRADIENT ANALYSIS")
print("=" * 70)

# --- Attention enrichment: does having prefix tokens matter? ---
print(f"\n--- Attention enrichment effect ---")
print(f"  no_prefix_posmatched vs repeat_token")
print(f"  (same position offset, no prefix tokens vs Q uniform prefix tokens)")
diff_enrich = arrays['no_prefix_posmatched'] - arrays['repeat_token']
d_enrich = cohens_d(diff_enrich)
_, p_enrich = stats.ttest_1samp(diff_enrich, 0)
sig_e = '***' if p_enrich < 0.001 else '**' if p_enrich < 0.01 else '*' if p_enrich < 0.05 else 'ns'
print(f"  d = {d_enrich:+.4f} ({sig_e}), p = {p_enrich:.2e}")
if d_enrich > 0.05:
    print(f"  -> Having prefix tokens to attend to HELPS (repeat_token < no_prefix)")
elif d_enrich < -0.05:
    print(f"  -> Having prefix tokens to attend to HURTS (repeat_token > no_prefix)")
else:
    print(f"  -> Attention enrichment is negligible")

# --- Semantic gradient: pairwise steps ---
print(f"\n--- Semantic gradient (pairwise contrasts) ---")
print(f"  Each step adds one dimension of semantic content.")
print(f"  Positive d = later condition is better.\n")

gradient_pairs = [
    ('repeat_token', 'random_tokens', 'Embedding variation'),
    ('random_tokens', 'unrelated_query', 'Natural language structure'),
    ('unrelated_query', 'shuffled_query', 'Vocabulary match (right words)'),
    ('shuffled_query', 'oracle', 'Word order / syntax'),
]

print(f"  {'Step':<40} {'d':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*70}")
for cond_a, cond_b, label in gradient_pairs:
    diff = arrays[cond_a] - arrays[cond_b]  # positive = cond_b better
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<40} {d:>+8.4f} {p:>12.2e} {sig:>5}")

# --- Doc keywords vs oracle ---
print(f"\n--- Doc relevance vs query relevance ---")
diff_dq = arrays['doc_keywords'] - arrays['oracle']
d_dq = cohens_d(diff_dq)
_, p_dq = stats.ttest_1samp(diff_dq, 0)
sig_dq = '***' if p_dq < 0.001 else '**' if p_dq < 0.01 else '*' if p_dq < 0.05 else 'ns'
print(f"  doc_keywords vs oracle: d = {d_dq:+.4f} ({sig_dq})")
if d_dq > 0.05:
    print(f"  -> Oracle is better: query-specific semantics matter")
elif d_dq < -0.05:
    print(f"  -> Doc keywords are better: doc-derived priming is stronger")
else:
    print(f"  -> No significant difference")

# --- Structure vs semantics ---
print(f"\n--- Structure vs semantics ---")
diff_ss = arrays['best_structural'] - arrays['oracle']
d_ss = cohens_d(diff_ss)
_, p_ss = stats.ttest_1samp(diff_ss, 0)
sig_ss = '***' if p_ss < 0.001 else '**' if p_ss < 0.01 else '*' if p_ss < 0.05 else 'ns'
print(f"  best_structural (prune_first_3) vs oracle: d = {d_ss:+.4f} ({sig_ss})")
if d_ss > 0.05:
    print(f"  -> Oracle beats best structural: semantic priming adds value")
elif d_ss < -0.05:
    print(f"  -> Best structural beats oracle: structure > semantics")
else:
    print(f"  -> No significant difference")

# --- Is there ANY semantic gradient at all? ---
print(f"\n--- Overall semantic test ---")
print(f"  ANOVA-like: do the 6 prefixed conditions differ significantly?")
prefixed_names = ['repeat_token', 'random_tokens', 'unrelated_query',
                  'shuffled_query', 'doc_keywords', 'oracle']
prefixed_arrays = [arrays[n] for n in prefixed_names]
F_stat, p_anova = stats.f_oneway(*prefixed_arrays)
sig_anova = '***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else 'ns'
print(f"  F = {F_stat:.2f}, p = {p_anova:.2e} ({sig_anova})")
if p_anova < 0.05:
    print(f"  -> YES: prefix content significantly affects NLL")
else:
    print(f"  -> NO: prefix content does not significantly affect NLL")
    print(f"  -> The structural effect is EVERYTHING; semantics are zero")

# --- Strongest single semantic contrast: repeat_token vs oracle ---
print(f"\n--- Strongest semantic contrast: repeat_token vs oracle ---")
diff_max = arrays['repeat_token'] - arrays['oracle']
d_max = cohens_d(diff_max)
_, p_max = stats.ttest_1samp(diff_max, 0)
sig_max = '***' if p_max < 0.001 else '**' if p_max < 0.01 else '*' if p_max < 0.05 else 'ns'
print(f"  d = {d_max:+.4f} ({sig_max}), p = {p_max:.2e}")
print(f"  repeat_token NLL = {arrays['repeat_token'].mean():.4f}")
print(f"  oracle NLL       = {arrays['oracle'].mean():.4f}")
""")


# ===== Cell 9: Verdict + save =====
code(r"""# Cell 9: Verdict
print("=" * 70)
print("VERDICT — Exp 05: Semantic Priming in Isolation")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")
print(f"Mean query tokens: {np.mean([r['Q'] for r in results]):.1f}")

# All conditions ranked
print(f"\n--- All conditions (ranked by d vs bare) ---")
all_ranked = sorted(analysis.items(),
                    key=lambda x: x[1].get('d', -999), reverse=True)
for name, info in all_ranked:
    if name == 'bare':
        print(f"  {name:<24} NLL = {info['mean_nll']:.4f}  (baseline)")
    else:
        print(f"  {name:<24} NLL = {info['mean_nll']:.4f}  "
              f"d = {info['d']:+.4f}  ({info['recovery']:.0f}% recovery)")

# Classification of result
print(f"\n--- Classification ---")
d_repeat = analysis.get('repeat_token', {}).get('d', 0)
d_oracle = analysis.get('oracle', {}).get('d', 0)
d_random = analysis.get('random_tokens', {}).get('d', 0)
d_shuffled = analysis.get('shuffled_query', {}).get('d', 0)

# Test repeat vs oracle
diff_ro = arrays['repeat_token'] - arrays['oracle']
_, p_ro = stats.ttest_1samp(diff_ro, 0)

if p_ro < 0.01 and d_oracle > d_repeat + 0.05:
    print(f"  SEMANTIC PRIMING IS REAL")
    print(f"  Oracle significantly outperforms matched-structure repeat_token")
    print(f"  Effect size of pure semantics: d = {cohens_d(diff_ro):+.4f}")

    # Where does the semantic benefit come from?
    diff_rv = arrays['repeat_token'] - arrays['random_tokens']
    diff_ru = arrays['random_tokens'] - arrays['unrelated_query']
    diff_us = arrays['unrelated_query'] - arrays['shuffled_query']
    diff_so = arrays['shuffled_query'] - arrays['oracle']

    total_sem = diff_ro.mean()
    print(f"\n  Semantic decomposition (total = {total_sem:+.4f}):")
    for label, diff_step in [
        ("Embedding variation (repeat→random)", diff_rv),
        ("Natural language (random→unrelated)", diff_ru),
        ("Vocabulary match (unrelated→shuffled)", diff_us),
        ("Word order (shuffled→oracle)", diff_so),
    ]:
        step_mean = diff_step.mean()
        _, step_p = stats.ttest_1samp(diff_step, 0)
        sig = '***' if step_p < 0.001 else '**' if step_p < 0.01 else '*' if step_p < 0.05 else 'ns'
        pct = step_mean / total_sem * 100 if total_sem != 0 else 0
        print(f"    {label}: {step_mean:+.4f} ({pct:>5.1f}%) ({sig})")
else:
    print(f"  SEMANTIC PRIMING IS NEGLIGIBLE")
    print(f"  Oracle does NOT significantly outperform repeat_token")
    print(f"  The entire benefit of prefix co-encoding is structural")
    print(f"  (BOS removal + position offset + attention enrichment)")

# Save
final_results = {
    'experiment': 'v4_decoder_only_exp05_semantic_isolation',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {k: v for k, v in analysis.items()},
    'query_token_stats': {
        'mean': float(np.mean([r['Q'] for r in results])),
        'median': float(np.median([r['Q'] for r in results])),
        'min': int(np.min([r['Q'] for r in results])),
        'max': int(np.max([r['Q'] for r in results])),
    },
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
out_path = "experiments/decoder_only/05/05_semantic_isolation.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
