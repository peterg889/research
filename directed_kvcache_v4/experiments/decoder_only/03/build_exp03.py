#!/usr/bin/env python3
"""Build Exp 03: Position vs Attention Isolation.

Exp 02 found structural dominance (82-88%) on Gemma 3 4B. This experiment
isolates three candidate mechanisms:
  1. BOS removal from Phase B cache (confound in all exp02 prefixed conditions)
  2. RoPE position offset (doc tokens at higher absolute positions)
  3. Attention enrichment (doc attending to prefix tokens during Phase A)

10 conditions, N=400, MS MARCO v1.1, Gemma 3 4B-PT.
"""

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/03", exist_ok=True)

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 03: Position vs Attention Isolation

## Motivation

Exp 02 found that the **structural effect** (any prefix helps, regardless of content)
accounts for 82-88% of the benefit on Gemma 3 4B. But what drives this structural
component? Three candidate mechanisms:

1. **BOS removal**: In all prefixed conditions, BOS is sliced from the Phase B cache
   (because BOS is part of the prefix). In bare, BOS stays. If BOS acts as an attention
   sink that wastes capacity, removing it from the cache could improve query attention.

2. **RoPE position offset**: With a prefix of S tokens, doc tokens sit at RoPE positions
   S..S+D instead of 1..D. Even though relative doc-query distances are preserved,
   the absolute positions differ. The model may have learned position-dependent behaviors.
   Also, BOS-to-doc distance increases from 1 to S, weakening the BOS attention sink.

3. **Attention enrichment**: During Phase A, doc tokens attend to prefix tokens in
   addition to BOS and prior doc tokens. This changes hidden states and thus KV values.
   Even random words provide alternative attention targets beyond BOS, yielding a richer
   weighted sum.

## Conditions (10 total)

### Diagnostic controls
| # | Condition | Prefix tokens? | BOS in cache? | Doc RoPE positions | Tests |
|---|-----------|---------------|---------------|-------------------|-------|
| 1 | bare | no | YES | 1..D | Baseline |
| 2 | oracle | query | no | S..S+D | Upper bound |
| 3 | bare_no_bos | no | **NO** | 1..D | Factor 1: BOS removal |
| 4 | pos_offset_4 | no | no | 4..4+D | Factor 2: small offset |
| 5 | pos_offset_20 | no | no | 20..20+D | Factor 2: large offset |

### Saturation curve (real prefix tokens)
| # | Condition | Prefix | BOS in cache? | Doc positions (approx) |
|---|-----------|--------|---------------|----------------------|
| 6 | newline_only | just `\n` | no | 2..2+D |
| 7 | single_word | 1 random word | no | ~4..4+D |
| 8 | random_3w | 3 random words | no | ~6..6+D |
| 9 | random_5w | 5 random words | no | ~8..8+D |
| 10 | random_15w | 15 random words | no | ~20..20+D |

## Key diagnostic comparisons

1. **BOS removal**: bare → bare_no_bos
2. **Position offset**: bare_no_bos → pos_offset_20
3. **Attention enrichment**: pos_offset_20 → random_15w
4. **Saturation curve**: newline → 1w → 3w → 5w → 15w
5. **Offset dose-response**: pos_offset_4 vs pos_offset_20

These three factors should sum to the total structural effect:
`(bare − random_15w) = (bare − bare_no_bos) + (bare_no_bos − pos_offset_20) + (pos_offset_20 − random_15w)`""")


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

RESULTS_DIR = Path("../../../results/decoder_only/exp03")
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

print(f"Exp 03: Position vs Attention Isolation")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
print(f"Vocab size: {getattr(text_cfg, 'vocab_size', 'N/A')}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
""")


# ===== Cell 3: Scoring functions =====
code(r"""# Cell 3: KV cache helpers and unified scoring function

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


def score(doc_text, query_text, answer_text, prefix_text=None,
          position_offset=0, remove_bos=False):
    # Score NLL of answer tokens using two-phase KV cache approach.
    #
    # Three modes:
    #   1. prefix_text given: [BOS + prefix + \n + doc], slice prefix+BOS from cache.
    #   2. position_offset > 0: doc at offset RoPE positions, no prefix tokens.
    #      BOS at position 0, doc at offset..offset+D. BOS always removed.
    #   3. remove_bos=True: like bare but BOS sliced from cache.
    # Default (all False/0): bare with BOS in cache.

    # --- Phase A: Conditioning ---
    if prefix_text is not None:
        # Standard prefixed mode
        prefix_ids = tokenizer(prefix_text + "\n", add_special_tokens=True,
                               truncation=True, max_length=512).input_ids
        doc_ids = tokenizer(doc_text, add_special_tokens=False,
                            truncation=True, max_length=1536).input_ids
        cond_ids = prefix_ids + doc_ids
        slice_start = len(prefix_ids)
        custom_pos = None
        # Last token at position len(cond_ids)-1, Phase B starts after
        phase_b_start = len(cond_ids)

    elif position_offset > 0:
        # Position offset mode: no prefix, doc at offset positions
        cond_ids = tokenizer(doc_text, add_special_tokens=True,
                             truncation=True, max_length=2048).input_ids
        slice_start = 1  # always remove BOS in this mode
        # BOS at position 0, doc tokens at offset..offset+D
        n_doc = len(cond_ids) - 1  # exclude BOS
        pos_list = [0] + list(range(position_offset, position_offset + n_doc))
        custom_pos = torch.tensor([pos_list], dtype=torch.long, device=DEVICE)
        # Last doc token at position offset + n_doc - 1
        phase_b_start = position_offset + n_doc

    else:
        # Bare mode (optionally remove BOS)
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

    # --- Phase B: Inference with query + answer ---
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


def score_full_sequence(doc_text, query_text, answer_text):
    # Single-pass scoring for validation (bare equivalent).
    doc_ids = tokenizer(doc_text, add_special_tokens=True,
                        truncation=True, max_length=2048).input_ids
    query_part_ids = tokenizer("\n" + query_text + "\n",
                               add_special_tokens=False).input_ids
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


print("Scoring functions defined.")
print(f"\nMode verification:")
print(f"  score(doc, q, a)                     → bare (BOS in cache)")
print(f"  score(doc, q, a, remove_bos=True)    → bare_no_bos")
print(f"  score(doc, q, a, position_offset=20) → pos_offset (BOS removed)")
print(f"  score(doc, q, a, prefix_text='...')   → prefixed (BOS+prefix sliced)")

# Show how newline_only tokenizes
nl_ids = tokenizer("\n", add_special_tokens=True).input_ids
print(f"\n  newline_only prefix_ids: {nl_ids} ({len(nl_ids)} tokens)")
print(f"  Token names: {[tokenizer.decode([t]) for t in nl_ids]}")
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

# Build word pool from passage text
all_words = []
for s in samples:
    all_words.extend(s['passage'].split())
pyrandom.seed(SEED + 99)
pyrandom.shuffle(all_words)
word_pool = all_words

# Generate per-sample prefixes at different lengths
for i, s in enumerate(samples):
    pool_offset = i * 50

    # single_word: 1 word from pool
    s['single_word'] = word_pool[pool_offset]

    # random_3w: 3 words
    s['random_3w'] = " ".join(word_pool[pool_offset + 1:pool_offset + 4])

    # random_5w: 5 words
    s['random_5w'] = " ".join(word_pool[pool_offset + 4:pool_offset + 9])

    # random_15w: 15 words
    s['random_15w'] = " ".join(word_pool[pool_offset + 9:pool_offset + 24])

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([len(s['query'].split()) for s in samples]):.1f}")

# Show actual token counts for each prefix type
print(f"\nPrefix token counts (sample 0):")
for name, text in [('newline_only', ''),
                    ('single_word', samples[0]['single_word']),
                    ('random_3w', samples[0]['random_3w']),
                    ('random_5w', samples[0]['random_5w']),
                    ('random_15w', samples[0]['random_15w'])]:
    prefix_ids = tokenizer(text + "\n", add_special_tokens=True).input_ids
    print(f"  {name:<16} {len(prefix_ids):>3} tok  S={len(prefix_ids):<4}  "
          f"text: {repr(text[:40])}")

print(f"\n  pos_offset_4:   S=4   (matches ~single_word)")
print(f"  pos_offset_20:  S=20  (matches ~random_15w)")
""")


# ===== Cell 5: Validation =====
code(r"""# Cell 5: Validate all scoring modes
print("=" * 70)
print("VALIDATION")
print("=" * 70)

# 1. Bare cached vs full-sequence
print("\n--- Bare: cached vs full-sequence ---")
max_diff = 0.0
for i in range(5):
    s = samples[i]
    nll_cached = score(s['passage'], s['query'], s['answer'])
    nll_full = score_full_sequence(s['passage'], s['query'], s['answer'])
    diff = abs(nll_cached - nll_full)
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 0.01 else "~"
    print(f"  Sample {i}: cached={nll_cached:.6f}, full={nll_full:.6f}, "
          f"diff={diff:.6f} [{status}]")
if max_diff < 0.1:
    print(f"  PASSED (max diff = {max_diff:.6f}, bf16 rounding)")
else:
    print(f"  WARNING: max diff = {max_diff:.6f}")

# 2. All modes run without error on sample 0
print(f"\n--- All modes on sample 0 ---")
s = samples[0]
nll_bare = score(s['passage'], s['query'], s['answer'])
nll_no_bos = score(s['passage'], s['query'], s['answer'], remove_bos=True)
nll_pos4 = score(s['passage'], s['query'], s['answer'], position_offset=4)
nll_pos20 = score(s['passage'], s['query'], s['answer'], position_offset=20)
nll_nl = score(s['passage'], s['query'], s['answer'], prefix_text="")
nll_1w = score(s['passage'], s['query'], s['answer'], prefix_text=s['single_word'])
nll_3w = score(s['passage'], s['query'], s['answer'], prefix_text=s['random_3w'])
nll_5w = score(s['passage'], s['query'], s['answer'], prefix_text=s['random_5w'])
nll_15w = score(s['passage'], s['query'], s['answer'], prefix_text=s['random_15w'])
nll_oracle = score(s['passage'], s['query'], s['answer'], prefix_text=s['query'])

print(f"  {'bare':<20} NLL = {nll_bare:.6f}")
print(f"  {'bare_no_bos':<20} NLL = {nll_no_bos:.6f}  delta = {nll_bare - nll_no_bos:+.4f}")
print(f"  {'pos_offset_4':<20} NLL = {nll_pos4:.6f}  delta = {nll_bare - nll_pos4:+.4f}")
print(f"  {'pos_offset_20':<20} NLL = {nll_pos20:.6f}  delta = {nll_bare - nll_pos20:+.4f}")
print(f"  {'newline_only':<20} NLL = {nll_nl:.6f}  delta = {nll_bare - nll_nl:+.4f}")
print(f"  {'single_word':<20} NLL = {nll_1w:.6f}  delta = {nll_bare - nll_1w:+.4f}")
print(f"  {'random_3w':<20} NLL = {nll_3w:.6f}  delta = {nll_bare - nll_3w:+.4f}")
print(f"  {'random_5w':<20} NLL = {nll_5w:.6f}  delta = {nll_bare - nll_5w:+.4f}")
print(f"  {'random_15w':<20} NLL = {nll_15w:.6f}  delta = {nll_bare - nll_15w:+.4f}")
print(f"  {'oracle':<20} NLL = {nll_oracle:.6f}  delta = {nll_bare - nll_oracle:+.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 10 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle', 'bare_no_bos',
    'pos_offset_4', 'pos_offset_20',
    'newline_only', 'single_word', 'random_3w', 'random_5w', 'random_15w',
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

    # 3. bare_no_bos
    result['nll_bare_no_bos'] = score(passage, query, answer, remove_bos=True)

    # 4. pos_offset_4 (S=4, matches ~single_word prefix length)
    result['nll_pos_offset_4'] = score(passage, query, answer, position_offset=4)

    # 5. pos_offset_20 (S=20, matches ~random_15w prefix length)
    result['nll_pos_offset_20'] = score(passage, query, answer, position_offset=20)

    # 6. newline_only (prefix_text="" → BOS + \n only)
    result['nll_newline_only'] = score(passage, query, answer, prefix_text="")

    # 7. single_word
    result['nll_single_word'] = score(passage, query, answer,
                                      prefix_text=s['single_word'])

    # 8. random_3w
    result['nll_random_3w'] = score(passage, query, answer,
                                    prefix_text=s['random_3w'])

    # 9. random_5w
    result['nll_random_5w'] = score(passage, query, answer,
                                    prefix_text=s['random_5w'])

    # 10. random_15w
    result['nll_random_15w'] = score(passage, query, answer,
                                     prefix_text=s['random_15w'])

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
for name in COND_NAMES:
    arrays[name] = np.array([r[f'nll_{name}'] for r in results])

bare = arrays['bare']
oracle = arrays['oracle']
oracle_delta_mean = (bare - oracle).mean()

print(f"\n  {'Condition':<20} {'NLL':>8} {'vs bare':>10} {'d':>8} "
      f"{'Win%':>8} {'p':>12} {'sig':>5} {'Recovery':>10}")
print(f"  {'-'*90}")

analysis = {}
for name in COND_NAMES:
    nlls = arrays[name]
    mean_nll = nlls.mean()

    if name == 'bare':
        print(f"  {name:<20} {mean_nll:>8.4f} {'--':>10} {'--':>8} "
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

        print(f"  {name:<20} {mean_nll:>8.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {rec_str:>10}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
            'recovery': float(recovery) if not np.isnan(recovery) else None,
        }
""")


# ===== Cell 8: Mechanism isolation analysis =====
code(r"""# Cell 8: Mechanism isolation — three-factor decomposition
print("=" * 70)
print("MECHANISM ISOLATION")
print("=" * 70)

# -----------------------------------------------------------------------
# Three-factor decomposition of the total structural effect
# Total: bare → random_15w (the effect of a 15-word random prefix)
# Factor 1: BOS removal           (bare → bare_no_bos)
# Factor 2: Position offset       (bare_no_bos → pos_offset_20)
# Factor 3: Attention enrichment  (pos_offset_20 → random_15w)
# -----------------------------------------------------------------------

total = bare - arrays['random_15w']
f1 = bare - arrays['bare_no_bos']
f2 = arrays['bare_no_bos'] - arrays['pos_offset_20']
f3 = arrays['pos_offset_20'] - arrays['random_15w']

print(f"\n{'='*60}")
print(f"  THREE-FACTOR DECOMPOSITION")
print(f"  Total structural effect: bare → random_15w")
print(f"{'='*60}")

total_mean = total.mean()
total_d = cohens_d(total)
_, total_p = stats.ttest_1samp(total, 0)
print(f"\n  TOTAL: NLL delta = {total_mean:+.4f}, d = {total_d:+.4f}, "
      f"p = {total_p:.2e}")

for label, factor, explanation in [
    ("Factor 1: BOS removal", f1, "bare → bare_no_bos"),
    ("Factor 2: Position offset", f2, "bare_no_bos → pos_offset_20"),
    ("Factor 3: Attention enrichment", f3, "pos_offset_20 → random_15w"),
]:
    f_mean = factor.mean()
    f_d = cohens_d(factor)
    _, f_p = stats.ttest_1samp(factor, 0)
    f_sig = '***' if f_p < 0.001 else '**' if f_p < 0.01 else '*' if f_p < 0.05 else 'ns'
    if total_mean > 0:
        pct = f_mean / total_mean * 100
    else:
        pct = 0
    print(f"\n  {label}  ({explanation})")
    print(f"    NLL delta = {f_mean:+.4f}  ({pct:>5.1f}% of total)")
    print(f"    d = {f_d:+.4f}, p = {f_p:.2e} ({f_sig})")

# Sum check
sum_factors = f1.mean() + f2.mean() + f3.mean()
print(f"\n  Sum check: {f1.mean():.4f} + {f2.mean():.4f} + {f3.mean():.4f} "
      f"= {sum_factors:.4f} vs total {total_mean:.4f}")

# -----------------------------------------------------------------------
# Position offset dose-response
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  POSITION OFFSET DOSE-RESPONSE")
print(f"  (all with BOS removed, no prefix tokens)")
print(f"{'='*60}")

print(f"\n  {'Condition':<20} {'S':>4} {'d vs bare':>10} {'d vs bare_no_bos':>18} {'p':>12}")
print(f"  {'-'*70}")
for name, S in [('bare_no_bos', 0), ('pos_offset_4', 4), ('pos_offset_20', 20)]:
    diff_vs_bare = bare - arrays[name]
    diff_vs_nobos = arrays['bare_no_bos'] - arrays[name]
    d_b = cohens_d(diff_vs_bare)
    d_nb = cohens_d(diff_vs_nobos)
    _, p_nb = stats.ttest_1samp(diff_vs_nobos, 0)
    sig = '***' if p_nb < 0.001 else '**' if p_nb < 0.01 else '*' if p_nb < 0.05 else 'ns'
    print(f"  {name:<20} {S:>4} {d_b:>+10.4f} {d_nb:>+18.4f} {p_nb:>12.2e} {sig}")

# -----------------------------------------------------------------------
# Saturation curve — where does the effect plateau?
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  SATURATION CURVE")
print(f"  (real prefix tokens: when does benefit plateau?)")
print(f"{'='*60}")

print(f"\n  {'Condition':<16} {'~words':>8} {'d vs bare':>10} {'recovery':>10} {'p':>12}")
print(f"  {'-'*62}")
for name, nw in [('newline_only', '0'),
                  ('single_word', '1'),
                  ('random_3w', '3'),
                  ('random_5w', '5'),
                  ('random_15w', '15')]:
    diff = bare - arrays[name]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    rec = diff.mean() / oracle_delta_mean * 100 if oracle_delta_mean > 0 else 0
    print(f"  {name:<16} {nw:>8} {d:>+10.4f} {rec:>9.1f}% {p:>12.2e} {sig}")

# Pairwise: does adding more words help significantly?
print(f"\n  Pairwise (does adding more words help beyond previous level?):")
pairs = [('newline_only', 'single_word'),
         ('single_word', 'random_3w'),
         ('random_3w', 'random_5w'),
         ('random_5w', 'random_15w')]
for a, b in pairs:
    diff_ab = arrays[a] - arrays[b]  # positive = b better
    d_ab = cohens_d(diff_ab)
    _, p_ab = stats.ttest_1samp(diff_ab, 0)
    sig = '***' if p_ab < 0.001 else '**' if p_ab < 0.01 else '*' if p_ab < 0.05 else 'ns'
    print(f"  {a:<16} → {b:<16} d = {d_ab:+.4f} ({sig})")

# -----------------------------------------------------------------------
# Matched comparison: pos_offset vs prefix at same approximate S
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  POSITION-MATCHED COMPARISONS")
print(f"  (same approximate offset, with vs without prefix tokens)")
print(f"{'='*60}")

for pos_name, prefix_name, approx_s in [
    ('pos_offset_4', 'single_word', '~4'),
    ('pos_offset_20', 'random_15w', '~20'),
]:
    diff = arrays[pos_name] - arrays[prefix_name]  # positive = prefix better
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\n  S≈{approx_s}: {pos_name} vs {prefix_name}")
    print(f"    pos_offset NLL = {arrays[pos_name].mean():.4f}")
    print(f"    prefix     NLL = {arrays[prefix_name].mean():.4f}")
    print(f"    diff d = {d:+.4f} ({sig})")
    if d > 0.05:
        print(f"    → Prefix tokens ADD value beyond position offset alone")
    elif d < -0.05:
        print(f"    → Prefix tokens HURT — position alone is better?!")
    else:
        print(f"    → No significant difference — position alone explains it")
""")


# ===== Cell 9: Verdict + save =====
code(r"""# Cell 9: Verdict and interpretation
print("=" * 70)
print("VERDICT — Exp 03: Position vs Attention Isolation")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} samples (MS MARCO v1.1)")

# Recall the three factors
total_mean = (bare - arrays['random_15w']).mean()
f1_mean = (bare - arrays['bare_no_bos']).mean()
f2_mean = (arrays['bare_no_bos'] - arrays['pos_offset_20']).mean()
f3_mean = (arrays['pos_offset_20'] - arrays['random_15w']).mean()

print(f"\n--- Structural effect decomposition ---")
print(f"  Total (bare → random_15w): {total_mean:+.4f}")
if total_mean > 0:
    print(f"  Factor 1 — BOS removal:     {f1_mean:+.4f} ({f1_mean/total_mean*100:>5.1f}%)")
    print(f"  Factor 2 — Position offset:  {f2_mean:+.4f} ({f2_mean/total_mean*100:>5.1f}%)")
    print(f"  Factor 3 — Attention enrich: {f3_mean:+.4f} ({f3_mean/total_mean*100:>5.1f}%)")

# Interpretation
print(f"\n--- Interpretation ---")
_, f1_p = stats.ttest_1samp(bare - arrays['bare_no_bos'], 0)
_, f2_p = stats.ttest_1samp(arrays['bare_no_bos'] - arrays['pos_offset_20'], 0)
_, f3_p = stats.ttest_1samp(arrays['pos_offset_20'] - arrays['random_15w'], 0)

for label, pct, p in [("BOS removal", f1_mean/total_mean*100 if total_mean > 0 else 0, f1_p),
                        ("Position offset", f2_mean/total_mean*100 if total_mean > 0 else 0, f2_p),
                        ("Attention enrichment", f3_mean/total_mean*100 if total_mean > 0 else 0, f3_p)]:
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {label:<24} {pct:>5.1f}%  ({sig}, p={p:.2e})")

# Saturation
d_1w = cohens_d(bare - arrays['single_word'])
d_15w = cohens_d(bare - arrays['random_15w'])
if d_15w > 0:
    print(f"\n  Saturation: 1 word achieves {d_1w/d_15w*100:.0f}% of 15-word effect")

# All conditions summary
print(f"\n--- All conditions ---")
for name in COND_NAMES:
    if name == 'bare':
        print(f"  {name:<20} NLL = {arrays[name].mean():.4f}")
    else:
        d = cohens_d(bare - arrays[name])
        _, p = stats.ttest_1samp(bare - arrays[name], 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {name:<20} NLL = {arrays[name].mean():.4f}  d = {d:+.4f} ({sig})")

# Save
final_results = {
    'experiment': 'v4_decoder_only_exp03_position_vs_attention',
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
out_path = "experiments/decoder_only/03/03_position_vs_attention.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
