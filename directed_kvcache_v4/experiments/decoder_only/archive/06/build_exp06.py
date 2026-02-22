#!/usr/bin/env python3
"""Build Exp 06: Graded Semantic Relevance — Decoder-Only.

Port of v3 Exp 12 to decoder-only two-phase KV cache scoring.
All structural confounds equalized via token-level prefix matching
(exactly Q prefix token IDs per condition).

Two-phase design:
  Phase 1: Gemma 3 12B-IT generates paraphrases + same-topic queries
  Phase 2: Gemma 3 4B-PT scores 7 conditions x 400 samples

7 conditions, N=400, SEED=42.
"""

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/06", exist_ok=True)

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Decoder-Only Exp 06: Graded Semantic Relevance

## Motivation

Port of v3 Exp 12 to decoder-only two-phase KV cache scoring.
v3 Exp 12 found a monotonic semantic gradient (Spearman rho=+0.94, p=0.005)
in the encoder-decoder T5Gemma. Does the same gradient appear with decoder-only
Gemma 3 4B-PT using KV cache priming?

## Design

All structural confounds (BOS removal, position offset, cache length) are
equalized by using token-level prefix matching: every prefixed condition
constructs exactly Q prefix token IDs (Q = number of real query tokens).

Phase A input: [BOS] + prefix_ids(Q) + [\n] + doc_ids
Slice from cache: first Q+2 entries (BOS + prefix + newline)
Result: only doc KV entries remain, at identical positions across all conditions.

## Conditions (7)

| # | Condition | Prefix content | Semantic relevance |
|---|-----------|---------------|--------------------|
| 1 | bare | (none) | N/A (lower bound) |
| 2 | oracle | real query tokens | maximal (exact query) |
| 3 | paraphrase | LLM paraphrase tokens | high (same meaning, diff words) |
| 4 | same_topic | LLM same-topic question | medium (right topic, wrong question) |
| 5 | unrelated_query | different sample's query | low (real syntax, wrong topic) |
| 6 | scrambled_oracle | query tokens shuffled | vocabulary only |
| 7 | random_matched | random passage word tokens | none (structural baseline) |

## Analysis

- Part 1: Standard condition table
- Part 2: Semantic gradient with monotonicity test
- Part 3: Fine-grained decomposition chain
- Part 4: Hardness interaction""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, re, gc, random as pyrandom
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

RESULTS_DIR = Path("../../../results/decoder_only/exp06")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_PATH = RESULTS_DIR / "surrogates.json"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

# Prompt templates for LLM generation
PROMPT_PARAPHRASE = (
    "Rephrase this search query using completely different words but keeping "
    "the same meaning. Keep it to 5-8 words. Output only the rephrased query."
)
PROMPT_SAME_TOPIC = (
    "Write a question about the same topic as this document but asking for "
    "DIFFERENT information. Keep it to 5-8 words. Output only the question."
)

print("Exp 06: Graded Semantic Relevance (Decoder-Only)")
print(f"N: {N_SAMPLES}")
print(f"Model: {MODEL_NAME} (generation + scoring)")
""")


# ===== Cell 3: Load MS MARCO =====
code(r"""# Cell 3: Load MS MARCO and select samples
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= N_SAMPLES * 3:
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
            samples.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

passage_words = np.array([s['word_count'] for s in samples])
query_words = np.array([len(s['query'].split()) for s in samples])
print(f"Selected {N_SAMPLES} samples")
print(f"Document lengths: {passage_words.min()}-{passage_words.max()} words, "
      f"mean={passage_words.mean():.0f}")
print(f"Query lengths: {query_words.min()}-{query_words.max()} words, "
      f"mean={query_words.mean():.1f}")

for i in range(5):
    s = samples[i]
    print(f"\nExample {i}:")
    print(f"  Q: {s['query']}")
    print(f"  A: {s['answer'][:80]}")
    print(f"  P ({s['word_count']}w): {s['passage'][:100]}...")
""")


# ===== Cell 4: Phase 1 — LLM surrogate generation =====
code(r"""# Cell 4: Phase 1 — Generate surrogates with Gemma 3 12B-IT
# Skip if surrogates already cached

if SURROGATES_PATH.exists():
    print("Loading cached surrogates...")
    surrogates = json.loads(SURROGATES_PATH.read_text())
    assert len(surrogates) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(surrogates)}"
    for i in range(min(10, N_SAMPLES)):
        assert surrogates[i]['query'][:50] == samples[i]['query'][:50], \
            f"Sample {i} query mismatch"
    print(f"Loaded {len(surrogates)} cached surrogates")
    print(f"Keys per sample: {list(surrogates[0].keys())}")
else:
    print(f"Loading {MODEL_NAME} for surrogate generation...")
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    HF_TOKEN = os.environ.get("HF_TOKEN")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
    )
    gen_model.eval()
    GEN_DEVICE = next(gen_model.parameters()).device
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def generate_text(input_text, prompt_text):
        # Generate text from a prompt + input using Gemma IT.
        messages = [
            {"role": "user",
             "content": f"{prompt_text}\n\n{input_text}"}
        ]
        chat_text = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = gen_tokenizer(chat_text, return_tensors="pt",
                               truncation=True, max_length=1024).to(GEN_DEVICE)

        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
        raw_text = gen_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Post-process: strip, take first line, remove quotes, truncate to 15 words
        cleaned = raw_text.strip().split("\n")[0].strip()
        cleaned = cleaned.strip('"').strip("'").strip()
        cleaned = " ".join(cleaned.split()[:15])
        return cleaned

    # Generate with checkpointing
    surrogates = []
    gen_ckpt_path = RESULTS_DIR / "gen_checkpoint.json"

    if gen_ckpt_path.exists():
        gen_ckpt = json.loads(gen_ckpt_path.read_text())
        if gen_ckpt.get('n_total') == N_SAMPLES:
            surrogates = gen_ckpt['surrogates']
            print(f"Resuming generation from {len(surrogates)}/{N_SAMPLES}")

    start_gen = len(surrogates)
    t0 = time.time()

    for i in tqdm(range(start_gen, N_SAMPLES), initial=start_gen, total=N_SAMPLES,
                  desc="Generating"):
        s = samples[i]
        entry = {'query': s['query']}

        # Paraphrase: rephrase the query
        torch.manual_seed(SEED + i * 10)
        entry['paraphrase'] = generate_text(
            f"Query: {s['query']}", PROMPT_PARAPHRASE
        )

        # Same-topic: question about same topic but different info
        torch.manual_seed(SEED + i * 10 + 1)
        words = s['passage'].split()[:150]
        entry['same_topic'] = generate_text(
            f"Document:\n{' '.join(words)}", PROMPT_SAME_TOPIC
        )

        surrogates.append(entry)

        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            gen_ckpt = {'n_total': N_SAMPLES, 'surrogates': surrogates,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            gen_ckpt_path.write_text(json.dumps(gen_ckpt))
            elapsed = time.time() - t0
            done = i - start_gen + 1
            eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
            tqdm.write(f"  Gen checkpoint {i+1}/{N_SAMPLES} | "
                       f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {len(surrogates)} samples in {elapsed/60:.1f} min")

    # Save final surrogates
    SURROGATES_PATH.write_text(json.dumps(surrogates, indent=2))
    print(f"Saved surrogates to {SURROGATES_PATH}")

    # Free VRAM
    print("Freeing generation model VRAM...")
    mem_before = torch.cuda.memory_allocated() / 1e9
    del gen_model, gen_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    mem_after = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
""")


# ===== Cell 5: Load scoring model + helpers =====
code(r"""# Cell 5: Load scoring model and define helpers
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

print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = getattr(text_cfg, 'vocab_size', 262208)
print(f"Vocab size: {VOCAB_SIZE}")

NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
BOS_ID = tokenizer.bos_token_id
print(f"BOS token ID: {BOS_ID}")
print(f"Newline token IDs: {NEWLINE_IDS} ({len(NEWLINE_IDS)} tokens)")


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


def score(doc_text, query_text, answer_text, prefix_token_ids=None):
    # Score NLL of answer tokens using two-phase KV cache.
    #
    # If prefix_token_ids is provided:
    #   Phase A: [BOS] + prefix_ids + [\n] + doc_ids
    #   Slice first 1+len(prefix_ids)+len(NEWLINE_IDS) entries
    # Otherwise (bare):
    #   Phase A: [BOS] + doc_ids (nothing sliced)

    # --- Phase A: Conditioning ---
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids

    if prefix_token_ids is not None:
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        slice_start = 1 + len(prefix_token_ids) + len(NEWLINE_IDS)
        phase_b_start = len(cond_ids)
    else:
        cond_ids = [BOS_ID] + doc_ids
        slice_start = 0
        phase_b_start = len(cond_ids)

    cond_tensor = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        phase_a = model(input_ids=cond_tensor, use_cache=True)

    cache = phase_a.past_key_values
    del phase_a

    if slice_start > 0:
        cache = slice_kv_cache(cache, slice_start)

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


print("Scoring function defined.")
""")


# ===== Cell 6: Build per-sample token-level prefixes =====
code(r"""# Cell 6: Build per-sample token-level prefix IDs

# Collect all query token IDs (for random pool)
all_query_token_ids = []
for s in samples:
    ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    all_query_token_ids.extend(ids)
query_vocab_pool = list(set(all_query_token_ids))
print(f"Query vocabulary pool: {len(query_vocab_pool)} unique token IDs")

pyrandom.seed(SEED + 200)

for i, s in enumerate(samples):
    surr = surrogates[i]
    q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    Q = len(q_ids)
    s['Q'] = Q

    # 1. oracle: actual query tokens
    s['prefix_oracle'] = q_ids

    # 2. paraphrase: LLM paraphrase, tokenized and truncated/padded to Q
    para_ids = tokenizer(surr['paraphrase'], add_special_tokens=False).input_ids
    if len(para_ids) >= Q:
        s['prefix_paraphrase'] = para_ids[:Q]
    else:
        padded = para_ids * ((Q // max(len(para_ids), 1)) + 1)
        s['prefix_paraphrase'] = padded[:Q]

    # 3. same_topic: LLM same-topic question, tokenized and truncated/padded to Q
    topic_ids = tokenizer(surr['same_topic'], add_special_tokens=False).input_ids
    if len(topic_ids) >= Q:
        s['prefix_same_topic'] = topic_ids[:Q]
    else:
        padded = topic_ids * ((Q // max(len(topic_ids), 1)) + 1)
        s['prefix_same_topic'] = padded[:Q]

    # 4. unrelated_query: different sample's query tokens, truncated/padded to Q
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_q_ids = tokenizer(samples[other_idx]['query'],
                            add_special_tokens=False).input_ids
    if len(other_q_ids) >= Q:
        s['prefix_unrelated'] = other_q_ids[:Q]
    else:
        padded = other_q_ids * ((Q // max(len(other_q_ids), 1)) + 1)
        s['prefix_unrelated'] = padded[:Q]

    # 5. scrambled_oracle: query tokens randomly permuted
    shuffled = list(q_ids)
    pyrandom.shuffle(shuffled)
    s['prefix_scrambled'] = shuffled

    # 6. random_matched: words from unrelated passage, tokenized and truncated/padded to Q
    other_words = samples[other_idx]['passage'].split()
    random_text = " ".join(other_words[:len(s['query'].split())])
    rand_ids = tokenizer(random_text, add_special_tokens=False).input_ids
    if len(rand_ids) >= Q:
        s['prefix_random'] = rand_ids[:Q]
    else:
        padded = rand_ids * ((Q // max(len(rand_ids), 1)) + 1)
        s['prefix_random'] = padded[:Q]

# Summary statistics
q_lens = [s['Q'] for s in samples]
print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Query token count — mean: {np.mean(q_lens):.1f}, "
      f"median: {np.median(q_lens):.0f}, "
      f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

# Verify all prefixes have exactly Q tokens
prefix_names = ['prefix_oracle', 'prefix_paraphrase', 'prefix_same_topic',
                'prefix_unrelated', 'prefix_scrambled', 'prefix_random']
for i, s in enumerate(samples[:5]):
    Q = s['Q']
    for name in prefix_names:
        assert len(s[name]) == Q, f"Sample {i} {name}: len={len(s[name])} != Q={Q}"
    print(f"  Sample {i}: Q={Q}, query='{s['query'][:50]}...'")
    for name in prefix_names:
        label = name.replace('prefix_', '')
        print(f"    {label:<15}: {tokenizer.decode(s[name][:8])}...")
print("All prefix lengths verified.")
""")


# ===== Cell 7: Validation =====
code(r"""# Cell 7: Validate scoring
print("=" * 70)
print("VALIDATION")
print("=" * 70)

s = samples[0]
Q = s['Q']

print(f"\nSample 0: Q={Q} query tokens")
print(f"  Query: '{s['query']}'")
print(f"  Doc position start (all prefixed): {Q + 2}")
print(f"  (BOS=1 + prefix={Q} + newline={len(NEWLINE_IDS)})")

print(f"\n--- NLL for each condition (sample 0) ---")
nll_bare = score(s['passage'], s['query'], s['answer'])
print(f"  {'bare':<20} NLL = {nll_bare:.4f}")

for name, prefix_key in [('oracle', 'prefix_oracle'),
                          ('paraphrase', 'prefix_paraphrase'),
                          ('same_topic', 'prefix_same_topic'),
                          ('unrelated_query', 'prefix_unrelated'),
                          ('scrambled_oracle', 'prefix_scrambled'),
                          ('random_matched', 'prefix_random')]:
    nll = score(s['passage'], s['query'], s['answer'],
                prefix_token_ids=s[prefix_key])
    print(f"  {name:<20} NLL = {nll:.4f}  delta = {nll_bare - nll:+.4f}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 8: Scoring loop =====
code(r"""# Cell 8: Scoring loop — 7 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle', 'paraphrase', 'same_topic',
    'unrelated_query', 'scrambled_oracle', 'random_matched',
]

# Semantic relevance ordering (for gradient analysis)
RELEVANCE_ORDER = [
    ('random_matched', 'Random matched', 0, 'none (structural baseline)'),
    ('scrambled_oracle', 'Scrambled oracle', 1, 'vocabulary only'),
    ('unrelated_query', 'Unrelated query', 2, 'low (wrong topic)'),
    ('same_topic', 'Same topic', 3, 'medium (right topic)'),
    ('paraphrase', 'Paraphrase', 4, 'high (same meaning)'),
    ('oracle', 'Oracle', 5, 'maximal (exact query)'),
]

PREFIX_MAP = {
    'oracle': 'prefix_oracle',
    'paraphrase': 'prefix_paraphrase',
    'same_topic': 'prefix_same_topic',
    'unrelated_query': 'prefix_unrelated',
    'scrambled_oracle': 'prefix_scrambled',
    'random_matched': 'prefix_random',
}

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
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {len(COND_NAMES) * N_SAMPLES} scorings")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
        'Q': s['Q'],
    }

    # bare
    result['nll_bare'] = score(s['passage'], s['query'], s['answer'])

    # All prefixed conditions
    for cond_name, prefix_key in PREFIX_MAP.items():
        result[f'nll_{cond_name}'] = score(
            s['passage'], s['query'], s['answer'],
            prefix_token_ids=s[prefix_key]
        )

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
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | "
                   f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 9: Part 1 — Standard condition table =====
code(r"""# Cell 9: Part 1 — Standard Condition Table
print("=" * 70)
print("PART 1: STANDARD CONDITION TABLE")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_nlls = np.array([r['nll_oracle'] for r in results])
oracle_benefit = bare_nlls - oracle_nlls
oracle_d = cohens_d(oracle_benefit)

all_conds = [
    ('oracle', 'Oracle (real query)'),
    ('paraphrase', 'Paraphrase (same meaning)'),
    ('same_topic', 'Same topic (diff question)'),
    ('unrelated_query', 'Unrelated query (wrong topic)'),
    ('scrambled_oracle', 'Scrambled oracle (vocab only)'),
    ('random_matched', 'Random matched (structural)'),
]

alpha_bonf = 0.05 / len(all_conds)

print(f"\n{'Condition':<38} {'NLL':>8} {'Delta':>8} {'d':>8} "
      f"{'Win%':>7} {'%Orc':>6} {'p':>12} {'sig':>5}")
print("-" * 100)

for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    _, p = stats.ttest_1samp(benefit, 0)
    sig = '***' if p < alpha_bonf / 10 else '**' if p < alpha_bonf else '*' if p < 0.05 else 'ns'
    print(f"  {desc:<36} {nlls.mean():>8.4f} {delta:>+8.4f} {d:>+8.3f} "
          f"{win:>6.1f}% {pct:>5.0f}% {p:>12.2e} {sig}")

print(f"\n  bare (lower bound): {bare_nlls.mean():.4f}")
print(f"  Bonferroni threshold: alpha={alpha_bonf:.4f}")
""")


# ===== Cell 10: Part 2 — Semantic Gradient =====
code(r"""# Cell 10: Part 2 — Semantic Gradient
print("=" * 70)
print("PART 2: SEMANTIC GRADIENT")
print("=" * 70)
print("Does NLL improvement increase monotonically with semantic relevance?\n")

random_nlls = np.array([r['nll_random_matched'] for r in results])
random_benefit = bare_nlls - random_nlls

print("--- Raw delta (benefit over bare) ---")
print(f"  {'Condition':<30} {'Relevance':>10} {'Mean delta':>12} {'d':>8} {'%Oracle':>8}")
print(f"  {'-'*75}")

gradient_ds = []
gradient_labels = []

for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"  {desc:<30} {rank:>10} {benefit.mean():>+12.4f} {d:>+8.3f} {pct:>7.0f}%")
    gradient_ds.append(d)
    gradient_labels.append(desc)

# Semantic delta (above structural baseline)
print(f"\n--- Semantic delta (above random_matched baseline) ---")
print(f"  {'Condition':<30} {'Semantic d':>12} {'p vs random':>14} {'sig':>5}")
print(f"  {'-'*65}")

semantic_ds = []

for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    diff = random_nlls - nlls  # positive = condition is better than random
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {desc:<30} {d:>+12.3f} {p:>14.2e} {sig}")
    semantic_ds.append(d)

# Monotonicity test (Spearman)
ranks = [rank for _, _, rank, _ in RELEVANCE_ORDER]
rho, p_mono = stats.spearmanr(ranks, gradient_ds)
sig_mono = '***' if p_mono < 0.001 else '**' if p_mono < 0.01 else '*' if p_mono < 0.05 else 'ns'
print(f"\n--- Monotonicity test ---")
print(f"  Spearman rho (relevance rank vs Cohen's d): rho={rho:+.3f}, "
      f"p={p_mono:.4f} {sig_mono}")

rho_sem, p_sem = stats.spearmanr(ranks, semantic_ds)
sig_sem = '***' if p_sem < 0.001 else '**' if p_sem < 0.01 else '*' if p_sem < 0.05 else 'ns'
print(f"  Spearman rho (relevance rank vs semantic d): rho={rho_sem:+.3f}, "
      f"p={p_sem:.4f} {sig_sem}")

if rho > 0.8 and p_mono < 0.05:
    print(f"  --> MONOTONIC: clear semantic gradient (rho={rho:+.3f})")
elif rho > 0.5:
    print(f"  --> PARTIAL: imperfect gradient (rho={rho:+.3f})")
else:
    print(f"  --> FLAT: no clear gradient (rho={rho:+.3f})")
""")


# ===== Cell 11: Part 3 — Fine-Grained Decomposition Chain =====
code(r"""# Cell 11: Part 3 — Decomposition Chain
print("=" * 70)
print("PART 3: FINE-GRAINED DECOMPOSITION CHAIN")
print("=" * 70)
print("bare -> random_matched -> scrambled_oracle -> unrelated_query -> "
      "same_topic -> paraphrase -> oracle\n")

scrambled_nlls = np.array([r['nll_scrambled_oracle'] for r in results])
unrelated_nlls = np.array([r['nll_unrelated_query'] for r in results])
same_topic_nlls = np.array([r['nll_same_topic'] for r in results])
paraphrase_nlls = np.array([r['nll_paraphrase'] for r in results])

chain = [
    ('Structure', bare_nlls - random_nlls),
    ('Vocabulary', random_nlls - scrambled_nlls),
    ('Query syntax', scrambled_nlls - unrelated_nlls),
    ('Topic relevance', unrelated_nlls - same_topic_nlls),
    ('Semantic precision', same_topic_nlls - paraphrase_nlls),
    ('Exact match', paraphrase_nlls - oracle_nlls),
]

total = bare_nlls - oracle_nlls
total_mean = total.mean()

print(f"  {'Component':<22} {'Delta':>10} {'%total':>8} {'d':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*70}")

chain_pcts = {}
for label, comp in chain:
    mu = comp.mean()
    pct = mu / total_mean * 100 if total_mean != 0 else 0
    d = cohens_d(comp)
    _, p = stats.ttest_1samp(comp, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<22} {mu:>+10.4f} {pct:>7.1f}% {d:>+8.3f} {p:>12.2e} {sig}")
    chain_pcts[label] = pct

print(f"  {'TOTAL':<22} {total_mean:>+10.4f} {'100.0%':>8}")
residual = total_mean - sum(comp.mean() for _, comp in chain)
print(f"\n  Decomposition residual: {residual:.6f} (should be ~0)")

struct_pct = chain_pcts['Structure']
print(f"\n--- Grouped Summary ---")
print(f"  Structure:                {struct_pct:>6.1f}%")
print(f"  All semantic components:  {100 - struct_pct:>6.1f}%")
for label in ['Vocabulary', 'Query syntax', 'Topic relevance',
              'Semantic precision', 'Exact match']:
    print(f"    {label}:{'':>{20-len(label)}} {chain_pcts[label]:>6.1f}%")

# Comparison with v3 Exp 12
print(f"\n--- v3 Exp 12 comparison (T5Gemma encoder-decoder) ---")
print(f"  v3: Structure=86.5%, Vocab=4.3%, Syntax=-1.6%, "
      f"Topic=-15.8%, Precision=7.4%, Exact=19.2%")
print(f"  v4: Structure={struct_pct:.1f}%, Vocab={chain_pcts['Vocabulary']:.1f}%, "
      f"Syntax={chain_pcts['Query syntax']:.1f}%, "
      f"Topic={chain_pcts['Topic relevance']:.1f}%, "
      f"Precision={chain_pcts['Semantic precision']:.1f}%, "
      f"Exact={chain_pcts['Exact match']:.1f}%")
""")


# ===== Cell 12: Part 4 — Hardness Interaction =====
code(r"""# Cell 12: Part 4 — Hardness Interaction
print("=" * 70)
print("PART 4: HARDNESS INTERACTION")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

print("--- Semantic delta by quintile (above random baseline) ---")
print(f"  {'Quintile':<12} {'Bare NLL':>10}", end="")
for _, desc, _, _ in RELEVANCE_ORDER[1:]:
    print(f"  {desc:>12}", end="")
print()
print(f"  {'-'*(14 + 14 * (len(RELEVANCE_ORDER) - 1))}")

for q in range(5):
    mask = quintiles == q
    row = f"  {q_labels[q]:<12} {bare_nlls[mask].mean():>10.3f}"
    for cond, desc, _, _ in RELEVANCE_ORDER[1:]:
        nlls_c = np.array([r[f'nll_{cond}'] for r in results])[mask]
        rand_c = random_nlls[mask]
        sem_delta = cohens_d(rand_c - nlls_c)
        row += f"  {sem_delta:>+12.3f}"
    print(row)

print(f"\n--- Structure vs semantic % by quintile ---")
print(f"  {'Quintile':<12} {'Struct%':>9} {'Vocab%':>8} {'Syntax%':>9} "
      f"{'Topic%':>8} {'Precis%':>9} {'Exact%':>8}")
print(f"  {'-'*65}")

for q in range(5):
    mask = quintiles == q
    total_q = (bare_nlls[mask] - oracle_nlls[mask]).mean()
    if total_q > 0:
        s_pct = (bare_nlls[mask] - random_nlls[mask]).mean() / total_q * 100
        v_pct = (random_nlls[mask] - scrambled_nlls[mask]).mean() / total_q * 100
        syn_pct = (scrambled_nlls[mask] - unrelated_nlls[mask]).mean() / total_q * 100
        top_pct = (unrelated_nlls[mask] - same_topic_nlls[mask]).mean() / total_q * 100
        pre_pct = (same_topic_nlls[mask] - paraphrase_nlls[mask]).mean() / total_q * 100
        ex_pct = (paraphrase_nlls[mask] - oracle_nlls[mask]).mean() / total_q * 100
    else:
        s_pct = v_pct = syn_pct = top_pct = pre_pct = ex_pct = 0
    print(f"  {q_labels[q]:<12} {s_pct:>8.1f}% {v_pct:>7.1f}% {syn_pct:>8.1f}% "
          f"{top_pct:>7.1f}% {pre_pct:>8.1f}% {ex_pct:>7.1f}%")

# Correlation: hardness vs semantic delta
print(f"\n--- Correlations: hardness vs semantic delta ---")
for cond, desc, rank, _ in RELEVANCE_ORDER[1:]:
    nlls_c = np.array([r[f'nll_{cond}'] for r in results])
    sem_delta = random_nlls - nlls_c
    r_val, p_val = stats.pearsonr(bare_nlls, sem_delta)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"  {desc:<25} r={r_val:+.3f} (p={p_val:.2e}) {sig}")
""")


# ===== Cell 13: Synthesis + Save =====
code(r"""# Cell 13: Synthesis + Save
print("=" * 70)
print("SYNTHESIS: GRADED SEMANTIC RELEVANCE (DECODER-ONLY)")
print("=" * 70)

print(f"\n1. SEMANTIC GRADIENT (d, % oracle):")
for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    d = cohens_d(bare_nlls - nlls)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"   [{rank}] {desc:<25} d={d:>+.3f} ({pct:>5.1f}% oracle) — {rel_desc}")

print(f"\n2. MONOTONICITY:")
print(f"   Spearman rho (raw d): {rho:+.3f} (p={p_mono:.4f})")
print(f"   Spearman rho (semantic d): {rho_sem:+.3f} (p={p_sem:.4f})")

print(f"\n3. DECOMPOSITION CHAIN:")
for label, pct in chain_pcts.items():
    print(f"   {label:<22} {pct:>6.1f}%")

print(f"\n{'='*70}")
print("CONCLUSIONS:")

if rho > 0.8 and p_mono < 0.05:
    print(f"  1. CLEAR SEMANTIC GRADIENT: monotonic (rho={rho:+.3f}, p={p_mono:.4f})")
elif rho > 0.5:
    print(f"  1. PARTIAL GRADIENT: imperfect (rho={rho:+.3f}, p={p_mono:.4f})")
else:
    print(f"  1. NO CLEAR GRADIENT (rho={rho:+.3f}, p={p_mono:.4f})")

if struct_pct > 75:
    print(f"  2. Structure dominates ({struct_pct:.0f}%)")
elif struct_pct > 50:
    print(f"  2. Structure largest ({struct_pct:.0f}%) but semantics substantial")
else:
    print(f"  2. Semantics dominate ({100-struct_pct:.0f}%)")

sem_components = [(k, v) for k, v in chain_pcts.items() if k != 'Structure']
largest_sem = max(sem_components, key=lambda x: abs(x[1]))
print(f"  3. Largest semantic component: {largest_sem[0]} ({largest_sem[1]:+.1f}%)")

# Cross-architecture comparison
print(f"\n--- Cross-architecture comparison ---")
print(f"  v3 (T5Gemma encoder-decoder): monotonic rho=+0.943, structure=86.5%")
print(f"  v4 (Gemma 3 decoder-only):    rho={rho:+.3f}, structure={struct_pct:.1f}%")
print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'v4_decoder_only_exp06_semantic_gradient',
    'scoring_model': MODEL_NAME,
    'generation_model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {},
    'gradient': {
        'spearman_rho_raw': float(rho),
        'spearman_p_raw': float(p_mono),
        'spearman_rho_semantic': float(rho_sem),
        'spearman_p_semantic': float(p_sem),
    },
    'decomposition_chain': {k: float(v) for k, v in chain_pcts.items()},
    'structure_pct': float(struct_pct),
    'query_token_stats': {
        'mean': float(np.mean([r['Q'] for r in results])),
        'median': float(np.median([r['Q'] for r in results])),
        'min': int(np.min([r['Q'] for r in results])),
        'max': int(np.max([r['Q'] for r in results])),
    },
}

for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    _, p = stats.ttest_1samp(benefit, 0)
    sem_diff = random_nlls - nlls
    sem_d = cohens_d(sem_diff)
    _, sem_p = stats.ttest_1samp(sem_diff, 0)
    final_results['conditions'][cond] = {
        'description': desc,
        'relevance_rank': rank,
        'relevance_label': rel_desc,
        'd': float(d),
        'mean_nll': float(nlls.mean()),
        'mean_delta': float(benefit.mean()),
        'pct_oracle': float(d / oracle_d * 100) if oracle_d > 0 else 0,
        'p': float(p),
        'semantic_d': float(sem_d),
        'semantic_p': float(sem_p),
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
out_path = "experiments/decoder_only/06/06_semantic_gradient.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
