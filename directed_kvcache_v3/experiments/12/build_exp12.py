#!/usr/bin/env python3
# Build Exp 12 notebook: Graded Semantic Relevance Sweep.
#
# Two-phase design:
#   Phase 1: Load Gemma 2 9B-IT, generate paraphrases + same-topic queries, save JSON, free VRAM
#   Phase 2: Load T5Gemma, score 7 conditions x 500 samples, save results
#
# 7 conditions, 500 samples = 3,500 scoring passes.
# Total runtime: ~50 min generation + ~30 min scoring + overhead ~ 90 min.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 12: Graded Semantic Relevance Sweep

## Motivation

Exp 2B/3E established the mechanism decomposition: ~85% structural, ~10% semantic,
~6% vocabulary. The structural effect is essentially binary (any 5+ token prefix
triggers it). But the semantic component was measured only as a binary contrast
(oracle vs scrambled). To properly isolate and characterize the semantic gradient,
we need experiments that **hold structure constant** and vary only semantic content.

This experiment traces the full semantic gradient across 6 levels of relevance,
all length-matched to control for structural effects.

## Conditions (7)

| # | Condition | Prefix | Semantic relevance |
|---|-----------|--------|--------------------|
| 1 | `bare` | (none) | N/A (lower bound) |
| 2 | `oracle_trunc` | real query | maximal |
| 3 | `paraphrase_trunc` | LLM paraphrase of query | high (same meaning, different words) |
| 4 | `same_topic_trunc` | LLM question about same topic | medium (right topic, wrong question) |
| 5 | `unrelated_query_trunc` | real query from different sample | low (real query syntax, wrong topic) |
| 6 | `scrambled_oracle_trunc` | oracle words shuffled | vocabulary only (right words, no structure) |
| 7 | `random_matched_trunc` | words from random passage | none (structural baseline) |

All non-bare conditions are length-matched to oracle query word count per sample.

## Analysis

- **Part 1**: Standard condition table (d, win%, p, % oracle)
- **Part 2**: Semantic gradient — delta by relevance rank, monotonicity test
- **Part 3**: Fine-grained decomposition chain
- **Part 4**: Hardness interaction — does the semantic gradient steepen for harder samples?""")


# ===== Cell 2: Setup =====
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

sys.path.insert(0, "../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
T5GEMMA_NAME = "google/t5gemma-2-4b-4b"
GEMMA_IT_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../results/exp12")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_PATH = RESULTS_DIR / "surrogates.json"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

# Prompt templates for Gemma 2 9B-IT
PROMPT_PARAPHRASE = (
    "Rephrase this search query using completely different words but keeping "
    "the same meaning. Keep it to 5-8 words. Output only the rephrased query."
)
PROMPT_SAME_TOPIC = (
    "Write a question about the same topic as this document but asking for "
    "DIFFERENT information. Keep it to 5-8 words. Output only the question."
)

print("Exp 12: Graded Semantic Relevance Sweep")
print(f"N: {N_SAMPLES}")
print(f"Generation model: {GEMMA_IT_NAME}")
print(f"Scoring model: {T5GEMMA_NAME}")
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

# Show 5 examples
for i in range(5):
    s = samples[i]
    print(f"\nExample {i}:")
    print(f"  Q: {s['query']}")
    print(f"  A: {s['answer'][:80]}")
    print(f"  P ({s['word_count']}w): {s['passage'][:100]}...")
""")


# ===== Cell 4: Phase 1 — LLM surrogate generation =====
code(r"""# Cell 4: Phase 1 — Generate surrogates with Gemma 2 9B-IT
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
    print(f"Loading {GEMMA_IT_NAME} for surrogate generation...")
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    HF_TOKEN = os.environ.get("HF_TOKEN")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    gen_tokenizer = AutoTokenizer.from_pretrained(GEMMA_IT_NAME, token=HF_TOKEN)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
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
            tqdm.write(f"  Gen checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

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


# ===== Cell 5: Inspect surrogates =====
code(r"""# Cell 5: Inspect surrogates — examples, word counts, vocabulary overlap

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

print("=" * 70)
print("SURROGATE INSPECTION")
print("=" * 70)

# Show 5 examples
for idx in range(5):
    s = samples[idx]
    surr = surrogates[idx]
    print(f"\n--- Sample {idx} ---")
    print(f"  Passage:     {s['passage'][:100]}...")
    print(f"  Query:       {s['query']}")
    print(f"  Paraphrase:  {surr['paraphrase']}")
    print(f"  Same-topic:  {surr['same_topic']}")

# Word count distributions
print(f"\n--- Word count distributions ---")
for label, key in [('oracle query', None), ('paraphrase', 'paraphrase'),
                    ('same_topic', 'same_topic')]:
    if key is None:
        wc = np.array([len(s['query'].split()) for s in samples])
    else:
        wc = np.array([len(surr[key].split()) for surr in surrogates])
    print(f"  {label}: mean={wc.mean():.1f}, median={np.median(wc):.0f}, "
          f"range=[{wc.min()}, {wc.max()}], std={wc.std():.1f}")

# Vocabulary overlap with document (content words only)
print(f"\n--- Vocabulary overlap with document (content words) ---")
for label, get_text in [
    ('oracle query', lambda i: samples[i]['query']),
    ('paraphrase', lambda i: surrogates[i]['paraphrase']),
    ('same_topic', lambda i: surrogates[i]['same_topic']),
]:
    overlaps = []
    for i in range(N_SAMPLES):
        doc_words = set(re.sub(r'[^\w\s]', '', samples[i]['passage'].lower()).split())
        doc_content = doc_words - STOP_WORDS
        text_words = set(re.sub(r'[^\w\s]', '', get_text(i).lower()).split())
        text_content = text_words - STOP_WORDS
        if len(text_content) > 0:
            overlap = len(text_content & doc_content) / len(text_content)
        else:
            overlap = 0.0
        overlaps.append(overlap)
    overlaps = np.array(overlaps)
    print(f"  {label}: mean={overlaps.mean():.3f}, median={np.median(overlaps):.3f}")

# Vocabulary overlap with oracle query (content words only)
print(f"\n--- Vocabulary overlap with oracle query (content words) ---")
for label, get_text in [
    ('paraphrase', lambda i: surrogates[i]['paraphrase']),
    ('same_topic', lambda i: surrogates[i]['same_topic']),
]:
    overlaps = []
    for i in range(N_SAMPLES):
        q_words = set(re.sub(r'[^\w\s]', '', samples[i]['query'].lower()).split())
        q_content = q_words - STOP_WORDS
        text_words = set(re.sub(r'[^\w\s]', '', get_text(i).lower()).split())
        text_content = text_words - STOP_WORDS
        if len(text_content) > 0 and len(q_content) > 0:
            overlap = len(text_content & q_content) / len(text_content)
        else:
            overlap = 0.0
        overlaps.append(overlap)
    overlaps = np.array(overlaps)
    print(f"  {label}: mean={overlaps.mean():.3f}, median={np.median(overlaps):.3f}")
""")


# ===== Cell 6: Phase 2 — Load T5Gemma + define helpers =====
code(r"""# Cell 6: Phase 2 — Load T5Gemma and define scoring helpers
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {T5GEMMA_NAME}...")
processor = AutoProcessor.from_pretrained(T5GEMMA_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    T5GEMMA_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)
    if ans_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            labels=ans_ids,
        )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    # Count how many tokens the prefix occupies in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


print("Helpers defined.")
""")


# ===== Cell 7: Generate all conditions per sample =====
code(r"""# Cell 7: Generate all 7 scoring conditions per sample

for i, s in enumerate(samples):
    surr = surrogates[i]
    query = s['query']
    passage = s['passage']
    query_words_list = query.split()
    n_query_words = len(query_words_list)

    # Unrelated query: query from a distant sample
    unrelated_idx = (i + N_SAMPLES // 2) % len(samples)
    unrelated_query = samples[unrelated_idx]['query']
    # Length-match to oracle: truncate or pad to same word count
    unrelated_words = unrelated_query.split()[:n_query_words]
    if len(unrelated_words) < n_query_words:
        # Pad with words from another sample if too short
        pad_idx = (i + N_SAMPLES // 3) % len(samples)
        pad_words = samples[pad_idx]['query'].split()
        unrelated_words = (unrelated_words + pad_words)[:n_query_words]
    s['unrelated_query'] = " ".join(unrelated_words)

    # Scrambled oracle: same words, random order
    rng = np.random.RandomState(SEED + i)
    shuffled = list(query_words_list)
    rng.shuffle(shuffled)
    s['scrambled_oracle'] = " ".join(shuffled)

    # Random matched: words from unrelated passage, same word count as oracle
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    s['random_matched'] = " ".join(other_words[:n_query_words])

    # Paraphrase and same-topic from LLM generation
    s['paraphrase'] = surr['paraphrase']
    s['same_topic'] = surr['same_topic']

    # Oracle (just the query)
    s['oracle'] = query

# Define all scoring conditions
COND_NAMES = [
    'bare',
    'oracle_trunc',
    'paraphrase_trunc',
    'same_topic_trunc',
    'unrelated_query_trunc',
    'scrambled_oracle_trunc',
    'random_matched_trunc',
]

# Semantic relevance ordering (for gradient analysis)
RELEVANCE_ORDER = [
    ('random_matched_trunc', 'Random matched', 0, 'none (structural baseline)'),
    ('scrambled_oracle_trunc', 'Scrambled oracle', 1, 'vocabulary only'),
    ('unrelated_query_trunc', 'Unrelated query', 2, 'low (wrong topic)'),
    ('same_topic_trunc', 'Same topic', 3, 'medium (right topic)'),
    ('paraphrase_trunc', 'Paraphrase', 4, 'high (same meaning)'),
    ('oracle_trunc', 'Oracle', 5, 'maximal (exact query)'),
]

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query:   {ex['query'][:80]}")
print(f"  Answer:  {ex['answer'][:80]}")
print(f"  Passage: {ex['passage'][:80]}...")
print()
for c in COND_NAMES:
    if c == 'bare':
        print(f"  {c:<30}: [document only]")
    else:
        key = c.replace('_trunc', '')
        text = ex[key]
        ptoks = count_prefix_tokens(text, ex['passage'])
        print(f"  {c:<30} ({ptoks:>3} toks): {str(text)[:55]}")

# Token count stats across first 50 samples
print(f"\nPrefix token counts (first 50 samples):")
for c in COND_NAMES:
    if c == 'bare':
        continue
    key = c.replace('_trunc', '')
    toks = [count_prefix_tokens(s[key], s['passage']) for s in samples[:50]]
    print(f"  {c:<30} mean={np.mean(toks):.1f}, range=[{min(toks)}, {max(toks)}]")
""")


# ===== Cell 8: Scoring loop with checkpointing =====
code(r"""# Cell 8: Scoring loop with checkpointing

print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

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
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['passage'], s['answer'])
            result['nll_bare'] = nll
        else:
            key = cond.replace('_trunc', '')
            prefix = s[key]
            enc_text = prefix + "\n" + s['passage']
            ptoks = count_prefix_tokens(prefix, s['passage'])
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result[f'nll_{cond}'] = nll
            result[f'ptoks_{cond}'] = ptoks

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


# ===== Cell 9: Part 1 — Standard condition table =====
code(r"""# Cell 9: Part 1 — Standard Condition Table

print("=" * 70)
print("PART 1: STANDARD CONDITION TABLE")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_nlls = np.array([r['nll_oracle_trunc'] for r in results])
oracle_benefit = bare_nlls - oracle_nlls
oracle_d = cohens_d(oracle_benefit)

all_conds = [
    ('oracle_trunc', 'Oracle (real query)'),
    ('paraphrase_trunc', 'Paraphrase (same meaning)'),
    ('same_topic_trunc', 'Same topic (diff question)'),
    ('unrelated_query_trunc', 'Unrelated query (wrong topic)'),
    ('scrambled_oracle_trunc', 'Scrambled oracle (vocab only)'),
    ('random_matched_trunc', 'Random matched (structural)'),
]

# Bonferroni threshold
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

# Compute delta = bare_nll - cond_nll for each condition
# Use random_matched as structural baseline: semantic_delta = delta - random_delta
random_nlls = np.array([r['nll_random_matched_trunc'] for r in results])
random_benefit = bare_nlls - random_nlls  # structural benefit

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
print(f"  Spearman rho (relevance rank vs Cohen's d): rho={rho:+.3f}, p={p_mono:.4f} {sig_mono}")

rho_sem, p_sem = stats.spearmanr(ranks, semantic_ds)
sig_sem = '***' if p_sem < 0.001 else '**' if p_sem < 0.01 else '*' if p_sem < 0.05 else 'ns'
print(f"  Spearman rho (relevance rank vs semantic d): rho={rho_sem:+.3f}, p={p_sem:.4f} {sig_sem}")

if rho > 0.8 and p_mono < 0.05:
    print(f"  --> MONOTONIC: clear semantic gradient (rho={rho:+.3f})")
elif rho > 0.5:
    print(f"  --> PARTIAL: imperfect gradient (rho={rho:+.3f})")
else:
    print(f"  --> FLAT: no clear gradient (rho={rho:+.3f})")
""")


# ===== Cell 11: Part 3 — Fine-Grained Decomposition Chain =====
code(r"""# Cell 11: Part 3 — Fine-Grained Decomposition Chain

print("=" * 70)
print("PART 3: FINE-GRAINED DECOMPOSITION CHAIN")
print("=" * 70)
print("bare -> random_matched -> scrambled_oracle -> unrelated_query -> "
      "same_topic -> paraphrase -> oracle\n")
print("Each step adds one aspect of semantic relevance:\n")
print("  bare -> random_matched:     STRUCTURE (any prefix helps)")
print("  random_matched -> scrambled: VOCABULARY (right words, wrong order)")
print("  scrambled -> unrelated:      QUERY SYNTAX (real query structure, wrong topic)")
print("  unrelated -> same_topic:     TOPIC RELEVANCE (right topic, different question)")
print("  same_topic -> paraphrase:    SEMANTIC PRECISION (same meaning, diff words)")
print("  paraphrase -> oracle:        EXACT MATCH (same words + meaning)")
print()

scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in results])
unrelated_nlls = np.array([r['nll_unrelated_query_trunc'] for r in results])
same_topic_nlls = np.array([r['nll_same_topic_trunc'] for r in results])
paraphrase_nlls = np.array([r['nll_paraphrase_trunc'] for r in results])

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

# Grouped summary: structure vs all semantic components
struct_pct = chain_pcts['Structure']
semantic_pct = 100 - struct_pct
print(f"\n--- Grouped Summary ---")
print(f"  Structure:                {struct_pct:>6.1f}%")
print(f"  All semantic components:  {semantic_pct:>6.1f}%")
print(f"    Vocabulary:             {chain_pcts['Vocabulary']:>6.1f}%")
print(f"    Query syntax:           {chain_pcts['Query syntax']:>6.1f}%")
print(f"    Topic relevance:        {chain_pcts['Topic relevance']:>6.1f}%")
print(f"    Semantic precision:     {chain_pcts['Semantic precision']:>6.1f}%")
print(f"    Exact match:            {chain_pcts['Exact match']:>6.1f}%")

# Compare with Exp 2B
print(f"\n--- Comparison with Exp 2B (3-way decomposition) ---")
print(f"  Exp 2B: Structure=84.7%, Vocabulary=5.5%, Semantics=9.7%")
vocab_total = chain_pcts['Vocabulary']
sem_total = (chain_pcts['Query syntax'] + chain_pcts['Topic relevance'] +
             chain_pcts['Semantic precision'] + chain_pcts['Exact match'])
print(f"  Exp 12: Structure={struct_pct:.1f}%, Vocabulary={vocab_total:.1f}%, "
      f"Semantics (broad)={sem_total:.1f}%")
""")


# ===== Cell 12: Part 4 — Hardness Interaction =====
code(r"""# Cell 12: Part 4 — Hardness Interaction

print("=" * 70)
print("PART 4: HARDNESS INTERACTION")
print("=" * 70)
print("Does the semantic gradient steepen for harder samples?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

# Semantic delta (cond_nll - random_nll) by quintile for each condition
print("--- Semantic delta by quintile (above random baseline) ---")
print(f"  {'Quintile':<12} {'Bare NLL':>10}", end="")
for _, desc, _, _ in RELEVANCE_ORDER[1:]:  # skip random itself
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

# Decomposition chain by quintile
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

# Correlation: hardness vs semantic delta for each condition
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
print("SYNTHESIS: GRADED SEMANTIC RELEVANCE RESULTS")
print("=" * 70)

# 1. Gradient summary
print(f"\n1. SEMANTIC GRADIENT (d, % oracle):")
for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    d = cohens_d(bare_nlls - nlls)
    pct = d / oracle_d * 100 if oracle_d > 0 else 0
    print(f"   [{rank}] {desc:<25} d={d:>+.3f} ({pct:>5.1f}% oracle) — {rel_desc}")

# 2. Monotonicity
print(f"\n2. MONOTONICITY:")
print(f"   Spearman rho (raw d): {rho:+.3f} (p={p_mono:.4f})")
print(f"   Spearman rho (semantic d): {rho_sem:+.3f} (p={p_sem:.4f})")

# 3. Decomposition chain
print(f"\n3. DECOMPOSITION CHAIN:")
for label, pct in chain_pcts.items():
    print(f"   {label:<22} {pct:>6.1f}%")

# 4. Conclusions
print(f"\n{'='*70}")
print("CONCLUSIONS:")

if rho > 0.8 and p_mono < 0.05:
    print(f"  1. CLEAR SEMANTIC GRADIENT: increasing relevance monotonically improves NLL")
    print(f"     (Spearman rho={rho:+.3f}, p={p_mono:.4f})")
elif rho > 0.5:
    print(f"  1. PARTIAL GRADIENT: general trend but imperfect monotonicity")
    print(f"     (Spearman rho={rho:+.3f}, p={p_mono:.4f})")
else:
    print(f"  1. NO CLEAR GRADIENT: semantic relevance does not reliably predict benefit")
    print(f"     (Spearman rho={rho:+.3f}, p={p_mono:.4f})")

if struct_pct > 75:
    print(f"  2. Structure still dominates ({struct_pct:.0f}%), consistent with Exp 2B")
elif struct_pct > 50:
    print(f"  2. Structure is largest component ({struct_pct:.0f}%) but semantic components "
          f"are substantial ({100-struct_pct:.0f}%)")
else:
    print(f"  2. Semantic components dominate ({100-struct_pct:.0f}%) — "
          f"the fine-grained gradient reveals structure is not dominant")

# Which semantic component is largest?
sem_components = [(k, v) for k, v in chain_pcts.items() if k != 'Structure']
largest_sem = max(sem_components, key=lambda x: abs(x[1]))
print(f"  3. Largest semantic component: {largest_sem[0]} ({largest_sem[1]:+.1f}%)")

print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp12_semantic_gradient',
    'generation_model': GEMMA_IT_NAME,
    'scoring_model': T5GEMMA_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
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
}

# Add per-condition results
for cond, desc, rank, rel_desc in RELEVANCE_ORDER:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    _, p = stats.ttest_1samp(benefit, 0)
    # Semantic delta vs random
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
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/12/12_semantic_gradient.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
