#!/usr/bin/env python3
# Build Exp 02: Token-Matched Semantic Probing with LLM Surrogates.
#
# Definitive experiment eliminating ALL structural confounds via token-level
# prefix matching. Every prefixed condition uses exactly Q token IDs per sample
# (Q = number of real query tokens), equalizing RoPE delta, cache length, and
# BOS removal across all 13 conditions.
#
# Scoring: BOS-retained repositioning on Gemma 3 12B-IT (same as Exp 01 rerun).
# LLM generation: same model generates 5 document-specific surrogates per sample.
#
# 13 conditions, N=400, SEED=42.

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 02: Token-Matched Semantic Probing with LLM Surrogates

## Motivation

Exp 01 rerun (Gemma 3 12B-IT, N=400, BOS-retained repositioning) established that
the semantic effect is real but **reversed from expectations**: oracle conditioning
HURTS (d=-0.151), while the data-extraction task-framing prefix HELPS (d=+0.264).
However, Exp 01 has a **prefix-length confound** — different conditions have different
token counts (P=14-20), producing different RoPE repositioning deltas.

Exps 02-05 were invalidated by the 1-token look-ahead bug. Exps 06-07 used
`slice_kv_cache` without BOS retention on a different model (4B-IT) and are not
directly comparable.

## Goal

Design and run a definitive experiment that:
1. Eliminates all structural confounds via token-level prefix matching
2. Spans the full semantic gradient
3. Introduces LLM-generated document-specific surrogates
4. Deeply probes where conditioning helps vs hurts

## Method — BOS-Retained Repositioning with Token-Level Matching

**Phase A:** `[BOS] + prefix_ids(Q) + [\n] + doc_ids(D)` at natural positions.
Select BOS + doc from cache (skip prefix + newline).
Reposition doc keys from `[1+Q+NL, ..., Q+NL+D]` to `[1, ..., D]`.
Cache has `1+D` entries (BOS at 0, doc at 1..D).

**Phase B:** `[\n + query + \n + answer]` at positions `[D+1, ...]`.
`cache_position` auto-generated from cache length (= 1+D = D+1). No look-ahead.

**Key invariant:** For ALL prefixed conditions, repositioning delta = -(Q+NL) is
**identical** per sample. Every condition uses exactly Q prefix token IDs.

## Conditions (13 total)

| # | Key | Source | Semantic relevance | Token construction |
|---|-----|--------|-------------------|--------------------|
| 1 | `bare` | — | baseline | No prefix |
| 2 | `random_tokens` | Random vocab IDs | none | Q random IDs from vocab |
| 3 | `repeat_token` | Single token x Q | none (structural) | Token ID 1000 repeated Q times |
| 4 | `scrambled_oracle` | Shuffled query | vocab match only | Random permutation of oracle IDs |
| 5 | `unrelated_query` | Other sample's query | low | Sample (i+N/2)%N query, pad/trunc to Q |
| 6 | `same_topic` | LLM-generated | medium | "Write a question about same topic..." pad/trunc to Q |
| 7 | `paraphrase` | LLM-generated | high | "Rephrase this query differently..." pad/trunc to Q |
| 8 | `oracle` | Real query | maximal | Exact query token IDs (already Q) |
| 9 | `llm_extract` | LLM doc-specific | task-framing (doc) | "List key facts from this document" pad/trunc to Q |
| 10 | `llm_question` | LLM doc-specific | query-like (doc) | "What question does this doc answer?" pad/trunc to Q |
| 11 | `llm_summarize` | LLM doc-specific | summary (doc) | "Summarize in one sentence" pad/trunc to Q |
| 12 | `extractor_matched` | Fixed text | task-framing (generic) | "Extract:" text tokenized, pad/trunc to Q |
| 13 | `adversarial_matched` | Fixed text | adversarial | Adversarial text tokenized, pad/trunc to Q |

## Key analyses

1. **Semantic gradient test**: Spearman rho of relevance rank vs delta_NLL
2. **Structural decomposition**: delta(random_tokens) / delta(oracle)
3. **LLM doc-specific vs generic**: paired test llm_extract vs extractor_matched
4. **Hardness interaction**: 5 quintile bins x 13 conditions
5. **Per-sample ranking**: which condition gives lowest NLL per sample""")


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
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp02")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_PATH = RESULTS_DIR / "surrogates.json"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = getattr(text_cfg, 'vocab_size', 262208)

print(f"Exp 02: Token-Matched Semantic Probing with LLM Surrogates")
print(f"Scoring: BOS-retained repositioning (look-ahead fix)")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
print(f"Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
rope_params = getattr(text_cfg, 'rope_parameters', {})
layer_types_list = getattr(text_cfg, 'layer_types', [])
print(f"Layer types: {set(layer_types_list)} ({len(layer_types_list)} layers)")
for ltype, params in rope_params.items():
    print(f"  {ltype}: theta={params.get('rope_theta')}, "
          f"type={params.get('rope_type')}, factor={params.get('factor', 'N/A')}")
n_global = sum(1 for t in layer_types_list if t == 'full_attention')
print(f"  Global layers: {n_global}/{len(layer_types_list)} "
      f"(indices: {[i for i, t in enumerate(layer_types_list) if t == 'full_attention']})")

# Load MS MARCO
from lib.data import count_words
from datasets import load_dataset

print("\nLoading MS MARCO v1.1 validation...")
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

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"\nFirst sample:")
print(f"  Query:  {samples[0]['query'][:70]}...")
print(f"  Answer: {samples[0]['answer'][:70]}...")
print(f"  Passage ({samples[0]['word_count']}w): {samples[0]['passage'][:70]}...")
""")


# ===== Cell 3: LLM surrogate generation =====
code(r"""# Cell 3: Generate LLM surrogates (5 per sample)

PROMPT_PARAPHRASE = (
    "Rephrase this search query using completely different words but keeping "
    "the same meaning. Keep it to 5-8 words. Output only the rephrased query."
)
PROMPT_SAME_TOPIC = (
    "Write a question about the same topic as this document but asking for "
    "DIFFERENT information. Keep it to 5-8 words. Output only the question."
)
PROMPT_EXTRACT = (
    "List the key facts from this document as a brief comma-separated list. "
    "Output only the fact list, nothing else."
)
PROMPT_QUESTION = (
    "What question does this document answer? Write only the question, "
    "nothing else. Keep it to 5-10 words."
)
PROMPT_SUMMARIZE = (
    "Summarize this document in one sentence. Output only the summary, nothing else."
)

def generate_text(input_text, prompt_text, max_new_tokens=50):
    # Generate text from a prompt + input using Gemma IT chat template.
    messages = [
        {"role": "user",
         "content": f"{prompt_text}\n\n{input_text}"}
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt",
                       truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Post-process: strip, take first line, remove quotes, truncate to 20 words
    cleaned = raw_text.strip().split("\n")[0].strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    cleaned = " ".join(cleaned.split()[:20])
    return cleaned


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
                  desc="Generating surrogates"):
        s = samples[i]
        entry = {'query': s['query']}

        # First 200 words of passage for doc-specific prompts
        doc_words = s['passage'].split()[:200]
        doc_input = f"Document:\n{' '.join(doc_words)}"

        # 1. Paraphrase: rephrase the query
        torch.manual_seed(SEED + i * 10)
        entry['paraphrase'] = generate_text(
            f"Query: {s['query']}", PROMPT_PARAPHRASE
        )

        # 2. Same-topic: question about same topic but different info
        torch.manual_seed(SEED + i * 10 + 1)
        entry['same_topic'] = generate_text(doc_input, PROMPT_SAME_TOPIC)

        # 3. LLM extract: key facts from document
        torch.manual_seed(SEED + i * 10 + 2)
        entry['llm_extract'] = generate_text(doc_input, PROMPT_EXTRACT)

        # 4. LLM question: what question does the doc answer?
        torch.manual_seed(SEED + i * 10 + 3)
        entry['llm_question'] = generate_text(doc_input, PROMPT_QUESTION)

        # 5. LLM summarize: one-sentence summary
        torch.manual_seed(SEED + i * 10 + 4)
        entry['llm_summarize'] = generate_text(doc_input, PROMPT_SUMMARIZE)

        surrogates.append(entry)

        if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
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

# Show examples
for i in range(3):
    s = surrogates[i]
    print(f"\nSample {i}: query='{s['query'][:60]}'")
    for key in ['paraphrase', 'same_topic', 'llm_extract', 'llm_question', 'llm_summarize']:
        print(f"  {key:<15}: {s.get(key, 'N/A')[:60]}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 4: Prefix construction =====
code(r"""# Cell 4: Build per-sample token-level prefix IDs (13 conditions)

# Fixed-text prefixes
EXTRACTOR_TEXT = "Extract all key data points, facts, entities, and specific attributes from the following text."
ADVERSARIAL_TEXT = "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt mixed together."

def make_prefix(token_ids, Q):
    # Pad or truncate token_ids to exactly Q tokens.
    if len(token_ids) >= Q:
        return token_ids[:Q]
    else:
        padded = token_ids * ((Q // max(len(token_ids), 1)) + 1)
        return padded[:Q]


pyrandom.seed(SEED + 200)
np.random.seed(SEED + 300)

# Pre-tokenize fixed texts
extractor_ids = tokenizer(EXTRACTOR_TEXT, add_special_tokens=False).input_ids
adversarial_ids = tokenizer(ADVERSARIAL_TEXT, add_special_tokens=False).input_ids

# Special token IDs to exclude from random sampling
special_ids = set(tokenizer.all_special_ids)

for i, s in enumerate(samples):
    surr = surrogates[i]
    q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
    Q = len(q_ids)
    s['Q'] = Q

    # 1. oracle: exact query token IDs (already Q tokens)
    s['prefix_oracle'] = q_ids

    # 2. random_tokens: random vocab IDs (excluding special tokens)
    rand_ids = []
    while len(rand_ids) < Q:
        tid = np.random.randint(0, VOCAB_SIZE)
        if tid not in special_ids:
            rand_ids.append(int(tid))
    s['prefix_random_tokens'] = rand_ids[:Q]

    # 3. repeat_token: single token repeated Q times
    s['prefix_repeat_token'] = [1000] * Q

    # 4. scrambled_oracle: random permutation of query IDs
    shuffled = list(q_ids)
    pyrandom.shuffle(shuffled)
    s['prefix_scrambled_oracle'] = shuffled

    # 5. unrelated_query: other sample's query, pad/trunc to Q
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_q_ids = tokenizer(samples[other_idx]['query'],
                            add_special_tokens=False).input_ids
    s['prefix_unrelated_query'] = make_prefix(other_q_ids, Q)

    # 6. same_topic: LLM-generated, pad/trunc to Q
    topic_ids = tokenizer(surr['same_topic'], add_special_tokens=False).input_ids
    s['prefix_same_topic'] = make_prefix(topic_ids, Q)

    # 7. paraphrase: LLM-generated, pad/trunc to Q
    para_ids = tokenizer(surr['paraphrase'], add_special_tokens=False).input_ids
    s['prefix_paraphrase'] = make_prefix(para_ids, Q)

    # 8. llm_extract: LLM doc-specific fact list, pad/trunc to Q
    extract_ids = tokenizer(surr['llm_extract'], add_special_tokens=False).input_ids
    s['prefix_llm_extract'] = make_prefix(extract_ids, Q)

    # 9. llm_question: LLM doc-specific question, pad/trunc to Q
    question_ids = tokenizer(surr['llm_question'], add_special_tokens=False).input_ids
    s['prefix_llm_question'] = make_prefix(question_ids, Q)

    # 10. llm_summarize: LLM doc-specific summary, pad/trunc to Q
    summarize_ids = tokenizer(surr['llm_summarize'], add_special_tokens=False).input_ids
    s['prefix_llm_summarize'] = make_prefix(summarize_ids, Q)

    # 11. extractor_matched: fixed extraction text, pad/trunc to Q
    s['prefix_extractor_matched'] = make_prefix(extractor_ids, Q)

    # 12. adversarial_matched: fixed adversarial text, pad/trunc to Q
    s['prefix_adversarial_matched'] = make_prefix(adversarial_ids, Q)

# Verify all prefixes have exactly Q tokens
PREFIX_KEYS = [
    'prefix_oracle', 'prefix_random_tokens', 'prefix_repeat_token',
    'prefix_scrambled_oracle', 'prefix_unrelated_query', 'prefix_same_topic',
    'prefix_paraphrase', 'prefix_llm_extract', 'prefix_llm_question',
    'prefix_llm_summarize', 'prefix_extractor_matched', 'prefix_adversarial_matched',
]

q_lens = [s['Q'] for s in samples]
print(f"Query token count — mean: {np.mean(q_lens):.1f}, "
      f"median: {np.median(q_lens):.0f}, "
      f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

errors = 0
for i, s in enumerate(samples):
    Q = s['Q']
    for key in PREFIX_KEYS:
        if len(s[key]) != Q:
            print(f"  ERROR: Sample {i} {key}: len={len(s[key])} != Q={Q}")
            errors += 1
assert errors == 0, f"{errors} prefix length mismatches!"

# Show examples
for i in range(3):
    Q = samples[i]['Q']
    print(f"\nSample {i}: Q={Q}, query='{samples[i]['query'][:50]}...'")
    for key in PREFIX_KEYS:
        label = key.replace('prefix_', '')
        decoded = tokenizer.decode(samples[i][key][:8])
        print(f"  {label:<22}: {decoded}...")

print(f"\nAll {len(PREFIX_KEYS)} prefix types verified for {len(samples)} samples.")
""")


# ===== Cell 5: Scoring functions + validation =====
code(r"""# Cell 5: Scoring functions with BOS-retained repositioning + validation

# --- RoPE repositioning helpers ---
layer_types = getattr(text_cfg, 'layer_types', [])

def build_layer_inv_freqs():
    # Build per-layer-type inverse frequency tensors for RoPE rotation.
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


def select_kv_cache(cache, indices):
    # Select specific cache indices (e.g., BOS + doc, skipping prefix).
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def reposition_kv_cache(cache, old_positions, new_positions, bos_start=0):
    # Reposition doc keys from old_positions to new_positions via RoPE rotation.
    # BOS entry at bos_start is left untouched. Doc entries start at bos_start+1.
    delta = new_positions - old_positions
    for L in range(len(cache.layers)):
        lt = layer_types[L]
        inv_freq = LAYER_INV_FREQS[lt]
        k = cache.layers[L].keys
        doc_keys = k[:, :, bos_start + 1:, :]
        freqs = torch.einsum('i,j->ij', delta.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_delta = emb.cos().to(k.dtype).unsqueeze(0).unsqueeze(0)
        sin_delta = emb.sin().to(k.dtype).unsqueeze(0).unsqueeze(0)
        doc_keys_new = doc_keys * cos_delta + rotate_half(doc_keys) * sin_delta
        cache.layers[L].keys = torch.cat([
            k[:, :, :bos_start + 1, :],
            doc_keys_new,
        ], dim=2)
    return cache


def score(doc_text, query_text, answer_text, prefix_token_ids=None):
    # BOS-retained repositioning.
    #
    # If prefix_token_ids provided:
    #   Phase A: [BOS] + prefix_ids(Q) + [\n] + doc_ids(D) at natural positions.
    #   Select BOS(0) + doc(1+Q+NL .. end) from cache.
    #   Reposition doc keys from [1+Q+NL, ..., Q+NL+D] to [1, ..., D].
    #   Cache: 1+D entries (BOS at 0, doc at 1..D).
    #
    # Bare: [BOS + doc] with default positions. Cache: 1+D entries.
    #
    # Phase B: score [\n + query + \n + answer] at positions [D+1, ...]
    #   cache_position auto-generated from cache length (= 1+D = D+1).

    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1536).input_ids
    D = len(doc_ids)

    if prefix_token_ids is not None:
        P = len(prefix_token_ids)
        NL = len(NEWLINE_IDS)

        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

        # Select BOS (index 0) + doc (indices 1+P+NL .. end)
        keep_indices = [0] + list(range(1 + P + NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)

        # Reposition doc keys from natural positions to bare positions
        old_pos = torch.arange(1 + P + NL, 1 + P + NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
    else:
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

    # Cache has 1+D entries. Phase B at D+1.
    phase_b_start = D + 1

    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        del cache
        return 0.0

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    # Phase B: NO explicit cache_position — auto-generated from cache length
    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
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
# VALIDATION TESTS
# ================================================================
print("=" * 70)
print("VALIDATION: BOS-Retained Repositioning with Token-Level Matching")
print("=" * 70)

# TEST 1: Bare two-phase matches single-pass
print("\n--- Test 1: Bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"
doc_ids_t = tokenizer(doc_text_t, add_special_tokens=False).input_ids
D_t = len(doc_ids_t)
query_ids_t = tokenizer("\n" + query_text_t + "\n", add_special_tokens=False).input_ids
answer_ids_t = tokenizer(answer_text_t, add_special_tokens=False).input_ids

# Single-pass reference
full_ids = [BOS_ID] + doc_ids_t + query_ids_t + answer_ids_t
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D_t + len(query_ids_t)
logits_full = out_full.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids_t), :].float()
targets_t = torch.tensor(answer_ids_t, device=DEVICE)
nll_single = -F.log_softmax(logits_full, dim=-1).gather(
    1, targets_t.unsqueeze(1)).squeeze(1).mean().item()
del out_full

# Two-phase bare
nll_bare = score(doc_text_t, query_text_t, answer_text_t)

diff_pct = abs(nll_single - nll_bare) / nll_single * 100
print(f"  Single-pass NLL: {nll_single:.6f}")
print(f"  Two-phase bare:  {nll_bare:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"
print(f"  PASSED — bare matches single-pass within {diff_pct:.2f}%")

# TEST 2: Prefixed scoring runs without error
print("\n--- Test 2: Prefixed scoring runs correctly ---")
s = samples[0]
nll_b = score(s['passage'], s['query'], s['answer'])
nll_o = score(s['passage'], s['query'], s['answer'],
              prefix_token_ids=s['prefix_oracle'])
nll_r = score(s['passage'], s['query'], s['answer'],
              prefix_token_ids=s['prefix_random_tokens'])
print(f"  Bare:          {nll_b:.4f}")
print(f"  Oracle:        {nll_o:.4f}  delta={nll_b - nll_o:+.4f}")
print(f"  Random tokens: {nll_r:.4f}  delta={nll_b - nll_r:+.4f}")
assert 0 < nll_b < 20, f"Bare NLL out of range: {nll_b}"
assert 0 < nll_o < 20, f"Oracle NLL out of range: {nll_o}"
assert 0 < nll_r < 20, f"Random NLL out of range: {nll_r}"
print("  PASSED — all NLLs in valid range")

# TEST 3: Token-matching invariant (all prefixed conds have same Q)
print("\n--- Test 3: Token-matching invariant ---")
Q = s['Q']
for key in PREFIX_KEYS:
    assert len(s[key]) == Q, f"{key}: {len(s[key])} != Q={Q}"
print(f"  All 12 prefixed conditions have Q={Q} tokens for sample 0")
print("  PASSED")

# TEST 4: 5-sample quick check
print("\n--- Test 4: 5-sample bare vs oracle vs random_tokens ---")
for i in range(5):
    st = samples[i]
    nb_ = score(st['passage'], st['query'], st['answer'])
    no_ = score(st['passage'], st['query'], st['answer'],
                prefix_token_ids=st['prefix_oracle'])
    nr_ = score(st['passage'], st['query'], st['answer'],
                prefix_token_ids=st['prefix_random_tokens'])
    print(f"  Sample {i}: bare={nb_:.4f}, oracle={no_:.4f} ({nb_-no_:+.4f}), "
          f"random={nr_:.4f} ({nb_-nr_:+.4f})")

gc.collect()
torch.cuda.empty_cache()
print("\n" + "=" * 70)
print("ALL VALIDATION TESTS PASSED")
print("=" * 70)
""")


# ===== Cell 6: Scoring loop =====
code(r"""# Cell 6: Scoring loop — 13 conditions x 400 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'random_tokens', 'repeat_token', 'scrambled_oracle',
    'unrelated_query', 'same_topic', 'paraphrase', 'oracle',
    'llm_extract', 'llm_question', 'llm_summarize',
    'extractor_matched', 'adversarial_matched',
]

# Map condition name -> prefix key in samples dict
COND_PREFIX_MAP = {
    'random_tokens': 'prefix_random_tokens',
    'repeat_token': 'prefix_repeat_token',
    'scrambled_oracle': 'prefix_scrambled_oracle',
    'unrelated_query': 'prefix_unrelated_query',
    'same_topic': 'prefix_same_topic',
    'paraphrase': 'prefix_paraphrase',
    'oracle': 'prefix_oracle',
    'llm_extract': 'prefix_llm_extract',
    'llm_question': 'prefix_llm_question',
    'llm_summarize': 'prefix_llm_summarize',
    'extractor_matched': 'prefix_extractor_matched',
    'adversarial_matched': 'prefix_adversarial_matched',
}

SCORING_KEY = 'bos_retained_token_matched_v02'

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and ckpt.get('scoring') == SCORING_KEY:
        if len(ckpt.get('results', [])) > 0:
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
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
        'Q': s['Q'],
    }

    # bare — no prefix
    result['nll_bare'] = score(passage, query, answer)

    # All prefixed conditions
    for cond_name, prefix_key in COND_PREFIX_MAP.items():
        result[f'nll_{cond_name}'] = score(
            passage, query, answer,
            prefix_token_ids=s[prefix_key]
        )

    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'scoring': SCORING_KEY,
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


# ===== Cell 7: Results & analysis =====
code(r"""# Cell 7: Results & analysis
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

# Build arrays for all conditions
cond_arrays = {}
for cond in COND_NAMES:
    cond_arrays[cond] = np.array([r[f'nll_{cond}'] for r in results])

bare = cond_arrays['bare']

# ================================================================
# PART 1: Basic condition table
# ================================================================
print(f"\n--- PART 1: Condition Table ---")
print(f"\n  {'Condition':<24} {'NLL':>8} {'Delta':>8} {'d':>8} {'Win%':>7} "
      f"{'p':>12} {'sig':>5}")
print(f"  {'-'*78}")

analysis = {}
for cond in COND_NAMES:
    nlls = cond_arrays[cond]
    mean_nll = nlls.mean()
    if cond == 'bare':
        print(f"  {cond:<24} {mean_nll:>8.4f} {'--':>8} {'--':>8} {'--':>7} "
              f"{'--':>12} {'--':>5}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls  # positive = condition has lower NLL (better)
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {cond:<24} {mean_nll:>8.4f} {diff.mean():>+8.4f} {d:>+8.3f} "
              f"{win_pct:>6.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# ================================================================
# PART 2: Semantic gradient test
# ================================================================
print(f"\n--- PART 2: Semantic Gradient Test ---")
print("Relevance ordering: random_tokens(0) < scrambled(1) < unrelated(2) "
      "< same_topic(3) < paraphrase(4) < oracle(5)")

GRADIENT_CONDS = [
    ('random_tokens', 0),
    ('scrambled_oracle', 1),
    ('unrelated_query', 2),
    ('same_topic', 3),
    ('paraphrase', 4),
    ('oracle', 5),
]

gradient_ranks = []
gradient_ds = []
for cond, rank in GRADIENT_CONDS:
    d = cohens_d(bare - cond_arrays[cond])
    gradient_ranks.append(rank)
    gradient_ds.append(d)
    print(f"  [{rank}] {cond:<22} d={d:+.4f}")

rho, p_mono = stats.spearmanr(gradient_ranks, gradient_ds)
sig_mono = '***' if p_mono < 0.001 else '**' if p_mono < 0.01 else '*' if p_mono < 0.05 else 'ns'
print(f"\n  Spearman rho (relevance rank vs d): rho={rho:+.3f}, p={p_mono:.4f} {sig_mono}")

if rho > 0.8 and p_mono < 0.05:
    print(f"  --> MONOTONIC: clear semantic gradient")
elif rho > 0.5:
    print(f"  --> PARTIAL: imperfect gradient")
else:
    print(f"  --> FLAT: no clear gradient")

# ================================================================
# PART 3: Structural decomposition
# ================================================================
print(f"\n--- PART 3: Structural Decomposition ---")
oracle_d = cohens_d(bare - cond_arrays['oracle'])
random_d = cohens_d(bare - cond_arrays['random_tokens'])
repeat_d = cohens_d(bare - cond_arrays['repeat_token'])

if oracle_d != 0:
    structural_frac_random = random_d / oracle_d * 100
    structural_frac_repeat = repeat_d / oracle_d * 100
else:
    structural_frac_random = structural_frac_repeat = float('nan')

print(f"  Oracle d:         {oracle_d:+.4f}")
print(f"  Random tokens d:  {random_d:+.4f} ({structural_frac_random:.1f}% of oracle)")
print(f"  Repeat token d:   {repeat_d:+.4f} ({structural_frac_repeat:.1f}% of oracle)")

if abs(structural_frac_random) > 80:
    print(f"  --> Structure dominates: random_tokens recovers {structural_frac_random:.0f}% of oracle")
elif abs(structural_frac_random) > 40:
    print(f"  --> Mixed: structure accounts for {structural_frac_random:.0f}%")
else:
    print(f"  --> Semantics dominate: structure only {structural_frac_random:.0f}%")

# ================================================================
# PART 4: LLM surrogates vs fixed task-framing
# ================================================================
print(f"\n--- PART 4: LLM Document-Specific vs Generic Task-Framing ---")

# Paired comparisons
pairs = [
    ('llm_extract', 'extractor_matched', 'doc-specific vs generic extraction'),
    ('llm_question', 'extractor_matched', 'doc question vs generic extraction'),
    ('llm_summarize', 'extractor_matched', 'doc summary vs generic extraction'),
]

for llm_cond, generic_cond, desc in pairs:
    diff = cond_arrays[generic_cond] - cond_arrays[llm_cond]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = 100 * np.mean(diff > 0)
    print(f"  {desc}:")
    print(f"    {llm_cond:<22} d={cohens_d(bare - cond_arrays[llm_cond]):+.4f}")
    print(f"    {generic_cond:<22} d={cohens_d(bare - cond_arrays[generic_cond]):+.4f}")
    print(f"    Paired diff: d={d:+.4f}, p={p:.2e} {sig}, "
          f"LLM wins {win:.1f}%")

# ================================================================
# PART 5: Hardness interaction
# ================================================================
print(f"\n--- PART 5: Hardness Interaction (5 quintiles x 13 conditions) ---")
quintile_bounds = np.percentile(bare, [20, 40, 60, 80])
quintiles = np.digitize(bare, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

# Print header
header = f"  {'Quintile':<12} {'N':>4} {'bare':>8}"
for cond in COND_NAMES[1:]:
    header += f"  {cond[:8]:>8}"
print(header)
print(f"  {'-'*(16 + 10 * len(COND_NAMES))}")

hardness_data = {}
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    row = f"  {q_labels[q]:<12} {n_q:>4} {bare[mask].mean():>8.3f}"
    hardness_data[q_labels[q]] = {}
    for cond in COND_NAMES[1:]:
        delta = (bare[mask] - cond_arrays[cond][mask]).mean()
        d = cohens_d(bare[mask] - cond_arrays[cond][mask])
        row += f"  {d:>+8.3f}"
        hardness_data[q_labels[q]][cond] = float(d)
    print(row)

# Correlation: hardness vs benefit for key conditions
print(f"\n  Hardness-benefit correlations:")
for cond in ['oracle', 'random_tokens', 'extractor_matched', 'llm_extract', 'paraphrase']:
    diff = bare - cond_arrays[cond]
    r_val, p_val = stats.spearmanr(bare, diff)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"    {cond:<22} rho={r_val:+.3f} (p={p_val:.2e}) {sig}")

# ================================================================
# PART 6: Per-sample ranking
# ================================================================
print(f"\n--- PART 6: Per-Sample Ranking ---")

stacked = np.stack([cond_arrays[c] for c in COND_NAMES], axis=1)
best_idx = stacked.argmin(axis=1)

print(f"  {'Condition':<24} {'Best count':>12} {'Best %':>8} {'Mean rank':>10}")
print(f"  {'-'*58}")

ranks = stacked.argsort(axis=1).argsort(axis=1) + 1
mean_ranks = ranks.mean(axis=0)
for ci, cname in enumerate(COND_NAMES):
    count = (best_idx == ci).sum()
    pct = 100 * count / len(best_idx)
    print(f"  {cname:<24} {count:>12} {pct:>7.1f}% {mean_ranks[ci]:>10.2f}")

# ================================================================
# PART 7: Document-specific vs generic — deep dive
# ================================================================
print(f"\n--- PART 7: Document-Specific vs Generic Deep Dive ---")
print("For each sample: is llm_extract better than extractor_matched?")

llm_better = cond_arrays['extractor_matched'] - cond_arrays['llm_extract']
print(f"  llm_extract wins: {100 * np.mean(llm_better > 0):.1f}%")
print(f"  Mean advantage: {llm_better.mean():+.4f}")
print(f"  Cohen's d: {cohens_d(llm_better):+.4f}")

# By quintile
print(f"\n  LLM advantage by hardness quintile:")
for q in range(5):
    mask = quintiles == q
    diff_q = llm_better[mask]
    d_q = cohens_d(diff_q)
    win_q = 100 * np.mean(diff_q > 0)
    print(f"    {q_labels[q]:<12} d={d_q:+.4f}, LLM wins {win_q:.1f}%")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 8: Verdict + save =====
code(r"""# Cell 8: Verdict and save
print("=" * 70)
print("VERDICT — Exp 02: Token-Matched Semantic Probing with LLM Surrogates")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"N: {len(results)} samples (MS MARCO v1.1)")
print(f"Conditions: {len(COND_NAMES)}")

# Key results
oracle_d = cohens_d(bare - cond_arrays['oracle'])
_, p_oracle = stats.ttest_1samp(bare - cond_arrays['oracle'], 0)

print(f"\n--- Key findings ---")
print(f"  1. Oracle (token-matched): d={oracle_d:+.4f} "
      f"({'***' if p_oracle < 0.001 else '**' if p_oracle < 0.01 else '*' if p_oracle < 0.05 else 'ns'})")

print(f"\n  2. Semantic gradient: rho={rho:+.3f} (p={p_mono:.4f})")
if rho > 0.8 and p_mono < 0.05:
    print(f"     -> MONOTONIC: semantic content drives the effect")
elif abs(rho) < 0.3:
    print(f"     -> FLAT: no semantic gradient")
else:
    print(f"     -> PARTIAL: imperfect gradient")

print(f"\n  3. Structural fraction: {structural_frac_random:.1f}% (random_tokens / oracle)")

print(f"\n  4. All conditions ranked by d:")
sorted_conds = sorted(
    [(c, cohens_d(bare - cond_arrays[c])) for c in COND_NAMES if c != 'bare'],
    key=lambda x: x[1], reverse=True
)
for cond, d in sorted_conds:
    _, p = stats.ttest_1samp(bare - cond_arrays[cond], 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"     {cond:<24} d={d:+.4f} ({sig})")

# Conclusions
print(f"\n--- Conclusions ---")
if abs(oracle_d) < 0.05:
    print(f"  Oracle conditioning has negligible effect (d={oracle_d:+.3f}).")
elif oracle_d < -0.1:
    print(f"  Oracle conditioning HURTS (d={oracle_d:+.3f}).")
elif oracle_d > 0.1:
    print(f"  Oracle conditioning HELPS (d={oracle_d:+.3f}).")
else:
    print(f"  Oracle conditioning has weak effect (d={oracle_d:+.3f}).")

best_cond, best_d = sorted_conds[0]
worst_cond, worst_d = sorted_conds[-1]
print(f"  Best condition: {best_cond} (d={best_d:+.3f})")
print(f"  Worst condition: {worst_cond} (d={worst_d:+.3f})")

# Save
final_results = {
    'experiment': 'v4_exp02_token_matched_semantic_probing',
    'model': MODEL_NAME,
    'scoring': 'bos_retained_repositioning_token_matched',
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'n_conditions': len(COND_NAMES),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': analysis,
    'gradient': {
        'spearman_rho': float(rho),
        'spearman_p': float(p_mono),
    },
    'structural_decomposition': {
        'oracle_d': float(oracle_d),
        'random_tokens_d': float(random_d),
        'repeat_token_d': float(repeat_d),
        'structural_frac_random': float(structural_frac_random),
        'structural_frac_repeat': float(structural_frac_repeat),
    },
    'hardness_interaction': hardness_data,
    'per_sample_mean_ranks': {
        cond: float(mean_ranks[ci]) for ci, cond in enumerate(COND_NAMES)
    },
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
out_path = "experiments/decoder_only/02/02_token_matched_semantic.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
