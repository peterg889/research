#!/usr/bin/env python3
# Build Exp 03: Hard-Example Semantic Isolation Across Datasets.
#
# Restricts to hard examples (top 40% by bare NLL) where conditioning helps,
# measures semantic delta above structural baseline (condition - random_tokens),
# and tests whether the semantic gradient generalizes across 4 diverse QA datasets.
#
# MS MARCO: reuses Exp 02 results (same model, same scoring approach).
# SQuAD 2.0, TriviaQA, HotpotQA: new scoring in this experiment.
#
# Scoring: BOS-retained repositioning + token-level prefix matching on
# Gemma 3 12B-IT (identical to Exp 02).
#
# 13 conditions, N=400 per dataset, top 40% hard = 160 per dataset, SEED=42.

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/03", exist_ok=True)

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
md(r"""# Experiment 03: Hard-Example Semantic Isolation Across Datasets

## Motivation

Exp 02 (token-matched, 13 conditions, N=400, Gemma 3 12B-IT) confirmed:
- **Oracle HURTS overall** (d=-0.151, p=0.003)
- **No semantic gradient overall** (Spearman rho=-0.43, p=0.40)
- **But in Q5 (hardest 20%)**, ALL conditions help, and there IS a semantic gradient:
  llm_summarize d=+0.493 > llm_extract +0.356 > same_topic +0.334 > paraphrase +0.291
  > random_tokens +0.218 > oracle +0.078
- The semantic effect is **real but masked** by the dominant structural effect in easy samples

## Goal

Isolate the semantic effect by:
1. Restricting to hard examples (top 40% by bare NLL) where conditioning helps
2. Measuring the **semantic delta** above the structural baseline (condition - random_tokens)
3. Testing whether this pattern generalizes across 4 diverse QA datasets

## Method — BOS-Retained Repositioning with Token-Level Matching

Identical to Exp 02. Phase A builds KV cache with `[BOS] + prefix_ids(Q) + [\n] + doc_ids(D)`,
selects BOS + doc, repositions doc keys. Phase B scores `[\n + query + \n + answer]` with
cache_position auto-generated from cache length. No look-ahead.

## Datasets (4 total)

| Dataset | Source | Question type | Passage type |
|---------|--------|--------------|--------------|
| MS MARCO | `microsoft/ms_marco` v1.1 | Web search queries | Selected passages (30-300w) |
| SQuAD 2.0 | `rajpurkar/squad_v2` | Factoid questions | Wikipedia paragraphs (30-500w) |
| TriviaQA | `mandarjoshi/trivia_qa` rc.wikipedia | Trivia questions | Wikipedia articles (first 500w) |
| HotpotQA | `hotpotqa/hotpot_qa` distractor | Multi-hop questions | Supporting fact sentences (30-500w) |

## Conditions (13 total, same as Exp 02)

| # | Key | Semantic relevance | Token construction |
|---|-----|-------------------|--------------------|
| 1 | `bare` | baseline | No prefix |
| 2 | `random_tokens` | none | Q random IDs from vocab |
| 3 | `repeat_token` | none (structural) | Token ID 1000 repeated Q times |
| 4 | `scrambled_oracle` | vocab match only | Random permutation of oracle IDs |
| 5 | `unrelated_query` | low | Other sample's query, pad/trunc to Q |
| 6 | `same_topic` | medium | LLM: "Write a question about same topic..." |
| 7 | `paraphrase` | high | LLM: "Rephrase this query differently..." |
| 8 | `oracle` | maximal | Exact query token IDs |
| 9 | `llm_extract` | task-framing (doc) | LLM: "List key facts from this document" |
| 10 | `llm_question` | query-like (doc) | LLM: "What question does this doc answer?" |
| 11 | `llm_summarize` | summary (doc) | LLM: "Summarize in one sentence" |
| 12 | `extractor_matched` | task-framing (generic) | Fixed extraction text |
| 13 | `adversarial_matched` | adversarial | Fixed adversarial text |

## Key metric: Semantic delta

For each condition C: `semantic_delta(C) = NLL(random_tokens) - NLL(C)`

Positive = semantic content helps beyond structural baseline.""")


# ===== Cell 2: Setup + model loading + scoring functions =====
code(r"""# Cell 2: Setup, model loading, and scoring functions
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
from lib.data import count_words

SEED = 42
N_SAMPLES = 400      # per dataset
HARD_FRAC = 0.40     # top 40% by bare NLL
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp03")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")

DATASET_NAMES = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa']
NEW_DATASETS = ['squad_v2', 'triviaqa', 'hotpotqa']

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
# Use actual embedding table size (config vocab_size may include padding rows)
VOCAB_SIZE = model.get_input_embeddings().num_embeddings
cfg_vocab = getattr(text_cfg, 'vocab_size', None)
if cfg_vocab != VOCAB_SIZE:
    print(f"WARNING: config vocab_size={cfg_vocab} != embedding size={VOCAB_SIZE}")
    print(f"Using embedding size {VOCAB_SIZE} for random token generation")
rope_params = getattr(text_cfg, 'rope_parameters', {})
layer_types = getattr(text_cfg, 'layer_types', [])
# Sliding attention layers cache only (sliding_window - 1) entries.
# select_kv_cache uses uniform indices across all layers, so total Phase A
# tokens must not exceed this limit when a prefix is used.
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1  # observed: 1024-1 = 1023 for Gemma 3

print(f"Exp 03: Hard-Example Semantic Isolation Across Datasets")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"N_SAMPLES: {N_SAMPLES} per dataset, HARD_FRAC: {HARD_FRAC}")
print(f"Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Sliding window: {SLIDING_WINDOW}, cache limit: {SLIDING_CACHE_LIMIT}")
print(f"Num layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")

# --- RoPE repositioning helpers ---
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


def select_kv_cache(cache, indices):
    selected = DynamicCache()
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys[:, :, idx_tensor, :]
        v = cache.layers[i].values[:, :, idx_tensor, :]
        selected.update(k, v, i)
    return selected


def reposition_kv_cache(cache, old_positions, new_positions, bos_start=0):
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
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids

    if prefix_token_ids is not None:
        P = len(prefix_token_ids)
        NL = len(NEWLINE_IDS)
        # Truncate doc so total Phase A tokens fit in sliding window cache.
        # Sliding attention layers store only (sliding_window - 1) KV entries;
        # select_kv_cache needs uniform indexing across all layers.
        max_doc = SLIDING_CACHE_LIMIT - 1 - P - NL  # 1 for BOS
        if len(doc_ids) > max_doc:
            doc_ids = doc_ids[:max_doc]
        D = len(doc_ids)
        cond_ids = [BOS_ID] + list(prefix_token_ids) + NEWLINE_IDS + doc_ids
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([cond_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa
        keep_indices = [0] + list(range(1 + P + NL, len(cond_ids)))
        cache = select_kv_cache(cache, keep_indices)
        old_pos = torch.arange(1 + P + NL, 1 + P + NL + D, device=DEVICE)
        new_pos = torch.arange(1, D + 1, device=DEVICE)
        cache = reposition_kv_cache(cache, old_pos, new_pos, bos_start=0)
    else:
        D = len(doc_ids)
        with torch.no_grad():
            pa = model(input_ids=torch.tensor([[BOS_ID] + doc_ids], device=DEVICE),
                       use_cache=True)
        cache = pa.past_key_values
        del pa

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


def generate_text(input_text, prompt_text, max_new_tokens=50):
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
    cleaned = raw_text.strip().split("\n")[0].strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    cleaned = " ".join(cleaned.split()[:20])
    return cleaned


def make_prefix(token_ids, Q):
    if len(token_ids) >= Q:
        return token_ids[:Q]
    else:
        padded = token_ids * ((Q // max(len(token_ids), 1)) + 1)
        return padded[:Q]


# LLM surrogate prompts
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

# Condition definitions
COND_NAMES = [
    'bare', 'random_tokens', 'repeat_token', 'scrambled_oracle',
    'unrelated_query', 'same_topic', 'paraphrase', 'oracle',
    'llm_extract', 'llm_question', 'llm_summarize',
    'extractor_matched', 'adversarial_matched',
]

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

PREFIX_KEYS = list(COND_PREFIX_MAP.values())

# Fixed-text prefixes
EXTRACTOR_TEXT = "Extract all key data points, facts, entities, and specific attributes from the following text."
ADVERSARIAL_TEXT = "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt mixed together."

SCORING_KEY = 'bos_retained_token_matched_v03'

print(f"\nSetup complete. Functions defined: score, generate_text, make_prefix")
print(f"Conditions: {len(COND_NAMES)} ({len(COND_PREFIX_MAP)} prefixed + bare)")
""")


# ===== Cell 3: Dataset loading =====
code(r"""# Cell 3: Load SQuAD 2.0, TriviaQA, HotpotQA (MS MARCO reused from Exp 02)
from datasets import load_dataset

all_samples = {}  # ds_name -> list of 400 sample dicts

# Per-dataset seeds for reproducible sampling
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
}

# ---- SQuAD 2.0 ----
print("=" * 70)
print("Loading SQuAD 2.0 validation...")
ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")

squad_candidates = []
for item in ds_squad:
    answers = item.get('answers', {})
    answer_texts = answers.get('text', [])
    if not answer_texts:
        continue
    passage = item['context']
    query = item['question']
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        squad_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })

print(f"SQuAD 2.0 candidates: {len(squad_candidates)}")
np.random.seed(DS_SEEDS['squad_v2'])
indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
all_samples['squad_v2'] = [squad_candidates[i] for i in indices]
del ds_squad, squad_candidates
gc.collect()

# ---- TriviaQA ----
print("\nLoading TriviaQA rc.wikipedia validation...")
ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")

trivia_candidates = []
for item in ds_trivia:
    entity_pages = item.get('entity_pages', {})
    wiki_contexts = entity_pages.get('wiki_context', [])
    if not wiki_contexts or not wiki_contexts[0]:
        continue
    # Take first 500 words of first wiki context
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])

    # Check if answer or any alias appears in passage (case-insensitive)
    passage_lower = passage.lower()
    found = answer_val.lower() in passage_lower
    if not found:
        for alias in aliases:
            if alias.lower() in passage_lower:
                found = True
                break
    if not found:
        continue

    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        trivia_candidates.append({
            'passage': passage, 'query': query, 'answer': answer_val,
            'word_count': wc,
        })

print(f"TriviaQA candidates: {len(trivia_candidates)}")
np.random.seed(DS_SEEDS['triviaqa'])
indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
all_samples['triviaqa'] = [trivia_candidates[i] for i in indices]
del ds_trivia, trivia_candidates
gc.collect()

# ---- HotpotQA ----
print("\nLoading HotpotQA distractor validation...")
ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

hotpot_candidates = []
for item in ds_hotpot:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    sf_titles = sf.get('title', [])
    sf_sent_ids = sf.get('sent_id', [])

    # Build title -> sentences mapping
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents

    # Extract supporting fact sentences
    passage_parts = []
    for title, sid in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents and sid < len(title_to_sents[title]):
            passage_parts.append(title_to_sents[title][sid])

    if not passage_parts:
        continue
    passage = ' '.join(passage_parts)
    query = item['question']
    answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        hotpot_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })

print(f"HotpotQA candidates: {len(hotpot_candidates)}")
np.random.seed(DS_SEEDS['hotpotqa'])
indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# Summary
print("\n" + "=" * 70)
print("Dataset loading summary:")
for ds_name in NEW_DATASETS:
    samps = all_samples[ds_name]
    print(f"\n  {ds_name}: {len(samps)} samples")
    print(f"    Mean passage words: {np.mean([s['word_count'] for s in samps]):.0f}")
    print(f"    Mean answer words: {np.mean([count_words(s['answer']) for s in samps]):.0f}")
    print(f"    Mean query words: {np.mean([count_words(s['query']) for s in samps]):.0f}")
    print(f"    Example query: {samps[0]['query'][:70]}...")
    print(f"    Example answer: {samps[0]['answer'][:70]}...")
""")


# ===== Cell 4: MS MARCO reuse from Exp 02 =====
code(r"""# Cell 4: Reuse MS MARCO results from Exp 02
print("=" * 70)
print("Loading MS MARCO results from Exp 02")
print("=" * 70)

assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
assert exp02_ckpt.get('scoring') == 'bos_retained_token_matched_v02', \
    f"Unexpected scoring key: {exp02_ckpt.get('scoring')}"
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES, \
    f"Expected {N_SAMPLES} results, got {len(exp02_results)}"

# Extract bare NLLs and select hard 40%
msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
N_HARD = int(N_SAMPLES * HARD_FRAC)
sorted_idx = np.argsort(msmarco_bare)[::-1]  # descending (hardest first)
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])  # restore original order

print(f"MS MARCO: {N_SAMPLES} total, selecting top {HARD_FRAC*100:.0f}% = {N_HARD} hard samples")
print(f"Bare NLL range: {msmarco_bare.min():.4f} - {msmarco_bare.max():.4f}")
print(f"Hard cutoff (min NLL in hard set): {msmarco_bare[msmarco_hard_idx].min():.4f}")
print(f"Hard samples mean bare NLL: {msmarco_bare[msmarco_hard_idx].mean():.4f}")

# Build hard_nlls for MS MARCO
hard_nlls = {}  # ds_name -> {cond_name: np.array}
hard_metadata = {}  # ds_name -> {n_total, n_hard, ...}

hard_nlls['ms_marco'] = {}
for cond in COND_NAMES:
    arr = np.array([exp02_results[i][f'nll_{cond}'] for i in msmarco_hard_idx])
    hard_nlls['ms_marco'][cond] = arr

hard_metadata['ms_marco'] = {
    'n_total': N_SAMPLES,
    'n_hard': N_HARD,
    'source': 'exp02_reuse',
    'mean_passage_words': float(np.mean([exp02_results[i]['passage_words']
                                          for i in msmarco_hard_idx])),
    'mean_query_tokens': float(np.mean([exp02_results[i]['Q']
                                         for i in msmarco_hard_idx])),
    'mean_answer_words': float(np.mean([count_words(exp02_results[i]['answer'])
                                         for i in msmarco_hard_idx])),
}

# Quick sanity check: bare should match
assert np.allclose(hard_nlls['ms_marco']['bare'],
                   msmarco_bare[msmarco_hard_idx]), "Bare NLL mismatch"

print(f"\nMS MARCO hard samples loaded:")
print(f"  N_hard: {N_HARD}")
print(f"  Mean passage words: {hard_metadata['ms_marco']['mean_passage_words']:.0f}")
print(f"  Mean query tokens: {hard_metadata['ms_marco']['mean_query_tokens']:.0f}")
print(f"  Conditions: {len(COND_NAMES)}")

# Show condition summary for MS MARCO hard set
bare_h = hard_nlls['ms_marco']['bare']
print(f"\n  {'Condition':<24} {'NLL':>8} {'d vs bare':>10} {'sem delta d':>12}")
print(f"  {'-'*58}")
for cond in COND_NAMES:
    nlls_h = hard_nlls['ms_marco'][cond]
    mean_nll = nlls_h.mean()
    if cond == 'bare':
        print(f"  {cond:<24} {mean_nll:>8.4f} {'--':>10} {'--':>12}")
    else:
        d_bare = cohens_d(bare_h - nlls_h)
        if cond == 'random_tokens':
            print(f"  {cond:<24} {mean_nll:>8.4f} {d_bare:>+10.3f} {'(ref)':>12}")
        else:
            sem_delta = hard_nlls['ms_marco']['random_tokens'] - nlls_h
            d_sem = cohens_d(sem_delta)
            print(f"  {cond:<24} {mean_nll:>8.4f} {d_bare:>+10.3f} {d_sem:>+12.3f}")

del exp02_ckpt, exp02_results
gc.collect()
""")


# ===== Cell 5: Bare scoring for new datasets =====
code(r"""# Cell 5: Bare NLL scoring for SQuAD, TriviaQA, HotpotQA — select hard 40%
print("=" * 70)
print("BARE SCORING — 3 new datasets x 400 samples")
print("=" * 70)

hard_indices = {}   # ds_name -> np.array of indices into all_samples
hard_samples = {}   # ds_name -> list of hard sample dicts

for ds_name in NEW_DATASETS:
    print(f"\n--- {ds_name} ({N_SAMPLES} samples) ---")
    samples = all_samples[ds_name]
    bare_ckpt_path = RESULTS_DIR / f"bare_{ds_name}.json"

    bare_nlls = []
    start_idx = 0

    # Try to resume from checkpoint
    if bare_ckpt_path.exists():
        ckpt = json.loads(bare_ckpt_path.read_text())
        if (ckpt.get('n_total') == N_SAMPLES and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('dataset') == ds_name):
            saved_queries = ckpt.get('queries_first50', [])
            current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
            if saved_queries == current_queries:
                bare_nlls = ckpt['bare_nlls']
                start_idx = len(bare_nlls)
                print(f"  Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

    if start_idx < N_SAMPLES:
        t0 = time.time()
        for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx,
                      total=N_SAMPLES, desc=f"Bare {ds_name}"):
            s = samples[i]
            nll = score(s['passage'], s['query'], s['answer'])
            bare_nlls.append(nll)

            if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_total': N_SAMPLES,
                    'scoring': SCORING_KEY,
                    'bare_nlls': bare_nlls,
                    'queries_first50': [s['query'][:50]
                                        for s in samples[:len(bare_nlls)]],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                bare_ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Bare scoring complete in {elapsed/60:.1f} min")

    bare_arr = np.array(bare_nlls)

    # Select hard 40%
    N_HARD = int(N_SAMPLES * HARD_FRAC)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])
    hard_indices[ds_name] = h_idx

    # Build hard_samples with bare NLL attached
    hs = []
    for idx in h_idx:
        s = dict(samples[idx])  # copy
        s['nll_bare'] = bare_arr[idx]
        s['original_idx'] = int(idx)
        hs.append(s)
    hard_samples[ds_name] = hs

    # Initialize hard_nlls with bare
    hard_nlls[ds_name] = {'bare': bare_arr[h_idx]}

    hard_metadata[ds_name] = {
        'n_total': N_SAMPLES,
        'n_hard': N_HARD,
        'source': 'scored',
        'mean_passage_words': float(np.mean([s['word_count'] for s in hs])),
        'mean_query_tokens': 0.0,  # filled in after tokenization
        'mean_answer_words': float(np.mean([count_words(s['answer']) for s in hs])),
    }

    print(f"  Hard cutoff: {bare_arr[h_idx].min():.4f}")
    print(f"  Hard mean bare NLL: {bare_arr[h_idx].mean():.4f}")
    print(f"  Easy mean bare NLL: {np.delete(bare_arr, h_idx).mean():.4f}")

gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("Hard sample selection complete:")
for ds_name in NEW_DATASETS:
    n_h = len(hard_samples[ds_name])
    print(f"  {ds_name}: {n_h} hard samples (mean bare NLL: "
          f"{hard_nlls[ds_name]['bare'].mean():.4f})")
""")


# ===== Cell 6: LLM surrogate generation =====
code(r"""# Cell 6: Generate LLM surrogates for hard samples of 3 new datasets
print("=" * 70)
print("LLM SURROGATE GENERATION — 3 datasets x ~160 hard samples x 5 surrogates")
print("=" * 70)

surrogates_all = {}  # ds_name -> list of surrogates for hard samples

for ds_name in NEW_DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    surr_path = RESULTS_DIR / f"surrogates_{ds_name}.json"

    print(f"\n--- {ds_name} ({n_hard} hard samples) ---")

    surrogates = []
    start_idx = 0

    if surr_path.exists():
        surr_ckpt = json.loads(surr_path.read_text())
        if (surr_ckpt.get('dataset') == ds_name and
            surr_ckpt.get('n_hard') == n_hard):
            saved_queries = [s.get('query', '')[:50]
                             for s in surr_ckpt.get('surrogates', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                surrogates = surr_ckpt['surrogates']
                start_idx = len(surrogates)
                print(f"  Resuming from {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()
        ds_seed_offset = DS_SEEDS[ds_name]

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Gen {ds_name}"):
            s = hs[i]
            entry = {'query': s['query']}

            doc_words = s['passage'].split()[:200]
            doc_input = f"Document:\n{' '.join(doc_words)}"

            torch.manual_seed(ds_seed_offset + i * 10)
            entry['paraphrase'] = generate_text(
                f"Query: {s['query']}", PROMPT_PARAPHRASE
            )

            torch.manual_seed(ds_seed_offset + i * 10 + 1)
            entry['same_topic'] = generate_text(doc_input, PROMPT_SAME_TOPIC)

            torch.manual_seed(ds_seed_offset + i * 10 + 2)
            entry['llm_extract'] = generate_text(doc_input, PROMPT_EXTRACT)

            torch.manual_seed(ds_seed_offset + i * 10 + 3)
            entry['llm_question'] = generate_text(doc_input, PROMPT_QUESTION)

            torch.manual_seed(ds_seed_offset + i * 10 + 4)
            entry['llm_summarize'] = generate_text(doc_input, PROMPT_SUMMARIZE)

            surrogates.append(entry)

            if (i + 1) % 20 == 0 or i == n_hard - 1:
                surr_ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'surrogates': surrogates,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                surr_path.write_text(json.dumps(surr_ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Gen checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Generation complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(surrogates)} cached surrogates")

    surrogates_all[ds_name] = surrogates

    # Show examples
    for j in range(min(2, n_hard)):
        print(f"\n  Sample {j}: query='{surrogates[j]['query'][:50]}'")
        for key in ['paraphrase', 'same_topic', 'llm_extract']:
            print(f"    {key:<15}: {surrogates[j].get(key, 'N/A')[:50]}")

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 7: Prefix construction + validation =====
code(r"""# Cell 7: Build per-sample token-level prefix IDs for hard samples + validation
print("=" * 70)
print("PREFIX CONSTRUCTION + VALIDATION")
print("=" * 70)

# Pre-tokenize fixed texts
extractor_ids = tokenizer(EXTRACTOR_TEXT, add_special_tokens=False).input_ids
adversarial_ids = tokenizer(ADVERSARIAL_TEXT, add_special_tokens=False).input_ids
special_ids = set(tokenizer.all_special_ids)

for ds_name in NEW_DATASETS:
    hs = hard_samples[ds_name]
    surrs = surrogates_all[ds_name]
    n_hard = len(hs)

    print(f"\n--- {ds_name} ({n_hard} hard samples) ---")

    pyrandom.seed(DS_SEEDS[ds_name] + 200)
    np.random.seed(DS_SEEDS[ds_name] + 300)

    for i, s in enumerate(hs):
        surr = surrs[i]
        q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
        Q = len(q_ids)
        s['Q'] = Q

        # 1. oracle: exact query token IDs
        s['prefix_oracle'] = q_ids

        # 2. random_tokens: random vocab IDs (excluding special)
        rand_ids = []
        while len(rand_ids) < Q:
            tid = np.random.randint(0, VOCAB_SIZE)
            if tid not in special_ids:
                rand_ids.append(int(tid))
        s['prefix_random_tokens'] = rand_ids[:Q]

        # 3. repeat_token
        s['prefix_repeat_token'] = [1000] * Q

        # 4. scrambled_oracle
        shuffled = list(q_ids)
        pyrandom.shuffle(shuffled)
        s['prefix_scrambled_oracle'] = shuffled

        # 5. unrelated_query: other hard sample's query
        other_idx = (i + n_hard // 2) % n_hard
        other_q_ids = tokenizer(hs[other_idx]['query'],
                                add_special_tokens=False).input_ids
        s['prefix_unrelated_query'] = make_prefix(other_q_ids, Q)

        # 6. same_topic
        topic_ids = tokenizer(surr['same_topic'],
                              add_special_tokens=False).input_ids
        s['prefix_same_topic'] = make_prefix(topic_ids, Q)

        # 7. paraphrase
        para_ids = tokenizer(surr['paraphrase'],
                             add_special_tokens=False).input_ids
        s['prefix_paraphrase'] = make_prefix(para_ids, Q)

        # 8. llm_extract
        extract_ids = tokenizer(surr['llm_extract'],
                                add_special_tokens=False).input_ids
        s['prefix_llm_extract'] = make_prefix(extract_ids, Q)

        # 9. llm_question
        question_ids = tokenizer(surr['llm_question'],
                                 add_special_tokens=False).input_ids
        s['prefix_llm_question'] = make_prefix(question_ids, Q)

        # 10. llm_summarize
        summarize_ids = tokenizer(surr['llm_summarize'],
                                  add_special_tokens=False).input_ids
        s['prefix_llm_summarize'] = make_prefix(summarize_ids, Q)

        # 11. extractor_matched
        s['prefix_extractor_matched'] = make_prefix(extractor_ids, Q)

        # 12. adversarial_matched
        s['prefix_adversarial_matched'] = make_prefix(adversarial_ids, Q)

    # Update metadata with query token stats
    q_lens = [s['Q'] for s in hs]
    hard_metadata[ds_name]['mean_query_tokens'] = float(np.mean(q_lens))
    print(f"  Q tokens — mean: {np.mean(q_lens):.1f}, "
          f"median: {np.median(q_lens):.0f}, "
          f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

    # Verify all prefixes have exactly Q tokens
    errors = 0
    for i, s in enumerate(hs):
        Q = s['Q']
        for key in PREFIX_KEYS:
            if len(s[key]) != Q:
                print(f"  ERROR: Sample {i} {key}: len={len(s[key])} != Q={Q}")
                errors += 1
    assert errors == 0, f"{ds_name}: {errors} prefix length mismatches!"
    print(f"  All {len(PREFIX_KEYS)} prefix types verified for {n_hard} samples")

# ================================================================
# VALIDATION TESTS
# ================================================================
print("\n" + "=" * 70)
print("VALIDATION TESTS")
print("=" * 70)

# Test 1: Bare two-phase matches single-pass
print("\n--- Test 1: Bare two-phase matches single-pass ---")
doc_text_t = "The cat sat on the mat near the door of the house by the lake"
query_text_t = "Where did the cat sit?"
answer_text_t = "on the mat"
doc_ids_t = tokenizer(doc_text_t, add_special_tokens=False).input_ids
D_t = len(doc_ids_t)
query_ids_t = tokenizer("\n" + query_text_t + "\n", add_special_tokens=False).input_ids
answer_ids_t = tokenizer(answer_text_t, add_special_tokens=False).input_ids

full_ids = [BOS_ID] + doc_ids_t + query_ids_t + answer_ids_t
with torch.no_grad():
    out_full = model(input_ids=torch.tensor([full_ids], device=DEVICE))
n_ctx = 1 + D_t + len(query_ids_t)
logits_full = out_full.logits[0, n_ctx - 1:n_ctx - 1 + len(answer_ids_t), :].float()
targets_t = torch.tensor(answer_ids_t, device=DEVICE)
nll_single = -F.log_softmax(logits_full, dim=-1).gather(
    1, targets_t.unsqueeze(1)).squeeze(1).mean().item()
del out_full

nll_bare = score(doc_text_t, query_text_t, answer_text_t)
diff_pct = abs(nll_single - nll_bare) / nll_single * 100
print(f"  Single-pass NLL: {nll_single:.6f}")
print(f"  Two-phase bare:  {nll_bare:.6f} (diff: {diff_pct:.2f}%)")
assert diff_pct < 1.0, f"Bare doesn't match single-pass: {diff_pct}%"
print(f"  PASSED")

# Test 2: Prefixed scoring on first hard sample from first new dataset
print("\n--- Test 2: Prefixed scoring runs correctly ---")
test_ds = NEW_DATASETS[0]
ts = hard_samples[test_ds][0]
nll_b = score(ts['passage'], ts['query'], ts['answer'])
nll_o = score(ts['passage'], ts['query'], ts['answer'],
              prefix_token_ids=ts['prefix_oracle'])
nll_r = score(ts['passage'], ts['query'], ts['answer'],
              prefix_token_ids=ts['prefix_random_tokens'])
print(f"  [{test_ds}] Bare:          {nll_b:.4f}")
print(f"  [{test_ds}] Oracle:        {nll_o:.4f}  delta={nll_b - nll_o:+.4f}")
print(f"  [{test_ds}] Random tokens: {nll_r:.4f}  delta={nll_b - nll_r:+.4f}")
assert 0 < nll_b < 20 and 0 < nll_o < 20 and 0 < nll_r < 20
print("  PASSED")

# Test 3: Token-matching invariant
print("\n--- Test 3: Token-matching invariant ---")
Q = ts['Q']
for key in PREFIX_KEYS:
    assert len(ts[key]) == Q, f"{key}: {len(ts[key])} != Q={Q}"
print(f"  All 12 prefixed conditions have Q={Q} tokens")
print("  PASSED")

gc.collect()
torch.cuda.empty_cache()
print("\nALL VALIDATION TESTS PASSED")
""")


# ===== Cell 8: Full scoring loop =====
code(r"""# Cell 8: Full 12-condition scoring for hard samples of 3 new datasets
print("=" * 70)
print("FULL SCORING — 12 prefixed conditions x ~160 hard samples x 3 datasets")
print("=" * 70)

for ds_name in NEW_DATASETS:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"

    print(f"\n--- {ds_name} ({n_hard} hard samples x 12 conditions) ---")

    ds_results = []
    start_idx = 0

    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if (ckpt.get('dataset') == ds_name and
            ckpt.get('scoring') == SCORING_KEY and
            ckpt.get('n_hard') == n_hard):
            saved_queries = [r['query'][:50] for r in ckpt.get('results', [])]
            current_queries = [s['query'][:50] for s in hs[:len(saved_queries)]]
            if saved_queries == current_queries:
                ds_results = ckpt['results']
                start_idx = len(ds_results)
                print(f"  Resuming from checkpoint: {start_idx}/{n_hard}")

    if start_idx < n_hard:
        t0 = time.time()

        for i in tqdm(range(start_idx, n_hard), initial=start_idx,
                      total=n_hard, desc=f"Score {ds_name}"):
            s = hs[i]
            result = {
                'query': s['query'],
                'answer': s['answer'],
                'passage_words': s['word_count'],
                'Q': s['Q'],
                'nll_bare': float(s['nll_bare']),
            }

            for cond_name, prefix_key in COND_PREFIX_MAP.items():
                result[f'nll_{cond_name}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=s[prefix_key]
                )

            ds_results.append(result)

            if (i + 1) % 20 == 0 or i == n_hard - 1:
                ckpt = {
                    'dataset': ds_name,
                    'n_hard': n_hard,
                    'scoring': SCORING_KEY,
                    'results': ds_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                ckpt_path.write_text(json.dumps(ckpt))
                elapsed = time.time() - t0
                done = i - start_idx + 1
                eta = (n_hard - i - 1) * elapsed / done if done > 0 else 0
                tqdm.write(f"  Checkpoint {i+1}/{n_hard} | "
                           f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Scoring complete in {elapsed/60:.1f} min")
    else:
        print(f"  Loaded {len(ds_results)} cached results")

    # Populate hard_nlls for this dataset
    for cond in COND_NAMES:
        hard_nlls[ds_name][cond] = np.array(
            [r[f'nll_{cond}'] for r in ds_results])

    # Sanity check: bare NLLs should match
    bare_from_results = np.array([r['nll_bare'] for r in ds_results])
    bare_from_cell5 = hard_nlls[ds_name]['bare']
    assert np.allclose(bare_from_results, bare_from_cell5, atol=0.01), \
        f"{ds_name}: bare NLL mismatch between Cell 5 and Cell 8"
    # Use the fresh bare from Cell 5 (scored directly)
    hard_nlls[ds_name]['bare'] = bare_from_cell5

gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("All scoring complete. Datasets in hard_nlls:")
for ds_name in DATASET_NAMES:
    n = len(hard_nlls[ds_name]['bare'])
    print(f"  {ds_name}: {n} hard samples x {len(COND_NAMES)} conditions")
""")


# ===== Cell 9: Per-dataset results =====
code(r"""# Cell 9: Per-dataset analysis — condition tables, semantic gradient, LLM vs generic
print("=" * 70)
print("PER-DATASET ANALYSIS")
print("=" * 70)

# Relevance ordering for gradient test (excluding random_tokens = reference)
GRADIENT_CONDS = [
    ('scrambled_oracle', 1),
    ('unrelated_query', 2),
    ('same_topic', 3),
    ('paraphrase', 4),
    ('oracle', 5),
]

per_dataset_analysis = {}

for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    n_hard = len(nlls['bare'])
    bare = nlls['bare']
    random_base = nlls['random_tokens']

    print(f"\n{'='*70}")
    print(f"  {ds_name.upper()} — {n_hard} hard samples")
    print(f"{'='*70}")

    analysis = {}

    # ---- Part A: Condition table ----
    print(f"\n  {'Cond':<24} {'NLL':>7} {'d bare':>8} {'sem d':>8} "
          f"{'win%':>6} {'p':>10} {'sig':>4}")
    print(f"  {'-'*72}")

    for cond in COND_NAMES:
        c_nlls = nlls[cond]
        mean_nll = c_nlls.mean()

        if cond == 'bare':
            print(f"  {cond:<24} {mean_nll:>7.3f} {'--':>8} {'--':>8} "
                  f"{'--':>6} {'--':>10} {'--':>4}")
            analysis[cond] = {'mean_nll': float(mean_nll)}
            continue

        diff_bare = bare - c_nlls
        d_bare = cohens_d(diff_bare)
        _, p_bare = stats.ttest_1samp(diff_bare, 0)

        if cond == 'random_tokens':
            win_pct = 100 * np.mean(diff_bare > 0)
            sig = ('***' if p_bare < 0.001 else '**' if p_bare < 0.01
                   else '*' if p_bare < 0.05 else 'ns')
            print(f"  {cond:<24} {mean_nll:>7.3f} {d_bare:>+8.3f} {'(ref)':>8} "
                  f"{win_pct:>5.1f}% {p_bare:>10.2e} {sig:>4}")
            analysis[cond] = {
                'mean_nll': float(mean_nll), 'd_bare': float(d_bare),
                'semantic_delta_d': 0.0, 'p_bare': float(p_bare),
            }
        else:
            sem_delta = random_base - c_nlls
            d_sem = cohens_d(sem_delta)
            _, p_sem = stats.ttest_1samp(sem_delta, 0)
            win_pct = 100 * np.mean(sem_delta > 0)
            sig = ('***' if p_sem < 0.001 else '**' if p_sem < 0.01
                   else '*' if p_sem < 0.05 else 'ns')
            print(f"  {cond:<24} {mean_nll:>7.3f} {d_bare:>+8.3f} {d_sem:>+8.3f} "
                  f"{win_pct:>5.1f}% {p_sem:>10.2e} {sig:>4}")
            analysis[cond] = {
                'mean_nll': float(mean_nll), 'd_bare': float(d_bare),
                'semantic_delta_d': float(d_sem), 'p_semantic': float(p_sem),
            }

    # ---- Part B: Semantic gradient test ----
    print(f"\n  Semantic gradient (within hard examples):")
    grad_ranks = []
    grad_ds = []
    for cond, rank in GRADIENT_CONDS:
        sem_d = cohens_d(random_base - nlls[cond])
        grad_ranks.append(rank)
        grad_ds.append(sem_d)
        print(f"    [{rank}] {cond:<22} sem_delta_d={sem_d:+.4f}")

    rho, p_grad = stats.spearmanr(grad_ranks, grad_ds)
    sig = ('***' if p_grad < 0.001 else '**' if p_grad < 0.01
           else '*' if p_grad < 0.05 else 'ns')
    print(f"    Spearman rho: {rho:+.3f} (p={p_grad:.4f}) {sig}")

    if rho > 0.8 and p_grad < 0.10:
        verdict = "MONOTONIC gradient"
    elif rho > 0.5:
        verdict = "PARTIAL gradient"
    elif rho > 0:
        verdict = "WEAK positive trend"
    else:
        verdict = "NO gradient"
    print(f"    --> {verdict}")
    analysis['gradient'] = {
        'rho': float(rho), 'p': float(p_grad), 'verdict': verdict,
    }

    # ---- Part C: LLM doc-specific vs generic ----
    print(f"\n  LLM doc-specific vs generic task-framing:")
    llm_pairs = [
        ('llm_extract', 'extractor_matched'),
        ('llm_question', 'extractor_matched'),
        ('llm_summarize', 'extractor_matched'),
    ]
    for llm_cond, gen_cond in llm_pairs:
        diff = nlls[gen_cond] - nlls[llm_cond]  # pos = LLM better
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        win = 100 * np.mean(diff > 0)
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')
        print(f"    {llm_cond} vs {gen_cond}: d={d:+.3f}, "
              f"LLM wins {win:.1f}%, p={p:.2e} {sig}")

    per_dataset_analysis[ds_name] = analysis

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 10: Cross-dataset analysis + verdict =====
code(r"""# Cell 10: Cross-dataset meta-analysis, consistency, and verdict
print("=" * 70)
print("CROSS-DATASET ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Cross-dataset condition comparison
# ================================================================
print(f"\n--- PART 1: Semantic delta d across datasets ---")
print(f"\n  {'Condition':<24}", end="")
for ds_name in DATASET_NAMES:
    print(f" {ds_name[:10]:>10}", end="")
print(f"  {'mean':>8} {'consistent':>10}")
print(f"  {'-'*90}")

cross_dataset = {}  # cond -> {ds_name: d, ...}
for cond in COND_NAMES:
    if cond in ('bare', 'random_tokens'):
        continue
    row = f"  {cond:<24}"
    ds_vals = []
    for ds_name in DATASET_NAMES:
        sem_delta = hard_nlls[ds_name]['random_tokens'] - hard_nlls[ds_name][cond]
        d = cohens_d(sem_delta)
        row += f" {d:>+10.3f}"
        ds_vals.append(d)
    mean_d = np.mean(ds_vals)
    same_sign = all(v >= 0 for v in ds_vals) or all(v <= 0 for v in ds_vals)
    row += f"  {mean_d:>+8.3f} {'YES' if same_sign else 'NO':>10}"
    print(row)
    cross_dataset[cond] = {ds: float(d) for ds, d in zip(DATASET_NAMES, ds_vals)}
    cross_dataset[cond]['mean'] = float(mean_d)
    cross_dataset[cond]['consistent_sign'] = same_sign

# ================================================================
# PART 2: Fixed-effects meta-analysis
# ================================================================
print(f"\n--- PART 2: Fixed-Effects Meta-Analysis ---")
print(f"\n  {'Condition':<24} {'pooled_d':>9} {'SE':>8} {'z':>8} "
      f"{'p':>10} {'95% CI':>16} {'sig':>4}")
print(f"  {'-'*82}")

meta_results = {}
for cond in COND_NAMES:
    if cond in ('bare', 'random_tokens'):
        continue

    # Per-dataset estimates
    ds_effects = []
    for ds_name in DATASET_NAMES:
        sem_delta = hard_nlls[ds_name]['random_tokens'] - hard_nlls[ds_name][cond]
        n = len(sem_delta)
        d = cohens_d(sem_delta)
        se = np.sqrt(1.0/n + d**2 / (2.0*n))
        ds_effects.append((d, se, n))

    # Fixed-effects pooling
    weights = [1.0 / (se**2) for _, se, _ in ds_effects]
    w_sum = sum(weights)
    pooled_d = sum(w * d for (d, _, _), w in zip(ds_effects, weights)) / w_sum
    pooled_se = 1.0 / np.sqrt(w_sum)
    z = pooled_d / pooled_se if pooled_se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    ci_lo = pooled_d - 1.96 * pooled_se
    ci_hi = pooled_d + 1.96 * pooled_se
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')

    print(f"  {cond:<24} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
          f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4}")
    meta_results[cond] = {
        'pooled_d': float(pooled_d), 'se': float(pooled_se),
        'z': float(z), 'p': float(p),
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
    }

# ================================================================
# PART 3: Semantic gradient across datasets
# ================================================================
print(f"\n--- PART 3: Semantic Gradient Per Dataset ---")
print(f"\n  {'Dataset':<16} {'rho':>8} {'p':>10} {'verdict':>20}")
print(f"  {'-'*58}")

all_rhos = []
for ds_name in DATASET_NAMES:
    grad_ranks = []
    grad_ds = []
    for cond, rank in GRADIENT_CONDS:
        sem_d = cohens_d(hard_nlls[ds_name]['random_tokens'] -
                         hard_nlls[ds_name][cond])
        grad_ranks.append(rank)
        grad_ds.append(sem_d)
    rho, p = stats.spearmanr(grad_ranks, grad_ds)
    all_rhos.append(rho)
    if rho > 0.8 and p < 0.10:
        verdict = "MONOTONIC"
    elif rho > 0.5:
        verdict = "PARTIAL"
    elif rho > 0:
        verdict = "WEAK"
    else:
        verdict = "NONE"
    print(f"  {ds_name:<16} {rho:>+8.3f} {p:>10.4f} {verdict:>20}")

mean_rho = np.mean(all_rhos)
print(f"\n  Mean rho across datasets: {mean_rho:+.3f}")
n_positive = sum(1 for r in all_rhos if r > 0)
print(f"  Datasets with positive gradient: {n_positive}/{len(DATASET_NAMES)}")

# ================================================================
# PART 4: Dataset properties
# ================================================================
print(f"\n--- PART 4: Dataset Properties ---")
print(f"\n  {'Dataset':<16} {'N_hard':>6} {'pass_w':>8} {'ans_w':>8} "
      f"{'Q_tok':>8} {'bare NLL':>10} {'best_sem_d':>10}")
print(f"  {'-'*70}")

for ds_name in DATASET_NAMES:
    meta = hard_metadata[ds_name]
    bare_mean = hard_nlls[ds_name]['bare'].mean()

    # Find condition with highest semantic delta d
    best_cond = None
    best_d = -999
    for cond in COND_NAMES:
        if cond in ('bare', 'random_tokens'):
            continue
        sem_d = cohens_d(hard_nlls[ds_name]['random_tokens'] -
                         hard_nlls[ds_name][cond])
        if sem_d > best_d:
            best_d = sem_d
            best_cond = cond

    print(f"  {ds_name:<16} {meta['n_hard']:>6} "
          f"{meta['mean_passage_words']:>8.0f} "
          f"{meta['mean_answer_words']:>8.1f} "
          f"{meta['mean_query_tokens']:>8.1f} "
          f"{bare_mean:>10.3f} {best_d:>+10.3f}")

# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 70)
print("VERDICT — Exp 03: Hard-Example Semantic Isolation Across Datasets")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"Datasets: {len(DATASET_NAMES)} ({', '.join(DATASET_NAMES)})")
print(f"Hard selection: top {HARD_FRAC*100:.0f}% by bare NLL")

# Key findings
print(f"\n--- Key findings ---")

# 1. Does the semantic gradient emerge in hard examples?
n_gradient = sum(1 for r in all_rhos if r > 0.5)
print(f"\n  1. Semantic gradient in hard examples:")
print(f"     {n_gradient}/{len(DATASET_NAMES)} datasets show partial/monotonic gradient")
print(f"     Mean Spearman rho: {mean_rho:+.3f}")
if n_gradient >= 3:
    print(f"     --> YES: gradient generalizes across datasets")
elif n_gradient >= 2:
    print(f"     --> PARTIAL: gradient in {n_gradient}/{len(DATASET_NAMES)} datasets")
else:
    print(f"     --> NO: gradient does not consistently emerge")

# 2. Which conditions have significant semantic benefit?
print(f"\n  2. Conditions with significant semantic benefit (pooled p<0.05):")
sig_conds = [(c, m) for c, m in meta_results.items() if m['p'] < 0.05]
sig_conds.sort(key=lambda x: x[1]['pooled_d'], reverse=True)
for cond, m in sig_conds:
    print(f"     {cond:<24} pooled_d={m['pooled_d']:+.3f} "
          f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]")
if not sig_conds:
    print(f"     (none)")

# 3. Cross-dataset consistency
n_consistent = sum(1 for v in cross_dataset.values() if v.get('consistent_sign'))
print(f"\n  3. Cross-dataset sign consistency:")
print(f"     {n_consistent}/{len(cross_dataset)} conditions have same sign across all 4 datasets")

# 4. LLM doc-specific vs generic
print(f"\n  4. LLM doc-specific vs generic (pooled):")
for cond in ['llm_extract', 'llm_question', 'llm_summarize']:
    if cond in meta_results:
        m = meta_results[cond]
        gen_cond = 'extractor_matched'
        gm = meta_results.get(gen_cond, {})
        print(f"     {cond:<22} pooled_d={m['pooled_d']:+.3f}  "
              f"{gen_cond} pooled_d={gm.get('pooled_d', 0):+.3f}")

# Ranked conditions by pooled d
print(f"\n  5. All conditions ranked by pooled semantic delta d:")
ranked = sorted(meta_results.items(), key=lambda x: x[1]['pooled_d'], reverse=True)
for cond, m in ranked:
    sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
           else '*' if m['p'] < 0.05 else 'ns')
    print(f"     {cond:<24} d={m['pooled_d']:+.4f} ({sig})")

# ================================================================
# SAVE RESULTS
# ================================================================
final_results = {
    'experiment': 'v4_exp03_hard_semantic_cross_dataset',
    'model': MODEL_NAME,
    'scoring': 'bos_retained_repositioning_token_matched',
    'hard_fraction': HARD_FRAC,
    'datasets': DATASET_NAMES,
    'n_samples_per_dataset': N_SAMPLES,
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'per_dataset': {},
    'meta_analysis': meta_results,
    'cross_dataset': cross_dataset,
    'gradient': {
        'per_dataset_rho': {ds: float(r) for ds, r in zip(DATASET_NAMES, all_rhos)},
        'mean_rho': float(mean_rho),
    },
    'hard_metadata': {ds: hard_metadata[ds] for ds in DATASET_NAMES},
}

for ds_name in DATASET_NAMES:
    final_results['per_dataset'][ds_name] = per_dataset_analysis.get(ds_name, {})

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
out_path = "experiments/decoder_only/03/03_hard_semantic_cross_dataset.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
