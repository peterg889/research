#!/usr/bin/env python3
# Build Exp 04: Instruction Framing Decomposition.
#
# Decomposes the extractor_matched effect (Exp 03's best prefix) into three
# components: structural (any tokens in cache), vocabulary (instruction token
# multiset vs random), and meaning (coherent instruction vs scrambled).
#
# Tests 8 coherent instructions (4 extraction + 3 non-extraction + 1 minimal)
# against 8 scrambled controls with identical token multisets.
# Loads 5 baseline conditions from Exp 03 (bare, random_tokens, repeat_token,
# extractor_matched, adversarial_matched).
#
# Scoring: BOS-retained repositioning + token-level prefix matching on
# Gemma 3 12B-IT (identical to Exps 02-03).
#
# 20 conditions (15 scored fresh + 5 loaded), hard examples only (top 40%),
# 4 datasets, SEED=42.

import os
import nbformat as nbf

os.makedirs("experiments/decoder_only/04", exist_ok=True)

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
md(r"""# Experiment 04: Instruction Framing Decomposition

## Motivation

Exp 03 found that `extractor_matched` — a generic extraction instruction
("Extract all key data points, facts, entities, and specific attributes from the
following text.") token-matched to Q tokens — is the **best** KV cache prefix
across all 4 datasets (pooled semantic delta d=+0.357, \*\*\*). Oracle HURTS
(d=-0.253), LLM doc-specific surrogates lose to the generic instruction, and
there is no semantic gradient.

**Why does the extraction instruction work?** Five hypotheses:
- **H1**: Extraction framing is specifically beneficial (vs other task framings)
- **H2**: Any coherent task-relevant instruction helps equally
- **H3**: Instruction structure (coherent syntax) matters, content doesn't
- **H4**: Extraction-related token vocabulary activates relevant representations
- **H5**: Short repeated instruction phrases combine structural repetition with semantic content

## Design

### Three-Level Decomposition (key metric)

For each instruction I:
- **Structural** = NLL(bare) - NLL(random_tokens) — effect of any tokens in cache
- **Vocabulary** = NLL(random_tokens) - NLL(scrambled_I) — effect of instruction token vocabulary
- **Meaning** = NLL(scrambled_I) - NLL(coherent_I) — effect of coherent instruction order
- **Total** = structural + vocabulary + meaning = NLL(bare) - NLL(coherent_I)

### Conditions (20 total: 15 scored fresh + 5 loaded from Exp 03)

**8 coherent instructions** (4 extraction + 3 non-extraction + 1 special):

| # | Name | Category | Text |
|---|------|----------|------|
| 1 | `extract_general` | extraction | "Extract all key data points, facts, entities, and specific attributes from the following text." (**LOADED** as `extractor_matched` from Exp 03) |
| 2 | `extract_entities` | extraction | "Identify and list every named entity including people, locations, organizations, and dates mentioned in this text." |
| 3 | `extract_claims` | extraction | "Extract all factual claims, statistics, numerical data, and specific assertions made in this passage." |
| 4 | `extract_qa` | extraction | "Extract information from this text that would help answer questions about its content and meaning." |
| 5 | `comprehend` | non-extraction | "Read and understand the main ideas, arguments, and supporting details presented in the following text." |
| 6 | `generate_qa` | non-extraction | "Generate questions that could be answered using the specific facts and details in this passage." |
| 7 | `classify` | non-extraction | "Determine the subject matter, text type, writing style, and intended audience of this passage." |
| 8 | `extract_minimal` | special | "Extract key facts from the text." (~7 tokens, heavily repeated in Q-token prefix) |

**8 scrambled controls**: For each coherent instruction, permute the Q-length
token-matched prefix (after `make_prefix`) with a per-condition+sample seed.
Same token multiset, destroyed meaning.

**5 loaded from Exp 03**: `bare`, `random_tokens`, `repeat_token`,
`extractor_matched` (= `extract_general`), `adversarial_matched`

### Key Analyses

1. Three-level decomposition for each of 8 instructions, pooled across datasets
2. Extraction vs non-extraction (H1 vs H2): paired test
3. Coherent vs scrambled (H3): per-instruction paired t-test
4. Vocabulary by category (H4): scrambled extraction vs scrambled non-extraction
5. Repetition effect (H5): extract_minimal vs extract_general
6. Cross-dataset consistency and instruction ranking""")


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

RESULTS_DIR = Path("../../../results/decoder_only/exp04")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")

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

print(f"Exp 04: Instruction Framing Decomposition")
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


def make_prefix(token_ids, Q):
    if len(token_ids) >= Q:
        return token_ids[:Q]
    else:
        padded = token_ids * ((Q // max(len(token_ids), 1)) + 1)
        return padded[:Q]


def scramble_prefix(prefix_ids, seed):
    # Permute Q-length prefix token IDs (after make_prefix).
    # Same token multiset, destroyed meaning.
    rng = pyrandom.Random(seed)
    shuffled = list(prefix_ids)
    rng.shuffle(shuffled)
    return shuffled


# --- Instruction definitions ---
INSTRUCTIONS = {
    # 4 extraction instructions
    'extract_general': "Extract all key data points, facts, entities, and specific attributes from the following text.",
    'extract_entities': "Identify and list every named entity including people, locations, organizations, and dates mentioned in this text.",
    'extract_claims': "Extract all factual claims, statistics, numerical data, and specific assertions made in this passage.",
    'extract_qa': "Extract information from this text that would help answer questions about its content and meaning.",
    # 3 non-extraction instructions
    'comprehend': "Read and understand the main ideas, arguments, and supporting details presented in the following text.",
    'generate_qa': "Generate questions that could be answered using the specific facts and details in this passage.",
    'classify': "Determine the subject matter, text type, writing style, and intended audience of this passage.",
    # 1 special (short, heavily repeated)
    'extract_minimal': "Extract key facts from the text.",
}

EXTRACTION_CONDS = ['extract_general', 'extract_entities', 'extract_claims', 'extract_qa']
NON_EXTRACTION_CONDS = ['comprehend', 'generate_qa', 'classify']
ALL_INSTRUCTION_CONDS = list(INSTRUCTIONS.keys())

# Conditions scored fresh in this experiment (exclude extract_general = loaded from Exp 03)
FRESH_COHERENT_CONDS = [c for c in ALL_INSTRUCTION_CONDS if c != 'extract_general']
FRESH_SCRAMBLED_CONDS = [f'scrambled_{c}' for c in ALL_INSTRUCTION_CONDS]
FRESH_COND_NAMES = FRESH_COHERENT_CONDS + FRESH_SCRAMBLED_CONDS

# Conditions loaded from Exp 03
LOADED_COND_NAMES = ['bare', 'random_tokens', 'repeat_token',
                     'extractor_matched', 'adversarial_matched']

# All 20 conditions (for analysis)
ALL_COND_NAMES = LOADED_COND_NAMES + ['extract_general'] + FRESH_COHERENT_CONDS + FRESH_SCRAMBLED_CONDS

SCORING_KEY = 'bos_retained_token_matched_v04'

# Pre-tokenize all instructions
INSTRUCTION_IDS = {}
for name, text in INSTRUCTIONS.items():
    ids = tokenizer(text, add_special_tokens=False).input_ids
    INSTRUCTION_IDS[name] = ids
    print(f"  {name:<20}: {len(ids)} tokens -> '{text[:60]}...'")

print(f"\nSetup complete. Functions defined: score, make_prefix, scramble_prefix")
print(f"Fresh conditions to score: {len(FRESH_COND_NAMES)} "
      f"({len(FRESH_COHERENT_CONDS)} coherent + {len(FRESH_SCRAMBLED_CONDS)} scrambled)")
print(f"Loaded from Exp 03: {len(LOADED_COND_NAMES)}")
print(f"Total conditions: {len(ALL_COND_NAMES)}")
""")


# ===== Cell 3: Dataset loading + Exp 03 baseline loading =====
code(r"""# Cell 3: Reload datasets (for passage text) + load Exp 03 baselines
from datasets import load_dataset

# Per-dataset seeds (same as Exp 03)
DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
}

# ================================================================
# PART 1: Load MS MARCO from Exp 02 checkpoint
# ================================================================
print("=" * 70)
print("LOADING MS MARCO from Exp 02 + Exp 03 baselines")
print("=" * 70)

assert EXP02_DIR.exists(), f"Exp 02 results not found at {EXP02_DIR}"
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
assert exp02_ckpt.get('scoring') == 'bos_retained_token_matched_v02', \
    f"Unexpected scoring key: {exp02_ckpt.get('scoring')}"
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES, \
    f"Expected {N_SAMPLES} results, got {len(exp02_results)}"

# Extract bare NLLs and select hard 40% (same selection as Exp 03)
msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
N_HARD = int(N_SAMPLES * HARD_FRAC)
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])

print(f"MS MARCO: {N_SAMPLES} total, top {HARD_FRAC*100:.0f}% = {N_HARD} hard samples")

# Build hard_nlls with loaded baselines
hard_nlls = {}  # ds_name -> {cond_name: np.array}
hard_metadata = {}

hard_nlls['ms_marco'] = {}
for cond in LOADED_COND_NAMES:
    arr = np.array([exp02_results[i][f'nll_{cond}'] for i in msmarco_hard_idx])
    hard_nlls['ms_marco'][cond] = arr

# Map extract_general = extractor_matched (same instruction text)
hard_nlls['ms_marco']['extract_general'] = hard_nlls['ms_marco']['extractor_matched'].copy()

# Get passage text for MS MARCO hard samples (need to reload dataset)
print("\nReloading MS MARCO v1.1 for passage text...")
ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES:
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
            msmarco_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES]
msmarco_all = [msmarco_candidates[i] for i in indices]
del ds_msmarco, msmarco_candidates
gc.collect()

# Verify alignment with Exp 02 results
for i in range(min(20, N_SAMPLES)):
    assert msmarco_all[i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"
print("MS MARCO alignment verified (first 20 queries match Exp 02)")

# Build hard_samples for MS MARCO
hard_samples = {}
hs_msmarco = []
for idx in msmarco_hard_idx:
    s = dict(msmarco_all[idx])
    s['nll_bare'] = float(msmarco_bare[idx])
    s['original_idx'] = int(idx)
    hs_msmarco.append(s)
hard_samples['ms_marco'] = hs_msmarco

hard_metadata['ms_marco'] = {
    'n_total': N_SAMPLES,
    'n_hard': N_HARD,
    'source': 'exp02_reuse',
    'mean_passage_words': float(np.mean([s['word_count'] for s in hs_msmarco])),
    'mean_query_tokens': float(np.mean([exp02_results[i]['Q'] for i in msmarco_hard_idx])),
    'mean_answer_words': float(np.mean([count_words(s['answer']) for s in hs_msmarco])),
}

del exp02_ckpt, exp02_results
gc.collect()

# ================================================================
# PART 2: Reload 3 new datasets (same seeds as Exp 03)
# ================================================================
all_samples = {}  # ds_name -> list of N_SAMPLES dicts (for query alignment check)

# ---- SQuAD 2.0 ----
print("\n" + "=" * 70)
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
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
all_samples['squad_v2'] = [squad_candidates[i] for i in sq_indices]
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
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
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
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
all_samples['triviaqa'] = [trivia_candidates[i] for i in tr_indices]
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
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
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
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# ================================================================
# PART 3: Load Exp 03 baselines for 3 new datasets
# ================================================================
print("\n" + "=" * 70)
print("LOADING EXP 03 BASELINES for new datasets")
print("=" * 70)

assert EXP03_DIR.exists(), f"Exp 03 results not found at {EXP03_DIR}"

for ds_name in NEW_DATASETS:
    samples_ds = all_samples[ds_name]

    # Load bare NLLs to replicate hard selection
    bare_path = EXP03_DIR / f"bare_{ds_name}.json"
    assert bare_path.exists(), f"Exp 03 bare checkpoint not found: {bare_path}"
    bare_ckpt = json.loads(bare_path.read_text())
    assert bare_ckpt.get('n_total') == N_SAMPLES
    bare_nlls_all = bare_ckpt['bare_nlls']
    assert len(bare_nlls_all) == N_SAMPLES

    # Verify query alignment
    saved_queries = bare_ckpt.get('queries_first50', [])
    current_queries = [s['query'][:50] for s in samples_ds[:len(saved_queries)]]
    assert saved_queries == current_queries, \
        f"{ds_name}: query alignment mismatch with Exp 03 bare checkpoint"
    print(f"\n{ds_name}: query alignment verified with Exp 03 bare checkpoint")

    # Replicate hard selection (same as Exp 03)
    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])

    # Build hard_samples
    hs = []
    for idx in h_idx:
        s = dict(samples_ds[idx])
        s['nll_bare'] = float(bare_arr[idx])
        s['original_idx'] = int(idx)
        hs.append(s)
    hard_samples[ds_name] = hs

    # Load Exp 03 scored conditions for hard samples
    ckpt_path = EXP03_DIR / f"checkpoint_{ds_name}.json"
    assert ckpt_path.exists(), f"Exp 03 checkpoint not found: {ckpt_path}"
    ckpt = json.loads(ckpt_path.read_text())
    exp03_results = ckpt['results']
    assert len(exp03_results) == N_HARD, \
        f"{ds_name}: expected {N_HARD} results, got {len(exp03_results)}"

    # Verify alignment: Exp 03 checkpoint queries match our hard samples
    for j in range(min(10, N_HARD)):
        assert exp03_results[j]['query'][:50] == hs[j]['query'][:50], \
            f"{ds_name}: query mismatch at hard sample {j}"

    # Extract baseline NLLs
    hard_nlls[ds_name] = {}
    hard_nlls[ds_name]['bare'] = bare_arr[h_idx]
    for cond in ['random_tokens', 'repeat_token', 'extractor_matched',
                 'adversarial_matched']:
        hard_nlls[ds_name][cond] = np.array(
            [r[f'nll_{cond}'] for r in exp03_results])

    # Map extract_general = extractor_matched
    hard_nlls[ds_name]['extract_general'] = hard_nlls[ds_name]['extractor_matched'].copy()

    hard_metadata[ds_name] = {
        'n_total': N_SAMPLES,
        'n_hard': N_HARD,
        'source': 'exp03_reuse',
        'mean_passage_words': float(np.mean([s['word_count'] for s in hs])),
        'mean_query_tokens': 0.0,  # filled after tokenization
        'mean_answer_words': float(np.mean([count_words(s['answer']) for s in hs])),
    }

    print(f"  Loaded {N_HARD} hard samples with 5 baseline conditions")
    print(f"  Bare NLL range: {bare_arr[h_idx].min():.4f} - {bare_arr[h_idx].max():.4f}")

del bare_ckpt, ckpt, exp03_results
gc.collect()

# Summary
print("\n" + "=" * 70)
print("Dataset + baseline loading summary:")
for ds_name in DATASET_NAMES:
    n_h = len(hard_samples[ds_name])
    loaded_conds = [c for c in hard_nlls[ds_name].keys()]
    print(f"  {ds_name}: {n_h} hard samples, loaded conditions: {loaded_conds}")
""")


# ===== Cell 4: Prefix construction + validation =====
code(r"""# Cell 4: Build prefixes for 15 fresh conditions + validation
print("=" * 70)
print("PREFIX CONSTRUCTION + VALIDATION")
print("=" * 70)

special_ids = set(tokenizer.all_special_ids)

for ds_name in DATASET_NAMES:
    hs = hard_samples[ds_name]
    n_hard = len(hs)

    print(f"\n--- {ds_name} ({n_hard} hard samples) ---")

    for i, s in enumerate(hs):
        q_ids = tokenizer(s['query'], add_special_tokens=False).input_ids
        Q = len(q_ids)
        s['Q'] = Q

        # Build coherent prefixes (7 fresh + extract_general already loaded)
        for cond_name in FRESH_COHERENT_CONDS:
            instr_ids = INSTRUCTION_IDS[cond_name]
            s[f'prefix_{cond_name}'] = make_prefix(instr_ids, Q)

        # Build scrambled prefixes (8 total, including scrambled_extract_general)
        for cond_name in ALL_INSTRUCTION_CONDS:
            instr_ids = INSTRUCTION_IDS[cond_name]
            coherent_prefix = make_prefix(instr_ids, Q)
            # Seed = hash(condition_name) % 2**31 + sample_index
            seed = hash(cond_name) % (2**31) + i
            s[f'prefix_scrambled_{cond_name}'] = scramble_prefix(coherent_prefix, seed)

    # Update metadata with query token stats
    q_lens = [s['Q'] for s in hs]
    hard_metadata[ds_name]['mean_query_tokens'] = float(np.mean(q_lens))
    print(f"  Q tokens -- mean: {np.mean(q_lens):.1f}, "
          f"median: {np.median(q_lens):.0f}, "
          f"min: {np.min(q_lens)}, max: {np.max(q_lens)}")

    # Verify all fresh prefixes have exactly Q tokens
    errors = 0
    for i, s in enumerate(hs):
        Q = s['Q']
        for cond in FRESH_COND_NAMES:
            prefix = s[f'prefix_{cond}']
            if len(prefix) != Q:
                print(f"  ERROR: Sample {i} {cond}: len={len(prefix)} != Q={Q}")
                errors += 1
    assert errors == 0, f"{ds_name}: {errors} prefix length mismatches!"
    print(f"  All {len(FRESH_COND_NAMES)} fresh prefix types verified for {n_hard} samples")

    # Verify scrambled prefixes have same token multiset as coherent
    multiset_errors = 0
    for i in range(min(5, n_hard)):
        s = hs[i]
        for cond_name in ALL_INSTRUCTION_CONDS:
            coherent = make_prefix(INSTRUCTION_IDS[cond_name], s['Q'])
            scrambled = s[f'prefix_scrambled_{cond_name}']
            if sorted(coherent) != sorted(scrambled):
                print(f"  ERROR: Sample {i} {cond_name}: multiset mismatch")
                multiset_errors += 1
    assert multiset_errors == 0, f"{ds_name}: {multiset_errors} multiset mismatches!"
    print(f"  Token multiset invariant verified (first 5 samples)")

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

# Test 2: Prefixed scoring on first hard sample
print("\n--- Test 2: Prefixed scoring runs correctly ---")
test_ds = DATASET_NAMES[0]
ts = hard_samples[test_ds][0]
nll_b = score(ts['passage'], ts['query'], ts['answer'])
nll_c = score(ts['passage'], ts['query'], ts['answer'],
              prefix_token_ids=ts['prefix_extract_entities'])
nll_s = score(ts['passage'], ts['query'], ts['answer'],
              prefix_token_ids=ts['prefix_scrambled_extract_entities'])
print(f"  [{test_ds}] Bare:              {nll_b:.4f}")
print(f"  [{test_ds}] extract_entities:  {nll_c:.4f}  delta={nll_b - nll_c:+.4f}")
print(f"  [{test_ds}] scrambled:         {nll_s:.4f}  delta={nll_b - nll_s:+.4f}")
assert 0 < nll_b < 20 and 0 < nll_c < 20 and 0 < nll_s < 20
print("  PASSED")

gc.collect()
torch.cuda.empty_cache()
print("\nALL VALIDATION TESTS PASSED")
""")


# ===== Cell 5: Scoring loop =====
code(r"""# Cell 5: Score 15 fresh conditions x ~160 hard samples x 4 datasets
print("=" * 70)
print(f"SCORING — {len(FRESH_COND_NAMES)} fresh conditions x ~{N_HARD} hard x "
      f"{len(DATASET_NAMES)} datasets")
print("=" * 70)

for ds_name in DATASET_NAMES:
    hs = hard_samples[ds_name]
    n_hard = len(hs)
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"

    print(f"\n--- {ds_name} ({n_hard} hard samples x {len(FRESH_COND_NAMES)} conditions) ---")

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
            }

            # Score all fresh conditions
            for cond_name in FRESH_COND_NAMES:
                result[f'nll_{cond_name}'] = score(
                    s['passage'], s['query'], s['answer'],
                    prefix_token_ids=s[f'prefix_{cond_name}']
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

    # Populate hard_nlls with fresh conditions
    for cond in FRESH_COND_NAMES:
        hard_nlls[ds_name][cond] = np.array(
            [r[f'nll_{cond}'] for r in ds_results])

gc.collect()
torch.cuda.empty_cache()

# Validation: extract_general (loaded) should be very close to extractor_matched
print("\n" + "=" * 70)
print("VALIDATION: extract_general loaded vs extractor_matched loaded")
for ds_name in DATASET_NAMES:
    eg = hard_nlls[ds_name]['extract_general']
    em = hard_nlls[ds_name]['extractor_matched']
    max_diff = np.abs(eg - em).max()
    mean_diff = np.abs(eg - em).mean()
    print(f"  {ds_name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert np.allclose(eg, em, atol=0.01), \
        f"{ds_name}: extract_general != extractor_matched (max diff {max_diff:.6f})"

print("\nAll scoring complete. Datasets in hard_nlls:")
for ds_name in DATASET_NAMES:
    n = len(hard_nlls[ds_name]['bare'])
    n_conds = len(hard_nlls[ds_name])
    print(f"  {ds_name}: {n} hard samples x {n_conds} conditions")
""")


# ===== Cell 6: Per-dataset results =====
code(r"""# Cell 6: Per-dataset analysis — condition tables, three-level decomposition
print("=" * 70)
print("PER-DATASET ANALYSIS")
print("=" * 70)

per_dataset_analysis = {}

for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    n_hard = len(nlls['bare'])
    bare = nlls['bare']
    random_base = nlls['random_tokens']

    print(f"\n{'='*70}")
    print(f"  {ds_name.upper()} -- {n_hard} hard samples")
    print(f"{'='*70}")

    analysis = {}

    # ---- Part A: Condition table (all 20 conditions) ----
    print(f"\n  {'Cond':<28} {'NLL':>7} {'d bare':>8} {'sem d':>8} "
          f"{'win%':>6} {'p':>10} {'sig':>4}")
    print(f"  {'-'*76}")

    for cond in ALL_COND_NAMES:
        if cond not in nlls:
            continue
        c_nlls = nlls[cond]
        mean_nll = c_nlls.mean()

        if cond == 'bare':
            print(f"  {cond:<28} {mean_nll:>7.3f} {'--':>8} {'--':>8} "
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
            print(f"  {cond:<28} {mean_nll:>7.3f} {d_bare:>+8.3f} {'(ref)':>8} "
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
            print(f"  {cond:<28} {mean_nll:>7.3f} {d_bare:>+8.3f} {d_sem:>+8.3f} "
                  f"{win_pct:>5.1f}% {p_sem:>10.2e} {sig:>4}")
            analysis[cond] = {
                'mean_nll': float(mean_nll), 'd_bare': float(d_bare),
                'semantic_delta_d': float(d_sem), 'p_semantic': float(p_sem),
            }

    # ---- Part B: Three-level decomposition ----
    print(f"\n  Three-Level Decomposition (NLL deltas, positive = better):")
    print(f"  {'Instruction':<22} {'Structural':>10} {'Vocab':>10} "
          f"{'Meaning':>10} {'Total':>10}")
    print(f"  {'-'*66}")

    structural = (bare - random_base).mean()

    decomp = {}
    for cond_name in ALL_INSTRUCTION_CONDS:
        coherent_key = cond_name
        scrambled_key = f'scrambled_{cond_name}'

        if coherent_key not in nlls or scrambled_key not in nlls:
            continue

        vocab = (random_base - nlls[scrambled_key]).mean()
        meaning = (nlls[scrambled_key] - nlls[coherent_key]).mean()
        total = structural + vocab + meaning

        print(f"  {cond_name:<22} {structural:>+10.4f} {vocab:>+10.4f} "
              f"{meaning:>+10.4f} {total:>+10.4f}")

        decomp[cond_name] = {
            'structural': float(structural),
            'vocabulary': float(vocab),
            'meaning': float(meaning),
            'total': float(total),
        }

    analysis['decomposition'] = decomp

    # ---- Part C: Coherent vs scrambled paired tests ----
    print(f"\n  Coherent vs Scrambled (paired t-test, positive = coherent better):")
    print(f"  {'Instruction':<22} {'d':>8} {'p':>10} {'sig':>4} {'coherent wins':>14}")
    print(f"  {'-'*62}")

    for cond_name in ALL_INSTRUCTION_CONDS:
        coherent_key = cond_name
        scrambled_key = f'scrambled_{cond_name}'
        if coherent_key not in nlls or scrambled_key not in nlls:
            continue
        diff = nlls[scrambled_key] - nlls[coherent_key]  # pos = coherent better
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        win_pct = 100 * np.mean(diff > 0)
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')
        print(f"  {cond_name:<22} {d:>+8.3f} {p:>10.2e} {sig:>4} {win_pct:>13.1f}%")

    per_dataset_analysis[ds_name] = analysis

gc.collect()
torch.cuda.empty_cache()
""")


# ===== Cell 7: Cross-dataset meta-analysis =====
code(r"""# Cell 7: Cross-dataset meta-analysis, hypothesis verdicts
print("=" * 70)
print("CROSS-DATASET META-ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Pooled three-level decomposition
# ================================================================
print(f"\n--- PART 1: Pooled Three-Level Decomposition ---")
print(f"\n  {'Instruction':<22} {'Structural':>10} {'Vocab':>10} "
      f"{'Meaning':>10} {'Total':>10}")
print(f"  {'-'*66}")

pooled_decomp = {}
for cond_name in ALL_INSTRUCTION_CONDS:
    struct_vals, vocab_vals, meaning_vals, total_vals = [], [], [], []
    for ds_name in DATASET_NAMES:
        nlls = hard_nlls[ds_name]
        bare = nlls['bare']
        random_base = nlls['random_tokens']
        coherent_key = cond_name
        scrambled_key = f'scrambled_{cond_name}'
        if coherent_key not in nlls or scrambled_key not in nlls:
            continue
        struct_vals.append((bare - random_base).mean())
        vocab_vals.append((random_base - nlls[scrambled_key]).mean())
        meaning_vals.append((nlls[scrambled_key] - nlls[coherent_key]).mean())
        total_vals.append((bare - nlls[coherent_key]).mean())

    s_mean = np.mean(struct_vals)
    v_mean = np.mean(vocab_vals)
    m_mean = np.mean(meaning_vals)
    t_mean = np.mean(total_vals)
    print(f"  {cond_name:<22} {s_mean:>+10.4f} {v_mean:>+10.4f} "
          f"{m_mean:>+10.4f} {t_mean:>+10.4f}")
    pooled_decomp[cond_name] = {
        'structural': float(s_mean), 'vocabulary': float(v_mean),
        'meaning': float(m_mean), 'total': float(t_mean),
    }

# ================================================================
# PART 2: Fixed-effects meta-analysis (semantic delta d)
# ================================================================
print(f"\n--- PART 2: Fixed-Effects Meta-Analysis (semantic delta d) ---")

# All scorable conditions (coherent + scrambled, excluding bare/random_tokens)
META_CONDS = (
    ALL_INSTRUCTION_CONDS +
    FRESH_SCRAMBLED_CONDS +
    ['repeat_token', 'adversarial_matched']
)

print(f"\n  {'Condition':<28} {'pooled_d':>9} {'SE':>8} {'z':>8} "
      f"{'p':>10} {'95% CI':>16} {'sig':>4}")
print(f"  {'-'*86}")

meta_results = {}
for cond in META_CONDS:
    ds_effects = []
    for ds_name in DATASET_NAMES:
        nlls = hard_nlls[ds_name]
        if cond not in nlls:
            continue
        sem_delta = nlls['random_tokens'] - nlls[cond]
        n = len(sem_delta)
        d = cohens_d(sem_delta)
        se = np.sqrt(1.0/n + d**2 / (2.0*n))
        ds_effects.append((d, se, n))

    if not ds_effects:
        continue

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

    print(f"  {cond:<28} {pooled_d:>+9.4f} {pooled_se:>8.4f} {z:>+8.2f} "
          f"{p:>10.2e} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {sig:>4}")
    meta_results[cond] = {
        'pooled_d': float(pooled_d), 'se': float(pooled_se),
        'z': float(z), 'p': float(p),
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
    }

# ================================================================
# PART 3: H1 vs H2 — Extraction vs Non-Extraction
# ================================================================
print(f"\n--- PART 3: Extraction vs Non-Extraction (H1 vs H2) ---")

extraction_ds = [meta_results[c]['pooled_d'] for c in EXTRACTION_CONDS if c in meta_results]
non_extraction_ds = [meta_results[c]['pooled_d'] for c in NON_EXTRACTION_CONDS if c in meta_results]
print(f"  Extraction (n={len(extraction_ds)}):     mean pooled_d = {np.mean(extraction_ds):+.4f}")
print(f"  Non-extraction (n={len(non_extraction_ds)}): mean pooled_d = {np.mean(non_extraction_ds):+.4f}")

# Per-sample paired test across all datasets
ext_nlls_pooled = []
non_ext_nlls_pooled = []
for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    random_base = nlls['random_tokens']
    ext_mean = np.mean([random_base - nlls[c] for c in EXTRACTION_CONDS if c in nlls], axis=0)
    non_ext_mean = np.mean([random_base - nlls[c] for c in NON_EXTRACTION_CONDS if c in nlls], axis=0)
    ext_nlls_pooled.append(ext_mean)
    non_ext_nlls_pooled.append(non_ext_mean)

ext_pooled = np.concatenate(ext_nlls_pooled)
non_ext_pooled = np.concatenate(non_ext_nlls_pooled)
diff_ext_vs_non = ext_pooled - non_ext_pooled
d_ext_vs_non = cohens_d(diff_ext_vs_non)
_, p_ext_vs_non = stats.ttest_1samp(diff_ext_vs_non, 0)
sig_ext = ('***' if p_ext_vs_non < 0.001 else '**' if p_ext_vs_non < 0.01
           else '*' if p_ext_vs_non < 0.05 else 'ns')
print(f"  Paired (extraction - non-extraction): d={d_ext_vs_non:+.4f}, "
      f"p={p_ext_vs_non:.2e} {sig_ext}")
if d_ext_vs_non > 0.05 and p_ext_vs_non < 0.05:
    print(f"  --> H1 SUPPORTED: extraction framing is specifically beneficial")
else:
    print(f"  --> H1 NOT SUPPORTED: extraction framing is not uniquely better")

# ================================================================
# PART 4: H3 — Coherent vs Scrambled (meaning effect)
# ================================================================
print(f"\n--- PART 4: Coherent vs Scrambled -- Meaning Effect (H3) ---")
print(f"  {'Instruction':<22} {'d(meaning)':>12} {'p':>10} {'sig':>4} {'consistent':>10}")
print(f"  {'-'*62}")

meaning_tests = {}
for cond_name in ALL_INSTRUCTION_CONDS:
    per_ds_d = []
    all_diffs = []
    for ds_name in DATASET_NAMES:
        nlls = hard_nlls[ds_name]
        coherent_key = cond_name
        scrambled_key = f'scrambled_{cond_name}'
        if coherent_key not in nlls or scrambled_key not in nlls:
            continue
        diff = nlls[scrambled_key] - nlls[coherent_key]
        per_ds_d.append(cohens_d(diff))
        all_diffs.append(diff)

    pooled_diff = np.concatenate(all_diffs)
    d = cohens_d(pooled_diff)
    _, p = stats.ttest_1samp(pooled_diff, 0)
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')
    consistent = all(x > 0 for x in per_ds_d) or all(x <= 0 for x in per_ds_d)
    print(f"  {cond_name:<22} {d:>+12.4f} {p:>10.2e} {sig:>4} "
          f"{'YES' if consistent else 'NO':>10}")
    meaning_tests[cond_name] = {
        'd': float(d), 'p': float(p), 'consistent': consistent,
        'per_ds_d': [float(x) for x in per_ds_d],
    }

n_sig_meaning = sum(1 for v in meaning_tests.values() if v['p'] < 0.05 and v['d'] > 0)
print(f"\n  {n_sig_meaning}/{len(meaning_tests)} instructions have significant meaning effect")
if n_sig_meaning == 0:
    print(f"  --> H3 SUPPORTED: instruction structure matters, content doesn't")
elif n_sig_meaning <= 2:
    print(f"  --> H3 PARTIALLY SUPPORTED: meaning effect is weak/inconsistent")
else:
    print(f"  --> H3 NOT SUPPORTED: meaning (word order) matters for coherent instructions")

# ================================================================
# PART 5: H4 — Vocabulary by Category
# ================================================================
print(f"\n--- PART 5: Scrambled Extraction vs Scrambled Non-Extraction (H4) ---")

scr_ext_nlls = []
scr_non_nlls = []
for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    random_base = nlls['random_tokens']
    ext_scr = [random_base - nlls[f'scrambled_{c}'] for c in EXTRACTION_CONDS
               if f'scrambled_{c}' in nlls]
    non_scr = [random_base - nlls[f'scrambled_{c}'] for c in NON_EXTRACTION_CONDS
               if f'scrambled_{c}' in nlls]
    if ext_scr and non_scr:
        scr_ext_nlls.append(np.mean(ext_scr, axis=0))
        scr_non_nlls.append(np.mean(non_scr, axis=0))

scr_ext_pooled = np.concatenate(scr_ext_nlls)
scr_non_pooled = np.concatenate(scr_non_nlls)
diff_scr = scr_ext_pooled - scr_non_pooled
d_scr = cohens_d(diff_scr)
_, p_scr = stats.ttest_1samp(diff_scr, 0)
sig_scr = ('***' if p_scr < 0.001 else '**' if p_scr < 0.01
           else '*' if p_scr < 0.05 else 'ns')

print(f"  Scrambled extraction mean sem delta: {np.mean(scr_ext_pooled):+.4f}")
print(f"  Scrambled non-extraction mean sem delta: {np.mean(scr_non_pooled):+.4f}")
print(f"  Paired diff: d={d_scr:+.4f}, p={p_scr:.2e} {sig_scr}")
if d_scr > 0.05 and p_scr < 0.05:
    print(f"  --> H4 SUPPORTED: extraction vocabulary alone activates relevant representations")
else:
    print(f"  --> H4 NOT SUPPORTED: vocabulary category doesn't matter when scrambled")

# ================================================================
# PART 6: H5 — Repetition effect
# ================================================================
print(f"\n--- PART 6: Repetition Effect (H5) ---")
print(f"  extract_minimal (~7 tokens, heavily repeated) vs extract_general (~18 tokens)")

for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    if 'extract_minimal' in nlls and 'extract_general' in nlls:
        diff = nlls['extract_general'] - nlls['extract_minimal']
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')
        win = 100 * np.mean(diff > 0)
        print(f"  {ds_name:<16}: extract_minimal wins {win:.1f}%, d={d:+.4f}, p={p:.2e} {sig}")

# Pooled
min_diffs = []
for ds_name in DATASET_NAMES:
    nlls = hard_nlls[ds_name]
    if 'extract_minimal' in nlls and 'extract_general' in nlls:
        min_diffs.append(nlls['extract_general'] - nlls['extract_minimal'])
min_pooled = np.concatenate(min_diffs)
d_min = cohens_d(min_pooled)
_, p_min = stats.ttest_1samp(min_pooled, 0)
sig_min = ('***' if p_min < 0.001 else '**' if p_min < 0.01
           else '*' if p_min < 0.05 else 'ns')
print(f"\n  Pooled: extract_minimal vs extract_general: d={d_min:+.4f}, p={p_min:.2e} {sig_min}")
if d_min > 0.05 and p_min < 0.05:
    print(f"  --> H5 SUPPORTED: short repeated instruction outperforms longer one")
elif d_min < -0.05 and p_min < 0.05:
    print(f"  --> H5 REFUTED: longer instruction outperforms short repeated one")
else:
    print(f"  --> H5 INCONCLUSIVE: no significant difference")

# ================================================================
# PART 7: Cross-dataset consistency
# ================================================================
print(f"\n--- PART 7: Cross-Dataset Consistency ---")
print(f"\n  {'Condition':<28}", end="")
for ds_name in DATASET_NAMES:
    print(f" {ds_name[:10]:>10}", end="")
print(f"  {'mean':>8} {'consistent':>10}")
print(f"  {'-'*90}")

cross_dataset = {}
for cond in META_CONDS:
    ds_vals = []
    for ds_name in DATASET_NAMES:
        nlls = hard_nlls[ds_name]
        if cond not in nlls:
            ds_vals.append(float('nan'))
            continue
        sem_delta = nlls['random_tokens'] - nlls[cond]
        d = cohens_d(sem_delta)
        ds_vals.append(d)

    valid = [v for v in ds_vals if not np.isnan(v)]
    if not valid:
        continue
    mean_d = np.mean(valid)
    same_sign = all(v >= 0 for v in valid) or all(v <= 0 for v in valid)
    row = f"  {cond:<28}"
    for v in ds_vals:
        row += f" {v:>+10.3f}" if not np.isnan(v) else f" {'N/A':>10}"
    row += f"  {mean_d:>+8.3f} {'YES' if same_sign else 'NO':>10}"
    print(row)
    cross_dataset[cond] = {
        ds: float(d) for ds, d in zip(DATASET_NAMES, ds_vals)
    }
    cross_dataset[cond]['mean'] = float(mean_d)
    cross_dataset[cond]['consistent_sign'] = same_sign

# ================================================================
# PART 8: Instruction ranking
# ================================================================
print(f"\n--- PART 8: Instruction Ranking (pooled semantic delta d) ---")
ranked = sorted(meta_results.items(), key=lambda x: x[1]['pooled_d'], reverse=True)
print(f"\n  {'Rank':>4} {'Condition':<28} {'pooled_d':>9} {'sig':>4}")
print(f"  {'-'*50}")
for rank, (cond, m) in enumerate(ranked, 1):
    sig = ('***' if m['p'] < 0.001 else '**' if m['p'] < 0.01
           else '*' if m['p'] < 0.05 else 'ns')
    print(f"  {rank:>4} {cond:<28} {m['pooled_d']:>+9.4f} {sig:>4}")

# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 70)
print("VERDICT -- Exp 04: Instruction Framing Decomposition")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Scoring: BOS-retained repositioning + token-level prefix matching")
print(f"Datasets: {len(DATASET_NAMES)} ({', '.join(DATASET_NAMES)})")
print(f"Hard selection: top {HARD_FRAC*100:.0f}% by bare NLL")

print(f"\n--- Hypothesis Verdicts ---")

# H1
print(f"\n  H1 (Extraction framing is specifically beneficial):")
print(f"    Extraction mean pooled_d: {np.mean(extraction_ds):+.4f}")
print(f"    Non-extraction mean pooled_d: {np.mean(non_extraction_ds):+.4f}")
print(f"    Paired diff d={d_ext_vs_non:+.4f}, p={p_ext_vs_non:.2e}")

# H2
print(f"\n  H2 (Any coherent task-relevant instruction helps equally):")
all_coherent_ds = [meta_results[c]['pooled_d'] for c in ALL_INSTRUCTION_CONDS if c in meta_results]
print(f"    All coherent instructions mean pooled_d: {np.mean(all_coherent_ds):+.4f}")
print(f"    Range: [{min(all_coherent_ds):+.4f}, {max(all_coherent_ds):+.4f}]")
n_sig_pos = sum(1 for c in ALL_INSTRUCTION_CONDS
                if c in meta_results and meta_results[c]['pooled_d'] > 0 and meta_results[c]['p'] < 0.05)
print(f"    {n_sig_pos}/{len(ALL_INSTRUCTION_CONDS)} significantly positive")

# H3
print(f"\n  H3 (Structure matters, content doesn't):")
print(f"    {n_sig_meaning}/{len(meaning_tests)} instructions have significant meaning effect")

# H4
print(f"\n  H4 (Extraction vocabulary activates relevant representations):")
print(f"    Scrambled ext vs non-ext: d={d_scr:+.4f}, p={p_scr:.2e}")

# H5
print(f"\n  H5 (Short repeated instruction combines repetition + semantics):")
print(f"    extract_minimal vs extract_general: d={d_min:+.4f}, p={p_min:.2e}")

# Summary
print(f"\n--- Key decomposition finding ---")
mean_struct = np.mean([pooled_decomp[c]['structural'] for c in ALL_INSTRUCTION_CONDS])
mean_vocab = np.mean([pooled_decomp[c]['vocabulary'] for c in ALL_INSTRUCTION_CONDS])
mean_meaning = np.mean([pooled_decomp[c]['meaning'] for c in ALL_INSTRUCTION_CONDS])
mean_total = np.mean([pooled_decomp[c]['total'] for c in ALL_INSTRUCTION_CONDS])
print(f"  Mean across all 8 instructions (pooled NLL delta):")
print(f"    Structural: {mean_struct:+.4f}")
print(f"    Vocabulary:  {mean_vocab:+.4f}")
print(f"    Meaning:     {mean_meaning:+.4f}")
print(f"    Total:       {mean_total:+.4f}")
if abs(mean_total) > 0.001:
    pct_struct = mean_struct / mean_total * 100
    pct_vocab = mean_vocab / mean_total * 100
    pct_meaning = mean_meaning / mean_total * 100
    print(f"    Decomposition: {pct_struct:.0f}% structural, "
          f"{pct_vocab:.0f}% vocabulary, {pct_meaning:.0f}% meaning")
""")


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save results
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

final_results = {
    'experiment': 'v4_exp04_instruction_framing_decomposition',
    'model': MODEL_NAME,
    'scoring': 'bos_retained_repositioning_token_matched',
    'hard_fraction': HARD_FRAC,
    'datasets': DATASET_NAMES,
    'n_samples_per_dataset': N_SAMPLES,
    'n_hard_per_dataset': N_HARD,
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'instructions': {name: text for name, text in INSTRUCTIONS.items()},
    'conditions': {
        'loaded_from_exp03': LOADED_COND_NAMES,
        'fresh_coherent': FRESH_COHERENT_CONDS,
        'fresh_scrambled': FRESH_SCRAMBLED_CONDS,
        'all': ALL_COND_NAMES,
        'extraction': EXTRACTION_CONDS,
        'non_extraction': NON_EXTRACTION_CONDS,
    },
    'per_dataset': {ds: per_dataset_analysis.get(ds, {}) for ds in DATASET_NAMES},
    'meta_analysis': meta_results,
    'pooled_decomposition': pooled_decomp,
    'cross_dataset': cross_dataset,
    'meaning_tests': meaning_tests,
    'hypotheses': {
        'H1_extraction_vs_non': {
            'd': float(d_ext_vs_non),
            'p': float(p_ext_vs_non),
        },
        'H3_meaning_n_sig': n_sig_meaning,
        'H4_vocab_category': {
            'd': float(d_scr),
            'p': float(p_scr),
        },
        'H5_repetition': {
            'd': float(d_min),
            'p': float(p_min),
        },
    },
    'hard_metadata': {ds: hard_metadata[ds] for ds in DATASET_NAMES},
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'results.json'}")

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
out_path = "experiments/decoder_only/04/04_instruction_framing.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
