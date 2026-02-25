#!/usr/bin/env python3
# Build Prefix LM Exp 05 notebook: Hero Run — 6 Datasets, 17 Conditions,
# 3 Wrong-Answer Types, Full Evaluation Battery.
#
# 04h proved contrastive evaluation with difficulty stratification reveals hidden
# semantic signal: oracle beats random on hard samples (d=+0.475, ***) even though
# aggregate metrics show 105% structural. Vocabulary bridging (oracle_plus_vocab,
# d=+0.311 vs random) and pointer instructions (d=+0.250) are the only approaches
# that beat the structural baseline.
#
# This hero run scales the finding across 6 datasets, 17 conditions, 3 wrong-answer
# types to produce the definitive comparison.
#
# 17 conditions, 6 datasets x N=500 = 3000 total samples, ~10 hrs total.

import nbformat as nbf

nb = nbf.v4.new_notebook()


DATASET_NAMES_BUILD = ["msmarco", "squad", "neuralbridge", "boolq", "triviaqa", "hotpotqa"]


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 05: Hero Run — 6 Datasets, 17 Conditions, 3 Wrong-Answer Types

## Motivation

Exp 04h proved that new evaluation methodologies reveal signal hidden by mean NLL:
- **Token stratification**: Oracle beats random on hard tokens (d gradient +0.090 Q1->Q4)
- **Contrastive evaluation**: Oracle increases discrimination (d=+0.193, ***),
  concentrated on hard samples (Q4: d=+0.475, ***)
- **Vocabulary bridging**: oracle_plus_vocab (d=+0.311 vs random) and pointer
  instructions (d=+0.250) are the only approaches that beat the structural baseline.

But 04h only tested 5 conditions on MS MARCO. This hero run applies the full evaluation
battery across **17 conditions** (including LLM-generated surrogates, vocabulary bridge
dose-response, and pointer instructions) and **6 datasets** (3000 total samples) with
**3 wrong-answer types** (random, hard, LLM-generated) to produce the definitive
surrogate comparison table.

## Datasets (6 x N=500 = 3000 total)

| Dataset | Source | Doc length | Answer type | Why |
|---------|--------|-----------|-------------|-----|
| MS MARCO v1.1 | `microsoft/ms_marco` val | ~60w | ~20w factoid | Established baseline |
| SQuAD v2 | `rajpurkar/squad_v2` val | ~130w | ~3w extractive | Medium docs, extractive |
| neural-bridge | `neural-bridge/rag-dataset-12000` train | ~600w | ~43w generative | Long docs |
| BoolQ | `google/boolq` val | ~100w | 1 token (yes/no) | Pure discrimination |
| TriviaQA | `trivia_qa` rc subset | ~300-500w | ~2w factoid | Long docs, diverse |
| HotpotQA | `hotpot_qa` distractor | ~500w (10 paragraphs) | ~2w multi-hop | Hardest |

## Conditions (17)

| # | Condition | Category | Description |
|---|-----------|----------|-------------|
| 1 | `no_doc` | control | Single-pass [BOS, query, answer] |
| 2 | `bare` | control | Two-pass, no prefix |
| 3 | `random` | structural | 8 random words from WORD_POOL |
| 4 | `oracle` | semantic | Real query as prefix |
| 5 | `oracle_plus_vocab` | semantic | Query + answer-doc overlap words (up to 10) |
| 6 | `pointer` | semantic | "the answer is about [5 overlap keywords]" |
| 7 | `doc_kw10` | extraction | Top-10 TF keywords from document |
| 8 | `instruct` | extraction | "Identify the key facts:" |
| 9 | `llm_query` | LLM | LLM-generated question about passage |
| 10 | `llm_summary` | LLM | LLM-generated 1-sentence summary |
| 11 | `llm_keywords` | LLM | LLM-generated keyword list |
| 12 | `vocab_bridge_1` | dose-response | 1 answer-doc overlap word |
| 13 | `vocab_bridge_3` | dose-response | 3 overlap words |
| 14 | `vocab_bridge_5` | dose-response | 5 overlap words |
| 15 | `vocab_bridge_10` | dose-response | 10 overlap words |
| 16 | `vocab_bridge_15` | dose-response | 15 overlap words |
| 17 | `vocab_bridge_20` | dose-response | 20 overlap words |

BoolQ: skip conditions 12-17 (degenerate -- "yes"/"no" has no content words). Uses 11 conditions.

## Wrong-Answer Pairing (3 types)

1. **Random negatives**: offset (i+250)%N
2. **Hard negatives**: per-dataset topic-matched (Jaccard, same-passage, yes/no flip)
3. **LLM-generated negatives**: Gemma generates plausible-but-wrong answer per sample

## Evaluation Battery (per dataset)

- A: Standard NLL ranking (d vs bare, win%, p-value, Bonferroni)
- B: Token stratification by difficulty (bare NLL quartiles, d(oracle-random) gradient)
- C: Document dependence (no_doc - bare quartiles)
- D: Contrastive AUC -- random negatives
- E: Contrastive AUC -- hard negatives
- F: Contrastive AUC -- LLM-generated negatives
- G: Targeted NLL (top-quartile hard + doc-dependent tokens only)
- H: Vocabulary bridge dose-response curve
- I: Cross-dataset comparison (Kendall's tau, structural fraction, category means)""")


# ===== Cell 1: Setup =====
code(r"""# Cell 1: Setup
import os
os.umask(0o000)

import sys, json, time, gc, re, copy
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
N_SAMPLES = 500

MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/prefix_lm_exp05")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- All 17 conditions --
ALL_CONDITIONS = [
    "no_doc", "bare", "random", "oracle", "oracle_plus_vocab",
    "pointer", "doc_kw10", "instruct",
    "llm_query", "llm_summary", "llm_keywords",
    "vocab_bridge_1", "vocab_bridge_3", "vocab_bridge_5",
    "vocab_bridge_10", "vocab_bridge_15", "vocab_bridge_20",
]

# BoolQ uses only 11 conditions (skip dose-response -- "yes"/"no" has no content words)
VOCAB_BRIDGE_CONDITIONS = [
    "vocab_bridge_1", "vocab_bridge_3", "vocab_bridge_5",
    "vocab_bridge_10", "vocab_bridge_15", "vocab_bridge_20",
]
BOOLQ_CONDITIONS = [c for c in ALL_CONDITIONS if c not in VOCAB_BRIDGE_CONDITIONS]

TWO_PASS_CONDITIONS = [c for c in ALL_CONDITIONS if c != "no_doc"]

CONDITION_CATEGORIES = {
    "no_doc": "control", "bare": "control",
    "random": "structural", "oracle": "semantic",
    "oracle_plus_vocab": "semantic", "pointer": "semantic",
    "doc_kw10": "extraction", "instruct": "extraction",
    "llm_query": "LLM", "llm_summary": "LLM", "llm_keywords": "LLM",
    "vocab_bridge_1": "dose-response", "vocab_bridge_3": "dose-response",
    "vocab_bridge_5": "dose-response", "vocab_bridge_10": "dose-response",
    "vocab_bridge_15": "dose-response", "vocab_bridge_20": "dose-response",
}

DATASET_NAMES = ["msmarco", "squad", "neuralbridge", "boolq", "triviaqa", "hotpotqa"]

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

WORD_POOL = [
    "computer", "mountain", "hospital", "children", "building", "national",
    "business", "research", "students", "american", "possible", "economic",
    "personal", "together", "products", "services", "actually", "remember",
    "practice", "training", "industry", "complete", "critical", "function",
    "language", "standard", "material", "original", "physical", "security",
    "interest", "problems", "consider", "response", "pressure", "politics",
    "movement", "evidence", "southern", "northern", "exchange", "decision",
    "position", "increase", "describe", "military", "required", "approach",
    "strategy", "customer", "resource", "employee", "audience", "location",
    "property", "cultural", "activity", "strength", "analysis", "powerful",
    "election", "argument", "campaign", "maintain", "question", "behavior",
    "majority", "solution", "software", "consumer", "creative", "reaction",
    "european", "delivery", "organize", "involved", "relative", "learning",
    "positive", "numerous", "familiar", "engineer", "platform", "indicate",
    "previous", "pleasure", "opposite", "magazine", "document", "religion",
    "scenario", "workshop", "minority", "guidance", "estimate", "recently",
    "surprise", "champion", "pleasant", "grateful", "moderate", "boundary",
]

INSTRUCT_PREFIX = "Identify the key facts:"


def content_words(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def jaccard(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def extract_keywords(text, n=10):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    counts = Counter(filtered)
    return " ".join(w for w, _ in counts.most_common(n))


def get_conditions_for_dataset(ds_name):
    if ds_name == "boolq":
        return BOOLQ_CONDITIONS
    return ALL_CONDITIONS


ANSWER_TYPES = ['correct', 'wrong_random', 'wrong_hard', 'wrong_llm']

def get_answer_types_for_dataset(ds_name):
    # BoolQ: skip wrong_llm (natural hard neg is sufficient)
    if ds_name == "boolq":
        return ['correct', 'wrong_random', 'wrong_hard']
    return ANSWER_TYPES


print(f"Prefix LM Exp 05: Hero Run -- 6 Datasets, 17 Conditions, 3 Wrong-Answer Types")
print(f"N: {N_SAMPLES} per dataset, Total: {N_SAMPLES * len(DATASET_NAMES)}")
print(f"Datasets: {DATASET_NAMES}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nAll conditions ({len(ALL_CONDITIONS)}): {ALL_CONDITIONS}")
print(f"BoolQ conditions ({len(BOOLQ_CONDITIONS)}): {BOOLQ_CONDITIONS}")
print(f"Answer types: {ANSWER_TYPES}")
""")


# ===== Cell 2: Load model + tokenizer =====
code(r"""# Cell 2: Load model + tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

print(f"transformers version: {transformers.__version__}")

print(f"Loading {MODEL_NAME}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    token=HF_TOKEN,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"Loaded: {n_params:.1f}B params, {gpu_mem:.1f} GB GPU, {time.time()-t0:.0f}s")
print(f"BOS token id: {tokenizer.bos_token_id}")
""")


# ===== Cell 3: Mask functions + deepcopy sanity check =====
code(r"""# Cell 3: Mask functions + deepcopy sanity check

def make_causal_mask(n, dtype=torch.bfloat16):
    # Standard causal mask: lower triangle = 0, upper triangle = min_val
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((n, n), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)


def make_phase_b_mask(n_s, n_d, n_q, n_a, dtype=torch.bfloat16):
    # Phase B: continuation [query, answer] sees [BOS, doc] but NOT prime (truncation)
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min
    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)
    # Allow attending to all prefix positions
    mask[:, :n_prefix] = 0.0
    # Mask out prime positions (truncation)
    if n_s > 0:
        mask[:, 1:1 + n_s] = min_val
    # Causal mask for continuation tokens among themselves
    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


# --- Mask sanity check ---
print("Mask sanity check: custom causal mask vs default forward...")
test_text = "The quick brown fox jumps over the lazy dog."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
Lt = test_ids.shape[1]

with torch.no_grad():
    out_default = model(input_ids=test_ids)

causal_mask = make_causal_mask(Lt)
causal_dict = make_mask_dict(causal_mask.to(DEVICE))
causal_pos = torch.arange(Lt, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_custom = model(input_ids=test_ids, attention_mask=causal_dict,
                       position_ids=causal_pos)

max_diff = (out_default.logits - out_custom.logits).abs().max().item()
print(f"  Max logit diff: {max_diff:.6f}")
assert max_diff < 0.1, f"FAIL: max_diff={max_diff:.4f}"
print(f"  PASS: Dict-based mask API verified.")

# --- deepcopy sanity check for KV cache reuse ---
print("\nDeepcopy sanity check for Phase A KV cache reuse...")
USE_DEEPCOPY = True

prefix_text = "The capital of France is known worldwide."
prefix_ids_t = tokenizer(prefix_text, return_tensors="pt",
                         add_special_tokens=True).input_ids.to(DEVICE)
n_pf = prefix_ids_t.shape[1]

pf_mask = make_causal_mask(n_pf)
pf_dict = make_mask_dict(pf_mask.to(DEVICE))
pf_pos = torch.arange(n_pf, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_pf = model(input_ids=prefix_ids_t, attention_mask=pf_dict,
                   position_ids=pf_pos, use_cache=True)
past_kv_orig = out_pf.past_key_values

# deepcopy and run Phase B with copy
past_kv_copy = copy.deepcopy(past_kv_orig)

cont_text = "Paris is beautiful."
cont_ids_t = tokenizer(cont_text, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(DEVICE)
n_ct = cont_ids_t.shape[1]

# Phase B mask: full attention to prefix, causal among continuation
min_val = torch.finfo(torch.bfloat16).min
b_mask = torch.full((n_ct, n_pf + n_ct), min_val, dtype=torch.bfloat16)
b_mask[:, :n_pf] = 0.0
b_mask[:, n_pf:] = torch.triu(
    torch.full((n_ct, n_ct), min_val, dtype=torch.bfloat16), diagonal=1
)
b_dict = make_mask_dict(b_mask.unsqueeze(0).unsqueeze(0).to(DEVICE))
b_pos = torch.arange(n_pf, n_pf + n_ct, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_copy = model(input_ids=cont_ids_t, attention_mask=b_dict,
                     position_ids=b_pos, past_key_values=past_kv_copy)

# Now run Phase B with original (should be identical if deepcopy worked)
with torch.no_grad():
    out_orig = model(input_ids=cont_ids_t, attention_mask=b_dict,
                     position_ids=b_pos, past_key_values=past_kv_orig)

dc_diff = (out_copy.logits - out_orig.logits).abs().max().item()
print(f"  Deepcopy vs original Phase B max diff: {dc_diff:.6f}")
if dc_diff < 0.01:
    print(f"  PASS: deepcopy KV reuse verified. Will share Phase A across answer types.")
else:
    print(f"  FAIL: deepcopy mismatch ({dc_diff:.4f}). Falling back to separate Phase A.")
    USE_DEEPCOPY = False

del out_default, out_custom, out_pf, out_copy, out_orig
del past_kv_orig, past_kv_copy
gc.collect(); torch.cuda.empty_cache()

print(f"\nUSE_DEEPCOPY = {USE_DEEPCOPY}")
""")


# ===== Cell 4: Data loading functions =====
code(r"""# Cell 4: Data loading -- 6 dataset loaders + prepare_samples + assign_hard_negatives
from datasets import load_dataset


def load_msmarco(n=N_SAMPLES, seed=SEED):
    print("Loading MS MARCO v1.1 validation...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    all_candidates = []
    for item in ds:
        if len(all_candidates) >= 3 * n:
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
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


def load_squad_v2(n=N_SAMPLES, seed=SEED):
    print("Loading SQuAD v2 validation...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    all_candidates = []
    for row in ds:
        answers_obj = row.get("answers", {})
        answer_texts = answers_obj.get("text", [])
        if not answer_texts or len(answer_texts) == 0:
            continue
        answer = answer_texts[0]
        context = row.get("context", "")
        question = row.get("question", "")
        if not answer or not context or not question:
            continue
        c_words = count_words(context)
        a_words = count_words(answer)
        if c_words >= 30 and a_words >= 2:
            all_candidates.append({
                'passage': context, 'query': question, 'answer': answer,
                'word_count': c_words,
                'passage_key': context[:100],  # for same-passage hard neg grouping
            })
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


def load_neuralbridge(n=N_SAMPLES, seed=SEED):
    print("Loading neural-bridge/rag-dataset-12000...")
    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    all_candidates = []
    for row in ds:
        q = row.get("question", "")
        doc = row.get("context", "")
        answer = row.get("answer", "")
        if not q or not doc or not answer:
            continue
        q_words = count_words(q)
        a_words = count_words(answer)
        if q_words >= 15 and a_words >= 5:
            all_candidates.append({
                'passage': doc, 'query': q, 'answer': answer,
                'word_count': count_words(doc),
            })
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


def load_boolq(n=N_SAMPLES, seed=SEED):
    print("Loading BoolQ validation...")
    ds = load_dataset("google/boolq", split="validation")
    all_candidates = []
    for row in ds:
        passage = row.get("passage", "")
        question = row.get("question", "")
        answer_bool = row.get("answer", None)
        if not passage or not question or answer_bool is None:
            continue
        wc = count_words(passage)
        if wc < 30:
            continue
        answer = "yes" if answer_bool else "no"
        all_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
            'answer_bool': answer_bool,
        })
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


def load_triviaqa(n=N_SAMPLES, seed=SEED):
    print("Loading TriviaQA (rc subset)...")
    ds = load_dataset("trivia_qa", "rc", split="validation")
    all_candidates = []
    for row in ds:
        answer_obj = row.get("answer", {})
        answer = answer_obj.get("value", "")
        if not answer:
            continue
        # Get first wiki context entry
        entity_pages = row.get("entity_pages", {})
        wiki_contexts = entity_pages.get("wiki_context", [])
        if not wiki_contexts:
            continue
        passage = wiki_contexts[0]
        question = row.get("question", "")
        if not passage or not question:
            continue
        wc = count_words(passage)
        a_words = count_words(answer)
        if wc >= 30 and a_words >= 1:
            # Truncate very long passages to ~800 words
            words = passage.split()
            if len(words) > 800:
                passage = " ".join(words[:800])
                wc = 800
            all_candidates.append({
                'passage': passage, 'query': question, 'answer': answer,
                'word_count': wc,
            })
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


def load_hotpotqa(n=N_SAMPLES, seed=SEED):
    print("Loading HotpotQA (distractor)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    all_candidates = []
    for row in ds:
        answer = row.get("answer", "")
        question = row.get("question", "")
        if not answer or not question:
            continue
        # Concatenate all context paragraphs: titles + sentences
        context = row.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        paragraphs = []
        for title, sents in zip(titles, sentences_list):
            para = title + ": " + " ".join(sents)
            paragraphs.append(para)
        passage = " ".join(paragraphs)
        wc = count_words(passage)
        a_words = count_words(answer)
        if wc >= 30 and a_words >= 1:
            # Truncate very long concatenated passages
            words = passage.split()
            if len(words) > 800:
                passage = " ".join(words[:800])
                wc = 800
            all_candidates.append({
                'passage': passage, 'query': question, 'answer': answer,
                'word_count': wc,
            })
    print(f"  Total candidates: {len(all_candidates)}")
    np.random.seed(seed)
    indices = np.random.permutation(len(all_candidates))
    samples = [all_candidates[i] for i in indices[:n]]
    del ds, all_candidates; gc.collect()
    return samples


DATASET_LOADERS = {
    'msmarco': load_msmarco,
    'squad': load_squad_v2,
    'neuralbridge': load_neuralbridge,
    'boolq': load_boolq,
    'triviaqa': load_triviaqa,
    'hotpotqa': load_hotpotqa,
}


def prepare_samples(samples, ds_name, seed=SEED):
    # Add per-sample fields needed for all conditions
    is_boolq = (ds_name == "boolq")
    conditions = get_conditions_for_dataset(ds_name)
    N = len(samples)

    for i, s in enumerate(samples):
        # Random prefix (8 words, deterministic per sample)
        rng = np.random.RandomState(seed + i + 20000)
        words = rng.choice(WORD_POOL, size=8, replace=False)
        s['random_prefix'] = " ".join(words)

        # Query-doc overlap
        q_words = set(content_words(s['query']))
        d_words = set(content_words(s['passage']))
        a_words = set(content_words(s['answer']))
        s['query_doc_overlap'] = jaccard(q_words, d_words)

        # Answer-doc overlap words (for oracle_plus_vocab and vocab_bridge)
        overlap_words = sorted(a_words & d_words)
        if not overlap_words:
            overlap_words = content_words(s['answer'])[:5]
        s['overlap_words'] = overlap_words
        s['n_overlap'] = len(overlap_words)

        # oracle_plus_vocab: query + up to 10 overlap words
        s['answer_vocab'] = " ".join(overlap_words[:10])
        s['oracle_plus_vocab'] = s['query'] + " " + s['answer_vocab']

        # pointer: "the answer is about [5 overlap keywords]"
        pointer_kws = " ".join(overlap_words[:5]) if overlap_words else "unknown"
        s['pointer_prefix'] = f"the answer is about {pointer_kws}"

        # Document keywords (top 10, for doc_kw10)
        s['doc_kw10'] = extract_keywords(s['passage'], n=10)

        # Instruct prefix (static)
        s['instruct_prefix'] = INSTRUCT_PREFIX

        # Vocab bridge conditions (various counts)
        if not is_boolq:
            for n_bridge in [1, 3, 5, 10, 15, 20]:
                key = f'vocab_bridge_{n_bridge}'
                bridge_words = overlap_words[:n_bridge]
                s[key] = " ".join(bridge_words) if bridge_words else "unknown"
                s[f'{key}_actual_n'] = len(bridge_words)

        # Wrong-answer pairing: random offset
        j = (i + 250) % N
        s['wrong_random'] = samples[j]['answer']

        s['answer_wc'] = count_words(s['answer'])

    print(f"  Prepared {N} samples for {ds_name}")
    print(f"  Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
    print(f"  Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
    print(f"  Mean answer words: {np.mean([s['answer_wc'] for s in samples]):.0f}")
    if not is_boolq:
        print(f"  Mean overlap words: {np.mean([s['n_overlap'] for s in samples]):.1f}")
    return samples


def assign_hard_negatives(samples, ds_name):
    # Per-dataset hard negative assignment
    N = len(samples)
    print(f"\n  Assigning hard negatives for {ds_name}...")

    if ds_name == "boolq":
        # BoolQ: yes/no flip (natural hard negative)
        for s in samples:
            s['wrong_hard'] = "no" if s['answer'] == "yes" else "yes"
            s['hard_neg_source'] = 'flip'
        print(f"    BoolQ: yes/no flip for all {N} samples")
        return

    if ds_name == "squad":
        # SQuAD: prefer same-passage sibling, fallback to Jaccard
        # Group by passage_key
        from collections import defaultdict
        passage_groups = defaultdict(list)
        for i, s in enumerate(samples):
            pkey = s.get('passage_key', s['passage'][:100])
            passage_groups[pkey].append(i)

        n_sibling = 0
        n_jaccard = 0
        for i, s in enumerate(samples):
            pkey = s.get('passage_key', s['passage'][:100])
            siblings = [j for j in passage_groups[pkey] if j != i]
            if siblings:
                # Pick sibling with most different answer
                best_j = siblings[0]
                s['wrong_hard'] = samples[best_j]['answer']
                s['hard_neg_source'] = 'sibling'
                n_sibling += 1
            else:
                # Fallback: highest query-Jaccard match
                q_words_i = set(content_words(s['query']))
                best_j, best_jac = -1, -1.0
                for j in range(N):
                    if j == i:
                        continue
                    q_words_j = set(content_words(samples[j]['query']))
                    jac = jaccard(q_words_i, q_words_j)
                    if jac > best_jac:
                        best_jac = jac
                        best_j = j
                s['wrong_hard'] = samples[best_j]['answer']
                s['hard_neg_source'] = 'jaccard'
                n_jaccard += 1
        print(f"    SQuAD: {n_sibling} sibling, {n_jaccard} Jaccard")
        return

    # MS MARCO, neural-bridge, TriviaQA, HotpotQA: highest query-Jaccard match
    q_word_sets = [set(content_words(s['query'])) for s in samples]
    for i, s in enumerate(samples):
        best_j, best_jac = -1, -1.0
        for j in range(N):
            if j == i:
                continue
            jac = jaccard(q_word_sets[i], q_word_sets[j])
            if jac > best_jac:
                best_jac = jac
                best_j = j
        s['wrong_hard'] = samples[best_j]['answer']
        s['hard_neg_source'] = 'jaccard'
        s['hard_neg_jaccard'] = best_jac
    mean_jac = np.mean([s['hard_neg_jaccard'] for s in samples])
    print(f"    {ds_name}: Jaccard matching, mean similarity = {mean_jac:.3f}")


print("Data loading functions defined.")
print("  Loaders:", list(DATASET_LOADERS.keys()))
print("  prepare_samples(), assign_hard_negatives()")
""")


# ===== Cell 5: LLM generation function =====
code(r"""# Cell 5: LLM generation -- 3 surrogates + 1 hard negative, checkpointed per dataset

LLM_PROMPTS = {
    'llm_query': "Read this passage and write a single question that this passage answers.",
    'llm_summary': "Summarize the main point of this passage in one sentence.",
    'llm_keywords': "List the 5 most important keywords from this passage, separated by spaces.",
}

LLM_WRONG_ANSWER_PROMPT = (
    "Given this passage and question, write a plausible but INCORRECT short answer.\n\n"
    "Passage: {passage}\nQuestion: {query}\n\nIncorrect answer:"
)


def generate_llm_fields(samples, ds_name, model, tokenizer, device):
    # Generate llm_query, llm_summary, llm_keywords surrogates + llm wrong answers
    # Checkpoints to JSON every 10 samples
    is_boolq = (ds_name == "boolq")
    surr_keys = list(LLM_PROMPTS.keys())
    all_keys = surr_keys + ([] if is_boolq else ['wrong_llm'])

    surr_path = RESULTS_DIR / f"llm_generated_{ds_name}.json"

    # Load existing if available
    existing = {}
    if surr_path.exists():
        existing = json.loads(surr_path.read_text())
        if existing.get("version") == 2 and len(existing.get(surr_keys[0], [])) >= len(samples):
            print(f"  LLM fields already generated for {ds_name}, loading...")
            for i, s in enumerate(samples):
                for key in all_keys:
                    s[key] = existing[key][i]
            return

    start_idx = len(existing.get(surr_keys[0], []))
    stored = {key: existing.get(key, []) for key in all_keys}

    # Apply previously generated fields
    for i in range(start_idx):
        for key in all_keys:
            samples[i][key] = stored[key][i]

    print(f"  Generating LLM fields for {ds_name}"
          f" (starting from {start_idx}/{len(samples)})...")
    t0 = time.time()

    for i in tqdm(range(start_idx, len(samples)), initial=start_idx,
                  total=len(samples), desc=f"LLM gen ({ds_name})"):
        passage = samples[i]['passage']
        passage_trunc = " ".join(passage.split()[:600])

        # Generate surrogates
        for key, prompt_text in LLM_PROMPTS.items():
            messages = [
                {"role": "user",
                 "content": f"{prompt_text}\n\nPassage: {passage_trunc}"},
            ]
            chat_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(chat_ids, torch.Tensor):
                chat_ids = chat_ids["input_ids"]
            chat_ids = chat_ids.to(device)
            n_input = chat_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=chat_ids, max_new_tokens=64, do_sample=False,
                )

            new_tokens = output_ids[0, n_input:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            first_line = generated.split('\n')[0].strip()
            if not first_line:
                first_line = generated[:200].strip()
            samples[i][key] = first_line
            stored[key].append(first_line)

            del chat_ids, output_ids

        # Generate LLM wrong answer (skip for BoolQ)
        if not is_boolq:
            query = samples[i]['query']
            prompt_wa = LLM_WRONG_ANSWER_PROMPT.format(
                passage=passage_trunc, query=query
            )
            messages = [{"role": "user", "content": prompt_wa}]
            chat_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(chat_ids, torch.Tensor):
                chat_ids = chat_ids["input_ids"]
            chat_ids = chat_ids.to(device)
            n_input = chat_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=chat_ids, max_new_tokens=64, do_sample=False,
                )

            new_tokens = output_ids[0, n_input:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            first_line = generated.split('\n')[0].strip()
            if not first_line:
                first_line = generated[:200].strip()
            samples[i]['wrong_llm'] = first_line
            stored['wrong_llm'].append(first_line)

            del chat_ids, output_ids

        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            surr_data = {
                "version": 2,
                "dataset": ds_name,
                "n_samples": len(stored[surr_keys[0]]),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            for key in all_keys:
                surr_data[key] = stored[key]
            surr_path.write_text(json.dumps(surr_data, indent=2))

        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Generated LLM fields for {len(samples)} samples in {elapsed/60:.1f} min")

    # Print examples
    print(f"\n  --- LLM Examples ({ds_name}) ---")
    for j in range(min(3, len(samples))):
        print(f"  Sample {j}:")
        print(f"    Passage: {samples[j]['passage'][:80]}...")
        for key in all_keys:
            print(f"    {key}: {samples[j][key][:80]}")


print("LLM generation function defined.")
""")


# ===== Cell 6: Scoring function =====
code(r"""# Cell 6: score_sample() -- conditions x answer_types, per-token NLLs, Phase A sharing

def score_sample(model, tokenizer, sample, device, ds_name):
    passage = sample['passage']
    query = sample['query']
    conditions = get_conditions_for_dataset(ds_name)
    answer_types = get_answer_types_for_dataset(ds_name)

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids

    # Tokenize all answers
    answer_id_map = {}
    for at in answer_types:
        if at == 'correct':
            text = sample['answer']
        elif at == 'wrong_random':
            text = sample['wrong_random']
        elif at == 'wrong_hard':
            text = sample['wrong_hard']
        elif at == 'wrong_llm':
            text = sample['wrong_llm']
        else:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=256).input_ids
        if len(ids) == 0:
            return None
        answer_id_map[at] = ids

    # Tokenize all prime variants
    prime_map = {"bare": []}
    for cn in conditions:
        if cn in ("no_doc", "bare"):
            continue
        if cn == "random":
            prime_map[cn] = tokenizer(sample['random_prefix'],
                                      add_special_tokens=False).input_ids
        elif cn == "oracle":
            prime_map[cn] = tokenizer(query, add_special_tokens=False,
                                      truncation=True, max_length=256).input_ids
        elif cn == "oracle_plus_vocab":
            prime_map[cn] = tokenizer(sample['oracle_plus_vocab'],
                                      add_special_tokens=False,
                                      truncation=True, max_length=256).input_ids
        elif cn == "pointer":
            prime_map[cn] = tokenizer(sample['pointer_prefix'],
                                      add_special_tokens=False,
                                      truncation=True, max_length=128).input_ids
        elif cn == "doc_kw10":
            prime_map[cn] = tokenizer(sample['doc_kw10'],
                                      add_special_tokens=False).input_ids
        elif cn == "instruct":
            prime_map[cn] = tokenizer(sample['instruct_prefix'],
                                      add_special_tokens=False).input_ids
        elif cn in ("llm_query", "llm_summary", "llm_keywords"):
            prime_map[cn] = tokenizer(sample[cn], add_special_tokens=False,
                                      truncation=True, max_length=128).input_ids
        elif cn.startswith("vocab_bridge_"):
            prime_map[cn] = tokenizer(sample[cn], add_special_tokens=False,
                                      truncation=True, max_length=128).input_ids

    n_q = len(query_ids)
    n_d = len(doc_ids)

    result = {
        'n_doc': n_d, 'n_query': n_q,
    }
    for at, ids in answer_id_map.items():
        result[f'{at}_answer_ids'] = ids

    two_pass_conds = [c for c in conditions if c not in ("no_doc",)]

    for answer_type, answer_ids in answer_id_map.items():
        n_a = len(answer_ids)
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)

        # --- no_doc: single-pass [BOS, query, answer] ---
        no_doc_tokens = [bos_id] + query_ids + answer_ids
        no_doc_input = torch.tensor([no_doc_tokens], dtype=torch.long, device=device)
        n_total = len(no_doc_tokens)

        no_doc_mask = make_causal_mask(n_total)
        no_doc_dict = make_mask_dict(no_doc_mask.to(device))
        no_doc_pos = torch.arange(n_total, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=no_doc_input, attention_mask=no_doc_dict,
                        position_ids=no_doc_pos)

        answer_logits = out.logits[0, n_q : n_q + n_a, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        result[f'token_nlls_no_doc_{answer_type}'] = token_nlls.cpu().tolist()
        result[f'nll_no_doc_{answer_type}'] = token_nlls.mean().item()

        del out, no_doc_input, no_doc_mask, no_doc_dict
        del answer_logits, log_probs, token_nlls

        # --- Two-pass conditions ---
        for cond_name in two_pass_conds:
            if cond_name not in prime_map:
                continue
            surr_ids = prime_map[cond_name]
            n_s = len(surr_ids)
            n_prefix = 1 + n_s + n_d

            # Phase A: cache [BOS, prime, doc]
            prefix_tokens = [bos_id] + surr_ids + doc_ids
            prefix_input = torch.tensor([prefix_tokens], dtype=torch.long,
                                        device=device)

            phase_a_mask = make_causal_mask(n_prefix)
            phase_a_dict = make_mask_dict(phase_a_mask.to(device))
            phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

            with torch.no_grad():
                out_a = model(input_ids=prefix_input,
                              attention_mask=phase_a_dict,
                              position_ids=phase_a_pos, use_cache=True)
            past_kv = out_a.past_key_values

            # Phase B: evaluate [query, answer] with truncation
            cont_tokens = query_ids + answer_ids
            n_cont = len(cont_tokens)
            cont_input = torch.tensor([cont_tokens], dtype=torch.long,
                                      device=device)

            phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a)
            phase_b_dict = make_mask_dict(phase_b_mask.to(device))
            phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                       device=device).unsqueeze(0)

            with torch.no_grad():
                out_b = model(input_ids=cont_input,
                              attention_mask=phase_b_dict,
                              position_ids=phase_b_pos,
                              past_key_values=past_kv)

            answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
            log_probs = F.log_softmax(answer_logits, dim=-1)
            token_nlls = -log_probs.gather(
                1, targets.unsqueeze(1)).squeeze(1)

            result[f'token_nlls_{cond_name}_{answer_type}'] = \
                token_nlls.cpu().tolist()
            result[f'nll_{cond_name}_{answer_type}'] = \
                token_nlls.mean().item()

            del out_a, out_b, past_kv, prefix_input, cont_input
            del phase_a_mask, phase_a_dict, phase_b_mask, phase_b_dict
            del answer_logits, log_probs, token_nlls

    return result


# Forward count estimate per sample:
# no_doc: 1 x n_answer_types
# two-pass: n_conditions x n_answer_types x 2 phases
# With 17 conds (16 two-pass) x 4 answer types: 4 + 16*4*2 = 132 forwards
# With BoolQ 11 conds (10 two-pass) x 3 answer types: 3 + 10*3*2 = 63 forwards
print(f"Scoring function defined.")
print(f"  Standard: {len(ALL_CONDITIONS)} conditions x {len(ANSWER_TYPES)} answers")
print(f"  BoolQ: {len(BOOLQ_CONDITIONS)} conditions x 3 answers")
""")


# ===== Cells 7-12: Per-dataset scoring loops =====
for ds_idx, ds_name in enumerate(DATASET_NAMES_BUILD):
    cell_num = 7 + ds_idx
    ds_var = ds_name.replace("-", "_")

    code(rf"""# Cell {cell_num}: Dataset {ds_idx+1} -- {ds_name}
print("=" * 70)
print("DATASET {ds_idx+1}: {ds_name.upper()}")
print("=" * 70)

ds_dir = RESULTS_DIR / "{ds_name}"
ds_dir.mkdir(parents=True, exist_ok=True)

# Load and prepare
{ds_var}_samples = DATASET_LOADERS["{ds_name}"]()
{ds_var}_samples = prepare_samples({ds_var}_samples, "{ds_name}")
assign_hard_negatives({ds_var}_samples, "{ds_name}")

# Generate LLM fields
generate_llm_fields({ds_var}_samples, "{ds_name}", model, tokenizer, DEVICE)

# --- Scoring loop ---
conditions = get_conditions_for_dataset("{ds_name}")
answer_types = get_answer_types_for_dataset("{ds_name}")
n_conds = len(conditions)
n_at = len(answer_types)
print(f"\n--- Scoring {ds_name} ({{N_SAMPLES}} samples, {{n_conds}} conditions, {{n_at}} answer types) ---")

CKPT_PATH = ds_dir / "checkpoint.json"

{ds_var}_results = []
start_idx = 0
if CKPT_PATH.exists():
    ckpt = json.loads(CKPT_PATH.read_text())
    if ckpt.get("version") == 2 and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in {ds_var}_samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            {ds_var}_results = ckpt['results']
            start_idx = len({ds_var}_results)
            print(f"Resuming from checkpoint: {{start_idx}}/{{N_SAMPLES}}")

if start_idx == 0:
    print(f"Starting fresh")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="{ds_name}"):
    s = {ds_var}_samples[i]
    try:
        result = score_sample(model, tokenizer, s, DEVICE, "{ds_name}")
    except Exception as e:
        print(f"ERROR at sample {{i}}: {{e}}")
        import traceback; traceback.print_exc()
        result = None

    if result is None:
        continue

    result['query'] = s['query'][:50]
    result['query_doc_overlap'] = s['query_doc_overlap']
    result['answer_wc'] = s['answer_wc']
    result['query_wc'] = count_words(s['query'])
    result['doc_wc'] = s['word_count']
    {ds_var}_results.append(result)

    if (i + 1) % 25 == 0 or i == N_SAMPLES - 1:
        ckpt = {{
            'version': 2,
            'dataset': '{ds_name}',
            'model': MODEL_NAME,
            'n_total': N_SAMPLES,
            'conditions': conditions,
            'answer_types': answer_types,
            'results': {ds_var}_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }}
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 50 == 0:
        gc.collect(); torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {{len({ds_var}_results)}} samples in {{elapsed/60:.1f}} min")

print(f"\nQuick summary (correct answer, mean NLL):")
for cn in conditions:
    vals = [r[f'nll_{{cn}}_correct'] for r in {ds_var}_results]
    print(f"  {{cn:<25}} NLL={{np.mean(vals):.4f}}")
""")


# ===== Cell 13: Analysis A — Standard NLL ranking =====
code(r"""# Cell 13: Analysis A -- Standard NLL ranking per dataset
print("=" * 70)
print("ANALYSIS A: STANDARD NLL RANKING")
print("=" * 70)

all_datasets = {
    'msmarco': msmarco_results,
    'squad': squad_results,
    'neuralbridge': neuralbridge_results,
    'boolq': boolq_results,
    'triviaqa': triviaqa_results,
    'hotpotqa': hotpotqa_results,
}

# Storage for cross-dataset comparison
dataset_summaries = {}

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N}, {len(conditions)} conditions)")
    print(f"{'='*50}")

    # Gather per-sample mean NLLs (correct answer)
    nll_correct = {}
    for cn in conditions:
        nll_correct[cn] = np.array([r[f'nll_{cn}_correct'] for r in results])

    # d vs bare, win%, p-value, Bonferroni
    n_tests = len(conditions) - 1  # exclude bare
    bonferroni = 0.05 / n_tests

    print(f"\n  {'Condition':<25} {'Category':<14} {'NLL':>8} {'d_vs_bare':>10}"
          f" {'win%':>6} {'p':>12} {'sig':>5} {'Bonf':>5}")
    print(f"  {'-'*90}")

    cond_stats = {}
    for cn in conditions:
        cat = CONDITION_CATEGORIES[cn]
        nll_m = nll_correct[cn].mean()
        if cn == 'bare':
            d_vs_bare = 0.0
            win_pct = 0.5
            p_val = 1.0
        else:
            diff = nll_correct['bare'] - nll_correct[cn]
            d_vs_bare = cohens_d(diff)
            win_pct = (diff > 0).mean()
            _, p_val = stats.ttest_1samp(diff, 0)
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        bonf_sig = '*' if p_val < bonferroni else ''
        print(f"  {cn:<25} {cat:<14} {nll_m:>8.4f} {d_vs_bare:>+10.3f}"
              f" {win_pct:>6.1%} {p_val:>12.2e} {sig:>5} {bonf_sig:>5}")

        cond_stats[cn] = {
            'nll_correct': float(nll_m),
            'd_correct_vs_bare': float(d_vs_bare),
            'win_pct': float(win_pct),
            'p_val': float(p_val),
        }

    # Structural fraction
    d_oracle = cond_stats['oracle']['d_correct_vs_bare']
    d_random = cond_stats['random']['d_correct_vs_bare']
    sf = d_random / d_oracle if d_oracle != 0 else float('nan')
    print(f"\n  Structural fraction: d_random/d_oracle = {d_random:+.3f}/{d_oracle:+.3f} = {sf:.0%}")

    dataset_summaries[ds_name] = {
        'n_samples': N,
        'conditions': conditions,
        'nll_correct': nll_correct,
        'cond_stats': cond_stats,
        'structural_fraction': float(sf),
    }

print(f"\n\nAnalysis A complete for all {len(all_datasets)} datasets.")
""")


# ===== Cell 14: Analysis B — Token stratification by difficulty =====
code(r"""# Cell 14: Analysis B -- Token stratification by difficulty quartiles
print("=" * 70)
print("ANALYSIS B: TOKEN STRATIFICATION BY DIFFICULTY")
print("=" * 70)

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    # Gather per-token NLLs for correct answers
    tok_data = {cn: [] for cn in conditions}
    tok_ids_flat = []

    for r in results:
        n_a = len(r['correct_answer_ids'])
        for cn in conditions:
            tok_data[cn].extend(r[f'token_nlls_{cn}_correct'])
        tok_ids_flat.extend(r['correct_answer_ids'])

    for cn in conditions:
        tok_data[cn] = np.array(tok_data[cn])
    tok_ids_flat = np.array(tok_ids_flat)
    n_tokens = len(tok_ids_flat)

    print(f"\n  Total tokens pooled: {n_tokens}")
    print(f"  Mean tokens/sample: {n_tokens / N:.1f}")

    # Difficulty quartiles (bare NLL)
    difficulty = tok_data['bare']
    q25, q50, q75 = np.percentile(difficulty, [25, 50, 75])
    quartile = np.digitize(difficulty, bins=[q25, q50, q75])

    q_labels = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
    print(f"\n  {'Quartile':<12} {'N':>6} {'bare':>8} {'random':>8}"
          f" {'oracle':>8} {'d(o-r)':>8} {'d(o-b)':>8}")
    print(f"  {'-'*60}")

    d_or_by_q = []
    for q in range(4):
        mask = quartile == q
        n_tok = mask.sum()
        bare_m = tok_data['bare'][mask].mean()
        rand_m = tok_data['random'][mask].mean()
        orac_m = tok_data['oracle'][mask].mean()
        d_or = cohens_d(tok_data['random'][mask] - tok_data['oracle'][mask])
        d_ob = cohens_d(tok_data['bare'][mask] - tok_data['oracle'][mask])
        d_or_by_q.append(d_or)
        print(f"  {q_labels[q]:<12} {n_tok:>6} {bare_m:>8.3f} {rand_m:>8.3f}"
              f" {orac_m:>8.3f} {d_or:>+8.3f} {d_ob:>+8.3f}")

    gradient = d_or_by_q[3] - d_or_by_q[0]
    print(f"\n  d(oracle-random) gradient (Q4-Q1): {gradient:+.3f}")

    dataset_summaries[ds_name]['tok_data'] = tok_data
    dataset_summaries[ds_name]['quartile'] = quartile
    dataset_summaries[ds_name]['d_gradient_q4_q1'] = float(gradient)
    dataset_summaries[ds_name]['n_tokens'] = n_tokens

print(f"\n\nAnalysis B complete.")
""")


# ===== Cell 15: Analysis C — Document dependence =====
code(r"""# Cell 15: Analysis C -- Document dependence quartiles
print("=" * 70)
print("ANALYSIS C: DOCUMENT DEPENDENCE")
print("=" * 70)

for ds_name, results in all_datasets.items():
    conditions = get_conditions_for_dataset(ds_name)
    tok_data = dataset_summaries[ds_name]['tok_data']
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*50}")

    doc_dep = tok_data['no_doc'] - tok_data['bare']
    dq25, dq50, dq75 = np.percentile(doc_dep, [25, 50, 75])
    dep_quartile = np.digitize(doc_dep, bins=[dq25, dq50, dq75])

    dep_labels = ["Q1 (doc-indep)", "Q2", "Q3", "Q4 (doc-dep)"]
    print(f"\n  {'Quartile':<16} {'N':>6} {'dep':>8} {'d(o-r)':>8} {'d(o-b)':>8}")
    print(f"  {'-'*50}")

    for q in range(4):
        mask = dep_quartile == q
        dep_m = doc_dep[mask].mean()
        d_or = cohens_d(tok_data['random'][mask] - tok_data['oracle'][mask])
        d_ob = cohens_d(tok_data['bare'][mask] - tok_data['oracle'][mask])
        print(f"  {dep_labels[q]:<16} {mask.sum():>6} {dep_m:>+8.3f}"
              f" {d_or:>+8.3f} {d_ob:>+8.3f}")

    dataset_summaries[ds_name]['dep_quartile'] = dep_quartile

print(f"\n\nAnalysis C complete.")
""")


# ===== Cell 16: Analysis D — Contrastive AUC (random negatives) =====
code(r"""# Cell 16: Analysis D -- Contrastive AUC (random negatives)
print("=" * 70)
print("ANALYSIS D: CONTRASTIVE AUC -- RANDOM NEGATIVES")
print("=" * 70)

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    nll_correct = dataset_summaries[ds_name]['nll_correct']
    nll_wrong_rand = {}
    for cn in conditions:
        nll_wrong_rand[cn] = np.array([r[f'nll_{cn}_wrong_random'] for r in results])

    gap_rand = {}
    for cn in conditions:
        gap_rand[cn] = nll_wrong_rand[cn] - nll_correct[cn]

    print(f"\n  {'Condition':<25} {'mean_gap':>10} {'AUC':>8} {'d(gap)':>8}"
          f" {'p':>12} {'sig':>5}")
    print(f"  {'-'*72}")

    for cn in conditions:
        g = gap_rand[cn]
        auc = (g > 0).mean()
        d_gap = cohens_d(g)
        _, p_gap = stats.ttest_1samp(g, 0)
        sig = ('***' if p_gap < 0.001 else '**' if p_gap < 0.01
               else '*' if p_gap < 0.05 else 'ns')
        print(f"  {cn:<25} {g.mean():>+10.3f} {auc:>8.1%} {d_gap:>+8.3f}"
              f" {p_gap:>12.2e} {sig:>5}")

    # Oracle vs random discrimination
    diff_d = gap_rand['oracle'] - gap_rand['random']
    d_d = cohens_d(diff_d)
    _, p_d = stats.ttest_1samp(diff_d, 0)
    sig_d = ('***' if p_d < 0.001 else '**' if p_d < 0.01
             else '*' if p_d < 0.05 else 'ns')
    print(f"\n  Oracle vs random discrimination: d={d_d:+.3f}, p={p_d:.2e} {sig_d}")

    dataset_summaries[ds_name]['gap_rand'] = gap_rand
    dataset_summaries[ds_name]['d_discrim_rand'] = float(d_d)
    dataset_summaries[ds_name]['nll_wrong_rand'] = nll_wrong_rand

    # Per-condition contrastive stats
    for cn in conditions:
        g = gap_rand[cn]
        dataset_summaries[ds_name]['cond_stats'][cn]['auc_rand'] = float((g > 0).mean())
        dataset_summaries[ds_name]['cond_stats'][cn]['d_gap_rand'] = float(cohens_d(g))

print(f"\n\nAnalysis D complete.")
""")


# ===== Cell 17: Analysis E — Contrastive AUC (hard negatives) =====
code(r"""# Cell 17: Analysis E -- Contrastive AUC (hard negatives)
print("=" * 70)
print("ANALYSIS E: CONTRASTIVE AUC -- HARD NEGATIVES")
print("=" * 70)

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    nll_correct = dataset_summaries[ds_name]['nll_correct']
    nll_wrong_hard = {}
    for cn in conditions:
        nll_wrong_hard[cn] = np.array([r[f'nll_{cn}_wrong_hard'] for r in results])

    gap_hard = {}
    for cn in conditions:
        gap_hard[cn] = nll_wrong_hard[cn] - nll_correct[cn]

    print(f"\n  {'Condition':<25} {'mean_gap':>10} {'AUC':>8} {'d(gap)':>8}"
          f" {'p':>12} {'sig':>5}")
    print(f"  {'-'*72}")

    for cn in conditions:
        g = gap_hard[cn]
        auc = (g > 0).mean()
        d_gap = cohens_d(g)
        _, p_gap = stats.ttest_1samp(g, 0)
        sig = ('***' if p_gap < 0.001 else '**' if p_gap < 0.01
               else '*' if p_gap < 0.05 else 'ns')
        print(f"  {cn:<25} {g.mean():>+10.3f} {auc:>8.1%} {d_gap:>+8.3f}"
              f" {p_gap:>12.2e} {sig:>5}")

    # Oracle vs random
    diff_d = gap_hard['oracle'] - gap_hard['random']
    d_d = cohens_d(diff_d)
    _, p_d = stats.ttest_1samp(diff_d, 0)
    sig_d = ('***' if p_d < 0.001 else '**' if p_d < 0.01
             else '*' if p_d < 0.05 else 'ns')
    print(f"\n  Oracle vs random discrimination (hard neg): d={d_d:+.3f}, p={p_d:.2e} {sig_d}")

    # Compare hard vs random negatives
    gap_rand = dataset_summaries[ds_name]['gap_rand']
    auc_rand_bare = (gap_rand['bare'] > 0).mean()
    auc_hard_bare = (gap_hard['bare'] > 0).mean()
    print(f"  AUC(bare): random_neg={auc_rand_bare:.1%} vs hard_neg={auc_hard_bare:.1%}")

    dataset_summaries[ds_name]['gap_hard'] = gap_hard
    dataset_summaries[ds_name]['d_discrim_hard'] = float(d_d)

    for cn in conditions:
        g = gap_hard[cn]
        dataset_summaries[ds_name]['cond_stats'][cn]['auc_hard'] = float((g > 0).mean())
        dataset_summaries[ds_name]['cond_stats'][cn]['d_gap_hard'] = float(cohens_d(g))

print(f"\n\nAnalysis E complete.")
""")


# ===== Cell 18: Analysis F — Contrastive AUC (LLM negatives) =====
code(r"""# Cell 18: Analysis F -- Contrastive AUC (LLM-generated negatives)
print("=" * 70)
print("ANALYSIS F: CONTRASTIVE AUC -- LLM-GENERATED NEGATIVES")
print("=" * 70)

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    answer_types = get_answer_types_for_dataset(ds_name)

    if 'wrong_llm' not in answer_types:
        print(f"\n  {ds_name}: skipping (no LLM negatives)")
        dataset_summaries[ds_name]['gap_llm'] = None
        continue

    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    nll_correct = dataset_summaries[ds_name]['nll_correct']
    nll_wrong_llm = {}
    for cn in conditions:
        nll_wrong_llm[cn] = np.array([r[f'nll_{cn}_wrong_llm'] for r in results])

    gap_llm = {}
    for cn in conditions:
        gap_llm[cn] = nll_wrong_llm[cn] - nll_correct[cn]

    print(f"\n  {'Condition':<25} {'mean_gap':>10} {'AUC':>8} {'d(gap)':>8}"
          f" {'p':>12} {'sig':>5}")
    print(f"  {'-'*72}")

    for cn in conditions:
        g = gap_llm[cn]
        auc = (g > 0).mean()
        d_gap = cohens_d(g)
        _, p_gap = stats.ttest_1samp(g, 0)
        sig = ('***' if p_gap < 0.001 else '**' if p_gap < 0.01
               else '*' if p_gap < 0.05 else 'ns')
        print(f"  {cn:<25} {g.mean():>+10.3f} {auc:>8.1%} {d_gap:>+8.3f}"
              f" {p_gap:>12.2e} {sig:>5}")

    # Oracle vs random
    diff_d = gap_llm['oracle'] - gap_llm['random']
    d_d = cohens_d(diff_d)
    _, p_d = stats.ttest_1samp(diff_d, 0)
    sig_d = ('***' if p_d < 0.001 else '**' if p_d < 0.01
             else '*' if p_d < 0.05 else 'ns')
    print(f"\n  Oracle vs random discrimination (LLM neg): d={d_d:+.3f}, p={p_d:.2e} {sig_d}")

    # Compare all 3 negative types for bare
    gap_rand = dataset_summaries[ds_name]['gap_rand']
    gap_hard = dataset_summaries[ds_name]['gap_hard']
    print(f"\n  AUC(bare) by negative type:")
    print(f"    Random neg: {(gap_rand['bare'] > 0).mean():.1%}")
    print(f"    Hard neg:   {(gap_hard['bare'] > 0).mean():.1%}")
    print(f"    LLM neg:    {(gap_llm['bare'] > 0).mean():.1%}")

    dataset_summaries[ds_name]['gap_llm'] = gap_llm
    dataset_summaries[ds_name]['d_discrim_llm'] = float(d_d)

    for cn in conditions:
        g = gap_llm[cn]
        dataset_summaries[ds_name]['cond_stats'][cn]['auc_llm'] = float((g > 0).mean())
        dataset_summaries[ds_name]['cond_stats'][cn]['d_gap_llm'] = float(cohens_d(g))

print(f"\n\nAnalysis F complete.")
""")


# ===== Cell 19: Analysis G — Targeted NLL =====
code(r"""# Cell 19: Analysis G -- Targeted NLL (hard + doc-dependent tokens only)
print("=" * 70)
print("ANALYSIS G: TARGETED NLL (HARD + DOC-DEPENDENT TOKENS)")
print("=" * 70)

for ds_name, results in all_datasets.items():
    N = len(results)
    conditions = get_conditions_for_dataset(ds_name)
    tok_data = dataset_summaries[ds_name]['tok_data']
    quartile = dataset_summaries[ds_name]['quartile']
    dep_quartile = dataset_summaries[ds_name]['dep_quartile']

    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    # Targeted tokens: Q4 difficulty AND Q4 doc-dependent
    target_mask = (quartile == 3) & (dep_quartile == 3)
    n_target = target_mask.sum()
    n_total = len(quartile)
    print(f"\n  Targeted tokens: {n_target}/{n_total} ({100*n_target/n_total:.1f}%)")

    if n_target < 50:
        print(f"  Too few targeted tokens, skipping.")
        dataset_summaries[ds_name]['targeted_d'] = {}
        continue

    print(f"\n  {'Condition':<25} {'NLL_target':>12} {'d_vs_bare':>10}"
          f" {'d(o-r)':>8}")
    print(f"  {'-'*58}")

    targeted_d = {}
    for cn in conditions:
        nll_t = tok_data[cn][target_mask].mean()
        d_vs_bare = cohens_d(tok_data['bare'][target_mask] - tok_data[cn][target_mask])
        targeted_d[cn] = float(d_vs_bare)
        print(f"  {cn:<25} {nll_t:>12.4f} {d_vs_bare:>+10.3f}", end="")
        if cn == conditions[-1] or cn == 'oracle':
            d_or = cohens_d(tok_data['random'][target_mask] - tok_data['oracle'][target_mask])
            print(f" {d_or:>+8.3f}")
        else:
            print()

    # Compare targeted vs aggregate
    d_orc_all = cohens_d(tok_data['bare'] - tok_data['oracle'])
    d_orc_target = targeted_d.get('oracle', 0.0)
    print(f"\n  d_oracle: aggregate={d_orc_all:+.3f}, targeted={d_orc_target:+.3f}")
    print(f"  Targeted amplification: {d_orc_target / d_orc_all:.1f}x" if d_orc_all != 0 else "  N/A")

    dataset_summaries[ds_name]['targeted_d'] = targeted_d

print(f"\n\nAnalysis G complete.")
""")


# ===== Cell 20: Analysis H — Vocabulary bridge dose-response =====
code(r"""# Cell 20: Analysis H -- Vocabulary bridge dose-response curve
print("=" * 70)
print("ANALYSIS H: VOCABULARY BRIDGE DOSE-RESPONSE")
print("=" * 70)

bridge_ns = [1, 3, 5, 10, 15, 20]
bridge_conds = [f"vocab_bridge_{n}" for n in bridge_ns]

all_sample_lists = {
    'msmarco': msmarco_samples,
    'squad': squad_samples,
    'neuralbridge': neuralbridge_samples,
    'boolq': boolq_samples,
    'triviaqa': triviaqa_samples,
    'hotpotqa': hotpotqa_samples,
}

for ds_name, results in all_datasets.items():
    conditions = get_conditions_for_dataset(ds_name)
    if 'vocab_bridge_1' not in conditions:
        print(f"\n  {ds_name}: skipping (no dose-response conditions)")
        dataset_summaries[ds_name]['dose_response'] = None
        continue

    N = len(results)
    nll_correct = dataset_summaries[ds_name]['nll_correct']
    print(f"\n{'='*50}")
    print(f"  Dataset: {ds_name} (N={N})")
    print(f"{'='*50}")

    # Degeneracy check: how many samples have fewer overlap words than requested?
    sample_list = all_sample_lists[ds_name]
    print(f"\n  Overlap word availability:")
    for n_bridge in bridge_ns:
        key = f'vocab_bridge_{n_bridge}'
        actual_ns = [s[f'{key}_actual_n'] for s in sample_list]
        degen_rate = sum(1 for a in actual_ns if a < n_bridge) / len(actual_ns)
        mean_actual = np.mean(actual_ns)
        print(f"    bridge_{n_bridge}: mean_actual={mean_actual:.1f},"
              f" degeneracy={degen_rate:.1%}")

    # Dose-response: d vs bare and d vs random
    print(f"\n  {'Condition':<25} {'d_vs_bare':>10} {'d_vs_random':>12}"
          f" {'NLL':>8}")
    print(f"  {'-'*60}")

    # Include reference conditions
    ref_conds = ['bare', 'random', 'oracle', 'oracle_plus_vocab']
    for cn in ref_conds + bridge_conds:
        if cn not in nll_correct:
            continue
        nll_m = nll_correct[cn].mean()
        d_bare = cohens_d(nll_correct['bare'] - nll_correct[cn])
        d_rand = cohens_d(nll_correct['random'] - nll_correct[cn])
        print(f"  {cn:<25} {d_bare:>+10.3f} {d_rand:>+12.3f} {nll_m:>8.4f}")

    # Also check contrastive AUC dose-response
    gap_rand = dataset_summaries[ds_name]['gap_rand']
    print(f"\n  Contrastive AUC dose-response (random negatives):")
    print(f"  {'Condition':<25} {'AUC':>8} {'d(gap)':>8}")
    print(f"  {'-'*44}")

    for cn in ref_conds + bridge_conds:
        if cn not in gap_rand:
            continue
        g = gap_rand[cn]
        auc = (g > 0).mean()
        d_gap = cohens_d(g)
        print(f"  {cn:<25} {auc:>8.1%} {d_gap:>+8.3f}")

    # Store dose-response data
    dose_data = {}
    for n_bridge in bridge_ns:
        cn = f'vocab_bridge_{n_bridge}'
        if cn in nll_correct:
            dose_data[n_bridge] = {
                'd_vs_bare': float(cohens_d(nll_correct['bare'] - nll_correct[cn])),
                'd_vs_random': float(cohens_d(nll_correct['random'] - nll_correct[cn])),
                'nll': float(nll_correct[cn].mean()),
            }
    dataset_summaries[ds_name]['dose_response'] = dose_data

print(f"\n\nAnalysis H complete.")
""")


# ===== Cell 21: Analysis I — Cross-dataset comparison =====
code(r"""# Cell 21: Analysis I -- Cross-dataset comparison
print("=" * 70)
print("ANALYSIS I: CROSS-DATASET COMPARISON")
print("=" * 70)

# I1. Full effect size table
print(f"\n--- I1. Effect Size Table (d_correct vs bare) ---\n")
# Use only conditions common to all datasets (exclude dose-response for clean comparison)
common_conds = BOOLQ_CONDITIONS  # 11 conditions available in all datasets

print(f"  {'Condition':<25} {'Category':<14}", end="")
for ds in DATASET_NAMES:
    print(f" {ds:>10}", end="")
print(f" {'Mean':>8}")
print(f"  {'-'*(25+14+10*len(DATASET_NAMES)+8)}")

mean_d_all = {}
for cn in common_conds:
    cat = CONDITION_CATEGORIES[cn]
    print(f"  {cn:<25} {cat:<14}", end="")
    ds_vals = []
    for ds in DATASET_NAMES:
        d = dataset_summaries[ds]['cond_stats'][cn]['d_correct_vs_bare']
        print(f" {d:>+10.3f}", end="")
        ds_vals.append(d)
    mean_v = np.mean(ds_vals)
    mean_d_all[cn] = mean_v
    print(f" {mean_v:>+8.3f}")

# I2. Structural fraction per dataset
print(f"\n--- I2. Structural Fraction ---")
for ds in DATASET_NAMES:
    sf = dataset_summaries[ds]['structural_fraction']
    d_o = dataset_summaries[ds]['cond_stats']['oracle']['d_correct_vs_bare']
    d_r = dataset_summaries[ds]['cond_stats']['random']['d_correct_vs_bare']
    print(f"  {ds:<15} d_oracle={d_o:+.3f}, d_random={d_r:+.3f},"
          f" structural={sf:.0%}")

# I3. Surrogate ranking stability (Kendall's tau)
print(f"\n--- I3. Surrogate Ranking Stability (Kendall's tau) ---")
surr_conds = [cn for cn in common_conds if cn not in ('no_doc', 'bare')]

rankings = {}
for ds in DATASET_NAMES:
    rankings[ds] = [dataset_summaries[ds]['cond_stats'][cn]['d_correct_vs_bare']
                    for cn in surr_conds]

import itertools
for ds_a, ds_b in itertools.combinations(DATASET_NAMES, 2):
    tau, p_tau = stats.kendalltau(rankings[ds_a], rankings[ds_b])
    sig = ('***' if p_tau < 0.001 else '**' if p_tau < 0.01
           else '*' if p_tau < 0.05 else 'ns')
    print(f"  {ds_a} vs {ds_b}: tau={tau:+.3f} (p={p_tau:.3f}) {sig}")

# I4. Category means across datasets
print(f"\n--- I4. Category Means ---")
categories = {
    'control': ['no_doc', 'bare'],
    'structural': ['random'],
    'semantic': ['oracle', 'oracle_plus_vocab', 'pointer'],
    'extraction': ['doc_kw10', 'instruct'],
    'LLM': ['llm_query', 'llm_summary', 'llm_keywords'],
}

print(f"\n  {'Category':<14} {'Conditions':<35}", end="")
for ds in DATASET_NAMES:
    print(f" {ds:>10}", end="")
print(f" {'Mean':>8}")
print(f"  {'-'*(14+35+10*len(DATASET_NAMES)+8)}")

for cat_name, cat_conds in categories.items():
    cond_str = ", ".join(cat_conds)
    print(f"  {cat_name:<14} {cond_str:<35}", end="")
    ds_vals = []
    for ds in DATASET_NAMES:
        mean_d = np.mean([
            dataset_summaries[ds]['cond_stats'][cn]['d_correct_vs_bare']
            for cn in cat_conds if cn in dataset_summaries[ds]['cond_stats']
        ])
        print(f" {mean_d:>+10.3f}", end="")
        ds_vals.append(mean_d)
    print(f" {np.mean(ds_vals):>+8.3f}")

# I5. Difficulty gradient generalization
print(f"\n--- I5. Difficulty Gradient ---")
for ds in DATASET_NAMES:
    g = dataset_summaries[ds]['d_gradient_q4_q1']
    print(f"  {ds:<15} gradient={g:+.3f}")

# I6. Discrimination comparison across datasets and negative types
print(f"\n--- I6. Discrimination (oracle vs random) by Negative Type ---")
print(f"  {'Dataset':<15} {'Random':>10} {'Hard':>10} {'LLM':>10}")
print(f"  {'-'*48}")

for ds in DATASET_NAMES:
    d_rand = dataset_summaries[ds].get('d_discrim_rand', float('nan'))
    d_hard = dataset_summaries[ds].get('d_discrim_hard', float('nan'))
    d_llm = dataset_summaries[ds].get('d_discrim_llm', float('nan'))
    print(f"  {ds:<15} {d_rand:>+10.3f} {d_hard:>+10.3f}", end="")
    if np.isnan(d_llm):
        print(f" {'N/A':>10}")
    else:
        print(f" {d_llm:>+10.3f}")

# I7. Contrastive AUC heatmap (all conditions x all datasets, random negatives)
print(f"\n--- I7. AUC Heatmap (random negatives) ---\n")
print(f"  {'Condition':<25}", end="")
for ds in DATASET_NAMES:
    print(f" {ds:>10}", end="")
print()
print(f"  {'-'*(25+10*len(DATASET_NAMES))}")

for cn in common_conds:
    print(f"  {cn:<25}", end="")
    for ds in DATASET_NAMES:
        auc = dataset_summaries[ds]['cond_stats'][cn].get('auc_rand', float('nan'))
        if np.isnan(auc):
            print(f" {'N/A':>10}", end="")
        else:
            print(f" {auc:>10.1%}", end="")
    print()

print(f"\n\nAnalysis I complete.")
""")


# ===== Cell 22: Summary — final tables, hypothesis tests, save =====
code(r"""# Cell 22: Summary -- final tables, hypothesis tests, save results
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 05: Hero Run")
print("=" * 70)

# --- Replication checks ---
print(f"\n--- Replication Checks ---")

# MS MARCO: d_oracle ~ +0.452, d_random ~ +0.475, structural ~ 105%
cs_mm = dataset_summaries['msmarco']['cond_stats']
d_orc_mm = cs_mm['oracle']['d_correct_vs_bare']
d_rnd_mm = cs_mm['random']['d_correct_vs_bare']
sf_mm = d_rnd_mm / d_orc_mm if d_orc_mm != 0 else float('nan')
print(f"  MS MARCO: d_oracle={d_orc_mm:+.3f} (expected ~+0.452),"
      f" d_random={d_rnd_mm:+.3f} (expected ~+0.475),"
      f" structural={sf_mm:.0%} (expected ~105%)")

# Contrastive sanity: AUC(bare) > 50% on all datasets
print(f"\n  Contrastive sanity (AUC_bare > 50%, random negatives):")
for ds in DATASET_NAMES:
    auc_bare = dataset_summaries[ds]['cond_stats']['bare'].get('auc_rand', 0)
    ok = "PASS" if auc_bare > 0.50 else "FAIL"
    print(f"    {ds}: AUC(bare) = {auc_bare:.1%} [{ok}]")

# --- Final ranked comparison ---
common_conds = BOOLQ_CONDITIONS
surr_conds_ranked = [cn for cn in common_conds if cn not in ('no_doc', 'bare')]

print(f"\n--- Final Surrogate Ranking (d_correct vs bare, mean across 6 datasets) ---\n")
print(f"  {'Rank':>4} {'Condition':<25} {'Category':<14} {'Mean_d':>8}", end="")
for ds in DATASET_NAMES:
    print(f" {ds[:6]:>8}", end="")
print()
print(f"  {'-'*(4+25+14+8+8*len(DATASET_NAMES))}")

# Compute mean d across ALL datasets
mean_d = {}
for cn in surr_conds_ranked:
    mean_d[cn] = np.mean([
        dataset_summaries[ds]['cond_stats'][cn]['d_correct_vs_bare']
        for ds in DATASET_NAMES
    ])

ranked = sorted(surr_conds_ranked, key=lambda c: -mean_d[c])
for rank, cn in enumerate(ranked, 1):
    cat = CONDITION_CATEGORIES[cn]
    print(f"  {rank:>4} {cn:<25} {cat:<14} {mean_d[cn]:>+8.3f}", end="")
    for ds in DATASET_NAMES:
        d = dataset_summaries[ds]['cond_stats'][cn]['d_correct_vs_bare']
        print(f" {d:>+8.3f}", end="")
    print()

# --- Hypothesis Tests ---
print(f"\n--- Hypothesis Tests ---")

# H1: LLM surrogates beat static instructions
d_lq = np.mean([dataset_summaries[ds]['cond_stats']['llm_query']['d_correct_vs_bare']
                 for ds in DATASET_NAMES])
d_inst = np.mean([dataset_summaries[ds]['cond_stats']['instruct']['d_correct_vs_bare']
                   for ds in DATASET_NAMES])
h1 = d_lq > d_inst + 0.02
print(f"  H1 (LLM > static): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}"
      f" (llm_query={d_lq:+.3f} vs instruct={d_inst:+.3f})")

# H2: Token stratification generalizes across all 6 datasets
gradients = [dataset_summaries[ds]['d_gradient_q4_q1'] for ds in DATASET_NAMES]
h2 = sum(1 for g in gradients if g > 0.03) >= 4  # majority
print(f"  H2 (stratification generalizes): {'SUPPORTED' if h2 else 'NOT SUPPORTED'}"
      f" (positive in {sum(1 for g in gradients if g > 0.03)}/6 datasets)")
print(f"      gradients: {dict(zip(DATASET_NAMES, [f'{g:+.3f}' for g in gradients]))}")

# H3: Hard negatives reveal more signal than random negatives
d_discrim_rand = [dataset_summaries[ds].get('d_discrim_rand', 0) for ds in DATASET_NAMES]
d_discrim_hard = [dataset_summaries[ds].get('d_discrim_hard', 0) for ds in DATASET_NAMES]
h3 = np.mean(d_discrim_hard) > np.mean(d_discrim_rand) + 0.02
print(f"  H3 (hard neg > random neg discrimination): {'SUPPORTED' if h3 else 'NOT SUPPORTED'}"
      f" (mean_hard={np.mean(d_discrim_hard):+.3f} vs mean_rand={np.mean(d_discrim_rand):+.3f})")

# H4: Structural fraction varies by dataset characteristics
sfs = [dataset_summaries[ds]['structural_fraction'] for ds in DATASET_NAMES]
sf_range = max(sfs) - min(sfs)
h4 = sf_range > 0.3
print(f"  H4 (structural fraction varies): {'SUPPORTED' if h4 else 'NOT SUPPORTED'}"
      f" (range={sf_range:.0%}, min={min(sfs):.0%}, max={max(sfs):.0%})")

# H5: oracle_plus_vocab consistently beats oracle
h5_wins = 0
for ds in DATASET_NAMES:
    d_opv = dataset_summaries[ds]['cond_stats']['oracle_plus_vocab']['d_correct_vs_bare']
    d_orc = dataset_summaries[ds]['cond_stats']['oracle']['d_correct_vs_bare']
    if d_opv > d_orc + 0.01:
        h5_wins += 1
h5 = h5_wins >= 4
print(f"  H5 (opv > oracle consistently): {'SUPPORTED' if h5 else 'NOT SUPPORTED'}"
      f" (wins {h5_wins}/6 datasets)")

# H6: Pointer instruction competitive with oracle_plus_vocab
d_ptr = np.mean([dataset_summaries[ds]['cond_stats']['pointer']['d_correct_vs_bare']
                  for ds in DATASET_NAMES])
d_opv = np.mean([dataset_summaries[ds]['cond_stats']['oracle_plus_vocab']['d_correct_vs_bare']
                  for ds in DATASET_NAMES])
h6 = abs(d_ptr - d_opv) < 0.05
print(f"  H6 (pointer ~ opv): {'SUPPORTED' if h6 else 'NOT SUPPORTED'}"
      f" (pointer={d_ptr:+.3f} vs opv={d_opv:+.3f})")

# --- Save all results ---
print(f"\n--- Saving Results ---")

# Per-dataset results
for ds in DATASET_NAMES:
    ds_results = {
        'experiment': 'prefix_lm_exp05_hero',
        'dataset': ds,
        'model': MODEL_NAME,
        'n_samples': dataset_summaries[ds]['n_samples'],
        'seed': SEED,
        'conditions': dataset_summaries[ds]['conditions'],
        'condition_categories': CONDITION_CATEGORIES,
        'cond_stats': dataset_summaries[ds]['cond_stats'],
        'structural_fraction': dataset_summaries[ds]['structural_fraction'],
        'd_gradient_q4_q1': dataset_summaries[ds]['d_gradient_q4_q1'],
        'dose_response': dataset_summaries[ds].get('dose_response'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = RESULTS_DIR / ds / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(ds_results, f, indent=2)
    print(f"  Saved {out_path}")

# Cross-dataset results
cross_results = {
    'experiment': 'prefix_lm_exp05_hero',
    'model': MODEL_NAME,
    'seed': SEED,
    'datasets': DATASET_NAMES,
    'conditions': ALL_CONDITIONS,
    'condition_categories': CONDITION_CATEGORIES,
    'mean_d_correct': {cn: float(mean_d.get(cn, 0)) for cn in surr_conds_ranked},
    'ranking': ranked,
    'structural_fraction': {
        ds: dataset_summaries[ds]['structural_fraction'] for ds in DATASET_NAMES
    },
    'hypotheses': {
        'H1_llm_beats_static': h1,
        'H2_stratification_generalizes': h2,
        'H3_hard_neg_better': h3,
        'H4_structural_varies': h4,
        'H5_opv_beats_oracle': h5,
        'H6_pointer_matches_opv': h6,
    },
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with open(RESULTS_DIR / 'cross_dataset_results.json', 'w') as f:
    json.dump(cross_results, f, indent=2)
print(f"  Saved {RESULTS_DIR / 'cross_dataset_results.json'}")

print(f"\n{'='*70}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*70}")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/05/05_hero_run.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
