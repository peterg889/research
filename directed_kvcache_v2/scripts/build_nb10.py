#!/usr/bin/env python3
"""Build script for 10_semantic_content_gradient.ipynb"""

import json

def make_cell(cell_type, source, cell_id=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split('\n')
    }
    if cell_id:
        cell["id"] = cell_id
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def s(text):
    """Convert multi-line string to notebook source lines (preserving newlines)."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    if result and result[-1] == '':
        result = result[:-1]
    return result

cells = []

# ========== Cell 0: Markdown overview ==========
cells.append(make_cell("markdown", s("""\
# Exp 10: Semantic Content Gradient — Does Content Matter?

## Motivation

Previous experiments show conflicting signals about whether semantic content matters:
- **FOR**: static_fact_trunc (d=+0.472) >> random_trunc (d=+0.125) in Exp 07 — 3.8× larger effect
- **FOR**: LLM-kw (d=+0.234) >> random (d=+0.125) in Exp 06 — coherence matters
- **FOR**: static_fact values-only (d=+0.466) >> random values-only (d=+0.310) in Exp 09
- **AGAINST**: oracle (d=+0.023, ns) ≈ random (d=+0.091) in Exp 01
- **AGAINST**: separator-only (d=+0.231) ≈ LLM-kw-trunc (d=+0.234) in Exp 06

This experiment resolves the question decisively by testing a **gradient of semantic relevance**
in **both truncated-prefix and suffix modes**. If semantic content helps in both modes, the
effect cannot be explained by structural artifacts alone.

## Key Design Principle

Each semantic level is tested in BOTH delivery modes (truncated prefix, suffix). This is
critical because:
- **Truncated prefix**: Affects document value vectors (value contamination)
- **Suffix**: Document KV entries are unchanged; effect must come from query → suffix attention

If the same semantic gradient appears in BOTH modes, it proves the benefit is genuinely semantic.

## 16 Conditions

| # | Condition | Mode | Content | Semantic Level |
|---|-----------|------|---------|----------------|
| 1 | bare | — | No prefix | Baseline |
| 2 | random_trunc | Trunc | Random tokens | 0 |
| 3 | random_words_trunc | Trunc | Random English words | 0.5 |
| 4 | wrong_doc_llm_trunc | Trunc | LLM-kw from wrong doc | 1 |
| 5 | tfidf_kw_trunc | Trunc | TF-IDF keywords (right doc) | 2 |
| 6 | llm_kw_trunc | Trunc | LLM-kw (right doc) | 3 |
| 7 | static_fact_trunc | Trunc | "What are the key facts?" | 4 |
| 8 | oracle_kw_trunc | Trunc | Oracle as keywords | 5 |
| 9 | oracle_raw_trunc | Trunc | Oracle (raw question format) | 5* |
| 10 | random_suffix | Suffix | Random tokens | 0 |
| 11 | random_words_suffix | Suffix | Random English words | 0.5 |
| 12 | wrong_doc_llm_suffix | Suffix | LLM-kw from wrong doc | 1 |
| 13 | tfidf_kw_suffix | Suffix | TF-IDF keywords (right doc) | 2 |
| 14 | llm_kw_suffix | Suffix | LLM-kw (right doc) | 3 |
| 15 | static_fact_suffix | Suffix | "What are the key facts?" | 4 |
| 16 | oracle_kw_suffix | Suffix | Oracle as keywords | 5 |

*oracle_raw_trunc uses question format (potential interference), included for comparison.

## 10 Primary Comparisons (Bonferroni alpha = 0.005)

| # | Comparison | Question |
|---|-----------|----------|
| C1 | llm_kw_trunc vs random_trunc | LLM > random in truncated? |
| C2 | static_fact_trunc vs random_trunc | Static > random in truncated? |
| C3 | llm_kw_trunc vs wrong_doc_llm_trunc | Right doc > wrong doc (trunc)? |
| C4 | tfidf_kw_trunc vs random_trunc | TF-IDF > random in truncated? |
| C5 | oracle_kw_trunc vs oracle_raw_trunc | Keyword > question format? |
| C6 | llm_kw_suffix vs random_suffix | LLM > random in suffix? |
| C7 | static_fact_suffix vs random_suffix | Static > random in suffix? |
| C8 | llm_kw_suffix vs wrong_doc_llm_suffix | Right doc > wrong doc (suffix)? |
| C9 | llm_kw_trunc vs llm_kw_suffix | Truncated > suffix (LLM content)? |
| C10 | static_fact_trunc vs static_fact_suffix | Truncated > suffix (static)? |

## Decisive Predictions

If semantic content matters, we expect:
1. **Truncated gradient**: random < wrong_doc < tfidf < llm_kw (C1, C3, C4 all sig)
2. **Suffix gradient**: random < wrong_doc < llm_kw (C6, C8 both sig)
3. **Static phrase dominance**: static_fact > random in BOTH modes (C2, C7 both sig)
4. **Format effect**: oracle_kw > oracle_raw in truncated (C5 sig, replicates Exp 06)""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import re
from collections import Counter
import numpy as np
import torch
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results/exp10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_DIR = RESULTS_DIR / "surrogates"
SURROGATES_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
FINAL_RESULTS_PATH = RESULTS_DIR / "results.json"

print(f"SEED: {SEED}")
print(f"Results directory: {RESULTS_DIR}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\
""")))

# ========== Cell 2: Load model ==========
cells.append(make_cell("code", s("""\
# Cell 2: Load model (Mistral-7B 4-bit)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading {MODEL_NAME} (4-bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print(f"Model loaded. dtype={model.dtype}, device={model.device}")\
""")))

# ========== Cell 3: Imports + config + constants + helpers ==========
cells.append(make_cell("code", s("""\
# Cell 3: Imports, config, constants, and helper functions
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    score_answer_with_cache,
    build_suffix_kv_cache,
)
from lib.data import load_ms_marco, load_evaluation_samples
from lib.analysis import cohens_d
from lib.surrogate import generate_all_5_surrogates, STATIC_SURROGATE_QUERIES
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=2000,
    min_passage_words=20,
    max_passage_words=500,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"

N_EVAL = 1000
N_COMPARISONS = 10
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
CHECKPOINT_EVERY = 50

SUFFIX_SEPARATOR = "\\n\\nRelated question: "
STATIC_FACTUAL_PHRASE = STATIC_SURROGATE_QUERIES['static_factual']['query']

CONDITION_NAMES = [
    'bare',
    # Truncated gradient (8)
    'random_trunc', 'random_words_trunc', 'wrong_doc_llm_trunc',
    'tfidf_kw_trunc', 'llm_kw_trunc', 'static_fact_trunc',
    'oracle_kw_trunc', 'oracle_raw_trunc',
    # Suffix gradient (7)
    'random_suffix', 'random_words_suffix', 'wrong_doc_llm_suffix',
    'tfidf_kw_suffix', 'llm_kw_suffix', 'static_fact_suffix',
    'oracle_kw_suffix',
]

# Semantic levels for gradient analysis
SEMANTIC_LEVELS = {
    'random': 0, 'random_words': 0.5, 'wrong_doc_llm': 1,
    'tfidf_kw': 2, 'llm_kw': 3, 'static_fact': 4, 'oracle_kw': 5,
}

# --- Stopwords for TF-IDF and oracle-as-keywords ---
STOPWORDS = set([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
    'where', 'why', 'how', 'what', 'if', 'up', 'also', 'well', 'back',
    'even', 'still', 'new', 'now', 'way', 'many', 'much', 'like', 'get',
    'got', 'make', 'made', 'take', 'come', 'go', 'see', 'know', 'think',
])

QUESTION_STOPWORDS = STOPWORDS | set([
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'does', 'did', 'can', 'could', 'would', 'should', 'will', 'shall',
    'may', 'might', 'must', 'isn', 'aren', 'wasn', 'weren', 'don', 'doesn',
    'didn', 'won', 'wouldn', 'couldn', 'shouldn',
])

# --- Common English words for random_words condition ---
COMMON_ENGLISH_WORDS = [
    "apple", "river", "mountain", "table", "chair", "window", "garden", "flower",
    "music", "dance", "piano", "guitar", "forest", "ocean", "desert", "island",
    "bridge", "castle", "village", "market", "kitchen", "bedroom", "library",
    "hospital", "church", "stadium", "airport", "highway", "bicycle", "telephone",
    "calendar", "newspaper", "magazine", "photograph", "umbrella", "birthday",
    "holiday", "vacation", "weekend", "summer", "winter", "autumn", "spring",
    "morning", "evening", "midnight", "afternoon", "sunrise", "sunset", "rainbow",
    "thunder", "lightning", "earthquake", "volcano", "diamond", "crystal", "silver",
    "golden", "copper", "bronze", "wooden", "plastic", "rubber", "leather", "cotton",
    "marble", "granite", "concrete", "gravel", "pebble", "boulder", "cliff",
    "valley", "meadow", "jungle", "canyon", "glacier", "waterfall", "harbor",
    "elephant", "dolphin", "penguin", "parrot", "butterfly", "crocodile",
    "salmon", "turtle", "spider", "mosquito", "sandwich", "chocolate", "vanilla",
    "cinnamon", "pepper", "mushroom", "tomato", "potato", "banana", "strawberry",
    "blanket", "pillow", "curtain", "mirror", "compass", "telescope", "microscope",
    "battery", "engine", "propeller", "satellite", "oxygen", "hydrogen", "nitrogen",
    "calcium", "protein", "vitamin", "bacteria", "molecule", "equation", "triangle",
    "rectangle", "cylinder", "sphere", "pentagon", "diameter", "fraction", "decimal",
    "giraffe", "kangaroo", "flamingo", "orchestra", "symphony", "painting",
    "sculpture", "pottery", "costume", "jewelry", "bracelet", "necklace",
    "backpack", "suitcase", "envelope", "receipt", "passport", "notebook",
    "keyboard", "monitor", "speaker", "printer", "cabinet", "corridor",
]


def extract_tfidf_keywords(passage, n_keywords=8):
    \"\"\"Extract top content words by frequency (stopwords removed).\"\"\"
    words = re.findall(r'\\b[a-zA-Z]+\\b', passage.lower())
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join([w for w, _ in Counter(content_words).most_common(n_keywords)])


def oracle_to_keywords(query):
    \"\"\"Strip question/function words from oracle query to get keyword format.\"\"\"
    words = re.findall(r'\\b[a-zA-Z]+\\b', query)
    return ' '.join([w for w in words if w.lower() not in QUESTION_STOPWORDS and len(w) > 2])


def generate_random_words(rng, n_words=8):
    \"\"\"Generate n random English words from the common words list.\"\"\"
    indices = rng.randint(0, len(COMMON_ENGLISH_WORDS), size=n_words)
    return ' '.join(COMMON_ENGLISH_WORDS[i] for i in indices)


def build_primed_and_truncated(prefix_text, bos_id, doc_ids, doc_len, model, tokenizer, config):
    \"\"\"Build a primed cache: tokenize prefix, concat [BOS][prefix][doc], forward, truncate+RoPE.

    Returns:
        (trunc_cache, prefix_token_len) where prefix_token_len includes BOS
    \"\"\"
    prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=prefix_text)
    prefix_enc = tokenizer(prefix_str, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
    prefix_ids = prefix_enc['input_ids'].to(config.device)
    prefix_token_len = 1 + prefix_ids.shape[1]  # BOS + prefix tokens

    full_ids = torch.cat([bos_id, prefix_ids, doc_ids], dim=1)

    with torch.no_grad():
        out = model(input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=True, return_dict=True)

    trunc_cache = extract_and_truncate_cache_with_bos(out.past_key_values, doc_len)
    correct_rope_positions_with_bos(trunc_cache, prefix_token_len - 1, model)

    del out
    return trunc_cache, prefix_token_len


print("Config ready")
print(f"  num_samples pool: {config.num_samples}")
print(f"  eval samples: {N_EVAL}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: {len(CONDITION_NAMES)}")
print(f"  suffix_separator: '{SUFFIX_SEPARATOR}'")
print(f"  static_factual_phrase: '{STATIC_FACTUAL_PHRASE}'")
print(f"  common_english_words: {len(COMMON_ENGLISH_WORDS)} words")
print(f"  semantic_levels: {SEMANTIC_LEVELS}")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (1000 samples)
dataset = load_ms_marco(config)

np.random.seed(SEED)
all_samples = load_evaluation_samples(dataset, config, require_answer=True)

samples = all_samples[:N_EVAL]
N = len(samples)
print(f"Loaded {len(all_samples)} candidates, using first {N} for evaluation")
print(f"Example passage ({len(samples[0]['passage'].split())} words): {samples[0]['passage'][:100]}...")
print(f"Example query: {samples[0]['query']}")
print(f"Example answer: {samples[0]['answer']}")\
""")))

# ========== Cell 5: Generate LLM keyword surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 5: Generate LLM keyword surrogates (fresh, independent)
print("=" * 70)
print("PHASE 1: LLM SURROGATE GENERATION (keyword only)")
print("=" * 70)

surrogates_path = SURROGATES_DIR / "keyword_surrogates.json"

if surrogates_path.exists():
    with open(surrogates_path, 'r') as f:
        surrogates_data = json.load(f)
    keyword_surrogates = surrogates_data['surrogates']
    print(f"Loaded {len(keyword_surrogates)} keyword surrogates from cache")
else:
    keyword_surrogates = []

start_idx_gen = len(keyword_surrogates)
if start_idx_gen < N:
    print(f"Generating keyword surrogates for samples {start_idx_gen} to {N-1}...")
    t_start = time.time()
    for idx in tqdm(range(start_idx_gen, N), initial=start_idx_gen, total=N,
                     desc="Keyword surrogates"):
        passage = samples[idx]['passage']
        try:
            s5 = generate_all_5_surrogates(passage, model, tokenizer, config)
            kw = s5.get('keyword_query', '')
        except Exception as e:
            print(f"  WARNING: Generation failed for sample {idx}: {e}")
            kw = ""
        keyword_surrogates.append(kw)

        if (idx + 1) % 100 == 0 or idx == N - 1:
            with open(surrogates_path, 'w') as f:
                json.dump({'surrogates': keyword_surrogates}, f)
            elapsed = time.time() - t_start
            rate = (idx - start_idx_gen + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"  Saved {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

    with open(surrogates_path, 'w') as f:
        json.dump({'surrogates': keyword_surrogates}, f)
    print(f"Keyword surrogates complete: {len(keyword_surrogates)} samples")
else:
    print(f"All keyword surrogates already cached ({len(keyword_surrogates)} samples)")

n_empty = sum(1 for s in keyword_surrogates if not s.strip())
print(f"Empty surrogates: {n_empty}/{N}")
print(f"Example: '{keyword_surrogates[0]}'")\
""")))

# ========== Cell 6: Pre-compute derived surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 6: Pre-compute TF-IDF keywords and oracle-as-keywords for all samples
print("=" * 70)
print("PRE-COMPUTING DERIVED SURROGATES")
print("=" * 70)

tfidf_keywords = []
oracle_keywords = []

for idx in range(N):
    passage = samples[idx]['passage']
    query = samples[idx]['query']

    tfidf_keywords.append(extract_tfidf_keywords(passage, n_keywords=8))
    oracle_keywords.append(oracle_to_keywords(query))

# Diagnostics
print(f"\\nTF-IDF keywords ({N} samples):")
print(f"  Example 0: '{tfidf_keywords[0]}'")
print(f"  Example 1: '{tfidf_keywords[1]}'")
print(f"  Empty: {sum(1 for t in tfidf_keywords if not t.strip())}/{N}")

print(f"\\nOracle-as-keywords ({N} samples):")
print(f"  Example 0: '{oracle_keywords[0]}'  (from: '{samples[0]['query']}')")
print(f"  Example 1: '{oracle_keywords[1]}'  (from: '{samples[1]['query']}')")
print(f"  Empty: {sum(1 for o in oracle_keywords if not o.strip())}/{N}")

# Token length comparison
llm_lens = [len(tokenizer.encode(kw, add_special_tokens=False)) for kw in keyword_surrogates]
tfidf_lens = [len(tokenizer.encode(kw, add_special_tokens=False)) for kw in tfidf_keywords]
oracle_kw_lens = [len(tokenizer.encode(kw, add_special_tokens=False)) for kw in oracle_keywords]
static_len = len(tokenizer.encode(STATIC_FACTUAL_PHRASE, add_special_tokens=False))

print(f"\\nToken lengths (mean ± std):")
print(f"  LLM keywords: {np.mean(llm_lens):.1f} ± {np.std(llm_lens):.1f}")
print(f"  TF-IDF keywords: {np.mean(tfidf_lens):.1f} ± {np.std(tfidf_lens):.1f}")
print(f"  Oracle-as-keywords: {np.mean(oracle_kw_lens):.1f} ± {np.std(oracle_kw_lens):.1f}")
print(f"  Static factual: {static_len}")\
""")))

# ========== Cell 7: Condition explanation ==========
cells.append(make_cell("code", s("""\
# Cell 7: Condition explanation with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

ex = {
    'passage': samples[0]['passage'][:80],
    'query': samples[0]['query'],
    'llm_kw': keyword_surrogates[0],
    'tfidf_kw': tfidf_keywords[0],
    'oracle_kw': oracle_keywords[0],
    'random_words': generate_random_words(np.random.RandomState(SEED), 8),
}

conditions_explained = [
    ("1. bare",
     "[BOS][doc]",
     "No prefix — baseline"),
    ("2. random_trunc",
     "[BOS][random_tokens\\\\n][doc] → truncate + RoPE",
     "Random vocabulary tokens. Semantic level: 0"),
    ("3. random_words_trunc",
     "[BOS][random_words\\\\n][doc] → truncate + RoPE",
     f"Random English words: '{ex['random_words']}'. Semantic level: 0.5"),
    ("4. wrong_doc_llm_trunc",
     "[BOS][prev_kw\\\\n][doc] → truncate + RoPE",
     "LLM keywords from PREVIOUS sample's doc. Right format, wrong content. Semantic level: 1"),
    ("5. tfidf_kw_trunc",
     "[BOS][tfidf\\\\n][doc] → truncate + RoPE",
     f"TF-IDF keywords: '{ex['tfidf_kw']}'. Semantic level: 2"),
    ("6. llm_kw_trunc",
     "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     f"LLM-generated keywords: '{ex['llm_kw']}'. Semantic level: 3"),
    ("7. static_fact_trunc",
     "[BOS][static_fact\\\\n][doc] → truncate + RoPE",
     f"Fixed phrase: '{STATIC_FACTUAL_PHRASE}'. Semantic level: 4"),
    ("8. oracle_kw_trunc",
     "[BOS][oracle_kw\\\\n][doc] → truncate + RoPE",
     f"Oracle as keywords: '{ex['oracle_kw']}'. Semantic level: 5"),
    ("9. oracle_raw_trunc",
     "[BOS][oracle_raw\\\\n][doc] → truncate + RoPE",
     f"Oracle raw question: '{ex['query']}'. Level 5 but with format interference"),
    ("10-16. *_suffix",
     "[BOS][doc][sep][content]",
     "Same 7 content types as suffix after doc. No value contamination; pure attention."),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")

print(f"\\n{'='*70}")
print("FORWARD PASSES PER SAMPLE: 16")
print("  Truncated (9): bare + 8 prefix types")
print("  Suffix (7): 7 suffix types via build_suffix_kv_cache")
print("  Total scoring calls: 16 (one per condition)")
print(f"{'='*70}")\
""")))

# ========== Cell 8: Main eval loop ==========
cells.append(make_cell("code", s("""\
# Cell 8: Main eval loop — 16 conditions × 1000 samples
print("=" * 70)
print("PHASE 2: MAIN EVALUATION (16 conditions × 1000 samples)")
print("=" * 70)

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in samples]
    if ckpt_queries == current_queries:
        results = ckpt['results']
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{N}")
    else:
        print("Checkpoint sample mismatch. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

print(f"Evaluating samples {start_idx} to {N-1}")
print(f"Conditions: {len(CONDITION_NAMES)}")

t_start = time.time()

for idx in tqdm(range(start_idx, N), initial=start_idx, total=N, desc="Evaluating"):
    sample = samples[idx]
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']

    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    # Get all content strings for this sample
    llm_kw_text = keyword_surrogates[idx]
    tfidf_kw_text = tfidf_keywords[idx]
    oracle_kw_text = oracle_keywords[idx]
    oracle_raw_text = query  # original question format

    # Wrong-doc: use previous sample's LLM keyword
    if idx > 0:
        wrong_doc_text = keyword_surrogates[idx - 1]
    else:
        wrong_doc_text = ""  # sample 0: handled below

    # Random tokens (deterministic per sample)
    n_random_tokens = max(5, len(tokenizer.encode(llm_kw_text, add_special_tokens=False)))
    rng_tokens = np.random.RandomState(SEED + idx)
    random_ids = torch.randint(100, tokenizer.vocab_size - 100, (n_random_tokens,), device='cpu')
    random_text = tokenizer.decode(random_ids, skip_special_tokens=True)

    # Random English words (deterministic per sample)
    rng_words = np.random.RandomState(SEED + idx + 10000)
    random_words_text = generate_random_words(rng_words, n_words=8)

    # --- Matched tokenization (for truncated conditions) ---
    oracle_prefix = SURROGATE_PREFIX_TEMPLATE.format(surrogate=query)
    document_text = DOCUMENT_TEMPLATE.format(document=passage)
    full_oracle_text = oracle_prefix + document_text

    full_oracle_enc = tokenizer(full_oracle_text, return_tensors="pt",
                                add_special_tokens=True, padding=False, truncation=False)
    full_oracle_ids = full_oracle_enc['input_ids'].to(config.device)

    oracle_prefix_enc = tokenizer(oracle_prefix, return_tensors="pt",
                                  add_special_tokens=True, padding=False, truncation=False)
    oracle_prefix_len = oracle_prefix_enc['input_ids'].shape[1]

    bos_id = full_oracle_ids[:, :1]
    doc_ids = full_oracle_ids[:, oracle_prefix_len:]
    doc_len = doc_ids.shape[1]

    # ===== FORWARD PASS 1: BARE =====
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    bare_cache = bare_out.past_key_values
    del bare_out

    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_cache), bare_ids.shape[1],
        query_prompt, answer_text, model, tokenizer, config)

    # ===== TRUNCATED CONDITIONS (8 forward passes) =====

    # 2. random_trunc
    trunc_cache, _ = build_primed_and_truncated(
        random_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_random_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 3. random_words_trunc
    trunc_cache, _ = build_primed_and_truncated(
        random_words_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_random_words_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 4. wrong_doc_llm_trunc (sample 0: NLL=0)
    if idx > 0:
        trunc_cache, _ = build_primed_and_truncated(
            wrong_doc_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
        nll_wrong_doc_trunc = score_answer_with_cache(
            deepcopy_cache(trunc_cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del trunc_cache
    else:
        nll_wrong_doc_trunc = 0.0

    # 5. tfidf_kw_trunc
    trunc_cache, _ = build_primed_and_truncated(
        tfidf_kw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_tfidf_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 6. llm_kw_trunc
    trunc_cache, _ = build_primed_and_truncated(
        llm_kw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_llm_kw_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 7. static_fact_trunc
    trunc_cache, _ = build_primed_and_truncated(
        STATIC_FACTUAL_PHRASE, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_static_fact_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 8. oracle_kw_trunc
    trunc_cache, _ = build_primed_and_truncated(
        oracle_kw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_oracle_kw_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    # 9. oracle_raw_trunc
    trunc_cache, _ = build_primed_and_truncated(
        oracle_raw_text, bos_id, doc_ids, doc_len, model, tokenizer, config)
    nll_oracle_raw_trunc = score_answer_with_cache(
        deepcopy_cache(trunc_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del trunc_cache

    del bare_cache, bare_ids
    torch.cuda.empty_cache()

    # ===== SUFFIX CONDITIONS (7 forward passes) =====

    # 10. random_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, random_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_random_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # 11. random_words_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, random_words_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_random_words_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # 12. wrong_doc_llm_suffix (sample 0: NLL=0)
    if idx > 0:
        suf_len, suf_cache = build_suffix_kv_cache(
            passage, wrong_doc_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
        nll_wrong_doc_suffix = score_answer_with_cache(
            deepcopy_cache(suf_cache), suf_len,
            query_prompt, answer_text, model, tokenizer, config)
        del suf_cache
    else:
        nll_wrong_doc_suffix = 0.0

    # 13. tfidf_kw_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, tfidf_kw_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_tfidf_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # 14. llm_kw_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, llm_kw_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_llm_kw_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # 15. static_fact_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, STATIC_FACTUAL_PHRASE, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_static_fact_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    # 16. oracle_kw_suffix
    suf_len, suf_cache = build_suffix_kv_cache(
        passage, oracle_kw_text, model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_oracle_kw_suffix = score_answer_with_cache(
        deepcopy_cache(suf_cache), suf_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suf_cache

    torch.cuda.empty_cache()

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len': doc_len,
        'passage_word_count': len(passage.split()),
        'bare': nll_bare,
        'random_trunc': nll_random_trunc,
        'random_words_trunc': nll_random_words_trunc,
        'wrong_doc_llm_trunc': nll_wrong_doc_trunc,
        'tfidf_kw_trunc': nll_tfidf_trunc,
        'llm_kw_trunc': nll_llm_kw_trunc,
        'static_fact_trunc': nll_static_fact_trunc,
        'oracle_kw_trunc': nll_oracle_kw_trunc,
        'oracle_raw_trunc': nll_oracle_raw_trunc,
        'random_suffix': nll_random_suffix,
        'random_words_suffix': nll_random_words_suffix,
        'wrong_doc_llm_suffix': nll_wrong_doc_suffix,
        'tfidf_kw_suffix': nll_tfidf_suffix,
        'llm_kw_suffix': nll_llm_kw_suffix,
        'static_fact_suffix': nll_static_fact_suffix,
        'oracle_kw_suffix': nll_oracle_kw_suffix,
    }
    results.append(result)

    if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == N - 1:
        ckpt_data = {
            'results': results,
            'sample_queries': [s['query'] for s in samples],
            'completed': len(results),
            'total': N,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)
        elapsed = time.time() - t_start
        rate = (idx - start_idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (N - idx - 1) / rate if rate > 0 else 0
        tqdm.write(f"  Checkpoint {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\\nEvaluation complete: {len(results)} samples in {elapsed_total/60:.1f} min")\
""")))

# ========== Cell 9: Primary analysis ==========
cells.append(make_cell("code", s("""\
# Cell 9: Primary analysis — NLL summary + 10 comparisons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — SEMANTIC CONTENT GRADIENT")
print("=" * 70)

# Extract arrays and filter zero NLLs
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

valid = np.ones(len(results), dtype=bool)
for cname in CONDITION_NAMES:
    valid &= (cond_arrays[cname] != 0)
n_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total: {len(results)}, Valid: {n_valid}, Excluded: {n_excluded}")

c = {}
for cname in CONDITION_NAMES:
    c[cname] = cond_arrays[cname][valid]

# NLL summary table
print(f"\\n{'Condition':<30} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10} {'Win%':>7}")
print("-" * 72)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        print(f"{cname:<30} {mean_nll:>10.4f} {std_nll:>10.4f} {'—':>10} {'—':>7}")
    else:
        delta = c['bare'] - c[cname]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        _, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
        print(f"{cname:<30} {mean_nll:>10.4f} {std_nll:>10.4f} {d:>+10.3f} {win:>5.1f}% {sig}")

# 10 primary comparisons
print(f"\\n{'='*95}")
print(f"10 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*95}")

comparisons = [
    ('C1: llm_kw vs random (trunc)',
     c['random_trunc'] - c['llm_kw_trunc'],
     'LLM > random in truncated?'),
    ('C2: static_fact vs random (trunc)',
     c['random_trunc'] - c['static_fact_trunc'],
     'Static > random in truncated?'),
    ('C3: llm_kw vs wrong_doc (trunc)',
     c['wrong_doc_llm_trunc'] - c['llm_kw_trunc'],
     'Right doc > wrong doc (trunc)?'),
    ('C4: tfidf vs random (trunc)',
     c['random_trunc'] - c['tfidf_kw_trunc'],
     'TF-IDF > random in truncated?'),
    ('C5: oracle_kw vs oracle_raw (trunc)',
     c['oracle_raw_trunc'] - c['oracle_kw_trunc'],
     'Keyword > question format?'),
    ('C6: llm_kw vs random (suffix)',
     c['random_suffix'] - c['llm_kw_suffix'],
     'LLM > random in suffix?'),
    ('C7: static_fact vs random (suffix)',
     c['random_suffix'] - c['static_fact_suffix'],
     'Static > random in suffix?'),
    ('C8: llm_kw vs wrong_doc (suffix)',
     c['wrong_doc_llm_suffix'] - c['llm_kw_suffix'],
     'Right doc > wrong doc (suffix)?'),
    ('C9: trunc vs suffix (llm_kw)',
     c['llm_kw_suffix'] - c['llm_kw_trunc'],
     'Truncated > suffix (LLM content)?'),
    ('C10: trunc vs suffix (static_fact)',
     c['static_fact_suffix'] - c['static_fact_trunc'],
     'Truncated > suffix (static)?'),
]

print(f"\\n{'Comparison':<40} {'Mean delta':>10} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 95)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<40} {np.mean(delta):>10.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        'question': question,
    }

# All vs Bare
print(f"\\n{'='*95}")
print("ALL CONDITIONS vs BARE (sorted by d)")
print(f"{'='*95}")
all_vs_bare = {}
all_conds = [(cn, cohens_d(c['bare'] - c[cn])) for cn in CONDITION_NAMES if cn != 'bare']
all_conds.sort(key=lambda x: x[1], reverse=True)
print(f"\\n{'Condition':<30} {'d vs Bare':>10} {'Win%':>7} {'p':>12}")
print("-" * 65)
for cname, d in all_conds:
    delta = c['bare'] - c[cname]
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname:<30} {d:>10.3f} {win:>6.1f}% {p_val:>11.2e} {sig:>5}")
    all_vs_bare[cname] = {'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val)}\
""")))

# ========== Cell 10: Gradient analysis + hardness ==========
cells.append(make_cell("code", s("""\
# Cell 10: Semantic gradient analysis + hardness breakdown

# --- SEMANTIC GRADIENT ---
print("=" * 70)
print("SEMANTIC GRADIENT ANALYSIS")
print("=" * 70)

# Build gradient data for both modes
content_types = ['random', 'random_words', 'wrong_doc_llm', 'tfidf_kw', 'llm_kw', 'static_fact', 'oracle_kw']
levels = [SEMANTIC_LEVELS[ct] for ct in content_types]

trunc_ds = []
suffix_ds = []
for ct in content_types:
    trunc_d = cohens_d(c['bare'] - c[f'{ct}_trunc'])
    suffix_d = cohens_d(c['bare'] - c[f'{ct}_suffix'])
    trunc_ds.append(trunc_d)
    suffix_ds.append(suffix_d)

print(f"\\n{'Content Type':<20} {'Level':>6} {'Trunc d':>10} {'Suffix d':>10} {'Trunc > Suffix':>15}")
print("-" * 65)
for ct, lev, td, sd in zip(content_types, levels, trunc_ds, suffix_ds):
    trunc_better = "YES" if td > sd else "no"
    print(f"{ct:<20} {lev:>6.1f} {td:>+10.3f} {sd:>+10.3f} {trunc_better:>15}")

# Correlation: semantic level vs d
from scipy.stats import spearmanr

r_trunc, p_trunc = spearmanr(levels, trunc_ds)
r_suffix, p_suffix = spearmanr(levels, suffix_ds)
print(f"\\nSemantic level vs d correlation:")
print(f"  Truncated: Spearman r = {r_trunc:+.3f}, p = {p_trunc:.4f}")
print(f"  Suffix:    Spearman r = {r_suffix:+.3f}, p = {p_suffix:.4f}")

# Also check: do both modes agree on ranking?
trunc_rank = np.argsort(np.argsort(trunc_ds))
suffix_rank = np.argsort(np.argsort(suffix_ds))
r_cross, p_cross = spearmanr(trunc_rank, suffix_rank)
print(f"  Cross-mode rank agreement: Spearman r = {r_cross:+.3f}, p = {p_cross:.4f}")

# Oracle raw vs oracle kw (format effect)
d_oracle_raw = cohens_d(c['bare'] - c['oracle_raw_trunc'])
d_oracle_kw = cohens_d(c['bare'] - c['oracle_kw_trunc'])
print(f"\\nFormat effect (truncated only):")
print(f"  Oracle-as-keywords: d = {d_oracle_kw:+.3f}")
print(f"  Oracle-raw-question: d = {d_oracle_raw:+.3f}")
print(f"  Format penalty: d = {d_oracle_raw - d_oracle_kw:+.3f}")

# --- HARDNESS QUINTILE BREAKDOWN ---
print(f"\\n{'='*70}")
print("HARDNESS QUINTILE BREAKDOWN")
print(f"{'='*70}")

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easy)', 'Q2', 'Q3', 'Q4', 'Q5 (hard)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

# Key conditions for hardness analysis
key_conds = [
    'random_trunc', 'random_words_trunc', 'wrong_doc_llm_trunc',
    'tfidf_kw_trunc', 'llm_kw_trunc', 'static_fact_trunc', 'oracle_kw_trunc',
    'random_suffix', 'llm_kw_suffix', 'static_fact_suffix', 'oracle_kw_suffix',
]

header = f"{'Condition':<30}" + "".join(f"{ql:>12}" for ql in quintile_labels) + f"{'Overall':>12}"
print(f"\\n{header}")
print("-" * (30 + 12 * 6))

hardness_breakdown = {}
for cname in key_conds:
    row = f"{cname:<30}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>12}"
            quintile_ds.append(None)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            d = cohens_d(delta)
            row += f"{d:>+12.3f}"
            quintile_ds.append(float(d))
    d_all = cohens_d(bare_valid - c[cname])
    row += f"{d_all:>+12.3f}"
    print(row)
    hardness_breakdown[cname] = {'quintile_ds': quintile_ds, 'overall_d': float(d_all)}

# Verdict
print(f"\\n{'='*70}")
print("VERDICT: DOES SEMANTIC CONTENT MATTER?")
print(f"{'='*70}")

# Check if gradient exists in both modes
trunc_gradient = trunc_ds[-1] - trunc_ds[0]  # oracle_kw - random
suffix_gradient = suffix_ds[-1] - suffix_ds[0]
both_positive = trunc_gradient > 0 and suffix_gradient > 0

# Check key comparisons
c1_sig = comparison_results['C1: llm_kw vs random (trunc)']['bonferroni_significant']
c2_sig = comparison_results['C2: static_fact vs random (trunc)']['bonferroni_significant']
c6_sig = comparison_results['C6: llm_kw vs random (suffix)']['bonferroni_significant']
c7_sig = comparison_results['C7: static_fact vs random (suffix)']['bonferroni_significant']

print(f"\\n  Truncated gradient (oracle_kw - random): {trunc_gradient:+.3f}")
print(f"  Suffix gradient (oracle_kw - random):    {suffix_gradient:+.3f}")
print(f"  Both positive: {'YES' if both_positive else 'NO'}")
print(f"\\n  C1 (llm_kw > random, trunc): {'YES ***' if c1_sig else 'NO (ns)'}")
print(f"  C2 (static > random, trunc):  {'YES ***' if c2_sig else 'NO (ns)'}")
print(f"  C6 (llm_kw > random, suffix): {'YES ***' if c6_sig else 'NO (ns)'}")
print(f"  C7 (static > random, suffix): {'YES ***' if c7_sig else 'NO (ns)'}")
print(f"\\n  Semantic level vs d (trunc): r = {r_trunc:+.3f} (p = {p_trunc:.4f})")
print(f"  Semantic level vs d (suffix): r = {r_suffix:+.3f} (p = {p_suffix:.4f})")

if both_positive and c1_sig and c6_sig:
    print(f"\\n  *** CONCLUSION: SEMANTIC CONTENT MATTERS ***")
    print(f"  The semantic gradient is positive in BOTH truncated and suffix modes.")
    print(f"  LLM > random is significant in BOTH modes.")
    print(f"  This cannot be explained by structural artifacts alone.")
elif c2_sig and c7_sig:
    print(f"\\n  *** CONCLUSION: STATIC FACTUAL PHRASE IS SPECIAL ***")
    print(f"  Static > random in both modes, but LLM may not beat random in suffix.")
else:
    print(f"\\n  CONCLUSION: MIXED EVIDENCE")
    print(f"  The gradient is not consistently significant across modes.")\
""")))

# ========== Cell 11: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 11: Plots (2x3 grid)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# Color scheme: truncated = blue shades, suffix = red shades
trunc_color = '#2166ac'
suffix_color = '#b2182b'

# --- Plot 1: All conditions bar chart (sorted by d) ---
ax = axes[0, 0]
all_sorted = sorted(
    [(cn, cohens_d(c['bare'] - c[cn])) for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda x: x[1], reverse=True
)
names_sorted = [x[0] for x in all_sorted]
ds_sorted = [x[1] for x in all_sorted]
colors_bar = [trunc_color if '_trunc' in cn else suffix_color if '_suffix' in cn else 'gray'
              for cn in names_sorted]
ax.barh(range(len(names_sorted)), ds_sorted, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=7)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare')
ax.invert_yaxis()

# --- Plot 2: Semantic Gradient — Truncated Mode ---
ax = axes[0, 1]
x_pos = range(len(content_types))
ax.bar(x_pos, trunc_ds, color=trunc_color, edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([ct.replace('_', '\\n') for ct in content_types], fontsize=7, rotation=45, ha='right')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Semantic Gradient — Truncated Prefix')
for i, v in enumerate(trunc_ds):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=7)

# --- Plot 3: Semantic Gradient — Suffix Mode ---
ax = axes[0, 2]
ax.bar(x_pos, suffix_ds, color=suffix_color, edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([ct.replace('_', '\\n') for ct in content_types], fontsize=7, rotation=45, ha='right')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Semantic Gradient — Suffix Mode')
for i, v in enumerate(suffix_ds):
    ax.text(i, v + 0.005, f"{v:+.3f}", ha='center', va='bottom', fontsize=7)

# --- Plot 4: Overlay — Both modes on same plot ---
ax = axes[1, 0]
width = 0.35
x = np.arange(len(content_types))
bars1 = ax.bar(x - width/2, trunc_ds, width, color=trunc_color, alpha=0.8, label='Truncated', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, suffix_ds, width, color=suffix_color, alpha=0.8, label='Suffix', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([ct.replace('_', '\\n') for ct in content_types], fontsize=7, rotation=45, ha='right')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Truncated vs Suffix — Same Content')
ax.legend(fontsize=9)

# --- Plot 5: Semantic level scatter with fit lines ---
ax = axes[1, 1]
ax.scatter(levels, trunc_ds, color=trunc_color, s=80, zorder=5, label='Truncated')
ax.scatter(levels, suffix_ds, color=suffix_color, s=80, zorder=5, marker='s', label='Suffix')
# Fit lines
z_trunc = np.polyfit(levels, trunc_ds, 1)
z_suffix = np.polyfit(levels, suffix_ds, 1)
x_fit = np.linspace(0, 5, 50)
ax.plot(x_fit, np.polyval(z_trunc, x_fit), '--', color=trunc_color, alpha=0.5,
        label=f'Trunc fit (slope={z_trunc[0]:.3f})')
ax.plot(x_fit, np.polyval(z_suffix, x_fit), '--', color=suffix_color, alpha=0.5,
        label=f'Suffix fit (slope={z_suffix[0]:.3f})')
for ct, lev, td, sd in zip(content_types, levels, trunc_ds, suffix_ds):
    ax.annotate(ct.replace('_', '\\n'), (lev, td), textcoords="offset points",
                xytext=(5, 5), fontsize=6)
ax.set_xlabel('Semantic Level')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('Semantic Level vs Effect Size')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 6: Hardness × condition heatmap ---
ax = axes[1, 2]
hm_conds = ['random_trunc', 'wrong_doc_llm_trunc', 'tfidf_kw_trunc',
            'llm_kw_trunc', 'static_fact_trunc', 'oracle_kw_trunc',
            'random_suffix', 'llm_kw_suffix', 'static_fact_suffix']
hm_data = []
for cname in hm_conds:
    row = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row.append(0)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            row.append(cohens_d(delta))
    hm_data.append(row)
hm_data = np.array(hm_data)
im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.5, vmax=0.7, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=7)
ax.set_yticks(range(len(hm_conds)))
ax.set_yticklabels(hm_conds, fontsize=7)
for i in range(len(hm_conds)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness × Condition')

plt.suptitle('Exp 10: Semantic Content Gradient', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 12: Save results JSON ==========
cells.append(make_cell("code", s("""\
# Cell 12: Save comprehensive results JSON

# Gradient summaries
gradient_trunc = {}
gradient_suffix = {}
for ct, lev, td, sd in zip(content_types, levels, trunc_ds, suffix_ds):
    gradient_trunc[ct] = {'semantic_level': lev, 'cohens_d': float(td)}
    gradient_suffix[ct] = {'semantic_level': lev, 'cohens_d': float(sd)}

final = {
    'experiment': 'exp10_semantic_content_gradient',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': config.model_name,
        'seed': SEED,
        'n_eval': N,
        'n_valid': n_valid,
        'n_excluded': n_excluded,
        'min_passage_words': config.min_passage_words,
        'max_passage_words': config.max_passage_words,
        'n_conditions': len(CONDITION_NAMES),
        'n_comparisons': N_COMPARISONS,
        'bonferroni_alpha': BONFERRONI_ALPHA,
        'suffix_separator': SUFFIX_SEPARATOR,
    },
    'condition_names': CONDITION_NAMES,
    'nll_summary': {
        cname: {
            'mean': float(np.mean(c[cname])),
            'std': float(np.std(c[cname])),
            'cohens_d_vs_bare': float(cohens_d(c['bare'] - c[cname])) if cname != 'bare' else 0.0,
        }
        for cname in CONDITION_NAMES
    },
    'gradient_summaries': {
        'truncated': gradient_trunc,
        'suffix': gradient_suffix,
        'spearman_trunc': {'r': float(r_trunc), 'p': float(p_trunc)},
        'spearman_suffix': {'r': float(r_suffix), 'p': float(p_suffix)},
        'cross_mode_rank_agreement': {'r': float(r_cross), 'p': float(p_cross)},
    },
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'hardness_breakdown': hardness_breakdown,
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 13: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 13: GPU cleanup
import gc

print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9

del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()
gc.collect()

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Cleanup complete.")\
""")))

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "cells": cells
}

output_path = "/home/jupyter/research/directed_kvcache_v2/10_semantic_content_gradient.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Cells: {len(cells)}")
