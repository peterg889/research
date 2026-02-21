#!/usr/bin/env python3
"""Build script for 06_surrogate_deep_dive.ipynb"""

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
    # Remove trailing empty string if present
    if result and result[-1] == '':
        result = result[:-1]
    return result

cells = []

# ========== Cell 0: Markdown overview ==========
cells.append(make_cell("markdown", s("""\
# Exp 06: LLM Surrogate Deep-Dive — Mechanism Decomposition

## Motivation

Exp 05 found LLM-generated keyword surrogates (d=0.37 vs bare) massively outperform oracle
queries (d=0.13, indistinguishable from random) on hard MS MARCO. But **WHY?** The LLM
surrogates look like "Cumulonimbus clouds height atmosphere" — keyword-dense topic phrases
with high vocabulary overlap with the passage. Oracle queries are natural-language questions
with lower overlap.

This experiment decomposes the mechanism through controlled ablations on the **full MS MARCO
distribution** (2000 samples, not hardness-filtered).

## Hypotheses

1. **Token Overlap** — LLM surrogates share tokens with the passage, creating coherent value contamination.
2. **Coherence** — Token ORDER matters beyond identity.
3. **Format** — Question syntax hurts (question words create unhelpful attention patterns).
4. **Passage Specificity** — Document-specific content words help more than generic content words.
5. **Mechanism Stacking** — Truncated prefix + separator suffix should combine.

## 15 Conditions

| # | Condition | Type | Tests |
|---|-----------|------|-------|
| 1 | Bare | Baseline | — |
| 2 | Random-truncated | Control | Structural control |
| 3 | Separator-only | Control | Suffix framing |
| 4 | Oracle-truncated | Oracle | Semantic control |
| 5 | Oracle-as-keywords | Oracle | Format (H3) |
| 6 | Anti-keywords | Overlap | Specificity (H4) |
| 7 | TF-IDF-keywords | Overlap | Cheap surrogate |
| 8 | Passage-echo | Overlap | Overlap ceiling |
| 9 | Shuffled-LLM | Overlap | Coherence (H2) |
| 10 | LLM-keyword | LLM | Keyword template |
| 11 | LLM-question | LLM | Question template |
| 12 | LLM-symptom | LLM | Symptom template |
| 13 | LLM-summary | LLM | Summary template |
| 14 | LLM-keyword+sep | Production | Stacking (H5) |
| 15 | LLM-messy | Production | Informal style |

## 10 Primary Comparisons (Bonferroni alpha = 0.005)

- M1: Shuffled-LLM vs LLM-keyword (coherence)
- M2: Oracle-as-keywords vs Oracle (format)
- M3: LLM-keyword vs LLM-question (keyword vs question for LLM)
- M4: TF-IDF vs Anti-keywords (specificity)
- M5: Passage-echo vs LLM-keyword (max overlap ceiling)
- M6: TF-IDF vs LLM-keyword (is LLM necessary?)
- M7: LLM-keyword+sep vs max(LLM-keyword, sep-only) (stacking)
- R1: LLM-keyword vs Random (replicates Exp 05)
- R2: Oracle vs Random (replicates null)
- R3: Template ranking""")))

# ========== Cell 1: Setup ==========
cells.append(make_cell("code", s("""\
# Cell 1: Setup — permissions, seeds, results directory
import os
os.umask(0o000)

import sys
import json
import time
import re
import numpy as np
import torch
from pathlib import Path
from collections import Counter

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("results/exp06")
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

# ========== Cell 3: Imports + config + templates + all helper functions ==========
cells.append(make_cell("code", s("""\
# Cell 3: Imports + config + templates + all helper functions
sys.path.insert(0, ".")

from lib.config import ExperimentConfig
from lib.kv_cache import (
    build_kv_cache,
    build_suffix_kv_cache,
    score_answer_with_cache,
    deepcopy_cache,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
)
from lib.data import load_ms_marco, load_evaluation_samples
from lib.analysis import cohens_d, compute_token_overlap
from lib.surrogate import (
    generate_all_5_surrogates,
    generate_summary,
    TOP_5_SURROGATE_TEMPLATES,
)
from scipy import stats
from tqdm.auto import tqdm

config = ExperimentConfig(
    model_name=MODEL_NAME,
    num_samples=4000,
    min_passage_words=20,
    max_passage_words=500,
    seed=SEED,
)

# Templates — bare text, no "Document:\\n" framing (hurts NLL, d=-0.45)
SURROGATE_PREFIX_TEMPLATE = "{surrogate}\\n"
DOCUMENT_TEMPLATE = "{document}"
QUERY_TEMPLATE = "\\nQuery: {query}\\nAnswer:"
ANSWER_TEMPLATE = " {answer}"
SUFFIX_SEPARATOR = "\\n\\nRelated question: "
CHECKPOINT_EVERY = 50
N_COMPARISONS = 10
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS
N_EVAL = 2000  # number of samples to evaluate

# Stopwords for TF-IDF and oracle-as-keywords
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
    'just', 'because', 'about', 'that', 'this', 'these', 'those', 'it',
    'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'him',
    'his', 'she', 'her', 'which', 'who', 'whom', 'there', 'here', 'when',
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


def generate_random_prefix_text(target_text, tokenizer, seed):
    # Generate random token text matching the token length of target_text.
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_len = len(target_ids)
    if target_len == 0:
        return ""
    rng = np.random.RandomState(seed)
    vocab_size = len(tokenizer)
    min_id = 3
    random_ids = rng.randint(min_id, vocab_size, size=target_len)
    random_text = tokenizer.decode(random_ids.tolist(), skip_special_tokens=True)
    reencoded = tokenizer.encode(random_text, add_special_tokens=False)
    if len(reencoded) != target_len:
        if len(reencoded) > target_len:
            random_text = tokenizer.decode(reencoded[:target_len], skip_special_tokens=True)
        else:
            extra_needed = target_len - len(reencoded)
            extra_ids = rng.randint(min_id, vocab_size, size=extra_needed)
            extra_text = tokenizer.decode(extra_ids.tolist(), skip_special_tokens=True)
            random_text = random_text + extra_text
            reencoded2 = tokenizer.encode(random_text, add_special_tokens=False)
            if len(reencoded2) > target_len:
                random_text = tokenizer.decode(reencoded2[:target_len], skip_special_tokens=True)
    return random_text


def extract_tfidf_keywords(passage, n_keywords=8):
    # Extract top content words by frequency (stopwords removed).
    words = re.findall(r'\\b[a-zA-Z]+\\b', passage.lower())
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join([w for w, _ in Counter(content_words).most_common(n_keywords)])


def extract_first_sentence(passage, max_words=30):
    # Extract first sentence of passage, up to max_words.
    first = passage.split('.')[0].strip()
    return ' '.join(first.split()[:max_words])


def oracle_to_keywords(query):
    # Strip question/function words from oracle query.
    words = re.findall(r'\\b[a-zA-Z]+\\b', query)
    return ' '.join([w for w in words if w.lower() not in QUESTION_STOPWORDS and len(w) > 2])


def shuffle_tokens(text, tokenizer, seed):
    # Shuffle token IDs of text while preserving token count.
    ids = tokenizer.encode(text, add_special_tokens=False)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


print("Config ready")
print(f"  num_samples pool: {config.num_samples}")
print(f"  eval samples: {N_EVAL}")
print(f"  passage words: {config.min_passage_words}-{config.max_passage_words}")
print(f"  bonferroni_alpha: {BONFERRONI_ALPHA:.4f} ({N_COMPARISONS} comparisons)")
print(f"  conditions: 15")\
""")))

# ========== Cell 4: Load MS MARCO ==========
cells.append(make_cell("code", s("""\
# Cell 4: Load MS MARCO (2000 samples, full distribution)
dataset = load_ms_marco(config)

np.random.seed(SEED)
all_samples = load_evaluation_samples(dataset, config, require_answer=True)

# Take first 2000 for manageable runtime
samples = all_samples[:N_EVAL]
N = len(samples)
print(f"Loaded {len(all_samples)} candidates, using first {N} for evaluation")
print(f"Example passage ({len(samples[0]['passage'].split())} words): {samples[0]['passage'][:100]}...")
print(f"Example query: {samples[0]['query']}")
print(f"Example answer: {samples[0]['answer']}")\
""")))

# ========== Cell 5: Generate ALL LLM surrogates ==========
cells.append(make_cell("code", s("""\
# Cell 5: Generate ALL LLM surrogates (5 templates + summary, checkpointed)
# generate_all_5_surrogates() gives: keyword_query, target_question, symptom_scenario,
#     misconception_negative, messy_realworld
# generate_summary() gives: 2-sentence summary
# Total: 6 LLM calls per sample

print("=" * 70)
print("PHASE 1: LLM SURROGATE GENERATION")
print("=" * 70)

surrogates_5_path = SURROGATES_DIR / "all_5_surrogates.json"
summaries_path = SURROGATES_DIR / "summaries.json"

# --- Load or generate all_5_surrogates ---
if surrogates_5_path.exists():
    with open(surrogates_5_path, 'r') as f:
        surrogates_5_data = json.load(f)
    surrogates_5 = surrogates_5_data['surrogates']
    print(f"Loaded {len(surrogates_5)} sets of 5-template surrogates from cache")
else:
    surrogates_5 = []

start_5 = len(surrogates_5)
if start_5 < N:
    print(f"Generating 5-template surrogates for samples {start_5} to {N-1}...")
    t_start = time.time()
    for idx in tqdm(range(start_5, N), initial=start_5, total=N,
                     desc="5-template surrogates"):
        passage = samples[idx]['passage']
        try:
            s5 = generate_all_5_surrogates(passage, model, tokenizer, config)
        except Exception as e:
            print(f"  WARNING: 5-template generation failed for sample {idx}: {e}")
            s5 = {k: "" for k in TOP_5_SURROGATE_TEMPLATES.keys()}
        surrogates_5.append(s5)

        if (idx + 1) % 100 == 0 or idx == N - 1:
            with open(surrogates_5_path, 'w') as f:
                json.dump({'surrogates': surrogates_5}, f)
            elapsed = time.time() - t_start
            rate = (idx - start_5 + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"  Saved {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

    with open(surrogates_5_path, 'w') as f:
        json.dump({'surrogates': surrogates_5}, f)
    print(f"5-template surrogates complete: {len(surrogates_5)} samples")
else:
    print(f"All 5-template surrogates already cached ({len(surrogates_5)} samples)")

# --- Load or generate summaries ---
if summaries_path.exists():
    with open(summaries_path, 'r') as f:
        summaries_data = json.load(f)
    summaries = summaries_data['summaries']
    print(f"Loaded {len(summaries)} summaries from cache")
else:
    summaries = []

start_sum = len(summaries)
if start_sum < N:
    print(f"Generating summaries for samples {start_sum} to {N-1}...")
    t_start = time.time()
    for idx in tqdm(range(start_sum, N), initial=start_sum, total=N, desc="Summaries"):
        passage = samples[idx]['passage']
        try:
            summary = generate_summary(passage, model, tokenizer, config)
        except Exception as e:
            print(f"  WARNING: Summary generation failed for sample {idx}: {e}")
            summary = ""
        summaries.append(summary)

        if (idx + 1) % 100 == 0 or idx == N - 1:
            with open(summaries_path, 'w') as f:
                json.dump({'summaries': summaries}, f)
            elapsed = time.time() - t_start
            rate = (idx - start_sum + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - idx - 1) / rate if rate > 0 else 0
            tqdm.write(f"  Saved {idx+1}/{N} | {rate:.2f} s/s | ETA: {remaining/60:.1f} min")

    with open(summaries_path, 'w') as f:
        json.dump({'summaries': summaries}, f)
    print(f"Summaries complete: {len(summaries)} samples")
else:
    print(f"All summaries already cached ({len(summaries)} samples)")

# Validate
n_empty_kw = sum(1 for s in surrogates_5 if not s.get('keyword_query', '').strip())
n_empty_q = sum(1 for s in surrogates_5 if not s.get('target_question', '').strip())
n_empty_sum = sum(1 for s in summaries if not s.strip())
print(f"\\nValidation:")
print(f"  Empty keyword surrogates: {n_empty_kw}/{N}")
print(f"  Empty question surrogates: {n_empty_q}/{N}")
print(f"  Empty summaries: {n_empty_sum}/{N}")

# Show examples
print(f"\\nExamples (sample 0):")
print(f"  Passage: {samples[0]['passage'][:80]}...")
for k, v in surrogates_5[0].items():
    print(f"  {k}: {v}")
print(f"  summary: {summaries[0][:100]}...")\
""")))

# ========== Cell 6: Compute non-LLM surrogates + token overlap diagnostics ==========
cells.append(make_cell("code", s("""\
# Cell 6: Compute non-LLM surrogates + token overlap diagnostics
print("=" * 70)
print("PHASE 1b: NON-LLM SURROGATES + TOKEN OVERLAP")
print("=" * 70)

# Pre-compute all non-LLM surrogates
tfidf_keywords = []
anti_keywords = []
passage_echos = []
oracle_as_kw = []
shuffled_llm = []

for i in range(N):
    passage = samples[i]['passage']
    query = samples[i]['query']
    llm_kw = surrogates_5[i].get('keyword_query', '')

    # TF-IDF keywords from THIS passage
    tfidf_keywords.append(extract_tfidf_keywords(passage, n_keywords=8))

    # Anti-keywords from WRONG passage (offset by 500)
    wrong_passage = samples[(i + 500) % N]['passage']
    anti_keywords.append(extract_tfidf_keywords(wrong_passage, n_keywords=8))

    # Passage echo: first sentence
    passage_echos.append(extract_first_sentence(passage, max_words=30))

    # Oracle-as-keywords: strip question/function words
    oracle_as_kw.append(oracle_to_keywords(query))

    # Shuffled-LLM: shuffle token IDs of LLM keyword surrogate
    shuffled_llm.append(shuffle_tokens(llm_kw, tokenizer, seed=SEED + i))

# Compute token overlaps for ALL non-bare conditions vs passage
print("Computing token overlaps...")
overlap_data = {}
overlap_labels = [
    ('random', 'Random prefix'),
    ('oracle', 'Oracle query'),
    ('oracle_kw', 'Oracle-as-keywords'),
    ('anti_kw', 'Anti-keywords (wrong doc)'),
    ('tfidf', 'TF-IDF keywords (right doc)'),
    ('echo', 'Passage echo (1st sentence)'),
    ('shuffled', 'Shuffled LLM'),
    ('llm_keyword', 'LLM keyword'),
    ('llm_question', 'LLM question'),
    ('llm_symptom', 'LLM symptom'),
    ('llm_summary', 'LLM summary'),
    ('llm_messy', 'LLM messy'),
]

for i in tqdm(range(N), desc="Token overlap"):
    passage = samples[i]['passage']
    query = samples[i]['query']
    random_text = generate_random_prefix_text(query, tokenizer, seed=SEED + i)

    overlaps_i = {
        'random': compute_token_overlap(random_text, passage, tokenizer),
        'oracle': compute_token_overlap(query, passage, tokenizer),
        'oracle_kw': compute_token_overlap(oracle_as_kw[i], passage, tokenizer),
        'anti_kw': compute_token_overlap(anti_keywords[i], passage, tokenizer),
        'tfidf': compute_token_overlap(tfidf_keywords[i], passage, tokenizer),
        'echo': compute_token_overlap(passage_echos[i], passage, tokenizer),
        'shuffled': compute_token_overlap(shuffled_llm[i], passage, tokenizer),
        'llm_keyword': compute_token_overlap(surrogates_5[i].get('keyword_query', ''), passage, tokenizer),
        'llm_question': compute_token_overlap(surrogates_5[i].get('target_question', ''), passage, tokenizer),
        'llm_symptom': compute_token_overlap(surrogates_5[i].get('symptom_scenario', ''), passage, tokenizer),
        'llm_summary': compute_token_overlap(summaries[i], passage, tokenizer),
        'llm_messy': compute_token_overlap(surrogates_5[i].get('messy_realworld', ''), passage, tokenizer),
    }
    overlap_data[i] = overlaps_i

# Report overlap gradient
print(f"\\n{'Condition':<30} {'Mean Overlap':>12} {'Std':>10}")
print("-" * 55)
for key, label in overlap_labels:
    vals = [overlap_data[i][key] for i in range(N)]
    print(f"{label:<30} {np.mean(vals):>12.4f} {np.std(vals):>10.4f}")

# Save overlap data
with open(RESULTS_DIR / "overlap_data.json", 'w') as f:
    json.dump({str(k): v for k, v in overlap_data.items()}, f)
print(f"\\nOverlap data saved to {RESULTS_DIR / 'overlap_data.json'}")\
""")))

# ========== Cell 7: Condition explanation cell ==========
cells.append(make_cell("code", s("""\
# Cell 7: Condition explanation with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("=" * 70)

ex_i = 0
ex_passage = samples[ex_i]['passage']
ex_query = samples[ex_i]['query']
ex_llm_kw = surrogates_5[ex_i].get('keyword_query', '')
ex_llm_q = surrogates_5[ex_i].get('target_question', '')
ex_llm_symp = surrogates_5[ex_i].get('symptom_scenario', '')
ex_llm_messy = surrogates_5[ex_i].get('messy_realworld', '')
ex_summary = summaries[ex_i]
ex_tfidf = tfidf_keywords[ex_i]
ex_anti = anti_keywords[ex_i]
ex_echo = passage_echos[ex_i]
ex_oracle_kw = oracle_as_kw[ex_i]
ex_shuffled = shuffled_llm[ex_i]

conditions_explained = [
    ("1. Bare", "[BOS][doc]", "No prefix — baseline"),
    ("2. Random-truncated", "[BOS][random_tokens\\\\n][doc] → truncate + RoPE",
     f"Random text: '{generate_random_prefix_text(ex_query, tokenizer, SEED)[:60]}...'"),
    ("3. Separator-only", "[BOS][doc][\\\\n\\\\nRelated question: ]",
     "Suffix appended after passage — structural framing only"),
    ("4. Oracle-truncated", "[BOS][query\\\\n][doc] → truncate + RoPE",
     f"Query: '{ex_query[:60]}...'"),
    ("5. Oracle-as-keywords", "[BOS][keywords\\\\n][doc] → truncate + RoPE",
     f"Keywords from oracle: '{ex_oracle_kw[:60]}'"),
    ("6. Anti-keywords", "[BOS][wrong_tfidf\\\\n][doc] → truncate + RoPE",
     f"TF-IDF from WRONG passage: '{ex_anti[:60]}'"),
    ("7. TF-IDF-keywords", "[BOS][tfidf\\\\n][doc] → truncate + RoPE",
     f"TF-IDF from THIS passage: '{ex_tfidf[:60]}'"),
    ("8. Passage-echo", "[BOS][first_sent\\\\n][doc] → truncate + RoPE",
     f"First sentence: '{ex_echo[:60]}...'"),
    ("9. Shuffled-LLM", "[BOS][shuffled\\\\n][doc] → truncate + RoPE",
     f"Shuffled LLM keyword tokens: '{ex_shuffled[:60]}'"),
    ("10. LLM-keyword", "[BOS][llm_kw\\\\n][doc] → truncate + RoPE",
     f"LLM keyword: '{ex_llm_kw[:60]}'"),
    ("11. LLM-question", "[BOS][llm_q\\\\n][doc] → truncate + RoPE",
     f"LLM question: '{ex_llm_q[:60]}'"),
    ("12. LLM-symptom", "[BOS][llm_symp\\\\n][doc] → truncate + RoPE",
     f"LLM symptom: '{ex_llm_symp[:60]}'"),
    ("13. LLM-summary", "[BOS][summary\\\\n][doc] → truncate + RoPE",
     f"Summary: '{ex_summary[:60]}...'"),
    ("14. LLM-keyword+sep", "[BOS][llm_kw\\\\n][doc][\\\\n\\\\nRelated question: ] (prefix+suffix)",
     f"Stacking: truncated prefix + suffix separator"),
    ("15. LLM-messy", "[BOS][llm_messy\\\\n][doc] → truncate + RoPE",
     f"LLM messy: '{ex_llm_messy[:60]}'"),
]

for name, pattern, detail in conditions_explained:
    print(f"\\n### {name} ###")
    print(f"  Cache: {pattern}")
    print(f"  Detail: {detail}")\
""")))

# ========== Cell 8: Main eval loop (15 conditions × 2000 samples) ==========
cells.append(make_cell("code", s("""\
# Cell 8: Main eval loop — 15 conditions × 2000 samples
print("=" * 70)
print("PHASE 2: MAIN EVALUATION (15 conditions × 2000 samples)")
print("=" * 70)

CONDITION_NAMES = [
    'bare', 'random_trunc', 'separator_only',
    'oracle_trunc', 'oracle_as_kw',
    'anti_keywords', 'tfidf_keywords', 'passage_echo', 'shuffled_llm',
    'llm_keyword', 'llm_question', 'llm_symptom', 'llm_summary',
    'llm_keyword_sep', 'llm_messy',
]

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

    # --- Matched tokenization (same doc_ids for ALL truncated conditions) ---
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

    # Pre-compute all prefix texts
    random_text = generate_random_prefix_text(query, tokenizer, seed=SEED + idx)
    llm_kw_text = surrogates_5[idx].get('keyword_query', '')
    llm_q_text = surrogates_5[idx].get('target_question', '')
    llm_symp_text = surrogates_5[idx].get('symptom_scenario', '')
    llm_messy_text = surrogates_5[idx].get('messy_realworld', '')
    summary_text = summaries[idx]
    tfidf_text = tfidf_keywords[idx]
    anti_kw_text = anti_keywords[idx]
    echo_text = passage_echos[idx]
    oracle_kw_text = oracle_as_kw[idx]
    shuffled_text = shuffled_llm[idx]

    # Helper: build truncated cache from prefix text
    def build_trunc(prefix_text):
        prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=prefix_text)
        prefix_enc = tokenizer(prefix_str, return_tensors="pt",
                               add_special_tokens=False, padding=False, truncation=False)
        prefix_ids = prefix_enc['input_ids'].to(config.device)
        full_ids = torch.cat([bos_id, prefix_ids, doc_ids], dim=1)
        prefix_token_len = 1 + prefix_ids.shape[1]

        with torch.no_grad():
            out = model(input_ids=full_ids,
                        attention_mask=torch.ones_like(full_ids),
                        use_cache=True, return_dict=True)
        cache = extract_and_truncate_cache_with_bos(out.past_key_values, doc_len)
        correct_rope_positions_with_bos(cache, prefix_token_len - 1, model)
        nll = score_answer_with_cache(
            deepcopy_cache(cache), 1 + doc_len,
            query_prompt, answer_text, model, tokenizer, config)
        del out, cache
        return nll

    # === Condition 1: BARE ===
    bare_ids = torch.cat([bos_id, doc_ids], dim=1)
    bare_len = bare_ids.shape[1]
    with torch.no_grad():
        bare_out = model(input_ids=bare_ids, attention_mask=torch.ones_like(bare_ids),
                         use_cache=True, return_dict=True)
    nll_bare = score_answer_with_cache(
        deepcopy_cache(bare_out.past_key_values), bare_len,
        query_prompt, answer_text, model, tokenizer, config)
    del bare_out

    # === Condition 2: RANDOM-TRUNCATED ===
    nll_random = build_trunc(random_text)

    # === Condition 3: SEPARATOR-ONLY ===
    sep_len, sep_cache = build_suffix_kv_cache(
        passage, "", model, tokenizer, config, separator=SUFFIX_SEPARATOR)
    nll_separator = score_answer_with_cache(
        deepcopy_cache(sep_cache), sep_len,
        query_prompt, answer_text, model, tokenizer, config)
    del sep_cache

    # === Condition 4: ORACLE-TRUNCATED ===
    with torch.no_grad():
        oracle_out = model(input_ids=full_oracle_ids,
                           attention_mask=torch.ones_like(full_oracle_ids),
                           use_cache=True, return_dict=True)
    oracle_cache = extract_and_truncate_cache_with_bos(oracle_out.past_key_values, doc_len)
    correct_rope_positions_with_bos(oracle_cache, oracle_prefix_len - 1, model)
    nll_oracle = score_answer_with_cache(
        deepcopy_cache(oracle_cache), 1 + doc_len,
        query_prompt, answer_text, model, tokenizer, config)
    del oracle_out, oracle_cache

    # === Condition 5: ORACLE-AS-KEYWORDS ===
    nll_oracle_kw = build_trunc(oracle_kw_text)

    # === Condition 6: ANTI-KEYWORDS ===
    nll_anti_kw = build_trunc(anti_kw_text)

    # === Condition 7: TF-IDF-KEYWORDS ===
    nll_tfidf = build_trunc(tfidf_text)

    # === Condition 8: PASSAGE-ECHO ===
    nll_echo = build_trunc(echo_text)

    # === Condition 9: SHUFFLED-LLM ===
    nll_shuffled = build_trunc(shuffled_text)

    # === Condition 10: LLM-KEYWORD ===
    nll_llm_kw = build_trunc(llm_kw_text)

    # === Condition 11: LLM-QUESTION ===
    nll_llm_q = build_trunc(llm_q_text)

    # === Condition 12: LLM-SYMPTOM ===
    nll_llm_symp = build_trunc(llm_symp_text)

    # === Condition 13: LLM-SUMMARY ===
    nll_llm_sum = build_trunc(summary_text)

    # === Condition 14: LLM-KEYWORD + SEPARATOR ===
    # Build truncated cache with LLM keyword prefix, then use build_suffix on THAT
    # Step 1: Build the primed document cache (truncated)
    kw_prefix_str = SURROGATE_PREFIX_TEMPLATE.format(surrogate=llm_kw_text)
    kw_prefix_enc = tokenizer(kw_prefix_str, return_tensors="pt",
                              add_special_tokens=False, padding=False, truncation=False)
    kw_prefix_ids = kw_prefix_enc['input_ids'].to(config.device)
    kw_full_ids = torch.cat([bos_id, kw_prefix_ids, doc_ids], dim=1)
    kw_prefix_token_len = 1 + kw_prefix_ids.shape[1]

    with torch.no_grad():
        kw_out = model(input_ids=kw_full_ids,
                       attention_mask=torch.ones_like(kw_full_ids),
                       use_cache=True, return_dict=True)
    kw_trunc_cache = extract_and_truncate_cache_with_bos(kw_out.past_key_values, doc_len)
    correct_rope_positions_with_bos(kw_trunc_cache, kw_prefix_token_len - 1, model)
    del kw_out

    # Step 2: Extend with suffix separator (forward pass through separator tokens)
    suffix_enc = tokenizer(SUFFIX_SEPARATOR, return_tensors="pt",
                           add_special_tokens=False, padding=False, truncation=False)
    suffix_ids = suffix_enc['input_ids'].to(config.device)
    suffix_len = suffix_ids.shape[1]
    cache_len_before_suffix = 1 + doc_len

    # Create position_ids continuing from doc end
    suffix_position_ids = torch.arange(
        cache_len_before_suffix, cache_len_before_suffix + suffix_len,
        device=config.device
    ).unsqueeze(0)

    with torch.no_grad():
        suffix_out = model(
            input_ids=suffix_ids,
            attention_mask=torch.ones(1, cache_len_before_suffix + suffix_len,
                                      device=config.device, dtype=torch.long),
            position_ids=suffix_position_ids,
            past_key_values=deepcopy_cache(kw_trunc_cache),
            use_cache=True,
            return_dict=True,
        )
    combo_cache = suffix_out.past_key_values
    combo_len = cache_len_before_suffix + suffix_len
    nll_llm_kw_sep = score_answer_with_cache(
        deepcopy_cache(combo_cache), combo_len,
        query_prompt, answer_text, model, tokenizer, config)
    del suffix_out, combo_cache, kw_trunc_cache

    # === Condition 15: LLM-MESSY ===
    nll_llm_messy = build_trunc(llm_messy_text)

    # --- Store result ---
    result = {
        'idx': idx,
        'doc_len': doc_len,
        'passage_word_count': len(passage.split()),
        'bare': nll_bare,
        'random_trunc': nll_random,
        'separator_only': nll_separator,
        'oracle_trunc': nll_oracle,
        'oracle_as_kw': nll_oracle_kw,
        'anti_keywords': nll_anti_kw,
        'tfidf_keywords': nll_tfidf,
        'passage_echo': nll_echo,
        'shuffled_llm': nll_shuffled,
        'llm_keyword': nll_llm_kw,
        'llm_question': nll_llm_q,
        'llm_symptom': nll_llm_symp,
        'llm_summary': nll_llm_sum,
        'llm_keyword_sep': nll_llm_kw_sep,
        'llm_messy': nll_llm_messy,
    }
    results.append(result)

    torch.cuda.empty_cache()

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
# Cell 9: Primary analysis — all 10 comparisons (Bonferroni alpha = 0.005)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYSIS — MECHANISM DECOMPOSITION")
print("=" * 70)

# Extract arrays and filter zero NLLs
cond_arrays = {}
for cname in CONDITION_NAMES:
    cond_arrays[cname] = np.array([r[cname] for r in results])

# Valid mask: no zero NLLs in any condition
valid = np.ones(len(results), dtype=bool)
for cname in CONDITION_NAMES:
    valid &= (cond_arrays[cname] != 0)
n_valid = int(np.sum(valid))
n_excluded = int(np.sum(~valid))
print(f"Total: {len(results)}, Valid: {n_valid}, Excluded: {n_excluded}")

# Apply mask
c = {}
for cname in CONDITION_NAMES:
    c[cname] = cond_arrays[cname][valid]

# NLL summary table
print(f"\\n{'Condition':<25} {'Mean NLL':>10} {'Std':>10} {'d vs Bare':>10}")
print("-" * 60)
for cname in CONDITION_NAMES:
    mean_nll = np.mean(c[cname])
    std_nll = np.std(c[cname])
    if cname == 'bare':
        d_str = "—"
    else:
        d = cohens_d(c['bare'] - c[cname])
        d_str = f"{d:+.3f}"
    print(f"{cname:<25} {mean_nll:>10.4f} {std_nll:>10.4f} {d_str:>10}")

# 10 primary comparisons
print(f"\\n{'='*80}")
print(f"10 PRIMARY COMPARISONS (Bonferroni alpha = {BONFERRONI_ALPHA:.4f})")
print(f"{'='*80}")

comparisons = [
    # (name, delta_array, question)
    # delta > 0 means first condition is better (lower NLL)
    ('M1: Shuffled vs LLM-kw', c['shuffled_llm'] - c['llm_keyword'],
     'Does coherence matter?'),
    ('M2: Oracle-kw vs Oracle', c['oracle_trunc'] - c['oracle_as_kw'],
     'Does question format hurt?'),
    ('M3: LLM-kw vs LLM-question', c['llm_question'] - c['llm_keyword'],
     'Keyword vs question format?'),
    ('M4: TF-IDF vs Anti-kw', c['anti_keywords'] - c['tfidf_keywords'],
     'Does passage specificity matter?'),
    ('M5: Echo vs LLM-kw', c['llm_keyword'] - c['passage_echo'],
     'Is max overlap the ceiling?'),
    ('M6: TF-IDF vs LLM-kw', c['llm_keyword'] - c['tfidf_keywords'],
     'Is LLM necessary beyond TF-IDF?'),
    ('M7: LLM-kw+sep vs best single',
     np.minimum(c['llm_keyword'], c['separator_only']) - c['llm_keyword_sep'],
     'Does stacking help?'),
    ('R1: LLM-kw vs Random', c['random_trunc'] - c['llm_keyword'],
     'Replicates Exp 05 (d≈0.2)?'),
    ('R2: Oracle vs Random', c['random_trunc'] - c['oracle_trunc'],
     'Replicates null (d≈0)?'),
    ('R3: LLM-kw vs Bare', c['bare'] - c['llm_keyword'],
     'Overall LLM benefit?'),
]

print(f"\\n{'Comparison':<30} {'Mean Δ':>8} {'d':>8} {'Win%':>7} {'t':>8} {'p':>12} {'Sig':>5}")
print("-" * 85)

comparison_results = {}
for name, delta, question in comparisons:
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{name:<30} {np.mean(delta):>8.4f} {d:>8.3f} {win:>6.1f}% {t_stat:>8.2f} {p_val:>11.2e} {sig:>5}")
    comparison_results[name] = {
        'mean_delta': float(np.mean(delta)),
        'cohens_d': float(d),
        'win_rate': float(win / 100),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'bonferroni_significant': bool(p_val < BONFERRONI_ALPHA),
        'question': question,
    }

# All conditions vs Bare
print(f"\\n{'='*80}")
print("ALL CONDITIONS vs BARE")
print(f"{'='*80}")
print(f"\\n{'Condition':<25} {'d vs Bare':>10} {'Win%':>7} {'p':>12}")
print("-" * 60)
all_vs_bare = {}
for cname in CONDITION_NAMES:
    if cname == 'bare':
        continue
    delta = c['bare'] - c[cname]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < BONFERRONI_ALPHA else "*" if p_val < 0.05 else "ns"
    print(f"{cname:<25} {d:>10.3f} {win:>6.1f}% {p_val:>11.2e} {sig:>5}")
    all_vs_bare[cname] = {'cohens_d': float(d), 'win_rate': float(win/100), 'p_value': float(p_val)}\
""")))

# ========== Cell 10: Token overlap mechanism analysis + regression ==========
cells.append(make_cell("code", s("""\
# Cell 10: Token overlap mechanism analysis + regression

print("=" * 70)
print("TOKEN OVERLAP MECHANISM ANALYSIS")
print("=" * 70)

# Map condition names to overlap keys
cond_to_overlap = {
    'random_trunc': 'random',
    'oracle_trunc': 'oracle',
    'oracle_as_kw': 'oracle_kw',
    'anti_keywords': 'anti_kw',
    'tfidf_keywords': 'tfidf',
    'passage_echo': 'echo',
    'shuffled_llm': 'shuffled',
    'llm_keyword': 'llm_keyword',
    'llm_question': 'llm_question',
    'llm_symptom': 'llm_symptom',
    'llm_summary': 'llm_summary',
    'llm_messy': 'llm_messy',
}

# 1. Pool all (sample, condition) pairs → overlap vs delta
all_overlaps = []
all_deltas = []
all_cond_labels = []
valid_indices = np.where(valid)[0]

for cname, okey in cond_to_overlap.items():
    for i_valid, i_orig in enumerate(valid_indices):
        ov = overlap_data[i_orig][okey]
        delta = c['bare'][i_valid] - c[cname][i_valid]
        all_overlaps.append(ov)
        all_deltas.append(delta)
        all_cond_labels.append(cname)

all_overlaps = np.array(all_overlaps)
all_deltas = np.array(all_deltas)

# Universal correlation
r_all, p_all = stats.pearsonr(all_overlaps, all_deltas)
print(f"\\nUniversal overlap-delta correlation (pooled across all conditions):")
print(f"  r = {r_all:.4f}, p = {p_all:.2e}, N = {len(all_overlaps)}")

# 2. Within-condition correlations
print(f"\\n{'Condition':<25} {'r(overlap,delta)':>18} {'p':>12} {'N':>6}")
print("-" * 65)
for cname, okey in cond_to_overlap.items():
    ovs = []
    deltas = []
    for i_valid, i_orig in enumerate(valid_indices):
        ovs.append(overlap_data[i_orig][okey])
        deltas.append(c['bare'][i_valid] - c[cname][i_valid])
    r, p = stats.pearsonr(ovs, deltas)
    print(f"{cname:<25} {r:>18.4f} {p:>11.2e} {len(ovs):>6}")

# 3. Cross-condition: median overlap vs mean effect size
print(f"\\nCross-condition: median overlap vs mean Cohen's d")
print(f"{'Condition':<25} {'Median Overlap':>15} {'Mean d':>10}")
print("-" * 55)
cond_median_overlap = []
cond_mean_d = []
for cname, okey in cond_to_overlap.items():
    ovs = [overlap_data[i_orig][okey] for i_orig in valid_indices]
    d = cohens_d(c['bare'] - c[cname])
    med_ov = np.median(ovs)
    cond_median_overlap.append(med_ov)
    cond_mean_d.append(d)
    print(f"{cname:<25} {med_ov:>15.4f} {d:>10.3f}")

r_cross, p_cross = stats.pearsonr(cond_median_overlap, cond_mean_d)
print(f"\\nCross-condition correlation: r = {r_cross:.4f}, p = {p_cross:.4f}")

# 4. Regression: delta ~ overlap + hardness + overlap*hardness
from numpy.polynomial import polynomial as P

print(f"\\n--- Regression: delta ~ overlap + hardness + overlap*hardness ---")
bare_valid = c['bare']
# Standardize
ov_std = (all_overlaps - np.mean(all_overlaps)) / (np.std(all_overlaps) + 1e-8)
# Hardness: repeat bare NLL for each condition
hardness_all = np.tile(bare_valid, len(cond_to_overlap))
h_std = (hardness_all - np.mean(hardness_all)) / (np.std(hardness_all) + 1e-8)
interaction = ov_std * h_std

X = np.column_stack([np.ones(len(all_deltas)), ov_std, h_std, interaction])
betas, residuals, rank, sv = np.linalg.lstsq(X, all_deltas, rcond=None)
y_pred = X @ betas
ss_res = np.sum((all_deltas - y_pred) ** 2)
ss_tot = np.sum((all_deltas - np.mean(all_deltas)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"  beta_0 (intercept):           {betas[0]:+.5f}")
print(f"  beta_1 (overlap):             {betas[1]:+.5f}")
print(f"  beta_2 (hardness):            {betas[2]:+.5f}")
print(f"  beta_3 (overlap × hardness):  {betas[3]:+.5f}")
print(f"  R² = {r_squared:.4f}")

# Decision criteria
print(f"\\n--- Decision Criteria ---")
if abs(r_all) > 0.3:
    print(f"  ✓ r(overlap, delta) = {r_all:.3f} > 0.3 → overlap IS the mechanism")
elif abs(r_all) > 0.1:
    print(f"  ~ r(overlap, delta) = {r_all:.3f} ∈ (0.1, 0.3) → overlap is PART of the mechanism")
else:
    print(f"  ✗ r(overlap, delta) = {r_all:.3f} < 0.1 → overlap is NOT the main mechanism")

# TF-IDF vs LLM comparison
d_tfidf = cohens_d(c['bare'] - c['tfidf_keywords'])
d_llm_kw = cohens_d(c['bare'] - c['llm_keyword'])
if abs(d_tfidf - d_llm_kw) < 0.05:
    print(f"  ✓ TF-IDF (d={d_tfidf:.3f}) ≈ LLM-kw (d={d_llm_kw:.3f}) → LLM may be unnecessary")
else:
    print(f"  ✗ TF-IDF (d={d_tfidf:.3f}) ≠ LLM-kw (d={d_llm_kw:.3f}) → LLM adds value beyond overlap")

# Shuffled vs ordered
d_shuffled = cohens_d(c['bare'] - c['shuffled_llm'])
if abs(d_shuffled - d_llm_kw) < 0.05:
    print(f"  ✓ Shuffled (d={d_shuffled:.3f}) ≈ Ordered (d={d_llm_kw:.3f}) → coherence doesn't matter")
else:
    print(f"  ✗ Shuffled (d={d_shuffled:.3f}) ≠ Ordered (d={d_llm_kw:.3f}) → coherence matters")\
""")))

# ========== Cell 11: Format & coherence ablation results ==========
cells.append(make_cell("code", s("""\
# Cell 11: Format & coherence ablation results

print("=" * 70)
print("FORMAT & COHERENCE ABLATION RESULTS")
print("=" * 70)

# H2: Coherence — shuffled vs ordered
delta_coherence = c['shuffled_llm'] - c['llm_keyword']
d_coherence = cohens_d(delta_coherence)
t_coh, p_coh = stats.ttest_1samp(delta_coherence, 0)
print(f"\\nH2 — COHERENCE: Shuffled-LLM vs LLM-keyword")
print(f"  d = {d_coherence:+.3f}, win% = {np.mean(delta_coherence > 0)*100:.1f}%, "
      f"t = {t_coh:.2f}, p = {p_coh:.2e}")
if p_coh < BONFERRONI_ALPHA and d_coherence > 0:
    print(f"  → Coherence MATTERS: ordered tokens outperform shuffled")
elif p_coh < BONFERRONI_ALPHA and d_coherence < 0:
    print(f"  → SURPRISING: shuffled tokens are BETTER than ordered")
else:
    print(f"  → Coherence does NOT matter: order is irrelevant, just token identity")

# H3: Format — question syntax
# Oracle
delta_fmt_oracle = c['oracle_trunc'] - c['oracle_as_kw']
d_fmt_oracle = cohens_d(delta_fmt_oracle)
t_fo, p_fo = stats.ttest_1samp(delta_fmt_oracle, 0)
print(f"\\nH3a — FORMAT (Oracle): Oracle-full-question vs Oracle-as-keywords")
print(f"  d = {d_fmt_oracle:+.3f}, win% = {np.mean(delta_fmt_oracle > 0)*100:.1f}%, "
      f"t = {t_fo:.2f}, p = {p_fo:.2e}")
if d_fmt_oracle > 0:
    print(f"  → Oracle-as-keywords IMPROVES over full question (question format hurts)")
else:
    print(f"  → Oracle-as-keywords does NOT improve over full question")

# LLM
delta_fmt_llm = c['llm_question'] - c['llm_keyword']
d_fmt_llm = cohens_d(delta_fmt_llm)
t_fl, p_fl = stats.ttest_1samp(delta_fmt_llm, 0)
print(f"\\nH3b — FORMAT (LLM): LLM-keyword vs LLM-question")
print(f"  d = {d_fmt_llm:+.3f}, win% = {np.mean(delta_fmt_llm > 0)*100:.1f}%, "
      f"t = {t_fl:.2f}, p = {p_fl:.2e}")
if d_fmt_llm > 0:
    print(f"  → LLM-keyword OUTPERFORMS LLM-question (keyword format is better)")
else:
    print(f"  → No advantage for keyword format in LLM surrogates")

# H4: Passage Specificity
delta_spec = c['anti_keywords'] - c['tfidf_keywords']
d_spec = cohens_d(delta_spec)
t_sp, p_sp = stats.ttest_1samp(delta_spec, 0)
print(f"\\nH4 — SPECIFICITY: TF-IDF (right doc) vs Anti-keywords (wrong doc)")
print(f"  d = {d_spec:+.3f}, win% = {np.mean(delta_spec > 0)*100:.1f}%, "
      f"t = {t_sp:.2f}, p = {p_sp:.2e}")
if p_sp < BONFERRONI_ALPHA and d_spec > 0:
    print(f"  → Passage-SPECIFIC keywords are BETTER than wrong-doc keywords")
elif p_sp >= BONFERRONI_ALPHA:
    print(f"  → Specificity does NOT matter: any content words work")

# H5: Stacking
delta_stack = np.minimum(c['llm_keyword'], c['separator_only']) - c['llm_keyword_sep']
d_stack = cohens_d(delta_stack)
t_st, p_st = stats.ttest_1samp(delta_stack, 0)
print(f"\\nH5 — STACKING: LLM-keyword+sep vs best-of(LLM-keyword, sep-only)")
print(f"  d = {d_stack:+.3f}, win% = {np.mean(delta_stack > 0)*100:.1f}%, "
      f"t = {t_st:.2f}, p = {p_st:.2e}")
if p_st < BONFERRONI_ALPHA and d_stack > 0:
    print(f"  → Stacking WORKS: combining prefix + suffix exceeds either alone")
elif p_st < 0.05 and d_stack > 0:
    print(f"  → Suggestive stacking benefit (not Bonferroni significant)")
else:
    print(f"  → No stacking benefit")\
""")))

# ========== Cell 12: Template ranking + production cost-benefit ==========
cells.append(make_cell("code", s("""\
# Cell 12: Template ranking + production cost-benefit analysis

print("=" * 70)
print("LLM TEMPLATE RANKING + COST-BENEFIT")
print("=" * 70)

llm_conditions = [
    ('llm_keyword', 'Keyword (3-6 words)', 'cheap'),
    ('llm_question', 'Question (5-12 words)', 'cheap'),
    ('llm_symptom', 'Symptom (4-10 words)', 'cheap'),
    ('llm_summary', 'Summary (2 sentences)', 'moderate'),
    ('llm_messy', 'Messy/informal (3-8 words)', 'cheap'),
]

non_llm_conditions = [
    ('tfidf_keywords', 'TF-IDF keywords', 'free'),
    ('passage_echo', 'First sentence echo', 'free'),
    ('oracle_as_kw', 'Oracle-as-keywords', 'oracle'),
]

print(f"\\n{'Template':<30} {'d vs Bare':>10} {'d vs Random':>12} {'Win% vs Bare':>13} {'Cost':>8}")
print("-" * 78)

# LLM templates
for cname, label, cost in llm_conditions:
    d_bare = cohens_d(c['bare'] - c[cname])
    d_random = cohens_d(c['random_trunc'] - c[cname])
    win = np.mean(c['bare'] > c[cname]) * 100
    print(f"{label:<30} {d_bare:>10.3f} {d_random:>12.3f} {win:>12.1f}% {cost:>8}")

print("-" * 78)
# Non-LLM alternatives
for cname, label, cost in non_llm_conditions:
    d_bare = cohens_d(c['bare'] - c[cname])
    d_random = cohens_d(c['random_trunc'] - c[cname])
    win = np.mean(c['bare'] > c[cname]) * 100
    print(f"{label:<30} {d_bare:>10.3f} {d_random:>12.3f} {win:>12.1f}% {cost:>8}")

# Pairwise template comparisons
print(f"\\n--- Pairwise LLM Template Comparisons ---")
llm_cnames = [cn for cn, _, _ in llm_conditions]
for i in range(len(llm_cnames)):
    for j in range(i+1, len(llm_cnames)):
        cn_a, cn_b = llm_cnames[i], llm_cnames[j]
        delta = c[cn_b] - c[cn_a]  # positive means a is better
        d = cohens_d(delta)
        t, p = stats.ttest_1samp(delta, 0)
        sig = "***" if p < 0.001 else "**" if p < BONFERRONI_ALPHA else "*" if p < 0.05 else "ns"
        print(f"  {cn_a} vs {cn_b}: d={d:+.3f}, p={p:.2e} {sig}")

# Production recommendation
print(f"\\n--- Production Recommendation ---")
best_d = -999
best_name = ""
for cname in ['llm_keyword', 'tfidf_keywords', 'passage_echo']:
    d = cohens_d(c['bare'] - c[cname])
    if d > best_d:
        best_d = d
        best_name = cname
print(f"Best overall: {best_name} (d={best_d:+.3f} vs bare)")

# Is LLM worth it?
d_tfidf_bare = cohens_d(c['bare'] - c['tfidf_keywords'])
d_llmkw_bare = cohens_d(c['bare'] - c['llm_keyword'])
improvement = d_llmkw_bare - d_tfidf_bare
print(f"LLM keyword over TF-IDF: Δd = {improvement:+.3f}")
if improvement > 0.05:
    print(f"→ LLM IS worth the cost (meaningful improvement over free TF-IDF)")
else:
    print(f"→ LLM may NOT be worth the cost (minimal improvement over free TF-IDF)")\
""")))

# ========== Cell 13: Hardness quintile breakdown ==========
cells.append(make_cell("code", s("""\
# Cell 13: Hardness quintile breakdown (all conditions)

print("=" * 70)
print("HARDNESS QUINTILE BREAKDOWN")
print("=" * 70)

bare_valid = c['bare']
quintile_boundaries = np.percentile(bare_valid, [20, 40, 60, 80])
quintile_labels = ['Q1 (easiest)', 'Q2', 'Q3', 'Q4', 'Q5 (hardest)']

def get_quintile(nll, boundaries):
    for i, b in enumerate(boundaries):
        if nll <= b:
            return i
    return len(boundaries)

quintiles = np.array([get_quintile(nll, quintile_boundaries) for nll in bare_valid])

# Header
conditions_to_show = [
    'random_trunc', 'oracle_trunc', 'oracle_as_kw', 'tfidf_keywords',
    'llm_keyword', 'llm_question', 'llm_symptom', 'passage_echo',
    'shuffled_llm', 'llm_keyword_sep',
]
header = f"{'Condition':<20}" + "".join(f"{ql:>14}" for ql in quintile_labels) + f"{'Overall':>14}"
print(f"\\n{header}")
print("-" * (20 + 14 * 6))

hardness_breakdown = {}
for cname in conditions_to_show:
    row = f"{cname:<20}"
    quintile_ds = []
    for q in range(5):
        mask_q = quintiles == q
        if np.sum(mask_q) < 10:
            row += f"{'n/a':>14}"
            quintile_ds.append(None)
        else:
            delta = bare_valid[mask_q] - c[cname][mask_q]
            d = cohens_d(delta)
            row += f"{d:>+14.3f}"
            quintile_ds.append(float(d))
    # Overall
    d_all = cohens_d(bare_valid - c[cname])
    row += f"{d_all:>+14.3f}"
    print(row)
    hardness_breakdown[cname] = {
        'quintile_ds': quintile_ds,
        'overall_d': float(d_all),
    }

# Hardness interaction correlations
print(f"\\nHardness interaction (r between bare NLL and benefit):")
for cname in conditions_to_show:
    delta = bare_valid - c[cname]
    r, p = stats.pearsonr(bare_valid, delta)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {cname:<20}: r={r:+.3f}, p={p:.2e} {sig}")\
""")))

# ========== Cell 14: Plots ==========
cells.append(make_cell("code", s("""\
# Cell 14: Plots (overlap scatter, condition bars, hardness heatmap)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# --- Plot 1: Overlap scatter (pooled across all conditions) ---
ax = axes[0, 0]
# Color by condition type
cond_colors = {
    'random_trunc': 'gray', 'oracle_trunc': 'royalblue', 'oracle_as_kw': 'cornflowerblue',
    'anti_keywords': 'salmon', 'tfidf_keywords': 'orange', 'passage_echo': 'gold',
    'shuffled_llm': 'mediumpurple', 'llm_keyword': 'forestgreen', 'llm_question': 'limegreen',
    'llm_symptom': 'darkgreen', 'llm_summary': 'olive', 'llm_messy': 'teal',
}
for cname, okey in cond_to_overlap.items():
    ovs = [overlap_data[i_orig][okey] for i_orig in valid_indices]
    deltas_plot = c['bare'] - c[cname]
    ax.scatter(ovs, deltas_plot, alpha=0.03, s=3, c=cond_colors.get(cname, 'gray'))
# Overlay condition means
for cname, okey in cond_to_overlap.items():
    ovs = [overlap_data[i_orig][okey] for i_orig in valid_indices]
    mean_ov = np.mean(ovs)
    mean_delta = np.mean(c['bare'] - c[cname])
    ax.scatter([mean_ov], [mean_delta], s=80, c=cond_colors.get(cname, 'gray'),
               edgecolors='black', linewidths=1, zorder=5)
    ax.annotate(cname.replace('_', '\\n'), (mean_ov, mean_delta), fontsize=5,
                ha='center', va='bottom')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Token Jaccard Overlap with Passage')
ax.set_ylabel('NLL Benefit vs Bare (positive = better)')
ax.set_title(f'Overlap vs Benefit (r={r_all:.3f})')

# --- Plot 2: All conditions bar chart (Cohen's d vs bare) ---
ax = axes[0, 1]
cnames_sorted = sorted(
    [cn for cn in CONDITION_NAMES if cn != 'bare'],
    key=lambda cn: cohens_d(c['bare'] - c[cn]),
    reverse=True
)
ds = [cohens_d(c['bare'] - c[cn]) for cn in cnames_sorted]
colors_bar = [cond_colors.get(cn, 'gray') for cn in cnames_sorted]
bars = ax.barh(range(len(cnames_sorted)), ds, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(cnames_sorted)))
ax.set_yticklabels(cnames_sorted, fontsize=8)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_xlabel("Cohen's d vs Bare")
ax.set_title('All Conditions vs Bare (d > 0 = better)')
ax.invert_yaxis()

# --- Plot 3: Mechanism ablation summary ---
ax = axes[0, 2]
mech_labels = ['M1:\\nCoherence', 'M2:\\nFormat\\n(Oracle)', 'M3:\\nFormat\\n(LLM)',
               'M4:\\nSpecificity', 'M5:\\nOverlap\\nCeiling', 'M6:\\nLLM vs\\nTF-IDF',
               'M7:\\nStacking']
mech_ds = [comparison_results[k]['cohens_d'] for k in
           ['M1: Shuffled vs LLM-kw', 'M2: Oracle-kw vs Oracle', 'M3: LLM-kw vs LLM-question',
            'M4: TF-IDF vs Anti-kw', 'M5: Echo vs LLM-kw', 'M6: TF-IDF vs LLM-kw',
            'M7: LLM-kw+sep vs best single']]
mech_sig = [comparison_results[k]['bonferroni_significant'] for k in
            ['M1: Shuffled vs LLM-kw', 'M2: Oracle-kw vs Oracle', 'M3: LLM-kw vs LLM-question',
             'M4: TF-IDF vs Anti-kw', 'M5: Echo vs LLM-kw', 'M6: TF-IDF vs LLM-kw',
             'M7: LLM-kw+sep vs best single']]
mech_colors = ['mediumpurple' if s else 'lightgray' for s in mech_sig]
ax.bar(range(len(mech_labels)), mech_ds, color=mech_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(mech_labels)))
ax.set_xticklabels(mech_labels, fontsize=7)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d")
ax.set_title('Mechanism Tests (colored = Bonferroni sig)')

# --- Plot 4: Hardness × condition heatmap ---
ax = axes[1, 0]
hm_conditions = conditions_to_show
hm_data = []
for cname in hm_conditions:
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
im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(quintile_labels, fontsize=7)
ax.set_yticks(range(len(hm_conditions)))
ax.set_yticklabels(hm_conditions, fontsize=7)
for i in range(len(hm_conditions)):
    for j in range(5):
        ax.text(j, i, f"{hm_data[i,j]:+.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label="Cohen's d vs bare")
ax.set_title('Hardness × Condition (d vs bare)')

# --- Plot 5: Cross-condition median overlap vs mean d ---
ax = axes[1, 1]
for i, (cname, okey) in enumerate(cond_to_overlap.items()):
    ax.scatter([cond_median_overlap[i]], [cond_mean_d[i]],
               s=80, c=cond_colors.get(cname, 'gray'),
               edgecolors='black', linewidths=1, zorder=5)
    ax.annotate(cname.replace('_', ' '), (cond_median_overlap[i], cond_mean_d[i]),
                fontsize=7, ha='left', va='bottom')
# Fit line
z = np.polyfit(cond_median_overlap, cond_mean_d, 1)
x_line = np.linspace(min(cond_median_overlap), max(cond_median_overlap), 50)
ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.7)
ax.set_xlabel('Median Token Overlap')
ax.set_ylabel("Mean Cohen's d vs Bare")
ax.set_title(f'Cross-Condition Overlap vs Effect (r={r_cross:.3f})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# --- Plot 6: LLM template ranking bar chart ---
ax = axes[1, 2]
template_conds = [
    ('llm_keyword', 'Keyword'), ('llm_question', 'Question'),
    ('llm_symptom', 'Symptom'), ('llm_summary', 'Summary'),
    ('llm_messy', 'Messy'),
]
tmpl_ds = [cohens_d(c['bare'] - c[cn]) for cn, _ in template_conds]
tmpl_names = [label for _, label in template_conds]
tmpl_colors = ['forestgreen', 'limegreen', 'darkgreen', 'olive', 'teal']
ax.bar(range(len(tmpl_ds)), tmpl_ds, color=tmpl_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(tmpl_ds)))
ax.set_xticklabels(tmpl_names, fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_ylabel("Cohen's d vs Bare")
ax.set_title('LLM Template Ranking')

plt.suptitle('Exp 06: Surrogate Deep-Dive — Mechanism Decomposition', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'analysis_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots saved to {RESULTS_DIR / 'analysis_plots.png'}")\
""")))

# ========== Cell 15: Save results JSON ==========
cells.append(make_cell("code", s("""\
# Cell 15: Save comprehensive results JSON

final = {
    'experiment': 'exp06_surrogate_deep_dive',
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
    'primary_comparisons': comparison_results,
    'all_vs_bare': all_vs_bare,
    'overlap_analysis': {
        'universal_r': float(r_all),
        'universal_p': float(p_all),
        'cross_condition_r': float(r_cross),
        'cross_condition_p': float(p_cross),
        'regression_betas': [float(b) for b in betas],
        'regression_r_squared': float(r_squared),
    },
    'hardness_breakdown': hardness_breakdown,
    'per_sample_results': results,
}

with open(FINAL_RESULTS_PATH, 'w') as f:
    json.dump(final, f, indent=2)

print(f"Results saved to {FINAL_RESULTS_PATH}")
print(f"File size: {FINAL_RESULTS_PATH.stat().st_size / 1024:.1f} KB")
print("\\nDone!")\
""")))

# ========== Cell 16: GPU cleanup ==========
cells.append(make_cell("code", s("""\
# Cell 16: GPU cleanup — free all VRAM
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

output_path = "/home/jupyter/research/directed_kvcache_v2/06_surrogate_deep_dive.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
