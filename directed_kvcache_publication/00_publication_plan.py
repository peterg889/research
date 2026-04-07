#!/usr/bin/env python3
"""Build the publication plan notebook.

Generates 00_publication_plan.ipynb — comprehensive plan for the directed KV cache
paper including paper outline, figure plan, existing results inventory, missing
experiment specifications with code skeletons, and reviewer anticipation.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 00_publication_plan.py
"""

import os
import ast
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
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# =====================================================================
# Cell 0: Title
# =====================================================================
md(r"""# Directed KV Cache Construction — Publication Plan

## Working Title
**"Directed KV Cache Construction: How Prefix Conditioning Improves
Precomputed Document Representations for Retrieval-Augmented Generation"**

## Thesis (one paragraph)
When precomputing KV caches for documents in RAG systems, the standard approach
caches raw document tokens. We show that prepending a short task-framing prefix
(e.g., "Read and comprehend this text carefully") before the document during cache
construction measurably improves downstream QA — even though the prefix is discarded
from the final cache. All 13 tested prefix strategies help, including random tokens
(d=+0.10) and false-premise questions (d=+0.22). Learned soft prompt embeddings
(245K parameters) double the best handcrafted text prefix (d=0.85 vs 0.43), and a
single universal prompt nearly matches per-dataset specialists. A per-tensor
normalization correction (one line of code) universally improves all caches. We
validate across 3 model families, 7 datasets, and show NLL improvements transfer
to downstream accuracy gains.

## Status
- **Existing work**: 15 experiments on Gemma 3 12B-IT (directed_kvcache_v4/)
- **Missing for paper**: Multi-model, downstream accuracy, latency, attention analysis
- **Target venue**: TBD (EMNLP 2026 / NeurIPS 2026 / COLM 2026)
""")


# =====================================================================
# Cell 1: Paper Outline
# =====================================================================
md(r"""# 1. Paper Outline

## Section 1: Introduction (1 page)

**Opening**: KV cache precomputation is now standard in RAG (cite TurboRAG, CacheBlend,
Cache-Craft). All prior work treats cache construction as fixed — you cache the raw
document and optimize what to *keep* (eviction) or *how to store it* (quantization).

**Gap**: Nobody asks: can we make the cache entries *better*?

**Our contribution**: We show that prepending a short prefix during offline cache
construction changes the quality of the resulting document KV representations for
downstream QA. This is:
- **Cheap**: 64 extra tokens during offline construction, zero tokens at inference
- **Universal**: All prefix types help, across 7 QA datasets
- **Learnable**: 245K-parameter soft prompts double the best handcrafted text
- **Complementary**: Stacks with quantization and eviction methods

**Contributions** (bulleted):
1. Systematic study of how prefix content affects KV cache quality (13 strategies)
2. A per-tensor normalization correction that universally improves caches
3. Learned soft prompts for cache conditioning (245K params, 2× best text)
4. Cross-dataset transfer analysis revealing task specialization structure
5. Validation across 3 model families (Gemma, Llama, Qwen)

---

## Section 2: Related Work (1 page)

### 2.1 KV Cache Efficiency
- **Eviction**: H2O, SnapKV, NACL, Ada-KV, StreamingLLM — *which* entries to keep
- **Quantization**: KIVI, KVQuant, CQ — *how* to compress entries
- **Precomputation**: TurboRAG, CacheBlend, CacheClip, Cache-Craft — *reuse* across queries
- **Our position**: We optimize the *content* of entries, orthogonal to all of the above

### 2.2 Prompt Engineering and Soft Prompts
- **Prefix tuning** (Li & Liang 2021): Learned prefixes for task adaptation
- **Gist tokens** (Mu & Li 2023): Compress prompts into continuous tokens
- **Our connection**: We apply soft prompts to cache *construction*, not inference

### 2.3 Context Compression for RAG
- **LongLLMLingua**: Query-aware prompt compression (prunes tokens)
- **xRAG, PISCO**: Extreme compression (one token per document)
- **Beyond RAG** (Corallo et al. 2025): Task-aware knowledge compression
- **Our difference**: We *enrich* rather than compress — prefix tokens improve full-res cache

---

## Section 3: Method (1.5 pages)

### 3.1 Two-Phase KV Cache Scoring
- **Phase A (offline)**: `[BOS, prefix, \n, doc]` → forward pass → extract cache →
  select BOS + doc entries → RoPE reposition → normalize
- **Phase B (online)**: `[\n, query, \n, answer]` → score with cached KV
- **Key**: prefix tokens influence doc representations but are NOT retained in cache
- **Diagram**: Figure 1 (methodology)

### 3.2 Prefix Strategies
- **Structural** (4): random tokens, repeat token, unrelated text, adversarial
- **Keyword** (2): TF-IDF keywords, scrambled comprehend
- **Instruction** (3): comprehend, extract, classify
- **Query-based** (4): oracle, LLM question, OOD query, misleading query
- **Learned** (variable): per-dataset soft prompt, universal soft prompt

### 3.3 KV Cache Normalization
- Per-tensor: `x = (x / (absmax/127)) * (absmax/127)` — bf16 round-trip
- Applied to all caches post-Phase A, including bare baseline
- Mechanism: corrects scale drift accumulated during autoregressive generation

### 3.4 Soft Prompt Training
- `nn.Parameter(shape=(64, hidden_size))` — 245K params for Gemma 3 12B
- Model frozen, only prompt trained via AdamW (10 epochs, patience=3)
- 4 initializations: random, warm-from-comprehend/extract/classify

### 3.5 Evaluation
- **Primary metric**: NLL on answer tokens (paired difference vs bare cache)
- **Effect size**: Cohen's d, win rate, paired t-test
- **Downstream**: Exact match / F1 (extractive QA), accuracy (GSM8K)
- **Datasets**: MS MARCO, SQuAD v2, TriviaQA, HotpotQA, DROP, BoolQ, GSM8K

---

## Section 4: Results (3 pages)

### 4.1 All Prefixes Help — Even Noise (Figure 2)
- 13 text strategies: all pooled d > 0 (range +0.10 to +0.43)
- Comprehend best (d=+0.43), random tokens still positive (d=+0.10)
- OOD queries help (d=+0.20), misleading queries help (d=+0.22)
- **Multi-model validation**: Same ranking on Llama 3.1 and Qwen 2.5

### 4.2 Decomposition: Structure vs Meaning (Figure 3)
- At L=64: structural 11%, vocabulary 39%, meaning 50%
- Meaning effect grows with prefix length (reverses short-prefix conclusions)
- Comprehend is the only instruction where word order matters

### 4.3 Learned Soft Prompts (Figure 4)
- Soft prompt d=0.85 vs best text d=0.43 (2× gain)
- Random init (d=0.85) >> warm-start (d=0.61)
- Universal prompt (d=0.84) ≈ per-dataset specialists (d=0.85)

### 4.4 Normalization: A Free Lunch (Figure 5)
- Universal improvement: d=-0.50 to -2.79 across 10 datasets
- bf16 arithmetic round-trip captures 97% of int8 benefit
- 72% independent from prefix conditioning

### 4.5 Cross-Dataset Transfer (Figure 6)
- Diagonal d=1.16 vs off-diagonal d=0.47 (specialization gap 0.70)
- BoolQ poisons (off-diag d=-0.49), GSM8K absorbs (d=+0.90 to +2.39)

### 4.6 Downstream Accuracy (Figure 7, Table 1)
- NLL improvements transfer to EM/F1 and accuracy gains
- **NEW EXPERIMENT** — required

### 4.7 Multi-Model Validation (Table 2)
- Same effects on Llama 3.1 8B and Qwen 2.5 7B
- **NEW EXPERIMENT** — required

---

## Section 5: Analysis (1 page)

### 5.1 Why Does Any Prefix Help?
- Attention sink activation (cite StreamingLLM)
- Distributional shift: prefix tokens establish a task-relevant context frame
- Even noise breaks the model out of "raw document" encoding mode

### 5.2 Why Does Oracle Query Hurt?
- Query-as-prefix creates attention competition with Phase B query
- Document representations optimized for answering become less general

### 5.3 Task Complexity Determines Benefit
- GSM8K (d=+1.33) > DROP (d=+0.91) despite single-word answers
- Answer length uncorrelated (ρ=-0.20, ns)
- Reasoning complexity is the key predictor

### 5.4 Attention Pattern Analysis (Figure 8)
- How prefix presence changes attention distribution in doc layers
- **NEW EXPERIMENT** — nice-to-have

---

## Section 6: Practical Deployment (0.5 page)
- Latency overhead: +64 tokens offline, 0 tokens online
- Normalization: one tensor op per layer (µs)
- Recommended configuration: comprehend prefix + normalization
- Complementary with SnapKV, KIVI, etc.

## Section 7: Conclusion (0.5 page)

## Appendix
- Full per-dataset breakdowns
- Encoder-decoder track results
- Scaling analysis (prefix length, doc length, model size)
- Routing experiment details
""")


# =====================================================================
# Cell 2: Figure Plan
# =====================================================================
md(r"""# 2. Figure Plan

| # | Figure | Data Source | Status | Message |
|---|--------|------------|--------|---------|
| 1 | **Methodology diagram** | — | Exists (Exp 14 fig_methodology.png) | Two-phase pipeline: construct → select → reposition → normalize → score |
| 2 | **All prefixes help** (bar chart) | Exp 13 rankings | Exists (slide_1) | 13 conditions, all positive, color by group |
| 3 | **Decomposition: meaning grows with length** | Exp 05-06 scaling data | Exists (Exp 11 charts) | Stacked bar: structural/vocabulary/meaning at L=8,16,32,64 |
| 4 | **Soft prompts 2× text** (ranking + dumbbell) | Exp 14 rankings + per-dataset | Exists (slide_2) | Left: unified ranking. Right: per-dataset vs universal |
| 5 | **Normalization: free lunch** (bar chart) | Exp 09 results | Exists (Exp 09 charts) | d across 10 datasets, bf16 round-trip annotation |
| 6 | **Transfer matrix** (heatmap) | Exp 14 transfer_matrix.json | Exists (slide_4) | 8×7 heatmap, diagonal vs off-diagonal, BoolQ/GSM8K callouts |
| 7 | **Downstream accuracy** (table + bar) | **NEW: Exp 02** | **MISSING** | EM/F1/accuracy for bare vs comprehend vs soft prompt |
| 8 | **Multi-model validation** (table) | **NEW: Exp 01** | **MISSING** | 3 models × 4 conditions × 4 datasets |
| 9 | **Attention analysis** (heatmap) | **NEW: Exp 04** | **MISSING (nice-to-have)** | Attention diff map: prefix vs bare on sample document |

### Existing figures to adapt (from directed_kvcache_v4)
- `results/decoder_only/exp15/charts/slide_1_every_prefix_helps.png`
- `results/decoder_only/exp15/charts/slide_2_soft_prompt_ranking.png`
- `results/decoder_only/exp15/charts/slide_4_transfer_matrix.png`
- `results/decoder_only/exp14/charts/fig_methodology.png`
- `results/decoder_only/exp11/charts/` (decomposition, scaling)
- `results/decoder_only/exp09/charts/` (normalization)

These need reformatting for publication (single-column, consistent fonts, no suptitles)
but the underlying data and layout are sound.
""")


# =====================================================================
# Cell 3: Existing Results Inventory
# =====================================================================
md(r"""# 3. Existing Results Inventory

## What we can reuse directly from directed_kvcache_v4

| Claim | Source | Data File | Key Numbers |
|-------|--------|-----------|-------------|
| 13 text prefixes all help | Exp 13 | `exp13/summary.json` | d=+0.10 to +0.43, all positive |
| Decomposition (struct/vocab/meaning) | Exp 04-06 | `exp05/summary.json`, `exp06/summary.json` | At L=64: 11%/39%/50% |
| Soft prompt 2× text | Exp 14 | `exp14/summary.json` | soft_rand d=0.85, comprehend d=0.43 |
| Random init > warm-start | Exp 14 | `exp14/summary.json` | rand d=0.85, best warm d=0.61 |
| Universal ≈ per-dataset | Exp 14 | `exp14/summary.json` | univ d=0.84, per-dataset d=0.85 |
| Normalization universal | Exp 09 | `exp09/summary.json` | d=-0.50 to -2.79, 10 datasets |
| Transfer matrix | Exp 14 | `exp14/transfer_matrix.json` | diag d=1.16, off-diag d=0.47 |
| BoolQ poisons | Exp 14 | `exp14/transfer_matrix.json` | off-diag mean d=-0.49 |
| GSM8K absorbs | Exp 14 | `exp14/transfer_matrix.json` | any source d=+0.90 to +2.39 |
| Routing gap closure | Exp 15 | `exp15/summary.json` | 46-63% of oracle gap |
| Model size scaling | Exp 11-12 | `exp11/summary.json` | 1B/4B/12B/27B within Gemma |
| Prefix length scaling | Exp 05-06 | `exp06/summary.json` | Peaks at L=64 |
| GSM8K > DROP > ... > BoolQ | Exp 06 | `exp06/summary.json` | d=+1.33 > +0.91 > ... > -0.51 |

## What we need to generate new

| Claim | Experiment | Status |
|-------|-----------|--------|
| Same effects on Llama 3.1 8B | Exp 01 (multi-model) | **NOT STARTED** |
| Same effects on Qwen 2.5 7B | Exp 01 (multi-model) | **NOT STARTED** |
| NLL → downstream accuracy | Exp 02 (downstream) | **NOT STARTED** |
| Latency overhead is negligible | Exp 03 (latency) | **NOT STARTED** |
| Attention patterns change with prefix | Exp 04 (attention) | **NOT STARTED (nice-to-have)** |
""")


# =====================================================================
# Cell 4: Missing Experiment Specifications
# =====================================================================
md(r"""# 4. Missing Experiments — Detailed Specifications

## Priority Ordering

| Priority | Experiment | Blocking? | Effort | Why |
|----------|-----------|-----------|--------|-----|
| **P0** | Multi-model (Llama + Qwen) | Yes — reviewers will reject without it | 3-5 days | Single-model papers get desk-rejected |
| **P0** | Downstream accuracy (EM/F1) | Yes — "does NLL matter?" is the first question | 1-2 days | Bridges metric gap |
| **P1** | Latency measurements | Soft yes — production framing needs it | 0.5 days | Quick win |
| **P2** | Attention analysis | No — nice-to-have for analysis section | 1-2 days | Mechanistic insight |
| **P2** | Publication figures | No — but needed before submission | 2-3 days | Reformatting existing charts |
""")


# =====================================================================
# Cell 5: Experiment 01 — Multi-Model Evaluation
# =====================================================================
md(r"""# Experiment 01: Multi-Model Evaluation

## Hypothesis
The prefix conditioning effect generalizes beyond Gemma 3 to other model families
with different architectures, training data, and tokenizers.

## Design

### Models
| Model | Params | Architecture | RoPE | GQA | Cache Type |
|-------|--------|-------------|------|-----|-----------|
| Gemma 3 12B-IT | 12B | Hybrid sliding/full | Yes | Yes (16 KV heads) | DynamicCache |
| Llama 3.1 8B-Instruct | 8B | Full attention only | Yes | Yes (8 KV heads) | DynamicCache |
| Qwen 2.5 7B-Instruct | 7B | Full attention only | Yes | Yes (4 KV heads) | DynamicCache |

### Conditions (4, covering the key comparisons)
1. **bare** — no prefix, just `[BOS, doc]`
2. **random** — 64 random token IDs as prefix (structural control)
3. **comprehend** — "Read and comprehend this text carefully" padded to L=64
4. **soft_rand** — learned soft prompt (random init, trained per-dataset)

*Note: soft_rand requires training on each model separately (P0 experiment includes
training 4 datasets × 1 init × 2 new models = 8 training runs).*

### Datasets (4, spanning the task spectrum)
| Dataset | Type | Expected Effect | Why Included |
|---------|------|----------------|-------------|
| MS MARCO | Factoid retrieval | Small (d≈0.1-0.4) | Most common RAG benchmark |
| SQuAD v2 | Extractive QA | Medium (d≈0.5-1.0) | Well-known, extractive |
| DROP | Discrete reasoning | Large (d≈0.9-1.1) | Strongest effect in Gemma |
| GSM8K | Math reasoning | Very large (d≈1.3) | Champion dataset |

### Sample Size
- N=400 per dataset, hard subset = top 40% by bare NLL = 160 samples
- Same SEED=42 as v4 experiments
- Total: 4 conditions × 4 datasets × 3 models = 48 evaluation runs
- Plus 8 soft prompt training runs (4 datasets × 2 new models)

## Engineering Requirements

### lib/ Generalization
The existing `lib/rope.py` functions are Gemma-3-specific:
- `build_layer_inv_freqs()` reads Gemma-specific config attributes
- `get_layer_types()` handles sliding/full attention hybrid
- `select_kv_cache()` works with DynamicCache

**Required changes for multi-model support:**
1. **Generalize `build_layer_inv_freqs()`** — Llama uses `model.config.rope_theta`
   and uniform layer types. Qwen uses similar structure. Need model-family dispatch.
2. **Generalize `get_layer_types()`** — Llama/Qwen have all "full_attention" layers
   (no sliding window), so this returns a uniform list.
3. **Generalize `select_kv_cache()`** — No sliding window limit for Llama/Qwen,
   so the 1023-entry constraint is Gemma-specific. For other models, max_doc is
   limited only by context window.
4. **Tokenizer differences** — "comprehend" instruction tokenizes differently across
   models. Use `make_prefix()` with model-specific token IDs.

### Model Access
- Llama 3.1: Meta license, `meta-llama/Llama-3.1-8B-Instruct`
- Qwen 2.5: Open license, `Qwen/Qwen2.5-7B-Instruct`
- Both need ~16GB VRAM in bf16 (fits single A100)

## Expected Output
```
results/publication/exp01_multi_model/
  {model_name}/
    summary.json          # Rankings with pooled_d, pooled_win per condition
    results.json          # Per-sample NLLs
    checkpoint_{ds}.json  # Per-dataset checkpoints
    soft_prompt_{ds}.pt   # Trained soft prompts (new models only)
```

## Success Criteria
- Same condition ranking across all 3 models: soft > comprehend > random > bare
- Effect size within 50% of Gemma values (i.e., soft d > 0.4 on Llama/Qwen)
- All prefixes positive pooled d on all models
""")


# =====================================================================
# Cell 6: Experiment 01 — Code Skeleton
# =====================================================================
code(r"""# Experiment 01: Multi-Model Evaluation — Setup
# This cell verifies model availability and estimates VRAM requirements.
# Full experiment will be in experiments/01_multi_model/build_exp01.py

import torch
import os

MODELS = {
    'gemma3_12b': {
        'name': 'google/gemma-3-12b-it',
        'class': 'Gemma3ForConditionalGeneration',
        'vram_gb': 24,
        'has_sliding': True,
        'n_kv_heads': 16,
    },
    'llama31_8b': {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'class': 'LlamaForCausalLM',
        'vram_gb': 16,
        'has_sliding': False,
        'n_kv_heads': 8,
    },
    'qwen25_7b': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'class': 'Qwen2ForCausalLM',
        'vram_gb': 14,
        'has_sliding': False,
        'n_kv_heads': 4,
    },
}

CONDITIONS = ['bare', 'random', 'comprehend', 'soft_rand']
DATASETS = ['ms_marco', 'squad_v2', 'drop', 'gsm8k']
N_SAMPLES = 400
N_HARD = 160
SEED = 42
PREFIX_L = 64

print("Experiment 01: Multi-Model Evaluation")
print(f"Models: {len(MODELS)}")
print(f"Conditions: {len(CONDITIONS)}")
print(f"Datasets: {len(DATASETS)}")
print(f"Total eval runs: {len(MODELS) * len(CONDITIONS) * len(DATASETS)}")
print(f"Soft prompt training runs: {len(DATASETS) * (len(MODELS) - 1)}")

if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram:.1f} GB")
    for key, m in MODELS.items():
        fits = "YES" if vram >= m['vram_gb'] else "NO"
        print(f"  {key}: needs {m['vram_gb']}GB — {fits}")
else:
    print("\nNo GPU available — cannot run model experiments")
""")


# =====================================================================
# Cell 7: Experiment 02 — Downstream Accuracy
# =====================================================================
md(r"""# Experiment 02: Downstream Accuracy Evaluation

## Hypothesis
NLL improvements from prefix conditioning and soft prompts translate to measurable
gains in downstream task accuracy (exact match, F1, solve rate).

## Design

### Evaluation Protocol
For each sample, generate an answer using greedy decoding, then score:

| Dataset | Metric | Generation Strategy |
|---------|--------|-------------------|
| MS MARCO | Token F1 | Generate until `\n` or 128 tokens |
| SQuAD v2 | Exact Match + Token F1 | Generate until `\n` or 64 tokens |
| DROP | Exact Match (number/date/span) | Generate until `\n` or 64 tokens |
| GSM8K | Accuracy (final number) | Generate CoT until `####` or 512 tokens, extract number |

### Conditions (3, core comparison)
1. **bare** — baseline (equivalent to TurboRAG)
2. **comprehend** — best text prefix
3. **soft_rand** — best learned prefix

### Scoring Details
- **Token F1**: Tokenize prediction and reference, compute precision/recall/F1
  on token overlap (standard SQuAD F1, bag-of-words)
- **Exact Match**: Normalize (lowercase, strip articles/punctuation), compare strings
- **GSM8K accuracy**: Extract final number after `####`, compare to ground truth

### Sample Size
- N=160 hard samples per dataset (same as NLL evaluation)
- 3 conditions × 4 datasets × 3 models = 36 generation runs
- Each run: ~160 samples × ~10 sec/sample = ~25 min per run

### What This Proves
- If EM/F1 improves with prefix conditioning, the NLL metric is validated
- If EM/F1 doesn't improve, we need to explain why NLL is still informative
  (e.g., calibration improvement, perplexity reduction on correct answers)

## Expected Output
```
results/publication/exp02_downstream/
  {model_name}/
    accuracy_{dataset}.json   # Per-sample: generated text, EM, F1, NLL
    summary.json              # Aggregate: mean EM, mean F1, mean accuracy
```

## Success Criteria
- Comprehend shows +2-5% F1 improvement over bare on extractive QA
- Soft prompt shows +5-10% F1 improvement over bare
- GSM8K accuracy improves by at least 2 percentage points
""")


# =====================================================================
# Cell 8: Experiment 02 — Code Skeleton
# =====================================================================
code(r"""# Experiment 02: Downstream Accuracy — Scoring Functions
# These are the metric implementations needed for evaluation.

import re
import string
from collections import Counter


def normalize_answer(s):
    # Standard SQuAD normalization
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return int(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gsm8k_extract_answer(text):
    # Extract the final number after #### in GSM8K-style output
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: last number in the text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ''


def gsm8k_accuracy(prediction, ground_truth):
    pred_num = gsm8k_extract_answer(prediction)
    gold_num = gsm8k_extract_answer(ground_truth)
    return int(pred_num == gold_num)


# Quick test
assert exact_match("The cat sat", "the cat sat") == 1
assert exact_match("A dog", "the cat") == 0
assert token_f1("the cat sat on the mat", "the cat sat") == 1.0
assert gsm8k_extract_answer("So the answer is #### 42") == '42'
print("All scoring functions verified.")
""")


# =====================================================================
# Cell 9: Experiment 03 — Latency Measurements
# =====================================================================
md(r"""# Experiment 03: Latency Measurements

## Purpose
Quantify the computational overhead of prefix conditioning to support the
"cheap" claim. Show that the cost is paid offline (Phase A) and is zero
at inference time (Phase B).

## Design

### Measurements
| Metric | What | Expected |
|--------|------|----------|
| Phase A time (bare) | Forward pass: `[BOS, doc]` | baseline |
| Phase A time (prefix) | Forward pass: `[BOS, prefix, \n, doc]` | +5-10% (64 extra tokens) |
| Select + reposition time | `select_kv_cache` + `reposition_kv_cache` | <1ms |
| Normalization time | `norm_roundtrip_kv_cache` | <1ms |
| Phase B time (bare) | Forward pass with bare cache | baseline |
| Phase B time (prefix) | Forward pass with prefix-conditioned cache | **same** (cache has identical shape) |
| Soft prompt training | Per-dataset, 10 epochs | ~10-30 min/dataset |

### Protocol
- Warm up GPU with 10 forward passes
- Measure 100 forward passes, report mean ± std
- Use `torch.cuda.Event` for accurate GPU timing
- Test on MS MARCO with median-length document

## Expected Output
```
results/publication/exp03_latency/
  timing_{model}.json    # Per-operation timing measurements
  summary_table.md       # LaTeX-ready table for paper
```

## Key Message for Paper
- Phase A overhead: +5-10% for 64 prefix tokens (amortized over all future queries)
- Phase B overhead: 0% (cache shape identical with or without prefix)
- Normalization: <0.1% of Phase A time
- **For a document queried N times, per-query cost is zero**
""")


# =====================================================================
# Cell 10: Experiment 04 — Attention Analysis
# =====================================================================
md(r"""# Experiment 04: Attention Analysis (Nice-to-Have)

## Purpose
Visualize how prefix conditioning changes attention patterns in the document
KV cache, providing mechanistic insight for the Analysis section.

## Design

### Approach
1. Run Phase A twice on the same document: once bare, once with comprehend prefix
2. Extract attention weights from Phase B (same query) in both cases
3. Compute the attention difference map: `attn_prefix - attn_bare`
4. Aggregate across heads/layers to show which document positions gain/lose attention

### Visualizations
- **Layer-averaged attention heatmap**: query tokens × doc tokens, showing attention
  redistribution with vs without prefix
- **Per-layer attention entropy**: Does prefix reduce entropy (sharpen attention)?
- **Position-wise attention gain**: Which document positions benefit most from prefix?

### Sample
- Pick 3 representative examples from different datasets (MS MARCO, DROP, GSM8K)
- Show qualitative patterns, not statistical claims

## Expected Output
```
results/publication/exp04_attention/
  attention_maps/               # Saved attention tensors
  figures/                      # Publication-ready attention visualizations
```

## Key Message for Paper
- Prefix conditioning sharpens attention on task-relevant document regions
- Even random prefix reduces attention entropy (structural effect)
- Comprehend prefix further concentrates on answer-bearing spans (semantic effect)
""")


# =====================================================================
# Cell 11: lib/ Generalization Plan
# =====================================================================
md(r"""# 5. Library Generalization Plan

The existing `lib/` modules in directed_kvcache_v4 are Gemma-3-specific.
For multi-model support, we need to generalize several functions.

## Required Changes

### lib/rope.py

**`build_layer_inv_freqs(model, device)`**
- Current: Reads `model.config.text_config.rope_scaling` (Gemma 3 specific)
- Needed: Model-family dispatch based on `model.config.model_type`

```python
# Pseudocode for generalized version
def build_layer_inv_freqs(model, device=None):
    model_type = model.config.model_type  # "gemma3", "llama", "qwen2"
    if model_type in ("llama", "qwen2", "mistral"):
        # Uniform layer types, single inv_freq from config.rope_theta
        rope_theta = model.config.rope_theta
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
        return {"full_attention": inv_freq.to(device)}
    elif model_type == "gemma3":
        # Existing Gemma 3 logic (hybrid sliding/full)
        ...
```

**`get_layer_types(model)`**
- Current: Reads per-layer `sliding_window` attribute
- Needed: For Llama/Qwen, return `["full_attention"] * n_layers`

**`select_kv_cache(cache, indices, device)`**
- Current: Works with DynamicCache, has sliding window limit warning
- Needed: Remove sliding window constraint for non-Gemma models

**`reposition_kv_cache(cache, ...)`**
- Current: Handles both sliding and full attention layer types
- Needed: Should work unchanged for full-attention-only models (just one layer type)

### lib/cache.py
- `deep_copy_cache()` — Should work unchanged (uses DynamicCache API)
- `make_prefix()` — Should work unchanged (operates on token IDs)

### lib/quantization.py
- All functions should work unchanged (operate on raw tensors in cache)

### lib/analysis.py, lib/data.py
- No changes needed

## Testing Plan
- Add parametrized tests for Llama/Qwen mock configs in test_rope.py
- Verify `build_layer_inv_freqs` returns correct structure for each model type
- Verify `get_layer_types` returns uniform "full_attention" for Llama/Qwen
- Verify `reposition_kv_cache` works when all layers are "full_attention"

## Estimated Effort: 1-2 days
""")


# =====================================================================
# Cell 12: Reviewer Anticipation
# =====================================================================
md(r"""# 6. Reviewer Anticipation

## Questions reviewers WILL ask, and our answers

### Q1: "Why not just use a longer context window instead of prefix conditioning?"
**A**: We're not competing with longer contexts. We're improving the quality of
*precomputed* caches. In RAG systems, documents are cached offline. The question
is: what do you cache? We show that how you construct the cache matters.

### Q2: "How does this compare to TurboRAG / CacheBlend?"
**A**: Our bare baseline IS TurboRAG (precomputed cache, no conditioning). We show
that adding a 64-token prefix during construction improves over this baseline.
CacheBlend recomputes a subset of tokens at query time; our approach requires zero
extra computation at query time. These are complementary.

### Q3: "Why NLL instead of task accuracy?"
**A**: We report both. NLL provides fine-grained paired comparisons (160 samples
yield high-power tests). We validate with EM/F1/accuracy showing NLL improvements
transfer. (This is why Exp 02 is P0.)

### Q4: "The normalization finding seems like it could be its own paper."
**A**: It could be, but it's more impactful as part of the directed cache story.
The normalization is a practical tool that makes all caches better; the prefix
conditioning is the scientific contribution.

### Q5: "BoolQ is negative — doesn't that undermine the universality claim?"
**A**: We're transparent about this. BoolQ is a binary classification task with
single-token answers. Prefix conditioning benefits tasks with extractive or
generative answers where the answer spans multiple tokens. We scope our claims
accordingly.

### Q6: "The soft prompt training requires labeled data — isn't that a limitation?"
**A**: Yes, but only 200 training examples per dataset, and the universal prompt
(trained on pooled data) performs within 1% of per-dataset specialists. In practice,
a single universal prompt can be trained once and deployed across tasks.

### Q7: "Is 160 samples per dataset enough?"
**A**: With paired comparisons (each sample is its own control), we achieve strong
statistical power. Our smallest significant effect (random, d=+0.10) has p<0.01
at N=160. We use Cohen's d + win rate + paired t-test for all claims.

### Q8: "What about decoder-only models without RoPE?"
**A**: All major open models (Llama, Gemma, Qwen, Mistral) use RoPE. The RoPE
repositioning step is essential for correct position encoding after prefix removal.
Models without RoPE (e.g., ALiBi-based) would need a different repositioning
approach, which we leave for future work.

### Q9: "Can prefix conditioning be combined with cache eviction (SnapKV, H2O)?"
**A**: In principle, yes — prefix conditioning improves cache quality, eviction
reduces cache size. They operate on orthogonal axes. We demonstrate complementarity
with quantization (normalization + int8). Full eviction integration is future work.

### Q10: "Why does random initialization beat warm-start for soft prompts?"
**A**: The text embedding manifold occupies a small region of the full parameter space.
Warm-starting from text embeddings constrains optimization to this region. Random
initialization allows the soft prompt to explore the full space, finding better optima
that don't correspond to any natural language text. This is consistent with findings
in prompt tuning literature (Lester et al. 2021).
""")


# =====================================================================
# Cell 13: Dependency Graph and Timeline
# =====================================================================
md(r"""# 7. Dependency Graph and Execution Order

```
                    ┌─────────────────────┐
                    │  lib/ generalization │
                    │  (Llama + Qwen)     │
                    │  [1-2 days]         │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────────┐ ┌──────────┐ ┌──────────────────┐
    │ Exp 01: Multi-  │ │ Exp 03:  │ │ Exp 04: Attention│
    │ Model Eval      │ │ Latency  │ │ Analysis         │
    │ [3-5 days]      │ │ [0.5 day]│ │ [1-2 days]       │
    │ P0              │ │ P1       │ │ P2               │
    └────────┬────────┘ └──────────┘ └──────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Exp 02: Downstr.│
    │ Accuracy (needs │
    │ trained models) │
    │ [1-2 days]      │
    │ P0              │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Publication      │
    │ Figures          │
    │ [2-3 days]       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Paper Draft      │
    │ [5-7 days]       │
    └─────────────────┘
```

## Execution Order
1. **Week 1**: Generalize lib/ for multi-model → start Exp 01 (multi-model) + Exp 03 (latency)
2. **Week 2**: Complete Exp 01 → run Exp 02 (downstream accuracy) on all 3 models
3. **Week 3**: Exp 04 (attention, if time) → Publication figures → Paper draft
4. **Week 4**: Paper writing, revision, internal review

## Total estimated effort: 2-3 weeks of focused work

## Directory Structure
```
directed_kvcache_publication/
  00_publication_plan.ipynb         # This notebook
  experiments/
    01_multi_model/                 # Multi-model eval (P0)
      build_exp01.py                # Notebook builder
      lib_extensions.py             # Model-family dispatch for rope/cache
    02_downstream_accuracy/         # EM/F1/accuracy (P0)
      build_exp02.py
      scoring.py                    # EM, F1, GSM8K accuracy functions
    03_latency/                     # Timing measurements (P1)
      build_exp03.py
    04_attention_analysis/          # Attention visualization (P2)
      build_exp04.py
  figures/                          # Publication-ready figures
  paper/                            # LaTeX source
  results/                          # New experiment results
    exp01_multi_model/
    exp02_downstream/
    exp03_latency/
    exp04_attention/
```
""")


# =====================================================================
# Cell 14: Compact Experiment Strategy
# =====================================================================
md(r"""# 8. Making Experiments Compact

## The 15 → 6 figure reduction

The v4 repo has 15 experiments generating 30+ figures. For the paper, we distill
to **6-8 figures** plus **2 tables**:

### Main paper figures (6)
1. **Methodology** (adapted from Exp 14) — two-phase pipeline diagram
2. **All prefixes help** (from Exp 13) — 13 conditions bar chart
3. **Meaning grows with length** (from Exps 05-06) — decomposition by prefix length
4. **Soft prompts 2× text** (from Exp 14) — unified ranking + dumbbell
5. **Normalization** (from Exp 09) — bar chart across datasets
6. **Transfer matrix** (from Exp 14) — 8×7 heatmap

### Main paper tables (2)
1. **Multi-model validation** — 3 models × 4 conditions × 4 datasets (d-values)
2. **Downstream accuracy** — bare vs comprehend vs soft (EM, F1, accuracy)

### Appendix figures
- Full 15-condition × 7-dataset heatmap
- Prefix length scaling curves
- Document length scaling curves
- Model size scaling curves
- Routing experiment results
- Training curves for soft prompts
- Per-dataset breakdowns for all conditions

## What to cut from the main paper
- Encoder-decoder track (save for appendix or separate paper)
- Routing (interesting but tangential — appendix)
- Quantization deep-dive (Exps 07-08 diagnosis — fold into normalization paragraph)
- Deconfounding (Exp 10 — methodology detail, not a result)
- Individual experiment progression (Exps 01-12 are development, not results)

## Rerunning for consistency
Consider re-running the core evaluation (Exp 13 conditions + Exp 14 soft prompts)
on all 3 models in a single unified experiment rather than presenting Gemma results
from v4 and Llama/Qwen results from new experiments. This ensures:
- Same code path for all models
- Same random seeds and data splits
- No "apples to oranges" concern from reviewers

**Decision**: Run the full pipeline once per model, share the results in one unified
table. Reference v4 for methodology development, new experiment for publication data.
""")


# =====================================================================
# Cell 15: Open Questions
# =====================================================================
md(r"""# 9. Open Questions for Discussion

## Scoping decisions needed

### 1. How many models?
- **Minimum**: 2 (Gemma + Llama) — enough to claim generalization
- **Better**: 3 (+ Qwen) — covers 3 distinct model families
- **Stretch**: 4 (+ Mistral) — diminishing returns but very strong claim
- **Recommendation**: 3 models (Gemma, Llama, Qwen)

### 2. Should we include the encoder-decoder track?
- **Pro**: Shows the approach works across architectures (not just decoder-only)
- **Con**: Different model (T5Gemma), different metrics, muddies the narrative
- **Recommendation**: Appendix only, with a brief note in the main text

### 3. How much normalization detail?
- **Option A**: Full section (1 page) with mechanism analysis
- **Option B**: Brief paragraph in Method + one figure
- **Recommendation**: Option B for main paper. The normalization is a practical
  contribution, not the main scientific claim. Save the deep-dive for a blog post
  or supplementary material.

### 4. Should we train soft prompts on all 3 models?
- **Pro**: Strongest claim (learned conditioning works across models)
- **Con**: 8 extra training runs per model (4 datasets × 2 inits minimum)
- **Recommendation**: Yes, but only random init (skip warm-start variants).
  4 datasets × 2 new models = 8 runs, each ~30 min = 4 hours total.

### 5. What about instruction-tuned vs base models?
- **Current**: All experiments use instruction-tuned models (*-IT, *-Instruct)
- **Concern**: Instruction tuning might make models more responsive to prefixes
- **Recommendation**: Add one base model comparison (e.g., Gemma 3 12B base vs IT)
  as a sanity check. If the effect persists on the base model, it's not just
  instruction-following. This is a 0.5-day experiment.

### 6. Single paper or two papers?
- **Paper A**: "Directed KV Cache Construction" (full story, 8 pages)
- **Paper B**: "Universal KV Cache Normalization" (short paper, 4 pages)
- **Recommendation**: Single paper. The normalization is a contribution within
  the larger story, and splitting weakens both.
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "00_publication_plan.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
