# Directed KV Cache Research Project

## Overview
Research on priming document KV caches with surrogate queries for ad-serving.
Model: Mistral-7B-Instruct-v0.2 (4-bit). Dataset: MS MARCO v1.1.

## CRITICAL: Read Before Creating New Notebooks
**Always read `EXPERIMENTAL_NOTES.md` before designing new experiments.**
It contains the full experimental history, all known bugs, and lessons learned.

**Also read `FUTURE_EXPERIMENTS.md`** for proposed next experiments (16-21) and the
reasoning behind them. Prioritized: prefix composition (Exp 16), practical hardness gate (Exp 18).

## Known Pitfalls (repeatedly cause bugs)

0. **File permissions — two-user environment**: JupyterLab runs as `jupyter` (uid=1000),
   while the CLI runs as `petergrabowski_google_com`. They share no group.

   **For notebooks**: Call `os.umask(0o000)` near the top (before any file I/O).

   **For CLI**: When creating files/directories, prefix with `umask 000 &&`:
   ```bash
   umask 000 && mkdir -p results/exp15
   umask 000 && touch results/exp15/checkpoint.json
   ```

   **If you get PermissionError**: Run the fix script:
   ```bash
   bash fix_permissions.sh
   ```
   Or manually: `find . -type f -exec chmod 666 {} \; && find . -type d -exec chmod 777 {} \;`

   **Why this happens**: Each CLI command runs in a fresh shell with default umask 022.
   The user's .bashrc isn't consistently sourced. Always use explicit `umask 000 &&`.

1. **BPE boundary mismatch**: Tokenizing `prefix + passage` together produces different
   BPE tokens than tokenizing them separately. **Always use `build_matched_bare_and_truncated()`**
   from the notebook helpers (or `build_matched_caches` from lib). Never compare a bare cache
   built from independently-tokenized passage against a truncated cache from concatenated input.

2. **`score_answer_with_cache()` mutates its cache**: The function extends the cache in-place
   via `use_cache=True`. If scoring multiple queries against the same cache, you must
   deep copy it before each call. **Use `from lib.kv_cache import deepcopy_cache`** — do NOT
   use `copy.deepcopy()` or manual iteration, as DynamicCache's iteration API varies across
   transformers versions.

3. **DynamicCache API varies across transformers versions**: **NEVER access cache internals directly.**
   The attributes `.key_cache`, `.value_cache`, and iteration behavior change between versions.

   **ALWAYS use the lib helper functions:**
   ```python
   from lib.kv_cache import (
       deepcopy_cache,           # Safe cache copying
       _get_cache_keys,          # Get keys: _get_cache_keys(cache, layer_idx)
       _get_cache_values,        # Get values: _get_cache_values(cache, layer_idx)
       _set_cache_keys,          # Set keys: _set_cache_keys(cache, layer_idx, tensor)
       _set_cache_values,        # Set values: _set_cache_values(cache, layer_idx, tensor)
       _ensure_dynamic_cache,    # Convert any cache format to DynamicCache
   )

   # WRONG - breaks on different transformers versions:
   k = cache.key_cache[layer_idx]
   for layer_idx, (k, v) in enumerate(cache):

   # RIGHT - works on all versions:
   k = _get_cache_keys(cache, layer_idx)
   for layer_idx in range(len(cache)):
       k = _get_cache_keys(cache, layer_idx)
       v = _get_cache_values(cache, layer_idx)
   ```

   **Why this keeps breaking:** New notebooks define cache-manipulation functions inline instead
   of using lib/kv_cache.py. The fix is to ALWAYS import from lib, never write raw cache access.

4. **"Document:\n" framing hurts**: Adding "Document:\n" before passages increases NLL (d=-0.45).
   Always use bare passage text with no framing for baselines.

5. **RoPE uses half-split, not interleaved**: HuggingFace's `rotate_half` splits dimensions as
   `x[..., :d/2]` and `x[..., d/2:]`, NOT interleaved `x[..., 0::2]` and `x[..., 1::2]`.

6. **BOS token must be preserved**: Truncated caches must keep the BOS token at position 0.
   Use `extract_and_truncate_cache_with_bos()`.

7. **Attention mask dtype must match model dtype**: When using custom attention masks with SDPA,
   the mask dtype must match the query dtype. BitsAndBytes 4-bit models often use `bfloat16`
   internally even if `bnb_4bit_compute_dtype=torch.float16`. **Always use `dtype=model.dtype`**
   when creating masks, or cast before passing to model:
   ```python
   # WRONG: hardcoded dtype
   mask = torch.full((seq, seq), float('-inf'), dtype=torch.float16, device=model.device)

   # RIGHT: use model's dtype
   mask = torch.full((seq, seq), float('-inf'), dtype=model.dtype, device=model.device)

   # OR cast before use
   mask = mask.to(dtype=model.dtype)
   ```

## Directory Structure

All experiment outputs (checkpoints, result JSONs, plots, analysis scripts) are stored in
per-experiment subdirectories under `results/`:

```
results/
  exp01/   — outputs from 01_directed_kvcache_experiment.ipynb
  exp04/   — outputs from 04_directed_kvcache_corrected_routing.ipynb
  exp05/   — outputs from 05_directed_kvcache_bugfix_rerun.ipynb
  ...
  exp14/   — outputs from 14_isolate_and_amplify.ipynb
```

**IMPORTANT: Always save new experiment outputs to `results/expXX/`**, never to the project
root. When creating a new experiment notebook (e.g., `15_foo.ipynb`), create `results/exp15/`
and point all `json.dump()`, `savefig()`, and checkpoint paths there. This keeps the project
root clean — only notebooks, documentation, config files, and subdirectories belong at the top level.

## Library Structure
- `lib/kv_cache.py` — Cache building, truncation, RoPE correction, scoring
- `lib/block_attention.py` — Block-diagonal and query-time attention masks
- `lib/surrogate.py` — Surrogate query generation templates and functions
- `lib/data.py` — MS MARCO dataset loading
- `lib/config.py` — ExperimentConfig dataclass
- `lib/analysis.py` — Statistical analysis utilities (ranking metrics, token overlap)

### Standard Import Pattern for Cache Manipulation

**Copy this block into any notebook that manipulates KV caches:**

```python
from lib.kv_cache import (
    # Cache access (ALWAYS use these, never access .key_cache/.value_cache directly)
    deepcopy_cache,
    _get_cache_keys,
    _get_cache_values,
    _set_cache_keys,
    _set_cache_values,
    _ensure_dynamic_cache,
    # Cache building
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    # Scoring
    score_answer_with_cache,
)
```

### Library Organization Guidelines

**Before defining functions in a notebook, check if they already exist in lib/:**

| Notebook Need | Library Location |
|---------------|------------------|
| **Access cache keys/values** | `lib/kv_cache._get_cache_keys()`, `_get_cache_values()` |
| **Set cache keys/values** | `lib/kv_cache._set_cache_keys()`, `_set_cache_values()` |
| **Deep copy a cache** | `lib/kv_cache.deepcopy_cache()` |
| **Convert to DynamicCache** | `lib/kv_cache._ensure_dynamic_cache()` |
| Build cache with custom mask | `lib/kv_cache.build_cache_with_mask()` |
| Score with query-time mask | `lib/kv_cache.score_answer_with_cache_flexible()` |
| Truncate cache with BOS | `lib/kv_cache.extract_and_truncate_cache_with_bos()` |
| RoPE position correction | `lib/kv_cache.correct_rope_positions_with_bos()` |
| Block-diagonal prefix mask | `lib/block_attention.create_block_diagonal_prefix_mask()` |
| Query-time masking (first/last/none) | `lib/block_attention.create_query_time_mask_flexible()` |
| Get prefix token boundaries | `lib/block_attention.get_prefix_boundaries_from_text()` |
| Compute MRR, Hit@k | `lib/analysis.compute_ranking_metrics()` |
| Compute token Jaccard similarity | `lib/analysis.compute_token_overlap()` |

**When to add to lib vs keep in notebook:**
- **Add to lib**: Reusable across experiments, general-purpose, tested patterns
- **Keep in notebook**: Experiment-specific combinations, one-off analysis, prototypes

**Always use `model.dtype` for attention masks** to avoid SDPA dtype mismatches.
The lib functions handle this automatically when you pass `dtype=model.dtype`.

## Notebook Documentation Guidelines

**All new experiment code must include clear printouts that illustrate experimental conditions.**
The reader should understand what each condition does without reading the code.

### Required: Condition Explanation Cell

After defining experimental conditions, include a cell that prints concrete examples:

```python
# Cell: Explain Experimental Conditions
print("="*70)
print("EXPERIMENTAL CONDITIONS EXPLAINED")
print("="*70)

# Show concrete example for each condition
example_query = "What is the capital of France?"
example_passage = "Paris is the capital and largest city of France..."

for cond_name, cond_config in CONDITIONS.items():
    print(f"\n### {cond_name} ###")
    print(f"Description: {cond_config['description']}")

    # Show what the prefix looks like
    if cond_config['prefix_fn']:
        prefix = cond_config['prefix_fn'](example_query, example_passage)
        print(f"Prefix (first 100 chars): {prefix[:100]}...")

    # Show attention pattern
    if cond_config.get('use_block_mask'):
        print("Cache-build: Block-diagonal (prefix reps can't see each other)")
    if cond_config.get('query_time_mask'):
        print(f"Query-time: {cond_config['query_time_mask']}")

    # Explain the key difference
    print(f"Key insight: {get_condition_insight(cond_name)}")
```

### Example: Query-Time Masking Conditions (Exp 18)

Good documentation explains *what makes conditions different*:

```
### oracle_5x_qmask_first ###
Cache: [BOS][Q Q Q Q Q][passage]  (5 copies of query)
Query attends to: [BOS][Q · · · ·][passage][query]
                       ↑ only first copy visible
Key insight: Tests if ONE prefix copy is better than ALL copies

### oracle_5x_qmask_last ###
Cache: [BOS][Q Q Q Q Q][passage]
Query attends to: [BOS][· · · · Q][passage][query]
                               ↑ only last copy visible
Key insight: Compared to first_only, tests if position matters

### oracle_5x_qmask_none ###
Cache: [BOS][Q Q Q Q Q][passage]
Query attends to: [BOS][· · · · ·][passage][query]
                       ↑ no prefix visible
Key insight: Tests if prefix visibility hurts (pure interference test)
```

### Why This Matters

1. **Reproducibility**: Future readers understand the experiment without reverse-engineering code
2. **Debugging**: Easy to verify conditions are implemented correctly
3. **Communication**: Results are interpretable by non-coders
4. **Self-documentation**: You'll thank yourself when revisiting in 6 months

## Core Experimental Framework

**Goal:** Build the most effective KV cache for ad-serving, trading storage for quality.

**Baseline (always):** `bare` — Document KV cache built in isolation

**Two Core Experimental Conditions:**

| Condition | Build | Score | Tests |
|-----------|-------|-------|-------|
| **TRUNCATED** | `[prefix][doc]` → truncate prefix → RoPE correct | Query sees only doc | Pure value contamination |
| **FULL-CONTEXT** | `[prefix][doc]` → keep full cache | Query sees prefix + doc | Value contamination + attention |

**⚠️ CRITICAL:** Experiments 15-17 used FULL-CONTEXT only. The "random beats oracle" finding may reflect attention interference, NOT value contamination. Truncation comparison is needed.

## Current State (as of v2 Exp 01)

### v2 Exp 01: First-Principles Priming Test (CLEAN RESTART)
- **Random prefix helps vs bare** (d=+0.091, p<0.001): Value contamination from ANY prefix is beneficial
- **Oracle prefix does NOT beat bare** (d=+0.023, p=0.26, ns): Semantic prefix adds nothing on average
- **Oracle is WORSE than random** (d=-0.051, p=0.015): Semantic content interferes vs random noise
- **Hardness gating replicates** (r=0.157 oracle, r=0.193 random): Hard samples benefit, easy hurt
- Matched tokenization eliminates BPE mismatch; `\n` gives 0% clean boundaries

### Legacy Findings (from v1, may be affected by BPE/framing bugs)
- Value contamination confirmed as the mechanism (Exp 10)
- **Semantic signal exists but is TINY** (Exp 12): r=0.034 (~0.1% variance explained)
- **Truncated oracle > truncated random** (Exp 18): 62% win rate, p<1e-6 — **NOT replicated in v2**
- **BUT: All priming hurts average performance on MS MARCO** (Exp 18): even oracle d=-0.11
- **Hardness gating works** (Exp 13B): r=0.302 between difficulty and benefit — **replicated in v2**
- **LLM surrogates work** (Exp 14D): Intent-based generation (d=0.274) — not yet retested

### ⚠️ CRITICAL: Priming Benefits are MS MARCO-SPECIFIC (Exp 19)

**Cross-dataset survey of 6 datasets revealed priming helps ONLY on MS MARCO:**

| Dataset | Trunc d | Win% | Verdict |
|---------|---------|------|---------|
| MS MARCO (hard) | **+0.19** | 59% | **Only dataset that benefits** |
| SQuAD v2 | +0.003 | 52% | Neutral (ceiling effect) |
| HotpotQA | -0.35 | 34% | HURTS (multi-hop disrupted) |
| PubMedQA | -0.73 | 16% | HURTS (domain mismatch) |
| CNN/DailyMail | **-1.31** | 8% | **CATASTROPHIC** (summarization) |
| NarrativeQA | -0.35 | 45% | HURTS (long-doc interference) |

**Key insight:** The positive results from Exps 05-14 do NOT generalize. Priming should be **OFF by default**.

### ⚠️ CRITICAL: Priming is Mistral-Specific, NOT Universal (Exp 16)

**Cross-model replication on Gemma 3 4B shows priming does NOT transfer:**

| Condition | Mistral d | Gemma d | Verdict |
|-----------|----------|---------|---------|
| static_fact_trunc | **+0.472** | -0.031 (ns) | Does NOT replicate |
| random_trunc | +0.091 | **-0.109** (***) | **Hurts on Gemma** |
| oracle_trunc | +0.023 (ns) | -0.020 (ns) | Neutral on both |
| values_only | +0.275 | +0.056 (**) | Weak positive |

**Key insight:** Value contamination signal exists on Gemma (values_only d=+0.056) but primed
keys carry negative interference (d flips from +0.056 to -0.031 when keys included). Cause is
content-based key interference, NOT bfloat16 precision (Exp 19 disproved). Layer-selective values
(layers 0-16 only) amplifies the effect to d=+0.211 on Gemma.

### ⚠️ Priming Improves Average NLL but NOT Ranking (Exp 22)

**Ranking evaluation: does values_early_layers help distinguish relevant from irrelevant passages?**

| Method | AUC | MRR@10 | Cohen's d |
|--------|-----|--------|-----------|
| Raw bare NLL | 0.828 | 0.860 | +1.201 |
| Raw primed NLL | 0.829 | 0.853 | +1.228 |
| **PMI bare** | **0.841** | **0.860** | **+1.647** |
| PMI primed | 0.832 | 0.853 | +1.588 |

Primed vs Bare MRR (PMI): 6 wins / 185 ties / 9 losses.

**Key insights:**
- **Raw NLL is already a strong ranker** (AUC=0.83, MRR=0.86) without any priming
- **PMI scoring** (NLL - baseline) boosts AUC to 0.841 for free (one BOS-only pass per query)
- **Priming provides zero ranking benefit** — it lowers NLL equally for relevant and irrelevant
  passages. The d=+0.211 from Exp 19 is a content-agnostic regularization, not a relevance signal
- **Static-fact priming is query-independent** — same prefix for all documents cannot create
  query-specific relevance discrimination

### ⚠️ Even Query-Aware Priming Does NOT Improve Ranking (Exp 23)

**Exhaustive test: 13 cache conditions × 3 scoring targets = 39 scores per passage (200 queries).**

Best results (PMI AUC, answer target):

| Condition | AUC | vs bare |
|-----------|-----|---------|
| oracle_interp (α=0.25 blend) | 0.842 | +0.001 |
| bare | 0.841 | — |
| oracle_vel_low (L0-8) | 0.841 | +0.000 |
| qvi_010 (query inject α=0.10) | 0.836 | -0.005 |
| qvi_050 (query inject α=0.50) | 0.767 | -0.074 |

**Key insights:**
- **Nothing beats bare PMI** — even oracle (actual query) priming yields AUC 0.842 vs 0.841 (+0.001, negligible)
- **Alternative targets fail** — `qdoc` (AUC~0.57) and `relevance` template (AUC~0.45-0.50) are poor rankers
- **QVI (query value injection) hurts** — blending query values into doc cache corrupts rather than enhances
- **Intent routing hurts** — min-across-5-intents (AUC 0.825) and per-query routing (0.828) are worse than bare
- **Two-stage pipeline** (bare PMI → oracle re-rank top-3) gives small MRR gain (+0.006) but requires per-passage oracle FP
- **Conclusion**: value contamination fundamentally cannot create ranking signal. NLL ranking comes from
  token overlap between passage and answer, not from cache modifications

### ⚠️ Length is the Primary Constraint (Exp 20)

**Controlled padding experiment: pad MS MARCO passages to long-doc lengths, measure when benefit disappears.**

| Target Length | Cohen's d | Win% | p | sig |
|---------------|-----------|------|---|-----|
| original (~130 tok) | **+0.303** | 67% | 7.7e-07 | *** |
| 256 tok | +0.114 | 65% | 0.059 | ns |
| 512 tok | +0.034 | 58% | 0.570 | ns |
| 1024 tok | -0.043 | 55% | 0.472 | ns |
| 2048 tok | -0.014 | 58% | 0.813 | ns |

**Key insight:** The failure on long NQ docs is a **length effect**, not a dataset effect. Same MS MARCO
content, same prefix, same question style — the benefit vanishes by ~256 tokens. Value contamination
from a ~10-token prefix gets diluted as the cache grows. No known mitigation (Exp 18 periodic beacons
also failed).

## When Priming HELPS (Very Narrow)

The "Goldilocks zone" is much narrower than originally thought:
- **Mistral-7B only** — does NOT replicate on Gemma 3 4B (Exp 16); layer-selective values help on Gemma (Exp 19) but full priming still fails
- **Very short passages only** (<200 tokens / ~100 words, like MS MARCO) — Exp 20 shows benefit
  vanishes by 256 tokens
- **Hard samples** (bare NLL > 1.5) within MS MARCO-like distributions
- **Generative/abstractive tasks** — NOT extractive, NOT summarization
- **Positive hardness correlation** — some datasets show INVERTED correlation (harder HURTS more)

## When Priming HURTS (Most Cases)

**DO NOT prime for:**
- **Ranking / document selection** — zero benefit; bare NLL is already a strong ranker (Exp 22)
- **Summarization** — catastrophic harm (d=-1.3)
- **Multi-hop reasoning** — disrupts attention routing
- **Long passages** (>200 tokens) — value contamination diluted below noise floor (Exp 20)
- **Scientific/specialized domains** — vocabulary mismatch
- **Extractive QA** — no benefit, ceiling effect

## Recommended Deployment Strategy (UPDATED after Exp 20)

**⚠️ Priming should be OFF BY DEFAULT. Only enable for very short MS MARCO-like content.**

```python
def should_prime(passage, tokenizer, task_type, domain):
    # NEVER prime these task types
    if task_type in ["summarization", "multi_hop", "extractive_qa"]:
        return False

    # NEVER prime long passages (Exp 20: benefit gone by ~200 tokens)
    n_tokens = len(tokenizer.encode(passage, add_special_tokens=False))
    if n_tokens > 200:
        return False

    # NEVER prime specialized domains without validation
    if domain in ["scientific", "legal", "technical"]:
        return False

    # Only prime hard MS MARCO-like content
    bare_nll = estimate_difficulty(passage)
    return bare_nll > 1.5


if should_prime(passage, tokenizer, task_type, domain):
    # Truncation + oracle prefix (semantic signal helps)
    prefix = best_query(click_history, repeat=5) or llm_intent_query(passage, repeat=5)
    cache = build_truncated_cache(passage, prefix)  # truncate + RoPE correct
else:
    cache = build_bare_cache(passage)  # don't prime
```

**Bottom line:** The approach works for a narrow slice of use cases (very short factoid QA passages under ~200 tokens). For most real-world applications, priming does more harm than good. Even where priming helps average NLL, it does NOT improve document ranking (Exp 22) — bare PMI scoring (NLL minus BOS-only baseline) is a better, cheaper ranker.
