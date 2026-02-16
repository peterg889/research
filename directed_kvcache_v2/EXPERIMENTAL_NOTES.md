# Experimental Notes — Directed KV Cache v2

## Fresh Start

This is a clean restart of the directed KV cache research. The previous 23 experiments
(in `directed_kvcache/`) were riddled with compounding bugs:

### Known Bugs in Previous Experiments
1. **Wrong RoPE dimension pairing**: Used interleaved instead of half-split convention
2. **Missing BOS tokens**: Truncated caches didn't preserve BOS at position 0
3. **BPE boundary mismatches**: Tokenizing prefix+document together vs separately produces different tokens
4. **Cache mutation**: `score_answer_with_cache()` mutates cache via `use_cache=True` — scoring multiple conditions with the same cache gave wrong results
5. **"Document:\n" framing artifact**: Adding "Document:\n" before passages increases NLL (d=-0.45), confounding baseline comparisons

### What we kept
- Battle-tested library code (`lib/`) with 258 passing tests
- All known-good utility functions with version-safe DynamicCache access

### What we're rebuilding
- Every experiment, from first principles, with explicit safeguards against each known bug

---

## Experiment 01: First-Principles Surrogate Priming Test

**Date started**: 2026-02-06
**Notebook**: `01_first_principles_priming.ipynb`
**Results**: `results/exp01/`

### Question
Does surrogate priming actually have an effect on NLL scoring? If so, is the effect
structural (any prefix changes values) or semantic (relevant prefix content matters)?

### Design
Three conditions per sample (N=2500, MS MARCO v1.1 validation, 2303 valid after
excluding 197 single-token answers):

| Condition | How cache is built | What it tests |
|-----------|-------------------|---------------|
| **Bare** | `[BOS] + doc_ids` | Baseline — no prefix |
| **Random prefix** | `[BOS][random_tokens][doc_ids]` → truncate + RoPE correct | Structural effect of value contamination |
| **Oracle prefix** | `[BOS][oracle_query][doc_ids]` → truncate + RoPE correct | Semantic effect of relevant prefix |

All three conditions use **identical document token IDs** extracted from the oracle
concatenated tokenization. This eliminates BPE boundary mismatch entirely.

### Key Comparisons
- **Bare vs Random**: Does truncation/RoPE-correction itself change NLL?
- **Bare vs Oracle**: Does the query as prefix help?
- **Random vs Oracle**: Is there semantic signal beyond structural noise?

### Safeguards Against Known Bugs
1. No template framing: `surrogate_prefix_template="{surrogate}\n"`, `document_template="{document}"`
2. **Matched tokenization**: `\n` alone gives 0% clean BPE boundaries; fixed by extracting
   `doc_ids` from concatenated tokenization and reusing across all 3 conditions
3. `deepcopy_cache()` before every `score_answer_with_cache()` call
4. Random tokens from vocabulary with decode→re-encode verification
5. `np.random.seed(SEED)` immediately before `load_evaluation_samples()`
6. Checkpoint every 50 samples with full sample list for resume correctness

### Results

**Primary comparisons** (N=2303, positive delta = lower NLL = better):

| Comparison | Mean Δ | Cohen's d | Win% | p-value | Sig |
|---|---|---|---|---|---|
| Bare vs Random | +0.029 | +0.091 | 59.5% | 1.3e-05 | *** |
| Bare vs Oracle | +0.009 | +0.023 | 50.0% | 0.263 | ns |
| Random vs Oracle | -0.020 | -0.051 | 44.9% | 0.015 | * |

**Key finding: Random prefix helps MORE than oracle prefix.**

1. **Any prefix helps** (d=+0.091, p<0.001): Even random nonsense tokens lower NLL.
   The structural act of value contamination itself — putting *something* before the
   document during cache building — is beneficial. Small but highly significant.

2. **Oracle doesn't beat bare** (d=+0.023, ns): The actual query as prefix provides
   no significant improvement. 50.0% win rate — pure coin flip.

3. **Oracle is worse than random** (d=-0.051, p=0.015): Semantically relevant content
   *interferes* compared to random noise. Random wins 55.1% of the time.

**Hardness interaction** (both correlations highly significant, p<1e-13):

| Quartile | Bare NLL | Oracle Δ | Oracle d | Oracle Win% | Random Δ |
|---|---|---|---|---|---|
| Q1 (easy) | 0.086 | -0.033 | -0.195 | 37.8% | +0.001 |
| Q2 | 0.417 | -0.021 | -0.127 | 45.7% | +0.013 |
| Q3 | 0.943 | +0.011 | +0.044 | 56.7% | +0.030 |
| Q4 (hard) | 3.136 | +0.078 | +0.121 | 59.7% | +0.069 |

Oracle priming *hurts* easy samples (Q1: d=-0.195) but helps hard ones (Q4: d=+0.121).
Random priming helps more uniformly. Hardness-benefit correlation: r=0.157 (oracle),
r=0.193 (random).

### Interpretation

The results suggest value contamination creates a small, content-agnostic regularization
effect that slightly helps scoring. But when the prefix is semantically relevant (oracle),
it creates specific attention patterns that **interfere** with easy passages while adding
marginal signal for hard ones. On average, the interference cancels out the benefit.

This is consistent with the old experiments' finding that "all priming hurts average
performance" — but with cleaner methodology we can now decompose the effect:
- **Structural component** (random vs bare): small positive (d=+0.091)
- **Semantic component** (oracle vs random): small negative (d=-0.051)
- **Net effect** (oracle vs bare): near zero (d=+0.023, ns)

### Lessons Learned

1. **`\n` does NOT create clean BPE boundaries** — 0% match rate with SentencePiece.
   Must extract doc token IDs from concatenated tokenization and reuse them.
   This was the #1 bug risk and caught early by the diagnostic cell.

2. **Single-token answers produce NLL=0.0** — `score_answer_with_cache` returns 0 when
   `answer_len=1` (zero transitions to score). 197/2500 samples (7.9%) affected.
   Must filter these before analysis.

3. **The semantic signal is negative on average** — oracle prefix *hurts* relative to
   random, not helps. Previous experiments may have been seeing structural (not semantic)
   effects through the BPE mismatch lens.

4. **Hardness gating is real** — the one finding from old experiments that replicates
   cleanly. Oracle priming only helps for hard samples (Q4: d=+0.121).

---

## Experiment 02: Suffix vs Truncated Priming

**Date started**: 2026-02-06
**Notebook**: `02_suffix_vs_truncated.ipynb`
**Results**: `results/exp02/`

### Question
Does semantic content in a suffix (appended AFTER the passage) improve NLL scoring?
This is the cleanest possible test of semantic signal: causal masking guarantees passage
KV entries are byte-identical to bare, so any benefit must come from query → suffix attention.

### Motivation (from Exp 01)
Exp 01 found random prefix helps MORE than oracle (d=+0.091 vs d=+0.023 ns). Oracle
actually *interferes* vs random (d=-0.051, p=0.015). The truncation mechanism conflates:
- **Structural value contamination** (beneficial): prefix alters passage value vectors
- **Semantic attention patterns** (harmful on average): oracle content creates interference

Suffix priming isolates the semantic signal: passage KV entries are unchanged, so any
effect must come from query tokens attending to suffix KV entries that encode passage-aware context.

### Design
Five conditions per sample (N=2500, same dataset as Exp 01):

| # | Condition | How cache is built | What it tests |
|---|-----------|-------------------|---------------|
| 1 | **Bare** | `[BOS] + doc_ids` | Baseline (matched tokenization) |
| 2 | **Oracle-truncated** | `[BOS][query\n][doc_ids]` → truncate + RoPE | Value contamination (Exp 01 replication) |
| 3 | **Random-truncated** | `[BOS][random\n][doc_ids]` → truncate + RoPE | Structural control (Exp 01 replication) |
| 4 | **Oracle-suffix** | `build_suffix_kv_cache(passage, query, sep)` | Clean semantic signal (NEW) |
| 5 | **Random-suffix** | `build_suffix_kv_cache(passage, random, sep)` | Structural control for suffix (NEW) |

### Six Comparisons (Bonferroni: alpha = 0.05/6 = 0.0083)

**Primary:**
- P1: Oracle-suffix vs Random-suffix — semantic signal in suffix?
- P2: Oracle-suffix vs Bare — does suffix priming help at all?
- P3: Oracle-truncated vs Bare — Exp 01 replication (expect d~+0.023, ns)

**Secondary:**
- S1: Random-suffix vs Bare — any suffix helps?
- S2: Oracle-suffix vs Oracle-truncated — which mechanism better?
- S3: Random-truncated vs Bare — Exp 01 random replication (expect d~+0.091)

### Key Design Details
1. Same matched tokenization for truncated conditions as Exp 01
2. Same random text per sample for both random-truncated and random-suffix
3. Suffix separator: `"\n\nRelated question: "` (identical for oracle/random suffix)
4. Bare fairness diagnostic (Cell 7) confirms matched bare and independent bare agree

### Predictions
- **If Oracle-suffix > Random-suffix (P1 sig)**: Unambiguous semantic signal
- **If Oracle-suffix = Random-suffix (P1 ns)**: No semantic signal even in cleanest test
- **If Random-suffix > Bare (S1 sig)**: Structural attention benefit from suffix
- **If P3 ≈ Exp 01**: Truncated results replicate; experiment is internally consistent

### Results

*(To be filled after experiment completes)*

---

## Experiment 06: LLM Surrogate Deep-Dive — Mechanism Decomposition

**Date started**: 2026-02-08
**Notebook**: `06_surrogate_deep_dive.ipynb`
**Results**: `results/exp06/`

### Question

Why do LLM surrogates work? Is it token overlap, coherence, format, passage specificity,
or something else? Can we decompose the mechanism into independent components?

### Design

15 conditions × 2000 samples (full MS MARCO distribution, N=1834 valid after excluding
166 zero-NLL samples). Bonferroni alpha = 0.005 (10 comparisons).

| # | Condition | Type | d vs Bare | Win% | p-value |
|---|-----------|------|-----------|------|---------|
| 1 | Bare | Baseline | — | — | — |
| 2 | Random-truncated | Control | +0.125 | 62.3% | 9.1e-8 |
| 3 | Separator-only | Control | +0.231 | 66.6% | 1.4e-22 |
| 4 | Oracle-truncated | Oracle | +0.034 | 50.8% | 0.14 ns |
| 5 | Oracle-as-keywords | Oracle | +0.098 | 60.3% | 2.7e-5 |
| 6 | Anti-keywords | Overlap | +0.142 | 61.5% | 1.4e-9 |
| 7 | TF-IDF-keywords | Overlap | +0.101 | 60.5% | 1.7e-5 |
| 8 | Passage-echo | Overlap | +0.007 | 55.1% | 0.75 ns |
| 9 | Shuffled-LLM | Overlap | +0.115 | 60.0% | 9.5e-7 |
| 10 | LLM-keyword | LLM | +0.234 | 67.6% | 5.2e-23 |
| 11 | LLM-question | LLM | +0.200 | 61.2% | 2.2e-17 |
| 12 | LLM-symptom | LLM | +0.252 | 66.7% | 2.0e-26 |
| 13 | LLM-summary | LLM | -0.023 | 53.4% | 0.32 ns |
| 14 | LLM-keyword+sep | Stacking | +0.297 | 69.0% | 1.4e-35 |
| 15 | LLM-messy | LLM | +0.143 | 61.4% | 1.0e-9 |

### 10 Primary Comparisons

| # | Comparison | d | p | Sig? | Finding |
|---|-----------|---|---|------|---------|
| M1 | Shuffled vs LLM-kw | +0.150 | 1.8e-10 | *** | Coherence matters: ordered > shuffled |
| M2 | Oracle-kw vs Oracle | +0.092 | 8.5e-5 | ** | Question format hurts oracle |
| M3 | LLM-kw vs LLM-question | -0.018 | 0.44 | ns | No keyword vs question difference for LLM |
| M4 | TF-IDF vs Anti-kw | -0.026 | 0.26 | ns | Passage specificity doesn't matter |
| M5 | Echo vs LLM-kw | -0.173 | 2.0e-13 | *** | LLM >> passage echo (max overlap ≠ ceiling) |
| M6 | TF-IDF vs LLM-kw | -0.137 | 4.7e-9 | *** | LLM adds value beyond TF-IDF |
| M7 | LLM-kw+sep vs best single | -0.119 | 3.5e-7 | *** | Stacking works: prefix + suffix > either |
| R1 | LLM-kw vs Random | +0.099 | 2.5e-5 | ** | LLM > random (replicates Exp 05) |
| R2 | Oracle vs Random | -0.077 | 1.0e-3 | ** | Random > oracle (replicates Exp 01 null) |
| R3 | LLM-kw vs Bare | +0.234 | 5.2e-23 | *** | Overall LLM benefit confirmed |

### Token Overlap Mechanism Analysis

Token overlap is NOT the mechanism:
- **Universal r(overlap, delta) = -0.024** (p=3.8e-4): essentially zero
- **Cross-condition r = -0.260** (p=0.41): not significant
- **Regression R² = 0.042**: overlap + hardness explain only 4.2% of variance
- Regression betas: intercept=+0.044, overlap=-0.010, hardness=+0.079, interaction=+0.002

Key evidence against overlap:
- Anti-keywords (wrong doc, d=+0.142) ≈ TF-IDF (right doc, d=+0.101) — M4 ns
- Passage-echo (highest overlap, d=+0.007) ≈ bare — overlap CEILING is near zero
- LLM-keyword (d=+0.234) >> TF-IDF (d=+0.101) despite similar overlap

### Key Mechanism Findings

1. **Separator-only is surprisingly powerful** (d=+0.231): Just appending "\n\nRelated question: "
   after the passage, with NO content, nearly matches LLM-keyword (d=+0.234). This is a
   suffix attention mechanism, not value contamination.

2. **Stacking works** (d=+0.297): LLM-keyword prefix + separator suffix exceeds either alone.
   The two mechanisms are additive.

3. **Coherence matters** (M1, d=+0.150, p<1e-10): Ordered LLM tokens > shuffled LLM tokens.
   This rules out "just token identity" — the model needs coherent prefix text.

4. **Question format hurts oracle but not LLM** (M2 sig, M3 ns): Oracle queries lose power
   from question syntax; LLM-generated queries are robust to format.

5. **LLM adds value beyond TF-IDF** (M6, d=0.137, p<1e-9): LLM surrogates capture something
   cheap extraction cannot — likely document coherence and relevance framing.

6. **LLM-summary hurts** (d=-0.023, ns): Long, verbose suffixes don't help. Short, focused
   surrogates are best.

### Hardness Quintile Breakdown (d vs bare)

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) | Overall |
|-----------|-----------|-----|-----|-----|-----------|---------|
| random_trunc | +0.075 | +0.244 | +0.221 | +0.246 | +0.117 | +0.125 |
| oracle_trunc | -0.259 | -0.151 | -0.067 | +0.160 | +0.124 | +0.034 |
| tfidf_keywords | -0.130 | +0.040 | +0.169 | +0.277 | +0.119 | +0.101 |
| llm_keyword | -0.082 | +0.222 | +0.349 | +0.355 | +0.340 | +0.234 |
| llm_symptom | -0.110 | +0.243 | +0.420 | +0.427 | +0.379 | +0.252 |
| llm_keyword_sep | -0.226 | -0.036 | +0.258 | +0.475 | +0.630 | +0.297 |

The hardness gradient is massive for LLM-keyword+sep: d ranges from -0.226 (Q1 easiest) to
+0.630 (Q5 hardest), with hardness-benefit r=0.574.

### Interpretation

The priming mechanism has two independent channels:
1. **Value contamination** (truncated prefix): Any prefix alters document value vectors.
   LLM surrogates > TF-IDF > random > passage-echo. Coherence matters.
2. **Suffix attention** (separator-only): Adding structural framing after the passage creates
   new KV entries that query tokens can attend to. Content-free separator alone matches LLM.

These channels are additive: combining them (LLM-keyword+sep) yields the best result (d=+0.297).
Token overlap is NOT the mechanism (r=-0.024). The LLM's value comes from generating
coherent, concise text that activates useful attention patterns, not from overlapping vocabulary.

### Lessons Learned

1. **Separator-only baseline is essential** — without it, you'd attribute all improvement to
   the LLM content, when much of it comes from structural framing.
2. **Token overlap is a red herring** — r=-0.024 across all conditions. Don't use overlap
   as a proxy for surrogate quality.
3. **LLM-summary hurts** — verbose surrogates add noise. Keep surrogates short (3-10 words).
4. **Passage-echo fails** despite maximal overlap — first-sentence echo (d=+0.007) performs
   like bare. The passage content is already in the cache; repeating it adds nothing.
5. **Anti-keywords work as well as TF-IDF** — specificity to the CORRECT document doesn't
   matter for overlap-based approaches. Only LLM surrogates capture true relevance.

---

## Exp 07: Static Surrogates, Dual-Mode Priming, Intent Routing

**Notebook:** `07_static_surrogates_and_routing.ipynb`
**Date:** 2026-02-09
**N:** 2000 evaluated, 1837 valid (163 excluded for zero NLLs), SEED=42
**Status:** Complete. Plots in `results/exp07/analysis_plots.png`.

### Core Question

Can cheap static surrogates (fixed phrases with no LLM generation) match or beat LLM-generated
surrogates? Does the optimal strategy depend on query intent, passage length, or difficulty?

### Design

**21 conditions** testing static vs LLM surrogates in both truncated-prefix and suffix modes:

| Category | Conditions | Description |
|----------|-----------|-------------|
| Baselines (3) | bare, random_trunc, separator_only | Controls |
| Static trunc (5) | static_{def,proc,quant,fact,prob}_trunc | Fixed phrases as truncated prefixes |
| Static suffix (5) | static_{def,proc,quant,fact,prob}_suffix | Same phrases appended as suffixes |
| LLM suffix (4) | llm_{keyword,symptom,question,messy}_suffix | LLM-generated surrogates as suffixes |
| LLM special (3) | llm_keyword_trunc, llm_keyword_sep, llm_keyword_full_ctx | LLM keyword in different modes |
| Novel (1) | novel_generic_trunc | "What is this page about?" as prefix |

Static phrases from `lib/surrogate.py::STATIC_SURROGATE_QUERIES`:
- definitional: "What is this and what does it mean?"
- procedural: "How do I do this step by step?"
- quantitative: "How much does this cost or how long does it take?"
- factual: "What are the key facts I need to know?"
- problem: "What problem does this solve?"

### Results — All Conditions vs Bare (N=1837)

| Condition | d vs bare | Win% | p-value | Sig |
|-----------|----------|------|---------|-----|
| static_fact_trunc | **+0.472** | 80.5% | 3.2e-82 | *** |
| static_quant_trunc | +0.410 | 74.5% | 6.9e-64 | *** |
| novel_generic_trunc | +0.342 | 75.1% | 3.9e-46 | *** |
| static_quant_suffix | +0.335 | 70.7% | 2.6e-44 | *** |
| llm_keyword_sep | +0.297 | 68.9% | 1.4e-35 | *** |
| static_prob_suffix | +0.245 | 66.9% | 3.7e-25 | *** |
| static_def_trunc | +0.237 | 66.0% | 1.1e-23 | *** |
| llm_keyword_trunc | +0.234 | 67.6% | 5.3e-23 | *** |
| separator_only | +0.231 | 66.5% | 1.4e-22 | *** |
| llm_symptom_suffix | +0.220 | 64.9% | 1.4e-20 | *** |
| static_prob_trunc | +0.215 | 62.8% | 9.4e-20 | *** |
| static_proc_trunc | +0.165 | 63.9% | 2.1e-12 | *** |
| static_fact_suffix | +0.141 | 63.9% | 1.9e-09 | *** |
| random_trunc | +0.125 | 62.3% | 9.1e-08 | *** |
| llm_keyword_suffix | +0.116 | 61.6% | 8.0e-07 | *** |
| llm_keyword_full_ctx | +0.110 | 58.5% | 2.6e-06 | *** |
| static_def_suffix | +0.057 | 53.3% | 1.5e-02 | * |
| llm_question_suffix | -0.026 | 51.1% | 2.7e-01 | ns |
| llm_messy_suffix | -0.033 | 54.2% | 1.5e-01 | ns |
| static_proc_suffix | -0.071 | 52.3% | 2.4e-03 | ** |

Bare mean NLL: 1.114

### Key Comparisons (Bonferroni alpha = 0.005)

| # | Comparison | d | p | Sig | Answer |
|---|-----------|---|---|-----|--------|
| C1 | Best-static-trunc (fact) vs Bare | +0.472 | 3.2e-82 | *** | YES, static prefixes help enormously |
| C2 | Best-static-suffix (quant) vs Bare | +0.335 | 2.6e-44 | *** | YES, static suffixes help |
| C3 | Best-static-trunc vs Best-static-suffix | +0.129 | 3.7e-08 | *** | Truncated prefix > suffix |
| C4 | Best-static (fact_trunc) vs LLM-kw-trunc | -0.260 | 5.9e-28 | *** | **Statics BEAT LLM by a lot** |
| C5 | Oracle-static-K5 vs Best-static | +0.489 | 1.8e-87 | *** | Routing helps massively |
| C6 | Oracle-static-K5 vs LLM-kw-suffix | -0.466 | 1.5e-80 | *** | Routed statics >> single LLM |
| C7 | LLM-kw-suffix vs LLM-kw-trunc | -0.075 | 1.4e-03 | ** | Truncated > suffix for LLM too |
| C8 | LLM-kw-full-ctx vs LLM-kw-trunc | -0.165 | 1.9e-12 | *** | **Full-context is WORSE** |
| C9 | LLM-kw-sep vs LLM-kw-suffix | +0.199 | 3.1e-17 | *** | Stacking helps (replicates Exp 06) |
| C10 | Embed-routed-LLM vs Oracle-LLM | +0.531 | 4.1e-101 | *** | Embedding routing fails (22.5% accuracy) |

### Oracle Routing

Best-of-K oracle routing (select min NLL per sample across top-K conditions):

| K | Static d | LLM suffix d |
|---|---------|-------------|
| 1 | +0.472 (fact_trunc) | +0.220 (symptom) |
| 2 | +0.540 | +0.447 |
| 3 | +0.623 | +0.486 |
| 4 | +0.629 | +0.519 |
| 5 | +0.632 | — |

Oracle routing across top-3 statics achieves d=+0.623. Diminishing returns after K=3.

### Practical Routing Results (NEW)

| Routing Strategy | d vs bare | Notes |
|-----------------|----------|-------|
| Oracle-static-K5 | +0.655 | Per-sample best of 10 static conditions |
| Oracle-LLM-K4 | +0.519 | Per-sample best of 4 LLM suffix conditions |
| Embedding-routed-LLM-K4 | **+0.037** | Cosine similarity routing — essentially useless |
| Intent-matched-static | **+0.187** | Rule-based intent → static phrase — worse than best single |
| Best single static (fact_trunc) | +0.472 | Zero-cost baseline beats all practical routing |

**Embedding routing is a failure**: 22.5% oracle accuracy (random = 25%). The cosine similarity
between queries and LLM surrogates does not predict which surrogate will minimize NLL. The
router heavily favors "question" format (57%) while oracle favors "symptom" (34%).

**Intent-matched routing is also worse than no routing**: Matching query intent to the
corresponding static phrase (d=+0.187) performs worse than just using static_fact_trunc for
everything (d=+0.472). The "factual" phrase is universally superior.

### Hardness Quintile Breakdown

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) | r(bare,benefit) |
|-----------|----------|----|----|----|----|------|
| static_fact_trunc | +0.187 | +0.813 | +0.934 | +0.886 | +0.700 | +0.610 |
| static_quant_trunc | — | — | — | — | — | — |
| novel_generic_trunc | +0.099 | +0.505 | +0.662 | +0.648 | +0.459 | +0.408 |
| llm_keyword_trunc | -0.082 | +0.217 | +0.351 | +0.356 | +0.339 | +0.284 |
| llm_keyword_sep | -0.226 | -0.030 | +0.249 | +0.474 | +0.630 | +0.575 |
| separator_only | -0.269 | -0.070 | +0.273 | +0.281 | +0.518 | +0.462 |
| random_trunc | +0.074 | +0.234 | +0.231 | +0.245 | +0.117 | +0.128 |

Key finding: **static_fact_trunc is unique** — it helps even Q1 (easiest) samples (+0.187),
while all other conditions hurt Q1. Its hardness correlation (r=+0.610) is the highest of any
condition, peaking at Q3 (d=+0.934). This makes it uniquely suitable for production where
hardness gating may not be available.

### Hardness-Gated Routing (using llm_keyword_sep)

| Threshold | % Primed | d vs bare |
|-----------|----------|----------|
| 0.0 | 100% | +0.297 |
| 0.3 | 70% | +0.331 |
| 0.5 | 56% | +0.323 |
| 0.8 | 42% | +0.313 |
| 1.0 | 35% | +0.299 |
| 1.5 | 23% | +0.250 |

Optimal gate threshold ~0.3 (prime 70% of samples), but the improvement over always-priming
is modest (+0.034). For static_fact_trunc (not shown), gating would help even less since it
already helps Q1 queries.

### Intent Stratification

| Intent | N | static_fact_trunc | llm_kw_trunc | llm_kw_sep | novel_generic |
|--------|---|------------------|-------------|------------|--------------|
| other | 845 | +0.448 | +0.225 | +0.345 | +0.318 |
| definitional | 503 | +0.607 | +0.364 | +0.450 | +0.493 |
| factual | 242 | +0.523 | +0.198 | +0.073 | +0.367 |
| transactional | 138 | +0.460 | +0.112 | +0.174 | +0.303 |
| procedural | 69 | +0.685 | +0.091 | +0.555 | +0.383 |
| medical | 26 | +0.553 | +0.389 | +0.573 | +0.693 |
| comparison | 14 | +1.105 | +0.587 | +0.442 | +0.311 |

**Best static per intent**: static_fact_trunc wins 5/7 categories. Exceptions:
medical (static_quant_suffix, d=+0.693) and transactional (static_quant_trunc, d=+0.549).

### Answer Length Stratification

| Length | N | separator_only | llm_kw_suffix | llm_kw_trunc | llm_kw_sep |
|--------|---|---------------|--------------|-------------|------------|
| Short (<5 tokens) | 203 | +0.011 | -0.038 | +0.097 | +0.103 |
| Medium (5-15) | 647 | +0.242 | +0.106 | +0.297 | +0.352 |
| Long (>15) | 987 | +0.660 | +0.397 | +0.473 | +0.687 |

Longer answers benefit dramatically more from priming. Short answers (<5 tokens) get
essentially no benefit. This makes sense: short factoid answers have low entropy regardless.

### Passage Length

| Length | N | separator_only | llm_kw_suffix | llm_kw_trunc | llm_kw_sep |
|--------|---|---------------|--------------|-------------|------------|
| Short (<80w) | 1065 | +0.265 | +0.140 | +0.302 | +0.341 |
| Medium (80-200w) | 772 | +0.181 | +0.075 | +0.152 | +0.237 |

Short passages benefit more from priming than medium ones. No long passages (>200w) in MS MARCO.

### Interpretation

**The headline finding is that static_fact_trunc ("What are the key facts I need to know?")
is the single best priming strategy we've tested, at d=+0.472 — 2x the effect of
LLM-keyword-trunc (d=+0.234).** This is remarkable because:

1. **Zero runtime cost** — the phrase is fixed, no LLM generation needed per document
2. **Universally effective** — works across all intent categories and even helps easy queries
3. **Dominates LLM surrogates** — C4 shows statics beat LLM by d=+0.260
4. **Strongest hardness correlation** — r=+0.610, peaks at Q3 (d=+0.934)

Other key insights:

1. **Truncated prefix >> suffix >> full-context**: The truncation+RoPE approach consistently
   outperforms suffix mode (C3: d=+0.129, C7: d=-0.075), which in turn beats full-context
   (C8: d=-0.165). Full-context being worst strongly supports value contamination as the
   mechanism — direct prefix attention actually interferes.

2. **Practical routing fails**: Both embedding routing (d=+0.037) and intent-matched routing
   (d=+0.187) perform far worse than the best single static (d=+0.472). The oracle shows
   massive headroom (K=3: d=+0.623), but we lack a practical way to capture it.

3. **novel_generic_trunc** ("What is this page about?") at d=+0.342 is an excellent ultra-cheap
   option — slightly worse than static_fact_trunc but simpler.

4. **LLM question/messy formats fail in suffix mode** (d≈0, ns): Only keyword and symptom
   formats work as suffixes. Format matters a lot for suffix mode.

5. **Stacking replicates** (C9: d=+0.199): Combining truncated prefix with suffix separator
   is additive, replicating Exp 06 findings.

6. **Answer length matters more than passage length**: Long answers (>15 tokens) benefit at
   d=+0.687 (llm_kw_sep) while short answers (<5 tokens) get d=+0.103. Priming helps most
   when the model needs to generate more text.

### Lessons Learned

1. **Don't need LLMs for priming** — static phrases can be 2x more effective than LLM surrogates.
   The key is the PHRASE CONTENT, not whether it was generated specifically for each passage.
2. **"What are the key facts?" is the magic phrase** — it's universally effective, probably
   because it primes the model for factual recall regardless of actual question type.
3. **Truncation is the superior delivery mechanism** — value contamination through forward pass
   is more effective than suffix attention. Full-context hurts because direct prefix attention
   interferes with passage comprehension.
4. **Hardness gating is less critical for static_fact_trunc** — unlike other conditions that
   hurt Q1 queries, static_fact_trunc even helps easy samples (d=+0.187 at Q1). A production
   system could use it universally with less risk.
5. **Practical routing is an unsolved problem** — oracle K=3 shows d=+0.623 but embedding
   cosine similarity and intent matching both fail. Need fundamentally different approaches.
6. **Answer length is the strongest moderator** — priming benefits scale with answer length,
   suggesting the effect accumulates across generated tokens.

---

## Exp 08: Mechanism Isolation + Prefix Amplification

**Notebook:** `08_mechanism_and_amplification.ipynb`
**Executed:** `08_mechanism_and_amplification_executed.ipynb`
**Date:** 2026-02-10
**N:** 1000 evaluated, 930 valid (70 excluded for zero NLLs), SEED=42
**Status:** Complete. Plots in `results/exp08/analysis_plots.png`.

### Core Questions

1. **Key vs Value isolation**: Is the priming effect carried by keys, values, or both?
2. **Prefix amplification**: Does repeating the surrogate prefix K times amplify the effect?

### Design

10 conditions × 1000 samples. LLM keyword surrogates only. Bonferroni alpha = 0.0083 (6 comparisons).

| # | Condition | Construction | Tests |
|---|-----------|-------------|-------|
| 1 | Bare | `[BOS][doc]` | Baseline |
| 2 | LLM-keyword-trunc | Standard truncated + RoPE | Reference |
| 3 | LLM-keyword-suffix | Suffix mode | Reference |
| 4 | LLM-keyword+sep | Trunc + suffix | Best method reference |
| 5 | Primed-values-only | Keys from bare, values from LLM-trunc | Key vs value |
| 6 | Primed-keys-only | Values from bare, keys from LLM-trunc | Key vs value |
| 7 | Prefix-1x | `[BOS][kw\n][doc]` → truncate (same as #2) | Baseline repetition |
| 8 | Prefix-3x | `[BOS][kw\n kw\n kw\n][doc]` → truncate | 3× amplification |
| 9 | Prefix-5x | `[BOS][kw\n ×5][doc]` → truncate | 5× amplification |
| 10 | Separator-only | `[BOS][doc][\n\nRelated question: ]` | Structural control |

### Results — All Conditions vs Bare (N=930)

| Condition | d vs bare | Win% | p-value | Sig |
|-----------|----------|------|---------|-----|
| llm_kw_sep | +0.288 | 67.4% | 8.3e-18 | *** |
| primed_values_only | +0.275 | 67.8% | 1.7e-16 | *** |
| llm_kw_trunc / prefix_1x | +0.254 | 69.2% | 2.5e-14 | *** |
| prefix_3x | +0.245 | 67.4% | 2.0e-13 | *** |
| separator_only | +0.214 | 65.1% | 1.2e-10 | *** |
| prefix_5x | +0.208 | 66.5% | 3.6e-10 | *** |
| llm_kw_suffix | +0.179 | 60.8% | 6.4e-08 | *** |
| primed_keys_only | -0.009 | 46.0% | 7.8e-01 | ns |

### 6 Primary Comparisons

| # | Comparison | d | p | Sig | Answer |
|---|-----------|---|---|-----|--------|
| C1 | Primed-values-only vs Bare | +0.275 | 1.7e-16 | *** | YES — values carry the signal |
| C2 | Primed-keys-only vs Bare | -0.009 | 7.8e-01 | ns | NO — keys contribute nothing |
| C3 | Primed-values-only vs LLM-trunc | -0.045 | 1.7e-01 | ns | Values capture ~100% of the effect |
| C4 | Prefix-3x vs Prefix-1x | -0.069 | 3.6e-02 | * | 3× does NOT amplify (marginal harm) |
| C5 | Prefix-5x vs Prefix-1x | -0.129 | 8.9e-05 | *** | 5× significantly HURTS |
| C6 | Prefix-5x vs Prefix-3x | -0.220 | 3.1e-11 | *** | More repetition = worse |

### Key/Value Decomposition

| Component | d vs Bare | % of Full Effect |
|-----------|----------|-----------------|
| Full LLM-trunc (keys + values) | +0.254 | 100% |
| Values only (bare keys) | +0.275 | 108% |
| Keys only (bare values) | -0.009 | -4% |
| Sum of parts (V + K) | +0.266 | 105% |

**The priming effect is carried ENTIRELY by value vectors.** Swapping in primed values
with bare keys (d=+0.275) actually slightly exceeds the full effect (d=+0.254), suggesting
that bare keys are marginally better than primed keys for the query-time attention routing.
Keys contribute nothing — primed-keys-only is indistinguishable from bare (d=-0.009, p=0.78).

### Prefix Amplification Curve

| Repetitions | d vs Bare | Δd from 1× |
|------------|----------|------------|
| 1× | +0.254 | — |
| 3× | +0.245 | -0.009 |
| 5× | +0.208 | -0.046 |

**Repetition does NOT amplify — it degrades.** The 5× condition is significantly worse
than 1× (p<0.001). This is consistent with a model where truncated prefix value contamination
has already saturated at 1×, and additional repetitions dilute the document representation
by consuming more of the model's representational capacity with redundant prefix content.

### Hardness Quintile Breakdown (d vs bare)

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) |
|-----------|----------|----|----|----|----|
| llm_kw_trunc | -0.030 | +0.192 | +0.280 | +0.530 | +0.342 |
| primed_values_only | -0.026 | +0.172 | +0.328 | +0.623 | +0.376 |
| primed_keys_only | -0.271 | -0.287 | -0.100 | +0.095 | +0.050 |
| prefix_1x | -0.030 | +0.192 | +0.280 | +0.530 | +0.342 |
| prefix_3x | -0.046 | +0.184 | +0.297 | +0.554 | +0.270 |
| prefix_5x | -0.069 | +0.181 | +0.224 | +0.484 | +0.223 |
| llm_kw_sep | -0.349 | +0.008 | +0.152 | +0.524 | +0.595 |
| separator_only | -0.382 | -0.070 | +0.134 | +0.271 | +0.498 |

Primed-values-only peaks at Q4 (d=+0.623) — stronger than any other single condition at
the hardest quintiles. Keys-only actually *hurts* easy samples (Q1: d=-0.271, Q2: d=-0.287),
confirming that the primed keys interfere with attention routing.

### Interpretation

1. **Values are the mechanism, not keys.** The truncated prefix forward pass alters the
   value vectors of document tokens (value contamination). The key vectors, which control
   attention routing via dot-product with queries, are either unaffected or slightly harmed
   by the prefix. This is the cleanest mechanistic evidence yet: you can literally swap
   just the values and get the full effect.

2. **Prefix repetition is counterproductive.** The value contamination signal saturates at
   a single pass. Repeating the prefix 3× or 5× doesn't strengthen the signal — it weakens
   it, likely because the longer prefix during the forward pass means document tokens attend
   more to prefix tokens and less to each other, diluting inter-document attention patterns.

3. **Values-only may be the purest priming mechanism.** The fact that primed-values + bare-keys
   (d=+0.275) slightly exceeds full priming (d=+0.254) suggests an optimal strategy:
   build the primed cache, then REPLACE the keys with bare-cache keys. This preserves the
   value contamination signal while maintaining clean attention routing. Whether this marginal
   gain (+0.021d) justifies the 2× cache-build cost is questionable.

### Lessons Learned

1. **Build hybrid caches for mechanism isolation** — `build_hybrid_cache()` from lib/kv_cache.py
   makes it easy to swap keys and values between caches for clean decomposition experiments.
2. **Don't repeat prefixes** — "more is better" intuition fails here. One pass is sufficient
   and optimal for value contamination.
3. **Primed keys are slightly harmful** — keys-only hurts easy samples substantially (Q1-Q2:
   d≈-0.28). In production, if you only need a partial priming approach, contaminating values
   alone is strictly better than contaminating keys alone.

---

## Exp 09: Values Deep Dive

**Notebook:** `09_values_deep_dive.ipynb`
**Executed:** `09_values_deep_dive_executed.ipynb`
**Date:** 2026-02-10
**N:** 1000 evaluated, 929 valid (71 excluded for zero NLLs), SEED=42
**Status:** Complete. Plots in `results/exp09/analysis_plots.png`.

### Core Question

Exp 08 showed value vectors carry 100% of the priming effect (d=+0.275) while keys contribute
nothing (d=-0.009). This experiment dissects the values-only mechanism across 5 directions:
layer-wise isolation, prefix type variation, value interpolation, cross-document transfer,
and positional isolation.

### Design

17 conditions × 1000 samples. All value-manipulation conditions use keys from bare cache,
values from the specified source (values-only mode). Bonferroni alpha = 0.005 (10 comparisons).

| # | Condition | Category | d vs bare | Win% | p-value | Sig |
|---|-----------|----------|-----------|------|---------|-----|
| 1 | bare | Baseline | — | — | — | — |
| 2 | full_llm_kw | Reference | +0.253 | 69.2% | 3.1e-14 | *** |
| 3 | values_only_llm_kw | Reference | +0.275 | 67.8% | 2.1e-16 | *** |
| 4 | values_layers_0_7 | Layer-wise | +0.233 | 66.1% | 2.3e-12 | *** |
| 5 | values_layers_8_15 | Layer-wise | +0.267 | 67.1% | 1.4e-15 | *** |
| 6 | values_layers_16_23 | Layer-wise | -0.017 | 48.4% | 0.60 | ns |
| 7 | values_layers_24_31 | Layer-wise | -0.068 | 45.3% | 0.040 | ns |
| 8 | values_only_static_fact | Prefix type | **+0.466** | 81.5% | 1.2e-41 | *** |
| 9 | values_only_oracle | Prefix type | +0.309 | 68.5% | 3.6e-20 | *** |
| 10 | values_only_random | Prefix type | +0.310 | 66.7% | 3.0e-20 | *** |
| 11 | values_interp_025 | Interpolation | +0.236 | 68.2% | 1.4e-12 | *** |
| 12 | values_interp_050 | Interpolation | +0.244 | 68.2% | 2.2e-13 | *** |
| 13 | values_interp_075 | Interpolation | +0.262 | 67.6% | 3.9e-15 | *** |
| 14 | values_cross_doc | Cross-doc | **-1.070** | 6.8% | 5.0e-156 | *** |
| 15 | values_first_quarter | Positional | +0.333 | 66.3% | 4.7e-23 | *** |
| 16 | values_last_quarter | Positional | +0.030 | 49.4% | 0.36 | ns |
| 17 | values_middle_half | Positional | +0.044 | 52.6% | 0.18 | ns |

### 10 Primary Comparisons

| # | Comparison | d | p | Sig | Answer |
|---|-----------|---|---|-----|--------|
| C1 | layers_0_7 vs bare | +0.233 | 2.3e-12 | *** | YES — early layers carry signal |
| C2 | layers_8_15 vs bare | +0.267 | 1.4e-15 | *** | YES — early-mid layers strongest |
| C3 | layers_16_23 vs bare | -0.017 | 0.60 | ns | NO — mid-late layers contribute nothing |
| C4 | layers_24_31 vs bare | -0.068 | 0.040 | ns | NO — late layers slightly harmful |
| C5 | static_fact vs llm_kw (values) | +0.252 | 3.9e-14 | *** | YES — static dominates even in values-only |
| C6 | random vs bare (values) | +0.310 | 3.0e-20 | *** | YES — even random-primed values help |
| C7 | interp_050 vs bare | +0.244 | 2.2e-13 | *** | YES — 50% blend helps |
| C8 | cross_doc vs bare | -1.070 | 5.0e-156 | *** | CATASTROPHIC — wrong-doc values destroy |
| C9 | cross_doc vs llm_kw (values) | -1.103 | 1.1e-162 | *** | Same-doc massively better |
| C10 | first_quarter vs last_quarter | +0.323 | 8.9e-22 | *** | YES — beginning positions carry signal |

### Direction 1: Layer-Wise Decomposition

| Layer Group | d vs Bare | % of Full Effect |
|-------------|-----------|-----------------|
| Layers 0-7 (early) | +0.233 | 85% |
| Layers 8-15 (early-mid) | +0.267 | 97% |
| Layers 16-23 (mid-late) | -0.017 | -6% |
| Layers 24-31 (late) | -0.068 | -25% |
| All layers (reference) | +0.275 | 100% |

**The priming signal lives in layers 0-15.** Early and early-mid layers each carry nearly the
full effect independently (85-97%). Late layers contribute nothing and are slightly harmful.
This is consistent with a representation-building mechanism: early layers form the semantic
representation that later layers read. Value contamination alters HOW early layers build
document representations, and those altered representations propagate forward.

### Direction 2: Prefix Type Comparison (All Values-Only)

| Prefix Type | d vs Bare | Description |
|-------------|-----------|-------------|
| Static factual | **+0.466** | "What are the key facts I need to know?" |
| Random tokens | +0.310 | Random vocabulary tokens |
| Oracle (actual query) | +0.309 | The real query as prefix |
| LLM keyword | +0.275 | LLM-generated keywords |

**Static factual dominates again** (d=+0.466), even when isolated to values-only mode. This
replicates Exp 07's finding (d=+0.472 full, d=+0.466 values-only) almost exactly, confirming
the effect is carried entirely by values.

**Critically: static_fact >> llm_kw in values-only mode** (C5: d=+0.252, p<1e-14). The
"magic phrase" effect is NOT about attention routing (keys) — it's encoded in the value vectors.

**Random ≈ Oracle in values-only** (d=+0.310 vs +0.309). The oracle query doesn't improve
values beyond random noise. This contrasts with static_fact, which is far superior. The
oracle's semantic content is irrelevant for value contamination — but static_fact's phrasing
creates uniquely beneficial value patterns.

### Direction 3: Value Interpolation

| Alpha (primed blend) | d vs Bare | Predicted (linear) |
|---------------------|-----------|-------------------|
| 0.00 (bare) | 0.000 | — |
| 0.25 | +0.236 | +0.069 |
| 0.50 | +0.244 | +0.137 |
| 0.75 | +0.262 | +0.206 |
| 1.00 (full primed) | +0.275 | +0.275 |

Linear fit: slope=0.053, intercept=0.221, **R²=0.961**. The relationship between primed
value fraction and effect size is nearly linear but with a massive intercept: even 25%
primed values already achieve 86% of the full effect (d=+0.236 vs +0.275). This suggests
the value contamination signal is highly redundant — a small fraction of primed values
is sufficient to shift the representation.

### Direction 4: Cross-Document Value Transfer

Cross-doc (d=-1.070, catastrophic): Using primed values from the WRONG document's cache
is far worse than bare. Mean NLL jumps from 1.163 (bare) to 3.196 (cross-doc). The
value contamination is **document-specific** — it encodes information about WHICH document
the prefix was processed with, not just the prefix content.

This is the strongest evidence yet that value contamination is NOT just structural noise.
If it were purely structural (any prefix changes values in a generic helpful way), then
wrong-document values should still help. Instead, they're catastrophic.

### Direction 5: Positional Isolation

| Position Range | d vs Bare | d Per 25% Fraction |
|---------------|-----------|-------------------|
| First 25% | **+0.333** | 1.333 |
| Middle 50% | +0.044 (ns) | 0.087 |
| Last 25% | +0.030 (ns) | 0.120 |

**Early token positions carry the signal.** The first quarter of the document (by token
position) carries essentially all the value contamination benefit. Middle and last positions
contribute nothing. Per-fraction efficiency: first quarter is 15× more efficient than
middle half.

This makes physical sense: in causal attention, early tokens influence every subsequent
token's representation. Contaminating early token values propagates through the entire
forward pass. Late token values only affect tokens after them — by which point the
representation is already formed.

### Key Synthesis Across 5 Directions

1. **Where**: Layers 0-15 (early/early-mid), first 25% of token positions
2. **What**: Value vectors only (keys contribute nothing, confirming Exp 08)
3. **How much**: Nearly linear in blend fraction, but saturates quickly (25% blend → 86% effect)
4. **Document specificity**: Catastrophically document-specific (cross-doc d=-1.070)
5. **Content sensitivity**: Static factual >> LLM ≈ oracle ≈ random in values-only mode

### Important Nuance: Semantic Content DOES Matter

While random ≈ oracle ≈ llm_kw in values-only mode (all around d≈+0.3), **static_fact is
dramatically better** (d=+0.466). This can't be explained by structural properties alone:
- static_fact and random have similar token counts
- Both use truncation + RoPE correction
- The only difference is the SEMANTIC CONTENT of the phrase

Furthermore, looking across experiments:
- **Exp 07**: static_fact_trunc (d=+0.472) >> random_trunc (d=+0.125) — 3.8× larger effect
- **Exp 06**: llm_kw (d=+0.234) >> random (d=+0.125) — 1.9× larger
- **Exp 06**: shuffled_llm (d=+0.115) < llm_kw (d=+0.234) — coherence matters (M1 sig)
- **Exp 06**: oracle-as-keywords (d=+0.098) < oracle-raw (d=+0.034) — format matters

**Semantic content matters, but the relationship is complex.** The "magic phrase" effect
of static_fact suggests that specific phrasings can create universally beneficial value
contamination patterns, independent of document content. This is NOT the same as "content
doesn't matter" — it means the MOST EFFECTIVE content is a general-purpose factual priming
phrase, not document-specific information.

### Lessons Learned

1. **Layer targeting could save computation** — only layers 0-15 need primed values.
   Late layers could keep bare values with no loss (and slight gain from removing interference).
2. **Position targeting works** — priming just the first 25% of positions gives d=+0.333,
   better than full priming (d=+0.275). This is surprising and suggests over-priming degrades
   late positions.
3. **Cross-doc is a strong control** — the catastrophic failure of wrong-document values
   (d=-1.070) proves the value contamination is content-specific, not just noise.
4. **Interpolation reveals redundancy** — 25% primed values gives 86% of the effect. This
   has practical implications: you might not need a full primed cache.
5. **Static factual is the magic phrase** — confirmed even in values-only mode (d=+0.466).

---

## Exp 10: Semantic Content Gradient (2026-02-11)

**Notebook**: `10_semantic_content_gradient.ipynb`
**Build script**: `scripts/build_nb10.py`
**Results**: `results/exp10/results.json`
**Runtime**: ~6 hours on NVIDIA L4
**Samples**: 1000 evaluated, 929 valid (71 zero-NLL excluded)

### Research Question

Does semantic content in prefixes follow a monotonic gradient (more relevant → more helpful),
or is the relationship non-monotonic? Test in BOTH truncated-prefix and suffix modes to
distinguish value contamination from attention-based effects.

### Design

16 conditions × 1000 samples. 7 semantic levels (0 = random tokens → 5 = oracle keywords)
tested in both truncated-prefix (9 conditions) and suffix (7 conditions) modes.

| # | Condition | Mode | Semantic Level | Cohen's d | Win% | p |
|---|-----------|------|:-:|:-:|:-:|---|
| 1 | bare | — | — | 0.000 | — | — |
| 2 | random_trunc | Trunc | 0 | +0.170 | 67.1% | 2.7e-07 |
| 3 | random_words_trunc | Trunc | 0.5 | +0.179 | 62.5% | 5.8e-08 |
| 4 | wrong_doc_llm_trunc | Trunc | 1 | +0.239 | 62.9% | 7.6e-13 |
| 5 | tfidf_kw_trunc | Trunc | 2 | +0.124 | 60.4% | 1.6e-04 |
| 6 | llm_kw_trunc | Trunc | 3 | +0.253 | 69.2% | 3.1e-14 |
| 7 | **static_fact_trunc** | **Trunc** | **4** | **+0.438** | **81.8%** | **2.7e-37** |
| 8 | oracle_kw_trunc | Trunc | 5 | +0.124 | 60.6% | 1.8e-04 |
| 9 | oracle_raw_trunc | Trunc | 5* | +0.069 | 52.5% | 0.036 |
| 10 | random_suffix | Suffix | 0 | +0.294 | 65.9% | 1.9e-18 |
| 11 | random_words_suffix | Suffix | 0.5 | +0.201 | 61.4% | 1.4e-09 |
| 12 | wrong_doc_llm_suffix | Suffix | 1 | +0.166 | 60.9% | 4.8e-07 |
| 13 | tfidf_kw_suffix | Suffix | 2 | +0.349 | 70.6% | 5.0e-25 |
| 14 | llm_kw_suffix | Suffix | 3 | +0.179 | 60.8% | 5.9e-08 |
| 15 | static_fact_suffix | Suffix | 4 | +0.188 | 63.3% | 1.3e-08 |
| 16 | oracle_kw_suffix | Suffix | 5 | +0.283 | 64.9% | 2.7e-17 |

### 10 Primary Comparisons (Bonferroni alpha = 0.005)

| # | Comparison | d | p | Sig | Answer |
|---|-----------|---|---|-----|--------|
| C1 | llm_kw vs random (trunc) | +0.073 | 0.026 | ns | NO — LLM barely beats random |
| C2 | **static_fact vs random (trunc)** | **+0.297** | **7.4e-19** | **\*\*\*** | **YES — static massively > random** |
| C3 | llm_kw vs wrong_doc (trunc) | +0.035 | 0.285 | ns | NO — right doc ≈ wrong doc |
| C4 | tfidf_kw vs random (trunc) | -0.033 | 0.310 | ns | NO — TF-IDF ≈ random |
| C5 | oracle_kw vs oracle_raw (trunc) | +0.050 | 0.131 | ns | NO — format doesn't matter |
| C6 | llm_kw vs random (suffix) | -0.080 | 0.015 | ns | NO — LLM WORSE than random |
| C7 | static_fact vs random (suffix) | -0.081 | 0.014 | ns | NO — static ≈ random in suffix |
| C8 | llm_kw vs wrong_doc (suffix) | +0.028 | 0.389 | ns | NO — right doc ≈ wrong doc |
| C9 | trunc vs suffix (llm_kw) | +0.021 | 0.517 | ns | NO — modes equivalent for LLM |
| C10 | **trunc vs suffix (static_fact)** | **+0.222** | **2.4e-11** | **\*\*\*** | **YES — trunc >> suffix for static** |

### Semantic Gradient Analysis

**Spearman rank correlation of semantic level vs Cohen's d:**

| Mode | r | p | Interpretation |
|------|---|---|----------------|
| Truncated | +0.036 | 0.939 | No gradient detected |
| Suffix | -0.143 | 0.760 | No gradient detected |
| Cross-mode rank agreement | -0.750 | 0.052 | Different conditions win in different modes |

**No monotonic semantic gradient was detected** with the 7 levels tested. The ranking of
conditions differs substantially between truncated and suffix modes. The cross-mode rank
correlation is r=-0.75 (marginally significant), meaning conditions that work well in
truncated mode tend to work differently in suffix mode.

Note: This does not rule out a semantic gradient under different operationalizations of
"relevance." Our 7-level scale (random → oracle) may not capture the relevant dimensions.
The static_fact outlier in particular suggests the relationship between content and priming
is more complex than a simple relevance gradient.

### Hardness Quintile Breakdown (Selected Conditions)

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) | Overall |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| static_fact_trunc | +0.256 | +0.453 | **+1.101** | **+1.050** | +0.604 | +0.438 |
| llm_kw_trunc | -0.030 | +0.192 | +0.280 | +0.524 | +0.342 | +0.253 |
| random_trunc | +0.256 | +0.257 | +0.233 | +0.449 | +0.137 | +0.170 |
| random_suffix | -0.056 | +0.204 | +0.350 | +0.378 | +0.503 | +0.294 |
| oracle_kw_suffix | -0.344 | +0.029 | +0.180 | +0.521 | +0.585 | +0.283 |

**static_fact_trunc** is again unique: it shows enormous benefit for mid-difficulty samples
(Q3: d=+1.101, Q4: d=+1.050) while still helping easy samples (Q1: d=+0.256). Other
conditions show the familiar pattern of hurting easy samples and helping hard ones.

### Key Findings

1. **No semantic gradient detected with this operationalization.** Under the 7-level scale
   tested, more semantically relevant content does not monotonically improve priming. Oracle
   keywords (level 5) perform worse than random tokens (level 0) in truncated mode
   (d=+0.124 vs +0.170). However, different operationalizations of "semantic relevance" or
   finer-grained content variations might reveal a gradient we haven't captured here.

2. **static_fact_trunc is an outlier, not a gradient endpoint.** Its d=+0.438 is 1.7× the
   next-best truncated condition (llm_kw_trunc, d=+0.253). It doesn't sit on a gradient —
   it appears to be a qualitatively different phenomenon. The phrase "What are the key facts
   I need to know?" creates uniquely beneficial value contamination that no other tested
   content matches.

3. **Document-specific content provides no detected advantage.** Right-doc LLM keywords
   (d=+0.253) are not significantly better than wrong-doc LLM keywords (d=+0.239) in
   truncated mode (C3: p=0.285). Same in suffix mode (C8: p=0.389). At least for
   keyword-style content, document relevance doesn't appear to matter for priming.

4. **Suffix and truncated modes have different winners.** Truncated: static_fact dominates.
   Suffix: tfidf_kw (d=+0.349) and random (d=+0.294) are best, while static_fact drops to
   d=+0.188. The mechanisms are genuinely different — value contamination (truncated) responds
   to different content than attention routing (suffix).

5. **static_fact's advantage is specific to value contamination.** C10 shows
   static_fact_trunc >> static_fact_suffix (d=+0.222, p<1e-11). The "magic phrase" effect
   requires value-level contamination; it does not work by providing useful attention targets
   at query time.

6. **Suffix mode: random and TF-IDF perform well.** In suffix mode, random tokens (d=+0.294)
   outperform LLM keywords (d=+0.179) and static_fact (d=+0.188). TF-IDF keywords are best
   (d=+0.349), possibly because they provide useful lexical overlap for attention.

### Interpretation

The results suggest the relationship between semantic content and priming benefit is more
complex than a simple relevance gradient. Two distinct mechanisms respond to different
content types:

- **Value contamination** (truncated mode) is dominated by the static_fact phrase, which
  appears to act as a "mode-switching prompt" rather than a semantic prime. Document-relevant
  content (oracle, LLM keywords) doesn't help more than random or wrong-doc content.

- **Attention routing** (suffix mode) responds to lexical/statistical features (TF-IDF
  keywords, random tokens) rather than semantic understanding. This mechanism may be
  exploitable via simpler, non-LLM methods.

The fact that these two mechanisms have almost opposite content preferences (r=-0.75 cross-mode)
means that single-condition experiments can be misleading. Future work should always test
both modes.

### Open Questions

1. **Why does static_fact create special value patterns?** This remains the central mystery.
   The phrase works via value contamination but is not "more relevant" in any standard sense.
2. **Would other "mode-switching" phrases work?** We've only tested one magic phrase. A
   systematic search over instructional phrasings might find even better ones.
3. **Is the lack of gradient an artifact of MS MARCO's short passages?** Longer documents
   might show different patterns where semantic relevance matters more. → **Answered by Exp 11: YES, longer docs show completely different (worse) behavior.**
4. **Does the suffix TF-IDF advantage hold for longer documents?** Lexical overlap effects
   might scale differently with document length.

---

## Exp 11: Long-Document Priming (Natural Questions)

**Notebook:** `11_long_document_priming.ipynb`
**Build script:** `scripts/build_nb11.py`
**Date:** 2026-02-12
**Results:** `results/exp11/results.json`
**Runtime:** ~72 minutes on NVIDIA L4
**Samples:** 390 evaluated, 369 valid (21 excluded for zero NLLs), SEED=42
**Status:** Complete. Plots in `results/exp11/analysis_plots.png`.

### Research Question

All v2 experiments (01-10) used MS MARCO v1.1, where passages average ~60 words (max 300).
Does our best priming approach (static_fact_trunc, d=+0.472 on MS MARCO) generalize to
longer documents? Prior evidence (v1 Exp 19) showed priming hurts on longer-document datasets
(CNN/DailyMail d=-1.31, NarrativeQA d=-0.35), but those used full-context mode with known
BPE/framing bugs. This experiment re-tests with clean v2 methodology.

### Design

5 conditions × 390 samples from Natural Questions (validation split, real Google queries
over full Wikipedia articles). Documents stratified into 4 length bins. Bonferroni alpha =
0.01 (5 comparisons).

| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | `[BOS][doc]` — baseline |
| 2 | static_fact_trunc | `[BOS]["What are the key facts?"\\n][doc]` → truncate + RoPE |
| 3 | random_trunc | `[BOS][random_tokens\\n][doc]` → truncate + RoPE |
| 4 | llm_kw_trunc | `[BOS][llm_keywords\\n][doc]` → truncate + RoPE |
| 5 | oracle_trunc | `[BOS][actual_NQ_query\\n][doc]` → truncate + RoPE |

**Length bins:**

| Bin | Word Range | N (valid) | Mean Words |
|-----|-----------|-----------|------------|
| short | 100-300 | 14 | ~200 |
| medium | 300-800 | 114 | ~550 |
| long | 800-2000 | 121 | ~1400 |
| very_long | 2000-4000 | 120 | ~3000 |

Note: Only 14 samples in the "short" bin (NQ documents are mostly long Wikipedia articles).
Short-bin results should be interpreted cautiously due to low N.

### Results — Overall (N=369)

| Condition | Mean NLL | d vs Bare | Win% | t | p-value | Sig |
|-----------|----------|----------|------|---|---------|-----|
| bare | 0.357 | — | — | — | — | — |
| static_fact_trunc | 0.359 | **-0.019** | 65.0% | -0.37 | 0.71 | ns |
| random_trunc | 0.363 | -0.034 | 64.2% | -0.66 | 0.51 | ns |
| llm_kw_trunc | 0.371 | -0.120 | 50.1% | -2.31 | 0.022 | * |
| oracle_trunc | 0.391 | **-0.188** | 61.5% | -3.61 | 3.5e-4 | *** |

**All conditions have negative Cohen's d — priming HURTS on Natural Questions.**

### 5 Primary Comparisons

| # | Comparison | d | p | Sig | Answer |
|---|-----------|---|---|-----|--------|
| C1 | static_fact vs bare | -0.019 | 0.71 | ns | NO — static_fact does NOT help |
| C2 | random vs bare | -0.034 | 0.51 | ns | NO — ANY prefix does NOT help |
| C3 | llm_kw vs bare | -0.120 | 0.022 | * | Marginal harm (not Bonferroni sig) |
| C4 | oracle vs bare | **-0.188** | **3.5e-4** | **\*\*\*** | **YES — oracle SIGNIFICANTLY HURTS** |
| C5 | static_fact vs random | +0.021 | 0.69 | ns | NO — no content advantage |

### Per Length Bin Analysis (d vs bare)

| Condition | Short (n=14) | Medium (n=114) | Long (n=121) | Very Long (n=120) |
|-----------|:---:|:---:|:---:|:---:|
| static_fact_trunc | **+0.378** | -0.053 | -0.163 | -0.088 |
| random_trunc | -0.067 | +0.057 | -0.202 | -0.195 |
| llm_kw_trunc | -0.202 | -0.100 | -0.166 | -0.127 |
| oracle_trunc | -0.358 | -0.157 | -0.224 | -0.157 |

- **Short bin (100-300w):** static_fact_trunc still shows a positive effect (d=+0.378),
  consistent with MS MARCO findings. But n=14 is too small for reliable inference.
- **Medium+ bins:** All conditions are negative across all bins. The failure is not gradual —
  it sets in immediately once documents exceed ~300 words.

### Length Interaction — Correlation Analysis

| Condition | Spearman r | p | Pearson r | p |
|-----------|:---------:|---|:---------:|---|
| static_fact_trunc | -0.040 | 0.44 | -0.054 | 0.31 |
| random_trunc | -0.004 | 0.94 | -0.036 | 0.49 |
| llm_kw_trunc | +0.038 | 0.47 | +0.033 | 0.53 |
| oracle_trunc | +0.059 | 0.26 | +0.110 | 0.03 |

No significant continuous length-effect correlations for any condition (except a marginal
oracle Pearson r=+0.11, p=0.03). The priming failure is a step function at ~300 words,
not a gradual decay.

### Hardness × Length Interaction

| Bin | static_fact Easy d | static_fact Hard d | oracle Easy d | oracle Hard d |
|-----|:---:|:---:|:---:|:---:|
| medium | -0.03 | -0.07 | -0.18 | -0.22 |
| long | **+0.40** | **-0.23** | -0.12 | -0.31 |
| very_long | **+0.49** | **-0.13** | -0.09 | -0.22 |

**Critical finding: The hardness interaction INVERTS on longer documents.** On MS MARCO,
hard samples benefit most from priming and easy samples benefit least. On NQ long/very_long
documents, the pattern reverses: static_fact_trunc helps easy samples (d=+0.40 to +0.49)
but hurts hard samples (d=-0.13 to -0.23). This inversion likely reflects a different
difficulty distribution — "hard" NQ items may involve complex multi-hop or multi-sentence
reasoning that priming disrupts, unlike MS MARCO "hard" items which are simply ambiguous
short passages.

### Key Findings

1. **Priming fails on NQ/Wikipedia documents**, confirming v1 Exp 19 with clean v2
   methodology. The result is now unambiguous: even with correct BPE matching, truncated
   prefix mode, and no framing artifacts, priming does not help on documents >300 words.

2. **static_fact_trunc's "magic" is MS MARCO-specific.** The phrase drops from d=+0.472
   (MS MARCO) to d=-0.019 (NQ overall). It shows a faint positive signal only in the
   short bin (d=+0.378, n=14), which is the closest to MS MARCO passage lengths.

3. **Oracle prefix significantly HURTS** (d=-0.188, p=3.5e-4, Bonferroni significant).
   Priming with the actual NQ query makes performance worse, especially on short and long
   documents. Semantic specificity actively interferes.

4. **The failure is a step function, not a gradient.** No significant continuous
   length-effect correlations. The transition happens at ~300 words — below that, priming
   may still work; above it, priming is neutral to harmful.

5. **Hardness interaction inverts on long documents.** Easy samples benefit, hard samples
   hurt — opposite of MS MARCO. The priming mechanism that helps with "difficult short
   factoid retrieval" is different from what's needed for "difficult long document
   comprehension."

6. **Win rates are misleadingly high.** Despite negative d values, win rates range 50-65%.
   This is because NLL differences are small on most samples (many near zero), so a slight
   majority of tiny positive deltas coexists with a few large negative deltas that dominate
   the mean. Win rate alone is a poor metric for this regime.

### Interpretation

The MS MARCO priming benefit (Exps 01-10) is specific to **short factoid passages** where:
- Documents are short enough that prefix value contamination meaningfully shifts early-position
  representations (Exp 09: first 25% of positions carry the signal)
- Queries are factoid-style, matching the "What are the key facts?" mode-switching effect
- The answer can be improved by small representational nudges

On NQ's longer Wikipedia articles:
- The prefix value contamination signal is diluted across thousands of tokens
- Even the first 25% of a 2000-word document is 500 words of value vectors — far more than
  the entire MS MARCO passage
- The model's task shifts from "retrieve fact from short passage" to "comprehend and navigate
  long document" — a qualitatively different challenge that priming disrupts

**This definitively narrows the applicability of KV cache priming to short-passage factoid QA
(MS MARCO-like scenarios).** For longer documents, priming should be OFF.

### Updated Deployment Recommendation

```python
def should_prime(passage, task_type):
    word_count = len(passage.split())
    # Only prime short passages (MS MARCO-like)
    if word_count > 300:
        return False
    # Only factoid QA benefits
    if task_type not in ["factoid_qa", "short_answer"]:
        return False
    return True
```

### Lessons Learned

1. **Always test across document lengths** — MS MARCO's short passages are not representative.
   A d=+0.472 finding on 60-word passages tells you nothing about 600-word passages.
2. **Win rate is misleading in small-effect regimes** — look at Cohen's d and mean delta,
   not win percentage.
3. **Hardness interaction is dataset-specific** — the "hard samples benefit" pattern from
   MS MARCO inverts on NQ. Don't assume hardness gating generalizes across datasets.
4. **The "short bin" (100-300w) positive signal is suggestive but unreliable** — n=14 is
   too few for confident inference. Would need targeted sampling to confirm.
5. **NQ dataset loading requires careful extraction** — Wikipedia HTML tokens must be
   filtered, short answers vary in format across annotators, and streaming mode is needed
   for the large dataset.

---

## Experiment 12: Why Does Priming Fail on Long Documents? — Diagnostic Battery

**Date started**: 2026-02-13
**Notebook**: `12_long_doc_priming_diagnostic.ipynb` (built by `scripts/build_nb12.py`)
**Results**: `results/exp12/`

### Question

Exp 11 showed that priming collapses from d=+0.472 (MS MARCO) to d=-0.019 (NQ long docs),
with oracle actively hurting (d=-0.188). **Three hypotheses:** (A) signal dilution (7 prefix
tokens contaminate 4000 doc values = 0.2% dose vs 8% on MARCO), (B) attention redistribution
(values contribute too little per position), (C) positional/RoPE interference.

### Design

- Dataset: Natural Questions validation, 100-4000 word documents
- N = 299 valid samples (16 excluded), 4 length bins (short/medium/long/very_long)
- 9 conditions, 8 Bonferroni-corrected comparisons (α = 0.00625)
- All conditions use static_fact = "What are the key facts I need to know?"

| # | Condition | Build | Tests |
|---|-----------|-------|-------|
| 1 | bare | [BOS][doc] | Baseline |
| 2 | prefix_1x | [BOS][sf\n][doc] → trunc+RoPE | Confirms exp 11 failure |
| 3 | prefix_5x | [BOS][sf\n ×5][doc] → trunc+RoPE | Hyp A: 5x dose |
| 4 | prefix_20x | [BOS][sf\n ×20][doc] → trunc+RoPE | Hyp A: 20x dose |
| 5 | amplify_2x | bare keys + 2x boosted values | Hyp A+B: amplify delta |
| 6 | amplify_5x | bare keys + 5x boosted values | Hyp A+B: stronger boost |
| 7 | layers_0_15 | primed values only at layers 0-15 | Signal localization |
| 8 | suffix | [BOS][doc][sep][sf] (full context) | Hyp C: no RoPE needed |
| 9 | no_rope | [BOS][sf\n][doc] → trunc, NO RoPE | Hyp C: direct test |

### Results

**Primary comparisons vs bare (Bonferroni α=0.00625):**

| Condition | Cohen's d | Mean Δ NLL | Win Rate | p-value | Significant? |
|-----------|----------|-----------|----------|---------|-------------|
| prefix_1x | -0.016 | -0.0018 | 65.2% | 0.778 | No |
| prefix_5x | -0.017 | -0.0043 | 65.2% | 0.770 | No |
| prefix_20x | -0.026 | -0.0102 | 59.2% | 0.651 | No |
| amplify_2x | **+0.090** | +0.0248 | 57.9% | 0.122 | No |
| amplify_5x | +0.060 | +0.0271 | 46.8% | 0.301 | No |
| layers_0_15 | +0.083 | +0.0144 | 56.2% | 0.152 | No |
| suffix | **-0.196** | -0.0985 | 35.8% | **0.0008** | **Yes** |
| no_rope | **-0.205** | -0.0975 | 56.5% | **0.0005** | **Yes** |

**Hypothesis tests (direct comparisons):**

| Comparison | Cohen's d | p-value | Verdict |
|-----------|----------|---------|---------|
| 5x vs 1x (repetition helps?) | -0.015 | 0.794 | No — more reps doesn't help |
| 20x vs 1x (strong repetition?) | -0.026 | 0.657 | No — even 20x fails |
| amplify_2x vs 1x | +0.131 | 0.024 | Marginal, not Bonf. sig. |
| layers_0_15 vs 1x | +0.148 | 0.011 | Marginal, not Bonf. sig. |
| suffix vs 1x (RoPE issue?) | -0.191 | 0.001 | Suffix significantly WORSE |
| no_rope vs 1x (correction helps?) | -0.192 | 0.001 | Removing RoPE significantly WORSE |

### Key Findings

1. **Hypothesis A (signal dilution) — REFUTED.** Repeating the prefix 5x or 20x does not
   help. More repetitions actually trend slightly worse (20x d=-0.026).

2. **Hypothesis C (RoPE interference) — REFUTED.** Suffix mode (no RoPE correction needed)
   and skipping RoPE correction both significantly hurt (d≈-0.20, Bonferroni significant).
   RoPE correction is essential; removing it destroys the signal.

3. **Amplification and layer targeting show faint positive signals** but neither reaches
   significance. amplify_2x (d=+0.090) and layers_0_15 (d=+0.083) are the best conditions
   but remain in noise territory for N=299.

4. **Diagnostic analysis revealed answer position as the key confound:** 80% of NQ answers
   are in the first 25% of the document, exactly where contamination is strongest. Priming
   HURTS when answers are early (d=-0.052), HELPS when answers are late (d=+0.145).

### Interpretation

The failure is not about signal dilution or RoPE artifacts. It's about **where the answer
lives relative to the contaminated positions**. Contamination is strongest at early positions
(exponential decay), and NQ answers overwhelmingly cluster there. The contamination disrupts
exactly the positions the model needs to read. This motivates Exp 13: position-selective
contamination.

### Lessons Learned

1. **Repetition is not the answer** — 20x the prefix tokens doesn't overcome the fundamental
   position-answer interaction.
2. **Suffix mode is worse, not better** — the intuition that "avoid RoPE correction" helps
   is wrong. Suffix introduces its own interference (the model sees sf after the document,
   changing what it attends to).
3. **Always check WHERE answers are in your documents** — position distributions dominate.

---

## Experiment 13: Position-Aware Value Contamination for Long Documents

**Date started**: 2026-02-14
**Notebook**: `13_position_aware_priming.ipynb` (built by `scripts/build_nb13.py`)
**Results**: `results/exp13/`

### Question

Exp 12's diagnostic found that contamination disrupts early positions where 80% of NQ
answers live. The delta distribution has extreme kurtosis (73-227): a few catastrophic
outliers drag the mean negative despite 65% win rate. `layers_0_15` and `amplify_2x` are
r=0.982 correlated (doing the same thing). **Can we recover the effect by modulating
contamination by position — reducing it where answers live and boosting it elsewhere?**

### Design

- Dataset: Same 315 NQ samples from Exp 12 (300 valid after exclusions)
- 10 conditions, 9 Bonferroni-corrected comparisons (α = 0.00556)
- Only 2 forward passes per sample (bare + primed); all conditions derive from delta manipulation
- Position-variant conditions manipulate which positions receive contamination

| # | Condition | Description | Tests |
|---|-----------|-------------|-------|
| 1 | bare | Baseline | — |
| 2 | standard_1x | static_fact_trunc (replicate exp 12) | Baseline priming |
| 3 | layers_0_15_amp2x | Layer targeting + 2x amplification | Combined best from exp 12 |
| 4 | layers_0_15_amp3x | Layer targeting + 3x amplification | Stronger boost |
| 5 | pos_normalized | Normalize delta to constant per-position L2 norm | Remove position bias |
| 6 | attenuate_first_25 | Scale delta ×0.25 at first 25% of positions | Reduce early disruption |
| 7 | skip_first_25 | Zero delta at first 25% of positions | Eliminate early disruption |
| 8 | last_50_only | Contaminate only positions 50-100% | Avoid answer region |
| 9 | window_25_75 | Contaminate only positions 25-75% | Middle band |
| 10 | pos_norm_L0_15 | Position normalization + layer targeting | Combined approach |

### Results

**Primary comparisons vs bare (Bonferroni α=0.00556):**

| Condition | Cohen's d | Mean Δ NLL | Win Rate | p-value | Significant? |
|-----------|----------|-----------|----------|---------|-------------|
| standard_1x | +0.073 | +0.0119 | 64.3% | 0.210 | No |
| layers_0_15_amp2x | +0.101 | +0.0295 | 51.3% | 0.082 | No |
| **layers_0_15_amp3x** | **+0.103** | +0.0384 | 45.7% | 0.076 | No |
| pos_normalized | -0.051 | -0.0300 | 62.0% | 0.378 | No |
| attenuate_first_25 | -0.030 | -0.0023 | 66.7% | 0.604 | No |
| skip_first_25 | -0.133 | -0.0069 | 64.3% | 0.022 | No |
| last_50_only | -0.128 | -0.0049 | 61.7% | 0.028 | No |
| window_25_75 | -0.078 | -0.0032 | 61.7% | 0.179 | No |
| pos_norm_L0_15 | -0.049 | -0.0289 | 60.0% | 0.394 | No |

**Position-targeting comparisons vs standard_1x:**

| Condition | Cohen's d | p-value | Verdict |
|-----------|----------|---------|---------|
| attenuate_first_25 | -0.148 | 0.011 | Significantly worse than standard |
| skip_first_25 | -0.144 | 0.013 | Significantly worse than standard |
| last_50_only | -0.120 | 0.038 | Worse than standard |
| window_25_75 | -0.112 | 0.054 | Marginal, worse |
| pos_normalized | -0.072 | 0.211 | Worse (ns) |
| pos_norm_L0_15 | -0.071 | 0.222 | Worse (ns) |

### Key Findings

1. **Position-selective contamination FAILS.** Every position-targeting condition (skip,
   attenuate, window, last_50) is worse than standard uniform contamination. The intuition
   "reduce contamination where answers live" is wrong.

2. **Layer-targeted amplification remains the best approach.** layers_0_15_amp3x (d=+0.103)
   is the strongest condition, but still not significant at N=300.

3. **Removing contamination from any position range hurts.** The contamination signal is
   holistic — it modulates representations globally, not just at the contaminated positions.
   Selectively zeroing positions breaks the coherence.

4. **Win rates are misleading.** standard_1x has 64.3% win rate but only d=+0.073 — the
   familiar pattern of many small wins and a few large losses that dominate the mean.

5. **Per-bin analysis:** standard_1x shows d=+0.370 on short docs (n=14, unreliable) but
   d=-0.108 on very_long docs (n=97). The length threshold is robust.

### Interpretation

The value contamination mechanism is not amenable to position-selective targeting. The
contamination from early layers (0-15) propagates through the entire sequence via attention
at later layers — it's not a local position-by-position effect. Trying to "protect" early
positions by removing contamination there actually removes the global representational shift
that sometimes helps.

**The amplification approach (layers_0_15_amp3x) represents the ceiling of what position-
and layer-level manipulation can achieve on long documents: d≈+0.10, not significant.**
Further progress requires a fundamentally different approach — either different use cases
(ranking instead of NLL) or different data regimes (short docs where priming works).

### Lessons Learned

1. **Position-selective manipulation is a dead end** — contamination is holistic, not local.
2. **Layer targeting + amplification is r=0.982 correlated with amplification alone** —
   they're not independent mechanisms.
3. **N=300 is borderline for detecting d=0.10 effects** — would need N~1500 for 80% power.
4. **Always compare position-targeting conditions against standard priming, not just bare** —
   the relevant question is "does selective improve on uniform?" not "does selective beat nothing?"

---

## Experiment 14: Ranking-Aware Priming — Does Priming Improve Ad Ranking?

**Date started**: 2026-02-14
**Notebook**: `14_ranking_aware_priming.ipynb` (built by `scripts/build_nb14.py`)
**Results**: `results/exp14/`

### Question

All prior experiments (Exps 01-13) measured priming as per-document NLL deltas. But for
ad serving, what matters is **ranking**: does priming help rank the relevant document higher
among candidates? MS MARCO v1.1 has ~10 candidate passages per query with `is_selected`
labels — a natural ranking evaluation. Does priming create a **differential effect** where
relevant passages benefit more than irrelevant ones?

### Design

- Dataset: MS MARCO v1.1 validation, short passages (≤300 words)
- N = 300 queries, ~8 passages per query (~2400 passages total)
- 6 conditions for ranking
- Per-query alpha optimization (21 values from 0 to 1) for the combined condition
- Cross-validation analysis to estimate honest improvement

| # | Condition | Ranking Signal | Tests |
|---|-----------|---------------|-------|
| 1 | bare | bare NLL | Baseline |
| 2 | primed_1x | static_fact truncated NLL | Does uniform priming help ranking? |
| 3 | primed_amp2x | L0-15 2x amplified NLL | Stronger contamination |
| 4 | oracle_gated | primed if bare_nll > median, else bare | Selective priming |
| 5 | delta_signal | bare_nll - primed_nll (as score) | Is delta a relevance signal? |
| 6 | combined | α·bare + (1-α)·(-delta), α tuned per query | Complementary signals? |

### Results

**Ranking performance (MRR):**

| Condition | MRR | Hit@1 | Hit@3 | Mean Rank |
|-----------|-----|-------|-------|-----------|
| bare | 0.801 | 71.3% | 84.7% | 1.88 |
| primed_1x | 0.806 | 72.0% | 85.7% | 1.86 |
| primed_amp2x | 0.809 | 72.0% | 86.3% | 1.85 |
| oracle_gated | 0.803 | 71.7% | 84.3% | 1.87 |
| delta_signal | 0.275 | 7.7% | 26.0% | 5.32 |
| combined (oracle α) | **0.851** | **78.0%** | **89.3%** | **1.61** |

**Significance tests (Wilcoxon signed-rank):**

| Comparison | ΔMRR | p-value | Significant? |
|-----------|------|---------|-------------|
| primed_1x vs bare | +0.005 | 0.580 | No |
| primed_amp2x vs bare | +0.008 | 0.390 | No |
| oracle_gated vs bare | +0.002 | 0.619 | No |
| delta_signal vs bare | -0.526 | 3.8e-37 | Yes (catastrophic) |
| combined vs bare | **+0.050** | **3.3e-7** | **Yes** |

**But the combined result is OVERFIT:**

The combined condition uses per-query oracle alpha (best of 21 alphas per query). This
inflates the result massively. Honest analysis:

| Metric | Value |
|--------|-------|
| Per-query oracle ΔMRR | +0.050 (overfit) |
| Best global alpha (0.75) MRR | 0.809 (+0.008 vs bare) |
| Global alpha landscape range | 0.801-0.809 (flat) |
| Cross-validated ΔMRR | +0.006 (ns, p≈0.21) |
| Fixed alpha=0.5 ΔMRR | +0.005 (ns, p≈0.58) |

**Differential effect analysis:**

| Metric | Value |
|--------|-------|
| Delta on relevant passages | 0.121 |
| Delta on irrelevant passages | 0.168 |
| Difference (relevant - irrelevant) | **-0.047** (wrong direction) |
| p-value | 0.004 (significant, but INVERTED) |
| Delta AUC for relevance | **0.429** (below 0.5 = anti-discriminative) |
| Delta-relevance correlation | r = -0.057 (weakly anti-correlated) |

### Key Findings

1. **Individual priming conditions don't significantly improve ranking.** primed_1x (+0.005)
   and primed_amp2x (+0.008) are both non-significant improvements over bare.

2. **The combined per-query oracle result (+0.050) is overfit.** Cross-validated honest
   estimate is only +0.006. The alpha landscape is flat (0.801-0.809 across 21 alphas).

3. **The differential effect is INVERTED.** Priming reduces NLL *more* for irrelevant
   passages than relevant ones (delta AUC = 0.429). This is the opposite of what's needed
   for ranking improvement.

4. **Delta-as-signal is catastrophic for ranking** (MRR = 0.275). The NLL delta is not
   a useful relevance predictor.

5. **Stratified results:** Medium-difficulty queries benefit most (MRR 0.905→0.930 for
   amp2x), but easy and hard queries show no meaningful improvement.

### Interpretation

Priming does not create a useful differential effect for ad ranking. The "combined" result
that looked impressive (+0.050 MRR) was entirely driven by per-query alpha overfitting.
The true ranking improvement from priming is approximately zero on MS MARCO.

The inverted differential effect (irrelevant passages benefit MORE from priming) makes
intuitive sense: irrelevant passages are harder to predict (higher bare NLL), and priming
acts as a "general fluency boost" that helps all text proportionally to its difficulty —
not proportionally to its relevance.

### Lessons Learned

1. **Per-query oracle optimization is a trap** — always cross-validate tuned parameters.
2. **Global alpha landscape flatness is the real signal** — when all alphas give ~same
   MRR, the combination has no real information.
3. **Differential effect can be inverted** — don't assume priming helps relevant docs
   more than irrelevant ones. Always measure empirically.
4. **Delta-as-signal is useless** — the NLL delta between bare and primed is noise, not
   a relevance predictor. AUC < 0.5 means it's anti-discriminative.

---

## Experiment 15: NLL Ensemble Ranking — Can Diverse Priming Improve Ranking?

**Date started**: 2026-02-15
**Notebook**: `15_nll_ensemble_ranking.ipynb` (built by `scripts/build_nb15.py`)
**Results**: `results/exp15/`

### Question

Exp 14 showed per-query alpha optimization is overfit and individual priming barely moves
ranking. But what about **ensembling diverse NLL signals**? Different priming prefixes may
make different scoring errors — averaging them could reduce noise. A critical control: does
a **non-primed rescore** (same bare cache, different prompt template) provide equal diversity,
or is the priming mechanism genuinely special?

### Design

- Dataset: MS MARCO v1.1 validation, short passages (≤300 words)
- N = 300 queries
- 5 individual scoring signals, 7 ensemble combinations
- Equal-weight NLL averaging (no tuning, no overfitting risk)
- Rescore control: bare cache scored with alt template ("Question:...Response:")

**Individual signals:**

| Signal | Source | Description |
|--------|--------|-------------|
| bare | bare cache, standard template | Baseline |
| rescore | bare cache, alt template | Control for prompt diversity |
| sf | static_fact primed cache | Best priming condition |
| rand | random-text primed cache | Non-semantic priming |
| intent | "What is this passage about?" primed cache | Semantic priming variant |

**Ensemble combinations (equal-weight NLL average):**

| Ensemble | Members | Tests |
|----------|---------|-------|
| ens_2_sf | bare + sf | Best single priming signal |
| ens_2_rand | bare + rand | Non-semantic noise reduction |
| ens_2_intent | bare + intent | Alternative semantic priming |
| ens_2_rescore | bare + rescore | Control: prompt diversity only |
| ens_3 | bare + sf + rand | Two diverse priming signals |
| ens_4 | bare + sf + rand + intent | Three priming signals |
| ens_5_all | bare + sf + rand + intent + rescore | Everything |

### Results

**Individual signal MRRs:**

| Signal | MRR | ΔMRR vs bare | p-value |
|--------|-----|-------------|---------|
| bare | 0.801 | — | — |
| rescore | 0.790 | -0.011 | 0.192 (ns) |
| sf | 0.806 | +0.005 | 0.580 (ns) |
| rand | 0.803 | +0.002 | 0.706 (ns) |
| intent | 0.793 | -0.008 | 0.288 (ns) |

**Ensemble MRRs:**

| Ensemble | MRR | ΔMRR | p-value | # Queries Changed |
|----------|-----|------|---------|-------------------|
| ens_2_sf | 0.806 | +0.005 | 0.363 | 21 |
| ens_2_rand | 0.808 | +0.007 | 0.160 | 20 |
| ens_2_intent | 0.801 | +0.000 | 0.942 | 27 |
| ens_2_rescore | 0.791 | -0.010 | 0.106 | 25 |
| **ens_3** | **0.808** | **+0.007** | 0.232 | 24 |
| ens_4 | 0.803 | +0.002 | 0.755 | 25 |
| ens_5_all | 0.806 | +0.004 | 0.497 | 26 |

**Priming vs control test (key result):**

| Comparison | MRR | Difference | p-value |
|-----------|-----|-----------|---------|
| ens_2_sf (primed) | 0.806 | — | — |
| ens_2_rescore (control) | 0.791 | — | — |
| sf - rescore | **+0.015** | | **0.039 (sig)** |

**Greedy scaling curve (add signals one at a time):**

| K | Added Signal | MRR |
|---|-------------|-----|
| 1 | bare | 0.801 |
| 2 | + rand | 0.808 |
| 3 | + sf | 0.808 |
| 4 | + rescore | 0.805 |
| 5 | + intent | 0.806 |

**NLL correlation matrix (key pairs):**

| Pair | r |
|------|---|
| bare–sf | 0.990 |
| bare–rand | 0.989 |
| bare–rescore | 0.982 |
| sf–rand | 0.990 |
| lowest pair | 0.972 (rescore–sf) |

### Key Findings

1. **No ensemble reaches statistical significance.** The best is ens_3 (bare+sf+rand)
   at ΔMRR=+0.007 (p=0.232). Ensembling diverse priming signals does not meaningfully
   improve ranking.

2. **Priming IS special vs the rescore control** (p=0.039). The ens_2_sf ensemble (0.806)
   significantly outperforms ens_2_rescore (0.791). This confirms that priming adds
   genuine signal beyond mere prompt diversity — but the signal is too small to matter.

3. **All NLL signals are extremely correlated** (r > 0.97 for all pairs). Different
   priming prefixes barely create different NLL patterns. Only 20-27 out of 300 queries
   change ranking under any ensemble.

4. **Adding more signals hurts after K=2.** The greedy curve peaks at K=2 (bare+rand,
   MRR=0.808) and adding more signals dilutes the improvement.

5. **Rescore (prompt diversity control) hurts ranking.** ens_2_rescore actually reduces
   MRR by 0.010 — prompt diversity without priming adds noise, not signal.

### Interpretation

The NLL ensemble approach is a dead end. The fundamental problem is that all NLL signals
are too correlated (r > 0.97) for ensembling to meaningfully reduce ranking noise. This
high correlation occurs because the NLL is dominated by document-intrinsic properties
(text perplexity, length, style) that priming barely perturbs.

The positive finding is that priming IS genuinely special vs the control (p=0.039), but
the effect size (ΔMRR ≈ +0.005-0.007) is too small for practical use. The priming mechanism
creates a real but tiny signal that is drowned by the much larger document-intrinsic NLL
variation.

**This closes the ranking-via-NLL-priming line of investigation.** If ranking improvement
is the goal, it would need fundamentally different representations (e.g., embedding-based
rather than NLL-based scoring) or dramatically different priming regimes.

### Lessons Learned

1. **Equal-weight ensembling avoids overfitting but can't create signal from noise** —
   when individual signals barely move the needle, averaging them doesn't help.
2. **Always include a non-priming control** — the rescore condition proved that priming
   is special (vs prompt diversity), which is the one positive finding.
3. **Correlation matrix is the first thing to check** — r > 0.97 between all signals
   means ensembling is mathematically limited (diversity ≈ 0).
4. **Greedy scaling curve is a simple diagnostic** — adding the 3rd signal gives +0.000,
   immediately showing diminishing returns.
5. **20-27 changed queries out of 300 is the ceiling** — even a perfect ensemble can
   only affect the ~7-9% of queries where ranking differs across signals.

---

## Experiment 20: Length-Controlled Padding — Is Priming Failure a Length or Dataset Effect?

**Date started**: 2026-02-15
**Notebook**: `20_length_controlled_padding.ipynb`
**Results**: `results/exp20/`

### Question

Priming works on short MS MARCO docs (Exp 07 d=+0.30, Exp 18 d=+0.30) and short NQ docs
(Exp 18 short bin d=+0.33) but fails on long NQ docs (Exp 18 medium/long/very_long all
negative). Exp 18 ruled out the distance hypothesis (periodic beacons didn't help).

**Is the failure on long docs a length effect or a dataset effect?** MS MARCO and NQ differ
in many ways (passage style, question type, answer format). The cleanest test: take MS MARCO
samples where priming works, artificially pad them to long-doc lengths, and see exactly when
the benefit disappears.

### Design

- Dataset: MS MARCO v1.1 validation, same 300 samples as Exp 17/18
- Prefix: `"What are the key facts I need to know?"` (static_factual, same as Exp 07/17/18)
- 2 conditions (bare, single_prefix) × 5 target lengths = 10 scores per sample
- Padding: unrelated MS MARCO passages appended at the **token level** (no BPE issues)
  - Random start offset into pre-tokenized padding pool for diversity
  - Answer span stays at the beginning — only total cache size changes

| Target Length | What it tests |
|---|---|
| original (~130 tok) | Baseline where priming works |
| 256 | Still short |
| 512 | Medium (~NQ medium bin) |
| 1024 | Long (~NQ long bin) |
| 2048 | Very long (~NQ very_long bin) |

### Results (N=278 valid after filtering)

| Length | Mean Bare NLL | Mean Prefix NLL | Mean Δ | Cohen's d | Win% | p | sig |
|--------|--------------|----------------|--------|-----------|------|---|-----|
| original (~130 tok) | 1.098 | 1.025 | +0.072 | **+0.303** | 66.5% | 7.7e-07 | *** |
| 256 tok | 1.022 | 0.996 | +0.026 | +0.114 | 64.7% | 0.059 | ns |
| 512 tok | 0.955 | 0.948 | +0.007 | +0.034 | 58.3% | 0.570 | ns |
| 1024 tok | 0.907 | 0.916 | -0.008 | -0.043 | 55.0% | 0.472 | ns |
| 2048 tok | 0.889 | 0.891 | -0.002 | -0.014 | 57.6% | 0.813 | ns |

### Key Findings

1. **Length IS the primary factor.** The priming benefit decays monotonically from d=+0.303
   at original length (~130 tokens) to effectively zero by 512 tokens, even though the
   content, domain, question type, and passage style are identical MS MARCO throughout.
   The benefit is already non-significant at 256 tokens (d=+0.114, p=0.059).

2. **The transition is sharp.** Between original (~130 tok, d=+0.30) and 256 tok (d=+0.11),
   priming loses ~2/3 of its effect. By 512 tokens the effect is gone (d=+0.03). This
   suggests a critical window of ~200 tokens beyond which value contamination is diluted
   below the noise floor.

3. **Bare NLL decreases with length** (1.098 → 0.889). Longer documents make the answer
   easier to predict because the padding provides additional context for the model. This
   means the *absolute* improvement from priming (+0.072 NLL at original) gets swamped by
   the much larger NLL reduction from document length itself (-0.209 NLL from original to
   2048).

4. **Weak per-sample correlation** (r=0.247, p=3.15e-05). Samples that benefit most from
   priming at original length tend to benefit slightly more at 2048 too, but the effect is
   massively attenuated — points cluster near zero at 2048 regardless of their original-
   length delta.

### Interpretation

This conclusively answers the open question from Exp 18: the failure of priming on long
NQ documents is primarily a **length effect**, not a dataset effect. The value contamination
from a short prefix (~10 tokens) gets diluted as the cache grows. Each additional document
token's KV representation is formed by attending to the full prefix+doc context, but when
the document dominates the cache (e.g., 2048 doc tokens vs 10 prefix tokens), the prefix's
contribution to each value vector is proportionally tiny.

Combined with Exp 18's finding that periodic beacons don't help, this paints a clear picture:
value contamination is a **local** phenomenon that only meaningfully affects documents
comparable in length to the prefix itself (~100-200 tokens). There is no known way to
extend it to longer documents.

### Implications for Deployment

The length constraint is more severe than previously understood:
- **< ~200 tokens**: Priming can help (d ≈ +0.30, but only for hard samples)
- **200-500 tokens**: Marginal, likely not worth the compute
- **> 500 tokens**: No benefit; priming is wasted effort

This means priming is viable only for very short content: search snippets, ad copy, product
descriptions, short social posts. Most real-world documents (articles, pages, long-form
content) are beyond the effective range.

### Lessons Learned

1. **Controlled padding is a clean way to isolate length effects** — avoids confounding
   domain, style, and question-type differences between datasets.
2. **Token-level padding avoids BPE boundary issues** — pre-tokenize the pool, then
   concatenate IDs directly.
3. **Always check if baseline NLL changes across conditions** — the bare NLL drop with
   length is a reminder that the denominator of the comparison shifts too.

---

## Experiment 16: Cross-Model Priming Replication (Gemma 3 4B)

**Date started**: 2026-02-15
**Notebook**: `16_cross_model_gemma3.ipynb` (built by `scripts/build_nb16.py`)
**Results**: `results/exp16/`

### Question

All 15 prior experiments used Mistral-7B exclusively. The critical open question: **is value
contamination via priming a universal transformer mechanism, or Mistral-specific?** Gemma 3 4B
has a substantially different architecture (34 layers, head_dim=256, per-layer RoPE with two
theta values, 4 KV heads, bfloat16) — if priming replicates here, the mechanism is likely
universal.

### Design

- **Model**: Gemma 3 4B (`google/gemma-3-4b-it`, 4-bit quantized, bfloat16)
- **Dataset**: MS MARCO v1.1 validation, ≤300 words, ≥2 passages, N=300 queries (2174 passages)
- **5 conditions**: bare, static_fact_trunc, random_trunc, oracle_trunc, values_only
- All conditions use truncation + RoPE correction (same pipeline as Mistral experiments)
- values_only = bare keys + static_fact primed values (hybrid cache, tests mechanism)

| # | Condition | Description |
|---|-----------|-------------|
| 1 | bare | Baseline — no prefix |
| 2 | static_fact_trunc | "What are the key facts?" prefix, truncated+RoPE |
| 3 | random_trunc | Random text prefix, truncated+RoPE |
| 4 | oracle_trunc | Actual query as prefix, truncated+RoPE |
| 5 | values_only | Bare keys + sf primed values (hybrid cache) |

### Results

**Priming does NOT replicate on Gemma 3 4B.** The effects are opposite in sign from Mistral.

| Condition | Mistral d | Gemma d | Gemma p | Gemma Win% | Verdict |
|-----------|----------|---------|---------|------------|---------|
| static_fact_trunc | **+0.472** | **-0.031** | 0.145 (ns) | 45.2% | Does NOT replicate |
| random_trunc | +0.091 | **-0.109** | 4.4e-7 (***) | 40.0% | **Significantly hurts** |
| oracle_trunc | +0.023 (ns) | -0.020 (ns) | 0.341 (ns) | 46.6% | Neutral on both |
| values_only | +0.275 | **+0.056** | 0.009 (**) | 49.2% | Small positive (only sig condition) |

**Mechanism decomposition (Gemma):**
- Full static_fact (keys + values): d = -0.031 (ns)
- Values-only (bare keys + primed values): d = +0.056 (**)
- Keys carry **negative** interference: when primed keys are included, the net effect flips
  from +0.056 to -0.031. On Mistral, keys were neutral (-0.009 ns).

**Hardness interaction — INVERTED on easy samples:**

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) |
|-----------|-----------|-----|-----|-----|-----------|
| static_fact_trunc | -0.275 | -0.248 | -0.134 | -0.036 | **+0.148** |
| random_trunc | -0.243 | -0.372 | -0.254 | -0.145 | +0.051 |
| values_only | -0.242 | -0.083 | +0.020 | +0.083 | **+0.219** |

The hardness gradient EXISTS on Gemma (hard samples benefit, easy samples hurt) but the crossover
point is much higher — only Q5 (hardest 20%) sees any benefit, and the damage to easy samples
overwhelms the gain. On Mistral, the crossover was around Q3 (median difficulty).

### Key Findings

1. **Priming is NOT a universal transformer mechanism.** The best Mistral condition
   (static_fact d=+0.472) is effectively zero on Gemma (d=-0.031, ns). Random prefix
   **significantly hurts** (d=-0.109, p<0.001) — the opposite of Mistral.

2. **Value contamination signal EXISTS but is overwhelmed by key interference.**
   values_only (bare keys + primed values) is the only significantly positive condition
   (d=+0.056, p=0.009). But when primed keys are included (full truncated cache),
   the effect flips negative. On Mistral, keys were negligible; on Gemma, keys actively
   interfere.

3. **The key interference may be due to Gemma's larger head_dim (256 vs 128).**
   With head_dim=256, each key dimension encodes a narrower frequency band in RoPE.
   The RoPE correction (inverse rotation by prefix offset) may introduce larger
   numerical errors in bfloat16 with 256 dimensions than float16 with 128 dimensions.
   Alternatively, Gemma's per-layer RoPE (10k/1M theta alternation) may create
   mismatches when a single correction offset is applied uniformly.

4. **The hardness gradient is weaker but directionally consistent.** Both models show
   that hard samples benefit more from priming, but on Gemma the crossover is shifted
   to only the hardest quintile, and the net effect is still negative.

5. **This further narrows the applicability of priming.** Combined with Exp 11 (fails
   on long docs) and Exp 19 v1 (fails on non-MARCO datasets), priming now also fails
   on a different model architecture. The Goldilocks zone is: Mistral-7B + short MARCO
   passages + hard samples.

### Interpretation

The result is a **strong negative** for the universality hypothesis. Value contamination
is detectable on Gemma (values_only d=+0.056) but too weak to overcome key interference.
The most likely explanation is architectural:

- **head_dim=256** means RoPE correction operates on 128 frequency pairs (vs 64 on Mistral).
  The bfloat16 precision (~3 significant digits) may be insufficient for accurate correction
  at higher frequencies, introducing systematic key noise that degrades attention.
- **Per-layer RoPE (10k sliding / 1M full)** means 5/6 of layers use theta=10k, where
  position corrections produce larger angular changes than theta=1M. This amplifies any
  numerical error from the correction.
- **4 KV heads (vs 8)** means each head covers more of the representation, so noise in
  any single head has a larger impact on the overall attention distribution.

### Lessons Learned

1. **Always test on multiple architectures before claiming a universal mechanism.** 15
   experiments on one model can produce compelling but model-specific results.
2. **Mechanism decomposition (values_only) is essential.** Without it, we'd only see
   "priming doesn't work on Gemma" — not that values help but keys hurt.
3. **RoPE correction precision depends on head_dim and dtype.** The float16/128-dim
   combination on Mistral may be in a sweet spot that bfloat16/256-dim on Gemma is not.
4. **The hardness gradient is the most robust finding across models** — even when the
   net effect changes sign, the direction (hard benefits, easy hurts) is consistent.

---

## Exp 19: Gemma Priming — Precision Fix & Selective Value Contamination

**Date:** 2026-02-15
**Notebook:** `19_gemma_precision_and_selectivity.ipynb`
**Results:** `results/exp19/results.json`
**Runtime:** ~2h 19m on L4 GPU

### Motivation

Exp 16 showed priming FAILS on Gemma 3 4B (static_fact d=-0.031, ns), but `values_only`
works (d=+0.056, p=0.009). The gap reveals -0.087 of key interference. Two hypotheses:

- **H1 (Precision):** RoPE correction in bfloat16 (7-bit mantissa) with head_dim=256
  introduces ~8.6x more quantization noise than Mistral's float16/128-dim. Computing
  correction in float32 may recover the effect.
- **H2 (Selectivity):** On Mistral (Exp 09), value contamination signal lives in layers
  0-15 (88%) and first 25% of positions. Targeting these on Gemma may amplify d=+0.056.

### Design

9 conditions on MS MARCO v1.1 (N=300 queries, 2174 passages), 2 forward passes per
passage + 9 scoring calls. 7 primary comparisons with Bonferroni α = 0.00714.

| # | Condition | Description | Tests |
|---|-----------|-------------|-------|
| 1 | `bare` | Baseline | — |
| 2 | `sf_trunc` | Standard truncated + bfloat16 RoPE correction | Exp 16 reference |
| 3 | `sf_trunc_fp32` | Truncated + float32 RoPE correction | H1: precision |
| 4 | `sf_trunc_nocorr` | Truncated, NO RoPE correction | Correction vs mismatch |
| 5 | `values_only` | Bare keys + sf primed values (all layers) | Exp 16 replication |
| 6 | `values_early_layers` | Values_only, layers 0-16 only | H2: layer selectivity |
| 7 | `values_early_pos` | Values_only, first 25% of doc positions | H2: position selectivity |
| 8 | `values_alpha_25` | 25% primed / 75% bare value blend | H2: dose reduction |
| 9 | `rope_roundtrip` | Bare + RoPE roundtrip noise on keys | Control: noise baseline |

### Results

| Condition | d vs bare | Win% | Interpretation |
|-----------|----------|------|----------------|
| `bare` | 0.000 | — | Baseline |
| `sf_trunc` | -0.031 | — | Replicates Exp 16 (keys+values hurt) |
| `sf_trunc_fp32` | -0.032 | — | fp32 makes zero difference |
| `sf_trunc_nocorr` | -0.009 | — | No correction ≈ neutral |
| `values_only` | +0.056 | 49.2% | Replicates Exp 16 (values help) |
| **`values_early_layers`** | **+0.211** | **62.4%** | **4x amplification — key finding** |
| `values_early_pos` | +0.010 | — | Position selectivity doesn't help |
| `values_alpha_25` | +0.081 | 50.6% | Dose reduction helps slightly |
| `rope_roundtrip` | -0.019 | — | Pure noise cost is small |

### Primary Comparisons (Bonferroni α = 0.00714)

| # | Comparison | d | p | Bonferroni sig? |
|---|-----------|---|---|-----------------|
| C1 | fp32 vs bf16 | -0.016 | 4.4e-01 | No |
| C2 | nocorr vs bare | -0.009 | 6.7e-01 | No |
| C3 | nocorr vs bf16 | +0.021 | 3.3e-01 | No |
| C4 | values_only vs bare | +0.056 | 8.7e-03 | No (marginal) |
| C5 | early_layers vs values_only | **+0.170** | **3.7e-15** | **Yes** |
| C6 | early_pos vs values_only | -0.124 | 7.7e-09 | Yes (worse) |
| C7 | alpha_25 vs values_only | -0.043 | 4.3e-02 | No |

### Key Derived Metrics

- **Precision gain (fp32 - bf16):** -0.001 — bfloat16 is NOT the bottleneck
- **Key interference (bf16):** +0.087 — keys subtract 0.087 from values' +0.056
- **Key interference (fp32):** +0.089 — identical to bf16, confirming precision is irrelevant
- **Noise baseline (rope_roundtrip):** -0.019 — pure bfloat16 noise is small
- **Best selective condition:** `values_early_layers` d=+0.211 (3.8x values_only)

### Verdicts

**H1 REJECTED: fp32 correction DOES NOT RECOVER priming on Gemma.**

The gain is -0.001 (effectively zero). bfloat16 quantization during RoPE correction is
not the mechanism causing key interference. The interference is content-based, not
precision-based — primed keys carry information that actively hurts attention routing
regardless of numeric precision.

**H2 SUPPORTED: Selective contamination AMPLIFIES the value signal.**

Restricting primed values to layers 0-16 (first 50% of 34 layers) quadruples the effect
from d=+0.056 to d=+0.211 (p=3.7e-15, Bonferroni significant). This reveals that
**late-layer values (17-33) carry interference that dilutes the early-layer signal**.

This mirrors Exp 09 on Mistral, where layers 0-15 carried 88% of the signal. But on
Gemma the relationship is stronger: late layers don't just contribute less — they
actively *hurt*. The net effect of all-layer values (+0.056) is the sum of a strong
early-layer benefit (+0.211) partially cancelled by late-layer harm.

Position selectivity (first 25% only) does NOT help on Gemma (d=+0.010 vs +0.056),
unlike Mistral where first-quarter positions were dominant. The alpha=0.25 blend (d=+0.081)
shows a modest improvement over full values (+0.056), consistent with the idea that
reducing the late-layer "dose" helps.

### Hardness Interaction

| Condition | Q1 (easy) | Q2 | Q3 | Q4 | Q5 (hard) | Overall |
|-----------|----------|-----|-----|-----|-----------|---------|
| sf_trunc | -0.275 | -0.248 | -0.134 | -0.036 | +0.148 | -0.031 |
| sf_trunc_fp32 | -0.277 | -0.248 | -0.134 | -0.040 | +0.145 | -0.032 |
| values_only | -0.242 | -0.083 | +0.020 | +0.083 | +0.219 | +0.056 |
| values_early_layers | -0.112 | +0.142 | +0.311 | +0.270 | +0.365 | +0.211 |
| values_alpha_25 | -0.074 | +0.031 | +0.138 | +0.117 | +0.144 | +0.081 |

The hardness gradient is consistent: hard samples benefit, easy samples hurt. But
`values_early_layers` shifts the crossover dramatically — even Q2 benefits (+0.142),
and Q3-Q5 all show d > +0.27. This suggests early-layer selective contamination
could be practically useful on Gemma for medium-to-hard queries.

### Interpretation

1. **The key interference problem on Gemma is NOT about numeric precision.** fp32 RoPE
   correction is identical to bfloat16. The interference is content-based: primed keys
   encode prefix information that misdirects attention to irrelevant positions.

2. **Late-layer value contamination is harmful on Gemma.** Unlike Mistral where all
   layers contribute positively (just with different magnitudes), Gemma's late layers
   (17-33) carry information that actively cancels the early-layer benefit. This may
   relate to Gemma's per-layer RoPE structure — full-attention layers (5, 11, 17, 23, 29)
   with theta=1M may process priming information differently than sliding-attention
   layers with theta=10k.

3. **Layer-selective values is a viable strategy on Gemma.** d=+0.211 approaches
   Mistral's values_only (d=+0.275) and is practically meaningful. If combined with
   hardness gating (Q3-Q5 only), the effective d would be even larger.

4. **The RoPE correction itself is neutral on Gemma.** `sf_trunc_nocorr` (d=-0.009) is
   essentially identical to `sf_trunc` (d=-0.031) — neither helps. This is surprising:
   on Mistral, RoPE correction is essential. On Gemma, the problem is upstream of
   the correction step.

### Lessons Learned

1. **Precision is rarely the bottleneck.** When something doesn't work, check content-level
   mechanisms before numerical ones. The -0.087 key interference is a content effect, not
   a quantization artifact.
2. **Layer selectivity can dramatically change outcomes.** The 4x amplification from
   all-layers to early-layers-only was unexpected. Always test selective conditions.
3. **"Late layers hurt" is a model-specific finding** that likely depends on Gemma's
   architecture (per-layer RoPE, sliding/full attention pattern). Don't assume Mistral's
   layer-wise profile generalizes.
4. **The hardness gradient remains the most robust finding.** It holds across both models,
   across all conditions, and across selective variants. Hard samples consistently benefit.

---

## Experiment 24: Gemma Layer-Selective Mechanism Deep Dive

**Date**: 2026-02-16
**Notebook**: `24_gemma_layer_mechanism.ipynb`
**Results**: `results/exp24/`
**Model**: Gemma 3 4B (4-bit, bfloat16)

### Question

Exp 21 confirmed layer-selective value contamination (layers 0-15, cutoff=16) produces
d=+0.227 on Gemma with MS MARCO. Four questions remained:

1. Which individual layers carry the signal?
2. Does this generalize to SQuAD v2?
3. Does prefix content matter under layer selectivity?
4. What makes early-layer values physically different?

### Design

| Part | Data | N | Conditions |
|------|------|---|------------|
| 1+4 | MS MARCO | 300 | 34 single-layer replacements + value features |
| 2 | SQuAD v2 | 400 | bare, values_all (34 layers), values_cutoff_16 (layers 0-15) |
| 3 | MS MARCO | 300 | bare + 3 prefix types (static_fact, random, oracle) at cutoff=16 |

### Part 1 Results: Individual Layer Contribution Map

The signal is **concentrated in a handful of layers**, not a smooth gradient:

| Layer | d | sig | Layer | d | sig |
|-------|---|----|-------|---|-----|
| L0 | -0.062 | ns | L17 | -0.010 | ns |
| L1 | +0.004 | ns | L18 | **-0.196** | *** |
| L2 | +0.081 | ns | L19 | +0.084 | ns |
| L3 | +0.026 | ns | L20 | **+0.195** | *** |
| L4 | +0.044 | ns | L21 | -0.114 | * |
| L5 | +0.002 | ns | L22 | -0.144 | * |
| L6 | +0.060 | ns | L23 | -0.110 | ns |
| L7 | +0.037 | ns | L24 | +0.067 | ns |
| L8 | +0.099 | ns | L25 | +0.098 | ns |
| L9 | +0.023 | ns | L26 | +0.051 | ns |
| **L10** | **+0.207** | *** | **L27** | **-0.213** | *** |
| L11 | -0.083 | ns | L28 | +0.026 | ns |
| **L12** | **+0.198** | *** | L29 | -0.044 | ns |
| L13 | -0.052 | ns | L30 | -0.025 | ns |
| **L14** | **+0.197** | *** | L31 | +0.080 | ns |
| **L15** | **+0.238** | *** | L32 | +0.030 | ns |
| L16 | -0.060 | ns | L33 | +0.164 | ** |

**Top 5**: L15 (+0.238), L10 (+0.207), L12 (+0.198), L14 (+0.197), L20 (+0.195)
**Bottom 5**: L23 (-0.110), L21 (-0.114), L22 (-0.144), L18 (-0.196), L27 (-0.213)

**Early (0-15)**: mean d=+0.064, 13/16 positive
**Late (16-33)**: mean d=-0.007, 9/18 positive

Key insight: The signal is NOT a smooth early-vs-late gradient. It clusters in a
"hero band" at **layers 10, 12, 14, 15** (all d ≈ +0.20, all p < 0.001), with a
secondary contributor at **L20** (+0.195). Destructive layers cluster at **L18, L21-23,
L27** (all d < -0.10). Most layers individually are non-significant noise.

This explains why cutoff=16 is optimal — it captures ALL four hero layers while
excluding the destructive L18/L21-23 block. But the early layers 0-9 are mostly
noise (mean d=+0.033), suggesting a cherry-pick of just {10,12,14,15,20} might
outperform the blanket cutoff.

### Part 2 Results: Cross-Dataset — SQuAD v2

| Condition | d | Win% | p | sig |
|-----------|---|------|---|-----|
| values_all (34 layers) | -0.093 | 27.7% | 0.095 | ns |
| values_cutoff_16 (layers 0-15) | -0.031 | 24.9% | 0.578 | ns |

**Verdict: Layer-selective value contamination does NOT generalize to SQuAD v2.**

Both conditions are non-significant. values_all trends negative (consistent with
late-layer interference). values_cutoff_16 is neutral. This reinforces the finding
from Exp 11 that priming benefits are MS MARCO-specific — even with the layer-selective
method that works well on MS MARCO (d=+0.227), SQuAD v2 shows zero benefit.

The low win rates (25-28%) suggest the method may be slightly harmful on this dataset,
possibly due to SQuAD's extractive nature (ceiling effect — passages already contain
exact answer spans).

### Part 3 Results: Prefix Content × Layer Selectivity

| Prefix | d | Win% | p | sig |
|--------|---|------|---|-----|
| static_fact | **+0.217** | 51.2% | 5.5e-04 | *** |
| random | +0.095 | 52.7% | 0.126 | ns |
| oracle | **+0.230** | 51.9% | 2.6e-04 | *** |

**Verdict: Prefix content DOES matter under layer selectivity.**

Comparison with Exp 16 (full-cache replacement):

| Prefix | Full-cache d | VEL@16 d | Gain |
|--------|-------------|----------|------|
| static_fact | -0.031 (ns) | **+0.217** (***) | +0.248 |
| random | -0.109 (***) | +0.095 (ns) | +0.204 |

Under full-cache replacement (Exp 16), ALL prefix types fail or hurt on Gemma. Under
layer-selective values, static_fact (+0.217) and oracle (+0.230) both work significantly.
Random is positive but non-significant (+0.095).

This means the priming failure on Gemma was never about prefix content — it was about
key interference in the late layers. Once late-layer values are excluded, the
semantic signal emerges: oracle ≈ static_fact > random, matching the Mistral pattern.

### Part 4 Results: Value Feature Analysis

| Metric | r with per-layer d | p |
|--------|--------------------|---|
| delta_norm | -0.060 | 0.737 |
| cosine_sim | -0.249 | 0.156 |

Neither feature shows significant correlation with per-layer d. The physical
properties of value vectors (L2 norms, cosine similarity, perturbation magnitude)
do NOT predict which layers carry useful signal. The mechanism appears to be
content-level rather than geometrically detectable from simple features.

### Key Takeaways

1. **Signal is concentrated, not distributed.** Only 5 of 34 layers matter significantly:
   L10, L12, L14, L15 (hero band), plus L20. These account for virtually all of the
   collective d=+0.227 from cutoff=16.

2. **NOT dataset-general.** SQuAD v2 shows zero benefit even with the optimal Gemma method.
   The priming benefit remains narrowly MS MARCO-specific.

3. **Prefix content matters again once keys are excluded.** The Exp 16 conclusion that
   "prefix content doesn't matter on Gemma" was wrong — it doesn't matter when primed
   keys are poisoning the cache. Under layer-selective values, oracle and static_fact
   both work (d≈+0.22), and random is weaker. The semantic hierarchy is restored.

4. **No simple geometric predictor.** You can't identify beneficial layers from value
   norms or cosine similarities. Layer selection must be empirically determined per model.

### Lessons Learned

1. **Don't iterate dict while deleting.** `for k in d: del d[k]` crashes Python.
   Use `del d` or `d.clear()`.
2. **Layer cherry-picking may beat blanket cutoffs.** The 5-layer hero set is a strong
   candidate for a more targeted approach.
3. **"Method X fails on Gemma" needs qualification.** Full-cache priming fails.
   Layer-selective priming works. The method matters more than the model.

## Exp 22: Ranking Evaluation & Contrastive Scoring (PMI)

**Date started**: 2026-02-15
**Notebook**: `22_ranking_and_pmi.ipynb`
**Results**: `results/exp22/`

### Question

All prior experiments measured *average NLL improvement* — does priming reduce NLL across
passages? But they never tested whether the signal **correlates with document relevance**
for ranking. This experiment asks: does values_early_layers produce lower NLL for relevant
documents than irrelevant ones? And does PMI scoring (subtracting the model's prior)
improve relevance discrimination?

### Motivation (from Exp 19)

Exp 19 established `values_early_layers` (layers 0-16) as the best Gemma mechanism
(d=+0.211 vs bare). But a positive average d doesn't prove ranking utility — if priming
lowers NLL equally for relevant and irrelevant passages, it helps prediction but not
selection.

### Design

| Parameter | Value |
|-----------|-------|
| Model | Gemma 3 4B (4-bit, bfloat16) |
| Method | `values_early_layers` (layers 0-16 of 34) |
| Dataset | MS MARCO v1.1 validation — natural ~8-10 candidate passages per query |
| N | 200 queries, 1,692 passages (221 relevant / 1,471 irrelevant) |

**Three NLL values per passage:**
1. `nll_primed` — values_early_layers cache → score answer
2. `nll_bare` — bare cache (no priming) → score answer
3. `nll_baseline` — BOS-only cache (no document context) → score answer (once per query)

**PMI scoring:** `Score_PMI = NLL(Answer | Query, Document) - NLL(Answer | Query, Empty)`.
More negative = document helps more. Removes "easy answer" bias.

**Four scoring methods compared:** raw bare NLL, raw primed NLL, PMI bare, PMI primed.

### Results

| Method | AUC | MRR@10 | Diff NLL | Cohen's d | p-value |
|---|---|---|---|---|---|
| Raw bare NLL | 0.828 | 0.860 | +1.920 | +1.201 | 1.1e-57 |
| Raw primed NLL | 0.829 | 0.853 | +1.874 | +1.228 | 4.8e-60 |
| **PMI bare** | **0.841** | **0.860** | **+1.963** | **+1.647** | **1.2e-100** |
| PMI primed | 0.832 | 0.853 | +1.918 | +1.588 | 2.0e-94 |

Primed vs Bare MRR (PMI): 6 wins / 185 ties / 9 losses.

### Key Findings

1. **NLL is an excellent document ranker.** All four methods achieve AUC > 0.82 and
   MRR@10 > 0.85 — far above chance. Raw NLL alone, without any priming or contrastive
   scoring, separates relevant from irrelevant passages with Cohen's d > 1.2 (a large
   effect). Relevant passages achieve mean NLL=0.56 vs irrelevant NLL=2.48.

2. **PMI improves discrimination.** Subtracting the baseline (BOS-only NLL) boosts AUC
   from 0.828 to 0.841 and Cohen's d from 1.20 to 1.65. PMI removes the "easy answer"
   confound: some answers are predictable regardless of the document. PMI isolates how
   much the *document specifically* helps predict the answer.

3. **Priming does NOT improve ranking.** values_early_layers provides essentially zero
   ranking benefit. MRR@10 is identical between bare and primed within each scoring type
   (0.860 vs 0.853). Per-query: 6 primed wins, 185 ties, 9 bare wins. Priming slightly
   *hurts* AUC by ~1 point (PMI bare 0.841 vs PMI primed 0.832).

4. **Priming improves NLL uniformly, not differentially.** Exp 19 showed priming reduces
   average NLL (d=+0.211). Exp 22 reveals this improvement is ~equal for relevant and
   irrelevant passages — it doesn't create differential signal. The value contamination is
   a content-agnostic regularization effect, not a relevance-aware one.

5. **Bare PMI is the best scorer.** The simplest method (no priming, just NLL minus
   baseline) achieves the highest AUC (0.841) and tied-highest MRR@10 (0.860). Priming
   adds computational cost (2 extra forward passes per passage) with no ranking benefit.

### Interpretation

The results separate two distinct questions:
- **"Does priming help predict the answer?"** — Yes (Exp 19: d=+0.211).
- **"Does priming help *rank* documents by relevance?"** — No (Exp 22: 0 MRR gain).

Priming helps the model predict answers better from *any* passage, but it doesn't help
distinguish which passage is *relevant*. This makes sense: value contamination from a
static fact prefix (`"What are the key facts I need to know?"`) is query-independent —
it injects the same information regardless of whether the passage matches the query.

For the ad-serving use case, what matters is ranking (which cache to serve), not
absolute NLL (how well to predict the answer). Priming improves the latter but not the
former. The most useful result from this experiment is that **bare NLL itself is a strong
ranker** (AUC=0.83), and **PMI scoring is an easy win** (+1.3 AUC points) that requires
only one extra BOS-only scoring pass per query.

### Lessons Learned

1. **Average NLL improvement ≠ ranking utility.** d=+0.211 on average says nothing about
   whether the improvement is differential across relevant vs irrelevant documents. Always
   test ranking metrics (AUC, MRR) separately from average effects.

2. **PMI is cheap and effective.** One BOS-only forward pass per query (not per document)
   gives a meaningful AUC boost. The baseline computation is amortized across all candidate
   passages.

3. **Raw NLL is already a strong relevance signal.** AUC=0.83 and MRR=0.86 from bare NLL
   alone is competitive. The model inherently assigns lower NLL to relevant passages —
   likely because relevant passages contain answer-overlapping content that reduces
   prediction uncertainty.

4. **Static-fact priming is not query-aware.** The prefix is the same for all documents,
   so any benefit it provides is query-independent. To improve ranking, you'd need
   query-specific priming — but that defeats the purpose of pre-computed caches.

## Exp 23: Multi-Signal Ranking & Query-Time Enhancement

**Date started**: 2026-02-15
**Notebook**: `23_multi_signal_ranking.ipynb`
**Results**: `results/exp23/`

### Question

Exp 22 showed static-fact VEL provides zero ranking benefit. Can we improve ranking with:
(a) query-aware cache modifications, (b) alternative scoring targets, (c) intent-routing
across multiple prefixes, or (d) two-stage pipelines?

### Motivation (from Exp 22)

The static-fact prefix is query-independent — same prefix for all documents, so any benefit
is uniform across relevant and irrelevant passages. Four ideas to break this:
1. **Query-Time Value Injection (QVI)**: Blend mean query-cache values into bare doc cache
2. **Multiple scoring targets**: Score relevance templates or query prediction, not just answers
3. **Intent-specialized prefixes**: 5 static prefixes, route to best per passage or per query
4. **Two-stage pipeline**: Cheap bare PMI rank → expensive oracle VEL re-rank on top-k

### Design

| Parameter | Value |
|-----------|-------|
| Model | Gemma 3 4B (4-bit, bfloat16) |
| Dataset | MS MARCO v1.1 validation |
| N | 200 queries, 1,692 passages (221 relevant / 1,471 irrelevant) |
| Conditions | 13 cache conditions × 3 scoring targets = 39 scores/passage |

**13 cache conditions:**
- `bare`: [BOS][doc] baseline
- 5 intent VELs (`fact_vel`, `def_vel`, `proc_vel`, `quant_vel`, `prob_vel`): static prefix, L0-16
- `oracle_vel`: actual query prefix, L0-16
- `oracle_vel_low`: actual query prefix, L0-8
- `oracle_interp`: interpolate_values(bare, oracle_corrected, α=0.25), all layers
- `oracle_full`: full [BOS][query][doc] cache, no truncation
- 3 QVI caches (`qvi_010`, `qvi_025`, `qvi_050`): v_new = (1-α)·v_doc + α·mean(v_query), L0-16

**3 scoring targets:**
- `answer`: "\nQuery: {q}\nAnswer:" → " {answer}" (standard)
- `qdoc`: "\nThis document is about:" → " {query}" (bidirectional)
- `relevance`: "\nQuery: {q}\nIs this document relevant?" → " Yes, this document is relevant to the query" (~8 tokens)

**Analysis-time derived methods** (no extra compute):
- PMI (score − baseline) for all 39 pairs
- Best-intent routing: min(5 intents) per passage
- Per-query intent routing: pick best intent per query
- Two-stage pipeline: bare PMI rank → oracle_vel re-rank top-k

### Results

**AUC-ROC (PMI, answer target) — primary metric:**

| Condition | AUC (PMI) | vs bare |
|-----------|-----------|---------|
| **oracle_interp** | **0.842** | **+0.001** |
| bare | 0.841 | — |
| oracle_vel_low | 0.841 | +0.000 |
| oracle_full | 0.837 | -0.004 |
| qvi_010 | 0.836 | -0.005 |
| def_vel | 0.835 | -0.006 |
| proc_vel | 0.834 | -0.007 |
| fact_vel | 0.832 | -0.009 |
| oracle_vel | 0.831 | -0.010 |
| quant_vel | 0.830 | -0.011 |
| prob_vel | 0.830 | -0.011 |
| qvi_025 | 0.811 | -0.030 |
| qvi_050 | 0.767 | -0.074 |

**MRR@10 (PMI, answer target):**

| Condition | MRR@10 |
|-----------|--------|
| oracle_vel | 0.865 |
| oracle_vel_low | 0.864 |
| quant_vel | 0.863 |
| proc_vel | 0.862 |
| oracle_interp | 0.861 |
| **bare** | **0.860** |
| prob_vel | 0.859 |
| fact_vel / def_vel | 0.853 |
| oracle_full | 0.850 |
| qvi_010 | 0.847 |
| qvi_025 | 0.840 |
| qvi_050 | 0.835 |

**Alternative targets (PMI AUC):**

| Target | Best condition | AUC | bare AUC |
|--------|---------------|-----|----------|
| answer | oracle_interp | 0.842 | 0.841 |
| qdoc | oracle_interp / bare | 0.574 | 0.574 |
| relevance | qvi_050 | 0.498 | 0.451 |

**Derived methods:**

| Method | AUC (PMI, answer) | MRR@10 |
|--------|-------------------|--------|
| Best-intent routing | 0.825 | 0.855 |
| Per-query intent routing | 0.828 | 0.858 |
| Two-stage (k=3) | — | 0.866 |
| Two-stage (k=5) | — | 0.865 |
| bare PMI (baseline) | 0.841 | 0.860 |

**Head-to-head vs bare (PMI answer MRR):**
- oracle_interp: 2 wins / 197 ties / 1 loss (most stable)
- oracle_vel: 8 wins / 187 ties / 5 losses
- All conditions: overwhelmingly ties (~93-99%)

### Key Findings

1. **No condition meaningfully beats bare for ranking.** The best condition (oracle_interp,
   AUC=0.842) beats bare (0.841) by +0.001 — statistically insignificant. Even with the
   actual query as prefix (oracle conditions), ranking improvement is negligible. The
   13-condition sweep confirms Exp 22's finding is robust.

2. **qdoc and relevance targets are poor rankers.** The `qdoc` target (does doc predict query?)
   achieves AUC ~0.57 — barely above chance. The `relevance` target (explicit relevance
   judgment) is worse at AUC ~0.45-0.50, actually below chance for most conditions. The
   `answer` target dominates. The model's relevance judgment via template scoring does not
   correlate with actual relevance.

3. **QVI hurts, especially at high alpha.** Blending query values into doc cache *degrades*
   ranking. α=0.10 is near-neutral (AUC 0.836), but α=0.50 is catastrophic (AUC 0.767).
   Injecting query information into value vectors corrupts the document representation
   rather than enhancing relevance discrimination.

4. **Intent routing underperforms bare.** Both best-intent routing (min across 5 intents
   per passage, AUC 0.825) and per-query routing (AUC 0.828) are *worse* than bare (0.841).
   Taking the minimum NLL across intents adds noise rather than signal.

5. **Two-stage pipeline shows marginal MRR gain.** Bare PMI rank → oracle VEL re-rank on
   top-3 yields MRR@10=0.866 vs bare 0.860 — a +0.006 improvement. This is the only method
   that shows any positive signal, but it requires a query-specific forward pass per passage
   in the top-k, which is expensive and defeats the pre-computation goal.

6. **oracle_vel_low (L0-8) ≈ oracle_vel (L0-16).** Restricting to fewer layers doesn't help
   or hurt. The value contamination from oracle prefix is irrelevant to ranking regardless
   of layer range.

7. **oracle_full slightly hurts.** Full context (query visible to attention during scoring)
   achieves AUC 0.837, slightly below bare 0.841. This is surprising: having the query
   explicitly present in context doesn't help ranking. The query is already provided in the
   scoring prompt; duplicating it in the cache adds no information.

### Interpretation

This experiment exhaustively tested whether query-aware cache modifications can improve
document ranking. The answer is definitively **no** — across 13 cache conditions, 3 scoring
targets, and 4 derived methods, nothing meaningfully beats bare PMI.

The fundamental issue: **NLL-based ranking already captures relevance well** (AUC=0.84). The
model assigns lower NLL to relevant passages because they contain answer-overlapping tokens.
This signal comes from the document content itself, not from any prefix or cache modification.
Priming (whether static or query-aware) modifies value representations but doesn't change
which tokens overlap with the answer — so it can't improve relevance discrimination.

The one exception is the two-stage pipeline (+0.006 MRR), which works by combining two
independent ranking signals (bare PMI + oracle VEL PMI). But this requires per-passage
oracle forward passes at query time, making it impractical for the ad-serving use case.

**For the ad-serving use case**: bare PMI scoring (AUC=0.841, MRR=0.860) remains the best
approach. It requires only one BOS-only pass per query (amortized across all candidates)
and zero modifications to document caches. No form of cache priming improves ranking.

### Lessons Learned

1. **Query-aware priming doesn't help ranking either.** Exp 22 showed static priming fails;
   Exp 23 shows even oracle (actual query) priming fails. The issue isn't that the prefix
   is wrong — it's that value contamination fundamentally doesn't create ranking signal.

2. **Alternative scoring targets don't help.** The `answer` target is by far the best ranker.
   Bidirectional scoring (doc predicts query) and explicit relevance templates perform near
   or below chance. The model's relevance judgment doesn't correlate with actual relevance.

3. **Value injection is destructive.** QVI (blending query values into doc cache) actively
   corrupts the representation. The mean query value is a poor summary of query semantics
   in value space — it overwrites document-specific information without adding useful signal.

4. **Routing across intents adds noise.** Min-across-intents selects the intent that happens
   to produce the lowest NLL, which is not necessarily the most discriminative. The selection
   is driven by passage characteristics (which intent fits the topic) rather than relevance.

5. **Two-stage re-ranking is the only positive signal.** Combining two independent scoring
   functions (bare + oracle) via re-ranking is more promising than modifying a single cache.
   Future work on ranking should explore score combination rather than cache modification.
