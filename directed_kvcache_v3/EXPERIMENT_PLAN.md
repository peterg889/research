# Experiment Plan: Directed KV Cache v3 (T5Gemma Encoder-Decoder)

## Guiding Principle

Every experiment measures the same three points on the spectrum:

| Label | What happens | Cost at serving time |
|-------|-------------|---------------------|
| **Upper bound** | `encode([real_query + document])`, truncate query → decoder | O(Q x D) encoder passes |
| **Lower bound** | `encode([document])` → decoder | O(1) encoder passes (pre-cached) |
| **Middle ground** | `encode([surrogate + document])`, truncate surrogate → decoder | O(1) encoder passes (pre-cached) |

The question is always: **how much of the upper-lower gap does the surrogate close?**

All conditions use truncation (mask surrogate tokens from decoder cross-attention),
since Exp 01 proved this is strictly better than full visibility.

---

## Datasets

### Primary: MS MARCO v1.1 (Exps 01-03)
- **Why**: Proven to work, has answer text for NLL scoring, short passages (~130 tokens),
  no ceiling effects on T5Gemma (bare mean NLL = 3.72, plenty of headroom).
- **Load**: `load_dataset("microsoft/ms_marco", "v1.1", split="validation")`
- **Triple**: (query, passage, answer) — all present in dataset.
- **Ranking pools**: ~8 candidate passages per query, 1 relevant. Config `v2.1` for ranking.
- **Role in v3**: Controlled baseline for Exps 02 (surrogate types) and 03 (length scaling).

### Commercial: Amazon ESCI (Exp 04)
- **Why**: Real product search with graded relevance (Exact/Substitute/Complement/Irrelevant).
  130K queries, 1.8M products, up to 40 candidates per query. The ad-serving use case.
- **Load**: `load_dataset("tasksource/esci")`, filter to `product_locale == 'us'`.
- **Document**: product_title + product_bullet_point + product_description (concatenated).
  Median ~80-150 words when description is present; many products are title-only (~15 words).
- **Scoring adaptation**: No gold answer text. Use query-likelihood scoring instead:
  `NLL(query | encode([surrogate + product_text]))`. The question: does the primed product
  representation make the user's query more predictable? This IS the ad-serving question —
  "which product best explains this query?"
- **Relevance**: E+S = relevant, C+I = irrelevant (or use all 4 levels for NDCG).
- **v2 history**: Attempted in Exp 32C but all 3 HuggingFace sources failed to load.
  `tasksource/esci` is the most reliable mirror (pre-joined, 2.03M train rows).

### Commercial: WANDS — Wayfair (Exp 04)
- **Why**: Different commercial domain (furniture/home goods), richer prose descriptions
  than ESCI, 3-class graded labels (Exact/Partial/Irrelevant). Cross-domain validation.
- **Load**: `load_dataset("napsternxg/wands")`.
- **Document**: product_name + product_description + product_features. Typically 100-400 words.
- **Scoring**: Same query-likelihood approach as ESCI.
- **Limitation**: Only 480 queries (but 233K relevance judgments). Small but dense.

### Commercial: Amazon Product QA — hetPQA (Exp 04, optional)
- **Why**: The ONLY product dataset with gold answer text. 7.5K questions about Amazon
  products (toys domain) with human-written answers + evidence from 6 source types
  (descriptions, bullets, reviews, community QA, editorial, attributes).
- **Load**: GitHub download from `amazon-science/contextual-product-qa`.
- **Scoring**: Standard NLL(answer | encode([surrogate + product_text])). Same as MS MARCO.
- **Value**: Bridges the gap between MS MARCO (has answers, not commercial) and
  ESCI (commercial, no answers). Tests whether surrogate priming helps a model
  generate product answers.

### Ceiling-Effect Pre-Screening Protocol
**Learned from v2 the hard way** (TriviaQA 77% ceiling, BoolQ 100%, CoQA 65%):

Before committing to any dataset, run a 20-sample pre-screen:
```python
# Pre-screen: check for ceiling effects
bare_nlls = [score_nll(doc, answer) for doc, answer in random.sample(samples, 20)]
pct_floor = sum(1 for n in bare_nlls if n < 0.05) / len(bare_nlls)
if pct_floor > 0.3:
    print(f"WARNING: {pct_floor:.0%} at floor — ceiling effect likely, skip dataset")
```
Only proceed if <30% of samples have near-zero bare NLL. This catches cases where
the model already knows the answer from parametric memory.

---

## Completed

### Exp 01 — Truncation Test (DONE)
**Question**: Is the benefit from the decoder reading surrogate tokens directly, or from
improved document representations via bidirectional co-encoding?

**Result**: Truncation makes it STRONGER. Document representations are genuinely improved.
Oracle_trunc d=+0.408, surr_doc_trunc d=+0.363 (89% of oracle). N=200.

### Exp 02 — Surrogate Type Sweep (DONE)
**Question**: What is the best surrogate to prime the encoder with?

**Result**: Mechanism is 85% structural, 10% semantic. Random prefix captures 81% of oracle.
Best doc-derived: surr_template ("What is [keyword]?") at 90% of oracle. N=500.

### Exp 2B — Structural vs Semantic Mechanism Decomposition (DONE)
**Question**: Is the benefit structural or semantic? Same as v2's value contamination?

**Result**: Binary switch — 1 random word (2.5 tokens) captures 85% of oracle. "the" x10
beats diverse random. 85% structure / 6% vocabulary (ns) / 10% semantics (sig). N=500.

### Exp 03 — Length Scaling (DONE)
**Question**: Does the surrogate benefit survive longer documents?

**Result**: NO DECAY. All conditions significant (p<1e-07) at every length up to 2048 tokens.
Oracle d=+0.38 to +0.45, surr_doc d=+0.33 to +0.41. Complete reversal of v2's cliff at ~200 tokens.
N=400, 6 length bins, Bonferroni-corrected (18 comparisons). See Results Log for full table.

---

## Planned Experiments

### Exp 2B — Structural vs Semantic Mechanism Decomposition
**Question**: Is the co-encoding benefit structural (any prefix helps) or semantic
(content-specific surrogates are genuinely better)? Is this the same implicit
regularization as v2's value contamination, or qualitatively different?

**Dataset**: MS MARCO v1.1 (same 500 samples as Exp 02, enables direct comparison).

**Why this is needed**: Exp 02 found no content gradient (Spearman rho=-0.167) yet
pairwise comparisons showed doc-specific surrogates DO beat random per-sample. Static
surrogates had higher Cohen's d despite lower absolute NLL improvement. These contradictions
need resolution before we can interpret Exps 03-04 correctly.

**Design** (4 parts, 10 new conditions, all with truncation, N=500):

Part 1 — Re-analysis of Exp 02 data (no new scoring):
- Document length stratification (does semantic gap change with doc length?)
- Hardness stratification (does semantic advantage emerge for hard samples?)
- Variance decomposition (explains the Cohen's d paradox)

Part 2 — Prefix length titration (random words: 1, 3, 5, 10, 20, 50):
- Saturation curve: does 1 word suffice (switch) or do we need many (gradual)?
- Distinguishes attention redistribution from regularization

Part 3 — Content ablation (all length-matched to oracle):
- bare → random_matched: STRUCTURE (any prefix helps)
- random_matched → scrambled_oracle: VOCABULARY (right words, wrong order)
- scrambled_oracle → oracle: SEMANTICS (right word order)
- Clean three-way decomposition of the total benefit

Part 4 — Token diversity (all ~10 words):
- "the" x10 vs doc_keyword x10 vs diverse random words
- Tests whether having DIFFERENT tokens matters or repetition suffices

**Key predictions**:
- If 1 random word ≈ 50 random words → switch mechanism (like v2)
- If structure > 70% of benefit → primarily structural (like v2 but different path)
- If repeat_the ≈ diverse_random → pure structural, token identity irrelevant
- If vocabulary or semantics contribute > 20% → different from v2

**Builds on**: Exp 02 (same samples, loads checkpoint)
**Informs**: Interpretation of Exps 03-04, whether LLM surrogates (Exp 05) are worth pursuing

---

### Exp 03 — Length Scaling (DONE)
**Question**: Does the surrogate benefit survive longer documents?

**Dataset**: MS MARCO v1.1 with controlled padding (same approach as v2 Exp 20).

**Result**: NO DECAY. Benefit is rock-solid at all lengths up to 2048 tokens.
Complete reversal of v2's cliff at ~200 tokens. See Results Log below.

---

### Exp 04 — Ranking on Commercial Datasets
**Question**: Can surrogate-primed encoder representations improve document ranking?
Does this generalize from MS MARCO to real product search?

**Datasets**: MS MARCO v2.1 (ranking pools) + Amazon ESCI + WANDS (Wayfair).

**Why this is the big one**: Ranking NEVER worked in v2. Across 6 experiments
(Exps 14, 15, 22, 23, 28, 31), no cache modification improved ranking. The reason was clear:
in decoder-only models, value contamination is document-independent — it lowers NLL equally
for relevant and irrelevant passages, providing zero differential signal.

T5Gemma's mechanism is fundamentally different. Bidirectional co-encoding creates
document-SPECIFIC representations. A surrogate like "best family hotels in Bahamas" will
reshape the document representation of a Bahamas hotel listing differently than it reshapes
a car insurance document. This document-specificity could create the differential signal
that v2 never achieved.

**Part A — MS MARCO Ranking** (comparison with v2):

Score = NLL(answer | encoder_output), same as Exps 01-03.

| # | Condition | Encoder input | Decoder scores |
|---|-----------|--------------|----------------|
| 1 | bare | document only | answer |
| 2 | oracle_trunc | real query + document (truncated) | answer |
| 3 | best_surr_trunc | best surrogate + document (truncated) | answer |

N=200 queries (~1600 passages). Metrics: AUC, MRR@10.
Compare directly to v2 Exp 22 (Gemma bare AUC=0.828, MRR=0.860, priming AUC=0.828).

**Part B — Amazon ESCI Ranking** (the ad-serving use case):

Scoring adaptation for product search (no gold answers):
- Score = NLL(query | encode([surrogate + product_text]))
- "Given this product representation, how likely is the user's query?"
- This is exactly the ad-serving question: which product best matches this query?

| # | Condition | Encoder input | Decoder scores |
|---|-----------|--------------|----------------|
| 1 | bare | product_text only | query |
| 2 | oracle_trunc | real query + product_text (truncated) | query |
| 3 | best_surr_trunc | best surrogate + product_text (truncated) | query |
| 4 | surr_title_trunc | product_title + product_text (truncated) | query |

The `surr_title` condition is natural for ads: the product title is always available
and serves as a document-derived surrogate (like doc_keywords for MS MARCO).

N=200 queries, filter to US locale, groups with 4+ candidates (at least 1 E/S, 1 C/I).
Metrics: NDCG@10 (graded: E=3, S=2, C=1, I=0), AUC (binary: E+S vs C+I), MRR@10.

**Why ESCI might work where MS MARCO didn't for query-likelihood**:
v2 Exp 31 showed QL barely above chance (AUC=0.578) on MS MARCO because all ~8 candidates
were retrieved for the SAME query — they're all topically similar. ESCI's candidate pools
include truly Irrelevant products (10% of pairs), providing more ranking headroom.

**Part C — WANDS Cross-Domain Validation** (optional, if B shows positive signal):

Same query-likelihood approach on Wayfair product search.
480 queries, 3-class labels (Exact/Partial/Irrelevant).
Tests whether the ranking benefit (if any) transfers across commercial domains.

**The critical test across all parts**: Does surrogate priming lower NLL MORE for relevant
documents than for irrelevant ones? Compute per-query:
- `delta_relevant = NLL_bare - NLL_primed` for relevant document(s)
- `delta_irrelevant = mean(NLL_bare - NLL_primed)` for irrelevant documents
- If `delta_relevant > delta_irrelevant` consistently, the surrogate creates differential signal

**Builds on**: Exp 02 (best surrogate), Exp 03 (length constraints)
**Informs**: Whether this approach has practical deployment value for ad ranking

---

## Future Experiments (after Exps 02-04)

### Exp 05 (tentative) — LLM-Generated Surrogate Queries
**Question**: Can an LLM generate better surrogates than heuristic extraction?

Exp 02 tests simple surrogates (keywords, templates, static phrases). These are cheap
but may not capture the full query intent. An LLM could generate a natural-language
query like "What are the best family-friendly hotels in the Bahamas?" from a product
description, potentially producing a much richer surrogate.

**Why not in Exp 02**: T5Gemma is pretrained-only (not instruction-tuned), so it cannot
generate queries on command. Generating surrogates requires a separate instruction-tuned
model (e.g., Gemma 2 9B-IT, Llama 3, or an API call). This adds infrastructure
complexity and GPU memory pressure that should be isolated from the controlled
surrogate comparison in Exp 02.

**Approach**:
- Pre-generate surrogate queries offline using an instruction-tuned LLM
- Prompt: "Given this product description, write a search query someone might use to find it: [doc]"
- Generate 1-3 surrogates per document, test each
- Compare to Exp 02's best heuristic surrogate
- Test both generic prompts and domain-specific prompts (e.g., "Write a product search query...")

**Key question**: How much of the oracle-heuristic gap can LLM surrogates close?
If Exp 02 shows doc_keywords at 85% of oracle, and LLM surrogates reach 95%,
the 10% uplift may not justify the generation cost. But if heuristics plateau at 60%
and LLM surrogates reach 90%, the case is strong.

**Depends on**: Exp 02 results (establishes the heuristic baseline to beat)

---

## Experiment Dependency Graph

```
Exp 01 (Truncation — DONE)
  │
  ▼
Exp 02 (Surrogate Types — DONE) ── best doc-derived: surr_template (90% oracle)
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
Exp 2B (Mechanism Decomposition)  Exp 03 (Length Scaling) [can run in parallel]
  │                                 │
  ▼                                 ▼
  Interpretation of structural    Exp 04 (Ranking: MARCO + ESCI + WANDS)
  vs semantic mechanism             ├─ Part A: MS MARCO
  │                                 ├─ Part B: Amazon ESCI
  ▼                                 └─ Part C: WANDS
Exp 05 (LLM Surrogates — only if
  vocab/semantics contribute >20%)
```

---

## Dataset Loading Reference

```python
# MS MARCO v1.1 (QA with answers)
msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
# Fields: query, passages.passage_text, passages.is_selected, answers

# MS MARCO v2.1 (ranking pools)
msmarco_rank = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
# Fields: query, passages.passage_text, passages.is_selected

# Amazon ESCI
esci = load_dataset("tasksource/esci", split="train")
esci_us = esci.filter(lambda x: x['product_locale'] == 'us')
# Fields: query, product_title, product_description, product_bullet_point,
#          product_text, esci_label (E/S/C/I)

# WANDS (Wayfair)
wands = load_dataset("napsternxg/wands")
# Fields: query, product_name, product_description, label (Exact/Partial/Irrelevant)

# Amazon Product QA (hetPQA) — GitHub download
# git clone https://github.com/amazon-science/contextual-product-qa
# Fields: question, answer, evidence (6 source types)
```

---

## Design Principles (Inherited from v2 Lessons)

1. **Always report the three-point spectrum**: upper bound (oracle), lower bound (bare),
   middle ground (surrogate). Every table should make it obvious where we are.

2. **Always truncate**: Exp 01 proved truncation is strictly better. All conditions
   mask surrogate tokens from decoder cross-attention.

3. **Statistical rigor**: Cohen's d, win%, AND p-value. Bonferroni for multiple comparisons.
   **N=500 minimum** (we have a 40 GB GPU and prefer quality over speed). This gives us
   power to detect d=0.13 in pairwise comparisons. Checkpoint every 20 samples for resume.

4. **Content controls**: Always include a content-agnostic control (random or static)
   to distinguish semantic from structural effects.

5. **Same evaluation framework**: NLL(answer | encoder_output) for QA datasets.
   NLL(query | encoder_output) for product search datasets (query-likelihood).
   No query in decoder for answer-likelihood; no answer in decoder for query-likelihood.

6. **Pre-screen for ceiling effects**: 20-sample bare NLL check before committing
   to any new dataset. Reject if >30% of samples have near-zero NLL.

7. **Document length awareness**: Filter or note document lengths. Exp 03 results
   determine the safe operating envelope. For ESCI/WANDS, report results stratified
   by product text length.

---

## Datasets NOT Used (and Why)

From v2 experience, these datasets are unsuitable for our method:

| Dataset | Problem | v2 Evidence |
|---------|---------|-------------|
| TriviaQA | 77% ceiling — model knows answers from training | v2 Exp 27 |
| BoolQ | 100% ceiling — binary Yes/No, trivial for model | v2 Exp 30 |
| CoQA | 65% ceiling | v2 Exp 29 |
| AdversarialQA | 72% ceiling | v2 Exp 29 |
| SQuAD v2 | Ceiling effect | v1 Exp 19 |
| HotpotQA | Multi-hop reasoning disrupted by priming | v2 Exp 27 (d=-0.35) |
| DROP | Numerical computation disrupted by priming | v2 Exp 30 (d=-0.198) |
| CNN/DailyMail | Summarization — catastrophic harm | v1 Exp 19 (d=-1.31) |
| PubMedQA | Specialized domain, priming hurts | v1 Exp 19 (d=-0.73) |
| NarrativeQA | Long-form, priming hurts | v1 Exp 19 (d=-0.35) |

**Note**: These exclusions are based on decoder-only models. T5Gemma's bidirectional
encoder may behave differently, but we prioritize datasets with the highest probability
of informative results. If Exps 02-04 show strong results, a targeted cross-dataset
test (e.g., NQ, which showed d=+0.213 on Gemma hero layers) could be a future experiment.

---

## Results Log

Results will be appended here as experiments complete.

### Exp 01 — Truncation Test
**Status**: COMPLETE | **Date**: 2026-02-17 | **N**: 200 | **Dataset**: MS MARCO v1.1

| Condition | d vs bare | Win% | p-value | % of oracle gap |
|-----------|-----------|------|---------|----------------|
| oracle_trunc | +0.408 | 94% | 3e-08 | 100% (upper bound) |
| surr_doc_trunc | +0.363 | 85% | 7e-07 | 89% |
| surr_para_trunc | +0.357 | 88% | 1e-06 | 87% |
| oracle_full | +0.345 | 82% | 2e-06 | 85% |
| surr_doc_full | +0.312 | 80% | 2e-05 | 77% |
| surr_para_full | +0.293 | 72% | 5e-05 | 72% |

**Conclusion**: Truncation is strictly better. Document representations are genuinely
improved by bidirectional co-encoding. Doc keywords capture 89% of oracle benefit.

### Exp 02 — Surrogate Type Sweep
**Status**: COMPLETE | **Date**: 2026-02-17 | **N**: 500 | **Dataset**: MS MARCO v1.1

**Question**: What is the best surrogate type? Does content matter (semantic) or is any prefix enough (structural)?

**Conditions**: 9 total (1 bare + 1 oracle + 7 surrogates), all with truncation.

| Rank | Condition | Mean NLL | Delta | d vs bare | Win% | p-value | % Oracle |
|------|-----------|----------|-------|-----------|------|---------|----------|
| — | bare | 3.6765 | — | — | — | — | 0% (LB) |
| 1 | oracle_trunc | 2.9929 | +0.684 | +0.376 | 92.6% | 4.8e-16 | 100% (UB) |
| 2 | static_fact_trunc | 3.2584 | +0.418 | +0.372 | 83.8% | 8.7e-16 | 99% |
| 3 | static_howto_trunc | 3.2280 | +0.448 | +0.346 | 85.6% | 6.0e-14 | 92% |
| 4 | surr_template_trunc | 3.1173 | +0.559 | +0.336 | 90.8% | 2.6e-13 | 90% |
| 5 | surr_doc_trunc | 3.0560 | +0.620 | +0.322 | 87.4% | 2.1e-12 | 86% |
| 6 | surr_para_trunc | 3.0870 | +0.589 | +0.305 | 89.2% | 2.8e-11 | 81% |
| 7 | random_trunc | 3.1432 | +0.533 | +0.303 | 87.6% | 3.6e-11 | 81% |
| 8 | surr_lead_trunc | 3.4672 | +0.209 | +0.151 | 64.2% | 7.9e-04 | 40% |

All conditions significant after Bonferroni correction (threshold p < 0.0063).

**Content gradient**: Spearman rho = -0.167 (p = 0.693). **NO content gradient**, same as v2.
Semantic relevance of the surrogate does NOT predict effect size (Cohen's d).

**However — pairwise head-to-head comparisons tell a nuanced story**:

| Comparison | d | Win% | p | Sig |
|------------|---|------|---|-----|
| Oracle vs surr_doc | +0.080 | 63.6% | 0.074 | ns |
| Oracle vs surr_para | +0.181 | 64.2% | 6.1e-05 | *** |
| surr_doc vs static_fact | +0.169 | 59.2% | 1.7e-04 | *** |
| surr_doc vs random | +0.130 | 54.6% | 3.9e-03 | ** |
| static_fact vs random | -0.129 | 43.6% | 4.0e-03 | ** |
| surr_lead vs surr_template | -0.427 | 32.6% | 5.5e-20 | *** |

The Cohen's d ranking and pairwise ranking diverge because of **variance differences**:
- Static surrogates have SMALLER absolute NLL improvements but LOWER variance (more consistent)
- Doc-specific surrogates have LARGER absolute improvements but HIGHER variance
- Pairwise: surr_doc beats both static_fact (d=+0.169***) and random (d=+0.130**) on a per-sample basis

**Group analysis**:
- Content-specific avg d: +0.270 (71.9% oracle) — but includes weak surr_lead
- Content-agnostic avg d: +0.340 (90.6% oracle) — more consistent
- Group difference: d=+0.121 (p=0.007), content-specific has lower NLL per-sample

**Hardness**: All conditions strongly correlated with hardness (r > +0.80). Benefit
scales with bare NLL: hard samples benefit more. Random shows r=+0.849 — same as oracle.

**Best document-derived surrogate**: surr_template_trunc ("What is [top_keyword]?")
d=+0.336 (90% of oracle). Cheapest to compute, requires only word frequency.

**surr_lead anomaly**: First sentence of document gets only 40% of oracle despite being
the most semantically rich doc-derived surrogate. Possibly because it overlaps too much
with the document itself (redundant information in bidirectional attention).

**Conclusions**:
1. The mechanism is primarily structural, not semantic — same as v2, even with bidirectional attention
2. BUT there IS a per-sample semantic advantage (pairwise surr_doc > random, p=0.004)
3. The "structural floor" (random) captures 81% of oracle — most of the benefit comes for free
4. For deployment: surr_template ("What is [keyword]?") is the practical winner — 90% of oracle, trivial to compute
5. LLM surrogates (Exp 05) may not be worth the cost: heuristics already capture 90% of oracle

### Exp 2B — Structural vs Semantic Mechanism Decomposition
**Status**: COMPLETE | **Date**: 2026-02-17 | **N**: 500 | **Dataset**: MS MARCO v1.1

**Question**: Is the co-encoding benefit structural (any prefix helps) or semantic
(content matters)? Is it the same implicit regularization as v2?

**Part 1: Re-analysis of Exp 02 data**
- Document length: semantic gap is STABLE across lengths (r=+0.056, p=0.208)
- Hardness: semantic gap grows Q1(+0.013) → Q5(+0.397) — content matters more for hard samples
- Variance: all conditions share r>0.88 benefit correlation — dominated by shared structural mechanism
- Oracle beats random only 70.6% of the time (63% for easy, 79% for hard)

**Part 2: Prefix length titration**

| Prefix words | ~Tokens | d | % of oracle |
|-------------|---------|------|-------------|
| 1 | 2.5 | +0.321 | 85% |
| 3 | 5.3 | +0.308 | 82% |
| 5 | 7.9 | +0.298 | 79% |
| 10 | 14.5 | +0.279 | 74% |
| 20 | 27.7 | +0.303 | 81% |
| 50 | 65.1 | +0.296 | 79% |

**SWITCH MECHANISM**: 1 random word (2.5 tokens) captures 85% of oracle. Curve is flat.

**Part 3: Content ablation** (length-matched to oracle)

| Component | NLL delta | % of total | d | sig |
|-----------|----------|-----------|------|-----|
| Structure (bare → random_matched) | +0.579 | **84.7%** | +0.296 | *** |
| Vocabulary (→ scrambled oracle) | +0.038 | 5.5% | +0.045 | ns |
| Semantics (→ oracle) | +0.067 | 9.7% | +0.152 | *** |

**Part 4: Token diversity** (~10 words each)

| Condition | d | % oracle |
|-----------|------|----------|
| "the" x10 | +0.338 | 90% |
| doc_keyword x10 | +0.332 | 88% |
| diverse random 10w | +0.279 | 74% |

Uniform repetition BEATS diverse random. Token identity barely matters.

**Conclusions**:
1. **85% of the headroom is pure structure** — any prefix triggers a mode shift
2. Vocabulary contributes 6% (not significant); semantics contributes 10% (significant but small)
3. The mechanism is a **binary switch**: 1 random word suffices (85% of oracle)
4. Uniform tokens ("the" x10) outperform diverse random words
5. Semantics matters slightly MORE for hard samples (Q5 gap = +0.397)
6. The mechanism is NOT v2's value contamination (truncation improves it) — it is
   structural representation enrichment through bidirectional attention
7. For deployment: prepend ANY short prefix. Content barely matters.
   The 10% semantic uplift may not justify surrogate generation cost.

### Exp 03 — Length Scaling
**Status**: COMPLETE | **Date**: 2026-02-17 | **N**: 400 | **Dataset**: MS MARCO v1.1 (padded)

| Length | oracle_trunc d | surr_doc d | surr_para d | v2 oracle d |
|--------|---------------|-----------|------------|-------------|
| original (~130 tok) | +0.384*** | +0.340*** | +0.334*** | +0.303*** |
| 256 tok | +0.435*** | +0.413*** | +0.369*** | +0.114 (ns) |
| 384 tok | +0.447*** | +0.363*** | +0.324*** | N/A |
| 512 tok | +0.442*** | +0.368*** | +0.349*** | +0.034 (ns) |
| 1024 tok | +0.452*** | +0.343*** | +0.280*** | -0.043 (ns) |
| 2048 tok | +0.392*** | +0.333*** | +0.365*** | N/A |

All p-values < 1e-07 (Bonferroni-corrected for 18 comparisons: 3 conditions × 6 lengths).

**Conclusion**: NO DECAY. The benefit is rock-solid at all lengths up to 2048 tokens —
a complete reversal of v2's cliff at ~200 tokens. Oracle effect actually *increases* slightly
with length (d=+0.384 → +0.452 at 1024). Bidirectional co-encoding distributes surrogate
influence uniformly via global self-attention, unlike causal forward-only propagation.
This removes the #1 practical limitation from v2 and confirms that ESCI product descriptions
(100-500 words) are well within the operating envelope for Exp 04.
