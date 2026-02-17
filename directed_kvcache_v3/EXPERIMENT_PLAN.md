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

---

## Planned Experiments

### Exp 02 — Surrogate Type Sweep
**Question**: What is the best surrogate to prime the encoder with?

**Dataset**: MS MARCO v1.1 (controlled — isolate the surrogate variable, same dataset as Exp 01).

**Why this is next**: Exp 01 tested only 2 surrogate types (paraphrased query, doc keywords).
v2 showed surprising results — static phrases beat LLM-generated surrogates 2x on Mistral
(Exp 07, d=+0.472 vs d=+0.234). The mechanism in T5Gemma is fundamentally different
(bidirectional co-encoding, not value contamination), so the surrogate hierarchy may change.
We need to map this landscape before investing in length/ranking experiments.

**Conditions** (all with truncation, N=200):

| # | Condition | Source | v2 analog |
|---|-----------|--------|-----------|
| 1 | bare | — (lower bound) | baseline |
| 2 | oracle_trunc | Real query (upper bound) | Exp 01 |
| 3 | surr_doc_trunc | Top-5 TF-IDF keywords from document | Exp 01 |
| 4 | surr_para_trunc | Query words reversed (paraphrase proxy) | Exp 01 |
| 5 | static_fact_trunc | "What are the key facts?" | v2 Exp 07 (best for Mistral) |
| 6 | static_howto_trunc | "How do I do this?" | v2 Exp 07 |
| 7 | surr_llm_trunc | LLM-generated query about the document | v2 Exp 06 |
| 8 | random_trunc | Random unrelated sentence | v2 Exp 01 (structural control) |
| 9 | category_trunc | Broad topic label from document (e.g., "health", "travel") | New |

**Key comparisons**:
- Content-specific (doc_kw, para, llm) vs content-agnostic (static, random, category)
- Document-derived (doc_kw, category) vs external (llm, static) — matters for deployment cost
- How much of oracle gap does each close?
- Does T5Gemma show a genuine CONTENT GRADIENT (more relevant → more helpful)?
  v2 found NO content gradient with Mistral (Exp 10, Spearman r=+0.036). If T5Gemma shows
  a monotonic gradient, this confirms the bidirectional mechanism is qualitatively different.

**What we expect**: Unlike v2 where the mechanism was value contamination (and content
barely mattered), T5Gemma's bidirectional attention should create a genuine content gradient —
more relevant surrogates should produce better document representations. If static_fact
still wins, that would suggest the encoder is doing something structural rather than semantic.

**Builds on**: Exp 01 (inherits scoring infrastructure, truncation approach)
**Informs**: Exps 03-04 (which surrogate to use for length/ranking tests)

---

### Exp 03 — Length Scaling
**Question**: Does the surrogate benefit survive longer documents?

**Dataset**: MS MARCO v1.1 with controlled padding (same approach as v2 Exp 20).

**Why this matters**: The #1 limitation from v2. Value contamination in decoder-only models
diluted at ~200 tokens (v2 Exp 20). This was a step function, not gradual decay.
T5Gemma's bidirectional encoder is qualitatively different: every document token attends to
the surrogate in BOTH directions, not just via causal forward propagation. The surrogate
influence should be distributed more uniformly. If this experiment shows the benefit persists
at 512+ tokens, it dramatically expands the practical operating envelope — and is critical
for Exp 04, where ESCI product descriptions can be 100-500 words.

**Method**: Same padding approach as v2 Exp 20. Take the same MS MARCO passages
(median ~130 tokens) and pad them to controlled lengths with unrelated MS MARCO text.
Same questions, same answers — only the document length changes.

**Conditions** (all with truncation, using best surrogate from Exp 02, N=200):

| # | Document length | What changes | v2 Exp 20 result |
|---|----------------|-------------|------------------|
| 1 | Original (~130 tok) | Replication of Exp 01/02 | d=+0.303*** |
| 2 | 256 tokens | Padded | d=+0.114 (ns) |
| 3 | 512 tokens | 4x original | d=+0.034 (ns) |
| 4 | 1024 tokens | 8x original | d=-0.043 (ns) |
| 5 | 2048 tokens | Approaching T5Gemma encoder limits | not tested in v2 |

Each length tested with: bare, oracle_trunc, best_surrogate_trunc.

**Key metric**: At what length does d drop below significance? Compare the decay curve
to v2's cliff at ~200 tokens. Plot the decay curves side-by-side.

**What we expect**: The bidirectional encoder should show a more gradual decay (vs v2's
step function). The surrogate's influence is propagated through global self-attention, not
just causal forward flow. Even if the effect diminishes with length, the critical question
is whether 512-token documents still show meaningful benefit — that's the range where
ESCI product descriptions live.

**Builds on**: Exp 02 (best surrogate type)
**Informs**: Exp 04 (practical document length constraints for ranking)

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

## Experiment Dependency Graph

```
Exp 01 (Truncation — DONE)
  │
  ▼
Exp 02 (Surrogate Types, MS MARCO) ── determines best surrogate
  │
  ▼
Exp 03 (Length Scaling, MS MARCO) ──── determines operating envelope
  │
  ▼
Exp 04 (Ranking: MS MARCO + ESCI + WANDS) ── the payoff
  ├─ Part A: MS MARCO ranking (compare to v2)
  ├─ Part B: Amazon ESCI (ad serving)
  └─ Part C: WANDS (cross-domain, conditional)
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
   N=200 minimum. Checkpoint every 20 samples for resume.

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
