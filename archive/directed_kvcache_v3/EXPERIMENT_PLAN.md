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

### Exp 03B — Extended Length Scaling (DONE)
**Question**: Does the benefit survive to 6144 tokens? Does the three-way decomposition hold at all lengths?

**Result**: NO DECAY even at 6144 tokens. All 8 conditions *** at all 7 lengths (Bonferroni 49).
Three-way decomposition (structure/vocabulary/semantics) holds at every length.
Encoder sliding window (1024 tokens) does NOT degrade the effect. See Results Log for full table.

### Exp 3D — Cross-Dataset Content Ablation (DONE)
**Question**: Does the 85% structural finding hold with longer queries on a different dataset?

**Result**: YES — Structure = 84.3% (vs 84.7% MS MARCO). Near-perfect replication.
But the semantic component flipped: vocabulary grew (5.5%→19.9%), semantics went
NEGATIVE (-4.2%). All surrogates beat oracle (150%+ of oracle d). The real query
creates semantic interference on this dataset. N=500.

### Exp 3E — Attention Mechanism Probing (DONE)
**Question**: WHY does prepending any text improve encoder representations?

**Result**: Attention sink redistribution. Bare encoder has a degenerate attention
pattern (position 0 absorbs 56-143x average attention). Any prefix absorbs the sink
role, freeing doc-doc attention to reorganize. See Results Log for full probe data.
N=500.

### Exp 3F — Semantic Amplification (DONE)
**Question**: Can repeating the prefix amplify the semantic component beyond the ~10% baseline?

**Result**: MODERATE amplification (1.7x). Semantic fraction grew 15% → 26% as repetition
increased from x1 to x20, while structural effect saturated (x20/x1 = 1.12). The growth
is mostly vocabulary exposure, not word-order semantics. Stripping stop words hurts (semantic
ratio drops 2.4x). Hardness interaction dramatic: Q5 semantic fraction triples (8% → 23%).
N=500.

### Exp 05 — LLM-Generated Surrogate Queries (DONE)
**Question**: Can an LLM generate better surrogates than the "What is [keyword]?" heuristic?

**Result**: NO. LLM surrogates provide no meaningful uplift over the template heuristic
(d uplift: -0.003 at x1, +0.021 at x4, both ns). The mechanism is 87% structural — any
short prefix works. "Need"-focused prompt is best among LLM variants; stop-word hypothesis
confirmed (need > keywords, p<0.001); semantic interference confirmed (need > question,
p<0.001). Surrogate generation has negative ROI. N=500.

### Exp 06 — Factoid Subsample Validation (DONE)
**Question**: Does filtering to short factoid answers (≤5 words) shift the decomposition
from ~85% structural toward a more semantic-dominated regime?

**Result**: PARTIAL — structural dropped from 85% to 76%, vocabulary tripled (6%→15%),
semantics nearly doubled (5%→9%). Oracle headroom doubled (d=0.767 vs 0.376). Oracle vs
random is significant (d=0.256, p<1e-8), but template vs random is NOT (d=0.012, p=0.79).
The semantic benefit requires the actual query — surrogates cannot capture it. N=500.

### Exp 07 — RoPE Position Isolation (DONE)
**Question**: Does the RoPE position shift from prepending a prefix contribute to
the co-encoding benefit, or is it purely attention redistribution?

**Result**: RoPE conclusively ruled out. Pure RoPE shift (shifted_bare): d=-0.034 (ns).
Invisible prefix with RoPE shift (prefix_encoder_blocked): d=+0.036 (ns). Attention
redistribution WITHOUT RoPE shift (random_rope_neutralized): d=+0.372 (***), actually
BETTER than standard random_trunc (d=+0.296). 2x2 factorial: attention row effect=+0.260,
RoPE column effect=-0.076. Primary mechanism: attention redistribution only. N=500.

### Exp 04A — MS MARCO Ranking (DONE)
**Question**: Can surrogate-primed encoder representations improve passage ranking?

**Result**: MARGINAL. Oracle AUC=0.853 vs bare AUC=0.845 (gain=+0.008, ns). Surrogates
beat oracle on AUC (surr_doc=0.867**, random=0.866*). BUT oracle differential signal
d=-0.007 (ns) — priming helps relevant and irrelevant passages equally. Surr_template
differential d=+0.130 (***) and static_fact d=+0.153 (***) show some differential, but
driven by hard queries (Q1: +0.08-0.13 AUC gain) while slightly hurting easy queries.
Overall: no clean ranking signal from oracle priming. N=400, ~8.2 passages/query.

### Exp 04B — Amazon ESCI Ranking (DONE)
**Question**: Can surrogate priming improve product ranking using query-likelihood scoring?

**Result**: NO. Oracle HURTS ranking: AUC=0.699 vs bare=0.709 (negative gain). Oracle
differential d=-0.269 — helps irrelevant products MORE than relevant ones. Surr_template
AUC=0.724 (**) and surr_doc AUC=0.718 (**) show modest AUC gains, but all differentials
are negative or near zero. The structural mechanism is document-independent — it cannot
create the differential signal needed for ranking. Same outcome as v2. N=400, ~21.6
products/query, query-likelihood scoring.

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

### Exp 03B — Extended Length Scaling (DONE)
**Question**: Does the benefit survive beyond the encoder's sliding window (1024 tokens)?
Does the three-way structure/vocabulary/semantics decomposition hold at all lengths?

**Dataset**: MS MARCO v1.1 with controlled padding to 512, 1024, 2048, 3072, 4096, 6144 tokens.

**Design**: 8 conditions × 7 length bins × N=400, Bonferroni = 49 comparisons.
- bare, oracle_trunc, scrambled_oracle_trunc, random_matched_trunc
- random_trunc, static_fact_trunc, surr_template_trunc, surr_doc_trunc

Three-way decomposition at each length:
- Structure = bare → random_matched (any prefix helps)
- Vocabulary = random_matched → scrambled_oracle (right words, wrong order)
- Semantics = scrambled_oracle → oracle (right word order)

**Encoder architecture**: sliding_window=1024 tokens, full attention every 6th layer
(5/34 layers global). At 6144 tokens, most layers see only local context.

**Result**: NO DECAY. All 8 conditions significant at all 7 lengths (all p < 1e-7).
Oracle d=+0.38 at both original and 6144 tokens — completely flat. The three-way
decomposition holds at every length. See Results Log below.

---

### Exp 04 — Ranking on Commercial Datasets (DONE)
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

## Future Experiments (after Exps 02-05)

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
Exp 2B (Mechanism — DONE)         Exp 03/03B (Length Scaling — DONE)
  │                                 │
  ├── Exp 3D (Cross-dataset — DONE) │
  │                                 │
  ├── Exp 3E (Attention probing     │
  │     — DONE: sink redistribution │
  │     mechanism identified)       │
  │                                 │
  ├── Exp 3F (Semantic amplification│
  │     — DONE: 1.7x amplification, │
  │     vocabulary not semantics)   │
  │                                 ▼
  ├── Exp 05 (LLM Surrogates       Exp 04 (Ranking: MARCO + ESCI + WANDS)
  │     — DONE: NEGATIVE ROI,        ├─ Part A: MS MARCO
  │     template heuristic is         ├─ Part B: Amazon ESCI
  │     sufficient)                   └─ Part C: WANDS
  │
  ├── Exp 06 (Factoid Subsample
  │     — DONE: structural drops
  │     85%→76%, oracle headroom
  │     doubles, but template can't
  │     capture semantic gap)
  │
  ├── Exp 07 (RoPE Isolation
  │     — DONE: RoPE ruled out,
  │     pure attention redistribution)
  ▼                                 ▼
  (Mechanism fully characterized)   Exp 04 (Ranking — DONE)
                                      ├─ A: MS MARCO (marginal, ns oracle)
                                      └─ B: ESCI (oracle HURTS ranking)
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

### Exp 03B — Extended Length Scaling
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 400 | **Dataset**: MS MARCO v1.1 (padded)

8 conditions × 7 length bins, Bonferroni = 49 comparisons. All p < 1e-7.

| Length | oracle d | scrambled d | rand_match d | random d | static d | template d | doc_kw d |
|--------|---------|------------|-------------|---------|---------|-----------|---------|
| orig (~98) | +0.384 | +0.374 | +0.316 | +0.314 | +0.316 | +0.363 | +0.340 |
| 512 | +0.432 | +0.388 | +0.381 | +0.456 | +0.446 | +0.442 | +0.400 |
| 1024 | +0.384 | +0.388 | +0.282 | +0.393 | +0.336 | +0.345 | +0.364 |
| 2048 | +0.335 | +0.309 | +0.274 | +0.346 | +0.327 | +0.316 | +0.397 |
| 3072 | +0.422 | +0.393 | +0.361 | +0.452 | +0.438 | +0.438 | +0.426 |
| 4096 | +0.389 | +0.343 | +0.318 | +0.401 | +0.406 | +0.386 | +0.356 |
| 6144 | +0.382 | +0.322 | +0.323 | +0.355 | +0.386 | +0.349 | +0.379 |

All entries *** (p < 1e-7 after Bonferroni correction for 49 comparisons).

**Three-way decomposition** (averaged across lengths):
- Structure (bare → random_matched): ~80-85% of oracle benefit
- Vocabulary (random_matched → scrambled_oracle): small, variable
- Semantics (scrambled_oracle → oracle): small but consistent

**Key observations**:
1. **Completely flat decay**: Oracle d=+0.38 at original AND at 6144 tokens
2. **Encoder sliding window is irrelevant**: At 6144 tokens, most encoder layers (29/34)
   use 1024-token sliding window, yet the effect is undiminished. The 5 global layers
   (every 6th) are sufficient to propagate prefix influence across the full sequence.
3. **Content-agnostic surrogates strengthen at longer lengths**: random_trunc and
   static_fact_trunc sometimes match or exceed oracle at 512-3072 tokens
4. **Three-way decomposition holds at all lengths**: structural floor is dominant everywhere
5. **No length × condition interaction**: all surrogates decay (or don't) similarly

**Conclusion**: The benefit is architecture-robust — it survives well beyond the sliding
window boundary and shows no sign of decay even at 6144 tokens. Combined with Exp 03,
this confirms documents of any practical length are within the operating envelope.
The 5 global attention layers (every 6th) are sufficient to distribute prefix influence.

### Exp 3D — Cross-Dataset Content Ablation (Long Queries)
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: neural-bridge/rag-dataset-12000

**Dataset**: Synthetic QA with retrieved contexts. Mean query=17.8w (3x MS MARCO),
mean document=604w (10x MS MARCO), mean answer=43w. Filtered to q≥15w, a≥5w.
Pre-screen: PASS (bare NLL=0.98, oracle headroom=+0.255, 0% ceiling).

**Baseline**: Oracle d=+0.592 (p=1.9e-34), headroom=+0.090 nats, win rate=85.0%.

**Content Ablation**:

| Step | Mean NLL | Delta | % total | d | sig |
|------|----------|-------|---------|------|-----|
| bare (baseline) | 1.3135 | — | — | — | — |
| + Structure (→ random_matched) | 1.2380 | +0.0755 | **84.3%** | +0.858 | *** |
| + Vocabulary (→ scrambled_oracle) | 1.2202 | +0.0178 | **19.9%** | +0.209 | *** |
| + Semantics (→ oracle) | 1.2239 | **-0.0037** | **-4.2%** | -0.031 | ns |

**Cross-dataset comparison**:

| Component | MS MARCO (6w queries) | neural-bridge (18w queries) | Change |
|-----------|----------------------|---------------------------|--------|
| Structure | 84.7% | 84.3% | -0.4pp |
| Vocabulary | 5.5% (ns) | 19.9% (***) | +14.4pp |
| Semantics | 9.7% (***) | -4.2% (ns) | -13.9pp |

**All conditions vs oracle (by Cohen's d)**:

| Condition | d | % oracle |
|-----------|------|---------|
| surr_template ("What is [kw]?") | +0.933 | 158% |
| scrambled_oracle | +0.889 | 150% |
| "the" x N | +0.888 | 150% |
| repeat_kw x N | +0.887 | 150% |
| random_matched | +0.858 | 145% |
| **oracle (real query)** | **+0.592** | **100% — WEAKEST** |

**Query length stratification (within-dataset)**:

| Bin | Struct% | Vocab% | Sem% |
|-----|---------|--------|------|
| Short (15w) | 75.5% | 13.9% | +10.6% |
| Medium (17w) | 84.5% | 17.8% | -2.3% |
| Long (20w) | 89.7% | 25.8% | -15.6% |

**Pairwise head-to-head**:

| Comparison | d | sig | Winner |
|------------|------|-----|--------|
| Word ORDER matters? (oracle vs scrambled) | -0.031 | ns | scrambled |
| Right WORDS matter? (scrambled vs random) | +0.209 | *** | scrambled |
| Any content? (oracle vs random) | +0.105 | * | oracle |
| Diversity? (repeat_the vs random) | -0.197 | *** | random |

**Conclusions**:
1. **Structure = 84.3%** — near-perfect replication of MS MARCO's 84.7%. Cross-dataset consistent.
2. **Vocabulary grew** (5.5% → 19.9%): having the right words helps more with longer queries.
3. **Semantics went NEGATIVE** (-4.2%): word order hurts. Real query creates semantic interference.
4. **ALL surrogates beat oracle**: the intact query is the WORST condition on this dataset.
5. The effect strengthens as queries get longer (semantic % goes 10.6% → -15.6%).
6. **The structural mechanism is genuine** — not an artifact of MS MARCO's short queries.
7. For deployment: prepend ANY prefix. Content doesn't help; specific semantic structure can hurt.

### Exp 3E — Attention Mechanism Probing
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: neural-bridge/rag-dataset-12000

**Question**: WHY does prepending any text improve encoder representations?
Three hypotheses: (1) attention redistribution, (2) RoPE position shift,
(3) representation regularization.

**Method**: Forward hooks on 6 encoder layers (0, 5, 11, 17, 23, 29) + final layer,
`attn_implementation="eager"`, 4 conditions (bare, oracle, random_matched, repeat_the),
6 probes on attention weights and hidden states. Runtime: 9.3 min (2000 forward passes).

**NLL cross-reference**: bare=1.313, oracle=1.223, d=+0.604 (matches Exp 3D).

**Probe A — Prefix attention mass** (full-attn layers):

| Layer | oracle | random | repeat_the |
|-------|--------|--------|------------|
| 5 | 13.2% | 13.4% | 9.8% |
| 11 | 14.2% | 12.0% | 9.9% |
| 17 | 15.2% | 13.2% | 11.9% |
| 23 | 24.9% | 23.5% | 22.5% |
| 29 | 26.3% | 25.1% | 24.0% |

Mass is uniform across doc positions (CV < 0.15). Oracle and random absorb nearly
identical mass (~1-2pp difference). Prefix acts as a uniform attention drain.

**Probe B — Entropy**: INCREASES in all 6 layers, all 3 conditions (all p < 10^-7).
Repeat_the causes the largest increase at layer 5 (d=+1.62). The prefix smooths
attention, not focuses it. Consistent with regularization/dilution.

**Probe C — Doc-doc redistribution**: KL divergence (bare vs prefixed doc-doc patterns)
grows to 2.97 nats at layer 29. Nearly identical across conditions:

| Layer | oracle KL | random KL | ratio |
|-------|-----------|-----------|-------|
| 5 | 0.480 | 0.503 | 0.95x |
| 17 | 0.946 | 0.885 | 1.07x |
| 29 | 2.966 | 2.794 | 1.06x |

The redistribution is structural — it barely matters what the prefix contains.

**Probe D — Shift magnitude** (L2 distance, bare vs prefixed doc representations):

| Layer | oracle | random | repeat_the | oracle/random |
|-------|--------|--------|------------|---------------|
| 0 | 15.0 | 15.6 | 16.5 | 0.96x |
| 17 | 666 | 685 | 528 | 0.97x |
| 33 | 4,331 | 4,851 | 3,636 | 0.89x |

Random shifts representations MORE than oracle (d=-0.40 *** at final layer). Repeat_the
shifts least. Yet all three produce similar NLL benefits. Shift magnitude does not
determine benefit — random words are more disruptive but no more helpful.
Early doc positions shift 3-4x more than late positions.

**Probe E — Shift direction** (cosine similarity of shift vectors vs oracle):

| Layer | random vs oracle | repeat vs oracle | interpretation |
|-------|-----------------|-----------------|----------------|
| 0 | 0.489 | 0.501 | MIXED |
| 5 | 0.395 | 0.223 | MIXED |
| 11 | 0.271 | 0.231 | SEMANTIC |
| 17 | 0.279 | 0.264 | SEMANTIC |
| 23 | 0.295 | 0.290 | SEMANTIC |
| 29 | 0.306 | 0.300 | MIXED |
| 33 | 0.299 | 0.302 | MIXED |

Overall mean cosine = **0.32**. Different prefixes push in **different directions** —
only 4.6% of samples have cosine > 0.5 at layer 29. Yet all produce similar NLL.
Resolution: bare state is suboptimal, all directions away from it reach the
good manifold.

**Probe F — Attention sinks**: In bare encoding, position 0 absorbs 56-143x
average attention (classic attention sink). With prefix, prefix tokens absorb the
sink role (4-12x more than doc tokens), freeing position 0 to contribute to semantics.
Oracle tokens are better sinks than random (d=+0.41 to +0.77 ***).

**Unified mechanism — attention sink redistribution**:

The bare encoder has a degenerate attention pattern: the `<bos>` token (ID 2, always
position 0) absorbs ~72-76% of doc-token attention budget at layer 23 (67-87x average
attention). Adding ANY prefix:

1. **Does NOT reduce attention to `<bos>`** — it stays at ~72% regardless of prefix
   (the sink is a property of the `<bos>` embedding, not just position)
2. **Prefix steals from doc-doc attention** (drops ~3pp), NOT from `<bos>` (Probe F)
3. **Remaining doc-doc attention reorganizes** — entropy increases +0.15-0.22 nats,
   KL up to 2.97 nats (Probe C)
4. **Smooths attention** — entropy increases everywhere (Probe B)
5. **Shifts representations** off the degenerate bare manifold (Probe D)
6. Direction depends on prefix content (cosine ~0.32), but **all directions
   away from bare are equally good for NLL** (Probe E)
7. **Single token is insufficient**: "X" barely changes anything (doc_to_prefix ~0.3%).
   Need 5+ tokens for meaningful redistribution.

**CORRECTION from initial Exp 3E narrative**: The prefix does NOT "absorb the sink
role away from `<bos>`". The `<bos>` token retains its massive attention share. Instead,
the prefix acts as a buffer that steals a small fraction of doc-doc attention, and this
perturbation causes the remaining doc-doc patterns to reorganize beneficially.

**This explains the paradoxes from Exp 2B/3D**:
- 85% structural: the doc-doc redistribution is content-independent (KL ratio ~1.0x)
- All surrogates beat oracle on neural-bridge: oracle semantics create interference
- 1 random word gets 85% of headroom: even small doc-doc perturbation triggers reorganization
- "the the the" works: repetitive tokens are perfectly good attention buffers
- Uniform tokens beat diverse random (Exp 2B): concentrated prefix = cleaner redistribution

**Conclusions**:
1. **Primary mechanism: attention buffer redistribution** — prefix perturbs doc-doc
   attention allocation, triggering beneficial reorganization
2. The `<bos>` sink is a learned property of the token embedding (follows the token,
   not the position). The sink is NOT displaced by the prefix.
3. Hypothesis 1 (attention redistribution) is **confirmed** as the dominant mechanism,
   but the redistribution is doc-doc, not `<bos>`-to-prefix
4. Hypothesis 2 (RoPE position shift) contributes to different shift directions but
   is not the primary driver (repeat_the shifts least despite longest token prefix)
5. Hypothesis 3 (regularization) is confirmed as a secondary effect — but the benefit
   is not from the noise itself, it's from escaping the degenerate bare state
6. The mechanism is a **geometric escape**: bare reps occupy a suboptimal manifold,
   any prefix moves them to a better manifold, and the specific direction doesn't matter
7. For deployment: confirmed — prepend ANY short prefix (5+ tokens). The mechanism is
   fully understood and the structural recommendation from Exp 2B/3D is validated

### Exp 3F — Semantic Amplification
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: MS MARCO v1.1

**Question**: Can repeating the prefix amplify the semantic component beyond the ~10%
baseline from Exp 2B? If the structural effect saturates, does the semantic share grow?

**Part 1: Repetition Sweep** (oracle_xN and random_xN, N=1,3,5,10,20)

| N | Struct delta | Sem delta | Total delta | Sem frac | Struct d | Sem d | Total d |
|---|-------------|-----------|-------------|----------|----------|-------|---------|
| 1 | +0.579 | +0.104 | +0.684 | 15.3% | +0.296 | +0.134 | +0.376 |
| 3 | +0.529 | +0.141 | +0.669 | 21.0% | +0.310 | +0.239 | +0.365 |
| 5 | +0.489 | +0.133 | +0.622 | 21.4% | +0.324 | +0.212 | +0.360 |
| 10 | +0.428 | +0.155 | +0.582 | 26.6% | +0.345 | +0.186 | +0.364 |
| 20 | +0.361 | +0.129 | +0.490 | 26.3% | +0.331 | +0.224 | +0.414 |

Semantic fraction: 15.3% → 21.0% → 21.4% → 26.6% → 26.3%. **Amplification factor: 1.7x.**
Structural d is saturated (x20/x1 = 1.12). All conditions significant after Bonferroni (k=19).

Verification: bare and oracle_x1 NLLs match Exp 02 exactly (max diff = 0.000000).

**Part 2: 3-Way Decomposition at N=5 and N=10**

| N | Structure | Vocabulary | Semantics |
|---|-----------|------------|-----------|
| 1 (Exp 2B) | 84.7% | 5.5% (ns) | 9.7% (***) |
| 5 | 78.6% (***) | 12.0% (***) | 9.4% (*) |
| 10 | 73.4% (***) | 15.8% (***) | 10.7% (***) |

Structure dropped 84.7% → 73.4%, but **vocabulary** grew 5.5% → 15.8% (not semantics).
Repeating the prefix amplifies vocabulary exposure (having the right words), not
word-order semantics (which stays flat at ~10%).

**Part 3: Content Concentration**

| N | Full query sem/struct | Content-only sem/struct |
|---|----------------------|------------------------|
| 5 | 0.272 | 0.108 |
| 10 | 0.362 | 0.154 |

Stripping stop words **hurts** the semantic ratio (p<0.001). Stop words provide
useful structural scaffolding. Content-only prefix has 60% as many tokens.

**Part 4: Short Documents** (30-word truncated docs)

| Metric | Full doc | Short doc (30w) |
|--------|----------|----------------|
| Cohen d | +0.376 | +0.458 |
| Prefix/doc ratio | 0.08 | 0.18 |

Prefix benefit is **larger** for shorter documents. The semantic signal is diluted
by document length — higher prefix/doc ratio amplifies the prefix effect.

**Part 5: Structural Saturation**

| Condition | d | % Oracle_x10 |
|-----------|------|-------------|
| the_matched10 (uniform, ~67 toks) | +0.322 | 88.6% |
| random_x10 (diverse, ~82 toks) | +0.345 | 94.8% |
| oracle_x10 (semantic, ~66 toks) | +0.364 | 100% |

"the" vs random_x10: d=+0.055, p=0.216 (ns). Token diversity does not significantly
matter — structural benefit comes from attention **mass**, not token content.
Structural d saturated: x1=+0.296, x3=+0.310, x5=+0.324, x10=+0.345, x20=+0.331.

**Part 6: Hardness x Repetition Interaction**

| Quintile | x1 sem frac | x20 sem frac | Growth |
|----------|------------|-------------|--------|
| Q1 easy | 31.4% | 25.3% | 0.8x |
| Q5 hard | 7.7% | 23.2% | 3.0x |

Hardness correlation flips with repetition: r=-0.161 (x1) → r=+0.416 (x10).
At x1, easy samples have higher semantic fraction. At x10+, hard samples catch up.
Repetition preferentially amplifies semantic content for hard samples.

**Conclusions**:
1. **Moderate amplification (1.7x)**: semantic fraction grew 15% → 26% with repetition
2. **Vocabulary, not semantics**: the growth is from repeated word exposure (5.5% → 15.8%),
   not word-order meaning (stays ~10%)
3. **Structural saturation confirmed**: x20/x1 = 1.12, saturates by x3
4. **Stop words help**: stripping them halves the semantic ratio — they provide scaffolding
5. **Short docs amplify**: higher prefix/doc ratio → larger prefix benefit (d=+0.458 vs +0.376)
6. **Hardness interaction**: repetition triples semantic fraction for hard samples (8% → 23%)
7. **Practical implication**: repeating the query 3-5x modestly boosts semantic content,
   but the mechanism remains dominated by structure at all repetition levels

### Exp 05 — LLM-Generated Surrogate Queries
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: MS MARCO v1.1

**Question**: Can Gemma 2 9B-IT generate surrogates that beat the "What is [keyword]?"
heuristic? Three prompt variants informed by Exp 2B-3F findings.

**Two-phase design**: (1) Gemma 2 9B-IT generates 3 × 500 surrogates → save JSON → free VRAM,
(2) T5Gemma scores 14 conditions × 500 samples. Runtime: ~45 min total.

**Prompt variants**:
- **need**: "Write a short web search someone would type to find this document" (complementary vocab)
- **question**: "Write a short question that this document answers" (traditional QG)
- **keywords**: "List 3-5 search keywords, only content words" (stop-word-free control)

**Surrogate characteristics**:
- Word counts: need=4.9w, question=7.6w, keywords=6.2w, oracle=6.0w
- Doc vocabulary overlap: need=0.72, question=0.78, keywords=0.93, oracle=0.73
- Stop word fraction: need=0.24, question=0.50, keywords=0.00, oracle=0.43

**All conditions (14)**:

| Condition | d vs bare | Delta | Win% | % Oracle | p |
|-----------|-----------|-------|------|----------|---|
| oracle_x1 (upper bound) | +0.376 | +0.684 | 92.6% | 100% | 4.8e-16 |
| LLM question x1 | +0.405 | +0.483 | 87.6% | 108% | 3.1e-18 |
| oracle_x4 | +0.359 | +0.649 | 92.0% | 96% | 7.1e-15 |
| scrambled LLM need x4 | +0.352 | +0.547 | 88.8% | 94% | 2.2e-14 |
| LLM need x4 | +0.347 | +0.577 | 90.8% | 92% | 5.0e-14 |
| LLM keywords x4 | +0.346 | +0.479 | 86.0% | 92% | 5.7e-14 |
| LLM question x4 | +0.345 | +0.469 | 84.8% | 92% | 6.7e-14 |
| surr_template x1 | +0.336 | +0.559 | 90.8% | 90% | 2.6e-13 |
| LLM need x1 | +0.333 | +0.595 | 87.6% | 89% | 4.1e-13 |
| surr_template x4 | +0.326 | +0.539 | 90.8% | 87% | 1.2e-12 |
| random_x4 | +0.312 | +0.502 | 84.8% | 83% | 1.0e-11 |
| random_x1 | +0.296 | +0.579 | 86.8% | 79% | 9.4e-11 |
| LLM keywords x1 | +0.291 | +0.501 | 86.0% | 78% | 1.7e-10 |

All significant after Bonferroni (k=13, alpha=0.0038). Verification: bare and oracle_x1
NLLs match Exp 02 exactly (max diff = 0.000000).

**Notable**: LLM question x1 achieves d=+0.405 (108% of oracle) — exceeding the real query.
This echoes Exp 3D where surrogates beat oracle due to semantic interference from the real query.

**Part 1: LLM vs Heuristic** (the key question):

| Comparison | d | Win% | p | sig |
|------------|------|------|---|-----|
| LLM need x1 vs template x1 | +0.048 | 47.0% | 0.28 | ns |
| LLM need x4 vs template x4 | +0.084 | 51.6% | 0.06 | ns |
| LLM need x1 vs random x1 | +0.029 | 56.4% | 0.52 | ns |
| LLM need x4 vs random x4 | +0.149 | 64.6% | 9.6e-4 | *** |

**No significant uplift** of LLM surrogates over template heuristic at either repetition level.
LLM need beats random only at x4 (vocabulary exposure with repetition).

**Part 2: Prompt Variant Comparison**:

| Comparison (x4) | d | p | sig |
|-----------------|------|---|-----|
| need vs question | +0.163 | 2.9e-4 | *** |
| need vs keywords | +0.229 | 4.3e-7 | *** |
| question vs keywords | -0.019 | 0.67 | ns |

**need is best** among LLM variants at x4. Stop-word hypothesis **confirmed**: need > keywords
(p<0.001) — natural language with stop words outperforms bare keywords. Semantic interference
**confirmed**: need > question (p<0.001) — question framing hurts, consistent with Exp 3D.

**Part 3: Repetition + Decomposition**:

x1 vs x4 uplift is not significant for any condition except random (p=0.001).

3-way decomposition of llm_need_x4:

| Component | Delta | % total | d | sig |
|-----------|-------|---------|------|-----|
| Structure (bare → random_x4) | +0.502 | 87.0% | +0.312 | *** |
| Vocabulary (random → scrambled) | +0.045 | 7.9% | +0.106 | * |
| Semantics (scrambled → llm_need) | +0.030 | 5.2% | +0.116 | ** |

Decomposition residual: 0.000000. Consistent with Exp 2B: mechanism is ~87% structural.

**Part 4: Hardness Interaction**:

| Quintile | x1 sem gap | x4 sem gap | x4 sem frac |
|----------|-----------|-----------|------------|
| Q1 easy | +0.010 | +0.013 | 18.0% |
| Q3 | +0.027 | +0.053 | 21.2% |
| Q5 hard | -0.005 | +0.191 | 9.4% |

LLM need vs template by quintile (x1): no significant differences at any quintile.
Hardness correlation for LLM need semantic gap: r=-0.225 (x1), r=+0.126 (x4).

**Conclusions**:
1. **LLM surrogates have NEGATIVE ROI**: no significant uplift over "What is [keyword]?" heuristic
2. **Best prompt: "need"** (complementary vocabulary, natural stop words), but the advantage
   is only over other LLM prompts, not over the template heuristic
3. **Stop-word hypothesis confirmed**: need > keywords (p<0.001 at x4)
4. **Semantic interference confirmed**: need > question (p<0.001 at x4), echoing Exp 3D
5. **Decomposition**: 87% structural / 8% vocabulary / 5% semantics — consistent with Exp 2B
6. **LLM question x1 at 108% of oracle**: another instance of surrogates beating the real
   query, confirming that specific semantic structure creates interference
7. **Practical recommendation**: Use "What is [keyword]?" heuristic. LLM generation costs
   ~75 min of Gemma 2 9B GPU time for zero uplift. The mechanism is structural.

### Exp 06 — Factoid Subsample Validation
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: MS MARCO v1.1 (answer ≤5 words)

**Question**: Does filtering to short factoid answers shift the decomposition from ~85%
structural toward a more semantic regime? Motivated by the finding that the 85% aggregate
is a Simpson's paradox-like average over two populations: short factoid answers (~37%
structural in post-hoc analysis) vs long answers (~113% structural, negative semantics).

**Dataset**: Fresh 500 MS MARCO v1.1 samples filtered to answer ≤5 words (SEED=43,
different from Exp 02's SEED=42). Mean answer length: 2.2 words. Bare NLL: 6.77
(much higher than full dataset's ~3.7 — these are harder questions).

**Baseline**: Oracle d=+0.767, headroom=+1.200 nats. **Oracle effect is 2x the full
dataset** (d=0.376). Factoid questions benefit much more from query co-encoding.

**3-Way Decomposition** (bare → random_x1 → scrambled_oracle → oracle):

| Component | Delta | % total | d | sig |
|-----------|-------|---------|------|-----|
| Structure (bare → random) | — | **76.0%** | +0.578 | *** |
| Vocabulary (random → scrambled) | — | **15.3%** | +0.164 | *** |
| Semantics (scrambled → oracle) | — | **8.7%** | +0.154 | *** |

**Cross-dataset comparison of decomposition**:

| Component | Full MS MARCO (Exp 2B) | Factoid subsample (Exp 06) | Change |
|-----------|----------------------|--------------------------|--------|
| Structure | 84.7% | 76.0% | -8.7pp |
| Vocabulary | 5.5% (ns) | 15.3% (***) | +9.8pp |
| Semantics | 9.7% (***) | 8.7% (***) | -1.0pp |

**Surrogate Comparison** (the practical question):

| Comparison | d | p | sig |
|------------|------|---|-----|
| Oracle vs random | +0.256 | 1.8e-08 | *** |
| Template vs random | +0.012 | 0.787 | ns |

**All conditions**:

| Condition | d vs bare | % Oracle | p |
|-----------|-----------|----------|---|
| oracle_x1 (upper bound) | +0.767 | 100% | 3.5e-52 |
| oracle_x4 | +0.734 | 96% | 1.0e-48 |
| scrambled_oracle | +0.722 | 94% | 1.9e-47 |
| surr_template x1 | +0.685 | 89% | 1.2e-43 |
| surr_template x4 | +0.639 | 83% | 4.6e-39 |
| random_x1 | +0.578 | 75% | 3.3e-33 |
| random_x4 | +0.514 | 67% | 2.7e-27 |

**Conclusions**:
1. **Structural dropped from 85% to 76%** — meaningful shift but still dominant
2. **Vocabulary tripled** (5.5% → 15.3%) — having the right words matters more for factoids
3. **Semantics stayed flat** (9.7% → 8.7%) — word order doesn't help more for factoids
4. **Oracle headroom doubled** (d=0.767 vs 0.376) — factoid QA has much larger benefit
5. **Oracle vs random is highly significant** (d=0.256, p<1e-8) — the semantic gap MATTERS here
6. **Template CANNOT capture the semantic gap** (d=0.012 vs random, p=0.79) — the "What is
   [keyword]?" heuristic provides zero semantic benefit beyond random on this subsample
7. **Only the real query captures the semantic benefit** — for factoid QA, query content
   genuinely matters, but no surrogate (template or LLM) can replicate it
8. **Two-population interpretation validated**: factoid answers ARE more semantic-sensitive,
   but even here the mechanism is still 76% structural. The prediction of 40-50% structural
   was too optimistic — the structural mechanism is robust even on the most favorable subsample

### Exp 07 — RoPE Position Isolation
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 500 | **Dataset**: MS MARCO v1.1

**Question**: Does the encoder RoPE position shift from prepending a prefix contribute to
the co-encoding benefit? Or is the benefit purely from attention redistribution?

**Design**: 2x2 factorial {attention redistribution ON/OFF} × {RoPE shift ON/OFF}, plus controls.

| # | Condition | Attention | RoPE shift | d vs bare | p | sig |
|---|-----------|-----------|-----------|-----------|---|-----|
| 1 | bare | — | — | 0.000 | — | — |
| 2 | oracle_trunc | both | both | +0.376 | 4.8e-16 | *** |
| 3 | random_trunc | ON | ON | +0.296 | 9.4e-11 | *** |
| 4 | shifted_bare | OFF | ON | -0.034 | 0.448 | ns |
| 5 | prefix_encoder_blocked | OFF (4D mask) | ON | +0.036 | 0.427 | ns |
| 6 | random_rope_neutralized | ON | OFF | +0.372 | 7.9e-16 | *** |

**Method**:
- `shifted_bare`: bare encoding with position_ids shifted by +6 (tests pure absolute RoPE shift)
- `prefix_encoder_blocked`: prefix present but invisible to doc tokens via 4D attention mask
  (tests RoPE shift + invisible prefix in attention computation)
- `random_rope_neutralized`: random prefix present in attention, but doc positions preserved
  at [0..doc_len-1] via custom position_ids (attention redistribution without RoPE shift)

**2x2 Factorial Analysis**:

|  | RoPE shift OFF | RoPE shift ON |
|--|---------------|--------------|
| Attention OFF | bare (0.000) | shifted_bare (-0.034) |
| Attention ON | rope_neutralized (+0.372) | random_trunc (+0.296) |

- **Row effect (attention)**: +0.260 — attention redistribution provides the benefit
- **Column effect (RoPE)**: -0.076 — RoPE shift actually HURTS slightly
- **Interaction**: RoPE shift reduces the attention benefit (0.372 → 0.296)

**Key result**: `random_rope_neutralized` (d=+0.372) is BETTER than `random_trunc` (d=+0.296),
and the difference is significant (d=-0.397, p=1.2e-17). Removing the RoPE shift actually
improves the effect.

**Caveat**: The neutralized condition has higher variance and lower raw NLL (2.242 vs 3.097)
due to unusual position arrangement (prefix at positions [doc_len..doc_len+N]). The d-value
comparison is reliable but raw NLLs are not directly comparable.

**Conclusions**:
1. **RoPE is conclusively ruled out** as a mechanism contributor
2. Pure RoPE shift does nothing (shifted_bare: d=-0.034, ns)
3. Invisible prefix with RoPE shift does nothing (prefix_encoder_blocked: d=+0.036, ns)
4. Attention redistribution WITHOUT RoPE works BETTER (d=+0.372 vs +0.296)
5. RoPE slightly attenuates the benefit (column effect = -0.076)
6. The mechanism is **purely attention redistribution** — consistent with Exp 3E
7. This rules out Hypothesis 2 (RoPE position shift) from the Exp 3E probe

### Exp 04A — MS MARCO Ranking
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 400 queries | **Dataset**: MS MARCO v2.1

**Question**: Can surrogate-primed encoder representations improve passage ranking?

**Method**: Answer-likelihood scoring — `NLL(answer | encode([condition + passage]))`.
400 queries, ~8.2 candidate passages per query (3,276 total scoring passes per condition).
6 conditions: bare, oracle_trunc, surr_template_trunc, surr_doc_trunc, static_fact_trunc, random_trunc.

**AUC Results**:

| Condition | AUC | Gain vs bare | p | sig |
|-----------|-----|-------------|---|-----|
| bare | 0.845 | — | — | — |
| oracle_trunc | 0.853 | +0.008 | 0.126 | ns |
| surr_doc_trunc | 0.867 | +0.022 | 0.001 | ** |
| surr_template_trunc | 0.861 | +0.016 | 0.010 | * |
| static_fact_trunc | 0.864 | +0.019 | 0.003 | ** |
| random_trunc | 0.866 | +0.021 | 0.012 | * |

**Critical test — Differential signal** (does priming help relevant MORE than irrelevant?):

| Condition | delta_relevant | delta_irrelevant | differential d | p | sig |
|-----------|---------------|-----------------|---------------|---|-----|
| oracle_trunc | +0.600 | +0.607 | -0.007 | 0.904 | ns |
| surr_template | +0.494 | +0.366 | +0.130 | <0.001 | *** |
| static_fact | +0.389 | +0.242 | +0.153 | <0.001 | *** |
| random_trunc | +0.537 | +0.516 | +0.024 | 0.658 | ns |

**Oracle has ZERO differential signal** — priming helps relevant and irrelevant equally (d=-0.007).
Surr_template and static_fact show some differential, but this is driven by:
- Hard queries (Q1): all conditions gain +0.08-0.13 AUC
- Easy queries (Q4-Q5): slight harm from priming

**Conclusions**:
1. Oracle priming does NOT create a ranking signal (AUC gain +0.008, ns)
2. The structural mechanism is document-independent — it helps all passages equally
3. Content-agnostic surrogates (random, static) perform as well as oracle for AUC
4. Some differential exists for surr_template/static_fact, but not for oracle or random
5. Hardness interaction: priming helps ranking for hard queries, slightly hurts for easy
6. v2's ranking failure (Exps 22/23/28) is replicated in encoder-decoder architecture

### Exp 04B — Amazon ESCI Ranking
**Status**: COMPLETE | **Date**: 2026-02-18 | **N**: 400 queries | **Dataset**: Amazon ESCI (US)

**Question**: Can surrogate priming improve product ranking? Query-likelihood scoring on
a dataset with truly irrelevant candidates (unlike MS MARCO's topically similar pools).

**Method**: Query-likelihood scoring — `NLL(query | encode([condition + product_text]))`.
400 queries, ~21.6 products per query (8,642 total scoring passes per condition).
Pre-screen: PASS (bare QL AUC=0.713, much better than v2's 0.578 on MS MARCO).
6 conditions: bare, oracle_trunc, surr_template_trunc, surr_doc_trunc, surr_title_trunc, random_trunc.

**AUC Results** (binary: E+S = relevant, C+I = irrelevant):

| Condition | AUC | Gain vs bare | p | sig |
|-----------|-----|-------------|---|-----|
| bare | 0.709 | — | — | — |
| oracle_trunc | 0.699 | **-0.010** | 0.189 | ns |
| surr_template_trunc | 0.724 | +0.015 | 0.002 | ** |
| surr_doc_trunc | 0.718 | +0.009 | 0.002 | ** |
| surr_title_trunc | 0.700 | -0.009 | 0.193 | ns |
| random_trunc | 0.717 | +0.008 | 0.100 | ns |

**Oracle HURTS ranking** — AUC drops from 0.709 to 0.699. The real query primes irrelevant
products MORE than relevant ones (differential d=-0.269).

**Differential signal** (all negative or near zero):

| Condition | differential d | p | sig |
|-----------|---------------|---|-----|
| oracle_trunc | -0.269 | <0.001 | *** |
| surr_title_trunc | -0.245 | <0.001 | *** |
| surr_template_trunc | -0.013 | 0.765 | ns |
| random_trunc | +0.032 | 0.432 | ns |

**Conclusions**:
1. **Oracle HURTS ranking on ESCI** — priming helps irrelevant products more than relevant
2. surr_title (product name) also hurts — same problem as oracle, too semantically specific
3. surr_template shows modest AUC gain (+0.015) but zero differential (d=-0.013, ns)
4. The structural mechanism is document-independent — it cannot create ranking signal
5. v2's ranking failure is definitively replicated on a commercial dataset
6. The mechanism fundamentally cannot do ranking: it reshapes ALL documents similarly
7. **Ranking is a dead end for this approach** — across v2 (6 experiments) and v3 (2 experiments),
   no configuration has ever produced a meaningful ranking signal from cache priming
