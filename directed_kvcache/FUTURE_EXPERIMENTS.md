# Future Experiment Proposals

Updated after Experiment 15 completed the end-to-end ranking validation.

## Summary of Completed Experiments (13-15)

### Experiment 13: Multi-Query Amplification (DONE)
- **13A:** 5q@sim>=0.50 > 1q@sim>=0.70 (p=0.011). Repetition helps (p=0.042).
- **13B:** Hardness-benefit r=0.302 (9x stronger than similarity). P25 gate improves d by 7.7%.

### Experiment 14: Isolate and Amplify (DONE)
- **14A (Repetition Scaling):** Peak at 5x repetitions (d=0.194). Diminishing returns beyond 5x.
- **14B (Diversity):** NULL RESULT. 5 similar queries (d=0.238) ≈ 5 diverse queries (d=0.180). Repetition > diversity.
- **14C (Answer Length):** Weak effect (partial r=0.051). Hardness dominates.
- **14D (LLM Surrogates):** Intent-based generation (d=0.274, win=73%) matches real queries. Cold-start solved.

### Experiment 15: End-to-End Ad Ranking (DONE)
- **NLL improvements translate to ranking improvements** — validated core hypothesis
- **Baseline:** MRR 0.630, Hit@1 48.2%
- **Oracle (P25 gate + oracle_5x):** MRR 0.657 (+2.7pp), Hit@1 51.0% (+2.8pp)
- **Practical (learned gate + historical/LLM):** MRR 0.683 (+5.3pp), Hit@1 54.6% (+6.4pp)
- **Surprising finding: Practical > Oracle!** LLM intent surrogates outperform ground-truth queries
- **Random_5x ≈ Oracle_5x** — structural benefit rivals semantic benefit
- **Per-difficulty:** Q3 (medium-hard) shows largest gains (+12.3pp MRR)

## Key Findings That Inform Future Work

1. **NLL proxy is valid** — ranking improvements confirmed (Exp 15)
2. **LLM surrogates > ground truth** — intent-based generation outperforms oracle queries
3. **Random prefixes provide structural benefit** — not purely semantic
4. **Oracle can HURT some samples** — semantic interference on 33/500 pages
5. **Optimal strategy = 5x repetition of best single query** — not 20x, not diverse queries
6. **Hardness gating is critical** — skip Q1 (easy) samples, focus on Q2-Q4

## The New Open Question

Random_5x performs nearly as well as oracle_5x (MRR 0.669 vs 0.657). **What is the optimal composition of semantic vs random tokens in the prefix?**

---

## Experiment 16: Prefix Composition (PRIORITY 1 - NEW)

**Core hypothesis:** The optimal prefix combines semantic signal (from oracle/LLM queries) with structural noise (from random queries). A mix captures both benefits while avoiding semantic interference.

**Motivation from Exp 15:**
- Oracle helps 91 pages, hurts 33 pages (net +58)
- Random helps 100 pages, hurts 24 pages (net +76) — better net effect!
- 6 pages where oracle actively hurts but random helps
- Random provides "regularization" that prevents over-focusing

**Design (MS MARCO, N=500, reuse Exp 15 ranking task):**

**Conditions (all 5 repetitions total):**

| Condition | Composition | Hypothesis |
|-----------|-------------|------------|
| `pure_oracle_5x` | oracle × 5 | Baseline semantic |
| `pure_random_5x` | random × 5 | Baseline structural |
| `mix_4o_1r` | oracle × 4 + random × 1 | Mostly semantic |
| `mix_3o_2r` | oracle × 3 + random × 2 | Balanced |
| `mix_2o_3r` | oracle × 2 + random × 3 | Mostly structural |
| `mix_1o_4r` | oracle × 1 + random × 4 | Light semantic |
| `interleaved` | oracle, random, oracle, random, oracle | Position effect |

**LLM surrogate variants:**

| Condition | Composition |
|-----------|-------------|
| `pure_llm_5x` | llm_intent × 5 |
| `mix_3llm_2r` | llm_intent × 3 + random × 2 |
| `mix_2llm_3r` | llm_intent × 2 + random × 3 |

**Metrics:** MRR@10, Hit@1, Hit@3, NDCG@10 (same as Exp 15)

**Analysis:**
- Find optimal mix ratio for oracle and LLM surrogates
- Test whether interleaving matters (position effects)
- Per-page analysis: which pages benefit from semantic vs structural?
- Develop heuristic: when to use pure semantic vs mixed

**Why this matters:** Could improve practical system by 10-20% by finding optimal composition. May explain why LLM surrogates outperform oracle (they're noisier = more structural benefit).

**Estimated cost:** ~25,000 forward passes, ~3 hours.

---

## Experiment 17: TriviaQA Failure Diagnosis

**Core hypothesis:** TriviaQA fails due to short answers (3 tokens avg) amplifying perturbation, not domain mismatch. Filtering to long-answer TriviaQA samples should recover the benefit.

**Design (TriviaQA, N=1000):**

**Investigation A — Answer length filtering:**
- Score bare + oracle_5x on all 1000 samples
- Stratify by answer token length: 1-2, 3-5, 6-10, 11+
- Hypothesis: d increases with answer length, turning positive at 6+ tokens

**Investigation B — Natural Questions as fourth dataset:**
- NQ has Wikipedia passages + real Google queries + variable answer lengths
- 5 conditions: bare, oracle_1x, oracle_5x, llm_intent_1x, llm_intent_5x
- Stratify by answer length
- Hypothesis: NQ behaves like MS MARCO where answers are long, like TriviaQA where short

**Why this matters:** Confirms the answer-length hypothesis for failure modes. Informs deployment: don't prime pages targeting short factoid answers.

**Estimated cost:** ~10,000 forward passes, ~1.5 hours.

---

## Experiment 18: Practical Hardness Gate Features

**Core hypothesis:** The oracle hardness gate (bare NLL) requires test-time information. Practical gates using indexing-time features can approximate it.

**Design (MS MARCO, N=2000):**

**Indexing-time features (computed without knowing the query):**
- Passage length (tokens)
- Passage perplexity (bare passage, no query)
- Vocabulary complexity (type-token ratio)
- Named entity density
- Average word frequency

**Evaluation:**
- Compute all features for 2000 passages
- Score bare + oracle_5x for all
- Train logistic regression: features → (oracle_5x wins)
- Evaluate: does feature-based gating match oracle gating performance?

**Target:** Achieve 80%+ of oracle gate improvement using only indexing-time features.

**Why this matters:** Makes hardness gating deployable. Oracle gate requires bare NLL which requires the query.

**Estimated cost:** ~4,000 forward passes (reuse from earlier exps), ~30 min.

---

## Experiment 19: Attention-Based Surrogate Optimization

**Core hypothesis:** Now that we know value contamination is the mechanism, we can optimize the surrogate to maximize document attention to relevant prefix tokens.

**Design (MS MARCO, N=200):**

**Investigation A — Attention analysis of winning surrogates:**
- For samples where oracle_5x wins big (delta > 0.3):
  - Which prefix tokens receive most attention from document tokens?
  - Which document tokens attend most to prefix?
  - Is there a pattern? (e.g., answer-adjacent tokens attend to query keywords)

**Investigation B — Attention-weighted surrogate generation:**
- Generate surrogates with LLM
- Score each by predicted attention pattern
- Select surrogate that maximizes attention from answer-adjacent tokens
- Compare against random selection from same LLM output

**Why this matters:** Understanding the attention mechanism could enable better surrogate selection or generation.

**Estimated cost:** ~2,000 forward passes + attention extraction, ~1 hour.

---

## Experiment 20: Cross-Model Generalization

**Core hypothesis:** The value contamination mechanism is architecture-general. Findings should transfer to other decoder-only LLMs.

**Design:**

**Models to test:**
- Llama-2-7B (same size, different architecture)
- Phi-2 (2.7B, smaller but strong)
- Gemma-7B (Google architecture)

**Conditions (minimal replication):**
- bare, oracle_1x, oracle_5x, random_5x
- N=500 MS MARCO samples

**Metrics:** Same as Exp 12 — win rate, Cohen's d, bootstrap CI on correlation.

**Why this matters:** Confirms whether findings are Mistral-specific or general. Production may use different models.

**Estimated cost:** ~8,000 forward passes per model, ~6 hours total.

---

## Experiment 21: Production-Scale Efficiency Analysis

**Core hypothesis:** The 5x repetition sweet spot may shift with larger context windows or longer documents.

**Design (MS MARCO long passages, N=500):**

**Document length buckets:** 100, 200, 500, 1000 tokens

**Repetition levels:** 1x, 3x, 5x, 10x, 20x

**Analysis:**
- Does optimal repetition count scale with document length?
- Is there a tokens-of-prefix / tokens-of-document ratio that's optimal?
- Compute efficiency: benefit per additional prefix token

**Why this matters:** Production documents vary in length. One-size-fits-all may not be optimal.

**Estimated cost:** ~10,000 forward passes, ~1.5 hours.

---

## Output Convention

All experiment outputs must be saved to `results/expXX/` subdirectories.
See `CLAUDE.md` for the full convention.

## Recommended Priority Order

1. **Exp 16 (Prefix Composition)** — motivated by Exp 15 finding that random ≈ oracle
2. **Exp 18 (Practical Hardness Gate)** — makes hardness gating deployable
3. **Exp 17 (TriviaQA Diagnosis)** — confirms failure mode theory
4. **Exp 20 (Cross-Model)** — tests generalization before production deployment
5. **Exp 19 (Attention Analysis)** — mechanistic understanding
6. **Exp 21 (Efficiency Analysis)** — production optimization

**Total estimated budget for priority 1-3:** ~39,000 forward passes, ~5 hours.
