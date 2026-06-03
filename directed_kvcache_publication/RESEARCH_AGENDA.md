# Directed KV-Cache Priming — Research Agenda

Goal: turn the current findings (contrastive-margin metric + a hand-found "extract"
prompt that improves discrimination) into a **general method** with a real task-accuracy
contribution and a mechanism. Each phase has a decision GATE; we do not proceed past a
gate that fails. Smoke-test every harness before a full run.

## Where we are
- exp05: contrastive margin (entropy-invariant) shows keyword/document priming is an
  entropy artifact (d_margin≈0); only an "extract" instruction helps (d_margin≈+0.27).
- exp06: the effect is a narrow semantic class — imperative *extract-salient-content*
  directives (extract/attend/summarize); `extract` is the single most reliable, and
  beats other imperatives by d+0.27. NOT one sentence, NOT "any instruction."
- Missing for publication: (1) a method (not prompt-hunting), (2) real task-accuracy
  gains (EM/F1, not a proxy), (3) a mechanism.

---

## PHASE 0 — De-risk: does the best intervention move REAL task accuracy?  [GATES ALL]

Rationale: we have intervention→margin. We have NOT shown intervention→accuracy.
exp04 showed keyword→EM was null (matching keyword→margin null). Prediction: if the
margin metric is valid, extract→margin(+) should give extract→EM(+). Testing this
validates the metric AND the intervention in one shot.

- **T0.1** Generation harness, consistent answer-elicitation format across conditions,
  paired Δ(primed − bare). bare/primed differ ONLY in the cache.
  - Smoke: 1 model / 1 dataset / 10 samples — verify non-degenerate generation, metrics compute.
- **T0.2** Full de-risk: 4 models (Qwen 1.5B, Qwen 7B, Gemma 12B, Ministral 8B) ×
  {SQuAD v2, HotpotQA, TriviaQA} × 300 samples. Conditions: bare, **extract** (test),
  comprehend (predicted-null ctrl), keywords (predicted-null ctrl). Metrics: EM, F1,
  contains; paired ΔEM / ΔF1 vs bare with bootstrap 95% CI.
- **GATE G0**: PASS if Δ(extract−bare) EM or F1 > 0 with CI excluding 0 on ≥2 models,
  AND keyword/comprehend Δ ≈ 0 (confirming margin predicts accuracy). 
  - PASS → Phase 1. FAIL → pivot to the metric/negative-result methods paper; the
    margin is clean but insufficient, and a prompt that only moves a proxy is not a method.

---

## PHASE 1 — Activation-space cache priming (centerpiece method)

Generalize "prepend a token prefix" to "add a steering vector to the residual stream
during document encoding." Elegant bonus: with no prefix, the document stays at
positions 1..D, so **no select/reposition step and no bf16 reposition error**.

- **T1.1** Extract the priming direction: over a TRAIN split of docs, capture residual
  activations at each layer for bare vs extract-prefixed encoding; v_ℓ = mean(h_extract −
  h_bare) at doc-token positions (try last-token and mean-over-doc variants).
  - Smoke: 5 docs, 1 model — verify per-layer shapes, norms, that v_ℓ is stable.
- **T1.2** Steering application via forward hooks: during bare encoding add α·v_ℓ to the
  residual at doc positions. Sweep α.
  - Smoke: 1 model / 20 samples — verify margin moves monotonically-ish with α; find sign/scale.
- **T1.3** Generalization: compute v on train docs, evaluate margin + EM on HELD-OUT docs
  and a held-out dataset (NaturalQuestions). Tune α on a val split only.
- **T1.4** Comparison: steering vs extract-prompt vs bare, on margin and EM.
- **GATE G1**: steering improves margin AND EM on held-out docs (ideally ≥ extract prompt,
  and generalizes across documents from a single learned direction). PASS → strong method
  result; proceed to mechanism + packaging. PARTIAL (per-doc needed) → frame as test-time
  cache optimization.

---

## PHASE 2 — Learned soft-prefix (alternative / upper bound; optional)

If steering's gains are small, learn the intervention directly.
- **T2.1** Differentiable two-phase pipeline (reposition is differentiable). Optimize P
  soft-token embeddings to maximize a contrastive margin loss (correct vs distractors,
  InfoNCE) on train docs.
  - Smoke: overfit 10 docs — loss decreases, margin rises.
- **T2.2** Train on N docs; test document-agnostic generalization; compare to steering/prompt.
- **GATE G2**: include only if it beats steering meaningfully (justifies the training cost).

---

## PHASE 3 — Mechanism (turns engineering into science)

Why does extract/steering help discrimination while comprehend/keywords don't?
- **T3.1** Attention analysis: does the intervention increase attention mass on
  answer-overlapping / high-IDF doc tokens during encoding, vs bare and comprehend?
- **T3.2** Value-norm analysis: does it amplify value-vector norms of content tokens
  (improving their retrievability by the Phase-B query)?
- **T3.3** Link: does the per-sample attention/value shift predict the per-sample margin
  gain? A positive correlation grounds the "salient-content amplification" hypothesis.
- Output: a mechanistic account that justifies the steering-vector construction.

---

## PHASE 4 — Packaging & robustness

- **T4.1** Full RAG accuracy eval of the best method across all models + held-out dataset.
- **T4.2** Composition with KV-cache compression: does the priming benefit survive
  int8/int4 quantization and H2O/SnapKV eviction? (Critical for practical relevance.)
- **T4.3** When-it-helps characterization (task type, model, baseline difficulty) and
  cost/benefit.

---

## Paper spine if gates pass
metric (entropy-invariant margin) → negative result (content priming is an entropy
artifact) → method (activation-space priming, learned without prompt search) →
mechanism (salient-content amplification) → task validation (RAG EM/F1) + composition
with compression.

## Test status
- [ ] T0.1 smoke  [ ] T0.2 full  [ ] G0
- [ ] T1.1 [ ] T1.2 [ ] T1.3 [ ] T1.4 [ ] G1
- [ ] T2.1 [ ] T2.2 [ ] G2
- [ ] T3.1 [ ] T3.2 [ ] T3.3
- [ ] T4.1 [ ] T4.2 [ ] T4.3
