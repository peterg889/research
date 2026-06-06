# Where is cache-priming "sharpening" most important? — rigorous findings

After correcting the "entropy artifact" misframing (the margin genuinely moves, so
there IS real differential signal for `extract`), we asked *where* the signal has
practical value. Built confound-controlled experiments. The answer has three
coordinates, none of which is the "classification win" we first hypothesized.

## 1. Mechanism: content amplification, not abstract sharpening

BoolQ (binary doc-grounded classification) prior-shift control — Δmargin(extract−bare)
split by gold class:
```
extract   gold=yes Δmgn=+0.328   gold=no Δmgn=-0.326   -> PRIOR SHIFT (not sharpening)
```
A near-perfectly symmetric label-prior shift toward "yes", net ~0 on real
discrimination; all accuracy/calibration deltas null. So on abstract-label tasks
the intervention does NOT sharpen the correct answer — it nudges the label prior.

This reconciles with extractive QA, where `extract` lowered NLL on the correct span
(document content) while distractors (other docs' content) stayed flat. The real
mechanism is **document-content amplification**: priming makes *this document's
content* more retrievable. It helps when the answer IS document content; it cannot
discriminate an abstract label absent from the document.

## 2. Answer-space / task type

K-way MC accuracy gain (extract−bare), extractive QA (answer = content), pooled:
```
              K=2     K=4     K=8
Gemma 12B   +0.017  +0.023  +0.028     (real, grows with #competitors)
Qwen 1.5B   +0.003  +0.009  +0.019
Qwen 7B     +0.002  +0.004  +0.005
Mistral 7B  -0.000  -0.004  -0.005     (non-responsive)
Ministral   +0.000  -0.001  -0.003     (non-responsive)
```
Content-grounded tasks benefit (small, model-dependent); abstract classification
(BoolQ) does not. Gain grows with the number of competitors, implying the effect
is largest in many-candidate retrieval/reranking settings.

## 3. Difficulty: concentrated on the uncertain/boundary cases (the key result)

Δtop1 (extract−bare) by baseline-confidence tercile, extractive QA:
```
              low-margin (uncertain)   mid     high (confident)
Gemma 12B            +0.092           +0.000      -0.007
Qwen 1.5B            +0.043           +0.013      +0.003
Qwen 7B              +0.010           +0.005       0.000
Ministral 8B         ~0               ~0          ~0
```
The gain is almost entirely on the samples the model is unsure about. The small
*overall* gain (~3pp) is a dilution artifact: on the hard cases that the model
would otherwise get wrong, Gemma 12B gains **+9.2pp**.

## Deployable thesis

Directed cache priming is a **targeted, document-content-amplification intervention**
whose value is concentrated on (a) content-grounded answers (extraction/retrieval/
reranking, NOT abstract classification — there it is only a prior shift), (b)
instruction-tuned responsive models, and (c) the model's uncertain/boundary cases.
Use it in confidence-aware / selective RAG: prime the cache for the model's hard
content-retrieval queries. Honest caveat: averaged over all samples the effect is
small; it pays off specifically on the close calls.

## Methodological contributions (the durable part)
- Contrastive margin (entropy-invariant) + the gold-class margin-split (prior-shift)
  control are necessary to evaluate cache interventions; absolute NLL, raw accuracy,
  and even MC top-1 each mislead in a different way.
- Demonstration that keyword priming = entropy, oracle = partly entropy, instruction
  priming = real-but-content-specific amplification, and that abstract-classification
  "gains" are label-prior shifts.

## Status / pending
- exp10 BoolQ full run (4 models × 3270) in progress to confirm the prior-shift null
  across all models (n=600 single-model so far, very clean).
- Suggested next: many-candidate passage reranking (predicted to be where the effect
  is largest, per the grows-with-K trend) + selective-RAG accuracy@coverage on the
  uncertain stratum.

## Deep dive 1 — difficulty (sharpened, exp05 re-analysis)

Rescue/break decomposition (extract vs bare, 8-way MC top1):
```
model         bareAcc  rescued  broke  net   rescue-rate-on-errors
Gemma 12B      0.778     72      38    +34        27.1%
Qwen 1.5B      0.781     40      17    +23        15.2%
Qwen 7B        0.772     26      20     +6         9.5%
Mistral 7B     0.767     47      53     -6   (non-responsive)
Ministral 8B   0.810      9      13     -4   (non-responsive)
```
Continuous curve (Gemma 12B, Δtop1 by bare-margin decile): peaks at +15.8pp in
decile 1 (bareAcc 0.28, the "winnable" fence-sitters), +11.7pp in decile 0, and
~0 / slightly negative once bareAcc>0.85. Priming tips uncertain cases over the
line; does nothing once confident.

Selective-deployment simulation (prime only bottom-K% by bare confidence):
```
model        bareAcc  prime-all   best selective
Gemma 12B     0.778    +0.028     +0.036 @ 45% coverage  (beats prime-all, half the cost)
Qwen 1.5B     0.781    +0.019     +0.019 @ 55%
Qwen 7B       0.772    +0.005     +0.007 @ 45%
```
Recipe: prime the model's low-confidence queries, leave confident ones bare —
better accuracy than priming everything, because it avoids the break cost.
(Uses oracle confidence ordering; serve-time proxy to be validated.)

## Deep dive 2 — mechanism confirmed within-task (content amplification)

Δmargin (extract−bare) stratified by whether the gold answer appears in the passage:
```
GSM8K (answer often computed, not in passage):
  Gemma 12B:  answer-in-doc(13%) +0.969   |  not-in-doc(87%) -0.003
  Qwen 7B:    answer-in-doc      +0.124   |  not-in-doc       -0.155
SQuAD (100% in-doc): uniformly large positive (Gemma +1.22).
```
Same task, same model: priming helps iff the answer is document content. Decisive
support for the content-amplification mechanism. (Substring overlap is too coarse
for long TriviaQA passages.)

## Next experiment (motivated): task-aligned priming
If priming amplifies content, prime toward the content most RELEVANT to the task's
answer type (entities/dates/numbers for factoid QA). Test whether task-aligned
prefixes beat generic "extract", and whether label-relevant priming can create
REAL sharpening (gold-class split both positive) on otherwise-prior-shift abstract
tasks. Plus validate the selective recipe with a serve-time confidence proxy.

## Deep dive 3 — exploitation attempts FAIL (exp11, honest negatives)

Task-aligned priming (margin gain d vs bare, pooled 4 datasets):
```
model        generic  entity   qa_directed
Qwen 1.5B     +0.392  +0.396    +0.443
Qwen 7B       +0.189  -0.066    -0.045
Gemma 12B     +0.228  -0.268    -0.248
```
DECISIVELY FALSIFIED: on capable models the specific content-targeting prefixes are
HARMFUL; generic "extract the key facts" wins. Implication: the mechanism is NOT
instruction-following (entity-priming doesn't help entity-answers) but a low-level
BROAD salience cue. Specificity backfires.

Serve-time selective recipe (Gemma 12B, generic): the oracle-margin concentration
(+9pp on uncertain) does NOT survive a deployable signal. With serve-time next-token
entropy, Δtop1 is flat across strata (confident +0.033 / uncertain +0.037), so
selective-by-entropy ~= prime-all (no benefit). The boundary signal that makes
selective work only exists when you score candidates (MC/rerank/classification),
not in open generation.

## Overall honest conclusion
Cache priming = real, mechanism-clear broad document-content amplification (d~0.2),
best induced by a SIMPLE generic extract cue. But: small accuracy gains (~3pp),
content-grounded only (prior-shift on abstract classification), boundary-concentrated
but not exploitable via a simple serve-time signal. The exploitation hooks tried
(task-aligned prompts, activation steering, selective deployment) each fail to turn
the modest effect into a large win. The durable contribution is the rigorous
CHARACTERIZATION + the methodology (contrastive margin + gold-class prior-shift
control) showing what cache priming actually does and why prior NLL/accuracy-based
claims mislead.

## Deep dive 4 — reranking (the bounded application test) + UNIFYING CONCLUSION

MS MARCO v2.1, 10-way reranking by query-likelihood (1 relevant + 9 BM25 hard negs):
```
model         bare MRR/R@1     ΔMRR     ΔR@1     ΔR@3        uncertain-query ΔR@1
Qwen 1.5B     0.499 / 0.287    +0.020   +0.030   +0.017      +0.103   (selective +0.043 @40%)
Qwen 7B       0.523 / 0.323    -0.028   -0.020   -0.070*     -0.058   (priming HURTS)
Gemma 12B     0.447 / 0.233    -0.025   -0.033   -0.010      -0.039   (priming hurts)
```
Reranking is NOT a general win: priming helps only the smallest model (and only on
its uncertain queries, where the selective recipe gives +4.3pp R@1 with a deployable
signal). On capable models priming HURTS (Qwen 7B significant on R@3).

WHY: MS MARCO hard negatives are BM25-retrieved -> they share the query's vocabulary.
Content amplification is NON-SELECTIVE: it boosts content salience in the relevant
passage AND in the lexically-similar hard negatives, so against hard alternatives it
fails to improve (or hurts) discrimination. Contrast: priming widened the margin
against EASY random distractors (QA-rerank ceiling) but fails against HARD ones.

UNIFYING CONCLUSION across all experiments:
Cache priming is NON-SELECTIVE content amplification. It uniformly raises the
salience of the primed document's content, but it cannot sharpen the DISTINCTION
between a correct answer and a PLAUSIBLE alternative. Every practically valuable
task (real reranking, classification, hard QA) requires discriminating among close
alternatives -- exactly what priming does NOT do. This is the deep reason no
practical application materialized: the mechanism amplifies content but does not
discriminate, while the tasks that matter require discrimination.

The durable contribution is the rigorous CHARACTERIZATION (what cache priming does
and, mechanistically, why it can't deliver) + the evaluation METHODOLOGY
(contrastive margin + gold-class prior-shift control + hard-vs-easy distractor
contrast) that exposes it. Prior NLL/perplexity-based cache-construction claims
overstate value because they never test against hard, plausible alternatives.

---

# UPDATE (2026-06-06): the non-selectivity conclusion was too strong — selectivity is a PREFIX property, achievable on PRIMABLE models

The conclusion above ("cache priming is non-selective content amplification, full
stop") conflated two separable things: the non-selectivity of the *generic* prefix
with an inherent limit of the *method*. Two new experiments (exp14 contrastive
reranking, exp15 needle) separate them. Headline: **a CONTRASTIVE prefix makes the
amplification selective enough to genuinely improve discrimination — but only on a
model that is "primable" in the first place, which in our 6-model sweep is the
Gemma family.**

## exp14 — Contrastive / distinctive priming (MS MARCO reranking, N=300, 10-way)

Instead of priming each passage with generic "Extract the key facts" (non-selective),
prime it with what DISTINGUISHES it from competitors: top TF-IDF terms of the
passage minus its neighbors. Two variants, both length-matched to L=16:
 - `distinctive_corpus`  vs nearest CORPUS neighbors -> query-agnostic, CACHEABLE
 - `distinctive_cand`    vs the actual 9 candidates  -> ORACLE upper bound

Result (ΔMRR vs generic = does selectivity add value over naive priming?):
```
model         primability  dist_corp_selectivity  dist_corp ΔMRR-vs-generic  ΔMRR-vs-bare
gemma3_12b       0.844          +0.114*                 +0.053*                +0.028
gemma3_27b       0.842          +0.086*                 +0.060*                +0.030
mistral_7b       0.552          +0.039                  -0.016                 -0.030*
qwen25_14b       0.391          +0.007                  -0.026                 -0.025
qwen25_7b        0.370          -0.007                  -0.007                 -0.036
qwen25_1_5b      0.195          +0.028                  ~0                     +0.020
```
- **Primability** = mean |Δquery-NLL| caused by generic priming (how much priming
  moves the representation at all). It is a model-architecture trait: Gemma ~0.84,
  Mistral ~0.55, Qwen ~0.2-0.4. NOT monotonic in scale.
- **Selectivity** (decomposition): on Gemma, distinctive priming lowers the RELEVANT
  passage's query-NLL ~2.3x more than the negatives' (Δrel=-0.201 vs Δneg=-0.088 on
  12B) -> the relevant passage rises in rank. Generic priming instead pushes ALL
  query-NLLs up (degrades) and scrambles ranking.
- **The two highest-primability models (both Gemma) are exactly the two that show
  significant contrastive selectivity**, and the effect grew with Gemma scale.
  Mistral 7B (predicted non-responsive) and the Qwen ladder show none — distinctive
  priming does not help (or hurts).

This OVERTURNS the categorical claim: priming CAN sharpen distinction among close
alternatives, via a contrastive prefix, on a primable model. Selective amplification
is real discrimination (entropy would move relevant and negatives equally; here they
diverge, CIs exclude 0).

### Honest limits (why this is a mechanism result, not a deployable technique)
1. Even on Gemma, distinctive_corpus beats generic significantly but is only +0.028
   (ns) vs the NO-PRIMING baseline. "Make priming selective" beats "naive priming";
   it does not reliably beat "don't prime."
2. Scope is the Gemma family (high primability). Cross-family (Mistral) and the Qwen
   ladder do not show it.
3. Cache-priming reranking (MRR ~0.45-0.52) is dominated by purpose-built rerankers.

The durable contribution: a CONTROLLED MANIPULATION isolating selectivity as the
causal lever (same length/position/machinery, only prefix content changes -> sign
flips on primable models), plus the primability x selectivity framework explaining
WHEN priming can discriminate.

## exp15 — Long-context positional rescue (needle-in-a-haystack)

Different regime: the "competitor" to the gold content is the model's own POSITIONAL
attention decay (lost-in-the-middle), not a semantic hard negative. Hypothesis: since
nothing competing benefits from the same boost, non-selective amplification might help
here. A needle fact is buried at fractional position p in a filler haystack; we score
the needle's answer NLL (lower=better) bare vs extract-primed, vs p.
Qwen 1.5B/7B at 2000 tokens; Gemma 12B at 700 (sliding window caps context).

```
model         bare mid-penalty   Δmid(ext-bare)  Δends    verdict
qwen25_1_5b     -0.028 (none)       -0.292        -0.236   uniform NLL drop; no penalty to rescue
qwen25_7b       +0.257 (real LITM)  +0.204 worse  +0.106   priming AGGRAVATES the middle
gemma3_12b      -0.054 (none)       -0.492        -0.658   huge drop (primability) but helps ENDS more
```

Verdict: **no positional rescue on any model.**
- On the only model with a genuine lost-in-the-middle penalty (Qwen 7B, 2000 tok),
  priming makes the middle WORSE, not better.
- On primable Gemma, priming lowers absolute NLL a lot (the entropy/primability
  artifact) but concentrates the help at the ENDS (esp. position 0), i.e. where
  content is already well-attended — the opposite of rescue.
- The absolute-NLL drops do NOT track where position hurts -> they are the entropy
  artifact, not retrieval improvement. (Gemma caveat: 700 tokens is too short to
  induce a strong lost-in-the-middle, a limit imposed by the sliding window.)

## Net of both idea-tests
The non-selectivity story stands as the DEFAULT, with one important refinement:
selectivity is a prefix property a CONTRASTIVE prefix can supply, and it yields real
discrimination on PRIMABLE (Gemma-family) models. That is a clean mechanism result.
But no deployable production win emerged: contrastive priming beats naive priming yet
not the no-priming baseline, is Gemma-scoped, and is dominated by purpose-built
methods; positional rescue fails outright. The durable contributions remain the
rigorous characterization (primability x selectivity) and the evaluation methodology
(contrastive margin + gold-class prior-shift control + selectivity decomposition).

---

# DEEPEN + GENERALIZE (2026-06-06): the contrastive win is a Gemma-FAMILY (QK-norm) trait

Three workstreams to turn the contrastive finding into a defensible centerpiece.

## WS1 — Primability x selectivity across the FULL Gemma ladder (exp14, +3 models)
Added gemma3_1b/4b/4b-base to the 6-model sweep (now 9 models, MS MARCO N=300):
```
model            primability  dc_selectivity  ΔMRR(dcorp vs generic)   QK-norm
gemma3_1b           0.429        +0.071          +0.013                  yes
gemma3_4b           0.597        +0.115          +0.050*                 yes
gemma3_4b_base      0.207        +0.076          +0.059*                 yes  (BASE model)
gemma3_12b          0.844        +0.114          +0.053*                 yes
gemma3_27b          0.842        +0.086          +0.060*                 yes
mistral_7b          0.552        +0.039          -0.016                  no
qwen25_1_5b         0.195        +0.028          -0.004                  no
qwen25_7b           0.370        -0.007          -0.007                  no
qwen25_14b          0.391        +0.007          -0.026                  no
```
Findings:
- **Every Gemma model has positive contrastive selectivity (+0.07..+0.12); the win
  (dcorp vs generic) is significant for ALL Gemma >=4B, instruct AND base. No other
  family shows it.** So contrastive priming is a Gemma-family phenomenon, not scale- or
  instruction-gated. (The earlier n=10 smoke hint that 1B fails was noise; at N=300 1B
  has the selectivity but too little primability to move MRR -> primability threshold ~0.43.)
- **Base vs instruct (4B) dissociates the two axes:** instruction-tuning ~3x's the
  primability MAGNITUDE (0.207 -> 0.597) but selectivity is already present in the base
  model (+0.076, win +0.059*). => the ARCHITECTURE supplies selectivity; instruction-
  tuning amplifies magnitude.

## WS3 — Why is Gemma primable? (architecture)
Resolved AutoConfig + model-class features. The feature that cleanly separates the
primable family from the rest:
```
                  QK-norm  head_dim  hybrid-attn   primability   dc_selectivity
Gemma 3 (all)      YES      256/128   YES           0.58 (mean)   +0.092 (mean)
Qwen2.5 / Mistral  NO       128       NO            0.38 (mean)   +0.017 (mean)
```
- **QK-norm** (RMSNorm applied to Q and K per head) is present in every Gemma 3 size and
  absent in Qwen2.5 and Mistral. It renormalizes per-head q,k magnitudes, plausibly
  making attention more responsive to context conditioning (the prefix) -> higher
  primability and, with the contrastive prefix, selective amplification.
- Gemma also has large head_dim (256 for <=12B), (1+w) RMSNorm, embedding x sqrt(d), and
  hybrid local/global attention; QK-norm is the cleanest single differentiator.
- HONEST: correlational (3 families, no ablation). A causal test = toggle QK-norm off in
  a Gemma forward and re-measure primability (model surgery; future work).
Reproduce: `experiments/13_contrastive/primability_architecture.py`.

## WS2 — Replication on a 2nd benchmark: HotpotQA (distractor)
Different corpus + natural multi-hop queries + pre-built hard negatives (incl. the
co-supporting paragraph). Single relevant = answer-bearing paragraph. N=300, 10-way.
```
model        bare   gen   dcorp   ΔMRR(dcorp vs bare)   ΔMRR(dcorp vs generic)
gemma3_4b    0.503  0.458  0.477   -0.026*               +0.019    (ns; weaker here)
gemma3_12b   0.521  0.474  0.520   -0.001                +0.046*   REPLICATES
gemma3_27b   0.527  0.506  0.534   +0.007                +0.027*   REPLICATES
qwen25_7b    0.553  0.533  0.546   -0.007                +0.013    (control: none, correct)
```
- The contrastive win REPLICATES on HotpotQA for the flagship Gemma models
  (12B +0.046*, 27B +0.027* vs generic), confirming it is NOT MS-MARCO-specific. The
  qwen25_7b control correctly shows no effect.
- The SAME honest pattern holds on both benchmarks: generic priming HURTS vs bare on
  primable Gemma (non-selective degradation); contrastive priming RECOVERS that loss
  (significantly beats generic, returns to ~bare) but does NOT exceed no-priming.

## CONSOLIDATED VERDICT (deepen+generalize)
1. Contrastive priming significantly beats generic priming on PRIMABLE models, across
   model scales (Gemma 4B-27B, base+instruct), 2 benchmarks (MS MARCO, HotpotQA), and
   the cacheable variant. Absent in Qwen/Mistral + control. Robust, well-bounded.
2. Mechanism: selective amplification (relevant query-NLL drops ~2.3x more than
   negatives'); architectural correlate = QK-norm (Gemma 3) -> primability.
3. Practical bound (unchanged, honest): contrastive priming RECOVERS the harm that
   naive priming causes; it does not beat the no-priming baseline. So the deployable
   claim is narrow: "IF you prime a reused cache (the RAG precompute premise) on a
   Gemma model, make the prefix contrastive, not generic — it avoids the damage."
   It is a mechanism/characterization result + a priming-hygiene recommendation, not a
   standalone technique that beats not-priming.

## WS3 CAUSAL TEST (exp17) — QK-norm hypothesis REFUTED
Ran the causal test the correlation invited: monkeypatch gemma3_4b to replace all 68
q_norm/k_norm RMSNorms with identities; re-measure primability (MS MARCO, 600
query-passage pairs, N_Q=60).
```
                    intact   ablated(QK-norm off)   change
primability(|Δnll|)  0.615    0.662                 +8%   (UP, not down)
repr_primability     0.362    0.749                 +107% (DOUBLED)
bare_nll (health)    4.974    7.760                 +2.79 nats (model degraded)
```
Disabling QK-norm does NOT reduce primability — it INCREASES it (and doubles the
representation-level perturbation of doc tokens by the prefix), while degrading the
model. So QK-norm is NOT the cause of Gemma's high primability; mechanistically it
REGULARIZES attention (removing it lets attention saturate -> doc tokens absorb the
prefix more strongly). The convenient correlational story (Gemma has QK-norm, Qwen/
Mistral don't, Gemma is primable) does NOT survive the causal test.

CORRECTED WS3 conclusion: primability is a robust, replicated GEMMA-FAMILY property
(all 5 Gemma models, base+instruct), but its architectural cause is NOT QK-norm and
remains OPEN (candidates not yet tested: embedding x sqrt(d) scaling, (1+w) RMSNorm,
hybrid local/global attention, or a training/distillation effect). This does not
affect the core result (contrastive priming helps primable Gemma models, replicated
across scales + 2 benchmarks); only the bonus mechanistic attribution is retracted.
Reproduce: `experiments/13_contrastive/ablate_qknorm.py`.

## WS3 DEEPER (exp18 att-temp, exp19 prefix-mass) — primability has NO single attention-side cause
After QK-norm was refuted, two more interventional/descriptive probes to pin down the cause:

(A) Attention-TEMPERATURE sweep (gentle, non-breaking: scale attn logits by 1/tau),
    gemma3_4b vs qwen25_7b, MS MARCO 300 pairs:
```
tau (sharp<-->soft)   0.5    0.7    1.0    1.5    2.0
gemma3_4b primability  0.68   0.55   0.57   0.49   0.40   (sharper => MORE primable)
qwen25_7b primability  0.35   0.32   0.37   0.42   0.58   (softer  => MORE primable)
```
The families respond in OPPOSITE directions -> there is NO universal "sharpness" knob.
The cross-over is real and survives normalizing by baseline NLL (Gemma/Qwen normalized
primability ratio = 1.51x @tau0.5, 1.29x @tau1.0, 0.58x @tau2.0). So the gap is not
merely Gemma's higher absolute NLL.

(B) Prefix-attention-mass (eager attn, N=20): does Gemma attend to the prefix more?
```
            prefix_mass  (x uniform)  BOS_sink  doc_attn_entropy
gemma3_4b     0.042       0.23x        0.479      1.73
qwen25_7b     0.056       0.32x        0.470      1.75
```
NO — Gemma attends to the prefix LESS than Qwen, both well BELOW uniform, with
near-identical attention entropy and the same ~47% BOS-sink. So primability is not
driven by direct doc->prefix attention or by Gemma having sharper/peakier attention;
the prefix's effect is INDIRECT.

CONCLUSION (honest): primability is a robust, replicated empirical GEMMA-FAMILY property
(~1.3x Qwen at natural settings, normalized), but it does NOT reduce to any single
obvious attention-side mechanism. Three plausible hypotheses were FALSIFIED:
  1. QK-norm        -> removing it INCREASES primability (regularizer, not cause)
  2. attn sharpness -> Gemma & Qwen respond OPPOSITELY (no universal knob)
  3. prefix salience-> Gemma attends to the prefix LESS, entropy ~identical
The families differ qualitatively in how context conditioning propagates (opposite
temperature response); a full reduction would need circuit-level analysis (future work).
This does NOT affect the core contrastive result. Scripts: ablate_attention_temp.py,
prefix_attention_mass.py.
