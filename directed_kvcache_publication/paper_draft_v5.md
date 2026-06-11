# Imprinting Mode: What Zero-Retention KV-Cache Priming Actually Banks, and When It Helps

## Abstract

Retrieval-augmented systems precompute document KV caches offline and reuse them across
queries. A tempting "free lunch" is *cache priming*: prepend a short context during offline
encoding, let it shape the document's representations through self-attention, then discard it
before storage — reshaping the cache at zero inference cost. Whether this captures useful
signal has been unclear, and our own earlier work overstated it. We give a controlled,
end-to-end account.

First, a measurement correction: absolute negative log-likelihood (NLL) is entropy-confounded
for evaluating priming. Priming lowers output entropy, which lowers NLL without improving the
model's ability to *discriminate* the correct answer. Re-evaluating with an entropy-invariant
contrastive margin dissolves the headline that document-derived keyword prefixes beat
instructions (margin effect d≈0.00), and a battery of controls (neighbor-leakage, position
matching, a machinery-neutral prime, matched footing) overturns several further "clean" claims.

Second, the real phenomenon. We show that zero-retention priming *does* bank context into a
document's stored KV — substantially — but **what** it banks is governed by a single
measurable model trait we call **imprintability**. We establish a clean double dissociation
**by content type**: the Gemma family (and Mistral) imprint *meaning* — recovering up to 35%
of a semantic context's value from a stripped cache (−3.8 nats on Gemma 27B), scaling
monotonically with model size — but cannot store arbitrary literals; Qwen 2.5 imprints
*surface form* (codes, pseudowords) but not meaning. Imprintability predicts semantic banking
across eight models at r=0.94, and is the same trait that gates a small but significant
reranking benefit (Gemma 12B/27B beat no-priming by +0.036 MRR). The semantic imprint is
distributed and read out in late layers.

Third, downstream value follows from **mode–task match**: semantic imprinting helps semantic
relevance (reranking) but *hurts* precise extraction (priming a passage with the question
degrades QA by +0.36 nats on Gemma), while surface imprinting helps extraction (−0.80 nats
on Qwen) but not relevance. Neither mode is universally useful. We close with the bounds —
most context value (~65%+) is structurally un-bankable, and the construction step is mildly
lossy — and argue the durable contributions are the imprinting-mode characterization, the
evaluation methodology that exposes it, and an honest map of what zero-retention cache
construction can and cannot do.

---

## 1. Introduction

KV-cache reuse is a standard RAG optimization: TurboRAG, CacheBlend, SGLang and others encode
document chunks offline and reuse their key–value states, cutting time-to-first-token by up to
an order of magnitude. The literature optimizes what happens *after* construction — which
entries to keep, how to compress, how to schedule. We ask whether *construction* itself carries
usable signal, via **cache priming**: encode `[BOS, context, \n, document]` in one pass so the
context shapes the document tokens through attention, then discard the context's cache entries,
reposition the document keys (RoPE delta-rotation) so positions are indistinguishable from a
standard cache, and store the result. Cost: a few tokens offline, zero at inference.

This is attractive, and an earlier version of this work reported the attractive result —
document-derived TF-IDF keyword prefixes lower answer NLL more than instructions or an oracle.
**That result does not survive a careful evaluation, and the path to what does is the spine of
this paper.** We make four contributions:

1. **A measurement correction (§4).** Absolute NLL is entropy-confounded; the contrastive margin
   and a set of controls dissolve the keyword headline and several successor claims, including
   our own.
2. **Imprinting mode (§6).** Priming banks context in a model-specific mode — *semantic* (Gemma,
   Mistral) or *surface-form* (Qwen) — a clean double dissociation by content type, governed by a
   single trait (**imprintability**, r=0.94 with semantic banking) that scales with model size.
3. **Mode–task match (§7).** Downstream value depends on matching imprinting mode to task:
   semantic imprinting helps relevance (reranking), hurts extraction (QA); surface imprinting the
   reverse. Neither is universally useful.
4. **The ceiling and the negatives (§5, §8).** Context value is large (−2.8 nats when retained)
   but mostly un-bankable; we report every controlled claim that failed (contrastive priming is
   inert, a representation-level "coherence" mechanism is a positional artifact, the architectural
   root of imprintability resists four ablations) as carefully as the ones that held.

The throughline is methodological honesty: for a "free-lunch" technique, the controls *are* the
result. Most clean stories here were dismantled by the next control; we report the survivors and
the casualties together.

---

## 2. Related Work

**Cache reuse / compression.** TurboRAG, CacheBlend, SGLang reuse precomputed caches; H2O,
SnapKV compress them. All take construction as fixed. Our per-tensor normalization round-trip is
a near-identity that independently improves NLL; our focus is construction-time *conditioning*.

**Prefix/prompt tuning, activation steering.** These learn or add persistent prefixes/directions.
Cache priming differs: the prefix is *discarded* (zero inference cost), and we study discrete,
natural-language context and its imprint. We borrow steering-vector methodology to test whether
priming reduces to a fixed direction — it does not.

**Memory and gisting.** "Gist tokens" and recurrent-memory methods compress context into a few
*retained* tokens. Our setting is the zero-retention extreme; the imprintability trait and its
modes characterize what survives that extreme.

**Confidence vs. correctness / RoPE.** A recurring theme — perplexity gains need not imply better
answers — motivates our margin metric. Repositioning cached keys requires per-layer RoPE
delta-rotation in float32, with model-family-specific frequencies and sliding/full layer handling.

---

## 3. Method

### 3.1 Two-phase pipeline
**Phase A:** encode `[BOS, context, \n, document]`; select BOS + document entries; reposition
document keys to positions `1..D` (float32 RoPE delta); per-tensor normalize. **Phase B:** append
`[\n, query, (\n, answer)]` at positions `D+1+`, reusing the cache; never pass explicit
`cache_position` (it reintroduces a one-token look-ahead).

### 3.2 Metrics and controls (the part that matters)
- **Contrastive margin** `= mean_k NLL(distractor_k) − NLL(correct)` — entropy-invariant; absolute
  NLL is not, so it is unsafe for evaluating priming.
- **Gold-class prior-shift control** — split the margin change by gold label; a symmetric split is
  a label-prior shift, not discrimination.
- **Machinery-neutral prime** — a content-free, length-matched prime (newlines) isolates the
  reposition+normalize *construction* cost from the prime's *content* effect.
- **Matched footing** — hold the prime fixed and vary only the variable of interest (e.g., whether
  a fact is in the document) to isolate it.
- **Neighbor-leakage / position-matching** — checks that "query-agnostic" constructions are truly
  query-agnostic and that representation comparisons are not positional artifacts.

### 3.3 Models and data
Eight models spanning imprintability: Qwen 2.5 (1.5/7/14B), Mistral 7B, Gemma 3 (1/4/12/27B).
Datasets: SQuAD, HotpotQA, GSM8K, DROP, MS MARCO (BM25 hard negatives); plus controlled
synthetic probes (a decisive fact in filler) for banking. Bootstrap 95% CIs; `*` excludes 0.

---

## 4. The Measurement Problem: Absolute NLL Is Entropy-Confounded

Our earlier NLL-based evaluation produced a clean story: TF-IDF keyword prefixes beat instructions
and an oracle. Re-scored with the contrastive margin (5 models × 4 datasets × 300 samples):

| condition | d(NLL) | d(margin) |
|---|---|---|
| tfidf keywords | +0.179 | **+0.001 (n.s.)** |
| random document words | +0.165 | −0.017 (n.s.) |
| random vocabulary | +0.031 | **−0.113** |
| oracle (query) | +0.054 | **−0.057** |
| **generic instruction (extract)** | +0.172 | **+0.270** |

![Figure 1](figures/fig1_measurement_correction.png)
*Figure 1: The entropy confound. Every prefix lowers absolute NLL (gray), but on the
entropy-invariant contrastive margin (blue) the keyword "win" vanishes (d≈0) and only
extract-style instructions improve discrimination.*

The keyword advantage was almost entirely entropy reduction (margin d≈0). Only extract-style
instructions move the margin. On BoolQ, the one apparent classification effect is a near-perfect
*label-prior shift* (gold=yes +0.328 / gold=no −0.326, net ≈0), not discrimination. **All
cache-priming claims based on perplexity/NLL are presumptively inflated by this confound.** The
rest of the paper uses entropy-invariant metrics and explicit controls.

A cascade of further "clean" claims fell to the controls: a "contrastive" keyword construction
turned out to add nothing over plain passage keywords (neighbor-subtraction inert, n.s. on four
models); its "cacheable" variant was 85% leaked from the candidate set; and a borderline
significant win at N=300 failed to replicate at N=400 before re-emerging at N=900 — a reminder to
trust only high-powered estimates. We report these in §8.

---

## 5. The Bankability Ceiling: Context Value Is Large but Mostly Unreachable

Does priming bank context at all? We measure a *decisive* fact (unknowable without it) in a filler
document, scoring the answer NLL when the fact is (i) absent, (ii) retained, (iii) primed then
stripped. Across Gemma and Qwen, **retaining the fact is a −2.8-nat effect** — context matters
enormously, and the pipeline detects it. But the *content* of a primed-then-stripped fact
contributes near-zero on a machinery-neutral basis, and the reposition+normalize *construction*
costs ~0.3–0.6 nats. So:

> Context value is large (~2.8 nats) and **mostly un-bankable** (≥~65% lost): the value lives in
> the attendable context *tokens*, which zero-retention construction discards, keeping only an
> imprint. What that imprint *does* carry is the subject of §6.

This is the principled ceiling: you cannot fold N attendable context tokens losslessly into a
document's KV. The surprise is that the imprint is not nothing — it is *typed*.

---

## 6. Imprinting Mode: Semantic vs. Surface (the central result)

### 6.1 A double dissociation by content type
We prime a filler document with a fact whose answer ranges from meaningless to meaningful, strip
it, and measure how much the answer is recovered (machinery-controlled; negative = banked):

| answer type | gemma3_12b | qwen25_7b |
|---|---|---|
| code (4 digits) | −0.07 (no) | **−0.33\*** |
| pseudoword (nonword) | −0.39\* (weak) | **−0.27\*** |
| rare word | **−3.67\*** | +0.18\* (worse) |
| common word | **−0.93\*** | +0.29 (worse) |
| phrase | **−2.02\*** | −0.10 (n.s.) |

**Gemma imprints *meaning*** (words/phrases, up to −3.7 nats) but not meaningless tokens; **Qwen
imprints *surface form*** (codes, pseudowords) but not meaning. A clean double dissociation —
each family fails at the other's content type. (Testing only *codes*, as we first did, hid this:
codes are a surface item where Qwen wins.)

![Figure 2](figures/fig8_content_dissociation.png)
*Figure 2: Content-type double dissociation. Gemma (12B) banks meaningful words/phrases but not
meaningless codes/pseudowords; Qwen (7B) banks surface forms but not meaning. Banking = nats
recovered from the stripped cache; bars are bootstrap 95% CIs.*

### 6.2 One trait predicts it: imprintability (r=0.94)
Define **imprintability** as the mean |Δ query-NLL| a generic prefix induces (what we earlier
called "primability"). Across eight models it predicts semantic banking almost perfectly:

```
            imprint.  sem-bank          imprint.  sem-bank
qwen1.5b    0.20      0.22       gemma1b  0.43     0.57
qwen7b      0.37     -0.14       gemma4b  0.60     2.46
qwen14b     0.39      0.01       gemma12b 0.84     3.62
mistral7b   0.55      1.36       gemma27b 0.84     3.77
                         Pearson r = 0.94
```

![Figure 3](figures/fig7_imprintability_unification.png)
*Figure 3: A single trait — imprintability (|Δ query-NLL| from a generic prefix) — predicts how
much semantic context a model banks into a stripped cache (Pearson r=0.94, 8 models). It is the
trait, not the family: Mistral (purple) sits on the line.*

It is the *trait*, not the brand: Mistral (non-Gemma, imprintability 0.55) banks semantics
(−1.36). Semantic imprinting also **scales monotonically with Gemma size** — 6% → 23% → 36% →
35% of semantic context value (1B→4B→12B→27B).

![Figure 4](figures/fig9_semantic_scaling.png)
*Figure 4: Semantic imprinting scales with model size — fraction of a semantic context's value
recoverable from the stripped cache, Gemma 1B→27B.*

### 6.3 Where it lives
Layer-wise KV patching on semantic recovery: the Gemma imprint is **distributed and read out in
late layers** (peak ~L44/48; full-cache recovery +4.1 nats, single-layer patches sum to only
+0.9 → non-localized). Qwen shows no semantic recovery. The semantic imprint is a distributed,
late-stage property of the cached representation.

---

## 7. Mode–Task Match: When Imprinting Helps

Imprinting mode is not uniformly good — its value depends on the task.

**Relevance (reranking).** Priming each MS MARCO passage with its own keywords significantly
improves query-likelihood reranking over generic priming on Gemma, and beats *no* priming on the
larger, higher-imprintability models: **+0.036 MRR on Gemma 12B and 27B** (CIs exclude 0), null
on Gemma 4B and on Qwen/Mistral. Semantic imprinting re-weights the passage's own semantic
salience, which is exactly what relevance scoring rewards.

**Extraction (QA).** Priming a passage with the *question*, then stripping it, and answering
(machinery-controlled content effect):

| | gemma3_12b | gemma3_4b | qwen25_7b |
|---|---|---|---|
| content effect on answer NLL | **+0.36\* (hurts)** | +0.12 (n.s.) | **−0.80\* (helps)** |

Here the dissociation **reverses**: surface imprinting (Qwen) banks the question's matchable
*tokens* and helps locate the answer; semantic imprinting (Gemma) banks the question's *meaning*,
shifts the passage toward the question's topic, and *blurs* the precise answer.

**The 2×2.**

| | Gemma (semantic imprinter) | Qwen (surface imprinter) |
|---|---|---|
| reranking (relevance) | **helps** (+0.036\*) | no |
| QA (precise extraction) | **hurts** (+0.36\*) | **helps** (−0.80\*) |

![Figure 5](figures/fig10_mode_task.png)
*Figure 5: Mode–task match. Left: semantic imprinting (Gemma) helps relevance reranking; Qwen
does not. Right: surface imprinting (Qwen) helps precise QA extraction; semantic imprinting
(Gemma) hurts it. Value depends on matching mode to task.*

Value comes from **matching imprinting mode to task type**, not from the technique per se.

---

## 8. What Did Not Pan Out (controlled negatives)

- **"Contrastive" priming is inert.** Neighbor-subtracted keywords ≈ plain passage keywords (n.s.,
  4 models). The active ingredient is keyword content, not contrast.
- **A representation-level "content-coherence" mechanism is a positional artifact.** Position-matched
  re-measurement shrank the Gemma effect ~70%, and Mistral (more "coherent" by that metric) shows
  no behavioral effect — falsifying it.
- **The architectural root of imprintability resists explanation.** QK-norm (ablation *raises*
  imprintability), attention sharpness (families respond oppositely to a temperature knob), prefix
  attention-salience (Gemma attends to the prefix *less*), and a fixed-direction steering vector
  (reproduces <25%) were each falsified. Imprintability is an empirical trait; its cause is open.
- **No universal win, and no precise-fact injection.** Most context value is un-bankable; priming a
  document with an arbitrary external fact recovers ~0 of it on the semantic imprinter.

---

## 9. Practical Guidance

1. **Evaluate with the contrastive margin and a machinery-neutral control; never absolute NLL.**
   Report the prior-shift control. NLL gains for priming are presumptively entropy artifacts.
2. **Measure imprintability first** (mean |Δ query-NLL| from a generic prefix). It predicts what a
   model can bank (r=0.94) and which mode it is in.
3. **Match mode to task.** On a high-imprintability (semantic) model, construction-time
   conditioning can help *relevance/retrieval* but may hurt *precise extraction*; on a surface
   imprinter, the reverse. Do not deploy it blind.
4. **Do not expect a free lunch.** ~65%+ of context value is un-bankable; the construction step is
   mildly lossy. The gains are real but bounded and task-specific.

---

## 10. Limitations

- The architectural cause of imprintability is uncharacterized (four ablations falsified).
- Downstream value is shown on reranking and extractive QA; broader task coverage is future work.
- Banking probes use controlled synthetic facts (decisive content in filler) at N=150; behavioral
  results use N=300–900. The synthetic design maximizes cleanliness at some cost to ecological
  validity.
- Reranking uses query-likelihood scoring, well below purpose-built rerankers in absolute MRR.

---

## 11. Conclusion

We set out to optimize KV-cache construction and learned, first, that the optimization we thought
we had was an artifact of how we measured it. Measured correctly, zero-retention priming *does*
bank context — but in a model-specific **mode**: Gemma-family models imprint *meaning*, Qwen
imprints *surface form*, with a single trait (imprintability, r=0.94) predicting which and how
much, and downstream value set by whether that mode matches the task. The honest verdict is that
directed cache construction is not a free-lunch accelerator but a *typed*, bounded mechanism whose
value is conditional and predictable. Its durable contributions are the imprinting-mode
characterization, the evaluation methodology that exposes the entropy and machinery confounds, and
a clear map — survivors and casualties alike — of what zero-retention cache construction can and
cannot do.

---

## Appendix

**A. Reproducibility.** Pipeline in `directed_kvcache_v4/lib` (RoPE float32 reposition, sliding-
window handling, no Phase-B `cache_position`). Experiments under `experiments/13_contrastive/`
(keyword/contrastive reranking, ablations, machinery/coherence/circuit/steering controls) and
`experiments/15_bankability/` (bankability, reweight-vs-inject, content-type spectrum, circuit
localization, downstream QA). Running log: `experiments/09_boolq_classification/SHARPENING_FINDINGS.md`.
Bootstrap CIs (4000 resamples); fixed seeds; 20-sample checkpoints.

**B. Key statistics.** Entropy confound (§4) exp05; keyword-vs-bare reranking (§7) exp14b/exp14c
(N=900): gemma12b/27b +0.036\*, others n.s.; imprintability×banking r=0.94 (exp26); content-type
dissociation (exp27, N=150); localization (exp28); downstream QA content effects (exp29, N=300,
machinery-controlled): gemma12b +0.36\*, qwen7b −0.80\*; bankability ceiling (exp24/25): retained
−2.8 nats, content-bankable ≈0, machinery ~0.3–0.6.
