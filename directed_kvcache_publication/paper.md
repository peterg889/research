# Content Imprinting in KV Caches: Token Presence, Structure, and the Role of Instruction Tuning

## Abstract

A document's key–value (KV) cache can be precomputed once and reused across queries, and it can be
*shaped* at encoding time by prepending a short context that is discarded before the cache is stored
— reshaping the representation at zero inference cost. We ask a basic question about this operation:
what does a discarded prefix actually leave behind in the cache? Across eleven instruction-tuned
models spanning seven families, we find that priming banks recoverable content into a document's
stored KV, and that a single model trait — *imprintability*, the mean shift a generic prefix induces
in query likelihood — predicts *how much* a model banks (Pearson r=0.94). The *kind* of content
banked is a separate axis, exposed by a word-order shuffle control: some models bank **token
presence** (banking is order-invariant, so a scrambled prefix works as well as an ordered one),
others bank **relational structure** (scrambling destroys it), and a third group banks little. This
token-presence-vs-structure distinction is a property of the *instruction-tuned* model, not its
architecture: five architectural accounts fail to predict imprintability, whereas comparing
pretrained base models to their instruction-tuned versions shows that alignment training sets — and
can invert — which content type is banked. Most of a context's value is nonetheless unbankable into
zero retention (a ceiling of roughly two-thirds), and the imprint is read out in a model's late
layers. The imprinting style is directly observable and downstream-relevant: it governs whether
priming a passage with its own keywords improves query-likelihood reranking. Our results give a
compact, measurable account of how instruction tuning shapes the content a model encodes into its
KV cache, and of what cache-compression methods can and cannot expect to preserve on a given model.

---

## 1. Introduction

Retrieval-augmented and long-context systems increasingly treat the KV cache as a reusable artifact:
document chunks are encoded offline and their key–value states are concatenated at inference to cut
time-to-first-token [@turborag; @cacheblend; @sglang]. This makes the cache itself an object worth
studying — not just *how much* of it to keep [@h2o; @snapkv], but *what it encodes* and how that
varies across models.

We study a minimal operation for shaping a cache at construction time, which we call **priming**:
encode `[BOS, context, document]` in a single pass so the context influences the document tokens
through self-attention, then discard the context's cache entries and reposition the document keys so
the stored cache is indistinguishable in shape from a standard one. The cost is a few extra tokens
offline and nothing at inference. Priming is a clean probe because its effect is entirely mediated by
what self-attention writes into the *retained* document representation — the prefix is gone.

The question we answer is: **what does a discarded prefix leave behind?** We find three things.

1. **Priming banks recoverable content, and one trait predicts how much.** Priming a document with a
   fact makes that fact recoverable from the stored cache even after the fact's tokens are removed.
   The magnitude varies by nearly an order of magnitude across models, and is predicted almost
   perfectly (r=0.94) by a single, cheaply measured trait we call *imprintability* (§4).

2. **Models bank different *kinds* of content.** A word-order shuffle control separates banking of
   *token presence* from banking of *relational structure*. Across eleven models we find a three-way
   split — token-presence imprinters (e.g. Gemma, Falcon-3, Yi), structure imprinters (e.g. Mistral,
   Llama-3, OLMo-2), and weak imprinters (the Qwen family) — and this *kind* axis is separable from
   the *magnitude* axis (§5).

3. **Instruction tuning sets it.** Imprintability resists architectural explanation: five
   architecture-based accounts fail to predict it. It is instead set in alignment training — every
   pretrained base model imprints content, and instruction tuning amplifies, preserves, or inverts
   which content type is banked (§6).

We bound the effect (most context value is unbankable into zero retention; §7.1), localize it (a
distributed, late-layer property; §7.2), and show it is downstream-relevant (it predicts a
query-likelihood reranking benefit; §8). Together these give a compact account of how instruction
tuning shapes what a model writes into its KV cache — with a direct implication for cache
compression, since what a method can preserve on a given model depends on that model's imprinting
style.

---

## 2. Related Work

**KV-cache reuse and compression.** Precompute-and-reuse systems encode chunks offline and
concatenate caches at inference [@turborag; @sglang], with quality-recovery methods that selectively
recompute tokens [@cacheblend; @cacheclip]. A large literature compresses a cache after construction
by evicting or reallocating entries [@h2o; @snapkv; @adakv; @kvzip] or by distilling context into a
few retained slots — gist tokens [@gisttokens], in-context autoencoders [@icae], learned cache
vectors [@kvdistill], and task-aware compression [@beyondrag; @cartridges]. These methods change
*which* entries survive; we instead ask what a discarded prefix writes into the entries that are
*kept*, and how much of a context is recoverable from a stripped cache at all — a training-free
measurement of the limit these methods operate under, and one that turns out to be model-specific.

**Continuous prompts and steering.** Prefix- and prompt-tuning learn continuous prefixes that
*persist* at inference [@prefixtuning; @prompttuning], and activation steering adds a fixed direction
to the residual stream [@caa]. Priming differs on both counts: the prefix is discrete natural
language and is discarded before storage. We use the steering-vector methodology as a control, to
test whether priming's effect reduces to a single fixed direction — it does not (§6).

**Mechanistic and instruction-tuning analysis.** We localize the imprint with activation patching in
the causal-tracing lineage [@rome]. That the imprint is read out in late layers is consistent with
evidence that in-context and task processing concentrate in middle-to-late layers
[@wheredoesicl; @layerbylayer]. Most directly, work relating in-context learning to instruction
tuning — that ICL reshapes hidden states akin to implicit tuning [@iclimplicitit] and that
instruction tuning reshapes middle-layer representations [@layerbylayer] — frames our central causal
result: instruction tuning sets which content type a model imprints into its cache.

**Long-context evaluation.** Studies of long-context behavior show that placement and surface
statistics strongly shape outcomes and that likelihood need not track task quality [@lostinmiddle].
We adopt an evaluation control in this spirit: a content-free "machinery" baseline that separates the
prefix's *content* effect from the mechanical cost of the reshaping itself (§3).

---

## 3. Method

**Priming pipeline.** Given a document `D` and a context prefix `C`, we encode `[BOS, C, \n, D]` in
one forward pass, select the BOS and document entries, reposition the document keys from their
encoded positions to `1..|D|` via an exact RoPE delta-rotation (float32), and per-tensor normalize
the cache. The result is a cache for `D` alone, of standard shape, that carries whatever `C` wrote
into `D`'s keys and values through attention. Scoring appends a query at positions `|D|+1…` with no
explicit cache-position override, which we found necessary to avoid a one-token look-ahead through
the causal mask.

**Banking measure and the machinery control.** To measure what a prefix banks, we prime a filler
document `D` with a fact `F` (e.g., *"The `city` office specializes in `topic`."*), strip `F`, and
measure the negative log-likelihood (NLL) of the fact's answer given a question about it. Because the
reposition-and-normalize step itself perturbs the cache, we subtract a **machinery-neutral** control:
the same pipeline run with a content-free, length-matched prefix (newlines). The banking magnitude
is `strip(F) − strip(neutral)`; negative means the fact is banked. All differences are reported with
bootstrap 95% confidence intervals (4000 resamples).

**The shuffle probe.** To ask *what kind* of content a prefix banks, we prime with the same tokens in
natural order versus a deterministic shuffle that preserves the token multiset but destroys word
order and relational structure. Define `ORDER = banking(ordered) − banking(shuffled)`. `ORDER ≈ 0`
means banking is carried by **token presence** (order is irrelevant); `ORDER < 0` means **structure**
is load-bearing (scrambling destroys the banking). We run two versions: a single-fact measure, and a
harder two-fact binding measure that primes two `city→topic` facts and asks about one, so that
recalling the right answer requires the city–topic association rather than mere token presence.

**Models and data.** We study eleven instruction-tuned models across seven families: Gemma 3
(1/4/12/27B), Qwen 2.5 (1.5/7/14B), Mistral-7B, Ministral-8B, Llama-3-8B, Yi-1.5-9B, Falcon3-7B, and
a DeepSeek-R1-distilled Qwen-7B; plus three pretrained base models (Gemma-3-4B, Qwen2.5-7B,
Mistral-7B) for the causal analysis (§6). Every RoPE-reposition adapter is validated against the
model's own rotary embedding (relative error < 1e-4) before use. Fact-banking probes use controlled
synthetic facts embedded in filler passages (N=150); the reranking evaluation uses MS MARCO (N=900).

---

## 4. Priming Banks Content, and One Trait Predicts How Much

Priming a filler document with a fact and then stripping the fact makes the answer measurably more
recoverable from the stored cache — the machinery-controlled banking magnitude is negative and
significant on most models, and it varies widely: from near zero to −3.8 nats.

This variation is captured by a single model trait. Define **imprintability** as the mean absolute
change in query NLL that a *generic* instruction prefix induces on a document, `mean |Δ NLL|` — a
cheap, content-agnostic measure of how strongly a prefix perturbs a model's document representation.
Across eight models spanning the imprintability range, imprintability predicts banking magnitude
almost perfectly (Pearson **r = 0.94**; Figure 1). It is a property of the model, not the family:
Mistral (imprintability 0.55) banks substantially and sits on the same line as the Gemma models. In
the Gemma family, banking magnitude also rises monotonically with scale, recovering up to about a
third of a meaningful context's value from the stripped cache at 12–27B.

![Figure 1](figures/fig7_imprintability_unification.png)
*Figure 1: A single trait — imprintability (mean |Δ query-NLL| from a generic prefix) — predicts how
much content a model banks into a stripped cache (r=0.94, eight models). Mistral (purple) lies on the
line, so the relationship tracks the trait, not the family.*

Imprintability tells us *how much* a model banks. It does not tell us *what* — a question the
magnitude alone cannot answer, because a single primed fact can be recalled from the mere presence of
its answer token in the cache. We turn to that next.

---

## 5. What Is Banked: Token Presence vs. Structure

Applying the shuffle probe (§3) across the eleven models exposes a three-way taxonomy of *what* a
model banks, with multiple independent families on each side (Table 1, Figure 2).

| model | family | banking magnitude | single-fact ORDER | binding ORDER | kind |
|---|---|---|---|---|---|
| Gemma-4B | Gemma | −2.46\* | −0.20 (n.s.) | −0.51\* | token presence |
| Gemma-12B | Gemma | −3.62\* | −0.28 (n.s.) | +1.43\* | token presence |
| Gemma-27B | Gemma | −3.77\* | +0.47\* | +0.23 (n.s.) | token presence |
| Falcon3-7B | Falcon | −2.80\* | −0.01 (n.s.) | +0.10 (n.s.) | token presence |
| Yi-1.5-9B | Yi | −0.75\* | +0.35\* | +0.91\* | token presence |
| Mistral-7B | Mistral | −1.36\* | −1.55\* | −1.07\* | structure |
| Ministral-8B | Mistral | −0.89\* | −0.80\* | −0.24\* | structure |
| Llama-3-8B | Llama | −0.59\* | −0.49\* | −0.17\* | structure |
| OLMo-2-7B | OLMo | −0.32\* | −0.38\* | −0.10 (n.s.) | structure |
| Qwen-7B | Qwen | +0.14 (n.s.) | −0.16 (n.s.) | +0.12 (n.s.) | weak |
| DeepSeek-R1-Qwen-7B | Qwen | +0.07 (n.s.) | 0.00 (n.s.) | +0.04 (n.s.) | weak |

*Table 1: What each model banks. Banking magnitude and single-fact/two-fact-binding ORDER (both in
nats); negative ORDER = structure (scrambling destroys banking), ≈0 = token presence. \* denotes a
bootstrap 95% CI excluding zero.*

- **Token-presence imprinters — Gemma, Falcon-3, Yi (three independent families).** These models bank
  with substantial magnitude but *order-invariantly*: scrambling the primed tokens leaves the banking
  intact or even increases it. The clearest case is Falcon3-7B, which banks strongly (−2.80 nats,
  comparable to Gemma-4B) yet is exactly order-neutral on both probes (−0.01 and +0.10, both n.s.).
  For these models the imprint is lexical — the salient answer token is written into the retained
  representation regardless of its context — not relational meaning.

- **Structure imprinters — Mistral, Ministral, Llama-3, OLMo-2 (four families).** These models bank
  order-*dependently*: scrambling destroys the banking (single-fact ORDER −1.55 / −0.80 / −0.49 /
  −0.38, all significant), and for Mistral a scrambled prefix even *anti-*banks. Mistral, Ministral,
  and Llama-3 show this on both the single-fact and the two-fact binding probe; OLMo-2 shows it on
  the single-fact probe but is too weak to bind two facts.

- **Weak imprinters — the Qwen family.** Both instruct Qwen-7B and a DeepSeek-R1-distilled Qwen-7B
  bank little of either kind. That the same backbone stays a null imprinter under a very different
  (reasoning-distillation) fine-tune suggests the pretrained backbone bounds imprinting.

![Figure 2](figures/fig12_shuffle_controls.png)
*Figure 2: The shuffle control across eleven models. ORDER = banking(ordered) − banking(shuffled);
≈0 is token presence (order-invariant), <0 is structure (order matters). Token-presence imprinters
(Gemma, Falcon-3, Yi) are order-invariant; structure imprinters (Mistral, Ministral, Llama-3,
OLMo-2) are order-dependent; the Qwen models bank little. Bars are bootstrap 95% CIs.*

**Magnitude and kind are separable axes.** OLMo-2 banks *weakly but structurally* (−0.32 nats,
order-dependent), whereas Falcon-3 banks *strongly but lexically* (−2.80 nats, order-invariant). A
consistent reading is that imprintability (§4) measures the strength of a model's content-token
imprint — how aggressively it abstracts a prefix into the kept representation — and that this
token-level imprint can trade off against literal structure, which is why the most imprintable models
(Gemma-12B, Yi) bank a *shuffled* two-fact prime at least as well as an ordered one. There is, in
short, no single "how a model encodes context" axis: models differ both in how much and in what kind.

---

## 6. Instruction Tuning Sets the Imprinting Kind

Two lines of evidence show that a model's imprinting style is a property of its *instruction tuning*,
not its architecture.

**Architecture does not predict imprintability.** We tested five architecture-based accounts and none
predicts a model's imprintability: query–key normalization (ablating it *raises* imprintability),
attention temperature (families respond in opposite directions to a sharpening knob), prefix
attention salience (high-imprintability Gemma attends to the prefix *less*), reducibility to a single
fixed steering direction (a contrastive steering vector reproduces under a quarter of the effect),
and residual-stream norm (falsified by Mistral, which has the most explosive residual stream yet high
imprintability). The failure of every architectural account is itself the clue: the cause lies in
training.

**Base-vs-instruct comparison localizes it to tuning.** Comparing pretrained base models to their
instruction-tuned versions on the banking probe (Table 2) shows two things. First, *every pretrained
base model banks the meaningful-content target* (−0.7 to −1.1 nats) — content imprinting is a
universal property of pretrained language models, not a quirk of one family. Second, *instruction
tuning changes which content type is banked*, and can invert it: Qwen 2.5's alignment destroys the
meaningful-content banking (−0.72 → +0.14) and creates literal-code banking (+0.17 → −0.37), while
Gemma's and Mistral's tuning preserve and amplify meaningful-content banking. The banked content type is thus a
trainable property, set in alignment — which is exactly why every architectural ablation failed.

| | code banking | meaningful-content banking |
|---|---|---|
| Gemma-4B base | +0.31\* (anti) | −1.11\* |
| Gemma-4B instruct | +0.15\* (anti) | −2.46\* |
| Qwen-7B base | +0.17\* (anti) | −0.72\* |
| Qwen-7B instruct | −0.37\* (banks code) | +0.14 (none) |
| Mistral-7B base | −0.16\* (banks code) | −1.08\* |
| Mistral-7B instruct | −0.56\* (banks code) | −1.36\* |

*Table 2: The banked content type is set by instruction tuning. Positive code-banking values are
significant anti-banking (priming raises the code's NLL). Qwen 2.5's tuning uniquely inverts the
content type banked. \* = CI excludes zero.*

![Figure 3](figures/fig11_base_vs_instruct.png)
*Figure 3: Every pretrained base model (gray) banks the meaningful-content target; instruction tuning
amplifies it for Gemma and Mistral but, for Qwen 2.5, destroys it (left) and creates literal-code
banking (right) — a change in the banked content type.*

---

## 7. Bounds and Locus

**7.1 The bankability ceiling.** Zero-retention priming banks *some* content, but most of a context's
value is unrecoverable once its tokens are discarded. Retaining a fact in the cache is worth about
−2.8 nats across models, but the machinery-controlled banking of a primed-then-stripped fact is near
zero on average and reaches its maximum of a few nats only on the most imprintable models — so the
construction step recovers at most roughly a third of a context's value, and typically much less. The
value lives in the attendable context tokens, which zero-retention construction discards; only an
imprint remains. This is a principled ceiling on how much any method can fold a context into a
document's cache without retaining tokens.

**7.2 Where the imprint lives.** Layer-wise KV patching localizes the imprint to a model's *late*
layers: on a high-imprintability Gemma model, full-cache recovery is +4.1 nats while single-layer
patches sum to only +0.9, and the peak is near layer 44 of 48 — the imprint is a distributed,
late-stage property of the cached representation rather than a single localized edit. Weak imprinters
show no such recovery.

---

## 8. Consequence: The Imprinting Style Predicts a Reranking Benefit

The imprinting style is not merely diagnostic; it predicts a downstream effect. In query-likelihood
reranking on MS MARCO, priming each candidate passage with its *own* top keywords significantly
improves ranking on the high-imprintability Gemma models — **+0.036 MRR** and **+4.0 points of top-1
rank accuracy** on Gemma-12B and 27B (both with CIs excluding zero; hit@3 +5.4 and +4.6), and null on
lower-imprintability Gemma-4B and on the Qwen and Mistral models. Consistent with the token-presence
character of Gemma's imprint (§5), the benefit survives shuffling the keyword tokens: the top-1
accuracy gain is preserved or strengthened under a scrambled keyword prefix (+4.7 and +6.9 points on
12B/27B). Keyword priming re-weights *which of a passage's own tokens are salient* — precisely what
relevance scoring rewards — and does so without needing the keywords in any particular order.

The general implication is for cache compression: what a method can preserve on a given model depends
on that model's imprinting style. A budget that keeps "the important tokens" preserves content that a
token-presence imprinter banks lexically, but is a poorer fit for structure-imprinted content;
methods validated on one model family may not transfer.

---

## 9. Limitations

The token-presence-vs-structure split rests on two shuffle probes over eleven models with multiple
families on each side, but we do not yet explain *why* a given family lands where it does — the
causal analysis (§6) localizes the effect to instruction tuning without identifying which alignment
objective is responsible; a controlled fine-tuning study is future work. The magnitude–imprintability
correlation (r=0.94) is measured on the eight models for which we ran the imprintability probe;
extending it to all eleven — in particular to the added token-presence exemplars Falcon-3 and Yi —
would further strengthen the magnitude claim. OLMo-2's structure
classification rests on the single-fact probe, since it is too weak an imprinter to bind two facts.
The banking probes use controlled synthetic facts embedded in filler at N=150, a design that
maximizes cleanliness at some cost to ecological validity; the reranking evaluation uses
query-likelihood scoring, which is below purpose-built rerankers in absolute MRR. Downstream evidence
is shown for reranking; the broader downstream consequences of the imprinting style — in particular
its interaction with token-eviction cache compression — are a natural next step.

---

## 10. Conclusion

A document's KV cache can be shaped at construction time by a prefix that is thrown away, and what
that prefix leaves behind is a compact window onto how a model encodes content. Priming banks
recoverable content whose *magnitude* is predicted by a single trait (imprintability, r=0.94) and
whose *kind* falls into a small taxonomy — token presence, structure, or little — that is set by
instruction tuning rather than architecture. Most of a context remains unbankable into zero
retention, and the imprint is a distributed, late-layer property that nonetheless governs concrete
downstream behavior. Beyond characterizing a zero-cost cache-shaping operation, these results give a
measurable account of an under-examined effect of instruction tuning — it changes not only what a
model *does* but what it *writes into its cache* — and a caution for cache compression, whose reach
on a given model is bounded by that model's imprinting style.

---

## Appendix

**A. Reproducibility.** The priming pipeline uses a float32 RoPE delta-reposition with per-layer-type
inverse frequencies validated against each model's rotary embedding, sliding-window-aware cache
selection, and no explicit Phase-B cache position. All reported numbers regenerate deterministically
from cached per-sample results with fixed seeds and 4000-resample bootstraps.

**B. Key statistics.** Imprintability × banking magnitude, r=0.94 (eight models). Shuffle controls,
N=150, eleven models: single-fact ORDER — token presence Gemma-4B/12B −0.20/−0.28 (n.s.), Gemma-27B
+0.47\*, Falcon3 −0.01 (n.s., magnitude −2.80\*), Yi +0.35\*; structure Mistral −1.55\*, Ministral
−0.80\*, Llama-3 −0.49\*, OLMo-2 −0.38\*; weak Qwen-7B −0.16 (n.s.), DeepSeek-Qwen 0.00. Two-fact
binding ORDER — Gemma-12B +1.43\*, Yi +0.91\*, Mistral −1.07\*, Ministral −0.24\*, Llama-3 −0.17\*.
Base→instruct: Qwen meaningful-content −0.72→+0.14, code +0.17→−0.37. Bankability ceiling: retained
−2.8 nats, content-bankable ≈0, machinery ~0.3–0.6 nats. Localization: peak layer ~44/48, full
recovery +4.1 nats, single-layer sum +0.9. Reranking (MS MARCO, N=900): Gemma-12B/27B +0.036\* MRR,
+4.0\* top-1 accuracy, preserved under keyword shuffling; null on other models.
