# Content Imprinting: What Zero-Retention KV-Cache Priming Actually Banks, and When It Helps

## Abstract

Retrieval-augmented systems precompute document KV caches offline and reuse them across
queries. A tempting "free lunch" is *cache priming*: prepend a short context during offline
encoding, let it shape the document's representations through self-attention, then discard it
before storage — reshaping the cache at zero inference cost. Whether this captures useful
signal has been unclear, and our own earlier work overstated it. We give a controlled,
end-to-end account.

First, a measurement correction: absolute negative log-likelihood (NLL) is entropy-confounded
for evaluating priming. Priming lowers output entropy, which lowers NLL without improving the
model's ability to *discriminate* the correct answer. Re-evaluating with a contrastive margin
(invariant to additive NLL shifts; paired with a lockstep-sharpening test and rank/top-1 metrics)
dissolves the headline that document-derived keyword prefixes beat instructions (margin effect
d≈0.00), and a battery of controls (neighbor-leakage, position matching, a machinery-neutral
prime, matched footing) overturns several further "clean" claims — including our own.

Second, the real phenomenon. We show that zero-retention priming *does* bank context into a
document's stored KV — substantially — and its **magnitude** is governed by a single measurable
model trait we call **imprintability**: it recovers up to 35% of a semantic context's value from a
stripped cache (−3.8 nats on Gemma 27B), scales monotonically across the Gemma family, is present
in Mistral and weak in instruct-tuned Qwen 2.5, with imprintability predicting banking magnitude at
r=0.94 across eight models. But **what** is banked is *not* a clean "semantic vs. surface" mode, as
we and others might assume. A word-order **shuffle control** (prime with the same tokens, ordered
vs. scrambled) shows the high-imprintability **Gemma family banks token presence** — its banking is
order-invariant (at 27B the *shuffled* prime banks *more*), so its "semantic banking" is lexical,
not relational meaning — while only **Mistral banks genuine structure** (shuffling destroys its
banking, −1.55 nats) and **Qwen banks little**. Imprintability thus measures the strength of a
content-*token* imprint that, at the extreme, trades off against literal structure. The semantic
imprint is distributed and read out in late layers, and is set by **instruction-tuning, not
architecture** (five architectural accounts fail): every pretrained *base* model imprints content,
and Qwen 2.5's alignment uniquely *suppresses* it while strengthening surface/code imprinting
(sem −0.72→+0.14, code +0.17→−0.37) — a controlled demonstration that the banked content type is a
*trainable* property.

Third, downstream value and an actionable, **task-aware** construction rule. The mode–task "match"
of earlier drafts is weaker than claimed: the QA effect of priming is **token presence** (shuffling
the question barely changes it) with a sign that does not track family — priming helps Qwen-7B
(−0.80 nats) but *hurts* the larger Qwen-14B (+0.44) and the Gemmas. What *is* robust and useful is
that the best build operation when the task is known is **mode-dependent**: against a SnapKV-style
task-aware **selection** baseline, conditioning (our discarded prime) *hurts* the token-presence
imprinters (Gemma, +0.5 to +1.0 nats) where selection is the right move, but *helps* Qwen-7B
(−0.6) where selection instead *hurts* by a full nat. A one-size selection method would therefore
damage Qwen-7B; the prescription is a decision rule indexed by imprinting character. We close with
the bounds — most context value (~65%+) is structurally un-bankable — and argue the durable
contributions are the content-imprint characterization (and its token-presence/structure
correction), the evaluation methodology that exposes it, and a mode-indexed rule for task-aware
cache construction.

---

## 1. Introduction

KV-cache reuse is a standard RAG optimization: systems such as TurboRAG [@turborag], CacheBlend
[@cacheblend], and SGLang [@sglang] encode document chunks offline and reuse their key–value
states, cutting time-to-first-token by up to an order of magnitude. The literature optimizes what
happens *after* construction — which entries to keep [@h2o; @snapkv], how to compress
[@gisttokens; @icae], how to schedule. We ask whether *construction* itself carries
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
2. **A content-imprint axis, and what it banks (§6).** Priming banks context with a *magnitude*
   governed by a single trait (**imprintability**, r=0.94) that scales with model size. A word-order
   shuffle control (§6.5) corrects the "semantic" framing: the high-imprintability Gemma family banks
   **token presence** (order-invariant), only Mistral banks genuine **structure**, and Qwen banks
   little — there is no clean two-way semantic/surface *mode*. The banked content type is *set by
   instruction-tuning, not architecture* (§6.4), with a controlled base→instruct flip in Qwen.
3. **Task-aware construction (§7).** The mode–task "match" is weaker than earlier claimed (the QA
   effect is token presence, and its sign is model/scale-specific, not a family law). The robust,
   actionable result: when the task is known, the best build operation is **mode-dependent** —
   against a SnapKV-style **selection** baseline, conditioning hurts the token-presence imprinters
   (select instead) but helps Qwen-7B (where selection itself hurts). The prescription is a
   decision rule indexed by imprinting character, not a universal "prime the cache."
4. **The ceiling and the negatives (§5, §8).** Context value is large (−2.8 nats when retained)
   but mostly un-bankable; we report every controlled claim that failed (contrastive priming is
   inert, a representation-level "coherence" mechanism is a positional artifact, and five
   architectural accounts of imprintability fail — which is itself the clue that pointed to
   training) as carefully as the ones that held.

The throughline is methodological honesty: for a "free-lunch" technique, the controls *are* the
result. Most clean stories here were dismantled by the next control; we report the survivors and
the casualties together.

---

## 2. Related Work

Our work sits at the intersection of six threads. For each we note what we build on and where
we differ; the recurring distinction is that prior work asks *how much* context can be stored or
*how to recover* what precomputation loses, whereas we ask *what kind* of context survives
zero-retention construction, *why it varies across models*, and *when it helps*.

**KV-cache reuse for RAG.** Precompute-and-reuse systems encode document chunks offline and
concatenate their caches at inference [@turborag; @sglang], cutting time-to-first-token by up to
an order of magnitude. Because precomputation omits cross-chunk attention and duplicates attention
sinks, it degrades quality; the dominant response is to *recover* the lost signal — CacheBlend
selectively recomputes a token subset [@cacheblend] and CacheClip uses an auxiliary model to pick
the tokens worth recomputing [@cacheclip]. We adopt the same construction primitive as TurboRAG
(precompute plus RoPE position-id repositioning) but pose the inverse question — can construction-
time *conditioning add* usable signal? Our bankability ceiling (§5) and imprinting-mode result
(§6) explain *why* precomputation loses quality (most cross-chunk semantic content is not bankable
into a stripped cache, and how much depends on a model-specific trait), making our analysis
complementary to these recovery methods.

**Task-aware and trained caches.** Closest to our motivating question — *can knowing the task at
build time improve the cache?* — are three recent lines. (i) *Task-aware compression*: Beyond RAG
[@beyondrag] precomputes a single compressed cache tuned to a task description rather than a query
("condense study material for an open-book exam"), beating query-agnostic compression and
approaching query-aware compression. (ii) *Query-aware vs. query-agnostic selection*: SnapKV
[@snapkv] and Ada-KV [@adakv] keep the tokens an observed query attends to, while KVzip [@kvzip]
targets the query-agnostic regime. (iii) *Trained caches*: Cartridges [@cartridges] distills a
corpus into a small trainable cache via self-study, matching in-context learning at a fraction of
the memory. All three change *which tokens (or trained slots) are retained*. We differ on the
mechanism: we *retain nothing* and instead ask whether construction-time *conditioning* — a
discarded natural-language prefix that reshapes the kept document keys/values — adds usable signal,
and we make the selection methods a first-class baseline (does conditioning beat task-aware
*selection* of the same budget?). Our contribution to this thread is the *mode* account: which
conditioning content a given model can bank is governed by a measurable trait (§6.2), so the right
task-aware prime is *model-specific* — a prescription these retention-based methods do not provide.

**KV-cache compression and prompt/context compression.** A large literature compresses the cache
post-hoc by evicting or summarizing tokens [@h2o; @snapkv], or compresses context into a *few
retained* learned slots — gist tokens [@gisttokens], in-context autoencoders [@icae], and
distillation into cache vectors [@kvdistill]. A complementary strand probes the *limits* of such
compression [@gistsilverbullet; @cramming]. Our zero-retention priming is the degenerate extreme
(zero retained tokens), and our bankability ceiling is a *training-free* measurement of that limit
(§5). Critically, this literature measures *how much* compresses; we introduce a *what-kind* axis —
a content-type dissociation (meaning vs. surface form) that is model-specific and, we show,
*trainable* (§6.4). This predicts what a compression method can preserve on a given model and warns
that methods validated on one model family may not transfer.

**Continuous-prompt and steering methods.** Prefix- and prompt-tuning learn continuous prefixes
that *persist* at inference [@prefixtuning; @prompttuning], and activation steering adds a fixed
direction to the residual stream [@caa]. Cache priming differs on both counts: the prefix is
*discrete natural language* and is *discarded* before storage (zero inference cost). We borrow the
steering-vector methodology to test whether priming's effect reduces to a single fixed direction —
it does not; the imprint is content-routed (§8).

**Long-context behavior and the evaluation confound.** Work on how models use long contexts shows
that placement and surface statistics strongly shape behavior and that perplexity need not track
downstream quality [@lostinmiddle]. We make a specific instance of this concrete and actionable:
absolute NLL conflates entropy with discrimination for cache-construction interventions, so we
evaluate with a contrastive margin (invariant to additive NLL offsets) backed by a battery of
controls — a lockstep sharpening test, rank/top-1 metrics, and per-model prior-shift splits
(§3.2, §4) — a correction that overturned several clean-looking claims, including our own.

**Mechanistic interpretability of context use.** We use activation patching from the causal-tracing
lineage [@rome] and contrastive steering vectors [@caa] as tools to localize and test the imprint
(§6.3, §8). Our finding that the semantic imprint is distributed and read out in *late* layers
connects to evidence that in-context and task processing concentrate in middle–late layers
[@wheredoesicl; @layerbylayer]. Most directly, the relationship between in-context learning and
instruction tuning — that ICL reshapes hidden states as implicit instruction tuning [@iclimplicitit]
and that instruction tuning reshapes middle-layer representations [@layerbylayer] — frames our most
novel result (§6.4): instruction tuning *sets a model's context-imprinting mode*, and can preserve,
amplify, or *flip* it. To our knowledge no prior work shows instruction tuning flipping a model from
semantic to surface-form context encoding.

---

## 3. Method

### 3.1 Two-phase pipeline
**Phase A:** encode `[BOS, context, \n, document]`; select BOS + document entries; reposition
document keys to positions `1..D` (float32 RoPE delta); per-tensor normalize. **Phase B:** append
`[\n, query, (\n, answer)]` at positions `D+1+`, reusing the cache; never pass explicit
`cache_position` (it reintroduces a one-token look-ahead).

### 3.2 Metrics and controls (the part that matters)
- **Contrastive margin** `= mean_k NLL(distractor_k) − NLL(correct)`. This is invariant to a
  *uniform additive* NLL shift but **not** to multiplicative logit sharpening (temperature `T`
  scales the margin by `1/T`), so it is not a complete entropy control on its own. We therefore
  pair it with a **lockstep test** — does priming move the correct answer's NLL while leaving
  distractors flat (genuine) or in lockstep with them (sharpening)? — and **rank / top-1**
  (which are invariant to any monotone per-example transform). Absolute NLL alone is unsafe.
- **Gold-class prior-shift control** — for binary/labelled tasks, split the *gold-aligned* margin
  change by gold class. A *symmetric* split (one class up, the other down) is a label-prior shift;
  *both classes up* is genuine discrimination. We report this per model, not pooled.
- **Machinery-neutral prime** — a content-free, length-matched prime (newlines) isolates the
  reposition+normalize *construction* cost from the prime's *content* effect.
- **Matched footing** — hold the prime fixed and vary only the variable of interest (e.g., whether
  a fact is in the document) to isolate it.
- **Neighbor-leakage / position-matching** — checks that "query-agnostic" constructions are truly
  query-agnostic and that representation comparisons are not positional artifacts.

### 3.3 Models and data
Eight instruction-tuned models spanning imprintability: Qwen 2.5 (1.5/7/14B), Mistral 7B, Gemma 3
(1/4/12/27B); plus three pretrained **base** models (Gemma-4B-pt, Qwen-7B, Mistral-7B-v0.3) for
the training analysis (§6.4). Datasets: SQuAD, HotpotQA, GSM8K, DROP, MS MARCO (BM25 hard
negatives); plus controlled synthetic probes (a decisive fact in filler) for banking. Bootstrap
95% CIs; `*` excludes 0.

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
*Figure 1: The entropy confound. Every prefix lowers absolute NLL (gray), but on the contrastive
margin (blue) the pooled keyword effect collapses to ≈0 and only extract-style instructions move
the pooled margin.*

The pooled keyword margin (d≈0.00) is, however, a **sign-cancellation artifact**, not a per-sample
null: the per-model TF-IDF margin effect is large and bidirectional — it *helps* low-imprintability
models (Qwen-1.5B +0.243\*, Mistral-7B +0.104\*) and *hurts* high-imprintability Gemma (−0.240\*),
averaging to zero across models. So the honest statement is not "keyword priming does nothing" but
"keyword priming's discrimination effect is **model-specific and bidirectional**" — which in fact
foreshadows the imprinting-mode thesis (§6). The robust pooled positive is the generic
*extract*-style instruction (+0.270), whose lockstep signature is genuine (the correct answer's
NLL falls while distractors' rise). On BoolQ, our earlier "label-prior shift only" reading does
**not** survive the full data: extract priming produces a *real* discrimination gain on the larger
instruct models (gemma-12B gold=yes +0.916\* / gold=no +0.782\*, balanced accuracy 0.875→0.883;
qwen-7B +0.390\*/+0.489\*; ministral-8B +0.450\*/+0.001) and improves accuracy, ECE, and Brier;
only qwen-1.5B shows the pure prior shift (+0.40/−0.33). The lesson stands — perplexity/NLL gains
are presumptively inflated — but the corrected control is the per-model gold-aligned margin, and
priming does sharpen discrimination, not merely shift the prior, on capable models.

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

## 6. Imprinting: a Content-Imprint Axis, and What It Actually Banks (the central result)

> **Reading guide / correction.** An earlier framing of this section called the headline axis
> "semantic imprinting" and split models into "semantic vs. surface" *modes*. Word-order shuffle
> controls (§6.5) show that label is too strong: the dominant, scaling axis is a **content-imprint**
> axis whose *magnitude* a single trait predicts (r=0.94, §6.2), but **what** is imprinted is
> model-specific — for the Gemma family it is **token presence** (order-invariant), and only for
> Mistral is it genuine **structure/meaning** (order-dependent). We keep the magnitude results,
> which are robust, and reinterpret the mechanism in §6.5.

### 6.1 What is banked: a content-imprint axis (magnitude) with a model-specific character
We prime a filler document with a fact whose answer ranges from meaningless to meaningful, strip
it, and measure how much the answer is recovered (machinery-controlled; negative = banked). For
one representative pair:

| answer type | gemma3_12b | qwen25_7b |
|---|---|---|
| code (4 digits) | −0.07 (no) | **−0.33\*** |
| pseudoword (nonword) | −0.39\* (weak) | **−0.27\*** |
| rare word | **−3.67\*** | +0.18\* (worse) |
| common word | **−0.93\*** | +0.29 (worse) |
| phrase | **−2.02\*** | −0.10 (n.s.) |

For *this pair* the picture is a tidy double dissociation — Gemma imprints meaning, Qwen imprints
surface form. **But that tidiness does not survive the full model set, and we do not claim it.**
The *semantic* axis is robust and law-like (§6.2): it scales monotonically across the Gemma family,
appears in Mistral, and is weak/absent in instruct Qwen. The *surface/literal* axis is real but
**idiosyncratic** — across the eight instruct models the code-banking column is non-monotonic and
sign-unstable: Gemma-27B *does* bank a literal code (−0.33\*), Mistral banks it *more* than Qwen-7B
(−0.57\*), and Qwen-14B strongly *anti-banks* it (+0.78\*). So "Gemma cannot store literals" and
"Qwen imprints surface form" are properties of the cherry-picked 12B-vs-7B pair, not general laws.
The principled, controlled evidence for surface imprinting is instead the *base→instruct flip*
(§6.4), where Qwen's tuning demonstrably trades meaning for surface form. We therefore present the
semantic axis as the headline and the surface axis as a model-specific, training-dependent
phenomenon.

![Figure 2](figures/fig8_content_dissociation.png)
*Figure 2: Content-type banking for one representative pair (Gemma-12B vs Qwen-7B). Gemma banks
meaningful words/phrases; Qwen banks surface forms (codes/pseudowords). This tidy pattern is
pair-specific — the surface axis is idiosyncratic across the full model set (§6.1); the semantic
axis (Fig. 3–4) is the robust, scaling result. Banking = nats recovered from the stripped cache;
bars are bootstrap 95% CIs.*

### 6.2 One trait predicts the magnitude: imprintability (r=0.94)
Define **imprintability** as the mean |Δ query-NLL| a generic prefix induces (what we earlier
called "primability"). Across eight models it predicts the *magnitude* of content banking almost
perfectly (the *kind* of banking — token presence vs. structure — is resolved in §6.5):

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

It is the *trait*, not the brand: Mistral (non-Gemma, imprintability 0.55) banks content
(−1.36). Banking magnitude also **scales monotonically with Gemma size** — 6% → 23% → 36% →
35% of a semantic context's value (1B→4B→12B→27B). (§6.5 shows that for Gemma this scaling
magnitude is *token-presence* imprint; for Mistral it is genuine structure.)

![Figure 4](figures/fig9_semantic_scaling.png)
*Figure 4: Semantic imprinting scales with model size — fraction of a semantic context's value
recoverable from the stripped cache, Gemma 1B→27B.*

### 6.3 Where it lives
Layer-wise KV patching on semantic recovery: the Gemma imprint is **distributed and read out in
late layers** (peak ~L44/48; full-cache recovery +4.1 nats, single-layer patches sum to only
+0.9 → non-localized). Qwen shows no semantic recovery. The semantic imprint is a distributed,
late-stage property of the cached representation.

### 6.4 The cause is instruction-tuning, not architecture
Five architectural accounts of imprintability fail (QK-norm, attention sharpness, prefix-salience,
fixed-direction sufficiency, residual-norm control — the last killed by Mistral, which has the
most explosive residual stream yet high imprintability; §8). The cause is in the *training*.
Comparing pretrained **base** models to their instruction-tuned versions on the banking probe:

| | code bank | semantic bank |
|---|---|---|
| Gemma-4B base | +0.31 (no) | **−1.11\*** |
| Gemma-4B instruct | +0.15 (no) | **−2.46\*** |
| Qwen-7B base | +0.17 (no) | **−0.72\*** |
| Qwen-7B instruct | **−0.37\* (code!)** | +0.14 (none) |
| Mistral-7B base | −0.16 (code) | **−1.08\*** |
| Mistral-7B instruct | −0.56\* (code) | **−1.36\*** |

Two facts: (i) **every pretrained base model semantically imprints** (−0.7 to −1.1) — semantic
imprinting is a *universal* property of pretrained LMs, not a Gemma quirk; (ii) **instruction-
tuning modulates it, and Qwen 2.5's tuning uniquely *flips the mode*** — destroying semantic
imprinting (−0.72→+0.14) and creating surface/code imprinting (+0.17→−0.37), while Gemma's and
Mistral's tuning preserve and amplify the semantic mode. So imprinting mode is a **trainable**
property set in alignment training, not a fixed architectural fact — which is exactly why every
architectural ablation failed. A practical corollary: a model could be tuned toward either mode.

![Figure 5](figures/fig11_base_vs_instruct.png)
*Figure 5: Imprinting mode is set by instruction-tuning. Every pretrained base model (gray) banks
meaning; instruction-tuning amplifies it for Gemma and Mistral but, for Qwen 2.5, destroys
semantic imprinting (left) and creates surface/code imprinting (right) — a mode flip.*

### 6.5 What is *actually* banked: token presence, not (mostly) meaning — a shuffle control
The §6.1–6.2 probes use a *single* fact, so recalling the one primed answer requires only that the
answer **token** be imprinted somewhere in the stripped cache — they cannot distinguish banking
*meaning* from banking *token presence*. We add the missing control: prime with the **same tokens**
either in natural order or **shuffled** (a deterministic permutation that destroys word order and
relational structure but preserves the exact token multiset). `ORDER ≡ banking(ordered) −
banking(shuffled)`; ORDER≈0 means the banking is token presence, ORDER<0 means order/meaning is
load-bearing. We run two independent versions: the single-fact measure of §6.1 (recall one fact),
and a harder **two-fact binding** measure (prime two `city→topic` facts, ask about one — recalling
the right topic *requires* the city→topic binding, not just token presence).

| model | single-fact ORDER (exp33) | two-fact binding ORDER (exp32) | what is banked |
|---|---|---|---|
| Gemma-4B  | −0.20 (n.s.) | −0.51\* | token presence (small structure at 4B) |
| Gemma-12B | −0.28 (n.s.) | **+1.43\*** (shuffle *better*) | **token presence** |
| Gemma-27B | +0.47\* (shuffle better) | +0.23 (n.s.) | **token presence** |
| Mistral-7B | **−1.55\*** (shuffle kills it) | **−1.07\*** | **genuine structure/meaning** |
| Qwen-7B   | −0.16 (n.s.) | +0.12 (n.s.) | banks little of either |

Two independent shuffle probes **agree**, and they overturn the binary "semantic (Gemma, Mistral)
vs. surface (Qwen)" mode of earlier drafts:

- **The Gemma family is a token-presence imprinter.** Its large, scaling "semantic banking" (§6.2)
  is **order-invariant** — shuffling the fact's tokens barely changes it, and at 27B the *shuffled*
  prime banks *more*. So Gemma's headline banking is **lexical/token-type** (a meaningful word
  imprints more than a digit code because it is a more imprintable *token*), not relational meaning.
- **Mistral is a genuine structure imprinter.** Its banking is strongly order-dependent — shuffling
  destroys it (single-fact −1.55\*; it even *anti-banks* shuffled facts), and the same holds for
  codes (CODE ORDER −0.74\*). Only here does zero-retention priming bank *meaning*.
- **Qwen-7B banks little** of either, single- or two-fact: consistent with a weak imprinter.

There is, in short, **no clean two-way semantic/surface mode**: the families differ in *what* they
bank, not on a single axis. A coherent reading is that **imprintability measures the strength of a
model's content-token imprint (≈ how aggressively it abstracts a prefix into the kept
representation), and this token-level imprint trades off against literal structure** — which is why
the most imprintable model (Gemma-12B) banks a *shuffled* two-fact prime *better* than an ordered
one (the ordered version is abstracted away from the literal answer token). The robust, defensible
claims are therefore: (i) a content-imprint axis whose **magnitude** scales with imprintability
(r=0.94); (ii) the imprint is predominantly **token presence** for the high-imprintability Gemma
family and genuine **structure** only for Mistral; (iii) the base→instruct flip (§6.4) is real at
the level of *what content type* is banked. We retract the stronger "Gemma banks meaning" reading.

---

## 7. Mode–Task Match: When Imprinting Helps

Imprinting mode is not uniformly good — its value depends on the task.

**Relevance (reranking).** Priming each MS MARCO passage with its own keywords significantly
improves query-likelihood reranking over generic priming on Gemma, and beats *no* priming on the
larger, higher-imprintability models: **+0.036 MRR on Gemma 12B and 27B** (CIs exclude 0), null
on Gemma 4B and on Qwen/Mistral. Semantic imprinting re-weights the passage's own semantic
salience, which is exactly what relevance scoring rewards.

**Extraction (QA).** Priming a passage with the *question*, then stripping it, and answering
(machinery-controlled content effect, pos = hurts; we also shuffle the question's tokens to test
the mechanism):

| | Gemma-12B | Gemma-4B | Qwen-7B | Qwen-14B |
|---|---|---|---|---|
| ordered-question prime | **+0.36\*** | +0.12 (n.s.) | **−0.80\*** | **+0.44\*** |
| shuffled-question prime | **+0.32\*** | −0.04 (n.s.) | **−0.73\*** | **+0.28\*** |
| order contribution | +0.04 (n.s.) | +0.16 (n.s.) | −0.07 (n.s.) | +0.16\* |

The mechanism is **not** semantic blurring. Shuffling the question's tokens barely changes the
effect (order contributes ≤0.16 nats, mostly n.s.): what moves the answer is the **token presence**
of the question in the cached representation, with a **sign that is model-and-scale-specific**.
Crucially the sign does *not* track family — Qwen-7B is *helped* (−0.80) but the larger Qwen-14B is
*hurt* (+0.44), like the Gemmas. So the earlier clean story ("surface imprinters help extraction,
semantic imprinters hurt it") is **not a law**: it is a property of the specific Qwen-7B vs.
Gemma-12B pair. We therefore demote the mode–task "2×2" below to an *illustrative* worked pair,
and replace the prescription with the directly actionable, mode-indexed result of §7.1 (which
construction operation to use when you know the task).

**The 2×2** (one worked pair — Gemma-12B and Qwen-7B — *not* a family law; see the per-model
caveats above and the shuffle controls in §6.5):

| | Gemma-12B | Qwen-7B |
|---|---|---|
| reranking (relevance) | **helps** (+0.036\* MRR) | no |
| QA (precise extraction) | **hurts** (+0.36\* NLL) | **helps** (−0.80\* NLL) |

![Figure 6](figures/fig10_mode_task.png)
*Figure 6: Mode–task match for one worked pair (Gemma-12B, Qwen-7B). Left: keyword priming helps
Gemma's relevance reranking. Right: question priming helps Qwen-7B's QA extraction but hurts
Gemma-12B's. This is illustrative, not a family law — Qwen-14B reverses Qwen-7B's QA sign (above),
and the QA effect is token-presence, not meaning (§6.5).*

### 7.1 Task-aware construction: *select* vs. *condition* is mode-dependent
The motivating practical question is: if we know the task at cache-build time, how should we build
the cache? Two families of operation are available — **selection** (keep the task-relevant tokens,
drop the rest: the SnapKV/Beyond-RAG move) and **conditioning** (keep the document but reshape its
kept keys/values with a discarded task prime: our move). We compare them iso-budget on extraction:
per passage we pick the top-`k`=32 document tokens by question→document attention (a SnapKV-style
probe), and score the answer under (a) the selected tokens alone, (b) the selected tokens
*conditioned* by the discarded question, and (c) the full document conditioned by the question.
`selVal` = selection − normalized full doc; `COND|sel` = conditioning the selected set − selection
alone; both machinery-matched (pos = hurts).

| model | selVal (selection) | COND\|sel (conditioning, given selection) | full conditioning |
|---|---|---|---|
| Gemma-12B | −0.03 (n.s.) | +0.16 (n.s.) | **+0.52\* (hurts)** |
| Gemma-4B  | −0.12 (n.s.) | **+1.03\* (hurts)** | **+1.05\* (hurts)** |
| Qwen-7B   | **+1.00\* (hurts)** | **−0.51\* (helps)** | **−0.63\* (helps)** |

The best build operation is **mode-dependent, and the two modes want opposite things**:

- On the **token-presence imprinters (Gemma)**, conditioning *hurts* extraction (full-doc priming
  +0.5 to +1.0 nats) and selection is neutral → the right move is to **select / not prime**.
- On **Qwen-7B**, *selection hurts* by a full nat (it needs the whole document) while *conditioning
  helps* (−0.5 to −0.6) → the right move is to **prime, not prune**.

So a one-size task-aware method built only on **selection** (Beyond RAG, SnapKV) would *hurt*
Qwen-7B by ≈1 nat, where our discarded-prefix conditioning recovers ≈0.6; conversely conditioning
is the wrong tool on Gemma. The deployable corollary is a **decision rule indexed by imprinting
mode**: *measure imprintability/banking once per model; for extraction, prime surface/weak
imprinters and select (or leave) high-imprintability ones.* (We demonstrate the rule on the
Qwen-7B vs. Gemma pair; §10 notes that the surface side rests on Qwen-7B and the larger Qwen-14B
behaves like the token-presence imprinters, so the rule is mode-indexed, not family-indexed.)

Value comes from **matching the construction operation to the model's imprinting character**, not
from priming per se.

---

## 8. What Did Not Pan Out (controlled negatives)

- **"Contrastive" priming is inert.** Neighbor-subtracted keywords ≈ plain passage keywords (n.s.,
  4 models). The active ingredient is keyword content, not contrast.
- **A representation-level "content-coherence" mechanism is a positional artifact.** Position-matched
  re-measurement shrank the Gemma effect ~70%, and Mistral (more "coherent" by that metric) shows
  no behavioral effect — falsifying it.
- **No *architectural* account of imprintability survives** — and that turned out to be the clue.
  QK-norm (ablation *raises* imprintability), attention sharpness (families respond oppositely to a
  temperature knob), prefix attention-salience (Gemma attends to the prefix *less*), a fixed-direction
  steering vector (reproduces <25%), and residual-norm control (killed by Mistral) were each
  falsified. All five failed because the cause is not architectural but in *training* (§6.4): the
  imprinting mode is set by instruction-tuning.
- **No universal win, and no precise-fact injection.** Most context value is un-bankable; priming a
  document with an arbitrary external fact recovers ~0 of it on the semantic imprinter.

---

## 9. Practical Guidance

1. **Evaluate with the contrastive margin and a machinery-neutral control; never absolute NLL.**
   Report the prior-shift control. NLL gains for priming are presumptively entropy artifacts.
2. **Measure imprintability first** (mean |Δ query-NLL| from a generic prefix). It predicts how
   *much* a model banks (r=0.94); a quick shuffle probe (ordered vs. scrambled prime) tells you
   *what* it banks — token presence (Gemma) vs. structure (Mistral) vs. little (Qwen).
3. **Choose the construction operation by imprinting character, not by default-priming.** For
   extraction with a known task: on a high-imprintability token-presence model (Gemma), conditioning
   *hurts* — prefer task-aware **selection** (keep relevant tokens) or no priming; on a weak/surface
   imprinter where selection *hurts* (Qwen-7B), **condition** instead. Do not deploy a single
   operation blind — a selection-only method damages Qwen-7B, and conditioning damages Gemma.
4. **Do not expect a free lunch.** ~65%+ of context value is un-bankable; the construction step is
   mildly lossy. The gains are real but bounded, model-specific, and task-specific.

---

## 10. Limitations

- We localize the cause of the banked content type to instruction-tuning (§6.4) but do not identify
  *which* alignment objective changes it; a causal training study (controlled fine-tunes) is future
  work.
- The token-presence-vs-structure split (§6.5) is shown with two shuffle probes on five models; the
  "Gemma = token presence, Mistral = structure" reading rests on one non-Gemma structure imprinter
  (Mistral) and would be strengthened by more model families.
- The task-aware select-vs-condition rule (§7.1) is demonstrated on three models; its surface side
  rests on **Qwen-7B specifically** — the larger Qwen-14B behaves like the token-presence imprinters
  in QA (§7), so the rule is **mode-indexed, not family-indexed**, and the surface→condition cell
  needs more weak/surface imprinters to generalize. The selection baseline is a single SnapKV-style
  attention probe at k=32; other budgets/selectors may shift the crossover.
- Downstream value is shown on reranking and extractive QA; broader task coverage is future work.
- Banking probes use controlled synthetic facts (decisive content in filler) at N=150; behavioral
  results use N=300–900. The synthetic design maximizes cleanliness at some cost to ecological
  validity.
- Reranking uses query-likelihood scoring, well below purpose-built rerankers in absolute MRR.

---

## 11. Conclusion

We set out to optimize KV-cache construction and learned, first, that the optimization we thought
we had was an artifact of how we measured it. Measured correctly, zero-retention priming *does*
bank context — with a **magnitude** governed by a single trait (imprintability, r=0.94) that scales
with model size. But a word-order shuffle control corrected our own next mistake: the banking is not
a clean "semantic vs. surface" mode. The high-imprintability Gemma family banks **token presence**
(its "semantic" banking is order-invariant), only Mistral banks genuine **structure**, and Qwen
banks little — so imprintability measures the strength of a content-*token* imprint that, at the
extreme, even competes with literal structure. The honest verdict is that directed cache
construction is not a free-lunch accelerator but a *typed*, bounded mechanism whose character is
model-specific, **trainable** (five architectural accounts fail; a base-vs-instruct comparison
localizes the content type to instruction-tuning, which for Qwen 2.5 flips it), and whose
*deployment* must be matched to that character: when the task is known, the better build operation —
keep-the-relevant-tokens **selection** vs. discarded-prime **conditioning** — depends on the model's
imprinting character, and using the wrong one *hurts*. Its durable contributions are the
content-imprint characterization (and the token-presence/structure correction), the evaluation
methodology that exposes the entropy and machinery confounds, and a mode-indexed rule — survivors
and casualties alike — for what zero-retention cache construction can and cannot do.

---

## Appendix

**A. Reproducibility.** Pipeline in `directed_kvcache_v4/lib` (RoPE float32 reposition, sliding-
window handling, no Phase-B `cache_position`). Experiments under `experiments/13_contrastive/`
(keyword/contrastive reranking, ablations, machinery/coherence/circuit/steering controls),
`experiments/15_bankability/` (bankability, reweight-vs-inject, content-type spectrum, circuit
localization, downstream QA, base-vs-instruct), `experiments/16_architecture/` (intrinsic-metric
probe), and `experiments/17_taskaware/` (shuffle controls and task-aware select-vs-condition).
Running log: `experiments/09_boolq_classification/SHARPENING_FINDINGS.md`; task-aware findings:
`experiments/17_taskaware/FINDINGS.md`. Contested headline numbers are regenerated deterministically
from `results/*.json` by `make_numbers.py` and `make_taskaware_numbers.py`. Bootstrap CIs (4000
resamples); fixed seeds; 20-sample checkpoints.

**B. Key statistics.** Entropy confound (§4) exp05; keyword-vs-bare reranking (§7) exp14b/exp14c
(N=900): gemma12b/27b +0.036\*, others n.s.; imprintability×banking-magnitude r=0.94 (exp26);
content-type dissociation (exp27, N=150); localization (exp28); base→instruct content-type flip
(§6.4) exp26: Qwen semantic −0.72→+0.14, code +0.17→−0.37; architecture probe (5th falsification)
exp30; bankability ceiling (exp24/25): retained −2.8 nats, content-bankable ≈0, machinery ~0.3–0.6.
*Shuffle controls (§6.5), N=150:* single-fact SEM order-effect (exp33) Gemma-4B/12B n.s.
(−0.20/−0.28), Gemma-27B +0.47\*, Mistral −1.55\*, Qwen-7B n.s.; two-fact binding order-effect
(exp32) Gemma-12B +1.43\*, Mistral −1.07\*, Qwen-7B n.s. *QA prime (§7), N=300, machinery-controlled
(pos=hurts):* ordered/shuffled question Gemma-12B +0.36\*/+0.32\*, Qwen-7B −0.80\*/−0.73\*, Qwen-14B
+0.44\*/+0.28\*. *Select-vs-condition (§7.1), N=300, k=32:* selVal Qwen-7B +1.00\* (else n.s.);
conditioning-given-selection Qwen-7B −0.51\*, Gemma-4B +1.03\*, Gemma-12B +0.16 n.s.; full
conditioning Gemma-12B/4B +0.52\*/+1.05\*, Qwen-7B −0.63\*.
