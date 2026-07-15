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
model trait we call **imprintability**: it recovers a large share of a semantic context's value from
a stripped cache (up to −3.8 nats on Gemma-27B; ~36% of context value at Gemma-12B), rising
monotonically in nats across the Gemma family, present
in Mistral and weak in instruct-tuned Qwen 2.5, with imprintability predicting banking magnitude at
r=0.94 across eight models. But **what** is banked is *not* a clean "semantic vs. surface" mode, as
we and others might assume. A word-order **shuffle control** (prime with the same tokens, ordered
vs. scrambled), run across **eleven models in seven families**, reveals a **three-way** split, with
multiple independent families on each side: **token-presence imprinters** (Gemma, Falcon-3, Yi-1.5)
bank with magnitude but order-invariantly — shuffling leaves the banking intact or *increases* it,
so their "semantic banking" is lexical, not relational meaning; **structure imprinters** (Mistral,
Ministral, Llama-3, OLMo-2) bank order-dependently, so shuffling destroys it (up to −1.55 nats); and
**the Qwen family banks little** (including a DeepSeek reasoning-distilled Qwen, showing the backbone
bounds imprinting). Magnitude and *kind* are separable — OLMo-2 banks weakly but structurally,
Falcon-3 strongly but lexically — so imprintability measures the strength of a content-*token*
imprint that can trade off against literal structure. This content
imprint (token presence for Gemma) is distributed and read out in late layers, and is set by
**instruction-tuning, not architecture** (five architectural accounts fail): every pretrained
*base* model imprints content,
and Qwen 2.5's alignment uniquely *suppresses* it while strengthening surface/code imprinting
(sem −0.72→+0.14, code +0.17→−0.37) — a controlled demonstration that the banked content type is a
*trainable* property.

Third, downstream value — and the limits of any tidy **task-aware** rule. Task-dependence is real
but not a law: the QA effect of priming is **token presence** (shuffling
the question barely changes it) with a sign that does not track family — priming helps Qwen-7B
(−0.80 nats) but *hurts* the larger Qwen-14B (+0.44) and Gemma-12B (+0.36). Comparing our discarded-prefix
**conditioning** to a SnapKV-style task-aware **selection** baseline across eight models, neither
operation dominates: conditioning *helps* four (Qwen-1.5B/3B/7B, Gemma-1B; up to −1.9 nats) and
*hurts* four, and which wins does **not** reduce to imprintability (r=0.29) or size — there is no
trait-indexed rule, and you must *probe both per model*. The one systematic, confound-controlled effect is that aggressive query-aware selection
*hurts the whole Qwen family and Mistral* (but never Gemma) even at matched answer-span survival —
a real risk of SnapKV-style pruning that conditioning sidesteps. We close with the bounds — most context value
(~65%+) is structurally un-bankable — and argue the durable contributions are the content-imprint
characterization (and its token-presence/structure correction), the evaluation methodology that
exposes it, and an honest account of when discarded-prefix conditioning beats task-aware selection.

---
