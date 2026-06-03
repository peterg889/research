# Instruction-Form Confirmatory Sweep — Findings

Tests whether the contrastive-discrimination gain is the instruction FORM (robust)
or specific to one sentence. 8 phrasings + 2 anchors, contrastive-margin metric,
5 models × 4 datasets × 300 samples. Identical seeds/distractors to exp05.

## Verdict: neither "one sentence" nor "any instruction" — it is a narrow semantic class

Two of the five models (Qwen 1.5B, Mistral 7B) have positive margins for *everything
including the anchors* — they cannot separate signal from noise. The decisive
evidence is the three DIAGNOSTIC models where the anchors are ≤0
(Qwen 7B, Gemma 12B, Ministral 8B). Pooled d(margin) on those three:

| group | d(margin) |
|---|---|
| **extract** ("Extract the key facts from this text") | **+0.180** |
| attend ("Pay close attention to the details…") | +0.175 |
| summarize ("Summarize the following text") | +0.095 |
| other imperatives (comprehend/index/identify) | −0.01 to −0.03 |
| interrogative ("What are the key facts…?") | −0.146 |
| declarative ("This text contains important information") | −0.078 |
| anchor: random_docwords | −0.149 |
| anchor: random_vocab | −0.244 |

Per-instruction ranking on diagnostic models:
`extract +0.18, attend +0.18, summarize +0.10` > `comprehend −0.01, index −0.02,
identify −0.03` > `declarative −0.08, question −0.15`.

`extract` beats the mean of the other five imperatives by **d=+0.265, CI [+0.23,+0.30]**
(paired) — so it is genuinely special, not merely "an imperative."

## Three conclusions

1. **Not a one-sentence artifact.** `extract` replicates as positive and
   CI-significant on all 5 models, including all 3 diagnostic ones (q7b +0.18,
   gemma12b +0.23, ministral8b +0.19). `attend` and `summarize` are also positive.
   The finding is real and reproducible.

2. **Not a generic instruction-form effect either.** Most instructions
   (comprehend, index, identify) sit at ≈0 on the diagnostic models — no better
   than the harmful anchors. Interrogative and declarative phrasings are
   *negative*. "Any instruction helps" is false.

3. **The active ingredient is a narrow semantic class:** imperative directives to
   *extract / attend to / condense salient content*. Mood matters (imperative >
   interrogative ≈ declarative), but mood alone is insufficient — the instruction
   must direct salient-content extraction. This is a SEMANTIC effect (what the
   instruction asks for), not a structural one (presence of an instruction).

The anchors reconfirm exp05 at pooled diagnostic level: random_docwords ≈ harmful
(repetition gives no discrimination), random_vocab strongly harmful.

## Refined paper claim

Replace "a generic task instruction improves discrimination" with: **priming the
cache with an imperative "extract the key facts"-style directive robustly improves
answer discrimination across model sizes and architectures — including models where
absolute NLL degrades — whereas document-derived prefixes (keywords, repetition),
the oracle query, and non-extraction instruction phrasings (questions, statements,
"read carefully") do not.** The effect is specific to extract/attend/summarize
semantics; `extract` is the single most reliable choice.
