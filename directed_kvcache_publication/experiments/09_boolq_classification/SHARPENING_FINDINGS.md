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
