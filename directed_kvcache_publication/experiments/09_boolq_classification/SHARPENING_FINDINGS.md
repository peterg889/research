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
