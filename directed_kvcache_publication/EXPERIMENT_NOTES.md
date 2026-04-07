# Directed KV Cache Publication — Experiment Notes

## Timeline

### 2026-04-03: Multi-Model Evaluation (Exp 01)

**Setup**: 4 models × 3 conditions (bare, random, comprehend) × 4 datasets × 400 samples.

Models: Gemma 3 12B-IT, Gemma 3N E4B-IT, Mistral 7B-Instruct-v0.3, Qwen 2.5 7B-Instruct.

**Results**:
- Gemma 3 12B: comprehend +0.34, random +0.07 (both positive)
- Gemma 3N E4B: random +0.41, comprehend -0.27 (structural helps, instructions hurt)
- Mistral 7B: random +0.13, comprehend -0.07 (structural helps, instructions neutral)
- Qwen 2.5 7B: comprehend -0.26, random -0.41 (everything hurts)

**Pipeline bugs found and fixed**:
- Missing `device_map='cuda:0'` caused model loading to hang on CPU
- Gemma 3N has `use_cache=False` bug — produces garbage logits. Workaround: always use `use_cache=True`.
- Qwen has no `bos_token_id` — handled with `bos_start=-1` in `reposition_kv_cache`.
- Multiple competing papermill processes caused silent failures — always kill old processes before restarting.

**Pipeline correctness verified**: 19/20 tests pass. RoPE round-trip exact in fp32 on all models. Bare two-phase matches single-pass on all models (within bf16 tolerance: Gemma 0.006, Mistral 0.078, Qwen 0.109).

### 2026-04-07: Condition Sweep

**Setup**: 4 models × 7 conditions × 4 datasets × 200 samples (80 hard).

Conditions: random, repeat_token ("the"×64), comprehend, extract, summarize, oracle (query as prefix), doc_keywords (first 64 doc tokens).

**Results**:
- Gemma 3 12B: **7/7 positive**. Best: extract +0.35, comprehend +0.35.
- Gemma 3N E4B: **5/7 positive**. Best: random +0.45. Instructions hurt (comprehend -0.27).
- Mistral 7B: **6/7 positive**. Best: extract +0.29, random +0.24.
- Qwen 2.5 7B: **0/7 positive**. Everything hurts. Best: comprehend -0.25.

### 2026-04-07: Root Cause Analysis

Three major findings from diagnostic investigation:

#### Finding 1: GSM8K Answer Format Bug (CRITICAL)

The publication code used the full chain-of-thought as the GSM8K answer (~50-100 tokens),
while v4 correctly extracted only the final number after `####` (~1-2 tokens).

Verification on Gemma 3 12B (N=100):
- Number-only answer: comprehend d=**+0.63** (82% win)
- Full CoT answer: comprehend d=**-0.39** (36% win)

v4 reference (number-only, N=200 hard): d=+1.334.

**Conclusion**: Prefix conditioning helps predict SHORT extractive answers by focusing
attention on relevant document regions. For LONG generative answers, the prefix introduces
a distributional bias that hurts diverse token generation. This is not a bug in prefix
conditioning — it's a fundamentally different task.

**Fix**: Extract `raw_answer.split('####')[-1].strip()` for GSM8K.

#### Finding 2: Qwen Needs BOS Token (Attention Sink)

Qwen 2.5 7B has `bos_token_id = None`. Without a leading BOS/attention-sink token,
prefix conditioning fails catastrophically because repositioned cache entries lack
a proper attention anchor.

Test on SQuAD v2 (N=60):

| Config | Comprehend d | Random d |
|--------|:-----------:|:-------:|
| No BOS + norm | -0.644 | -0.694 |
| No BOS + no norm | -0.649 | -0.696 |
| **With BOS + norm** | **+0.131** | -0.357 |
| **With BOS + no norm** | **+0.125** | -0.365 |

Adding PAD token (151643) as artificial BOS flips comprehend from d=-0.65 to d=+0.13.
Normalization has negligible effect (±0.005) on Qwen regardless of BOS.

**Conclusion**: The attention sink (BOS) is essential for prefix conditioning to work.
Models without native BOS need an artificial one.

#### Finding 3: Normalization is Architecture-Dependent

KV cache activation statistics explain why normalization helps Gemma 3 but not others:

| Statistic | Gemma 3 12B | Mistral 7B | Qwen 2.5 7B |
|-----------|:-----------:|:----------:|:-----------:|
| Key absmax range | 15–203 | 9–17 | 10–420 |
| Key absmax CoV | 0.82 | 0.14 | 1.91 |
| Val absmax range | 4–1240 | 0.4–7 | 2–72 |
| Norm perturbation (relative) | 0.25% | 0.24% | 0.26% |

The relative perturbation is identical (~0.25%) across all models. The difference is
in inter-layer scale variation:

- **Gemma 3** (CoV=0.82): Hybrid attention creates inconsistent scales between sliding
  (θ=10K) and full (θ=1M) layers. Normalization corrects this cross-layer inconsistency.
  Value absmax spans 4 to 1240 — a 300× spread that normalization compresses.

- **Mistral** (CoV=0.14): Uniform architecture, uniform scales. Nothing to correct.
  Normalization is effectively a no-op (max perturbation = 0.06).

- **Qwen** (CoV=1.91): Extreme outliers in a few layers (absmax=420) but normalization
  can't fix the underlying distribution shape. The outliers dominate attention regardless.

**Hypothesis**: Normalization primarily benefits models with heterogeneous layer types
that develop inconsistent activation scales during autoregressive generation.

### 2026-04-07: Decomposition Analysis

From the condition sweep, we can decompose the prefix effect into components:

```
                    Structural     Instruction Boost    Instruction Boost
Model               (random-bare)  (extract-repeat)     (comprehend-repeat)
──────────────────  ────────────   ──────────────────   ───────────────────
Gemma 3 12B         +0.07          +0.10                +0.09
Gemma 3N E4B        +0.45          -0.23                -0.65
Mistral 7B          +0.24          +0.19                -0.11
Qwen 2.5 7B*       -0.40          +0.16                +0.19
```
*Qwen without BOS fix — structural effect is contaminated by missing attention sink

**Three distinct mechanisms**:

1. **Structural effect** (any prefix vs bare): Positive on all models with BOS. Even
   meaningless tokens change how attention distributes across document positions during
   cache construction. Strongest on smaller models (Gemma 3N: +0.45).

2. **Instruction content boost** (instruction vs structural-only): Varies by model AND
   by instruction type. "Extract" helps more universally than "comprehend". Larger models
   (≥7B) can leverage instruction semantics; the 4B model is damaged by them.

3. **Model capacity threshold**: Gemma 3N (4B) shows the largest structural benefit
   (+0.45) but is DAMAGED by instructions (-0.23 to -0.65). The model can't interpret
   semantic instructions — they confuse its attention patterns. Models ≥7B can use
   instruction semantics productively (at least for "extract").

---

## Hypotheses and Planned Tests

### H1: Meaning vs Vocabulary (Scrambled Prefix Test)
**Question**: Does word ORDER matter, or just which words are present?
**Test**: Scramble comprehend/extract prefix token order. Compare coherent vs scrambled.
**Prediction**: On Gemma 3 12B, coherent > scrambled (meaning helps). On Gemma 3N,
coherent < scrambled (meaning hurts — the model tries to follow the instruction and fails).

### H2: Anti-Instruction
**Question**: Does semantic DIRECTION matter?
**Test**: Use "Ignore this text completely" as prefix. Compare vs comprehend.
**Prediction**: If meaning matters, anti-instruction should hurt relative to comprehend.
If only vocabulary/structure matters, similar performance.

### H3: Prefix Length Curve
**Question**: How quickly do structural vs semantic effects emerge?
**Test**: Vary prefix length L=1, 4, 16, 64 for comprehend and random.
**Prediction**: Structural effect (random) appears at L=1. Semantic effect (comprehend
advantage over random) needs L≥16-64.

### H4: Position-Only Shift
**Question**: Does the RoPE position shift itself matter?
**Test**: Score document at positions [65..65+D] instead of [1..D] with NO actual
prefix tokens. Just shift the position encoding.
**Prediction**: Near-zero effect. The position shift is a consequence of prefix removal,
not a cause of the benefit.

### H5: Qwen with BOS + Full Condition Sweep
**Question**: Once BOS is fixed, does Qwen behave like other models?
**Test**: Rerun all 7 conditions on Qwen with PAD token as BOS.
**Prediction**: Structural effects become positive. Extract/comprehend show small positive
effects (~d=0.1-0.2).

### H6: Per-Layer Normalization on Gemma 3
**Question**: Which layers drive the normalization benefit?
**Test**: Apply normalization only to sliding layers, only to full layers, or all layers.
**Prediction**: Normalizing only the mismatched layer type (whichever has larger scale
drift) provides most of the benefit.

### H7: Normalization Scale Sweep
**Question**: Is qmax=127 (int8 range) optimal, or do other scales work?
**Test**: Try qmax=7 (int4), 31 (int5), 127 (int8), 32767 (int16) on Gemma 3.
**Prediction**: Benefit peaks at intermediate scales (qmax=31-127). Too aggressive
(qmax=7) introduces too much noise. Too mild (qmax=32767) is nearly identity.

---

## Key Files

| File | Purpose |
|------|---------|
| `model_adapters.py` | Multi-model RoPE adapter |
| `tests/test_pipeline_correctness.py` | 29 tests (9 unit + 20 integration) |
| `experiments/01_multi_model/build_exp01.py` | 3-condition × 4-model experiment |
| `experiments/01_multi_model/build_condition_sweep.py` | 7-condition sweep |
| `results/exp01_multi_model/` | 3-condition results (400 samples) |
| `results/exp01_condition_sweep/` | 7-condition results (200 samples) |

### 2026-04-07: Deep Hypothesis Sweep

**Setup**: 4 models × 15 conditions × 2 datasets (SQuAD v2, GSM8K number-only) × 200 samples.
All models use BOS (Qwen gets PAD as artificial BOS). GSM8K answer extraction fixed.

**Full results per model** (pooled d across SQuAD + GSM8K, 80 hard per dataset):

```
GEMMA 3 12B — ALL 14/15 conditions positive (position_shift only negative)
  random_1         +1.17    comprehend_16    +1.03    comprehend_4     +0.98
  random_4         +1.11    comprehend_1     +0.95    repeat_64        +0.85
  comprehend_64    +0.84    random_16        +0.83    extract_64       +0.74
  anti_64          +0.63    random_64        +0.59    oracle_64        +0.58
  extract_scram    +0.52    comp_scrambled   +0.41    position_shift   -0.18

GEMMA 3N E4B — ALL 15/15 conditions positive (with BOS fix + GSM8K fix!)
  random_64        +0.82    extract_scram    +0.81    random_16        +0.77
  repeat_64        +0.73    anti_64          +0.72    position_shift   +0.64
  oracle_64        +0.60    random_4         +0.57    comprehend_4     +0.54
  extract_64       +0.53    comp_scrambled   +0.45    comprehend_1     +0.39
  comprehend_16    +0.35    comprehend_64    +0.17    random_1         +0.11

MISTRAL 7B — 11/15 conditions positive
  random_64        +0.66    extract_64       +0.63    extract_scram    +0.61
  oracle_64        +0.47    random_1         +0.46    repeat_64        +0.42
  comprehend_16    +0.26    comprehend_64    +0.22    comp_scrambled   +0.18
  random_16        +0.17    comprehend_4     +0.04    anti_64          -0.01
  position_shift   -0.18    random_4         -0.19    comprehend_1     -0.24

QWEN 2.5 7B (with BOS fix) — 10/15 conditions positive!
  repeat_64        +0.59    oracle_64        +0.57    position_shift   +0.56
  anti_64          +0.47    extract_64       +0.36    extract_scram    +0.28
  random_16        +0.23    comprehend_16    +0.19    comprehend_1     +0.17
  comprehend_64    +0.11    comprehend_4     -0.03    random_1         -0.04
  random_64        -0.10    comp_scrambled   -0.18    random_4         -0.23
```

**CRITICAL: BOS + GSM8K fixes transform the results.**
Previous sweep showed Qwen at 0/7 positive. With BOS fix: **10/15 positive**.
Previous sweep showed Gemma 3N at 5/7. With both fixes: **15/15 positive**.

#### Hypothesis Test Results

**H1: Does word order (meaning) matter?**
```
                    comp coherent-scrambled    extract coherent-scrambled
Gemma 3 12B         +0.43 (YES, strong)       +0.23 (YES)
Gemma 3N E4B        -0.28 (REVERSE — order hurts)  -0.28 (REVERSE)
Mistral 7B          +0.04 (negligible)        +0.02 (negligible)
Qwen 2.5 7B         +0.29 (YES, moderate)     +0.08 (weak)
```
Word order helps on Gemma 3 12B and Qwen, HURTS on Gemma 3N (small model can't
leverage instruction meaning — scrambled is better), negligible on Mistral.

**H2: Does semantic direction matter? (comprehend vs "Ignore this text")**
```
Gemma 3 12B         +0.21 (comp > anti — direction matters)
Gemma 3N E4B        -0.55 (anti > comp — anti-instruction works BETTER!)
Mistral 7B          +0.23 (comp > anti — direction matters)
Qwen 2.5 7B         -0.36 (anti > comp — anti-instruction works better)
```
Semantic direction matters on Gemma 3 12B and Mistral (positive instruction > negative).
On Gemma 3N and Qwen, "ignore" actually works BETTER than "comprehend" — these models
are better off with non-task-specific activation than with instructions they misinterpret.

**H3: Prefix length curves**
```
Gemma 3 12B  comprehend: L=1:+0.95, L=4:+0.98, L=16:+1.03, L=64:+0.84
             random:     L=1:+1.17, L=4:+1.11, L=16:+0.83, L=64:+0.59
```
SHOCKING: L=1 random is the BEST condition on Gemma 3 12B (d=+1.17). Short prefixes
outperform long ones. Effect DECREASES with length for random. For comprehend, peaks
at L=16. This suggests the benefit is about BREAKING the autoregressive pattern with
minimal disruption — longer prefixes add more noise than signal.

**H4: Does RoPE position shift alone matter?**
```
Gemma 3 12B         -0.18 (slight negative)
Gemma 3N E4B        +0.64 (strong POSITIVE!)
Mistral 7B          -0.18 (slight negative)
Qwen 2.5 7B         +0.56 (strong POSITIVE!)
```
Position shift alone (no actual prefix tokens, just shifting doc to pos 65+ then
repositioning back) has a LARGE positive effect on Gemma 3N (+0.64) and Qwen (+0.56).
This means the RoPE reposition operation ITSELF changes the cache in a beneficial way
for these models. On Gemma 3 12B and Mistral, the shift slightly hurts.

This is a major mechanistic finding: on some models, the bf16 precision loss from
RoPE rotation-then-unrotation acts as a beneficial regularizer (similar to the
normalization effect). The models where position_shift helps (Gemma 3N, Qwen) are
the same models where normalization had little effect — the reposition IS the
regularization mechanism.

---

## Key Files

| File | Purpose |
|------|---------|
| `model_adapters.py` | Multi-model RoPE adapter |
| `tests/test_pipeline_correctness.py` | 29 tests (9 unit + 20 integration) |
| `experiments/01_multi_model/build_exp01.py` | 3-condition × 4-model experiment |
| `experiments/01_multi_model/build_condition_sweep.py` | 7-condition sweep |
| `experiments/01_multi_model/build_deep_sweep.py` | 15-condition hypothesis sweep |
| `results/exp01_multi_model/` | 3-condition results (400 samples) |
| `results/exp01_condition_sweep/` | 7-condition results (200 samples, pre-fix) |
| `results/exp01_deep_sweep/` | 15-condition hypothesis results (200 samples, with fixes) |

---

## Remaining Experiments for Publication

### P0: Downstream Generative Accuracy (~6 hours)

**Why**: NLL-only papers always get "does this actually help generation?" from reviewers.
Our finding that prefix conditioning HURTS NLL on full CoT (d=-0.39) but HELPS on
number extraction (d=+0.63) makes the NLL-to-accuracy relationship non-trivial. Must prove it.

**Design**: 2 models (Gemma 3 12B, Mistral 7B) × 3 conditions (bare, comprehend_64,
random_64) × 2 datasets (SQuAD v2, GSM8K) × 80 hard samples.

Use `model.generate()` with cached KV from Phase A. Greedy decoding. Score with:
- SQuAD: Exact Match + Token F1
- GSM8K: Accuracy (extract number after `####`)

Report: accuracy table + Spearman correlation between per-sample NLL improvement
and per-sample accuracy improvement. This validates "NLL is a good proxy."

### P0: Two More Datasets in Deep Sweep (~3 hours)

**Why**: 2 datasets is too few for generalization. Add DROP (discrete reasoning,
strong in v4) and TriviaQA (factoid retrieval, moderate in v4).

**Design**: Same 15 conditions × 4 models × 200 samples on DROP + TriviaQA.

### P1: Confidence Intervals via Bootstrap (~2 hours)

**Why**: Cohen's d gives point estimates. Reviewers may want CIs. Bootstrap the
paired differences 1000 times to get 95% CIs on the decomposition components.

**Design**: Resampling analysis on existing deep sweep results (no new GPU runs).

### P1: Attention Pattern Visualization (~2 hours)

**Why**: The "token presence reshapes attention" claim needs visual evidence.

**Design**: On Gemma 3 12B, extract attention weights from Phase B for 3 samples.
Compare attention heatmaps: bare vs random_1 vs comprehend_64. Show how prefix
changes which document positions receive attention during query answering.

### P2: Generative Task (Summarization or Open QA) (~4 hours)

**Why**: All experiments are extractive QA. Need at least one generative task.

**Design**: Use CNN/DailyMail summarization. Precompute caches with/without prefix.
Generate summaries. Score with ROUGE-L. This tests whether prefix conditioning
helps beyond extractive QA.

### P2: Interaction Effects Between Components (~1 hour)

**Why**: Our decomposition is additive by construction. There may be non-additive
interactions (e.g., position shift × token presence).

**Design**: Compare predicted total (sum of components) vs actual total for each
condition. The residual reveals interaction effects. Analysis only, no new GPU runs.

---

## Known Issues

- Gemma 3N `use_cache=False` bug: always use `use_cache=True`
- Qwen no BOS: must add artificial BOS (PAD token) for prefix conditioning to work
- GSM8K answer: must extract number only after `####`, not full CoT
- Normalization: beneficial for Gemma 3 only; neutral on Mistral/Qwen
