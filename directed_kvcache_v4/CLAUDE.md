# Directed KV Cache v4: Production-Realistic Encoder-Decoder Testing

## Overview
Tests whether surrogate-enriched encoder caching helps in a **production-realistic** setting
where the decoder already has the query as input.

v3 proved co-encoding [surrogate + document] enriches document representations (d=+0.408).
But the decoder never saw the query. In production:
1. **Offline**: Encode [surrogate + document] → cache encoder hidden states
2. **Online**: Query arrives → decoder receives query as input, cross-attends to cache

**Key question**: Does enrichment become redundant once the decoder has the query?

## Model
T5Gemma 2 4B-4B (same as v3). See `directed_kvcache_v3/CLAUDE.md` for architecture details.

## Key difference from v3
- **v3**: Decoder only receives answer tokens (no query)
- **v4**: Decoder receives `[BOS] + query_tokens + answer_tokens` as `decoder_input_ids`
- NLL is computed only on answer token positions

## Technical approach
Pass `decoder_input_ids` explicitly to `model.forward()` alongside `encoder_outputs`.
When both are provided, the model uses decoder_input_ids directly (labels are only
shifted to create decoder_input_ids when decoder_input_ids is None).

## Experiment plan
See `EXPERIMENT_PLAN.md` for the full experiment log and forward plan.

## Key results

**Exp 01** (MS MARCO, ~98 tok docs): Enrichment survives with query in decoder: d=+0.228
(61% of v3 baseline). Mechanism shifted: structural component collapsed (85%→35%),
content now dominant. Doc-keyword surrogate works (d=+0.148, 65% of oracle). Template fails.

**Exp 02** (MS MARCO, padded 98–4096 tok): Enrichment GROWS with doc length. Oracle d
rises from +0.238 to +0.439. Random prefix significant at 256+ tokens — structural
mechanism regains dominance on longer docs even with query in decoder.

**Exp 03** (neural-bridge, naturally ~604w docs): Surrogates beat oracle. Random d=+0.624
> surr_doc d=+0.502 > oracle d=+0.306. Structural fraction 204%. Real query creates
semantic interference on long docs. ANY short prefix works — no surrogate generation needed.

**Exp 04** (MS MARCO, prefix content sweep): kw10 is best surrogate (d=+0.186, 82% oracle).
Keyword density is inverted-U (kw10 > kw5 > kw20). First sentence is catastrophic
(d=-0.298). Document-specific keywords ≈ wrong-doc keywords (ns). Benefit is ~72%
vocabulary-type, ~28% semantic. Optimal prefix: 10-15 disconnected keyword-like tokens.

**Exp 05** (Truncation wound mechanism): The first-sentence catastrophe requires BOTH
coherence AND overlap — interaction effect d=-0.361. Wrong-doc coherent sentence: +0.063
(harmless). Shuffled same-doc sentence: +0.078 (harmless). Only the combination is
catastrophic. Deep bidirectional bonds + truncation = dangling references.

**Exp 06** (Factoid split): Reverses v3 pattern — long answers (>5w) show STRONGER v4
enrichment (d=+0.412 vs +0.284). Structural fraction increases with answer length (33%
at 1-2w to 74% at 21+w). 3-5w sweet spot: surrogate beats oracle (101%).

**Exp 07** (Decoder attention probing): 2×2 factorial confirms 35% redundancy between
encoder prefix and decoder query (interaction d=+0.316, ***). Query tokens absorb only
5.5% of answer attention — modest buffer. Main finding: oracle encoding shifts answer-token
budget from cross-attention (37%→13%) to self-attention (63%→87%). The structural collapse
is because decoder query reduces cross-attention reliance, not because query acts as a
large attention buffer.

**Exp 08** (Decoder length control, 500/500): CORRECTS Exp 07. The "35% redundancy" is
**entirely a length artifact**. 2×3 factorial {bare,oracle}×{nq,random_q,q}: with bare
encoder, nq→q improvement is 55% length + 45% semantic. With oracle encoder, length
effect vanishes (d=-0.014, ns) — 100% semantic. Interaction: length component d=+0.324
(102% of total d=+0.316), semantic component d=-0.097 (slightly super-additive, ns).
Encoder and decoder provide INDEPENDENT semantic benefits but REDUNDANT structural
benefits. Random decoder tokens absorb 14.3% attention (vs 5.5% for real query, 2.6×).

**Exp 09** (Cross-model sweep): Enrichment GENERALIZES to all 4 models tested.
flan-t5-base d=+0.251 (***), flan-t5-large d=+0.320 (***), flan-t5-xl d=+0.430 (***),
bart-large d=+0.144 (**). Structural fraction varies wildly: 7% (flan-t5-xl) to 77%
(flan-t5-large). Flan-T5 models have negative d_dec_q (query in decoder HURTS — against
their training distribution). Inverted-U for structural fraction vs model quality.

## Directory structure
```
directed_kvcache_v4/
  .env                          # symlink -> v3/.env
  CLAUDE.md                     # This file
  EXPERIMENT_PLAN.md            # Experiment log + forward plan
  lib/                          # symlinks -> v3/lib
  results/exp01/                # Experiment outputs
  results/exp02/
  results/exp03/
  results/exp08/
  results/exp09/
  experiments/
    encoder_decoder/            # T5Gemma encoder-decoder experiments
      01/
        01_production_kv_cache.ipynb
        build_exp01.py
        build_examples.py
        01_examples.ipynb
      02/
        02_length_scaling.ipynb
        build_exp02.py
      03/
        03_neural_bridge.ipynb
        build_exp03.py
      04/
        04_prefix_optimization.ipynb
        build_exp04.py
      05/
        05_truncation_wound.ipynb
        build_exp05.py
      06/
        06_factoid_split.ipynb
        build_exp06.py
      07/
        07_decoder_attention_probing.ipynb
        build_exp07.py
      08/
        08_decoder_length_control.ipynb
        build_exp08.py
      09/
        09_cross_model_sweep.ipynb
        build_exp09.py
```

## Path convention
Notebooks live at `experiments/encoder_decoder/XX/` — three levels below v4 root.
Inside notebook code cells: `sys.path.insert(0, "../../..")` to reach lib/.
Results: `Path("../../../results/expXX")`.

## Known pitfalls
- See `directed_kvcache_v3/CLAUDE.md` for all architecture notes and pitfalls
- `os.umask(0o000)` at top of every notebook (JupyterLab vs CLI permissions)
- Never commit `.env` or HF tokens
- Use `#` comments inside `code(r"""...""")` blocks, not `"""..."""` docstrings
