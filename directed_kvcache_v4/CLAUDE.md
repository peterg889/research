# Directed KV Cache v4

## Overview

Two parallel experiment tracks:

1. **Encoder-decoder** (Exps 01-10): Production-realistic test — does enrichment become
   redundant once the decoder already has the query as input?
2. **Decoder-only** (Exps 01-03): Systematic investigation of KV cache priming in
   causal LMs. Isolates structural vs semantic mechanisms. Old Exps 02-07 archived.
3. **Prefix LM** (Exps 01-03): Causal vs bidirectional prefix attention on decoder-only models.

## Models
- **T5Gemma 2 4B-4B**: Encoder-decoder (v4 Exps 01-10). See `directed_kvcache_v3/CLAUDE.md`.
- **Gemma 3 12B-IT** (`google/gemma-3-12b-it`): Decoder-only Exps 01 (rerun), 02, and 03;
  Prefix LM Exps 01-03. BF16, single GPU. Used for both LLM surrogate generation and scoring.
- **Gemma 3 4B-IT** (`google/gemma-3-4b-it`): Archived decoder-only Exps 06-07.
  Prior executed notebooks used Gemma 2 2B (Exp 01) and Gemma 3 4B-PT (Exps 02-05).
- **Cross-model** (Exp 09): flan-t5-base, flan-t5-large, flan-t5-xl, bart-large.

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

**Exp 10** (T5 size scaling, planned): Tests standard (non-instruction-tuned) T5 models
(t5-small through t5-3b) for a clean size-scaling curve. Flan-T5 had negative d_dec_q
(query in decoder hurts — against training distribution), so standard T5 should give
cleaner results.

**Prefix LM Exp 01** (Gemma 3 12B IT, COMPLETE): Bidirectional attention HURTS
decoder-only models (d=-0.727, ***) — disrupts causal-trained representations. BUT
causal prefixes work: d_causal_oracle=+0.452 (***), d_causal_random=+0.475 (***),
d_causal_surr_doc=+0.461 (***). All surrogates help under causal attention. Under
prefix_lm, oracle enrichment is ns (d=+0.059, p=0.19). Structural fraction 140%.
The enrichment transfers via the CAUSAL channel, not bidirectionality. Two-pass
design: Phase A caches [BOS,surr,doc] KVs, Phase B evaluates [query,answer].

## Directory structure
```
directed_kvcache_v4/
  .env                          # symlink -> v3/.env
  CLAUDE.md                     # This file
  EXPERIMENT_PLAN.md            # Experiment log + forward plan
  lib/                          # Shared analysis/data utilities
  results/exp01/                # Encoder-decoder experiment outputs
  results/exp02/
  results/exp03/
  results/exp08/
  results/exp09/
  results/prefix_lm_exp01/
  results/decoder_only/exp01/   # Decoder-only experiment outputs
  results/decoder_only/exp02/
  results/decoder_only/exp03/
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
      10/
        10_t5_size_scaling.ipynb
        build_exp10.py
    prefix_lm/              # Decoder-only prefix LM experiments
      01/
        01_prefix_lm_enrichment.ipynb
        build_exp01.py
        test_attention_masks.py
      02/
        02_semantic_isolation.ipynb
        build_exp02.py
      03/
        03_surrogate_content_sweep.ipynb
        build_exp03.py
    decoder_only/01/                # Exp 01: Surrogate prefix conditioning (rerun)
    decoder_only/02/                # Exp 02: Token-matched semantic probing
    decoder_only/03/                # Exp 03: Hard-example semantic isolation (cross-dataset)
    decoder_only/archive/           # Archived old 02-07 (look-ahead bug / different model)
```

## Path convention
Notebooks live at `experiments/{encoder_decoder,prefix_lm}/XX/` — three levels below v4 root.
Inside notebook code cells: `sys.path.insert(0, "../../..")` to reach lib/.
Results: `Path("../../../results/expXX")` or `Path("../../../results/prefix_lm_expXX")`.
Decoder-only notebooks use the same convention from `experiments/decoder_only/XX/`.

## Encoder-decoder technical approach
- `decoder_input_ids = [BOS] + query_tokens + answer_tokens` passed explicitly
- NLL computed only on answer token positions
- Encoder outputs pre-computed with optional prefix truncation (cross-attention masking)

## Decoder-only technical approach (CORRECTED — BOS-retained repositioning)
Two-phase KV cache scoring:
- **Phase A** (conditioning): `[BOS] + prefix_ids + [\n] + doc_ids` → build KV cache
  → `select_kv_cache()` keeps BOS + doc entries (skips prefix + \n)
  → `reposition_kv_cache()` rotates doc keys from `[1+P+NL,...,P+NL+D]` to `[1,...,D]`
  → Cache has `1+D` entries (BOS at 0, doc at positions 1..D)
- **Phase B** (inference): `[\n + query + \n] + answer_ids` with `position_ids`
  starting at `D+1`. `cache_position` auto-generated from `cache.get_seq_length()=1+D`.
  No explicit `cache_position` — this prevents the 1-token look-ahead bug.
- **CRITICAL BUG FIX**: Previous approach sliced BOS, leaving cache length=D but
  cache_position=D+1. The causal mask `kv_idx <= q_idx` with q_idx=D+1 allowed
  attending to the NEXT Phase B token (1-token look-ahead). ALL Exps 01-05 results
  were inflated by this bug. Fix: retain BOS so cache length=D+1=cache_position.
- Token-level prefix matching (Exp 02): all prefixed conditions use exactly Q
  token IDs (Q = number of query tokens), equalizing BOS removal, position offset,
  and cache length across conditions

---

## Decoder-Only Experiment Log

### Exp 01: Surrogate KV Caching — RERUN with look-ahead fix (Gemma 3 12B-IT, N=400, SEED=42)
**BOS-retained repositioning (corrected).** Previous results (Gemma 2 2B) were entirely
driven by a 1-token look-ahead bug. With correct masking:
- **Oracle HURTS**: d=-0.151 (**), win=32.0% — conditioning worsens NLL
- **surr_extractor is BEST**: d=+0.264 (***), win=68.0% — data extraction prompt helps
- **surr_universal weak**: d=+0.079 (ns). Other surrogates near zero.
- **adversarial neutral**: d=+0.007 (ns). Off-topic prefix has no effect.
- **adv_instruct HURTS**: d=-0.199 (***). Anti-instruction corrupts representations.
- **oracle_full HURTS MOST**: d=-0.362 (***). Full cache (Phase B attends to prefix) worst.
- **Ranking**: surr_extractor best in 28.2% of samples (mean rank 3.55), oracle worst
  among semantic conditions (mean rank 6.71).
- **Hardness gradient**: Oracle hurts in Q1-Q4, helps slightly only in Q5 (hardest).
The effect IS semantic (content-dependent), but oracle conditioning is counter-productive.
Task-framing prefixes (data extraction) can modestly improve document representations.

### Exp 02: Token-Matched Semantic Probing with LLM Surrogates (Gemma 3 12B-IT, N=400, SEED=42) — PENDING
Definitive experiment eliminating ALL structural confounds via token-level prefix matching.
13 conditions spanning full semantic gradient + LLM document-specific surrogates.
BOS-retained repositioning on same model as Exp 01 rerun.
- **Semantic gradient**: random_tokens, repeat_token, scrambled_oracle, unrelated_query,
  same_topic (LLM), paraphrase (LLM), oracle — tests monotonicity with Spearman rho
- **LLM surrogates**: llm_extract, llm_question, llm_summarize — document-specific
  conditioning generated by same model
- **Fixed controls**: extractor_matched (generic task-framing), adversarial_matched
- **Key analyses**: structural decomposition (random/oracle ratio), LLM doc-specific vs
  generic (paired test), hardness interaction (5 quintiles x 13 conditions), per-sample
  ranking
- **Token invariant**: all 12 prefixed conditions use exactly Q tokens per sample

### Exp 03: Hard-Example Semantic Isolation Across Datasets (Gemma 3 12B-IT, N=400×4, SEED=42)
Isolates the semantic effect by restricting to hard examples (top 40% by bare NLL) and
measuring semantic delta above the structural baseline (condition - random_tokens).
Tests generalization across 4 diverse QA datasets: MS MARCO, SQuAD 2.0, TriviaQA, HotpotQA.
- **MS MARCO**: reuses Exp 02 results (same model, same scoring)
- **3 new datasets**: SQuAD 2.0, TriviaQA (rc.wikipedia), HotpotQA (distractor)
- **13 conditions**: same as Exp 02 (token-level matched)
- **Hard selection**: top 40% by bare NLL per dataset (~160 samples each)
- **Key metric**: semantic_delta(C) = NLL(random_tokens) - NLL(C)
- BOS-retained repositioning + token-level prefix matching (identical to Exp 02)
- **Bug fix**: Sliding attention layers cache only (sliding_window-1) entries. When
  Phase A total tokens > 1023, select_kv_cache indexed OOB. Fix: dynamically truncate
  doc to fit within sliding cache limit when prefix is present.
- **Results** (pooled across 4 datasets, ranked by semantic delta d):
  - **extractor_matched d=+0.357 (\*\*\*)**: generic task-framing BEST by far
  - adversarial_matched d=+0.169 (\*\*\*)
  - repeat_token d=+0.124 (\*\*)
  - llm_summarize d=+0.052 (ns), llm_extract d=-0.023 (ns)
  - paraphrase d=-0.122 (\*\*), same_topic d=-0.229 (\*\*\*)
  - **oracle d=-0.253 (\*\*\*)**: oracle HURTS even in hard examples
  - **llm_question d=-0.343 (\*\*\*)**: worst LLM surrogate
- **Semantic gradient**: 0/4 datasets show monotonic gradient (mean rho=+0.15, ns)
- **LLM doc-specific vs generic**: generic extractor_matched BEATS all LLM surrogates
  across all 4 datasets (p<0.001). Document-specific content does NOT help.
- **Cross-dataset consistency**: 7/11 conditions have same sign across all 4 datasets

### Old Exps 02-07: ARCHIVED
Old Exps 02-05 invalidated by 1-token look-ahead bug. Old Exps 06-07 used slice_kv_cache
without BOS retention on Gemma 3 4B-IT — not directly comparable. All moved to
`experiments/decoder_only/archive/`.

---

## Key Findings (Decoder-Only)

### CRITICAL: 1-token look-ahead bug invalidates Exps 01-05
All previous Exps 01-05 used `cache_position = position_ids = [D+1,...]` after slicing
BOS from cache (cache length = D). The causal mask `kv_idx <= q_idx` with q_idx=D+1
allowed attending to the NEXT Phase B token. This inflated all results (oracle d=+0.44
to +0.80). The "structural benefit" (RoPE shift, BOS removal) was entirely look-ahead.

### Corrected findings (Exp 01 rerun, Gemma 3 12B-IT, N=400)
1. **Oracle conditioning HURTS** (d=-0.151, **). The real query as prefix worsens NLL.
2. **Task-framing prefixes can help.** surr_extractor d=+0.264 (***). Data extraction
   prompt improves doc representations for downstream QA.
3. **The effect IS semantic** — content determines direction: surr_extractor (+0.26),
   adversarial (0.00), oracle (-0.15), adv_instruct (-0.20), oracle_full (-0.36).
4. **Full cache worst.** Phase B attending to prefix cache entries hurts most (d=-0.36).
5. **Hardness interaction.** Oracle helps only for Q5 (hardest 20% of questions).
6. **Previous "structural" findings were look-ahead artifacts.** RoPE position shift,
   BOS removal, attention sink pruning — all driven by the masking bug.
7. **Exps 02-07 archived.** Exps 02-05 had look-ahead bug; Exps 06-07 used different
   model/approach. Exp 02 supersedes with token-matched design on same model.

### Exp 03: Cross-dataset hard-example analysis (Gemma 3 12B-IT, N=160×4)
1. **No semantic gradient in hard examples.** 0/4 datasets show monotonic gradient
   (mean Spearman rho=+0.15, all ns). Even restricting to hard samples doesn't reveal
   a meaningful content-relevance → NLL relationship.
2. **Generic task-framing BEST.** extractor_matched pooled d=+0.357 (***), consistent
   across all 4 datasets. The effect is robust and domain-general.
3. **LLM doc-specific content does NOT help beyond generic framing.** All 3 LLM surrogates
   (extract, question, summarize) LOSE to generic extractor_matched (p<0.001 each).
   Document-specific conditioning hurts compared to generic task framing.
4. **Oracle HURTS in hard examples too.** oracle pooled d=-0.253 (***). The actual query
   as prefix worsens NLL even when restricted to the hardest questions.
5. **Structural effects dominate.** repeat_token (d=+0.124, **) outperforms all semantic
   conditions except extractor_matched and adversarial_matched. Content is less important
   than the structural "activation" of task-framing tokens.
6. **Cross-dataset consistency.** 7/11 conditions have consistent sign across all 4 datasets.
   The pattern is robust: generic framing helps, semantic content hurts or is neutral.

## Known pitfalls
- See `directed_kvcache_v3/CLAUDE.md` for all architecture notes and pitfalls
- `os.umask(0o000)` at top of every notebook (JupyterLab vs CLI permissions)
- Never commit `.env` or HF tokens
- Use `#` comments inside `code(r"""...""")` blocks, not `"""..."""` docstrings
- Decoder-only notebooks use `sys.path.insert(0, "../../..")` (3 levels up from experiment dir)
- **CRITICAL: cache_position in Phase B** — NEVER pass explicit `cache_position` to Phase B
  when using `past_key_values`. Let the model auto-generate it from `cache.get_seq_length()`.
  Passing `cache_position > cache.get_seq_length()` creates a look-ahead in `kv_idx <= q_idx`.
  The fix: retain BOS in cache so length matches expected position, or omit `cache_position`.
- **CRITICAL: sliding window cache limit** — Gemma 3 sliding attention layers cache only
  `sliding_window - 1` entries (1023 for window=1024). When `select_kv_cache()` uses
  uniform indices across all layers, total Phase A tokens must not exceed this limit.
  Fix: dynamically truncate doc to `SLIDING_CACHE_LIMIT - 1 - P - NL` when prefix is used.
  Otherwise `IndexKernel.cu:111` assert (OOB index on sliding layers).
- **CRITICAL: use `dtype=` not `torch_dtype=`** for model loading — `torch_dtype` is deprecated
  in newer transformers and may cause warnings or errors.
- Run papermill from experiment directory (`cd experiments/decoder_only/01 && papermill ...`)
  to ensure `sys.path.insert(0, "../../..")` resolves correctly.
