# Directed KV Cache v4

## Overview

Two parallel experiment tracks:

1. **Encoder-decoder** (Exp 01): Production-realistic test — does enrichment become
   redundant once the decoder already has the query as input?
2. **Decoder-only** (Exps 01-07): Systematic investigation of KV cache priming in
   causal LMs. Isolates structural vs semantic mechanisms.

## Models
- **T5Gemma 2 4B-4B**: Encoder-decoder (v4 Exp 01). See `directed_kvcache_v3/CLAUDE.md`.
- **Gemma 3 4B-IT** (`google/gemma-3-4b-it`): All decoder-only experiments (Exps 01-07).
  Also used for LLM surrogate generation in Exp 06. BF16, ~8GB VRAM.
  Prior executed notebooks used Gemma 2 2B (Exp 01) and Gemma 3 4B-PT (Exps 02-05).

## Directory structure
```
directed_kvcache_v4/
  .env                              # symlink -> v3/.env
  CLAUDE.md                         # This file
  lib/                              # symlinks -> v3/lib
  archive/
    01_encoder_decoder/             # Archived encoder-decoder experiment (T5Gemma)
  results/
    decoder_only/exp01-07/          # Decoder-only experiment outputs
  experiments/
    decoder_only/01-07/             # Decoder-only experiments
```

## Decoder-only technical approach
Two-phase KV cache scoring:
- **Phase A** (conditioning): `[BOS] + prefix_ids + [\n] + doc_ids` → build KV cache
  → slice first `1 + len(prefix) + len(\n)` entries → only doc KV remains
- **Phase B** (inference): `[\n + query + \n] + answer_ids` with custom `position_ids`
  starting at `phase_b_start` → score NLL on answer tokens only
- `slice_kv_cache()` helper removes prefix entries from `DynamicCache`
- Token-level prefix matching (Exps 05-07): all prefixed conditions use exactly Q
  token IDs (Q = number of query tokens), equalizing BOS removal, position offset,
  and cache length across conditions

## Encoder-decoder technical approach
- `decoder_input_ids = [BOS] + query_tokens + answer_tokens` passed explicitly
- NLL computed only on answer token positions
- Encoder outputs pre-computed with optional prefix truncation (cross-attention masking)

---

## Decoder-Only Experiment Log

### Exp 01: Surrogate KV Caching (Gemma 2 2B, N=400, SEED=42)
ALL surrogates beat oracle (recovery >100%), including adversarial. Oracle d=+0.440,
surr_reasonant d=+0.647 (146% recovery). Effect is primarily structural — prefix
length, not content, drives the benefit.

### Exp 02: Length vs Content Decomposition (Gemma 3 4B-PT, N=400)
Switched to larger model. Oracle d=+0.638. Random text of oracle length achieves 94%
recovery (d=+0.561). Semantic surrogates degrade relative to oracle on Gemma 3. Confirms
length, not content, is the driver. Structure accounts for 82-88% of the effect.

### Exp 03: Position vs Attention Isolation (Gemma 3 4B-PT, N=400)
Mechanism is **RoPE position offset** — doc tokens attending at higher absolute positions.
pos_offset_4 d=+0.779 (144% recovery, beats oracle). BOS removal alone d=+0.456 (56%).
Single newline prefix d=+0.754 (128%). Peak benefit at small position shifts (S=4 >> S=20).

### Exp 04: Position Sweep + Cache Surgery (Gemma 3 4B-PT, N=400)
Smooth position offset curve peaking at S=4. **prune_first_3** (remove BOS + first 3 doc
tokens from cache) achieves d=+0.803 (136% recovery) — best condition overall. Cache
surgery (pruning attention sinks) is more effective than prefix co-encoding.

### Exp 05: Semantic Priming in Isolation (Gemma 3 4B-PT, N=400)
All structural confounds equalized via **token-level prefix matching** (exactly Q tokens
per condition). With structure controlled: oracle d=+0.638 (100%), repeat_token d=+0.589
(97%), random_tokens d=+0.575 (98%), unrelated_query d=+0.614 (99%). All semantic
variants cluster within 96-102%. Semantic content has **negligible** effect on decoder-only
priming. best_structural (prune_first_3) still dominant at d=+0.803 (136%).

### Exp 06: Graded Semantic Relevance (Gemma 3 4B-IT, N=400, SEED=42) — PENDING
Port of v3 Exp 12. Single model (Gemma 3 4B-IT) for both surrogate generation and scoring.
7 conditions across the full semantic gradient (random → scrambled →
unrelated → same_topic → paraphrase → oracle), all token-level matched. Tests whether a
monotonic semantic gradient exists in decoder-only (v3 found rho=+0.943, p=0.005 in
encoder-decoder).

### Exp 07: Swapped-Query Paired Contrasts (Gemma 3 4B-IT, N=400, SEED=43) — PENDING
Port of v3 Exp 13. Maximum-power paired test: same doc scored with real vs swapped query,
token-level matched. 4 conditions (bare, oracle, swapped, random_matched). Tests whether
the paired semantic signal (v3: d=+0.166, p=2.3e-04, 33.4% semantic fraction) replicates
in decoder-only.

---

## Key Findings (Decoder-Only)

1. **Mechanism is structural (RoPE + attention sinks), NOT semantic.**
   Position offset and BOS/sink pruning explain >95% of the benefit.
2. **Cache surgery beats co-encoding.** prune_first_3 (d=+0.803) > oracle (d=+0.638).
3. **Token-level matching confirms near-zero semantics.** With structure equalized,
   all prefix content variants (oracle, random, repeat) perform within 4% of each other.
4. **Sharp contrast with v3 encoder-decoder.** v3 T5Gemma shows 10-15% semantic component
   via bidirectional encoder enrichment. Decoder-only has no such mechanism.
5. **Exps 06-07 will test whether any residual semantic gradient exists** when measured
   with the same fine-grained methodology used in v3.

## Known pitfalls
- See `directed_kvcache_v3/CLAUDE.md` for all architecture notes and pitfalls
- `os.umask(0o000)` at top of every notebook (JupyterLab vs CLI permissions)
- Never commit `.env` or HF tokens
- Use `#` comments inside `code(r"""...""")` blocks, not `"""..."""` docstrings
- Decoder-only notebooks use `sys.path.insert(0, "../../..")` (3 levels up from experiment dir)
