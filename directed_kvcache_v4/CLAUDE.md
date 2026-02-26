# Directed KV Cache v4

## Overview

Three parallel experiment tracks investigating whether surrogate-enriched KV caching
improves document representations for downstream QA:

1. **Encoder-decoder** (Exps 01-10): Production-realistic test — does enrichment become
   redundant once the decoder already has the query as input?
2. **Decoder-only** (Exps 01-09): Systematic investigation of KV cache priming in
   causal LMs. Isolates structural vs semantic mechanisms. Exps 07-09 discover that
   KV cache normalization universally improves NLL. Old Exps 02-07 archived.
3. **Prefix LM** (Exps 01-03): Causal vs bidirectional prefix attention on decoder-only models.

## Experiment Notes

- **`ENCODER_DECODER_NOTES.md`** — Encoder-decoder experiment log (Exps 01-10),
  Prefix LM experiments, v3→v4 mechanism analysis, research arc summary.
- **`DECODER_ONLY_NOTES.md`** — Decoder-only experiment log (Exps 01-09),
  technical approach (BOS-retained repositioning), key findings across 14 datasets,
  quantization diagnosis (Exps 07-08), and KV cache normalization (Exp 09).
- **`EXPERIMENT_PLAN.md`** — Legacy file (same content now in `ENCODER_DECODER_NOTES.md`).

## Models

- **T5Gemma 2 4B-4B**: Encoder-decoder (v4 Exps 01-10). See `directed_kvcache_v3/CLAUDE.md`.
- **Gemma 3 12B-IT** (`google/gemma-3-12b-it`): Decoder-only Exps 01-09;
  Prefix LM Exps 01-03. BF16, single GPU. Used for both LLM surrogate generation and scoring.
- **Gemma 3 4B-IT** (`google/gemma-3-4b-it`): Archived decoder-only Exps 06-07.
  Prior executed notebooks used Gemma 2 2B (Exp 01) and Gemma 3 4B-PT (Exps 02-05).
- **Cross-model** (Exp 09): flan-t5-base, flan-t5-large, flan-t5-xl, bart-large.

## Directory Structure

```
directed_kvcache_v4/
  .env                          # symlink -> v3/.env
  CLAUDE.md                     # This file (project overview)
  ENCODER_DECODER_NOTES.md      # Enc-dec + prefix LM experiment notes
  DECODER_ONLY_NOTES.md         # Decoder-only experiment notes
  EXPERIMENT_PLAN.md            # Legacy (enc-dec log, kept for reference)
  lib/                          # Shared library modules
    __init__.py                 # Module docstring
    analysis.py                 # cohens_d, win_rate, paired_ttest
    cache.py                    # deep_copy_cache, make_prefix
    data.py                     # count_words
    quantization.py             # simulated_quantize, quantize_kv_cache,
                                #   norm_roundtrip_kv_cache, clip_kv_cache
    rope.py                     # build_layer_inv_freqs, get_layer_types,
                                #   rotate_half, select_kv_cache,
                                #   reposition_kv_cache
    tests/                      # pytest suite (76 tests)
  results/exp01/ ... exp09/     # Encoder-decoder experiment outputs
  results/prefix_lm_exp01/
  results/decoder_only/exp01/ ... exp09/  # Decoder-only experiment outputs
  diagrams/                     # Manim presentation diagrams (01-08 PNGs + source)
  experiments/
    encoder_decoder/            # T5Gemma encoder-decoder experiments
      01/ 02/ 03/ 04/ 05/ 06/ 07/ 08/ 09/ 10/
    prefix_lm/                  # Decoder-only prefix LM experiments
      01/ 02/ 03/ 04/ 05/
    decoder_only/               # Decoder-only KV cache priming
      01/ 02/ 03/ 04/ 05/ 06/ 07/ 08/ 09/
      archive/                  # Archived old 02-07 (look-ahead bug / different model)
```

## Path Convention

Notebooks live at `experiments/{encoder_decoder,prefix_lm,decoder_only}/XX/` — three
levels below v4 root. Inside notebook code cells: `sys.path.insert(0, "../../..")`
to reach lib/. Results: `Path("../../../results/expXX")` or
`Path("../../../results/decoder_only/expXX")`.

## Scoring Approaches

### Encoder-decoder
- `decoder_input_ids = [BOS] + query_tokens + answer_tokens` passed explicitly
- NLL computed only on answer token positions
- Encoder outputs pre-computed with optional prefix truncation (cross-attention masking)

### Decoder-only (CORRECTED — BOS-retained repositioning)
Two-phase KV cache scoring:
- **Phase A** (conditioning): `[BOS] + prefix_ids + [\n] + doc_ids` → build KV cache
  → `select_kv_cache()` keeps BOS + doc entries (skips prefix + \n)
  → `reposition_kv_cache()` rotates doc keys via RoPE
  → Cache has `1+D` entries (BOS at 0, doc at positions 1..D)
- **Phase B** (inference): `[\n + query + \n] + answer_ids` with `position_ids`
  starting at `D+1`. `cache_position` auto-generated from `cache.get_seq_length()`.
  No explicit `cache_position` — this prevents the 1-token look-ahead bug.

## Known Pitfalls

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
