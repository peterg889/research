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

## Directory structure
```
directed_kvcache_v4/
  .env                          # symlink -> v3/.env
  CLAUDE.md                     # This file
  lib/                          # symlinks -> v3/lib
  results/exp01/                # Experiment outputs
  experiments/
    01/
      01_production_kv_cache.ipynb
      build_exp01.py
```

## Known pitfalls
- See `directed_kvcache_v3/CLAUDE.md` for all architecture notes and pitfalls
- `os.umask(0o000)` at top of every notebook (JupyterLab vs CLI permissions)
- Never commit `.env` or HF tokens
- Use `#` comments inside `code(r"""...""")` blocks, not `"""..."""` docstrings
