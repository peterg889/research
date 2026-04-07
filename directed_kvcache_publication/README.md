# Directed KV Cache Construction — Publication

How prefix tokens during KV cache construction reshape document representations.

## Quick Orientation

| File | What It Is |
|------|-----------|
| **`paper/draft_v1.md`** | Paper draft (Sections 1-6 + appendix, ~430 lines) |
| **`TASKS.md`** | Remaining work with owners and priorities |
| **`EXPERIMENT_NOTES.md`** | Detailed experiment log with all findings |
| **`PLAN.md`** | High-level publication plan (quick reference) |

## Key Results

We decompose prefix conditioning into four independent components:

| Component | Gemma 3 12B | Gemma 3N 4B | Mistral 7B | Qwen 7B |
|-----------|:----------:|:----------:|:----------:|:-------:|
| 1. Position shift | -0.18 | **+0.64** | -0.18 | **+0.56** |
| 2. Token presence | **+0.76** | +0.18 | **+0.84** | -0.66 |
| 3. Vocabulary | -0.18 | -0.37 | -0.49 | -0.08 |
| 4. Word order | **+0.43** | -0.28 | +0.04 | **+0.29** |

Token presence is the dominant mechanism. Semantic word order only helps on larger models.

## Repository Structure

```
directed_kvcache_publication/
  README.md                          # This file
  TASKS.md                           # Remaining work (start here for contributors)
  EXPERIMENT_NOTES.md                # Detailed experiment log and findings
  PLAN.md                            # High-level publication plan

  paper/
    draft_v1.md                      # Paper draft with [TODO] placeholders

  model_adapters.py                  # Multi-model RoPE parameter extraction
  tests/
    test_pipeline_correctness.py     # 44 tests (24 unit + 20 GPU integration)

  experiments/01_multi_model/
    build_exp01.py                   # 3-condition × 4-model experiment
    build_condition_sweep.py         # 7-condition sweep (pre-fixes)
    build_deep_sweep.py              # 15-condition hypothesis sweep (with fixes)
    01_deep_sweep_executed.ipynb     # Executed deep sweep notebook

  results/
    exp01_multi_model/               # 3-condition results (400 samples)
    exp01_condition_sweep/           # 7-condition results (200 samples, pre-fix)
    exp01_deep_sweep/                # 15-condition results (200 samples, with fixes)

  00_publication_plan.ipynb          # Original planning notebook
```

## Setup

```bash
# Dependencies (same as directed_kvcache_v4)
pip install -r ../directed_kvcache_v4/requirements.txt
pip install timm  # needed for Gemma 3N

# HF token (needed for model access)
cp ../directed_kvcache_v4/.env.example .env
# Edit .env with your token

# Run tests (no GPU needed for unit tests)
cd directed_kvcache_publication
PYTHONPATH="../directed_kvcache_v4:." pytest tests/ -v -k "not slow"

# Run GPU integration tests (loads each model sequentially)
PYTHONPATH="../directed_kvcache_v4:." pytest tests/ -v -k "slow" -s
```

## Models Tested

| Model | Access | Notes |
|-------|--------|-------|
| Gemma 3 12B-IT | `google/gemma-3-12b-it` | Primary model, hybrid sliding+full attention |
| Gemma 3N E4B-IT | `google/gemma-3n-e4b-it` | Gemma 4 family, has `use_cache=False` bug |
| Mistral 7B-Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Apache 2.0, full attention |
| Qwen 2.5 7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | No native BOS — uses PAD as attention sink |

## Known Issues

See `EXPERIMENT_NOTES.md` for details on:
- Gemma 3N `use_cache=False` bug and workaround
- Qwen BOS token handling
- GSM8K answer extraction (must use number-only after `####`)
- Normalization is architecture-dependent (helps Gemma 3 only)

## Relationship to directed_kvcache_v4

This publication directory builds on 15 experiments in `../directed_kvcache_v4/`.
The v4 repo contains the original single-model (Gemma 3 12B) experiments, shared
library code (`lib/`), and detailed experiment notes. This directory adds multi-model
validation, the four-level decomposition analysis, and the paper draft.

The `lib/` modules from v4 are imported via `sys.path` — not copied. The
`model_adapters.py` in this directory extends `lib/rope.py` functions with
multi-model support (Gemma, Mistral, Qwen config differences).
