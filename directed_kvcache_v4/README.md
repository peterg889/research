# Directed KV Cache v4

Can surrogate-enriched KV caches improve document representations for downstream QA?
This repository investigates prefix conditioning of KV caches in large language models,
progressing from static text prefixes through learned soft prompts to cache routing.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up HuggingFace token (required for model access)
#    Get a token at https://huggingface.co/settings/tokens
#    Then create a .env file:
cp .env.example .env
# Edit .env and replace the placeholder with your actual token

# 3. Accept the Gemma 3 model license at:
#    https://huggingface.co/google/gemma-3-12b-it
```

## Quick Start

```bash
# Run tests (no GPU needed)
pytest lib/tests/ -v

# Run an experiment (example: Exp 13)
cd /path/to/directed_kvcache_v4
python3 experiments/decoder_only/13/build_exp13.py          # Generate notebook
cd experiments/decoder_only/13
papermill 13_ood_misleading_hero.ipynb 13_ood_misleading_hero_executed.ipynb --no-progress-bar
```

**GPU requirements**: Most experiments need a single GPU with ~24GB VRAM
(Gemma 3 12B-IT in BF16). Exp 15 (routing) is CPU-only.

## Repository Structure

```
directed_kvcache_v4/
  README.md                       # This file
  CLAUDE.md                       # Architecture notes, scoring approaches, known pitfalls
  DECODER_ONLY_NOTES.md           # Decoder-only experiment log (Exps 01-15)
  ENCODER_DECODER_NOTES.md        # Encoder-decoder experiment log (Exps 01-10)
  requirements.txt                # Python dependencies
  lib/                            # Shared library (analysis, cache ops, RoPE, quantization)
    tests/                        # 76 pytest tests
  experiments/
    decoder_only/01-15/           # Main experiment track (see below)
    encoder_decoder/01-10/        # T5Gemma encoder-decoder experiments
    prefix_lm/01-05/              # Prefix attention experiments (notebooks only)
  results/
    decoder_only/exp01-exp15/     # JSON results, checkpoints, charts
    exp01-exp09/                  # Encoder-decoder results
  diagrams/                       # ManimCE presentation diagrams
```

## Experiment Arc (Decoder-Only)

The decoder-only track is the primary research thread (15 experiments on Gemma 3 12B-IT):

| Phase | Exps | Question | Key Finding |
|-------|------|----------|-------------|
| **Semantic probing** | 01-03 | Does prefix content matter? | Generic task-framing beats LLM surrogates; oracle query *hurts* |
| **Decomposition** | 04-05 | Structure vs vocabulary vs meaning? | Meaning grows with length; dominates at L=64 (50% of effect) |
| **Scaling** | 06 | How does this generalize? | 14 datasets: GSM8K champion (d=+1.33); reasoning complexity drives benefit |
| **Normalization** | 07-09 | Why does int8 *improve* NLL? | absmax/qmax normalization is the mechanism (97% of benefit); universal fix |
| **Deconfounding** | 10 | Is compression truly robust? | True compression cost isolated after removing normalization confound |
| **Hero runs** | 11-13 | Unified story across conditions? | 13-15 conditions, 7 datasets; all prefixes help, even noise (d=+0.10) |
| **Soft prompts** | 14 | Can we learn better prefixes? | 245K params double the best text (d=0.85 vs 0.43); random init > warm-start |
| **Routing** | 15 | Which cache for which query? | Routing closes 54% of oracle gap; dataset identity is strongest signal |

## How Experiments Work

Each experiment follows the same pattern:

1. **`build_expNN.py`** generates a Jupyter notebook using `nbformat` with `ast.parse()` syntax checking
2. **`papermill`** executes the notebook, producing `*_executed.ipynb`
3. Results checkpoint every 20 samples to `results/decoder_only/expNN/`
4. Presentation charts are built by separate `build_presentation_charts.py` scripts

**Two-phase KV cache scoring** (all decoder-only experiments):
- **Phase A**: `[BOS, prefix, \n, doc]` → build cache → select BOS+doc entries → RoPE reposition → normalize
- **Phase B**: `[\n, query, \n, answer]` → score with cached KV → NLL on answer tokens only

See `CLAUDE.md` for detailed scoring mechanics and critical pitfalls.

## Shared Library (`lib/`)

| Module | Purpose |
|--------|---------|
| `analysis.py` | Cohen's d, win rate, paired t-test |
| `cache.py` | Deep copy, prefix construction, scrambling |
| `data.py` | Word counting utility |
| `quantization.py` | Simulated quantization, normalization round-trip, clipping |
| `rope.py` | RoPE repositioning, KV cache selection, layer introspection |

All functions have docstrings, type hints, and examples. Run `pytest lib/tests/ -v` (76 tests).

## Key Documentation

- **`CLAUDE.md`** — Start here for architecture, scoring approaches, and critical bugs
- **`DECODER_ONLY_NOTES.md`** — Detailed experiment log with results for all 15 experiments
- **`ENCODER_DECODER_NOTES.md`** — Encoder-decoder track (T5Gemma, production-realistic)
- **`EXPERIMENT_PLAN.md`** — Prefix-LM mechanistic analysis (vocabulary bridging, semantic forcing)
