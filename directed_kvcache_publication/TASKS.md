# Remaining Tasks — Directed KV Cache Publication

Status key: `[ ]` not started, `[~]` in progress, `[x]` done

## Technical: Experiments to Run

### P0 — Blocking for Submission

- [ ] **Downstream generative accuracy** (~6 hours GPU)
  - Run on Gemma 3 12B + Mistral 7B, 3 conditions (bare, comprehend, random), SQuAD + GSM8K
  - Use `model.generate()` with cached KV from Phase A, greedy decoding
  - Score: EM + Token F1 (SQuAD), Accuracy (GSM8K — extract number after `####`)
  - Report: accuracy table + Spearman correlation between NLL improvement and accuracy improvement
  - Owner: ___
  - Files: build from `experiments/01_multi_model/build_deep_sweep.py` template; scoring functions in `00_publication_plan.ipynb` Cell 8

- [ ] **Expand deep sweep to 4 datasets** (~3 hours GPU)
  - Add DROP and TriviaQA to the 15-condition deep sweep across all 4 models
  - Strengthens generalization claims beyond SQuAD + GSM8K
  - Owner: ___
  - Files: modify `DS_SPECS` in `build_deep_sweep.py`

### P1 — Strengthens Paper

- [ ] **Bootstrap confidence intervals** (~30 min CPU)
  - 10,000 bootstrap resamples on each decomposition component
  - Report 95% CIs in all results tables
  - Add `bootstrap_ci()` function to analysis utilities
  - Owner: ___
  - Files: pure analysis on existing `results/exp01_deep_sweep/` checkpoint files

- [ ] **Latency measurements** (~30 min GPU)
  - Time Phase A with/without prefix, select+reposition, normalization, Phase B
  - Use `torch.cuda.Event` for accurate GPU timing, 100 iterations
  - Show Phase A overhead ~5-10%, Phase B overhead = 0%
  - Owner: ___

- [ ] **Normalization ablation** (~2 hours GPU)
  - Run bare + comprehend_64 WITH and WITHOUT normalization on all 4 models × 2 datasets
  - 2×2 factorial (norm × prefix) to quantify interaction
  - Owner: ___

### P2 — Nice to Have

- [ ] **Attention pattern visualization** (~2 hours GPU)
  - Extract attention weights from Phase B: bare vs random_1 vs comprehend_64
  - Show which document positions gain/lose attention with prefix
  - 3 samples from SQuAD on Gemma 3 12B
  - Owner: ___

- [ ] **Generative task (summarization)** (~4 hours GPU)
  - CNN/DailyMail or XSum summarization with ROUGE-L scoring
  - Tests whether prefix effect extends beyond extractive QA
  - Owner: ___

- [ ] **Interaction effects analysis** (~1 hour CPU)
  - Compare predicted decomposition sum vs actual total for each condition
  - Residual reveals non-additive interactions between components
  - Owner: ___

---

## Technical: Code & Testing

- [x] **Pipeline correctness tests** — 44 tests (24 unit + 20 integration)
  - Select, reposition, normalization, BOS handling, scoring logic all tested
  - Run: `cd directed_kvcache_publication && PYTHONPATH="../directed_kvcache_v4:." pytest tests/ -v`

- [ ] **Fix GSM8K answer extraction** in experiment code
  - Must use `raw_answer.split('####')[-1].strip()` (number only)
  - Currently correct in `build_deep_sweep.py`, but `build_exp01.py` and `build_condition_sweep.py` still have the bug
  - Owner: ___

- [ ] **Fix Qwen BOS** in all experiment code
  - Must use PAD token as artificial BOS for models with `bos_token_id = None`
  - Currently correct in `build_deep_sweep.py` but not in older experiment files
  - Owner: ___

---

## Writing: Paper Draft

The draft is at `paper/draft_v1.md`. Sections with `[TODO]` need work.

- [ ] **Section 4.1**: Insert full condition ranking table (data in `results/exp01_deep_sweep/combined_summary.json`)
- [ ] **Section 4.2**: Insert decomposition table (the key result — data computed, needs formatting)
- [ ] **Section 4.3**: Create prefix length figure (data in deep sweep results)
- [ ] **Section 4.4**: Write after downstream accuracy experiments are done
- [ ] **Section 4.5**: Write after generative task experiments are done
- [ ] **Appendix B**: Insert full per-model per-dataset tables
- [ ] **Add EPIC and KVLink citations** (arXiv:2410.15332 and arXiv:2502.16002) — directly relevant to our positioning

---

## Writing: Framing & Positioning

- [ ] **Review the "four-level decomposition" framing** — is this the clearest way to present it?
  - Alternative: lead with the practical finding ("one random token helps") and decompose afterward
  - Alternative: lead with the model capacity threshold finding
  - Needs group discussion

- [ ] **Sharpen the "why does it work" story**
  - Token presence: connects to attention sinks [Xiao et al., 2024]
  - Position shift: connects to EPIC [Hu et al., 2025] (position-independent caching)
  - Word order: connects to prefix tuning [Li & Liang, 2021]
  - Vocabulary: novel negative finding — scrambled instructions are worse than random

- [ ] **Clarify positioning vs CacheBlend and TurboRAG**
  - Our "bare" baseline IS TurboRAG (precompute cache, use for queries)
  - CacheBlend recomputes tokens at query time; we modify at construction time
  - We're complementary to both — our prefix conditioning could be added to either system

- [ ] **Decide on venue/format**
  - 8-page conference paper (EMNLP, NeurIPS, COLM)?
  - Workshop paper (shorter, faster turnaround)?
  - Preprint first, then submit?

---

## Organizational

- [ ] **Internal review** of paper draft by team members
- [ ] **Sanity check decomposition math** — verify the formula is correct and additive
- [ ] **Reproduce key results** — have a second person run the deep sweep independently
- [ ] **Figure generation** — create publication-quality figures from results data
  - Decomposition table (main result)
  - Length curves (4 panels, one per model)
  - Condition ranking heatmap (models × conditions)
  - Methodology diagram (Phase A → select → reposition → Phase B)
