# Directed KV Cache: Experimental Lab Notebook

## Research Question
Can we improve a language model's ability to answer queries about a document by "priming" the document's KV cache with a surrogate query generated at indexing time? This would allow pre-computing better document representations without knowing the user's query in advance.

## Setup
- **Models:** Mistral-7B-Instruct-v0.2 (4-bit, experiments 01-08), ChatGLM-6B (prefix LM, experiment 09)
- **Dataset:** MS MARCO v1.1 (passages, queries, answers)
- **Metric:** Mean negative log-likelihood (NLL) of gold answer — lower is better
- **Seed:** 42

---

## Experiment 01: Initial 15-Condition Test (`01_directed_kvcache_experiment.ipynb`)

**Date:** Early in project
**Samples:** 200
**Hypothesis:** When you build a KV cache from `[surrogate][document]` and truncate the surrogate entries, the document keys have wrong RoPE positions (position S+i instead of i). Applying RoPE(-S) correction should fix this and preserve the surrogate's influence on document representations.

**Conditions:** 15 across 5 groups (baselines, full-context surrogates, truncated caches, suffix placement, random prefix controls).

**Key Results:**
- RoPE correction mechanically worked: broken truncation (NLL ~2.58) improved to corrected (NLL ~1.24)
- But corrected truncation only returned to baseline — no improvement beyond it
- Full-context surrogates (surrogate kept visible) significantly improved over baseline (~71% win rate)
- Random prefix in full context ALSO helped (~77% win rate) — first hint that the benefit might be positional, not semantic

**Decision:** Full-context surrogates clearly help. Scale up and add routing to determine if the benefit is semantic.

---

## Experiment 02: Production Simulation at Scale (`02_production_simulation_experiment.ipynb`)

**Samples:** 2,500
**Hypothesis:** Generated surrogates routed by cosine similarity will outperform static surrogates and provide meaningful improvement over baseline.

**Key Results:**
- Generated Routed: **69.2% win rate**, mean improvement 0.2349 NLL, Cohen's d=0.39
- Static Routed: **66.7% win rate**, mean improvement 0.1972 NLL, Cohen's d=0.37
- Generated significantly better than static (p<0.001), but only by 0.0377 NLL
- Oracle (hindsight best-of-5): 84.4% win rate — room for better routing

**Decision:** Full-context surrogates consistently help. But surrogates hurt when baseline is already good. Need to diagnose failure cases.

---

## Experiment 03: Diagnostic Deep Dive (`03_production_simulation_diagnostic.ipynb`)

**Hypothesis:** Competing query signal, template framing, or attention dilution explains why surrogates sometimes hurt.

**Key Findings:**
- When baseline NLL is low (easy samples), surrogates make things worse
- When baseline NLL > 3.0 (hard samples), surrogates help
- Even "perfect surrogate" (actual query) doesn't consistently beat baseline
- Truncating surrogate after building cache removes the benefit entirely

**Decision:** Need to test truncated+corrected caches at scale with proper RoPE correction, and determine if full-context benefit is semantic.

---

## Experiment 04: Corrected Routing (`04_directed_kvcache_corrected_routing.ipynb`)

**Samples:** 921
**Hypothesis:** Truncated+corrected caches should work now that we have proper RoPE correction. Generated surrogates should outperform random prefixes (semantic > positional).

**Key Results:**
- **Truncated+corrected caches do NOT beat baseline** (gen routed: 42.6% win, p=0.70)
- **Full-context surrogates still help** (gen routed: 73.1% win, p<0.0001)
- **Random prefix ALSO helps** (80.1% win rate!)
- Full-ctx generated vs random prefix: p=0.03 — statistically significant but tiny effect
- **"Document:\n" framing hurts performance** (d=-0.45) — first discovery of this artifact

**Critical question:** Why don't truncated caches work? Is the RoPE correction code buggy?

**Decision:** Audit the RoPE correction code carefully.

---

## Experiment 05: Bug Fix Rerun (`05_directed_kvcache_bugfix_rerun.ipynb`)

**Samples:** 91 (incomplete run)

### Three Critical Bugs Discovered

**Bug 1 — Wrong RoPE dimension pairing (CRITICAL):**
`correct_rope_positions()` split keys into interleaved even/odd pairs (`keys[..., 0::2]`, `keys[..., 1::2]`). But Mistral's HuggingFace implementation uses half-split pairing: `x1 = x[..., :d/2]`, `x2 = x[..., d/2:]`. The inverse rotation was applied to the wrong dimension pairs, **scrambling keys instead of correcting them**. This is why truncated caches never worked in experiments 01-04.

**Bug 2 — Missing BOS token:**
`build_truncated_kv_cache_corrected` computed `doc_len` without the BOS token. The truncated cache started with `[Document, :, \n, ...]` instead of `[<s>, Document, :, \n, ...]`. Since baseline caches always start with BOS, this mismatch caused systematic differences unrelated to surrogate content.

**Bug 3 — Tokenizer boundary mismatch:**
`doc_len` was computed by tokenizing the document text in isolation, but BPE tokenization produces different tokens at join boundaries (`_Document` vs `Document`). Fixed by computing `doc_len = len(full_tokens) - len(prefix_tokens)`.

### Results After Bug Fixes
- **Truncated+corrected NOW works!** Generated routed: **83.5% win rate**, Cohen's d=0.66, p<0.0001
- Perfect surrogate (truncated): **78.0% win rate**, Cohen's d=0.53, p<0.0001
- Full-context random prefix still at 80.2% win rate
- Full-ctx gen vs random: p=0.45 — no significant difference

**Lesson learned:** The RoPE dimension pairing bug was devastating. Three experiments (01, 03, 04) produced misleading results because of it. Always verify mathematical operations against the actual model implementation, not the paper's notation.

**Decision:** Truncation works mechanically. Need rigorous semantic vs positional test.

---

## Experiment 06: Semantic Priming Hypothesis Test (`06_semantic_priming_hypothesis_test.ipynb`)

**Samples:** 677
**Purpose:** Definitive test of whether truncated cache benefit is semantic (content of surrogate matters) or structural (any prefix works).

**16 Conditions:** Bare, framed, generated/oracle/perfect/irrelevant/shuffled/random-passage/random-tokens (all truncated), full-context controls, 4 routing strategies.

**Key Results:**
| Hypothesis | Result | p-value |
|-----------|--------|---------|
| H1: Surrogate priming helps vs bare | NOT SUPPORTED | 0.62 |
| H2: Semantic content matters (gen vs irrelevant) | SUPPORTED | 0.0004 |
| H3: Word order matters (gen vs shuffled) | NOT SUPPORTED | 0.19 |
| H5: Coherence matters (gen vs random tokens) | SUPPORTED | 0.000001 |
| H6: Full-ctx benefit is semantic | NOT SUPPORTED | 0.03 (wrong direction) |

**Critical finding:** Correlation between generated and shuffled prefix deltas: **r=0.924**. Effects are overwhelmingly content-independent. The forward pass through any coherent text contaminates the value vectors in the same way.

**"Document:\n" framing confirmed harmful** (d=-0.45).

**Decision:** Prefix contamination is fundamental — the forward pass through the surrogate changes values, not just keys. Try suffix placement where document KV entries are provably unmodified.

---

## Experiment 07: Suffix Priming (`07_suffix_priming_experiment.ipynb`)

**Samples:** 200
**Hypothesis:** Place surrogate AFTER document. In causal attention, document tokens cannot attend to suffix tokens, so document KV entries are byte-identical to bare cache. Any improvement must come from query tokens attending to suffix tokens that have "read" the full document.

**Sanity check confirmed:** Document KV entries with suffix are byte-identical to bare cache. ✓

**18 Conditions:** Suffix variants (gen routed, oracle, perfect, irrelevant, shuffled, random passage, random tokens), format variations (raw, newline, multi-query), prefix comparison, summary.

**Key Results:**
- Content-independent effects persisted — relevant, irrelevant, and shuffled suffixes performed similarly
- Win rates ~50% against baseline (no improvement)
- High correlation between gen_routed and shuffled deltas (mirrors prefix r=0.924)

**Decision:** Suffix approach doesn't work either. Need to understand WHY. Three hypotheses: (1) query makes suffix redundant, (2) model ignores suffix, (3) MS MARCO is too easy.

---

## Experiment 08: Diagnostic — Is There Any Signal? (`08_diagnostic_suffix_signal.ipynb`)

**Samples:** 200 (Investigation A), 30 (Investigation B)

### Investigation A: Query-Free Scoring
Remove the query entirely. Score with `[passage + suffix] + "\n\nAnswer:"`. Suffix is the ONLY intent signal.

**Results:**
- All generated suffixes **hurt** performance (NLL increases 0.46-0.62, p<0.001)
- Win rates only 25-30%
- Relevant vs irrelevant: **not significant** (p=0.226)
- Relevant vs shuffled: **not significant** (p=0.135)
- Only "perfect" suffix (actual query) avoids harm
- Content-independence: r=0.797

### Investigation B: Attention Analysis
- Suffixes receive 14-16% of total attention (model isn't ignoring them)
- Relevant suffixes get +2.5% more attention than irrelevant (p<0.001)
- **Query attention drops from 20% to 9-10%** when suffix present — suffix STEALS attention from query
- Early layers (2-5) attend most to suffix; late layers (28-31) barely attend

**Verdict:** Suffix priming does NOT work for causal LLMs. The fundamental problem: **causal attention prevents backward information flow from suffix into passage representations.** The suffix is noise that competes with the query for attention budget.

**Decision:** The architecture is the bottleneck. Need a prefix LM with bidirectional attention on the prefix region, where passage tokens CAN attend to suffix tokens.

---

## Experiment 09: Prefix LM Test with ChatGLM-6B (`09_prefix_lm_experiment.ipynb`)

**Samples:** 200 planned (20 conditions)
**Model:** ChatGLM-6B — the only widely-available decoder-only prefix LM (bidirectional attention on prefix, causal on generation).

### Why ChatGLM-6B?
- ChatGLM-2 and ChatGLM-3 switched to fully causal attention (unsuitable)
- UL2/FLAN-UL2 are encoder-decoder (different KV cache structure)
- ChatGLM-6B is essentially the last available decoder-only prefix LM

### Compatibility Issues with Transformers 5.0.0
ChatGLM-6B was built for transformers ~4.27. Four patches required:

1. **`all_tied_weights_keys` missing:** transformers 5.0.0 calls `model.all_tied_weights_keys.keys()` during weight loading. Added `all_tied_weights_keys = {}` to `ChatGLMPreTrainedModel`.

2. **`sp_tokenizer` init ordering:** transformers 5.0.0's `PythonBackend.__init__` calls `get_vocab()` → `vocab_size` before the tokenizer subclass finishes init. Moved `self.sp_tokenizer = SPTokenizer(...)` before `super().__init__()`.

3. **`_pad()` missing kwargs:** transformers 5.0.0 passes `padding_side` kwarg to `_pad()`. Added `**kwargs` to signature.

4. **`_extract_past_from_model_output` removed:** This method was on `GenerationMixin` in transformers 4.x but removed in 5.0.0. ChatGLM's `_update_model_kwargs_for_generation` calls it. Added the method back to the model class.

### Local Model Copy
To prevent HF Hub re-downloads from overwriting patches, model files are stored locally at `models/chatglm-6b/` and loaded with `local_files_only=True`. All HF cache copies were removed.

### Architecture Differences from Mistral
- KV cache: tuple of tuples, shape `[seq_len, batch, num_heads, head_dim]` (seq-first vs Mistral's batch-first)
- 2D RoPE: first 64 dims = absolute position, second 64 dims = block position
- Inverted attention mask (True = masked)
- Custom tokenizer with `[gMASK]` + BOS

### Validation Confirmed
- Bidirectional attention works: passage KV entries DIFFER when suffix present (unlike Mistral where they were byte-identical) ✓
- 2D RoPE correction round-trips correctly (max error 3.91e-03) ✓
- Model generates coherent text ✓

### 20 Conditions
- **Group A (2):** Baselines (bare, bare padded)
- **Group B (6):** Suffix priming — THE MAIN TEST (gen routed, perfect, irrelevant, shuffled, random tokens, summary)
- **Group C (3):** Full prefix priming (gen routed, perfect, irrelevant)
- **Group D (3):** Truncated prefix + 2D RoPE correction
- **Group E (2):** Format sensitivity (template separator, raw concatenation)
- **Group F (4):** Query-free scoring (direct comparison to Exp 08)

**Status:** In progress.

---

## Cross-Experiment Reference Table

| Exp | Samples | Key Question | Key Finding |
|-----|---------|-------------|-------------|
| 01 | 200 | Does RoPE correction work? | Mechanically yes, but no improvement over baseline |
| 02 | 2500 | Do full-context surrogates help at scale? | Yes, 69% win rate |
| 03 | varies | Why do surrogates sometimes hurt? | Easy samples get worse; hard samples improve |
| 04 | 921 | Truncated vs full-context? | Truncated fails; random prefix = semantic prefix |
| 05 | 91 | Were there bugs? | YES — 3 critical bugs. Truncated now works (83.5% win) |
| 06 | 677 | Semantic or positional? | r=0.924 content-independence. Mostly positional |
| 07 | 200 | Does suffix placement help? | No. ~50% win rate |
| 08 | 200 | Any signal at all in suffix? | No. Suffix steals query attention. Causal mask is blocker |
| 09 | 200* | Does bidirectional attention fix it? | In progress |

## Key Bugs and Fixes

| Bug | Found in | Impact | Fix |
|-----|----------|--------|-----|
| Wrong RoPE dim pairing | Exp 05 | Exps 01-04 truncation results invalid | Use half-split `[:d/2]`, `[d/2:]` matching HF `rotate_half` |
| Missing BOS in truncated cache | Exp 05 | Systematic baseline mismatch | `extract_and_truncate_cache_with_bos` preserves BOS |
| BPE boundary mismatch | Exp 05 | Wrong doc_len computation | Compute `doc_len = len(full) - len(prefix)` |
| "Document:\n" framing hurts | Exp 04 | Baseline artificially degraded | Use bare passage, no framing text |

## Key Surprises

1. **Random prefixes help as much as semantic ones** (Exp 04, 06) — the benefit of full-context surrogates is predominantly structural/positional, not semantic.

2. **The RoPE bug was devastating** (Exp 05) — three experiments produced misleading null results because keys were scrambled instead of corrected. Mathematical verification against the actual code (not paper notation) is essential.

3. **Suffixes steal attention from queries** (Exp 08) — even though the model attends to suffixes (14-16% of attention), this comes at the cost of query attention (20% → 9-10%), causing net harm.

4. **Framing text matters more than surrogate content** (Exp 04, 06) — "Document:\n" causes d=-0.45 degradation. The structural aspects of the prompt template have outsized impact.

5. **Causal attention is the fundamental architectural blocker** (Exp 07, 08) — in causal LMs, passage tokens cannot attend to suffix tokens. The suffix is invisible to passage representations regardless of content.
