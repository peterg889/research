# Directed KV Cache v4 — Experiment Plan

## Critical v3→v4 Difference

In v3, the decoder received **only answer tokens**: `[BOS] + answer_ids`. The decoder
had to read the query entirely from encoder cross-attention output. This made the
co-encoding benefit look enormous (d=+0.408) but didn't model any real system — in
production, the query arrives at inference time and the decoder always has it.

In v4, the decoder receives **query + answer**: `[BOS] + query_ids + answer_ids`.
NLL is computed only on answer positions. This models the real production pipeline:

1. **Offline**: Encode [surrogate + document] → cache encoder hidden states
2. **Online**: Query arrives → decoder input = [BOS, query, answer], cross-attends to cache

**Every experiment in v4 must pass `decoder_input_ids` explicitly** with the query
as prefix. The v3 scoring function (`score_nll`) is only used for `_nq` replication
conditions.

### What changed mechanistically

The v3→v4 shift fundamentally altered the mechanism landscape:

| Property | v3 (no query in decoder) | v4 (query in decoder) |
|----------|--------------------------|----------------------|
| Oracle enrichment d | +0.408 | +0.228 (61% preserved) |
| Structural fraction (random/oracle) | 81-85% | ~35% (0.080/0.228) |
| Template surrogate | d=+0.336 (90% oracle) | d=-0.069 (HURTS) |
| Doc-keyword surrogate | d=+0.363 (89% oracle) | d=+0.148 (65% oracle) |
| Random prefix | d=+0.303 (81% oracle) | d=+0.080 (ns, p=0.07) |

**Key insight**: When the decoder already has the query, the structural "binary switch"
mechanism is largely neutralized. The decoder's own query representation provides
most of the attention redistribution benefit that the encoder prefix gave in v3.
What remains is the **content-dependent** component — the encoder prefix must contain
information that genuinely enriches document representations beyond what the decoder
query provides through cross-attention.

This is why:
- **Doc keywords work**: They give the encoder semantic signal about document content
  that bidirectional attention can use to reorganize document representations.
- **Template fails**: "What is [keyword]?" is structurally generic — in v3, its
  structure alone was enough; in v4, it needs content, and one keyword isn't enough.
- **Random fails**: Pure structural noise no longer triggers the benefit when the
  decoder already provides the structural component via its own query.

## Design Principles (inherited from v3, updated)

1. **Always report the three-point spectrum**: oracle (upper bound), bare (lower bound),
   and surrogate(s) in between. Every table shows where we are.

2. **Always truncate**: Exp 01 confirmed truncation is strictly better in v4 too
   (oracle_full d=0.167 < oracle_trunc d=0.228). Mask prefix tokens from cross-attention.

3. **Statistical rigor**: Cohen's d, win%, AND p-value. Bonferroni for multiple comparisons.
   **N=500 minimum**. Checkpoint every 20 samples. 40 GB GPU can handle it.

4. **Content controls**: Always include random prefix (structural control) AND bare
   (no-prefix baseline). In v4, also always include `_nq` replication pair for calibration.

5. **Query-in-decoder is mandatory**: All primary conditions use `decoder_input_ids =
   [BOS] + query_ids + answer_ids`. The `_nq` conditions are replication controls only.

6. **Same evaluation framework**: Mean NLL over answer tokens. Paired comparisons.

7. **Document length awareness**: Stratify or control for document length.

---

## Experiment Log

### Exp 01: Production-Realistic KV Cache (COMPLETE)

**Question**: Does surrogate-enriched encoder caching still help when the decoder
already has the query?

**Answer**: YES, but the benefit is reduced and the mechanism shifts.

**Conditions** (8 total):

| # | Condition | Enc prefix | Trunc | Dec query | Mean NLL | d vs bare | p |
|---|-----------|-----------|-------|-----------|----------|-----------|---|
| 1 | bare | (none) | — | yes | 2.554 | — | — |
| 2 | oracle_trunc | real query | yes | yes | 2.406 | +0.228 | 5.1e-7 *** |
| 3 | oracle_full | real query | no | yes | 2.260 | +0.167 | 2.2e-4 *** |
| 4 | surr_template_trunc | "What is [kw]?" | yes | yes | 2.593 | -0.069 | 0.12 ns |
| 5 | surr_doc_trunc | top-5 doc kw | yes | yes | 2.447 | +0.148 | 1.0e-3 ** |
| 6 | random_trunc | unrelated words | yes | yes | 2.490 | +0.080 | 0.07 ns |
| 7 | bare_nq | (none) | — | no | 3.676 | — | — |
| 8 | oracle_trunc_nq | real query | yes | no | 2.993 | +0.376 | 4.8e-16 *** |

**Key results**:
- Oracle enrichment with query in decoder: d=+0.228 (***), 61% of v3 baseline preserved
- v3 replication (no query): d=+0.376 (***) — matches v3 Exp 01
- Truncation still better: oracle_trunc d=+0.228 > oracle_full d=+0.167
- Doc-keyword surrogate: d=+0.148 (**), 65% of oracle — **works in production**
- Template surrogate: d=-0.069 (ns) — fails (was 90% of oracle in v3!)
- Random prefix: d=+0.080 (ns, p=0.07) — structural alone is marginal
- Per-sample correlation (with-q vs no-q enrichment): r=0.475
- Hardness gradient flattened vs v3

**Files**: `experiments/encoder_decoder/01/`, `results/exp01/`

### Exp 02: Length Scaling with Query in Decoder (COMPLETE)

**Question**: Does the v4 enrichment benefit (d=+0.228) decay with document length?

**Answer**: NO — enrichment **grows** with document length, not decays. All conditions
significant at ALL lengths after Bonferroni (24 comparisons).

**Design**: MS MARCO v1.1, pad documents to target lengths by appending unrelated
passages. 6 length bins × 6 conditions × N=500, SEED=42.

**Results** (d vs bare at each length bin):

| Length (tokens) | oracle_trunc | surr_doc_trunc | random_trunc | oracle_nq (v3 repl.) |
|-----------------|-------------|----------------|--------------|---------------------|
| ~98 (original)  | +0.238 ***  | +0.165 ***     | -0.028 ns    | +0.427 ***          |
| 256             | +0.352 ***  | +0.364 ***     | +0.254 ***   | +0.519 ***          |
| 512             | +0.323 ***  | +0.330 ***     | +0.271 ***   | +0.517 ***          |
| 1024            | +0.439 ***  | +0.399 ***     | +0.357 ***   | +0.448 ***          |
| 2048            | +0.392 ***  | +0.396 ***     | +0.284 ***   | +0.383 ***          |
| 4096            | +0.426 ***  | +0.396 ***     | +0.221 ***   | +0.342 ***          |

**Key findings**:
- Oracle enrichment GROWS from d=+0.238 (98 tok) to d=+0.426 (4096 tok)
- Random prefix becomes highly significant at 256+ tokens (structural mechanism regains
  importance with longer documents, even when decoder has query)
- At 1024+ tokens, surr_doc ≈ oracle — the gap between content-dependent and structural
  enrichment closes as documents get longer
- v3 replication (oracle_nq) decays slightly from d=+0.427 to d=+0.342
- The v4/v3 ratio INCREASES with length: from 56% (original) to 125% (4096)

**Interpretation**: On short MS MARCO docs (~98 tokens), the decoder's own query
representation provides most of the structural redistribution, leaving only the
content-dependent residual (35% structural). But on longer documents, the encoder
prefix's structural benefit becomes important again — the decoder's query alone
isn't sufficient to reorganize attention across 4096 encoder tokens. This was
tested with **padded** documents (unrelated passages appended), not naturally long
documents — see Exp 03 for ecologically valid confirmation.

**Files**: `experiments/encoder_decoder/02/`, `results/exp02/`

### Exp 03: Neural-Bridge Cross-Dataset Validation (COMPLETE)

**Question**: Does the enrichment pattern hold on naturally long documents (neural-bridge
RAG dataset, ~604 words) without artificial padding?

**Answer**: YES, and surrogates beat oracle — the v3 "semantic interference" phenomenon
persists in v4.

**Design**: neural-bridge/rag-dataset-12000. Filter: query >= 15 words, answer >= 5 words.
Mean query 17.8w, doc 604w, answer 42.9w. 6 conditions × N=500, SEED=42.

**Results**:

| # | Condition | Mean NLL | d vs bare | win% | p |
|---|-----------|----------|-----------|------|---|
| 1 | bare | 0.675 | — | — | — |
| 2 | oracle_trunc | 0.654 | +0.306 *** | 64.6% | 2.1e-11 |
| 3 | surr_doc_trunc | 0.652 | +0.502 *** | 71.4% | 2.9e-26 |
| 4 | random_trunc | 0.644 | +0.624 *** | 75.6% | 1.5e-37 |
| 5 | bare_nq | 1.314 | — | — | — |
| 6 | oracle_trunc_nq | 1.224 | +0.592 *** | 85.0% | 1.9e-34 |

**Key findings**:
- Oracle enrichment with query in decoder: d=+0.306 (***) — stronger than Exp 01 MS MARCO (+0.228)
- **Surrogates beat oracle**: random d=+0.624 > surr_doc d=+0.502 > oracle d=+0.306
- Structural fraction: 204% — the real query creates semantic interference, same as v3
- v3 replication (oracle_nq): d=+0.592 (***), strong confirmation
- Confirms Exp 02 pattern: longer documents → stronger enrichment, structural mechanism dominant

**Interpretation**: On naturally long documents (~600 words), the structural mechanism
completely dominates even in v4 (structural fraction 204%). The real query in the
encoder prefix actually hurts relative to random/doc-keyword prefixes because it creates
semantic interference with the document's own content — the same phenomenon observed in
v3 Exp 3D on this dataset. The decoder already has the query, so putting it in the
encoder too is redundant AND harmful. A random prefix provides pure structural benefit
without the interference cost.

**Implications for production**: On long documents, the optimal encoder prefix is NOT
the query — it's any short arbitrary text. This is a simpler and cheaper solution than
query prediction or keyword extraction.

**Files**: `experiments/encoder_decoder/03/`, `results/exp03/`

### Exp 04: Prefix Content Optimization (COMPLETE)

**Question**: What prefix content maximizes enrichment on short documents where content
matters (structural fraction only 35%)?

**Answer**: kw10 (top-10 TF keywords) captures 82% of oracle — but document-specific
keywords don't matter, and natural text is catastrophic.

**Design**: MS MARCO v1.1, 8 content types + 2 v3 replication conditions, N=500, SEED=42.

**Results** (ranked by d vs bare):

| Rank | Condition | Prefix tokens | d vs bare | % oracle | sig |
|------|-----------|--------------|-----------|----------|-----|
| 1 | oracle_trunc | 7.5 | +0.228 | 100% | *** |
| 2 | surr_kw10 | 13.6 | +0.186 | 82% | *** |
| 3 | surr_kw5 | 7.3 | +0.148 | 65% | ** |
| 4 | surr_random_kw5 | 7.3 | +0.129 | 57% | * |
| 5 | surr_kw20 | 25.8 | +0.115 | 50% | ns |
| 6 | random_trunc | 9.1 | +0.080 | 35% | ns |
| 7 | surr_first_sent | 22.1 | **-0.298** | -131% | *** |

v3 replication: oracle_trunc_nq d=+0.376 (***). Exp 01 replication exact: oracle +0.228,
kw5 +0.148, random +0.080.

**Key findings**:

1. **Keyword density is inverted-U**: kw10 (82%) > kw5 (65%) > kw20 (50%). kw20 vs kw10
   is significantly worse (d=-0.099, *). Too many keywords (~26 tokens) creates deep
   attention dependencies during encoding that leave "truncation wounds" when masked.
   Sweet spot: 10-15 prefix tokens.

2. **First sentence is catastrophic** (d=-0.298 ***): Coherent natural text from the
   document creates the strongest bidirectional attention connections with matching
   document content. Masking these after encoding leaves representation gaps far worse
   than bare encoding. This is the most practically important finding — do NOT use
   natural document text as a prefix.

3. **Document specificity is irrelevant**: kw5 (this doc) vs random_kw5 (wrong doc):
   d=-0.024 (ns). Decomposition of the kw5-vs-random gap:
   - Vocabulary component (random→random_kw5): +0.049 (72%)
   - Semantic component (random_kw5→kw5): +0.019 (28%)
   Having keyword-*shaped* tokens matters; having the RIGHT keywords barely matters.

4. **Length penalty**: Across conditions, longer prefixes tend worse (r=-0.546). The
   mechanism is not just "more tokens to mask" but the nature of longer prefixes —
   more redundant words, more attention dependencies, more truncation damage.

5. **Oracle's remaining advantage**: The real query is short (~7.5 tokens) and doesn't
   overlap document content (it's a question *about* the document). It gets structural
   benefit without creating destructive attention connections, plus genuine semantic
   signal (question intent, answer framing) that no surrogate captures.

**Mechanistic interpretation**: Truncation creates two opposing forces:
- **Beneficial**: Prefix tokens perturb the bidirectional attention landscape, causing
  document tokens to redistribute attention beneficially (the v3 mechanism)
- **Destructive**: When prefix-document attention connections are too strong (coherent
  overlapping text), masking leaves "wounds" in the document representations

The ideal prefix is: short (7-15 tokens), disconnected (keyword bags, not sentences),
and content-bearing (any keywords, not necessarily document-specific). The content
headroom closed by kw10: 72% of the random→oracle gap.

**Files**: `experiments/encoder_decoder/04/`, `results/exp04/`

### Exp 05: Truncation Wound Mechanism (COMPLETE)

**Question**: WHY does coherent overlapping text (first sentence) cause catastrophic
damage (d=-0.298) when other prefix types are beneficial?

**Answer**: The wound requires BOTH coherence AND overlap — neither alone is sufficient.
It's an interaction effect, not a main effect.

**Design**: 2×2 factorial crossing Coherence (intact sentence vs shuffled/keywords) ×
Overlap (same-doc content vs wrong-doc/generic). MS MARCO v1.1, N=500, SEED=42.

**Results**:

| # | Condition | Coherent? | Overlaps? | d vs bare | sig |
|---|-----------|-----------|-----------|-----------|-----|
| 1 | bare | — | — | — | — |
| 2 | oracle_trunc | — | — | +0.228 | *** |
| 3 | surr_kw10 | No | Yes (partial) | +0.186 | *** |
| 4 | surr_first_sent | **Yes** | **Yes** | **-0.298** | *** |
| 5 | surr_wrong_first_sent | Yes | No | +0.063 | ns |
| 6 | surr_shuffled_sent | No | Yes | +0.078 | ns |
| 7 | surr_generic_sent | Yes | No (generic) | -0.062 | ns |
| 8 | surr_wrong_kw10 | No | No | +0.131 | ** |

**Hypothesis tests** (2×2 factorial decomposition):
- Overlap main effect: d = -0.207 (overlap makes things worse ON AVERAGE)
- Coherence main effect: d = -0.222 (coherence makes things worse ON AVERAGE)
- **Interaction (overlap × coherence)**: d = -0.361 (the COMBINATION is catastrophic)
- Neither factor alone causes damage: wrong-doc coherent (+0.063), shuffled same-doc (+0.078)

**Mechanistic interpretation**: During bidirectional encoding, a coherent sentence that
shares vocabulary with the document forms deep cross-attention connections — the syntax
and word co-occurrences create strong bidirectional bonds between prefix and document
tokens. When the prefix is then masked from decoder cross-attention (truncation), the
document representations retain "dangling references" to tokens the decoder can never
access. The result is worse than bare encoding because the document representations
were actively reshaped around now-invisible information.

Keywords (even overlapping ones) don't trigger this because they lack syntactic structure
— isolated words create weaker, more superficial attention connections that leave only
minor artifacts when masked. Shuffled sentences have the same words but broken syntax,
so they also don't form the deep bonds. Wrong-doc sentences have coherent syntax but
share no vocabulary with the document, so there's nothing to form strong bonds with.

**Length within first_sent predicts damage**: r=0.230, p=2.1e-7. Longer coherent
overlapping prefixes form more connections, causing more wound.

**Practical implication**: The "no natural text" rule from Exp 04 is now mechanistically
explained. It's not natural text per se that's dangerous — it's specifically natural text
that OVERLAPS document content. A coherent sentence about an unrelated topic is harmless
(d=+0.063). But since surrogates are typically derived from the target document, this
means: NEVER use extractive surrogates (summaries, first sentences, key phrases in
natural language). Keyword bags are safe.

**Files**: `experiments/encoder_decoder/05/`, `results/exp05/`

### Exp 06: Factoid vs Long-Answer Split (COMPLETE)

**Question**: Does the v4 enrichment differ between factoid (≤5 word answers) and
long-answer (>5 word) samples?

**Answer**: YES — and the pattern REVERSES from v3: long answers show STRONGER
enrichment in v4 (opposite of v3).

**Design**: Pure reanalysis of Exp 01 checkpoint data (no GPU needed). Binary split
(factoid ≤5w vs long >5w) plus 5-bin breakdown.

**Results** (subgroup analysis):

| Subgroup | N | d_oracle | d_surr (kw5) | d_random | struct_frac | surr % oracle |
|----------|---|----------|-------------|----------|-------------|--------------|
| ALL | 500 | +0.228 *** | +0.148 ** | +0.080 ns | 35% | 65% |
| Factoid (≤5w) | 210 | +0.284 *** | +0.194 ** | +0.099 ns | 35% | 68% |
| Long (>5w) | 290 | +0.412 *** | +0.259 *** | +0.209 *** | 51% | 63% |

**Answer-length bins**:

| Bin | N | d_oracle | d_surr | struct_frac | surr % oracle |
|-----|---|----------|--------|-------------|--------------|
| 1-2w | 130 | +0.299 *** | +0.197 * | 33% | 66% |
| 3-5w | 80 | +0.385 *** | +0.391 *** | 78% | **101%** |
| 6-10w | 63 | +0.452 *** | +0.292 * | 48% | 65% |
| 11-20w | 100 | +0.548 *** | +0.326 ** | 44% | 60% |
| 21+w | 127 | +0.341 *** | +0.210 * | 74% | 62% |

**Key findings**:

1. **v3→v4 reversal**: In v3, factoid answers had 2× the oracle headroom (d=0.767 vs
   0.376). In v4, LONG answers have stronger enrichment (d=+0.412 vs +0.284). When the
   decoder has the query, factoid QA benefits less because the decoder's query already
   provides most of the semantic signal needed for short answers.

2. **3-5 word sweet spot**: Surrogate BEATS oracle (101% of oracle d). These answers
   are short enough to benefit from structural enrichment but long enough that content
   helps. The structural fraction (78%) is the highest of any bin.

3. **11-20 word peak**: Strongest oracle enrichment (d=+0.548). These longer answers
   require the most from encoder representations — the decoder needs rich cross-attention
   targets to generate multi-sentence answers correctly.

4. **Structural fraction increases with answer length**: From 33% (1-2w) to 74% (21+w).
   Longer answers rely more on structural enrichment, consistent with the v4 mechanism
   shift — content matters most for short answers, structure for long ones.

5. **v4/v3 ratio**: Around 54-60% across subgroups (v4 preserves ~55% of v3 effect).
   The 3-5w bin drops to 38% — these answers had the most to lose from the structural
   collapse in v4.

**Files**: `experiments/encoder_decoder/06/`, `results/exp06/`

### Exp 07: Decoder Attention Probing (COMPLETE)

**Question**: Do the decoder's query tokens act as attention buffers (same mechanism
as encoder prefix), and does this explain the v3→v4 structural collapse (85%→35%)?

**Answer**: PARTIALLY CONFIRMED. Query tokens absorb 5.5% of answer-token attention
(modest buffer). The two mechanisms are **35% redundant** — significant interaction
(d=+0.316, p<1e-11). The main finding is that oracle encoding shifts budget from
cross-attention (37%→13%) to self-attention (63%→87%).

**Design**: 2×2 factorial {encoder: bare, oracle_trunc} × {decoder: nq, q}.
Attention probes from 6 decoder layers [0, 5, 11, 17, 22, 33].
N=500, MS MARCO v1.1, SEED=42.

**NLL Results** (2×2):

| | No query (nq) | With query (q) |
|---|---|---|
| Bare encoder | 3.683 | 2.554 |
| Oracle encoder | 2.997 | 2.397 |

| Effect | d | sig |
|--------|---|-----|
| Encoder prefix (w/o query) | +0.366 | *** |
| Encoder prefix (w/ query) | +0.238 | *** |
| Decoder query (w/o prefix) | +0.309 | *** |
| Decoder query (w/ prefix) | +0.228 | *** |
| **Interaction** | **+0.316** | *** |

Encoder prefix benefit reduced by **35%** when decoder has query.
Decoder query benefit reduced by **26%** when encoder has prefix.

**Attention Budget (layer 33, answer tokens)**:

| Condition | BOS sink | Query buffer | Self-answer | Self total | Cross total |
|-----------|----------|-------------|-------------|------------|-------------|
| bare_nq | 13.0% | 0% | 50.0% | 63.0% | 37.0% |
| bare_q | 10.9% | 5.5% | 47.9% | 64.2% | 35.8% |
| oracle_trunc_nq | 21.2% | 0% | 65.9% | 87.1% | 12.9% |
| oracle_trunc_q | 17.3% | 6.1% | 65.9% | 89.3% | 10.7% |

**Key findings**:

1. **Query buffer is modest (5.5%)**: Not a massive attention sink like the encoder BOS.
   The query tokens absorb some attention from answer tokens, but this is a minor
   channel compared to the cross-attention shift.

2. **Oracle encoding massively shifts budget**: The enriched encoder reduces cross-attention
   from 37% to 13% and increases self-attention from 63% to 87%. The enriched encoder
   representations are so good that the decoder doesn't need to cross-attend as much.

3. **BOS sink grows with oracle**: 13%→21%. The enriched encoder makes decoder self-attention
   more useful, and BOS absorbs more of the self-attention budget.

4. **Query mostly steals from cross-attention**: The query buffer's 5.5% comes primarily
   from cross-attention, not from self-answer attention (which barely changes: 50.0%→47.9%).

5. **Self-attention entropy increases with query**: +0.186 nats (more uniform distribution
   when query provides additional context). Cross-attention entropy decreases: -0.113 nats
   (more focused on relevant encoder positions).

6. **Interaction is sub-additive**: Both mechanisms operate on the same pathway
   (cross-attention → self-attention rebalancing). Having one partially satisfies the
   need the other addresses, creating the 35% redundancy.

**Mechanistic interpretation**: The v3→v4 structural collapse is NOT primarily due to
the query acting as an attention buffer in the decoder (5.5% is too small). Instead,
the decoder's query provides direct semantic signal through its representation, which
reduces the MODEL'S NEED for cross-attention to the encoder. Since encoder prefix
enrichment works by improving the encoder representations that cross-attention reads,
any reduction in cross-attention reliance reduces the prefix's impact. The 35%
redundancy reflects the overlap between "making encoder reps better" (prefix) and
"needing encoder reps less" (decoder query).

**Files**: `experiments/encoder_decoder/07/`, `results/exp07/`

### Exp 08: Decoder Length Control (COMPLETE)

**Question**: Is the Exp 07 "35% redundancy" between encoder prefix and decoder query
actually a decoder sequence LENGTH artifact? (The _q conditions have longer decoder
sequences than _nq, which could affect attention independently of semantic content.)

**Answer**: YES — **the redundancy is entirely a length artifact**.

**Design**: 2×3 factorial {encoder: bare, oracle_trunc} × {decoder: nq, random_q, q}.
`random_q` uses random token IDs matching the query length per sample. 3 probe layers
[0, 17, 33]. MS MARCO v1.1, N=500, SEED=42.

**NLL Results** (Mean NLL, 500/500 final):

| | No query (nq) | Random Q (rand_q) | Real query (q) |
|---|---|---|---|
| Bare encoder | 3.683 | 3.114 | 2.554 |
| Oracle encoder | 2.997 | 3.026 | 2.397 |

**Decomposition of nq → q improvement**:

| | Bare encoder | Oracle encoder |
|---|---|---|
| Length effect (rand_q − nq) | d=+0.170 (55%) | d=-0.014 ns (0%) |
| Semantic effect (q − rand_q) | d=+0.317 (45%) | d=+0.324 (100%) |
| Total (q − nq) | d=+0.309 *** | d=+0.228 *** |

**Interaction decomposition**:
- Full enc × q interaction: d=+0.316, ***
- Length component: d=+0.324 (102% of total interaction)
- Semantic component: d=-0.097 (slightly super-additive, ns)

**Attention**:
- Random decoder tokens absorb 14.3% of answer attention (vs 5.5% for real query, 2.6×)

**Key findings**:

1. **The 35% redundancy from Exp 07 is entirely a length artifact.** Both encoder
   co-encoding and decoder prefix tokens act through the same mechanism: attention
   redistribution. Once random tokens provide the structural escape, the encoder has
   nothing additional to contribute structurally.

2. **With oracle encoder, adding random decoder tokens provides ZERO length benefit**
   (d=-0.014, ns). The oracle encoder already captures all structural benefit. The only
   remaining decoder benefit is semantic — having the actual query content.

3. **For bare encoders, the nq→q improvement is ~55% length + 45% semantic.** The random
   tokens provide significant structural benefit (d=+0.170).

4. **Random tokens steal more attention than real queries** (14.3% vs 5.5% at layer 33).
   Random tokens cause maximal attention redistribution because the decoder can't focus
   on them meaningfully.

5. **Encoder and decoder are semantically INDEPENDENT, not redundant.** The semantic
   interaction is d=-0.097 (slightly super-additive, ns). The encoder enriches document
   representations with query content; the decoder has query tokens for generation.
   These are additive benefits. Only the STRUCTURAL benefit is redundant.

**Implication for Exp 07 reinterpretation**: The Exp 07 "35% redundancy" and "the decoder
query reduces the model's need for cross-attention" interpretation needs correction. The
redundancy is not about the query's semantic content reducing cross-attention reliance —
it's about the query's TOKENS providing the same structural benefit that the encoder
prefix provides. Once you control for this length effect, the encoder and decoder query
are NOT semantically redundant at all.

**Files**: `experiments/encoder_decoder/08/`, `results/exp08/`

### Exp 09: Cross-Model Generalization Sweep (COMPLETE)

**Question**: Does the enrichment effect generalize beyond T5Gemma 2 4B-4B?

**Answer**: YES — oracle enrichment is significant for ALL 4 models tested. But the
structural fraction varies wildly by architecture.

**Design**: Test 4 additional encoder-decoder models:
- google/flan-t5-base (250M)
- google/flan-t5-large (780M)
- google/flan-t5-xl (3B)
- facebook/bart-large (400M)

4 conditions: bare, oracle_trunc, random_trunc, bare_nq. Same 500 MS MARCO samples.
NLL only (no attention probes), SDPA attention for speed. Sequential model loading/unloading.

**Results**:

| Model | Params | d_oracle | d_random | Struct % | d_dec_q | oracle sig |
|-------|--------|----------|----------|----------|---------|------------|
| T5Gemma-2-4B (ref) | 8B | +0.228 | +0.080 ns | 35% | +0.309 | *** |
| flan-t5-base | 250M | +0.251 | +0.107 * | 43% | -0.438 | *** |
| flan-t5-large | 780M | +0.320 | +0.247 *** | 77% | -0.416 | *** |
| flan-t5-xl | 3B | +0.430 | +0.030 ns | 7% | -0.341 | *** |
| bart-large | 400M | +0.144 | +0.043 ns | 30% | +0.680 | ** |

**Key findings**:

1. **Enrichment generalizes universally.** All 4 models show significant oracle enrichment
   (p < 0.002 for all). The effect is not a T5Gemma quirk.

2. **Structural fraction varies wildly (7%–77%)**: flan-t5-xl is almost pure content
   (7% structural), flan-t5-large is dominantly structural (77%), bart-large and T5Gemma
   are moderate (30–35%). Architecture matters more than size for the mechanism mix.

3. **flan-t5-xl has the STRONGEST oracle effect** (d=+0.430, stronger than T5Gemma's
   +0.228) but near-zero structural component. This larger model can extract genuine
   content from the oracle prefix but gains nothing from random tokens.

4. **Flan-T5 models have NEGATIVE d_dec_q** — putting the query in the decoder HURTS
   answer prediction. These models were instruction-tuned with query-in-encoder /
   answer-in-decoder, so query-in-decoder goes against their training distribution.
   This means the v4 setup (query in both encoder and decoder) doesn't map cleanly
   onto flan-t5's intended usage.

5. **BART has the weakest enrichment** (d=+0.144) but the strongest decoder query effect
   (d=+0.680). BART's denoising pre-training may make it more dependent on decoder
   context and less sensitive to encoder perturbations.

6. **Model quality vs structural fraction**: Using bare NLL as a quality proxy, there's
   an inverted-U — structural fraction peaks at medium quality (flan-t5-large: 77%) and
   drops for both the weakest (bart: 30%) and strongest (flan-t5-xl: 7%) models. Better
   models may need real content to benefit, while weaker models can't effectively use
   even structural perturbations. However, the flan-t5 negative d_dec_q complicates
   direct comparison with T5Gemma, since these models handle decoder queries differently.

**Files**: `experiments/encoder_decoder/09/`, `results/exp09/`

---

## Deferred Experiments

### Ranking Revisited (LOW PRIORITY)

**Motivation**: v3 Exp 04A/04B showed ranking was a dead end because the structural
mechanism is document-independent. Exp 04 confirmed that even in v4, document
specificity doesn't matter (kw5 vs random_kw5: ns). Exp 05 showed the mechanism is
purely about attention restructuring, not information content. The oracle's remaining
18% edge over kw10 is genuinely query-specific, but unlikely to create differential
signal.

**Risk**: HIGH. Given Exp 04 (specificity irrelevant) and Exp 05 (mechanism is
structural), ranking signal is extremely unlikely.

**Status**: Deferred pending a stronger theoretical motivation.

---

## Experiment Priority

| Priority | Experiment | Status |
|----------|-----------|--------|
| 1 | Exp 08: Decoder Length Control | COMPLETE |
| 2 | Exp 09: Cross-Model Sweep | COMPLETE |
| — | Ranking Revisited | Deferred — high risk, no new evidence for signal |

---

## Summary of Research Arc

### v3 → v4 story so far

In v3, we discovered that prepending ANY short text to the encoder input improves
document representations for answer prediction (d=+0.408), and that 85% of this
benefit is structural — a single random word triggers attention redistribution that
escapes the bare-encoding manifold. But v3's decoder never had the query, which is
unrealistic. In v4, we gave the decoder the query (modeling real production), and
found that 61% of the enrichment survives (d=+0.228 on short MS MARCO docs). Crucially,
the **mechanism shifted**: the structural component collapsed from 85% to ~35% (the
decoder's own query representation already provides the redistribution), while the
content-dependent component became dominant (doc-keyword prefix: 65% of oracle vs
random: 35%).

However, this structural collapse was **specific to short documents**. Exp 02 (length
scaling) showed that enrichment GROWS with document length — oracle d rises from +0.238
(~98 tokens) to +0.439 (1024 tokens). At 256+ tokens, even the random prefix becomes
highly significant, meaning the structural mechanism regains importance when documents
are long enough that the decoder's query alone can't reorganize attention across all
encoder positions. Exp 03 (neural-bridge, naturally ~604-word documents) confirmed this
dramatically: structural fraction hit 204%, and surrogates beat the oracle.

Exp 04 (prefix content optimization) then dissected WHAT makes a good prefix on short
documents. The key insight: **truncation creates both benefit and damage**. A prefix
that perturbs the attention pattern with short, disconnected tokens (keywords) causes
beneficial reorganization. But a prefix that creates deep attention connections with
the document (coherent natural text, especially overlapping content) leaves
"truncation wounds" when masked — the first sentence of the document is catastrophic
(d=-0.298). Document-specific keywords don't outperform wrong-document keywords
(d=-0.024, ns), confirming the benefit is still largely structural (operating through
vocabulary-type features rather than semantic matching). The sweet spot is 10-15
keyword-like tokens (kw10 captures 82% of oracle).

**Bottom line**: The practical recipe is document-length-dependent:
- **Short documents (<256 tokens)**: Prepend 10 keyword-like tokens. Document-specific
  keywords (kw10) are best at 82% of oracle, but wrong-doc keywords also work (57%).
  NEVER use natural text — especially extractive text that overlaps document content
  (Exp 05: coherent + overlapping = catastrophic interaction). Closes 72% of the
  random→oracle content gap.
- **Long documents (500+ tokens)**: ANY short prefix works. The structural mechanism
  dominates, surrogates beat oracle (semantic interference). No optimization needed.
- The crossover is around 256 tokens. The oracle's remaining 18% edge over kw10 comes
  from genuine query-specific semantic signal that no surrogate can capture.

### v4-specific mechanism insights (Exp 05 + 06)

**Truncation wound** (Exp 05): The first-sentence catastrophe is an interaction effect
requiring BOTH coherence AND overlap. Coherent natural text forms deep bidirectional
attention bonds during encoding. If those tokens also share vocabulary with the document,
the bonds are especially strong. Masking after encoding leaves "dangling references" —
the document representations were reshaped around now-invisible information. Keywords
are safe because they form weaker, more superficial connections.

**Answer-length dependence** (Exp 06): v4 reverses v3's factoid advantage. Long answers
(>5w) now show STRONGER enrichment (d=+0.412 vs +0.284) because the decoder's query
already provides most of the semantic signal for short factoid answers. The structural
fraction increases from 33% (1-2w answers) to 74% (21+w answers). The 3-5 word bin is
the sweet spot where surrogates beat oracle (101% of oracle d).

**Decoder attention probing** (Exp 07): The v3→v4 structural collapse (85%→35%) is NOT
primarily because decoder query tokens act as attention buffers (they absorb only 5.5%
of answer attention). Instead, the decoder's query reduces the MODEL'S RELIANCE on
cross-attention to the encoder — oracle encoding shifts answer-token cross-attention
from 37% to 13%, redirecting to self-attention (63%→87%). Since encoder prefix enrichment
works by improving what cross-attention reads, anything that reduces cross-attention
reliance (like having query in the decoder) reduces the prefix's impact. The two
mechanisms are 35% redundant (interaction d=+0.316, ***), operating on the same
cross-attention → self-attention rebalancing pathway.

**CORRECTION (Exp 08)**: The Exp 07 "35% redundancy" is entirely a decoder sequence
LENGTH artifact. Adding random tokens (same length as query) to the decoder eliminates
85% of the encoder effect (d=-0.120 → d=-0.026). The length-based interaction accounts
for 110% of the total interaction. Semantically, the encoder and decoder are NOT
redundant — the semantic interaction is -0.050 (slightly super-additive, ns). The correct
interpretation: both encoder prefix and decoder prefix tokens provide structural attention
redistribution; this structural benefit is redundant (both do the same thing). But the
SEMANTIC benefit of the encoder (enriching document representations with query content)
and the SEMANTIC benefit of the decoder (having query tokens available during generation)
are independent and additive.
