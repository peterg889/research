# Directed KV Cache — Presentation Slides

## Part 1: Encoder-Decoder

---

### Slide: Two-Phase Process — Step-by-Step

#### Step 1: Encode prefix + document with bidirectional attention (offline)

```
Encoder input:
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Where were the 2024 Olympics held?   │ The 2024 Summer Olympics were held   │
│            (prefix)                  │ in Paris, France, from 26 July to    │
│                                      │ 11 August 2024...                    │
│                                      │           (document)                 │
└──────────────────────────────────────┴──────────────────────────────────────┘

                 Bidirectional self-attention:
                 ┌───────────────────────────────────────────┐
                 │  Every token attends to EVERY other token │
                 │                                           │
                 │  "Where" ↔ "The"  ↔ "Paris"  ↔ "held"    │
                 │  "2024"  ↔ "2024" ↔ "Olympics"↔ "Summer"  │
                 │                                           │
                 │  Prefix and document form deep            │
                 │  bidirectional attention bonds.            │
                 │  Document representations are shaped      │
                 │  by the prefix in BOTH directions.        │
                 └───────────────────────────────────────────┘
```

The encoder processes the prefix and document together with full bidirectional attention. Every token attends to every other token — the prefix reshapes document representations, and vice versa. This is the key step where the prefix's influence gets "baked in" to the document hidden states.

#### Step 2: Truncate — mask prefix from decoder cross-attention

```
Full encoder hidden states:
┌──────┬──────┬─────┬──────┬──────┬──────┬─────┬──────┬────────┬──────┐
│Where │ were │ the │ 2024 │Olymp.│held? │ The │ 2024 │ Summer │ ...  │
└──────┴──────┴─────┴──────┴──────┴──────┴─────┴──────┴────────┴──────┘
   ✗      ✗     ✗     ✗      ✗      ✗      ✓     ✓       ✓      ✓
  ◄──── masked from cross-attention ────►  ◄──── visible to decoder ──►

What the decoder can see:
┌──────┬──────┬────────┬──────────────────────────────────┐
│ The  │ 2024 │ Summer │ Olympics were held in Paris ...   │
└──────┴──────┴────────┴──────────────────────────────────┘
```

The prefix tokens are masked so the decoder cannot cross-attend to them. Only document hidden states are visible. These states were shaped by bidirectional attention with the prefix, but the prefix itself is gone. No RoPE correction is needed — T5-style encoders don't use RoPE on cross-attention keys.

#### Step 3: Cache the truncated encoder states to disk

```
┌──────────────────────────────────────────────────────────┐
│                  Encoder hidden states                    │
│                                                          │
│  ┌──────┬──────┬────────┬────────────────────────────┐   │
│  │ The  │ 2024 │ Summer │ Olympics were held in ...   │   │
│  │      │      │        │                            │   │
│  │  (shaped by prefix, but prefix is gone now)       │   │
│  └──────┴──────┴────────┴────────────────────────────┘   │
│                                                          │
│                    Save to disk / cache                   │
└──────────────────────────────────────────────────────────┘
```

This is the production use case: encode documents once offline with a prefix, then serve many queries against the cached states.

#### Step 4: Query arrives — decoder cross-attends to cached states (online)

```
Cached encoder states (from Steps 1-3):
┌──────┬──────┬────────┬──────────────────────────────────────┐
│ The  │ 2024 │ Summer │ Olympics were held in Paris, France  │
│      │      │        │ from 26 July to 11 August 2024...    │
└──────┴──────┴────────┴──────────────────────────────────────┘
                           ▲
                           │ cross-attention
                           │
┌──────────────────────────────────────────────────────────────┐
│                         Decoder                              │
│                                                              │
│  [BOS]  "Where" "were" "the" "2024" "Olympics" "held" "?"    │
│          \___________________  ___________________/          │
│                              \/                              │
│                        query tokens                          │
│                                                              │
│                       "Paris"  ","  "France"                 │
│                        \__________  _____/                   │
│                                   \/                         │
│                            answer tokens                     │
│                         (NLL scored here)                     │
└──────────────────────────────────────────────────────────────┘
```

At inference time, the real query arrives and the decoder cross-attends to the cached document states. Each decoder token has two attention paths: causal self-attention to preceding decoder tokens, and cross-attention to all visible encoder states. NLL is computed only on the answer token positions.

#### What changes between conditions

```
Condition:       bare              random                  oracle
                 ┌──────────┐     ┌────────────────────┐  ┌──────────────────────┐
Encoder input:   │   doc    │     │ climate  │   doc   │  │ Where were  │  doc   │
                 │          │     │ fish     │         │  │ the 2024    │        │
                 │          │     │ motor    │         │  │ Olympics    │        │
                 │          │     │          │         │  │ held?       │        │
                 └──────────┘     └────────────────────┘  └──────────────────────┘
                      │                  │                        │
                      ▼                  ▼                        ▼
                  [encode]           [encode]                 [encode]
                      │                  │                        │
                      ▼                  ▼                        ▼
Cached states:   [d1 d2 d3 ...]    [d1' d2' d3' ...]      [d1'' d2'' d3''...]
                      │                  │                        │
                 (unchanged)     (attention perturbed)     (attention perturbed
                                                           + semantic signal)
                      │                  │                        │
                      ▼                  ▼                        ▼
d vs bare:           —              +0.080 (ns)              +0.228 (***)
```

The same decoder, the same query, the same answer. Only the document hidden states differ — shaped by what prefix was present during encoding. Three random words ("climate fish motor") get 35% of the oracle benefit simply by perturbing the attention pattern.

---

### Slide: Conditions and Results (Exp 01)

**T5Gemma 2 4B-4B, N=500, MS MARCO**

Example: Document about the 2024 Paris Olympics.
- Query: "Where were the 2024 Summer Olympics held?"
- Answer: "Paris, France"

| Condition | Encoder prefix | d vs bare | Win% |
|-----------|---------------|----------|------|
| bare | (none — document only) | — | — |
| oracle | "Where were the 2024 Summer Olympics held?" | +0.228 (***) | 62% |
| surr_doc (kw5) | "Olympics Paris 2024 Games July" | +0.148 (**) | 58% |
| random | "climate fish motor" | +0.080 (ns) | 53% |
| surr_template | "What is Olympics?" | -0.069 (ns) | 47% |

The oracle prefix (the real query) is the best condition, but random words still get 35% of the benefit. Document keywords get 65%. A simple template that worked well in the previous version (v3) now fails — when the decoder already has the query, structural perturbation alone isn't enough on short documents.

---

### Slide: Why Structural Effects Are So Strong

**The key evidence across experiments:**

| Condition | Encoder prefix | d vs bare |
|-----------|---------------|----------|
| bare | (none) | — |
| oracle | "Where were the 2024 Olympics held?" | +0.228 (***) |
| doc keywords | "Olympics Paris 2024 Games July" | +0.148 (**) |
| wrong-doc keywords | "inflation GDP markets trade exports" | +0.129 (*) |
| random | "climate fish motor" | +0.080 (ns) |

"Climate fish motor" gets 35% of the oracle benefit. Keywords from the wrong document get 57%. The actual question only adds an 18% edge over keywords that have nothing to do with this passage.

**On long documents (~600 words), it gets more extreme (Exp 03, neural-bridge):**

| Condition | d vs bare | |
|-----------|----------|---|
| random prefix | **+0.624** (***) | Best |
| doc keywords | +0.502 (***) | |
| oracle (real query) | +0.306 (***) | Worst of the three |

Random words beat the real query by 2x. The structural fraction is 204% — the oracle's semantic content actively hurts compared to meaningless tokens.

The `[BOS]` token absorbs ~72-76% of all document-token attention. Adding ANY prefix tokens — even nonsense — perturbs this attention pattern, causing document tokens to redistribute attention among themselves (+0.15-0.22 nats entropy). The content of the prefix barely matters; its mere presence triggers beneficial reorganization. On long documents, the real query creates semantic interference — bidirectional attention bonds between query words and matching document words get severed at inference time, leaving "truncation wounds."

---

### Slide: Length Scaling (Exp 02)

**MS MARCO, documents padded to target lengths, N=500**

| Length (tokens) | d_oracle | d_random | Structural fraction |
|-----------------|----------|----------|-------------------|
| ~98 (original) | +0.238 (***) | -0.028 (ns) | ~0% |
| 256 | +0.352 (***) | +0.254 (***) | 72% |
| 512 | +0.323 (***) | +0.271 (***) | 84% |
| 1024 | +0.439 (***) | +0.357 (***) | 81% |
| 2048 | +0.392 (***) | +0.284 (***) | 72% |
| 4096 | +0.426 (***) | +0.221 (***) | 52% |

Enrichment grows with document length — oracle d rises from +0.238 to +0.439. On short documents (~98 tokens), the decoder's own query representation provides most of the structural redistribution, so random prefixes don't help. But at 256+ tokens, random prefixes become highly significant — the decoder's query alone can't reorganize attention across thousands of encoder positions.

---

### Slide: Cross-Dataset (Exp 03, neural-bridge)

**Naturally long documents (~604 words), N=500**

| Condition | d vs bare | Win% |
|-----------|----------|------|
| random | **+0.624** (***) | 75.6% |
| doc keywords | +0.502 (***) | 71.4% |
| oracle | +0.306 (***) | 64.6% |

Surrogates beat the oracle. The structural fraction is 204%. On naturally long documents, the real query in the encoder creates semantic interference with the document content — bidirectional attention bonds between overlapping words get severed at truncation. ANY short prefix provides pure structural benefit without this interference cost.

---

### Slide: Cross-Model Generalization (Exp 09)

**4 additional encoder-decoder models, same 500 MS MARCO samples**

| Model | Params | d_oracle | d_random | Structural % |
|-------|--------|----------|----------|-------------|
| T5Gemma-2-4B (ref) | 8B | +0.228 (***) | +0.080 (ns) | 35% |
| flan-t5-base | 250M | +0.251 (***) | +0.107 (*) | 43% |
| flan-t5-large | 780M | +0.320 (***) | +0.247 (***) | 77% |
| flan-t5-xl | 3B | +0.430 (***) | +0.030 (ns) | 7% |
| bart-large | 400M | +0.144 (**) | +0.043 (ns) | 30% |

Enrichment generalizes to all 4 models tested. The structural fraction varies wildly: flan-t5-xl is almost pure content (7% structural), flan-t5-large is dominantly structural (77%). The largest model (flan-t5-xl) has the strongest oracle effect but near-zero structural component — it can extract genuine content from the oracle prefix but gains nothing from random tokens. Architecture matters more than size for the mechanism mix.

---

### Slide: Summary of All Encoder-Decoder Investigations

| Investigation | Experiment | Key finding |
|--------------|-----------|-------------|
| Baseline enrichment | Exp 01 | Oracle d=+0.228, 61% of v3 preserved. Structural collapsed 85%→35%. |
| Length scaling | Exp 02 | Enrichment GROWS with doc length. Random significant at 256+ tokens. |
| Cross-dataset | Exp 03 | neural-bridge: surrogates beat oracle (structural fraction 204%). |
| Prefix content | Exp 04 | kw10 best surrogate (82% oracle). First sentence catastrophic (d=-0.298). |
| Truncation wound | Exp 05 | Requires BOTH coherence AND overlap (interaction d=-0.361). |
| Answer-length split | Exp 06 | Long answers > factoid (reverses v3). Structural 33%→74% by length. |
| Attention probing | Exp 07 | Query buffer modest (5.5%). Oracle shifts cross→self (37%→13%). |
| Length control | Exp 08 | "35% redundancy" was entirely a length artifact. Semantic benefits are independent. |
| Cross-model | Exp 09 | All 4 models significant. Structural fraction: 7% (xl) to 77% (large). |

---

## Part 2: Decoder-Only

---

### Slide 1: Initial Results Were Disappointing

**Setup**: Gemma 3 12B-IT, two-phase KV cache scoring.

Phase A caches `[BOS, prefix, \n, doc]` under causal attention, extracts doc-only cache with RoPE correction. Phase B evaluates `[\n, query, \n, answer]` against the primed cache.

#### Step-by-step process:

**Step 1: Encode prefix + document under causal attention**

```
Input tokens (left to right, causal attention):
┌─────┬────────────────────────────────────┬────┬──────────────────────────────────┐
│ BOS │ Where were the 2024 Olympics held? │ \n │ The 2024 Summer Olympics were     │
│     │            (prefix)                │    │ held in Paris, France, from 26    │
│     │                                    │    │ July to 11 August 2024...         │
│     │                                    │    │           (document)              │
└─────┴────────────────────────────────────┴────┴──────────────────────────────────┘
pos:  0     1    2    3   4     5       6    7     8    9     10      11     12

                    Causal attention:
                    Each token attends only to tokens BEFORE it.
                    Doc tokens see the prefix, but prefix never sees the doc.
```

Under causal attention, each token can only attend to preceding tokens. Document tokens see the prefix tokens that came before them, allowing the prefix to shape how the document is encoded. Unlike the encoder-decoder setup, the prefix never sees the document.

**Step 2: Remove the prefix from the KV cache**

```
Full KV cache:
┌─────┬──────┬──────┬─────┬──────┬──────┬──────┬────┬─────┬──────┬────────┐
│ BOS │Where │ were │ the │ 2024 │Olymp.│held? │ \n │ The │ 2024 │ Summer │
│pos 0│pos 1 │pos 2 │pos 3│pos 4 │pos 5 │pos 6 │pos7│pos 8│pos 9 │pos 10  │
└─────┴──────┴──────┴─────┴──────┴──────┴──────┴────┴─────┴──────┴────────┘
   ✓     ✗      ✗     ✗     ✗      ✗      ✗     ✗    ✓     ✓       ✓
  keep  ◄────────── remove prefix + \n ──────────►   ◄── keep doc ──────►

Pruned KV cache:
┌─────┬─────┬──────┬────────┬──────────────────┐
│ BOS │ The │ 2024 │ Summer │ Olympics were ... │
│pos 0│pos 8│pos 9 │pos 10  │ pos 11 ...       │  ← positions still wrong!
└─────┴─────┴──────┴────────┴──────────────────┘
```

The prefix shaped these representations during Step 1, but is now removed from the cache. BOS is retained so the cache length matches the expected position offset (this was critical for fixing the look-ahead bug). Problem: doc positions are still 8, 9, 10... but the cache now has only 1+D entries.

**Step 3: RoPE-correct the document positions**

```
Before correction:
┌─────┬─────┬──────┬────────┬──────────────────┐
│ BOS │ The │ 2024 │ Summer │ Olympics were ... │
│pos 0│pos 8│pos 9 │pos 10  │ pos 11 ...       │
└─────┴─────┴──────┴────────┴──────────────────┘
         ▲
         │ Gap! Positions jump 0 → 8.
         │ RoPE encodes position in key vectors.

                    reposition_kv_cache()
                    Undo RoPE(old_pos) → apply RoPE(new_pos)

After correction:
┌─────┬─────┬──────┬────────┬──────────────────┐
│ BOS │ The │ 2024 │ Summer │ Olympics were ... │
│pos 0│pos 1│pos 2 │pos 3   │ pos 4 ...        │
└─────┴─────┴──────┴────────┴──────────────────┘
```

Unlike the encoder-decoder model (which doesn't use RoPE on cross-attention keys), the decoder-only model encodes position directly in the key vectors via RoPE. After removing the prefix, the document keys still carry the old position information. We must undo the old RoPE rotation and apply new rotations for the corrected positions.

**Step 4: Evaluate the real query + answer against the primed cache**

```
Primed KV cache (from Steps 1-3):
┌─────┬─────┬──────┬────────┬──────────────────────────────────┐
│ BOS │ The │ 2024 │ Summer │ Olympics were held in Paris ...   │
│pos 0│pos 1│pos 2 │pos 3   │ pos 4 ...              pos D    │
└─────┴─────┴──────┴────────┴──────────────────────────────────┘
                                                          │
                                               causal attention
                                                          │
New tokens (Phase B):                                     │
┌────┬───────────────────────────────────┬────┬───────────┘
│ \n │ Where were the 2024 Olympics held?│ \n │ Paris, France
│    │          (real query)             │    │  (answer)
│D+1 │ D+2  D+3  D+4  D+5  D+6    D+7  │D+8 │ D+9  D+10
└────┴───────────────────────────────────┴────┴───────────
                                                ▲
                                          NLL scored here
```

Each Phase B token attends to all cached doc tokens (primed by the prefix) and all preceding Phase B tokens. The prefix is gone, but its influence remains in the document's key-value representations.

#### Exp 01 Results (N=400, MS MARCO)

| Condition | Prefix content | d vs bare | Win% |
|-----------|---------------|----------|------|
| oracle_full | "Where were the 2024 Olympics held?" (kept in cache) | **-0.362** (***) | 25% |
| oracle | "Where were the 2024 Olympics held?" (removed) | **-0.151** (**) | 32% |
| adversarial | "The recipe calls for two cups of flour and one egg." | +0.007 (ns) | 50% |
| surr_universal | "Read and understand the following passage carefully." | +0.079 (ns) | 55% |
| surr_extractor | "Extract the key information from the following text." | +0.264 (***) | 68% |

The real query as prefix hurts (d=-0.15, win only 32%) — the opposite of encoder-decoder (+0.23). The only thing that helped was a generic extraction prompt. Giving the model MORE information (oracle, oracle_full) made things worse, not better.

---

### Slide 2: Token-Matched Probing — No Semantic Gradient

**Exp 02-03**: 13 conditions, all token-matched (same prefix length per sample), tested on hard examples across 4 datasets (MS MARCO, SQuAD, TriviaQA, HotpotQA).

Example sample:
- **Document**: "The 2024 Summer Olympics, officially the Games of the XXXIII Olympiad, were held in Paris, France, from 26 July to 11 August 2024. Paris was awarded the Games in 2017 at the 131st IOC Session..."
- **Query**: "Where were the 2024 Summer Olympics held?"
- **Answer**: "Paris, France"

**Phase A** (conditioning): `[BOS] prefix \n doc...` — prefix is removed after encoding, doc cache is RoPE-corrected.

**Phase B** (inference): `\n Where were the 2024 Summer Olympics held? \n Paris, France` — NLL scored on answer tokens only.

| Condition | Prefix content (all exactly Q tokens) | Type | Delta d |
|-----------|---------------------------------------|------|---------|
| extractor_matched | "Extract the key information from the following" | generic task-framing | **+0.357** (***) |
| adversarial_matched | "The recipe calls for two cups of flour" | generic unrelated | +0.169 (***) |
| repeat_token | "the the the the the the the the the" | purely structural | +0.124 (**) |
| llm_summarize | "The passage covers the 2024 Paris Olympics hosting" | LLM doc-specific | +0.052 (ns) |
| llm_extract | "2024 Olympics Paris Summer Games host city venue" | LLM doc-specific | -0.023 (ns) |
| paraphrase | "In which city were the Summer 2024 Games held" | LLM query paraphrase | -0.122 (**) |
| oracle | "Where were the 2024 Summer Olympics held?" | real query | **-0.253** (***) |
| llm_question | "What city hosted the 2024 Summer Olympic Games" | LLM doc-specific | **-0.343** (***) |

The gradient goes the wrong direction: the more the prefix resembles the actual question, the worse performance gets. Generic task-framing with zero document knowledge is the best. LLM surrogates that "know" about the passage all lose to a fixed instruction string.

Key stats:
- 0/4 datasets show monotonic semantic gradient (Spearman rho=+0.15, ns)
- LLM doc-specific surrogates all LOSE to generic framing (p<0.001)
- 7/11 conditions have consistent sign across all 4 datasets

---

### Slide 3: Instruction Framing — "Comprehend" Changes the Game

**Exp 04**: 8 instruction prefixes tested coherent vs scrambled across 4 datasets. Three-level decomposition isolates why each instruction helps or hurts.

**Decomposition method**:
- **Total** = coherent instruction - bare
- **Structural** = scrambled random words - bare (any tokens in the prefix slot help)
- **Vocabulary** = scrambled instruction - scrambled random words (the specific words matter, but not their order)
- **Meaning** = coherent instruction - scrambled instruction (word order matters on top of vocabulary)

These three components are additive: structural + vocabulary + meaning = total. Percentages are each component divided by total. When meaning is negative (coherent order hurts), vocabulary can exceed 100%.

Example — "comprehend" (d=+0.470):
```
Structural:  +0.028  →   6%   (any prefix helps a little)
Vocabulary:  +0.207  →  44%   (the words "comprehend", "thoroughly" help)
Meaning:     +0.235  →  50%   (putting them in order helps more)
```

| Instruction | Example prefix | Total d | Vocab | Meaning | |
|------------|---------------|---------|-------|---------|---|
| **comprehend** | "Comprehend the following passage thoroughly and carefully." | **+0.470** (***) | 44% | **+50%** | Only positive meaning |
| extract_general | "Extract the key information from the following text." | +0.357 (***) | 84% | 8% | Vocabulary-driven |
| classify | "Classify the content and meaning of this passage." | +0.301 (***) | 71% | 19% | |
| summarize | "Summarize the main points of the following passage." | +0.245 (***) | 68% | 15% | |
| extract_claims | "Extract the central claims and assertions from this text." | +0.198 (**) | 76% | -3% | |
| extract_entities | "Extract the named entities and key terms from this text." | +0.132 (*) | 104% | **-27%** | Coherent order hurts |
| answer_question | "Answer the following question based on the passage." | +0.088 (ns) | 59% | 12% | |
| generate_qa | "Generate a question and answer pair about this passage." | +0.014 (ns) | — | — | No benefit at all |

"Comprehend" is the only instruction where putting the words in the right order actually helps. For extract_entities, the words "extract", "entities", "key" activate useful representations — but in coherent order they prime the model to list entities rather than answer questions, so meaning is negative. "Comprehend" activates a general deep-reading mode without priming any specific subtask.

Key contrast with encoder-decoder: In enc-dec, ANY prefix worked (85% structural — a binary switch). In decoder-only, what you say matters. Scrambled extraction tokens work almost as well as coherent ones for most instructions — vocabulary alone carries 49-84% of the effect. "Comprehend" is the exception where coherent semantics contribute half the benefit.

---

### Slide 4: Meaning Grows With Length — The Key Discovery

**Exp 05-06**: Scaled "comprehend" from 8 to 256 tokens across 14 datasets.

| Length | Structural | Vocabulary | Meaning | Total d |
|--------|-----------|-----------|---------|---------|
| L=8 | 48% | 29% | 23% | +0.25 |
| L=16 | 38% | 33% | 29% | +0.30 |
| L=32 | 10% | 65% | 25% | +0.35 |
| **L=64** | **29%** | **21%** | **50%** | **+0.39** |
| L=128 | — | — | — | +0.36 |
| L=256 | — | — | — | +0.33 |

At short prefix lengths (Q-matched, ~7-19 tokens), meaning was -11% of the total effect. At L=64, meaning rises to 50%. This reverses Exp 04's conclusion that meaning is negligible — the instruction just needs enough repetitions for the model to internalize the task framing. Vocabulary saturates while meaning continues to grow.

Performance peaks at L=64 then declines at 128-256. The model appears to overfit to the instruction at longer lengths.

---

### Slide 5: 14-Dataset Meta-Analysis — GSM8K Is Champion

**Exp 06**: comprehend @ L=64, N=200 hard examples per dataset.

| Tier | Dataset | d (comprehend @ L=64) | Task type |
|------|---------|----------------------|-----------|
| **Strong** | GSM8K | **+1.334** | Multi-step math |
| (d > 0.6) | DROP | +0.914 | Discrete reasoning |
| | HotpotQA | +0.762 | Multi-hop QA |
| | SQuAD | +0.693 | Extractive QA |
| | RACE-high | +0.660 | Exam comprehension |
| **Moderate** | Quoref | +0.540 | Coreference QA |
| (0.3-0.6) | ROPES | +0.486 | Reasoning over text |
| | RACE-mid | +0.477 | Exam comprehension |
| | MultiRC | +0.440 | Multi-sentence |
| | TriviaQA | +0.380 | Factoid QA |
| **Weak/Neg** | MS MARCO | +0.100 | Short factoid |
| | ReCoRD | +0.040 | Cloze QA |
| | QuALITY | -0.170 | Long article (truncated) |
| | BoolQ | **-0.510** | Boolean QA |

GSM8K and BoolQ both have 1-word answers — but d=+1.33 vs d=-0.51. Answer length correlation: rho=-0.195, p=0.59 (not significant). Reasoning complexity, not answer length, determines prefix benefit. Three tiers emerge: complex reasoning tasks with extractive answers benefit most; simple factoid and binary QA benefit least.

---

### Slide 6: Decoder-Only vs Encoder-Decoder — Two Different Mechanisms

| | Encoder-Decoder | Decoder-Only |
|---|----------------|-------------|
| Does prefix help? | Yes (d=+0.23 to +0.45) | Yes, with right instruction (d=+0.39) |
| Does content matter? | Barely — 85% structural | Yes — "comprehend" is uniquely effective |
| Oracle vs random | Oracle better (d=+0.15 gap) | Oracle hurts (d=-0.15 vs bare) |
| Best prefix | Anything (even 1 random word) | "Comprehend" repeated to 64 tokens |
| Mechanism | Binary attention redistribution | Vocabulary + meaning (50% at L=64) |
| LLM surrogates | Beat oracle on long docs | Lose to generic framing (p<0.001) |
| Strongest dataset | neural-bridge (long docs) | GSM8K (complex reasoning, d=+1.33) |
| Practical recipe | Prepend any short prefix | "Comprehend the following carefully:" x L=64 |

The encoder-decoder mechanism is a binary switch — any prefix triggers attention redistribution. The decoder-only mechanism is richer: the model responds to task-framing semantics, with meaning growing as the instruction is reinforced through repetition.

---

### Key difference: Bidirectional vs Causal

```
Encoder-Decoder                          Decoder-Only
┌──────────────────────┐                 ┌──────────────────────┐
│   BIDIRECTIONAL      │                 │   CAUSAL (left→right)│
│                      │                 │                      │
│  prefix ↔ doc        │                 │  prefix → doc        │
│  doc ↔ prefix        │                 │  doc ✗ prefix        │
│  doc ↔ doc           │                 │  doc → doc           │
│                      │                 │                      │
│  Every doc token     │                 │  Doc tokens see      │
│  sees every prefix   │                 │  prefix tokens       │
│  token AND vice      │                 │  before them only.   │
│  versa.              │                 │  Prefix never sees   │
│                      │                 │  the document.       │
│  No RoPE correction  │                 │  RoPE correction     │
│  needed after        │                 │  required after      │
│  truncation.         │                 │  removing prefix.    │
└──────────────────────┘                 └──────────────────────┘
```

In the encoder-decoder model, bidirectional attention means the prefix and document form deep mutual bonds — every token attends to every other token. This is what makes the structural effect so powerful (any perturbation to this full attention graph reshapes all representations) but also what makes truncation dangerous (severing strong bonds leaves wounds).

In the decoder-only model, causal attention means the prefix influences the document but the document doesn't influence the prefix. This is a weaker coupling, which is why the structural effect is smaller and content matters more — the prefix has to actively guide the model through its semantic content rather than just perturbing a bidirectional attention graph.
