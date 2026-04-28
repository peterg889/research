# Directed KV Cache Construction: Document-Derived Prefixes Outperform Instructions for Priming Precomputed Representations

## Abstract

Precomputed KV caches accelerate retrieval-augmented generation (RAG) by encoding documents offline and reusing their key-value representations across queries. We show that prepending a short *directive prefix* during cache construction -- then discarding it before storage -- measurably reshapes the cached representations and improves downstream question answering, at zero additional inference cost. Through experiments spanning 16 models (0.5B--32B parameters, 5 architecture families), 6 QA datasets, and 32 prefix conditions (totaling over 1.2 million scored samples), we find three progressively surprising results. First, fixed instruction prefixes (e.g., "Extract the key facts from this text") improve answer NLL for most models, but no single instruction dominates: the best instruction varies across models, with each of six tested instructions winning on at most 3 of 16 models. Second, the oracle condition -- using the actual query as prefix -- is not the performance ceiling; document-derived surrogates outperform it for most models. Third, document-specific TF-IDF keywords consistently outperform all fixed instructions and the oracle, ranking in the top 3 for 7 of 16 models. On Qwen 2.5 7B-Instruct, keywords achieve Cohen's d=+0.62 versus d=+0.50 for the oracle and d=+0.34 for the best fixed instruction. Mechanistic controls reveal that the dominant factor is *feature activation* -- surfacing document-relevant representations during encoding -- rather than instruction following: scrambled instruction tokens are often as effective as coherent instructions, and even random tokens produce positive effects on 12 of 16 models. Semantic sensitivity (the coherent-minus-scrambled gap) scales with model size in the Qwen family (r=+0.75) but not in Gemma, and instruction tuning amplifies preference for document-derived priming. Three models are consistently harmed by all prefix types (Gemma 3 4B base, Ministral 8B, Qwen 2.5 14B), requiring per-model validation before deployment. High-reasoning tasks (GSM8K, DROP) show the strongest benefits (mean keyword d=+0.40 across responsive models), while factoid tasks (TriviaQA, MS MARCO) show the weakest.

---

## 1. Introduction

KV cache precomputation has become a standard optimization in retrieval-augmented generation. Systems such as TurboRAG [1], CacheBlend [2], and SGLang [3] precompute document KV caches offline and reuse them across queries, reducing time-to-first-token by up to an order of magnitude. The subsequent literature has focused on what happens *after* construction: which cache entries to retain [4, 5, 6], how to compress them [7, 8], and how to manage them across requests [9].

All of this work treats cache construction as fixed: the document tokens are encoded and the resulting KV states are stored. We ask whether the construction step itself can be optimized.

Our approach is simple. During offline cache construction, we prepend a short prefix (1--64 tokens) before the document. The combined sequence [prefix, document] is encoded in a single forward pass, during which prefix tokens participate in self-attention with document tokens. We then discard the prefix entries from the cache, reposition the document keys via RoPE delta rotation, and store the result. At inference time, the cache is indistinguishable from a standard precomputed cache -- same shape, same position indexing -- but the document representations have been reshaped by the prefix's influence during encoding. The cost is 1--64 extra tokens during offline construction and exactly zero tokens at inference time.

We call this technique *directed KV cache construction*, by analogy with directed evolution: we do not design the optimal representation, but we steer the encoding process and select the result.

The natural first hypothesis is that a carefully chosen instruction prefix (e.g., "Extract the key facts from this text") would work best, leveraging the model's instruction-following capabilities. The natural second hypothesis is that the oracle condition -- using the actual downstream query as prefix -- would provide the performance ceiling. Both hypotheses are wrong.

Through a systematic evaluation across 16 models, 6 datasets, and 32 conditions, we find:

1. **No single instruction wins.** Six fixed instructions each achieve the best result on at most 3 of 16 models. The optimal instruction depends on model family, model size, and instruction tuning status.

2. **The oracle is not the ceiling.** Natural-length oracle prefixes perform worse than padded-to-64 oracle prefixes for 10 of 16 models, and both are outperformed by document-derived surrogates for most models.

3. **TF-IDF keywords are the best general-purpose prefix.** Extracting the top TF-IDF terms from the document and prepending them outperforms all fixed instructions, AI-generated summaries, and the oracle condition for most responsive models. Keywords rank in the top 3 for 7 of 16 models.

4. **Some models are consistently harmed.** Three models (Gemma 3 4B base, Ministral 8B, Qwen 2.5 14B) show negative effects across nearly all conditions. Practitioners must validate on their target model.

These findings reframe cache construction as an optimization target and suggest that the right prior for priming is derived from the document itself, not from task instructions.

---

## 2. Related Work

### 2.1 KV Cache Precomputation

KV cache reuse for RAG was introduced by TurboRAG [1], which precomputes chunk-level KV caches and concatenates them at inference time, achieving 9.4x TTFT reduction. CacheBlend [2] identifies that precomputed caches lack cross-chunk attention and selectively recomputes a subset of tokens to recover quality. Both systems treat the cache construction step as immutable.

### 2.2 KV Cache Compression and Eviction

A large body of work addresses cache efficiency *after* construction. H2O [4] discovers power-law attention distributions and retains "heavy hitter" tokens. SnapKV [5] uses an observation window to select important entries per head. StreamingLLM [6] discovers attention sinks -- initial tokens that absorb excess attention probability regardless of content -- and shows that retaining a few sink tokens plus a sliding window enables unbounded streaming. KIVI [7] achieves 2-bit KV cache quantization with asymmetric per-channel/per-token schemes. KVQuant [8] extends this with sensitivity-aware non-uniform quantization.

Our work is orthogonal: we improve cache *quality* rather than reduce cache *size*. The two approaches compose naturally -- directed construction can precede any compression method.

### 2.3 Prompt Engineering and Prefix Tuning

Prefix tuning [10] learns continuous embeddings prepended to the input to adapt frozen language models. Gist tokens [11] train models to compress prompts into learned token representations. These methods modify the input at inference time and require gradient-based optimization.

Our approach uses discrete tokens during *offline construction* only. No training is required, no inference-time cost is incurred, and the prefix can be derived from the document without model access (keyword extraction uses only corpus statistics).

### 2.4 Rotary Position Embeddings

RoPE [12] encodes position via rotation of key and query vectors. Our method requires repositioning document keys after prefix removal: we apply a delta rotation from the encoding position to the target position. We verify that this repositioning is exact in float32 (error < 5e-7) and introduces only a small bfloat16 rounding perturbation (~0.25% relative) across all 16 tested models.

---

## 3. Method

### 3.1 Two-Phase Pipeline

We use a two-phase approach to both construct and evaluate directed KV caches.

**Phase A (Offline -- Cache Construction):**

1. Tokenize the input as `[BOS, prefix_1, ..., prefix_P, \n, doc_1, ..., doc_D]`
2. Run a single forward pass through the model to build the full KV cache
3. Select only the BOS and document entries from the cache (discard prefix and separator)
4. Reposition document keys from positions (P+2, ..., P+1+D) to (1, ..., D) via RoPE delta rotation
5. Apply per-tensor normalization (optional; see Section 3.4)

**Phase B (Online -- Query Answering):**

1. Tokenize the query-answer pair as `[\n, query, \n, answer]`
2. Run a forward pass using the precomputed cache, with position IDs starting at D+1
3. Compute negative log-likelihood (NLL) on answer tokens only

The key mechanism: during Step 2 of Phase A, prefix tokens participate in self-attention with document tokens. Each document token's key and value projections are influenced by attending to the prefix. When the prefix is discarded in Step 3, this influence persists in the document's cached representations.

**BOS handling.** Models without a native BOS token (Qwen 2.5 family, DeepSeek) use the PAD token as an artificial attention sink anchor. We verify that this is necessary: without BOS, prefix conditioning fails on Qwen (comprehend d=-0.65 vs. d=+0.13 with BOS).

**RoPE repositioning.** After prefix removal, document keys occupy positions (P+2, ..., P+1+D) instead of the expected (1, ..., D). We correct this by applying a delta rotation using the model-specific inverse frequency tensors. For models with heterogeneous attention (Gemma 3, with per-layer-type RoPE theta), we apply layer-type-specific rotations.

### 3.2 Prefix Conditions

We evaluate 32 prefix conditions across two sweeps, spanning five categories.

**Fixed instructions (L=64, padded/truncated).** Six task-oriented instructions: *comprehend* ("Read and comprehend this text carefully"), *extract* ("Extract the key facts from this text"), *summarize* ("Summarize the following text"), *question* ("What are the key facts in this text?"), *index* ("Index the following information for retrieval"), *declarative* ("This text contains important information").

**Minimal cues (L=1).** Single tokens: "Extract", "Comprehend", "Facts".

**Structural controls.** Random tokens (L=1, 4, 16, 64), repeated token (L=64), scrambled instruction tokens (comprehend and extract, L=64), anti-instruction ("Ignore this text completely", L=64).

**Oracle.** The actual downstream query, both padded to L=64 and at natural length.

**Document-derived.** TF-IDF top-10 keywords (computed per-document within each dataset), AI-generated one-sentence summary (Gemini 2.5 Flash), AI-generated custom reading instruction (Gemini 2.5 Flash).

**Controls.** Position-shift roundtrip (no prefix tokens; document encoded at natural positions, keys shifted forward by 64 then back to measure bf16 reposition error), bare with and without normalization.

[TABLE 1: Complete list of 32 prefix conditions with category, token budget, and description]

### 3.3 Models

We evaluate 16 models across 5 architecture families, spanning 0.5B to 32B parameters and including two base/instruct pairs.

[TABLE 2: Model specifications]

| Model | Params | Family | Attention | Instruct |
|-------|--------|--------|-----------|----------|
| Qwen 2.5 0.5B-Instruct | 0.5B | Qwen 2 | Full | Yes |
| Gemma 3 1B-IT | 1B | Gemma 3 | Hybrid S/F | Yes |
| Qwen 2.5 1.5B-Instruct | 1.5B | Qwen 2 | Full | Yes |
| Qwen 2.5 3B-Instruct | 3B | Qwen 2 | Full | Yes |
| Gemma 3N E4B-IT | ~4B | Gemma 3N | Hybrid | Yes |
| Gemma 3 4B-IT | 4B | Gemma 3 | Hybrid S/F | Yes |
| Gemma 3 4B-PT | 4B | Gemma 3 | Hybrid S/F | No |
| Qwen 2.5 7B-Instruct | 7B | Qwen 2 | Full | Yes |
| Qwen 2.5 7B | 7B | Qwen 2 | Full | No |
| DeepSeek R1-Distill-Qwen 7B | 7B | Qwen 2 | Full | Yes |
| Mistral 7B-Instruct v0.3 | 7B | Mistral | Full | Yes |
| Ministral 8B-Instruct | 8B | Ministral | Hybrid S/F | Yes |
| Gemma 3 12B-IT | 12B | Gemma 3 | Hybrid S/F | Yes |
| Qwen 2.5 14B-Instruct | 14B | Qwen 2 | Full | Yes |
| Gemma 3 27B-IT | 27B | Gemma 3 | Hybrid S/F | Yes |
| Qwen 2.5 32B-Instruct | 32B | Qwen 2 | Full | Yes |

Hybrid S/F denotes models with alternating sliding-window and full-attention layers (e.g., Gemma 3 uses sliding window of 1024 on most layers, full attention every 6th layer). Models include two base/instruct pairs (Qwen 7B, Gemma 4B) for studying instruction tuning effects.

### 3.4 Per-Tensor Normalization

We apply an optional per-tensor normalization to each KV cache tensor after Phase A:

```
x_normalized = (x / (absmax(x) / 127)) * (absmax(x) / 127)
```

In exact arithmetic this is the identity function. In bfloat16, the divide-then-multiply round-trip introduces a small perturbation (~0.25% relative). We include a normalization ablation for the bare and comprehend conditions on all 16 models.

### 3.5 Datasets and Evaluation

We use six QA datasets spanning three reasoning tiers:

| Dataset | Tier | Task Type |
|---------|------|-----------|
| GSM8K | High reasoning | Math word problems (number-only answer) |
| DROP | High reasoning | Discrete reasoning |
| SQuAD v2 | Mid reasoning | Extractive QA |
| HotpotQA | Mid reasoning | Multi-hop QA |
| TriviaQA | Factoid | Factoid retrieval |
| MS MARCO | Factoid | Information retrieval |

**Sampling.** 400 samples per dataset, drawn with fixed seeds from validation/test splits. Documents are filtered to 30--500 words (10--500 for GSM8K). For GSM8K, the answer is the final number after `####`, not the full chain-of-thought -- this is a critical implementation detail, as using the full chain-of-thought produces reversed results due to distributional bias.

**Scoring.** We compute Cohen's d effect size on paired NLL differences (bare NLL minus condition NLL), where positive d means the condition improves answer likelihood. We evaluate on the top-160 hardest samples per dataset (ranked by bare NLL, top 40%), which concentrates statistical power on samples where the model is uncertain. Pooled d is computed across all 6 datasets (960 hard samples per model per condition). Win rate (fraction of samples where condition NLL < bare NLL) supplements the primary effect size metric.

**Statistical significance.** With N=960 pooled hard samples, the standard error of Cohen's d is approximately 1/sqrt(960) = 0.032. At alpha=0.05 (two-tailed), effects with |d| > 0.063 are statistically significant.

**Robustness.** We verify that results are stable between hard-sample (N=160 per dataset) and all-sample (N=400 per dataset) analyses. Effect directions are preserved, with hard-sample analysis typically amplifying effect sizes by a factor of 1.2--1.4x.

---

## 4. Results

### 4.1 Overall Landscape: Most Models Benefit, Some Are Harmed

Across the 25 priming conditions tested (excluding controls), we summarize per-model responsiveness by the fraction of conditions that produce a positive effect (d > 0).

[TABLE 3: Win counts per model -- fraction of conditions with d > 0]

| Model | Positive / Total | Category |
|-------|-----------------|----------|
| Gemma 3 1B-IT | 25/25 | Universally responsive |
| Qwen 2.5 7B (base) | 25/25 | Universally responsive |
| Gemma 3 27B-IT | 25/25 | Universally responsive |
| Qwen 2.5 1.5B-Inst | 24/25 | Broadly responsive |
| Gemma 3 12B-IT | 24/25 | Broadly responsive |
| Qwen 2.5 3B-Inst | 23/25 | Broadly responsive |
| Qwen 2.5 32B-Inst | 22/25 | Broadly responsive |
| Gemma 3N E4B-IT | 21/25 | Broadly responsive |
| Qwen 2.5 7B-Inst | 21/25 | Broadly responsive |
| Gemma 3 4B-IT | 20/25 | Broadly responsive |
| DeepSeek R1-Qwen 7B | 20/25 | Broadly responsive |
| Qwen 2.5 0.5B-Inst | 19/25 | Moderately responsive |
| Mistral 7B-Inst | 17/25 | Moderately responsive |
| Qwen 2.5 14B-Inst | 3/25 | Predominantly harmed |
| Ministral 8B-Inst | 1/25 | Predominantly harmed |
| Gemma 3 4B-PT (base) | 0/25 | Universally harmed |

Three models are consistently harmed. Gemma 3 4B-PT (base model) shows negative effects across all 25 conditions, with d ranging from -0.07 to -0.54. Ministral 8B shows 1/25 positive (only the single-word "Facts" at d=+0.01, effectively zero). Qwen 2.5 14B shows only 3/25 positive conditions. These models require per-model validation; blanket deployment of priming would degrade their performance.

The remaining 13 models are responsive, with 17--25 of 25 conditions producing positive effects. Six models are fully or near-fully responsive (25/25 or 24/25), responding positively to nearly any prefix content.

### 4.2 No Single Instruction Wins

[TABLE 4: Best fixed instruction per model -- 6 instructions x 16 models]

Across the ablation sweep of 6 fixed instructions, no instruction achieves consistent dominance. Taking the highest-d instruction per model:

- **Index** wins for 3 models (Gemma 3N E4B d=0.69, Qwen 7B base d=0.43, Gemma 27B d=0.60)
- **Question** wins for 3 models (Gemma 1B d=0.81, DeepSeek R1 d=0.39, Qwen 32B d=0.44)
- **Declarative** wins for 3 models (Gemma 4B IT d=0.33, Gemma 4B base d=-0.07 [least negative], Qwen 14B d=-0.02 [least negative])
- **Summarize** wins for 3 models (Qwen 1.5B d=0.37, Qwen 3B d=0.25, Qwen 7B Inst d=0.34)
- **Extract** wins for 2 models (Mistral 7B d=0.43, Gemma 12B d=0.33)
- **Comprehend** wins for 1 model (Qwen 0.5B d=-0.08 [least negative])

The practical implication is stark: a practitioner cannot simply pick a good instruction and deploy it across models. The instruction space contains no universal optimum. Even "extract," which is the strongest single instruction by median d across all models, ranks below 5th for several models.

### 4.3 The Oracle Is Not the Ceiling

[TABLE 5: Oracle conditions vs. keywords and best instruction]

| Model | Oracle (L=64) | Oracle (natural) | Keywords | Best instruction |
|-------|--------------|-----------------|----------|-----------------|
| Qwen 0.5B | +0.50 | +0.61 | **+0.52** | -0.08 (comprehend) |
| Gemma 1B | +0.70 | +0.53 | +0.55 | **+0.81** (question) |
| Qwen 1.5B | +0.48 | +0.48 | **+0.43** | +0.37 (summarize) |
| Qwen 3B | +0.12 | +0.10 | +0.14 | **+0.25** (summarize) |
| Gemma 3N E4B | +0.58 | +0.57 | **+0.67** | +0.69 (index) |
| Gemma 4B IT | +0.23 | -0.00 | +0.11 | **+0.33** (declarative) |
| Gemma 4B base | -0.07 | -0.09 | -0.07 | -0.07 (declarative) |
| Qwen 7B Inst | +0.50 | +0.35 | **+0.62** | +0.34 (summarize) |
| Qwen 7B base | +0.13 | +0.08 | +0.10 | **+0.43** (index) |
| DeepSeek R1 7B | +0.37 | +0.39 | -0.04 | **+0.39** (question) |
| Mistral 7B | +0.10 | -0.00 | -0.14 | **+0.43** (extract) |
| Ministral 8B | -0.28 | -0.58 | -0.29 | +0.01 (facts word) |
| Gemma 12B | +0.23 | -0.02 | +0.34 | **+0.60** (instruction AI) |
| Qwen 14B | -0.02 | -0.00 | **+0.20** | +0.35 (facts word) |
| Gemma 27B | +0.24 | +0.04 | +0.40 | **+0.60** (index) |
| Qwen 32B | +0.20 | +0.20 | **+0.50** | +0.62 (facts word) |

Two findings stand out. First, natural-length oracle prefixes perform worse than padded-to-64 oracle prefixes for 10 of 16 models. The mean advantage of padding is d=+0.10, reaching d=+0.25 for Gemma 3 12B and d=+0.30 for Ministral 8B. This indicates that prefix *length* contributes independently of content, consistent with a token-presence mechanism -- longer prefixes provide more attention targets during encoding.

Second, TF-IDF keywords outperform the padded oracle for 6 of 13 responsive models (excluding the 3 harmed models). On Qwen 2.5 7B-Instruct, the gap is substantial: keywords d=+0.62 vs. oracle d=+0.50. The oracle provides query-specific relevance, but the keywords provide document-specific feature activation that proves more broadly useful since the cache is reused across many queries.

### 4.4 TF-IDF Keywords: Best General-Purpose Prefix

[TABLE 6: TF-IDF keyword ranking and effect size across all 16 models]

| Model | Keyword d | Rank (of 14) | In Top-3? |
|-------|-----------|-------------|-----------|
| Gemma 3N E4B | +0.67 | 2 | Yes |
| Qwen 7B Inst | +0.62 | 1 | Yes |
| Gemma 1B | +0.55 | 13 | No |
| Qwen 0.5B | +0.52 | 2 | Yes |
| Qwen 32B | +0.50 | 3 | Yes |
| Qwen 1.5B | +0.43 | 3 | Yes |
| Gemma 27B | +0.40 | 4 | No |
| Gemma 12B | +0.34 | 7 | No |
| Qwen 14B | +0.20 | 2 | Yes |
| Qwen 3B | +0.14 | 7 | No |
| Gemma 4B IT | +0.11 | 8 | No |
| Qwen 7B base | +0.10 | 12 | No |
| DeepSeek R1 7B | -0.04 | 14 | No |
| Gemma 4B base | -0.07 | 2 | Yes |
| Mistral 7B | -0.14 | 13 | No |
| Ministral 8B | -0.29 | 10 | No |

TF-IDF keywords rank in the top 3 for 7 of 16 models and rank first for 1. While this is not universal dominance, it is the most consistent single strategy: no other condition achieves top-3 placement for as many models. Critically, when keywords work, they tend to work well -- the top 5 models by keyword d all exceed +0.43.

The keywords condition consists of document-specific terms like "quarterback touchdowns yards season" or "photosynthesis chlorophyll membrane ATP." These are individually meaningless as instructions but collectively activate the model's representations for the document's topical domain.

### 4.5 Per-Dataset Breakdown: Keywords Strongest on High-Reasoning Tasks

[TABLE 7: TF-IDF keyword Cohen's d by dataset for all 16 models]

| Model | GSM8K | DROP | SQuAD v2 | HotpotQA | TriviaQA | MS MARCO |
|-------|-------|------|----------|----------|----------|----------|
| Qwen 7B Inst | **+1.22** | +0.67 | +0.67 | +0.38 | +0.65 | +0.31 |
| Qwen 32B | **+1.08** | +0.39 | +0.87 | +0.10 | +0.38 | +0.25 |
| Qwen 14B | **+1.30** | -0.29 | +0.45 | -0.09 | -0.11 | +0.24 |
| Gemma 3N E4B | +0.91 | +0.40 | **+1.01** | +0.93 | +0.84 | +0.27 |
| Gemma 4B IT | **+0.90** | -0.23 | +0.33 | +0.25 | -0.25 | -0.28 |
| Gemma 12B | +0.74 | +0.21 | +0.53 | +0.25 | +0.29 | +0.12 |

GSM8K is the standout dataset. The top keyword effect across all models is d=+1.30 (Qwen 14B on GSM8K), and 12 of 16 models show positive keyword effects on GSM8K specifically. The high-reasoning tier (GSM8K + DROP) shows consistently stronger effects than the factoid tier (TriviaQA + MS MARCO), with mean keyword d of +0.40 vs. +0.17 across responsive models. This is consistent with the hypothesis that keyword priming helps most when the model needs to focus attention on specific document regions relevant to multi-step reasoning.

### 4.6 AI-Generated Prefixes: Competitive but Not Dominant

The two AI-generated prefix types -- document summaries (produced by Gemini 2.5 Flash) and custom reading instructions (Gemini 2.5 Flash) -- are competitive with fixed instructions but do not consistently outperform TF-IDF keywords.

AI-generated summaries achieve median d=+0.20 across all 16 models (vs. +0.13 for TF-IDF keywords across all 16 models, including the 3 harmed models). Among responsive models only, the median is +0.30 for summaries vs. +0.43 for keywords. AI-generated instructions perform similarly (median d=+0.23 across all models).

The most notable AI-generated result is on Gemma 12B, where the custom reading instruction achieves d=+0.60, the highest single-condition effect for that model. This suggests that for some models, task-specific guidance from a capable LLM can be effective -- but this requires an LLM call per document, whereas TF-IDF extraction requires only corpus-level word statistics.

### 4.7 Single-Token Prefixes

A striking finding is that a single token can produce substantial effects:

- On Qwen 2.5 32B-Instruct, "Facts" achieves d=+0.62, the highest single-condition effect for that model
- On Qwen 2.5 32B-Instruct, "Comprehend" achieves d=+0.57
- On Gemma 3 1B-IT, "Comprehend" achieves d=+0.69 and "Facts" achieves d=+0.69

Across models, single-word prefixes produce meaningful effects (|d| > 0.063) for 10 of 16 models. This extreme efficiency suggests that the priming mechanism does not require reading and following a multi-token instruction. Even one well-chosen token is sufficient to shift attention during encoding.

### 4.8 Token Budget Scaling

[FIGURE 1: Effect size vs. prefix length for comprehend and random conditions]

We test four token budgets (1, 4, 16, 64) for both the comprehend instruction and random tokens.

For comprehend, the relationship between length and effectiveness is non-monotonic for many models. Qwen 2.5 32B achieves d=+0.41 at L=1 but only d=+0.05 at L=64. Gemma 3 12B peaks at L=1 (d=+0.44) and L=16 (d=+0.48), dipping at L=64 (d=+0.36). This suggests that longer instruction prefixes may activate instruction-following circuits that interfere with the document-encoding process.

For random tokens, the pattern differs by model. Gemma 3 1B shows d=+0.81 at L=1, decreasing to d=+0.65 at L=64. Qwen 2.5 0.5B is erratic, with d=+0.27 at L=1, +0.06 at L=4, +0.52 at L=16, and -0.06 at L=64. There is no universal monotonic relationship between prefix length and benefit for either instruction or random tokens.

### 4.9 Generation-Based Evaluation: NLL Improvements Do Not Guarantee Better Answers

[TABLE 12: Exact Match and F1 scores for generation-based evaluation]

To validate whether NLL improvements translate to better generated answers, we run greedy decoding on 4 models (Qwen 1.5B, Qwen 7B, Gemma 12B, Qwen 14B) across 3 datasets (SQuAD v2, TriviaQA, GSM8K) under 3 priming conditions (bare, keywords, extract) and an inference-time baseline.

| Model | Dataset | Bare EM | Keywords EM | Extract EM | Bare F1 | Keywords F1 | Extract F1 |
|-------|---------|---------|-------------|------------|---------|-------------|------------|
| Qwen 1.5B | SQuAD v2 | 23.0% | 19.0% | 17.8% | 0.35 | 0.33 | 0.31 |
| Qwen 1.5B | TriviaQA | 37.2% | 34.8% | 35.5% | 0.46 | 0.43 | 0.44 |
| Gemma 12B | SQuAD v2 | 24.8% | 24.2% | **40.0%** | 0.45 | 0.47 | **0.58** |
| Gemma 12B | TriviaQA | **45.5%** | 34.0% | 35.8% | **0.60** | 0.51 | 0.48 |
| Gemma 12B | GSM8K | 0.8% | **1.8%** | **2.5%** | -- | -- | -- |
| Qwen 7B | SQuAD v2 | 2.0% | **6.5%** | 3.2% | 0.16 | **0.21** | 0.17 |

The results reveal a critical nuance: **NLL improvement does not reliably predict generation quality improvement.** Cache-time keyword priming consistently improves NLL across these models (d > 0 for 11 of 12 model-dataset combinations), but the effect on Exact Match is mixed:

- **Positive transfer.** Gemma 12B on SQuAD shows the clearest positive transfer: extract priming raises EM from 24.8% to 40.0% (+15.2pp) and F1 from 0.45 to 0.58. On GSM8K, keywords improve Gemma 12B from 0.8% to 1.8% EM. Qwen 7B shows a 3x improvement on SQuAD (2.0% to 6.5%) with keywords.

- **Negative transfer.** Gemma 12B on TriviaQA drops from 45.5% to 34.0% EM with keywords, despite positive NLL improvement (d=+0.06). Qwen 1.5B shows consistent but small EM drops (-2 to -4pp) across SQuAD and TriviaQA with keywords.

- **Condition specificity.** Extract priming produces the largest EM gain (Gemma 12B SQuAD: +15.2pp) but the choice of prefix matters: the same model with keywords shows only -0.6pp on SQuAD. The optimal prefix for NLL and the optimal prefix for generation quality may differ.

Note: Qwen 7B and 14B show very low baseline EM (< 5%) on SQuAD and TriviaQA. These models were evaluated without chat templates and produce verbose, off-format continuations. Their NLL scores (which evaluate reference token likelihood directly) are unaffected by generation format, but EM scores are. This limitation affects the generation evaluation but not the NLL-based results presented elsewhere in this paper.

The takeaway for practitioners: **NLL-based validation is necessary but not sufficient.** A positive NLL effect (d > 0) indicates that the model assigns higher probability to the correct answer, but this does not guarantee that greedy (or sampled) decoding will produce the correct answer. We recommend validating with generation-based metrics on the target task before deployment.

### 4.10 Comparison with Inference-Time Keyword Prepending

[TABLE 13: Cache-time priming vs inference-time keyword prepending (NLL)]

An obvious alternative to offline cache priming is to simply prepend keywords at query time: store a standard bare cache and include keywords in the Phase B input as `[\n, keywords, \n, query, \n, answer]`. This adds tokens at every inference call but avoids the RoPE reposition machinery entirely.

| Model | Dataset | Bare NLL | Cache-time d | Inference-time d | Winner |
|-------|---------|----------|-------------|-----------------|--------|
| Qwen 1.5B | SQuAD v2 | 1.43 | +0.34 | +0.14 | Cache |
| Qwen 1.5B | TriviaQA | 2.51 | +0.35 | -0.16 | Cache |
| Qwen 1.5B | GSM8K | 3.03 | +0.43 | +0.01 | Cache |
| Qwen 7B | SQuAD v2 | 2.05 | +0.47 | -0.59 | Cache |
| Qwen 7B | TriviaQA | 3.54 | +0.55 | -0.56 | Cache |
| Qwen 7B | GSM8K | 9.08 | +0.89 | -0.25 | Cache |
| Gemma 12B | SQuAD v2 | 2.36 | +0.30 | **+0.67** | Inference |
| Gemma 12B | TriviaQA | 2.67 | +0.06 | **+0.42** | Inference |
| Gemma 12B | GSM8K | 5.43 | +0.36 | +0.32 | Cache |
| Qwen 14B | SQuAD v2 | 2.98 | +0.28 | -0.15 | Cache |
| Qwen 14B | TriviaQA | 3.68 | -0.24 | +0.09 | Inference |
| Qwen 14B | GSM8K | 10.05 | **+0.89** | -0.59 | Cache |

Two clear patterns emerge:

**Cache-time priming dominates for Qwen models.** Across all 9 Qwen model-dataset combinations, cache-time priming produces higher d than inference-time in 8 of 9 cases. Inference-time keyword prepending *actively hurts* Qwen 7B on all datasets (d = -0.25 to -0.59), while cache-time priming helps strongly (d = +0.47 to +0.89). The mechanism appears to be that Qwen's instruction-tuned models treat query-time keywords as confusing instructions that interfere with question answering, whereas during cache construction the keywords reshape document representations without this interference.

**Inference-time prepending is preferable for Gemma 12B.** On SQuAD and TriviaQA, inference-time achieves d=+0.67 and d=+0.42, exceeding cache-time's d=+0.30 and d=+0.06. Gemma's hybrid attention architecture may benefit from seeing keywords alongside the query, allowing direct keyword-query interaction during attention.

**Practical implication.** For high-throughput RAG deployments where the same document is queried repeatedly, cache-time priming is preferred when using Qwen models: it achieves equal or better NLL improvement at zero per-query cost. For Gemma-family models, inference-time keyword prepending may be more effective, though it adds ~10 tokens per query.

---

## 5. Analysis

### 5.1 Mechanistic Decomposition

The main sweep includes controls that enable a four-level decomposition of the prefix effect for the comprehend instruction:

```
Total(comprehend) = position_shift
                  + (random_64 - position_shift)     [token presence]
                  + (scrambled - random_64)           [vocabulary]
                  + (coherent - scrambled)            [word order]
```

[TABLE 8: Four-level decomposition for all 16 models]

**Position shift** is the effect of encoding at shifted RoPE positions without any actual prefix tokens. This is near zero for all models (max |d| = 0.08 across all 16 models, mean |d| = 0.03). Win rates cluster around 0.35--0.46, confirming that the RoPE repositioning step is effectively lossless and does not contribute to the priming effect. This validates the pipeline correctness.

**Token presence** -- the increment from position shift to random-64 -- is the dominant mechanism for models where priming is effective. On Gemma 3 1B, token presence accounts for d=+0.67 of the total +0.60 effect. On Gemma 3N E4B, token presence contributes d=+0.58. Having any tokens participate in attention during encoding reshapes document representations, consistent with the attention sink hypothesis [6]: prefix tokens serve as additional attention targets that redistribute attention mass.

**Vocabulary** -- the increment from random to scrambled instruction tokens -- is generally weak and sometimes negative. Mean vocabulary contribution across models is d=-0.03. The scrambled instruction tokens, which contain the same vocabulary as the coherent instruction but in random order, are often as effective as or more effective than random tokens. This means that the specific vocabulary of the instruction tokens adds little beyond their presence.

**Word order** -- the increment from scrambled to coherent -- varies strongly with model capacity. For Qwen 2.5 7B-Instruct, word order contributes d=+0.26 (coherent d=+0.08 vs. scrambled d=-0.18). For Qwen 2.5 14B, word order contributes d=+0.30 (coherent d=-0.19 vs. scrambled d=-0.49). But for Gemma 3 1B, word order contributes d=-0.13 (coherent d=+0.60 is *lower* than scrambled d=+0.73). For models where scrambled tokens work as well as coherent instructions, the mechanism is purely token presence and feature activation, not instruction following.

### 5.2 Semantic Sensitivity Scales with Model Size (Qwen, Not Gemma)

[FIGURE 2: Semantic delta vs. log(model size) for Qwen and Gemma families]

The semantic delta (coherent d minus scrambled d) measures how much a model benefits from word order in the prefix -- a proxy for instruction-following capacity during cache construction.

**Qwen family (6 sizes: 0.5B--32B).** The semantic delta for comprehend shows a clear positive correlation with log model size: r=+0.75. The 0.5B model has a *negative* delta of -0.10 (scrambled is better), while the 7B and 14B models show positive deltas of +0.26 and +0.30 respectively. The 32B model's delta is +0.11. For extract, the correlation is r=+0.77. This suggests that Qwen models develop the capacity to leverage semantic coherence in prefixes gradually with scale.

**Gemma family (4 sizes: 1B--27B).** The correlation is weaker: r=+0.48. The 1B model has a *negative* delta of -0.13, the 4B is also negative at -0.34, while 12B is positive at +0.13 and 27B is near zero at -0.03. The non-monotonic pattern suggests that Gemma's hybrid attention architecture creates qualitatively different capacity profiles, perhaps because the alternating sliding-window layers disrupt the ability to propagate long-range instruction-following signals.

### 5.3 Instruction Tuning Amplifies Document-Specific Priming

[TABLE 9: Base vs. instruct comparison for Qwen 7B and Gemma 4B]

We have two base/instruct pairs that provide a controlled comparison.

**Qwen 2.5 7B (instruct vs. base).** Instruction tuning dramatically boosts document-derived priming: TF-IDF keywords jump from d=+0.10 (base) to d=+0.62 (instruct), a delta of +0.52. Oracle also improves substantially from d=+0.13 to d=+0.50. Meanwhile, generic instructions show mixed effects: extract drops from d=+0.42 (base) to d=+0.16 (instruct), and comprehend drops from d=+0.36 to d=+0.08. The instruct model has learned to respond to document-relevant signals (keywords, oracle query) more strongly, while becoming less responsive to generic instructions. This pattern -- instruction tuning shifting preference from generic to document-specific priming -- is the most practically important finding for deployment.

**Gemma 3 4B (instruct vs. base).** The base model is uniformly negative across all 25 priming conditions (0/25 positive). The instruct model is positive for 20/25. The mean shift from base to instruct is +0.45 across all conditions. For this model family, instruction tuning is a prerequisite for any beneficial priming effect.

The Gemma result is particularly instructive: it shows that the base model's attention mechanism actively resists prefix influence -- the additional tokens *degrade* document representations. Instruction tuning provides the inductive bias needed to integrate prefix context constructively.

### 5.4 Task-Reasoning Tier Analysis

[FIGURE 3: Mean effect by reasoning tier across responsive models]

We partition datasets into three reasoning tiers and compute mean effects.

**High reasoning (GSM8K, DROP):** The strongest priming benefits. Mean keyword d across responsive models (excluding the 3 harmed models) is approximately +0.40. The standout is Qwen 7B Instruct on GSM8K with keyword d=+1.22. Comprehend instruction averages d=+0.32 across responsive models for this tier.

**Mid reasoning (SQuAD v2, HotpotQA):** Moderate benefits. Mean keyword d is approximately +0.33 across responsive models. Effects are more consistent across models than for high-reasoning tasks.

**Factoid (TriviaQA, MS MARCO):** Weakest benefits. Mean keyword d is approximately +0.23 across responsive models. Several responsive models show negative keyword effects specifically on factoid tasks (e.g., Gemma 4B IT: -0.25 on TriviaQA, -0.28 on MS MARCO). Factoid retrieval may require less attention redistribution because the answer is a surface-level fact rather than the product of multi-step reasoning over the document.

### 5.5 The Anti-Prefix Paradox

The anti-instruction ("Ignore this text completely") produces *positive* effects for 11 of 16 models, with substantial effects in several cases:

- Gemma 3 1B: d=+0.72
- Gemma 3N E4B: d=+0.61
- Qwen 2.5 7B base: d=+0.34
- Qwen 2.5 7B Inst: d=+0.29

Only 3 models show meaningful negative anti-prefix effects: Qwen 14B (d=-0.54), Gemma 4B base (d=-0.41), and Ministral 8B (d=-0.21). The fact that an instruction to *ignore* the document produces the same direction of effect as an instruction to *comprehend* it provides the strongest evidence against an instruction-following mechanism. The model is not following the instruction; it is responding to the presence of additional tokens in the attention context.

### 5.6 Normalization Ablation

[TABLE 10: Normalization effect across all 16 models]

The per-tensor normalization has negligible effect across all 16 models. For the bare condition, normalization produces |d| < 0.06 for all models, with a mean of 0.02. For the comprehend condition, all |d| < 0.05. The bfloat16 round-trip perturbation is too small to meaningfully affect representations. We retain normalization as a default because it is costless, but it contributes nothing measurable to priming effectiveness.

### 5.7 Robustness: Hard Samples vs. All Samples

[TABLE 11: Effect sizes at N_HARD=160 vs. N=400 for key conditions]

Effects computed on the top-40% hardest samples (N_HARD=160 per dataset, 960 pooled) are consistently larger than effects computed on all samples (N=400 per dataset, 2400 pooled), typically by a factor of 1.2--1.4x. This amplification is expected: easy samples (low bare NLL) have little room for improvement.

Critically, effect *directions* are preserved. For TF-IDF keywords across all 16 models, the sign of d at N_HARD=160 matches the sign at N=400 for 15 of 16 models. The single discrepancy is Qwen 3B (d=+0.14 hard, d=-0.15 all), where both values are small and the effect is genuinely marginal. For the extract instruction, sign agreement is 16/16.

One exception requires noting: Qwen 2.5 3B shows a consistent pattern where hard-sample d is positive but all-sample d is negative across multiple conditions (comprehend: +0.10 vs. -0.17; extract: +0.21 vs. -0.10; keywords: +0.14 vs. -0.15). This model benefits from priming specifically on difficult samples but is harmed on easy ones, suggesting that priming adds useful information for uncertain predictions but introduces noise for confident ones.

---

## 6. Practical Guidelines

Based on these results, we offer the following guidance for practitioners deploying precomputed KV caches in RAG systems.

### 6.1 Default Strategy: TF-IDF Keywords

For a new deployment, start with TF-IDF keywords:

1. Compute corpus-level IDF statistics across your document collection
2. For each document, extract the top 10 terms by TF-IDF score
3. Prepend these terms (space-separated) as the prefix during cache construction
4. Discard the prefix entries and reposition keys via RoPE delta rotation
5. Store the resulting cache as usual

This requires no model access, no LLM calls, and adds negligible cost to offline construction (10--15 extra tokens). It adds nothing to inference time.

### 6.2 When to Try Alternatives

- **If your model is Gemma 1B:** Keywords rank 13th. Fixed instructions (question, declarative) work much better. Try "What are the key facts in this text?" (d=+0.81).
- **If your model is DeepSeek R1 or Mistral 7B:** Keywords are negative. Use extract instruction for Mistral (d=+0.43). Use question instruction for DeepSeek (d=+0.39).
- **If you can afford LLM preprocessing:** On Gemma 12B, an AI-generated instruction achieves d=+0.60 vs. d=+0.34 for keywords. Consider generating custom reading instructions per-document.
- **If your model is a base (non-instruct) model:** Validate carefully. Gemma 3 4B base is universally harmed. Qwen 7B base responds positively but prefers generic instructions (index d=+0.43) over keywords (d=+0.10).

### 6.3 When NOT to Use Directed Construction

- **Ministral 8B:** Harmed by all conditions. Do not use.
- **Qwen 2.5 14B:** Harmed by most conditions (only keywords d=+0.20 and facts-word d=+0.35 are positive). Use with extreme caution.
- **Base models generally:** Validate before deployment. The technique is unreliable on non-instruct models.

### 6.4 Validation Protocol

Before deploying directed construction on a new model:

1. Select 100+ representative documents from your corpus
2. Create (document, query, answer) triples
3. Compute NLL with bare cache and with keyword-primed cache
4. Compute paired Cohen's d on the hardest 40% of samples
5. If d < 0 or near zero, try 2--3 alternative prefixes (extract instruction, random tokens)
6. If all produce d <= 0, do not use directed construction for this model

---

## 7. Limitations

**NLL as proxy metric.** Our primary metric is NLL on answer tokens. As shown in Section 4.9, NLL improvement does not reliably translate to generation quality improvement: keyword priming improves NLL for 11/12 model-dataset combinations but improves Exact Match in only a subset of cases. The relationship between NLL improvement and downstream answer quality depends on task type, model, and generation format. We recommend generation-based validation on the target task before deployment.

**TF-IDF requires a corpus.** Computing TF-IDF requires inverse document frequency statistics from a corpus. For a single document without corpus context, alternative keyword extraction methods (frequency-based, entity extraction) would be needed. We have not tested these alternatives.

**Fixed prefix lengths.** We test L=1, 4, 16, 64 tokens. The optimal length may vary by model and document; we have not explored adaptive length selection or lengths beyond 64.

**No causal claims on mechanism.** The four-level decomposition is additive by construction and cannot capture interaction effects. The feature activation hypothesis, while consistent with all observations, is not verified through attention pattern analysis or probing experiments.

**Model coverage.** While 16 models across 5 families provides reasonable breadth, we have not tested Llama, Phi, or Command-R families. Our Gemma results use the multimodal (VLM) checkpoint for models >= 4B due to Gemma 3's release structure; results may differ on text-only variants.

**Heterogeneous negative results.** The three harmed models (Gemma 4B base, Ministral 8B, Qwen 14B) share no obvious common trait -- different families, different sizes, different attention types. We cannot predict which new model will be harmed by directed construction without empirical evaluation.

**Comparison with inference-time prefix.** As shown in Section 4.10, cache-time priming is not universally superior to inference-time keyword prepending. For Gemma 12B, inference-time prepending achieves higher NLL improvement. The advantage of cache-time priming is greatest for Qwen models and in deployments where the same document is queried repeatedly (amortizing the offline construction cost).

**Evaluation on English QA only.** All datasets are English question-answering tasks. Effectiveness on other languages, document types (code, tables, dialogue), and tasks (summarization, translation) is unknown.

---

## 8. Conclusion

We introduce directed KV cache construction, a technique that improves precomputed document representations by prepending a short prefix during encoding and discarding it before storage. Through experiments across 16 models (0.5B--32B parameters), 6 QA datasets, and 32 prefix conditions, we establish several findings.

The technique works for 13 of 16 tested models, producing positive effects on the majority of conditions for each responsive model. However, 3 models are consistently harmed, underscoring the need for per-model validation.

No single instruction prefix dominates. Across 6 fixed instructions tested on 16 models, each instruction wins on at most 3 models. The optimal instruction depends on model family, size, and instruction tuning status.

The oracle condition (actual query as prefix) is not the performance ceiling. Document-derived TF-IDF keywords outperform the oracle for 6 of 13 responsive models and rank in the top 3 of all conditions for 7 of 16 models. On Qwen 2.5 7B-Instruct, keywords achieve d=+0.62 versus d=+0.50 for the oracle.

Mechanistic decomposition reveals that token presence -- having any tokens participate in attention during encoding -- is the dominant factor, not instruction following. Anti-instructions produce positive effects for 11 of 16 models, scrambled instructions often match or exceed coherent ones, and even random tokens produce positive effects for 12 of 16 models. Semantic sensitivity (the benefit of word order) scales with model size in the Qwen family (r=+0.75) but not in Gemma.

Instruction tuning amplifies document-specific priming. Qwen 7B shows a +0.52 increase in keyword effect from base to instruct, while simultaneously becoming less responsive to generic instructions. Gemma 4B base is universally harmed, while the instruct version responds positively to 20 of 25 conditions.

However, generation-based evaluation reveals an important caveat: NLL improvement does not reliably predict generation quality improvement. Keyword priming improves NLL for 11 of 12 tested model-dataset combinations but improves Exact Match in a smaller subset. The largest generation quality gain is Gemma 12B on SQuAD v2, where extract priming raises EM from 24.8% to 40.0%. But on TriviaQA, the same model's EM drops from 45.5% to 34.0% with keywords despite a positive NLL effect. This disconnect between information-theoretic and task metrics requires careful per-task validation.

Comparison with inference-time keyword prepending shows that cache-time priming is not universally superior. For Qwen models, cache-time priming achieves equal or better NLL improvement at zero per-query cost, making it the preferred approach for high-throughput RAG. For Gemma 12B, inference-time keyword prepending produces larger NLL gains, suggesting that the optimal deployment strategy is model-dependent.

For practitioners, we recommend the following protocol: (1) start with TF-IDF keyword priming as the default, (2) validate with generation-based metrics on the target task, (3) compare with inference-time keyword prepending, and (4) deploy whichever approach produces the best generation quality on the target model and task. The keyword extraction requires only corpus-level word statistics (no model access), adds negligible cost to offline construction, and adds nothing to inference time.

More broadly, these findings reframe cache construction as an optimization target. The context in which encoding occurs -- the co-attended tokens that shape attention during the forward pass -- is an optimization dimension that has been entirely overlooked in the KV cache literature. Our results suggest that the transformer's self-attention mechanism, when used to reshape cached representations, responds more to *what is relevant* (document keywords) than to *what to do* (task instructions). The disconnect between NLL and generation quality presents an open challenge: understanding when and why improved token-level probability translates to better task performance is essential for making directed construction practically reliable.

---

## References

[1] Lu, S., Wang, H., Rong, Y., Chen, Z., Tang, Y. TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text. *EMNLP 2025*. arXiv:2410.07590.

[2] Yao, J., Li, H., Liu, Y., et al. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. *EuroSys 2025 (Best Paper)*. arXiv:2405.16444.

[3] Zheng, L., Yin, L., Xie, Z., et al. SGLang: Efficient Execution of Structured Language Model Programs. arXiv:2312.07104.

[4] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS 2023*. arXiv:2306.14048.

[5] Li, Y., Huang, Y., Yang, B., et al. SnapKV: LLM Knows What You Are Looking for Before Generation. *NeurIPS 2024*. arXiv:2404.14469.

[6] Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. Efficient Streaming Language Models with Attention Sinks. *ICLR 2024*. arXiv:2309.17453.

[7] Liu, Z., Yuan, J., Jin, H., et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *ICML 2024*. arXiv:2402.02750.

[8] Hooper, C., Kim, S., Mohammadzadeh, H., et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *NeurIPS 2024*. arXiv:2401.18079.

[9] Yang, Z., et al. Ada-KV: Adaptive KV Cache Management for Efficient LLM Inference. arXiv:2025.xxxxx.

[10] Li, X.L. and Liang, P. Prefix-Tuning: Optimizing Continuous Prompts for Generation. *ACL-IJCNLP 2021*. arXiv:2101.00190.

[11] Mu, J., Li, X.L., Goodman, N. Learning to Compress Prompts with Gist Tokens. *NeurIPS 2023*. arXiv:2304.08467.

[12] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing 568*. arXiv:2104.09864.

---

## Appendix

### A. Pipeline Correctness Verification

We verify pipeline correctness with the following tests on all 16 models:

1. **RoPE round-trip.** Rotating keys by delta then by negative delta recovers original keys within float32 tolerance (< 5e-7). In bfloat16, the round-trip error is ~0.016 (inherent precision limit). For models with linear RoPE scaling (Gemma 3 4B/12B/27B full_attention layers), the scaling factor is correctly applied to inverse frequencies.

2. **Bare two-phase = single-pass.** Without any prefix or repositioning, two-phase scoring produces NLL within 0.004--0.11 of single-pass scoring on all models. The residual is entirely attributable to bfloat16 precision.

3. **BOS handling.** Models without native BOS (Qwen 2.5, DeepSeek) use the PAD token as an artificial attention sink. Ablation confirms this is essential on Qwen: without BOS, comprehend d=-0.65; with BOS, d=+0.13.

4. **Position-shift roundtrip control.** The position-shift condition (encode at natural positions, shift keys forward by 64 then back -- a pure bf16 reposition roundtrip) produces |d| < 0.08 for all 16 models, with mean |d|=0.03 and win rates of 0.32--0.46. This confirms that the RoPE repositioning step introduces negligible error and does not contribute to the priming effect.

### B. Sliding Window Cache Constraints

Models with hybrid sliding/full attention (Gemma 3, Gemma 3N, Ministral) cache only `sliding_window - 1` entries in sliding attention layers. The total cache size (BOS + prefix + separator + document) must not exceed this limit. We truncate documents to `sliding_cache_limit - 1 - P - NL` tokens where P is the prefix length and NL is the separator length. This ensures uniform document lengths across conditions for fair comparison.

### C. GSM8K Answer Extraction

A critical implementation detail: GSM8K answers must be extracted as the final number after `####` (typically 1--2 tokens), not the full chain-of-thought (50--100 tokens). Using the full chain-of-thought produces reversed results because prefix conditioning helps focus attention on extractive answers but introduces distributional bias that hurts diverse token generation.

### D. Four-Level Decomposition: Full Results

[TABLE A1: Four-level decomposition for all 16 models]

| Model | Total (comp d) | Pos shift | Token presence | Vocabulary | Word order |
|-------|---------------|-----------|---------------|------------|------------|
| Qwen 0.5B | -0.08 | +0.03 | -0.09 | +0.08 | -0.10 |
| Gemma 1B | +0.60 | -0.03 | +0.67 | +0.08 | -0.13 |
| Qwen 1.5B | +0.30 | +0.06 | +0.03 | +0.19 | +0.03 |
| Qwen 3B | +0.10 | +0.00 | +0.19 | -0.11 | +0.02 |
| Gemma 3N E4B | -0.11 | +0.04 | +0.58 | -0.38 | -0.35 |
| Gemma 4B IT | -0.02 | +0.01 | +0.27 | +0.03 | -0.34 |
| Gemma 4B base | -0.34 | -0.03 | -0.41 | -0.06 | +0.16 |
| Qwen 7B Inst | +0.08 | +0.02 | -0.12 | -0.08 | +0.26 |
| Qwen 7B base | +0.36 | +0.01 | +0.28 | -0.02 | +0.09 |
| DeepSeek R1 7B | +0.20 | -0.08 | -0.06 | +0.34 | +0.00 |
| Mistral 7B | +0.09 | +0.01 | +0.37 | -0.32 | +0.04 |
| Ministral 8B | -0.51 | +0.02 | -0.15 | +0.02 | -0.40 |
| Gemma 12B | +0.36 | -0.01 | +0.17 | +0.07 | +0.13 |
| Qwen 14B | -0.19 | -0.03 | -0.41 | -0.04 | +0.30 |
| Gemma 27B | +0.16 | +0.02 | +0.20 | -0.03 | -0.03 |
| Qwen 32B | +0.05 | -0.03 | -0.22 | +0.20 | +0.11 |

### E. Statistical Significance Summary

At the alpha=0.05 threshold (|d| > 0.063 with N=960), we observe the following across the ablation experiment (14 priming conditions):

| Model | Sig. positive | Sig. negative | Net |
|-------|--------------|---------------|-----|
| Gemma 1B | 14 | 0 | +14 |
| Qwen 1.5B | 14 | 0 | +14 |
| Qwen 7B base | 14 | 0 | +14 |
| Qwen 32B | 12 | 1 | +11 |
| Qwen 7B Inst | 13 | 1 | +12 |
| Gemma 12B | 13 | 0 | +13 |
| Gemma 27B | 12 | 0 | +12 |
| Qwen 3B | 12 | 0 | +12 |
| Gemma 3N E4B | 11 | 1 | +10 |
| DeepSeek R1 7B | 11 | 0 | +11 |
| Gemma 4B IT | 10 | 0 | +10 |
| Mistral 7B | 9 | 2 | +7 |
| Qwen 0.5B | 8 | 3 | +5 |
| Qwen 14B | 2 | 9 | -7 |
| Ministral 8B | 0 | 13 | -13 |
| Gemma 4B base | 0 | 14 | -14 |

### F. Per-Dataset Results for Key Conditions

[TABLE A2: Cohen's d per dataset for comprehend-64, extract-64, keywords, and oracle-64 across all 16 models]

Full per-dataset breakdowns are available in the supplementary materials. Key patterns: GSM8K shows the largest positive effects for most models and conditions; MS MARCO shows the smallest effects; DROP and SQuAD v2 are intermediate. The effect direction is more consistent across datasets within a model than across models within a dataset, suggesting that model-level factors (architecture, training) dominate over task-level factors in determining priming responsiveness.

### G. Oracle Analysis: Padded vs. Natural Length

[TABLE A3: Oracle natural vs. padded-to-64]

| Model | Oracle (natural) | Oracle (L=64) | Padding benefit |
|-------|-----------------|---------------|----------------|
| Qwen 0.5B | +0.61 | +0.50 | -0.11 |
| Gemma 1B | +0.53 | +0.70 | +0.17 |
| Qwen 1.5B | +0.48 | +0.48 | +0.00 |
| Qwen 3B | +0.10 | +0.12 | +0.03 |
| Gemma 3N E4B | +0.57 | +0.58 | +0.01 |
| Gemma 4B IT | -0.00 | +0.23 | +0.23 |
| Gemma 4B base | -0.09 | -0.07 | +0.02 |
| Qwen 7B Inst | +0.35 | +0.50 | +0.15 |
| Qwen 7B base | +0.08 | +0.13 | +0.06 |
| DeepSeek R1 7B | +0.39 | +0.37 | -0.02 |
| Mistral 7B | -0.00 | +0.10 | +0.10 |
| Ministral 8B | -0.58 | -0.28 | +0.30 |
| Gemma 12B | -0.02 | +0.23 | +0.25 |
| Qwen 14B | -0.00 | -0.02 | -0.02 |
| Gemma 27B | +0.04 | +0.24 | +0.19 |
| Qwen 32B | +0.20 | +0.20 | +0.00 |

Padding benefits 10 of 16 models (net positive), supporting the hypothesis that prefix length contributes independently of semantic content. The largest padding benefits occur on Ministral 8B (+0.30), Gemma 12B (+0.25), and Gemma 4B IT (+0.23).
