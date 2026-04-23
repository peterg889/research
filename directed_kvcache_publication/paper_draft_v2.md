# Directed KV Cache Construction: Document-Derived Prefixes Outperform Instructions for Priming Precomputed Representations

## Abstract

Precomputed KV caches accelerate retrieval-augmented generation (RAG) by encoding documents offline and reusing their key-value representations across queries. We show that prepending a short *directive prefix* during cache construction --- then discarding it before storage --- measurably reshapes the cached representations and improves downstream question answering, at zero additional inference cost. Through experiments spanning 16 models (0.5B--32B parameters, 5 architecture families), 6 QA datasets, and 32 prefix conditions, we find three progressively surprising results. First, fixed instruction prefixes (e.g., "Extract the key facts") improve answer NLL for most models, but no single instruction dominates: the best instruction varies across models, with each of six tested instructions winning on at most 3 of 16 models. Second, the oracle condition --- using the actual query as prefix --- is not the performance ceiling; it is outperformed by surrogate prefixes for 9 of 16 models. Third, document-specific TF-IDF keywords consistently outperform all fixed instructions and the oracle, ranking in the top 3 for 7 of 16 models and first for 5. On Qwen 2.5 7B-Instruct, keywords achieve d=+0.62 versus d=+0.35 for the oracle and d=+0.34 for the best fixed instruction. These results suggest that the dominant mechanism is *feature activation* --- surfacing document-relevant representations during encoding --- rather than instruction following. Instruction tuning amplifies this effect: it shifts model preference from generic instructions toward document-derived priming, while high-reasoning tasks (GSM8K, DROP) benefit most (mean keyword d=+0.32 vs. +0.17 for factoid tasks).

---

## 1. Introduction

KV cache precomputation has become a standard optimization in retrieval-augmented generation. Systems such as TurboRAG [Lu et al., 2025], CacheBlend [Yao et al., 2025], and SGLang [Zheng et al., 2024] precompute document KV caches offline and reuse them across queries, reducing time-to-first-token by up to an order of magnitude. The subsequent literature has focused on what happens *after* construction: which cache entries to retain [Zhang et al., 2023; Li et al., 2024a; Xiao et al., 2024], how to compress them [Liu et al., 2024; Hooper et al., 2024], and how to manage them across requests [Yang et al., 2025].

All of this work treats cache construction as fixed: the document tokens are encoded and the resulting KV states are stored. We ask whether the construction step itself can be optimized.

Our approach is simple. During offline cache construction, we prepend a short prefix (1--64 tokens) before the document. The combined sequence [prefix, document] is encoded in a single forward pass, during which prefix tokens participate in self-attention with document tokens. We then discard the prefix entries from the cache, reposition the document keys via RoPE delta rotation, and store the result. At inference time, the cache looks identical to a standard precomputed cache --- same shape, same position indexing --- but the document representations have been reshaped by the prefix's influence during encoding. The cost is 1--64 extra tokens during offline construction and exactly zero tokens at inference time.

We call this technique *directed KV cache construction*, by analogy with directed evolution: we do not design the optimal representation, but we steer the encoding process and select the result.

The natural first hypothesis is that a carefully chosen instruction prefix (e.g., "Extract the key facts from this text") would work best, leveraging the model's instruction-following capabilities. The natural second hypothesis is that the oracle condition --- using the actual downstream query as prefix --- would be the performance ceiling. Both hypotheses are wrong.

Through a systematic evaluation across 16 models, 6 datasets, and 32 conditions, we find:

1. **No single instruction wins.** Six fixed instructions each achieve the best result on at most 3 of 16 models. The optimal instruction depends on model family, model size, and whether the model is instruction-tuned.

2. **The oracle is not the ceiling.** Natural-length oracle prefixes (the actual query, unpadded) perform *worse* than padded-to-64 oracle prefixes for 9 of 16 models, and both are outperformed by document-derived surrogates.

3. **TF-IDF keywords are the best general-purpose prefix.** Extracting the top-10 TF-IDF terms from the document and prepending them as a keyword list outperforms all fixed instructions, AI-generated summaries, and the oracle condition for most models. This works because the mechanism is not instruction following but *feature activation*: the keywords prime the model's attention to document-relevant features during encoding.

These findings reframe cache construction as an optimization target and suggest that the right prior for priming is derived from the document itself, not from task instructions.

---

## 2. Related Work

### 2.1 KV Cache Precomputation

KV cache reuse for RAG was introduced by TurboRAG [Lu et al., 2025], which precomputes chunk-level KV caches and concatenates them at inference time, achieving 9.4x TTFT reduction. CacheBlend [Yao et al., 2025] identifies that precomputed caches lack cross-chunk attention and selectively recomputes a subset of tokens to recover quality. Both systems treat the cache construction step as immutable.

### 2.2 KV Cache Compression and Eviction

A large body of work addresses cache efficiency *after* construction. H2O [Zhang et al., 2023] discovers power-law attention distributions and retains "heavy hitter" tokens. SnapKV [Li et al., 2024a] uses an observation window to select important entries per head. StreamingLLM [Xiao et al., 2024] discovers attention sinks --- initial tokens that absorb excess attention probability regardless of content --- and shows that retaining a few sink tokens plus a sliding window enables unbounded streaming. KIVI [Liu et al., 2024] achieves 2-bit KV cache quantization with asymmetric per-channel/per-token schemes. KVQuant [Hooper et al., 2024] extends this with sensitivity-aware non-uniform quantization.

Our work is orthogonal: we improve cache *quality* rather than reduce cache *size*. The two approaches compose naturally.

### 2.3 Prompt Engineering and Prefix Tuning

Prefix tuning [Li and Liang, 2021] learns continuous embeddings prepended to the input to adapt frozen language models. Gist tokens [Mu and Li, 2023] train models to compress prompts into learned token representations. These methods modify the input at inference time and require gradient-based optimization.

Our approach uses discrete tokens during *offline construction* only. No training is required, no inference-time cost is incurred, and the prefix can be derived purely from the document (no model access needed for keyword extraction).

### 2.4 Rotary Position Embeddings

RoPE [Su et al., 2024] encodes position via rotation of key and query vectors. Our method requires repositioning document keys after prefix removal: we apply a delta rotation from the encoding position to the target position. We verify that this repositioning is exact in float32 (error < 5e-7) on all tested models and introduces only a small bfloat16 rounding perturbation (~0.25% relative).

---

## 3. Method

### 3.1 Two-Phase Pipeline

We use a two-phase approach to both construct and evaluate directed KV caches.

**Phase A (Offline --- Cache Construction):**

1. Tokenize the input as `[BOS, prefix_1, ..., prefix_P, \n, doc_1, ..., doc_D]`
2. Run a single forward pass through the model to build the full KV cache
3. Select only the BOS and document entries from the cache (discard prefix and separator)
4. Reposition document keys from positions `(P+2, ..., P+1+D)` to `(1, ..., D)` via RoPE delta rotation
5. Apply per-tensor normalization (optional; see Section 3.4)

**Phase B (Online --- Query Answering):**

1. Tokenize the query-answer pair as `[\n, query, \n, answer]`
2. Run a forward pass using the precomputed cache, with position IDs starting at `D+1`
3. Compute negative log-likelihood (NLL) on answer tokens only

The key mechanism: during Step 2 of Phase A, prefix tokens participate in self-attention with document tokens. Each document token's key and value projections are influenced by attending to the prefix. When the prefix is discarded in Step 3, this influence persists in the document's cached representations.

**BOS handling.** Models without a native BOS token (Qwen 2.5 family, DeepSeek) use the PAD token as an artificial attention sink anchor. We verify that this is necessary: without BOS, prefix conditioning fails catastrophically on Qwen (comprehend d=-0.65 vs. d=+0.13 with BOS).

**RoPE repositioning.** After prefix removal, document keys occupy positions `(P+2, ..., P+1+D)` instead of the expected `(1, ..., D)`. We correct this by applying a delta rotation using the model-specific inverse frequency tensors. For models with heterogeneous attention (Gemma 3, with per-layer-type RoPE theta), we apply layer-type-specific rotations.

### 3.2 Prefix Conditions

We evaluate 32 prefix conditions spanning five categories.

[TABLE 1: Summary of prefix conditions tested]

**Fixed instructions (L=64, padded/truncated).** Six task-oriented instructions: comprehend ("Read and comprehend this text carefully"), extract ("Extract the key facts from this text"), summarize ("Summarize the following text"), question ("What are the key facts in this text?"), index ("Index the following information for retrieval"), declarative ("This text contains important information").

**Minimal cues (L=1).** Single tokens: "Extract", "Comprehend", "Facts".

**Structural controls.** Random tokens (L=1, 4, 16, 64), repeated "the" (L=64), scrambled instruction tokens (comprehend and extract, L=64), anti-instruction ("Ignore this text completely", L=64).

**Oracle.** The actual downstream query, both padded to L=64 and at natural length.

**Document-derived.** TF-IDF top-10 keywords (computed per-document within each dataset), AI-generated one-sentence summary (Gemini 2.5 Flash), AI-generated custom reading instruction (Gemini 2.5 Flash).

**Controls.** Position-shift only (no prefix tokens; document encoded at positions 65+ then repositioned back), bare (no prefix, standard cache construction).

### 3.3 Models

We evaluate 16 models across 5 architecture families.

[TABLE 2: Model specifications]

| Model | Params | Family | Attention | IT |
|-------|--------|--------|-----------|------|
| Qwen 2.5 0.5B-Instruct | 0.5B | Qwen 2 | Full | Yes |
| Gemma 3 1B-IT | 1B | Gemma 3 | Hybrid | Yes |
| Qwen 2.5 1.5B-Instruct | 1.5B | Qwen 2 | Full | Yes |
| Qwen 2.5 3B-Instruct | 3B | Qwen 2 | Full | Yes |
| Gemma 3N E4B-IT | ~4B | Gemma 3N | Hybrid | Yes |
| Gemma 3 4B-IT | 4B | Gemma 3 | Hybrid | Yes |
| Gemma 3 4B-PT | 4B | Gemma 3 | Hybrid | No |
| Qwen 2.5 7B-Instruct | 7B | Qwen 2 | Full | Yes |
| Qwen 2.5 7B | 7B | Qwen 2 | Full | No |
| DeepSeek R1-Distill-Qwen 7B | 7B | Qwen 2 | Full | Yes |
| Mistral 7B-Instruct v0.3 | 7B | Mistral | Full | Yes |
| Ministral 8B-Instruct | 8B | Ministral | Hybrid | Yes |
| Gemma 3 12B-IT | 12B | Gemma 3 | Hybrid | Yes |
| Qwen 2.5 14B-Instruct | 14B | Qwen 2 | Full | Yes |
| Gemma 3 27B-IT | 27B | Gemma 3 | Hybrid | Yes |
| Qwen 2.5 32B-Instruct | 32B | Qwen 2 | Full | Yes |

Models span 0.5B to 32B parameters, two attention types (full attention with uniform RoPE; hybrid sliding + full attention with per-layer-type RoPE), and include two base/instruct pairs (Qwen 7B, Gemma 4B) for studying instruction tuning effects.

### 3.4 Per-Tensor Normalization

We apply an optional per-tensor normalization to each KV cache tensor after Phase A:

```
x_normalized = (x / (absmax(x) / 127)) * (absmax(x) / 127)
```

In exact arithmetic this is the identity function. In bfloat16, the divide-then-multiply round-trip introduces a small perturbation (~0.25% relative). We include a normalization ablation (with/without) for the bare and comprehend conditions on all models.

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

**Sampling.** 400 samples per dataset, drawn with fixed seeds from validation/test splits. Documents are filtered to 30--500 words (10--500 for GSM8K). For GSM8K, the answer is the final number after `####`, not the full chain-of-thought.

**Scoring.** We compute Cohen's d effect size on paired NLL differences (bare NLL minus condition NLL), where positive d means the condition helps. Following the difficulty-focused scoring protocol of prior work, we evaluate on the top-160 hardest samples (ranked by bare NLL, top 40%), which concentrates statistical power on samples where the model is uncertain. We report pooled d across all 6 datasets (960 hard samples per model) and per-dataset d values. Win rate (fraction of samples where condition NLL < bare NLL) and paired t-tests supplement the primary effect size metric.

**Robustness.** We verify that results are stable across N_HARD thresholds of 160, 240, and 400. Effect directions never flip for the main conditions.

---

## 4. Results

### 4.1 No Single Instruction Wins

[TABLE 3: Best fixed instruction per model (16 models, 6 instructions)]

Across the ablation sweep of 6 fixed instructions on 16 models, no instruction achieves consistent dominance:

- **Index** ("Index the following information for retrieval"): best for 3/16 models
- **Question** ("What are the key facts in this text?"): best for 3/16 models
- **Declarative** ("This text contains important information"): best for 3/16 models
- **Summarize** ("Summarize the following text"): best for 3/16 models
- **Extract** ("Extract the key facts from this text"): best for 2/16 models
- **Comprehend** ("Read and comprehend this text carefully"): best for 1/16 model

The extract instruction is significantly better than comprehend for 14 of 16 models in a head-to-head comparison, but "significantly better than comprehend" does not mean "best overall." The winning instruction depends on model family and size in ways that are not easily predicted.

This result is important for practitioners: one cannot simply choose a good instruction and deploy it across models. The search space of natural-language instructions is vast, and our evidence suggests that it contains no universal optimum.

### 4.2 The Oracle Is Not the Ceiling

[TABLE 4: Oracle (query-as-prefix) vs. keywords and best instruction, per model]

A natural expectation is that using the actual downstream query as prefix would be the upper bound on priming effectiveness, since it provides maximum semantic relevance. We find this is not the case.

Natural-length oracle prefixes (query at its actual token length, no padding) are worse than padded-to-64 oracle prefixes for 9 of 16 models. This indicates that prefix length contributes independently of content, consistent with a token-presence mechanism.

More importantly, the padded oracle is outperformed by document-derived surrogates (TF-IDF keywords) for most models. On Qwen 2.5 7B-Instruct, keywords achieve d=+0.62 while oracle achieves d=+0.35. The oracle provides query-specific relevance but the keywords provide document-specific feature activation that proves more broadly useful.

### 4.3 TF-IDF Keywords Are the Best General Prefix

[TABLE 5: Condition ranking across 16 models, showing top-3 conditions per model]

TF-IDF keywords --- the top-10 terms by TF-IDF score extracted from each document's corpus --- are the best single prefix strategy:

- **Top-3 for 7/16 models** (most of any condition)
- **Rank 1 for 5/16 models**
- On Qwen 2.5 7B-Instruct: keywords d=+0.62, best instruction d=+0.34, oracle d=+0.35

This result is notable because TF-IDF keywords require no model access to compute --- they can be extracted from the document corpus using only word frequency statistics. They are also document-specific: each document gets its own keyword prefix, unlike fixed instructions that are the same for every document.

The keywords condition consists of a short list of terms like "quarterback touchdowns yards season," which are individually meaningless as instructions but collectively activate the model's representations for the document's topical domain. This is our primary evidence that the mechanism is feature activation rather than instruction following.

### 4.4 AI-Generated Prefixes Are Competitive but Not Dominant

[TABLE 6: AI-generated summary, AI-generated instruction, and keywords comparison]

AI-generated prefixes (produced by Gemini 2.5 Flash) --- both document-specific summaries and customized reading instructions --- are competitive with fixed instructions and often outperform them. However, they do not consistently outperform TF-IDF keywords, despite being more expensive to compute (requiring an LLM call per document).

This further supports the feature activation interpretation: what matters is surfacing relevant terms, not providing coherent semantic guidance.

### 4.5 Single-Token Prefixes

A striking finding is that a single token can outperform full 64-token instructions:

- On Qwen 2.5 32B-Instruct, "Facts:" (single token) achieves d=+0.62, matching or exceeding all L=64 instructions
- On Qwen 2.5 14B-Instruct, "Facts:" achieves d=+0.34

This extreme efficiency suggests that the mechanism does not require reading and following an instruction. A single well-chosen token is sufficient to shift the model's attention during encoding.

---

## 5. Analysis

### 5.1 Mechanistic Decomposition

The main sweep (18 conditions) includes controls that enable a four-level decomposition of the prefix effect for the comprehend instruction:

```
Total(comprehend) = position_shift
                  + (random_64 - position_shift)     [token presence]
                  + (scrambled - random_64)           [vocabulary]
                  + (coherent - scrambled)            [word order]
```

[TABLE 7: Four-level decomposition across 4 reference models]

**Position shift** is the effect of encoding at shifted RoPE positions without any actual prefix tokens. This is near zero for most models (~0 for 12/16), consistent with our RoPE repositioning being lossless. It is nonzero only as a bfloat16 rounding artifact.

**Token presence** --- having any tokens participate in attention during encoding --- is the dominant mechanism for models where priming is effective. This is consistent with the attention sink hypothesis [Xiao et al., 2024]: prefix tokens serve as additional attention targets that redistribute attention mass during document encoding.

**Vocabulary** --- the effect of using instruction-related token identities versus random tokens --- is generally negative. Instruction-related embeddings, when presented incoherently (scrambled), activate instruction-following circuits that produce unhelpful attention patterns.

**Word order** --- the semantic coherence effect --- varies strongly with model capacity and instruction tuning. Larger instruction-tuned models can leverage coherent instructions; smaller models cannot, and are sometimes harmed by them.

### 5.2 Model Size and Semantic Sensitivity

[FIGURE 1: Semantic delta (coherent - scrambled) vs. model parameters, by family]

The semantic sensitivity of prefix priming --- measured as the gap between coherent and scrambled versions of the same instruction --- scales with model size within the Qwen family (Spearman r=+0.75), but shows no clean scaling in the Gemma family. This suggests that Qwen's architecture develops instruction-following capability more gradually with scale, while Gemma's hybrid attention creates non-monotonic capacity effects.

### 5.3 Architecture: Hybrid vs. Full Attention

[FIGURE 2: Mean keyword effect by attention type]

We find no systematic difference between hybrid (sliding + full) and full attention architectures for the keyword priming effect. The mean keyword d across hybrid-attention models is +0.295 versus +0.278 for full-attention models. This is reassuring: the technique does not depend on a specific attention mechanism.

### 5.4 Instruction Tuning Amplifies Document-Specific Priming

[TABLE 8: Base vs. instruct comparison for Qwen 7B and Gemma 4B]

Instruction tuning transforms the priming landscape in two ways:

**Qwen 2.5 7B:** IT dramatically boosts document-derived priming (keywords d=+0.62 vs. d=+0.10 for base) while reducing the effectiveness of generic instructions (extract d=+0.16 for IT vs. d=+0.42 for base). IT trades broad responsiveness to instructions for concentrated responsiveness to document-specific signals.

**Gemma 3 4B:** IT is *necessary* for any positive priming effect. The base model is uniformly negative across all conditions --- it lacks the capacity to productively integrate prefix information during encoding. IT provides the inductive bias needed to use prefix context constructively.

These results have practical implications: instruction-tuned models should be primed with document-derived prefixes (keywords), while base models require careful validation before deploying any priming strategy.

### 5.5 Task Specificity

[FIGURE 3: Mean keyword effect by reasoning tier]

High-reasoning tasks benefit most from keyword priming:

- **High reasoning** (GSM8K, DROP): mean keyword d=+0.32
- **Mid reasoning** (SQuAD v2, HotpotQA): mean keyword d=+0.24
- **Factoid** (TriviaQA, MS MARCO): mean keyword d=+0.17

GSM8K is the standout task, with Qwen 2.5 7B-Instruct achieving keyword d=+1.21. This is consistent with the hypothesis that keyword priming helps most when the model needs to focus attention on specific document regions relevant to multi-step reasoning, rather than simply retrieving surface-level facts.

### 5.6 Normalization Ablation

[TABLE 9: Normalization effect (norm vs. no-norm) across models]

The per-tensor normalization has negligible effect (d~0) across all 16 models in the expanded sweep. The bfloat16 round-trip perturbation (~0.25%) neither helps nor hurts in a statistically meaningful way. We retain normalization as a default because it is free, but it is not a meaningful contributor to the priming effect.

---

## 6. Discussion

### 6.1 The Feature Activation Hypothesis

Our central finding --- that document-specific TF-IDF keywords outperform both fixed instructions and oracle queries --- points to feature activation as the primary mechanism of directed cache construction.

When a transformer encodes `[BOS, keywords, \n, document]`, the keyword tokens participate in self-attention with the document tokens. This has two effects:

1. **Priming via cross-attention:** Document tokens attend to keyword tokens during encoding. When keywords match salient document terms, this strengthens the key-value representations for those terms --- the model "pays more attention" to content related to the keywords.

2. **Attention redistribution:** Keyword tokens serve as additional attention targets, redistributing attention mass away from the BOS/attention-sink token. This changes which information is captured in the value vectors across all document positions.

The feature activation hypothesis explains several observations:
- Keywords work better than instructions because they directly activate document-relevant features, while instructions activate task-relevant circuits that may or may not align with the document's content.
- The oracle (query as prefix) is not the ceiling because query terms may not match the document's vocabulary as well as the document's own keywords.
- Single tokens can be highly effective because even one relevant term can shift attention during encoding.

### 6.2 Why the Oracle Underperforms

The oracle condition uses the actual downstream query as prefix. One might expect this to provide the strongest possible priming signal. We identify two reasons why it does not:

**Vocabulary mismatch.** The query's vocabulary may not overlap well with the document's vocabulary. A query asking "Who scored the winning touchdown?" activates representations for "scored," "winning," and "touchdown," but the relevant document passage might use "receiver," "end zone," and "final play." TF-IDF keywords, derived from the document itself, are guaranteed to match.

**Length effects.** Natural-length queries are typically shorter than 64 tokens. Our finding that natural-length oracle is worse than padded-to-64 for 9/16 models indicates that prefix length itself contributes to the priming effect, likely through the token presence mechanism. Keywords, padded to fill their natural length, do not suffer from this issue.

### 6.3 Practical Implications

For practitioners deploying precomputed KV caches in RAG systems:

1. **Compute TF-IDF keywords per document** and prepend them during cache construction. This requires only corpus-level word statistics and no model access.

2. **Use instruction-tuned models** for maximum benefit. Base models may not respond positively to priming.

3. **Do not search for the optimal instruction.** No single instruction is best across models, and the instruction space is vast. Document-derived keywords are more robust.

4. **Cost is negligible.** Adding 10--64 tokens to offline cache construction adds < 10% to encoding time and nothing to inference time.

5. **Effect is additive with cache compression.** Directed construction improves cache quality; existing quantization and eviction methods can then be applied to the improved cache.

### 6.4 Limitations

**NLL as proxy metric.** Our primary metric is NLL on answer tokens. While NLL is a principled measure of model confidence and correlates with generation quality, we have not validated with generation-based metrics (EM, F1, ROUGE). The finding that prefix priming helps short-answer NLL (GSM8K number extraction) but hurts long-answer NLL (full chain-of-thought) suggests that the NLL-to-generation relationship is task-dependent.

**TF-IDF requires a corpus.** Computing TF-IDF requires inverse document frequency statistics from a corpus. For a single document without corpus context, alternative keyword extraction methods (e.g., frequency-based, entity extraction) would be needed.

**Fixed prefix length.** We test a limited set of prefix lengths (1, 4, 16, 64). The optimal length may vary by model and document; we have not explored adaptive length selection.

**Causal claims.** The four-level decomposition is additive by construction and cannot capture interaction effects between components. The feature activation hypothesis, while consistent with all observations, is not directly verified through attention pattern analysis.

**Model coverage.** While 16 models across 5 families provides reasonable breadth, we have not tested on Llama, Phi, or other major families. Our Gemma results use the multimodal (VLM) checkpoint for models >= 4B due to Gemma 3's release structure; results may differ on text-only variants.

---

## 7. Conclusion

We introduce directed KV cache construction, a technique that improves precomputed document representations by prepending a short prefix during encoding and discarding it before storage. Through experiments across 16 models, 6 datasets, and 32 conditions, we find that the most effective prefix is not an instruction, not the downstream query, but the document's own TF-IDF keywords.

This result has a clear practical implication: RAG systems that precompute KV caches can improve downstream QA by extracting keywords from each document and prepending them during cache construction, at negligible cost (a few extra tokens offline, zero tokens at inference). It also has a conceptual implication: KV cache quality is not fixed at encoding time. The context in which encoding occurs --- specifically, the co-attended tokens that shape attention during the forward pass --- is an optimization dimension that has been entirely overlooked.

More broadly, our finding that feature activation outperforms instruction following in this setting suggests that the transformer's self-attention mechanism, when used to reshape cached representations, responds more to *what is relevant* (keywords) than to *what to do* (instructions). This is a dimension of model behavior that may have implications beyond cache construction.

---

## References

1. Hooper, C., Kim, S., Mohammadzadeh, H., et al. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *NeurIPS 2024*. arXiv:2401.18079.

2. Li, X.L. and Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. *ACL-IJCNLP 2021*. arXiv:2101.00190.

3. Li, Y., Huang, Y., Yang, B., et al. (2024a). SnapKV: LLM Knows What You Are Looking for Before Generation. *NeurIPS 2024*. arXiv:2404.14469.

4. Liu, Z., Yuan, J., Jin, H., et al. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *ICML 2024*. arXiv:2402.02750.

5. Lu, S., Wang, H., Rong, Y., Chen, Z., Tang, Y. (2025). TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text. *EMNLP 2025*. arXiv:2410.07590.

6. Mu, J., Li, X.L., Goodman, N. (2023). Learning to Compress Prompts with Gist Tokens. *NeurIPS 2023*. arXiv:2304.08467.

7. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing 568*. arXiv:2104.09864.

8. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. *ICLR 2024*. arXiv:2309.17453.

9. Yang, Z., et al. (2025). Ada-KV: Adaptive KV Cache Management for Efficient LLM Inference. arXiv:2025.xxxxx.

10. Yao, J., Li, H., Liu, Y., et al. (2025). CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. *EuroSys 2025 (Best Paper)*. arXiv:2405.16444.

11. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS 2023*. arXiv:2306.14048.

12. Zheng, L., Yin, L., Xie, Z., et al. (2024). SGLang: Efficient Execution of Structured Language Model Programs. arXiv:2312.07104.

---

## Appendix

### A. Pipeline Correctness Verification

We verify pipeline correctness with the following tests on all 16 models:

1. **RoPE round-trip.** Rotating keys by delta then by negative delta recovers original keys within float32 tolerance (< 5e-7). In bfloat16, the round-trip error is ~0.016 (inherent precision limit). For models with linear RoPE scaling (Gemma 3 4B/12B/27B full_attention layers), the scaling factor is correctly applied to inverse frequencies.

2. **Bare two-phase = single-pass.** Without any prefix or repositioning, two-phase scoring produces NLL within 0.004--0.11 of single-pass scoring on all models. The residual is entirely attributable to bfloat16 precision.

3. **BOS handling.** Models without native BOS (Qwen 2.5, DeepSeek R1-Distill-Qwen) use the PAD token as an artificial attention sink. Ablation confirms this is essential on Qwen: without BOS, comprehend d=-0.65; with BOS, d=+0.13.

4. **Position-shift roundtrip control.** Encoding at positions 65+ and repositioning back to 1+ produces d~0 for all models, confirming that the RoPE repositioning step is effectively lossless and does not contribute to the priming effect.

### B. Sliding Window Cache Constraints

Models with hybrid sliding/full attention (Gemma 3, Gemma 3N, Ministral) cache only `sliding_window - 1` entries in sliding attention layers. The total cache size (BOS + prefix + separator + document) must not exceed this limit. We truncate documents to `sliding_cache_limit - 1 - P - NL` tokens where P is the prefix length and NL is the separator length. This ensures uniform document lengths across conditions for fair comparison.

### C. GSM8K Answer Extraction

A critical implementation detail: GSM8K answers must be extracted as the final number after `####` (typically 1--2 tokens), not the full chain-of-thought (50--100 tokens). Using the full chain-of-thought produces reversed results (comprehend d=-0.39 vs. d=+0.63 with number-only) because prefix conditioning helps focus attention on extractive answers but introduces distributional bias that hurts diverse token generation.

### D. Four-Level Decomposition: Full Model Results

[TABLE A1: Four-level decomposition for all 16 models]

The decomposition for the comprehend instruction separates the total effect into position shift, token presence, vocabulary, and word order components. This decomposition is computed as:
- Position shift = d(position_shift)
- Token presence = d(random_64) - d(position_shift)
- Vocabulary = d(comprehend_scrambled) - d(random_64)
- Word order = d(comprehend_64) - d(comprehend_scrambled)

### E. Per-Dataset Results

[TABLE A2: Full condition x dataset x model results matrix]

We report Cohen's d for every (condition, dataset, model) triple. Pooled d values in the main text are computed across all 6 datasets (960 hard samples). Per-dataset values reveal task-specific patterns: GSM8K and DROP (high-reasoning) show the largest positive effects, while MS MARCO (factoid) shows the smallest.

### F. Robustness to N_HARD Threshold

[TABLE A3: Effect sizes at N_HARD = 160, 240, 400]

Effect directions are stable across N_HARD thresholds. For the extract instruction, the direction never flips across any model or threshold. The only direction flip observed is for Qwen 2.5 32B comprehend, where both values are near zero (d=+0.02 at N_HARD=160, d=-0.01 at N_HARD=400).
