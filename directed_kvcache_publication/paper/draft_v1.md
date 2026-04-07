# Directed KV Cache Construction: How Prefix Tokens Reshape Document Representations in Large Language Models

## Abstract

Precomputed KV caches are widely used to accelerate retrieval-augmented generation
(RAG), yet the standard approach — caching raw document tokens — ignores the
possibility that cache construction itself can be optimized. We show that prepending
short prefix tokens before a document during KV cache construction measurably changes
the quality of the resulting cache for downstream question answering, even though the
prefix tokens are discarded from the final cache. Through systematic experiments across
four models from three architecture families (Gemma, Mistral, Qwen), we decompose the
prefix effect into four independent components: (1) a RoPE position shift effect that
varies by architecture, (2) a token presence effect where prefix tokens participate in
attention during encoding and reshape document representations, (3) a vocabulary effect
from the specific words used, and (4) a semantic word-order effect that only emerges in
larger models capable of instruction following. We find that the token presence effect
is the dominant mechanism (d=+0.76 to +0.84 on Gemma 3 12B and Mistral 7B), while
semantic instruction following (d=+0.43 on Gemma 3 12B) requires model capacity above
~7B parameters. These findings reveal that KV cache quality is not fixed at encoding
time but is influenced by the context in which encoding occurs — a dimension orthogonal
to existing work on cache compression and eviction.

---

## 1. Introduction

KV cache precomputation has become a standard optimization in retrieval-augmented
generation (RAG) systems. Rather than re-encoding retrieved documents for every query,
systems like TurboRAG [Lu et al., 2025], CacheBlend [Yao et al., 2024], and
Cache-Craft [2025] precompute and store document KV caches offline, reusing them
across queries to reduce time-to-first-token by up to 9.4x.

All prior work treats the cache construction step as fixed: the document tokens are
encoded by the model and the resulting KV states are stored. Research has focused on
what happens *after* construction — which entries to keep (eviction: H2O [Zhang et al.,
2023], SnapKV [Li et al., 2024], StreamingLLM [Xiao et al., 2024]), how to compress
them (quantization: KIVI [Liu et al., 2024], KVQuant [2024]), and how to manage them
(CacheClip [Yang et al., 2025], Ada-KV [2025]).

We ask a different question: **can we make the cache entries themselves better?**

We show that prepending a short prefix (as few as 1 token) before a document during
cache construction changes the resulting KV representations in ways that improve
downstream QA performance. The prefix is discarded after encoding — it influences the
document representations through attention during the forward pass but is not retained
in the final cache. This costs 1–64 extra tokens during offline construction and
zero tokens at inference time.

Through a systematic four-level decomposition across four models, we identify the
mechanisms behind this effect:

1. **Position shift**: Encoding the document at shifted RoPE positions (due to the
   prefix occupying the initial positions) changes how the model processes relative
   position information. This effect varies by architecture and can be positive or
   negative.

2. **Token presence**: Having any tokens — even random ones — participate in
   self-attention during document encoding reshapes the document's value vectors. This
   is the dominant mechanism, contributing d=+0.76 to +0.84 on models where it is
   positive. We connect this to the attention sink phenomenon [Xiao et al., 2024]:
   prefix tokens serve as additional attention sinks that redistribute attention mass
   during encoding.

3. **Vocabulary**: The specific token identities in the prefix (independent of their
   order) have a small effect, often negative. Scrambled instruction words perform
   worse than random tokens on most models, suggesting that instruction-related
   embeddings can interfere when presented without coherent meaning.

4. **Semantic word order**: Coherent instructions ("Read and comprehend this text
   carefully") outperform their scrambled counterparts on larger models (d=+0.43 on
   Gemma 3 12B, d=+0.29 on Qwen 2.5 7B) but actively hurt smaller models (d=-0.28
   on Gemma 3N 4B). This reveals a model capacity threshold for instruction-following
   in the cache construction context.

Our contributions:

- A four-level decomposition framework that cleanly separates positional, structural,
  vocabulary, and semantic components of prefix conditioning
- Evidence that KV cache quality is malleable — not fixed by the document alone — and
  that the encoding context matters
- Cross-architecture validation showing that the dominant mechanism (token presence)
  generalizes across model families, while semantic effects are capacity-dependent
- Practical guidance: for production RAG systems, even prepending a single random
  token during cache construction can improve downstream QA (d=+1.17 on Gemma 3 12B)

---

## 2. Background and Related Work

### 2.1 KV Cache in Transformer Inference

In autoregressive transformer inference, the key-value (KV) cache stores the key and
value projections from previous tokens, avoiding redundant recomputation during
generation. For a model with $L$ layers, $H$ attention heads, sequence length $S$,
and head dimension $d_h$, the KV cache stores $2 \times L \times H \times S \times d_h$
values.

In RAG systems, the document is typically much longer than the query. Precomputing
the document's KV cache offline and reusing it across queries eliminates the dominant
cost of the first forward pass. TurboRAG [Lu et al., 2025] precomputes chunk-level
KV caches and stitches them at inference time with RoPE reordering, achieving 9.4x
TTFT reduction. CacheBlend [Yao et al., 2024] identifies that precomputed caches
lack cross-chunk attention and selectively recomputes a subset of tokens.

### 2.2 KV Cache Efficiency

**Eviction** methods reduce cache size by removing less-important entries. H2O
[Zhang et al., 2023] discovers power-law attention distributions and retains
"heavy hitter" tokens. SnapKV [Li et al., 2024] uses an observation window to
identify important entries per attention head. StreamingLLM [Xiao et al., 2024]
discovers "attention sinks" — initial tokens that absorb excess attention probability
regardless of content — and shows that retaining just 4 sink tokens plus a sliding
window enables infinite-length streaming.

**Quantization** methods compress cache values. KIVI [Liu et al., 2024] discovers
that keys need per-channel quantization while values need per-token quantization,
achieving 2-bit precision with negligible quality loss. KVQuant [2024] uses
non-uniform quantization with sensitivity-aware calibration.

### 2.3 Prompt Engineering and Soft Prompts

Prefix tuning [Li and Liang, 2021] learns continuous embeddings prepended to the
input to adapt frozen language models. Gist tokens [Mu and Li, 2023] train models
to compress prompts into learned "gist" token representations. These methods modify
the model's input at inference time.

Our approach is distinct: we prepend tokens during *cache construction* (offline),
not during inference. The prefix influences how the document is encoded into the KV
cache but is discarded before the cache is used for query answering. This means
zero additional cost at inference time.

### 2.4 Rotary Position Embeddings

RoPE [Su et al., 2021] encodes position information by rotating key and query vectors
by position-dependent angles. For position $p$ and frequency $\theta_i$:

$$k_p = k \cdot \cos(p \cdot \theta_i) + \text{rotate\_half}(k) \cdot \sin(p \cdot \theta_i)$$

where $\text{rotate\_half}$ splits the vector in half and swaps with negation:
$[-x_2, x_1]$. This encoding allows the model to attend based on relative position
through the inner product of rotated keys and queries.

RoPE is central to our method: when a prefix of length $P$ is prepended, document
tokens are encoded at positions $P+1, P+2, \ldots$ instead of $1, 2, \ldots$. After
discarding the prefix from the cache, we reposition the document keys back to
positions $1, 2, \ldots$ using a delta rotation. This repositioning is exact in
float32 but introduces small perturbations in bfloat16 (~0.26% relative noise).

---

## 3. Method

### 3.1 Two-Phase KV Cache Scoring

We use a two-phase approach to evaluate how prefix conditioning affects downstream
QA performance:

**Phase A (Offline — Cache Construction):**
1. Concatenate: $[\text{BOS}, \text{prefix}_1, \ldots, \text{prefix}_P, \backslash\text{n}, \text{doc}_1, \ldots, \text{doc}_D]$
2. Forward pass through the model to build the KV cache
3. Select only BOS + document entries from the cache (discard prefix + newline)
4. Reposition document keys via RoPE delta rotation from positions $(P+2, \ldots, P+1+D)$ to $(1, \ldots, D)$
5. Apply per-tensor normalization (optional, model-dependent)

**Phase B (Online — Query Answering):**
1. Concatenate: $[\backslash\text{n}, \text{query}, \backslash\text{n}, \text{answer}]$
2. Forward pass using the precomputed cache with position IDs starting at $D+1$
3. Compute NLL on answer tokens only

The key insight: during Step 2 of Phase A, the prefix tokens participate in
self-attention with the document tokens. Each document token's key and value
projections are influenced by attending to the prefix tokens. When we discard the
prefix in Step 3, this influence is retained in the document's KV representations.

### 3.2 Prefix Conditions

We test a range of prefix strategies spanning four mechanistic categories:

| Category | Conditions | Mechanism Tested |
|----------|-----------|-----------------|
| Structural | random tokens, repeat "the" | Token presence without meaning |
| Instruction | "comprehend", "extract", "summarize" | Task-directed attention guidance |
| Instruction (scrambled) | permuted instruction tokens | Vocabulary without word order |
| Anti-instruction | "Ignore this text completely" | Opposite semantic direction |
| Query-based | Oracle (actual query as prefix) | Maximum semantic relevance |
| Control | Position shift only (no tokens) | RoPE position effect in isolation |

All prefixed conditions use the same prefix length ($L=64$ tokens unless otherwise
specified), ensuring equal position shifts for fair comparison.

### 3.3 Four-Level Decomposition

We decompose the total prefix effect into four additive components:

$$\text{Total}(\text{comprehend}) = \underbrace{\Delta_{\text{shift}}}_{\text{Position}} + \underbrace{(\Delta_{\text{random}} - \Delta_{\text{shift}})}_{\text{Token presence}} + \underbrace{(\Delta_{\text{scrambled}} - \Delta_{\text{random}})}_{\text{Vocabulary}} + \underbrace{(\Delta_{\text{coherent}} - \Delta_{\text{scrambled}})}_{\text{Word order}}$$

where $\Delta_X = \text{NLL}_{\text{bare}} - \text{NLL}_X$ (positive means $X$ helps).

This decomposition is clean because:
- All $L=64$ conditions share the same position shift
- Random and scrambled conditions control for token count and vocabulary respectively
- The scrambled condition uses the same tokens as the coherent instruction, just permuted

### 3.4 Models and Datasets

**Models**: We select four models spanning three architecture families and two
size tiers:

| Model | Params | Architecture | RoPE $\theta$ | Attention |
|-------|--------|-------------|---------------|-----------|
| Gemma 3 12B-IT | 12B | Hybrid sliding+full | 10K / 1M | Mixed |
| Gemma 3N E4B-IT | ~4B | Hybrid sliding+full | 10K / 1M | Mixed |
| Mistral 7B-Instruct | 7B | Full attention | 1M | Uniform |
| Qwen 2.5 7B-Instruct | 7B | Full attention | 1M | Uniform |

**Datasets**: SQuAD v2 (extractive QA) and GSM8K (math reasoning, number-only answer).

**Evaluation**: Cohen's d effect size on paired NLL differences (condition vs bare),
computed on the hardest 40% of samples (top by bare NLL). Win rate and paired t-test
for significance. $N=200$ samples per dataset, 80 hard.

### 3.5 Implementation Details

**BOS handling**: Models without a native BOS token (Qwen 2.5) use the PAD token
as an attention sink anchor. We verify that bare two-phase NLL matches single-pass
NLL on all models (within bfloat16 tolerance of 0.004–0.11).

**RoPE repositioning**: We verify exact fp32 round-trip recovery (error < $5 \times 10^{-7}$)
on all models using the model-specific inverse frequency tensors.

**Normalization**: Per-tensor scale normalization
$(x / (\text{absmax}/127)) \times (\text{absmax}/127)$ is applied to all caches
including bare. This was found to benefit Gemma 3 models [prior work] but has
negligible effect on Mistral and Qwen.

---

## 4. Results

### 4.1 Prefix Conditioning Works Across Architectures

[TABLE: Full condition ranking across 4 models, 15 conditions]

With proper BOS handling and answer extraction, prefix conditioning produces
positive effects on all four models: 14/15 conditions positive on Gemma 3 12B,
15/15 on Gemma 3N E4B, 11/15 on Mistral 7B, and 10/15 on Qwen 2.5 7B.

### 4.2 Four-Level Decomposition

[TABLE: Decomposition table from our analysis — the key result]

**Position shift** (Component 1): The effect of encoding at shifted positions
without any prefix tokens varies by architecture. Gemma 3N (+0.64) and Qwen (+0.56)
benefit from the position shift alone, while Gemma 3 12B (-0.18) and Mistral (-0.18)
are slightly hurt. [TODO: investigate why — may relate to how each architecture handles
non-contiguous position sequences]

**Token presence** (Component 2): The dominant mechanism. Having random tokens
participate in attention during document encoding adds d=+0.76 (Gemma 3 12B) and
d=+0.84 (Mistral 7B) beyond the position shift. These tokens serve as additional
attention targets during encoding, redistributing attention mass and changing which
information is captured in the document's value vectors. On Qwen, this effect is
strongly negative (-0.66), suggesting that random attention targets interfere with
Qwen's encoding strategy.

**Vocabulary** (Component 3): The specific token identities in the prefix
(scrambled instruction words vs random tokens) have a generally negative effect
(-0.08 to -0.49). Instruction-related embeddings, when presented without coherent
order, are worse than arbitrary tokens. This is consistent with the embeddings
activating instruction-related circuits that then produce unhelpful attention
patterns when the instruction is incoherent.

**Semantic word order** (Component 4): Coherent instructions outperform their
scrambled counterparts only on models with sufficient capacity for instruction
following: Gemma 3 12B (+0.43), Qwen 2.5 7B (+0.29), negligible on Mistral 7B
(+0.04), and harmful on Gemma 3N 4B (-0.28). This reveals a capacity threshold:
models below ~7B parameters cannot leverage instruction semantics in the cache
construction context and are actively confused by them.

### 4.3 Prefix Length Analysis

[FIGURE: Length curves for comprehend and random across models]

A surprising finding: shorter prefixes often outperform longer ones. On Gemma 3 12B,
a single random token (L=1) achieves d=+1.17, the highest effect of any condition.
Random prefix effectiveness *decreases* monotonically with length:
L=1 (+1.17) > L=4 (+1.11) > L=16 (+0.83) > L=64 (+0.59).

This suggests that the benefit of token presence saturates quickly while additional
tokens add noise. One interpretation: the first token creates an attention sink that
reshapes encoding; subsequent tokens dilute rather than reinforce this effect.

For comprehend prefixes, the pattern differs: L=16 (+1.03) > L=4 (+0.98) > L=1 (+0.95)
> L=64 (+0.84). The semantic content needs ~16 tokens to manifest but degrades at 64,
possibly due to the model over-attending to a long instruction prefix.

### 4.4 Downstream Accuracy

[TODO: EM/F1 on SQuAD, accuracy on GSM8K — experiments needed]

### 4.5 Generative Task Evaluation

[TODO: generation quality metrics — experiments needed]

---

## 5. Discussion

### 5.1 Why Does Token Presence Dominate?

The largest single mechanism is Component 2: having any tokens (even random ones)
before the document during encoding. We propose this operates through attention
redistribution. During the Phase A forward pass, document tokens attend to both
the prefix tokens and other document tokens. The prefix tokens serve as additional
attention targets that:

(a) Absorb attention mass that would otherwise concentrate on the first few document
tokens (extending the attention sink phenomenon of [Xiao et al., 2024])

(b) Create a richer context for the first document tokens, which no longer need to
serve as the sole attention anchors

(c) Change the layer-by-layer processing of document tokens in ways that persist
in the value vectors even after prefix removal

The observation that this effect is *negative* on Qwen but positive on Gemma and
Mistral suggests that different model architectures develop different strategies for
handling initial-position tokens, and disrupting that strategy can be harmful.

### 5.2 The Model Capacity Threshold

Our most publishable finding may be the clean separation between structural and
semantic effects by model capacity. Models above ~7B parameters (Gemma 3 12B,
Qwen 2.5 7B) show positive word-order effects (+0.29 to +0.43): they can
"understand" that "Read and comprehend this text" means something different from
the same words scrambled, and this understanding reshapes attention during encoding.

The 4B model (Gemma 3N) not only fails to leverage this meaning but is actively
damaged by it (-0.28). Coherent instructions create expectations the model cannot
fulfill, leading to worse representations than if no instruction were given.

This has practical implications: for smaller models deployed in production RAG
systems, structural prefixes (random tokens or repeated tokens) are safer and more
effective than semantic instructions.

### 5.3 Implications for RAG System Design

For production deployment:
- **Large models (>7B)**: Use a short task-relevant instruction prefix (L=16–64).
  "Extract the key facts" is more universally effective than "comprehend."
- **Small models (<7B)**: Use a minimal structural prefix (L=1–4 random tokens).
  Avoid instructions — they hurt.
- **All models**: Ensure a BOS/attention-sink token anchors the cache. Models without
  native BOS need an artificial one.
- **Cost**: Near-zero. The prefix adds 1–64 tokens to the offline encoding pass and
  nothing to the online query path.

### 5.4 Limitations

- **NLL metric**: Our primary metric is NLL on answer tokens. While NLL correlates
  with answer quality, we have not yet validated with generation-based metrics
  (EM, F1, ROUGE). [TODO: downstream accuracy experiments]
- **Dataset scope**: The deep decomposition is validated on two datasets (SQuAD v2,
  GSM8K). Broader dataset coverage is needed.
- **Causal claims**: The four-level decomposition is additive by construction. There
  may be interaction effects between components that this framework does not capture.
- **Generative tasks**: All experiments use extractive QA. The effect on open-ended
  generation, summarization, or multi-turn dialogue is unknown.

---

## 6. Conclusion

We show that KV cache construction is not a fixed operation — the context in which a
document is encoded into its KV representation affects the quality of that
representation for downstream tasks. Through a four-level decomposition, we identify
token presence (attention redistribution during encoding) as the dominant mechanism,
with semantic instruction following as a secondary effect available only to larger
models. These findings open a new dimension for RAG optimization: rather than only
compressing or evicting cache entries after construction, we can improve them during
construction at near-zero cost.

---

## References

1. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453
2. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS 2023. arXiv:2306.14048
3. Li, Y., Huang, Y., Yang, B., et al. (2024). SnapKV: LLM Knows What You Are Looking for Before Generation. NeurIPS 2024. arXiv:2404.14469
4. Lu, S., Wang, H., Rong, Y., Chen, Z., Tang, Y. (2025). TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text. EMNLP 2025. arXiv:2410.07590
5. Yao, J., Li, H., Liu, Y., et al. (2025). CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. EuroSys 2025 (Best Paper). arXiv:2405.16444
6. Liu, Z., Yuan, J., Jin, H., et al. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. arXiv:2402.02750
7. Li, X.L., Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL-IJCNLP 2021. arXiv:2101.00190
8. Mu, J., Li, X.L., Goodman, N. (2023). Learning to Compress Prompts with Gist Tokens. NeurIPS 2023. arXiv:2304.08467
9. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568. arXiv:2104.09864
10. Hooper, C., Kim, S., Mohammadzadeh, H., et al. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. NeurIPS 2024. arXiv:2401.18079

---

## Appendix

### A. Pipeline Correctness Verification

We verify pipeline correctness with the following tests on all four models:

1. **RoPE round-trip**: Rotating keys by delta then by -delta recovers original keys
   within fp32 tolerance ($< 5 \times 10^{-7}$). In bfloat16, the round-trip error is
   ~0.016 (inherent precision limit).

2. **Bare two-phase = single-pass**: Without any prefix or repositioning, two-phase
   scoring produces NLL within 0.004–0.11 of single-pass scoring on all models.

3. **BOS handling**: Qwen 2.5 7B lacks a native BOS token. We use PAD (token 151643)
   as an artificial attention sink. Ablation confirms this is essential: without BOS,
   comprehend d=-0.65; with BOS, d=+0.13.

4. **Gemma 3N use_cache bug**: Gemma3nForConditionalGeneration produces incorrect
   logits when use_cache=False. We use use_cache=True for all forward passes and
   verify equivalence on other models.

### B. Full Per-Model Per-Dataset Results

[TODO: Tables with all condition × dataset × model results]

### C. Normalization Analysis

Per-tensor KV cache normalization $(x / (\text{absmax}/127)) \times (\text{absmax}/127)$
produces a bfloat16 perturbation that benefits Gemma 3 models (which have high
inter-layer scale variation, CoV=0.82) but is neutral on Mistral (CoV=0.14) and
Qwen (CoV=1.91). The normalization's relative perturbation is identical (~0.25%)
across models; the difference in impact reflects architectural differences in
activation scale consistency.
