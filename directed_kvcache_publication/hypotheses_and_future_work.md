# Why Does Directed KV Cache Priming Help Some Models and Not Others?

## Hypotheses Tested, Results, and Proposed Follow-On Experiments

---

## Part 1: Hypotheses We Have Tested

### H1: Priming effectiveness scales with model size
**Status: Not supported (no clean correlation)**

Within the Qwen instruct family (0.5B → 32B), keyword effectiveness shows no monotonic trend: d = +0.52, +0.43, +0.14, +0.62, +0.19, +0.50. The correlation between log2(params) and keyword d is r = -0.21 across all instruct models.

However, *semantic sensitivity* (the ability to leverage word order in the prefix) does scale with size in Qwen: coherent-minus-scrambled d correlates at r = +0.75 with log2(params). Larger models don't benefit MORE from priming overall, but they become more discriminating about WHAT kind of prefix helps.

**Implication**: Size determines the quality of response to priming (semantic vs token-presence), not the magnitude.

---

### H2: Full attention models respond differently from hybrid sliding/full attention models
**Status: Partially supported (different preference patterns, not different magnitude)**

| Metric | Full attention (n=9) | Hybrid (n=7) |
|--------|---------------------|--------------|
| Mean keyword d | +0.28 | +0.30 |
| Mean extract d | +0.22 | +0.30 |
| Mean semantic delta | +0.08 | -0.14 |

Overall priming magnitude is nearly identical between architectures. But they diverge on semantic sensitivity: full-attention models develop positive semantic delta (coherent > scrambled), while hybrid models show negative semantic delta (scrambled often better). This suggests hybrid sliding-window attention disrupts the propagation of long-range instruction-following signals, making token presence the dominant mechanism.

**Implication**: Architecture determines the *mechanism* (semantic vs token-presence) but not the *magnitude* of priming.

---

### H3: Instruction tuning is necessary for priming to work
**Status: Model-family dependent**

| Pair | Base positive/25 | IT positive/25 | Base kw d | IT kw d |
|------|-----------------|----------------|-----------|---------|
| Gemma 3 4B | 0/25 | 20/25 | -0.07 | +0.12 |
| Qwen 2.5 7B | 25/25 | 21/25 | +0.10 | +0.62 |

For Gemma, IT is **necessary** — the base model is universally harmed. For Qwen, IT is **not necessary** (base responds to 25/25 conditions) but IT **amplifies document-derived priming** (keywords jump from +0.10 to +0.62) while reducing generic instruction response.

IT changes which prefixes work: base models prefer generic instructions (Qwen 7B base: extract d=+0.42, index d=+0.43), IT models prefer document-derived priming (Qwen 7B IT: keywords d=+0.62).

**Implication**: IT doesn't gate priming on/off universally; it shifts preference from generic to document-specific priming. Whether IT is required depends on the model family.

---

### H4: Task complexity predicts priming benefit
**Status: Supported (high reasoning > mid > factoid)**

| Tier | Mean keyword d (responsive models) |
|------|-----------------------------------|
| High reasoning (GSM8K, DROP) | +0.40 |
| Mid reasoning (SQuAD, HotpotQA) | +0.33 |
| Factoid (TriviaQA, MS MARCO) | +0.17 |

GSM8K is the standout: Qwen 7B IT achieves d=+1.22, Qwen 14B achieves d=+1.30 with keywords. Math reasoning benefits enormously, possibly because surfacing numerical terms helps the model attend to quantities during encoding.

Factoid retrieval shows the weakest effects. When the answer is a surface-level fact, there may be less room for attention reallocation to improve encoding.

**Implication**: Priming helps most when the task requires multi-step reasoning over the document, not simple fact lookup.

---

### H5: The oracle (query-as-prefix) is the performance ceiling
**Status: Refuted**

Document-derived surrogates outperform the oracle for most models. Natural-length oracle is worse than padded-to-64 for 10/16 models, and both are outperformed by keywords or fixed instructions for 9/16 models.

The mechanism is not query approximation. The best prefix activates document-relevant features, not query-relevant features.

---

### H6: Prefix semantic content matters (instruction following)
**Status: Largely refuted for most models**

- Anti-instruction ("Ignore this text completely") produces positive effects for 11/16 models
- Scrambled instructions match or exceed coherent instructions for most models (especially hybrid)
- Random tokens produce positive effects for 12/16 models
- Single tokens ("Facts") can match full 64-token instructions

The dominant mechanism is token presence / feature activation, not instruction following. Semantic content matters only for larger full-attention instruct models (Qwen ≥ 7B).

---

### H7: NLL improvement predicts generation quality improvement
**Status: Not reliably**

Tested on 4 models × 3 datasets. Keywords improve NLL for 11/12 combinations but improve Exact Match for only a subset. Worst case: Gemma 12B on TriviaQA, NLL improves (d=+0.06) but EM drops from 45.5% to 34.0% (-11.5pp).

Best case: Gemma 12B + extract on SQuAD, EM goes from 24.8% to 40.0% (+15.2pp).

---

### H8: Cache-time priming is better than inference-time keyword prepending
**Status: Model-dependent**

For Qwen models: cache-time dominates (8/9 model-dataset wins). Inference-time keywords actively hurt Qwen 7B (d = -0.25 to -0.59).

For Gemma 12B: inference-time is better on SQuAD (d=+0.67 vs +0.30) and TriviaQA (d=+0.42 vs +0.06).

---

### H9: There exists a single best prefix for all models
**Status: Definitively refuted**

6 fixed instructions tested, each wins on at most 3/16 models. TF-IDF keywords are the most consistent (top 3 for 7/16) but rank 10th-14th for several models. The optimal prefix depends on model family, size, and instruction tuning.

---

### H10: Prefix length scales with effectiveness
**Status: Refuted (non-monotonic)**

L=1 matches or exceeds L=64 for many models:
- Qwen 32B: "Facts" (L=1) achieves d=+0.62, comprehend L=64 achieves d=+0.05
- Gemma 12B: comprehend L=1 (d=+0.44) ≈ L=64 (d=+0.36)

Longer prefixes sometimes hurt, possibly by activating instruction-following circuits that interfere with document encoding.

---

## Part 2: Variables That May Explain the Outliers

### The three harmed models share no obvious trait:
| Model | Family | Params | Attn | IT | theta | SW | Mean bare NLL |
|-------|--------|--------|------|------|-------|-------|----------|
| Gemma 4B base | Gemma 3 | 4B | hybrid | **No** | 10K | 1024 | **4.15** (low) |
| Ministral 8B | Ministral | 8B | hybrid | Yes | **100M** | **32768** | **2.66** (lowest) |
| Qwen 14B | Qwen | 14B | full | Yes | 1M | none | 7.37 |

### Strongest predictor found: baseline task difficulty (r = +0.49)
Models with higher mean bare NLL (worse at the task) benefit more from keyword priming. This is intuitive: models that are already confident have less room for improvement and more risk of perturbation.

| Group | Models | Mean keyword d |
|-------|--------|---------------|
| High bare NLL (> 6.0) | Gemma3N E4B, Qwen 7B IT, DeepSeek, Qwen 14B, Gemma 1B, Qwen 32B | +0.39 |
| Low bare NLL (< 6.0) | Qwen 0.5B, Qwen 1.5B, Qwen 3B, Gemma 4B IT, Mistral, Ministral, Gemma 12B, Gemma 27B | +0.19 |

This partially explains Ministral 8B (bare NLL = 2.66, lowest — already very confident) and Gemma 4B base (bare NLL = 4.15, second lowest). But Qwen 14B has high bare NLL (7.37) and is still harmed, so baseline difficulty is not the full story.

### Moderate predictor: RoPE theta (r = -0.36)
Higher theta correlates with lower keyword effectiveness. Ministral's theta = 100M is 10,000× larger than Gemma's 10K. Very high theta means position encodings vary extremely slowly with position, which might make the RoPE reposition step less meaningful (everything already looks like everything else positionally).

---

## Part 3: Hypotheses We Have NOT Tested (Proposed Follow-On Experiments)

### H11: Baseline task confidence predicts per-sample priming benefit
**Can test with existing data (no new experiments needed)**

For each sample, correlate the bare NLL with (bare NLL - keyword NLL). If the relationship is positive and monotonic, it confirms that priming helps uncertain predictions more than confident ones. If there's a threshold below which priming consistently hurts, that's actionable guidance (only prime when bare NLL exceeds the threshold).

### H12: Document length interacts with priming effectiveness
**Can test with existing data**

Our samples range from 30-500 words. Bin samples by document length and compute keyword d per bin. If priming helps more on short documents (where the prefix is a larger fraction of total tokens), the mechanism is token-count-driven. If it helps more on long documents, the mechanism may be attention-redistribution.

### H13: Answer-in-document overlap predicts priming benefit
**Can test with existing data**

For extractive QA (SQuAD, TriviaQA), check if priming helps more when the answer appears verbatim in the document versus when it requires paraphrasing or inference. Keyword priming might surface the answer tokens specifically.

### H14: Per-sample keyword-query semantic similarity predicts benefit
**Can test with existing data + embeddings**

Compute cosine similarity between TF-IDF keyword embedding and query embedding per sample. If priming helps more when keywords overlap with the query, the mechanism involves query-relevant feature activation. If there's no correlation, it's generic feature activation.

### H15: The number of sliding vs full attention layers predicts priming response
**Can test with existing data**

Gemma models have different ratios of sliding/full layers:
- 1B: 22 sliding / 4 full (85% sliding)
- 4B: 29 / 5 (85% sliding)
- 12B: 40 / 8 (83% sliding)
- 27B: 52 / 10 (84% sliding)

The ratios are similar, so this likely doesn't explain within-Gemma variation. But Ministral has 27/9 (75% sliding) with a much larger window (32768), which means its sliding layers are functionally equivalent to full attention for our document lengths. This effectively makes Ministral a full-attention model that happens to have a hybrid config — which might explain why it behaves differently from Gemma hybrid models.

### H16: RoPE reposition fidelity differs across models and explains effectiveness
**Can test with existing data (preflight results)**

We have layer-0 reposition error for all 16 models (0.03 to 1.0). Does reposition error correlate with priming effectiveness? If models with higher reposition error show worse priming, the RoPE correction is a bottleneck.

From our data: DeepSeek (error = 1.0) has keyword d = -0.04, while Gemma 12B (error = 0.016) has keyword d = +0.34. But the sample is too small and confounded to draw conclusions.

### H17: Priming composes with KV cache compression
**Requires new experiments**

Does a primed cache retain its advantage after quantization (KIVI 2-bit, KVQuant) or eviction (H2O, SnapKV)? If the priming signal is stored in high-precision dimensions that get discarded by compression, the benefit disappears. If it's stored in the dominant components, it composes well. This is critical for practical deployment where caches are typically compressed.

### H18: Multi-document / multi-chunk RAG priming
**Requires new experiments**

Production RAG retrieves 3-10 document chunks and concatenates their caches. Does priming each chunk independently help, or does cross-chunk interaction (as in CacheBlend) change the picture? Can you prime the combined cache rather than individual chunks?

### H19: Learned prefix optimization (soft prompts)
**Requires new experiments (training)**

Instead of TF-IDF keywords or fixed instructions, learn a continuous prefix embedding that maximizes downstream QA performance when used for cache priming. This is related to prefix tuning (Li & Liang, 2021) but applied to offline cache construction. The learned prefix might capture model-specific priming signals that our discrete prefixes miss.

### H20: Model-specific keyword selection
**Requires new experiments**

Instead of corpus-level TF-IDF, select keywords that are most salient to each specific model. For example, use attention rollout on a bare forward pass to identify which document tokens receive the most attention, then use those as the prefix. This adapts the keyword set to the model's own attention patterns.

### H21: Other model families (Llama, Phi, Command-R, Falcon)
**Requires new experiments**

Our 5 families may not cover the full space. Llama 3 (which we can't access due to licensing) is the most widely deployed family. Phi models have unusual training (synthetic data heavy). Testing on more families would strengthen generalization claims and potentially reveal what model properties predict priming responsiveness.

### H22: Non-English and non-QA tasks
**Requires new experiments**

All our data is English QA. The mechanism (token presence modifying attention) should be language-agnostic, but verification is needed. Non-QA tasks (summarization, translation, code generation) would test whether priming generalizes beyond the QA format.

### H23: Priming with model's own outputs (self-priming)
**Requires new experiments**

Instead of external keywords or instructions, use the model's own preliminary output as the prefix. For example: (1) encode document bare, (2) generate a brief summary, (3) re-encode document with that summary as prefix, (4) store the re-encoded cache. This is a form of self-distillation at cache construction time. It requires 2x forward passes but uses the model's own understanding of what's important.

### H24: Attention pattern analysis (mechanistic understanding)
**Requires new experiments (attention extraction)**

Extract attention matrices from bare vs primed encoding and quantify how attention distributions change. Specifically: does keyword priming increase attention to answer-bearing tokens? Does it reduce attention sink concentration at BOS? Does it change the entropy of attention distributions? This would move the "feature activation" hypothesis from plausible to verified.

### H25: Training-data-driven explanation for negative models
**Requires access to training details**

Ministral 8B, Qwen 14B, and Gemma 4B base are all harmed. If we could examine their training data composition or training procedure, we might identify what makes them resistant. For example:
- Were they trained with prefix-augmented data? (If so, a mismatched prefix might be worse than none.)
- Is their attention sink behavior unusual?
- Do they have unusually strong position-dependent biases that make reposition harmful?

This is hard to test without insider knowledge of training procedures but would be the most explanatory.

---

## Part 4: Recommended Priority Order

### Can do now (existing data, no new experiments):
1. **H11**: Per-sample bare NLL vs priming benefit correlation
2. **H12**: Document length interaction
3. **H13**: Answer-in-document overlap analysis
4. **H16**: Reposition fidelity vs effectiveness correlation

### Medium effort (need embeddings or new scoring runs):
5. **H14**: Keyword-query semantic similarity
6. **H15**: Sliding/full layer ratio analysis (mostly done)
7. **H20**: Model-specific keyword selection via attention rollout

### Requires new experiments (significant GPU time):
8. **H17**: Composition with cache compression
9. **H18**: Multi-chunk RAG evaluation
10. **H22**: Non-English / non-QA tasks

### Requires training:
11. **H19**: Learned soft prefix optimization
12. **H23**: Self-priming (2-pass encoding)

### Requires external information:
13. **H21**: New model families
14. **H24**: Attention pattern extraction
15. **H25**: Training data analysis
