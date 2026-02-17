# Research Framing: Directed KV Caches for Ad Serving

## The Core Idea

When serving ads (or any document-retrieval-then-generation pipeline), we need to
score user queries against pre-computed document representations. The question is:
**can we build better document representations by pre-encoding documents alongside
surrogate queries?**

## The Three Points on the Spectrum

### Upper Bound: Full Cross-Attention at Runtime (Expensive, Best Quality)
```
User query arrives → encode([real_query + document]) → generate answer
```
- Query and document see each other with full bidirectional attention
- Document representation is perfectly tailored to this specific query
- **Cost**: O(Q × D) encoder forward passes at serving time
- **Quality**: Best possible — this is what we're trying to approximate

### Lower Bound: Document Cache in Isolation (Cheap, Worst Quality)
```
Offline: encode([document]) → store encoder representation
Online:  user query arrives → decode answer using stored representation
```
- Document representation has ZERO query awareness
- Decoder must do all the work of connecting query to document
- **Cost**: O(D) offline, O(1) encoder work at serving time
- **Quality**: Worst — the document doesn't "know" what question to answer

### Our Proposal: Surrogate-Primed Document Cache (Middle Ground)
```
Offline: encode([surrogate_query + document]) → store encoder representation
Online:  user query arrives → decode answer using stored representation
```
- Document representation has SOME query awareness baked in
- The surrogate is a proxy for the kinds of queries this document will serve
- Bidirectional attention means document tokens are influenced by the surrogate
- **Cost**: Same as lower bound at serving time (O(1) encoder work)
- **Quality**: Between lower and upper bound — how close to upper bound?

## What the Surrogate Could Be

In an ad-serving context, the surrogate query could be:
1. **Keywords from the ad/document itself** (cheapest, no external knowledge)
2. **A representative query** from historical search logs for this ad
3. **An LLM-generated query** ("what would someone search to find this ad?")
4. **A category label** or intent template ("best deals on [product]")

## What We're Measuring

For each (query, document, answer) triple:
- **NLL(answer | encoder_output)**: How well can the decoder predict the answer
  given only the encoder representation (no query in decoder)?
- Lower NLL = better encoder representation for this query

## The Truncation Question

When we encode [surrogate + document], the encoder output contains representations
for BOTH surrogate tokens and document tokens. The decoder cross-attends to all of them.

Is the benefit from:
**(a)** The decoder reading the surrogate tokens from the encoder output (trivial —
   just an indirect way of giving the decoder the query), OR
**(b)** The document token representations being genuinely improved by having been
   co-encoded with the surrogate (the real value proposition)?

We test this by **masking** surrogate tokens from decoder cross-attention:
- Encoder still processes [surrogate + document] with full bidirectional attention
- But decoder can only cross-attend to document token positions
- If (b) is the mechanism, the benefit persists after masking
- If (a) is the mechanism, the benefit disappears

## Results So Far

### Exp 33b (v2): Proof of Concept (no truncation)
| Condition | d vs bare | Captures |
|-----------|-----------|----------|
| oracle (real query) | +0.345 | 100% |
| surr_doc (doc keywords) | +0.312 | 96% |
| surr_para (paraphrased query) | +0.293 | 70% |
| static prefix | +0.103 | 9% |

### Exp 01 (v3): Truncation Test — THE KEY RESULT
Truncation makes it STRONGER, not weaker. Masking surrogate tokens from
decoder cross-attention INCREASES the benefit.

| Condition | d vs bare | Win% |
|-----------|-----------|------|
| oracle_trunc | **+0.408** | 94% |
| surr_doc_trunc | +0.363 | 85% |
| surr_para_trunc | +0.357 | 88% |
| oracle_full | +0.345 | 82% |

**What this proves:**
1. Document representations ARE genuinely improved by co-encoding with query/surrogate
2. Having query tokens visible to decoder is actually a DISTRACTION — removing them helps
3. Surrogate keywords from the document itself capture 89% of the oracle benefit
4. The mechanism is real and the practical deployment path is clear:
   - Offline: encode([doc_keywords + document]) with T5Gemma encoder
   - Store only the DOCUMENT PORTION of the encoder output (truncate surrogate tokens)
   - Online: decoder cross-attends to stored document representations to answer queries
