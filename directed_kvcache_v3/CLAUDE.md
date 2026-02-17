# Directed KV Cache v3: T5Gemma Encoder-Decoder Research

## Overview
Research on priming document encoder representations with surrogate queries for ad-serving.
Model: T5Gemma 2 4B-4B (encoder-decoder, bidirectional encoder).
Dataset: MS MARCO v1.1.

**This is a continuation of directed_kvcache_v2 (32 experiments, decoder-only models).**
The v2 project established that decoder-only causal attention cannot leverage query-conditioned
encoding because the query is already available at inference time. T5Gemma's bidirectional
encoder changes the picture: Exp 33b showed oracle d=+0.345, surrogate transfer captures 70-96%
of the oracle gap.

## CRITICAL: Key Findings from v2 to Build On

### What Worked (Exp 33b)
- **Oracle query in encoder: d=+0.345, 81.5% win rate (p<0.001)**
- **Doc-keyword surrogate: captures 96% of oracle gap**
- **Paraphrase surrogate: captures 70% of oracle gap**
- Static prefix: only 9% of oracle gap (content-agnostic, not the mechanism here)
- Hardness gradient: oracle benefit scales Q1(+0.05) → Q5(+1.05)
- Setup: query in encoder ONLY, decoder scores answer ONLY

### What Didn't Work in v2 (Don't Repeat)
- Priming for ranking in decoder-only models (Exps 22/23/28)
- Query-likelihood ranking (Exp 31)
- Long documents >200 tokens (Exp 20, value contamination diluted)
- Cross-dataset generalization beyond MS MARCO (Exp 19)
- Contrastive/hinge loss for ranking (Exp 28)

### Open Question: Truncation
Exp 33b did NOT truncate query tokens from encoder output. The decoder cross-attended
to ALL encoder tokens including query/surrogate. The benefit could be:
(a) Decoder reads query directly from encoder output (trivial)
(b) Document representations improved by bidirectional co-encoding (the real prize)
Exp 01 in v3 tests this via masking.

## T5Gemma 2 Architecture Notes

### Critical: Cross-attention has NO RoPE
In `T5Gemma2MergedAttention`, cross-attention keys are projected but RoPE is NOT applied:
```python
cross_key_states = self.k_proj(encoder_hidden_states)  # projected
cross_key_states = self.k_norm(cross_key_states)        # normalized
# NO apply_rotary_pos_emb!
```
This means we can safely mask/slice encoder outputs without position correction.

### Merged Self+Cross Attention
The decoder uses a single attention op with concatenated [self_KV ; cross_KV]:
```python
key_states = torch.cat([self_key_states, cross_key_states], dim=2)
value_states = torch.cat([self_value_states, cross_value_states], dim=2)
```
Self-attention portion has RoPE + causal mask. Cross-attention portion has no RoPE + bidirectional mask.

### Masking Encoder Tokens (Preferred Over Slicing)
To hide query tokens from decoder cross-attention, modify `attention_mask`:
```python
# Encode [query + document] with full bidirectional attention
encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=full_mask)

# Mask query tokens for decoder cross-attention
cross_attn_mask = torch.ones(1, total_len, device=device)
cross_attn_mask[:, :query_token_count] = 0  # mask query positions

# Decoder sees only document representations (but those reps contain query info
# from encoder's bidirectional attention)
outputs = model(encoder_outputs=encoder_outputs, attention_mask=cross_attn_mask, labels=labels)
```

### Model Loading
```python
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM
processor = AutoProcessor.from_pretrained("google/t5gemma-2-4b-4b", token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-2-4b-4b", device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
```

### Key Properties
- Encoder: bidirectional attention, dual-frequency RoPE (global + sliding window)
- Decoder: causal self-attention + merged cross-attention (no RoPE on cross keys)
- Sliding window: every other layer uses 4096-token window
- Tied word embeddings (encoder + decoder share)
- ~7B total params (4B encoder + 4B decoder), ~15GB VRAM in BF16
- Pretrained only (NOT instruction-tuned)
- Gated repo: requires HF_TOKEN

## Known Pitfalls (Inherited from v2)

### 0. File Permissions
JupyterLab runs as `jupyter`, CLI as `petergrabowski_google_com`. No shared group.
**For notebooks**: `os.umask(0o000)` at top.
**For CLI**: prefix with `umask 000 &&`.

### 1. T5Gemma is Pretrained Only
No instruction-tuned checkpoint. Don't expect it to follow instructions or use
chat templates for scoring. Just use raw NLL scoring.

### 2. attention_mask Serves Double Duty
When passing pre-computed encoder_outputs, the `attention_mask` parameter becomes
`encoder_attention_mask` in the decoder (controls cross-attention). The encoder is
skipped. This is exactly the mechanism for masking/truncation experiments.

### 3. Encoder Representations Are Context-Dependent
Even when masking query tokens from cross-attention, document token representations
ALREADY contain query information from the encoder's bidirectional self-attention.
"Truncation" here means: preventing the decoder from directly reading query tokens,
while preserving query influence on document representations.

### 4. Statistical Methodology (from v2)
- Always report Cohen's d, win%, AND p-value
- Cross-validate any tuned parameters (Exp 14 overfitting lesson)
- Bonferroni correction for multiple comparisons
- Checkpoint every 20 samples for resume
- Separate ranking metrics (AUC, MRR) from average NLL

### 5. HuggingFace Token
Stored in `.env` file. Load with:
```python
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
```

## Directory Structure
```
directed_kvcache_v3/
  .env                    -- HuggingFace token
  CLAUDE.md               -- This file
  RESEARCH_FRAMING.md     -- Upper/lower/middle bound framing
  EXPERIMENT_PLAN.md      -- Experiment roadmap and results log
  lib/                    -- Shared utilities
  results/expXX/          -- Per-experiment outputs
  tests/                  -- Tests
  scripts/                -- Build scripts
  build_expXX.py          -- Notebook build scripts
  XX_name.ipynb           -- Experiment notebooks
```

## Experiment Log
See EXPERIMENT_PLAN.md for full roadmap and detailed results.

- **Exp 01**: Truncation test — BREAKTHROUGH. Masking query tokens from decoder
  cross-attention makes the benefit STRONGER (d=+0.408 trunc vs +0.345 full).
  Document representations are genuinely improved by co-encoding. Surrogate
  keywords from the document capture 89% of oracle benefit (d=+0.363).
  Truncation parallels the decoder-only values-only finding.
- **Exp 02**: Surrogate type sweep — NO content gradient (Spearman rho=-0.167, p=0.693),
  mechanism is primarily structural. BUT pairwise comparisons show per-sample semantic
  advantage (surr_doc > random, d=+0.130, p=0.004). Static_fact achieves 99% of oracle
  by Cohen's d but lower absolute NLL improvement. Best doc-derived: surr_template
  ("What is [keyword]?") at d=+0.336 (90% of oracle). Random gets 81% — large structural floor.
  surr_lead anomalously weak (40% oracle). N=500, all sig after Bonferroni.
- **Exp 2B**: Mechanism decomposition — **85% of headroom is pure structure**, only 10%
  is semantic (vocabulary 6% ns, semantics 10% ***). Binary SWITCH: 1 random word gets
  85% of oracle. "the" x10 gets 90% of oracle. Uniform tokens beat diverse random.
  NOT v2's value contamination (truncation improves it) — structural representation
  enrichment through bidirectional attention. Semantic component grows for harder samples.
- **Exp 03** (planned): Length scaling — does the benefit survive longer documents?
- **Exp 04** (planned): Ranking on MS MARCO + Amazon ESCI + WANDS
- **Exp 05** (future): LLM-generated surrogates — can an instruction-tuned LLM beat heuristics?
