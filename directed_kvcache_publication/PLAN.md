# Directed KV Cache — Publication Plan

Quick reference. Full details in `00_publication_plan.ipynb`.

## Working Title
"Directed KV Cache Construction: How Prefix Conditioning Improves
Precomputed Document Representations for Retrieval-Augmented Generation"

## Missing Experiments (P0 = blocking)

| Priority | Experiment | Effort | Directory |
|----------|-----------|--------|-----------|
| P0 | Multi-model (Llama 3.1 8B + Qwen 2.5 7B) | 3-5 days | experiments/01_multi_model/ |
| P0 | Downstream accuracy (EM/F1/accuracy) | 1-2 days | experiments/02_downstream_accuracy/ |
| P1 | Latency measurements | 0.5 days | experiments/03_latency/ |
| P2 | Attention analysis | 1-2 days | experiments/04_attention_analysis/ |

## Prerequisite
Generalize `lib/rope.py` for Llama/Qwen (1-2 days). No sliding attention,
different RoPE config paths.

## Execution Order
1. lib/ generalization → Exp 01 (multi-model) + Exp 03 (latency)
2. Exp 01 complete → Exp 02 (downstream accuracy)
3. Exp 04 (attention, if time) → figures → paper draft

## Paper Figures (6 main + 2 tables)
1. Methodology diagram (exists)
2. All prefixes help — bar chart (exists)
3. Decomposition by prefix length (exists)
4. Soft prompts 2× text — ranking + dumbbell (exists)
5. Normalization — bar chart (exists)
6. Transfer matrix — heatmap (exists)
7. **Table 1**: Multi-model validation (NEW)
8. **Table 2**: Downstream accuracy (NEW)

## Data Sources
All existing results in `../directed_kvcache_v4/results/decoder_only/exp01-exp15/`.
New results in `results/exp01-exp04/`.
