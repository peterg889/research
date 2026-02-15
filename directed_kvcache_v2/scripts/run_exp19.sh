#!/bin/bash
# Run Exp 19: Gemma Priming — Precision Fix & Selective Value Contamination
#
# Tests two hypotheses for why priming fails on Gemma 3 4B:
# H1: bfloat16 RoPE precision is the bottleneck (try float32)
# H2: Selective value contamination amplifies the weak signal
# 9 conditions, ~300 queries × ~8 passages × (2 forward + 9 scoring) ≈ 2-3h on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 19: Gemma Precision & Selectivity"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb19.py

# Create results directory
umask 000 && mkdir -p results/exp19

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 19_gemma_precision_and_selectivity_executed.ipynb \
    19_gemma_precision_and_selectivity.ipynb

echo ""
echo "=============================================="
echo "Exp 19 complete!"
echo "End time: $(date)"
echo "Output notebook: 19_gemma_precision_and_selectivity_executed.ipynb"
echo "Results: results/exp19/results.json"
echo "=============================================="
