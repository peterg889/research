#!/bin/bash
# Run Exp 27b: Cross-Dataset Generalization on Gemma 3 4B
#
# Tests whether the Gemma-specific toolkit (values-early-layers, hero layers)
# generalizes beyond MS MARCO to TriviaQA, NQ, and HotpotQA.
# MAX_DOC_TOKENS = 900 (sliding window constraint)
# Expected runtime: ~5-7 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 27b: Cross-Dataset Gemma 3 4B"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb27b.py

# Create results directory
umask 000 && mkdir -p results/exp27b

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 27b_cross_dataset_gemma_executed.ipynb \
    27b_cross_dataset_gemma.ipynb

echo ""
echo "=============================================="
echo "Exp 27b complete!"
echo "End time: $(date)"
echo "Output notebook: 27b_cross_dataset_gemma_executed.ipynb"
echo "Results: results/exp27b/results.json"
echo "CSV: results/exp27b/results.csv"
echo "=============================================="
