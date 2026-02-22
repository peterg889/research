#!/bin/bash
# Run Exp 29: Cross-Dataset Generalization on Hard QA Datasets (Gemma 3 4B)
#
# Tests whether the Gemma-specific toolkit (values-early-layers, hero layers)
# generalizes to datasets with NO ceiling effects:
#   1. DROP — numerical/discrete reasoning
#   2. AdversarialQA — adversarially hard extractive QA
#   3. CoQA — abstractive conversational QA
#
# MAX_DOC_TOKENS = 900 (sliding window constraint)
# Expected runtime: ~3-5 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 29: Hard QA Datasets (Gemma 3 4B)"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb29.py

# Create results directory
umask 000 && mkdir -p results/exp29

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 29_hard_datasets_gemma_executed.ipynb \
    29_hard_datasets_gemma.ipynb

echo ""
echo "=============================================="
echo "Exp 29 complete!"
echo "End time: $(date)"
echo "Output notebook: 29_hard_datasets_gemma_executed.ipynb"
echo "Results: results/exp29/results.json"
echo "CSV: results/exp29/results.csv"
echo "=============================================="
