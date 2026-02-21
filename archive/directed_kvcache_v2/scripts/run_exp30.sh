#!/bin/bash
# Run Exp 30: Retrieval vs Reasoning Task-Type Dissociation (Gemma 3 4B)
#
# Tests whether task type (retrieval vs computation) predicts hero layer effect
# beyond difficulty. Three datasets:
#   1. NQ — retrieval (factoid), known positive control
#   2. DROP — mixed (computation + extraction), known negative
#   3. BoolQ — retrieval (binary judgment), new data point
#
# Only 4 conditions (bare, sf_trunc, values_early, values_hero)
# MAX_DOC_TOKENS = 900 (sliding window constraint)
# Expected runtime: ~4-5 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 30: Retrieval vs Reasoning (Gemma 3 4B)"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb30.py

# Create results directory
umask 000 && mkdir -p results/exp30

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 30_retrieval_vs_reasoning_executed.ipynb \
    30_retrieval_vs_reasoning.ipynb

echo ""
echo "=============================================="
echo "Exp 30 complete!"
echo "End time: $(date)"
echo "Output notebook: 30_retrieval_vs_reasoning_executed.ipynb"
echo "Results: results/exp30/results.json"
echo "CSV: results/exp30/results.csv"
echo "=============================================="
