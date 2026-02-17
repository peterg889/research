#!/bin/bash
# Run Exp 27: Cross-Dataset Generalization with Attention Forcing
#
# Tests whether attention forcing (from Exp 26) generalizes beyond MS MARCO
# to TriviaQA, Natural Questions, and HotpotQA.
# Conditions: bare, sf_trunc, sf_trunc_bias2, sf_trunc_bias4, values_only
# Expected runtime: ~5 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 27: Cross-Dataset Attention Forcing"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb27.py

# Create results directory
umask 000 && mkdir -p results/exp27

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 27_cross_dataset_attention_forcing_executed.ipynb \
    27_cross_dataset_attention_forcing.ipynb

echo ""
echo "=============================================="
echo "Exp 27 complete!"
echo "End time: $(date)"
echo "Output notebook: 27_cross_dataset_attention_forcing_executed.ipynb"
echo "Results: results/exp27/results.json"
echo "CSV: results/exp27/results.csv"
echo "=============================================="
