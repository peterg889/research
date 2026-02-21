#!/bin/bash
# Run Exp 26: Attention Forcing for Long Documents
#
# Tests whether adding a positive logit bias to doc->prefix attention
# during cache building can recover priming benefit at 1024 tokens.
# Conditions: bare, bias=0.0, bias=2.0, bias=5.0, bias=10.0
# Expected runtime: ~1-2 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 26: Attention Forcing for Long Documents"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb26.py

# Create results directory
umask 000 && mkdir -p results/exp26

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 26_attention_forcing_executed.ipynb \
    26_attention_forcing.ipynb

echo ""
echo "=============================================="
echo "Exp 26 complete!"
echo "End time: $(date)"
echo "Output notebook: 26_attention_forcing_executed.ipynb"
echo "Results: results/exp26/results.json"
echo "CSV: results/exp26/results.csv"
echo "=============================================="
