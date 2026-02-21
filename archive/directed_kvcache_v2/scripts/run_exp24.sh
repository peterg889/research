#!/bin/bash
# Run Exp 24: Gemma Layer-Selective Mechanism Deep Dive
#
# Part 1+4: Individual layer contribution map + value features (MS MARCO, N=300)
# Part 2:   Cross-dataset SQuAD v2 (N=400)
# Part 3:   Prefix content x layer selectivity (MS MARCO, N=300)
# Expected runtime: ~2.5-3 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 24: Gemma Layer-Selective Mechanism Deep Dive"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb24.py

# Create results directory
umask 000 && mkdir -p results/exp24

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 24_gemma_layer_mechanism_executed.ipynb \
    24_gemma_layer_mechanism.ipynb

echo ""
echo "=============================================="
echo "Exp 24 complete!"
echo "End time: $(date)"
echo "Output notebook: 24_gemma_layer_mechanism_executed.ipynb"
echo "Results: results/exp24/results.json"
echo "CSVs: results/exp24/part1_layer_map.csv"
echo "       results/exp24/part2_squad.csv"
echo "       results/exp24/part3_prefix_content.csv"
echo "       results/exp24/part4_value_features.csv"
echo "=============================================="
