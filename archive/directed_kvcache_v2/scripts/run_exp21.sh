#!/bin/bash
# Run Exp 21: Gemma Mechanism Robustness & Tuning
#
# Part 1: Length generalization test (N=500, 4 pad lengths, cutoff=16)
# Part 2: Layer boundary sweep (N=200, 5 cutoffs, no padding)
# Expected runtime: ~2-3 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 21: Gemma Mechanism Robustness & Tuning"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb21.py

# Create results directory
umask 000 && mkdir -p results/exp21

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 21_gemma_robustness_tuning_executed.ipynb \
    21_gemma_robustness_tuning.ipynb

echo ""
echo "=============================================="
echo "Exp 21 complete!"
echo "End time: $(date)"
echo "Output notebook: 21_gemma_robustness_tuning_executed.ipynb"
echo "Results: results/exp21/results.json"
echo "CSVs: results/exp21/part1_results.csv, results/exp21/part2_results.csv"
echo "=============================================="
