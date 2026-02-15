#!/bin/bash
# Run Exp 16: Cross-Model Priming Replication (Gemma 3 4B)
#
# Tests whether value contamination via priming replicates on a different
# architecture. 5 conditions: bare, static_fact_trunc, random_trunc,
# oracle_trunc, values_only.
# ~300 queries × ~8 passages × (4 forward + 5 scoring) ≈ 1.5-2.5h on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 16: Cross-Model Priming (Gemma 3 4B)"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb16.py

# Create results directory
umask 000 && mkdir -p results/exp16

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 16_cross_model_gemma3_executed.ipynb \
    16_cross_model_gemma3.ipynb

echo ""
echo "=============================================="
echo "Exp 16 complete!"
echo "End time: $(date)"
echo "Output notebook: 16_cross_model_gemma3_executed.ipynb"
echo "Results: results/exp16/results.json"
echo "=============================================="
