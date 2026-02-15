#!/bin/bash
# Run Exp 14: Ranking-Aware Priming — Does Priming Improve Ad Ranking?
#
# Uses ALL passages per query from MS MARCO v1.1 validation.
# Tests whether priming creates a differential effect that improves ranking.
#
# Expected runtime: ~60-90 min on NVIDIA L4
# Forward passes: ~300 queries × ~8 passages × 2 = ~4800 forward + ~4800 scoring

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 14: Ranking-Aware Priming"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb14.py

# Create results directory
umask 000 && mkdir -p results/exp14

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 14_ranking_aware_priming_executed.ipynb \
    14_ranking_aware_priming.ipynb

echo ""
echo "=============================================="
echo "Exp 14 complete!"
echo "End time: $(date)"
echo "Output notebook: 14_ranking_aware_priming_executed.ipynb"
echo "Results: results/exp14/results.json"
echo "=============================================="
