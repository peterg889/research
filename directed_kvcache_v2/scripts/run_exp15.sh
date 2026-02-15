#!/bin/bash
# Run Exp 15: NLL Ensemble Ranking
#
# Tests whether diverse priming caches improve ranking via NLL ensembling.
# 5 signals: bare, rescore (control), sf, rand, intent
# ~300 queries × ~8 passages × (4 forward + 5 scoring) ≈ 2-3h on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 15: NLL Ensemble Ranking"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb15.py

# Create results directory
umask 000 && mkdir -p results/exp15

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 15_nll_ensemble_ranking_executed.ipynb \
    15_nll_ensemble_ranking.ipynb

echo ""
echo "=============================================="
echo "Exp 15 complete!"
echo "End time: $(date)"
echo "Output notebook: 15_nll_ensemble_ranking_executed.ipynb"
echo "Results: results/exp15/results.json"
echo "=============================================="
