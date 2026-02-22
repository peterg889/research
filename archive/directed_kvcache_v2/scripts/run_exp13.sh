#!/bin/bash
# Run Exp 13: Position-Aware Value Contamination for Long Documents
#
# Uses same NQ samples as exp 12. Tests position-selective contamination
# strategies motivated by the answer-position diagnostic finding.
#
# Expected runtime: ~3-4 hours on NVIDIA L4
# Forward passes: ~630 (2 per sample) + ~3000 scoring = ~3600 total

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 13: Position-Aware Priming"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb13.py

# Create results directory
umask 000 && mkdir -p results/exp13

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 13_position_aware_priming_executed.ipynb \
    13_position_aware_priming.ipynb

echo ""
echo "=============================================="
echo "Exp 13 complete!"
echo "End time: $(date)"
echo "Output notebook: 13_position_aware_priming_executed.ipynb"
echo "Results: results/exp13/results.json"
echo "=============================================="
