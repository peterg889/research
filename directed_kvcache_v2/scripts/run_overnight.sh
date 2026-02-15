#!/bin/bash
# run_overnight.sh — Run Exp 04 + Exp 05 sequentially overnight
# Usage: nohup bash scripts/run_overnight.sh > overnight.log 2>&1 &
#
# Exp 04: Cross-dataset semantic priming battery (~7 hours)
#   - MS MARCO multi-passage, SQuAD 2.0, Natural Questions, TriviaQA
#   - 4 conditions × 4 datasets
#
# Exp 05: Hardness-gated + LLM surrogates (~3.5 hours)
#   - Two-pass: bare NLL scoring → filter hard → LLM surrogate generation → eval
#   - 5 conditions on hardest 2000 MS MARCO samples
#
# Total estimated runtime: ~10.5 hours on single L4 GPU

set -e
umask 000

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Overnight experiment run started: $(date)"
echo "=============================================="

# Create results directories
mkdir -p results/exp04 results/exp05

echo ""
echo "=== Starting Exp 04: Cross-Dataset Battery ==="
echo "Start time: $(date)"
echo ""

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    04_semantic_priming_battery.ipynb \
    --output 04_semantic_priming_battery_executed.ipynb

echo ""
echo "=== Exp 04 finished: $(date) ==="
echo ""

# Brief pause to let any residual CUDA processes clean up
echo "Waiting 10s for GPU memory cleanup..."
sleep 10

echo "=== Starting Exp 05: Hardness-Gated + LLM Surrogates ==="
echo "Start time: $(date)"
echo ""

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    05_hardness_gated_surrogates.ipynb \
    --output 05_hardness_gated_surrogates_executed.ipynb

echo ""
echo "=== Exp 05 finished: $(date) ==="
echo ""

echo "=============================================="
echo "All experiments completed: $(date)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  results/exp04/results.json"
echo "  results/exp05/results.json"
