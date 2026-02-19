#!/bin/bash
# Run Exp 31: Ad-Content Benchmark & Generation Quality (Gemma 3 4B)
#
# Phase 1: NLL evaluation on short-passage QA datasets
#   - Amazon SubjQA (electronics + grocery): product review QA
#   - MS MARCO short (<250 words): known positive control
#   - SQuAD short (<250 words): extractive QA baseline
#
# Phase 2: Generation quality on hard subset (bare vs hero)
#   - Greedy generation with bare and hero caches
#   - Metrics: Exact Match, Token F1, ROUGE-L, generation confidence
#
# 4 conditions (bare, sf_trunc, values_early, values_hero)
# MAX_DOC_TOKENS = 900 (sliding window constraint)
# Expected runtime: ~4-6 hours on L4

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jupyter/research/directed_kvcache_v2

echo "=============================================="
echo "Exp 31: Ad-Content Benchmark & Generation Quality (Gemma 3 4B)"
echo "Start time: $(date)"
echo "=============================================="

# Build the notebook
umask 000 && python3 scripts/build_nb31.py

# Create results directory
umask 000 && mkdir -p results/exp31

# Execute the notebook
echo "Executing notebook..."
umask 000 && jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=36000 \
    --output 31_ad_benchmark_and_generation_executed.ipynb \
    31_ad_benchmark_and_generation.ipynb

echo ""
echo "=============================================="
echo "Exp 31 complete!"
echo "End time: $(date)"
echo "Output notebook: 31_ad_benchmark_and_generation_executed.ipynb"
echo "Results: results/exp31/results.json"
echo "NLL CSV: results/exp31/results.csv"
echo "Gen CSV: results/exp31/gen_results.csv"
echo "=============================================="
