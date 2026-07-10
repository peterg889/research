#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/17_taskaware
export ONLY_MODELS="olmo2_7b,ministral_8b,deepseek_r1_qwen7b"
echo "=== exp33 single-fact shuffle (new models) $(date) ==="
python3 run_singlefact_shuffle.py
echo "=== exp32 binding shuffle (new models) $(date) ==="
python3 run_binding_shuffle.py
echo "=== NEW-MODEL SHUFFLE RUNS DONE $(date) ==="
