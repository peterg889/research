#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/17_taskaware
export ONLY_MODELS="llama3_8b,yi15_9b,falcon3_7b"
echo "=== R2 exp33 single-fact (llama3/yi/falcon) $(date) ==="
python3 run_singlefact_shuffle.py
echo "=== R2 exp32 binding $(date) ==="
python3 run_binding_shuffle.py
echo "=== ROUND-2 SHUFFLE DONE $(date) ==="
