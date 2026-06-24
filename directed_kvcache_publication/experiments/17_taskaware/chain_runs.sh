#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/17_taskaware
# wait for the QA run to finish releasing the GPU
while kill -0 1584826 2>/dev/null; do sleep 30; done
echo "=== QA run finished; starting exp32 binding shuffle $(date) ===" 
python3 run_binding_shuffle.py
echo "=== exp32 done; starting exp31b selection $(date) ==="
python3 run_taskaware_select.py
echo "=== ALL TASK-AWARE RUNS DONE $(date) ==="
