#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/17_taskaware
while kill -0 1587471 2>/dev/null; do sleep 30; done
echo "=== prior chain done; starting exp33 single-fact shuffle $(date) ==="
python3 run_singlefact_shuffle.py
echo "=== exp33 DONE $(date) ==="
