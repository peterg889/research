#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/13_contrastive
export ONLY_MODELS="gemma3_12b,gemma3_4b,gemma3_27b"
export CONDS="tfidf_shuffled"
export RESULTS_NAME="exp34_rerank_shuffle"
export N_EVAL=900
echo "=== rerank tfidf_shuffled-only (pairs vs exp14c_highN) $(date) ==="
python3 run_contrastive_ablations.py
echo "=== RERANK SHUFFLE DONE $(date) ==="
