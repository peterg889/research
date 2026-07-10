#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication/experiments/17_taskaware
# wait for the new-model shuffle chain to finish (marker in its log)
while ! grep -q "NEW-MODEL SHUFFLE RUNS DONE" /home/jupyter/research/directed_kvcache_publication/taskaware_newmodels.log 2>/dev/null; do sleep 30; done
echo "=== new-model chain done; starting reranking shuffle $(date) ==="
cd /home/jupyter/research/directed_kvcache_publication/experiments/13_contrastive
export ONLY_MODELS="gemma3_4b,gemma3_12b,gemma3_27b"
export CONDS="bare,generic,tfidf_plain,tfidf_shuffled"
export RESULTS_NAME="exp34_rerank_shuffle"
export N_EVAL=900
python3 run_contrastive_ablations.py
echo "=== RERANK SHUFFLE DONE $(date) ==="
