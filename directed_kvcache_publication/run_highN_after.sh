#!/bin/bash
cd /home/jupyter/research/directed_kvcache_publication
export $(grep HF_TOKEN /home/jupyter/research/.env)
export PYTHONPATH="../directed_kvcache_v4:."
# wait for the current audit pipeline (exp14b 4 models + exp21b) to fully finish
while ! grep -q "ALL AUDIT FOLLOW-UPS DONE" audit_followups.log 2>/dev/null; do sleep 60; done
echo "high-N vs-bare launching $(date)" > highN.log
# decisive vs-bare test across the full Gemma ladder + qwen control.
# 12b first (confirm the borderline n=300 result soonest); 27b last (slow/large).
RESULTS_NAME="exp14c_highN" N_EVAL=900 CONDS="bare,generic,tfidf_plain" \
  ONLY_MODELS="gemma3_12b,gemma3_4b,gemma3_27b,qwen25_7b" \
  python3 -u experiments/13_contrastive/run_contrastive_ablations.py >> highN.log 2>&1
echo "HIGH-N DONE $(date)" >> highN.log
