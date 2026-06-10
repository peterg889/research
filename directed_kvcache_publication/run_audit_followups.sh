#!/bin/bash
# Audit double-down pipeline: fig1 regen -> leakage quant -> exp14b smoke ->
# exp14b full (gemma3_4b -> gemma3_12b -> qwen25_7b -> gemma3_4b_base) -> exp21b coherence v2.
# Logs: audit_followups.log (this), exp14b.log, exp21b.log
set -u
cd /home/jupyter/research/directed_kvcache_publication
export $(grep HF_TOKEN /home/jupyter/research/.env)
export PYTHONPATH="../directed_kvcache_v4:."
LOG=audit_followups.log
echo "=== audit follow-ups started $(date) ===" > $LOG

echo "--- [1/5] syntax checks ---" >> $LOG
python3 -c "import ast; ast.parse(open('experiments/13_contrastive/run_contrastive_ablations.py').read()); ast.parse(open('experiments/13_contrastive/coherence_probe2.py').read()); print('syntax OK')" >> $LOG 2>&1 || exit 1

echo "--- [2/5] regenerate fig1 (corrected exp05 values) ---" >> $LOG
python3 make_figures.py >> $LOG 2>&1

echo "--- [3/5] leakage quantification (CPU) ---" >> $LOG
python3 - >> $LOG 2>&1 <<'EOF'
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from lib.data import count_words
N_EVAL=300
ds = load_dataset("microsoft/ms_marco","v2.1",split="validation")
samples=[]
for x in ds:
    pt=x["passages"]["passage_text"]; sel=x["passages"]["is_selected"]
    if sum(sel)!=1: continue
    rel=sel.index(1)
    if not (5<=count_words(pt[rel])<=300): continue
    samples.append({"query":x["query"],"passages":pt,"relevant_idx":rel})
    if len(samples)>=N_EVAL: break
all_passages=[]; offsets=[]
for s in samples:
    offsets.append(len(all_passages)); all_passages.extend(s["passages"])
vec=TfidfVectorizer(stop_words="english",max_features=50000)
mat=vec.fit_transform(all_passages); feats=vec.get_feature_names_out()
def topterms(idx, nb, topk=10):
    v=mat[idx].toarray().ravel()
    comp=mat[nb].toarray().mean(axis=0) if len(nb) else np.zeros_like(v)
    d=v-comp; top=np.argsort(-d)[:topk]
    return set(feats[t] for t in top if d[t]>0)
import random; random.seed(0)
leak=[]; jac=[]
for qi in random.sample(range(len(samples)),60):
    s=samples[qi]; base=offsets[qi]; ci=s["relevant_idx"]; g=base+ci
    sims=(mat@mat[g].T).toarray().ravel(); sims[g]=-1
    nb=np.argsort(-sims)[:10]
    same=set(range(base,base+len(s["passages"])))
    leak.append(sum(1 for j in nb if j in same)/10)
    tc=topterms(g,nb); others=np.array([base+j for j in range(len(s["passages"])) if j!=ci])
    td=topterms(g,others)
    if tc|td: jac.append(len(tc&td)/len(tc|td))
print(f"LEAKAGE: same-query fraction of top-10 corpus neighbors: mean={np.mean(leak):.2f} median={np.median(leak):.2f}  >=50%: {np.mean(np.array(leak)>=0.5)*100:.0f}%")
print(f"TERM OVERLAP dist_corpus vs dist_cand (Jaccard): {np.mean(jac):.2f}")
EOF

echo "--- [4/5] exp14b smoke (gemma3_4b, N=10) ---" >> $LOG
rm -rf results/exp14b_ablations/gemma3_4b
SMOKE=1 ONLY_MODELS="gemma3_4b" timeout 600 python3 experiments/13_contrastive/run_contrastive_ablations.py 2>&1 | grep -vE "it/s|Loading|Fetching|%\|" | tail -6 >> $LOG
rm -rf results/exp14b_ablations/gemma3_4b

echo "--- [5/5] launching exp14b full + exp21b chained ---" >> $LOG
( ONLY_MODELS="gemma3_4b,gemma3_12b,qwen25_7b,gemma3_4b_base" python3 -u experiments/13_contrastive/run_contrastive_ablations.py > exp14b.log 2>&1
  echo "exp14b finished $(date)" >> $LOG
  python3 -u experiments/13_contrastive/coherence_probe2.py > exp21b.log 2>&1
  echo "exp21b finished $(date)" >> $LOG
  echo "ALL AUDIT FOLLOW-UPS DONE $(date)" >> $LOG ) &
echo "background pipeline PID $! — monitor: tail -f exp14b.log" >> $LOG
echo "launcher done $(date)" >> $LOG
cat $LOG
