#!/usr/bin/env python3
"""Deterministic regeneration of every contested headline number from results/*.json.
Source of truth for the paper (fixes the stale-interim class of error the review found).
Run: cd <repo> && PYTHONPATH="../directed_kvcache_v4:." python3 make_numbers.py"""
import json, glob, numpy as np
from pathlib import Path
RES = Path("results")
def load(p):
    f = RES / p
    return json.loads(f.read_text())["samples"] if f.exists() else None
def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
def f(t): return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]{'*' if (t[1]>0 or t[2]<0) else ' '}"

print("="*78); print("DETERMINISTIC NUMBERS REGEN — source of truth for the paper"); print("="*78)

# 1. BoolQ: gold-aligned margin split (prior-shift vs sharpening) + balanced accuracy
print("\n## 1. BoolQ (exp10): extract vs bare — gold-aligned Δmargin by class, + balanced-acc")
def isyes(g): return str(g).lower() in ('yes','true','1')
for d in sorted(glob.glob("results/exp10_boolq/*/")):
    S = json.loads((Path(d)/"results.json").read_text())["samples"]; m = Path(d).name
    dmy, dmn = [], []
    for s in S:
        if isyes(s['gold']):
            dmy.append((s['extract__yes']-s['extract__no'])-(s['bare__yes']-s['bare__no']))
        else:
            dmn.append((s['extract__no']-s['extract__yes'])-(s['bare__no']-s['bare__yes']))
    # balanced accuracy (argmax yes/no) for bare vs extract
    def bal(cond):
        yacc=np.mean([ (s[f'{cond}__yes']>s[f'{cond}__no'])==isyes(s['gold']) for s in S if isyes(s['gold'])])
        nacc=np.mean([ (s[f'{cond}__no']>s[f'{cond}__yes'])==(not isyes(s['gold'])) for s in S if not isyes(s['gold'])])
        return 0.5*(yacc+nacc)
    print(f"  {m:14s} gold=yes {f(boot(dmy))}  gold=no {f(boot(dmn))}  balAcc bare={bal('bare'):.3f}->extract={bal('extract'):.3f}")

# 2. exp05 tfidf / generic margin per model (sign heterogeneity)
print("\n## 2. exp05 discrimination: Δ(margin_mean) vs bare, PER MODEL (pooling masks signs)")
for d in sorted(glob.glob("results/exp05_discrimination/*/")):
    S=[];
    for ff in glob.glob(str(Path(d)/"*.json")): S+=json.loads(open(ff).read()).get("samples",[])
    if not S: continue
    t=boot([s['tfidf_16__margin_mean']-s['bare__margin_mean'] for s in S if np.isfinite(s.get('tfidf_16__margin_mean',np.nan))])
    g=boot([s['generic_instr_16__margin_mean']-s['bare__margin_mean'] for s in S if np.isfinite(s.get('generic_instr_16__margin_mean',np.nan))])
    print(f"  {Path(d).name:14s} tfidf16 {f(t)}   generic16 {f(g)}")

# 3. content-type banking: semantic + code across all models (exp26) + spectrum (exp27)
print("\n## 3. banking: SEMANTIC (exp26, all models) — neg=banks")
for m in ["qwen25_1_5b","qwen25_7b","qwen25_14b","mistral_7b","gemma3_1b","gemma3_4b","gemma3_12b","gemma3_27b"]:
    S=load(f"exp26_bank_semantic/{m}/results.json")
    if not S: continue
    sem=boot([s['sem_strip']-s['sem_neutral'] for s in S]); code=boot([s['code_strip']-s['code_neutral'] for s in S])
    print(f"  {m:14s} sem {f(sem)}   code {f(code)}")
print("## 3b. content spectrum (exp27, gemma12b vs qwen7b) — neg=banks")
for m in ["gemma3_12b","qwen25_7b"]:
    S=load(f"exp27_bank_types/{m}/results.json")
    if not S: continue
    print(f"  {m}: "+"  ".join(f"{t}={f(boot([s[f'{t}_strip']-s[f'{t}_neutral'] for s in S]))}" for t in ["code","pseudoword","rare_word","common_word","phrase"]))

# 4. imprintability x semantic banking r
print("\n## 4. imprintability (|Δ generic| from exp14) x semantic banking (exp26)  r")
prim={"qwen25_1_5b":0.195,"qwen25_7b":0.370,"qwen25_14b":0.391,"mistral_7b":0.552,"gemma3_1b":0.429,"gemma3_4b":0.597,"gemma3_12b":0.844,"gemma3_27b":0.842}
xs,ys=[],[]
for m,p in prim.items():
    S=load(f"exp26_bank_semantic/{m}/results.json")
    if not S: continue
    xs.append(p); ys.append(-np.mean([s['sem_strip']-s['sem_neutral'] for s in S]))
print(f"  Pearson r = {np.corrcoef(xs,ys)[0,1]:.3f}  (n={len(xs)})")

# 5. reranking vs bare ladder (exp14c)
print("\n## 5. reranking (exp14c N=900): tfidf_plain vs bare / vs generic")
def ranks(nl,ri): return 1+sum(1 for j,x in enumerate(nl) if j!=ri and x<nl[ri])
def mrr(S,c): return np.array([1.0/ranks(r[f"{c}__q"],r["relevant_idx"]) for r in S])
for m in ["gemma3_4b","gemma3_12b","gemma3_27b","qwen25_7b","qwen25_14b","mistral_7b"]:
    S=load(f"exp14c_highN/{m}/results.json")
    if not S: continue
    b,g,t=mrr(S,"bare"),mrr(S,"generic"),mrr(S,"tfidf_plain")
    print(f"  {m:12s} (n={len(S)}) tfidf-bare {f(boot(t-b))}  tfidf-gen {f(boot(t-g))}  gen-bare {f(boot(g-b))}")

# 6. base vs instruct flip (exp26)
print("\n## 6. base vs instruct (exp26): sem + code banking")
for base,inst in [("gemma3_4b_base","gemma3_4b"),("qwen25_7b_base","qwen25_7b"),("mistral_7b_base","mistral_7b")]:
    for m in [base,inst]:
        S=load(f"exp26_bank_semantic/{m}/results.json")
        if not S: print(f"  {m}: none"); continue
        print(f"  {m:18s} sem {f(boot([s['sem_strip']-s['sem_neutral'] for s in S]))}  code {f(boot([s['code_strip']-s['code_neutral'] for s in S]))}")

# 7. downstream QA (exp29) machinery-controlled content effect
print("\n## 7. downstream QA (exp29, machinery-controlled): q_primed - q_neutral (pos=hurts)")
for m in ["gemma3_12b","gemma3_4b","qwen25_7b"]:
    S=load(f"exp29_downstream_qa/{m}/results.json")
    if not S: continue
    print(f"  {m:12s} content {f(boot([s['q_primed']-s['q_neutral'] for s in S]))}  machinery {f(boot([s['q_neutral']-s['bare'] for s in S]))}")
print("\n"+"="*78+"\nDONE\n"+"="*78)
