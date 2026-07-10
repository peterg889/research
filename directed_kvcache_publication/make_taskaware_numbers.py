#!/usr/bin/env python3
"""Single source of truth for the task-aware / shuffle-control experiments (exp31a/b, 32, 33).
Run: cd <repo> && PYTHONPATH="../directed_kvcache_v4:." python3 make_taskaware_numbers.py"""
import json, numpy as np
from pathlib import Path
RES = Path("results")
def S(p):
    f = RES / p / "results.json"
    return json.loads(f.read_text())["samples"] if f.exists() else None
def boot(d, n=4000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
def f(t): return f"{t[0]:+.3f}[{t[1]:+.3f},{t[2]:+.3f}]" + ("*" if (t[1] > 0 or t[2] < 0) else " ")

print("="*82); print("TASK-AWARE + SHUFFLE-CONTROL NUMBERS — source of truth"); print("="*82)

print("\n## exp31a QA: sem(ordered Q) vs surf(shuffled Q) prime, machinery-controlled (pos=hurts)")
for m in ["gemma3_12b","gemma3_4b","qwen25_7b","qwen25_14b"]:
    s = S(f"exp31_taskaware_qa/{m}")
    if not s: continue
    print(f"  {m:11s} n={len(s)} semC={f(boot([x['sem_prime']-x['neutral'] for x in s]))} "
          f"surfC={f(boot([x['surf_prime']-x['neutral'] for x in s]))} "
          f"order={f(boot([x['sem_prime']-x['surf_prime'] for x in s]))}")

print("\n## exp31b QA: conditioning vs SELECTION (k=32, 8 models). pos=hurts. COND|sel<0 => conditioning helps given selection")
print("   imprintability does NOT cleanly predict the direction; ans-surv = answer-span survival under top-k selection")
IMP = {"qwen25_1_5b":0.20,"qwen25_3b":0.30,"qwen25_7b":0.37,"qwen25_14b":0.39,
       "gemma3_1b":0.43,"mistral_7b":0.55,"gemma3_4b":0.60,"gemma3_12b":0.84}
import numpy as _np
_pv, _cs, _imp = [], [], []
for m in ["qwen25_1_5b","qwen25_3b","qwen25_7b","qwen25_14b","gemma3_1b","mistral_7b","gemma3_4b","gemma3_12b"]:
    s = S(f"exp31_taskaware_select/{m}")
    if not s: continue
    pv = boot([x['prime_full']-x['bare_norm'] for x in s]); cs = boot([x['sel_k_primed']-x['sel_k_plain'] for x in s])
    sv = boot([x['sel_k_plain']-x['bare_norm'] for x in s])
    surv = sum(x.get('a_in_sel',0) for x in s) / max(sum(x.get('a_span',0) for x in s), 1)
    print(f"  {m:12s} imp={IMP[m]:.2f} n={len(s)} primeVal={f(pv)} COND|sel={f(cs)} selVal={f(sv)} ans-surv={surv:.2f}")
    _pv.append(pv[0]); _cs.append(cs[0]); _imp.append(IMP[m])
if len(_imp) >= 4:
    print(f"  -> r(imprintability,primeVal)={_np.corrcoef(_imp,_pv)[0,1]:.3f}  r(imprintability,COND|sel)={_np.corrcoef(_imp,_cs)[0,1]:.3f}  (NO clean trait law)")

print("\n## exp32 binding (2-fact): ORDER=strip_ord-strip_shuf. neg=ordered banks more (structure); pos=shuffled (tokens)")
for m in ["gemma3_4b","gemma3_12b","gemma3_27b","mistral_7b","ministral_8b","olmo2_7b","deepseek_r1_qwen7b","qwen25_7b"]:
    s = S(f"exp32_binding_shuffle/{m}")
    if not s: continue
    print(f"  {m:11s} n={len(s)} bankOrd={f(boot([x['strip_ord']-x['neutral'] for x in s]))} "
          f"bankShuf={f(boot([x['strip_shuf']-x['neutral'] for x in s]))} "
          f"ORDER={f(boot([x['strip_ord']-x['strip_shuf'] for x in s]))}")

print("\n## exp33 single-fact: SEM & CODE banking, ORDER=ord-shuf. neg=meaning/structure; ~0=token presence")
for m in ["gemma3_4b","gemma3_12b","gemma3_27b","mistral_7b","ministral_8b","olmo2_7b","deepseek_r1_qwen7b","qwen25_7b"]:
    s = S(f"exp33_singlefact_shuffle/{m}")
    if not s: continue
    semO = boot([x['sem_ord']-x['sem_shuf'] for x in s]); codO = boot([x['code_ord']-x['code_shuf'] for x in s])
    semBank = boot([x['sem_ord']-x['sem_neutral'] for x in s])
    print(f"  {m:11s} n={len(s)} SEMbank={f(semBank)} SEM_ORDER={f(semO)} CODE_ORDER={f(codO)}")

print("\n" + "="*82 + "\nDONE\n" + "="*82)
