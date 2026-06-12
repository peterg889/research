#!/usr/bin/env python3
"""v5 figures: imprinting-mode finding. Generates from results/ to figures/."""
import os
os.umask(0o000)
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True, mode=0o777)
RNG = np.random.RandomState(0)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": "--"})
GEMMA="#0b7a75"; QWEN="#d1495b"; MISTRAL="#8338ec"
def famc(m): return GEMMA if m.startswith("gemma") else (QWEN if m.startswith("qwen") else MISTRAL)
SHORT={"gemma3_1b":"G-1B","gemma3_4b":"G-4B","gemma3_12b":"G-12B","gemma3_27b":"G-27B",
       "mistral_7b":"Mistral-7B","qwen25_1_5b":"Q-1.5B","qwen25_7b":"Q-7B","qwen25_14b":"Q-14B"}
PRIM={"qwen25_1_5b":0.195,"qwen25_7b":0.370,"qwen25_14b":0.391,"mistral_7b":0.552,
      "gemma3_1b":0.429,"gemma3_4b":0.597,"gemma3_12b":0.844,"gemma3_27b":0.842}
def sembank(m):
    p=RES/f"exp26_bank_semantic/{m}/results.json"
    if not p.exists(): return None
    S=json.loads(p.read_text())["samples"]
    return -np.mean([s["sem_strip"]-s["sem_neutral"] for s in S])  # positive = banks

# Fig 7: imprintability x semantic banking (r=0.94)
def fig7():
    xs,ys,ms=[],[],[]
    for m in PRIM:
        b=sembank(m)
        if b is None: continue
        xs.append(PRIM[m]); ys.append(b); ms.append(m)
    xs,ys=np.array(xs),np.array(ys); r=np.corrcoef(xs,ys)[0,1]
    fig,ax=plt.subplots(figsize=(7,5.2))
    for x,y,m in zip(xs,ys,ms):
        ax.scatter(x,y,s=130,color=famc(m),edgecolor="k",lw=0.6,zorder=3)
        ax.annotate(SHORT[m],(x,y),textcoords="offset points",xytext=(7,4),fontsize=8.5)
    a,b=np.polyfit(xs,ys,1); xr=np.linspace(xs.min(),xs.max(),10)
    ax.plot(xr,a*xr+b,"k--",lw=1,alpha=0.6)
    ax.axhline(0,color="k",lw=0.7)
    ax.set_xlabel("imprintability  (mean |Δ query-NLL| from a generic prefix)")
    ax.set_ylabel("semantic banking  (nats recovered from stripped KV)")
    ax.set_title(f"One trait predicts semantic banking  (Pearson r = {r:.2f})")
    for c,l in [(GEMMA,"Gemma 3"),(QWEN,"Qwen 2.5"),(MISTRAL,"Mistral")]:
        ax.plot([],[],"o",color=c,label=l)
    ax.legend(fontsize=9,loc="upper left")
    fig.savefig(FIG/"fig7_imprintability_unification.png"); plt.close(fig); print("fig7 ok")

# Fig 8: content-type double dissociation
def fig8():
    RNG2=np.random.RandomState(1)
    def boot(d,n=3000):
        d=np.asarray(d,float); idx=RNG2.randint(0,len(d),(n,len(d)))
        return d.mean(), *np.percentile(d[idx].mean(1),[2.5,97.5])
    TYPES=["code","pseudoword","rare_word","common_word","phrase"]
    LAB=["code","pseudo-\nword","rare\nword","common\nword","phrase"]
    fig,ax=plt.subplots(figsize=(8,4.6)); x=np.arange(len(TYPES)); w=0.38
    for off,m,c in [(-w/2,"gemma3_12b",GEMMA),(w/2,"qwen25_7b",QWEN)]:
        S=json.loads((RES/f"exp27_bank_types/{m}/results.json").read_text())["samples"]
        vals=[boot([s[f"{t}_strip"]-s[f"{t}_neutral"] for s in S]) for t in TYPES]
        # plot banking magnitude = -content (positive = banks)
        means=[-v[0] for v in vals]; err=[[v[2]-v[0] for v in vals],[v[0]-v[1] for v in vals]]
        ax.bar(x+off,means,w,yerr=err,color=c,alpha=0.9,error_kw=dict(lw=1,capsize=2),
               label=("Gemma 12B (meaning)" if m.startswith("gemma") else "Qwen 7B (surface)"))
    ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(LAB)
    ax.set_ylabel("banking  (nats; higher = banks more)")
    ax.set_title("Content-type double dissociation: Gemma banks MEANING, Qwen banks SURFACE")
    ax.annotate("meaningless",xy=(0.5,ax.get_ylim()[1]*0.92),fontsize=8,ha="center",color="gray")
    ax.annotate("meaningful",xy=(3,ax.get_ylim()[1]*0.92),fontsize=8,ha="center",color="gray")
    ax.legend(fontsize=9)
    fig.savefig(FIG/"fig8_content_dissociation.png"); plt.close(fig); print("fig8 ok")

# Fig 9: semantic banking scales with Gemma size (% of context value)
def fig9():
    ladder=["gemma3_1b","gemma3_4b","gemma3_12b","gemma3_27b"]; sizes=[1,4,12,27]
    pct=[]
    for m in ladder:
        S=json.loads((RES/f"exp26_bank_semantic/{m}/results.json").read_text())["samples"]
        bank=np.mean([s["sem_strip"]-s["sem_neutral"] for s in S])
        val=np.mean([s["sem_bare"]-s["sem_retain"] for s in S])
        pct.append(-bank/val*100)
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    ax.plot(sizes,pct,"-o",color=GEMMA,lw=2,ms=9)
    for s,p,m in zip(sizes,pct,ladder): ax.annotate(f"{p:.0f}%",(s,p),textcoords="offset points",xytext=(6,5),fontsize=9)
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.set_xlabel("Gemma 3 model size"); ax.set_ylabel("semantic context value banked (%)")
    ax.set_title("Semantic imprinting scales with model size")
    ax.set_ylim(0,40)
    fig.savefig(FIG/"fig9_semantic_scaling.png"); plt.close(fig); print("fig9 ok")

# Fig 10: downstream mode-task 2x2
def fig10():
    # reranking dcorp-vs-bare (exp14c N=900) and QA content effect (exp29)
    def rerank_vs_bare(m):
        p=RES/f"exp14c_highN/{m}/results.json"
        if not p.exists(): return None
        S=json.loads(p.read_text())["samples"]
        def ranks(nl,ri): return 1+sum(1 for j,x in enumerate(nl) if j!=ri and x<nl[ri])
        def mrr(c): return np.mean([1.0/ranks(r[f"{c}__q"],r["relevant_idx"]) for r in S])
        return mrr("tfidf_plain")-mrr("bare")
    def qa_content(m):
        p=RES/f"exp29_downstream_qa/{m}/results.json"
        if not p.exists(): return None
        S=json.loads(p.read_text())["samples"]
        return np.mean([s["q_primed"]-s["q_neutral"] for s in S])  # positive = hurts
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
    # left: reranking (positive MRR = helps)
    rr=[("gemma3_12b",rerank_vs_bare("gemma3_12b")),("gemma3_27b",rerank_vs_bare("gemma3_27b")),
        ("qwen25_7b",rerank_vs_bare("qwen25_7b"))]
    rr=[(m,v) for m,v in rr if v is not None]
    a1.bar([SHORT[m] for m,_ in rr],[v for _,v in rr],color=[famc(m) for m,_ in rr],alpha=0.9)
    a1.axhline(0,color="k",lw=0.8); a1.set_title("Relevance (reranking)\n↑ ΔMRR = helps")
    a1.set_ylabel("Δ MRR vs no-priming")
    # right: QA (negative content = helps; flip sign so up=helps)
    qa=[("gemma3_12b",qa_content("gemma3_12b")),("gemma3_4b",qa_content("gemma3_4b")),
        ("qwen25_7b",qa_content("qwen25_7b"))]
    qa=[(m,v) for m,v in qa if v is not None]
    a2.bar([SHORT[m] for m,_ in qa],[-v for _,v in qa],color=[famc(m) for m,_ in qa],alpha=0.9)
    a2.axhline(0,color="k",lw=0.8); a2.set_title("Extraction (QA)\n↑ = helps (−ΔNLL)")
    a2.set_ylabel("answer NLL improvement (content)")
    fig.suptitle("Mode–task match: semantic imprinting helps relevance not extraction; surface the reverse",y=1.03)
    fig.savefig(FIG/"fig10_mode_task.png"); plt.close(fig); print("fig10 ok")

# Fig 11: base-vs-instruct mode flip (the cause = instruction-tuning)
def fig11():
    RNG2=np.random.RandomState(2)
    def boot(d,n=3000):
        d=np.asarray(d,float); idx=RNG2.randint(0,len(d),(n,len(d)))
        return d.mean(), *np.percentile(d[idx].mean(1),[2.5,97.5])
    def bank(m,typ):
        p=RES/f"exp26_bank_semantic/{m}/results.json"
        if not p.exists(): return None
        S=json.loads(p.read_text())["samples"]
        return boot([s[f"{typ}_strip"]-s[f"{typ}_neutral"] for s in S])
    pairs=[("Gemma-4B","gemma3_4b_base","gemma3_4b"),
           ("Qwen-7B","qwen25_7b_base","qwen25_7b"),
           ("Mistral-7B","mistral_7b_base","mistral_7b")]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4),sharey=True)
    x=np.arange(len(pairs)); w=0.38
    for ax,typ,title in [(a1,"sem","SEMANTIC (meaning)"),(a2,"code","SURFACE (code)")]:
        for off,which,hatch,lab in [(-w/2,1,"","base (pretrained)"),(w/2,2,"//","instruct-tuned")]:
            means,err,cols=[],[[],[]],[]
            for _,b,i in pairs:
                v=bank(b if which==1 else i, typ)
                mag=-v[0]  # positive = banks
                means.append(mag); err[0].append(v[2]-v[0]); err[1].append(v[0]-v[1])
                cols.append("#888" if which==1 else "#1a73e8")
            ax.bar(x+off,means,w,yerr=err,color=cols,hatch=hatch,alpha=0.9,
                   error_kw=dict(lw=1,capsize=2),label=lab,edgecolor="k",lw=0.4)
        ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs])
        ax.set_title(title)
    a1.set_ylabel("banking  (nats; higher = banks more)"); a1.legend(fontsize=9,loc="upper right")
    a2.annotate("Qwen tuning\nFLIPS the mode",xy=(1+w/2,0.3),xytext=(1.3,2.0),fontsize=9,ha="left",
                arrowprops=dict(arrowstyle="->",lw=1))
    a1.annotate("Qwen tuning\nDESTROYS meaning",xy=(1+w/2,0.1),xytext=(0.0,2.2),fontsize=9,ha="left",
                arrowprops=dict(arrowstyle="->",lw=1))
    fig.suptitle("Imprinting mode is set by INSTRUCTION-TUNING: all base models bank meaning; Qwen's tuning flips to surface",y=1.03,fontsize=11.5)
    fig.savefig(FIG/"fig11_base_vs_instruct.png"); plt.close(fig); print("fig11 ok")

if __name__=="__main__":
    fig7(); fig8(); fig9(); fig10(); fig11()
    print("v5 figures in", FIG)
