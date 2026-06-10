# Audit-driven revisions for paper v4 → v5 (2026-06-10)

End-to-end audit + double-down (exp14b ablations, exp21b position-matched coherence,
exp14c high-N vs-bare). Every change below is data-backed; citations to results/ dirs.
**Net: the behavioral core holds but is RENAMED and NARROWED; one mechanism claim is
retracted; several numbers corrected. The vs-bare bound is UNSETTLED — a small significant
win on gemma3_12b (+0.036*, N=900) but null on 4B; 27B tiebreaker re-running.**

## CONFIRMED / UNCHANGED
- Generic instruction priming HURTS reranking on capable models (high-N: generic vs bare
  −0.032* on gemma3_12b). [exp14c]
- Per-passage keyword priming is SELECTIVE on Gemma (Δneg−Δrel = +0.10*/+0.14* on 4B/12B;
  generic NOT selective on 4B). Gemma-specific (qwen control: no benefit). [exp14b]
- Primability is the empirical gate (Gemma ≫ Qwen/Mistral; not scale-monotonic). [exp14]
- §4 entropy-confound result (keyword NLL win is an entropy artifact, d(margin)≈0). [exp05]
- §7.5 layer-wise KV patching is position-matched by construction (reposition→1..D) — keep,
  but add an explicit position-matched control note. [exp20]

## CHANGE 1 — RENAME: "contrastive priming" → "keyword priming"  (§ title, abstract, §6, §7)
T1 ablation: dist_clean − tfidf_plain = −0.003 (4b), +0.003 (12b), −0.008 (4b_base),
−0.005 (qwen) — ALL n.s. The neighbor-subtraction that DEFINES "contrastive" is INERT.
The active ingredient is plain passage TF-IDF keyword priming. [exp14b]
→ Retitle; recast §6 around keyword priming; note contrast adds nothing as an explicit
  ablation result (it is informative that contrast is unnecessary).

## CHANGE 2 — CACHEABILITY: dist_corpus was leaky; use dist_clean  (§3.3, §6)
exp14's `distinctive_corpus` neighbor search did NOT exclude same-query BM25 candidates:
59% of top-10 neighbors are same-query (73% of passages ≥50% leaked); Jaccard vs the oracle
`distinctive_cand` = 0.85. So it was ~85% the oracle, not query-agnostic. [leakage quant]
→ Report the leak honestly; the genuinely cacheable variant is `dist_clean` (neighbors
  exclude same-query candidates) OR simply `tfidf_plain` (trivially cacheable, equal MRR).
  De-leaking cost almost nothing (dist_corpus − dist_clean ≈ 0 on 12b/4b_base).

## CHANGE 3 — vs-BARE BOUND: small, model-dependent win on 12B (UNDER RE-TEST) (§6.3, §8, abstract)
CORRECTED: full N=900 gemma3_12b tfidf vs bare = +0.036 [+0.017,+0.055]* (SIGNIFICANT;
the n=400 interim +0.022 n.s. was the low subset). gemma3_4b +0.009 n.s.; qwen -0.007 n.s.
So: small SIGNIFICANT win on 12B, null on 4B -> model-dependent within Gemma. 27b tiebreaker
+ mistral + qwen14b re-running. Bound is NOT settled; do not finalize until 27b is in.
→ State carefully: keyword priming RECOVERS the harm generic priming causes (beats generic
  +0.05*). vs no-priming it is a SMALL effect that is significant on 12B (+0.036*) but null
  on 4B — model-dependent, ≤+0.04 MRR. Whether to call this a practical win depends on 27B
  and on whether +0.04 MRR matters; frame as "marginal, model-dependent," not "beats bare"
  or "no win." Do NOT finalize until 27B + Mistral + qwen14b land.

## CHANGE 4 — RETRACT §7.6/§7.8 "content-structured coherence" mechanism
exp21b position-matched coherence + Mistral falsification:
- Position-matching shrinks Gemma content-alignment from −0.40 to −0.123 (~70% was the
  BOS-distance positional artifact).
- Mistral content-align −0.152 is MORE structured than Gemma (−0.123) yet shows NO priming
  benefit → content-alignment does not track behavior. [exp21b]
→ Retract the representation-level "content-routed directional coherence" claim. Replace
  with: the deep architectural cause of primability is OPEN; what is established is
  (a) behavioral selectivity (Δnll decomposition), (b) primability as the gate, and
  (c) §7.5's position-matched directional patching effect. The 8-probe section becomes a
  "what it is NOT" characterization (QK-norm, sharpness, prefix-salience, content-coherence,
  fixed-direction sufficiency all falsified) + the two surviving behavioral facts.

## CHANGE 5 — NUMBER CORRECTIONS (§4, fig1) [done in source]
Exact exp05 pooled Cohen's d: tfidf d(NLL)=0.179/d(margin)=0.001; random_docwords
0.165/−0.017; random_vocab 0.031/−0.113; oracle 0.054/−0.057; extract 0.172/0.270.
(Previously fig1 plotted random_vocab 0.10, oracle 0.22, extract 0.20; text said
random_docwords margin "+0.03".) Fixed in paper table + make_figures.py.

## FIGURE CHANGES
- fig1: regenerated with exact values. [done]
- fig2: relabel "contrastive" → "keyword priming (vs generic)"; it remains valid (the
  dcorp-vs-generic effect == tfidf_plain-vs-generic effect, T1 null).
- NEW fig (recommended): per-model bare/generic/tfidf_plain MRR with the two key contrasts
  (generic hurts; keyword recovers to ≈bare) — the honest practical summary. Build from
  exp14b + exp14c when the ladder completes.

## PENDING before final v5
- Full high-N ladder (4b/27b/qwen at N=900) for ladder-wide vs-bare/vs-generic numbers.
- (optional) Mistral plain-keyword reranking to make the §7.6 falsification airtight
  (currently Mistral only tested with the contrast version).
- (optional) position-matched re-verify of §7.5 patching directionality.

## ONE-LINE NET
"Generic cache priming hurts retrieval; per-passage keyword priming is a selective,
primability-gated correction that recovers the loss on Gemma-family models and yields a
small significant gain over no-priming on the largest tested model (12B) but not the
smaller one — and the elegant representation-level mechanism we proposed does not survive
its controls." A rigorous-characterization + methodology paper; whether it is also a
(marginal) technique hinges on the 27B tiebreaker.
