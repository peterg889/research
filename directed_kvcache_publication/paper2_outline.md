# Companion paper (outline): Query-Aware KV-Cache Pruning Has a Dose-Dependent Downstream-Accuracy Cost

Distinct audience (systems/efficiency) and distinct thesis from the imprinting paper (`paper.md`).
Self-contained; cites only published work; no reference to unpublished prior attempts.

**One-line thesis.** When a document's KV cache is built for a known task, aggressive query-aware
*selection* (SnapKV-style top-k pruning) degrades downstream QA answer accuracy in a smooth,
dose-dependent, model-specific way — often severely — whereas reshaping the kept cache with a
discarded prefix leaves accuracy intact. The result is a caution for query-aware KV compression and a
demonstration that answer-NLL improvements do not imply answer-correctness improvements.

**Why it's a clean standalone story.** It speaks directly to the deployed KV-compression literature
[@snapkv; @adakv; @kvzip; @beyondrag]; the central metric is task accuracy, not perplexity; and it
carries a methodological contribution (verbosity-robust answer-recall for generation-based QA eval).

**Findings (all run, verified, committed):**
1. **Selection hurts answer correctness, dose-dependently (headline).** k-sweep on SQuAD (Qwen-1.5B/7B):
   answer-recall vs full-doc bare degrades smoothly — k=8 −67/−62, k=32 −36/−24, k=64 −18/−8 (keeping
   HALF still significantly hurts), recovering only near the document length (~128 tokens). So on
   short passages, *any real query-aware compression* costs Qwen answer accuracy. (exp37, Fig fig14)
2. **Cross-task.** Replicates and amplifies on multi-hop HotpotQA (two gold paragraphs): selection
   answer-recall −31.5*/−34*/−41* (gemma12b/qwen7b/qwen1.5b), vs −8.5*/−24*/−36* on SQuAD. Gemma is
   hit hard once the task needs multi-paragraph context (k=32 can't span two paragraphs). (exp35/36)
3. **Conditioning is correctness-neutral.** Discarded-prefix conditioning shifts answer-NLL and EM/F1
   but does not change answer-recall on either task (all n.s.) — its effect is calibration/verbosity,
   not whether the answer is found. Honest deflation: conditioning's value is that it does not destroy
   answer-bearing tokens the way pruning does, not that it adds accuracy.
4. **Methodological control.** Generation-based EM/F1 is confounded by answer verbosity (interventions
   change how models answer); verbosity-robust answer-recall is required for a fair correctness
   comparison. (This control is itself a contribution for QA-generation eval.)
5. **Select-vs-condition is not trait-predictable.** Across 8 models the winner (in NLL) does not
   reduce to imprintability (r=0.29) or size — a null that motivates "probe per model," and that
   frames the accuracy study as the decisive metric.

**Numbers/scripts:** analyze_qa_accuracy.py (exp35 SQuAD, exp36 Hotpot), run_ksweep.py (exp37),
QA_ACCURACY_NUMBERS.txt, and the select-vs-condition tables in make_taskaware_numbers.py. Figures
fig13 (select-vs-condition), fig14 (k-sweep dose-response) already generated.

**Proposed structure:** intro (KV compression + the query-aware pruning assumption) → related work
(KV eviction/compression) → setup (selection vs conditioning; cache construction) → method (greedy
generation from constructed cache; verbosity-robust answer-recall) → §selection hurts (dose-response
+ cross-task) → §conditioning neutral → §no trait rule → limitations (N=200 generation power; single
selector family; extractive QA) → conclusion.
