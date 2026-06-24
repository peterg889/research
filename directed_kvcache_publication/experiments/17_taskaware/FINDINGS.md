# Task-aware KV-cache experiments — findings (exp31a/b, exp32)

Goal (user): improve the constructed KV cache when the downstream task is KNOWN at build time.
Prior art (verified): Beyond RAG (2503.04973, task-aware *compression*), SnapKV/Ada-KV/KVzip
(query-aware vs query-agnostic *selection*), Cartridges (2506.06266, *trained* cache). All change
WHICH tokens/slots are retained. Our angle: task-aware *conditioning* (discarded prime) + mode
routing. These experiments stress-test that angle and the mechanism behind it.

## exp31a — prescriptive sem-vs-surf prime on QA (SQuAD, N=300, machinery-controlled)
Hold the question's TOKEN SET fixed; vary only order/meaning. content = prime − neutral (pos=hurts).

| model | sem (ordered Q) | surf (shuffled Q) | order = sem−surf |
|---|---|---|---|
| gemma3_12b | +0.358* hurts | +0.322* hurts | +0.036 (n.s.) |
| gemma3_4b  | +0.116 (n.s.) | −0.042 (n.s.) | +0.158 (n.s.) |
| qwen25_7b  | −0.800* helps | −0.729* helps | −0.070 (n.s.) |
| qwen25_14b | +0.439* hurts | +0.276* hurts | +0.163* |

**Findings.**
1. The QA effect is **token-presence-driven**: shuffling the question's tokens barely changes it
   (|order| ≤ 0.16, mostly n.s.). Word order / meaning contributes little. → the §7 claim that
   "Gemma banks the question's *meaning* and blurs the answer" is WRONG for QA; it is token presence.
2. The SIGN is model-specific and **does not track family**: qwen25_7b HELPS (−0.80), but
   qwen25_14b HURTS (+0.44) like the Gemmas. → the mode-task 2×2 "surface-imprinter helps
   extraction" is a **qwen7b-specific** result, not a Qwen-family or general surface-mode result.
   §7 must be demoted to a single-model illustration, not a law.
3. (qwen25_14b shows a large machinery cost neutral−bare = +1.89; the reposition/normalize step
   damages its cache more than others. Content contrast is still machinery-matched and valid.)

## exp31b — conditioning vs selection (the make-or-break control) [RUNNING]
Per item: pick top-k=32 doc tokens by question→doc attention (SnapKV-style). Compare iso-budget.
Decisive contrast: sel_k_primed − sel_k_plain (value of conditioning, retained set held fixed).
Smoke (gemma3_4b, n=8): selection HELPS strongly (selVal vs bare_norm ≈ −1.6) and conditioning on
top of selection ~0/slightly hurts (COND|sel ≈ +0.5). If this holds: for extraction on a semantic
imprinter, task-aware SELECTION dominates CONDITIONING. Open cell: Qwen-7B (where conditioning helped).

## exp32 — binding shuffle: does priming bank MEANING or TOKEN PRESENCE? [RUNNING]
Two facts primed, ask about one (requires city→topic BINDING). ordered vs token-shuffled prime.
BINDING = strip_ord − strip_shuf (neg = ordered beats shuffled = genuine relational banking).
Smoke (gemma3_4b, n=6): bankOrd ≈ −2.84, bankShuf ≈ −1.18, BINDING ≈ −1.67 → ordered banks the
correct association ~1.7 nats better than shuffled. → unlike QA, when the task REQUIRES binding,
word order matters; genuine structure is banked. Full run tests whether BINDING splits by mode
(negative for Gemma/Mistral semantic imprinters, ~0 for Qwen surface imprinter).

## Emerging honest reframe (pending full exp32/31b)
- A downstream task's response to zero-retention priming is dominated by TOKEN PRESENCE
  (order-invariant), sign set by model (and scale), NOT by a clean semantic-vs-surface split.
- The semantic/structure ("meaning") component appears specifically when the task REQUIRES
  relational binding (exp32), and that is the right regime to test imprinting mode.
- Practically: if you know the task is extraction, SELECT relevant tokens (cheaper + better) rather
  than condition; conditioning's residual value (if any) is mode/scale-specific.
