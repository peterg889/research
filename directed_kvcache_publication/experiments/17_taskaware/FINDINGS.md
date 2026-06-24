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

## exp32 — binding shuffle: does priming bank MEANING or TOKEN PRESENCE? (N=150, 4/5 done)
Two facts primed, ask about one (requires city→topic BINDING). ordered vs token-shuffled prime.
ORDER = strip_ord − strip_shuf (neg = ordered beats shuffled = genuine structure banking; pos =
shuffled/token-presence beats ordered/meaning).

| model | bankOrd | bankShuf | ORDER (ord−shuf) |
|---|---|---|---|
| gemma3_12b | −0.39 (n.s.) | −1.82* | **+1.43*** (shuffled/tokens bank MORE) |
| gemma3_4b  | −1.92* | −1.41* | **−0.51*** (ordered banks more) |
| mistral_7b | −0.71* | +0.36* | **−1.07*** (ordered banks more; shuffle HURTS) |
| qwen25_7b  | +0.07 (n.s.) | −0.05 (n.s.) | +0.12 (n.s.) (banks neither) |
| gemma3_27b | (pending) | | |

**There is NO clean mapping from "imprinting mode" to structure banking.** Within the Gemma family
the ORDER sign FLIPS with scale (4b −0.51 → 12b +1.43). Interpretation that fits all four:
- Imprintability (scales with size) = strength of SEMANTIC ABSTRACTION of primed content.
- Abstraction TRADES OFF against literal token traces. For LITERAL recall, token presence
  (order-invariant) is what helps; semantic integration of ordered input competes with it.
- Small/mid models (gemma3_4b, mistral) still leave enough structure trace that ordered > shuffled.
- The most imprintable model (gemma3_12b) abstracts ordered meaning away from literal traces, so
  shuffled (pure token presence) recalls the literal answer better (+1.43).
- The surface model (qwen7b) banks neither structure nor tokens for a 2-fact bind (can't bind).

## exp33 — single-fact shuffle on exp26's EXACT measure [QUEUED]
Direct test of whether the paper's headline "semantic axis" is meaning or token presence. With ONE
fact (one topic to recall), token presence should suffice → predict ORDER≈0 (banking survives
shuffle) → "semantic" label = token-type/token-presence, not meaning/structure. Confirms or refutes.

## Emerging honest reframe (firming up)
1. Zero-retention priming banks TOKEN PRESENCE / token-type imprint, NOT relational meaning.
   Imprintability = strength of that imprint (≈ semantic abstraction), scales with size.
2. Effect on a downstream task is order-invariant with a model+scale-specific SIGN; no clean
   semantic-vs-surface split (qwen7b helps QA, qwen14b hurts; gemma4b vs 12b flip on binding).
3. Genuine RELATIONAL/structure banking is weak, model-idiosyncratic, and at high imprintability
   COMPETES with literal recall. The "semantic axis" framing overstates meaning; it is largely a
   content-token-imprint axis. (exp33 will confirm on the headline measure.)
4. Practical task-aware takeaway (exp31b, pending): for EXTRACTION, task-aware SELECTION (drop
   irrelevant tokens, à la SnapKV/Beyond RAG) is cheaper AND better than conditioning; conditioning's
   residual value is mode/scale-specific and small.

## Consequences for paper_draft_v5.md (to apply once exp33/31b land)
- §6 "robust semantic axis": relabel toward content-token-imprint; report the shuffle controls;
  keep imprintability r=0.94 but reinterpret it as abstraction strength, not meaning-banking.
- §7 mode-task 2×2: demote hard (qwen7b-specific; QA effect is token presence; no family law).
- Add a task-aware section: selection ≥ conditioning for extraction (honest negative for our method),
  positioned against Beyond RAG / SnapKV.
