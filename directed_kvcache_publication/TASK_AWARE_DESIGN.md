# Task-aware KV-cache construction — design memo (to merge with review workflow output)

**Goal (user):** improve the constructed KV cache when the downstream task is KNOWN at cache-build
time. Cache a document once, offline, knowing how it will be used (but not the exact query).

**Foundation (our findings, to be re-confirmed by the audit):**
- Priming imprints context in a model-specific MODE: semantic (Gemma/Mistral) vs surface (Qwen).
- **Mode–task match** governs value: semantic imprinting helps *relevance* (reranking), hurts
  *precise extraction* (QA); surface imprinting the reverse.
- Mode is fixed per (trained) model at inference — you cannot make Gemma a surface imprinter,
  but you CAN choose the PRIME CONTENT to match what the model's mode can bank.
- Most context value is un-bankable into zero retention (~65%+); the construction step is lossy.

**Levels of "knowing the task":**
1. task TYPE known (relevance vs extraction vs ...), exact query unknown — the realistic RAG case.
2. query DISTRIBUTION/type known (e.g., "who/role" questions, a domain), exact query unknown.
3. exact query known — degenerate (oracle-stripped ≈ 0; retaining it is trivial RAG).

## Strategy A — Mode-matched task-aware priming (the headline prescriptive test)
Turn the descriptive mode–task finding into a PRESCRIPTIVE rule. 2×2×conditions:
{Gemma (semantic), Qwen (surface)} × {relevance=reranking, extraction=QA} ×
{no-prime, semantic-prime (topic/"about" summary), surface-prime (key entities/numbers)}.
- Prediction: best cell is mode-matched — Gemma+relevance+semantic-prime > bare; Qwen+extraction+
  surface-prime > bare; mismatched primes ≤ bare (Gemma+extraction+semantic HURTS, per exp29).
- Payoff: knowing the task lets you pick the prime (or pick NOT to prime) → beats task-blind priming.
- Reuse: exp14c reranking harness + exp29 QA harness; add a `surface-prime` condition (entities/digits)
  and a `semantic-prime` condition (topic words / one-line gist). Machinery-neutral control throughout.

## Strategy B — Query-TYPE priming (granularity of task knowledge)
Exact-question priming HURT Gemma QA (exp29) because the question's meaning blurs the answer. Test a
COARSER signal: prime with the question TYPE (e.g., "who/role", "when/date", "where", a domain tag) or
a representative/centroid query, not the exact question. Strip, measure QA on questions of that type.
- Prediction: type-level priming is less answer-blurring; may help (or at least not hurt) where exact-
  question priming hurt — quantifies how much task knowledge helps vs how specific it must be.
- Reuse: exp29 harness; replace `q_primed` (exact question) with `qtype_primed` (question type/centroid).

## Strategy C — Task-conditioned gist RETENTION (lift the ceiling)
Bankability ceiling says zero-retention loses ~65%+. If the task is known, retain a FEW task-relevant
tokens (relax zero-retention): a one-line task-conditioned summary, or top-k query-type-relevant tokens.
- Conditions: zero-retention prime (current) | retain k∈{1,4,16} task-relevant gist tokens | full retain.
- Prediction: a few task-relevant retained tokens recover a large fraction of the un-bankable value at
  near-zero inference cost — connects our ceiling to the gisting literature, but task-conditioned.
- Reuse: bankability harness; add a "retain-k" mode (keep BOS + k chosen tokens + doc).

## Strategy D — Mode-aware deployment rule (no-priming is a valid task-aware choice)
The cleanest practical contribution may be a DECISION RULE, validated empirically:
  given (model imprinting mode, task type) → {prime semantic | prime surface | DO NOT prime}.
Show that following the rule beats both task-blind priming (always prime) and never-priming, averaged
over a task mix. This is the "actionable corollary" of mode–task match.

## Open risks (depend on audit)
- If exp29's "semantic hurts extraction" is a FORMAT confound (flagged for audit), Strategy A/B's
  extraction predictions need re-grounding on a clean QA measure first.
- If the base-vs-instruct flip is real and tuning sets mode, a further direction: can a small fine-tune
  STEER a model's imprinting mode toward the deployed task mix? (Phase-3 follow-up, larger effort.)

## Execution order (after review lands)
1. Fix any confirmed confounds (esp. downstream QA format, machinery-neutral fairness).
2. Run Strategy A (mode-matched priming) — the headline task-aware result.
3. Run Strategy C (task-conditioned gist retention) — the ceiling-lifting result.
4. Strategy B (query-type granularity) if A's extraction side is clean.
