# Contrastive Discrimination Sweep — Findings

5 models × 4 datasets × 300 samples × 8-way type-matched discrimination (n=1200/model,
6000 pooled). Margin = mean(NLL_distractor) − NLL_correct; entropy-invariant.

## Headline: the keyword effect is an entropy artifact

Pooled across all 5 models (6000 samples):

| condition | d(nll) | d(margin) | entropy gap |
|---|---|---|---|
| tfidf_16 | **+0.179*** | **+0.001** (ns) | +0.178 |
| random_docwords_16 | +0.165* | −0.017 (ns) | +0.182 |
| random_vocab_16 | +0.031* | −0.113* | +0.144 |
| **generic_instr_16** | +0.172* | **+0.270*** | −0.097 |
| oracle_16 | +0.054* | −0.057* | +0.110 |
| tfidf_4 | +0.162* | +0.033* | +0.128 |
| tfidf_64 | +0.189* | −0.009 (ns) | +0.198 |

(* = bootstrap 95% CI excludes 0)

**TF-IDF keywords improve absolute NLL (d=+0.18) but have ZERO effect on
discrimination (d=+0.001, CI [−0.02,+0.03]).** The entire keyword effect — the
headline of paper draft v3 — was entropy reduction (uniform confidence gain),
not improved answer selection. This directly explains the earlier NLL↔EM
disconnect in the generation eval: NLL improved, EM did not.

The oracle (query-as-prefix) is even worse: net NEGATIVE margin (−0.057) despite
positive NLL. random_vocab is actively harmful to discrimination (−0.113).

## The one thing that genuinely works: a generic instruction

**`generic_instr_16` ("Extract the key facts from this text") is the only
condition with a large, consistent positive margin effect — and the ONLY
condition positive for all 5 models**, including the two "harmed" models:

| model | generic_instr d(margin) | tfidf_16 d(margin) | oracle d(margin) |
|---|---|---|---|
| qwen25_1_5b | +0.38* | +0.35* | +0.29* |
| qwen25_7b | +0.19* | −0.03 (ns) | +0.09* |
| mistral_7b | +0.54* | +0.13* | +0.04 (ns) |
| gemma3_12b | +0.23* | −0.19* | −0.34* |
| ministral_8b | +0.19* | −0.18* | −0.31* |

Strikingly, for Gemma 12B and Ministral 8B — which were "harmed by everything"
on absolute NLL — the generic instruction is the *only* positive-margin
condition. Its entropy gap is negative (−0.097), i.e. its discrimination effect
is even larger than its NLL effect: it is a genuine representation improvement,
not a confidence shift.

## The four designed questions, answered

- **Q1 (confidence vs discrimination):** Most of the absolute-NLL "priming"
  signal is entropy reduction. Only `generic_instr` survives as discrimination.
- **Q2 (salience vs repetition):** tfidf − random_docwords margin d=+0.027,
  CI [+0.00,+0.05]. Statistically ~0, practically negligible (0.016 nats).
  **TF-IDF salience adds nothing beyond in-document repetition** — and neither
  helps discrimination anyway.
- **Q3 (document relevance):** random_docwords − random_vocab margin d=+0.107*.
  Document words beat random vocab — but mostly because random vocab is *harmful*
  (−0.11), not because document words help (docwords margin ≈ 0).
- **Q4 (oracle entropy dissociation):** positive gap for Qwen 1.5B (+0.16),
  Qwen 7B (+0.16), Gemma 12B (+0.23). The oracle's apparent NLL strength is
  substantially an entropy artifact.

## Implication for the paper

The narrative must flip:
1. Retire the TF-IDF-keyword headline — it has no discrimination value; the NLL
   gain was an entropy artifact (now demonstrated, not merely suspected).
2. New headline: **a generic task instruction is the only prefix that robustly
   improves answer discrimination**, across sizes and architectures, including
   models where absolute NLL gets worse.
3. Reframe the contribution around the **contrastive margin metric** as the
   correct way to evaluate cache priming — absolute NLL is misleading because
   priming trivially lowers entropy.

Per-dataset: where keyword margin is positive at all, it is confined to the
smallest model (Qwen 1.5B) on SQuAD/HotpotQA/TriviaQA, and vanishes on GSM8K.
