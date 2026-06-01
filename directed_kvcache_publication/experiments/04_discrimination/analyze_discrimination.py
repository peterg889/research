#!/usr/bin/env python3
"""Rigorous analysis of the contrastive answer-discrimination sweep (exp05).

Answers four questions, each with effect sizes and bootstrap 95% CIs:

Q1. Does priming improve DISCRIMINATION (margin), or only CONFIDENCE (nll)?
    For each condition: d(nll) vs d(margin). If d(nll) >> d(margin), the effect
    is largely entropy reduction, not useful representation.

Q2. SALIENCE vs REPETITION: tfidf_16 vs random_docwords_16 (paired, per-sample).
    If ~0, the keyword effect is in-document repetition, not salience.

Q3. DOCUMENT RELEVANCE: random_docwords_16 vs random_vocab_16 (paired).
    If > 0, document words beat generic tokens.

Q4. ORACLE ENTROPY DISSOCIATION: d(nll) - d(margin) for oracle_16.
    A large positive gap means the oracle's NLL advantage is an entropy artifact.

Runs on partial or complete results (skips missing checkpoints).

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/04_discrimination/analyze_discrimination.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp05_discrimination"
DATASETS = ["squad_v2", "hotpotqa", "triviaqa", "gsm8k"]
MODELS = ["qwen25_1_5b", "qwen25_7b", "mistral_7b", "gemma3_12b", "ministral_8b"]
PRIMED = ["tfidf_16", "random_docwords_16", "random_vocab_16",
          "generic_instr_16", "oracle_16", "tfidf_4", "tfidf_64"]

RNG = np.random.RandomState(42)
N_BOOT = 5000


def cohens_d(a):
    a = np.asarray(a, dtype=float)
    return a.mean() / (a.std(ddof=1) + 1e-12) if len(a) > 1 else 0.0


def boot_ci_d(a, n_boot=N_BOOT):
    """Bootstrap 95% CI for Cohen's d of a paired difference array."""
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return (0.0, 0.0)
    idx = RNG.randint(0, len(a), size=(n_boot, len(a)))
    samples = a[idx]
    ds = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) + 1e-12)
    return tuple(np.percentile(ds, [2.5, 97.5]))


def load_model(model_key):
    """Return dict: condition -> list of per-sample dicts pooled across datasets,
    plus which datasets are present."""
    md = RESULTS / model_key
    rows = []
    present = []
    for ds in DATASETS:
        ck = md / f"checkpoint_{ds}.json"
        if not ck.exists():
            continue
        data = json.loads(ck.read_text())
        if not data.get("samples"):
            continue
        present.append((ds, len(data["samples"])))
        for s in data["samples"]:
            s["_dataset"] = ds
            rows.append(s)
    return rows, present


def fmt_ci(d, ci):
    star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
    return f"{d:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}]{star}"


def main():
    print("=" * 100)
    print("CONTRASTIVE DISCRIMINATION ANALYSIS (exp05)")
    print("  d>0 = priming helps.  * = bootstrap 95% CI excludes 0.")
    print("=" * 100)

    all_present = {}
    pooled = {c: {"dnll": [], "dmgn": [], "dmgn1": [], "dtop1": []} for c in PRIMED}
    # paired contrast accumulators (per-sample, pooled across models+datasets)
    salience = []   # margin(tfidf_16) - margin(random_docwords_16)
    docrel = []     # margin(random_docwords_16) - margin(random_vocab_16)
    oracle_gap = {}  # per model: d(nll) - d(margin) for oracle

    for model_key in MODELS:
        rows, present = load_model(model_key)
        if not rows:
            continue
        all_present[model_key] = present
        n = len(rows)
        print(f"\n{'='*100}\n{model_key}  (n={n} pooled; " +
              ", ".join(f"{d}:{c}" for d, c in present) + ")")
        print(f"{'='*100}")
        print(f"{'condition':<20s} | {'d(nll_correct)':>22s} | {'d(margin_mean)':>22s} | {'d(margin_1st)':>22s} | {'top1Δ':>6s}")
        print("-" * 104)

        for c in PRIMED:
            dnll = [s["bare__nll_correct"] - s[f"{c}__nll_correct"]
                    for s in rows if f"{c}__nll_correct" in s]
            dmgn = [s[f"{c}__margin_mean"] - s["bare__margin_mean"]
                    for s in rows if f"{c}__margin_mean" in s]
            dmgn1 = [s[f"{c}__margin_first"] - s["bare__margin_first"]
                     for s in rows if f"{c}__margin_first" in s]
            dtop1 = [s[f"{c}__top1"] - s["bare__top1"]
                     for s in rows if f"{c}__top1" in s]
            if not dnll:
                continue
            pooled[c]["dnll"] += dnll
            pooled[c]["dmgn"] += dmgn
            pooled[c]["dmgn1"] += dmgn1
            pooled[c]["dtop1"] += dtop1
            print(f"{c:<20s} | {fmt_ci(cohens_d(dnll), boot_ci_d(dnll)):>22s} | "
                  f"{fmt_ci(cohens_d(dmgn), boot_ci_d(dmgn)):>22s} | "
                  f"{fmt_ci(cohens_d(dmgn1), boot_ci_d(dmgn1)):>22s} | {np.mean(dtop1):>+6.1%}")

        # oracle entropy gap for this model
        on = [s["bare__nll_correct"] - s["oracle_16__nll_correct"]
              for s in rows if "oracle_16__nll_correct" in s]
        om = [s["oracle_16__margin_mean"] - s["bare__margin_mean"]
              for s in rows if "oracle_16__margin_mean" in s]
        if on and om:
            oracle_gap[model_key] = (cohens_d(on), cohens_d(om))

        # paired salience & docrel contrasts (same samples)
        for s in rows:
            if "tfidf_16__margin_mean" in s and "random_docwords_16__margin_mean" in s:
                salience.append(s["tfidf_16__margin_mean"] - s["random_docwords_16__margin_mean"])
            if "random_docwords_16__margin_mean" in s and "random_vocab_16__margin_mean" in s:
                docrel.append(s["random_docwords_16__margin_mean"] - s["random_vocab_16__margin_mean"])

    # ---------- Pooled cross-model summary ----------
    print(f"\n\n{'='*100}")
    print("POOLED ACROSS ALL MODELS & DATASETS")
    print(f"{'='*100}")
    print(f"{'condition':<20s} | {'d(nll)':>22s} | {'d(margin)':>22s} | {'entropy gap':>11s}")
    print("-" * 84)
    for c in PRIMED:
        if not pooled[c]["dnll"]:
            continue
        dn = cohens_d(pooled[c]["dnll"])
        dm = cohens_d(pooled[c]["dmgn"])
        print(f"{c:<20s} | {fmt_ci(dn, boot_ci_d(pooled[c]['dnll'])):>22s} | "
              f"{fmt_ci(dm, boot_ci_d(pooled[c]['dmgn'])):>22s} | {dn - dm:>+11.3f}")

    # ---------- Q2: salience vs repetition ----------
    print(f"\n\n{'='*100}")
    print("Q2. SALIENCE vs REPETITION  (margin: tfidf_16 - random_docwords_16, paired)")
    print("    H0: TF-IDF salience adds nothing beyond in-document repetition (d=0).")
    print(f"{'='*100}")
    if salience:
        d = cohens_d(salience); ci = boot_ci_d(salience)
        print(f"  n={len(salience)}  d = {fmt_ci(d, ci)}")
        print(f"  mean margin diff = {np.mean(salience):+.4f} nats")
        verdict = ("SALIENCE MATTERS (tfidf > random docwords)" if ci[0] > 0 else
                   "REPETITION DOMINATES (tfidf ~= random docwords)" if ci[1] < 0.1 else
                   "INCONCLUSIVE")
        print(f"  VERDICT: {verdict}")

    # ---------- Q3: document relevance ----------
    print(f"\n{'='*100}")
    print("Q3. DOCUMENT RELEVANCE  (margin: random_docwords_16 - random_vocab_16, paired)")
    print("    H0: document words add nothing beyond generic token presence (d=0).")
    print(f"{'='*100}")
    if docrel:
        d = cohens_d(docrel); ci = boot_ci_d(docrel)
        print(f"  n={len(docrel)}  d = {fmt_ci(d, ci)}")
        print(f"  mean margin diff = {np.mean(docrel):+.4f} nats")
        verdict = ("DOCUMENT WORDS MATTER" if ci[0] > 0 else
                   "NO DOCUMENT-RELEVANCE EFFECT" if ci[1] < 0.1 else "INCONCLUSIVE")
        print(f"  VERDICT: {verdict}")

    # ---------- Q4: oracle entropy dissociation ----------
    print(f"\n{'='*100}")
    print("Q4. ORACLE ENTROPY DISSOCIATION  (d(nll) - d(margin) for oracle_16, per model)")
    print("    A large positive gap = the oracle's NLL advantage is an entropy artifact.")
    print(f"{'='*100}")
    print(f"  {'model':<16s} | {'d(nll)':>7s} | {'d(margin)':>9s} | {'gap':>6s}")
    print("  " + "-" * 46)
    for mk, (dn, dm) in oracle_gap.items():
        print(f"  {mk:<16s} | {dn:>+7.2f} | {dm:>+9.2f} | {dn-dm:>+6.2f}")

    # ---------- per-dataset headline ----------
    print(f"\n\n{'='*100}")
    print("PER-DATASET d(margin) for tfidf_16 and random_docwords_16")
    print(f"{'='*100}")
    print(f"  {'model':<16s} {'cond':<18s} | " + " ".join(f"{d[:8]:>8s}" for d in DATASETS))
    for model_key in MODELS:
        md = RESULTS / model_key
        for c in ["tfidf_16", "random_docwords_16"]:
            cells = []
            any_data = False
            for ds in DATASETS:
                ck = md / f"checkpoint_{ds}.json"
                if not ck.exists():
                    cells.append(f"{'--':>8s}"); continue
                samples = json.loads(ck.read_text()).get("samples", [])
                diffs = [s[f"{c}__margin_mean"] - s["bare__margin_mean"]
                         for s in samples if f"{c}__margin_mean" in s]
                if len(diffs) > 1:
                    cells.append(f"{cohens_d(diffs):>+8.2f}"); any_data = True
                else:
                    cells.append(f"{'--':>8s}")
            if any_data:
                print(f"  {model_key:<16s} {c:<18s} | " + " ".join(cells))


if __name__ == "__main__":
    main()
