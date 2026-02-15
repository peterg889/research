"""
Statistical analysis utilities for experiment results.

This module provides:
- Effect size computation (Cohen's d)
- Ranking metrics (MRR, Hit@k)
- Token overlap / similarity metrics
- Experiment result analysis
"""

from typing import List, Dict, Any, Set
import numpy as np
from scipy import stats

from .surrogate import TOP_5_SURROGATE_TEMPLATES, STATIC_SURROGATE_QUERIES


def compute_ranking_metrics(scores: Dict[int, float], relevant_idx: int = 0) -> dict:
    """
    Compute ranking metrics from NLL scores.

    Args:
        scores: Dict mapping candidate index to NLL score (lower is better)
        relevant_idx: Index of the relevant/correct candidate (default 0)

    Returns:
        dict with:
            - mrr: Mean Reciprocal Rank (1/rank if rank <= 10, else 0)
            - hit_at_1: 1.0 if relevant is ranked first, else 0.0
            - hit_at_3: 1.0 if relevant is in top 3, else 0.0
            - relevant_rank: Actual rank of the relevant candidate
            - relevant_nll: NLL score of the relevant candidate

    Example:
        scores = {0: 1.5, 1: 2.0, 2: 1.8}  # candidate 0 has lowest NLL
        metrics = compute_ranking_metrics(scores, relevant_idx=0)
        # metrics['hit_at_1'] = 1.0, metrics['mrr'] = 1.0
    """
    # Sort by NLL (lower is better)
    ranked = sorted(scores.items(), key=lambda x: x[1])
    ranked_ids = [idx for idx, _ in ranked]
    relevant_rank = ranked_ids.index(relevant_idx) + 1

    return {
        'mrr': 1.0 / relevant_rank if relevant_rank <= 10 else 0.0,
        'hit_at_1': 1.0 if relevant_rank == 1 else 0.0,
        'hit_at_3': 1.0 if relevant_rank <= 3 else 0.0,
        'relevant_rank': relevant_rank,
        'relevant_nll': scores[relevant_idx]
    }


def compute_token_overlap(text1: str, text2: str, tokenizer) -> float:
    """
    Compute Jaccard similarity of token sets between two texts.

    This measures literal token overlap, useful for analyzing
    interference effects from repeated tokens.

    Args:
        text1: First text
        text2: Second text
        tokenizer: Tokenizer to use for encoding

    Returns:
        Jaccard similarity (intersection / union) in [0, 1]

    Example:
        overlap = compute_token_overlap("What is the capital?", "What is the city?", tokenizer)
        # High overlap due to shared tokens "What", "is", "the", "?"
    """
    tokens1 = set(tokenizer.encode(text1.lower(), add_special_tokens=False))
    tokens2 = set(tokenizer.encode(text2.lower(), add_special_tokens=False))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def compute_pairwise_ranking_stats(
    condition_results: List[Dict],
    baseline_key: str = 'bare',
) -> Dict[str, Any]:
    """
    Compute pairwise ranking statistics across samples.

    Args:
        condition_results: List of dicts, each containing 'metrics' with ranking info
        baseline_key: Key for the baseline condition to compare against

    Returns:
        dict with aggregated statistics
    """
    mrrs = [r['metrics']['mrr'] for r in condition_results]
    hit1s = [r['metrics']['hit_at_1'] for r in condition_results]
    hit3s = [r['metrics']['hit_at_3'] for r in condition_results]
    ranks = [r['metrics']['relevant_rank'] for r in condition_results]

    return {
        'n': len(condition_results),
        'mrr_mean': np.mean(mrrs),
        'mrr_std': np.std(mrrs),
        'hit_at_1_mean': np.mean(hit1s),
        'hit_at_3_mean': np.mean(hit3s),
        'rank_mean': np.mean(ranks),
        'rank_median': np.median(ranks),
    }


def cohens_d(diff: np.ndarray) -> float:
    """Compute Cohen's d effect size for a difference array."""
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0


def analyze_experiment_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Comprehensive analysis of experiment results for both generated and static surrogates.

    This function computes:
    - Summary statistics (mean, std) for all conditions
    - Win rates vs baseline
    - Statistical significance tests (paired t-tests)
    - Effect sizes (Cohen's d)
    - Per-template analysis

    Args:
        results: List of result dictionaries from evaluate_sample_with_top5_routing

    Returns:
        Dictionary containing all analysis metrics
    """
    baseline_nlls = np.array([r['baseline_nll'] for r in results])

    # Generated surrogates
    gen_routed_nlls = np.array([r['gen_routed_nll'] for r in results])
    gen_oracle_nlls = np.array([r['gen_oracle_nll'] for r in results])
    deltas_gen_routed = np.array([r['delta_gen_routed'] for r in results])
    deltas_gen_oracle = np.array([r['delta_gen_oracle'] for r in results])

    # Static surrogates
    static_routed_nlls = np.array([r['static_routed_nll'] for r in results])
    static_oracle_nlls = np.array([r['static_oracle_nll'] for r in results])
    deltas_static_routed = np.array([r['delta_static_routed'] for r in results])
    deltas_static_oracle = np.array([r['delta_static_oracle'] for r in results])

    # Per-template analysis for GENERATED
    gen_template_stats = {}
    for key in TOP_5_SURROGATE_TEMPLATES.keys():
        nlls = [r['generated_nlls'][key] for r in results]
        deltas = [r['baseline_nll'] - r['generated_nlls'][key] for r in results]
        times_routed = sum(1 for r in results if r['gen_routed_key'] == key)
        times_oracle = sum(1 for r in results if r['gen_oracle_key'] == key)

        gen_template_stats[key] = {
            'mean_nll': np.mean(nlls),
            'mean_delta': np.mean(deltas),
            'win_rate': np.mean([d > 0 for d in deltas]),
            'times_routed': times_routed,
            'times_oracle': times_oracle,
        }

    # Per-template analysis for STATIC
    static_template_stats = {}
    for key in STATIC_SURROGATE_QUERIES.keys():
        nlls = [r['static_nlls'][key] for r in results]
        deltas = [r['baseline_nll'] - r['static_nlls'][key] for r in results]
        times_routed = sum(1 for r in results if r['static_routed_key'] == key)
        times_oracle = sum(1 for r in results if r['static_oracle_key'] == key)

        static_template_stats[key] = {
            'mean_nll': np.mean(nlls),
            'mean_delta': np.mean(deltas),
            'win_rate': np.mean([d > 0 for d in deltas]),
            'times_routed': times_routed,
            'times_oracle': times_oracle,
        }

    # Statistical tests
    t_gen_routed, p_gen_routed = stats.ttest_rel(baseline_nlls, gen_routed_nlls)
    t_gen_oracle, p_gen_oracle = stats.ttest_rel(baseline_nlls, gen_oracle_nlls)
    t_static_routed, p_static_routed = stats.ttest_rel(baseline_nlls, static_routed_nlls)
    t_static_oracle, p_static_oracle = stats.ttest_rel(baseline_nlls, static_oracle_nlls)

    # Generated vs Static comparison
    t_gen_vs_static, p_gen_vs_static = stats.ttest_rel(gen_routed_nlls, static_routed_nlls)

    return {
        'n_samples': len(results),

        # Baseline
        'mean_baseline_nll': np.mean(baseline_nlls),
        'std_baseline_nll': np.std(baseline_nlls),

        # Generated surrogates
        'mean_gen_routed_nll': np.mean(gen_routed_nlls),
        'std_gen_routed_nll': np.std(gen_routed_nlls),
        'mean_gen_oracle_nll': np.mean(gen_oracle_nlls),
        'std_gen_oracle_nll': np.std(gen_oracle_nlls),
        'mean_delta_gen_routed': np.mean(deltas_gen_routed),
        'median_delta_gen_routed': np.median(deltas_gen_routed),
        'mean_delta_gen_oracle': np.mean(deltas_gen_oracle),
        'win_rate_gen_routed': np.mean(deltas_gen_routed > 0),
        'win_rate_gen_oracle': np.mean(deltas_gen_oracle > 0),
        't_stat_gen_routed': t_gen_routed,
        'p_value_gen_routed': p_gen_routed,
        't_stat_gen_oracle': t_gen_oracle,
        'p_value_gen_oracle': p_gen_oracle,
        'cohens_d_gen_routed': cohens_d(deltas_gen_routed),

        # Static surrogates
        'mean_static_routed_nll': np.mean(static_routed_nlls),
        'std_static_routed_nll': np.std(static_routed_nlls),
        'mean_static_oracle_nll': np.mean(static_oracle_nlls),
        'std_static_oracle_nll': np.std(static_oracle_nlls),
        'mean_delta_static_routed': np.mean(deltas_static_routed),
        'median_delta_static_routed': np.median(deltas_static_routed),
        'mean_delta_static_oracle': np.mean(deltas_static_oracle),
        'win_rate_static_routed': np.mean(deltas_static_routed > 0),
        'win_rate_static_oracle': np.mean(deltas_static_oracle > 0),
        't_stat_static_routed': t_static_routed,
        'p_value_static_routed': p_static_routed,
        't_stat_static_oracle': t_static_oracle,
        'p_value_static_oracle': p_static_oracle,
        'cohens_d_static_routed': cohens_d(deltas_static_routed),

        # Generated vs Static comparison
        't_stat_gen_vs_static': t_gen_vs_static,
        'p_value_gen_vs_static': p_gen_vs_static,
        'gen_beats_static_rate': np.mean(gen_routed_nlls < static_routed_nlls),

        # Per-template stats
        'gen_template_stats': gen_template_stats,
        'static_template_stats': static_template_stats,
    }


def print_analysis_summary(analysis: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the analysis results.

    Args:
        analysis: Dictionary from analyze_experiment_results
    """
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nSamples analyzed: {analysis['n_samples']}")

    print(f"\n{'=' * 80}")
    print("ANSWER NLL BY CONDITION (lower = better)")
    print(f"{'=' * 80}")
    print(f"{'Condition':<30} {'Mean NLL':>12} {'Std':>10}")
    print("-" * 55)
    print(f"{'Baseline (doc only)':<30} {analysis['mean_baseline_nll']:>12.4f} {analysis['std_baseline_nll']:>10.4f}")
    print("-" * 55)
    print(f"{'Generated Routed':<30} {analysis['mean_gen_routed_nll']:>12.4f} {analysis['std_gen_routed_nll']:>10.4f}")
    print(f"{'Generated Oracle':<30} {analysis['mean_gen_oracle_nll']:>12.4f} {analysis['std_gen_oracle_nll']:>10.4f}")
    print("-" * 55)
    print(f"{'Static Routed':<30} {analysis['mean_static_routed_nll']:>12.4f} {analysis['std_static_routed_nll']:>10.4f}")
    print(f"{'Static Oracle':<30} {analysis['mean_static_oracle_nll']:>12.4f} {analysis['std_static_oracle_nll']:>10.4f}")

    print(f"\n{'=' * 80}")
    print("WIN RATES vs BASELINE (positive delta = surrogate better)")
    print(f"{'=' * 80}")
    print(f"Generated Routed vs Baseline: {analysis['win_rate_gen_routed']*100:.1f}%")
    print(f"Generated Oracle vs Baseline: {analysis['win_rate_gen_oracle']*100:.1f}%")
    print(f"Static Routed vs Baseline:    {analysis['win_rate_static_routed']*100:.1f}%")
    print(f"Static Oracle vs Baseline:    {analysis['win_rate_static_oracle']*100:.1f}%")

    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'=' * 80}")
    print(f"Generated Routed: t={analysis['t_stat_gen_routed']:.3f}, p={analysis['p_value_gen_routed']:.4f}")
    print(f"Static Routed:    t={analysis['t_stat_static_routed']:.3f}, p={analysis['p_value_static_routed']:.4f}")
    print(f"Gen vs Static:    t={analysis['t_stat_gen_vs_static']:.3f}, p={analysis['p_value_gen_vs_static']:.4f}")
