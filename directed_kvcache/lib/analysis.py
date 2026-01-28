"""
Statistical analysis utilities for experiment results.
"""

from typing import List, Dict, Any
import numpy as np
from scipy import stats

from .surrogate import TOP_5_SURROGATE_TEMPLATES, STATIC_SURROGATE_QUERIES


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
