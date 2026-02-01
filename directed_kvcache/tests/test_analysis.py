"""Tests for cohens_d and analyze_experiment_results."""

import numpy as np
import pytest
from lib.analysis import cohens_d, analyze_experiment_results
from lib.surrogate import TOP_5_SURROGATE_TEMPLATES, STATIC_SURROGATE_QUERIES


class TestCohensD:
    def test_all_same_returns_zero(self):
        diff = np.array([5.0, 5.0, 5.0, 5.0])
        assert cohens_d(diff) == 0

    def test_known_d_equals_1(self):
        """If mean == std, d = 1."""
        # mean=1, std=1 => d=1
        diff = np.array([0.0, 2.0])  # mean=1, std=sqrt(2) with ddof=1 => std=sqrt(2)
        # Actually for [0, 2]: mean=1, std(ddof=1)=sqrt(2), d=1/sqrt(2)
        # For d=1: need mean/std(ddof=1) = 1
        # Use [1-s, 1+s] => mean=1, std(ddof=1) = s*sqrt(2/(n-1)) for n=2 => std=s*sqrt(2)
        # Want mean/std = 1 => 1/(s*sqrt(2))=1 => s=1/sqrt(2)
        s = 1.0 / np.sqrt(2)
        diff = np.array([1 - s, 1 + s])
        result = cohens_d(diff)
        assert abs(result - 1.0) < 1e-10

    def test_negative_d(self):
        """Negative mean gives negative d."""
        diff = np.array([-3.0, -5.0, -4.0])
        assert cohens_d(diff) < 0

    def test_zero_std_zero_mean(self):
        """All zeros: std=0, returns 0."""
        diff = np.array([0.0, 0.0, 0.0])
        assert cohens_d(diff) == 0


class TestAnalyzeExperimentResults:
    @pytest.fixture
    def synthetic_results(self):
        """Create synthetic experiment results with known properties."""
        gen_keys = list(TOP_5_SURROGATE_TEMPLATES.keys())
        static_keys = list(STATIC_SURROGATE_QUERIES.keys())
        np.random.seed(42)

        results = []
        for i in range(20):
            baseline_nll = 3.0 + np.random.randn() * 0.5

            # Generated NLLs: slightly better than baseline
            gen_nlls = {}
            for k in gen_keys:
                gen_nlls[k] = baseline_nll - 0.1 + np.random.randn() * 0.3

            # Static NLLs: slightly worse than generated
            static_nlls = {}
            for k in static_keys:
                static_nlls[k] = baseline_nll + 0.05 + np.random.randn() * 0.3

            # Routed = best (lowest NLL) across templates
            gen_routed_key = min(gen_nlls, key=gen_nlls.get)
            gen_oracle_key = min(gen_nlls, key=gen_nlls.get)
            static_routed_key = min(static_nlls, key=static_nlls.get)
            static_oracle_key = min(static_nlls, key=static_nlls.get)

            results.append({
                'baseline_nll': baseline_nll,
                'generated_nlls': gen_nlls,
                'static_nlls': static_nlls,
                'gen_routed_nll': gen_nlls[gen_routed_key],
                'gen_oracle_nll': gen_nlls[gen_oracle_key],
                'gen_routed_key': gen_routed_key,
                'gen_oracle_key': gen_oracle_key,
                'delta_gen_routed': baseline_nll - gen_nlls[gen_routed_key],
                'delta_gen_oracle': baseline_nll - gen_nlls[gen_oracle_key],
                'static_routed_nll': static_nlls[static_routed_key],
                'static_oracle_nll': static_nlls[static_oracle_key],
                'static_routed_key': static_routed_key,
                'static_oracle_key': static_oracle_key,
                'delta_static_routed': baseline_nll - static_nlls[static_routed_key],
                'delta_static_oracle': baseline_nll - static_nlls[static_oracle_key],
            })

        return results

    def test_all_output_keys_present(self, synthetic_results):
        analysis = analyze_experiment_results(synthetic_results)
        expected_keys = [
            'n_samples',
            'mean_baseline_nll', 'std_baseline_nll',
            'mean_gen_routed_nll', 'std_gen_routed_nll',
            'mean_gen_oracle_nll', 'std_gen_oracle_nll',
            'mean_delta_gen_routed', 'median_delta_gen_routed',
            'mean_delta_gen_oracle',
            'win_rate_gen_routed', 'win_rate_gen_oracle',
            't_stat_gen_routed', 'p_value_gen_routed',
            't_stat_gen_oracle', 'p_value_gen_oracle',
            'cohens_d_gen_routed',
            'mean_static_routed_nll', 'std_static_routed_nll',
            'mean_static_oracle_nll', 'std_static_oracle_nll',
            'mean_delta_static_routed', 'median_delta_static_routed',
            'mean_delta_static_oracle',
            'win_rate_static_routed', 'win_rate_static_oracle',
            't_stat_static_routed', 'p_value_static_routed',
            't_stat_static_oracle', 'p_value_static_oracle',
            'cohens_d_static_routed',
            't_stat_gen_vs_static', 'p_value_gen_vs_static',
            'gen_beats_static_rate',
            'gen_template_stats', 'static_template_stats',
        ]
        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"

    def test_n_samples(self, synthetic_results):
        analysis = analyze_experiment_results(synthetic_results)
        assert analysis['n_samples'] == 20

    def test_win_rates_bounded(self, synthetic_results):
        analysis = analyze_experiment_results(synthetic_results)
        for key in ['win_rate_gen_routed', 'win_rate_gen_oracle',
                     'win_rate_static_routed', 'win_rate_static_oracle',
                     'gen_beats_static_rate']:
            assert 0.0 <= analysis[key] <= 1.0

    def test_gen_routed_better_than_static(self, synthetic_results):
        """With our synthetic data, generated surrogates should beat static."""
        analysis = analyze_experiment_results(synthetic_results)
        # Generated has lower NLL (better) by construction
        assert analysis['mean_gen_routed_nll'] < analysis['mean_static_routed_nll']

    def test_template_stats_keys(self, synthetic_results):
        analysis = analyze_experiment_results(synthetic_results)
        for key in TOP_5_SURROGATE_TEMPLATES:
            assert key in analysis['gen_template_stats']
            stats = analysis['gen_template_stats'][key]
            assert 'mean_nll' in stats
            assert 'mean_delta' in stats
            assert 'win_rate' in stats
            assert 'times_routed' in stats
            assert 'times_oracle' in stats

        for key in STATIC_SURROGATE_QUERIES:
            assert key in analysis['static_template_stats']

    def test_t_stat_sign_gen_routed(self, synthetic_results):
        """With generated surrogates better than baseline, t-stat should be positive
        (ttest_rel(baseline, routed) where baseline > routed => positive t)."""
        analysis = analyze_experiment_results(synthetic_results)
        # Generated NLLs are lower by construction, so baseline > gen_routed
        # ttest_rel(baseline, gen_routed) => positive t
        assert analysis['t_stat_gen_routed'] > 0
