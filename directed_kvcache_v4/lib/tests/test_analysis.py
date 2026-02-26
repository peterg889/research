"""Tests for lib.analysis — statistical utilities.

Covers cohens_d, win_rate, paired_ttest with extensive edge cases,
mathematical property checks, and cross-function consistency tests.
"""

import numpy as np
import pytest
from scipy import stats

from lib.analysis import cohens_d, paired_ttest, win_rate


# ======================================================================
# cohens_d
# ======================================================================

class TestCohensDBasic:
    """Basic correctness for cohens_d."""

    def test_positive_effect(self):
        d = cohens_d([0.9, 1.0, 1.1, 1.0])
        assert d > 0

    def test_negative_effect(self):
        d = cohens_d([-0.5, -0.6, -0.4, -0.5])
        assert d < 0

    def test_zero_std_returns_zero(self):
        assert cohens_d([0.5, 0.5, 0.5]) == 0.0

    def test_all_zeros(self):
        assert cohens_d([0.0, 0.0, 0.0]) == 0.0

    def test_single_element(self):
        assert cohens_d([1.0]) == 0.0

    def test_empty_array(self):
        assert cohens_d([]) == 0.0

    def test_two_elements(self):
        d = cohens_d([1.0, 2.0])
        expected = np.mean([1.0, 2.0]) / np.std([1.0, 2.0], ddof=1)
        assert abs(d - expected) < 1e-10

    def test_known_value(self):
        diff = [1.0, 2.0, 3.0, 4.0]
        expected = np.mean(diff) / np.std(diff, ddof=1)
        assert abs(cohens_d(diff) - expected) < 1e-10


class TestCohensDMathProperties:
    """Mathematical properties and invariants."""

    def test_sign_follows_mean(self):
        """d should be positive when mean is positive, negative when negative."""
        pos = np.random.RandomState(42).normal(1.0, 0.5, 50)
        neg = np.random.RandomState(42).normal(-1.0, 0.5, 50)
        assert cohens_d(pos) > 0
        assert cohens_d(neg) < 0

    def test_scaling_invariance(self):
        """Multiplying all values by a constant should not change d."""
        diff = [0.5, 0.3, 0.7, 0.4, 0.6]
        d1 = cohens_d(diff)
        d2 = cohens_d([x * 100 for x in diff])
        assert abs(d1 - d2) < 1e-10

    def test_negation_flips_sign(self):
        """Negating all values should negate d."""
        diff = [0.5, 0.3, 0.7, 0.4]
        d_pos = cohens_d(diff)
        d_neg = cohens_d([-x for x in diff])
        assert abs(d_pos + d_neg) < 1e-10

    def test_shift_changes_d(self):
        """Adding a constant shifts d proportionally to the constant / std."""
        diff = [0.0, 0.1, -0.1, 0.05, -0.05]
        d_orig = cohens_d(diff)
        shifted = [x + 10.0 for x in diff]
        d_shifted = cohens_d(shifted)
        assert d_shifted > d_orig  # Positive shift increases d

    def test_large_n_converges(self):
        """For large samples from N(mu, sigma), d ≈ mu/sigma."""
        rng = np.random.RandomState(42)
        mu, sigma = 0.5, 1.0
        diff = rng.normal(mu, sigma, 10000)
        d = cohens_d(diff)
        assert abs(d - mu / sigma) < 0.1  # Within 0.1 of theoretical

    def test_magnitude_ordering(self):
        """Stronger effects should have larger |d|."""
        rng = np.random.RandomState(42)
        weak = rng.normal(0.1, 1.0, 200)
        strong = rng.normal(1.0, 1.0, 200)
        assert abs(cohens_d(strong)) > abs(cohens_d(weak))

    def test_returns_float(self):
        assert isinstance(cohens_d([1.0, 2.0, 3.0]), float)
        assert isinstance(cohens_d(np.array([1.0, 2.0])), float)


class TestCohensDInputTypes:
    """Input type handling."""

    def test_numpy_array(self):
        arr = np.array([0.5, 0.3, 0.7])
        assert isinstance(cohens_d(arr), float)

    def test_tuple_input(self):
        d = cohens_d((0.5, 0.3, 0.7))
        assert isinstance(d, float)

    def test_integer_input(self):
        d = cohens_d([1, 2, 3, 4])
        assert abs(d - cohens_d([1.0, 2.0, 3.0, 4.0])) < 1e-10

    def test_mixed_types(self):
        d = cohens_d([1, 2.0, 3, 4.5])
        assert isinstance(d, float)


class TestCohensDRealisticExperiment:
    """Tests mimicking actual experiment data patterns."""

    def test_typical_nll_improvement(self):
        """Typical experiment: NLL drops ~0.5 nats with high consistency."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0.5, 0.3, 160)  # 160 hard samples
        d = cohens_d(diff)
        assert 1.0 < d < 2.5  # Strong effect

    def test_weak_nll_improvement(self):
        """Weak effect: NLL drops ~0.05 with high variance."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0.05, 0.5, 160)
        d = cohens_d(diff)
        assert abs(d) < 0.3  # Small effect

    def test_oracle_hurts(self):
        """Oracle condition: NLL increases (negative diff = hurt)."""
        rng = np.random.RandomState(42)
        diff = rng.normal(-0.15, 0.6, 400)
        d = cohens_d(diff)
        assert d < 0

    def test_large_sample_many_datasets(self):
        """Pooled across 10 datasets: 2000 samples."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0.3, 0.8, 2000)
        d = cohens_d(diff)
        assert 0.2 < d < 0.6


# ======================================================================
# win_rate
# ======================================================================

class TestWinRateBasic:
    def test_all_positive(self):
        assert win_rate([1.0, 2.0, 3.0]) == 1.0

    def test_all_negative(self):
        assert win_rate([-1.0, -2.0, -3.0]) == 0.0

    def test_mixed(self):
        assert abs(win_rate([1.0, -1.0, 1.0, -1.0]) - 0.5) < 1e-10

    def test_zeros_are_not_wins(self):
        """Zero values should NOT count as wins."""
        assert win_rate([0.0, 0.0, 0.0]) == 0.0

    def test_empty(self):
        assert win_rate([]) == 0.0

    def test_single_positive(self):
        assert win_rate([0.1]) == 1.0

    def test_single_negative(self):
        assert win_rate([-0.1]) == 0.0

    def test_single_zero(self):
        assert win_rate([0.0]) == 0.0


class TestWinRateMathProperties:
    def test_complement(self):
        """win_rate(x) + win_rate(-x) + zero_rate = 1.0."""
        diff = [0.5, -0.1, 0.3, 0.0, -0.2]
        wr = win_rate(diff)
        loss_rate = win_rate([-x for x in diff])
        zero_rate = np.mean(np.array(diff) == 0)
        assert abs(wr + loss_rate + zero_rate - 1.0) < 1e-10

    def test_bounded_zero_one(self):
        """Win rate should always be in [0, 1]."""
        for seed in range(10):
            rng = np.random.RandomState(seed)
            diff = rng.normal(0, 1, 100)
            wr = win_rate(diff)
            assert 0.0 <= wr <= 1.0

    def test_monotonic_with_shift(self):
        """Shifting all values up should increase win rate."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 1, 200)
        wr_base = win_rate(base)
        wr_shifted = win_rate(base + 2.0)
        assert wr_shifted >= wr_base

    def test_numpy_input(self):
        arr = np.array([0.5, -0.1, 0.3])
        assert abs(win_rate(arr) - 2 / 3) < 1e-10

    def test_large_sample_converges(self):
        """For N(1, 1), win rate should be ~84% (Φ(1))."""
        rng = np.random.RandomState(42)
        diff = rng.normal(1.0, 1.0, 100000)
        wr = win_rate(diff)
        assert abs(wr - 0.8413) < 0.02


class TestWinRateRealisticExperiment:
    def test_strong_improvement_high_win(self):
        """Strong effect should give >80% win rate."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0.5, 0.3, 160)
        assert win_rate(diff) > 0.8

    def test_null_effect_near_50(self):
        """Null effect should give ~50% win rate."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0, 1, 1000)
        wr = win_rate(diff)
        assert 0.45 < wr < 0.55

    def test_very_tiny_improvements_counted(self):
        """Even epsilon-positive values should count as wins."""
        diff = [1e-10, 1e-15, 1e-20, -1e-10]
        assert win_rate(diff) == 0.75


# ======================================================================
# paired_ttest
# ======================================================================

class TestPairedTtestBasic:
    def test_significant_positive(self):
        diff = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.5]
        t, p = paired_ttest(diff)
        assert t > 0
        assert p < 0.01

    def test_significant_negative(self):
        diff = [-0.5, -0.6, -0.4, -0.7, -0.5]
        t, p = paired_ttest(diff)
        assert t < 0
        assert p < 0.01

    def test_null_effect_not_significant(self):
        np.random.seed(42)
        diff = np.random.normal(0, 1, 1000)
        _, p = paired_ttest(diff)
        assert p > 0.01

    def test_single_element(self):
        t, p = paired_ttest([1.0])
        assert t == 0.0
        assert p == 1.0

    def test_empty(self):
        t, p = paired_ttest([])
        assert t == 0.0
        assert p == 1.0

    def test_two_elements(self):
        t, p = paired_ttest([1.0, 2.0])
        t_scipy, p_scipy = stats.ttest_1samp([1.0, 2.0], 0)
        assert abs(t - t_scipy) < 1e-10
        assert abs(p - p_scipy) < 1e-10


class TestPairedTtestMathProperties:
    def test_matches_scipy(self):
        diff = [0.1, 0.3, -0.1, 0.2, 0.4]
        t_ours, p_ours = paired_ttest(diff)
        t_scipy, p_scipy = stats.ttest_1samp(diff, 0)
        assert abs(t_ours - t_scipy) < 1e-10
        assert abs(p_ours - p_scipy) < 1e-10

    def test_t_sign_follows_mean(self):
        """t-statistic should be positive when mean is positive."""
        pos_diff = [0.5, 0.3, 0.7, 0.4]
        neg_diff = [-0.5, -0.3, -0.7, -0.4]
        t_pos, _ = paired_ttest(pos_diff)
        t_neg, _ = paired_ttest(neg_diff)
        assert t_pos > 0
        assert t_neg < 0

    def test_negation_flips_t_sign(self):
        diff = [0.1, 0.3, -0.1, 0.2, 0.4]
        t_pos, p_pos = paired_ttest(diff)
        t_neg, p_neg = paired_ttest([-x for x in diff])
        assert abs(t_pos + t_neg) < 1e-10
        assert abs(p_pos - p_neg) < 1e-10  # p-value unchanged

    def test_scaling_does_not_change_t(self):
        """Multiplying all values by a constant should not change t or p."""
        diff = [0.5, 0.3, 0.7, 0.4, 0.6]
        t1, p1 = paired_ttest(diff)
        t2, p2 = paired_ttest([x * 1000 for x in diff])
        assert abs(t1 - t2) < 1e-8
        assert abs(p1 - p2) < 1e-8

    def test_constant_array_gives_inf_t(self):
        """Constant nonzero array: std=0, scipy gives inf t and p≈0."""
        diff = [1.0, 1.0, 1.0, 1.0, 1.0]
        t, p = paired_ttest(diff)
        assert np.isinf(t)  # scipy returns inf
        assert p < 1e-10

    def test_larger_n_lower_p(self):
        """More samples from the same distribution should give lower p."""
        rng = np.random.RandomState(42)
        diff_small = rng.normal(0.5, 1.0, 10)
        diff_large = rng.normal(0.5, 1.0, 1000)
        _, p_small = paired_ttest(diff_small)
        _, p_large = paired_ttest(diff_large)
        assert p_large < p_small

    def test_returns_float_tuple(self):
        t, p = paired_ttest([0.1, 0.2, 0.3])
        assert isinstance(t, float)
        assert isinstance(p, float)


class TestPairedTtestRealistic:
    def test_exp09_normalization_significance(self):
        """Simulated Exp 09-like data: normalization drops NLL by ~1.5 nats."""
        rng = np.random.RandomState(42)
        diff = rng.normal(1.5, 1.0, 200)
        t, p = paired_ttest(diff)
        assert t > 10  # Very significant
        assert p < 1e-20

    def test_nonsignificant_small_sample(self):
        """Small n=5 with noisy data should not be significant."""
        diff = [0.1, -0.3, 0.2, -0.1, 0.05]
        _, p = paired_ttest(diff)
        assert p > 0.05


# ======================================================================
# Cross-function consistency
# ======================================================================

class TestCrossFunctionConsistency:
    """Verify that cohens_d, win_rate, and paired_ttest agree."""

    def test_strong_effect_all_agree(self):
        """Strong positive effect: high d, high win_rate, low p."""
        rng = np.random.RandomState(42)
        diff = rng.normal(1.0, 0.3, 200)
        d = cohens_d(diff)
        wr = win_rate(diff)
        _, p = paired_ttest(diff)
        assert d > 2.0
        assert wr > 0.95
        assert p < 1e-10

    def test_null_effect_all_agree(self):
        """Null effect: d ≈ 0, win_rate ≈ 0.5, p > 0.05."""
        rng = np.random.RandomState(42)
        diff = rng.normal(0, 1, 500)
        d = cohens_d(diff)
        wr = win_rate(diff)
        _, p = paired_ttest(diff)
        assert abs(d) < 0.2
        assert 0.4 < wr < 0.6
        assert p > 0.05

    def test_negative_effect_all_agree(self):
        """Negative effect: negative d, low win_rate, low p."""
        rng = np.random.RandomState(42)
        diff = rng.normal(-0.8, 0.4, 200)
        d = cohens_d(diff)
        wr = win_rate(diff)
        _, p = paired_ttest(diff)
        assert d < -1.0
        assert wr < 0.1
        assert p < 1e-10

    def test_d_sign_matches_win_rate(self):
        """If d > 0, win_rate should be > 0.5 (for large enough n)."""
        rng = np.random.RandomState(42)
        for mu in [-1.0, -0.5, 0.5, 1.0]:
            diff = rng.normal(mu, 1.0, 500)
            d = cohens_d(diff)
            wr = win_rate(diff)
            if d > 0.2:
                assert wr > 0.5
            elif d < -0.2:
                assert wr < 0.5


# ======================================================================
# NaN and Inf input handling
# ======================================================================

class TestNaNInfInputs:
    """Verify behavior when inputs contain NaN or Inf values.

    These can appear in experiment data from degenerate samples (e.g.
    division by zero in NLL computation). The functions should propagate
    NaN/Inf following NumPy conventions, not crash.
    """

    def test_cohens_d_with_nan(self):
        """NaN in input should propagate to NaN output."""
        result = cohens_d([1.0, float("nan"), 2.0])
        assert np.isnan(result)

    def test_cohens_d_with_inf(self):
        """Inf in input should not crash."""
        result = cohens_d([1.0, float("inf"), 2.0])
        # std will be inf, so d = mean/inf = 0 or nan depending on mean
        assert isinstance(result, float)

    def test_cohens_d_all_nan(self):
        result = cohens_d([float("nan"), float("nan")])
        assert np.isnan(result)

    def test_win_rate_with_nan(self):
        """NaN comparisons are False, so NaN should not count as a win."""
        result = win_rate([1.0, float("nan"), -1.0])
        # nan > 0 is False, so only 1 win out of 3
        assert abs(result - 1 / 3) < 1e-10

    def test_win_rate_with_inf(self):
        """Positive inf should count as a win."""
        result = win_rate([float("inf"), float("-inf"), 1.0])
        assert abs(result - 2 / 3) < 1e-10

    def test_win_rate_all_nan(self):
        """All NaN should give 0% win rate (nan > 0 is False)."""
        result = win_rate([float("nan"), float("nan")])
        assert result == 0.0

    def test_paired_ttest_with_nan(self):
        """NaN in input should propagate to NaN t-stat and p-value."""
        t, p = paired_ttest([1.0, float("nan"), 2.0])
        assert np.isnan(t)
        assert np.isnan(p)

    def test_paired_ttest_with_inf(self):
        """Inf in input should not crash."""
        t, p = paired_ttest([1.0, float("inf"), 2.0])
        assert isinstance(t, float)
        assert isinstance(p, float)
