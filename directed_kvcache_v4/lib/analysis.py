"""Statistical analysis utilities for KV cache experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def cohens_d(diff: ArrayLike) -> float:
    """Compute Cohen's d effect size for a paired-difference array.

    Cohen's d is the mean difference divided by the standard deviation,
    giving a standardized measure of effect size.  Conventions:
    ``|d| < 0.2`` = small, ``0.5`` = medium, ``0.8`` = large.

    Args:
        diff: Array-like of paired differences (e.g. ``nll_baseline - nll_treatment``).
            A positive value means the treatment is better (lower NLL).

    Returns:
        Cohen's d as a float.  Returns ``0.0`` if the standard deviation
        is zero (constant array).

    Example::

        >>> cohens_d([0.5, 0.3, 0.7, 0.4])
        2.853...
        >>> cohens_d([0.0, 0.0, 0.0])
        0.0
    """
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def win_rate(diff: ArrayLike) -> float:
    """Compute the win rate (fraction of positive differences).

    Args:
        diff: Array-like of paired differences.  Positive values count as
            "wins" for the treatment.

    Returns:
        Fraction in ``[0, 1]``.  Returns ``0.0`` for empty input.

    Example::

        >>> win_rate([0.5, -0.1, 0.3, 0.0])
        0.5
    """
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return 0.0
    return float(np.mean(diff > 0))


def paired_ttest(diff: ArrayLike) -> tuple[float, float]:
    """One-sample t-test on paired differences (H0: mean = 0).

    Args:
        diff: Array-like of paired differences.

    Returns:
        Tuple of ``(t_statistic, p_value)``.  Returns ``(0.0, 1.0)`` if
        the array has fewer than 2 elements.

    Example::

        >>> t, p = paired_ttest([0.5, 0.3, 0.7, 0.4, 0.6])
        >>> t > 0
        True
    """
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    return float(t_stat), float(p_val)
