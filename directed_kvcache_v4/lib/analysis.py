"""Statistical analysis utilities for v3 experiments."""
import numpy as np
from scipy import stats


def cohens_d(diff):
    """Compute Cohen's d effect size for a difference array."""
    diff = np.asarray(diff)
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0
