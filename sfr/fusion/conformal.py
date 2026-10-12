"""Conformal p-values against healthy cases, and what small samples allow.

An expert's score is turned into a p-value by ranking it among the scores of healthy cases the
expert has not been fitted on. Flagging when p <= alpha then flags a healthy case with
probability at most alpha, whatever the score distribution (exchangeability is the only
assumption). With n healthy cases the smallest possible p-value is 1/(n+1), so an alarm level
below that can never fire.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def p_value(score: float, healthy_scores: np.ndarray) -> float:
    """Share of healthy scores at least as high as ``score``, counting the case itself."""
    healthy_scores = np.asarray(healthy_scores, dtype=float)
    healthy_scores = healthy_scores[np.isfinite(healthy_scores)]
    if not np.isfinite(score):
        return float("nan")
    return float((1 + np.sum(healthy_scores >= score)) / (healthy_scores.size + 1))


def leave_one_out_p_values(healthy_scores: np.ndarray) -> np.ndarray:
    """p-value of every healthy case against the other healthy cases."""
    healthy_scores = np.asarray(healthy_scores, dtype=float)
    return np.array([p_value(s, np.delete(healthy_scores, i)) for i, s in enumerate(healthy_scores)])


def smallest_p(n_healthy: int) -> float:
    return 1.0 / (n_healthy + 1)


def realised_false_alarm_range(n_healthy: int, alpha: float, level: float = 0.90) -> tuple[float, float]:
    """Range in which the false-alarm rate of a fixed calibration set falls with probability ``level``.

    With n calibration cases and alarm level alpha, the rate actually obtained on future healthy
    cases follows Beta(k, n + 1 - k) with k = floor((n + 1) alpha) (Angelopoulos and Bates,
    "A gentle introduction to conformal prediction", section 3.2). If k = 0 no alarm can fire.
    """
    k = int(np.floor((n_healthy + 1) * alpha))
    if k == 0:
        return 0.0, 0.0
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    return float(stats.beta.ppf(low, k, n_healthy + 1 - k)), float(stats.beta.ppf(high, k, n_healthy + 1 - k))
