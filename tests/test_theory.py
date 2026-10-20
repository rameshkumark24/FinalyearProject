"""Standard results the decision rule relies on, checked numerically (references: docs/theory_notes.md).

- Conformal p-value (1 + #{calibration >= score}) / (n + 1): smallest value 1/(n+1), values on
  multiples of 1/(n+1), healthy alarm rate at most floor(alpha (n+1)) / (n+1) on average.
- For a fixed calibration set the alarm rate on future healthy cases follows Beta(k, n+1-k),
  k = floor(alpha (n+1)) (continuous scores).
- Under a shift between calibration and use, the alarm rate at a fixed threshold moves by at most
  the Kolmogorov-Smirnov distance between the two score distributions.
- Group-conditional (Mondrian) calibration keeps the rate at the level inside every group; a
  group needs ceil(1/alpha) - 1 healthy calibration cases to alarm at all.

Synthetic scores only.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sfr.fusion.conformal import p_value
from sfrtool.allocation import effective_level
from sfrtool.conformal_groups import group_conditional_p_values, minimum_group_size


def alarm_rate(calibration: np.ndarray, test: np.ndarray, level: float) -> float:
    ordered = np.sort(calibration)
    at_least = ordered.size - np.searchsorted(ordered, test, side="left")
    return float(np.mean((1 + at_least) / (ordered.size + 1) <= level + 1e-12))


def test_p_value_formula_smallest_value_and_grid():
    calibration = np.arange(14, dtype=float)
    assert p_value(100.0, calibration) == pytest.approx(1 / 15)
    assert p_value(-1.0, calibration) == pytest.approx(1.0)
    values = np.array([p_value(s, calibration) for s in np.linspace(-2, 16, 101)])
    assert np.allclose(values * 15, np.round(values * 15))


@pytest.mark.parametrize("n,level", [(19, 0.05), (14, 0.10), (60, 0.05)])
def test_conformal_alarm_rate_is_at_most_the_effective_level_on_average(n, level):
    rng = np.random.default_rng(n)
    rates = np.array([alarm_rate(rng.normal(size=n), rng.normal(size=400), level) for _ in range(3000)])
    assert rates.mean() == pytest.approx(effective_level(level, n), abs=4 * rates.std(ddof=1) / np.sqrt(rates.size))


def test_too_few_calibration_cases_never_alarm():
    rng = np.random.default_rng(0)
    assert alarm_rate(rng.normal(size=14), rng.normal(size=5000), 0.05) == 0.0


def test_conditional_alarm_rate_follows_the_beta_law():
    n, level = 14, 0.10
    k = int(np.floor(level * (n + 1)))
    rng = np.random.default_rng(1)
    # uniform scores: a new case alarms when it exceeds the k-th largest calibration score
    rates = np.array([1 - np.sort(rng.random(n))[n - k] for _ in range(5000)])
    assert stats.kstest(rates, stats.beta(k, n + 1 - k).cdf).pvalue > 0.001


def test_shift_moves_the_rate_at_a_threshold_by_at_most_the_ks_distance():
    rng = np.random.default_rng(2)
    p_scores, q_scores = rng.normal(size=300), rng.normal(0.7, 1.3, size=500)
    ks = stats.ks_2samp(p_scores, q_scores).statistic
    for threshold in np.linspace(-3, 4, 71):
        assert abs(np.mean(q_scores >= threshold) - np.mean(p_scores >= threshold)) <= ks + 1e-12


def test_group_conditional_calibration_keeps_the_level_inside_each_group():
    rng = np.random.default_rng(3)
    level, rates = 0.05, {0: [], 1: []}
    for _ in range(400):
        g_cal = rng.random(400) < 0.3
        cal = rng.normal(size=400) + g_cal
        g_test = rng.random(2000) < 0.3
        test = rng.normal(size=2000) + g_test
        p = group_conditional_p_values(test, g_test, cal, g_cal)
        for group in (0, 1):
            rates[group].append(np.mean(p[g_test == group] <= level))
    for group in (0, 1):
        assert np.mean(rates[group]) <= level + 0.004


def test_minimum_group_size_and_groups_without_calibration():
    assert minimum_group_size(0.05) == 19
    assert minimum_group_size(0.10) == 9
    p = group_conditional_p_values([1.0, 2.0], ["a", "b"], [0.0, 0.5], ["a", "a"])
    assert p[0] == pytest.approx(1 / 3) and np.isnan(p[1])
