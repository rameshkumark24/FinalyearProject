"""Budgeted alarm levels and the combination rules of sfrtool.allocation.

- An allocation never spends more than its budget, counted in effective levels.
- The search finds the same optimum as a brute-force search, and for smooth concave sensitivity
  curves it lands on the analytic optimum (where the marginal gains are equal).
- For the yes/no referral, Holm's procedure refers exactly the people Bonferroni refers.
- Bonferroni (and so Holm) keeps the false-referral rate at or below the total for any dependence;
  Simes does so under positive dependence, and exceeds it for a dependence built against it.

Random numbers draw synthetic p-values for these tests only.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sfrtool.allocation import (best_allocation, bonferroni_any, effective_level, evaluate, gaussian_null_p, holm_any,
                                holm_rejections, net_benefit, sensitivity_at, simes_any, simes_worst_case_two, spendable_levels)

N = 100_000


def se(rate: float, n: int = N) -> float:
    return float(np.sqrt(rate * (1 - rate) / n))


def test_effective_level_is_the_largest_multiple_of_one_over_n_plus_one():
    assert effective_level(0.10, 14) == pytest.approx(1 / 15)
    assert effective_level(0.05, 14) == 0.0
    assert effective_level(0.05, 19) == pytest.approx(0.05)
    assert effective_level(0.07, None) == pytest.approx(0.07)
    assert np.allclose(spendable_levels(14, 0.2), [0, 1 / 15, 2 / 15, 3 / 15])


@pytest.mark.parametrize("seed", range(5))
def test_allocation_never_exceeds_the_budget(seed):
    rng = np.random.default_rng(seed)
    finding = {"structure": rng.beta(0.4, 2, 120), "function": None, "rhythm": rng.beta(0.3, 2, 12)}
    n_cal = {"structure": None, "function": None, "rhythm": int(rng.integers(9, 30))}
    prevalence = {"structure": 0.009, "function": 0.002, "rhythm": float(rng.uniform(0.0005, 0.01))}
    for budget in (0.08, 0.15, 0.2, 0.3):
        fixed = {"function": budget / 4}
        best = best_allocation(budget, finding, prevalence, n_cal, fixed, ("rhythm", "structure"), step=0.005)
        assert best.effective_ceiling <= budget + 1e-9
        assert best.levels["function"] == pytest.approx(budget / 4)


def test_search_matches_brute_force():
    rng = np.random.default_rng(1)
    finding = {"a": rng.beta(0.5, 3, 80), "b": rng.beta(0.3, 3, 15)}
    n_cal = {"a": None, "b": 14}
    prevalence = {"a": 0.01, "b": 0.003}
    budget = 0.2
    best = best_allocation(budget, finding, prevalence, n_cal, {}, ("b", "a"), step=0.001)
    grid_a = np.unique(np.concatenate([np.arange(0, budget + 1e-12, 0.001), finding["a"][finding["a"] <= budget]]))
    brute = max(prevalence["a"] * sensitivity_at(finding["a"], a) + prevalence["b"] * sensitivity_at(finding["b"], b)
                for a, b in itertools.product(grid_a, spendable_levels(14, budget)) if a + b <= budget + 1e-12)
    assert best.value == pytest.approx(brute)


def test_concave_curves_reach_the_analytic_optimum():
    # Sensitivity curves s_k(a) = a ** g (p-values with distribution function x ** g, concave for g < 1).
    rng = np.random.default_rng(2)
    g, budget = 0.5, 0.2
    finding = {"a": rng.random(200_000) ** (1 / g), "b": rng.random(200_000) ** (1 / g)}
    prevalence = {"a": 0.009, "b": 0.003}
    best = best_allocation(budget, finding, prevalence, {"a": None, "b": None}, {}, ("a", "b"), step=0.0005)
    ratio = (prevalence["a"] / prevalence["b"]) ** (1 / (1 - g))          # equal marginal gains: a / b
    analytic_a = budget * ratio / (1 + ratio)
    assert best.levels["a"] == pytest.approx(analytic_a, abs=0.003)


def test_holm_refers_exactly_whom_bonferroni_refers():
    rng = np.random.default_rng(3)
    p = rng.random((20_000, 3)) ** 2
    for levels in ([0.05, 0.05, 0.10], [0.1, 0.0, 0.05], [1 / 15, 1 / 15, 1 / 15]):
        assert np.array_equal(holm_any(p, levels), bonferroni_any(p, levels))
    # Holm may name more experts than Bonferroni once one has alarmed
    rejected = holm_rejections(np.array([[0.01, 0.12, 0.9]]), [0.1, 0.1, 0.1])
    assert rejected.tolist() == [[True, True, False]]


def _dependent_nulls(kind: str, rng: np.random.Generator, total: float) -> tuple[np.ndarray, list[float]]:
    if kind == "independent":
        return rng.random((N, 3)), [total / 3] * 3
    if kind == "negative":
        return gaussian_null_p(N, 3, -0.45, rng), [total / 3] * 3
    if kind == "never_together":
        u = rng.random(N)
        levels = [total / 3] * 3
        offsets = np.array([0.0, total / 3, 2 * total / 3])
        return np.mod(u[:, None] - offsets[None, :], 1.0), levels
    if kind == "simes_worst_case":
        return simes_worst_case_two(N, total, rng), [total / 2] * 2
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["independent", "negative", "never_together", "simes_worst_case"])
def test_bonferroni_and_holm_keep_the_total_under_any_dependence(kind):
    rng = np.random.default_rng(4)
    total = 0.2
    p, levels = _dependent_nulls(kind, rng, total)
    rate = bonferroni_any(p, levels).mean()
    assert rate <= total + 3 * se(total)
    assert holm_any(p[:5000], levels).mean() == pytest.approx(bonferroni_any(p[:5000], levels).mean())


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8])
def test_simes_keeps_the_total_under_positive_dependence(rho):
    rng = np.random.default_rng(5)
    p = rng.random((N, 3)) if rho == 0 else gaussian_null_p(N, 3, rho, rng)
    assert simes_any(p, 0.2).mean() <= 0.2 + 3 * se(0.2)


def test_simes_exceeds_the_total_when_the_dependence_works_against_it():
    rng = np.random.default_rng(6)
    p = simes_worst_case_two(N, 0.2, rng)
    assert simes_any(p, 0.2).mean() == pytest.approx(0.3, abs=4 * se(0.3))
    assert bonferroni_any(p, [0.1, 0.1]).mean() == pytest.approx(0.2, abs=4 * se(0.2))   # the union bound is reached


def test_unknown_sensitivity_adds_nothing_to_the_value():
    finding = {"structure": np.array([0.01, 0.2]), "function": None}
    alloc = evaluate("x", {"structure": 0.05, "function": 0.05}, finding, {"structure": 0.01, "function": 0.5}, {})
    assert alloc.value == pytest.approx(0.01 * 0.5)
    assert "function" not in alloc.sensitivity


def test_net_benefit():
    assert net_benefit(0.01, 0.1, 0.05) == pytest.approx(0.005)
    assert net_benefit(0.0, 0.0, 0.3) == 0.0
