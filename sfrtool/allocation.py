"""Splitting a false-referral budget between the experts of an any-alarm rule.

The rule refers a person when any expert's conformal p-value is at or below that expert's alarm
level. For a healthy person the chance of a referral is then at most the sum of the levels,
whatever the dependence between the tests (the union, or Bonferroni, bound). That bound is
standard; nothing here improves it. This module only asks how a fixed sum - the budget - is best
split between the experts, given each expert's measured sensitivity curve and assumed
prevalences, and how the result compares with other ways of combining the same p-values.

Two facts shape the search:

- A conformal p-value computed against n healthy calibration cases is a multiple of 1/(n+1). A
  level between two multiples acts as the lower one, so budget placed there is never spent.
  Budgets here are sums of these effective levels.
- Combining experts never makes any one condition easier to detect: a person with a structural
  defect is referred by the structure expert at its own level, whatever the other experts do
  (other experts only add chance alarms). Splitting the budget trades detection of one condition
  against another.

Combination rules for the yes/no referral decision:

- Bonferroni: refer if any p_k <= level_k. Valid for any dependence.
- Holm (weighted step-down): its first step is the Bonferroni test, so for the yes/no referral it
  refers exactly the same people. It differs only in which further experts are named.
- Simes: refer if the i-th smallest p-value is <= i * total / m for some i. More people referred
  at the same nominal total, but the total is guaranteed only when the p-values are positively
  dependent (PRDS); under other dependence the false-referral rate can reach
  total * (1 + 1/2 + ... + 1/m). Dependence between tests that were never done on the same person
  cannot be checked, so Simes is shown for comparison only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

TOLERANCE = 1e-12


def effective_level(level: float, n_calibration: int | None) -> float:
    """The level a conformal alarm actually works at: the largest multiple of 1/(n+1) not above ``level``."""
    if n_calibration is None:
        return float(level)
    return float(np.floor(level * (n_calibration + 1) + TOLERANCE) / (n_calibration + 1))


def spendable_levels(n_calibration: int, largest: float) -> np.ndarray:
    """Every level that changes a conformal decision with n calibration cases, up to ``largest`` (0 included)."""
    steps = int(np.floor(largest * (n_calibration + 1) + TOLERANCE))
    return np.arange(steps + 1) / (n_calibration + 1)


def sensitivity_at(finding_p: np.ndarray, level: float) -> float:
    """Share of cases with the finding whose p-value is at or below ``level``."""
    finding_p = np.asarray(finding_p, dtype=float)
    finding_p = finding_p[np.isfinite(finding_p)]
    return float(np.mean(finding_p <= level + TOLERANCE)) if finding_p.size else float("nan")


@dataclass(frozen=True)
class Allocation:
    kind: str
    levels: Mapping[str, float]                  # as set
    effective: Mapping[str, float]               # as they act, given the calibration sizes
    sensitivity: Mapping[str, float] = field(default_factory=dict)
    value: float = float("nan")                  # expected true referrals per person screened

    @property
    def ceiling(self) -> float:
        return float(sum(self.levels.values()))

    @property
    def effective_ceiling(self) -> float:
        return float(sum(self.effective.values()))


def evaluate(kind: str, levels: Mapping[str, float], finding_p: Mapping[str, np.ndarray | None],
             prevalence: Mapping[str, float], n_calibration: Mapping[str, int | None]) -> Allocation:
    """Effective levels, sensitivities and expected true referrals per person of one allocation.

    Experts without cases of their finding (``finding_p`` None) add nothing to the value: their
    sensitivity is unknown, not zero, so the value is a lower bound whenever such an expert is used.
    """
    effective = {k: effective_level(v, n_calibration.get(k)) for k, v in levels.items()}
    sensitivity = {k: sensitivity_at(finding_p[k], effective[k]) for k in levels if finding_p.get(k) is not None}
    value = float(sum(prevalence[k] * s for k, s in sensitivity.items()))
    return Allocation(kind, dict(levels), effective, sensitivity, value)


MAX_OBSERVED = 1000          # above this many observed p-values the grid alone is searched


def candidate_levels(finding_p: np.ndarray | None, n_calibration: int | None, largest: float, step: float) -> np.ndarray:
    """Levels worth trying for one expert: the multiples of 1/(n+1) if n is known, else a grid plus the observed p-values.

    Sensitivity only changes at an observed p-value, so with few cases those are the exact break
    points; with many (simulations) the grid alone is fine enough.
    """
    if n_calibration is not None:
        return spendable_levels(n_calibration, largest)
    grid = np.arange(0.0, largest + TOLERANCE, step)
    if finding_p is not None:
        observed = np.asarray(finding_p, dtype=float)
        observed = observed[np.isfinite(observed) & (observed <= largest + TOLERANCE)]
        if np.unique(observed).size <= MAX_OBSERVED:
            grid = np.concatenate([grid, observed])
    return np.unique(np.round(grid, 12))


def best_allocation(budget: float, finding_p: Mapping[str, np.ndarray | None], prevalence: Mapping[str, float],
                    n_calibration: Mapping[str, int | None], fixed: Mapping[str, float], free: Sequence[str],
                    step: float = 0.0025) -> Allocation:
    """Levels of the ``free`` experts that maximise expected true referrals, the ``fixed`` levels kept.

    The constraint is on effective levels: the sum over all experts never exceeds ``budget``.
    Among allocations with the same value the one with the smaller effective ceiling is chosen
    (the same detection for fewer false referrals), then the first in the order searched.
    """
    spent = sum(effective_level(v, n_calibration.get(k)) for k, v in fixed.items())
    remaining = budget - spent
    if remaining < -TOLERANCE:
        raise ValueError(f"the fixed levels already exceed the budget ({spent:.4f} > {budget:.4f})")
    options = [candidate_levels(finding_p.get(k), n_calibration.get(k), max(remaining, 0.0), step) for k in free]
    gains = []                                   # prevalence x sensitivity at every candidate level, per free expert
    for k, levels in zip(free, options):
        found = finding_p.get(k)
        if found is None:
            gains.append(np.zeros(levels.size))
            continue
        ordered = np.sort(np.asarray(found, dtype=float)[np.isfinite(found)])
        gains.append(prevalence[k] * np.searchsorted(ordered, levels + TOLERANCE, side="right") / ordered.size)
    best_value, best_spend, best_index = -np.inf, np.inf, None
    for index in itertools.product(*(range(len(levels)) for levels in options)):
        spend = sum(options[i][j] for i, j in enumerate(index))
        if spend > remaining + TOLERANCE:
            continue
        value = sum(gains[i][j] for i, j in enumerate(index))
        if value > best_value + TOLERANCE or (abs(value - best_value) <= TOLERANCE and spend < best_spend - TOLERANCE):
            best_value, best_spend, best_index = value, spend, index
    if best_index is None:
        raise ValueError("no allocation fits the budget")
    levels = {**fixed, **{k: float(options[i][j]) for i, (k, j) in enumerate(zip(free, best_index))}}
    return evaluate("optimised", levels, finding_p, prevalence, n_calibration)


# ---- combination rules for the yes/no referral ---------------------------------------------------------

def bonferroni_any(p: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Refer if any p-value is at or below its own level. ``p`` has one row per person, one column per expert."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return np.any(p <= np.asarray(levels, dtype=float) + TOLERANCE, axis=1)


def holm_rejections(p: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Weighted Holm step-down with total = sum(levels) and weights proportional to the levels.

    Step j looks at the remaining expert with the smallest p / level and rejects it if
    p <= total * level / (sum of the remaining levels); it stops at the first failure.
    Returns a boolean array of rejections, one row per person.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    levels = np.asarray(levels, dtype=float)
    total = levels.sum()
    rejected = np.zeros(p.shape, dtype=bool)
    for row in range(p.shape[0]):
        remaining = list(range(p.shape[1]))
        while remaining:
            ratios = [p[row, k] / levels[k] if levels[k] > 0 else np.inf for k in remaining]
            k = remaining[int(np.argmin(ratios))]
            if levels[k] > 0 and p[row, k] <= total * levels[k] / levels[remaining].sum() + TOLERANCE:
                rejected[row, k] = True
                remaining.remove(k)
            else:
                break
    return rejected


def holm_any(p: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    return holm_rejections(p, levels).any(axis=1)


def simes_any(p: np.ndarray, total: float) -> np.ndarray:
    """Simes's global test with equal weights: refer if p_(i) <= i * total / m for some i."""
    p = np.sort(np.atleast_2d(np.asarray(p, dtype=float)), axis=1)
    m = p.shape[1]
    return np.any(p <= total * np.arange(1, m + 1) / m + TOLERANCE, axis=1)


# ---- null p-values under chosen dependence, for size checks --------------------------------------------

def gaussian_null_p(n: int, m: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """One-sided p-values of m equicorrelated standard normal statistics (rho >= -1/(m-1))."""
    if rho < -1.0 / (m - 1) - TOLERANCE:
        raise ValueError("equicorrelation below -1/(m-1) is not a valid correlation matrix")
    cov = np.full((m, m), rho) + (1 - rho) * np.eye(m)
    z = rng.multivariate_normal(np.zeros(m), cov, size=n, method="eigh")
    return stats.norm.sf(z)


def simes_worst_case_two(n: int, total: float, rng: np.random.Generator) -> np.ndarray:
    """Two uniform p-values whose dependence makes Simes refer healthy people at 1.5 x total (total < 2/3).

    p1 is uniform; p2 = 1 - p1 when p1 is below total/2 or above 1 - total/2, else p2 = p1.
    Both marginals are uniform. Simes refers when p1 <= total/2, when p1 >= 1 - total/2, and when
    both lie in (total/2, total]: probability 1.5 x total. Bonferroni (total/2 each) refers with
    probability exactly total, so the union bound is reached.
    """
    if not 0 < total < 2 / 3:
        raise ValueError("the construction needs 0 < total < 2/3")
    p1 = rng.random(n)
    flip = (p1 <= total / 2) | (p1 >= 1 - total / 2)
    p2 = np.where(flip, 1 - p1, p1)
    return np.column_stack([p1, p2])


# ---- per 1000 people and net benefit ---------------------------------------------------------------------

def per_thousand(value_per_person: float) -> float:
    return 1000.0 * value_per_person


def net_benefit(true_per_person: float, false_per_person: float, harm_ratio: float) -> float:
    """True referrals minus ``harm_ratio`` x false referrals, per person screened (Vickers and Elkin's net benefit).

    ``harm_ratio`` is the harm of one unnecessary referral relative to the benefit of one needed
    referral - a clinical judgement that is not made here.
    """
    return float(true_per_person - harm_ratio * false_per_person)
