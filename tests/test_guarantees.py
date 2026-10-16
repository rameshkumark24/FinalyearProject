"""The two guarantees of the decision rule hold for any dependence between the experts.

1. False referrals. On every sample of healthy fetuses, the number referred lies between the
   largest number of alarms of one expert and the sum of the experts' alarm numbers, whatever
   the joint behaviour of the tests. With valid p-values each expert alarms on at most its level
   of healthy fetuses, so the share referred is at most the sum of the levels.
2. Sensitivity. A fetus whose finding makes its expert alarm is always referred, whatever the
   other tests show, so for every finding the rule detects at least what its expert detects.

The dependence structures cover independent, tied and opposed experts, alarms that never come
together (the worst case), Gaussian copulas with random correlations, and conformal p-values
from finite healthy calibration sets. Random numbers draw the cases (synthetic, for these
tests only). The helpers of scripts/run_wp7_dependence_bounds.py are tested as well.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from sfr.fusion.conformal import p_value
from sfr.fusion.decision import EXPERTS, DecisionConfig, ExpertResult, decide

ROOT = Path(__file__).resolve().parent.parent
CONFIG = DecisionConfig()
ALPHA = np.array([CONFIG.alpha[name] for name in EXPERTS])
SEEDS = range(6)
N_CASES = 2500
CALIBRATION_SIZES = (62, 60, 14)          # healthy calibration cases of the three experts in this project


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bounds = load("run_wp7_dependence_bounds")


# ---- healthy p-values with different dependence between the experts ----------------------------------

def random_correlation(rng: np.random.Generator, size: int = 3) -> np.ndarray:
    loadings = rng.normal(size=(size, size))
    covariance = loadings @ loadings.T
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def healthy_p_values(kind: str, rng: np.random.Generator, n: int = N_CASES) -> np.ndarray:
    """n x 3 p-values of healthy fetuses; every column is a valid p-value."""
    u = rng.uniform(size=n)
    if kind == "independent":
        return rng.uniform(size=(n, 3))
    if kind == "tied":
        return np.repeat(u[:, None], 3, axis=1)
    if kind == "opposed":
        return np.column_stack([u, 1 - u, rng.uniform(size=n)])
    if kind == "never_together":                       # alarm regions of the three experts do not overlap
        offsets = np.concatenate([[0.0], np.cumsum(ALPHA)[:-1]])
        return np.mod(u[:, None] - offsets[None, :], 1.0)
    if kind == "gaussian_copula":
        latent = rng.multivariate_normal(np.zeros(3), random_correlation(rng), size=n)
        return stats.norm.cdf(latent)
    if kind == "conformal":                            # finite calibration sets, dependent test scores
        latent = rng.multivariate_normal(np.zeros(3), random_correlation(rng), size=n)
        p = np.empty((n, 3))
        for k, size in enumerate(CALIBRATION_SIZES):
            calibration = rng.normal(size=size)
            p[:, k] = [p_value(float(score), calibration) for score in latent[:, k]]
        return p
    raise ValueError(kind)


KINDS = ("independent", "tied", "opposed", "never_together", "gaussian_copula", "conformal")


def refer(p_row: np.ndarray) -> bool:
    results = {name: ExpertResult(present=True, quality_ok=True, p_value=float(p)) for name, p in zip(EXPERTS, p_row)}
    return decide(results, True, CONFIG).action == "REFER"


@pytest.mark.parametrize("kind", KINDS)
def test_referrals_lie_within_the_frechet_bounds_of_the_alarms_for_any_dependence(kind):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        p = healthy_p_values(kind, rng)
        alarms = p <= ALPHA[None, :]
        referred = np.array([refer(row) for row in p])
        assert np.array_equal(referred, alarms.any(axis=1))           # the rule is the OR of the alarms
        per_expert = alarms.sum(axis=0)
        assert per_expert.max() <= referred.sum() <= per_expert.sum()  # exact, on every sample


@pytest.mark.parametrize("kind", KINDS)
def test_healthy_referral_rate_is_at_most_the_sum_of_the_levels(kind):
    referred = total = 0
    for seed in SEEDS:
        p = healthy_p_values(kind, np.random.default_rng(100 + seed))
        referred += int(sum(refer(row) for row in p))
        total += p.shape[0]
    ceiling = CONFIG.false_referral_ceiling
    assert referred / total <= ceiling + 4 * np.sqrt(ceiling * (1 - ceiling) / total)


def test_the_ceiling_is_reached_when_alarms_never_come_together_and_halved_when_tied():
    u = (np.arange(20000) + 0.5) / 20000                  # evenly spread, no sampling noise
    offsets = np.concatenate([[0.0], np.cumsum(ALPHA)[:-1]])
    apart = np.mod(u[:, None] - offsets[None, :], 1.0)
    tied = np.repeat(u[:, None], 3, axis=1)
    assert (apart <= ALPHA).any(axis=1).mean() == pytest.approx(ALPHA.sum(), abs=1e-4)
    assert (tied <= ALPHA).any(axis=1).mean() == pytest.approx(ALPHA.max(), abs=1e-4)


def test_rule_detects_a_finding_at_least_as_often_as_its_expert_for_any_dependence():
    rng = np.random.default_rng(7)
    for kind in KINDS:
        for finding, name in enumerate(EXPERTS):
            p = healthy_p_values(kind, rng, n=600)
            p[:, finding] = rng.beta(0.4, 2.0, size=p.shape[0])     # the finding makes its expert's p-value small
            caught_by_expert = caught_by_rule = 0
            for row in p:
                results = {n: ExpertResult(present=True, p_value=float(v)) for n, v in zip(EXPERTS, row)}
                # whatever else happens: other tests missing, unreadable, or the trust check failing
                other = [n for n in EXPERTS if n != name][int(rng.integers(2))]
                state = int(rng.integers(3))
                if state == 1:
                    results[other] = ExpertResult(present=False)
                elif state == 2:
                    results[other] = ExpertResult(present=True, quality_ok=False, p_value=float(rng.uniform()))
                trust = [True, False, None][int(rng.integers(3))]
                expert_alarm = row[finding] <= CONFIG.alpha[name]
                referred = decide(results, trust, CONFIG).action == "REFER"
                assert referred or not expert_alarm
                caught_by_expert += expert_alarm
                caught_by_rule += referred
            assert caught_by_rule >= caught_by_expert


# ---- helpers of the dependence-bounds run -------------------------------------------------------------

def test_frechet_range_brackets_the_dependence_dial():
    rng = np.random.default_rng(3)
    for _ in range(2):
        rates = rng.uniform(0.01, 0.15, size=3)
        low, high = bounds.frechet_range(rates)
        for rho in (-0.45, -0.2, 0.0, 0.4, 0.8, 1.0):
            value = bounds.gaussian_copula_any_alarm(rates, rho)
            assert low - 1e-6 <= value <= high + 1e-6
        assert bounds.gaussian_copula_any_alarm(rates, 0.0) == pytest.approx(bounds.independent_rate(rates), abs=1e-6)
        assert bounds.gaussian_copula_any_alarm(rates, 1.0) == pytest.approx(low)
    assert bounds.gaussian_copula_any_alarm([0.1, 0.05], -1.0) == pytest.approx(0.15)


def test_dependence_dial_matches_the_one_factor_formula():
    rates = np.array([0.065, 0.033, 0.071])
    z = stats.norm.ppf(1 - rates)
    for rho in (0.2, 0.5, 0.8):
        def integrand(f):
            return stats.norm.pdf(f) * np.prod(stats.norm.cdf((z - np.sqrt(rho) * f) / np.sqrt(1 - rho)))
        none, _ = integrate.quad(integrand, -10, 10, epsabs=1e-12)
        assert bounds.gaussian_copula_any_alarm(rates, rho) == pytest.approx(1 - none, abs=1e-6)


def test_widened_range_contains_the_range_at_the_measured_rates():
    result = bounds.range_with_uncertainty([(40, 619), (2, 60), (1, 14)])
    assert result["range_with_uncertainty"][0] <= result["range"][0] <= result["range"][1] <= result["range_with_uncertainty"][1]
    assert result["one_sided_level_each"] == pytest.approx(1 - 0.05 / 3)


def test_widened_upper_limit_covers_the_true_largest_rate_at_least_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    true_rates, sizes = np.array([0.06, 0.04, 0.08]), np.array([619, 60, 14])
    covered, trials = 0, 1500
    for _ in range(trials):
        counts = [(int(rng.binomial(n, r)), int(n)) for r, n in zip(true_rates, sizes)]
        covered += bounds.range_with_uncertainty(counts)["range_with_uncertainty"][1] >= min(1.0, true_rates.sum())
    assert covered / trials >= 0.95 - 0.01


def test_precision_rises_with_prevalence_and_falls_with_false_referrals():
    values = [bounds.precision(0.5, 0.1, prevalence) for prevalence in (0.001, 0.01, 0.1)]
    assert values == sorted(values)
    assert bounds.precision(0.5, 0.05, 0.01) > bounds.precision(0.5, 0.2, 0.01)
    assert bounds.precision(0.5, 0.0, 0.01) == pytest.approx(1.0)
