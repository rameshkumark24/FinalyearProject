"""Properties of the opinions, the comparison rules and the decision.

These are the claims the paper makes about the fusion step. They hold by construction; the tests
are there so that a later change cannot break them silently. Random numbers are used only to
draw test cases.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sfr.fusion import rules
from sfr.fusion.conformal import leave_one_out_p_values, p_value, realised_false_alarm_range, smallest_p
from sfr.fusion.decision import EXPERTS, DecisionConfig, ExpertResult, decide
from sfr.fusion.opinion import Opinion, discount_reassurance, disjunction, disjunction_all, risk_interval

RNG = np.random.default_rng(20261011)


def random_opinion() -> Opinion:
    b, d, u = RNG.dirichlet([1, 1, 1])
    return Opinion(float(b), float(d), float(u), float(RNG.uniform(0.001, 0.999)))


def random_result(alarm: bool | None = None, config: DecisionConfig = DecisionConfig(), name: str = "rhythm") -> ExpertResult:
    if RNG.random() < 0.25:
        return ExpertResult(present=False)
    p = float(RNG.uniform(0, 1))
    if alarm is True:
        p = float(RNG.uniform(0, config.alpha[name]))
    if alarm is False:
        p = float(RNG.uniform(config.alpha[name] + 1e-6, 1))
    return ExpertResult(present=True, quality_ok=bool(RNG.random() < 0.85), p_value=p,
                        probability=float(RNG.uniform(0, 1)), uncertainty=float(RNG.uniform(0, 0.9)))


# ---- opinions -------------------------------------------------------------------------------------

def test_projected_probability_of_or_is_noisy_or():
    for _ in range(2000):
        x, y = random_opinion(), random_opinion()
        z = disjunction(x, y)
        assert z.projected == pytest.approx(x.projected + y.projected - x.projected * y.projected, abs=1e-9)
        assert z.belief + z.disbelief + z.uncertainty == pytest.approx(1.0, abs=1e-9)
        assert min(z.belief, z.disbelief, z.uncertainty) >= -1e-9


def test_belief_in_or_is_never_below_either_belief():
    for _ in range(2000):
        x, y = random_opinion(), random_opinion()
        z = disjunction(x, y)
        assert z.belief >= max(x.belief, y.belief) - 1e-12


def test_or_of_many_matches_pairwise_and_ignores_order():
    for _ in range(500):
        x, y, w = random_opinion(), random_opinion(), random_opinion()
        pair = disjunction(x, y)
        both = disjunction_all([x, y])
        for name in ("belief", "disbelief", "uncertainty", "base_rate"):
            assert getattr(both, name) == pytest.approx(getattr(pair, name), abs=1e-9)
        reference = disjunction_all([x, y, w])
        for order in itertools.permutations([x, y, w]):
            other = disjunction_all(order)
            for name in ("belief", "disbelief", "uncertainty", "base_rate"):
                assert getattr(other, name) == pytest.approx(getattr(reference, name), abs=1e-9)
        assert min(reference.disbelief, reference.uncertainty) >= -1e-9


def test_missing_test_adds_its_prior_and_no_reassurance():
    for _ in range(500):
        x = random_opinion()
        prior = float(RNG.uniform(0.001, 0.2))
        missing = Opinion.vacuous(prior)
        assert missing.disbelief == 0.0 and missing.projected == pytest.approx(prior)
        assert disjunction(x, missing).projected >= x.projected - 1e-12
        assert disjunction(x, missing).projected == pytest.approx(x.projected + prior - x.projected * prior, abs=1e-9)


def test_lower_trust_never_lowers_the_alarm():
    for _ in range(2000):
        x = random_opinion()
        high, low = sorted(RNG.uniform(0, 1, 2), reverse=True)
        more, less = discount_reassurance(x, float(high)), discount_reassurance(x, float(low))
        assert more.belief == x.belief == less.belief
        assert less.projected >= more.projected - 1e-12
        assert more.projected >= x.projected - 1e-12


def test_risk_interval_brackets_every_single_risk():
    for _ in range(500):
        opinions = [random_opinion() for _ in range(3)]
        low, high = risk_interval(opinions)
        assert low == pytest.approx(max(o.projected for o in opinions))
        assert low <= high <= min(1.0, sum(o.projected for o in opinions)) + 1e-12


# ---- comparison rules -----------------------------------------------------------------------------

def test_or_type_rules_cannot_be_lowered_by_a_normal_result():
    for _ in range(500):
        alarm = float(RNG.uniform(0.5, 1.0))
        normal = float(RNG.uniform(0.0, 0.2))
        alone = np.array([[alarm, np.nan, np.nan]])
        with_normals = np.array([[alarm, normal, normal]])
        for name in ("maximum", "noisy_or"):
            assert rules.RULES[name](with_normals)[0] >= rules.RULES[name](alone)[0] - 1e-12


def test_averaging_type_rules_are_lowered_by_normal_results():
    alone = np.array([[0.9, np.nan, np.nan]])
    with_normals = np.array([[0.9, 0.05, 0.05]])
    for name in rules.CANCELLING:
        assert rules.RULES[name](with_normals)[0] < rules.RULES[name](alone)[0]


def test_rules_return_nan_when_no_test_was_done():
    nothing = np.array([[np.nan, np.nan, np.nan]])
    for rule in rules.RULES.values():
        assert np.isnan(rule(nothing)[0])


# ---- conformal p-values ---------------------------------------------------------------------------

def test_p_value_is_a_rank():
    healthy = np.array([0.1, 0.2, 0.3, 0.4])
    assert p_value(0.5, healthy) == pytest.approx(1 / 5)
    assert p_value(0.25, healthy) == pytest.approx(3 / 5)
    assert p_value(0.0, healthy) == pytest.approx(1.0)
    assert smallest_p(14) == pytest.approx(1 / 15)


def test_healthy_cases_are_flagged_no_more_often_than_the_level():
    alpha, n, flagged, total = 0.10, 39, 0, 0
    for _ in range(400):
        p = leave_one_out_p_values(RNG.normal(size=n))
        flagged += int((p <= alpha).sum())
        total += n
    assert flagged / total <= alpha + 0.01


def test_alarm_level_below_the_smallest_p_can_never_fire():
    assert realised_false_alarm_range(14, 0.05) == (0.0, 0.0)
    low, high = realised_false_alarm_range(39, 0.05)
    assert 0 < low < 0.05 < high < 0.2


# ---- decision ---------------------------------------------------------------------------------------

def test_one_alarm_always_refers_whatever_the_others_say():
    config = DecisionConfig()
    for _ in range(3000):
        alarming = str(RNG.choice(EXPERTS))
        results = {name: random_result(None, config, name) for name in EXPERTS}
        results[alarming] = ExpertResult(present=True, quality_ok=True, p_value=float(RNG.uniform(0, config.alpha[alarming])),
                                         probability=float(RNG.uniform(0, 1)))
        for trust in (True, False, None):
            decision = decide(results, trust, config)
            assert decision.action == "REFER" and alarming in decision.alarms


def test_no_test_done_is_never_routine():
    decision = decide({}, None)
    assert decision.action == "GET_TEST" and decision.get
    for trust in (True, False):
        assert decide({name: ExpertResult(present=False) for name in EXPERTS}, trust).action != "ROUTINE"


def test_failed_trust_check_is_never_routine_and_never_removes_an_alarm():
    config = DecisionConfig()
    for _ in range(3000):
        results = {name: random_result(None, config, name) for name in EXPERTS}
        with_trust, without_trust = decide(results, True, config), decide(results, False, config)
        if with_trust.action == "REFER":
            assert without_trust.action == "REFER" and set(with_trust.alarms) == set(without_trust.alarms)
        if results["function"].usable and results["rhythm"].usable:
            assert without_trust.action in ("REFER", "REPEAT")
        assert without_trust.risk_low >= with_trust.risk_low - 1e-12
        assert without_trust.risk_high >= with_trust.risk_high - 1e-12


def test_unreadable_test_is_repeated_not_trusted():
    results = {"rhythm": ExpertResult(present=True, quality_ok=False, p_value=0.9, probability=0.01),
               "function": ExpertResult(present=True, p_value=0.8), "structure": ExpertResult(present=True, p_value=0.7, probability=0.01)}
    decision = decide(results, True)
    assert decision.action == "REPEAT" and decision.repeat == ["rhythm"]
    assert decision.opinions["rhythm"].disbelief == 0.0


def test_missing_echo_is_asked_for_but_missing_fetal_ecg_is_not():
    config = DecisionConfig()
    normal = ExpertResult(present=True, p_value=0.9, probability=0.001)
    no_echo = decide({"function": normal, "rhythm": normal}, True, config)
    assert no_echo.action == "GET_TEST" and no_echo.get == ["structure"]
    no_ecg = decide({"structure": normal, "function": normal}, None, config)
    assert no_ecg.action == "ROUTINE"
    assert no_ecg.opinions["rhythm"].projected == pytest.approx(config.prior["rhythm"])


def test_order_of_experts_does_not_matter():
    config = DecisionConfig()
    for _ in range(500):
        results = {name: random_result(None, config, name) for name in EXPERTS}
        reference = decide(results, True, config)
        for order in itertools.permutations(EXPERTS):
            other = decide({name: results[name] for name in order}, True, config)
            assert other.action == reference.action
            assert other.risk_low == pytest.approx(reference.risk_low) and other.risk_high == pytest.approx(reference.risk_high)


def test_false_referral_ceiling_is_the_sum_of_levels():
    config = DecisionConfig()
    assert config.false_referral_ceiling == pytest.approx(sum(config.alpha.values()))
    n, referred = 20000, 0
    for _ in range(n):
        results = {name: ExpertResult(present=True, p_value=float(RNG.uniform(0, 1)), probability=0.01) for name in EXPERTS}
        referred += decide(results, True, config).action == "REFER"
    assert referred / n <= config.false_referral_ceiling + 0.01
