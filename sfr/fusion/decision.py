"""The screening decision: REFER, REPEAT, GET_TEST or ROUTINE.

Each expert looks at one test and answers one question. The decision needs no model of how the
three tests behave together, so it can be stated, and its safety properties proved, without one
paired patient:

1. any expert that alarms sends the case to referral - no other result can cancel it;
2. a test of poor quality, or an ECG and a Doppler that do not show the same heart, are repeated;
3. a test that was not done is asked for only if the prior risk of its finding is above a limit;
4. "routine care" is given only when nothing alarms, nothing must be repeated and nothing that
   matters is missing. It is never given when no test could be read.

An alarm is a conformal p-value at or below the expert's level, so the share of healthy fetuses
referred by the whole system is at most the sum of the levels of the experts that were used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .opinion import Opinion, discount_reassurance, risk_interval

EXPERTS = ("structure", "function", "rhythm")
ACTIONS = ("REFER", "REPEAT", "GET_TEST", "ROUTINE")
PAIRED_BY_TRUST_CHECK = ("function", "rhythm")          # Doppler and fetal ECG


@dataclass(frozen=True)
class ExpertResult:
    present: bool = False                 # was the test done
    quality_ok: bool = True               # did the expert's own quality check pass
    p_value: float | None = None          # conformal p-value against healthy cases (small = unusual)
    probability: float | None = None      # probability of the finding at the screening prior, if available
    uncertainty: float = 0.2              # how much of that probability is held back as "not sure"

    @property
    def usable(self) -> bool:
        return bool(self.present and self.quality_ok and self.p_value is not None and np.isfinite(self.p_value))


@dataclass(frozen=True)
class DecisionConfig:
    """Alarm levels and priors. The priors of function and rhythm findings are assumptions."""
    alpha: Mapping[str, float] = field(default_factory=lambda: {"structure": 0.05, "function": 0.05, "rhythm": 0.10})
    prior: Mapping[str, float] = field(default_factory=lambda: {"structure": 0.009, "function": 0.002, "rhythm": 0.001})
    get_test_prior_limit: float = 0.005

    @property
    def false_referral_ceiling(self) -> float:
        return float(sum(self.alpha[name] for name in EXPERTS))


@dataclass
class Decision:
    action: str
    reasons: list[str]
    alarms: list[str]
    repeat: list[str]
    get: list[str]
    risk_low: float
    risk_high: float
    opinions: dict[str, Opinion]


def opinion_of(name: str, result: ExpertResult, config: DecisionConfig, trusted: bool) -> Opinion:
    """What the expert's result says about its finding, as an opinion.

    A test that is missing, unreadable, or that only gives a p-value (no calibrated probability)
    is the vacuous opinion: prior only, no reassurance. If the ECG-Doppler trust check failed,
    the reassuring part of those two opinions is removed and the alarm part kept.
    """
    prior = config.prior[name]
    if not result.usable or result.probability is None or not np.isfinite(result.probability):
        return Opinion.vacuous(prior)
    opinion = Opinion.from_probability(result.probability, result.uncertainty, prior)
    if not trusted and name in PAIRED_BY_TRUST_CHECK:
        opinion = discount_reassurance(opinion, 0.0)
    return opinion


def decide(results: Mapping[str, ExpertResult], trust_ok: bool | None = None,
           config: DecisionConfig = DecisionConfig()) -> Decision:
    """Apply the rule. ``trust_ok`` is the ECG-Doppler check: True, False, or None if not applicable."""
    results = {name: results.get(name, ExpertResult()) for name in EXPERTS}
    alarms = [name for name in EXPERTS if results[name].usable and results[name].p_value <= config.alpha[name]]
    repeat = [name for name in EXPERTS if results[name].present and not results[name].usable]
    both_paired = all(results[name].usable for name in PAIRED_BY_TRUST_CHECK)
    trusted = not (trust_ok is False and both_paired)
    if not trusted:
        repeat += [name for name in PAIRED_BY_TRUST_CHECK if name not in repeat]
    missing = [name for name in EXPERTS if not results[name].present]
    get = [name for name in missing if config.prior[name] > config.get_test_prior_limit]

    opinions = {name: opinion_of(name, results[name], config, trusted) for name in EXPERTS}
    risk_low, risk_high = risk_interval(opinions.values())
    any_usable = any(results[name].usable for name in EXPERTS)

    if alarms:
        action, reasons = "REFER", [f"{name} expert alarms (p = {results[name].p_value:.3f} <= {config.alpha[name]})" for name in alarms]
    elif repeat:
        action = "REPEAT"
        reasons = ["fetal ECG and Doppler do not show the same heart" if not trusted else ""]
        reasons += [f"{name} test could not be read" for name in EXPERTS if results[name].present and not results[name].usable]
        reasons = [r for r in reasons if r]
    elif get:
        action, reasons = "GET_TEST", [f"{name} test not done and its prior risk {config.prior[name]} is above {config.get_test_prior_limit}" for name in get]
    elif not any_usable:
        most_likely = max(EXPERTS, key=lambda name: config.prior[name])
        action, get, reasons = "GET_TEST", [most_likely], ["no test was done"]
    else:
        action, reasons = "ROUTINE", ["no alarm, nothing to repeat, no missing test above the prior limit"]
    return Decision(action, reasons, alarms, repeat, get, risk_low, risk_high, opinions)
