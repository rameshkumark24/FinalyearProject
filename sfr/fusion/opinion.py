"""Subjective-logic opinions and their OR.

An opinion about a statement ("this fetus has finding X") is four numbers: belief, disbelief,
uncertainty (they sum to one) and a base rate (the prior). Its projected probability is
belief + base rate x uncertainty.

The three experts judge three *different* statements (structure, function, rhythm), and the
question for screening is whether *any* of them is true. In Josang and McAnally (2004,
"Multiplication and comultiplication of beliefs", Int. J. Approx. Reasoning 38:19-51) that is
comultiplication (OR), not belief fusion: fusion rules such as Dempster's combine several
observers of the *same* statement and let one confident "normal" cancel another's alarm.

Facts used below (checked numerically in ``tests/test_fusion.py``):
* the projected probability of the OR is the noisy-OR of the projected probabilities;
* the belief of the OR is the noisy-OR of the beliefs, so a strong belief in one finding can
  never be reduced by the other opinions;
* a missing test is the vacuous opinion (no belief, no disbelief): it contributes its prior and
  no reassurance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

TOLERANCE = 1e-9


@dataclass(frozen=True)
class Opinion:
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float

    def __post_init__(self) -> None:
        for name in ("belief", "disbelief", "uncertainty", "base_rate"):
            value = getattr(self, name)
            if not -TOLERANCE <= value <= 1 + TOLERANCE:
                raise ValueError(f"{name} = {value} is outside [0, 1]")
        if abs(self.belief + self.disbelief + self.uncertainty - 1.0) > 1e-6:
            raise ValueError("belief, disbelief and uncertainty must sum to one")

    @property
    def projected(self) -> float:
        return self.belief + self.base_rate * self.uncertainty

    @staticmethod
    def vacuous(base_rate: float) -> "Opinion":
        """A test that was not done: nothing is known beyond the prior."""
        return Opinion(0.0, 0.0, 1.0, base_rate)

    @staticmethod
    def from_probability(probability: float, uncertainty: float, base_rate: float) -> "Opinion":
        """An expert's probability held with a given uncertainty; shrinks to the prior as it grows."""
        probability, uncertainty = float(np.clip(probability, 0, 1)), float(np.clip(uncertainty, 0, 1))
        return Opinion(probability * (1 - uncertainty), (1 - probability) * (1 - uncertainty), uncertainty, base_rate)


def disjunction(x: Opinion, y: Opinion) -> Opinion:
    """OR of two opinions about different statements (normal comultiplication, Josang & McAnally 2004)."""
    a = x.base_rate + y.base_rate - x.base_rate * y.base_rate
    b = x.belief + y.belief - x.belief * y.belief
    if a <= TOLERANCE:                                   # both statements impossible a priori
        return Opinion(b, x.disbelief * y.disbelief, 1.0 - b - x.disbelief * y.disbelief, 0.0)
    d = x.disbelief * y.disbelief + (x.base_rate * (1 - y.base_rate) * x.disbelief * y.uncertainty
                                     + (1 - x.base_rate) * y.base_rate * x.uncertainty * y.disbelief) / a
    u = x.uncertainty * y.uncertainty + (y.base_rate * x.disbelief * y.uncertainty
                                         + x.base_rate * x.uncertainty * y.disbelief) / a
    return Opinion(b, d, u, a)


def disjunction_all(opinions: Iterable[Opinion]) -> Opinion:
    """OR of any number of opinions. The result does not depend on their order.

    Belief, base rate and projected probability of the OR are each the noisy-OR of the parts;
    uncertainty and disbelief follow from them. For two opinions this is ``disjunction``.
    """
    opinions = list(opinions)
    if not opinions:
        raise ValueError("At least one opinion is needed")
    b = 1.0 - float(np.prod([1.0 - o.belief for o in opinions]))
    a = 1.0 - float(np.prod([1.0 - o.base_rate for o in opinions]))
    projected = 1.0 - float(np.prod([1.0 - o.projected for o in opinions]))
    u = (projected - b) / a if a > TOLERANCE else 1.0 - b - float(np.prod([o.disbelief for o in opinions]))
    u = float(np.clip(u, 0.0, 1.0 - b))
    return Opinion(b, 1.0 - b - u, u, a)


def discount_reassurance(opinion: Opinion, trust: float) -> Opinion:
    """Keep the alarm, weaken only the reassurance.

    ``trust`` in [0, 1] scales the disbelief; what is removed becomes uncertainty. Belief is
    untouched, so the projected probability can only stay or rise: a failed trust check never
    lowers an alarm.
    """
    trust = float(np.clip(trust, 0.0, 1.0))
    disbelief = trust * opinion.disbelief
    return Opinion(opinion.belief, disbelief, 1.0 - opinion.belief - disbelief, opinion.base_rate)


def risk_interval(opinions: Iterable[Opinion]) -> tuple[float, float]:
    """Bounds on the probability of "any finding" when the joint behaviour of the tests is unknown.

    Lower end: the largest single projected probability (findings always occur together).
    Upper end used here: the noisy-OR (findings unrelated). If findings tend to occur together,
    as heart block and structural disease do, the truth lies between the two.
    """
    projected = [o.projected for o in opinions]
    return float(max(projected)), float(1.0 - np.prod([1.0 - p for p in projected]))
