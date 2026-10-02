"""
SFR Framework — Subjective Logic Implementation
================================================
Based on Audun Jøsang (2016) "Subjective Logic: A Formalism for Reasoning Under Uncertainty".

In the Structure-Function-Rhythm (SFR) framework:
- Each expert provides an opinion: omega = (b, d, u, a)
  - b: belief of abnormality (CHD, arrhythmia, mechanical dysfunction)
  - d: disbelief of abnormality (evidence of health)
  - u: uncertainty (due to signal noise, low model confidence, or missing test)
  - a: base rate / clinical prior (prevalence ~ 0.009 or 9 per 1000)
- Constraint: b + d + u = 1.0, with b, d, u >= 0
- If an expert / modality is MISSING (e.g. Doppler not taken):
  Vacuous opinion: (b = 0.0, d = 0.0, u = 1.0)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Opinion:
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.009  # Saxena et al. ~9 per 1000 live births

    def __post_init__(self):
        # Clip to valid ranges to prevent floating point drift
        self.belief = max(0.0, min(1.0, float(self.belief)))
        self.disbelief = max(0.0, min(1.0, float(self.disbelief)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))
        
        total = self.belief + self.disbelief + self.uncertainty
        if total > 0:
            self.belief /= total
            self.disbelief /= total
            self.uncertainty /= total
        else:
            self.uncertainty = 1.0
            self.belief = 0.0
            self.disbelief = 0.0

    @property
    def expected_probability(self) -> float:
        """Projected probability: P(x) = b + a * u"""
        return self.belief + self.base_rate * self.uncertainty

    @classmethod
    def vacuous(cls, base_rate: float = 0.009) -> "Opinion":
        """Vacuous opinion representing complete absence of evidence / missing modality."""
        return cls(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=base_rate)

    @classmethod
    def from_evidence(cls, r: float, s: float, W: float = 2.0, base_rate: float = 0.009) -> "Opinion":
        """
        Derive opinion from Dirichlet positive (r) and negative (s) evidence counts.
        b = r / (r + s + W)
        d = s / (r + s + W)
        u = W / (r + s + W)
        """
        denom = r + s + W
        if denom <= 0:
            return cls.vacuous(base_rate)
        return cls(
            belief=r / denom,
            disbelief=s / denom,
            uncertainty=W / denom,
            base_rate=base_rate
        )

    @classmethod
    def from_probability_and_confidence(
        cls, prob: float, confidence: float, base_rate: float = 0.009
    ) -> "Opinion":
        """
        Derive opinion from model predicted probability p and epistemic/trust confidence c in [0, 1].
        Uncertainty u = 1 - c
        b = c * prob
        d = c * (1 - prob)
        """
        c = max(0.0, min(1.0, confidence))
        p = max(0.0, min(1.0, prob))
        b = c * p
        d = c * (1.0 - p)
        u = 1.0 - c
        return cls(belief=b, disbelief=d, uncertainty=u, base_rate=base_rate)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.belief, self.disbelief, self.uncertainty, self.base_rate)

    def __repr__(self) -> str:
        return f"Opinion(b={self.belief:.3f}, d={self.disbelief:.3f}, u={self.uncertainty:.3f}, E[P]={self.expected_probability:.3f})"


def subjective_logic_or(op_a: Opinion, op_b: Opinion) -> Opinion:
    """
    Cumulative Disjunctive (OR) operator for Subjective Logic.
    Used for screening: if EITHER Structure, Function, OR Rhythm is abnormal,
    the alarm is maintained.

    Formulas for independent binary frames:
    b_{A v B} = b_A + b_B - b_A * b_B
    d_{A v B} = d_A * d_B
    u_{A v B} = d_A * u_B + d_B * u_A + u_A * u_B
    """
    b1, d1, u1 = op_a.belief, op_a.disbelief, op_a.uncertainty
    b2, d2, u2 = op_b.belief, op_b.disbelief, op_b.uncertainty

    b_or = b1 + b2 - (b1 * b2)
    d_or = d1 * d2
    u_or = (d1 * u2) + (d2 * u1) + (u1 * u2)

    # Use combined or weighted base rate
    base_rate = (op_a.base_rate + op_b.base_rate) / 2.0
    return Opinion(belief=b_or, disbelief=d_or, uncertainty=u_or, base_rate=base_rate)


def fuse_opinions_or(opinions: List[Opinion]) -> Opinion:
    """
    Sequentially fuse N opinions using the Subjective Logic OR-operator.
    Vacuous opinions (b=0, d=0, u=1) act as the identity element for disbelief
    and preserve evidence from other modalities.
    """
    valid_opinions = [op for op in opinions if op is not None]
    if not valid_opinions:
        return Opinion.vacuous()

    result = valid_opinions[0]
    for op in valid_opinions[1:]:
        result = subjective_logic_or(result, op)
    return result
