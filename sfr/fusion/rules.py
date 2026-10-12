"""Rules that turn the experts' alarm strengths into one score, for comparison.

Every rule takes, per case, the alarm strength of each expert in [0, 1] (NaN = test not done)
and returns one score; a case is referred when the score reaches a threshold. The project's own
rule is ``maximum`` with a threshold per expert, i.e. "refer if any expert alarms".

``maximum`` and ``noisy_or`` cannot be lowered by adding a normal result. ``mean``,
``product_of_experts`` and ``dempster`` can: that is the cancellation the paper is about.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-6


def _clean(strengths: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(strengths, dtype=float))


def maximum(strengths: np.ndarray) -> np.ndarray:
    s = _clean(strengths)
    return np.where(np.isnan(s).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(s), -np.inf, s), axis=1))


def noisy_or(strengths: np.ndarray) -> np.ndarray:
    s = _clean(strengths)
    out = 1.0 - np.prod(np.where(np.isnan(s), 1.0, 1.0 - s), axis=1)
    return np.where(np.isnan(s).all(axis=1), np.nan, out)


def mean(strengths: np.ndarray) -> np.ndarray:
    s = _clean(strengths)
    count = np.sum(~np.isnan(s), axis=1)
    total = np.nansum(s, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def product_of_experts(strengths: np.ndarray) -> np.ndarray:
    """Normalised product of the experts' probabilities (each treated as P(abnormal))."""
    s = np.clip(_clean(strengths), EPSILON, 1 - EPSILON)
    yes = np.prod(np.where(np.isnan(s), 1.0, s), axis=1)
    no = np.prod(np.where(np.isnan(s), 1.0, 1.0 - s), axis=1)
    return np.where(np.isnan(_clean(strengths)).all(axis=1), np.nan, yes / (yes + no))


def dempster(strengths: np.ndarray, ignorance: float = 0.1) -> np.ndarray:
    """Dempster's rule on the frame {abnormal, normal}; returns the belief in "abnormal".

    Each expert gives mass s(1-i) to abnormal, (1-s)(1-i) to normal and i to "either".
    A missing test gives all its mass to "either".
    """
    s = _clean(strengths)
    out = np.empty(s.shape[0])
    for row in range(s.shape[0]):
        abnormal, normal, either = 0.0, 0.0, 1.0
        for value in s[row]:
            if np.isnan(value):
                continue
            a2, n2, e2 = value * (1 - ignorance), (1 - value) * (1 - ignorance), ignorance
            conflict = abnormal * n2 + normal * a2
            scale = 1.0 - conflict
            if scale <= EPSILON:
                abnormal, normal, either = 0.5, 0.5, 0.0
                continue
            abnormal, normal, either = ((abnormal * a2 + abnormal * e2 + either * a2) / scale,
                                        (normal * n2 + normal * e2 + either * n2) / scale,
                                        either * e2 / scale)
        out[row] = np.nan if np.isnan(s[row]).all() else abnormal
    return out


RULES = {"maximum": maximum, "noisy_or": noisy_or, "mean": mean,
         "product_of_experts": product_of_experts, "dempster": dempster}
CANCELLING = ("mean", "product_of_experts", "dempster")
