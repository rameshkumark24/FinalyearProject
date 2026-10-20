"""Group-conditional ("Mondrian") conformal p-values.

A plain conformal p-value ranks a new score among all healthy calibration scores, so the false
alarm rate is at the level on average over the groups that make up the calibration set. If the
score depends on the group (for example on the image size of the scanner), one group can get far
more false alarms than another. Ranking each case only among the healthy calibration cases of its
own group keeps the false-alarm rate at the level inside every group, provided the cases are
exchangeable within the group (Mondrian conformal prediction; Vovk, "Conditional validity of
inductive conformal predictors", Machine Learning 92, 2013).

The price is sample size: with n healthy calibration cases a group's smallest p-value is 1/(n+1),
so it can alarm at level alpha only if n >= ceil(1/alpha) - 1.
"""

from __future__ import annotations

import numpy as np


def minimum_group_size(alpha: float) -> int:
    """Fewest healthy calibration cases with which a group can alarm at ``alpha`` (smallest p = 1/(n+1) <= alpha)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    return int(np.ceil(1 / alpha - 1 - 1e-12))


def group_conditional_p_values(scores, groups, calibration_scores, calibration_groups) -> np.ndarray:
    """p-value of each score against the healthy calibration scores of its own group (NaN if the group has none)."""
    scores, groups = np.asarray(scores, dtype=float), np.asarray(groups)
    calibration_scores, calibration_groups = np.asarray(calibration_scores, dtype=float), np.asarray(calibration_groups)
    out = np.full(scores.size, np.nan)
    for group in np.unique(groups):
        member = groups == group
        reference = np.sort(calibration_scores[(calibration_groups == group) & np.isfinite(calibration_scores)])
        if reference.size == 0:
            continue
        at_least = reference.size - np.searchsorted(reference, scores[member], side="left")      # same count as p_value
        out[member] = (1 + at_least) / (reference.size + 1)
    out[~np.isfinite(scores)] = np.nan
    return out
