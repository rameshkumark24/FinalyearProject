"""Small-sample statistics used across the project.

Intervals are always computed over the independent unit (recording, woman or patient),
never over windows or beats taken from the same unit.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(values: Sequence[float], statistic: Callable = np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval for a statistic of independent values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(n_boot, values.size))
    draws = np.array([statistic(values[row]) for row in index])
    return float(statistic(values)), float(np.percentile(draws, 100 * alpha / 2)), \
        float(np.percentile(draws, 100 * (1 - alpha / 2)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve by the rank (Mann-Whitney) formula. NaN if one class is absent."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([positives, negatives]))
    return float((ranks[: positives.size].sum() - positives.size * (positives.size + 1) / 2)
                 / (positives.size * negatives.size))


def auroc_ci(labels: Sequence[int], scores: Sequence[float], groups: Sequence | None = None,
             n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """AUROC with a bootstrap interval. If ``groups`` is given, whole groups are resampled."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    point = auroc(labels, scores)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(labels.size)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([members[g] for g in chosen])
        value = auroc(labels[index], scores[index])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))


def permutation_p_auroc(labels: Sequence[int], scores: Sequence[float], n_perm: int = 10000,
                        seed: int = 0) -> float:
    """One-sided p-value for AUROC above chance, by permuting the labels."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    observed = auroc(labels, scores)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        if auroc(rng.permutation(labels), scores) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def bland_altman(a: Sequence[float], b: Sequence[float]) -> dict:
    """Bias and 95% limits of agreement between two measurements of the same quantity."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return {"n": int(diff.size), "bias": float("nan"), "sd": float("nan"),
                "loa_low": float("nan"), "loa_high": float("nan")}
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    return {"n": int(diff.size), "bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}
