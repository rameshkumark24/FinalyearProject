"""Leave-one-subject-out evaluation for small cohorts.

Every score that is reported for a subject comes from a model that never saw that subject.
With 12 against 14 subjects the uncertainty is large, so each result carries a bootstrap
interval over subjects and a permutation p-value in which the model is refitted for every
shuffle of the labels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.stats import auroc, auroc_ci, clopper_pearson


def logistic(c: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=2000))


def forest(seed: int = 0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=2,
                                                random_state=seed, n_jobs=1))


def out_of_fold(x: np.ndarray, y: np.ndarray, model, groups: np.ndarray | None = None) -> np.ndarray:
    """Probability of class 1 for every row, from a model fitted without that row's group."""
    x, y = np.asarray(x, dtype=float), np.asarray(y).astype(int)
    groups = np.arange(y.size) if groups is None else np.asarray(groups)
    scores = np.full(y.size, np.nan)
    for group in np.unique(groups):
        held = groups == group
        if np.unique(y[~held]).size < 2:
            continue
        fitted = clone(model).fit(x[~held], y[~held])
        scores[held] = fitted.predict_proba(x[held])[:, 1]
    return scores


def permutation_p_refit(x: np.ndarray, y: np.ndarray, model, observed: float, n_perm: int = 1000,
                        seed: int = 0, groups: np.ndarray | None = None) -> float:
    """One-sided p-value for the out-of-fold AUROC: labels shuffled over subjects, model refitted."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        value = auroc(shuffled, out_of_fold(x, shuffled, model, groups))
        if np.isfinite(value) and value >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def summarise(y: np.ndarray, scores: np.ndarray, threshold: float | None = 0.5, groups: np.ndarray | None = None,
              seed: int = 0) -> dict:
    """AUROC with a subject-level bootstrap interval; sensitivity and specificity at ``threshold``."""
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    point, low, high = auroc_ci(y[keep], scores[keep], groups=None if groups is None else np.asarray(groups)[keep], seed=seed)
    out = {"n": int(keep.sum()), "n_positive": int(y[keep].sum()), "n_negative": int((1 - y[keep]).sum()),
           "auroc": point, "auroc_ci": [low, high]}
    positive, negative = scores[keep][y[keep] == 1], scores[keep][y[keep] == 0]
    if positive.size and negative.size:
        out["mann_whitney_p_one_sided"] = float(stats.mannwhitneyu(positive, negative, alternative="greater").pvalue)
    if threshold is not None:
        predicted = scores[keep] >= threshold
        tp, fn = int((predicted & (y[keep] == 1)).sum()), int((~predicted & (y[keep] == 1)).sum())
        tn, fp = int((~predicted & (y[keep] == 0)).sum()), int((predicted & (y[keep] == 0)).sum())
        out.update({"threshold": threshold, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                    "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                    "sensitivity_ci": list(clopper_pearson(tp, tp + fn)),
                    "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                    "specificity_ci": list(clopper_pearson(tn, tn + fp))})
    return out


def evaluate(x: np.ndarray, y: np.ndarray, model, n_perm: int = 1000, seed: int = 0) -> tuple[dict, np.ndarray]:
    scores = out_of_fold(x, y, model)
    result = summarise(y, scores, seed=seed)
    result["permutation_p"] = permutation_p_refit(x, y, model, result["auroc"], n_perm=n_perm, seed=seed)
    result["n_permutations"] = n_perm
    return result, scores


def single_index(y: np.ndarray, values: np.ndarray, n_perm: int = 10000, seed: int = 0) -> dict:
    """AUROC of one pre-specified index used as it is: nothing is fitted, so no folds are needed."""
    y, values = np.asarray(y).astype(int), np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    result = summarise(y[keep], values[keep], threshold=None, seed=seed)
    rng = np.random.default_rng(seed)
    count = sum(auroc(rng.permutation(y[keep]), values[keep]) >= result["auroc"] for _ in range(n_perm))
    result["permutation_p"] = (count + 1) / (n_perm + 1)
    result["n_permutations"] = n_perm
    result["n_missing"] = int((~keep).sum())
    return result


FitFunction = Callable[[np.ndarray, np.ndarray], object]
