"""Scoring of detected fetal beats against reference annotations."""

from __future__ import annotations

import numpy as np


def match_pairs(predicted: np.ndarray, reference: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """One-to-one matching of sorted event times within ``tolerance``.

    Returns (index into sorted predicted, index into sorted reference) for every match.
    """
    predicted = np.sort(np.asarray(predicted, dtype=float))
    reference = np.sort(np.asarray(reference, dtype=float))
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < predicted.size and j < reference.size:
        diff = predicted[i] - reference[j]
        if abs(diff) <= tolerance:
            pairs.append((i, j))
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return pairs


def match_beats(predicted: np.ndarray, reference: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """One-to-one matching of sorted event times within ``tolerance``. Returns TP, FP, FN."""
    tp = len(match_pairs(predicted, reference, tolerance))
    return tp, int(np.size(predicted)) - tp, int(np.size(reference)) - tp


def beat_scores(predicted: np.ndarray, reference: np.ndarray, fs: float, tolerance_ms: float = 50.0) -> dict:
    tp, fp, fn = match_beats(predicted, reference, tolerance_ms * fs / 1000.0)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sensitivity, "ppv": ppv, "f1": f1}


def heart_rate_bpm(peaks: np.ndarray, fs: float) -> float:
    """Median instantaneous rate. Robust to a few missed or extra beats."""
    if np.size(peaks) < 2:
        return float("nan")
    return float(60.0 * fs / np.median(np.diff(np.sort(peaks))))


def follows_mother(predicted_bpm: float, fetal_bpm: float, maternal_bpm: float) -> bool:
    """True when the detected rate is closer to the mother's rate than to the true fetal rate."""
    if not np.isfinite(predicted_bpm) or not np.isfinite(maternal_bpm):
        return False
    return abs(predicted_bpm - maternal_bpm) < abs(predicted_bpm - fetal_bpm)
