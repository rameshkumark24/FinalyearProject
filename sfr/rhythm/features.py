"""Rhythm features from fetal beat times.

Computed per one-minute segment from the raw, threshold-only detections (the tracker smooths an
irregular rhythm, so its output is used only for the heart rate), then summarised per recording.
The set of features and the direction of each (higher = more irregular) were fixed before any
of them was compared with the arrhythmia labels.
"""

from __future__ import annotations

import numpy as np

SEGMENT_FEATURES = (
    "hr_bpm",             # median rate from the tracked beats
    "rr_mad_rel",         # spread of the intervals (median absolute deviation over the median)
    "succ_rel",           # typical change from one interval to the next, over the median
    "outlier_share",      # share of intervals more than 15% away from the running median
    "short_long_share",   # share of "early beat then pause" pairs
    "long_share",         # share of intervals longer than 1.7 running medians (dropped beats)
    "tracked_raw_gap",    # how much the regularising tracker changed the raw detections
)
IRREGULARITY_FEATURES = SEGMENT_FEATURES[1:]          # higher = more irregular (hr_bpm has no direction)
RECORD_FEATURES = tuple(f"{name}_median" for name in SEGMENT_FEATURES) + ("outlier_share_p90", "short_long_share_p90")
QUALITY_FEATURES = ("usable_share", "score_median", "template_corr_median")
MIN_BEATS = 20


def running_median(x: np.ndarray, window: int = 9) -> np.ndarray:
    half = window // 2
    return np.array([np.median(x[max(0, i - half): i + half + 1]) for i in range(x.size)])


def segment_features(raw_s: np.ndarray, tracked_s: np.ndarray) -> dict[str, float]:
    """Features of one segment. NaN if there are too few beats to describe a rhythm."""
    raw_s, tracked_s = np.sort(np.asarray(raw_s, dtype=float)), np.sort(np.asarray(tracked_s, dtype=float))
    out = {name: float("nan") for name in SEGMENT_FEATURES}
    if raw_s.size < MIN_BEATS or tracked_s.size < MIN_BEATS:
        return out
    rr = np.diff(raw_s)
    typical = float(np.median(rr))
    local = running_median(rr)
    out["hr_bpm"] = float(60.0 / np.median(np.diff(tracked_s)))
    out["rr_mad_rel"] = float(np.median(np.abs(rr - typical)) / typical)
    out["succ_rel"] = float(np.median(np.abs(np.diff(rr))) / typical)
    out["outlier_share"] = float(np.mean(np.abs(rr - local) > 0.15 * local))
    out["short_long_share"] = float(np.mean((rr[:-1] < 0.85 * local[:-1]) & (rr[1:] > 1.10 * local[1:])))
    out["long_share"] = float(np.mean(rr > 1.7 * local))
    out["tracked_raw_gap"] = float(abs(raw_s.size - tracked_s.size) / tracked_s.size)
    return out


def record_features(segments: list[dict[str, float]]) -> dict[str, float]:
    """Summary over the segments of one recording: medians, and the 90th percentile of the two
    features that an occasional arrhythmia would raise only in some segments."""
    out = {name: float("nan") for name in RECORD_FEATURES}
    for name in SEGMENT_FEATURES:
        values = np.array([s[name] for s in segments], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[f"{name}_median"] = float(np.median(values))
            if name in ("outlier_share", "short_long_share"):
                out[f"{name}_p90"] = float(np.percentile(values, 90))
    return out
