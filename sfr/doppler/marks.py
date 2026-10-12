"""Scoring detected cycle starts against the cardiologist's window marks.

What the marks are (Sulas et al., PLoS ONE 2021;16(4):e0248114, and the viewer shipped with the
data): the left edge of a 128-column window that holds one complete, measurable cardiac cycle.
The labelling tool also accepted the windows shifted by up to 15 columns either way as the same
cycle, and many marks sit exactly 128 columns after the previous one (tiled windows), so a mark
is not an exact event time. Matching therefore uses a tolerance of 15 columns by default.

Only complete and measurable cycles were marked. A detected cycle with no mark nearby may be a
real but unmarked cycle, so false detections are counted only where the marks are dense: between
two consecutive marks that are one cycle apart.
"""

from __future__ import annotations

import numpy as np

from ..ecg.metrics import match_pairs

TILE = 128


def tiled(marks: np.ndarray) -> np.ndarray:
    """True for a mark placed exactly one window after the previous mark."""
    flag = np.zeros(marks.size, dtype=bool)
    flag[1:] = np.diff(marks) == TILE
    return flag


def single_cycle_gaps(marks: np.ndarray, low: float = 0.75, high: float = 1.35) -> np.ndarray:
    """Index k of every gap (marks[k], marks[k+1]) that spans one cycle."""
    if marks.size < 3:
        return np.array([], dtype=int)
    gaps = np.diff(marks).astype(float)
    typical = float(np.median(gaps[(gaps > 80) & (gaps < 200)])) if np.any((gaps > 80) & (gaps < 200)) else float(TILE)
    return np.flatnonzero((gaps >= low * typical) & (gaps <= high * typical))


def score_starts(starts: np.ndarray, marks: np.ndarray, tolerance: int = 15, offset: float = 0.0) -> dict:
    """Recall over all marks; precision where the marks are dense."""
    starts = np.sort(np.asarray(starts, dtype=float) + offset)
    marks = np.sort(np.asarray(marks, dtype=float))
    pairs = match_pairs(starts, marks, tolerance)
    matched_starts = {i for i, _ in pairs}
    matched_marks = {j for _, j in pairs}
    residual = np.array([starts[i] - marks[j] for i, j in pairs])

    judged = np.zeros(starts.size, dtype=bool)
    for k in single_cycle_gaps(marks):
        judged |= (starts >= marks[k] - tolerance) & (starts <= marks[k + 1] + tolerance)
    judged[list(matched_starts)] = True
    false_pos = int(sum(1 for i in np.flatnonzero(judged) if i not in matched_starts))
    true_pos = len(pairs)
    false_neg = int(marks.size - len(matched_marks))
    recall = true_pos / marks.size if marks.size else float("nan")
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if true_pos else 0.0
    flag = tiled(marks.astype(int))
    hand = [j for j in range(marks.size) if not flag[j]]
    return {
        "n_marks": int(marks.size), "n_starts": int(starts.size), "n_judged_starts": int(judged.sum()),
        "tp": true_pos, "fp": false_pos, "fn": false_neg,
        "recall": recall, "precision": precision, "f1": f1,
        "n_hand_placed": len(hand),
        "recall_hand_placed": (sum(1 for j in hand if j in matched_marks) / len(hand)) if hand else float("nan"),
        "residual_median": float(np.median(residual)) if residual.size else float("nan"),
        "residual_iqr": float(np.subtract(*np.percentile(residual, [75, 25]))) if residual.size else float("nan"),
        "matched_mark_index": sorted(matched_marks),
    }


def rates_in_marked_spans(starts: np.ndarray, marks: np.ndarray, pps: float, offset: float = 0.0,
                          min_gaps: int = 8) -> tuple[float, float, int]:
    """Heart rate from the marks and from the detector over the same stretches of the strip.

    Marks: total span of the single-cycle gaps over their number. Tiled marks drift against the
    true beats, but over a run the drift stays inside the labelling tolerance, so the mean is
    usable. Detector: median of the detected intervals that lie in those gaps.
    Returns (marked bpm, detected bpm, number of gaps); NaN if there are fewer than ``min_gaps``.
    """
    marks = np.sort(np.asarray(marks, dtype=float))
    index = single_cycle_gaps(marks)
    if index.size < min_gaps:
        return float("nan"), float("nan"), int(index.size)
    marked = float(60.0 * pps * index.size / np.diff(marks)[index].sum())
    starts = np.sort(np.asarray(starts, dtype=float) + offset)
    if starts.size < 2:
        return marked, float("nan"), int(index.size)
    middle = 0.5 * (starts[:-1] + starts[1:])
    inside = np.zeros(middle.size, dtype=bool)
    for k in index:
        inside |= (middle >= marks[k]) & (middle <= marks[k + 1])
    if not inside.any():
        return marked, float("nan"), int(index.size)
    return marked, float(60.0 * pps / np.median(np.diff(starts)[inside])), int(index.size)
