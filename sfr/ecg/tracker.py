"""Beat tracking by dynamic programming.

Given candidate peak times and their strengths, choose the sequence of beats that is both
strong and rhythmically smooth. Unlike threshold-and-search-back, the whole recording is
optimised at once, so a burst of noise cannot pull the rhythm away: inside the burst the
tracker keeps the beat positions that agree with the rhythm on either side.

This is a regularising detector. It is meant for heart-rate work on regular rhythms and must
not be used to judge whether a rhythm is irregular (the rhythm expert uses raw detections).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def candidate_peaks(envelope: np.ndarray, fs: float, level: float, min_gap_s: float = 0.05,
                    floor: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the detection envelope and their strengths scaled to 0..1."""
    peaks, _ = signal.find_peaks(envelope, distance=max(1, int(round(min_gap_s * fs))))
    if peaks.size == 0 or level <= 0:
        return np.array([], dtype=int), np.array([])
    strength = np.clip((envelope[peaks] / level - floor) / (1.0 - floor), 0.0, 1.0)
    return peaks, strength


def track_beats(
    times: np.ndarray, strength: np.ndarray, fs: float, min_interval_s: float = 0.27,
    max_interval_s: float = 1.20, smoothness: float = 4.0, beat_cost: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Return the indices of the selected candidates and the objective value per beat.

    State = (previous beat, current beat). The value of a path is the sum of
    (strength - beat_cost) over its beats minus ``smoothness`` times the absolute log-ratio
    of successive intervals.
    """
    n = times.size
    if n < 3:
        return np.arange(n), 0.0
    t = times.astype(np.float64) / fs
    node = strength - beat_cost
    lo = np.searchsorted(t, t - max_interval_s, side="left")
    hi = np.searchsorted(t, t - min_interval_s, side="right")

    values: list[np.ndarray] = [np.empty(0)] * n       # values[i][a]: best path ending (j=lo[i]+a, i)
    backs: list[np.ndarray] = [np.empty(0, dtype=int)] * n
    best_value, best_state = -np.inf, (-1, -1)
    for i in range(n):
        j_index = np.arange(lo[i], hi[i])
        if j_index.size == 0:
            continue
        interval_ji = t[i] - t[j_index]
        v = np.empty(j_index.size)
        b = np.empty(j_index.size, dtype=int)
        for a, j in enumerate(j_index):
            start = node[j]                              # path that begins at j
            previous = values[j]
            if previous.size:
                interval_kj = t[j] - t[lo[j]:hi[j]]
                option = previous - smoothness * np.abs(np.log(interval_ji[a] / interval_kj))
                k = int(np.argmax(option))
                if option[k] > start:
                    v[a], b[a] = option[k] + node[i], lo[j] + k
                    continue
            v[a], b[a] = start + node[i], -1
        values[i], backs[i] = v, b
        a_best = int(np.argmax(v))
        if v[a_best] > best_value:
            best_value, best_state = float(v[a_best]), (int(j_index[a_best]), i)

    if best_state[1] < 0:
        return np.array([], dtype=int), 0.0
    path = []
    j, i = best_state
    while True:
        path.append(i)
        k = int(backs[i][j - lo[i]])
        if k < 0:
            path.append(j)
            break
        j, i = k, j
    selected = np.array(path[::-1], dtype=int)
    return selected, best_value / max(1, selected.size)
