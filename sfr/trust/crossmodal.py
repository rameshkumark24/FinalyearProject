"""ECG-Doppler trust check: do the two recordings show the same heart, beat for beat?

The fetal ECG gives the time of every electrical beat (R peak). The Doppler gives the time of
every mechanical beat (aortic valve closure). If both come from the same heart at the same
time, every closure-to-closure interval equals the R-R interval that surrounds it, to within a
few milliseconds. If the "fetal" ECG is really the mother's, or belongs to another fetus, or
either detector has lost the signal, the intervals stop agreeing.

Three numbers are computed for a pair of beat series:

* ``locked_share``  - share of Doppler intervals within ``tau_ms`` of the matching R-R interval.
* ``drift_ms``      - median difference (Doppler interval minus R-R interval). Two regular
  rhythms of slightly different rate agree on most single beats but drift steadily.
* ``delay_ms``      - median time from an R peak to the next valve closure. For the same heart
  this is the length of electromechanical systole and lies in a narrow range.

Interval differences are used rather than a global phase measure because the NInFEA strips
were assembled from video frames and contain occasional jumps of the time base, which change
the delay by a step but leave all other intervals intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrustRule:
    """Thresholds of the check. Set on development women only."""
    tau_ms: float = 10.0                 # about three Doppler columns
    min_locked_share: float = 0.5        # at least half of the beats agree
    max_abs_drift_ms: float = 2.5        # about 0.8 bpm at 140 bpm
    delay_range_ms: tuple[float, float] = (130.0, 275.0)   # development range 157-248 ms, with a margin
    min_pairs: int = 8


def paired_intervals(r_s: np.ndarray, c_s: np.ndarray) -> dict[str, np.ndarray]:
    """For every Doppler interval, the R-R interval that contains its midpoint.

    ``r_s`` and ``c_s`` are beat times in seconds on the same clock. Returns the two interval
    series (seconds), the midpoint times, and the R-to-closure delay for every closure that has
    an R peak before it.
    """
    r_s, c_s = np.sort(np.asarray(r_s, dtype=float)), np.sort(np.asarray(c_s, dtype=float))
    empty = np.array([])
    if r_s.size < 3 or c_s.size < 3:
        return {"cc": empty, "rr": empty, "t": empty, "delay": empty}
    middle = 0.5 * (c_s[:-1] + c_s[1:])
    index = np.searchsorted(r_s, middle, side="right") - 1
    inside = (index >= 0) & (index < r_s.size - 1)
    cc = np.diff(c_s)[inside]
    rr = (r_s[np.clip(index + 1, 0, r_s.size - 1)] - r_s[np.clip(index, 0, r_s.size - 1)])[inside]
    before = np.searchsorted(r_s, c_s, side="right") - 1
    has = before >= 0
    delay = c_s[has] - r_s[before[has]]
    return {"cc": cc, "rr": rr, "t": middle[inside], "delay": delay}


def trust_statistics(r_s: np.ndarray, c_s: np.ndarray, tau_ms: float = 10.0) -> dict[str, float]:
    pairs = paired_intervals(r_s, c_s)
    n = int(pairs["cc"].size)
    out = {"n_pairs": n, "locked_share": float("nan"), "drift_ms": float("nan"), "abs_diff_median_ms": float("nan"),
           "delay_ms": float("nan"), "delay_iqr_ms": float("nan"), "rate_ecg_bpm": float("nan"),
           "rate_doppler_bpm": float("nan"), "coverage": 0.0}
    if np.size(r_s) > 2:
        out["rate_ecg_bpm"] = float(60.0 / np.median(np.diff(np.sort(r_s))))
    if np.size(c_s) > 2:
        out["rate_doppler_bpm"] = float(60.0 / np.median(np.diff(np.sort(c_s))))
        out["coverage"] = n / (np.size(c_s) - 1)
    if n == 0:
        return out
    diff_ms = 1000.0 * (pairs["cc"] - pairs["rr"])
    out["locked_share"] = float(np.mean(np.abs(diff_ms) <= tau_ms))
    out["drift_ms"] = float(np.median(diff_ms))
    out["abs_diff_median_ms"] = float(np.median(np.abs(diff_ms)))
    if pairs["delay"].size:
        out["delay_ms"] = float(1000.0 * np.median(pairs["delay"]))
        out["delay_iqr_ms"] = float(1000.0 * np.subtract(*np.percentile(pairs["delay"], [75, 25])))
    return out


def verdict(stats: dict[str, float], rule: TrustRule = TrustRule()) -> dict:
    """PASS only if the beats are locked, do not drift and the delay is physiological."""
    reasons = []
    if stats["n_pairs"] < rule.min_pairs or not np.isfinite(stats["locked_share"]):
        return {"trusted": False, "reasons": ["too few paired beats"]}
    if stats["locked_share"] < rule.min_locked_share:
        reasons.append("beats not locked")
    if abs(stats["drift_ms"]) > rule.max_abs_drift_ms:
        reasons.append("rates drift apart")
    if not rule.delay_range_ms[0] <= stats["delay_ms"] <= rule.delay_range_ms[1]:
        reasons.append("delay outside the physiological range")
    return {"trusted": not reasons, "reasons": reasons}


def plausible_intervals(intervals: np.ndarray, window: int = 9, tolerance: float = 0.20) -> np.ndarray:
    """True where an interval is within ``tolerance`` of the running median of its neighbours."""
    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        return np.zeros(0, dtype=bool)
    half = window // 2
    # At the two ends the window is cut short rather than padded: padding with the end value
    # would make a wrong first or last interval look typical.
    local = np.array([np.median(intervals[max(0, i - half): i + half + 1]) for i in range(intervals.size)])
    return np.abs(intervals - local) <= tolerance * local
