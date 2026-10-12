"""Which side of the zero line shows ejection?

The display orientation of a five-chamber strip is chosen at the scanner and is not stored with
the NInFEA images. Two ways to recover it:

* ``from_ecg``: with a simultaneous fetal ECG, ejection is the flow that follows the R peak.
* ``from_doppler``: without an ECG, the ejection wave is usually the larger of the two flows,
  and the main filling peak (A wave) comes shortly before it.
"""

from __future__ import annotations

import numpy as np

from .cycles import CycleParams, Prepared, track_waves
from .envelope import PIXELS_PER_SECOND

EJECTION_WINDOW_S = (0.05, 0.20)     # after the R peak; fixed from physiology, not tuned


def from_ecg(prepared: Prepared, r_columns: np.ndarray, pps: float = PIXELS_PER_SECOND,
             window_s: tuple[float, float] = EJECTION_WINDOW_S) -> dict:
    """Ejection side = the side whose flow is concentrated in the window after the R peaks."""
    r_columns = np.asarray(r_columns, dtype=float)
    lo, hi = int(round(window_s[0] * pps)), int(round(window_s[1] * pps))
    width = prepared.upper_flow.size
    nothing = {"side": None, "contrast_upper": float("nan"), "contrast_lower": float("nan"), "margin": float("nan")}
    if r_columns.size < 3:
        return nothing
    inside = np.zeros(width, dtype=bool)
    for r in r_columns:
        a, b = int(round(r)) + lo, int(round(r)) + hi
        if a >= 0 and b < width:
            inside[a:b] = True
    covered = np.zeros(width, dtype=bool)
    covered[int(max(0, r_columns.min())):int(min(width, r_columns.max() + hi))] = True
    contrast = {}
    for name, flow in (("upper", prepared.upper_flow), ("lower", prepared.lower_flow)):
        total = flow[covered].mean() if covered.any() else 0.0
        contrast[name] = float(flow[inside].mean() / total) if inside.any() and total > 0 else float("nan")
    if not np.isfinite(contrast["upper"]) or not np.isfinite(contrast["lower"]) or min(contrast.values()) <= 0:
        return {**nothing, "contrast_upper": contrast["upper"], "contrast_lower": contrast["lower"]}
    side = "upper" if contrast["upper"] > contrast["lower"] else "lower"
    return {"side": side, "contrast_upper": contrast["upper"], "contrast_lower": contrast["lower"],
            "margin": float(abs(np.log(contrast["upper"] / contrast["lower"])))}


def _profile(x: np.ndarray, reference: np.ndarray, bins: int = 64) -> np.ndarray:
    intervals = np.diff(reference)
    if intervals.size == 0:
        return np.zeros(bins)
    typical = np.median(intervals)
    rows = [np.interp(a + np.arange(bins) / bins * (b - a), np.arange(x.size), x)
            for a, b in zip(reference[:-1], reference[1:]) if 0.8 * typical <= b - a <= 1.2 * typical]
    return np.median(rows, axis=0) if rows else np.zeros(bins)


def from_doppler(prepared: Prepared, pps: float = PIXELS_PER_SECOND, params: CycleParams = CycleParams(),
                 decisive_log_ratio: float = 0.30, coincident: float = 0.10) -> dict:
    """Ejection side from the Doppler alone: the larger flow, with wave order as the tie-break.

    ``area`` is the mean floor-corrected flow of a side. If the two areas differ clearly
    (|log ratio| above ``decisive_log_ratio``) the larger side is ejection. Otherwise the order of
    the main peaks decides: the main filling peak precedes the ejection peak by less than half a
    cycle. When the two main peaks coincide (a mirrored spectrum) only the areas can be used.
    """
    area_upper, area_lower = float(prepared.upper_flow.mean()), float(prepared.lower_flow.mean())
    if area_upper <= 0 or area_lower <= 0:
        return {"side": None, "log_area_ratio": float("nan"), "lag": float("nan"), "rule": "no flow"}
    log_ratio = float(np.log(area_lower / area_upper))            # > 0: lower side is larger

    tracks = {name: track_waves(flow, pps, prepared.duration_s, params.track_sigma, params.smoothness)
              for name, flow in (("upper", prepared.upper_flow), ("lower", prepared.lower_flow))}
    reference_side = max(tracks, key=lambda name: tracks[name][1])
    reference = tracks[reference_side][0]
    upper_profile = _profile(prepared.upper_flow, reference)
    lower_profile = _profile(prepared.lower_flow, reference)
    bins = upper_profile.size
    lag = ((int(np.argmax(lower_profile)) - int(np.argmax(upper_profile))) % bins) / bins   # upper peak -> lower peak

    mirrored = lag < coincident or lag > 1.0 - coincident
    if abs(log_ratio) >= decisive_log_ratio or mirrored:
        side = "lower" if log_ratio > 0 else "upper"
        rule = "area (peaks coincide)" if mirrored and abs(log_ratio) < decisive_log_ratio else "area"
    else:
        side, rule = ("lower" if lag < 0.5 else "upper"), "wave order"
    return {"side": side, "log_area_ratio": log_ratio, "lag": float(lag), "rule": rule}
