"""Cardiac cycles from Doppler velocity envelopes.

A five-chamber pulsed-wave Doppler strip shows ventricular filling (E wave, then A wave) on one
side of the zero line and aortic ejection (V wave) on the other. Which side is which is a
display choice made at the scanner, so it is an input here (``ejection_side``); see
``orientation.py`` for ways to infer it when it was not recorded.

One cycle runs from one aortic valve closure to the next. The closure is found in two steps:
the ejection wave of every beat is tracked, and the valve click next to the end of that wave
(a narrow line that reaches both sides of the zero line) gives the exact column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from ..ecg.tracker import candidate_peaks, track_beats
from .envelope import PIXELS_PER_SECOND, Envelopes


@dataclass(frozen=True)
class CycleParams:
    click_width: int = 13            # columns; a valve click is narrower than this
    click_relative: float = 0.25     # click threshold as a share of the strip's 95th percentile height
    spike_width: int = 11            # median filter that removes what is left of the clicks
    smooth: int = 15                 # Savitzky-Golay window of the cleaned envelope
    floor_s: float = 0.60            # window of the running floor (longer than one cycle)
    track_sigma: float = 4.0         # Gaussian smoothing before wave tracking, columns
    smoothness: float = 4.0          # rhythm penalty of the tracker
    edge_sigma: float = 1.5          # light smoothing of the signal used for wave edges
    edge_fraction: float = 0.40      # a wave starts and ends at this share of its peak height
    click_before: int = 6            # search window for the closure click around the end of ejection
    click_after: int = 14
    click_accept: float = 0.30       # a beat's click counts if it reaches this share of the typical click
    use_click_start: bool = True     # cycle start = closure click (False: end of the ejection wave)
    use_click_removal: bool = True   # the switches below exist for the ablation runs
    use_floor: bool = True
    use_tracker: bool = True


def click_strength(upper: np.ndarray, lower: np.ndarray, width: int = 13) -> np.ndarray:
    """Valve clicks are narrow and reach both sides of the zero line; flow is wide or one-sided."""
    both = np.minimum(upper, lower).astype(float)
    return both - ndimage.grey_opening(both, size=width)


def click_mask(upper: np.ndarray, lower: np.ndarray, width: int = 13, relative: float = 0.25,
               grow: int = 2) -> tuple[np.ndarray, np.ndarray]:
    strength = click_strength(upper, lower, width)
    scale = float(np.percentile(np.maximum(upper, lower), 95)) if upper.size else 0.0
    if scale <= 0:
        return np.zeros(upper.size, dtype=bool), strength
    return ndimage.binary_dilation(strength > relative * scale, iterations=grow), strength


def clean_envelope(envelope: np.ndarray, mask: np.ndarray | None = None, spike_width: int = 11,
                   smooth: int = 15) -> np.ndarray:
    """Bridge the click columns, remove remaining spikes and smooth lightly."""
    x = envelope.astype(float).copy()
    if x.size < max(spike_width, smooth) + 2:
        return x
    if mask is not None and mask.any() and (~mask).sum() > 10:
        index = np.arange(x.size)
        x[mask] = np.interp(index[mask], index[~mask], x[~mask])
    x = signal.medfilt(x, spike_width)
    return signal.savgol_filter(x, smooth, 2, mode="interp").clip(min=0.0)


def running_floor(x: np.ndarray, window: int) -> np.ndarray:
    """Slowly varying lower bound: clutter next to the zero line that never goes away."""
    window = max(3, int(window))
    return ndimage.uniform_filter1d(ndimage.minimum_filter1d(x, window), window)


def _level(values: np.ndarray, duration_s: float, min_rate_bpm: float = 100.0) -> float:
    if values.size == 0:
        return 0.0
    n_top = int(max(3, min(values.size, round(duration_s * min_rate_bpm / 60.0))))
    return float(np.median(np.sort(values)[-n_top:]))


def track_waves(flow: np.ndarray, pps: float, duration_s: float, sigma: float = 4.0,
                smoothness: float = 4.0, use_tracker: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
    """One peak per cycle on one side, chosen by the rhythm-aware tracker."""
    smooth = ndimage.gaussian_filter1d(flow, sigma)
    strict, _ = signal.find_peaks(smooth, distance=max(1, int(0.25 * pps)))
    level = _level(smooth[strict], duration_s)
    if level <= 0:
        return np.array([], dtype=int), 0.0, smooth
    if not use_tracker:                       # ablation: plain peak picking above half the typical height
        plain, _ = signal.find_peaks(smooth, height=0.5 * level, distance=max(1, int(0.27 * pps)))
        return plain, 0.0, smooth
    candidates, strength = candidate_peaks(smooth, pps, level, min_gap_s=0.07)
    selected, per_beat = track_beats(candidates, strength, pps, smoothness=smoothness)
    return candidates[selected], float(per_beat), smooth


@dataclass
class Prepared:
    """Both sides of a strip after click removal and floor correction."""
    upper_clean: np.ndarray
    lower_clean: np.ndarray
    upper_flow: np.ndarray
    lower_flow: np.ndarray
    click: np.ndarray                 # click strength per column
    click_columns: np.ndarray         # boolean mask
    scale: float                      # 95th percentile of the raw envelope height
    duration_s: float

    def side(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "upper":
            return self.upper_clean, self.upper_flow
        if name == "lower":
            return self.lower_clean, self.lower_flow
        raise ValueError("side must be 'upper' or 'lower'")


def prepare(envelopes: Envelopes, pps: float = PIXELS_PER_SECOND, params: CycleParams = CycleParams()) -> Prepared:
    upper, lower = envelopes.upper.astype(float), envelopes.lower.astype(float)
    mask, strength = click_mask(upper, lower, params.click_width, params.click_relative)
    if not params.use_click_removal:
        mask = np.zeros(upper.size, dtype=bool)
    window = int(params.floor_s * pps)
    cleaned = [clean_envelope(e, mask, params.spike_width, params.smooth) for e in (upper, lower)]
    if params.use_floor:
        flows = [np.clip(c - running_floor(c, window), 0.0, None) for c in cleaned]
    else:
        flows = [c.copy() for c in cleaned]
    scale = float(np.percentile(np.maximum(upper, lower), 95)) if upper.size else 0.0
    return Prepared(cleaned[0], cleaned[1], flows[0], flows[1], strength, mask, scale, upper.size / pps)


def wave_edges(edge_signal: np.ndarray, peaks: np.ndarray, heights: np.ndarray, fraction: float,
               max_half: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns where each wave rises above and falls below ``fraction`` of its peak height."""
    starts, ends = np.empty(peaks.size, dtype=int), np.empty(peaks.size, dtype=int)
    last = edge_signal.size - 1
    for i, peak in enumerate(peaks):
        limit = fraction * heights[i]
        left_bound = max(0, peak - max_half, peaks[i - 1] if i > 0 else 0)
        right_bound = min(last, peak + max_half, peaks[i + 1] if i + 1 < peaks.size else last)
        left = peak
        while left > left_bound and edge_signal[left] > limit:
            left -= 1
        right = peak
        while right < right_bound and edge_signal[right] > limit:
            right += 1
        starts[i], ends[i] = left, right
    return starts, ends


def closure_clicks(click: np.ndarray, ends: np.ndarray, scale: float, before: int = 6, after: int = 14,
                   accept: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Aortic valve closure click next to the end of each ejection wave.

    Returns the click column for every beat and whether a click was actually seen. Where none
    was seen, the usual distance between the end of ejection and the click is used instead.
    """
    position = np.empty(ends.size, dtype=int)
    strength = np.empty(ends.size)
    for i, end in enumerate(ends):
        lo, hi = max(0, end - before), min(click.size, end + after + 1)
        k = lo + int(np.argmax(click[lo:hi])) if hi > lo else int(end)
        position[i], strength[i] = k, click[min(k, click.size - 1)]
    level = float(np.median(strength)) if strength.size else 0.0
    if level < 0.05 * scale:                       # this strip shows no usable closure click
        return ends.copy(), np.zeros(ends.size, dtype=bool)
    seen = strength >= accept * level
    usual = int(round(float(np.median((position - ends)[seen])))) if seen.any() else 0
    return np.where(seen, position, ends + usual), seen


@dataclass
class Cycles:
    ejection_side: str
    ejection_peak: np.ndarray        # column of each V-wave peak
    ejection_start: np.ndarray       # V wave rises above the edge threshold
    ejection_end: np.ndarray         # V wave falls below it
    closure: np.ndarray              # aortic valve closure click (or its usual place): the cycle start
    has_click: np.ndarray            # True where the click itself was seen
    track_score: float               # tracker objective per beat on the ejection side

    @property
    def n(self) -> int:
        return int(self.ejection_peak.size)

    @property
    def starts(self) -> np.ndarray:
        return self.closure


def detect_cycles(envelopes: Envelopes, ejection_side: str = "lower", pps: float = PIXELS_PER_SECOND,
                  params: CycleParams = CycleParams(), prepared: Prepared | None = None) -> Cycles:
    """Track the ejection wave of every beat and place the cycle start at its closure click."""
    prepared = prepared or prepare(envelopes, pps, params)
    _, ejection_flow = prepared.side(ejection_side)
    peaks, score, smooth = track_waves(ejection_flow, pps, prepared.duration_s, params.track_sigma,
                                       params.smoothness, params.use_tracker)
    if peaks.size < 2:
        return Cycles(ejection_side, peaks, peaks.copy(), peaks.copy(), peaks.copy(),
                      np.zeros(peaks.size, dtype=bool), score)
    edge = ndimage.gaussian_filter1d(ejection_flow, params.edge_sigma)
    starts, ends = wave_edges(edge, peaks, smooth[peaks], params.edge_fraction, int(0.5 * pps))
    if params.use_click_start:
        closure, has_click = closure_clicks(prepared.click, ends, prepared.scale, params.click_before,
                                            params.click_after, params.click_accept)
    else:
        closure, has_click = ends.copy(), np.zeros(ends.size, dtype=bool)
    return Cycles(ejection_side, peaks, starts, ends, closure, has_click, score)


def heart_rate_bpm(cycles: Cycles, pps: float = PIXELS_PER_SECOND) -> float:
    if cycles.n < 3:
        return float("nan")
    return float(60.0 * pps / np.median(np.diff(cycles.starts)))
