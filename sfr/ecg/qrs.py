"""QRS detection for maternal and fetal beats.

Two modes are offered for fetal beats:

* ``regularise=True``  - after thresholding, missed beats are searched for and extra beats
  removed using the local beat-to-beat interval. Best for heart-rate work on regular rhythms.
* ``regularise=False`` - threshold and refractory period only. Nothing is added or removed on
  the grounds of regularity, so an irregular rhythm stays irregular. Used by the rhythm expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .preprocess import bandpass


@dataclass(frozen=True)
class QRSParams:
    band: tuple[float, float]
    window_s: float        # energy integration window
    refractory_s: float    # minimum distance between beats
    min_rate_bpm: float    # slowest plausible rate, used to size the reference level
    threshold: float = 0.30
    low_threshold: float = 0.12


MATERNAL = QRSParams(band=(5.0, 30.0), window_s=0.10, refractory_s=0.33, min_rate_bpm=45.0)
FETAL = QRSParams(band=(12.0, 45.0), window_s=0.04, refractory_s=0.25, min_rate_bpm=100.0)


def energy_envelope(x: np.ndarray, fs: float, params: QRSParams) -> tuple[np.ndarray, np.ndarray]:
    """Band-passed signal and its smoothed squared energy."""
    y = bandpass(x, fs, *params.band)
    width = max(3, int(round(params.window_s * fs)))
    envelope = signal.convolve(y * y, np.ones(width) / width, mode="same")
    return y, envelope


def _reference_level(heights: np.ndarray, duration_s: float, params: QRSParams) -> float:
    """Typical height of a true beat: median of the largest peaks a slowest-rate rhythm would give."""
    if heights.size == 0:
        return 0.0
    n_top = int(max(3, min(heights.size, round(duration_s * params.min_rate_bpm / 60.0))))
    top = np.sort(heights)[-n_top:]
    return float(np.median(top))


def _regularise(
    peaks: np.ndarray, heights_at: np.ndarray, candidates: np.ndarray, envelope: np.ndarray,
    level: float, params: QRSParams, fs: float, rounds: int = 4,
) -> tuple[np.ndarray, int, int]:
    """Remove extra beats and search back for missed ones using the local interval."""
    peaks = np.array(sorted(set(peaks.tolist())), dtype=int)
    n_added = n_removed = 0
    low = params.low_threshold * level
    for _ in range(rounds):
        if peaks.size < 4:
            break
        changed = False
        rr = np.diff(peaks)
        local = signal.medfilt(rr.astype(float), kernel_size=min(11, rr.size if rr.size % 2 else rr.size - 1) or 1)
        # 1. remove the weaker beat of a pair that is much too close
        keep = np.ones(peaks.size, dtype=bool)
        for i in range(rr.size):
            if keep[i] and keep[i + 1] and rr[i] < 0.70 * local[i]:
                weaker = i if envelope[peaks[i]] < envelope[peaks[i + 1]] else i + 1
                keep[weaker] = False
                n_removed += 1
                changed = True
        peaks = peaks[keep]
        # 2. search back in gaps that are much too long
        rr = np.diff(peaks)
        if rr.size < 3:
            break
        local = signal.medfilt(rr.astype(float), kernel_size=min(11, rr.size if rr.size % 2 else rr.size - 1) or 1)
        additions = []
        for i in range(rr.size):
            if rr[i] > 1.55 * local[i]:
                n_missing = int(round(rr[i] / local[i])) - 1
                for k in range(1, n_missing + 1):
                    expected = peaks[i] + k * rr[i] / (n_missing + 1)
                    lo, hi = expected - 0.18 * local[i], expected + 0.18 * local[i]
                    inside = candidates[(candidates >= lo) & (candidates <= hi)]
                    if inside.size:
                        best = inside[np.argmax(envelope[inside])]
                        if envelope[best] >= low:
                            additions.append(int(best))
        if additions:
            peaks = np.array(sorted(set(peaks.tolist()) | set(additions)), dtype=int)
            n_added += len(additions)
            changed = True
        if not changed:
            break
    return peaks, n_added, n_removed


def detect_qrs(
    x: np.ndarray, fs: float, params: QRSParams, regularise: bool = True
) -> dict:
    """Detect beats in one channel. Returns sample positions and diagnostic values."""
    y, envelope = energy_envelope(x, fs, params)
    distance = max(1, int(round(params.refractory_s * fs)))
    candidates, props = signal.find_peaks(envelope, distance=max(1, distance // 2))
    duration = x.size / fs
    strict, _ = signal.find_peaks(envelope, distance=distance)
    level = _reference_level(envelope[strict], duration, params)
    if level <= 0 or strict.size == 0:
        return {"peaks": np.array([], dtype=int), "level": 0.0, "n_added": 0, "n_removed": 0,
                "raw_peaks": np.array([], dtype=int), "bandpassed": y, "envelope": envelope}
    raw = strict[envelope[strict] >= params.threshold * level]
    peaks, n_added, n_removed = raw, 0, 0
    if regularise:
        peaks, n_added, n_removed = _regularise(raw, envelope, candidates, envelope, level, params, fs)
    peaks = refine_to_extremum(y, peaks, fs)
    return {"peaks": peaks, "raw_peaks": refine_to_extremum(y, raw, fs), "level": level,
            "n_added": n_added, "n_removed": n_removed, "bandpassed": y, "envelope": envelope}


def refine_to_extremum(y: np.ndarray, peaks: np.ndarray, fs: float, half_s: float = 0.03) -> np.ndarray:
    """Move each detection to the extremum of the band-passed signal with the dominant polarity."""
    if peaks.size == 0:
        return peaks
    half = max(1, int(round(half_s * fs)))
    signs = []
    for p in peaks:
        segment = y[max(0, p - half): p + half + 1]
        signs.append(np.sign(segment[np.argmax(np.abs(segment))]))
    polarity = 1.0 if np.sum(signs) >= 0 else -1.0
    refined = []
    for p in peaks:
        start = max(0, p - half)
        segment = polarity * y[start: p + half + 1]
        refined.append(start + int(np.argmax(segment)))
    refined = np.array(sorted(set(refined)), dtype=int)
    return refined


def template_correlation(y: np.ndarray, peaks: np.ndarray, fs: float, half_s: float = 0.04) -> float:
    """Mean correlation of single beats with their median template (0 to 1). A quality index."""
    half = int(round(half_s * fs))
    beats = [y[p - half: p + half + 1] for p in peaks if p - half >= 0 and p + half + 1 <= y.size]
    if len(beats) < 5:
        return 0.0
    beats = np.asarray(beats)
    template = np.median(beats, axis=0)
    template = template - template.mean()
    norm_t = np.linalg.norm(template)
    if norm_t < 1e-12:
        return 0.0
    centred = beats - beats.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    valid = norms > 1e-12
    if not valid.any():
        return 0.0
    corr = (centred[valid] @ template) / (norms[valid] * norm_t)
    return float(np.clip(np.mean(corr), 0.0, 1.0))


def interval_regularity(peaks: np.ndarray, tolerance: float = 0.12) -> float:
    """Share of beat-to-beat intervals within ``tolerance`` of the median interval."""
    if peaks.size < 4:
        return 0.0
    rr = np.diff(peaks).astype(float)
    median = np.median(rr)
    return float(np.mean(np.abs(rr - median) <= tolerance * median))
