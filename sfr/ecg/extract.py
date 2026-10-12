"""Fetal ECG extraction: maternal beats -> maternal cancellation -> source separation -> fetal beats.

The pipeline follows the structure of published classical methods (maternal cancellation by
template subtraction with principal components; independent component analysis; choice of the
best source by a quality index). It adds one guard that the first build lacked: a source whose
beats coincide with the mother's, or sit at any fixed point of her cardiac cycle, is rejected,
and if no acceptable source is found the recording is reported as failed instead of returning
the mother's rhythm. The fixed-point part of the guard was added on 11 Oct 2026 after the first
held-out NInFEA run, in which two recordings came back with the mother's rhythm.

Two sets of fetal beats are returned:

* ``fetal_peaks``     - beat tracking with a rhythm prior (``sfr.ecg.tracker``). For heart rate.
* ``fetal_peaks_raw`` - threshold and refractory period only. For rhythm analysis, where the
  detector must not smooth an irregular rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .maternal import cancel_maternal, find_maternal_qrs, ica_sources
from .preprocess import preprocess
from .qrs import FETAL, detect_qrs, interval_regularity, refine_to_extremum, template_correlation
from .tracker import candidate_peaks, track_beats

MIN_FETAL_BPM = 90.0
MAX_FETAL_BPM = 220.0


@dataclass
class Extraction:
    fs: int
    fetal_peaks: np.ndarray            # samples at ``fs``; tracked with a rhythm prior
    fetal_peaks_raw: np.ndarray        # threshold-only detections, nothing added or removed
    maternal_peaks: np.ndarray
    fetal_signal: np.ndarray           # the selected source (band 3-100 Hz)
    source: str
    ok: bool                           # False when no acceptable fetal source was found
    quality: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)


def coincidence(fetal_peaks: np.ndarray, maternal_peaks: np.ndarray, fs: float, window_s: float = 0.04) -> float:
    """Share of detected beats that fall within ``window_s`` of a maternal beat."""
    if fetal_peaks.size == 0 or maternal_peaks.size == 0:
        return 0.0
    position = np.searchsorted(maternal_peaks, fetal_peaks)
    left = maternal_peaks[np.clip(position - 1, 0, maternal_peaks.size - 1)]
    right = maternal_peaks[np.clip(position, 0, maternal_peaks.size - 1)]
    nearest = np.minimum(np.abs(fetal_peaks - left), np.abs(fetal_peaks - right))
    return float(np.mean(nearest <= window_s * fs))


def maternal_phase_lock(peaks: np.ndarray, maternal_peaks: np.ndarray) -> tuple[float, float]:
    """How firmly the detected beats sit at one fixed point of the mother's cardiac cycle.

    Returns the resultant length of the beat phases inside the mother's R-R intervals (0 = no
    relation, 1 = every beat at the same point) and its Rayleigh p-value. A fetal rhythm runs
    free of the mother's, so its phases are spread out. What is left of the mother after
    cancellation is locked to her beats, at her R peak or at any other fixed delay.
    Only the same-rate relation is tested: a fetal rate close to twice the maternal rate is
    common and does lock at two points over a short recording (seen on development data).
    """
    peaks, maternal_peaks = np.asarray(peaks, dtype=float), np.asarray(maternal_peaks, dtype=float)
    if peaks.size < 8 or maternal_peaks.size < 4:
        return 0.0, 1.0
    index = np.searchsorted(maternal_peaks, peaks, side="right") - 1
    inside = (index >= 0) & (index < maternal_peaks.size - 1)
    if inside.sum() < 8:
        return 0.0, 1.0
    index, beats = index[inside], peaks[inside]
    phase = (beats - maternal_peaks[index]) / (maternal_peaks[index + 1] - maternal_peaks[index])
    length = float(np.abs(np.mean(np.exp(2j * np.pi * phase))))
    return length, float(np.exp(-beats.size * length * length))


def is_maternal(rate: float, overlap: float, maternal_bpm: float,
                phase_lock: tuple[float, float] = (0.0, 1.0)) -> bool:
    """A source is treated as the mother's if its beats coincide with hers far above chance,
    or sit at a fixed point of her cycle (resultant length at least 0.5, p below 0.001)."""
    chance = min(1.0, 2 * 0.04 * maternal_bpm / 60.0) if maternal_bpm > 0 else 0.0
    locked = phase_lock[0] >= 0.5 and phase_lock[1] < 1e-3
    return bool(locked or overlap > max(0.5, 3.0 * chance) or (abs(rate - maternal_bpm) < 6.0 and overlap > 0.35))


def screen_candidate(source: np.ndarray, fs: float, maternal_peaks: np.ndarray, maternal_bpm: float) -> dict:
    """Quick look at one source: threshold detection and a first quality score."""
    detection = detect_qrs(source, fs, FETAL, regularise=True)
    peaks, raw = detection["peaks"], detection["raw_peaks"]
    result = {"detection": detection, "raw_peaks": raw, "screen_score": 0.0, "screen_rate_bpm": float("nan"),
              "template_corr": 0.0, "regularity_raw": 0.0}
    if peaks.size < 6:
        return result
    rate = 60.0 * fs / float(np.median(np.diff(peaks)))
    corr = template_correlation(detection["bandpassed"], peaks, fs)
    regularity_raw = interval_regularity(raw)
    overlap = coincidence(peaks, maternal_peaks, fs)
    score = corr * (0.3 + 0.7 * regularity_raw)
    lock = is_maternal(rate, overlap, maternal_bpm, maternal_phase_lock(peaks, maternal_peaks))
    if not MIN_FETAL_BPM <= rate <= MAX_FETAL_BPM or lock:
        score = 0.0
    result.update({"screen_score": float(score), "screen_rate_bpm": float(rate),
                   "template_corr": float(corr), "regularity_raw": float(regularity_raw)})
    return result


def track_candidate(screened: dict, fs: float, maternal_peaks: np.ndarray, maternal_bpm: float,
                    smoothness: float = 4.0) -> dict:
    """Beat tracking on one source and the quality values used to choose between sources."""
    detection = screened["detection"]
    candidates, strength = candidate_peaks(detection["envelope"], fs, detection["level"])
    selected, per_beat = track_beats(candidates, strength, fs, smoothness=smoothness)
    result = dict(screened)
    result.update({"peaks": np.array([], dtype=int), "score": 0.0, "rate_bpm": float("nan"),
                   "regularity": 0.0, "weak_share": 1.0, "maternal_coincidence": 0.0, "maternal_lock": False,
                   "maternal_phase_lock": 0.0})
    if selected.size < 6:
        return result
    peaks = refine_to_extremum(detection["bandpassed"], candidates[selected], fs)
    rate = 60.0 * fs / float(np.median(np.diff(peaks)))
    overlap = coincidence(peaks, maternal_peaks, fs)
    phase_lock = maternal_phase_lock(peaks, maternal_peaks)
    lock = is_maternal(rate, overlap, maternal_bpm, phase_lock)
    plausible = MIN_FETAL_BPM <= rate <= MAX_FETAL_BPM
    regularity = interval_regularity(peaks)
    chance = min(1.0, 2 * 0.04 * maternal_bpm / 60.0) if maternal_bpm > 0 else 0.0
    excess = float(np.clip((overlap - chance) / max(1e-6, 1.0 - chance), 0.0, 1.0))
    # Strong beats (per_beat), a smooth rhythm after tracking, and no more coincidence with the
    # mother's beats than chance allows.
    score = float(per_beat) * (0.3 + 0.7 * regularity) * (1.0 - excess)
    result.update({
        "peaks": peaks,
        "score": score if plausible and not lock else 0.0,
        "rate_bpm": float(rate),
        "regularity": float(regularity),
        "weak_share": float(np.mean(strength[selected] < 0.25)),
        "maternal_coincidence": float(overlap),
        "maternal_lock": lock,
        "maternal_phase_lock": float(phase_lock[0]),
    })
    return result


def untracked_candidate(screened: dict, fs: float, maternal_peaks: np.ndarray, maternal_bpm: float) -> dict:
    """Ablation: keep the threshold-and-search-back detections instead of tracked beats."""
    peaks = screened["detection"]["peaks"]
    result = dict(screened)
    overlap = coincidence(peaks, maternal_peaks, fs)
    rate = screened["screen_rate_bpm"]
    phase_lock = maternal_phase_lock(peaks, maternal_peaks)
    result.update({"peaks": peaks, "score": screened["screen_score"], "rate_bpm": rate,
                   "regularity": interval_regularity(peaks), "weak_share": float("nan"),
                   "maternal_coincidence": float(overlap), "maternal_phase_lock": float(phase_lock[0]),
                   "maternal_lock": bool(np.isfinite(rate) and is_maternal(rate, overlap, maternal_bpm, phase_lock))})
    return result


def extract_fetal_ecg(
    x: np.ndarray, fs: float, maternal_reference: np.ndarray | None = None,
    n_components: int = 3, max_sources: int = 8, n_tracked: int = 3, seed: int = 0,
    min_score: float = 0.20, smoothness: float = 4.0,
    cancel: bool = True, separate: bool = True, track: bool = True,
) -> Extraction:
    """Run the whole pipeline on a (channels, samples) abdominal recording.

    ``cancel``, ``separate`` and ``track`` switch off maternal cancellation, independent
    component analysis and beat tracking. They exist for the ablation table only.
    """
    filtered, fs_out, info = preprocess(x, fs)
    reference = None
    if maternal_reference is not None:
        reference, _, _ = preprocess(np.atleast_2d(maternal_reference), fs)
        if reference.shape[0] == 0:
            reference = None

    maternal = find_maternal_qrs(filtered, fs_out, reference=reference, seed=seed)
    maternal_peaks = maternal["peaks"]
    residual = filtered
    if cancel:
        residual = cancel_maternal(filtered, maternal_peaks, fs_out, n_components=n_components)

    sources: list[tuple[str, np.ndarray]] = [(f"residual_{i}", r) for i, r in enumerate(residual)]
    if separate and residual.shape[0] > 1:
        separated = ica_sources(residual, n_components=min(max_sources, residual.shape[0]), seed=seed)
        sources += [(f"ica_{i}", s) for i, s in enumerate(separated)]

    screened = []
    for name, source in sources:
        values = screen_candidate(source, fs_out, maternal_peaks, maternal["rate_bpm"])
        values["name"] = name
        screened.append((values, source))
    screened.sort(key=lambda item: item[0]["screen_score"], reverse=True)

    tracked = []
    for values, source in screened[:n_tracked]:
        if track:
            tracked.append((track_candidate(values, fs_out, maternal_peaks, maternal["rate_bpm"], smoothness), source))
        else:
            tracked.append((untracked_candidate(values, fs_out, maternal_peaks, maternal["rate_bpm"]), source))
    tracked.sort(key=lambda item: item[0]["score"], reverse=True)
    best, best_source = tracked[0]
    ok = bool(best["score"] >= min_score)

    keys = ("score", "rate_bpm", "template_corr", "regularity_raw", "regularity", "weak_share",
            "maternal_coincidence", "maternal_lock", "maternal_phase_lock", "screen_score")
    quality = {k: best[k] for k in keys}
    quality.update({"maternal_source": maternal["source"], "maternal_bpm": maternal["rate_bpm"],
                    "maternal_regularity": maternal["regularity"], "mains_hz": info["mains_hz"],
                    "nan_fraction": info["nan_fraction"], "n_channels": int(filtered.shape[0])})
    summary = [{"name": values["name"], **{k: values[k] for k in keys}} for values, _ in tracked]
    return Extraction(fs=fs_out, fetal_peaks=best["peaks"], fetal_peaks_raw=best["raw_peaks"],
                      maternal_peaks=maternal_peaks, fetal_signal=best_source, source=best["name"],
                      ok=ok, quality=quality, candidates=summary)
