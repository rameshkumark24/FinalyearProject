"""Finding the mother's beats and removing her ECG from abdominal channels."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

from .qrs import MATERNAL, detect_qrs, interval_regularity


def ica_sources(x: np.ndarray, n_components: int | None = None, seed: int = 0) -> np.ndarray:
    """Independent components of a (channels, samples) array, as (components, samples).

    Falls back to principal components if FastICA does not converge.
    """
    n_channels = x.shape[0]
    k = n_channels if n_components is None else min(n_components, n_channels)
    if n_channels == 1:
        return x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            ica = FastICA(n_components=k, whiten="unit-variance", max_iter=600, tol=1e-4, random_state=seed)
            sources = ica.fit_transform(x.T).T
            if np.all(np.isfinite(sources)):
                return sources
        except (ValueError, np.linalg.LinAlgError):
            pass
    return PCA(n_components=k, random_state=seed).fit_transform(x.T).T


def explained_energy(x: np.ndarray, peaks: np.ndarray, fs: float, half_s: float = 0.08) -> float:
    """Share of the channels' energy explained by a beat-locked average at ``peaks``.

    The mother's ECG dominates abdominal channels, so her beats explain far more energy than
    the fetus's. This is how the maternal source is recognised when no chest lead exists.
    """
    half = int(round(half_s * fs))
    inside = peaks[(peaks - half >= 0) & (peaks + half + 1 <= x.shape[1])]
    if inside.size < 5:
        return 0.0
    index = inside[:, None] + np.arange(-half, half + 1)[None, :]
    beats = x[:, index]                       # channels, beats, samples
    template = np.median(beats, axis=1)       # channels, samples
    explained = inside.size * float(np.sum(template ** 2))
    total = float(np.sum(x ** 2)) + 1e-20
    return explained / total


def find_maternal_qrs(x: np.ndarray, fs: float, reference: np.ndarray | None = None, seed: int = 0) -> dict:
    """Locate maternal beats.

    If a chest or thoracic lead is supplied it is used directly. Otherwise each independent
    component and the first principal component are tried, and the one whose beats explain
    most of the abdominal signal energy is kept.
    """
    candidates: list[tuple[str, np.ndarray]] = []
    if reference is not None:
        reference = np.atleast_2d(reference)
        candidates += [(f"reference_{i}", ref) for i, ref in enumerate(reference)]
    else:
        for i, source in enumerate(ica_sources(x, seed=seed)):
            candidates.append((f"ica_{i}", source))
        candidates.append(("pc_0", PCA(n_components=1, random_state=seed).fit_transform(x.T).T[0]))
        candidates += [(f"channel_{i}", channel) for i, channel in enumerate(x)]

    best: dict | None = None
    for name, source in candidates:
        detection = detect_qrs(source, fs, MATERNAL, regularise=True)
        peaks = detection["peaks"]
        if peaks.size < 5:
            continue
        rate = 60.0 * fs / float(np.median(np.diff(peaks)))
        # A chest lead is the mother by construction, so her rate is not limited to the usual
        # resting range (NInFEA record 22: mother at 140 bpm, faster than her fetus).
        if not 40.0 <= rate <= (200.0 if reference is not None else 140.0):
            continue
        regularity = interval_regularity(peaks, tolerance=0.20)
        energy = explained_energy(x, peaks, fs) if reference is None else regularity
        score = energy * (0.25 + 0.75 * regularity)
        if best is None or score > best["score"]:
            best = {"peaks": peaks, "source": name, "score": float(score), "rate_bpm": float(rate),
                    "regularity": float(regularity)}
    if best is None:
        return {"peaks": np.array([], dtype=int), "source": "none", "score": 0.0, "rate_bpm": 0.0,
                "regularity": 0.0}
    return best


def cancel_maternal(
    x: np.ndarray, maternal_peaks: np.ndarray, fs: float, n_components: int = 3,
    block_beats: int = 80, pre_s: float = 0.25, post_s: float = 0.45, ramp_s: float = 0.02,
) -> np.ndarray:
    """Template subtraction with principal components (TS-PCA), channel by channel.

    For every maternal beat a window covering P, QRS and T is taken. Within blocks of
    consecutive beats, each beat is rebuilt from the block's median beat plus its first few
    principal components and the rebuilt beat is subtracted. The fetal ECG is not locked to
    the mother's beats, so it survives.
    """
    x = np.atleast_2d(x).astype(np.float64)
    if maternal_peaks.size < 8:
        return x.copy()
    rr = np.diff(maternal_peaks)
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    shortest = float(np.percentile(rr, 5))
    if pre + post > 0.96 * shortest:
        scale = 0.96 * shortest / (pre + post)
        pre, post = int(pre * scale), int(post * scale)
    length = pre + post
    ramp = max(2, int(round(ramp_s * fs)))
    taper = np.ones(length)
    edge = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    taper[:ramp], taper[-ramp:] = edge, edge[::-1]

    pad = length
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    starts = maternal_peaks + pad - pre
    index = starts[:, None] + np.arange(length)[None, :]

    n_beats = maternal_peaks.size
    edges = list(range(0, n_beats, block_beats))
    if len(edges) > 1 and n_beats - edges[-1] < block_beats // 2:
        edges.pop()
    edges.append(n_beats)

    estimate = np.zeros_like(padded)
    for channel in range(x.shape[0]):
        beats = padded[channel][index]                       # beats, samples
        for lo, hi in zip(edges[:-1], edges[1:]):
            block = beats[lo:hi]
            centre = np.median(block, axis=0)
            deviation = block - centre
            k = min(n_components, block.shape[0] - 1)
            if k > 0:
                _, _, vt = np.linalg.svd(deviation, full_matrices=False)
                basis = vt[:k]
                rebuilt = centre + (deviation @ basis.T) @ basis
            else:
                rebuilt = np.tile(centre, (block.shape[0], 1))
            np.add.at(estimate[channel], index[lo:hi], rebuilt * taper)
    return (padded - estimate)[:, pad:-pad]
