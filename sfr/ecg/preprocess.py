"""Filtering of abdominal ECG recordings before maternal cancellation."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

TARGET_FS = 1000


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over missing samples; a channel with no valid sample becomes zero."""
    x = np.array(x, dtype=np.float64, copy=True)
    for channel in x.reshape(-1, x.shape[-1]):
        bad = ~np.isfinite(channel)
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            channel[:] = 0.0
        else:
            channel[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), channel[good])
    return x


def resample_to(x: np.ndarray, fs: float, target_fs: int = TARGET_FS) -> np.ndarray:
    if int(round(fs)) == target_fs:
        return x
    ratio = Fraction(target_fs, int(round(fs))).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def mains_frequency(x: np.ndarray, fs: float) -> int:
    """50 or 60 Hz, whichever carries more narrow-band power."""
    freqs, power = signal.welch(x, fs=fs, nperseg=min(x.shape[-1], int(4 * fs)), axis=-1)
    power = np.atleast_2d(power).mean(axis=0)

    def peak_ratio(f0: float) -> float:
        inside = power[(freqs > f0 - 1) & (freqs < f0 + 1)].max()
        around = np.median(power[(freqs > f0 - 6) & (freqs < f0 + 6)])
        return float(inside / (around + 1e-20))

    return 50 if peak_ratio(50.0) >= peak_ratio(60.0) else 60


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    sos = signal.butter(order, [low / nyquist, min(high / nyquist, 0.99)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, fs: float, f0: float, q: float = 35.0) -> np.ndarray:
    if f0 >= fs / 2.0:
        return x
    b, a = signal.iirnotch(f0 / (fs / 2.0), q)
    return signal.filtfilt(b, a, x, axis=-1)


def preprocess(
    x: np.ndarray, fs: float, low: float = 3.0, high: float = 100.0, target_fs: int = TARGET_FS
) -> tuple[np.ndarray, int, dict]:
    """Interpolate gaps, resample to 1 kHz, remove mains and band-pass.

    x has shape (channels, samples). Returns the filtered array, the new sampling rate and
    a small info dict. Channels that are flat after filtering are dropped.
    """
    x = np.atleast_2d(x)
    nan_fraction = float(np.mean(~np.isfinite(x)))
    x = fill_nans(x)
    x = resample_to(x, fs, target_fs)
    mains = mains_frequency(x, target_fs)
    x = notch(x, target_fs, mains)
    if 2 * mains < high + 20:
        x = notch(x, target_fs, 2 * mains)
    x = bandpass(x, target_fs, low, high)
    spread = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    keep = spread > 1e-9
    info = {"mains_hz": mains, "nan_fraction": nan_fraction, "channels_kept": np.flatnonzero(keep).tolist()}
    return x[keep], target_fs, info
