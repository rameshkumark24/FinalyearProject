"""Velocity envelopes from a pulsed-wave Doppler strip image.

A Python port of the method shipped with the NInFEA dataset (``code/envelope_extraction.m``,
Ortu, Sulas and Pani, University of Cagliari, 2020): grey conversion, removal of the zero
line, a two-dimensional median-Otsu threshold, removal of small connected regions, and, for
every image column, the outermost bright pixel above and below the zero line.

The strips used here are five-chamber recordings: ventricular filling (E and A waves) on one
side of the zero line and aortic ejection (V wave) on the other. One image column is 1/284 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

PIXELS_PER_SECOND = 284.0


@dataclass
class Envelopes:
    upper: np.ndarray        # height above the zero line, pixels, one value per column
    lower: np.ndarray        # depth below the zero line, pixels (positive), one value per column
    baseline_row: int
    threshold: int
    height: int
    width: int

    @property
    def duration_s(self) -> float:
        return self.width / PIXELS_PER_SECOND


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Grey conversion with the same weights as MATLAB's rgb2gray."""
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    weights = np.array([0.2989, 0.5870, 0.1140])
    return np.clip(np.rint(rgb[..., :3].astype(np.float64) @ weights), 0, 255).astype(np.uint8)


def find_baseline(gray: np.ndarray) -> int:
    """Row of the white zero-velocity line: the brightest row of the image."""
    row_mean = gray.mean(axis=1)
    row = int(np.argmax(row_mean))
    if row_mean[row] < 150:
        raise ValueError("No bright zero line found; this does not look like a Doppler strip")
    return row


def colour_marker_mask(rgb: np.ndarray, min_spread: int = 50) -> np.ndarray:
    """Coloured pixels (the scale tick marks). The Doppler spectrum itself is grey."""
    if rgb.ndim == 2:
        return np.zeros(rgb.shape, dtype=bool)
    channels = rgb[..., :3].astype(np.int16)
    return (channels.max(axis=2) - channels.min(axis=2)) > min_spread


def median_otsu_2d(gray: np.ndarray, kernel: int = 5) -> int:
    """Two-dimensional Otsu threshold on (pixel value, local median), as in the dataset's code.

    Returns the threshold on the median axis, using the same index convention as the
    MATLAB function so that ``gray > threshold`` reproduces ``imbinarize(x, T/255)``.
    """
    median = cv2.medianBlur(gray, kernel)
    half = kernel // 2
    f = gray[half:-half, half:-half].ravel().astype(np.intp)
    g = median[half:-half, half:-half].ravel().astype(np.intp)
    joint = np.bincount(f * 256 + g, minlength=256 * 256).reshape(256, 256).astype(np.float64)
    p = joint / joint.sum()
    levels = np.arange(256, dtype=np.float64)

    def cumulative(matrix: np.ndarray) -> np.ndarray:
        padded = np.zeros((257, 257))
        padded[1:, 1:] = matrix.cumsum(axis=0).cumsum(axis=1)
        return padded                     # padded[s, t] = sum over rows < s and columns < t

    c_p = cumulative(p)
    c_i = cumulative(p * levels[:, None])
    c_j = cumulative(p * levels[None, :])
    total_i, total_j = c_i[256, 256], c_j[256, 256]

    s = np.arange(256)[:, None]
    t = np.arange(256)[None, :]
    w0 = c_p[s, t]
    m0_i, m0_j = c_i[s, t], c_j[s, t]
    # quadrant with rows >= s and columns >= t, by inclusion-exclusion
    w1 = c_p[256, 256] - c_p[s, 256] - c_p[256, t] + w0
    m1_i = total_i - c_i[s, 256] - c_i[256, t] + m0_i
    m1_j = total_j - c_j[s, 256] - c_j[256, t] + m0_j
    with np.errstate(divide="ignore", invalid="ignore"):
        u0_i, u0_j = m0_i / w0, m0_j / w0
        u1_i, u1_j = m1_i / w1, m1_j / w1
        trace = (w0 * ((u0_i - total_i) ** 2 + (u0_j - total_j) ** 2)
                 + w1 * ((u1_i - total_i) ** 2 + (u1_j - total_j) ** 2))
    trace[~np.isfinite(trace)] = -np.inf
    best_t = int(np.argmax(trace.max(axis=0)))
    return best_t + 1


def remove_small_regions(mask: np.ndarray, min_pixels: int = 70) -> np.ndarray:
    """Drop 4-connected bright regions smaller than ``min_pixels`` (MATLAB bwareaopen(BW, 70, 4))."""
    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def extract_envelopes(rgb: np.ndarray, min_region: int = 70) -> Envelopes:
    """Upper and lower velocity envelopes of a Doppler strip (RGB or grey image array)."""
    gray = to_gray(rgb).copy()
    gray[colour_marker_mask(rgb)] = 0
    baseline = find_baseline(gray)
    row_mean = gray.mean(axis=1)
    thick = [baseline]
    for step in (-1, 1):                      # the zero line can be more than one pixel thick
        row = baseline + step
        while 0 <= row < gray.shape[0] and row_mean[row] > 0.8 * row_mean[baseline]:
            thick.append(row)
            row += step
    top, bottom = min(thick), max(thick)
    gray[top:bottom + 1, :] = 0

    threshold = median_otsu_2d(gray)
    mask = remove_small_regions(gray > threshold, min_region)

    above = mask[:top, :]
    below = mask[bottom + 1:, :]
    any_above = above.any(axis=0)
    first_above = np.argmax(above, axis=0)                      # first bright row from the top
    upper = np.where(any_above, top - first_above, 0).astype(np.float64)
    any_below = below.any(axis=0)
    last_below = below.shape[0] - 1 - np.argmax(below[::-1, :], axis=0)
    lower = np.where(any_below, last_below + 1, 0).astype(np.float64)
    return Envelopes(upper=upper, lower=lower, baseline_row=baseline, threshold=threshold,
                     height=gray.shape[0], width=gray.shape[1])
