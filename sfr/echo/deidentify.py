"""Keep the ultrasound picture, drop everything written on or around it.

A saved fetal echo frame holds the scan (a grey sector on a black screen) plus text burned into
the pixels: patient name and number, hospital, date, machine settings, and often calipers and
measurement boxes drawn on the scan. That text must go for two reasons: privacy, and shortcut
learning (calipers appear more often on abnormal hearts, so a model can learn to read them).

Steps:
1. find the scan: the largest solid region of non-black pixels once thin strokes (text, rulers)
   have been removed by a morphological opening;
2. blank every pixel outside that region;
3. inside it, blank coloured pixels (calipers, colour boxes) and thin saturated-white strokes;
4. crop to the region.

This is a rule-based tool, not a guarantee. Before any image leaves the machine a person must
look at a sample of the output at full resolution.

Only NumPy and SciPy are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Deidentified:
    image: np.ndarray            # grey, cropped to the scan, everything else zero
    mask: np.ndarray             # boolean, the scan region in the original frame
    box: tuple[int, int, int, int]            # top, bottom, left, right of the crop
    scan_share: float            # share of the frame kept
    coloured_removed: int        # coloured overlay pixels blanked inside the scan
    white_strokes_removed: int   # thin saturated-white pixels blanked inside the scan
    outside_nonzero: int         # non-black pixels outside the scan before blanking (text, logos)


def _disk(diameter: int) -> np.ndarray:
    radius = diameter / 2.0
    yy, xx = np.mgrid[:diameter, :diameter]
    return (yy - radius + 0.5) ** 2 + (xx - radius + 0.5) ** 2 <= radius ** 2


def to_grey(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114])
    return np.clip(np.rint(rgb[..., :3].astype(np.float64) @ weights), 0, 255).astype(np.uint8)


def hue_degrees(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue (0-360) and saturation (0-1) of an RGB array."""
    x = rgb[..., :3].astype(np.float64) / 255.0
    high, low = x.max(axis=-1), x.min(axis=-1)
    spread = high - low
    safe = np.where(spread > 0, spread, 1.0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hue = np.where(high == r, ((g - b) / safe) % 6, np.where(high == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    hue = np.where(spread > 0, hue, 0.0)
    saturation = np.where(high > 0, spread / np.where(high > 0, high, 1.0), 0.0)
    return hue, saturation


def scan_region(grey: np.ndarray, floor: int = 12, stroke: int = 7, close: int = 21) -> np.ndarray:
    """Boolean mask of the scan."""
    bright = grey > floor
    opened = ndimage.binary_opening(bright, structure=_disk(stroke))
    closed = ndimage.binary_closing(opened, structure=_disk(close), border_value=0)
    labels, count = ndimage.label(closed)
    if count == 0:
        return np.zeros(grey.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)))


def deidentify(rgb: np.ndarray, colour_spread: int = 40, white_level: int = 245) -> Deidentified:
    """Blank text and overlays of one frame (RGB or grey array)."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
    grey = to_grey(rgb)
    mask = scan_region(grey)
    if not mask.any():
        raise ValueError("No scan region found; this does not look like an ultrasound frame")

    channels = rgb.astype(np.int16)
    coloured = (channels.max(axis=2) - channels.min(axis=2)) > colour_spread
    if coloured[mask].mean() > 0.30:
        # The scan itself is shown with a colour tint (sepia or blue maps). Then only pixels whose
        # hue differs from the tint are overlays.
        hue, saturation = hue_degrees(rgb)
        tint = float(np.median(hue[mask & coloured]))
        away = np.abs((hue - tint + 180.0) % 360.0 - 180.0) > 30.0
        coloured = coloured & away & (saturation > 0.47)
    coloured = ndimage.binary_dilation(coloured, structure=np.ones((3, 3), dtype=bool))
    white = grey >= white_level
    strokes = white & ~ndimage.binary_opening(white, structure=np.ones((5, 5), dtype=bool))

    out = grey.copy()
    outside_nonzero = int(np.count_nonzero(out[~mask] > 12))
    out[~mask] = 0
    out[mask & coloured] = 0
    out[mask & strokes] = 0
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return Deidentified(out[top:bottom, left:right], mask, (top, bottom, left, right), float(mask.mean()),
                        int(np.count_nonzero(mask & coloured)), int(np.count_nonzero(mask & strokes)), outside_nonzero)


def overlay_only(rgb: np.ndarray) -> np.ndarray:
    """The complement: keep only what ``deidentify`` removes, blank the scan.

    Used for the shortcut audit: a classifier trained on these images sees text, calipers and
    boxes but no anatomy. If it predicts the label, the label leaks through the overlays.
    """
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    grey = to_grey(rgb)
    result = deidentify(rgb)
    top, bottom, left, right = result.box
    kept = np.zeros_like(grey)
    kept[top:bottom, left:right] = result.image
    removed = grey.copy()
    removed[kept > 0] = 0
    return removed
