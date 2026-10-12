"""Timing measures of one recording, taken from its cycle-averaged waveform.

Single beats of a fetal Doppler strip are noisy, the E wave is often only a shoulder on the A
wave, and the foot of the ejection wave is hidden by low flow next to the zero line. Wave
onsets can therefore not be read reliably from the envelope. What can be read reliably are the
valve closure click, the peaks of the A and V waves, and the widths of those waves at half
height. The measures below use only these.

The cycles of a recording are stretched to a common length and their median is taken. Phase 0
of this template is the aortic valve closure: filling comes first, then the ejection wave of
the next beat. One set of measures is produced per recording, not per beat.

These are envelope-derived measures. They are related to, but not the same as, the clinical
ejection time, atrioventricular interval and E/A ratio, which are measured by hand from wave
onsets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .cycles import Cycles, Prepared
from .envelope import PIXELS_PER_SECOND

TEMPLATE_MEASURES = (
    "cycle_length_ms",          # closure to closure
    "ejection_half_width_ms",   # width of the V wave at half its height
    "a_to_v_peak_ms",           # A-wave peak to V-wave peak (atrial contraction to ejection)
    "v_peak_to_closure_ms",     # V-wave peak to valve closure
    "closure_to_a_peak_ms",     # valve closure to A-wave peak
    "a_half_width_ms",          # width of the A wave at half its height
    "ea_ratio",                 # E height over A height, only where E is a peak of its own
    "e_to_a_ms",                # E-wave peak to A-wave peak, same condition
)
MIN_CYCLES = 8


@dataclass
class Template:
    ejection: np.ndarray           # floor-corrected flow of the ejection side, one cycle
    filling: np.ndarray            # floor-corrected flow of the filling side
    filling_height: np.ndarray     # envelope height above the zero line (for the E/A ratio)
    n_cycles: int
    cycle_px: float                # mean length of the averaged cycles, columns
    rows_ejection: np.ndarray      # the single cycles that were averaged (n_cycles x bins)
    rows_filling: np.ndarray
    cycle_index: np.ndarray        # which cycle each row is

    @property
    def bins(self) -> int:
        return int(self.ejection.size)


def build_template(cycles: Cycles, prepared: Prepared, bins: int = 128, tolerance: float = 0.15,
                   keep: np.ndarray | None = None) -> Template | None:
    """Median waveform over the cycles whose length is within ``tolerance`` of the typical one."""
    starts = cycles.starts
    if starts.size < 4:
        return None
    lengths = np.diff(starts).astype(float)
    typical = float(np.median(lengths))
    use = (lengths >= (1 - tolerance) * typical) & (lengths <= (1 + tolerance) * typical)
    if keep is not None:
        use &= np.asarray(keep, dtype=bool)[: use.size]
    index = np.flatnonzero(use)
    if index.size < 3:
        return None
    _, ejection_flow = prepared.side(cycles.ejection_side)
    filling_clean, filling_flow = prepared.side("upper" if cycles.ejection_side == "lower" else "lower")
    columns = np.arange(ejection_flow.size)
    phase = np.arange(bins) / bins
    e_rows, f_rows, h_rows = [], [], []
    for i in index:
        at = starts[i] + phase * lengths[i]
        e_rows.append(np.interp(at, columns, ejection_flow))
        f_rows.append(np.interp(at, columns, filling_flow))
        h_rows.append(np.interp(at, columns, filling_clean))
    e_rows, f_rows = np.array(e_rows), np.array(f_rows)
    return Template(np.median(e_rows, axis=0), np.median(f_rows, axis=0), np.median(np.array(h_rows), axis=0),
                    int(index.size), float(lengths[index].mean()), e_rows, f_rows, index)


def _half_width(x: np.ndarray, peak: int) -> float:
    """Width at half the peak height, measured from zero flow, with linear interpolation."""
    half = 0.5 * x[peak]
    left = peak
    while left > 0 and x[left] > half:
        left -= 1
    right = peak
    while right < x.size - 1 and x[right] > half:
        right += 1
    if x[left] > half or x[right] > half:
        return float("nan")
    left_edge = left + (half - x[left]) / max(x[left + 1] - x[left], 1e-9)
    right_edge = right - (half - x[right]) / max(x[right - 1] - x[right], 1e-9)
    return float(right_edge - left_edge)


def template_measures(template: Template | None, pps: float = PIXELS_PER_SECOND,
                      min_e_prominence: float = 0.08) -> dict[str, float]:
    """Measures read from the averaged waveform. NaN where a wave cannot be read."""
    out = {name: float("nan") for name in TEMPLATE_MEASURES}
    if template is None or template.n_cycles < MIN_CYCLES:
        return out
    bins = template.bins
    ms_per_bin = 1000.0 * template.cycle_px / bins / pps
    out["cycle_length_ms"] = bins * ms_per_bin

    ejection = ndimage.gaussian_filter1d(template.ejection, 1.0, mode="wrap")
    filling = ndimage.gaussian_filter1d(template.filling, 1.0, mode="wrap")
    if ejection.max() <= 0 or filling.max() <= 0:
        return out

    # The ejection wave ends at the closure (phase 1), so its peak lies in the second half.
    lo = int(0.45 * bins)
    v_peak = lo + int(np.argmax(ejection[lo:]))
    out["v_peak_to_closure_ms"] = (bins - v_peak) * ms_per_bin
    out["ejection_half_width_ms"] = _half_width(np.concatenate([ejection, ejection[: bins // 4]]), v_peak) * ms_per_bin

    # The A wave is the largest filling peak between the closure and the rise of the ejection wave.
    lo, hi = int(0.05 * bins), v_peak - int(0.08 * bins)
    if hi - lo < 8:
        return out
    a_peak = lo + int(np.argmax(filling[lo:hi]))
    if filling[a_peak] <= 0:
        return out
    out["closure_to_a_peak_ms"] = a_peak * ms_per_bin
    out["a_to_v_peak_ms"] = (v_peak - a_peak) * ms_per_bin
    out["a_half_width_ms"] = _half_width(filling, a_peak) * ms_per_bin

    # E wave: a peak of its own before the A wave.
    found, props = signal.find_peaks(filling[:a_peak], prominence=min_e_prominence * filling[a_peak])
    found = found[found <= a_peak - int(0.06 * bins)]
    if found.size:
        e_peak = int(found[np.argmax(filling[found])])
        out["ea_ratio"] = float(template.filling_height[e_peak] / max(template.filling_height[a_peak], 1e-9))
        out["e_to_a_ms"] = (a_peak - e_peak) * ms_per_bin
    return out


def cycle_similarity(template: Template | None) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of each averaged cycle with the template (both sides together).

    Returns (cycle index, correlation). A low value marks a cycle that does not look like the
    recording's typical cycle.
    """
    if template is None:
        return np.array([], dtype=int), np.array([])
    reference = np.concatenate([template.ejection, template.filling])
    reference = reference - reference.mean()
    norm = np.linalg.norm(reference)
    values = []
    for e_row, f_row in zip(template.rows_ejection, template.rows_filling):
        row = np.concatenate([e_row, f_row])
        row = row - row.mean()
        denominator = norm * np.linalg.norm(row)
        values.append(float(row @ reference / denominator) if denominator > 0 else 0.0)
    return template.cycle_index, np.array(values)
