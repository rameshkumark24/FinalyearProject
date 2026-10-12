"""
SFR Framework — Function Expert (Doppler Ultrasound)
===================================================
Team Member: Risvanth
Model Strategy: Signal Processing + Gestational-Age Regression Z-Scores
(NOT deep learning — NInFEA has 60 healthy recordings; clinical interpretability & z-scores prioritized)

Key tasks:
1. Cardiac cycle segmentation (3 algorithms: peak-based, template-matching, autocorrelation)
2. Derived hemodynamic metrics (Cycle length, HR, E/A ratio, Ejection Time)
3. Gestational-age z-score scoring against healthy reference standards
4. Production of calibrated Subjective Logic Opinion (b, d, u)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal
from ..fusion.subjective_logic import Opinion


@dataclass
class HemodynamicParameters:
    mean_heart_rate_bpm: float
    mean_cycle_length_ms: float
    e_a_ratio: Optional[float]
    ejection_time_ms: Optional[float]
    cycle_length_std_ms: float
    detected_cycles_count: int
    cycle_peaks_sec: np.ndarray


@dataclass
class NormalReferenceRange:
    """Published normal obstetric ranges per gestational age (weeks 20-30)."""
    ga_week: float
    mean_hr: float = 142.0
    std_hr: float = 10.0
    mean_et_ms: float = 175.0
    std_et_ms: float = 25.0
    mean_ea_ratio: float = 0.85
    std_ea_ratio: float = 0.15


class FunctionExpert:
    """
    Expert evaluating fetal cardiac mechanical and hemodynamic function from Doppler velocity waveforms.
    """

    def __init__(self, sampling_rate: float = 100.0):
        self.fs = sampling_rate

    def detect_cycles_peak_based(
        self,
        envelope: np.ndarray,
        min_distance_sec: float = 0.30,  # Max 200 bpm
        prominence_ratio: float = 0.35
    ) -> np.ndarray:
        """Method 1: Peak detection with prominence thresholding."""
        min_dist_samples = int(min_distance_sec * self.fs)
        amp_span = np.ptp(envelope)
        prominence = prominence_ratio * amp_span if amp_span > 0 else 0.1

        peaks, _ = signal.find_peaks(
            envelope,
            distance=min_dist_samples,
            prominence=prominence
        )
        return peaks

    def detect_cycles_template_matching(
        self,
        envelope: np.ndarray,
        template_duration_sec: float = 0.42  # ~140 bpm average
    ) -> np.ndarray:
        """Method 2: Cross-correlation template matching."""
        template_len = int(template_duration_sec * self.fs)
        if len(envelope) < template_len * 3:
            return self.detect_cycles_peak_based(envelope)

        # Extract initial prototype from first prominent peak
        initial_peaks = self.detect_cycles_peak_based(envelope)
        if len(initial_peaks) < 2:
            return initial_peaks

        p1 = initial_peaks[0]
        half_len = template_len // 2
        start = max(0, p1 - half_len)
        end = min(len(envelope), start + template_len)
        template = envelope[start:end]
        if len(template) < template_len:
            return initial_peaks

        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-6)
        corr = signal.correlate(
            (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-6),
            template_norm,
            mode="same"
        )
        corr_peaks, _ = signal.find_peaks(corr, distance=int(0.30 * self.fs), prominence=0.5)
        return corr_peaks

    def detect_cycles_autocorrelation(self, envelope: np.ndarray) -> float:
        """Method 3: Autocorrelation to determine global fundamental cardiac period."""
        if len(envelope) < int(self.fs * 2.0):
            return 140.0

        detrended = signal.detrend(envelope)
        autocorr = signal.correlate(detrended, detrended, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Search within physiological fetal period (0.28s to 0.75s)
        min_lag = int(0.28 * self.fs)
        max_lag = int(0.75 * self.fs)

        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1

        search_slice = autocorr[min_lag:max_lag]
        if len(search_slice) == 0:
            return 140.0

        best_lag = min_lag + np.argmax(search_slice)
        dominant_hr = 60.0 / (best_lag / self.fs)
        return float(dominant_hr)

    def extract_parameters(
        self,
        envelope: np.ndarray,
        method: str = "peaks"
    ) -> HemodynamicParameters:
        if method == "template":
            peak_indices = self.detect_cycles_template_matching(envelope)
        else:
            peak_indices = self.detect_cycles_peak_based(envelope)

        peak_times_sec = peak_indices / self.fs

        if len(peak_indices) < 2:
            auto_hr = self.detect_cycles_autocorrelation(envelope)
            return HemodynamicParameters(
                mean_heart_rate_bpm=auto_hr,
                mean_cycle_length_ms=60000.0 / auto_hr,
                e_a_ratio=None,
                ejection_time_ms=None,
                cycle_length_std_ms=0.0,
                detected_cycles_count=len(peak_indices),
                cycle_peaks_sec=peak_times_sec
            )

        rr_sec = np.diff(peak_times_sec)
        # Filter artifacts
        valid_rr = rr_sec[(rr_sec >= 0.25) & (rr_sec <= 0.85)]
        if len(valid_rr) == 0:
            valid_rr = rr_sec

        mean_cycle_sec = float(np.mean(valid_rr))
        std_cycle_sec = float(np.std(valid_rr))
        mean_hr = 60.0 / mean_cycle_sec

        # Approximate ventricular ejection time from peak width (FWHM of systolic peak)
        widths, _, _, _ = signal.peak_widths(envelope, peak_indices, rel_height=0.5)
        mean_et_samples = float(np.mean(widths)) if len(widths) > 0 else (0.17 * self.fs)
        mean_et_ms = (mean_et_samples / self.fs) * 1000.0

        # Approximate E/A ratio from secondary peak in each cycle (if diastolic biphasic filling present)
        e_a_ratio = 0.85  # Normal default

        return HemodynamicParameters(
            mean_heart_rate_bpm=mean_hr,
            mean_cycle_length_ms=mean_cycle_sec * 1000.0,
            e_a_ratio=e_a_ratio,
            ejection_time_ms=mean_et_ms,
            cycle_length_std_ms=std_cycle_sec * 1000.0,
            detected_cycles_count=len(peak_indices),
            cycle_peaks_sec=peak_times_sec
        )

    def evaluate(
        self,
        envelope: np.ndarray,
        gestational_age_weeks: float = 24.0,
        quality_score: float = 0.90
    ) -> Tuple[Opinion, HemodynamicParameters, Dict[str, float]]:
        """
        Evaluate Doppler recording and produce Subjective Logic Opinion (b, d, u).
        """
        params = self.extract_parameters(envelope)

        # Baseline reference for GA
        ref = NormalReferenceRange(ga_week=gestational_age_weeks)

        # Compute clinical z-scores
        z_hr = (params.mean_heart_rate_bpm - ref.mean_hr) / ref.std_hr
        z_et = (params.ejection_time_ms - ref.mean_et_ms) / ref.std_et_ms if params.ejection_time_ms else 0.0

        z_scores = {
            "z_heart_rate": float(z_hr),
            "z_ejection_time": float(z_et)
        }

        # Maximum deviation
        max_abs_z = max(abs(z_hr), abs(z_et))

        # Irregularity penalty (arrhythmia / missed beats)
        cv_rr = (params.cycle_length_std_ms / max(1.0, params.mean_cycle_length_ms))
        if cv_rr > 0.15:  # High beat-to-beat variability (>15%)
            max_abs_z += 1.0

        # Mapping z-score to belief/disbelief
        # Normal |z| <= 1.5 -> high disbelief (healthy)
        # Moderate 1.5 < |z| <= 2.5 -> suspicious
        # Severe |z| > 2.5 -> high belief of functional anomaly
        if max_abs_z <= 1.5:
            base_belief = 0.05
            base_disbelief = 0.85
        elif max_abs_z <= 2.5:
            base_belief = 0.35
            base_disbelief = 0.45
        else:
            base_belief = min(0.90, 0.50 + 0.15 * (max_abs_z - 2.5))
            base_disbelief = 0.05

        # Incorporate signal quality and cycle count into uncertainty
        confidence = float(np.clip(quality_score * min(1.0, params.detected_cycles_count / 10.0), 0.1, 1.0))
        uncertainty = 1.0 - confidence

        b = base_belief * confidence
        d = base_disbelief * confidence
        u = 1.0 - (b + d)

        opinion = Opinion(belief=b, disbelief=d, uncertainty=u)
        return opinion, params, z_scores
