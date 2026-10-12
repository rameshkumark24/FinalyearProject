"""
SFR Framework — Cross-Modal Trust Check (ECG-Doppler Agreement)
==============================================================
Validates physiological consistency between Fetal ECG and Doppler Ultrasound.
Both modalities capture mechanical / electrical cardiac beats.

Key failure mode in fetal monitoring:
- Maternal ECG leakage: Maternal heart rate (~70-90 bpm) is detected instead of fetal (~110-160 bpm).
- If ECG detects 140 bpm (fetal) but Doppler detects 75 bpm (maternal uterine artery instead of fetal aorta/umbilical),
  the system MUST NOT blindly fuse them.
- Instead, the Trust Check flags sensor discrepancy and raises epistemic uncertainty u.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import signal
from ..fusion.subjective_logic import Opinion


@dataclass
class TrustCheckResult:
    is_trusted: bool
    hr_ecg_bpm: float
    hr_doppler_bpm: float
    hr_delta_bpm: float
    correlation_score: float
    trust_score: float  # In [0, 1], 1 = full trust, 0 = completely untrusted
    explanation: str


class CrossModalTrustChecker:
    """
    Evaluates temporal and rate consistency between Fetal ECG and Doppler Ultrasound.
    """

    def __init__(
        self,
        hr_tolerance_bpm: float = 10.0,
        min_correlation: float = 0.50,
        uncertainty_inflation_factor: float = 1.5
    ):
        self.hr_tolerance_bpm = hr_tolerance_bpm
        self.min_correlation = min_correlation
        self.uncertainty_inflation = uncertainty_inflation_factor

    def compute_instantaneous_hr(self, event_times_sec: np.ndarray) -> float:
        """Calculate mean heart rate in beats per minute (BPM) from peak timestamps."""
        if len(event_times_sec) < 2:
            return 0.0
        rr_intervals = np.diff(event_times_sec)
        # Filter physiological outliers for fetal HR (80 to 220 bpm -> 0.27s to 0.75s)
        valid_rr = rr_intervals[(rr_intervals >= 0.25) & (rr_intervals <= 0.85)]
        if len(valid_rr) == 0:
            valid_rr = rr_intervals
        median_rr = np.median(valid_rr)
        if median_rr <= 0:
            return 0.0
        return float(60.0 / median_rr)

    def compute_beat_train_correlation(
        self,
        ecg_r_peaks_sec: np.ndarray,
        doppler_peaks_sec: np.ndarray,
        duration_sec: float,
        fs_resample: int = 100
    ) -> float:
        """
        Convert sparse peak times into continuous binary/Gaussian impulse trains and compute normalized cross-correlation.
        """
        if len(ecg_r_peaks_sec) < 2 or len(doppler_peaks_sec) < 2 or duration_sec <= 0:
            return 0.0

        n_samples = int(duration_sec * fs_resample)
        t_grid = np.linspace(0, duration_sec, n_samples)

        train_ecg = np.zeros(n_samples)
        train_dop = np.zeros(n_samples)

        # Place impulses
        ecg_idx = np.clip((ecg_r_peaks_sec * fs_resample).astype(int), 0, n_samples - 1)
        dop_idx = np.clip((doppler_peaks_sec * fs_resample).astype(int), 0, n_samples - 1)

        train_ecg[ecg_idx] = 1.0
        train_dop[dop_idx] = 1.0

        # Smooth with Gaussian kernel (sigma ~ 50ms = 5 samples) to account for electromechanical delay (EMD ~ 40-70ms)
        kernel = signal.windows.gaussian(21, std=3)
        kernel /= np.sum(kernel)

        smooth_ecg = signal.convolve(train_ecg, kernel, mode="same")
        smooth_dop = signal.convolve(train_dop, kernel, mode="same")

        std_e = np.std(smooth_ecg)
        std_d = np.std(smooth_dop)

        if std_e < 1e-6 or std_d < 1e-6:
            return 0.0

        # Maximum cross-correlation within physical lag window (±150ms = ±15 samples)
        max_lag = int(0.15 * fs_resample)
        corr = signal.correlate(smooth_ecg - np.mean(smooth_ecg), smooth_dop - np.mean(smooth_dop), mode="same")
        mid = len(corr) // 2
        local_corr = corr[mid - max_lag : mid + max_lag + 1]
        norm = np.sqrt(np.sum((smooth_ecg - np.mean(smooth_ecg)) ** 2) * np.sum((smooth_dop - np.mean(smooth_dop)) ** 2))
        if norm < 1e-6:
            return 0.0
        max_val = np.max(local_corr) / norm
        return float(np.clip(max_val, -1.0, 1.0))

    def evaluate(
        self,
        ecg_r_peaks_sec: np.ndarray,
        doppler_peaks_sec: np.ndarray,
        duration_sec: float
    ) -> TrustCheckResult:
        hr_ecg = self.compute_instantaneous_hr(ecg_r_peaks_sec)
        hr_doppler = self.compute_instantaneous_hr(doppler_peaks_sec)
        hr_delta = abs(hr_ecg - hr_doppler)

        corr = self.compute_beat_train_correlation(ecg_r_peaks_sec, doppler_peaks_sec, duration_sec)

        # Trust score formula: penalty on HR delta and correlation drop
        hr_penalty = max(0.0, min(1.0, hr_delta / (2.0 * self.hr_tolerance_bpm)))
        corr_credit = max(0.0, min(1.0, (corr - 0.2) / 0.8)) if corr > 0.2 else 0.0

        trust_score = float(np.clip((1.0 - hr_penalty) * (0.5 + 0.5 * corr_credit), 0.0, 1.0))

        is_trusted = (hr_delta <= self.hr_tolerance_bpm) and (corr >= self.min_correlation or hr_delta <= 4.0)

        if is_trusted:
            explanation = (
                f"Cross-modal verification PASSED: ECG HR ({hr_ecg:.1f} bpm) matches "
                f"Doppler HR ({hr_doppler:.1f} bpm) within {hr_delta:.1f} bpm, corr={corr:.2f}."
            )
        else:
            explanation = (
                f"Cross-modal verification WARNING: Discrepancy detected! ECG HR ({hr_ecg:.1f} bpm) "
                f"vs Doppler HR ({hr_doppler:.1f} bpm) (delta={hr_delta:.1f} bpm > {self.hr_tolerance_bpm} bpm). "
                f"Possible maternal signal contamination or transducer misalignment."
            )

        return TrustCheckResult(
            is_trusted=is_trusted,
            hr_ecg_bpm=hr_ecg,
            hr_doppler_bpm=hr_doppler,
            hr_delta_bpm=hr_delta,
            correlation_score=corr,
            trust_score=trust_score,
            explanation=explanation
        )

    def calibrate_opinions(
        self,
        ecg_opinion: Optional[Opinion],
        doppler_opinion: Optional[Opinion],
        trust_result: TrustCheckResult
    ) -> Tuple[Optional[Opinion], Optional[Opinion]]:
        """
        Adjust expert opinions based on cross-modal trust result.
        If trust fails, scale down belief & disbelief, inflating uncertainty u!
        """
        if trust_result.is_trusted:
            return ecg_opinion, doppler_opinion

        factor = max(0.2, trust_result.trust_score)

        def penalize(op: Optional[Opinion]) -> Optional[Opinion]:
            if op is None or op.uncertainty >= 0.99:
                return op
            new_b = op.belief * factor
            new_d = op.disbelief * factor
            new_u = 1.0 - (new_b + new_d)
            return Opinion(belief=new_b, disbelief=new_d, uncertainty=new_u, base_rate=op.base_rate)

        return penalize(ecg_opinion), penalize(doppler_opinion)
