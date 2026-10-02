"""
SFR Framework — Fetal ECG Extraction & Maternal Cancellation
=============================================================
Team Member: Niranjana
Implements non-invasive fetal ECG extraction from abdominal leads:
1. Pre-filtering: 50Hz notch, 1.0 - 100.0 Hz bandpass
2. Maternal QRS detection (Pan-Tompkins on chest/abdominal channel)
3. Maternal ECG removal:
   - Method 1: Adaptive Template Subtraction (TS)
   - Method 2: FastICA (Blind Source Separation)
   - Method 3: Combined TS + PCA residual filtering
4. Fetal QRS peak detection on cleaned residual signal
5. Benchmark metric: F1 score within +/- 50ms tolerance (CinC 2013 standard)
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA, PCA


class FetalECGExtractor:
    """
    Extracts fetal ECG from multi-channel abdominal recordings (e.g. CinC 2013, NIFEADB, NInFEA).
    """

    def __init__(
        self,
        sampling_rate: int = 500,
        notch_freq: float = 50.0,
        bandpass_low: float = 1.0,
        bandpass_high: float = 100.0
    ):
        self.fs = sampling_rate
        self.notch_freq = notch_freq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high

    def filter_signal(self, ecg_channels: np.ndarray) -> np.ndarray:
        """
        Apply notch filter (mains hum) and 4th-order Butterworth bandpass filter.
        ecg_channels: shape (n_channels, n_samples) or (n_samples,)
        """
        is_1d = ecg_channels.ndim == 1
        if is_1d:
            ecg_channels = ecg_channels[np.newaxis, :]

        n_channels, n_samples = ecg_channels.shape
        filtered = np.zeros_like(ecg_channels)

        # 1. Notch filter
        nyq = 0.5 * self.fs
        notch_w0 = self.notch_freq / nyq
        if notch_w0 < 1.0:
            b_notch, a_notch = signal.iirnotch(notch_w0, Q=30.0)
        else:
            b_notch, a_notch = None, None

        # 2. Bandpass filter
        bp_low = max(0.01, self.bandpass_low / nyq)
        bp_high = min(0.99, self.bandpass_high / nyq)
        b_bp, a_bp = signal.butter(4, [bp_low, bp_high], btype="bandpass")

        for ch in range(n_channels):
            sig = ecg_channels[ch]
            if b_notch is not None:
                sig = signal.filtfilt(b_notch, a_notch, sig)
            sig = signal.filtfilt(b_bp, a_bp, sig)
            filtered[ch] = sig

        return filtered[0] if is_1d else filtered

    def detect_maternal_qrs(
        self,
        maternal_channel: np.ndarray,
        min_distance_sec: float = 0.50  # Maternal HR rarely > 120 bpm at rest (min distance ~ 500ms)
    ) -> np.ndarray:
        """
        Detect prominent maternal R-peaks using derivative and squaring integration (Pan-Tompkins style).
        """
        # Derivative
        diff_sig = np.diff(maternal_channel)
        diff_sig = np.pad(diff_sig, (0, 1), mode="edge")
        squared = diff_sig ** 2

        # Moving average integration window (~80ms)
        win_size = int(0.08 * self.fs)
        kernel = np.ones(win_size) / win_size
        integrated = np.convolve(squared, kernel, mode="same")

        # Peak detection
        min_dist = int(min_distance_sec * self.fs)
        threshold = 0.35 * np.max(integrated)
        peaks, _ = signal.find_peaks(integrated, distance=min_dist, height=threshold)
        return peaks

    def cancel_maternal_template_subtraction(
        self,
        abdominal_channel: np.ndarray,
        maternal_r_peaks: np.ndarray,
        window_ms: int = 120
    ) -> np.ndarray:
        """
        Method 1: Adaptive template subtraction.
        Constructs an ensemble average maternal P-QRS-T complex and subtracts it
        at each maternal R-peak location.
        """
        n_samples = len(abdominal_channel)
        cleaned = abdominal_channel.copy()
        half_win = int((window_ms / 1000.0) * self.fs // 2)

        # Collect maternal templates
        templates = []
        for r in maternal_r_peaks:
            start = r - half_win
            end = r + half_win
            if start >= 0 and end <= n_samples:
                templates.append(abdominal_channel[start:end])

        if len(templates) < 3:
            return cleaned

        mean_template = np.median(templates, axis=0)

        # Adaptive scaling and subtraction
        for r in maternal_r_peaks:
            start = r - half_win
            end = r + half_win
            if start >= 0 and end <= n_samples:
                seg = abdominal_channel[start:end]
                # Scale factor via least-squares
                dot_prod = np.dot(seg, mean_template)
                norm_sq = np.dot(mean_template, mean_template)
                scale = dot_prod / (norm_sq + 1e-8) if norm_sq > 0 else 1.0
                scale = np.clip(scale, 0.5, 2.0)
                cleaned[start:end] -= scale * mean_template

        return cleaned

    def cancel_maternal_fastica(
        self,
        multi_lead_abdominal: np.ndarray,
        n_components: int = 4
    ) -> np.ndarray:
        """
        Method 2: FastICA decomposition across multiple abdominal leads.
        multi_lead_abdominal: shape (n_leads, n_samples)
        Returns the independent component that best isolates fetal cardiac activity.
        """
        n_leads, n_samples = multi_lead_abdominal.shape
        n_comp = min(n_components, n_leads)
        
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        # FastICA expects (n_samples, n_features)
        sources = ica.fit_transform(multi_lead_abdominal.T).T

        # Fetal component typically has higher fundamental frequency (~130-160 bpm) than maternal (~70-90 bpm)
        # Select component with highest high-frequency energy ratio or kurtosis
        best_comp = sources[0]
        max_kurt = -1e9
        for c in range(n_comp):
            comp = sources[c]
            # Kurtosis: QRS complexes are spiky outliers -> high kurtosis
            kurt = np.mean(((comp - np.mean(comp)) / (np.std(comp) + 1e-8)) ** 4) - 3.0
            if kurt > max_kurt:
                max_kurt = kurt
                best_comp = comp

        return best_comp

    def detect_fetal_qrs(
        self,
        cleaned_signal: np.ndarray,
        min_fetal_distance_sec: float = 0.28  # Fetal max HR ~ 210 bpm
    ) -> np.ndarray:
        """
        Detect fetal R-peaks in the cleaned residual signal.
        """
        diff_sig = np.diff(cleaned_signal)
        diff_sig = np.pad(diff_sig, (0, 1), mode="edge")
        squared = diff_sig ** 2

        win_size = int(0.04 * self.fs)  # ~40ms window for narrow fetal QRS
        kernel = np.ones(win_size) / win_size
        integrated = np.convolve(squared, kernel, mode="same")

        min_dist = int(min_fetal_distance_sec * self.fs)
        med_val = np.median(integrated)
        std_val = np.std(integrated)
        threshold = med_val + 1.8 * std_val

        peaks, _ = signal.find_peaks(integrated, distance=min_dist, height=threshold)
        return peaks

    def evaluate_cinc2013(
        self,
        predicted_peaks_samples: np.ndarray,
        reference_peaks_samples: np.ndarray,
        tolerance_ms: float = 50.0
    ) -> Dict[str, float]:
        """
        CinC 2013 Challenge Evaluation Standard:
        Match predicted fetal R-peaks to reference within +/- 50ms window.
        Returns: Sensitivity (Se), Positive Predictive Value (PPV), and F1-Score.
        """
        tol_samples = int((tolerance_ms / 1000.0) * self.fs)

        if len(reference_peaks_samples) == 0:
            return {"sensitivity": 0.0, "ppv": 0.0, "f1_score": 0.0}
        if len(predicted_peaks_samples) == 0:
            return {"sensitivity": 0.0, "ppv": 0.0, "f1_score": 0.0}

        matched_ref = set()
        matched_pred = set()

        for i, pred in enumerate(predicted_peaks_samples):
            diffs = np.abs(reference_peaks_samples - pred)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tol_samples and min_idx not in matched_ref:
                matched_ref.add(min_idx)
                matched_pred.add(i)

        tp = len(matched_ref)
        fn = len(reference_peaks_samples) - tp
        fp = len(predicted_peaks_samples) - len(matched_pred)

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * se * ppv) / (se + ppv) if (se + ppv) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "sensitivity": float(se),
            "ppv": float(ppv),
            "f1_score": float(f1)
        }
