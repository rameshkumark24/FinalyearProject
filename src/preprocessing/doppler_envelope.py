"""
SFR Framework — Doppler Ultrasound Signal & Envelope Preprocessing
==================================================================
Processes Doppler ultrasound video/image data (e.g. from NInFEA):
1. Frame extraction & region of interest (ROI) crop
2. Speckle noise reduction & Otsu thresholding
3. Velocity envelope tracing (maximum velocity envelope)
4. Smoothing & normalization
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy import signal, ndimage


class DopplerEnvelopeExtractor:
    """
    Extracts 1D spectral Doppler maximum velocity envelope from Doppler sonogram images or time-series frames.
    """

    def __init__(
        self,
        blur_kernel_size: int = 5,
        morph_size: int = 3,
        smooth_window: int = 11
    ):
        self.blur_kernel_size = blur_kernel_size
        self.morph_size = morph_size
        self.smooth_window = smooth_window

    def extract_from_sonogram_image(
        self,
        sonogram_gray: np.ndarray,
        baseline_row: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract velocity envelope from a 2D spectral Doppler spectrogram image.
        Rows: frequency/velocity, Columns: time.
        
        Steps:
        1. Smooth image with median filter to reduce acoustic speckle.
        2. Otsu thresholding to segment blood flow signals from dark background.
        3. For each time column, find the topmost (highest velocity) illuminated pixel.
        4. Subtract baseline to obtain signed/positive velocity.
        """
        if sonogram_gray.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {sonogram_gray.shape}")

        height, width = sonogram_gray.shape
        if baseline_row is None:
            baseline_row = int(height * 0.75)  # Common clinical default (zero velocity line)

        # 1. Speckle reduction
        denoised = ndimage.median_filter(sonogram_gray, size=self.blur_kernel_size)

        # 2. Otsu thresholding
        threshold = self._compute_otsu_threshold(denoised)
        binary_mask = (denoised > threshold).astype(np.uint8)

        # 3. Morphological opening to remove isolated spurious pixels
        struct = ndimage.generate_binary_structure(2, 1)
        cleaned_mask = ndimage.binary_opening(binary_mask, structure=struct, iterations=self.morph_size)

        # 4. Envelope detection (highest non-zero pixel index above baseline for each time slice)
        envelope = np.zeros(width, dtype=np.float32)
        for col in range(width):
            col_slice = cleaned_mask[:baseline_row, col]
            active_pixels = np.where(col_slice > 0)[0]
            if len(active_pixels) > 0:
                # Topmost pixel has smallest row index
                top_row = active_pixels[0]
                velocity_px = baseline_row - top_row
                envelope[col] = float(velocity_px)
            else:
                envelope[col] = 0.0

        # 5. Savitzky-Golay or moving average smoothing
        if len(envelope) > self.smooth_window:
            envelope_smoothed = signal.savgol_filter(
                envelope,
                window_length=self.smooth_window if self.smooth_window % 2 != 0 else self.smooth_window + 1,
                polyorder=2
            )
            envelope = np.maximum(0.0, envelope_smoothed)

        return envelope

    @staticmethod
    def _compute_otsu_threshold(image: np.ndarray) -> float:
        """Compute Otsu's optimal binarization threshold."""
        hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
        total_pixels = image.size
        current_max = 0.0
        threshold = 128.0

        weight_bg = 0
        sum_bg = 0
        total_sum = np.sum(np.arange(256) * hist)

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0:
                break

            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (total_sum - sum_bg) / weight_fg

            # Inter-class variance
            var_between = float(weight_bg) * float(weight_fg) * ((mean_bg - mean_fg) ** 2)
            if var_between > current_max:
                current_max = var_between
                threshold = float(t)

        return threshold
