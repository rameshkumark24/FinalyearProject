"""
SFR Framework — Model Calibration & Uncertainty Evaluation
===========================================================
Calibrates probabilities and opinions for clinical trustworthiness:
1. Expected Calibration Error (ECE)
2. Maximum Calibration Error (MCE)
3. Brier Score
4. Temperature Scaling Optimizer
"""

from typing import Tuple, Dict
import numpy as np
from scipy.optimize import minimize


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute ECE, MCE, and Brier Score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    # Brier Score = mean((prob - true)^2)
    brier_score = float(np.mean((y_probs - y_true) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_confidence = np.mean(y_probs[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            gap = abs(avg_confidence - avg_accuracy)

            ece += (bin_size / total_samples) * gap
            mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score
    }


class TemperatureScaler:
    """
    Post-hoc temperature scaling for calibrated clinical risks.
    Solves for optimal T > 0 on validation set log-loss.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_true: np.ndarray, uncalibrated_probs: np.ndarray):
        y_true = np.asarray(y_true).astype(float)
        # Avoid log(0)
        p = np.clip(uncalibrated_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))

        def loss_fn(t_arr):
            t = t_arr[0]
            scaled_logits = logits / t
            scaled_p = 1.0 / (1.0 + np.exp(-scaled_logits))
            # Binary cross entropy
            loss = -np.mean(y_true * np.log(scaled_p + 1e-9) + (1.0 - y_true) * np.log(1.0 - scaled_p + 1e-9))
            return loss

        res = minimize(loss_fn, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def transform(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        p = np.clip(uncalibrated_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))
