"""
SFR Framework — Non-Parametric Bootstrap & Cohort Simulation
=============================================================
Enforces:
1. Non-parametric bootstrap (1,000 iterations) for 95% Confidence Intervals.
2. Simulated cohort generator for Run X3 (evaluating Subjective Logic OR vs baselines
   across missing modality combinations without violating paired patient constraints).
"""

from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np
from ..fusion.subjective_logic import Opinion
from ..fusion.fusion_rules import FusionBenchmark
from .metrics import compute_screening_metrics


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute point estimate and 95% bootstrap percentile confidence intervals.
    Returns: {metric_name: (point_est, ci_lower, ci_upper)}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_metrics = compute_screening_metrics(y_true, y_pred_probs)

    boot_results: Dict[str, List[float]] = {k: [] for k in point_metrics}

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Check if resampled slice has both classes
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_screening_metrics(y_true[idx], y_pred_probs[idx])
        for k, v in m.items():
            boot_results[k].append(v)

    summary = {}
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    for k in point_metrics:
        vals = np.array(boot_results[k])
        if len(vals) > 0:
            low = float(np.percentile(vals, lower_pct))
            high = float(np.percentile(vals, upper_pct))
        else:
            low, high = point_metrics[k], point_metrics[k]
        summary[k] = (point_metrics[k], low, high)

    return summary


class SimulatedCohortGenerator:
    """
    Simulates multi-modal screening cohorts (Run X3 in plan) from marginal expert distributions.
    Allows testing fusion behavior across all 7 missingness patterns:
    (S,F,R), (S,F,_), (S,_,R), (_,F,R), (S,_,_), (_,F,_), (_,_,R)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_cohort(
        self,
        n_samples: int = 2000,
        prevalence: float = 0.009,  # Saxena ~9 per 1,000
        missing_rate_echo: float = 0.15,
        missing_rate_doppler: float = 0.25,
        missing_rate_ecg: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        Generate cohort with ground truth diagnosis and individual expert outputs.
        """
        cohort = []
        n_abnormal = int(n_samples * prevalence)
        labels = np.zeros(n_samples, dtype=int)
        labels[:n_abnormal] = 1
        self.rng.shuffle(labels)

        for i in range(n_samples):
            is_abnormal = bool(labels[i] == 1)

            # Determine clinical subtype if abnormal
            # Subtype 1: Isolated Structural CHD (60%)
            # Subtype 2: Isolated Arrhythmia/Conduction (25%)
            # Subtype 3: Hemodynamic/Functional dysfunction (15%)
            if is_abnormal:
                sub_r = self.rng.random()
                if sub_r < 0.60:
                    subtype = "structural_chd"
                elif sub_r < 0.85:
                    subtype = "arrhythmia"
                else:
                    subtype = "functional_decompensation"
            else:
                subtype = "healthy"

            # Structure Expert simulation (Echo)
            if self.rng.random() < missing_rate_echo:
                op_structure = None
            else:
                if subtype == "structural_chd":
                    p = self.rng.beta(7, 2)  # High risk
                    c = self.rng.uniform(0.75, 0.95)
                else:
                    p = self.rng.beta(1, 15) # Low risk
                    c = self.rng.uniform(0.70, 0.90)
                op_structure = Opinion.from_probability_and_confidence(p, c)

            # Function Expert simulation (Doppler)
            if self.rng.random() < missing_rate_doppler:
                op_function = None
            else:
                if subtype in ("functional_decompensation", "structural_chd"):
                    p = self.rng.beta(6, 3)
                    c = self.rng.uniform(0.70, 0.90)
                else:
                    p = self.rng.beta(1, 18)
                    c = self.rng.uniform(0.75, 0.95)
                op_function = Opinion.from_probability_and_confidence(p, c)

            # Rhythm Expert simulation (Fetal ECG)
            if self.rng.random() < missing_rate_ecg:
                op_rhythm = None
            else:
                if subtype == "arrhythmia":
                    p = self.rng.beta(8, 2)  # High risk
                    c = self.rng.uniform(0.80, 0.95)
                else:
                    p = self.rng.beta(1, 20) # Normal sinus rhythm
                    c = self.rng.uniform(0.70, 0.90)
                op_rhythm = Opinion.from_probability_and_confidence(p, c)

            cohort.append({
                "patient_id": f"SIM_{i+1:05d}",
                "label": int(is_abnormal),
                "subtype": subtype,
                "op_structure": op_structure,
                "op_function": op_function,
                "op_rhythm": op_rhythm
            })

        return cohort
