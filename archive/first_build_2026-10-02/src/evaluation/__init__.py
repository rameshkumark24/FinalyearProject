"""
SFR Framework — Evaluation Module
"""

from .metrics import compute_screening_metrics
from .calibration import compute_calibration_metrics, TemperatureScaler
from .bootstrap import bootstrap_metric_ci, SimulatedCohortGenerator

__all__ = [
    "compute_screening_metrics",
    "compute_calibration_metrics",
    "TemperatureScaler",
    "bootstrap_metric_ci",
    "SimulatedCohortGenerator"
]
