"""
Tests for Cross-Modal Trust Check and Function Expert
"""

import numpy as np
from src.trust_check.cross_modal_trust import CrossModalTrustChecker
from src.experts.function_expert import FunctionExpert
from src.fusion.subjective_logic import Opinion


def test_cross_modal_trust_checker_matching():
    checker = CrossModalTrustChecker(hr_tolerance_bpm=10.0)

    # 140 bpm -> interval 60/140 = 0.428s
    ecg_peaks = np.arange(1.0, 10.0, 0.428)
    # Doppler with small electromechanical delay (~50ms)
    doppler_peaks = ecg_peaks + 0.05

    res = checker.evaluate(ecg_peaks, doppler_peaks, duration_sec=11.0)
    assert res.is_trusted is True
    assert res.hr_delta_bpm < 2.0
    assert res.trust_score > 0.70


def test_cross_modal_trust_checker_maternal_mismatch():
    checker = CrossModalTrustChecker(hr_tolerance_bpm=10.0)

    # Fetal ECG: 145 bpm (~0.413s)
    ecg_peaks = np.arange(1.0, 10.0, 0.413)
    # Maternal Doppler leakage: 75 bpm (~0.800s)
    maternal_doppler_peaks = np.arange(1.0, 10.0, 0.800)

    res = checker.evaluate(ecg_peaks, maternal_doppler_peaks, duration_sec=11.0)
    assert res.is_trusted is False
    assert res.hr_delta_bpm > 50.0  # Significant discrepancy flagged!


def test_function_expert_evaluation():
    expert = FunctionExpert(sampling_rate=100.0)

    # Generate synthetic Doppler velocity envelope (~140 bpm = 2.33 Hz, period ~ 430 ms)
    # Systolic ejection pulse duration ~ 170 ms (17 samples at 100Hz)
    t = np.linspace(0, 10, 1000)
    envelope = np.zeros(1000)
    cycle_samples = int(100.0 / 2.33)  # 42 samples
    for start in range(0, 1000 - 18, cycle_samples):
        # Half sine systolic ejection peak lasting 18 samples (~180ms)
        envelope[start : start + 18] = np.sin(np.pi * np.linspace(0, 1, 18)) * 60.0

    op, params, z_scores = expert.evaluate(envelope, gestational_age_weeks=24.0)

    assert 130.0 <= params.mean_heart_rate_bpm <= 150.0
    assert abs(z_scores["z_heart_rate"]) < 2.0
    # Normal hemodynamics -> high disbelief of abnormality
    assert op.disbelief > op.belief
