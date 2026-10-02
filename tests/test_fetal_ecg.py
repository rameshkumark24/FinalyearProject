"""
Tests for Fetal ECG Extraction and Data Loading
"""

import numpy as np
from src.utils.data_loading import PhysioNetDataLoader
from src.preprocessing.fetal_ecg_extraction import FetalECGExtractor


def test_synthetic_data_loading_and_filtering():
    loader = PhysioNetDataLoader()
    signals, ref_peaks, fs = loader.load_cinc2013_record("synthetic_test")

    assert signals.shape[0] == 4
    assert signals.shape[1] == 60 * fs
    assert ref_peaks is not None
    assert len(ref_peaks) > 50

    extractor = FetalECGExtractor(sampling_rate=fs)
    filtered = extractor.filter_signal(signals)
    assert filtered.shape == signals.shape


def test_maternal_cancellation_and_cinc_eval():
    loader = PhysioNetDataLoader()
    signals, ref_peaks, fs = loader.load_cinc2013_record("synthetic_test")

    extractor = FetalECGExtractor(sampling_rate=fs)
    maternal_peaks = extractor.detect_maternal_qrs(signals[0])
    assert len(maternal_peaks) > 40

    # Template subtraction on lead 1
    cleaned = extractor.cancel_maternal_template_subtraction(signals[0], maternal_peaks)
    assert len(cleaned) == len(signals[0])

    # Evaluate F1
    eval_metrics = extractor.evaluate_cinc2013(
        predicted_peaks_samples=ref_peaks,  # Perfect match
        reference_peaks_samples=ref_peaks,
        tolerance_ms=50.0
    )
    assert eval_metrics["f1_score"] == 1.0
    assert eval_metrics["sensitivity"] == 1.0
