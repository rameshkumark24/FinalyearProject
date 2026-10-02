"""
SFR Framework — Real Fetal ECG Extraction & Benchmark (Runs R1–R2)
==================================================================
Runs maternal cancellation on REAL clinical PhysioNet Challenge 2013 records:
1. Ingests record 'set-a/a01' (4 abdominal channels, 60 seconds, 1000 Hz).
2. Filters signal (Notch 50Hz + Bandpass 1-100Hz).
3. Detects maternal QRS complexes.
4. Performs Adaptive Template Subtraction (TS) and FastICA.
5. Detects fetal R-peaks on residual signal.
6. Evaluates against ground truth reference .qrs annotations (+/- 50ms tolerance).
7. Computes and saves CinC 2013 F1-score, Sensitivity, and PPV.
"""

import sys
import json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig
from src.preprocessing.fetal_ecg_extraction import FetalECGExtractor

def run_real_extraction():
    print("=" * 65)
    print("[RUN R1-R2] Real Fetal ECG Extraction & CinC 2013 Evaluation")
    print("=" * 65)

    paths = PathConfig()
    record_path = paths.cinc2013_dir / "set-a" / "a01"

    if not record_path.with_suffix(".hea").exists():
        print(f"[-] Record {record_path} not found. Please run download script first.")
        return

    import wfdb

    print(f"[*] Loading real clinical record: {record_path}...")
    rec = wfdb.rdrecord(str(record_path))
    signals = rec.p_signal.T  # (4, 60000)
    fs = rec.fs
    print(f"[+] Loaded signals: {signals.shape[0]} channels, {signals.shape[1]} samples at {fs} Hz")

    # Load ground truth annotations (.fqrs)
    try:
        ann = wfdb.rdann(str(record_path), "fqrs")
    except Exception:
        ann = wfdb.rdann(str(record_path), "qrs")
    ref_peaks = ann.sample
    print(f"[+] Loaded ground truth fetal QRS annotations: {len(ref_peaks)} beats")
    # True fetal heart rate
    true_fhr = (len(ref_peaks) / (signals.shape[1] / fs)) * 60.0
    print(f"    -> True fetal heart rate: {true_fhr:.1f} bpm")

    extractor = FetalECGExtractor(sampling_rate=fs, notch_freq=50.0, bandpass_low=1.0, bandpass_high=100.0)

    # 1. Pre-filtering
    print("\n[*] Step 1: Bandpass and notch filtering...")
    filtered_signals = extractor.filter_signal(signals)

    # 2. Maternal QRS detection on lead with largest amplitude
    print("[*] Step 2: Maternal QRS detection...")
    lead_stds = np.std(filtered_signals, axis=1)
    lead_idx = int(np.argmax(lead_stds))
    maternal_lead = filtered_signals[lead_idx]

    maternal_peaks = extractor.detect_maternal_qrs(maternal_lead)
    maternal_hr = (len(maternal_peaks) / (signals.shape[1] / fs)) * 60.0
    print(f"[+] Detected {len(maternal_peaks)} maternal beats (Maternal HR: {maternal_hr:.1f} bpm)")

    # 3. Method 1: Adaptive Template Subtraction
    print("\n[*] Step 3: Maternal cancellation via Adaptive Template Subtraction (TS)...")
    cleaned_lead = extractor.cancel_maternal_template_subtraction(maternal_lead, maternal_peaks, window_ms=120)

    # 4. Fetal QRS detection on cleaned residual
    print("[*] Step 4: Detecting fetal R-peaks on residual signal...")
    detected_fetal_peaks = extractor.detect_fetal_qrs(cleaned_lead)
    pred_fhr = (len(detected_fetal_peaks) / (signals.shape[1] / fs)) * 60.0
    print(f"[+] Detected {len(detected_fetal_peaks)} fetal candidate peaks (Detected HR: {pred_fhr:.1f} bpm)")

    # 5. Method 2: FastICA decomposition across all 4 leads
    print("\n[*] Step 5: Maternal cancellation via FastICA across 4 channels...")
    try:
        fetal_ica_comp = extractor.cancel_maternal_fastica(filtered_signals, n_components=4)
        detected_ica_peaks = extractor.detect_fetal_qrs(fetal_ica_comp)
        ica_metrics = extractor.evaluate_cinc2013(detected_ica_peaks, ref_peaks, tolerance_ms=50.0)
        print(f"    -> FastICA F1-Score: {ica_metrics['f1_score']:.4f} (Se: {ica_metrics['sensitivity']:.4f}, PPV: {ica_metrics['ppv']:.4f})")
    except Exception as e:
        print(f"    [-] FastICA note: {e}")
        ica_metrics = None

    # 6. Evaluation of Template Subtraction against CinC 2013 standard (+/- 50ms)
    print("\n[*] Step 6: CinC 2013 Benchmark Evaluation (+/- 50ms tolerance)...")
    ts_metrics = extractor.evaluate_cinc2013(detected_fetal_peaks, ref_peaks, tolerance_ms=50.0)
    print(f"    -> Template Subtraction F1-Score: {ts_metrics['f1_score']:.4f}")
    print(f"    -> Sensitivity (Se): {ts_metrics['sensitivity']:.4f}")
    print(f"    -> Positive Predictive Value (PPV): {ts_metrics['ppv']:.4f}")
    print(f"    -> True Positives: {ts_metrics['true_positives']} / {len(ref_peaks)}")

    results = {
        "record_id": "set-a/a01",
        "ground_truth_beats": len(ref_peaks),
        "true_fetal_hr_bpm": round(true_fhr, 2),
        "maternal_hr_bpm": round(maternal_hr, 2),
        "template_subtraction_metrics": ts_metrics,
        "fastica_metrics": ica_metrics
    }

    out_file = paths.results_dir / "r1_r2_real_cinc2013_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Successfully saved real clinical extraction metrics to {out_file}")
    return results

if __name__ == "__main__":
    run_real_extraction()
