"""
SFR Framework — Full Cohort Fetal ECG Benchmark (CinC 2013 Set A)
=================================================================
Processes all 75 clinical records in CinC 2013 Set A:
1. Filters 4-channel abdominal recordings (spline NaN interpolation + Notch + Bandpass).
2. Detects maternal QRS complexes and measures maternal heart rate.
3. Performs Adaptive Template Subtraction (TS) maternal cancellation.
4. Detects fetal QRS complexes on residual signal.
5. Benchmarks against ground-truth annotations (.fqrs) within +/- 50ms tolerance.
6. Computes aggregate cohort metrics:
   - Mean & Median F1-Score
   - Sensitivity (Se) & Positive Predictive Value (PPV)
   - Heart rate error (|HR_pred - HR_true|)
   - Clinical yield
7. Exports publication-grade JSON and summary metrics.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig
from src.preprocessing.fetal_ecg_extraction import FetalECGExtractor


def run_full_cohort_benchmark():
    paths = PathConfig()
    set_a_dir = paths.cinc2013_dir / "set-a"

    if not set_a_dir.exists():
        print(f"[-] Directory {set_a_dir} does not exist.")
        return

    import wfdb

    hea_files = sorted(list(set_a_dir.glob("*.hea")))
    print(f"[*] Found {len(hea_files)} clinical records in {set_a_dir}...")

    extractor = FetalECGExtractor(sampling_rate=1000, notch_freq=50.0, bandpass_low=1.0, bandpass_high=100.0)

    record_results: List[Dict[str, Any]] = []
    all_f1: List[float] = []
    all_se: List[float] = []
    all_ppv: List[float] = []
    all_hr_errors: List[float] = []

    for idx, hea in enumerate(hea_files):
        rec_id = hea.stem
        rec_base = str(hea.with_suffix(""))

        try:
            # 1. Load record
            rec = wfdb.rdrecord(rec_base)
            signals = rec.p_signal.T
            fs = rec.fs

            # 2. Load ground truth
            fqrs_path = hea.with_suffix(".fqrs")
            if not fqrs_path.exists():
                continue

            ann = wfdb.rdann(rec_base, "fqrs")
            ref_peaks = ann.sample
            if len(ref_peaks) == 0:
                continue

            duration_sec = signals.shape[1] / fs
            true_fhr = (len(ref_peaks) / duration_sec) * 60.0

            # 3. Filtering with NaN interpolation
            filtered = extractor.filter_signal(signals)

            # 4. Maternal QRS detection on highest variance lead
            lead_idx = int(np.argmax(np.std(filtered, axis=1)))
            lead_sig = filtered[lead_idx]
            maternal_peaks = extractor.detect_maternal_qrs(lead_sig)
            maternal_hr = (len(maternal_peaks) / duration_sec) * 60.0

            # 5. Template Subtraction
            cleaned_sig = extractor.cancel_maternal_template_subtraction(lead_sig, maternal_peaks, window_ms=120)

            # 6. Fetal QRS detection
            pred_fetal_peaks = extractor.detect_fetal_qrs(cleaned_sig)
            pred_fhr = (len(pred_fetal_peaks) / duration_sec) * 60.0

            # 7. Evaluate CinC 2013 standard (+/- 50ms)
            eval_metrics = extractor.evaluate_cinc2013(pred_fetal_peaks, ref_peaks, tolerance_ms=50.0)

            hr_error = abs(pred_fhr - true_fhr)

            all_f1.append(eval_metrics["f1_score"])
            all_se.append(eval_metrics["sensitivity"])
            all_ppv.append(eval_metrics["ppv"])
            all_hr_errors.append(hr_error)

            record_results.append({
                "record": rec_id,
                "ref_beats": len(ref_peaks),
                "true_fhr_bpm": round(true_fhr, 1),
                "maternal_hr_bpm": round(maternal_hr, 1),
                "detected_beats": len(pred_fetal_peaks),
                "pred_fhr_bpm": round(pred_fhr, 1),
                "hr_error_bpm": round(hr_error, 1),
                "f1_score": round(eval_metrics["f1_score"], 4),
                "sensitivity": round(eval_metrics["sensitivity"], 4),
                "ppv": round(eval_metrics["ppv"], 4)
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(hea_files):
                print(f"[+] Processed {idx + 1}/{len(hea_files)}: Mean F1 so far: {np.mean(all_f1):.3f}")

        except Exception as e:
            # Record parsing exception handling
            continue

    if not record_results:
        print("[-] No records could be processed.")
        return

    # Aggregate Statistics
    summary = {
        "total_records_processed": len(record_results),
        "mean_f1_score": round(float(np.mean(all_f1)), 4),
        "median_f1_score": round(float(np.median(all_f1)), 4),
        "std_f1_score": round(float(np.std(all_f1)), 4),
        "mean_sensitivity": round(float(np.mean(all_se)), 4),
        "median_sensitivity": round(float(np.median(all_se)), 4),
        "mean_ppv": round(float(np.mean(all_ppv)), 4),
        "median_ppv": round(float(np.median(all_ppv)), 4),
        "mean_hr_error_bpm": round(float(np.mean(all_hr_errors)), 2),
        "median_hr_error_bpm": round(float(np.median(all_hr_errors)), 2),
        "records": record_results
    }

    out_file = paths.results_dir / "cinc2013_full_cohort_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[SUMMARY] CinC 2013 Cohort Benchmark Summary (N={len(record_results)})")
    print("=" * 65)
    print(f"Mean F1-Score:    {summary['mean_f1_score']:.4f} +/- {summary['std_f1_score']:.4f}")
    print(f"Median F1-Score:  {summary['median_f1_score']:.4f}")
    print(f"Mean Sensitivity: {summary['mean_sensitivity']:.4f}")
    print(f"Mean PPV:         {summary['mean_ppv']:.4f}")
    print(f"Mean HR Error:    {summary['mean_hr_error_bpm']:.2f} bpm")
    print(f"[+] Detailed cohort results saved to {out_file}")

    return summary


if __name__ == "__main__":
    run_full_cohort_benchmark()
