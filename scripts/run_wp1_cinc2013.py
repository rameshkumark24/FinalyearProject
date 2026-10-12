"""WP1 - fetal ECG extraction on PhysioNet/CinC Challenge 2013, set A (runs R1, R2).

The pipeline settings were tuned on the 25 development recordings only. This script runs the
frozen pipeline, and four ablations of it, on all 75 recordings and reports the development
and test splits separately.

    python scripts/run_wp1_cinc2013.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.cinc2013 import ALL_RECORDS, load_record, record_files  # noqa: E402
from sfr.ecg.extract import extract_fetal_ecg  # noqa: E402
from sfr.ecg.metrics import beat_scores, follows_mother, heart_rate_bpm  # noqa: E402
from sfr.eval.stats import bootstrap_ci, clopper_pearson  # noqa: E402
from sfr.paths import SPLITS  # noqa: E402
from sfr.provenance import Run  # noqa: E402

CONFIGS = {
    "full": dict(cancel=True, separate=True, track=True),
    "no_tracking": dict(cancel=True, separate=True, track=False),
    "no_ica": dict(cancel=True, separate=False, track=True),
    "no_cancellation": dict(cancel=False, separate=True, track=True),
    "channel_only": dict(cancel=False, separate=False, track=False),
}
FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)
TOLERANCE_MS = 50.0


def summarise(rows: list[dict]) -> dict:
    f1 = np.array([r["f1"] for r in rows])
    hr_error = np.array([r["hr_abs_error_bpm"] for r in rows])
    n_lock = int(sum(r["follows_mother"] for r in rows))
    mean = bootstrap_ci(f1, np.mean)
    median = bootstrap_ci(f1, np.median)
    lock_low, lock_high = clopper_pearson(n_lock, len(rows))
    return {
        "n_recordings": len(rows),
        "f1_mean": mean[0], "f1_mean_ci": [mean[1], mean[2]],
        "f1_median": median[0], "f1_median_ci": [median[1], median[2]],
        "f1_min": float(f1.min()),
        "share_f1_at_least_0.9": float(np.mean(f1 >= 0.9)),
        "sensitivity_mean": float(np.mean([r["sensitivity"] for r in rows])),
        "ppv_mean": float(np.mean([r["ppv"] for r in rows])),
        "hr_abs_error_median_bpm": float(np.nanmedian(hr_error)),
        "hr_abs_error_mean_bpm": float(np.nanmean(hr_error)),
        "follows_mother_n": n_lock,
        "follows_mother_rate": n_lock / len(rows),
        "follows_mother_rate_ci": [lock_low, lock_high],
        "flagged_failed_n": int(sum(not r["ok"] for r in rows)),
    }


def main() -> None:
    split = json.loads((SPLITS / "cinc2013.json").read_text(encoding="utf-8"))
    development, test = set(split["development"]), set(split["test"])
    doubted = set(split["reference_doubted_in_literature"]["records"])
    assert development | test == set(ALL_RECORDS) and not development & test

    inputs = [p for name in ALL_RECORDS for p in record_files(name)]
    run = Run(
        "wp1_cinc2013",
        description="Fetal QRS detection on CinC 2013 set A: frozen pipeline and four ablations",
        script="scripts/run_wp1_cinc2013.py",
        config={"frozen": FROZEN, "configs": CONFIGS, "tolerance_ms": TOLERANCE_MS,
                "split_file": "splits/cinc2013.json"},
        inputs={"cinc2013_set_a": inputs},
        n_real=len(ALL_RECORDS),
        seed=FROZEN["seed"],
    )

    rows: list[dict] = []
    started = time.time()
    for name in ALL_RECORDS:
        x, fs, reference = load_record(name)
        true_bpm = heart_rate_bpm(reference, fs)
        for config_name, switches in CONFIGS.items():
            extraction = extract_fetal_ecg(x, fs, **FROZEN, **switches)
            scores = beat_scores(extraction.fetal_peaks, reference, extraction.fs, TOLERANCE_MS)
            predicted_bpm = heart_rate_bpm(extraction.fetal_peaks, extraction.fs)
            maternal_bpm = extraction.quality["maternal_bpm"]
            rows.append({
                "record": name,
                "split": "development" if name in development else "test",
                "reference_doubted": name in doubted,
                "config": config_name,
                "n_reference": int(reference.size),
                "n_detected": int(extraction.fetal_peaks.size),
                **scores,
                "true_bpm": true_bpm, "predicted_bpm": predicted_bpm, "maternal_bpm": maternal_bpm,
                "hr_abs_error_bpm": abs(predicted_bpm - true_bpm) if np.isfinite(predicted_bpm) else float("nan"),
                "follows_mother": follows_mother(predicted_bpm, true_bpm, maternal_bpm),
                "ok": extraction.ok, "quality_score": extraction.quality["score"],
                "source": extraction.source, "maternal_source": extraction.quality["maternal_source"],
            })
        print(f"{name} done ({time.time() - started:.0f} s)", flush=True)

    run.write_csv("per_record.csv", rows)

    metrics: dict = {"tolerance_ms": TOLERANCE_MS, "by_config": {}}
    for config_name in CONFIGS:
        selected = [r for r in rows if r["config"] == config_name]
        subsets = {
            "development": [r for r in selected if r["split"] == "development"],
            "test": [r for r in selected if r["split"] == "test"],
            "test_without_doubted_references": [r for r in selected if r["split"] == "test" and not r["reference_doubted"]],
            "all": selected,
            "all_without_doubted_references": [r for r in selected if not r["reference_doubted"]],
        }
        metrics["by_config"][config_name] = {key: summarise(value) for key, value in subsets.items()}

    full_test = metrics["by_config"]["full"]["test"]
    metrics["gate_G1"] = {
        "rule": "median F1 >= 0.90 and follows-mother rate <= 0.05 on the held-out test recordings",
        "median_f1_test": full_test["f1_median"],
        "follows_mother_rate_test": full_test["follows_mother_rate"],
        "passed": bool(full_test["f1_median"] >= 0.90 and full_test["follows_mother_rate"] <= 0.05),
    }
    run.finish(metrics)

    print("\nConfig            split         n   mean F1 [95% CI]        median F1   F1>=0.9  follows mother")
    for config_name in CONFIGS:
        for subset in ("development", "test", "test_without_doubted_references"):
            m = metrics["by_config"][config_name][subset]
            print(f"{config_name:17s} {subset[:12]:12s} {m['n_recordings']:3d}   {m['f1_mean']:.3f} "
                  f"[{m['f1_mean_ci'][0]:.3f}, {m['f1_mean_ci'][1]:.3f}]   {m['f1_median']:.3f}       "
                  f"{m['share_f1_at_least_0.9']:.2f}     {m['follows_mother_n']}/{m['n_recordings']}")
    print("\nGate G1:", metrics["gate_G1"])


if __name__ == "__main__":
    main()
