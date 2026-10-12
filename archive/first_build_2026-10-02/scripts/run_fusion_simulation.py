"""
SFR Framework — Fusion Benchmark Simulation (Runs X1 & X3)
===========================================================
Benchmarks Subjective Logic OR against baseline fusion operators:
1. Subjective Logic OR (Proposed)
2. Max Rule
3. Noisy-OR
4. Dempster-Shafer
5. Mean Probability

Evaluates across all 7 clinical missingness patterns:
Pattern 1: (S, F, R) — Full screening triage
Pattern 2: (S, F, _) — Echo + Doppler (no ECG)
Pattern 3: (S, _, R) — Echo + ECG (no Doppler)
Pattern 4: (_, F, R) — Doppler + ECG (no Echo)
Pattern 5: (S, _, _) — Echo only
Pattern 6: (_, F, _) — Doppler only
Pattern 7: (_, _, R) — ECG only
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.fusion.subjective_logic import Opinion, fuse_opinions_or
from src.fusion.fusion_rules import FusionBenchmark
from src.evaluation.bootstrap import SimulatedCohortGenerator, bootstrap_metric_ci
from src.evaluation.metrics import compute_screening_metrics
from src.evaluation.calibration import compute_calibration_metrics


PATTERNS = {
    "All_Three_(S,F,R)": (True, True, True),
    "Echo+Doppler_(S,F,_)": (True, True, False),
    "Echo+ECG_(S,_,R)": (True, False, True),
    "Doppler+ECG_(_,F,R)": (False, True, True),
    "Echo_Only_(S,_,_)": (True, False, False),
    "Doppler_Only_(_,F,_)": (False, True, False),
    "ECG_Only_(_,_,R)": (False, False, True),
}


def run_benchmark(n_samples: int = 2000, seed: int = 42) -> Dict[str, Any]:
    print(f"[*] Simulating clinical screening cohort (N={n_samples}, prevalence=0.009)...")
    gen = SimulatedCohortGenerator(seed=seed)
    # Generate population cohort without missingness in generation, then mask per pattern
    cohort = gen.generate_cohort(
        n_samples=n_samples,
        prevalence=0.009,
        missing_rate_echo=0.0,
        missing_rate_doppler=0.0,
        missing_rate_ecg=0.0
    )

    y_true = np.array([p["label"] for p in cohort], dtype=int)
    results = {}

    for pattern_name, (use_s, use_f, use_r) in PATTERNS.items():
        print(f"\n--- Evaluating Pattern: {pattern_name} ---")
        rule_predictions: Dict[str, List[float]] = {
            "Subjective_Logic_OR": [],
            "Max_Rule": [],
            "Noisy_OR": [],
            "Dempster_Shafer": [],
            "Mean_Rule": []
        }

        for p in cohort:
            ops: List[Optional[Opinion]] = []
            probs: List[Optional[float]] = []

            # Structure
            if use_s and p["op_structure"] is not None:
                ops.append(p["op_structure"])
                probs.append(p["op_structure"].expected_probability)
            else:
                ops.append(None)
                probs.append(None)

            # Function
            if use_f and p["op_function"] is not None:
                ops.append(p["op_function"])
                probs.append(p["op_function"].expected_probability)
            else:
                ops.append(None)
                probs.append(None)

            # Rhythm
            if use_r and p["op_rhythm"] is not None:
                ops.append(p["op_rhythm"])
                probs.append(p["op_rhythm"].expected_probability)
            else:
                ops.append(None)
                probs.append(None)

            # 1. Subjective Logic OR
            sl_op = FusionBenchmark.subjective_logic(ops)
            rule_predictions["Subjective_Logic_OR"].append(sl_op.expected_probability)

            # 2. Max Rule
            rule_predictions["Max_Rule"].append(FusionBenchmark.max_rule(probs))

            # 3. Noisy-OR
            rule_predictions["Noisy_OR"].append(FusionBenchmark.noisy_or(probs))

            # 4. Dempster-Shafer
            ds_b, _, _ = FusionBenchmark.dempster_shafer([op for op in ops if op is not None])
            rule_predictions["Dempster_Shafer"].append(ds_b)

            # 5. Mean Rule
            rule_predictions["Mean_Rule"].append(FusionBenchmark.mean_rule(probs))

        pattern_summary = {}
        for rule, preds in rule_predictions.items():
            preds_arr = np.array(preds)
            metrics = compute_screening_metrics(y_true, preds_arr, threshold=0.30)
            calib = compute_calibration_metrics(y_true, preds_arr)
            pattern_summary[rule] = {
                "auroc": round(metrics["auroc"], 4),
                "sensitivity": round(metrics["sensitivity"], 4),
                "specificity": round(metrics["specificity"], 4),
                "sens_at_spec90": round(metrics["sens_at_spec90"], 4),
                "ece": round(calib["ece"], 4),
                "brier_score": round(calib["brier_score"], 4)
            }
            print(f"  [{rule}] AUROC: {metrics['auroc']:.3f} | Sens@Spec90: {metrics['sens_at_spec90']:.3f} | ECE: {calib['ece']:.3f}")

        results[pattern_name] = pattern_summary

    # Save to results
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "fusion_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Successfully saved benchmark results to {out_file}")
    return results


if __name__ == "__main__":
    run_benchmark()
