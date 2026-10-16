"""WP8 - alarm levels against sensitivity: what each false-referral ceiling buys.

Added on 16 October 2026 in answer to the review comment that a 20% ceiling is high; a check
made after seeing the results. Nothing is fitted: the conformal p-values already computed by
the expert runs are compared with a grid of alarm levels.

For every level of every expert: its sensitivity (structure on DELTA, rhythm on NIFEADB; the
function expert has no abnormal cases) and its alarm rate on healthy cases. For every
combination of levels: the false-referral ceiling (sum of the levels), the range of the
measured healthy alarm rates for any dependence (as in wp7), and the two sensitivities.

The rhythm expert has 14 healthy calibration recordings, so its smallest possible p-value is
1/15: at a lower level it can never alarm. In the leave-one-out evaluation of the 14 healthy
recordings each is ranked among the other 13 (smallest p 1/14), so at the level 1/15 their
measured alarm rate is 0 by construction. Both facts are stated in the output.

Choosing a level from this table and reporting performance at that level on the same data
would be optimistic; the table shows the trade-off, it does not validate an operating point.

    python scripts/run_wp8_tradeoff.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import clopper_pearson  # noqa: E402
from sfr.fusion.conformal import leave_one_out_p_values, p_value, smallest_p  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN = "wp5_structure_delta", "wp2b_function_reference", "wp4b_rhythm_intervals"
RHYTHM_SCORE = "oof_logistic_rhythm"
DESIGN = DecisionConfig().alpha
CONFIDENCE = 0.95
N_RHYTHM_HEALTHY_EXPECTED = 14
LEVELS = {
    "structure": (0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
    "function": (0.01, 0.02, 0.05, 0.10),
    "rhythm": (0.05, 1 / 15, 0.10, 0.15, 0.20),
}
CURVE_LEVELS = tuple(round(0.01 * i, 2) for i in range(1, 31))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def p_values() -> dict:
    """Conformal p-values of every expert, as computed for the reported results."""
    for name in (STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN):
        load_run(name)
    cases = read_csv(RUNS / STRUCTURE_RUN / "per_case.csv")
    structure_label = np.array([int(r["label"]) for r in cases])
    structure_p = np.array([float(r["p_value"]) for r in cases])
    function_p = np.array([float(r["p_value"]) for r in read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")])
    records = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    label = np.array([int(r["label"]) for r in records])
    score = np.array([float(r[RHYTHM_SCORE]) for r in records])
    normal, abnormal = score[label == 0], score[label == 1]
    if normal.size != N_RHYTHM_HEALTHY_EXPECTED:
        raise ValueError(f"expected {N_RHYTHM_HEALTHY_EXPECTED} healthy NIFEADB recordings, found {normal.size}")
    return {
        "structure": {"healthy": structure_p[structure_label == 0], "finding": structure_p[structure_label == 1]},
        "function": {"healthy": function_p, "finding": None},
        "rhythm": {"healthy": leave_one_out_p_values(normal), "finding": np.array([p_value(s, normal) for s in abnormal]),
                   "smallest_p_for_a_new_case": smallest_p(int(normal.size)), "smallest_p_healthy_leave_one_out": smallest_p(int(normal.size) - 1)},
    }


def rate(p: np.ndarray, level: float) -> tuple[int, int]:
    p = p[np.isfinite(p)]
    return int((p <= level + 1e-12).sum()), int(p.size)


def main() -> None:
    p = p_values()
    curve_rows = []
    for expert in ("structure", "function", "rhythm"):
        for level in sorted(set(CURVE_LEVELS) | set(LEVELS[expert])):
            k_h, n_h = rate(p[expert]["healthy"], level)
            low_h, high_h = clopper_pearson(k_h, n_h)
            row = {"expert": expert, "alarm_level": level, "healthy_flagged": k_h, "healthy_n": n_h, "healthy_rate": k_h / n_h,
                   "healthy_ci_low": low_h, "healthy_ci_high": high_h,
                   "finding_caught": "", "finding_n": "", "sensitivity": "", "sensitivity_ci_low": "", "sensitivity_ci_high": ""}
            if p[expert]["finding"] is not None:
                k_f, n_f = rate(p[expert]["finding"], level)
                low_f, high_f = clopper_pearson(k_f, n_f)
                row.update({"finding_caught": k_f, "finding_n": n_f, "sensitivity": k_f / n_f, "sensitivity_ci_low": low_f, "sensitivity_ci_high": high_f})
            curve_rows.append(row)

    design_rows = []
    for a_s, a_f, a_r in itertools.product(LEVELS["structure"], LEVELS["function"], LEVELS["rhythm"]):
        counts = [rate(p["structure"]["healthy"], a_s), rate(p["function"]["healthy"], a_f), rate(p["rhythm"]["healthy"], a_r)]
        rates = [k / n for k, n in counts]
        level_each = 1 - (1 - CONFIDENCE) / len(counts)
        upper = sum(clopper_pearson(k, n, alpha=2 * (1 - level_each))[1] for k, n in counts)
        caught_s, n_s = rate(p["structure"]["finding"], a_s)
        caught_r, n_r = rate(p["rhythm"]["finding"], a_r)
        design_rows.append({
            "structure_level": a_s, "function_level": a_f, "rhythm_level": a_r, "ceiling": a_s + a_f + a_r,
            "healthy_rate_lowest_any_dependence": max(rates), "healthy_rate_highest_any_dependence": min(1.0, sum(rates)),
            "healthy_rate_highest_with_uncertainty": min(1.0, upper),
            "structural_defect_sensitivity": caught_s / n_s, "arrhythmia_sensitivity": caught_r / n_r,
            "rhythm_can_alarm": a_r >= p["rhythm"]["smallest_p_for_a_new_case"] - 1e-12,
            "is_the_design": (a_s, a_f, a_r) == (DESIGN["structure"], DESIGN["function"], DESIGN["rhythm"])})

    design = next(row for row in design_rows if row["is_the_design"])
    published = load_run("wp6b_fusion_structure")[0]
    a, b = published["A_alarms_of_three_experts"], published["B_healthy_people_with_three_tests"]["false_referral_rate_three_tests"]
    assert abs(design["structural_defect_sensitivity"] - a["structure"]["sensitivity"]) < 1e-12
    assert abs(design["arrhythmia_sensitivity"] - a["rhythm"]["sensitivity"]) < 1e-12
    assert abs(design["healthy_rate_lowest_any_dependence"] - b["tied"]) < 1e-12
    assert abs(design["healthy_rate_highest_any_dependence"] - b["alarms_never_overlap"]) < 1e-12
    metrics = {
        "made_after_seeing_results": True,
        "design_levels": dict(DESIGN),
        "design_row": design,
        "rhythm_smallest_level": {
            "n_healthy_calibration": N_RHYTHM_HEALTHY_EXPECTED,
            "smallest_p_for_a_new_case": p["rhythm"]["smallest_p_for_a_new_case"],
            "smallest_p_healthy_leave_one_out": p["rhythm"]["smallest_p_healthy_leave_one_out"],
            # p-values with n calibration cases take the values j/(n+1), so a healthy case alarms at level a with probability floor(a(n+1))/(n+1)
            "effective_level_of_the_design": float(np.floor(DESIGN["rhythm"] * (N_RHYTHM_HEALTHY_EXPECTED + 1) + 1e-12) / (N_RHYTHM_HEALTHY_EXPECTED + 1)),
            "effective_ceiling_of_the_design": DESIGN["structure"] + DESIGN["function"]
            + float(np.floor(DESIGN["rhythm"] * (N_RHYTHM_HEALTHY_EXPECTED + 1) + 1e-12) / (N_RHYTHM_HEALTHY_EXPECTED + 1)),
            "statement": "with 14 healthy calibration recordings the rhythm expert cannot alarm at a level below 1/15, and its p-values are multiples "
                         "of 1/15, so the design level 0.10 acts as 1/15 for a new case; in the leave-one-out evaluation each healthy recording is "
                         "ranked among the other 13, so at 1/15 its measured healthy alarm rate is 0 by construction",
        },
        "smallest_p_structure_per_fold": "about 1/125 (calibration normal cases of each fold); see wp5_structure_delta",
        "caution": "the curves are measured on the evaluation data; choosing a level from them and reporting performance at that level on the same data would be optimistic",
        "n_designs": len(design_rows),
    }
    inputs = {"structure": [RUNS / STRUCTURE_RUN / "per_case.csv"], "function": [RUNS / FUNCTION_RUN / "per_recording.csv"],
              "rhythm": [RUNS / RHYTHM_RUN / "per_record_scores.csv"]}
    run = Run("wp8_tradeoff",
              description="Alarm levels against sensitivity and healthy alarm rate for each expert, and the false-referral ceiling of every "
                          "combination of levels. A check made after seeing the results.",
              script="scripts/run_wp8_tradeoff.py",
              config={"source_runs": [STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN], "rhythm_score": RHYTHM_SCORE, "levels": {k: list(v) for k, v in LEVELS.items()},
                      "curve_levels": list(CURVE_LEVELS), "confidence": CONFIDENCE, "made_after_seeing_results": True},
              inputs=inputs, n_real=sum(int(np.isfinite(v).sum()) for e in p.values() for k, v in e.items() if k in ("healthy", "finding") and v is not None))
    run.write_csv("expert_curves.csv", curve_rows)
    run.write_csv("designs.csv", design_rows)
    run.finish(metrics)
    print(f"design {DESIGN}: ceiling {design['ceiling']:.2f}, healthy range [{design['healthy_rate_lowest_any_dependence']:.3f}, "
          f"{design['healthy_rate_highest_any_dependence']:.3f}], structure sensitivity {design['structural_defect_sensitivity']:.3f}, "
          f"arrhythmia sensitivity {design['arrhythmia_sensitivity']:.3f}")
    for row in design_rows:
        if row["function_level"] == DESIGN["function"] and row["structure_level"] == DESIGN["structure"]:
            print(f"  rhythm level {row['rhythm_level']:.4f}: ceiling {row['ceiling']:.3f}, arrhythmia sensitivity {row['arrhythmia_sensitivity']:.3f}, "
                  f"can alarm {row['rhythm_can_alarm']}")


if __name__ == "__main__":
    main()
