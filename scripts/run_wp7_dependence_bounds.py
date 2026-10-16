"""WP7 - what the fused system guarantees without paired data.

Added on 16 October 2026 in answer to a review, so every part of this run is a check made
after seeing the results. Nothing is fitted. The inputs are alarm counts that earlier runs
already report; they are recounted here from the per-case files of those runs and must match.

1. False-referral range for any dependence. If the experts alarm on healthy fetuses at rates
   r_1..r_K, the OR rule refers a healthy fetus at a rate between max r_k (the alarms always
   come together) and min(1, sum r_k) (they never come together), whatever the dependence
   between the tests (Frechet bounds). The range is given at the measured rates and with their
   uncertainty: one-sided Clopper-Pearson limits at 1 - 0.05/K for each of the K experts
   (Bonferroni), so the widened range holds with at least 95% confidence for all rates together.
2. On the 60 real ECG-Doppler pairs of NInFEA the joint rate was measured (run wp6_fusion); it
   is compared with the range computed from the two experts' own rates on the same recordings.
3. A dependence dial: the rate under a Gaussian copula with one correlation shared by every
   pair of experts. It shows how dependence moves the rate; it is not a model of real fetuses.
4. With a fixed calibration set the rate actually obtained varies around the alarm level
   (Beta distribution, sfr.fusion.conformal.realised_false_alarm_range).
5. Sensitivity floor: the OR rule refers whenever the expert for a finding alarms, so for that
   finding its sensitivity is at least the expert's, measured on that expert's own dataset.
6. Precision of a referral over a grid of prevalences. The priors in the decision rule are
   assumptions without a cited source, so no single prevalence is used.
7. Gate G7, defined after seeing the results: the range for any dependence lies inside the
   design ceiling (the sum of the alarm levels), at the measured rates and with uncertainty.

    python scripts/run_wp7_dependence_bounds.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import clopper_pearson  # noqa: E402
from sfr.fusion.conformal import leave_one_out_p_values, p_value, realised_false_alarm_range  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN = "wp5_structure_delta", "wp2b_function_reference", "wp4b_rhythm_intervals"
PAIRS_RUN, FUSED_RUN = "wp6_fusion", "wp6b_fusion_structure"
RHYTHM_SCORE = "oof_logistic_rhythm"                 # the rhythm expert fixed in the plan (as in wp6 and wp6b)
CONFIG = DecisionConfig()
CONFIDENCE = 0.95
CALIBRATION_LEVEL = 0.90                             # as in wp6_fusion
RHO_GRID = tuple(round(-0.45 + 0.05 * i, 2) for i in range(29)) + (1.0,)   # three variables cannot share a correlation below -0.5
# TODO(prevalence): the priors of the decision rule (sfr/fusion/decision.py, DecisionConfig.prior:
# structure 0.009, function 0.002, rhythm 0.001) are assumptions without a cited source. When the
# team has a cited prevalence, add it to this grid and to DecisionConfig, re-run, and cite it in
# the paper; docs/analysis_notes.md lists every place that would change.
PREVALENCE_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
SEED = 0


# ---- bounds (pure functions, tested in tests/test_guarantees.py) -----------------------------------------

def frechet_range(rates) -> tuple[float, float]:
    """Smallest and largest probability that at least one of several events happens, for any dependence."""
    rates = np.asarray(rates, dtype=float)
    return float(rates.max()), float(min(1.0, rates.sum()))


def independent_rate(rates) -> float:
    return float(1.0 - np.prod(1.0 - np.asarray(rates, dtype=float)))


def gaussian_copula_any_alarm(rates, rho: float, seed: int = SEED) -> float:
    """Probability that at least one expert alarms when the alarms come from correlated normal variables.

    Expert k alarms when its latent normal value exceeds the quantile that gives rate r_k; every
    pair of latent values has correlation ``rho``. rho = 0 gives independence, rho = 1 tied alarms.
    """
    rates = np.asarray(rates, dtype=float)
    rates = rates[rates > 0]
    if rates.size == 0:
        return 0.0
    if np.any(rates >= 1):
        return 1.0
    if rates.size == 1 or rho >= 1:
        return float(rates.max())
    if rho <= -1.0 / (rates.size - 1):
        if rates.size == 2:                         # rho = -1: alarms never together when possible
            return float(min(1.0, rates.sum()))
        raise ValueError(f"{rates.size} variables cannot share a correlation of {rho}")
    covariance = np.full((rates.size, rates.size), float(rho))
    np.fill_diagonal(covariance, 1.0)
    none = stats.multivariate_normal.cdf(stats.norm.ppf(1.0 - rates), mean=np.zeros(rates.size), cov=covariance,
                                         abseps=1e-8, releps=1e-8, rng=np.random.default_rng(seed))
    return float(1.0 - none)


def one_sided_limits(k: int, n: int, level: float) -> tuple[float, float]:
    """Lower and upper one-sided Clopper-Pearson limits, each holding with probability ``level``."""
    return clopper_pearson(k, n, alpha=2 * (1 - level))


def range_with_uncertainty(counts: list[tuple[int, int]], confidence: float = CONFIDENCE) -> dict:
    """Frechet range at the measured rates, and widened by the uncertainty of every rate (Bonferroni)."""
    rates = [k / n for k, n in counts]
    level = 1 - (1 - confidence) / len(counts)
    limits = [one_sided_limits(k, n, level) for k, n in counts]
    low, high = frechet_range(rates)
    return {"rates": rates, "range": [low, high], "independent": independent_rate(rates),
            "one_sided_level_each": level, "limits_each": [list(pair) for pair in limits],
            "range_with_uncertainty": [max(pair[0] for pair in limits), float(min(1.0, sum(pair[1] for pair in limits)))]}


def precision(sensitivity: float, false_referral: float, prevalence: float) -> float:
    """Share of referred fetuses that have the finding."""
    true = sensitivity * prevalence
    return float(true / (true + false_referral * (1 - prevalence)))


# ---- inputs, recounted from per-case files ------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def recount() -> dict:
    """Alarm counts of every expert at its level, from the per-case files; checked against the published metrics."""
    fused, _ = load_run(FUSED_RUN)
    pairs, _ = load_run(PAIRS_RUN)
    for name in (STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN):
        load_run(name)
    a = fused["A_alarms_of_three_experts"]

    cases = read_csv(RUNS / STRUCTURE_RUN / "per_case.csv")
    label = np.array([int(r["label"]) for r in cases])
    alarm = np.array([float(r["p_value"]) <= CONFIG.alpha["structure"] for r in cases])
    structure = {"healthy": (int(alarm[label == 0].sum()), int((label == 0).sum())),
                 "finding": (int(alarm[label == 1].sum()), int((label == 1).sum()))}
    assert structure["healthy"] == (a["structure"]["normal_flagged"], a["structure"]["n_normal"])
    assert structure["finding"] == (a["structure"]["caught"], a["structure"]["n_abnormal"])

    recordings = read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")
    p_function = np.array([float(r["p_value"]) for r in recordings])
    function = {"healthy": (int((p_function <= CONFIG.alpha["function"]).sum()), int(p_function.size))}
    assert function["healthy"] == (a["function"]["flagged"], a["function"]["n_healthy_recordings"])

    records = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    rhythm_label = np.array([int(r["label"]) for r in records])
    score = np.array([float(r[RHYTHM_SCORE]) for r in records])
    normal, abnormal = score[rhythm_label == 0], score[rhythm_label == 1]
    p_normal = leave_one_out_p_values(normal)
    p_abnormal = np.array([p_value(s, normal) for s in abnormal])
    rhythm = {"healthy": (int((p_normal <= CONFIG.alpha["rhythm"]).sum()), int(normal.size)),
              "finding": (int((p_abnormal <= CONFIG.alpha["rhythm"]).sum()), int(abnormal.size)),
              "calibration_cases": int(normal.size)}
    assert rhythm["healthy"] == (a["rhythm"]["normal_flagged"], a["rhythm"]["n_normal"])
    assert rhythm["finding"] == (a["rhythm"]["arrhythmia_caught"], a["rhythm"]["n_arrhythmia"])

    cohort = read_csv(RUNS / PAIRS_RUN / "paired_cohort_ninfea.csv")
    readable = sum(r["ecg_readable"] == "True" for r in cohort)
    local = {"rhythm": sum("rhythm" in r["alarms_local"].split(";") for r in cohort),
             "function": sum("function" in r["alarms_local"].split(";") for r in cohort),
             "referred": sum(r["action_local_calibration"] == "REFER" for r in cohort)}
    moved = {"rhythm": sum("rhythm" in r["alarms"].split(";") for r in cohort),
             "function": sum("function" in r["alarms"].split(";") for r in cohort),
             "referred": sum(r["action_if_echo_normal"] == "REFER" for r in cohort)}
    b = pairs["B_real_paired_cohort"]
    published_local = b["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]["false_referrals"]
    assert (local["referred"], local["rhythm"], local["function"]) == (published_local["recordings"], published_local["by_expert"]["rhythm"],
                                                                      published_local["by_expert"]["function"])
    assert (moved["referred"], moved["rhythm"], moved["function"]) == (b["false_referrals"]["recordings"], b["false_referrals"]["by_expert"]["rhythm"],
                                                                      b["false_referrals"]["by_expert"]["function"])
    assert readable == b["ecg_readable"]
    own = load_run(STRUCTURE_RUN)[0]["own_partition"]["fetalclip_scan"]
    return {"structure": structure, "function": function, "rhythm": rhythm,
            "pairs": {"n": len(cohort), "ecg_readable": readable, "local": local, "moved": moved},
            "structure_calibration_cases_own_partition": own["calibration_normal_cases"]}


# ---- the run ------------------------------------------------------------------------------------------

def main() -> None:
    counts = recount()
    alpha = CONFIG.alpha
    s, f, r, pairs = counts["structure"], counts["function"], counts["rhythm"], counts["pairs"]
    n_pairs = pairs["n"]
    local, moved = pairs["local"], pairs["moved"]
    combinations = [
        {"combination": "three experts, each at its rate on its own healthy data", "people": "three different groups combined",
         "experts": [("structure (DELTA)", *s["healthy"], alpha["structure"]), ("function (NInFEA)", *f["healthy"], alpha["function"]),
                     ("rhythm (NIFEADB)", *r["healthy"], alpha["rhythm"])], "measured": None, "gate": True},
        {"combination": "three experts, rhythm calibrated on healthy NInFEA (leave one woman out)", "people": "three different groups combined",
         "experts": [("structure (DELTA)", *s["healthy"], alpha["structure"]), ("function (NInFEA)", *f["healthy"], alpha["function"]),
                     ("rhythm, local calibration (NInFEA, readable ECG)", local["rhythm"], pairs["ecg_readable"], alpha["rhythm"])],
         "measured": None, "gate": False},
        {"combination": "ECG and Doppler of the same fetuses, rhythm calibrated on NInFEA", "people": "the same 60 recordings (real pairs)",
         "experts": [("function", local["function"], n_pairs, alpha["function"]), ("rhythm, local calibration", local["rhythm"], n_pairs, alpha["rhythm"])],
         "measured": (local["referred"], n_pairs), "gate": False},
        {"combination": "ECG and Doppler of the same fetuses, rhythm calibrated on NIFEADB (another device)", "people": "the same 60 recordings (real pairs)",
         "experts": [("function", moved["function"], n_pairs, alpha["function"]), ("rhythm, calibrated on NIFEADB", moved["rhythm"], n_pairs, alpha["rhythm"])],
         "measured": (moved["referred"], n_pairs), "gate": False},
        {"combination": "echo, plus the ECG-Doppler pair as measured on real pairs (local calibration)", "people": "two different groups combined",
         "experts": [("structure (DELTA)", *s["healthy"], alpha["structure"]),
                     ("ECG-Doppler pair, measured", local["referred"], n_pairs, alpha["function"] + alpha["rhythm"])],
         "measured": None, "gate": False},
    ]
    range_rows = []
    for item in combinations:
        result = range_with_uncertainty([(k, n) for _, k, n, _ in item["experts"]])
        ceiling = float(sum(level for *_, level in item["experts"]))
        row = {"combination": item["combination"], "people": item["people"],
               "experts": "; ".join(f"{name} {k}/{n}" for name, k, n, _ in item["experts"]),
               "ceiling": ceiling, "lowest_rate_any_dependence": result["range"][0], "rate_if_independent": result["independent"],
               "highest_rate_any_dependence": result["range"][1],
               "lowest_with_uncertainty": result["range_with_uncertainty"][0], "highest_with_uncertainty": result["range_with_uncertainty"][1],
               "one_sided_level_each": result["one_sided_level_each"],
               "inside_ceiling_at_measured_rates": result["range"][1] <= ceiling,
               "inside_ceiling_with_uncertainty": result["range_with_uncertainty"][1] <= ceiling,
               "measured_joint_rate": "", "measured_joint_ci": "", "measured_inside_range": ""}
        if item["measured"]:
            k, n = item["measured"]
            row["measured_joint_rate"] = k / n
            row["measured_joint_ci"] = json.dumps([round(v, 4) for v in clopper_pearson(k, n)])
            row["measured_inside_range"] = result["range"][0] - 1e-12 <= k / n <= result["range"][1] + 1e-12
        row["_limits_each"] = result["limits_each"]
        range_rows.append(row)

    design = range_rows[0]
    rates_design = [k / n for _, k, n, _ in combinations[0]["experts"]]
    high_rates_design = [pair[1] for pair in design["_limits_each"]]
    dial_rows = [{"rho": rho, "rate_at_measured_rates": gaussian_copula_any_alarm(rates_design, rho),
                  "rate_at_upper_limits": gaussian_copula_any_alarm(high_rates_design, rho)} for rho in RHO_GRID]

    calibration_rows = []
    for expert, n, level, source in (("structure", counts["structure_calibration_cases_own_partition"], alpha["structure"], "normal cases of the DELTA Val split"),
                                     ("function", f["healthy"][1], alpha["function"], "healthy NInFEA recordings"),
                                     ("rhythm, calibrated on NIFEADB", r["calibration_cases"], alpha["rhythm"], "normal NIFEADB recordings"),
                                     ("rhythm, calibrated on NInFEA", pairs["ecg_readable"], alpha["rhythm"], "healthy NInFEA recordings with a readable ECG")):
        low, high = realised_false_alarm_range(int(n), level, CALIBRATION_LEVEL)
        calibration_rows.append({"expert": expert, "calibration_set": source, "n_healthy": int(n), "alarm_level": level,
                                 "smallest_possible_p": 1 / (n + 1), "range_low": low, "range_high": high, "level": CALIBRATION_LEVEL})

    floor_rows = []
    for finding, expert, caught, dataset, note in (
            ("structural heart defect", "structure", s["finding"], "DELTA, case-level folds",
             "inputs with little or no cardiac anatomy also separate the classes; an upper bound on what heart structure contributes"),
            ("arrhythmia", "rhythm", r["finding"], "NIFEADB, leave one subject out",
             "the four recordings sampled at 500 Hz are all arrhythmia"),
            ("abnormal Doppler timing", "function", None, "NInFEA (healthy only)", "no abnormal cases: cannot be measured")):
        if caught is None:
            floor_rows.append({"finding": finding, "expert": expert, "caught": "", "n": "", "sensitivity_floor": "", "ci_low": "", "ci_high": "",
                               "dataset": dataset, "note": note})
            continue
        k, n = caught
        low, high = clopper_pearson(k, n)
        floor_rows.append({"finding": finding, "expert": expert, "caught": k, "n": n, "sensitivity_floor": k / n, "ci_low": low, "ci_high": high,
                           "dataset": dataset, "note": note})

    precision_rows = []
    for floor in floor_rows[:2]:
        for prevalence in PREVALENCE_GRID:
            precision_rows.append({
                "finding": floor["finding"], "prevalence": prevalence, "sensitivity_used": floor["sensitivity_floor"],
                "precision_alarms_tied": precision(floor["sensitivity_floor"], design["lowest_rate_any_dependence"], prevalence),
                "precision_independent": precision(floor["sensitivity_floor"], design["rate_if_independent"], prevalence),
                "precision_alarms_never_together": precision(floor["sensitivity_floor"], design["highest_rate_any_dependence"], prevalence),
                "precision_pessimistic": precision(floor["ci_low"], design["highest_with_uncertainty"], prevalence)})

    gate = {
        "rule": "the false-referral range for any dependence lies inside the design ceiling (sum of the alarm levels); three experts, "
                "each at its alarm rate on its own healthy data",
        "defined": "after seeing the results (16 October 2026)",
        "ceiling": design["ceiling"],
        "range_at_measured_rates": [design["lowest_rate_any_dependence"], design["highest_rate_any_dependence"]],
        "passed_at_measured_rates": bool(design["inside_ceiling_at_measured_rates"]),
        "range_with_uncertainty": [design["lowest_with_uncertainty"], design["highest_with_uncertainty"]],
        "uncertainty": f"one-sided Clopper-Pearson limit of each rate at level {design['one_sided_level_each']:.4f}, Bonferroni over the "
                       f"three rates, so the range holds with at least {CONFIDENCE:.0%} confidence for all of them together",
        "passed_with_uncertainty": bool(design["inside_ceiling_with_uncertainty"]),
        "upper_limits_each": {name: limits[1] for (name, *_), limits in zip(combinations[0]["experts"], design["_limits_each"])},
    }
    metrics = {
        "made_after_seeing_results": True,
        "inputs_recounted_from_per_case_files_and_checked": True,
        "alarm_levels": dict(alpha),
        "ranges": [{k: v for k, v in row.items() if not k.startswith("_")} | {"limits_each": row["_limits_each"]} for row in range_rows],
        "realised_rate_with_a_fixed_calibration_set": calibration_rows,
        "sensitivity_floor": floor_rows,
        "dependence_dial": {"model": "Gaussian copula, one correlation shared by every pair of experts; an illustration, not data",
                            "rates": rates_design, "upper_limit_rates": high_rates_design, "rho_grid": list(RHO_GRID),
                            "rate_at_rho_0": next(row["rate_at_measured_rates"] for row in dial_rows if row["rho"] == 0),
                            "rate_at_rho_1": next(row["rate_at_measured_rates"] for row in dial_rows if row["rho"] == 1)},
        "priors": {"values": dict(CONFIG.prior), "status": "the function and rhythm priors are assumptions without a cited source",
                   "prevalence_grid": list(PREVALENCE_GRID)},
        "gate_G7": gate,
        "not_measured": "the sensitivity gained by combining experts, and any rate on fetuses that had all three tests",
    }

    inputs = {"structure": [RUNS / STRUCTURE_RUN / "per_case.csv", RUNS / STRUCTURE_RUN / "metrics.json"],
              "function": [RUNS / FUNCTION_RUN / "per_recording.csv"],
              "rhythm": [RUNS / RHYTHM_RUN / "per_record_scores.csv"],
              "pairs": [RUNS / PAIRS_RUN / "paired_cohort_ninfea.csv", RUNS / PAIRS_RUN / "metrics.json"],
              "fused": [RUNS / FUSED_RUN / "metrics.json"]}
    run = Run("wp7_dependence_bounds",
              description="What the fused system guarantees without paired data: false-referral range for any dependence, with uncertainty; "
                          "dependence dial; sensitivity floor; precision over a prevalence grid; gate G7. A check made after seeing the results.",
              script="scripts/run_wp7_dependence_bounds.py",
              config={"source_runs": [STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN, PAIRS_RUN, FUSED_RUN], "rhythm_score": RHYTHM_SCORE,
                      "alpha": dict(alpha), "confidence": CONFIDENCE, "calibration_level": CALIBRATION_LEVEL, "rho_grid": list(RHO_GRID),
                      "prevalence_grid": list(PREVALENCE_GRID), "made_after_seeing_results": True},
              inputs=inputs, n_real=s["healthy"][1] + s["finding"][1] + f["healthy"][1] + r["healthy"][1] + r["finding"][1], seed=SEED)
    run.write_csv("dependence_ranges.csv", [{k: v for k, v in row.items() if not k.startswith("_")} for row in range_rows])
    run.write_csv("dependence_dial.csv", dial_rows)
    run.write_csv("calibration_ranges.csv", calibration_rows)
    run.write_csv("sensitivity_floor.csv", floor_rows)
    run.write_csv("precision_grid.csv", precision_rows)
    run.finish(metrics)
    for row in range_rows:
        print(f"{row['combination'][:70]:70s} [{row['lowest_rate_any_dependence']:.3f}, {row['highest_rate_any_dependence']:.3f}]  "
              f"with uncertainty [{row['lowest_with_uncertainty']:.3f}, {row['highest_with_uncertainty']:.3f}]  ceiling {row['ceiling']:.2f}"
              + (f"  measured {row['measured_joint_rate']:.3f}" if row["measured_joint_rate"] != "" else ""))
    print("G7 at measured rates:", gate["passed_at_measured_rates"], " with uncertainty:", gate["passed_with_uncertainty"])


if __name__ == "__main__":
    main()
