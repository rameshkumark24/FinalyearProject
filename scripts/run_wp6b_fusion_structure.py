"""WP6b - the decision rule with the structure expert's real scores.

Run wp6_fusion had no structure scores. This run adds, from the structure run on labelled echo
images, what could not be done before:

A. Alarms of all three experts at their own levels, each on its own dataset.
B. False referrals of healthy people when all three tests are done. Nobody has all three tests,
   so the healthy scores of the three experts are combined in three ways: independently, tied in
   rank, and in the worst case that the three alarm rates allow (alarms never fall on the same
   person).
C. The comparison of rules for a structural defect. A virtual patient pairs the structure score
   of one real case (normal or abnormal) with the function score of one real healthy recording,
   under three couplings, with the Doppler done in every patient or in half of them. This repeats
   part C of wp6_fusion with far more abnormal and normal scores.

Virtual patients are combinations of real held-out scores of different people. They are labelled
as such in every output and are not a measurement of a fused system on patients.

    python scripts/run_wp6b_fusion_structure.py --structure-run wp5_structure_delta
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import clopper_pearson  # noqa: E402
from sfr.fusion import rules  # noqa: E402
from sfr.fusion.conformal import leave_one_out_p_values  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

RHYTHM_RUN, FUNCTION_RUN, FUSION_RUN = "wp4b_rhythm_intervals", "wp2b_function_reference", "wp6_fusion"
RHYTHM_SCORE = "oof_logistic_rhythm"
CONFIG = DecisionConfig()
TARGET_RATES = (0.10, 0.15)
COUPLINGS = ("independent", "tied", "opposed")
N_BOOT, SEED = 500, 0


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dempster_columns(strengths: np.ndarray, ignorance: float = 0.1) -> np.ndarray:
    """``sfr.fusion.rules.dempster`` computed for all cases at once (same numbers, tested)."""
    s = np.atleast_2d(np.asarray(strengths, dtype=float))
    abnormal, normal, either = np.zeros(len(s)), np.zeros(len(s)), np.ones(len(s))
    for column in s.T:
        present = ~np.isnan(column)
        value = np.where(present, column, 0.0)
        a2, n2, e2 = value * (1 - ignorance), (1 - value) * (1 - ignorance), ignorance
        scale = 1.0 - (abnormal * n2 + normal * a2)
        degenerate = present & (scale <= rules.EPSILON)
        usual = present & ~degenerate
        safe = np.where(usual, scale, 1.0)
        new = ((abnormal * a2 + abnormal * e2 + either * a2) / safe,
               (normal * n2 + normal * e2 + either * n2) / safe, either * e2 / safe)
        abnormal = np.where(usual, new[0], np.where(degenerate, 0.5, abnormal))
        normal = np.where(usual, new[1], np.where(degenerate, 0.5, normal))
        either = np.where(usual, new[2], np.where(degenerate, 0.0, either))
    return np.where(np.isnan(s).all(axis=1), np.nan, abnormal)


RULES = {**rules.RULES, "dempster": dempster_columns}


def pair(first: np.ndarray, second: np.ndarray, coupling: str) -> tuple[np.ndarray, np.ndarray]:
    """Strengths of two tests in virtual patients under a stated coupling (as in run_wp6_fusion.py)."""
    if coupling == "independent":
        return np.repeat(first, second.size), np.tile(second, first.size)
    order_second = np.sort(second)
    order_first = np.sort(first) if coupling == "tied" else np.sort(first)[::-1]
    index = np.minimum((np.arange(order_second.size) + 0.5) / order_second.size * order_first.size, order_first.size - 1).astype(int)
    return order_first[index], order_second


def threshold_for_rate(normal_scores: np.ndarray, target: float) -> tuple[float, float]:
    """Smallest threshold whose false-referral rate on normal patients does not exceed ``target``."""
    ordered = np.sort(normal_scores)[::-1]
    allowed = int(np.floor(target * ordered.size + 1e-9))
    if allowed == 0:
        return float("inf"), 0.0
    value = ordered[allowed - 1]
    if allowed < ordered.size and ordered[allowed] == value:      # a tie crosses the limit: step above it
        above = ordered[ordered > value]
        if above.size == 0:
            return float("inf"), 0.0
        value = above.min()
    return float(value), float(np.mean(normal_scores >= value))


def compare_rules(structure_normal, structure_abnormal, function_healthy, coupling: str, availability: str, target: float) -> dict:
    """Sensitivity of every rule for a structural defect at a matched false-referral rate."""
    normal_s, normal_f = pair(structure_normal, function_healthy, coupling)
    abnormal_s, abnormal_f = pair(structure_abnormal, function_healthy, coupling)
    if availability == "mixed":                                   # half of the patients have no Doppler
        normal = np.vstack([np.column_stack([normal_s, normal_f]), np.column_stack([normal_s, np.full(normal_s.size, np.nan)])])
        abnormal = np.vstack([np.column_stack([abnormal_s, abnormal_f]), np.column_stack([abnormal_s, np.full(abnormal_s.size, np.nan)])])
    else:
        normal, abnormal = np.column_stack([normal_s, normal_f]), np.column_stack([abnormal_s, abnormal_f])
    out = {"n_virtual_normal": int(normal.shape[0]), "n_virtual_abnormal": int(abnormal.shape[0])}
    for name, rule in RULES.items():
        threshold, rate = threshold_for_rate(rule(normal), target)
        out[name] = {"false_referral_rate": rate, "sensitivity": float((rule(abnormal) >= threshold).mean())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--structure-run", required=True, help="name of the structure run folder, e.g. wp5_structure_delta")
    args = parser.parse_args()
    for name in (RHYTHM_RUN, FUNCTION_RUN, FUSION_RUN, args.structure_run):
        load_run(name)
    fusion, _ = load_run(FUSION_RUN)
    structure_metrics, _ = load_run(args.structure_run)
    alpha = CONFIG.alpha

    cases = read_csv(RUNS / args.structure_run / "per_case.csv")
    label = np.array([int(row["label"]) for row in cases])
    p_structure = np.array([float(row["p_value"]) for row in cases])
    function_rows = read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")
    p_function = np.array([float(row["p_value"]) for row in function_rows])
    woman = np.array([row["woman"] for row in function_rows])
    rhythm_rows = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    rhythm_label = np.array([int(row["label"]) for row in rhythm_rows])
    rhythm_score = np.array([float(row[RHYTHM_SCORE]) for row in rhythm_rows])
    p_rhythm_normal = leave_one_out_p_values(rhythm_score[rhythm_label == 0])

    # ---- A. the three experts at their alarm levels ---------------------------------------------------
    caught, flagged = int((p_structure[label == 1] <= alpha["structure"]).sum()), int((p_structure[label == 0] <= alpha["structure"]).sum())
    n_abnormal, n_normal = int((label == 1).sum()), int((label == 0).sum())
    part_a = {
        "structure": {"alpha": alpha["structure"], "dataset": structure_metrics["dataset"]["name"], "n_abnormal": n_abnormal, "n_normal": n_normal,
                      "caught": caught, "sensitivity": caught / n_abnormal, "sensitivity_ci": list(clopper_pearson(caught, n_abnormal)),
                      "normal_flagged": flagged, "false_alarm_rate": flagged / n_normal, "false_alarm_ci": list(clopper_pearson(flagged, n_normal))},
        "function": fusion["A_conformal_alarms"]["function"],
        "rhythm": fusion["A_conformal_alarms"]["rhythm"],
        "false_referral_ceiling_three_experts": alpha["structure"] + alpha["function"] + alpha["rhythm"],
    }

    # ---- B. healthy people with all three tests ---------------------------------------------------------
    rate = {"structure": float(np.mean(p_structure[label == 0] <= alpha["structure"])),
            "function": float(np.mean(p_function <= alpha["function"])),
            "rhythm": float(np.mean(p_rhythm_normal <= alpha["rhythm"]))}
    ceiling = part_a["false_referral_ceiling_three_experts"]
    combined = {"independent": 1.0 - float(np.prod([1.0 - value for value in rate.values()])),
                "tied": max(rate.values()), "alarms_never_overlap": min(1.0, sum(rate.values()))}
    part_b = {"false_alarm_rate_of_each_expert": rate, "false_referral_rate_three_tests": combined, "ceiling_sum_of_levels": ceiling,
              "ceiling_respected": {key: bool(value <= ceiling + 1e-9) for key, value in combined.items()},
              "note": "healthy scores of three different groups of people combined; the last row is the largest rate any dependence could give"}

    # ---- C. rules for a structural defect, on virtual patients --------------------------------------------
    strength_normal, strength_abnormal, strength_function = 1.0 - p_structure[label == 0], 1.0 - p_structure[label == 1], 1.0 - p_function
    rng = np.random.default_rng(SEED)
    women = np.unique(woman)
    comparison = []
    for target in TARGET_RATES:
        for availability in ("both_tests", "mixed"):
            for coupling in COUPLINGS:
                point = compare_rules(strength_normal, strength_abnormal, strength_function, coupling, availability, target)
                draws = {name: [] for name in RULES}
                for _ in range(N_BOOT):
                    normal_sample = strength_normal[rng.integers(0, strength_normal.size, strength_normal.size)]
                    abnormal_sample = strength_abnormal[rng.integers(0, strength_abnormal.size, strength_abnormal.size)]
                    function_sample = np.concatenate([strength_function[woman == w] for w in rng.choice(women, women.size, replace=True)])
                    sample = compare_rules(normal_sample, abnormal_sample, function_sample, coupling, availability, target)
                    for name in RULES:
                        draws[name].append(sample[name]["sensitivity"])
                gap = np.array(draws["maximum"]) - np.array(draws["mean"])
                row = {"target_false_referral_rate": target, "availability": availability, "coupling": coupling,
                       "n_virtual_normal": point["n_virtual_normal"], "n_virtual_abnormal": point["n_virtual_abnormal"],
                       "n_real_scores": int(strength_normal.size + strength_abnormal.size + strength_function.size),
                       "max_minus_mean_sensitivity": point["maximum"]["sensitivity"] - point["mean"]["sensitivity"],
                       "max_minus_mean_ci_low": float(np.percentile(gap, 2.5)), "max_minus_mean_ci_high": float(np.percentile(gap, 97.5))}
                for name in RULES:
                    row[f"{name}_sensitivity"] = point[name]["sensitivity"]
                    row[f"{name}_ci_low"] = float(np.percentile(draws[name], 2.5))
                    row[f"{name}_ci_high"] = float(np.percentile(draws[name], 97.5))
                    row[f"{name}_false_referral_rate"] = point[name]["false_referral_rate"]
                comparison.append(row)
                print(f"{target:.2f} {availability:10s} {coupling:11s} " + " ".join(f"{name[:7]} {point[name]['sensitivity']:.3f}" for name in RULES)
                      + f"  max-mean {row['max_minus_mean_sensitivity']:+.3f} [{row['max_minus_mean_ci_low']:+.3f}, {row['max_minus_mean_ci_high']:+.3f}]", flush=True)

    n_virtual = int(sum(row["n_virtual_normal"] + row["n_virtual_abnormal"] for row in comparison))
    run = Run(
        "wp6b_fusion_structure",
        description="Decision rule with the structure expert's real scores: alarms of three experts, false referrals with three tests, and the rule comparison for a structural defect on virtual patients",
        script="scripts/run_wp6b_fusion_structure.py",
        config={"alpha": dict(alpha), "source_runs": [args.structure_run, FUNCTION_RUN, RHYTHM_RUN, FUSION_RUN], "rhythm_score": RHYTHM_SCORE,
                "couplings": COUPLINGS, "target_false_referral_rates": TARGET_RATES, "bootstrap": N_BOOT},
        inputs={"structure": [RUNS / args.structure_run / "per_case.csv"], "function": [RUNS / FUNCTION_RUN / "per_recording.csv"],
                "rhythm": [RUNS / RHYTHM_RUN / "per_record_scores.csv"]},
        n_real=int(label.size + p_function.size + rhythm_label.size),
        n_synthetic=n_virtual,
        synthetic_kind="virtual patients made by pairing real held-out scores of different people (part C; coupling stated in every row). Parts A and B use real scores only.",
        seed=SEED,
    )
    run.write_csv("rule_comparison_structure.csv", comparison)
    mixed = [row for row in comparison if row["availability"] == "mixed"]
    both = [row for row in comparison if row["availability"] == "both_tests"]
    metrics = {
        "A_alarms_of_three_experts": part_a, "B_healthy_people_with_three_tests": part_b,
        "C_rule_comparison_structural_defect": {"rows": len(comparison), "structure_run": args.structure_run},
        "hypothesis_H7_structure": {
            "statement": fusion["hypothesis_H7"]["statement"],
            "rows_checked": len(mixed),
            "rows_where_mean_rule_has_lower_sensitivity": int(sum(row["max_minus_mean_sensitivity"] > 0 for row in mixed)),
            "rows_where_interval_excludes_zero": int(sum(row["max_minus_mean_ci_low"] > 0 for row in mixed)),
            "established": bool(mixed) and all(row["max_minus_mean_ci_low"] > 0 for row in mixed),
            "both_tests_rows_where_mean_rule_has_lower_sensitivity": int(sum(row["max_minus_mean_sensitivity"] > 0 for row in both)),
            "both_tests_rows_where_interval_excludes_zero": int(sum(row["max_minus_mean_ci_low"] > 0 for row in both)),
        },
    }
    run.finish(metrics)
    print("A:", part_a["structure"])
    print("B:", part_b)
    print("H7 (structure):", metrics["hypothesis_H7_structure"])


if __name__ == "__main__":
    main()
