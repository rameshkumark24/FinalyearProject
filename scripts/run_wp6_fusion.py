"""WP6 - fusion and decision: what can be shown without paired patients, and what was measured.

Three parts.

A. Conformal alarms of the two experts that have real held-out scores (rhythm: NIFEADB,
   leave-one-subject-out; function: NInFEA, leave-one-woman-out). The structure expert has no
   labelled data yet and enters only as "test not done".

B. A real paired cohort: the 60 NInFEA recordings have a fetal ECG and a Doppler of the same
   healthy fetus at the same time. The decision rule is applied to them as they are. Every
   referral here is a false referral, every "repeat" is a recording the system would not accept.
   Caveat: the rhythm expert was fitted on another dataset (other device, later gestation).

C. Comparison of combination rules on virtual patients built by pairing real held-out scores of
   different people (labelled semi-synthetic). Because the joint behaviour of the tests is
   unknown, each result is given for three couplings: independent, tied (the two scores rise
   together) and opposed. Rules are compared at the same false-referral rate.

    python scripts/run_wp6_fusion.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import clopper_pearson  # noqa: E402
from sfr.fusion import rules  # noqa: E402
from sfr.fusion.conformal import leave_one_out_p_values, p_value, realised_false_alarm_range, smallest_p  # noqa: E402
from sfr.fusion.decision import DecisionConfig, ExpertResult, decide  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.rhythm.evaluate import logistic  # noqa: E402
from sfr.rhythm.features import RECORD_FEATURES, record_features, segment_features  # noqa: E402

RHYTHM_RUN, FUNCTION_RUN, TRUST_RUN = "wp4b_rhythm_intervals", "wp2b_function_reference", "wp3_trust_check"
RHYTHM_SCORE = "oof_logistic_rhythm"                  # the rhythm expert fixed in the plan: interval features, logistic
CONFIG = DecisionConfig()
TARGET_RATES = (0.10, 0.15)
N_BOOT, SEED = 500, 0
COUPLINGS = ("independent", "tied", "opposed")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair(rhythm: np.ndarray, function: np.ndarray, coupling: str) -> tuple[np.ndarray, np.ndarray]:
    """Rhythm and function strengths of virtual patients under a stated coupling."""
    if coupling == "independent":                       # every pair once: exact, no random draw
        return np.repeat(rhythm, function.size), np.tile(function, rhythm.size)
    order_f = np.sort(function)
    order_r = np.sort(rhythm) if coupling == "tied" else np.sort(rhythm)[::-1]
    index = np.minimum((np.arange(order_f.size) + 0.5) / order_f.size * order_r.size, order_r.size - 1).astype(int)
    return order_r[index], order_f


def threshold_for_rate(normal_scores: np.ndarray, target: float) -> tuple[float, float]:
    """Smallest threshold whose false-referral rate on normal patients does not exceed ``target``."""
    candidates = np.unique(normal_scores)[::-1]
    best, best_rate = np.inf, 0.0
    for value in candidates:
        rate = float(np.mean(normal_scores >= value))
        if rate <= target + 1e-12:
            best, best_rate = float(value), rate
        else:
            break
    return best, best_rate


def compare_rules(rhythm_normal, rhythm_abnormal, function_healthy, coupling: str, availability: str, target: float) -> dict:
    """Sensitivity of every rule for arrhythmia patients at a matched false-referral rate."""
    normal_r, normal_f = pair(rhythm_normal, function_healthy, coupling)
    abnormal_r, abnormal_f = pair(rhythm_abnormal, function_healthy, coupling)
    if availability == "mixed":                          # half of the patients have no Doppler
        normal = np.vstack([np.column_stack([normal_r, normal_f]), np.column_stack([normal_r, np.full(normal_r.size, np.nan)])])
        abnormal = np.vstack([np.column_stack([abnormal_r, abnormal_f]), np.column_stack([abnormal_r, np.full(abnormal_r.size, np.nan)])])
        has_doppler = np.r_[np.ones(abnormal_r.size, bool), np.zeros(abnormal_r.size, bool)]
    else:
        normal, abnormal = np.column_stack([normal_r, normal_f]), np.column_stack([abnormal_r, abnormal_f])
        has_doppler = np.ones(abnormal_r.size, bool)
    out = {"n_virtual_normal": int(normal.shape[0]), "n_virtual_arrhythmia": int(abnormal.shape[0])}
    for name, rule in rules.RULES.items():
        threshold, rate = threshold_for_rate(rule(normal), target)
        hit = rule(abnormal) >= threshold
        out[name] = {"false_referral_rate": rate, "sensitivity": float(hit.mean()),
                     "sensitivity_with_doppler": float(hit[has_doppler].mean()),
                     "sensitivity_without_doppler": float(hit[~has_doppler].mean()) if (~has_doppler).any() else float("nan")}
    return out


def main() -> None:
    for name in (RHYTHM_RUN, FUNCTION_RUN, TRUST_RUN):
        load_run(name)
    rhythm_rows = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    function_rows = read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")
    trust_rows = {int(r["record"]): r for r in read_csv(RUNS / TRUST_RUN / "per_recording.csv")}
    beats = json.loads((RUNS / TRUST_RUN / "beats.json").read_text(encoding="utf-8"))

    # ---- A. conformal alarms of the two experts -----------------------------------------------------------
    labels = np.array([int(r["label"]) for r in rhythm_rows])
    scores = np.array([float(r[RHYTHM_SCORE]) for r in rhythm_rows])
    normal_scores, abnormal_scores = scores[labels == 0], scores[labels == 1]
    p_normal = leave_one_out_p_values(normal_scores)
    p_abnormal = np.array([p_value(s, normal_scores) for s in abnormal_scores])
    alpha_r, alpha_f = CONFIG.alpha["rhythm"], CONFIG.alpha["function"]
    caught, flagged = int((p_abnormal <= alpha_r).sum()), int((p_normal <= alpha_r).sum())
    function_p = np.array([float(r["p_value"]) for r in function_rows])
    function_woman = np.array([r["woman"] for r in function_rows])
    function_flagged = int((function_p <= alpha_f).sum())
    part_a = {
        "rhythm": {"alpha": alpha_r, "n_arrhythmia": int(p_abnormal.size), "n_normal": int(p_normal.size),
                   "smallest_possible_p": smallest_p(int(normal_scores.size)),
                   "arrhythmia_caught": caught, "sensitivity": caught / p_abnormal.size,
                   "sensitivity_ci": list(clopper_pearson(caught, p_abnormal.size)),
                   "normal_flagged": flagged, "false_alarm_rate": flagged / p_normal.size,
                   "false_alarm_ci": list(clopper_pearson(flagged, p_normal.size)),
                   "false_alarm_range_of_a_fixed_calibration_set_90pct": list(realised_false_alarm_range(int(normal_scores.size), alpha_r))},
        "function": {"alpha": alpha_f, "n_healthy_recordings": int(function_p.size), "n_women": int(np.unique(function_woman).size),
                     "flagged": function_flagged, "false_alarm_rate": function_flagged / function_p.size,
                     "no_abnormal_cases": "sensitivity cannot be measured"},
        "structure": {"status": "no labelled data; the expert is 'test not done' in every analysis below"},
        "false_referral_ceiling_two_experts": alpha_r + alpha_f,
    }
    # Other rhythm scores of run wp4b at the same alarm level. The logistic model on interval
    # features is the expert fixed in the plan; the others are shown for comparison only.
    alternatives = {}
    for name, sign in (("oof_forest_rhythm", 1.0), ("succ_rel_median", 1.0), ("outlier_share_p90", 1.0)):
        values = sign * np.array([float(r[name]) for r in rhythm_rows])
        normal_values, abnormal_values = values[labels == 0], values[labels == 1]
        hit = int(sum(p_value(v, normal_values) <= alpha_r for v in abnormal_values))
        false = int((leave_one_out_p_values(normal_values) <= alpha_r).sum())
        alternatives[name] = {"arrhythmia_caught": hit, "n_arrhythmia": int(abnormal_values.size),
                              "sensitivity_ci": list(clopper_pearson(hit, abnormal_values.size)),
                              "normal_flagged": false, "n_normal": int(normal_values.size)}
    part_a["rhythm"]["other_scores_same_level_not_preselected"] = alternatives

    # ---- B. the real paired healthy cohort (NInFEA) -----------------------------------------------------------
    x_train = np.array([[float(r[f]) for f in RECORD_FEATURES] for r in rhythm_rows])
    model = logistic().fit(x_train, labels)              # the rhythm expert fitted on all 26 NIFEADB recordings
    cohort_rows = []
    for row in function_rows:
        number = int(row["record"])
        item, trust = beats[str(number)], trust_rows[number]
        raw, tracked = np.array(item["raw_s"]), np.array(item["r_s"])
        edges = np.arange(0.0, item["duration_s"], 60.0)
        segments = []
        for start in edges:
            end = min(start + 60.0, item["duration_s"])
            if end - start >= 30.0 or len(edges) == 1:
                segments.append(segment_features(raw[(raw >= start) & (raw < end)], tracked[(tracked >= start) & (tracked < end)]))
        features = record_features(segments)
        readable = trust["ecg_ok"] == "True" and all(np.isfinite(features[f]) for f in RECORD_FEATURES)
        rhythm_score = float(model.predict_proba(np.array([[features[f] for f in RECORD_FEATURES]]))[0, 1]) if readable else float("nan")
        rhythm_p = p_value(rhythm_score, normal_scores) if readable else float("nan")
        rhythm = ExpertResult(present=True, quality_ok=readable, p_value=rhythm_p if readable else None)
        function = ExpertResult(present=True, quality_ok=np.isfinite(float(row["p_value"])), p_value=float(row["p_value"]))
        trusted = trust["trusted"] == "True"
        as_is = decide({"rhythm": rhythm, "function": function}, trusted, CONFIG)
        echo_normal = decide({"rhythm": rhythm, "function": function,
                              "structure": ExpertResult(present=True, p_value=1.0)}, trusted, CONFIG)
        cohort_rows.append({"record": number, "woman": row["woman"], "ecg_readable": readable, "rhythm_score": rhythm_score,
                            "rhythm_p": rhythm_p, "function_p": float(row["p_value"]), "trust_check_passed": trusted,
                            "action_echo_missing": as_is.action, "action_if_echo_normal": echo_normal.action,
                            "alarms": ";".join(echo_normal.alarms), "repeat": ";".join(echo_normal.repeat)})
    women = sorted({r["woman"] for r in cohort_rows})

    # The same recordings with the rhythm score calibrated where it is used: each recording is
    # ranked among the healthy NInFEA recordings of other women (leave-one-woman-out).
    local_scores = np.array([r["rhythm_score"] for r in cohort_rows])
    local_women = np.array([r["woman"] for r in cohort_rows])
    for r in cohort_rows:
        local_p = p_value(r["rhythm_score"], local_scores[local_women != r["woman"]]) if r["ecg_readable"] else float("nan")
        rhythm = ExpertResult(present=True, quality_ok=bool(r["ecg_readable"]), p_value=local_p if r["ecg_readable"] else None)
        function = ExpertResult(present=True, p_value=r["function_p"])
        local = decide({"rhythm": rhythm, "function": function, "structure": ExpertResult(present=True, p_value=1.0)},
                       bool(r["trust_check_passed"]), CONFIG)
        r["rhythm_p_local"], r["action_local_calibration"], r["alarms_local"] = local_p, local.action, ";".join(local.alarms)

    def tally(key: str) -> dict:
        counts = {action: int(sum(r[key] == action for r in cohort_rows)) for action in ("REFER", "REPEAT", "GET_TEST", "ROUTINE")}
        counts["n"] = len(cohort_rows)
        return counts

    referred = sum(r["action_if_echo_normal"] == "REFER" for r in cohort_rows)
    women_referred = sum(any(r["action_if_echo_normal"] == "REFER" for r in cohort_rows if r["woman"] == w) for w in women)
    local_referred = sum(r["action_local_calibration"] == "REFER" for r in cohort_rows)
    local_both = sum(("rhythm" in r["alarms_local"]) and ("function" in r["alarms_local"]) for r in cohort_rows)
    part_b = {
        "cohort": "NInFEA: 60 simultaneous fetal ECG and Doppler recordings of 39 healthy fetuses (real pairs)",
        "structure_expert": "test not done (no echo in NInFEA)",
        "actions_as_is_echo_missing": tally("action_echo_missing"),
        "actions_if_echo_were_done_and_normal": tally("action_if_echo_normal"),
        "false_referrals": {"recordings": int(referred), "n": len(cohort_rows), "share": referred / len(cohort_rows),
                            "ci": list(clopper_pearson(referred, len(cohort_rows))),
                            "women_with_a_referred_recording": int(women_referred), "n_women": len(women),
                            "by_expert": {"rhythm": int(sum("rhythm" in r["alarms"] for r in cohort_rows)),
                                          "function": int(sum("function" in r["alarms"] for r in cohort_rows))},
                            "ceiling_from_alarm_levels": alpha_r + alpha_f},
        "ecg_readable": int(sum(r["ecg_readable"] for r in cohort_rows)),
        "with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out": {
            "actions_if_echo_were_done_and_normal": tally("action_local_calibration"),
            "false_referrals": {"recordings": int(local_referred), "n": len(cohort_rows), "share": local_referred / len(cohort_rows),
                                "ci": list(clopper_pearson(local_referred, len(cohort_rows))),
                                "by_expert": {"rhythm": int(sum("rhythm" in r["alarms_local"] for r in cohort_rows)),
                                              "function": int(sum("function" in r["alarms_local"] for r in cohort_rows)),
                                              "both_on_the_same_recording": int(local_both)},
                                "ceiling_from_alarm_levels": alpha_r + alpha_f},
            "what_this_shows": "false referrals of the two experts on real pairs of healthy fetuses, with both calibrated on healthy cases from the same setting. Sensitivity of this calibration is unknown: NInFEA has no arrhythmia.",
        },
        "caveat": "The rhythm expert was fitted on NIFEADB (other device, later gestation, 60-second segments). Its alarms on NInFEA measure how it travels, not a tuned false-alarm rate.",
        "note": "'if echo were done and normal' replaces the missing echo by an assumed normal result to show the ECG-Doppler part alone; it is a stated assumption, not data.",
    }

    # ---- C. rule comparison on virtual patients built from real scores ------------------------------------------
    strength_normal, strength_abnormal, strength_function = 1.0 - p_normal, 1.0 - p_abnormal, 1.0 - function_p
    rng = np.random.default_rng(SEED)
    comparison = []
    for target in TARGET_RATES:
        for availability in ("both_tests", "mixed"):
            for coupling in COUPLINGS:
                point = compare_rules(strength_normal, strength_abnormal, strength_function, coupling, availability, target)
                draws = {name: [] for name in rules.RULES}
                women_all = np.unique(function_woman)
                for _ in range(N_BOOT):
                    n_index = rng.integers(0, strength_normal.size, strength_normal.size)
                    a_index = rng.integers(0, strength_abnormal.size, strength_abnormal.size)
                    chosen = rng.choice(women_all, women_all.size, replace=True)
                    f_sample = np.concatenate([strength_function[function_woman == w] for w in chosen])
                    sample = compare_rules(strength_normal[n_index], strength_abnormal[a_index], f_sample, coupling, availability, target)
                    for name in rules.RULES:
                        draws[name].append(sample[name]["sensitivity"])
                gap = np.array(draws["maximum"]) - np.array(draws["mean"])
                row = {"target_false_referral_rate": target, "availability": availability, "coupling": coupling,
                       "n_virtual_normal": point["n_virtual_normal"], "n_virtual_arrhythmia": point["n_virtual_arrhythmia"],
                       "n_real_scores": int(strength_normal.size + strength_abnormal.size + strength_function.size),
                       "max_minus_mean_sensitivity": point["maximum"]["sensitivity"] - point["mean"]["sensitivity"],
                       "max_minus_mean_ci_low": float(np.percentile(gap, 2.5)), "max_minus_mean_ci_high": float(np.percentile(gap, 97.5))}
                for name in rules.RULES:
                    row[f"{name}_sensitivity"] = point[name]["sensitivity"]
                    row[f"{name}_ci_low"] = float(np.percentile(draws[name], 2.5))
                    row[f"{name}_ci_high"] = float(np.percentile(draws[name], 97.5))
                    row[f"{name}_false_referral_rate"] = point[name]["false_referral_rate"]
                    row[f"{name}_sensitivity_with_doppler"] = point[name]["sensitivity_with_doppler"]
                    row[f"{name}_sensitivity_without_doppler"] = point[name]["sensitivity_without_doppler"]
                comparison.append(row)

    # the project's own rule: an alarm of either expert at its own level (no threshold fitted)
    own_rule = {}
    for coupling in COUPLINGS:
        n_r, n_f = pair(p_normal, function_p, coupling)
        a_r, a_f = pair(p_abnormal, function_p, coupling)
        own_rule[coupling] = {
            "false_referral_rate": float(np.mean((n_r <= alpha_r) | (n_f <= alpha_f))),
            "sensitivity_for_arrhythmia": float(np.mean((a_r <= alpha_r) | (a_f <= alpha_f))),
            "rhythm_expert_alone": float(np.mean(a_r <= alpha_r)),
        }
    bounds = {
        "ceiling_sum_of_levels": alpha_r + alpha_f,
        "false_referral_rate_by_coupling": {c: own_rule[c]["false_referral_rate"] for c in COUPLINGS},
        "ceiling_respected": all(own_rule[c]["false_referral_rate"] <= alpha_r + alpha_f + 1e-9 for c in COUPLINGS),
        "sensitivity_floor_respected": all(own_rule[c]["sensitivity_for_arrhythmia"] >= own_rule[c]["rhythm_expert_alone"] - 1e-12 for c in COUPLINGS),
    }
    n_virtual = int(sum(r["n_virtual_normal"] + r["n_virtual_arrhythmia"] for r in comparison))

    run = Run(
        "wp6_fusion",
        description="Decision rule on the real paired NInFEA cohort, conformal alarms of the experts, and a rule comparison on virtual patients built from real held-out scores",
        script="scripts/run_wp6_fusion.py",
        config={"decision": {"alpha": dict(CONFIG.alpha), "prior": dict(CONFIG.prior), "get_test_prior_limit": CONFIG.get_test_prior_limit},
                "rhythm_score": RHYTHM_SCORE, "source_runs": [RHYTHM_RUN, FUNCTION_RUN, TRUST_RUN], "couplings": COUPLINGS,
                "target_false_referral_rates": TARGET_RATES, "bootstrap": N_BOOT,
                "priors_are_assumptions": "structure 0.009 from the literature on CHD prevalence; function 0.002 and rhythm 0.001 are assumptions"},
        inputs={"rhythm": [RUNS / RHYTHM_RUN / "per_record_scores.csv"], "function": [RUNS / FUNCTION_RUN / "per_recording.csv"],
                "trust": [RUNS / TRUST_RUN / "per_recording.csv", RUNS / TRUST_RUN / "beats.json"]},
        n_real=int(labels.size + function_p.size),
        n_synthetic=n_virtual,
        synthetic_kind="virtual patients made by pairing real held-out scores of different people (part C only; coupling stated in every row). Parts A and B use real cases only.",
        seed=SEED,
    )
    run.write_csv("paired_cohort_ninfea.csv", cohort_rows)
    run.write_csv("rule_comparison.csv", comparison)

    mixed = [r for r in comparison if r["availability"] == "mixed"]
    lost = [r for r in mixed if r["max_minus_mean_sensitivity"] > 0]
    metrics = {
        "A_conformal_alarms": part_a, "B_real_paired_cohort": part_b,
        "C_rule_comparison": {"rows": len(comparison), "own_rule": own_rule, "bounds": bounds,
                              "note": "virtual patients; each row is one coupling, so the three rows together bracket the unknown joint behaviour"},
        "hypothesis_H7": {
            "statement": "with mixed test availability and one threshold, averaging-type rules lose alarms that the OR rule keeps at the same false-referral rate",
            "rows_checked": len(mixed), "rows_where_mean_rule_has_lower_sensitivity": len(lost),
            "rows_where_interval_excludes_zero": int(sum(r["max_minus_mean_ci_low"] > 0 for r in mixed)),
            "direction_as_predicted_in_all_rows": len(lost) == len(mixed),
            "established": bool(mixed) and all(r["max_minus_mean_ci_low"] > 0 for r in mixed),
        },
        "gate_G6": {"property_tests": "tests/test_fusion.py (19 tests)", "every_row_states_real_and_virtual_counts_and_coupling": True},
        "decision_config": {"alpha": dict(CONFIG.alpha), "prior": dict(CONFIG.prior), "get_test_prior_limit": CONFIG.get_test_prior_limit},
    }
    run.finish(metrics)

    print("A. conformal alarms:", json.dumps(part_a, indent=1, default=float))
    print("B. real paired cohort:", json.dumps(part_b, indent=1, default=float))
    print("C. own rule by coupling:", own_rule)
    print("   bounds:", bounds)
    print("   target  availability  coupling     " + "  ".join(f"{n[:9]:>9s}" for n in rules.RULES) + "   max-mean [CI]")
    for r in comparison:
        print(f"   {r['target_false_referral_rate']:.2f}    {r['availability']:11s}  {r['coupling']:11s}  "
              + "  ".join(f"{r[f'{n}_sensitivity']:9.3f}" for n in rules.RULES)
              + f"   {r['max_minus_mean_sensitivity']:+.3f} [{r['max_minus_mean_ci_low']:+.3f}, {r['max_minus_mean_ci_high']:+.3f}]")
    print("H7:", metrics["hypothesis_H7"])


if __name__ == "__main__":
    main()
