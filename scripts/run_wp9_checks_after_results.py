"""WP9 - checks made after seeing the results, from saved outputs only (nothing is re-fitted).

Added on 16 October 2026 in answer to a review. Every number here re-summarises per-case
outputs that earlier runs already saved; no model is fitted and no signal is re-extracted.

A. Rhythm, 1 kHz recordings only. The four NIFEADB recordings sampled at 500 Hz are all
   arrhythmia recordings, so the sampling rate could carry part of the label. The AUROC of the
   same out-of-fold scores is given without them, with a bootstrap interval over subjects. The
   sampling rate of every recording is read from its header, not from a list.
B. Paired comparisons of rhythm scores: AUROC differences on the same subjects, with a paired
   bootstrap interval. Two scores whose difference interval contains 0 cannot be ranked.
C. Trust check, robust agreement. The beat-to-beat differences of the held-out recordings that
   pass the ECG quality check, recomputed from the beat times saved by wp3 (rounded to 0.1 ms):
   median, 90th and 95th percentile of the absolute difference, and percentile limits of
   agreement next to the classical limits. Gate G3 keeps its pre-specified rule.
D. Subgroups. Structure: alarms on normal and abnormal cases by image size (DELTA has no field
   for the device or the gestational age; image size is the only acquisition variable).
   Function: alarms on healthy recordings by gestational age, and the correlation of the score
   with gestational age (the reference adjusts cycle length for it).
E. Structural defects the expert missed at its level (case identifiers of the public dataset).

    python scripts/run_wp9_checks_after_results.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data import nifeadb  # noqa: E402
from sfr.eval.stats import auroc, auroc_ci, bland_altman, clopper_pearson  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import NIFEADB_DIR, RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.trust.crossmodal import paired_intervals, plausible_intervals  # noqa: E402

INTERVALS_RUN, FOUNDATION_RUN = "wp4b_rhythm_intervals", "wp4c_foundation"
TRUST_RUN, FUNCTION_RUN, STRUCTURE_RUN = "wp3_trust_check", "wp2b_function_reference", "wp5_structure_delta"
CONFIG = DecisionConfig()
N_BOOT, SEED = 2000, 0
RHYTHM_SCORES = (
    (INTERVALS_RUN, "oof_logistic_rhythm", "rhythm expert: interval features, logistic"),
    (INTERVALS_RUN, "oof_forest_rhythm", "interval features, random forest"),
    (INTERVALS_RUN, "succ_rel_median", "single index: successive interval change"),
    (INTERVALS_RUN, "hr_bpm_median", "baseline: heart rate only (higher = arrhythmia)"),
    ("derived", "slow_hr_bpm_median", "baseline: heart rate only, lower = arrhythmia (direction chosen after seeing the data)"),
    (FOUNDATION_RUN, "oof_ecgfounder__fetal", "ECGFounder, fetal input"),
    (FOUNDATION_RUN, "oof_ecgfounder__fetal_stretch2", "ECGFounder, fetal input stretched"),
    (FOUNDATION_RUN, "oof_random_init__fetal", "random weights, fetal input"),
    (FOUNDATION_RUN, "oof_random_init__fetal_stretch2", "random weights, fetal input stretched"),
    (FOUNDATION_RUN, "oof_ecgfounder__maternal_chest", "ECGFounder, mother's chest lead (control)"),
)
PAIRS = (
    ("oof_ecgfounder__fetal", "oof_random_init__fetal_stretch2"),
    ("oof_ecgfounder__fetal", "oof_random_init__fetal"),
    ("oof_ecgfounder__fetal", "oof_logistic_rhythm"),
    ("oof_logistic_rhythm", "succ_rel_median"),
    ("oof_logistic_rhythm", "slow_hr_bpm_median"),
    ("oof_ecgfounder__fetal", "slow_hr_bpm_median"),
)
GESTATION_BINS_WEEKS = ((21, 23), (23, 25), (25, 27), (27, 29))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_auroc_difference(labels, a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """AUROC(a) - AUROC(b) on the same subjects, with a percentile bootstrap interval over subjects."""
    labels, a, b = np.asarray(labels).astype(int), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, labels.size, labels.size)
        if labels[index].min() == labels[index].max():
            continue
        draws.append(auroc(labels[index], a[index]) - auroc(labels[index], b[index]))
    draws = np.array(draws)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": auroc(labels, a) - auroc(labels, b), "ci": [float(low), float(high)],
            "share_of_resamples_above_0": float(np.mean(draws > 0)), "n_resamples": int(draws.size),
            "can_be_ranked": bool(low > 0 or high < 0)}


def proportion(k: int, n: int) -> dict:
    low, high = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "share": k / n if n else float("nan"), "ci": [low, high]}


# ---- A and B ------------------------------------------------------------------------------------------------

def rhythm_checks() -> tuple[dict, list[dict], list[dict]]:
    by_run = {run: {r["record"]: r for r in read_csv(RUNS / run / "per_record_scores.csv")} for run in (INTERVALS_RUN, FOUNDATION_RUN)}
    records = list(nifeadb.ALL_RECORDS)
    labels = np.array([nifeadb.label_of(name) for name in records])
    for run, rows in by_run.items():
        if sorted(rows) != sorted(records) or any(int(rows[name]["label"]) != label for name, label in zip(records, labels)):
            raise ValueError(f"{run}: records or labels differ from the NIFEADB loader")
    rates = {name: float(wfdb.rdheader(str(NIFEADB_DIR / name)).fs) for name in records}
    one_khz = np.array([round(rates[name]) == 1000 for name in records])
    rate_table = {f"{int(round(fs))} Hz": {"arrhythmia": int(sum(1 for n, l in zip(records, labels) if rates[n] == fs and l == 1)),
                                           "normal": int(sum(1 for n, l in zip(records, labels) if rates[n] == fs and l == 0))}
                  for fs in sorted(set(rates.values()))}
    scores = {key: np.array([float(by_run[run][name][key]) for name in records]) for run, key, _ in RHYTHM_SCORES if run != "derived"}
    scores["slow_hr_bpm_median"] = -scores["hr_bpm_median"]

    # The per-record files hold scores rounded to 6 decimals. Rounding can tie scores of the two
    # classes (several network outputs round to 1.0), which changes the AUROC. A score is used
    # here only if its AUROC over all 26 recordings equals the published value exactly: then no
    # tie crosses the classes, and every subset and resample is exact as well.
    m_int, m_found = load_run(INTERVALS_RUN)[0], load_run(FOUNDATION_RUN)[0]
    published = {f"oof_{k}": v["auroc"] for k, v in m_int["recording_level"].items()}
    published |= {k: v["auroc"] for k, v in m_int["single_index"].items()}
    published |= {f"oof_{k}": v["auroc"] for k, v in m_found["arrhythmia_probe_nifeadb"].items()}
    published["slow_hr_bpm_median"] = 1 - published["hr_bpm_median"]
    published_1khz = {f"oof_{k}": v["auroc_without_500hz_recordings"] for k, v in m_found["sampling_rate_check_added_after_first_run"].items()}
    exact = {key: abs(auroc(labels, scores[key]) - published[key]) < 1e-9 for _, key, _ in RHYTHM_SCORES}

    rows = []
    for run, key, title in RHYTHM_SCORES:
        row = {"score": key, "description": title, "run": run, "published_auroc_all": published[key], "exact_from_saved_file": exact[key]}
        if exact[key]:
            value, low, high = auroc_ci(labels, scores[key], seed=SEED)
            value_1k, low_1k, high_1k = auroc_ci(labels[one_khz], scores[key][one_khz], seed=SEED)
            if key in published_1khz:
                assert abs(published_1khz[key] - value_1k) < 1e-9
            row.update({"n_all": int(labels.size), "auroc_all": value, "ci_all_low": low, "ci_all_high": high, "n_1khz": int(one_khz.sum()),
                        "arrhythmia_1khz": int(labels[one_khz].sum()), "auroc_1khz": value_1k, "ci_1khz_low": low_1k, "ci_1khz_high": high_1k})
        else:
            row["note"] = "not recomputed: rounding in the saved file ties scores of the two classes"
        rows.append(row)
    pair_rows = []
    for subset, keep in (("all 26 recordings", np.ones(labels.size, dtype=bool)), ("1 kHz recordings only", one_khz)):
        for first, second in PAIRS:
            row = {"subset": subset, "first": first, "second": second, "n": int(keep.sum())}
            if exact[first] and exact[second]:
                result = paired_auroc_difference(labels[keep], scores[first][keep], scores[second][keep])
                row.update({"difference": result["difference"], "ci_low": result["ci"][0], "ci_high": result["ci"][1],
                            "share_above_0": result["share_of_resamples_above_0"], "resamples": result["n_resamples"],
                            "can_be_ranked": result["can_be_ranked"]})
            else:
                row["note"] = "not recomputed: rounding in the saved file ties scores of the two classes"
            pair_rows.append(row)
    return {"sampling_rates_from_headers": rate_table, "records_500hz": [n for n in records if round(rates[n]) != 1000],
            "scores_not_recomputable_from_saved_files": [key for key, ok in exact.items() if not ok]}, rows, pair_rows


# ---- C --------------------------------------------------------------------------------------------------

def trust_agreement() -> dict:
    metrics, _ = load_run(TRUST_RUN)
    beats = json.loads((RUNS / TRUST_RUN / "beats.json").read_text(encoding="utf-8"))
    chosen = [r for r in read_csv(RUNS / TRUST_RUN / "per_recording.csv") if r["split"] == "test" and r["ecg_ok"] == "True"]
    diff_ms, rate_diff, keep = [], [], []
    for row in chosen:
        r = np.array(beats[row["record"]]["r_s"])
        c = np.array(beats[row["record"]]["closure_s"])
        pairs = paired_intervals(r, c)
        if pairs["cc"].size == 0:
            continue
        rr_ok, cc_ok = plausible_intervals(np.diff(r)), plausible_intervals(np.diff(c))
        middle = 0.5 * (c[:-1] + c[1:])
        which_c = np.clip(np.searchsorted(middle, pairs["t"]), 0, cc_ok.size - 1)
        which_r = np.clip(np.searchsorted(r, pairs["t"], side="right") - 1, 0, rr_ok.size - 1)
        diff_ms += list(1000.0 * (pairs["cc"] - pairs["rr"]))
        rate_diff += list(60.0 / pairs["cc"] - 60.0 / pairs["rr"])
        keep += list(cc_ok[which_c] & rr_ok[which_r])
    diff_ms, rate_diff, keep = np.array(diff_ms), np.array(rate_diff), np.array(keep, dtype=bool)
    stored = metrics["by_split"]["test"]["beat_level_ecg_quality_passed_recordings"]
    plausible = rate_diff[keep]
    result = {
        "recordings": len(chosen),
        "pairs": int(diff_ms.size), "pairs_stored_by_wp3": stored["n_pairs"],
        "plausible_pairs": int(keep.sum()),
        "abs_interval_difference_ms": {q: float(np.percentile(np.abs(diff_ms), q)) for q in (50, 90, 95)},
        "abs_interval_difference_median_stored_by_wp3_ms": stored["abs_diff_median_ms"],
        "abs_rate_difference_bpm_plausible_pairs": {q: float(np.percentile(np.abs(plausible), q)) for q in (50, 90, 95)},
        "percentile_limits_bpm_plausible_pairs": [float(np.percentile(plausible, 2.5)), float(np.percentile(plausible, 97.5))],
        "percentile_limits_stored_by_wp3_bpm": [stored["rate_diff_percentiles_plausible_bpm"]["2.5"], stored["rate_diff_percentiles_plausible_bpm"]["97.5"]],
        "classical_limits_bpm_plausible_pairs": bland_altman(plausible, np.zeros(plausible.size)),
        "share_within_10_bpm_plausible_pairs": float(np.mean(np.abs(plausible) <= 10)),
        "gate_G3_target_bpm": 10,
        "note": "beat times were saved rounded to 0.1 ms, so recomputed values can differ from wp3 in the last digit",
    }
    if result["pairs"] != stored["n_pairs"] or abs(result["abs_interval_difference_ms"][50] - stored["abs_diff_median_ms"]) > 0.2:
        raise ValueError("the recomputed beat pairs do not match those of wp3")
    return result


# ---- D and E --------------------------------------------------------------------------------------------

def subgroups() -> tuple[list[dict], list[dict], dict, list[dict]]:
    cases = read_csv(RUNS / STRUCTURE_RUN / "per_case.csv")
    alpha_s = CONFIG.alpha["structure"]
    size_rows = []
    for size in sorted({c["image_size"] for c in cases}, key=lambda s: -sum(c["image_size"] == s for c in cases)):
        member = [c for c in cases if c["image_size"] == size]
        normal = [c for c in member if c["label"] == "0"]
        abnormal = [c for c in member if c["label"] == "1"]
        fa = proportion(sum(float(c["p_value"]) <= alpha_s for c in normal), len(normal))
        se = proportion(sum(float(c["p_value"]) <= alpha_s for c in abnormal), len(abnormal))
        size_rows.append({"image_size": size, "normal": len(normal), "normal_flagged": fa["k"], "false_alarm_rate": fa["share"],
                          "fa_ci_low": fa["ci"][0], "fa_ci_high": fa["ci"][1], "abnormal": len(abnormal), "abnormal_caught": se["k"],
                          "sensitivity": se["share"], "se_ci_low": se["ci"][0], "se_ci_high": se["ci"][1]})
    missed = [{"case": c["case"], "subtype": c["subtype"], "split": c["split"], "image_size": c["image_size"], "p_value": float(c["p_value"])}
              for c in cases if c["label"] == "1" and float(c["p_value"]) > alpha_s]

    recordings = read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")
    alpha_f = CONFIG.alpha["function"]
    week_rows = []
    for low, high in GESTATION_BINS_WEEKS:
        member = [r for r in recordings if low * 7 <= int(r["gestation_days"]) < high * 7]
        women = {r["woman"] for r in member}
        flagged = proportion(sum(float(r["p_value"]) <= alpha_f for r in member), len(member))
        week_rows.append({"gestation_weeks": f"{low}+0 to {high - 1}+6", "recordings": len(member), "women": len(women),
                          "flagged": flagged["k"], "share": flagged["share"], "ci_low": flagged["ci"][0], "ci_high": flagged["ci"][1]})
    days = np.array([float(r["gestation_days"]) for r in recordings])
    score = np.array([float(r["score"]) for r in recordings])
    finite = np.isfinite(score)
    rho, p = stats.spearmanr(days[finite], score[finite])
    correlation = {"spearman_score_vs_gestation_days": float(rho), "p_value": float(p), "n": int(finite.sum()),
                   "note": "repeated recordings of a woman share her gestational age, so the recordings are not independent"}
    return size_rows, week_rows, correlation, missed


def main() -> None:
    for name in (INTERVALS_RUN, FOUNDATION_RUN, TRUST_RUN, FUNCTION_RUN, STRUCTURE_RUN):
        load_run(name)
    sampling, rhythm_rows, pair_rows = rhythm_checks()
    agreement = trust_agreement()
    size_rows, week_rows, correlation, missed = subgroups()
    metrics = {
        "made_after_seeing_results": True,
        "A_rhythm_1khz_only": {**sampling, "table": "rhythm_1khz_only.csv",
                               "note": "same out-of-fold scores as wp4b and wp4c (models fitted on all 26); nothing re-fitted"},
        "B_rhythm_paired_differences": {"table": "rhythm_paired_differences.csv", "bootstrap": N_BOOT,
                                        "rows": pair_rows},
        "C_trust_check_robust_agreement": agreement,
        "D_subgroups": {"structure_by_image_size": size_rows, "function_by_gestational_age": week_rows,
                        "function_score_and_gestational_age": correlation,
                        "not_available": "DELTA records neither the device nor the gestational age, and NInFEA was recorded with one device"},
        "E_structure_missed": {"n_missed": len(missed), "n_abnormal": sum(r["abnormal"] for r in size_rows), "table": "structure_missed_cases.csv"},
    }
    inputs = {"rhythm": [RUNS / INTERVALS_RUN / "per_record_scores.csv", RUNS / FOUNDATION_RUN / "per_record_scores.csv",
                         *[NIFEADB_DIR / f"{name}.hea" for name in nifeadb.ALL_RECORDS]],
              "trust": [RUNS / TRUST_RUN / "beats.json", RUNS / TRUST_RUN / "per_recording.csv"],
              "function": [RUNS / FUNCTION_RUN / "per_recording.csv"], "structure": [RUNS / STRUCTURE_RUN / "per_case.csv"]}
    run = Run("wp9_checks_after_results",
              description="Checks made after seeing the results: rhythm AUROC on 1 kHz recordings only, paired differences of rhythm scores, "
                          "robust ECG-Doppler agreement, alarms by subgroup, missed structural defects. Saved outputs only; nothing re-fitted.",
              script="scripts/run_wp9_checks_after_results.py",
              config={"source_runs": [INTERVALS_RUN, FOUNDATION_RUN, TRUST_RUN, FUNCTION_RUN, STRUCTURE_RUN], "bootstrap": N_BOOT,
                      "gestation_bins_weeks": [list(b) for b in GESTATION_BINS_WEEKS], "alpha": dict(CONFIG.alpha), "made_after_seeing_results": True},
              inputs=inputs, n_real=len(nifeadb.ALL_RECORDS) + agreement["recordings"] + sum(r["recordings"] for r in week_rows)
              + sum(r["normal"] + r["abnormal"] for r in size_rows), seed=SEED)
    run.write_csv("rhythm_1khz_only.csv", rhythm_rows)
    run.write_csv("rhythm_paired_differences.csv", pair_rows)
    run.write_csv("structure_by_image_size.csv", size_rows)
    run.write_csv("function_by_gestational_age.csv", week_rows)
    run.write_csv("structure_missed_cases.csv", missed)
    run.finish(metrics)

    print("sampling rates:", sampling["sampling_rates_from_headers"], "500 Hz:", sampling["records_500hz"])
    for row in rhythm_rows:
        if not row["exact_from_saved_file"]:
            print(f"  {row['score']:34s} {row['note']}")
            continue
        print(f"  {row['score']:34s} all {row['auroc_all']:.3f} [{row['ci_all_low']:.3f}, {row['ci_all_high']:.3f}]   "
              f"1 kHz {row['auroc_1khz']:.3f} [{row['ci_1khz_low']:.3f}, {row['ci_1khz_high']:.3f}] (n {row['n_1khz']})")
    for row in pair_rows:
        if "difference" not in row:
            print(f"  {row['subset']:22s} {row['first']} - {row['second']}: {row['note']}")
            continue
        print(f"  {row['subset']:22s} {row['first']} - {row['second']}: {row['difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
    print("agreement:", json.dumps({k: v for k, v in agreement.items() if k != "classical_limits_bpm_plausible_pairs"}, indent=1))
    for row in size_rows:
        print("  size", row["image_size"], f"normal {row['normal_flagged']}/{row['normal']}", f"abnormal {row['abnormal_caught']}/{row['abnormal']}")
    for row in week_rows:
        print("  weeks", row["gestation_weeks"], f"{row['flagged']}/{row['recordings']}")
    print("  correlation", correlation, " missed", len(missed))


if __name__ == "__main__":
    main()
