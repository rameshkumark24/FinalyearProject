"""WP13 - how many local healthy recordings the rhythm expert needs before its alarm level holds.

Run wp6 found that the rhythm expert, fitted and calibrated on NIFEADB (one device), alarmed on
about half of the healthy NInFEA recordings (another device, earlier gestation), while the same
scores calibrated on healthy NInFEA recordings of other women alarmed on about one in ten. This run
turns that single comparison into a curve. Added on 20 October 2026 in answer to a review; a check
made after seeing the results. Nothing is refitted: the scores saved by wp6 and wp4b are reused.

One device pair only (NIFEADB to NInFEA) and only healthy fetuses: the curve says how many local
healthy recordings keep false alarms near the level; it says nothing about sensitivity after local
calibration, because NInFEA has no arrhythmia.

Settings, fixed before computing:
  level            0.10, the rhythm level of the decision rule (its effective level depends on the
                   number of calibration recordings: floor(0.10 (n + 1)) / (n + 1))
  calibrations     transported: the 14 healthy NIFEADB recordings only (as published);
                   local: healthy NInFEA recordings of other women only (site-conditional);
                   pooled: the 14 NIFEADB recordings plus the local ones
  learning curve   1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25 and 30 calibration women out of the 37 with a
                   readable fetal ECG; 500 random draws each (seed 0); all readable recordings of the
                   chosen women calibrate, the recordings of the other women are tested (no woman on
                   both sides)
  tolerance        "holds" = the expert can alarm (every calibration set has at least 9 recordings,
                   so that its smallest p-value 1/(n+1) is at most 0.10) and the mean realised
                   false-alarm rate over the draws is at most 0.10 + 0.05; the smallest number of
                   women (and their median number of recordings) for which this is true is reported.
                   Revised after a trial run: the first criterion (rate within 0.15 in 90% of draws)
                   counted draws in which the expert could not alarm at all, and with only a few test
                   women left a single draw's rate is mostly test-set noise. The 5th and 95th
                   percentiles over draws are still reported, next to the range expected from the
                   calibration set alone (Beta law, 90%)
  shift            Kolmogorov-Smirnov distance between the healthy scores of the two datasets, with
                   a bootstrap interval (NIFEADB by recording, NInFEA by woman, 2000 resamples). At
                   any fixed threshold the false-alarm rate on the new data exceeds the calibration
                   rate by at most this distance

    python scripts/run_wp13_calibration_transfer.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import auroc  # noqa: E402
from sfr.fusion.conformal import p_value, realised_false_alarm_range  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfrtool.allocation import effective_level  # noqa: E402

COHORT_RUN, RHYTHM_RUN = "wp6_fusion", "wp4b_rhythm_intervals"
RHYTHM_SCORE = "oof_logistic_rhythm"
LEVEL, TOLERANCE = 0.10, 0.05
MIN_TO_ALARM = int(np.ceil(1 / LEVEL)) - 1                         # 9 calibration recordings
N_WOMEN_GRID = (1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30)
N_DRAWS, N_BOOT, SEED = 500, 2000, 0


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def false_alarm_rate(test_scores: np.ndarray, calibration: np.ndarray) -> float:
    return float(np.mean([p_value(float(s), calibration) <= LEVEL + 1e-12 for s in test_scores]))


def ks_with_interval(a: np.ndarray, b: np.ndarray, b_groups: np.ndarray, rng: np.random.Generator) -> dict:
    point = float(stats.ks_2samp(a, b).statistic)
    unique = np.unique(b_groups)
    members = {g: np.flatnonzero(b_groups == g) for g in unique}
    draws = []
    for _ in range(N_BOOT):
        a_star = a[rng.integers(0, a.size, a.size)]
        b_star = b[np.concatenate([members[g] for g in rng.choice(unique, unique.size, replace=True)])]
        draws.append(float(stats.ks_2samp(a_star, b_star).statistic))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"distance": point, "ci": [float(low), float(high)]}


def main() -> None:
    load_run(COHORT_RUN)
    load_run(RHYTHM_RUN)
    cohort = [r for r in read_csv(RUNS / COHORT_RUN / "paired_cohort_ninfea.csv") if r["ecg_readable"] == "True"]
    scores = np.array([float(r["rhythm_score"]) for r in cohort])
    women = np.array([r["woman"] for r in cohort])
    published_p = np.array([float(r["rhythm_p"]) for r in cohort])
    rhythm_rows = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    published = np.array([float(r[RHYTHM_SCORE]) for r in rhythm_rows if r["label"] == "0"])
    if published.size != 14:
        raise ValueError("expected the 14 healthy NIFEADB recordings")
    transported_p = np.array([p_value(float(s), published) for s in scores])
    if not np.allclose(transported_p, published_p, atol=5e-7):
        raise ValueError("the transported p-values differ from run wp6")
    unique_women = np.unique(women)

    run = Run("wp13_calibration_transfer",
              description="Rhythm false-alarm rate on healthy NInFEA recordings against the number of local calibration recordings "
                          "(transported, local and pooled calibration). A check made after seeing the results.",
              script="scripts/run_wp13_calibration_transfer.py",
              config={"made_after_seeing_results": True, "requested_by": "review of 20 Oct 2026", "level": LEVEL,
                      "tolerance": TOLERANCE, "holds": "can alarm in every draw and mean realised rate <= level + tolerance",
                      "criterion_revised_after_trial_run": "first criterion: realised rate <= level + tolerance in 90% of draws",
                      "minimum_calibration_recordings_to_alarm": MIN_TO_ALARM, "n_women_grid": list(N_WOMEN_GRID),
                      "draws": N_DRAWS, "bootstrap": N_BOOT, "seed": SEED, "device_pair": "NIFEADB (fitting, calibration) to NInFEA (use)",
                      "unit_of_sampling": "woman"},
              inputs={"cohort": [RUNS / COHORT_RUN / "paired_cohort_ninfea.csv"], "nifeadb_scores": [RUNS / RHYTHM_RUN / "per_record_scores.csv"]},
              n_real=int(scores.size + published.size), seed=SEED)

    rng = np.random.default_rng(SEED)
    curve, draws_out = [], []
    for n_women in N_WOMEN_GRID:
        local_rates, pooled_rates, sizes = [], [], []
        for draw in range(N_DRAWS):
            chosen = rng.choice(unique_women, n_women, replace=False)
            calibrate = np.isin(women, chosen)
            local = scores[calibrate]
            test = scores[~calibrate]
            local_rates.append(false_alarm_rate(test, local))
            pooled_rates.append(false_alarm_rate(test, np.concatenate([published, local])))
            sizes.append(int(calibrate.sum()))
        local_rates, pooled_rates, sizes = np.array(local_rates), np.array(pooled_rates), np.array(sizes)
        row = {"calibration_women": n_women, "calibration_recordings_median": float(np.median(sizes)),
               "calibration_recordings_min": int(sizes.min()), "calibration_recordings_max": int(sizes.max()),
               "effective_level_at_median_size": effective_level(LEVEL, int(np.median(sizes)))}
        row["can_alarm_in_every_draw"] = bool(sizes.min() >= MIN_TO_ALARM)
        row["beta_range_90_at_median_size"] = (list(realised_false_alarm_range(int(np.median(sizes)), LEVEL))
                                               if np.median(sizes) >= MIN_TO_ALARM else [0.0, 0.0])
        for name, rates in (("local", local_rates), ("pooled", pooled_rates)):
            row[f"{name}_mean"] = float(rates.mean())
            row[f"{name}_p5"], row[f"{name}_p95"] = (float(v) for v in np.percentile(rates, [5, 95]))
            can_alarm = row["can_alarm_in_every_draw"] or name == "pooled"      # pooled sets always hold the 14 NIFEADB recordings
            row[f"{name}_holds"] = bool(can_alarm and row[f"{name}_mean"] <= LEVEL + TOLERANCE + 1e-12)
        curve.append(row)
        print(f"{n_women:2d} women (median {np.median(sizes):.0f} recordings): local mean {local_rates.mean():.3f} "
              f"[{np.percentile(local_rates, 5):.3f}, {np.percentile(local_rates, 95):.3f}]  pooled mean {pooled_rates.mean():.3f}", flush=True)

    transported_rate = float(np.mean(transported_p <= LEVEL + 1e-12))
    leave_one_woman_out = np.array([p_value(float(s), scores[women != w]) for s, w in zip(scores, women)])
    smallest = {}
    for name in ("local", "pooled"):
        holding = [row for row in curve if row[f"{name}_holds"]]
        smallest[name] = ({"calibration_women": holding[0]["calibration_women"],
                           "calibration_recordings_median": holding[0]["calibration_recordings_median"]}
                          if holding else "not reached on this grid")
    shift = ks_with_interval(published, scores, women, np.random.default_rng(SEED))
    shift["auroc_ninfea_above_nifeadb"] = auroc(np.r_[np.ones(scores.size), np.zeros(published.size)], np.r_[scores, published])
    shift["observed_excess_false_alarms"] = transported_rate - effective_level(LEVEL, published.size)
    shift["bound_note"] = "at a fixed threshold, false alarms on the new data exceed those on the calibration data by at most the KS distance"
    metrics = {
        "made_after_seeing_results": True,
        "n_recordings_readable": int(scores.size), "n_women_readable": int(unique_women.size),
        "transported": {"calibration": "14 healthy NIFEADB recordings", "false_alarm_rate": transported_rate,
                        "flagged": int(np.sum(transported_p <= LEVEL + 1e-12)), "n": int(scores.size),
                        "effective_level": effective_level(LEVEL, published.size)},
        "local_leave_one_woman_out": {"false_alarm_rate": float(np.mean(leave_one_woman_out <= LEVEL + 1e-12)),
                                      "flagged": int(np.sum(leave_one_woman_out <= LEVEL + 1e-12)), "n": int(scores.size)},
        "curve": curve, "smallest_calibration_that_holds": smallest, "shift": shift,
        "scope": "one device pair (NIFEADB to NInFEA), 57 readable recordings of 37 healthy fetuses; sensitivity after local calibration "
                 "is unknown because NInFEA has no arrhythmia",
    }
    run.write_csv("learning_curve.csv", curve)
    run.finish(metrics)
    print("transported", metrics["transported"])
    print("local, leave one woman out", metrics["local_leave_one_woman_out"])
    print("smallest that holds", smallest)
    print("shift", shift)


if __name__ == "__main__":
    main()
