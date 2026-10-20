"""WP3b - simple baselines for the ECG-Doppler trust check.

The trust check (run wp3) asks whether the fetal ECG and the Doppler strip of one simultaneous
recording follow the same heart, beat by beat. It separated real pairs from pairs in which the
ECG beats were the mother's (AUROC 0.984 on the test women). A review on 20 October 2026 asked
whether that needs beat-level timing at all. A check made after seeing the results.

Baselines, fixed before computing:
  heart rate only      minus the absolute difference between the ECG rate and the Doppler rate
                       (median intervals, as in wp3). For the mother's beats the rate is the
                       maternal rate reported by the extractor
  pulse-train
  cross-correlation    the detected ECG beats and Doppler cycle starts as two trains of Gaussian
                       pulses (sd 10 ms, the check's tolerance), sampled every millisecond over the
                       common duration; the largest normalised cross-correlation over lags of
                       -400 to +400 ms. The Doppler envelope itself was not saved by wp3, so the
                       detected event times stand in for it. Not computed for the mother's beats,
                       whose times wp3 did not save

Comparisons, on the development women, the test women and all 39 women:
  real pairs against the mother's beats (maternal substitution): locked share vs heart rate
  real pairs against another woman's fetal ECG (swap): locked share vs heart rate vs
  cross-correlation
AUROC with a bootstrap interval over women; paired AUROC differences with the same women
resampled for both scores (2000 resamples, seed 0).

    python scripts/run_wp3b_trust_baselines.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import auroc, auroc_ci  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

SOURCE_RUN = "wp3_trust_check"
SIGMA_MS, MAX_LAG_MS, MIN_COMMON_S = 10.0, 400, 8.0
N_BOOT, SEED = 2000, 0
UNTESTED = [
    "both signals following the mother (Doppler gate on a maternal vessel and maternal ECG): they would agree and pass",
    "the other twin in a twin pregnancy: two fetal hearts at similar rates",
    "fetal arrhythmias with irregular beats or atrioventricular block (none in NInFEA): a correct pair may fail to lock",
    "recordings that are not simultaneous: the check cannot run, and the rule then asks for a repeat",
    "other ECG or ultrasound devices, sampling rates and Doppler gate placements than NInFEA's",
    "long periods of fetal movement or very low ECG quality beyond what the 60 recordings contain",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value) if value not in ("", "nan") else float("nan")


def pulse_train(times: np.ndarray, duration_s: float) -> np.ndarray:
    n = int(np.floor(duration_s * 1000)) + 1
    train = np.zeros(n)
    index = np.round(np.asarray(times) * 1000).astype(int)
    np.add.at(train, index[(index >= 0) & (index < n)], 1.0)
    return ndimage.gaussian_filter1d(train, SIGMA_MS)


def cross_correlation(r_s: np.ndarray, c_s: np.ndarray, duration_s: float) -> float:
    a, b = pulse_train(r_s, duration_s), pulse_train(c_s, duration_s)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    full = signal.correlate(b, a, mode="full", method="fft") / a.size
    middle = a.size - 1
    return float(full[middle - MAX_LAG_MS: middle + MAX_LAG_MS + 1].max())


def paired_difference(labels, a, b, women) -> dict:
    """AUROC(a) - AUROC(b), with whole women resampled for both scores at once."""
    labels, a, b, women = np.asarray(labels).astype(int), np.asarray(a, float), np.asarray(b, float), np.asarray(women)
    unique = np.unique(women)
    members = {w: np.flatnonzero(women == w) for w in unique}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        index = np.concatenate([members[w] for w in rng.choice(unique, unique.size, replace=True)])
        if labels[index].min() == labels[index].max():
            continue
        draws.append(auroc(labels[index], a[index]) - auroc(labels[index], b[index]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": auroc(labels, a) - auroc(labels, b), "ci": [float(low), float(high)],
            "n_resamples": len(draws), "can_be_ranked": bool(low > 0 or high < 0)}


def scored(labels, scores, women) -> dict:
    value, low, high = auroc_ci(labels, np.nan_to_num(np.asarray(scores, float), nan=-np.inf), groups=women, seed=SEED)
    return {"auroc": value, "ci": [low, high], "n_positive": int(np.sum(labels)), "n_negative": int(np.size(labels) - np.sum(labels))}


def main() -> None:
    load_run(SOURCE_RUN)
    rows = read_csv(RUNS / SOURCE_RUN / "per_recording.csv")
    swaps = read_csv(RUNS / SOURCE_RUN / "swaps.csv")
    beats = json.loads((RUNS / SOURCE_RUN / "beats.json").read_text(encoding="utf-8"))
    run = Run("wp3b_trust_baselines",
              description="Heart-rate-only and pulse-train cross-correlation baselines for the ECG-Doppler trust check "
                          "(check made after seeing the results)",
              script="scripts/run_wp3b_trust_baselines.py",
              config={"made_after_seeing_results": True, "requested_by": "review of 20 Oct 2026", "source_run": SOURCE_RUN,
                      "heart_rate_baseline": "minus |ECG rate - Doppler rate|; maternal rate from the extractor for the mother's beats",
                      "cross_correlation": {"pulse_sd_ms": SIGMA_MS, "lags_ms": [-MAX_LAG_MS, MAX_LAG_MS], "sampling": "1 ms",
                                            "events": "detected ECG beats and Doppler cycle starts saved by wp3"},
                      "bootstrap": {"resamples": N_BOOT, "unit": "woman", "seed": SEED}},
              inputs={"wp3": [RUNS / SOURCE_RUN / "per_recording.csv", RUNS / SOURCE_RUN / "swaps.csv", RUNS / SOURCE_RUN / "beats.json"]},
              n_real=len(rows), seed=SEED)

    genuine = []
    for r in rows:
        item = beats[r["record"]]
        r_s, c_s = np.array(item["r_s"]), np.array(item["closure_s"])
        genuine.append({"record": int(r["record"]), "woman": r["woman"], "split": r["split"],
                        "locked_share": number(r["locked_share"]), "maternal_locked_share": number(r["maternal_locked_share"]),
                        "rate_agreement": -abs(number(r["rate_ecg_bpm"]) - number(r["rate_doppler_bpm"])),
                        "maternal_rate_agreement": -abs(number(r["maternal_bpm"]) - number(r["rate_doppler_bpm"])),
                        "cross_correlation": cross_correlation(r_s, c_s, float(item["duration_s"]))})
    swap_rows = []
    for s in swaps:
        a, b = beats[s["doppler_record"]], beats[s["ecg_record"]]
        common = float(s["common_s"])
        r_s, c_s = np.array(b["r_s"]), np.array(a["closure_s"])
        swap_rows.append({"split": s["split"], "doppler_record": int(s["doppler_record"]), "ecg_record": int(s["ecg_record"]),
                          "doppler_woman": s["doppler_woman"], "locked_share": number(s["locked_share"]),
                          "rate_agreement": -number(s["rate_gap_bpm"]),
                          "cross_correlation": cross_correlation(r_s[r_s < common], c_s[c_s < common], common)})
    run.write_csv("genuine_scores.csv", genuine)
    run.write_csv("swap_scores.csv", swap_rows)

    results, table = {}, []
    for part in ("development", "test", "all"):
        mine = [g for g in genuine if part in ("all", g["split"])]
        theirs = [s for s in swap_rows if part in ("all", s["split"])]
        women_g = [g["woman"] for g in mine]
        # maternal substitution: real pair = 1, the mother's beats with the same Doppler = 0
        labels = np.r_[np.ones(len(mine)), np.zeros(len(mine))]
        women = np.r_[women_g, women_g]
        locked = np.nan_to_num(np.r_[[g["locked_share"] for g in mine], [g["maternal_locked_share"] for g in mine]])
        rate = np.r_[[g["rate_agreement"] for g in mine], [g["maternal_rate_agreement"] for g in mine]]
        rate = np.nan_to_num(rate, nan=-np.inf)
        maternal = {"locked_share": scored(labels, locked, women), "heart_rate_only": scored(labels, rate, women),
                    "locked_minus_heart_rate": paired_difference(labels, locked, np.nan_to_num(rate, neginf=-1e6), women)}
        # swap: real pair = 1, another woman's fetal ECG with this Doppler = 0
        labels = np.r_[np.ones(len(mine)), np.zeros(len(theirs))]
        women = np.r_[women_g, [s["doppler_woman"] for s in theirs]]
        locked = np.nan_to_num(np.r_[[g["locked_share"] for g in mine], [s["locked_share"] for s in theirs]])
        rate = np.nan_to_num(np.r_[[g["rate_agreement"] for g in mine], [s["rate_agreement"] for s in theirs]], nan=-1e6)
        xcorr = np.nan_to_num(np.r_[[g["cross_correlation"] for g in mine], [s["cross_correlation"] for s in theirs]], nan=-1e6)
        swap = {"locked_share": scored(labels, locked, women), "heart_rate_only": scored(labels, rate, women),
                "cross_correlation": scored(labels, xcorr, women),
                "locked_minus_heart_rate": paired_difference(labels, locked, rate, women),
                "locked_minus_cross_correlation": paired_difference(labels, locked, xcorr, women)}
        results[part] = {"n_recordings": len(mine), "n_women": len(set(women_g)), "n_swaps": len(theirs),
                         "maternal_substitution": maternal, "swap": swap}
        for mismatch, block in (("mother's beats", maternal), ("another woman's fetal ECG", swap)):
            for score, value in block.items():
                if "minus" in score:
                    table.append({"women": part, "mismatch": mismatch, "row": score.replace("_", " "), "auroc": "", "ci_low": value["ci"][0],
                                  "ci_high": value["ci"][1], "difference": value["difference"], "can_be_ranked": value["can_be_ranked"]})
                else:
                    table.append({"women": part, "mismatch": mismatch, "row": score.replace("_", " "), "auroc": value["auroc"],
                                  "ci_low": value["ci"][0], "ci_high": value["ci"][1], "difference": "", "can_be_ranked": ""})
    run.write_csv("baselines.csv", table)
    published = load_run(SOURCE_RUN)[0]["gate_G3"]
    metrics = {
        "made_after_seeing_results": True,
        "by_women": results,
        "published_maternal_auroc_test": published["auroc_maternal_test"],
        "locked_share_matches_published": bool(np.isclose(results["test"]["maternal_substitution"]["locked_share"]["auroc"],
                                                          published["auroc_maternal_test"])),
        "gate_G3_unchanged": {"beat_level_passed": published["beat_level_passed"], "note": "this run does not change gate G3, which stays not met"},
        "untested_failure_modes": UNTESTED,
        "wording": ("a consistency (substitution) check between two simultaneous recordings of the same heart, not a measure of "
                    "diagnostic accuracy; using Doppler as the reference for fetal ECG timing is established practice, not new"),
    }
    run.finish(metrics)
    for row in table:
        if row["auroc"] != "":
            print(f"{row['women']:12s} {row['mismatch']:26s} {row['row']:24s} AUROC {row['auroc']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
        else:
            print(f"{row['women']:12s} {row['mismatch']:26s} {row['row']:24s} diff {row['difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
