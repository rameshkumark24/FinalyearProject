"""WP17 - does the structure expert's alarm level travel to other hospitals, and how many local healthy
fetuses repair it? With heart defects in the data, detection after local calibration is measured too.

The rhythm expert's alarm level did not travel from NIFEADB to NInFEA (runs wp6, wp13), but NInFEA
holds healthy fetuses only, so detection after local calibration was unknown. Heartbeat and CARDIUM
hold fetuses with and without CHD, so here both sides are measured for the frozen DELTA structure
expert. Exploratory (after the preregistered external test); nothing is refitted: the scores are
the ones the preregistered runs saved (runs wp11_external_echo_*, private files).

Settings, fixed before computing:
  datasets         Heartbeat 2nd trimester, Heartbeat 3rd trimester, CARDIUM (all images); each alone
  level            0.05, the structure level of the decision rule
  transported      the 62 DELTA calibration scores of the structure artifact
  local            0, 5, 10, 19, 25, 40, 60, 100, 150, 200 and 300 healthy patients drawn at random as
                   the calibration set (500 draws each, seed 0); false alarms are counted on the other
                   healthy patients and detection on every patient with CHD (never used to calibrate)
  pooled           the 62 DELTA scores plus the local ones
  holds            the expert can alarm in every draw (at least 19 calibration scores) and the mean false
                   alarm rate is at most 0.05 + 0.05; the smallest local size that holds is reported
  shift            Kolmogorov-Smirnov distance between the DELTA calibration scores and the dataset's
                   healthy scores, bootstrap interval over patients (2000 resamples)

Only aggregate numbers are written (restricted data: no per-patient output at all).

    python scripts/run_wp17_structure_calibration.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.fusion.conformal import realised_false_alarm_range  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfrtool.allocation import effective_level  # noqa: E402
from sfrtool.artifacts import load_artifact  # noqa: E402

DATASETS = ("heartbeat_2T", "heartbeat_3T", "cardium")
LEVEL, TOLERANCE = 0.05, 0.05
SIZES = (0, 5, 10, 19, 25, 40, 60, 100, 150, 200, 300)
N_DRAWS, N_BOOT, SEED = 500, 2000, 0
MIN_TO_ALARM = int(np.ceil(1 / LEVEL)) - 1


def per_case(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = RUNS / f"wp11_external_echo_{dataset}" / "private" / "per_case.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["score"]) for r in rows])


def rate(test: np.ndarray, calibration: np.ndarray) -> float:
    """Share of ``test`` scores whose conformal p-value against ``calibration`` is at most the level."""
    if test.size == 0:
        return float("nan")
    ordered = np.sort(calibration)
    at_least = ordered.size - np.searchsorted(ordered, test, side="left")
    return float(np.mean((1 + at_least) / (ordered.size + 1) <= LEVEL + 1e-12))


def ks_interval(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    draws = [stats.ks_2samp(a[rng.integers(0, a.size, a.size)], b[rng.integers(0, b.size, b.size)]).statistic for _ in range(N_BOOT)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"distance": float(stats.ks_2samp(a, b).statistic), "ci": [float(low), float(high)]}


def main() -> None:
    arrays, card = load_artifact("structure")
    delta = np.asarray(arrays["calibration"], dtype=float)
    for dataset in DATASETS:
        load_run(f"wp11_external_echo_{dataset}")
    run = Run("wp17_structure_calibration",
              description="Structure expert: false alarms and detection on Heartbeat and CARDIUM with transported, local and pooled "
                          "calibration (exploratory)",
              script="scripts/run_wp17_structure_calibration.py",
              config={"restricted": True, "datasets": list(DATASETS), "level": LEVEL, "tolerance": TOLERANCE, "sizes": list(SIZES),
                      "draws": N_DRAWS, "bootstrap": N_BOOT, "seed": SEED, "transported": f"{delta.size} DELTA calibration scores",
                      "holds": "can alarm in every draw and mean false-alarm rate <= level + tolerance",
                      "made_after_seeing_results": True},
              inputs={"scores": [RUNS / f"wp11_external_echo_{d}" / "private" / "per_case.csv" for d in DATASETS]},
              n_real=sum(per_case(d)[0].size for d in DATASETS), seed=SEED)
    results = {}
    for dataset in DATASETS:
        label, score = per_case(dataset)
        healthy, chd = score[label == 0], score[label == 1]
        rng = np.random.default_rng(SEED)
        transported = {"false_alarms": rate(healthy, delta), "detection": rate(chd, delta),
                       "effective_level": effective_level(LEVEL, delta.size)}
        curve = []
        for size in SIZES:
            if size > healthy.size - 50:
                continue
            local_fa, local_det, pooled_fa, pooled_det = [], [], [], []
            for _ in range(N_DRAWS):
                order = rng.permutation(healthy.size)
                calibration, test = healthy[order[:size]], healthy[order[size:]]
                if size >= 1:
                    local_fa.append(rate(test, calibration))
                    local_det.append(rate(chd, calibration))
                pooled = np.concatenate([delta, calibration])
                pooled_fa.append(rate(test, pooled))
                pooled_det.append(rate(chd, pooled))
            row = {"local_healthy": size, "can_alarm_local": bool(size >= MIN_TO_ALARM),
                   "effective_level_local": effective_level(LEVEL, size) if size else 0.0,
                   "beta_range_90_local": list(realised_false_alarm_range(size, LEVEL)) if size >= MIN_TO_ALARM else [0.0, 0.0]}
            for name, values in (("local_false_alarms", local_fa), ("local_detection", local_det),
                                 ("pooled_false_alarms", pooled_fa), ("pooled_detection", pooled_det)):
                if values:
                    row[name] = {"mean": float(np.mean(values)), "p5": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}
            row["local_holds"] = bool(row["can_alarm_local"] and "local_false_alarms" in row
                                      and row["local_false_alarms"]["mean"] <= LEVEL + TOLERANCE + 1e-12)
            row["pooled_holds"] = bool(row["pooled_false_alarms"]["mean"] <= LEVEL + TOLERANCE + 1e-12)
            curve.append(row)
        smallest = {}
        for name in ("local", "pooled"):
            holding = [r for r in curve if r[f"{name}_holds"]]
            smallest[name] = ({"healthy": holding[0]["local_healthy"],
                               "detection_mean": holding[0][f"{name}_detection"]["mean"]} if holding else "not reached")
        results[dataset] = {"n_healthy": int(healthy.size), "n_chd": int(chd.size), "transported": transported,
                            "curve": curve, "smallest_that_holds": smallest,
                            "shift": {**ks_interval(delta, healthy, rng),
                                      "observed_excess_false_alarms": transported["false_alarms"] - transported["effective_level"]}}
        print(f"{dataset}: transported FA {transported['false_alarms']:.3f} detection {transported['detection']:.3f}; smallest that holds "
              f"{smallest}; KS {results[dataset]['shift']['distance']:.3f}", flush=True)
    run.finish({"by_dataset": results, "made_after_seeing_results": True,
                "reading": "transported vs local calibration of the frozen DELTA structure expert on two external datasets with CHD cases"})


if __name__ == "__main__":
    main()
