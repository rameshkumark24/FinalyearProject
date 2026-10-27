"""WP18 - the same fetus scanned twice: does the structure expert give the same answer?

296 Heartbeat patients were scanned in the 2nd and in the 3rd trimester. A structural defect does
not come and go, so a reader of heart structure should rank these fetuses alike at both scans.
Exploratory (after the preregistered external test); nothing is refitted: the scores are those of
the preregistered runs wp11_external_echo_heartbeat_2T and _3T (private files).

Settings, fixed before computing:
  score agreement    Spearman correlation of a patient's two scores, bootstrap interval over patients
                     (2000 resamples, seed 0)
  alarm agreement    alarms at level 0.05 with the transported (DELTA) calibration: 2 x 2 table and
                     Cohen's kappa
  labels             patients whose label differs between the two trimesters are counted and left out
  per label          the same correlation among patients with CHD and among those without

Only aggregate numbers are written (restricted data).

    python scripts/run_wp18_echo_retest.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

RUNS_USED = ("wp11_external_echo_heartbeat_2T", "wp11_external_echo_heartbeat_3T")
LEVEL, N_BOOT, SEED = 0.05, 2000, 0


def per_patient(run: str) -> dict[str, dict]:
    with (RUNS / run / "private" / "per_case.csv").open(encoding="utf-8", newline="") as handle:
        return {r["case"].rsplit("_", 1)[0]: {"label": int(r["label"]), "score": float(r["score"]),
                                               "alarm": float(r["p_transported"]) <= LEVEL + 1e-12}
                for r in csv.DictReader(handle)}


def spearman_with_interval(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    point = stats.spearmanr(a, b).statistic
    draws = []
    for _ in range(N_BOOT):
        index = rng.integers(0, a.size, a.size)
        draws.append(stats.spearmanr(a[index], b[index]).statistic)
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"rho": float(point), "ci": [float(low), float(high)], "n": int(a.size)}


def main() -> None:
    for run_name in RUNS_USED:
        load_run(run_name)
    second, third = (per_patient(r) for r in RUNS_USED)
    both = sorted(set(second) & set(third))
    disagree = [p for p in both if second[p]["label"] != third[p]["label"]]
    kept = [p for p in both if p not in disagree]
    run = Run("wp18_echo_retest",
              description="Structure expert on the same fetuses scanned in the 2nd and the 3rd trimester (Heartbeat; exploratory)",
              script="scripts/run_wp18_echo_retest.py",
              config={"restricted": True, "runs": list(RUNS_USED), "level": LEVEL, "bootstrap": N_BOOT, "seed": SEED,
                      "made_after_seeing_results": True},
              inputs={"scores": [RUNS / r / "private" / "per_case.csv" for r in RUNS_USED]},
              n_real=len(both), seed=SEED)
    rng = np.random.default_rng(SEED)
    a = np.array([second[p]["score"] for p in kept])
    b = np.array([third[p]["score"] for p in kept])
    labels = np.array([second[p]["label"] for p in kept])
    alarm_a = np.array([second[p]["alarm"] for p in kept])
    alarm_b = np.array([third[p]["alarm"] for p in kept])
    table = {"both": int(np.sum(alarm_a & alarm_b)), "2T only": int(np.sum(alarm_a & ~alarm_b)),
             "3T only": int(np.sum(~alarm_a & alarm_b)), "neither": int(np.sum(~alarm_a & ~alarm_b))}
    observed = (table["both"] + table["neither"]) / len(kept)
    expected = (np.mean(alarm_a) * np.mean(alarm_b) + np.mean(~alarm_a) * np.mean(~alarm_b))
    kappa = float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")
    metrics = {
        "patients_in_both_trimesters": len(both), "label_differs_between_trimesters": len(disagree), "kept": len(kept),
        "kept_with_chd": int(labels.sum()),
        "score_agreement_all": spearman_with_interval(a, b, rng),
        "score_agreement_chd": spearman_with_interval(a[labels == 1], b[labels == 1], rng) if labels.sum() > 3 else "too few",
        "score_agreement_no_chd": spearman_with_interval(a[labels == 0], b[labels == 0], rng),
        "alarm_table_transported": table, "alarm_kappa_transported": kappa,
        "made_after_seeing_results": True,
    }
    run.finish(metrics)
    print({k: v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
