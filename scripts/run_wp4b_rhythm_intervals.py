"""WP4b - rhythm expert on beat intervals (NIFEADB: 12 arrhythmia, 14 normal rhythm).

Input: the fetal beats of run ``wp4a_nifeadb_beats``. Unit of analysis: the recording (one
fetus). Every reported score comes from a model that did not see that recording
(leave-one-subject-out). The feature set, the models and their settings were fixed before any
of them was compared with the labels; nothing is tuned.

Rows of the result table:
  single index      one pre-specified irregularity feature used as it is (nothing fitted)
  logistic / forest the nine recording-level rhythm features
  quality only      extractor quality values. The extractor's score includes rhythm regularity, so
                    only the beat-shape consistency value is a neutral control
  segment split     the leakage comparison: the same model scored with segments of a recording
                    on both sides of the split, and with the recording held out

    python scripts/run_wp4b_rhythm_intervals.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.nifeadb import ALL_RECORDS, label_of, record_files  # noqa: E402
from sfr.eval.stats import auroc_ci  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.rhythm.evaluate import evaluate, forest, logistic, out_of_fold, single_index, summarise  # noqa: E402
from sfr.rhythm.features import (IRREGULARITY_FEATURES, QUALITY_FEATURES, RECORD_FEATURES, SEGMENT_FEATURES,  # noqa: E402
                                 record_features, segment_features)

SOURCE_RUN = "wp4a_nifeadb_beats"
N_PERM_LOGISTIC, N_PERM_FOREST, N_PERM_INDEX = 500, 50, 10000
SEED = 0


def main() -> None:
    _, source_provenance = load_run(SOURCE_RUN)
    source = RUNS / SOURCE_RUN
    beats = json.loads((source / "beats.json").read_text(encoding="utf-8"))
    with open(source / "segments.csv", encoding="utf-8") as handle:
        segment_rows = list(csv.DictReader(handle))
    assert sorted(beats) == sorted(ALL_RECORDS)

    run = Run(
        "wp4b_rhythm_intervals",
        description="Rhythm expert on beat-interval features, NIFEADB, leave-one-subject-out, with controls and the leakage comparison",
        script="scripts/run_wp4b_rhythm_intervals.py",
        config={"source_run": SOURCE_RUN, "source_code_sha256": source_provenance["code_sha256"],
                "segment_features": SEGMENT_FEATURES, "record_features": RECORD_FEATURES, "quality_features": QUALITY_FEATURES,
                "logistic": "median imputation, standardisation, L2 logistic regression, C=1",
                "forest": "300 trees, depth 3, at least 2 per leaf", "threshold": 0.5,
                "permutations": {"logistic": N_PERM_LOGISTIC, "forest": N_PERM_FOREST, "single_index": N_PERM_INDEX}},
        inputs={"nifeadb": [p for name in ALL_RECORDS for p in record_files(name)],
                "beats": [source / "beats.json", source / "segments.csv"]},
        n_real=len(ALL_RECORDS),
        seed=SEED,
    )

    # ---- features -------------------------------------------------------------------------
    quality_of = {(r["record"], int(r["segment"])): r for r in segment_rows}
    segments: list[dict] = []
    for name in ALL_RECORDS:
        for item in beats[name]:
            values = segment_features(np.array(item["raw_s"]), np.array(item["tracked_s"]))
            q = quality_of[(name, item["segment"])]
            segments.append({"record": name, "label": label_of(name), "segment": item["segment"], "ok": bool(item["ok"]),
                             "score": float(q["score"]), "template_corr": float(q["template_corr"]), **values})
    run.write_csv("segment_features.csv", segments)

    def table(only_ok: bool) -> tuple[np.ndarray, list[dict]]:
        rows = []
        for name in ALL_RECORDS:
            mine = [s for s in segments if s["record"] == name]
            used = [s for s in mine if s["ok"]] if only_ok else mine
            row = {"record": name, "label": label_of(name), "n_segments": len(mine), "n_used": len(used),
                   **record_features(used),
                   "usable_share": float(np.mean([s["ok"] for s in mine])),
                   "score_median": float(np.median([s["score"] for s in mine])),
                   "template_corr_median": float(np.median([s["template_corr"] for s in mine]))}
            rows.append(row)
        return np.array([[r[f] for f in RECORD_FEATURES] for r in rows], dtype=float), rows

    y = np.array([label_of(name) for name in ALL_RECORDS])
    x_primary, rows_primary = table(only_ok=True)
    x_all, rows_all = table(only_ok=False)
    x_quality = np.array([[r[f] for f in QUALITY_FEATURES] for r in rows_primary], dtype=float)
    run.write_csv("record_features.csv", rows_primary)

    # ---- recording-level models --------------------------------------------------------------
    results: dict = {}
    oof: dict[str, np.ndarray] = {}
    results["logistic_rhythm"], oof["logistic_rhythm"] = evaluate(x_primary, y, logistic(), N_PERM_LOGISTIC, SEED)
    results["forest_rhythm"], oof["forest_rhythm"] = evaluate(x_primary, y, forest(SEED), N_PERM_FOREST, SEED)
    results["logistic_quality_only"], oof["logistic_quality_only"] = evaluate(x_quality, y, logistic(), N_PERM_LOGISTIC, SEED)
    results["logistic_rhythm_all_segments"], oof["logistic_rhythm_all_segments"] = evaluate(x_all, y, logistic(), N_PERM_LOGISTIC, SEED)
    results["logistic_rhythm_plus_quality"], oof["logistic_rhythm_plus_quality"] = evaluate(
        np.hstack([x_primary, x_quality]), y, logistic(), N_PERM_LOGISTIC, SEED)
    for key in results:
        results[key]["kind"] = "leave-one-subject-out"
    results["logistic_quality_only"]["role"] = "extractor quality values only; its score includes rhythm regularity, so this is not a neutral control"

    single: dict = {}
    for feature in IRREGULARITY_FEATURES:
        column = RECORD_FEATURES.index(f"{feature}_median")
        single[f"{feature}_median"] = single_index(y, x_primary[:, column], N_PERM_INDEX, SEED)
    for feature in ("outlier_share_p90", "short_long_share_p90"):
        single[feature] = single_index(y, x_primary[:, RECORD_FEATURES.index(feature)], N_PERM_INDEX, SEED)
    rate = x_primary[:, RECORD_FEATURES.index("hr_bpm_median")]
    single["hr_bpm_median"] = single_index(y, rate, N_PERM_INDEX, SEED)
    single["hr_bpm_median"]["note"] = "no direction was pre-specified for heart rate; AUROC below 0.5 means arrhythmia recordings are slower"
    quality_roles = {"usable_share_low": "share of usable segments (lower = arrhythmia)",
                     "score_median_low": "extractor score (lower = arrhythmia); it includes rhythm regularity, so it is not a neutral control",
                     "template_corr_median_low": "beat-shape consistency only (lower = arrhythmia): the neutral quality control"}
    for key, column in (("usable_share_low", 0), ("score_median_low", 1), ("template_corr_median_low", 2)):
        single[key] = single_index(y, -x_quality[:, column], N_PERM_INDEX, SEED)
        single[key]["role"] = quality_roles[key]
    for value in single.values():
        value["kind"] = "single pre-specified index, nothing fitted"

    per_record = []
    for i, name in enumerate(ALL_RECORDS):
        per_record.append({"record": name, "label": int(y[i]), **{f"oof_{k}": float(v[i]) for k, v in oof.items()},
                           **{f: rows_primary[i][f] for f in RECORD_FEATURES + QUALITY_FEATURES}})
    run.write_csv("per_record_scores.csv", per_record)

    # ---- leakage comparison on real data (segment level) ------------------------------------------
    usable = [s for s in segments if s["ok"] and np.isfinite(s["hr_bpm"])]
    xs = np.array([[s[f] for f in SEGMENT_FEATURES] for s in usable], dtype=float)
    ys = np.array([s["label"] for s in usable])
    groups = np.array([s["record"] for s in usable])
    leakage: dict = {"n_segments": int(ys.size), "n_records": int(np.unique(groups).size)}
    for model_name, model in (("forest", forest(SEED)), ("logistic", logistic())):
        subject_scores = out_of_fold(xs, ys, model, groups)
        leaky_scores = np.full(ys.size, np.nan)
        for train, test in StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(xs, ys):
            leaky_scores[test] = clone(model).fit(xs[train], ys[train]).predict_proba(xs[test])[:, 1]
        subject = auroc_ci(ys, subject_scores, groups=groups, seed=SEED)
        leaky = auroc_ci(ys, leaky_scores, groups=groups, seed=SEED)
        record_names = np.unique(groups)
        record_level = np.array([np.mean(subject_scores[groups == g]) for g in record_names])
        record_labels = np.array([label_of(g) for g in record_names])
        leakage[model_name] = {
            "segment_split_auroc": leaky[0], "segment_split_ci": [leaky[1], leaky[2]],
            "subject_split_auroc": subject[0], "subject_split_ci": [subject[1], subject[2]],
            "gap": leaky[0] - subject[0],
            "subject_split_recording_level": summarise(record_labels, record_level, seed=SEED),
        }
    leakage["note"] = ("segment split = 10-fold over one-minute segments, so segments of a recording fall on both sides; "
                       "subject split = all segments of a recording held out together. Intervals resample recordings.")

    metrics = {
        "dataset": {"n_records": int(y.size), "n_arrhythmia": int(y.sum()), "n_normal": int((1 - y).sum()),
                    "n_segments": len(segments), "n_usable_segments": int(sum(s["ok"] for s in segments))},
        "recording_level": results, "single_index": single, "leakage_comparison": leakage,
        "gate_G4_partial": {"negative_control_row_present": True,
                            "note": "the raw-lead control and the checkpoint check belong to run wp4c"},
    }
    run.finish(metrics)

    def show(name: str, r: dict) -> None:
        extra = f"  sens {r['sensitivity']:.2f} spec {r['specificity']:.2f}" if "sensitivity" in r else ""
        print(f"{name:34s} n {r['n']:2d}  AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]  p {r['permutation_p']:.4f}{extra}")

    print("Recording level, leave-one-subject-out")
    for name, r in results.items():
        show(name, r)
    print("Single pre-specified index (nothing fitted)")
    for name, r in single.items():
        show(name, r)
    print("Leakage comparison (segment-level AUROC)")
    for model_name in ("forest", "logistic"):
        m = leakage[model_name]
        print(f"  {model_name:9s} segment split {m['segment_split_auroc']:.3f} [{m['segment_split_ci'][0]:.3f}, {m['segment_split_ci'][1]:.3f}]"
              f"   subject split {m['subject_split_auroc']:.3f} [{m['subject_split_ci'][0]:.3f}, {m['subject_split_ci'][1]:.3f}]   gap {m['gap']:+.3f}")


if __name__ == "__main__":
    main()
