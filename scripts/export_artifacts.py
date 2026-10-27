"""Write the artifacts of the one-patient tool (sfrtool) from the saved outputs of the runs.

    python scripts/export_artifacts.py

Nothing new is learned. The linear models of the runs are refitted on the features those runs
saved, with the same settings, and every refit is checked against the published numbers before
it is written. The checks are kept in runs/tool_artifacts/metrics.json.

- rhythm: logistic regression on the 9 interval features of the 26 NIFEADB recordings
  (run wp4b_rhythm_intervals), refitted as in scripts/run_wp6_fusion.py. Check: the NInFEA rhythm
  scores of run wp6_fusion are reproduced from the beat times saved by run wp3_trust_check.
  Calibration sets: "published", the out-of-fold scores of the 14 normal NIFEADB recordings;
  "ninfea", the scores of the readable healthy NInFEA recordings (run wp6_fusion).
- function: reference lines fitted on the 60 healthy NInFEA recordings (run wp2b); calibration,
  their leave-one-woman-out scores. Check: those scores are reproduced from the saved measures.
- structure: the probe of official_split() in scripts/run_wp5_structure.py, fitted on the DELTA
  Train cases (embeddings saved by run wp5_structure_delta, not in git); calibration, the scores
  of the normal Val cases. Check: AUROC and alarms on the Test cases equal the published result.
  Skipped, with a note, when the embeddings are not on this machine.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.doppler.reference import REFERENCE_MEASURES, fit_reference, leave_one_woman_out  # noqa: E402
from sfr.echo.fetalclip import FETALCLIP_SHA256  # noqa: E402
from sfr.eval.stats import auroc  # noqa: E402
from sfr.fusion.conformal import p_value, smallest_p  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import RUNS, SPLITS  # noqa: E402
from sfr.provenance import Run, code_sha256, load_run  # noqa: E402
from sfr.rhythm.evaluate import logistic  # noqa: E402
from sfr.rhythm.features import RECORD_FEATURES, record_features, segment_features  # noqa: E402
from sfrtool import ARTIFACTS, CONDITIONS, TESTS, TOOL_VERSION  # noqa: E402
from sfrtool.artifacts import artifact_sha256, write_artifact  # noqa: E402
from sfrtool.experts import MIN_LAST_S, PLANES, SEGMENT_S, rhythm_probability, structure_probability  # noqa: E402

RHYTHM_RUN, PAIRS_RUN, TRUST_RUN = "wp4b_rhythm_intervals", "wp6_fusion", "wp3_trust_check"
FUNCTION_RUN, STRUCTURE_RUN, CHECKS_RUN = "wp2b_function_reference", "wp5_structure_delta", "wp9_checks_after_results"
RHYTHM_SCORE = "oof_logistic_rhythm"
CONFIG = DecisionConfig()
N_COMPONENTS, SEED = 32, 0                     # the probe of scripts/run_wp5_structure.py


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def common_card() -> dict:
    return {"tool_version": TOOL_VERSION, "created": date.today().isoformat(), "code_sha256": code_sha256()}


def export_rhythm() -> tuple[dict, list[Path]]:
    pairs_metrics = load_run(PAIRS_RUN)[0]
    rows = read_csv(RUNS / RHYTHM_RUN / "per_record_scores.csv")
    labels = np.array([int(r["label"]) for r in rows])
    x = np.array([[float(r[f]) for f in RECORD_FEATURES] for r in rows])
    model = logistic().fit(x, labels)                         # as in scripts/run_wp6_fusion.py, part B
    imputer, scaler, regression = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    cohort = read_csv(RUNS / PAIRS_RUN / "paired_cohort_ninfea.csv")
    arrays = {"imputer_median": imputer.statistics_, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
              "coef": regression.coef_[0], "intercept": regression.intercept_,
              "calibration_published": np.array([float(r[RHYTHM_SCORE]) for r in rows if r["label"] == "0"]),
              "calibration_ninfea": np.array([float(r["rhythm_score"]) for r in cohort if r["ecg_readable"] == "True"])}

    own = np.array([rhythm_probability(dict(zip(RECORD_FEATURES, row)), arrays) for row in x])
    beats = json.loads((RUNS / TRUST_RUN / "beats.json").read_text(encoding="utf-8"))
    differences, p_differences = [], []
    for r in cohort:                                          # NInFEA features from the saved beats, as run wp6 made them
        if r["ecg_readable"] != "True":
            continue
        item = beats[r["record"]]
        raw, tracked = np.array(item["raw_s"]), np.array(item["r_s"])
        edges = np.arange(0.0, item["duration_s"], SEGMENT_S)
        segments = [segment_features(raw[(raw >= a) & (raw < min(a + SEGMENT_S, item["duration_s"]))],
                                     tracked[(tracked >= a) & (tracked < min(a + SEGMENT_S, item["duration_s"]))])
                    for a in edges if min(a + SEGMENT_S, item["duration_s"]) - a >= MIN_LAST_S or len(edges) == 1]
        score = rhythm_probability(record_features(segments), arrays)
        differences.append(abs(score - float(r["rhythm_score"])))
        p_differences.append(abs(p_value(score, arrays["calibration_published"]) - float(r["rhythm_p"])))
    checks = {"training_scores_equal_sklearn_max_abs_diff": float(np.max(np.abs(own - model.predict_proba(x)[:, 1]))),
              "ninfea_scores_of_wp6_reproduced_max_abs_diff": float(max(differences)), "ninfea_recordings_checked": len(differences),
              "ninfea_p_values_of_wp6_reproduced_max_abs_diff": float(max(p_differences))}
    assert checks["training_scores_equal_sklearn_max_abs_diff"] < 1e-12
    assert checks["ninfea_scores_of_wp6_reproduced_max_abs_diff"] < 1e-5, checks   # wp6 saved scores rounded to 6 decimals
    assert checks["ninfea_p_values_of_wp6_reproduced_max_abs_diff"] < 1e-5, checks

    published = pairs_metrics["B_real_paired_cohort"]
    local = published["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]["false_referrals"]
    limits = ["fitted on 26 NIFEADB recordings of one dataset (12 arrhythmia, 14 normal); later gestation than NInFEA",
              f"calibrated on NIFEADB, it referred {published['false_referrals']['share']:.1%} of 60 healthy NInFEA recordings; "
              f"calibrated on healthy NInFEA recordings of other women, {local['share']:.1%} (run {PAIRS_RUN})",
              f"with {arrays['calibration_published'].size} healthy calibration recordings the smallest possible p-value is "
              f"{smallest_p(arrays['calibration_published'].size):.3f}; the alarm level {CONFIG.alpha['rhythm']} acts as that value"]
    if (RUNS / CHECKS_RUN / "metrics.json").exists():
        row = next(r for r in read_csv(RUNS / CHECKS_RUN / "rhythm_1khz_only.csv") if r["score"] == RHYTHM_SCORE)
        limits.append(f"the four 500 Hz NIFEADB recordings are all arrhythmia; on the 1 kHz recordings its AUROC is {float(row['auroc_1khz']):.3f} "
                      f"[{float(row['ci_1khz_low']):.3f}, {float(row['ci_1khz_high']):.3f}] (run {CHECKS_RUN})")
    card = {**common_card(), "condition": CONDITIONS["rhythm"], "test": TESTS["rhythm"],
            "dataset": "NIFEADB (PhysioNet, ODC-By 1.0): 12 arrhythmia and 14 normal-rhythm recordings",
            "model": {"name": "logistic regression on fetal interval features (sfr.rhythm.evaluate.logistic)", "sha256": None},
            "features": list(RECORD_FEATURES), "alarm_level": CONFIG.alpha["rhythm"],
            "calibration_sets": {
                "published": {"array": "calibration_published", "n": int(arrays["calibration_published"].size), "setting": "nifeadb",
                              "source": f"out-of-fold scores of the normal NIFEADB recordings (run {RHYTHM_RUN})"},
                "ninfea": {"array": "calibration_ninfea", "n": int(arrays["calibration_ninfea"].size), "setting": "ninfea",
                           "source": f"scores of the healthy NInFEA recordings with a readable ECG (run {PAIRS_RUN})"}},
            "transport": {"false_referral_share_on_ninfea_with_published_calibration": published["false_referrals"]["share"],
                          "false_referral_share_on_ninfea_with_ninfea_calibration": local["share"], "run": PAIRS_RUN},
            "intended_input": "WFDB record (.hea and .dat) in the NInFEA layout (24 abdominal channels uni_abd1-24, thoracic leads bi_tho1-3; "
                              "whole record, as run wp3) or the NIFEADB layout (channel ECG from the chest, then Abdomen_1...; 60-second segments, as run wp4a)",
            "known_limits": limits, "source_runs": [RHYTHM_RUN, PAIRS_RUN, TRUST_RUN], "verification": checks}
    write_artifact("rhythm", arrays, card)
    return {"exported": True, "cases": int(labels.size), **checks, "sha256": artifact_sha256("rhythm")}, [RUNS / RHYTHM_RUN / "per_record_scores.csv",
                                                                               RUNS / PAIRS_RUN / "paired_cohort_ninfea.csv", RUNS / TRUST_RUN / "beats.json"]


def export_function() -> tuple[dict, list[Path]]:
    rows = read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")
    usable = [{"record": int(r["record"]), "woman": r["woman"], "gestation_days": float(r["gestation_days"]),
               **{m: float(r[m]) for m in REFERENCE_MEASURES}} for r in rows if all(np.isfinite(float(r[m])) for m in REFERENCE_MEASURES)]
    reference = fit_reference(usable)
    names = list(REFERENCE_MEASURES)
    saved = {int(r["record"]): r for r in rows}
    calibration = np.array([float(saved[u["record"]]["score"]) for u in usable])
    again = {s["record"]: s for s in leave_one_woman_out(usable)}
    checks = {"loo_scores_of_wp2b_reproduced_max_abs_diff": float(max(abs(again[u["record"]]["score"] - float(saved[u["record"]]["score"])) for u in usable)),
              "loo_p_values_of_wp2b_reproduced_max_abs_diff": float(max(abs(again[u["record"]]["p_value"] - float(saved[u["record"]]["p_value"])) for u in usable)),
              "recordings": len(usable)}
    assert checks["loo_scores_of_wp2b_reproduced_max_abs_diff"] < 1e-4, checks     # wp2b saved measures rounded to 6 decimals
    assert checks["loo_p_values_of_wp2b_reproduced_max_abs_diff"] < 1e-5, checks
    days = np.array([u["gestation_days"] for u in usable])
    arrays = {"slope": [reference[n]["slope"] for n in names], "intercept": [reference[n]["intercept"] for n in names],
              "sd": [reference[n]["sd"] for n in names], "calibration": calibration, "gestation_days_range": [days.min(), days.max()]}
    card = {**common_card(), "condition": CONDITIONS["function"], "test": TESTS["function"],
            "dataset": f"NInFEA (PhysioNet, ODC-By 1.0): {len(usable)} healthy recordings of {len({u['woman'] for u in usable})} women, weeks 21 to 27",
            "model": {"name": "healthy reference lines (sfr.doppler.reference)", "sha256": None},
            "measures": names, "covariates": {n: reference[n]["covariate"] for n in names}, "alarm_level": CONFIG.alpha["function"],
            "calibration": {"n": int(calibration.size), "source": f"leave-one-woman-out scores of the healthy recordings (run {FUNCTION_RUN})"},
            "intended_input": "pulsed-wave Doppler strip image at 284 columns per second (NInFEA format) and the gestational age; "
                              f"the reference covers {int(days.min()) // 7}+{int(days.min()) % 7} to {int(days.max()) // 7}+{int(days.max()) % 7} weeks",
            "known_limits": ["healthy recordings only: its sensitivity for abnormal timing is unknown",
                             "one dataset, one device; only the heart rate measure is clearly repeatable (run wp2b)"],
            "source_runs": [FUNCTION_RUN], "verification": checks}
    write_artifact("function", arrays, card)
    return {"exported": True, "cases": len(usable), **checks, "sha256": artifact_sha256("function")}, [RUNS / FUNCTION_RUN / "per_recording.csv"]


def export_structure() -> tuple[dict, list[Path]]:
    embeddings_file = RUNS / STRUCTURE_RUN / "embeddings.npz"
    if not embeddings_file.exists():
        return {"exported": False, "reason": f"{embeddings_file} is not on this machine; it is written by run {STRUCTURE_RUN} (DELTA dataset and FetalCLIP, GPU advised)"}, []
    metrics = load_run(STRUCTURE_RUN)[0]
    published = metrics["own_partition"]["fetalclip_scan"]
    saved = np.load(embeddings_file, allow_pickle=False)
    cases, y, x = [str(c) for c in saved["cases"]], saved["labels"].astype(int), saved["fetalclip_scan"]
    split_of = {r["case"]: r["split"] for r in read_csv(SPLITS / "echo_delta_index.csv")}
    split = np.array([split_of[c] for c in cases])
    train, val, test = split == "Train", split == "Val", split == "Test"
    model = make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=SEED),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(x[train], y[train])
    scaler, pca, regression = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    healthy = model.predict_proba(x[val & (y == 0)])[:, 1]
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "pca_mean": pca.mean_, "pca_components": pca.components_,
              "coef": regression.coef_[0], "intercept": regression.intercept_, "calibration": healthy}
    scores = model.predict_proba(x[test])[:, 1]
    mine = np.array([structure_probability(v, arrays) for v in x[test]])
    alarm = np.array([p_value(float(s), healthy) for s in scores]) <= CONFIG.alpha["structure"]
    checks = {"test_auroc": auroc(y[test], scores), "published_test_auroc": published["auroc"],
              "probe_equals_sklearn_max_abs_diff": float(np.max(np.abs(mine - scores))),
              "calibration_normal_cases": int(healthy.size), "published_calibration_normal_cases": published["calibration_normal_cases"],
              "test_defects_caught": int(alarm[y[test] == 1].sum()), "published_test_defects_caught": published["alarm"]["sensitivity"]["caught"],
              "test_normal_flagged": int(alarm[y[test] == 0].sum()), "published_test_normal_flagged": published["alarm"]["false_alarms"]["flagged"]}
    assert checks["test_auroc"] == checks["published_test_auroc"], checks
    assert checks["calibration_normal_cases"] == checks["published_calibration_normal_cases"]
    assert checks["test_defects_caught"] == checks["published_test_defects_caught"]
    assert checks["test_normal_flagged"] == checks["published_test_normal_flagged"]
    assert checks["probe_equals_sklearn_max_abs_diff"] < 1e-10
    model_record = metrics["models"]["published"]
    limits = [f"one open dataset from one centre (DELTA); inputs with little or no cardiac anatomy also separate the classes (AUROC "
              f"{metrics['probes']['no_anatomy_numbers']['auroc']:.3f} to {metrics['probes']['fetalclip_overlay_only']['auroc']:.3f}, run {STRUCTURE_RUN}), "
              "so its performance is an upper bound on what heart structure contributes"]
    if (RUNS / CHECKS_RUN / "metrics.json").exists():
        sizes = load_run(CHECKS_RUN)[0]["D_subgroups"]["structure_by_image_size"]
        limits.append("defects caught by image size: " + "; ".join(f"{r['abnormal_caught']} of {r['abnormal']} in {r['image_size']}" for r in sizes if r["abnormal"] >= 10)
                      + f" (run {CHECKS_RUN}, cross-validated probe)")
    limits.append("the run computed its embeddings on a GPU; on a CPU they differ in the last digits (tests/test_predict.py checks the agreement)")
    card = {**common_card(), "condition": CONDITIONS["structure"], "test": TESTS["structure"],
            "dataset": "DELTA (figshare doi:10.6084/m9.figshare.32358771, CC BY 4.0): probe fitted on the Train split "
                       f"({int(train.sum())} cases), calibrated on the normal cases of the Val split",
            "model": {"name": "FetalCLIP", "sha256": FETALCLIP_SHA256, "file": model_record["file"], "licence": model_record["licence"],
                      "source": model_record["source"], "probe": f"standardise, PCA to {N_COMPONENTS}, logistic regression (balanced)"},
            "planes": list(PLANES), "alarm_level": CONFIG.alpha["structure"],
            "calibration": {"n": int(healthy.size), "source": f"probe scores of the normal Val cases (run {STRUCTURE_RUN}, own partition)"},
            "intended_input": "a folder with one image of each plane, named Abdomen, 4CH, LVOT, RVOT and 3VT (.jpg, .png or .bmp); "
                              "every image is cleaned by sfr.echo.deidentify before it is embedded",
            "known_limits": limits, "source_runs": [STRUCTURE_RUN], "verification": checks}
    write_artifact("structure", arrays, card)
    return {"exported": True, "cases": len(cases), **checks, "sha256": artifact_sha256("structure")}, [embeddings_file, SPLITS / "echo_delta_index.csv"]


def main() -> None:
    results, inputs = {}, {}
    for name, export in (("rhythm", export_rhythm), ("function", export_function), ("structure", export_structure)):
        results[name], files = export()
        if files:
            inputs[name] = files
        print(name, json.dumps(results[name]))
    run = Run("tool_artifacts",
              description="Artifacts of the one-patient research tool (sfrtool): linear models of the runs refitted on their saved features, "
                          "each checked against the published numbers before it is written to artifacts/",
              script="scripts/export_artifacts.py",
              config={"artifacts_folder": ARTIFACTS.relative_to(ARTIFACTS.parent).as_posix(), "tool_version": TOOL_VERSION,
                      "rhythm_score": RHYTHM_SCORE, "structure_probe_components": N_COMPONENTS, "seed": SEED},
              inputs=inputs, n_real=sum(r.get("cases", 0) for r in results.values()), seed=SEED)
    run.finish({"artifacts": results})


if __name__ == "__main__":
    main()
