"""WP5c - the structure expert by image size, with alarms calibrated inside each size group.

Run wp9 found that the structure expert caught 60 of 114 defects in 900x752 images but 0 of 11 in
642x506 images, and run wp5b that inputs with little or no cardiac anatomy separate the classes.
Image size marks the acquisition setting, so it is a candidate shortcut. Added on 20 October 2026
in answer to a review; a check made after seeing the results.

Everything is computed from the embeddings saved by run wp5_structure_delta; no image is read and
nothing is retrained except the linear probe, refitted exactly as in that run.

1. The run's conformal p-values are recomputed with the same folds, probe and calibration split,
   and must equal the published ones; otherwise the script stops.
2. The same folds again, but each case is ranked only among the calibration normal cases of its
   own image-size group (group-conditional, "Mondrian", conformal). This keeps the false-alarm
   rate at the level inside every group, if cases are exchangeable within a group. It cannot
   create detection that the score does not carry.
3. By group: AUROC of the run's cross-validated scores (bootstrap interval over cases), and
   sensitivity and false-alarm rate at level 0.05 with both calibrations (Clopper-Pearson).

Groups, fixed before computing: 900x752, 642x506, 806x648, and every other size pooled (744x658,
1114x766, 448x404; 21 normal and 3 abnormal cases). With n calibration normal cases a group's
smallest p-value is 1/(n+1), so a group needs at least 19 of them in a fold to alarm at 0.05;
groups below that are reported as unable to alarm, not as missing.

    python scripts/run_wp5c_structure_by_size.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import auroc_ci, clopper_pearson  # noqa: E402
from sfr.fusion.conformal import p_value  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfrtool.conformal_groups import group_conditional_p_values  # noqa: E402

SOURCE_RUN = "wp5_structure_delta"
FEATURES = "fetalclip_scan"
SCORE_COLUMN = "score_fetalclip_scan"
ALPHA = 0.05
N_FOLDS, N_COMPONENTS, CALIBRATION_SHARE, SEED = 5, 32, 0.25, 0          # as in run wp5_structure_delta
GROUPS = ("900x752", "642x506", "806x648")
OTHER = "other sizes"
MIN_CALIBRATION = int(np.ceil(1 / ALPHA)) - 1                              # 19: below this a group cannot alarm at 0.05


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def probe(seed: int = SEED):
    return make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=seed),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def folds_with_calibration(x: np.ndarray, y: np.ndarray):
    """The folds of run wp5: stratified 5-fold over cases, 25% of each training fold kept for calibration."""
    for train, test in StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x, y):
        fit, calibration = train_test_split(train, test_size=CALIBRATION_SHARE, stratify=y[train], random_state=SEED)
        yield fit, calibration, test


def proportion(k: int, n: int) -> dict:
    low, high = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "share": k / n if n else float("nan"), "ci": [low, high]}


def main() -> None:
    metrics_wp5, _ = load_run(SOURCE_RUN)
    saved = np.load(RUNS / SOURCE_RUN / "embeddings.npz", allow_pickle=False)
    cases, y, x = [str(c) for c in saved["cases"]], saved["labels"].astype(int), saved[FEATURES]
    rows = {r["case"]: r for r in read_csv(RUNS / SOURCE_RUN / "per_case.csv")}
    if sorted(rows) != sorted(cases) or any(int(rows[c]["label"]) != label for c, label in zip(cases, y)):
        raise ValueError("embeddings and per_case.csv disagree on cases or labels")
    size = np.array([rows[c]["image_size"] for c in cases])
    group = np.array([s if s in GROUPS else OTHER for s in size])
    published_p = np.array([float(rows[c]["p_value"]) for c in cases])
    score = np.array([float(rows[c][SCORE_COLUMN]) for c in cases])

    run = Run("wp5c_structure_by_size",
              description="Structure expert by image size with alarms calibrated inside each size group (group-conditional conformal), "
                          "from the saved embeddings. A check made after seeing the results.",
              script="scripts/run_wp5c_structure_by_size.py",
              config={"made_after_seeing_results": True, "requested_by": "review of 20 Oct 2026", "source_run": SOURCE_RUN,
                      "features": FEATURES, "groups": list(GROUPS) + [OTHER], "alpha": ALPHA,
                      "folds": f"stratified {N_FOLDS}-fold, seed {SEED}, calibration share {CALIBRATION_SHARE} (as in wp5)",
                      "probe": f"standardise, PCA({N_COMPONENTS}), logistic regression C=1 balanced (as in wp5)",
                      "minimum_calibration_normals_to_alarm": MIN_CALIBRATION,
                      "auroc": "run wp5's cross-validated scores, bootstrap over cases, 2000 resamples, seed 0"},
              inputs={"embeddings": [RUNS / SOURCE_RUN / "embeddings.npz"], "per_case": [RUNS / SOURCE_RUN / "per_case.csv"]},
              n_real=len(cases), seed=SEED)

    marginal = np.full(y.size, np.nan)
    conditional = np.full(y.size, np.nan)
    calibration_counts = {g: [] for g in list(GROUPS) + [OTHER]}
    for fit, calibration, test in folds_with_calibration(x, y):
        model = probe().fit(x[fit], y[fit])
        healthy = calibration[y[calibration] == 0]
        healthy_scores = model.predict_proba(x[healthy])[:, 1]
        test_scores = model.predict_proba(x[test])[:, 1]
        marginal[test] = [p_value(float(s), healthy_scores) for s in test_scores]
        conditional[test] = group_conditional_p_values(test_scores, group[test], healthy_scores, group[healthy])
        for g in calibration_counts:
            calibration_counts[g].append(int(np.sum(group[healthy] == g)))
    reproduced = bool(np.allclose(marginal, published_p, atol=5e-7))
    if not reproduced:
        raise ValueError("the recomputed conformal p-values differ from run wp5_structure_delta; stopping")

    by_group, per_case = [], []
    for g in list(GROUPS) + [OTHER]:
        member = group == g
        normal, abnormal = member & (y == 0), member & (y == 1)
        entry = {"group": g, "sizes": sorted(set(size[member])), "n_normal": int(normal.sum()), "n_abnormal": int(abnormal.sum()),
                 "calibration_normals_per_fold": calibration_counts[g],
                 "can_alarm_in_every_fold": bool(min(calibration_counts[g]) >= MIN_CALIBRATION)}
        if normal.any() and abnormal.any():
            value, low, high = auroc_ci(y[member], score[member], seed=SEED)
            entry["auroc"], entry["auroc_ci"] = value, [low, high]
        for name, p in (("marginal", marginal), ("group_conditional", conditional)):
            alarm = p <= ALPHA + 1e-12
            entry[f"{name}_false_alarms"] = proportion(int(alarm[normal].sum()), int(normal.sum()))
            entry[f"{name}_sensitivity"] = proportion(int(alarm[abnormal].sum()), int(abnormal.sum()))
        by_group.append(entry)
    for c, label, s, g, pm, pc in zip(cases, y, size, group, marginal, conditional):
        per_case.append({"case": c, "label": int(label), "image_size": s, "group": g, "p_marginal": pm, "p_group_conditional": pc})

    table = [{"group": e["group"], "n_normal": e["n_normal"], "n_abnormal": e["n_abnormal"],
              "auroc": e.get("auroc", ""), "auroc_ci_low": e.get("auroc_ci", ["", ""])[0], "auroc_ci_high": e.get("auroc_ci", ["", ""])[1],
              "calibration_normals_min": min(e["calibration_normals_per_fold"]), "can_alarm_in_every_fold": e["can_alarm_in_every_fold"],
              **{f"{name}_{what}_{part}": (e[f"{name}_{what}"][part] if part != "ci_low" and part != "ci_high"
                                           else e[f"{name}_{what}"]["ci"][0 if part == "ci_low" else 1])
                 for name in ("marginal", "group_conditional") for what in ("false_alarms", "sensitivity")
                 for part in ("k", "n", "share", "ci_low", "ci_high")}} for e in by_group]
    overall = {name: {"false_alarms": proportion(int((p[y == 0] <= ALPHA + 1e-12).sum()), int((y == 0).sum())),
                      "sensitivity": proportion(int((p[y == 1] <= ALPHA + 1e-12).sum()), int((y == 1).sum()))}
               for name, p in (("marginal", marginal), ("group_conditional", conditional))}
    published = metrics_wp5.get("alarm", {}) if isinstance(metrics_wp5.get("alarm"), dict) else {}
    flag = next(e for e in by_group if e["group"] == "642x506")
    metrics = {
        "made_after_seeing_results": True,
        "published_p_values_reproduced": reproduced,
        "by_group": by_group, "overall": overall,
        "flag_642x506": {"marginal_caught": flag["marginal_sensitivity"]["k"], "group_conditional_caught": flag["group_conditional_sensitivity"]["k"],
                         "n_abnormal": flag["n_abnormal"], "auroc": flag.get("auroc"), "auroc_ci": flag.get("auroc_ci")},
        "published_alarm_summary_present": bool(published),
        "reading": ("Group-conditional calibration fixes the false-alarm rate inside each size group; it does not add detection. A group "
                    "whose AUROC interval includes 0.5, or whose sensitivity stays near zero, gives no evidence that the expert reads "
                    "heart structure there."),
    }
    run.write_csv("by_group.csv", table)
    run.write_csv("per_case_p_values.csv", per_case)
    run.finish(metrics)
    for e in by_group:
        auc = f"{e['auroc']:.3f} [{e['auroc_ci'][0]:.3f}, {e['auroc_ci'][1]:.3f}]" if "auroc" in e else "n/a"
        print(f"{e['group']:12s} normal {e['n_normal']:3d} abnormal {e['n_abnormal']:3d} AUROC {auc}  calibration min "
              f"{min(e['calibration_normals_per_fold'])}  marginal caught {e['marginal_sensitivity']['k']}/{e['n_abnormal']} "
              f"FA {e['marginal_false_alarms']['k']}/{e['n_normal']}  group caught {e['group_conditional_sensitivity']['k']}/{e['n_abnormal']} "
              f"FA {e['group_conditional_false_alarms']['k']}/{e['n_normal']}")
    print("overall", overall)


if __name__ == "__main__":
    main()
