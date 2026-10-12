"""WP5 - structure expert on a labelled fetal echo dataset (frozen FetalCLIP, linear probe).

Works on any dataset described by an index file (CSV) with one row per image:

    case,label,subtype,plane,path
    C0001,0,normal,4CH,images/C0001/4ch.png

``label`` is 0 (normal) or 1 (abnormal); ``subtype`` and ``plane`` may be empty; ``path`` is
relative to ``--root``. All images of a case stay on the same side of every split.

What is computed:

1. Shortcut audit. The same probe is fitted on (a) images that hold only what the
   de-identification step removes (text, calipers, colour boxes) and (b) six numbers that carry
   no anatomy (image size, brightness, share of the frame taken by the scan, overlay counts).
   If either separates the classes, the label leaks through something other than the heart.
2. The structure expert: frozen FetalCLIP features of the de-identified scan, averaged over the
   images of a case, then standardise, PCA and logistic regression, in repeated stratified folds.
3. The same network with random weights - the control used for ECGFounder in WP4c.
4. Alarm at the level used by the decision rule: split-conformal p-values against normal cases
   on which the probe was not fitted.

Nothing is tuned here: the probe, the folds and the alarm level are fixed in this file, and the
script is run once per dataset.

    python scripts/run_wp5_structure.py --name delta --root data/delta/extracted --index splits/echo_delta_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch        # must be imported before NumPy/SciPy code runs: the other order crashes on this machine
import open_clip    # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.echo.deidentify import deidentify, overlay_only  # noqa: E402
from sfr.echo.fetalclip import FETALCLIP_CONFIG, embed_images, load_fetalclip  # noqa: E402
from sfr.eval.stats import auroc, auroc_ci, clopper_pearson  # noqa: E402
from sfr.fusion.conformal import p_value  # noqa: E402
from sfr.provenance import Run  # noqa: E402

ALPHA = 0.05                 # alarm level of the structure expert in the decision rule
N_FOLDS, N_REPEATS = 5, 5
N_COMPONENTS = 32
N_PERM = 200
CALIBRATION_SHARE = 0.25     # share of each training fold kept aside for conformal calibration
MIN_PER_CLASS = 20           # a plane or subtype with fewer cases is listed, not analysed
CHUNK = 32
SEED = 0
TRIVIAL = ("width", "height", "mean_grey", "scan_share", "coloured_removed", "white_strokes_removed")


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"case", "label", "path"} <= set(rows[0]):
        raise ValueError(f"{path} must have the columns case, label and path (subtype and plane are optional).")
    for row in rows:
        if row["label"] not in ("0", "1"):
            raise ValueError(f"label must be 0 or 1, found {row['label']!r} for case {row['case']}")
        row.setdefault("subtype", "")
        row.setdefault("plane", "")
    return rows


def by_case(rows: list[dict], features: np.ndarray, keep: np.ndarray | None = None):
    """Mean feature vector, label and subtype of every case. ``keep`` selects images (for one plane)."""
    keep = np.ones(len(rows), dtype=bool) if keep is None else keep
    cases = sorted({row["case"] for row, use in zip(rows, keep) if use})
    position = {case: i for i, case in enumerate(cases)}
    total = np.zeros((len(cases), features.shape[1]))
    count = np.zeros(len(cases))
    label = np.full(len(cases), -1)
    subtype = [""] * len(cases)
    for row, feature, use in zip(rows, features, keep):
        if not use:
            continue
        i = position[row["case"]]
        if label[i] not in (-1, int(row["label"])):
            raise ValueError(f"case {row['case']} has images with different labels")
        label[i], subtype[i] = int(row["label"]), row["subtype"]
        total[i] += feature
        count[i] += 1
    return cases, total / count[:, None], label, subtype


def probe(x: np.ndarray, seed: int = SEED):
    components = min(N_COMPONENTS, x.shape[1])
    return make_pipeline(StandardScaler(), PCA(n_components=components, random_state=seed),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def fold_scores(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold probability of 'abnormal' from one pass of stratified folds over cases."""
    scores = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train, test in folds.split(x, y):
        scores[test] = probe(x, seed).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    return scores


def cross_validated_scores(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Out-of-fold scores averaged over repeated folds. Every score comes from probes that never saw the case."""
    return np.mean([fold_scores(x, y, seed + repeat) for repeat in range(N_REPEATS)], axis=0)


def permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """One-sided p-value of the cross-validated AUROC: labels shuffled over cases, probe refitted."""
    observed = auroc(y, fold_scores(x, y, seed))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        count += auroc(shuffled, fold_scores(x, shuffled, seed)) >= observed
    return (count + 1) / (n_perm + 1)


def conformal_p_values(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Split-conformal p-value of every case against normal cases the probe was not fitted on.

    In each fold the training cases are split once more. The probe is fitted on one part; the
    normal cases of the other part give the calibration scores; the test cases of the fold are
    ranked among them.
    """
    p = np.full(len(y), np.nan)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train, test in folds.split(x, y):
        fit, calibration = train_test_split(train, test_size=CALIBRATION_SHARE, stratify=y[train], random_state=seed)
        model = probe(x, seed).fit(x[fit], y[fit])
        healthy = model.predict_proba(x[calibration[y[calibration] == 0]])[:, 1]
        for i, score in zip(test, model.predict_proba(x[test])[:, 1]):
            p[i] = p_value(float(score), healthy)
    return p


def evaluate(x: np.ndarray, y: np.ndarray, with_permutation: bool = True, seed: int = SEED) -> tuple[dict, np.ndarray]:
    scores = cross_validated_scores(x, y, seed)
    value, low, high = auroc_ci(y, scores, seed=seed)
    result = {"auroc": value, "ci": [low, high], "n_abnormal": int((y == 1).sum()), "n_normal": int((y == 0).sum())}
    if with_permutation:
        result["p_value"] = permutation_p(x, y, seed=seed)
        result["p_method"] = f"labels shuffled over cases, probe refitted, {N_PERM} permutations"
    return result, scores


def alarm_summary(y: np.ndarray, p: np.ndarray, subtype: list[str]) -> dict:
    alarm = p <= ALPHA
    caught, n_abnormal = int(alarm[y == 1].sum()), int((y == 1).sum())
    false, n_normal = int(alarm[y == 0].sum()), int((y == 0).sum())
    rows = []
    for name in sorted({s for s, label in zip(subtype, y) if label == 1 and s}):
        member = np.array([s == name and label == 1 for s, label in zip(subtype, y)])
        low, high = clopper_pearson(int(alarm[member].sum()), int(member.sum()))
        rows.append({"subtype": name, "cases": int(member.sum()), "caught": int(alarm[member].sum()),
                     "ci": [low, high], "too_few_to_interpret": bool(member.sum() < MIN_PER_CLASS)})
    return {"alpha": ALPHA,
            "sensitivity": {"caught": caught, "n": n_abnormal, "share": caught / max(n_abnormal, 1),
                            "ci": list(clopper_pearson(caught, n_abnormal))},
            "false_alarms": {"flagged": false, "n": n_normal, "share": false / max(n_normal, 1),
                             "ci": list(clopper_pearson(false, n_normal))},
            "by_subtype": rows}


def twin_record(seed: int) -> dict:
    return {"name": "random-init network with the FetalCLIP architecture", "seed": seed, "verified": False}


def random_twin(seed: int, device: str):
    """The FetalCLIP architecture with untrained weights. Never called by the published name."""
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = json.loads(FETALCLIP_CONFIG.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP")
    model.eval().to(device)
    return model, preprocess, twin_record(seed)


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def embed_dataset(paths: list[Path], device: str, model, preprocess, variants: tuple[str, ...]):
    """Embeddings of every image for the asked variants, and the numbers that carry no anatomy."""
    features = {name: [] for name in variants}
    trivial = []
    for start in range(0, len(paths), CHUNK):
        images = [load_rgb(path) for path in paths[start:start + CHUNK]]
        cleaned = [deidentify(image) for image in images]
        for image, result in zip(images, cleaned):
            trivial.append([image.shape[1], image.shape[0], float(image.mean()), result.scan_share,
                            float(result.coloured_removed), float(result.white_strokes_removed)])
        stacks = {"scan": [result.image for result in cleaned],
                  "overlay": [overlay_only(image) for image in images] if "overlay" in variants else [],
                  "raw": images}
        for name in variants:
            features[name].append(embed_images(model, preprocess, stacks[name], device, batch=CHUNK))
        if (start // CHUNK) % 10 == 0:
            print(f"  embedded {min(start + CHUNK, len(paths))} of {len(paths)} images", flush=True)
    return {name: np.concatenate(parts) for name, parts in features.items()}, np.array(trivial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="short dataset name; the run is called wp5_structure_<name>")
    parser.add_argument("--root", required=True, type=Path, help="folder that the index paths are relative to")
    parser.add_argument("--index", required=True, type=Path, help="CSV with case, label, subtype, plane, path")
    parser.add_argument("--source", default="", help="where the dataset comes from (DOI, licence), written to the run")
    args = parser.parse_args()

    rows = read_index(args.index)
    paths = [args.root / row["path"] for row in rows]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images of the index are missing, first: {missing[0]}")

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, record = load_fetalclip(device)
    run = Run(
        f"wp5_structure_{args.name}",
        description="Structure expert: frozen FetalCLIP features, linear probe with case-level folds, shortcut audit, conformal alarm.",
        script="scripts/run_wp5_structure.py",
        config={"dataset": args.name, "source": args.source, "alpha": ALPHA, "folds": N_FOLDS, "repeats": N_REPEATS,
                "probe": f"standardise, PCA({N_COMPONENTS}), logistic regression C=1 with balanced class weights",
                "permutations": N_PERM, "calibration_share": CALIBRATION_SHARE, "device": device},
        inputs={"index": [args.index], "images": paths},
        n_real=len({row["case"] for row in rows}),
        models=[record, twin_record(SEED)],
        seed=SEED,
    )
    print(f"{len(rows)} images of {len({row['case'] for row in rows})} cases; embedding with the published weights", flush=True)
    published, trivial = embed_dataset(paths, device, model, preprocess, ("scan", "overlay", "raw"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    twin, twin_preprocess, control_record = random_twin(SEED, device)
    print("embedding with random weights", flush=True)
    untrained, _ = embed_dataset(paths, device, twin, twin_preprocess, ("scan",))
    del twin

    inputs = {"fetalclip_scan": published["scan"], "fetalclip_overlay_only": published["overlay"],
              "fetalclip_unaltered_image": published["raw"], "no_anatomy_numbers": trivial,
              "random_weights_scan": untrained["scan"]}
    cases, _, y, subtype = by_case(rows, published["scan"])
    results, scores = {}, {}
    for name, features in inputs.items():
        _, x, _, _ = by_case(rows, features)
        results[name], scores[name] = evaluate(x, y)
        print(name, results[name], flush=True)

    _, x_scan, _, _ = by_case(rows, published["scan"])
    p = conformal_p_values(x_scan, y)
    alarms = alarm_summary(y, p, subtype)

    planes = []
    for plane in sorted({row["plane"] for row in rows if row["plane"]}):
        keep = np.array([row["plane"] == plane for row in rows])
        _, x_plane, y_plane, _ = by_case(rows, published["scan"], keep)
        entry = {"plane": plane, "cases": int(len(y_plane)), "abnormal": int((y_plane == 1).sum())}
        if min((y_plane == 1).sum(), (y_plane == 0).sum()) >= MIN_PER_CLASS:
            entry.update(evaluate(x_plane, y_plane, with_permutation=False)[0])
        else:
            entry["note"] = f"fewer than {MIN_PER_CLASS} cases in a class; not analysed"
        planes.append(entry)

    run.write_csv("per_case.csv", [
        {"case": case, "label": int(label), "subtype": kind, "p_value": float(p_i), "alarm": bool(p_i <= ALPHA),
         **{f"score_{name}": float(scores[name][i]) for name in inputs}}
        for i, (case, label, kind, p_i) in enumerate(zip(cases, y, subtype, p))])
    np.savez_compressed(run.path / "embeddings.npz", cases=np.array(cases), labels=y, **{
        name: by_case(rows, features)[1] for name, features in inputs.items()})

    audit_overlay, audit_trivial = results["fetalclip_overlay_only"], results["no_anatomy_numbers"]
    metrics = {
        "dataset": {"name": args.name, "source": args.source, "images": len(rows), "cases": len(cases),
                    "abnormal_cases": int((y == 1).sum()), "normal_cases": int((y == 0).sum())},
        "probes": results,
        "shortcut_audit": {
            "overlay_only_auroc": audit_overlay["auroc"], "overlay_only_ci": audit_overlay["ci"],
            "no_anatomy_numbers_auroc": audit_trivial["auroc"], "no_anatomy_numbers_ci": audit_trivial["ci"],
            "reading": "an interval that excludes 0.5 means the label can be read without looking at the heart",
        },
        "alarm": alarms,
        "by_plane": planes,
        "models": {"published": record, "control": control_record},
        "gate_G5": {"auroc_with_interval": True, "shortcut_audit_reported": True,
                    "reference_model_reproduced": False,
                    "reason": "no released reference model exists for this dataset; the other two parts are reported"},
        "minutes": (time.time() - started) / 60,
    }
    run.finish(metrics)
    print(json.dumps({"alarm": alarms["sensitivity"], "false_alarms": alarms["false_alarms"]}, indent=1))


if __name__ == "__main__":
    main()
