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
5. Two checks added on 12 Oct 2026 after reading the documentation of the first dataset and
   before any image was processed: (a) if the index has a ``split`` column with Train, Val and
   Test, the dataset's own partition is also used - probe fitted on Train, alarm calibrated on
   the normal cases of Val, judged on Test; (b) the main probes are repeated inside the most
   common image size, because image size can differ between normal and abnormal cases.

Nothing is tuned here: the probe, the folds and the alarm level are fixed in this file, and the
script is run once per dataset.

    python scripts/run_wp5_structure.py --name delta --root data/delta/extracted --index splits/echo_delta_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch        # must be imported before NumPy/SciPy code runs: the other order crashes on this machine
if multiprocessing.current_process().name == "MainProcess":    # the image-cleaning workers do not need the model code
    import open_clip    # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.echo.deidentify import deidentify, to_grey  # noqa: E402
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
        row.setdefault("split", "")
    return rows


def case_attribute(rows: list[dict], cases: list[str], key: str) -> np.ndarray:
    value = {row["case"]: row.get(key, "") for row in rows}
    return np.array([value[case] for case in cases])


def official_split(x: np.ndarray, y: np.ndarray, split: np.ndarray, subtype: list[str], seed: int = SEED) -> dict:
    """The dataset's own partition: probe fitted on Train, alarm calibrated on the normal cases of Val, judged on Test."""
    train, val, test = split == "Train", split == "Val", split == "Test"
    model = probe(x, seed).fit(x[train], y[train])
    scores = model.predict_proba(x[test])[:, 1]
    value, low, high = auroc_ci(y[test], scores, seed=seed)
    healthy = model.predict_proba(x[val & (y == 0)])[:, 1]
    p = np.array([p_value(float(score), healthy) for score in scores])
    return {"auroc": value, "ci": [low, high], "fitted_on_cases": int(train.sum()),
            "calibration_normal_cases": int((val & (y == 0)).sum()),
            "test_abnormal": int((y[test] == 1).sum()), "test_normal": int((y[test] == 0).sum()),
            "alarm": alarm_summary(y[test], p, [kind for kind, use in zip(subtype, test) if use])}


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


def clean_one(path: str):
    """One image: the cleaned scan, what the cleaning removed, the image as released, and six numbers without anatomy.

    The removed part is built by the steps of ``sfr.echo.deidentify.overlay_only`` without cleaning
    the image a second time (tests/test_structure_probe.py checks that the two are identical).
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    result = deidentify(rgb)
    grey = to_grey(rgb)
    top, bottom, left, right = result.box
    kept = np.zeros_like(grey)
    kept[top:bottom, left:right] = result.image
    removed = grey.copy()
    removed[kept > 0] = 0
    numbers = [rgb.shape[1], rgb.shape[0], float(rgb.mean()), result.scan_share,
               float(result.coloured_removed), float(result.white_strokes_removed)]
    return result.image, removed, rgb, numbers


def to_model_size(preprocess):
    """The resizing part of the model's own preprocessing, applied once so that both networks see the same pixels."""
    from PIL import Image

    steps = []
    for step in preprocess.transforms:
        if "ToTensor" in type(step).__name__:
            break
        steps.append(step)

    def apply(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
        for step in steps:
            image = step(image)
        return np.asarray(image)

    return apply, [type(step).__name__ for step in steps]


def embed_dataset(paths: list[Path], device: str, model, preprocess, workers: int):
    """Embeddings of the cleaned scan, the removed overlays and the released image; the resized scans; the six numbers."""
    resize, steps = to_model_size(preprocess)
    print(f"  resizing steps taken from the model's preprocessing: {steps}; {workers} cleaning processes", flush=True)
    features = {"scan": [], "overlay": [], "raw": []}
    scans, trivial = [], []
    chunks = [[str(path) for path in paths[start:start + CHUNK]] for start in range(0, len(paths), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = pool.map(clean_one, chunks[0])
        for index in range(len(chunks)):
            cleaned = list(pending)
            if index + 1 < len(chunks):                # clean the next chunk while this one is embedded
                pending = pool.map(clean_one, chunks[index + 1])
            stacks = {"scan": [resize(item[0]) for item in cleaned], "overlay": [resize(item[1]) for item in cleaned],
                      "raw": [resize(item[2]) for item in cleaned]}
            trivial.extend(item[3] for item in cleaned)
            scans.extend(stacks["scan"])
            for name, stack in stacks.items():
                features[name].append(embed_images(model, preprocess, stack, device, batch=CHUNK))
            if index % 10 == 0:
                print(f"  embedded {min((index + 1) * CHUNK, len(paths))} of {len(paths)} images", flush=True)
    return {name: np.concatenate(parts) for name, parts in features.items()}, np.array(trivial), scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="short dataset name; the run is called wp5_structure_<name>")
    parser.add_argument("--root", required=True, type=Path, help="folder that the index paths are relative to")
    parser.add_argument("--index", required=True, type=Path, help="CSV with case, label, subtype, plane, path")
    parser.add_argument("--source", default="", help="where the dataset comes from (DOI, licence), written to the run")
    parser.add_argument("--workers", type=int, default=5, help="processes that clean images")
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
    published, trivial, scans = embed_dataset(paths, device, model, preprocess, args.workers)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    twin, twin_preprocess, control_record = random_twin(SEED, device)
    print("embedding the same resized scans with random weights", flush=True)
    untrained = {"scan": embed_images(twin, twin_preprocess, scans, device, batch=CHUNK)}
    del twin, scans

    inputs = {"fetalclip_scan": published["scan"], "fetalclip_overlay_only": published["overlay"],
              "fetalclip_unaltered_image": published["raw"], "no_anatomy_numbers": trivial,
              "random_weights_scan": untrained["scan"]}
    cases, _, y, subtype = by_case(rows, published["scan"])
    case_x = {name: by_case(rows, features)[1] for name, features in inputs.items()}
    results, scores = {}, {}
    for name in inputs:
        results[name], scores[name] = evaluate(case_x[name], y)
        print(name, results[name], flush=True)

    x_scan = case_x["fetalclip_scan"]
    p = conformal_p_values(x_scan, y)
    alarms = alarm_summary(y, p, subtype)

    split = case_attribute(rows, cases, "split")
    own_partition = {}
    if {"Train", "Val", "Test"} <= set(split):
        own_partition = {name: official_split(case_x[name], y, split, subtype) for name in inputs}
        print("dataset's own partition:", {name: round(value["auroc"], 3) for name, value in own_partition.items()}, flush=True)

    size = np.array([f"{int(round(width))}x{int(round(height))}" for width, height in case_x["no_anatomy_numbers"][:, :2]])
    size_table = [{"size": name, "normal": int(((size == name) & (y == 0)).sum()), "abnormal": int(((size == name) & (y == 1)).sum())}
                  for name in sorted(set(size))]
    common = max(size_table, key=lambda entry: entry["normal"] + entry["abnormal"])["size"]
    member = size == common
    same_size = {"size": common, "cases": int(member.sum()), "abnormal": int((y[member] == 1).sum())}
    if min((y[member] == 1).sum(), (y[member] == 0).sum()) >= MIN_PER_CLASS:
        same_size["probes"] = {name: evaluate(case_x[name][member], y[member], with_permutation=False)[0]
                               for name in ("fetalclip_scan", "no_anatomy_numbers", "random_weights_scan")}

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
        {"case": case, "label": int(label), "subtype": kind, "split": split[i], "image_size": size[i],
         "p_value": float(p_i), "alarm": bool(p_i <= ALPHA),
         **{f"score_{name}": float(scores[name][i]) for name in inputs}}
        for i, (case, label, kind, p_i) in enumerate(zip(cases, y, subtype, p))])
    np.savez_compressed(run.path / "embeddings.npz", cases=np.array(cases), labels=y, **case_x)

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
        "own_partition": own_partition,
        "image_size_by_label": size_table,
        "same_image_size": same_size,
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
