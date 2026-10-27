"""WP11 - the frozen DELTA structure expert on a second echo dataset (prepared, not yet run).

The structure expert was fitted and calibrated on one open dataset (DELTA), and inputs with
little or no cardiac anatomy also separate its classes. Only a second dataset, collected
elsewhere, can show whether it reads heart structure. Heartbeat and CARDIUM are behind access
forms that have not been sent; this script is ready for either, or for any dataset described by
an index file. The analysis is fixed in docs/preregistration_external_echo.md before any external
image is seen.

What it does, and nothing else:
  1. checks the frozen artifact (artifacts/structure: probe fitted on DELTA Train, calibrated on
     the normal cases of DELTA Val) and the FetalCLIP weights against their published hashes;
  2. cleans every image with sfr.echo.deidentify and embeds it with FetalCLIP; one mean embedding
     per case (as in run wp5);
  3. scores each case with the frozen probe. It never refits: there is no fitting code here, and
     --refit (or any request to adapt the probe) stops the script;
  4. alarms at level 0.05 with two calibrations: transported (the DELTA calibration in the
     artifact) and local (the normal cases of the new dataset, half chosen by a fixed seed for
     calibration, the other half judged with the abnormal cases);
  5. reports AUROC with a bootstrap interval, sensitivity and false-alarm rate with Clopper-Pearson
     intervals, by image size, and the single-number shortcut checks of the protocol (image width,
     height and mean brightness used as scores, nothing fitted).

Needs data and a GPU-sized amount of time: nothing to run until an external dataset is on disk.

    python scripts/run_wp11_external_echo.py --name heartbeat --root D:/heartbeat --index splits/echo_heartbeat_index.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHA = 0.05
SEED = 0
REFIT_WORDS = ("--refit", "--fit", "--finetune", "--fine-tune", "--retrain", "--adapt")
NEEDS_DATA = ("needs data: no external echo dataset is on disk. Heartbeat and CARDIUM are behind access forms "
              "(request them with the guide's name); then write an index file with case,label,subtype,plane,path")


class RefitRefused(SystemExit):
    pass


def refuse_refit(argv: list[str]) -> None:
    asked = [word for word in argv if word.split("=")[0].lower() in REFIT_WORDS]
    if asked:
        raise RefitRefused(f"refused: {asked[0]} - this script applies the frozen DELTA probe only; refitting on the "
                           "external data would turn an external test into another internal one (see the preregistration)")


def score_cases(features: np.ndarray, arrays: dict) -> np.ndarray:
    """Probability of 'abnormal' from the frozen probe: standardise, project on 32 components, logistic. No fitting."""
    from sfrtool.experts import structure_probability

    return np.array([structure_probability(row, arrays) for row in np.atleast_2d(features)])


def p_values(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    from sfr.fusion.conformal import p_value

    return np.array([p_value(float(s), calibration) for s in scores])


def local_split(labels: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Half of the normal cases (fixed seed) calibrate; the other half and every abnormal case are judged."""
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    calibrate = rng.choice(normal, normal.size // 2, replace=False)
    judged = np.setdiff1d(np.arange(labels.size), calibrate)
    return calibrate, judged


def summarise(labels: np.ndarray, scores: np.ndarray, p: np.ndarray) -> dict:
    from sfr.eval.stats import auroc_ci, clopper_pearson

    out = {"n_abnormal": int((labels == 1).sum()), "n_normal": int((labels == 0).sum())}
    if out["n_abnormal"] and out["n_normal"]:
        value, low, high = auroc_ci(labels, scores, seed=SEED)
        out["auroc"], out["auroc_ci"] = value, [low, high]
    alarm = p <= ALPHA + 1e-12
    for name, member in (("sensitivity", labels == 1), ("false_alarms", labels == 0)):
        k, n = int(alarm[member].sum()), int(member.sum())
        out[name] = {"k": k, "n": n, "share": k / n if n else float("nan"),
                     "ci": list(clopper_pearson(k, n)) if n else [float("nan"), float("nan")]}
    return out


def shortcut_aurocs(labels: np.ndarray, plain: np.ndarray) -> dict:
    """AUROC of image width, height and mean brightness used as scores (nothing fitted). Far from 0.5 either way = a shortcut."""
    from sfr.eval.stats import auroc

    if np.unique(labels).size < 2:
        return {}
    return {name: auroc(labels, plain[:, i]) for i, name in enumerate(("width", "height", "mean_brightness"))}


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"case", "label", "path"} <= set(rows[0]):
        raise SystemExit(f"{path} must have the columns case, label and path")
    return rows


def outside_repository(root: Path) -> bool:
    """Patient images must not live in the repository unless the folder is git-ignored."""
    repo = Path(__file__).resolve().parent.parent
    try:
        root.resolve().relative_to(repo)
    except ValueError:
        return True
    ignored = subprocess.run(["git", "check-ignore", "-q", str(root.resolve())], cwd=repo).returncode == 0
    return ignored


def clean_image(path: str):
    """One image: the cleaned scan (None if no scan region is found), and width, height and mean brightness of the image as released."""
    from PIL import Image

    from sfr.echo.deidentify import deidentify

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    numbers = [rgb.shape[1], rgb.shape[0], float(rgb.mean())]
    try:
        return deidentify(rgb).image, numbers
    except ValueError:
        return None, numbers


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    refuse_refit(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", help="short dataset name; the run is called wp11_external_echo_<name>")
    parser.add_argument("--root", type=Path, help="folder that the index paths are relative to")
    parser.add_argument("--index", type=Path, help="CSV with case, label, subtype, plane, path (image_size optional)")
    parser.add_argument("--source", default="", help="where the dataset comes from (DOI, licence)")
    parser.add_argument("--workers", type=int, default=5, help="processes that clean images")
    args = parser.parse_args(argv)
    if not (args.name and args.root and args.index) or not args.index.exists() or not args.root.exists():
        raise SystemExit(NEEDS_DATA)
    if not outside_repository(args.root):
        raise SystemExit("the image folder is inside the repository and not git-ignored; move it or ignore it first")

    import torch  # noqa: F401  (before the image work: the other order crashes on this machine)
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor

    from sfr.echo.fetalclip import FETALCLIP_SHA256, embed_images, load_fetalclip
    from sfr.provenance import Run
    from sfrtool.artifacts import load_artifact
    from sfrtool.experts import model_resizer

    arrays, card = load_artifact("structure", model={"name": "FetalCLIP", "sha256": FETALCLIP_SHA256})
    rows = read_index(args.index)
    paths = [args.root / r["path"] for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, record = load_fetalclip(device)
    resize = model_resizer(preprocess)
    run = Run(f"wp11_external_echo_{args.name}",
              description="Frozen DELTA structure expert on an external echo dataset (no refitting), as preregistered",
              script="scripts/run_wp11_external_echo.py",
              config={"alpha": ALPHA, "seed": SEED, "artifact_card": {k: card[k] for k in ("dataset", "calibration", "files")},
                      "preregistration": "docs/preregistration_external_echo.md", "source": args.source, "refit": False,
                      "restricted": True,
                      "case_embedding": "mean over every usable image of the case (as preregistered)",
                      "case_image_size": "the most common width x height among the case's images",
                      "unusable_image": "no scan region found by sfr.echo.deidentify; skipped and counted"},
              inputs={"index": [args.index], "images": paths}, n_real=len({r["case"] for r in rows}), models=[record], seed=SEED)
    by_case: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_case.setdefault(r["case"], []).append(i)
    cases = sorted(by_case)
    features, labels, sizes, plain, kept_cases = [], [], [], [], []
    skipped_images, skipped_cases = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, case in enumerate(cases):
            members = by_case[case]
            cleaned = list(pool.map(clean_image, [str(paths[i]) for i in members]))
            images = [resize(image) for image, _ in cleaned if image is not None]
            skipped_images += sum(image is None for image, _ in cleaned)
            if not images:
                skipped_cases += 1
                continue
            numbers = [nums for _, nums in cleaned]
            features.append(embed_images(model, preprocess, images, device).mean(axis=0))
            labels.append(int(rows[members[0]]["label"]))
            sizes.append(rows[members[0]].get("image_size") or Counter(f"{w}x{h}" for w, h, _ in numbers).most_common(1)[0][0])
            plain.append(np.mean(numbers, axis=0))
            kept_cases.append(case)
            if n % 100 == 0:
                print(f"{n} of {len(cases)} cases embedded", flush=True)
    features, labels, sizes, plain = np.array(features), np.array(labels), np.array(sizes), np.array(plain)
    scores = score_cases(features, arrays)
    transported = p_values(scores, arrays["calibration"])
    calibrate, judged = local_split(labels)
    local = np.full(labels.size, np.nan)
    local[judged] = p_values(scores[judged], scores[calibrate])

    metrics = {"transported_calibration": summarise(labels, scores, transported),
               "local_calibration": summarise(labels[judged], scores[judged], local[judged]),
               "local_calibration_normals": int(calibrate.size),
               "by_image_size": {s: summarise(labels[sizes == s], scores[sizes == s], transported[sizes == s]) for s in sorted(set(sizes))},
               "shortcut_single_numbers_auroc": shortcut_aurocs(labels, plain),
               "images": len(rows), "unusable_images_skipped": skipped_images, "cases_without_a_usable_image": skipped_cases,
               "cases": len(kept_cases), "refit": False}
    (run.path / "private").mkdir(parents=True, exist_ok=True)
    run.write_csv("private/per_case.csv", [{"case": c, "label": int(y), "image_size": s, "score": float(v), "p_transported": float(pt),
                                            "p_local": float(pl)} for c, y, s, v, pt, pl in zip(kept_cases, labels, sizes, scores,
                                                                                                  transported, local)])
    run.finish(metrics)
    print({k: v for k, v in metrics.items() if k != "by_image_size"})


if __name__ == "__main__":
    main()
