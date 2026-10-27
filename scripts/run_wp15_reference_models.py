"""WP15 - the Heartbeat authors' released models, run with their own code on their own splits.

Gate G5 asks whether a released reference model can be reproduced. Heartbeat releases Heart-ViT
(4 fold models per trimester) and five ImageNet-initialised baselines (2nd trimester). This run
loads every released checkpoint into the authors' architecture, predicts with the authors'
inference and metric code (github.com/BCV-Uniandes/Heartbeat, kept outside this repository in
data/third_party/Heartbeat because its terms do not let us copy it), and compares the 2nd-trimester
test results with the numbers in the authors' README. Exploratory, after the preregistered runs.

Nothing is trained. Models are built without downloading ImageNet weights (every parameter comes
from the released checkpoint, loaded strictly); the per-fold settings and thresholds are read from
the authors' test scripts, not typed here.

Settings fixed before running:
  reproduced      each of mean sensitivity, specificity and AUROC of Heart-ViT on the 2nd-trimester
                  test set lies within 0.5 percentage points of the README value
  trimester shift (exploratory) every 2nd-trimester model is also applied to the 3rd-trimester
                  patients who were never in the 2nd-trimester data (so no fetus is seen twice);
                  AUROC and sensitivity / specificity at the fold's own test threshold

Per-patient predictions go to runs/wp15_reference_models/private/ (not in git).

    python scripts/run_wp15_reference_models.py
"""

from __future__ import annotations

import torch  # noqa: F401  (first: importing it after SciPy work crashes on this machine)

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sfr.eval.stats import auroc_ci  # noqa: E402
from sfr.paths import DATA  # noqa: E402
from sfr.provenance import Run  # noqa: E402

AUTHORS = DATA / "third_party" / "Heartbeat"
DATASET = DATA / "heartbeat" / "dataset"
CHECKPOINTS = {"heart-vit": DATA / "heartbeat" / "checkpoints - heartbeat",
               "baselines": DATA / "heartbeat" / "checkpoints - Baseline"}
REPORTED_SOURCE = "github.com/BCV-Uniandes/Heartbeat README (commit 91d9772), read on 27 Oct 2026"
REPORTED_2T_TEST = {   # mean and SD over the four fold models, in percent, as printed in the README
    "mobilenet": {"sensitivity": [29.17, 25.00], "specificity": [87.73, 7.62], "ppv": [15.30, 11.87], "auroc": [82.73, 5.06], "f1": [19.90, 15.98]},
    "vgg16": {"sensitivity": [62.50, 15.96], "specificity": [88.64, 6.86], "ppv": [41.28, 12.77], "auroc": [85.46, 2.98], "f1": [47.48, 5.02]},
    "resnet18": {"sensitivity": [66.67, 23.57], "specificity": [87.73, 4.78], "ppv": [37.30, 4.89], "auroc": [86.44, 4.32], "f1": [46.67, 9.03]},
    "resnet50": {"sensitivity": [75.00, 9.62], "specificity": [85.46, 4.92], "ppv": [37.45, 8.60], "auroc": [85.53, 2.48], "f1": [49.22, 6.23]},
    "vit": {"sensitivity": [70.84, 8.33], "specificity": [88.64, 5.01], "ppv": [42.70, 11.13], "auroc": [85.83, 2.73], "f1": [52.40, 7.97]},
    "heart-vit": {"sensitivity": [79.17, 8.33], "specificity": [93.18, 2.29], "ppv": [56.95, 6.99], "auroc": [88.56, 3.80], "f1": [65.63, 2.09]},
}
REPORTED_CV_F1 = {"2T": [52.17, 71.43, 60.00, 66.67], "3T": [72.00, 66.67, 66.67, 66.67]}   # Heart-ViT, per fold
TOLERANCE_POINTS = 0.5
BASELINES = ("vit", "resnet18", "resnet50", "vgg16", "mobilenet")
SEED = 0


def settings() -> dict:
    """Per-fold settings and thresholds, read from the authors' test scripts."""
    out = {}
    for cohort in ("2T", "3T"):
        text = (AUTHORS / "scripts" / "test" / cohort / "heart-vit.sh").read_text(encoding="utf-8")
        for split, fold, dim, layers, heads, threshold in re.findall(
                r"^run_fold\s+(\w+)\s+(\d)\s+(\d+)\s+(\d+)\s+(\d+)\s*([\d.]+)?\s*$", text, flags=re.M):
            out[("heart-vit", cohort, split, int(fold))] = {"dim": int(dim), "layers": int(layers), "heads": int(heads),
                                                            "threshold": float(threshold) if threshold else None}
    text = (AUTHORS / "scripts" / "test" / "2T" / "baselines.sh").read_text(encoding="utf-8")
    for arch, split, fold, threshold in re.findall(r"^run_fold\s+(\w+)\s+(\w+)\s+(\d)\s*([\d.]+)?\s*$", text, flags=re.M):
        out[(arch, "2T", split, int(fold))] = {"threshold": float(threshold) if threshold else None}
    return out


def build(arch: str, setting: dict):
    """The authors' architectures, without fetching ImageNet weights (the checkpoint holds every parameter)."""
    import torch.nn as nn
    import torchvision

    from models.baselines import ViT
    from models.heart_vit import HeartViT

    class FlatBatch(nn.Module):              # same layout as the authors' _FlatBatch: the network sits in .model
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    if arch == "heart-vit":
        return HeartViT(2, view_embedding_dim=setting["dim"], adanorm_layers=setting["layers"],
                        adanorm_attention_heads=setting["heads"], dropout=0.0, metadata_values=4, pretrained=False), True
    if arch == "vit":
        return ViT(2, pretrained=False), False
    if arch in ("resnet18", "resnet50"):
        network = getattr(torchvision.models, arch)(weights=None)
        network.fc = nn.Linear(network.fc.in_features, 2, bias=True)
        return FlatBatch(network), False
    if arch == "vgg16":
        network = torchvision.models.vgg16(weights=None)
        network.classifier[-1] = nn.Linear(4096, 2)
        return FlatBatch(network), False
    if arch == "mobilenet":
        network = torchvision.models.mobilenet_v2(weights=None)
        network.classifier[1] = nn.Linear(network.last_channel, 2)
        return FlatBatch(network), False
    raise ValueError(arch)


def checkpoint(arch: str, cohort: str, fold: int) -> Path:
    folder = CHECKPOINTS["heart-vit" if arch == "heart-vit" else "baselines"]
    return folder / cohort / arch / f"{arch}_fold_{fold}.pth"


def val_codes(cohort: str, fold: int) -> set[str]:
    with (DATASET / cohort / "cross_val" / f"fold{fold}.csv").open(encoding="utf-8", newline="") as handle:
        return {r["patient_code"] for r in csv.DictReader(handle) if r["split"] == "val"}


def all_codes(cohort: str, part: str) -> set[str]:
    with (DATASET / cohort / part / "metadata_raw.csv").open(encoding="utf-8", newline="") as handle:
        return {r["patient_code"] for r in csv.DictReader(handle)}


def summary(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd_population": float(values.std(ddof=0)),
            "sd_sample": float(values.std(ddof=1)) if values.size > 1 else float("nan"), "per_fold": values.tolist()}


def main() -> None:
    if not AUTHORS.exists():
        raise SystemExit(f"needs the authors' code at {AUTHORS} (git clone --depth 1 https://github.com/BCV-Uniandes/Heartbeat)")
    sys.path.insert(0, str(AUTHORS))
    import metrics as authors_metrics                            # the authors' package
    from dataloader import DelfosPatientDataset
    from evaluation import predict

    plan = settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_2t = all_codes("2T", "cross_val") | all_codes("2T", "test")
    unseen_3t = all_codes("3T", "cross_val") - in_2t
    run = Run("wp15_reference_models",
              description="Heartbeat's released Heart-ViT and baseline checkpoints, run with the authors' code on their splits "
                          "(exploratory; after the preregistered external test)",
              script="scripts/run_wp15_reference_models.py",
              config={"restricted": True, "authors_code": "github.com/BCV-Uniandes/Heartbeat (data/third_party/Heartbeat)",
                      "reported_source": REPORTED_SOURCE, "reported_2T_test": REPORTED_2T_TEST, "reported_cv_f1": REPORTED_CV_F1,
                      "tolerance_points": TOLERANCE_POINTS, "gate_rule": "Heart-ViT 2T test: mean sensitivity, specificity and AUROC each within the tolerance",
                      "settings": {"|".join(map(str, k)): v for k, v in plan.items()}, "device": device,
                      "trimester_shift": "2T models on the 3T patients never seen in 2T data, at each fold's own test threshold"},
              inputs={"checkpoints": sorted(p for folder in CHECKPOINTS.values() for p in folder.rglob("*.pth")),
                      "metadata": sorted(DATASET.rglob("*.csv"))},
              n_real=len(in_2t | all_codes("3T", "cross_val")), seed=SEED)
    (run.path / "private").mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict = {}
    datasets: dict = {}

    def dataset(cohort: str, part: str, codes: set[str] | None):
        key = (cohort, part, None if codes is None else len(codes), None if codes is None else min(codes))
        if key not in datasets:
            root = DATASET / cohort / part
            datasets[key] = DelfosPatientDataset(str(root), str(root / "metadata_normalized.csv"), patient_codes=codes)
        return datasets[key]

    jobs = []
    for (arch, cohort, split, fold), setting in sorted(plan.items()):
        jobs.append((arch, cohort, split, fold, setting))
        if cohort == "2T" and split == "test":
            jobs.append((arch, "3T-unseen", "cross_val", fold, setting))
    private = {}
    for arch, cohort, split, fold, setting in jobs:
        source_cohort = "2T" if cohort == "3T-unseen" else cohort
        model, conditioned = build(arch, setting)
        state = torch.load(checkpoint(arch, source_cohort, fold), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        if cohort == "3T-unseen":
            data = dataset("3T", "cross_val", unseen_3t)
        elif split == "test":
            data = dataset(cohort, "test", None)
        else:
            data = dataset(cohort, "cross_val", val_codes(cohort, fold))
        rows = predict(model, data, device=device, conditioned=conditioned)
        labels = [label for _code, label, _prob, _ in rows]
        probs = [prob for _code, _label, prob, _ in rows]
        scored = authors_metrics.evaluate(labels, probs, threshold=setting["threshold"] if split == "test" or cohort == "3T-unseen" else None)
        point, low, high = auroc_ci(labels, probs, seed=SEED)
        scored.update({"n": len(rows), "n_chd": int(sum(labels)), "auroc_ci": [low, high]})
        results.setdefault(f"{arch}|{cohort}|{split}", {})[fold] = scored
        private[f"{arch}|{cohort}|{split}|{fold}"] = {code: [label, prob] for code, label, prob, _ in rows}
        print(f"{arch:10s} {cohort:9s} {split:9s} fold {fold}: AUROC {scored['auroc']:.4f} sens {scored['sensitivity']:.4f} "
              f"spec {scored['specificity']:.4f} thr {scored['threshold']:.2f} n {len(rows)} ({time.time() - started:.0f} s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    aggregated = {}
    for key, folds in results.items():
        aggregated[key] = {name: summary([folds[f][name] for f in sorted(folds)]) for name in ("auroc", "sensitivity", "specificity", "ppv", "f1")}
        aggregated[key]["n"] = [folds[f]["n"] for f in sorted(folds)]
        aggregated[key]["n_chd"] = [folds[f]["n_chd"] for f in sorted(folds)]
        aggregated[key]["auroc_ci_per_fold"] = [folds[f]["auroc_ci"] for f in sorted(folds)]
    versus_readme = {arch: {name: 100 * aggregated[f"{arch}|2T|test"][name]["mean"] - reported[name][0] for name in reported}
                     for arch, reported in REPORTED_2T_TEST.items() if f"{arch}|2T|test" in aggregated}
    cv_f1 = {cohort: [100 * results[f"heart-vit|{cohort}|cross_val"][fold]["f1"] - value
                      for fold, value in zip(sorted(results[f"heart-vit|{cohort}|cross_val"]), values)]
             for cohort, values in REPORTED_CV_F1.items() if f"heart-vit|{cohort}|cross_val" in results}
    differences = {name: versus_readme["heart-vit"][name] for name in ("sensitivity", "specificity", "auroc")}
    reproduced = all(abs(d) <= TOLERANCE_POINTS for d in differences.values())
    (run.path / "private" / "predictions.json").write_text(json.dumps(private, indent=1), encoding="utf-8")
    run.finish({"results": aggregated, "heart_vit_2T_test_minus_reported_points": differences,
                "all_models_2T_test_minus_readme_points": versus_readme, "heart_vit_cv_f1_minus_readme_points_per_fold": cv_f1,
                "gate_G5_reference_model_reproduced": reproduced,
                "unseen_3T_patients": len(unseen_3t),
                "note": "means over the 4 fold models, as the authors report; per-patient predictions are in private/ (not in git)"})
    print("Heart-ViT 2T test minus README (points):", {k: round(v, 2) for k, v in differences.items()}, "reproduced:", reproduced)


if __name__ == "__main__":
    main()
