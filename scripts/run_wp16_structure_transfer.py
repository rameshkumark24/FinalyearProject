"""WP16 - structure probes moved between datasets, and from the 2nd to the 3rd trimester.

A probe fitted on one echo dataset is applied to another; the drop from the within-dataset AUROC
shows how much of what a probe learns belongs to its dataset. The trimester part compares the
FetalCLIP probe with the authors' ImageNet-initialised baselines and Heart-ViT (hypothesis H6 of
the plan: FetalCLIP loses less when the trimester changes). Exploratory, after the preregistered
external test; only linear probes are fitted, on saved FetalCLIP case embeddings.

Settings, fixed before computing:
  embeddings     FetalCLIP scan embeddings, mean per case, from runs wp5_structure_delta,
                 wp5_structure_heartbeat_2T, wp5_structure_heartbeat_3T and wp5_structure_cardium
  probe          standardise, PCA(32), logistic regression C=1 with balanced class weights (as wp5),
                 fitted on every case of the source dataset
  no fetus twice the target leaves out patients seen in the source: Heartbeat 2T <-> 3T by patient
                 code, Heartbeat <-> CARDIUM by the matches of run wp14 (exact or possible)
  trimester      FetalCLIP: fitted on the training patients of each of the authors' 4 folds (2T),
                 AUROC on that fold's validation patients and on the 3T patients never seen in 2T;
                 the authors' models: the same two AUROCs from run wp15
  intervals      bootstrap over cases, 2000 resamples, seed 0

Only aggregate numbers are written (restricted data).

    python scripts/run_wp16_structure_transfer.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import auroc, auroc_ci  # noqa: E402
from sfr.paths import DATA, RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

SOURCES = {"delta": RUNS / "wp5_structure_delta" / "embeddings.npz",
           "heartbeat_2T": RUNS / "wp5_structure_heartbeat_2T" / "private" / "embeddings.npz",
           "heartbeat_3T": RUNS / "wp5_structure_heartbeat_3T" / "private" / "embeddings.npz",
           "cardium": RUNS / "wp5_structure_cardium" / "private" / "embeddings.npz"}
MATCHES = RUNS / "wp14_echo_overlap" / "private" / "matched_patients.csv"
INDEX_2T = DATA / "indexes" / "echo_heartbeat_2T_index.csv"
FEATURES, N_COMPONENTS, SEED = "fetalclip_scan", 32, 0


def probe():
    return make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=SEED),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def patient(dataset: str, case: str) -> str:
    return case.rsplit("_", 1)[0] if dataset.startswith("heartbeat") else case


def load(dataset: str) -> dict:
    data = np.load(SOURCES[dataset], allow_pickle=False)
    cases = [str(c) for c in data["cases"]]
    return {"cases": cases, "patients": [patient(dataset, c) for c in cases], "y": data["labels"].astype(int), "x": data[FEATURES]}


def shared_patients() -> dict[tuple[str, str], set[tuple[str, str]]]:
    pairs: dict[tuple[str, str], set[tuple[str, str]]] = {}
    with MATCHES.open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            pairs.setdefault((r["first"], r["second"]), set()).add((r["first_patient"], r["second_patient"]))
    return pairs


def seen_in(source: str, target: str, data: dict, pairs: dict) -> set[str]:
    """Patients of the target that also appear in the source: by patient code across the Heartbeat
    trimesters, by the image matches of run wp14 otherwise."""
    if {source, target} == {"heartbeat_2T", "heartbeat_3T"}:
        return set(data[source]["patients"]) & set(data[target]["patients"])
    seen: set[str] = set()
    for (first, second), matched in pairs.items():
        if (first, second) == (source, target):
            seen |= {q for _, q in matched}
        elif (first, second) == (target, source):
            seen |= {p for p, _ in matched}
    return seen


def main() -> None:
    for name in ("wp5_structure_delta", "wp5_structure_heartbeat_2T", "wp5_structure_heartbeat_3T", "wp5_structure_cardium",
                 "wp14_echo_overlap", "wp15_reference_models"):
        load_run(name)
    data = {name: load(name) for name in SOURCES}
    pairs = shared_patients()
    run = Run("wp16_structure_transfer",
              description="FetalCLIP structure probes moved between echo datasets and from the 2nd to the 3rd trimester (exploratory)",
              script="scripts/run_wp16_structure_transfer.py",
              config={"restricted": True, "features": FEATURES, "probe": f"standardise, PCA({N_COMPONENTS}), logistic C=1 balanced",
                      "fit": "every case of the source", "exclusion": "target patients seen in the source are left out",
                      "made_after_seeing_results": True},
              inputs={"embeddings": list(SOURCES.values()), "matches": [MATCHES], "folds": [INDEX_2T]},
              n_real=sum(len(d["y"]) for d in data.values()), seed=SEED)

    matrix = {}
    for source in data:
        fitted = probe().fit(data[source]["x"], data[source]["y"])
        for target in data:
            if target == source:
                continue
            seen = seen_in(source, target, data, pairs)
            keep = np.array([p not in seen for p in data[target]["patients"]])
            scores = fitted.predict_proba(data[target]["x"][keep])[:, 1]
            value, low, high = auroc_ci(data[target]["y"][keep], scores, seed=SEED)
            matrix[f"{source}->{target}"] = {"auroc": value, "ci": [low, high], "n": int(keep.sum()),
                                             "n_chd": int(data[target]["y"][keep].sum()), "left_out_seen": int((~keep).sum())}
            print(f"{source:13s} -> {target:13s} AUROC {value:.3f} [{low:.3f}, {high:.3f}] n {int(keep.sum())} (left out {int((~keep).sum())})", flush=True)
    within = {name: load_run(run_name)[0]["probes"][FEATURES]["auroc"] for name, run_name in
              (("delta", "wp5_structure_delta"), ("heartbeat_2T", "wp5_structure_heartbeat_2T"),
               ("heartbeat_3T", "wp5_structure_heartbeat_3T"), ("cardium", "wp5_structure_cardium"))}

    # trimester shift with the authors' folds
    with INDEX_2T.open(encoding="utf-8", newline="") as handle:
        fold_of = {}
        for r in csv.DictReader(handle):
            fold_of[r["case"]] = {k: r[f"fold{k}"] for k in (1, 2, 3, 4)}
    hb2, hb3 = data["heartbeat_2T"], data["heartbeat_3T"]
    unseen = np.array([p not in set(hb2["patients"]) for p in hb3["patients"]])
    fetalclip = {"val_auroc": [], "unseen_3T_auroc": []}
    for k in (1, 2, 3, 4):
        train = np.array([fold_of.get(c, {}).get(k) == "train" for c in hb2["cases"]])
        val = np.array([fold_of.get(c, {}).get(k) == "val" for c in hb2["cases"]])
        fitted = probe().fit(hb2["x"][train], hb2["y"][train])
        fetalclip["val_auroc"].append(auroc(hb2["y"][val], fitted.predict_proba(hb2["x"][val])[:, 1]))
        fetalclip["unseen_3T_auroc"].append(auroc(hb3["y"][unseen], fitted.predict_proba(hb3["x"][unseen])[:, 1]))
    reference = load_run("wp15_reference_models")[0]["results"]
    trimester = {"fetalclip_probe": {"val_auroc_mean": float(np.mean(fetalclip["val_auroc"])),
                                     "unseen_3T_auroc_mean": float(np.mean(fetalclip["unseen_3T_auroc"])),
                                     "drop": float(np.mean(fetalclip["val_auroc"]) - np.mean(fetalclip["unseen_3T_auroc"])),
                                     "per_fold": fetalclip}}
    for arch in ("heart-vit", "vit", "resnet18", "resnet50", "vgg16", "mobilenet"):
        cv, shifted = reference.get(f"{arch}|2T|cross_val"), reference.get(f"{arch}|3T-unseen|cross_val")
        if cv and shifted:
            trimester[arch] = {"val_auroc_mean": cv["auroc"]["mean"], "unseen_3T_auroc_mean": shifted["auroc"]["mean"],
                               "drop": cv["auroc"]["mean"] - shifted["auroc"]["mean"]}
    run.finish({"transfer": matrix, "within_dataset_auroc": within, "trimester_shift": trimester,
                "unseen_3T_patients": int(unseen.sum()), "made_after_seeing_results": True})
    for name, entry in trimester.items():
        print(f"{name:16s} 2T val {entry['val_auroc_mean']:.3f} -> unseen 3T {entry['unseen_3T_auroc_mean']:.3f} (drop {entry['drop']:+.3f})")


if __name__ == "__main__":
    main()
