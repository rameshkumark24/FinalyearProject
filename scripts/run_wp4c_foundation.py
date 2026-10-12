"""WP4c - a frozen adult ECG foundation model (ECGFounder, single lead) on fetal ECG.

Question (hypotheses H4 and H5 of the plan): do features of a model trained on adult ECG carry
fetal rhythm information, does a 2x time stretch help, and does anything leak through signals
that hold no fetal rhythm information?

The checkpoint is used only after its SHA-256 matches the published file. Every row that uses
it is paired with a row from the same network with untrained weights ("random-init").

Inputs (one-minute segments in which the extractor found a fetal source):
  fetal            extracted fetal ECG, resampled to the model's 500 Hz (10 s per window)
  fetal_stretch2   extracted fetal ECG presented at half speed (5 s per window), so that the
                   fetal rate and QRS width look like an adult's
  abdominal_raw    first abdominal channel, unprocessed: mostly the mother (negative control)
  maternal_chest   the mother's chest lead: no fetal signal at all (negative control)

Probes:
  arrhythmia vs normal on NIFEADB, leave-one-subject-out, PCA(8) + logistic regression
  fetal heart rate on CinC 2013 (75 recordings with reference beats), leave-one-recording-out
  maternal heart rate from the chest lead on NIFEADB (positive control: an adult ECG)

    python scripts/run_wp4c_foundation.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal, stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data import cinc2013, nifeadb  # noqa: E402
from sfr.ecg.extract import extract_fetal_ecg  # noqa: E402
from sfr.ecg.metrics import heart_rate_bpm  # noqa: E402
from sfr.eval.stats import auroc, bootstrap_ci  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.rhythm.evaluate import evaluate  # noqa: E402
from sfr.rhythm.foundation import MODEL_FS, WINDOW, embed, load_ecgfounder, prepare_windows, random_init  # noqa: E402

SOURCE_RUN = "wp4a_nifeadb_beats"
ECG_FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)
INPUTS = ("fetal", "fetal_stretch2", "abdominal_raw", "maternal_chest")
N_COMPONENTS, N_PERM, SEED = 8, 300, 0
NATIVE_500_HZ = ("ARR_06", "ARR_07", "ARR_08", "ARR_09")       # all four are arrhythmia recordings


def windows_of(x: np.ndarray) -> np.ndarray:
    n = x.size // WINDOW
    return x[: n * WINDOW].reshape(n, WINDOW) if n else np.zeros((0, WINDOW))


def to_model_rate(x: np.ndarray, fs: float) -> np.ndarray:
    if int(round(fs)) == MODEL_FS:
        return x
    ratio = int(round(fs)) // MODEL_FS
    if ratio * MODEL_FS != int(round(fs)):
        raise ValueError(f"Sampling rate {fs} is not a multiple of {MODEL_FS}")
    return signal.resample_poly(x, 1, ratio)


def upright(source: np.ndarray, beat_samples: np.ndarray) -> np.ndarray:
    """Independent components have an arbitrary sign; make the beats point upwards."""
    inside = beat_samples[(beat_samples >= 0) & (beat_samples < source.size)]
    if inside.size and np.median(source[inside]) < 0:
        return -source
    return source


def classifier():
    return make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=SEED),
                         LogisticRegression(C=1.0, max_iter=2000))


def regressor():
    return make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=SEED), Ridge(alpha=1.0))


def rate_probe(x: np.ndarray, target: np.ndarray) -> dict:
    """Leave-one-recording-out prediction of a heart rate from mean embeddings."""
    predicted, baseline = np.empty(target.size), np.empty(target.size)
    for i in range(target.size):
        train = np.arange(target.size) != i
        predicted[i] = clone(regressor()).fit(x[train], target[train]).predict(x[i:i + 1])[0]
        baseline[i] = target[train].mean()
    error, base_error = np.abs(predicted - target), np.abs(baseline - target)
    mae = bootstrap_ci(error, np.mean, seed=SEED)
    rho = stats.spearmanr(predicted, target)
    return {"n": int(target.size), "mae_bpm": mae[0], "mae_ci": [mae[1], mae[2]],
            "mae_of_mean_predictor_bpm": float(base_error.mean()),
            "r2": float(1.0 - np.sum((predicted - target) ** 2) / np.sum((baseline - target) ** 2)),
            "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "target_sd_bpm": float(target.std(ddof=1))}


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    founder, founder_record = load_ecgfounder(device)            # raises unless the file is the published one
    untrained, untrained_record = random_init(SEED, device)
    models = {"ecgfounder": founder, "random_init": untrained}

    load_run(SOURCE_RUN)
    source = RUNS / SOURCE_RUN
    beats = json.loads((source / "beats.json").read_text(encoding="utf-8"))
    sources = np.load(source / "fetal_sources.npz")

    run = Run(
        "wp4c_foundation",
        description="Frozen ECGFounder single-lead features on fetal ECG: arrhythmia probe with controls, and heart-rate probes",
        script="scripts/run_wp4c_foundation.py",
        config={"source_run": SOURCE_RUN, "inputs": INPUTS, "probe": f"standardise, PCA({N_COMPONENTS}), logistic regression C=1",
                "rate_probe": f"standardise, PCA({N_COMPONENTS}), ridge alpha=1", "permutations": N_PERM,
                "preprocessing": "official ECGFounder dataset.py: notch 50 Hz, band 0.67-40 Hz, median baseline removal, z-score",
                "ecg": ECG_FROZEN, "device": device, "random_init": untrained_record},
        inputs={"nifeadb": [p for name in nifeadb.ALL_RECORDS for p in nifeadb.record_files(name)],
                "cinc2013": [p for name in cinc2013.ALL_RECORDS for p in cinc2013.record_files(name)],
                "beats": [source / "beats.json"]},
        n_real=len(nifeadb.ALL_RECORDS) + len(cinc2013.ALL_RECORDS),
        models=[founder_record],
        seed=SEED,
    )

    started = time.time()
    # ---- NIFEADB: one mean embedding per recording, model and input -------------------------------
    embeddings = {m: {k: [] for k in INPUTS} for m in models}
    n_windows = {k: [] for k in INPUTS}
    maternal_rate = []
    for name in nifeadb.ALL_RECORDS:
        record = nifeadb.load_record(name)
        stack = {k: [] for k in INPUTS}
        for item in beats[name]:
            if not item["ok"]:
                continue
            k = item["segment"]
            fetal = sources[f"{name}__{k}"].astype(np.float64)                      # 1 kHz
            tracked = np.round((np.array(item["tracked_s"]) - item["start_s"]) * 1000).astype(int)
            fetal = upright(fetal, tracked)
            a = int(round(item["start_s"] * record.fs))
            b = a + int(round(fetal.size / 1000 * record.fs))
            stack["fetal"].append(windows_of(signal.resample_poly(fetal, 1, 2)))
            stack["fetal_stretch2"].append(windows_of(fetal))
            stack["abdominal_raw"].append(windows_of(to_model_rate(record.abdominal[0, a:b], record.fs)))
            stack["maternal_chest"].append(windows_of(to_model_rate(record.chest[a:b], record.fs)))
        for key in INPUTS:
            prepared = prepare_windows(np.concatenate(stack[key]))
            n_windows[key].append(int(prepared.shape[0]))
            for model_name, model in models.items():
                embeddings[model_name][key].append(embed(model, prepared, device).mean(axis=0))
        maternal = np.concatenate([np.array(item["maternal_s"]) for item in beats[name] if item["ok"]])
        maternal_rate.append(float(60.0 / np.median(np.diff(maternal)[np.diff(maternal) < 2.0])))
        print(f"{name} embedded ({time.time() - started:.0f} s)", flush=True)
    y = np.array([nifeadb.label_of(name) for name in nifeadb.ALL_RECORDS])
    maternal_rate = np.array(maternal_rate)
    embeddings = {m: {k: np.array(v) for k, v in d.items()} for m, d in embeddings.items()}

    arrhythmia: dict = {}
    oof_rows = [{"record": name, "label": int(label)} for name, label in zip(nifeadb.ALL_RECORDS, y)]
    for model_name in models:
        for key in INPUTS:
            result, scores = evaluate(embeddings[model_name][key], y, classifier(), N_PERM, SEED)
            result.update({"model": model_name, "input": key, "windows_per_recording_median": int(np.median(n_windows[key])),
                           "role": "negative control" if key in ("abdominal_raw", "maternal_chest") else "fetal input"})
            arrhythmia[f"{model_name}__{key}"] = result
            for row, value in zip(oof_rows, scores):
                row[f"oof_{model_name}__{key}"] = float(value)
            print(f"probe {model_name} {key}: AUROC {result['auroc']:.3f} ({time.time() - started:.0f} s)", flush=True)
    run.write_csv("per_record_scores.csv", oof_rows)

    maternal_probe = {m: rate_probe(embeddings[m]["maternal_chest"], maternal_rate) for m in models}

    # Added after the first run showed the maternal-lead control above chance: the four recordings
    # sampled at 500 Hz are all arrhythmia recordings. The same out-of-fold scores are summarised
    # again without them (no refitting). This is a check made after seeing the result.
    native = np.array([name in NATIVE_500_HZ for name in nifeadb.ALL_RECORDS])
    sampling_check = {}
    for key in arrhythmia:
        values = np.array([row[f"oof_{key}"] for row in oof_rows])
        sampling_check[key] = {
            "auroc_all": auroc(y, values), "auroc_without_500hz_recordings": auroc(y[~native], values[~native]),
            "n_without": int((~native).sum()),
            "mean_score_500hz_arrhythmia": float(values[native].mean()),
            "mean_score_1khz_arrhythmia": float(values[(y == 1) & ~native].mean()),
            "mean_score_normal": float(values[y == 0].mean())}

    # ---- CinC 2013: fetal heart rate from the embeddings ------------------------------------------
    cinc = {m: {"fetal": [], "fetal_stretch2": []} for m in models}
    reference_rate = []
    for name in cinc2013.ALL_RECORDS:
        x, fs, reference = cinc2013.load_record(name)
        extraction = extract_fetal_ecg(x, fs, **ECG_FROZEN)
        fetal = upright(extraction.fetal_signal.astype(np.float64), extraction.fetal_peaks)
        prepared = {"fetal": prepare_windows(windows_of(signal.resample_poly(fetal, 1, 2))),
                    "fetal_stretch2": prepare_windows(windows_of(fetal))}
        for model_name, model in models.items():
            for key, windows in prepared.items():
                cinc[model_name][key].append(embed(model, windows, device).mean(axis=0))
        reference_rate.append(heart_rate_bpm(reference, fs))
    reference_rate = np.array(reference_rate)
    fetal_probe = {f"{m}__{k}": rate_probe(np.array(v), reference_rate) for m, d in cinc.items() for k, v in d.items()}
    print(f"CinC probes done ({time.time() - started:.0f} s)", flush=True)

    np.savez_compressed(run.path / "record_embeddings.npz",
                        **{f"nifeadb__{m}__{k}": v for m, d in embeddings.items() for k, v in d.items()},
                        **{f"cinc__{m}__{k}": np.array(v) for m, d in cinc.items() for k, v in d.items()})
    metrics = {
        "checkpoint": founder_record,
        "arrhythmia_probe_nifeadb": arrhythmia,
        "fetal_heart_rate_probe_cinc2013": fetal_probe,
        "maternal_heart_rate_probe_nifeadb_positive_control": maternal_probe,
        "sampling_rate_check_added_after_first_run": sampling_check,
        "gate_G4": {
            "checkpoint_hash_matches_published_file": bool(founder_record["verified"]),
            "negative_control_rows_present": all(f"ecgfounder__{k}" in arrhythmia for k in ("abdominal_raw", "maternal_chest")),
            "every_row_has_n_and_interval": all("n" in r and "auroc_ci" in r for r in arrhythmia.values()),
        },
    }
    metrics["gate_G4"]["passed"] = all(metrics["gate_G4"].values())
    run.finish(metrics)

    print("\nArrhythmia vs normal, NIFEADB, leave-one-subject-out")
    for key, r in arrhythmia.items():
        print(f"  {key:32s} n {r['n']}  AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]  p {r['permutation_p']:.3f}  ({r['role']})")
    print("Fetal heart rate from embeddings, CinC 2013")
    for key, r in fetal_probe.items():
        print(f"  {key:32s} n {r['n']}  MAE {r['mae_bpm']:.2f} [{r['mae_ci'][0]:.2f}, {r['mae_ci'][1]:.2f}] bpm (mean predictor {r['mae_of_mean_predictor_bpm']:.2f})  R2 {r['r2']:.2f}  rho {r['spearman']:.2f}")
    print("Maternal heart rate from the chest lead (positive control)")
    for key, r in maternal_probe.items():
        print(f"  {key:32s} n {r['n']}  MAE {r['mae_bpm']:.2f} bpm (mean predictor {r['mae_of_mean_predictor_bpm']:.2f})  R2 {r['r2']:.2f}  rho {r['spearman']:.2f}")
    print("Sampling-rate check (added after the first run)")
    for key, r in sampling_check.items():
        print(f"  {key:32s} all {r['auroc_all']:.3f}  without the four 500 Hz recordings {r['auroc_without_500hz_recordings']:.3f} (n {r['n_without']})")
    print("Gate G4:", metrics["gate_G4"])


if __name__ == "__main__":
    main()
