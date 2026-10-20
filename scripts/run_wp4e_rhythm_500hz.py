"""WP4e - the rhythm expert again, with every NIFEADB recording brought to 500 Hz first.

Why: four of the twelve arrhythmia recordings (ARR_06 to ARR_09) were sampled at 500 Hz and all
other recordings at 1 kHz, so the sampling rate alone can separate part of the classes. Run wp4c
found the mother's chest lead (no fetal signal) scoring above chance, and run wp9 found the
interval expert's AUROC on the 1 kHz recordings alone not distinguishable from heart rate.

Here the 1 kHz recordings are reduced to 500 Hz before anything else (scipy.signal.resample_poly
with factor 1/2 and its default anti-aliasing filter), so all 26 recordings pass through the same
rate. Everything after that is the frozen pipeline of runs wp4a, wp4b and wp4c: the same extractor
and settings, the same one-minute segments, the same nine interval features, the same logistic
model with leave-one-subject-out folds, the same verified ECGFounder checkpoint (and its
random-weight copy) with a refitted linear probe. Nothing is tuned; nothing is retrained except the
linear probes and the logistic model, as in the original runs.

This is a check made after seeing the results (requested in a review on 20 Oct 2026). The settings
and comparisons below are fixed in the run's config before anything is computed:
  - interval expert: AUROC with a bootstrap interval over subjects, permutation p (500 refits)
  - single indices of wp4b, nothing fitted (10000 permutations)
  - ECGFounder / random-init probes on the fetal inputs and the two negative controls (300 refits)
  - the same rows without the four natively 500 Hz recordings (no refitting)
  - paired AUROC differences, new minus original, over the same 26 subjects (2000 resamples),
    only where the original AUROC is reproduced exactly from its saved per-record scores
  - the four natively 500 Hz recordings get identical input, so their beats must equal run wp4a's

Not repeated here: the random forest, the leakage comparison, the CinC 2013 heart-rate probes and
the maternal heart-rate probe, which do not depend on the NIFEADB sampling rates.

    python -u scripts/run_wp4e_rhythm_500hz.py
"""

from __future__ import annotations

import torch  # noqa: F401  (first: importing it after SciPy work crashes on this machine)

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data import nifeadb  # noqa: E402
from sfr.ecg.extract import extract_fetal_ecg  # noqa: E402
from sfr.eval.stats import auroc, auroc_ci  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfr.rhythm.evaluate import evaluate, logistic, single_index  # noqa: E402
from sfr.rhythm.features import IRREGULARITY_FEATURES, QUALITY_FEATURES, RECORD_FEATURES, record_features, segment_features  # noqa: E402
from sfr.rhythm.foundation import MODEL_FS, WINDOW, embed, load_ecgfounder, prepare_windows, random_init  # noqa: E402

BEATS_RUN, INTERVALS_RUN, FOUNDATION_RUN = "wp4a_nifeadb_beats", "wp4b_rhythm_intervals", "wp4c_foundation"
ECG_FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)
SEGMENT_S, MIN_LAST_S = 60.0, 30.0
COMMON_FS = 500
NATIVE_500_HZ = ("ARR_06", "ARR_07", "ARR_08", "ARR_09")
INPUTS = ("fetal", "fetal_stretch2", "abdominal_raw", "maternal_chest")
N_PERM_LOGISTIC, N_PERM_INDEX, N_PERM_PROBE = 500, 10000, 300
N_COMPONENTS, N_BOOT, SEED = 8, 2000, 0
COMPARED = {"logistic_rhythm": (INTERVALS_RUN, "oof_logistic_rhythm"),
            "ecgfounder__fetal": (FOUNDATION_RUN, "oof_ecgfounder__fetal"),
            "random_init__fetal": (FOUNDATION_RUN, "oof_random_init__fetal"),
            "ecgfounder__maternal_chest": (FOUNDATION_RUN, "oof_ecgfounder__maternal_chest"),
            "ecgfounder__abdominal_raw": (FOUNDATION_RUN, "oof_ecgfounder__abdominal_raw")}


def to_common_rate(record: nifeadb.Recording) -> tuple[nifeadb.Recording, str]:
    fs = int(round(record.fs))
    if fs == COMMON_FS:
        return record, "unchanged (recorded at 500 Hz)"
    if fs != 2 * COMMON_FS:
        raise ValueError(f"{record.name}: unexpected sampling rate {record.fs}")
    chest = signal.resample_poly(record.chest, 1, 2)
    abdominal = signal.resample_poly(record.abdominal, 1, 2, axis=-1)
    return (nifeadb.Recording(record.name, record.label, float(COMMON_FS), chest, abdominal, chest.size / COMMON_FS),
            "resample_poly 1/2 from 1000 Hz")


def windows_of(x: np.ndarray) -> np.ndarray:
    n = x.size // WINDOW
    return x[: n * WINDOW].reshape(n, WINDOW) if n else np.zeros((0, WINDOW))


def upright(source: np.ndarray, beat_samples: np.ndarray) -> np.ndarray:
    """Independent components have an arbitrary sign; make the beats point upwards (as in wp4c)."""
    inside = beat_samples[(beat_samples >= 0) & (beat_samples < source.size)]
    if inside.size and np.median(source[inside]) < 0:
        return -source
    return source


def classifier():
    return make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=SEED),
                         LogisticRegression(C=1.0, max_iter=2000))


def paired_auroc_difference(labels, a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """AUROC(a) - AUROC(b) on the same subjects, with a percentile bootstrap interval over subjects."""
    labels, a, b = np.asarray(labels).astype(int), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, labels.size, labels.size)
        if labels[index].min() == labels[index].max():
            continue
        draws.append(auroc(labels[index], a[index]) - auroc(labels[index], b[index]))
    draws = np.array(draws)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": auroc(labels, a) - auroc(labels, b), "ci": [float(low), float(high)],
            "share_of_resamples_above_0": float(np.mean(draws > 0)), "n_resamples": int(draws.size),
            "can_be_ranked": bool(low > 0 or high < 0)}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    founder, founder_record = load_ecgfounder(device)            # raises unless the file is the published one
    untrained, untrained_record = random_init(SEED, device)
    models = {"ecgfounder": founder, "random_init": untrained}
    beats_metrics, _ = load_run(BEATS_RUN)
    intervals_metrics, _ = load_run(INTERVALS_RUN)
    foundation_metrics, _ = load_run(FOUNDATION_RUN)
    records = list(nifeadb.ALL_RECORDS)

    run = Run(
        "wp4e_rhythm_500hz",
        description="Rhythm expert and ECGFounder probes again with every NIFEADB recording brought to 500 Hz first "
                    "(check made after seeing the results)",
        script="scripts/run_wp4e_rhythm_500hz.py",
        config={"made_after_seeing_results": True, "requested_by": "review of 20 Oct 2026",
                "common_sampling_hz": COMMON_FS, "resampling": "scipy.signal.resample_poly(x, 1, 2), default Kaiser anti-aliasing filter",
                "ecg": ECG_FROZEN, "segment_s": SEGMENT_S, "min_last_segment_s": MIN_LAST_S,
                "interval_model": "wp4b logistic: median imputation, standardisation, L2 logistic regression C=1, threshold 0.5",
                "probe": f"wp4c: standardise, PCA({N_COMPONENTS}), logistic regression C=1", "inputs": INPUTS,
                "permutations": {"logistic": N_PERM_LOGISTIC, "single_index": N_PERM_INDEX, "probe": N_PERM_PROBE},
                "bootstrap": {"resamples": N_BOOT, "unit": "subject (recording)", "seed": SEED},
                "compared_with": {k: f"{run_name}:{column}" for k, (run_name, column) in COMPARED.items()},
                "rule_for_paired_differences": "only where the original AUROC is reproduced exactly from its saved per-record scores",
                "not_repeated": ["random forest", "leakage comparison", "CinC 2013 heart-rate probes", "maternal heart-rate probe"],
                "device": device, "random_init": untrained_record},
        inputs={"nifeadb": [p for name in records for p in nifeadb.record_files(name)],
                "original_scores": [RUNS / INTERVALS_RUN / "per_record_scores.csv", RUNS / FOUNDATION_RUN / "per_record_scores.csv",
                                    RUNS / BEATS_RUN / "beats.json", RUNS / FOUNDATION_RUN / "record_embeddings.npz"]},
        n_real=len(records),
        models=[founder_record],
        seed=SEED,
    )
    original_beats = json.loads((RUNS / BEATS_RUN / "beats.json").read_text(encoding="utf-8"))

    # ---- extraction at a common 500 Hz, then embeddings ---------------------------------------
    started = time.time()
    segment_rows, beats, segments = [], {}, []
    embeddings = {m: {k: [] for k in INPUTS} for m in models}
    resampling = {}
    for name in records:
        record, resampling[name] = to_common_rate(nifeadb.load_record(name))
        step = int(round(SEGMENT_S * record.fs))
        n_samples = record.abdominal.shape[1]
        bounds = [(a, min(a + step, n_samples)) for a in range(0, n_samples, step)
                  if min(a + step, n_samples) - a >= MIN_LAST_S * record.fs]
        beats[name] = []
        stack = {k: [] for k in INPUTS}
        for k, (a, b) in enumerate(bounds):
            result = extract_fetal_ecg(record.abdominal[:, a:b], record.fs, maternal_reference=record.chest[a:b], **ECG_FROZEN)
            offset = a / record.fs
            quality = result.quality
            item = {"segment": k, "start_s": offset, "ok": bool(result.ok),
                    "tracked_s": np.round(offset + result.fetal_peaks / result.fs, 4).tolist(),
                    "raw_s": np.round(offset + result.fetal_peaks_raw / result.fs, 4).tolist(),
                    "maternal_s": np.round(offset + result.maternal_peaks / result.fs, 4).tolist()}
            beats[name].append(item)
            segment_rows.append({"record": name, "label": record.label, "segment": k, "start_s": offset, "ok": bool(result.ok),
                                 "score": quality["score"], "template_corr": quality["template_corr"],
                                 "rate_bpm": quality["rate_bpm"], "n_tracked": int(result.fetal_peaks.size)})
            values = segment_features(np.array(item["raw_s"]), np.array(item["tracked_s"]))
            segments.append({"record": name, "label": record.label, "segment": k, "ok": bool(result.ok),
                             "score": float(quality["score"]), "template_corr": float(quality["template_corr"]), **values})
            if not result.ok:
                continue
            fetal = upright(result.fetal_signal.astype(np.float64), result.fetal_peaks)      # extractor output, 1 kHz
            n_model = int(round(fetal.size / result.fs * record.fs))
            stack["fetal"].append(windows_of(signal.resample_poly(fetal, 1, 2)))
            stack["fetal_stretch2"].append(windows_of(fetal))
            stack["abdominal_raw"].append(windows_of(record.abdominal[0, a:a + n_model]))
            stack["maternal_chest"].append(windows_of(record.chest[a:a + n_model]))
        for key in INPUTS:
            prepared = prepare_windows(np.concatenate(stack[key]), fs=MODEL_FS)
            for model_name, model in models.items():
                embeddings[model_name][key].append(embed(model, prepared, device).mean(axis=0))
        print(f"{name}: {len(bounds)} segments, {sum(i['ok'] for i in beats[name])} usable ({time.time() - started:.0f} s)", flush=True)
    run.write_csv("segments.csv", segment_rows)
    run.write_json("beats.json", beats)
    run.write_csv("segment_features.csv", segments)

    identical_native = {name: beats[name] == original_beats[name] for name in NATIVE_500_HZ}

    # ---- interval expert (wp4b settings) -------------------------------------------------------
    y = np.array([nifeadb.label_of(name) for name in records])
    rows = []
    for name in records:
        mine = [s for s in segments if s["record"] == name]
        used = [s for s in mine if s["ok"]]
        rows.append({"record": name, "label": nifeadb.label_of(name), "n_segments": len(mine), "n_used": len(used),
                     **record_features(used),
                     "usable_share": float(np.mean([s["ok"] for s in mine])),
                     "score_median": float(np.median([s["score"] for s in mine])),
                     "template_corr_median": float(np.median([s["template_corr"] for s in mine]))})
    x_rhythm = np.array([[r[f] for f in RECORD_FEATURES] for r in rows], dtype=float)
    run.write_csv("record_features.csv", rows)

    results, oof = {}, {}
    results["logistic_rhythm"], oof["logistic_rhythm"] = evaluate(x_rhythm, y, logistic(), N_PERM_LOGISTIC, SEED)
    print(f"interval expert done ({time.time() - started:.0f} s)", flush=True)
    single = {}
    for feature in IRREGULARITY_FEATURES:
        column = RECORD_FEATURES.index(f"{feature}_median")
        single[f"{feature}_median"] = single_index(y, x_rhythm[:, column], N_PERM_INDEX, SEED)
    single["hr_bpm_median"] = single_index(y, x_rhythm[:, RECORD_FEATURES.index("hr_bpm_median")], N_PERM_INDEX, SEED)
    single["hr_bpm_median"]["note"] = "no direction was pre-specified for heart rate; below 0.5 means arrhythmia recordings are slower"

    # ---- ECGFounder and random-init probes (wp4c settings) ---------------------------------------
    for model_name in models:
        for key in INPUTS:
            x = np.array(embeddings[model_name][key])
            result, scores = evaluate(x, y, classifier(), N_PERM_PROBE, SEED)
            result.update({"model": model_name, "input": key,
                           "role": "negative control" if key in ("abdominal_raw", "maternal_chest") else "fetal input"})
            results[f"{model_name}__{key}"], oof[f"{model_name}__{key}"] = result, scores
            print(f"probe {model_name} {key}: AUROC {result['auroc']:.3f} ({time.time() - started:.0f} s)", flush=True)
    np.savez_compressed(run.path / "record_embeddings.npz",
                        **{f"nifeadb__{m}__{k}": np.array(v) for m, d in embeddings.items() for k, v in d.items()})

    per_record = [{"record": name, "label": int(y[i]), "resampling": resampling[name],
                   **{f"oof_{k}": float(v[i]) for k, v in oof.items()},
                   **{f: rows[i][f] for f in RECORD_FEATURES + QUALITY_FEATURES}} for i, name in enumerate(records)]
    run.write_csv("per_record_scores.csv", per_record)

    # ---- without the four natively 500 Hz recordings (no refitting) -------------------------------
    native = np.array([name in NATIVE_500_HZ for name in records])
    without_native = {}
    for key, scores in oof.items():
        value, low, high = auroc_ci(y[~native], scores[~native], seed=SEED)
        without_native[key] = {"auroc": value, "auroc_ci": [low, high], "n": int((~native).sum()), "n_arrhythmia": int(y[~native].sum())}

    # The two control inputs already reached the network at 500 Hz in wp4c (its to_model_rate uses the same
    # resample_poly step), so their embeddings should be unchanged here; checked against wp4c's saved embeddings.
    saved = np.load(RUNS / FOUNDATION_RUN / "record_embeddings.npz")
    controls_unchanged = {f"{m}__{k}": bool(np.allclose(np.array(embeddings[m][k]), saved[f"nifeadb__{m}__{k}"], atol=1e-5))
                          for m in models for k in ("abdominal_raw", "maternal_chest")}

    # ---- paired differences against the original runs -----------------------------------------------
    published = {"logistic_rhythm": intervals_metrics["recording_level"]["logistic_rhythm"]["auroc"],
                 **{k: foundation_metrics["arrhythmia_probe_nifeadb"][k]["auroc"] for k in COMPARED if k != "logistic_rhythm"}}
    original_rows = {run_name: {r["record"]: r for r in read_csv(RUNS / run_name / "per_record_scores.csv")}
                     for run_name in (INTERVALS_RUN, FOUNDATION_RUN)}
    comparison = {}
    for key, (run_name, column) in COMPARED.items():
        before = np.array([float(original_rows[run_name][name][column]) for name in records])
        reproduced = bool(np.isclose(auroc(y, before), published[key]))
        entry = {"original_run": run_name, "original_auroc_published": published[key],
                 "original_auroc_from_saved_scores": auroc(y, before), "reproduced": reproduced,
                 "auroc_500hz": results[key]["auroc"], "auroc_500hz_ci": results[key]["auroc_ci"]}
        if reproduced:
            entry["paired_difference_500hz_minus_original"] = paired_auroc_difference(y, oof[key], before)
            entry["original_without_native_500hz"] = auroc(y[~native], before[~native])
        comparison[key] = entry

    metrics = {
        "dataset": {"n_records": int(y.size), "n_arrhythmia": int(y.sum()), "n_normal": int((1 - y).sum()),
                    "n_segments": len(segments), "n_usable_segments": int(sum(s["ok"] for s in segments)),
                    "n_usable_segments_original": int(sum(item["ok"] for name in records for item in original_beats[name])),
                    "resampled_to_500hz": int(sum(v.startswith("resample") for v in resampling.values()))},
        "native_500hz_recordings_beats_identical_to_wp4a": identical_native,
        "recording_level": results, "single_index": single,
        "without_native_500hz_recordings": without_native,
        "comparison_with_original": comparison,
        "control_inputs_unchanged_from_wp4c": controls_unchanged,
        "made_after_seeing_results": True,
        "reading": ("If the interval expert's AUROC holds after the common 500 Hz step, the sampling rate itself was not what it used. "
                    "The control inputs reached the network at 500 Hz already in wp4c, so a control above chance points to another "
                    "difference of the four natively 500 Hz recordings (all arrhythmia): their recording setting, which resampling "
                    "cannot remove. Only recordings of both classes from one setting can separate rhythm from setting."),
    }
    run.finish(metrics)

    print("\nRecording level, leave-one-subject-out, all recordings at 500 Hz")
    for key, r in results.items():
        w = without_native[key]
        print(f"  {key:30s} AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]  p {r['permutation_p']:.4f}"
              f"  without ARR_06-09 {w['auroc']:.3f} [{w['auroc_ci'][0]:.3f}, {w['auroc_ci'][1]:.3f}]")
    print("Single indices (nothing fitted)")
    for key, r in single.items():
        print(f"  {key:30s} AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]")
    print("Against the original runs")
    for key, c in comparison.items():
        diff = c.get("paired_difference_500hz_minus_original")
        text = f"{diff['difference']:+.3f} [{diff['ci'][0]:+.3f}, {diff['ci'][1]:+.3f}]" if diff else "not compared (original not reproduced)"
        print(f"  {key:30s} original {c['original_auroc_published']:.3f}  500 Hz {c['auroc_500hz']:.3f}  difference {text}")
    print("Native 500 Hz recordings unchanged:", identical_native)
    print("Control inputs unchanged from wp4c:", controls_unchanged)


if __name__ == "__main__":
    main()
