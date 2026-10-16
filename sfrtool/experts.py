"""Read one test of one patient and score it with its expert, exactly as the runs scored theirs.

Missing files raise an error (loaders fail loudly). A file that is there but cannot be used by
the expert - no fetal source, too few beats, no cardiac cycles, a missing echo plane - gives a
reading with ``readable = False`` and the reason. Nothing is guessed. Images are read into
memory only and never written anywhere.

The frozen settings are copied from the runs named next to them; tests/test_predict.py checks
on a real NInFEA recording that the tool reproduces the values those runs saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sfr.data.pwd import load_strip
from sfr.doppler.cycles import CycleParams, detect_cycles, prepare
from sfr.doppler.envelope import PIXELS_PER_SECOND, extract_envelopes
from sfr.doppler.measures import build_template, template_measures
from sfr.doppler.orientation import from_doppler, from_ecg
from sfr.doppler.reference import REFERENCE_MEASURES, nonconformity, z_scores
from sfr.ecg.extract import extract_fetal_ecg
from sfr.rhythm.features import RECORD_FEATURES, record_features, segment_features
from sfr.trust.crossmodal import TrustRule, trust_statistics, verdict

ECG_FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)  # runs wp3 and wp4a
SEGMENT_S, MIN_LAST_S = 60.0, 30.0              # rhythm segments of runs wp4a and wp6
CYCLE_FROZEN = CycleParams()                     # runs wp2, wp2b and wp3
RULE = TrustRule()                               # run wp3
ECG_LAYOUTS = ("ninfea", "nifeadb")
NINFEA_ABDOMINAL = tuple(f"uni_abd{i}" for i in range(1, 25))
NINFEA_THORACIC = tuple(f"bi_tho{i}" for i in range(1, 4))
PLANES = ("Abdomen", "4CH", "LVOT", "RVOT", "3VT")   # the five DELTA planes; the probe uses their mean embedding
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")


# ---- fetal ECG ---------------------------------------------------------------------------------------

@dataclass
class EcgReading:
    readable: bool
    reason: str
    layout: str
    duration_s: float
    extractor_ok: bool
    r_s: np.ndarray = field(repr=False)          # tracked fetal beats, seconds from the record start
    raw_s: np.ndarray = field(repr=False)        # threshold-only fetal detections
    features: dict = field(default_factory=dict)
    segments_used: int = 0


def _record_stem(path: Path) -> Path:
    path = Path(path)
    stem = path.with_suffix("") if path.suffix.lower() in (".hea", ".dat") else path
    for suffix in (".hea", ".dat"):
        if not stem.with_suffix(suffix).exists():
            raise FileNotFoundError(f"ECG file missing: {stem.with_suffix(suffix)}")
    return stem


def read_ecg(path: Path, layout: str = "ninfea") -> EcgReading:
    """Fetal beats and rhythm features of one WFDB record in the NInFEA or the NIFEADB channel layout."""
    import wfdb

    if layout not in ECG_LAYOUTS:
        raise ValueError(f"ECG layout must be one of {ECG_LAYOUTS}, not {layout!r}")
    record = wfdb.rdrecord(str(_record_stem(path)))
    names, fs = list(record.sig_name), float(record.fs)
    signals = record.p_signal.T.astype(np.float64)
    duration = signals.shape[1] / fs
    if layout == "ninfea":
        missing = [n for n in NINFEA_ABDOMINAL + NINFEA_THORACIC if n not in names]
        if missing:
            raise ValueError(f"not the NInFEA channel layout; missing channels {missing[:4]}")
        ecg = extract_fetal_ecg(signals[[names.index(n) for n in NINFEA_ABDOMINAL]], fs,
                                maternal_reference=signals[[names.index(n) for n in NINFEA_THORACIC]], **ECG_FROZEN)
        r_s, raw_s = ecg.fetal_peaks / ecg.fs, ecg.fetal_peaks_raw / ecg.fs
        edges = np.arange(0.0, duration, SEGMENT_S)                  # 60-second windows, as run wp6 cut NInFEA
        segments = []
        for start in edges:
            end = min(start + SEGMENT_S, duration)
            if end - start >= MIN_LAST_S or len(edges) == 1:
                segments.append(segment_features(raw_s[(raw_s >= start) & (raw_s < end)], r_s[(r_s >= start) & (r_s < end)]))
        features, extractor_ok = record_features(segments), bool(ecg.ok)
    else:
        if names[0] != "ECG" or not all(n.startswith("Abdomen_") for n in names[1:]):
            raise ValueError(f"not the NIFEADB channel layout: {names}")
        chest, abdominal = signals[0], signals[1:]
        step, n = int(round(SEGMENT_S * fs)), signals.shape[1]
        bounds = [(a, min(a + step, n)) for a in range(0, n, step) if min(a + step, n) - a >= MIN_LAST_S * fs]
        segments, tracked_all, raw_all, oks = [], [], [], []
        for a, b in bounds:                                           # one extraction per segment, as run wp4a
            result = extract_fetal_ecg(abdominal[:, a:b], fs, maternal_reference=chest[a:b], **ECG_FROZEN)
            tracked, raw = a / fs + result.fetal_peaks / result.fs, a / fs + result.fetal_peaks_raw / result.fs
            tracked_all.append(tracked)
            raw_all.append(raw)
            oks.append(bool(result.ok))
            if result.ok:                                             # run wp4b uses usable segments only
                segments.append(segment_features(raw, tracked))
        r_s = np.concatenate(tracked_all) if tracked_all else np.array([])
        raw_s = np.concatenate(raw_all) if raw_all else np.array([])
        features = record_features(segments) if segments else {name: float("nan") for name in RECORD_FEATURES}
        extractor_ok = any(oks)
    finite = all(np.isfinite(features[name]) for name in RECORD_FEATURES)
    readable = extractor_ok and finite
    reason = "" if readable else ("no acceptable fetal ECG source was found" if not extractor_ok else "too few fetal beats to describe the rhythm")
    return EcgReading(readable, reason, layout, duration, extractor_ok, r_s, raw_s, features, len(segments))


def rhythm_probability(features: dict, arrays: dict) -> float:
    """The logistic rhythm expert: median imputation, standardisation, logistic regression."""
    x = np.array([features[name] for name in RECORD_FEATURES], dtype=float)
    x = np.where(np.isfinite(x), x, arrays["imputer_median"])
    z = (x - arrays["scaler_mean"]) / arrays["scaler_scale"]
    return float(1.0 / (1.0 + np.exp(-(z @ arrays["coef"] + arrays["intercept"][0]))))


# ---- Doppler -------------------------------------------------------------------------------------------

@dataclass
class DopplerReading:
    readable: bool
    reason: str
    side: str
    duration_s: float
    n_cycles: int
    c_s: np.ndarray = field(repr=False)          # cycle starts (valve closure), seconds from the strip start
    measures: dict = field(default_factory=dict)


def read_doppler(path: Path, r_s: np.ndarray | None = None, ecg_ok: bool = False) -> DopplerReading:
    """Cardiac cycles and timing measures of one strip (284 columns per second, as in NInFEA).

    The ejection side comes from the ECG if the extractor succeeded, otherwise from the Doppler
    alone - the rule of run wp3.
    """
    envelopes = extract_envelopes(load_strip(Path(path)))
    prepared = prepare(envelopes, params=CYCLE_FROZEN)
    side_ecg = from_ecg(prepared, np.asarray(r_s) * PIXELS_PER_SECOND) if r_s is not None else {"side": None}
    side_doppler = from_doppler(prepared, params=CYCLE_FROZEN)
    side = (side_ecg["side"] if ecg_ok else None) or side_doppler["side"] or side_ecg["side"] or "lower"
    cycles = detect_cycles(envelopes, side, params=CYCLE_FROZEN, prepared=prepared)
    template = build_template(cycles, prepared)
    measures = template_measures(template)
    readable = template is not None and all(np.isfinite(measures.get(m, float("nan"))) for m in REFERENCE_MEASURES)
    reason = "" if readable else "too few clear cardiac cycles to measure the timing"
    return DopplerReading(readable, reason, side, envelopes.duration_s, int(cycles.n), cycles.starts / PIXELS_PER_SECOND, measures)


def function_score(measures: dict, gestation_days: float, arrays: dict, card: dict) -> tuple[float, dict]:
    """Largest standardised deviation from the healthy reference lines (sfr.doppler.reference)."""
    reference = {name: {"covariate": card["covariates"][name], "slope": float(arrays["slope"][k]), "intercept": float(arrays["intercept"][k]),
                        "sd": float(arrays["sd"][k])} for k, name in enumerate(card["measures"])}
    z = z_scores({**measures, "gestation_days": float(gestation_days)}, reference)
    return nonconformity(z), z


# ---- trust check ---------------------------------------------------------------------------------------

def trust_check(r_s: np.ndarray, c_s: np.ndarray) -> dict:
    stats = trust_statistics(r_s, c_s, RULE.tau_ms)
    decision = verdict(stats, RULE)
    keys = ("n_pairs", "locked_share", "drift_ms", "abs_diff_median_ms", "delay_ms", "rate_ecg_bpm", "rate_doppler_bpm")
    return {"trusted": bool(decision["trusted"]), "reasons": decision["reasons"], **{k: float(stats[k]) for k in keys}}


# ---- echo ----------------------------------------------------------------------------------------------

@dataclass
class EchoReading:
    readable: bool
    reason: str
    planes: list
    embedding: np.ndarray | None = field(default=None, repr=False)
    model: dict = field(default_factory=dict)


def find_planes(folder: Path) -> dict[str, Path]:
    found = {}
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in IMAGE_TYPES:
            for plane in PLANES:
                if path.stem.lower() == plane.lower():
                    found[plane] = path
    return found


def model_resizer(preprocess):
    """The resizing steps of the model's own preprocessing (as ``to_model_size`` in scripts/run_wp5_structure.py)."""
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

    return apply


def read_echo(folder: Path, device: str = "cpu") -> EchoReading:
    """Mean FetalCLIP embedding of the five cleaned planes of one case.

    Every image goes through ``sfr.echo.deidentify`` before anything else; only the cleaned
    scan, resized, reaches the network. Nothing is written to disk.
    """
    from PIL import Image

    from sfr.echo.deidentify import deidentify
    from sfr.echo.fetalclip import embed_images, load_fetalclip

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"echo input missing: {folder}")
    if not folder.is_dir():
        return EchoReading(False, "the structure expert needs a folder with one image of each plane "
                                  f"({', '.join(PLANES)}); a single image is outside its intended input", [])
    planes = find_planes(folder)
    missing = [plane for plane in PLANES if plane not in planes]
    if missing:
        return EchoReading(False, f"echo planes missing: {', '.join(missing)}", sorted(planes))
    cleaned = []
    for plane in PLANES:
        with Image.open(planes[plane]) as image:
            rgb = np.asarray(image.convert("RGB"))
        try:
            cleaned.append(deidentify(rgb).image)
        except ValueError:
            return EchoReading(False, f"no ultrasound scan region found in the {plane} image", list(PLANES))
    model, preprocess, record = load_fetalclip(device)
    resize = model_resizer(preprocess)
    embeddings = embed_images(model, preprocess, [resize(image) for image in cleaned], device)
    return EchoReading(True, "", list(PLANES), embeddings.mean(axis=0), record)


def structure_probability(embedding: np.ndarray, arrays: dict) -> float:
    """The linear probe of the structure expert: standardise, project on 32 components, logistic regression."""
    z = (np.asarray(embedding, dtype=float) - arrays["scaler_mean"]) / arrays["scaler_scale"]
    h = (z - arrays["pca_mean"]) @ arrays["pca_components"].T
    return float(1.0 / (1.0 + np.exp(-(h @ arrays["coef"] + arrays["intercept"][0]))))
