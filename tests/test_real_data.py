"""Tests on real recordings with a known answer.

Each test runs a piece of the pipeline on a real file and checks a value that was established
in a recorded run. They are skipped, with the reason shown, when the data folder is not on this
machine; a skipped test certifies nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sfr.paths import CINC_DIR, MENDELEY_PWD_DIR, MODELS, NINFEA_ECG_DIR, NINFEA_PWD_DIR, SPLITS
from sfr.provenance import Run

needs_cinc = pytest.mark.skipif(not (CINC_DIR / "a05.hea").exists(), reason="CinC 2013 set A is not on this machine")
needs_marks = pytest.mark.skipif(not (MENDELEY_PWD_DIR / "annotations.mat").exists(), reason="annotated Doppler strips are not on this machine")
needs_ninfea = pytest.mark.skipif(not (NINFEA_ECG_DIR / "7.hea").exists() or not (NINFEA_PWD_DIR / "7.bmp").exists(),
                                  reason="NInFEA is not on this machine")
FROZEN = dict(n_components=3, max_sources=8, n_tracked=3, smoothness=4.0, min_score=0.20, seed=0)


# ---- loaders fail loudly ------------------------------------------------------------------------------

def test_missing_record_raises_instead_of_inventing_data(tmp_path):
    from sfr.data import cinc2013, nifeadb, ninfea
    from sfr.data.pwd import load_strip

    with pytest.raises(FileNotFoundError):
        cinc2013.load_record("a05", tmp_path)
    with pytest.raises(FileNotFoundError):
        ninfea.load_record(7, tmp_path)
    with pytest.raises(FileNotFoundError):
        nifeadb.load_record("ARR_01", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_strip(tmp_path / "nothing.bmp")


# ---- provenance refuses unlabelled synthetic data and unverified model names ------------------------------

def test_run_refuses_synthetic_cases_without_a_stated_kind(tmp_path, monkeypatch):
    monkeypatch.setattr("sfr.provenance.RUNS", tmp_path)
    with pytest.raises(ValueError):
        Run("x", description="d", script="s", config={}, inputs={}, n_real=1, n_synthetic=5)


def test_run_refuses_a_published_model_name_without_verified_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("sfr.provenance.RUNS", tmp_path)
    with pytest.raises(ValueError):
        Run("x", description="d", script="s", config={}, inputs={}, n_real=1,
            models=[{"claimed_name": "ECGFounder single-lead", "verified": False}])


def test_wrong_weight_file_is_not_accepted_as_ecgfounder(tmp_path):
    from sfr.rhythm.foundation import load_ecgfounder

    fake = tmp_path / "1_lead_ECGFounder.pth"
    fake.write_bytes(b"not the published file")
    with pytest.raises(ValueError):
        load_ecgfounder("cpu", fake)
    with pytest.raises(FileNotFoundError):
        load_ecgfounder("cpu", tmp_path / "absent.pth")


# ---- fetal ECG extraction -----------------------------------------------------------------------------

@needs_cinc
def test_fetal_beats_of_a_cinc_record_match_the_reference_and_are_not_the_mothers():
    from sfr.data.cinc2013 import load_record
    from sfr.ecg.extract import extract_fetal_ecg, maternal_phase_lock
    from sfr.ecg.metrics import beat_scores, follows_mother, heart_rate_bpm

    x, fs, reference = load_record("a05")
    result = extract_fetal_ecg(x, fs, **FROZEN)
    scores = beat_scores(result.fetal_peaks, reference, result.fs, 50.0)
    assert result.ok and scores["f1"] >= 0.97
    assert not follows_mother(heart_rate_bpm(result.fetal_peaks, result.fs), heart_rate_bpm(reference, fs),
                              result.quality["maternal_bpm"])
    assert maternal_phase_lock(result.fetal_peaks, result.maternal_peaks)[0] < 0.3
    # the mother's own beats, moved by a fixed delay, are recognised as locked to her
    shifted = result.maternal_peaks + int(0.1 * result.fs)
    assert maternal_phase_lock(shifted, result.maternal_peaks)[0] > 0.9


@needs_cinc
def test_a_single_abdominal_channel_without_the_pipeline_follows_the_mother():
    """The failure of the first build, kept as a test: without cancellation the detector locks to the mother.

    Record a13 is one of the development recordings where this happens (run wp1_cinc2013).
    """
    from sfr.data.cinc2013 import load_record
    from sfr.ecg.extract import extract_fetal_ecg
    from sfr.ecg.metrics import beat_scores, follows_mother, heart_rate_bpm

    x, fs, reference = load_record("a13")
    crude = extract_fetal_ecg(x, fs, **FROZEN, cancel=False, separate=False, track=False)
    full = extract_fetal_ecg(x, fs, **FROZEN)
    true_bpm = heart_rate_bpm(reference, fs)
    assert follows_mother(heart_rate_bpm(crude.fetal_peaks, crude.fs), true_bpm, crude.quality["maternal_bpm"])
    assert beat_scores(crude.fetal_peaks, reference, crude.fs)["f1"] < 0.3
    assert not follows_mother(heart_rate_bpm(full.fetal_peaks, full.fs), true_bpm, full.quality["maternal_bpm"])
    assert beat_scores(full.fetal_peaks, reference, full.fs)["f1"] > 0.97


# ---- Doppler cycles -----------------------------------------------------------------------------------

@needs_marks
def test_cycle_starts_of_an_annotated_strip_match_the_cardiologist_marks():
    from sfr.data.pwd import annotated_path, load_cycle_marks, load_strip
    from sfr.doppler.cycles import detect_cycles
    from sfr.doppler.envelope import extract_envelopes
    from sfr.doppler.marks import score_starts

    number = json.loads((SPLITS / "pwd_cycle_marks.json").read_text(encoding="utf-8"))["development"][3]   # strip 4
    envelopes = extract_envelopes(load_strip(annotated_path(number)))
    assert abs(envelopes.duration_s - envelopes.width / 284.0) < 1e-9
    cycles = detect_cycles(envelopes, "lower")
    scores = score_starts(cycles.starts, load_cycle_marks()[number], tolerance=15, offset=4.0)
    assert scores["f1"] >= 0.93 and scores["recall"] >= 0.93
    assert 120 < 60 * 284.0 / np.median(np.diff(cycles.starts)) < 180


@needs_marks
def test_annotated_strips_and_their_ninfea_copies_are_mapped():
    overlap = json.loads((SPLITS / "pwd_overlap.json").read_text(encoding="utf-8"))
    assert overlap["findings"] == {"identical": 13, "vertical_flip": 10, "same_orientation_extra_rows": 1, "not_in_ninfea": 19}
    split = json.loads((SPLITS / "pwd_cycle_marks.json").read_text(encoding="utf-8"))
    in_ninfea = {s["img"] for s in overlap["strips"] if s["ninfea_record"]}
    assert in_ninfea == set(split["test"]) and not in_ninfea & set(split["development"])


# ---- trust check ------------------------------------------------------------------------------------------

@needs_ninfea
def test_ecg_and_doppler_of_one_recording_agree_and_the_mothers_beats_do_not():
    from sfr.data.ninfea import load_record, strip_file
    from sfr.data.pwd import load_strip
    from sfr.doppler.cycles import detect_cycles, prepare
    from sfr.doppler.envelope import PIXELS_PER_SECOND, extract_envelopes
    from sfr.doppler.orientation import from_ecg
    from sfr.ecg.extract import extract_fetal_ecg
    from sfr.trust.crossmodal import trust_statistics, verdict

    record = load_record(7)                                   # a development recording
    ecg = extract_fetal_ecg(record.abdominal, record.fs, maternal_reference=record.thoracic, **FROZEN)
    envelopes = extract_envelopes(load_strip(strip_file(7)))
    prepared = prepare(envelopes)
    side = from_ecg(prepared, ecg.fetal_peaks / ecg.fs * PIXELS_PER_SECOND)["side"]
    assert side == "upper"                                    # known: this strip is stored upside down in the annotated set
    closures = detect_cycles(envelopes, side, prepared=prepared).starts / PIXELS_PER_SECOND
    genuine = trust_statistics(ecg.fetal_peaks / ecg.fs, closures)
    maternal = trust_statistics(ecg.maternal_peaks / ecg.fs, closures)
    assert verdict(genuine)["trusted"] and genuine["locked_share"] > 0.9 and abs(genuine["rate_ecg_bpm"] - genuine["rate_doppler_bpm"]) < 1.0
    assert not verdict(maternal)["trusted"] and maternal["locked_share"] < 0.1


def test_trust_check_flags_two_unrelated_regular_rhythms():
    from sfr.trust.crossmodal import trust_statistics, verdict

    a = np.arange(0.2, 30, 0.430)             # 139.5 bpm
    b = np.arange(0.4, 30, 0.400)             # 150 bpm
    same = trust_statistics(a, a + 0.2)
    other = trust_statistics(b, a + 0.2)
    assert verdict(same)["trusted"] and same["locked_share"] == 1.0
    assert not verdict(other)["trusted"] and "rates drift apart" in verdict(other)["reasons"]


# ---- models ------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (MODELS / "ecgfounder" / "1_lead_ECGFounder.pth").exists(), reason="ECGFounder weights are not on this machine")
def test_published_ecgfounder_weights_load_strictly_and_differ_from_random_init():
    from sfr.rhythm.foundation import embed, load_ecgfounder, prepare_windows, random_init

    model, record = load_ecgfounder("cpu")
    assert record["verified"] and record["n_parameters"] > 30_000_000
    windows = prepare_windows(np.sin(np.linspace(0, 60, 5000))[None, :] + np.random.default_rng(0).normal(0, 0.1, (2, 5000)))
    features = embed(model, windows)
    untrained, other = random_init(0)
    assert features.shape == (2, 1024) and np.isfinite(features).all()
    assert "claimed_name" not in other
    assert not np.allclose(features, embed(untrained, windows), atol=1e-3)
