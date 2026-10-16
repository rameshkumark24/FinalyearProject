"""The one-patient research tool (sfrtool, scripts/predict_patient.py).

1. The output says exactly what was checked, for every subset of tests, and never reassures
   without a checked test; an alarm always refers; a missing test is never reported as checked.
2. Artifacts are refused when a file, the expert or the model does not match.
3. On real data the tool reproduces the runs: one NInFEA recording (fetal ECG and Doppler)
   against runs wp3, wp2b and wp6, and one DELTA case against run wp5. These tests are skipped,
   with the reason, when the data are not on this machine; there is no synthetic stand-in.

Random p-values in part 1 are synthetic and used only to draw decision cases.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401  (before NumPy work on this machine)
except ImportError:
    torch = None

import csv
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sfr.fusion.conformal import p_value
from sfr.fusion.decision import EXPERTS, DecisionConfig, ExpertResult, decide
from sfr.paths import NINFEA_ECG_DIR, NINFEA_PWD_DIR, RUNS
from sfrtool import ARTIFACTS, CONDITIONS, ROOT
from sfrtool.artifacts import ArtifactError, load_artifact, write_artifact
from sfrtool.labels import NOT_NORMAL, describe

CONFIG = DecisionConfig()
RNG = np.random.default_rng(20261016)
SUBSETS = [subset for k in (1, 2, 3) for subset in itertools.combinations(EXPERTS, k)]


def quiet_result() -> ExpertResult:
    return ExpertResult(present=True, p_value=float(RNG.uniform(0.5, 1.0)))


def label_of(results: dict, trust=None) -> tuple:
    decision = decide(results, trust, CONFIG)
    return decision, describe(decision, results, trust)


def checked_part(label: str) -> str:
    """The part of a label before 'NOT checked'."""
    return label.split("NOT checked")[0]


# ---- what the output says -----------------------------------------------------------------------------

@pytest.mark.parametrize("subset", SUBSETS)
def test_every_subset_of_tests_is_reported_as_checked_and_nothing_else(subset):
    for _ in range(50):
        results = {name: quiet_result() for name in subset}
        decision, text = label_of(results, True)
        assert decision.action in ("ROUTINE", "GET_TEST")
        assert text["covered"] == [CONDITIONS[n] for n in EXPERTS if n in subset]
        assert text["not_covered"] == [CONDITIONS[n] for n in EXPERTS if n not in subset]
        for name in EXPERTS:
            if name not in subset:
                assert CONDITIONS[name] not in checked_part(text["display_label"])
                assert text["not_covered_reasons"][CONDITIONS[name]] == "not done"
        assert NOT_NORMAL in text["display_label"]                    # "no alarm" is never a statement that the heart is normal


def test_routine_without_an_ecg_says_that_the_rhythm_was_not_checked():
    results = {"structure": quiet_result(), "function": quiet_result()}
    decision, text = label_of(results, None)
    assert decision.action == "ROUTINE"                                 # stored action name unchanged
    assert text["display_label"].startswith("No alarm on the tests done (structural heart defect (fetal echo), Doppler timing (pulsed-wave Doppler)).")
    assert "NOT checked: rhythm disturbance (fetal ECG: not done)." in text["display_label"]
    assert text["display_label"].endswith(NOT_NORMAL)


def test_no_usable_test_never_gives_reassuring_words():
    for results in ({}, {n: ExpertResult(present=True, quality_ok=False, p_value=0.9) for n in EXPERTS},
                    {"rhythm": ExpertResult(present=True, quality_ok=False)}):
        decision, text = label_of(results, None)
        assert decision.action != "ROUTINE"
        assert text["covered"] == []
        for phrase in ("No alarm", "normal", "Checked without alarm"):
            assert phrase not in text["display_label"]


def test_an_alarm_in_any_present_test_refers_whatever_the_others_show():
    for alarming in EXPERTS:
        for others in itertools.product(("missing", "unreadable", "quiet", "alarm"), repeat=2):
            results = {alarming: ExpertResult(present=True, p_value=float(RNG.uniform(0, CONFIG.alpha[alarming])))}
            for name, state in zip([n for n in EXPERTS if n != alarming], others):
                if state == "unreadable":
                    results[name] = ExpertResult(present=True, quality_ok=False)
                elif state == "quiet":
                    results[name] = quiet_result()
                elif state == "alarm":
                    results[name] = ExpertResult(present=True, p_value=float(RNG.uniform(0, CONFIG.alpha[name])))
            for trust in (True, False, None):
                decision, text = label_of(results, trust)
                assert decision.action == "REFER" and alarming in decision.alarms
                assert text["display_label"].startswith("Refer: alarm on") and CONDITIONS[alarming] in text["covered"]


def test_a_missing_or_unreadable_test_is_never_reported_as_checked():
    for _ in range(300):
        results, expected = {}, []
        for name in EXPERTS:
            state = RNG.choice(["missing", "unreadable", "quiet"])
            if state == "unreadable":
                results[name] = ExpertResult(present=True, quality_ok=False)
            elif state == "quiet":
                results[name] = quiet_result()
                expected.append(name)
        decision, text = label_of(results, True)
        assert text["covered"] == [CONDITIONS[n] for n in expected]
        for name in EXPERTS:
            if name not in expected:
                assert CONDITIONS[name] in text["not_covered"]
                assert CONDITIONS[name] not in checked_part(text["display_label"])


def test_failed_trust_check_moves_ecg_and_doppler_to_not_checked():
    results = {name: quiet_result() for name in EXPERTS}
    decision, text = label_of(results, False)
    assert decision.action == "REPEAT"
    assert text["covered"] == [CONDITIONS["structure"]]
    assert set(text["not_covered"]) == {CONDITIONS["function"], CONDITIONS["rhythm"]}
    assert "do not show the same heart" in text["display_label"]


# ---- artifacts ----------------------------------------------------------------------------------------

def test_artifact_round_trip_and_refusal_of_changed_files(tmp_path):
    folder = write_artifact("function", {"slope": [1.0, 2.0], "calibration": [0.1, 0.2, 0.3]}, {"measures": ["a", "b"]}, folder=tmp_path)
    arrays, card = load_artifact("function", tmp_path)
    assert np.allclose(arrays["slope"], [1.0, 2.0]) and card["measures"] == ["a", "b"]
    np.save(folder / "slope.npy", np.array([1.0, 2.5]))                      # one number changed
    with pytest.raises(ArtifactError, match="changed"):
        load_artifact("function", tmp_path)
    with pytest.raises(ArtifactError, match="belongs to expert"):
        shutil.copytree(folder, tmp_path / "rhythm")
        load_artifact("rhythm", tmp_path)


@pytest.mark.parametrize("expert", ["rhythm", "function", "structure"])
def test_shipped_artifacts_load_and_refuse_a_wrong_model_or_a_changed_file(expert, tmp_path):
    if not (ARTIFACTS / expert / "card.json").exists():
        pytest.skip(f"artifact {expert} not exported on this machine (python scripts/export_artifacts.py)")
    arrays, card = load_artifact(expert)
    assert card["verification"]
    if expert == "structure":
        with pytest.raises(ArtifactError, match="model sha256"):
            load_artifact(expert, model={"name": "FetalCLIP", "sha256": "0" * 64})
    shutil.copytree(ARTIFACTS / expert, tmp_path / expert)
    target = sorted((tmp_path / expert).glob("*.npy"))[0]
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 1                                                             # flip one bit of the last number
    target.write_bytes(bytes(blob))
    with pytest.raises(ArtifactError, match="changed"):
        load_artifact(expert, tmp_path)


def run_tool(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "predict_patient.py"), *arguments], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)


def test_tool_refuses_a_result_file_that_git_would_track():
    done = run_tool("--echo", "nowhere", "--out", str(ROOT / "result_of_a_patient.json"))
    assert done.returncode != 0 and "never be committed" in done.stderr + done.stdout
    assert not (ROOT / "result_of_a_patient.json").exists()


def test_tool_refuses_a_rhythm_calibration_too_small_for_its_alarm_level(tmp_path):
    if not (ARTIFACTS / "rhythm" / "card.json").exists():
        pytest.skip("rhythm artifact not exported on this machine")
    few = tmp_path / "healthy_scores.csv"
    few.write_text("score\n0.1\n0.2\n0.3\n0.4\n0.5\n", encoding="utf-8")          # 5 scores: smallest p 1/6 > 0.10
    done = run_tool("--ecg", str(tmp_path / "not_read"), "--calibration", str(few))
    assert done.returncode != 0 and "smallest possible p-value" in done.stderr + done.stdout


def test_tool_needs_at_least_one_test_and_a_gestation_for_the_doppler():
    assert run_tool().returncode != 0
    assert "--gestation" in run_tool("--doppler", "strip.bmp").stderr


# ---- reproduces the runs on real data ----------------------------------------------------------------------

RECORD = 1                       # a NInFEA recording whose ECG is readable and passes the trust check (run wp3)


def read_rows(path: Path, key: str) -> dict:
    with open(path, encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


@pytest.mark.skipif(not (NINFEA_ECG_DIR / f"{RECORD}.dat").exists() or not (NINFEA_PWD_DIR / f"{RECORD}.bmp").exists(),
                    reason="NInFEA is not on this machine (python scripts/download_ninfea.py)")
def test_tool_reproduces_the_runs_on_one_real_ninfea_recording():
    from sfrtool.experts import read_doppler, read_ecg, rhythm_probability, trust_check

    trust_row = read_rows(RUNS / "wp3_trust_check" / "per_recording.csv", "record")[str(RECORD)]
    function_row = read_rows(RUNS / "wp2b_function_reference" / "per_recording.csv", "record")[str(RECORD)]
    pair_row = read_rows(RUNS / "wp6_fusion" / "paired_cohort_ninfea.csv", "record")[str(RECORD)]
    ecg = read_ecg(NINFEA_ECG_DIR / str(RECORD), "ninfea")
    doppler = read_doppler(NINFEA_PWD_DIR / f"{RECORD}.bmp", r_s=ecg.r_s, ecg_ok=ecg.extractor_ok)
    trust = trust_check(ecg.r_s, doppler.c_s)
    assert ecg.extractor_ok == (trust_row["ecg_ok"] == "True") and ecg.readable
    assert doppler.side == trust_row["side_used"]
    assert trust["trusted"] == (trust_row["trusted"] == "True")
    assert trust["n_pairs"] == int(trust_row["n_pairs"])
    for key in ("locked_share", "drift_ms", "abs_diff_median_ms", "delay_ms"):
        assert trust[key] == pytest.approx(float(trust_row[key]), abs=1e-5)
    for key in ("cycle_length_ms", "a_to_v_peak_ms", "v_peak_to_closure_ms", "ejection_half_width_ms"):
        assert doppler.measures[key] == pytest.approx(float(function_row[key]), abs=1e-4)

    arrays, card = load_artifact("rhythm")
    score = rhythm_probability(ecg.features, arrays)
    assert score == pytest.approx(float(pair_row["rhythm_score"]), abs=1e-4)          # wp6 used beat times rounded to 0.1 ms
    assert p_value(score, arrays["calibration_published"]) == pytest.approx(float(pair_row["rhythm_p"]), abs=1e-6)

    results = {"rhythm": ExpertResult(present=True, p_value=p_value(score, arrays["calibration_published"])),
               "function": ExpertResult(present=True, p_value=float(function_row["p_value"]))}
    decision, text = label_of(results, trust["trusted"])
    assert decision.action == pair_row["action_echo_missing"]
    assert text["not_covered"] == [CONDITIONS["structure"]]


DELTA_CASE = Path("Test") / "00_Normal" / "Patient_0599"


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_tool_reproduces_the_structure_run_on_one_delta_case():
    from sfr.echo.fetalclip import FETALCLIP_FILE
    from sfrtool.experts import read_echo, structure_probability

    folder = ROOT / "data" / "delta" / "extracted" / "DELTA_Dataset" / DELTA_CASE
    embeddings = RUNS / "wp5_structure_delta" / "embeddings.npz"
    for needed, why in ((folder, "DELTA is not on this machine"), (FETALCLIP_FILE, "FetalCLIP weights are not on this machine"),
                        (embeddings, "the embeddings of run wp5_structure_delta are not on this machine"),
                        (ARTIFACTS / "structure" / "card.json", "the structure artifact is not exported")):
        if not needed.exists():
            pytest.skip(why)
    reading = read_echo(folder, "cpu")
    saved = np.load(embeddings, allow_pickle=False)
    stored = saved["fetalclip_scan"][[str(c) for c in saved["cases"]].index(DELTA_CASE.name)]
    assert reading.readable and reading.model["verified"]
    assert np.max(np.abs(reading.embedding - stored)) < 1e-4                    # run on a GPU, tool on a CPU
    arrays, _ = load_artifact("structure")
    assert structure_probability(reading.embedding, arrays) == pytest.approx(structure_probability(stored, arrays), abs=1e-4)
