"""Run every analysis in order, on the current code, and print the gates.

    python scripts/run_all.py            # everything (about four and a half hours on the development laptop)
    python scripts/run_all.py wp3 wp6    # only the named steps
    python scripts/run_all.py --list     # the steps and what each needs (data, weights, GPU)

Each step writes its run folder under ``runs/`` and a log under ``logs/``. The chain stops
at the first step that fails to run. A gate that is not met does not stop the chain: it is a
result and is printed at the end.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("wp1", "run_wp1_cinc2013.py", "wp1_cinc2013", "gate_G1"),
    ("wp2", "run_wp2_doppler_cycles.py", "wp2_doppler_cycles", "gate_G2"),
    ("wp3", "run_wp3_trust_check.py", "wp3_trust_check", "gate_G3"),
    ("wp2b", "run_wp2b_function_reference.py", "wp2b_function_reference", None),
    ("wp4a", "run_wp4a_nifeadb_beats.py", "wp4a_nifeadb_beats", None),
    ("wp4b", "run_wp4b_rhythm_intervals.py", "wp4b_rhythm_intervals", None),
    ("wp4c", "run_wp4c_foundation.py", "wp4c_foundation", "gate_G4"),
    ("wp4d", "run_wp4d_cnn.py", "wp4d_cnn", None),
    ("wp5s", "run_wp5_structure.py", "wp5_structure_delta", "gate_G5"),
    ("wp5b", "run_wp5b_structure_controls.py", "wp5b_structure_controls_delta", None),
    ("wp6", "run_wp6_fusion.py", "wp6_fusion", "gate_G6"),
    ("wp6b", "run_wp6b_fusion_structure.py", "wp6b_fusion_structure", None),
    # Checks added after seeing the results (16 October 2026); they read saved outputs only.
    ("wp7", "run_wp7_dependence_bounds.py", "wp7_dependence_bounds", "gate_G7"),
    ("wp8", "run_wp8_tradeoff.py", "wp8_tradeoff", None),
    ("wp9", "run_wp9_checks_after_results.py", "wp9_checks_after_results", None),
    # Checks requested in the review of 20 October 2026, also made after seeing the results.
    ("wp4e", "run_wp4e_rhythm_500hz.py", "wp4e_rhythm_500hz", None),
    ("wp5c", "run_wp5c_structure_by_size.py", "wp5c_structure_by_size", None),
    ("wp3b", "run_wp3b_trust_baselines.py", "wp3b_trust_baselines", None),
    ("wp13", "run_wp13_calibration_transfer.py", "wp13_calibration_transfer", None),
    ("wp12", "run_wp12_theory_simulation.py", "wp12_theory_simulation", None),
    ("wp10", "run_wp10_allocation.py", "wp10_allocation", None),
    # External echo data (Heartbeat, CARDIUM; received 24 and 27 October 2026). Restricted: per-patient
    # files stay in runs/*/private/. The wp11 steps are the preregistered test; the rest is exploratory.
    ("ixhb", "make_echo_index_heartbeat.py", "", None),
    ("ixcd", "make_echo_index_cardium.py", "", None),
    ("wp14", "run_wp14_echo_overlap.py", "wp14_echo_overlap", None),
    ("wp14b", "run_wp14b_echo_overlap_scan.py", "wp14b_echo_overlap_scan", None),
    ("x2T", "run_wp11_external_echo.py", "wp11_external_echo_heartbeat_2T", None),
    ("x2Tt", "run_wp11_external_echo.py", "wp11_external_echo_heartbeat_2T_test", None),
    ("x3T", "run_wp11_external_echo.py", "wp11_external_echo_heartbeat_3T", None),
    ("xcd", "run_wp11_external_echo.py", "wp11_external_echo_cardium", None),
    ("xcdg", "run_wp11_external_echo.py", "wp11_external_echo_cardium_grey", None),
    ("wp17", "run_wp17_structure_calibration.py", "wp17_structure_calibration", None),
    ("wp18", "run_wp18_echo_retest.py", "wp18_echo_retest", None),
    ("wp15", "run_wp15_reference_models.py", "wp15_reference_models", None),
    ("s2T", "run_wp5_structure.py", "wp5_structure_heartbeat_2T", None),
    ("s3T", "run_wp5_structure.py", "wp5_structure_heartbeat_3T", None),
    ("scd", "run_wp5_structure.py", "wp5_structure_cardium", None),
    ("wp16", "run_wp16_structure_transfer.py", "wp16_structure_transfer", None),
    # Artifacts of the one-patient research tool, each refit checked against its run.
    ("tool", "export_artifacts.py", "tool_artifacts", None),
]
# The steps that need the DELTA echo dataset (see docs/data_cards/echo.md) or its structure run. They
# are skipped, with a note, when the extracted dataset or the structure run is not there.
DELTA_ROOT = ROOT / "data" / "delta" / "extracted"
ARGUMENTS = {
    "wp5s": ["--name", "delta", "--root", str(DELTA_ROOT), "--index", str(ROOT / "splits" / "echo_delta_index.csv"),
             "--source", "figshare doi:10.6084/m9.figshare.32358771 (version 2), CC BY 4.0"],
    "wp5b": ["--structure-run", "wp5_structure_delta"],
    "wp6b": ["--structure-run", "wp5_structure_delta"],
}
HEARTBEAT = ROOT / "data" / "heartbeat" / "dataset"
CARDIUM = ROOT / "data" / "cardium_clinical" / "CARDIUM_dataset"
INDEXES = ROOT / "data" / "indexes"
SOURCE_HB = "Heartbeat (BCV-Uniandes), Front Cardiovasc Med 2026, doi:10.3389/fcvm.2026.1726484; academic and research use only"
SOURCE_CD = "CARDIUM (BCV-Uniandes), arXiv:2510.15208; data CC BY-NC 4.0, academic and research use"
for key, name, root, index, source in (("x2T", "heartbeat_2T", HEARTBEAT, "echo_heartbeat_2T_index.csv", SOURCE_HB),
                                       ("x2Tt", "heartbeat_2T_test", HEARTBEAT, "echo_heartbeat_2T_test_index.csv", SOURCE_HB),
                                       ("x3T", "heartbeat_3T", HEARTBEAT, "echo_heartbeat_3T_index.csv", SOURCE_HB),
                                       ("xcd", "cardium", CARDIUM, "echo_cardium_index.csv", SOURCE_CD),
                                       ("xcdg", "cardium_grey", CARDIUM, "echo_cardium_grey_index.csv", SOURCE_CD)):
    ARGUMENTS[key] = ["--name", name, "--root", str(root), "--index", str(INDEXES / index), "--source", source]
for key, name, root, index, source in (("s2T", "heartbeat_2T", HEARTBEAT, "echo_heartbeat_2T_index.csv", SOURCE_HB),
                                       ("s3T", "heartbeat_3T", HEARTBEAT, "echo_heartbeat_3T_index.csv", SOURCE_HB),
                                       ("scd", "cardium", CARDIUM, "echo_cardium_index.csv", SOURCE_CD)):
    ARGUMENTS[key] = ["--name", name, "--root", str(root), "--index", str(INDEXES / index), "--source", source, "--restricted"]
NEEDS = {"wp5s": DELTA_ROOT / "DELTA_Dataset" / "metadata.csv",
         "wp5b": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "wp6b": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "wp7": ROOT / "runs" / "wp6b_fusion_structure" / "metrics.json",
         "wp8": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "wp9": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "wp5c": ROOT / "runs" / "wp5_structure_delta" / "embeddings.npz",
         "wp10": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "ixhb": HEARTBEAT / "2T" / "cross_val" / "metadata_raw.csv", "ixcd": CARDIUM,
         "wp14": INDEXES / "echo_cardium_index.csv", "wp14b": INDEXES / "echo_cardium_index.csv",
         "x2T": INDEXES / "echo_heartbeat_2T_index.csv", "x2Tt": INDEXES / "echo_heartbeat_2T_test_index.csv",
         "x3T": INDEXES / "echo_heartbeat_3T_index.csv", "xcd": INDEXES / "echo_cardium_index.csv",
         "xcdg": INDEXES / "echo_cardium_grey_index.csv",
         "wp17": ROOT / "runs" / "wp11_external_echo_cardium" / "metrics.json",
         "wp18": ROOT / "runs" / "wp11_external_echo_heartbeat_3T" / "metrics.json",
         "wp15": ROOT / "data" / "third_party" / "Heartbeat" / "run_eval.py",
         "s2T": INDEXES / "echo_heartbeat_2T_index.csv", "s3T": INDEXES / "echo_heartbeat_3T_index.csv",
         "scd": INDEXES / "echo_cardium_index.csv",
         "wp16": ROOT / "runs" / "wp5_structure_cardium" / "metrics.json"}
# What each step needs besides the CPU and the public data under data/ (python scripts/run_all.py --list).
RESOURCES = {"wp4c": "ECGFounder weights; a GPU is advised", "wp4d": "a GPU is advised",
             "wp5s": "the DELTA dataset and the FetalCLIP weights; a GPU is advised (about half an hour)",
             "wp4e": "ECGFounder weights; a GPU is advised (about 17 minutes on the development laptop)",
             "wp5b": "run wp5s", "wp6b": "run wp5s", "wp7": "run wp6b", "wp8": "run wp5s", "wp9": "run wp5s",
             "wp5c": "the embeddings saved by run wp5s", "wp10": "run wp5s; uses run wp4e if present",
             "tool": "runs wp2b, wp4b and wp5s; the FetalCLIP weights",
             "ixhb": "the Heartbeat dataset (request form; academic use)", "ixcd": "the CARDIUM dataset (request form; CC BY-NC 4.0)",
             "wp14": "both index files and DELTA", "wp14b": "both index files",
             "x2T": "Heartbeat, FetalCLIP weights, structure artifact; a GPU is advised",
             "x2Tt": "Heartbeat, FetalCLIP weights, structure artifact", "x3T": "Heartbeat, FetalCLIP weights, structure artifact; a GPU is advised",
             "xcd": "CARDIUM, FetalCLIP weights, structure artifact; a GPU is advised", "xcdg": "CARDIUM, FetalCLIP weights, structure artifact",
             "wp17": "the private outputs of x2T, x3T and xcd", "wp18": "the private outputs of x2T and x3T",
             "wp15": "the authors' code in data/third_party/Heartbeat and their checkpoints; a GPU is advised",
             "s2T": "Heartbeat and FetalCLIP; a GPU is advised", "s3T": "Heartbeat and FetalCLIP; a GPU is advised",
             "scd": "CARDIUM and FetalCLIP; a GPU is advised", "wp16": "runs wp5s, s2T, s3T, scd, wp14b and wp15"}


def main() -> int:
    if "--list" in sys.argv[1:]:
        for key, script, run, _ in STEPS:
            print(f"{key:5s} {script:36s} {RESOURCES.get(key, 'CPU and the public data')}")
        return 0
    wanted = set(sys.argv[1:])
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for key, script, _, _ in STEPS:
        if wanted and key not in wanted:
            continue
        if key in NEEDS and not NEEDS[key].exists():
            print(f"[{time.strftime('%H:%M:%S')}] {key}: skipped, {NEEDS[key].relative_to(ROOT)} is not there", flush=True)
            continue
        started = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {key}: {script}", flush=True)
        with open(logs / f"{key}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *ARGUMENTS.get(key, [])], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"           finished in {time.time() - started:.0f} s, exit code {code}", flush=True)
        if code != 0:
            print(f"Stopped: {key} did not run. See logs/{key}.log")
            return code

    print("\nGates")
    for key, _, run, gate in STEPS:
        metrics_file = ROOT / "runs" / run / "metrics.json"
        if gate is None or not metrics_file.exists():
            continue
        entry = json.loads(metrics_file.read_text(encoding="utf-8")).get(gate, {})
        if "passed_at_measured_rates" in entry:
            state = (("MET" if entry["passed_at_measured_rates"] else "NOT MET") + " at the measured rates, "
                     + ("MET" if entry["passed_with_uncertainty"] else "NOT MET") + " with uncertainty (defined after seeing the results)")
        elif entry.get("passed") is True:
            state = "MET"
        elif entry.get("auroc_with_interval"):
            state = "PARTLY MET (" + entry["reason"] + ")"
        elif "reason" in entry:
            state = "CANNOT BE EVALUATED (" + entry["reason"] + ")"
        elif entry.get("passed") is False:
            state = "NOT MET"
        else:
            state = "decided by the property tests: python -m pytest tests/test_fusion.py"
        print(f"  {gate[5:]}  {run:26s} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
