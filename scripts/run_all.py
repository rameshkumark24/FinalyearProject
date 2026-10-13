"""Run every analysis in order, on the current code, and print the gates.

    python scripts/run_all.py            # everything (about two hours on the development laptop)
    python scripts/run_all.py wp3 wp6    # only the named steps

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
]
# The three steps that need the DELTA echo dataset (see docs/data_cards/echo.md). They are skipped,
# with a note, when the extracted dataset or the structure run is not there.
DELTA_ROOT = ROOT / "data" / "delta" / "extracted"
ARGUMENTS = {
    "wp5s": ["--name", "delta", "--root", str(DELTA_ROOT), "--index", str(ROOT / "splits" / "echo_delta_index.csv"),
             "--source", "figshare doi:10.6084/m9.figshare.32358771 (version 2), CC BY 4.0"],
    "wp5b": ["--structure-run", "wp5_structure_delta"],
    "wp6b": ["--structure-run", "wp5_structure_delta"],
}
NEEDS = {"wp5s": DELTA_ROOT / "DELTA_Dataset" / "metadata.csv",
         "wp5b": ROOT / "runs" / "wp5_structure_delta" / "metrics.json",
         "wp6b": ROOT / "runs" / "wp5_structure_delta" / "metrics.json"}


def main() -> int:
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
        if entry.get("passed") is True:
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
