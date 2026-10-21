"""The external echo script applies the frozen DELTA probe and never refits.

- Any request to refit stops the script before anything is read.
- Without an external dataset it stops with a message that says what is needed.
- Its scoring reproduces the artifact's own check on DELTA's Test split (from the embeddings saved
  by run wp5_structure_delta), so the code path that will meet external data is the published one.
- The script holds no fitting call at all.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_wp11_external_echo.py"


def load():
    spec = importlib.util.spec_from_file_location("run_wp11_external_echo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_wp11_external_echo"] = module
    spec.loader.exec_module(module)
    return module


external = load()


@pytest.mark.parametrize("argv", [["--refit"], ["--name", "x", "--fit=1"], ["--retrain"], ["--fine-tune"]])
def test_refit_requests_are_refused(argv):
    with pytest.raises(external.RefitRefused):
        external.main(argv)


def test_without_data_it_says_what_is_needed():
    with pytest.raises(SystemExit) as stop:
        external.main(["--name", "none", "--root", str(ROOT / "no_such_folder"), "--index", str(ROOT / "no_such_index.csv")])
    assert "needs data" in str(stop.value)


def test_the_script_has_no_fitting_code():
    text = SCRIPT.read_text(encoding="utf-8")
    assert ".fit(" not in text and "fit_transform" not in text and "partial_fit" not in text


def test_scoring_reproduces_the_artifact_check_on_delta_test():
    saved = ROOT / "runs" / "wp5_structure_delta" / "embeddings.npz"
    if not saved.exists() or not (ROOT / "artifacts" / "structure" / "card.json").exists():
        pytest.skip("run wp5_structure_delta or the structure artifact is missing")
    from sfr.eval.stats import auroc
    from sfrtool.artifacts import load_artifact

    arrays, card = load_artifact("structure")
    data = np.load(saved, allow_pickle=False)
    with open(ROOT / "runs" / "wp5_structure_delta" / "per_case.csv", encoding="utf-8", newline="") as handle:
        split = {r["case"]: r["split"] for r in csv.DictReader(handle)}
    test = np.array([split[str(c)] == "Test" for c in data["cases"]])
    labels = data["labels"][test].astype(int)
    scores = external.score_cases(data["fetalclip_scan"][test], arrays)
    p = external.p_values(scores, arrays["calibration"])
    check = card["verification"]
    assert auroc(labels, scores) == pytest.approx(check["test_auroc"], abs=1e-9)
    assert int((p[labels == 1] <= external.ALPHA).sum()) == check["test_defects_caught"]
    assert int((p[labels == 0] <= external.ALPHA).sum()) == check["test_normal_flagged"]
