"""One-patient research tool built on the analysis package ``sfr``.

It scores one new case with the experts fitted in the runs, applies the decision rule of
``sfr.fusion.decision`` unchanged, and states which conditions were and were not checked.

The tool lives outside ``sfr`` so that the code hash recorded by every run still identifies the
code that produced it. Nothing here changes a reported result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

TOOL_VERSION = "0.1.0"
DISCLAIMER = "Research prototype. Not validated on patients with all three tests. Not for clinical use."
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# What each test looks for, in the plain words used in every output.
CONDITIONS = {"structure": "structural heart defect", "function": "Doppler timing", "rhythm": "rhythm disturbance"}
TESTS = {"structure": "fetal echo", "function": "pulsed-wave Doppler", "rhythm": "fetal ECG"}


def tool_code_sha256() -> str:
    """Hash of the tool's own code (this package and its command-line script)."""
    digest = hashlib.sha256()
    files = sorted((ROOT / "sfrtool").glob("*.py")) + [ROOT / "scripts" / "predict_patient.py"]
    for path in files:
        if path.exists():
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
