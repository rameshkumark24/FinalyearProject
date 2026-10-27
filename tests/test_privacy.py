"""Nothing from a dataset whose terms forbid redistribution reaches the public repository.

Heartbeat and CARDIUM may be used for academic research but not passed on. Their images, their
index files (patient codes with labels) and every per-patient output stay in git-ignored places:
data/, and a private/ folder inside each run. A run made on such data marks itself
"restricted" in its config and may put only metrics.json and provenance.json in git. (Provenance
records only counts and one hash per set of input files, never their names.)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git is not available")
    return out.splitlines()


def ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def test_private_places_are_ignored_by_git():
    for path in ("data/indexes/echo_heartbeat_index.csv", "data/heartbeat/dataset/x.png", "data/cardium_clinical/x.json",
                 "runs/wp11_external_echo_x/private/per_case.csv", "runs/wp14_echo_overlap/private/matched_patients.csv"):
        assert ignored(path), f"{path} would be committed"


def test_no_tracked_file_is_in_a_private_place():
    bad = [p for p in tracked() if p.startswith("data/") or "/private/" in p]
    assert not bad, f"tracked files in private places: {bad[:5]}"


def test_restricted_runs_publish_only_their_summary():
    files = tracked()
    for provenance in sorted((ROOT / "runs").glob("*/provenance.json")):
        record = json.loads(provenance.read_text(encoding="utf-8"))
        if not record.get("config", {}).get("restricted"):
            continue
        folder = provenance.parent.name
        in_git = [p for p in files if p.startswith(f"runs/{folder}/")]
        assert set(in_git) <= {f"runs/{folder}/metrics.json", f"runs/{folder}/provenance.json"}, f"{folder}: {in_git}"
        for entry in record.get("data", {}).values():
            assert set(entry) <= {"n_files", "total_bytes", "manifest_sha256"}, f"{folder}: provenance lists more than counts and a hash"
