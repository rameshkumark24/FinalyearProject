"""Small files that hold what an expert needs to score one new case.

An artifact is a folder ``artifacts/<expert>/`` with plain NumPy arrays (``.npy``, read without
pickle) and a ``card.json``: dataset, number of healthy calibration cases, code hash, model
hash, date, intended input, known limits, and the SHA-256 of every array file. Loading checks
all of it and refuses on any mismatch. No raw data and no images are ever stored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import ARTIFACTS

FORMAT = "sfrtool-artifact-1"
EXPERTS = ("rhythm", "function", "structure")


class ArtifactError(ValueError):
    """An artifact is missing, altered or not the one the tool expects."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact(expert: str, arrays: Mapping[str, np.ndarray], card: Mapping[str, Any], folder: Path = ARTIFACTS) -> Path:
    """Write the arrays and the card. Existing files of this expert are replaced."""
    if expert not in EXPERTS:
        raise ArtifactError(f"unknown expert {expert!r}")
    target = Path(folder) / expert
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.npy"):
        if old.stem not in arrays:
            old.unlink()
    files = {}
    for name, value in arrays.items():
        path = target / f"{name}.npy"
        np.save(path, np.asarray(value, dtype=np.float64), allow_pickle=False)
        files[f"{name}.npy"] = _sha256(path)
    record = {"format": FORMAT, "expert": expert, **dict(card), "files": files}
    (target / "card.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def artifact_sha256(expert: str, folder: Path = ARTIFACTS) -> str:
    """One hash for the whole artifact: its card, which lists the hash of every array file."""
    return _sha256(Path(folder) / expert / "card.json")


def load_artifact(expert: str, folder: Path = ARTIFACTS, model: Mapping[str, str] | None = None) -> tuple[dict, dict]:
    """Arrays and card of one expert, after checking format, expert, file hashes and the model.

    ``model`` (name and sha256) must match the card when given; the structure expert passes the
    published FetalCLIP hash here.
    """
    target = Path(folder) / expert
    card_file = target / "card.json"
    if not card_file.exists():
        raise ArtifactError(f"{card_file} is missing. Run: python scripts/export_artifacts.py")
    card = json.loads(card_file.read_text(encoding="utf-8"))
    if card.get("format") != FORMAT:
        raise ArtifactError(f"{card_file}: format {card.get('format')!r}, expected {FORMAT!r}")
    if card.get("expert") != expert:
        raise ArtifactError(f"{card_file} belongs to expert {card.get('expert')!r}, not {expert!r}")
    if model is not None:
        stored = card.get("model") or {}
        for key in ("name", "sha256"):
            if stored.get(key) != model.get(key):
                raise ArtifactError(f"{card_file}: model {key} {stored.get(key)!r} does not match {model.get(key)!r}")
    listed = card.get("files") or {}
    present = {path.name for path in target.glob("*.npy")}
    if set(listed) != present:
        raise ArtifactError(f"{target}: files on disk {sorted(present)} differ from the card {sorted(listed)}")
    arrays = {}
    for name, digest in listed.items():
        path = target / name
        if _sha256(path) != digest:
            raise ArtifactError(f"{path} has been changed: its SHA-256 does not match the card")
        arrays[path.stem] = np.load(path, allow_pickle=False)
    return arrays, card
