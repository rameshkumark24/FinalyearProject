"""Run folders and provenance.

Rule: no reported number without a run folder. A run folder holds the per-subject
outputs, a ``metrics.json`` and a ``provenance.json`` that records which data files,
code and model weights produced them, and how many cases were real or synthetic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .paths import DATA, ROOT, RUNS

_HASH_CACHE_FILE = DATA / ".sha256_cache.json"

REQUIRED_KEYS = (
    "run", "description", "created_utc", "script", "code_sha256", "git_commit",
    "config", "data", "n_real", "n_synthetic", "synthetic", "models",
)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path, use_cache: bool = True) -> str:
    """SHA-256 of a file. Results for data files are cached by path, size and mtime."""
    path = Path(path)
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    cache: dict[str, str] = {}
    if use_cache and _HASH_CACHE_FILE.exists():
        try:
            cache = json.loads(_HASH_CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
        if key in cache:
            return cache[key]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if use_cache:
        cache[key] = value
        _HASH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HASH_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return value


def manifest(paths: Iterable[Path]) -> dict[str, Any]:
    """One hash that identifies a set of input files."""
    entries = []
    total = 0
    for path in sorted(Path(p) for p in paths):
        if not path.exists():
            raise FileNotFoundError(f"Input file listed for provenance is missing: {path}")
        total += path.stat().st_size
        entries.append(f"{path.name}:{sha256_file(path)}")
    return {
        "n_files": len(entries),
        "total_bytes": total,
        "manifest_sha256": sha256_bytes("\n".join(entries).encode("utf-8")),
    }


def code_sha256() -> str:
    """Hash of every Python file in the package, so a run can be tied to exact code."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "sfr").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "sklearn", "torch", "wfdb", "PIL", "cv2"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


class Run:
    """A run folder under ``runs/<name>/``. Re-running the same script replaces it."""

    def __init__(
        self,
        name: str,
        *,
        description: str,
        script: str,
        config: Mapping[str, Any],
        inputs: Mapping[str, Sequence[Path]],
        n_real: int,
        n_synthetic: int = 0,
        synthetic_kind: str = "",
        models: Sequence[Mapping[str, Any]] = (),
        seed: int | None = None,
    ) -> None:
        if n_real < 0 or n_synthetic < 0:
            raise ValueError("Case counts cannot be negative")
        if n_synthetic > 0 and not synthetic_kind:
            raise ValueError("A run that uses synthetic cases must say what kind they are")
        for model in models:
            if model.get("claimed_name") and not model.get("verified"):
                raise ValueError(
                    f"Model '{model.get('claimed_name')}' was not verified against its published "
                    "hash. A model may carry a published name only if its weights match."
                )
        self.name = name
        self.path = RUNS / name
        self.path.mkdir(parents=True, exist_ok=True)
        self._provenance: dict[str, Any] = {
            "run": name,
            "description": description,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": script,
            "code_sha256": code_sha256(),
            "git_commit": git_commit(),
            "packages": _package_versions(),
            "platform": platform.platform(),
            "config": dict(config),
            "data": {key: manifest(paths) for key, paths in inputs.items()},
            "n_real": int(n_real),
            "n_synthetic": int(n_synthetic),
            "synthetic": bool(n_synthetic > 0),
            "synthetic_kind": synthetic_kind,
            "models": [dict(m) for m in models],
            "seed": seed,
            "argv": sys.argv,
        }

    def write_json(self, filename: str, obj: Any) -> Path:
        target = self.path / filename
        target.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        return target

    def write_csv(self, filename: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        target = self.path / filename
        if not rows:
            target.write_text("", encoding="utf-8")
            return target
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})
        return target

    def finish(self, metrics: Mapping[str, Any]) -> Path:
        self.write_json("metrics.json", dict(metrics))
        return self.write_json("provenance.json", self._provenance)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Cannot serialise {type(obj)!r}")


def _csv_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float):
        return round(value, 6)
    return value


def load_run(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metrics, provenance) of a finished run, or raise if either is missing."""
    path = RUNS / name
    metrics_file = path / "metrics.json"
    provenance_file = path / "provenance.json"
    if not metrics_file.exists() or not provenance_file.exists():
        raise FileNotFoundError(
            f"Run '{name}' has no metrics.json and provenance.json in {path}. "
            "Numbers may be reported only from finished runs."
        )
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"Run '{name}' provenance lacks: {missing}")
    return metrics, provenance
