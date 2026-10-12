"""Download published model weights and verify them against the hash published with them.

A model may be called by its published name only if this check passes (rule 2 of the project).
The files go to ``models_cache/`` and are never committed.

    python scripts/download_models.py ecgfounder
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import MODELS  # noqa: E402
from sfr.provenance import sha256_file  # noqa: E402

REGISTRY = {
    "ecgfounder": {
        "repo": "PKUDigitalHealth/ECGFounder",
        "revision": "d9b1793951b2342f5f7e84f1ac03cd37f8a08724",
        "file": "1_lead_ECGFounder.pth",
        "sha256": "f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603",
        "bytes": 369807481,
        "licence": "MIT",
        "folder": "ecgfounder",
    },
    "fetalclip": {
        "repo": "numansaeed/fetalclip-model",
        "revision": "2c02ad7b3eea314a328c5ebe5b81be4ef3a4d5db",
        "file": "FetalCLIP_weights.pt",
        "sha256": "90916caa408baf9374ee3f8c1533463331a4ace37a13e61385e782905b35c310",
        "bytes": 1710766661,
        "licence": "CC BY-NC 4.0",
        "folder": "fetalclip",
    },
}


def fetch(key: str) -> Path:
    from huggingface_hub import hf_hub_download

    entry = REGISTRY[key]
    folder = MODELS / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / entry["file"]
    if not target.exists():
        hf_hub_download(entry["repo"], entry["file"], revision=entry["revision"], local_dir=str(folder))
    digest = sha256_file(target, use_cache=False)
    if digest != entry["sha256"]:
        raise SystemExit(f"{target} does not match the published SHA-256 ({digest}). Delete it and download again.")
    print(f"{key}: {target} verified ({target.stat().st_size} bytes, sha256 {digest[:16]}...)")
    return target


if __name__ == "__main__":
    for name in sys.argv[1:] or list(REGISTRY):
        fetch(name)
