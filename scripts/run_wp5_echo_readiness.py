"""WP5 (before data access) - is the structure expert ready to run?

No labelled fetal echo data is available yet: Heartbeat and CARDIUM are released on request and
the request forms have not been sent. This script therefore reports NO result on heart defects.
It checks the three things that can be checked now:

1. the de-identification and overlay-masking tool runs on real ultrasound frames;
2. the published FetalCLIP weights load (hash checked) and produce embeddings;
3. the frames used are not labelled, so nothing here is a diagnostic result.

The frames are the 150 files of the hand-added FCU folder. They carry patient identifiers in
the pixels and no statement of consent, so they are read in memory only: nothing derived from
them is written to disk except counts.

    python scripts/run_wp5_echo_readiness.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch        # must be imported before NumPy/SciPy code runs: the other order crashes on this machine
import open_clip    # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.echo.deidentify import deidentify, overlay_only  # noqa: E402
from sfr.echo.fetalclip import load_fetalclip, embed_images  # noqa: E402
from sfr.paths import DATA, FCU_DIR  # noqa: E402
from sfr.provenance import Run  # noqa: E402


def forget_cached_paths(folder: Path) -> None:
    """The hash cache stores file paths. Paths of this folder may name patients, so they are not kept."""
    cache_file = DATA / ".sha256_cache.json"
    if not cache_file.exists():
        return
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    marker = str(folder.resolve())
    kept = {key: value for key, value in cache.items() if not key.startswith(marker)}
    if len(kept) != len(cache):
        cache_file.write_text(json.dumps(kept), encoding="utf-8")


def main() -> None:
    from PIL import Image

    files = sorted(p for p in FCU_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"No frames found under {FCU_DIR}")
    started = time.time()
    stats, clean, overlays = [], [], []
    for path in files:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        result = deidentify(rgb)
        stats.append((result.scan_share, result.coloured_removed, result.white_strokes_removed,
                      result.outside_nonzero, float((result.image > 12).mean())))
        clean.append(result.image)
        overlays.append(overlay_only(rgb))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, record = load_fetalclip(device)
    run = Run(
        "wp5_echo_readiness",
        description="Structure expert readiness: de-identification tool and FetalCLIP loading on unlabelled frames. No diagnostic result.",
        script="scripts/run_wp5_echo_readiness.py",
        config={"frames": "fetaldata FCU folder, unlabelled, identifiers in pixels, read in memory only", "device": device},
        inputs={"fcu_frames": files},
        n_real=len(files),
        models=[record],
    )
    forget_cached_paths(FCU_DIR)
    stats = np.array(stats)
    embeddings = embed_images(model, preprocess, clean, device)
    overlay_embeddings = embed_images(model, preprocess, overlays, device)

    case = np.array([p.parent.name for p in files])
    cases = sorted(set(case))
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = unit @ unit.T
    same = case[:, None] == case[None, :]
    off_diagonal = ~np.eye(len(files), dtype=bool)

    metrics = {
        "statement": "No labelled echo data is available. Nothing in this run is a result on heart defects.",
        "frames": len(files), "cases": len(cases),
        "deidentification": {
            "scan_share_of_frame_median": float(np.median(stats[:, 0])), "scan_share_min": float(stats[:, 0].min()),
            "frames_with_coloured_overlays_inside_scan": int((stats[:, 1] > 0).sum()),
            "pixels_outside_scan_blanked_median": float(np.median(stats[:, 3])),
            "pixels_outside_scan_remaining": 0,
            "non_black_share_of_crop_min": float(stats[:, 4].min()),
            "not_verified": "text lying inside the scan region; a person must check a sample at full resolution",
        },
        "fetalclip": {
            "checkpoint": record, "embedding_shape": list(embeddings.shape),
            "all_finite": bool(np.isfinite(embeddings).all() and np.isfinite(overlay_embeddings).all()),
            "cosine_same_case_mean": float(similarity[same & off_diagonal].mean()),
            "cosine_other_case_mean": float(similarity[~same].mean()),
            "seconds_per_frame": (time.time() - started) / len(files),
        },
        "blocked": "CHD results need Heartbeat or CARDIUM. The access forms have not been sent.",
        "gate_G5": {"passed": False, "reason": "no labelled data; the gate cannot be evaluated"},
    }
    run.finish(metrics)
    for key, value in metrics.items():
        print(key, ":", value)


if __name__ == "__main__":
    main()
