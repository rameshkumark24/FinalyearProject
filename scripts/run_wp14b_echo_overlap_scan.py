"""WP14b - shared fetuses between Heartbeat and CARDIUM, compared on the cleaned scan region.

Run wp14 found no shared image between Heartbeat and CARDIUM, but it compared whole frames, and the
two datasets store frames of different sizes (mostly 1136 x 787 against 1071 x 852): the same scan
cropped differently would not match. Here every image is first reduced to its ultrasound scan by
sfr.echo.deidentify, so the screen around it no longer matters. The same run counts repeated
images inside each dataset that belong to different patients (a leak across any split by patient).

Rules, fixed before computing:
  candidate     256-bit difference hash of the cleaned scan (grey, resized to 17 x 16) within
                Hamming distance 24
  match         a candidate whose 32 x 32 thumbnails of the cleaned scans correlate at 0.95 or more
  same patient  two or more matched images between two patients; one matched image is reported
                separately as "single"
  repeats       identical decoded pixels inside one dataset, counted when they belong to different
                patients, and when those patients have different labels

Counts only in the public files; the matched patients go to private/ (not in git).

    python scripts/run_wp14b_echo_overlap_scan.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import DATA  # noqa: E402
from sfr.provenance import Run  # noqa: E402

CANDIDATE_BITS, MIN_CORRELATION, MIN_IMAGES = 24, 0.95, 2
SETS = {"heartbeat": (DATA / "indexes" / "echo_heartbeat_index.csv", DATA / "heartbeat" / "dataset"),
        "cardium": (DATA / "indexes" / "echo_cardium_index.csv", DATA / "cardium_clinical" / "CARDIUM_dataset")}
BYTE_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
WORKERS = 5


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(path: str):
    """Pixel digest of the image as stored; difference hash and thumbnail of its cleaned scan (None if no scan)."""
    from PIL import Image

    from sfr.echo.deidentify import deidentify

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes() + str(rgb.shape).encode()).hexdigest()
    try:
        scan = deidentify(rgb).image
    except ValueError:
        return digest, None, None
    picture = Image.fromarray(scan.astype(np.uint8))
    grey = np.asarray(picture.resize((17, 16), Image.BILINEAR), dtype=np.int16)
    bits = np.packbits((grey[:, 1:] > grey[:, :-1]).astype(np.uint8).ravel())
    thumb = np.asarray(picture.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return digest, bits, thumb


def main() -> None:
    indexes = {name: read_index(index) for name, (index, _) in SETS.items()}
    run = Run("wp14b_echo_overlap_scan",
              description="Shared fetuses between Heartbeat and CARDIUM compared on the cleaned scan region, and repeated images across "
                          "patients inside each dataset",
              script="scripts/run_wp14b_echo_overlap_scan.py",
              config={"candidate": f"256-bit difference hash of the cleaned scan, Hamming distance <= {CANDIDATE_BITS}",
                      "match": f"thumbnail correlation >= {MIN_CORRELATION}", "same_patient": f">= {MIN_IMAGES} matched images",
                      "repeats": "identical decoded pixels belonging to different patients of one dataset", "restricted": True},
              inputs={"indexes": [index for index, _ in SETS.values()]},
              n_real=sum(len(rows) for rows in indexes.values()), seed=None)
    started = time.time()
    data = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for name, (index, root) in SETS.items():
            rows = indexes[name]
            results = list(pool.map(fingerprint, [str(root / r["path"]) for r in rows], chunksize=16))
            usable = [i for i, (_, bits, _) in enumerate(results) if bits is not None]
            data[name] = {"rows": rows, "digests": [d for d, _, _ in results], "usable": usable,
                          "bits": np.array([results[i][1] for i in usable], dtype=np.uint8),
                          "thumbs": np.array([results[i][2] for i in usable], dtype=np.float32)}
            print(f"{name}: {len(rows)} images, {len(usable)} with a scan region ({time.time() - started:.0f} s)", flush=True)

    repeats = {}
    for name, d in data.items():
        groups: dict[str, set[tuple[str, str]]] = {}
        for row, digest in zip(d["rows"], d["digests"]):
            groups.setdefault(digest, set()).add((row["patient"], row["label"]))
        across = [g for g in groups.values() if len({p for p, _ in g}) > 1]
        repeats[name] = {"images_repeated_across_patients": len(across),
                         "patients_involved": len({p for g in across for p, _ in g}),
                         "with_different_labels": sum(len({label for _, label in g}) > 1 for g in across)}

    a, b = data["heartbeat"], data["cardium"]
    candidates, matches = 0, []
    for i in range(a["bits"].shape[0]):
        distance = BYTE_BITS[np.bitwise_xor(b["bits"], a["bits"][i])].sum(axis=1)
        for j in np.flatnonzero(distance <= CANDIDATE_BITS):
            candidates += 1
            correlation = float(np.dot(a["thumbs"][i], b["thumbs"][j]) / a["thumbs"].shape[1])
            if correlation >= MIN_CORRELATION:
                matches.append((a["usable"][i], b["usable"][int(j)], correlation))
    pairs: dict[tuple[str, str], int] = {}
    for i, j, _ in matches:
        key = (a["rows"][i]["patient"], b["rows"][j]["patient"])
        pairs[key] = pairs.get(key, 0) + 1
    same = {key for key, count in pairs.items() if count >= MIN_IMAGES}
    single = {key for key, count in pairs.items() if count < MIN_IMAGES}
    label_a = {r["patient"]: r["label"] for r in a["rows"]}
    label_b = {r["patient"]: r["label"] for r in b["rows"]}
    trimesters = {}
    for p, _ in same:
        for r in a["rows"]:
            if r["patient"] == p:
                trimesters[r["trimester"]] = trimesters.get(r["trimester"], 0) + 1
                break
    (run.path / "private").mkdir(parents=True, exist_ok=True)
    run.write_csv("private/matched_patients.csv",
                  [{"first": "heartbeat", "first_patient": p, "second": "cardium", "second_patient": q, "matched_images": pairs[(p, q)],
                    "evidence": "same" if (p, q) in same else "single", "labels": f"{label_a[p]}/{label_b[q]}"}
                   for p, q in sorted(pairs)])
    metrics = {"usable_images": {name: len(d["usable"]) for name, d in data.items()},
               "candidate_image_pairs": candidates, "matched_image_pairs": len(matches),
               "patient_pairs_same": len(same), "patient_pairs_single_image": len(single),
               "heartbeat_patients_found_in_cardium": len({p for p, _ in same}),
               "cardium_patients_found_in_heartbeat": len({q for _, q in same}),
               "labels_agree_in_same_pairs": sum(label_a[p] == label_b[q] for p, q in same),
               "first_listed_trimester_of_matched_heartbeat_patients": trimesters,
               "repeats_inside_datasets": repeats,
               "note": "counts only; matched patients are in private/matched_patients.csv (not in git)"}
    run.finish(metrics)
    print(metrics)


if __name__ == "__main__":
    main()
