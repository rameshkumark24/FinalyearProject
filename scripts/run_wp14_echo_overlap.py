"""WP14 - do the echo datasets share images or patients?

Heartbeat and CARDIUM come from the same group and the same hospitals, so they may hold the same
fetuses; DELTA comes from another country. Before any model is run on the new data, every image of
DELTA, Heartbeat (2nd and 3rd trimester) and CARDIUM is compared with every image of the other
datasets. Nothing here uses a model or a label for the matching itself.

Rules, fixed before computing:
  exact match     identical decoded pixels (RGB, same width and height)
  near match      a 256-bit difference hash (grey image resized to 17 x 16, sign of the horizontal
                  gradient) at Hamming distance 10 or less
  same patient    at least one exact match between their images; "possible" if they share no exact
                  match but two or more near matches
Labels are compared only for patients found to be the same.

Only counts are written to the public files. The list of matched patients is written to
runs/wp14_echo_overlap/private/, which git ignores, because the access terms of Heartbeat and
CARDIUM do not let us redistribute patient-level information.

    python scripts/run_wp14_echo_overlap.py
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import DATA, SPLITS  # noqa: E402
from sfr.provenance import Run  # noqa: E402

NEAR_BITS = 10
SETS = {
    "delta": (SPLITS / "echo_delta_index.csv", DATA / "delta" / "extracted"),
    "heartbeat_2T": (DATA / "indexes" / "echo_heartbeat_2T_index.csv", DATA / "heartbeat" / "dataset"),
    "heartbeat_3T": (DATA / "indexes" / "echo_heartbeat_3T_index.csv", DATA / "heartbeat" / "dataset"),
    "cardium": (DATA / "indexes" / "echo_cardium_index.csv", DATA / "cardium_clinical" / "CARDIUM_dataset"),
}
BYTE_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprints(path: Path) -> tuple[str, np.ndarray]:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
        grey = np.asarray(rgb.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    digest = hashlib.sha256(np.ascontiguousarray(pixels).tobytes() + str(pixels.shape).encode()).hexdigest()
    bits = (grey[:, 1:] > grey[:, :-1]).astype(np.uint8).ravel()          # 16 x 16 = 256 bits
    return digest, np.packbits(bits)


def near_pairs(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs (i, j) with Hamming distance <= NEAR_BITS between packed 256-bit hashes."""
    found = []
    for i in range(a.shape[0]):
        distance = BYTE_BITS[np.bitwise_xor(b, a[i])].sum(axis=1)
        for j in np.flatnonzero(distance <= NEAR_BITS):
            found.append((i, int(j)))
    return found


def main() -> None:
    for name, (index, root) in SETS.items():
        if not index.exists():
            raise FileNotFoundError(f"{index} is missing; run the index script for {name} first")
    indexes = {name: read_index(index) for name, (index, _) in SETS.items()}
    run = Run("wp14_echo_overlap",
              description="Exact and near-duplicate images, and shared patients, between DELTA, Heartbeat and CARDIUM",
              script="scripts/run_wp14_echo_overlap.py",
              config={"exact": "identical decoded RGB pixels", "near": f"256-bit difference hash, Hamming distance <= {NEAR_BITS}",
                      "same_patient": "at least one exact image match; possible if no exact match but two or more near matches",
                      "restricted": True, "public_output": "counts only; matched patients in private/"},
              inputs={"indexes": [index for index, _ in SETS.values()]},
              n_real=sum(len(rows) for rows in indexes.values()), seed=None)
    started = time.time()
    data = {}
    for name, (index, root) in SETS.items():
        rows = indexes[name]
        digests, hashes = [], []
        for row in rows:
            digest, packed = fingerprints(root / row["path"])
            digests.append(digest)
            hashes.append(packed)
        data[name] = {"rows": rows, "digests": digests, "hashes": np.array(hashes, dtype=np.uint8)}
        print(f"{name}: {len(rows)} images fingerprinted ({time.time() - started:.0f} s)", flush=True)

    summary, private = {}, []
    for first, second in itertools.combinations(SETS, 2):
        a, b = data[first], data[second]
        where = {}
        for j, digest in enumerate(b["digests"]):
            where.setdefault(digest, []).append(j)
        exact = [(i, j) for i, digest in enumerate(a["digests"]) for j in where.get(digest, [])]
        near = near_pairs(a["hashes"], b["hashes"])
        patient_a = [r.get("patient") or r["case"] for r in a["rows"]]
        patient_b = [r.get("patient") or r["case"] for r in b["rows"]]
        exact_patients = {(patient_a[i], patient_b[j]) for i, j in exact}
        near_counts: dict[tuple[str, str], int] = {}
        for i, j in near:
            key = (patient_a[i], patient_b[j])
            near_counts[key] = near_counts.get(key, 0) + 1
        possible = {key for key, count in near_counts.items() if count >= 2} - exact_patients
        label_a = {(r.get("patient") or r["case"]): r["label"] for r in a["rows"]}
        label_b = {(r.get("patient") or r["case"]): r["label"] for r in b["rows"]}
        agree = sum(label_a[p] == label_b[q] for p, q in exact_patients)
        summary[f"{first}__{second}"] = {
            "images": [len(a["rows"]), len(b["rows"])],
            "exact_image_pairs": len(exact), "near_image_pairs": len(near),
            "patients_with_an_exact_match": [len({p for p, _ in exact_patients}), len({q for _, q in exact_patients})],
            "patient_pairs_exact": len(exact_patients), "patient_pairs_possible_near_only": len(possible),
            "labels_agree_in_exact_pairs": agree,
            "patients": [len(set(patient_a)), len(set(patient_b))],
        }
        for p, q in sorted(exact_patients):
            private.append({"first": first, "first_patient": p, "second": second, "second_patient": q, "evidence": "exact",
                            "labels": f"{label_a[p]}/{label_b[q]}"})
        for p, q in sorted(possible):
            private.append({"first": first, "first_patient": p, "second": second, "second_patient": q,
                            "evidence": f"near x{near_counts[(p, q)]}", "labels": f"{label_a[p]}/{label_b[q]}"})
        s = summary[f"{first}__{second}"]
        print(f"{first} vs {second}: exact image pairs {s['exact_image_pairs']}, near {s['near_image_pairs']}; patients with an exact "
              f"match {s['patients_with_an_exact_match']}; possible (near only) {s['patient_pairs_possible_near_only']}; labels agree "
              f"{agree}/{len(exact_patients)} ({time.time() - started:.0f} s)", flush=True)

    within = {}
    for name in SETS:
        digests = data[name]["digests"]
        within[name] = {"images": len(digests), "distinct_pixel_contents": len(set(digests))}
    (run.path / "private").mkdir(parents=True, exist_ok=True)
    run.write_csv("private/matched_patients.csv", private)
    run.finish({"between_datasets": summary, "within_datasets": within, "near_bits": NEAR_BITS,
                "note": "counts only; the matched patients are listed in private/matched_patients.csv (not in git)"})


if __name__ == "__main__":
    main()
