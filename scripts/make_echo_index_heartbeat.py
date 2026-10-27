"""Write the index files of the Heartbeat dataset (kept private: they list patients and labels).

Heartbeat: fetal echo images with CHD and normal labels from Colombian hospitals, released by the
BCV group of Universidad de los Andes through a request form, for academic and research use only
(github.com/BCV-Uniandes/Heartbeat; Front Cardiovasc Med 2026, doi:10.3389/fcvm.2026.1726484).
The data are not in this repository and must not be: put the released ``dataset`` folder in
``data/heartbeat/dataset/``. The index files go to ``data/indexes/``, which git ignores, because
they list patient codes with their labels, which the access terms do not let us redistribute.

One row per image: case, patient, trimester, label, subtype, plane, path, split, fold1..fold4 and
the four clinical fields of the authors' metadata. A case is one patient in one trimester (296
patients were scanned in both trimesters). ``split`` follows the authors' partition for the 2nd
trimester (fold 1: train = Train, val = Val; the test set = Test); the 3rd trimester has no test set.
Views are renamed to the names used for DELTA (4C -> 4CH). Nothing is filtered.

    python scripts/make_echo_index_heartbeat.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import DATA  # noqa: E402

ROOT = DATA / "heartbeat" / "dataset"
TARGET = DATA / "indexes"
PLANE = {"4C": "4CH", "LVOT": "LVOT", "RVOT": "RVOT", "3VT": "3VT"}
LABEL = {"CHD": "1", "No_CHD": "0"}
CLINICAL = ("maternal_age", "gestational_age_wk", "three_vessels", "growth_percentile")
SETS = (("2T", "cross_val"), ("2T", "test"), ("3T", "cross_val"))


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_of(trimester: str, part: str) -> list[dict]:
    folder = ROOT / trimester / part
    meta = read(folder / "metadata_raw.csv")
    folds = {}
    if part == "cross_val":
        for k in range(1, 5):
            for row in read(folder / f"fold{k}.csv"):
                folds.setdefault(row["image"], {})[f"fold{k}"] = row["split"]
    rows = []
    for row in meta:
        label = LABEL[row["class"]]
        path = Path(trimester) / part / row["image"]               # the authors' column holds class/patient/file
        if not (ROOT / path).exists():
            raise FileNotFoundError(f"image listed in the metadata is missing: {path}")
        fold = folds.get(row["image"], {})
        if part == "test":
            split = "Test"
        elif trimester == "2T":
            split = {"train": "Train", "val": "Val"}[fold["fold1"]]
        else:
            split = ""
        rows.append({"case": f"{row['patient_code']}_{trimester}", "patient": row["patient_code"], "trimester": trimester,
                     "label": label, "subtype": "", "plane": PLANE[row["view"]], "path": path.as_posix(), "split": split,
                     **{f"fold{k}": fold.get(f"fold{k}", "test" if part == "test" else "") for k in range(1, 5)},
                     **{name: row[name] for name in CLINICAL}})
    return rows


def write(name: str, rows: list[dict]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with (TARGET / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = {r["case"]: r["label"] for r in rows}
    print(f"{TARGET / name}: {len(rows)} images of {len(cases)} cases ({sum(v == '1' for v in cases.values())} with CHD)")


def main() -> None:
    if not (ROOT / "2T" / "cross_val" / "metadata_raw.csv").exists():
        raise FileNotFoundError(f"{ROOT} does not hold the released Heartbeat dataset folder")
    parts = {(t, p): rows_of(t, p) for t, p in SETS}
    everything = [row for rows in parts.values() for row in rows]
    write("echo_heartbeat_index.csv", everything)
    write("echo_heartbeat_2T_index.csv", parts[("2T", "cross_val")] + parts[("2T", "test")])
    write("echo_heartbeat_2T_test_index.csv", parts[("2T", "test")])
    write("echo_heartbeat_3T_index.csv", parts[("3T", "cross_val")])


if __name__ == "__main__":
    main()
