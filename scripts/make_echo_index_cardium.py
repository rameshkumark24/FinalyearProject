"""Write the index files of the CARDIUM dataset (kept private: they list patients and labels).

CARDIUM: prenatal ultrasound images and clinical data with CHD labels, released by the BCV group
of Universidad de los Andes through a request form (github.com/BCV-Uniandes/Cardium; data CC BY-NC
4.0, academic and research use; arXiv:2510.15208). The data are not in this repository and must not
be: put the released folders in ``data/cardium_clinical/``. The index files go to ``data/indexes/``,
which git ignores.

One row per image of ``CARDIUM_dataset/`` (the full set, by label and patient): case, patient,
label, subtype, plane (CARDIUM does not name the view), path, split (empty), fold1..fold3 (train or
test in the authors' 3-fold cross-validation, read from ``cardium_images/``), and colour_share,
the share of pixels whose colour channels differ by more than 40 (the threshold of
sfr.echo.deidentify). An image with colour_share above 0.01 is counted as a colour (Doppler or
overlay) still; ``echo_cardium_grey_index.csv`` keeps only the other images. Nothing else is
filtered; the labels are checked against the authors' clinical table.

    python scripts/make_echo_index_cardium.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import DATA  # noqa: E402

BASE = DATA / "cardium_clinical"
ROOT = BASE / "CARDIUM_dataset"
TARGET = DATA / "indexes"
LABEL = {"CHD": "1", "Non_CHD": "0"}
COLOUR_SPREAD, COLOUR_SHARE = 40, 0.01
CLINICAL = BASE / "cardium_clinical_data_wnm_translated_final_cleaned.json"


def colour_share(path: Path) -> float:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return float(np.mean(spread > COLOUR_SPREAD))


def fold_membership() -> dict[str, dict[str, str]]:
    member: dict[str, dict[str, str]] = {}
    for k in (1, 2, 3):
        for part in ("train", "test"):
            for label in LABEL:
                folder = BASE / "cardium_images" / f"fold_{k}" / part / label
                for patient in sorted(p.name for p in folder.iterdir() if p.is_dir()):
                    member.setdefault(patient, {})[f"fold{k}"] = part
    return member


def write(name: str, rows: list[dict]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with (TARGET / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = {r["case"]: r["label"] for r in rows}
    print(f"{TARGET / name}: {len(rows)} images of {len(cases)} cases ({sum(v == '1' for v in cases.values())} with CHD)")


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(f"{ROOT} does not hold the released CARDIUM dataset")
    clinical = {r["id"]: str(r["CHD"]) for r in json.loads(CLINICAL.read_text(encoding="utf-8"))}
    folds = fold_membership()
    rows = []
    for label_name, label in LABEL.items():
        for patient in sorted(p.name for p in (ROOT / label_name).iterdir() if p.is_dir()):
            if clinical.get(patient) != label:
                raise ValueError(f"label of {patient} differs between the image folders and the clinical table")
            for image in sorted((ROOT / label_name / patient).iterdir()):
                share = colour_share(image)
                rows.append({"case": patient, "patient": patient, "label": label, "subtype": "", "plane": "",
                             "path": image.relative_to(ROOT).as_posix(), "split": "",
                             **{f"fold{k}": folds.get(patient, {}).get(f"fold{k}", "") for k in (1, 2, 3)},
                             "colour_share": round(share, 6), "colour_still": share > COLOUR_SHARE})
    write("echo_cardium_index.csv", rows)
    grey = [r for r in rows if not r["colour_still"]]
    write("echo_cardium_grey_index.csv", grey)
    missing_folds = sum(1 for r in {r["case"]: r for r in rows}.values() if not all(r[f"fold{k}"] for k in (1, 2, 3)))
    print(f"colour stills: {len(rows) - len(grey)} of {len(rows)} images; cases without a fold assignment: {missing_folds}")


if __name__ == "__main__":
    main()
