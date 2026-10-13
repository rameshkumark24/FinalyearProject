"""Write the index file of the DELTA dataset for scripts/run_wp5_structure.py.

DELTA: Zhang, Yao, Qian, Ye, "DELTA: A Dual-Modality Fetal Echocardiography Dataset for Long-Tail
Congenital Heart Disease Diagnosis", figshare, doi:10.6084/m9.figshare.32358771, CC BY 4.0.

The archive is not in this repository. Put ``DELTA_Dataset.zip`` in ``data/delta/``, check its MD5
(865d952adc0d8fe2be9df326e46cb763) and extract the still images and ``metadata.csv`` to
``data/delta/extracted/``. This script then turns the authors' ``metadata.csv`` (one row per case)
into one row per image: case, label, subtype, plane, path, split. Nothing is changed or filtered.

    python scripts/make_echo_index_delta.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import DATA, SPLITS  # noqa: E402

ROOT = DATA / "delta" / "extracted"
PLANES = ("Abdomen", "4CH", "LVOT", "RVOT", "3VT")
LABEL = {"normal": "0", "abnormal": "1"}


def main() -> None:
    source = ROOT / "DELTA_Dataset" / "metadata.csv"
    if not source.exists():
        raise FileNotFoundError(f"{source} is missing. Extract the DELTA archive first (see this file's header).")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    rows = []
    for case in cases:
        for plane in PLANES:
            path = case[f"{plane}_path"]
            if not (ROOT / path).exists():
                raise FileNotFoundError(f"image listed in metadata.csv is missing: {path}")
            rows.append({"case": case["case_id"], "label": LABEL[case["binary_label"]], "subtype": case["class_code"],
                         "plane": plane, "path": path, "split": case["split"]})
    target = SPLITS / "echo_delta_index.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    abnormal = sum(case["binary_label"] == "abnormal" for case in cases)
    print(f"{target}: {len(rows)} images of {len(cases)} cases ({abnormal} abnormal, {len(cases) - abnormal} normal)")


if __name__ == "__main__":
    main()
