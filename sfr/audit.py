"""Checks that make invented results hard to produce by accident.

    python -m sfr.audit

Exit code 0 only if every check passes. The checks follow section 10 of the rebuild plan:
numbers come from run folders with provenance, synthetic cases are labelled, published model
names need a matching hash, splits contain only real identifiers, report code holds no typed-in
metric values, and patient images are neither tracked by git nor outside the ignored folder.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from .paths import CINC_DIR, FETALDATA, MENDELEY_PWD_DIR, NIFEADB_DIR, NINFEA_ECG_DIR, REPORTS, ROOT, RUNS, SPLITS
from .provenance import REQUIRED_KEYS

REPORT_CODE = [ROOT / "sfr" / "report", ROOT / "scripts" / "build_report.py"]
ALLOWED_NUMBERS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 50, 95, 100, 1000}


def check_runs() -> list[str]:
    problems = []
    for folder in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        metrics, provenance = folder / "metrics.json", folder / "provenance.json"
        if not metrics.exists() or not provenance.exists():
            problems.append(f"{folder.name}: metrics.json or provenance.json is missing")
            continue
        record = json.loads(provenance.read_text(encoding="utf-8"))
        missing = [key for key in REQUIRED_KEYS if key not in record]
        if missing:
            problems.append(f"{folder.name}: provenance lacks {missing}")
            continue
        if record["n_synthetic"] > 0 and not record.get("synthetic_kind"):
            problems.append(f"{folder.name}: synthetic cases without a stated kind")
        if bool(record["synthetic"]) != (record["n_synthetic"] > 0):
            problems.append(f"{folder.name}: synthetic flag does not match the count")
        for model in record["models"]:
            if model.get("claimed_name") and not (model.get("verified") and model.get("sha256")):
                problems.append(f"{folder.name}: model '{model.get('claimed_name')}' is named without a verified hash")
        if not record["data"]:
            problems.append(f"{folder.name}: no input data recorded")
    return problems


def check_splits() -> list[str]:
    problems = []
    cinc = json.loads((SPLITS / "cinc2013.json").read_text(encoding="utf-8"))
    for name in cinc["development"] + cinc["test"]:
        if not (CINC_DIR / f"{name}.hea").exists():
            problems.append(f"splits/cinc2013.json names {name}, which is not on disk")
    if set(cinc["development"]) & set(cinc["test"]):
        problems.append("splits/cinc2013.json: a record is in both development and test")

    women = json.loads((SPLITS / "ninfea_by_woman.json").read_text(encoding="utf-8"))["women"]
    records = [r for w in women for r in w["records"]]
    if sorted(records) != list(range(1, 61)):
        problems.append("splits/ninfea_by_woman.json does not list records 1 to 60 exactly once")
    for record in records:
        if not (NINFEA_ECG_DIR / f"{record}.hea").exists():
            problems.append(f"NInFEA record {record} is not on disk")
    dev_test = json.loads((SPLITS / "ninfea_dev_test.json").read_text(encoding="utf-8"))
    if set(dev_test["development"]) & set(dev_test["test"]):
        problems.append("splits/ninfea_dev_test.json: a woman is in both development and test")
    if sorted(dev_test["development"] + dev_test["test"]) != sorted(w["woman"] for w in women):
        problems.append("splits/ninfea_dev_test.json does not cover the 39 women exactly once")

    marks = json.loads((SPLITS / "pwd_cycle_marks.json").read_text(encoding="utf-8"))
    if set(marks["development"]) & set(marks["test"]):
        problems.append("splits/pwd_cycle_marks.json: a strip is in both development and test")
    if MENDELEY_PWD_DIR.exists():
        for number in marks["development"] + marks["test"]:
            if not (MENDELEY_PWD_DIR / f"img{number}.bmp").exists():
                problems.append(f"annotated strip img{number} is not on disk")
    overlap = json.loads((SPLITS / "pwd_overlap.json").read_text(encoding="utf-8"))
    in_ninfea = {s["img"] for s in overlap["strips"] if s["ninfea_record"]}
    if in_ninfea & set(marks["development"]):
        problems.append("a development strip of the cycle detector is also a NInFEA recording")

    loso = SPLITS / "nifeadb_loso.json"
    if loso.exists():
        for name in json.loads(loso.read_text(encoding="utf-8"))["records"]:
            if not (NIFEADB_DIR / f"{name}.hea").exists():
                problems.append(f"splits/nifeadb_loso.json names {name}, which is not on disk")
    else:
        problems.append("splits/nifeadb_loso.json is missing")
    return problems


def check_patient_images() -> list[str]:
    problems = []
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
        ignored = subprocess.run(["git", "check-ignore", "fetaldata/x"], cwd=ROOT, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["git is not available, so tracking of patient images could not be checked"]
    if any(line.startswith("fetaldata/") for line in tracked):
        problems.append("files under fetaldata/ are tracked by git")
    if FETALDATA.exists() and ignored.returncode != 0:
        problems.append("fetaldata/ is not ignored by git")
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".dcm")
    for line in tracked:
        on_disk = (ROOT / line).exists()             # a tracked file already deleted here is a pending removal
        if on_disk and line.lower().endswith(image_types) and not line.startswith(("reports/figures/", "archive/")):
            problems.append(f"an image file is tracked outside reports/figures: {line}")
    return problems


def check_report_code() -> list[str]:
    """Report code must read its numbers from run folders, never contain them."""
    problems = []
    files = []
    for target in REPORT_CODE:
        files += sorted(target.rglob("*.py")) if target.is_dir() else ([target] if target.exists() else [])
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value not in ALLOWED_NUMBERS and not float(node.value).is_integer():
                    problems.append(f"{path.relative_to(ROOT)}:{node.lineno} holds the number {node.value}")
    return problems


def check_reports() -> list[str]:
    problems = []
    manifest = REPORTS / "manifest.json"
    if not manifest.exists():
        return problems
    for entry in json.loads(manifest.read_text(encoding="utf-8")).get("outputs", []):
        for run in entry.get("runs", []):
            if not (RUNS / run / "provenance.json").exists():
                problems.append(f"{entry['file']} cites run '{run}', which has no provenance")
    return problems


CHECKS = {
    "run folders have provenance, labelled synthetic cases and verified model names": check_runs,
    "split files contain only real identifiers and do not overlap": check_splits,
    "patient images are not tracked by git": check_patient_images,
    "report code holds no typed-in metric values": check_report_code,
    "reports cite only runs that exist": check_reports,
}


def main() -> int:
    failed = 0
    for title, check in CHECKS.items():
        problems = check()
        print(("PASS  " if not problems else "FAIL  ") + title)
        for problem in problems:
            print("      - " + problem)
        failed += bool(problems)
    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
