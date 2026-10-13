"""Compare run folders with a saved copy, file by file (used for the final re-run check).

    python scripts/compare_runs.py <saved-runs-folder> [<runs-folder>]

provenance.json is skipped (it holds run times and the git commit). JSON and CSV files are
compared value by value; numbers that differ are reported with the largest absolute difference.
Values whose key names a run time (minutes, seconds) are listed separately, because they change
from one run to the next by nature.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TIMING_WORDS = ("minutes", "seconds")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append((f"{path}.{key}", "present in one copy only", None))
            else:
                walk(a[key], b[key], f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} against {len(b)}", None))
        for index, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{index}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if a != b and not (a != a and b != b):          # NaN equals NaN here
            out.append((path, f"{a!r} against {b!r}", abs(a - b)))
    elif a != b:
        out.append((path, f"{str(a)[:60]!r} against {str(b)[:60]!r}", None))


def compare_csv(old: Path, new: Path, out):
    with old.open(encoding="utf-8") as h1, new.open(encoding="utf-8") as h2:
        rows_old, rows_new = list(csv.reader(h1)), list(csv.reader(h2))
    if len(rows_old) != len(rows_new):
        out.append(("rows", f"{len(rows_old)} against {len(rows_new)}", None))
        return
    for r, (x, y) in enumerate(zip(rows_old, rows_new)):
        for c, (u, v) in enumerate(zip(x, y)):
            if u == v:
                continue
            try:
                diff = abs(float(u) - float(v))
            except ValueError:
                diff = None
            out.append((f"row {r} column {c}", f"{u[:40]!r} against {v[:40]!r}", diff))


def main() -> int:
    saved = Path(sys.argv[1])
    current = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "runs"
    all_identical = True
    for run in sorted(p for p in saved.iterdir() if p.is_dir()):
        if not (current / run.name).exists():
            print(f"{run.name}: not in the current runs folder")
            continue
        notes = []
        for old in sorted(run.rglob("*")):
            if old.is_dir() or old.name == "provenance.json" or old.suffix in (".npz", ".npy", ".pkl", ".log"):
                continue
            new = current / run.name / old.relative_to(run)
            if not new.exists():
                notes.append(f"  {old.name}: missing in the new run")
                continue
            if old.read_bytes() == new.read_bytes():
                continue
            out = []
            if old.suffix == ".json":
                walk(json.loads(old.read_text(encoding="utf-8")), json.loads(new.read_text(encoding="utf-8")), "", out)
            elif old.suffix == ".csv":
                compare_csv(old, new, out)
            else:
                out.append(("bytes", "files differ", None))
            timing = [d for d in out if any(word in d[0] for word in TIMING_WORDS)]
            real = [d for d in out if d not in timing]
            if real:
                largest = max((d[2] for d in real if d[2] is not None), default=None)
                notes.append(f"  {old.relative_to(run)}: {len(real)} values differ"
                             + (f", largest absolute difference {largest:.3g}" if largest is not None else "")
                             + f"; first: {real[0][0]} {real[0][1]}")
            if timing:
                notes.append(f"  {old.relative_to(run)}: {len(timing)} run-time values differ (expected)")
        identical = not any("values differ" in n and "run-time" not in n or "missing" in n or "files differ" in n for n in notes)
        all_identical &= identical
        print(f"{run.name}: {'identical' if identical else 'DIFFERENT'}")
        for note in notes:
            print(note)
    print("ALL IDENTICAL" if all_identical else "SOME RUNS DIFFER")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
