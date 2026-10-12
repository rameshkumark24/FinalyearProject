"""Development helper for the Doppler cycle detector. Uses the development strips only.

    python scripts/dev_pwd.py [edge_fraction ...]
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.pwd import annotated_path, load_cycle_marks, load_strip  # noqa: E402
from sfr.doppler.cycles import CycleParams, detect_cycles, heart_rate_bpm  # noqa: E402
from sfr.doppler.envelope import PIXELS_PER_SECOND, extract_envelopes  # noqa: E402
from sfr.doppler.marks import marked_rate_bpm, score_starts  # noqa: E402
from sfr.paths import RUNS, SPLITS  # noqa: E402

CACHE = RUNS / "_cache_mendeley_env.pkl"


def envelopes_for(numbers):
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    for n in numbers:
        if n not in cache:
            cache[n] = extract_envelopes(load_strip(annotated_path(n)))
    return cache


def evaluate(numbers, params: CycleParams, offset: float | None = None, verbose: bool = True):
    marks = load_cycle_marks()
    env = envelopes_for(numbers)
    cycles = {n: detect_cycles(env[n], "lower", params=params) for n in numbers}
    if offset is None:                      # constant shift between "end of ejection" and the marks
        residual = []
        for n in numbers:
            s = score_starts(cycles[n].starts, marks[n], tolerance=40)
            starts = np.sort(cycles[n].starts)
            from sfr.ecg.metrics import match_pairs
            residual += [marks[n][j] - starts[i] for i, j in match_pairs(starts, marks[n], 40)]
        offset = float(np.median(residual))
    rows = []
    for n in numbers:
        s15 = score_starts(cycles[n].starts, marks[n], 15, offset)
        s25 = score_starts(cycles[n].starts, marks[n], 25, offset)
        rate, marked = heart_rate_bpm(cycles[n]), marked_rate_bpm(marks[n], PIXELS_PER_SECOND)
        rows.append(dict(img=n, **{k: s15[k] for k in ("n_marks", "recall", "precision", "f1", "recall_hand_placed", "residual_median", "residual_iqr", "fp", "fn")},
                         recall25=s25["recall"], f1_25=s25["f1"], rate=rate, marked_rate=marked,
                         score=cycles[n].track_score, clicks=float(np.mean(cycles[n].has_click)) if cycles[n].n else 0.0))
        if verbose:
            r = rows[-1]
            print("img%-3d marks %3d  F1@15 %.3f (R %.3f P %.3f; hand R %.3f)  F1@25 %.3f  resid %+5.1f iqr %4.1f  rate %5.1f vs %5.1f  track %.2f  clicks seen %.2f"
                  % (n, r["n_marks"], r["f1"], r["recall"], r["precision"], r["recall_hand_placed"], r["f1_25"], r["residual_median"], r["residual_iqr"],
                     r["rate"], r["marked_rate"], r["score"], r["clicks"]))
    tp = sum(round(r["recall"] * r["n_marks"]) for r in rows)
    total = sum(r["n_marks"] for r in rows)
    f1 = np.array([r["f1"] for r in rows])
    rate_ok = [abs(r["rate"] - r["marked_rate"]) <= 5 for r in rows if np.isfinite(r["marked_rate"])]
    summary = dict(offset=offset, pooled_recall=tp / total, mean_f1=float(f1.mean()), median_f1=float(np.median(f1)),
                   min_f1=float(f1.min()), mean_f1_25=float(np.mean([r["f1_25"] for r in rows])),
                   rate_within_5=float(np.mean(rate_ok)) if rate_ok else float("nan"), n_rate=len(rate_ok))
    return summary, rows


if __name__ == "__main__":
    development = json.loads((SPLITS / "pwd_cycle_marks.json").read_text(encoding="utf-8"))["development"]
    fractions = [float(a) for a in sys.argv[1:]] or [CycleParams().edge_fraction]
    for k, fraction in enumerate(fractions):
        summary, _ = evaluate(development, replace(CycleParams(), edge_fraction=fraction), verbose=(len(fractions) == 1))
        print(f"edge_fraction {fraction:.2f}:", {key: round(value, 3) for key, value in summary.items()})
